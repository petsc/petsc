/*$Id: ex7.c,v 1.58 2001/08/07 21:30:54 bsmith Exp $*/

static char help[] = "Block Jacobi preconditioner for solving a linear system in parallel with SLES.\n\
The code indicates the\n\
procedures for setting the particular block sizes and for using different\n\
linear solvers on the individual blocks.\n\n";

/*
   Note:  This example focuses on ways to customize the block Jacobi
   preconditioner. See ex1.c and ex2.c for more detailed comments on
   the basic usage of SLES (including working with matrices and vectors).

   Recall: The block Jacobi method is equivalent to the ASM preconditioner
   with zero overlap.
 */

/*T
   Concepts: SLES^customizing the block Jacobi preconditioner
   Processors: n
T*/

/* 
  Include "petscsles.h" so that we can use SLES solvers.  Note that this file
  automatically includes:
     petsc.h       - base PETSc routines   petscvec.h - vectors
     petscsys.h    - system routines       petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
*/
#include "petscsles.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Vec          x,b,u;      /* approx solution, RHS, exact solution */
  Mat          A;            /* linear system matrix */
  SLES         sles;         /* SLES context */
  SLES         *subsles;     /* array of local SLES contexts on this processor */
  PC           pc;           /* PC context */
  PC           subpc;        /* PC context for subdomain */
  KSP          subksp;       /* KSP context for subdomain */
  PetscReal    norm;         /* norm of solution error */
  int          i,j,I,J,ierr,*blks,m = 8,n;
  int          rank,size,its,nlocal,first,Istart,Iend;
  PetscScalar  v,one = 1.0,none = -1.0;
  PetscTruth   isbjacobi,flg;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-m",&m,PETSC_NULL);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  n = m+2;

  /* -------------------------------------------------------------------
         Compute the matrix and right-hand-side vector that define
         the linear system, Ax = b.
     ------------------------------------------------------------------- */

  /* 
     Create and assemble parallel matrix
  */
  ierr = MatCreate(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n,&A);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);
  for (I=Istart; I<Iend; I++) { 
    v = -1.0; i = I/n; j = I - i*n;  
    if (i>0)   {J = I - n; ierr = MatSetValues(A,1,&I,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
    if (i<m-1) {J = I + n; ierr = MatSetValues(A,1,&I,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
    if (j>0)   {J = I - 1; ierr = MatSetValues(A,1,&I,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
    if (j<n-1) {J = I + 1; ierr = MatSetValues(A,1,&I,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
    v = 4.0; ierr = MatSetValues(A,1,&I,1,&I,&v,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /*
     Create parallel vectors
  */
  ierr = VecCreate(PETSC_COMM_WORLD,&u);CHKERRQ(ierr);
  ierr = VecSetSizes(u,PETSC_DECIDE,m*n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(u);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&b);CHKERRQ(ierr);
  ierr = VecDuplicate(b,&x);CHKERRQ(ierr);

  /*
     Set exact solution; then compute right-hand-side vector.
  */
  ierr = VecSet(&one,u);CHKERRQ(ierr);
  ierr = MatMult(A,u,b);CHKERRQ(ierr);

  /*
     Create linear solver context
  */
  ierr = SLESCreate(PETSC_COMM_WORLD,&sles);CHKERRQ(ierr);

  /* 
     Set operators. Here the matrix that defines the linear system
     also serves as the preconditioning matrix.
  */
  ierr = SLESSetOperators(sles,A,A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);

  /*
     Set default preconditioner for this program to be block Jacobi.
     This choice can be overridden at runtime with the option
        -pc_type <type>
  */
  ierr = SLESGetPC(sles,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCBJACOBI);CHKERRQ(ierr);


  /* -------------------------------------------------------------------
                   Define the problem decomposition
     ------------------------------------------------------------------- */

  /* 
     Call PCBJacobiSetTotalBlocks() to set individually the size of
     each block in the preconditioner.  This could also be done with
     the runtime option
         -pc_bjacobi_blocks <blocks>
     Also, see the command PCBJacobiSetLocalBlocks() to set the
     local blocks.

      Note: The default decomposition is 1 block per processor.
  */
  ierr = PetscMalloc(m*sizeof(int),&blks);CHKERRQ(ierr);
  for (i=0; i<m; i++) blks[i] = n;
  ierr = PCBJacobiSetTotalBlocks(pc,m,blks);CHKERRQ(ierr);
  ierr = PetscFree(blks);CHKERRQ(ierr);


  /* -------------------------------------------------------------------
               Set the linear solvers for the subblocks
     ------------------------------------------------------------------- */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
       Basic method, should be sufficient for the needs of most users.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

     By default, the block Jacobi method uses the same solver on each
     block of the problem.  To set the same solver options on all blocks, 
     use the prefix -sub before the usual PC and KSP options, e.g.,
          -sub_pc_type <pc> -sub_ksp_type <ksp> -sub_ksp_rtol 1.e-4
  */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        Advanced method, setting different solvers for various blocks.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

     Note that each block's SLES context is completely independent of
     the others, and the full range of uniprocessor SLES options is
     available for each block. The following section of code is intended
     to be a simple illustration of setting different linear solvers for
     the individual blocks.  These choices are obviously not recommended
     for solving this particular problem.
  */
  ierr = PetscTypeCompare((PetscObject)pc,PCBJACOBI,&isbjacobi);CHKERRQ(ierr);
  if (isbjacobi) {
    /* 
       Call SLESSetUp() to set the block Jacobi data structures (including
       creation of an internal SLES context for each block).

       Note: SLESSetUp() MUST be called before PCBJacobiGetSubSLES().
    */
    ierr = SLESSetUp(sles,b,x);CHKERRQ(ierr);

    /*
       Extract the array of SLES contexts for the local blocks
    */
    ierr = PCBJacobiGetSubSLES(pc,&nlocal,&first,&subsles);CHKERRQ(ierr);

    /*
       Loop over the local blocks, setting various SLES options
       for each block.  
    */
    for (i=0; i<nlocal; i++) {
      ierr = SLESGetPC(subsles[i],&subpc);CHKERRQ(ierr);
      ierr = SLESGetKSP(subsles[i],&subksp);CHKERRQ(ierr);
      if (!rank) {
        if (i%2) {
          ierr = PCSetType(subpc,PCILU);CHKERRQ(ierr);
        } else {
          ierr = PCSetType(subpc,PCNONE);CHKERRQ(ierr);
          ierr = KSPSetType(subksp,KSPBCGS);CHKERRQ(ierr);
          ierr = KSPSetTolerances(subksp,1.e-6,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
        }
      } else {
        ierr = PCSetType(subpc,PCJACOBI);CHKERRQ(ierr);
        ierr = KSPSetType(subksp,KSPGMRES);CHKERRQ(ierr);
        ierr = KSPSetTolerances(subksp,1.e-7,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
      }
    }
  }

  /* -------------------------------------------------------------------
                      Solve the linear system
     ------------------------------------------------------------------- */

  /* 
    Set runtime options
  */
  ierr = SLESSetFromOptions(sles);CHKERRQ(ierr);

  /*
     Solve the linear system
  */
  ierr = SLESSolve(sles,b,x,&its);CHKERRQ(ierr);

  /*
     View info about the solver
  */
  ierr = PetscOptionsHasName(PETSC_NULL,"-noslesview",&flg);CHKERRQ(ierr);
  if (!flg) {
    ierr = SLESView(sles,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  /* -------------------------------------------------------------------
                      Check solution and clean up
     ------------------------------------------------------------------- */

  /*
     Check the error
  */
  ierr = VecAXPY(&none,u,x);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error %A iterations %d\n",norm,its);CHKERRQ(ierr);

  /* 
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  ierr = SLESDestroy(sles);CHKERRQ(ierr);
  ierr = VecDestroy(u);CHKERRQ(ierr);  ierr = VecDestroy(x);CHKERRQ(ierr);
  ierr = VecDestroy(b);CHKERRQ(ierr);  ierr = MatDestroy(A);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
