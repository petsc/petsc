#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex7.c,v 1.37 1999/01/12 23:16:17 bsmith Exp bsmith $";
#endif

static char help[] = "Illustrates use of the block Jacobi preconditioner for\n\
solving a linear system in parallel with SLES.  The code indicates the\n\
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
   Concepts: SLES^Customizing the block Jacobi preconditioner
   Routines: SLESCreate(); SLESSetOperators(); SLESSetFromOptions();
   Routines: SLESGetPC(); SLESGetKSP(); SLESSolve(); SLESView();
   Routines: PCSetType(); PCGetType();
   Routines: PCBJacobiSetTotalBlocks(); PCBJacobiGetSubSLES();
   Routines: KSPSetType(); KSPSetTolerances();
   Processors: n
T*/

/* 
  Include "sles.h" so that we can use SLES solvers.  Note that this file
  automatically includes:
     petsc.h  - base PETSc routines   vec.h - vectors
     sys.h    - system routines       mat.h - matrices
     is.h     - index sets            ksp.h - Krylov subspace methods
     viewer.h - viewers               pc.h  - preconditioners
*/
#include "sles.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Vec     x, b, u;      /* approx solution, RHS, exact solution */
  Mat     A;            /* linear system matrix */
  SLES    sles;         /* SLES context */
  SLES    *subsles;     /* array of local SLES contexts on this processor */
  PC      pc;           /* PC context */
  PC      subpc;        /* PC context for subdomain */
  KSP     subksp;       /* KSP context for subdomain */
  PCType  pctype;       /* preconditioning technique */
  double  norm;         /* norm of solution error */
  int       i, j, I, J, ierr, *blks, m = 8, n;
  int       rank, size, its, nlocal, first, Istart, Iend, flg;
  Scalar    v, one = 1.0, none = -1.0;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = OptionsGetInt(PETSC_NULL,"-m",&m,&flg); CHKERRA(ierr);
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  MPI_Comm_size(PETSC_COMM_WORLD,&size);
  n = m+2;

  /* -------------------------------------------------------------------
         Compute the matrix and right-hand-side vector that define
         the linear system, Ax = b.
     ------------------------------------------------------------------- */

  /* 
     Create and assemble parallel matrix
  */
  ierr = MatCreate(PETSC_COMM_WORLD,m*n,m*n,&A); CHKERRA(ierr);
  ierr = MatGetOwnershipRange(A,&Istart,&Iend); CHKERRA(ierr);
  for ( I=Istart; I<Iend; I++ ) { 
    v = -1.0; i = I/n; j = I - i*n;  
    if ( i>0 )   {J = I - n; MatSetValues(A,1,&I,1,&J,&v,ADD_VALUES);}
    if ( i<m-1 ) {J = I + n; MatSetValues(A,1,&I,1,&J,&v,ADD_VALUES);}
    if ( j>0 )   {J = I - 1; MatSetValues(A,1,&I,1,&J,&v,ADD_VALUES);}
    if ( j<n-1 ) {J = I + 1; MatSetValues(A,1,&I,1,&J,&v,ADD_VALUES);}
    v = 4.0; MatSetValues(A,1,&I,1,&I,&v,ADD_VALUES);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);

  /*
     Create parallel vectors
  */
  ierr = VecCreate(PETSC_COMM_WORLD,PETSC_DECIDE,m*n,&u); CHKERRA(ierr);
  ierr = VecSetFromOptions(u);CHKERRA(ierr);
  ierr = VecDuplicate(u,&b); CHKERRA(ierr);
  ierr = VecDuplicate(b,&x); CHKERRA(ierr);

  /*
     Set exact solution; then compute right-hand-side vector.
  */
  ierr = VecSet(&one,u); CHKERRA(ierr);
  ierr = MatMult(A,u,b); CHKERRA(ierr);

  /*
     Create linear solver context
  */
  ierr = SLESCreate(PETSC_COMM_WORLD,&sles); CHKERRA(ierr);

  /* 
     Set operators. Here the matrix that defines the linear system
     also serves as the preconditioning matrix.
  */
  ierr = SLESSetOperators(sles,A,A,DIFFERENT_NONZERO_PATTERN); CHKERRA(ierr);

  /*
     Set default preconditioner for this program to be block Jacobi.
     This choice can be overridden at runtime with the option
        -pc_type <type>
  */
  ierr = SLESGetPC(sles,&pc); CHKERRA(ierr);
  ierr = PCSetType(pc,PCBJACOBI); CHKERRA(ierr);


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
  blks = (int *) PetscMalloc( m*sizeof(int) ); CHKPTRA(blks);
  for ( i=0; i<m; i++ ) blks[i] = n;
  ierr = PCBJacobiSetTotalBlocks(pc,m,blks);
  PetscFree(blks); 


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
  ierr = PCGetType(pc,&pctype); CHKERRA(ierr);
  if (PetscTypeCompare(pctype,PCBJACOBI)) {
    /* 
       Call SLESSetUp() to set the block Jacobi data structures (including
       creation of an internal SLES context for each block).

       Note: SLESSetUp() MUST be called before PCBJacobiGetSubSLES().
    */
    ierr = SLESSetUp(sles,x,b); CHKERRA(ierr);

    /*
       Extract the array of SLES contexts for the local blocks
    */
    ierr = PCBJacobiGetSubSLES(pc,&nlocal,&first,&subsles); CHKERRA(ierr);

    /*
       Loop over the local blocks, setting various SLES options
       for each block.  
    */
    for (i=0; i<nlocal; i++) {
      ierr = SLESGetPC(subsles[i],&subpc); CHKERRA(ierr);
      ierr = SLESGetKSP(subsles[i],&subksp); CHKERRA(ierr);
      if (rank == 0) {
        if (i%2) {
          ierr = PCSetType(subpc,PCILU); CHKERRA(ierr);
        } else {
          ierr = PCSetType(subpc,PCNONE); CHKERRA(ierr);
          ierr = KSPSetType(subksp,KSPBCGS); CHKERRA(ierr);
          ierr = KSPSetTolerances(subksp,1.e-6,PETSC_DEFAULT,PETSC_DEFAULT,
                 PETSC_DEFAULT); CHKERRA(ierr);
        }
      } else {
        ierr = PCSetType(subpc,PCJACOBI); CHKERRA(ierr);
        ierr = KSPSetType(subksp,KSPGMRES); CHKERRA(ierr);
        ierr = KSPSetTolerances(subksp,1.e-7,PETSC_DEFAULT,PETSC_DEFAULT,
               PETSC_DEFAULT); CHKERRA(ierr);
      }
    }
  }

  /* -------------------------------------------------------------------
                      Solve the linear system
     ------------------------------------------------------------------- */

  /* 
    Set runtime options
  */
  ierr = SLESSetFromOptions(sles); CHKERRA(ierr);

  /*
     Solve the linear system
  */
  ierr = SLESSolve(sles,b,x,&its); CHKERRA(ierr);

  /*
     View info about the solver
  */
  ierr = OptionsHasName(PETSC_NULL,"-noslesview",&flg); CHKERRA(ierr);
  if (!flg) {
    ierr = SLESView(sles,VIEWER_STDOUT_WORLD); CHKERRA(ierr);
  }

  /* -------------------------------------------------------------------
                      Check solution and clean up
     ------------------------------------------------------------------- */

  /*
     Check the error
  */
  ierr = VecAXPY(&none,u,x); CHKERRA(ierr);
  ierr = VecNorm(x,NORM_2,&norm); CHKERRA(ierr);
  if (norm > 1.e-12)
    PetscPrintf(PETSC_COMM_WORLD,"Norm of error %g iterations %d\n",norm,its);
  else 
    PetscPrintf(PETSC_COMM_WORLD,"Norm of error < 1.e-12 Iterations %d\n",its);

  /* 
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  ierr = SLESDestroy(sles); CHKERRA(ierr);
  ierr = VecDestroy(u); CHKERRA(ierr);  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(b); CHKERRA(ierr);  ierr = MatDestroy(A); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
