
static char help[] = "Illustrates use of the preconditioner GASM.\n\
The Additive Schwarz Method for solving a linear system in parallel with KSP.  The\n\
code indicates the procedure for setting user-defined subdomains.  Input\n\
parameters include:\n\
  -M:                           Number of mesh points in the x direction\n\
  -N:                           Number of mesh points in the y direction\n\
  -user_set_subdomain_solvers:  User explicitly sets subdomain solvers\n\
  -user_set_subdomains:         Use the user-provided subdomain partitioning routine\n\
With -user_set_subdomains on, the following options are meaningful:\n\
  -Mdomains:                    Number of subdomains in the x direction \n\
  -Ndomains:                    Number of subdomains in the y direction \n\
  -overlap:                     Size of domain overlap in terms of the number of mesh lines in x and y\n\
General useful options:\n\
  -pc_gasm_print_subdomains:    Print the index sets defining the subdomains\n\
\n";

/*
   Note:  This example focuses on setting the subdomains for the GASM 
   preconditioner for a problem on a 2D rectangular grid.  See ex1.c
   and ex2.c for more detailed comments on the basic usage of KSP
   (including working with matrices and vectors).

   The GASM preconditioner is fully parallel.  The user-space routine
   CreateSubdomains2D that computes the domain decomposition is also parallel 
   and attempts to generate both subdomains straddling processors and multiple 
   domains per processor.


   This matrix in this linear system arises from the discretized Laplacian,
   and thus is not very interesting in terms of experimenting with variants
   of the GASM preconditioner.  
*/

/*T
   Concepts: KSP^Additive Schwarz Method (GASM) with user-defined subdomains
   Processors: n
T*/

/* 
  Include "petscksp.h" so that we can use KSP solvers.  Note that this file
  automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
*/
#include <petscksp.h>



#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Vec            x,b,u;                  /* approx solution, RHS, exact solution */
  Mat            A;                      /* linear system matrix */
  KSP            ksp;                    /* linear solver context */
  PC             pc;                     /* PC context */
  IS             *is,*is_local;          /* array of index sets that define the subdomains */
  PetscInt       overlap = 1;            /* width of subdomain overlap */
  PetscInt       Nsub;                   /* number of subdomains */
  PetscInt       m = 15,n = 17;          /* mesh dimensions in x- and y- directions */
  PetscInt       M = 2,N = 1;            /* number of subdomains in x- and y- directions */
  PetscInt       i,j,Ii,J,Istart,Iend;
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscBool      flg;
  PetscBool      user_set_subdomains = PETSC_FALSE;     
  PetscScalar    v, one = 1.0;
  PetscReal      e;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-M",&m,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-N",&n,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(PETSC_NULL,"-user_set_subdomains",&user_set_subdomains,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-Mdomains",&M,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-Ndomains",&N,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-overlap",&overlap,PETSC_NULL);CHKERRQ(ierr);

  /* -------------------------------------------------------------------
         Compute the matrix and right-hand-side vector that define
         the linear system, Ax = b.
     ------------------------------------------------------------------- */

  /* 
     Assemble the matrix for the five point stencil, YET AGAIN 
  */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart; Ii<Iend; Ii++) { 
    v = -1.0; i = Ii/n; j = Ii - i*n;  
    if (i>0)   {J = Ii - n; ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
    if (i<m-1) {J = Ii + n; ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
    if (j>0)   {J = Ii - 1; ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
    if (j<n-1) {J = Ii + 1; ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
    v = 4.0; ierr = MatSetValues(A,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* 
     Create and set vectors 
  */
  ierr = VecCreate(PETSC_COMM_WORLD,&b);CHKERRQ(ierr);
  ierr = VecSetSizes(b,PETSC_DECIDE,m*n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(b);CHKERRQ(ierr);
  ierr = VecDuplicate(b,&u);CHKERRQ(ierr);
  ierr = VecDuplicate(b,&x);CHKERRQ(ierr);
  ierr = VecSet(u,one);CHKERRQ(ierr);
  ierr = MatMult(A,u,b);CHKERRQ(ierr);

  /* 
     Create linear solver context 
  */
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);

  /* 
     Set operators. Here the matrix that defines the linear system
     also serves as the preconditioning matrix.
  */
  ierr = KSPSetOperators(ksp,A,A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);

  /* 
     Set the default preconditioner for this program to be GASM
  */
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCGASM);CHKERRQ(ierr);

  /* -------------------------------------------------------------------
                  Define the problem decomposition
     ------------------------------------------------------------------- */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
       Basic method, should be sufficient for the needs of many users.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

     Set the overlap, using the default PETSc decomposition via
         PCGASMSetOverlap(pc,overlap);
     Could instead use the option -pc_gasm_overlap <ovl> 

     Set the total number of blocks via -pc_gasm_blocks <blks>
     Note:  The GASM default is to use 1 block per processor.  To
     experiment on a single processor with various overlaps, you
     must specify use of multiple blocks!
  */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
       More advanced method, setting user-defined subdomains
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

     Firstly, create index sets that define the subdomains.  The utility
     routine PCGASMCreateSubdomains2D() is a simple example, which partitions
     the 2D grid into MxN subdomains with an optional overlap.  
     More generally, the user should write a custom routine for a particular 
     problem geometry.

     Then call PCGASMSetLocalSubdomains() with resulting index sets
     to set the subdomains for the GASM preconditioner.
  */

  if (!user_set_subdomains) { /* basic version */
    ierr = PCGASMSetOverlap(pc,overlap);CHKERRQ(ierr);
  } else { /* advanced version */
    ierr = PCGASMCreateSubdomains2D(pc, m,n,M,N,1,overlap,&Nsub,&is,&is_local);CHKERRQ(ierr);
    ierr = PCGASMSetLocalSubdomains(pc,Nsub,is,is_local);CHKERRQ(ierr);
    ierr = PCView(pc, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
  }

  /* -------------------------------------------------------------------
                Set the linear solvers for the subblocks
     ------------------------------------------------------------------- */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
       Basic method, should be sufficient for the needs of most users.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

     By default, the GASM preconditioner uses the same solver on each
     block of the problem.  To set the same solver options on all blocks,
     use the prefix -sub before the usual PC and KSP options, e.g.,
          -sub_pc_type <pc> -sub_ksp_type <ksp> -sub_ksp_rtol 1.e-4

     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        Advanced method, setting different solvers for various blocks.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

     Note that each block's KSP context is completely independent of
     the others, and the full range of uniprocessor KSP options is
     available for each block.

     - Use PCGASMGetSubKSP() to extract the array of KSP contexts for
       the local blocks.
     - See ex7.c for a simple example of setting different linear solvers
       for the individual blocks for the block Jacobi method (which is
       equivalent to the GASM method with zero overlap).
  */

  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PETSC_NULL,"-user_set_subdomain_solvers",&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg) {
    KSP        *subksp;       /* array of KSP contexts for local subblocks */
    PetscInt   nlocal,first;  /* number of local subblocks, first local subblock */
    PC         subpc;          /* PC context for subblock */
    PetscBool  isasm;

    ierr = PetscPrintf(PETSC_COMM_WORLD,"User explicitly sets subdomain solvers.\n");CHKERRQ(ierr);

    /* 
       Set runtime options
    */
    ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

    /* 
       Flag an error if PCTYPE is changed from the runtime options
     */
    ierr = PetscTypeCompare((PetscObject)pc,PCGASM,&isasm);CHKERRQ(ierr);
    if (!isasm) SETERRQ(PETSC_COMM_WORLD,1,"Cannot Change the PCTYPE when manually changing the subdomain solver settings");

    /* 
       Call KSPSetUp() to set the block Jacobi data structures (including
       creation of an internal KSP context for each block).

       Note: KSPSetUp() MUST be called before PCGASMGetSubKSP().
    */
    ierr = KSPSetUp(ksp);CHKERRQ(ierr);

    /*
       Extract the array of KSP contexts for the local blocks
    */
    ierr = PCGASMGetSubKSP(pc,&nlocal,&first,&subksp);CHKERRQ(ierr);

    /*
       Loop over the local blocks, setting various KSP options
       for each block.  
    */
    for (i=0; i<nlocal; i++) {
      ierr = KSPGetPC(subksp[i],&subpc);CHKERRQ(ierr);
      ierr = PCSetType(subpc,PCILU);CHKERRQ(ierr);
      ierr = KSPSetType(subksp[i],KSPGMRES);CHKERRQ(ierr);
      ierr = KSPSetTolerances(subksp[i],1.e-7,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
    }
  } else {
    /* 
       Set runtime options
    */
    ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  }

  /* -------------------------------------------------------------------
                      Solve the linear system
     ------------------------------------------------------------------- */

  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

  /* -------------------------------------------------------------------
                      Compare result to the exact solution
     ------------------------------------------------------------------- */
  ierr = VecAXPY(x,-1.0,u); CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_INFINITY, &e); CHKERRQ(ierr);

  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PETSC_NULL,"-print_error",&flg,PETSC_NULL);CHKERRQ(ierr);
  if(flg) {
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Infinity norm of the error: %G\n", e); CHKERRQ(ierr);
  }

  /* 
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */

  if (user_set_subdomains) {
    for (i=0; i<Nsub; i++) {
      ierr = ISDestroy(&is[i]);CHKERRQ(ierr);
      ierr = ISDestroy(&is_local[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(is);CHKERRQ(ierr);
    ierr = PetscFree(is_local);CHKERRQ(ierr);
  }
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
