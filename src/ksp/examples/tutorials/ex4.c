
static char help[] = "Uses a different preconditioner matrix and linear system matrix in the KSP solvers.\n\
Note that different storage formats\n\
can be used for the different matrices.\n\n";

/*T
   Concepts: KSP^different matrices for linear system and preconditioner;
   Processors: n
T*/

/* 
  Include "petscksp.h" so that we can use KSP solvers.  Note that this file
  automatically includes:
     petsc.h       - base PETSc routines   petscvec.h - vectors
     petscsys.h    - system routines       petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
*/
#include "petscksp.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  KSP            ksp;      /* linear solver context */
  Mat            A,B;      /* linear system matrix, preconditioning matrix */
  PetscRandom    rctx;      /* random number generator context */
  Vec            x,b,u;   /* approx solution, RHS, exact solution */
  Vec            tmp;       /* work vector */
  PetscScalar    v,one = 1.0,scale = 0.0;
  PetscInt       i,j,m = 15,n = 17,I,J,Istart,Iend;
  PetscErrorCode ierr;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-m",&m,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(PETSC_NULL,"-scale",&scale,PETSC_NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
         Compute the matrix and right-hand-side vector that define
         the linear system,Ax = b.  Also, create a different
         preconditioner matrix.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create the linear system matrix (A).
      - Here we use a block diagonal matrix format (MATBDIAG) and
        specify only the global size.  The parallel partitioning of
        the matrix will be determined at runtime by PETSc.
  */
  ierr = MatCreate(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n,&A);CHKERRQ(ierr);
  ierr = MatSetType(A,MATBDIAG);CHKERRQ(ierr);
  ierr = MatSeqBDiagSetPreallocation(A,0,1,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatMPIBDiagSetPreallocation(A,0,1,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);

  /* 
     Create a different preconditioner matrix (B).  This is usually
     done to form a cheaper (or sparser) preconditioner matrix
     compared to the linear system matrix.
      - Here we use MatCreate() followed by MatSetFromOptions(),
        so that the matrix format and parallel partitioning will be
        determined at runtime.
  */
  ierr = MatCreate(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n,&B);CHKERRQ(ierr);
  ierr = MatSetFromOptions(B);CHKERRQ(ierr);

  /* 
     Currently, all PETSc parallel matrix formats are partitioned by
     contiguous chunks of rows across the processors.  Determine which
     rows of the matrix are locally owned. 
  */
  ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);

  /*
     Set entries within the two matrices
  */
  for (I=Istart; I<Iend; I++) { 
    v = -1.0; i = I/n; j = I - i*n;  
    if (i>0) {
      J=I-n; 
      ierr = MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatSetValues(B,1,&I,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
    if (i<m-1) {
      J=I+n; 
      ierr = MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatSetValues(B,1,&I,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
    if (j>0) {
      J=I-1; 
      ierr = MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
    if (j<n-1) {
      J=I+1; 
      ierr = MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
    v = 5.0; ierr = MatSetValues(A,1,&I,1,&I,&v,INSERT_VALUES);CHKERRQ(ierr);
    v = 3.0; ierr = MatSetValues(B,1,&I,1,&I,&v,INSERT_VALUES);CHKERRQ(ierr);
  }

  /*
     Assemble the preconditioner matrix (B), using the 2-step process
       MatAssemblyBegin(), MatAssemblyEnd()
     Note that computations can be done while messages are in
     transition by placing code between these two statements.
  */
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  for (I=Istart; I<Iend; I++) { 
    v = -0.5; i = I/n;
    if (i>1) { 
      J=I-(n+1); ierr = MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
    if (i<m-2) {
      J=I+n+1; ierr = MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* 
     Assemble the linear system matrix, (A)
  */
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* 
     Create parallel vectors.
      - When using VecSeSizes(), we specify only the vector's global
        dimension; the parallel partitioning is determined at runtime. 
      - Note: We form 1 vector from scratch and then duplicate as needed.
  */
  ierr = VecCreate(PETSC_COMM_WORLD,&b);CHKERRQ(ierr);
  ierr = VecSetSizes(b,PETSC_DECIDE,m*n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(b);CHKERRQ(ierr);
  ierr = VecDuplicate(b,&u);CHKERRQ(ierr);
  ierr = VecDuplicate(b,&x);CHKERRQ(ierr);

  /* 
     Make solution vector be 1 to random noise
  */
  ierr = VecSet(u,one);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&tmp);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,RANDOM_DEFAULT,&rctx);CHKERRQ(ierr);
  ierr = VecSetRandom(rctx,tmp);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(rctx);CHKERRQ(ierr);
  ierr = VecAXPY(u,scale,tmp);CHKERRQ(ierr);
  ierr = VecDestroy(tmp);CHKERRQ(ierr);

  /*
     Compute right-hand-side vector 
  */
  ierr = MatMult(A,u,b);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* 
    Create linear solver context
  */
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);

  /* 
     Set operators. Note that we use different matrices to define the
     linear system and to precondition it.
  */
  ierr = KSPSetOperators(ksp,A,B,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);

  /* 
     Set runtime options (e.g., -ksp_type <type> -pc_type <type>)
  */
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

  /* 
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  ierr = KSPDestroy(ksp);CHKERRQ(ierr); ierr = VecDestroy(u);CHKERRQ(ierr);
  ierr = MatDestroy(B);CHKERRQ(ierr);   ierr = VecDestroy(x);CHKERRQ(ierr);
  ierr = MatDestroy(A);CHKERRQ(ierr);   ierr = VecDestroy(b);CHKERRQ(ierr);

  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_summary). 
  */
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
