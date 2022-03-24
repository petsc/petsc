
static char help[] = "Solves a linear system in parallel with KSP.\n\n";

/*T
   Concepts: KSP^solving a Helmholtz equation
   Concepts: complex numbers;
   Concepts: Helmholtz equation
   Processors: n
T*/

/*
   Description: Solves a complex linear system in parallel with KSP.

   The model problem:
      Solve Helmholtz equation on the unit square: (0,1) x (0,1)
          -delta u - sigma1*u + i*sigma2*u = f,
           where delta = Laplace operator
      Dirichlet b.c.'s on all sides
      Use the 2-D, five-point finite difference stencil.

   Compiling the code:
      This code uses the complex numbers version of PETSc, so configure
      must be run to enable this
*/

/*
  Include "petscksp.h" so that we can use KSP solvers.  Note that this file
  automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
*/
#include <petscksp.h>

int main(int argc,char **args)
{
  Vec            x,b,u;      /* approx solution, RHS, exact solution */
  Mat            A;            /* linear system matrix */
  KSP            ksp;         /* linear solver context */
  PetscReal      norm;         /* norm of solution error */
  PetscInt       dim,i,j,Ii,J,Istart,Iend,n = 6,its,use_random;
  PetscScalar    v,none = -1.0,sigma2,pfive = 0.5,*xa;
  PetscRandom    rctx;
  PetscReal      h2,sigma1 = 100.0;
  PetscBool      flg = PETSC_FALSE;

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-sigma1",&sigma1,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  dim  = n*n;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Compute the matrix and right-hand-side vector that define
         the linear system, Ax = b.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Create parallel matrix, specifying only its global dimensions.
     When using MatCreate(), the matrix format can be specified at
     runtime. Also, the parallel partitioning of the matrix is
     determined by PETSc at runtime.
  */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,dim,dim));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));

  /*
     Currently, all PETSc parallel matrix formats are partitioned by
     contiguous chunks of rows across the processors.  Determine which
     rows of the matrix are locally owned.
  */
  CHKERRQ(MatGetOwnershipRange(A,&Istart,&Iend));

  /*
     Set matrix elements in parallel.
      - Each processor needs to insert only elements that it owns
        locally (but any non-local elements will be sent to the
        appropriate processor during matrix assembly).
      - Always specify global rows and columns of matrix entries.
  */

  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-norandom",&flg,NULL));
  if (flg) use_random = 0;
  else use_random = 1;
  if (use_random) {
    CHKERRQ(PetscRandomCreate(PETSC_COMM_WORLD,&rctx));
    CHKERRQ(PetscRandomSetFromOptions(rctx));
    CHKERRQ(PetscRandomSetInterval(rctx,0.0,PETSC_i));
  } else {
    sigma2 = 10.0*PETSC_i;
  }
  h2 = 1.0/((n+1)*(n+1));
  for (Ii=Istart; Ii<Iend; Ii++) {
    v = -1.0; i = Ii/n; j = Ii - i*n;
    if (i>0) {
      J = Ii-n; CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES));
    }
    if (i<n-1) {
      J = Ii+n; CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES));
    }
    if (j>0) {
      J = Ii-1; CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES));
    }
    if (j<n-1) {
      J = Ii+1; CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES));
    }
    if (use_random) CHKERRQ(PetscRandomGetValue(rctx,&sigma2));
    v    = 4.0 - sigma1*h2 + sigma2*h2;
    CHKERRQ(MatSetValues(A,1,&Ii,1,&Ii,&v,ADD_VALUES));
  }
  if (use_random) CHKERRQ(PetscRandomDestroy(&rctx));

  /*
     Assemble matrix, using the 2-step process:
       MatAssemblyBegin(), MatAssemblyEnd()
     Computations can be done while messages are in transition
     by placing code between these two statements.
  */
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /*
     Create parallel vectors.
      - When using VecCreate(), VecSetSizes() and VecSetFromOptions(),
      we specify only the vector's global
        dimension; the parallel partitioning is determined at runtime.
      - Note: We form 1 vector from scratch and then duplicate as needed.
  */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&u));
  CHKERRQ(VecSetSizes(u,PETSC_DECIDE,dim));
  CHKERRQ(VecSetFromOptions(u));
  CHKERRQ(VecDuplicate(u,&b));
  CHKERRQ(VecDuplicate(b,&x));

  /*
     Set exact solution; then compute right-hand-side vector.
  */

  if (use_random) {
    CHKERRQ(PetscRandomCreate(PETSC_COMM_WORLD,&rctx));
    CHKERRQ(PetscRandomSetFromOptions(rctx));
    CHKERRQ(VecSetRandom(u,rctx));
  } else {
    CHKERRQ(VecSet(u,pfive));
  }
  CHKERRQ(MatMult(A,u,b));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create linear solver context
  */
  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));

  /*
     Set operators. Here the matrix that defines the linear system
     also serves as the preconditioning matrix.
  */
  CHKERRQ(KSPSetOperators(ksp,A,A));

  /*
    Set runtime options, e.g.,
        -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
  */
  CHKERRQ(KSPSetFromOptions(ksp));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(KSPSolve(ksp,b,x));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Check solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
      Print the first 3 entries of x; this demonstrates extraction of the
      real and imaginary components of the complex vector, x.
  */
  flg  = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-print_x3",&flg,NULL));
  if (flg) {
    CHKERRQ(VecGetArray(x,&xa));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"The first three entries of x are:\n"));
    for (i=0; i<3; i++) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"x[%D] = %g + %g i\n",i,(double)PetscRealPart(xa[i]),(double)PetscImaginaryPart(xa[i])));
    }
    CHKERRQ(VecRestoreArray(x,&xa));
  }

  /*
     Check the error
  */
  CHKERRQ(VecAXPY(x,none,u));
  CHKERRQ(VecNorm(x,NORM_2,&norm));
  CHKERRQ(KSPGetIterationNumber(ksp,&its));
  if (norm < 1.e-12) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Norm of error < 1.e-12 iterations %D\n",its));
  } else {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Norm of error %g iterations %D\n",(double)norm,its));
  }

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  CHKERRQ(KSPDestroy(&ksp));
  if (use_random) CHKERRQ(PetscRandomDestroy(&rctx));
  CHKERRQ(VecDestroy(&u)); CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&b)); CHKERRQ(MatDestroy(&A));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: complex

   test:
      args: -n 6 -norandom -pc_type none -ksp_monitor_short -ksp_gmres_cgs_refinement_type refine_always

   testset:
      suffix: deflation
      args: -norandom -pc_type deflation -ksp_monitor_short
      requires: superlu_dist

      test:
        nsize: 6

      test:
        nsize: 3
        args: -pc_deflation_compute_space {{db2 aggregation}}

      test:
        suffix: pc_deflation_init_only-0
        nsize: 4
        args: -ksp_type fgmres -pc_deflation_compute_space db4 -pc_deflation_compute_space_size 2 -pc_deflation_levels 2 -deflation_ksp_max_it 10
        #TODO remove suffix and next test when this works
        #args: -pc_deflation_init_only {{0 1}separate output}
        args: -pc_deflation_init_only 0

      test:
        suffix: pc_deflation_init_only-1
        nsize: 4
        args: -ksp_type fgmres -pc_deflation_compute_space db4 -pc_deflation_compute_space_size 2 -pc_deflation_levels 2 -deflation_ksp_max_it 10
        args: -pc_deflation_init_only 1

TEST*/
