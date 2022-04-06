
static char help[] = "Solves a linear system in parallel with KSP.\n\n";

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

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-sigma1",&sigma1,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
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
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,dim,dim));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));

  /*
     Currently, all PETSc parallel matrix formats are partitioned by
     contiguous chunks of rows across the processors.  Determine which
     rows of the matrix are locally owned.
  */
  PetscCall(MatGetOwnershipRange(A,&Istart,&Iend));

  /*
     Set matrix elements in parallel.
      - Each processor needs to insert only elements that it owns
        locally (but any non-local elements will be sent to the
        appropriate processor during matrix assembly).
      - Always specify global rows and columns of matrix entries.
  */

  PetscCall(PetscOptionsGetBool(NULL,NULL,"-norandom",&flg,NULL));
  if (flg) use_random = 0;
  else use_random = 1;
  if (use_random) {
    PetscCall(PetscRandomCreate(PETSC_COMM_WORLD,&rctx));
    PetscCall(PetscRandomSetFromOptions(rctx));
    PetscCall(PetscRandomSetInterval(rctx,0.0,PETSC_i));
  } else {
    sigma2 = 10.0*PETSC_i;
  }
  h2 = 1.0/((n+1)*(n+1));
  for (Ii=Istart; Ii<Iend; Ii++) {
    v = -1.0; i = Ii/n; j = Ii - i*n;
    if (i>0) {
      J = Ii-n; PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES));
    }
    if (i<n-1) {
      J = Ii+n; PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES));
    }
    if (j>0) {
      J = Ii-1; PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES));
    }
    if (j<n-1) {
      J = Ii+1; PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES));
    }
    if (use_random) PetscCall(PetscRandomGetValue(rctx,&sigma2));
    v    = 4.0 - sigma1*h2 + sigma2*h2;
    PetscCall(MatSetValues(A,1,&Ii,1,&Ii,&v,ADD_VALUES));
  }
  if (use_random) PetscCall(PetscRandomDestroy(&rctx));

  /*
     Assemble matrix, using the 2-step process:
       MatAssemblyBegin(), MatAssemblyEnd()
     Computations can be done while messages are in transition
     by placing code between these two statements.
  */
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /*
     Create parallel vectors.
      - When using VecCreate(), VecSetSizes() and VecSetFromOptions(),
      we specify only the vector's global
        dimension; the parallel partitioning is determined at runtime.
      - Note: We form 1 vector from scratch and then duplicate as needed.
  */
  PetscCall(VecCreate(PETSC_COMM_WORLD,&u));
  PetscCall(VecSetSizes(u,PETSC_DECIDE,dim));
  PetscCall(VecSetFromOptions(u));
  PetscCall(VecDuplicate(u,&b));
  PetscCall(VecDuplicate(b,&x));

  /*
     Set exact solution; then compute right-hand-side vector.
  */

  if (use_random) {
    PetscCall(PetscRandomCreate(PETSC_COMM_WORLD,&rctx));
    PetscCall(PetscRandomSetFromOptions(rctx));
    PetscCall(VecSetRandom(u,rctx));
  } else {
    PetscCall(VecSet(u,pfive));
  }
  PetscCall(MatMult(A,u,b));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create linear solver context
  */
  PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));

  /*
     Set operators. Here the matrix that defines the linear system
     also serves as the preconditioning matrix.
  */
  PetscCall(KSPSetOperators(ksp,A,A));

  /*
    Set runtime options, e.g.,
        -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
  */
  PetscCall(KSPSetFromOptions(ksp));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(KSPSolve(ksp,b,x));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Check solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
      Print the first 3 entries of x; this demonstrates extraction of the
      real and imaginary components of the complex vector, x.
  */
  flg  = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-print_x3",&flg,NULL));
  if (flg) {
    PetscCall(VecGetArray(x,&xa));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"The first three entries of x are:\n"));
    for (i=0; i<3; i++) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"x[%D] = %g + %g i\n",i,(double)PetscRealPart(xa[i]),(double)PetscImaginaryPart(xa[i])));
    }
    PetscCall(VecRestoreArray(x,&xa));
  }

  /*
     Check the error
  */
  PetscCall(VecAXPY(x,none,u));
  PetscCall(VecNorm(x,NORM_2,&norm));
  PetscCall(KSPGetIterationNumber(ksp,&its));
  if (norm < 1.e-12) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Norm of error < 1.e-12 iterations %D\n",its));
  } else {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Norm of error %g iterations %D\n",(double)norm,its));
  }

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  PetscCall(KSPDestroy(&ksp));
  if (use_random) PetscCall(PetscRandomDestroy(&rctx));
  PetscCall(VecDestroy(&u)); PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b)); PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
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
