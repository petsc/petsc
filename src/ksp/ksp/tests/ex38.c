/*

mpiexec -n 8 ./ex38 -ksp_type fbcgs -ksp_rtol 1.e-6 -sub_ksp_type bcgs -sub_ksp_rtol 1.e-3 -pc_type bjacobi -ksp_converged_reason -ksp_monitor -n1 64 -n2 64

  Contributed by Jie Chen for testing flexible BiCGStab algorithm
*/

static char help[] = "Solves the PDE (in 2D) -laplacian(u) + gamma x dot grad(u) + beta u = 1\n\
with zero Dirichlet condition. The discretization is standard centered\n\
difference. Input parameters include:\n\
  -n1        : number of mesh points in 1st dimension (default 64)\n\
  -n2        : number of mesh points in 2nd dimension (default 64)\n\
  -h         : spacing between mesh points (default 1/n1)\n\
  -gamma     : gamma (default 4/h)\n\
  -beta      : beta (default 0.01/h^2)\n\n";

/*T
   Concepts: KSP^basic parallel example;
   Concepts: KSP^Laplacian, 2d
   Concepts: Laplacian, 2d
   Processors: n
T*/

/*
  Include "petscksp.h" so that we can use KSP solvers.  Note that this file
  automatically includes:
     petscsys.h    - base PETSc routines   petscvec.h - vectors
     petscmat.h    - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
*/
#include <petscksp.h>

int main(int argc,char **args)
{
  Vec            x,b,u;                 /* approx solution, RHS, working vector */
  Mat            A;                     /* linear system matrix */
  KSP            ksp;                   /* linear solver context */
  PetscInt       n1, n2;                /* parameters */
  PetscReal      h, gamma, beta;        /* parameters */
  PetscInt       i,j,Ii,J,Istart,Iend;
  PetscErrorCode ierr;
  PetscScalar    v, co1, co2;
#if defined(PETSC_USE_LOG)
  PetscLogStage stage;
#endif

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  n1 = 64;
  n2 = 64;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n1",&n1,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n2",&n2,NULL));

  h     = 1.0/n1;
  gamma = 4.0;
  beta  = 0.01;
  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-h",&h,NULL));
  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-gamma",&gamma,NULL));
  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-beta",&beta,NULL));
  gamma = gamma/h;
  beta  = beta/(h*h);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Compute the matrix and set right-hand-side vector.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Create parallel matrix, specifying only its global dimensions.
     When using MatCreate(), the matrix format can be specified at
     runtime. Also, the parallel partitioning of the matrix is
     determined by PETSc at runtime.

     Performance tuning note:  For problems of substantial size,
     preallocation of matrix memory is crucial for attaining good
     performance. See the matrix chapter of the users manual for details.
  */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n1*n2,n1*n2));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatMPIAIJSetPreallocation(A,5,NULL,5,NULL));
  CHKERRQ(MatSeqAIJSetPreallocation(A,5,NULL));
  CHKERRQ(MatSetUp(A));

  /*
     Currently, all PETSc parallel matrix formats are partitioned by
     contiguous chunks of rows across the processors.  Determine which
     rows of the matrix are locally owned.
  */
  CHKERRQ(MatGetOwnershipRange(A,&Istart,&Iend));

  /*
     Set matrix elements for the 2-D, five-point stencil in parallel.
      - Each processor needs to insert only elements that it owns
        locally (but any non-local elements will be sent to the
        appropriate processor during matrix assembly).
      - Always specify global rows and columns of matrix entries.
   */
  CHKERRQ(PetscLogStageRegister("Assembly", &stage));
  CHKERRQ(PetscLogStagePush(stage));
  co1  = gamma * h * h / 2.0;
  co2  = beta * h * h;
  for (Ii=Istart; Ii<Iend; Ii++) {
    i = Ii/n2; j = Ii - i*n2;
    if (i>0) {
      J    = Ii - n2;  v = -1.0 + co1*(PetscScalar)i;
      CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));
    }
    if (i<n1-1) {
      J    = Ii + n2;  v = -1.0 + co1*(PetscScalar)i;
      CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));
    }
    if (j>0) {
      J    = Ii - 1;  v = -1.0 + co1*(PetscScalar)j;
      CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));
    }
    if (j<n2-1) {
      J    = Ii + 1;  v = -1.0 + co1*(PetscScalar)j;
      CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));
    }
    v    = 4.0 + co2;
    CHKERRQ(MatSetValues(A,1,&Ii,1,&Ii,&v,INSERT_VALUES));
  }

  /*
     Assemble matrix, using the 2-step process:
       MatAssemblyBegin(), MatAssemblyEnd()
     Computations can be done while messages are in transition
     by placing code between these two statements.
  */
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(PetscLogStagePop());

  /*
     Create parallel vectors.
      - We form 1 vector from scratch and then duplicate as needed.
      - When using VecCreate(), VecSetSizes and VecSetFromOptions()
        in this example, we specify only the
        vector's global dimension; the parallel partitioning is determined
        at runtime.
      - When solving a linear system, the vectors and matrices MUST
        be partitioned accordingly.  PETSc automatically generates
        appropriately partitioned matrices and vectors when MatCreate()
        and VecCreate() are used with the same communicator.
      - The user can alternatively specify the local vector and matrix
        dimensions when more sophisticated partitioning is needed
        (replacing the PETSC_DECIDE argument in the VecSetSizes() statement
        below).
  */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&b));
  CHKERRQ(VecSetSizes(b,PETSC_DECIDE,n1*n2));
  CHKERRQ(VecSetFromOptions(b));
  CHKERRQ(VecDuplicate(b,&x));
  CHKERRQ(VecDuplicate(b,&u));

  /*
     Set right-hand side.
  */
  CHKERRQ(VecSet(b,1.0));

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
     Set linear solver defaults for this problem (optional).
     - By extracting the KSP and PC contexts from the KSP context,
       we can then directly call any KSP and PC routines to set
       various options.
  */
  CHKERRQ(KSPSetTolerances(ksp,1.e-6,PETSC_DEFAULT,PETSC_DEFAULT,200));

  /*
    Set runtime options, e.g.,
        -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
    These options will override those specified above as long as
    KSPSetFromOptions() is called _after_ any other customization
    routines.
  */
  CHKERRQ(KSPSetFromOptions(ksp));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(KSPSolve(ksp,b,x));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  CHKERRQ(KSPDestroy(&ksp));
  CHKERRQ(VecDestroy(&u));  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&b));  CHKERRQ(MatDestroy(&A));

  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_view).
  */
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: 8
      args: -ksp_type fbcgs -ksp_rtol 1.e-6 -sub_ksp_type bcgs -sub_ksp_rtol 1.e-3 -pc_type bjacobi -ksp_converged_reason -n1 64 -n2 64

   test:
      suffix: 2
      nsize: 8
      args: -ksp_type qmrcgs -ksp_rtol 1.e-6 -sub_ksp_type bcgs -sub_ksp_rtol 1.e-3 -pc_type bjacobi -ksp_converged_reason -n1 64 -n2 64
      output_file: output/ex38_1.out

TEST*/
