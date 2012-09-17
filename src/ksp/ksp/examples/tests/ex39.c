/*

mpiexec -n 8 ./ex39 -ksp_type fbcgs -ksp_rtol 1.e-6 -sub_ksp_type bcgs -sub_ksp_rtol 1.e-3 -pc_type bjacobi -ksp_converged_reason -ksp_monitor -n1 32 -n2 32 -n3 32

  Contributed by Jie Chen for testing flexible BiCGStab algorithm
*/

static char help[] = "Solves the PDE (in 3D) - laplacian(u) + gamma x dot grad(u) + beta u = 1\n\
with zero Dirichlet condition. The discretization is standard centered\n\
difference. Input parameters include:\n\
  -n1        : number of mesh points in 1st dimension (default 32)\n\
  -n2        : number of mesh points in 2nd dimension (default 32)\n\
  -n3        : number of mesh points in 3nd dimension (default 32)\n\
  -h         : spacing between mesh points (default 1/n1)\n\
  -gamma     : gamma (default 4/h)\n\
  -beta      : beta (default 0.01/h^2)\n\n";

/*T
   Concepts: KSP^basic parallel example;
   Concepts: KSP^Laplacian, 3d
   Concepts: Laplacian, 3d
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

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Vec            x,b,u;                 /* approx solution, RHS, working vector */
  Mat            A;                     /* linear system matrix */
  KSP            ksp;                   /* linear solver context */
  PetscInt       n1, n2, n3;            /* parameters */
  PetscReal      h, gamma, beta;        /* parameters */
  PetscInt       i,j,k,Ii,J,Istart,Iend;
  PetscErrorCode ierr;
  PetscScalar    v, co1, co2;
#if defined(PETSC_USE_LOG)
  PetscLogStage  stage;
#endif

  PetscInitialize(&argc,&args,(char *)0,help);

  n1 = 32;
  n2 = 32;
  n3 = 32;

  ierr = PetscOptionsGetInt(PETSC_NULL,"-n1",&n1,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n2",&n2,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n3",&n3,PETSC_NULL);CHKERRQ(ierr);

  h = 1.0/n1;
  gamma = 4.0/h;
  beta = 0.01/(h*h);

  ierr = PetscOptionsGetReal(PETSC_NULL,"-h",&h,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-gamma",&gamma,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-beta",&beta,PETSC_NULL);CHKERRQ(ierr);

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
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n1*n2*n3,n1*n2*n3);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(A,7,PETSC_NULL,7,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(A,7,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  /* 
     Currently, all PETSc parallel matrix formats are partitioned by
     contiguous chunks of rows across the processors.  Determine which
     rows of the matrix are locally owned. 
  */
  ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);

  /* 
     Set matrix elements for the 3-D, seven-point stencil in parallel.
      - Each processor needs to insert only elements that it owns
        locally (but any non-local elements will be sent to the
        appropriate processor during matrix assembly). 
      - Always specify global rows and columns of matrix entries.
   */
  ierr = PetscLogStageRegister("Assembly", &stage);CHKERRQ(ierr);
  ierr = PetscLogStagePush(stage);CHKERRQ(ierr);
  co1 = gamma * h * h / 2.0;
  co2 = beta * h * h;
  for (Ii=Istart; Ii<Iend; Ii++) { 
    i = Ii/(n2*n3); j = (Ii - i*n2*n3)/n3; k = Ii - i*n2*n3 - j*n3;
    if (i>0)   {
      J = Ii - n2*n3;  v = -1.0 + co1*(PetscScalar)i;
      ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
    if (i<n1-1) {
      J = Ii + n2*n3;  v = -1.0 + co1*(PetscScalar)i;
      ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
    if (j>0)   {
      J = Ii - n3;  v = -1.0 + co1*(PetscScalar)j;
      ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
    if (j<n2-1) {
      J = Ii + n3;  v = -1.0 + co1*(PetscScalar)j;
      ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
    if (k>0)   {
      J = Ii - 1;  v = -1.0 + co1*(PetscScalar)k;
      ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
    if (k<n3-1) {
      J = Ii + 1;  v = -1.0 + co1*(PetscScalar)k;
      ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
    v = 6.0 + co2;
    ierr = MatSetValues(A,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
  }

  /* 
     Assemble matrix, using the 2-step process:
       MatAssemblyBegin(), MatAssemblyEnd()
     Computations can be done while messages are in transition
     by placing code between these two statements.
  */
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr);

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
  ierr = VecCreate(PETSC_COMM_WORLD,&b);CHKERRQ(ierr);
  ierr = VecSetSizes(b,PETSC_DECIDE,n1*n2*n3);CHKERRQ(ierr);
  ierr = VecSetFromOptions(b);CHKERRQ(ierr);
  ierr = VecDuplicate(b,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(b,&u);CHKERRQ(ierr);

  /* 
     Set right-hand side.
  */
  ierr = VecSet(b,1.0);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
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
     Set linear solver defaults for this problem (optional).
     - By extracting the KSP and PC contexts from the KSP context,
       we can then directly call any KSP and PC routines to set
       various options.
  */
  ierr = KSPSetTolerances(ksp,1.e-6,1.e-50,PETSC_DEFAULT,200);CHKERRQ(ierr);

  /* 
    Set runtime options, e.g.,
        -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
    These options will override those specified above as long as
    KSPSetFromOptions() is called _after_ any other customization
    routines.
  */
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                      Clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);  ierr = MatDestroy(&A);CHKERRQ(ierr);

  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_summary). 
  */
  ierr = PetscFinalize();
  return 0;
}
