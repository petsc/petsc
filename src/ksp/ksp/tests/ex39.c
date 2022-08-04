/*
mpiexec -n 8 ./ex39 -ksp_type fbcgs -ksp_rtol 1.e-6 -sub_ksp_type bcgs -sub_ksp_rtol 1.e-3 -pc_type bjacobi -ksp_converged_reason -ksp_monitor -n1 32 -n2 32 -n3 32

  Contributed by Jie Chen for testing flexible BiCGStab algorithm
*/

static char help[] = "Solves the PDE (in 3D) - laplacian(u) + gamma x dot grad(u) + beta u = 1\n\
with zero Dirichlet condition. The discretization is standard centered\n\
difference. Input parameters include:\n\
  -n1        : number of mesh points in 1st dimension (default 32)\n\
  -n2        : number of mesh points in 2nd dimension (default 32)\n\
  -n3        : number of mesh points in 3rd dimension (default 32)\n\
  -h         : spacing between mesh points (default 1/n1)\n\
  -gamma     : gamma (default 4/h)\n\
  -beta      : beta (default 0.01/h^2)\n\n";

#include <petscksp.h>
int main(int argc,char **args)
{
  Vec            x,b,u;                 /* approx solution, RHS, working vector */
  Mat            A;                     /* linear system matrix */
  KSP            ksp;                   /* linear solver context */
  PetscInt       n1, n2, n3;            /* parameters */
  PetscReal      h, gamma, beta;        /* parameters */
  PetscInt       i,j,k,Ii,J,Istart,Iend;
  PetscScalar    v, co1, co2;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  n1 = 32;
  n2 = 32;
  n3 = 32;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n1",&n1,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n2",&n2,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n3",&n3,NULL));

  h     = 1.0/n1;
  gamma = 4.0/h;
  beta  = 0.01/(h*h);
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-h",&h,NULL));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-gamma",&gamma,NULL));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-beta",&beta,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Compute the matrix and set right-hand-side vector.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n1*n2*n3,n1*n2*n3));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatMPIAIJSetPreallocation(A,7,NULL,7,NULL));
  PetscCall(MatSeqAIJSetPreallocation(A,7,NULL));
  PetscCall(MatSetUp(A));
  PetscCall(MatGetOwnershipRange(A,&Istart,&Iend));

  /*
     Set matrix elements for the 3-D, seven-point stencil in parallel.
      - Each processor needs to insert only elements that it owns
        locally (but any non-local elements will be sent to the
        appropriate processor during matrix assembly).
      - Always specify global rows and columns of matrix entries.
   */
  co1  = gamma * h * h / 2.0;
  co2  = beta * h * h;
  for (Ii=Istart; Ii<Iend; Ii++) {
    i = Ii/(n2*n3); j = (Ii - i*n2*n3)/n3; k = Ii - i*n2*n3 - j*n3;
    if (i>0) {
      J    = Ii - n2*n3;  v = -1.0 + co1*(PetscScalar)i;
      PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));
    }
    if (i<n1-1) {
      J    = Ii + n2*n3;  v = -1.0 + co1*(PetscScalar)i;
      PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));
    }
    if (j>0) {
      J    = Ii - n3;  v = -1.0 + co1*(PetscScalar)j;
      PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));
    }
    if (j<n2-1) {
      J    = Ii + n3;  v = -1.0 + co1*(PetscScalar)j;
      PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));
    }
    if (k>0) {
      J    = Ii - 1;  v = -1.0 + co1*(PetscScalar)k;
      PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));
    }
    if (k<n3-1) {
      J    = Ii + 1;  v = -1.0 + co1*(PetscScalar)k;
      PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));
    }
    v    = 6.0 + co2;
    PetscCall(MatSetValues(A,1,&Ii,1,&Ii,&v,INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* Create parallel vectors and Set right-hand side. */
  PetscCall(VecCreate(PETSC_COMM_WORLD,&b));
  PetscCall(VecSetSizes(b,PETSC_DECIDE,n1*n2*n3));
  PetscCall(VecSetFromOptions(b));
  PetscCall(VecDuplicate(b,&x));
  PetscCall(VecDuplicate(b,&u));
  PetscCall(VecSet(b,1.0));

  /* Create linear solver context */
  PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
  PetscCall(KSPSetOperators(ksp,A,A));
  PetscCall(KSPSetTolerances(ksp,1.e-6,PETSC_DEFAULT,PETSC_DEFAULT,200));
  PetscCall(KSPSetFromOptions(ksp));

  /* Solve the linear system */
  PetscCall(KSPSolve(ksp,b,x));

  /* Free work space.  */
  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&u));  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 8
      args: -ksp_type fbcgs -ksp_rtol 1.e-6 -sub_ksp_type bcgs -sub_ksp_rtol 1.e-3 -pc_type bjacobi -ksp_converged_reason -n1 32 -n2 32 -n3 32

   test:
      suffix: 2
      nsize: 8
      args: -ksp_type fbcgsr -ksp_rtol 1.e-6 -sub_ksp_type bcgs -sub_ksp_rtol 1.e-3 -pc_type bjacobi -ksp_converged_reason -n1 32 -n2 32 -n3 32
      output_file: output/ex39_1.out

   test:
      suffix: 3
      nsize: 8
      args: -ksp_type qmrcgs -ksp_rtol 1.e-6 -sub_ksp_type bcgs -sub_ksp_rtol 1.e-3 -pc_type bjacobi -ksp_converged_reason -n1 32 -n2 32 -n3 32
      output_file: output/ex39_1.out

TEST*/
