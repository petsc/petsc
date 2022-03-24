static char help[] = "Test sequential FFTW interface \n\n";

/*
  Compiling the code:
      This code uses the complex numbers version of PETSc, so configure
      must be run to enable this

*/

#include <petscmat.h>
int main(int argc,char **args)
{
  typedef enum {RANDOM, CONSTANT, TANH, NUM_FUNCS} FuncType;
  const char     *funcNames[NUM_FUNCS] = {"random", "constant", "tanh"};
  Mat            A;
  PetscMPIInt    size;
  PetscInt       n = 10,N,ndim=4,dim[4],DIM,i;
  Vec            x,y,z;
  PetscScalar    s;
  PetscRandom    rdm;
  PetscReal      enorm, tol = PETSC_SMALL;
  PetscInt       func;
  FuncType       function = RANDOM;
  PetscBool      view     = PETSC_FALSE;
  PetscErrorCode ierr;

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheckFalse(size != 1,PETSC_COMM_WORLD,PETSC_ERR_SUP, "This is a uniprocessor example only!");
  ierr     = PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "FFTW Options", "ex112");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsEList("-function", "Function type", "ex112", funcNames, NUM_FUNCS, funcNames[function], &func, NULL));
  CHKERRQ(PetscOptionsBool("-vec_view_draw", "View the functions", "ex112", view, &view, NULL));
  function = (FuncType) func;
  ierr     = PetscOptionsEnd();CHKERRQ(ierr);

  for (DIM = 0; DIM < ndim; DIM++) {
    dim[DIM] = n;  /* size of transformation in DIM-dimension */
  }
  CHKERRQ(PetscRandomCreate(PETSC_COMM_SELF, &rdm));
  CHKERRQ(PetscRandomSetFromOptions(rdm));

  for (DIM = 1; DIM < 5; DIM++) {
    for (i = 0, N = 1; i < DIM; i++) N *= dim[i];
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "\n %d-D: FFTW on vector of size %d \n",DIM,N));

    /* create FFTW object */
    CHKERRQ(MatCreateFFT(PETSC_COMM_SELF,DIM,dim,MATFFTW,&A));

    /* create vectors of length N=n^DIM */
    CHKERRQ(MatCreateVecs(A,&x,&y));
    CHKERRQ(MatCreateVecs(A,&z,NULL));
    CHKERRQ(PetscObjectSetName((PetscObject) x, "Real space vector"));
    CHKERRQ(PetscObjectSetName((PetscObject) y, "Frequency space vector"));
    CHKERRQ(PetscObjectSetName((PetscObject) z, "Reconstructed vector"));

    /* set values of space vector x */
    if (function == RANDOM) {
      CHKERRQ(VecSetRandom(x, rdm));
    } else if (function == CONSTANT) {
      CHKERRQ(VecSet(x, 1.0));
    } else if (function == TANH) {
      PetscScalar *a;
      CHKERRQ(VecGetArray(x, &a));
      for (i = 0; i < N; ++i) {
        a[i] = tanh((i - N/2.0)*(10.0/N));
      }
      CHKERRQ(VecRestoreArray(x, &a));
    }
    if (view) CHKERRQ(VecView(x, PETSC_VIEWER_DRAW_WORLD));

    /* apply FFTW_FORWARD and FFTW_BACKWARD several times on same x, y, and z */
    for (i=0; i<3; i++) {
      CHKERRQ(MatMult(A,x,y));
      if (view && i == 0) CHKERRQ(VecView(y, PETSC_VIEWER_DRAW_WORLD));
      CHKERRQ(MatMultTranspose(A,y,z));

      /* compare x and z. FFTW computes an unnormalized DFT, thus z = N*x */
      s    = 1.0/(PetscReal)N;
      CHKERRQ(VecScale(z,s));
      if (view && i == 0) CHKERRQ(VecView(z, PETSC_VIEWER_DRAW_WORLD));
      CHKERRQ(VecAXPY(z,-1.0,x));
      CHKERRQ(VecNorm(z,NORM_1,&enorm));
      if (enorm > tol) {
        CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"  Error norm of |x - z| %g\n",(double)enorm));
      }
    }

    /* apply FFTW_FORWARD and FFTW_BACKWARD several times on different x */
    for (i=0; i<3; i++) {
      CHKERRQ(VecDestroy(&x));
      CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,N,&x));
      CHKERRQ(VecSetRandom(x, rdm));

      CHKERRQ(MatMult(A,x,y));
      CHKERRQ(MatMultTranspose(A,y,z));

      /* compare x and z. FFTW computes an unnormalized DFT, thus z = N*x */
      s    = 1.0/(PetscReal)N;
      CHKERRQ(VecScale(z,s));
      if (view && i == 0) CHKERRQ(VecView(z, PETSC_VIEWER_DRAW_WORLD));
      CHKERRQ(VecAXPY(z,-1.0,x));
      CHKERRQ(VecNorm(z,NORM_1,&enorm));
      if (enorm > tol) {
        CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"  Error norm of new |x - z| %g\n",(double)enorm));
      }
    }

    /* free spaces */
    CHKERRQ(VecDestroy(&x));
    CHKERRQ(VecDestroy(&y));
    CHKERRQ(VecDestroy(&z));
    CHKERRQ(MatDestroy(&A));
  }
  CHKERRQ(PetscRandomDestroy(&rdm));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires:  fftw complex

   test:
      args: -mat_fftw_plannerflags FFTW_ESTIMATE
      output_file: output/ex112.out

   test:
      suffix: 2
      args: -mat_fftw_plannerflags FFTW_MEASURE
      output_file: output/ex112.out
      requires: !defined(PETSC_USE_CXXCOMPLEX)

   test:
      suffix: 3
      args: -mat_fftw_plannerflags FFTW_PATIENT
      output_file: output/ex112.out

   test:
      suffix: 4
      args: -mat_fftw_plannerflags FFTW_EXHAUSTIVE
      output_file: output/ex112.out

TEST*/
