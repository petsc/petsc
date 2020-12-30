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

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);CHKERRMPI(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP, "This is a uniprocessor example only!");
  ierr     = PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "FFTW Options", "ex112");CHKERRQ(ierr);
  ierr     = PetscOptionsEList("-function", "Function type", "ex112", funcNames, NUM_FUNCS, funcNames[function], &func, NULL);CHKERRQ(ierr);
  ierr     = PetscOptionsBool("-vec_view_draw", "View the functions", "ex112", view, &view, NULL);CHKERRQ(ierr);
  function = (FuncType) func;
  ierr     = PetscOptionsEnd();CHKERRQ(ierr);

  for (DIM = 0; DIM < ndim; DIM++) {
    dim[DIM] = n;  /* size of transformation in DIM-dimension */
  }
  ierr = PetscRandomCreate(PETSC_COMM_SELF, &rdm);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rdm);CHKERRQ(ierr);

  for (DIM = 1; DIM < 5; DIM++) {
    for (i = 0, N = 1; i < DIM; i++) N *= dim[i];
    ierr = PetscPrintf(PETSC_COMM_SELF, "\n %d-D: FFTW on vector of size %d \n",DIM,N);CHKERRQ(ierr);

    /* create FFTW object */
    ierr = MatCreateFFT(PETSC_COMM_SELF,DIM,dim,MATFFTW,&A);CHKERRQ(ierr);

    /* create vectors of length N=n^DIM */
    ierr = MatCreateVecs(A,&x,&y);CHKERRQ(ierr);
    ierr = MatCreateVecs(A,&z,NULL);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) x, "Real space vector");CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) y, "Frequency space vector");CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) z, "Reconstructed vector");CHKERRQ(ierr);

    /* set values of space vector x */
    if (function == RANDOM) {
      ierr = VecSetRandom(x, rdm);CHKERRQ(ierr);
    } else if (function == CONSTANT) {
      ierr = VecSet(x, 1.0);CHKERRQ(ierr);
    } else if (function == TANH) {
      PetscScalar *a;
      ierr = VecGetArray(x, &a);CHKERRQ(ierr);
      for (i = 0; i < N; ++i) {
        a[i] = tanh((i - N/2.0)*(10.0/N));
      }
      ierr = VecRestoreArray(x, &a);CHKERRQ(ierr);
    }
    if (view) {ierr = VecView(x, PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}

    /* apply FFTW_FORWARD and FFTW_BACKWARD several times on same x, y, and z */
    for (i=0; i<3; i++) {
      ierr = MatMult(A,x,y);CHKERRQ(ierr);
      if (view && i == 0) {ierr = VecView(y, PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}
      ierr = MatMultTranspose(A,y,z);CHKERRQ(ierr);

      /* compare x and z. FFTW computes an unnormalized DFT, thus z = N*x */
      s    = 1.0/(PetscReal)N;
      ierr = VecScale(z,s);CHKERRQ(ierr);
      if (view && i == 0) {ierr = VecView(z, PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}
      ierr = VecAXPY(z,-1.0,x);CHKERRQ(ierr);
      ierr = VecNorm(z,NORM_1,&enorm);CHKERRQ(ierr);
      if (enorm > tol) {
        ierr = PetscPrintf(PETSC_COMM_SELF,"  Error norm of |x - z| %g\n",(double)enorm);CHKERRQ(ierr);
      }
    }

    /* apply FFTW_FORWARD and FFTW_BACKWARD several times on different x */
    for (i=0; i<3; i++) {
      ierr = VecDestroy(&x);CHKERRQ(ierr);
      ierr = VecCreateSeq(PETSC_COMM_SELF,N,&x);CHKERRQ(ierr);
      ierr = VecSetRandom(x, rdm);CHKERRQ(ierr);

      ierr = MatMult(A,x,y);CHKERRQ(ierr);
      ierr = MatMultTranspose(A,y,z);CHKERRQ(ierr);

      /* compare x and z. FFTW computes an unnormalized DFT, thus z = N*x */
      s    = 1.0/(PetscReal)N;
      ierr = VecScale(z,s);CHKERRQ(ierr);
      if (view && i == 0) {ierr = VecView(z, PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}
      ierr = VecAXPY(z,-1.0,x);CHKERRQ(ierr);
      ierr = VecNorm(z,NORM_1,&enorm);CHKERRQ(ierr);
      if (enorm > tol) {
        ierr = PetscPrintf(PETSC_COMM_SELF,"  Error norm of new |x - z| %g\n",(double)enorm);CHKERRQ(ierr);
      }
    }

    /* free spaces */
    ierr = VecDestroy(&x);CHKERRQ(ierr);
    ierr = VecDestroy(&y);CHKERRQ(ierr);
    ierr = VecDestroy(&z);CHKERRQ(ierr);
    ierr = MatDestroy(&A);CHKERRQ(ierr);
  }
  ierr = PetscRandomDestroy(&rdm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
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
      requires: !define(PETSC_USE_CXXCOMPLEX)

   test:
      suffix: 3
      args: -mat_fftw_plannerflags FFTW_PATIENT
      output_file: output/ex112.out

   test:
      suffix: 4
      args: -mat_fftw_plannerflags FFTW_EXHAUSTIVE
      output_file: output/ex112.out

TEST*/
