static char help[] = "Test sequential FFTW convolution\n\n";

/*
  Compiling the code:
    This code uses the complex numbers, so configure must be given --with-scalar-type=complex to enable this
*/

#include <petscmat.h>

int main(int argc,char **args)
{
  typedef enum {RANDOM, CONSTANT, TANH, NUM_FUNCS} FuncType;
  const char     *funcNames[NUM_FUNCS] = {"random", "constant", "tanh"};
  Mat            A;
  PetscMPIInt    size;
  PetscInt       n = 10,N,ndim=4,dim[4],DIM,i,j;
  Vec            w,x,y1,y2,z1,z2;
  PetscScalar    *a, *a2, *a3;
  PetscScalar    s;
  PetscRandom    rdm;
  PetscReal      enorm;
  PetscInt       func     = 0;
  FuncType       function = RANDOM;
  PetscBool      view     = PETSC_FALSE;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheckFalse(size != 1,PETSC_COMM_WORLD,PETSC_ERR_SUP, "This is a uniprocessor example only!");
  ierr     = PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "FFTW Options", "ex112");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsEList("-function", "Function type", "ex121", funcNames, NUM_FUNCS, funcNames[function], &func, NULL));
  CHKERRQ(PetscOptionsBool("-vec_view draw", "View the functions", "ex112", view, &view, NULL));
  function = (FuncType) func;
  ierr     = PetscOptionsEnd();CHKERRQ(ierr);

  for (DIM = 0; DIM < ndim; DIM++) {
    dim[DIM] = n;  /* size of transformation in DIM-dimension */
  }
  CHKERRQ(PetscRandomCreate(PETSC_COMM_SELF, &rdm));
  CHKERRQ(PetscRandomSetFromOptions(rdm));

  for (DIM = 1; DIM < 5; DIM++) {
    /* create vectors of length N=n^DIM */
    for (i = 0, N = 1; i < DIM; i++) N *= dim[i];
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "\n %d-D: FFTW on vector of size %d \n",DIM,N));
    CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,N,&x));
    CHKERRQ(PetscObjectSetName((PetscObject) x, "Real space vector"));
    CHKERRQ(VecDuplicate(x,&w));
    CHKERRQ(PetscObjectSetName((PetscObject) w, "Window vector"));
    CHKERRQ(VecDuplicate(x,&y1));
    CHKERRQ(PetscObjectSetName((PetscObject) y1, "Frequency space vector"));
    CHKERRQ(VecDuplicate(x,&y2));
    CHKERRQ(PetscObjectSetName((PetscObject) y2, "Frequency space window vector"));
    CHKERRQ(VecDuplicate(x,&z1));
    CHKERRQ(PetscObjectSetName((PetscObject) z1, "Reconstructed convolution"));
    CHKERRQ(VecDuplicate(x,&z2));
    CHKERRQ(PetscObjectSetName((PetscObject) z2, "Real space convolution"));

    if (function == RANDOM) {
      CHKERRQ(VecSetRandom(x, rdm));
    } else if (function == CONSTANT) {
      CHKERRQ(VecSet(x, 1.0));
    } else if (function == TANH) {
      CHKERRQ(VecGetArray(x, &a));
      for (i = 0; i < N; ++i) {
        a[i] = tanh((i - N/2.0)*(10.0/N));
      }
      CHKERRQ(VecRestoreArray(x, &a));
    }
    if (view) CHKERRQ(VecView(x, PETSC_VIEWER_DRAW_WORLD));

    /* Create window function */
    CHKERRQ(VecGetArray(w, &a));
    for (i = 0; i < N; ++i) {
      /* Step Function */
      a[i] = (i > N/4 && i < 3*N/4) ? 1.0 : 0.0;
      /* Delta Function */
      /*a[i] = (i == N/2)? 1.0: 0.0; */
    }
    CHKERRQ(VecRestoreArray(w, &a));
    if (view) CHKERRQ(VecView(w, PETSC_VIEWER_DRAW_WORLD));

    /* create FFTW object */
    CHKERRQ(MatCreateFFT(PETSC_COMM_SELF,DIM,dim,MATFFTW,&A));

    /* Convolve x with w*/
    CHKERRQ(MatMult(A,x,y1));
    CHKERRQ(MatMult(A,w,y2));
    CHKERRQ(VecPointwiseMult(y1, y1, y2));
    if (view && i == 0) CHKERRQ(VecView(y1, PETSC_VIEWER_DRAW_WORLD));
    CHKERRQ(MatMultTranspose(A,y1,z1));

    /* Compute the real space convolution */
    CHKERRQ(VecGetArray(x, &a));
    CHKERRQ(VecGetArray(w, &a2));
    CHKERRQ(VecGetArray(z2, &a3));
    for (i = 0; i < N; ++i) {
      /* PetscInt checkInd = (i > N/2-1)? i-N/2: i+N/2;*/

      a3[i] = 0.0;
      for (j = -N/2+1; j < N/2; ++j) {
        PetscInt xpInd   = (j < 0) ? N+j : j;
        PetscInt diffInd = (i-j < 0) ? N-(j-i) : (i-j > N-1) ? i-j-N : i-j;

        a3[i] += a[xpInd]*a2[diffInd];
      }
    }
    CHKERRQ(VecRestoreArray(x, &a));
    CHKERRQ(VecRestoreArray(w, &a2));
    CHKERRQ(VecRestoreArray(z2, &a3));

    /* compare z1 and z2. FFTW computes an unnormalized DFT, thus z1 = N*z2 */
    s    = 1.0/(PetscReal)N;
    CHKERRQ(VecScale(z1,s));
    if (view) CHKERRQ(VecView(z1, PETSC_VIEWER_DRAW_WORLD));
    if (view) CHKERRQ(VecView(z2, PETSC_VIEWER_DRAW_WORLD));
    CHKERRQ(VecAXPY(z1,-1.0,z2));
    CHKERRQ(VecNorm(z1,NORM_1,&enorm));
    if (enorm > 1.e-11) {
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"  Error norm of |z1 - z2| %g\n",(double)enorm));
    }

    /* free spaces */
    CHKERRQ(VecDestroy(&x));
    CHKERRQ(VecDestroy(&y1));
    CHKERRQ(VecDestroy(&y2));
    CHKERRQ(VecDestroy(&z1));
    CHKERRQ(VecDestroy(&z2));
    CHKERRQ(VecDestroy(&w));
    CHKERRQ(MatDestroy(&A));
  }
  CHKERRQ(PetscRandomDestroy(&rdm));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
      requires: fftw complex

   test:
      output_file: output/ex121.out
      TODO: Example or FFTW interface is broken

TEST*/
