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

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE, "This is a uniprocessor example only!");
  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "FFTW Options", "ex112");
  PetscCall(PetscOptionsEList("-function", "Function type", "ex121", funcNames, NUM_FUNCS, funcNames[function], &func, NULL));
  PetscCall(PetscOptionsBool("-vec_view draw", "View the functions", "ex112", view, &view, NULL));
  function = (FuncType) func;
  PetscOptionsEnd();

  for (DIM = 0; DIM < ndim; DIM++) {
    dim[DIM] = n;  /* size of transformation in DIM-dimension */
  }
  PetscCall(PetscRandomCreate(PETSC_COMM_SELF, &rdm));
  PetscCall(PetscRandomSetFromOptions(rdm));

  for (DIM = 1; DIM < 5; DIM++) {
    /* create vectors of length N=n^DIM */
    for (i = 0, N = 1; i < DIM; i++) N *= dim[i];
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "\n %d-D: FFTW on vector of size %d \n",DIM,N));
    PetscCall(VecCreateSeq(PETSC_COMM_SELF,N,&x));
    PetscCall(PetscObjectSetName((PetscObject) x, "Real space vector"));
    PetscCall(VecDuplicate(x,&w));
    PetscCall(PetscObjectSetName((PetscObject) w, "Window vector"));
    PetscCall(VecDuplicate(x,&y1));
    PetscCall(PetscObjectSetName((PetscObject) y1, "Frequency space vector"));
    PetscCall(VecDuplicate(x,&y2));
    PetscCall(PetscObjectSetName((PetscObject) y2, "Frequency space window vector"));
    PetscCall(VecDuplicate(x,&z1));
    PetscCall(PetscObjectSetName((PetscObject) z1, "Reconstructed convolution"));
    PetscCall(VecDuplicate(x,&z2));
    PetscCall(PetscObjectSetName((PetscObject) z2, "Real space convolution"));

    if (function == RANDOM) {
      PetscCall(VecSetRandom(x, rdm));
    } else if (function == CONSTANT) {
      PetscCall(VecSet(x, 1.0));
    } else if (function == TANH) {
      PetscCall(VecGetArray(x, &a));
      for (i = 0; i < N; ++i) {
        a[i] = tanh((i - N/2.0)*(10.0/N));
      }
      PetscCall(VecRestoreArray(x, &a));
    }
    if (view) PetscCall(VecView(x, PETSC_VIEWER_DRAW_WORLD));

    /* Create window function */
    PetscCall(VecGetArray(w, &a));
    for (i = 0; i < N; ++i) {
      /* Step Function */
      a[i] = (i > N/4 && i < 3*N/4) ? 1.0 : 0.0;
      /* Delta Function */
      /*a[i] = (i == N/2)? 1.0: 0.0; */
    }
    PetscCall(VecRestoreArray(w, &a));
    if (view) PetscCall(VecView(w, PETSC_VIEWER_DRAW_WORLD));

    /* create FFTW object */
    PetscCall(MatCreateFFT(PETSC_COMM_SELF,DIM,dim,MATFFTW,&A));

    /* Convolve x with w*/
    PetscCall(MatMult(A,x,y1));
    PetscCall(MatMult(A,w,y2));
    PetscCall(VecPointwiseMult(y1, y1, y2));
    if (view && i == 0) PetscCall(VecView(y1, PETSC_VIEWER_DRAW_WORLD));
    PetscCall(MatMultTranspose(A,y1,z1));

    /* Compute the real space convolution */
    PetscCall(VecGetArray(x, &a));
    PetscCall(VecGetArray(w, &a2));
    PetscCall(VecGetArray(z2, &a3));
    for (i = 0; i < N; ++i) {
      /* PetscInt checkInd = (i > N/2-1)? i-N/2: i+N/2;*/

      a3[i] = 0.0;
      for (j = -N/2+1; j < N/2; ++j) {
        PetscInt xpInd   = (j < 0) ? N+j : j;
        PetscInt diffInd = (i-j < 0) ? N-(j-i) : (i-j > N-1) ? i-j-N : i-j;

        a3[i] += a[xpInd]*a2[diffInd];
      }
    }
    PetscCall(VecRestoreArray(x, &a));
    PetscCall(VecRestoreArray(w, &a2));
    PetscCall(VecRestoreArray(z2, &a3));

    /* compare z1 and z2. FFTW computes an unnormalized DFT, thus z1 = N*z2 */
    s    = 1.0/(PetscReal)N;
    PetscCall(VecScale(z1,s));
    if (view) PetscCall(VecView(z1, PETSC_VIEWER_DRAW_WORLD));
    if (view) PetscCall(VecView(z2, PETSC_VIEWER_DRAW_WORLD));
    PetscCall(VecAXPY(z1,-1.0,z2));
    PetscCall(VecNorm(z1,NORM_1,&enorm));
    if (enorm > 1.e-11) {
      PetscCall(PetscPrintf(PETSC_COMM_SELF,"  Error norm of |z1 - z2| %g\n",(double)enorm));
    }

    /* free spaces */
    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroy(&y1));
    PetscCall(VecDestroy(&y2));
    PetscCall(VecDestroy(&z1));
    PetscCall(VecDestroy(&z2));
    PetscCall(VecDestroy(&w));
    PetscCall(MatDestroy(&A));
  }
  PetscCall(PetscRandomDestroy(&rdm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: fftw complex

   test:
      output_file: output/ex121.out
      TODO: Example or FFTW interface is broken

TEST*/
