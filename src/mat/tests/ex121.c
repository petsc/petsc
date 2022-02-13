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
  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);CHKERRMPI(ierr);
  PetscCheckFalse(size != 1,PETSC_COMM_WORLD,PETSC_ERR_SUP, "This is a uniprocessor example only!");
  ierr     = PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "FFTW Options", "ex112");CHKERRQ(ierr);
  ierr     = PetscOptionsEList("-function", "Function type", "ex121", funcNames, NUM_FUNCS, funcNames[function], &func, NULL);CHKERRQ(ierr);
  ierr     = PetscOptionsBool("-vec_view draw", "View the functions", "ex112", view, &view, NULL);CHKERRQ(ierr);
  function = (FuncType) func;
  ierr     = PetscOptionsEnd();CHKERRQ(ierr);

  for (DIM = 0; DIM < ndim; DIM++) {
    dim[DIM] = n;  /* size of transformation in DIM-dimension */
  }
  ierr = PetscRandomCreate(PETSC_COMM_SELF, &rdm);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rdm);CHKERRQ(ierr);

  for (DIM = 1; DIM < 5; DIM++) {
    /* create vectors of length N=n^DIM */
    for (i = 0, N = 1; i < DIM; i++) N *= dim[i];
    ierr = PetscPrintf(PETSC_COMM_SELF, "\n %d-D: FFTW on vector of size %d \n",DIM,N);CHKERRQ(ierr);
    ierr = VecCreateSeq(PETSC_COMM_SELF,N,&x);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) x, "Real space vector");CHKERRQ(ierr);
    ierr = VecDuplicate(x,&w);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) w, "Window vector");CHKERRQ(ierr);
    ierr = VecDuplicate(x,&y1);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) y1, "Frequency space vector");CHKERRQ(ierr);
    ierr = VecDuplicate(x,&y2);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) y2, "Frequency space window vector");CHKERRQ(ierr);
    ierr = VecDuplicate(x,&z1);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) z1, "Reconstructed convolution");CHKERRQ(ierr);
    ierr = VecDuplicate(x,&z2);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) z2, "Real space convolution");CHKERRQ(ierr);

    if (function == RANDOM) {
      ierr = VecSetRandom(x, rdm);CHKERRQ(ierr);
    } else if (function == CONSTANT) {
      ierr = VecSet(x, 1.0);CHKERRQ(ierr);
    } else if (function == TANH) {
      ierr = VecGetArray(x, &a);CHKERRQ(ierr);
      for (i = 0; i < N; ++i) {
        a[i] = tanh((i - N/2.0)*(10.0/N));
      }
      ierr = VecRestoreArray(x, &a);CHKERRQ(ierr);
    }
    if (view) {ierr = VecView(x, PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}

    /* Create window function */
    ierr = VecGetArray(w, &a);CHKERRQ(ierr);
    for (i = 0; i < N; ++i) {
      /* Step Function */
      a[i] = (i > N/4 && i < 3*N/4) ? 1.0 : 0.0;
      /* Delta Function */
      /*a[i] = (i == N/2)? 1.0: 0.0; */
    }
    ierr = VecRestoreArray(w, &a);CHKERRQ(ierr);
    if (view) {ierr = VecView(w, PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}

    /* create FFTW object */
    ierr = MatCreateFFT(PETSC_COMM_SELF,DIM,dim,MATFFTW,&A);CHKERRQ(ierr);

    /* Convolve x with w*/
    ierr = MatMult(A,x,y1);CHKERRQ(ierr);
    ierr = MatMult(A,w,y2);CHKERRQ(ierr);
    ierr = VecPointwiseMult(y1, y1, y2);CHKERRQ(ierr);
    if (view && i == 0) {ierr = VecView(y1, PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}
    ierr = MatMultTranspose(A,y1,z1);CHKERRQ(ierr);

    /* Compute the real space convolution */
    ierr = VecGetArray(x, &a);CHKERRQ(ierr);
    ierr = VecGetArray(w, &a2);CHKERRQ(ierr);
    ierr = VecGetArray(z2, &a3);CHKERRQ(ierr);
    for (i = 0; i < N; ++i) {
      /* PetscInt checkInd = (i > N/2-1)? i-N/2: i+N/2;*/

      a3[i] = 0.0;
      for (j = -N/2+1; j < N/2; ++j) {
        PetscInt xpInd   = (j < 0) ? N+j : j;
        PetscInt diffInd = (i-j < 0) ? N-(j-i) : (i-j > N-1) ? i-j-N : i-j;

        a3[i] += a[xpInd]*a2[diffInd];
      }
    }
    ierr = VecRestoreArray(x, &a);CHKERRQ(ierr);
    ierr = VecRestoreArray(w, &a2);CHKERRQ(ierr);
    ierr = VecRestoreArray(z2, &a3);CHKERRQ(ierr);

    /* compare z1 and z2. FFTW computes an unnormalized DFT, thus z1 = N*z2 */
    s    = 1.0/(PetscReal)N;
    ierr = VecScale(z1,s);CHKERRQ(ierr);
    if (view) {ierr = VecView(z1, PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}
    if (view) {ierr = VecView(z2, PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}
    ierr = VecAXPY(z1,-1.0,z2);CHKERRQ(ierr);
    ierr = VecNorm(z1,NORM_1,&enorm);CHKERRQ(ierr);
    if (enorm > 1.e-11) {
      ierr = PetscPrintf(PETSC_COMM_SELF,"  Error norm of |z1 - z2| %g\n",(double)enorm);CHKERRQ(ierr);
    }

    /* free spaces */
    ierr = VecDestroy(&x);CHKERRQ(ierr);
    ierr = VecDestroy(&y1);CHKERRQ(ierr);
    ierr = VecDestroy(&y2);CHKERRQ(ierr);
    ierr = VecDestroy(&z1);CHKERRQ(ierr);
    ierr = VecDestroy(&z2);CHKERRQ(ierr);
    ierr = VecDestroy(&w);CHKERRQ(ierr);
    ierr = MatDestroy(&A);CHKERRQ(ierr);
  }
  ierr = PetscRandomDestroy(&rdm);CHKERRQ(ierr);
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
