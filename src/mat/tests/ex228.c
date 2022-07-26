static char help[] = "Test duplication/destruction of FFTW vecs \n\n";

/*
 Compiling the code:
   This code uses the FFTW interface.
   Use one of the options below to configure:
   --with-fftw-dir=/.... or --download-fftw
 Usage:
   mpiexec -np <np> ./ex228
*/

#include <petscmat.h>
int main(int argc,char **args)
{
  Mat            A;         /* FFT Matrix */
  Vec            x,y,z;     /* Work vectors */
  Vec            x1,y1,z1;  /* Duplicate vectors */
  PetscInt       i,k;       /* for iterating over dimensions */
  PetscRandom    rdm;       /* for creating random input */
  PetscScalar    a;         /* used to scale output */
  PetscReal      enorm;     /* norm for sanity check */
  PetscInt       n=10,N=1;  /* FFT dimension params */
  PetscInt       DIM,dim[5];/* FFT params */

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));

  /* To create random input vector */
  PetscCall(PetscRandomCreate(PETSC_COMM_SELF, &rdm));
  PetscCall(PetscRandomSetFromOptions(rdm));

  /* Iterate over dimensions, use PETSc-FFTW interface */
  for (i=1; i<5; i++) {
    DIM = i;
    N = 1;
    for (k=0; k<i; k++){dim[k] = n; N*=n;}

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n %" PetscInt_FMT " dimensions: FFTW on vector of size %" PetscInt_FMT " \n",DIM,N));

    /* create FFTW object */
    PetscCall(MatCreateFFT(PETSC_COMM_SELF,DIM,dim,MATFFTW,&A));
    /* create vectors of length N */
    PetscCall(MatCreateVecsFFTW(A,&x,&y,&z));

    PetscCall(PetscObjectSetName((PetscObject) x, "Real space vector"));
    PetscCall(PetscObjectSetName((PetscObject) y, "Frequency space vector"));
    PetscCall(PetscObjectSetName((PetscObject) z, "Reconstructed vector"));

    /* Test vector duplication*/
    PetscCall(VecDuplicate(x,&x1));
    PetscCall(VecDuplicate(y,&y1));
    PetscCall(VecDuplicate(z,&z1));

    /* Set values of space vector x, copy to duplicate */
    PetscCall(VecSetRandom(x,rdm));
    PetscCall(VecCopy(x,x1));

    /* Apply FFTW_FORWARD and FFTW_BACKWARD */
    PetscCall(MatMult(A,x,y));
    PetscCall(MatMultTranspose(A,y,z));

    /* Apply FFTW_FORWARD and FFTW_BACKWARD for duplicate vecs */
    PetscCall(MatMult(A,x1,y1));
    PetscCall(MatMultTranspose(A,y1,z1));

    /* Compare x and z1. FFTW computes an unnormalized DFT, thus z1 = N*x */
    a    = 1.0/(PetscReal)N;
    PetscCall(VecScale(z1,a));
    PetscCall(VecAXPY(z1,-1.0,x));
    PetscCall(VecNorm(z1,NORM_1,&enorm));
    if (enorm > 1.e-9) PetscCall(PetscPrintf(PETSC_COMM_WORLD,"  Error norm of |x - z1| %g\n",enorm));

    /* free spaces */
    PetscCall(VecDestroy(&x1));
    PetscCall(VecDestroy(&y1));
    PetscCall(VecDestroy(&z1));
    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroy(&y));
    PetscCall(VecDestroy(&z));
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
      suffix: 2
      nsize : 4
      args: -mat_fftw_plannerflags FFTW_ESTIMATE -n 16

    test:
      suffix: 3
      nsize : 2
      args: -mat_fftw_plannerflags FFTW_MEASURE -n 12

    test:
      suffix: 4
      nsize : 2
      args: -mat_fftw_plannerflags FFTW_PATIENT -n 10

    test:
      suffix: 5
      nsize : 1
      args: -mat_fftw_plannerflags FFTW_EXHAUSTIVE -n 5

TEST*/
