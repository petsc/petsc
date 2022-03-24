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

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));

  /* To create random input vector */
  CHKERRQ(PetscRandomCreate(PETSC_COMM_SELF, &rdm));
  CHKERRQ(PetscRandomSetFromOptions(rdm));

  /* Iterate over dimensions, use PETSc-FFTW interface */
  for (i=1; i<5; i++) {
    DIM = i;
    N = 1;
    for (k=0; k<i; k++){dim[k] = n; N*=n;}

    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "\n %" PetscInt_FMT " dimensions: FFTW on vector of size %" PetscInt_FMT " \n",DIM,N));

    /* create FFTW object */
    CHKERRQ(MatCreateFFT(PETSC_COMM_SELF,DIM,dim,MATFFTW,&A));
    /* create vectors of length N */
    CHKERRQ(MatCreateVecsFFTW(A,&x,&y,&z));

    CHKERRQ(PetscObjectSetName((PetscObject) x, "Real space vector"));
    CHKERRQ(PetscObjectSetName((PetscObject) y, "Frequency space vector"));
    CHKERRQ(PetscObjectSetName((PetscObject) z, "Reconstructed vector"));

    /* Test vector duplication*/
    CHKERRQ(VecDuplicate(x,&x1));
    CHKERRQ(VecDuplicate(y,&y1));
    CHKERRQ(VecDuplicate(z,&z1));

    /* Set values of space vector x, copy to duplicate */
    CHKERRQ(VecSetRandom(x,rdm));
    CHKERRQ(VecCopy(x,x1));

    /* Apply FFTW_FORWARD and FFTW_BACKWARD */
    CHKERRQ(MatMult(A,x,y));
    CHKERRQ(MatMultTranspose(A,y,z));

    /* Apply FFTW_FORWARD and FFTW_BACKWARD for duplicate vecs */
    CHKERRQ(MatMult(A,x1,y1));
    CHKERRQ(MatMultTranspose(A,y1,z1));

    /* Compare x and z1. FFTW computes an unnormalized DFT, thus z1 = N*x */
    a    = 1.0/(PetscReal)N;
    CHKERRQ(VecScale(z1,a));
    CHKERRQ(VecAXPY(z1,-1.0,x));
    CHKERRQ(VecNorm(z1,NORM_1,&enorm));
    if (enorm > 1.e-9) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"  Error norm of |x - z1| %g\n",enorm));

    /* free spaces */
    CHKERRQ(VecDestroy(&x1));
    CHKERRQ(VecDestroy(&y1));
    CHKERRQ(VecDestroy(&z1));
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
