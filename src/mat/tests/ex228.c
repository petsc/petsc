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
  PetscErrorCode ierr;      /* to catch bugs, if any */
  PetscInt       n=10,N=1;  /* FFT dimension params */
  PetscInt       DIM,dim[5];/* FFT params */

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);

  /* To create random input vector */
  ierr = PetscRandomCreate(PETSC_COMM_SELF, &rdm);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rdm);CHKERRQ(ierr);

  /* Iterate over dimensions, use PETSc-FFTW interface */
  for (i=1; i<5; i++) {
    DIM = i;
    N = 1;
    for (k=0; k<i; k++){dim[k] = n; N*=n;}

    ierr = PetscPrintf(PETSC_COMM_WORLD, "\n %" PetscInt_FMT " dimensions: FFTW on vector of size %" PetscInt_FMT " \n",DIM,N);CHKERRQ(ierr);

    /* create FFTW object */
    ierr = MatCreateFFT(PETSC_COMM_SELF,DIM,dim,MATFFTW,&A);CHKERRQ(ierr);
    /* create vectors of length N */
    ierr = MatCreateVecsFFTW(A,&x,&y,&z);CHKERRQ(ierr);

    ierr = PetscObjectSetName((PetscObject) x, "Real space vector");CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) y, "Frequency space vector");CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) z, "Reconstructed vector");CHKERRQ(ierr);

    /* Test vector duplication*/
    ierr = VecDuplicate(x,&x1);CHKERRQ(ierr);
    ierr = VecDuplicate(y,&y1);CHKERRQ(ierr);
    ierr = VecDuplicate(z,&z1);CHKERRQ(ierr);

    /* Set values of space vector x, copy to duplicate */
    ierr = VecSetRandom(x,rdm);CHKERRQ(ierr);
    ierr = VecCopy(x,x1);CHKERRQ(ierr);

    /* Apply FFTW_FORWARD and FFTW_BACKWARD */
    ierr = MatMult(A,x,y);CHKERRQ(ierr);
    ierr = MatMultTranspose(A,y,z);CHKERRQ(ierr);

    /* Apply FFTW_FORWARD and FFTW_BACKWARD for duplicate vecs */
    ierr = MatMult(A,x1,y1);CHKERRQ(ierr);
    ierr = MatMultTranspose(A,y1,z1);CHKERRQ(ierr);

    /* Compare x and z1. FFTW computes an unnormalized DFT, thus z1 = N*x */
    a    = 1.0/(PetscReal)N;
    ierr = VecScale(z1,a);CHKERRQ(ierr);
    ierr = VecAXPY(z1,-1.0,x);CHKERRQ(ierr);
    ierr = VecNorm(z1,NORM_1,&enorm);CHKERRQ(ierr);
    if (enorm > 1.e-9){ierr = PetscPrintf(PETSC_COMM_WORLD,"  Error norm of |x - z1| %g\n",enorm);CHKERRQ(ierr);}

    /* free spaces */
    ierr = VecDestroy(&x1);CHKERRQ(ierr);
    ierr = VecDestroy(&y1);CHKERRQ(ierr);
    ierr = VecDestroy(&z1);CHKERRQ(ierr);
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
