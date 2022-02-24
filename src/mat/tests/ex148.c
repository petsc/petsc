static char help[]="This program illustrates the use of PETSc-fftw interface for real 2D DFT.\n\
                    See ~petsc/src/mat/tests/ex158.c for general cases. \n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank,size;
  PetscInt       N0=50,N1=50,N=N0*N1;
  PetscRandom    rdm;
  PetscReal      enorm;
  Vec            x,y,z,input,output;
  Mat            A;
  PetscInt       DIM,dim[2];
  PetscReal      fac;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
#if defined(PETSC_USE_COMPLEX)
  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP, "This example requires real numbers");
#endif

  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  /* Create and set PETSc vectors 'input' and 'output' */
  CHKERRQ(PetscRandomCreate(PETSC_COMM_WORLD, &rdm));
  CHKERRQ(PetscRandomSetFromOptions(rdm));

  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&input));
  CHKERRQ(VecSetSizes(input,PETSC_DECIDE,N0*N1));
  CHKERRQ(VecSetFromOptions(input));
  CHKERRQ(VecSetRandom(input,rdm));
  CHKERRQ(VecDuplicate(input,&output));
  CHKERRQ(PetscObjectSetName((PetscObject)input, "Real space vector"));
  CHKERRQ(PetscObjectSetName((PetscObject)output, "Reconstructed vector"));

  /* Get FFTW vectors 'x', 'y' and 'z' */
  DIM    = 2;
  dim[0] = N0; dim[1] = N1;
  CHKERRQ(MatCreateFFT(PETSC_COMM_WORLD,DIM,dim,MATFFTW,&A));
  CHKERRQ(MatCreateVecsFFTW(A,&x,&y,&z));

  /* Scatter PETSc vector 'input' to FFTW vector 'x' */
  CHKERRQ(VecScatterPetscToFFTW(A,input,x));

  /* Apply forward FFT */
  CHKERRQ(MatMult(A,x,y));

  /* Apply backward FFT */
  CHKERRQ(MatMultTranspose(A,y,z));

  /* Scatter FFTW vector 'z' to PETSc vector 'output' */
  CHKERRQ(VecScatterFFTWToPetsc(A,z,output));

  /* Check accuracy */
  fac  = 1.0/(PetscReal)N;
  CHKERRQ(VecScale(output,fac));
  CHKERRQ(VecAXPY(output,-1.0,input));
  CHKERRQ(VecNorm(output,NORM_1,&enorm));
  if (enorm > 1.e-11 && rank == 0) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"  Error norm of |x - z| %e\n",enorm));
  }

  /* Free spaces */
  CHKERRQ(PetscRandomDestroy(&rdm));
  CHKERRQ(VecDestroy(&input));
  CHKERRQ(VecDestroy(&output));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&y));
  CHKERRQ(VecDestroy(&z));
  CHKERRQ(MatDestroy(&A));

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
      requires: fftw !complex

   test:
      output_file: output/ex148.out

   test:
      suffix: 2
      nsize: 3
      output_file: output/ex148.out

TEST*/
