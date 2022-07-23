static char help[]="This program illustrates the use of PETSc-fftw interface for real 2D DFT.\n\
                    See ~petsc/src/mat/tests/ex158.c for general cases. \n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  PetscMPIInt    rank,size;
  PetscInt       N0=50,N1=50,N=N0*N1;
  PetscRandom    rdm;
  PetscReal      enorm;
  Vec            x,y,z,input,output;
  Mat            A;
  PetscInt       DIM,dim[2];
  PetscReal      fac;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
#if defined(PETSC_USE_COMPLEX)
  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP, "This example requires real numbers");
#endif

  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  /* Create and set PETSc vectors 'input' and 'output' */
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rdm));
  PetscCall(PetscRandomSetFromOptions(rdm));

  PetscCall(VecCreate(PETSC_COMM_WORLD,&input));
  PetscCall(VecSetSizes(input,PETSC_DECIDE,N0*N1));
  PetscCall(VecSetFromOptions(input));
  PetscCall(VecSetRandom(input,rdm));
  PetscCall(VecDuplicate(input,&output));
  PetscCall(PetscObjectSetName((PetscObject)input, "Real space vector"));
  PetscCall(PetscObjectSetName((PetscObject)output, "Reconstructed vector"));

  /* Get FFTW vectors 'x', 'y' and 'z' */
  DIM    = 2;
  dim[0] = N0; dim[1] = N1;
  PetscCall(MatCreateFFT(PETSC_COMM_WORLD,DIM,dim,MATFFTW,&A));
  PetscCall(MatCreateVecsFFTW(A,&x,&y,&z));

  /* Scatter PETSc vector 'input' to FFTW vector 'x' */
  PetscCall(VecScatterPetscToFFTW(A,input,x));

  /* Apply forward FFT */
  PetscCall(MatMult(A,x,y));

  /* Apply backward FFT */
  PetscCall(MatMultTranspose(A,y,z));

  /* Scatter FFTW vector 'z' to PETSc vector 'output' */
  PetscCall(VecScatterFFTWToPetsc(A,z,output));

  /* Check accuracy */
  fac  = 1.0/(PetscReal)N;
  PetscCall(VecScale(output,fac));
  PetscCall(VecAXPY(output,-1.0,input));
  PetscCall(VecNorm(output,NORM_1,&enorm));
  if (enorm > 1.e-11 && rank == 0) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"  Error norm of |x - z| %e\n",enorm));
  }

  /* Free spaces */
  PetscCall(PetscRandomDestroy(&rdm));
  PetscCall(VecDestroy(&input));
  PetscCall(VecDestroy(&output));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&z));
  PetscCall(MatDestroy(&A));

  PetscCall(PetscFinalize());
  return 0;
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
