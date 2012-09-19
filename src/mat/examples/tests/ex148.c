static char help[]="This program illustrates the use of PETSc-fftw interface for real 2D DFT.\n\
                    See ~petsc/src/mat/examples/tests/ex158.c for general cases. \n\n";
#include <petscmat.h>

#undef __FUNCT__
#define __FUNCT__ "main"
PetscInt main(PetscInt argc,char **args)
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

  ierr = PetscInitialize(&argc,&args,(char *)0,help);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP, "This example requires real numbers");
#endif

  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRQ(ierr);

  /* Create and set PETSc vectors 'input' and 'output' */
  ierr = PetscRandomCreate(PETSC_COMM_WORLD, &rdm);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rdm);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&input);CHKERRQ(ierr);
  ierr = VecSetSizes(input,PETSC_DECIDE,N0*N1);CHKERRQ(ierr);
  ierr = VecSetFromOptions(input);CHKERRQ(ierr);
  ierr = VecSetRandom(input,rdm);CHKERRQ(ierr);
  ierr = VecDuplicate(input,&output);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)input, "Real space vector");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)output, "Reconstructed vector");CHKERRQ(ierr);

  /* Get FFTW vectors 'x', 'y' and 'z' */
  DIM = 2;
  dim[0] = N0; dim[1] = N1;
  ierr = MatCreateFFT(PETSC_COMM_WORLD,DIM,dim,MATFFTW,&A);CHKERRQ(ierr);
  ierr = MatGetVecsFFTW(A,&x,&y,&z);CHKERRQ(ierr);

  /* Scatter PETSc vector 'input' to FFTW vector 'x' */
  ierr = VecScatterPetscToFFTW(A,input,x);CHKERRQ(ierr);

  /* Apply forward FFT */
  ierr = MatMult(A,x,y);CHKERRQ(ierr);

  /* Apply backward FFT */
  ierr = MatMultTranspose(A,y,z);CHKERRQ(ierr);

  /* Scatter FFTW vector 'z' to PETSc vector 'output' */
  ierr = VecScatterFFTWToPetsc(A,z,output);CHKERRQ(ierr);

  /* Check accuracy */
  fac = 1.0/(PetscReal)N;
  ierr = VecScale(output,fac);CHKERRQ(ierr);
  ierr = VecAXPY(output,-1.0,input);CHKERRQ(ierr);
  ierr = VecNorm(output,NORM_1,&enorm);CHKERRQ(ierr);
  if (enorm > 1.e-11 && !rank){
    ierr = PetscPrintf(PETSC_COMM_SELF,"  Error norm of |x - z| %e\n",enorm);CHKERRQ(ierr);
  }

  /* Free spaces */
  ierr = PetscRandomDestroy(&rdm);CHKERRQ(ierr);
  ierr = VecDestroy(&input);CHKERRQ(ierr);
  ierr = VecDestroy(&output);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = VecDestroy(&z);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return 0;
}



