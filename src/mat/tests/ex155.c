static char help[]="This program illustrates the use of PETSc-fftw interface for parallel real DFT\n";
#include <petscmat.h>
#include <fftw3-mpi.h>
/*extern PetscErrorCode MatCreateVecsFFT(Mat,Vec *,Vec *,Vec *);*/
int main(int argc,char **args)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank,size;
  PetscInt       N0=4096,N1=4096,N2=256,N3=10,N4=10,N=N0*N1;
  PetscRandom    rdm;
  PetscReal      enorm;
  Vec            x,y,z,input,output;
  Mat            A;
  PetscInt       DIM, dim[5],vsize,row,col;
  PetscReal      fac;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

#if defined(PETSC_USE_COMPLEX)
  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP, "Example for Real DFT. Your current data type is complex!");
#endif
  CHKERRQ(PetscRandomCreate(PETSC_COMM_WORLD, &rdm));
  CHKERRQ(PetscRandomSetFromOptions(rdm));
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&input));
  CHKERRQ(VecSetSizes(input,PETSC_DECIDE,N));
  CHKERRQ(VecSetFromOptions(input));
  CHKERRQ(VecSetRandom(input,rdm));
  CHKERRQ(VecDuplicate(input,&output));

  DIM  = 2; dim[0] = N0; dim[1] = N1; dim[2] = N2; dim[3] = N3; dim[4] = N4;
  CHKERRQ(MatCreateFFT(PETSC_COMM_WORLD,DIM,dim,MATFFTW,&A));
  CHKERRQ(MatGetLocalSize(A,&row,&col));
  printf("The Matrix size  is %d and %d from process %d\n",row,col,rank);
  CHKERRQ(MatCreateVecsFFTW(A,&x,&y,&z));

  CHKERRQ(VecGetSize(x,&vsize));

  CHKERRQ(VecGetSize(z,&vsize));
  printf("The vector size of output from the main routine is %d\n",vsize);

  CHKERRQ(VecScatterPetscToFFTW(A,input,x));
  /*CHKERRQ(VecDestroy(&input));*/
  CHKERRQ(MatMult(A,x,y));
  CHKERRQ(MatMultTranspose(A,y,z));
  CHKERRQ(VecScatterFFTWToPetsc(A,z,output));
  /*CHKERRQ(VecDestroy(&z));*/
  fac  = 1.0/(PetscReal)N;
  CHKERRQ(VecScale(output,fac));

  CHKERRQ(VecAssemblyBegin(input));
  CHKERRQ(VecAssemblyEnd(input));
  CHKERRQ(VecAssemblyBegin(output));
  CHKERRQ(VecAssemblyEnd(output));

/*  CHKERRQ(VecView(input,PETSC_VIEWER_STDOUT_WORLD));*/
/*  CHKERRQ(VecView(output,PETSC_VIEWER_STDOUT_WORLD));*/

  CHKERRQ(VecAXPY(output,-1.0,input));
  CHKERRQ(VecNorm(output,NORM_1,&enorm));
  if (enorm > 1.e-14) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"  Error norm of |x - z| %e\n",enorm));
  }

  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&y));
  CHKERRQ(VecDestroy(&z));
  CHKERRQ(VecDestroy(&output));
  CHKERRQ(VecDestroy(&input));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(PetscRandomDestroy(&rdm));
  ierr = PetscFinalize();
  return ierr;

}
