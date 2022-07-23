static char help[]="This program illustrates the use of PETSc-fftw interface for parallel real DFT\n";
#include <petscmat.h>
#include <fftw3-mpi.h>
/*extern PetscErrorCode MatCreateVecsFFT(Mat,Vec *,Vec *,Vec *);*/
int main(int argc,char **args)
{
  PetscMPIInt    rank,size;
  PetscInt       N0=4096,N1=4096,N2=256,N3=10,N4=10,N=N0*N1;
  PetscRandom    rdm;
  PetscReal      enorm;
  Vec            x,y,z,input,output;
  Mat            A;
  PetscInt       DIM, dim[5],vsize,row,col;
  PetscReal      fac;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

#if defined(PETSC_USE_COMPLEX)
  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP, "Example for Real DFT. Your current data type is complex!");
#endif
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rdm));
  PetscCall(PetscRandomSetFromOptions(rdm));
  PetscCall(VecCreate(PETSC_COMM_WORLD,&input));
  PetscCall(VecSetSizes(input,PETSC_DECIDE,N));
  PetscCall(VecSetFromOptions(input));
  PetscCall(VecSetRandom(input,rdm));
  PetscCall(VecDuplicate(input,&output));

  DIM  = 2; dim[0] = N0; dim[1] = N1; dim[2] = N2; dim[3] = N3; dim[4] = N4;
  PetscCall(MatCreateFFT(PETSC_COMM_WORLD,DIM,dim,MATFFTW,&A));
  PetscCall(MatGetLocalSize(A,&row,&col));
  printf("The Matrix size  is %d and %d from process %d\n",row,col,rank);
  PetscCall(MatCreateVecsFFTW(A,&x,&y,&z));

  PetscCall(VecGetSize(x,&vsize));

  PetscCall(VecGetSize(z,&vsize));
  printf("The vector size of output from the main routine is %d\n",vsize);

  PetscCall(VecScatterPetscToFFTW(A,input,x));
  /*PetscCall(VecDestroy(&input));*/
  PetscCall(MatMult(A,x,y));
  PetscCall(MatMultTranspose(A,y,z));
  PetscCall(VecScatterFFTWToPetsc(A,z,output));
  /*PetscCall(VecDestroy(&z));*/
  fac  = 1.0/(PetscReal)N;
  PetscCall(VecScale(output,fac));

  PetscCall(VecAssemblyBegin(input));
  PetscCall(VecAssemblyEnd(input));
  PetscCall(VecAssemblyBegin(output));
  PetscCall(VecAssemblyEnd(output));

/*  PetscCall(VecView(input,PETSC_VIEWER_STDOUT_WORLD));*/
/*  PetscCall(VecView(output,PETSC_VIEWER_STDOUT_WORLD));*/

  PetscCall(VecAXPY(output,-1.0,input));
  PetscCall(VecNorm(output,NORM_1,&enorm));
  if (enorm > 1.e-14) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"  Error norm of |x - z| %e\n",enorm));
  }

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&z));
  PetscCall(VecDestroy(&output));
  PetscCall(VecDestroy(&input));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscRandomDestroy(&rdm));
  PetscCall(PetscFinalize());
  return 0;

}
