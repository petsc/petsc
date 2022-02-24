static char help[]="This program illustrates the use of PETSc-fftw interface for sequential real DFT\n";
#include <petscmat.h>
#include <fftw3-mpi.h>
int main(int argc,char **args)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank,size;
  PetscInt       N0=10,N1=10,N2=10,N3=10,N4=10,N=N0*N1*N2*N3*N4;
  PetscRandom    rdm;
  PetscReal      enorm;
  Vec            x,y,z,input,output;
  Mat            A;
  PetscInt       DIM, dim[5],vsize;
  PetscReal      fac;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
#if defined(PETSC_USE_COMPLEX)
  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP, "This example requires real numbers");
#endif
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  PetscCheckFalse(size!=1,PETSC_COMM_WORLD,PETSC_ERR_SUP, "This is a uni-processor example only");
  CHKERRQ(PetscRandomCreate(PETSC_COMM_SELF, &rdm));
  CHKERRQ(PetscRandomSetFromOptions(rdm));
  CHKERRQ(VecCreate(PETSC_COMM_SELF,&input));
  CHKERRQ(VecSetSizes(input,N,N));
  CHKERRQ(VecSetFromOptions(input));
  CHKERRQ(VecSetRandom(input,rdm));
  CHKERRQ(VecDuplicate(input,&output));

  DIM  = 5; dim[0] = N0; dim[1] = N1; dim[2] = N2; dim[3] = N3; dim[4] = N4;
  CHKERRQ(MatCreateFFT(PETSC_COMM_SELF,DIM,dim,MATFFTW,&A));
  CHKERRQ(MatCreateVecs(A,&x,&y));
  CHKERRQ(MatCreateVecs(A,&z,NULL));

  CHKERRQ(VecGetSize(x,&vsize));
  printf("The vector size  of input from the main routine is %d\n",vsize);

  CHKERRQ(VecGetSize(z,&vsize));
  printf("The vector size of output from the main routine is %d\n",vsize);

  CHKERRQ(InputTransformFFT(A,input,x));

  CHKERRQ(MatMult(A,x,y));
  CHKERRQ(MatMultTranspose(A,y,z));

  CHKERRQ(OutputTransformFFT(A,z,output));
  fac  = 1.0/(PetscReal)N;
  CHKERRQ(VecScale(output,fac));
/*
  CHKERRQ(VecAssemblyBegin(input));
  CHKERRQ(VecAssemblyEnd(input));
  CHKERRQ(VecAssemblyBegin(output));
  CHKERRQ(VecAssemblyEnd(output));

  CHKERRQ(VecView(input,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(VecView(output,PETSC_VIEWER_STDOUT_WORLD));
*/
  CHKERRQ(VecAXPY(output,-1.0,input));
  CHKERRQ(VecNorm(output,NORM_1,&enorm));
/*  if (enorm > 1.e-14) { */
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"  Error norm of |x - z| %e\n",enorm));
/*      } */

  CHKERRQ(VecDestroy(&output));
  CHKERRQ(VecDestroy(&input));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&y));
  CHKERRQ(VecDestroy(&z));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(PetscRandomDestroy(&rdm));
  ierr = PetscFinalize();
  return ierr;

}
