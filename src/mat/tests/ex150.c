static char help[]="This program illustrates the use of PETSc-fftw interface for real DFT\n";
#include <petscmat.h>
#include <fftw3-mpi.h>

extern PetscErrorCode InputTransformFFT(Mat,Vec,Vec);
extern PetscErrorCode OutputTransformFFT(Mat,Vec,Vec);
int main(int argc,char **args)
{
  PetscMPIInt    rank,size;
  PetscInt       N0=3,N1=3,N2=3,N3=3,N4=3,N=N0*N1*N2*N3;
  PetscRandom    rdm;
  PetscReal      enorm;
  Vec            x,y,z,input,output;
  Mat            A;
  PetscInt       DIM, dim[5],vsize;
  PetscReal      fac;
  PetscScalar    one=1;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
#if defined(PETSC_USE_COMPLEX)
  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP, "This example requires real numbers");
#endif

  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rdm));
  PetscCall(PetscRandomSetFromOptions(rdm));
  PetscCall(VecCreate(PETSC_COMM_WORLD,&input));
  PetscCall(VecSetSizes(input,PETSC_DECIDE,N));
  PetscCall(VecSetFromOptions(input));
  PetscCall(VecSetRandom(input,rdm));
  PetscCall(VecAssemblyBegin(input));
  PetscCall(VecAssemblyEnd(input));
/*  PetscCall(VecView(input,PETSC_VIEWER_STDOUT_WORLD)); */
  PetscCall(VecDuplicate(input,&output));

  DIM    = 4;
  dim[0] = N0; dim[1] = N1; dim[2] = N2; dim[3] = N3; dim[4] = N4;

  PetscCall(MatCreateFFT(PETSC_COMM_WORLD,DIM,dim,MATFFTW,&A));
  PetscCall(MatCreateVecs(A,&x,&y));
  PetscCall(MatCreateVecs(A,&z,NULL));
  PetscCall(VecGetSize(x,&vsize));
  printf("The vector size from the main routine is %d\n",vsize);

  PetscCall(InputTransformFFT(A,input,x));

  PetscCall(MatMult(A,x,y));
  PetscCall(VecAssemblyBegin(y));
  PetscCall(VecAssemblyEnd(y));
  PetscCall(VecView(y,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(MatMultTranspose(A,y,z));
  PetscCall(VecGetSize(z,&vsize));
  printf("The vector size of zfrom the main routine is %d\n",vsize);

  PetscCall(OutputTransformFFT(A,z,output));

  fac  = 1.0/(PetscReal)N;
  PetscCall(VecScale(output,fac));

  PetscCall(VecAssemblyBegin(input));
  PetscCall(VecAssemblyEnd(input));
  PetscCall(VecAssemblyBegin(output));
  PetscCall(VecAssemblyEnd(output));

  PetscCall(VecView(input,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecView(output,PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(VecAXPY(output,-1.0,input));
  PetscCall(VecNorm(output,NORM_1,&enorm));
/*  if (enorm > 1.e-14) { */
  if (rank == 0) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"  Error norm of |x - z| %e\n",enorm));
  }
/*      } */

/* PetscCall(MatCreateVecs(A,&z,NULL)); */
/*  printf("Vector size from ex148 %d\n",vsize); */
/*  PetscCall(PetscObjectSetName((PetscObject) x, "Real space vector")); */
/*      PetscCall(PetscObjectSetName((PetscObject) y, "Frequency space vector")); */
/*      PetscCall(PetscObjectSetName((PetscObject) z, "Reconstructed vector")); */

  PetscCall(VecDestroy(&output));
  PetscCall(VecDestroy(&input));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&z));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscRandomDestroy(&rdm));
  PetscCall(PetscFinalize());
  return 0;
}
