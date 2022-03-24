static char help[]="This program illustrates the use of PETSc-fftw interface for real DFT\n";
#include <petscmat.h>
#include <fftw3-mpi.h>

extern PetscErrorCode InputTransformFFT(Mat,Vec,Vec);
extern PetscErrorCode OutputTransformFFT(Mat,Vec,Vec);
int main(int argc,char **args)
{
  PetscMPIInt    rank,size;
  PetscInt       N0=3,N1=3,N2=3,N=N0*N1*N2;
  PetscRandom    rdm;
  PetscScalar    a;
  PetscReal      enorm;
  Vec            x,y,z,input,output;
  PetscBool      view=PETSC_FALSE,use_interface=PETSC_TRUE;
  Mat            A;
  PetscInt       DIM, dim[3],vsize;
  PetscReal      fac;

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
#if defined(PETSC_USE_COMPLEX)
  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP, "This example requires real numbers");
#endif

  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  CHKERRQ(PetscRandomCreate(PETSC_COMM_WORLD, &rdm));
  CHKERRQ(PetscRandomSetFromOptions(rdm));

  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&input));
  CHKERRQ(VecSetSizes(input,PETSC_DECIDE,N));
  CHKERRQ(VecSetFromOptions(input));
  CHKERRQ(VecSetRandom(input,rdm));
  CHKERRQ(VecDuplicate(input,&output));
/*  CHKERRQ(VecGetSize(input,&vsize)); */
/*  printf("Size of the input Vector is %d\n",vsize); */

  DIM    = 3;
  dim[0] = N0; dim[1] = N1; dim[2] = N2;

  CHKERRQ(MatCreateFFT(PETSC_COMM_WORLD,DIM,dim,MATFFTW,&A));
  CHKERRQ(MatCreateVecs(A,&x,&y));
  CHKERRQ(MatCreateVecs(A,&z,NULL));
  CHKERRQ(VecGetSize(y,&vsize));
  printf("The vector size from the main routine is %d\n",vsize);

  CHKERRQ(InputTransformFFT(A,input,x));
  CHKERRQ(MatMult(A,x,y));
  CHKERRQ(MatMultTranspose(A,y,z));
  CHKERRQ(OutputTransformFFT(A,z,output));

  fac  = 1.0/(PetscReal)N;
  CHKERRQ(VecScale(output,fac));

  CHKERRQ(VecAssemblyBegin(input));
  CHKERRQ(VecAssemblyEnd(input));
  CHKERRQ(VecAssemblyBegin(output));
  CHKERRQ(VecAssemblyEnd(output));

  CHKERRQ(VecView(input,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(VecView(output,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(VecAXPY(output,-1.0,input));
  CHKERRQ(VecNorm(output,NORM_1,&enorm));
/*  if (enorm > 1.e-14) { */
  if (rank == 0) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"  Error norm of |x - z| %e\n",enorm));
  }
/*      } */

/* CHKERRQ(MatCreateVecs(A,&z,NULL)); */
/*  printf("Vector size from ex148 %d\n",vsize); */
/*  CHKERRQ(PetscObjectSetName((PetscObject) x, "Real space vector")); */
/*      CHKERRQ(PetscObjectSetName((PetscObject) y, "Frequency space vector")); */
/*      CHKERRQ(PetscObjectSetName((PetscObject) z, "Reconstructed vector")); */

  CHKERRQ(PetscFinalize());
  return 0;

}
