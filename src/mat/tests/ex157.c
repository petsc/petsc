static char help[]="This program illustrates the use of PETSc-fftw interface for parallel real DFT\n";
#include <petscmat.h>
#include <fftw3-mpi.h>
int main(int argc,char **args)
{
  PetscMPIInt    rank,size;
  PetscInt       N0=2048,N1=2048,N2=3,N3=5,N4=5,N=N0*N1;
  PetscRandom    rdm;
  PetscReal      enorm;
  Vec            x,y,z,input,output;
  Mat            A;
  PetscInt       DIM, dim[5],vsize;
  PetscReal      fac;
  PetscScalar    one=1,two=2,three=3;

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
/*  PetscCall(VecSet(input,one)); */
/*  PetscCall(VecSetValue(input,1,two,INSERT_VALUES)); */
/*  PetscCall(VecSetValue(input,2,three,INSERT_VALUES)); */
/*  PetscCall(VecSetValue(input,3,three,INSERT_VALUES)); */
  PetscCall(VecSetRandom(input,rdm));
/*  PetscCall(VecSetRandom(input,rdm)); */
/*  PetscCall(VecSetRandom(input,rdm)); */
  PetscCall(VecDuplicate(input,&output));

  DIM  = 2; dim[0] = N0; dim[1] = N1; dim[2] = N2; dim[3] = N3; dim[4] = N4;
  PetscCall(MatCreateFFT(PETSC_COMM_WORLD,DIM,dim,MATFFTW,&A));
  PetscCall(MatCreateVecsFFTW(A,&x,&y,&z));
/*  PetscCall(MatCreateVecs(A,&x,&y)); */
/*  PetscCall(MatCreateVecs(A,&z,NULL)); */

  PetscCall(VecGetSize(x,&vsize));
  printf("The vector size  of input from the main routine is %d\n",vsize);

  PetscCall(VecGetSize(z,&vsize));
  printf("The vector size of output from the main routine is %d\n",vsize);

  PetscCall(InputTransformFFT(A,input,x));

  PetscCall(MatMult(A,x,y));
  PetscCall(VecAssemblyBegin(y));
  PetscCall(VecAssemblyEnd(y));
  PetscCall(VecView(y,PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(MatMultTranspose(A,y,z));

  PetscCall(OutputTransformFFT(A,z,output));
  fac  = 1.0/(PetscReal)N;
  PetscCall(VecScale(output,fac));

  PetscCall(VecAssemblyBegin(input));
  PetscCall(VecAssemblyEnd(input));
  PetscCall(VecAssemblyBegin(output));
  PetscCall(VecAssemblyEnd(output));

/*  PetscCall(VecView(input,PETSC_VIEWER_STDOUT_WORLD)); */
/*  PetscCall(VecView(output,PETSC_VIEWER_STDOUT_WORLD)); */

  PetscCall(VecAXPY(output,-1.0,input));
  PetscCall(VecNorm(output,NORM_1,&enorm));
/*  if (enorm > 1.e-14) { */
  PetscCall(PetscPrintf(PETSC_COMM_SELF,"  Error norm of |x - z| %e\n",enorm));
/*      } */

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
