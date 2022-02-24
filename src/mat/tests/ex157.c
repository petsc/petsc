static char help[]="This program illustrates the use of PETSc-fftw interface for parallel real DFT\n";
#include <petscmat.h>
#include <fftw3-mpi.h>
int main(int argc,char **args)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank,size;
  PetscInt       N0=2048,N1=2048,N2=3,N3=5,N4=5,N=N0*N1;
  PetscRandom    rdm;
  PetscReal      enorm;
  Vec            x,y,z,input,output;
  Mat            A;
  PetscInt       DIM, dim[5],vsize;
  PetscReal      fac;
  PetscScalar    one=1,two=2,three=3;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
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
/*  CHKERRQ(VecSet(input,one)); */
/*  CHKERRQ(VecSetValue(input,1,two,INSERT_VALUES)); */
/*  CHKERRQ(VecSetValue(input,2,three,INSERT_VALUES)); */
/*  CHKERRQ(VecSetValue(input,3,three,INSERT_VALUES)); */
  CHKERRQ(VecSetRandom(input,rdm));
/*  CHKERRQ(VecSetRandom(input,rdm)); */
/*  CHKERRQ(VecSetRandom(input,rdm)); */
  CHKERRQ(VecDuplicate(input,&output));

  DIM  = 2; dim[0] = N0; dim[1] = N1; dim[2] = N2; dim[3] = N3; dim[4] = N4;
  CHKERRQ(MatCreateFFT(PETSC_COMM_WORLD,DIM,dim,MATFFTW,&A));
  CHKERRQ(MatCreateVecsFFTW(A,&x,&y,&z));
/*  CHKERRQ(MatCreateVecs(A,&x,&y)); */
/*  CHKERRQ(MatCreateVecs(A,&z,NULL)); */

  CHKERRQ(VecGetSize(x,&vsize));
  printf("The vector size  of input from the main routine is %d\n",vsize);

  CHKERRQ(VecGetSize(z,&vsize));
  printf("The vector size of output from the main routine is %d\n",vsize);

  CHKERRQ(InputTransformFFT(A,input,x));

  CHKERRQ(MatMult(A,x,y));
  CHKERRQ(VecAssemblyBegin(y));
  CHKERRQ(VecAssemblyEnd(y));
  CHKERRQ(VecView(y,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(MatMultTranspose(A,y,z));

  CHKERRQ(OutputTransformFFT(A,z,output));
  fac  = 1.0/(PetscReal)N;
  CHKERRQ(VecScale(output,fac));

  CHKERRQ(VecAssemblyBegin(input));
  CHKERRQ(VecAssemblyEnd(input));
  CHKERRQ(VecAssemblyBegin(output));
  CHKERRQ(VecAssemblyEnd(output));

/*  CHKERRQ(VecView(input,PETSC_VIEWER_STDOUT_WORLD)); */
/*  CHKERRQ(VecView(output,PETSC_VIEWER_STDOUT_WORLD)); */

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
