static char help[]="This program illustrates the use of PETSc-fftw interface for real DFT\n";
#include <petscmat.h>
#include <fftw3-mpi.h>

#undef __FUNCT__
#define __FUNCT__ "main"
extern PetscErrorCode InputTransformFFT(Mat,Vec,Vec);
extern PetscErrorCode OutputTransformFFT(Mat,Vec,Vec);
PetscInt main(PetscInt argc,char **args)
{
  PetscErrorCode  ierr;
  PetscMPIInt     rank,size;
  PetscInt        N0=3,N1=3,N=N0*N1;
  PetscRandom     rdm;
  PetscScalar     a;
  PetscReal       enorm;
  Vec             x,y,z,input,output;
  PetscBool       view=PETSC_FALSE,use_interface=PETSC_TRUE;
  Mat             A;
  PetscInt        DIM, dim[2],vsize;
  PetscReal       fac;

  ierr = PetscInitialize(&argc,&args,(char *)0,help);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
 SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP, "This example requires real numbers");
#endif

  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRQ(ierr);


  ierr = PetscRandomCreate(PETSC_COMM_WORLD, &rdm);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rdm);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&input);CHKERRQ(ierr);
  ierr = VecSetSizes(input,PETSC_DECIDE,N0*N1);CHKERRQ(ierr);
  ierr = VecSetFromOptions(input);CHKERRQ(ierr);
  ierr = VecSetRandom(input,rdm);CHKERRQ(ierr);
  ierr = VecDuplicate(input,&output); 
//  ierr = VecGetSize(input,&vsize);CHKERRQ(ierr);
//  printf("Size of the input Vector is %d\n",vsize);
  
  DIM = 2;
  dim[0] = N0; dim[1] = N1;

  ierr = MatCreateFFT(PETSC_COMM_WORLD,DIM,dim,MATFFTW,&A);CHKERRQ(ierr);
  ierr = MatGetVecs(A,&x,&y);CHKERRQ(ierr);
  ierr = MatGetVecs(A,&z,PETSC_NULL);CHKERRQ(ierr);
  ierr = VecGetSize(y,&vsize);CHKERRQ(ierr);

  ierr = InputTransformFFT(A,input,x);CHKERRQ(ierr);
  ierr = MatMult(A,x,y);CHKERRQ(ierr);
  ierr = MatMultTranspose(A,y,z);CHKERRQ(ierr);
  ierr = OutputTransformFFT(A,z,output);CHKERRQ(ierr);

  fac = 1.0/(PetscReal)N;
  ierr = VecScale(output,fac);CHKERRQ(ierr);

  VecAssemblyBegin(input);
  VecAssemblyEnd(input);
  VecAssemblyBegin(output);
  VecAssemblyEnd(output);

//  VecView(input,PETSC_VIEWER_STDOUT_WORLD);
//  VecView(output,PETSC_VIEWER_STDOUT_WORLD);

  
 
// ierr = MatGetVecs(A,&z,PETSC_NULL);CHKERRQ(ierr);
//  printf("Vector size from ex148 %d\n",vsize);
//  ierr = PetscObjectSetName((PetscObject) x, "Real space vector");CHKERRQ(ierr);
//      ierr = PetscObjectSetName((PetscObject) y, "Frequency space vector");CHKERRQ(ierr);
//      ierr = PetscObjectSetName((PetscObject) z, "Reconstructed vector");CHKERRQ(ierr);

  ierr = PetscFinalize();
  return 0;
  
}



