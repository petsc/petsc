static char help[]="This program illustrates the use of PETSc-fftw interface\n";
#include <petscmat.h>
#include <fftw3-mpi.h>

#undef __FUNCT__
#define __FUNCT__ "main"
PetscInt main(PetscInt argc,char **args)
{
  PetscErrorCode  ierr;
  PetscMPIInt     rank,size;
  PetscInt        N0=50,N1=20,N=N0*N1;
  PetscRandom     rdm;
  PetscScalar     a;
  PetscReal       enorm;
  Vec             x,y,z;
  PetscBool       view=PETSC_FALSE,use_interface=PETSC_TRUE;
  Mat             A;
  PetscInt        DIM, dim[2],vsize;

  ierr = PetscInitialize(&argc,&args,(char *)0,help);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
 SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP, "This example requires real numbers");
#endif

  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRQ(ierr);


  ierr = PetscRandomCreate(PETSC_COMM_WORLD, &rdm);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rdm);CHKERRQ(ierr);
  
  DIM = 2;
  dim[0] = N0; dim[1] = N1;

 ierr = MatCreateFFT(PETSC_COMM_WORLD,DIM,dim,MATFFTW,&A);CHKERRQ(ierr);
 ierr = MatGetVecs(A,&x,&y);CHKERRQ(ierr);
 ierr = VecGetSize(x,&vsize);CHKERRQ(ierr);
 
// ierr = MatGetVecs(A,&z,PETSC_NULL);CHKERRQ(ierr);
 printf("Vector size from ex148 %d\n",vsize);
 ierr = PetscFinalize();
  return 0;
  
}



