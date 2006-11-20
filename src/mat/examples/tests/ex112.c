static char help[] = "Test sequential FFTW interface \n\n";

/*
  Compiling the code:
      This code uses the complex numbers version of PETSc, so configure
      must be run to enable this

*/

#include "petscmat.h"
#undef __FUNCT__
#define __FUNCT__ "main"
PetscInt main(PetscInt argc,char **args)
{
  Mat            A;    
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscInt       n = 10,N,ndim=4,dim[4],DIM,i;
  Vec            x,y,z;
  PetscScalar    *x_array,*y_array,*z_array,s;  
  PetscRandom    rdm;
  PetscReal      enorm;

  PetscInitialize(&argc,&args,(char *)0,help);
#if !defined(PETSC_USE_COMPLEX)
  SETERRQ(1,"This example requires complex numbers");
#endif
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(1,"This is a uniprocessor example only!");

  for (DIM=0; DIM<ndim; DIM++){
    dim[DIM] = n;  /* size of transformation in DIM-dimension */
  }
  ierr = PetscRandomCreate(PETSC_COMM_SELF,&rdm);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rdm);CHKERRQ(ierr);

  for (DIM=1; DIM<5; DIM++){
    /* create vectors of length N=n^DIM */
    N = 1; for (i=0; i<DIM; i++) N *= dim[i];   
    PetscPrintf(PETSC_COMM_SELF, "\n %d-D: FFTW on vector of size %d \n",DIM,N);
    ierr = VecCreateSeq(PETSC_COMM_SELF,N,&x);CHKERRQ(ierr);
    ierr = VecDuplicate(x,&y);CHKERRQ(ierr);
    ierr = VecDuplicate(x,&z);CHKERRQ(ierr);
    ierr = VecSetRandom(x,rdm);CHKERRQ(ierr);

    /* create FFTW object */
    ierr = MatCreateSeqFFTW(PETSC_COMM_SELF,DIM,dim,&A);CHKERRQ(ierr);

    /* apply FFTW_FORWARD several times, so the fftw_plan can be reused on different vectors */
    ierr = MatMult(A,x,z);CHKERRQ(ierr);
    for (i=0; i<3; i++){
      ierr = MatMult(A,x,y);CHKERRQ(ierr); 

      /* apply FFTW_BACKWARD several times */  
      ierr = MatMultTranspose(A,y,z);CHKERRQ(ierr);
    }
 
    /* compare x and z. FFTW computes an unnormalized DFT, thus z = N*x */
    s = 1.0/(PetscReal)N;
    ierr = VecScale(z,s);CHKERRQ(ierr);
    ierr = VecAXPY(z,-1.0,x);CHKERRQ(ierr);
    ierr = VecNorm(z,NORM_1,&enorm);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"  Error norm of |x - z| = %A\n",enorm);CHKERRQ(ierr);

    /* free spaces */
    ierr = VecDestroy(x);CHKERRQ(ierr);
    ierr = VecDestroy(y);CHKERRQ(ierr);
    ierr = VecDestroy(z);CHKERRQ(ierr);
    ierr = MatDestroy(A);CHKERRQ(ierr);
  }
  ierr = PetscRandomDestroy(rdm);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
