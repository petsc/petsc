static char help[] = "Test FFTW interface \n\n";

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
  PetscInt       n = 10,dim[1];
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

  dim[0] = n;
  ierr = MatCreateSeqFFTW(PETSC_COMM_SELF,1,dim,&A);CHKERRQ(ierr);

  /* create vectors */
  ierr = VecCreateSeq(PETSC_COMM_SELF,n,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&y);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&z);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_SELF,&rdm);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rdm);CHKERRQ(ierr);
  ierr = VecSetRandom(x,rdm);CHKERRQ(ierr);
  /*
  printf("input vector x:\n");
  ierr = VecView(x,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  */

  /* apply FFTW_FORWARD */
  ierr = MatMult(A,x,y);CHKERRQ(ierr);
  /*
  printf("output vector y:\n");
  ierr = VecView(y,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  */

  /* apply FFTW_BACKWARD */
  ierr = MatMultTranspose(A,y,z);CHKERRQ(ierr);
  /*
  printf("output vector z:\n");
  ierr = VecView(z,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  */
 
  /* compare x and z. FFTW computes an unnormalized DFT, thus z = n*x */
  s = 1.0/(PetscReal)n;
  ierr = VecScale(z,s);CHKERRQ(ierr);
  ierr = VecAXPY(z,-1.0,x);CHKERRQ(ierr);
  ierr = VecNorm(z,NORM_1,&enorm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"  Error norm of |x - z| = %g\n",enorm);CHKERRQ(ierr);
  
  /* free spaces */
  ierr = PetscRandomDestroy(rdm);CHKERRQ(ierr);
  ierr = VecDestroy(x);CHKERRQ(ierr);
  ierr = VecDestroy(y);CHKERRQ(ierr);
  ierr = VecDestroy(z);CHKERRQ(ierr);
  ierr = MatDestroy(A);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
