
static char help[] = "Tests converting a parallel AIJ formatted matrix to the parallel Row format.\n\
 This also tests MatGetRow() and MatRestoreRow() for the parallel case.\n\n";

#include "petscmat.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat               C,A;
  PetscInt          i,j,m = 3,n = 2,Ii,J,rstart,rend,nz;
  PetscMPIInt       rank,size;
  PetscErrorCode    ierr;
  const PetscInt    *idx;
  PetscScalar       v;
  const PetscScalar *values;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  n = 2*size;

  /* create the matrix for the five point stencil, YET AGAIN*/
  ierr = MatCreate(PETSC_COMM_WORLD,&C);CHKERRQ(ierr);
  ierr = MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(C);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(C,5,PETSC_NULL,5,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(C,5,PETSC_NULL);CHKERRQ(ierr);
  for (i=0; i<m; i++) { 
    for (j=2*rank; j<2*rank+2; j++) {
      v = -1.0;  Ii = j + n*i;
      if (i>0)   {J = Ii - n; ierr = MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (i<m-1) {J = Ii + n; ierr = MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (j>0)   {J = Ii - 1; ierr = MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (j<n-1) {J = Ii + 1; ierr = MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      v = 4.0; ierr = MatSetValues(C,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);
  ierr = MatView(C,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MatView(C,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(C,&rstart,&rend);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) {
    ierr = MatGetRow(C,i,&nz,&idx,&values);CHKERRQ(ierr);
    ierr = PetscSynchronizedFPrintf(PETSC_COMM_WORLD,stdout,"[%d] get row %D: ",rank,i);CHKERRQ(ierr);
    for (j=0; j<nz; j++) {
#if defined(PETSC_USE_COMPLEX)
      ierr = PetscSynchronizedFPrintf(PETSC_COMM_WORLD,stdout,"%D %G  ",idx[j],PetscRealPart(values[j]));CHKERRQ(ierr);
#else
      ierr = PetscSynchronizedFPrintf(PETSC_COMM_WORLD,stdout,"%D %G  ",idx[j],values[j]);CHKERRQ(ierr);
#endif
    }
    ierr = PetscSynchronizedFPrintf(PETSC_COMM_WORLD,stdout,"\n");CHKERRQ(ierr);
    ierr = MatRestoreRow(C,i,&nz,&idx,&values);CHKERRQ(ierr);
  }
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD);CHKERRQ(ierr);CHKERRQ(ierr);

  ierr = MatConvert(C,MATSAME,MAT_INITIAL_MATRIX,&A);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);
  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); 
  ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); 

  ierr = MatDestroy(A);CHKERRQ(ierr);
  ierr = MatDestroy(C);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
