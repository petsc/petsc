/*$Id: ex55.c,v 1.18 2001/04/10 19:35:44 bsmith Exp $*/

static char help[] = "Tests converting a matrix to another format with MatConvert().\n\n";

#include "petscmat.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat         C,A,B,D; 
  int         ierr,i,j,ntypes = 2,size;
  MatType     type[9] = {MATSEQAIJ,MATSEQBAIJ,MATMPIROWBS};
  char        file[128];
  Vec         v;
  PetscViewer fd;
  PetscTruth  equal;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = PetscOptionsGetString(PETSC_NULL,"-f",file,127,PETSC_NULL);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size > 1) ntypes = 5;

  /* 
     Open binary file.  Note that we use PETSC_FILE_RDONLY to indicate
     reading from this file.
  */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,PETSC_FILE_RDONLY,&fd);CHKERRQ(ierr);

  /*
     Load the matrix and vector; then destroy the viewer.
  */
  ierr = MatLoad(fd,MATMPIAIJ,&C);CHKERRQ(ierr);
  ierr = VecLoad(fd,&v);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(fd);CHKERRQ(ierr);

  
  for (i=0; i<ntypes; i++) {
    ierr = MatConvert(C,type[i],&A);CHKERRQ(ierr);
    for (j=0; j<ntypes; j++) {
      ierr = MatConvert(A,type[j],&B);CHKERRQ(ierr);
      ierr = MatConvert(B,type[i],&D);CHKERRQ(ierr);
      ierr = MatEqual(A,D,&equal);CHKERRQ(ierr);
      if (!equal){
        MatView(A,PETSC_VIEWER_STDOUT_WORLD);
        MatView(B,PETSC_VIEWER_STDOUT_WORLD);
        MatView(D,PETSC_VIEWER_STDOUT_WORLD);
        SETERRQ2(1,"Error in conversion from %s to %s",type[i],type[j]);
      }
      ierr = MatDestroy(B);CHKERRQ(ierr);
      ierr = MatDestroy(D);CHKERRQ(ierr);
    }
    ierr = MatDestroy(A);CHKERRQ(ierr);
  }


  ierr = MatDestroy(C);CHKERRQ(ierr);
  ierr = VecDestroy(v);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}











