/*$Id: ex55.c,v 1.18 2001/04/10 19:35:44 bsmith Exp $*/

static char help[] = "Tests converting a matrix to another format with MatConvert().\n\n";

#include "petscmat.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat         C,A,B; 
  int         ierr,i,j,ntypes = 9,size;
  MatType     type[9] = {MATMPIAIJ, MATMPIROWBS, MATMPIBDIAG,MATMPIDENSE,
                         MATMPIBAIJ,MATSEQDENSE,MATSEQAIJ,  MATSEQBDIAG,MATSEQBAIJ};
  char        file[128];
  Vec         v;
  PetscViewer fd;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = PetscOptionsGetString(PETSC_NULL,"-f",file,127,PETSC_NULL);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size > 1) ntypes = 5;

  /* 
     Open binary file.  Note that we use PETSC_BINARY_RDONLY to indicate
     reading from this file.
  */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,PETSC_BINARY_RDONLY,&fd);CHKERRQ(ierr);

  /*
     Load the matrix and vector; then destroy the viewer.
  */
  ierr = MatLoad(fd,MATMPIAIJ,&C);CHKERRQ(ierr);
  ierr = VecLoad(fd,&v);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(fd);CHKERRQ(ierr);

  
  for (i=0; i<ntypes; i++) {
    ierr = MatConvert(C,type[i],&A);CHKERRQ(ierr);
    for (j=0; j<ntypes; j++) {
      ierr = MatConvert(A,type[i],&B);CHKERRQ(ierr);
      ierr = MatDestroy(B);CHKERRQ(ierr);
    }
    ierr = MatDestroy(A);CHKERRQ(ierr);
  }

  if (size == 1) {
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"testmat",PETSC_BINARY_CREATE,&fd);CHKERRQ(ierr);
    ierr = MatView(C,fd);CHKERRQ(ierr);
    ierr = VecView(v,fd);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(fd);CHKERRQ(ierr);
  }

  ierr = MatDestroy(C);CHKERRQ(ierr);
  ierr = VecDestroy(v);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}











