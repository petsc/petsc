/*$Id: ex55.c,v 1.13 2000/05/05 22:16:17 balay Exp bsmith $*/

static char help[] = "Tests converting a matrix to another format with MatConvert()\n\n";

#include "petscmat.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Mat     C,A,B; 
  int     ierr,i,j,ntypes = 9,size;
  MatType type[9] = {MATMPIAIJ, MATMPIROWBS, MATMPIBDIAG,MATMPIDENSE,
                     MATMPIBAIJ,MATSEQDENSE,MATSEQAIJ,  MATSEQBDIAG,MATSEQBAIJ};
  char    file[128];
  Vec     v;
  PetscViewer  fd;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = PetscOptionsGetString(PETSC_NULL,"-f",file,127,PETSC_NULL);CHKERRA(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRA(ierr);
  if (size > 1) ntypes = 5;

  /* 
     Open binary file.  Note that we use PETSC_BINARY_RDONLY to indicate
     reading from this file.
  */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,PETSC_BINARY_RDONLY,&fd);CHKERRA(ierr);

  /*
     Load the matrix and vector; then destroy the viewer.
  */
  ierr = MatLoad(fd,MATMPIAIJ,&C);CHKERRA(ierr);
  ierr = VecLoad(fd,&v);CHKERRA(ierr);
  ierr = PetscViewerDestroy(fd);CHKERRA(ierr);

  
  for (i=0; i<ntypes; i++) {
    ierr = MatConvert(C,type[i],&A);CHKERRA(ierr);
    for (j=0; j<ntypes; j++) {
      ierr = MatConvert(A,type[i],&B);CHKERRA(ierr);
      ierr = MatDestroy(B);CHKERRA(ierr);
    }
    ierr = MatDestroy(A);CHKERRA(ierr);
  }

  if (size == 1) {
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"testmat",PETSC_BINARY_CREATE,&fd);CHKERRA(ierr);
    ierr = MatView(C,fd);CHKERRA(ierr);
    ierr = VecView(v,fd);CHKERRA(ierr);
    ierr = PetscViewerDestroy(fd);CHKERRA(ierr);
  }

  ierr = MatDestroy(C);CHKERRA(ierr);
  ierr = VecDestroy(v);CHKERRA(ierr);

  PetscFinalize();
  return 0;
}











