/*$Id: ex55.c,v 1.12 2000/01/11 21:01:03 bsmith Exp balay $*/

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
  Viewer  fd;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = OptionsGetString(PETSC_NULL,"-f",file,127,PETSC_NULL);CHKERRA(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRA(ierr);
  if (size > 1) ntypes = 5;

  /* 
     Open binary file.  Note that we use BINARY_RDONLY to indicate
     reading from this file.
  */
  ierr = ViewerBinaryOpen(PETSC_COMM_WORLD,file,BINARY_RDONLY,&fd);CHKERRA(ierr);

  /*
     Load the matrix and vector; then destroy the viewer.
  */
  ierr = MatLoad(fd,MATMPIAIJ,&C);CHKERRA(ierr);
  ierr = VecLoad(fd,&v);CHKERRA(ierr);
  ierr = ViewerDestroy(fd);CHKERRA(ierr);

  
  for (i=0; i<ntypes; i++) {
    ierr = MatConvert(C,type[i],&A);CHKERRA(ierr);
    for (j=0; j<ntypes; j++) {
      ierr = MatConvert(A,type[i],&B);CHKERRA(ierr);
      ierr = MatDestroy(B);CHKERRA(ierr);
    }
    ierr = MatDestroy(A);CHKERRA(ierr);
  }

  if (size == 1) {
    ierr = ViewerBinaryOpen(PETSC_COMM_WORLD,"testmat",BINARY_CREATE,&fd);CHKERRA(ierr);
    ierr = MatView(C,fd);CHKERRA(ierr);
    ierr = VecView(v,fd);CHKERRA(ierr);
    ierr = ViewerDestroy(fd);CHKERRA(ierr);
  }

  ierr = MatDestroy(C);CHKERRA(ierr);
  ierr = VecDestroy(v);CHKERRA(ierr);

  PetscFinalize();
  return 0;
}











