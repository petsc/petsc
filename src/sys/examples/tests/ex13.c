#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex13.c,v 1.4 1999/03/19 21:17:16 bsmith Exp bsmith $";
#endif

/*
     Tests PetscSetCommWorld()
*/
#include "petsc.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc, char **argv) 
{
  int      ierr,rank,size;
  MPI_Comm newcomm;

  MPI_Init(&argc,&argv);
  ierr = MPI_Comm_rank(MPI_COMM_WORLD,&rank);CHKERRA(ierr);
  ierr = MPI_Comm_size(MPI_COMM_WORLD,&size);CHKERRA(ierr);

  /*
       make two new communicators each half the size of original
  */
  ierr = MPI_Comm_split(MPI_COMM_WORLD,2*rank<size,0,&newcomm);CHKERRA(ierr);

  ierr = PetscSetCommWorld(newcomm);
  if (ierr) {
    fprintf(stderr,"Unable to set PETSC_COMM_WORLD\n");
  }

  PetscInitialize(&argc,&argv,PETSC_NULL,PETSC_NULL);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  printf("rank = %3d\n",rank);

  PetscFinalize();

  MPI_Finalize();
  return 0;
}
