/*$Id: ex13.c,v 1.7 1999/10/24 14:01:38 bsmith Exp bsmith $*/

/*
     Tests PetscSetCommWorld()
*/
#include "petsc.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv) 
{
  int      ierr,rank,size;
  MPI_Comm newcomm;

  MPI_Init(&argc,&argv);

  /* Note cannot use PETSc error handlers here,since PETSc not yet initialized */
  ierr = MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  if (ierr) {
    printf("Error in getting rank");
    return 1;
  }
  ierr = MPI_Comm_size(MPI_COMM_WORLD,&size);CHKERRA(ierr);
  if (ierr) {
    printf("Error in getting size");
    return 1;
  }

  /*
       make two new communicators each half the size of original
  */
  ierr = MPI_Comm_split(MPI_COMM_WORLD,2*rank<size,0,&newcomm);
  if (ierr) {
    printf("Error in splitting comm");
    return 1;
  }

  ierr = PetscSetCommWorld(newcomm);
  if (ierr) {
    fprintf(stderr,"Unable to set PETSC_COMM_WORLD\n");
  }

  PetscInitialize(&argc,&argv,PETSC_NULL,PETSC_NULL);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRA(ierr);
  printf("rank = %3d\n",rank);

  PetscFinalize();

  MPI_Finalize();
  return 0;
}
