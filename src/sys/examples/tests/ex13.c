#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex13.c,v 1.1 1998/05/19 15:41:22 bsmith Exp bsmith $";
#endif

/*
     Tests PetscSetCommWorld()
*/
#include "petsc.h"

int main(int argc, char **argv) 
{
  int      ierr,rank,size;
  MPI_Comm newcomm;

  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);

  /*
       make two new communicators each half the size of original
  */
  MPI_Comm_split(MPI_COMM_WORLD,2*rank<size,0,&newcomm);

  ierr = PetscSetCommWorld(newcomm);
  if (ierr) {
    fprintf(stderr,"Unable to set PETSC_COMM_WORLD\n");
  }

  PetscInitialize(&argc,&argv,PETSC_NULL,PETSC_NULL);
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  printf("rank %d\n",rank);

  PetscFinalize();

  MPI_Finalize();
  return 0;
}
