#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex3.c,v 1.2 1997/10/28 14:20:59 bsmith Exp bsmith $";
#endif
/*
       Tests ISAllGather()
*/

static char help[] = "Tests ISAllGather()\n\n";

#include "is.h"

int main(int argc,char **argv)
{
  int        i, n, ierr,*indices,rank,size;
  IS         is,newis;

  PetscInitialize(&argc,&argv,(char*)0,help);
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  MPI_Comm_size(PETSC_COMM_WORLD,&size);

  /*
     Create IS
  */
  n = 4 + rank;
  indices = (int *) PetscMalloc( n*sizeof(int) ); CHKPTRQ(indices);
  for ( i=0; i<n; i++ ) {
    indices[i] = rank + i;
  }
  ierr = ISCreateGeneral(PETSC_COMM_WORLD,n,indices,&is); CHKERRA(ierr);
  PetscFree(indices);

  /*
      Stick them together from all processors 
  */
  ierr = ISAllGather(is,&newis);CHKERRA(ierr);

  if (rank == 0) {
    ierr = ISView(newis,VIEWER_STDOUT_SELF); CHKERRA(ierr);
  }

  ierr = ISDestroy(newis); CHKERRA(ierr);
  ierr = ISDestroy(is); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
 






