/*$Id: ex3.c,v 1.10 2000/01/11 20:59:59 bsmith Exp balay $*/
/*
       Tests ISAllGather()
*/

static char help[] = "Tests ISAllGather()\n\n";

#include "petscis.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int        i,n,ierr,*indices,rank,size;
  IS         is,newis;

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRA(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRA(ierr);

  /*
     Create IS
  */
  n = 4 + rank;
  indices = (int*)PetscMalloc(n*sizeof(int));CHKPTRQ(indices);
  for (i=0; i<n; i++) {
    indices[i] = rank + i;
  }
  ierr = ISCreateGeneral(PETSC_COMM_WORLD,n,indices,&is);CHKERRA(ierr);
  ierr = PetscFree(indices);CHKERRQ(ierr);

  /*
      Stick them together from all processors 
  */
  ierr = ISAllGather(is,&newis);CHKERRA(ierr);

  if (!rank) {
    ierr = ISView(newis,VIEWER_STDOUT_SELF);CHKERRA(ierr);
  }

  ierr = ISDestroy(newis);CHKERRA(ierr);
  ierr = ISDestroy(is);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
 






