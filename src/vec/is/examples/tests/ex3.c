/*$Id: ex3.c,v 1.11 2000/05/05 22:14:45 balay Exp bsmith $*/
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
ierr = PetscMalloc(n*sizeof(int),&(  indices ));CHKERRQ(ierr);
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
    ierr = ISView(newis,PETSC_VIEWER_STDOUT_SELF);CHKERRA(ierr);
  }

  ierr = ISDestroy(newis);CHKERRA(ierr);
  ierr = ISDestroy(is);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
 






