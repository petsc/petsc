/*$Id: ex3.c,v 1.15 2001/01/22 23:02:58 bsmith Exp balay $*/
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

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr); 
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);

  /*
     Create IS
  */
  n = 4 + rank;
  ierr = PetscMalloc(n*sizeof(int),&indices);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    indices[i] = rank + i;
  }
  ierr = ISCreateGeneral(PETSC_COMM_WORLD,n,indices,&is);CHKERRQ(ierr);
  ierr = PetscFree(indices);CHKERRQ(ierr);

  /*
      Stick them together from all processors 
  */
  ierr = ISAllGather(is,&newis);CHKERRQ(ierr);

  if (!rank) {
    ierr = ISView(newis,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  }

  ierr = ISDestroy(newis);CHKERRQ(ierr);
  ierr = ISDestroy(is);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
 






