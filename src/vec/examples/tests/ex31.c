/*$Id: ex16.c,v 1.2 2000/05/04 16:24:58 bsmith Exp $*/

/* 
   Demonstrates PetscMatlabEngineXXX()
*/

#include "petsc.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int        ierr,rank;
  char       buffer[256],*output,user[256];
  PetscTruth userhappy = PETSC_FALSE;

  PetscInitialize(&argc,&argv,(char *)0,0);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

  ierr = PetscMatlabEngineGetOutput(PETSC_COMM_WORLD,&output);

  ierr = PetscMatlabEngineEvaluate(PETSC_COMM_WORLD,"MPI_Comm_rank");
  ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d]Processor rank is %s",rank,output);CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,">>");CHKERRQ(ierr);
  ierr = PetscSynchronizedFGets(PETSC_COMM_WORLD,stdin,256,user);CHKERRQ(ierr); 
  ierr = PetscStrncmp(user,"exit",4,&userhappy);CHKERRQ(ierr);
  while (!userhappy) {
    ierr = PetscMatlabEngineEvaluate(PETSC_COMM_WORLD,user);
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d]The result is %s",rank,output);CHKERRQ(ierr);
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,">>");CHKERRQ(ierr);
    ierr = PetscSynchronizedFGets(PETSC_COMM_WORLD,stdin,256,user);CHKERRQ(ierr);
    ierr = PetscStrncmp(user,"exit",4,&userhappy);CHKERRQ(ierr);
  }
  PetscFinalize();
  return 0;
}
 
