/*$Id: ex31.c,v 1.2 2000/05/05 18:29:33 bsmith Exp balay $*/

/* 
   Demonstrates PetscMatlabEngineXXX()
*/

#include "petscvec.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int        ierr,rank,n = 5;
  char       *output;
  Vec        x;

  PetscInitialize(&argc,&argv,(char *)0,0);

  ierr = VecCreate(PETSC_COMM_WORLD,PETSC_DECIDE,n,&x);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);

  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  ierr = PetscMatlabEngineGetOutput(PETSC_COMM_WORLD,&output);
  ierr = PetscMatlabEngineEvaluate(PETSC_COMM_WORLD,"MPI_Comm_rank");
  ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d]Processor rank is %s",rank,output);CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD);CHKERRQ(ierr);

  ierr = PetscObjectSetName((PetscObject)x,"x");CHKERRQ(ierr);
  ierr = PetscMatlabEnginePut((PetscObject)x);CHKERRQ(ierr);
  ierr = PetscMatlabEngineEvaluate(PETSC_COMM_WORLD,"x = x + MPI_Comm_rank;\n");
  ierr = PetscMatlabEngineGet((PetscObject)x);CHKERRQ(ierr);

  ierr = PetscMatlabEngineEvaluate(PETSC_COMM_WORLD,"whos\n");
  ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d]The result is %s",rank,output);CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD);CHKERRQ(ierr);

  ierr = VecView(x,VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecDestroy(x);CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}
 
