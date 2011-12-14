static const char help[] = "Test bipartite graph communication (PetscBG)\n\n";

/*T
    Description: Creates a bipartite graph based on a set of integers, communicates broadcasts values using the graph,
    views the graph, then destroys it.
T*/

/*
  Include petscbg.h so we can use PetscBG objects. Note that this automatically
  includes petscsys.h.
*/
#include <petscbg.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       i,nowned,nlocal,*owned,*local;
  PetscBGNode    *remote;
  PetscMPIInt    rank,size;
  PetscBG        bg;
  PetscBool      test_bcast,test_reduce;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);

  test_bcast = PETSC_FALSE;
  test_reduce = PETSC_FALSE;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","PetscBG Test Options","none");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_bcast","Test broadcast","",test_bcast,&test_bcast,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_reduce","Test reduction","",test_reduce,&test_reduce,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  nowned = 2 + (PetscInt)(rank == 0);
  nlocal = 2 + (PetscInt)(rank > 0);
  ierr = PetscMalloc(nlocal*sizeof(*remote),&remote);CHKERRQ(ierr);
  remote[0].rank = (rank+size-1)%size;
  remote[0].index = 1;
  remote[1].rank = (rank+1)%size;
  remote[1].index = 0;
  if (rank > 0) {
    remote[2].rank = 0;
    remote[2].index = 2;
  }

  ierr = PetscBGCreate(PETSC_COMM_WORLD,&bg);CHKERRQ(ierr);
  ierr = PetscBGSetGraph(bg,nowned,nlocal,PETSC_NULL,PETSC_COPY_VALUES,remote,PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr = PetscBGView(bg,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = PetscMalloc2(nowned,PetscInt,&owned,nlocal,PetscInt,&local);CHKERRQ(ierr);

  if (test_bcast) {
    for (i=0; i<nowned; i++) owned[i] = 100*(rank+1) + i;
    for (i=0; i<nlocal; i++) local[i] = -1;
    ierr = PetscBGBcastBegin(bg,MPIU_INT,owned,local);CHKERRQ(ierr);
    ierr = PetscBGBcastEnd(bg,MPIU_INT,owned,local);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## Bcast Owned\n");CHKERRQ(ierr);
    ierr = PetscIntView(nowned,owned,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## Bcast Local\n");CHKERRQ(ierr);
    ierr = PetscIntView(nlocal,local,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  if (test_reduce) {
    for (i=0; i<nowned; i++) owned[i] = 100*(rank+1) + i;
    for (i=0; i<nlocal; i++) local[i] = 1000*(rank+1) + 10*i;
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## Pre-Reduce Owned\n");CHKERRQ(ierr);
    ierr = PetscIntView(nowned,owned,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscBGReduceBegin(bg,MPIU_INT,local,owned,MPIU_SUM);CHKERRQ(ierr);
    ierr = PetscBGReduceEnd(bg,MPIU_INT,local,owned,MPIU_SUM);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## Reduce Local\n");CHKERRQ(ierr);
    ierr = PetscIntView(nlocal,local,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## Reduce Owned\n");CHKERRQ(ierr);
    ierr = PetscIntView(nowned,owned,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  ierr = PetscFree2(owned,local);CHKERRQ(ierr);
  ierr = PetscBGDestroy(&bg);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
