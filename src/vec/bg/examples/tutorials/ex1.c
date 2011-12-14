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
  PetscBool      test_bcast,test_reduce,test_degree;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);

  test_bcast = PETSC_FALSE;
  test_reduce = PETSC_FALSE;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","PetscBG Test Options","none");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_bcast","Test broadcast","",test_bcast,&test_bcast,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_reduce","Test reduction","",test_reduce,&test_reduce,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_degree","Test computation of vertex degree","",test_degree,&test_degree,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  nowned = 2 + (PetscInt)(rank == 0);
  nlocal = 2 + (PetscInt)(rank > 0);
  ierr = PetscMalloc(nlocal*sizeof(*remote),&remote);CHKERRQ(ierr);
  /* Left periodic neighbor */
  remote[0].rank = (rank+size-1)%size;
  remote[0].index = 1;
  /* Right periodic neighbor */
  remote[1].rank = (rank+1)%size;
  remote[1].index = 0;
  if (rank > 0) {               /* All processes reference rank 0, index 1 */
    remote[2].rank = 0;
    remote[2].index = 2;
  }

  /* Create a bipartite graph for communication. In this example, the local space is dense, so we pass PETSC_NULL. */
  ierr = PetscBGCreate(PETSC_COMM_WORLD,&bg);CHKERRQ(ierr);
  ierr = PetscBGSetGraph(bg,nowned,nlocal,PETSC_NULL,PETSC_COPY_VALUES,remote,PETSC_OWN_POINTER);CHKERRQ(ierr);

  /* View graph, mostly useful for debugging purposes. */
  ierr = PetscBGView(bg,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* Allocate space for send and recieve buffers. This example communicates PetscInt, but other types, including
   * user-defined structures, could also be used. */
  ierr = PetscMalloc2(nowned,PetscInt,&owned,nlocal,PetscInt,&local);CHKERRQ(ierr);

  if (test_bcast) {
    /* Set owned buffer to be broadcast */
    for (i=0; i<nowned; i++) owned[i] = 100*(rank+1) + i;
    /* Initialize local buffer, these values are never used. */
    for (i=0; i<nlocal; i++) local[i] = -1;
    /* Broadcast entries from owned to local. Computation or other communication can be performed between the begin and end calls. */
    ierr = PetscBGBcastBegin(bg,MPIU_INT,owned,local);CHKERRQ(ierr);
    ierr = PetscBGBcastEnd(bg,MPIU_INT,owned,local);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## Bcast Owned\n");CHKERRQ(ierr);
    ierr = PetscIntView(nowned,owned,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## Bcast Local\n");CHKERRQ(ierr);
    ierr = PetscIntView(nlocal,local,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  if (test_reduce) {
    /* Initialize owned buffer in which the result of the reduction will appear. */
    for (i=0; i<nowned; i++) owned[i] = 100*(rank+1) + i;
    /* Set local values to reduce. */
    for (i=0; i<nlocal; i++) local[i] = 1000*(rank+1) + 10*i;
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## Pre-Reduce Owned\n");CHKERRQ(ierr);
    ierr = PetscIntView(nowned,owned,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    /* Perform reduction. Computation or other communication can be performed between the begin and end calls. */
    ierr = PetscBGReduceBegin(bg,MPIU_INT,local,owned,MPIU_SUM);CHKERRQ(ierr);
    ierr = PetscBGReduceEnd(bg,MPIU_INT,local,owned,MPIU_SUM);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## Reduce Local\n");CHKERRQ(ierr);
    ierr = PetscIntView(nlocal,local,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## Reduce Owned\n");CHKERRQ(ierr);
    ierr = PetscIntView(nowned,owned,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  if (test_degree) {
    const PetscInt *degree;
    ierr = PetscBGComputeDegreeBegin(bg,&degree);CHKERRQ(ierr);
    ierr = PetscBGComputeDegreeEnd(bg,&degree);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## Degree Owned\n");CHKERRQ(ierr);
    ierr = PetscIntView(nowned,degree,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  /* Clean up local space and storage for bipartite graph. */
  ierr = PetscFree2(owned,local);CHKERRQ(ierr);
  ierr = PetscBGDestroy(&bg);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
