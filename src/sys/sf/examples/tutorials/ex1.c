static const char help[] = "Test star forest communication (PetscSF)\n\n";

/*T
    Description: A star forest is a simple tree with one root and zero or more leaves.
    Many common communication patterns can be expressed as updates of rootdata using leafdata and vice-versa.
    This example creates a star forest, communicates values using the graph (see options for types of communication), views the graph, then destroys it.
T*/

/*
  Include petscsf.h so we can use PetscSF objects. Note that this automatically
  includes petscsys.h.
*/
#include <petscsf.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       i,nowned,nlocal,*owned,*local;
  PetscSFNode    *remote;
  PetscMPIInt    rank,size;
  PetscSF        sf;
  PetscBool      test_bcast,test_reduce,test_degree,test_fetchandop,test_gather,test_scatter;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","PetscSF Test Options","none");CHKERRQ(ierr);
  test_bcast = PETSC_FALSE;
  ierr = PetscOptionsBool("-test_bcast","Test broadcast","",test_bcast,&test_bcast,PETSC_NULL);CHKERRQ(ierr);
  test_reduce = PETSC_FALSE;
  ierr = PetscOptionsBool("-test_reduce","Test reduction","",test_reduce,&test_reduce,PETSC_NULL);CHKERRQ(ierr);
  test_degree = PETSC_FALSE;
  ierr = PetscOptionsBool("-test_degree","Test computation of vertex degree","",test_degree,&test_degree,PETSC_NULL);CHKERRQ(ierr);
  test_fetchandop = PETSC_FALSE;
  ierr = PetscOptionsBool("-test_fetchandop","Test atomic Fetch-And-Op","",test_fetchandop,&test_fetchandop,PETSC_NULL);CHKERRQ(ierr);
  test_gather = PETSC_FALSE;
  ierr = PetscOptionsBool("-test_gather","Test point gather","",test_gather,&test_gather,PETSC_NULL);CHKERRQ(ierr);
  test_scatter = PETSC_FALSE;
  ierr = PetscOptionsBool("-test_scatter","Test point scatter","",test_scatter,&test_scatter,PETSC_NULL);CHKERRQ(ierr);
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

  /* Create a star forest for communication. In this example, the local space is dense, so we pass PETSC_NULL. */
  ierr = PetscSFCreate(PETSC_COMM_WORLD,&sf);CHKERRQ(ierr);
  ierr = PetscSFSetFromOptions(sf);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(sf,nowned,nlocal,PETSC_NULL,PETSC_COPY_VALUES,remote,PETSC_OWN_POINTER);CHKERRQ(ierr);

  /* View graph, mostly useful for debugging purposes. */
  ierr = PetscSFView(sf,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* Allocate space for send and recieve buffers. This example communicates PetscInt, but other types, including
   * user-defined structures, could also be used. */
  ierr = PetscMalloc2(nowned,PetscInt,&owned,nlocal,PetscInt,&local);CHKERRQ(ierr);

  if (test_bcast) {
    /* Set owned buffer to be broadcast */
    for (i=0; i<nowned; i++) owned[i] = 100*(rank+1) + i;
    /* Initialize local buffer, these values are never used. */
    for (i=0; i<nlocal; i++) local[i] = -1;
    /* Broadcast entries from owned to local. Computation or other communication can be performed between the begin and end calls. */
    ierr = PetscSFBcastBegin(sf,MPIU_INT,owned,local);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(sf,MPIU_INT,owned,local);CHKERRQ(ierr);
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
    ierr = PetscSFReduceBegin(sf,MPIU_INT,local,owned,MPIU_SUM);CHKERRQ(ierr);
    ierr = PetscSFReduceEnd(sf,MPIU_INT,local,owned,MPIU_SUM);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## Reduce Local\n");CHKERRQ(ierr);
    ierr = PetscIntView(nlocal,local,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## Reduce Owned\n");CHKERRQ(ierr);
    ierr = PetscIntView(nowned,owned,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  if (test_degree) {
    const PetscInt *degree;
    ierr = PetscSFComputeDegreeBegin(sf,&degree);CHKERRQ(ierr);
    ierr = PetscSFComputeDegreeEnd(sf,&degree);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## Degree Owned\n");CHKERRQ(ierr);
    ierr = PetscIntView(nowned,degree,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  if (test_fetchandop) {
    /* Cannot use text compare here because token ordering is not deterministic */
    PetscInt    *outgoing,*token,*incoming;
    ierr = PetscMalloc3(nlocal,PetscInt,&outgoing,nlocal,PetscInt,&token,nowned,PetscInt,&incoming);CHKERRQ(ierr);
    for (i=0; i<nlocal; i++) outgoing[i] = 1;
    for (i=0; i<nowned; i++) incoming[i] = 0;
    ierr = PetscSFFetchAndOpBegin(sf,MPIU_INT,incoming,outgoing,token,MPIU_SUM);CHKERRQ(ierr);
    ierr = PetscSFFetchAndOpEnd(sf,MPIU_INT,incoming,outgoing,token,MPIU_SUM);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## Incoming Count\n");CHKERRQ(ierr);
    ierr = PetscIntView(nowned,incoming,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## Outgoing Token\n");CHKERRQ(ierr);
    ierr = PetscIntView(nlocal,token,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscFree3(outgoing,token,incoming);CHKERRQ(ierr);
  }

  if (test_gather) {
    const PetscInt *degree;
    PetscInt inedges,*indata,*outdata;
    ierr = PetscSFComputeDegreeBegin(sf,&degree);CHKERRQ(ierr);
    ierr = PetscSFComputeDegreeEnd(sf,&degree);CHKERRQ(ierr);
    for (i=0,inedges=0; i<nowned; i++) inedges += degree[i];
    ierr = PetscMalloc2(inedges,PetscInt,&indata,nlocal,PetscInt,&outdata);CHKERRQ(ierr);
    for (i=0; i<nlocal; i++) outdata[i] = 1000*(rank+1) + i;
    ierr = PetscSFGatherBegin(sf,MPIU_INT,outdata,indata);CHKERRQ(ierr);
    ierr = PetscSFGatherEnd(sf,MPIU_INT,outdata,indata);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## Gathered data from incoming edges\n");CHKERRQ(ierr);
    ierr = PetscIntView(inedges,indata,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscFree2(indata,outdata);CHKERRQ(ierr);
  }

  if (test_scatter) {
    const PetscInt *degree;
    PetscInt j,count,inedges,*indata,*outdata;
    ierr = PetscSFComputeDegreeBegin(sf,&degree);CHKERRQ(ierr);
    ierr = PetscSFComputeDegreeEnd(sf,&degree);CHKERRQ(ierr);
    for (i=0,inedges=0; i<nowned; i++) inedges += degree[i];
    ierr = PetscMalloc2(inedges,PetscInt,&indata,nlocal,PetscInt,&outdata);CHKERRQ(ierr);
    for (i=0,count=0; i<nowned; i++) {
      for (j=0; j<degree[i]; j++) indata[count++] = 1000*(rank+1) + 100*i + j;
    }
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## Data at incoming edges, to scatter\n");CHKERRQ(ierr);
    ierr = PetscIntView(inedges,indata,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

    ierr = PetscSFScatterBegin(sf,MPIU_INT,indata,outdata);CHKERRQ(ierr);
    ierr = PetscSFScatterEnd(sf,MPIU_INT,indata,outdata);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## Scattered data to outgoing edges\n");CHKERRQ(ierr);
    ierr = PetscIntView(nlocal,outdata,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscFree2(indata,outdata);CHKERRQ(ierr);
  }

  /* Clean up local space and storage for bipartite graph. */
  ierr = PetscFree2(owned,local);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
