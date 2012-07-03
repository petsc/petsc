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
  PetscInt       i,nroots,nleaves;
  PetscSFNode    *remote;
  PetscMPIInt    rank,size;
  PetscSF        sf;
  PetscBool      test_bcast,test_reduce,test_degree,test_fetchandop,test_gather,test_scatter,test_embed,test_invert;

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
  test_embed = PETSC_FALSE;
  ierr = PetscOptionsBool("-test_embed","Test point embed","",test_embed,&test_embed,PETSC_NULL);CHKERRQ(ierr);
  test_invert = PETSC_FALSE;
  ierr = PetscOptionsBool("-test_invert","Test point invert","",test_invert,&test_invert,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  nroots = 2 + (PetscInt)(rank == 0);
  nleaves = 2 + (PetscInt)(rank > 0);
  ierr = PetscMalloc(nleaves*sizeof(*remote),&remote);CHKERRQ(ierr);
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

  /* Create a star forest for communication. In this example, the leaf space is dense, so we pass PETSC_NULL. */
  ierr = PetscSFCreate(PETSC_COMM_WORLD,&sf);CHKERRQ(ierr);
  ierr = PetscSFSetFromOptions(sf);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(sf,nroots,nleaves,PETSC_NULL,PETSC_COPY_VALUES,remote,PETSC_OWN_POINTER);CHKERRQ(ierr);

  /* View graph, mostly useful for debugging purposes. */
  ierr = PetscSFView(sf,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);


  if (test_bcast) {             /* broadcast rootdata into leafdata */
    PetscInt *rootdata,*leafdata;
    /* Allocate space for send and recieve buffers. This example communicates PetscInt, but other types, including
     * user-defined structures, could also be used. */
    ierr = PetscMalloc2(nroots,PetscInt,&rootdata,nleaves,PetscInt,&leafdata);CHKERRQ(ierr);
    /* Set rootdata buffer to be broadcast */
    for (i=0; i<nroots; i++) rootdata[i] = 100*(rank+1) + i;
    /* Initialize local buffer, these values are never used. */
    for (i=0; i<nleaves; i++) leafdata[i] = -1;
    /* Broadcast entries from rootdata to leafdata. Computation or other communication can be performed between the begin and end calls. */
    ierr = PetscSFBcastBegin(sf,MPIU_INT,rootdata,leafdata);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(sf,MPIU_INT,rootdata,leafdata);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## Bcast Rootdata\n");CHKERRQ(ierr);
    ierr = PetscIntView(nroots,rootdata,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## Bcast Leafdata\n");CHKERRQ(ierr);
    ierr = PetscIntView(nleaves,leafdata,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscFree2(rootdata,leafdata);CHKERRQ(ierr);
  }

  if (test_reduce) {            /* Reduce leafdata into rootdata */
    PetscInt *rootdata,*leafdata;
    ierr = PetscMalloc2(nroots,PetscInt,&rootdata,nleaves,PetscInt,&leafdata);CHKERRQ(ierr);
    /* Initialize rootdata buffer in which the result of the reduction will appear. */
    for (i=0; i<nroots; i++) rootdata[i] = 100*(rank+1) + i;
    /* Set leaf values to reduce. */
    for (i=0; i<nleaves; i++) leafdata[i] = 1000*(rank+1) + 10*i;
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## Pre-Reduce Rootdata\n");CHKERRQ(ierr);
    ierr = PetscIntView(nroots,rootdata,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    /* Perform reduction. Computation or other communication can be performed between the begin and end calls.
     * This example sums the values, but other MPI_Ops can be used (e.g MPI_MAX, MPI_PROD). */
    ierr = PetscSFReduceBegin(sf,MPIU_INT,leafdata,rootdata,MPIU_SUM);CHKERRQ(ierr);
    ierr = PetscSFReduceEnd(sf,MPIU_INT,leafdata,rootdata,MPIU_SUM);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## Reduce Leafdata\n");CHKERRQ(ierr);
    ierr = PetscIntView(nleaves,leafdata,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## Reduce Rootdata\n");CHKERRQ(ierr);
    ierr = PetscIntView(nroots,rootdata,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscFree2(rootdata,leafdata);CHKERRQ(ierr);
  }

  if (test_degree) {
    const PetscInt *degree;
    ierr = PetscSFComputeDegreeBegin(sf,&degree);CHKERRQ(ierr);
    ierr = PetscSFComputeDegreeEnd(sf,&degree);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## Root degrees\n");CHKERRQ(ierr);
    ierr = PetscIntView(nroots,degree,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  if (test_fetchandop) {
    /* Cannot use text compare here because token ordering is not deterministic */
    PetscInt    *leafdata,*leafupdate,*rootdata;
    ierr = PetscMalloc3(nleaves,PetscInt,&leafdata,nleaves,PetscInt,&leafupdate,nroots,PetscInt,&rootdata);CHKERRQ(ierr);
    for (i=0; i<nleaves; i++) leafdata[i] = 1;
    for (i=0; i<nroots; i++) rootdata[i] = 0;
    ierr = PetscSFFetchAndOpBegin(sf,MPIU_INT,rootdata,leafdata,leafupdate,MPIU_SUM);CHKERRQ(ierr);
    ierr = PetscSFFetchAndOpEnd(sf,MPIU_INT,rootdata,leafdata,leafupdate,MPIU_SUM);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## Rootdata (sum of 1 from each leaf)\n");CHKERRQ(ierr);
    ierr = PetscIntView(nroots,rootdata,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## Leafupdate (value at roots prior to my atomic update)\n");CHKERRQ(ierr);
    ierr = PetscIntView(nleaves,leafupdate,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscFree3(leafdata,leafupdate,rootdata);CHKERRQ(ierr);
  }

  if (test_gather) {
    const PetscInt *degree;
    PetscInt inedges,*indata,*outdata;
    ierr = PetscSFComputeDegreeBegin(sf,&degree);CHKERRQ(ierr);
    ierr = PetscSFComputeDegreeEnd(sf,&degree);CHKERRQ(ierr);
    for (i=0,inedges=0; i<nroots; i++) inedges += degree[i];
    ierr = PetscMalloc2(inedges,PetscInt,&indata,nleaves,PetscInt,&outdata);CHKERRQ(ierr);
    for (i=0; i<nleaves; i++) outdata[i] = 1000*(rank+1) + i;
    ierr = PetscSFGatherBegin(sf,MPIU_INT,outdata,indata);CHKERRQ(ierr);
    ierr = PetscSFGatherEnd(sf,MPIU_INT,outdata,indata);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## Gathered data at multi-roots from leaves\n");CHKERRQ(ierr);
    ierr = PetscIntView(inedges,indata,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscFree2(indata,outdata);CHKERRQ(ierr);
  }

  if (test_scatter) {
    const PetscInt *degree;
    PetscInt j,count,inedges,*indata,*outdata;
    ierr = PetscSFComputeDegreeBegin(sf,&degree);CHKERRQ(ierr);
    ierr = PetscSFComputeDegreeEnd(sf,&degree);CHKERRQ(ierr);
    for (i=0,inedges=0; i<nroots; i++) inedges += degree[i];
    ierr = PetscMalloc2(inedges,PetscInt,&indata,nleaves,PetscInt,&outdata);CHKERRQ(ierr);
    for (i=0,count=0; i<nroots; i++) {
      for (j=0; j<degree[i]; j++) indata[count++] = 1000*(rank+1) + 100*i + j;
    }
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## Data at multi-roots, to scatter to leaves\n");CHKERRQ(ierr);
    ierr = PetscIntView(inedges,indata,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

    ierr = PetscSFScatterBegin(sf,MPIU_INT,indata,outdata);CHKERRQ(ierr);
    ierr = PetscSFScatterEnd(sf,MPIU_INT,indata,outdata);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## Scattered data at leaves\n");CHKERRQ(ierr);
    ierr = PetscIntView(nleaves,outdata,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscFree2(indata,outdata);CHKERRQ(ierr);
  }

  if (test_embed) {
    const PetscInt nroots = 1 + (PetscInt)!rank,selected[] = {1,2};
    PetscSF esf;
    ierr = PetscSFCreateEmbeddedSF(sf,nroots,selected,&esf);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## Embedded PetscSF\n");CHKERRQ(ierr);
    ierr = PetscSFView(esf,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&esf);CHKERRQ(ierr);
  }

  if (test_invert) {
    PetscSF msf,imsf;
    ierr = PetscSFGetMultiSF(sf,&msf);CHKERRQ(ierr);
    ierr = PetscSFCreateInverseSF(msf,&imsf);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## Multi-SF\n");CHKERRQ(ierr);
    ierr = PetscSFView(msf,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## Inverse of Multi-SF\n");CHKERRQ(ierr);
    ierr = PetscSFView(imsf,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&imsf);CHKERRQ(ierr);
  }

  /* Clean storage for star forest. */
  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
