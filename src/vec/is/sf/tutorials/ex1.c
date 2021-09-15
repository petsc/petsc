static const char help[] = "Test star forest communication (PetscSF)\n\n";

/*T
    Description: A star is a simple tree with one root and zero or more leaves.
    A star forest is a union of disjoint stars.
    Many common communication patterns can be expressed as updates of rootdata using leafdata and vice-versa.
    This example creates a star forest, communicates values using the graph (see options for types of communication), views the graph, then destroys it.
T*/

/*
  Include petscsf.h so we can use PetscSF objects. Note that this automatically
  includes petscsys.h.
*/
#include <petscsf.h>
#include <petscviewer.h>

/* like PetscSFView() but with alternative array of local indices */
static PetscErrorCode PetscSFViewCustomLocals_Private(PetscSF sf,const PetscInt locals[],PetscViewer viewer)
{
  const PetscSFNode *iremote;
  PetscInt          i,nroots,nleaves,nranks;
  PetscMPIInt       rank;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)sf),&rank);CHKERRMPI(ierr);
  ierr = PetscSFGetGraph(sf,&nroots,&nleaves,NULL,&iremote);CHKERRQ(ierr);
  ierr = PetscSFGetRootRanks(sf,&nranks,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Number of roots=%D, leaves=%D, remote ranks=%D\n",rank,nroots,nleaves,nranks);CHKERRQ(ierr);
  for (i=0; i<nleaves; i++) {
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] %D <- (%D,%D)\n",rank,locals[i],iremote[i].rank,iremote[i].index);CHKERRQ(ierr);
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       i,nroots,nrootsalloc,nleaves,nleavesalloc,*mine,stride;
  PetscSFNode    *remote;
  PetscMPIInt    rank,size;
  PetscSF        sf;
  PetscBool      test_all,test_bcast,test_bcastop,test_reduce,test_degree,test_fetchandop,test_gather,test_scatter,test_embed,test_invert,test_sf_distribute,test_char;
  MPI_Op         mop=MPI_OP_NULL; /* initialize to prevent compiler warnings with cxx_quad build */
  char           opstring[256];
  PetscBool      strflg;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);

  ierr            = PetscOptionsBegin(PETSC_COMM_WORLD,"","PetscSF Test Options","none");CHKERRQ(ierr);
  test_all        = PETSC_FALSE;
  ierr            = PetscOptionsBool("-test_all","Test all SF communications","",test_all,&test_all,NULL);CHKERRQ(ierr);
  test_bcast      = test_all;
  ierr            = PetscOptionsBool("-test_bcast","Test broadcast","",test_bcast,&test_bcast,NULL);CHKERRQ(ierr);
  test_bcastop    = test_all;
  ierr            = PetscOptionsBool("-test_bcastop","Test broadcast and reduce","",test_bcastop,&test_bcastop,NULL);CHKERRQ(ierr);
  test_reduce     = test_all;
  ierr            = PetscOptionsBool("-test_reduce","Test reduction","",test_reduce,&test_reduce,NULL);CHKERRQ(ierr);
  test_char       = test_all;
  ierr            = PetscOptionsBool("-test_char","Test signed char, unsigned char, and char","",test_char,&test_char,NULL);CHKERRQ(ierr);
  mop             = MPI_SUM;
  ierr            = PetscStrcpy(opstring,"sum");CHKERRQ(ierr);
  ierr            = PetscOptionsString("-test_op","Designate which MPI_Op to use","",opstring,opstring,sizeof(opstring),NULL);CHKERRQ(ierr);
  ierr = PetscStrcmp("sum",opstring,&strflg);CHKERRQ(ierr);
  if (strflg) {
    mop = MPIU_SUM;
  }
  ierr = PetscStrcmp("prod",opstring,&strflg);CHKERRQ(ierr);
  if (strflg) {
    mop = MPI_PROD;
  }
  ierr = PetscStrcmp("max",opstring,&strflg);CHKERRQ(ierr);
  if (strflg) {
    mop = MPI_MAX;
  }
  ierr = PetscStrcmp("min",opstring,&strflg);CHKERRQ(ierr);
  if (strflg) {
    mop = MPI_MIN;
  }
  ierr = PetscStrcmp("land",opstring,&strflg);CHKERRQ(ierr);
  if (strflg) {
    mop = MPI_LAND;
  }
  ierr = PetscStrcmp("band",opstring,&strflg);CHKERRQ(ierr);
  if (strflg) {
    mop = MPI_BAND;
  }
  ierr = PetscStrcmp("lor",opstring,&strflg);CHKERRQ(ierr);
  if (strflg) {
    mop = MPI_LOR;
  }
  ierr = PetscStrcmp("bor",opstring,&strflg);CHKERRQ(ierr);
  if (strflg) {
    mop = MPI_BOR;
  }
  ierr = PetscStrcmp("lxor",opstring,&strflg);CHKERRQ(ierr);
  if (strflg) {
    mop = MPI_LXOR;
  }
  ierr = PetscStrcmp("bxor",opstring,&strflg);CHKERRQ(ierr);
  if (strflg) {
    mop = MPI_BXOR;
  }
  test_degree     = test_all;
  ierr            = PetscOptionsBool("-test_degree","Test computation of vertex degree","",test_degree,&test_degree,NULL);CHKERRQ(ierr);
  test_fetchandop = test_all;
  ierr            = PetscOptionsBool("-test_fetchandop","Test atomic Fetch-And-Op","",test_fetchandop,&test_fetchandop,NULL);CHKERRQ(ierr);
  test_gather     = test_all;
  ierr            = PetscOptionsBool("-test_gather","Test point gather","",test_gather,&test_gather,NULL);CHKERRQ(ierr);
  test_scatter    = test_all;
  ierr            = PetscOptionsBool("-test_scatter","Test point scatter","",test_scatter,&test_scatter,NULL);CHKERRQ(ierr);
  test_embed      = test_all;
  ierr            = PetscOptionsBool("-test_embed","Test point embed","",test_embed,&test_embed,NULL);CHKERRQ(ierr);
  test_invert     = test_all;
  ierr            = PetscOptionsBool("-test_invert","Test point invert","",test_invert,&test_invert,NULL);CHKERRQ(ierr);
  stride          = 1;
  ierr            = PetscOptionsInt("-stride","Stride for leaf and root data","",stride,&stride,NULL);CHKERRQ(ierr);
  test_sf_distribute = PETSC_FALSE;
  ierr            = PetscOptionsBool("-test_sf_distribute","Create an SF that 'distributes' to each process, like an alltoall","",test_sf_distribute,&test_sf_distribute,NULL);CHKERRQ(ierr);
  ierr            = PetscOptionsString("-test_op","Designate which MPI_Op to use","",opstring,opstring,sizeof(opstring),NULL);CHKERRQ(ierr);
  ierr            = PetscOptionsEnd();CHKERRQ(ierr);

  if (test_sf_distribute) {
    nroots = size;
    nrootsalloc = size;
    nleaves = size;
    nleavesalloc = size;
    mine = NULL;
    ierr = PetscMalloc1(nleaves,&remote);CHKERRQ(ierr);
    for (i=0; i<size; i++) {
      remote[i].rank = i;
      remote[i].index = rank;
    }
  } else {
    nroots       = 2 + (PetscInt)(rank == 0);
    nrootsalloc  = nroots * stride;
    nleaves      = 2 + (PetscInt)(rank > 0);
    nleavesalloc = nleaves * stride;
    mine         = NULL;
    if (stride > 1) {
      PetscInt i;

      ierr = PetscMalloc1(nleaves,&mine);CHKERRQ(ierr);
      for (i = 0; i < nleaves; i++) {
        mine[i] = stride * i;
      }
    }
    ierr = PetscMalloc1(nleaves,&remote);CHKERRQ(ierr);
    /* Left periodic neighbor */
    remote[0].rank  = (rank+size-1)%size;
    remote[0].index = 1 * stride;
    /* Right periodic neighbor */
    remote[1].rank  = (rank+1)%size;
    remote[1].index = 0 * stride;
    if (rank > 0) {               /* All processes reference rank 0, index 1 */
      remote[2].rank  = 0;
      remote[2].index = 2 * stride;
    }
  }

  /* Create a star forest for communication. In this example, the leaf space is dense, so we pass NULL. */
  ierr = PetscSFCreate(PETSC_COMM_WORLD,&sf);CHKERRQ(ierr);
  ierr = PetscSFSetFromOptions(sf);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(sf,nrootsalloc,nleaves,mine,PETSC_OWN_POINTER,remote,PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr = PetscSFSetUp(sf);CHKERRQ(ierr);

  /* View graph, mostly useful for debugging purposes. */
  ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL);CHKERRQ(ierr);
  ierr = PetscSFView(sf,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  if (test_bcast) {             /* broadcast rootdata into leafdata */
    PetscInt *rootdata,*leafdata;
    /* Allocate space for send and receive buffers. This example communicates PetscInt, but other types, including
     * user-defined structures, could also be used. */
    ierr = PetscMalloc2(nrootsalloc,&rootdata,nleavesalloc,&leafdata);CHKERRQ(ierr);
    /* Set rootdata buffer to be broadcast */
    for (i=0; i<nrootsalloc; i++) rootdata[i] = -1;
    for (i=0; i<nroots; i++) rootdata[i*stride] = 100*(rank+1) + i;
    /* Initialize local buffer, these values are never used. */
    for (i=0; i<nleavesalloc; i++) leafdata[i] = -1;
    /* Broadcast entries from rootdata to leafdata. Computation or other communication can be performed between the begin and end calls. */
    ierr = PetscSFBcastBegin(sf,MPIU_INT,rootdata,leafdata,MPI_REPLACE);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(sf,MPIU_INT,rootdata,leafdata,MPI_REPLACE);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## Bcast Rootdata\n");CHKERRQ(ierr);
    ierr = PetscIntView(nrootsalloc,rootdata,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## Bcast Leafdata\n");CHKERRQ(ierr);
    ierr = PetscIntView(nleavesalloc,leafdata,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscFree2(rootdata,leafdata);CHKERRQ(ierr);
  }

  if (test_bcast && test_char) { /* Bcast with char */
    PetscInt len;
    char buf[256];
    char *rootdata,*leafdata;
    ierr = PetscMalloc2(nrootsalloc,&rootdata,nleavesalloc,&leafdata);CHKERRQ(ierr);
    /* Set rootdata buffer to be broadcast */
    for (i=0; i<nrootsalloc; i++) rootdata[i] = '*';
    for (i=0; i<nroots; i++) rootdata[i*stride] = 'A' + rank*3 + i; /* rank is very small, so it is fine to compute a char */
    /* Initialize local buffer, these values are never used. */
    for (i=0; i<nleavesalloc; i++) leafdata[i] = '?';

    ierr = PetscSFBcastBegin(sf,MPI_CHAR,rootdata,leafdata,MPI_REPLACE);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(sf,MPI_CHAR,rootdata,leafdata,MPI_REPLACE);CHKERRQ(ierr);

    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## Bcast Rootdata in type of char\n");CHKERRQ(ierr);
    len  = 0; ierr = PetscSNPrintf(buf,256,"%4d:",rank);CHKERRQ(ierr); len += 5;
    for (i=0; i<nrootsalloc; i++) {ierr = PetscSNPrintf(buf+len,256-len,"%5c",rootdata[i]);CHKERRQ(ierr); len += 5;}
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"%s\n",buf);CHKERRQ(ierr);
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);CHKERRQ(ierr);

    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## Bcast Leafdata in type of char\n");CHKERRQ(ierr);
    len = 0; ierr = PetscSNPrintf(buf,256,"%4d:",rank);CHKERRQ(ierr); len += 5;
    for (i=0; i<nleavesalloc; i++) {ierr = PetscSNPrintf(buf+len,256-len,"%5c",leafdata[i]);CHKERRQ(ierr); len += 5;}
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"%s\n",buf);CHKERRQ(ierr);
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);CHKERRQ(ierr);

    ierr = PetscFree2(rootdata,leafdata);CHKERRQ(ierr);
  }

  if (test_bcastop) {         /* Reduce rootdata into leafdata */
    PetscInt *rootdata,*leafdata;
    /* Allocate space for send and receive buffers. This example communicates PetscInt, but other types, including
     * user-defined structures, could also be used. */
    ierr = PetscMalloc2(nrootsalloc,&rootdata,nleavesalloc,&leafdata);CHKERRQ(ierr);
    /* Set rootdata buffer to be broadcast */
    for (i=0; i<nrootsalloc; i++) rootdata[i] = -1;
    for (i=0; i<nroots; i++) rootdata[i*stride] = 100*(rank+1) + i;
    /* Set leaf values to reduce with */
    for (i=0; i<nleavesalloc; i++) leafdata[i] = -10*(rank+1) - i;
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## Pre-BcastAndOp Leafdata\n");CHKERRQ(ierr);
    ierr = PetscIntView(nleavesalloc,leafdata,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    /* Broadcast entries from rootdata to leafdata. Computation or other communication can be performed between the begin and end calls. */
    ierr = PetscSFBcastBegin(sf,MPIU_INT,rootdata,leafdata,mop);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(sf,MPIU_INT,rootdata,leafdata,mop);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## BcastAndOp Rootdata\n");CHKERRQ(ierr);
    ierr = PetscIntView(nrootsalloc,rootdata,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## BcastAndOp Leafdata\n");CHKERRQ(ierr);
    ierr = PetscIntView(nleavesalloc,leafdata,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscFree2(rootdata,leafdata);CHKERRQ(ierr);
  }

  if (test_reduce) {            /* Reduce leafdata into rootdata */
    PetscInt *rootdata,*leafdata;
    ierr = PetscMalloc2(nrootsalloc,&rootdata,nleavesalloc,&leafdata);CHKERRQ(ierr);
    /* Initialize rootdata buffer in which the result of the reduction will appear. */
    for (i=0; i<nrootsalloc; i++) rootdata[i] = -1;
    for (i=0; i<nroots; i++) rootdata[i*stride] = 100*(rank+1) + i;
    /* Set leaf values to reduce. */
    for (i=0; i<nleavesalloc; i++) leafdata[i] = -1;
    for (i=0; i<nleaves; i++) leafdata[i*stride] = 1000*(rank+1) + 10*i;
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## Pre-Reduce Rootdata\n");CHKERRQ(ierr);
    ierr = PetscIntView(nrootsalloc,rootdata,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    /* Perform reduction. Computation or other communication can be performed between the begin and end calls.
     * This example sums the values, but other MPI_Ops can be used (e.g MPI_MAX, MPI_PROD). */
    ierr = PetscSFReduceBegin(sf,MPIU_INT,leafdata,rootdata,mop);CHKERRQ(ierr);
    ierr = PetscSFReduceEnd(sf,MPIU_INT,leafdata,rootdata,mop);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## Reduce Leafdata\n");CHKERRQ(ierr);
    ierr = PetscIntView(nleavesalloc,leafdata,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## Reduce Rootdata\n");CHKERRQ(ierr);
    ierr = PetscIntView(nrootsalloc,rootdata,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscFree2(rootdata,leafdata);CHKERRQ(ierr);
  }

  if (test_reduce && test_char) { /* Reduce with signed char */
    PetscInt len;
    char buf[256];
    signed char *rootdata,*leafdata;
    ierr = PetscMalloc2(nrootsalloc,&rootdata,nleavesalloc,&leafdata);CHKERRQ(ierr);
    /* Initialize rootdata buffer in which the result of the reduction will appear. */
    for (i=0; i<nrootsalloc; i++) rootdata[i] = -1;
    for (i=0; i<nroots; i++) rootdata[i*stride] = 10*(rank+1) + i;
    /* Set leaf values to reduce. */
    for (i=0; i<nleavesalloc; i++) leafdata[i] = -1;
    for (i=0; i<nleaves; i++) leafdata[i*stride] = 50*(rank+1) + 10*i;
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## Pre-Reduce Rootdata in type of signed char\n");CHKERRQ(ierr);

    len = 0; ierr = PetscSNPrintf(buf,256,"%4d:",rank);CHKERRQ(ierr); len += 5;
    for (i=0; i<nrootsalloc; i++) {ierr = PetscSNPrintf(buf+len,256-len,"%5d",rootdata[i]);CHKERRQ(ierr); len += 5;}
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"%s\n",buf);CHKERRQ(ierr);
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);CHKERRQ(ierr);

    /* Using MPI_CHAR should trigger an error since MPI standard does not support reduction on MPI_CHAR.
       Testing with -test_op max, one can see the sign does take effect in MPI_MAX.
     */
    ierr = PetscSFReduceBegin(sf,MPI_SIGNED_CHAR,leafdata,rootdata,mop);CHKERRQ(ierr);
    ierr = PetscSFReduceEnd(sf,MPI_SIGNED_CHAR,leafdata,rootdata,mop);CHKERRQ(ierr);

    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## Reduce Leafdata in type of signed char\n");CHKERRQ(ierr);
    len  = 0; ierr = PetscSNPrintf(buf,256,"%4d:",rank);CHKERRQ(ierr); len += 5;
    for (i=0; i<nleavesalloc; i++) {ierr = PetscSNPrintf(buf+len,256-len,"%5d",leafdata[i]);CHKERRQ(ierr); len += 5;}
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"%s\n",buf);CHKERRQ(ierr);
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);CHKERRQ(ierr);

    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## Reduce Rootdata in type of signed char\n");CHKERRQ(ierr);
    len = 0; ierr = PetscSNPrintf(buf,256,"%4d:",rank);CHKERRQ(ierr); len += 5;
    for (i=0; i<nrootsalloc; i++) {ierr = PetscSNPrintf(buf+len,256-len,"%5d",rootdata[i]);CHKERRQ(ierr); len += 5;}
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"%s\n",buf);CHKERRQ(ierr);
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);CHKERRQ(ierr);

    ierr = PetscFree2(rootdata,leafdata);CHKERRQ(ierr);
  }

  if (test_reduce && test_char) { /* Reduce with unsigned char */
    PetscInt len;
    char buf[256];
    unsigned char *rootdata,*leafdata;
    ierr = PetscMalloc2(nrootsalloc,&rootdata,nleavesalloc,&leafdata);CHKERRQ(ierr);
    /* Initialize rootdata buffer in which the result of the reduction will appear. */
    for (i=0; i<nrootsalloc; i++) rootdata[i] = 0;
    for (i=0; i<nroots; i++) rootdata[i*stride] = 10*(rank+1) + i;
    /* Set leaf values to reduce. */
    for (i=0; i<nleavesalloc; i++) leafdata[i] = 0;
    for (i=0; i<nleaves; i++) leafdata[i*stride] = 50*(rank+1) + 10*i;
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## Pre-Reduce Rootdata in type of unsigned char\n");CHKERRQ(ierr);

    len = 0; ierr = PetscSNPrintf(buf,256,"%4d:",rank);CHKERRQ(ierr); len += 5;
    for (i=0; i<nrootsalloc; i++) {ierr = PetscSNPrintf(buf+len,256-len,"%5u",rootdata[i]);CHKERRQ(ierr); len += 5;}
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"%s\n",buf);CHKERRQ(ierr);
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);CHKERRQ(ierr);

    /* Using MPI_CHAR should trigger an error since MPI standard does not support reduction on MPI_CHAR.
       Testing with -test_op max, one can see the sign does take effect in MPI_MAX.
     */
    ierr = PetscSFReduceBegin(sf,MPI_UNSIGNED_CHAR,leafdata,rootdata,mop);CHKERRQ(ierr);
    ierr = PetscSFReduceEnd(sf,MPI_UNSIGNED_CHAR,leafdata,rootdata,mop);CHKERRQ(ierr);

    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## Reduce Leafdata in type of unsigned char\n");CHKERRQ(ierr);
    len  = 0; ierr = PetscSNPrintf(buf,256,"%4d:",rank);CHKERRQ(ierr); len += 5;
    for (i=0; i<nleavesalloc; i++) {ierr = PetscSNPrintf(buf+len,256-len,"%5u",leafdata[i]);CHKERRQ(ierr); len += 5;}
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"%s\n",buf);CHKERRQ(ierr);
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);CHKERRQ(ierr);

    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## Reduce Rootdata in type of unsigned char\n");CHKERRQ(ierr);
    len = 0; ierr = PetscSNPrintf(buf,256,"%4d:",rank);CHKERRQ(ierr); len += 5;
    for (i=0; i<nrootsalloc; i++) {ierr = PetscSNPrintf(buf+len,256-len,"%5u",rootdata[i]);CHKERRQ(ierr); len += 5;}
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"%s\n",buf);CHKERRQ(ierr);
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);CHKERRQ(ierr);

    ierr = PetscFree2(rootdata,leafdata);CHKERRQ(ierr);
  }

  if (test_degree) {
    const PetscInt *degree;
    ierr = PetscSFComputeDegreeBegin(sf,&degree);CHKERRQ(ierr);
    ierr = PetscSFComputeDegreeEnd(sf,&degree);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## Root degrees\n");CHKERRQ(ierr);
    ierr = PetscIntView(nrootsalloc,degree,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  if (test_fetchandop) {
    /* Cannot use text compare here because token ordering is not deterministic */
    PetscInt *leafdata,*leafupdate,*rootdata;
    ierr = PetscMalloc3(nleavesalloc,&leafdata,nleavesalloc,&leafupdate,nrootsalloc,&rootdata);CHKERRQ(ierr);
    for (i=0; i<nleavesalloc; i++) leafdata[i] = -1;
    for (i=0; i<nleaves; i++) leafdata[i*stride] = 1;
    for (i=0; i<nrootsalloc; i++) rootdata[i] = -1;
    for (i=0; i<nroots; i++) rootdata[i*stride] = 0;
    ierr = PetscSFFetchAndOpBegin(sf,MPIU_INT,rootdata,leafdata,leafupdate,mop);CHKERRQ(ierr);
    ierr = PetscSFFetchAndOpEnd(sf,MPIU_INT,rootdata,leafdata,leafupdate,mop);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## Rootdata (sum of 1 from each leaf)\n");CHKERRQ(ierr);
    ierr = PetscIntView(nrootsalloc,rootdata,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## Leafupdate (value at roots prior to my atomic update)\n");CHKERRQ(ierr);
    ierr = PetscIntView(nleavesalloc,leafupdate,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscFree3(leafdata,leafupdate,rootdata);CHKERRQ(ierr);
  }

  if (test_gather) {
    const PetscInt *degree;
    PetscInt       inedges,*indata,*outdata;
    ierr = PetscSFComputeDegreeBegin(sf,&degree);CHKERRQ(ierr);
    ierr = PetscSFComputeDegreeEnd(sf,&degree);CHKERRQ(ierr);
    for (i=0,inedges=0; i<nrootsalloc; i++) inedges += degree[i];
    ierr = PetscMalloc2(inedges,&indata,nleavesalloc,&outdata);CHKERRQ(ierr);
    for (i=0; i<nleavesalloc; i++) outdata[i] = -1;
    for (i=0; i<nleaves; i++) outdata[i*stride] = 1000*(rank+1) + i;
    ierr = PetscSFGatherBegin(sf,MPIU_INT,outdata,indata);CHKERRQ(ierr);
    ierr = PetscSFGatherEnd(sf,MPIU_INT,outdata,indata);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## Gathered data at multi-roots from leaves\n");CHKERRQ(ierr);
    ierr = PetscIntView(inedges,indata,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscFree2(indata,outdata);CHKERRQ(ierr);
  }

  if (test_scatter) {
    const PetscInt *degree;
    PetscInt       j,count,inedges,*indata,*outdata;
    ierr = PetscSFComputeDegreeBegin(sf,&degree);CHKERRQ(ierr);
    ierr = PetscSFComputeDegreeEnd(sf,&degree);CHKERRQ(ierr);
    for (i=0,inedges=0; i<nrootsalloc; i++) inedges += degree[i];
    ierr = PetscMalloc2(inedges,&indata,nleavesalloc,&outdata);CHKERRQ(ierr);
    for (i=0; i<nleavesalloc; i++) outdata[i] = -1;
    for (i=0,count=0; i<nrootsalloc; i++) {
      for (j=0; j<degree[i]; j++) indata[count++] = 1000*(rank+1) + 100*i + j;
    }
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## Data at multi-roots, to scatter to leaves\n");CHKERRQ(ierr);
    ierr = PetscIntView(inedges,indata,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

    ierr = PetscSFScatterBegin(sf,MPIU_INT,indata,outdata);CHKERRQ(ierr);
    ierr = PetscSFScatterEnd(sf,MPIU_INT,indata,outdata);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## Scattered data at leaves\n");CHKERRQ(ierr);
    ierr = PetscIntView(nleavesalloc,outdata,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscFree2(indata,outdata);CHKERRQ(ierr);
  }

  if (test_embed) {
    const PetscInt nroots = 1 + (PetscInt) (rank == 0);
    PetscInt       selected[2];
    PetscSF        esf;

    selected[0] = stride;
    selected[1] = 2*stride;
    ierr = PetscSFCreateEmbeddedRootSF(sf,nroots,selected,&esf);CHKERRQ(ierr);
    ierr = PetscSFSetUp(esf);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## Embedded PetscSF\n");CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL);CHKERRQ(ierr);
    ierr = PetscSFView(esf,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&esf);CHKERRQ(ierr);
  }

  if (test_invert) {
    const PetscInt *degree;
    PetscInt *mRootsOrigNumbering;
    PetscInt inedges;
    PetscSF msf,imsf;

    ierr = PetscSFGetMultiSF(sf,&msf);CHKERRQ(ierr);
    ierr = PetscSFCreateInverseSF(msf,&imsf);CHKERRQ(ierr);
    ierr = PetscSFSetUp(msf);CHKERRQ(ierr);
    ierr = PetscSFSetUp(imsf);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## Multi-SF\n");CHKERRQ(ierr);
    ierr = PetscSFView(msf,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## Multi-SF roots indices in original SF roots numbering\n");CHKERRQ(ierr);
    ierr = PetscSFComputeDegreeBegin(sf,&degree);CHKERRQ(ierr);
    ierr = PetscSFComputeDegreeEnd(sf,&degree);CHKERRQ(ierr);
    ierr = PetscSFComputeMultiRootOriginalNumbering(sf,degree,&inedges,&mRootsOrigNumbering);CHKERRQ(ierr);
    ierr = PetscIntView(inedges,mRootsOrigNumbering,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## Inverse of Multi-SF\n");CHKERRQ(ierr);
    ierr = PetscSFView(imsf,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## Inverse of Multi-SF, original numbering\n");CHKERRQ(ierr);
    ierr = PetscSFViewCustomLocals_Private(imsf,mRootsOrigNumbering,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&imsf);CHKERRQ(ierr);
    ierr = PetscFree(mRootsOrigNumbering);CHKERRQ(ierr);
  }

  /* Clean storage for star forest. */
  ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: 4
      filter: grep -v "type" | grep -v "sort"
      args: -test_bcast -sf_type window -sf_window_sync {{fence active lock}} -sf_window_flavor {{create dynamic allocate}}
      requires: defined(PETSC_HAVE_MPI_ONE_SIDED) defined(PETSC_HAVE_MPI_FEATURE_DYNAMIC_WINDOW)

   test:
      suffix: 2
      nsize: 4
      filter: grep -v "type" | grep -v "sort"
      args: -test_reduce -sf_type window -sf_window_sync {{fence active lock}} -sf_window_flavor {{create dynamic allocate}}
      requires: defined(PETSC_HAVE_MPI_ONE_SIDED) defined(PETSC_HAVE_MPI_FEATURE_DYNAMIC_WINDOW)

   test:
      suffix: 2_basic
      nsize: 4
      args: -test_reduce -sf_type basic

   test:
      suffix: 3
      nsize: 4
      filter: grep -v "type" | grep -v "sort"
      args: -test_degree -sf_type window -sf_window_sync {{fence active lock}} -sf_window_flavor {{create dynamic allocate}}
      requires: defined(PETSC_HAVE_MPI_ONE_SIDED) defined(PETSC_HAVE_MPI_FEATURE_DYNAMIC_WINDOW)

   test:
      suffix: 3_basic
      nsize: 4
      args: -test_degree -sf_type basic

   test:
      suffix: 4
      nsize: 4
      filter: grep -v "type" | grep -v "sort"
      args: -test_gather -sf_type window -sf_window_sync {{fence active lock}} -sf_window_flavor {{create dynamic allocate}}
      requires: defined(PETSC_HAVE_MPI_ONE_SIDED) defined(PETSC_HAVE_MPI_FEATURE_DYNAMIC_WINDOW)

   test:
      suffix: 4_basic
      nsize: 4
      args: -test_gather -sf_type basic

   test:
      suffix: 4_stride
      nsize: 4
      args: -test_gather -sf_type basic -stride 2

   test:
      suffix: 5
      nsize: 4
      filter: grep -v "type" | grep -v "sort"
      args: -test_scatter -sf_type window -sf_window_sync {{fence active lock}} -sf_window_flavor {{create dynamic allocate}}
      requires: defined(PETSC_HAVE_MPI_ONE_SIDED) defined(PETSC_HAVE_MPI_FEATURE_DYNAMIC_WINDOW)

   test:
      suffix: 5_basic
      nsize: 4
      args: -test_scatter -sf_type basic

   test:
      suffix: 5_stride
      nsize: 4
      args: -test_scatter -sf_type basic -stride 2

   test:
      suffix: 6
      nsize: 4
      filter: grep -v "type" | grep -v "sort"
      # No -sf_window_flavor dynamic due to bug https://gitlab.com/petsc/petsc/issues/555
      args: -test_embed -sf_type window -sf_window_sync {{fence active lock}} -sf_window_flavor {{create allocate}}
      requires: defined(PETSC_HAVE_MPI_ONE_SIDED) defined(PETSC_HAVE_MPI_FEATURE_DYNAMIC_WINDOW)

   test:
      suffix: 6_basic
      nsize: 4
      args: -test_embed -sf_type basic

   test:
      suffix: 7
      nsize: 4
      filter: grep -v "type" | grep -v "sort"
      args: -test_invert -sf_type window -sf_window_sync {{fence active lock}} -sf_window_flavor {{create dynamic allocate}}
      requires: defined(PETSC_HAVE_MPI_ONE_SIDED) defined(PETSC_HAVE_MPI_FEATURE_DYNAMIC_WINDOW)

   test:
      suffix: 7_basic
      nsize: 4
      args: -test_invert -sf_type basic

   test:
      suffix: basic
      nsize: 4
      args: -test_bcast -sf_type basic
      output_file: output/ex1_1_basic.out

   test:
      suffix: bcastop_basic
      nsize: 4
      args: -test_bcastop -sf_type basic
      output_file: output/ex1_bcastop_basic.out

   test:
      suffix: 8
      nsize: 3
      filter: grep -v "type" | grep -v "sort"
      args: -test_bcast -test_sf_distribute -sf_type window -sf_window_sync {{fence active lock}} -sf_window_flavor {{create dynamic allocate}}
      requires: defined(PETSC_HAVE_MPI_ONE_SIDED) defined(PETSC_HAVE_MPI_FEATURE_DYNAMIC_WINDOW)

   test:
      suffix: 8_basic
      nsize: 3
      args: -test_bcast -test_sf_distribute -sf_type basic

   test:
      suffix: 9_char
      nsize: 4
      args: -sf_type basic -test_bcast -test_reduce -test_op max -test_char

   # Here we do not test -sf_window_flavor dynamic since it is designed for repeated SFs with few different rootdata pointers
   test:
      suffix: 10
      filter: grep -v "type" | grep -v "sort"
      nsize: 4
      args: -sf_type window -sf_window_sync {{fence active lock}} -sf_window_flavor {{create allocate}} -test_all -test_bcastop 0 -test_fetchandop 0
      requires: defined(PETSC_HAVE_MPI_ONE_SIDED) defined(PETSC_HAVE_MPI_FEATURE_DYNAMIC_WINDOW)

   # The nightly test suite with MPICH uses ch3:sock, which is broken when winsize == 0 in some of the processes
   test:
      suffix: 10_shared
      output_file: output/ex1_10.out
      filter: grep -v "type" | grep -v "sort"
      nsize: 4
      args: -sf_type window -sf_window_sync {{fence active lock}} -sf_window_flavor shared -test_all -test_bcastop 0 -test_fetchandop 0
      requires: defined(PETSC_HAVE_MPI_PROCESS_SHARED_MEMORY) !defined(PETSC_HAVE_MPICH_NUMVERSION) defined(PETSC_HAVE_MPI_ONE_SIDED) defined(PETSC_HAVE_MPI_FEATURE_DYNAMIC_WINDOW)

   test:
      suffix: 10_basic
      nsize: 4
      args: -sf_type basic -test_all -test_bcastop 0 -test_fetchandop 0

TEST*/
