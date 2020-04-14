#include <petscsys.h>        /*I  "petscsys.h"  I*/
#include <petsc/private/petscimpl.h>

PetscLogEvent PETSC_BuildTwoSided;
PetscLogEvent PETSC_BuildTwoSidedF;

const char *const PetscBuildTwoSidedTypes[] = {
  "ALLREDUCE",
  "IBARRIER",
  "REDSCATTER",
  "PetscBuildTwoSidedType",
  "PETSC_BUILDTWOSIDED_",
  NULL
};

static PetscBuildTwoSidedType _twosided_type = PETSC_BUILDTWOSIDED_NOTSET;

/*@
   PetscCommBuildTwoSidedSetType - set algorithm to use when building two-sided communication

   Logically Collective

   Input Arguments:
+  comm - PETSC_COMM_WORLD
-  twosided - algorithm to use in subsequent calls to PetscCommBuildTwoSided()

   Level: developer

   Note:
   This option is currently global, but could be made per-communicator.

.seealso: PetscCommBuildTwoSided(), PetscCommBuildTwoSidedGetType()
@*/
PetscErrorCode PetscCommBuildTwoSidedSetType(MPI_Comm comm,PetscBuildTwoSidedType twosided)
{
  PetscFunctionBegin;
  if (PetscDefined(USE_DEBUG)) {                             /* We don't have a PetscObject so can't use PetscValidLogicalCollectiveEnum */
    PetscMPIInt ierr;
    PetscMPIInt b1[2],b2[2];
    b1[0] = -(PetscMPIInt)twosided;
    b1[1] = (PetscMPIInt)twosided;
    ierr  = MPIU_Allreduce(b1,b2,2,MPI_INT,MPI_MAX,comm);CHKERRQ(ierr);
    if (-b2[0] != b2[1]) SETERRQ(comm,PETSC_ERR_ARG_WRONG,"Enum value must be same on all processes");
  }
  _twosided_type = twosided;
  PetscFunctionReturn(0);
}

/*@
   PetscCommBuildTwoSidedGetType - set algorithm to use when building two-sided communication

   Logically Collective

   Output Arguments:
+  comm - communicator on which to query algorithm
-  twosided - algorithm to use for PetscCommBuildTwoSided()

   Level: developer

.seealso: PetscCommBuildTwoSided(), PetscCommBuildTwoSidedSetType()
@*/
PetscErrorCode PetscCommBuildTwoSidedGetType(MPI_Comm comm,PetscBuildTwoSidedType *twosided)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  *twosided = PETSC_BUILDTWOSIDED_NOTSET;
  if (_twosided_type == PETSC_BUILDTWOSIDED_NOTSET) {
    ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
    _twosided_type = PETSC_BUILDTWOSIDED_ALLREDUCE; /* default for small comms, see https://gitlab.com/petsc/petsc/-/merge_requests/2611 */
#if defined(PETSC_HAVE_MPI_IBARRIER)
    if (size > 1024) _twosided_type = PETSC_BUILDTWOSIDED_IBARRIER;
#endif
    ierr = PetscOptionsGetEnum(NULL,NULL,"-build_twosided",PetscBuildTwoSidedTypes,(PetscEnum*)&_twosided_type,NULL);CHKERRQ(ierr);
  }
  *twosided = _twosided_type;
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_MPI_IBARRIER) || defined(PETSC_HAVE_MPIX_IBARRIER)

static PetscErrorCode PetscCommBuildTwoSided_Ibarrier(MPI_Comm comm,PetscMPIInt count,MPI_Datatype dtype,PetscMPIInt nto,const PetscMPIInt *toranks,const void *todata,PetscMPIInt *nfrom,PetscMPIInt **fromranks,void *fromdata)
{
  PetscErrorCode ierr;
  PetscMPIInt    nrecvs,tag,done,i;
  MPI_Aint       lb,unitbytes;
  char           *tdata;
  MPI_Request    *sendreqs,barrier;
  PetscSegBuffer segrank,segdata;
  PetscBool      barrier_started;

  PetscFunctionBegin;
  ierr = PetscCommDuplicate(comm,&comm,&tag);CHKERRQ(ierr);
  ierr = MPI_Type_get_extent(dtype,&lb,&unitbytes);CHKERRQ(ierr);
  if (lb != 0) SETERRQ1(comm,PETSC_ERR_SUP,"Datatype with nonzero lower bound %ld\n",(long)lb);
  tdata = (char*)todata;
  ierr  = PetscMalloc1(nto,&sendreqs);CHKERRQ(ierr);
  for (i=0; i<nto; i++) {
    ierr = MPI_Issend((void*)(tdata+count*unitbytes*i),count,dtype,toranks[i],tag,comm,sendreqs+i);CHKERRQ(ierr);
  }
  ierr = PetscSegBufferCreate(sizeof(PetscMPIInt),4,&segrank);CHKERRQ(ierr);
  ierr = PetscSegBufferCreate(unitbytes,4*count,&segdata);CHKERRQ(ierr);

  nrecvs  = 0;
  barrier = MPI_REQUEST_NULL;
  /* MPICH-3.2 sometimes does not create a request in some "optimized" cases.  This is arguably a standard violation,
   * but we need to work around it. */
  barrier_started = PETSC_FALSE;
  for (done=0; !done; ) {
    PetscMPIInt flag;
    MPI_Status  status;
    ierr = MPI_Iprobe(MPI_ANY_SOURCE,tag,comm,&flag,&status);CHKERRQ(ierr);
    if (flag) {                 /* incoming message */
      PetscMPIInt *recvrank;
      void        *buf;
      ierr      = PetscSegBufferGet(segrank,1,&recvrank);CHKERRQ(ierr);
      ierr      = PetscSegBufferGet(segdata,count,&buf);CHKERRQ(ierr);
      *recvrank = status.MPI_SOURCE;
      ierr      = MPI_Recv(buf,count,dtype,status.MPI_SOURCE,tag,comm,MPI_STATUS_IGNORE);CHKERRQ(ierr);
      nrecvs++;
    }
    if (!barrier_started) {
      PetscMPIInt sent,nsends;
      ierr = PetscMPIIntCast(nto,&nsends);CHKERRQ(ierr);
      ierr = MPI_Testall(nsends,sendreqs,&sent,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
      if (sent) {
#if defined(PETSC_HAVE_MPI_IBARRIER)
        ierr = MPI_Ibarrier(comm,&barrier);CHKERRQ(ierr);
#elif defined(PETSC_HAVE_MPIX_IBARRIER)
        ierr = MPIX_Ibarrier(comm,&barrier);CHKERRQ(ierr);
#endif
        barrier_started = PETSC_TRUE;
        ierr = PetscFree(sendreqs);CHKERRQ(ierr);
      }
    } else {
      ierr = MPI_Test(&barrier,&done,MPI_STATUS_IGNORE);CHKERRQ(ierr);
    }
  }
  *nfrom = nrecvs;
  ierr   = PetscSegBufferExtractAlloc(segrank,fromranks);CHKERRQ(ierr);
  ierr   = PetscSegBufferDestroy(&segrank);CHKERRQ(ierr);
  ierr   = PetscSegBufferExtractAlloc(segdata,fromdata);CHKERRQ(ierr);
  ierr   = PetscSegBufferDestroy(&segdata);CHKERRQ(ierr);
  ierr   = PetscCommDestroy(&comm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

static PetscErrorCode PetscCommBuildTwoSided_Allreduce(MPI_Comm comm,PetscMPIInt count,MPI_Datatype dtype,PetscMPIInt nto,const PetscMPIInt *toranks,const void *todata,PetscMPIInt *nfrom,PetscMPIInt **fromranks,void *fromdata)
{
  PetscErrorCode   ierr;
  PetscMPIInt      size,rank,*iflags,nrecvs,tag,*franks,i,flg;
  MPI_Aint         lb,unitbytes;
  char             *tdata,*fdata;
  MPI_Request      *reqs,*sendreqs;
  MPI_Status       *statuses;
  PetscCommCounter *counter;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = PetscCommDuplicate(comm,&comm,&tag);CHKERRQ(ierr);
  ierr = MPI_Comm_get_attr(comm,Petsc_Counter_keyval,&counter,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Inner PETSc communicator does not have its tag/name counter attribute set");
  if (!counter->iflags) {
    ierr   = PetscCalloc1(size,&counter->iflags);CHKERRQ(ierr);
    iflags = counter->iflags;
  } else {
    iflags = counter->iflags;
    ierr   = PetscArrayzero(iflags,size);CHKERRQ(ierr);
  }
  for (i=0; i<nto; i++) iflags[toranks[i]] = 1;
  ierr     = MPIU_Allreduce(MPI_IN_PLACE,iflags,size,MPI_INT,MPI_SUM,comm);CHKERRQ(ierr);
  nrecvs   = iflags[rank];
  ierr     = MPI_Type_get_extent(dtype,&lb,&unitbytes);CHKERRQ(ierr);
  if (lb != 0) SETERRQ1(comm,PETSC_ERR_SUP,"Datatype with nonzero lower bound %ld\n",(long)lb);
  ierr     = PetscMalloc(nrecvs*count*unitbytes,&fdata);CHKERRQ(ierr);
  tdata    = (char*)todata;
  ierr     = PetscMalloc2(nto+nrecvs,&reqs,nto+nrecvs,&statuses);CHKERRQ(ierr);
  sendreqs = reqs + nrecvs;
  for (i=0; i<nrecvs; i++) {
    ierr = MPI_Irecv((void*)(fdata+count*unitbytes*i),count,dtype,MPI_ANY_SOURCE,tag,comm,reqs+i);CHKERRQ(ierr);
  }
  for (i=0; i<nto; i++) {
    ierr = MPI_Isend((void*)(tdata+count*unitbytes*i),count,dtype,toranks[i],tag,comm,sendreqs+i);CHKERRQ(ierr);
  }
  ierr = MPI_Waitall(nto+nrecvs,reqs,statuses);CHKERRQ(ierr);
  ierr = PetscMalloc1(nrecvs,&franks);CHKERRQ(ierr);
  for (i=0; i<nrecvs; i++) franks[i] = statuses[i].MPI_SOURCE;
  ierr = PetscFree2(reqs,statuses);CHKERRQ(ierr);
  ierr = PetscCommDestroy(&comm);CHKERRQ(ierr);

  *nfrom            = nrecvs;
  *fromranks        = franks;
  *(void**)fromdata = fdata;
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_MPI_REDUCE_SCATTER_BLOCK)
static PetscErrorCode PetscCommBuildTwoSided_RedScatter(MPI_Comm comm,PetscMPIInt count,MPI_Datatype dtype,PetscMPIInt nto,const PetscMPIInt *toranks,const void *todata,PetscMPIInt *nfrom,PetscMPIInt **fromranks,void *fromdata)
{
  PetscErrorCode ierr;
  PetscMPIInt    size,*iflags,nrecvs,tag,*franks,i,flg;
  MPI_Aint       lb,unitbytes;
  char           *tdata,*fdata;
  MPI_Request    *reqs,*sendreqs;
  MPI_Status     *statuses;
  PetscCommCounter *counter;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = PetscCommDuplicate(comm,&comm,&tag);CHKERRQ(ierr);
  ierr = MPI_Comm_get_attr(comm,Petsc_Counter_keyval,&counter,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Inner PETSc communicator does not have its tag/name counter attribute set");
  if (!counter->iflags) {
    ierr   = PetscCalloc1(size,&counter->iflags);CHKERRQ(ierr);
    iflags = counter->iflags;
  } else {
    iflags = counter->iflags;
    ierr   = PetscArrayzero(iflags,size);CHKERRQ(ierr);
  }
  for (i=0; i<nto; i++) iflags[toranks[i]] = 1;
  ierr     = MPI_Reduce_scatter_block(iflags,&nrecvs,1,MPI_INT,MPI_SUM,comm);CHKERRQ(ierr);
  ierr     = MPI_Type_get_extent(dtype,&lb,&unitbytes);CHKERRQ(ierr);
  if (lb != 0) SETERRQ1(comm,PETSC_ERR_SUP,"Datatype with nonzero lower bound %ld\n",(long)lb);
  ierr     = PetscMalloc(nrecvs*count*unitbytes,&fdata);CHKERRQ(ierr);
  tdata    = (char*)todata;
  ierr     = PetscMalloc2(nto+nrecvs,&reqs,nto+nrecvs,&statuses);CHKERRQ(ierr);
  sendreqs = reqs + nrecvs;
  for (i=0; i<nrecvs; i++) {
    ierr = MPI_Irecv((void*)(fdata+count*unitbytes*i),count,dtype,MPI_ANY_SOURCE,tag,comm,reqs+i);CHKERRQ(ierr);
  }
  for (i=0; i<nto; i++) {
    ierr = MPI_Isend((void*)(tdata+count*unitbytes*i),count,dtype,toranks[i],tag,comm,sendreqs+i);CHKERRQ(ierr);
  }
  ierr = MPI_Waitall(nto+nrecvs,reqs,statuses);CHKERRQ(ierr);
  ierr = PetscMalloc1(nrecvs,&franks);CHKERRQ(ierr);
  for (i=0; i<nrecvs; i++) franks[i] = statuses[i].MPI_SOURCE;
  ierr = PetscFree2(reqs,statuses);CHKERRQ(ierr);
  ierr = PetscCommDestroy(&comm);CHKERRQ(ierr);

  *nfrom            = nrecvs;
  *fromranks        = franks;
  *(void**)fromdata = fdata;
  PetscFunctionReturn(0);
}
#endif

/*@C
   PetscCommBuildTwoSided - discovers communicating ranks given one-sided information, moving constant-sized data in the process (often message lengths)

   Collective

   Input Arguments:
+  comm - communicator
.  count - number of entries to send/receive (must match on all ranks)
.  dtype - datatype to send/receive from each rank (must match on all ranks)
.  nto - number of ranks to send data to
.  toranks - ranks to send to (array of length nto)
-  todata - data to send to each rank (packed)

   Output Arguments:
+  nfrom - number of ranks receiving messages from
.  fromranks - ranks receiving messages from (length nfrom; caller should PetscFree())
-  fromdata - packed data from each rank, each with count entries of type dtype (length nfrom, caller responsible for PetscFree())

   Level: developer

   Options Database Keys:
.  -build_twosided <allreduce|ibarrier|redscatter> - algorithm to set up two-sided communication. Default is allreduce for communicators with <= 1024 ranks, otherwise ibarrier.

   Notes:
   This memory-scalable interface is an alternative to calling PetscGatherNumberOfMessages() and
   PetscGatherMessageLengths(), possibly with a subsequent round of communication to send other constant-size data.

   Basic data types as well as contiguous types are supported, but non-contiguous (e.g., strided) types are not.

   References:
.  1. - Hoefler, Siebert and Lumsdaine, The MPI_Ibarrier implementation uses the algorithm in
   Scalable communication protocols for dynamic sparse data exchange, 2010.

.seealso: PetscGatherNumberOfMessages(), PetscGatherMessageLengths()
@*/
PetscErrorCode PetscCommBuildTwoSided(MPI_Comm comm,PetscMPIInt count,MPI_Datatype dtype,PetscMPIInt nto,const PetscMPIInt *toranks,const void *todata,PetscMPIInt *nfrom,PetscMPIInt **fromranks,void *fromdata)
{
  PetscErrorCode         ierr;
  PetscBuildTwoSidedType buildtype = PETSC_BUILDTWOSIDED_NOTSET;

  PetscFunctionBegin;
  ierr = PetscSysInitializePackage();CHKERRQ(ierr);
  ierr = PetscLogEventSync(PETSC_BuildTwoSided,comm);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(PETSC_BuildTwoSided,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscCommBuildTwoSidedGetType(comm,&buildtype);CHKERRQ(ierr);
  switch (buildtype) {
  case PETSC_BUILDTWOSIDED_IBARRIER:
#if defined(PETSC_HAVE_MPI_IBARRIER) || defined(PETSC_HAVE_MPIX_IBARRIER)
    ierr = PetscCommBuildTwoSided_Ibarrier(comm,count,dtype,nto,toranks,todata,nfrom,fromranks,fromdata);CHKERRQ(ierr);
#else
    SETERRQ(comm,PETSC_ERR_PLIB,"MPI implementation does not provide MPI_Ibarrier (part of MPI-3)");
#endif
    break;
  case PETSC_BUILDTWOSIDED_ALLREDUCE:
    ierr = PetscCommBuildTwoSided_Allreduce(comm,count,dtype,nto,toranks,todata,nfrom,fromranks,fromdata);CHKERRQ(ierr);
    break;
  case PETSC_BUILDTWOSIDED_REDSCATTER:
#if defined(PETSC_HAVE_MPI_REDUCE_SCATTER_BLOCK)
    ierr = PetscCommBuildTwoSided_RedScatter(comm,count,dtype,nto,toranks,todata,nfrom,fromranks,fromdata);CHKERRQ(ierr);
#else
    SETERRQ(comm,PETSC_ERR_PLIB,"MPI implementation does not provide MPI_Reduce_scatter_block (part of MPI-2.2)");
#endif
    break;
  default: SETERRQ(comm,PETSC_ERR_PLIB,"Unknown method for building two-sided communication");
  }
  ierr = PetscLogEventEnd(PETSC_BuildTwoSided,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscCommBuildTwoSidedFReq_Reference(MPI_Comm comm,PetscMPIInt count,MPI_Datatype dtype,PetscMPIInt nto,const PetscMPIInt *toranks,const void *todata,
                                                           PetscMPIInt *nfrom,PetscMPIInt **fromranks,void *fromdata,PetscMPIInt ntags,MPI_Request **toreqs,MPI_Request **fromreqs,
                                                           PetscErrorCode (*send)(MPI_Comm,const PetscMPIInt[],PetscMPIInt,PetscMPIInt,void*,MPI_Request[],void*),
                                                           PetscErrorCode (*recv)(MPI_Comm,const PetscMPIInt[],PetscMPIInt,void*,MPI_Request[],void*),void *ctx)
{
  PetscErrorCode ierr;
  PetscMPIInt i,*tag;
  MPI_Aint    lb,unitbytes;
  MPI_Request *sendreq,*recvreq;

  PetscFunctionBegin;
  ierr = PetscMalloc1(ntags,&tag);CHKERRQ(ierr);
  if (ntags > 0) {
    ierr = PetscCommDuplicate(comm,&comm,&tag[0]);CHKERRQ(ierr);
  }
  for (i=1; i<ntags; i++) {
    ierr = PetscCommGetNewTag(comm,&tag[i]);CHKERRQ(ierr);
  }

  /* Perform complete initial rendezvous */
  ierr = PetscCommBuildTwoSided(comm,count,dtype,nto,toranks,todata,nfrom,fromranks,fromdata);CHKERRQ(ierr);

  ierr = PetscMalloc1(nto*ntags,&sendreq);CHKERRQ(ierr);
  ierr = PetscMalloc1(*nfrom*ntags,&recvreq);CHKERRQ(ierr);

  ierr = MPI_Type_get_extent(dtype,&lb,&unitbytes);CHKERRQ(ierr);
  if (lb != 0) SETERRQ1(comm,PETSC_ERR_SUP,"Datatype with nonzero lower bound %ld\n",(long)lb);
  for (i=0; i<nto; i++) {
    PetscMPIInt k;
    for (k=0; k<ntags; k++) sendreq[i*ntags+k] = MPI_REQUEST_NULL;
    ierr = (*send)(comm,tag,i,toranks[i],((char*)todata)+count*unitbytes*i,sendreq+i*ntags,ctx);CHKERRQ(ierr);
  }
  for (i=0; i<*nfrom; i++) {
    void *header = (*(char**)fromdata) + count*unitbytes*i;
    PetscMPIInt k;
    for (k=0; k<ntags; k++) recvreq[i*ntags+k] = MPI_REQUEST_NULL;
    ierr = (*recv)(comm,tag,(*fromranks)[i],header,recvreq+i*ntags,ctx);CHKERRQ(ierr);
  }
  ierr = PetscFree(tag);CHKERRQ(ierr);
  ierr = PetscCommDestroy(&comm);CHKERRQ(ierr);
  *toreqs = sendreq;
  *fromreqs = recvreq;
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_MPI_IBARRIER) || defined(PETSC_HAVE_MPIX_IBARRIER)

static PetscErrorCode PetscCommBuildTwoSidedFReq_Ibarrier(MPI_Comm comm,PetscMPIInt count,MPI_Datatype dtype,PetscMPIInt nto,const PetscMPIInt *toranks,const void *todata,
                                                          PetscMPIInt *nfrom,PetscMPIInt **fromranks,void *fromdata,PetscMPIInt ntags,MPI_Request **toreqs,MPI_Request **fromreqs,
                                                          PetscErrorCode (*send)(MPI_Comm,const PetscMPIInt[],PetscMPIInt,PetscMPIInt,void*,MPI_Request[],void*),
                                                          PetscErrorCode (*recv)(MPI_Comm,const PetscMPIInt[],PetscMPIInt,void*,MPI_Request[],void*),void *ctx)
{
  PetscErrorCode ierr;
  PetscMPIInt    nrecvs,tag,*tags,done,i;
  MPI_Aint       lb,unitbytes;
  char           *tdata;
  MPI_Request    *sendreqs,*usendreqs,*req,barrier;
  PetscSegBuffer segrank,segdata,segreq;
  PetscBool      barrier_started;

  PetscFunctionBegin;
  ierr = PetscCommDuplicate(comm,&comm,&tag);CHKERRQ(ierr);
  ierr = PetscMalloc1(ntags,&tags);CHKERRQ(ierr);
  for (i=0; i<ntags; i++) {
    ierr = PetscCommGetNewTag(comm,&tags[i]);CHKERRQ(ierr);
  }
  ierr = MPI_Type_get_extent(dtype,&lb,&unitbytes);CHKERRQ(ierr);
  if (lb != 0) SETERRQ1(comm,PETSC_ERR_SUP,"Datatype with nonzero lower bound %ld\n",(long)lb);
  tdata = (char*)todata;
  ierr = PetscMalloc1(nto,&sendreqs);CHKERRQ(ierr);
  ierr = PetscMalloc1(nto*ntags,&usendreqs);CHKERRQ(ierr);
  /* Post synchronous sends */
  for (i=0; i<nto; i++) {
    ierr = MPI_Issend((void*)(tdata+count*unitbytes*i),count,dtype,toranks[i],tag,comm,sendreqs+i);CHKERRQ(ierr);
  }
  /* Post actual payloads.  These are typically larger messages.  Hopefully sending these later does not slow down the
   * synchronous messages above. */
  for (i=0; i<nto; i++) {
    PetscMPIInt k;
    for (k=0; k<ntags; k++) usendreqs[i*ntags+k] = MPI_REQUEST_NULL;
    ierr = (*send)(comm,tags,i,toranks[i],tdata+count*unitbytes*i,usendreqs+i*ntags,ctx);CHKERRQ(ierr);
  }

  ierr = PetscSegBufferCreate(sizeof(PetscMPIInt),4,&segrank);CHKERRQ(ierr);
  ierr = PetscSegBufferCreate(unitbytes,4*count,&segdata);CHKERRQ(ierr);
  ierr = PetscSegBufferCreate(sizeof(MPI_Request),4,&segreq);CHKERRQ(ierr);

  nrecvs  = 0;
  barrier = MPI_REQUEST_NULL;
  /* MPICH-3.2 sometimes does not create a request in some "optimized" cases.  This is arguably a standard violation,
   * but we need to work around it. */
  barrier_started = PETSC_FALSE;
  for (done=0; !done; ) {
    PetscMPIInt flag;
    MPI_Status  status;
    ierr = MPI_Iprobe(MPI_ANY_SOURCE,tag,comm,&flag,&status);CHKERRQ(ierr);
    if (flag) {                 /* incoming message */
      PetscMPIInt *recvrank,k;
      void        *buf;
      ierr = PetscSegBufferGet(segrank,1,&recvrank);CHKERRQ(ierr);
      ierr = PetscSegBufferGet(segdata,count,&buf);CHKERRQ(ierr);
      *recvrank = status.MPI_SOURCE;
      ierr = MPI_Recv(buf,count,dtype,status.MPI_SOURCE,tag,comm,MPI_STATUS_IGNORE);CHKERRQ(ierr);
      ierr = PetscSegBufferGet(segreq,ntags,&req);CHKERRQ(ierr);
      for (k=0; k<ntags; k++) req[k] = MPI_REQUEST_NULL;
      ierr = (*recv)(comm,tags,status.MPI_SOURCE,buf,req,ctx);CHKERRQ(ierr);
      nrecvs++;
    }
    if (!barrier_started) {
      PetscMPIInt sent,nsends;
      ierr = PetscMPIIntCast(nto,&nsends);CHKERRQ(ierr);
      ierr = MPI_Testall(nsends,sendreqs,&sent,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
      if (sent) {
#if defined(PETSC_HAVE_MPI_IBARRIER)
        ierr = MPI_Ibarrier(comm,&barrier);CHKERRQ(ierr);
#elif defined(PETSC_HAVE_MPIX_IBARRIER)
        ierr = MPIX_Ibarrier(comm,&barrier);CHKERRQ(ierr);
#endif
        barrier_started = PETSC_TRUE;
      }
    } else {
      ierr = MPI_Test(&barrier,&done,MPI_STATUS_IGNORE);CHKERRQ(ierr);
    }
  }
  *nfrom = nrecvs;
  ierr = PetscSegBufferExtractAlloc(segrank,fromranks);CHKERRQ(ierr);
  ierr = PetscSegBufferDestroy(&segrank);CHKERRQ(ierr);
  ierr = PetscSegBufferExtractAlloc(segdata,fromdata);CHKERRQ(ierr);
  ierr = PetscSegBufferDestroy(&segdata);CHKERRQ(ierr);
  *toreqs = usendreqs;
  ierr = PetscSegBufferExtractAlloc(segreq,fromreqs);CHKERRQ(ierr);
  ierr = PetscSegBufferDestroy(&segreq);CHKERRQ(ierr);
  ierr = PetscFree(sendreqs);CHKERRQ(ierr);
  ierr = PetscFree(tags);CHKERRQ(ierr);
  ierr = PetscCommDestroy(&comm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

/*@C
   PetscCommBuildTwoSidedF - discovers communicating ranks given one-sided information, calling user-defined functions during rendezvous

   Collective

   Input Arguments:
+  comm - communicator
.  count - number of entries to send/receive in initial rendezvous (must match on all ranks)
.  dtype - datatype to send/receive from each rank (must match on all ranks)
.  nto - number of ranks to send data to
.  toranks - ranks to send to (array of length nto)
.  todata - data to send to each rank (packed)
.  ntags - number of tags needed by send/recv callbacks
.  send - callback invoked on sending process when ready to send primary payload
.  recv - callback invoked on receiving process after delivery of rendezvous message
-  ctx - context for callbacks

   Output Arguments:
+  nfrom - number of ranks receiving messages from
.  fromranks - ranks receiving messages from (length nfrom; caller should PetscFree())
-  fromdata - packed data from each rank, each with count entries of type dtype (length nfrom, caller responsible for PetscFree())

   Level: developer

   Notes:
   This memory-scalable interface is an alternative to calling PetscGatherNumberOfMessages() and
   PetscGatherMessageLengths(), possibly with a subsequent round of communication to send other data.

   Basic data types as well as contiguous types are supported, but non-contiguous (e.g., strided) types are not.

   References:
.  1. - Hoefler, Siebert and Lumsdaine, The MPI_Ibarrier implementation uses the algorithm in
   Scalable communication protocols for dynamic sparse data exchange, 2010.

.seealso: PetscCommBuildTwoSided(), PetscCommBuildTwoSidedFReq(), PetscGatherNumberOfMessages(), PetscGatherMessageLengths()
@*/
PetscErrorCode PetscCommBuildTwoSidedF(MPI_Comm comm,PetscMPIInt count,MPI_Datatype dtype,PetscMPIInt nto,const PetscMPIInt *toranks,const void *todata,PetscMPIInt *nfrom,PetscMPIInt **fromranks,void *fromdata,PetscMPIInt ntags,
                                       PetscErrorCode (*send)(MPI_Comm,const PetscMPIInt[],PetscMPIInt,PetscMPIInt,void*,MPI_Request[],void*),
                                       PetscErrorCode (*recv)(MPI_Comm,const PetscMPIInt[],PetscMPIInt,void*,MPI_Request[],void*),void *ctx)
{
  PetscErrorCode ierr;
  MPI_Request    *toreqs,*fromreqs;

  PetscFunctionBegin;
  ierr = PetscCommBuildTwoSidedFReq(comm,count,dtype,nto,toranks,todata,nfrom,fromranks,fromdata,ntags,&toreqs,&fromreqs,send,recv,ctx);CHKERRQ(ierr);
  ierr = MPI_Waitall(nto*ntags,toreqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
  ierr = MPI_Waitall(*nfrom*ntags,fromreqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
  ierr = PetscFree(toreqs);CHKERRQ(ierr);
  ierr = PetscFree(fromreqs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscCommBuildTwoSidedFReq - discovers communicating ranks given one-sided information, calling user-defined functions during rendezvous, returns requests

   Collective

   Input Arguments:
+  comm - communicator
.  count - number of entries to send/receive in initial rendezvous (must match on all ranks)
.  dtype - datatype to send/receive from each rank (must match on all ranks)
.  nto - number of ranks to send data to
.  toranks - ranks to send to (array of length nto)
.  todata - data to send to each rank (packed)
.  ntags - number of tags needed by send/recv callbacks
.  send - callback invoked on sending process when ready to send primary payload
.  recv - callback invoked on receiving process after delivery of rendezvous message
-  ctx - context for callbacks

   Output Arguments:
+  nfrom - number of ranks receiving messages from
.  fromranks - ranks receiving messages from (length nfrom; caller should PetscFree())
.  fromdata - packed data from each rank, each with count entries of type dtype (length nfrom, caller responsible for PetscFree())
.  toreqs - array of nto*ntags sender requests (caller must wait on these, then PetscFree())
-  fromreqs - array of nfrom*ntags receiver requests (caller must wait on these, then PetscFree())

   Level: developer

   Notes:
   This memory-scalable interface is an alternative to calling PetscGatherNumberOfMessages() and
   PetscGatherMessageLengths(), possibly with a subsequent round of communication to send other data.

   Basic data types as well as contiguous types are supported, but non-contiguous (e.g., strided) types are not.

   References:
.  1. - Hoefler, Siebert and Lumsdaine, The MPI_Ibarrier implementation uses the algorithm in
   Scalable communication protocols for dynamic sparse data exchange, 2010.

.seealso: PetscCommBuildTwoSided(), PetscCommBuildTwoSidedF(), PetscGatherNumberOfMessages(), PetscGatherMessageLengths()
@*/
PetscErrorCode PetscCommBuildTwoSidedFReq(MPI_Comm comm,PetscMPIInt count,MPI_Datatype dtype,PetscMPIInt nto,const PetscMPIInt *toranks,const void *todata,
                                          PetscMPIInt *nfrom,PetscMPIInt **fromranks,void *fromdata,PetscMPIInt ntags,MPI_Request **toreqs,MPI_Request **fromreqs,
                                          PetscErrorCode (*send)(MPI_Comm,const PetscMPIInt[],PetscMPIInt,PetscMPIInt,void*,MPI_Request[],void*),
                                          PetscErrorCode (*recv)(MPI_Comm,const PetscMPIInt[],PetscMPIInt,void*,MPI_Request[],void*),void *ctx)
{
  PetscErrorCode         ierr,(*f)(MPI_Comm,PetscMPIInt,MPI_Datatype,PetscMPIInt,const PetscMPIInt[],const void*,
                                   PetscMPIInt*,PetscMPIInt**,void*,PetscMPIInt,MPI_Request**,MPI_Request**,
                                   PetscErrorCode (*send)(MPI_Comm,const PetscMPIInt[],PetscMPIInt,PetscMPIInt,void*,MPI_Request[],void*),
                                   PetscErrorCode (*recv)(MPI_Comm,const PetscMPIInt[],PetscMPIInt,void*,MPI_Request[],void*),void *ctx);
  PetscBuildTwoSidedType buildtype = PETSC_BUILDTWOSIDED_NOTSET;
  PetscMPIInt i,size;

  PetscFunctionBegin;
  ierr = PetscSysInitializePackage();CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  for (i=0; i<nto; i++) {
    if (toranks[i] < 0 || size <= toranks[i]) SETERRQ3(comm,PETSC_ERR_ARG_OUTOFRANGE,"toranks[%d] %d not in comm size %d",i,toranks[i],size);
  }
  ierr = PetscLogEventSync(PETSC_BuildTwoSidedF,comm);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(PETSC_BuildTwoSidedF,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscCommBuildTwoSidedGetType(comm,&buildtype);CHKERRQ(ierr);
  switch (buildtype) {
  case PETSC_BUILDTWOSIDED_IBARRIER:
#if defined(PETSC_HAVE_MPI_IBARRIER) || defined(PETSC_HAVE_MPIX_IBARRIER)
    f = PetscCommBuildTwoSidedFReq_Ibarrier;
#else
    SETERRQ(comm,PETSC_ERR_PLIB,"MPI implementation does not provide MPI_Ibarrier (part of MPI-3)");
#endif
    break;
  case PETSC_BUILDTWOSIDED_ALLREDUCE:
  case PETSC_BUILDTWOSIDED_REDSCATTER:
    f = PetscCommBuildTwoSidedFReq_Reference;
    break;
  default: SETERRQ(comm,PETSC_ERR_PLIB,"Unknown method for building two-sided communication");
  }
  ierr = (*f)(comm,count,dtype,nto,toranks,todata,nfrom,fromranks,fromdata,ntags,toreqs,fromreqs,send,recv,ctx);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PETSC_BuildTwoSidedF,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
