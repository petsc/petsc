#include <petscsys.h>        /*I  "petscsys.h"  I*/

const char *const PetscBuildTwoSidedTypes[] = {
  "ALLREDUCE",
  "IBARRIER",
  "PetscBuildTwoSidedType",
  "PETSC_BUILDTWOSIDED_",
  0
};

static PetscBuildTwoSidedType _twosided_type = PETSC_BUILDTWOSIDED_NOTSET;

#undef __FUNCT__
#define __FUNCT__ "PetscCommBuildTwoSidedSetType"
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
#if defined(PETSC_USE_DEBUG)
  {                             /* We don't have a PetscObject so can't use PetscValidLogicalCollectiveEnum */
    PetscMPIInt ierr;
    PetscMPIInt b1[2],b2[2];
    b1[0] = -(PetscMPIInt)twosided;
    b1[1] = (PetscMPIInt)twosided;
    ierr  = MPI_Allreduce(b1,b2,2,MPI_INT,MPI_MAX,comm);CHKERRQ(ierr);
    if (-b2[0] != b2[1]) SETERRQ(comm,PETSC_ERR_ARG_WRONG,"Enum value must be same on all processes");
  }
#endif
  _twosided_type = twosided;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscCommBuildTwoSidedGetType"
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

  PetscFunctionBegin;
  *twosided = PETSC_BUILDTWOSIDED_NOTSET;
  if (_twosided_type == PETSC_BUILDTWOSIDED_NOTSET) {
#if defined(PETSC_HAVE_MPI_IBARRIER)
#  if defined(PETSC_HAVE_MPICH_CH3_SOCK) && !defined(PETSC_HAVE_MPICH_CH3_SOCK_FIXED_NBC_PROGRESS)
    /* Deadlock in Ibarrier: http://trac.mpich.org/projects/mpich/ticket/1785 */
    _twosided_type = PETSC_BUILDTWOSIDED_ALLREDUCE;
#  else
    _twosided_type = PETSC_BUILDTWOSIDED_IBARRIER;
#  endif
#else
    _twosided_type = PETSC_BUILDTWOSIDED_ALLREDUCE;
#endif
    ierr = PetscOptionsGetEnum(NULL,"-build_twosided",PetscBuildTwoSidedTypes,(PetscEnum*)&_twosided_type,NULL);CHKERRQ(ierr);
  }
  *twosided = _twosided_type;
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_MPI_IBARRIER)

#undef __FUNCT__
#define __FUNCT__ "PetscCommBuildTwoSided_Ibarrier"
static PetscErrorCode PetscCommBuildTwoSided_Ibarrier(MPI_Comm comm,PetscMPIInt count,MPI_Datatype dtype,PetscInt nto,const PetscMPIInt *toranks,const void *todata,PetscInt *nfrom,PetscMPIInt **fromranks,void *fromdata)
{
  PetscErrorCode ierr;
  PetscMPIInt    nrecvs,tag,unitbytes,done;
  PetscInt       i;
  char           *tdata;
  MPI_Request    *sendreqs,barrier;
  PetscSegBuffer segrank,segdata;

  PetscFunctionBegin;
  ierr  = PetscCommGetNewTag(comm,&tag);CHKERRQ(ierr);
  ierr  = MPI_Type_size(dtype,&unitbytes);CHKERRQ(ierr);
  tdata = (char*)todata;
  ierr  = PetscMalloc(nto*sizeof(MPI_Request),&sendreqs);CHKERRQ(ierr);
  for (i=0; i<nto; i++) {
    ierr = MPI_Issend((void*)(tdata+count*unitbytes*i),count,dtype,toranks[i],tag,comm,sendreqs+i);CHKERRQ(ierr);
  }
  ierr = PetscSegBufferCreate(sizeof(PetscMPIInt),4,&segrank);CHKERRQ(ierr);
  ierr = PetscSegBufferCreate(unitbytes,4*count,&segdata);CHKERRQ(ierr);

  nrecvs  = 0;
  barrier = MPI_REQUEST_NULL;
  for (done=0; !done; ) {
    PetscMPIInt flag;
    MPI_Status  status;
    ierr = MPI_Iprobe(MPI_ANY_SOURCE,tag,comm,&flag,&status);CHKERRQ(ierr);
    if (flag) {                 /* incoming message */
      PetscMPIInt *recvrank;
      void        *buf;
      ierr      = PetscSegBufferGet(&segrank,1,&recvrank);CHKERRQ(ierr);
      ierr      = PetscSegBufferGet(&segdata,count,&buf);CHKERRQ(ierr);
      *recvrank = status.MPI_SOURCE;
      ierr      = MPI_Recv(buf,count,dtype,status.MPI_SOURCE,tag,comm,MPI_STATUS_IGNORE);CHKERRQ(ierr);
      nrecvs++;
    }
    if (barrier == MPI_REQUEST_NULL) {
      PetscMPIInt sent,nsends;
      ierr = PetscMPIIntCast(nto,&nsends);CHKERRQ(ierr);
      ierr = MPI_Testall(nsends,sendreqs,&sent,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
      if (sent) {
        ierr = MPI_Ibarrier(comm,&barrier);CHKERRQ(ierr);
        ierr = PetscFree(sendreqs);CHKERRQ(ierr);
      }
    } else {
      ierr = MPI_Test(&barrier,&done,MPI_STATUS_IGNORE);CHKERRQ(ierr);
    }
  }
  *nfrom = nrecvs;
  ierr   = PetscSegBufferExtractAlloc(&segrank,fromranks);CHKERRQ(ierr);
  ierr   = PetscSegBufferDestroy(&segrank);CHKERRQ(ierr);
  ierr   = PetscSegBufferExtractAlloc(&segdata,fromdata);CHKERRQ(ierr);
  ierr   = PetscSegBufferDestroy(&segdata);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#undef __FUNCT__
#define __FUNCT__ "PetscCommBuildTwoSided_Allreduce"
static PetscErrorCode PetscCommBuildTwoSided_Allreduce(MPI_Comm comm,PetscMPIInt count,MPI_Datatype dtype,PetscInt nto,const PetscMPIInt *toranks,const void *todata,PetscInt *nfrom,PetscMPIInt **fromranks,void *fromdata)
{
  PetscErrorCode ierr;
  PetscMPIInt    size,*iflags,nrecvs,tag,unitbytes,*franks;
  PetscInt       i;
  char           *tdata,*fdata;
  MPI_Request    *reqs,*sendreqs;
  MPI_Status     *statuses;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = PetscMalloc(size*sizeof(*iflags),&iflags);CHKERRQ(ierr);
  ierr = PetscMemzero(iflags,size*sizeof(*iflags));CHKERRQ(ierr);
  for (i=0; i<nto; i++) iflags[toranks[i]] = 1;
  ierr = PetscGatherNumberOfMessages(comm,iflags,NULL,&nrecvs);CHKERRQ(ierr);
  ierr = PetscFree(iflags);CHKERRQ(ierr);

  ierr     = PetscCommGetNewTag(comm,&tag);CHKERRQ(ierr);
  ierr     = MPI_Type_size(dtype,&unitbytes);CHKERRQ(ierr);
  ierr     = PetscMalloc(nrecvs*count*unitbytes,&fdata);CHKERRQ(ierr);
  tdata    = (char*)todata;
  ierr     = PetscMalloc2(nto+nrecvs,MPI_Request,&reqs,nto+nrecvs,MPI_Status,&statuses);CHKERRQ(ierr);
  sendreqs = reqs + nrecvs;
  for (i=0; i<nrecvs; i++) {
    ierr = MPI_Irecv((void*)(fdata+count*unitbytes*i),count,dtype,MPI_ANY_SOURCE,tag,comm,reqs+i);CHKERRQ(ierr);
  }
  for (i=0; i<nto; i++) {
    ierr = MPI_Isend((void*)(tdata+count*unitbytes*i),count,dtype,toranks[i],tag,comm,sendreqs+i);CHKERRQ(ierr);
  }
  ierr = MPI_Waitall(nto+nrecvs,reqs,statuses);CHKERRQ(ierr);
  ierr = PetscMalloc(nrecvs*sizeof(PetscMPIInt),&franks);CHKERRQ(ierr);
  for (i=0; i<nrecvs; i++) franks[i] = statuses[i].MPI_SOURCE;
  ierr = PetscFree2(reqs,statuses);CHKERRQ(ierr);

  *nfrom            = nrecvs;
  *fromranks        = franks;
  *(void**)fromdata = fdata;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscCommBuildTwoSided"
/*@C
   PetscCommBuildTwoSided - discovers communicating ranks given one-sided information, moving constant-sized data in the process (often message lengths)

   Collective on MPI_Comm

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

   Notes:
   This memory-scalable interface is an alternative to calling PetscGatherNumberOfMessages() and
   PetscGatherMessageLengths(), possibly with a subsequent round of communication to send other constant-size data.

   Basic data types as well as contiguous types are supported, but non-contiguous (e.g., strided) types are not.

   References:
   The MPI_Ibarrier implementation uses the algorithm in
   Hoefler, Siebert and Lumsdaine, Scalable communication protocols for dynamic sparse data exchange, 2010.

.seealso: PetscGatherNumberOfMessages(), PetscGatherMessageLengths()
@*/
PetscErrorCode PetscCommBuildTwoSided(MPI_Comm comm,PetscMPIInt count,MPI_Datatype dtype,PetscInt nto,const PetscMPIInt *toranks,const void *todata,PetscInt *nfrom,PetscMPIInt **fromranks,void *fromdata)
{
  PetscErrorCode         ierr;
  PetscBuildTwoSidedType buildtype = PETSC_BUILDTWOSIDED_NOTSET;

  PetscFunctionBegin;
  ierr = PetscCommBuildTwoSidedGetType(comm,&buildtype);CHKERRQ(ierr);
  switch (buildtype) {
  case PETSC_BUILDTWOSIDED_IBARRIER:
#if defined(PETSC_HAVE_MPI_IBARRIER)
    ierr = PetscCommBuildTwoSided_Ibarrier(comm,count,dtype,nto,toranks,todata,nfrom,fromranks,fromdata);CHKERRQ(ierr);
#else
    SETERRQ(comm,PETSC_ERR_PLIB,"MPI implementation does not provide MPI_Ibarrier (part of MPI-3)");
#endif
    break;
  case PETSC_BUILDTWOSIDED_ALLREDUCE:
    ierr = PetscCommBuildTwoSided_Allreduce(comm,count,dtype,nto,toranks,todata,nfrom,fromranks,fromdata);CHKERRQ(ierr);
    break;
  default: SETERRQ(comm,PETSC_ERR_PLIB,"Unknown method for building two-sided communication");
  }
  PetscFunctionReturn(0);
}
