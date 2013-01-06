#include <petscsys.h>        /*I  "petscsys.h"  I*/
#include <stddef.h>

typedef enum {BUILDTWOSIDED_NOTSET = -1,
#if defined(PETSC_HAVE_MPI_IBARRIER)
              BUILDTWOSIDED_IBARRIER,
#endif
              BUILDTWOSIDED_ALLREDUCE} BuildTwoSidedType;

static const char *const BuildTwoSidedTypes[] = {
#if defined(PETSC_HAVE_MPI_IBARRIER)
  "IBARRIER",
#endif
  "ALLREDUCE",
  "BuildTwoSidedType",
  "BUILDTWOSIDED_",
  0
};

static BuildTwoSidedType _twosided_type = BUILDTWOSIDED_NOTSET;

#undef __FUNCT__
#define __FUNCT__ "PetscBuildTwoSidedGetType"
static PetscErrorCode PetscBuildTwoSidedGetType(BuildTwoSidedType *twosided)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (_twosided_type == BUILDTWOSIDED_NOTSET) {
#if defined(PETSC_HAVE_MPI_IBARRIER)
    _twosided_type = BUILDTWOSIDED_IBARRIER;
#else
    _twosided_type = BUILDTWOSIDED_ALLREDUCE;
#endif
    ierr = PetscOptionsGetEnum(PETSC_NULL,"-build_twosided",BuildTwoSidedTypes,(PetscEnum*)&_twosided_type,PETSC_NULL);CHKERRQ(ierr);
  }
  *twosided = _twosided_type;
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_MPI_IBARRIER)
/* Segmented (extendable) array implementation */
typedef struct _SegArray *SegArray;
struct _SegArray {
  PetscInt unitbytes;
  PetscInt alloc;
  PetscInt used;
  PetscInt tailused;
  SegArray tail;
  union {                       /* Dummy types to ensure alignment */
    PetscReal dummy_real;
    PetscInt dummy_int;
    char array[1];
  } u;
};

#undef __FUNCT__
#define __FUNCT__ "SegArrayCreate"
static PetscErrorCode SegArrayCreate(PetscInt unitbytes,PetscInt expected,SegArray *seg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc(offsetof(struct _SegArray,u)+expected*unitbytes,seg);CHKERRQ(ierr);
  ierr = PetscMemzero(*seg,offsetof(struct _SegArray,u));CHKERRQ(ierr);
  (*seg)->unitbytes = unitbytes;
  (*seg)->alloc = expected;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SegArrayAlloc_Private"
static PetscErrorCode SegArrayAlloc_Private(SegArray *seg,PetscInt count)
{
  PetscErrorCode ierr;
  SegArray newseg,s;
  PetscInt alloc;

  PetscFunctionBegin;
  s = *seg;
  /* Grow at least fast enough to hold next item, like Fibonacci otherwise (up to 1MB chunks) */
  alloc = PetscMax(s->used+count,PetscMin(1000000/s->unitbytes+1,s->alloc+s->tailused));
  ierr = PetscMalloc(offsetof(struct _SegArray,u)+alloc*s->unitbytes,&newseg);CHKERRQ(ierr);
  ierr = PetscMemzero(newseg,offsetof(struct _SegArray,u));CHKERRQ(ierr);
  newseg->unitbytes = s->unitbytes;
  newseg->tailused = s->used + s->tailused;
  newseg->tail = s;
  newseg->alloc = alloc;
  *seg = newseg;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SegArrayGet"
static PetscErrorCode SegArrayGet(SegArray *seg,PetscInt count,void *array)
{
  PetscErrorCode ierr;
  SegArray s;

  PetscFunctionBegin;
  s = *seg;
  if (PetscUnlikely(s->used + count > s->alloc)) {ierr = SegArrayAlloc_Private(seg,count);CHKERRQ(ierr);}
  s = *seg;
  *(char**)array = &s->u.array[s->used*s->unitbytes];
  s->used += count;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SegArrayDestroy"
static PetscErrorCode SegArrayDestroy(SegArray *seg)
{
  PetscErrorCode ierr;
  SegArray s;

  PetscFunctionBegin;
  for (s=*seg; s; ) {
    SegArray tail = s->tail;
    ierr = PetscFree(s);CHKERRQ(ierr);
    s = tail;
  }
  *seg = PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SegArrayExtract"
/* Extracts contiguous data and resets segarray */
static PetscErrorCode SegArrayExtract(SegArray *seg,void *contiguous)
{
  PetscErrorCode ierr;
  PetscInt unitbytes;
  SegArray s,t;
  char *contig,*ptr;

  PetscFunctionBegin;
  s = *seg;
  unitbytes = s->unitbytes;
  ierr = PetscMalloc((s->used+s->tailused)*unitbytes,&contig);CHKERRQ(ierr);
  ptr = contig + s->tailused*unitbytes;
  ierr = PetscMemcpy(ptr,s->u.array,s->used*unitbytes);CHKERRQ(ierr);
  for (t=s->tail; t; ) {
    SegArray tail = t->tail;
    ptr -= t->used*unitbytes;
    ierr = PetscMemcpy(ptr,t->u.array,t->used*unitbytes);CHKERRQ(ierr);
    ierr = PetscFree(t);CHKERRQ(ierr);
    t = tail;
  }
  if (ptr != contig) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Tail count does not match");
  s->tailused = 0;
  s->tail = PETSC_NULL;
  *(char**)contiguous = contig;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "BuildTwoSided_Ibarrier"
static PetscErrorCode BuildTwoSided_Ibarrier(MPI_Comm comm,PetscMPIInt count,MPI_Datatype dtype,PetscInt nto,const PetscMPIInt *toranks,const void *todata,PetscInt *nfrom,PetscMPIInt **fromranks,void *fromdata)
{
  PetscErrorCode ierr;
  PetscMPIInt    nrecvs,tag,unitbytes,done;
  PetscInt       i;
  char           *tdata;
  MPI_Request    *sendreqs,barrier;
  SegArray       segrank,segdata;

  PetscFunctionBegin;
  ierr = PetscCommGetNewTag(comm,&tag);CHKERRQ(ierr);
  ierr = MPI_Type_size(dtype,&unitbytes);CHKERRQ(ierr);
  tdata = (char*)todata;
  ierr = PetscMalloc(nto*sizeof(MPI_Request),&sendreqs);CHKERRQ(ierr);
  for (i=0; i<nto; i++) {
    ierr = MPI_Issend((void*)(tdata+count*unitbytes*i),count,dtype,toranks[i],tag,comm,sendreqs+i);CHKERRQ(ierr);
  }
  ierr = SegArrayCreate(sizeof(PetscMPIInt),4,&segrank);CHKERRQ(ierr);
  ierr = SegArrayCreate(unitbytes,4*count,&segdata);CHKERRQ(ierr);

  nrecvs = 0;
  barrier = MPI_REQUEST_NULL;
  for (done=0; !done; ) {
    PetscMPIInt flag;
    MPI_Status status;
    ierr = MPI_Iprobe(MPI_ANY_SOURCE,tag,comm,&flag,&status);CHKERRQ(ierr);
    if (flag) {                 /* incoming message */
      PetscMPIInt *recvrank;
      void *buf;
      ierr = SegArrayGet(&segrank,1,&recvrank);CHKERRQ(ierr);
      ierr = SegArrayGet(&segdata,count,&buf);CHKERRQ(ierr);
      *recvrank = status.MPI_SOURCE;
      ierr = MPI_Recv(buf,count,dtype,status.MPI_SOURCE,tag,comm,MPI_STATUS_IGNORE);CHKERRQ(ierr);
      nrecvs++;
    }
    if (barrier == MPI_REQUEST_NULL) {
      PetscMPIInt sent,nsends = PetscMPIIntCast(nto);
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
  ierr = SegArrayExtract(&segrank,fromranks);CHKERRQ(ierr);
  ierr = SegArrayDestroy(&segrank);CHKERRQ(ierr);
  ierr = SegArrayExtract(&segdata,fromdata);CHKERRQ(ierr);
  ierr = SegArrayDestroy(&segdata);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#undef __FUNCT__
#define __FUNCT__ "BuildTwoSided_Allreduce"
static PetscErrorCode BuildTwoSided_Allreduce(MPI_Comm comm,PetscMPIInt count,MPI_Datatype dtype,PetscInt nto,const PetscMPIInt *toranks,const void *todata,PetscInt *nfrom,PetscMPIInt **fromranks,void *fromdata)
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
  ierr = PetscGatherNumberOfMessages(comm,iflags,PETSC_NULL,&nrecvs);CHKERRQ(ierr);
  ierr = PetscFree(iflags);CHKERRQ(ierr);

  ierr = PetscCommGetNewTag(comm,&tag);CHKERRQ(ierr);
  ierr = MPI_Type_size(dtype,&unitbytes);CHKERRQ(ierr);
  ierr = PetscMalloc(nrecvs*count*unitbytes,&fdata);CHKERRQ(ierr);
  tdata = (char*)todata;
  ierr = PetscMalloc2(nto+nrecvs,MPI_Request,&reqs,nto+nrecvs,MPI_Status,&statuses);CHKERRQ(ierr);
  sendreqs = reqs + nrecvs;
  for (i=0; i<nrecvs; i++) {
    ierr = MPI_Irecv((void*)(fdata+count*unitbytes*i),count,dtype,MPI_ANY_SOURCE,tag,comm,reqs+i);CHKERRQ(ierr);
  }
  for (i=0; i<nto; i++) {
    ierr = MPI_Isend((void*)(tdata+count*unitbytes*i),count,dtype,toranks[i],tag,comm,sendreqs+i);CHKERRQ(ierr);
  }
  ierr = MPI_Waitall(nto+nrecvs,reqs,statuses);CHKERRQ(ierr);
  ierr = PetscMalloc(nrecvs*sizeof(PetscMPIInt),&franks);CHKERRQ(ierr);
  for (i=0; i<nrecvs; i++) {
    franks[i] = statuses[i].MPI_SOURCE;
  }
  ierr = PetscFree2(reqs,statuses);CHKERRQ(ierr);

  *nfrom = nrecvs;
  *fromranks = franks;
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
  PetscErrorCode ierr;
  BuildTwoSidedType buildtype;

  PetscFunctionBegin;
  ierr = PetscBuildTwoSidedGetType(&buildtype);CHKERRQ(ierr);
  switch (buildtype) {
#if defined(PETSC_HAVE_MPI_IBARRIER)
  case BUILDTWOSIDED_IBARRIER:
    ierr = BuildTwoSided_Ibarrier(comm,count,dtype,nto,toranks,todata,nfrom,fromranks,fromdata);CHKERRQ(ierr);
    break;
#endif
  case BUILDTWOSIDED_ALLREDUCE:
    ierr = BuildTwoSided_Allreduce(comm,count,dtype,nto,toranks,todata,nfrom,fromranks,fromdata);CHKERRQ(ierr);
    break;
  default: SETERRQ(comm,PETSC_ERR_PLIB,"Unknown method for building two-sided communication");
  }
  PetscFunctionReturn(0);
}
