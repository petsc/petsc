#if !defined(PETSCSFIMPL_H)
#define PETSCSFIMPL_H

#include <petscsf.h>
#include <petsc/private/petscimpl.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_CUDA)
#include <../src/vec/vec/impls/seq/seqcuda/cudavecimpl.h>
#endif

PETSC_EXTERN PetscLogEvent PETSCSF_SetGraph;
PETSC_EXTERN PetscLogEvent PETSCSF_SetUp;
PETSC_EXTERN PetscLogEvent PETSCSF_BcastBegin;
PETSC_EXTERN PetscLogEvent PETSCSF_BcastEnd;
PETSC_EXTERN PetscLogEvent PETSCSF_BcastAndOpBegin;
PETSC_EXTERN PetscLogEvent PETSCSF_BcastAndOpEnd;
PETSC_EXTERN PetscLogEvent PETSCSF_ReduceBegin;
PETSC_EXTERN PetscLogEvent PETSCSF_ReduceEnd;
PETSC_EXTERN PetscLogEvent PETSCSF_FetchAndOpBegin;
PETSC_EXTERN PetscLogEvent PETSCSF_FetchAndOpEnd;
PETSC_EXTERN PetscLogEvent PETSCSF_EmbedSF;
PETSC_EXTERN PetscLogEvent PETSCSF_DistSect;
PETSC_EXTERN PetscLogEvent PETSCSF_SectSF;
PETSC_EXTERN PetscLogEvent PETSCSF_RemoteOff;

typedef enum {PETSC_MEMTYPE_HOST=0, PETSC_MEMTYPE_DEVICE} PetscMemType;

struct _PetscSFOps {
  PetscErrorCode (*Reset)(PetscSF);
  PetscErrorCode (*Destroy)(PetscSF);
  PetscErrorCode (*SetUp)(PetscSF);
  PetscErrorCode (*SetFromOptions)(PetscOptionItems*,PetscSF);
  PetscErrorCode (*View)(PetscSF,PetscViewer);
  PetscErrorCode (*Duplicate)(PetscSF,PetscSFDuplicateOption,PetscSF);
  PetscErrorCode (*BcastAndOpBegin)(PetscSF,MPI_Datatype,PetscMemType,const void*,PetscMemType,      void*,      MPI_Op);
  PetscErrorCode (*BcastAndOpEnd)  (PetscSF,MPI_Datatype,PetscMemType,const void*,PetscMemType,      void*,      MPI_Op);
  PetscErrorCode (*ReduceBegin)    (PetscSF,MPI_Datatype,PetscMemType,const void*,PetscMemType,      void*,      MPI_Op);
  PetscErrorCode (*ReduceEnd)      (PetscSF,MPI_Datatype,PetscMemType,const void*,PetscMemType,      void*,      MPI_Op);
  PetscErrorCode (*FetchAndOpBegin)(PetscSF,MPI_Datatype,PetscMemType,      void*,PetscMemType,const void*,void*,MPI_Op);
  PetscErrorCode (*FetchAndOpEnd)  (PetscSF,MPI_Datatype,PetscMemType,      void*,PetscMemType,const void*,void*,MPI_Op);
  PetscErrorCode (*BcastToZero)    (PetscSF,MPI_Datatype,PetscMemType,const void*,PetscMemType,      void*); /* For interal use only */
  PetscErrorCode (*GetRootRanks)(PetscSF,PetscInt*,const PetscMPIInt**,const PetscInt**,const PetscInt**,const PetscInt**);
  PetscErrorCode (*GetLeafRanks)(PetscSF,PetscInt*,const PetscMPIInt**,const PetscInt**,const PetscInt**);
  PetscErrorCode (*CreateLocalSF)(PetscSF,PetscSF*);
  PetscErrorCode (*GetGraph)(PetscSF,PetscInt*,PetscInt*,const PetscInt**,const PetscSFNode**);
  PetscErrorCode (*CreateEmbeddedSF)(PetscSF,PetscInt,const PetscInt*,PetscSF*);
  PetscErrorCode (*CreateEmbeddedLeafSF)(PetscSF,PetscInt,const PetscInt*,PetscSF*);
};

typedef struct _n_PetscSFPackOpt *PetscSFPackOpt;

struct _p_PetscSF {
  PETSCHEADER(struct _PetscSFOps);
  PetscInt        nroots;          /* Number of root vertices on current process (candidates for incoming edges) */
  PetscInt        nleaves;         /* Number of leaf vertices on current process (this process specifies a root for each leaf) */
  PetscInt        *mine;           /* Location of leaves in leafdata arrays provided to the communication routines */
  PetscInt        *mine_alloc;
  PetscInt        minleaf,maxleaf;
  PetscSFNode     *remote;         /* Remote references to roots for each local leaf */
  PetscSFNode     *remote_alloc;
  PetscInt        nranks;          /* Number of ranks owning roots connected to my leaves */
  PetscInt        ndranks;         /* Number of ranks in distinguished group holding roots connected to my leaves */
  PetscMPIInt     *ranks;          /* List of ranks referenced by "remote" */
  PetscInt        *roffset;        /* Array of length nranks+1, offset in rmine/rremote for each rank */
  PetscInt        *rmine;          /* Concatenated array holding local indices referencing each remote rank */
  PetscInt        *rremote;        /* Concatenated array holding remote indices referenced for each remote rank */
  PetscBool       degreeknown;     /* The degree is currently known, do not have to recompute */
  PetscInt        *degree;         /* Degree of each of my root vertices */
  PetscInt        *degreetmp;      /* Temporary local array for computing degree */
  PetscBool       rankorder;       /* Sort ranks for gather and scatter operations */
  MPI_Group       ingroup;         /* Group of processes connected to my roots */
  MPI_Group       outgroup;        /* Group of processes connected to my leaves */
  PetscSF         multi;           /* Internal graph used to implement gather and scatter operations */
  PetscBool       graphset;        /* Flag indicating that the graph has been set, required before calling communication routines */
  PetscBool       setupcalled;     /* Type and communication structures have been set up */
  PetscSFPackOpt  leafpackopt;     /* Optimization plans to (un)pack leaves connected to remote roots, based on index patterns in rmine[]. NULL for no optimization */
  PetscSFPackOpt  selfleafpackopt; /* Optimization plans to (un)pack leaves connected to local roots */
  PetscBool       selfleafdups;    /* Indices of leaves in rmine[0,roffset[ndranks]) have dups, implying theads working ... */
                                   /* ... on these leaves in parallel may have data race. */
  PetscBool       remoteleafdups;  /* Indices of leaves in rmine[roffset[ndranks],roffset[nranks]) have dups */

  PetscSFPattern  pattern;         /* Pattern of the graph */
  PetscLayout     map;             /* Layout of leaves over all processes when building a patterned graph */
#if defined(PETSC_HAVE_CUDA)
  PetscInt        *rmine_d;        /* A copy of rmine in device memory */
  PetscInt        MAX_CORESIDENT_THREADS;
#endif
  void *data;                      /* Pointer to implementation */
};

PETSC_EXTERN PetscBool PetscSFRegisterAllCalled;
PETSC_EXTERN PetscErrorCode PetscSFRegisterAll(void);

PETSC_INTERN PetscErrorCode PetscSFCreateLocalSF_Private(PetscSF,PetscSF*);
PETSC_INTERN PetscErrorCode PetscSFBcastToZero_Private(PetscSF,MPI_Datatype,const void*,void*);

PETSC_EXTERN PetscErrorCode MPIPetsc_Type_unwrap(MPI_Datatype,MPI_Datatype*,PetscBool*);
PETSC_EXTERN PetscErrorCode MPIPetsc_Type_compare(MPI_Datatype,MPI_Datatype,PetscBool*);
PETSC_EXTERN PetscErrorCode MPIPetsc_Type_compare_contig(MPI_Datatype,MPI_Datatype,PetscInt*);

#if defined(PETSC_HAVE_MPI_NONBLOCKING_COLLECTIVES)
#define MPIU_Iscatter(a,b,c,d,e,f,g,h,req)     MPI_Iscatter(a,b,c,d,e,f,g,h,req)
#define MPIU_Iscatterv(a,b,c,d,e,f,g,h,i,req)  MPI_Iscatterv(a,b,c,d,e,f,g,h,i,req)
#define MPIU_Igather(a,b,c,d,e,f,g,h,req)      MPI_Igather(a,b,c,d,e,f,g,h,req)
#define MPIU_Igatherv(a,b,c,d,e,f,g,h,i,req)   MPI_Igatherv(a,b,c,d,e,f,g,h,i,req)
#define MPIU_Iallgather(a,b,c,d,e,f,g,req)     MPI_Iallgather(a,b,c,d,e,f,g,req)
#define MPIU_Iallgatherv(a,b,c,d,e,f,g,h,req)  MPI_Iallgatherv(a,b,c,d,e,f,g,h,req)
#define MPIU_Ialltoall(a,b,c,d,e,f,g,req)      MPI_Ialltoall(a,b,c,d,e,f,g,req)
#else
/* Ignore req, the MPI_Request argument, and use MPI blocking collectives. One should initialize req
   to MPI_REQUEST_NULL so that one can do MPI_Wait(req,status) no matter the call is blocking or not.
 */
#define MPIU_Iscatter(a,b,c,d,e,f,g,h,req)     MPI_Scatter(a,b,c,d,e,f,g,h)
#define MPIU_Iscatterv(a,b,c,d,e,f,g,h,i,req)  MPI_Scatterv(a,b,c,d,e,f,g,h,i)
#define MPIU_Igather(a,b,c,d,e,f,g,h,req)      MPI_Gather(a,b,c,d,e,f,g,h)
#define MPIU_Igatherv(a,b,c,d,e,f,g,h,i,req)   MPI_Gatherv(a,b,c,d,e,f,g,h,i)
#define MPIU_Iallgather(a,b,c,d,e,f,g,req)     MPI_Allgather(a,b,c,d,e,f,g)
#define MPIU_Iallgatherv(a,b,c,d,e,f,g,h,req)  MPI_Allgatherv(a,b,c,d,e,f,g,h)
#define MPIU_Ialltoall(a,b,c,d,e,f,g,req)      MPI_Alltoall(a,b,c,d,e,f,g)
#endif

PETSC_STATIC_INLINE PetscErrorCode PetscGetMemType(const void *data,PetscMemType *mtype)
{
  PetscFunctionBegin;
  PetscValidPointer(mtype,2);
  *mtype = PETSC_MEMTYPE_HOST;
#if defined(PETSC_HAVE_CUDA)
  {
    struct cudaPointerAttributes attr;
    if (data) {
#if (CUDART_VERSION < 10000)
      attr.memoryType = cudaMemoryTypeHost;
      cudaPointerGetAttributes(&attr,data);
      cudaGetLastError();
      if (attr.memoryType == cudaMemoryTypeDevice) *mtype = PETSC_MEMTYPE_DEVICE;
#else
      attr.type = cudaMemoryTypeHost;
      cudaPointerGetAttributes(&attr,data); /* Do not check error since before CUDA 11.0, passing host pointer will return cudaErrorInvalidValue */
      cudaGetLastError(); /* Get and then clear the last error */
      if (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged) *mtype = PETSC_MEMTYPE_DEVICE;
#endif
    }
  }
#endif
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscMallocWithMemType(PetscMemType mtype,size_t size,void** ptr)
{
  PetscFunctionBegin;
  if (mtype == PETSC_MEMTYPE_HOST) {PetscErrorCode ierr = PetscMalloc(size,ptr);CHKERRQ(ierr);}
#if defined(PETSC_HAVE_CUDA)
  else if (mtype == PETSC_MEMTYPE_DEVICE) {cudaError_t err = cudaMalloc(ptr,size);CHKERRCUDA(err);}
#endif
  else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Wrong PetscMemType %d", (int)mtype);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscFreeWithMemType_Private(PetscMemType mtype,void* ptr)
{
  PetscFunctionBegin;
  if (mtype == PETSC_MEMTYPE_HOST) {PetscErrorCode ierr = PetscFree(ptr);CHKERRQ(ierr);}
#if defined(PETSC_HAVE_CUDA)
  else if (mtype == PETSC_MEMTYPE_DEVICE) {cudaError_t err = cudaFree(ptr);CHKERRCUDA(err);}
#endif
  else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Wrong PetscMemType %d",(int)mtype);
  PetscFunctionReturn(0);
}

/* Free memory and set ptr to NULL when succeeded */
#define PetscFreeWithMemType(t,p) ((p) && (PetscFreeWithMemType_Private((t),(p)) || ((p)=NULL,0)))

PETSC_STATIC_INLINE PetscErrorCode PetscMemcpyWithMemType(PetscMemType dstmtype,PetscMemType srcmtype,void* dst,const void*src,size_t n)
{
  PetscFunctionBegin;
  if (n) {
    if (dstmtype == PETSC_MEMTYPE_HOST && srcmtype == PETSC_MEMTYPE_HOST) {PetscErrorCode ierr = PetscMemcpy(dst,src,n);CHKERRQ(ierr);}
#if defined(PETSC_HAVE_CUDA)
    else if (dstmtype == PETSC_MEMTYPE_DEVICE && srcmtype == PETSC_MEMTYPE_HOST)   {cudaError_t err = cudaMemcpy(dst,src,n,cudaMemcpyHostToDevice);CHKERRCUDA(err);}
    else if (dstmtype == PETSC_MEMTYPE_HOST   && srcmtype == PETSC_MEMTYPE_DEVICE) {cudaError_t err = cudaMemcpy(dst,src,n,cudaMemcpyDeviceToHost);CHKERRCUDA(err);}
    else if (dstmtype == PETSC_MEMTYPE_DEVICE && srcmtype == PETSC_MEMTYPE_DEVICE) {cudaError_t err = cudaMemcpy(dst,src,n,cudaMemcpyDeviceToDevice);CHKERRCUDA(err);}
#endif
    else SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Wrong PetscMemType for dst %d and src %d",(int)dstmtype,(int)srcmtype);
  }
  PetscFunctionReturn(0);
}

#endif
