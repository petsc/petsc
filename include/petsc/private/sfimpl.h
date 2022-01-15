#if !defined(PETSCSFIMPL_H)
#define PETSCSFIMPL_H

#include <petscvec.h>
#include <petscsf.h>
#include <petsc/private/deviceimpl.h>
#include <petsc/private/mpiutils.h>
#include <petsc/private/petscimpl.h>

PETSC_EXTERN PetscLogEvent PETSCSF_SetGraph;
PETSC_EXTERN PetscLogEvent PETSCSF_SetUp;
PETSC_EXTERN PetscLogEvent PETSCSF_BcastBegin;
PETSC_EXTERN PetscLogEvent PETSCSF_BcastEnd;
PETSC_EXTERN PetscLogEvent PETSCSF_BcastBegin;
PETSC_EXTERN PetscLogEvent PETSCSF_BcastEnd;
PETSC_EXTERN PetscLogEvent PETSCSF_ReduceBegin;
PETSC_EXTERN PetscLogEvent PETSCSF_ReduceEnd;
PETSC_EXTERN PetscLogEvent PETSCSF_FetchAndOpBegin;
PETSC_EXTERN PetscLogEvent PETSCSF_FetchAndOpEnd;
PETSC_EXTERN PetscLogEvent PETSCSF_EmbedSF;
PETSC_EXTERN PetscLogEvent PETSCSF_DistSect;
PETSC_EXTERN PetscLogEvent PETSCSF_SectSF;
PETSC_EXTERN PetscLogEvent PETSCSF_RemoteOff;
PETSC_EXTERN PetscLogEvent PETSCSF_Pack;
PETSC_EXTERN PetscLogEvent PETSCSF_Unpack;

typedef enum {PETSCSF_ROOT2LEAF=0, PETSCSF_LEAF2ROOT} PetscSFDirection;
typedef enum {PETSCSF_BCAST=0, PETSCSF_REDUCE, PETSCSF_FETCH} PetscSFOperation;
/* When doing device-aware MPI, a backend refers to the SF/device interface */
typedef enum {PETSCSF_BACKEND_INVALID=0,PETSCSF_BACKEND_CUDA,PETSCSF_BACKEND_HIP,PETSCSF_BACKEND_KOKKOS} PetscSFBackend;

struct _PetscSFOps {
  PetscErrorCode (*Reset)(PetscSF);
  PetscErrorCode (*Destroy)(PetscSF);
  PetscErrorCode (*SetUp)(PetscSF);
  PetscErrorCode (*SetFromOptions)(PetscOptionItems*,PetscSF);
  PetscErrorCode (*View)(PetscSF,PetscViewer);
  PetscErrorCode (*Duplicate)(PetscSF,PetscSFDuplicateOption,PetscSF);
  PetscErrorCode (*BcastBegin)     (PetscSF,MPI_Datatype,PetscMemType,const void*,PetscMemType,void*,MPI_Op);
  PetscErrorCode (*BcastEnd)       (PetscSF,MPI_Datatype,const void*,void*,MPI_Op);
  PetscErrorCode (*ReduceBegin)    (PetscSF,MPI_Datatype,PetscMemType,const void*,PetscMemType,void*,MPI_Op);
  PetscErrorCode (*ReduceEnd)      (PetscSF,MPI_Datatype,const void*,void*,MPI_Op);
  PetscErrorCode (*FetchAndOpBegin)(PetscSF,MPI_Datatype,PetscMemType,void*,PetscMemType,const void*,void*,MPI_Op);
  PetscErrorCode (*FetchAndOpEnd)  (PetscSF,MPI_Datatype,void*,const void*,void*,MPI_Op);
  PetscErrorCode (*BcastToZero)    (PetscSF,MPI_Datatype,PetscMemType,const void*,PetscMemType,      void*); /* For interal use only */
  PetscErrorCode (*GetRootRanks)(PetscSF,PetscInt*,const PetscMPIInt**,const PetscInt**,const PetscInt**,const PetscInt**);
  PetscErrorCode (*GetLeafRanks)(PetscSF,PetscInt*,const PetscMPIInt**,const PetscInt**,const PetscInt**);
  PetscErrorCode (*CreateLocalSF)(PetscSF,PetscSF*);
  PetscErrorCode (*GetGraph)(PetscSF,PetscInt*,PetscInt*,const PetscInt**,const PetscSFNode**);
  PetscErrorCode (*CreateEmbeddedRootSF)(PetscSF,PetscInt,const PetscInt*,PetscSF*);
  PetscErrorCode (*CreateEmbeddedLeafSF)(PetscSF,PetscInt,const PetscInt*,PetscSF*);

  PetscErrorCode (*Malloc)(PetscMemType,size_t,void**);
  PetscErrorCode (*Free)(PetscMemType,void*);
};

typedef struct _n_PetscSFPackOpt *PetscSFPackOpt;

struct _p_PetscSF {
  PETSCHEADER(struct _PetscSFOps);
  struct { /* Fields needed to implement VecScatter behavior */
    PetscInt          from_n,to_n;   /* Recorded local sizes of the input from/to vectors in VecScatterCreate(). Used subsequently for error checking. */
    PetscBool         beginandendtogether;  /* Indicates that the scatter begin and end  function are called together, VecScatterEnd() is then treated as a nop */
    PetscBool         packongpu;     /* For GPU vectors, pack needed entries on GPU instead of pulling the whole vector down to CPU and then packing on CPU */
    const PetscScalar *xdata;        /* Vector data to read from */
    PetscScalar       *ydata;        /* Vector data to write to. The two pointers are recorded in VecScatterBegin. Memory is not managed by SF. */
    PetscSF           lsf;           /* The local part of the scatter, used in SCATTER_LOCAL. Built on demand. */
    PetscInt          bs;            /* Block size, determined by IS passed to VecScatterCreate */
    MPI_Datatype      unit;          /* one unit = bs PetscScalars */
    PetscBool         logging;       /* Indicate if vscat log events are happening. If yes, avoid duplicated SF logging to have clear -log_view */
  } vscat;

  /* Fields for generic PetscSF functionality */
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
  PetscInt        *rmine_d[2];     /* A copy of rmine[local/remote] in device memory if needed */

  /* Some results useful in packing by analyzing rmine[] */
  PetscInt        leafbuflen[2];   /* Length (in unit) of leaf buffers, in layout of [PETSCSF_LOCAL/REMOTE] */
  PetscBool       leafcontig[2];   /* True means indices in rmine[self part] or rmine[remote part] are contiguous, and they start from ... */
  PetscInt        leafstart[2];    /* ... leafstart[0] and leafstart[1] respectively */
  PetscSFPackOpt  leafpackopt[2];  /* Optimization plans to (un)pack leaves connected to remote roots, based on index patterns in rmine[]. NULL for no optimization */
  PetscSFPackOpt  leafpackopt_d[2];/* Copy of leafpackopt_d[] on device if needed */
  PetscBool       leafdups[2];     /* Indices in rmine[] for self(0)/remote(1) communication have dups respectively? TRUE implies theads working on them in parallel may have data race. */

  PetscInt        nleafreqs;       /* Number of MPI reqests for leaves */
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
  PetscSFPattern  pattern;         /* Pattern of the graph */
  PetscBool       persistent;      /* Does this SF use MPI persistent requests for communication */
  PetscLayout     map;             /* Layout of leaves over all processes when building a patterned graph */
  PetscBool       unknown_input_stream;/* If true, SF does not know which streams root/leafdata is on. Default is false, since we only use petsc default stream */
  PetscBool       use_gpu_aware_mpi;   /* If true, SF assumes it can pass GPU pointers to MPI */
  PetscBool       use_stream_aware_mpi;/* If true, SF assumes the underlying MPI is cuda-stream aware and we won't sync streams for send/recv buffers passed to MPI */
  PetscInt        maxResidentThreadsPerGPU;
  PetscSFBackend  backend;         /* The device backend (if any) SF will use */
  void *data;                      /* Pointer to implementation */

 #if defined(PETSC_HAVE_NVSHMEM)
  PetscBool       use_nvshmem;     /* TRY to use nvshmem on cuda devices with this SF when possible */
  PetscBool       use_nvshmem_get; /* If true, use nvshmem_get based protocal, otherwise, use nvshmem_put based protocol */
  PetscBool       checked_nvshmem_eligibility; /* Have we checked eligibility of using NVSHMEM on this sf? */
  PetscBool       setup_nvshmem;   /* Have we already set up NVSHMEM related fields below? These fields are built on-demand */
  PetscInt        leafbuflen_rmax; /* max leafbuflen[REMOTE] over comm */
  PetscInt        nRemoteRootRanks;/* nranks - ndranks */
  PetscInt        nRemoteRootRanksMax; /* max nranks-ndranks over comm */

  /* The following two fields look confusing but actually make sense: They are offsets of buffers at the remote side. We're doing one-sided communication! */
  PetscInt        *rootsigdisp;    /* [nRemoteRootRanks]. For my i-th remote root rank, I will access its rootsigdisp[i]-th root signal */
  PetscInt        *rootbufdisp;    /* [nRemoteRootRanks]. For my i-th remote root rank, I will access its root buf at offset rootbufdisp[i], in <unit> to be set */

  PetscInt        *rootbufdisp_d;
  PetscInt        *rootsigdisp_d;  /* Copy of rootsigdisp[] on device */
  PetscMPIInt     *ranks_d;        /* Copy of the remote part of (root) ranks[] on device */
  PetscInt        *roffset_d;      /* Copy of the remote part of roffset[] on device */
 #endif
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

PETSC_EXTERN PetscErrorCode VecScatterGetRemoteCount_Private(VecScatter,PetscBool,PetscInt*,PetscInt*);
PETSC_EXTERN PetscErrorCode VecScatterGetRemote_Private(VecScatter,PetscBool,PetscInt*,const PetscInt**,const PetscInt**,const PetscMPIInt**,PetscInt*);
PETSC_EXTERN PetscErrorCode VecScatterGetRemoteOrdered_Private(VecScatter,PetscBool,PetscInt*,const PetscInt**,const PetscInt**,const PetscMPIInt**,PetscInt*);
PETSC_EXTERN PetscErrorCode VecScatterRestoreRemote_Private(VecScatter,PetscBool,PetscInt*,const PetscInt**,const PetscInt**,const PetscMPIInt**,PetscInt*);
PETSC_EXTERN PetscErrorCode VecScatterRestoreRemoteOrdered_Private(VecScatter,PetscBool,PetscInt*,const PetscInt**,const PetscInt**,const PetscMPIInt**,PetscInt*);

#if defined(PETSC_HAVE_CUDA)
PETSC_EXTERN PetscErrorCode PetscSFMalloc_CUDA(PetscMemType,size_t,void**);
PETSC_EXTERN PetscErrorCode PetscSFFree_CUDA(PetscMemType,void*);
#endif
#if defined(PETSC_HAVE_HIP)
PETSC_EXTERN PetscErrorCode PetscSFMalloc_HIP(PetscMemType,size_t,void**);
PETSC_EXTERN PetscErrorCode PetscSFFree_HIP(PetscMemType,void*);
#endif

#if defined(PETSC_HAVE_KOKKOS)
  PETSC_EXTERN PetscErrorCode PetscSFMalloc_Kokkos(PetscMemType,size_t,void**);
  PETSC_EXTERN PetscErrorCode PetscSFFree_Kokkos(PetscMemType,void*);
 #if defined(PETSC_HAVE_CUDA)
  static const PetscMemType PETSC_MEMTYPE_KOKKOS = PETSC_MEMTYPE_CUDA;
 #elif defined(PETSC_HAVE_HIP)
  static const PetscMemType PETSC_MEMTYPE_KOKKOS = PETSC_MEMTYPE_HIP;
 #elif defined(PETSC_HAVE_SYCL)
  static const PetscMemType PETSC_MEMTYPE_KOKKOS = PETSC_MEMTYPE_SYCL;
 #else
  static const PetscMemType PETSC_MEMTYPE_KOKKOS = PETSC_MEMTYPE_HOST;
 #endif
#endif

/* SF only supports CUDA and Kokkos devices. Even VIENNACL is a device, its device pointers are invisible to SF.
   Through VecGetArray(), we copy data of VECVIENNACL from device to host and pass host pointers to SF.
 */
#if defined(PETSC_HAVE_CUDA) || defined(PETSC_HAVE_KOKKOS) || defined(PETSC_HAVE_HIP)
  #define PetscSFMalloc(sf,mtype,sz,ptr)  ((*(sf)->ops->Malloc)(mtype,sz,ptr))
  /* Free memory and set ptr to NULL when succeeded */
  #define PetscSFFree(sf,mtype,ptr)       ((ptr) && ((*(sf)->ops->Free)(mtype,ptr) || ((ptr)=NULL,0)))
#else
  /* If pure host code, do with less indirection */
  #define PetscSFMalloc(sf,mtype,sz,ptr)  PetscMalloc(sz,ptr)
  #define PetscSFFree(sf,mtype,ptr)       PetscFree(ptr)
#endif

#endif
