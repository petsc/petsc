#if !defined(__SFPACK_H)
#define __SFPACK_H

#include <../src/vec/is/sf/impls/basic/sfbasic.h>
#if defined(PETSC_HAVE_CUDA)
  #include <petsccublas.h>
  typedef cudaStream_t  cupmStream_t;
  typedef cudaEvent_t   cupmEvent_t;
#endif

#if defined(PETSC_HAVE_HIP)
  #include <petschipblas.h>
  typedef hipStream_t   cupmStream_t;
  typedef hipEvent_t    cupmEvent_t;
#endif

/* In terms of function overloading, long long int is a different type than int64_t, which PetscInt might be defined to.
   We perfer long long int over PetscInt (int64_t), since CUDA atomics are built around (unsigned) long long int.
 */
typedef long long int          llint;
typedef unsigned long long int ullint;

/* We separate SF communications for SFBasic and SFNeighbor in two parts: local (self,intra-rank) and remote (inter-rank) */
typedef enum {PETSCSF_LOCAL=0, PETSCSF_REMOTE} PetscSFScope;

/* Optimizations in packing & unpacking for destination ranks.

  Suppose there are m indices stored in idx[], and two addresses u, p. We want to do packing:
     p[i] = u[idx[i]], for i in [0,m)

  Indices are associated with n ranks and each rank's indices are stored consecutively in idx[].
  We go through indices for each rank and see if they are indices of a 3D submatrix of size [dx,dy,dz] in
  a parent matrix of size [X,Y,Z], with the submatrix's first index being <start>.

  E.g., for indices 1,2,3, 6,7,8, 11,12,13, the submatrix size is [3,3,1] with start=1, and the parent matrix's size
  is [5,3,1]. For simplicity, if any destination rank does not have this pattern, we give up the optimization.

  Note before using this per-rank optimization, one should check leafcontig[], rootcontig[], which say
  indices in whole are contiguous, and therefore much more useful than this one when true.
 */
struct _n_PetscSFPackOpt {
  PetscInt       *array;      /* [7*n+2] Memory pool for other fields in this struct. Used to easily copy this struct to GPU */
  PetscInt       n;           /* Number of destination ranks */
  PetscInt       *offset;     /* [n+1] Offsets of indices for each rank. offset[0]=0, offset[i+1]=offset[i]+dx[i]*dy[i]*dz[i] */
  PetscInt       *start;      /* [n] First index */
  PetscInt       *dx,*dy,*dz; /* [n] Lengths of the submatrix in X, Y, Z dimension. */
  PetscInt       *X,*Y;       /* [n] Lengths of the outer matrix in X, Y. We do not care Z. */
};

/* An abstract class that defines a communication link, which includes how to pack/unpack data and send/recv buffers
 */
struct _n_PetscSFLink {
  PetscErrorCode (*Memcpy)              (PetscSFLink,PetscMemType,void*,PetscMemType,const void*,size_t); /* Async device memcopy might use stream in the link */
  PetscErrorCode (*PrePack)             (PetscSF,PetscSFLink,PetscSFDirection);
  PetscErrorCode (*PostUnpack)          (PetscSF,PetscSFLink,PetscSFDirection);
  PetscErrorCode (*StartCommunication)  (PetscSF,PetscSFLink,PetscSFDirection);
  PetscErrorCode (*FinishCommunication) (PetscSF,PetscSFLink,PetscSFDirection);
  PetscErrorCode (*SyncDevice)          (PetscSFLink);
  PetscErrorCode (*SyncStream)          (PetscSFLink);
  PetscErrorCode (*Destroy)             (PetscSF,PetscSFLink);

  PetscErrorCode (*BuildDependenceBegin)(PetscSF,PetscSFLink,PetscSFDirection);
  PetscErrorCode (*BuildDependenceEnd)  (PetscSF,PetscSFLink,PetscSFDirection);

  PetscErrorCode (*h_Pack)            (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,const void*,void*);
  PetscErrorCode (*h_UnpackAndInsert) (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,void*,const void*);
  PetscErrorCode (*h_UnpackAndAdd)    (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,void*,const void*);
  PetscErrorCode (*h_UnpackAndMin)    (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,void*,const void*);
  PetscErrorCode (*h_UnpackAndMax)    (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,void*,const void*);
  PetscErrorCode (*h_UnpackAndMinloc) (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,void*,const void*);
  PetscErrorCode (*h_UnpackAndMaxloc) (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,void*,const void*);
  PetscErrorCode (*h_UnpackAndMult)   (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,void*,const void*);
  PetscErrorCode (*h_UnpackAndLAND)   (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,void*,const void*);
  PetscErrorCode (*h_UnpackAndBAND)   (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,void*,const void*);
  PetscErrorCode (*h_UnpackAndLOR)    (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,void*,const void*);
  PetscErrorCode (*h_UnpackAndBOR)    (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,void*,const void*);
  PetscErrorCode (*h_UnpackAndLXOR)   (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,void*,const void*);
  PetscErrorCode (*h_UnpackAndBXOR)   (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,void*,const void*);
  PetscErrorCode (*h_FetchAndAdd)     (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,void*,      void*);

  PetscErrorCode (*h_ScatterAndInsert)(PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,const void*,PetscInt,PetscSFPackOpt,const PetscInt*,void*);
  PetscErrorCode (*h_ScatterAndAdd)   (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,const void*,PetscInt,PetscSFPackOpt,const PetscInt*,void*);
  PetscErrorCode (*h_ScatterAndMin)   (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,const void*,PetscInt,PetscSFPackOpt,const PetscInt*,void*);
  PetscErrorCode (*h_ScatterAndMax)   (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,const void*,PetscInt,PetscSFPackOpt,const PetscInt*,void*);
  PetscErrorCode (*h_ScatterAndMinloc)(PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,const void*,PetscInt,PetscSFPackOpt,const PetscInt*,void*);
  PetscErrorCode (*h_ScatterAndMaxloc)(PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,const void*,PetscInt,PetscSFPackOpt,const PetscInt*,void*);
  PetscErrorCode (*h_ScatterAndMult)  (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,const void*,PetscInt,PetscSFPackOpt,const PetscInt*,void*);
  PetscErrorCode (*h_ScatterAndLAND)  (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,const void*,PetscInt,PetscSFPackOpt,const PetscInt*,void*);
  PetscErrorCode (*h_ScatterAndBAND)  (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,const void*,PetscInt,PetscSFPackOpt,const PetscInt*,void*);
  PetscErrorCode (*h_ScatterAndLOR)   (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,const void*,PetscInt,PetscSFPackOpt,const PetscInt*,void*);
  PetscErrorCode (*h_ScatterAndBOR)   (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,const void*,PetscInt,PetscSFPackOpt,const PetscInt*,void*);
  PetscErrorCode (*h_ScatterAndLXOR)  (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,const void*,PetscInt,PetscSFPackOpt,const PetscInt*,void*);
  PetscErrorCode (*h_ScatterAndBXOR)  (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,const void*,PetscInt,PetscSFPackOpt,const PetscInt*,void*);

  PetscErrorCode (*h_FetchAndAddLocal)(PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,void*,PetscInt,PetscSFPackOpt,const PetscInt*,const void*,void*);

  PetscBool      deviceinited;        /* Are device related fields initialized? */
#if defined(PETSC_HAVE_DEVICE)
  /* These fields are lazily initialized in a sense that only when device pointers are passed to an SF, the SF
     will set them, otherwise it just leaves them alone. Packing routines using regular ops when there are no data race chances.
  */
  PetscErrorCode (*d_Pack)            (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,const void*,void*);
  PetscErrorCode (*d_UnpackAndInsert) (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,void*,const void*);
  PetscErrorCode (*d_UnpackAndAdd)    (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,void*,const void*);
  PetscErrorCode (*d_UnpackAndMin)    (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,void*,const void*);
  PetscErrorCode (*d_UnpackAndMax)    (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,void*,const void*);
  PetscErrorCode (*d_UnpackAndMinloc) (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,void*,const void*);
  PetscErrorCode (*d_UnpackAndMaxloc) (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,void*,const void*);
  PetscErrorCode (*d_UnpackAndMult)   (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,void*,const void*);
  PetscErrorCode (*d_UnpackAndLAND)   (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,void*,const void*);
  PetscErrorCode (*d_UnpackAndBAND)   (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,void*,const void*);
  PetscErrorCode (*d_UnpackAndLOR)    (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,void*,const void*);
  PetscErrorCode (*d_UnpackAndBOR)    (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,void*,const void*);
  PetscErrorCode (*d_UnpackAndLXOR)   (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,void*,const void*);
  PetscErrorCode (*d_UnpackAndBXOR)   (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,void*,const void*);
  PetscErrorCode (*d_FetchAndAdd)     (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,void*,      void*);

  PetscErrorCode (*d_ScatterAndInsert)(PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,const void*,PetscInt,PetscSFPackOpt,const PetscInt*,void*);
  PetscErrorCode (*d_ScatterAndAdd)   (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,const void*,PetscInt,PetscSFPackOpt,const PetscInt*,void*);
  PetscErrorCode (*d_ScatterAndMin)   (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,const void*,PetscInt,PetscSFPackOpt,const PetscInt*,void*);
  PetscErrorCode (*d_ScatterAndMax)   (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,const void*,PetscInt,PetscSFPackOpt,const PetscInt*,void*);
  PetscErrorCode (*d_ScatterAndMinloc)(PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,const void*,PetscInt,PetscSFPackOpt,const PetscInt*,void*);
  PetscErrorCode (*d_ScatterAndMaxloc)(PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,const void*,PetscInt,PetscSFPackOpt,const PetscInt*,void*);
  PetscErrorCode (*d_ScatterAndMult)  (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,const void*,PetscInt,PetscSFPackOpt,const PetscInt*,void*);
  PetscErrorCode (*d_ScatterAndLAND)  (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,const void*,PetscInt,PetscSFPackOpt,const PetscInt*,void*);
  PetscErrorCode (*d_ScatterAndBAND)  (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,const void*,PetscInt,PetscSFPackOpt,const PetscInt*,void*);
  PetscErrorCode (*d_ScatterAndLOR)   (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,const void*,PetscInt,PetscSFPackOpt,const PetscInt*,void*);
  PetscErrorCode (*d_ScatterAndBOR)   (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,const void*,PetscInt,PetscSFPackOpt,const PetscInt*,void*);
  PetscErrorCode (*d_ScatterAndLXOR)  (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,const void*,PetscInt,PetscSFPackOpt,const PetscInt*,void*);
  PetscErrorCode (*d_ScatterAndBXOR)  (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,const void*,PetscInt,PetscSFPackOpt,const PetscInt*,void*);
  PetscErrorCode (*d_FetchAndAddLocal)(PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,void*,PetscInt,PetscSFPackOpt,const PetscInt*,const void*,void*);

  /* Packing routines using atomics when there are data race chances */
  PetscErrorCode (*da_UnpackAndInsert)(PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,void*,const void*);
  PetscErrorCode (*da_UnpackAndAdd)   (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,void*,const void*);
  PetscErrorCode (*da_UnpackAndMin)   (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,void*,const void*);
  PetscErrorCode (*da_UnpackAndMax)   (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,void*,const void*);
  PetscErrorCode (*da_UnpackAndMinloc)(PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,void*,const void*);
  PetscErrorCode (*da_UnpackAndMaxloc)(PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,void*,const void*);
  PetscErrorCode (*da_UnpackAndMult)  (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,void*,const void*);
  PetscErrorCode (*da_UnpackAndLAND)  (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,void*,const void*);
  PetscErrorCode (*da_UnpackAndBAND)  (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,void*,const void*);
  PetscErrorCode (*da_UnpackAndLOR)   (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,void*,const void*);
  PetscErrorCode (*da_UnpackAndBOR)   (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,void*,const void*);
  PetscErrorCode (*da_UnpackAndLXOR)  (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,void*,const void*);
  PetscErrorCode (*da_UnpackAndBXOR)  (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,void*,const void*);
  PetscErrorCode (*da_FetchAndAdd)    (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,void*,      void*);

  PetscErrorCode (*da_ScatterAndInsert)(PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,const void*,PetscInt,PetscSFPackOpt,const PetscInt*,void*);
  PetscErrorCode (*da_ScatterAndAdd)   (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,const void*,PetscInt,PetscSFPackOpt,const PetscInt*,void*);
  PetscErrorCode (*da_ScatterAndMin)   (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,const void*,PetscInt,PetscSFPackOpt,const PetscInt*,void*);
  PetscErrorCode (*da_ScatterAndMax)   (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,const void*,PetscInt,PetscSFPackOpt,const PetscInt*,void*);
  PetscErrorCode (*da_ScatterAndMinloc)(PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,const void*,PetscInt,PetscSFPackOpt,const PetscInt*,void*);
  PetscErrorCode (*da_ScatterAndMaxloc)(PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,const void*,PetscInt,PetscSFPackOpt,const PetscInt*,void*);
  PetscErrorCode (*da_ScatterAndMult)  (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,const void*,PetscInt,PetscSFPackOpt,const PetscInt*,void*);
  PetscErrorCode (*da_ScatterAndLAND)  (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,const void*,PetscInt,PetscSFPackOpt,const PetscInt*,void*);
  PetscErrorCode (*da_ScatterAndBAND)  (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,const void*,PetscInt,PetscSFPackOpt,const PetscInt*,void*);
  PetscErrorCode (*da_ScatterAndLOR)   (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,const void*,PetscInt,PetscSFPackOpt,const PetscInt*,void*);
  PetscErrorCode (*da_ScatterAndBOR)   (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,const void*,PetscInt,PetscSFPackOpt,const PetscInt*,void*);
  PetscErrorCode (*da_ScatterAndLXOR)  (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,const void*,PetscInt,PetscSFPackOpt,const PetscInt*,void*);
  PetscErrorCode (*da_ScatterAndBXOR)  (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,const void*,PetscInt,PetscSFPackOpt,const PetscInt*,void*);
  PetscErrorCode (*da_FetchAndAddLocal)(PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,void*,PetscInt,PetscSFPackOpt,const PetscInt*,const void*,void*);
 #if defined (PETSC_HAVE_CUDA) || defined(PETSC_HAVE_HIP)
  PetscInt       maxResidentThreadsPerGPU;   /* It is a copy from SF for convenience */
  cupmStream_t   stream;                     /* stream on which input/output root/leafdata is computed on (default is PetscDefaultCudaStream) */
 #endif
#endif
  PetscMPIInt  tag;                          /* Each link has a tag so we can perform multiple SF ops at the same time */
  MPI_Datatype unit;                         /* The MPI datatype this PetscSFLink is built for */
  MPI_Datatype basicunit;                    /* unit is made of MPI builtin dataype basicunit */
  PetscBool    isbuiltin;                    /* Is unit an MPI/PETSc builtin datatype? If it is true, then bs=1 and basicunit is equivalent to unit */
  size_t       unitbytes;                    /* Number of bytes in a unit */
  PetscInt     bs;                           /* Number of basic units in a unit */
  const void   *rootdata,*leafdata;          /* rootdata and leafdata the link is working on. They are used as keys for pending links. */
  PetscMemType rootmtype,leafmtype;          /* root/leafdata's memory type */

  /* For local and remote communication */
  PetscMemType rootmtype_mpi,leafmtype_mpi;  /* Mtypes of buffers passed to MPI. If use_gpu_aware_mpi, they are same as root/leafmtype. Otherwise they are PETSC_MEMTYPE_HOST */
  PetscBool    rootdirect[2],leafdirect[2];  /* Can root/leafdata be directly passed to SF (i.e., without buffering). In layout of [PETSCSF_LOCAL/REMOTE]. See more in PetscSFLinkCreate() */
  PetscInt     rootdirect_mpi,leafdirect_mpi;/* Can root/leafdata for remote be directly passed to MPI? 1: yes, 0: no. See more in PetscSFLinkCreate() */
  const void   *rootdatadirect[2][2];        /* The root/leafdata used to init root/leaf requests, in layout of [PETSCSF_DIRECTION][PETSC_MEMTYPE]. */
  const void   *leafdatadirect[2][2];        /* ... We need them to look up links when root/leafdirect_mpi are true */
  char         *rootbuf[2][2];               /* Buffers for packed roots, in layout of [PETSCSF_LOCAL/REMOTE][PETSC_MEMTYPE]. PETSCSF_LOCAL does not need MPI, .. */
                                             /* .. but in case rootmtype is different from leafmtype, we still need to pack local roots and then copy them to memory of leafmtype */
  char         *rootbuf_alloc[2][2];         /* Log memory allocated by petsc. We need it since rootbuf[][] may point to rootdata given by user */
  char         *leafbuf[2][2];               /* Buffers for packed leaves, in layout of [PETSCSF_LOCAL/REMOTE][PETSC_MEMTYPE] */
  char         *leafbuf_alloc[2][2];
  MPI_Request  *rootreqs[2][2][2];           /* Root requests in layout of [PETSCSF_DIRECTION][PETSC_MEMTYPE][rootdirect_mpi] */
  MPI_Request  *leafreqs[2][2][2];           /* Leaf requests in layout of [PETSCSF_DIRECTION][PETSC_MEMTYPE][leafdirect_mpi] */
  PetscBool    rootreqsinited[2][2][2];      /* Are root requests initialized? Also in layout of [PETSCSF_DIRECTION][PETSC_MEMTYPE][rootdirect_mpi]*/
  PetscBool    leafreqsinited[2][2][2];      /* Are leaf requests initialized? Also in layout of [PETSCSF_DIRECTION][PETSC_MEMTYPE][leafdirect_mpi]*/
  MPI_Request  *reqs;                        /* An array of length (nrootreqs+nleafreqs)*8. Pointers in rootreqs[][][] and leafreqs[][][] point here */
  PetscSFLink  next;

  PetscBool    use_nvshmem;                  /* Does this link use nvshem (vs. MPI) for communication? */
#if defined(PETSC_HAVE_NVSHMEM)
  cupmEvent_t  dataReady;                    /* Events to mark readiness of root/leafdata */
  cupmEvent_t  endRemoteComm;              /* Events to mark end of local/remote communication */
  cupmStream_t remoteCommStream;             /* Streams for remote (i.e., inter-rank) communication */

  /* The buffers are allocated in device symmetric heap. Their length is the maximal length over all ranks in the comm, and therefore is the same. */
  uint64_t     *rootSendSig,*rootRecvSig;    /* [max{niranks-ndiranks}], signals used when rootbuf works as send/recv buf */
  uint64_t     *leafSendSig,*leafRecvSig;    /* [max{nranks-ndranks}], signals used when leafbuf works as send/recv buf */
#endif
};

PETSC_INTERN PetscErrorCode PetscSFSetErrorOnUnsupportedOverlap(PetscSF,MPI_Datatype,const void*,const void*);

/* Create/setup/retrieve/destroy a link */
PETSC_INTERN PetscErrorCode PetscSFLinkCreate(PetscSF,MPI_Datatype,PetscMemType,const void*,PetscMemType,const void*,MPI_Op,PetscSFOperation,PetscSFLink*);
PETSC_INTERN PetscErrorCode PetscSFLinkSetUp_Host(PetscSF,PetscSFLink,MPI_Datatype);
PETSC_INTERN PetscErrorCode PetscSFLinkGetInUse(PetscSF,MPI_Datatype,const void*,const void*,PetscCopyMode,PetscSFLink*);
PETSC_INTERN PetscErrorCode PetscSFLinkReclaim(PetscSF,PetscSFLink*);
PETSC_INTERN PetscErrorCode PetscSFLinkDestroy(PetscSF,PetscSFLink);

/* Get pack/unpack function pointers from a link */
PETSC_STATIC_INLINE PetscErrorCode PetscSFLinkGetPack(PetscSFLink link,PetscMemType mtype,PetscErrorCode (**Pack)(PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,const void*,void*))
{
  PetscFunctionBegin;
  if (PetscMemTypeHost(mtype)) *Pack = link->h_Pack;
#if defined(PETSC_HAVE_DEVICE)
  else *Pack = link->d_Pack;
#endif
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode PetscSFLinkGetUnpackAndOp(PetscSFLink,PetscMemType,MPI_Op,PetscBool,PetscErrorCode (**UnpackAndOp)(PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,void*,const void*));
PETSC_INTERN PetscErrorCode PetscSFLinkGetFetchAndOp (PetscSFLink,PetscMemType,MPI_Op,PetscBool,PetscErrorCode (**FetchAndOp) (PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,void*,void*));
PETSC_INTERN PetscErrorCode PetscSFLinkGetScatterAndOp(PetscSFLink,PetscMemType,MPI_Op,PetscBool,PetscErrorCode (**ScatterAndOp)(PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,const void*,PetscInt,PetscSFPackOpt,const PetscInt*,void*));
PETSC_INTERN PetscErrorCode PetscSFLinkGetFetchAndOpLocal(PetscSFLink,PetscMemType,MPI_Op,PetscBool,PetscErrorCode (**FetchAndOpLocal)(PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,void*,PetscInt,PetscSFPackOpt,const PetscInt*,const void*,void*));
PETSC_INTERN PetscErrorCode PetscSFLinkGetMPIBuffersAndRequests(PetscSF,PetscSFLink,PetscSFDirection,void**,void**,MPI_Request**,MPI_Request**);

/* Do Pack/Unpack/Fetch/Scatter with the link */
PETSC_INTERN PetscErrorCode PetscSFLinkPackRootData  (PetscSF,PetscSFLink,PetscSFScope,const void*);
PETSC_INTERN PetscErrorCode PetscSFLinkPackLeafData  (PetscSF,PetscSFLink,PetscSFScope,const void*);
PETSC_INTERN PetscErrorCode PetscSFLinkUnpackRootData(PetscSF,PetscSFLink,PetscSFScope,void*,MPI_Op);
PETSC_INTERN PetscErrorCode PetscSFLinkUnpackLeafData(PetscSF,PetscSFLink,PetscSFScope,void*,MPI_Op);
PETSC_INTERN PetscErrorCode PetscSFLinkFetchAndOpRemote (PetscSF,PetscSFLink,void*,MPI_Op);

PETSC_INTERN PetscErrorCode PetscSFLinkScatterLocal(PetscSF,PetscSFLink,PetscSFDirection,void*,void*,MPI_Op);
PETSC_INTERN PetscErrorCode PetscSFLinkFetchAndOpLocal(PetscSF,PetscSFLink,void*,const void*,void*,MPI_Op);

PETSC_INTERN PetscErrorCode PetscSFSetUpPackFields(PetscSF);
PETSC_INTERN PetscErrorCode PetscSFResetPackFields(PetscSF);
PETSC_INTERN PetscErrorCode PetscSFLinkCreate_MPI(PetscSF,MPI_Datatype,PetscMemType,const void*,PetscMemType,const void*,MPI_Op,PetscSFOperation,PetscSFLink*);

#if defined(PETSC_HAVE_CUDA)
PETSC_INTERN PetscErrorCode PetscSFLinkSetUp_CUDA(PetscSF,PetscSFLink,MPI_Datatype);
#endif

#if defined(PETSC_HAVE_HIP)
PETSC_INTERN PetscErrorCode PetscSFLinkSetUp_HIP(PetscSF,PetscSFLink,MPI_Datatype);
#endif

#if defined(PETSC_HAVE_KOKKOS)
PETSC_INTERN PetscErrorCode PetscSFLinkSetUp_Kokkos(PetscSF,PetscSFLink,MPI_Datatype);
#endif

#if defined(PETSC_HAVE_NVSHMEM)
PETSC_INTERN PetscErrorCode PetscSFLinkCreate_NVSHMEM(PetscSF,MPI_Datatype,PetscMemType,const void*,PetscMemType,const void*,MPI_Op,PetscSFOperation,PetscSFLink*);
PETSC_INTERN PetscErrorCode PetscSFLinkNvshmemCheck(PetscSF,PetscMemType,const void*,PetscMemType,const void*,PetscBool*);
#endif

PETSC_STATIC_INLINE PetscErrorCode PetscSFLinkStartCommunication(PetscSF sf,PetscSFLink link,PetscSFDirection direction)
{
  PetscFunctionBegin;
  if (link->StartCommunication) {PetscErrorCode ierr = (*link->StartCommunication)(sf,link,direction);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscSFLinkFinishCommunication(PetscSF sf,PetscSFLink link,PetscSFDirection direction)
{
  PetscFunctionBegin;
  if (link->FinishCommunication) {PetscErrorCode ierr=(*link->FinishCommunication)(sf,link,direction);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/* A set of helper routines for Pack/Unpack/Scatter on GPUs */
#if defined(PETSC_HAVE_CUDA) || defined(PETSC_HAVE_HIP)
/* PetscSFLinkCopyXxxxBufferInCaseNotUseGpuAwareMPI routines are simple: if not use_gpu_aware_mpi, we need
   to copy the buffer from GPU to CPU before MPI calls, and from CPU to GPU after MPI calls.
*/
PETSC_STATIC_INLINE PetscErrorCode PetscSFLinkCopyRootBufferInCaseNotUseGpuAwareMPI(PetscSF sf,PetscSFLink link,PetscBool device2host)
{
  PetscErrorCode ierr;
  PetscSF_Basic  *bas = (PetscSF_Basic*)sf->data;

  PetscFunctionBegin;
  /* rootdata is on device but we use regular MPI for communication */
  if (PetscMemTypeDevice(link->rootmtype) && PetscMemTypeHost(link->rootmtype_mpi) && bas->rootbuflen[PETSCSF_REMOTE]) {
    void  *h_buf = link->rootbuf[PETSCSF_REMOTE][PETSC_MEMTYPE_HOST];
    void  *d_buf = link->rootbuf[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE];
    size_t count = bas->rootbuflen[PETSCSF_REMOTE]*link->unitbytes;
    if (device2host) {
      ierr = (*link->Memcpy)(link,PETSC_MEMTYPE_HOST,h_buf,PETSC_MEMTYPE_DEVICE,d_buf,count);CHKERRQ(ierr);
      ierr = PetscLogGpuToCpu(count);CHKERRQ(ierr);
    } else {
      ierr = (*link->Memcpy)(link,PETSC_MEMTYPE_DEVICE,d_buf,PETSC_MEMTYPE_HOST,h_buf,count);CHKERRQ(ierr);
      ierr = PetscLogCpuToGpu(count);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscSFLinkCopyLeafBufferInCaseNotUseGpuAwareMPI(PetscSF sf,PetscSFLink link,PetscBool device2host)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscMemTypeDevice(link->leafmtype) && PetscMemTypeHost(link->leafmtype_mpi) && sf->leafbuflen[PETSCSF_REMOTE]) {
    void  *h_buf = link->leafbuf[PETSCSF_REMOTE][PETSC_MEMTYPE_HOST];
    void  *d_buf = link->leafbuf[PETSCSF_REMOTE][PETSC_MEMTYPE_DEVICE];
    size_t count = sf->leafbuflen[PETSCSF_REMOTE]*link->unitbytes;
    if (device2host) {
      ierr = (*link->Memcpy)(link,PETSC_MEMTYPE_HOST,h_buf,PETSC_MEMTYPE_DEVICE,d_buf,count);CHKERRQ(ierr);
      ierr = PetscLogGpuToCpu(count);CHKERRQ(ierr);
    } else {
      ierr = (*link->Memcpy)(link,PETSC_MEMTYPE_DEVICE,d_buf,PETSC_MEMTYPE_HOST,h_buf,count);CHKERRQ(ierr);
      ierr = PetscLogCpuToGpu(count);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/* Make sure root/leafbuf for the remote is ready for MPI */
PETSC_STATIC_INLINE PetscErrorCode PetscSFLinkSyncStreamBeforeCallMPI(PetscSF sf,PetscSFLink link,PetscSFDirection direction)
{
  PetscErrorCode ierr;
  PetscSF_Basic  *bas;
  PetscInt       buflen;
  PetscMemType   mtype;

  PetscFunctionBegin;
  if (direction == PETSCSF_ROOT2LEAF) {
    bas    = (PetscSF_Basic*)sf->data;
    mtype  = link->rootmtype;
    buflen = bas->rootbuflen[PETSCSF_REMOTE];
  } else {
    mtype  = link->leafmtype;
    buflen = sf->leafbuflen[PETSCSF_REMOTE];
  }

  if (PetscMemTypeDevice(mtype) && buflen) {
    ierr = (*link->SyncStream)(link);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
#else /* Host only */
  #define PetscSFLinkCopyRootBufferInCaseNotUseGpuAwareMPI(a,b,c)                  0
  #define PetscSFLinkCopyLeafBufferInCaseNotUseGpuAwareMPI(a,b,c)                  0
  #define PetscSFLinkSyncStreamBeforeCallMPI(a,b,c)                                0
#endif

/* Get root indices used for pack/unpack

Input arguments:
  +sf    - StarForest
  .link  - The link, which provides the stream for the async memcpy (In SF, we make all GPU operations asynchronous to avoid unexpected pipeline stalls)
  .mtype - In what type of memory? (PETSC_MEMTYPE_DEVICE or PETSC_MEMTYPE_HOST)
  -scope - Which part of the indices? (PETSCSF_LOCAL or PETSCSF_REMOTE)

 Output arguments:
  +count   - Count of indices
  .start   - The first index (only useful when indices is NULL)
  .opt     - Packing optimizations
  -indices - Indices of roots for pack/unpack. NULL means indices are contiguous
 */
PETSC_STATIC_INLINE PetscErrorCode PetscSFLinkGetRootPackOptAndIndices(PetscSF sf,PetscSFLink link,PetscMemType mtype,PetscSFScope scope,PetscInt *count,PetscInt *start,PetscSFPackOpt *opt,const PetscInt **indices)
{
  PetscSF_Basic  *bas = (PetscSF_Basic*)sf->data;
  PetscInt       offset;

  PetscFunctionBegin;
  *count   = bas->rootbuflen[scope];
  *start   = bas->rootstart[scope];
  *opt     = NULL;
  *indices = NULL;

  /* We have these rules:
    1) opt == NULL && indices == NULL ==> indices are contiguous.
    2) opt != NULL ==> indices are in 3D but not contiguous. On host, indices != NULL since indices are already available and we do not
       want to enforce all operations to use opt; but on device, indices = NULL since we do not want to copy indices to device.
  */
  if (!bas->rootcontig[scope]) {
    offset = (scope == PETSCSF_LOCAL)? 0 : bas->ioffset[bas->ndiranks];
    if (PetscMemTypeHost(mtype)) {*opt = bas->rootpackopt[scope]; *indices = bas->irootloc + offset;}
    else {
      PetscErrorCode ierr;
      size_t         size;
      if (bas->rootpackopt[scope]) {
        if (!bas->rootpackopt_d[scope]) {
          ierr = PetscMalloc1(1,&bas->rootpackopt_d[scope]);CHKERRQ(ierr);
          ierr = PetscArraycpy(bas->rootpackopt_d[scope],bas->rootpackopt[scope],1);CHKERRQ(ierr); /* Make pointers in bas->rootpackopt_d[] still work on host */
          size = (bas->rootpackopt[scope]->n*7+2)*sizeof(PetscInt); /* See comments at struct _n_PetscSFPackOpt*/
          ierr = PetscSFMalloc(sf,PETSC_MEMTYPE_DEVICE,size,(void **)&bas->rootpackopt_d[scope]->array);CHKERRQ(ierr);
          ierr = (*link->Memcpy)(link,PETSC_MEMTYPE_DEVICE,bas->rootpackopt_d[scope]->array,PETSC_MEMTYPE_HOST,bas->rootpackopt[scope]->array,size);CHKERRQ(ierr);
        }
        *opt = bas->rootpackopt_d[scope];
      } else { /* On device, we only provide indices when there is no optimization. We're reluctant to copy indices to device. */
        if (!bas->irootloc_d[scope]) {
          size = bas->rootbuflen[scope]*sizeof(PetscInt);
          ierr = PetscSFMalloc(sf,PETSC_MEMTYPE_DEVICE,size,(void **)&bas->irootloc_d[scope]);CHKERRQ(ierr);
          ierr = (*link->Memcpy)(link,PETSC_MEMTYPE_DEVICE,bas->irootloc_d[scope],PETSC_MEMTYPE_HOST,bas->irootloc+offset,size);CHKERRQ(ierr);
        }
        *indices = bas->irootloc_d[scope];
      }
    }
  }
  PetscFunctionReturn(0);
}

/* Get leaf indices used for pack/unpack

  See also PetscSFLinkGetRootPackOptAndIndices()
 */
PETSC_STATIC_INLINE PetscErrorCode PetscSFLinkGetLeafPackOptAndIndices(PetscSF sf,PetscSFLink link,PetscMemType mtype,PetscSFScope scope,PetscInt *count,PetscInt *start,PetscSFPackOpt *opt,const PetscInt **indices)
{
  PetscInt   offset;

  PetscFunctionBegin;
  *count   = sf->leafbuflen[scope];
  *start   = sf->leafstart[scope];
  *opt     = NULL;
  *indices = NULL;
  if (!sf->leafcontig[scope]) {
    offset = (scope == PETSCSF_LOCAL)? 0 : sf->roffset[sf->ndranks];
    if (PetscMemTypeHost(mtype)) {*opt = sf->leafpackopt[scope]; *indices = sf->rmine + offset;}
    else {
      PetscErrorCode ierr;
      size_t         size;
      if (sf->leafpackopt[scope]) {
        if (!sf->leafpackopt_d[scope]) {
          ierr = PetscMalloc1(1,&sf->leafpackopt_d[scope]);CHKERRQ(ierr);
          ierr = PetscArraycpy(sf->leafpackopt_d[scope],sf->leafpackopt[scope],1);CHKERRQ(ierr);
          size = (sf->leafpackopt[scope]->n*7+2)*sizeof(PetscInt); /* See comments at struct _n_PetscSFPackOpt*/
          ierr = PetscSFMalloc(sf,PETSC_MEMTYPE_DEVICE,size,(void **)&sf->leafpackopt_d[scope]->array);CHKERRQ(ierr); /* Change ->array to a device pointer */
          ierr = (*link->Memcpy)(link,PETSC_MEMTYPE_DEVICE,sf->leafpackopt_d[scope]->array,PETSC_MEMTYPE_HOST,sf->leafpackopt[scope]->array,size);CHKERRQ(ierr);
        }
        *opt = sf->leafpackopt_d[scope];
      } else {
        if (!sf->rmine_d[scope]) {
          size = sf->leafbuflen[scope]*sizeof(PetscInt);
          ierr = PetscSFMalloc(sf,PETSC_MEMTYPE_DEVICE,size,(void **)&sf->rmine_d[scope]);CHKERRQ(ierr);
          ierr = (*link->Memcpy)(link,PETSC_MEMTYPE_DEVICE,sf->rmine_d[scope],PETSC_MEMTYPE_HOST,sf->rmine+offset,size);CHKERRQ(ierr);
        }
        *indices = sf->rmine_d[scope];
      }
    }
  }
  PetscFunctionReturn(0);
}
#endif
