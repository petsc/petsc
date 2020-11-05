#if !defined(__SFPACK_H)
#define __SFPACK_H

#include <../src/vec/is/sf/impls/basic/sfbasic.h>
#if defined(PETSC_HAVE_CUDA)
  #include <cuda_runtime.h> /* For cudaStream_t */
  #include <petsccublas.h>  /* For CHKERRCUDA */
#endif

#if defined(PETSC_HAVE_HIP)
  #include <hip/hip_runtime.h>  /* For hipStream_t */
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
  PetscErrorCode (*Memcpy)            (PetscSFLink,PetscMemType,void*,PetscMemType,const void*,size_t); /* Asynchronous copy might use stream in the link */

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
  PetscErrorCode (*d_SyncDevice)      (PetscSFLink);
  PetscErrorCode (*d_SyncStream)      (PetscSFLink);

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
#if defined (PETSC_HAVE_CUDA)
  PetscInt       maxResidentThreadsPerGPU;   /* It is a copy from SF for convenience */
  cudaStream_t   stream;                     /* Stream to launch pack/unapck kernels if not using the default stream */
#elif defined (PETSC_HAVE_HIP)
  hipStream_t    stream;
#endif

  PetscErrorCode (*Destroy)(PetscSFLink);    /* These two fields are meant to be used by SF_Kokkos, with spptr pointing to an execution space object */
  void           *spptr;                     /* for a given stream, but unused now due to a Kokkos bug, so that SF_Kokkos only supports null stream. */
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
  char         *rootbuf[2][2];               /* Buffers for packed roots, in layout of [PETSCSF_LOCAL/REMOTE][PETSC_MEMTYPE] */
  char         *rootbuf_alloc[2][2];         /* Log memory allocated by petsc. We need it since rootbuf[][] may point to rootdata given by user */
  char         *leafbuf[2][2];               /* Buffers for packed leaves, in layout of [PETSCSF_LOCAL/REMOTE][PETSC_MEMTYPE] */
  char         *leafbuf_alloc[2][2];
  MPI_Request  *rootreqs[2][2][2];           /* Root requests in layout of [PETSCSF_DIRECTION][PETSC_MEMTYPE][rootdirect_mpi] */
  MPI_Request  *leafreqs[2][2][2];           /* Leaf requests in layout of [PETSCSF_DIRECTION][PETSC_MEMTYPE][leafdirect_mpi] */
  PetscBool    rootreqsinited[2][2][2];      /* Are root requests initialized? Also in layout of [PETSCSF_DIRECTION][PETSC_MEMTYPE][rootdirect_mpi]*/
  PetscBool    leafreqsinited[2][2][2];      /* Are leaf requests initialized? Also in layout of [PETSCSF_DIRECTION][PETSC_MEMTYPE][leafdirect_mpi]*/
  MPI_Request  *reqs;                        /* An array of length (nrootreqs+nleafreqs)*8. Pointers in rootreqs[][][] and leafreqs[][][] point here */
  PetscSFLink  next;
};

PETSC_INTERN PetscErrorCode PetscSFSetErrorOnUnsupportedOverlap(PetscSF,MPI_Datatype,const void*,const void*);

/* Create/setup/retrieve/destroy a link */
PETSC_INTERN PetscErrorCode PetscSFLinkCreate(PetscSF,MPI_Datatype,PetscMemType,const void*,PetscMemType,const void*,MPI_Op,PetscSFOperation,PetscSFLink*);
PETSC_INTERN PetscErrorCode PetscSFLinkSetUp_Host(PetscSF,PetscSFLink,MPI_Datatype);
PETSC_INTERN PetscErrorCode PetscSFLinkGetInUse(PetscSF,MPI_Datatype,const void*,const void*,PetscCopyMode,PetscSFLink*);
PETSC_INTERN PetscErrorCode PetscSFLinkReclaim(PetscSF,PetscSFLink*);
PETSC_INTERN PetscErrorCode PetscSFLinkDestroy(PetscSF,PetscSFLink*);

/* Get pack/unpack function pointers from a link */
PETSC_STATIC_INLINE PetscErrorCode PetscSFLinkGetPack(PetscSFLink link,PetscMemType mtype,PetscErrorCode (**Pack)(PetscSFLink,PetscInt,PetscInt,PetscSFPackOpt,const PetscInt*,const void*,void*))
{
  PetscFunctionBegin;
  if (mtype == PETSC_MEMTYPE_HOST) *Pack = link->h_Pack;
#if defined(PETSC_HAVE_DEVICE)
  else *Pack = link->d_Pack;
#endif
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscSFLinkMPIWaitall(PetscSF sf,PetscSFLink link,PetscSFDirection direction)
{
  PetscErrorCode       ierr;
  PetscSF_Basic        *bas = (PetscSF_Basic*)sf->data;
  const PetscMemType   rootmtype_mpi = link->rootmtype_mpi,leafmtype_mpi = link->leafmtype_mpi;
  const PetscInt       rootdirect_mpi = link->rootdirect_mpi,leafdirect_mpi = link->leafdirect_mpi;

  PetscFunctionBegin;
  ierr = MPI_Waitall(bas->nrootreqs,link->rootreqs[direction][rootmtype_mpi][rootdirect_mpi],MPI_STATUSES_IGNORE);CHKERRQ(ierr);
  ierr = MPI_Waitall(sf->nleafreqs, link->leafreqs[direction][leafmtype_mpi][leafdirect_mpi],MPI_STATUSES_IGNORE);CHKERRQ(ierr);
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
PETSC_INTERN PetscErrorCode PetscSFLinkFetchRootData (PetscSF,PetscSFLink,PetscSFScope,void*,MPI_Op);

PETSC_INTERN PetscErrorCode PetscSFLinkBcastAndOpLocal(PetscSF,PetscSFLink,const void*,void*,MPI_Op);
PETSC_INTERN PetscErrorCode PetscSFLinkReduceLocal(PetscSF,PetscSFLink,const void*,void*,MPI_Op);
PETSC_INTERN PetscErrorCode PetscSFLinkFetchAndOpLocal(PetscSF,PetscSFLink,void*,const void*,void*,MPI_Op);

PETSC_INTERN PetscErrorCode PetscSFSetUpPackFields(PetscSF);
PETSC_INTERN PetscErrorCode PetscSFResetPackFields(PetscSF);

#if defined(PETSC_HAVE_CUDA)
PETSC_INTERN PetscErrorCode PetscSFLinkSetUp_Cuda(PetscSF,PetscSFLink,MPI_Datatype);
#endif

#if defined(PETSC_HAVE_KOKKOS)
PETSC_INTERN PetscErrorCode PetscSFLinkSetUp_Kokkos(PetscSF,PetscSFLink,MPI_Datatype);
#endif

/* A set of helper routines for Pack/Unpack/Scatter on GPUs */
#if defined(PETSC_HAVE_DEVICE)
/* If SF does not know which stream root/leafdata is being computed on, it has to sync the device to
   make sure the data is ready for packing.
 */
PETSC_STATIC_INLINE PetscErrorCode PetscSFLinkSyncDeviceBeforePackData(PetscSF sf,PetscSFLink link)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (sf->use_default_stream) PetscFunctionReturn(0);
  if (link->rootmtype == PETSC_MEMTYPE_DEVICE || link->leafmtype == PETSC_MEMTYPE_DEVICE) {ierr = (*link->d_SyncDevice)(link);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/* PetscSFLinkSyncStreamAfterPackXxxData routines make sure root/leafbuf for the remote is ready for MPI */
PETSC_STATIC_INLINE PetscErrorCode PetscSFLinkSyncStreamAfterPackRootData(PetscSF sf,PetscSFLink link)
{
  PetscErrorCode ierr;
  PetscSF_Basic  *bas = (PetscSF_Basic*)sf->data;

  PetscFunctionBegin;
  /* Do nothing if we use stream aware mpi || has nothing for remote */
  if (sf->use_stream_aware_mpi || link->rootmtype != PETSC_MEMTYPE_DEVICE || !bas->rootbuflen[PETSCSF_REMOTE]) PetscFunctionReturn(0);
  /* If we called a packing kernel || we async-copied rootdata from device to host || No cudaDeviceSynchronize was called (since default stream is assumed) */
  if (!link->rootdirect[PETSCSF_REMOTE] || !sf->use_gpu_aware_mpi || sf->use_default_stream) {ierr = (*link->d_SyncStream)(link);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscSFLinkSyncStreamAfterPackLeafData(PetscSF sf,PetscSFLink link)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  /* See comments above */
  if (sf->use_stream_aware_mpi || link->leafmtype != PETSC_MEMTYPE_DEVICE || !sf->leafbuflen[PETSCSF_REMOTE]) PetscFunctionReturn(0);
  if (!link->leafdirect[PETSCSF_REMOTE] || !sf->use_gpu_aware_mpi || sf->use_default_stream) {ierr = (*link->d_SyncStream)(link);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/* PetscSFLinkSyncStreamAfterUnpackXxx routines make sure root/leafdata (local & remote) is ready to use for SF callers, when SF
   does not know which stream the callers will use.
*/
PETSC_STATIC_INLINE PetscErrorCode PetscSFLinkSyncStreamAfterUnpackRootData(PetscSF sf,PetscSFLink link)
{
  PetscErrorCode ierr;
  PetscSF_Basic  *bas = (PetscSF_Basic*)sf->data;
  PetscBool      host2host = (link->rootmtype == PETSC_MEMTYPE_HOST) && (link->leafmtype == PETSC_MEMTYPE_HOST) ? PETSC_TRUE : PETSC_FALSE;

  PetscFunctionBegin;
  /* Do nothing if host2host OR we are allowed to asynchronously put rootdata on device through the default stream */
  if (host2host || (link->rootmtype == PETSC_MEMTYPE_DEVICE && sf->use_default_stream)) PetscFunctionReturn(0);

  /* If rootmtype is HOST or DEVICE:
     If we have data from local, then we called a scatter kernel (on link->stream), then we must sync it;
     If we have data from remote && no rootdirect(i.e., we called an unpack kernel), then we must also sycn it (if rootdirect,
     i.e., no unpack kernel after MPI, MPI guarentees rootbuf is ready to use so that we do not need the sync).

     Note a tricky case is when leafmtype=DEVICE, rootmtype=HOST on uni-processor, we must sync the stream otherwise
     CPU thread might use the yet-to-be-updated rootdata pending in the stream.
   */
  if (bas->rootbuflen[PETSCSF_LOCAL] || (bas->rootbuflen[PETSCSF_REMOTE] && !link->rootdirect[PETSCSF_REMOTE])) {ierr = (*link->d_SyncStream)(link);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscSFLinkSyncStreamAfterUnpackLeafData(PetscSF sf,PetscSFLink link)
{
  PetscErrorCode ierr;
  PetscBool      host2host = (link->rootmtype == PETSC_MEMTYPE_HOST) && (link->leafmtype == PETSC_MEMTYPE_HOST) ? PETSC_TRUE : PETSC_FALSE;

  PetscFunctionBegin;
  /* See comments in PetscSFLinkSyncStreamAfterUnpackRootData*/
  if (host2host || (link->leafmtype == PETSC_MEMTYPE_DEVICE && sf->use_default_stream)) PetscFunctionReturn(0);
  if (sf->leafbuflen[PETSCSF_LOCAL] || (sf->leafbuflen[PETSCSF_REMOTE] && !link->leafdirect[PETSCSF_REMOTE])) {ierr = (*link->d_SyncStream)(link);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/* PetscSFLinkCopyXxxxBufferInCaseNotUseGpuAwareMPI routines are simple: if not use_gpu_aware_mpi, we need
   to copy the buffer from GPU to CPU before MPI calls, and from CPU to GPU after MPI calls.
*/
PETSC_STATIC_INLINE PetscErrorCode PetscSFLinkCopyRootBufferInCaseNotUseGpuAwareMPI(PetscSF sf,PetscSFLink link,PetscBool device2host)
{
  PetscErrorCode ierr;
  PetscSF_Basic  *bas = (PetscSF_Basic*)sf->data;

  PetscFunctionBegin;
  if (link->rootmtype == PETSC_MEMTYPE_DEVICE && (link->rootmtype_mpi != link->rootmtype) && bas->rootbuflen[PETSCSF_REMOTE]) {
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
  if (link->leafmtype == PETSC_MEMTYPE_DEVICE && (link->leafmtype_mpi != link->leafmtype) && sf->leafbuflen[PETSCSF_REMOTE]) {
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

#else /* Host only */
#define PetscSFLinkSyncDeviceBeforePackData(a,b)                0
#define PetscSFLinkSyncStreamAfterPackRootData(a,b)             0
#define PetscSFLinkSyncStreamAfterPackLeafData(a,b)             0
#define PetscSFLinkSyncStreamAfterUnpackRootData(a,b)           0
#define PetscSFLinkSyncStreamAfterUnpackLeafData(a,b)           0
#define PetscSFLinkCopyRootBufferInCaseNotUseGpuAwareMPI(a,b,c) 0
#define PetscSFLinkCopyLeafBufferInCaseNotUseGpuAwareMPI(a,b,c) 0
#endif

/* Get root indices used for pack/unpack

Input arguments:
  +sf    - StarForest
  .link  - The link, which provides the stream for the async memcpy (In SF, we make all GPU operations asynchronous to avoid unexpected pipeline stalls)
  .scope - Which part of the indices? (PETSCSF_LOCAL or PETSCSF_REMOTE)
  .mtype - In what type of memory? (PETSC_MEMTYPE_DEVICE or PETSC_MEMTYPE_HOST)

 Output arguments:
  +count   - Count of indices
  .start   - The first index (only useful when indices is NULL)
  -indices - indices of roots for pack/unpack. NULL means indices are contiguous
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
    if (mtype == PETSC_MEMTYPE_HOST) {*opt = bas->rootpackopt[scope]; *indices = bas->irootloc + offset;}
#if defined(PETSC_HAVE_DEVICE)
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
#endif
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
    if (mtype == PETSC_MEMTYPE_HOST) {*opt = sf->leafpackopt[scope]; *indices = sf->rmine + offset;}
#if defined(PETSC_HAVE_DEVICE)
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
#endif
  }
  PetscFunctionReturn(0);
}
#endif
