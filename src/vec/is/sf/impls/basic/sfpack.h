#if !defined(__SFPACK_H)
#define __SFPACK_H

#include <../src/vec/is/sf/impls/basic/sfbasic.h>

/* We separate SF communications for SFBasic and SFNeighbor in two parts: local (self,intra-rank) and remote (inter-rank) */
typedef enum {PETSCSF_LOCAL=0, PETSCSF_REMOTE} PetscSFScope;

/* Optimization plans in packing & unpacking for destination ranks.

  Suppose there are count indices stored in idx[], and two addresses u, p. We want to do packing:
     p[i] = u[idx[i]], for i in [0,count)

  Often, the indices are associated with n ranks. Each rank's indices are stored consecutively in idx[].
  We analyze indices for each rank and see if they are patterns that can be used to optimize the packing.
  The result is stored in PetscSFPackOpt. Packing for a rank might be non-optimizable, or optimized into
  a small number of contiguous memory copies or one strided memory copy.

  Note before using this per-rank optimization, one should check rleafloccontig, irootloccontig, which say
  indices in whole start from 0 and are contiguous, and therefore much useful when true.
 */
typedef enum {PETSCSF_PACKOPT_NONE=0, PETSCSF_PACKOPT_MULTICOPY, PETSCSF_PACKOPT_STRIDE} PetscSFPackOptType;
struct _n_PetscSFPackOpt {
  PetscInt           n;             /* Number of destination ranks */
  PetscSFPackOptType *type;         /* [n] Optimization types for the n ranks */
  PetscInt           *offset;       /* [n+1] Indices for i-th rank are in [offset[i],offset[i+1]) of idx[] */
  PetscInt           *copy_offset;  /* [n+1] If type[i] = PETSCSF_PACKOPT_MULTICOPY, packing for i-th rank is optimized into copies numbered between [copy_offset[i],copy_offset[i+1]) */
  PetscInt           *copy_start;   /* [*]     j-th copy starts at copy_start[j] in idx[]. In other words, there are copy_length[j] contiguous indices */
  PetscInt           *copy_length;  /* [*]     starting at idx[copy_start[j]] */
  PetscInt           *stride_step;  /* [n]   If type[i] = PETSCSF_PACKOPT_STRIDE, then packing for i-th rank is strided, with first index being idx[offset[i]] and step stride_step[i], */
  PetscInt           *stride_n;     /* [n]     and total stride_n[i] steps */
};

/* An abstract class that defines a communication link, which includes how to pack/unpack data and send/recv buffers
 */
struct _n_PetscSFLink {                /* link,    count,  start,   indices,      pack plan, root/leafdata, buffer*/
  PetscErrorCode (*h_Pack)            (PetscSFLink,PetscInt,PetscInt,const PetscInt*,PetscSFPackOpt,const void*,void*);
  PetscErrorCode (*h_UnpackAndInsert) (PetscSFLink,PetscInt,PetscInt,const PetscInt*,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*h_UnpackAndAdd)    (PetscSFLink,PetscInt,PetscInt,const PetscInt*,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*h_UnpackAndMin)    (PetscSFLink,PetscInt,PetscInt,const PetscInt*,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*h_UnpackAndMax)    (PetscSFLink,PetscInt,PetscInt,const PetscInt*,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*h_UnpackAndMinloc) (PetscSFLink,PetscInt,PetscInt,const PetscInt*,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*h_UnpackAndMaxloc) (PetscSFLink,PetscInt,PetscInt,const PetscInt*,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*h_UnpackAndMult)   (PetscSFLink,PetscInt,PetscInt,const PetscInt*,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*h_UnpackAndLAND)   (PetscSFLink,PetscInt,PetscInt,const PetscInt*,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*h_UnpackAndBAND)   (PetscSFLink,PetscInt,PetscInt,const PetscInt*,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*h_UnpackAndLOR)    (PetscSFLink,PetscInt,PetscInt,const PetscInt*,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*h_UnpackAndBOR)    (PetscSFLink,PetscInt,PetscInt,const PetscInt*,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*h_UnpackAndLXOR)   (PetscSFLink,PetscInt,PetscInt,const PetscInt*,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*h_UnpackAndBXOR)   (PetscSFLink,PetscInt,PetscInt,const PetscInt*,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*h_FetchAndAdd)     (PetscSFLink,PetscInt,PetscInt,const PetscInt*,PetscSFPackOpt,void*,      void*);
                                      /*     link,   count, startx,            idx,      xdata,  starty,         idy,   ydata */
  PetscErrorCode (*h_ScatterAndInsert)(PetscSFLink,PetscInt,PetscInt,const PetscInt*,const void*,PetscInt,const PetscInt*,void*);
  PetscErrorCode (*h_ScatterAndAdd)   (PetscSFLink,PetscInt,PetscInt,const PetscInt*,const void*,PetscInt,const PetscInt*,void*);
  PetscErrorCode (*h_ScatterAndMin)   (PetscSFLink,PetscInt,PetscInt,const PetscInt*,const void*,PetscInt,const PetscInt*,void*);
  PetscErrorCode (*h_ScatterAndMax)   (PetscSFLink,PetscInt,PetscInt,const PetscInt*,const void*,PetscInt,const PetscInt*,void*);
  PetscErrorCode (*h_ScatterAndMinloc)(PetscSFLink,PetscInt,PetscInt,const PetscInt*,const void*,PetscInt,const PetscInt*,void*);
  PetscErrorCode (*h_ScatterAndMaxloc)(PetscSFLink,PetscInt,PetscInt,const PetscInt*,const void*,PetscInt,const PetscInt*,void*);
  PetscErrorCode (*h_ScatterAndMult)  (PetscSFLink,PetscInt,PetscInt,const PetscInt*,const void*,PetscInt,const PetscInt*,void*);
  PetscErrorCode (*h_ScatterAndLAND)  (PetscSFLink,PetscInt,PetscInt,const PetscInt*,const void*,PetscInt,const PetscInt*,void*);
  PetscErrorCode (*h_ScatterAndBAND)  (PetscSFLink,PetscInt,PetscInt,const PetscInt*,const void*,PetscInt,const PetscInt*,void*);
  PetscErrorCode (*h_ScatterAndLOR)   (PetscSFLink,PetscInt,PetscInt,const PetscInt*,const void*,PetscInt,const PetscInt*,void*);
  PetscErrorCode (*h_ScatterAndBOR)   (PetscSFLink,PetscInt,PetscInt,const PetscInt*,const void*,PetscInt,const PetscInt*,void*);
  PetscErrorCode (*h_ScatterAndLXOR)  (PetscSFLink,PetscInt,PetscInt,const PetscInt*,const void*,PetscInt,const PetscInt*,void*);
  PetscErrorCode (*h_ScatterAndBXOR)  (PetscSFLink,PetscInt,PetscInt,const PetscInt*,const void*,PetscInt,const PetscInt*,void*);
                                       /* link, count,rootstart,rootindices,rootdata,leafstart,leafindices,leafdata,leafupdate*/
  PetscErrorCode (*h_FetchAndAddLocal)(PetscSFLink,PetscInt,PetscInt,const PetscInt*,void*,PetscInt,const PetscInt*,const void*,void*);

  PetscBool      deviceinited;        /* Are device related fields initialized? */
#if defined(PETSC_HAVE_CUDA)
  /* These fields are lazily initialized in a sense that only when device pointers are passed to an SF, the SF
     will set them, otherwise it just leaves them alone even though PETSC_HAVE_CUDA. Packing routines using
     regular ops when there are no data race chances.
  */
  PetscErrorCode (*d_Pack)            (PetscSFLink,PetscInt,PetscInt,const PetscInt*,PetscSFPackOpt,const void*,void*);

  PetscErrorCode (*d_UnpackAndInsert) (PetscSFLink,PetscInt,PetscInt,const PetscInt*,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*d_UnpackAndAdd)    (PetscSFLink,PetscInt,PetscInt,const PetscInt*,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*d_UnpackAndMin)    (PetscSFLink,PetscInt,PetscInt,const PetscInt*,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*d_UnpackAndMax)    (PetscSFLink,PetscInt,PetscInt,const PetscInt*,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*d_UnpackAndMinloc) (PetscSFLink,PetscInt,PetscInt,const PetscInt*,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*d_UnpackAndMaxloc) (PetscSFLink,PetscInt,PetscInt,const PetscInt*,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*d_UnpackAndMult)   (PetscSFLink,PetscInt,PetscInt,const PetscInt*,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*d_UnpackAndLAND)   (PetscSFLink,PetscInt,PetscInt,const PetscInt*,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*d_UnpackAndBAND)   (PetscSFLink,PetscInt,PetscInt,const PetscInt*,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*d_UnpackAndLOR)    (PetscSFLink,PetscInt,PetscInt,const PetscInt*,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*d_UnpackAndBOR)    (PetscSFLink,PetscInt,PetscInt,const PetscInt*,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*d_UnpackAndLXOR)   (PetscSFLink,PetscInt,PetscInt,const PetscInt*,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*d_UnpackAndBXOR)   (PetscSFLink,PetscInt,PetscInt,const PetscInt*,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*d_FetchAndAdd)     (PetscSFLink,PetscInt,PetscInt,const PetscInt*,PetscSFPackOpt,void*,      void*);

  PetscErrorCode (*d_ScatterAndInsert)(PetscSFLink,PetscInt,PetscInt,const PetscInt*,const void*,PetscInt,const PetscInt*,void*);
  PetscErrorCode (*d_ScatterAndAdd)   (PetscSFLink,PetscInt,PetscInt,const PetscInt*,const void*,PetscInt,const PetscInt*,void*);
  PetscErrorCode (*d_ScatterAndMin)   (PetscSFLink,PetscInt,PetscInt,const PetscInt*,const void*,PetscInt,const PetscInt*,void*);
  PetscErrorCode (*d_ScatterAndMax)   (PetscSFLink,PetscInt,PetscInt,const PetscInt*,const void*,PetscInt,const PetscInt*,void*);
  PetscErrorCode (*d_ScatterAndMinloc)(PetscSFLink,PetscInt,PetscInt,const PetscInt*,const void*,PetscInt,const PetscInt*,void*);
  PetscErrorCode (*d_ScatterAndMaxloc)(PetscSFLink,PetscInt,PetscInt,const PetscInt*,const void*,PetscInt,const PetscInt*,void*);
  PetscErrorCode (*d_ScatterAndMult)  (PetscSFLink,PetscInt,PetscInt,const PetscInt*,const void*,PetscInt,const PetscInt*,void*);
  PetscErrorCode (*d_ScatterAndLAND)  (PetscSFLink,PetscInt,PetscInt,const PetscInt*,const void*,PetscInt,const PetscInt*,void*);
  PetscErrorCode (*d_ScatterAndBAND)  (PetscSFLink,PetscInt,PetscInt,const PetscInt*,const void*,PetscInt,const PetscInt*,void*);
  PetscErrorCode (*d_ScatterAndLOR)   (PetscSFLink,PetscInt,PetscInt,const PetscInt*,const void*,PetscInt,const PetscInt*,void*);
  PetscErrorCode (*d_ScatterAndBOR)   (PetscSFLink,PetscInt,PetscInt,const PetscInt*,const void*,PetscInt,const PetscInt*,void*);
  PetscErrorCode (*d_ScatterAndLXOR)  (PetscSFLink,PetscInt,PetscInt,const PetscInt*,const void*,PetscInt,const PetscInt*,void*);
  PetscErrorCode (*d_ScatterAndBXOR)  (PetscSFLink,PetscInt,PetscInt,const PetscInt*,const void*,PetscInt,const PetscInt*,void*);
  PetscErrorCode (*d_FetchAndAddLocal)(PetscSFLink,PetscInt,PetscInt,const PetscInt*,void*,PetscInt,const PetscInt*,const void*,void*);

  /* Packing routines using atomics when there are data race chances */
  PetscErrorCode (*da_UnpackAndInsert)(PetscSFLink,PetscInt,PetscInt,const PetscInt*,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*da_UnpackAndAdd)   (PetscSFLink,PetscInt,PetscInt,const PetscInt*,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*da_UnpackAndMin)   (PetscSFLink,PetscInt,PetscInt,const PetscInt*,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*da_UnpackAndMax)   (PetscSFLink,PetscInt,PetscInt,const PetscInt*,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*da_UnpackAndMinloc)(PetscSFLink,PetscInt,PetscInt,const PetscInt*,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*da_UnpackAndMaxloc)(PetscSFLink,PetscInt,PetscInt,const PetscInt*,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*da_UnpackAndMult)  (PetscSFLink,PetscInt,PetscInt,const PetscInt*,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*da_UnpackAndLAND)  (PetscSFLink,PetscInt,PetscInt,const PetscInt*,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*da_UnpackAndBAND)  (PetscSFLink,PetscInt,PetscInt,const PetscInt*,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*da_UnpackAndLOR)   (PetscSFLink,PetscInt,PetscInt,const PetscInt*,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*da_UnpackAndBOR)   (PetscSFLink,PetscInt,PetscInt,const PetscInt*,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*da_UnpackAndLXOR)  (PetscSFLink,PetscInt,PetscInt,const PetscInt*,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*da_UnpackAndBXOR)  (PetscSFLink,PetscInt,PetscInt,const PetscInt*,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*da_FetchAndAdd)    (PetscSFLink,PetscInt,PetscInt,const PetscInt*,PetscSFPackOpt,void*,      void*);

  PetscErrorCode (*da_ScatterAndInsert)(PetscSFLink,PetscInt,PetscInt,const PetscInt*,const void*,PetscInt,const PetscInt*,void*);
  PetscErrorCode (*da_ScatterAndAdd)   (PetscSFLink,PetscInt,PetscInt,const PetscInt*,const void*,PetscInt,const PetscInt*,void*);
  PetscErrorCode (*da_ScatterAndMin)   (PetscSFLink,PetscInt,PetscInt,const PetscInt*,const void*,PetscInt,const PetscInt*,void*);
  PetscErrorCode (*da_ScatterAndMax)   (PetscSFLink,PetscInt,PetscInt,const PetscInt*,const void*,PetscInt,const PetscInt*,void*);
  PetscErrorCode (*da_ScatterAndMinloc)(PetscSFLink,PetscInt,PetscInt,const PetscInt*,const void*,PetscInt,const PetscInt*,void*);
  PetscErrorCode (*da_ScatterAndMaxloc)(PetscSFLink,PetscInt,PetscInt,const PetscInt*,const void*,PetscInt,const PetscInt*,void*);
  PetscErrorCode (*da_ScatterAndMult)  (PetscSFLink,PetscInt,PetscInt,const PetscInt*,const void*,PetscInt,const PetscInt*,void*);
  PetscErrorCode (*da_ScatterAndLAND)  (PetscSFLink,PetscInt,PetscInt,const PetscInt*,const void*,PetscInt,const PetscInt*,void*);
  PetscErrorCode (*da_ScatterAndBAND)  (PetscSFLink,PetscInt,PetscInt,const PetscInt*,const void*,PetscInt,const PetscInt*,void*);
  PetscErrorCode (*da_ScatterAndLOR)   (PetscSFLink,PetscInt,PetscInt,const PetscInt*,const void*,PetscInt,const PetscInt*,void*);
  PetscErrorCode (*da_ScatterAndBOR)   (PetscSFLink,PetscInt,PetscInt,const PetscInt*,const void*,PetscInt,const PetscInt*,void*);
  PetscErrorCode (*da_ScatterAndLXOR)  (PetscSFLink,PetscInt,PetscInt,const PetscInt*,const void*,PetscInt,const PetscInt*,void*);
  PetscErrorCode (*da_ScatterAndBXOR)  (PetscSFLink,PetscInt,PetscInt,const PetscInt*,const void*,PetscInt,const PetscInt*,void*);
  PetscErrorCode (*da_FetchAndAddLocal)(PetscSFLink,PetscInt,PetscInt,const PetscInt*,void*,PetscInt,const PetscInt*,const void*,void*);

  PetscInt     maxResidentThreadsPerGPU;     /* It is a copy from SF for convenience */
  cudaStream_t stream;                       /* Stream to launch pack/unapck kernels if not using the default stream */
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

#if defined(PETSC_USE_DEBUG)
PETSC_INTERN PetscErrorCode PetscSFSetErrorOnUnsupportedOverlap(PetscSF,MPI_Datatype,const void*,const void*);
#else
#define PetscSFSetErrorOnUnsupportedOverlap(a,b,c,d) 0
#endif

/* Create/setup/retrieve/destroy a link */
PETSC_INTERN PetscErrorCode PetscSFLinkCreate(PetscSF,MPI_Datatype,PetscMemType,const void*,PetscMemType,const void*,MPI_Op,PetscSFOperation,PetscSFLink*);
PETSC_INTERN PetscErrorCode PetscSFLinkSetUp_Host(PetscSF,PetscSFLink,MPI_Datatype);
#if defined(PETSC_HAVE_CUDA)
PETSC_INTERN PetscErrorCode PetscSFLinkSetUp_Device(PetscSF,PetscSFLink,MPI_Datatype);
#else
#define PetscSFLinkSetUp_Device(a,b,c) 0
#endif
PETSC_INTERN PetscErrorCode PetscSFLinkGetInUse(PetscSF,MPI_Datatype,const void*,const void*,PetscCopyMode,PetscSFLink*);
PETSC_INTERN PetscErrorCode PetscSFLinkReclaim(PetscSF,PetscSFLink*);
PETSC_INTERN PetscErrorCode PetscSFLinkDestroy(PetscSF,PetscSFLink*);

/* Get pack/unpack function pointers from a link */
PETSC_STATIC_INLINE PetscErrorCode PetscSFLinkGetPack(PetscSFLink link,PetscMemType mtype,PetscErrorCode (**Pack)(PetscSFLink,PetscInt,PetscInt,const PetscInt*,PetscSFPackOpt,const void*,void*))
{
  PetscFunctionBegin;
  if (mtype == PETSC_MEMTYPE_HOST) *Pack = link->h_Pack;
#if defined(PETSC_HAVE_CUDA)
  else *Pack = link->d_Pack;
#endif
  PetscFunctionReturn(0);
}
PETSC_INTERN PetscErrorCode PetscSFLinkGetUnpackAndOp(PetscSFLink,PetscMemType,MPI_Op,PetscBool,PetscErrorCode (**UnpackAndOp)(PetscSFLink,PetscInt,PetscInt,const PetscInt*,PetscSFPackOpt,void*,const void*));
PETSC_INTERN PetscErrorCode PetscSFLinkGetFetchAndOp (PetscSFLink,PetscMemType,MPI_Op,PetscBool,PetscErrorCode (**FetchAndOp) (PetscSFLink,PetscInt,PetscInt,const PetscInt*,PetscSFPackOpt,void*,void*));
PETSC_INTERN PetscErrorCode PetscSFLinkGetScatterAndOp(PetscSFLink,PetscMemType,MPI_Op,PetscBool,PetscErrorCode (**ScatterAndOp)(PetscSFLink,PetscInt,PetscInt,const PetscInt*,const void*,PetscInt,const PetscInt*,void*));
PETSC_INTERN PetscErrorCode PetscSFLinkGetFetchAndOpLocal(PetscSFLink,PetscMemType,MPI_Op,PetscBool,PetscErrorCode (**FetchAndOpLocal)(PetscSFLink,PetscInt,PetscInt,const PetscInt*,void*,PetscInt,const PetscInt*,const void*,void*));
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

PETSC_INTERN PetscErrorCode PetscSFSetUpPackFields(PetscSF sf);
PETSC_INTERN PetscErrorCode PetscSFResetPackFields(PetscSF sf);
PETSC_INTERN PetscErrorCode PetscSFCreatePackOpt(PetscInt,const PetscInt*,const PetscInt*,PetscSFPackOpt*);
PETSC_INTERN PetscErrorCode PetscSFDestroyPackOpt(PetscSFPackOpt *out);


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
PETSC_STATIC_INLINE PetscErrorCode PetscSFLinkGetRootIndices(PetscSF sf,PetscSFLink link,PetscMemType mtype,PetscSFScope scope,PetscInt *count,PetscInt *start,const PetscInt **indices)
{
  PetscSF_Basic  *bas = (PetscSF_Basic*)sf->data;
  PetscInt       offset;

  PetscFunctionBegin;
  if (count) *count = bas->rootbuflen[scope];
  if (start) *start = bas->rootstart[scope];
  if (indices) {
    if (bas->rootcontig[scope]) *indices = NULL;
    else {
      offset = (scope == PETSCSF_LOCAL)? 0 : bas->ioffset[bas->ndiranks];
      if (mtype == PETSC_MEMTYPE_HOST) {*indices = bas->irootloc + offset;}
#if defined(PETSC_HAVE_CUDA)
      else {
        if (!bas->irootloc_d[scope]) {
          cudaError_t    cerr;
          size_t         size = bas->rootbuflen[scope]*sizeof(PetscInt);
          cerr = cudaMalloc((void **)&bas->irootloc_d[scope],size);CHKERRCUDA(cerr);
          cerr = cudaMemcpyAsync(bas->irootloc_d[scope],bas->irootloc+offset,size,cudaMemcpyHostToDevice,link->stream);CHKERRCUDA(cerr);
        }
        *indices = bas->irootloc_d[scope];
      }
#endif
    }
  }
  PetscFunctionReturn(0);
}

/* Get leaf indices used for pack/unpack

  See also PetscSFLinkGetRootIndices()
 */
PETSC_STATIC_INLINE PetscErrorCode PetscSFLinkGetLeafIndices(PetscSF sf,PetscSFLink link,PetscMemType mtype,PetscSFScope scope,PetscInt *count,PetscInt *start,const PetscInt **indices)
{
  PetscInt   offset;

  PetscFunctionBegin;
  if (count) *count = sf->leafbuflen[scope];
  if (start) *start = sf->leafstart[scope];
  if (indices) {
    if (sf->leafcontig[scope]) *indices = NULL;
    else {
      offset = (scope == PETSCSF_LOCAL)? 0 : sf->roffset[sf->ndranks];
      if (mtype == PETSC_MEMTYPE_HOST) {*indices = sf->rmine + offset;}
  #if defined(PETSC_HAVE_CUDA)
      else {
        if (!sf->rmine_d[scope]) {
          cudaError_t  cerr;
          size_t       size = sf->leafbuflen[scope]*sizeof(PetscInt);
          cerr = cudaMalloc((void **)&sf->rmine_d[scope],size);CHKERRCUDA(cerr);
          cerr = cudaMemcpyAsync(sf->rmine_d[scope],sf->rmine+offset,size,cudaMemcpyHostToDevice,link->stream);CHKERRCUDA(cerr);
        }
        *indices = sf->rmine_d[scope];
      }
  #endif
    }
  }
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

#endif
