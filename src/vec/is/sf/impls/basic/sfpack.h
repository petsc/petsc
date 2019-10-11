#if !defined(__SFPACK_H)
#define __SFPACK_H

#include <petsc/private/sfimpl.h> /*I "petscsf.h" I*/

/* Optimization plans in packing & unpacking for destination ranks.

  Suppose there are count indices stored in idx[], and two addresses u, p. We want to do packing:
     p[i] = u[idx[i]], for i in [0,count)

  Often, the indices are associated with n ranks. Each rank's indices are stored consecutively in idx[].
  We analyze indices for each rank and see if they are patterns that can be used to optimize the packing.
  The result is stored in PetscSFPackOpt. Packing for a rank might be non-optimizable, or optimized into
  a small number of contiguous memory copies or one strided memory copy.
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

typedef struct _n_PetscSFPack* PetscSFPack;

/* An abstract class that defines a communication link, which includes how to pack/unpack data and send/recv buffers
 */
struct _n_PetscSFPack {
  PetscErrorCode (*h_Pack)            (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,const void*,void*);
  PetscErrorCode (*h_UnpackAndInsert) (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*h_UnpackAndAdd)    (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*h_UnpackAndMin)    (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*h_UnpackAndMax)    (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*h_UnpackAndMinloc) (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*h_UnpackAndMaxloc) (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*h_UnpackAndMult)   (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*h_UnpackAndLAND)   (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*h_UnpackAndBAND)   (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*h_UnpackAndLOR)    (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*h_UnpackAndBOR)    (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*h_UnpackAndLXOR)   (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*h_UnpackAndBXOR)   (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*h_FetchAndInsert)  (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,      void*);
  PetscErrorCode (*h_FetchAndAdd)     (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,      void*);
  PetscErrorCode (*h_FetchAndMin)     (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,      void*);
  PetscErrorCode (*h_FetchAndMax)     (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,      void*);
  PetscErrorCode (*h_FetchAndMinloc)  (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,      void*);
  PetscErrorCode (*h_FetchAndMaxloc)  (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,      void*);
  PetscErrorCode (*h_FetchAndMult)    (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,      void*);
  PetscErrorCode (*h_FetchAndLAND)    (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,      void*);
  PetscErrorCode (*h_FetchAndBAND)    (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,      void*);
  PetscErrorCode (*h_FetchAndLOR)     (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,      void*);
  PetscErrorCode (*h_FetchAndBOR)     (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,      void*);
  PetscErrorCode (*h_FetchAndLXOR)    (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,      void*);
  PetscErrorCode (*h_FetchAndBXOR)    (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,      void*);
#if defined(PETSC_HAVE_CUDA)
  /* These fields are lazily initialized in a sense that only when device pointers are passed to an SF, the SF
     will set them, otherwise it just leaves them alone even though PETSC_HAVE_CUDA. Packing routines using
     regular ops when there are no data race chances.
  */
  PetscBool      deviceinited;        /* Are device related fields initialized? */
  PetscErrorCode (*d_Pack)            (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,const void*,void*);

  PetscErrorCode (*d_UnpackAndInsert) (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*d_UnpackAndAdd)    (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*d_UnpackAndMin)    (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*d_UnpackAndMax)    (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*d_UnpackAndMinloc) (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*d_UnpackAndMaxloc) (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*d_UnpackAndMult)   (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*d_UnpackAndLAND)   (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*d_UnpackAndBAND)   (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*d_UnpackAndLOR)    (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*d_UnpackAndBOR)    (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*d_UnpackAndLXOR)   (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*d_UnpackAndBXOR)   (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*d_FetchAndInsert)  (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,      void*);
  PetscErrorCode (*d_FetchAndAdd)     (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,      void*);
  PetscErrorCode (*d_FetchAndMin)     (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,      void*);
  PetscErrorCode (*d_FetchAndMax)     (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,      void*);
  PetscErrorCode (*d_FetchAndMinloc)  (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,      void*);
  PetscErrorCode (*d_FetchAndMaxloc)  (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,      void*);
  PetscErrorCode (*d_FetchAndMult)    (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,      void*);
  PetscErrorCode (*d_FetchAndLAND)    (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,      void*);
  PetscErrorCode (*d_FetchAndBAND)    (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,      void*);
  PetscErrorCode (*d_FetchAndLOR)     (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,      void*);
  PetscErrorCode (*d_FetchAndBOR)     (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,      void*);
  PetscErrorCode (*d_FetchAndLXOR)    (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,      void*);
  PetscErrorCode (*d_FetchAndBXOR)    (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,      void*);

  /* Packing routines using atomics when there are data race chances */
  PetscErrorCode (*da_UnpackAndInsert)(PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*da_UnpackAndAdd)   (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*da_UnpackAndMin)   (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*da_UnpackAndMax)   (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*da_UnpackAndMinloc)(PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*da_UnpackAndMaxloc)(PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*da_UnpackAndMult)  (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*da_UnpackAndLAND)  (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*da_UnpackAndBAND)  (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*da_UnpackAndLOR)   (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*da_UnpackAndBOR)   (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*da_UnpackAndLXOR)  (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*da_UnpackAndBXOR)  (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,const void*);
  PetscErrorCode (*da_FetchAndInsert) (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,      void*);
  PetscErrorCode (*da_FetchAndAdd)    (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,      void*);
  PetscErrorCode (*da_FetchAndMin)    (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,      void*);
  PetscErrorCode (*da_FetchAndMax)    (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,      void*);
  PetscErrorCode (*da_FetchAndMinloc) (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,      void*);
  PetscErrorCode (*da_FetchAndMaxloc) (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,      void*);
  PetscErrorCode (*da_FetchAndMult)   (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,      void*);
  PetscErrorCode (*da_FetchAndLAND)   (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,      void*);
  PetscErrorCode (*da_FetchAndBAND)   (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,      void*);
  PetscErrorCode (*da_FetchAndLOR)    (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,      void*);
  PetscErrorCode (*da_FetchAndBOR)    (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,      void*);
  PetscErrorCode (*da_FetchAndLXOR)   (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,      void*);
  PetscErrorCode (*da_FetchAndBXOR)   (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,      void*);

  PetscInt       MAX_CORESIDENT_THREADS; /* It is a copy from SF for convenience. */
  cudaStream_t   stream;                 /* Stream to launch pack/unapck kernels if not using the default stream */
#endif
  PetscMPIInt    tag;                    /* Each link has a tag so we can perform multiple SF ops at the same time */
  MPI_Datatype   unit;                   /* The MPI datatype this PetscSFPack is built for */
  MPI_Datatype   basicunit;              /* unit is made of MPI builtin dataype basicunit */
  PetscBool      isbuiltin;              /* Is unit an MPI/PETSc builtin datatype? If it is true, basicunit=unit, bs=1 */
  size_t         unitbytes;              /* Number of bytes in a unit */
  PetscInt       bs;                     /* Number of basic units in a unit */
  const void     *rkey,*lkey;            /* rootdata and leafdata used as keys for operation */
  char           *rootbuf[2];            /* Buffer for packed roots on Host (0 or PETSC_MEMTYPE_HOST) or Device (1 or PETSC_MEMTYPE_DEVICE) */
  char           *leafbuf[2];            /* Buffer for packed leaves on Host (0) or Device (1) */
  char           *selfbuf[2];            /* Buffer for roots in self to self communication on Host (0) or Device (1) */
  PetscInt       rootbuflen;             /* Length of root buffer in <unit> */
  PetscInt       leafbuflen;             /* Length of leaf buffer in <unit> */
  PetscInt       selfbuflen;             /* Length of self buffer in <unit> */
  PetscMemType   rootmtype;              /* rootdata's memory type */
  PetscMemType   leafmtype;              /* leafdata's memory type */
  PetscMPIInt    nrootreqs;              /* Number of root requests */
  PetscMPIInt    nleafreqs;              /* Number of leaf requests */
  MPI_Request    *rootreqs[2][2];        /* Pointers to root requests in this layout [PETSCSF_DIRECTION][PETSC_MEMTYPE] */
  MPI_Request    *leafreqs[2][2];        /* Pointers to leaf requests in this layout [PETSCSF_DIRECTION][PETSC_MEMTYPE] */
  PetscBool      rootreqsinited[2][2];   /* Are root requests initialized? Also in layout of [PETSCSF_DIRECTION][PETSC_MEMTYPE]*/
  PetscBool      leafreqsinited[2][2];   /* Are leaf requests initialized? Also in layout of [PETSCSF_DIRECTION][PETSC_MEMTYPE]*/
  MPI_Request    *reqs;                  /* An array of length (nrootreqs+nleafreqs)*4. Pointers in rootreqs[][] and leafreqs[][] point here */
  PetscSFPack    next;
};

PETSC_INTERN PetscErrorCode PetscSFPackGetInUse(PetscSF,MPI_Datatype,const void*,const void*,PetscCopyMode,PetscSFPack*);
PETSC_INTERN PetscErrorCode PetscSFPackReclaim(PetscSF,PetscSFPack*);
PETSC_INTERN PetscErrorCode PetscSFPackDestoryAvailable(PetscSFPack*);
PETSC_STATIC_INLINE PetscErrorCode PetscSFPackGetPack(PetscSFPack link,PetscMemType mtype,PetscErrorCode (**Pack)(PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,const void*,void*))
{
  PetscFunctionBegin;
  *Pack = NULL;
  if (mtype == PETSC_MEMTYPE_HOST)        *Pack = link->h_Pack;
#if defined(PETSC_HAVE_CUDA)
  else if (mtype == PETSC_MEMTYPE_DEVICE) *Pack = link->d_Pack;
#endif
  else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Wrong PetscMemType %D",mtype);
  PetscFunctionReturn(0);
}
PETSC_INTERN PetscErrorCode PetscSFPackGetUnpackAndOp(PetscSFPack,PetscMemType,MPI_Op,PetscBool,PetscErrorCode (**UnpackAndOp)(PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,const void*));
PETSC_INTERN PetscErrorCode PetscSFPackGetFetchAndOp (PetscSFPack,PetscMemType,MPI_Op,PetscBool,PetscErrorCode (**FetchAndOp) (PetscInt,const PetscInt*,PetscSFPack,PetscSFPackOpt,void*,void*));
PETSC_INTERN PetscErrorCode PetscSFPackSetErrorOnUnsupportedOverlap(PetscSF,MPI_Datatype,const void*,const void*);
PETSC_INTERN PetscErrorCode PetscSFPackSetUp_Host(PetscSF,PetscSFPack,MPI_Datatype);
#if defined(PETSC_HAVE_CUDA)
PETSC_INTERN PetscErrorCode PetscSFPackSetUp_Device(PetscSF,PetscSFPack,MPI_Datatype);
#endif
PETSC_INTERN PetscErrorCode PetscSFPackOptCreate(PetscInt,const PetscInt*,const PetscInt*,PetscSFPackOpt*);
PETSC_INTERN PetscErrorCode PetscSFPackOptDestory(PetscSFPackOpt *out);
#endif
