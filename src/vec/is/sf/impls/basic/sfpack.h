#if !defined(__SFPACK_H)
#define __SFPACK_H

#include <petsc/private/sfimpl.h> /*I "petscsf.h" I*/

/* Optimization plans in packing & unpacking for destination ranks.

  Suppose there are count indices stored in idx[], and two addresses u, p. We want to do packing:
     p[i] = u[idx[i]], for i in [0,count)

  Often, the indices are associated with n ranks. Each rank's indices are stored consecutively in idx[].
  We analyze indices for each rank and see if they are patterns that can be used to optimize the packing.
  The result is stored in PetscSFPackOpt. Packing for a rank might be not optimizable, or optimized in
  to a small number of contiguous memory copies or one strided memory copy.
 */
typedef enum {PETSCSF_PACKOPT_NONE=0, PETSCSF_PACKOPT_MULTICOPY, PETSCSF_PACKOPT_STRIDE} PetscSFPackOptType;

struct _n_PetscSFPackOpt {
  PetscInt           n;             /* Number of destination ranks */
  PetscSFPackOptType *type;         /* [n] Optimization types for the n ranks */
  PetscInt           *offset;       /* [n+1] Indices for i-th rank are in [offset[i],offset[i+1]) of idx[] */
  PetscInt           *copy_offset;  /* [n+1] If type[i] = PETSCSF_PACKOPT_MULTICOPY, packing for i-th rank is optimized into copies numbered between [copy_offset[i],copy_offset[i+1]) */
  PetscInt           *copy_start;   /* [*]     j-th copy starts at copy_start[j] in idx[]. In other words, there are copy_length[j] contiguous indices */
  PetscInt           *copy_length;  /* [*]     starting from idx[copy_start[j]] */
  PetscInt           *stride_step;  /* [n]   If type[i] = PETSCSF_PACKOPT_STRIDE, then packing for i-th rank is strided, with first index being idx[offset[i]] and step stride_step[i], */
  PetscInt           *stride_n;     /* [n]     and total stride_n[i] steps */
};

typedef struct _n_PetscSFPack* PetscSFPack;

#define SFPACKHEADER \
  PetscErrorCode (*Pack)           (PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,const void*,void*);  \
  PetscErrorCode (*UnpackAndInsert)(PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,void*,const void*);  \
  PetscErrorCode (*UnpackAndAdd)   (PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,void*,const void*);  \
  PetscErrorCode (*UnpackAndMin)   (PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,void*,const void*);  \
  PetscErrorCode (*UnpackAndMax)   (PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,void*,const void*);  \
  PetscErrorCode (*UnpackAndMinloc)(PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,void*,const void*);  \
  PetscErrorCode (*UnpackAndMaxloc)(PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,void*,const void*);  \
  PetscErrorCode (*UnpackAndMult)  (PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,void*,const void*);  \
  PetscErrorCode (*UnpackAndLAND)  (PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,void*,const void*);  \
  PetscErrorCode (*UnpackAndBAND)  (PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,void*,const void*);  \
  PetscErrorCode (*UnpackAndLOR)   (PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,void*,const void*);  \
  PetscErrorCode (*UnpackAndBOR)   (PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,void*,const void*);  \
  PetscErrorCode (*UnpackAndLXOR)  (PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,void*,const void*);  \
  PetscErrorCode (*UnpackAndBXOR)  (PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,void*,const void*);  \
  PetscErrorCode (*FetchAndInsert) (PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,void*,void*);        \
  PetscErrorCode (*FetchAndAdd)    (PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,void*,void*);        \
  PetscErrorCode (*FetchAndMin)    (PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,void*,void*);        \
  PetscErrorCode (*FetchAndMax)    (PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,void*,void*);        \
  PetscErrorCode (*FetchAndMinloc) (PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,void*,void*);        \
  PetscErrorCode (*FetchAndMaxloc) (PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,void*,void*);        \
  PetscErrorCode (*FetchAndMult)   (PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,void*,void*);        \
  PetscErrorCode (*FetchAndLAND)   (PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,void*,void*);        \
  PetscErrorCode (*FetchAndBAND)   (PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,void*,void*);        \
  PetscErrorCode (*FetchAndLOR)    (PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,void*,void*);        \
  PetscErrorCode (*FetchAndBOR)    (PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,void*,void*);        \
  PetscErrorCode (*FetchAndLXOR)   (PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,void*,void*);        \
  PetscErrorCode (*FetchAndBXOR)   (PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,void*,void*);        \
  PetscMPIInt    tag;         /* Each link has a tag so we can perform multiple SF ops at the same time */\
  MPI_Datatype   unit;                                                                                    \
  MPI_Datatype   basicunit;   /* unit is made of MPI builtin dataype basicunit */                         \
  PetscBool      isbuiltin;   /* Is unit an MPI builtin datatype? If it is true, basicunit=unit, bs=1 */  \
  size_t         unitbytes;   /* Number of bytes in a unit */                                             \
  PetscInt       bs;          /* Number of basic units in a unit */                                       \
  const void     *rkey,*lkey; /* rootdata and leafdata used as keys for operation */                      \
  char           *rootbuf;       /* Buffer for packed roots in send/recv */                                                         \
  char           *leafbuf;       /* Buffer for packed leaves in send/recv */                                                        \
  char           *selfbuf;       /* If self communication does not use MPI, this is the shared buffer for packed roots or leaves */ \
  PetscSFPack    next

/* An abstract class that defines a communication link, which includes how to
   pack/unpack data. Subclasses may further contain fields for send/recv buffers,
   MPI_Requests etc used in communication.
 */
struct _n_PetscSFPack {
  SFPACKHEADER;
};

PETSC_INTERN PetscErrorCode PetscSFPackGetInUse(PetscSF,MPI_Datatype,const void*,const void*,PetscCopyMode,PetscSFPack*);
PETSC_INTERN PetscErrorCode PetscSFPackReclaim(PetscSF,PetscSFPack*);
PETSC_INTERN PetscErrorCode PetscSFPackSetupType(PetscSFPack,MPI_Datatype);
PETSC_INTERN PetscErrorCode PetscSFPackGetUnpackAndOp(PetscSF,PetscSFPack,MPI_Op,PetscErrorCode (**UnpackAndOp)(PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,void*,const void*));
PETSC_INTERN PetscErrorCode PetscSFPackGetFetchAndOp (PetscSF,PetscSFPack,MPI_Op,PetscErrorCode (**FetchAndOp) (PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,void*,      void*));
PETSC_INTERN PetscErrorCode PetscSFPackSetupOptimization(PetscInt,const PetscInt*,const PetscInt*,PetscSFPackOpt*);
PETSC_INTERN PetscErrorCode PetscSFPackDestoryOptimization(PetscSFPackOpt *out);
PETSC_INTERN PetscErrorCode PetscSFPackSetErrorOnUnsupportedOverlap(PetscSF,MPI_Datatype,const void*,const void*);

#endif
