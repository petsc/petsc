#if !defined(__SFPACK_H)
#define __SFPACK_H

#include <petsc/private/sfimpl.h> /*I "petscsf.h" I*/

/* Optimization plans in packing(unpacking) for destination ranks.

   Indirect accesses in packing like p[i] = u[idx[i]] are expensive and are not vectorization friendly. We
   try to optimize them if we found cenrtain patterns among indices in idx[]. As a result, a pack might be
   optimized into 1) a small number of contiguous memory copies; OR 2) one strided memory copy.

   Each target has its own plan. n, the number of destination ranks, is nranks or niranks depending on the context.
 */
struct _n_PetscSFPackOpt {
  PetscInt  n;               /*  The number of destination ranks */
  PetscBool *optimized;      /* [n]   Is the packing to i-th rank optimized? If yes, other fields give the opt plan */
  PetscInt  *copy_offset;    /* [n+1] We number all memory copies. Packing for i-th rank is optimized into copies in [copy_offset[i],copy_offset[i+1]) */
  PetscInt  *copy_start;     /* [*]   j-th copy starts at index copy_start[j] */
  PetscInt  *copy_length;    /* [*]     with length copy_length[j] in unit of the <unit> used in for example, PetscSFReduceBegin(sf,unit,...) */
  PetscInt  *stride_first;   /* [n]   If optimized[i] is TRUE but copy_offset[i] == copy_offset[i+1], then packing for i-th rank is strided. The first */
  PetscInt  *stride_step;    /* [n]     index is stride_first[i], step is stride_step[i], */
  PetscInt  *stride_n;       /* [n]     and total stride_n[i] steps */
};

typedef struct _n_PetscSFPack* PetscSFPack;

#define SFPACKHEADER \
  PetscErrorCode (*Pack)           (PetscInt,PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,const void*,void*);  \
  PetscErrorCode (*UnpackAndInsert)(PetscInt,PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,void*,const void*);  \
  PetscErrorCode (*UnpackAndAdd)   (PetscInt,PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,void*,const void*);  \
  PetscErrorCode (*UnpackAndMin)   (PetscInt,PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,void*,const void*);  \
  PetscErrorCode (*UnpackAndMax)   (PetscInt,PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,void*,const void*);  \
  PetscErrorCode (*UnpackAndMinloc)(PetscInt,PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,void*,const void*);  \
  PetscErrorCode (*UnpackAndMaxloc)(PetscInt,PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,void*,const void*);  \
  PetscErrorCode (*UnpackAndMult)  (PetscInt,PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,void*,const void*);  \
  PetscErrorCode (*UnpackAndLAND)  (PetscInt,PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,void*,const void*);  \
  PetscErrorCode (*UnpackAndBAND)  (PetscInt,PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,void*,const void*);  \
  PetscErrorCode (*UnpackAndLOR)   (PetscInt,PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,void*,const void*);  \
  PetscErrorCode (*UnpackAndBOR)   (PetscInt,PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,void*,const void*);  \
  PetscErrorCode (*UnpackAndLXOR)  (PetscInt,PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,void*,const void*);  \
  PetscErrorCode (*UnpackAndBXOR)  (PetscInt,PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,void*,const void*);  \
  PetscErrorCode (*FetchAndInsert) (PetscInt,PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,void*,void*);        \
  PetscErrorCode (*FetchAndAdd)    (PetscInt,PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,void*,void*);        \
  PetscErrorCode (*FetchAndMin)    (PetscInt,PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,void*,void*);        \
  PetscErrorCode (*FetchAndMax)    (PetscInt,PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,void*,void*);        \
  PetscErrorCode (*FetchAndMinloc) (PetscInt,PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,void*,void*);        \
  PetscErrorCode (*FetchAndMaxloc) (PetscInt,PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,void*,void*);        \
  PetscErrorCode (*FetchAndMult)   (PetscInt,PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,void*,void*);        \
  PetscErrorCode (*FetchAndLAND)   (PetscInt,PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,void*,void*);        \
  PetscErrorCode (*FetchAndBAND)   (PetscInt,PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,void*,void*);        \
  PetscErrorCode (*FetchAndLOR)    (PetscInt,PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,void*,void*);        \
  PetscErrorCode (*FetchAndBOR)    (PetscInt,PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,void*,void*);        \
  PetscErrorCode (*FetchAndLXOR)   (PetscInt,PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,void*,void*);        \
  PetscErrorCode (*FetchAndBXOR)   (PetscInt,PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,void*,void*);        \
  PetscMPIInt    tag;         /* Each link has a tag so we can perform multiple SF ops at the same time */         \
  MPI_Datatype   unit;                                                                                             \
  MPI_Datatype   basicunit;   /* unit is made of MPI builtin dataype basicunit */                                  \
  PetscBool      isbuiltin;   /* Is unit an MPI builtin datatype? If it is true, basicunit=unit, bs=1 */           \
  size_t         unitbytes;   /* Number of bytes in a unit */                                                      \
  PetscInt       bs;          /* Number of basic units in a unit */                                                \
  const void     *rkey,*lkey; /* rootdata and leafdata used as keys for operation */                                                \
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
PETSC_INTERN PetscErrorCode PetscSFPackGetUnpackAndOp(PetscSF,PetscSFPack,MPI_Op,PetscErrorCode (**UnpackAndOp)(PetscInt,PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,void*,const void*));
PETSC_INTERN PetscErrorCode PetscSFPackGetFetchAndOp(PetscSF,PetscSFPack,MPI_Op,PetscErrorCode (**FetchAndOp)(PetscInt,PetscInt,const PetscInt*,PetscInt,PetscSFPackOpt,void*,void*));
PETSC_INTERN PetscErrorCode PetscSFPackSetupOptimization(PetscInt,const PetscInt*,const PetscInt*,PetscSFPackOpt*);
PETSC_INTERN PetscErrorCode PetscSFPackDestoryOptimization(PetscSFPackOpt *out);
PETSC_INTERN PetscErrorCode PetscSFPackSetErrorOnUnsupportedOverlap(PetscSF,MPI_Datatype,const void*,const void*);

#endif
