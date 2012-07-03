#if !defined(_STASHIJ_H)
#define _STASHIJ_H

/* Need PetscLayout */
#include <petsc-private/vecimpl.h> /*I "petscvec.h" */

/* Need PetscHash */
#include <../src/sys/utils/hash.h>

struct _MatStashSeqIJ {
  PetscInt      n;
  PetscBool     multivalued;
  PetscHashIJ   h;
};
typedef struct _MatStashSeqIJ *MatStashSeqIJ;

struct _MatStashMPIIJ {
  PetscLayout rmap;
  MatStashSeqIJ  astash, bstash;
  PetscBool   assembled;
};
typedef struct _MatStashMPIIJ *MatStashMPIIJ;

extern PetscErrorCode MatStashSeqIJCreate_Private(MatStashSeqIJ*);
extern PetscErrorCode MatStashSeqIJGetMultivalued_Private(MatStashSeqIJ, PetscBool*);
extern PetscErrorCode MatStashSeqIJSetMultivalued_Private(MatStashSeqIJ, PetscBool);
extern PetscErrorCode MatStashSeqIJExtend_Private(MatStashSeqIJ, PetscInt, const PetscInt*, const PetscInt*);

extern PetscErrorCode MatStashSeqIJSetPreallocation_Private(MatStashSeqIJ, PetscInt);
extern PetscErrorCode MatStashSeqIJGetIndices_Private(MatStashSeqIJ, PetscInt*,PetscInt**, PetscInt**);
extern PetscErrorCode MatStashSeqIJClear_Private(MatStashSeqIJ);
extern PetscErrorCode MatStashSeqIJDestroy_Private(MatStashSeqIJ *);


extern PetscErrorCode MatStashMPIIJCreate_Private(PetscLayout, MatStashMPIIJ*);
extern PetscErrorCode MatStashMPIIJGetMultivalued_Private(MatStashMPIIJ, PetscBool*);
extern PetscErrorCode MatStashMPIIJSetMultivalued_Private(MatStashMPIIJ, PetscBool);
extern PetscErrorCode MatStashMPIIJDestroy_Private(MatStashMPIIJ *);
extern PetscErrorCode MatStashMPIIJClear_Private(MatStashMPIIJ);

extern PetscErrorCode MatStashMPIIJSetPreallocation_Private(MatStashMPIIJ, PetscInt, PetscInt);
extern PetscErrorCode MatStashMPIIJExtend_Private(MatStashMPIIJ, PetscInt, const PetscInt*, const PetscInt*);
extern PetscErrorCode MatStashMPIIJGetIndices_Private(MatStashMPIIJ, PetscInt*,PetscInt**, PetscInt**,PetscInt*,PetscInt**,PetscInt**);

extern PetscErrorCode MatStashMPIIJGetIndicesMerged_Private(MatStashMPIIJ, PetscInt*,PetscInt**, PetscInt**);

extern PetscErrorCode MatStashMPIIJAssemble_Private(MatStashMPIIJ);
#endif
