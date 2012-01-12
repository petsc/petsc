#if !defined(_STASHIJ_H)
#define _STASHIJ_H

/* Need PetscLayout */
#include <private/vecimpl.h> /*I "petscvec.h" */

/* Need PetscHash */
#include <../src/sys/utils/hash.h>

struct _StashSeqIJ {
  PetscInt      n;
  PetscBool     multivalued;
  PetscHashIJ   h;
};
typedef struct _StashSeqIJ *StashSeqIJ;

struct _StashMPIIJ {
  PetscLayout rmap;
  StashSeqIJ astash, bstash;
  PetscBool  assembled;
};
typedef struct _StashMPIIJ *StashMPIIJ;



extern PetscErrorCode StashSeqIJCreate_Private(StashSeqIJ*);
extern PetscErrorCode StashSeqIJGetMultivalued_Private(StashSeqIJ, PetscBool*);
extern PetscErrorCode StashSeqIJSetMultivalued_Private(StashSeqIJ, PetscBool);
extern PetscErrorCode StashSeqIJExtend_Private(StashSeqIJ, PetscInt, const PetscInt*, const PetscInt*);
#endif
extern PetscErrorCode StashSeqIJSetPreallocation_Private(StashSeqIJ, PetscInt);
extern PetscErrorCode StashSeqIJGetIndices_Private(StashSeqIJ, PetscInt*,PetscInt**, PetscInt**);
extern PetscErrorCode StashSeqIJClear_Private(StashSeqIJ);
extern PetscErrorCode StashSeqIJDestroy_Private(StashSeqIJ *);


extern PetscErrorCode StashMPIIJCreate_Private(PetscLayout, StashMPIIJ*);
extern PetscErrorCode StashMPIIJGetMultivalued_Private(StashMPIIJ, PetscBool*);
extern PetscErrorCode StashMPIIJSetMultivalued_Private(StashMPIIJ, PetscBool);
extern PetscErrorCode StashMPIIJDestroy_Private(StashMPIIJ *);
extern PetscErrorCode StashMPIIJClear_Private(StashMPIIJ);

extern PetscErrorCode StashMPIIJSetPreallocation_Private(StashMPIIJ, PetscInt, PetscInt);
extern PetscErrorCode StashMPIIJExtend_Private(StashMPIIJ, PetscInt, const PetscInt*, const PetscInt*);
extern PetscErrorCode StashMPIIJGetIndices_Private(StashMPIIJ, PetscInt*,PetscInt**, PetscInt**,PetscInt*,PetscInt**,PetscInt**);

extern PetscErrorCode StashMPIIJGetIndicesMerged_Private(StashMPIIJ, PetscInt*,PetscInt**, PetscInt**);

extern PetscErrorCode StashMPIIJAssemble_Private(StashMPIIJ);
