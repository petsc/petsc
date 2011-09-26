#if !defined(_STASHIJ_H)
#define _STASHIJ_H

/* Need PetscLayout */
#include <private/vecimpl.h> /*I "petscvec.h" */
/* Need khash */
#include <../src/mat/impls/ij/petsckhash.h>

/* Linked list of values in a bucket. */
struct _IJNode {
  PetscInt k;
  struct _IJNode *next;
};
typedef struct _IJNode IJNode;

/* Value (holds a linked list of nodes) in the bucket. */
struct _IJVal {
  PetscInt n;
  IJNode *head, *tail;
};
typedef struct _IJVal IJVal;

/* Key (a pair of integers). */
struct _IJKey {
  PetscInt i, j;
};
typedef struct _IJKey IJKey;

/* Hash function: mix two integers into one. 
   Shift by half the number of bits in PetscInt to the left and then XOR.  If the indices fit into the lowest half part of PetscInt, this is a bijection.
 */
#define IJKeyHash(key) ((((key).i) << (4*sizeof(PetscInt)))^((key).j))


/* Compare two keys (integer pairs). */
#define IJKeyEqual(k1,k2) (((k1).i==(k2).i)?((k1).j==(k2).j):0)

KHASH_INIT(IJ,IJKey,IJVal,1,IJKeyHash,IJKeyEqual)

struct _StashSeqIJ {
  PetscInt n;
  PetscBool   multivalued;
  khash_t(IJ) *h;
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
