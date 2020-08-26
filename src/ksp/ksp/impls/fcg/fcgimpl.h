#if !defined(__KSP_FCG_H)
#define __KSP_FCG_H

#include <petsc/private/kspimpl.h>        /*I "petscksp.h" I*/

typedef struct {
  KSPCGType    type;        /* type of system (symmetric of Hermitian) */
  PetscScalar  emin,emax;           /* eigenvalues */
  PetscInt     ned;                 /* size of following arrays */
  PetscScalar  *e,*d;
  PetscReal    *ee,*dd;             /* work space for Lanczos algorithm */

  PetscInt     mmax;        /* The maximum number of P/C vectors to store */
  PetscInt     nprealloc;   /* How many vectors to preallocate */
  PetscInt     nvecs;       /* How many P/C vecs are actually stored */
  PetscInt     vecb;        /* How many vecs to allocate at a time in a chunk */
  Vec          *Pvecs, *Cvecs, **pPvecs, **pCvecs; /* Arrays of vectors, and arrays of pointers to them */
  PetscInt     *chunksizes; /* Chunk sizes allocated */
  PetscInt     nchunks;     /* Number of chunks */
  KSPFCDTruncationType truncstrat;
} KSP_FCG;

#endif
