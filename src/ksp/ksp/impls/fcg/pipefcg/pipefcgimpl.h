#if !defined(__KSP_PIPEFCG_H)
#define __KSP_PIPEFCG_H

#include <petsc/private/kspimpl.h>        /*I "petscksp.h" I*/

typedef struct {
  KSPCGType            type;        /* type of system (symmetric or Hermitian) */
  PetscInt             mmax;        /* The maximum number of P/C vectors to store */
  PetscInt             nprealloc;   /* How many vectors to preallocate */
  PetscInt             nvecs;       /* How many P/C vecs are actually stored */
  PetscInt             vecb;        /* How many vecs to allocate at a time in a chunk */
  Vec                  *Qvecs, *ZETAvecs, *Pvecs, *Cvecs, *Svecs, **pQvecs, **pZETAvecs, **pPvecs, **pCvecs, **pSvecs,*Qold,*ZETAold,*Pold,*Sold;
  PetscInt             *chunksizes; /* Chunk sizes allocated */
  PetscInt             nchunks;     /* Number of chunks */
  KSPFCDTruncationType truncstrat;
  PetscInt             n_restarts;
  PetscScalar          *dots;
  PetscReal            *etas;
  Vec                  *redux;
  PetscBool            norm_breakdown;
} KSP_PIPEFCG;

#endif
