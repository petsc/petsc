#if !defined(__KSP_PIPEGCR_H)
#define __KSP_PIPEGCR_H

#include <petsc/private/kspimpl.h>        /*I "petscksp.h" I*/

typedef struct {
  PetscInt     mmax;        /* The maximum number vectors of each type to store */
  PetscInt     nprealloc;   /* How many vectors to preallocate */
  PetscInt     nvecs;       /* How many vectors are actually stored */
  PetscInt     vecb;        /* How many vectors to allocate at a time in a chunk */
  Vec          *pvecs,*svecs,*qvecs,*tvecs,**ppvecs,**psvecs,**pqvecs,**ptvecs,*qold,*pold,*sold,*told; /* Arrays of vectors, ...
    and arrays of pointers to them */
  PetscInt     *chunksizes; /* Chunk sizes allocated */
  PetscInt     nchunks;     /* Number of chunks */
  KSPFCDTruncationType trunctype;
  PetscInt     n_restarts;
  PetscInt     n_search_space_resets;
  PetscScalar *dots;
  PetscReal   *etas;
  Vec         *redux;
  PetscErrorCode (*modifypc)(KSP,PetscInt,PetscReal,void*);  /* function to modify the preconditioner*/
  PetscErrorCode (*modifypc_destroy)(void*);                 /* function to destroy the user context for the modifypc function */
  void *modifypc_ctx;                                        /* user defined data for the modifypc function */
  PetscBool   norm_breakdown; /* set this if the recurred norm eta breaks down -> restart triggered */
  PetscBool   unroll_w;
} KSP_PIPEGCR;

#endif
