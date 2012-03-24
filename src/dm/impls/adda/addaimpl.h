#if !defined(_ADDA_H)
#define _ADDA_H

#include <petscdmadda.h>
#include <petsc-private/dmimpl.h>

/* vector was allocated and never referenced, clearly some task was not finished */
#define ADDA_HAS_LOCAL_VECTOR 0


typedef struct {
  PetscInt            dim;                   /* dimension of lattice */
  PetscInt            dof;                   /* degrees of freedom per node */
  PetscInt            *nodes;                /* array of number of nodes in each dimension */
  PetscInt            *procs;                /* processor layout */
  PetscBool           *periodic;             /* true, if that dimension is periodic */
  PetscInt            *lcs, *lce;            /* corners of the locally stored portion of the grid */
  PetscInt            *lgs, *lge;            /* corners of the local portion of the grid
						including the ghost points */
  PetscInt            lsize;                 /* number of nodes in local region */
  PetscInt            lgsize;                /* number of nodes in local region including ghost points */
  Vec                 global;                /* global prototype vector */
#if ADDA_HAS_LOCAL_VECTOR
  Vec                 local;                 /* local prototype vector */
#endif
  PetscInt            *refine;               /* refinement factors for each dimension */
  PetscInt            dofrefine;             /* refinement factor for the dof */
} DM_ADDA;

#endif
