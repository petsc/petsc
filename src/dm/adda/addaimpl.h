#if !defined(_ADDA_H)
#define _ADDA_H

#include "private/dmimpl.h"

/* vector was allocated and never referenced, clearly some task was not finished */
#define ADDA_HAS_LOCAL_VECTOR 0

typedef struct _ADDAOps *ADDAOps;
struct _ADDAOps {
  DMOPS(ADDA)
};

struct _p_ADDA {
  PETSCHEADER(struct _ADDAOps);
  PetscInt            dim;                   /* dimension of lattice */
  PetscInt            dof;                   /* degrees of freedom per node */
  PetscInt            *nodes;                /* array of number of nodes in each dimension */
  PetscInt            *procs;                /* processor layout */
  PetscTruth          *periodic;             /* true, if that dimension is periodic */
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
};

#endif
