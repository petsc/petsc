#if !defined(_MatStashSpace_h_)
#define _MatStashSpace_h_

#include "petsc.h"

typedef struct _MatStashSpace *PetscMatStashSpace;

struct _MatStashSpace {
  PetscMatStashSpace next;
  MatScalar          *space_head,*val;
  PetscInt           *idx,*idy;
  PetscInt           total_space_size;
  PetscInt           local_used;
  PetscInt           local_remaining;
};

EXTERN PetscErrorCode PetscMatStashSpaceGet(PetscInt,PetscInt,PetscMatStashSpace *);
EXTERN PetscErrorCode PetscMatStashSpaceContiguous(PetscInt,PetscMatStashSpace *,PetscScalar *,PetscInt *,PetscInt *);
EXTERN PetscErrorCode PetscMatStashSpaceDestroy(PetscMatStashSpace);

#endif
