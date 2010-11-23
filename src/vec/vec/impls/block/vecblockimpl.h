
#ifndef __PETSc_VecBlock_implementation__
#define __PETSc_VecBlock_implementation__

#include <petsc.h>
#include <petscvec.h>

typedef struct {
	PetscInt   nb; /* n blocks */
	Vec        *v;
	PetscBool  setup_called;
} Vec_Block;

#endif
