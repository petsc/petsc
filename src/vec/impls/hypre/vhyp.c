/*$Id: bvec2.c,v 1.202 2001/09/12 03:26:24 bsmith Exp $*/
/*
    Creates hypre ijvector from PETSc vector
*/

#include "vecimpl.h"          /*I "petscvec.h" I*/
EXTERN_C_BEGIN
#include "HYPRE.h"
#include "IJ_mv.h"
EXTERN_C_END

int VecHYPRE_IJVectorCreate(Vec v,HYPRE_IJVector *ij)
{
  int         ierr;

  PetscFunctionBegin;
  ierr = HYPRE_IJVectorCreate(v->comm,v->map->rstart,v->map->rend-1,ij);CHKERRQ(ierr);
  ierr = HYPRE_IJVectorSetObjectType(*ij,HYPRE_PARCSR);CHKERRQ(ierr);
  ierr = HYPRE_IJVectorInitialize(*ij);CHKERRQ(ierr);
  ierr = HYPRE_IJVectorAssemble(*ij);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int VecHYPRE_IJVectorCopy(Vec v,HYPRE_IJVector ij)
{
  int         ierr;
  PetscScalar *array;

  PetscFunctionBegin;
  ierr = HYPRE_IJVectorInitialize(ij);CHKERRQ(ierr);
  ierr = VecGetArray(v,&array);CHKERRQ(ierr);
  ierr = HYPRE_IJVectorSetValues(ij,v->map->n,PETSC_NULL,array);CHKERRQ(ierr);
  ierr = VecRestoreArray(v,&array);CHKERRQ(ierr);
  ierr = HYPRE_IJVectorAssemble(ij);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int VecHYPRE_IJVectorCopyFrom(HYPRE_IJVector ij,Vec v)
{
  int         ierr;
  PetscScalar *array;

  PetscFunctionBegin;
  ierr = VecGetArray(v,&array);CHKERRQ(ierr);
  ierr = HYPRE_IJVectorGetValues(ij,v->map->n,PETSC_NULL,array);CHKERRQ(ierr);
  ierr = VecRestoreArray(v,&array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
