
/*
    Creates hypre ijvector from PETSc vector
*/

#include <petsc/private/vecimpl.h>          /*I "petscvec.h" I*/
#include <../src/vec/vec/impls/hypre/vhyp.h>
#include <HYPRE.h>

PETSC_EXTERN PetscErrorCode VecHYPRE_IJVectorCreate(Vec v,HYPRE_IJVector *ij)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = HYPRE_IJVectorCreate(PetscObjectComm((PetscObject)v),v->map->rstart,v->map->rend-1,ij);CHKERRQ(ierr);
  ierr = HYPRE_IJVectorSetObjectType(*ij,HYPRE_PARCSR);CHKERRQ(ierr);
  ierr = HYPRE_IJVectorInitialize(*ij);CHKERRQ(ierr);
  ierr = HYPRE_IJVectorAssemble(*ij);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode VecHYPRE_IJVectorCopy(Vec v,HYPRE_IJVector ij)
{
  PetscErrorCode ierr;
  PetscScalar    *array;

  PetscFunctionBegin;
  ierr = HYPRE_IJVectorInitialize(ij);CHKERRQ(ierr);
  ierr = VecGetArray(v,&array);CHKERRQ(ierr);
  ierr = HYPRE_IJVectorSetValues(ij,v->map->n,NULL,(HYPRE_Complex*)array);CHKERRQ(ierr);
  ierr = VecRestoreArray(v,&array);CHKERRQ(ierr);
  ierr = HYPRE_IJVectorAssemble(ij);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode VecHYPRE_IJVectorCopyFrom(HYPRE_IJVector ij,Vec v)
{
  PetscErrorCode ierr;
  PetscScalar    *array;

  PetscFunctionBegin;
  ierr = VecGetArray(v,&array);CHKERRQ(ierr);
  ierr = HYPRE_IJVectorGetValues(ij,v->map->n,NULL,(HYPRE_Complex*)array);CHKERRQ(ierr);
  ierr = VecRestoreArray(v,&array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
