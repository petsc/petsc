
/*
    Creates hypre ijvector from PETSc vector
*/

#include <petsc-private/vecimpl.h>          /*I "petscvec.h" I*/
EXTERN_C_BEGIN
#include <HYPRE.h>
#include <HYPRE_parcsr_ls.h>
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "VecHYPRE_IJVectorCreate"
PetscErrorCode VecHYPRE_IJVectorCreate(Vec v,HYPRE_IJVector *ij)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = HYPRE_IJVectorCreate(((PetscObject)v)->comm,v->map->rstart,v->map->rend-1,ij);CHKERRQ(ierr);
  ierr = HYPRE_IJVectorSetObjectType(*ij,HYPRE_PARCSR);CHKERRQ(ierr);
  ierr = HYPRE_IJVectorInitialize(*ij);CHKERRQ(ierr);
  ierr = HYPRE_IJVectorAssemble(*ij);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecHYPRE_IJVectorCopy"
PetscErrorCode VecHYPRE_IJVectorCopy(Vec v,HYPRE_IJVector ij)
{
  PetscErrorCode ierr;
  PetscScalar    *array;

  PetscFunctionBegin;
  ierr = HYPRE_IJVectorInitialize(ij);CHKERRQ(ierr);
  ierr = VecGetArray(v,&array);CHKERRQ(ierr);
  ierr = HYPRE_IJVectorSetValues(ij,v->map->n,PETSC_NULL,array);CHKERRQ(ierr);
  ierr = VecRestoreArray(v,&array);CHKERRQ(ierr);
  ierr = HYPRE_IJVectorAssemble(ij);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecHYPRE_IJVectorCopyFrom"
PetscErrorCode VecHYPRE_IJVectorCopyFrom(HYPRE_IJVector ij,Vec v)
{
  PetscErrorCode ierr;
  PetscScalar    *array;

  PetscFunctionBegin;
  ierr = VecGetArray(v,&array);CHKERRQ(ierr);
  ierr = HYPRE_IJVectorGetValues(ij,v->map->n,PETSC_NULL,array);CHKERRQ(ierr);
  ierr = VecRestoreArray(v,&array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
