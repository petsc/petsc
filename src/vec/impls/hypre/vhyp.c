/*$Id: bvec2.c,v 1.202 2001/09/12 03:26:24 bsmith Exp $*/
/*
    Creates hypre ijvector from PETSc vector
*/

#include "src/vec/vecimpl.h"          /*I "petscvec.h" I*/
#include "HYPRE.h"
#include "IJ_mv.h"

int VecHYPRE_IJVectorCreate(Vec v,HYPRE_IJVector *ij)
{
  int         ierr;

  PetscFunctionBegin;
  ierr = HYPRE_IJVectorCreate(v->comm,v->map->rstart,v->map->rend,ij);CHKERRQ(ierr);
  ierr = HYPRE_IJVectorSetObjectType(*ij,HYPRE_PARCSR);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int VecHYPRE_IJVectorCopy(Vec v,HYPRE_IJVector ij)
{
  int         ierr;
  PetscScalar *array;

  PetscFunctionBegin;
  ierr = HYPRE_IJVectorInitialize(ij);CHKERRQ(ierr);
  ierr = VecGetArrayFast(v,&array);CHKERRQ(ierr);
  ierr = HYPRE_IJVectorSetValues(ij,v->map->n,PETSC_NULL,array);CHKERRQ(ierr);
  ierr = VecRestoreArrayFast(v,&array);CHKERRQ(ierr);
  ierr = HYPRE_IJVectorAssemble(ij);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
