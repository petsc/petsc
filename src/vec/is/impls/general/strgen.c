
#include "src/vec/is/impls/general/general.h" /*I  "petscis.h"  I*/

EXTERN PetscErrorCode ISDuplicate_General(IS,IS *);
EXTERN PetscErrorCode ISDestroy_General(IS);
EXTERN PetscErrorCode ISGetIndices_General(IS,PetscInt **);
EXTERN PetscErrorCode ISRestoreIndices_General(IS,PetscInt **);
EXTERN PetscErrorCode ISGetSize_General(IS,PetscInt *);
EXTERN PetscErrorCode ISGetLocalSize_General(IS,PetscInt *);
EXTERN PetscErrorCode ISInvertPermutation_General(IS,PetscInt,IS *);
EXTERN PetscErrorCode ISView_General(IS,PetscViewer);
EXTERN PetscErrorCode ISSort_General(IS);
EXTERN PetscErrorCode ISSorted_General(IS,PetscTruth*);

static struct _ISOps myops = { ISGetSize_General,
                               ISGetLocalSize_General,
                               ISGetIndices_General,
                               ISRestoreIndices_General,
                               ISInvertPermutation_General,
                               ISSort_General,
                               ISSorted_General,
                               ISDuplicate_General,
                               ISDestroy_General,
                               ISView_General};

#undef __FUNCT__  
#define __FUNCT__ "ISStrideToGeneral" 
/*@C
   ISStrideToGeneral - Converts a stride index set to a general index set.

   Collective on IS

   Input Parameters:
.    is - the index set

   Level: advanced

   Concepts: index sets^converting
   Concepts: stride^converting index sets

.seealso: ISCreateStride(), ISCreateBlock(), ISCreateGeneral()
@*/
PetscErrorCode ISStrideToGeneral(IS inis)
{
  PetscErrorCode ierr;
  PetscInt       step;
  IS_General     *sub;
  PetscTruth     stride,flg;

  PetscFunctionBegin;
  ierr = ISStride(inis,&stride);CHKERRQ(ierr);
  if (!stride) SETERRQ(PETSC_ERR_SUP,"Can only convert stride index sets");

  ierr = PetscNew(IS_General,&sub);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(inis,sizeof(IS_General));CHKERRQ(ierr);
  
  ierr   = ISGetIndices(inis,&sub->idx);CHKERRQ(ierr);
  /* Note: we never restore the indices, since we need to keep the copy generated */
  ierr   = ISGetLocalSize(inis,&sub->n);CHKERRQ(ierr);

  ierr = ISStrideGetInfo(inis,PETSC_NULL,&step);CHKERRQ(ierr);
  if (step > 0) sub->sorted = PETSC_TRUE; else sub->sorted = PETSC_FALSE;
  sub->allocated = PETSC_TRUE;

  /* Remove the old stride data set */
  ierr = PetscFree(inis->data);CHKERRQ(ierr);

  inis->type         = IS_GENERAL;
  inis->data         = (void*)sub;
  inis->isperm       = PETSC_FALSE;
  ierr = PetscMemcpy(inis->ops,&myops,sizeof(myops));CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-is_view",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = ISView(inis,PETSC_VIEWER_STDOUT_(inis->comm));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}




