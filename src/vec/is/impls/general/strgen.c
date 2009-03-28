#define PETSCVEC_DLL
#include "../src/vec/is/impls/general/general.h" /*I  "petscis.h"  I*/

EXTERN PetscErrorCode ISDuplicate_General(IS,IS *);
EXTERN PetscErrorCode ISDestroy_General(IS);
EXTERN PetscErrorCode ISGetIndices_General(IS,const PetscInt *[]);
EXTERN PetscErrorCode ISRestoreIndices_General(IS,const PetscInt *[]);
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
PetscErrorCode PETSCVEC_DLLEXPORT ISStrideToGeneral(IS inis)
{
  PetscErrorCode ierr;
  PetscInt       step;
  IS_General     *sub;
  PetscTruth     stride,flg = PETSC_FALSE;
  const PetscInt *idx;

  PetscFunctionBegin;
  ierr = ISStride(inis,&stride);CHKERRQ(ierr);
  if (!stride) SETERRQ(PETSC_ERR_SUP,"Can only convert stride index sets");

  ierr = PetscNewLog(inis,IS_General,&sub);CHKERRQ(ierr);
  
  ierr   = ISGetLocalSize(inis,&sub->n);CHKERRQ(ierr);
  ierr   = ISGetIndices(inis,&idx);CHKERRQ(ierr);
  ierr   = PetscMalloc(sub->n*sizeof(PetscInt),&sub->idx);CHKERRQ(ierr);
  ierr   = PetscMemcpy(sub->idx,idx,sub->n*sizeof(PetscInt));CHKERRQ(ierr);
  ierr   = ISRestoreIndices(inis,&idx);CHKERRQ(ierr);

  ierr = ISStrideGetInfo(inis,PETSC_NULL,&step);CHKERRQ(ierr);
  if (step > 0) sub->sorted = PETSC_TRUE; else sub->sorted = PETSC_FALSE;
  sub->allocated = PETSC_TRUE;

  /* Remove the old stride data set */
  ierr = PetscFree(inis->data);CHKERRQ(ierr);

  ((PetscObject)inis)->type         = IS_GENERAL;
  inis->data         = (void*)sub;
  inis->isperm       = PETSC_FALSE;
  ierr = PetscMemcpy(inis->ops,&myops,sizeof(myops));CHKERRQ(ierr);
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-is_view",&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg) {
    PetscViewer viewer;
    ierr = PetscViewerASCIIGetStdout(((PetscObject)inis)->comm,&viewer);CHKERRQ(ierr);
    ierr = ISView(inis,viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}




