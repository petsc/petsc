/*$Id: strgen.c,v 1.13 2000/05/04 16:25:02 bsmith Exp balay $*/

#include "src/vec/is/impls/general/general.h" /*I  "petscis.h"  I*/

extern int ISDuplicate_General(IS,IS *);
extern int ISDestroy_General(IS);
extern int ISGetIndices_General(IS,int **);
extern int ISRestoreIndices_General(IS,int **);
extern int ISGetSize_General(IS,int *);
extern int ISInvertPermutation_General(IS,int,IS *);
extern int ISView_General(IS,Viewer);
extern int ISSort_General(IS);
extern int ISSorted_General(IS,PetscTruth*);

static struct _ISOps myops = { ISGetSize_General,
                               ISGetSize_General,
                               ISGetIndices_General,
                               ISRestoreIndices_General,
                               ISInvertPermutation_General,
                               ISSort_General,
                               ISSorted_General,
                               ISDuplicate_General,
                               ISDestroy_General,
                               ISView_General};

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"ISStrideToGeneral" 
/*@C
   ISStrideToGeneral - Converts a stride index set to a general index set.

   Collective on IS

   Input Parameters:
.    is - the index set

   Level: advanced

.keywords: IS, general, index set, create, convert, stride

.seealso: ISCreateStride(), ISCreateBlock(), ISCreateGeneral()
@*/
int ISStrideToGeneral(IS inis)
{
  int        ierr,step;
  IS_General *sub;
  PetscTruth stride,flg;

  PetscFunctionBegin;
  ierr = ISStride(inis,&stride);CHKERRQ(ierr);
  if (!stride) SETERRQ(1,1,"Can only convert stride index sets");

  sub        = PetscNew(IS_General);CHKPTRQ(sub);
  PLogObjectMemory(inis,sizeof(IS_General));
  
  ierr   = ISGetIndices(inis,&sub->idx);CHKERRQ(ierr);
  /* Note: we never restore the indices, since we need to keep the copy generated */
  ierr   = ISGetSize(inis,&sub->n);CHKERRQ(ierr);

  ierr = ISStrideGetInfo(inis,PETSC_NULL,&step);CHKERRQ(ierr);
  if (step > 0) sub->sorted = PETSC_TRUE; else sub->sorted = PETSC_FALSE;

  /* Remove the old stride data set */
  ierr = PetscFree(inis->data);CHKERRQ(ierr);

  inis->type         = IS_GENERAL;
  inis->data         = (void*)sub;
  inis->isperm       = 0;
  ierr = PetscMemcpy(inis->ops,&myops,sizeof(myops));CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-is_view",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = ISView(inis,VIEWER_STDOUT_(inis->comm));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}




