#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: strgen.c,v 1.3 1999/01/31 16:03:26 bsmith Exp bsmith $";
#endif

#include "src/vec/is/impls/general/general.h" /*I  "is.h"  I*/

extern int ISDuplicate_General(IS, IS *);
extern int ISDestroy_General(IS);
extern int ISGetIndices_General(IS,int **);
extern int ISRestoreIndices_General(IS,int **);
extern int ISGetSize_General(IS,int *);
extern int ISInvertPermutation_General(IS, IS *);
extern int ISView_General(IS, Viewer);
extern int ISSort_General(IS);
extern int ISSorted_General(IS, PetscTruth*);

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
#define __FUNC__ "ISStrideToGeneral" 
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
  int        flg, ierr;
  IS_General *sub;
  PetscTruth stride;

  PetscFunctionBegin;
  ierr = ISStride(inis,&stride); CHKERRQ(ierr);
  if (!stride) SETERRQ(1,1,"Can only convert stride index sets");

  sub        = PetscNew(IS_General); CHKPTRQ(sub);
  PLogObjectMemory(inis,sizeof(IS_General));
  
  ierr   = ISGetIndices(inis,&sub->idx); CHKERRQ(ierr);
  /* Note: we never restore the indices, since we need to keep the copy generated */
  ierr   = ISGetSize(inis,&sub->n);CHKERRQ(ierr);

  ierr = ISStrideGetInfo(inis,PETSC_NULL,&sub->sorted);CHKERRQ(ierr);
  if (sub->sorted > 0) sub->sorted = 1; else sub->sorted = 0;

  /* Remove the old stride data set */
  PetscFree(inis->data);

  inis->type         = IS_GENERAL;
  inis->data         = (void *) sub;
  inis->isperm       = 0;
  PetscMemcpy(inis->ops,&myops,sizeof(myops));
  ierr = OptionsHasName(PETSC_NULL,"-is_view",&flg); CHKERRQ(ierr);
  if (flg) {
    ierr = ISView(inis,VIEWER_STDOUT_(inis->comm)); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}




