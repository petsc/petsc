/*$Id: general.c,v 1.89 1999/11/05 14:44:37 bsmith Exp bsmith $*/
/*
     Provides the functions for index sets (IS) defined by a list of integers.
*/
#include "src/vec/is/impls/general/general.h" /*I  "is.h"  I*/

#undef __FUNC__  
#define __FUNC__ "ISDuplicate_General" 
int ISDuplicate_General(IS is,IS *newIS)
{
  int        ierr;
  IS_General *sub = (IS_General *)is->data;

  PetscFunctionBegin;
  ierr = ISCreateGeneral(is->comm,sub->n,sub->idx,newIS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ISDestroy_General" 
int ISDestroy_General(IS is)
{
  IS_General *is_general = (IS_General*)is->data;
  int        ierr;

  PetscFunctionBegin;
  ierr = PetscFree(is_general->idx);CHKERRQ(ierr);
  ierr = PetscFree(is_general);CHKERRQ(ierr);
  PLogObjectDestroy(is);
  PetscHeaderDestroy(is);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ISIdentity_General" 
int ISIdentity_General(IS is,PetscTruth *ident)
{
  IS_General *is_general = (IS_General*)is->data;
  int        i,n = is_general->n,*idx = is_general->idx;

  PetscFunctionBegin;
  is->isidentity = 1;
  *ident         = PETSC_TRUE;
  for (i=0; i<n; i++) {
    if (idx[i] != i) {
      is->isidentity = 0;
      *ident         = PETSC_FALSE;
      break;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ISGetIndices_General" 
int ISGetIndices_General(IS in,int **idx)
{
  IS_General *sub = (IS_General*)in->data;

  PetscFunctionBegin;
  *idx = sub->idx; 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ISRestoreIndices_General" 
int ISRestoreIndices_General(IS in,int **idx)
{
  IS_General *sub = (IS_General*)in->data;

  PetscFunctionBegin;
  if (*idx != sub->idx) {
    SETERRQ(PETSC_ERR_ARG_WRONG,0,"Must restore with value from ISGetIndices()");
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ISGetSize_General" 
int ISGetSize_General(IS is,int *size)
{
  IS_General *sub = (IS_General *)is->data;

  PetscFunctionBegin;
  *size = sub->n; 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ISInvertPermutation_General" 
int ISInvertPermutation_General(IS is,IS *isout)
{
  IS_General *sub = (IS_General *)is->data;
  int        i,ierr,*ii,n = sub->n,*idx = sub->idx;

  PetscFunctionBegin;
  ii = (int*)PetscMalloc(n*sizeof(int));CHKPTRQ(ii);
  for (i=0; i<n; i++) {
    ii[idx[i]] = i;
  }
  ierr = ISCreateGeneral(PETSC_COMM_SELF,n,ii,isout);CHKERRQ(ierr);
  ierr = ISSetPermutation(*isout);CHKERRQ(ierr);
  ierr = PetscFree(ii);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ISView_General" 
int ISView_General(IS is,Viewer viewer)
{
  IS_General  *sub = (IS_General *)is->data;
  int         i,n = sub->n,*idx = sub->idx,ierr;
  PetscTruth  isascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,ASCII_VIEWER,&isascii);CHKERRQ(ierr);
  if (isascii) {
    MPI_Comm comm;
    int      rank,size;

    ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
    ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);

    if (size > 1) {
      if (is->isperm) {
        ierr = ViewerASCIISynchronizedPrintf(viewer,"[%d] Index set is permutation\n",rank);CHKERRQ(ierr);
      }
      ierr = ViewerASCIISynchronizedPrintf(viewer,"[%d] Number of indices in set %d\n",rank,n);CHKERRQ(ierr);
      for (i=0; i<n; i++) {
        ierr = ViewerASCIISynchronizedPrintf(viewer,"[%d] %d %d\n",rank,i,idx[i]);CHKERRQ(ierr);
      }
    } else {
      if (is->isperm) {
        ierr = ViewerASCIISynchronizedPrintf(viewer,"Index set is permutation\n");CHKERRQ(ierr);
      }
      ierr = ViewerASCIISynchronizedPrintf(viewer,"Number of indices in set %d\n",n);CHKERRQ(ierr);
      for (i=0; i<n; i++) {
        ierr = ViewerASCIISynchronizedPrintf(viewer,"%d %d\n",i,idx[i]);CHKERRQ(ierr);
      }
    }
    ierr = ViewerFlush(viewer);CHKERRQ(ierr);
  } else {
    SETERRQ1(1,1,"Viewer type %s not supported for this object",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ISSort_General" 
int ISSort_General(IS is)
{
  IS_General *sub = (IS_General *)is->data;
  int        ierr;

  PetscFunctionBegin;
  if (sub->sorted) PetscFunctionReturn(0);
  ierr = PetscSortInt(sub->n,sub->idx);CHKERRQ(ierr);
  sub->sorted = 1;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ISSorted_General" 
int ISSorted_General(IS is,PetscTruth *flg)
{
  IS_General *sub = (IS_General *)is->data;

  PetscFunctionBegin;
  *flg = (PetscTruth) sub->sorted;
  PetscFunctionReturn(0);
}

static struct _ISOps myops = { ISGetSize_General,
                               ISGetSize_General,
                               ISGetIndices_General,
                               ISRestoreIndices_General,
                               ISInvertPermutation_General,
                               ISSort_General,
                               ISSorted_General,
                               ISDuplicate_General,
                               ISDestroy_General,
                               ISView_General,
                               ISIdentity_General };

#undef __FUNC__  
#define __FUNC__ "ISCreateGeneral" 
/*@C
   ISCreateGeneral - Creates a data structure for an index set 
   containing a list of integers.

   Collective on MPI_Comm

   Input Parameters:
+  n - the length of the index set
.  idx - the list of integers
-  comm - the MPI communicator

   Output Parameter:
.  is - the new index set

   Notes:
   When the communicator is not MPI_COMM_SELF, the operations on IS are NOT
   conceptually the same as MPI_Group operations. The IS are then
   distributed sets of indices. 

   Level: beginner

.keywords: IS, general, index set, create

.seealso: ISCreateStride(), ISCreateBlock(), ISAllGather()
@*/
int ISCreateGeneral(MPI_Comm comm,int n,const int idx[],IS *is)
{
  int        i,sorted = 1,min,max,ierr;
  IS         Nindex;
  IS_General *sub;
  PetscTruth flg;

  PetscFunctionBegin;
  PetscValidPointer(is);
  if (n < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"length < 0");
  if (n) {PetscValidIntPointer(idx);}

  *is = 0;
  PetscHeaderCreate(Nindex,_p_IS,struct _ISOps,IS_COOKIE,IS_GENERAL,"IS",comm,ISDestroy,ISView); 
  PLogObjectCreate(Nindex);
  sub            = PetscNew(IS_General);CHKPTRQ(sub);
  PLogObjectMemory(Nindex,sizeof(IS_General)+n*sizeof(int)+sizeof(struct _p_IS));
  sub->idx       = (int*)PetscMalloc((n+1)*sizeof(int));CHKPTRQ(sub->idx);
  sub->n         = n;
  for (i=1; i<n; i++) {
    if (idx[i] < idx[i-1]) {sorted = 0; break;}
  }
  if (n) {min = max = idx[0];} else {min = max = 0;}
  for (i=1; i<n; i++) {
    if (idx[i] < min) min = idx[i];
    if (idx[i] > max) max = idx[i];
  }
  ierr = PetscMemcpy(sub->idx,idx,n*sizeof(int));CHKERRQ(ierr);
  sub->sorted     = sorted;
  Nindex->min     = min;
  Nindex->max     = max;
  Nindex->data    = (void*)sub;
  ierr = PetscMemcpy(Nindex->ops,&myops,sizeof(myops));CHKERRQ(ierr);
  Nindex->isperm     = 0;
  Nindex->isidentity = 0;
  ierr = OptionsHasName(PETSC_NULL,"-is_view",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = ISView(Nindex,VIEWER_STDOUT_(Nindex->comm));CHKERRQ(ierr);
  }
  *is = Nindex; PetscFunctionReturn(0);
}




