/*
     Provides the functions for index sets (IS) defined by a list of integers.
*/
#include "src/vec/is/impls/general/general.h" /*I  "petscis.h"  I*/

EXTERN int VecInitializePackage(char *);

#undef __FUNCT__  
#define __FUNCT__ "ISDuplicate_General" 
int ISDuplicate_General(IS is,IS *newIS)
{
  int        ierr;
  IS_General *sub = (IS_General *)is->data;

  PetscFunctionBegin;
  ierr = ISCreateGeneral(is->comm,sub->n,sub->idx,newIS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISDestroy_General" 
int ISDestroy_General(IS is)
{
  IS_General *is_general = (IS_General*)is->data;
  int        ierr;

  PetscFunctionBegin;
  ierr = PetscFree(is_general->idx);CHKERRQ(ierr);
  ierr = PetscFree(is_general);CHKERRQ(ierr);
  PetscLogObjectDestroy(is);
  PetscHeaderDestroy(is);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISIdentity_General" 
int ISIdentity_General(IS is,PetscTruth *ident)
{
  IS_General *is_general = (IS_General*)is->data;
  int        i,n = is_general->n,*idx = is_general->idx;

  PetscFunctionBegin;
  is->isidentity = PETSC_TRUE;
  *ident         = PETSC_TRUE;
  for (i=0; i<n; i++) {
    if (idx[i] != i) {
      is->isidentity = PETSC_FALSE;
      *ident         = PETSC_FALSE;
      break;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISGetIndices_General" 
int ISGetIndices_General(IS in,int **idx)
{
  IS_General *sub = (IS_General*)in->data;

  PetscFunctionBegin;
  *idx = sub->idx; 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISRestoreIndices_General" 
int ISRestoreIndices_General(IS in,int **idx)
{
  IS_General *sub = (IS_General*)in->data;

  PetscFunctionBegin;
  if (*idx != sub->idx) {
    SETERRQ(PETSC_ERR_ARG_WRONG,"Must restore with value from ISGetIndices()");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISGetSize_General" 
int ISGetSize_General(IS is,int *size)
{
  IS_General *sub = (IS_General *)is->data;

  PetscFunctionBegin;
  *size = sub->N; 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISGetLocalSize_General" 
int ISGetLocalSize_General(IS is,int *size)
{
  IS_General *sub = (IS_General *)is->data;

  PetscFunctionBegin;
  *size = sub->n; 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISInvertPermutation_General" 
int ISInvertPermutation_General(IS is,int nlocal,IS *isout)
{
  IS_General *sub = (IS_General *)is->data;
  int        i,ierr,*ii,n = sub->n,*idx = sub->idx,size,nstart;
  IS         istmp,nistmp;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(is->comm,&size);CHKERRQ(ierr);
  if (size == 1) {
    ierr = PetscMalloc(n*sizeof(int),&ii);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      ii[idx[i]] = i;
    }
    ierr = ISCreateGeneral(PETSC_COMM_SELF,n,ii,isout);CHKERRQ(ierr);
    ierr = ISSetPermutation(*isout);CHKERRQ(ierr);
    ierr = PetscFree(ii);CHKERRQ(ierr);
  } else {
    /* crude, nonscalable get entire IS on each processor */
    ierr = ISAllGather(is,&istmp);CHKERRQ(ierr);
    ierr = ISSetPermutation(istmp);CHKERRQ(ierr);
    ierr = ISInvertPermutation(istmp,PETSC_DECIDE,&nistmp);CHKERRQ(ierr);
    ierr = ISDestroy(istmp);CHKERRQ(ierr);
    /* get the part we need */
    ierr    = MPI_Scan(&nlocal,&nstart,1,MPI_INT,MPI_SUM,is->comm);CHKERRQ(ierr);
    nstart -= nlocal;
    ierr    = ISGetIndices(nistmp,&idx);CHKERRQ(ierr);
    ierr    = ISCreateGeneral(is->comm,nlocal,idx+nstart,isout);CHKERRQ(ierr);    
    ierr    = ISRestoreIndices(nistmp,&idx);CHKERRQ(ierr);
    ierr    = ISDestroy(nistmp);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISView_General" 
int ISView_General(IS is,PetscViewer viewer)
{
  IS_General  *sub = (IS_General *)is->data;
  int         i,n = sub->n,*idx = sub->idx,ierr;
  PetscTruth  iascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    MPI_Comm comm;
    int      rank,size;

    ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
    ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);

    if (size > 1) {
      if (is->isperm) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Index set is permutation\n",rank);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Number of indices in set %d\n",rank,n);CHKERRQ(ierr);
      for (i=0; i<n; i++) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] %d %d\n",rank,i,idx[i]);CHKERRQ(ierr);
      }
    } else {
      if (is->isperm) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Index set is permutation\n");CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Number of indices in set %d\n",n);CHKERRQ(ierr);
      for (i=0; i<n; i++) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"%d %d\n",i,idx[i]);CHKERRQ(ierr);
      }
    }
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported for this object",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISSort_General" 
int ISSort_General(IS is)
{
  IS_General *sub = (IS_General *)is->data;
  int        ierr;

  PetscFunctionBegin;
  if (sub->sorted) PetscFunctionReturn(0);
  ierr = PetscSortInt(sub->n,sub->idx);CHKERRQ(ierr);
  sub->sorted = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISSorted_General" 
int ISSorted_General(IS is,PetscTruth *flg)
{
  IS_General *sub = (IS_General *)is->data;

  PetscFunctionBegin;
  *flg = sub->sorted;
  PetscFunctionReturn(0);
}

static struct _ISOps myops = { ISGetSize_General,
                               ISGetLocalSize_General,
                               ISGetIndices_General,
                               ISRestoreIndices_General,
                               ISInvertPermutation_General,
                               ISSort_General,
                               ISSorted_General,
                               ISDuplicate_General,
                               ISDestroy_General,
                               ISView_General,
                               ISIdentity_General };

#undef __FUNCT__  
#define __FUNCT__ "ISCreateGeneral" 
/*@C
   ISCreateGeneral - Creates a data structure for an index set 
   containing a list of integers.

   Collective on MPI_Comm

   Input Parameters:
+  comm - the MPI communicator
.  n - the length of the index set
-  idx - the list of integers

   Output Parameter:
.  is - the new index set

   Notes:
   When the communicator is not MPI_COMM_SELF, the operations on IS are NOT
   conceptually the same as MPI_Group operations. The IS are then
   distributed sets of indices and thus certain operations on them are collective.

   Level: beginner

  Concepts: index sets^creating
  Concepts: IS^creating

.seealso: ISCreateStride(), ISCreateBlock(), ISAllGather()
@*/
int ISCreateGeneral(MPI_Comm comm,int n,const int idx[],IS *is)
{
  int        i,min,max,ierr;
  PetscTruth sorted = PETSC_TRUE;
  IS         Nindex;
  IS_General *sub;
  PetscTruth flg;

  PetscFunctionBegin;
  PetscValidPointer(is,4);
  if (n < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"length < 0");
  if (n) {PetscValidIntPointer(idx,3);}
  *is = PETSC_NULL;
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = VecInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif

  PetscHeaderCreate(Nindex,_p_IS,struct _ISOps,IS_COOKIE,IS_GENERAL,"IS",comm,ISDestroy,ISView); 
  PetscLogObjectCreate(Nindex);
  ierr           = PetscNew(IS_General,&sub);CHKERRQ(ierr);
  PetscLogObjectMemory(Nindex,sizeof(IS_General)+n*sizeof(int)+sizeof(struct _p_IS));
  ierr           = PetscMalloc((n+1)*sizeof(int),&sub->idx);CHKERRQ(ierr);
  sub->n         = n;

  ierr = MPI_Allreduce(&n,&sub->N,1,MPI_INT,MPI_SUM,comm);CHKERRQ(ierr);
  for (i=1; i<n; i++) {
    if (idx[i] < idx[i-1]) {sorted = PETSC_FALSE; break;}
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
  Nindex->isperm     = PETSC_FALSE;
  Nindex->isidentity = PETSC_FALSE;
  ierr = PetscOptionsHasName(PETSC_NULL,"-is_view",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = ISView(Nindex,PETSC_VIEWER_STDOUT_(Nindex->comm));CHKERRQ(ierr);
  }
  *is = Nindex;
  PetscFunctionReturn(0);
}




