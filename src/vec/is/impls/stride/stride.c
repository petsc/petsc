
/*
       Index sets of evenly space integers, defined by a 
    start, stride and length.
*/
#include <petsc-private/isimpl.h>             /*I   "petscis.h"   I*/
#include <petscvec.h>

typedef struct {
  PetscInt N,n,first,step;
} IS_Stride;

#undef __FUNCT__  
#define __FUNCT__ "ISIdentity_Stride" 
PetscErrorCode ISIdentity_Stride(IS is,PetscBool  *ident)
{
  IS_Stride *is_stride = (IS_Stride*)is->data;

  PetscFunctionBegin;
  is->isidentity = PETSC_FALSE;
  *ident         = PETSC_FALSE;
  if (is_stride->first != 0) PetscFunctionReturn(0);
  if (is_stride->step  != 1) PetscFunctionReturn(0);
  *ident          = PETSC_TRUE;
  is->isidentity  = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISCopy_Stride"
static PetscErrorCode ISCopy_Stride(IS is,IS isy)
{
  IS_Stride      *is_stride = (IS_Stride*)is->data,*isy_stride = (IS_Stride*)isy->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMemcpy(isy_stride,is_stride,sizeof(IS_Stride));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISDuplicate_Stride" 
PetscErrorCode ISDuplicate_Stride(IS is,IS *newIS)
{
  PetscErrorCode ierr;
  IS_Stride      *sub = (IS_Stride*)is->data;

  PetscFunctionBegin;
  ierr = ISCreateStride(((PetscObject)is)->comm,sub->n,sub->first,sub->step,newIS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISInvertPermutation_Stride" 
PetscErrorCode ISInvertPermutation_Stride(IS is,PetscInt nlocal,IS *perm)
{
  IS_Stride      *isstride = (IS_Stride*)is->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (is->isidentity) {
    ierr = ISCreateStride(PETSC_COMM_SELF,isstride->n,0,1,perm);CHKERRQ(ierr);
  } else {
    IS             tmp;
    const PetscInt *indices,n = isstride->n;
    ierr = ISGetIndices(is,&indices);CHKERRQ(ierr);
    ierr = ISCreateGeneral(((PetscObject)is)->comm,n,indices,PETSC_COPY_VALUES,&tmp);CHKERRQ(ierr);
    ierr = ISSetPermutation(tmp); CHKERRQ(ierr);
    ierr = ISRestoreIndices(is,&indices);CHKERRQ(ierr);
    ierr = ISInvertPermutation(tmp,nlocal,perm);CHKERRQ(ierr);
    ierr = ISDestroy(&tmp);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
    
#undef __FUNCT__  
#define __FUNCT__ "ISStrideGetInfo" 
/*@
   ISStrideGetInfo - Returns the first index in a stride index set and 
   the stride width.

   Not Collective

   Input Parameter:
.  is - the index set

   Output Parameters:
.  first - the first index
.  step - the stride width

   Level: intermediate

   Notes:
   Returns info on stride index set. This is a pseudo-public function that
   should not be needed by most users.

   Concepts: index sets^getting information
   Concepts: IS^getting information

.seealso: ISCreateStride(), ISGetSize()
@*/
PetscErrorCode  ISStrideGetInfo(IS is,PetscInt *first,PetscInt *step)
{
  IS_Stride *sub;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  if (first) PetscValidIntPointer(first,2);
  if (step) PetscValidIntPointer(step,3);

  sub = (IS_Stride*)is->data;
  if (first) *first = sub->first; 
  if (step)  *step  = sub->step;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISDestroy_Stride" 
PetscErrorCode ISDestroy_Stride(IS is)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)is,"ISStrideSetStride_C","",0);CHKERRQ(ierr);
  ierr = PetscFree(is->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISToGeneral_Stride" 
PetscErrorCode  ISToGeneral_Stride(IS inis)
{
  PetscErrorCode ierr;
  const PetscInt *idx;
  PetscInt       n;

  PetscFunctionBegin;
  ierr = ISGetLocalSize(inis,&n);CHKERRQ(ierr);
  ierr = ISGetIndices(inis,&idx);CHKERRQ(ierr);
  ierr = ISSetType(inis,ISGENERAL);CHKERRQ(ierr);
  ierr = ISGeneralSetIndices(inis,n,idx,PETSC_OWN_POINTER);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*
     Returns a legitimate index memory even if 
   the stride index set is empty.
*/
#undef __FUNCT__  
#define __FUNCT__ "ISGetIndices_Stride" 
PetscErrorCode ISGetIndices_Stride(IS in,const PetscInt *idx[])
{
  IS_Stride      *sub = (IS_Stride*)in->data;
  PetscErrorCode ierr;
  PetscInt       i,**dx = (PetscInt**)idx;

  PetscFunctionBegin;
  ierr      = PetscMalloc(sub->n*sizeof(PetscInt),idx);CHKERRQ(ierr);
  if (sub->n) {
    (*dx)[0] = sub->first;
    for (i=1; i<sub->n; i++) (*dx)[i] = (*dx)[i-1] + sub->step;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISRestoreIndices_Stride" 
PetscErrorCode ISRestoreIndices_Stride(IS in,const PetscInt *idx[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(*(void**)idx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISGetSize_Stride" 
PetscErrorCode ISGetSize_Stride(IS is,PetscInt *size)
{
  IS_Stride *sub = (IS_Stride *)is->data;

  PetscFunctionBegin;
  *size = sub->N; 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISGetLocalSize_Stride" 
PetscErrorCode ISGetLocalSize_Stride(IS is,PetscInt *size)
{
  IS_Stride *sub = (IS_Stride *)is->data;

  PetscFunctionBegin;
  *size = sub->n; 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISView_Stride" 
PetscErrorCode ISView_Stride(IS is,PetscViewer viewer)
{
  IS_Stride      *sub = (IS_Stride *)is->data;
  PetscInt       i,n = sub->n;
  PetscMPIInt    rank,size;
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) { 
    ierr = MPI_Comm_rank(((PetscObject)is)->comm,&rank);CHKERRQ(ierr);
    ierr = MPI_Comm_size(((PetscObject)is)->comm,&size);CHKERRQ(ierr);
    if (size == 1) {
      if (is->isperm) {
        ierr = PetscViewerASCIIPrintf(viewer,"Index set is permutation\n");CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer,"Number of indices in (stride) set %D\n",n);CHKERRQ(ierr);
      for (i=0; i<n; i++) {
        ierr = PetscViewerASCIIPrintf(viewer,"%D %D\n",i,sub->first + i*sub->step);CHKERRQ(ierr);
      }
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_TRUE);CHKERRQ(ierr);      
      if (is->isperm) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Index set is permutation\n",rank);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Number of indices in (stride) set %D\n",rank,n);CHKERRQ(ierr);
      for (i=0; i<n; i++) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] %D %D\n",rank,i,sub->first + i*sub->step);CHKERRQ(ierr);
      }
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_FALSE);CHKERRQ(ierr);      
    }
  } else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Viewer type %s not supported for this object",((PetscObject)viewer)->type_name);
  PetscFunctionReturn(0);
}
  
#undef __FUNCT__  
#define __FUNCT__ "ISSort_Stride" 
PetscErrorCode ISSort_Stride(IS is)
{
  IS_Stride *sub = (IS_Stride*)is->data;

  PetscFunctionBegin;
  if (sub->step >= 0) PetscFunctionReturn(0);
  sub->first += (sub->n - 1)*sub->step;
  sub->step *= -1;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISSorted_Stride" 
PetscErrorCode ISSorted_Stride(IS is,PetscBool * flg)
{
  IS_Stride *sub = (IS_Stride*)is->data;

  PetscFunctionBegin;
  if (sub->step >= 0) *flg = PETSC_TRUE;
  else *flg = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISOnComm_Stride"
static PetscErrorCode ISOnComm_Stride(IS is,MPI_Comm comm,PetscCopyMode mode,IS *newis)
{
  PetscErrorCode ierr;
  IS_Stride      *sub = (IS_Stride*)is->data;

  PetscFunctionBegin;
  ierr = ISCreateStride(comm,sub->n,sub->first,sub->step,newis);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISSetBlockSize_Stride"
static PetscErrorCode ISSetBlockSize_Stride(IS is,PetscInt bs)
{
  IS_Stride      *sub = (IS_Stride*)is->data;

  PetscFunctionBegin;
  if (sub->step != 1 && bs != 1) SETERRQ2(((PetscObject)is)->comm,PETSC_ERR_ARG_SIZ,"ISSTRIDE has stride %D, cannot be blocked of size %D",sub->step,bs);
  is->bs = bs;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ISContiguousLocal_Stride"
static PetscErrorCode ISContiguousLocal_Stride(IS is,PetscInt gstart,PetscInt gend,PetscInt *start,PetscBool *contig)
{
  IS_Stride *sub = (IS_Stride*)is->data;

  PetscFunctionBegin;
  if (sub->step == 1 && sub->first >= gstart && sub->first+sub->n <= gend) {
    *start  = sub->first - gstart;
    *contig = PETSC_TRUE;
  } else {
    *start  = -1;
    *contig = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}


static struct _ISOps myops = { ISGetSize_Stride,
                               ISGetLocalSize_Stride,
                               ISGetIndices_Stride,
                               ISRestoreIndices_Stride,
                               ISInvertPermutation_Stride,
                               ISSort_Stride,
                               ISSorted_Stride,
                               ISDuplicate_Stride,
                               ISDestroy_Stride,
                               ISView_Stride,
                               ISIdentity_Stride,
                               ISCopy_Stride,
                               ISToGeneral_Stride,
                               ISOnComm_Stride,
                               ISSetBlockSize_Stride,
                               ISContiguousLocal_Stride
};


#undef __FUNCT__  
#define __FUNCT__ "ISStrideSetStride" 
/*@
   ISStrideSetStride - Sets the stride information for a stride index set.

   Collective on IS

   Input Parameters:
+  is - the index set
.  n - the length of the locally owned portion of the index set
.  first - the first element of the locally owned portion of the index set
-  step - the change to the next index

   Level: beginner

  Concepts: IS^stride
  Concepts: index sets^stride
  Concepts: stride^index set

.seealso: ISCreateGeneral(), ISCreateBlock(), ISAllGather()
@*/
PetscErrorCode  ISStrideSetStride(IS is,PetscInt n,PetscInt first,PetscInt step)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (n < 0) SETERRQ1(((PetscObject) is)->comm, PETSC_ERR_ARG_OUTOFRANGE, "Negative length %d not valid", n);
  ierr = PetscUseMethod(is,"ISStrideSetStride_C",(IS,PetscInt,PetscInt,PetscInt),(is,n,first,step));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "ISStrideSetStride_Stride" 
PetscErrorCode  ISStrideSetStride_Stride(IS is,PetscInt n,PetscInt first,PetscInt step)
{
  PetscErrorCode ierr;
  PetscInt       min,max;
  IS_Stride      *sub = (IS_Stride*)is->data;

  PetscFunctionBegin;
  sub->n         = n;
  ierr = MPI_Allreduce(&n,&sub->N,1,MPIU_INT,MPI_SUM,((PetscObject)is)->comm);CHKERRQ(ierr);
  sub->first     = first;
  sub->step      = step;
  if (step > 0) {min = first; max = first + step*(n-1);}
  else          {max = first; min = first + step*(n-1);}

  is->min     = min;
  is->max     = max;
  is->data    = (void*)sub;

  if ((!first && step == 1) || (first == max && step == -1 && !min)) {
    is->isperm  = PETSC_TRUE;
  } else {
    is->isperm  = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "ISCreateStride" 
/*@
   ISCreateStride - Creates a data structure for an index set 
   containing a list of evenly spaced integers.

   Collective on MPI_Comm

   Input Parameters:
+  comm - the MPI communicator
.  n - the length of the locally owned portion of the index set
.  first - the first element of the locally owned portion of the index set
-  step - the change to the next index

   Output Parameter:
.  is - the new index set

   Notes: 
   When the communicator is not MPI_COMM_SELF, the operations on IS are NOT
   conceptually the same as MPI_Group operations. The IS are the 
   distributed sets of indices and thus certain operations on them are collective. 

   Level: beginner

  Concepts: IS^stride
  Concepts: index sets^stride
  Concepts: stride^index set

.seealso: ISCreateGeneral(), ISCreateBlock(), ISAllGather()
@*/
PetscErrorCode  ISCreateStride(MPI_Comm comm,PetscInt n,PetscInt first,PetscInt step,IS *is)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = ISCreate(comm,is);CHKERRQ(ierr);
  ierr = ISSetType(*is,ISSTRIDE);CHKERRQ(ierr);
  ierr = ISStrideSetStride(*is,n,first,step);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "ISCreate_Stride" 
PetscErrorCode  ISCreate_Stride(IS is)
{
  PetscErrorCode ierr;
  IS_Stride      *sub;

  PetscFunctionBegin;
  ierr = PetscMemcpy(is->ops,&myops,sizeof(myops));CHKERRQ(ierr);
  ierr = PetscNewLog(is,IS_Stride,&sub);CHKERRQ(ierr);
  is->bs   = 1;
  is->data = sub;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)is,"ISStrideSetStride_C","ISStrideSetStride_Stride",ISStrideSetStride_Stride);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END



