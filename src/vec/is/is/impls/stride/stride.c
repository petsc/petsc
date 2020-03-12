
/*
       Index sets of evenly space integers, defined by a
    start, stride and length.
*/
#include <petsc/private/isimpl.h>             /*I   "petscis.h"   I*/
#include <petscviewer.h>

typedef struct {
  PetscInt first,step;
} IS_Stride;

static PetscErrorCode ISCopy_Stride(IS is,IS isy)
{
  IS_Stride      *is_stride = (IS_Stride*)is->data,*isy_stride = (IS_Stride*)isy->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMemcpy(isy_stride,is_stride,sizeof(IS_Stride));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode ISDuplicate_Stride(IS is,IS *newIS)
{
  PetscErrorCode ierr;
  IS_Stride      *sub = (IS_Stride*)is->data;

  PetscFunctionBegin;
  ierr = ISCreateStride(PetscObjectComm((PetscObject)is),is->map->n,sub->first,sub->step,newIS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode ISInvertPermutation_Stride(IS is,PetscInt nlocal,IS *perm)
{
  PetscBool      isident;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = ISGetInfo(is,IS_IDENTITY,IS_GLOBAL,PETSC_TRUE,&isident);CHKERRQ(ierr);
  if (isident) {
    PetscInt rStart, rEnd;

    ierr = PetscLayoutGetRange(is->map, &rStart, &rEnd);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,PetscMax(rEnd - rStart, 0),rStart,1,perm);CHKERRQ(ierr);
  } else {
    IS             tmp;
    const PetscInt *indices,n = is->map->n;

    ierr = ISGetIndices(is,&indices);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PetscObjectComm((PetscObject)is),n,indices,PETSC_COPY_VALUES,&tmp);CHKERRQ(ierr);
    ierr = ISSetPermutation(tmp);CHKERRQ(ierr);
    ierr = ISRestoreIndices(is,&indices);CHKERRQ(ierr);
    ierr = ISInvertPermutation(tmp,nlocal,perm);CHKERRQ(ierr);
    ierr = ISDestroy(&tmp);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
   ISStrideGetInfo - Returns the first index in a stride index set and
   the stride width.

   Not Collective

   Input Parameter:
.  is - the index set

   Output Parameters:
+  first - the first index
-  step - the stride width

   Level: intermediate

   Notes:
   Returns info on stride index set. This is a pseudo-public function that
   should not be needed by most users.


.seealso: ISCreateStride(), ISGetSize()
@*/
PetscErrorCode  ISStrideGetInfo(IS is,PetscInt *first,PetscInt *step)
{
  IS_Stride      *sub;
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  if (first) PetscValidIntPointer(first,2);
  if (step) PetscValidIntPointer(step,3);
  ierr = PetscObjectTypeCompare((PetscObject)is,ISSTRIDE,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PetscObjectComm((PetscObject)is),PETSC_ERR_ARG_WRONG,"IS must be of type ISSTRIDE");

  sub = (IS_Stride*)is->data;
  if (first) *first = sub->first;
  if (step)  *step  = sub->step;
  PetscFunctionReturn(0);
}

PetscErrorCode ISDestroy_Stride(IS is)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectComposeFunction((PetscObject)is,"ISStrideSetStride_C",NULL);CHKERRQ(ierr);
  ierr = PetscFree(is->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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

PetscErrorCode ISLocate_Stride(IS is,PetscInt key,PetscInt *location)
{
  IS_Stride      *sub = (IS_Stride*)is->data;
  PetscInt       rem, step;

  PetscFunctionBegin;
  *location = -1;
  step      = sub->step;
  key      -= sub->first;
  rem       = key / step;
  if ((rem < is->map->n) && !(key % step)) {
    *location = rem;
  }
  PetscFunctionReturn(0);
}

/*
     Returns a legitimate index memory even if
   the stride index set is empty.
*/
PetscErrorCode ISGetIndices_Stride(IS is,const PetscInt *idx[])
{
  IS_Stride      *sub = (IS_Stride*)is->data;
  PetscErrorCode ierr;
  PetscInt       i,**dx = (PetscInt**)idx;

  PetscFunctionBegin;
  ierr      = PetscMalloc1(is->map->n,(PetscInt**)idx);CHKERRQ(ierr);
  if (is->map->n) {
    (*dx)[0] = sub->first;
    for (i=1; i<is->map->n; i++) (*dx)[i] = (*dx)[i-1] + sub->step;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode ISRestoreIndices_Stride(IS in,const PetscInt *idx[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(*(void**)idx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode ISView_Stride(IS is,PetscViewer viewer)
{
  IS_Stride         *sub = (IS_Stride*)is->data;
  PetscInt          i,n = is->map->n;
  PetscMPIInt       rank,size;
  PetscBool         iascii,ibinary;
  PetscViewerFormat fmt;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&ibinary);CHKERRQ(ierr);
  if (iascii) {
    PetscBool matl, isperm;

    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)is),&rank);CHKERRQ(ierr);
    ierr = MPI_Comm_size(PetscObjectComm((PetscObject)is),&size);CHKERRQ(ierr);
    ierr = PetscViewerGetFormat(viewer,&fmt);CHKERRQ(ierr);
    matl = (PetscBool)(fmt == PETSC_VIEWER_ASCII_MATLAB);
    ierr = ISGetInfo(is,IS_PERMUTATION,IS_GLOBAL,PETSC_FALSE,&isperm);CHKERRQ(ierr);
    if (isperm && !matl) {ierr = PetscViewerASCIIPrintf(viewer,"Index set is permutation\n");CHKERRQ(ierr);}
    if (size == 1) {
      if (matl) {
        const char* name;

        ierr = PetscObjectGetName((PetscObject)is,&name);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"%s = [%D : %D : %D];\n",name,sub->first+1,sub->step,sub->first + sub->step*(n-1)+1);CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(viewer,"Number of indices in (stride) set %D\n",n);CHKERRQ(ierr);
        for (i=0; i<n; i++) {
          ierr = PetscViewerASCIIPrintf(viewer,"%D %D\n",i,sub->first + i*sub->step);CHKERRQ(ierr);
        }
      }
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
      if (matl) {
        const char* name;

        ierr = PetscObjectGetName((PetscObject)is,&name);CHKERRQ(ierr);
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"%s_%d = [%D : %D : %D];\n",name,rank,sub->first+1,sub->step,sub->first + sub->step*(n-1)+1);CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Number of indices in (stride) set %D\n",rank,n);CHKERRQ(ierr);
        for (i=0; i<n; i++) {
          ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d] %D %D\n",rank,i,sub->first + i*sub->step);CHKERRQ(ierr);
        }
      }
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);
    }
  } else if (ibinary) {
    ierr = ISView_Binary(is,viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode ISSort_Stride(IS is)
{
  IS_Stride *sub = (IS_Stride*)is->data;

  PetscFunctionBegin;
  if (sub->step >= 0) PetscFunctionReturn(0);
  sub->first += (is->map->n - 1)*sub->step;
  sub->step  *= -1;
  PetscFunctionReturn(0);
}

PetscErrorCode ISSorted_Stride(IS is,PetscBool * flg)
{
  IS_Stride *sub = (IS_Stride*)is->data;

  PetscFunctionBegin;
  if (sub->step >= 0) *flg = PETSC_TRUE;
  else *flg = PETSC_FALSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode ISUniqueLocal_Stride(IS is, PetscBool *flg)
{
  IS_Stride *sub = (IS_Stride*)is->data;

  PetscFunctionBegin;
  if (!(is->map->n) || sub->step != 0) *flg = PETSC_TRUE;
  else *flg = PETSC_FALSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode ISPermutationLocal_Stride(IS is, PetscBool *flg)
{
  IS_Stride *sub = (IS_Stride*)is->data;

  PetscFunctionBegin;
  if (!(is->map->n) || (PetscAbsInt(sub->step) == 1 && is->min == 0)) *flg = PETSC_TRUE;
  else *flg = PETSC_FALSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode ISIntervalLocal_Stride(IS is, PetscBool *flg)
{
  IS_Stride *sub = (IS_Stride*)is->data;

  PetscFunctionBegin;
  if (!(is->map->n) || sub->step == 1) *flg = PETSC_TRUE;
  else *flg = PETSC_FALSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode ISOnComm_Stride(IS is,MPI_Comm comm,PetscCopyMode mode,IS *newis)
{
  PetscErrorCode ierr;
  IS_Stride      *sub = (IS_Stride*)is->data;

  PetscFunctionBegin;
  ierr = ISCreateStride(comm,is->map->n,sub->first,sub->step,newis);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ISSetBlockSize_Stride(IS is,PetscInt bs)
{
  IS_Stride     *sub = (IS_Stride*)is->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (sub->step != 1 && bs != 1) SETERRQ2(PetscObjectComm((PetscObject)is),PETSC_ERR_ARG_SIZ,"ISSTRIDE has stride %D, cannot be blocked of size %D",sub->step,bs);
  ierr = PetscLayoutSetBlockSize(is->map, bs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ISContiguousLocal_Stride(IS is,PetscInt gstart,PetscInt gend,PetscInt *start,PetscBool *contig)
{
  IS_Stride *sub = (IS_Stride*)is->data;

  PetscFunctionBegin;
  if (sub->step == 1 && sub->first >= gstart && sub->first+is->map->n <= gend) {
    *start  = sub->first - gstart;
    *contig = PETSC_TRUE;
  } else {
    *start  = -1;
    *contig = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}


static struct _ISOps myops = { ISGetIndices_Stride,
                               ISRestoreIndices_Stride,
                               ISInvertPermutation_Stride,
                               ISSort_Stride,
                               ISSort_Stride,
                               ISSorted_Stride,
                               ISDuplicate_Stride,
                               ISDestroy_Stride,
                               ISView_Stride,
                               ISLoad_Default,
                               ISCopy_Stride,
                               ISToGeneral_Stride,
                               ISOnComm_Stride,
                               ISSetBlockSize_Stride,
                               ISContiguousLocal_Stride,
                               ISLocate_Stride,
                               ISSorted_Stride,
                               NULL,
                               ISUniqueLocal_Stride,
                               NULL,
                               ISPermutationLocal_Stride,
                               NULL,
                               ISIntervalLocal_Stride,
                               NULL};


/*@
   ISStrideSetStride - Sets the stride information for a stride index set.

   Collective on IS

   Input Parameters:
+  is - the index set
.  n - the length of the locally owned portion of the index set
.  first - the first element of the locally owned portion of the index set
-  step - the change to the next index

   Level: beginner

.seealso: ISCreateGeneral(), ISCreateBlock(), ISAllGather()
@*/
PetscErrorCode  ISStrideSetStride(IS is,PetscInt n,PetscInt first,PetscInt step)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (n < 0) SETERRQ1(PetscObjectComm((PetscObject)is), PETSC_ERR_ARG_OUTOFRANGE, "Negative length %D not valid", n);
  ierr = ISClearInfoCache(is,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscUseMethod(is,"ISStrideSetStride_C",(IS,PetscInt,PetscInt,PetscInt),(is,n,first,step));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode  ISStrideSetStride_Stride(IS is,PetscInt n,PetscInt first,PetscInt step)
{
  PetscErrorCode ierr;
  PetscInt       min,max;
  IS_Stride      *sub = (IS_Stride*)is->data;
  PetscLayout    map;

  PetscFunctionBegin;
  ierr = PetscLayoutCreateFromSizes(PetscObjectComm((PetscObject)is),n,is->map->N,is->map->bs,&map);CHKERRQ(ierr);
  ierr = PetscLayoutDestroy(&is->map);CHKERRQ(ierr);
  is->map = map;

  sub->first = first;
  sub->step  = step;
  if (step > 0) {min = first; max = first + step*(n-1);}
  else          {max = first; min = first + step*(n-1);}

  is->min  = n > 0 ? min : PETSC_MAX_INT;
  is->max  = n > 0 ? max : PETSC_MIN_INT;
  is->data = (void*)sub;
  PetscFunctionReturn(0);
}

/*@
   ISCreateStride - Creates a data structure for an index set
   containing a list of evenly spaced integers.

   Collective

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

PETSC_EXTERN PetscErrorCode ISCreate_Stride(IS is)
{
  PetscErrorCode ierr;
  IS_Stride      *sub;

  PetscFunctionBegin;
  ierr = PetscNewLog(is,&sub);CHKERRQ(ierr);
  is->data = (void *) sub;
  ierr = PetscMemcpy(is->ops,&myops,sizeof(myops));CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)is,"ISStrideSetStride_C",ISStrideSetStride_Stride);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
