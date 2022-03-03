
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

  PetscFunctionBegin;
  CHKERRQ(PetscMemcpy(isy_stride,is_stride,sizeof(IS_Stride)));
  PetscFunctionReturn(0);
}

PetscErrorCode ISDuplicate_Stride(IS is,IS *newIS)
{
  IS_Stride      *sub = (IS_Stride*)is->data;

  PetscFunctionBegin;
  CHKERRQ(ISCreateStride(PetscObjectComm((PetscObject)is),is->map->n,sub->first,sub->step,newIS));
  PetscFunctionReturn(0);
}

PetscErrorCode ISInvertPermutation_Stride(IS is,PetscInt nlocal,IS *perm)
{
  PetscBool      isident;

  PetscFunctionBegin;
  CHKERRQ(ISGetInfo(is,IS_IDENTITY,IS_GLOBAL,PETSC_TRUE,&isident));
  if (isident) {
    PetscInt rStart, rEnd;

    CHKERRQ(PetscLayoutGetRange(is->map, &rStart, &rEnd));
    CHKERRQ(ISCreateStride(PETSC_COMM_SELF,PetscMax(rEnd - rStart, 0),rStart,1,perm));
  } else {
    IS             tmp;
    const PetscInt *indices,n = is->map->n;

    CHKERRQ(ISGetIndices(is,&indices));
    CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)is),n,indices,PETSC_COPY_VALUES,&tmp));
    CHKERRQ(ISSetPermutation(tmp));
    CHKERRQ(ISRestoreIndices(is,&indices));
    CHKERRQ(ISInvertPermutation(tmp,nlocal,perm));
    CHKERRQ(ISDestroy(&tmp));
  }
  PetscFunctionReturn(0);
}

/*@
   ISStrideGetInfo - Returns the first index in a stride index set and the stride width.

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

.seealso: ISCreateStride(), ISGetSize(), ISSTRIDE
@*/
PetscErrorCode  ISStrideGetInfo(IS is,PetscInt *first,PetscInt *step)
{
  IS_Stride      *sub;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is,IS_CLASSID,1);
  if (first) PetscValidIntPointer(first,2);
  if (step) PetscValidIntPointer(step,3);
  CHKERRQ(PetscObjectTypeCompare((PetscObject)is,ISSTRIDE,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)is),PETSC_ERR_ARG_WRONG,"IS must be of type ISSTRIDE");

  sub = (IS_Stride*)is->data;
  if (first) *first = sub->first;
  if (step)  *step  = sub->step;
  PetscFunctionReturn(0);
}

PetscErrorCode ISDestroy_Stride(IS is)
{
  PetscFunctionBegin;
  CHKERRQ(PetscObjectComposeFunction((PetscObject)is,"ISStrideSetStride_C",NULL));
  CHKERRQ(PetscFree(is->data));
  PetscFunctionReturn(0);
}

PetscErrorCode  ISToGeneral_Stride(IS inis)
{
  const PetscInt *idx;
  PetscInt       n;

  PetscFunctionBegin;
  CHKERRQ(ISGetLocalSize(inis,&n));
  CHKERRQ(ISGetIndices(inis,&idx));
  CHKERRQ(ISSetType(inis,ISGENERAL));
  CHKERRQ(ISGeneralSetIndices(inis,n,idx,PETSC_OWN_POINTER));
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
  PetscInt       i,**dx = (PetscInt**)idx;

  PetscFunctionBegin;
  CHKERRQ(PetscMalloc1(is->map->n,(PetscInt**)idx));
  if (is->map->n) {
    (*dx)[0] = sub->first;
    for (i=1; i<is->map->n; i++) (*dx)[i] = (*dx)[i-1] + sub->step;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode ISRestoreIndices_Stride(IS in,const PetscInt *idx[])
{
  PetscFunctionBegin;
  CHKERRQ(PetscFree(*(void**)idx));
  PetscFunctionReturn(0);
}

PetscErrorCode ISView_Stride(IS is,PetscViewer viewer)
{
  IS_Stride         *sub = (IS_Stride*)is->data;
  PetscInt          i,n = is->map->n;
  PetscMPIInt       rank,size;
  PetscBool         iascii,ibinary;
  PetscViewerFormat fmt;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&ibinary));
  if (iascii) {
    PetscBool matl, isperm;

    CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)is),&rank));
    CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)is),&size));
    CHKERRQ(PetscViewerGetFormat(viewer,&fmt));
    matl = (PetscBool)(fmt == PETSC_VIEWER_ASCII_MATLAB);
    CHKERRQ(ISGetInfo(is,IS_PERMUTATION,IS_GLOBAL,PETSC_FALSE,&isperm));
    if (isperm && !matl) CHKERRQ(PetscViewerASCIIPrintf(viewer,"Index set is permutation\n"));
    if (size == 1) {
      if (matl) {
        const char* name;

        CHKERRQ(PetscObjectGetName((PetscObject)is,&name));
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"%s = [%" PetscInt_FMT " : %" PetscInt_FMT " : %" PetscInt_FMT "];\n",name,sub->first+1,sub->step,sub->first + sub->step*(n-1)+1));
      } else {
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"Number of indices in (stride) set %" PetscInt_FMT "\n",n));
        for (i=0; i<n; i++) {
          CHKERRQ(PetscViewerASCIIPrintf(viewer,"%" PetscInt_FMT " %" PetscInt_FMT "\n",i,sub->first + i*sub->step));
        }
      }
      CHKERRQ(PetscViewerFlush(viewer));
    } else {
      CHKERRQ(PetscViewerASCIIPushSynchronized(viewer));
      if (matl) {
        const char* name;

        CHKERRQ(PetscObjectGetName((PetscObject)is,&name));
        CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer,"%s_%d = [%" PetscInt_FMT " : %" PetscInt_FMT " : %" PetscInt_FMT "];\n",name,rank,sub->first+1,sub->step,sub->first + sub->step*(n-1)+1));
      } else {
        CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer,"[%d] Number of indices in (stride) set %" PetscInt_FMT "\n",rank,n));
        for (i=0; i<n; i++) {
          CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer,"[%d] %" PetscInt_FMT " %" PetscInt_FMT "\n",rank,i,sub->first + i*sub->step));
        }
      }
      CHKERRQ(PetscViewerFlush(viewer));
      CHKERRQ(PetscViewerASCIIPopSynchronized(viewer));
    }
  } else if (ibinary) {
    CHKERRQ(ISView_Binary(is,viewer));
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
  IS_Stride      *sub = (IS_Stride*)is->data;

  PetscFunctionBegin;
  CHKERRQ(ISCreateStride(comm,is->map->n,sub->first,sub->step,newis));
  PetscFunctionReturn(0);
}

static PetscErrorCode ISSetBlockSize_Stride(IS is,PetscInt bs)
{
  IS_Stride     *sub = (IS_Stride*)is->data;

  PetscFunctionBegin;
  PetscCheckFalse(sub->step != 1 && bs != 1,PetscObjectComm((PetscObject)is),PETSC_ERR_ARG_SIZ,"ISSTRIDE has stride %" PetscInt_FMT ", cannot be blocked of size %" PetscInt_FMT,sub->step,bs);
  CHKERRQ(PetscLayoutSetBlockSize(is->map, bs));
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

static struct _ISOps myops = {
  PetscDesignatedInitializer(getindices,ISGetIndices_Stride),
  PetscDesignatedInitializer(restoreindices,ISRestoreIndices_Stride),
  PetscDesignatedInitializer(invertpermutation,ISInvertPermutation_Stride),
  PetscDesignatedInitializer(sort,ISSort_Stride),
  PetscDesignatedInitializer(sortremovedups,ISSort_Stride),
  PetscDesignatedInitializer(sorted,ISSorted_Stride),
  PetscDesignatedInitializer(duplicate,ISDuplicate_Stride),
  PetscDesignatedInitializer(destroy,ISDestroy_Stride),
  PetscDesignatedInitializer(view,ISView_Stride),
  PetscDesignatedInitializer(load,ISLoad_Default),
  PetscDesignatedInitializer(copy,ISCopy_Stride),
  PetscDesignatedInitializer(togeneral,ISToGeneral_Stride),
  PetscDesignatedInitializer(oncomm,ISOnComm_Stride),
  PetscDesignatedInitializer(setblocksize,ISSetBlockSize_Stride),
  PetscDesignatedInitializer(contiguous,ISContiguousLocal_Stride),
  PetscDesignatedInitializer(locate,ISLocate_Stride),
  PetscDesignatedInitializer(sortedlocal,ISSorted_Stride),
  NULL,
  PetscDesignatedInitializer(uniquelocal,ISUniqueLocal_Stride),
  NULL,
  PetscDesignatedInitializer(permlocal,ISPermutationLocal_Stride),
  NULL,
  PetscDesignatedInitializer(intervallocal,ISIntervalLocal_Stride),
  NULL
};

/*@
   ISStrideSetStride - Sets the stride information for a stride index set.

   Collective on IS

   Input Parameters:
+  is - the index set
.  n - the length of the locally owned portion of the index set
.  first - the first element of the locally owned portion of the index set
-  step - the change to the next index

   Level: beginner

.seealso: ISCreateGeneral(), ISCreateBlock(), ISAllGather(), ISSTRIDE, ISCreateStride(), ISStrideGetInfo()
@*/
PetscErrorCode  ISStrideSetStride(IS is,PetscInt n,PetscInt first,PetscInt step)
{
  PetscFunctionBegin;
  PetscCheckFalse(n < 0,PetscObjectComm((PetscObject)is), PETSC_ERR_ARG_OUTOFRANGE, "Negative length %" PetscInt_FMT " not valid", n);
  CHKERRQ(ISClearInfoCache(is,PETSC_FALSE));
  CHKERRQ(PetscUseMethod(is,"ISStrideSetStride_C",(IS,PetscInt,PetscInt,PetscInt),(is,n,first,step)));
  PetscFunctionReturn(0);
}

PetscErrorCode  ISStrideSetStride_Stride(IS is,PetscInt n,PetscInt first,PetscInt step)
{
  PetscInt       min,max;
  IS_Stride      *sub = (IS_Stride*)is->data;
  PetscLayout    map;

  PetscFunctionBegin;
  CHKERRQ(PetscLayoutCreateFromSizes(PetscObjectComm((PetscObject)is),n,is->map->N,is->map->bs,&map));
  CHKERRQ(PetscLayoutDestroy(&is->map));
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

.seealso: ISCreateGeneral(), ISCreateBlock(), ISAllGather(), ISSTRIDE
@*/
PetscErrorCode  ISCreateStride(MPI_Comm comm,PetscInt n,PetscInt first,PetscInt step,IS *is)
{
  PetscFunctionBegin;
  CHKERRQ(ISCreate(comm,is));
  CHKERRQ(ISSetType(*is,ISSTRIDE));
  CHKERRQ(ISStrideSetStride(*is,n,first,step));
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode ISCreate_Stride(IS is)
{
  IS_Stride      *sub;

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(is,&sub));
  is->data = (void *) sub;
  CHKERRQ(PetscMemcpy(is->ops,&myops,sizeof(myops)));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)is,"ISStrideSetStride_C",ISStrideSetStride_Stride));
  PetscFunctionReturn(0);
}
