#include <petsc/private/matimpl.h>      /*I "petscmat.h" I*/
#include <petsc/private/hashsetij.h>

typedef struct {
  PetscHSetIJ ht;
  PetscInt   *dnz, *onz;
  PetscInt   *dnzu, *onzu;
  PetscBool   nooffproc;
  PetscBool   used;
} Mat_Preallocator;

PetscErrorCode MatDestroy_Preallocator(Mat A)
{
  Mat_Preallocator *p = (Mat_Preallocator *) A->data;

  PetscFunctionBegin;
  PetscCall(MatStashDestroy_Private(&A->stash));
  PetscCall(PetscHSetIJDestroy(&p->ht));
  PetscCall(PetscFree4(p->dnz, p->onz, p->dnzu, p->onzu));
  PetscCall(PetscFree(A->data));
  PetscCall(PetscObjectChangeTypeName((PetscObject) A, NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject) A, "MatPreallocatorPreallocate_C", NULL));
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetUp_Preallocator(Mat A)
{
  Mat_Preallocator *p = (Mat_Preallocator *) A->data;
  PetscInt          m, bs, mbs;

  PetscFunctionBegin;
  PetscCall(PetscLayoutSetUp(A->rmap));
  PetscCall(PetscLayoutSetUp(A->cmap));
  PetscCall(MatGetLocalSize(A, &m, NULL));
  PetscCall(PetscHSetIJCreate(&p->ht));
  PetscCall(MatGetBlockSize(A, &bs));
  /* Do not bother bstash since MatPreallocator does not implement MatSetValuesBlocked */
  PetscCall(MatStashCreate_Private(PetscObjectComm((PetscObject) A), 1, &A->stash));
  /* arrays are for blocked rows/cols */
  mbs  = m/bs;
  PetscCall(PetscCalloc4(mbs, &p->dnz, mbs, &p->onz, mbs, &p->dnzu, mbs, &p->onzu));
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetValues_Preallocator(Mat A, PetscInt m, const PetscInt *rows, PetscInt n, const PetscInt *cols, const PetscScalar *values, InsertMode addv)
{
  Mat_Preallocator *p = (Mat_Preallocator *) A->data;
  PetscInt          rStart, rEnd, r, cStart, cEnd, c, bs;

  PetscFunctionBegin;
  PetscCall(MatGetBlockSize(A, &bs));
  PetscCall(MatGetOwnershipRange(A, &rStart, &rEnd));
  PetscCall(MatGetOwnershipRangeColumn(A, &cStart, &cEnd));
  for (r = 0; r < m; ++r) {
    PetscHashIJKey key;
    PetscBool      missing;

    key.i = rows[r];
    if (key.i < 0) continue;
    if ((key.i < rStart) || (key.i >= rEnd)) {
      PetscCall(MatStashValuesRow_Private(&A->stash, key.i, n, cols, values, PETSC_FALSE));
    } else { /* Hash table is for blocked rows/cols */
      key.i = rows[r]/bs;
      for (c = 0; c < n; ++c) {
        key.j = cols[c]/bs;
        if (key.j < 0) continue;
        PetscCall(PetscHSetIJQueryAdd(p->ht, key, &missing));
        if (missing) {
          if ((key.j >= cStart/bs) && (key.j < cEnd/bs)) {
            ++p->dnz[key.i-rStart/bs];
            if (key.j >= key.i) ++p->dnzu[key.i-rStart/bs];
          } else {
            ++p->onz[key.i-rStart/bs];
            if (key.j >= key.i) ++p->onzu[key.i-rStart/bs];
          }
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatAssemblyBegin_Preallocator(Mat A, MatAssemblyType type)
{
  PetscInt       nstash, reallocs;

  PetscFunctionBegin;
  PetscCall(MatStashScatterBegin_Private(A, &A->stash, A->rmap->range));
  PetscCall(MatStashGetInfo_Private(&A->stash, &nstash, &reallocs));
  PetscCall(PetscInfo(NULL, "Stash has %" PetscInt_FMT " entries, uses %" PetscInt_FMT " mallocs.\n", nstash, reallocs));
  PetscFunctionReturn(0);
}

PetscErrorCode MatAssemblyEnd_Preallocator(Mat A, MatAssemblyType type)
{
  PetscScalar      *val;
  PetscInt         *row, *col;
  PetscInt         i, j, rstart, ncols, flg;
  PetscMPIInt      n;
  Mat_Preallocator *p = (Mat_Preallocator *) A->data;

  PetscFunctionBegin;
  p->nooffproc = PETSC_TRUE;
  while (1) {
    PetscCall(MatStashScatterGetMesg_Private(&A->stash, &n, &row, &col, &val, &flg));
    if (flg) p->nooffproc = PETSC_FALSE;
    if (!flg) break;

    for (i = 0; i < n;) {
      /* Now identify the consecutive vals belonging to the same row */
      for (j = i, rstart = row[j]; j < n; j++) {
        if (row[j] != rstart) break;
      }
      if (j < n) ncols = j-i;
      else       ncols = n-i;
      /* Now assemble all these values with a single function call */
      PetscCall(MatSetValues_Preallocator(A, 1, row+i, ncols, col+i, val+i, INSERT_VALUES));
      i = j;
    }
  }
  PetscCall(MatStashScatterEnd_Private(&A->stash));
  PetscCall(MPIU_Allreduce(MPI_IN_PLACE,&p->nooffproc,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)A)));
  PetscFunctionReturn(0);
}

PetscErrorCode MatView_Preallocator(Mat A, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetOption_Preallocator(Mat A, MatOption op, PetscBool flg)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode MatPreallocatorPreallocate_Preallocator(Mat mat, PetscBool fill, Mat A)
{
  Mat_Preallocator *p = (Mat_Preallocator *) mat->data;
  PetscInt          bs;

  PetscFunctionBegin;
  PetscCheck(!p->used,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"MatPreallocatorPreallocate() can only be used once for a give MatPreallocator object. Consider using MatDuplicate() after preallocation.");
  p->used = PETSC_TRUE;
  if (!fill) PetscCall(PetscHSetIJDestroy(&p->ht));
  PetscCall(MatGetBlockSize(mat, &bs));
  PetscCall(MatXAIJSetPreallocation(A, bs, p->dnz, p->onz, p->dnzu, p->onzu));
  PetscCall(MatSetUp(A));
  PetscCall(MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
  if (fill) {
    PetscCall(MatSetOption(A, MAT_NO_OFF_PROC_ENTRIES, p->nooffproc));
    PetscHashIter  hi;
    PetscHashIJKey key;
    PetscScalar    *zeros;
    PetscInt       n,maxrow=1,*cols,rStart,rEnd,*rowstarts;

    PetscCall(MatGetOwnershipRange(A, &rStart, &rEnd));
    // Ownership range is in terms of scalar entries, but we deal with blocks
    rStart /= bs;
    rEnd /= bs;
    PetscCall(PetscHSetIJGetSize(p->ht,&n));
    PetscCall(PetscMalloc2(n,&cols,rEnd-rStart+1,&rowstarts));
    rowstarts[0] = 0;
    for (PetscInt i=0; i<rEnd-rStart; i++) {
      rowstarts[i+1] = rowstarts[i] + p->dnz[i] + p->onz[i];
      maxrow = PetscMax(maxrow, p->dnz[i] + p->onz[i]);
    }
    PetscCheck(rowstarts[rEnd-rStart] == n,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Hash claims %" PetscInt_FMT " entries, but dnz+onz counts %" PetscInt_FMT,n,rowstarts[rEnd-rStart]);

    PetscHashIterBegin(p->ht,hi);
    for (PetscInt i=0; !PetscHashIterAtEnd(p->ht,hi); i++) {
      PetscHashIterGetKey(p->ht,hi,key);
      PetscInt lrow = key.i - rStart;
      cols[rowstarts[lrow]] = key.j;
      rowstarts[lrow]++;
      PetscHashIterNext(p->ht,hi);
    }
    PetscCall(PetscHSetIJDestroy(&p->ht));

    PetscCall(PetscCalloc1(maxrow*bs*bs,&zeros));
    for (PetscInt i=0; i<rEnd-rStart; i++) {
      PetscInt grow = rStart + i;
      PetscInt end = rowstarts[i], start = end - p->dnz[i] - p->onz[i];
      PetscCall(PetscSortInt(end-start,&cols[start]));
      PetscCall(MatSetValuesBlocked(A, 1, &grow, end-start, &cols[start], zeros, INSERT_VALUES));
    }
    PetscCall(PetscFree(zeros));
    PetscCall(PetscFree2(cols,rowstarts));

    PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
    PetscCall(MatSetOption(A, MAT_NO_OFF_PROC_ENTRIES, PETSC_FALSE));
  }
  PetscFunctionReturn(0);
}

/*@
  MatPreallocatorPreallocate - Preallocates the A matrix, using information from mat, optionally filling A with zeros

  Input Parameters:
+ mat  - the preallocator
. fill - fill the matrix with zeros
- A    - the matrix to be preallocated

  Notes:
  This Mat implementation provides a helper utility to define the correct
  preallocation data for a given nonzero structure. Use this object like a
  regular matrix, e.g. loop over the nonzero structure of the matrix and
  call MatSetValues() or MatSetValuesBlocked() to indicate the nonzero locations.
  The matrix entries provided to MatSetValues() will be ignored, it only uses
  the row / col indices provided to determine the information required to be
  passed to MatXAIJSetPreallocation(). Once you have looped over the nonzero
  structure, you must call MatAssemblyBegin(), MatAssemblyEnd() on mat.

  After you have assembled the preallocator matrix (mat), call MatPreallocatorPreallocate()
  to define the preallocation information on the matrix (A). Setting the parameter
  fill = PETSC_TRUE will insert zeros into the matrix A. Internally MatPreallocatorPreallocate()
  will call MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE);

  This function may only be called once for a given MatPreallocator object. If
  multiple Mats need to be preallocated, consider using MatDuplicate() after
  this function.

  Level: advanced

.seealso: `MATPREALLOCATOR`
@*/
PetscErrorCode MatPreallocatorPreallocate(Mat mat, PetscBool fill, Mat A)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  PetscValidLogicalCollectiveBool(mat,fill,2);
  PetscValidHeaderSpecific(A,MAT_CLASSID,3);
  PetscUseMethod(mat,"MatPreallocatorPreallocate_C",(Mat,PetscBool,Mat),(mat,fill,A));
  PetscFunctionReturn(0);
}

/*MC
   MATPREALLOCATOR - MATPREALLOCATOR = "preallocator" - A matrix type to be used for computing a matrix preallocation.

   Operations Provided:
.vb
  MatSetValues()
.ve

   Options Database Keys:
. -mat_type preallocator - sets the matrix type to "preallocator" during a call to MatSetFromOptions()

  Level: advanced

.seealso: `Mat`, `MatPreallocatorPreallocate()`

M*/

PETSC_EXTERN PetscErrorCode MatCreate_Preallocator(Mat A)
{
  Mat_Preallocator *p;

  PetscFunctionBegin;
  PetscCall(PetscNewLog(A, &p));
  A->data = (void *) p;

  p->ht   = NULL;
  p->dnz  = NULL;
  p->onz  = NULL;
  p->dnzu = NULL;
  p->onzu = NULL;
  p->used = PETSC_FALSE;

  /* matrix ops */
  PetscCall(PetscMemzero(A->ops, sizeof(struct _MatOps)));

  A->ops->destroy       = MatDestroy_Preallocator;
  A->ops->setup         = MatSetUp_Preallocator;
  A->ops->setvalues     = MatSetValues_Preallocator;
  A->ops->assemblybegin = MatAssemblyBegin_Preallocator;
  A->ops->assemblyend   = MatAssemblyEnd_Preallocator;
  A->ops->view          = MatView_Preallocator;
  A->ops->setoption     = MatSetOption_Preallocator;
  A->ops->setblocksizes = MatSetBlockSizes_Default; /* once set, user is not allowed to change the block sizes */

  /* special MATPREALLOCATOR functions */
  PetscCall(PetscObjectComposeFunction((PetscObject) A, "MatPreallocatorPreallocate_C", MatPreallocatorPreallocate_Preallocator));
  PetscCall(PetscObjectChangeTypeName((PetscObject) A, MATPREALLOCATOR));
  PetscFunctionReturn(0);
}
