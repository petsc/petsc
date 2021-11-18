#include <petsc/private/matimpl.h>      /*I "petscmat.h" I*/
#include <petsc/private/hashsetij.h>

typedef struct {
  PetscHSetIJ ht;
  PetscInt   *dnz, *onz;
  PetscInt   *dnzu, *onzu;
  PetscBool   nooffproc;
} Mat_Preallocator;

PetscErrorCode MatDestroy_Preallocator(Mat A)
{
  Mat_Preallocator *p = (Mat_Preallocator *) A->data;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatStashDestroy_Private(&A->stash);CHKERRQ(ierr);
  ierr = PetscHSetIJDestroy(&p->ht);CHKERRQ(ierr);
  ierr = PetscFree4(p->dnz, p->onz, p->dnzu, p->onzu);CHKERRQ(ierr);
  ierr = PetscFree(A->data);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject) A, NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) A, "MatPreallocatorPreallocate_C", NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetUp_Preallocator(Mat A)
{
  Mat_Preallocator *p = (Mat_Preallocator *) A->data;
  PetscInt          m, bs, mbs;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscLayoutSetUp(A->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(A->cmap);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A, &m, NULL);CHKERRQ(ierr);
  ierr = PetscHSetIJCreate(&p->ht);CHKERRQ(ierr);
  ierr = MatGetBlockSize(A, &bs);CHKERRQ(ierr);
  /* Do not bother bstash since MatPreallocator does not implement MatSetValuesBlocked */
  ierr = MatStashCreate_Private(PetscObjectComm((PetscObject) A), 1, &A->stash);CHKERRQ(ierr);
  /* arrays are for blocked rows/cols */
  mbs  = m/bs;
  ierr = PetscCalloc4(mbs, &p->dnz, mbs, &p->onz, mbs, &p->dnzu, mbs, &p->onzu);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetValues_Preallocator(Mat A, PetscInt m, const PetscInt *rows, PetscInt n, const PetscInt *cols, const PetscScalar *values, InsertMode addv)
{
  Mat_Preallocator *p = (Mat_Preallocator *) A->data;
  PetscInt          rStart, rEnd, r, cStart, cEnd, c, bs;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatGetBlockSize(A, &bs);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A, &rStart, &rEnd);CHKERRQ(ierr);
  ierr = MatGetOwnershipRangeColumn(A, &cStart, &cEnd);CHKERRQ(ierr);
  for (r = 0; r < m; ++r) {
    PetscHashIJKey key;
    PetscBool      missing;

    key.i = rows[r];
    if (key.i < 0) continue;
    if ((key.i < rStart) || (key.i >= rEnd)) {
      ierr = MatStashValuesRow_Private(&A->stash, key.i, n, cols, values, PETSC_FALSE);CHKERRQ(ierr);
    } else { /* Hash table is for blocked rows/cols */
      key.i = rows[r]/bs;
      for (c = 0; c < n; ++c) {
        key.j = cols[c]/bs;
        if (key.j < 0) continue;
        ierr = PetscHSetIJQueryAdd(p->ht, key, &missing);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatStashScatterBegin_Private(A, &A->stash, A->rmap->range);CHKERRQ(ierr);
  ierr = MatStashGetInfo_Private(&A->stash, &nstash, &reallocs);CHKERRQ(ierr);
  ierr = PetscInfo2(A, "Stash has %D entries, uses %D mallocs.\n", nstash, reallocs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatAssemblyEnd_Preallocator(Mat A, MatAssemblyType type)
{
  PetscScalar      *val;
  PetscInt         *row, *col;
  PetscInt         i, j, rstart, ncols, flg;
  PetscMPIInt      n;
  Mat_Preallocator *p = (Mat_Preallocator *) A->data;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  p->nooffproc = PETSC_TRUE;
  while (1) {
    ierr = MatStashScatterGetMesg_Private(&A->stash, &n, &row, &col, &val, &flg);CHKERRQ(ierr);
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
      ierr = MatSetValues_Preallocator(A, 1, row+i, ncols, col+i, val+i, INSERT_VALUES);CHKERRQ(ierr);
      i = j;
    }
  }
  ierr = MatStashScatterEnd_Private(&A->stash);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(MPI_IN_PLACE,&p->nooffproc,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)A));CHKERRMPI(ierr);
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
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (!fill) {ierr = PetscHSetIJDestroy(&p->ht);CHKERRQ(ierr);}
  ierr = MatGetBlockSize(mat, &bs);CHKERRQ(ierr);
  ierr = MatXAIJSetPreallocation(A, bs, p->dnz, p->onz, p->dnzu, p->onzu);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);
  ierr = MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatSetOption(A, MAT_NO_OFF_PROC_ENTRIES, p->nooffproc);CHKERRQ(ierr);
  if (fill) {
    PetscHashIter  hi;
    PetscHashIJKey key;
    PetscScalar    *zeros;
    PetscInt       n,maxrow=1,*cols,rStart,rEnd,*rowstarts;

    ierr = MatGetOwnershipRange(A, &rStart, &rEnd);CHKERRQ(ierr);
    // Ownership range is in terms of scalar entries, but we deal with blocks
    rStart /= bs;
    rEnd /= bs;
    ierr = PetscHSetIJGetSize(p->ht,&n);CHKERRQ(ierr);
    ierr = PetscMalloc2(n,&cols,rEnd-rStart+1,&rowstarts);CHKERRQ(ierr);
    rowstarts[0] = 0;
    for (PetscInt i=0; i<rEnd-rStart; i++) {
      rowstarts[i+1] = rowstarts[i] + p->dnz[i] + p->onz[i];
      maxrow = PetscMax(maxrow, p->dnz[i] + p->onz[i]);
    }
    if (rowstarts[rEnd-rStart] != n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Hash claims %D entries, but dnz+onz counts %D",n,rowstarts[rEnd-rStart]);

    PetscHashIterBegin(p->ht,hi);
    for (PetscInt i=0; !PetscHashIterAtEnd(p->ht,hi); i++) {
      PetscHashIterGetKey(p->ht,hi,key);
      PetscInt lrow = key.i - rStart;
      cols[rowstarts[lrow]] = key.j;
      rowstarts[lrow]++;
      PetscHashIterNext(p->ht,hi);
    }
    ierr = PetscHSetIJDestroy(&p->ht);CHKERRQ(ierr);

    ierr = PetscCalloc1(maxrow*bs*bs,&zeros);CHKERRQ(ierr);
    for (PetscInt i=0; i<rEnd-rStart; i++) {
      PetscInt grow = rStart + i;
      PetscInt end = rowstarts[i], start = end - p->dnz[i] - p->onz[i];
      ierr = PetscSortInt(end-start,&cols[start]);CHKERRQ(ierr);
      ierr = MatSetValuesBlocked(A, 1, &grow, end-start, &cols[start], zeros, INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = PetscFree(zeros);CHKERRQ(ierr);
    ierr = PetscFree2(cols,rowstarts);CHKERRQ(ierr);

    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
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

  Level: advanced

.seealso: MATPREALLOCATOR
@*/
PetscErrorCode MatPreallocatorPreallocate(Mat mat, PetscBool fill, Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat, MAT_CLASSID, 1);
  PetscValidHeaderSpecific(A,   MAT_CLASSID, 3);
  ierr = PetscUseMethod(mat, "MatPreallocatorPreallocate_C", (Mat,PetscBool,Mat),(mat,fill,A));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
   MATPREALLOCATOR - MATPREALLOCATOR = "preallocator" - A matrix type to be used for computing a matrix preallocation.

   Operations Provided:
.  MatSetValues()

   Options Database Keys:
. -mat_type preallocator - sets the matrix type to "preallocator" during a call to MatSetFromOptions()

  Level: advanced

.seealso: Mat

M*/

PETSC_EXTERN PetscErrorCode MatCreate_Preallocator(Mat A)
{
  Mat_Preallocator *p;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(A, &p);CHKERRQ(ierr);
  A->data = (void *) p;

  p->ht   = NULL;
  p->dnz  = NULL;
  p->onz  = NULL;
  p->dnzu = NULL;
  p->onzu = NULL;

  /* matrix ops */
  ierr = PetscMemzero(A->ops, sizeof(struct _MatOps));CHKERRQ(ierr);

  A->ops->destroy       = MatDestroy_Preallocator;
  A->ops->setup         = MatSetUp_Preallocator;
  A->ops->setvalues     = MatSetValues_Preallocator;
  A->ops->assemblybegin = MatAssemblyBegin_Preallocator;
  A->ops->assemblyend   = MatAssemblyEnd_Preallocator;
  A->ops->view          = MatView_Preallocator;
  A->ops->setoption     = MatSetOption_Preallocator;
  A->ops->setblocksizes = MatSetBlockSizes_Default; /* once set, user is not allowed to change the block sizes */

  /* special MATPREALLOCATOR functions */
  ierr = PetscObjectComposeFunction((PetscObject) A, "MatPreallocatorPreallocate_C", MatPreallocatorPreallocate_Preallocator);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject) A, MATPREALLOCATOR);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
