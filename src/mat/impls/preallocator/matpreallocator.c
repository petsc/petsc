#include <petsc/private/matimpl.h>      /*I "petscmat.h" I*/
#include <petsc/private/hash.h>

typedef struct {
  PetscHashJK ht;
  PetscInt   *dnz, *onz;
} Mat_Preallocator;

PetscErrorCode MatDestroy_Preallocator(Mat A)
{
  Mat_Preallocator *p = (Mat_Preallocator *) A->data;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatStashDestroy_Private(&A->stash);CHKERRQ(ierr);
  ierr = PetscHashJKDestroy(&p->ht);CHKERRQ(ierr);
  ierr = PetscFree2(p->dnz, p->onz);CHKERRQ(ierr);
  ierr = PetscFree(A->data);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject) A, 0);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) A, "MatPreallocatorPreallocate_C", NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetUp_Preallocator(Mat A)
{
  Mat_Preallocator *p = (Mat_Preallocator *) A->data;
  PetscInt          m, bs;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscLayoutSetUp(A->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(A->cmap);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A, &m, NULL);CHKERRQ(ierr);
  ierr = PetscHashJKCreate(&p->ht);CHKERRQ(ierr);
  ierr = MatGetBlockSize(A, &bs);CHKERRQ(ierr);
  ierr = MatStashCreate_Private(PetscObjectComm((PetscObject) A), bs, &A->stash);CHKERRQ(ierr);
  ierr = PetscCalloc2(m, &p->dnz, m, &p->onz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetValues_Preallocator(Mat A, PetscInt m, const PetscInt *rows, PetscInt n, const PetscInt *cols, const PetscScalar *values, InsertMode addv)
{
  Mat_Preallocator *p = (Mat_Preallocator *) A->data;
  PetscInt          rStart, rEnd, r, cStart, cEnd, c;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  /* TODO: Handle blocksize */
  ierr = MatGetOwnershipRange(A, &rStart, &rEnd);CHKERRQ(ierr);
  ierr = MatGetOwnershipRangeColumn(A, &cStart, &cEnd);CHKERRQ(ierr);
  for (r = 0; r < m; ++r) {
    PetscHashJKKey  key;
    PetscHashJKIter missing, iter;

    key.j = rows[r];
    if (key.j < 0) continue;
    if ((key.j < rStart) || (key.j >= rEnd)) {
      ierr = MatStashValuesRow_Private(&A->stash, key.j, n, cols, values, PETSC_FALSE);CHKERRQ(ierr);
    } else {
      for (c = 0; c < n; ++c) {
        key.k = cols[c];
        if (key.k < 0) continue;
        ierr = PetscHashJKPut(p->ht, key, &missing, &iter);CHKERRQ(ierr);
        if (missing) {
          ierr = PetscHashJKSet(p->ht, iter, 1);CHKERRQ(ierr);
          if ((key.k >= cStart) && (key.k < cEnd)) ++p->dnz[key.j-rStart];
          else                                     ++p->onz[key.j-rStart];
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
  ierr = PetscLayoutSetUp(A->rmap);CHKERRQ(ierr);
  ierr = MatStashScatterBegin_Private(A, &A->stash, A->rmap->range);CHKERRQ(ierr);
  ierr = MatStashGetInfo_Private(&A->stash, &nstash, &reallocs);CHKERRQ(ierr);
  ierr = PetscInfo2(A, "Stash has %D entries, uses %D mallocs.\n", nstash, reallocs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatAssemblyEnd_Preallocator(Mat A, MatAssemblyType type)
{
  PetscScalar   *val;
  PetscInt      *row, *col;
  PetscInt       i, j, rstart, ncols, flg;
  PetscMPIInt    n;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  while (1) {
    ierr = MatStashScatterGetMesg_Private(&A->stash, &n, &row, &col, &val, &flg);CHKERRQ(ierr);
    if (!flg) break;

    for (i = 0; i < n; ) {
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
  PetscInt         *udnz = NULL, *uonz = NULL;
  PetscInt          bs;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MatGetBlockSize(mat, &bs);CHKERRQ(ierr);
  ierr = MatXAIJSetPreallocation(A, bs, p->dnz, p->onz, udnz, uonz);CHKERRQ(ierr);
  ierr = MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  MatPreallocatorPreallocate - Preallocates the input matrix, optionally filling it with zeros

  Input Parameter:
+ mat  - the preallocator
- fill - fill the matrix with zeros

  Output Parameter:
. A    - the matrix

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

  p->ht  = NULL;
  p->dnz = NULL;
  p->onz = NULL;

  /* matrix ops */
  ierr = PetscMemzero(A->ops, sizeof(struct _MatOps));CHKERRQ(ierr);
  A->ops->destroy                 = MatDestroy_Preallocator;
  A->ops->setup                   = MatSetUp_Preallocator;
  A->ops->setvalues               = MatSetValues_Preallocator;
  A->ops->assemblybegin           = MatAssemblyBegin_Preallocator;
  A->ops->assemblyend             = MatAssemblyEnd_Preallocator;
  A->ops->view                    = MatView_Preallocator;
  A->ops->setoption               = MatSetOption_Preallocator;

  /* special MATPREALLOCATOR functions */
  ierr = PetscObjectComposeFunction((PetscObject) A, "MatPreallocatorPreallocate_C", MatPreallocatorPreallocate_Preallocator);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject) A, MATPREALLOCATOR);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
