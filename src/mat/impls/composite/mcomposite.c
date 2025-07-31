#include <../src/mat/impls/shell/shell.h> /*I "petscmat.h" I*/

const char *const MatCompositeMergeTypes[] = {"left", "right", "MatCompositeMergeType", "MAT_COMPOSITE_", NULL};

typedef struct _Mat_CompositeLink *Mat_CompositeLink;
struct _Mat_CompositeLink {
  Mat               mat;
  Vec               work;
  Mat_CompositeLink next, prev;
};

typedef struct {
  MatCompositeType      type;
  Mat_CompositeLink     head, tail;
  Vec                   work;
  PetscInt              nmat;
  PetscBool             merge;
  MatCompositeMergeType mergetype;
  MatStructure          structure;

  PetscScalar *scalings;
  PetscBool    merge_mvctx; /* Whether need to merge mvctx of component matrices */
  Vec         *lvecs;       /* [nmat] Basically, they are Mvctx->lvec of each component matrix */
  PetscScalar *larray;      /* [len] Data arrays of lvecs[] are stored consecutively in larray */
  PetscInt     len;         /* Length of larray[] */
  Vec          gvec;        /* Union of lvecs[] without duplicated entries */
  PetscInt    *location;    /* A map that maps entries in garray[] to larray[] */
  VecScatter   Mvctx;
} Mat_Composite;

static PetscErrorCode MatDestroy_Composite(Mat mat)
{
  Mat_Composite    *shell;
  Mat_CompositeLink next, oldnext;
  PetscInt          i;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(mat, &shell));
  next = shell->head;
  while (next) {
    PetscCall(MatDestroy(&next->mat));
    if (next->work && (!next->next || next->work != next->next->work)) PetscCall(VecDestroy(&next->work));
    oldnext = next;
    next    = next->next;
    PetscCall(PetscFree(oldnext));
  }
  PetscCall(VecDestroy(&shell->work));

  if (shell->Mvctx) {
    for (i = 0; i < shell->nmat; i++) PetscCall(VecDestroy(&shell->lvecs[i]));
    PetscCall(PetscFree3(shell->location, shell->larray, shell->lvecs));
    PetscCall(PetscFree(shell->larray));
    PetscCall(VecDestroy(&shell->gvec));
    PetscCall(VecScatterDestroy(&shell->Mvctx));
  }

  PetscCall(PetscFree(shell->scalings));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "MatCompositeAddMat_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "MatCompositeSetType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "MatCompositeGetType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "MatCompositeSetMergeType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "MatCompositeSetMatStructure_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "MatCompositeGetMatStructure_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "MatCompositeMerge_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "MatCompositeGetNumberMat_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "MatCompositeGetMat_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "MatCompositeSetScalings_C", NULL));
  PetscCall(PetscFree(shell));
  PetscCall(PetscObjectComposeFunction((PetscObject)mat, "MatShellSetContext_C", NULL)); // needed to avoid a call to MatShellSetContext_Immutable()
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMult_Composite_Multiplicative(Mat A, Vec x, Vec y)
{
  Mat_Composite    *shell;
  Mat_CompositeLink next;
  Vec               out;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A, &shell));
  next = shell->head;
  PetscCheck(next, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Must provide at least one matrix with MatCompositeAddMat()");
  while (next->next) {
    if (!next->work) { /* should reuse previous work if the same size */
      PetscCall(MatCreateVecs(next->mat, NULL, &next->work));
    }
    out = next->work;
    PetscCall(MatMult(next->mat, x, out));
    x    = out;
    next = next->next;
  }
  PetscCall(MatMult(next->mat, x, y));
  if (shell->scalings) {
    PetscScalar scale = 1.0;
    for (PetscInt i = 0; i < shell->nmat; i++) scale *= shell->scalings[i];
    PetscCall(VecScale(y, scale));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMultTranspose_Composite_Multiplicative(Mat A, Vec x, Vec y)
{
  Mat_Composite    *shell;
  Mat_CompositeLink tail;
  Vec               out;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A, &shell));
  tail = shell->tail;
  PetscCheck(tail, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Must provide at least one matrix with MatCompositeAddMat()");
  while (tail->prev) {
    if (!tail->prev->work) { /* should reuse previous work if the same size */
      PetscCall(MatCreateVecs(tail->mat, NULL, &tail->prev->work));
    }
    out = tail->prev->work;
    PetscCall(MatMultTranspose(tail->mat, x, out));
    x    = out;
    tail = tail->prev;
  }
  PetscCall(MatMultTranspose(tail->mat, x, y));
  if (shell->scalings) {
    PetscScalar scale = 1.0;
    for (PetscInt i = 0; i < shell->nmat; i++) scale *= shell->scalings[i];
    PetscCall(VecScale(y, scale));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMult_Composite(Mat mat, Vec x, Vec y)
{
  Mat_Composite     *shell;
  Mat_CompositeLink  cur;
  Vec                y2, xin;
  Mat                A, B;
  PetscInt           i, j, k, n, nuniq, lo, hi, mid, *gindices, *buf, *tmp, tot;
  const PetscScalar *vals;
  const PetscInt    *garray;
  IS                 ix, iy;
  PetscBool          match;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(mat, &shell));
  cur = shell->head;
  PetscCheck(cur, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Must provide at least one matrix with MatCompositeAddMat()");

  /* Try to merge Mvctx when instructed but not yet done. We did not do it in MatAssemblyEnd() since at that time
     we did not know whether mat is ADDITIVE or MULTIPLICATIVE. Only now we are assured mat is ADDITIVE and
     it is legal to merge Mvctx, because all component matrices have the same size.
   */
  if (shell->merge_mvctx && !shell->Mvctx) {
    /* Currently only implemented for MATMPIAIJ */
    for (cur = shell->head; cur; cur = cur->next) {
      PetscCall(PetscObjectTypeCompare((PetscObject)cur->mat, MATMPIAIJ, &match));
      if (!match) {
        shell->merge_mvctx = PETSC_FALSE;
        goto skip_merge_mvctx;
      }
    }

    /* Go through matrices first time to count total number of nonzero off-diag columns (may have dups) */
    tot = 0;
    for (cur = shell->head; cur; cur = cur->next) {
      PetscCall(MatMPIAIJGetSeqAIJ(cur->mat, NULL, &B, NULL));
      PetscCall(MatGetLocalSize(B, NULL, &n));
      tot += n;
    }
    PetscCall(PetscMalloc3(tot, &shell->location, tot, &shell->larray, shell->nmat, &shell->lvecs));
    shell->len = tot;

    /* Go through matrices second time to sort off-diag columns and remove dups */
    PetscCall(PetscMalloc1(tot, &gindices)); /* No Malloc2() since we will give one to PETSc and free the other */
    PetscCall(PetscMalloc1(tot, &buf));
    nuniq = 0; /* Number of unique nonzero columns */
    for (cur = shell->head; cur; cur = cur->next) {
      PetscCall(MatMPIAIJGetSeqAIJ(cur->mat, NULL, &B, &garray));
      PetscCall(MatGetLocalSize(B, NULL, &n));
      /* Merge pre-sorted garray[0,n) and gindices[0,nuniq) to buf[] */
      i = j = k = 0;
      while (i < n && j < nuniq) {
        if (garray[i] < gindices[j]) buf[k++] = garray[i++];
        else if (garray[i] > gindices[j]) buf[k++] = gindices[j++];
        else {
          buf[k++] = garray[i++];
          j++;
        }
      }
      /* Copy leftover in garray[] or gindices[] */
      if (i < n) {
        PetscCall(PetscArraycpy(buf + k, garray + i, n - i));
        nuniq = k + n - i;
      } else if (j < nuniq) {
        PetscCall(PetscArraycpy(buf + k, gindices + j, nuniq - j));
        nuniq = k + nuniq - j;
      } else nuniq = k;
      /* Swap gindices and buf to merge garray of the next matrix */
      tmp      = gindices;
      gindices = buf;
      buf      = tmp;
    }
    PetscCall(PetscFree(buf));

    /* Go through matrices third time to build a map from gindices[] to garray[] */
    tot = 0;
    for (cur = shell->head, j = 0; cur; cur = cur->next, j++) { /* j-th matrix */
      PetscCall(MatMPIAIJGetSeqAIJ(cur->mat, NULL, &B, &garray));
      PetscCall(MatGetLocalSize(B, NULL, &n));
      PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF, 1, n, NULL, &shell->lvecs[j]));
      /* This is an optimized PetscFindInt(garray[i],nuniq,gindices,&shell->location[tot+i]), using the fact that garray[] is also sorted */
      lo = 0;
      for (i = 0; i < n; i++) {
        hi = nuniq;
        while (hi - lo > 1) {
          mid = lo + (hi - lo) / 2;
          if (garray[i] < gindices[mid]) hi = mid;
          else lo = mid;
        }
        shell->location[tot + i] = lo; /* gindices[lo] = garray[i] */
        lo++;                          /* Since garray[i+1] > garray[i], we can safely advance lo */
      }
      tot += n;
    }

    /* Build merged Mvctx */
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, nuniq, gindices, PETSC_OWN_POINTER, &ix));
    PetscCall(ISCreateStride(PETSC_COMM_SELF, nuniq, 0, 1, &iy));
    PetscCall(VecCreateMPIWithArray(PetscObjectComm((PetscObject)mat), 1, mat->cmap->n, mat->cmap->N, NULL, &xin));
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, nuniq, &shell->gvec));
    PetscCall(VecScatterCreate(xin, ix, shell->gvec, iy, &shell->Mvctx));
    PetscCall(VecDestroy(&xin));
    PetscCall(ISDestroy(&ix));
    PetscCall(ISDestroy(&iy));
  }

skip_merge_mvctx:
  PetscCall(VecSet(y, 0));
  if (!((Mat_Shell *)mat->data)->left_work) PetscCall(VecDuplicate(y, &(((Mat_Shell *)mat->data)->left_work)));
  y2 = ((Mat_Shell *)mat->data)->left_work;

  if (shell->Mvctx) { /* Have a merged Mvctx */
    /* Suppose we want to compute y = sMx, where s is the scaling factor and A, B are matrix M's diagonal/off-diagonal part. We could do
       in y = s(Ax1 + Bx2) or y = sAx1 + sBx2. The former incurs less FLOPS than the latter, but the latter provides an opportunity to
       overlap communication/computation since we can do sAx1 while communicating x2. Here, we use the former approach.
     */
    PetscCall(VecScatterBegin(shell->Mvctx, x, shell->gvec, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(shell->Mvctx, x, shell->gvec, INSERT_VALUES, SCATTER_FORWARD));

    PetscCall(VecGetArrayRead(shell->gvec, &vals));
    for (i = 0; i < shell->len; i++) shell->larray[i] = vals[shell->location[i]];
    PetscCall(VecRestoreArrayRead(shell->gvec, &vals));

    for (cur = shell->head, tot = i = 0; cur; cur = cur->next, i++) { /* i-th matrix */
      PetscCall(MatMPIAIJGetSeqAIJ(cur->mat, &A, &B, NULL));
      PetscUseTypeMethod(A, mult, x, y2);
      PetscCall(MatGetLocalSize(B, NULL, &n));
      PetscCall(VecPlaceArray(shell->lvecs[i], &shell->larray[tot]));
      PetscUseTypeMethod(B, multadd, shell->lvecs[i], y2, y2);
      PetscCall(VecResetArray(shell->lvecs[i]));
      PetscCall(VecAXPY(y, shell->scalings ? shell->scalings[i] : 1.0, y2));
      tot += n;
    }
  } else {
    if (shell->scalings) {
      for (cur = shell->head, i = 0; cur; cur = cur->next, i++) {
        PetscCall(MatMult(cur->mat, x, y2));
        PetscCall(VecAXPY(y, shell->scalings[i], y2));
      }
    } else {
      for (cur = shell->head; cur; cur = cur->next) PetscCall(MatMultAdd(cur->mat, x, y, y));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMultTranspose_Composite(Mat A, Vec x, Vec y)
{
  Mat_Composite    *shell;
  Mat_CompositeLink next;
  Vec               y2 = NULL;
  PetscInt          i;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A, &shell));
  next = shell->head;
  PetscCheck(next, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Must provide at least one matrix with MatCompositeAddMat()");

  PetscCall(MatMultTranspose(next->mat, x, y));
  if (shell->scalings) {
    PetscCall(VecScale(y, shell->scalings[0]));
    if (!((Mat_Shell *)A->data)->right_work) PetscCall(VecDuplicate(y, &(((Mat_Shell *)A->data)->right_work)));
    y2 = ((Mat_Shell *)A->data)->right_work;
  }
  i = 1;
  while ((next = next->next)) {
    if (!shell->scalings) PetscCall(MatMultTransposeAdd(next->mat, x, y, y));
    else {
      PetscCall(MatMultTranspose(next->mat, x, y2));
      PetscCall(VecAXPY(y, shell->scalings[i++], y2));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatGetDiagonal_Composite(Mat A, Vec v)
{
  Mat_Composite    *shell;
  Mat_CompositeLink next;
  PetscInt          i;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A, &shell));
  next = shell->head;
  PetscCheck(next, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Must provide at least one matrix with MatCompositeAddMat()");
  PetscCall(MatGetDiagonal(next->mat, v));
  if (shell->scalings) PetscCall(VecScale(v, shell->scalings[0]));

  if (next->next && !shell->work) PetscCall(VecDuplicate(v, &shell->work));
  i = 1;
  while ((next = next->next)) {
    PetscCall(MatGetDiagonal(next->mat, shell->work));
    PetscCall(VecAXPY(v, shell->scalings ? shell->scalings[i++] : 1.0, shell->work));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatAssemblyEnd_Composite(Mat Y, MatAssemblyType t)
{
  Mat_Composite *shell;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(Y, &shell));
  if (shell->merge) PetscCall(MatCompositeMerge(Y));
  else PetscCall(MatAssemblyEnd_Shell(Y, t));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetFromOptions_Composite(Mat A, PetscOptionItems PetscOptionsObject)
{
  Mat_Composite *a;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A, &a));
  PetscOptionsHeadBegin(PetscOptionsObject, "MATCOMPOSITE options");
  PetscCall(PetscOptionsBool("-mat_composite_merge", "Merge at MatAssemblyEnd", "MatCompositeMerge", a->merge, &a->merge, NULL));
  PetscCall(PetscOptionsEnum("-mat_composite_merge_type", "Set composite merge direction", "MatCompositeSetMergeType", MatCompositeMergeTypes, (PetscEnum)a->mergetype, (PetscEnum *)&a->mergetype, NULL));
  PetscCall(PetscOptionsBool("-mat_composite_merge_mvctx", "Merge MatMult() vecscat contexts", "MatCreateComposite", a->merge_mvctx, &a->merge_mvctx, NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatCreateComposite - Creates a matrix as the sum or product of one or more matrices

  Collective

  Input Parameters:
+ comm - MPI communicator
. nmat - number of matrices to put in
- mats - the matrices

  Output Parameter:
. mat - the matrix

  Options Database Keys:
+ -mat_composite_merge       - merge in `MatAssemblyEnd()`
. -mat_composite_merge_mvctx - merge Mvctx of component matrices to optimize communication in `MatMult()` for ADDITIVE matrices
- -mat_composite_merge_type  - set merge direction

  Level: advanced

  Note:
  Alternative construction
.vb
       MatCreate(comm,&mat);
       MatSetSizes(mat,m,n,M,N);
       MatSetType(mat,MATCOMPOSITE);
       MatCompositeAddMat(mat,mats[0]);
       ....
       MatCompositeAddMat(mat,mats[nmat-1]);
       MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);
       MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);
.ve

  For the multiplicative form the product is mat[nmat-1]*mat[nmat-2]*....*mat[0]

.seealso: [](ch_matrices), `Mat`, `MatDestroy()`, `MatMult()`, `MatCompositeAddMat()`, `MatCompositeGetMat()`, `MatCompositeMerge()`, `MatCompositeSetType()`,
          `MATCOMPOSITE`, `MatCompositeType`
@*/
PetscErrorCode MatCreateComposite(MPI_Comm comm, PetscInt nmat, const Mat *mats, Mat *mat)
{
  PetscFunctionBegin;
  PetscCheck(nmat >= 1, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Must pass in at least one matrix");
  PetscAssertPointer(mat, 4);
  PetscCall(MatCreate(comm, mat));
  PetscCall(MatSetType(*mat, MATCOMPOSITE));
  for (PetscInt i = 0; i < nmat; i++) PetscCall(MatCompositeAddMat(*mat, mats[i]));
  PetscCall(MatAssemblyBegin(*mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*mat, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCompositeAddMat_Composite(Mat mat, Mat smat)
{
  Mat_Composite    *shell;
  Mat_CompositeLink ilink, next;
  VecType           vtype_mat, vtype_smat;
  PetscBool         match;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(mat, &shell));
  next = shell->head;
  PetscCall(PetscNew(&ilink));
  ilink->next = NULL;
  PetscCall(PetscObjectReference((PetscObject)smat));
  ilink->mat = smat;

  if (!next) shell->head = ilink;
  else {
    while (next->next) next = next->next;
    next->next  = ilink;
    ilink->prev = next;
  }
  shell->tail = ilink;
  shell->nmat += 1;

  /* If all of the partial matrices have the same default vector type, then the composite matrix should also have this default type.
     Otherwise, the default type should be "standard". */
  PetscCall(MatGetVecType(smat, &vtype_smat));
  if (shell->nmat == 1) PetscCall(MatSetVecType(mat, vtype_smat));
  else {
    PetscCall(MatGetVecType(mat, &vtype_mat));
    PetscCall(PetscStrcmp(vtype_smat, vtype_mat, &match));
    if (!match) PetscCall(MatSetVecType(mat, VECSTANDARD));
  }

  /* Retain the old scalings (if any) and expand it with a 1.0 for the newly added matrix */
  if (shell->scalings) {
    PetscCall(PetscRealloc(sizeof(PetscScalar) * shell->nmat, &shell->scalings));
    shell->scalings[shell->nmat - 1] = 1.0;
  }

  /* The composite matrix requires PetscLayouts for its rows and columns; we copy these from the constituent partial matrices. */
  if (shell->nmat == 1) PetscCall(PetscLayoutReference(smat->cmap, &mat->cmap));
  PetscCall(PetscLayoutReference(smat->rmap, &mat->rmap));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatCompositeAddMat - Add another matrix to a composite matrix.

  Collective

  Input Parameters:
+ mat  - the composite matrix
- smat - the partial matrix

  Level: advanced

.seealso: [](ch_matrices), `Mat`, `MatCreateComposite()`, `MatCompositeGetMat()`, `MATCOMPOSITE`
@*/
PetscErrorCode MatCompositeAddMat(Mat mat, Mat smat)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat, MAT_CLASSID, 1);
  PetscValidHeaderSpecific(smat, MAT_CLASSID, 2);
  PetscUseMethod(mat, "MatCompositeAddMat_C", (Mat, Mat), (mat, smat));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCompositeSetType_Composite(Mat mat, MatCompositeType type)
{
  Mat_Composite *b;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(mat, &b));
  b->type = type;
  if (type == MAT_COMPOSITE_MULTIPLICATIVE) {
    PetscCall(MatShellSetOperation(mat, MATOP_GET_DIAGONAL, NULL));
    PetscCall(MatShellSetOperation(mat, MATOP_MULT, (PetscErrorCodeFn *)MatMult_Composite_Multiplicative));
    PetscCall(MatShellSetOperation(mat, MATOP_MULT_TRANSPOSE, (PetscErrorCodeFn *)MatMultTranspose_Composite_Multiplicative));
    b->merge_mvctx = PETSC_FALSE;
  } else {
    PetscCall(MatShellSetOperation(mat, MATOP_GET_DIAGONAL, (PetscErrorCodeFn *)MatGetDiagonal_Composite));
    PetscCall(MatShellSetOperation(mat, MATOP_MULT, (PetscErrorCodeFn *)MatMult_Composite));
    PetscCall(MatShellSetOperation(mat, MATOP_MULT_TRANSPOSE, (PetscErrorCodeFn *)MatMultTranspose_Composite));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatCompositeSetType - Indicates if the matrix is defined as the sum of a set of matrices or the product.

  Logically Collective

  Input Parameters:
+ mat  - the composite matrix
- type - the `MatCompositeType` to use for the matrix

  Level: advanced

.seealso: [](ch_matrices), `Mat`, `MatDestroy()`, `MatMult()`, `MatCompositeAddMat()`, `MatCreateComposite()`, `MatCompositeGetType()`, `MATCOMPOSITE`,
          `MatCompositeType`
@*/
PetscErrorCode MatCompositeSetType(Mat mat, MatCompositeType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat, MAT_CLASSID, 1);
  PetscValidLogicalCollectiveEnum(mat, type, 2);
  PetscUseMethod(mat, "MatCompositeSetType_C", (Mat, MatCompositeType), (mat, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCompositeGetType_Composite(Mat mat, MatCompositeType *type)
{
  Mat_Composite *shell;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(mat, &shell));
  *type = shell->type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatCompositeGetType - Returns type of composite.

  Not Collective

  Input Parameter:
. mat - the composite matrix

  Output Parameter:
. type - type of composite

  Level: advanced

.seealso: [](ch_matrices), `Mat`, `MatCreateComposite()`, `MatCompositeSetType()`, `MATCOMPOSITE`, `MatCompositeType`
@*/
PetscErrorCode MatCompositeGetType(Mat mat, MatCompositeType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat, MAT_CLASSID, 1);
  PetscAssertPointer(type, 2);
  PetscUseMethod(mat, "MatCompositeGetType_C", (Mat, MatCompositeType *), (mat, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCompositeSetMatStructure_Composite(Mat mat, MatStructure str)
{
  Mat_Composite *shell;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(mat, &shell));
  shell->structure = str;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatCompositeSetMatStructure - Indicates structure of matrices in the composite matrix.

  Not Collective

  Input Parameters:
+ mat - the composite matrix
- str - either `SAME_NONZERO_PATTERN`, `DIFFERENT_NONZERO_PATTERN` (default) or `SUBSET_NONZERO_PATTERN`

  Level: advanced

  Note:
  Information about the matrices structure is used in `MatCompositeMerge()` for additive composite matrix.

.seealso: [](ch_matrices), `Mat`, `MatAXPY()`, `MatCreateComposite()`, `MatCompositeMerge()` `MatCompositeGetMatStructure()`, `MATCOMPOSITE`
@*/
PetscErrorCode MatCompositeSetMatStructure(Mat mat, MatStructure str)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat, MAT_CLASSID, 1);
  PetscUseMethod(mat, "MatCompositeSetMatStructure_C", (Mat, MatStructure), (mat, str));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCompositeGetMatStructure_Composite(Mat mat, MatStructure *str)
{
  Mat_Composite *shell;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(mat, &shell));
  *str = shell->structure;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatCompositeGetMatStructure - Returns the structure of matrices in the composite matrix.

  Not Collective

  Input Parameter:
. mat - the composite matrix

  Output Parameter:
. str - structure of the matrices

  Level: advanced

.seealso: [](ch_matrices), `Mat`, `MatCreateComposite()`, `MatCompositeSetMatStructure()`, `MATCOMPOSITE`
@*/
PetscErrorCode MatCompositeGetMatStructure(Mat mat, MatStructure *str)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat, MAT_CLASSID, 1);
  PetscAssertPointer(str, 2);
  PetscUseMethod(mat, "MatCompositeGetMatStructure_C", (Mat, MatStructure *), (mat, str));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCompositeSetMergeType_Composite(Mat mat, MatCompositeMergeType type)
{
  Mat_Composite *shell;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(mat, &shell));
  shell->mergetype = type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatCompositeSetMergeType - Sets order of `MatCompositeMerge()`.

  Logically Collective

  Input Parameters:
+ mat  - the composite matrix
- type - `MAT_COMPOSITE_MERGE RIGHT` (default) to start merge from right with the first added matrix (mat[0]),
          `MAT_COMPOSITE_MERGE_LEFT` to start merge from left with the last added matrix (mat[nmat-1])

  Level: advanced

  Note:
  The resulting matrix is the same regardless of the `MatCompositeMergeType`. Only the order of operation is changed.
  If set to `MAT_COMPOSITE_MERGE_RIGHT` the order of the merge is mat[nmat-1]*(mat[nmat-2]*(...*(mat[1]*mat[0])))
  otherwise the order is (((mat[nmat-1]*mat[nmat-2])*mat[nmat-3])*...)*mat[0].

.seealso: [](ch_matrices), `Mat`, `MatCreateComposite()`, `MatCompositeMerge()`, `MATCOMPOSITE`
@*/
PetscErrorCode MatCompositeSetMergeType(Mat mat, MatCompositeMergeType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat, MAT_CLASSID, 1);
  PetscValidLogicalCollectiveEnum(mat, type, 2);
  PetscUseMethod(mat, "MatCompositeSetMergeType_C", (Mat, MatCompositeMergeType), (mat, type));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCompositeMerge_Composite(Mat mat)
{
  Mat_Composite    *shell;
  Mat_CompositeLink next, prev;
  Mat               tmat, newmat;
  Vec               left, right, dshift;
  PetscScalar       scale, shift;
  PetscInt          i;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(mat, &shell));
  next = shell->head;
  prev = shell->tail;
  PetscCheck(next, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Must provide at least one matrix with MatCompositeAddMat()");
  PetscCall(MatShellGetScalingShifts(mat, &shift, &scale, &dshift, &left, &right, (Mat *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED));
  if (shell->type == MAT_COMPOSITE_ADDITIVE) {
    if (shell->mergetype == MAT_COMPOSITE_MERGE_RIGHT) {
      i = 0;
      PetscCall(MatDuplicate(next->mat, MAT_COPY_VALUES, &tmat));
      if (shell->scalings) PetscCall(MatScale(tmat, shell->scalings[i++]));
      while ((next = next->next)) PetscCall(MatAXPY(tmat, shell->scalings ? shell->scalings[i++] : 1.0, next->mat, shell->structure));
    } else {
      i = shell->nmat - 1;
      PetscCall(MatDuplicate(prev->mat, MAT_COPY_VALUES, &tmat));
      if (shell->scalings) PetscCall(MatScale(tmat, shell->scalings[i--]));
      while ((prev = prev->prev)) PetscCall(MatAXPY(tmat, shell->scalings ? shell->scalings[i--] : 1.0, prev->mat, shell->structure));
    }
  } else {
    if (shell->mergetype == MAT_COMPOSITE_MERGE_RIGHT) {
      PetscCall(MatDuplicate(next->mat, MAT_COPY_VALUES, &tmat));
      while ((next = next->next)) {
        PetscCall(MatMatMult(next->mat, tmat, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &newmat));
        PetscCall(MatDestroy(&tmat));
        tmat = newmat;
      }
    } else {
      PetscCall(MatDuplicate(prev->mat, MAT_COPY_VALUES, &tmat));
      while ((prev = prev->prev)) {
        PetscCall(MatMatMult(tmat, prev->mat, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &newmat));
        PetscCall(MatDestroy(&tmat));
        tmat = newmat;
      }
    }
    if (shell->scalings) {
      for (i = 0; i < shell->nmat; i++) scale *= shell->scalings[i];
    }
  }

  if (left) PetscCall(PetscObjectReference((PetscObject)left));
  if (right) PetscCall(PetscObjectReference((PetscObject)right));
  if (dshift) PetscCall(PetscObjectReference((PetscObject)dshift));

  PetscCall(MatHeaderReplace(mat, &tmat));

  PetscCall(MatDiagonalScale(mat, left, right));
  PetscCall(MatScale(mat, scale));
  PetscCall(MatShift(mat, shift));
  PetscCall(VecDestroy(&left));
  PetscCall(VecDestroy(&right));
  if (dshift) {
    PetscCall(MatDiagonalSet(mat, dshift, ADD_VALUES));
    PetscCall(VecDestroy(&dshift));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatCompositeMerge - Given a composite matrix, replaces it with a "regular" matrix
  by summing or computing the product of all the matrices inside the composite matrix.

  Collective

  Input Parameter:
. mat - the composite matrix

  Options Database Keys:
+ -mat_composite_merge      - merge in `MatAssemblyEnd()`
- -mat_composite_merge_type - set merge direction

  Level: advanced

  Note:
  The `MatType` of the resulting matrix will be the same as the `MatType` of the FIRST matrix in the composite matrix.

.seealso: [](ch_matrices), `Mat`, `MatDestroy()`, `MatMult()`, `MatCompositeAddMat()`, `MatCreateComposite()`, `MatCompositeSetMatStructure()`, `MatCompositeSetMergeType()`, `MATCOMPOSITE`
@*/
PetscErrorCode MatCompositeMerge(Mat mat)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat, MAT_CLASSID, 1);
  PetscUseMethod(mat, "MatCompositeMerge_C", (Mat), (mat));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCompositeGetNumberMat_Composite(Mat mat, PetscInt *nmat)
{
  Mat_Composite *shell;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(mat, &shell));
  *nmat = shell->nmat;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatCompositeGetNumberMat - Returns the number of matrices in the composite matrix.

  Not Collective

  Input Parameter:
. mat - the composite matrix

  Output Parameter:
. nmat - number of matrices in the composite matrix

  Level: advanced

.seealso: [](ch_matrices), `Mat`, `MatCreateComposite()`, `MatCompositeGetMat()`, `MATCOMPOSITE`
@*/
PetscErrorCode MatCompositeGetNumberMat(Mat mat, PetscInt *nmat)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat, MAT_CLASSID, 1);
  PetscAssertPointer(nmat, 2);
  PetscUseMethod(mat, "MatCompositeGetNumberMat_C", (Mat, PetscInt *), (mat, nmat));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCompositeGetMat_Composite(Mat mat, PetscInt i, Mat *Ai)
{
  Mat_Composite    *shell;
  Mat_CompositeLink ilink;
  PetscInt          k;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(mat, &shell));
  PetscCheck(i < shell->nmat, PetscObjectComm((PetscObject)mat), PETSC_ERR_ARG_OUTOFRANGE, "index out of range: %" PetscInt_FMT " >= %" PetscInt_FMT, i, shell->nmat);
  ilink = shell->head;
  for (k = 0; k < i; k++) ilink = ilink->next;
  *Ai = ilink->mat;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatCompositeGetMat - Returns the ith matrix from the composite matrix.

  Logically Collective

  Input Parameters:
+ mat - the composite matrix
- i   - the number of requested matrix

  Output Parameter:
. Ai - ith matrix in composite

  Level: advanced

.seealso: [](ch_matrices), `Mat`, `MatCreateComposite()`, `MatCompositeGetNumberMat()`, `MatCompositeAddMat()`, `MATCOMPOSITE`
@*/
PetscErrorCode MatCompositeGetMat(Mat mat, PetscInt i, Mat *Ai)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat, MAT_CLASSID, 1);
  PetscValidLogicalCollectiveInt(mat, i, 2);
  PetscAssertPointer(Ai, 3);
  PetscUseMethod(mat, "MatCompositeGetMat_C", (Mat, PetscInt, Mat *), (mat, i, Ai));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCompositeSetScalings_Composite(Mat mat, const PetscScalar *scalings)
{
  Mat_Composite *shell;
  PetscInt       nmat;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(mat, &shell));
  PetscCall(MatCompositeGetNumberMat(mat, &nmat));
  if (!shell->scalings) PetscCall(PetscMalloc1(nmat, &shell->scalings));
  PetscCall(PetscArraycpy(shell->scalings, scalings, nmat));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatCompositeSetScalings - Sets separate scaling factors for component matrices.

  Logically Collective

  Input Parameters:
+ mat      - the composite matrix
- scalings - array of scaling factors with scalings[i] being factor of i-th matrix, for i in [0, nmat)

  Level: advanced

.seealso: [](ch_matrices), `Mat`, `MatScale()`, `MatDiagonalScale()`, `MATCOMPOSITE`
@*/
PetscErrorCode MatCompositeSetScalings(Mat mat, const PetscScalar *scalings)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat, MAT_CLASSID, 1);
  PetscAssertPointer(scalings, 2);
  PetscValidLogicalCollectiveScalar(mat, *scalings, 2);
  PetscUseMethod(mat, "MatCompositeSetScalings_C", (Mat, const PetscScalar *), (mat, scalings));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   MATCOMPOSITE - A matrix defined by the sum (or product) of one or more matrices.
    The matrices need to have a correct size and parallel layout for the sum or product to be valid.

  Level: advanced

   Note:
   To use the product of the matrices call `MatCompositeSetType`(mat,`MAT_COMPOSITE_MULTIPLICATIVE`);

  Developer Notes:
  This is implemented on top of `MATSHELL` to get support for scaling and shifting without requiring duplicate code

  Users can not call `MatShellSetOperation()` operations on this class, there is some error checking for that incorrect usage

.seealso: [](ch_matrices), `Mat`, `MatCreateComposite()`, `MatCompositeSetScalings()`, `MatCompositeAddMat()`, `MatSetType()`, `MatCompositeSetType()`, `MatCompositeGetType()`,
          `MatCompositeSetMatStructure()`, `MatCompositeGetMatStructure()`, `MatCompositeMerge()`, `MatCompositeSetMergeType()`, `MatCompositeGetNumberMat()`, `MatCompositeGetMat()`
M*/

PETSC_EXTERN PetscErrorCode MatCreate_Composite(Mat A)
{
  Mat_Composite *b;

  PetscFunctionBegin;
  PetscCall(PetscNew(&b));

  b->type        = MAT_COMPOSITE_ADDITIVE;
  b->nmat        = 0;
  b->merge       = PETSC_FALSE;
  b->mergetype   = MAT_COMPOSITE_MERGE_RIGHT;
  b->structure   = DIFFERENT_NONZERO_PATTERN;
  b->merge_mvctx = PETSC_TRUE;

  PetscCall(MatSetType(A, MATSHELL));
  PetscCall(MatShellSetContext(A, b));
  PetscCall(MatShellSetOperation(A, MATOP_DESTROY, (PetscErrorCodeFn *)MatDestroy_Composite));
  PetscCall(MatShellSetOperation(A, MATOP_MULT, (PetscErrorCodeFn *)MatMult_Composite));
  PetscCall(MatShellSetOperation(A, MATOP_MULT_TRANSPOSE, (PetscErrorCodeFn *)MatMultTranspose_Composite));
  PetscCall(MatShellSetOperation(A, MATOP_GET_DIAGONAL, (PetscErrorCodeFn *)MatGetDiagonal_Composite));
  PetscCall(MatShellSetOperation(A, MATOP_ASSEMBLY_END, (PetscErrorCodeFn *)MatAssemblyEnd_Composite));
  PetscCall(MatShellSetOperation(A, MATOP_SET_FROM_OPTIONS, (PetscErrorCodeFn *)MatSetFromOptions_Composite));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatCompositeAddMat_C", MatCompositeAddMat_Composite));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatCompositeSetType_C", MatCompositeSetType_Composite));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatCompositeGetType_C", MatCompositeGetType_Composite));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatCompositeSetMergeType_C", MatCompositeSetMergeType_Composite));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatCompositeSetMatStructure_C", MatCompositeSetMatStructure_Composite));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatCompositeGetMatStructure_C", MatCompositeGetMatStructure_Composite));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatCompositeMerge_C", MatCompositeMerge_Composite));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatCompositeGetNumberMat_C", MatCompositeGetNumberMat_Composite));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatCompositeGetMat_C", MatCompositeGetMat_Composite));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatCompositeSetScalings_C", MatCompositeSetScalings_Composite));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatShellSetContext_C", MatShellSetContext_Immutable));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatShellSetContextDestroy_C", MatShellSetContextDestroy_Immutable));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatShellSetManageScalingShifts_C", MatShellSetManageScalingShifts_Immutable));
  PetscCall(PetscObjectChangeTypeName((PetscObject)A, MATCOMPOSITE));
  PetscFunctionReturn(PETSC_SUCCESS);
}
