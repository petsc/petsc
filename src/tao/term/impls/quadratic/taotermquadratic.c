#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/

typedef struct _n_TaoTerm_Quadratic TaoTerm_Quadratic;

struct _n_TaoTerm_Quadratic {
  Mat A;
  Vec _diff;
  Vec Adiff;
};

static PetscErrorCode TaoTermDestroy_Quadratic(TaoTerm term)
{
  TaoTerm_Quadratic *quad = (TaoTerm_Quadratic *)term->data;

  PetscFunctionBegin;
  PetscCall(MatDestroy(&quad->A));
  PetscCall(VecDestroy(&quad->_diff));
  PetscCall(VecDestroy(&quad->Adiff));
  PetscCall(PetscFree(quad));
  term->data = NULL;
  PetscCall(TaoTermDestroy_ElementwiseDivergence_Internal(term));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermQuadraticSetMat_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermQuadraticGetMat_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermView_Quadratic(TaoTerm term, PetscViewer viewer)
{
  TaoTerm_Quadratic *quad = (TaoTerm_Quadratic *)term->data;

  PetscFunctionBegin;
  if (quad->A) {
    PetscBool iascii;

    PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));

    if (iascii) {
      PetscViewerFormat format;
      PetscBool         pop = PETSC_FALSE;

      PetscCall(PetscViewerGetFormat(viewer, &format));
      if (format != PETSC_VIEWER_ASCII_INFO_DETAIL) {
        PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_INFO));
        pop = PETSC_TRUE;
      }
      PetscCall(MatView(quad->A, viewer));
      if (pop) PetscCall(PetscViewerPopFormat(viewer));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermQuadraticDiff(TaoTerm term, Vec x, Vec params, Vec *diff)
{
  PetscFunctionBegin;
  *diff = x;
  if (params) {
    TaoTerm_Quadratic *quad = (TaoTerm_Quadratic *)term->data;
    if (quad->_diff == NULL) PetscCall(VecDuplicate(x, &quad->_diff));
    PetscCall(VecCopy(x, quad->_diff));
    PetscCall(VecAXPY(quad->_diff, -1.0, params));
    *diff = quad->_diff;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermComputeObjective_Quadratic(TaoTerm term, Vec x, Vec params, PetscReal *value)
{
  TaoTerm_Quadratic *quad = (TaoTerm_Quadratic *)term->data;
  Vec                diff;
  PetscScalar        sval;

  PetscFunctionBegin;
  PetscCheck(quad->A, PetscObjectComm((PetscObject)term), PETSC_ERR_ORDER, "Quadratic matrix not set, call TaoTermQuadraticSetMat() first");
  PetscCall(TaoTermQuadraticDiff(term, x, params, &diff));
  if (quad->Adiff == NULL) PetscCall(VecDuplicate(diff, &quad->Adiff));
  PetscCall(MatMult(quad->A, diff, quad->Adiff));
  PetscCall(VecDot(diff, quad->Adiff, &sval));
  *value = 0.5 * PetscRealPart(sval);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermComputeGradient_Quadratic(TaoTerm term, Vec x, Vec params, Vec g)
{
  TaoTerm_Quadratic *quad = (TaoTerm_Quadratic *)term->data;
  Vec                diff;

  PetscFunctionBegin;
  PetscCheck(quad->A, PetscObjectComm((PetscObject)term), PETSC_ERR_ORDER, "Quadratic matrix not set, call TaoTermQuadraticSetMat() first");
  PetscCall(TaoTermQuadraticDiff(term, x, params, &diff));
  PetscCall(MatMult(quad->A, diff, g));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermComputeObjectiveAndGradient_Quadratic(TaoTerm term, Vec x, Vec params, PetscReal *value, Vec g)
{
  TaoTerm_Quadratic *quad = (TaoTerm_Quadratic *)term->data;
  Vec                diff;
  PetscScalar        sval;

  PetscFunctionBegin;
  PetscCheck(quad->A, PetscObjectComm((PetscObject)term), PETSC_ERR_ORDER, "Quadratic matrix not set, call TaoTermQuadraticSetMat() first");
  PetscCall(TaoTermQuadraticDiff(term, x, params, &diff));
  PetscCall(MatMult(quad->A, diff, g));
  PetscCall(VecDot(diff, g, &sval));
  *value = 0.5 * PetscRealPart(sval);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermComputeHessian_Quadratic(TaoTerm term, Vec x, Vec params, Mat H, Mat Hpre)
{
  TaoTerm_Quadratic *quad = (TaoTerm_Quadratic *)term->data;

  PetscFunctionBegin;
  PetscCheck(quad->A, PetscObjectComm((PetscObject)term), PETSC_ERR_ORDER, "Quadratic matrix not set, call TaoTermQuadraticSetMat() first");
  // TODO caching to avoid unnecessary computation
  if (H) PetscCall(MatCopy(quad->A, H, UNKNOWN_NONZERO_PATTERN));
  if (Hpre && Hpre != H) PetscCall(MatCopy(quad->A, Hpre, UNKNOWN_NONZERO_PATTERN));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermCreateHessianMatrices_Quadratic(TaoTerm term, Mat *H, Mat *Hpre)
{
  TaoTerm_Quadratic *quad = (TaoTerm_Quadratic *)term->data;

  PetscFunctionBegin;
  PetscCheck(quad->A, PetscObjectComm((PetscObject)term), PETSC_ERR_ORDER, "Quadratic matrix not set, call TaoTermQuadraticSetMat() first");
  PetscCall(PetscInfo(term, "%s: Creating TAOTERMQUADRATIC Hessian Matrices by duplicating quadratic matrix set by TaoTermQuadraticSetMat, overriding custom MatType options.\n", ((PetscObject)term)->prefix));
  if (H) PetscCall(MatDuplicate(quad->A, MAT_DO_NOT_COPY_VALUES, H));
  if (Hpre) {
    if (term->Hpre_is_H && H) {
      PetscCall(PetscObjectReference((PetscObject)*H));
      *Hpre = *H;
    } else PetscCall(MatDuplicate(quad->A, MAT_DO_NOT_COPY_VALUES, Hpre));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermQuadraticGetMat - Get the matrix defining a `TaoTerm` of type `TAOTERMQUADRATIC`

  Not collective

  Input Parameter:
. term - a `TaoTerm` of type `TAOTERMQUADRATIC`

  Output Parameter:
. A - the matrix

  Level: intermediate

  Note:
  This function will return `NULL` if the term is not a `TAOTERMQUADRATIC`.

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TAOTERMQUADRATIC`,
          `TaoTermQuadraticSetMat()`
@*/
PetscErrorCode TaoTermQuadraticGetMat(TaoTerm term, Mat *A)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscAssertPointer(A, 2);
  *A = NULL;
  PetscTryMethod(term, "TaoTermQuadraticGetMat_C", (TaoTerm, Mat *), (term, A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermQuadraticGetMat_Quadratic(TaoTerm term, Mat *A)
{
  TaoTerm_Quadratic *quad = (TaoTerm_Quadratic *)term->data;

  PetscFunctionBegin;
  *A = quad->A;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermQuadraticSetMat - Set the matrix defining a `TaoTerm` of type `TAOTERMQUADRATIC`

  Collective

  Input Parameters:
+ term - a `TaoTerm` of type `TAOTERMQUADRATIC`
- A    - the matrix

  Level: intermediate

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TAOTERMQUADRATIC`,
          `TaoTermQuadraticGetMat()`
@*/
PetscErrorCode TaoTermQuadraticSetMat(TaoTerm term, Mat A)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscValidHeaderSpecific(A, MAT_CLASSID, 2);
  PetscCheckSameComm(term, 1, A, 2);
  PetscTryMethod(term, "TaoTermQuadraticSetMat_C", (TaoTerm, Mat), (term, A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermQuadraticSetMat_Quadratic(TaoTerm term, Mat A)
{
  TaoTerm_Quadratic *quad = (TaoTerm_Quadratic *)term->data;
  PetscLayout        rmap, cmap;
  VecType            vec_type;
  PetscBool          is_square;

  PetscFunctionBegin;
  if (A != quad->A) {
    MatType mat_type;
    PetscCall(MatGetLayouts(A, &rmap, &cmap));
    PetscCall(PetscLayoutCompare(rmap, cmap, &is_square));
    PetscCheck(is_square, PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_SIZ, "Matrix for quadratic term must be square");
    PetscCall(PetscObjectReference((PetscObject)A));
    PetscCall(MatDestroy(&quad->A));
    PetscCall(VecDestroy(&quad->_diff));
    PetscCall(VecDestroy(&quad->Adiff));
    quad->A = A;
    PetscCall(MatGetVecType(A, &vec_type));
    PetscCall(TaoTermSetSolutionVecType(term, vec_type));
    PetscCall(TaoTermSetParametersVecType(term, vec_type));
    PetscCall(TaoTermSetSolutionLayout(term, rmap));
    PetscCall(TaoTermSetParametersLayout(term, rmap));
    PetscCall(PetscFree(term->H_mattype));
    PetscCall(PetscFree(term->Hpre_mattype));
    PetscCall(MatGetType(A, &mat_type));
    PetscCall(PetscStrallocpy(mat_type, (char **)&term->H_mattype));
    PetscCall(PetscStrallocpy(mat_type, (char **)&term->Hpre_mattype));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermIsComputeHessianFDPossible_Quadratic(TaoTerm term, PetscBool3 *ispossible)
{
  PetscFunctionBegin;
  *ispossible = PETSC_BOOL3_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  TAOTERMQUADRATIC - A `TaoTerm` that computes $\tfrac{1}{2}(x - p)^T A (x - p)$, for a fixed matrix $A$, solution $x$ and parameters $p$.

  Level: intermediate

  Notes:
  This term is `TAOTERM_PARAMETERS_OPTIONAL`.  If the parameters argument is `NULL` for
  evaluation routines the term computes $\tfrac{1}{2}x^T A x$.

  The matrix $A$ must be symmetric.

  The default Hessian creation mode (see `TaoTermGetCreateHessianMode()`) is `H == Hpre` and `TaoTermCreateHessianMatrices()`
  will create a matrix with the same type as $A$.

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermType`,
          `TaoTermCreateQuadratic()`,
          `TAOTERMHALFL2SQUARED`,
          `TAOTERML1`, `TaoTermQuadraticSetMat()`
M*/
PETSC_INTERN PetscErrorCode TaoTermCreate_Quadratic(TaoTerm term)
{
  TaoTerm_Quadratic *quad;

  PetscFunctionBegin;
  PetscCall(TaoTermCreate_ElementwiseDivergence_Internal(term));
  PetscCall(PetscNew(&quad));
  term->data = (void *)quad;

  PetscCall(PetscFree(term->H_mattype));
  PetscCall(PetscFree(term->Hpre_mattype));

  term->ops->destroy                    = TaoTermDestroy_Quadratic;
  term->ops->view                       = TaoTermView_Quadratic;
  term->ops->objective                  = TaoTermComputeObjective_Quadratic;
  term->ops->gradient                   = TaoTermComputeGradient_Quadratic;
  term->ops->objectiveandgradient       = TaoTermComputeObjectiveAndGradient_Quadratic;
  term->ops->hessian                    = TaoTermComputeHessian_Quadratic;
  term->ops->createhessianmatrices      = TaoTermCreateHessianMatrices_Quadratic;
  term->ops->iscomputehessianfdpossible = TaoTermIsComputeHessianFDPossible_Quadratic;

  term->Hpre_is_H = PETSC_TRUE;

  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermQuadraticGetMat_C", TaoTermQuadraticGetMat_Quadratic));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermQuadraticSetMat_C", TaoTermQuadraticSetMat_Quadratic));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermCreateQuadratic - Create a `TAOTERMQUADRATIC` for a given matrix

  Collective

  Input Parameter:
. A - a square matrix

  Output Parameter:
. term - a `TaoTerm` that implements $\tfrac{1}{2}(x - p)^T A (x - p)$

  Level: beginner

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermCreate()`,
          `TAOTERMQUADRATIC`,
          `TaoTermCreateHalfL2Squared()`,
          `TaoTermCreateL1()`, `TaoTermQuadraticSetMat()`
@*/
PetscErrorCode TaoTermCreateQuadratic(Mat A, TaoTerm *term)
{
  TaoTerm _term;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscAssertPointer(term, 2);
  PetscCall(TaoTermCreate(PetscObjectComm((PetscObject)A), &_term));
  PetscCall(TaoTermSetType(_term, TAOTERMQUADRATIC));
  PetscCall(TaoTermQuadraticSetMat(_term, A));
  *term = _term;
  PetscFunctionReturn(PETSC_SUCCESS);
}
