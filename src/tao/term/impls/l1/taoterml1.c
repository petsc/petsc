#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/

typedef struct _n_TaoTerm_L1 TaoTerm_L1;

struct _n_TaoTerm_L1 {
  PetscReal        epsilon;
  PetscBool        epsilon_warning;
  Vec              diff; // caches $x - p$
  Vec              d;    // caches $d_i = sqrt((diff_i)^2 + epsilon^2)
  PetscReal        d_epsilon;
  PetscReal        d_sum;
  PetscObjectId    d_x_id, d_p_id;
  PetscObjectState d_x_state, d_p_state;
  Vec              diag;
  PetscObjectId    diag_x_id, diag_p_id;
  PetscObjectState diag_x_state, diag_p_state;
  PetscReal        diag_epsilon;
};

static PetscErrorCode TaoTermDestroy_L1(TaoTerm term)
{
  TaoTerm_L1 *l1 = (TaoTerm_L1 *)term->data;

  PetscFunctionBegin;
  PetscCall(VecDestroy(&l1->diff));
  PetscCall(VecDestroy(&l1->d));
  PetscCall(VecDestroy(&l1->diag));
  PetscCall(PetscFree(l1));
  term->data = NULL;
  PetscCall(TaoTermDestroy_ElementwiseDivergence_Internal(term));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermL1SetEpsilon_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermL1GetEpsilon_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermL1ComputeData(TaoTerm term, Vec x, Vec params, Vec *_diff, Vec *d)
{
  TaoTerm_L1      *l1 = (TaoTerm_L1 *)term->data;
  PetscObjectId    x_id, p_id       = 0;
  PetscObjectState x_state, p_state = 0;
  Vec              diff = x;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetId((PetscObject)x, &x_id));
  PetscCall(PetscObjectStateGet((PetscObject)x, &x_state));
  if (params) {
    PetscCall(PetscObjectGetId((PetscObject)params, &p_id));
    PetscCall(PetscObjectStateGet((PetscObject)params, &p_state));
  }
  if (l1->d_x_id != x_id || l1->d_x_state != x_state || l1->d_p_id != p_id || l1->d_p_state != p_state || l1->d_epsilon != l1->epsilon) {
    l1->d_x_id    = x_id;
    l1->d_x_state = x_state;
    l1->d_p_id    = p_id;
    l1->d_p_state = p_state;
    l1->d_epsilon = l1->epsilon;
    diff          = x;
    if (params) {
      PetscCall(VecIfNotCongruentGetSameLayoutVec(x, &l1->diff));
      PetscCall(VecWAXPY(l1->diff, -1.0, params, x));
      diff = l1->diff;
    }
    if (l1->epsilon != 0.0) {
      PetscCall(VecIfNotCongruentGetSameLayoutVec(x, &l1->d));
      PetscCall(VecPointwiseMult(l1->d, diff, diff));
      PetscCall(VecShift(l1->d, l1->epsilon * l1->epsilon));
      PetscCall(VecSqrtAbs(l1->d));
    }
  }
  if (params) diff = l1->diff;
  *_diff = diff;
  *d     = l1->d;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermL1ComputeDiag(TaoTerm term, Vec x, Vec params, Vec *diag)
{
  Vec              diff, d;
  TaoTerm_L1      *l1 = (TaoTerm_L1 *)term->data;
  PetscObjectId    x_id, p_id       = 0;
  PetscObjectState x_state, p_state = 0;

  PetscFunctionBegin;
  PetscCall(TaoTermL1ComputeData(term, x, params, &diff, &d));
  PetscCall(PetscObjectGetId((PetscObject)x, &x_id));
  PetscCall(PetscObjectStateGet((PetscObject)x, &x_state));
  if (params) {
    PetscCall(PetscObjectGetId((PetscObject)params, &p_id));
    PetscCall(PetscObjectStateGet((PetscObject)params, &p_state));
  }
  if (l1->diag_x_id != x_id || l1->diag_x_state != x_state || l1->diag_p_id != p_id || l1->diag_p_state != p_state || l1->diag_epsilon != l1->epsilon) {
    l1->diag_x_id    = x_id;
    l1->diag_x_state = x_state;
    l1->diag_p_id    = p_id;
    l1->diag_p_state = p_state;
    l1->diag_epsilon = l1->epsilon;
    if (l1->epsilon != 0.0) {
      PetscCall(VecIfNotCongruentGetSameLayoutVec(x, &l1->diag));
      PetscCall(VecCopy(d, l1->diag));
      PetscCall(VecPointwiseMult(l1->diag, l1->diag, d));
      PetscCall(VecPointwiseMult(l1->diag, l1->diag, d));
      PetscCall(VecReciprocal(l1->diag));
      PetscCall(VecScale(l1->diag, l1->epsilon * l1->epsilon));
    }
  }
  *diag = l1->diag;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermComputeObjective_L1_Internal(TaoTerm term, Vec diff, Vec d, PetscReal *value)
{
  TaoTerm_L1 *l1 = (TaoTerm_L1 *)term->data;

  PetscFunctionBegin;
  if (l1->epsilon == 0.0) {
    PetscCall(VecNorm(diff, NORM_1, value));
  } else {
    PetscScalar sum;
    PetscInt    n;

    PetscCall(VecGetSize(d, &n));
    PetscCall(VecSum(d, &sum));
    *value = PetscRealPart(sum) - n * l1->epsilon;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermComputeObjective_L1(TaoTerm term, Vec x, Vec params, PetscReal *value)
{
  Vec diff, d;

  PetscFunctionBegin;
  PetscCall(TaoTermL1ComputeData(term, x, params, &diff, &d));
  PetscCall(TaoTermComputeObjective_L1_Internal(term, diff, d, value));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermL1DerivativeCheck(TaoTerm term)
{
  TaoTerm_L1 *l1 = (TaoTerm_L1 *)term->data;

  PetscFunctionBegin;
  if (l1->epsilon_warning == PETSC_FALSE) {
    l1->epsilon_warning = PETSC_TRUE;
    PetscCall(PetscInfo(term, "%s: Asking for derivatives of l1 norm, which is not smooth.  Consider smoothing the TaoTerm with TaoTermL1SetEpsilon() or using a derivative-free Tao solver\n", ((PetscObject)term)->prefix));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermComputeGradient_L1_Internal(TaoTerm term, Vec diff, Vec d, Vec g)
{
  TaoTerm_L1 *l1 = (TaoTerm_L1 *)term->data;

  PetscFunctionBegin;
  if (l1->epsilon == 0.0) {
    PetscCall(TaoTermL1DerivativeCheck(term));
    PetscCall(VecPointwiseSign(g, diff, VEC_SIGN_ZERO_TO_ZERO));
  } else {
    PetscCall(VecPointwiseDivide(g, diff, d));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermComputeGradient_L1(TaoTerm term, Vec x, Vec params, Vec g)
{
  Vec diff, d;

  PetscFunctionBegin;
  PetscCall(TaoTermL1ComputeData(term, x, params, &diff, &d));
  PetscCall(TaoTermComputeGradient_L1_Internal(term, diff, d, g));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermComputeObjectiveAndGradient_L1(TaoTerm term, Vec x, Vec params, PetscReal *value, Vec g)
{
  Vec diff, d;

  PetscFunctionBegin;
  PetscCall(TaoTermL1ComputeData(term, x, params, &diff, &d));
  PetscCall(TaoTermComputeObjective_L1_Internal(term, diff, d, value));
  PetscCall(TaoTermComputeGradient_L1_Internal(term, diff, d, g));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermComputeHessian_L1_Internal(TaoTerm term, Vec diag, Mat H)
{
  TaoTerm_L1 *l1 = (TaoTerm_L1 *)term->data;

  PetscFunctionBegin;
  if (l1->epsilon == 0.0) {
    PetscCall(TaoTermL1DerivativeCheck(term));
    PetscCall(MatZeroEntries(H));
    PetscCall(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));
  } else {
    PetscCall(MatDiagonalSet(H, diag, INSERT_VALUES));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermComputeHessian_L1(TaoTerm term, Vec x, Vec params, Mat H, Mat Hpre)
{
  Vec diag = NULL; /* Appease -Wmaybe-uninitialized */

  PetscFunctionBegin;
  if (H == NULL && Hpre == NULL) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(TaoTermL1ComputeDiag(term, x, params, &diag));
  if (H) PetscCall(TaoTermComputeHessian_L1_Internal(term, diag, H));
  if (Hpre && Hpre != H) PetscCall(TaoTermComputeHessian_L1_Internal(term, diag, Hpre));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermCreateHessianMatrices_L1(TaoTerm term, Mat *H, Mat *Hpre)
{
  PetscBool is_hdiag, is_hprediag;

  PetscFunctionBegin;
  PetscCall(PetscInfo(term, "%s: Creating TAOTERML1 Hessian Matrices. TAOTERML1 only accepts MATDIAGONAL for MatType, overriding any user-set MatType.\n", ((PetscObject)term)->prefix));
  PetscCall(PetscStrcmp(term->H_mattype, MATDIAGONAL, &is_hdiag));
  PetscCall(PetscStrcmp(term->Hpre_mattype, MATDIAGONAL, &is_hprediag));
  if (!is_hdiag) {
    PetscCall(PetscFree(term->H_mattype));
    PetscCall(PetscStrallocpy(MATDIAGONAL, (char **)&term->H_mattype));
  }
  if (!is_hprediag) {
    PetscCall(PetscFree(term->Hpre_mattype));
    PetscCall(PetscStrallocpy(MATDIAGONAL, (char **)&term->Hpre_mattype));
  }
  PetscCall(TaoTermCreateHessianMatricesDefault(term, H, Hpre));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermL1SetEpsilon - Set an $\epsilon$ smoothing parameter.

  Logically collective

  Input Parameters:
+ term    - a `TaoTerm` of type `TAOTERML1`
- epsilon - a real number $\geq 0$

  Options Database Keys:
. -tao_term_l1_epsilon <real> - $\epsilon$

  Level: advanced

  If $\epsilon = 0$ (the default), then `term` computes $\|x - p\|_1$, but if $\epsilon > 0$, then it computes
  $\sum_{i=0}^{n-1} \left(\sqrt{(x_i-p_i)^2 + \epsilon^2} - \epsilon\right)$.

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TAOTERML1`,
          `TaoTermL1GetEpsilon()`
@*/
PetscErrorCode TaoTermL1SetEpsilon(TaoTerm term, PetscReal epsilon)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscValidLogicalCollectiveReal(term, epsilon, 2);
  PetscTryMethod(term, "TaoTermL1SetEpsilon_C", (TaoTerm, PetscReal), (term, epsilon));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermL1SetEpsilon_L1(TaoTerm term, PetscReal epsilon)
{
  TaoTerm_L1 *l1 = (TaoTerm_L1 *)term->data;

  PetscFunctionBegin;
  PetscCheck(epsilon >= 0, PetscObjectComm((PetscObject)term), PETSC_ERR_ARG_OUTOFRANGE, "L1 epsilon (%g) cannot be < 0.0", (double)epsilon);
  l1->epsilon = epsilon;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermL1GetEpsilon - Get the $\epsilon$ smoothing parameter set by `TaoTermL1SetEpsilon()`.

  Not collective

  Input Parameter:
. term - a `TaoTerm` of type `TAOTERML1`

  Output Parameter:
. epsilon - the smoothing parameter

  Level: advanced

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TAOTERML1`,
          `TaoTermL1SetEpsilon()`
@*/
PetscErrorCode TaoTermL1GetEpsilon(TaoTerm term, PetscReal *epsilon)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(term, TAOTERM_CLASSID, 1);
  PetscAssertPointer(epsilon, 2);
  PetscUseMethod(term, "TaoTermL1GetEpsilon_C", (TaoTerm, PetscReal *), (term, epsilon));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermL1GetEpsilon_L1(TaoTerm term, PetscReal *epsilon)
{
  TaoTerm_L1 *l1 = (TaoTerm_L1 *)term->data;

  PetscFunctionBegin;
  *epsilon = l1->epsilon;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermView_L1(TaoTerm term, PetscViewer viewer)
{
  TaoTerm_L1 *l1 = (TaoTerm_L1 *)term->data;
  PetscBool   is_ascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &is_ascii));
  if (is_ascii) PetscCall(PetscViewerASCIIPrintf(viewer, "epsilon (tao_term_l1_epsilon): %g\n", (double)l1->epsilon));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermSetFromOptions_L1(TaoTerm term, PetscOptionItems PetscOptionsObject)
{
  TaoTerm_L1 *l1 = (TaoTerm_L1 *)term->data;
  PetscBool   is_hdiag, is_hprediag;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "TaoTerm l1 options");
  PetscCall(PetscOptionsBoundedReal("-tao_term_l1_epsilon", "smoothing parameter", "TaoTermL1SetEpsilon", l1->epsilon, &l1->epsilon, NULL, 0.0));
  PetscOptionsHeadEnd();
  PetscCall(PetscStrcmp(term->H_mattype, MATDIAGONAL, &is_hdiag));
  PetscCall(PetscStrcmp(term->Hpre_mattype, MATDIAGONAL, &is_hprediag));
  if (!is_hdiag) {
    PetscCall(PetscFree(term->H_mattype));
    PetscCall(PetscStrallocpy(MATDIAGONAL, (char **)&term->H_mattype));
  }
  if (!is_hprediag) {
    PetscCall(PetscFree(term->Hpre_mattype));
    PetscCall(PetscStrallocpy(MATDIAGONAL, (char **)&term->Hpre_mattype));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermIsComputeHessianFDPossible_L1(TaoTerm term, PetscBool3 *ispossible)
{
  PetscFunctionBegin;
  *ispossible = PETSC_BOOL3_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  TAOTERML1 - A `TaoTerm` that computes $\|x - p\|_1$, for solution $x$ and parameters $p$.

  Level: intermediate

  Options Database Keys:
. -tao_term_l1_epsilon <real> - (default 0.0) a smoothing parameter (see `TaoTermL1SetEpsilon()`)

  Notes:
  This term is `TAOTERM_PARAMETERS_OPTIONAL`.  If the parameters argument is `NULL` for
  evaluation routines the term computes $\|x\|_1$.

  This term has a smoothing parameter $\epsilon$ that defaults to 0: if $\epsilon > 0$,
  the term computes a smooth approximation of $\|x - p\|_1$, see `TaoTermL1SetEpsilon()`.

  The default Hessian creation mode (see `TaoTermGetCreateHessianMode()`) is `H == Hpre` and `TaoTermCreateHessianMatrices()`
  will create a `MATDIAGONAL` for the Hessian.

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermType`,
          `TaoTermCreateL1()`,
          `TaoTermL1GetEpsilon()`,
          `TaoTermL1SetEpsilon()`,
          `TAOTERMHALFL2SQUARED`,
          `TAOTERMQUADRATIC`
M*/
PETSC_INTERN PetscErrorCode TaoTermCreate_L1(TaoTerm term)
{
  TaoTerm_L1 *l1;

  PetscFunctionBegin;
  PetscCall(TaoTermCreate_ElementwiseDivergence_Internal(term));
  PetscCall(PetscNew(&l1));
  term->data = (void *)l1;

  PetscCall(PetscFree(term->H_mattype));
  PetscCall(PetscFree(term->Hpre_mattype));

  PetscCall(PetscStrallocpy(MATDIAGONAL, (char **)&term->H_mattype));
  PetscCall(PetscStrallocpy(MATDIAGONAL, (char **)&term->Hpre_mattype));

  term->ops->destroy                    = TaoTermDestroy_L1;
  term->ops->view                       = TaoTermView_L1;
  term->ops->setfromoptions             = TaoTermSetFromOptions_L1;
  term->ops->objective                  = TaoTermComputeObjective_L1;
  term->ops->gradient                   = TaoTermComputeGradient_L1;
  term->ops->objectiveandgradient       = TaoTermComputeObjectiveAndGradient_L1;
  term->ops->hessian                    = TaoTermComputeHessian_L1;
  term->ops->createhessianmatrices      = TaoTermCreateHessianMatrices_L1;
  term->ops->iscomputehessianfdpossible = TaoTermIsComputeHessianFDPossible_L1;

  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermL1SetEpsilon_C", TaoTermL1SetEpsilon_L1));
  PetscCall(PetscObjectComposeFunction((PetscObject)term, "TaoTermL1GetEpsilon_C", TaoTermL1GetEpsilon_L1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermCreateL1 - Create a `TaoTerm` for the objective function term $\|x - p\|_1$.

  Collective

  Input Parameters:
+ comm    - the MPI communicator where the term will be computed
. n       - the local size of the $x$ and $p$ vectors (or `PETSC_DECIDE`)
. N       - the global size of the $x$ and $p$ vectors (or `PETSC_DECIDE`)
- epsilon - a non-negative smoothing parameter (see `TaoTermL1SetEpsilon()`)

  Output Parameter:
. term - the `TaoTerm`

  Level: beginner

  Note:
  If you would like to add an L1 regularization term $\alpha \|x\|_1$ to the objective function of a `Tao`, do the following\:
.vb
  VecGetLocalSize(x, &n);
  VecGetSize(x, &N);
  TaoTermCreateL1(PetscObjectComm((PetscObject)x), n, N, 0.0, &term);
  TaoAddTerm(tao, "reg_", alpha, term, NULL, NULL);
  TaoTermDestroy(&term);
.ve
  If you would like to have a dictionary matrix term $\alpha \|D x\|_1$, do the same but pass `D` as the map of the term\:
.vb
  MatGetLocalSize(D, &m, NULL);
  MatGetSize(D, &M, NULL);
  TaoTermCreateL1(PetscObjectComm((PetscObject)D), m, M, 0.0, &term);
  TaoAddTerm(tao, "reg_", alpha, term, NULL, D);
  TaoTermDestroy(&term);
.ve

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TAOTERML1`,
          `TaoTermL1GetEpsilon()`,
          `TaoTermL1SetEpsilon()`,
          `TaoTermCreateHalfL2Squared()`,
          `TaoTermCreateQuadratic()`
@*/
PetscErrorCode TaoTermCreateL1(MPI_Comm comm, PetscInt n, PetscInt N, PetscReal epsilon, TaoTerm *term)
{
  TaoTerm _term;

  PetscFunctionBegin;
  PetscAssertPointer(term, 5);
  PetscCheck(epsilon >= 0, comm, PETSC_ERR_ARG_OUTOFRANGE, "L1 epsilon (%g) cannot be < 0.0", (double)epsilon);
  PetscCall(TaoTermCreate(comm, &_term));
  PetscCall(TaoTermSetType(_term, TAOTERML1));
  PetscCall(TaoTermSetSolutionSizes(_term, n, N, 1));
  PetscCall(TaoTermL1SetEpsilon(_term, epsilon));
  *term = _term;
  PetscFunctionReturn(PETSC_SUCCESS);
}
