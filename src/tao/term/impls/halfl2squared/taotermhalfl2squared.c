#include <petsc/private/taoimpl.h> /*I "petsctao.h" I*/

typedef struct _n_TaoTerm_HalfL2Squared TaoTerm_HalfL2Squared;

struct _n_TaoTerm_HalfL2Squared {
  Vec pdiff_work;
};

static PetscErrorCode TaoTermDestroy_Halfl2squared(TaoTerm term)
{
  TaoTerm_HalfL2Squared *l2 = (TaoTerm_HalfL2Squared *)term->data;

  PetscFunctionBegin;
  PetscCall(VecDestroy(&l2->pdiff_work));
  PetscCall(PetscFree(l2));
  term->data = NULL;
  PetscCall(TaoTermDestroy_ElementwiseDivergence_Internal(term));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermComputeObjective_Halfl2squared(TaoTerm term, Vec x, Vec params, PetscReal *value)
{
  PetscScalar            v;
  TaoTerm_HalfL2Squared *l2 = (TaoTerm_HalfL2Squared *)term->data;

  PetscFunctionBegin;
  if (params) {
    if (l2->pdiff_work == NULL) PetscCall(VecDuplicate(x, &l2->pdiff_work));

    PetscCall(VecCopy(x, l2->pdiff_work));
    PetscCall(VecAXPY(l2->pdiff_work, -1.0, params));
    PetscCall(VecDot(l2->pdiff_work, l2->pdiff_work, &v));
  } else PetscCall(VecDot(x, x, &v));
  *value = 0.5 * PetscRealPart(v);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermComputeObjectiveAndGradient_Halfl2squared(TaoTerm term, Vec x, Vec params, PetscReal *value, Vec g)
{
  PetscScalar v;

  PetscFunctionBegin;
  if (params) PetscCall(VecWAXPY(g, -1.0, params, x));
  else PetscCall(VecCopy(x, g));
  PetscCall(VecDot(g, g, &v));
  *value = 0.5 * PetscRealPart(v);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermComputeGradient_Halfl2squared(TaoTerm term, Vec x, Vec params, Vec g)
{
  PetscFunctionBegin;
  if (params) PetscCall(VecWAXPY(g, -1.0, params, x));
  else PetscCall(VecCopy(x, g));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermComputeHessian_Halfl2squared(TaoTerm term, Vec x, Vec params, Mat H, Mat Hpre)
{
  PetscFunctionBegin;
  if (H) {
    PetscCall(MatZeroEntries(H));
    PetscCall(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));
    PetscCall(MatShift(H, 1.0));
  }
  if (Hpre && Hpre != H) {
    PetscCall(MatZeroEntries(Hpre));
    PetscCall(MatAssemblyBegin(Hpre, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Hpre, MAT_FINAL_ASSEMBLY));
    PetscCall(MatShift(Hpre, 1.0));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermCreateHessianMatrices_Halfl2squared(TaoTerm term, Mat *H, Mat *Hpre)
{
  PetscBool is_hdiag, is_hprediag;

  PetscFunctionBegin;
  PetscCall(PetscInfo(term, "%s: Creating TAOTERMHALFL2SQUARED Hessian Matrices. TAOTERMHALFL2SQUARED only accepts MATDIAGONAL for MatType, overriding any user-set MatType.\n", ((PetscObject)term)->prefix));
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

static PetscErrorCode TaoTermIsComputeHessianFDPossible_Halfl2squared(TaoTerm term, PetscBool3 *ispossible)
{
  PetscFunctionBegin;
  *ispossible = PETSC_BOOL3_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  TAOTERMHALFL2SQUARED - A `TaoTerm` that computes $\tfrac{1}{2}\|x - p\|_2^2$, for solution $x$ and parameters $p$.

  Level: intermediate

  Notes:
  By default this term is `TAOTERM_PARAMETERS_OPTIONAL`.  If the parameters
  argument is `NULL` in the evaluation routines (`TaoTermComputeObjective()`,
  `TaoTermComputeGradient()`, etc.), then it is assumed $p = 0$ and the term computes
  $\tfrac{1}{2}\|x\|_2^2$.

  The default Hessian creation mode (see `TaoTermGetCreateHessianMode()`) is `H == Hpre` and `TaoTermCreateHessianMatrices()`
  will create a `MATDIAGONAL` for the Hessian.

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TaoTermType`,
          `TaoTermCreateHalfL2Squared()`,
          `TAOTERML1`,
          `TAOTERMQUADRATIC`
M*/
PETSC_INTERN PetscErrorCode TaoTermCreate_Halfl2squared(TaoTerm term)
{
  TaoTerm_HalfL2Squared *l2;

  PetscFunctionBegin;
  PetscCall(TaoTermCreate_ElementwiseDivergence_Internal(term));
  PetscCall(PetscNew(&l2));
  term->data = (void *)l2;

  PetscCall(PetscFree(term->H_mattype));
  PetscCall(PetscFree(term->Hpre_mattype));

  PetscCall(PetscStrallocpy(MATDIAGONAL, (char **)&term->H_mattype));
  PetscCall(PetscStrallocpy(MATDIAGONAL, (char **)&term->Hpre_mattype));

  term->ops->destroy                    = TaoTermDestroy_Halfl2squared;
  term->ops->objective                  = TaoTermComputeObjective_Halfl2squared;
  term->ops->gradient                   = TaoTermComputeGradient_Halfl2squared;
  term->ops->objectiveandgradient       = TaoTermComputeObjectiveAndGradient_Halfl2squared;
  term->ops->hessian                    = TaoTermComputeHessian_Halfl2squared;
  term->ops->createhessianmatrices      = TaoTermCreateHessianMatrices_Halfl2squared;
  term->ops->iscomputehessianfdpossible = TaoTermIsComputeHessianFDPossible_Halfl2squared;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  TaoTermCreateHalfL2Squared - Create a `TaoTerm` for the objective term $\tfrac{1}{2}\|x - p\|_2^2$, for solution $x$ and parameters $p$.

  Collective

  Input Parameters:
+ comm - the MPI communicator where the `TaoTerm` will be computed
. n    - the local size of the $x$ and $p$ vectors (or `PETSC_DECIDE`)
- N    - the global size of the $x$ and $p$ vectors (or `PETSC_DECIDE`)

  Output Parameter:
. term - the `TaoTerm`

  Level: beginner

  Note:
  If you would like to add a Tikhonov regularization term $\alpha \tfrac{1}{2}\|x\|_2^2$ to the objective function of a `Tao`, do the following\:
.vb
  VecGetSizes(x, &n, &N);
  TaoTermCreateHalfL2Squared(PetscObjectComm((PetscObject)x), n, N, &term);
  TaoAddTerm(tao, "reg_", alpha, term, NULL, NULL);
  TaoTermDestroy(&term);
.ve
  If you would like to add a biased regularization term $\alpha \tfrac{1}{2}\|x - p \|_2^2$, do the same but pass `p` as the parameters of the term\:
.vb
  TaoAddTerm(tao, "reg_", alpha, term, p, NULL);
.ve

.seealso: [](sec_tao_term),
          `TaoTerm`,
          `TAOTERMHALFL2SQUARED`,
          `TaoTermCreateL1()`,
          `TaoTermCreateQuadratic()`
@*/
PetscErrorCode TaoTermCreateHalfL2Squared(MPI_Comm comm, PetscInt n, PetscInt N, TaoTerm *term)
{
  TaoTerm _term;

  PetscFunctionBegin;
  PetscAssertPointer(term, 4);
  PetscCall(TaoTermCreate(comm, &_term));
  PetscCall(TaoTermSetType(_term, TAOTERMHALFL2SQUARED));
  PetscCall(TaoTermSetSolutionSizes(_term, n, N, 1));
  *term = _term;
  PetscFunctionReturn(PETSC_SUCCESS);
}
