#include <../src/tao/unconstrained/impls/neldermead/neldermead.h>
#include <petscvec.h>

/*------------------------------------------------------------*/
static PetscErrorCode NelderMeadSort(TAO_NelderMead *nm)
{
  PetscReal *values  = nm->f_values;
  PetscInt  *indices = nm->indices;
  PetscInt   dim     = nm->N + 1;
  PetscInt   i, j, index;
  PetscReal  val;

  PetscFunctionBegin;
  for (i = 1; i < dim; i++) {
    index = indices[i];
    val   = values[index];
    for (j = i - 1; j >= 0 && values[indices[j]] > val; j--) indices[j + 1] = indices[j];
    indices[j + 1] = index;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
static PetscErrorCode NelderMeadReplace(TAO_NelderMead *nm, PetscInt index, Vec Xmu, PetscReal f)
{
  PetscFunctionBegin;
  /*  Add new vector's fraction of average */
  PetscCall(VecAXPY(nm->Xbar, nm->oneOverN, Xmu));
  PetscCall(VecCopy(Xmu, nm->simplex[index]));
  nm->f_values[index] = f;

  PetscCall(NelderMeadSort(nm));

  /*  Subtract last vector from average */
  PetscCall(VecAXPY(nm->Xbar, -nm->oneOverN, nm->simplex[nm->indices[nm->N]]));
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------- */
static PetscErrorCode TaoSetUp_NM(Tao tao)
{
  TAO_NelderMead *nm = (TAO_NelderMead *)tao->data;
  PetscInt        n;

  PetscFunctionBegin;
  PetscCall(VecGetSize(tao->solution, &n));
  nm->N        = n;
  nm->oneOverN = 1.0 / n;
  PetscCall(VecDuplicateVecs(tao->solution, nm->N + 1, &nm->simplex));
  PetscCall(PetscMalloc1(nm->N + 1, &nm->f_values));
  PetscCall(PetscMalloc1(nm->N + 1, &nm->indices));
  PetscCall(VecDuplicate(tao->solution, &nm->Xbar));
  PetscCall(VecDuplicate(tao->solution, &nm->Xmur));
  PetscCall(VecDuplicate(tao->solution, &nm->Xmue));
  PetscCall(VecDuplicate(tao->solution, &nm->Xmuc));

  tao->gradient = NULL;
  tao->step     = 0;
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------- */
static PetscErrorCode TaoDestroy_NM(Tao tao)
{
  TAO_NelderMead *nm = (TAO_NelderMead *)tao->data;

  PetscFunctionBegin;
  if (tao->setupcalled) {
    PetscCall(VecDestroyVecs(nm->N + 1, &nm->simplex));
    PetscCall(VecDestroy(&nm->Xmuc));
    PetscCall(VecDestroy(&nm->Xmue));
    PetscCall(VecDestroy(&nm->Xmur));
    PetscCall(VecDestroy(&nm->Xbar));
  }
  PetscCall(PetscFree(nm->indices));
  PetscCall(PetscFree(nm->f_values));
  PetscCall(PetscFree(tao->data));
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
static PetscErrorCode TaoSetFromOptions_NM(Tao tao, PetscOptionItems *PetscOptionsObject)
{
  TAO_NelderMead *nm = (TAO_NelderMead *)tao->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "Nelder-Mead options");
  PetscCall(PetscOptionsReal("-tao_nm_lamda", "initial step length", "", nm->lamda, &nm->lamda, NULL));
  PetscCall(PetscOptionsReal("-tao_nm_mu", "mu", "", nm->mu_oc, &nm->mu_oc, NULL));
  nm->mu_ic = -nm->mu_oc;
  nm->mu_r  = nm->mu_oc * 2.0;
  nm->mu_e  = nm->mu_oc * 4.0;
  PetscOptionsHeadEnd();
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
static PetscErrorCode TaoView_NM(Tao tao, PetscViewer viewer)
{
  TAO_NelderMead *nm = (TAO_NelderMead *)tao->data;
  PetscBool       isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer, "expansions: %" PetscInt_FMT "\n", nm->nexpand));
    PetscCall(PetscViewerASCIIPrintf(viewer, "reflections: %" PetscInt_FMT "\n", nm->nreflect));
    PetscCall(PetscViewerASCIIPrintf(viewer, "inside contractions: %" PetscInt_FMT "\n", nm->nincontract));
    PetscCall(PetscViewerASCIIPrintf(viewer, "outside contractionss: %" PetscInt_FMT "\n", nm->noutcontract));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Shrink steps: %" PetscInt_FMT "\n", nm->nshrink));
    PetscCall(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
static PetscErrorCode TaoSolve_NM(Tao tao)
{
  TAO_NelderMead *nm = (TAO_NelderMead *)tao->data;
  PetscReal      *x;
  PetscInt        i;
  Vec             Xmur = nm->Xmur, Xmue = nm->Xmue, Xmuc = nm->Xmuc, Xbar = nm->Xbar;
  PetscReal       fr, fe, fc;
  PetscInt        shrink;
  PetscInt        low, high;

  PetscFunctionBegin;
  nm->nshrink      = 0;
  nm->nreflect     = 0;
  nm->nincontract  = 0;
  nm->noutcontract = 0;
  nm->nexpand      = 0;

  if (tao->XL || tao->XU || tao->ops->computebounds) PetscCall(PetscInfo(tao, "WARNING: Variable bounds have been set but will be ignored by NelderMead algorithm\n"));

  PetscCall(VecCopy(tao->solution, nm->simplex[0]));
  PetscCall(TaoComputeObjective(tao, nm->simplex[0], &nm->f_values[0]));
  nm->indices[0] = 0;
  for (i = 1; i < nm->N + 1; i++) {
    PetscCall(VecCopy(tao->solution, nm->simplex[i]));
    PetscCall(VecGetOwnershipRange(nm->simplex[i], &low, &high));
    if (i - 1 >= low && i - 1 < high) {
      PetscCall(VecGetArray(nm->simplex[i], &x));
      x[i - 1 - low] += nm->lamda;
      PetscCall(VecRestoreArray(nm->simplex[i], &x));
    }

    PetscCall(TaoComputeObjective(tao, nm->simplex[i], &nm->f_values[i]));
    nm->indices[i] = i;
  }

  /*  Xbar  = (Sum of all simplex vectors - worst vector)/N */
  PetscCall(NelderMeadSort(nm));
  PetscCall(VecSet(Xbar, 0.0));
  for (i = 0; i < nm->N; i++) PetscCall(VecAXPY(Xbar, 1.0, nm->simplex[nm->indices[i]]));
  PetscCall(VecScale(Xbar, nm->oneOverN));
  tao->reason = TAO_CONTINUE_ITERATING;
  while (1) {
    /* Call general purpose update function */
    PetscTryTypeMethod(tao, update, tao->niter, tao->user_update);
    ++tao->niter;
    shrink = 0;
    PetscCall(VecCopy(nm->simplex[nm->indices[0]], tao->solution));
    PetscCall(TaoLogConvergenceHistory(tao, nm->f_values[nm->indices[0]], nm->f_values[nm->indices[nm->N]] - nm->f_values[nm->indices[0]], 0.0, tao->ksp_its));
    PetscCall(TaoMonitor(tao, tao->niter, nm->f_values[nm->indices[0]], nm->f_values[nm->indices[nm->N]] - nm->f_values[nm->indices[0]], 0.0, 1.0));
    PetscUseTypeMethod(tao, convergencetest, tao->cnvP);
    if (tao->reason != TAO_CONTINUE_ITERATING) break;

    /* x(mu) = (1 + mu)Xbar - mu*X_N+1 */
    PetscCall(VecAXPBYPCZ(Xmur, 1 + nm->mu_r, -nm->mu_r, 0, Xbar, nm->simplex[nm->indices[nm->N]]));
    PetscCall(TaoComputeObjective(tao, Xmur, &fr));

    if (nm->f_values[nm->indices[0]] <= fr && fr < nm->f_values[nm->indices[nm->N - 1]]) {
      /*  reflect */
      nm->nreflect++;
      PetscCall(PetscInfo(0, "Reflect\n"));
      PetscCall(NelderMeadReplace(nm, nm->indices[nm->N], Xmur, fr));
    } else if (fr < nm->f_values[nm->indices[0]]) {
      /*  expand */
      nm->nexpand++;
      PetscCall(PetscInfo(0, "Expand\n"));
      PetscCall(VecAXPBYPCZ(Xmue, 1 + nm->mu_e, -nm->mu_e, 0, Xbar, nm->simplex[nm->indices[nm->N]]));
      PetscCall(TaoComputeObjective(tao, Xmue, &fe));
      if (fe < fr) {
        PetscCall(NelderMeadReplace(nm, nm->indices[nm->N], Xmue, fe));
      } else {
        PetscCall(NelderMeadReplace(nm, nm->indices[nm->N], Xmur, fr));
      }
    } else if (nm->f_values[nm->indices[nm->N - 1]] <= fr && fr < nm->f_values[nm->indices[nm->N]]) {
      /* outside contraction */
      nm->noutcontract++;
      PetscCall(PetscInfo(0, "Outside Contraction\n"));
      PetscCall(VecAXPBYPCZ(Xmuc, 1 + nm->mu_oc, -nm->mu_oc, 0, Xbar, nm->simplex[nm->indices[nm->N]]));

      PetscCall(TaoComputeObjective(tao, Xmuc, &fc));
      if (fc <= fr) {
        PetscCall(NelderMeadReplace(nm, nm->indices[nm->N], Xmuc, fc));
      } else shrink = 1;
    } else {
      /* inside contraction */
      nm->nincontract++;
      PetscCall(PetscInfo(0, "Inside Contraction\n"));
      PetscCall(VecAXPBYPCZ(Xmuc, 1 + nm->mu_ic, -nm->mu_ic, 0, Xbar, nm->simplex[nm->indices[nm->N]]));
      PetscCall(TaoComputeObjective(tao, Xmuc, &fc));
      if (fc < nm->f_values[nm->indices[nm->N]]) {
        PetscCall(NelderMeadReplace(nm, nm->indices[nm->N], Xmuc, fc));
      } else shrink = 1;
    }

    if (shrink) {
      nm->nshrink++;
      PetscCall(PetscInfo(0, "Shrink\n"));

      for (i = 1; i < nm->N + 1; i++) {
        PetscCall(VecAXPBY(nm->simplex[nm->indices[i]], 1.5, -0.5, nm->simplex[nm->indices[0]]));
        PetscCall(TaoComputeObjective(tao, nm->simplex[nm->indices[i]], &nm->f_values[nm->indices[i]]));
      }
      PetscCall(VecAXPBY(Xbar, 1.5 * nm->oneOverN, -0.5, nm->simplex[nm->indices[0]]));

      /*  Add last vector's fraction of average */
      PetscCall(VecAXPY(Xbar, nm->oneOverN, nm->simplex[nm->indices[nm->N]]));
      PetscCall(NelderMeadSort(nm));
      /*  Subtract new last vector from average */
      PetscCall(VecAXPY(Xbar, -nm->oneOverN, nm->simplex[nm->indices[nm->N]]));
    }
  }
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------- */
/*MC
 TAONM - Nelder-Mead solver for derivative free, unconstrained minimization

 Options Database Keys:
+ -tao_nm_lamda - initial step length
- -tao_nm_mu - expansion/contraction factor

 Level: beginner
M*/

PETSC_EXTERN PetscErrorCode TaoCreate_NM(Tao tao)
{
  TAO_NelderMead *nm;

  PetscFunctionBegin;
  PetscCall(PetscNew(&nm));
  tao->data = (void *)nm;

  tao->ops->setup          = TaoSetUp_NM;
  tao->ops->solve          = TaoSolve_NM;
  tao->ops->view           = TaoView_NM;
  tao->ops->setfromoptions = TaoSetFromOptions_NM;
  tao->ops->destroy        = TaoDestroy_NM;

  /* Override default settings (unless already changed) */
  if (!tao->max_it_changed) tao->max_it = 2000;
  if (!tao->max_funcs_changed) tao->max_funcs = 4000;

  nm->simplex = NULL;
  nm->lamda   = 1;

  nm->mu_ic = -0.5;
  nm->mu_oc = 0.5;
  nm->mu_r  = 1.0;
  nm->mu_e  = 2.0;

  PetscFunctionReturn(0);
}
