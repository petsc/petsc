#include <../src/tao/unconstrained/impls/neldermead/neldermead.h>
#include <petscvec.h>

/*------------------------------------------------------------*/
static PetscErrorCode NelderMeadSort(TAO_NelderMead *nm)
{
  PetscReal *values = nm->f_values;
  PetscInt  *indices = nm->indices;
  PetscInt  dim = nm->N+1;
  PetscInt  i,j,index;
  PetscReal val;

  PetscFunctionBegin;
  for (i=1;i<dim;i++) {
    index = indices[i];
    val = values[index];
    for (j=i-1; j>=0 && values[indices[j]] > val; j--) {
      indices[j+1] = indices[j];
    }
    indices[j+1] = index;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
static PetscErrorCode NelderMeadReplace(TAO_NelderMead *nm, PetscInt index, Vec Xmu, PetscReal f)
{
  PetscFunctionBegin;
  /*  Add new vector's fraction of average */
  CHKERRQ(VecAXPY(nm->Xbar,nm->oneOverN,Xmu));
  CHKERRQ(VecCopy(Xmu,nm->simplex[index]));
  nm->f_values[index] = f;

  CHKERRQ(NelderMeadSort(nm));

  /*  Subtract last vector from average */
  CHKERRQ(VecAXPY(nm->Xbar,-nm->oneOverN,nm->simplex[nm->indices[nm->N]]));
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------- */
static PetscErrorCode TaoSetUp_NM(Tao tao)
{
  TAO_NelderMead *nm = (TAO_NelderMead *)tao->data;
  PetscInt       n;

  PetscFunctionBegin;
  CHKERRQ(VecGetSize(tao->solution,&n));
  nm->N = n;
  nm->oneOverN = 1.0/n;
  CHKERRQ(VecDuplicateVecs(tao->solution,nm->N+1,&nm->simplex));
  CHKERRQ(PetscMalloc1(nm->N+1,&nm->f_values));
  CHKERRQ(PetscMalloc1(nm->N+1,&nm->indices));
  CHKERRQ(VecDuplicate(tao->solution,&nm->Xbar));
  CHKERRQ(VecDuplicate(tao->solution,&nm->Xmur));
  CHKERRQ(VecDuplicate(tao->solution,&nm->Xmue));
  CHKERRQ(VecDuplicate(tao->solution,&nm->Xmuc));

  tao->gradient=NULL;
  tao->step=0;
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------- */
static PetscErrorCode TaoDestroy_NM(Tao tao)
{
  TAO_NelderMead *nm = (TAO_NelderMead*)tao->data;

  PetscFunctionBegin;
  if (tao->setupcalled) {
    CHKERRQ(VecDestroyVecs(nm->N+1,&nm->simplex));
    CHKERRQ(VecDestroy(&nm->Xmuc));
    CHKERRQ(VecDestroy(&nm->Xmue));
    CHKERRQ(VecDestroy(&nm->Xmur));
    CHKERRQ(VecDestroy(&nm->Xbar));
  }
  CHKERRQ(PetscFree(nm->indices));
  CHKERRQ(PetscFree(nm->f_values));
  CHKERRQ(PetscFree(tao->data));
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
static PetscErrorCode TaoSetFromOptions_NM(PetscOptionItems *PetscOptionsObject,Tao tao)
{
  TAO_NelderMead *nm = (TAO_NelderMead*)tao->data;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"Nelder-Mead options"));
  CHKERRQ(PetscOptionsReal("-tao_nm_lamda","initial step length","",nm->lamda,&nm->lamda,NULL));
  CHKERRQ(PetscOptionsReal("-tao_nm_mu","mu","",nm->mu_oc,&nm->mu_oc,NULL));
  nm->mu_ic = -nm->mu_oc;
  nm->mu_r = nm->mu_oc*2.0;
  nm->mu_e = nm->mu_oc*4.0;
  CHKERRQ(PetscOptionsTail());
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
static PetscErrorCode TaoView_NM(Tao tao,PetscViewer viewer)
{
  TAO_NelderMead *nm = (TAO_NelderMead*)tao->data;
  PetscBool      isascii;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  if (isascii) {
    CHKERRQ(PetscViewerASCIIPushTab(viewer));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"expansions: %D\n",nm->nexpand));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"reflections: %D\n",nm->nreflect));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"inside contractions: %D\n",nm->nincontract));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"outside contractionss: %D\n",nm->noutcontract));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"Shrink steps: %D\n",nm->nshrink));
    CHKERRQ(PetscViewerASCIIPopTab(viewer));
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
static PetscErrorCode TaoSolve_NM(Tao tao)
{
  TAO_NelderMead     *nm = (TAO_NelderMead*)tao->data;
  PetscReal          *x;
  PetscInt           i;
  Vec                Xmur=nm->Xmur, Xmue=nm->Xmue, Xmuc=nm->Xmuc, Xbar=nm->Xbar;
  PetscReal          fr,fe,fc;
  PetscInt           shrink;
  PetscInt           low,high;

  PetscFunctionBegin;
  nm->nshrink =      0;
  nm->nreflect =     0;
  nm->nincontract =  0;
  nm->noutcontract = 0;
  nm->nexpand =      0;

  if (tao->XL || tao->XU || tao->ops->computebounds) {
    CHKERRQ(PetscInfo(tao,"WARNING: Variable bounds have been set but will be ignored by NelderMead algorithm\n"));
  }

  CHKERRQ(VecCopy(tao->solution,nm->simplex[0]));
  CHKERRQ(TaoComputeObjective(tao,nm->simplex[0],&nm->f_values[0]));
  nm->indices[0]=0;
  for (i=1;i<nm->N+1;i++) {
    CHKERRQ(VecCopy(tao->solution,nm->simplex[i]));
    CHKERRQ(VecGetOwnershipRange(nm->simplex[i],&low,&high));
    if (i-1 >= low && i-1 < high) {
      CHKERRQ(VecGetArray(nm->simplex[i],&x));
      x[i-1-low] += nm->lamda;
      CHKERRQ(VecRestoreArray(nm->simplex[i],&x));
    }

    CHKERRQ(TaoComputeObjective(tao,nm->simplex[i],&nm->f_values[i]));
    nm->indices[i] = i;
  }

  /*  Xbar  = (Sum of all simplex vectors - worst vector)/N */
  CHKERRQ(NelderMeadSort(nm));
  CHKERRQ(VecSet(Xbar,0.0));
  for (i=0;i<nm->N;i++) {
    CHKERRQ(VecAXPY(Xbar,1.0,nm->simplex[nm->indices[i]]));
  }
  CHKERRQ(VecScale(Xbar,nm->oneOverN));
  tao->reason = TAO_CONTINUE_ITERATING;
  while (1) {
    /* Call general purpose update function */
    if (tao->ops->update) {
      CHKERRQ((*tao->ops->update)(tao, tao->niter, tao->user_update));
    }
    ++tao->niter;
    shrink = 0;
    CHKERRQ(VecCopy(nm->simplex[nm->indices[0]],tao->solution));
    CHKERRQ(TaoLogConvergenceHistory(tao, nm->f_values[nm->indices[0]], nm->f_values[nm->indices[nm->N]]-nm->f_values[nm->indices[0]], 0.0, tao->ksp_its));
    CHKERRQ(TaoMonitor(tao,tao->niter, nm->f_values[nm->indices[0]], nm->f_values[nm->indices[nm->N]]-nm->f_values[nm->indices[0]], 0.0, 1.0));
    CHKERRQ((*tao->ops->convergencetest)(tao,tao->cnvP));
    if (tao->reason != TAO_CONTINUE_ITERATING) break;

    /* x(mu) = (1 + mu)Xbar - mu*X_N+1 */
    CHKERRQ(VecAXPBYPCZ(Xmur,1+nm->mu_r,-nm->mu_r,0,Xbar,nm->simplex[nm->indices[nm->N]]));
    CHKERRQ(TaoComputeObjective(tao,Xmur,&fr));

    if (nm->f_values[nm->indices[0]] <= fr && fr < nm->f_values[nm->indices[nm->N-1]]) {
      /*  reflect */
      nm->nreflect++;
      CHKERRQ(PetscInfo(0,"Reflect\n"));
      CHKERRQ(NelderMeadReplace(nm,nm->indices[nm->N],Xmur,fr));
    } else if (fr < nm->f_values[nm->indices[0]]) {
      /*  expand */
      nm->nexpand++;
      CHKERRQ(PetscInfo(0,"Expand\n"));
      CHKERRQ(VecAXPBYPCZ(Xmue,1+nm->mu_e,-nm->mu_e,0,Xbar,nm->simplex[nm->indices[nm->N]]));
      CHKERRQ(TaoComputeObjective(tao,Xmue,&fe));
      if (fe < fr) {
        CHKERRQ(NelderMeadReplace(nm,nm->indices[nm->N],Xmue,fe));
      } else {
        CHKERRQ(NelderMeadReplace(nm,nm->indices[nm->N],Xmur,fr));
      }
    } else if (nm->f_values[nm->indices[nm->N-1]] <= fr && fr < nm->f_values[nm->indices[nm->N]]) {
      /* outside contraction */
      nm->noutcontract++;
      CHKERRQ(PetscInfo(0,"Outside Contraction\n"));
      CHKERRQ(VecAXPBYPCZ(Xmuc,1+nm->mu_oc,-nm->mu_oc,0,Xbar,nm->simplex[nm->indices[nm->N]]));

      CHKERRQ(TaoComputeObjective(tao,Xmuc,&fc));
      if (fc <= fr) {
        CHKERRQ(NelderMeadReplace(nm,nm->indices[nm->N],Xmuc,fc));
      } else shrink=1;
    } else {
      /* inside contraction */
      nm->nincontract++;
      CHKERRQ(PetscInfo(0,"Inside Contraction\n"));
      CHKERRQ(VecAXPBYPCZ(Xmuc,1+nm->mu_ic,-nm->mu_ic,0,Xbar,nm->simplex[nm->indices[nm->N]]));
      CHKERRQ(TaoComputeObjective(tao,Xmuc,&fc));
      if (fc < nm->f_values[nm->indices[nm->N]]) {
        CHKERRQ(NelderMeadReplace(nm,nm->indices[nm->N],Xmuc,fc));
      } else shrink = 1;
    }

    if (shrink) {
      nm->nshrink++;
      CHKERRQ(PetscInfo(0,"Shrink\n"));

      for (i=1;i<nm->N+1;i++) {
        CHKERRQ(VecAXPBY(nm->simplex[nm->indices[i]],1.5,-0.5,nm->simplex[nm->indices[0]]));
        CHKERRQ(TaoComputeObjective(tao,nm->simplex[nm->indices[i]], &nm->f_values[nm->indices[i]]));
      }
      CHKERRQ(VecAXPBY(Xbar,1.5*nm->oneOverN,-0.5,nm->simplex[nm->indices[0]]));

      /*  Add last vector's fraction of average */
      CHKERRQ(VecAXPY(Xbar,nm->oneOverN,nm->simplex[nm->indices[nm->N]]));
      CHKERRQ(NelderMeadSort(nm));
      /*  Subtract new last vector from average */
      CHKERRQ(VecAXPY(Xbar,-nm->oneOverN,nm->simplex[nm->indices[nm->N]]));
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
  CHKERRQ(PetscNewLog(tao,&nm));
  tao->data = (void*)nm;

  tao->ops->setup = TaoSetUp_NM;
  tao->ops->solve = TaoSolve_NM;
  tao->ops->view = TaoView_NM;
  tao->ops->setfromoptions = TaoSetFromOptions_NM;
  tao->ops->destroy = TaoDestroy_NM;

  /* Override default settings (unless already changed) */
  if (!tao->max_it_changed) tao->max_it = 2000;
  if (!tao->max_funcs_changed) tao->max_funcs = 4000;

  nm->simplex = NULL;
  nm->lamda = 1;

  nm->mu_ic = -0.5;
  nm->mu_oc = 0.5;
  nm->mu_r = 1.0;
  nm->mu_e = 2.0;

  PetscFunctionReturn(0);
}
