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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*  Add new vector's fraction of average */
  ierr = VecAXPY(nm->Xbar,nm->oneOverN,Xmu);CHKERRQ(ierr);
  ierr = VecCopy(Xmu,nm->simplex[index]);CHKERRQ(ierr);
  nm->f_values[index] = f;

  ierr = NelderMeadSort(nm);CHKERRQ(ierr);

  /*  Subtract last vector from average */
  ierr = VecAXPY(nm->Xbar,-nm->oneOverN,nm->simplex[nm->indices[nm->N]]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------- */
static PetscErrorCode TaoSetUp_NM(Tao tao)
{
  PetscErrorCode ierr;
  TAO_NelderMead *nm = (TAO_NelderMead *)tao->data;
  PetscInt       n;

  PetscFunctionBegin;
  ierr = VecGetSize(tao->solution,&n);CHKERRQ(ierr);
  nm->N = n;
  nm->oneOverN = 1.0/n;
  ierr = VecDuplicateVecs(tao->solution,nm->N+1,&nm->simplex);CHKERRQ(ierr);
  ierr = PetscMalloc1(nm->N+1,&nm->f_values);CHKERRQ(ierr);
  ierr = PetscMalloc1(nm->N+1,&nm->indices);CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution,&nm->Xbar);CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution,&nm->Xmur);CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution,&nm->Xmue);CHKERRQ(ierr);
  ierr = VecDuplicate(tao->solution,&nm->Xmuc);CHKERRQ(ierr);

  tao->gradient=NULL;
  tao->step=0;
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------- */
static PetscErrorCode TaoDestroy_NM(Tao tao)
{
  TAO_NelderMead *nm = (TAO_NelderMead*)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (tao->setupcalled) {
    ierr = VecDestroyVecs(nm->N+1,&nm->simplex);CHKERRQ(ierr);
    ierr = VecDestroy(&nm->Xmuc);CHKERRQ(ierr);
    ierr = VecDestroy(&nm->Xmue);CHKERRQ(ierr);
    ierr = VecDestroy(&nm->Xmur);CHKERRQ(ierr);
    ierr = VecDestroy(&nm->Xbar);CHKERRQ(ierr);
  }
  ierr = PetscFree(nm->indices);CHKERRQ(ierr);
  ierr = PetscFree(nm->f_values);CHKERRQ(ierr);
  ierr = PetscFree(tao->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
static PetscErrorCode TaoSetFromOptions_NM(PetscOptionItems *PetscOptionsObject,Tao tao)
{
  TAO_NelderMead *nm = (TAO_NelderMead*)tao->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"Nelder-Mead options");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nm_lamda","initial step length","",nm->lamda,&nm->lamda,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-tao_nm_mu","mu","",nm->mu_oc,&nm->mu_oc,NULL);CHKERRQ(ierr);
  nm->mu_ic = -nm->mu_oc;
  nm->mu_r = nm->mu_oc*2.0;
  nm->mu_e = nm->mu_oc*4.0;
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
static PetscErrorCode TaoView_NM(Tao tao,PetscViewer viewer)
{
  TAO_NelderMead *nm = (TAO_NelderMead*)tao->data;
  PetscBool      isascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"expansions: %D\n",nm->nexpand);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"reflections: %D\n",nm->nreflect);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"inside contractions: %D\n",nm->nincontract);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"outside contractionss: %D\n",nm->noutcontract);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Shrink steps: %D\n",nm->nshrink);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
static PetscErrorCode TaoSolve_NM(Tao tao)
{
  PetscErrorCode     ierr;
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
    ierr = PetscInfo(tao,"WARNING: Variable bounds have been set but will be ignored by NelderMead algorithm\n");CHKERRQ(ierr);
  }

  ierr = VecCopy(tao->solution,nm->simplex[0]);CHKERRQ(ierr);
  ierr = TaoComputeObjective(tao,nm->simplex[0],&nm->f_values[0]);CHKERRQ(ierr);
  nm->indices[0]=0;
  for (i=1;i<nm->N+1;i++) {
    ierr = VecCopy(tao->solution,nm->simplex[i]);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(nm->simplex[i],&low,&high);CHKERRQ(ierr);
    if (i-1 >= low && i-1 < high) {
      ierr = VecGetArray(nm->simplex[i],&x);CHKERRQ(ierr);
      x[i-1-low] += nm->lamda;
      ierr = VecRestoreArray(nm->simplex[i],&x);CHKERRQ(ierr);
    }

    ierr = TaoComputeObjective(tao,nm->simplex[i],&nm->f_values[i]);CHKERRQ(ierr);
    nm->indices[i] = i;
  }

  /*  Xbar  = (Sum of all simplex vectors - worst vector)/N */
  ierr = NelderMeadSort(nm);CHKERRQ(ierr);
  ierr = VecSet(Xbar,0.0);CHKERRQ(ierr);
  for (i=0;i<nm->N;i++) {
    ierr = VecAXPY(Xbar,1.0,nm->simplex[nm->indices[i]]);CHKERRQ(ierr);
  }
  ierr = VecScale(Xbar,nm->oneOverN);CHKERRQ(ierr);
  tao->reason = TAO_CONTINUE_ITERATING;
  while (1) {
    /* Call general purpose update function */
    if (tao->ops->update) {
      ierr = (*tao->ops->update)(tao, tao->niter, tao->user_update);CHKERRQ(ierr);
    }
    ++tao->niter;
    shrink = 0;
    ierr = VecCopy(nm->simplex[nm->indices[0]],tao->solution);CHKERRQ(ierr);
    ierr = TaoLogConvergenceHistory(tao, nm->f_values[nm->indices[0]], nm->f_values[nm->indices[nm->N]]-nm->f_values[nm->indices[0]], 0.0, tao->ksp_its);CHKERRQ(ierr);
    ierr = TaoMonitor(tao,tao->niter, nm->f_values[nm->indices[0]], nm->f_values[nm->indices[nm->N]]-nm->f_values[nm->indices[0]], 0.0, 1.0);CHKERRQ(ierr);
    ierr = (*tao->ops->convergencetest)(tao,tao->cnvP);CHKERRQ(ierr);
    if (tao->reason != TAO_CONTINUE_ITERATING) break;

    /* x(mu) = (1 + mu)Xbar - mu*X_N+1 */
    ierr = VecAXPBYPCZ(Xmur,1+nm->mu_r,-nm->mu_r,0,Xbar,nm->simplex[nm->indices[nm->N]]);CHKERRQ(ierr);
    ierr = TaoComputeObjective(tao,Xmur,&fr);CHKERRQ(ierr);

    if (nm->f_values[nm->indices[0]] <= fr && fr < nm->f_values[nm->indices[nm->N-1]]) {
      /*  reflect */
      nm->nreflect++;
      ierr = PetscInfo(0,"Reflect\n");CHKERRQ(ierr);
      ierr = NelderMeadReplace(nm,nm->indices[nm->N],Xmur,fr);CHKERRQ(ierr);
    } else if (fr < nm->f_values[nm->indices[0]]) {
      /*  expand */
      nm->nexpand++;
      ierr = PetscInfo(0,"Expand\n");CHKERRQ(ierr);
      ierr = VecAXPBYPCZ(Xmue,1+nm->mu_e,-nm->mu_e,0,Xbar,nm->simplex[nm->indices[nm->N]]);CHKERRQ(ierr);
      ierr = TaoComputeObjective(tao,Xmue,&fe);CHKERRQ(ierr);
      if (fe < fr) {
        ierr = NelderMeadReplace(nm,nm->indices[nm->N],Xmue,fe);CHKERRQ(ierr);
      } else {
        ierr = NelderMeadReplace(nm,nm->indices[nm->N],Xmur,fr);CHKERRQ(ierr);
      }
    } else if (nm->f_values[nm->indices[nm->N-1]] <= fr && fr < nm->f_values[nm->indices[nm->N]]) {
      /* outside contraction */
      nm->noutcontract++;
      ierr = PetscInfo(0,"Outside Contraction\n");CHKERRQ(ierr);
      ierr = VecAXPBYPCZ(Xmuc,1+nm->mu_oc,-nm->mu_oc,0,Xbar,nm->simplex[nm->indices[nm->N]]);CHKERRQ(ierr);

      ierr = TaoComputeObjective(tao,Xmuc,&fc);CHKERRQ(ierr);
      if (fc <= fr) {
        ierr = NelderMeadReplace(nm,nm->indices[nm->N],Xmuc,fc);CHKERRQ(ierr);
      } else shrink=1;
    } else {
      /* inside contraction */
      nm->nincontract++;
      ierr = PetscInfo(0,"Inside Contraction\n");CHKERRQ(ierr);
      ierr = VecAXPBYPCZ(Xmuc,1+nm->mu_ic,-nm->mu_ic,0,Xbar,nm->simplex[nm->indices[nm->N]]);CHKERRQ(ierr);
      ierr = TaoComputeObjective(tao,Xmuc,&fc);CHKERRQ(ierr);
      if (fc < nm->f_values[nm->indices[nm->N]]) {
        ierr = NelderMeadReplace(nm,nm->indices[nm->N],Xmuc,fc);CHKERRQ(ierr);
      } else shrink = 1;
    }

    if (shrink) {
      nm->nshrink++;
      ierr = PetscInfo(0,"Shrink\n");CHKERRQ(ierr);

      for (i=1;i<nm->N+1;i++) {
        ierr = VecAXPBY(nm->simplex[nm->indices[i]],1.5,-0.5,nm->simplex[nm->indices[0]]);CHKERRQ(ierr);
        ierr = TaoComputeObjective(tao,nm->simplex[nm->indices[i]], &nm->f_values[nm->indices[i]]);CHKERRQ(ierr);
      }
      ierr = VecAXPBY(Xbar,1.5*nm->oneOverN,-0.5,nm->simplex[nm->indices[0]]);CHKERRQ(ierr);

      /*  Add last vector's fraction of average */
      ierr = VecAXPY(Xbar,nm->oneOverN,nm->simplex[nm->indices[nm->N]]);CHKERRQ(ierr);
      ierr = NelderMeadSort(nm);CHKERRQ(ierr);
      /*  Subtract new last vector from average */
      ierr = VecAXPY(Xbar,-nm->oneOverN,nm->simplex[nm->indices[nm->N]]);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(tao,&nm);CHKERRQ(ierr);
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

