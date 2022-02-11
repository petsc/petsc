/*
  Code for time stepping with the multi-rate Runge-Kutta method

  Notes:
  1) The general system is written as
     Udot = F(t,U) for the nonsplit version of multi-rate RK,
     user should give the indexes for both slow and fast components;
  2) The general system is written as
     Usdot = Fs(t,Us,Uf)
     Ufdot = Ff(t,Us,Uf) for multi-rate RK with RHS splits,
     user should partioned RHS by themselves and also provide the indexes for both slow and fast components.
*/

#include <petsc/private/tsimpl.h>
#include <petscdm.h>
#include <../src/ts/impls/explicit/rk/rk.h>
#include <../src/ts/impls/explicit/rk/mrk.h>

static PetscErrorCode TSReset_RK_MultirateNonsplit(TS ts)
{
  TS_RK          *rk = (TS_RK*)ts->data;
  RKTableau      tab = rk->tableau;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&rk->X0);CHKERRQ(ierr);
  ierr = VecDestroyVecs(tab->s,&rk->YdotRHS_slow);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSInterpolate_RK_MultirateNonsplit(TS ts,PetscReal itime,Vec X)
{
  TS_RK            *rk = (TS_RK*)ts->data;
  PetscInt         s = rk->tableau->s,p = rk->tableau->p,i,j;
  PetscReal        h = ts->time_step;
  PetscReal        tt,t;
  PetscScalar      *b;
  const PetscReal  *B = rk->tableau->binterp;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscCheckFalse(!B,PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"TSRK %s does not have an interpolation formula",rk->tableau->name);
  t = (itime - rk->ptime)/h;
  ierr = PetscMalloc1(s,&b);CHKERRQ(ierr);
  for (i=0; i<s; i++) b[i] = 0;
  for (j=0,tt=t; j<p; j++,tt*=t) {
    for (i=0; i<s; i++) {
      b[i]  += h * B[i*p+j] * tt;
    }
  }
  ierr = VecCopy(rk->X0,X);CHKERRQ(ierr);
  ierr = VecMAXPY(X,s,b,rk->YdotRHS_slow);CHKERRQ(ierr);
  ierr = PetscFree(b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSStepRefine_RK_MultirateNonsplit(TS ts)
{
  TS              previousts,subts;
  TS_RK           *rk = (TS_RK*)ts->data;
  RKTableau       tab = rk->tableau;
  Vec             *Y = rk->Y,*YdotRHS = rk->YdotRHS;
  Vec             vec_fast,subvec_fast;
  const PetscInt  s = tab->s;
  const PetscReal *A = tab->A,*c = tab->c;
  PetscScalar     *w = rk->work;
  PetscInt        i,j,k;
  PetscReal       t = ts->ptime,h = ts->time_step;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = VecDuplicate(ts->vec_sol,&vec_fast);CHKERRQ(ierr);
  previousts = rk->subts_current;
  ierr = TSRHSSplitGetSubTS(rk->subts_current,"fast",&subts);CHKERRQ(ierr);
  ierr = TSRHSSplitGetSubTS(subts,"fast",&subts);CHKERRQ(ierr);
  for (k=0; k<rk->dtratio; k++) {
    for (i=0; i<s; i++) {
      ierr = TSInterpolate_RK_MultirateNonsplit(ts,t+k*h/rk->dtratio+h/rk->dtratio*c[i],Y[i]);CHKERRQ(ierr);
      for (j=0; j<i; j++) w[j] = h/rk->dtratio*A[i*s+j];
      /* update the fast components in the stage value, the slow components will be ignored, so we do not care the slow part in vec_fast */
      ierr = VecCopy(ts->vec_sol,vec_fast);CHKERRQ(ierr);
      ierr = VecMAXPY(vec_fast,i,w,YdotRHS);CHKERRQ(ierr);
      /* update the fast components in the stage value */
      ierr = VecGetSubVector(vec_fast,rk->is_fast,&subvec_fast);CHKERRQ(ierr);
      ierr = VecISCopy(Y[i],rk->is_fast,SCATTER_FORWARD,subvec_fast);CHKERRQ(ierr);
      ierr = VecRestoreSubVector(vec_fast,rk->is_fast,&subvec_fast);CHKERRQ(ierr);
      /* compute the stage RHS */
      ierr = TSComputeRHSFunction(ts,t+k*h/rk->dtratio+h/rk->dtratio*c[i],Y[i],YdotRHS[i]);CHKERRQ(ierr);
    }
    ierr = VecCopy(ts->vec_sol,vec_fast);CHKERRQ(ierr);
    ierr = TSEvaluateStep(ts,tab->order,vec_fast,NULL);CHKERRQ(ierr);
    /* update the fast components in the solution */
    ierr = VecGetSubVector(vec_fast,rk->is_fast,&subvec_fast);CHKERRQ(ierr);
    ierr = VecISCopy(ts->vec_sol,rk->is_fast,SCATTER_FORWARD,subvec_fast);CHKERRQ(ierr);
    ierr = VecRestoreSubVector(vec_fast,rk->is_fast,&subvec_fast);CHKERRQ(ierr);

    if (subts) {
      Vec *YdotRHS_copy;
      ierr = VecDuplicateVecs(ts->vec_sol,s,&YdotRHS_copy);CHKERRQ(ierr);
      rk->subts_current = rk->subts_fast;
      ts->ptime = t+k*h/rk->dtratio;
      ts->time_step = h/rk->dtratio;
      ierr = TSRHSSplitGetIS(rk->subts_current,"fast",&rk->is_fast);CHKERRQ(ierr);
      for (i=0; i<s; i++) {
        ierr = VecCopy(rk->YdotRHS_slow[i],YdotRHS_copy[i]);CHKERRQ(ierr);
        ierr = VecCopy(YdotRHS[i],rk->YdotRHS_slow[i]);CHKERRQ(ierr);
      }

      ierr = TSStepRefine_RK_MultirateNonsplit(ts);CHKERRQ(ierr);

      rk->subts_current = previousts;
      ts->ptime = t;
      ts->time_step = h;
      ierr = TSRHSSplitGetIS(previousts,"fast",&rk->is_fast);CHKERRQ(ierr);
      for (i=0; i<s; i++) {
        ierr = VecCopy(YdotRHS_copy[i],rk->YdotRHS_slow[i]);CHKERRQ(ierr);
      }
      ierr = VecDestroyVecs(s,&YdotRHS_copy);CHKERRQ(ierr);
    }
  }
  ierr = VecDestroy(&vec_fast);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSStep_RK_MultirateNonsplit(TS ts)
{
  TS_RK           *rk = (TS_RK*)ts->data;
  RKTableau       tab = rk->tableau;
  Vec             *Y = rk->Y,*YdotRHS = rk->YdotRHS,*YdotRHS_slow = rk->YdotRHS_slow;
  Vec             stage_slow,sol_slow; /* vectors store the slow components */
  Vec             subvec_slow; /* sub vector to store the slow components */
  IS              is_slow = rk->is_slow;
  const PetscInt  s = tab->s;
  const PetscReal *A = tab->A,*c = tab->c;
  PetscScalar     *w = rk->work;
  PetscInt        i,j,dtratio = rk->dtratio;
  PetscReal       next_time_step = ts->time_step,t = ts->ptime,h = ts->time_step;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  rk->status = TS_STEP_INCOMPLETE;
  ierr = VecDuplicate(ts->vec_sol,&stage_slow);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&sol_slow);CHKERRQ(ierr);
  ierr = VecCopy(ts->vec_sol,rk->X0);CHKERRQ(ierr);
  for (i=0; i<s; i++) {
    rk->stage_time = t + h*c[i];
    ierr = TSPreStage(ts,rk->stage_time);CHKERRQ(ierr);
    ierr = VecCopy(ts->vec_sol,Y[i]);CHKERRQ(ierr);
    for (j=0; j<i; j++) w[j] = h*A[i*s+j];
    ierr = VecMAXPY(Y[i],i,w,YdotRHS_slow);CHKERRQ(ierr);
    ierr = TSPostStage(ts,rk->stage_time,i,Y);CHKERRQ(ierr);
    /* compute the stage RHS */
    ierr = TSComputeRHSFunction(ts,t+h*c[i],Y[i],YdotRHS_slow[i]);CHKERRQ(ierr);
  }
  /* update the slow components in the solution */
  rk->YdotRHS = YdotRHS_slow;
  rk->dtratio = 1;
  ierr = TSEvaluateStep(ts,tab->order,sol_slow,NULL);CHKERRQ(ierr);
  rk->dtratio = dtratio;
  rk->YdotRHS = YdotRHS;
  /* update the slow components in the solution */
  ierr = VecGetSubVector(sol_slow,is_slow,&subvec_slow);CHKERRQ(ierr);
  ierr = VecISCopy(ts->vec_sol,is_slow,SCATTER_FORWARD,subvec_slow);CHKERRQ(ierr);
  ierr = VecRestoreSubVector(sol_slow,is_slow,&subvec_slow);CHKERRQ(ierr);

  rk->subts_current = ts;
  rk->ptime = t;
  rk->time_step = h;
  ierr = TSStepRefine_RK_MultirateNonsplit(ts);CHKERRQ(ierr);

  ts->ptime = t + ts->time_step;
  ts->time_step = next_time_step;
  rk->status = TS_STEP_COMPLETE;

  /* free memory of work vectors */
  ierr = VecDestroy(&stage_slow);CHKERRQ(ierr);
  ierr = VecDestroy(&sol_slow);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSSetUp_RK_MultirateNonsplit(TS ts)
{
  TS_RK          *rk = (TS_RK*)ts->data;
  RKTableau      tab = rk->tableau;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSRHSSplitGetIS(ts,"slow",&rk->is_slow);CHKERRQ(ierr);
  ierr = TSRHSSplitGetIS(ts,"fast",&rk->is_fast);CHKERRQ(ierr);
  PetscCheckFalse(!rk->is_slow || !rk->is_fast,PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"Must set up RHSSplits with TSRHSSplitSetIS() using split names 'slow' and 'fast' respectively in order to use multirate RK");
  ierr = TSRHSSplitGetSubTS(ts,"slow",&rk->subts_slow);CHKERRQ(ierr);
  ierr = TSRHSSplitGetSubTS(ts,"fast",&rk->subts_fast);CHKERRQ(ierr);
  PetscCheckFalse(!rk->subts_slow || !rk->subts_fast,PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"Must set up the RHSFunctions for 'slow' and 'fast' components using TSRHSSplitSetRHSFunction() or calling TSSetRHSFunction() for each sub-TS");
  ierr = VecDuplicate(ts->vec_sol,&rk->X0);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(ts->vec_sol,tab->s,&rk->YdotRHS_slow);CHKERRQ(ierr);
  rk->subts_current = rk->subts_fast;

  ts->ops->step        = TSStep_RK_MultirateNonsplit;
  ts->ops->interpolate = TSInterpolate_RK_MultirateNonsplit;
  PetscFunctionReturn(0);
}

/*
  Copy DM from tssrc to tsdest, while keeping the original DMTS and DMSNES in tsdest.
*/
static PetscErrorCode TSCopyDM(TS tssrc,TS tsdest)
{
  DM             newdm,dmsrc,dmdest;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetDM(tssrc,&dmsrc);CHKERRQ(ierr);
  ierr = DMClone(dmsrc,&newdm);CHKERRQ(ierr);
  ierr = TSGetDM(tsdest,&dmdest);CHKERRQ(ierr);
  ierr = DMCopyDMTS(dmdest,newdm);CHKERRQ(ierr);
  ierr = DMCopyDMSNES(dmdest,newdm);CHKERRQ(ierr);
  ierr = TSSetDM(tsdest,newdm);CHKERRQ(ierr);
  ierr = DMDestroy(&newdm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSReset_RK_MultirateSplit(TS ts)
{
  TS_RK          *rk = (TS_RK*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (rk->subts_slow) {
    ierr = PetscFree(rk->subts_slow->data);CHKERRQ(ierr);
    rk->subts_slow = NULL;
  }
  if (rk->subts_fast) {
    ierr = PetscFree(rk->YdotRHS_fast);CHKERRQ(ierr);
    ierr = PetscFree(rk->YdotRHS_slow);CHKERRQ(ierr);
    ierr = VecDestroy(&rk->X0);CHKERRQ(ierr);
    ierr = TSReset_RK_MultirateSplit(rk->subts_fast);CHKERRQ(ierr);
    ierr = PetscFree(rk->subts_fast->data);CHKERRQ(ierr);
    rk->subts_fast = NULL;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSInterpolate_RK_MultirateSplit(TS ts,PetscReal itime,Vec X)
{
  TS_RK           *rk = (TS_RK*)ts->data;
  Vec             Xslow;
  PetscInt        s = rk->tableau->s,p = rk->tableau->p,i,j;
  PetscReal       h;
  PetscReal       tt,t;
  PetscScalar     *b;
  const PetscReal *B = rk->tableau->binterp;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscCheckFalse(!B,PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"TSRK %s does not have an interpolation formula",rk->tableau->name);

  switch (rk->status) {
    case TS_STEP_INCOMPLETE:
    case TS_STEP_PENDING:
      h = ts->time_step;
      t = (itime - ts->ptime)/h;
      break;
    case TS_STEP_COMPLETE:
      h = ts->ptime - ts->ptime_prev;
      t = (itime - ts->ptime)/h + 1; /* In the interval [0,1] */
      break;
    default: SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_PLIB,"Invalid TSStepStatus");
  }
  ierr = PetscMalloc1(s,&b);CHKERRQ(ierr);
  for (i=0; i<s; i++) b[i] = 0;
  for (j=0,tt=t; j<p; j++,tt*=t) {
    for (i=0; i<s; i++) {
      b[i]  += h * B[i*p+j] * tt;
    }
  }
  for (i=0; i<s; i++) {
    ierr = VecGetSubVector(rk->YdotRHS[i],rk->is_slow,&rk->YdotRHS_slow[i]);CHKERRQ(ierr);
  }
  ierr = VecGetSubVector(X,rk->is_slow,&Xslow);CHKERRQ(ierr);
  ierr = VecISCopy(rk->X0,rk->is_slow,SCATTER_REVERSE,Xslow);CHKERRQ(ierr);
  ierr = VecMAXPY(Xslow,s,b,rk->YdotRHS_slow);CHKERRQ(ierr);
  ierr = VecRestoreSubVector(X,rk->is_slow,&Xslow);CHKERRQ(ierr);
  for (i=0; i<s; i++) {
    ierr = VecRestoreSubVector(rk->YdotRHS[i],rk->is_slow,&rk->YdotRHS_slow[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
 This is for partitioned RHS multirate RK method
 The step completion formula is

 x1 = x0 + h b^T YdotRHS

*/
static PetscErrorCode TSEvaluateStep_RK_MultirateSplit(TS ts,PetscInt order,Vec X,PetscBool *done)
{
  TS_RK          *rk = (TS_RK*)ts->data;
  RKTableau      tab = rk->tableau;
  Vec            Xslow,Xfast;                  /* subvectors of X which store slow components and fast components respectively */
  PetscScalar    *w = rk->work;
  PetscReal      h = ts->time_step;
  PetscInt       s = tab->s,j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCopy(ts->vec_sol,X);CHKERRQ(ierr);
  if (rk->slow) {
    for (j=0; j<s; j++) w[j] = h*tab->b[j];
    ierr = VecGetSubVector(ts->vec_sol,rk->is_slow,&Xslow);CHKERRQ(ierr);
    ierr = VecMAXPY(Xslow,s,w,rk->YdotRHS_slow);CHKERRQ(ierr);
    ierr = VecRestoreSubVector(ts->vec_sol,rk->is_slow,&Xslow);CHKERRQ(ierr);
  } else {
    for (j=0; j<s; j++) w[j] = h/rk->dtratio*tab->b[j];
    ierr = VecGetSubVector(X,rk->is_fast,&Xfast);CHKERRQ(ierr);
    ierr = VecMAXPY(Xfast,s,w,rk->YdotRHS_fast);CHKERRQ(ierr);
    ierr = VecRestoreSubVector(X,rk->is_fast,&Xfast);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSStepRefine_RK_MultirateSplit(TS ts)
{
  TS_RK           *rk = (TS_RK*)ts->data;
  TS              subts_fast = rk->subts_fast,currentlevelts;
  TS_RK           *subrk_fast = (TS_RK*)subts_fast->data;
  RKTableau       tab = rk->tableau;
  Vec             *Y = rk->Y;
  Vec             *YdotRHS = rk->YdotRHS,*YdotRHS_fast = rk->YdotRHS_fast;
  Vec             Yfast,Xfast;
  const PetscInt  s = tab->s;
  const PetscReal *A = tab->A,*c = tab->c;
  PetscScalar     *w = rk->work;
  PetscInt        i,j,k;
  PetscReal       t = ts->ptime,h = ts->time_step;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  for (k=0; k<rk->dtratio; k++) {
    ierr = VecGetSubVector(ts->vec_sol,rk->is_fast,&Xfast);CHKERRQ(ierr);
    for (i=0; i<s; i++) {
      ierr = VecGetSubVector(YdotRHS[i],rk->is_fast,&YdotRHS_fast[i]);CHKERRQ(ierr);
    }
    /* propagate fast component using small time steps */
    for (i=0; i<s; i++) {
      /* stage value for slow components */
      ierr = TSInterpolate_RK_MultirateSplit(rk->ts_root,t+k*h/rk->dtratio+h/rk->dtratio*c[i],Y[i]);CHKERRQ(ierr);
      currentlevelts = rk->ts_root;
      while (currentlevelts != ts) { /* all the slow parts need to be interpolated separated */
        currentlevelts = ((TS_RK*)currentlevelts->data)->subts_fast;
        ierr = TSInterpolate_RK_MultirateSplit(currentlevelts,t+k*h/rk->dtratio+h/rk->dtratio*c[i],Y[i]);CHKERRQ(ierr);
      }
      for (j=0; j<i; j++) w[j] = h/rk->dtratio*A[i*s+j];
      subrk_fast->stage_time = t + h/rk->dtratio*c[i];
      ierr = TSPreStage(subts_fast,subrk_fast->stage_time);CHKERRQ(ierr);
      /* stage value for fast components */
      ierr = VecGetSubVector(Y[i],rk->is_fast,&Yfast);CHKERRQ(ierr);
      ierr = VecCopy(Xfast,Yfast);CHKERRQ(ierr);
      ierr = VecMAXPY(Yfast,i,w,YdotRHS_fast);CHKERRQ(ierr);
      ierr = VecRestoreSubVector(Y[i],rk->is_fast,&Yfast);CHKERRQ(ierr);
      ierr = TSPostStage(subts_fast,subrk_fast->stage_time,i,Y);CHKERRQ(ierr);
      /* compute the stage RHS for fast components */
      ierr = TSComputeRHSFunction(subts_fast,t+k*h*rk->dtratio+h/rk->dtratio*c[i],Y[i],YdotRHS_fast[i]);CHKERRQ(ierr);
    }
    ierr = VecRestoreSubVector(ts->vec_sol,rk->is_fast,&Xfast);CHKERRQ(ierr);
    /* update the value of fast components using fast time step */
    rk->slow = PETSC_FALSE;
    ierr = TSEvaluateStep_RK_MultirateSplit(ts,tab->order,ts->vec_sol,NULL);CHKERRQ(ierr);
    for (i=0; i<s; i++) {
      ierr = VecRestoreSubVector(YdotRHS[i],rk->is_fast,&YdotRHS_fast[i]);CHKERRQ(ierr);
    }

    if (subrk_fast->subts_fast) {
      subts_fast->ptime = t+k*h/rk->dtratio;
      subts_fast->time_step = h/rk->dtratio;
      ierr = TSStepRefine_RK_MultirateSplit(subts_fast);CHKERRQ(ierr);
    }
    /* update the fast components of the solution */
    ierr = VecGetSubVector(ts->vec_sol,rk->is_fast,&Xfast);CHKERRQ(ierr);
    ierr = VecISCopy(rk->X0,rk->is_fast,SCATTER_FORWARD,Xfast);CHKERRQ(ierr);
    ierr = VecRestoreSubVector(ts->vec_sol,rk->is_fast,&Xfast);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSStep_RK_MultirateSplit(TS ts)
{
  TS_RK           *rk = (TS_RK*)ts->data;
  RKTableau       tab = rk->tableau;
  Vec             *Y = rk->Y,*YdotRHS = rk->YdotRHS;
  Vec             *YdotRHS_fast = rk->YdotRHS_fast,*YdotRHS_slow = rk->YdotRHS_slow;
  Vec             Yslow,Yfast; /* subvectors store the stges of slow components and fast components respectively                           */
  const PetscInt  s = tab->s;
  const PetscReal *A = tab->A,*c = tab->c;
  PetscScalar     *w = rk->work;
  PetscInt        i,j;
  PetscReal       next_time_step = ts->time_step,t = ts->ptime,h = ts->time_step;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  rk->status = TS_STEP_INCOMPLETE;
  for (i=0; i<s; i++) {
    ierr = VecGetSubVector(YdotRHS[i],rk->is_slow,&YdotRHS_slow[i]);CHKERRQ(ierr);
    ierr = VecGetSubVector(YdotRHS[i],rk->is_fast,&YdotRHS_fast[i]);CHKERRQ(ierr);
  }
  ierr = VecCopy(ts->vec_sol,rk->X0);CHKERRQ(ierr);
  /* propagate both slow and fast components using large time steps */
  for (i=0; i<s; i++) {
    rk->stage_time = t + h*c[i];
    ierr = TSPreStage(ts,rk->stage_time);CHKERRQ(ierr);
    ierr = VecCopy(ts->vec_sol,Y[i]);CHKERRQ(ierr);
    ierr = VecGetSubVector(Y[i],rk->is_fast,&Yfast);CHKERRQ(ierr);
    ierr = VecGetSubVector(Y[i],rk->is_slow,&Yslow);CHKERRQ(ierr);
    for (j=0; j<i; j++) w[j] = h*A[i*s+j];
    ierr = VecMAXPY(Yfast,i,w,YdotRHS_fast);CHKERRQ(ierr);
    ierr = VecMAXPY(Yslow,i,w,YdotRHS_slow);CHKERRQ(ierr);
    ierr = VecRestoreSubVector(Y[i],rk->is_fast,&Yfast);CHKERRQ(ierr);
    ierr = VecRestoreSubVector(Y[i],rk->is_slow,&Yslow);CHKERRQ(ierr);
    ierr = TSPostStage(ts,rk->stage_time,i,Y);CHKERRQ(ierr);
    ierr = TSComputeRHSFunction(rk->subts_slow,t+h*c[i],Y[i],YdotRHS_slow[i]);CHKERRQ(ierr);
    ierr = TSComputeRHSFunction(rk->subts_fast,t+h*c[i],Y[i],YdotRHS_fast[i]);CHKERRQ(ierr);
  }
  rk->slow = PETSC_TRUE;
  /* update the slow components of the solution using slow time step */
  ierr = TSEvaluateStep_RK_MultirateSplit(ts,tab->order,ts->vec_sol,NULL);CHKERRQ(ierr);
  /* YdotRHS will be used for interpolation during refinement */
  for (i=0; i<s; i++) {
    ierr = VecRestoreSubVector(YdotRHS[i],rk->is_slow,&YdotRHS_slow[i]);CHKERRQ(ierr);
    ierr = VecRestoreSubVector(YdotRHS[i],rk->is_fast,&YdotRHS_fast[i]);CHKERRQ(ierr);
  }

  ierr = TSStepRefine_RK_MultirateSplit(ts);CHKERRQ(ierr);

  ts->ptime = t + ts->time_step;
  ts->time_step = next_time_step;
  rk->status = TS_STEP_COMPLETE;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSSetUp_RK_MultirateSplit(TS ts)
{
  TS_RK          *rk = (TS_RK*)ts->data,*nextlevelrk,*currentlevelrk;
  TS             nextlevelts;
  Vec            X0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSRHSSplitGetIS(ts,"slow",&rk->is_slow);CHKERRQ(ierr);
  ierr = TSRHSSplitGetIS(ts,"fast",&rk->is_fast);CHKERRQ(ierr);
  PetscCheckFalse(!rk->is_slow || !rk->is_fast,PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"Must set up RHSSplits with TSRHSSplitSetIS() using split names 'slow' and 'fast' respectively in order to use -ts_type bsi");
  ierr = TSRHSSplitGetSubTS(ts,"slow",&rk->subts_slow);CHKERRQ(ierr);
  ierr = TSRHSSplitGetSubTS(ts,"fast",&rk->subts_fast);CHKERRQ(ierr);
  PetscCheckFalse(!rk->subts_slow || !rk->subts_fast,PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"Must set up the RHSFunctions for 'slow' and 'fast' components using TSRHSSplitSetRHSFunction() or calling TSSetRHSFunction() for each sub-TS");

  ierr = VecDuplicate(ts->vec_sol,&X0);CHKERRQ(ierr);
  /* The TS at each level share the same tableau, work array, solution vector, stage values and stage derivatives */
  currentlevelrk = rk;
  while (currentlevelrk->subts_fast) {
    ierr = PetscMalloc1(rk->tableau->s,&currentlevelrk->YdotRHS_fast);CHKERRQ(ierr);
    ierr = PetscMalloc1(rk->tableau->s,&currentlevelrk->YdotRHS_slow);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)X0);CHKERRQ(ierr);
    currentlevelrk->X0 = X0;
    currentlevelrk->ts_root = ts;

    /* set up the ts for the slow part */
    nextlevelts = currentlevelrk->subts_slow;
    ierr = PetscNewLog(nextlevelts,&nextlevelrk);CHKERRQ(ierr);
    nextlevelrk->tableau = rk->tableau;
    nextlevelrk->work = rk->work;
    nextlevelrk->Y = rk->Y;
    nextlevelrk->YdotRHS = rk->YdotRHS;
    nextlevelts->data = (void*)nextlevelrk;
    ierr = TSCopyDM(ts,nextlevelts);CHKERRQ(ierr);
    ierr = TSSetSolution(nextlevelts,ts->vec_sol);CHKERRQ(ierr);

    /* set up the ts for the fast part */
    nextlevelts = currentlevelrk->subts_fast;
    ierr = PetscNewLog(nextlevelts,&nextlevelrk);CHKERRQ(ierr);
    nextlevelrk->tableau = rk->tableau;
    nextlevelrk->work = rk->work;
    nextlevelrk->Y = rk->Y;
    nextlevelrk->YdotRHS = rk->YdotRHS;
    nextlevelrk->dtratio = rk->dtratio;
    ierr = TSRHSSplitGetIS(nextlevelts,"slow",&nextlevelrk->is_slow);CHKERRQ(ierr);
    ierr = TSRHSSplitGetSubTS(nextlevelts,"slow",&nextlevelrk->subts_slow);CHKERRQ(ierr);
    ierr = TSRHSSplitGetIS(nextlevelts,"fast",&nextlevelrk->is_fast);CHKERRQ(ierr);
    ierr = TSRHSSplitGetSubTS(nextlevelts,"fast",&nextlevelrk->subts_fast);CHKERRQ(ierr);
    nextlevelts->data = (void*)nextlevelrk;
    ierr = TSCopyDM(ts,nextlevelts);CHKERRQ(ierr);
    ierr = TSSetSolution(nextlevelts,ts->vec_sol);CHKERRQ(ierr);

    currentlevelrk = nextlevelrk;
  }
  ierr = VecDestroy(&X0);CHKERRQ(ierr);

  ts->ops->step         = TSStep_RK_MultirateSplit;
  ts->ops->evaluatestep = TSEvaluateStep_RK_MultirateSplit;
  ts->ops->interpolate  = TSInterpolate_RK_MultirateSplit;
  PetscFunctionReturn(0);
}

PetscErrorCode TSRKSetMultirate_RK(TS ts,PetscBool use_multirate)
{
  TS_RK          *rk = (TS_RK*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  rk->use_multirate = use_multirate;
  if (use_multirate) {
    rk->dtratio = 2;
    ierr = PetscObjectComposeFunction((PetscObject)ts,"TSSetUp_RK_MultirateSplit_C",TSSetUp_RK_MultirateSplit);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)ts,"TSReset_RK_MultirateSplit_C",TSReset_RK_MultirateSplit);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)ts,"TSSetUp_RK_MultirateNonsplit_C",TSSetUp_RK_MultirateNonsplit);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)ts,"TSReset_RK_MultirateNonsplit_C",TSReset_RK_MultirateNonsplit);CHKERRQ(ierr);
  } else {
    rk->dtratio = 0;
    ierr = PetscObjectComposeFunction((PetscObject)ts,"TSSetUp_RK_MultirateSplit_C",NULL);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)ts,"TSReset_RK_MultirateSplit_C",NULL);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)ts,"TSSetUp_RK_MultirateNonsplit_C",NULL);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)ts,"TSReset_RK_MultirateNonsplit_C",NULL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode TSRKGetMultirate_RK(TS ts,PetscBool *use_multirate)
{
  TS_RK *rk = (TS_RK*)ts->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  *use_multirate = rk->use_multirate;
  PetscFunctionReturn(0);
}
