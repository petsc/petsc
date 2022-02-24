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

  PetscFunctionBegin;
  CHKERRQ(VecDestroy(&rk->X0));
  CHKERRQ(VecDestroyVecs(tab->s,&rk->YdotRHS_slow));
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

  PetscFunctionBegin;
  PetscCheck(B,PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"TSRK %s does not have an interpolation formula",rk->tableau->name);
  t = (itime - rk->ptime)/h;
  CHKERRQ(PetscMalloc1(s,&b));
  for (i=0; i<s; i++) b[i] = 0;
  for (j=0,tt=t; j<p; j++,tt*=t) {
    for (i=0; i<s; i++) {
      b[i]  += h * B[i*p+j] * tt;
    }
  }
  CHKERRQ(VecCopy(rk->X0,X));
  CHKERRQ(VecMAXPY(X,s,b,rk->YdotRHS_slow));
  CHKERRQ(PetscFree(b));
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

  PetscFunctionBegin;
  CHKERRQ(VecDuplicate(ts->vec_sol,&vec_fast));
  previousts = rk->subts_current;
  CHKERRQ(TSRHSSplitGetSubTS(rk->subts_current,"fast",&subts));
  CHKERRQ(TSRHSSplitGetSubTS(subts,"fast",&subts));
  for (k=0; k<rk->dtratio; k++) {
    for (i=0; i<s; i++) {
      CHKERRQ(TSInterpolate_RK_MultirateNonsplit(ts,t+k*h/rk->dtratio+h/rk->dtratio*c[i],Y[i]));
      for (j=0; j<i; j++) w[j] = h/rk->dtratio*A[i*s+j];
      /* update the fast components in the stage value, the slow components will be ignored, so we do not care the slow part in vec_fast */
      CHKERRQ(VecCopy(ts->vec_sol,vec_fast));
      CHKERRQ(VecMAXPY(vec_fast,i,w,YdotRHS));
      /* update the fast components in the stage value */
      CHKERRQ(VecGetSubVector(vec_fast,rk->is_fast,&subvec_fast));
      CHKERRQ(VecISCopy(Y[i],rk->is_fast,SCATTER_FORWARD,subvec_fast));
      CHKERRQ(VecRestoreSubVector(vec_fast,rk->is_fast,&subvec_fast));
      /* compute the stage RHS */
      CHKERRQ(TSComputeRHSFunction(ts,t+k*h/rk->dtratio+h/rk->dtratio*c[i],Y[i],YdotRHS[i]));
    }
    CHKERRQ(VecCopy(ts->vec_sol,vec_fast));
    CHKERRQ(TSEvaluateStep(ts,tab->order,vec_fast,NULL));
    /* update the fast components in the solution */
    CHKERRQ(VecGetSubVector(vec_fast,rk->is_fast,&subvec_fast));
    CHKERRQ(VecISCopy(ts->vec_sol,rk->is_fast,SCATTER_FORWARD,subvec_fast));
    CHKERRQ(VecRestoreSubVector(vec_fast,rk->is_fast,&subvec_fast));

    if (subts) {
      Vec *YdotRHS_copy;
      CHKERRQ(VecDuplicateVecs(ts->vec_sol,s,&YdotRHS_copy));
      rk->subts_current = rk->subts_fast;
      ts->ptime = t+k*h/rk->dtratio;
      ts->time_step = h/rk->dtratio;
      CHKERRQ(TSRHSSplitGetIS(rk->subts_current,"fast",&rk->is_fast));
      for (i=0; i<s; i++) {
        CHKERRQ(VecCopy(rk->YdotRHS_slow[i],YdotRHS_copy[i]));
        CHKERRQ(VecCopy(YdotRHS[i],rk->YdotRHS_slow[i]));
      }

      CHKERRQ(TSStepRefine_RK_MultirateNonsplit(ts));

      rk->subts_current = previousts;
      ts->ptime = t;
      ts->time_step = h;
      CHKERRQ(TSRHSSplitGetIS(previousts,"fast",&rk->is_fast));
      for (i=0; i<s; i++) {
        CHKERRQ(VecCopy(YdotRHS_copy[i],rk->YdotRHS_slow[i]));
      }
      CHKERRQ(VecDestroyVecs(s,&YdotRHS_copy));
    }
  }
  CHKERRQ(VecDestroy(&vec_fast));
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

  PetscFunctionBegin;
  rk->status = TS_STEP_INCOMPLETE;
  CHKERRQ(VecDuplicate(ts->vec_sol,&stage_slow));
  CHKERRQ(VecDuplicate(ts->vec_sol,&sol_slow));
  CHKERRQ(VecCopy(ts->vec_sol,rk->X0));
  for (i=0; i<s; i++) {
    rk->stage_time = t + h*c[i];
    CHKERRQ(TSPreStage(ts,rk->stage_time));
    CHKERRQ(VecCopy(ts->vec_sol,Y[i]));
    for (j=0; j<i; j++) w[j] = h*A[i*s+j];
    CHKERRQ(VecMAXPY(Y[i],i,w,YdotRHS_slow));
    CHKERRQ(TSPostStage(ts,rk->stage_time,i,Y));
    /* compute the stage RHS */
    CHKERRQ(TSComputeRHSFunction(ts,t+h*c[i],Y[i],YdotRHS_slow[i]));
  }
  /* update the slow components in the solution */
  rk->YdotRHS = YdotRHS_slow;
  rk->dtratio = 1;
  CHKERRQ(TSEvaluateStep(ts,tab->order,sol_slow,NULL));
  rk->dtratio = dtratio;
  rk->YdotRHS = YdotRHS;
  /* update the slow components in the solution */
  CHKERRQ(VecGetSubVector(sol_slow,is_slow,&subvec_slow));
  CHKERRQ(VecISCopy(ts->vec_sol,is_slow,SCATTER_FORWARD,subvec_slow));
  CHKERRQ(VecRestoreSubVector(sol_slow,is_slow,&subvec_slow));

  rk->subts_current = ts;
  rk->ptime = t;
  rk->time_step = h;
  CHKERRQ(TSStepRefine_RK_MultirateNonsplit(ts));

  ts->ptime = t + ts->time_step;
  ts->time_step = next_time_step;
  rk->status = TS_STEP_COMPLETE;

  /* free memory of work vectors */
  CHKERRQ(VecDestroy(&stage_slow));
  CHKERRQ(VecDestroy(&sol_slow));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSSetUp_RK_MultirateNonsplit(TS ts)
{
  TS_RK          *rk = (TS_RK*)ts->data;
  RKTableau      tab = rk->tableau;

  PetscFunctionBegin;
  CHKERRQ(TSRHSSplitGetIS(ts,"slow",&rk->is_slow));
  CHKERRQ(TSRHSSplitGetIS(ts,"fast",&rk->is_fast));
  PetscCheck(rk->is_slow && rk->is_fast,PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"Must set up RHSSplits with TSRHSSplitSetIS() using split names 'slow' and 'fast' respectively in order to use multirate RK");
  CHKERRQ(TSRHSSplitGetSubTS(ts,"slow",&rk->subts_slow));
  CHKERRQ(TSRHSSplitGetSubTS(ts,"fast",&rk->subts_fast));
  PetscCheck(rk->subts_slow && rk->subts_fast,PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"Must set up the RHSFunctions for 'slow' and 'fast' components using TSRHSSplitSetRHSFunction() or calling TSSetRHSFunction() for each sub-TS");
  CHKERRQ(VecDuplicate(ts->vec_sol,&rk->X0));
  CHKERRQ(VecDuplicateVecs(ts->vec_sol,tab->s,&rk->YdotRHS_slow));
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

  PetscFunctionBegin;
  CHKERRQ(TSGetDM(tssrc,&dmsrc));
  CHKERRQ(DMClone(dmsrc,&newdm));
  CHKERRQ(TSGetDM(tsdest,&dmdest));
  CHKERRQ(DMCopyDMTS(dmdest,newdm));
  CHKERRQ(DMCopyDMSNES(dmdest,newdm));
  CHKERRQ(TSSetDM(tsdest,newdm));
  CHKERRQ(DMDestroy(&newdm));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSReset_RK_MultirateSplit(TS ts)
{
  TS_RK          *rk = (TS_RK*)ts->data;

  PetscFunctionBegin;
  if (rk->subts_slow) {
    CHKERRQ(PetscFree(rk->subts_slow->data));
    rk->subts_slow = NULL;
  }
  if (rk->subts_fast) {
    CHKERRQ(PetscFree(rk->YdotRHS_fast));
    CHKERRQ(PetscFree(rk->YdotRHS_slow));
    CHKERRQ(VecDestroy(&rk->X0));
    CHKERRQ(TSReset_RK_MultirateSplit(rk->subts_fast));
    CHKERRQ(PetscFree(rk->subts_fast->data));
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

  PetscFunctionBegin;
  PetscCheck(B,PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"TSRK %s does not have an interpolation formula",rk->tableau->name);

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
  CHKERRQ(PetscMalloc1(s,&b));
  for (i=0; i<s; i++) b[i] = 0;
  for (j=0,tt=t; j<p; j++,tt*=t) {
    for (i=0; i<s; i++) {
      b[i]  += h * B[i*p+j] * tt;
    }
  }
  for (i=0; i<s; i++) {
    CHKERRQ(VecGetSubVector(rk->YdotRHS[i],rk->is_slow,&rk->YdotRHS_slow[i]));
  }
  CHKERRQ(VecGetSubVector(X,rk->is_slow,&Xslow));
  CHKERRQ(VecISCopy(rk->X0,rk->is_slow,SCATTER_REVERSE,Xslow));
  CHKERRQ(VecMAXPY(Xslow,s,b,rk->YdotRHS_slow));
  CHKERRQ(VecRestoreSubVector(X,rk->is_slow,&Xslow));
  for (i=0; i<s; i++) {
    CHKERRQ(VecRestoreSubVector(rk->YdotRHS[i],rk->is_slow,&rk->YdotRHS_slow[i]));
  }
  CHKERRQ(PetscFree(b));
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

  PetscFunctionBegin;
  CHKERRQ(VecCopy(ts->vec_sol,X));
  if (rk->slow) {
    for (j=0; j<s; j++) w[j] = h*tab->b[j];
    CHKERRQ(VecGetSubVector(ts->vec_sol,rk->is_slow,&Xslow));
    CHKERRQ(VecMAXPY(Xslow,s,w,rk->YdotRHS_slow));
    CHKERRQ(VecRestoreSubVector(ts->vec_sol,rk->is_slow,&Xslow));
  } else {
    for (j=0; j<s; j++) w[j] = h/rk->dtratio*tab->b[j];
    CHKERRQ(VecGetSubVector(X,rk->is_fast,&Xfast));
    CHKERRQ(VecMAXPY(Xfast,s,w,rk->YdotRHS_fast));
    CHKERRQ(VecRestoreSubVector(X,rk->is_fast,&Xfast));
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

  PetscFunctionBegin;
  for (k=0; k<rk->dtratio; k++) {
    CHKERRQ(VecGetSubVector(ts->vec_sol,rk->is_fast,&Xfast));
    for (i=0; i<s; i++) {
      CHKERRQ(VecGetSubVector(YdotRHS[i],rk->is_fast,&YdotRHS_fast[i]));
    }
    /* propagate fast component using small time steps */
    for (i=0; i<s; i++) {
      /* stage value for slow components */
      CHKERRQ(TSInterpolate_RK_MultirateSplit(rk->ts_root,t+k*h/rk->dtratio+h/rk->dtratio*c[i],Y[i]));
      currentlevelts = rk->ts_root;
      while (currentlevelts != ts) { /* all the slow parts need to be interpolated separated */
        currentlevelts = ((TS_RK*)currentlevelts->data)->subts_fast;
        CHKERRQ(TSInterpolate_RK_MultirateSplit(currentlevelts,t+k*h/rk->dtratio+h/rk->dtratio*c[i],Y[i]));
      }
      for (j=0; j<i; j++) w[j] = h/rk->dtratio*A[i*s+j];
      subrk_fast->stage_time = t + h/rk->dtratio*c[i];
      CHKERRQ(TSPreStage(subts_fast,subrk_fast->stage_time));
      /* stage value for fast components */
      CHKERRQ(VecGetSubVector(Y[i],rk->is_fast,&Yfast));
      CHKERRQ(VecCopy(Xfast,Yfast));
      CHKERRQ(VecMAXPY(Yfast,i,w,YdotRHS_fast));
      CHKERRQ(VecRestoreSubVector(Y[i],rk->is_fast,&Yfast));
      CHKERRQ(TSPostStage(subts_fast,subrk_fast->stage_time,i,Y));
      /* compute the stage RHS for fast components */
      CHKERRQ(TSComputeRHSFunction(subts_fast,t+k*h*rk->dtratio+h/rk->dtratio*c[i],Y[i],YdotRHS_fast[i]));
    }
    CHKERRQ(VecRestoreSubVector(ts->vec_sol,rk->is_fast,&Xfast));
    /* update the value of fast components using fast time step */
    rk->slow = PETSC_FALSE;
    CHKERRQ(TSEvaluateStep_RK_MultirateSplit(ts,tab->order,ts->vec_sol,NULL));
    for (i=0; i<s; i++) {
      CHKERRQ(VecRestoreSubVector(YdotRHS[i],rk->is_fast,&YdotRHS_fast[i]));
    }

    if (subrk_fast->subts_fast) {
      subts_fast->ptime = t+k*h/rk->dtratio;
      subts_fast->time_step = h/rk->dtratio;
      CHKERRQ(TSStepRefine_RK_MultirateSplit(subts_fast));
    }
    /* update the fast components of the solution */
    CHKERRQ(VecGetSubVector(ts->vec_sol,rk->is_fast,&Xfast));
    CHKERRQ(VecISCopy(rk->X0,rk->is_fast,SCATTER_FORWARD,Xfast));
    CHKERRQ(VecRestoreSubVector(ts->vec_sol,rk->is_fast,&Xfast));
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

  PetscFunctionBegin;
  rk->status = TS_STEP_INCOMPLETE;
  for (i=0; i<s; i++) {
    CHKERRQ(VecGetSubVector(YdotRHS[i],rk->is_slow,&YdotRHS_slow[i]));
    CHKERRQ(VecGetSubVector(YdotRHS[i],rk->is_fast,&YdotRHS_fast[i]));
  }
  CHKERRQ(VecCopy(ts->vec_sol,rk->X0));
  /* propagate both slow and fast components using large time steps */
  for (i=0; i<s; i++) {
    rk->stage_time = t + h*c[i];
    CHKERRQ(TSPreStage(ts,rk->stage_time));
    CHKERRQ(VecCopy(ts->vec_sol,Y[i]));
    CHKERRQ(VecGetSubVector(Y[i],rk->is_fast,&Yfast));
    CHKERRQ(VecGetSubVector(Y[i],rk->is_slow,&Yslow));
    for (j=0; j<i; j++) w[j] = h*A[i*s+j];
    CHKERRQ(VecMAXPY(Yfast,i,w,YdotRHS_fast));
    CHKERRQ(VecMAXPY(Yslow,i,w,YdotRHS_slow));
    CHKERRQ(VecRestoreSubVector(Y[i],rk->is_fast,&Yfast));
    CHKERRQ(VecRestoreSubVector(Y[i],rk->is_slow,&Yslow));
    CHKERRQ(TSPostStage(ts,rk->stage_time,i,Y));
    CHKERRQ(TSComputeRHSFunction(rk->subts_slow,t+h*c[i],Y[i],YdotRHS_slow[i]));
    CHKERRQ(TSComputeRHSFunction(rk->subts_fast,t+h*c[i],Y[i],YdotRHS_fast[i]));
  }
  rk->slow = PETSC_TRUE;
  /* update the slow components of the solution using slow time step */
  CHKERRQ(TSEvaluateStep_RK_MultirateSplit(ts,tab->order,ts->vec_sol,NULL));
  /* YdotRHS will be used for interpolation during refinement */
  for (i=0; i<s; i++) {
    CHKERRQ(VecRestoreSubVector(YdotRHS[i],rk->is_slow,&YdotRHS_slow[i]));
    CHKERRQ(VecRestoreSubVector(YdotRHS[i],rk->is_fast,&YdotRHS_fast[i]));
  }

  CHKERRQ(TSStepRefine_RK_MultirateSplit(ts));

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

  PetscFunctionBegin;
  CHKERRQ(TSRHSSplitGetIS(ts,"slow",&rk->is_slow));
  CHKERRQ(TSRHSSplitGetIS(ts,"fast",&rk->is_fast));
  PetscCheck(rk->is_slow && rk->is_fast,PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"Must set up RHSSplits with TSRHSSplitSetIS() using split names 'slow' and 'fast' respectively in order to use -ts_type bsi");
  CHKERRQ(TSRHSSplitGetSubTS(ts,"slow",&rk->subts_slow));
  CHKERRQ(TSRHSSplitGetSubTS(ts,"fast",&rk->subts_fast));
  PetscCheck(rk->subts_slow && rk->subts_fast,PetscObjectComm((PetscObject)ts),PETSC_ERR_USER,"Must set up the RHSFunctions for 'slow' and 'fast' components using TSRHSSplitSetRHSFunction() or calling TSSetRHSFunction() for each sub-TS");

  CHKERRQ(VecDuplicate(ts->vec_sol,&X0));
  /* The TS at each level share the same tableau, work array, solution vector, stage values and stage derivatives */
  currentlevelrk = rk;
  while (currentlevelrk->subts_fast) {
    CHKERRQ(PetscMalloc1(rk->tableau->s,&currentlevelrk->YdotRHS_fast));
    CHKERRQ(PetscMalloc1(rk->tableau->s,&currentlevelrk->YdotRHS_slow));
    CHKERRQ(PetscObjectReference((PetscObject)X0));
    currentlevelrk->X0 = X0;
    currentlevelrk->ts_root = ts;

    /* set up the ts for the slow part */
    nextlevelts = currentlevelrk->subts_slow;
    CHKERRQ(PetscNewLog(nextlevelts,&nextlevelrk));
    nextlevelrk->tableau = rk->tableau;
    nextlevelrk->work = rk->work;
    nextlevelrk->Y = rk->Y;
    nextlevelrk->YdotRHS = rk->YdotRHS;
    nextlevelts->data = (void*)nextlevelrk;
    CHKERRQ(TSCopyDM(ts,nextlevelts));
    CHKERRQ(TSSetSolution(nextlevelts,ts->vec_sol));

    /* set up the ts for the fast part */
    nextlevelts = currentlevelrk->subts_fast;
    CHKERRQ(PetscNewLog(nextlevelts,&nextlevelrk));
    nextlevelrk->tableau = rk->tableau;
    nextlevelrk->work = rk->work;
    nextlevelrk->Y = rk->Y;
    nextlevelrk->YdotRHS = rk->YdotRHS;
    nextlevelrk->dtratio = rk->dtratio;
    CHKERRQ(TSRHSSplitGetIS(nextlevelts,"slow",&nextlevelrk->is_slow));
    CHKERRQ(TSRHSSplitGetSubTS(nextlevelts,"slow",&nextlevelrk->subts_slow));
    CHKERRQ(TSRHSSplitGetIS(nextlevelts,"fast",&nextlevelrk->is_fast));
    CHKERRQ(TSRHSSplitGetSubTS(nextlevelts,"fast",&nextlevelrk->subts_fast));
    nextlevelts->data = (void*)nextlevelrk;
    CHKERRQ(TSCopyDM(ts,nextlevelts));
    CHKERRQ(TSSetSolution(nextlevelts,ts->vec_sol));

    currentlevelrk = nextlevelrk;
  }
  CHKERRQ(VecDestroy(&X0));

  ts->ops->step         = TSStep_RK_MultirateSplit;
  ts->ops->evaluatestep = TSEvaluateStep_RK_MultirateSplit;
  ts->ops->interpolate  = TSInterpolate_RK_MultirateSplit;
  PetscFunctionReturn(0);
}

PetscErrorCode TSRKSetMultirate_RK(TS ts,PetscBool use_multirate)
{
  TS_RK          *rk = (TS_RK*)ts->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  rk->use_multirate = use_multirate;
  if (use_multirate) {
    rk->dtratio = 2;
    CHKERRQ(PetscObjectComposeFunction((PetscObject)ts,"TSSetUp_RK_MultirateSplit_C",TSSetUp_RK_MultirateSplit));
    CHKERRQ(PetscObjectComposeFunction((PetscObject)ts,"TSReset_RK_MultirateSplit_C",TSReset_RK_MultirateSplit));
    CHKERRQ(PetscObjectComposeFunction((PetscObject)ts,"TSSetUp_RK_MultirateNonsplit_C",TSSetUp_RK_MultirateNonsplit));
    CHKERRQ(PetscObjectComposeFunction((PetscObject)ts,"TSReset_RK_MultirateNonsplit_C",TSReset_RK_MultirateNonsplit));
  } else {
    rk->dtratio = 0;
    CHKERRQ(PetscObjectComposeFunction((PetscObject)ts,"TSSetUp_RK_MultirateSplit_C",NULL));
    CHKERRQ(PetscObjectComposeFunction((PetscObject)ts,"TSReset_RK_MultirateSplit_C",NULL));
    CHKERRQ(PetscObjectComposeFunction((PetscObject)ts,"TSSetUp_RK_MultirateNonsplit_C",NULL));
    CHKERRQ(PetscObjectComposeFunction((PetscObject)ts,"TSReset_RK_MultirateNonsplit_C",NULL));
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
