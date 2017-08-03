/*
  Code for timestepping with implicit Theta method
*/
#include <petsc/private/tsimpl.h>                /*I   "petscts.h"   I*/
#include <petscsnes.h>
#include <petscdm.h>
#include <petscmat.h>

typedef struct {
  /* context for time stepping */
  PetscReal    stage_time;
  Vec          X0,X,Xdot;                /* Storage for stages and time derivative */
  Vec          affine;                   /* Affine vector needed for residual at beginning of step in endpoint formulation */
  PetscReal    Theta;
  PetscReal    ptime;
  PetscReal    time_step;
  PetscInt     order;
  PetscBool    endpoint;
  PetscBool    extrapolate;
  TSStepStatus status;
  Vec          VecCostIntegral0;         /* Backup for roll-backs due to events */

  /* context for sensitivity analysis */
  PetscInt     num_tlm;                  /* Total number of tangent linear equations */
  Vec          *VecsDeltaLam;            /* Increment of the adjoint sensitivity w.r.t IC at stage */
  Vec          *VecsDeltaMu;             /* Increment of the adjoint sensitivity w.r.t P at stage */
  Vec          *VecsSensiTemp;           /* Vector to be timed with Jacobian transpose */
  Vec          *VecsDeltaFwdSensi;       /* Increment of the forward sensitivity at stage */
  Vec          VecIntegralSensiTemp;     /* Working vector for forward integral sensitivity */
  Vec          VecIntegralSensipTemp;    /* Working vector for forward integral sensitivity */
  Vec          *VecsFwdSensi0;           /* backup for roll-backs due to events */
  Vec          *VecsIntegralSensi0;      /* backup for roll-backs due to events */
  Vec          *VecsIntegralSensip0;     /* backup for roll-backs due to events */

  /* context for error estimation */
  Vec          vec_sol_prev;
  Vec          vec_lte_work;
} TS_Theta;

static PetscErrorCode TSThetaGetX0AndXdot(TS ts,DM dm,Vec *X0,Vec *Xdot)
{
  TS_Theta       *th = (TS_Theta*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (X0) {
    if (dm && dm != ts->dm) {
      ierr = DMGetNamedGlobalVector(dm,"TSTheta_X0",X0);CHKERRQ(ierr);
    } else *X0 = ts->vec_sol;
  }
  if (Xdot) {
    if (dm && dm != ts->dm) {
      ierr = DMGetNamedGlobalVector(dm,"TSTheta_Xdot",Xdot);CHKERRQ(ierr);
    } else *Xdot = th->Xdot;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSThetaRestoreX0AndXdot(TS ts,DM dm,Vec *X0,Vec *Xdot)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (X0) {
    if (dm && dm != ts->dm) {
      ierr = DMRestoreNamedGlobalVector(dm,"TSTheta_X0",X0);CHKERRQ(ierr);
    }
  }
  if (Xdot) {
    if (dm && dm != ts->dm) {
      ierr = DMRestoreNamedGlobalVector(dm,"TSTheta_Xdot",Xdot);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCoarsenHook_TSTheta(DM fine,DM coarse,void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMRestrictHook_TSTheta(DM fine,Mat restrct,Vec rscale,Mat inject,DM coarse,void *ctx)
{
  TS             ts = (TS)ctx;
  PetscErrorCode ierr;
  Vec            X0,Xdot,X0_c,Xdot_c;

  PetscFunctionBegin;
  ierr = TSThetaGetX0AndXdot(ts,fine,&X0,&Xdot);CHKERRQ(ierr);
  ierr = TSThetaGetX0AndXdot(ts,coarse,&X0_c,&Xdot_c);CHKERRQ(ierr);
  ierr = MatRestrict(restrct,X0,X0_c);CHKERRQ(ierr);
  ierr = MatRestrict(restrct,Xdot,Xdot_c);CHKERRQ(ierr);
  ierr = VecPointwiseMult(X0_c,rscale,X0_c);CHKERRQ(ierr);
  ierr = VecPointwiseMult(Xdot_c,rscale,Xdot_c);CHKERRQ(ierr);
  ierr = TSThetaRestoreX0AndXdot(ts,fine,&X0,&Xdot);CHKERRQ(ierr);
  ierr = TSThetaRestoreX0AndXdot(ts,coarse,&X0_c,&Xdot_c);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMSubDomainHook_TSTheta(DM dm,DM subdm,void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMSubDomainRestrictHook_TSTheta(DM dm,VecScatter gscat,VecScatter lscat,DM subdm,void *ctx)
{
  TS             ts = (TS)ctx;
  PetscErrorCode ierr;
  Vec            X0,Xdot,X0_sub,Xdot_sub;

  PetscFunctionBegin;
  ierr = TSThetaGetX0AndXdot(ts,dm,&X0,&Xdot);CHKERRQ(ierr);
  ierr = TSThetaGetX0AndXdot(ts,subdm,&X0_sub,&Xdot_sub);CHKERRQ(ierr);

  ierr = VecScatterBegin(gscat,X0,X0_sub,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(gscat,X0,X0_sub,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  ierr = VecScatterBegin(gscat,Xdot,Xdot_sub,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(gscat,Xdot,Xdot_sub,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  ierr = TSThetaRestoreX0AndXdot(ts,dm,&X0,&Xdot);CHKERRQ(ierr);
  ierr = TSThetaRestoreX0AndXdot(ts,subdm,&X0_sub,&Xdot_sub);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSThetaEvaluateCostIntegral(TS ts)
{
  TS_Theta       *th = (TS_Theta*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (th->endpoint) {
    /* Evolve ts->vec_costintegral to compute integrals */
    if (th->Theta!=1.0) {
      ierr = TSComputeCostIntegrand(ts,th->ptime,th->X0,ts->vec_costintegrand);CHKERRQ(ierr);
      ierr = VecAXPY(ts->vec_costintegral,th->time_step*(1.0-th->Theta),ts->vec_costintegrand);CHKERRQ(ierr);
    }
    ierr = TSComputeCostIntegrand(ts,ts->ptime,ts->vec_sol,ts->vec_costintegrand);CHKERRQ(ierr);
    ierr = VecAXPY(ts->vec_costintegral,th->time_step*th->Theta,ts->vec_costintegrand);CHKERRQ(ierr);
  } else {
    ierr = TSComputeCostIntegrand(ts,th->stage_time,th->X,ts->vec_costintegrand);CHKERRQ(ierr);
    ierr = VecAXPY(ts->vec_costintegral,th->time_step,ts->vec_costintegrand);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSForwardCostIntegral_Theta(TS ts)
{
  TS_Theta       *th = (TS_Theta*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* backup cost integral */
  ierr = VecCopy(ts->vec_costintegral,th->VecCostIntegral0);CHKERRQ(ierr);
  ierr = TSThetaEvaluateCostIntegral(ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSAdjointCostIntegral_Theta(TS ts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSThetaEvaluateCostIntegral(ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TS_SNESSolve(TS ts,Vec b,Vec x)
{
  PetscInt       nits,lits;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESSolve(ts->snes,b,x);CHKERRQ(ierr);
  ierr = SNESGetIterationNumber(ts->snes,&nits);CHKERRQ(ierr);
  ierr = SNESGetLinearSolveIterations(ts->snes,&lits);CHKERRQ(ierr);
  ts->snes_its += nits; ts->ksp_its += lits;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSStep_Theta(TS ts)
{
  TS_Theta       *th = (TS_Theta*)ts->data;
  PetscInt       rejections = 0;
  PetscBool      stageok,accept = PETSC_TRUE;
  PetscReal      next_time_step = ts->time_step;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!ts->steprollback) {
    if (th->vec_sol_prev) { ierr = VecCopy(th->X0,th->vec_sol_prev);CHKERRQ(ierr); }
    ierr = VecCopy(ts->vec_sol,th->X0);CHKERRQ(ierr);
  }

  th->status = TS_STEP_INCOMPLETE;
  while (!ts->reason && th->status != TS_STEP_COMPLETE) {

    PetscReal shift = 1/(th->Theta*ts->time_step);
    th->stage_time = ts->ptime + (th->endpoint ? (PetscReal)1 : th->Theta)*ts->time_step;

    ierr = VecCopy(th->X0,th->X);CHKERRQ(ierr);
    if (th->extrapolate && !ts->steprestart) {
      ierr = VecAXPY(th->X,1/shift,th->Xdot);CHKERRQ(ierr);
    }
    if (th->endpoint) { /* This formulation assumes linear time-independent mass matrix */
      if (!th->affine) {ierr = VecDuplicate(ts->vec_sol,&th->affine);CHKERRQ(ierr);}
      ierr = VecZeroEntries(th->Xdot);CHKERRQ(ierr);
      ierr = TSComputeIFunction(ts,ts->ptime,th->X0,th->Xdot,th->affine,PETSC_FALSE);CHKERRQ(ierr);
      ierr = VecScale(th->affine,(th->Theta-1)/th->Theta);CHKERRQ(ierr);
    } else if (th->affine) { /* Just in case th->endpoint is changed between calls to TSStep_Theta() */
      ierr = VecZeroEntries(th->affine);CHKERRQ(ierr);
    }
    ierr = TSPreStage(ts,th->stage_time);CHKERRQ(ierr);
    ierr = TS_SNESSolve(ts,th->affine,th->X);CHKERRQ(ierr);
    ierr = TSPostStage(ts,th->stage_time,0,&th->X);CHKERRQ(ierr);
    ierr = TSAdaptCheckStage(ts->adapt,ts,th->stage_time,th->X,&stageok);CHKERRQ(ierr);
    if (!stageok) goto reject_step;

    th->status = TS_STEP_PENDING;
    if (th->endpoint) {
      ierr = VecCopy(th->X,ts->vec_sol);CHKERRQ(ierr);
    } else {
      ierr = VecAXPBYPCZ(th->Xdot,-shift,shift,0,th->X0,th->X);CHKERRQ(ierr);
      ierr = VecAXPY(ts->vec_sol,ts->time_step,th->Xdot);CHKERRQ(ierr);
    }
    ierr = TSAdaptChoose(ts->adapt,ts,ts->time_step,NULL,&next_time_step,&accept);CHKERRQ(ierr);
    th->status = accept ? TS_STEP_COMPLETE : TS_STEP_INCOMPLETE;
    if (!accept) {
      ierr = VecCopy(th->X0,ts->vec_sol);CHKERRQ(ierr);
      ts->time_step = next_time_step;
      goto reject_step;
    }

    if (ts->forward_solve || ts->costintegralfwd) { /* Save the info for the later use in cost integral evaluation */
      th->ptime     = ts->ptime;
      th->time_step = ts->time_step;
    }

    ts->ptime += ts->time_step;
    ts->time_step = next_time_step;
    break;

  reject_step:
    ts->reject++; accept = PETSC_FALSE;
    if (!ts->reason && ++rejections > ts->max_reject && ts->max_reject >= 0) {
      ts->reason = TS_DIVERGED_STEP_REJECTED;
      ierr = PetscInfo2(ts,"Step=%D, step rejections %D greater than current TS allowed, stopping solve\n",ts->steps,rejections);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSAdjointStep_Theta(TS ts)
{
  TS_Theta            *th = (TS_Theta*)ts->data;
  Vec                 *VecsDeltaLam = th->VecsDeltaLam,*VecsDeltaMu = th->VecsDeltaMu,*VecsSensiTemp = th->VecsSensiTemp;
  PetscInt            nadj;
  PetscErrorCode      ierr;
  Mat                 J,Jp;
  KSP                 ksp;
  PetscReal           shift;

  PetscFunctionBegin;
  th->status = TS_STEP_INCOMPLETE;
  ierr = SNESGetKSP(ts->snes,&ksp);CHKERRQ(ierr);
  ierr = TSGetIJacobian(ts,&J,&Jp,NULL,NULL);CHKERRQ(ierr);

  /* If endpoint=1, th->ptime and th->X0 will be used; if endpoint=0, th->stage_time and th->X will be used. */
  th->stage_time = th->endpoint ? ts->ptime : (ts->ptime+(1.-th->Theta)*ts->time_step); /* time_step is negative*/
  th->ptime      = ts->ptime + ts->time_step;
  th->time_step  = -ts->time_step;

  /* Build RHS */
  if (ts->vec_costintegral) { /* Cost function has an integral term */
    if (th->endpoint) {
      ierr = TSAdjointComputeDRDYFunction(ts,th->stage_time,ts->vec_sol,ts->vecs_drdy);CHKERRQ(ierr);
    } else {
      ierr = TSAdjointComputeDRDYFunction(ts,th->stage_time,th->X,ts->vecs_drdy);CHKERRQ(ierr);
    }
  }
  for (nadj=0; nadj<ts->numcost; nadj++) {
    ierr = VecCopy(ts->vecs_sensi[nadj],VecsSensiTemp[nadj]);CHKERRQ(ierr);
    ierr = VecScale(VecsSensiTemp[nadj],1./(th->Theta*th->time_step));CHKERRQ(ierr);
    if (ts->vec_costintegral) {
      ierr = VecAXPY(VecsSensiTemp[nadj],1.,ts->vecs_drdy[nadj]);CHKERRQ(ierr);
    }
  }

  /* Build LHS */
  shift = 1./(th->Theta*th->time_step);
  if (th->endpoint) {
    ierr = TSComputeIJacobian(ts,th->stage_time,ts->vec_sol,th->Xdot,shift,J,Jp,PETSC_FALSE);CHKERRQ(ierr);
  } else {
    ierr = TSComputeIJacobian(ts,th->stage_time,th->X,th->Xdot,shift,J,Jp,PETSC_FALSE);CHKERRQ(ierr);
  }
  ierr = KSPSetOperators(ksp,J,Jp);CHKERRQ(ierr);

  /* Solve LHS X = RHS */
  for (nadj=0; nadj<ts->numcost; nadj++) {
    ierr = KSPSolveTranspose(ksp,VecsSensiTemp[nadj],VecsDeltaLam[nadj]);CHKERRQ(ierr);
  }

  /* Update sensitivities, and evaluate integrals if there is any */
  if(th->endpoint) { /* two-stage case */
    if (th->Theta!=1.) {
      shift = 1./((th->Theta-1.)*th->time_step);
      ierr  = TSComputeIJacobian(ts,th->ptime,th->X0,th->Xdot,shift,J,Jp,PETSC_FALSE);CHKERRQ(ierr);
      if (ts->vec_costintegral) {
        ierr = TSAdjointComputeDRDYFunction(ts,th->ptime,th->X0,ts->vecs_drdy);CHKERRQ(ierr);
      }
      for (nadj=0; nadj<ts->numcost; nadj++) {
        ierr = MatMultTranspose(J,VecsDeltaLam[nadj],ts->vecs_sensi[nadj]);CHKERRQ(ierr);
        if (ts->vec_costintegral) {
          ierr = VecAXPY(ts->vecs_sensi[nadj],-1.,ts->vecs_drdy[nadj]);CHKERRQ(ierr);
        }
        ierr = VecScale(ts->vecs_sensi[nadj],1./shift);CHKERRQ(ierr);
      }
    } else { /* backward Euler */
      shift = 0.0;
      ierr  = TSComputeIJacobian(ts,th->stage_time,ts->vec_sol,th->Xdot,shift,J,Jp,PETSC_FALSE);CHKERRQ(ierr); /* get -f_y */
      for (nadj=0; nadj<ts->numcost; nadj++) {
        ierr = MatMultTranspose(J,VecsDeltaLam[nadj],VecsSensiTemp[nadj]);CHKERRQ(ierr);
        ierr = VecAXPY(ts->vecs_sensi[nadj],-th->time_step,VecsSensiTemp[nadj]);CHKERRQ(ierr);
        if (ts->vec_costintegral) {
          ierr = VecAXPY(ts->vecs_sensi[nadj],th->time_step,ts->vecs_drdy[nadj]);CHKERRQ(ierr);
        }
      }
    }

    if (ts->vecs_sensip) { /* sensitivities wrt parameters */
      ierr = TSAdjointComputeRHSJacobian(ts,th->stage_time,ts->vec_sol,ts->Jacp);CHKERRQ(ierr);
      for (nadj=0; nadj<ts->numcost; nadj++) {
        ierr = MatMultTranspose(ts->Jacp,VecsDeltaLam[nadj],VecsDeltaMu[nadj]);CHKERRQ(ierr);
        ierr = VecAXPY(ts->vecs_sensip[nadj],th->time_step*th->Theta,VecsDeltaMu[nadj]);CHKERRQ(ierr);
      }
      if (th->Theta!=1.) {
        ierr = TSAdjointComputeRHSJacobian(ts,th->ptime,th->X0,ts->Jacp);CHKERRQ(ierr);
        for (nadj=0; nadj<ts->numcost; nadj++) {
          ierr = MatMultTranspose(ts->Jacp,VecsDeltaLam[nadj],VecsDeltaMu[nadj]);CHKERRQ(ierr);
          ierr = VecAXPY(ts->vecs_sensip[nadj],th->time_step*(1.-th->Theta),VecsDeltaMu[nadj]);CHKERRQ(ierr);
        }
      }
      if (ts->vec_costintegral) {
        ierr = TSAdjointComputeDRDPFunction(ts,th->stage_time,ts->vec_sol,ts->vecs_drdp);CHKERRQ(ierr);
        for (nadj=0; nadj<ts->numcost; nadj++) {
          ierr = VecAXPY(ts->vecs_sensip[nadj],th->time_step*th->Theta,ts->vecs_drdp[nadj]);CHKERRQ(ierr);
        }
        if (th->Theta!=1.) {
          ierr = TSAdjointComputeDRDPFunction(ts,th->ptime,th->X0,ts->vecs_drdp);CHKERRQ(ierr);
          for (nadj=0; nadj<ts->numcost; nadj++) {
            ierr = VecAXPY(ts->vecs_sensip[nadj],th->time_step*(1.-th->Theta),ts->vecs_drdp[nadj]);CHKERRQ(ierr);
          }
        }
      }
    }
  } else { /* one-stage case */
    shift = 0.0;
    ierr  = TSComputeIJacobian(ts,th->stage_time,th->X,th->Xdot,shift,J,Jp,PETSC_FALSE);CHKERRQ(ierr); /* get -f_y */
    if (ts->vec_costintegral) {
      ierr = TSAdjointComputeDRDYFunction(ts,th->stage_time,th->X,ts->vecs_drdy);CHKERRQ(ierr);
    }
    for (nadj=0; nadj<ts->numcost; nadj++) {
      ierr = MatMultTranspose(J,VecsDeltaLam[nadj],VecsSensiTemp[nadj]);CHKERRQ(ierr);
      ierr = VecAXPY(ts->vecs_sensi[nadj],-th->time_step,VecsSensiTemp[nadj]);CHKERRQ(ierr);
      if (ts->vec_costintegral) {
        ierr = VecAXPY(ts->vecs_sensi[nadj],th->time_step,ts->vecs_drdy[nadj]);CHKERRQ(ierr);
      }
    }
    if (ts->vecs_sensip) {
      ierr = TSAdjointComputeRHSJacobian(ts,th->stage_time,th->X,ts->Jacp);CHKERRQ(ierr);
      for (nadj=0; nadj<ts->numcost; nadj++) {
        ierr = MatMultTranspose(ts->Jacp,VecsDeltaLam[nadj],VecsDeltaMu[nadj]);CHKERRQ(ierr);
        ierr = VecAXPY(ts->vecs_sensip[nadj],th->time_step,VecsDeltaMu[nadj]);CHKERRQ(ierr);
      }
      if (ts->vec_costintegral) {
        ierr = TSAdjointComputeDRDPFunction(ts,th->stage_time,th->X,ts->vecs_drdp);CHKERRQ(ierr);
        for (nadj=0; nadj<ts->numcost; nadj++) {
          ierr = VecAXPY(ts->vecs_sensip[nadj],th->time_step,ts->vecs_drdp[nadj]);CHKERRQ(ierr);
        }
      }
    }
  }

  th->status = TS_STEP_COMPLETE;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSInterpolate_Theta(TS ts,PetscReal t,Vec X)
{
  TS_Theta       *th = (TS_Theta*)ts->data;
  PetscReal      dt  = t - ts->ptime;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCopy(ts->vec_sol,th->X);CHKERRQ(ierr);
  if (th->endpoint) dt *= th->Theta;
  ierr = VecWAXPY(X,dt,th->Xdot,th->X);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSEvaluateWLTE_Theta(TS ts,NormType wnormtype,PetscInt *order,PetscReal *wlte)
{
  TS_Theta       *th = (TS_Theta*)ts->data;
  Vec            X = ts->vec_sol;      /* X = solution */
  Vec            Y = th->vec_lte_work; /* Y = X + LTE  */
  PetscReal      wltea,wlter;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!th->vec_sol_prev) {*wlte = -1; PetscFunctionReturn(0);}
  /* Cannot compute LTE in first step or in restart after event */
  if (ts->steprestart) {*wlte = -1; PetscFunctionReturn(0);}
  /* Compute LTE using backward differences with non-constant time step */
  {
    PetscReal   h = ts->time_step, h_prev = ts->ptime - ts->ptime_prev;
    PetscReal   a = 1 + h_prev/h;
    PetscScalar scal[3]; Vec vecs[3];
    scal[0] = +1/a; scal[1] = -1/(a-1); scal[2] = +1/(a*(a-1));
    vecs[0] = X;    vecs[1] = th->X0;   vecs[2] = th->vec_sol_prev;
    ierr = VecCopy(X,Y);CHKERRQ(ierr);
    ierr = VecMAXPY(Y,3,scal,vecs);CHKERRQ(ierr);
    ierr = TSErrorWeightedNorm(ts,X,Y,wnormtype,wlte,&wltea,&wlter);CHKERRQ(ierr);
  }
  if (order) *order = 2;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSRollBack_Theta(TS ts)
{
  TS_Theta       *th = (TS_Theta*)ts->data;
  PetscInt       ntlm,ncost;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCopy(th->X0,ts->vec_sol);CHKERRQ(ierr);
  if (ts->vec_costintegral && ts->costintegralfwd) {
    ierr = VecCopy(th->VecCostIntegral0,ts->vec_costintegral);CHKERRQ(ierr);
  }
  th->status = TS_STEP_INCOMPLETE;

  for (ntlm=0;ntlm<th->num_tlm;ntlm++) {
    ierr = VecCopy(th->VecsFwdSensi0[ntlm],ts->vecs_fwdsensipacked[ntlm]);CHKERRQ(ierr);
  }
  if (ts->vecs_integral_sensi) {
    for (ncost=0;ncost<ts->numcost;ncost++) {
      ierr = VecCopy(th->VecsIntegralSensi0[ncost],ts->vecs_integral_sensi[ncost]);CHKERRQ(ierr);
    }
  }
  if (ts->vecs_integral_sensip) {
    for (ncost=0;ncost<ts->numcost;ncost++) {
      ierr = VecCopy(th->VecsIntegralSensip0[ncost],ts->vecs_integral_sensip[ncost]);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSThetaIntegrandDerivative(TS ts,PetscInt numtlm,Vec DRncostDY,Vec* s,Vec DRncostDP,Vec VecIntegrandDerivative)
{
  PetscInt    ntlm,low,high;
  PetscScalar *v;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = VecGetOwnershipRange(VecIntegrandDerivative,&low,&high);CHKERRQ(ierr);
  ierr = VecGetArray(VecIntegrandDerivative,&v);CHKERRQ(ierr);
  for (ntlm=low; ntlm<high; ntlm++) {
    ierr = VecDot(DRncostDY,s[ntlm],&v[ntlm-low]);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(VecIntegrandDerivative,&v);CHKERRQ(ierr);
  if (DRncostDP) {
    ierr = VecAXPY(VecIntegrandDerivative,1.,DRncostDP);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSForwardStep_Theta(TS ts)
{
  TS_Theta       *th = (TS_Theta*)ts->data;
  Vec            *VecsDeltaFwdSensi  = th->VecsDeltaFwdSensi;
  PetscInt       ntlm,ncost,np;
  KSP            ksp;
  Mat            J,Jp;
  PetscReal      shift;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (ntlm=0; ntlm<th->num_tlm; ntlm++) {
    ierr = VecCopy(ts->vecs_fwdsensipacked[ntlm],th->VecsFwdSensi0[ntlm]);CHKERRQ(ierr);
  }
  for (ncost=0; ncost<ts->numcost; ncost++) {
    if (ts->vecs_integral_sensi) {
      ierr = VecCopy(ts->vecs_integral_sensi[ncost],th->VecsIntegralSensi0[ncost]);CHKERRQ(ierr);
    }
    if (ts->vecs_integral_sensip) {
      ierr = VecCopy(ts->vecs_integral_sensip[ncost],th->VecsIntegralSensip0[ncost]);CHKERRQ(ierr);
    }
  }

  ierr = SNESGetKSP(ts->snes,&ksp);CHKERRQ(ierr);
  ierr = TSGetIJacobian(ts,&J,&Jp,NULL,NULL);CHKERRQ(ierr);

  /* Build RHS */
  if (th->endpoint) { /* 2-stage method*/
    shift = 1./((th->Theta-1.)*th->time_step);
    ierr = TSComputeIJacobian(ts,th->ptime,th->X0,th->Xdot,shift,J,Jp,PETSC_FALSE);CHKERRQ(ierr);

    for (ntlm=0; ntlm<th->num_tlm; ntlm++) {
      ierr = MatMult(J,ts->vecs_fwdsensipacked[ntlm],VecsDeltaFwdSensi[ntlm]);CHKERRQ(ierr);
      ierr = VecScale(VecsDeltaFwdSensi[ntlm],(th->Theta-1.)/th->Theta);CHKERRQ(ierr);
    }
    /* Add the f_p forcing terms */
    ierr = TSForwardComputeRHSJacobianP(ts,th->ptime,th->X0,ts->vecs_jacp);CHKERRQ(ierr);
    for (np=0; np<ts->num_parameters; np++) {
      ierr = VecAXPY(VecsDeltaFwdSensi[np+ts->num_initialvalues],(1.-th->Theta)/th->Theta,ts->vecs_jacp[np]);CHKERRQ(ierr);
    }
    ierr = TSForwardComputeRHSJacobianP(ts,th->stage_time,ts->vec_sol,ts->vecs_jacp);CHKERRQ(ierr);
    for (np=0; np<ts->num_parameters; np++) {
      ierr = VecAXPY(VecsDeltaFwdSensi[np+ts->num_initialvalues],1,ts->vecs_jacp[np]);CHKERRQ(ierr);
    }
  } else { /* 1-stage method */
    shift = 0.0;
    ierr = TSComputeIJacobian(ts,th->stage_time,th->X,th->Xdot,shift,J,Jp,PETSC_FALSE);CHKERRQ(ierr);
    for (ntlm=0; ntlm<ts->num_parameters; ntlm++) {
      ierr = MatMult(J,ts->vecs_fwdsensipacked[ntlm],VecsDeltaFwdSensi[ntlm]);CHKERRQ(ierr);
      ierr = VecScale(VecsDeltaFwdSensi[ntlm],-1);CHKERRQ(ierr);
    }
    /* Add the f_p forcing terms */
    ierr = TSForwardComputeRHSJacobianP(ts,th->stage_time,th->X,ts->vecs_jacp);CHKERRQ(ierr);
    for (np=0; np<ts->num_parameters; np++) {
      ierr = VecAXPY(VecsDeltaFwdSensi[np+ts->num_initialvalues],1,ts->vecs_jacp[np]);CHKERRQ(ierr);
    }
  }

  /* Build LHS */
  shift = 1/(th->Theta*th->time_step);
  if (th->endpoint) {
    ierr  = TSComputeIJacobian(ts,th->stage_time,ts->vec_sol,th->Xdot,shift,J,Jp,PETSC_FALSE);CHKERRQ(ierr);
  } else {
    ierr = TSComputeIJacobian(ts,th->stage_time,th->X,th->Xdot,shift,J,Jp,PETSC_FALSE);CHKERRQ(ierr);
  }
  ierr = KSPSetOperators(ksp,J,Jp);CHKERRQ(ierr);

  /*
    Evaluate the first stage of integral gradients with the 2-stage method:
    drdy|t_n*S(t_n) + drdp|t_n
    This is done before the linear solve because the sensitivity variable S(t_n) will be propagated to S(t_{n+1})
    It is important to note that sensitivitesi to parameters (sensip) and sensitivities to initial values (sensi) are independent of each other, but integral sensip relies on sensip and integral sensi requires sensi
   */
  if (th->endpoint) { /* 2-stage method only */
    if (ts->vecs_integral_sensi || ts->vecs_integral_sensip) {
      ierr = TSAdjointComputeDRDYFunction(ts,th->ptime,th->X0,ts->vecs_drdy);CHKERRQ(ierr);
    }
    if (ts->vecs_integral_sensip) {
      ierr = TSAdjointComputeDRDPFunction(ts,th->ptime,th->X0,ts->vecs_drdp);CHKERRQ(ierr);
      for (ncost=0; ncost<ts->numcost; ncost++) {
        ierr = TSThetaIntegrandDerivative(ts,ts->num_parameters,ts->vecs_drdy[ncost],&ts->vecs_fwdsensipacked[ts->num_initialvalues],ts->vecs_drdp[ncost],th->VecIntegralSensipTemp);CHKERRQ(ierr);
        ierr = VecAXPY(ts->vecs_integral_sensip[ncost],th->time_step*(1.-th->Theta),th->VecIntegralSensipTemp);CHKERRQ(ierr);
      }
    }
    if (ts->vecs_integral_sensi) {
      for (ncost=0; ncost<ts->numcost; ncost++) {
        ierr = TSThetaIntegrandDerivative(ts,ts->num_initialvalues,ts->vecs_drdy[ncost],ts->vecs_fwdsensipacked,NULL,th->VecIntegralSensiTemp);CHKERRQ(ierr);
        ierr = VecAXPY(ts->vecs_integral_sensi[ncost],th->time_step*(1.-th->Theta),th->VecIntegralSensiTemp);CHKERRQ(ierr);
      }
    }
  }

  /* Solve the tangent linear equation for forward sensitivities to parameters */
  for (ntlm=0; ntlm<th->num_tlm; ntlm++) {
    if (th->endpoint) {
      ierr = KSPSolve(ksp,VecsDeltaFwdSensi[ntlm],ts->vecs_fwdsensipacked[ntlm]);CHKERRQ(ierr);
    } else {
      ierr = KSPSolve(ksp,VecsDeltaFwdSensi[ntlm],VecsDeltaFwdSensi[ntlm]);CHKERRQ(ierr);
      ierr = VecAXPY(ts->vecs_fwdsensipacked[ntlm],1.,VecsDeltaFwdSensi[ntlm]);CHKERRQ(ierr);
    }
  }
  /* Evaluate the second stage of integral gradients with the 2-stage method:
    drdy|t_{n+1}*S(t_{n+1}) + drdp|t_{n+1}
  */
  if (ts->vecs_integral_sensi || ts->vecs_integral_sensip) {
    ierr = TSAdjointComputeDRDYFunction(ts,th->stage_time,th->X,ts->vecs_drdy);CHKERRQ(ierr);
  }
  if (ts->vecs_integral_sensip) {
    ierr = TSAdjointComputeDRDPFunction(ts,th->stage_time,th->X,ts->vecs_drdp);CHKERRQ(ierr);
    for (ncost=0; ncost<ts->numcost; ncost++) {
      ierr = TSThetaIntegrandDerivative(ts,ts->num_parameters,ts->vecs_drdy[ncost],&ts->vecs_fwdsensipacked[ts->num_initialvalues],ts->vecs_drdp[ncost],th->VecIntegralSensipTemp);CHKERRQ(ierr);
      if (th->endpoint) {
        ierr = VecAXPY(ts->vecs_integral_sensip[ncost],th->time_step*th->Theta,th->VecIntegralSensipTemp);CHKERRQ(ierr);
      } else {
        ierr = VecAXPY(ts->vecs_integral_sensip[ncost],th->time_step,th->VecIntegralSensipTemp);CHKERRQ(ierr);
      }
    }
  }
  if (ts->vecs_integral_sensi) {
    for (ncost=0; ncost<ts->numcost; ncost++) {
      ierr = TSThetaIntegrandDerivative(ts,ts->num_initialvalues,ts->vecs_drdy[ncost],ts->vecs_fwdsensipacked,NULL,th->VecIntegralSensiTemp);CHKERRQ(ierr);
      if (th->endpoint) {
        ierr = VecAXPY(ts->vecs_integral_sensi[ncost],th->time_step*th->Theta,th->VecIntegralSensiTemp);CHKERRQ(ierr);
      } else {
        ierr = VecAXPY(ts->vecs_integral_sensi[ncost],th->time_step,th->VecIntegralSensiTemp);CHKERRQ(ierr);
      }
    }
  }

  if (!th->endpoint) { /* VecsDeltaFwdSensip are the increment for the stage values for the 1-stage method */
    for (ntlm=0; ntlm<th->num_tlm; ntlm++) {
      ierr = VecAXPY(ts->vecs_fwdsensipacked[ntlm],(1.-th->Theta)/th->Theta,VecsDeltaFwdSensi[ntlm]);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
static PetscErrorCode TSReset_Theta(TS ts)
{
  TS_Theta       *th = (TS_Theta*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&th->X);CHKERRQ(ierr);
  ierr = VecDestroy(&th->Xdot);CHKERRQ(ierr);
  ierr = VecDestroy(&th->X0);CHKERRQ(ierr);
  ierr = VecDestroy(&th->affine);CHKERRQ(ierr);

  ierr = VecDestroy(&th->vec_sol_prev);CHKERRQ(ierr);
  ierr = VecDestroy(&th->vec_lte_work);CHKERRQ(ierr);

  ierr = VecDestroy(&th->VecCostIntegral0);CHKERRQ(ierr);
  if (ts->forward_solve) {
    ierr = VecDestroyVecs(th->num_tlm,&th->VecsDeltaFwdSensi);CHKERRQ(ierr);
    ierr = VecDestroyVecs(th->num_tlm,&th->VecsFwdSensi0);CHKERRQ(ierr);
    if (ts->vecs_integral_sensi) {
      ierr = VecDestroy(&th->VecIntegralSensiTemp);CHKERRQ(ierr);
      ierr = VecDestroyVecs(ts->numcost,&th->VecsIntegralSensi0);CHKERRQ(ierr);
    }
    if (ts->vecs_integral_sensip) {
      ierr = VecDestroy(&th->VecIntegralSensipTemp);CHKERRQ(ierr);
      ierr = VecDestroyVecs(ts->numcost,&th->VecsIntegralSensip0);CHKERRQ(ierr);
    }
  }
  ierr = VecDestroyVecs(ts->numcost,&th->VecsDeltaLam);CHKERRQ(ierr);
  ierr = VecDestroyVecs(ts->numcost,&th->VecsDeltaMu);CHKERRQ(ierr);
  ierr = VecDestroyVecs(ts->numcost,&th->VecsSensiTemp);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode TSDestroy_Theta(TS ts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSReset_Theta(ts);CHKERRQ(ierr);
  ierr = PetscFree(ts->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSThetaGetTheta_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSThetaSetTheta_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSThetaGetEndpoint_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSThetaSetEndpoint_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  This defines the nonlinear equation that is to be solved with SNES
  G(U) = F[t0+Theta*dt, U, (U-U0)*shift] = 0
*/
static PetscErrorCode SNESTSFormFunction_Theta(SNES snes,Vec x,Vec y,TS ts)
{
  TS_Theta       *th = (TS_Theta*)ts->data;
  PetscErrorCode ierr;
  Vec            X0,Xdot;
  DM             dm,dmsave;
  PetscReal      shift = 1/(th->Theta*ts->time_step);

  PetscFunctionBegin;
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  /* When using the endpoint variant, this is actually 1/Theta * Xdot */
  ierr = TSThetaGetX0AndXdot(ts,dm,&X0,&Xdot);CHKERRQ(ierr);
  ierr = VecAXPBYPCZ(Xdot,-shift,shift,0,X0,x);CHKERRQ(ierr);

  /* DM monkey-business allows user code to call TSGetDM() inside of functions evaluated on levels of FAS */
  dmsave = ts->dm;
  ts->dm = dm;
  ierr   = TSComputeIFunction(ts,th->stage_time,x,Xdot,y,PETSC_FALSE);CHKERRQ(ierr);
  ts->dm = dmsave;
  ierr   = TSThetaRestoreX0AndXdot(ts,dm,&X0,&Xdot);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESTSFormJacobian_Theta(SNES snes,Vec x,Mat A,Mat B,TS ts)
{
  TS_Theta       *th = (TS_Theta*)ts->data;
  PetscErrorCode ierr;
  Vec            Xdot;
  DM             dm,dmsave;
  PetscReal      shift = 1/(th->Theta*ts->time_step);

  PetscFunctionBegin;
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  /* Xdot has already been computed in SNESTSFormFunction_Theta (SNES guarantees this) */
  ierr = TSThetaGetX0AndXdot(ts,dm,NULL,&Xdot);CHKERRQ(ierr);

  dmsave = ts->dm;
  ts->dm = dm;
  ierr   = TSComputeIJacobian(ts,th->stage_time,x,Xdot,shift,A,B,PETSC_FALSE);CHKERRQ(ierr);
  ts->dm = dmsave;
  ierr   = TSThetaRestoreX0AndXdot(ts,dm,NULL,&Xdot);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSForwardSetUp_Theta(TS ts)
{
  TS_Theta       *th = (TS_Theta*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* combine sensitivities to parameters and sensitivities to initial values into one array */
  th->num_tlm = ts->num_parameters + ts->num_initialvalues;
  ierr = VecDuplicateVecs(ts->vecs_fwdsensipacked[0],th->num_tlm,&th->VecsDeltaFwdSensi);CHKERRQ(ierr);
  if (ts->vecs_integral_sensi) {
    ierr = VecDuplicate(ts->vecs_integral_sensi[0],&th->VecIntegralSensiTemp);CHKERRQ(ierr);
  }
  if (ts->vecs_integral_sensip) {
    ierr = VecDuplicate(ts->vecs_integral_sensip[0],&th->VecIntegralSensipTemp);CHKERRQ(ierr);
  }
  /* backup sensitivity results for roll-backs */
  ierr = VecDuplicateVecs(ts->vecs_fwdsensipacked[0],th->num_tlm,&th->VecsFwdSensi0);CHKERRQ(ierr);
  if (ts->vecs_integral_sensi) {
    ierr = VecDuplicateVecs(ts->vecs_integral_sensi[0],ts->numcost,&th->VecsIntegralSensi0);CHKERRQ(ierr);
  }
  if (ts->vecs_integral_sensip) {
    ierr = VecDuplicateVecs(ts->vecs_integral_sensip[0],ts->numcost,&th->VecsIntegralSensip0);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSSetUp_Theta(TS ts)
{
  TS_Theta       *th = (TS_Theta*)ts->data;
  PetscBool      match;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!th->VecCostIntegral0 && ts->vec_costintegral && ts->costintegralfwd) { /* back up cost integral */
    ierr = VecDuplicate(ts->vec_costintegral,&th->VecCostIntegral0);CHKERRQ(ierr);
  }
  if (!th->X) {
    ierr = VecDuplicate(ts->vec_sol,&th->X);CHKERRQ(ierr);
  }
  if (!th->Xdot) {
    ierr = VecDuplicate(ts->vec_sol,&th->Xdot);CHKERRQ(ierr);
  }
  if (!th->X0) {
    ierr = VecDuplicate(ts->vec_sol,&th->X0);CHKERRQ(ierr);
  }
  if (th->endpoint) {
    ierr = VecDuplicate(ts->vec_sol,&th->affine);CHKERRQ(ierr);
  }

  th->order = (th->Theta == 0.5) ? 2 : 1;

  ierr = TSGetDM(ts,&ts->dm);CHKERRQ(ierr);
  ierr = DMCoarsenHookAdd(ts->dm,DMCoarsenHook_TSTheta,DMRestrictHook_TSTheta,ts);CHKERRQ(ierr);
  ierr = DMSubDomainHookAdd(ts->dm,DMSubDomainHook_TSTheta,DMSubDomainRestrictHook_TSTheta,ts);CHKERRQ(ierr);

  ierr = TSGetAdapt(ts,&ts->adapt);CHKERRQ(ierr);
  ierr = TSAdaptCandidatesClear(ts->adapt);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)ts->adapt,TSADAPTNONE,&match);CHKERRQ(ierr);
  if (!match) {
    ierr = VecDuplicate(ts->vec_sol,&th->vec_sol_prev);CHKERRQ(ierr);
    ierr = VecDuplicate(ts->vec_sol,&th->vec_lte_work);CHKERRQ(ierr);
  }
  ierr = TSGetSNES(ts,&ts->snes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode TSAdjointSetUp_Theta(TS ts)
{
  TS_Theta       *th = (TS_Theta*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDuplicateVecs(ts->vecs_sensi[0],ts->numcost,&th->VecsDeltaLam);CHKERRQ(ierr);
  if(ts->vecs_sensip) {
    ierr = VecDuplicateVecs(ts->vecs_sensip[0],ts->numcost,&th->VecsDeltaMu);CHKERRQ(ierr);
  }
  ierr = VecDuplicateVecs(ts->vecs_sensi[0],ts->numcost,&th->VecsSensiTemp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSSetFromOptions_Theta(PetscOptionItems *PetscOptionsObject,TS ts)
{
  TS_Theta       *th = (TS_Theta*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"Theta ODE solver options");CHKERRQ(ierr);
  {
    ierr = PetscOptionsReal("-ts_theta_theta","Location of stage (0<Theta<=1)","TSThetaSetTheta",th->Theta,&th->Theta,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-ts_theta_endpoint","Use the endpoint instead of midpoint form of the Theta method","TSThetaSetEndpoint",th->endpoint,&th->endpoint,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-ts_theta_initial_guess_extrapolate","Extrapolate stage initial guess from previous solution (sometimes unstable)","TSThetaSetExtrapolate",th->extrapolate,&th->extrapolate,NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSView_Theta(TS ts,PetscViewer viewer)
{
  TS_Theta       *th = (TS_Theta*)ts->data;
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  Theta=%g\n",(double)th->Theta);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  Extrapolation=%s\n",th->extrapolate ? "yes" : "no");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSThetaGetTheta_Theta(TS ts,PetscReal *theta)
{
  TS_Theta *th = (TS_Theta*)ts->data;

  PetscFunctionBegin;
  *theta = th->Theta;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSThetaSetTheta_Theta(TS ts,PetscReal theta)
{
  TS_Theta *th = (TS_Theta*)ts->data;

  PetscFunctionBegin;
  if (theta <= 0 || 1 < theta) SETERRQ1(PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_OUTOFRANGE,"Theta %g not in range (0,1]",(double)theta);
  th->Theta = theta;
  th->order = (th->Theta == 0.5) ? 2 : 1;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSThetaGetEndpoint_Theta(TS ts,PetscBool *endpoint)
{
  TS_Theta *th = (TS_Theta*)ts->data;

  PetscFunctionBegin;
  *endpoint = th->endpoint;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSThetaSetEndpoint_Theta(TS ts,PetscBool flg)
{
  TS_Theta *th = (TS_Theta*)ts->data;

  PetscFunctionBegin;
  th->endpoint = flg;
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_COMPLEX)
static PetscErrorCode TSComputeLinearStability_Theta(TS ts,PetscReal xr,PetscReal xi,PetscReal *yr,PetscReal *yi)
{
  PetscComplex   z   = xr + xi*PETSC_i,f;
  TS_Theta       *th = (TS_Theta*)ts->data;
  const PetscReal one = 1.0;

  PetscFunctionBegin;
  f   = (one + (one - th->Theta)*z)/(one - th->Theta*z);
  *yr = PetscRealPartComplex(f);
  *yi = PetscImaginaryPartComplex(f);
  PetscFunctionReturn(0);
}
#endif

static PetscErrorCode TSGetStages_Theta(TS ts,PetscInt *ns,Vec **Y)
{
  TS_Theta     *th = (TS_Theta*)ts->data;

  PetscFunctionBegin;
  if (ns) *ns = 1;
  if (Y)  *Y  = th->endpoint ? &(th->X0) : &(th->X);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------ */
/*MC
      TSTHETA - DAE solver using the implicit Theta method

   Level: beginner

   Options Database:
+  -ts_theta_theta <Theta> - Location of stage (0<Theta<=1)
.  -ts_theta_endpoint <flag> - Use the endpoint (like Crank-Nicholson) instead of midpoint form of the Theta method
-  -ts_theta_initial_guess_extrapolate <flg> - Extrapolate stage initial guess from previous solution (sometimes unstable)

   Notes:
$  -ts_type theta -ts_theta_theta 1.0 corresponds to backward Euler (TSBEULER)
$  -ts_type theta -ts_theta_theta 0.5 corresponds to the implicit midpoint rule
$  -ts_type theta -ts_theta_theta 0.5 -ts_theta_endpoint corresponds to Crank-Nicholson (TSCN)

   This method can be applied to DAE.

   This method is cast as a 1-stage implicit Runge-Kutta method.

.vb
  Theta | Theta
  -------------
        |  1
.ve

   For the default Theta=0.5, this is also known as the implicit midpoint rule.

   When the endpoint variant is chosen, the method becomes a 2-stage method with first stage explicit:

.vb
  0 | 0         0
  1 | 1-Theta   Theta
  -------------------
    | 1-Theta   Theta
.ve

   For the default Theta=0.5, this is the trapezoid rule (also known as Crank-Nicolson, see TSCN).

   To apply a diagonally implicit RK method to DAE, the stage formula

$  Y_i = X + h sum_j a_ij Y'_j

   is interpreted as a formula for Y'_i in terms of Y_i and known values (Y'_j, j<i)

.seealso:  TSCreate(), TS, TSSetType(), TSCN, TSBEULER, TSThetaSetTheta(), TSThetaSetEndpoint()

M*/
PETSC_EXTERN PetscErrorCode TSCreate_Theta(TS ts)
{
  TS_Theta       *th;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ts->ops->reset           = TSReset_Theta;
  ts->ops->destroy         = TSDestroy_Theta;
  ts->ops->view            = TSView_Theta;
  ts->ops->setup           = TSSetUp_Theta;
  ts->ops->adjointsetup    = TSAdjointSetUp_Theta;
  ts->ops->step            = TSStep_Theta;
  ts->ops->interpolate     = TSInterpolate_Theta;
  ts->ops->evaluatewlte    = TSEvaluateWLTE_Theta;
  ts->ops->rollback        = TSRollBack_Theta;
  ts->ops->setfromoptions  = TSSetFromOptions_Theta;
  ts->ops->snesfunction    = SNESTSFormFunction_Theta;
  ts->ops->snesjacobian    = SNESTSFormJacobian_Theta;
#if defined(PETSC_HAVE_COMPLEX)
  ts->ops->linearstability = TSComputeLinearStability_Theta;
#endif
  ts->ops->getstages       = TSGetStages_Theta;
  ts->ops->adjointstep     = TSAdjointStep_Theta;
  ts->ops->adjointintegral = TSAdjointCostIntegral_Theta;
  ts->ops->forwardintegral = TSForwardCostIntegral_Theta;
  ts->default_adapt_type   = TSADAPTNONE;
  ts->ops->forwardsetup    = TSForwardSetUp_Theta;
  ts->ops->forwardstep     = TSForwardStep_Theta;

  ts->usessnes = PETSC_TRUE;

  ierr = PetscNewLog(ts,&th);CHKERRQ(ierr);
  ts->data = (void*)th;

  th->VecsDeltaLam   = NULL;
  th->VecsDeltaMu    = NULL;
  th->VecsSensiTemp  = NULL;

  th->extrapolate = PETSC_FALSE;
  th->Theta       = 0.5;
  th->order       = 2;
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSThetaGetTheta_C",TSThetaGetTheta_Theta);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSThetaSetTheta_C",TSThetaSetTheta_Theta);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSThetaGetEndpoint_C",TSThetaGetEndpoint_Theta);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSThetaSetEndpoint_C",TSThetaSetEndpoint_Theta);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  TSThetaGetTheta - Get the abscissa of the stage in (0,1].

  Not Collective

  Input Parameter:
.  ts - timestepping context

  Output Parameter:
.  theta - stage abscissa

  Note:
  Use of this function is normally only required to hack TSTHETA to use a modified integration scheme.

  Level: Advanced

.seealso: TSThetaSetTheta()
@*/
PetscErrorCode  TSThetaGetTheta(TS ts,PetscReal *theta)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidPointer(theta,2);
  ierr = PetscUseMethod(ts,"TSThetaGetTheta_C",(TS,PetscReal*),(ts,theta));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  TSThetaSetTheta - Set the abscissa of the stage in (0,1].

  Not Collective

  Input Parameter:
+  ts - timestepping context
-  theta - stage abscissa

  Options Database:
.  -ts_theta_theta <theta>

  Level: Intermediate

.seealso: TSThetaGetTheta()
@*/
PetscErrorCode  TSThetaSetTheta(TS ts,PetscReal theta)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = PetscTryMethod(ts,"TSThetaSetTheta_C",(TS,PetscReal),(ts,theta));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  TSThetaGetEndpoint - Gets whether to use the endpoint variant of the method (e.g. trapezoid/Crank-Nicolson instead of midpoint rule).

  Not Collective

  Input Parameter:
.  ts - timestepping context

  Output Parameter:
.  endpoint - PETSC_TRUE when using the endpoint variant

  Level: Advanced

.seealso: TSThetaSetEndpoint(), TSTHETA, TSCN
@*/
PetscErrorCode TSThetaGetEndpoint(TS ts,PetscBool *endpoint)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidPointer(endpoint,2);
  ierr = PetscUseMethod(ts,"TSThetaGetEndpoint_C",(TS,PetscBool*),(ts,endpoint));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  TSThetaSetEndpoint - Sets whether to use the endpoint variant of the method (e.g. trapezoid/Crank-Nicolson instead of midpoint rule).

  Not Collective

  Input Parameter:
+  ts - timestepping context
-  flg - PETSC_TRUE to use the endpoint variant

  Options Database:
.  -ts_theta_endpoint <flg>

  Level: Intermediate

.seealso: TSTHETA, TSCN
@*/
PetscErrorCode TSThetaSetEndpoint(TS ts,PetscBool flg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = PetscTryMethod(ts,"TSThetaSetEndpoint_C",(TS,PetscBool),(ts,flg));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
 * TSBEULER and TSCN are straightforward specializations of TSTHETA.
 * The creation functions for these specializations are below.
 */

static PetscErrorCode TSSetUp_BEuler(TS ts)
{
  TS_Theta       *th = (TS_Theta*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (th->Theta != 1.0) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_OPT_OVERWRITE,"Can not change the default value (1) of theta when using backward Euler\n");
  if (th->endpoint) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_OPT_OVERWRITE,"Can not change to the endpoint form of the Theta methods when using backward Euler\n");
  ierr = TSSetUp_Theta(ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSView_BEuler(TS ts,PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/*MC
      TSBEULER - ODE solver using the implicit backward Euler method

  Level: beginner

  Notes:
  TSBEULER is equivalent to TSTHETA with Theta=1.0

$  -ts_type theta -ts_theta_theta 1.0

.seealso:  TSCreate(), TS, TSSetType(), TSEULER, TSCN, TSTHETA

M*/
PETSC_EXTERN PetscErrorCode TSCreate_BEuler(TS ts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSCreate_Theta(ts);CHKERRQ(ierr);
  ierr = TSThetaSetTheta(ts,1.0);CHKERRQ(ierr);
  ierr = TSThetaSetEndpoint(ts,PETSC_FALSE);CHKERRQ(ierr);
  ts->ops->setup = TSSetUp_BEuler;
  ts->ops->view  = TSView_BEuler;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSSetUp_CN(TS ts)
{
  TS_Theta       *th = (TS_Theta*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (th->Theta != 0.5) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_OPT_OVERWRITE,"Can not change the default value (0.5) of theta when using Crank-Nicolson\n");
  if (!th->endpoint) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_OPT_OVERWRITE,"Can not change to the midpoint form of the Theta methods when using Crank-Nicolson\n");
  ierr = TSSetUp_Theta(ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSView_CN(TS ts,PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/*MC
      TSCN - ODE solver using the implicit Crank-Nicolson method.

  Level: beginner

  Notes:
  TSCN is equivalent to TSTHETA with Theta=0.5 and the "endpoint" option set. I.e.

$  -ts_type theta -ts_theta_theta 0.5 -ts_theta_endpoint

.seealso:  TSCreate(), TS, TSSetType(), TSBEULER, TSTHETA

M*/
PETSC_EXTERN PetscErrorCode TSCreate_CN(TS ts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSCreate_Theta(ts);CHKERRQ(ierr);
  ierr = TSThetaSetTheta(ts,0.5);CHKERRQ(ierr);
  ierr = TSThetaSetEndpoint(ts,PETSC_TRUE);CHKERRQ(ierr);
  ts->ops->setup = TSSetUp_CN;
  ts->ops->view  = TSView_CN;
  PetscFunctionReturn(0);
}
