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
  Vec          Stages[2];                 /* Storage for stage solutions */
  Vec          X0,X,Xdot;                /* Storage for u^n, u^n + dt a_{11} k_1, and time derivative u^{n+1}_t */
  Vec          affine;                   /* Affine vector needed for residual at beginning of step in endpoint formulation */
  PetscReal    Theta;
  PetscReal    shift;                    /* Shift parameter for SNES Jacobian, used by forward, TLM and adjoint */
  PetscInt     order;
  PetscBool    endpoint;
  PetscBool    extrapolate;
  TSStepStatus status;
  Vec          VecCostIntegral0;         /* Backup for roll-backs due to events, used by cost integral */
  PetscReal    ptime0;                   /* Backup for ts->ptime, the start time of current time step, used by TLM and cost integral */
  PetscReal    time_step0;               /* Backup for ts->timestep, the step size of current time step, used by TLM and cost integral*/

  /* context for sensitivity analysis */
  PetscInt     num_tlm;                  /* Total number of tangent linear equations */
  Vec          *VecsDeltaLam;            /* Increment of the adjoint sensitivity w.r.t IC at stage */
  Vec          *VecsDeltaMu;             /* Increment of the adjoint sensitivity w.r.t P at stage */
  Vec          *VecsSensiTemp;           /* Vector to be multiplied with Jacobian transpose */
  Mat          MatFwdStages[2];          /* TLM Stages */
  Mat          MatDeltaFwdSensip;        /* Increment of the forward sensitivity at stage */
  Vec          VecDeltaFwdSensipCol;     /* Working vector for holding one column of the sensitivity matrix */
  Mat          MatFwdSensip0;            /* backup for roll-backs due to events */
  Mat          MatIntegralSensipTemp;    /* Working vector for forward integral sensitivity */
  Mat          MatIntegralSensip0;       /* backup for roll-backs due to events */
  Vec          *VecsDeltaLam2;           /* Increment of the 2nd-order adjoint sensitivity w.r.t IC at stage */
  Vec          *VecsDeltaMu2;            /* Increment of the 2nd-order adjoint sensitivity w.r.t P at stage */
  Vec          *VecsSensi2Temp;          /* Working vectors that holds the residual for the second-order adjoint */
  Vec          *VecsAffine;              /* Working vectors to store residuals */
  /* context for error estimation */
  Vec          vec_sol_prev;
  Vec          vec_lte_work;
} TS_Theta;

static PetscErrorCode TSThetaGetX0AndXdot(TS ts,DM dm,Vec *X0,Vec *Xdot)
{
  TS_Theta       *th = (TS_Theta*)ts->data;

  PetscFunctionBegin;
  if (X0) {
    if (dm && dm != ts->dm) {
      PetscCall(DMGetNamedGlobalVector(dm,"TSTheta_X0",X0));
    } else *X0 = ts->vec_sol;
  }
  if (Xdot) {
    if (dm && dm != ts->dm) {
      PetscCall(DMGetNamedGlobalVector(dm,"TSTheta_Xdot",Xdot));
    } else *Xdot = th->Xdot;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSThetaRestoreX0AndXdot(TS ts,DM dm,Vec *X0,Vec *Xdot)
{
  PetscFunctionBegin;
  if (X0) {
    if (dm && dm != ts->dm) {
      PetscCall(DMRestoreNamedGlobalVector(dm,"TSTheta_X0",X0));
    }
  }
  if (Xdot) {
    if (dm && dm != ts->dm) {
      PetscCall(DMRestoreNamedGlobalVector(dm,"TSTheta_Xdot",Xdot));
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
  Vec            X0,Xdot,X0_c,Xdot_c;

  PetscFunctionBegin;
  PetscCall(TSThetaGetX0AndXdot(ts,fine,&X0,&Xdot));
  PetscCall(TSThetaGetX0AndXdot(ts,coarse,&X0_c,&Xdot_c));
  PetscCall(MatRestrict(restrct,X0,X0_c));
  PetscCall(MatRestrict(restrct,Xdot,Xdot_c));
  PetscCall(VecPointwiseMult(X0_c,rscale,X0_c));
  PetscCall(VecPointwiseMult(Xdot_c,rscale,Xdot_c));
  PetscCall(TSThetaRestoreX0AndXdot(ts,fine,&X0,&Xdot));
  PetscCall(TSThetaRestoreX0AndXdot(ts,coarse,&X0_c,&Xdot_c));
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
  Vec            X0,Xdot,X0_sub,Xdot_sub;

  PetscFunctionBegin;
  PetscCall(TSThetaGetX0AndXdot(ts,dm,&X0,&Xdot));
  PetscCall(TSThetaGetX0AndXdot(ts,subdm,&X0_sub,&Xdot_sub));

  PetscCall(VecScatterBegin(gscat,X0,X0_sub,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(gscat,X0,X0_sub,INSERT_VALUES,SCATTER_FORWARD));

  PetscCall(VecScatterBegin(gscat,Xdot,Xdot_sub,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(gscat,Xdot,Xdot_sub,INSERT_VALUES,SCATTER_FORWARD));

  PetscCall(TSThetaRestoreX0AndXdot(ts,dm,&X0,&Xdot));
  PetscCall(TSThetaRestoreX0AndXdot(ts,subdm,&X0_sub,&Xdot_sub));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSThetaEvaluateCostIntegral(TS ts)
{
  TS_Theta       *th = (TS_Theta*)ts->data;
  TS             quadts = ts->quadraturets;

  PetscFunctionBegin;
  if (th->endpoint) {
    /* Evolve ts->vec_costintegral to compute integrals */
    if (th->Theta!=1.0) {
      PetscCall(TSComputeRHSFunction(quadts,th->ptime0,th->X0,ts->vec_costintegrand));
      PetscCall(VecAXPY(quadts->vec_sol,th->time_step0*(1.0-th->Theta),ts->vec_costintegrand));
    }
    PetscCall(TSComputeRHSFunction(quadts,ts->ptime,ts->vec_sol,ts->vec_costintegrand));
    PetscCall(VecAXPY(quadts->vec_sol,th->time_step0*th->Theta,ts->vec_costintegrand));
  } else {
    PetscCall(TSComputeRHSFunction(quadts,th->stage_time,th->X,ts->vec_costintegrand));
    PetscCall(VecAXPY(quadts->vec_sol,th->time_step0,ts->vec_costintegrand));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSForwardCostIntegral_Theta(TS ts)
{
  TS_Theta       *th = (TS_Theta*)ts->data;
  TS             quadts = ts->quadraturets;

  PetscFunctionBegin;
  /* backup cost integral */
  PetscCall(VecCopy(quadts->vec_sol,th->VecCostIntegral0));
  PetscCall(TSThetaEvaluateCostIntegral(ts));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSAdjointCostIntegral_Theta(TS ts)
{
  TS_Theta       *th = (TS_Theta*)ts->data;

  PetscFunctionBegin;
  /* Like TSForwardCostIntegral(), the adjoint cost integral evaluation relies on ptime0 and time_step0. */
  th->ptime0     = ts->ptime + ts->time_step;
  th->time_step0 = -ts->time_step;
  PetscCall(TSThetaEvaluateCostIntegral(ts));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSTheta_SNESSolve(TS ts,Vec b,Vec x)
{
  PetscInt       nits,lits;

  PetscFunctionBegin;
  PetscCall(SNESSolve(ts->snes,b,x));
  PetscCall(SNESGetIterationNumber(ts->snes,&nits));
  PetscCall(SNESGetLinearSolveIterations(ts->snes,&lits));
  ts->snes_its += nits; ts->ksp_its += lits;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSStep_Theta(TS ts)
{
  TS_Theta       *th = (TS_Theta*)ts->data;
  PetscInt       rejections = 0;
  PetscBool      stageok,accept = PETSC_TRUE;
  PetscReal      next_time_step = ts->time_step;

  PetscFunctionBegin;
  if (!ts->steprollback) {
    if (th->vec_sol_prev) PetscCall(VecCopy(th->X0,th->vec_sol_prev));
    PetscCall(VecCopy(ts->vec_sol,th->X0));
  }

  th->status     = TS_STEP_INCOMPLETE;
  while (!ts->reason && th->status != TS_STEP_COMPLETE) {
    th->shift      = 1/(th->Theta*ts->time_step);
    th->stage_time = ts->ptime + (th->endpoint ? (PetscReal)1 : th->Theta)*ts->time_step;
    PetscCall(VecCopy(th->X0,th->X));
    if (th->extrapolate && !ts->steprestart) {
      PetscCall(VecAXPY(th->X,1/th->shift,th->Xdot));
    }
    if (th->endpoint) { /* This formulation assumes linear time-independent mass matrix */
      if (!th->affine) PetscCall(VecDuplicate(ts->vec_sol,&th->affine));
      PetscCall(VecZeroEntries(th->Xdot));
      PetscCall(TSComputeIFunction(ts,ts->ptime,th->X0,th->Xdot,th->affine,PETSC_FALSE));
      PetscCall(VecScale(th->affine,(th->Theta-1)/th->Theta));
    } else if (th->affine) { /* Just in case th->endpoint is changed between calls to TSStep_Theta() */
      PetscCall(VecZeroEntries(th->affine));
    }
    PetscCall(TSPreStage(ts,th->stage_time));
    PetscCall(TSTheta_SNESSolve(ts,th->affine,th->X));
    PetscCall(TSPostStage(ts,th->stage_time,0,&th->X));
    PetscCall(TSAdaptCheckStage(ts->adapt,ts,th->stage_time,th->X,&stageok));
    if (!stageok) goto reject_step;

    th->status = TS_STEP_PENDING;
    if (th->endpoint) {
      PetscCall(VecCopy(th->X,ts->vec_sol));
    } else {
      PetscCall(VecAXPBYPCZ(th->Xdot,-th->shift,th->shift,0,th->X0,th->X));
      PetscCall(VecAXPY(ts->vec_sol,ts->time_step,th->Xdot));
    }
    PetscCall(TSAdaptChoose(ts->adapt,ts,ts->time_step,NULL,&next_time_step,&accept));
    th->status = accept ? TS_STEP_COMPLETE : TS_STEP_INCOMPLETE;
    if (!accept) {
      PetscCall(VecCopy(th->X0,ts->vec_sol));
      ts->time_step = next_time_step;
      goto reject_step;
    }

    if (ts->forward_solve || ts->costintegralfwd) { /* Save the info for the later use in cost integral evaluation */
      th->ptime0     = ts->ptime;
      th->time_step0 = ts->time_step;
    }
    ts->ptime += ts->time_step;
    ts->time_step = next_time_step;
    break;

  reject_step:
    ts->reject++; accept = PETSC_FALSE;
    if (!ts->reason && ++rejections > ts->max_reject && ts->max_reject >= 0) {
      ts->reason = TS_DIVERGED_STEP_REJECTED;
      PetscCall(PetscInfo(ts,"Step=%" PetscInt_FMT ", step rejections %" PetscInt_FMT " greater than current TS allowed, stopping solve\n",ts->steps,rejections));
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSAdjointStepBEuler_Private(TS ts)
{
  TS_Theta       *th = (TS_Theta*)ts->data;
  TS             quadts = ts->quadraturets;
  Vec            *VecsDeltaLam = th->VecsDeltaLam,*VecsDeltaMu = th->VecsDeltaMu,*VecsSensiTemp = th->VecsSensiTemp;
  Vec            *VecsDeltaLam2 = th->VecsDeltaLam2,*VecsDeltaMu2 = th->VecsDeltaMu2,*VecsSensi2Temp = th->VecsSensi2Temp;
  PetscInt       nadj;
  Mat            J,Jpre,quadJ = NULL,quadJp = NULL;
  KSP            ksp;
  PetscScalar    *xarr;
  TSEquationType eqtype;
  PetscBool      isexplicitode = PETSC_FALSE;
  PetscReal      adjoint_time_step;

  PetscFunctionBegin;
  PetscCall(TSGetEquationType(ts,&eqtype));
  if (eqtype == TS_EQ_ODE_EXPLICIT) {
    isexplicitode  = PETSC_TRUE;
    VecsDeltaLam  = ts->vecs_sensi;
    VecsDeltaLam2 = ts->vecs_sensi2;
  }
  th->status = TS_STEP_INCOMPLETE;
  PetscCall(SNESGetKSP(ts->snes,&ksp));
  PetscCall(TSGetIJacobian(ts,&J,&Jpre,NULL,NULL));
  if (quadts) {
    PetscCall(TSGetRHSJacobian(quadts,&quadJ,NULL,NULL,NULL));
    PetscCall(TSGetRHSJacobianP(quadts,&quadJp,NULL,NULL));
  }

  th->stage_time    = ts->ptime;
  adjoint_time_step = -ts->time_step; /* always positive since time_step is negative */

  /* Build RHS for first-order adjoint lambda_{n+1}/h + r_u^T(n+1) */
  if (quadts) {
    PetscCall(TSComputeRHSJacobian(quadts,th->stage_time,ts->vec_sol,quadJ,NULL));
  }

  for (nadj=0; nadj<ts->numcost; nadj++) {
    PetscCall(VecCopy(ts->vecs_sensi[nadj],VecsSensiTemp[nadj]));
    PetscCall(VecScale(VecsSensiTemp[nadj],1./adjoint_time_step)); /* lambda_{n+1}/h */
    if (quadJ) {
      PetscCall(MatDenseGetColumn(quadJ,nadj,&xarr));
      PetscCall(VecPlaceArray(ts->vec_drdu_col,xarr));
      PetscCall(VecAXPY(VecsSensiTemp[nadj],1.,ts->vec_drdu_col));
      PetscCall(VecResetArray(ts->vec_drdu_col));
      PetscCall(MatDenseRestoreColumn(quadJ,&xarr));
    }
  }

  /* Build LHS for first-order adjoint */
  th->shift = 1./adjoint_time_step;
  PetscCall(TSComputeSNESJacobian(ts,ts->vec_sol,J,Jpre));
  PetscCall(KSPSetOperators(ksp,J,Jpre));

  /* Solve stage equation LHS*lambda_s = RHS for first-order adjoint */
  for (nadj=0; nadj<ts->numcost; nadj++) {
    KSPConvergedReason kspreason;
    PetscCall(KSPSolveTranspose(ksp,VecsSensiTemp[nadj],VecsDeltaLam[nadj]));
    PetscCall(KSPGetConvergedReason(ksp,&kspreason));
    if (kspreason < 0) {
      ts->reason = TSADJOINT_DIVERGED_LINEAR_SOLVE;
      PetscCall(PetscInfo(ts,"Step=%" PetscInt_FMT ", %" PetscInt_FMT "th cost function, transposed linear solve fails, stopping 1st-order adjoint solve\n",ts->steps,nadj));
    }
  }

  if (ts->vecs_sensi2) { /* U_{n+1} */
    /* Get w1 at t_{n+1} from TLM matrix */
    PetscCall(MatDenseGetColumn(ts->mat_sensip,0,&xarr));
    PetscCall(VecPlaceArray(ts->vec_sensip_col,xarr));
    /* lambda_s^T F_UU w_1 */
    PetscCall(TSComputeIHessianProductFunctionUU(ts,th->stage_time,ts->vec_sol,VecsDeltaLam,ts->vec_sensip_col,ts->vecs_fuu));
    /* lambda_s^T F_UP w_2 */
    PetscCall(TSComputeIHessianProductFunctionUP(ts,th->stage_time,ts->vec_sol,VecsDeltaLam,ts->vec_dir,ts->vecs_fup));
    for (nadj=0; nadj<ts->numcost; nadj++) { /* compute the residual */
      PetscCall(VecCopy(ts->vecs_sensi2[nadj],VecsSensi2Temp[nadj]));
      PetscCall(VecScale(VecsSensi2Temp[nadj],1./adjoint_time_step));
      PetscCall(VecAXPY(VecsSensi2Temp[nadj],-1.,ts->vecs_fuu[nadj]));
      if (ts->vecs_fup) {
        PetscCall(VecAXPY(VecsSensi2Temp[nadj],-1.,ts->vecs_fup[nadj]));
      }
    }
    /* Solve stage equation LHS X = RHS for second-order adjoint */
    for (nadj=0; nadj<ts->numcost; nadj++) {
      KSPConvergedReason kspreason;
      PetscCall(KSPSolveTranspose(ksp,VecsSensi2Temp[nadj],VecsDeltaLam2[nadj]));
      PetscCall(KSPGetConvergedReason(ksp,&kspreason));
      if (kspreason < 0) {
        ts->reason = TSADJOINT_DIVERGED_LINEAR_SOLVE;
        PetscCall(PetscInfo(ts,"Step=%" PetscInt_FMT ", %" PetscInt_FMT "th cost function, transposed linear solve fails, stopping 2nd-order adjoint solve\n",ts->steps,nadj));
      }
    }
  }

  /* Update sensitivities, and evaluate integrals if there is any */
  if (!isexplicitode) {
    th->shift = 0.0;
    PetscCall(TSComputeSNESJacobian(ts,ts->vec_sol,J,Jpre));
    PetscCall(KSPSetOperators(ksp,J,Jpre));
    for (nadj=0; nadj<ts->numcost; nadj++) {
      /* Add f_U \lambda_s to the original RHS */
      PetscCall(VecScale(VecsSensiTemp[nadj],-1.));
      PetscCall(MatMultTransposeAdd(J,VecsDeltaLam[nadj],VecsSensiTemp[nadj],VecsSensiTemp[nadj]));
      PetscCall(VecScale(VecsSensiTemp[nadj],-adjoint_time_step));
      PetscCall(VecCopy(VecsSensiTemp[nadj],ts->vecs_sensi[nadj]));
      if (ts->vecs_sensi2) {
        PetscCall(MatMultTransposeAdd(J,VecsDeltaLam2[nadj],VecsSensi2Temp[nadj],VecsSensi2Temp[nadj]));
        PetscCall(VecScale(VecsSensi2Temp[nadj],-adjoint_time_step));
        PetscCall(VecCopy(VecsSensi2Temp[nadj],ts->vecs_sensi2[nadj]));
      }
    }
  }
  if (ts->vecs_sensip) {
    PetscCall(VecAXPBYPCZ(th->Xdot,-1./adjoint_time_step,1.0/adjoint_time_step,0,th->X0,ts->vec_sol));
    PetscCall(TSComputeIJacobianP(ts,th->stage_time,ts->vec_sol,th->Xdot,1./adjoint_time_step,ts->Jacp,PETSC_FALSE)); /* get -f_p */
    if (quadts) {
      PetscCall(TSComputeRHSJacobianP(quadts,th->stage_time,ts->vec_sol,quadJp));
    }
    if (ts->vecs_sensi2p) {
      /* lambda_s^T F_PU w_1 */
      PetscCall(TSComputeIHessianProductFunctionPU(ts,th->stage_time,ts->vec_sol,VecsDeltaLam,ts->vec_sensip_col,ts->vecs_fpu));
      /* lambda_s^T F_PP w_2 */
      PetscCall(TSComputeIHessianProductFunctionPP(ts,th->stage_time,ts->vec_sol,VecsDeltaLam,ts->vec_dir,ts->vecs_fpp));
    }

    for (nadj=0; nadj<ts->numcost; nadj++) {
      PetscCall(MatMultTranspose(ts->Jacp,VecsDeltaLam[nadj],VecsDeltaMu[nadj]));
      PetscCall(VecAXPY(ts->vecs_sensip[nadj],-adjoint_time_step,VecsDeltaMu[nadj]));
      if (quadJp) {
        PetscCall(MatDenseGetColumn(quadJp,nadj,&xarr));
        PetscCall(VecPlaceArray(ts->vec_drdp_col,xarr));
        PetscCall(VecAXPY(ts->vecs_sensip[nadj],adjoint_time_step,ts->vec_drdp_col));
        PetscCall(VecResetArray(ts->vec_drdp_col));
        PetscCall(MatDenseRestoreColumn(quadJp,&xarr));
      }
      if (ts->vecs_sensi2p) {
        PetscCall(MatMultTranspose(ts->Jacp,VecsDeltaLam2[nadj],VecsDeltaMu2[nadj]));
        PetscCall(VecAXPY(ts->vecs_sensi2p[nadj],-adjoint_time_step,VecsDeltaMu2[nadj]));
        if (ts->vecs_fpu) {
          PetscCall(VecAXPY(ts->vecs_sensi2p[nadj],-adjoint_time_step,ts->vecs_fpu[nadj]));
        }
        if (ts->vecs_fpp) {
          PetscCall(VecAXPY(ts->vecs_sensi2p[nadj],-adjoint_time_step,ts->vecs_fpp[nadj]));
        }
      }
    }
  }

  if (ts->vecs_sensi2) {
    PetscCall(VecResetArray(ts->vec_sensip_col));
    PetscCall(MatDenseRestoreColumn(ts->mat_sensip,&xarr));
  }
  th->status = TS_STEP_COMPLETE;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSAdjointStep_Theta(TS ts)
{
  TS_Theta       *th = (TS_Theta*)ts->data;
  TS             quadts = ts->quadraturets;
  Vec            *VecsDeltaLam = th->VecsDeltaLam,*VecsDeltaMu = th->VecsDeltaMu,*VecsSensiTemp = th->VecsSensiTemp;
  Vec            *VecsDeltaLam2 = th->VecsDeltaLam2,*VecsDeltaMu2 = th->VecsDeltaMu2,*VecsSensi2Temp = th->VecsSensi2Temp;
  PetscInt       nadj;
  Mat            J,Jpre,quadJ = NULL,quadJp = NULL;
  KSP            ksp;
  PetscScalar    *xarr;
  PetscReal      adjoint_time_step;
  PetscReal      adjoint_ptime; /* end time of the adjoint time step (ts->ptime is the start time, ususally ts->ptime is larger than adjoint_ptime) */

  PetscFunctionBegin;
  if (th->Theta == 1.) {
    PetscCall(TSAdjointStepBEuler_Private(ts));
    PetscFunctionReturn(0);
  }
  th->status = TS_STEP_INCOMPLETE;
  PetscCall(SNESGetKSP(ts->snes,&ksp));
  PetscCall(TSGetIJacobian(ts,&J,&Jpre,NULL,NULL));
  if (quadts) {
    PetscCall(TSGetRHSJacobian(quadts,&quadJ,NULL,NULL,NULL));
    PetscCall(TSGetRHSJacobianP(quadts,&quadJp,NULL,NULL));
  }
  /* If endpoint=1, th->ptime and th->X0 will be used; if endpoint=0, th->stage_time and th->X will be used. */
  th->stage_time    = th->endpoint ? ts->ptime : (ts->ptime+(1.-th->Theta)*ts->time_step);
  adjoint_ptime     = ts->ptime + ts->time_step;
  adjoint_time_step = -ts->time_step;  /* always positive since time_step is negative */

  if (!th->endpoint) {
    /* recover th->X0 using vec_sol and the stage value th->X */
    PetscCall(VecAXPBYPCZ(th->X0,1.0/(1.0-th->Theta),th->Theta/(th->Theta-1.0),0,th->X,ts->vec_sol));
  }

  /* Build RHS for first-order adjoint */
  /* Cost function has an integral term */
  if (quadts) {
    if (th->endpoint) {
      PetscCall(TSComputeRHSJacobian(quadts,th->stage_time,ts->vec_sol,quadJ,NULL));
    } else {
      PetscCall(TSComputeRHSJacobian(quadts,th->stage_time,th->X,quadJ,NULL));
    }
  }

  for (nadj=0; nadj<ts->numcost; nadj++) {
    PetscCall(VecCopy(ts->vecs_sensi[nadj],VecsSensiTemp[nadj]));
    PetscCall(VecScale(VecsSensiTemp[nadj],1./(th->Theta*adjoint_time_step)));
    if (quadJ) {
      PetscCall(MatDenseGetColumn(quadJ,nadj,&xarr));
      PetscCall(VecPlaceArray(ts->vec_drdu_col,xarr));
      PetscCall(VecAXPY(VecsSensiTemp[nadj],1.,ts->vec_drdu_col));
      PetscCall(VecResetArray(ts->vec_drdu_col));
      PetscCall(MatDenseRestoreColumn(quadJ,&xarr));
    }
  }

  /* Build LHS for first-order adjoint */
  th->shift = 1./(th->Theta*adjoint_time_step);
  if (th->endpoint) {
    PetscCall(TSComputeSNESJacobian(ts,ts->vec_sol,J,Jpre));
  } else {
    PetscCall(TSComputeSNESJacobian(ts,th->X,J,Jpre));
  }
  PetscCall(KSPSetOperators(ksp,J,Jpre));

  /* Solve stage equation LHS*lambda_s = RHS for first-order adjoint */
  for (nadj=0; nadj<ts->numcost; nadj++) {
    KSPConvergedReason kspreason;
    PetscCall(KSPSolveTranspose(ksp,VecsSensiTemp[nadj],VecsDeltaLam[nadj]));
    PetscCall(KSPGetConvergedReason(ksp,&kspreason));
    if (kspreason < 0) {
      ts->reason = TSADJOINT_DIVERGED_LINEAR_SOLVE;
      PetscCall(PetscInfo(ts,"Step=%" PetscInt_FMT ", %" PetscInt_FMT "th cost function, transposed linear solve fails, stopping 1st-order adjoint solve\n",ts->steps,nadj));
    }
  }

  /* Second-order adjoint */
  if (ts->vecs_sensi2) { /* U_{n+1} */
    PetscCheck(th->endpoint,PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"Operation not implemented in TS_Theta");
    /* Get w1 at t_{n+1} from TLM matrix */
    PetscCall(MatDenseGetColumn(ts->mat_sensip,0,&xarr));
    PetscCall(VecPlaceArray(ts->vec_sensip_col,xarr));
    /* lambda_s^T F_UU w_1 */
    PetscCall(TSComputeIHessianProductFunctionUU(ts,th->stage_time,ts->vec_sol,VecsDeltaLam,ts->vec_sensip_col,ts->vecs_fuu));
    PetscCall(VecResetArray(ts->vec_sensip_col));
    PetscCall(MatDenseRestoreColumn(ts->mat_sensip,&xarr));
    /* lambda_s^T F_UP w_2 */
    PetscCall(TSComputeIHessianProductFunctionUP(ts,th->stage_time,ts->vec_sol,VecsDeltaLam,ts->vec_dir,ts->vecs_fup));
    for (nadj=0; nadj<ts->numcost; nadj++) { /* compute the residual */
      PetscCall(VecCopy(ts->vecs_sensi2[nadj],VecsSensi2Temp[nadj]));
      PetscCall(VecScale(VecsSensi2Temp[nadj],th->shift));
      PetscCall(VecAXPY(VecsSensi2Temp[nadj],-1.,ts->vecs_fuu[nadj]));
      if (ts->vecs_fup) {
        PetscCall(VecAXPY(VecsSensi2Temp[nadj],-1.,ts->vecs_fup[nadj]));
      }
    }
    /* Solve stage equation LHS X = RHS for second-order adjoint */
    for (nadj=0; nadj<ts->numcost; nadj++) {
      KSPConvergedReason kspreason;
      PetscCall(KSPSolveTranspose(ksp,VecsSensi2Temp[nadj],VecsDeltaLam2[nadj]));
      PetscCall(KSPGetConvergedReason(ksp,&kspreason));
      if (kspreason < 0) {
        ts->reason = TSADJOINT_DIVERGED_LINEAR_SOLVE;
        PetscCall(PetscInfo(ts,"Step=%" PetscInt_FMT ", %" PetscInt_FMT "th cost function, transposed linear solve fails, stopping 2nd-order adjoint solve\n",ts->steps,nadj));
      }
    }
  }

  /* Update sensitivities, and evaluate integrals if there is any */
  if (th->endpoint) { /* two-stage Theta methods with th->Theta!=1, th->Theta==1 leads to BEuler */
    th->shift      = 1./((th->Theta-1.)*adjoint_time_step);
    th->stage_time = adjoint_ptime;
    PetscCall(TSComputeSNESJacobian(ts,th->X0,J,Jpre));
    PetscCall(KSPSetOperators(ksp,J,Jpre));
    /* R_U at t_n */
    if (quadts) {
      PetscCall(TSComputeRHSJacobian(quadts,adjoint_ptime,th->X0,quadJ,NULL));
    }
    for (nadj=0; nadj<ts->numcost; nadj++) {
      PetscCall(MatMultTranspose(J,VecsDeltaLam[nadj],ts->vecs_sensi[nadj]));
      if (quadJ) {
        PetscCall(MatDenseGetColumn(quadJ,nadj,&xarr));
        PetscCall(VecPlaceArray(ts->vec_drdu_col,xarr));
        PetscCall(VecAXPY(ts->vecs_sensi[nadj],-1.,ts->vec_drdu_col));
        PetscCall(VecResetArray(ts->vec_drdu_col));
        PetscCall(MatDenseRestoreColumn(quadJ,&xarr));
      }
      PetscCall(VecScale(ts->vecs_sensi[nadj],1./th->shift));
    }

    /* Second-order adjoint */
    if (ts->vecs_sensi2) { /* U_n */
      /* Get w1 at t_n from TLM matrix */
      PetscCall(MatDenseGetColumn(th->MatFwdSensip0,0,&xarr));
      PetscCall(VecPlaceArray(ts->vec_sensip_col,xarr));
      /* lambda_s^T F_UU w_1 */
      PetscCall(TSComputeIHessianProductFunctionUU(ts,adjoint_ptime,th->X0,VecsDeltaLam,ts->vec_sensip_col,ts->vecs_fuu));
      PetscCall(VecResetArray(ts->vec_sensip_col));
      PetscCall(MatDenseRestoreColumn(th->MatFwdSensip0,&xarr));
      /* lambda_s^T F_UU w_2 */
      PetscCall(TSComputeIHessianProductFunctionUP(ts,adjoint_ptime,th->X0,VecsDeltaLam,ts->vec_dir,ts->vecs_fup));
      for (nadj=0; nadj<ts->numcost; nadj++) {
        /* M^T Lambda_s + h(1-theta) F_U^T Lambda_s + h(1-theta) lambda_s^T F_UU w_1 + lambda_s^T F_UP w_2  */
        PetscCall(MatMultTranspose(J,VecsDeltaLam2[nadj],ts->vecs_sensi2[nadj]));
        PetscCall(VecAXPY(ts->vecs_sensi2[nadj],1.,ts->vecs_fuu[nadj]));
        if (ts->vecs_fup) {
          PetscCall(VecAXPY(ts->vecs_sensi2[nadj],1.,ts->vecs_fup[nadj]));
        }
        PetscCall(VecScale(ts->vecs_sensi2[nadj],1./th->shift));
      }
    }

    th->stage_time = ts->ptime; /* recover the old value */

    if (ts->vecs_sensip) { /* sensitivities wrt parameters */
      /* U_{n+1} */
      th->shift = 1.0/(adjoint_time_step*th->Theta);
      PetscCall(VecAXPBYPCZ(th->Xdot,-th->shift,th->shift,0,th->X0,ts->vec_sol));
      PetscCall(TSComputeIJacobianP(ts,th->stage_time,ts->vec_sol,th->Xdot,-1./(th->Theta*adjoint_time_step),ts->Jacp,PETSC_FALSE));
      if (quadts) {
        PetscCall(TSComputeRHSJacobianP(quadts,th->stage_time,ts->vec_sol,quadJp));
      }
      for (nadj=0; nadj<ts->numcost; nadj++) {
        PetscCall(MatMultTranspose(ts->Jacp,VecsDeltaLam[nadj],VecsDeltaMu[nadj]));
        PetscCall(VecAXPY(ts->vecs_sensip[nadj],-adjoint_time_step*th->Theta,VecsDeltaMu[nadj]));
        if (quadJp) {
          PetscCall(MatDenseGetColumn(quadJp,nadj,&xarr));
          PetscCall(VecPlaceArray(ts->vec_drdp_col,xarr));
          PetscCall(VecAXPY(ts->vecs_sensip[nadj],adjoint_time_step*th->Theta,ts->vec_drdp_col));
          PetscCall(VecResetArray(ts->vec_drdp_col));
          PetscCall(MatDenseRestoreColumn(quadJp,&xarr));
        }
      }
      if (ts->vecs_sensi2p) { /* second-order */
        /* Get w1 at t_{n+1} from TLM matrix */
        PetscCall(MatDenseGetColumn(ts->mat_sensip,0,&xarr));
        PetscCall(VecPlaceArray(ts->vec_sensip_col,xarr));
        /* lambda_s^T F_PU w_1 */
        PetscCall(TSComputeIHessianProductFunctionPU(ts,th->stage_time,ts->vec_sol,VecsDeltaLam,ts->vec_sensip_col,ts->vecs_fpu));
        PetscCall(VecResetArray(ts->vec_sensip_col));
        PetscCall(MatDenseRestoreColumn(ts->mat_sensip,&xarr));

        /* lambda_s^T F_PP w_2 */
        PetscCall(TSComputeIHessianProductFunctionPP(ts,th->stage_time,ts->vec_sol,VecsDeltaLam,ts->vec_dir,ts->vecs_fpp));
        for (nadj=0; nadj<ts->numcost; nadj++) {
          /* Mu2 <- Mu2 + h theta F_P^T Lambda_s + h theta (lambda_s^T F_UU w_1 + lambda_s^T F_UP w_2)  */
          PetscCall(MatMultTranspose(ts->Jacp,VecsDeltaLam2[nadj],VecsDeltaMu2[nadj]));
          PetscCall(VecAXPY(ts->vecs_sensi2p[nadj],-adjoint_time_step*th->Theta,VecsDeltaMu2[nadj]));
          if (ts->vecs_fpu) {
            PetscCall(VecAXPY(ts->vecs_sensi2p[nadj],-adjoint_time_step*th->Theta,ts->vecs_fpu[nadj]));
          }
          if (ts->vecs_fpp) {
            PetscCall(VecAXPY(ts->vecs_sensi2p[nadj],-adjoint_time_step*th->Theta,ts->vecs_fpp[nadj]));
          }
        }
      }

      /* U_s */
      PetscCall(VecZeroEntries(th->Xdot));
      PetscCall(TSComputeIJacobianP(ts,adjoint_ptime,th->X0,th->Xdot,1./((th->Theta-1.0)*adjoint_time_step),ts->Jacp,PETSC_FALSE));
      if (quadts) {
        PetscCall(TSComputeRHSJacobianP(quadts,adjoint_ptime,th->X0,quadJp));
      }
      for (nadj=0; nadj<ts->numcost; nadj++) {
        PetscCall(MatMultTranspose(ts->Jacp,VecsDeltaLam[nadj],VecsDeltaMu[nadj]));
        PetscCall(VecAXPY(ts->vecs_sensip[nadj],-adjoint_time_step*(1.0-th->Theta),VecsDeltaMu[nadj]));
        if (quadJp) {
          PetscCall(MatDenseGetColumn(quadJp,nadj,&xarr));
          PetscCall(VecPlaceArray(ts->vec_drdp_col,xarr));
          PetscCall(VecAXPY(ts->vecs_sensip[nadj],adjoint_time_step*(1.0-th->Theta),ts->vec_drdp_col));
          PetscCall(VecResetArray(ts->vec_drdp_col));
          PetscCall(MatDenseRestoreColumn(quadJp,&xarr));
        }
        if (ts->vecs_sensi2p) { /* second-order */
          /* Get w1 at t_n from TLM matrix */
          PetscCall(MatDenseGetColumn(th->MatFwdSensip0,0,&xarr));
          PetscCall(VecPlaceArray(ts->vec_sensip_col,xarr));
          /* lambda_s^T F_PU w_1 */
          PetscCall(TSComputeIHessianProductFunctionPU(ts,adjoint_ptime,th->X0,VecsDeltaLam,ts->vec_sensip_col,ts->vecs_fpu));
          PetscCall(VecResetArray(ts->vec_sensip_col));
          PetscCall(MatDenseRestoreColumn(th->MatFwdSensip0,&xarr));
          /* lambda_s^T F_PP w_2 */
          PetscCall(TSComputeIHessianProductFunctionPP(ts,adjoint_ptime,th->X0,VecsDeltaLam,ts->vec_dir,ts->vecs_fpp));
          for (nadj=0; nadj<ts->numcost; nadj++) {
            /* Mu2 <- Mu2 + h(1-theta) F_P^T Lambda_s + h(1-theta) (lambda_s^T F_UU w_1 + lambda_s^T F_UP w_2) */
            PetscCall(MatMultTranspose(ts->Jacp,VecsDeltaLam2[nadj],VecsDeltaMu2[nadj]));
            PetscCall(VecAXPY(ts->vecs_sensi2p[nadj],-adjoint_time_step*(1.0-th->Theta),VecsDeltaMu2[nadj]));
            if (ts->vecs_fpu) {
              PetscCall(VecAXPY(ts->vecs_sensi2p[nadj],-adjoint_time_step*(1.0-th->Theta),ts->vecs_fpu[nadj]));
            }
            if (ts->vecs_fpp) {
              PetscCall(VecAXPY(ts->vecs_sensi2p[nadj],-adjoint_time_step*(1.0-th->Theta),ts->vecs_fpp[nadj]));
            }
          }
        }
      }
    }
  } else { /* one-stage case */
    th->shift = 0.0;
    PetscCall(TSComputeSNESJacobian(ts,th->X,J,Jpre)); /* get -f_y */
    PetscCall(KSPSetOperators(ksp,J,Jpre));
    if (quadts) {
      PetscCall(TSComputeRHSJacobian(quadts,th->stage_time,th->X,quadJ,NULL));
    }
    for (nadj=0; nadj<ts->numcost; nadj++) {
      PetscCall(MatMultTranspose(J,VecsDeltaLam[nadj],VecsSensiTemp[nadj]));
      PetscCall(VecAXPY(ts->vecs_sensi[nadj],-adjoint_time_step,VecsSensiTemp[nadj]));
      if (quadJ) {
        PetscCall(MatDenseGetColumn(quadJ,nadj,&xarr));
        PetscCall(VecPlaceArray(ts->vec_drdu_col,xarr));
        PetscCall(VecAXPY(ts->vecs_sensi[nadj],adjoint_time_step,ts->vec_drdu_col));
        PetscCall(VecResetArray(ts->vec_drdu_col));
        PetscCall(MatDenseRestoreColumn(quadJ,&xarr));
      }
    }
    if (ts->vecs_sensip) {
      th->shift = 1.0/(adjoint_time_step*th->Theta);
      PetscCall(VecAXPBYPCZ(th->Xdot,-th->shift,th->shift,0,th->X0,th->X));
      PetscCall(TSComputeIJacobianP(ts,th->stage_time,th->X,th->Xdot,th->shift,ts->Jacp,PETSC_FALSE));
      if (quadts) {
        PetscCall(TSComputeRHSJacobianP(quadts,th->stage_time,th->X,quadJp));
      }
      for (nadj=0; nadj<ts->numcost; nadj++) {
        PetscCall(MatMultTranspose(ts->Jacp,VecsDeltaLam[nadj],VecsDeltaMu[nadj]));
        PetscCall(VecAXPY(ts->vecs_sensip[nadj],-adjoint_time_step,VecsDeltaMu[nadj]));
        if (quadJp) {
          PetscCall(MatDenseGetColumn(quadJp,nadj,&xarr));
          PetscCall(VecPlaceArray(ts->vec_drdp_col,xarr));
          PetscCall(VecAXPY(ts->vecs_sensip[nadj],adjoint_time_step,ts->vec_drdp_col));
          PetscCall(VecResetArray(ts->vec_drdp_col));
          PetscCall(MatDenseRestoreColumn(quadJp,&xarr));
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

  PetscFunctionBegin;
  PetscCall(VecCopy(ts->vec_sol,th->X));
  if (th->endpoint) dt *= th->Theta;
  PetscCall(VecWAXPY(X,dt,th->Xdot,th->X));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSEvaluateWLTE_Theta(TS ts,NormType wnormtype,PetscInt *order,PetscReal *wlte)
{
  TS_Theta       *th = (TS_Theta*)ts->data;
  Vec            X = ts->vec_sol;      /* X = solution */
  Vec            Y = th->vec_lte_work; /* Y = X + LTE  */
  PetscReal      wltea,wlter;

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
    PetscCall(VecCopy(X,Y));
    PetscCall(VecMAXPY(Y,3,scal,vecs));
    PetscCall(TSErrorWeightedNorm(ts,X,Y,wnormtype,wlte,&wltea,&wlter));
  }
  if (order) *order = 2;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSRollBack_Theta(TS ts)
{
  TS_Theta       *th = (TS_Theta*)ts->data;
  TS             quadts = ts->quadraturets;

  PetscFunctionBegin;
  PetscCall(VecCopy(th->X0,ts->vec_sol));
  if (quadts && ts->costintegralfwd) {
    PetscCall(VecCopy(th->VecCostIntegral0,quadts->vec_sol));
  }
  th->status = TS_STEP_INCOMPLETE;
  if (ts->mat_sensip) {
    PetscCall(MatCopy(th->MatFwdSensip0,ts->mat_sensip,SAME_NONZERO_PATTERN));
  }
  if (quadts && quadts->mat_sensip) {
    PetscCall(MatCopy(th->MatIntegralSensip0,quadts->mat_sensip,SAME_NONZERO_PATTERN));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSForwardStep_Theta(TS ts)
{
  TS_Theta       *th = (TS_Theta*)ts->data;
  TS             quadts = ts->quadraturets;
  Mat            MatDeltaFwdSensip = th->MatDeltaFwdSensip;
  Vec            VecDeltaFwdSensipCol = th->VecDeltaFwdSensipCol;
  PetscInt       ntlm;
  KSP            ksp;
  Mat            J,Jpre,quadJ = NULL,quadJp = NULL;
  PetscScalar    *barr,*xarr;
  PetscReal      previous_shift;

  PetscFunctionBegin;
  previous_shift = th->shift;
  PetscCall(MatCopy(ts->mat_sensip,th->MatFwdSensip0,SAME_NONZERO_PATTERN));

  if (quadts && quadts->mat_sensip) {
    PetscCall(MatCopy(quadts->mat_sensip,th->MatIntegralSensip0,SAME_NONZERO_PATTERN));
  }
  PetscCall(SNESGetKSP(ts->snes,&ksp));
  PetscCall(TSGetIJacobian(ts,&J,&Jpre,NULL,NULL));
  if (quadts) {
    PetscCall(TSGetRHSJacobian(quadts,&quadJ,NULL,NULL,NULL));
    PetscCall(TSGetRHSJacobianP(quadts,&quadJp,NULL,NULL));
  }

  /* Build RHS */
  if (th->endpoint) { /* 2-stage method*/
    th->shift = 1./((th->Theta-1.)*th->time_step0);
    PetscCall(TSComputeIJacobian(ts,th->ptime0,th->X0,th->Xdot,th->shift,J,Jpre,PETSC_FALSE));
    PetscCall(MatMatMult(J,ts->mat_sensip,MAT_REUSE_MATRIX,PETSC_DEFAULT,&MatDeltaFwdSensip));
    PetscCall(MatScale(MatDeltaFwdSensip,(th->Theta-1.)/th->Theta));

    /* Add the f_p forcing terms */
    if (ts->Jacp) {
      PetscCall(VecZeroEntries(th->Xdot));
      PetscCall(TSComputeIJacobianP(ts,th->ptime0,th->X0,th->Xdot,th->shift,ts->Jacp,PETSC_FALSE));
      PetscCall(MatAXPY(MatDeltaFwdSensip,(th->Theta-1.)/th->Theta,ts->Jacp,SUBSET_NONZERO_PATTERN));
      th->shift = previous_shift;
      PetscCall(VecAXPBYPCZ(th->Xdot,-th->shift,th->shift,0,th->X0,ts->vec_sol));
      PetscCall(TSComputeIJacobianP(ts,th->stage_time,ts->vec_sol,th->Xdot,th->shift,ts->Jacp,PETSC_FALSE));
      PetscCall(MatAXPY(MatDeltaFwdSensip,-1.,ts->Jacp,SUBSET_NONZERO_PATTERN));
    }
  } else { /* 1-stage method */
    th->shift = 0.0;
    PetscCall(TSComputeIJacobian(ts,th->stage_time,th->X,th->Xdot,th->shift,J,Jpre,PETSC_FALSE));
    PetscCall(MatMatMult(J,ts->mat_sensip,MAT_REUSE_MATRIX,PETSC_DEFAULT,&MatDeltaFwdSensip));
    PetscCall(MatScale(MatDeltaFwdSensip,-1.));

    /* Add the f_p forcing terms */
    if (ts->Jacp) {
      th->shift = previous_shift;
      PetscCall(VecAXPBYPCZ(th->Xdot,-th->shift,th->shift,0,th->X0,th->X));
      PetscCall(TSComputeIJacobianP(ts,th->stage_time,th->X,th->Xdot,th->shift,ts->Jacp,PETSC_FALSE));
      PetscCall(MatAXPY(MatDeltaFwdSensip,-1.,ts->Jacp,SUBSET_NONZERO_PATTERN));
    }
  }

  /* Build LHS */
  th->shift = previous_shift; /* recover the previous shift used in TSStep_Theta() */
  if (th->endpoint) {
    PetscCall(TSComputeIJacobian(ts,th->stage_time,ts->vec_sol,th->Xdot,th->shift,J,Jpre,PETSC_FALSE));
  } else {
    PetscCall(TSComputeIJacobian(ts,th->stage_time,th->X,th->Xdot,th->shift,J,Jpre,PETSC_FALSE));
  }
  PetscCall(KSPSetOperators(ksp,J,Jpre));

  /*
    Evaluate the first stage of integral gradients with the 2-stage method:
    drdu|t_n*S(t_n) + drdp|t_n
    This is done before the linear solve because the sensitivity variable S(t_n) will be propagated to S(t_{n+1})
  */
  if (th->endpoint) { /* 2-stage method only */
    if (quadts && quadts->mat_sensip) {
      PetscCall(TSComputeRHSJacobian(quadts,th->ptime0,th->X0,quadJ,NULL));
      PetscCall(TSComputeRHSJacobianP(quadts,th->ptime0,th->X0,quadJp));
      PetscCall(MatTransposeMatMult(ts->mat_sensip,quadJ,MAT_REUSE_MATRIX,PETSC_DEFAULT,&th->MatIntegralSensipTemp));
      PetscCall(MatAXPY(th->MatIntegralSensipTemp,1,quadJp,SAME_NONZERO_PATTERN));
      PetscCall(MatAXPY(quadts->mat_sensip,th->time_step0*(1.-th->Theta),th->MatIntegralSensipTemp,SAME_NONZERO_PATTERN));
    }
  }

  /* Solve the tangent linear equation for forward sensitivities to parameters */
  for (ntlm=0; ntlm<th->num_tlm; ntlm++) {
    KSPConvergedReason kspreason;
    PetscCall(MatDenseGetColumn(MatDeltaFwdSensip,ntlm,&barr));
    PetscCall(VecPlaceArray(VecDeltaFwdSensipCol,barr));
    if (th->endpoint) {
      PetscCall(MatDenseGetColumn(ts->mat_sensip,ntlm,&xarr));
      PetscCall(VecPlaceArray(ts->vec_sensip_col,xarr));
      PetscCall(KSPSolve(ksp,VecDeltaFwdSensipCol,ts->vec_sensip_col));
      PetscCall(VecResetArray(ts->vec_sensip_col));
      PetscCall(MatDenseRestoreColumn(ts->mat_sensip,&xarr));
    } else {
      PetscCall(KSPSolve(ksp,VecDeltaFwdSensipCol,VecDeltaFwdSensipCol));
    }
    PetscCall(KSPGetConvergedReason(ksp,&kspreason));
    if (kspreason < 0) {
      ts->reason = TSFORWARD_DIVERGED_LINEAR_SOLVE;
      PetscCall(PetscInfo(ts,"Step=%" PetscInt_FMT ", %" PetscInt_FMT "th tangent linear solve, linear solve fails, stopping tangent linear solve\n",ts->steps,ntlm));
    }
    PetscCall(VecResetArray(VecDeltaFwdSensipCol));
    PetscCall(MatDenseRestoreColumn(MatDeltaFwdSensip,&barr));
  }

  /*
    Evaluate the second stage of integral gradients with the 2-stage method:
    drdu|t_{n+1}*S(t_{n+1}) + drdp|t_{n+1}
  */
  if (quadts && quadts->mat_sensip) {
    if (!th->endpoint) {
      PetscCall(MatAXPY(ts->mat_sensip,1,MatDeltaFwdSensip,SAME_NONZERO_PATTERN)); /* stage sensitivity */
      PetscCall(TSComputeRHSJacobian(quadts,th->stage_time,th->X,quadJ,NULL));
      PetscCall(TSComputeRHSJacobianP(quadts,th->stage_time,th->X,quadJp));
      PetscCall(MatTransposeMatMult(ts->mat_sensip,quadJ,MAT_REUSE_MATRIX,PETSC_DEFAULT,&th->MatIntegralSensipTemp));
      PetscCall(MatAXPY(th->MatIntegralSensipTemp,1,quadJp,SAME_NONZERO_PATTERN));
      PetscCall(MatAXPY(quadts->mat_sensip,th->time_step0,th->MatIntegralSensipTemp,SAME_NONZERO_PATTERN));
      PetscCall(MatAXPY(ts->mat_sensip,(1.-th->Theta)/th->Theta,MatDeltaFwdSensip,SAME_NONZERO_PATTERN));
    } else {
      PetscCall(TSComputeRHSJacobian(quadts,th->stage_time,ts->vec_sol,quadJ,NULL));
      PetscCall(TSComputeRHSJacobianP(quadts,th->stage_time,ts->vec_sol,quadJp));
      PetscCall(MatTransposeMatMult(ts->mat_sensip,quadJ,MAT_REUSE_MATRIX,PETSC_DEFAULT,&th->MatIntegralSensipTemp));
      PetscCall(MatAXPY(th->MatIntegralSensipTemp,1,quadJp,SAME_NONZERO_PATTERN));
      PetscCall(MatAXPY(quadts->mat_sensip,th->time_step0*th->Theta,th->MatIntegralSensipTemp,SAME_NONZERO_PATTERN));
    }
  } else {
    if (!th->endpoint) {
      PetscCall(MatAXPY(ts->mat_sensip,1./th->Theta,MatDeltaFwdSensip,SAME_NONZERO_PATTERN));
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSForwardGetStages_Theta(TS ts,PetscInt *ns,Mat *stagesensip[])
{
  TS_Theta       *th = (TS_Theta*)ts->data;

  PetscFunctionBegin;
  if (ns) {
    if (!th->endpoint && th->Theta != 1.0) *ns = 1; /* midpoint form */
    else *ns = 2; /* endpoint form */
  }
  if (stagesensip) {
    if (!th->endpoint && th->Theta != 1.0) {
      th->MatFwdStages[0] = th->MatDeltaFwdSensip;
    } else {
      th->MatFwdStages[0] = th->MatFwdSensip0;
      th->MatFwdStages[1] = ts->mat_sensip; /* stiffly accurate */
    }
    *stagesensip = th->MatFwdStages;
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
static PetscErrorCode TSReset_Theta(TS ts)
{
  TS_Theta       *th = (TS_Theta*)ts->data;

  PetscFunctionBegin;
  PetscCall(VecDestroy(&th->X));
  PetscCall(VecDestroy(&th->Xdot));
  PetscCall(VecDestroy(&th->X0));
  PetscCall(VecDestroy(&th->affine));

  PetscCall(VecDestroy(&th->vec_sol_prev));
  PetscCall(VecDestroy(&th->vec_lte_work));

  PetscCall(VecDestroy(&th->VecCostIntegral0));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSAdjointReset_Theta(TS ts)
{
  TS_Theta       *th = (TS_Theta*)ts->data;

  PetscFunctionBegin;
  PetscCall(VecDestroyVecs(ts->numcost,&th->VecsDeltaLam));
  PetscCall(VecDestroyVecs(ts->numcost,&th->VecsDeltaMu));
  PetscCall(VecDestroyVecs(ts->numcost,&th->VecsDeltaLam2));
  PetscCall(VecDestroyVecs(ts->numcost,&th->VecsDeltaMu2));
  PetscCall(VecDestroyVecs(ts->numcost,&th->VecsSensiTemp));
  PetscCall(VecDestroyVecs(ts->numcost,&th->VecsSensi2Temp));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSDestroy_Theta(TS ts)
{
  PetscFunctionBegin;
  PetscCall(TSReset_Theta(ts));
  if (ts->dm) {
    PetscCall(DMCoarsenHookRemove(ts->dm,DMCoarsenHook_TSTheta,DMRestrictHook_TSTheta,ts));
    PetscCall(DMSubDomainHookRemove(ts->dm,DMSubDomainHook_TSTheta,DMSubDomainRestrictHook_TSTheta,ts));
  }
  PetscCall(PetscFree(ts->data));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts,"TSThetaGetTheta_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts,"TSThetaSetTheta_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts,"TSThetaGetEndpoint_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts,"TSThetaSetEndpoint_C",NULL));
  PetscFunctionReturn(0);
}

/*
  This defines the nonlinear equation that is to be solved with SNES
  G(U) = F[t0+Theta*dt, U, (U-U0)*shift] = 0

  Note that U here is the stage argument. This means that U = U_{n+1} only if endpoint = true,
  otherwise U = theta U_{n+1} + (1 - theta) U0, which for the case of implicit midpoint is
  U = (U_{n+1} + U0)/2
*/
static PetscErrorCode SNESTSFormFunction_Theta(SNES snes,Vec x,Vec y,TS ts)
{
  TS_Theta       *th = (TS_Theta*)ts->data;
  Vec            X0,Xdot;
  DM             dm,dmsave;
  PetscReal      shift = th->shift;

  PetscFunctionBegin;
  PetscCall(SNESGetDM(snes,&dm));
  /* When using the endpoint variant, this is actually 1/Theta * Xdot */
  PetscCall(TSThetaGetX0AndXdot(ts,dm,&X0,&Xdot));
  if (x != X0) {
    PetscCall(VecAXPBYPCZ(Xdot,-shift,shift,0,X0,x));
  } else {
    PetscCall(VecZeroEntries(Xdot));
  }
  /* DM monkey-business allows user code to call TSGetDM() inside of functions evaluated on levels of FAS */
  dmsave = ts->dm;
  ts->dm = dm;
  PetscCall(TSComputeIFunction(ts,th->stage_time,x,Xdot,y,PETSC_FALSE));
  ts->dm = dmsave;
  PetscCall(TSThetaRestoreX0AndXdot(ts,dm,&X0,&Xdot));
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESTSFormJacobian_Theta(SNES snes,Vec x,Mat A,Mat B,TS ts)
{
  TS_Theta       *th = (TS_Theta*)ts->data;
  Vec            Xdot;
  DM             dm,dmsave;
  PetscReal      shift = th->shift;

  PetscFunctionBegin;
  PetscCall(SNESGetDM(snes,&dm));
  /* Xdot has already been computed in SNESTSFormFunction_Theta (SNES guarantees this) */
  PetscCall(TSThetaGetX0AndXdot(ts,dm,NULL,&Xdot));

  dmsave = ts->dm;
  ts->dm = dm;
  PetscCall(TSComputeIJacobian(ts,th->stage_time,x,Xdot,shift,A,B,PETSC_FALSE));
  ts->dm = dmsave;
  PetscCall(TSThetaRestoreX0AndXdot(ts,dm,NULL,&Xdot));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSForwardSetUp_Theta(TS ts)
{
  TS_Theta       *th = (TS_Theta*)ts->data;
  TS             quadts = ts->quadraturets;

  PetscFunctionBegin;
  /* combine sensitivities to parameters and sensitivities to initial values into one array */
  th->num_tlm = ts->num_parameters;
  PetscCall(MatDuplicate(ts->mat_sensip,MAT_DO_NOT_COPY_VALUES,&th->MatDeltaFwdSensip));
  if (quadts && quadts->mat_sensip) {
    PetscCall(MatDuplicate(quadts->mat_sensip,MAT_DO_NOT_COPY_VALUES,&th->MatIntegralSensipTemp));
    PetscCall(MatDuplicate(quadts->mat_sensip,MAT_DO_NOT_COPY_VALUES,&th->MatIntegralSensip0));
  }
  /* backup sensitivity results for roll-backs */
  PetscCall(MatDuplicate(ts->mat_sensip,MAT_DO_NOT_COPY_VALUES,&th->MatFwdSensip0));

  PetscCall(VecDuplicate(ts->vec_sol,&th->VecDeltaFwdSensipCol));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSForwardReset_Theta(TS ts)
{
  TS_Theta       *th = (TS_Theta*)ts->data;
  TS             quadts = ts->quadraturets;

  PetscFunctionBegin;
  if (quadts && quadts->mat_sensip) {
    PetscCall(MatDestroy(&th->MatIntegralSensipTemp));
    PetscCall(MatDestroy(&th->MatIntegralSensip0));
  }
  PetscCall(VecDestroy(&th->VecDeltaFwdSensipCol));
  PetscCall(MatDestroy(&th->MatDeltaFwdSensip));
  PetscCall(MatDestroy(&th->MatFwdSensip0));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSSetUp_Theta(TS ts)
{
  TS_Theta       *th = (TS_Theta*)ts->data;
  TS             quadts = ts->quadraturets;
  PetscBool      match;

  PetscFunctionBegin;
  if (!th->VecCostIntegral0 && quadts && ts->costintegralfwd) { /* back up cost integral */
    PetscCall(VecDuplicate(quadts->vec_sol,&th->VecCostIntegral0));
  }
  if (!th->X) {
    PetscCall(VecDuplicate(ts->vec_sol,&th->X));
  }
  if (!th->Xdot) {
    PetscCall(VecDuplicate(ts->vec_sol,&th->Xdot));
  }
  if (!th->X0) {
    PetscCall(VecDuplicate(ts->vec_sol,&th->X0));
  }
  if (th->endpoint) {
    PetscCall(VecDuplicate(ts->vec_sol,&th->affine));
  }

  th->order = (th->Theta == 0.5) ? 2 : 1;
  th->shift = 1/(th->Theta*ts->time_step);

  PetscCall(TSGetDM(ts,&ts->dm));
  PetscCall(DMCoarsenHookAdd(ts->dm,DMCoarsenHook_TSTheta,DMRestrictHook_TSTheta,ts));
  PetscCall(DMSubDomainHookAdd(ts->dm,DMSubDomainHook_TSTheta,DMSubDomainRestrictHook_TSTheta,ts));

  PetscCall(TSGetAdapt(ts,&ts->adapt));
  PetscCall(TSAdaptCandidatesClear(ts->adapt));
  PetscCall(PetscObjectTypeCompare((PetscObject)ts->adapt,TSADAPTNONE,&match));
  if (!match) {
    PetscCall(VecDuplicate(ts->vec_sol,&th->vec_sol_prev));
    PetscCall(VecDuplicate(ts->vec_sol,&th->vec_lte_work));
  }
  PetscCall(TSGetSNES(ts,&ts->snes));

  ts->stifflyaccurate = (!th->endpoint && th->Theta != 1.0) ? PETSC_FALSE : PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode TSAdjointSetUp_Theta(TS ts)
{
  TS_Theta       *th = (TS_Theta*)ts->data;

  PetscFunctionBegin;
  PetscCall(VecDuplicateVecs(ts->vecs_sensi[0],ts->numcost,&th->VecsDeltaLam));
  PetscCall(VecDuplicateVecs(ts->vecs_sensi[0],ts->numcost,&th->VecsSensiTemp));
  if (ts->vecs_sensip) {
    PetscCall(VecDuplicateVecs(ts->vecs_sensip[0],ts->numcost,&th->VecsDeltaMu));
  }
  if (ts->vecs_sensi2) {
    PetscCall(VecDuplicateVecs(ts->vecs_sensi[0],ts->numcost,&th->VecsDeltaLam2));
    PetscCall(VecDuplicateVecs(ts->vecs_sensi2[0],ts->numcost,&th->VecsSensi2Temp));
    /* hack ts to make implicit TS solver work when users provide only explicit versions of callbacks (RHSFunction,RHSJacobian,RHSHessian etc.) */
    if (!ts->ihessianproduct_fuu) ts->vecs_fuu = ts->vecs_guu;
    if (!ts->ihessianproduct_fup) ts->vecs_fup = ts->vecs_gup;
  }
  if (ts->vecs_sensi2p) {
    PetscCall(VecDuplicateVecs(ts->vecs_sensi2p[0],ts->numcost,&th->VecsDeltaMu2));
    /* hack ts to make implicit TS solver work when users provide only explicit versions of callbacks (RHSFunction,RHSJacobian,RHSHessian etc.) */
    if (!ts->ihessianproduct_fpu) ts->vecs_fpu = ts->vecs_gpu;
    if (!ts->ihessianproduct_fpp) ts->vecs_fpp = ts->vecs_gpp;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSSetFromOptions_Theta(PetscOptionItems *PetscOptionsObject,TS ts)
{
  TS_Theta       *th = (TS_Theta*)ts->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject,"Theta ODE solver options");
  {
    PetscCall(PetscOptionsReal("-ts_theta_theta","Location of stage (0<Theta<=1)","TSThetaSetTheta",th->Theta,&th->Theta,NULL));
    PetscCall(PetscOptionsBool("-ts_theta_endpoint","Use the endpoint instead of midpoint form of the Theta method","TSThetaSetEndpoint",th->endpoint,&th->endpoint,NULL));
    PetscCall(PetscOptionsBool("-ts_theta_initial_guess_extrapolate","Extrapolate stage initial guess from previous solution (sometimes unstable)","TSThetaSetExtrapolate",th->extrapolate,&th->extrapolate,NULL));
  }
  PetscOptionsHeadEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode TSView_Theta(TS ts,PetscViewer viewer)
{
  TS_Theta       *th = (TS_Theta*)ts->data;
  PetscBool      iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Theta=%g\n",(double)th->Theta));
    PetscCall(PetscViewerASCIIPrintf(viewer,"  Extrapolation=%s\n",th->extrapolate ? "yes" : "no"));
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
  PetscCheck(theta > 0 && theta <= 1,PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_OUTOFRANGE,"Theta %g not in range (0,1]",(double)theta);
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

static PetscErrorCode TSGetStages_Theta(TS ts,PetscInt *ns,Vec *Y[])
{
  TS_Theta       *th = (TS_Theta*)ts->data;

  PetscFunctionBegin;
  if (ns) {
    if (!th->endpoint && th->Theta != 1.0) *ns = 1; /* midpoint form */
    else *ns = 2; /* endpoint form */
  }
  if (Y) {
    if (!th->endpoint && th->Theta != 1.0) {
      th->Stages[0] = th->X;
    } else {
      th->Stages[0] = th->X0;
      th->Stages[1] = ts->vec_sol; /* stiffly accurate */
    }
    *Y = th->Stages;
  }
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

   The endpoint variant of the Theta method and backward Euler can be applied to DAE. The midpoint variant is not suitable for DAEs because it is not stiffly accurate.

   The midpoint variant is cast as a 1-stage implicit Runge-Kutta method.

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

.seealso: `TSCreate()`, `TS`, `TSSetType()`, `TSCN`, `TSBEULER`, `TSThetaSetTheta()`, `TSThetaSetEndpoint()`

M*/
PETSC_EXTERN PetscErrorCode TSCreate_Theta(TS ts)
{
  TS_Theta       *th;

  PetscFunctionBegin;
  ts->ops->reset           = TSReset_Theta;
  ts->ops->adjointreset    = TSAdjointReset_Theta;
  ts->ops->destroy         = TSDestroy_Theta;
  ts->ops->view            = TSView_Theta;
  ts->ops->setup           = TSSetUp_Theta;
  ts->ops->adjointsetup    = TSAdjointSetUp_Theta;
  ts->ops->adjointreset    = TSAdjointReset_Theta;
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

  ts->ops->forwardsetup     = TSForwardSetUp_Theta;
  ts->ops->forwardreset     = TSForwardReset_Theta;
  ts->ops->forwardstep      = TSForwardStep_Theta;
  ts->ops->forwardgetstages = TSForwardGetStages_Theta;

  ts->usessnes = PETSC_TRUE;

  PetscCall(PetscNewLog(ts,&th));
  ts->data = (void*)th;

  th->VecsDeltaLam    = NULL;
  th->VecsDeltaMu     = NULL;
  th->VecsSensiTemp   = NULL;
  th->VecsSensi2Temp  = NULL;

  th->extrapolate = PETSC_FALSE;
  th->Theta       = 0.5;
  th->order       = 2;
  PetscCall(PetscObjectComposeFunction((PetscObject)ts,"TSThetaGetTheta_C",TSThetaGetTheta_Theta));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts,"TSThetaSetTheta_C",TSThetaSetTheta_Theta));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts,"TSThetaGetEndpoint_C",TSThetaGetEndpoint_Theta));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts,"TSThetaSetEndpoint_C",TSThetaSetEndpoint_Theta));
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

.seealso: `TSThetaSetTheta()`
@*/
PetscErrorCode  TSThetaGetTheta(TS ts,PetscReal *theta)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidRealPointer(theta,2);
  PetscUseMethod(ts,"TSThetaGetTheta_C",(TS,PetscReal*),(ts,theta));
  PetscFunctionReturn(0);
}

/*@
  TSThetaSetTheta - Set the abscissa of the stage in (0,1].

  Not Collective

  Input Parameters:
+  ts - timestepping context
-  theta - stage abscissa

  Options Database:
.  -ts_theta_theta <theta> - set theta

  Level: Intermediate

.seealso: `TSThetaGetTheta()`
@*/
PetscErrorCode  TSThetaSetTheta(TS ts,PetscReal theta)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscTryMethod(ts,"TSThetaSetTheta_C",(TS,PetscReal),(ts,theta));
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

.seealso: `TSThetaSetEndpoint()`, `TSTHETA`, `TSCN`
@*/
PetscErrorCode TSThetaGetEndpoint(TS ts,PetscBool *endpoint)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidBoolPointer(endpoint,2);
  PetscUseMethod(ts,"TSThetaGetEndpoint_C",(TS,PetscBool*),(ts,endpoint));
  PetscFunctionReturn(0);
}

/*@
  TSThetaSetEndpoint - Sets whether to use the endpoint variant of the method (e.g. trapezoid/Crank-Nicolson instead of midpoint rule).

  Not Collective

  Input Parameters:
+  ts - timestepping context
-  flg - PETSC_TRUE to use the endpoint variant

  Options Database:
.  -ts_theta_endpoint <flg> - use the endpoint variant

  Level: Intermediate

.seealso: `TSTHETA`, `TSCN`
@*/
PetscErrorCode TSThetaSetEndpoint(TS ts,PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscTryMethod(ts,"TSThetaSetEndpoint_C",(TS,PetscBool),(ts,flg));
  PetscFunctionReturn(0);
}

/*
 * TSBEULER and TSCN are straightforward specializations of TSTHETA.
 * The creation functions for these specializations are below.
 */

static PetscErrorCode TSSetUp_BEuler(TS ts)
{
  TS_Theta       *th = (TS_Theta*)ts->data;

  PetscFunctionBegin;
  PetscCheck(th->Theta == 1.0,PetscObjectComm((PetscObject)ts),PETSC_ERR_OPT_OVERWRITE,"Can not change the default value (1) of theta when using backward Euler");
  PetscCheck(!th->endpoint,PetscObjectComm((PetscObject)ts),PETSC_ERR_OPT_OVERWRITE,"Can not change to the endpoint form of the Theta methods when using backward Euler");
  PetscCall(TSSetUp_Theta(ts));
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

.seealso: `TSCreate()`, `TS`, `TSSetType()`, `TSEULER`, `TSCN`, `TSTHETA`

M*/
PETSC_EXTERN PetscErrorCode TSCreate_BEuler(TS ts)
{
  PetscFunctionBegin;
  PetscCall(TSCreate_Theta(ts));
  PetscCall(TSThetaSetTheta(ts,1.0));
  PetscCall(TSThetaSetEndpoint(ts,PETSC_FALSE));
  ts->ops->setup = TSSetUp_BEuler;
  ts->ops->view  = TSView_BEuler;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSSetUp_CN(TS ts)
{
  TS_Theta       *th = (TS_Theta*)ts->data;

  PetscFunctionBegin;
  PetscCheck(th->Theta == 0.5,PetscObjectComm((PetscObject)ts),PETSC_ERR_OPT_OVERWRITE,"Can not change the default value (0.5) of theta when using Crank-Nicolson");
  PetscCheck(th->endpoint,PetscObjectComm((PetscObject)ts),PETSC_ERR_OPT_OVERWRITE,"Can not change to the midpoint form of the Theta methods when using Crank-Nicolson");
  PetscCall(TSSetUp_Theta(ts));
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

.seealso: `TSCreate()`, `TS`, `TSSetType()`, `TSBEULER`, `TSTHETA`

M*/
PETSC_EXTERN PetscErrorCode TSCreate_CN(TS ts)
{
  PetscFunctionBegin;
  PetscCall(TSCreate_Theta(ts));
  PetscCall(TSThetaSetTheta(ts,0.5));
  PetscCall(TSThetaSetEndpoint(ts,PETSC_TRUE));
  ts->ops->setup = TSSetUp_CN;
  ts->ops->view  = TSView_CN;
  PetscFunctionReturn(0);
}
