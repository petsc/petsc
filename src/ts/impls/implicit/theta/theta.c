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
  TS             quadts = ts->quadraturets;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (th->endpoint) {
    /* Evolve ts->vec_costintegral to compute integrals */
    if (th->Theta!=1.0) {
      ierr = TSComputeRHSFunction(quadts,th->ptime0,th->X0,ts->vec_costintegrand);CHKERRQ(ierr);
      ierr = VecAXPY(quadts->vec_sol,th->time_step0*(1.0-th->Theta),ts->vec_costintegrand);CHKERRQ(ierr);
    }
    ierr = TSComputeRHSFunction(quadts,ts->ptime,ts->vec_sol,ts->vec_costintegrand);CHKERRQ(ierr);
    ierr = VecAXPY(quadts->vec_sol,th->time_step0*th->Theta,ts->vec_costintegrand);CHKERRQ(ierr);
  } else {
    ierr = TSComputeRHSFunction(quadts,th->stage_time,th->X,ts->vec_costintegrand);CHKERRQ(ierr);
    ierr = VecAXPY(quadts->vec_sol,th->time_step0,ts->vec_costintegrand);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSForwardCostIntegral_Theta(TS ts)
{
  TS_Theta       *th = (TS_Theta*)ts->data;
  TS             quadts = ts->quadraturets;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* backup cost integral */
  ierr = VecCopy(quadts->vec_sol,th->VecCostIntegral0);CHKERRQ(ierr);
  ierr = TSThetaEvaluateCostIntegral(ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSAdjointCostIntegral_Theta(TS ts)
{
  TS_Theta       *th = (TS_Theta*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Like TSForwardCostIntegral(), the adjoint cost integral evaluation relies on ptime0 and time_step0. */
  th->ptime0     = ts->ptime + ts->time_step;
  th->time_step0 = -ts->time_step;
  ierr = TSThetaEvaluateCostIntegral(ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSTheta_SNESSolve(TS ts,Vec b,Vec x)
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

  th->status     = TS_STEP_INCOMPLETE;
  while (!ts->reason && th->status != TS_STEP_COMPLETE) {
    th->shift      = 1/(th->Theta*ts->time_step);
    th->stage_time = ts->ptime + (th->endpoint ? (PetscReal)1 : th->Theta)*ts->time_step;
    ierr = VecCopy(th->X0,th->X);CHKERRQ(ierr);
    if (th->extrapolate && !ts->steprestart) {
      ierr = VecAXPY(th->X,1/th->shift,th->Xdot);CHKERRQ(ierr);
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
    ierr = TSTheta_SNESSolve(ts,th->affine,th->X);CHKERRQ(ierr);
    ierr = TSPostStage(ts,th->stage_time,0,&th->X);CHKERRQ(ierr);
    ierr = TSAdaptCheckStage(ts->adapt,ts,th->stage_time,th->X,&stageok);CHKERRQ(ierr);
    if (!stageok) goto reject_step;

    th->status = TS_STEP_PENDING;
    if (th->endpoint) {
      ierr = VecCopy(th->X,ts->vec_sol);CHKERRQ(ierr);
    } else {
      ierr = VecAXPBYPCZ(th->Xdot,-th->shift,th->shift,0,th->X0,th->X);CHKERRQ(ierr);
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
      ierr = PetscInfo2(ts,"Step=%D, step rejections %D greater than current TS allowed, stopping solve\n",ts->steps,rejections);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetEquationType(ts,&eqtype);CHKERRQ(ierr);
  if (eqtype == TS_EQ_ODE_EXPLICIT) {
    isexplicitode  = PETSC_TRUE;
    VecsDeltaLam  = ts->vecs_sensi;
    VecsDeltaLam2 = ts->vecs_sensi2;
  }
  th->status = TS_STEP_INCOMPLETE;
  ierr = SNESGetKSP(ts->snes,&ksp);CHKERRQ(ierr);
  ierr = TSGetIJacobian(ts,&J,&Jpre,NULL,NULL);CHKERRQ(ierr);
  if (quadts) {
    ierr = TSGetRHSJacobian(quadts,&quadJ,NULL,NULL,NULL);CHKERRQ(ierr);
    ierr = TSGetRHSJacobianP(quadts,&quadJp,NULL,NULL);CHKERRQ(ierr);
  }

  th->stage_time    = ts->ptime;
  adjoint_time_step = -ts->time_step; /* always positive since time_step is negative */

  /* Build RHS for first-order adjoint lambda_{n+1}/h + r_u^T(n+1) */
  if (quadts) {
    ierr = TSComputeRHSJacobian(quadts,th->stage_time,th->X,quadJ,NULL);CHKERRQ(ierr);
  }

  for (nadj=0; nadj<ts->numcost; nadj++) {
    ierr = VecCopy(ts->vecs_sensi[nadj],VecsSensiTemp[nadj]);CHKERRQ(ierr);
    ierr = VecScale(VecsSensiTemp[nadj],1./adjoint_time_step);CHKERRQ(ierr); /* lambda_{n+1}/h */
    if (quadJ) {
      ierr = MatDenseGetColumn(quadJ,nadj,&xarr);CHKERRQ(ierr);
      ierr = VecPlaceArray(ts->vec_drdu_col,xarr);CHKERRQ(ierr);
      ierr = VecAXPY(VecsSensiTemp[nadj],1.,ts->vec_drdu_col);CHKERRQ(ierr);
      ierr = VecResetArray(ts->vec_drdu_col);CHKERRQ(ierr);
      ierr = MatDenseRestoreColumn(quadJ,&xarr);CHKERRQ(ierr);
    }
  }

  /* Build LHS for first-order adjoint */
  th->shift = 1./adjoint_time_step;
  ierr = TSComputeSNESJacobian(ts,th->X,J,Jpre);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,J,Jpre);CHKERRQ(ierr);

  /* Solve stage equation LHS*lambda_s = RHS for first-order adjoint */
  for (nadj=0; nadj<ts->numcost; nadj++) {
    KSPConvergedReason kspreason;
    ierr = KSPSolveTranspose(ksp,VecsSensiTemp[nadj],VecsDeltaLam[nadj]);CHKERRQ(ierr);
    ierr = KSPGetConvergedReason(ksp,&kspreason);CHKERRQ(ierr);
    if (kspreason < 0) {
      ts->reason = TSADJOINT_DIVERGED_LINEAR_SOLVE;
      ierr = PetscInfo2(ts,"Step=%D, %Dth cost function, transposed linear solve fails, stopping 1st-order adjoint solve\n",ts->steps,nadj);CHKERRQ(ierr);
    }
  }

  if (ts->vecs_sensi2) { /* U_{n+1} */
    /* Get w1 at t_{n+1} from TLM matrix */
    ierr = MatDenseGetColumn(ts->mat_sensip,0,&xarr);CHKERRQ(ierr);
    ierr = VecPlaceArray(ts->vec_sensip_col,xarr);CHKERRQ(ierr);
    /* lambda_s^T F_UU w_1 */
    ierr = TSComputeIHessianProductFunctionUU(ts,th->stage_time,th->X,VecsDeltaLam,ts->vec_sensip_col,ts->vecs_fuu);CHKERRQ(ierr);
    /* lambda_s^T F_UP w_2 */
    ierr = TSComputeIHessianProductFunctionUP(ts,th->stage_time,th->X,VecsDeltaLam,ts->vec_dir,ts->vecs_fup);CHKERRQ(ierr);
    for (nadj=0; nadj<ts->numcost; nadj++) { /* compute the residual */
      ierr = VecCopy(ts->vecs_sensi2[nadj],VecsSensi2Temp[nadj]);CHKERRQ(ierr);
      ierr = VecScale(VecsSensi2Temp[nadj],1./adjoint_time_step);CHKERRQ(ierr);
      ierr = VecAXPY(VecsSensi2Temp[nadj],-1.,ts->vecs_fuu[nadj]);CHKERRQ(ierr);
      if (ts->vecs_fup) {
        ierr = VecAXPY(VecsSensi2Temp[nadj],-1.,ts->vecs_fup[nadj]);CHKERRQ(ierr);
      }
    }
    /* Solve stage equation LHS X = RHS for second-order adjoint */
    for (nadj=0; nadj<ts->numcost; nadj++) {
      KSPConvergedReason kspreason;
      ierr = KSPSolveTranspose(ksp,VecsSensi2Temp[nadj],VecsDeltaLam2[nadj]);CHKERRQ(ierr);
      ierr = KSPGetConvergedReason(ksp,&kspreason);CHKERRQ(ierr);
      if (kspreason < 0) {
        ts->reason = TSADJOINT_DIVERGED_LINEAR_SOLVE;
        ierr = PetscInfo2(ts,"Step=%D, %Dth cost function, transposed linear solve fails, stopping 2nd-order adjoint solve\n",ts->steps,nadj);CHKERRQ(ierr);
      }
    }
  }

  /* Update sensitivities, and evaluate integrals if there is any */
  if (!isexplicitode) {
    th->shift = 0.0;
    ierr = TSComputeSNESJacobian(ts,th->X,J,Jpre);CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,J,Jpre);CHKERRQ(ierr);
    ierr = MatScale(J,-1.);CHKERRQ(ierr);
    for (nadj=0; nadj<ts->numcost; nadj++) {
      /* Add f_U \lambda_s to the original RHS */
      ierr = MatMultTransposeAdd(J,VecsDeltaLam[nadj],VecsSensiTemp[nadj],VecsSensiTemp[nadj]);CHKERRQ(ierr);
      ierr = VecScale(VecsSensiTemp[nadj],adjoint_time_step);CHKERRQ(ierr);
      ierr = VecCopy(VecsSensiTemp[nadj],ts->vecs_sensi[nadj]);CHKERRQ(ierr);
      if (ts->vecs_sensi2) {
        ierr = MatMultTransposeAdd(J,VecsDeltaLam2[nadj],VecsSensi2Temp[nadj],VecsSensi2Temp[nadj]);CHKERRQ(ierr);
        ierr = VecScale(VecsSensi2Temp[nadj],adjoint_time_step);CHKERRQ(ierr);
        ierr = VecCopy(VecsSensi2Temp[nadj],ts->vecs_sensi2[nadj]);CHKERRQ(ierr);
      }
    }
  }
  if (ts->vecs_sensip) {
    ierr = TSComputeIJacobianP(ts,th->stage_time,th->X,th->Xdot,1./adjoint_time_step,ts->Jacp,PETSC_FALSE);CHKERRQ(ierr); /* get -f_p */
    if (quadts) {
      ierr = TSComputeRHSJacobianP(quadts,th->stage_time,th->X,quadJp);CHKERRQ(ierr);
    }
    if (ts->vecs_sensi2p) {
      /* lambda_s^T F_PU w_1 */
      ierr = TSComputeIHessianProductFunctionPU(ts,th->stage_time,th->X,VecsDeltaLam,ts->vec_sensip_col,ts->vecs_fpu);CHKERRQ(ierr);
      /* lambda_s^T F_PP w_2 */
      ierr = TSComputeIHessianProductFunctionPP(ts,th->stage_time,th->X,VecsDeltaLam,ts->vec_dir,ts->vecs_fpp);CHKERRQ(ierr);
    }

    for (nadj=0; nadj<ts->numcost; nadj++) {
      ierr = MatMultTranspose(ts->Jacp,VecsDeltaLam[nadj],VecsDeltaMu[nadj]);CHKERRQ(ierr);
      ierr = VecAXPY(ts->vecs_sensip[nadj],-adjoint_time_step,VecsDeltaMu[nadj]);CHKERRQ(ierr);
      if (quadJp) {
        ierr = MatDenseGetColumn(quadJp,nadj,&xarr);CHKERRQ(ierr);
        ierr = VecPlaceArray(ts->vec_drdp_col,xarr);CHKERRQ(ierr);
        ierr = VecAXPY(ts->vecs_sensip[nadj],adjoint_time_step,ts->vec_drdp_col);CHKERRQ(ierr);
        ierr = VecResetArray(ts->vec_drdp_col);CHKERRQ(ierr);
        ierr = MatDenseRestoreColumn(quadJp,&xarr);CHKERRQ(ierr);
      }
      if (ts->vecs_sensi2p) {
        ierr = MatMultTranspose(ts->Jacp,VecsDeltaLam2[nadj],VecsDeltaMu2[nadj]);CHKERRQ(ierr);
        ierr = VecAXPY(ts->vecs_sensi2p[nadj],-adjoint_time_step,VecsDeltaMu2[nadj]);CHKERRQ(ierr);
        if (ts->vecs_fpu) {
          ierr = VecAXPY(ts->vecs_sensi2p[nadj],-adjoint_time_step,ts->vecs_fpu[nadj]);CHKERRQ(ierr);
        }
        if (ts->vecs_fpp) {
          ierr = VecAXPY(ts->vecs_sensi2p[nadj],-adjoint_time_step,ts->vecs_fpp[nadj]);CHKERRQ(ierr);
        }
      }
    }
  }

  if (ts->vecs_sensi2) {
    ierr = VecResetArray(ts->vec_sensip_col);CHKERRQ(ierr);
    ierr = MatDenseRestoreColumn(ts->mat_sensip,&xarr);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (th->Theta == 1.) {
    ierr = TSAdjointStepBEuler_Private(ts);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  th->status = TS_STEP_INCOMPLETE;
  ierr = SNESGetKSP(ts->snes,&ksp);CHKERRQ(ierr);
  ierr = TSGetIJacobian(ts,&J,&Jpre,NULL,NULL);CHKERRQ(ierr);
  if (quadts) {
    ierr = TSGetRHSJacobian(quadts,&quadJ,NULL,NULL,NULL);CHKERRQ(ierr);
    ierr = TSGetRHSJacobianP(quadts,&quadJp,NULL,NULL);CHKERRQ(ierr);
  }
  /* If endpoint=1, th->ptime and th->X0 will be used; if endpoint=0, th->stage_time and th->X will be used. */
  th->stage_time    = th->endpoint ? ts->ptime : (ts->ptime+(1.-th->Theta)*ts->time_step);
  adjoint_ptime     = ts->ptime + ts->time_step;
  adjoint_time_step = -ts->time_step;  /* always positive since time_step is negative */

  /* Build RHS for first-order adjoint */
  /* Cost function has an integral term */
  if (quadts) {
    if (th->endpoint) {
      ierr = TSComputeRHSJacobian(quadts,th->stage_time,ts->vec_sol,quadJ,NULL);CHKERRQ(ierr);
    } else {
      ierr = TSComputeRHSJacobian(quadts,th->stage_time,th->X,quadJ,NULL);CHKERRQ(ierr);
    }
  }

  for (nadj=0; nadj<ts->numcost; nadj++) {
    ierr = VecCopy(ts->vecs_sensi[nadj],VecsSensiTemp[nadj]);CHKERRQ(ierr);
    ierr = VecScale(VecsSensiTemp[nadj],1./(th->Theta*adjoint_time_step));CHKERRQ(ierr);
    if (quadJ) {
      ierr = MatDenseGetColumn(quadJ,nadj,&xarr);CHKERRQ(ierr);
      ierr = VecPlaceArray(ts->vec_drdu_col,xarr);CHKERRQ(ierr);
      ierr = VecAXPY(VecsSensiTemp[nadj],1.,ts->vec_drdu_col);CHKERRQ(ierr);
      ierr = VecResetArray(ts->vec_drdu_col);CHKERRQ(ierr);
      ierr = MatDenseRestoreColumn(quadJ,&xarr);CHKERRQ(ierr);
    }
  }

  /* Build LHS for first-order adjoint */
  th->shift = 1./(th->Theta*adjoint_time_step);
  if (th->endpoint) {
    ierr = TSComputeSNESJacobian(ts,ts->vec_sol,J,Jpre);CHKERRQ(ierr);
  } else {
    ierr = TSComputeSNESJacobian(ts,th->X,J,Jpre);CHKERRQ(ierr);
  }
  ierr = KSPSetOperators(ksp,J,Jpre);CHKERRQ(ierr);

  /* Solve stage equation LHS*lambda_s = RHS for first-order adjoint */
  for (nadj=0; nadj<ts->numcost; nadj++) {
    KSPConvergedReason kspreason;
    ierr = KSPSolveTranspose(ksp,VecsSensiTemp[nadj],VecsDeltaLam[nadj]);CHKERRQ(ierr);
    ierr = KSPGetConvergedReason(ksp,&kspreason);CHKERRQ(ierr);
    if (kspreason < 0) {
      ts->reason = TSADJOINT_DIVERGED_LINEAR_SOLVE;
      ierr = PetscInfo2(ts,"Step=%D, %Dth cost function, transposed linear solve fails, stopping 1st-order adjoint solve\n",ts->steps,nadj);CHKERRQ(ierr);
    }
  }

  /* Second-order adjoint */
  if (ts->vecs_sensi2) { /* U_{n+1} */
    if (!th->endpoint) SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"Operation not implemented in TS_Theta");
    /* Get w1 at t_{n+1} from TLM matrix */
    ierr = MatDenseGetColumn(ts->mat_sensip,0,&xarr);CHKERRQ(ierr);
    ierr = VecPlaceArray(ts->vec_sensip_col,xarr);CHKERRQ(ierr);
    /* lambda_s^T F_UU w_1 */
    ierr = TSComputeIHessianProductFunctionUU(ts,th->stage_time,ts->vec_sol,VecsDeltaLam,ts->vec_sensip_col,ts->vecs_fuu);CHKERRQ(ierr);
    ierr = VecResetArray(ts->vec_sensip_col);CHKERRQ(ierr);
    ierr = MatDenseRestoreColumn(ts->mat_sensip,&xarr);CHKERRQ(ierr);
    /* lambda_s^T F_UP w_2 */
    ierr = TSComputeIHessianProductFunctionUP(ts,th->stage_time,ts->vec_sol,VecsDeltaLam,ts->vec_dir,ts->vecs_fup);CHKERRQ(ierr);
    for (nadj=0; nadj<ts->numcost; nadj++) { /* compute the residual */
      ierr = VecCopy(ts->vecs_sensi2[nadj],VecsSensi2Temp[nadj]);CHKERRQ(ierr);
      ierr = VecScale(VecsSensi2Temp[nadj],th->shift);CHKERRQ(ierr);
      ierr = VecAXPY(VecsSensi2Temp[nadj],-1.,ts->vecs_fuu[nadj]);CHKERRQ(ierr);
      if (ts->vecs_fup) {
        ierr = VecAXPY(VecsSensi2Temp[nadj],-1.,ts->vecs_fup[nadj]);CHKERRQ(ierr);
      }
    }
    /* Solve stage equation LHS X = RHS for second-order adjoint */
    for (nadj=0; nadj<ts->numcost; nadj++) {
      KSPConvergedReason kspreason;
      ierr = KSPSolveTranspose(ksp,VecsSensi2Temp[nadj],VecsDeltaLam2[nadj]);CHKERRQ(ierr);
      ierr = KSPGetConvergedReason(ksp,&kspreason);CHKERRQ(ierr);
      if (kspreason < 0) {
        ts->reason = TSADJOINT_DIVERGED_LINEAR_SOLVE;
        ierr = PetscInfo2(ts,"Step=%D, %Dth cost function, transposed linear solve fails, stopping 2nd-order adjoint solve\n",ts->steps,nadj);CHKERRQ(ierr);
      }
    }
  }

  /* Update sensitivities, and evaluate integrals if there is any */
  if(th->endpoint) { /* two-stage Theta methods with th->Theta!=1, th->Theta==1 leads to BEuler */
    th->shift      = 1./((th->Theta-1.)*adjoint_time_step);
    th->stage_time = adjoint_ptime;
    ierr           = TSComputeSNESJacobian(ts,th->X0,J,Jpre);CHKERRQ(ierr);
    ierr           = KSPSetOperators(ksp,J,Jpre);CHKERRQ(ierr);
    /* R_U at t_n */
    if (quadts) {
      ierr = TSComputeRHSJacobian(quadts,adjoint_ptime,th->X0,quadJ,NULL);CHKERRQ(ierr);
    }
    for (nadj=0; nadj<ts->numcost; nadj++) {
      ierr = MatMultTranspose(J,VecsDeltaLam[nadj],ts->vecs_sensi[nadj]);CHKERRQ(ierr);
      if (quadJ) {
        ierr = MatDenseGetColumn(quadJ,nadj,&xarr);CHKERRQ(ierr);
        ierr = VecPlaceArray(ts->vec_drdu_col,xarr);CHKERRQ(ierr);
        ierr = VecAXPY(ts->vecs_sensi[nadj],-1.,ts->vec_drdu_col);CHKERRQ(ierr);
        ierr = VecResetArray(ts->vec_drdu_col);CHKERRQ(ierr);
        ierr = MatDenseRestoreColumn(quadJ,&xarr);CHKERRQ(ierr);
      }
      ierr = VecScale(ts->vecs_sensi[nadj],1./th->shift);CHKERRQ(ierr);
    }

    /* Second-order adjoint */
    if (ts->vecs_sensi2) { /* U_n */
      /* Get w1 at t_n from TLM matrix */
      ierr = MatDenseGetColumn(th->MatFwdSensip0,0,&xarr);CHKERRQ(ierr);
      ierr = VecPlaceArray(ts->vec_sensip_col,xarr);CHKERRQ(ierr);
      /* lambda_s^T F_UU w_1 */
      ierr = TSComputeIHessianProductFunctionUU(ts,adjoint_ptime,th->X0,VecsDeltaLam,ts->vec_sensip_col,ts->vecs_fuu);CHKERRQ(ierr);
      ierr = VecResetArray(ts->vec_sensip_col);CHKERRQ(ierr);
      ierr = MatDenseRestoreColumn(th->MatFwdSensip0,&xarr);CHKERRQ(ierr);
      /* lambda_s^T F_UU w_2 */
      ierr = TSComputeIHessianProductFunctionUP(ts,adjoint_ptime,th->X0,VecsDeltaLam,ts->vec_dir,ts->vecs_fup);CHKERRQ(ierr);
      for (nadj=0; nadj<ts->numcost; nadj++) {
        /* M^T Lambda_s + h(1-theta) F_U^T Lambda_s + h(1-theta) lambda_s^T F_UU w_1 + lambda_s^T F_UP w_2  */
        ierr = MatMultTranspose(J,VecsDeltaLam2[nadj],ts->vecs_sensi2[nadj]);CHKERRQ(ierr);
        ierr = VecAXPY(ts->vecs_sensi2[nadj],1.,ts->vecs_fuu[nadj]);CHKERRQ(ierr);
        if (ts->vecs_fup) {
          ierr = VecAXPY(ts->vecs_sensi2[nadj],1.,ts->vecs_fup[nadj]);CHKERRQ(ierr);
        }
        ierr = VecScale(ts->vecs_sensi2[nadj],1./th->shift);CHKERRQ(ierr);
      }
    }

    th->stage_time = ts->ptime; /* recover the old value */

    if (ts->vecs_sensip) { /* sensitivities wrt parameters */
      /* U_{n+1} */
      ierr = TSComputeIJacobianP(ts,th->stage_time,ts->vec_sol,th->Xdot,-1./(th->Theta*adjoint_time_step),ts->Jacp,PETSC_FALSE);CHKERRQ(ierr);
      if (quadts) {
        ierr = TSComputeRHSJacobianP(quadts,th->stage_time,ts->vec_sol,quadJp);CHKERRQ(ierr);
      }
      for (nadj=0; nadj<ts->numcost; nadj++) {
        ierr = MatMultTranspose(ts->Jacp,VecsDeltaLam[nadj],VecsDeltaMu[nadj]);CHKERRQ(ierr);
        ierr = VecAXPY(ts->vecs_sensip[nadj],-adjoint_time_step*th->Theta,VecsDeltaMu[nadj]);CHKERRQ(ierr);
      }
      if (ts->vecs_sensi2p) { /* second-order */
        /* Get w1 at t_{n+1} from TLM matrix */
        ierr = MatDenseGetColumn(ts->mat_sensip,0,&xarr);CHKERRQ(ierr);
        ierr = VecPlaceArray(ts->vec_sensip_col,xarr);CHKERRQ(ierr);
        /* lambda_s^T F_PU w_1 */
        ierr = TSComputeIHessianProductFunctionPU(ts,th->stage_time,ts->vec_sol,VecsDeltaLam,ts->vec_sensip_col,ts->vecs_fpu);CHKERRQ(ierr);
        ierr = VecResetArray(ts->vec_sensip_col);CHKERRQ(ierr);
        ierr = MatDenseRestoreColumn(ts->mat_sensip,&xarr);CHKERRQ(ierr);

        /* lambda_s^T F_PP w_2 */
        ierr = TSComputeIHessianProductFunctionPP(ts,th->stage_time,ts->vec_sol,VecsDeltaLam,ts->vec_dir,ts->vecs_fpp);CHKERRQ(ierr);
        for (nadj=0; nadj<ts->numcost; nadj++) {
          /* Mu2 <- Mu2 + h theta F_P^T Lambda_s + h theta (lambda_s^T F_UU w_1 + lambda_s^T F_UP w_2)  */
          ierr = MatMultTranspose(ts->Jacp,VecsDeltaLam2[nadj],VecsDeltaMu2[nadj]);CHKERRQ(ierr);
          ierr = VecAXPY(ts->vecs_sensi2p[nadj],-adjoint_time_step*th->Theta,VecsDeltaMu2[nadj]);CHKERRQ(ierr);
          if (ts->vecs_fpu) {
            ierr = VecAXPY(ts->vecs_sensi2p[nadj],-adjoint_time_step*th->Theta,ts->vecs_fpu[nadj]);CHKERRQ(ierr);
          }
          if (ts->vecs_fpp) {
            ierr = VecAXPY(ts->vecs_sensi2p[nadj],-adjoint_time_step*th->Theta,ts->vecs_fpp[nadj]);CHKERRQ(ierr);
          }
        }
      }

      /* U_s */
      ierr = TSComputeIJacobianP(ts,adjoint_ptime,th->X0,th->Xdot,1./((th->Theta-1.0)*adjoint_time_step),ts->Jacp,PETSC_FALSE);CHKERRQ(ierr);
      if (quadts) {
        ierr = TSComputeRHSJacobianP(quadts,adjoint_ptime,th->X0,quadJp);CHKERRQ(ierr);
      }
      for (nadj=0; nadj<ts->numcost; nadj++) {
        ierr = MatMultTranspose(ts->Jacp,VecsDeltaLam[nadj],VecsDeltaMu[nadj]);CHKERRQ(ierr);
        ierr = VecAXPY(ts->vecs_sensip[nadj],-adjoint_time_step*(1.0-th->Theta),VecsDeltaMu[nadj]);CHKERRQ(ierr);
        if (ts->vecs_sensi2p) { /* second-order */
          /* Get w1 at t_n from TLM matrix */
          ierr = MatDenseGetColumn(th->MatFwdSensip0,0,&xarr);CHKERRQ(ierr);
          ierr = VecPlaceArray(ts->vec_sensip_col,xarr);CHKERRQ(ierr);
          /* lambda_s^T F_PU w_1 */
          ierr = TSComputeIHessianProductFunctionPU(ts,adjoint_ptime,th->X0,VecsDeltaLam,ts->vec_sensip_col,ts->vecs_fpu);CHKERRQ(ierr);
          ierr = VecResetArray(ts->vec_sensip_col);CHKERRQ(ierr);
          ierr = MatDenseRestoreColumn(th->MatFwdSensip0,&xarr);CHKERRQ(ierr);
          /* lambda_s^T F_PP w_2 */
          ierr = TSComputeIHessianProductFunctionPP(ts,adjoint_ptime,th->X0,VecsDeltaLam,ts->vec_dir,ts->vecs_fpp);CHKERRQ(ierr);
          for (nadj=0; nadj<ts->numcost; nadj++) {
            /* Mu2 <- Mu2 + h(1-theta) F_P^T Lambda_s + h(1-theta) (lambda_s^T F_UU w_1 + lambda_s^T F_UP w_2) */
            ierr = MatMultTranspose(ts->Jacp,VecsDeltaLam2[nadj],VecsDeltaMu2[nadj]);CHKERRQ(ierr);
            ierr = VecAXPY(ts->vecs_sensi2p[nadj],-adjoint_time_step*(1.0-th->Theta),VecsDeltaMu2[nadj]);CHKERRQ(ierr);
            if (ts->vecs_fpu) {
              ierr = VecAXPY(ts->vecs_sensi2p[nadj],-adjoint_time_step*(1.0-th->Theta),ts->vecs_fpu[nadj]);CHKERRQ(ierr);
            }
            if (ts->vecs_fpp) {
              ierr = VecAXPY(ts->vecs_sensi2p[nadj],-adjoint_time_step*(1.0-th->Theta),ts->vecs_fpp[nadj]);CHKERRQ(ierr);
            }
          }
        }
      }
    }
  } else { /* one-stage case */
    th->shift = 0.0;
    ierr      = TSComputeSNESJacobian(ts,th->X,J,Jpre);CHKERRQ(ierr); /* get -f_y */
    ierr      = KSPSetOperators(ksp,J,Jpre);CHKERRQ(ierr);
    if (quadts) {
      ierr  = TSComputeRHSJacobian(quadts,th->stage_time,th->X,quadJ,NULL);CHKERRQ(ierr);
    }
    for (nadj=0; nadj<ts->numcost; nadj++) {
      ierr = MatMultTranspose(J,VecsDeltaLam[nadj],VecsSensiTemp[nadj]);CHKERRQ(ierr);
      ierr = VecAXPY(ts->vecs_sensi[nadj],-adjoint_time_step,VecsSensiTemp[nadj]);CHKERRQ(ierr);
      if (quadJ) {
        ierr = MatDenseGetColumn(quadJ,nadj,&xarr);CHKERRQ(ierr);
        ierr = VecPlaceArray(ts->vec_drdu_col,xarr);CHKERRQ(ierr);
        ierr = VecAXPY(ts->vecs_sensi[nadj],adjoint_time_step,ts->vec_drdu_col);CHKERRQ(ierr);
        ierr = VecResetArray(ts->vec_drdu_col);CHKERRQ(ierr);
        ierr = MatDenseRestoreColumn(quadJ,&xarr);CHKERRQ(ierr);
      }
    }
    if (ts->vecs_sensip) {
      ierr = TSComputeIJacobianP(ts,th->stage_time,th->X,th->Xdot,th->shift,ts->Jacp,PETSC_FALSE);CHKERRQ(ierr);
      if (quadts) {
        ierr = TSComputeRHSJacobianP(quadts,th->stage_time,th->X,quadJp);CHKERRQ(ierr);
      }
      for (nadj=0; nadj<ts->numcost; nadj++) {
        ierr = MatMultTranspose(ts->Jacp,VecsDeltaLam[nadj],VecsDeltaMu[nadj]);CHKERRQ(ierr);
        ierr = VecAXPY(ts->vecs_sensip[nadj],-adjoint_time_step,VecsDeltaMu[nadj]);CHKERRQ(ierr);
        if (quadJp) {
          ierr = MatDenseGetColumn(quadJp,nadj,&xarr);CHKERRQ(ierr);
          ierr = VecPlaceArray(ts->vec_drdp_col,xarr);CHKERRQ(ierr);
          ierr = VecAXPY(ts->vecs_sensip[nadj],adjoint_time_step,ts->vec_drdp_col);CHKERRQ(ierr);
          ierr = VecResetArray(ts->vec_drdp_col);CHKERRQ(ierr);
          ierr = MatDenseRestoreColumn(quadJp,&xarr);CHKERRQ(ierr);
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
  TS             quadts = ts->quadraturets;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCopy(th->X0,ts->vec_sol);CHKERRQ(ierr);
  if (quadts && ts->costintegralfwd) {
    ierr = VecCopy(th->VecCostIntegral0,quadts->vec_sol);CHKERRQ(ierr);
  }
  th->status = TS_STEP_INCOMPLETE;
  if (ts->mat_sensip) {
    ierr = MatCopy(th->MatFwdSensip0,ts->mat_sensip,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  }
  if (quadts && quadts->mat_sensip) {
    ierr = MatCopy(th->MatIntegralSensip0,quadts->mat_sensip,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  previous_shift = th->shift;
  ierr = MatCopy(ts->mat_sensip,th->MatFwdSensip0,SAME_NONZERO_PATTERN);CHKERRQ(ierr);

  if (quadts && quadts->mat_sensip) {
    ierr = MatCopy(quadts->mat_sensip,th->MatIntegralSensip0,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  }
  ierr = SNESGetKSP(ts->snes,&ksp);CHKERRQ(ierr);
  ierr = TSGetIJacobian(ts,&J,&Jpre,NULL,NULL);CHKERRQ(ierr);
  if (quadts) {
    ierr = TSGetRHSJacobian(quadts,&quadJ,NULL,NULL,NULL);CHKERRQ(ierr);
    ierr = TSGetRHSJacobianP(quadts,&quadJp,NULL,NULL);CHKERRQ(ierr);
  }

  /* Build RHS */
  if (th->endpoint) { /* 2-stage method*/
    th->shift = 1./((th->Theta-1.)*th->time_step0);
    ierr = TSComputeIJacobian(ts,th->ptime0,th->X0,th->Xdot,th->shift,J,Jpre,PETSC_FALSE);CHKERRQ(ierr);
    ierr = MatMatMult(J,ts->mat_sensip,MAT_REUSE_MATRIX,PETSC_DEFAULT,&MatDeltaFwdSensip);CHKERRQ(ierr);
    ierr = MatScale(MatDeltaFwdSensip,(th->Theta-1.)/th->Theta);CHKERRQ(ierr);

    /* Add the f_p forcing terms */
    if (ts->Jacp) {
      ierr = TSComputeIJacobianP(ts,th->ptime0,th->X0,th->Xdot,th->shift,ts->Jacp,PETSC_FALSE);CHKERRQ(ierr);
      ierr = MatAXPY(MatDeltaFwdSensip,(th->Theta-1.)/th->Theta,ts->Jacp,SUBSET_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = TSComputeIJacobianP(ts,th->stage_time,ts->vec_sol,th->Xdot,th->shift,ts->Jacp,PETSC_FALSE);CHKERRQ(ierr);
      ierr = MatAXPY(MatDeltaFwdSensip,-1.,ts->Jacp,SUBSET_NONZERO_PATTERN);CHKERRQ(ierr);
    }
  } else { /* 1-stage method */
    th->shift = 0.0;
    ierr = TSComputeIJacobian(ts,th->stage_time,th->X,th->Xdot,th->shift,J,Jpre,PETSC_FALSE);CHKERRQ(ierr);
    ierr = MatMatMult(J,ts->mat_sensip,MAT_REUSE_MATRIX,PETSC_DEFAULT,&MatDeltaFwdSensip);CHKERRQ(ierr);
    ierr = MatScale(MatDeltaFwdSensip,-1.);CHKERRQ(ierr);

    /* Add the f_p forcing terms */
    if (ts->Jacp) {
      ierr = TSComputeIJacobianP(ts,th->stage_time,th->X,th->Xdot,th->shift,ts->Jacp,PETSC_FALSE);CHKERRQ(ierr);
      ierr = MatAXPY(MatDeltaFwdSensip,-1.,ts->Jacp,SUBSET_NONZERO_PATTERN);CHKERRQ(ierr);
    }
  }

  /* Build LHS */
  th->shift = previous_shift; /* recover the previous shift used in TSStep_Theta() */
  if (th->endpoint) {
    ierr = TSComputeIJacobian(ts,th->stage_time,ts->vec_sol,th->Xdot,th->shift,J,Jpre,PETSC_FALSE);CHKERRQ(ierr);
  } else {
    ierr = TSComputeIJacobian(ts,th->stage_time,th->X,th->Xdot,th->shift,J,Jpre,PETSC_FALSE);CHKERRQ(ierr);
  }
  ierr = KSPSetOperators(ksp,J,Jpre);CHKERRQ(ierr);

  /*
    Evaluate the first stage of integral gradients with the 2-stage method:
    drdu|t_n*S(t_n) + drdp|t_n
    This is done before the linear solve because the sensitivity variable S(t_n) will be propagated to S(t_{n+1})
  */
  if (th->endpoint) { /* 2-stage method only */
    if (quadts && quadts->mat_sensip) {
      ierr = TSComputeRHSJacobian(quadts,th->ptime0,th->X0,quadJ,NULL);CHKERRQ(ierr);
      ierr = TSComputeRHSJacobianP(quadts,th->ptime0,th->X0,quadJp);CHKERRQ(ierr);
      ierr = MatTransposeMatMult(ts->mat_sensip,quadJ,MAT_REUSE_MATRIX,PETSC_DEFAULT,&th->MatIntegralSensipTemp);CHKERRQ(ierr);
      ierr = MatAXPY(th->MatIntegralSensipTemp,1,quadJp,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatAXPY(quadts->mat_sensip,th->time_step0*(1.-th->Theta),th->MatIntegralSensipTemp,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    }
  }

  /* Solve the tangent linear equation for forward sensitivities to parameters */
  for (ntlm=0; ntlm<th->num_tlm; ntlm++) {
    KSPConvergedReason kspreason;
    ierr = MatDenseGetColumn(MatDeltaFwdSensip,ntlm,&barr);CHKERRQ(ierr);
    ierr = VecPlaceArray(VecDeltaFwdSensipCol,barr);CHKERRQ(ierr);
    if (th->endpoint) {
      ierr = MatDenseGetColumn(ts->mat_sensip,ntlm,&xarr);CHKERRQ(ierr);
      ierr = VecPlaceArray(ts->vec_sensip_col,xarr);CHKERRQ(ierr);
      ierr = KSPSolve(ksp,VecDeltaFwdSensipCol,ts->vec_sensip_col);CHKERRQ(ierr);
      ierr = VecResetArray(ts->vec_sensip_col);CHKERRQ(ierr);
      ierr = MatDenseRestoreColumn(ts->mat_sensip,&xarr);CHKERRQ(ierr);
    } else {
      ierr = KSPSolve(ksp,VecDeltaFwdSensipCol,VecDeltaFwdSensipCol);CHKERRQ(ierr);
    }
    ierr = KSPGetConvergedReason(ksp,&kspreason);CHKERRQ(ierr);
    if (kspreason < 0) {
      ts->reason = TSFORWARD_DIVERGED_LINEAR_SOLVE;
      ierr = PetscInfo2(ts,"Step=%D, %Dth tangent linear solve, linear solve fails, stopping tangent linear solve\n",ts->steps,ntlm);CHKERRQ(ierr);
    }
    ierr = VecResetArray(VecDeltaFwdSensipCol);CHKERRQ(ierr);
    ierr = MatDenseRestoreColumn(MatDeltaFwdSensip,&barr);CHKERRQ(ierr);
  }

  /*
    Evaluate the second stage of integral gradients with the 2-stage method:
    drdu|t_{n+1}*S(t_{n+1}) + drdp|t_{n+1}
  */
  if (quadts && quadts->mat_sensip) {
    if (!th->endpoint) {
      ierr = MatAXPY(ts->mat_sensip,1,MatDeltaFwdSensip,SAME_NONZERO_PATTERN);CHKERRQ(ierr); /* stage sensitivity */
      ierr = TSComputeRHSJacobian(quadts,th->stage_time,th->X,quadJ,NULL);CHKERRQ(ierr);
      ierr = TSComputeRHSJacobianP(quadts,th->stage_time,th->X,quadJp);CHKERRQ(ierr);
      ierr = MatTransposeMatMult(ts->mat_sensip,quadJ,MAT_REUSE_MATRIX,PETSC_DEFAULT,&th->MatIntegralSensipTemp);CHKERRQ(ierr);
      ierr = MatAXPY(th->MatIntegralSensipTemp,1,quadJp,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatAXPY(quadts->mat_sensip,th->time_step0,th->MatIntegralSensipTemp,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatAXPY(ts->mat_sensip,(1.-th->Theta)/th->Theta,MatDeltaFwdSensip,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    } else {
      ierr = TSComputeRHSJacobian(quadts,th->stage_time,ts->vec_sol,quadJ,NULL);CHKERRQ(ierr);
      ierr = TSComputeRHSJacobianP(quadts,th->stage_time,ts->vec_sol,quadJp);CHKERRQ(ierr);
      ierr = MatTransposeMatMult(ts->mat_sensip,quadJ,MAT_REUSE_MATRIX,PETSC_DEFAULT,&th->MatIntegralSensipTemp);CHKERRQ(ierr);
      ierr = MatAXPY(th->MatIntegralSensipTemp,1,quadJp,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatAXPY(quadts->mat_sensip,th->time_step0*th->Theta,th->MatIntegralSensipTemp,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    }
  } else {
    if (!th->endpoint) {
      ierr = MatAXPY(ts->mat_sensip,1./th->Theta,MatDeltaFwdSensip,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSForwardGetStages_Theta(TS ts,PetscInt *ns,Mat **stagesensip)
{
  TS_Theta *th = (TS_Theta*)ts->data;

  PetscFunctionBegin;
  if (ns) *ns = 1;
  if (stagesensip) *stagesensip = th->endpoint ? &(th->MatFwdSensip0) : &(th->MatDeltaFwdSensip);
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
  PetscFunctionReturn(0);
}

static PetscErrorCode TSAdjointReset_Theta(TS ts)
{
  TS_Theta       *th = (TS_Theta*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroyVecs(ts->numcost,&th->VecsDeltaLam);CHKERRQ(ierr);
  ierr = VecDestroyVecs(ts->numcost,&th->VecsDeltaMu);CHKERRQ(ierr);
  ierr = VecDestroyVecs(ts->numcost,&th->VecsDeltaLam2);CHKERRQ(ierr);
  ierr = VecDestroyVecs(ts->numcost,&th->VecsDeltaMu2);CHKERRQ(ierr);
  ierr = VecDestroyVecs(ts->numcost,&th->VecsSensiTemp);CHKERRQ(ierr);
  ierr = VecDestroyVecs(ts->numcost,&th->VecsSensi2Temp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSDestroy_Theta(TS ts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSReset_Theta(ts);CHKERRQ(ierr);
  if (ts->dm) {
    ierr = DMCoarsenHookRemove(ts->dm,DMCoarsenHook_TSTheta,DMRestrictHook_TSTheta,ts);CHKERRQ(ierr);
    ierr = DMSubDomainHookRemove(ts->dm,DMSubDomainHook_TSTheta,DMSubDomainRestrictHook_TSTheta,ts);CHKERRQ(ierr);
  }
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
  PetscReal      shift = th->shift;

  PetscFunctionBegin;
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  /* When using the endpoint variant, this is actually 1/Theta * Xdot */
  ierr = TSThetaGetX0AndXdot(ts,dm,&X0,&Xdot);CHKERRQ(ierr);
  if (x != X0) {
    ierr = VecAXPBYPCZ(Xdot,-shift,shift,0,X0,x);CHKERRQ(ierr);
  } else {
    ierr = VecZeroEntries(Xdot);CHKERRQ(ierr);
  }
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
  PetscReal      shift = th->shift;

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
  TS             quadts = ts->quadraturets;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* combine sensitivities to parameters and sensitivities to initial values into one array */
  th->num_tlm = ts->num_parameters;
  ierr = MatDuplicate(ts->mat_sensip,MAT_DO_NOT_COPY_VALUES,&th->MatDeltaFwdSensip);CHKERRQ(ierr);
  if (quadts && quadts->mat_sensip) {
    ierr = MatDuplicate(quadts->mat_sensip,MAT_DO_NOT_COPY_VALUES,&th->MatIntegralSensipTemp);CHKERRQ(ierr);
    ierr = MatDuplicate(quadts->mat_sensip,MAT_DO_NOT_COPY_VALUES,&th->MatIntegralSensip0);CHKERRQ(ierr);
  }
  /* backup sensitivity results for roll-backs */
  ierr = MatDuplicate(ts->mat_sensip,MAT_DO_NOT_COPY_VALUES,&th->MatFwdSensip0);CHKERRQ(ierr);

  ierr = VecDuplicate(ts->vec_sol,&th->VecDeltaFwdSensipCol);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSForwardReset_Theta(TS ts)
{
  TS_Theta       *th = (TS_Theta*)ts->data;
  TS             quadts = ts->quadraturets;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (quadts && quadts->mat_sensip) {
    ierr = MatDestroy(&th->MatIntegralSensipTemp);CHKERRQ(ierr);
    ierr = MatDestroy(&th->MatIntegralSensip0);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&th->VecDeltaFwdSensipCol);CHKERRQ(ierr);
  ierr = MatDestroy(&th->MatDeltaFwdSensip);CHKERRQ(ierr);
  ierr = MatDestroy(&th->MatFwdSensip0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSSetUp_Theta(TS ts)
{
  TS_Theta       *th = (TS_Theta*)ts->data;
  TS             quadts = ts->quadraturets;
  PetscBool      match;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!th->VecCostIntegral0 && quadts && ts->costintegralfwd) { /* back up cost integral */
    ierr = VecDuplicate(quadts->vec_sol,&th->VecCostIntegral0);CHKERRQ(ierr);
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
  th->shift = 1/(th->Theta*ts->time_step);

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
  ierr = VecDuplicateVecs(ts->vecs_sensi[0],ts->numcost,&th->VecsSensiTemp);CHKERRQ(ierr);
  if (ts->vecs_sensip) {
    ierr = VecDuplicateVecs(ts->vecs_sensip[0],ts->numcost,&th->VecsDeltaMu);CHKERRQ(ierr);
  }
  if (ts->vecs_sensi2) {
    ierr = VecDuplicateVecs(ts->vecs_sensi[0],ts->numcost,&th->VecsDeltaLam2);CHKERRQ(ierr);
    ierr = VecDuplicateVecs(ts->vecs_sensi2[0],ts->numcost,&th->VecsSensi2Temp);CHKERRQ(ierr);
    /* hack ts to make implicit TS solver work when users provide only explicit versions of callbacks (RHSFunction,RHSJacobian,RHSHessian etc.) */
    if (!ts->ihessianproduct_fuu) ts->vecs_fuu = ts->vecs_guu;
    if (!ts->ihessianproduct_fup) ts->vecs_fup = ts->vecs_gup;
  }
  if (ts->vecs_sensi2p) {
    ierr = VecDuplicateVecs(ts->vecs_sensi2p[0],ts->numcost,&th->VecsDeltaMu2);CHKERRQ(ierr);
    /* hack ts to make implicit TS solver work when users provide only explicit versions of callbacks (RHSFunction,RHSJacobian,RHSHessian etc.) */
    if (!ts->ihessianproduct_fpu) ts->vecs_fpu = ts->vecs_gpu;
    if (!ts->ihessianproduct_fpp) ts->vecs_fpp = ts->vecs_gpp;
  }
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

  ierr = PetscNewLog(ts,&th);CHKERRQ(ierr);
  ts->data = (void*)th;

  th->VecsDeltaLam    = NULL;
  th->VecsDeltaMu     = NULL;
  th->VecsSensiTemp   = NULL;
  th->VecsSensi2Temp  = NULL;

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
