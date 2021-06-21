/*
       Code for Timestepping with my makeshift IMEX.
*/
#include <petsc/private/tsimpl.h>                /*I   "petscts.h"   I*/
#include <petscds.h>
#include <petscsection.h>
#include <petscdmplex.h>

typedef struct {
  Vec       Xdot, update;
  PetscReal stage_time;
  PetscInt  version;
} TS_Mimex;

static PetscErrorCode TSMimexGetX0AndXdot(TS ts, DM dm, Vec *X0, Vec *Xdot)
{
  TS_Mimex      *mimex = (TS_Mimex *) ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (X0) {
    if (dm && dm != ts->dm) {ierr = DMGetNamedGlobalVector(dm, "TSMimex_X0", X0);CHKERRQ(ierr);}
    else                    {*X0  = ts->vec_sol;}
  }
  if (Xdot) {
    if (dm && dm != ts->dm) {ierr  = DMGetNamedGlobalVector(dm, "TSMimex_Xdot", Xdot);CHKERRQ(ierr);}
    else                    {*Xdot = mimex->Xdot;}
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSMimexRestoreX0AndXdot(TS ts, DM dm, Vec *X0, Vec *Xdot)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (X0)   if (dm && dm != ts->dm) {ierr = DMRestoreNamedGlobalVector(dm, "TSMimex_X0", X0);CHKERRQ(ierr);}
  if (Xdot) if (dm && dm != ts->dm) {ierr = DMRestoreNamedGlobalVector(dm, "TSMimex_Xdot", Xdot);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode TSMimexGetXstarAndG(TS ts, DM dm, Vec *Xstar, Vec *G)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetNamedGlobalVector(dm, "TSMimex_Xstar", Xstar);CHKERRQ(ierr);
  ierr = DMGetNamedGlobalVector(dm, "TSMimex_G", G);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSMimexRestoreXstarAndG(TS ts, DM dm, Vec *Xstar, Vec *G)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMRestoreNamedGlobalVector(dm, "TSMimex_Xstar", Xstar);CHKERRQ(ierr);
  ierr = DMRestoreNamedGlobalVector(dm, "TSMimex_G", G);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  This defines the nonlinear equation that is to be solved with SNES
  G(U) = F[t0+dt, U, (U-U0)*shift] = 0
*/
static PetscErrorCode SNESTSFormFunction_Mimex(SNES snes, Vec x, Vec y, TS ts)
{
  TS_Mimex      *mimex = (TS_Mimex *) ts->data;
  DM             dm, dmsave;
  Vec            X0, Xdot;
  PetscReal      shift = 1./ts->time_step;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESGetDM(snes, &dm);CHKERRQ(ierr);
  ierr = TSMimexGetX0AndXdot(ts, dm, &X0, &Xdot);CHKERRQ(ierr);
  ierr = VecAXPBYPCZ(Xdot, -shift, shift, 0, X0, x);CHKERRQ(ierr);

  /* DM monkey-business allows user code to call TSGetDM() inside of functions evaluated on levels of FAS */
  dmsave = ts->dm;
  ts->dm = dm;
  ierr   = TSComputeIFunction(ts, mimex->stage_time, x, Xdot, y, PETSC_TRUE);CHKERRQ(ierr);
  if (mimex->version == 1) {
    DM                 dm;
    PetscDS            prob;
    PetscSection       s;
    Vec                Xstar = NULL, G = NULL;
    const PetscScalar *ax;
    PetscScalar       *axstar;
    PetscInt           Nf, f, pStart, pEnd, p;

    ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
    ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
    ierr = DMGetLocalSection(dm, &s);CHKERRQ(ierr);
    ierr = PetscDSGetNumFields(prob, &Nf);CHKERRQ(ierr);
    ierr = PetscSectionGetChart(s, &pStart, &pEnd);CHKERRQ(ierr);
    ierr = TSMimexGetXstarAndG(ts, dm, &Xstar, &G);CHKERRQ(ierr);
    ierr = VecCopy(X0, Xstar);CHKERRQ(ierr);
    ierr = VecGetArrayRead(x, &ax);CHKERRQ(ierr);
    ierr = VecGetArray(Xstar, &axstar);CHKERRQ(ierr);
    for (f = 0; f < Nf; ++f) {
      PetscBool implicit;

      ierr = PetscDSGetImplicit(prob, f, &implicit);CHKERRQ(ierr);
      if (!implicit) continue;
      for (p = pStart; p < pEnd; ++p) {
        PetscScalar *a, *axs;
        PetscInt     fdof, fcdof, d;

        ierr = PetscSectionGetFieldDof(s, p, f, &fdof);CHKERRQ(ierr);
        ierr = PetscSectionGetFieldConstraintDof(s, p, f, &fcdof);CHKERRQ(ierr);
        ierr = DMPlexPointGlobalFieldRead(dm, p, f, ax, &a);CHKERRQ(ierr);
        ierr = DMPlexPointGlobalFieldRef(dm, p, f, axstar, &axs);CHKERRQ(ierr);
        for (d = 0; d < fdof-fcdof; ++d) axs[d] = a[d];
      }
    }
    ierr = VecRestoreArrayRead(x, &ax);CHKERRQ(ierr);
    ierr = VecRestoreArray(Xstar, &axstar);CHKERRQ(ierr);
    ierr = TSComputeRHSFunction(ts, ts->ptime, Xstar, G);CHKERRQ(ierr);
    ierr = VecAXPY(y, -1.0, G);CHKERRQ(ierr);
    ierr = TSMimexRestoreXstarAndG(ts, dm, &Xstar, &G);CHKERRQ(ierr);
  }
  ts->dm = dmsave;
  ierr   = TSMimexRestoreX0AndXdot(ts, dm, &X0, &Xdot);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESTSFormJacobian_Mimex(SNES snes, Vec x, Mat A, Mat B, TS ts)
{
  TS_Mimex      *mimex = (TS_Mimex *) ts->data;
  DM             dm, dmsave;
  Vec            Xdot;
  PetscReal      shift = 1./ts->time_step;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* th->Xdot has already been computed in SNESTSFormFunction_Mimex (SNES guarantees this) */
  ierr = SNESGetDM(snes, &dm);CHKERRQ(ierr);
  ierr = TSMimexGetX0AndXdot(ts, dm, NULL, &Xdot);CHKERRQ(ierr);

  /* DM monkey-business allows user code to call TSGetDM() inside of functions evaluated on levels of FAS */
  dmsave = ts->dm;
  ts->dm = dm;
  ierr   = TSComputeIJacobian(ts, mimex->stage_time, x, Xdot, shift, A, B, PETSC_TRUE);CHKERRQ(ierr);
  ts->dm = dmsave;
  ierr   = TSMimexRestoreX0AndXdot(ts, dm, NULL, &Xdot);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSStep_Mimex_Split(TS ts)
{
  TS_Mimex          *mimex = (TS_Mimex *) ts->data;
  DM                 dm;
  PetscDS            prob;
  PetscSection       s;
  Vec                sol = ts->vec_sol, update = mimex->update;
  const PetscScalar *aupdate;
  PetscScalar       *asol, dt = ts->time_step;
  PetscInt           Nf, f, pStart, pEnd, p;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = DMGetLocalSection(dm, &s);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(prob, &Nf);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(s, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = TSPreStage(ts, ts->ptime);CHKERRQ(ierr);
  /* Compute implicit update */
  mimex->stage_time = ts->ptime + ts->time_step;
  ierr = VecCopy(sol, update);CHKERRQ(ierr);
  ierr = SNESSolve(ts->snes, NULL, update);CHKERRQ(ierr);
  ierr = VecGetArrayRead(update, &aupdate);CHKERRQ(ierr);
  ierr = VecGetArray(sol, &asol);CHKERRQ(ierr);
  for (f = 0; f < Nf; ++f) {
    PetscBool implicit;

    ierr = PetscDSGetImplicit(prob, f, &implicit);CHKERRQ(ierr);
    if (!implicit) continue;
    for (p = pStart; p < pEnd; ++p) {
      PetscScalar *au, *as;
      PetscInt     fdof, fcdof, d;

      ierr = PetscSectionGetFieldDof(s, p, f, &fdof);CHKERRQ(ierr);
      ierr = PetscSectionGetFieldConstraintDof(s, p, f, &fcdof);CHKERRQ(ierr);
      ierr = DMPlexPointGlobalFieldRead(dm, p, f, aupdate, &au);CHKERRQ(ierr);
      ierr = DMPlexPointGlobalFieldRef(dm, p, f, asol, &as);CHKERRQ(ierr);
      for (d = 0; d < fdof-fcdof; ++d) as[d] = au[d];
    }
  }
  ierr = VecRestoreArrayRead(update, &aupdate);CHKERRQ(ierr);
  ierr = VecRestoreArray(sol, &asol);CHKERRQ(ierr);
  /* Compute explicit update */
  ierr = TSComputeRHSFunction(ts, ts->ptime, sol, update);CHKERRQ(ierr);
  ierr = VecGetArrayRead(update, &aupdate);CHKERRQ(ierr);
  ierr = VecGetArray(sol, &asol);CHKERRQ(ierr);
  for (f = 0; f < Nf; ++f) {
    PetscBool implicit;

    ierr = PetscDSGetImplicit(prob, f, &implicit);CHKERRQ(ierr);
    if (implicit) continue;
    for (p = pStart; p < pEnd; ++p) {
      PetscScalar *au, *as;
      PetscInt     fdof, fcdof, d;

      ierr = PetscSectionGetFieldDof(s, p, f, &fdof);CHKERRQ(ierr);
      ierr = PetscSectionGetFieldConstraintDof(s, p, f, &fcdof);CHKERRQ(ierr);
      ierr = DMPlexPointGlobalFieldRead(dm, p, f, aupdate, &au);CHKERRQ(ierr);
      ierr = DMPlexPointGlobalFieldRef(dm, p, f, asol, &as);CHKERRQ(ierr);
      for (d = 0; d < fdof-fcdof; ++d) as[d] += dt*au[d];
    }
  }
  ierr = VecRestoreArrayRead(update, &aupdate);CHKERRQ(ierr);
  ierr = VecRestoreArray(sol, &asol);CHKERRQ(ierr);
  ierr = TSPostStage(ts, ts->ptime, 0, &sol);CHKERRQ(ierr);
  ts->ptime += ts->time_step;
  PetscFunctionReturn(0);
}

/* Evalute F at U and G at U0 for explicit fields and U for implicit fields */
static PetscErrorCode TSStep_Mimex_Implicit(TS ts)
{
  TS_Mimex      *mimex  = (TS_Mimex *) ts->data;
  Vec            sol    = ts->vec_sol;
  Vec            update = mimex->update;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSPreStage(ts, ts->ptime);CHKERRQ(ierr);
  /* Compute implicit update */
  mimex->stage_time = ts->ptime + ts->time_step;
  ts->ptime += ts->time_step;
  ierr = VecCopy(sol, update);CHKERRQ(ierr);
  ierr = SNESSolve(ts->snes, NULL, update);CHKERRQ(ierr);
  ierr = VecCopy(update, sol);CHKERRQ(ierr);
  ierr = TSPostStage(ts, ts->ptime, 0, &sol);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSStep_Mimex(TS ts)
{
  TS_Mimex       *mimex = (TS_Mimex*)ts->data;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  switch(mimex->version) {
  case 0:
    ierr = TSStep_Mimex_Split(ts);CHKERRQ(ierr); break;
  case 1:
    ierr = TSStep_Mimex_Implicit(ts);CHKERRQ(ierr); break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject) ts), PETSC_ERR_ARG_OUTOFRANGE, "Unknown MIMEX version %d", mimex->version);
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode TSSetUp_Mimex(TS ts)
{
  TS_Mimex       *mimex = (TS_Mimex*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDuplicate(ts->vec_sol, &mimex->update);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol, &mimex->Xdot);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSReset_Mimex(TS ts)
{
  TS_Mimex       *mimex = (TS_Mimex*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&mimex->update);CHKERRQ(ierr);
  ierr = VecDestroy(&mimex->Xdot);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSDestroy_Mimex(TS ts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSReset_Mimex(ts);CHKERRQ(ierr);
  ierr = PetscFree(ts->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*------------------------------------------------------------*/

static PetscErrorCode TSSetFromOptions_Mimex(PetscOptionItems *PetscOptionsObject, TS ts)
{
  TS_Mimex      *mimex = (TS_Mimex *) ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject, "MIMEX ODE solver options");CHKERRQ(ierr);
  {
    ierr = PetscOptionsInt("-ts_mimex_version", "Algorithm version", "TSMimexSetVersion", mimex->version, &mimex->version, NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSView_Mimex(TS ts,PetscViewer viewer)
{
  TS_Mimex      *mimex = (TS_Mimex *) ts->data;
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer, "  Version = %D\n", mimex->version);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSInterpolate_Mimex(TS ts,PetscReal t,Vec X)
{
  PetscReal      alpha = (ts->ptime - t)/ts->time_step;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecAXPBY(ts->vec_sol,1.0-alpha,alpha,X);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSComputeLinearStability_Mimex(TS ts,PetscReal xr,PetscReal xi,PetscReal *yr,PetscReal *yi)
{
  PetscFunctionBegin;
  *yr = 1.0 + xr;
  *yi = xi;
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------------ */

/*MC
      TSMIMEX - ODE solver using the explicit forward Mimex method

  Level: beginner

.seealso:  TSCreate(), TS, TSSetType(), TSBEULER

M*/
PETSC_EXTERN PetscErrorCode TSCreate_Mimex(TS ts)
{
  TS_Mimex       *mimex;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ts->ops->setup           = TSSetUp_Mimex;
  ts->ops->step            = TSStep_Mimex;
  ts->ops->reset           = TSReset_Mimex;
  ts->ops->destroy         = TSDestroy_Mimex;
  ts->ops->setfromoptions  = TSSetFromOptions_Mimex;
  ts->ops->view            = TSView_Mimex;
  ts->ops->interpolate     = TSInterpolate_Mimex;
  ts->ops->linearstability = TSComputeLinearStability_Mimex;
  ts->ops->snesfunction    = SNESTSFormFunction_Mimex;
  ts->ops->snesjacobian    = SNESTSFormJacobian_Mimex;
  ts->default_adapt_type   = TSADAPTNONE;

  ierr = PetscNewLog(ts,&mimex);CHKERRQ(ierr);
  ts->data = (void*)mimex;

  mimex->version = 1;
  PetscFunctionReturn(0);
}
