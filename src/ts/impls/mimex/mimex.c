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

  PetscFunctionBegin;
  if (X0) {
    if (dm && dm != ts->dm) CHKERRQ(DMGetNamedGlobalVector(dm, "TSMimex_X0", X0));
    else                    {*X0  = ts->vec_sol;}
  }
  if (Xdot) {
    if (dm && dm != ts->dm) CHKERRQ(DMGetNamedGlobalVector(dm, "TSMimex_Xdot", Xdot));
    else                    {*Xdot = mimex->Xdot;}
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSMimexRestoreX0AndXdot(TS ts, DM dm, Vec *X0, Vec *Xdot)
{
  PetscFunctionBegin;
  if (X0)   if (dm && dm != ts->dm) CHKERRQ(DMRestoreNamedGlobalVector(dm, "TSMimex_X0", X0));
  if (Xdot) if (dm && dm != ts->dm) CHKERRQ(DMRestoreNamedGlobalVector(dm, "TSMimex_Xdot", Xdot));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSMimexGetXstarAndG(TS ts, DM dm, Vec *Xstar, Vec *G)
{
  PetscFunctionBegin;
  CHKERRQ(DMGetNamedGlobalVector(dm, "TSMimex_Xstar", Xstar));
  CHKERRQ(DMGetNamedGlobalVector(dm, "TSMimex_G", G));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSMimexRestoreXstarAndG(TS ts, DM dm, Vec *Xstar, Vec *G)
{
  PetscFunctionBegin;
  CHKERRQ(DMRestoreNamedGlobalVector(dm, "TSMimex_Xstar", Xstar));
  CHKERRQ(DMRestoreNamedGlobalVector(dm, "TSMimex_G", G));
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

  PetscFunctionBegin;
  CHKERRQ(SNESGetDM(snes, &dm));
  CHKERRQ(TSMimexGetX0AndXdot(ts, dm, &X0, &Xdot));
  CHKERRQ(VecAXPBYPCZ(Xdot, -shift, shift, 0, X0, x));

  /* DM monkey-business allows user code to call TSGetDM() inside of functions evaluated on levels of FAS */
  dmsave = ts->dm;
  ts->dm = dm;
  CHKERRQ(TSComputeIFunction(ts, mimex->stage_time, x, Xdot, y, PETSC_TRUE));
  if (mimex->version == 1) {
    DM                 dm;
    PetscDS            prob;
    PetscSection       s;
    Vec                Xstar = NULL, G = NULL;
    const PetscScalar *ax;
    PetscScalar       *axstar;
    PetscInt           Nf, f, pStart, pEnd, p;

    CHKERRQ(TSGetDM(ts, &dm));
    CHKERRQ(DMGetDS(dm, &prob));
    CHKERRQ(DMGetLocalSection(dm, &s));
    CHKERRQ(PetscDSGetNumFields(prob, &Nf));
    CHKERRQ(PetscSectionGetChart(s, &pStart, &pEnd));
    CHKERRQ(TSMimexGetXstarAndG(ts, dm, &Xstar, &G));
    CHKERRQ(VecCopy(X0, Xstar));
    CHKERRQ(VecGetArrayRead(x, &ax));
    CHKERRQ(VecGetArray(Xstar, &axstar));
    for (f = 0; f < Nf; ++f) {
      PetscBool implicit;

      CHKERRQ(PetscDSGetImplicit(prob, f, &implicit));
      if (!implicit) continue;
      for (p = pStart; p < pEnd; ++p) {
        PetscScalar *a, *axs;
        PetscInt     fdof, fcdof, d;

        CHKERRQ(PetscSectionGetFieldDof(s, p, f, &fdof));
        CHKERRQ(PetscSectionGetFieldConstraintDof(s, p, f, &fcdof));
        CHKERRQ(DMPlexPointGlobalFieldRead(dm, p, f, ax, &a));
        CHKERRQ(DMPlexPointGlobalFieldRef(dm, p, f, axstar, &axs));
        for (d = 0; d < fdof-fcdof; ++d) axs[d] = a[d];
      }
    }
    CHKERRQ(VecRestoreArrayRead(x, &ax));
    CHKERRQ(VecRestoreArray(Xstar, &axstar));
    CHKERRQ(TSComputeRHSFunction(ts, ts->ptime, Xstar, G));
    CHKERRQ(VecAXPY(y, -1.0, G));
    CHKERRQ(TSMimexRestoreXstarAndG(ts, dm, &Xstar, &G));
  }
  ts->dm = dmsave;
  CHKERRQ(TSMimexRestoreX0AndXdot(ts, dm, &X0, &Xdot));
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESTSFormJacobian_Mimex(SNES snes, Vec x, Mat A, Mat B, TS ts)
{
  TS_Mimex      *mimex = (TS_Mimex *) ts->data;
  DM             dm, dmsave;
  Vec            Xdot;
  PetscReal      shift = 1./ts->time_step;

  PetscFunctionBegin;
  /* th->Xdot has already been computed in SNESTSFormFunction_Mimex (SNES guarantees this) */
  CHKERRQ(SNESGetDM(snes, &dm));
  CHKERRQ(TSMimexGetX0AndXdot(ts, dm, NULL, &Xdot));

  /* DM monkey-business allows user code to call TSGetDM() inside of functions evaluated on levels of FAS */
  dmsave = ts->dm;
  ts->dm = dm;
  CHKERRQ(TSComputeIJacobian(ts, mimex->stage_time, x, Xdot, shift, A, B, PETSC_TRUE));
  ts->dm = dmsave;
  CHKERRQ(TSMimexRestoreX0AndXdot(ts, dm, NULL, &Xdot));
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

  PetscFunctionBegin;
  CHKERRQ(TSGetDM(ts, &dm));
  CHKERRQ(DMGetDS(dm, &prob));
  CHKERRQ(DMGetLocalSection(dm, &s));
  CHKERRQ(PetscDSGetNumFields(prob, &Nf));
  CHKERRQ(PetscSectionGetChart(s, &pStart, &pEnd));
  CHKERRQ(TSPreStage(ts, ts->ptime));
  /* Compute implicit update */
  mimex->stage_time = ts->ptime + ts->time_step;
  CHKERRQ(VecCopy(sol, update));
  CHKERRQ(SNESSolve(ts->snes, NULL, update));
  CHKERRQ(VecGetArrayRead(update, &aupdate));
  CHKERRQ(VecGetArray(sol, &asol));
  for (f = 0; f < Nf; ++f) {
    PetscBool implicit;

    CHKERRQ(PetscDSGetImplicit(prob, f, &implicit));
    if (!implicit) continue;
    for (p = pStart; p < pEnd; ++p) {
      PetscScalar *au, *as;
      PetscInt     fdof, fcdof, d;

      CHKERRQ(PetscSectionGetFieldDof(s, p, f, &fdof));
      CHKERRQ(PetscSectionGetFieldConstraintDof(s, p, f, &fcdof));
      CHKERRQ(DMPlexPointGlobalFieldRead(dm, p, f, aupdate, &au));
      CHKERRQ(DMPlexPointGlobalFieldRef(dm, p, f, asol, &as));
      for (d = 0; d < fdof-fcdof; ++d) as[d] = au[d];
    }
  }
  CHKERRQ(VecRestoreArrayRead(update, &aupdate));
  CHKERRQ(VecRestoreArray(sol, &asol));
  /* Compute explicit update */
  CHKERRQ(TSComputeRHSFunction(ts, ts->ptime, sol, update));
  CHKERRQ(VecGetArrayRead(update, &aupdate));
  CHKERRQ(VecGetArray(sol, &asol));
  for (f = 0; f < Nf; ++f) {
    PetscBool implicit;

    CHKERRQ(PetscDSGetImplicit(prob, f, &implicit));
    if (implicit) continue;
    for (p = pStart; p < pEnd; ++p) {
      PetscScalar *au, *as;
      PetscInt     fdof, fcdof, d;

      CHKERRQ(PetscSectionGetFieldDof(s, p, f, &fdof));
      CHKERRQ(PetscSectionGetFieldConstraintDof(s, p, f, &fcdof));
      CHKERRQ(DMPlexPointGlobalFieldRead(dm, p, f, aupdate, &au));
      CHKERRQ(DMPlexPointGlobalFieldRef(dm, p, f, asol, &as));
      for (d = 0; d < fdof-fcdof; ++d) as[d] += dt*au[d];
    }
  }
  CHKERRQ(VecRestoreArrayRead(update, &aupdate));
  CHKERRQ(VecRestoreArray(sol, &asol));
  CHKERRQ(TSPostStage(ts, ts->ptime, 0, &sol));
  ts->ptime += ts->time_step;
  PetscFunctionReturn(0);
}

/* Evalute F at U and G at U0 for explicit fields and U for implicit fields */
static PetscErrorCode TSStep_Mimex_Implicit(TS ts)
{
  TS_Mimex      *mimex  = (TS_Mimex *) ts->data;
  Vec            sol    = ts->vec_sol;
  Vec            update = mimex->update;

  PetscFunctionBegin;
  CHKERRQ(TSPreStage(ts, ts->ptime));
  /* Compute implicit update */
  mimex->stage_time = ts->ptime + ts->time_step;
  ts->ptime += ts->time_step;
  CHKERRQ(VecCopy(sol, update));
  CHKERRQ(SNESSolve(ts->snes, NULL, update));
  CHKERRQ(VecCopy(update, sol));
  CHKERRQ(TSPostStage(ts, ts->ptime, 0, &sol));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSStep_Mimex(TS ts)
{
  TS_Mimex       *mimex = (TS_Mimex*)ts->data;

  PetscFunctionBegin;
  switch(mimex->version) {
  case 0:
    CHKERRQ(TSStep_Mimex_Split(ts)); break;
  case 1:
    CHKERRQ(TSStep_Mimex_Implicit(ts)); break;
  default:
    SETERRQ(PetscObjectComm((PetscObject) ts), PETSC_ERR_ARG_OUTOFRANGE, "Unknown MIMEX version %d", mimex->version);
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

static PetscErrorCode TSSetUp_Mimex(TS ts)
{
  TS_Mimex       *mimex = (TS_Mimex*)ts->data;

  PetscFunctionBegin;
  CHKERRQ(VecDuplicate(ts->vec_sol, &mimex->update));
  CHKERRQ(VecDuplicate(ts->vec_sol, &mimex->Xdot));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSReset_Mimex(TS ts)
{
  TS_Mimex       *mimex = (TS_Mimex*)ts->data;

  PetscFunctionBegin;
  CHKERRQ(VecDestroy(&mimex->update));
  CHKERRQ(VecDestroy(&mimex->Xdot));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSDestroy_Mimex(TS ts)
{
  PetscFunctionBegin;
  CHKERRQ(TSReset_Mimex(ts));
  CHKERRQ(PetscFree(ts->data));
  PetscFunctionReturn(0);
}
/*------------------------------------------------------------*/

static PetscErrorCode TSSetFromOptions_Mimex(PetscOptionItems *PetscOptionsObject, TS ts)
{
  TS_Mimex      *mimex = (TS_Mimex *) ts->data;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsHead(PetscOptionsObject, "MIMEX ODE solver options"));
  {
    CHKERRQ(PetscOptionsInt("-ts_mimex_version", "Algorithm version", "TSMimexSetVersion", mimex->version, &mimex->version, NULL));
  }
  CHKERRQ(PetscOptionsTail());
  PetscFunctionReturn(0);
}

static PetscErrorCode TSView_Mimex(TS ts,PetscViewer viewer)
{
  TS_Mimex      *mimex = (TS_Mimex *) ts->data;
  PetscBool      iascii;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer, "  Version = %D\n", mimex->version));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSInterpolate_Mimex(TS ts,PetscReal t,Vec X)
{
  PetscReal      alpha = (ts->ptime - t)/ts->time_step;

  PetscFunctionBegin;
  CHKERRQ(VecAXPBY(ts->vec_sol,1.0-alpha,alpha,X));
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

  CHKERRQ(PetscNewLog(ts,&mimex));
  ts->data = (void*)mimex;

  mimex->version = 1;
  PetscFunctionReturn(0);
}
