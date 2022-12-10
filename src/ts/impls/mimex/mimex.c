/*
       Code for Timestepping with my makeshift IMEX.
*/
#include <petsc/private/tsimpl.h> /*I   "petscts.h"   I*/
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
  TS_Mimex *mimex = (TS_Mimex *)ts->data;

  PetscFunctionBegin;
  if (X0) {
    if (dm && dm != ts->dm) PetscCall(DMGetNamedGlobalVector(dm, "TSMimex_X0", X0));
    else *X0 = ts->vec_sol;
  }
  if (Xdot) {
    if (dm && dm != ts->dm) PetscCall(DMGetNamedGlobalVector(dm, "TSMimex_Xdot", Xdot));
    else *Xdot = mimex->Xdot;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSMimexRestoreX0AndXdot(TS ts, DM dm, Vec *X0, Vec *Xdot)
{
  PetscFunctionBegin;
  if (X0)
    if (dm && dm != ts->dm) PetscCall(DMRestoreNamedGlobalVector(dm, "TSMimex_X0", X0));
  if (Xdot)
    if (dm && dm != ts->dm) PetscCall(DMRestoreNamedGlobalVector(dm, "TSMimex_Xdot", Xdot));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSMimexGetXstarAndG(TS ts, DM dm, Vec *Xstar, Vec *G)
{
  PetscFunctionBegin;
  PetscCall(DMGetNamedGlobalVector(dm, "TSMimex_Xstar", Xstar));
  PetscCall(DMGetNamedGlobalVector(dm, "TSMimex_G", G));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSMimexRestoreXstarAndG(TS ts, DM dm, Vec *Xstar, Vec *G)
{
  PetscFunctionBegin;
  PetscCall(DMRestoreNamedGlobalVector(dm, "TSMimex_Xstar", Xstar));
  PetscCall(DMRestoreNamedGlobalVector(dm, "TSMimex_G", G));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  This defines the nonlinear equation that is to be solved with SNES
  G(U) = F[t0+dt, U, (U-U0)*shift] = 0
*/
static PetscErrorCode SNESTSFormFunction_Mimex(SNES snes, Vec x, Vec y, TS ts)
{
  TS_Mimex *mimex = (TS_Mimex *)ts->data;
  DM        dm, dmsave;
  Vec       X0, Xdot;
  PetscReal shift = 1. / ts->time_step;

  PetscFunctionBegin;
  PetscCall(SNESGetDM(snes, &dm));
  PetscCall(TSMimexGetX0AndXdot(ts, dm, &X0, &Xdot));
  PetscCall(VecAXPBYPCZ(Xdot, -shift, shift, 0, X0, x));

  /* DM monkey-business allows user code to call TSGetDM() inside of functions evaluated on levels of FAS */
  dmsave = ts->dm;
  ts->dm = dm;
  PetscCall(TSComputeIFunction(ts, mimex->stage_time, x, Xdot, y, PETSC_TRUE));
  if (mimex->version == 1) {
    DM                 dm;
    PetscDS            prob;
    PetscSection       s;
    Vec                Xstar = NULL, G = NULL;
    const PetscScalar *ax;
    PetscScalar       *axstar;
    PetscInt           Nf, f, pStart, pEnd, p;

    PetscCall(TSGetDM(ts, &dm));
    PetscCall(DMGetDS(dm, &prob));
    PetscCall(DMGetLocalSection(dm, &s));
    PetscCall(PetscDSGetNumFields(prob, &Nf));
    PetscCall(PetscSectionGetChart(s, &pStart, &pEnd));
    PetscCall(TSMimexGetXstarAndG(ts, dm, &Xstar, &G));
    PetscCall(VecCopy(X0, Xstar));
    PetscCall(VecGetArrayRead(x, &ax));
    PetscCall(VecGetArray(Xstar, &axstar));
    for (f = 0; f < Nf; ++f) {
      PetscBool implicit;

      PetscCall(PetscDSGetImplicit(prob, f, &implicit));
      if (!implicit) continue;
      for (p = pStart; p < pEnd; ++p) {
        PetscScalar *a, *axs;
        PetscInt     fdof, fcdof, d;

        PetscCall(PetscSectionGetFieldDof(s, p, f, &fdof));
        PetscCall(PetscSectionGetFieldConstraintDof(s, p, f, &fcdof));
        PetscCall(DMPlexPointGlobalFieldRead(dm, p, f, ax, &a));
        PetscCall(DMPlexPointGlobalFieldRef(dm, p, f, axstar, &axs));
        for (d = 0; d < fdof - fcdof; ++d) axs[d] = a[d];
      }
    }
    PetscCall(VecRestoreArrayRead(x, &ax));
    PetscCall(VecRestoreArray(Xstar, &axstar));
    PetscCall(TSComputeRHSFunction(ts, ts->ptime, Xstar, G));
    PetscCall(VecAXPY(y, -1.0, G));
    PetscCall(TSMimexRestoreXstarAndG(ts, dm, &Xstar, &G));
  }
  ts->dm = dmsave;
  PetscCall(TSMimexRestoreX0AndXdot(ts, dm, &X0, &Xdot));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SNESTSFormJacobian_Mimex(SNES snes, Vec x, Mat A, Mat B, TS ts)
{
  TS_Mimex *mimex = (TS_Mimex *)ts->data;
  DM        dm, dmsave;
  Vec       Xdot;
  PetscReal shift = 1. / ts->time_step;

  PetscFunctionBegin;
  /* th->Xdot has already been computed in SNESTSFormFunction_Mimex (SNES guarantees this) */
  PetscCall(SNESGetDM(snes, &dm));
  PetscCall(TSMimexGetX0AndXdot(ts, dm, NULL, &Xdot));

  /* DM monkey-business allows user code to call TSGetDM() inside of functions evaluated on levels of FAS */
  dmsave = ts->dm;
  ts->dm = dm;
  PetscCall(TSComputeIJacobian(ts, mimex->stage_time, x, Xdot, shift, A, B, PETSC_TRUE));
  ts->dm = dmsave;
  PetscCall(TSMimexRestoreX0AndXdot(ts, dm, NULL, &Xdot));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSStep_Mimex_Split(TS ts)
{
  TS_Mimex          *mimex = (TS_Mimex *)ts->data;
  DM                 dm;
  PetscDS            prob;
  PetscSection       s;
  Vec                sol = ts->vec_sol, update = mimex->update;
  const PetscScalar *aupdate;
  PetscScalar       *asol, dt = ts->time_step;
  PetscInt           Nf, f, pStart, pEnd, p;

  PetscFunctionBegin;
  PetscCall(TSGetDM(ts, &dm));
  PetscCall(DMGetDS(dm, &prob));
  PetscCall(DMGetLocalSection(dm, &s));
  PetscCall(PetscDSGetNumFields(prob, &Nf));
  PetscCall(PetscSectionGetChart(s, &pStart, &pEnd));
  PetscCall(TSPreStage(ts, ts->ptime));
  /* Compute implicit update */
  mimex->stage_time = ts->ptime + ts->time_step;
  PetscCall(VecCopy(sol, update));
  PetscCall(SNESSolve(ts->snes, NULL, update));
  PetscCall(VecGetArrayRead(update, &aupdate));
  PetscCall(VecGetArray(sol, &asol));
  for (f = 0; f < Nf; ++f) {
    PetscBool implicit;

    PetscCall(PetscDSGetImplicit(prob, f, &implicit));
    if (!implicit) continue;
    for (p = pStart; p < pEnd; ++p) {
      PetscScalar *au, *as;
      PetscInt     fdof, fcdof, d;

      PetscCall(PetscSectionGetFieldDof(s, p, f, &fdof));
      PetscCall(PetscSectionGetFieldConstraintDof(s, p, f, &fcdof));
      PetscCall(DMPlexPointGlobalFieldRead(dm, p, f, aupdate, &au));
      PetscCall(DMPlexPointGlobalFieldRef(dm, p, f, asol, &as));
      for (d = 0; d < fdof - fcdof; ++d) as[d] = au[d];
    }
  }
  PetscCall(VecRestoreArrayRead(update, &aupdate));
  PetscCall(VecRestoreArray(sol, &asol));
  /* Compute explicit update */
  PetscCall(TSComputeRHSFunction(ts, ts->ptime, sol, update));
  PetscCall(VecGetArrayRead(update, &aupdate));
  PetscCall(VecGetArray(sol, &asol));
  for (f = 0; f < Nf; ++f) {
    PetscBool implicit;

    PetscCall(PetscDSGetImplicit(prob, f, &implicit));
    if (implicit) continue;
    for (p = pStart; p < pEnd; ++p) {
      PetscScalar *au, *as;
      PetscInt     fdof, fcdof, d;

      PetscCall(PetscSectionGetFieldDof(s, p, f, &fdof));
      PetscCall(PetscSectionGetFieldConstraintDof(s, p, f, &fcdof));
      PetscCall(DMPlexPointGlobalFieldRead(dm, p, f, aupdate, &au));
      PetscCall(DMPlexPointGlobalFieldRef(dm, p, f, asol, &as));
      for (d = 0; d < fdof - fcdof; ++d) as[d] += dt * au[d];
    }
  }
  PetscCall(VecRestoreArrayRead(update, &aupdate));
  PetscCall(VecRestoreArray(sol, &asol));
  PetscCall(TSPostStage(ts, ts->ptime, 0, &sol));
  ts->ptime += ts->time_step;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Evaluate F at U and G at U0 for explicit fields and U for implicit fields */
static PetscErrorCode TSStep_Mimex_Implicit(TS ts)
{
  TS_Mimex *mimex  = (TS_Mimex *)ts->data;
  Vec       sol    = ts->vec_sol;
  Vec       update = mimex->update;

  PetscFunctionBegin;
  PetscCall(TSPreStage(ts, ts->ptime));
  /* Compute implicit update */
  mimex->stage_time = ts->ptime + ts->time_step;
  ts->ptime += ts->time_step;
  PetscCall(VecCopy(sol, update));
  PetscCall(SNESSolve(ts->snes, NULL, update));
  PetscCall(VecCopy(update, sol));
  PetscCall(TSPostStage(ts, ts->ptime, 0, &sol));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSStep_Mimex(TS ts)
{
  TS_Mimex *mimex = (TS_Mimex *)ts->data;

  PetscFunctionBegin;
  switch (mimex->version) {
  case 0:
    PetscCall(TSStep_Mimex_Split(ts));
    break;
  case 1:
    PetscCall(TSStep_Mimex_Implicit(ts));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)ts), PETSC_ERR_ARG_OUTOFRANGE, "Unknown MIMEX version %" PetscInt_FMT, mimex->version);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*------------------------------------------------------------*/

static PetscErrorCode TSSetUp_Mimex(TS ts)
{
  TS_Mimex *mimex = (TS_Mimex *)ts->data;

  PetscFunctionBegin;
  PetscCall(VecDuplicate(ts->vec_sol, &mimex->update));
  PetscCall(VecDuplicate(ts->vec_sol, &mimex->Xdot));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSReset_Mimex(TS ts)
{
  TS_Mimex *mimex = (TS_Mimex *)ts->data;

  PetscFunctionBegin;
  PetscCall(VecDestroy(&mimex->update));
  PetscCall(VecDestroy(&mimex->Xdot));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSDestroy_Mimex(TS ts)
{
  PetscFunctionBegin;
  PetscCall(TSReset_Mimex(ts));
  PetscCall(PetscFree(ts->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}
/*------------------------------------------------------------*/

static PetscErrorCode TSSetFromOptions_Mimex(TS ts, PetscOptionItems *PetscOptionsObject)
{
  TS_Mimex *mimex = (TS_Mimex *)ts->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "MIMEX ODE solver options");
  {
    PetscCall(PetscOptionsInt("-ts_mimex_version", "Algorithm version", "TSMimexSetVersion", mimex->version, &mimex->version, NULL));
  }
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSView_Mimex(TS ts, PetscViewer viewer)
{
  TS_Mimex *mimex = (TS_Mimex *)ts->data;
  PetscBool iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) PetscCall(PetscViewerASCIIPrintf(viewer, "  Version = %" PetscInt_FMT "\n", mimex->version));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSInterpolate_Mimex(TS ts, PetscReal t, Vec X)
{
  PetscReal alpha = (ts->ptime - t) / ts->time_step;

  PetscFunctionBegin;
  PetscCall(VecAXPBY(ts->vec_sol, 1.0 - alpha, alpha, X));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TSComputeLinearStability_Mimex(TS ts, PetscReal xr, PetscReal xi, PetscReal *yr, PetscReal *yi)
{
  PetscFunctionBegin;
  *yr = 1.0 + xr;
  *yi = xi;
  PetscFunctionReturn(PETSC_SUCCESS);
}
/* ------------------------------------------------------------ */

/*MC
      TSMIMEX - ODE solver using the explicit forward Mimex method

  Level: beginner

.seealso: [](chapter_ts), `TSCreate()`, `TS`, `TSSetType()`, `TSBEULER`
M*/
PETSC_EXTERN PetscErrorCode TSCreate_Mimex(TS ts)
{
  TS_Mimex *mimex;

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

  PetscCall(PetscNew(&mimex));
  ts->data = (void *)mimex;

  mimex->version = 1;
  PetscFunctionReturn(PETSC_SUCCESS);
}
