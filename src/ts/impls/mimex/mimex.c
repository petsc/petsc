/*
       Code for Timestepping with my makeshift IMEX.
*/
#include <petsc-private/tsimpl.h>                /*I   "petscts.h"   I*/
#include <petscds.h>
#include <petscdmplex.h>

typedef struct {
  Vec       Xdot, update;
  PetscReal stage_time;
} TS_Mimex;

#undef __FUNCT__
#define __FUNCT__ "TSMimexGetX0AndXdot"
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


#undef __FUNCT__
#define __FUNCT__ "TSMimexRestoreX0AndXdot"
static PetscErrorCode TSMimexRestoreX0AndXdot(TS ts, DM dm, Vec *X0, Vec *Xdot)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (X0)   if (dm && dm != ts->dm) {ierr = DMRestoreNamedGlobalVector(dm, "TSMimex_X0", X0);CHKERRQ(ierr);}
  if (Xdot) if (dm && dm != ts->dm) {ierr = DMRestoreNamedGlobalVector(dm, "TSMimex_Xdot", Xdot);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*
  This defines the nonlinear equation that is to be solved with SNES
  G(U) = F[t0+dt, U, (U-U0)*shift] = 0
*/
#undef __FUNCT__
#define __FUNCT__ "SNESTSFormFunction_Mimex"
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
  ts->dm = dmsave;
  ierr   = TSMimexRestoreX0AndXdot(ts, dm, &X0, &Xdot);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESTSFormJacobian_Mimex"
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

#undef __FUNCT__
#define __FUNCT__ "TSStep_Mimex"
static PetscErrorCode TSStep_Mimex(TS ts)
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
  ierr = DMGetDefaultSection(dm, &s);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(prob, &Nf);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(s, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = TSPreStep(ts);CHKERRQ(ierr);
  ierr = TSPreStage(ts, ts->ptime);CHKERRQ(ierr);
  /* Compute implicit update */
  mimex->stage_time = ts->ptime + ts->time_step;
  ierr = VecCopy(sol, update);CHKERRQ(ierr);
  ierr = SNESSolve(ts->snes, NULL, update);CHKERRQ(ierr);
  ierr = VecGetArrayRead(update, &aupdate);CHKERRQ(ierr);
  ierr = VecGetArray(sol, &asol);CHKERRQ(ierr);
  for (f = 0; f < Nf; ++f) {
    PetscObject  obj;
    PetscClassId id;
    PetscInt     pdim, d;

    ierr = PetscDSGetDiscretization(prob, f, &obj);CHKERRQ(ierr);
    ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
    if (id == PETSCFV_CLASSID) continue;
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
    PetscObject  obj;
    PetscClassId id;
    PetscInt     pdim, d;

    ierr = PetscDSGetDiscretization(prob, f, &obj);CHKERRQ(ierr);
    ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
    if (id != PETSCFV_CLASSID) continue;
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
  ts->steps++;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

#undef __FUNCT__
#define __FUNCT__ "TSSetUp_Mimex"
static PetscErrorCode TSSetUp_Mimex(TS ts)
{
  TS_Mimex       *mimex = (TS_Mimex*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDuplicate(ts->vec_sol, &mimex->update);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol, &mimex->Xdot);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSReset_Mimex"
static PetscErrorCode TSReset_Mimex(TS ts)
{
  TS_Mimex       *mimex = (TS_Mimex*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&mimex->update);CHKERRQ(ierr);
  ierr = VecDestroy(&mimex->Xdot);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSDestroy_Mimex"
static PetscErrorCode TSDestroy_Mimex(TS ts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSReset_Mimex(ts);CHKERRQ(ierr);
  ierr = PetscFree(ts->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*------------------------------------------------------------*/

#undef __FUNCT__
#define __FUNCT__ "TSSetFromOptions_Mimex"
static PetscErrorCode TSSetFromOptions_Mimex(PetscOptions *PetscOptionsObject,TS ts)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSView_Mimex"
static PetscErrorCode TSView_Mimex(TS ts,PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSInterpolate_Mimex"
static PetscErrorCode TSInterpolate_Mimex(TS ts,PetscReal t,Vec X)
{
  PetscReal      alpha = (ts->ptime - t)/ts->time_step;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecAXPBY(ts->vec_sol,1.0-alpha,alpha,X);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSComputeLinearStability_Mimex"
PetscErrorCode TSComputeLinearStability_Mimex(TS ts,PetscReal xr,PetscReal xi,PetscReal *yr,PetscReal *yi)
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
#undef __FUNCT__
#define __FUNCT__ "TSCreate_Mimex"
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

  ierr = PetscNewLog(ts,&mimex);CHKERRQ(ierr);
  ts->data = (void*)mimex;
  PetscFunctionReturn(0);
}
