/*
  Code for timestepping with discrete gradient integrators
*/
#include <petsc/private/tsimpl.h> /*I   "petscts.h"   I*/
#include <petscdm.h>

PetscBool  DGCite       = PETSC_FALSE;
const char DGCitation[] = "@article{Gonzalez1996,\n"
                          "  title   = {Time integration and discrete Hamiltonian systems},\n"
                          "  author  = {Oscar Gonzalez},\n"
                          "  journal = {Journal of Nonlinear Science},\n"
                          "  volume  = {6},\n"
                          "  pages   = {449--467},\n"
                          "  doi     = {10.1007/978-1-4612-1246-1_10},\n"
                          "  year    = {1996}\n}\n";

typedef struct {
  PetscReal stage_time;
  Vec       X0, X, Xdot;
  void     *funcCtx;
  PetscBool gonzalez;
  PetscErrorCode (*Sfunc)(TS, PetscReal, Vec, Mat, void *);
  PetscErrorCode (*Ffunc)(TS, PetscReal, Vec, PetscScalar *, void *);
  PetscErrorCode (*Gfunc)(TS, PetscReal, Vec, Vec, void *);
} TS_DiscGrad;

static PetscErrorCode TSDiscGradGetX0AndXdot(TS ts, DM dm, Vec *X0, Vec *Xdot)
{
  TS_DiscGrad *dg = (TS_DiscGrad *)ts->data;

  PetscFunctionBegin;
  if (X0) {
    if (dm && dm != ts->dm) PetscCall(DMGetNamedGlobalVector(dm, "TSDiscGrad_X0", X0));
    else *X0 = ts->vec_sol;
  }
  if (Xdot) {
    if (dm && dm != ts->dm) PetscCall(DMGetNamedGlobalVector(dm, "TSDiscGrad_Xdot", Xdot));
    else *Xdot = dg->Xdot;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSDiscGradRestoreX0AndXdot(TS ts, DM dm, Vec *X0, Vec *Xdot)
{
  PetscFunctionBegin;
  if (X0) {
    if (dm && dm != ts->dm) PetscCall(DMRestoreNamedGlobalVector(dm, "TSDiscGrad_X0", X0));
  }
  if (Xdot) {
    if (dm && dm != ts->dm) PetscCall(DMRestoreNamedGlobalVector(dm, "TSDiscGrad_Xdot", Xdot));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCoarsenHook_TSDiscGrad(DM fine, DM coarse, void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMRestrictHook_TSDiscGrad(DM fine, Mat restrct, Vec rscale, Mat inject, DM coarse, void *ctx)
{
  TS  ts = (TS)ctx;
  Vec X0, Xdot, X0_c, Xdot_c;

  PetscFunctionBegin;
  PetscCall(TSDiscGradGetX0AndXdot(ts, fine, &X0, &Xdot));
  PetscCall(TSDiscGradGetX0AndXdot(ts, coarse, &X0_c, &Xdot_c));
  PetscCall(MatRestrict(restrct, X0, X0_c));
  PetscCall(MatRestrict(restrct, Xdot, Xdot_c));
  PetscCall(VecPointwiseMult(X0_c, rscale, X0_c));
  PetscCall(VecPointwiseMult(Xdot_c, rscale, Xdot_c));
  PetscCall(TSDiscGradRestoreX0AndXdot(ts, fine, &X0, &Xdot));
  PetscCall(TSDiscGradRestoreX0AndXdot(ts, coarse, &X0_c, &Xdot_c));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMSubDomainHook_TSDiscGrad(DM dm, DM subdm, void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMSubDomainRestrictHook_TSDiscGrad(DM dm, VecScatter gscat, VecScatter lscat, DM subdm, void *ctx)
{
  TS  ts = (TS)ctx;
  Vec X0, Xdot, X0_sub, Xdot_sub;

  PetscFunctionBegin;
  PetscCall(TSDiscGradGetX0AndXdot(ts, dm, &X0, &Xdot));
  PetscCall(TSDiscGradGetX0AndXdot(ts, subdm, &X0_sub, &Xdot_sub));

  PetscCall(VecScatterBegin(gscat, X0, X0_sub, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(gscat, X0, X0_sub, INSERT_VALUES, SCATTER_FORWARD));

  PetscCall(VecScatterBegin(gscat, Xdot, Xdot_sub, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(gscat, Xdot, Xdot_sub, INSERT_VALUES, SCATTER_FORWARD));

  PetscCall(TSDiscGradRestoreX0AndXdot(ts, dm, &X0, &Xdot));
  PetscCall(TSDiscGradRestoreX0AndXdot(ts, subdm, &X0_sub, &Xdot_sub));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSSetUp_DiscGrad(TS ts)
{
  TS_DiscGrad *dg = (TS_DiscGrad *)ts->data;
  DM           dm;

  PetscFunctionBegin;
  if (!dg->X) PetscCall(VecDuplicate(ts->vec_sol, &dg->X));
  if (!dg->X0) PetscCall(VecDuplicate(ts->vec_sol, &dg->X0));
  if (!dg->Xdot) PetscCall(VecDuplicate(ts->vec_sol, &dg->Xdot));

  PetscCall(TSGetDM(ts, &dm));
  PetscCall(DMCoarsenHookAdd(dm, DMCoarsenHook_TSDiscGrad, DMRestrictHook_TSDiscGrad, ts));
  PetscCall(DMSubDomainHookAdd(dm, DMSubDomainHook_TSDiscGrad, DMSubDomainRestrictHook_TSDiscGrad, ts));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSSetFromOptions_DiscGrad(TS ts, PetscOptionItems *PetscOptionsObject)
{
  TS_DiscGrad *dg = (TS_DiscGrad *)ts->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "Discrete Gradients ODE solver options");
  {
    PetscCall(PetscOptionsBool("-ts_discgrad_gonzalez", "Use Gonzalez term in discrete gradients formulation", "TSDiscGradUseGonzalez", dg->gonzalez, &dg->gonzalez, NULL));
  }
  PetscOptionsHeadEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode TSView_DiscGrad(TS ts, PetscViewer viewer)
{
  PetscBool iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) PetscCall(PetscViewerASCIIPrintf(viewer, "  Discrete Gradients\n"));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSDiscGradIsGonzalez_DiscGrad(TS ts, PetscBool *gonzalez)
{
  TS_DiscGrad *dg = (TS_DiscGrad *)ts->data;

  PetscFunctionBegin;
  *gonzalez = dg->gonzalez;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSDiscGradUseGonzalez_DiscGrad(TS ts, PetscBool flg)
{
  TS_DiscGrad *dg = (TS_DiscGrad *)ts->data;

  PetscFunctionBegin;
  dg->gonzalez = flg;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSReset_DiscGrad(TS ts)
{
  TS_DiscGrad *dg = (TS_DiscGrad *)ts->data;

  PetscFunctionBegin;
  PetscCall(VecDestroy(&dg->X));
  PetscCall(VecDestroy(&dg->X0));
  PetscCall(VecDestroy(&dg->Xdot));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSDestroy_DiscGrad(TS ts)
{
  DM dm;

  PetscFunctionBegin;
  PetscCall(TSReset_DiscGrad(ts));
  PetscCall(TSGetDM(ts, &dm));
  if (dm) {
    PetscCall(DMCoarsenHookRemove(dm, DMCoarsenHook_TSDiscGrad, DMRestrictHook_TSDiscGrad, ts));
    PetscCall(DMSubDomainHookRemove(dm, DMSubDomainHook_TSDiscGrad, DMSubDomainRestrictHook_TSDiscGrad, ts));
  }
  PetscCall(PetscFree(ts->data));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSDiscGradGetFormulation_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSDiscGradSetFormulation_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSDiscGradIsGonzalez_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSDiscGradUseGonzalez_C", NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSInterpolate_DiscGrad(TS ts, PetscReal t, Vec X)
{
  TS_DiscGrad *dg = (TS_DiscGrad *)ts->data;
  PetscReal    dt = t - ts->ptime;

  PetscFunctionBegin;
  PetscCall(VecCopy(ts->vec_sol, dg->X));
  PetscCall(VecWAXPY(X, dt, dg->Xdot, dg->X));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSDiscGrad_SNESSolve(TS ts, Vec b, Vec x)
{
  SNES     snes;
  PetscInt nits, lits;

  PetscFunctionBegin;
  PetscCall(TSGetSNES(ts, &snes));
  PetscCall(SNESSolve(snes, b, x));
  PetscCall(SNESGetIterationNumber(snes, &nits));
  PetscCall(SNESGetLinearSolveIterations(snes, &lits));
  ts->snes_its += nits;
  ts->ksp_its += lits;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSStep_DiscGrad(TS ts)
{
  TS_DiscGrad *dg = (TS_DiscGrad *)ts->data;
  TSAdapt      adapt;
  TSStepStatus status     = TS_STEP_INCOMPLETE;
  PetscInt     rejections = 0;
  PetscBool    stageok, accept = PETSC_TRUE;
  PetscReal    next_time_step = ts->time_step;

  PetscFunctionBegin;
  PetscCall(TSGetAdapt(ts, &adapt));
  if (!ts->steprollback) PetscCall(VecCopy(ts->vec_sol, dg->X0));

  while (!ts->reason && status != TS_STEP_COMPLETE) {
    PetscReal shift = 1 / (0.5 * ts->time_step);

    dg->stage_time = ts->ptime + 0.5 * ts->time_step;

    PetscCall(VecCopy(dg->X0, dg->X));
    PetscCall(TSPreStage(ts, dg->stage_time));
    PetscCall(TSDiscGrad_SNESSolve(ts, NULL, dg->X));
    PetscCall(TSPostStage(ts, dg->stage_time, 0, &dg->X));
    PetscCall(TSAdaptCheckStage(adapt, ts, dg->stage_time, dg->X, &stageok));
    if (!stageok) goto reject_step;

    status = TS_STEP_PENDING;
    PetscCall(VecAXPBYPCZ(dg->Xdot, -shift, shift, 0, dg->X0, dg->X));
    PetscCall(VecAXPY(ts->vec_sol, ts->time_step, dg->Xdot));
    PetscCall(TSAdaptChoose(adapt, ts, ts->time_step, NULL, &next_time_step, &accept));
    status = accept ? TS_STEP_COMPLETE : TS_STEP_INCOMPLETE;
    if (!accept) {
      PetscCall(VecCopy(dg->X0, ts->vec_sol));
      ts->time_step = next_time_step;
      goto reject_step;
    }
    ts->ptime += ts->time_step;
    ts->time_step = next_time_step;
    break;

  reject_step:
    ts->reject++;
    accept = PETSC_FALSE;
    if (!ts->reason && ts->max_reject >= 0 && ++rejections > ts->max_reject) {
      ts->reason = TS_DIVERGED_STEP_REJECTED;
      PetscCall(PetscInfo(ts, "Step=%" PetscInt_FMT ", step rejections %" PetscInt_FMT " greater than current TS allowed, stopping solve\n", ts->steps, rejections));
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSGetStages_DiscGrad(TS ts, PetscInt *ns, Vec **Y)
{
  TS_DiscGrad *dg = (TS_DiscGrad *)ts->data;

  PetscFunctionBegin;
  if (ns) *ns = 1;
  if (Y) *Y = &(dg->X);
  PetscFunctionReturn(0);
}

/*
  This defines the nonlinear equation that is to be solved with SNES
    G(U) = F[t0 + 0.5*dt, U, (U-U0)/dt] = 0
*/

/* x = (x+x')/2 */
/* NEED TO CALCULATE x_{n+1} from x and x_{n}*/
static PetscErrorCode SNESTSFormFunction_DiscGrad(SNES snes, Vec x, Vec y, TS ts)
{
  TS_DiscGrad *dg = (TS_DiscGrad *)ts->data;
  PetscReal    norm, shift = 1 / (0.5 * ts->time_step);
  PetscInt     n;
  Vec          X0, Xdot, Xp, Xdiff;
  Mat          S;
  PetscScalar  F = 0, F0 = 0, Gp;
  Vec          G, SgF;
  DM           dm, dmsave;

  PetscFunctionBegin;
  PetscCall(SNESGetDM(snes, &dm));

  PetscCall(VecDuplicate(y, &Xp));
  PetscCall(VecDuplicate(y, &Xdiff));
  PetscCall(VecDuplicate(y, &SgF));
  PetscCall(VecDuplicate(y, &G));

  PetscCall(VecGetLocalSize(y, &n));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &S));
  PetscCall(MatSetSizes(S, PETSC_DECIDE, PETSC_DECIDE, n, n));
  PetscCall(MatSetFromOptions(S));
  PetscCall(MatSetUp(S));
  PetscCall(MatAssemblyBegin(S, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(S, MAT_FINAL_ASSEMBLY));

  PetscCall(TSDiscGradGetX0AndXdot(ts, dm, &X0, &Xdot));
  PetscCall(VecAXPBYPCZ(Xdot, -shift, shift, 0, X0, x)); /* Xdot = shift (x - X0) */

  PetscCall(VecAXPBYPCZ(Xp, -1, 2, 0, X0, x));     /* Xp = 2*x - X0 + (0)*Xmid */
  PetscCall(VecAXPBYPCZ(Xdiff, -1, 1, 0, X0, Xp)); /* Xdiff = xp - X0 + (0)*Xdiff */

  if (dg->gonzalez) {
    PetscCall((*dg->Sfunc)(ts, dg->stage_time, x, S, dg->funcCtx));
    PetscCall((*dg->Ffunc)(ts, dg->stage_time, Xp, &F, dg->funcCtx));
    PetscCall((*dg->Ffunc)(ts, dg->stage_time, X0, &F0, dg->funcCtx));
    PetscCall((*dg->Gfunc)(ts, dg->stage_time, x, G, dg->funcCtx));

    /* Adding Extra Gonzalez Term */
    PetscCall(VecDot(Xdiff, G, &Gp));
    PetscCall(VecNorm(Xdiff, NORM_2, &norm));
    if (norm < PETSC_SQRT_MACHINE_EPSILON) {
      Gp = 0;
    } else {
      /* Gp = (1/|xn+1 - xn|^2) * (F(xn+1) - F(xn) - Gp) */
      Gp = (F - F0 - Gp) / PetscSqr(norm);
    }
    PetscCall(VecAXPY(G, Gp, Xdiff));
    PetscCall(MatMult(S, G, SgF)); /* S*gradF */

  } else {
    PetscCall((*dg->Sfunc)(ts, dg->stage_time, x, S, dg->funcCtx));
    PetscCall((*dg->Gfunc)(ts, dg->stage_time, x, G, dg->funcCtx));

    PetscCall(MatMult(S, G, SgF)); /* Xdot = S*gradF */
  }
  /* DM monkey-business allows user code to call TSGetDM() inside of functions evaluated on levels of FAS */
  dmsave = ts->dm;
  ts->dm = dm;
  PetscCall(VecAXPBYPCZ(y, 1, -1, 0, Xdot, SgF));
  ts->dm = dmsave;
  PetscCall(TSDiscGradRestoreX0AndXdot(ts, dm, &X0, &Xdot));

  PetscCall(VecDestroy(&Xp));
  PetscCall(VecDestroy(&Xdiff));
  PetscCall(VecDestroy(&SgF));
  PetscCall(VecDestroy(&G));
  PetscCall(MatDestroy(&S));

  PetscFunctionReturn(0);
}

static PetscErrorCode SNESTSFormJacobian_DiscGrad(SNES snes, Vec x, Mat A, Mat B, TS ts)
{
  TS_DiscGrad *dg    = (TS_DiscGrad *)ts->data;
  PetscReal    shift = 1 / (0.5 * ts->time_step);
  Vec          Xdot;
  DM           dm, dmsave;

  PetscFunctionBegin;
  PetscCall(SNESGetDM(snes, &dm));
  /* Xdot has already been computed in SNESTSFormFunction_DiscGrad (SNES guarantees this) */
  PetscCall(TSDiscGradGetX0AndXdot(ts, dm, NULL, &Xdot));

  dmsave = ts->dm;
  ts->dm = dm;
  PetscCall(TSComputeIJacobian(ts, dg->stage_time, x, Xdot, shift, A, B, PETSC_FALSE));
  ts->dm = dmsave;
  PetscCall(TSDiscGradRestoreX0AndXdot(ts, dm, NULL, &Xdot));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSDiscGradGetFormulation_DiscGrad(TS ts, PetscErrorCode (**Sfunc)(TS, PetscReal, Vec, Mat, void *), PetscErrorCode (**Ffunc)(TS, PetscReal, Vec, PetscScalar *, void *), PetscErrorCode (**Gfunc)(TS, PetscReal, Vec, Vec, void *), void *ctx)
{
  TS_DiscGrad *dg = (TS_DiscGrad *)ts->data;

  PetscFunctionBegin;
  *Sfunc = dg->Sfunc;
  *Ffunc = dg->Ffunc;
  *Gfunc = dg->Gfunc;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSDiscGradSetFormulation_DiscGrad(TS ts, PetscErrorCode (*Sfunc)(TS, PetscReal, Vec, Mat, void *), PetscErrorCode (*Ffunc)(TS, PetscReal, Vec, PetscScalar *, void *), PetscErrorCode (*Gfunc)(TS, PetscReal, Vec, Vec, void *), void *ctx)
{
  TS_DiscGrad *dg = (TS_DiscGrad *)ts->data;

  PetscFunctionBegin;
  dg->Sfunc   = Sfunc;
  dg->Ffunc   = Ffunc;
  dg->Gfunc   = Gfunc;
  dg->funcCtx = ctx;
  PetscFunctionReturn(0);
}

/*MC
  TSDISCGRAD - ODE solver using the discrete gradients version of the implicit midpoint method

  Level: intermediate

  Notes:
  This is the implicit midpoint rule, with an optional term that guarantees the discrete gradient property. This
  timestepper applies to systems of the form
$ u_t = S(u) grad F(u)
  where S(u) is a linear operator, and F is a functional of u.

.seealso: [](chapter_ts), `TSCreate()`, `TSSetType()`, `TS`, `TSDISCGRAD`, `TSDiscGradSetFormulation()`
M*/
PETSC_EXTERN PetscErrorCode TSCreate_DiscGrad(TS ts)
{
  TS_DiscGrad *th;

  PetscFunctionBegin;
  PetscCall(PetscCitationsRegister(DGCitation, &DGCite));
  ts->ops->reset          = TSReset_DiscGrad;
  ts->ops->destroy        = TSDestroy_DiscGrad;
  ts->ops->view           = TSView_DiscGrad;
  ts->ops->setfromoptions = TSSetFromOptions_DiscGrad;
  ts->ops->setup          = TSSetUp_DiscGrad;
  ts->ops->step           = TSStep_DiscGrad;
  ts->ops->interpolate    = TSInterpolate_DiscGrad;
  ts->ops->getstages      = TSGetStages_DiscGrad;
  ts->ops->snesfunction   = SNESTSFormFunction_DiscGrad;
  ts->ops->snesjacobian   = SNESTSFormJacobian_DiscGrad;
  ts->default_adapt_type  = TSADAPTNONE;

  ts->usessnes = PETSC_TRUE;

  PetscCall(PetscNew(&th));
  ts->data = (void *)th;

  th->gonzalez = PETSC_FALSE;

  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSDiscGradGetFormulation_C", TSDiscGradGetFormulation_DiscGrad));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSDiscGradSetFormulation_C", TSDiscGradSetFormulation_DiscGrad));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSDiscGradIsGonzalez_C", TSDiscGradIsGonzalez_DiscGrad));
  PetscCall(PetscObjectComposeFunction((PetscObject)ts, "TSDiscGradUseGonzalez_C", TSDiscGradUseGonzalez_DiscGrad));
  PetscFunctionReturn(0);
}

/*@C
  TSDiscGradGetFormulation - Get the construction method for S, F, and grad F from the formulation u_t = S grad F for `TSDISCGRAD`

  Not Collective

  Input Parameter:
. ts - timestepping context

  Output Parameters:
+ Sfunc - constructor for the S matrix from the formulation
. Ffunc - functional F from the formulation
. Gfunc - constructor for the gradient of F from the formulation
- ctx   - the user context

  Calling sequence of Sfunc:
$ PetscErrorCode func(TS ts, PetscReal time, Vec u, Mat S, void *)

  Calling sequence of Ffunc:
$ PetscErrorCode func(TS ts, PetscReal time, Vec u, PetscScalar *F, void *)

  Calling sequence of Gfunc:
$ PetscErrorCode func(TS ts, PetscReal time, Vec u, Vec G, void *)

  Level: intermediate

.seealso: [](chapter_ts), `TS`, `TSDISCGRAD`, `TSDiscGradSetFormulation()`
@*/
PetscErrorCode TSDiscGradGetFormulation(TS ts, PetscErrorCode (**Sfunc)(TS, PetscReal, Vec, Mat, void *), PetscErrorCode (**Ffunc)(TS, PetscReal, Vec, PetscScalar *, void *), PetscErrorCode (**Gfunc)(TS, PetscReal, Vec, Vec, void *), void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidPointer(Sfunc, 2);
  PetscValidPointer(Ffunc, 3);
  PetscValidPointer(Gfunc, 4);
  PetscUseMethod(ts, "TSDiscGradGetFormulation_C", (TS, PetscErrorCode(**Sfunc)(TS, PetscReal, Vec, Mat, void *), PetscErrorCode(**Ffunc)(TS, PetscReal, Vec, PetscScalar *, void *), PetscErrorCode(**Gfunc)(TS, PetscReal, Vec, Vec, void *), void *), (ts, Sfunc, Ffunc, Gfunc, ctx));
  PetscFunctionReturn(0);
}

/*@C
  TSDiscGradSetFormulation - Set the construction method for S, F, and grad F from the formulation u_t = S(u) grad F(u) for `TSDISCGRAD`

  Not Collective

  Input Parameters:
+ ts    - timestepping context
. Sfunc - constructor for the S matrix from the formulation
. Ffunc - functional F from the formulation
- Gfunc - constructor for the gradient of F from the formulation
  Calling sequence of Sfunc:
$ PetscErrorCode func(TS ts, PetscReal time, Vec u, Mat S, void *)

  Calling sequence of Ffunc:
$ PetscErrorCode func(TS ts, PetscReal time, Vec u, PetscScalar *F, void *)

  Calling sequence of Gfunc:
$ PetscErrorCode func(TS ts, PetscReal time, Vec u, Vec G, void *)

  Level: Intermediate

.seealso: [](chapter_ts), `TSDISCGRAD`, `TSDiscGradGetFormulation()`
@*/
PetscErrorCode TSDiscGradSetFormulation(TS ts, PetscErrorCode (*Sfunc)(TS, PetscReal, Vec, Mat, void *), PetscErrorCode (*Ffunc)(TS, PetscReal, Vec, PetscScalar *, void *), PetscErrorCode (*Gfunc)(TS, PetscReal, Vec, Vec, void *), void *ctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidFunction(Sfunc, 2);
  PetscValidFunction(Ffunc, 3);
  PetscValidFunction(Gfunc, 4);
  PetscTryMethod(ts, "TSDiscGradSetFormulation_C", (TS, PetscErrorCode(*Sfunc)(TS, PetscReal, Vec, Mat, void *), PetscErrorCode(*Ffunc)(TS, PetscReal, Vec, PetscScalar *, void *), PetscErrorCode(*Gfunc)(TS, PetscReal, Vec, Vec, void *), void *), (ts, Sfunc, Ffunc, Gfunc, ctx));
  PetscFunctionReturn(0);
}

/*@
  TSDiscGradIsGonzalez - Checks flag for whether to use additional conservative terms in discrete gradient formulation for `TSDISCGRAD`

  Not Collective

  Input Parameter:
.  ts - timestepping context

  Output Parameter:
.  gonzalez - `PETSC_TRUE` when using the Gonzalez term

  Level: Advanced

.seealso: [](chapter_ts), `TSDISCGRAD`, `TSDiscGradUseGonzalez()`, `TSDISCGRAD`
@*/
PetscErrorCode TSDiscGradIsGonzalez(TS ts, PetscBool *gonzalez)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidBoolPointer(gonzalez, 2);
  PetscUseMethod(ts, "TSDiscGradIsGonzalez_C", (TS, PetscBool *), (ts, gonzalez));
  PetscFunctionReturn(0);
}

/*@
  TSDiscGradUseGonzalez - Sets discrete gradient formulation with or without additional conservative terms.  Without flag, the discrete gradients timestepper is just backwards euler

  Not Collective

  Input Parameters:
+ ts - timestepping context
- flg - `PETSC_TRUE` to use the Gonzalez term

  Options Database Key:
. -ts_discgrad_gonzalez <flg> - use the Gonzalez term for the discrete gradient formulation

  Level: Intermediate

.seealso: [](chapter_ts), `TSDISCGRAD`
@*/
PetscErrorCode TSDiscGradUseGonzalez(TS ts, PetscBool flg)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscTryMethod(ts, "TSDiscGradUseGonzalez_C", (TS, PetscBool), (ts, flg));
  PetscFunctionReturn(0);
}
