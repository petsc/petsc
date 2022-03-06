/*
  Code for timestepping with discrete gradient integrators
*/
#include <petsc/private/tsimpl.h>                /*I   "petscts.h"   I*/
#include <petscdm.h>

PetscBool DGCite = PETSC_FALSE;
const char DGCitation[] = "@article{Gonzalez1996,\n"
                          "  title   = {Time integration and discrete Hamiltonian systems},\n"
                          "  author  = {Oscar Gonzalez},\n"
                          "  journal = {Journal of Nonlinear Science},\n"
                          "  volume  = {6},\n"
                          "  pages   = {449--467},\n"
                          "  doi     = {10.1007/978-1-4612-1246-1_10},\n"
                          "  year    = {1996}\n}\n";

typedef struct {
  PetscReal    stage_time;
  Vec          X0, X, Xdot;
  void        *funcCtx;
  PetscBool    gonzalez;
  PetscErrorCode (*Sfunc)(TS, PetscReal, Vec, Mat, void *);
  PetscErrorCode (*Ffunc)(TS, PetscReal, Vec, PetscScalar *, void *);
  PetscErrorCode (*Gfunc)(TS, PetscReal, Vec, Vec, void *);
} TS_DiscGrad;

static PetscErrorCode TSDiscGradGetX0AndXdot(TS ts, DM dm, Vec *X0, Vec *Xdot)
{
  TS_DiscGrad   *dg = (TS_DiscGrad *) ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (X0) {
    if (dm && dm != ts->dm) {ierr = DMGetNamedGlobalVector(dm, "TSDiscGrad_X0", X0);CHKERRQ(ierr);}
    else                    {*X0  = ts->vec_sol;}
  }
  if (Xdot) {
    if (dm && dm != ts->dm) {ierr = DMGetNamedGlobalVector(dm, "TSDiscGrad_Xdot", Xdot);CHKERRQ(ierr);}
    else                    {*Xdot = dg->Xdot;}
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSDiscGradRestoreX0AndXdot(TS ts, DM dm, Vec *X0, Vec *Xdot)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (X0) {
    if (dm && dm != ts->dm) {ierr = DMRestoreNamedGlobalVector(dm, "TSDiscGrad_X0", X0);CHKERRQ(ierr);}
  }
  if (Xdot) {
    if (dm && dm != ts->dm) {ierr = DMRestoreNamedGlobalVector(dm, "TSDiscGrad_Xdot", Xdot);CHKERRQ(ierr);}
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
  TS             ts = (TS) ctx;
  Vec            X0, Xdot, X0_c, Xdot_c;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSDiscGradGetX0AndXdot(ts, fine, &X0, &Xdot);CHKERRQ(ierr);
  ierr = TSDiscGradGetX0AndXdot(ts, coarse, &X0_c, &Xdot_c);CHKERRQ(ierr);
  ierr = MatRestrict(restrct, X0, X0_c);CHKERRQ(ierr);
  ierr = MatRestrict(restrct, Xdot, Xdot_c);CHKERRQ(ierr);
  ierr = VecPointwiseMult(X0_c, rscale, X0_c);CHKERRQ(ierr);
  ierr = VecPointwiseMult(Xdot_c, rscale, Xdot_c);CHKERRQ(ierr);
  ierr = TSDiscGradRestoreX0AndXdot(ts, fine, &X0, &Xdot);CHKERRQ(ierr);
  ierr = TSDiscGradRestoreX0AndXdot(ts, coarse, &X0_c, &Xdot_c);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMSubDomainHook_TSDiscGrad(DM dm, DM subdm, void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMSubDomainRestrictHook_TSDiscGrad(DM dm, VecScatter gscat, VecScatter lscat, DM subdm, void *ctx)
{
  TS             ts = (TS) ctx;
  Vec            X0, Xdot, X0_sub, Xdot_sub;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSDiscGradGetX0AndXdot(ts, dm, &X0, &Xdot);CHKERRQ(ierr);
  ierr = TSDiscGradGetX0AndXdot(ts, subdm, &X0_sub, &Xdot_sub);CHKERRQ(ierr);

  ierr = VecScatterBegin(gscat, X0, X0_sub, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(gscat, X0, X0_sub, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);

  ierr = VecScatterBegin(gscat, Xdot, Xdot_sub, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(gscat, Xdot, Xdot_sub, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);

  ierr = TSDiscGradRestoreX0AndXdot(ts, dm, &X0, &Xdot);CHKERRQ(ierr);
  ierr = TSDiscGradRestoreX0AndXdot(ts, subdm, &X0_sub, &Xdot_sub);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSSetUp_DiscGrad(TS ts)
{
  TS_DiscGrad   *dg = (TS_DiscGrad *) ts->data;
  DM             dm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!dg->X)    {ierr = VecDuplicate(ts->vec_sol, &dg->X);CHKERRQ(ierr);}
  if (!dg->X0)   {ierr = VecDuplicate(ts->vec_sol, &dg->X0);CHKERRQ(ierr);}
  if (!dg->Xdot) {ierr = VecDuplicate(ts->vec_sol, &dg->Xdot);CHKERRQ(ierr);}

  ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
  ierr = DMCoarsenHookAdd(dm, DMCoarsenHook_TSDiscGrad, DMRestrictHook_TSDiscGrad, ts);CHKERRQ(ierr);
  ierr = DMSubDomainHookAdd(dm, DMSubDomainHook_TSDiscGrad, DMSubDomainRestrictHook_TSDiscGrad, ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSSetFromOptions_DiscGrad(PetscOptionItems *PetscOptionsObject, TS ts)
{
  TS_DiscGrad   *dg = (TS_DiscGrad *) ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject, "Discrete Gradients ODE solver options");CHKERRQ(ierr);
  {
    ierr = PetscOptionsBool("-ts_discgrad_gonzalez","Use Gonzalez term in discrete gradients formulation","TSDiscGradUseGonzalez",dg->gonzalez,&dg->gonzalez,NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSView_DiscGrad(TS ts,PetscViewer viewer)
{
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  Discrete Gradients\n");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSDiscGradIsGonzalez_DiscGrad(TS ts,PetscBool *gonzalez)
{
  TS_DiscGrad *dg = (TS_DiscGrad*)ts->data;

  PetscFunctionBegin;
  *gonzalez = dg->gonzalez;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSDiscGradUseGonzalez_DiscGrad(TS ts,PetscBool flg)
{
  TS_DiscGrad *dg = (TS_DiscGrad*)ts->data;

  PetscFunctionBegin;
  dg->gonzalez = flg;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSReset_DiscGrad(TS ts)
{
  TS_DiscGrad   *dg = (TS_DiscGrad *) ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDestroy(&dg->X);CHKERRQ(ierr);
  ierr = VecDestroy(&dg->X0);CHKERRQ(ierr);
  ierr = VecDestroy(&dg->Xdot);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSDestroy_DiscGrad(TS ts)
{
  DM             dm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSReset_DiscGrad(ts);CHKERRQ(ierr);
  ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
  if (dm) {
    ierr = DMCoarsenHookRemove(dm, DMCoarsenHook_TSDiscGrad, DMRestrictHook_TSDiscGrad, ts);CHKERRQ(ierr);
    ierr = DMSubDomainHookRemove(dm, DMSubDomainHook_TSDiscGrad, DMSubDomainRestrictHook_TSDiscGrad, ts);CHKERRQ(ierr);
  }
  ierr = PetscFree(ts->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSDiscGradGetFormulation_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSDiscGradSetFormulation_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSInterpolate_DiscGrad(TS ts, PetscReal t, Vec X)
{
  TS_DiscGrad   *dg = (TS_DiscGrad*)ts->data;
  PetscReal      dt = t - ts->ptime;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCopy(ts->vec_sol, dg->X);CHKERRQ(ierr);
  ierr = VecWAXPY(X, dt, dg->Xdot, dg->X);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSDiscGrad_SNESSolve(TS ts, Vec b, Vec x)
{
  SNES           snes;
  PetscInt       nits, lits;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetSNES(ts, &snes);CHKERRQ(ierr);
  ierr = SNESSolve(snes, b, x);CHKERRQ(ierr);
  ierr = SNESGetIterationNumber(snes, &nits);CHKERRQ(ierr);
  ierr = SNESGetLinearSolveIterations(snes, &lits);CHKERRQ(ierr);
  ts->snes_its += nits;
  ts->ksp_its  += lits;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSStep_DiscGrad(TS ts)
{
  TS_DiscGrad   *dg = (TS_DiscGrad *) ts->data;
  TSAdapt        adapt;
  TSStepStatus   status          = TS_STEP_INCOMPLETE;
  PetscInt       rejections      = 0;
  PetscBool      stageok, accept = PETSC_TRUE;
  PetscReal      next_time_step  = ts->time_step;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSGetAdapt(ts, &adapt);CHKERRQ(ierr);
  if (!ts->steprollback) {ierr = VecCopy(ts->vec_sol, dg->X0);CHKERRQ(ierr);}

  while (!ts->reason && status != TS_STEP_COMPLETE) {
    PetscReal shift = 1/(0.5*ts->time_step);

    dg->stage_time = ts->ptime + 0.5*ts->time_step;

    ierr = VecCopy(dg->X0, dg->X);CHKERRQ(ierr);
    ierr = TSPreStage(ts, dg->stage_time);CHKERRQ(ierr);
    ierr = TSDiscGrad_SNESSolve(ts, NULL, dg->X);CHKERRQ(ierr);
    ierr = TSPostStage(ts, dg->stage_time, 0, &dg->X);CHKERRQ(ierr);
    ierr = TSAdaptCheckStage(adapt, ts, dg->stage_time, dg->X, &stageok);CHKERRQ(ierr);
    if (!stageok) goto reject_step;

    status = TS_STEP_PENDING;
    ierr = VecAXPBYPCZ(dg->Xdot, -shift, shift, 0, dg->X0, dg->X);CHKERRQ(ierr);
    ierr = VecAXPY(ts->vec_sol, ts->time_step, dg->Xdot);CHKERRQ(ierr);
    ierr = TSAdaptChoose(adapt, ts, ts->time_step, NULL, &next_time_step, &accept);CHKERRQ(ierr);
    status = accept ? TS_STEP_COMPLETE : TS_STEP_INCOMPLETE;
    if (!accept) {
      ierr = VecCopy(dg->X0, ts->vec_sol);CHKERRQ(ierr);
      ts->time_step = next_time_step;
      goto reject_step;
    }
    ts->ptime    += ts->time_step;
    ts->time_step = next_time_step;
    break;

  reject_step:
    ts->reject++; accept = PETSC_FALSE;
    if (!ts->reason && ts->max_reject >= 0 && ++rejections > ts->max_reject) {
      ts->reason = TS_DIVERGED_STEP_REJECTED;
      ierr = PetscInfo(ts, "Step=%D, step rejections %D greater than current TS allowed, stopping solve\n", ts->steps, rejections);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSGetStages_DiscGrad(TS ts, PetscInt *ns, Vec **Y)
{
  TS_DiscGrad *dg = (TS_DiscGrad *) ts->data;

  PetscFunctionBegin;
  if (ns) *ns = 1;
  if (Y)  *Y  = &(dg->X);
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

  TS_DiscGrad   *dg    = (TS_DiscGrad *) ts->data;
  PetscReal      norm, shift = 1/(0.5*ts->time_step);
  PetscInt       n;
  Vec            X0, Xdot, Xp, Xdiff;
  Mat            S;
  PetscScalar    F=0, F0=0, Gp;
  Vec            G, SgF;
  DM             dm, dmsave;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESGetDM(snes, &dm);CHKERRQ(ierr);

  ierr = VecDuplicate(y, &Xp);CHKERRQ(ierr);
  ierr = VecDuplicate(y, &Xdiff);CHKERRQ(ierr);
  ierr = VecDuplicate(y, &SgF);CHKERRQ(ierr);
  ierr = VecDuplicate(y, &G);CHKERRQ(ierr);

  ierr = VecGetLocalSize(y, &n);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD, &S);CHKERRQ(ierr);
  ierr = MatSetSizes(S, PETSC_DECIDE, PETSC_DECIDE, n, n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(S);CHKERRQ(ierr);
  ierr = MatSetUp(S);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(S,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(S,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = TSDiscGradGetX0AndXdot(ts, dm, &X0, &Xdot);CHKERRQ(ierr);
  ierr = VecAXPBYPCZ(Xdot, -shift, shift, 0, X0, x);CHKERRQ(ierr); /* Xdot = shift (x - X0) */

  ierr = VecAXPBYPCZ(Xp, -1, 2, 0, X0, x);CHKERRQ(ierr); /* Xp = 2*x - X0 + (0)*Xmid */
  ierr = VecAXPBYPCZ(Xdiff, -1, 1, 0, X0, Xp);CHKERRQ(ierr); /* Xdiff = xp - X0 + (0)*Xdiff */

  if (dg->gonzalez) {
    ierr = (*dg->Sfunc)(ts, dg->stage_time, x ,   S,  dg->funcCtx);CHKERRQ(ierr);
    ierr = (*dg->Ffunc)(ts, dg->stage_time, Xp,  &F,  dg->funcCtx);CHKERRQ(ierr);
    ierr = (*dg->Ffunc)(ts, dg->stage_time, X0,  &F0, dg->funcCtx);CHKERRQ(ierr);
    ierr = (*dg->Gfunc)(ts, dg->stage_time, x ,   G,  dg->funcCtx);CHKERRQ(ierr);

    /* Adding Extra Gonzalez Term */
    ierr = VecDot(Xdiff, G, &Gp);CHKERRQ(ierr);
    ierr = VecNorm(Xdiff, NORM_2, &norm);CHKERRQ(ierr);
    if (norm < PETSC_SQRT_MACHINE_EPSILON) {
      Gp = 0;
    } else {
      /* Gp = (1/|xn+1 - xn|^2) * (F(xn+1) - F(xn) - Gp) */
      Gp = (F - F0 - Gp)/PetscSqr(norm);
    }
    ierr = VecAXPY(G, Gp, Xdiff);CHKERRQ(ierr);
    ierr = MatMult(S, G , SgF);CHKERRQ(ierr); /* S*gradF */

  } else {
    ierr = (*dg->Sfunc)(ts, dg->stage_time, x, S,  dg->funcCtx);CHKERRQ(ierr);
    ierr = (*dg->Gfunc)(ts, dg->stage_time, x, G,  dg->funcCtx);CHKERRQ(ierr);

    ierr = MatMult(S, G , SgF);CHKERRQ(ierr); /* Xdot = S*gradF */
  }
  /* DM monkey-business allows user code to call TSGetDM() inside of functions evaluated on levels of FAS */
  dmsave = ts->dm;
  ts->dm = dm;
  ierr = VecAXPBYPCZ(y, 1, -1, 0, Xdot, SgF);CHKERRQ(ierr);
  ts->dm = dmsave;
  ierr   = TSDiscGradRestoreX0AndXdot(ts, dm, &X0, &Xdot);CHKERRQ(ierr);

  ierr = VecDestroy(&Xp);CHKERRQ(ierr);
  ierr = VecDestroy(&Xdiff);CHKERRQ(ierr);
  ierr = VecDestroy(&SgF);CHKERRQ(ierr);
  ierr = VecDestroy(&G);CHKERRQ(ierr);
  ierr = MatDestroy(&S);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode SNESTSFormJacobian_DiscGrad(SNES snes, Vec x, Mat A, Mat B, TS ts)
{
  TS_DiscGrad   *dg    = (TS_DiscGrad *) ts->data;
  PetscReal      shift = 1/(0.5*ts->time_step);
  Vec            Xdot;
  DM             dm,dmsave;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESGetDM(snes, &dm);CHKERRQ(ierr);
  /* Xdot has already been computed in SNESTSFormFunction_DiscGrad (SNES guarantees this) */
  ierr = TSDiscGradGetX0AndXdot(ts, dm, NULL, &Xdot);CHKERRQ(ierr);

  dmsave = ts->dm;
  ts->dm = dm;
  ierr   = TSComputeIJacobian(ts, dg->stage_time, x, Xdot, shift, A, B, PETSC_FALSE);CHKERRQ(ierr);
  ts->dm = dmsave;
  ierr   = TSDiscGradRestoreX0AndXdot(ts, dm, NULL, &Xdot);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSDiscGradGetFormulation_DiscGrad(TS ts, PetscErrorCode (**Sfunc)(TS, PetscReal, Vec, Mat, void *), PetscErrorCode (**Ffunc)(TS, PetscReal, Vec, PetscScalar *, void *), PetscErrorCode (**Gfunc)(TS, PetscReal, Vec, Vec, void *), void *ctx)
{
  TS_DiscGrad *dg = (TS_DiscGrad *) ts->data;

  PetscFunctionBegin;
  *Sfunc = dg->Sfunc;
  *Ffunc = dg->Ffunc;
  *Gfunc = dg->Gfunc;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSDiscGradSetFormulation_DiscGrad(TS ts, PetscErrorCode (*Sfunc)(TS, PetscReal, Vec, Mat, void *), PetscErrorCode (*Ffunc)(TS, PetscReal, Vec, PetscScalar *, void *), PetscErrorCode (*Gfunc)(TS, PetscReal, Vec, Vec, void *), void *ctx)
{
  TS_DiscGrad *dg = (TS_DiscGrad *) ts->data;

  PetscFunctionBegin;
  dg->Sfunc = Sfunc;
  dg->Ffunc = Ffunc;
  dg->Gfunc = Gfunc;
  dg->funcCtx = ctx;
  PetscFunctionReturn(0);
}

/*MC
  TSDISCGRAD - ODE solver using the discrete gradients version of the implicit midpoint method

  Notes:
  This is the implicit midpoint rule, with an optional term that guarantees the discrete gradient property. This
  timestepper applies to systems of the form
$ u_t = S(u) grad F(u)
  where S(u) is a linear operator, and F is a functional of u.

  Level: intermediate

.seealso: TSCreate(), TSSetType(), TS, TSDISCGRAD, TSDiscGradSetFormulation()
M*/
PETSC_EXTERN PetscErrorCode TSCreate_DiscGrad(TS ts)
{
  TS_DiscGrad       *th;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscCitationsRegister(DGCitation, &DGCite);CHKERRQ(ierr);
  ts->ops->reset           = TSReset_DiscGrad;
  ts->ops->destroy         = TSDestroy_DiscGrad;
  ts->ops->view            = TSView_DiscGrad;
  ts->ops->setfromoptions  = TSSetFromOptions_DiscGrad;
  ts->ops->setup           = TSSetUp_DiscGrad;
  ts->ops->step            = TSStep_DiscGrad;
  ts->ops->interpolate     = TSInterpolate_DiscGrad;
  ts->ops->getstages       = TSGetStages_DiscGrad;
  ts->ops->snesfunction    = SNESTSFormFunction_DiscGrad;
  ts->ops->snesjacobian    = SNESTSFormJacobian_DiscGrad;
  ts->default_adapt_type   = TSADAPTNONE;

  ts->usessnes = PETSC_TRUE;

  ierr = PetscNewLog(ts,&th);CHKERRQ(ierr);
  ts->data = (void*)th;

  th->gonzalez = PETSC_FALSE;

  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSDiscGradGetFormulation_C",TSDiscGradGetFormulation_DiscGrad);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSDiscGradSetFormulation_C",TSDiscGradSetFormulation_DiscGrad);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSDiscGradIsGonzalez_C",TSDiscGradIsGonzalez_DiscGrad);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSDiscGradUseGonzalez_C",TSDiscGradUseGonzalez_DiscGrad);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  TSDiscGradGetFormulation - Get the construction method for S, F, and grad F from the formulation u_t = S grad F

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

.seealso: TSDiscGradSetFormulation()
@*/
PetscErrorCode TSDiscGradGetFormulation(TS ts, PetscErrorCode (**Sfunc)(TS, PetscReal, Vec, Mat, void *), PetscErrorCode (**Ffunc)(TS, PetscReal, Vec, PetscScalar *, void *), PetscErrorCode (**Gfunc)(TS, PetscReal, Vec, Vec, void *), void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidPointer(Sfunc, 2);
  PetscValidPointer(Ffunc, 3);
  PetscValidPointer(Gfunc, 4);
  ierr = PetscUseMethod(ts,"TSDiscGradGetFormulation_C",(TS,PetscErrorCode(**Sfunc)(TS,PetscReal,Vec,Mat,void*),PetscErrorCode(**Ffunc)(TS,PetscReal,Vec,PetscScalar*,void*),PetscErrorCode(**Gfunc)(TS,PetscReal,Vec,Vec,void*), void*),(ts,Sfunc,Ffunc,Gfunc,ctx));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  TSDiscGradSetFormulation - Set the construction method for S, F, and grad F from the formulation u_t = S(u) grad F(u)

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

.seealso: TSDiscGradGetFormulation()
@*/
PetscErrorCode TSDiscGradSetFormulation(TS ts, PetscErrorCode (*Sfunc)(TS, PetscReal, Vec, Mat, void *), PetscErrorCode (*Ffunc)(TS, PetscReal, Vec , PetscScalar *, void *), PetscErrorCode (*Gfunc)(TS, PetscReal, Vec, Vec, void *), void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts, TS_CLASSID, 1);
  PetscValidFunction(Sfunc, 2);
  PetscValidFunction(Ffunc, 3);
  PetscValidFunction(Gfunc, 4);
  ierr = PetscTryMethod(ts,"TSDiscGradSetFormulation_C",(TS,PetscErrorCode(*Sfunc)(TS,PetscReal,Vec,Mat,void*),PetscErrorCode(*Ffunc)(TS,PetscReal,Vec,PetscScalar*,void*),PetscErrorCode(*Gfunc)(TS,PetscReal,Vec,Vec,void*), void*),(ts,Sfunc,Ffunc,Gfunc,ctx));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  TSDiscGradIsGonzalez - Checks flag for whether to use additional conservative terms in discrete gradient formulation.

  Not Collective

  Input Parameter:
.  ts - timestepping context

  Output Parameter:
.  gonzalez - PETSC_TRUE when using the Gonzalez term

  Level: Advanced

.seealso: TSDiscGradUseGonzalez(), TSDISCGRAD
@*/
PetscErrorCode TSDiscGradIsGonzalez(TS ts,PetscBool *gonzalez)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidPointer(gonzalez,2);
  ierr = PetscUseMethod(ts,"TSDiscGradIsGonzalez_C",(TS,PetscBool*),(ts,gonzalez));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  TSDiscGradUseGonzalez - Sets discrete gradient formulation with or without additional conservative terms.  Without flag, the discrete gradients timestepper is just backwards euler

  Not Collective

  Input Parameters:
+ ts - timestepping context
- flg - PETSC_TRUE to use the Gonzalez term

  Options Database:
. -ts_discgrad_gonzalez <flg>

  Level: Intermediate

.seealso: TSDISCGRAD
@*/
PetscErrorCode TSDiscGradUseGonzalez(TS ts,PetscBool flg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = PetscTryMethod(ts,"TSDiscGradUseGonzalez_C",(TS,PetscBool),(ts,flg));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
