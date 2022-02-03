/*
  Code for timestepping with BDF methods
*/
#include <petsc/private/tsimpl.h>  /*I "petscts.h" I*/
#include <petscdm.h>

static PetscBool  cited = PETSC_FALSE;
static const char citation[] =
  "@book{Brenan1995,\n"
  "  title     = {Numerical Solution of Initial-Value Problems in Differential-Algebraic Equations},\n"
  "  author    = {Brenan, K. and Campbell, S. and Petzold, L.},\n"
  "  publisher = {Society for Industrial and Applied Mathematics},\n"
  "  year      = {1995},\n"
  "  doi       = {10.1137/1.9781611971224},\n}\n";

typedef struct {
  PetscInt  k,n;
  PetscReal time[6+2];
  Vec       work[6+2];
  Vec       tvwork[6+2];
  PetscReal shift;
  Vec       vec_dot;            /* Xdot when !transientvar, else Cdot where C(X) is the transient variable. */
  Vec       vec_wrk;
  Vec       vec_lte;

  PetscBool    transientvar;
  PetscInt     order;
  TSStepStatus status;
} TS_BDF;

/* Compute Lagrange polynomials on T[:n] evaluated at t.
 * If one has data (T[i], Y[i]), then the interpolation/extrapolation is f(t) = \sum_i L[i]*Y[i].
 */
static inline void LagrangeBasisVals(PetscInt n,PetscReal t,const PetscReal T[],PetscScalar L[])
{
  PetscInt k,j;
  for (k=0; k<n; k++)
    for (L[k]=1, j=0; j<n; j++)
      if (j != k)
        L[k] *= (t - T[j])/(T[k] - T[j]);
}

static inline void LagrangeBasisDers(PetscInt n,PetscReal t,const PetscReal T[],PetscScalar dL[])
{
  PetscInt  k,j,i;
  for (k=0; k<n; k++)
    for (dL[k]=0, j=0; j<n; j++)
      if (j != k) {
        PetscReal L = 1/(T[k] - T[j]);
        for (i=0; i<n; i++)
          if (i != j && i != k)
            L *= (t - T[i])/(T[k] - T[i]);
        dL[k] += L;
      }
}

static PetscErrorCode TSBDF_GetVecs(TS ts,DM dm,Vec *Xdot,Vec *Ydot)
{
  TS_BDF         *bdf = (TS_BDF*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (dm && dm != ts->dm) {
    ierr = DMGetNamedGlobalVector(dm,"TSBDF_Vec_Xdot",Xdot);CHKERRQ(ierr);
    ierr = DMGetNamedGlobalVector(dm,"TSBDF_Vec_Ydot",Ydot);CHKERRQ(ierr);
  } else {
    *Xdot = bdf->vec_dot;
    *Ydot = bdf->vec_wrk;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSBDF_RestoreVecs(TS ts,DM dm,Vec *Xdot,Vec *Ydot)
{
  TS_BDF         *bdf = (TS_BDF*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (dm && dm != ts->dm) {
    ierr = DMRestoreNamedGlobalVector(dm,"TSBDF_Vec_Xdot",Xdot);CHKERRQ(ierr);
    ierr = DMRestoreNamedGlobalVector(dm,"TSBDF_Vec_Ydot",Ydot);CHKERRQ(ierr);
  } else {
    PetscAssertFalse(*Xdot != bdf->vec_dot,PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_INCOMP,"Vec does not match the cache");
    PetscAssertFalse(*Ydot != bdf->vec_wrk,PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_INCOMP,"Vec does not match the cache");
    *Xdot = NULL;
    *Ydot = NULL;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCoarsenHook_TSBDF(DM fine,DM coarse,void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMRestrictHook_TSBDF(DM fine,Mat restrct,Vec rscale,Mat inject,DM coarse,void *ctx)
{
  TS             ts = (TS)ctx;
  Vec            Ydot,Ydot_c;
  Vec            Xdot,Xdot_c;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSBDF_GetVecs(ts,fine,&Xdot,&Ydot);CHKERRQ(ierr);
  ierr = TSBDF_GetVecs(ts,coarse,&Xdot_c,&Ydot_c);CHKERRQ(ierr);

  ierr = MatRestrict(restrct,Ydot,Ydot_c);CHKERRQ(ierr);
  ierr = VecPointwiseMult(Ydot_c,rscale,Ydot_c);CHKERRQ(ierr);

  ierr = TSBDF_RestoreVecs(ts,fine,&Xdot,&Ydot);CHKERRQ(ierr);
  ierr = TSBDF_RestoreVecs(ts,coarse,&Xdot_c,&Ydot_c);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSBDF_Advance(TS ts,PetscReal t,Vec X)
{
  TS_BDF         *bdf = (TS_BDF*)ts->data;
  PetscInt       i,n = (PetscInt)(sizeof(bdf->work)/sizeof(Vec));
  Vec            tail = bdf->work[n-1],tvtail = bdf->tvwork[n-1];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (i=n-1; i>=2; i--) {
    bdf->time[i] = bdf->time[i-1];
    bdf->work[i] = bdf->work[i-1];
    bdf->tvwork[i] = bdf->tvwork[i-1];
  }
  bdf->n       = PetscMin(bdf->n+1,n-1);
  bdf->time[1] = t;
  bdf->work[1] = tail;
  bdf->tvwork[1] = tvtail;
  ierr = VecCopy(X,tail);CHKERRQ(ierr);
  ierr = TSComputeTransientVariable(ts,tail,tvtail);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSBDF_VecLTE(TS ts,PetscInt order,Vec lte)
{
  TS_BDF         *bdf = (TS_BDF*)ts->data;
  PetscInt       i,n = order+1;
  PetscReal      *time = bdf->time;
  Vec            *vecs = bdf->work;
  PetscScalar    a[8],b[8],alpha[8];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  LagrangeBasisDers(n+0,time[0],time,a); a[n] =0;
  LagrangeBasisDers(n+1,time[0],time,b);
  for (i=0; i<n+1; i++) alpha[i] = (a[i]-b[i])/a[0];
  ierr = VecZeroEntries(lte);CHKERRQ(ierr);
  ierr = VecMAXPY(lte,n+1,alpha,vecs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSBDF_Extrapolate(TS ts,PetscInt order,PetscReal t,Vec X)
{
  TS_BDF         *bdf = (TS_BDF*)ts->data;
  PetscInt       n = order+1;
  PetscReal      *time = bdf->time+1;
  Vec            *vecs = bdf->work+1;
  PetscScalar    alpha[7];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  n = PetscMin(n,bdf->n);
  LagrangeBasisVals(n,t,time,alpha);
  ierr = VecZeroEntries(X);CHKERRQ(ierr);
  ierr = VecMAXPY(X,n,alpha,vecs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSBDF_Interpolate(TS ts,PetscInt order,PetscReal t,Vec X)
{
  TS_BDF         *bdf = (TS_BDF*)ts->data;
  PetscInt       n = order+1;
  PetscReal      *time = bdf->time;
  Vec            *vecs = bdf->work;
  PetscScalar    alpha[7];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  LagrangeBasisVals(n,t,time,alpha);
  ierr = VecZeroEntries(X);CHKERRQ(ierr);
  ierr = VecMAXPY(X,n,alpha,vecs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Compute the affine term V0 such that Xdot = shift*X + V0.
 *
 * When using transient variables, we're computing Cdot = shift*C(X) + V0, and thus choose a linear combination of tvwork.
 */
static PetscErrorCode TSBDF_PreSolve(TS ts)
{
  TS_BDF         *bdf = (TS_BDF*)ts->data;
  PetscInt       i,n = PetscMax(bdf->k,1) + 1;
  Vec            V,V0;
  Vec            vecs[7];
  PetscScalar    alpha[7];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSBDF_GetVecs(ts,NULL,&V,&V0);CHKERRQ(ierr);
  LagrangeBasisDers(n,bdf->time[0],bdf->time,alpha);
  for (i=1; i<n; i++) {
    vecs[i] = bdf->transientvar ? bdf->tvwork[i] : bdf->work[i];
  }
  ierr = VecZeroEntries(V0);CHKERRQ(ierr);
  ierr = VecMAXPY(V0,n-1,alpha+1,vecs+1);CHKERRQ(ierr);
  bdf->shift = PetscRealPart(alpha[0]);
  ierr = TSBDF_RestoreVecs(ts,NULL,&V,&V0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSBDF_SNESSolve(TS ts,Vec b,Vec x)
{
  PetscInt       nits,lits;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSBDF_PreSolve(ts);CHKERRQ(ierr);
  ierr = SNESSolve(ts->snes,b,x);CHKERRQ(ierr);
  ierr = SNESGetIterationNumber(ts->snes,&nits);CHKERRQ(ierr);
  ierr = SNESGetLinearSolveIterations(ts->snes,&lits);CHKERRQ(ierr);
  ts->snes_its += nits; ts->ksp_its += lits;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSBDF_Restart(TS ts,PetscBool *accept)
{
  TS_BDF         *bdf = (TS_BDF*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  bdf->k = 1; bdf->n = 0;
  ierr = TSBDF_Advance(ts,ts->ptime,ts->vec_sol);CHKERRQ(ierr);

  bdf->time[0] = ts->ptime + ts->time_step/2;
  ierr = VecCopy(bdf->work[1],bdf->work[0]);CHKERRQ(ierr);
  ierr = TSPreStage(ts,bdf->time[0]);CHKERRQ(ierr);
  ierr = TSBDF_SNESSolve(ts,NULL,bdf->work[0]);CHKERRQ(ierr);
  ierr = TSPostStage(ts,bdf->time[0],0,&bdf->work[0]);CHKERRQ(ierr);
  ierr = TSAdaptCheckStage(ts->adapt,ts,bdf->time[0],bdf->work[0],accept);CHKERRQ(ierr);
  if (!*accept) PetscFunctionReturn(0);

  bdf->k = PetscMin(2,bdf->order); bdf->n++;
  ierr = VecCopy(bdf->work[0],bdf->work[2]);CHKERRQ(ierr);
  bdf->time[2] = bdf->time[0];
  ierr = TSComputeTransientVariable(ts,bdf->work[2],bdf->tvwork[2]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static const char *const BDF_SchemeName[] = {"", "1", "2", "3", "4", "5", "6"};

static PetscErrorCode TSStep_BDF(TS ts)
{
  TS_BDF         *bdf = (TS_BDF*)ts->data;
  PetscInt       rejections = 0;
  PetscBool      stageok,accept = PETSC_TRUE;
  PetscReal      next_time_step = ts->time_step;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscCitationsRegister(citation,&cited);CHKERRQ(ierr);

  if (!ts->steprollback && !ts->steprestart) {
    bdf->k = PetscMin(bdf->k+1,bdf->order);
    ierr = TSBDF_Advance(ts,ts->ptime,ts->vec_sol);CHKERRQ(ierr);
  }

  bdf->status = TS_STEP_INCOMPLETE;
  while (!ts->reason && bdf->status != TS_STEP_COMPLETE) {

    if (ts->steprestart) {
      ierr = TSBDF_Restart(ts,&stageok);CHKERRQ(ierr);
      if (!stageok) goto reject_step;
    }

    bdf->time[0] = ts->ptime + ts->time_step;
    ierr = TSBDF_Extrapolate(ts,bdf->k-(accept?0:1),bdf->time[0],bdf->work[0]);CHKERRQ(ierr);
    ierr = TSPreStage(ts,bdf->time[0]);CHKERRQ(ierr);
    ierr = TSBDF_SNESSolve(ts,NULL,bdf->work[0]);CHKERRQ(ierr);
    ierr = TSPostStage(ts,bdf->time[0],0,&bdf->work[0]);CHKERRQ(ierr);
    ierr = TSAdaptCheckStage(ts->adapt,ts,bdf->time[0],bdf->work[0],&stageok);CHKERRQ(ierr);
    if (!stageok) goto reject_step;

    bdf->status = TS_STEP_PENDING;
    ierr = TSAdaptCandidatesClear(ts->adapt);CHKERRQ(ierr);
    ierr = TSAdaptCandidateAdd(ts->adapt,BDF_SchemeName[bdf->k],bdf->k,1,1.0,1.0,PETSC_TRUE);CHKERRQ(ierr);
    ierr = TSAdaptChoose(ts->adapt,ts,ts->time_step,NULL,&next_time_step,&accept);CHKERRQ(ierr);
    bdf->status = accept ? TS_STEP_COMPLETE : TS_STEP_INCOMPLETE;
    if (!accept) { ts->time_step = next_time_step; goto reject_step; }

    ierr = VecCopy(bdf->work[0],ts->vec_sol);CHKERRQ(ierr);
    ts->ptime += ts->time_step;
    ts->time_step = next_time_step;
    break;

  reject_step:
    ts->reject++; accept = PETSC_FALSE;
    if (!ts->reason && ++rejections > ts->max_reject && ts->max_reject >= 0) {
      ierr = PetscInfo(ts,"Step=%D, step rejections %D greater than current TS allowed, stopping solve\n",ts->steps,rejections);CHKERRQ(ierr);
      ts->reason = TS_DIVERGED_STEP_REJECTED;
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSInterpolate_BDF(TS ts,PetscReal t,Vec X)
{
  TS_BDF         *bdf = (TS_BDF*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSBDF_Interpolate(ts,bdf->k,t,X);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSEvaluateWLTE_BDF(TS ts,NormType wnormtype,PetscInt *order,PetscReal *wlte)
{
  TS_BDF         *bdf = (TS_BDF*)ts->data;
  PetscInt       k = bdf->k;
  PetscReal      wltea,wlter;
  Vec            X = bdf->work[0], Y = bdf->vec_lte;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  k = PetscMin(k,bdf->n-1);
  ierr = TSBDF_VecLTE(ts,k,Y);CHKERRQ(ierr);
  ierr = VecAXPY(Y,1,X);CHKERRQ(ierr);
  ierr = TSErrorWeightedNorm(ts,X,Y,wnormtype,wlte,&wltea,&wlter);CHKERRQ(ierr);
  if (order) *order = k + 1;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSRollBack_BDF(TS ts)
{
  TS_BDF         *bdf = (TS_BDF*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCopy(bdf->work[1],ts->vec_sol);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESTSFormFunction_BDF(SNES snes,Vec X,Vec F,TS ts)
{
  TS_BDF         *bdf = (TS_BDF*)ts->data;
  DM             dm, dmsave = ts->dm;
  PetscReal      t = bdf->time[0];
  PetscReal      shift = bdf->shift;
  Vec            V,V0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = TSBDF_GetVecs(ts,dm,&V,&V0);CHKERRQ(ierr);
  if (bdf->transientvar) {      /* shift*C(X) + V0 */
    ierr = TSComputeTransientVariable(ts,X,V);CHKERRQ(ierr);
    ierr = VecAYPX(V,shift,V0);CHKERRQ(ierr);
  } else {                      /* shift*X + V0 */
    ierr = VecWAXPY(V,shift,X,V0);CHKERRQ(ierr);
  }

  /* F = Function(t,X,V) */
  ts->dm = dm;
  ierr = TSComputeIFunction(ts,t,X,V,F,PETSC_FALSE);CHKERRQ(ierr);
  ts->dm = dmsave;

  ierr = TSBDF_RestoreVecs(ts,dm,&V,&V0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESTSFormJacobian_BDF(SNES snes,Vec X,Mat J,Mat P,TS ts)
{
  TS_BDF         *bdf = (TS_BDF*)ts->data;
  DM             dm, dmsave = ts->dm;
  PetscReal      t = bdf->time[0];
  PetscReal      shift = bdf->shift;
  Vec            V,V0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SNESGetDM(snes,&dm);CHKERRQ(ierr);
  ierr = TSBDF_GetVecs(ts,dm,&V,&V0);CHKERRQ(ierr);

  /* J,P = Jacobian(t,X,V) */
  ts->dm = dm;
  ierr = TSComputeIJacobian(ts,t,X,V,shift,J,P,PETSC_FALSE);CHKERRQ(ierr);
  ts->dm = dmsave;

  ierr = TSBDF_RestoreVecs(ts,dm,&V,&V0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSReset_BDF(TS ts)
{
  TS_BDF         *bdf = (TS_BDF*)ts->data;
  size_t         i,n = sizeof(bdf->work)/sizeof(Vec);
  PetscErrorCode ierr;

  PetscFunctionBegin;
  bdf->k = bdf->n = 0;
  for (i=0; i<n; i++) {
    ierr = VecDestroy(&bdf->work[i]);CHKERRQ(ierr);
    ierr = VecDestroy(&bdf->tvwork[i]);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&bdf->vec_dot);CHKERRQ(ierr);
  ierr = VecDestroy(&bdf->vec_wrk);CHKERRQ(ierr);
  ierr = VecDestroy(&bdf->vec_lte);CHKERRQ(ierr);
  if (ts->dm) {ierr = DMCoarsenHookRemove(ts->dm,DMCoarsenHook_TSBDF,DMRestrictHook_TSBDF,ts);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode TSDestroy_BDF(TS ts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSReset_BDF(ts);CHKERRQ(ierr);
  ierr = PetscFree(ts->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSBDFSetOrder_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSBDFGetOrder_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSSetUp_BDF(TS ts)
{
  TS_BDF         *bdf = (TS_BDF*)ts->data;
  size_t         i,n = sizeof(bdf->work)/sizeof(Vec);
  PetscReal      low,high,two = 2;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSHasTransientVariable(ts,&bdf->transientvar);CHKERRQ(ierr);
  bdf->k = bdf->n = 0;
  for (i=0; i<n; i++) {
    ierr = VecDuplicate(ts->vec_sol,&bdf->work[i]);CHKERRQ(ierr);
    if (i && bdf->transientvar) {
      ierr = VecDuplicate(ts->vec_sol,&bdf->tvwork[i]);CHKERRQ(ierr);
    }
  }
  ierr = VecDuplicate(ts->vec_sol,&bdf->vec_dot);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&bdf->vec_wrk);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&bdf->vec_lte);CHKERRQ(ierr);
  ierr = TSGetDM(ts,&ts->dm);CHKERRQ(ierr);
  ierr = DMCoarsenHookAdd(ts->dm,DMCoarsenHook_TSBDF,DMRestrictHook_TSBDF,ts);CHKERRQ(ierr);

  ierr = TSGetAdapt(ts,&ts->adapt);CHKERRQ(ierr);
  ierr = TSAdaptCandidatesClear(ts->adapt);CHKERRQ(ierr);
  ierr = TSAdaptGetClip(ts->adapt,&low,&high);CHKERRQ(ierr);
  ierr = TSAdaptSetClip(ts->adapt,low,PetscMin(high,two));CHKERRQ(ierr);

  ierr = TSGetSNES(ts,&ts->snes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSSetFromOptions_BDF(PetscOptionItems *PetscOptionsObject,TS ts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"BDF ODE solver options");CHKERRQ(ierr);
  {
    PetscBool flg;
    PetscInt  order;
    ierr = TSBDFGetOrder(ts,&order);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-ts_bdf_order","Order of the BDF method","TSBDFSetOrder",order,&order,&flg);CHKERRQ(ierr);
    if (flg) {ierr = TSBDFSetOrder(ts,order);CHKERRQ(ierr);}
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSView_BDF(TS ts,PetscViewer viewer)
{
  TS_BDF         *bdf = (TS_BDF*)ts->data;
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  Order=%D\n",bdf->order);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------ */

static PetscErrorCode TSBDFSetOrder_BDF(TS ts,PetscInt order)
{
  TS_BDF *bdf = (TS_BDF*)ts->data;

  PetscFunctionBegin;
  if (order == bdf->order) PetscFunctionReturn(0);
  PetscAssertFalse(order < 1 || order > 6,PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_OUTOFRANGE,"BDF Order %D not implemented",order);
  bdf->order = order;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSBDFGetOrder_BDF(TS ts,PetscInt *order)
{
  TS_BDF *bdf = (TS_BDF*)ts->data;

  PetscFunctionBegin;
  *order = bdf->order;
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------ */

/*MC
      TSBDF - DAE solver using BDF methods

  Level: beginner

.seealso:  TS, TSCreate(), TSSetType()
M*/
PETSC_EXTERN PetscErrorCode TSCreate_BDF(TS ts)
{
  TS_BDF         *bdf;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ts->ops->reset          = TSReset_BDF;
  ts->ops->destroy        = TSDestroy_BDF;
  ts->ops->view           = TSView_BDF;
  ts->ops->setup          = TSSetUp_BDF;
  ts->ops->setfromoptions = TSSetFromOptions_BDF;
  ts->ops->step           = TSStep_BDF;
  ts->ops->evaluatewlte   = TSEvaluateWLTE_BDF;
  ts->ops->rollback       = TSRollBack_BDF;
  ts->ops->interpolate    = TSInterpolate_BDF;
  ts->ops->snesfunction   = SNESTSFormFunction_BDF;
  ts->ops->snesjacobian   = SNESTSFormJacobian_BDF;
  ts->default_adapt_type  = TSADAPTBASIC;

  ts->usessnes = PETSC_TRUE;

  ierr = PetscNewLog(ts,&bdf);CHKERRQ(ierr);
  ts->data = (void*)bdf;

  bdf->status = TS_STEP_COMPLETE;

  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSBDFSetOrder_C",TSBDFSetOrder_BDF);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSBDFGetOrder_C",TSBDFGetOrder_BDF);CHKERRQ(ierr);
  ierr = TSBDFSetOrder(ts,2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------ */

/*@
  TSBDFSetOrder - Set the order of the BDF method

  Logically Collective on TS

  Input Parameters:
+  ts - timestepping context
-  order - order of the method

  Options Database:
.  -ts_bdf_order <order>

  Level: intermediate

@*/
PetscErrorCode TSBDFSetOrder(TS ts,PetscInt order)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidLogicalCollectiveInt(ts,order,2);
  ierr = PetscTryMethod(ts,"TSBDFSetOrder_C",(TS,PetscInt),(ts,order));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  TSBDFGetOrder - Get the order of the BDF method

  Not Collective

  Input Parameter:
.  ts - timestepping context

  Output Parameter:
.  order - order of the method

  Level: intermediate

@*/
PetscErrorCode TSBDFGetOrder(TS ts,PetscInt *order)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidIntPointer(order,2);
  ierr = PetscUseMethod(ts,"TSBDFGetOrder_C",(TS,PetscInt*),(ts,order));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
