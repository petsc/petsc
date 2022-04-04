#include <petsc/private/tsimpl.h> /*I "petscts.h" I*/
#include <petscdm.h>

static const char *citation[] = {
  "@article{Soderlind2003,\n"
  " author = {S\"{o}derlind, Gustaf},\n"
  " title = {Digital Filters in Adaptive Time-stepping},\n"
  " journal = {ACM Transactions on Mathematical Software},\n"
  " volume = {29},\n"
  " number = {1},\n"
  " pages = {1--26},\n"
  " year = {2003},\n"
  " issn = {0098-3500},\n"
  " doi = {http://dx.doi.org/10.1145/641876.641877},\n"
  "}\n",
  "@article{Soderlind2006,\n"
  " author = {Gustaf S\"{o}derlind and Lina Wang},\n"
  " title = {Adaptive time-stepping and computational stability},\n"
  " journal = {Journal of Computational and Applied Mathematics},\n"
  " volume = {185},\n"
  " number = {2},\n"
  " pages = {225--243},\n"
  " year = {2006},\n"
  " issn = {0377-0427},\n"
  " doi = {http://dx.doi.org/10.1016/j.cam.2005.03.008},\n"
  "}\n",
};
static PetscBool cited[] = {PETSC_FALSE,PETSC_FALSE};

typedef struct {
  PetscReal kBeta[3];  /* filter parameters */
  PetscReal Alpha[2];  /* filter parameters */
  PetscReal cerror[3]; /* control error (controller input) history */
  PetscReal hratio[3]; /* stepsize ratio (controller output) history */
  PetscBool rollback;
} TSAdapt_DSP;

static PetscReal Limiter(PetscReal value,PetscReal kappa)
{
  return 1 + kappa*PetscAtanReal((value - 1)/kappa);
}

static PetscErrorCode TSAdaptRestart_DSP(TSAdapt adapt)
{
  TSAdapt_DSP *dsp = (TSAdapt_DSP*)adapt->data;
  PetscFunctionBegin;
  dsp->cerror[0] = dsp->hratio[0] = 1.0;
  dsp->cerror[1] = dsp->hratio[1] = 1.0;
  dsp->cerror[2] = dsp->hratio[2] = 1.0;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSAdaptRollBack_DSP(TSAdapt adapt)
{
  TSAdapt_DSP *dsp = (TSAdapt_DSP*)adapt->data;
  PetscFunctionBegin;
  dsp->cerror[0] = dsp->cerror[1];
  dsp->cerror[1] = dsp->cerror[2];
  dsp->cerror[2] = 1.0;
  dsp->hratio[0] = dsp->hratio[1];
  dsp->hratio[1] = dsp->hratio[2];
  dsp->hratio[2] = 1.0;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSAdaptChoose_DSP(TSAdapt adapt,TS ts,PetscReal h,PetscInt *next_sc,PetscReal *next_h,PetscBool *accept,PetscReal *wlte,PetscReal *wltea,PetscReal *wlter)
{
  TSAdapt_DSP    *dsp = (TSAdapt_DSP*)adapt->data;
  PetscInt       order = PETSC_DECIDE;
  PetscReal      enorm = -1;
  PetscReal      enorma,enormr;
  PetscReal      safety = adapt->safety * (PetscReal)0.9;
  PetscReal      hnew,hfac = PETSC_INFINITY;
  PetscReal      hmin = adapt->dt_min*(1 + PETSC_SQRT_MACHINE_EPSILON);

  PetscFunctionBegin;
  *next_sc = 0;   /* Reuse the same order scheme */
  *wltea   = -1;  /* Weighted absolute local truncation error is not used */
  *wlter   = -1;  /* Weighted relative local truncation error is not used */

  if (ts->ops->evaluatewlte) {
    PetscCall(TSEvaluateWLTE(ts,adapt->wnormtype,&order,&enorm));
    PetscCheckFalse(enorm >= 0 && order < 1,PetscObjectComm((PetscObject)adapt),PETSC_ERR_ARG_OUTOFRANGE,"Computed error order %D must be positive",order);
  } else if (ts->ops->evaluatestep) {
    DM  dm;
    Vec Y;

    PetscCheck(adapt->candidates.n >= 1,PetscObjectComm((PetscObject)adapt),PETSC_ERR_ARG_WRONGSTATE,"No candidate has been registered");
    PetscCheck(adapt->candidates.inuse_set,PetscObjectComm((PetscObject)adapt),PETSC_ERR_ARG_WRONGSTATE,"The current in-use scheme is not among the %D candidates",adapt->candidates.n);
    order = adapt->candidates.order[0];
    PetscCall(TSGetDM(ts,&dm));
    PetscCall(DMGetGlobalVector(dm,&Y));
    PetscCall(TSEvaluateStep(ts,order-1,Y,NULL));
    PetscCall(TSErrorWeightedNorm(ts,ts->vec_sol,Y,adapt->wnormtype,&enorm,&enorma,&enormr));
    PetscCall(DMRestoreGlobalVector(dm,&Y));
  }
  if (enorm < 0) {
    PetscCall(TSAdaptRestart_DSP(adapt));
    *accept = PETSC_TRUE;  /* Accept the step */
    *next_h = h;           /* Reuse the old step size */
    *wlte   = -1;          /* Weighted local truncation error was not evaluated */
    PetscFunctionReturn(0);
  }

  PetscCall(PetscCitationsRegister(citation[0],&cited[0]));
  PetscCall(PetscCitationsRegister(citation[1],&cited[1]));

  /* Update history after rollback */
  if (!ts->steprollback)
    dsp->rollback = PETSC_FALSE;
  else if (!dsp->rollback) {
    dsp->rollback = PETSC_TRUE;
    PetscCall(TSAdaptRollBack_DSP(adapt));
  }
  /* Reset history after restart */
  if (ts->steprestart) {
    PetscCall(TSAdaptRestart_DSP(adapt));
  }

  {
    PetscReal k = (PetscReal)order;
    PetscReal b1 = dsp->kBeta[0];
    PetscReal b2 = dsp->kBeta[1];
    PetscReal b3 = dsp->kBeta[2];
    PetscReal a2 = dsp->Alpha[0];
    PetscReal a3 = dsp->Alpha[1];

    PetscReal ctr0;
    PetscReal ctr1 = dsp->cerror[0];
    PetscReal ctr2 = dsp->cerror[1];
    PetscReal rho0;
    PetscReal rho1 = dsp->hratio[0];
    PetscReal rho2 = dsp->hratio[1];

    /* Compute the step size ratio */
    enorm = PetscMax(enorm,PETSC_SMALL);
    ctr0  = PetscPowReal(1/enorm,1/k);
    rho0  = PetscPowReal(ctr0,b1);
    rho0 *= PetscPowReal(ctr1,b2);
    rho0 *= PetscPowReal(ctr2,b3);
    rho0 *= PetscPowReal(rho1,-a2);
    rho0 *= PetscPowReal(rho2,-a3);
    rho0  = Limiter(rho0,1);

    /* Determine whether the step is accepted or rejected */
    if (rho0 >= safety)
      *accept = PETSC_TRUE;
    else if (adapt->always_accept)
      *accept = PETSC_TRUE;
    else if (h < hmin)
      *accept = PETSC_TRUE;
    else
      *accept = PETSC_FALSE;

    /* Update history after accept */
    if (*accept) {
      dsp->cerror[2] = dsp->cerror[1];
      dsp->cerror[1] = dsp->cerror[0];
      dsp->cerror[0] = ctr0;
      dsp->hratio[2] = dsp->hratio[1];
      dsp->hratio[1] = dsp->hratio[0];
      dsp->hratio[0] = rho0;
      dsp->rollback  = PETSC_FALSE;
    }

    hfac = rho0;
  }

  hnew    = h * PetscClipInterval(hfac,adapt->clip[0],adapt->clip[1]);
  *next_h = PetscClipInterval(hnew,adapt->dt_min,adapt->dt_max);
  *wlte   = enorm;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSAdaptDestroy_DSP(TSAdapt adapt)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectComposeFunction((PetscObject)adapt,"TSAdaptDSPSetFilter_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)adapt,"TSAdaptDSPSetPID_C",NULL));
  PetscCall(PetscFree(adapt->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode TSAdaptView_DSP(TSAdapt adapt,PetscViewer viewer)
{
  TSAdapt_DSP    *dsp = (TSAdapt_DSP*)adapt->data;
  PetscBool      iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    double a2 = (double)dsp->Alpha[0], a3 = (double)dsp->Alpha[1];
    double b1 = (double)dsp->kBeta[0], b2 = (double)dsp->kBeta[1], b3 = (double)dsp->kBeta[2];
    PetscCall(PetscViewerASCIIPrintf(viewer,"filter parameters kBeta=[%g,%g,%g] Alpha=[%g,%g]\n",b1,b2,b3,a2,a3));
  }
  PetscFunctionReturn(0);
}

struct FilterTab {
  const char *name;
  PetscReal scale;
  PetscReal kBeta[3];
  PetscReal Alpha[2];
};

static struct FilterTab filterlist[] = {
  {"basic",    1, {  1,  0,  0 }, {   0,  0 }},

  {"PI30",     3, {  1,  0,  0 }, {   0,  0 }},
  {"PI42",     5, {  3, -1,  0 }, {   0,  0 }},
  {"PI33",     3, {  2, -1,  0 }, {   0,  0 }},
  {"PI34",    10, {  7, -4,  0 }, {   0,  0 }},

  {"PC11",     1, {  2, -1,  0 }, {  -1,  0 }},
  {"PC47",    10, { 11, -7,  0 }, { -10,  0 }},
  {"PC36",    10, {  9, -6,  0 }, { -10,  0 }},

  {"H0211",    2, {  1,  1,  0 }, {   1,  0 }},
  {"H211b",    4, {  1,  1,  0 }, {   1,  0 }},
  {"H211PI",   6, {  1,  1,  0 }, {   0,  0 }},

  {"H0312",    4, {  1,  2,  1 }, {   3,  1 }},
  {"H312b",    8, {  1,  2,  1 }, {   3,  1 }},
  {"H312PID", 18, {  1,  2,  1 }, {   0,  0 }},

  {"H0321",    4, {  5,  2, -3 }, {  -1, -3 }},
  {"H321",    18, {  6,  1, -5 }, { -15, -3 }},
};

static PetscErrorCode TSAdaptDSPSetFilter_DSP(TSAdapt adapt,const char *name)
{
  TSAdapt_DSP       *dsp = (TSAdapt_DSP*)adapt->data;
  PetscInt          i,count = (PetscInt)(sizeof(filterlist)/sizeof(filterlist[0]));
  struct FilterTab* tab = NULL;
  PetscBool         match;

  PetscFunctionBegin;
  for (i=0; i<count; i++) {
    PetscCall(PetscStrcasecmp(name,filterlist[i].name,&match));
    if (match) { tab = &filterlist[i]; break; }
  }
  PetscCheck(tab,PetscObjectComm((PetscObject)adapt),PETSC_ERR_ARG_UNKNOWN_TYPE,"Filter name %s not found",name);
  dsp->kBeta[0] = tab->kBeta[0]/tab->scale;
  dsp->kBeta[1] = tab->kBeta[1]/tab->scale;
  dsp->kBeta[2] = tab->kBeta[2]/tab->scale;
  dsp->Alpha[0] = tab->Alpha[0]/tab->scale;
  dsp->Alpha[1] = tab->Alpha[1]/tab->scale;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSAdaptDSPSetPID_DSP(TSAdapt adapt,PetscReal kkI,PetscReal kkP,PetscReal kkD)
{
  TSAdapt_DSP    *dsp = (TSAdapt_DSP*)adapt->data;

  PetscFunctionBegin;
  dsp->kBeta[0] = kkI + kkP + kkD;
  dsp->kBeta[1] = -(kkP + 2*kkD);
  dsp->kBeta[2] = kkD;
  dsp->Alpha[0] = 0;
  dsp->Alpha[1] = 0;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSAdaptSetFromOptions_DSP(PetscOptionItems *PetscOptionsObject,TSAdapt adapt)
{
  TSAdapt_DSP    *dsp = (TSAdapt_DSP*)adapt->data;
  const char     *names[sizeof(filterlist)/sizeof(filterlist[0])];
  PetscInt       count = (PetscInt)(sizeof(filterlist)/sizeof(filterlist[0]));
  PetscInt       index = 2; /* PI42 */
  PetscReal      pid[3] = {1,0,0};
  PetscInt       i,n;
  PetscBool      set;

  PetscFunctionBegin;
  for (i=0; i<count; i++) names[i] = filterlist[i].name;
  PetscCall(PetscOptionsHead(PetscOptionsObject,"DSP adaptive controller options"));

  PetscCall(PetscOptionsEList("-ts_adapt_dsp_filter","Filter name","TSAdaptDSPSetFilter",names,count,names[index],&index,&set));
  if (set) PetscCall(TSAdaptDSPSetFilter(adapt,names[index]));

  PetscCall(PetscOptionsRealArray("-ts_adapt_dsp_pid","PID parameters <kkI,kkP,kkD>","TSAdaptDSPSetPID",pid,(n=3,&n),&set));
  PetscCheck(!set || n,PetscObjectComm((PetscObject)adapt),PETSC_ERR_ARG_WRONG,"Must provide at least one value for PID parameters");
  if (set) PetscCall(TSAdaptDSPSetPID(adapt,pid[0],pid[1],pid[2]));

  PetscCall(PetscOptionsRealArray("-ts_adapt_dsp_kbeta","Filter parameters","",dsp->kBeta,(n=3,&n),&set));
  PetscCheck(!set || n,PetscObjectComm((PetscObject)adapt),PETSC_ERR_ARG_WRONG,"Must provide at least one value for parameter kbeta");
  if (set) for (i=n; i<3; i++) dsp->kBeta[i] = 0;

  PetscCall(PetscOptionsRealArray("-ts_adapt_dsp_alpha","Filter parameters","",dsp->Alpha,(n=2,&n),&set));
  PetscCheck(!set || n,PetscObjectComm((PetscObject)adapt),PETSC_ERR_ARG_WRONG,"Must provide at least one value for parameter alpha");
  if (set) for (i=n; i<2; i++) dsp->Alpha[i] = 0;

  PetscCall(PetscOptionsTail());
  PetscFunctionReturn(0);
}

/*@C
   TSAdaptDSPSetFilter - Sets internal parameters corresponding to the named filter

   Collective on TSAdapt

   Input Parameters:
+  adapt - adaptive controller context
-  name - filter name

   Level: intermediate

   References:
.  * - http://dx.doi.org/10.1145/641876.641877

   Notes:
    Valid filter names are
+  "basic" - similar to TSADAPTBASIC but with different criteria for step rejections.
.  "PI30", "PI42", "PI33", "PI34" - PI controllers.
.  "PC11", "PC47", "PC36" - predictive controllers.
.  "H0211", "H211b", "H211PI" - digital filters with orders dynamics=2, adaptivity=1, filter=1.
.  "H0312", "H312b", "H312PID" - digital filters with orders dynamics=3, adaptivity=1, filter=2.
-  "H0321", "H321" - digital filters with orders dynamics=3, adaptivity=2, filter=1.

   Options Database:
.   -ts_adapt_dsp_filter <name> - Sets predefined controller by name; use -help for a list of available controllers

.seealso: TS, TSAdapt, TSGetAdapt(), TSAdaptDSPSetPID()
@*/
PetscErrorCode TSAdaptDSPSetFilter(TSAdapt adapt,const char *name)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt,TSADAPT_CLASSID,1);
  PetscValidCharPointer(name,2);
  PetscTryMethod(adapt,"TSAdaptDSPSetFilter_C",(TSAdapt,const char*),(adapt,name));
  PetscFunctionReturn(0);
}

/*@
   TSAdaptDSPSetPID - Set the PID controller parameters

   Input Parameters:
+  adapt - adaptive controller context
.  kkI - Integral parameter
.  kkP - Proportional parameter
-  kkD - Derivative parameter

   Level: intermediate

   References:
.  * - http://dx.doi.org/10.1016/j.cam.2005.03.008

   Options Database:
.   -ts_adapt_dsp_pid <kkI,kkP,kkD> - Sets PID controller parameters

.seealso: TS, TSAdapt, TSGetAdapt(), TSAdaptDSPSetFilter()
@*/
PetscErrorCode TSAdaptDSPSetPID(TSAdapt adapt,PetscReal kkI,PetscReal kkP,PetscReal kkD)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adapt,TSADAPT_CLASSID,1);
  PetscValidLogicalCollectiveReal(adapt,kkI,2);
  PetscValidLogicalCollectiveReal(adapt,kkP,3);
  PetscValidLogicalCollectiveReal(adapt,kkD,4);
  PetscTryMethod(adapt,"TSAdaptDSPSetPID_C",(TSAdapt,PetscReal,PetscReal,PetscReal),(adapt,kkI,kkP,kkD));
  PetscFunctionReturn(0);
}

/*MC
   TSADAPTDSP - Adaptive controller for time-stepping based on digital signal processing (DSP)

   Level: intermediate

   References:
+  * - http://dx.doi.org/10.1145/641876.641877
-  * - http://dx.doi.org/10.1016/j.cam.2005.03.008

   Options Database:
+   -ts_adapt_dsp_filter <name> - Sets predefined controller by name; use -help for a list of available controllers
.   -ts_adapt_dsp_pid <kkI,kkP,kkD> - Sets PID controller parameters
.   -ts_adapt_dsp_kbeta <b1,b2,b2> - Sets general filter parameters
-   -ts_adapt_dsp_alpha <a2,a3> - Sets general filter parameters

.seealso: TS, TSAdapt, TSGetAdapt(), TSAdaptDSPSetPID(), TSAdaptDSPSetFilter()
M*/
PETSC_EXTERN PetscErrorCode TSAdaptCreate_DSP(TSAdapt adapt)
{
  TSAdapt_DSP    *dsp;

  PetscFunctionBegin;
  PetscCall(PetscNewLog(adapt,&dsp));
  adapt->reject_safety = 1.0; /* unused */

  adapt->data                = (void*)dsp;
  adapt->ops->choose         = TSAdaptChoose_DSP;
  adapt->ops->setfromoptions = TSAdaptSetFromOptions_DSP;
  adapt->ops->destroy        = TSAdaptDestroy_DSP;
  adapt->ops->view           = TSAdaptView_DSP;

  PetscCall(PetscObjectComposeFunction((PetscObject)adapt,"TSAdaptDSPSetFilter_C",TSAdaptDSPSetFilter_DSP));
  PetscCall(PetscObjectComposeFunction((PetscObject)adapt,"TSAdaptDSPSetPID_C",TSAdaptDSPSetPID_DSP));

  PetscCall(TSAdaptDSPSetFilter_DSP(adapt,"PI42"));
  PetscCall(TSAdaptRestart_DSP(adapt));
  PetscFunctionReturn(0);
}
