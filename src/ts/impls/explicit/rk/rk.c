/*
  Code for time stepping with the Runge-Kutta method

  Notes:
  The general system is written as

  Udot = F(t,U)

*/
#include <petsc/private/tsimpl.h>                /*I   "petscts.h"   I*/
#include <petscdm.h>

static TSRKType  TSRKDefault = TSRK3BS;
static PetscBool TSRKRegisterAllCalled;
static PetscBool TSRKPackageInitialized;

typedef struct _RKTableau *RKTableau;
struct _RKTableau {
  char      *name;
  PetscInt   order;               /* Classical approximation order of the method i              */
  PetscInt   s;                   /* Number of stages                                           */
  PetscBool  FSAL;                /* flag to indicate if tableau is FSAL                        */
  PetscInt   pinterp;             /* Interpolation order                                        */
  PetscReal *A,*b,*c;             /* Tableau                                                    */
  PetscReal *bembed;              /* Embedded formula of order one less (order-1)               */
  PetscReal *binterp;             /* Dense output formula                                       */
  PetscReal  ccfl;                /* Placeholder for CFL coefficient relative to forward Euler  */
};
typedef struct _RKTableauLink *RKTableauLink;
struct _RKTableauLink {
  struct _RKTableau tab;
  RKTableauLink     next;
};
static RKTableauLink RKTableauList;

typedef struct {
  RKTableau    tableau;
  Vec          *Y;               /* States computed during the step */
  Vec          *YdotRHS;         /* Function evaluations for the non-stiff part */
  Vec          *VecDeltaLam;     /* Increment of the adjoint sensitivity w.r.t IC at stage */
  Vec          *VecDeltaMu;      /* Increment of the adjoint sensitivity w.r.t P at stage */
  Vec          *VecSensiTemp;    /* Vector to be timed with Jacobian transpose */
  Vec          VecCostIntegral0; /* backup for roll-backs due to events */
  PetscScalar  *work;            /* Scalar work */
  PetscReal    stage_time;
  TSStepStatus status;
  PetscReal    ptime;
  PetscReal    time_step;
} TS_RK;

/*MC
     TSRK1 - First order forward Euler scheme.

     This method has one stage.

     Level: advanced

.seealso: TSRK
M*/
/*MC
     TSRK2A - Second order RK scheme.

     This method has two stages.

     Level: advanced

.seealso: TSRK
M*/
/*MC
     TSRK3 - Third order RK scheme.

     This method has three stages.

     Level: advanced

.seealso: TSRK
M*/
/*MC
     TSRK3BS - Third order RK scheme of Bogacki-Shampine with 2nd order embedded method.

     This method has four stages.

     Level: advanced

.seealso: TSRK
M*/
/*MC
     TSRK4 - Fourth order RK scheme.

     This is the classical Runge-Kutta method with four stages.

     Level: advanced

.seealso: TSRK
M*/
/*MC
     TSRK5F - Fifth order Fehlberg RK scheme with a 4th order embedded method.

     This method has six stages.

     Level: advanced

.seealso: TSRK
M*/
/*MC
     TSRK5DP - Fifth order Dormand-Prince RK scheme with the 4th order embedded method.

     This method has seven stages.

     Level: advanced

.seealso: TSRK
M*/

#undef __FUNCT__
#define __FUNCT__ "TSRKRegisterAll"
/*@C
  TSRKRegisterAll - Registers all of the Runge-Kutta explicit methods in TSRK

  Not Collective, but should be called by all processes which will need the schemes to be registered

  Level: advanced

.keywords: TS, TSRK, register, all

.seealso:  TSRKRegisterDestroy()
@*/
PetscErrorCode TSRKRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (TSRKRegisterAllCalled) PetscFunctionReturn(0);
  TSRKRegisterAllCalled = PETSC_TRUE;

  {
    const PetscReal
      A[1][1] = {{0.0}},
      b[1]    = {1.0};
    ierr = TSRKRegister(TSRK1FE,1,1,&A[0][0],b,NULL,NULL,1,b);CHKERRQ(ierr);
  }
  {
    const PetscReal
      A[2][2]     = {{0.0,0.0},
                    {1.0,0.0}},
      b[2]        = {0.5,0.5},
      bembed[2]   = {1.0,0};
    ierr = TSRKRegister(TSRK2A,2,2,&A[0][0],b,NULL,bembed,2,b);CHKERRQ(ierr);
  }
  {
    const PetscReal
      A[3][3] = {{0,0,0},
                 {2.0/3.0,0,0},
                 {-1.0/3.0,1.0,0}},
      b[3]    = {0.25,0.5,0.25};
    ierr = TSRKRegister(TSRK3,3,3,&A[0][0],b,NULL,NULL,3,b);CHKERRQ(ierr);
  }
  {
    const PetscReal
      A[4][4] = {{0,0,0,0},
                 {1.0/2.0,0,0,0},
                 {0,3.0/4.0,0,0},
                 {2.0/9.0,1.0/3.0,4.0/9.0,0}},
      b[4]    = {2.0/9.0,1.0/3.0,4.0/9.0,0},
      bembed[4] = {7.0/24.0,1.0/4.0,1.0/3.0,1.0/8.0};
    ierr = TSRKRegister(TSRK3BS,3,4,&A[0][0],b,NULL,bembed,3,b);CHKERRQ(ierr);
  }
  {
    const PetscReal
      A[4][4] = {{0,0,0,0},
                 {0.5,0,0,0},
                 {0,0.5,0,0},
                 {0,0,1.0,0}},
      b[4]    = {1.0/6.0,1.0/3.0,1.0/3.0,1.0/6.0};
    ierr = TSRKRegister(TSRK4,4,4,&A[0][0],b,NULL,NULL,4,b);CHKERRQ(ierr);
  }
  {
    const PetscReal
      A[6][6]   = {{0,0,0,0,0,0},
                   {0.25,0,0,0,0,0},
                   {3.0/32.0,9.0/32.0,0,0,0,0},
                   {1932.0/2197.0,-7200.0/2197.0,7296.0/2197.0,0,0,0},
                   {439.0/216.0,-8.0,3680.0/513.0,-845.0/4104.0,0,0},
                   {-8.0/27.0,2.0,-3544.0/2565.0,1859.0/4104.0,-11.0/40.0,0}},
      b[6]      = {16.0/135.0,0,6656.0/12825.0,28561.0/56430.0,-9.0/50.0,2.0/55.0},
      bembed[6] = {25.0/216.0,0,1408.0/2565.0,2197.0/4104.0,-1.0/5.0,0};
    ierr = TSRKRegister(TSRK5F,5,6,&A[0][0],b,NULL,bembed,5,b);CHKERRQ(ierr);
  }
  {
    const PetscReal
      A[7][7]   = {{0,0,0,0,0,0,0},
                   {1.0/5.0,0,0,0,0,0,0},
                   {3.0/40.0,9.0/40.0,0,0,0,0,0},
                   {44.0/45.0,-56.0/15.0,32.0/9.0,0,0,0,0},
                   {19372.0/6561.0,-25360.0/2187.0,64448.0/6561.0,-212.0/729.0,0,0,0},
                   {9017.0/3168.0,-355.0/33.0,46732.0/5247.0,49.0/176.0,-5103.0/18656.0,0,0},
                   {35.0/384.0,0,500.0/1113.0,125.0/192.0,-2187.0/6784.0,11.0/84.0,0}},
      b[7]      = {35.0/384.0,0,500.0/1113.0,125.0/192.0,-2187.0/6784.0,11.0/84.0,0},
      bembed[7] = {5179.0/57600.0,0,7571.0/16695.0,393.0/640.0,-92097.0/339200.0,187.0/2100.0,1.0/40.0};
    ierr = TSRKRegister(TSRK5DP,5,7,&A[0][0],b,NULL,bembed,5,b);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSRKRegisterDestroy"
/*@C
   TSRKRegisterDestroy - Frees the list of schemes that were registered by TSRKRegister().

   Not Collective

   Level: advanced

.keywords: TSRK, register, destroy
.seealso: TSRKRegister(), TSRKRegisterAll()
@*/
PetscErrorCode TSRKRegisterDestroy(void)
{
  PetscErrorCode ierr;
  RKTableauLink link;

  PetscFunctionBegin;
  while ((link = RKTableauList)) {
    RKTableau t = &link->tab;
    RKTableauList = link->next;
    ierr = PetscFree3(t->A,t->b,t->c);  CHKERRQ(ierr);
    ierr = PetscFree (t->bembed);       CHKERRQ(ierr);
    ierr = PetscFree (t->binterp);      CHKERRQ(ierr);
    ierr = PetscFree (t->name);         CHKERRQ(ierr);
    ierr = PetscFree (link);            CHKERRQ(ierr);
  }
  TSRKRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSRKInitializePackage"
/*@C
  TSRKInitializePackage - This function initializes everything in the TSRK package. It is called
  from PetscDLLibraryRegister() when using dynamic libraries, and on the first call to TSCreate_RK()
  when using static libraries.

  Level: developer

.keywords: TS, TSRK, initialize, package
.seealso: PetscInitialize()
@*/
PetscErrorCode TSRKInitializePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (TSRKPackageInitialized) PetscFunctionReturn(0);
  TSRKPackageInitialized = PETSC_TRUE;
  ierr = TSRKRegisterAll();CHKERRQ(ierr);
  ierr = PetscRegisterFinalize(TSRKFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSRKFinalizePackage"
/*@C
  TSRKFinalizePackage - This function destroys everything in the TSRK package. It is
  called from PetscFinalize().

  Level: developer

.keywords: Petsc, destroy, package
.seealso: PetscFinalize()
@*/
PetscErrorCode TSRKFinalizePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TSRKPackageInitialized = PETSC_FALSE;
  ierr = TSRKRegisterDestroy();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSRKRegister"
/*@C
   TSRKRegister - register an RK scheme by providing the entries in the Butcher tableau and optionally embedded approximations and interpolation

   Not Collective, but the same schemes should be registered on all processes on which they will be used

   Input Parameters:
+  name - identifier for method
.  order - approximation order of method
.  s - number of stages, this is the dimension of the matrices below
.  A - stage coefficients (dimension s*s, row-major)
.  b - step completion table (dimension s; NULL to use last row of A)
.  c - abscissa (dimension s; NULL to use row sums of A)
.  bembed - completion table for embedded method (dimension s; NULL if not available)
.  pinterp - Order of the interpolation scheme, equal to the number of columns of binterp
-  binterp - Coefficients of the interpolation formula (dimension s*pinterp; NULL to reuse binterpt)

   Notes:
   Several RK methods are provided, this function is only needed to create new methods.

   Level: advanced

.keywords: TS, register

.seealso: TSRK
@*/
PetscErrorCode TSRKRegister(TSRKType name,PetscInt order,PetscInt s,
                                 const PetscReal A[],const PetscReal b[],const PetscReal c[],
                                 const PetscReal bembed[],
                                 PetscInt pinterp,const PetscReal binterp[])
{
  PetscErrorCode  ierr;
  RKTableauLink   link;
  RKTableau       t;
  PetscInt        i,j;

  PetscFunctionBegin;
  ierr     = PetscMalloc(sizeof(*link),&link);CHKERRQ(ierr);
  ierr     = PetscMemzero(link,sizeof(*link));CHKERRQ(ierr);
  t        = &link->tab;
  ierr     = PetscStrallocpy(name,&t->name);CHKERRQ(ierr);
  t->order = order;
  t->s     = s;
  ierr     = PetscMalloc3(s*s,&t->A,s,&t->b,s,&t->c);CHKERRQ(ierr);
  ierr     = PetscMemcpy(t->A,A,s*s*sizeof(A[0]));CHKERRQ(ierr);
  if (b)  { ierr = PetscMemcpy(t->b,b,s*sizeof(b[0]));CHKERRQ(ierr); }
  else for (i=0; i<s; i++) t->b[i] = A[(s-1)*s+i];
  if (c)  { ierr = PetscMemcpy(t->c,c,s*sizeof(c[0]));CHKERRQ(ierr); }
  else for (i=0; i<s; i++) for (j=0,t->c[i]=0; j<s; j++) t->c[i] += A[i*s+j];
  t->FSAL = PETSC_TRUE;
  for (i=0; i<s; i++) if (t->A[(s-1)*s+i] != t->b[i]) t->FSAL = PETSC_FALSE;
  if (bembed) {
    ierr = PetscMalloc1(s,&t->bembed);CHKERRQ(ierr);
    ierr = PetscMemcpy(t->bembed,bembed,s*sizeof(bembed[0]));CHKERRQ(ierr);
  }

  t->pinterp     = pinterp;
  ierr           = PetscMalloc1(s*pinterp,&t->binterp);CHKERRQ(ierr);
  ierr           = PetscMemcpy(t->binterp,binterp,s*pinterp*sizeof(binterp[0]));CHKERRQ(ierr);
  link->next     = RKTableauList;
  RKTableauList = link;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSEvaluateStep_RK"
/*
 The step completion formula is

 x1 = x0 + h b^T YdotRHS

 This function can be called before or after ts->vec_sol has been updated.
 Suppose we have a completion formula (b) and an embedded formula (be) of different order.
 We can write

 x1e = x0 + h be^T YdotRHS
     = x1 - h b^T YdotRHS + h be^T YdotRHS
     = x1 + h (be - b)^T YdotRHS

 so we can evaluate the method with different order even after the step has been optimistically completed.
*/
static PetscErrorCode TSEvaluateStep_RK(TS ts,PetscInt order,Vec X,PetscBool *done)
{
  TS_RK         *rk   = (TS_RK*)ts->data;
  RKTableau      tab  = rk->tableau;
  PetscScalar   *w    = rk->work;
  PetscReal      h;
  PetscInt       s    = tab->s,j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (rk->status) {
  case TS_STEP_INCOMPLETE:
  case TS_STEP_PENDING:
    h = ts->time_step; break;
  case TS_STEP_COMPLETE:
    h = ts->ptime - ts->ptime_prev; break;
  default: SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_PLIB,"Invalid TSStepStatus");
  }
  if (order == tab->order) {
    if (rk->status == TS_STEP_INCOMPLETE) {
      ierr = VecCopy(ts->vec_sol,X);CHKERRQ(ierr);
      for (j=0; j<s; j++) w[j] = h*tab->b[j];
      ierr = VecMAXPY(X,s,w,rk->YdotRHS);CHKERRQ(ierr);
    } else {ierr = VecCopy(ts->vec_sol,X);CHKERRQ(ierr);}
    PetscFunctionReturn(0);
  } else if (order == tab->order-1) {
    if (!tab->bembed) goto unavailable;
    if (rk->status == TS_STEP_INCOMPLETE) { /* Complete with the embedded method (be) */
      ierr = VecCopy(ts->vec_sol,X);CHKERRQ(ierr);
      for (j=0; j<s; j++) w[j] = h*tab->bembed[j];
      ierr = VecMAXPY(X,s,w,rk->YdotRHS);CHKERRQ(ierr);
    } else { /* Rollback and re-complete using (be-b) */
      ierr = VecCopy(ts->vec_sol,X);CHKERRQ(ierr);
      for (j=0; j<s; j++) w[j] = h*(tab->bembed[j] - tab->b[j]);
      ierr = VecMAXPY(X,s,w,rk->YdotRHS);CHKERRQ(ierr);
      if (ts->vec_costintegral && ts->costintegralfwd) {
        ierr = VecCopy(rk->VecCostIntegral0,ts->vec_costintegral);CHKERRQ(ierr);
      }
    }
    if (done) *done = PETSC_TRUE;
    PetscFunctionReturn(0);
  }
unavailable:
  if (done) *done = PETSC_FALSE;
  else SETERRQ3(PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"RK '%s' of order %D cannot evaluate step at order %D. Consider using -ts_adapt_type none or a different method that has an embedded estimate.",tab->name,tab->order,order);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSForwardCostIntegral_RK"
static PetscErrorCode TSForwardCostIntegral_RK(TS ts)
{
  TS_RK           *rk = (TS_RK*)ts->data;
  RKTableau       tab = rk->tableau;
  const PetscInt  s = tab->s;
  const PetscReal *b = tab->b,*c = tab->c;
  Vec             *Y = rk->Y;
  PetscInt        i;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  /* backup cost integral */
  ierr = VecCopy(ts->vec_costintegral,rk->VecCostIntegral0);CHKERRQ(ierr);
  for (i=s-1; i>=0; i--) {
    /* Evolve ts->vec_costintegral to compute integrals */
    ierr = TSAdjointComputeCostIntegrand(ts,rk->ptime+rk->time_step*(1.0-c[i]),Y[i],ts->vec_costintegrand);CHKERRQ(ierr);
    ierr = VecAXPY(ts->vec_costintegral,rk->time_step*b[i],ts->vec_costintegrand);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAdjointCostIntegral_RK"
static PetscErrorCode TSAdjointCostIntegral_RK(TS ts)
{
  TS_RK           *rk = (TS_RK*)ts->data;
  RKTableau       tab = rk->tableau;
  const PetscInt  s = tab->s;
  const PetscReal *b = tab->b,*c = tab->c;
  Vec             *Y = rk->Y;
  PetscInt        i;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  for (i=s-1; i>=0; i--) {
    /* Evolve ts->vec_costintegral to compute integrals */
    ierr = TSAdjointComputeCostIntegrand(ts,ts->ptime-ts->time_step*(1.0-c[i]),Y[i],ts->vec_costintegrand);CHKERRQ(ierr);
    ierr = VecAXPY(ts->vec_costintegral,-ts->time_step*b[i],ts->vec_costintegrand);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSRollBack_RK"
static PetscErrorCode TSRollBack_RK(TS ts)
{
  TS_RK           *rk = (TS_RK*)ts->data;
  RKTableau       tab = rk->tableau;
  const PetscInt  s  = tab->s;
  const PetscReal *b = tab->b;
  PetscScalar     *w = rk->work;
  Vec             *YdotRHS = rk->YdotRHS;
  PetscInt        j;
  PetscReal       h;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  switch (rk->status) {
  case TS_STEP_INCOMPLETE:
  case TS_STEP_PENDING:
    h = ts->time_step; break;
  case TS_STEP_COMPLETE:
    h = ts->ptime - ts->ptime_prev; break;
  default: SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_PLIB,"Invalid TSStepStatus");
  }
  for (j=0; j<s; j++) w[j] = -h*b[j];
  ierr = VecMAXPY(ts->vec_sol,s,w,YdotRHS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSStep_RK"
static PetscErrorCode TSStep_RK(TS ts)
{
  TS_RK           *rk   = (TS_RK*)ts->data;
  RKTableau        tab  = rk->tableau;
  const PetscInt   s = tab->s;
  const PetscReal *A = tab->A,*c = tab->c;
  PetscScalar     *w = rk->work;
  Vec             *Y = rk->Y,*YdotRHS = rk->YdotRHS;
  TSAdapt          adapt;
  PetscInt         i,j;
  PetscInt         rejections = 0;
  PetscBool        stageok,accept = PETSC_TRUE;
  PetscReal        next_time_step = ts->time_step;
  PetscErrorCode   ierr;

  PetscFunctionBegin;

  rk->status = TS_STEP_INCOMPLETE;
  while (!ts->reason && rk->status != TS_STEP_COMPLETE) {
    PetscReal t = ts->ptime;
    PetscReal h = ts->time_step;
    for (i=0; i<s; i++) {
      rk->stage_time = t + h*c[i];
      ierr = TSPreStage(ts,rk->stage_time); CHKERRQ(ierr);
      ierr = VecCopy(ts->vec_sol,Y[i]);CHKERRQ(ierr);
      for (j=0; j<i; j++) w[j] = h*A[i*s+j];
      ierr = VecMAXPY(Y[i],i,w,YdotRHS);CHKERRQ(ierr);
      ierr = TSPostStage(ts,rk->stage_time,i,Y); CHKERRQ(ierr);
      ierr = TSGetAdapt(ts,&adapt);CHKERRQ(ierr);
      ierr = TSAdaptCheckStage(adapt,ts,rk->stage_time,Y[i],&stageok);CHKERRQ(ierr);
      if (!stageok) goto reject_step;
      ierr = TSComputeRHSFunction(ts,t+h*c[i],Y[i],YdotRHS[i]);CHKERRQ(ierr);
    }

    rk->status = TS_STEP_INCOMPLETE;
    ierr = TSEvaluateStep(ts,tab->order,ts->vec_sol,NULL);CHKERRQ(ierr);
    rk->status = TS_STEP_PENDING;
    ierr = TSGetAdapt(ts,&adapt);CHKERRQ(ierr);
    ierr = TSAdaptCandidatesClear(adapt);CHKERRQ(ierr);
    ierr = TSAdaptCandidateAdd(adapt,tab->name,tab->order,1,tab->ccfl,1.*tab->s,PETSC_TRUE);CHKERRQ(ierr);
    ierr = TSAdaptChoose(adapt,ts,ts->time_step,NULL,&next_time_step,&accept);CHKERRQ(ierr);
    rk->status = accept ? TS_STEP_COMPLETE : TS_STEP_INCOMPLETE;
    if (!accept) { /* Roll back the current step */
      ierr = TSRollBack_RK(ts);CHKERRQ(ierr);
      ts->time_step = next_time_step;
      goto reject_step;
    }

    if (ts->costintegralfwd) { /* Save the info for the later use in cost integral evaluation*/
      rk->ptime     = ts->ptime;
      rk->time_step = ts->time_step;
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

#undef __FUNCT__
#define __FUNCT__ "TSAdjointSetUp_RK"
static PetscErrorCode TSAdjointSetUp_RK(TS ts)
{
  TS_RK         *rk  = (TS_RK*)ts->data;
  RKTableau      tab = rk->tableau;
  PetscInt       s   = tab->s;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ts->adjointsetupcalled++) PetscFunctionReturn(0);
  ierr = VecDuplicateVecs(ts->vecs_sensi[0],s*ts->numcost,&rk->VecDeltaLam);CHKERRQ(ierr);
  if(ts->vecs_sensip) {
    ierr = VecDuplicateVecs(ts->vecs_sensip[0],s*ts->numcost,&rk->VecDeltaMu);CHKERRQ(ierr);
  }
  ierr = VecDuplicateVecs(ts->vecs_sensi[0],ts->numcost,&rk->VecSensiTemp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSAdjointStep_RK"
static PetscErrorCode TSAdjointStep_RK(TS ts)
{
  TS_RK           *rk   = (TS_RK*)ts->data;
  RKTableau        tab  = rk->tableau;
  const PetscInt   s    = tab->s;
  const PetscReal *A = tab->A,*b = tab->b,*c = tab->c;
  PetscScalar     *w    = rk->work;
  Vec             *Y    = rk->Y,*VecDeltaLam = rk->VecDeltaLam,*VecDeltaMu = rk->VecDeltaMu,*VecSensiTemp = rk->VecSensiTemp;
  PetscInt         i,j,nadj;
  PetscReal        t = ts->ptime;
  PetscErrorCode   ierr;
  PetscReal        h = ts->time_step;
  Mat              J,Jp;

  PetscFunctionBegin;
  rk->status = TS_STEP_INCOMPLETE;
  for (i=s-1; i>=0; i--) {
    rk->stage_time = t + h*(1.0-c[i]);
    for (nadj=0; nadj<ts->numcost; nadj++) {
      ierr = VecCopy(ts->vecs_sensi[nadj],VecSensiTemp[nadj]);CHKERRQ(ierr);
      ierr = VecScale(VecSensiTemp[nadj],-h*b[i]);CHKERRQ(ierr);
      for (j=i+1; j<s; j++) {
        ierr = VecAXPY(VecSensiTemp[nadj],-h*A[j*s+i],VecDeltaLam[nadj*s+j]);CHKERRQ(ierr);
      }
    }
    /* Stage values of lambda */
    ierr = TSGetRHSJacobian(ts,&J,&Jp,NULL,NULL);CHKERRQ(ierr);
    ierr = TSComputeRHSJacobian(ts,rk->stage_time,Y[i],J,Jp);CHKERRQ(ierr);
    if (ts->vec_costintegral) {
      ierr = TSAdjointComputeDRDYFunction(ts,rk->stage_time,Y[i],ts->vecs_drdy);CHKERRQ(ierr);
    }
    for (nadj=0; nadj<ts->numcost; nadj++) {
      ierr = MatMultTranspose(J,VecSensiTemp[nadj],VecDeltaLam[nadj*s+i]);CHKERRQ(ierr);
      if (ts->vec_costintegral) {
        ierr = VecAXPY(VecDeltaLam[nadj*s+i],-h*b[i],ts->vecs_drdy[nadj]);CHKERRQ(ierr);
      }
    }

    /* Stage values of mu */
    if(ts->vecs_sensip) {
      ierr = TSAdjointComputeRHSJacobian(ts,rk->stage_time,Y[i],ts->Jacp);CHKERRQ(ierr);
      if (ts->vec_costintegral) {
        ierr = TSAdjointComputeDRDPFunction(ts,rk->stage_time,Y[i],ts->vecs_drdp);CHKERRQ(ierr);
      }

      for (nadj=0; nadj<ts->numcost; nadj++) {
        ierr = MatMultTranspose(ts->Jacp,VecSensiTemp[nadj],VecDeltaMu[nadj*s+i]);CHKERRQ(ierr);
        if (ts->vec_costintegral) {
          ierr = VecAXPY(VecDeltaMu[nadj*s+i],-h*b[i],ts->vecs_drdp[nadj]);CHKERRQ(ierr);
        }
      }
    }
  }

  for (j=0; j<s; j++) w[j] = 1.0;
  for (nadj=0; nadj<ts->numcost; nadj++) {
    ierr = VecMAXPY(ts->vecs_sensi[nadj],s,w,&VecDeltaLam[nadj*s]);CHKERRQ(ierr);
    if(ts->vecs_sensip) {
      ierr = VecMAXPY(ts->vecs_sensip[nadj],s,w,&VecDeltaMu[nadj*s]);CHKERRQ(ierr);
    }
  }
  rk->status = TS_STEP_COMPLETE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSInterpolate_RK"
static PetscErrorCode TSInterpolate_RK(TS ts,PetscReal itime,Vec X)
{
  TS_RK           *rk = (TS_RK*)ts->data;
  PetscInt         s  = rk->tableau->s,pinterp = rk->tableau->pinterp,i,j;
  PetscReal        h;
  PetscReal        tt,t;
  PetscScalar     *b;
  const PetscReal *B = rk->tableau->binterp;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (!B) SETERRQ1(PetscObjectComm((PetscObject)ts),PETSC_ERR_SUP,"TSRK %s does not have an interpolation formula",rk->tableau->name);

  switch (rk->status) {
  case TS_STEP_INCOMPLETE:
  case TS_STEP_PENDING:
    h = ts->time_step;
    t = (itime - ts->ptime)/h;
    break;
  case TS_STEP_COMPLETE:
    h = ts->ptime - ts->ptime_prev;
    t = (itime - ts->ptime)/h + 1; /* In the interval [0,1] */
    break;
  default: SETERRQ(PetscObjectComm((PetscObject)ts),PETSC_ERR_PLIB,"Invalid TSStepStatus");
  }
  ierr = PetscMalloc1(s,&b);CHKERRQ(ierr);
  for (i=0; i<s; i++) b[i] = 0;
  for (j=0,tt=t; j<pinterp; j++,tt*=t) {
    for (i=0; i<s; i++) {
      b[i]  += h * B[i*pinterp+j] * tt;
    }
  }

  ierr = VecCopy(rk->Y[0],X);CHKERRQ(ierr);
  ierr = VecMAXPY(X,s,b,rk->YdotRHS);CHKERRQ(ierr);

  ierr = PetscFree(b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/

#undef __FUNCT__
#define __FUNCT__ "TSRKTableauReset"
static PetscErrorCode TSRKTableauReset(TS ts)
{
  TS_RK          *rk = (TS_RK*)ts->data;
  RKTableau      tab = rk->tableau;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!tab) PetscFunctionReturn(0);
  ierr = PetscFree(rk->work);CHKERRQ(ierr);
  ierr = VecDestroyVecs(tab->s,&rk->Y);CHKERRQ(ierr);
  ierr = VecDestroyVecs(tab->s,&rk->YdotRHS);CHKERRQ(ierr);
  ierr = VecDestroyVecs(tab->s*ts->numcost,&rk->VecDeltaLam);CHKERRQ(ierr);
  ierr = VecDestroyVecs(tab->s*ts->numcost,&rk->VecDeltaMu);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSReset_RK"
static PetscErrorCode TSReset_RK(TS ts)
{
  TS_RK         *rk = (TS_RK*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSRKTableauReset(ts);CHKERRQ(ierr);
  ierr = VecDestroy(&rk->VecCostIntegral0);CHKERRQ(ierr);
  ierr = VecDestroyVecs(ts->numcost,&rk->VecSensiTemp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSDestroy_RK"
static PetscErrorCode TSDestroy_RK(TS ts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSReset_RK(ts);CHKERRQ(ierr);
  ierr = PetscFree(ts->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSRKGetType_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSRKSetType_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMCoarsenHook_TSRK"
static PetscErrorCode DMCoarsenHook_TSRK(DM fine,DM coarse,void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMRestrictHook_TSRK"
static PetscErrorCode DMRestrictHook_TSRK(DM fine,Mat restrct,Vec rscale,Mat inject,DM coarse,void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMSubDomainHook_TSRK"
static PetscErrorCode DMSubDomainHook_TSRK(DM dm,DM subdm,void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSubDomainRestrictHook_TSRK"
static PetscErrorCode DMSubDomainRestrictHook_TSRK(DM dm,VecScatter gscat,VecScatter lscat,DM subdm,void *ctx)
{

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
/*
#undef __FUNCT__
#define __FUNCT__ "RKSetAdjCoe"
static PetscErrorCode RKSetAdjCoe(RKTableau tab)
{
  PetscReal *A,*b;
  PetscInt        s,i,j;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  s    = tab->s;
  ierr = PetscMalloc2(s*s,&A,s,&b);CHKERRQ(ierr);

  for (i=0; i<s; i++)
    for (j=0; j<s; j++) {
      A[i*s+j] = (tab->b[s-1-i]==0) ? 0: -tab->A[s-1-i+(s-1-j)*s] * tab->b[s-1-j] / tab->b[s-1-i];
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Coefficients: A[%D][%D]=%.6f\n",i,j,A[i*s+j]);CHKERRQ(ierr);
    }

  for (i=0; i<s; i++) b[i] = (tab->b[s-1-i]==0)? 0: -tab->b[s-1-i];

  ierr  = PetscMemcpy(tab->A,A,s*s*sizeof(A[0]));CHKERRQ(ierr);
  ierr  = PetscMemcpy(tab->b,b,s*sizeof(b[0]));CHKERRQ(ierr);
  ierr  = PetscFree2(A,b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
*/

#undef __FUNCT__
#define __FUNCT__ "TSRKTableauSetUp"
static PetscErrorCode TSRKTableauSetUp(TS ts)
{
  TS_RK         *rk  = (TS_RK*)ts->data;
  RKTableau      tab = rk->tableau;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc1(tab->s,&rk->work);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(ts->vec_sol,tab->s,&rk->Y);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(ts->vec_sol,tab->s,&rk->YdotRHS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "TSSetUp_RK"
static PetscErrorCode TSSetUp_RK(TS ts)
{
  TS_RK         *rk = (TS_RK*)ts->data;
  PetscErrorCode ierr;
  DM             dm;

  PetscFunctionBegin;
  ierr = TSRKTableauSetUp(ts);CHKERRQ(ierr);
  if (!rk->VecCostIntegral0 && ts->vec_costintegral && ts->costintegralfwd) { /* back up cost integral */
    ierr = VecDuplicate(ts->vec_costintegral,&rk->VecCostIntegral0);CHKERRQ(ierr);
  }
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  if (dm) {
    ierr = DMCoarsenHookAdd(dm,DMCoarsenHook_TSRK,DMRestrictHook_TSRK,ts);CHKERRQ(ierr);
    ierr = DMSubDomainHookAdd(dm,DMSubDomainHook_TSRK,DMSubDomainRestrictHook_TSRK,ts);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


/*------------------------------------------------------------*/

#undef __FUNCT__
#define __FUNCT__ "TSSetFromOptions_RK"
static PetscErrorCode TSSetFromOptions_RK(PetscOptionItems *PetscOptionsObject,TS ts)
{
  TS_RK         *rk = (TS_RK*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"RK ODE solver options");CHKERRQ(ierr);
  {
    RKTableauLink  link;
    PetscInt       count,choice;
    PetscBool      flg;
    const char   **namelist;
    for (link=RKTableauList,count=0; link; link=link->next,count++) ;
    ierr = PetscMalloc1(count,&namelist);CHKERRQ(ierr);
    for (link=RKTableauList,count=0; link; link=link->next,count++) namelist[count] = link->tab.name;
    ierr = PetscOptionsEList("-ts_rk_type","Family of RK method","TSRKSetType",(const char*const*)namelist,count,rk->tableau->name,&choice,&flg);CHKERRQ(ierr);
    if (flg) {ierr = TSRKSetType(ts,namelist[choice]);CHKERRQ(ierr);}
    ierr = PetscFree(namelist);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFormatRealArray"
static PetscErrorCode PetscFormatRealArray(char buf[],size_t len,const char *fmt,PetscInt n,const PetscReal x[])
{
  PetscErrorCode ierr;
  PetscInt       i;
  size_t         left,count;
  char           *p;

  PetscFunctionBegin;
  for (i=0,p=buf,left=len; i<n; i++) {
    ierr = PetscSNPrintfCount(p,left,fmt,&count,x[i]);CHKERRQ(ierr);
    if (count >= left) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Insufficient space in buffer");
    left -= count;
    p    += count;
    *p++  = ' ';
  }
  p[i ? 0 : -1] = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSView_RK"
static PetscErrorCode TSView_RK(TS ts,PetscViewer viewer)
{
  TS_RK          *rk = (TS_RK*)ts->data;
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    RKTableau tab  = rk->tableau;
    TSRKType  rktype;
    char      buf[512];
    ierr = TSRKGetType(ts,&rktype);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  RK %s\n",rktype);CHKERRQ(ierr);
    ierr = PetscFormatRealArray(buf,sizeof(buf),"% 8.6f",tab->s,tab->c);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  Abscissa     c = %s\n",buf);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"FSAL: %s\n",tab->FSAL ? "yes" : "no");CHKERRQ(ierr);
  }
  if (ts->adapt) {ierr = TSAdaptView(ts->adapt,viewer);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSLoad_RK"
static PetscErrorCode TSLoad_RK(TS ts,PetscViewer viewer)
{
  PetscErrorCode ierr;
  TSAdapt        adapt;

  PetscFunctionBegin;
  ierr = TSGetAdapt(ts,&adapt);CHKERRQ(ierr);
  ierr = TSAdaptLoad(adapt,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSRKSetType"
/*@C
  TSRKSetType - Set the type of RK scheme

  Logically collective

  Input Parameter:
+  ts - timestepping context
-  rktype - type of RK-scheme

  Level: intermediate

.seealso: TSRKGetType(), TSRK, TSRK2, TSRK3, TSRKPRSSP2, TSRK4, TSRK5
@*/
PetscErrorCode TSRKSetType(TS ts,TSRKType rktype)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = PetscTryMethod(ts,"TSRKSetType_C",(TS,TSRKType),(ts,rktype));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSRKGetType"
/*@C
  TSRKGetType - Get the type of RK scheme

  Logically collective

  Input Parameter:
.  ts - timestepping context

  Output Parameter:
.  rktype - type of RK-scheme

  Level: intermediate

.seealso: TSRKGetType()
@*/
PetscErrorCode TSRKGetType(TS ts,TSRKType *rktype)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = PetscUseMethod(ts,"TSRKGetType_C",(TS,TSRKType*),(ts,rktype));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSRKGetType_RK"
static PetscErrorCode TSRKGetType_RK(TS ts,TSRKType *rktype)
{
  TS_RK *rk = (TS_RK*)ts->data;

  PetscFunctionBegin;
  *rktype = rk->tableau->name;
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "TSRKSetType_RK"
static PetscErrorCode TSRKSetType_RK(TS ts,TSRKType rktype)
{
  TS_RK          *rk = (TS_RK*)ts->data;
  PetscErrorCode ierr;
  PetscBool      match;
  RKTableauLink  link;

  PetscFunctionBegin;
  if (rk->tableau) {
    ierr = PetscStrcmp(rk->tableau->name,rktype,&match);CHKERRQ(ierr);
    if (match) PetscFunctionReturn(0);
  }
  for (link = RKTableauList; link; link=link->next) {
    ierr = PetscStrcmp(link->tab.name,rktype,&match);CHKERRQ(ierr);
    if (match) {
      if (ts->setupcalled) {ierr = TSRKTableauReset(ts);CHKERRQ(ierr);}
      rk->tableau = &link->tab;
      if (ts->setupcalled) {ierr = TSRKTableauSetUp(ts);CHKERRQ(ierr);}
      PetscFunctionReturn(0);
    }
  }
  SETERRQ1(PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_UNKNOWN_TYPE,"Could not find '%s'",rktype);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSGetStages_RK"
static PetscErrorCode  TSGetStages_RK(TS ts,PetscInt *ns,Vec **Y)
{
  TS_RK          *rk = (TS_RK*)ts->data;

  PetscFunctionBegin;
  *ns = rk->tableau->s;
  if(Y) *Y  = rk->Y;
  PetscFunctionReturn(0);
}


/* ------------------------------------------------------------ */
/*MC
      TSRK - ODE and DAE solver using Runge-Kutta schemes

  The user should provide the right hand side of the equation
  using TSSetRHSFunction().

  Notes:
  The default is TSRK3, it can be changed with TSRKSetType() or -ts_rk_type

  Level: beginner

.seealso:  TSCreate(), TS, TSSetType(), TSRKSetType(), TSRKGetType(), TSRKSetFullyImplicit(), TSRK2D, TTSRK2E, TSRK3,
           TSRK4, TSRK5, TSRKPRSSP2, TSRKBPR3, TSRKType, TSRKRegister()

M*/
#undef __FUNCT__
#define __FUNCT__ "TSCreate_RK"
PETSC_EXTERN PetscErrorCode TSCreate_RK(TS ts)
{
  TS_RK          *rk;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSRKInitializePackage();CHKERRQ(ierr);

  ts->ops->reset          = TSReset_RK;
  ts->ops->destroy        = TSDestroy_RK;
  ts->ops->view           = TSView_RK;
  ts->ops->load           = TSLoad_RK;
  ts->ops->setup          = TSSetUp_RK;
  ts->ops->adjointsetup   = TSAdjointSetUp_RK;
  ts->ops->step           = TSStep_RK;
  ts->ops->interpolate    = TSInterpolate_RK;
  ts->ops->evaluatestep   = TSEvaluateStep_RK;
  ts->ops->rollback       = TSRollBack_RK;
  ts->ops->setfromoptions = TSSetFromOptions_RK;
  ts->ops->getstages      = TSGetStages_RK;
  ts->ops->adjointstep    = TSAdjointStep_RK;

  ts->ops->adjointintegral = TSAdjointCostIntegral_RK;
  ts->ops->forwardintegral = TSForwardCostIntegral_RK;

  ierr = PetscNewLog(ts,&rk);CHKERRQ(ierr);
  ts->data = (void*)rk;

  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSRKGetType_C",TSRKGetType_RK);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSRKSetType_C",TSRKSetType_RK);CHKERRQ(ierr);

  ierr = TSRKSetType(ts,TSRKDefault);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
