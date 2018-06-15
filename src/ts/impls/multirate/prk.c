/*
  Code for time stepping with the Partitioned Runge-Kutta method

  Notes:
  1) The general system is written as
     Udot = F(t,U) for combined RHS multi-rate RK,
     user should give the indexes for both slow and fast components;
  2) The general system is written as
     Usdot = Fs(t,Us,Uf)
     Ufdot = Ff(t,Us,Uf) for partitioned RHS multi-rate RK,
     user should partioned RHS by themselves and also provide the indexes for both slow and fast components.
*/

#include <petsc/private/tsimpl.h>                /*I   "petscts.h"   I*/
#include <petscdm.h>

static TSPRKType  TSPRKDefault = TSPRKM2;
static PetscBool  TSPRKRegisterAllCalled;
static PetscBool  TSPRKPackageInitialized;

typedef struct _PRKTableau *PRKTableau;
struct _PRKTableau {
  char       *name;
  PetscInt   order;                          /* Classical approximation order of the method i  */
  PetscInt   s;                              /* Number of stages                               */
  PetscReal  *Af,*bf,*cf;                    /* Tableau for fast components                    */
  PetscReal  *As,*bs,*cs;                    /* Tableau for slow components                    */
};
typedef struct _PRKTableauLink *PRKTableauLink;
struct _PRKTableauLink {
  struct _PRKTableau tab;
  PRKTableauLink     next;
};
static PRKTableauLink PRKTableauList;

typedef struct {
  PRKTableau    tableau;
  Vec           *Y;                          /* States computed during the step                           */
  Vec           *YdotRHSfast_combine;        /* Function evaluations by fast tableau for all components   */
  Vec           *YdotRHSslow_combine;        /* Function evaluations by slow tableau for all components   */
  Vec           *YdotRHS_fast;               /* Function evaluations by fast tableau for fast components  */
  Vec           *YdotRHS_slow;               /* Function evaluations by slow tableau for slow components  */
  PetscScalar   *work_fast;                  /* Scalar work_fast by fast tableau                          */
  PetscScalar   *work_slow;                  /* Scalar work_slow by slow tableau                          */
  PetscReal     stage_time;
  TSStepStatus  status;
  PetscReal     ptime;
  PetscReal     time_step;
} TS_PRK;

/*MC
     TSPRKM2 - Second Order Partitioned Runge Kutta scheme.

     This method has four stages for both slow and fast parts.

     Options database:
.     -ts_prk_type pm2

     Level: advanced

.seealso: TSPRK, TSPRKType, TSPRKSetType()
M*/
/*MC
     TSPRKM3 - Third Order Partitioned Runge Kutta scheme.

     This method has eight stages for both slow and fast parts.

     Options database:
.     -ts_prk_type pm3

     Level: advanced

.seealso: TSPRK, TSPRKType, TSPRKSetType()
M*/
/*MC
     TSPRKRFSMR2 - Second Order Partitioned Runge Kutta scheme.

     This method has five stages for both slow and fast parts.

     Options database:
.     -ts_prk_type p2

     Level: advanced

.seealso: TSPRK, TSPRKType, TSPRKSetType()
M*/
/*MC
     TSPRKRFSMR3 - Third Order Partitioned Runge Kutta scheme.

     This method has ten stages for both slow and fast parts.

     Options database:
.     -ts_prk_type p3

     Level: advanced

.seealso: TSPRK, TSPRKType, TSPRKSetType()
M*/

/*@C
  TSPRKRegisterAll - Registers all of the Partirioned Runge-Kutta explicit methods in TSPRK

  Not Collective, but should be called by all processes which will need the schemes to be registered

  Level: advanced

.keywords: TS, TSPRK, register, all

.seealso:  TSPRKRegisterDestroy()
@*/
PetscErrorCode TSPRKRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (TSPRKRegisterAllCalled) PetscFunctionReturn(0);
  TSPRKRegisterAllCalled = PETSC_TRUE;

#define RC PetscRealConstant
  {
      const PetscReal
       As[4][4]  = {{0,0,0,0},
                   {RC(1.0),0,0,0},
                   {0,0,0,0},
                   {0,0,RC(1.0),0}},
        A[4][4]  = {{0,0,0,0},
                   {RC(0.5),0,0,0},
                   {RC(0.25),RC(0.25),0,0},
                   {RC(0.25),RC(0.25),RC(0.5),0}},
          bs[4]  = {RC(0.25),RC(0.25),RC(0.25),RC(0.25)},
           b[4]  = {RC(0.25),RC(0.25),RC(0.25),RC(0.25)};
           ierr  = TSPRKRegister(TSPRKM2,2,4,&As[0][0],bs,NULL,&A[0][0],b,NULL);CHKERRQ(ierr);
  }

  {
      const PetscReal
        As[8][8] = {{0,0,0,0,0,0,0,0},
                    {RC(1.0)/RC(2.0),0,0,0,0,0,0,0},
                    {RC(-1.0)/RC(6.0),RC(2.0)/RC(3.0),0,0,0,0,0,0},
                    {RC(1.0)/RC(3.0),RC(-1.0)/RC(3.0),RC(1.0),0,0,0,0,0},
                    {0,0,0,0,0,0,0,0},
                    {0,0,0,0,RC(1.0)/RC(2.0),0,0,0},
                    {0,0,0,0,RC(-1.0)/RC(6.0),RC(2.0)/RC(3.0),0,0},
                    {0,0,0,0,RC(1.0)/RC(3.0),RC(-1.0)/RC(3.0),RC(1.0),0}},
         A[8][8] = {{0,0,0,0,0,0,0,0},
                    {RC(1.0)/RC(4.0),0,0,0,0,0,0,0},
                    {RC(-1.0)/RC(12.0),RC(1.0)/RC(3.0),0,0,0,0,0,0},
                    {RC(1.0)/RC(6.0),RC(-1.0)/RC(6.0),RC(1.0)/RC(2.0),0,0,0,0,0},
                    {RC(1.0)/RC(12.0),RC(1.0)/RC(6.0),RC(1.0)/RC(6.0),RC(1.0)/RC(12.0),0,0,0,0},
                    {RC(1.0)/RC(12.0),RC(1.0)/RC(6.0),RC(1.0)/RC(6.0),RC(1.0)/RC(12.0),RC(1.0)/RC(4.0),0,0,0},
                    {RC(1.0)/RC(12.0),RC(1.0)/RC(6.0),RC(1.0)/RC(6.0),RC(1.0)/RC(12.0),RC(-1.0)/RC(12.0),RC(1.0)/RC(3.0),0,0},
                    {RC(1.0)/RC(12.0),RC(1.0)/RC(6.0),RC(1.0)/RC(6.0),RC(1.0)/RC(12.0),RC(1.0)/RC(6.0),RC(-1.0)/RC(6.0),RC(1.0)/RC(2.0),0}},
          bs[8] = {RC(1.0)/RC(12.0),RC(1.0)/RC(6.0),RC(1.0)/RC(6.0),RC(1.0)/RC(12.0),RC(1.0)/RC(12.0),RC(1.0)/RC(6.0),RC(1.0)/RC(6.0),RC(1.0)/RC(12.0)},
           b[8] = {RC(1.0)/RC(12.0),RC(1.0)/RC(6.0),RC(1.0)/RC(6.0),RC(1.0)/RC(12.0),RC(1.0)/RC(12.0),RC(1.0)/RC(6.0),RC(1.0)/RC(6.0),RC(1.0)/RC(12.0)};
           ierr = TSPRKRegister(TSPRKM3,3,8,&As[0][0],bs,NULL,&A[0][0],b,NULL);CHKERRQ(ierr);
  }

  {
     const PetscReal
       As[5][5] = {{0,0,0,0,0},
                   {RC(1.0)/RC(2.0),0,0,0,0},
                   {RC(1.0)/RC(2.0),0,0,0,0},
                   {RC(1.0),0,0,0,0},
                   {RC(1.0),0,0,0,0}},
        A[5][5] = {{0,0,0,0,0},
                   {RC(1.0)/RC(2.0),0,0,0,0},
                   {RC(1.0)/RC(4.0),RC(1.0)/RC(4.0),0,0,0},
                   {RC(1.0)/RC(4.0),RC(1.0)/RC(4.0),RC(1.0)/RC(2.0),0,0},
                   {RC(1.0)/RC(4.0),RC(1.0)/RC(4.0),RC(1.0)/RC(4.0),RC(1.0)/RC(4.0),0}},
          bs[5] = {RC(1.0)/RC(2.0),0,0,0,RC(1.0)/RC(2.0)},
           b[5] = {RC(1.0)/RC(4.0),RC(1.0)/RC(4.0),RC(1.0)/RC(4.0),RC(1.0)/RC(4.0),0};
           ierr = TSPRKRegister(TSPRKRFSMR2,2,5,&As[0][0],bs,NULL,&A[0][0],b,NULL);CHKERRQ(ierr);
  }

  {
     const PetscReal
       As[10][10] = {{0,0,0,0,0,0,0,0,0,0},
                     {RC(1.0)/RC(4.0),0,0,0,0,0,0,0,0,0},
                     {RC(1.0)/RC(4.0),0,0,0,0,0,0,0,0,0},
                     {RC(1.0)/RC(2.0),0,0,0,0,0,0,0,0,0},
                     {RC(1.0)/RC(2.0),0,0,0,0,0,0,0,0,0},
                     {RC(-1.0)/RC(6.0),0,0,0,RC(2.0)/RC(3.0),0,0,0,0,0},
                     {RC(1.0)/RC(12.0),0,0,0,RC(1.0)/RC(6.0),RC(1.0)/RC(2.0),0,0,0,0},
                     {RC(1.0)/RC(12.0),0,0,0,RC(1.0)/RC(6.0),RC(1.0)/RC(2.0),0,0,0,0},
                     {RC(1.0)/RC(3.0),0,0,0,RC(-1.0)/RC(3.0),RC(1.0),0,0,0,0},
                     {RC(1.0)/RC(3.0),0,0,0,RC(-1.0)/RC(3.0),RC(1.0),0,0,0,0}},
        A[10][10] = {{0,0,0,0,0,0,0,0,0,0},
                     {RC(1.0)/RC(4.0),0,0,0,0,0,0,0,0,0},
                     {RC(-1.0)/RC(12.0),RC(1.0)/RC(3.0),0,0,0,0,0,0,0,0},
                     {RC(1.0)/RC(6.0),RC(-1.0)/RC(6.0),RC(1.0)/RC(2.0),0,0,0,0,0,0,0},
                     {RC(1.0)/RC(12.0),RC(1.0)/RC(6.0),RC(1.0)/RC(6.0),RC(1.0)/RC(12.0),0,0,0,0,0,0},
                     {RC(1.0)/RC(12.0),RC(1.0)/RC(6.0),RC(1.0)/RC(6.0),RC(1.0)/RC(12.0),0,0,0,0,0,0},
                     {RC(1.0)/RC(12.0),RC(1.0)/RC(6.0),RC(1.0)/RC(6.0),RC(1.0)/RC(12.0),0,RC(1.0)/RC(4.0),0,0,0,0},
                     {RC(1.0)/RC(12.0),RC(1.0)/RC(6.0),RC(1.0)/RC(6.0),RC(1.0)/RC(12.0),0,RC(-1.0)/RC(12.0),RC(1.0)/RC(3.0),0,0,0},
                     {RC(1.0)/RC(12.0),RC(1.0)/RC(6.0),RC(1.0)/RC(6.0),RC(1.0)/RC(12.0),0,RC(1.0)/RC(6.0),RC(-1.0)/RC(6.0),RC(1.0)/RC(2.0),0,0},
                     {RC(1.0)/RC(12.0),RC(1.0)/RC(6.0),RC(1.0)/RC(6.0),RC(1.0)/RC(12.0),0,RC(1.0)/RC(12.0),RC(1.0)/RC(6.0),RC(1.0)/RC(6.0),RC(1.0)/RC(12.0),0}},
          bs[10] = {RC(1.0)/RC(6.0),0,0,0,RC(1.0)/RC(3.0),RC(1.0)/RC(3.0),0,0,0,RC(1.0)/RC(6.0)},
           b[10] = {RC(1.0)/RC(12.0),RC(1.0)/RC(6.0),RC(1.0)/RC(6.0),RC(1.0)/RC(12.0),0,RC(1.0)/RC(12.0),RC(1.0)/RC(6.0),RC(1.0)/RC(6.0),RC(1.0)/RC(12.0),0};
            ierr = TSPRKRegister(TSPRKRFSMR3,3,10,&As[0][0],bs,NULL,&A[0][0],b,NULL);CHKERRQ(ierr);
  }
#undef RC
  PetscFunctionReturn(0);
}

/*@C
   TSPRKRegisterDestroy - Frees the list of schemes that were registered by TSPRKRegister().

   Not Collective

   Level: advanced

.keywords: TSPRK, register, destroy
.seealso: TSPRKRegister(), TSPRKRegisterAll()
@*/
PetscErrorCode TSPRKRegisterDestroy(void)
{
  PetscErrorCode ierr;
  PRKTableauLink link;

  PetscFunctionBegin;
  while ((link = PRKTableauList)) {
    PRKTableau t = &link->tab;
    PRKTableauList = link->next;
    ierr = PetscFree3(t->Af,t->bf,t->cf);  CHKERRQ(ierr);
    ierr = PetscFree3(t->As,t->bs,t->cs);  CHKERRQ(ierr);
    ierr = PetscFree (t->name);            CHKERRQ(ierr);
    ierr = PetscFree (link);               CHKERRQ(ierr);
  }
  TSPRKRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  TSPRKInitializePackage - This function initializes everything in the TSPRK package. It is called
  from PetscDLLibraryRegister() when using dynamic libraries, and on the first call to TSCreate_PRK()
  when using static libraries.

  Level: developer

.keywords: TS, TSPRK, initialize, package
.seealso: PetscInitialize()
@*/
PetscErrorCode TSPRKInitializePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (TSPRKPackageInitialized) PetscFunctionReturn(0);
  TSPRKPackageInitialized = PETSC_TRUE;
  ierr = TSPRKRegisterAll();CHKERRQ(ierr);
  ierr = PetscRegisterFinalize(TSPRKFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  TSRKFinalizePackage - This function destroys everything in the TSPRK package. It is
  called from PetscFinalize().

  Level: developer

.keywords: Petsc, destroy, package
.seealso: PetscFinalize()
@*/
PetscErrorCode TSPRKFinalizePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TSPRKPackageInitialized = PETSC_FALSE;
  ierr = TSPRKRegisterDestroy();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   TSPRKRegister - register a PRK scheme by providing the entries in the Butcher tableau

   Not Collective, but the same schemes should be registered on all processes on which they will be used

   Input Parameters:
+  name - identifier for method
.  order - approximation order of method
.  s  - number of stages, this is the dimension of the matrices below
.  Af - stage coefficients for fast components(dimension s*s, row-major)
.  bf - step completion table for fast components(dimension s)
.  cf - abscissa for fast components(dimension s)
.  As - stage coefficients for slow components(dimension s*s, row-major)
.  bs - step completion table for slow components(dimension s)
-  cs - abscissa for slow components(dimension s)

   Notes:
   Several PRK methods are provided, this function is only needed to create new methods.

   Level: advanced

.keywords: TS, register

.seealso: TSPRK
@*/
PetscErrorCode TSPRKRegister(TSPRKType name,PetscInt order,PetscInt s,
                            const PetscReal As[],const PetscReal bs[],const PetscReal cs[],
                            const PetscReal Af[],const PetscReal bf[],const PetscReal cf[])
{
  PetscErrorCode   ierr;
  PRKTableauLink   link;
  PRKTableau       t;
  PetscInt         i,j;

  PetscFunctionBegin;
  PetscValidCharPointer(name,1);
  PetscValidRealPointer(Af,7);
  if (bf) PetscValidRealPointer(bf,8);
  if (cf) PetscValidRealPointer(cf,9);
  PetscValidRealPointer(As,4);
  if (bs) PetscValidRealPointer(bs,5);
  if (cs) PetscValidRealPointer(cs,6);

  ierr = PetscNew(&link);CHKERRQ(ierr);
  t = &link->tab;

  ierr = PetscStrallocpy(name,&t->name);CHKERRQ(ierr);
  t->order = order;
  t->s = s;
  ierr = PetscMalloc3(s*s,&t->Af,s,&t->bf,s,&t->cf);CHKERRQ(ierr);
  ierr = PetscMemcpy(t->Af,Af,s*s*sizeof(Af[0]));CHKERRQ(ierr);
  if (bf)  { ierr = PetscMemcpy(t->bf,bf,s*sizeof(bf[0]));CHKERRQ(ierr); }
  else for (i=0; i<s; i++) t->bf[i] = Af[(s-1)*s+i];
  if (cf)  { ierr = PetscMemcpy(t->cf,cf,s*sizeof(cf[0]));CHKERRQ(ierr); }
  else for (i=0; i<s; i++) for (j=0,t->cf[i]=0; j<s; j++) t->cf[i] += Af[i*s+j];
  ierr = PetscMalloc3(s*s,&t->As,s,&t->bs,s,&t->cs);CHKERRQ(ierr);
  ierr = PetscMemcpy(t->As,As,s*s*sizeof(As[0]));CHKERRQ(ierr);
  if (bs)  { ierr = PetscMemcpy(t->bs,bs,s*sizeof(bs[0]));CHKERRQ(ierr); }
  else for (i=0; i<s; i++) t->bs[i] = As[(s-1)*s+i];
  if (cs)  { ierr = PetscMemcpy(t->cs,cs,s*sizeof(cs[0]));CHKERRQ(ierr); }
  else for (i=0; i<s; i++) for (j=0,t->cs[i]=0; j<s; j++) t->cs[i] += As[i*s+j];

  link->next = PRKTableauList;
  PRKTableauList = link;
  PetscFunctionReturn(0);
}

/*
 This if for combined RHS PRK
 The step completion formula is

 x1 = x0 + h b^T YdotRHS

*/
static PetscErrorCode TSEvaluateStep_PRK(TS ts,PetscInt order,Vec X,PetscBool *done)
{
  TS_PRK          *prk = (TS_PRK*)ts->data;
  PRKTableau      tab  = prk->tableau;
  PetscScalar     *wf  = prk->work_fast, *ws = prk->work_slow;
  PetscReal       h    = ts->time_step;
  PetscInt        s    = tab->s,j,len_slow,len_fast;             /* len_slow: the number of slow components, len_fast: the number of fast components */
  Vec             Xslow_combine,Xfast_combine;                   /* work vector for X by slow tableau and fast tableau repectively                   */
  PetscScalar     *xslowcombine_ptr,*xfastcombine_ptr,*x_ptr;    /* location to put the pointer to Xslow_combine, Xfast_combine and X respectively   */
  const PetscInt  *is_slow,*is_fast;                             /* index for slow and fast components respectively                                  */
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  /* update Xslow_combine by slow tableau and Xfast_combine by fast tableau */
  ierr = VecCopy(ts->vec_sol,X);CHKERRQ(ierr);
  for (j=0; j<s; j++) wf[j] = h*tab->bf[j];
  for (j=0; j<s; j++) ws[j] = h*tab->bs[j];
  ierr = VecDuplicate(X,&Xslow_combine);CHKERRQ(ierr);
  ierr = VecCopy(X,Xslow_combine);CHKERRQ(ierr);
  ierr = VecDuplicate(X,&Xfast_combine);CHKERRQ(ierr);
  ierr = VecCopy(X,Xfast_combine);CHKERRQ(ierr);
  ierr = VecMAXPY(Xslow_combine,s,ws,prk->YdotRHSslow_combine);CHKERRQ(ierr);
  ierr = VecMAXPY(Xfast_combine,s,wf,prk->YdotRHSfast_combine);CHKERRQ(ierr);
  /* combine slow coponents of Xslow_combine and fast components of Xfast_combine to update X */
  ierr = ISGetSize(ts->iss,&len_slow);CHKERRQ(ierr);
  ierr = ISGetSize(ts->isf,&len_fast);CHKERRQ(ierr);
  ierr = ISGetIndices(ts->iss,&is_slow);CHKERRQ(ierr);
  ierr = ISGetIndices(ts->isf,&is_fast);CHKERRQ(ierr);
  ierr = ISRestoreIndices(ts->isf,&is_fast);CHKERRQ(ierr);
  ierr = ISRestoreIndices(ts->iss,&is_slow);CHKERRQ(ierr);
  ierr = VecGetArray(Xslow_combine,&xslowcombine_ptr);CHKERRQ(ierr);
  ierr = VecGetArray(Xfast_combine,&xfastcombine_ptr);CHKERRQ(ierr);
  ierr = VecGetArray(X,&x_ptr);CHKERRQ(ierr);
  for (j=0; j<len_slow;j++) x_ptr[is_slow[j]] = xslowcombine_ptr[is_slow[j]];
  for (j=0; j<len_fast;j++) x_ptr[is_fast[j]] = xfastcombine_ptr[is_fast[j]];
  ierr = VecRestoreArray(X,&x_ptr);CHKERRQ(ierr);
  ierr = VecRestoreArray(Xfast_combine,&xfastcombine_ptr);CHKERRQ(ierr);
  ierr = VecRestoreArray(Xslow_combine,&xslowcombine_ptr);CHKERRQ(ierr);
  /* free memory of work vectors */
  ierr = VecDestroy(&Xslow_combine);CHKERRQ(ierr);
  ierr = VecDestroy(&Xfast_combine);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSStep_PRK(TS ts)
{
  TS_PRK            *prk = (TS_PRK*)ts->data;
  Vec               *Y   = prk->Y,*YdotRHSfast_combine = prk->YdotRHSfast_combine;
  Vec               *YdotRHSslow_combine = prk->YdotRHSslow_combine;
  Vec               Yslow_combine,Yfast_combine;                                  /* work vectors for Y by slow and fast tableau respectively                       */
  PRKTableau        tab = prk->tableau;
  const PetscInt    s   = tab->s,*is_slow,*is_fast;                               /* is_slow: index of slow components, is_fast: index of fast components           */
  const PetscReal   *Af = tab->Af,*cf = tab->cf,*As = tab->As,*cs = tab->cs;
  PetscScalar       *wf = prk->work_fast, *ws = prk->work_slow;
  PetscScalar       *y_ptr,*yslowcombine_ptr,*yfastcombine_ptr;                   /* location to put the pointer to Y, Yslow_combine, Yfast_combine respectively     */
  PetscInt          i,j,len_slow,len_fast;                                        /* len_slow: the number of slow coponents, len_fast: the number of fast components */
  PetscReal         next_time_step = ts->time_step,t = ts->ptime,h = ts->time_step;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecDuplicate(ts->vec_sol,&Yslow_combine);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&Yfast_combine);CHKERRQ(ierr);
  for (i=0; i<s; i++) {
    prk->stage_time = t + h*cf[i];
    ierr = TSPreStage(ts,prk->stage_time);CHKERRQ(ierr);
    ierr = VecCopy(ts->vec_sol,Y[i]);CHKERRQ(ierr);
    /* update the satge value for all components by fast and tableau respectively */
    for (j=0; j<i; j++) wf[j] = h*Af[i*s+j];
    for (j=0; j<i; j++) ws[j] = h*As[i*s+j];
    ierr = VecCopy(ts->vec_sol,Yslow_combine);CHKERRQ(ierr);
    ierr = VecCopy(ts->vec_sol,Yfast_combine);CHKERRQ(ierr);
    ierr = VecMAXPY(Yslow_combine,i,ws,YdotRHSslow_combine);CHKERRQ(ierr);
    ierr = VecMAXPY(Yfast_combine,i,wf,YdotRHSfast_combine);CHKERRQ(ierr);
    /* combine the value of slow components in Yslow_combine and the value of fast components in Yfast_combine to update Y[i] */
    ierr = ISGetSize(ts->iss,&len_slow);CHKERRQ(ierr);
    ierr = ISGetSize(ts->isf,&len_fast);CHKERRQ(ierr);
    ierr = ISGetIndices(ts->iss,&is_slow);CHKERRQ(ierr);
    ierr = ISGetIndices(ts->isf,&is_fast);CHKERRQ(ierr);
    ierr = ISRestoreIndices(ts->isf,&is_fast);CHKERRQ(ierr);
    ierr = ISRestoreIndices(ts->iss,&is_slow);CHKERRQ(ierr);
    ierr = VecGetArray(Yslow_combine,&yslowcombine_ptr);CHKERRQ(ierr);
    ierr = VecGetArray(Yfast_combine,&yfastcombine_ptr);CHKERRQ(ierr);
    ierr = VecGetArray(Y[i],&y_ptr);CHKERRQ(ierr);
    for (j=0; j<len_slow;j++) y_ptr[is_slow[j]] = yslowcombine_ptr[is_slow[j]];
    for (j=0; j<len_fast;j++) y_ptr[is_fast[j]] = yfastcombine_ptr[is_fast[j]];
    ierr = VecRestoreArray(Y[i],&y_ptr);CHKERRQ(ierr);
    ierr = VecRestoreArray(Yfast_combine,&yfastcombine_ptr);CHKERRQ(ierr);
    ierr = VecRestoreArray(Yslow_combine,&yslowcombine_ptr);CHKERRQ(ierr);
    ierr = TSPostStage(ts,prk->stage_time,i,Y); CHKERRQ(ierr);
    /* compute the stage RHS by fast and slow tableau respectively */
    ierr = TSComputeRHSFunction(ts,t+h*cf[i],Y[i],YdotRHSfast_combine[i]);CHKERRQ(ierr);
    ierr = TSComputeRHSFunction(ts,t+h*cs[i],Y[i],YdotRHSslow_combine[i]);CHKERRQ(ierr);
  }
  ierr = TSEvaluateStep(ts,tab->order,ts->vec_sol,NULL);CHKERRQ(ierr);
  ts->ptime += ts->time_step;
  ts->time_step = next_time_step;
  /* free memory of work vectors*/
  ierr = VecDestroy(&Yslow_combine);CHKERRQ(ierr);
  ierr = VecDestroy(&Yfast_combine);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
 This if for partitioned RHS PRK
 The step completion formula is

 x1 = x0 + h b^T YdotRHS

*/
static PetscErrorCode TSEvaluateStep_PRKPARTITIONED(TS ts,PetscInt order,Vec X,PetscBool *done)
{
  TS_PRK          *prk = (TS_PRK*)ts->data;
  PRKTableau      tab  = prk->tableau;
  Vec             Xslow,Xfast;                                  /* subvectors for slow and fast componets in X respectively */
  PetscScalar     *wf = prk->work_fast,*ws = prk->work_slow;
  PetscReal       h = ts->time_step;
  PetscInt        s = tab->s,j;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = VecCopy(ts->vec_sol,X);CHKERRQ(ierr);
  for (j=0; j<s; j++) wf[j] = h*tab->bf[j];
  for (j=0; j<s; j++) ws[j] = h*tab->bs[j];
  ierr = VecGetSubVector(X,ts->iss,&Xslow);CHKERRQ(ierr);
  ierr = VecGetSubVector(X,ts->isf,&Xfast);CHKERRQ(ierr);
  ierr = VecMAXPY(Xslow,s,ws,prk->YdotRHS_slow);CHKERRQ(ierr);
  ierr = VecMAXPY(Xfast,s,wf,prk->YdotRHS_fast);CHKERRQ(ierr);
  ierr = VecRestoreSubVector(X,ts->iss,&Xfast);CHKERRQ(ierr);
  ierr = VecRestoreSubVector(X,ts->isf,&Xslow);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSStep_PRKPARTITIONED(TS ts)
{
  TS_PRK            *prk = (TS_PRK*)ts->data;
  PRKTableau        tab  = prk->tableau;
  Vec               *Y = prk->Y,*YdotRHS_fast = prk->YdotRHS_fast, *YdotRHS_slow = prk->YdotRHS_slow;
  Vec               Yslow,Yfast;           /* subvectors for slow and fast components in Y[i] respectively */
  const PetscInt    s = tab->s;
  const PetscReal   *Af = tab->Af,*cf = tab->cf,*As = tab->As,*cs = tab->cs;
  PetscScalar       *wf = prk->work_fast, *ws = prk->work_slow;
  PetscInt          i,j;
  PetscReal         next_time_step = ts->time_step,t = ts->ptime,h = ts->time_step;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  for (i=0; i<s; i++) {
    prk->stage_time = t + h*cf[i];
    ierr = TSPreStage(ts,prk->stage_time);CHKERRQ(ierr);
    /* calculate the stage value for fast and slow components respectively */
    ierr = VecCopy(ts->vec_sol,Y[i]);CHKERRQ(ierr);
    for (j=0; j<i; j++) wf[j] = h*Af[i*s+j];
    for (j=0; j<i; j++) ws[j] = h*As[i*s+j];
    ierr = VecGetSubVector(Y[i],ts->iss,&Yslow);CHKERRQ(ierr);
    ierr = VecGetSubVector(Y[i],ts->isf,&Yfast);CHKERRQ(ierr);
    ierr = VecMAXPY(Yslow,i,ws,YdotRHS_slow);CHKERRQ(ierr);
    ierr = VecMAXPY(Yfast,i,wf,YdotRHS_fast);CHKERRQ(ierr);
    ierr = VecRestoreSubVector(Y[i],ts->isf,&Yfast);CHKERRQ(ierr);
    ierr = VecRestoreSubVector(Y[i],ts->iss,&Yslow);CHKERRQ(ierr);
    ierr = TSPostStage(ts,prk->stage_time,i,Y); CHKERRQ(ierr);
    /* calculate the stage RHS for slow and fast components respectively */
    ierr = TSComputeRHSFunctionslow(ts,t+h*cs[i],Y[i],YdotRHS_slow[i]);CHKERRQ(ierr);
    ierr = TSComputeRHSFunctionfast(ts,t+h*cf[i],Y[i],YdotRHS_fast[i]);CHKERRQ(ierr);
  }
  ierr = TSEvaluateStep(ts,tab->order,ts->vec_sol,NULL);CHKERRQ(ierr);
  ts->ptime += ts->time_step;
  ts->time_step = next_time_step;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSPRKTableauReset(TS ts)
{
  TS_PRK          *prk = (TS_PRK*)ts->data;
  PRKTableau       tab = prk->tableau;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (!tab) PetscFunctionReturn(0);
  ierr = PetscFree(prk->work_fast);CHKERRQ(ierr);
  ierr = PetscFree(prk->work_slow);CHKERRQ(ierr);
  ierr = VecDestroyVecs(tab->s,&prk->Y);CHKERRQ(ierr);
  ierr = VecDestroyVecs(tab->s,&prk->YdotRHSfast_combine);CHKERRQ(ierr);
  ierr = VecDestroyVecs(tab->s,&prk->YdotRHSslow_combine);CHKERRQ(ierr);
  ierr = VecDestroyVecs(tab->s,&prk->YdotRHS_fast);CHKERRQ(ierr);
  ierr = VecDestroyVecs(tab->s,&prk->YdotRHS_slow);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSReset_PRK(TS ts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSPRKTableauReset(ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCoarsenHook_TSPRK(DM fine,DM coarse,void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMRestrictHook_TSPRK(DM fine,Mat restrct,Vec rscale,Mat inject,DM coarse,void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}


static PetscErrorCode DMSubDomainHook_TSPRK(DM dm,DM subdm,void *ctx)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMSubDomainRestrictHook_TSPRK(DM dm,VecScatter gscat,VecScatter lscat,DM subdm,void *ctx)
{

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSPRKTableauSetUp(TS ts)
{
  TS_PRK         *prk  = (TS_PRK*)ts->data;
  PRKTableau      tab = prk->tableau;
  Vec             YdotRHS_fast,YdotRHS_slow;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc1(tab->s,&prk->work_fast);CHKERRQ(ierr);
  ierr = PetscMalloc1(tab->s,&prk->work_slow);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(ts->vec_sol,tab->s,&prk->Y);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(ts->vec_sol,tab->s,&prk->YdotRHSfast_combine);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(ts->vec_sol,tab->s,&prk->YdotRHSslow_combine);CHKERRQ(ierr);
  ierr = VecGetSubVector(ts->vec_sol,ts->iss,&YdotRHS_slow);CHKERRQ(ierr);
  ierr = VecGetSubVector(ts->vec_sol,ts->isf,&YdotRHS_fast);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(YdotRHS_slow,tab->s,&prk->YdotRHS_slow);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(YdotRHS_fast,tab->s,&prk->YdotRHS_fast);CHKERRQ(ierr);
  ierr = VecRestoreSubVector(ts->vec_sol,ts->iss,&YdotRHS_slow);CHKERRQ(ierr);
  ierr = VecRestoreSubVector(ts->vec_sol,ts->isf,&YdotRHS_fast);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


static PetscErrorCode TSSetUp_PRK(TS ts)
{
  PetscErrorCode ierr;
  DM             dm;

  PetscFunctionBegin;
  ierr = TSCheckImplicitTerm(ts);CHKERRQ(ierr);
  ierr = TSPRKTableauSetUp(ts);CHKERRQ(ierr);
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMCoarsenHookAdd(dm,DMCoarsenHook_TSPRK,DMRestrictHook_TSPRK,ts);CHKERRQ(ierr);
  ierr = DMSubDomainHookAdd(dm,DMSubDomainHook_TSPRK,DMSubDomainRestrictHook_TSPRK,ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* construct a database to chose combined rhs mutirate prk method or partitioned rhs prk method */
const char *const TSPRKMultirateTypes[] = {"COMBINED","PARTITIONED","TSPRKMultirateType","PRKM_",0};

static PetscErrorCode TSSetFromOptions_PRK(PetscOptionItems *PetscOptionsObject,TS ts)
{
  TS_PRK         *prk = (TS_PRK*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"PRK ODE solver options");CHKERRQ(ierr);
  {
    PRKTableauLink  link;
    PetscInt        count,choice;
    PetscBool       flg;
    const char      **namelist;
    PetscInt        prkmtype = 0;
    for (link=PRKTableauList,count=0; link; link=link->next,count++) ;
    ierr = PetscMalloc1(count,(char***)&namelist);CHKERRQ(ierr);
    for (link=PRKTableauList,count=0; link; link=link->next,count++) namelist[count] = link->tab.name;
    ierr = PetscOptionsEList("-ts_prk_type","Family of PRK method","TSPRKSetType",(const char*const*)namelist,count,prk->tableau->name,&choice,&flg);CHKERRQ(ierr);
    if (flg) {ierr = TSPRKSetType(ts,namelist[choice]);CHKERRQ(ierr);}
    ierr = PetscFree(namelist);CHKERRQ(ierr);
    ierr = PetscOptionsEList("-ts_prk_multirate_type","Use Combined RHS Multirate or Partioned RHS Multirat PRK method","TSPRKSetMultirateType",TSPRKMultirateTypes,2,TSPRKMultirateTypes[0],&prkmtype,&flg);CHKERRQ(ierr);
     if (flg) {ierr = TSPRKSetMultirateType(ts,prkmtype);CHKERRQ(ierr);}
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TSView_PRK(TS ts,PetscViewer viewer)
{
  TS_PRK          *prk = (TS_PRK*)ts->data;
  PetscBool        iascii;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    PRKTableau tab  = prk->tableau;
    TSPRKType  prktype;
    char       fbuf[512];
    char       sbuf[512];
    ierr = TSPRKGetType(ts,&prktype);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  PRK type %s\n",prktype);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  Order: %D\n",tab->order);CHKERRQ(ierr);
    ierr = PetscFormatRealArray(fbuf,sizeof(fbuf),"% 8.6f",tab->s,tab->cf);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  Abscissa cf = %s\n",fbuf);CHKERRQ(ierr);
    ierr = PetscFormatRealArray(sbuf,sizeof(sbuf),"% 8.6f",tab->s,tab->cs);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  Abscissa cs = %s\n",sbuf);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSLoad_PRK(TS ts,PetscViewer viewer)
{
  PetscErrorCode ierr;
  TSAdapt        adapt;

  PetscFunctionBegin;
  ierr = TSGetAdapt(ts,&adapt);CHKERRQ(ierr);
  ierr = TSAdaptLoad(adapt,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  TSPRKSetType - Set the type of PRK scheme

  Logically collective

  Input Parameter:
+  ts - timestepping context
-  prktype - type of PRK-scheme

  Options Database:
.   -ts_prk_type - <pm2,pm3,p2,p3>

  Level: intermediate

.seealso: TSPRKGetType(), TSPRK, TSPRKType, TSPRKM2FULL
@*/
PetscErrorCode TSPRKSetType(TS ts,TSPRKType prktype)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidCharPointer(prktype,2);
  ierr = PetscTryMethod(ts,"TSPRKSetType_C",(TS,TSPRKType),(ts,prktype));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  TSPRKGetType - Get the type of PRK scheme

  Logically collective

  Input Parameter:
.  ts - timestepping context

  Output Parameter:
.  prktype - type of PRK-scheme

  Level: intermediate

.seealso: TSPRKGetType()
@*/
PetscErrorCode TSPRKGetType(TS ts,TSPRKType *prktype)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = PetscUseMethod(ts,"TSPRKGetType_C",(TS,TSPRKType*),(ts,prktype));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  TSPRKSetMultirateType - Set the type of PRK Multirate scheme

  Logically collective

  Input Parameter:
+  ts - timestepping context
-  prkmtype - type of PRKM-scheme

  Options Database:
.   -ts_prk_multirate_type - <combined,partitioned>

  Level: intermediate
@*/
PetscErrorCode TSPRKSetMultirateType(TS ts, TSPRKMultirateType prkmtype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  switch(prkmtype){
    case PRKM_COMBINED:
      break;
    case PRKM_PARTITIONED:
      ts->ops->step           = TSStep_PRKPARTITIONED;
      ts->ops->evaluatestep   = TSEvaluateStep_PRKPARTITIONED;
      break;
    default :
      SETERRQ1(PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown type '%s'",prkmtype);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode TSPRKGetType_PRK(TS ts,TSPRKType *prktype)
{
  TS_PRK *prk = (TS_PRK*)ts->data;

  PetscFunctionBegin;
  *prktype = prk->tableau->name;
  PetscFunctionReturn(0);
}
static PetscErrorCode TSPRKSetType_PRK(TS ts,TSPRKType prktype)
{
  TS_PRK          *prk = (TS_PRK*)ts->data;
  PetscErrorCode   ierr;
  PetscBool        match;
  PRKTableauLink   link;

  PetscFunctionBegin;
  if (prk->tableau) {
    ierr = PetscStrcmp(prk->tableau->name,prktype,&match);CHKERRQ(ierr);
    if (match) PetscFunctionReturn(0);
  }
  for (link = PRKTableauList; link; link=link->next) {
    ierr = PetscStrcmp(link->tab.name,prktype,&match);CHKERRQ(ierr);
    if (match) {
      if (ts->setupcalled) {ierr = TSPRKTableauReset(ts);CHKERRQ(ierr);}
      prk->tableau = &link->tab;
      if (ts->setupcalled) {ierr = TSPRKTableauSetUp(ts);CHKERRQ(ierr);}
      PetscFunctionReturn(0);
    }
  }
  SETERRQ1(PetscObjectComm((PetscObject)ts),PETSC_ERR_ARG_UNKNOWN_TYPE,"Could not find '%s'",prktype);
  PetscFunctionReturn(0);
}

static PetscErrorCode  TSGetStages_PRK(TS ts,PetscInt *ns,Vec **Y)
{
  TS_PRK *prk = (TS_PRK*)ts->data;

  PetscFunctionBegin;
  *ns = prk->tableau->s;
  if (Y) *Y = prk->Y;
  PetscFunctionReturn(0);
}

static PetscErrorCode TSDestroy_PRK(TS ts)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TSReset_PRK(ts);CHKERRQ(ierr);
  if (ts->dm) {
    ierr = DMCoarsenHookRemove(ts->dm,DMCoarsenHook_TSPRK,DMRestrictHook_TSPRK,ts);CHKERRQ(ierr);
    ierr = DMSubDomainHookRemove(ts->dm,DMSubDomainHook_TSPRK,DMSubDomainRestrictHook_TSPRK,ts);CHKERRQ(ierr);
  }
  ierr = PetscFree(ts->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSPRKGetType_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSPRKSetType_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSPRKSetMultirateType_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
      TSPRK - ODE solver using Partitioned Runge-Kutta schemes

  The user should provide the right hand side of the equation
  using TSSetRHSFunction().

  Notes:
  The default is TSPRKM2, it can be changed with TSRKSetType() or -ts_prk_type

  Level: beginner

.seealso:  TSCreate(), TS, TSSetType(), TSPRKSetType(), TSPRKGetType(), TSPRKType, TSPRKRegister(), TSPRKSetMultirateType()
           TSPRKM2, TSPRKM3, TSPRKRFSMR3, TSPRKRFSMR2

M*/
PETSC_EXTERN PetscErrorCode TSCreate_PRK(TS ts)
{
  TS_PRK          *prk;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = TSPRKInitializePackage();CHKERRQ(ierr);

  ts->ops->reset          = TSReset_PRK;
  ts->ops->destroy        = TSDestroy_PRK;
  ts->ops->view           = TSView_PRK;
  ts->ops->load           = TSLoad_PRK;
  ts->ops->setup          = TSSetUp_PRK;
  ts->ops->step           = TSStep_PRK;
  ts->ops->evaluatestep   = TSEvaluateStep_PRK;
  ts->ops->setfromoptions = TSSetFromOptions_PRK;
  ts->ops->getstages      = TSGetStages_PRK;


  ierr = PetscNewLog(ts,&prk);CHKERRQ(ierr);
  ts->data = (void*)prk;

  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSPRKGetType_C",TSPRKGetType_PRK);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSPRKSetType_C",TSPRKSetType_PRK);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ts,"TSPRKSetMultirateType_C",TSPRKSetMultirateType);CHKERRQ(ierr);

  ierr = TSPRKSetType(ts,TSPRKDefault);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
