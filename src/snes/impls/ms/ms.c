#include <petsc/private/snesimpl.h>   /*I "petscsnes.h" I*/

static SNESMSType SNESMSDefault = SNESMSM62;
static PetscBool  SNESMSRegisterAllCalled;
static PetscBool  SNESMSPackageInitialized;

typedef struct _SNESMSTableau *SNESMSTableau;
struct _SNESMSTableau {
  char      *name;
  PetscInt  nstages;            /* Number of stages */
  PetscInt  nregisters;         /* Number of registers */
  PetscReal stability;          /* Scaled stability region */
  PetscReal *gamma;             /* Coefficients of 3S* method */
  PetscReal *delta;             /* Coefficients of 3S* method */
  PetscReal *betasub;           /* Subdiagonal of beta in Shu-Osher form */
};

typedef struct _SNESMSTableauLink *SNESMSTableauLink;
struct _SNESMSTableauLink {
  struct _SNESMSTableau tab;
  SNESMSTableauLink     next;
};
static SNESMSTableauLink SNESMSTableauList;

typedef struct {
  SNESMSTableau tableau;        /* Tableau in low-storage form */
  PetscReal     damping;        /* Damping parameter, like length of (pseudo) time step */
  PetscBool     norms;          /* Compute norms, usually only for monitoring purposes */
} SNES_MS;

/*@C
  SNESMSRegisterAll - Registers all of the multi-stage methods in SNESMS

  Not Collective, but should be called by all processes which will need the schemes to be registered

  Level: advanced

.seealso:  SNESMSRegisterDestroy()
@*/
PetscErrorCode SNESMSRegisterAll(void)
{
  PetscFunctionBegin;
  if (SNESMSRegisterAllCalled) PetscFunctionReturn(0);
  SNESMSRegisterAllCalled = PETSC_TRUE;

  {
    PetscReal alpha[1] = {1.0};
    CHKERRQ(SNESMSRegister(SNESMSEULER,1,1,1.0,NULL,NULL,alpha));
  }

  {
    PetscReal gamma[3][6] = {
      {0.0000000000000000E+00, -7.0304722367110606E-01, -1.9836719667506464E-01, -1.6023843981863788E+00,  9.4483822882855284E-02, -1.4204296130641869E-01},
      {1.0000000000000000E+00,  1.1111025767083920E+00,  5.6150921583923230E-01,  7.4151723494934041E-01,  3.1714538168600587E-01,  4.6479276238548706E-01},
      {0.0000000000000000E+00,  0.0000000000000000E+00,  0.0000000000000000E+00,  6.7968174970583317E-01, -4.1755042846051737E-03, -1.9115668129923846E-01}
    };
    PetscReal delta[6]   = {1.0000000000000000E+00, 5.3275427433201750E-01, 6.0143544663985238E-01, 4.5874077053842177E-01, 2.7544386906104651E-01, 0.0000000000000000E+00};
    PetscReal betasub[6] = {8.4753115429481929E-01, 7.4018896368655618E-01, 6.5963574086583309E-03, 4.6747795645517759E-01, 1.3314545813643919E-01, 5.3260800028018784E-01};
    CHKERRQ(SNESMSRegister(SNESMSM62,6,3,1.0,&gamma[0][0],delta,betasub));
  }

  { /* Jameson (1983) */
    PetscReal alpha[4] = {0.25, 0.5, 0.55, 1.0};
    CHKERRQ(SNESMSRegister(SNESMSJAMESON83,4,1,1.0,NULL,NULL,alpha));
  }

  { /* Van Leer, Tai, and Powell (1989) 1 stage, order 1 */
    PetscReal alpha[1]  = {1.0};
    CHKERRQ(SNESMSRegister(SNESMSVLTP11,1,1,0.5,NULL,NULL,alpha));
  }
  { /* Van Leer, Tai, and Powell (1989) 2 stage, order 1 */
    PetscReal alpha[2] = {0.3333, 1.0};
    CHKERRQ(SNESMSRegister(SNESMSVLTP21,2,1,1.0,NULL,NULL,alpha));
  }
  { /* Van Leer, Tai, and Powell (1989) 3 stage, order 1 */
    PetscReal alpha[3] = {0.1481, 0.4000, 1.0};
    CHKERRQ(SNESMSRegister(SNESMSVLTP31,3,1,1.5,NULL,NULL,alpha));
  }
  { /* Van Leer, Tai, and Powell (1989) 4 stage, order 1 */
    PetscReal alpha[4] = {0.0833, 0.2069, 0.4265, 1.0};
    CHKERRQ(SNESMSRegister(SNESMSVLTP41,4,1,2.0,NULL,NULL,alpha));
  }
  { /* Van Leer, Tai, and Powell (1989) 5 stage, order 1 */
    PetscReal alpha[5] = {0.0533, 0.1263, 0.2375, 0.4414,1.0};
    CHKERRQ(SNESMSRegister(SNESMSVLTP51,5,1,2.5,NULL,NULL,alpha));
  }
  { /* Van Leer, Tai, and Powell (1989) 6 stage, order 1 */
    PetscReal alpha[6] = {0.0370, 0.0851, 0.1521, 0.2562, 0.4512, 1.0};
    CHKERRQ(SNESMSRegister(SNESMSVLTP61,6,1,3.0,NULL,NULL,alpha));
  }
  PetscFunctionReturn(0);
}

/*@C
   SNESMSRegisterDestroy - Frees the list of schemes that were registered by SNESMSRegister().

   Not Collective

   Level: advanced

.seealso: SNESMSRegister(), SNESMSRegisterAll()
@*/
PetscErrorCode SNESMSRegisterDestroy(void)
{
  SNESMSTableauLink link;

  PetscFunctionBegin;
  while ((link = SNESMSTableauList)) {
    SNESMSTableau t = &link->tab;
    SNESMSTableauList = link->next;

    CHKERRQ(PetscFree(t->name));
    CHKERRQ(PetscFree(t->gamma));
    CHKERRQ(PetscFree(t->delta));
    CHKERRQ(PetscFree(t->betasub));
    CHKERRQ(PetscFree(link));
  }
  SNESMSRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  SNESMSInitializePackage - This function initializes everything in the SNESMS package. It is called
  from SNESInitializePackage().

  Level: developer

.seealso: PetscInitialize()
@*/
PetscErrorCode SNESMSInitializePackage(void)
{
  PetscFunctionBegin;
  if (SNESMSPackageInitialized) PetscFunctionReturn(0);
  SNESMSPackageInitialized = PETSC_TRUE;

  CHKERRQ(SNESMSRegisterAll());
  CHKERRQ(PetscRegisterFinalize(SNESMSFinalizePackage));
  PetscFunctionReturn(0);
}

/*@C
  SNESMSFinalizePackage - This function destroys everything in the SNESMS package. It is
  called from PetscFinalize().

  Level: developer

.seealso: PetscFinalize()
@*/
PetscErrorCode SNESMSFinalizePackage(void)
{
  PetscFunctionBegin;
  SNESMSPackageInitialized = PETSC_FALSE;

  CHKERRQ(SNESMSRegisterDestroy());
  PetscFunctionReturn(0);
}

/*@C
   SNESMSRegister - register a multistage scheme

   Not Collective, but the same schemes should be registered on all processes on which they will be used

   Input Parameters:
+  name - identifier for method
.  nstages - number of stages
.  nregisters - number of registers used by low-storage implementation
.  stability - scaled stability region
.  gamma - coefficients, see Ketcheson's paper
.  delta - coefficients, see Ketcheson's paper
-  betasub - subdiagonal of Shu-Osher form

   Notes:
   The notation is described in Ketcheson (2010) Runge-Kutta methods with minimum storage implementations.

   Many multistage schemes are of the form
   $ X_0 = X^{(n)}
   $ X_k = X_0 + \alpha_k * F(X_{k-1}), k = 1,\ldots,s
   $ X^{(n+1)} = X_s
   These methods can be registered with
.vb
   SNESMSRegister("name",s,1,stability,NULL,NULL,alpha);
.ve

   Level: advanced

.seealso: SNESMS
@*/
PetscErrorCode SNESMSRegister(SNESMSType name,PetscInt nstages,PetscInt nregisters,PetscReal stability,const PetscReal gamma[],const PetscReal delta[],const PetscReal betasub[])
{
  SNESMSTableauLink link;
  SNESMSTableau     t;

  PetscFunctionBegin;
  PetscValidCharPointer(name,1);
  PetscCheckFalse(nstages < 1,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Must have at least one stage");
  if (gamma || delta) {
    PetscCheckFalse(nregisters != 3,PETSC_COMM_SELF,PETSC_ERR_SUP,"Only support for methods written in 3-register form");
    PetscValidRealPointer(gamma,5);
    PetscValidRealPointer(delta,6);
  } else {
    PetscCheckFalse(nregisters != 1,PETSC_COMM_SELF,PETSC_ERR_SUP,"Only support for methods written in 1-register form");
  }
  PetscValidRealPointer(betasub,7);

  CHKERRQ(SNESMSInitializePackage());
  CHKERRQ(PetscNew(&link));
  t             = &link->tab;
  CHKERRQ(PetscStrallocpy(name,&t->name));
  t->nstages    = nstages;
  t->nregisters = nregisters;
  t->stability  = stability;

  if (gamma && delta) {
    CHKERRQ(PetscMalloc1(nstages*nregisters,&t->gamma));
    CHKERRQ(PetscMalloc1(nstages,&t->delta));
    CHKERRQ(PetscArraycpy(t->gamma,gamma,nstages*nregisters));
    CHKERRQ(PetscArraycpy(t->delta,delta,nstages));
  }
  CHKERRQ(PetscMalloc1(nstages,&t->betasub));
  CHKERRQ(PetscArraycpy(t->betasub,betasub,nstages));

  link->next        = SNESMSTableauList;
  SNESMSTableauList = link;
  PetscFunctionReturn(0);
}

/*
  X - initial state, updated in-place.
  F - residual, computed at the initial X on input
*/
static PetscErrorCode SNESMSStep_3Sstar(SNES snes,Vec X,Vec F)
{
  SNES_MS         *ms    = (SNES_MS*)snes->data;
  SNESMSTableau   t      = ms->tableau;
  const PetscReal *gamma = t->gamma,*delta = t->delta,*betasub = t->betasub;
  Vec             S1,S2,S3,Y;
  PetscInt        i,nstages = t->nstages;

  PetscFunctionBegin;
  Y    = snes->work[0];
  S1   = X;
  S2   = snes->work[1];
  S3   = snes->work[2];
  CHKERRQ(VecZeroEntries(S2));
  CHKERRQ(VecCopy(X,S3));
  for (i = 0; i < nstages; i++) {
    Vec         Ss[4];
    PetscScalar scoeff[4];

    Ss[0] = S1; Ss[1] = S2; Ss[2] = S3; Ss[3] = Y;

    scoeff[0] = gamma[0*nstages+i] - 1;
    scoeff[1] = gamma[1*nstages+i];
    scoeff[2] = gamma[2*nstages+i];
    scoeff[3] = -betasub[i]*ms->damping;

    CHKERRQ(VecAXPY(S2,delta[i],S1));
    if (i > 0) {
      CHKERRQ(SNESComputeFunction(snes,S1,F));
    }
    CHKERRQ(KSPSolve(snes->ksp,F,Y));
    CHKERRQ(VecMAXPY(S1,4,scoeff,Ss));
  }
  PetscFunctionReturn(0);
}

/*
  X - initial state, updated in-place.
  F - residual, computed at the initial X on input
*/
static PetscErrorCode SNESMSStep_Basic(SNES snes,Vec X,Vec F)
{
  SNES_MS         *ms    = (SNES_MS*)snes->data;
  SNESMSTableau   tab    = ms->tableau;
  const PetscReal *alpha = tab->betasub, h = ms->damping;
  PetscInt        i,nstages = tab->nstages;
  Vec             X0 = snes->work[0];

  PetscFunctionBegin;
  CHKERRQ(VecCopy(X,X0));
  for (i = 0; i < nstages; i++) {
    if (i > 0) {
      CHKERRQ(SNESComputeFunction(snes,X,F));
    }
    CHKERRQ(KSPSolve(snes->ksp,F,X));
    CHKERRQ(VecAYPX(X,-alpha[i]*h,X0));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESMSStep_Step(SNES snes,Vec X,Vec F)
{
  SNES_MS        *ms = (SNES_MS*)snes->data;
  SNESMSTableau  tab = ms->tableau;

  PetscFunctionBegin;
  if (tab->gamma && tab->delta) {
    CHKERRQ(SNESMSStep_3Sstar(snes,X,F));
  } else {
    CHKERRQ(SNESMSStep_Basic(snes,X,F));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESMSStep_Norms(SNES snes,PetscInt iter,Vec F)
{
  SNES_MS        *ms = (SNES_MS*)snes->data;
  PetscReal      fnorm;

  PetscFunctionBegin;
  if (ms->norms) {
    CHKERRQ(VecNorm(F,NORM_2,&fnorm)); /* fnorm <- ||F||  */
    SNESCheckFunctionNorm(snes,fnorm);
    /* Monitor convergence */
    CHKERRQ(PetscObjectSAWsTakeAccess((PetscObject)snes));
    snes->iter = iter;
    snes->norm = fnorm;
    CHKERRQ(PetscObjectSAWsGrantAccess((PetscObject)snes));
    CHKERRQ(SNESLogConvergenceHistory(snes,snes->norm,0));
    CHKERRQ(SNESMonitor(snes,snes->iter,snes->norm));
    /* Test for convergence */
    CHKERRQ((*snes->ops->converged)(snes,snes->iter,0.0,0.0,fnorm,&snes->reason,snes->cnvP));
  } else if (iter > 0) {
    CHKERRQ(PetscObjectSAWsTakeAccess((PetscObject)snes));
    snes->iter = iter;
    CHKERRQ(PetscObjectSAWsGrantAccess((PetscObject)snes));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESSolve_MS(SNES snes)
{
  SNES_MS        *ms = (SNES_MS*)snes->data;
  Vec            X   = snes->vec_sol,F = snes->vec_func;
  PetscInt       i;

  PetscFunctionBegin;
  PetscCheckFalse(snes->xl || snes->xu || snes->ops->computevariablebounds,PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_WRONGSTATE, "SNES solver %s does not support bounds", ((PetscObject)snes)->type_name);
  CHKERRQ(PetscCitationsRegister(SNESCitation,&SNEScite));

  snes->reason = SNES_CONVERGED_ITERATING;
  CHKERRQ(PetscObjectSAWsTakeAccess((PetscObject)snes));
  snes->iter   = 0;
  snes->norm   = 0;
  CHKERRQ(PetscObjectSAWsGrantAccess((PetscObject)snes));

  if (!snes->vec_func_init_set) {
    CHKERRQ(SNESComputeFunction(snes,X,F));
  } else snes->vec_func_init_set = PETSC_FALSE;

  CHKERRQ(SNESMSStep_Norms(snes,0,F));
  if (snes->reason) PetscFunctionReturn(0);

  for (i = 0; i < snes->max_its; i++) {

    /* Call general purpose update function */
    if (snes->ops->update) {
      CHKERRQ((*snes->ops->update)(snes,snes->iter));
    }

    if (i == 0 && snes->jacobian) {
      /* This method does not require a Jacobian, but it is usually preconditioned by PBJacobi */
      CHKERRQ(SNESComputeJacobian(snes,snes->vec_sol,snes->jacobian,snes->jacobian_pre));
      SNESCheckJacobianDomainerror(snes);
      CHKERRQ(KSPSetOperators(snes->ksp,snes->jacobian,snes->jacobian_pre));
    }

    CHKERRQ(SNESMSStep_Step(snes,X,F));

    if (i < snes->max_its-1 || ms->norms) {
      CHKERRQ(SNESComputeFunction(snes,X,F));
    }

    CHKERRQ(SNESMSStep_Norms(snes,i+1,F));
    if (snes->reason) PetscFunctionReturn(0);
  }
  if (!snes->reason) snes->reason = SNES_CONVERGED_ITS;
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESSetUp_MS(SNES snes)
{
  SNES_MS        *ms   = (SNES_MS*)snes->data;
  SNESMSTableau  tab   = ms->tableau;
  PetscInt       nwork = tab->nregisters;

  PetscFunctionBegin;
  CHKERRQ(SNESSetWorkVecs(snes,nwork));
  CHKERRQ(SNESSetUpMatrices(snes));
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESReset_MS(SNES snes)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESDestroy_MS(SNES snes)
{
  PetscFunctionBegin;
  CHKERRQ(SNESReset_MS(snes));
  CHKERRQ(PetscFree(snes->data));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)snes,"SNESMSGetType_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)snes,"SNESMSSetType_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)snes,"SNESMSGetDamping_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)snes,"SNESMSSetDamping_C",NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESView_MS(SNES snes,PetscViewer viewer)
{
  PetscBool      iascii;
  SNES_MS        *ms = (SNES_MS*)snes->data;
  SNESMSTableau  tab = ms->tableau;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  multi-stage method type: %s\n",tab->name));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESSetFromOptions_MS(PetscOptionItems *PetscOptionsObject,SNES snes)
{
  SNES_MS        *ms = (SNES_MS*)snes->data;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"SNES MS options"));
  {
    SNESMSTableauLink link;
    PetscInt          count,choice;
    PetscBool         flg;
    const char        **namelist;
    SNESMSType        mstype;
    PetscReal         damping;

    CHKERRQ(SNESMSGetType(snes,&mstype));
    for (link=SNESMSTableauList,count=0; link; link=link->next,count++) ;
    CHKERRQ(PetscMalloc1(count,(char***)&namelist));
    for (link=SNESMSTableauList,count=0; link; link=link->next,count++) namelist[count] = link->tab.name;
    CHKERRQ(PetscOptionsEList("-snes_ms_type","Multistage smoother type","SNESMSSetType",(const char*const*)namelist,count,mstype,&choice,&flg));
    if (flg) CHKERRQ(SNESMSSetType(snes,namelist[choice]));
    CHKERRQ(PetscFree(namelist));
    CHKERRQ(SNESMSGetDamping(snes,&damping));
    CHKERRQ(PetscOptionsReal("-snes_ms_damping","Damping for multistage method","SNESMSSetDamping",damping,&damping,&flg));
    if (flg) CHKERRQ(SNESMSSetDamping(snes,damping));
    CHKERRQ(PetscOptionsBool("-snes_ms_norms","Compute norms for monitoring","none",ms->norms,&ms->norms,NULL));
  }
  CHKERRQ(PetscOptionsTail());
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESMSGetType_MS(SNES snes,SNESMSType *mstype)
{
  SNES_MS        *ms = (SNES_MS*)snes->data;
  SNESMSTableau  tab = ms->tableau;

  PetscFunctionBegin;
  *mstype = tab->name;
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESMSSetType_MS(SNES snes,SNESMSType mstype)
{
  SNES_MS           *ms = (SNES_MS*)snes->data;
  SNESMSTableauLink link;
  PetscBool         match;

  PetscFunctionBegin;
  if (ms->tableau) {
    CHKERRQ(PetscStrcmp(ms->tableau->name,mstype,&match));
    if (match) PetscFunctionReturn(0);
  }
  for (link = SNESMSTableauList; link; link=link->next) {
    CHKERRQ(PetscStrcmp(link->tab.name,mstype,&match));
    if (match) {
      if (snes->setupcalled)  CHKERRQ(SNESReset_MS(snes));
      ms->tableau = &link->tab;
      if (snes->setupcalled)  CHKERRQ(SNESSetUp_MS(snes));
      PetscFunctionReturn(0);
    }
  }
  SETERRQ(PetscObjectComm((PetscObject)snes),PETSC_ERR_ARG_UNKNOWN_TYPE,"Could not find '%s'",mstype);
}

/*@C
  SNESMSGetType - Get the type of multistage smoother

  Not collective

  Input Parameter:
.  snes - nonlinear solver context

  Output Parameter:
.  mstype - type of multistage method

  Level: beginner

.seealso: SNESMSSetType(), SNESMSType, SNESMS
@*/
PetscErrorCode SNESMSGetType(SNES snes,SNESMSType *mstype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidPointer(mstype,2);
  CHKERRQ(PetscUseMethod(snes,"SNESMSGetType_C",(SNES,SNESMSType*),(snes,mstype)));
  PetscFunctionReturn(0);
}

/*@C
  SNESMSSetType - Set the type of multistage smoother

  Logically collective

  Input Parameters:
+  snes - nonlinear solver context
-  mstype - type of multistage method

  Level: beginner

.seealso: SNESMSGetType(), SNESMSType, SNESMS
@*/
PetscErrorCode SNESMSSetType(SNES snes,SNESMSType mstype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidCharPointer(mstype,2);
  CHKERRQ(PetscTryMethod(snes,"SNESMSSetType_C",(SNES,SNESMSType),(snes,mstype)));
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESMSGetDamping_MS(SNES snes,PetscReal *damping)
{
  SNES_MS        *ms = (SNES_MS*)snes->data;

  PetscFunctionBegin;
  *damping = ms->damping;
  PetscFunctionReturn(0);
}

static PetscErrorCode SNESMSSetDamping_MS(SNES snes,PetscReal damping)
{
  SNES_MS           *ms = (SNES_MS*)snes->data;

  PetscFunctionBegin;
  ms->damping = damping;
  PetscFunctionReturn(0);
}

/*@C
  SNESMSGetDamping - Get the damping parameter

  Not collective

  Input Parameter:
.  snes - nonlinear solver context

  Output Parameter:
.  damping - damping parameter

  Level: advanced

.seealso: SNESMSSetDamping(), SNESMS
@*/
PetscErrorCode SNESMSGetDamping(SNES snes,PetscReal *damping)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidRealPointer(damping,2);
  CHKERRQ(PetscUseMethod(snes,"SNESMSGetDamping_C",(SNES,PetscReal*),(snes,damping)));
  PetscFunctionReturn(0);
}

/*@C
  SNESMSSetDamping - Set the damping parameter

  Logically collective

  Input Parameters:
+  snes - nonlinear solver context
-  damping - damping parameter

  Level: advanced

.seealso: SNESMSGetDamping(), SNESMS
@*/
PetscErrorCode SNESMSSetDamping(SNES snes,PetscReal damping)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  PetscValidLogicalCollectiveReal(snes,damping,2);
  CHKERRQ(PetscTryMethod(snes,"SNESMSSetDamping_C",(SNES,PetscReal),(snes,damping)));
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------- */
/*MC
      SNESMS - multi-stage smoothers

      Options Database:

+     -snes_ms_type - type of multi-stage smoother
-     -snes_ms_damping - damping for multi-stage method

      Notes:
      These multistage methods are explicit Runge-Kutta methods that are often used as smoothers for
      FAS multigrid for transport problems. In the linear case, these are equivalent to polynomial smoothers (such as Chebyshev).

      Multi-stage smoothers should usually be preconditioned by point-block Jacobi to ensure proper scaling and to normalize the wave speeds.

      The methods are specified in low storage form (Ketcheson 2010). New methods can be registered with SNESMSRegister().

      References:
+     * - Ketcheson (2010) Runge Kutta methods with minimum storage implementations (https://doi.org/10.1016/j.jcp.2009.11.006).
.     * - Jameson (1983) Solution of the Euler equations for two dimensional transonic flow by a multigrid method (https://doi.org/10.1016/0096-3003(83)90019-X).
.     * - Pierce and Giles (1997) Preconditioned multigrid methods for compressible flow calculations on stretched meshes (https://doi.org/10.1006/jcph.1997.5772).
-     * - Van Leer, Tai, and Powell (1989) Design of optimally smoothing multi-stage schemes for the Euler equations (https://doi.org/10.2514/6.1989-1933).

      Level: beginner

.seealso:  SNESCreate(), SNES, SNESSetType(), SNESMS, SNESFAS, KSPCHEBYSHEV

M*/
PETSC_EXTERN PetscErrorCode SNESCreate_MS(SNES snes)
{
  SNES_MS        *ms;

  PetscFunctionBegin;
  CHKERRQ(SNESMSInitializePackage());

  snes->ops->setup          = SNESSetUp_MS;
  snes->ops->solve          = SNESSolve_MS;
  snes->ops->destroy        = SNESDestroy_MS;
  snes->ops->setfromoptions = SNESSetFromOptions_MS;
  snes->ops->view           = SNESView_MS;
  snes->ops->reset          = SNESReset_MS;

  snes->usesnpc = PETSC_FALSE;
  snes->usesksp = PETSC_TRUE;

  snes->alwayscomputesfinalresidual = PETSC_FALSE;

  CHKERRQ(PetscNewLog(snes,&ms));
  snes->data  = (void*)ms;
  ms->damping = 0.9;
  ms->norms   = PETSC_FALSE;

  CHKERRQ(PetscObjectComposeFunction((PetscObject)snes,"SNESMSGetType_C",SNESMSGetType_MS));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)snes,"SNESMSSetType_C",SNESMSSetType_MS));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)snes,"SNESMSGetDamping_C",SNESMSGetDamping_MS));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)snes,"SNESMSSetDamping_C",SNESMSSetDamping_MS));

  CHKERRQ(SNESMSSetType(snes,SNESMSDefault));
  PetscFunctionReturn(0);
}
