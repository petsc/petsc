#include <petsc-private/snesimpl.h>   /*I "petscsnes.h" I*/

static SNESMSType SNESMSDefault = SNESMSM62;
static PetscBool SNESMSRegisterAllCalled;
static PetscBool SNESMSPackageInitialized;

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
  SNESMSTableauLink next;
};
static SNESMSTableauLink SNESMSTableauList;

typedef struct {
  SNESMSTableau tableau;        /* Tableau in low-storage form */
  PetscReal     damping;        /* Damping parameter, like length of (pseudo) time step */
  PetscBool     norms;          /* Compute norms, usually only for monitoring purposes */
} SNES_MS;

#undef __FUNCT__
#define __FUNCT__ "SNESMSRegisterAll"
/*@C
  SNESMSRegisterAll - Registers all of the multi-stage methods in SNESMS

  Not Collective, but should be called by all processes which will need the schemes to be registered

  Level: advanced

.keywords: SNES, SNESMS, register, all

.seealso:  SNESMSRegisterDestroy()
@*/
PetscErrorCode SNESMSRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (SNESMSRegisterAllCalled) PetscFunctionReturn(0);
  SNESMSRegisterAllCalled = PETSC_TRUE;

  {
    PetscReal gamma[3][1] = {{1.0},{0.0},{0.0}};
    PetscReal delta[1] = {0.0};
    PetscReal betasub[1] = {1.0};
    ierr = SNESMSRegister(SNESMSEULER,1,3,1.0,&gamma[0][0],delta,betasub);CHKERRQ(ierr);
  }

  {
    PetscReal gamma[3][6] = {
      {0.0000000000000000E+00,  -7.0304722367110606E-01, -1.9836719667506464E-01, -1.6023843981863788E+00, 9.4483822882855284E-02,  -1.4204296130641869E-01},
      {1.0000000000000000E+00,  1.1111025767083920E+00,  5.6150921583923230E-01,  7.4151723494934041E-01,  3.1714538168600587E-01,  4.6479276238548706E-01},
      {0.0000000000000000E+00,  0.0000000000000000E+00,  0.0000000000000000E+00,  6.7968174970583317E-01,  -4.1755042846051737E-03, -1.9115668129923846E-01}
    };
    PetscReal delta[6] = {1.0000000000000000E+00, 5.3275427433201750E-01, 6.0143544663985238E-01, 4.5874077053842177E-01, 2.7544386906104651E-01, 0.0000000000000000E+00};
    PetscReal betasub[6] = {8.4753115429481929E-01, 7.4018896368655618E-01, 6.5963574086583309E-03, 4.6747795645517759E-01, 1.3314545813643919E-01, 5.3260800028018784E-01};
    ierr = SNESMSRegister(SNESMSM62,6,3,1.0,&gamma[0][0],delta,betasub);CHKERRQ(ierr);
  }
  {
    PetscReal gamma[3][4] = {{0,0,0,0},{0,0,0,0},{1,1,1,1}};
    PetscReal delta[4] = {0,0,0,0};
    PetscReal betasub[4] = {0.25, 0.5, 0.55, 1.0};
    ierr = SNESMSRegister(SNESMSJAMESON83,4,3,1.0,&gamma[0][0],delta,betasub);CHKERRQ(ierr);
  }
  {                             /* Van Leer, Tai, and Powell (1989) 2 stage, order 1 */
    PetscReal gamma[3][2] = {{0,0},{0,0},{1,1}};
    PetscReal delta[2] = {0,0};
    PetscReal betasub[2] = {0.3333,1.0};
    ierr = SNESMSRegister(SNESMSVLTP21,2,3,1.0,&gamma[0][0],delta,betasub);CHKERRQ(ierr);
  }
  {                             /* Van Leer, Tai, and Powell (1989) 3 stage, order 1 */
    PetscReal gamma[3][3] = {{0,0,0},{0,0,0},{1,1,1}};
    PetscReal delta[3] = {0,0,0};
    PetscReal betasub[3] = {0.1481,0.4000,1.0};
    ierr = SNESMSRegister(SNESMSVLTP31,3,3,1.5,&gamma[0][0],delta,betasub);CHKERRQ(ierr);
  }
  {                             /* Van Leer, Tai, and Powell (1989) 4 stage, order 1 */
    PetscReal gamma[3][4] = {{0,0,0,0},{0,0,0,0},{1,1,1,1}};
    PetscReal delta[4] = {0,0,0,0};
    PetscReal betasub[4] = {0.0833,0.2069,0.4265,1.0};
    ierr = SNESMSRegister(SNESMSVLTP41,4,3,2.0,&gamma[0][0],delta,betasub);CHKERRQ(ierr);
  }
  {                             /* Van Leer, Tai, and Powell (1989) 5 stage, order 1 */
    PetscReal gamma[3][5] = {{0,0,0,0,0},{0,0,0,0,0},{1,1,1,1,1}};
    PetscReal delta[5] = {0,0,0,0,0};
    PetscReal betasub[5] = {0.0533,0.1263,0.2375,0.4414,1.0};
    ierr = SNESMSRegister(SNESMSVLTP51,5,3,2.5,&gamma[0][0],delta,betasub);CHKERRQ(ierr);
  }
  {                             /* Van Leer, Tai, and Powell (1989) 6 stage, order 1 */
    PetscReal gamma[3][6] = {{0,0,0,0,0,0},{0,0,0,0,0,0},{1,1,1,1,1,1}};
    PetscReal delta[6] = {0,0,0,0,0,0};
    PetscReal betasub[6] = {0.0370,0.0851,0.1521,0.2562,0.4512,1.0};
    ierr = SNESMSRegister(SNESMSVLTP61,6,3,3.0,&gamma[0][0],delta,betasub);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESMSRegisterDestroy"
/*@C
   SNESMSRegisterDestroy - Frees the list of schemes that were registered by TSRosWRegister().

   Not Collective

   Level: advanced

.keywords: TSRosW, register, destroy
.seealso: TSRosWRegister(), TSRosWRegisterAll(), TSRosWRegisterDynamic()
@*/
PetscErrorCode SNESMSRegisterDestroy(void)
{
  PetscErrorCode ierr;
  SNESMSTableauLink link;

  PetscFunctionBegin;
  while ((link = SNESMSTableauList)) {
    SNESMSTableau t = &link->tab;
    SNESMSTableauList = link->next;
    ierr = PetscFree3(t->gamma,t->delta,t->betasub);CHKERRQ(ierr);
    ierr = PetscFree(t->name);CHKERRQ(ierr);
    ierr = PetscFree(link);CHKERRQ(ierr);
  }
  SNESMSRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESMSInitializePackage"
/*@C
  SNESMSInitializePackage - This function initializes everything in the SNESMS package. It is called
  from PetscDLLibraryRegister() when using dynamic libraries, and on the first call to SNESCreate_MS()
  when using static libraries.

  Input Parameter:
  path - The dynamic library path, or PETSC_NULL

  Level: developer

.keywords: SNES, SNESMS, initialize, package
.seealso: PetscInitialize()
@*/
PetscErrorCode SNESMSInitializePackage(const char path[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (SNESMSPackageInitialized) PetscFunctionReturn(0);
  SNESMSPackageInitialized = PETSC_TRUE;
  ierr = SNESMSRegisterAll();CHKERRQ(ierr);
  ierr = PetscRegisterFinalize(SNESMSFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESMSFinalizePackage"
/*@C
  SNESMSFinalizePackage - This function destroys everything in the SNESMS package. It is
  called from PetscFinalize().

  Level: developer

.keywords: Petsc, destroy, package
.seealso: PetscFinalize()
@*/
PetscErrorCode SNESMSFinalizePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  SNESMSPackageInitialized = PETSC_FALSE;
  ierr = SNESMSRegisterDestroy();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESMSRegister"
/*@C
   SNESMSRegister - register a multistage scheme

   Not Collective, but the same schemes should be registered on all processes on which they will be used

   Input Parameters:
+  name - identifier for method
.  nstages - number of stages
.  nregisters - number of registers used by low-storage implementation
.  gamma - coefficients, see Ketcheson's paper
.  delta - coefficients, see Ketcheson's paper
-  betasub - subdiagonal of Shu-Osher form

   Notes:
   The notation is described in Ketcheson (2010) Runge-Kutta methods with minimum storage implementations.

   Level: advanced

.keywords: SNES, register

.seealso: SNESMS
@*/
PetscErrorCode SNESMSRegister(SNESMSType name,PetscInt nstages,PetscInt nregisters,PetscReal stability,const PetscReal gamma[],const PetscReal delta[],const PetscReal betasub[])
{
  PetscErrorCode ierr;
  SNESMSTableauLink link;
  SNESMSTableau t;

  PetscFunctionBegin;
  PetscValidCharPointer(name,1);
  if (nstages < 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Must have at least one stage");
  if (nregisters != 3) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only support for methods written in 3-register form");
  PetscValidPointer(gamma,4);
  PetscValidPointer(delta,5);
  PetscValidPointer(betasub,6);

  ierr = PetscMalloc(sizeof(*link),&link);CHKERRQ(ierr);
  ierr = PetscMemzero(link,sizeof(*link));CHKERRQ(ierr);
  t = &link->tab;
  ierr = PetscStrallocpy(name,&t->name);CHKERRQ(ierr);
  t->nstages    = nstages;
  t->nregisters = nregisters;
  t->stability  = stability;
  ierr = PetscMalloc3(nstages*nregisters,PetscReal,&t->gamma,nstages,PetscReal,&t->delta,nstages,PetscReal,&t->betasub);CHKERRQ(ierr);
  ierr = PetscMemcpy(t->gamma,gamma,nstages*nregisters*sizeof(PetscReal));CHKERRQ(ierr);
  ierr = PetscMemcpy(t->delta,delta,nstages*sizeof(PetscReal));CHKERRQ(ierr);
  ierr = PetscMemcpy(t->betasub,betasub,nstages*sizeof(PetscReal));CHKERRQ(ierr);

  link->next = SNESMSTableauList;
  SNESMSTableauList = link;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESMSStep_3Sstar"
/*
  X - initial state, updated in-place.
  F - residual, computed at the initial X on input
*/
static PetscErrorCode SNESMSStep_3Sstar(SNES snes,Vec X,Vec F)
{
  PetscErrorCode ierr;
  SNES_MS        *ms = (SNES_MS*)snes->data;
  SNESMSTableau  t = ms->tableau;
  const PetscReal *gamma = t->gamma,*delta = t->delta,*betasub = t->betasub;
  Vec            S1,S2,S3,Y;
  PetscInt       i,nstages = t->nstages;;


  PetscFunctionBegin;
  Y = snes->work[0];
  S1 = X;
  S2 = snes->work[1];
  S3 = snes->work[2];
  ierr = VecZeroEntries(S2);CHKERRQ(ierr);
  ierr = VecCopy(X,S3);CHKERRQ(ierr);
  for (i=0; i<nstages; i++) {
    Vec Ss[4] = {S1,S2,S3,Y};
    PetscScalar scoeff[4] = {gamma[0*nstages+i]-(PetscReal)1.0,gamma[1*nstages+i],gamma[2*nstages+i],-betasub[i]*ms->damping};
    ierr = VecAXPY(S2,delta[i],S1);CHKERRQ(ierr);
    if (i>0) {
      ierr = SNESComputeFunction(snes,S1,F);CHKERRQ(ierr);
      if (snes->domainerror) {
        snes->reason = SNES_DIVERGED_FUNCTION_DOMAIN;
        PetscFunctionReturn(0);
      }
    }
    ierr = SNES_KSPSolve(snes,snes->ksp,F,Y);CHKERRQ(ierr);
    ierr = VecMAXPY(S1,4,scoeff,Ss);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSolve_MS"
static PetscErrorCode SNESSolve_MS(SNES snes)
{
  SNES_MS        *ms = (SNES_MS*)snes->data;
  Vec            X = snes->vec_sol,F = snes->vec_func;
  PetscReal      fnorm;
  MatStructure   mstruct;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  snes->reason = SNES_CONVERGED_ITERATING;
  ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
  snes->iter = 0;
  snes->norm = 0.;
  ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
  if (!snes->vec_func_init_set) {
    ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);
    if (snes->domainerror) {
      snes->reason = SNES_DIVERGED_FUNCTION_DOMAIN;
      PetscFunctionReturn(0);
    }
  } else {
    snes->vec_func_init_set = PETSC_FALSE;
  }
  if (snes->jacobian) {         /* This method does not require a Jacobian, but it is usually preconditioned by PBJacobi */
    ierr = SNESComputeJacobian(snes,snes->vec_sol,&snes->jacobian,&snes->jacobian_pre,&mstruct);CHKERRQ(ierr);
    ierr = KSPSetOperators(snes->ksp,snes->jacobian,snes->jacobian_pre,mstruct);CHKERRQ(ierr);
  }
  if (ms->norms) {
    if (!snes->norm_init_set) {
      ierr = VecNorm(F,NORM_2,&fnorm);CHKERRQ(ierr); /* fnorm <- ||F||  */
      if (PetscIsInfOrNanReal(fnorm)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"Infinite or not-a-number generated in norm");
    } else {
      fnorm = snes->norm_init;
      snes->norm_init_set = PETSC_FALSE;
    }
    /* Monitor convergence */
    ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
    snes->iter = 0;
    snes->norm = fnorm;
    ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
    SNESLogConvHistory(snes,snes->norm,0);
    ierr = SNESMonitor(snes,snes->iter,snes->norm);CHKERRQ(ierr);

    /* set parameter for default relative tolerance convergence test */
    snes->ttol = fnorm*snes->rtol;
    /* Test for convergence */
    ierr = (*snes->ops->converged)(snes,snes->iter,0.0,0.0,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
    if (snes->reason) PetscFunctionReturn(0);
  }

  /* Call general purpose update function */
  if (snes->ops->update) {
    ierr = (*snes->ops->update)(snes,snes->iter);CHKERRQ(ierr);
  }
  for (i = 0; i < snes->max_its; i++) {
    ierr = SNESMSStep_3Sstar(snes,X,F);CHKERRQ(ierr);

    if (i+1 < snes->max_its || ms->norms) {
      ierr = SNESComputeFunction(snes,X,F);CHKERRQ(ierr);
      if (snes->domainerror) {
        snes->reason = SNES_DIVERGED_FUNCTION_DOMAIN;
        PetscFunctionReturn(0);
      }
    }

    if (ms->norms) {
      ierr = VecNorm(F,NORM_2,&fnorm);CHKERRQ(ierr); /* fnorm <- ||F||  */
      if (PetscIsInfOrNanReal(fnorm)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FP,"Infinite or not-a-number generated in norm");
      /* Monitor convergence */
      ierr = PetscObjectTakeAccess(snes);CHKERRQ(ierr);
      snes->iter = i+1;
      snes->norm = fnorm;
      ierr = PetscObjectGrantAccess(snes);CHKERRQ(ierr);
      SNESLogConvHistory(snes,snes->norm,0);
      ierr = SNESMonitor(snes,snes->iter,snes->norm);CHKERRQ(ierr);

      /* Test for convergence */
      ierr = (*snes->ops->converged)(snes,snes->iter,0.0,0.0,fnorm,&snes->reason,snes->cnvP);CHKERRQ(ierr);
      if (snes->reason) PetscFunctionReturn(0);
    }

    /* Call general purpose update function */
    if (snes->ops->update) {
      ierr = (*snes->ops->update)(snes, snes->iter);CHKERRQ(ierr);
    }
  }
  if (!snes->reason) snes->reason = SNES_CONVERGED_ITS;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSetUp_MS"
static PetscErrorCode SNESSetUp_MS(SNES snes)
{
  SNES_MS * ms = (SNES_MS *)snes->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!ms->tableau) {ierr = SNESMSSetType(snes,SNESMSDefault);CHKERRQ(ierr);}
  ierr = SNESDefaultGetWork(snes,3);CHKERRQ(ierr);
  ierr = SNESSetUpMatrices(snes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESReset_MS"
static PetscErrorCode SNESReset_MS(SNES snes)
{

  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESDestroy_MS"
static PetscErrorCode SNESDestroy_MS(SNES snes)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(snes->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)snes,"","",PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESView_MS"
static PetscErrorCode SNESView_MS(SNES snes,PetscViewer viewer)
{
  PetscBool        iascii;
  PetscErrorCode   ierr;
  SNES_MS          *ms = (SNES_MS*)snes->data;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {
    SNESMSTableau tab = ms->tableau;
    ierr = PetscViewerASCIIPrintf(viewer,"  multi-stage method type: %s\n",tab?tab->name:"not yet set");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESSetFromOptions_MS"
static PetscErrorCode SNESSetFromOptions_MS(SNES snes)
{
  SNES_MS        *ms = (SNES_MS*)snes->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("SNES MS options");CHKERRQ(ierr);
  {
    SNESMSTableauLink link;
    PetscInt count,choice;
    PetscBool flg;
    const char **namelist;
    char mstype[256];

    ierr = PetscStrncpy(mstype,SNESMSDefault,sizeof(mstype));CHKERRQ(ierr);
    for (link=SNESMSTableauList,count=0; link; link=link->next,count++) ;
    ierr = PetscMalloc(count*sizeof(char*),&namelist);CHKERRQ(ierr);
    for (link=SNESMSTableauList,count=0; link; link=link->next,count++) namelist[count] = link->tab.name;
    ierr = PetscOptionsEList("-snes_ms_type","Multistage smoother type","SNESMSSetType",(const char*const*)namelist,count,mstype,&choice,&flg);CHKERRQ(ierr);
    ierr = SNESMSSetType(snes,flg ? namelist[choice] : mstype);CHKERRQ(ierr);
    ierr = PetscFree(namelist);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-snes_ms_damping","Damping for multistage method","SNESMSSetDamping",ms->damping,&ms->damping,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-snes_ms_norms","Compute norms for monitoring","none",ms->norms,&ms->norms,PETSC_NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "SNESMSSetType_MS"
PetscErrorCode  SNESMSSetType_MS(SNES snes,SNESMSType mstype)
{
  PetscErrorCode    ierr;
  SNES_MS           *ms = (SNES_MS*)snes->data;
  SNESMSTableauLink link;
  PetscBool         match;

  PetscFunctionBegin;
  if (ms->tableau) {
    ierr = PetscStrcmp(ms->tableau->name,mstype,&match);CHKERRQ(ierr);
    if (match) PetscFunctionReturn(0);
  }
  for (link = SNESMSTableauList; link; link=link->next) {
    ierr = PetscStrcmp(link->tab.name,mstype,&match);CHKERRQ(ierr);
    if (match) {
      ierr = SNESReset_MS(snes);CHKERRQ(ierr);
      ms->tableau = &link->tab;
      PetscFunctionReturn(0);
    }
  }
  SETERRQ1(((PetscObject)snes)->comm,PETSC_ERR_ARG_UNKNOWN_TYPE,"Could not find '%s'",mstype);
  PetscFunctionReturn(0);
}
EXTERN_C_END


#undef __FUNCT__
#define __FUNCT__ "SNESMSSetType"
/*@C
  SNESMSSetType - Set the type of multistage smoother

  Logically collective

  Input Parameter:
+  snes - nonlinear solver context
-  mstype - type of multistage method

  Level: beginner

.seealso: SNESMSGetType(), SNESMS
@*/
PetscErrorCode SNESMSSetType(SNES snes,SNESMSType rostype)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes,SNES_CLASSID,1);
  ierr = PetscTryMethod(snes,"SNESMSSetType_C",(SNES,SNESMSType),(snes,rostype));CHKERRQ(ierr);
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

      Ketcheson (2010) Runge-Kutta methods with minimum storage implementations.

      Jameson (1983) Solution of the Euler equations for two dimensional transonic flow by a multigrid method.

      Pierce and Giles (1997) Preconditioned multigrid methods for compressible flow calculations on stretched meshes.

      Level: beginner

.seealso:  SNESCreate(), SNES, SNESSetType(), SNESMS, SNESFAS, KSPCHEBYSHEV

M*/
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "SNESCreate_MS"
PetscErrorCode  SNESCreate_MS(SNES snes)
{
  PetscErrorCode ierr;
  SNES_MS        *ms;

  PetscFunctionBegin;
#if !defined(PETSC_USE_DYNAMIC_LIBRARIES)
  ierr = SNESMSInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif

  snes->ops->setup          = SNESSetUp_MS;
  snes->ops->solve          = SNESSolve_MS;
  snes->ops->destroy        = SNESDestroy_MS;
  snes->ops->setfromoptions = SNESSetFromOptions_MS;
  snes->ops->view           = SNESView_MS;
  snes->ops->reset          = SNESReset_MS;

  snes->usespc  = PETSC_FALSE;
  snes->usesksp = PETSC_TRUE;

  ierr = PetscNewLog(snes,SNES_MS,&ms);CHKERRQ(ierr);
  snes->data = (void*)ms;
  ms->damping = 0.9;
  ms->norms   = PETSC_FALSE;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)snes,"SNESMSSetType_C","SNESMSSetType_MS",SNESMSSetType_MS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
