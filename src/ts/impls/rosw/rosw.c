/*
  Code for timestepping with Rosenbrock W methods

  Notes:
  The general system is written as

  G(t,X,Xdot) = F(t,X)

  where G represents the stiff part of the physics and F represents the non-stiff part.
  This method is designed to be linearly implicit on G and can use an approximate and lagged Jacobian.

*/
#include <private/tsimpl.h>                /*I   "petscts.h"   I*/

#include <../src/mat/blockinvert.h>

static const TSRosWType TSRosWDefault = TSROSW2P;
static PetscBool TSRosWRegisterAllCalled;
static PetscBool TSRosWPackageInitialized;

typedef struct _RosWTableau *RosWTableau;
struct _RosWTableau {
  char *name;
  PetscInt order;               /* Classical approximation order of the method */
  PetscInt s;                   /* Number of stages */
  PetscInt pinterp;             /* Interpolation order */
  PetscReal *A;                 /* Propagation table, strictly lower triangular */
  PetscReal *Gamma;             /* Stage table, lower triangular with nonzero diagonal */
  PetscReal *b;                 /* Step completion table */
  PetscReal *ASum;              /* Row sum of A */
  PetscReal *GammaSum;          /* Row sum of Gamma, only needed for non-autonomous systems */
  PetscReal *At;                /* Propagation table in transformed variables */
  PetscReal *bt;                /* Step completion table in transformed variables  */
  PetscReal *GammaInv;          /* Inverse of Gamma, used for transformed variables */
};
typedef struct _RosWTableauLink *RosWTableauLink;
struct _RosWTableauLink {
  struct _RosWTableau tab;
  RosWTableauLink next;
};
static RosWTableauLink RosWTableauList;

typedef struct {
  RosWTableau tableau;
  Vec         *Y;               /* States computed during the step, used to complete the step */
  Vec         Ydot;             /* Work vector holding Ydot during residual evaluation */
  Vec         Ystage;           /* Work vector for the state value at each stage */
  Vec         Zdot;             /* Ydot = Zdot + shift*Y */
  Vec         Zstage;           /* Y = Zstage + Y */
  PetscScalar *work;            /* Scalar work */
  PetscReal   shift;
  PetscReal   stage_time;
  PetscBool   recompute_jacobian; /* Recompute the Jacobian at each stage, default is to freeze the Jacobian at the start of each step */
} TS_RosW;

#undef __FUNCT__
#define __FUNCT__ "TSRosWRegisterAll"
/*@C
  TSRosWRegisterAll - Registers all of the additive Runge-Kutta implicit-explicit methods in TSRosW

  Not Collective, but should be called by all processes which will need the schemes to be registered

  Level: advanced

.keywords: TS, TSRosW, register, all

.seealso:  TSRosWRegisterDestroy()
@*/
PetscErrorCode TSRosWRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (TSRosWRegisterAllCalled) PetscFunctionReturn(0);
  TSRosWRegisterAllCalled = PETSC_TRUE;

  {
    const PetscReal g = 1. + 1./PetscSqrtReal(2.0);
    const PetscReal
      A[2][2] = {{0,0}, {1.,0}},
      Gamma[2][2] = {{g,0}, {-2.*g,g}},
      b[2] = {0.5,0.5};
    ierr = TSRosWRegister(TSROSW2P,2,2,&A[0][0],&Gamma[0][0],b);CHKERRQ(ierr);
  }
  {
    const PetscReal g = 1. - 1./PetscSqrtReal(2.0);
    const PetscReal
      A[2][2] = {{0,0}, {1.,0}},
      Gamma[2][2] = {{g,0}, {-2.*g,g}},
      b[2] = {0.5,0.5};
    ierr = TSRosWRegister(TSROSW2M,2,2,&A[0][0],&Gamma[0][0],b);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSRosWRegisterDestroy"
/*@C
   TSRosWRegisterDestroy - Frees the list of schemes that were registered by TSRosWRegister().

   Not Collective

   Level: advanced

.keywords: TSRosW, register, destroy
.seealso: TSRosWRegister(), TSRosWRegisterAll(), TSRosWRegisterDynamic()
@*/
PetscErrorCode TSRosWRegisterDestroy(void)
{
  PetscErrorCode ierr;
  RosWTableauLink link;

  PetscFunctionBegin;
  while ((link = RosWTableauList)) {
    RosWTableau t = &link->tab;
    RosWTableauList = link->next;
    ierr = PetscFree5(t->A,t->Gamma,t->b,t->ASum,t->GammaSum);CHKERRQ(ierr);
    ierr = PetscFree3(t->At,t->bt,t->GammaInv);CHKERRQ(ierr);
    ierr = PetscFree(t->name);CHKERRQ(ierr);
    ierr = PetscFree(link);CHKERRQ(ierr);
  }
  TSRosWRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSRosWInitializePackage"
/*@C
  TSRosWInitializePackage - This function initializes everything in the TSRosW package. It is called
  from PetscDLLibraryRegister() when using dynamic libraries, and on the first call to TSCreate_RosW()
  when using static libraries.

  Input Parameter:
  path - The dynamic library path, or PETSC_NULL

  Level: developer

.keywords: TS, TSRosW, initialize, package
.seealso: PetscInitialize()
@*/
PetscErrorCode TSRosWInitializePackage(const char path[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (TSRosWPackageInitialized) PetscFunctionReturn(0);
  TSRosWPackageInitialized = PETSC_TRUE;
  ierr = TSRosWRegisterAll();CHKERRQ(ierr);
  ierr = PetscRegisterFinalize(TSRosWFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSRosWFinalizePackage"
/*@C
  TSRosWFinalizePackage - This function destroys everything in the TSRosW package. It is
  called from PetscFinalize().

  Level: developer

.keywords: Petsc, destroy, package
.seealso: PetscFinalize()
@*/
PetscErrorCode TSRosWFinalizePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TSRosWPackageInitialized = PETSC_FALSE;
  ierr = TSRosWRegisterDestroy();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSRosWRegister"
/*@C
   TSRosWRegister - register a Rosenbrock W scheme by providing the entries in the Butcher tableau and optionally embedded approximations and interpolation

   Not Collective, but the same schemes should be registered on all processes on which they will be used

   Input Parameters:
+  name - identifier for method
.  order - approximation order of method
.  s - number of stages, this is the dimension of the matrices below
.  A - Table of propagated stage coefficients (dimension s*s, row-major), strictly lower triangular
.  Gamma - Table of coefficients in implicit stage equations (dimension s*s, row-major), lower triangular with nonzero diagonal
-  b - Step completion table (dimension s)

   Notes:
   Several Rosenbrock W methods are provided, this function is only needed to create new methods.

   Level: advanced

.keywords: TS, register

.seealso: TSRosW
@*/
PetscErrorCode TSRosWRegister(const TSRosWType name,PetscInt order,PetscInt s,
                              const PetscReal A[],const PetscReal Gamma[],const PetscReal b[])
{
  PetscErrorCode ierr;
  RosWTableauLink link;
  RosWTableau t;
  PetscInt i,j,k;
  PetscScalar *GammaInv;

  PetscFunctionBegin;
  ierr = PetscMalloc(sizeof(*link),&link);CHKERRQ(ierr);
  ierr = PetscMemzero(link,sizeof(*link));CHKERRQ(ierr);
  t = &link->tab;
  ierr = PetscStrallocpy(name,&t->name);CHKERRQ(ierr);
  t->order = order;
  t->s = s;
  ierr = PetscMalloc5(s*s,PetscReal,&t->A,s*s,PetscReal,&t->Gamma,s,PetscReal,&t->b,s,PetscReal,&t->ASum,s,PetscReal,&t->GammaSum);CHKERRQ(ierr);
  ierr = PetscMalloc3(s*s,PetscReal,&t->At,s,PetscReal,&t->bt,s*s,PetscReal,&t->GammaInv);CHKERRQ(ierr);
  ierr = PetscMemcpy(t->A,A,s*s*sizeof(A[0]));CHKERRQ(ierr);
  ierr = PetscMemcpy(t->Gamma,Gamma,s*s*sizeof(Gamma[0]));CHKERRQ(ierr);
  ierr = PetscMemcpy(t->b,b,s*sizeof(b[0]));CHKERRQ(ierr);
  for (i=0; i<s; i++) {
    t->ASum[i] = 0;
    t->GammaSum[i] = 0;
    for (j=0; j<s; j++) {
      t->ASum[i] += A[i*s+j];
      t->GammaSum[i] = Gamma[i*s+j];
    }
  }
  ierr = PetscMalloc(s*s*sizeof(PetscScalar),&GammaInv);CHKERRQ(ierr); /* Need to use Scalar for inverse, then convert back to Real */
  for (i=0; i<s*s; i++) GammaInv[i] = Gamma[i];
  switch (s) {
  case 1: GammaInv[0] = 1./GammaInv[0]; break;
  case 2: ierr = Kernel_A_gets_inverse_A_2(GammaInv,0);CHKERRQ(ierr); break;
  case 3: ierr = Kernel_A_gets_inverse_A_3(GammaInv,0);CHKERRQ(ierr); break;
  case 4: ierr = Kernel_A_gets_inverse_A_4(GammaInv,0);CHKERRQ(ierr); break;
  case 5: {
    PetscInt ipvt5[5];
    MatScalar work5[5*5];
    ierr = Kernel_A_gets_inverse_A_5(GammaInv,ipvt5,work5,0);CHKERRQ(ierr); break;
  }
  case 6: ierr = Kernel_A_gets_inverse_A_6(GammaInv,0);CHKERRQ(ierr); break;
  case 7: ierr = Kernel_A_gets_inverse_A_7(GammaInv,0);CHKERRQ(ierr); break;
  default: SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not implemented for %D stages",s);
  }
  for (i=0; i<s*s; i++) t->GammaInv[i] = PetscRealPart(GammaInv[i]);
  ierr = PetscFree(GammaInv);CHKERRQ(ierr);
  for (i=0; i<s; i++) {
    for (j=0; j<s; j++) {
      t->At[i*s+j] = 0;
      for (k=0; k<s; k++) {
        t->At[i*s+j] += t->A[i*s+k] * t->GammaInv[k*s+j];
      }
    }
    t->bt[i] = 0;
    for (j=0; j<s; j++) {
      t->bt[i] += t->b[j] * t->GammaInv[j*s+i];
    }
  }
  link->next = RosWTableauList;
  RosWTableauList = link;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSStep_RosW"
static PetscErrorCode TSStep_RosW(TS ts)
{
  TS_RosW         *ros = (TS_RosW*)ts->data;
  RosWTableau     tab  = ros->tableau;
  const PetscInt  s    = tab->s;
  const PetscReal *At  = tab->At,*Gamma = tab->Gamma,*bt = tab->bt,*ASum = tab->ASum,*GammaInv = tab->GammaInv;
  PetscScalar     *w   = ros->work;
  Vec             *Y   = ros->Y,Zdot = ros->Zdot,Zstage = ros->Zstage;
  SNES            snes;
  PetscInt        i,j,its,lits;
  PetscReal       next_time_step;
  PetscReal       h,t;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  next_time_step = ts->time_step;
  h = ts->time_step;
  t = ts->ptime;

  for (i=0; i<s; i++) {
    ros->stage_time = t + h*ASum[i];
    ros->shift = 1./(h*Gamma[i*s+i]);

    ierr = VecCopy(ts->vec_sol,Zstage);CHKERRQ(ierr);
    ierr = VecMAXPY(Zstage,i,&At[i*s+0],Y);CHKERRQ(ierr);

    for (j=0; j<i; j++) w[j] = 1./h * GammaInv[i*s+j];
    ierr = VecZeroEntries(Zdot);CHKERRQ(ierr);
    ierr = VecMAXPY(Zdot,i,w,Y);CHKERRQ(ierr);

    /* Initial guess taken from last stage */
    ierr = VecZeroEntries(Y[i]);CHKERRQ(ierr);

    if (!ros->recompute_jacobian && !i) {
      ierr = SNESSetLagJacobian(snes,-2);CHKERRQ(ierr); /* Recompute the Jacobian on this solve, but not again */
    }

    ierr = SNESSolve(snes,PETSC_NULL,Y[i]);CHKERRQ(ierr);
    ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);
    ierr = SNESGetLinearSolveIterations(snes,&lits);CHKERRQ(ierr);
    ts->nonlinear_its += its; ts->linear_its += lits;
  }
  ierr = VecMAXPY(ts->vec_sol,s,bt,Y);CHKERRQ(ierr);

  ts->ptime += ts->time_step;
  ts->time_step = next_time_step;
  ts->steps++;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSInterpolate_RosW"
static PetscErrorCode TSInterpolate_RosW(TS ts,PetscReal itime,Vec X)
{
  TS_RosW        *ros = (TS_RosW*)ts->data;

  PetscFunctionBegin;
  SETERRQ1(((PetscObject)ts)->comm,PETSC_ERR_SUP,"TSRosW %s does not have an interpolation formula",ros->tableau->name);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "TSReset_RosW"
static PetscErrorCode TSReset_RosW(TS ts)
{
  TS_RosW        *ros = (TS_RosW*)ts->data;
  PetscInt       s;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!ros->tableau) PetscFunctionReturn(0);
   s = ros->tableau->s;
  ierr = VecDestroyVecs(s,&ros->Y);CHKERRQ(ierr);
  ierr = VecDestroy(&ros->Ydot);CHKERRQ(ierr);
  ierr = VecDestroy(&ros->Ystage);CHKERRQ(ierr);
  ierr = VecDestroy(&ros->Zdot);CHKERRQ(ierr);
  ierr = VecDestroy(&ros->Zstage);CHKERRQ(ierr);
  ierr = PetscFree(ros->work);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSDestroy_RosW"
static PetscErrorCode TSDestroy_RosW(TS ts)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = TSReset_RosW(ts);CHKERRQ(ierr);
  ierr = PetscFree(ts->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSRosWGetType_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSRosWSetType_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSRosWSetRecomputeJacobian_C","",PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  This defines the nonlinear equation that is to be solved with SNES
  G(U) = F[t0+Theta*dt, U, (U-U0)*shift] = 0
*/
#undef __FUNCT__
#define __FUNCT__ "SNESTSFormFunction_RosW"
static PetscErrorCode SNESTSFormFunction_RosW(SNES snes,Vec X,Vec F,TS ts)
{
  TS_RosW        *ros = (TS_RosW*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecWAXPY(ros->Ydot,ros->shift,X,ros->Zdot);CHKERRQ(ierr); /* Ydot = shift*X + Zdot */
  ierr = VecWAXPY(ros->Ystage,1.0,X,ros->Zstage);CHKERRQ(ierr);    /* Ystage = X + Zstage */
  ierr = TSComputeIFunction(ts,ros->stage_time,ros->Ystage,ros->Ydot,F,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESTSFormJacobian_RosW"
static PetscErrorCode SNESTSFormJacobian_RosW(SNES snes,Vec X,Mat *A,Mat *B,MatStructure *str,TS ts)
{
  TS_RosW        *ros = (TS_RosW*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* ros->Ydot and ros->Ystage have already been computed in SNESTSFormFunction_RosW (SNES guarantees this) */
  ierr = TSComputeIJacobian(ts,ros->stage_time,ros->Ystage,ros->Ydot,ros->shift,A,B,str,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSetUp_RosW"
static PetscErrorCode TSSetUp_RosW(TS ts)
{
  TS_RosW        *ros = (TS_RosW*)ts->data;
  RosWTableau    tab  = ros->tableau;
  PetscInt       s    = tab->s;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!ros->tableau) {
    ierr = TSRosWSetType(ts,TSRosWDefault);CHKERRQ(ierr);
  }
  ierr = VecDuplicateVecs(ts->vec_sol,s,&ros->Y);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&ros->Ydot);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&ros->Ystage);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&ros->Zdot);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&ros->Zstage);CHKERRQ(ierr);
  ierr = PetscMalloc(s*sizeof(ros->work[0]),&ros->work);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*------------------------------------------------------------*/

#undef __FUNCT__
#define __FUNCT__ "TSSetFromOptions_RosW"
static PetscErrorCode TSSetFromOptions_RosW(TS ts)
{
  TS_RosW        *ros = (TS_RosW*)ts->data;
  PetscErrorCode ierr;
  char           rostype[256];

  PetscFunctionBegin;
  ierr = PetscOptionsHead("RosW ODE solver options");CHKERRQ(ierr);
  {
    RosWTableauLink link;
    PetscInt count,choice;
    PetscBool flg;
    const char **namelist;
    SNES snes;

    ierr = PetscStrncpy(rostype,TSRosWDefault,sizeof rostype);CHKERRQ(ierr);
    for (link=RosWTableauList,count=0; link; link=link->next,count++) ;
    ierr = PetscMalloc(count*sizeof(char*),&namelist);CHKERRQ(ierr);
    for (link=RosWTableauList,count=0; link; link=link->next,count++) namelist[count] = link->tab.name;
    ierr = PetscOptionsEList("-ts_rosw_type","Family of Rosenbrock-W method","TSRosWSetType",(const char*const*)namelist,count,rostype,&choice,&flg);CHKERRQ(ierr);
    ierr = TSRosWSetType(ts,flg ? namelist[choice] : rostype);CHKERRQ(ierr);
    ierr = PetscFree(namelist);CHKERRQ(ierr);

    ierr = PetscOptionsBool("-ts_rosw_recompute_jacobian","Recompute the Jacobian at each stage","TSRosWSetRecomputeJacobian",ros->recompute_jacobian,&ros->recompute_jacobian,PETSC_NULL);CHKERRQ(ierr);

    /* Rosenbrock methods are linearly implicit, so set that unless the user has specifically asked for something else */
    ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
    if (!((PetscObject)snes)->type_name) {
      ierr = SNESSetType(snes,SNESKSPONLY);CHKERRQ(ierr);
    }
    ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFormatRealArray"
static PetscErrorCode PetscFormatRealArray(char buf[],size_t len,const char *fmt,PetscInt n,const PetscReal x[])
{
  PetscErrorCode ierr;
  PetscInt i;
  size_t left,count;
  char *p;

  PetscFunctionBegin;
  for (i=0,p=buf,left=len; i<n; i++) {
    ierr = PetscSNPrintfCount(p,left,fmt,&count,x[i]);CHKERRQ(ierr);
    if (count >= left) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Insufficient space in buffer");
    left -= count;
    p += count;
    *p++ = ' ';
  }
  p[i ? 0 : -1] = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSView_RosW"
static PetscErrorCode TSView_RosW(TS ts,PetscViewer viewer)
{
  TS_RosW        *ros = (TS_RosW*)ts->data;
  RosWTableau    tab  = ros->tableau;
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    const TSRosWType rostype;
    PetscInt i;
    PetscReal abscissa[512];
    char buf[512];
    ierr = TSRosWGetType(ts,&rostype);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  Rosenbrock-W %s\n",rostype);CHKERRQ(ierr);
    ierr = PetscFormatRealArray(buf,sizeof buf,"% 8.6f",tab->s,tab->ASum);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  Abscissa of A       = %s\n",buf);CHKERRQ(ierr);
    for (i=0; i<tab->s; i++) abscissa[i] = tab->ASum[i] + tab->Gamma[i];
    ierr = PetscFormatRealArray(buf,sizeof buf,"% 8.6f",tab->s,abscissa);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  Abscissa of A+Gamma = %s\n",buf);CHKERRQ(ierr);
  }
  ierr = SNESView(ts->snes,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSRosWSetType"
/*@C
  TSRosWSetType - Set the type of Rosenbrock-W scheme

  Logically collective

  Input Parameter:
+  ts - timestepping context
-  rostype - type of Rosenbrock-W scheme

  Level: intermediate

.seealso: TSRosWGetType()
@*/
PetscErrorCode TSRosWSetType(TS ts,const TSRosWType rostype)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = PetscTryMethod(ts,"TSRosWSetType_C",(TS,const TSRosWType),(ts,rostype));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSRosWGetType"
/*@C
  TSRosWGetType - Get the type of Rosenbrock-W scheme

  Logically collective

  Input Parameter:
.  ts - timestepping context

  Output Parameter:
.  rostype - type of Rosenbrock-W scheme

  Level: intermediate

.seealso: TSRosWGetType()
@*/
PetscErrorCode TSRosWGetType(TS ts,const TSRosWType *rostype)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = PetscUseMethod(ts,"TSRosWGetType_C",(TS,const TSRosWType*),(ts,rostype));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSRosWSetRecomputeJacobian"
/*@C
  TSRosWSetRecomputeJacobian - Set whether to recompute the Jacobian at each stage. The default is to update the Jacobian once per step.

  Logically collective

  Input Parameter:
+  ts - timestepping context
-  flg - PETSC_TRUE to recompute the Jacobian at each stage

  Level: intermediate

.seealso: TSRosWGetType()
@*/
PetscErrorCode TSRosWSetRecomputeJacobian(TS ts,PetscBool flg)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = PetscTryMethod(ts,"TSRosWSetRecomputeJacobian_C",(TS,PetscBool),(ts,flg));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "TSRosWGetType_RosW"
PetscErrorCode  TSRosWGetType_RosW(TS ts,const TSRosWType *rostype)
{
  TS_RosW        *ros = (TS_RosW*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!ros->tableau) {ierr = TSRosWSetType(ts,TSRosWDefault);CHKERRQ(ierr);}
  *rostype = ros->tableau->name;
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "TSRosWSetType_RosW"
PetscErrorCode  TSRosWSetType_RosW(TS ts,const TSRosWType rostype)
{
  TS_RosW         *ros = (TS_RosW*)ts->data;
  PetscErrorCode  ierr;
  PetscBool       match;
  RosWTableauLink link;

  PetscFunctionBegin;
  if (ros->tableau) {
    ierr = PetscStrcmp(ros->tableau->name,rostype,&match);CHKERRQ(ierr);
    if (match) PetscFunctionReturn(0);
  }
  for (link = RosWTableauList; link; link=link->next) {
    ierr = PetscStrcmp(link->tab.name,rostype,&match);CHKERRQ(ierr);
    if (match) {
      ierr = TSReset_RosW(ts);CHKERRQ(ierr);
      ros->tableau = &link->tab;
      PetscFunctionReturn(0);
    }
  }
  SETERRQ1(((PetscObject)ts)->comm,PETSC_ERR_ARG_UNKNOWN_TYPE,"Could not find '%s'",rostype);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSRosWSetRecomputeJacobian_RosW"
PetscErrorCode  TSRosWSetRecomputeJacobian_RosW(TS ts,PetscBool flg)
{
  TS_RosW *ros = (TS_RosW*)ts->data;

  PetscFunctionBegin;
  ros->recompute_jacobian = flg;
  PetscFunctionReturn(0);
}
EXTERN_C_END

/* ------------------------------------------------------------ */
/*MC
      TSRosW - ODE solver using Rosenbrock-W schemes

  These methods are intended for problems with well-separated time scales, especially when a slow scale is strongly
  nonlinear such that it is expensive to solve with a fully implicit method. The user should provide the stiff part
  of the equation using TSSetIFunction() and the non-stiff part with TSSetRHSFunction().

  Notes:
  This method currently only works with autonomous ODE and DAE.

  Developer notes:
  Rosenbrock-W methods are typically specified for autonomous ODE

$  xdot = f(x)

  by the stage equations

$  k_i = h f(x_0 + sum_j alpha_ij k_j) + h J sum_j gamma_ij k_j

  and step completion formula

$  x_1 = x_0 + sum_j b_j k_j

  with step size h and coefficients alpha_ij, gamma_ij, and b_i. Implementing the method in this form would require f(x)
  and the Jacobian J to be available, in addition to the shifted matrix I - h gamma_ii J. Following Hairer and Wanner,
  we define new variables for the stage equations

$  y_i = gamma_ij k_j

  The k_j can be recovered because Gamma is invertible. Let C be the lower triangular part of Gamma^{-1} and define

$  A = Alpha Gamma^{-1}, bt^T = b^T Gamma^{-i}

  to rewrite the method as

$  [M/(h gamma_ii) - J] y_i = f(x_0 + sum_j a_ij y_j) + M sum_j (c_ij/h) y_j
$  x_1 = x_0 + sum_j bt_j y_j

   where we have introduced the mass matrix M. Continue by defining

$  ydot_i = 1/(h gamma_ii) y_i - sum_j (c_ij/h) y_j

   or, more compactly in tensor notation

$  Ydot = 1/h (Gamma^{-1} \otimes I) Y .

   Note that Gamma^{-1} is lower triangular. With this definition of Ydot in terms of known quantities and the current
   stage y_i, the stage equations reduce to performing one Newton step (typically with a lagged Jacobian) on the
   equation

$  g(x_0 + sum_j a_ij y_j + y_i, ydot_i) = 0

   with initial guess y_i = 0.

  Level: beginner

.seealso:  TSCreate(), TS, TSSetType(), TSRosWRegister()

M*/
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "TSCreate_RosW"
PetscErrorCode  TSCreate_RosW(TS ts)
{
  TS_RosW        *ros;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if !defined(PETSC_USE_DYNAMIC_LIBRARIES)
  ierr = TSRosWInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif

  ts->ops->reset          = TSReset_RosW;
  ts->ops->destroy        = TSDestroy_RosW;
  ts->ops->view           = TSView_RosW;
  ts->ops->setup          = TSSetUp_RosW;
  ts->ops->step           = TSStep_RosW;
  ts->ops->interpolate    = TSInterpolate_RosW;
  ts->ops->setfromoptions = TSSetFromOptions_RosW;
  ts->ops->snesfunction   = SNESTSFormFunction_RosW;
  ts->ops->snesjacobian   = SNESTSFormJacobian_RosW;

  ierr = PetscNewLog(ts,TS_RosW,&ros);CHKERRQ(ierr);
  ts->data = (void*)ros;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSRosWGetType_C","TSRosWGetType_RosW",TSRosWGetType_RosW);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSRosWSetType_C","TSRosWSetType_RosW",TSRosWSetType_RosW);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSRosWSetRecomputeJacobian_C","TSRosWSetRecomputeJacobian_RosW",TSRosWSetRecomputeJacobian_RosW);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
