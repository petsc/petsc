/*
  Code for timestepping with additive Runge-Kutta IMEX method

  Notes:
  The general system is written as

  F(t,X,Xdot) = G(t,X)

  where F represents the stiff part of the physics and G represents the non-stiff part.

*/
#include <private/tsimpl.h>                /*I   "petscts.h"   I*/

static const TSARKIMEXType TSARKIMEXDefault = TSARKIMEX2D;
static PetscBool TSARKIMEXRegisterAllCalled;
static PetscBool TSARKIMEXPackageInitialized;

typedef struct _ARKTableau *ARKTableau;
struct _ARKTableau {
  char *name;
  PetscInt order;
  PetscInt s;
  PetscReal *At,*bt,*ct;
  PetscReal *A,*b,*c;           /* Non-stiff tableau */
};
typedef struct _ARKTableauLink *ARKTableauLink;
struct _ARKTableauLink {
  struct _ARKTableau tab;
  ARKTableauLink next;
};
static ARKTableauLink ARKTableauList;

typedef struct {
  ARKTableau  tableau;
  Vec         *Y;               /* States computed during the step */
  Vec         *YdotI;           /* Time derivatives for the stiff part */
  Vec         *YdotRHS;         /* Function evaluations for the non-stiff part */
  Vec         Ydot;             /* Work vector holding Ydot during residual evaluation */
  Vec         Work;             /* Generic work vector */
  Vec         Z;                /* Ydot = shift(Y-Z) */
  PetscScalar *work;            /* Scalar work */
  PetscReal   shift;
  PetscReal   stage_time;
} TS_ARKIMEX;

#undef __FUNCT__
#define __FUNCT__ "TSARKIMEXRegisterAll"
/*@C
  TSARKIMEXRegisterAll - Registers all of the additive Runge-Kutta implicit-explicit methods in TSARKIMEX

  Not Collective

  Level: advanced

.keywords: TS, TSARKIMEX, register, all

.seealso:  TSARKIMEXRegisterDestroy()
@*/
PetscErrorCode TSARKIMEXRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (TSARKIMEXRegisterAllCalled) PetscFunctionReturn(0);
  TSARKIMEXRegisterAllCalled = PETSC_TRUE;

  {
    const PetscReal
      A[3][3] = {{0,0,0},
                 {0.41421356237309504880,0,0},
                 {0.75,0.25,0}},
      At[3][3] = {{0,0,0},
                  {0.12132034355964257320,0.29289321881345247560,0},
                  {0.20710678118654752440,0.50000000000000000000,0.29289321881345247560}};
      ierr = TSARKIMEXRegister(TSARKIMEX2D,2,3,&At[0][0],PETSC_NULL,PETSC_NULL,&A[0][0],PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  }
  {
    const PetscReal s2 = sqrt(2),
      A[3][3] = {{0,0,0},
                 {2-s2,0,0},
                 {(3-2*s2)/6,(3+2*s2)/6,0}},
      At[3][3] = {{0,0,0},
                  {1-1/s2,1-1/s2,0},
                  {1/(2*s2),1/(2*s2),1-1/s2}};
      ierr = TSARKIMEXRegister(TSARKIMEX2E,2,3,&At[0][0],PETSC_NULL,PETSC_NULL,&A[0][0],PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  }
  {
    const PetscReal
      A[4][4] = {{0,0,0,0},
                 {1767732205903./2027836641118,0,0,0},
                 {5535828885825./10492691773637,788022342437./10882634858940,0,0},
                 {6485989280629./16251701735622,-4246266847089./9704473918619,10755448449292./10357097424841,0}},
      At[4][4] = {{0,0,0,0},
                  {1767732205903./4055673282236,1767732205903./4055673282236,0,0},
                  {2746238789719./10658868560708,-640167445237./6845629431997,1767732205903./4055673282236,0},
                  {1471266399579./7840856788654,-4482444167858./7529755066697,11266239266428./11593286722821,1767732205903./4055673282236}};
      ierr = TSARKIMEXRegister(TSARKIMEX3,3,4,&At[0][0],PETSC_NULL,PETSC_NULL,&A[0][0],PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  }
  {
    const PetscReal
      A[6][6] = {{0,0,0,0,0,0},
                 {1./2,0,0,0,0,0},
                 {13861./62500,6889./62500,0,0,0,0},
                 {-116923316275./2393684061468,-2731218467317./15368042101831,9408046702089./11113171139209,0,0,0},
                 {-451086348788./2902428689909,-2682348792572./7519795681897,12662868775082./11960479115383,3355817975965./11060851509271,0,0},
                 {647845179188./3216320057751,73281519250./8382639484533,552539513391./3454668386233,3354512671639./8306763924573,4040./17871,0}},
      At[6][6] = {{0,0,0,0,0,0},
                  {1./4,1./4,0,0,0,0},
                  {8611./62500,-1743./31250,1./4,0,0,0},
                  {5012029./34652500,-654441./2922500,174375./388108,1./4,0,0},
                  {15267082809./155376265600,-71443401./120774400,730878875./902184768,2285395./8070912,1./4,0},
                  {82889./524892,0,15625./83664,69875./102672,-2260./8211,1./4}};
      ierr = TSARKIMEXRegister(TSARKIMEX4,4,6,&At[0][0],PETSC_NULL,PETSC_NULL,&A[0][0],PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  }
  {
    const PetscReal
      A[8][8] = {{0,0,0,0,0,0,0,0},
                 {41./100,0,0,0,0,0,0,0},
                 {367902744464./2072280473677,677623207551./8224143866563,0,0,0,0,0,0},
                 {1268023523408./10340822734521,0,1029933939417./13636558850479,0,0,0,0,0},
                 {14463281900351./6315353703477,0,66114435211212./5879490589093,-54053170152839./4284798021562,0,0,0,0},
                 {14090043504691./34967701212078,0,15191511035443./11219624916014,-18461159152457./12425892160975,-281667163811./9011619295870,0,0,0},
                 {19230459214898./13134317526959,0,21275331358303./2942455364971,-38145345988419./4862620318723,-1./8,-1./8,0,0},
                 {-19977161125411./11928030595625,0,-40795976796054./6384907823539,177454434618887./12078138498510,782672205425./8267701900261,-69563011059811./9646580694205,7356628210526./4942186776405,0}},
      At[8][8] = {{0,0,0,0,0,0,0,0},
                  {41./200,41./200,0,0,0,0,0,0},
                  {41./400,-567603406766./11931857230679,41./200,0,0,0,0,0},
                  {683785636431./9252920307686,0,-110385047103./1367015193373,41./200,0,0,0,0},
                  {3016520224154./10081342136671,0,30586259806659./12414158314087,-22760509404356./11113319521817,41./200,0,0,0},
                  {218866479029./1489978393911,0,638256894668./5436446318841,-1179710474555./5321154724896,-60928119172./8023461067671,41./200,0,0},
                  {1020004230633./5715676835656,0,25762820946817./25263940353407,-2161375909145./9755907335909,-211217309593./5846859502534,-4269925059573./7827059040749,41./200,0},
                  {-872700587467./9133579230613,0,0,22348218063261./9555858737531,-1143369518992./8141816002931,-39379526789629./19018526304540,32727382324388./42900044865799,41./200}};
      ierr = TSARKIMEXRegister(TSARKIMEX5,5,8,&At[0][0],PETSC_NULL,PETSC_NULL,&A[0][0],PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSARKIMEXRegisterDestroy"
/*@C
   TSARKIMEXRegisterDestroy - Frees the list of schemes that were registered by TSARKIMEXRegister().

   Not Collective

   Level: advanced

.keywords: TSARKIMEX, register, destroy
.seealso: TSARKIMEXRegister(), TSARKIMEXRegisterAll(), TSARKIMEXRegisterDynamic()
@*/
PetscErrorCode TSARKIMEXRegisterDestroy(void)
{
  PetscErrorCode ierr;
  ARKTableauLink link;

  PetscFunctionBegin;
  while ((link = ARKTableauList)) {
    ARKTableau t = &link->tab;
    ARKTableauList = link->next;
    ierr = PetscFree6(t->At,t->bt,t->ct,t->A,t->b,t->c);CHKERRQ(ierr);
    ierr = PetscFree(t->name);CHKERRQ(ierr);
    ierr = PetscFree(link);CHKERRQ(ierr);
  }
  TSARKIMEXRegisterAllCalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSARKIMEXInitializePackage"
/*@C
  TSARKIMEXInitializePackage - This function initializes everything in the TSARKIMEX package. It is called
  from PetscDLLibraryRegister() when using dynamic libraries, and on the first call to TSCreate_ARKIMEX()
  when using static libraries.

  Input Parameter:
  path - The dynamic library path, or PETSC_NULL

  Level: developer

.keywords: TS, TSARKIMEX, initialize, package
.seealso: PetscInitialize()
@*/
PetscErrorCode TSARKIMEXInitializePackage(const char path[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (TSARKIMEXPackageInitialized) PetscFunctionReturn(0);
  TSARKIMEXPackageInitialized = PETSC_TRUE;
  ierr = TSARKIMEXRegisterAll();CHKERRQ(ierr);
  ierr = PetscRegisterFinalize(TSARKIMEXFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSARKIMEXFinalizePackage"
/*@C
  TSARKIMEXFinalizePackage - This function destroys everything in the TSARKIMEX package. It is
  called from PetscFinalize().

  Level: developer

.keywords: Petsc, destroy, package
.seealso: PetscFinalize()
@*/
PetscErrorCode TSARKIMEXFinalizePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  TSARKIMEXPackageInitialized = PETSC_FALSE;
  ierr = TSARKIMEXRegisterDestroy();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSARKIMEXRegister"
PetscErrorCode TSARKIMEXRegister(const TSARKIMEXType name,PetscInt order,PetscInt s,
                                 const PetscReal At[],const PetscReal bt[],const PetscReal ct[],
                                 const PetscReal A[],const PetscReal b[],const PetscReal c[])
{
  PetscErrorCode ierr;
  ARKTableauLink link;
  ARKTableau t;
  PetscInt i,j;

  PetscFunctionBegin;
  ierr = PetscMalloc(sizeof(*link),&link);CHKERRQ(ierr);
  t = &link->tab;
  ierr = PetscStrallocpy(name,&t->name);CHKERRQ(ierr);
  t->order = order;
  t->s = s;
  ierr = PetscMalloc6(s*s,PetscReal,&t->At,s,PetscReal,&t->bt,s,PetscReal,&t->ct,s*s,PetscReal,&t->A,s,PetscReal,&t->b,s,PetscReal,&t->c);CHKERRQ(ierr);
  ierr = PetscMemcpy(t->At,At,s*s*sizeof(At[0]));CHKERRQ(ierr);
  ierr = PetscMemcpy(t->A,A,s*s*sizeof(A[0]));CHKERRQ(ierr);
  if (bt) {ierr = PetscMemcpy(t->bt,bt,s*sizeof(bt[0]));CHKERRQ(ierr);}
  else for (i=0; i<s; i++) t->bt[i] = At[(s-1)*s+i];
  if (b) {ierr = PetscMemcpy(t->b,b,s*sizeof(b[0]));CHKERRQ(ierr);}
  else for (i=0; i<s; i++) t->b[i] = At[(s-1)*s+i];
  if (ct) {ierr = PetscMemcpy(t->ct,ct,s*sizeof(ct[0]));CHKERRQ(ierr);}
  else for (i=0; i<s; i++) for (j=0,t->ct[i]=0; j<s; j++) t->ct[i] += At[i*s+j];
  if (c) {ierr = PetscMemcpy(t->c,c,s*sizeof(c[0]));CHKERRQ(ierr);}
  else for (i=0; i<s; i++) for (j=0,t->c[i]=0; j<s; j++) t->c[i] += A[i*s+j];
  link->next = ARKTableauList;
  ARKTableauList = link;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSStep_ARKIMEX"
static PetscErrorCode TSStep_ARKIMEX(TS ts)
{
  TS_ARKIMEX      *ark = (TS_ARKIMEX*)ts->data;
  ARKTableau      tab  = ark->tableau;
  const PetscInt  s    = tab->s;
  const PetscReal *At  = tab->At,*A = tab->A,*bt = tab->bt,*b = tab->b,*ct = tab->ct,*c = tab->c;
  PetscScalar     *w   = ark->work;
  Vec             *Y   = ark->Y,*YdotI = ark->YdotI,*YdotRHS = ark->YdotRHS,Ydot = ark->Ydot,W = ark->Work,Z = ark->Z;
  SNES            snes;
  PetscInt        i,j,its,lits;
  PetscReal       h,t;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  h = ts->time_step = ts->next_time_step;
  t = ts->ptime;

  for (i=0; i<s; i++) {
    if (At[i*s+i] == 0) {           /* This stage is explicit */
      ierr = VecCopy(ts->vec_sol,Y[i]);CHKERRQ(ierr);
      for (j=0; j<i; j++) w[j] = -h*At[i*s+j];
      ierr = VecMAXPY(Y[i],i,w,YdotI);CHKERRQ(ierr);
      for (j=0; j<i; j++) w[j] = h*A[i*s+j];
      ierr = VecMAXPY(Y[i],i,w,YdotRHS);CHKERRQ(ierr);
    } else {
      ark->stage_time = t + h*ct[i];
      ark->shift = 1./(h*At[i*s+i]);
      /* Affine part */
      ierr = VecZeroEntries(W);CHKERRQ(ierr);
      for (j=0; j<i; j++) w[j] = h*A[i*s+j];
      ierr = VecMAXPY(W,i,w,YdotRHS);CHKERRQ(ierr);
      /* Ydot = shift*(Y-Z) */
      ierr = VecCopy(ts->vec_sol,Z);CHKERRQ(ierr);
      for (j=0; j<i; j++) w[j] = h*At[i*s+j];
      ierr = VecMAXPY(Z,i,w,YdotRHS);CHKERRQ(ierr);
      /* Initial guess taken from last stage */
      ierr = VecCopy(i>0?Y[i-1]:ts->vec_sol,Y[i]);CHKERRQ(ierr);
      ierr = SNESSolve(snes,W,Y[i]);CHKERRQ(ierr);
      ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);
      ierr = SNESGetLinearSolveIterations(snes,&lits);CHKERRQ(ierr);
      ts->nonlinear_its += its; ts->linear_its += lits;
    }
    ierr = VecZeroEntries(Ydot);CHKERRQ(ierr);
    ierr = TSComputeIFunction(ts,t+h*ct[i],Y[i],Ydot,YdotI[i],PETSC_TRUE);CHKERRQ(ierr);
    ierr = TSComputeRHSFunction(ts,t+h*c[i],Y[i],YdotRHS[i]);CHKERRQ(ierr);
  }
  for (j=0; j<s; j++) w[j] = -h*bt[j];
  ierr = VecMAXPY(ts->vec_sol,s,w,YdotI);CHKERRQ(ierr);
  for (j=0; j<s; j++) w[j] = h*b[j];
  ierr = VecMAXPY(ts->vec_sol,s,w,YdotRHS);CHKERRQ(ierr);

  ts->ptime          += ts->time_step;
  ts->next_time_step  = ts->time_step;
  ts->steps++;
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "TSReset_ARKIMEX"
static PetscErrorCode TSReset_ARKIMEX(TS ts)
{
  TS_ARKIMEX      *ark = (TS_ARKIMEX*)ts->data;
  PetscInt        s;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (!ark->tableau) PetscFunctionReturn(0);
   s = ark->tableau->s;
  ierr = VecDestroyVecs(s,&ark->Y);CHKERRQ(ierr);
  ierr = VecDestroyVecs(s,&ark->YdotI);CHKERRQ(ierr);
  ierr = VecDestroyVecs(s,&ark->YdotRHS);CHKERRQ(ierr);
  ierr = VecDestroy(&ark->Ydot);CHKERRQ(ierr);
  ierr = VecDestroy(&ark->Work);CHKERRQ(ierr);
  ierr = VecDestroy(&ark->Z);CHKERRQ(ierr);
  ierr = PetscFree(ark->work);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSDestroy_ARKIMEX"
static PetscErrorCode TSDestroy_ARKIMEX(TS ts)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = TSReset_ARKIMEX(ts);CHKERRQ(ierr);
  ierr = PetscFree(ts->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSARKIMEXGetType_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSARKIMEXSetType_C","",PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  This defines the nonlinear equation that is to be solved with SNES
  G(U) = F[t0+Theta*dt, U, (U-U0)*shift] = 0
*/
#undef __FUNCT__
#define __FUNCT__ "SNESTSFormFunction_ARKIMEX"
static PetscErrorCode SNESTSFormFunction_ARKIMEX(SNES snes,Vec X,Vec F,TS ts)
{
  TS_ARKIMEX     *ark = (TS_ARKIMEX*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecAXPBYPCZ(ark->Ydot,-ark->shift,ark->shift,0,ark->Z,X);CHKERRQ(ierr); /* Ydot = shift*(X-Z) */
  ierr = TSComputeIFunction(ts,ark->stage_time,X,ark->Ydot,F,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SNESTSFormJacobian_ARKIMEX"
static PetscErrorCode SNESTSFormJacobian_ARKIMEX(SNES snes,Vec X,Mat *A,Mat *B,MatStructure *str,TS ts)
{
  TS_ARKIMEX       *ark = (TS_ARKIMEX*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* ark->Ydot has already been computed in SNESTSFormFunction_ARKIMEX (SNES guarantees this) */
  ierr = TSComputeIJacobian(ts,ark->stage_time,X,ark->Ydot,ark->shift,A,B,str,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSSetUp_ARKIMEX"
static PetscErrorCode TSSetUp_ARKIMEX(TS ts)
{
  TS_ARKIMEX     *ark = (TS_ARKIMEX*)ts->data;
  ARKTableau     tab  = ark->tableau;
  PetscInt       s = tab->s;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!ark->tableau) {
    ierr = TSARKIMEXSetType(ts,TSARKIMEXDefault);CHKERRQ(ierr);
  }
  ierr = VecDuplicateVecs(ts->vec_sol,s,&ark->Y);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(ts->vec_sol,s,&ark->YdotI);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(ts->vec_sol,s,&ark->YdotRHS);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&ark->Ydot);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&ark->Work);CHKERRQ(ierr);
  ierr = VecDuplicate(ts->vec_sol,&ark->Z);CHKERRQ(ierr);
  ierr = PetscMalloc(s*sizeof(ark->work[0]),&ark->work);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*------------------------------------------------------------*/

#undef __FUNCT__
#define __FUNCT__ "TSSetFromOptions_ARKIMEX"
static PetscErrorCode TSSetFromOptions_ARKIMEX(TS ts)
{
  PetscErrorCode ierr;
  char           arktype[256];

  PetscFunctionBegin;
  ierr = PetscOptionsHead("ARKIMEX ODE solver options");CHKERRQ(ierr);
  {
    ARKTableauLink link;
    PetscInt count,choice;
    PetscBool flg;
    const char **namelist;
    ierr = PetscStrncpy(arktype,TSARKIMEXDefault,sizeof arktype);CHKERRQ(ierr);
    for (link=ARKTableauList,count=0; link; link=link->next,count++) ;
    ierr = PetscMalloc(count*sizeof(char*),&namelist);CHKERRQ(ierr);
    for (link=ARKTableauList,count=0; link; link=link->next,count++) namelist[count] = link->tab.name;
    ierr = PetscOptionsEList("-ts_arkimex_type","Family of ARK IMEX method","TSARKIMEXSetType",(const char*const*)namelist,count,arktype,&choice,&flg);CHKERRQ(ierr);
    ierr = TSARKIMEXSetType(ts,flg ? namelist[choice] : arktype);CHKERRQ(ierr);
    ierr = PetscFree(namelist);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFormatRealArray"
static PetscErrorCode PetscFormatRealArray(char buf[],size_t len,const char *fmt,PetscInt n,const PetscReal x[])
{
  int i,left,count;
  char *p;

  PetscFunctionBegin;
  for (i=0,p=buf,left=(int)len; i<n; i++) {
    count = snprintf(p,left,fmt,x[i]);
    if (count < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SYS,"snprintf()");
    if (count >= left) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Insufficient space in buffer");
    left -= count;
    p += count;
    *p++ = ' ';
  }
  p[i ? 0 : -1] = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSView_ARKIMEX"
static PetscErrorCode TSView_ARKIMEX(TS ts,PetscViewer viewer)
{
  TS_ARKIMEX     *ark = (TS_ARKIMEX*)ts->data;
  ARKTableau     tab = ark->tableau;
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    const TSARKIMEXType arktype;
    char buf[512];
    ierr = TSARKIMEXGetType(ts,&arktype);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  ARK IMEX %s\n",arktype);CHKERRQ(ierr);
    ierr = PetscFormatRealArray(buf,sizeof buf,"% 8.6f",tab->s,tab->ct);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  Stiff abscissa       ct = %s\n",buf);CHKERRQ(ierr);
    ierr = PetscFormatRealArray(buf,sizeof buf,"% 8.6f",tab->s,tab->c);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  Nonstiff abscissa     c = %s\n",buf);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSARKIMEXSetType"
/*@C
  TSARKIMEXSetType - Set the type of ARK IMEX scheme

  Logically collective

  Input Parameter:
+  ts - timestepping context
-  arktype - type of ARK-IMEX scheme

  Level: intermediate

.seealso: TSARKIMEXGetType()
@*/
PetscErrorCode TSARKIMEXSetType(TS ts,const TSARKIMEXType arktype)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = PetscTryMethod(ts,"TSARKIMEXSetType_C",(TS,const TSARKIMEXType),(ts,arktype));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TSARKIMEXGetType"
/*@C
  TSARKIMEXGetType - Get the type of ARK IMEX scheme

  Logically collective

  Input Parameter:
.  ts - timestepping context

  Output Parameter:
.  arktype - type of ARK-IMEX scheme

  Level: intermediate

.seealso: TSARKIMEXGetType()
@*/
PetscErrorCode TSARKIMEXGetType(TS ts,const TSARKIMEXType *arktype)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  ierr = PetscUseMethod(ts,"TSARKIMEXGetType_C",(TS,const TSARKIMEXType*),(ts,arktype));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "TSARKIMEXGetType_ARKIMEX"
PetscErrorCode  TSARKIMEXGetType_ARKIMEX(TS ts,const TSARKIMEXType *arktype)
{
  TS_ARKIMEX *ark = (TS_ARKIMEX*)ts->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!ark->tableau) {ierr = TSARKIMEXSetType(ts,TSARKIMEXDefault);CHKERRQ(ierr);}
  *arktype = ark->tableau->name;
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "TSARKIMEXSetType_ARKIMEX"
PetscErrorCode  TSARKIMEXSetType_ARKIMEX(TS ts,const TSARKIMEXType arktype)
{
  TS_ARKIMEX *ark = (TS_ARKIMEX*)ts->data;
  PetscErrorCode ierr;
  PetscBool match;
  ARKTableauLink link;

  PetscFunctionBegin;
  if (ark->tableau) {
    ierr = PetscStrcmp(ark->tableau->name,arktype,&match);CHKERRQ(ierr);
    if (match) PetscFunctionReturn(0);
  }
  for (link = ARKTableauList; link; link=link->next) {
    ierr = PetscStrcmp(link->tab.name,arktype,&match);CHKERRQ(ierr);
    if (match) {
      ierr = TSReset_ARKIMEX(ts);CHKERRQ(ierr);
      ark->tableau = &link->tab;
      PetscFunctionReturn(0);
    }
  }
  SETERRQ1(((PetscObject)ts)->comm,PETSC_ERR_ARG_UNKNOWN_TYPE,"Could not find '%s'",arktype);
  PetscFunctionReturn(0);
}
EXTERN_C_END

/* ------------------------------------------------------------ */
/*MC
      TSARKIMEX - ODE solver using Additive Runge-Kutta IMEX schemes

  Level: beginner

.seealso:  TSCreate(), TS, TSSetType()

M*/
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "TSCreate_ARKIMEX"
PetscErrorCode  TSCreate_ARKIMEX(TS ts)
{
  TS_ARKIMEX       *th;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if !defined(PETSC_USE_DYNAMIC_LIBRARIES)
  ierr = TSARKIMEXInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif

  ts->ops->reset          = TSReset_ARKIMEX;
  ts->ops->destroy        = TSDestroy_ARKIMEX;
  ts->ops->view           = TSView_ARKIMEX;
  ts->ops->setup          = TSSetUp_ARKIMEX;
  ts->ops->step           = TSStep_ARKIMEX;
  ts->ops->setfromoptions = TSSetFromOptions_ARKIMEX;
  ts->ops->snesfunction   = SNESTSFormFunction_ARKIMEX;
  ts->ops->snesjacobian   = SNESTSFormJacobian_ARKIMEX;

  ierr = PetscNewLog(ts,TS_ARKIMEX,&th);CHKERRQ(ierr);
  ts->data = (void*)th;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSARKIMEXGetType_C","TSARKIMEXGetType_ARKIMEX",TSARKIMEXGetType_ARKIMEX);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ts,"TSARKIMEXSetType_C","TSARKIMEXSetType_ARKIMEX",TSARKIMEXSetType_ARKIMEX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
