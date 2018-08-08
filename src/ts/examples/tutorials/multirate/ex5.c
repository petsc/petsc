static const char help[] = "1D periodic Finite Volume solver in slope-limiter form with semidiscrete time stepping.\n"
  " advection   - Variable coefficient scalar advection\n"
  "                u_t       + (a*u)_x               = 0\n"
  " a is a piecewise function with two values say a0 and a1 (both a0 and a1 are constant).\n"
  " in this toy problem we assume a=a0 when x<0 and a=a1 when x>0 with a0<a1 which has the same discontinuous point with initial condition.\n"
  " we don't provide an exact solution, so you need to generate reference solution to do convergnece staudy,\n"
  " more precisely, you need firstly generate a reference to a binary file say file.bin, then on commend line,\n"
  " you should type -simulation -f file.bin.\n"
  " you can choose the number of grids by -da_grid_x.\n"
  " you can choose the value of a by -physics_advect_a1 and -physics_advect_a2.\n";

#include <petscts.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdraw.h>
#include <petsc/private/tsimpl.h>

#include <petsc/private/kernels/blockinvert.h> /* For the Kernel_*_gets_* stuff for BAIJ */

PETSC_STATIC_INLINE PetscReal Sgn(PetscReal a) { return (a<0) ? -1 : 1; }
PETSC_STATIC_INLINE PetscReal Abs(PetscReal a) { return (a<0) ? 0 : a; }
PETSC_STATIC_INLINE PetscReal Sqr(PetscReal a) { return a*a; }
PETSC_STATIC_INLINE PetscReal MaxAbs(PetscReal a,PetscReal b) { return (PetscAbs(a) > PetscAbs(b)) ? a : b; }
PETSC_UNUSED PETSC_STATIC_INLINE PetscReal MinAbs(PetscReal a,PetscReal b) { return (PetscAbs(a) < PetscAbs(b)) ? a : b; }
PETSC_STATIC_INLINE PetscReal MinMod2(PetscReal a,PetscReal b) { return (a*b<0) ? 0 : Sgn(a)*PetscMin(PetscAbs(a),PetscAbs(b)); }
PETSC_STATIC_INLINE PetscReal MaxMod2(PetscReal a,PetscReal b) { return (a*b<0) ? 0 : Sgn(a)*PetscMax(PetscAbs(a),PetscAbs(b)); }
PETSC_STATIC_INLINE PetscReal MinMod3(PetscReal a,PetscReal b,PetscReal c) {return (a*b<0 || a*c<0) ? 0 : Sgn(a)*PetscMin(PetscAbs(a),PetscMin(PetscAbs(b),PetscAbs(c))); }

PETSC_STATIC_INLINE PetscReal RangeMod(PetscReal a,PetscReal xmin,PetscReal xmax) { PetscReal range = xmax-xmin; return xmin +PetscFmodReal(range+PetscFmodReal(a,range),range); }


/* ----------------------- Lots of limiters, these could go in a separate library ------------------------- */
typedef struct _LimitInfo {
  PetscReal hx;
  PetscInt  m;
} *LimitInfo;
static void Limit_Upwind(LimitInfo info,const PetscScalar *jL,const PetscScalar *jR,PetscScalar *lmt)
{
  PetscInt i;
  for (i=0; i<info->m; i++) lmt[i] = 0;
}
static void Limit_LaxWendroff(LimitInfo info,const PetscScalar *jL,const PetscScalar *jR,PetscScalar *lmt)
{
  PetscInt i;
  for (i=0; i<info->m; i++) lmt[i] = jR[i];
}
static void Limit_BeamWarming(LimitInfo info,const PetscScalar *jL,const PetscScalar *jR,PetscScalar *lmt)
{
  PetscInt i;
  for (i=0; i<info->m; i++) lmt[i] = jL[i];
}
static void Limit_Fromm(LimitInfo info,const PetscScalar *jL,const PetscScalar *jR,PetscScalar *lmt)
{
  PetscInt i;
  for (i=0; i<info->m; i++) lmt[i] = 0.5*(jL[i] + jR[i]);
}
static void Limit_Minmod(LimitInfo info,const PetscScalar *jL,const PetscScalar *jR,PetscScalar *lmt)
{
  PetscInt i;
  for (i=0; i<info->m; i++) lmt[i] = MinMod2(jL[i],jR[i]);
}
static void Limit_Superbee(LimitInfo info,const PetscScalar *jL,const PetscScalar *jR,PetscScalar *lmt)
{
  PetscInt i;
  for (i=0; i<info->m; i++) lmt[i] = MaxMod2(MinMod2(jL[i],2*jR[i]),MinMod2(2*jL[i],jR[i]));
}
static void Limit_MC(LimitInfo info,const PetscScalar *jL,const PetscScalar *jR,PetscScalar *lmt)
{
  PetscInt i;
  for (i=0; i<info->m; i++) lmt[i] = MinMod3(2*jL[i],0.5*(jL[i]+jR[i]),2*jR[i]);
}
static void Limit_VanLeer(LimitInfo info,const PetscScalar *jL,const PetscScalar *jR,PetscScalar *lmt)
{ /* phi = (t + abs(t)) / (1 + abs(t)) */
  PetscInt i;
  for (i=0; i<info->m; i++) lmt[i] = (jL[i]*Abs(jR[i]) + Abs(jL[i])*jR[i]) / (Abs(jL[i]) + Abs(jR[i]) + 1e-15);
}
static void Limit_VanAlbada(LimitInfo info,const PetscScalar *jL,const PetscScalar *jR,PetscScalar *lmt) /* differentiable */
{ /* phi = (t + t^2) / (1 + t^2) */
  PetscInt i;
  for (i=0; i<info->m; i++) lmt[i] = (jL[i]*Sqr(jR[i]) + Sqr(jL[i])*jR[i]) / (Sqr(jL[i]) + Sqr(jR[i]) + 1e-15);
}
static void Limit_VanAlbadaTVD(LimitInfo info,const PetscScalar *jL,const PetscScalar *jR,PetscScalar *lmt)
{ /* phi = (t + t^2) / (1 + t^2) */
  PetscInt i;
  for (i=0; i<info->m; i++) lmt[i] = (jL[i]*jR[i]<0) ? 0 : (jL[i]*Sqr(jR[i]) + Sqr(jL[i])*jR[i]) / (Sqr(jL[i]) + Sqr(jR[i]) + 1e-15);
}
static void Limit_Koren(LimitInfo info,const PetscScalar *jL,const PetscScalar *jR,PetscScalar *lmt) /* differentiable */
{ /* phi = (t + 2*t^2) / (2 - t + 2*t^2) */
  PetscInt i;
  for (i=0; i<info->m; i++) lmt[i] = ((jL[i]*Sqr(jR[i]) + 2*Sqr(jL[i])*jR[i])/(2*Sqr(jL[i]) - jL[i]*jR[i] + 2*Sqr(jR[i]) + 1e-15));
}
static void Limit_KorenSym(LimitInfo info,const PetscScalar *jL,const PetscScalar *jR,PetscScalar *lmt) /* differentiable */
{ /* Symmetric version of above */
  PetscInt i;
  for (i=0; i<info->m; i++) lmt[i] = (1.5*(jL[i]*Sqr(jR[i]) + Sqr(jL[i])*jR[i])/(2*Sqr(jL[i]) - jL[i]*jR[i] + 2*Sqr(jR[i]) + 1e-15));
}
static void Limit_Koren3(LimitInfo info,const PetscScalar *jL,const PetscScalar *jR,PetscScalar *lmt)
{ /* Eq 11 of Cada-Torrilhon 2009 */
  PetscInt i;
  for (i=0; i<info->m; i++) lmt[i] = MinMod3(2*jL[i],(jL[i]+2*jR[i])/3,2*jR[i]);
}
static PetscReal CadaTorrilhonPhiHatR_Eq13(PetscReal L,PetscReal R)
{
  return PetscMax(0,PetscMin((L+2*R)/3,PetscMax(-0.5*L,PetscMin(2*L,PetscMin((L+2*R)/3,1.6*R)))));
}
static void Limit_CadaTorrilhon2(LimitInfo info,const PetscScalar *jL,const PetscScalar *jR,PetscScalar *lmt)
{ /* Cada-Torrilhon 2009, Eq 13 */
  PetscInt i;
  for (i=0; i<info->m; i++) lmt[i] = CadaTorrilhonPhiHatR_Eq13(jL[i],jR[i]);
}
static void Limit_CadaTorrilhon3R(PetscReal r,LimitInfo info,const PetscScalar *jL,const PetscScalar *jR,PetscScalar *lmt)
{ /* Cada-Torrilhon 2009, Eq 22 */
  /* They recommend 0.001 < r < 1, but larger values are more accurate in smooth regions */
  const PetscReal eps = 1e-7,hx = info->hx;
  PetscInt        i;
  for (i=0; i<info->m; i++) {
    const PetscReal eta = (Sqr(jL[i]) + Sqr(jR[i])) / Sqr(r*hx);
    lmt[i] = ((eta < 1-eps) ? (jL[i] + 2*jR[i]) / 3 : ((eta > 1+eps) ? CadaTorrilhonPhiHatR_Eq13(jL[i],jR[i]) : 0.5*((1-(eta-1)/eps)*(jL[i]+2*jR[i])/3 + (1+(eta+1)/eps)*CadaTorrilhonPhiHatR_Eq13(jL[i],jR[i]))));
  }
}
static void Limit_CadaTorrilhon3R0p1(LimitInfo info,const PetscScalar *jL,const PetscScalar *jR,PetscScalar *lmt)
{
  Limit_CadaTorrilhon3R(0.1,info,jL,jR,lmt);
}
static void Limit_CadaTorrilhon3R1(LimitInfo info,const PetscScalar *jL,const PetscScalar *jR,PetscScalar *lmt)
{
  Limit_CadaTorrilhon3R(1,info,jL,jR,lmt);
}
static void Limit_CadaTorrilhon3R10(LimitInfo info,const PetscScalar *jL,const PetscScalar *jR,PetscScalar *lmt)
{
  Limit_CadaTorrilhon3R(10,info,jL,jR,lmt);
}
static void Limit_CadaTorrilhon3R100(LimitInfo info,const PetscScalar *jL,const PetscScalar *jR,PetscScalar *lmt)
{
  Limit_CadaTorrilhon3R(100,info,jL,jR,lmt);
}


/* --------------------------------- Finite Volume data structures ----------------------------------- */

typedef enum {FVBC_PERIODIC, FVBC_OUTFLOW} FVBCType;
static const char *FVBCTypes[] = {"PERIODIC","OUTFLOW","FVBCType","FVBC_",0};
/* we add three new variables at the end of input parameters of function to be position of cell center, left bounday of domain, right boundary fo domain */
typedef PetscErrorCode (*RiemannFunction)(void*,PetscInt,const PetscScalar*,const PetscScalar*,PetscScalar*,PetscReal*,PetscReal,PetscReal,PetscReal);
typedef PetscErrorCode (*ReconstructFunction)(void*,PetscInt,const PetscScalar*,PetscScalar*,PetscScalar*,PetscReal*,PetscReal);

typedef struct {
  PetscErrorCode      (*sample)(void*,PetscInt,FVBCType,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal*);
  RiemannFunction     riemann;
  ReconstructFunction characteristic;
  PetscErrorCode      (*destroy)(void*);
  void                *user;
  PetscInt            dof;
  char                *fieldname[16];
} PhysicsCtx;

typedef struct {
  void        (*limit)(LimitInfo,const PetscScalar*,const PetscScalar*,PetscScalar*);
  PhysicsCtx  physics;
  MPI_Comm    comm;
  char        prefix[256];

  /* Local work arrays */
  PetscScalar *R,*Rinv;         /* Characteristic basis, and it's inverse.  COLUMN-MAJOR */
  PetscScalar *cjmpLR;          /* Jumps at left and right edge of cell, in characteristic basis, len=2*dof */
  PetscScalar *cslope;          /* Limited slope, written in characteristic basis */
  PetscScalar *uLR;             /* Solution at left and right of interface, conservative variables, len=2*dof */
  PetscScalar *flux;            /* Flux across interface */
  PetscReal   *speeds;          /* Speeds of each wave */

  PetscReal   cfl_idt;          /* Max allowable value of 1/Delta t */
  PetscReal   cfl;
  PetscReal   xmin,xmax;
  PetscInt    initial;
  PetscBool   simulation;
  FVBCType    bctype;
} FVCtx;

PetscErrorCode RiemannListAdd(PetscFunctionList *flist,const char *name,RiemannFunction rsolve)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscFunctionListAdd(flist,name,rsolve);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode RiemannListFind(PetscFunctionList flist,const char *name,RiemannFunction *rsolve)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscFunctionListFind(flist,name,rsolve);CHKERRQ(ierr);
  if (!*rsolve) SETERRQ1(PETSC_COMM_SELF,1,"Riemann solver \"%s\" could not be found",name);
  PetscFunctionReturn(0);
}

PetscErrorCode ReconstructListAdd(PetscFunctionList *flist,const char *name,ReconstructFunction r)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscFunctionListAdd(flist,name,r);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode ReconstructListFind(PetscFunctionList flist,const char *name,ReconstructFunction *r)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscFunctionListFind(flist,name,r);CHKERRQ(ierr);
  if (!*r) SETERRQ1(PETSC_COMM_SELF,1,"Reconstruction \"%s\" could not be found",name);
  PetscFunctionReturn(0);
}

/* --------------------------------- Physics ----------------------------------- */

static PetscErrorCode PhysicsDestroy_SimpleFree(void *vctx)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscFree(vctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* --------------------------------- Advection ----------------------------------- */

typedef struct {
  PetscReal a[2];                  /* advective velocity */
} AdvectCtx;

static PetscErrorCode PhysicsRiemann_Advect(void *vctx,PetscInt m,const PetscScalar *uL,const PetscScalar *uR,PetscScalar *flux,PetscReal *maxspeed,PetscReal x,PetscReal xmin, PetscReal xmax)
{
  AdvectCtx *ctx = (AdvectCtx*)vctx;
  PetscReal *speed;

  PetscFunctionBeginUser;
  speed = ctx->a;
  if(x==0 || x == xmin || x == xmax) flux[0] = PetscMax(0,(speed[0]+speed[1])/2.0)*uL[0] + PetscMin(0,(speed[0]+speed[1])/2.0)*uR[0]; /* if condition need to be changed base on different problem, '0' is the discontinuous point of a */
  else if(x<0) flux[0] = PetscMax(0,speed[0])*uL[0] + PetscMin(0,speed[0])*uR[0];  /* else if condition need to be changed based on diferent problem, 'x = 0' is discontinuous point of a */
  else flux[0] = PetscMax(0,speed[1])*uL[0] + PetscMin(0,speed[1])*uR[0];
  *maxspeed = *speed;
  PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsCharacteristic_Advect(void *vctx,PetscInt m,const PetscScalar *u,PetscScalar *X,PetscScalar *Xi,PetscReal *speeds,PetscReal x)
{
  AdvectCtx *ctx = (AdvectCtx*)vctx;

  PetscFunctionBeginUser;
  X[0]      = 1.;
  Xi[0]     = 1.;
  if(x<0) speeds[0] = ctx->a[0];  /* x is discontinuous point of a */
  else    speeds[0] = ctx->a[1];
  PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsSample_Advect(void *vctx,PetscInt initial,FVBCType bctype,PetscReal xmin,PetscReal xmax,PetscReal t,PetscReal x,PetscReal *u)
{
  AdvectCtx *ctx = (AdvectCtx*)vctx;
  PetscReal *a    = ctx->a,x0;

  PetscFunctionBeginUser;
  if(x<0){   /* x is cell center */
    switch (bctype) {
      case FVBC_OUTFLOW:  x0 = x-a[0]*t; break;
      case FVBC_PERIODIC: x0 = RangeMod(x-a[0]*t,xmin,xmax); break;
      default: SETERRQ(PETSC_COMM_SELF,1,"unknown BCType");
    }
    switch (initial) {
      case 0: u[0] = (x0 < 0) ? 1 : -1; break;
      case 1: u[0] = (x0 < 0) ? -1 : 1; break;
      case 2: u[0] = (0 < x0 && x0 < 1) ? 1 : 0; break;
      case 3: u[0] = PetscSinReal(2*PETSC_PI*x0); break;
      case 4: u[0] = PetscAbs(x0); break;
      case 5: u[0] = (x0 < 0 || x0 > 0.5) ? 0 : PetscSqr(PetscSinReal(2*PETSC_PI*x0)); break;
      case 6: u[0] = (x0 < 0) ? 0 : ((x0 < 1) ? x0 : ((x0 < 2) ? 2-x0 : 0)); break;
      case 7: u[0] = PetscPowReal(PetscSinReal(PETSC_PI*x0),10.0);break;
      default: SETERRQ(PETSC_COMM_SELF,1,"unknown initial condition");
    }
  }
  else{
    switch (bctype) {
      case FVBC_OUTFLOW:  x0 = x-a[1]*t; break;
      case FVBC_PERIODIC: x0 = RangeMod(x-a[1]*t,xmin,xmax); break;
      default: SETERRQ(PETSC_COMM_SELF,1,"unknown BCType");
    }
    switch (initial) {
      case 0: u[0] = (x0 < 0) ? 1 : -1; break;
      case 1: u[0] = (x0 < 0) ? -1 : 1; break;
      case 2: u[0] = (0 < x0 && x0 < 1) ? 1 : 0; break;
      case 3: u[0] = PetscSinReal(2*PETSC_PI*x0); break;
      case 4: u[0] = PetscAbs(x0); break;
      case 5: u[0] = (x0 < 0 || x0 > 0.5) ? 0 : PetscSqr(PetscSinReal(2*PETSC_PI*x0)); break;
      case 6: u[0] = (x0 < 0) ? 0 : ((x0 < 1) ? x0 : ((x0 < 2) ? 2-x0 : 0)); break;
      case 7: u[0] = PetscPowReal(PetscSinReal(PETSC_PI*x0),10.0);break;
      default: SETERRQ(PETSC_COMM_SELF,1,"unknown initial condition");
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsCreate_Advect(FVCtx *ctx)
{
  PetscErrorCode ierr;
  AdvectCtx      *user;

  PetscFunctionBeginUser;
  ierr = PetscNew(&user);CHKERRQ(ierr);
  ctx->physics.sample         = PhysicsSample_Advect;
  ctx->physics.riemann        = PhysicsRiemann_Advect;
  ctx->physics.characteristic = PhysicsCharacteristic_Advect;
  ctx->physics.destroy        = PhysicsDestroy_SimpleFree;
  ctx->physics.user           = user;
  ctx->physics.dof            = 1;
  ierr = PetscStrallocpy("u",&ctx->physics.fieldname[0]);CHKERRQ(ierr);
  user->a[0] = 1;
  user->a[1] = 1;
  ierr = PetscOptionsBegin(ctx->comm,ctx->prefix,"Options for advection","");CHKERRQ(ierr);
  {
    ierr = PetscOptionsReal("-physics_advect_a1","Speed1","",user->a[0],&user->a[0],NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-physics_advect_a2","Speed2","",user->a[1],&user->a[1],NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* --------------------------------- Finite Volume Solver ----------------------------------- */

static PetscErrorCode FVRHSFunction(TS ts,PetscReal time,Vec X,Vec F,void *vctx)
{
  FVCtx             *ctx = (FVCtx*)vctx;
  PetscErrorCode    ierr;
  PetscInt          i,j,k,Mx,dof,xs,xm;
  PetscReal         hx,cfl_idt = 0;
  PetscScalar       *x,*f,*slope;
  Vec               Xloc;
  DM                da;

  PetscFunctionBeginUser;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&Xloc);CHKERRQ(ierr);                          /* Xloc contains ghost points */
  ierr = DMDAGetInfo(da,0, &Mx,0,0, 0,0,0, &dof,0,0,0,0,0);CHKERRQ(ierr);   /* Mx is the number of center points */
  hx   = (ctx->xmax - ctx->xmin)/Mx;
  ierr = DMGlobalToLocalBegin(da,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);       /* X is solution vector which does not contain ghost points */
  ierr = DMGlobalToLocalEnd  (da,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
  ierr = VecZeroEntries(F);CHKERRQ(ierr);                                   /* F is the right hand side function corresponds to center points */
  ierr = DMDAVecGetArray(da,Xloc,&x);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,F,&f);CHKERRQ(ierr);
  ierr = DMDAGetArray(da,PETSC_TRUE,&slope);CHKERRQ(ierr);                  /* contains ghost points */
  ierr = DMDAGetCorners(da,&xs,0,0,&xm,0,0);CHKERRQ(ierr);

  if (ctx->bctype == FVBC_OUTFLOW) {
    for (i=xs-2; i<0; i++) {
      for (j=0; j<dof; j++) x[i*dof+j] = x[j];
    }
    for (i=Mx; i<xs+xm+2; i++) {
      for (j=0; j<dof; j++) x[i*dof+j] = x[(xs+xm-1)*dof+j];
    }
  }
  for (i=xs-1; i<xs+xm+1; i++) {
    struct _LimitInfo info;
    PetscScalar       *cjmpL,*cjmpR;
    /* Determine the right eigenvectors R, where A = R \Lambda R^{-1} */
    ierr = (*ctx->physics.characteristic)(ctx->physics.user,dof,&x[i*dof],ctx->R,ctx->Rinv,ctx->speeds,ctx->xmin+hx*i);CHKERRQ(ierr);
    /* Evaluate jumps across interfaces (i-1, i) and (i, i+1), put in characteristic basis */
    ierr  = PetscMemzero(ctx->cjmpLR,2*dof*sizeof(ctx->cjmpLR[0]));CHKERRQ(ierr);
    cjmpL = &ctx->cjmpLR[0];
    cjmpR = &ctx->cjmpLR[dof];
    for (j=0; j<dof; j++) {
      PetscScalar jmpL,jmpR;
      jmpL = x[(i+0)*dof+j] - x[(i-1)*dof+j];
      jmpR = x[(i+1)*dof+j] - x[(i+0)*dof+j];
      for (k=0; k<dof; k++) {
        cjmpL[k] += ctx->Rinv[k+j*dof] * jmpL;
        cjmpR[k] += ctx->Rinv[k+j*dof] * jmpR;
      }
    }
    /* Apply limiter to the left and right characteristic jumps */
    info.m  = dof;
    info.hx = hx;
    (*ctx->limit)(&info,cjmpL,cjmpR,ctx->cslope);
    for (j=0; j<dof; j++) ctx->cslope[j] /= hx; /* rescale to a slope */
    for (j=0; j<dof; j++) {
      PetscScalar tmp = 0;
      for (k=0; k<dof; k++) tmp += ctx->R[j+k*dof] * ctx->cslope[k];
      slope[i*dof+j] = tmp;
    }
  }

  for (i=xs; i<xs+xm+1; i++) {
    PetscReal   maxspeed;
    PetscScalar *uL,*uR;
    uL = &ctx->uLR[0];
    uR = &ctx->uLR[dof];
    for (j=0; j<dof; j++) {
      uL[j] = x[(i-1)*dof+j] + slope[(i-1)*dof+j]*hx/2;
      uR[j] = x[(i-0)*dof+j] - slope[(i-0)*dof+j]*hx/2;
    }
    ierr    = (*ctx->physics.riemann)(ctx->physics.user,dof,uL,uR,ctx->flux,&maxspeed,ctx->xmin+hx*i,ctx->xmin,ctx->xmax);CHKERRQ(ierr);
    cfl_idt = PetscMax(cfl_idt,PetscAbsScalar(maxspeed/hx)); /* Max allowable value of 1/Delta t */

    if (i > xs) {
      for (j=0; j<dof; j++) f[(i-1)*dof+j] -= ctx->flux[j]/hx;
    }
    if (i < xs+xm) {
      for (j=0; j<dof; j++) f[i*dof+j] += ctx->flux[j]/hx;
    }
  }
  ierr = DMDAVecRestoreArray(da,Xloc,&x);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,F,&f);CHKERRQ(ierr);
  ierr = DMDARestoreArray(da,PETSC_TRUE,&slope);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&Xloc);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&cfl_idt,&ctx->cfl_idt,1,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)da));CHKERRQ(ierr);
  if (0) {
    /* We need to a way to inform the TS of a CFL constraint, this is a debugging fragment */
    PetscReal dt,tnow;
    ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
    ierr = TSGetTime(ts,&tnow);CHKERRQ(ierr);
    if (dt > 0.5/ctx->cfl_idt) {
      if (1) {
        ierr = PetscPrintf(ctx->comm,"Stability constraint exceeded at t=%g, dt %g > %g\n",(double)tnow,(double)dt,(double)(0.5/ctx->cfl_idt));CHKERRQ(ierr);
      } else SETERRQ2(PETSC_COMM_SELF,1,"Stability constraint exceeded, %g > %g",(double)dt,(double)(ctx->cfl/ctx->cfl_idt));
    }
  }
  PetscFunctionReturn(0);
}

/* --------------------------------- Finite Volume Solver for slow parts ----------------------------------- */

static PetscErrorCode FVRHSFunctionslow(TS ts,PetscReal time,Vec X,Vec F,void *vctx)
{
  FVCtx             *ctx = (FVCtx*)vctx;
  PetscErrorCode    ierr;
  PetscInt          i,j,k,Mx,dof,xs,xm,len_slow;
  PetscReal         hx,cfl_idt = 0;
  PetscScalar       *x,*f,*slope;
  Vec               Xloc;
  DM                da;

  PetscFunctionBeginUser;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&Xloc);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,0, &Mx,0,0, 0,0,0, &dof,0,0,0,0,0);CHKERRQ(ierr);
  hx   = (ctx->xmax - ctx->xmin)/Mx;
  ierr = DMGlobalToLocalBegin(da,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (da,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
  ierr = VecZeroEntries(F);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,Xloc,&x);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  ierr = DMDAGetArray(da,PETSC_TRUE,&slope);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&xs,0,0,&xm,0,0);CHKERRQ(ierr);
  ierr = ISGetSize(ts->iss,&len_slow);CHKERRQ(ierr);

  if (ctx->bctype == FVBC_OUTFLOW) {
    for (i=xs-2; i<0; i++) {
      for (j=0; j<dof; j++) x[i*dof+j] = x[j];
    }
    for (i=Mx; i<xs+xm+2; i++) {
      for (j=0; j<dof; j++) x[i*dof+j] = x[(xs+xm-1)*dof+j];
    }
  }
  for (i=xs-1; i<xs+xm+1; i++) {
    struct _LimitInfo info;
    PetscScalar       *cjmpL,*cjmpR;
    if(i<len_slow+1){
      /* Determine the right eigenvectors R, where A = R \Lambda R^{-1} */
      ierr = (*ctx->physics.characteristic)(ctx->physics.user,dof,&x[i*dof],ctx->R,ctx->Rinv,ctx->speeds,ctx->xmin+hx*i);CHKERRQ(ierr);
      /* Evaluate jumps across interfaces (i-1, i) and (i, i+1), put in characteristic basis */
      ierr  = PetscMemzero(ctx->cjmpLR,2*dof*sizeof(ctx->cjmpLR[0]));CHKERRQ(ierr);
      cjmpL = &ctx->cjmpLR[0];
      cjmpR = &ctx->cjmpLR[dof];
      for (j=0; j<dof; j++) {
        PetscScalar jmpL,jmpR;
        jmpL = x[(i+0)*dof+j] - x[(i-1)*dof+j];
        jmpR = x[(i+1)*dof+j] - x[(i+0)*dof+j];
        for (k=0; k<dof; k++) {
          cjmpL[k] += ctx->Rinv[k+j*dof] * jmpL;
          cjmpR[k] += ctx->Rinv[k+j*dof] * jmpR;
        }
      }
      /* Apply limiter to the left and right characteristic jumps */
      info.m  = dof;
      info.hx = hx;
      (*ctx->limit)(&info,cjmpL,cjmpR,ctx->cslope);
      for (j=0; j<dof; j++) ctx->cslope[j] /= hx; /* rescale to a slope */
      for (j=0; j<dof; j++) {
      PetscScalar tmp = 0;
      for (k=0; k<dof; k++) tmp += ctx->R[j+k*dof] * ctx->cslope[k];
        slope[i*dof+j] = tmp;
      }
    }
  }

  for (i=xs; i<xs+xm+1; i++) {
    PetscReal   maxspeed;
    PetscScalar *uL,*uR;
    if(i<len_slow+1){ /* slow parts can be changed based on a */
      uL = &ctx->uLR[0];
      uR = &ctx->uLR[dof];
      for (j=0; j<dof; j++) {
        uL[j] = x[(i-1)*dof+j] + slope[(i-1)*dof+j]*hx/2;
        uR[j] = x[(i-0)*dof+j] - slope[(i-0)*dof+j]*hx/2;
      }
      ierr    = (*ctx->physics.riemann)(ctx->physics.user,dof,uL,uR,ctx->flux,&maxspeed,ctx->xmin+hx*i,ctx->xmin,ctx->xmax);CHKERRQ(ierr);
      cfl_idt = PetscMax(cfl_idt,PetscAbsScalar(maxspeed/hx)); /* Max allowable value of 1/Delta t */

      if (i > xs) {
        for (j=0; j<dof; j++) f[(i-1)*dof+j] -= ctx->flux[j]/hx;
      }
      if (i < len_slow) {
        for (j=0; j<dof; j++) f[i*dof+j] += ctx->flux[j]/hx;
      }
    }
  }

  ierr = DMDAVecRestoreArray(da,Xloc,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  ierr = DMDARestoreArray(da,PETSC_TRUE,&slope);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&Xloc);CHKERRQ(ierr);

  ierr = MPI_Allreduce(&cfl_idt,&ctx->cfl_idt,1,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)da));CHKERRQ(ierr);
  if (0) {
    /* We need to a way to inform the TS of a CFL constraint, this is a debugging fragment */
    PetscReal dt,tnow;
    ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
    ierr = TSGetTime(ts,&tnow);CHKERRQ(ierr);
    if (dt > 0.5/ctx->cfl_idt) {
      if (1) {
        ierr = PetscPrintf(ctx->comm,"Stability constraint exceeded at t=%g, dt %g > %g\n",(double)tnow,(double)dt,(double)(0.5/ctx->cfl_idt));CHKERRQ(ierr);
      } else SETERRQ2(PETSC_COMM_SELF,1,"Stability constraint exceeded, %g > %g",(double)dt,(double)(ctx->cfl/ctx->cfl_idt));
    }
  }
  PetscFunctionReturn(0);
}

/* --------------------------------- Finite Volume Solver for fast  parts ----------------------------------- */

static PetscErrorCode FVRHSFunctionfast(TS ts,PetscReal time,Vec X,Vec F,void *vctx)
{
  FVCtx             *ctx = (FVCtx*)vctx;
  PetscErrorCode    ierr;
  PetscInt          i,j,k,Mx,dof,xs,xm,len_slow;
  PetscReal         hx,cfl_idt = 0;
  PetscScalar       *x,*f,*slope;
  Vec               Xloc;
  DM                da;

  PetscFunctionBeginUser;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&Xloc);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,0, &Mx,0,0, 0,0,0, &dof,0,0,0,0,0);CHKERRQ(ierr);
  hx   = (ctx->xmax - ctx->xmin)/Mx;
  ierr = DMGlobalToLocalBegin(da,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd  (da,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
  ierr = VecZeroEntries(F);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,Xloc,&x);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  ierr = DMDAGetArray(da,PETSC_TRUE,&slope);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&xs,0,0,&xm,0,0);CHKERRQ(ierr);
  ierr = ISGetSize(ts->iss,&len_slow);CHKERRQ(ierr);
  if (ctx->bctype == FVBC_OUTFLOW) {
    for (i=xs-2; i<0; i++) {
      for (j=0; j<dof; j++) x[i*dof+j] = x[j];
    }
    for (i=Mx; i<xs+xm+2; i++) {
      for (j=0; j<dof; j++) x[i*dof+j] = x[(xs+xm-1)*dof+j];
    }
  }
  for (i=xs-1; i<xs+xm+1; i++) {
    struct _LimitInfo info;
    PetscScalar       *cjmpL,*cjmpR;
    if(i>len_slow-2){
      ierr = (*ctx->physics.characteristic)(ctx->physics.user,dof,&x[i*dof],ctx->R,ctx->Rinv,ctx->speeds,ctx->xmin+hx*i);CHKERRQ(ierr);
      ierr  = PetscMemzero(ctx->cjmpLR,2*dof*sizeof(ctx->cjmpLR[0]));CHKERRQ(ierr);
      cjmpL = &ctx->cjmpLR[0];
      cjmpR = &ctx->cjmpLR[dof];
      for (j=0; j<dof; j++) {
        PetscScalar jmpL,jmpR;
        jmpL = x[(i+0)*dof+j] - x[(i-1)*dof+j];
        jmpR = x[(i+1)*dof+j] - x[(i+0)*dof+j];
        for (k=0; k<dof; k++) {
          cjmpL[k] += ctx->Rinv[k+j*dof] * jmpL;
          cjmpR[k] += ctx->Rinv[k+j*dof] * jmpR;
        }
      }
      /* Apply limiter to the left and right characteristic jumps */
      info.m  = dof;
      info.hx = hx;
      (*ctx->limit)(&info,cjmpL,cjmpR,ctx->cslope);
      for (j=0; j<dof; j++) ctx->cslope[j] /= hx; /* rescale to a slope */
      for (j=0; j<dof; j++) {
      PetscScalar tmp = 0;
      for (k=0; k<dof; k++) tmp += ctx->R[j+k*dof] * ctx->cslope[k];
        slope[i*dof+j] = tmp;
      }
    }
  }

  for (i=xs; i<xs+xm+1; i++) {
    PetscReal   maxspeed;
    PetscScalar *uL,*uR;
    if(i>len_slow-1){
      uL = &ctx->uLR[0];
      uR = &ctx->uLR[dof];
      for (j=0; j<dof; j++) {
        uL[j] = x[(i-1)*dof+j] + slope[(i-1)*dof+j]*hx/2;
        uR[j] = x[(i-0)*dof+j] - slope[(i-0)*dof+j]*hx/2;
      }
      ierr    = (*ctx->physics.riemann)(ctx->physics.user,dof,uL,uR,ctx->flux,&maxspeed,ctx->xmin+hx*i,ctx->xmin,ctx->xmax);CHKERRQ(ierr);
      cfl_idt = PetscMax(cfl_idt,PetscAbsScalar(maxspeed/hx)); /* Max allowable value of 1/Delta t */

      if (i > xs && i!= len_slow) {
        for (j=0; j<dof; j++) f[(i-len_slow-1)*dof+j] -= ctx->flux[j]/hx;
      }
      if (i < xs+xm) {
        for (j=0; j<dof; j++) f[(i-len_slow)*dof+j] += ctx->flux[j]/hx;
      }
    }
  }
  ierr = DMDAVecRestoreArray(da,Xloc,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  ierr = DMDARestoreArray(da,PETSC_TRUE,&slope);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&Xloc);CHKERRQ(ierr);
  ierr = MPI_Allreduce(&cfl_idt,&ctx->cfl_idt,1,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)da));CHKERRQ(ierr);
  if (0) {
    /* We need to a way to inform the TS of a CFL constraint, this is a debugging fragment */
    PetscReal dt,tnow;
    ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
    ierr = TSGetTime(ts,&tnow);CHKERRQ(ierr);
    if (dt > 0.5/ctx->cfl_idt) {
      if (1) {
        ierr = PetscPrintf(ctx->comm,"Stability constraint exceeded at t=%g, dt %g > %g\n",(double)tnow,(double)dt,(double)(0.5/ctx->cfl_idt));CHKERRQ(ierr);
      } else SETERRQ2(PETSC_COMM_SELF,1,"Stability constraint exceeded, %g > %g",(double)dt,(double)(ctx->cfl/ctx->cfl_idt));
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode FVSample(FVCtx *ctx,DM da,PetscReal time,Vec U)
{
  PetscErrorCode ierr;
  PetscScalar    *u,*uj;
  PetscInt       i,j,k,dof,xs,xm,Mx;

  PetscFunctionBeginUser;
  if (!ctx->physics.sample) SETERRQ(PETSC_COMM_SELF,1,"Physics has not provided a sampling function");
  ierr = DMDAGetInfo(da,0, &Mx,0,0, 0,0,0, &dof,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&xs,0,0,&xm,0,0);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,U,&u);CHKERRQ(ierr);
  ierr = PetscMalloc1(dof,&uj);CHKERRQ(ierr);
  for (i=xs; i<xs+xm; i++) {
    const PetscReal h = (ctx->xmax-ctx->xmin)/Mx,xi = ctx->xmin+h/2+i*h;
    const PetscInt  N = 200;
    /* Integrate over cell i using trapezoid rule with N points. */
    for (k=0; k<dof; k++) u[i*dof+k] = 0;
    for (j=0; j<N+1; j++) {
      PetscScalar xj = xi+h*(j-N/2)/(PetscReal)N;
      ierr = (*ctx->physics.sample)(ctx->physics.user,ctx->initial,ctx->bctype,ctx->xmin,ctx->xmax,time,xj,uj);CHKERRQ(ierr);
      for (k=0; k<dof; k++) u[i*dof+k] += ((j==0 || j==N) ? 0.5 : 1.0)*uj[k]/N;
    }
  }
  ierr = DMDAVecRestoreArray(da,U,&u);CHKERRQ(ierr);
  ierr = PetscFree(uj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SolutionStatsView(DM da,Vec X,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscReal         xmin,xmax;
  PetscScalar       sum,tvsum,tvgsum;
  const PetscScalar *x;
  PetscInt          imin,imax,Mx,i,j,xs,xm,dof;
  Vec               Xloc;
  PetscBool         iascii;

  PetscFunctionBeginUser;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    /* PETSc lacks a function to compute total variation norm (difficult in multiple dimensions), we do it here */
    ierr  = DMGetLocalVector(da,&Xloc);CHKERRQ(ierr);
    ierr  = DMGlobalToLocalBegin(da,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
    ierr  = DMGlobalToLocalEnd  (da,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
    ierr  = DMDAVecGetArrayRead(da,Xloc,(void*)&x);CHKERRQ(ierr);
    ierr  = DMDAGetCorners(da,&xs,0,0,&xm,0,0);CHKERRQ(ierr);
    ierr  = DMDAGetInfo(da,0, &Mx,0,0, 0,0,0, &dof,0,0,0,0,0);CHKERRQ(ierr);
    tvsum = 0;
    for (i=xs; i<xs+xm; i++) {
      for (j=0; j<dof; j++) tvsum += PetscAbsScalar(x[i*dof+j] - x[(i-1)*dof+j]);
    }
    ierr = MPI_Allreduce(&tvsum,&tvgsum,1,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)da));CHKERRQ(ierr);
    ierr = DMDAVecRestoreArrayRead(da,Xloc,(void*)&x);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(da,&Xloc);CHKERRQ(ierr);
    ierr = VecMin(X,&imin,&xmin);CHKERRQ(ierr);
    ierr = VecMax(X,&imax,&xmax);CHKERRQ(ierr);
    ierr = VecSum(X,&sum);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Solution range [%8.5f,%8.5f] with extrema at %D and %D, mean %8.5f, ||x||_TV %8.5f\n",(double)xmin,(double)xmax,imin,imax,(double)(sum/Mx),(double)(tvgsum/Mx));CHKERRQ(ierr);
  } else SETERRQ(PETSC_COMM_SELF,1,"Viewer type not supported");
  PetscFunctionReturn(0);
}

int main(int argc,char *argv[])
{
  char              lname[256] = "mc",physname[256] = "advect",final_fname[256] = "solution.m";
  PetscFunctionList limiters   = 0,physics = 0;
  MPI_Comm          comm;
  TS                ts;
  DM                da;
  Vec               X,X0,R;
  FVCtx             ctx;
  IS                iss;
  IS                isf;
  PetscInt          i,dof,xs,xm,Mx,draw = 0,counts=0,*index_slow,*index_fast;
  PetscBool         view_final = PETSC_FALSE;
  PetscReal         ptime;
  PetscErrorCode    ierr;

  ierr = PetscInitialize(&argc,&argv,0,help);CHKERRQ(ierr);
  comm = PETSC_COMM_WORLD;
  ierr = PetscMemzero(&ctx,sizeof(ctx));CHKERRQ(ierr);

  /* Register limiters to be available on the command line */
  ierr = PetscFunctionListAdd(&limiters,"upwind"              ,Limit_Upwind);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&limiters,"lax-wendroff"        ,Limit_LaxWendroff);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&limiters,"beam-warming"        ,Limit_BeamWarming);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&limiters,"fromm"               ,Limit_Fromm);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&limiters,"minmod"              ,Limit_Minmod);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&limiters,"superbee"            ,Limit_Superbee);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&limiters,"mc"                  ,Limit_MC);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&limiters,"vanleer"             ,Limit_VanLeer);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&limiters,"vanalbada"           ,Limit_VanAlbada);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&limiters,"vanalbadatvd"        ,Limit_VanAlbadaTVD);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&limiters,"koren"               ,Limit_Koren);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&limiters,"korensym"            ,Limit_KorenSym);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&limiters,"koren3"              ,Limit_Koren3);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&limiters,"cada-torrilhon2"     ,Limit_CadaTorrilhon2);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&limiters,"cada-torrilhon3-r0p1",Limit_CadaTorrilhon3R0p1);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&limiters,"cada-torrilhon3-r1"  ,Limit_CadaTorrilhon3R1);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&limiters,"cada-torrilhon3-r10" ,Limit_CadaTorrilhon3R10);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&limiters,"cada-torrilhon3-r100",Limit_CadaTorrilhon3R100);CHKERRQ(ierr);

  /* Register physical models to be available on the command line */
  ierr = PetscFunctionListAdd(&physics,"advect"          ,PhysicsCreate_Advect);CHKERRQ(ierr);

  ctx.comm = comm;
  ctx.cfl  = 0.9; ctx.bctype = FVBC_PERIODIC;
  ctx.xmin = -1; ctx.xmax = 1;
  ierr = PetscOptionsBegin(comm,NULL,"Finite Volume solver options","");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-xmin","X min","",ctx.xmin,&ctx.xmin,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-xmax","X max","",ctx.xmax,&ctx.xmax,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsFList("-limit","Name of flux limiter to use","",limiters,lname,lname,sizeof(lname),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-draw","Draw solution vector, bitwise OR of (1=initial,2=final,4=final error)","",draw,&draw,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-view_final","Write final solution in ASCII MATLAB format to given file name","",final_fname,final_fname,sizeof(final_fname),&view_final);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-initial","Initial condition (depends on the physics)","",ctx.initial,&ctx.initial,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simulation","Compare errors with reference solution","",ctx.simulation,&ctx.simulation,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-cfl","CFL number to time step at","",ctx.cfl,&ctx.cfl,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-bc_type","Boundary condition","",FVBCTypes,(PetscEnum)ctx.bctype,(PetscEnum*)&ctx.bctype,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* Choose the limiter from the list of registered limiters */
  ierr = PetscFunctionListFind(limiters,lname,&ctx.limit);CHKERRQ(ierr);
  if (!ctx.limit) SETERRQ1(PETSC_COMM_SELF,1,"Limiter '%s' not found",lname);

  /* Choose the physics from the list of registered models */
  {
    PetscErrorCode (*r)(FVCtx*);
    ierr = PetscFunctionListFind(physics,physname,&r);CHKERRQ(ierr);
    if (!r) SETERRQ1(PETSC_COMM_SELF,1,"Physics '%s' not found",physname);
    /* Create the physics, will set the number of fields and their names */
    ierr = (*r)(&ctx);CHKERRQ(ierr);
  }

  /* Create a DMDA to manage the parallel grid */
  ierr = DMDACreate1d(comm,DM_BOUNDARY_PERIODIC,50,ctx.physics.dof,2,NULL,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  /* Inform the DMDA of the field names provided by the physics. */
  /* The names will be shown in the title bars when run with -ts_monitor_draw_solution */
  for (i=0; i<ctx.physics.dof; i++) {
    ierr = DMDASetFieldName(da,i,ctx.physics.fieldname[i]);CHKERRQ(ierr);
  }
  ierr = DMDAGetInfo(da,0, &Mx,0,0, 0,0,0, &dof,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&xs,0,0,&xm,0,0);CHKERRQ(ierr);

  /* Set coordinates of cell centers */
  ierr = DMDASetUniformCoordinates(da,ctx.xmin+0.5*(ctx.xmax-ctx.xmin)/Mx,ctx.xmax+0.5*(ctx.xmax-ctx.xmin)/Mx,0,0,0,0);CHKERRQ(ierr);

  /* Allocate work space for the Finite Volume solver (so it doesn't have to be reallocated on each function evaluation) */
  ierr = PetscMalloc4(dof*dof,&ctx.R,dof*dof,&ctx.Rinv,2*dof,&ctx.cjmpLR,1*dof,&ctx.cslope);CHKERRQ(ierr);
  ierr = PetscMalloc3(2*dof,&ctx.uLR,dof,&ctx.flux,dof,&ctx.speeds);CHKERRQ(ierr);

  /* Create a vector to store the solution and to save the initial state */
  ierr = DMCreateGlobalVector(da,&X);CHKERRQ(ierr);
  ierr = VecDuplicate(X,&X0);CHKERRQ(ierr);
  ierr = VecDuplicate(X,&R);CHKERRQ(ierr);

  /*-------------------------------- create index for slow parts and fast parts ----------------------------------------*/
  for (i=0; i<Mx; i++){if(ctx.xmin+i*(ctx.xmax-ctx.xmin)/(PetscReal)Mx+0.5*(ctx.xmax-ctx.xmin)/(PetscReal)Mx < 0) counts = counts+1;}  /* this step need to be changed based on discontinuous point of a */
  ierr = PetscMalloc1(counts,&index_slow);CHKERRQ(ierr);
  for (i=0; i<counts; i++) index_slow[i]=i;
  ierr = PetscMalloc1(Mx-counts,&index_fast);CHKERRQ(ierr);
  for (i=0; i<Mx-counts; i++) index_fast[i]=counts+i;
  ierr = ISCreateGeneral(PETSC_COMM_SELF,counts,index_slow,PETSC_COPY_VALUES,&iss);CHKERRQ(ierr);
  counts = Mx-counts;
  ierr = ISCreateGeneral(PETSC_COMM_SELF,counts,index_fast,PETSC_COPY_VALUES,&isf);CHKERRQ(ierr);
  /*--------------------------------------------------------------------------------------------------------------------*/

  /* Create a time-stepping object */
  ierr = TSCreate(comm,&ts);CHKERRQ(ierr);
  ierr = TSSetIS(ts,iss,isf);CHKERRQ(ierr);
  ierr = TSSetDM(ts,da);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,R,FVRHSFunction,&ctx);CHKERRQ(ierr);
  ierr = TSSetRHSFunctionslow(ts,NULL,FVRHSFunctionslow,&ctx);CHKERRQ(ierr);
  ierr = TSSetRHSFunctionfast(ts,NULL,FVRHSFunctionfast,&ctx);CHKERRQ(ierr);
  /*ierr = TSSetType(ts,TSSSP);CHKERRQ(ierr);*/
  ierr = TSSetType(ts,TSPRK);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,10);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);

  /* Compute initial conditions and starting time step */
  ierr = FVSample(&ctx,da,0,X0);CHKERRQ(ierr);
  ierr = FVRHSFunction(ts,0,X0,X,(void*)&ctx);CHKERRQ(ierr); /* Initial function evaluation, only used to determine max speed */
  ierr = VecCopy(X0,X);CHKERRQ(ierr);                        /* The function value was not used so we set X=X0 again */
  ierr = TSSetTimeStep(ts,ctx.cfl/ctx.cfl_idt);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr); /* Take runtime options */
  ierr = SolutionStatsView(da,X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  {
    PetscInt    steps;
    PetscScalar mass_initial,mass_final,mass_difference;

    ierr = TSSolve(ts,X);CHKERRQ(ierr);
    ierr = TSGetSolveTime(ts,&ptime);CHKERRQ(ierr);
    ierr = TSGetStepNumber(ts,&steps);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"Final time %8.5f, steps %D\n",(double)ptime,steps);CHKERRQ(ierr);
    /* calculate the total mass at initial time and final time */
    mass_initial = 0.0;
    mass_final   = 0.0;
    ierr = VecSum(X0,&mass_initial);CHKERRQ(ierr);
    ierr = VecSum(X,&mass_final);CHKERRQ(ierr);
    mass_difference = (ctx.xmax-ctx.xmin)/(PetscScalar)Mx*(mass_final - mass_initial);
    ierr = PetscPrintf(comm,"mass difference %.6g\n",mass_difference);CHKERRQ(ierr);
    if (ctx.simulation) {
      PetscViewer  fd;
      char         filename[PETSC_MAX_PATH_LEN] = "binaryoutput";
      Vec          XR;
      PetscReal    nrm1,nrmsup;
      PetscBool    flg;
      ierr = PetscOptionsGetString(NULL,NULL,"-f",filename,sizeof(filename),&flg);CHKERRQ(ierr);
      if (!flg) SETERRQ(PETSC_COMM_WORLD,1,"Must indicate binary file with the -f option");
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&fd);CHKERRQ(ierr);
      ierr = VecDuplicate(X0,&XR);CHKERRQ(ierr);
      ierr = VecLoad(XR,fd);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);
      ierr = VecAYPX(XR,-1,X);CHKERRQ(ierr);
      ierr = VecNorm(XR,NORM_1,&nrm1);CHKERRQ(ierr);
      ierr = VecNorm(XR,NORM_INFINITY,&nrmsup);CHKERRQ(ierr);
      nrm1 /= Mx;
      ierr = PetscPrintf(comm,"Error ||x-x_e||_1 %8.4e  ||x-x_e||_sup %8.4e\n",(double)nrm1,(double)nrmsup);CHKERRQ(ierr);
      ierr   = VecDestroy(&XR);CHKERRQ(ierr);
    }
  }

  ierr = SolutionStatsView(da,X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  if (draw & 0x1) {ierr = VecView(X0,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}
  if (draw & 0x2) {ierr = VecView(X,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}
  if (draw & 0x4) {
    Vec Y;
    ierr = VecDuplicate(X,&Y);CHKERRQ(ierr);
    ierr = FVSample(&ctx,da,ptime,Y);CHKERRQ(ierr);
    ierr = VecAYPX(Y,-1,X);CHKERRQ(ierr);
    ierr = VecView(Y,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
    ierr = VecDestroy(&Y);CHKERRQ(ierr);
  }

  if (view_final) {
    PetscViewer viewer;
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,final_fname,&viewer);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
    ierr = VecView(X,viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }

  /* Clean up */
  ierr = (*ctx.physics.destroy)(ctx.physics.user);CHKERRQ(ierr);
  for (i=0; i<ctx.physics.dof; i++) {ierr = PetscFree(ctx.physics.fieldname[i]);CHKERRQ(ierr);}
  ierr = PetscFree4(ctx.R,ctx.Rinv,ctx.cjmpLR,ctx.cslope);CHKERRQ(ierr);
  ierr = PetscFree3(ctx.uLR,ctx.flux,ctx.speeds);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = VecDestroy(&X0);CHKERRQ(ierr);
  ierr = VecDestroy(&R);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = ISDestroy(&iss);CHKERRQ(ierr);
  ierr = ISDestroy(&isf);CHKERRQ(ierr);
  ierr = PetscFree(index_slow);CHKERRQ(ierr);
  ierr = PetscFree(index_fast);CHKERRQ(ierr);
  ierr = PetscFunctionListDestroy(&limiters);CHKERRQ(ierr);
  ierr = PetscFunctionListDestroy(&physics);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

    build:
      requires: !complex c99

    test:
      args: -da_grid_x 50 -initial 1 -xmin -1 -xmax 1 -limit mc -physics_advect_a1 1 -physics_advect_a2 2 -ts_dt 0.01 -ts_max_steps 200 -ts_type rk -ts_rk_type 2a -ts_rk_multirate_type partitioned -ts_rk_dtratio 2 -simulation -f "yourfilename.bin"
      requires: !complex !single
      note: yourfilename.bin need to be generated by userselves and this is only for the convergent study
            -ts_rk_dtratio is the ratio between larger time step size and small time step size
            -ts_rk_multirate_type has three choices: none (for single step RK)
                                                     combined (for multirate method and user just need to provide the same RHS with the single step RK)
                                                     partitioned (for multiraet method and user need to provide two RHS, one is for fast components and the orther is for slow components
      test:
      args: -da_grid_x 50 -initial 1 -xmin -1 -xmax 1 -simulation -f "yourfilename.bin" -limit mc -physics_advect_a1 1 -physics_advect_a2 2 -ts_dt 0.01 -ts_max_steps 200 -ts_type prk -ts_prk_type pm2 -ts_prk_multirate_type combined
      note: yourfilename.bin need to be generated by userselves and this is only for the convergent study
            -ts_rk_multirate_type has teo choices: combined (for multirate method and user just need to provide the same RHS with the single step RK)
                                                   partitioned (for multiraet method and user need to provide two RHS, one is for fast components and the orther is for slow components)
TEST*/
