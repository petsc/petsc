static const char help[] = "1D periodic Finite Volume solver in slope-limiter form with semidiscrete time stepping.\n"
  "Solves scalar and vector problems, choose the physical model with -physics\n"
  "  advection   - Constant coefficient scalar advection\n"
  "                u_t       + (a*u)_x               = 0\n"
  "  burgers     - Burgers equation\n"
  "                u_t       + (u^2/2)_x             = 0\n"
  "  traffic     - Traffic equation\n"
  "                u_t       + (u*(1-u))_x           = 0\n"
  "  isogas      - Isothermal gas dynamics\n"
  "                rho_t     + (rho*u)_x             = 0\n"
  "                (rho*u)_t + (rho*u^2 + c^2*rho)_x = 0\n"
  "  shallow     - Shallow water equations\n"
  "                h_t       + (h*u)_x               = 0\n"
  "                (h*u)_t   + (h*u^2 + g*h^2/2)_x   = 0\n"
  "Some of these physical models have multiple Riemann solvers, select these with -physics_xxx_riemann\n"
  "  exact       - Exact Riemann solver which usually needs to perform a Newton iteration to connect\n"
  "                the states across shocks and rarefactions\n"
  "  roe         - Linearized scheme, usually with an entropy fix inside sonic rarefactions\n"
  "The systems provide a choice of reconstructions with -physics_xxx_reconstruct\n"
  "  characteristic - Limit the characteristic variables, this is usually preferred (default)\n"
  "  conservative   - Limit the conservative variables directly, can cause undesired interaction of waves\n\n"
  "A variety of limiters for high-resolution TVD limiters are available with -limit\n"
  "  upwind,minmod,superbee,mc,vanleer,vanalbada,koren,cada-torillhon (last two are nominally third order)\n"
  "  and non-TVD schemes lax-wendroff,beam-warming,fromm\n\n"
  "To preserve the TVD property, one should time step with a strong stability preserving method.\n"
  "The optimal high order explicit Runge-Kutta methods in TSSSP are recommended for non-stiff problems.\n\n"
  "Several initial conditions can be chosen with -initial N\n\n"
  "The problem size should be set with -da_grid_x M\n\n";

/* To get isfinite in math.h */
#define _XOPEN_SOURCE 600

#include <unistd.h>             /* usleep */
#include "petscts.h"
#include "petscda.h"

#include "../src/mat/blockinvert.h" /* For the Kernel_*_gets_* stuff for BAIJ */

static inline PetscReal Sgn(PetscReal a) { return (a<0) ? -1 : 1; }
static inline PetscReal Abs(PetscReal a) { return (a<0) ? 0 : a; }
static inline PetscReal Sqr(PetscReal a) { return a*a; }
static inline PetscReal MaxAbs(PetscReal a,PetscReal b) { return (PetscAbs(a) > PetscAbs(b)) ? a : b; }
static inline PetscReal MinAbs(PetscReal a,PetscReal b) { return (PetscAbs(a) < PetscAbs(b)) ? a : b; }
static inline PetscReal MinMod2(PetscReal a,PetscReal b)
{ return (a*b<0) ? 0 : Sgn(a)*PetscMin(PetscAbs(a),PetscAbs(b)); }
static inline PetscReal MaxMod2(PetscReal a,PetscReal b)
{ return (a*b<0) ? 0 : Sgn(a)*PetscMax(PetscAbs(a),PetscAbs(b)); }
static inline PetscReal MinMod3(PetscReal a,PetscReal b,PetscReal c)
{return (a*b<0 || a*c<0) ? 0 : Sgn(a)*PetscMin(PetscAbs(a),PetscMin(PetscAbs(b),PetscAbs(c))); }

static inline PetscReal RangeMod(PetscReal a,PetscReal xmin,PetscReal xmax)
{ PetscReal range = xmax-xmin; return xmin + fmod(range+fmod(a,range),range); }


/* ----------------------- Lots of limiters, these could go in a separate library ------------------------- */
typedef struct _LimitInfo {
  PetscReal hx;
  PetscInt m;
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
  for (i=0; i<info->m; i++) lmt[i] = (jL[i]*jR[i]<0) ? 0
                        : (jL[i]*Sqr(jR[i]) + Sqr(jL[i])*jR[i]) / (Sqr(jL[i]) + Sqr(jR[i]) + 1e-15);
}
static void Limit_Koren(LimitInfo info,const PetscScalar *jL,const PetscScalar *jR,PetscScalar *lmt) /* differentiable */
{ /* phi = (t + 2*t^2) / (2 - t + 2*t^2) */
  PetscInt i;
  for (i=0; i<info->m; i++) lmt[i] = ((jL[i]*Sqr(jR[i]) + 2*Sqr(jL[i])*jR[i])
                                / (2*Sqr(jL[i]) - jL[i]*jR[i] + 2*Sqr(jR[i]) + 1e-15));
}
static void Limit_KorenSym(LimitInfo info,const PetscScalar *jL,const PetscScalar *jR,PetscScalar *lmt) /* differentiable */
{ /* Symmetric version of above */
  PetscInt i;
  for (i=0; i<info->m; i++) lmt[i] = (1.5*(jL[i]*Sqr(jR[i]) + Sqr(jL[i])*jR[i])
                                / (2*Sqr(jL[i]) - jL[i]*jR[i] + 2*Sqr(jR[i]) + 1e-15));
}
static void Limit_Koren3(LimitInfo info,const PetscScalar *jL,const PetscScalar *jR,PetscScalar *lmt)
{ /* Eq 11 of Cada-Torrilhon 2009 */
  PetscInt i;
  for (i=0; i<info->m; i++) lmt[i] = MinMod3(2*jL[i],(jL[i]+2*jR[i])/3,2*jR[i]);
}

static PetscReal CadaTorrilhonPhiHatR_Eq13(PetscReal L,PetscReal R)
{ return PetscMax(0,PetscMin((L+2*R)/3,
                              PetscMax(-0.5*L,
                                       PetscMin(2*L,
                                                PetscMin((L+2*R)/3,1.6*R)))));
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
  PetscInt i;
  for (i=0; i<info->m; i++) {
    const PetscReal eta = (Sqr(jL[i]) + Sqr(jR[i])) / Sqr(r*hx);
    lmt[i] = ((eta < 1-eps)
              ? (jL[i] + 2*jR[i]) / 3
              : ((eta > 1+eps)
                 ? CadaTorrilhonPhiHatR_Eq13(jL[i],jR[i])
                 : 0.5*((1-(eta-1)/eps)*(jL[i]+2*jR[i])/3
                        + (1+(eta+1)/eps)*CadaTorrilhonPhiHatR_Eq13(jL[i],jR[i]))));
  }
}
static void Limit_CadaTorrilhon3R0p1(LimitInfo info,const PetscScalar *jL,const PetscScalar *jR,PetscScalar *lmt)
{ Limit_CadaTorrilhon3R(0.1,info,jL,jR,lmt); }
static void Limit_CadaTorrilhon3R1(LimitInfo info,const PetscScalar *jL,const PetscScalar *jR,PetscScalar *lmt)
{ Limit_CadaTorrilhon3R(1,info,jL,jR,lmt); }
static void Limit_CadaTorrilhon3R10(LimitInfo info,const PetscScalar *jL,const PetscScalar *jR,PetscScalar *lmt)
{ Limit_CadaTorrilhon3R(10,info,jL,jR,lmt); }
static void Limit_CadaTorrilhon3R100(LimitInfo info,const PetscScalar *jL,const PetscScalar *jR,PetscScalar *lmt)
{ Limit_CadaTorrilhon3R(100,info,jL,jR,lmt); }


/* --------------------------------- Finite Volume data structures ----------------------------------- */

typedef enum {FVBC_PERIODIC, FVBC_OUTFLOW} FVBCType;
static const char *FVBCTypes[] = {"PERIODIC","OUTFLOW","FVBCType","FVBC_",0};
typedef PetscErrorCode (*RiemannFunction)(void*,PetscInt,const PetscScalar*,const PetscScalar*,PetscScalar*,PetscReal*);
typedef PetscErrorCode (*ReconstructFunction)(void*,PetscInt,const PetscScalar*,PetscScalar*,PetscScalar*);

typedef struct {
  PetscErrorCode (*sample)(void*,PetscInt,FVBCType,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal*);
  RiemannFunction riemann;
  ReconstructFunction characteristic;
  PetscErrorCode (*destroy)(void*);
  void *user;
  PetscInt dof;
  char *fieldname[16];
} PhysicsCtx;

typedef struct {
  void (*limit)(LimitInfo,const PetscScalar*,const PetscScalar*,PetscScalar*);
  PhysicsCtx physics;

  MPI_Comm comm;
  char prefix[256];
  DA da;
  /* Local work arrays */
  PetscScalar *R,*Rinv;         /* Characteristic basis, and it's inverse.  COLUMN-MAJOR */
  PetscScalar *cjmpLR;          /* Jumps at left and right edge of cell, in characteristic basis, len=2*dof */
  PetscScalar *cslope;          /* Limited slope, written in characteristic basis */
  PetscScalar *uLR;             /* Solution at left and right of interface, conservative variables, len=2*dof */
  PetscScalar *flux;            /* Flux across interface */

  PetscReal cfl_idt;            /* Max allowable value of 1/Delta t */
  PetscReal cfl;
  PetscReal xmin,xmax;
  PetscInt initial;
  PetscTruth exact;
  FVBCType bctype;
} FVCtx;


/* Utility */

#undef __FUNCT__  
#define __FUNCT__ "RiemannListAdd"
PetscErrorCode RiemannListAdd(PetscFList *flist,const char *name,RiemannFunction rsolve)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFListAdd(flist,name,"",(void(*)(void))rsolve);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "RiemannListFind"
PetscErrorCode RiemannListFind(PetscFList flist,const char *name,RiemannFunction *rsolve)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFListFind(flist,PETSC_COMM_WORLD,name,(void(**)(void))rsolve);CHKERRQ(ierr);
  if (!*rsolve) SETERRQ1(1,"Riemann solver \"%s\" could not be found",name);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ReconstructListAdd"
PetscErrorCode ReconstructListAdd(PetscFList *flist,const char *name,ReconstructFunction r)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFListAdd(flist,name,"",(void(*)(void))r);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "ReconstructListFind"
PetscErrorCode ReconstructListFind(PetscFList flist,const char *name,ReconstructFunction *r)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFListFind(flist,PETSC_COMM_WORLD,name,(void(**)(void))r);CHKERRQ(ierr);
  if (!*r) SETERRQ1(1,"Reconstruction \"%s\" could not be found",name);
  PetscFunctionReturn(0);
}


/* --------------------------------- Physics ----------------------------------- */
/**
* Each physical model consists of Riemann solver and a function to determine the basis to use for reconstruction.  These
* are set with the PhysicsCreate_XXX function which allocates private storage and sets these methods as well as the
* number of fields and their names, and a function to deallocate private storage.
**/

/* First a few functions useful to several different physics */
#undef __FUNCT__
#define __FUNCT__ "PhysicsCharacteristic_Conservative"  
static PetscErrorCode PhysicsCharacteristic_Conservative(void *vctx,PetscInt m,const PetscScalar *u,PetscScalar *X,PetscScalar *Xi)
{
  PetscInt i,j;

  PetscFunctionBegin;
  for (i=0; i<m; i++) {
    for (j=0; j<m; j++) {
      Xi[i*m+j] = X[i*m+j] = (PetscScalar)(i==j);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PhysicsDestroy_SimpleFree"  
static PetscErrorCode PhysicsDestroy_SimpleFree(void *vctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(vctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



/* --------------------------------- Advection ----------------------------------- */

typedef struct {
  PetscReal a;                  /* advective velocity */
} AdvectCtx;

#undef __FUNCT__  
#define __FUNCT__ "PhysicsRiemann_Advect"
static PetscErrorCode PhysicsRiemann_Advect(void *vctx,PetscInt m,const PetscScalar *uL,const PetscScalar *uR,PetscScalar *flux,PetscReal *maxspeed)
{
  AdvectCtx *ctx = (AdvectCtx*)vctx;
  PetscReal speed;

  PetscFunctionBegin;
  speed = ctx->a;
  flux[0] = PetscMax(0,speed)*uL[0] + PetscMin(0,speed)*uR[0];
  *maxspeed = speed;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PhysicsSample_Advect"
static PetscErrorCode PhysicsSample_Advect(void *vctx,PetscInt initial,FVBCType bctype,PetscReal xmin,PetscReal xmax,PetscReal t,PetscReal x,PetscReal *u)
{
  AdvectCtx *ctx = (AdvectCtx*)vctx;
  PetscReal a = ctx->a,x0;

  PetscFunctionBegin;
  switch (bctype) {
    case FVBC_OUTFLOW: x0 = x-a*t; break;
    case FVBC_PERIODIC: x0 = RangeMod(x-a*t,xmin,xmax); break;
    default: SETERRQ(1,"unknown BCType");
  }
  switch (initial) {
    case 0: u[0] = (x0 < 0) ? 1 : -1; break;
    case 1: u[0] = (x0 < 0) ? -1 : 1; break;
    case 2: u[0] = (0 < x0 && x0 < 1) ? 1 : 0; break;
    case 3: u[0] = sin(2*M_PI*x0); break;
    case 4: u[0] = PetscAbs(x0); break;
    default: SETERRQ(1,"unknown initial condition");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PhysicsCreate_Advect"
static PetscErrorCode PhysicsCreate_Advect(FVCtx *ctx)
{
  PetscErrorCode ierr;
  AdvectCtx *user;

  PetscFunctionBegin;
  ierr = PetscNew(*user,&user);CHKERRQ(ierr);
  ctx->physics.sample         = PhysicsSample_Advect;
  ctx->physics.riemann        = PhysicsRiemann_Advect;
  ctx->physics.characteristic = PhysicsCharacteristic_Conservative;
  ctx->physics.destroy        = PhysicsDestroy_SimpleFree;
  ctx->physics.user           = user;
  ctx->physics.dof            = 1;
  ierr = PetscStrallocpy("u",&ctx->physics.fieldname[0]);CHKERRQ(ierr);
  user->a = 1;
  ierr = PetscOptionsBegin(ctx->comm,ctx->prefix,"Options for advection","");CHKERRQ(ierr);
  {
    ierr = PetscOptionsReal("-physics_advect_a","Speed","",user->a,&user->a,PETSC_NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



/* --------------------------------- Burgers ----------------------------------- */

typedef struct {
  PetscReal lxf_speed;
} BurgersCtx;

#undef __FUNCT__  
#define __FUNCT__ "PhysicsSample_Burgers"
static PetscErrorCode PhysicsSample_Burgers(void *vctx,PetscInt initial,FVBCType bctype,PetscReal xmin,PetscReal xmax,PetscReal t,PetscReal x,PetscReal *u)
{

  PetscFunctionBegin;
  if (bctype == FVBC_PERIODIC && t > 0) SETERRQ(1,"Exact solution not implemented for periodic");
  switch (initial) {
    case 0: u[0] = (x < 0) ? 1 : -1; break;
    case 1:
      if       (x < -t) u[0] = -1;
      else if  (x < t)  u[0] = x/t;
      else              u[0] = 1;
      break;
    case 2:
      if      (x < 0)       u[0] = 0;
      else if (x <= t)      u[0] = x/t;
      else if (x < 1+0.5*t) u[0] = 1;
      else                  u[0] = 0;
      break;
    case 3:
      if       (x < 0.2*t) u[0] = 0.2;
      else if  (x < t) u[0] = x/t;
      else             u[0] = 1;
      break;
    case 4:
      if (t > 0) SETERRQ(1,"Only initial condition available");
      u[0] = 0.7 + 0.3*sin(2*M_PI*((x-xmin)/(xmax-xmin)));
      break;
    default: SETERRQ(1,"unknown initial condition");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PhysicsRiemann_Burgers_Exact"
static PetscErrorCode PhysicsRiemann_Burgers_Exact(void *vctx,PetscInt m,const PetscScalar *uL,const PetscScalar *uR,PetscScalar *flux,PetscReal *maxspeed)
{

  PetscFunctionBegin;
  if (uL[0] < uR[0]) {          /* rarefaction */
    flux[0] = (uL[0]*uR[0] < 0)
      ? 0                       /* sonic rarefaction */
      : 0.5*PetscMin(PetscSqr(uL[0]),PetscSqr(uR[0]));
  } else {                      /* shock */
    flux[0] = 0.5*PetscMax(PetscSqr(uL[0]),PetscSqr(uR[0]));
  }
  *maxspeed = (PetscAbs(uL[0]) > PetscAbs(uR[0])) ? uL[0] : uR[0];
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PhysicsRiemann_Burgers_Roe"
static PetscErrorCode PhysicsRiemann_Burgers_Roe(void *vctx,PetscInt m,const PetscScalar *uL,const PetscScalar *uR,PetscScalar *flux,PetscReal *maxspeed)
{
  PetscReal speed;

  PetscFunctionBegin;
  speed = 0.5*(uL[0] + uR[0]);
  flux[0] = 0.25*(PetscSqr(uL[0]) + PetscSqr(uR[0])) - 0.5*PetscAbs(speed)*(uR[0]-uL[0]);
  if (uL[0] <= 0 && 0 <= uR[0]) flux[0] = 0; /* Entropy fix for sonic rarefaction */
  *maxspeed = speed;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PhysicsRiemann_Burgers_LxF"
static PetscErrorCode PhysicsRiemann_Burgers_LxF(void *vctx,PetscInt m,const PetscScalar *uL,const PetscScalar *uR,PetscScalar *flux,PetscReal *maxspeed)
{
  PetscReal c;
  PetscScalar fL,fR;

  PetscFunctionBegin;
  c = ((BurgersCtx*)vctx)->lxf_speed;
  fL = 0.5*PetscSqr(uL[0]);
  fR = 0.5*PetscSqr(uR[0]);
  flux[0] = 0.5*(fL + fR) - 0.5*c*(uR[0] - uL[0]);
  *maxspeed = c;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PhysicsRiemann_Burgers_Rusanov"
static PetscErrorCode PhysicsRiemann_Burgers_Rusanov(void *vctx,PetscInt m,const PetscScalar *uL,const PetscScalar *uR,PetscScalar *flux,PetscReal *maxspeed)
{
  PetscReal c;
  PetscScalar fL,fR;

  PetscFunctionBegin;
  c = PetscMax(PetscAbs(uL[0]),PetscAbs(uR[0]));
  fL = 0.5*PetscSqr(uL[0]);
  fR = 0.5*PetscSqr(uR[0]);
  flux[0] = 0.5*(fL + fR) - 0.5*c*(uR[0] - uL[0]);
  *maxspeed = c;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PhysicsCreate_Burgers"
static PetscErrorCode PhysicsCreate_Burgers(FVCtx *ctx)
{
  BurgersCtx *user;
  PetscErrorCode ierr;
  RiemannFunction r;
  PetscFList rlist = 0;
  char rname[256] = "exact";

  PetscFunctionBegin;
  ierr = PetscNew(*user,&user);CHKERRQ(ierr);
  ctx->physics.sample         = PhysicsSample_Burgers;
  ctx->physics.characteristic = PhysicsCharacteristic_Conservative;
  ctx->physics.destroy        = PhysicsDestroy_SimpleFree;
  ctx->physics.user           = user;
  ctx->physics.dof            = 1;
  ierr = PetscStrallocpy("u",&ctx->physics.fieldname[0]);CHKERRQ(ierr);
  ierr = RiemannListAdd(&rlist,"exact",  PhysicsRiemann_Burgers_Exact);CHKERRQ(ierr);
  ierr = RiemannListAdd(&rlist,"roe",    PhysicsRiemann_Burgers_Roe);CHKERRQ(ierr);
  ierr = RiemannListAdd(&rlist,"lxf",    PhysicsRiemann_Burgers_LxF);CHKERRQ(ierr);
  ierr = RiemannListAdd(&rlist,"rusanov",PhysicsRiemann_Burgers_Rusanov);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(ctx->comm,ctx->prefix,"Options for advection","");CHKERRQ(ierr);
  {
    ierr = PetscOptionsList("-physics_burgers_riemann","Riemann solver","",rlist,rname,rname,sizeof rname,PETSC_NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  ierr = RiemannListFind(rlist,rname,&r);CHKERRQ(ierr);
  ierr = PetscFListDestroy(&rlist);CHKERRQ(ierr);
  ctx->physics.riemann = r;

  /* *
  * Hack to deal with LxF in semi-discrete form
  * max speed is 1 for the basic initial conditions (where |u| <= 1)
  * */
  if (r == PhysicsRiemann_Burgers_LxF) user->lxf_speed = 1;
  PetscFunctionReturn(0);
}



/* --------------------------------- Traffic ----------------------------------- */

typedef struct {
  PetscReal lxf_speed;
  PetscReal a;
} TrafficCtx;

static inline PetscScalar TrafficFlux(PetscScalar a,PetscScalar u) { return a*u*(1-u); }

#undef __FUNCT__  
#define __FUNCT__ "PhysicsSample_Traffic"
static PetscErrorCode PhysicsSample_Traffic(void *vctx,PetscInt initial,FVBCType bctype,PetscReal xmin,PetscReal xmax,PetscReal t,PetscReal x,PetscReal *u)
{
  PetscReal a = ((TrafficCtx*)vctx)->a;

  PetscFunctionBegin;
  if (bctype == FVBC_PERIODIC && t > 0) SETERRQ(1,"Exact solution not implemented for periodic");
  switch (initial) {
    case 0:
      u[0] = (-a*t < x) ? 2 : 0; break;
    case 1:
      if      (x < PetscMin(2*a*t,0.5+a*t)) u[0] = -1;
      else if (x < 1)                       u[0] = 0;
      else                                  u[0] = 1;
      break;
    case 2:
      if (t > 0) SETERRQ(1,"Only initial condition available");
      u[0] = 0.7 + 0.3*sin(2*M_PI*((x-xmin)/(xmax-xmin)));
      break;
    default: SETERRQ(1,"unknown initial condition");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PhysicsRiemann_Traffic_Exact"
static PetscErrorCode PhysicsRiemann_Traffic_Exact(void *vctx,PetscInt m,const PetscScalar *uL,const PetscScalar *uR,PetscScalar *flux,PetscReal *maxspeed)
{
  PetscReal a = ((TrafficCtx*)vctx)->a;

  PetscFunctionBegin;
  if (uL[0] < uR[0]) {
    flux[0] = PetscMin(TrafficFlux(a,uL[0]),
                       TrafficFlux(a,uR[0]));
  } else {
    flux[0] = (uR[0] < 0.5 && 0.5 < uL[0])
      ? TrafficFlux(a,0.5)
      : PetscMax(TrafficFlux(a,uL[0]),
                 TrafficFlux(a,uR[0]));
  }
  *maxspeed = a*MaxAbs(1-2*uL[0],1-2*uR[0]);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PhysicsRiemann_Traffic_Roe"
static PetscErrorCode PhysicsRiemann_Traffic_Roe(void *vctx,PetscInt m,const PetscScalar *uL,const PetscScalar *uR,PetscScalar *flux,PetscReal *maxspeed)
{
  PetscReal a = ((TrafficCtx*)vctx)->a;
  PetscReal speed;

  PetscFunctionBegin;
  speed = a*(1 - (uL[0] + uR[0]));
  flux[0] = 0.5*(TrafficFlux(a,uL[0]) + TrafficFlux(a,uR[0])) - 0.5*PetscAbs(speed)*(uR[0]-uL[0]);
  *maxspeed = speed;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PhysicsRiemann_Traffic_LxF"
static PetscErrorCode PhysicsRiemann_Traffic_LxF(void *vctx,PetscInt m,const PetscScalar *uL,const PetscScalar *uR,PetscScalar *flux,PetscReal *maxspeed)
{
  TrafficCtx *phys = (TrafficCtx*)vctx;
  PetscReal a = phys->a;
  PetscReal speed;

  PetscFunctionBegin;
  speed = a*(1 - (uL[0] + uR[0]));
  flux[0] = 0.5*(TrafficFlux(a,uL[0]) + TrafficFlux(a,uR[0])) - 0.5*phys->lxf_speed*(uR[0]-uL[0]);
  *maxspeed = speed;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PhysicsRiemann_Traffic_Rusanov"
static PetscErrorCode PhysicsRiemann_Traffic_Rusanov(void *vctx,PetscInt m,const PetscScalar *uL,const PetscScalar *uR,PetscScalar *flux,PetscReal *maxspeed)
{
  PetscReal a = ((TrafficCtx*)vctx)->a;
  PetscReal speed;

  PetscFunctionBegin;
  speed = a*PetscMax(PetscAbs(1-2*uL[0]),PetscAbs(1-2*uR[0]));
  flux[0] = 0.5*(TrafficFlux(a,uL[0]) + TrafficFlux(a,uR[0])) - 0.5*speed*(uR[0]-uL[0]);
  *maxspeed = speed;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PhysicsCreate_Traffic"
static PetscErrorCode PhysicsCreate_Traffic(FVCtx *ctx)
{
  PetscErrorCode ierr;
  TrafficCtx *user;
  RiemannFunction r;
  PetscFList rlist = 0;
  char rname[256] = "exact";

  PetscFunctionBegin;
  ierr = PetscNew(*user,&user);CHKERRQ(ierr);
  ctx->physics.sample         = PhysicsSample_Traffic;
  ctx->physics.characteristic = PhysicsCharacteristic_Conservative;
  ctx->physics.destroy        = PhysicsDestroy_SimpleFree;
  ctx->physics.user           = user;
  ctx->physics.dof            = 1;
  ierr = PetscStrallocpy("density",&ctx->physics.fieldname[0]);CHKERRQ(ierr);
  user->a = 0.5;
  ierr = RiemannListAdd(&rlist,"exact",  PhysicsRiemann_Traffic_Exact);CHKERRQ(ierr);
  ierr = RiemannListAdd(&rlist,"roe",    PhysicsRiemann_Traffic_Roe);CHKERRQ(ierr);
  ierr = RiemannListAdd(&rlist,"lxf",    PhysicsRiemann_Traffic_LxF);CHKERRQ(ierr);
  ierr = RiemannListAdd(&rlist,"rusanov",PhysicsRiemann_Traffic_Rusanov);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(ctx->comm,ctx->prefix,"Options for Traffic","");CHKERRQ(ierr);
  {
    ierr = PetscOptionsReal("-physics_traffic_a","Flux = a*u*(1-u)","",user->a,&user->a,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsList("-physics_traffic_riemann","Riemann solver","",rlist,rname,rname,sizeof rname,PETSC_NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = RiemannListFind(rlist,rname,&r);CHKERRQ(ierr);
  ierr = PetscFListDestroy(&rlist);CHKERRQ(ierr);
  ctx->physics.riemann = r;

  /* *
  * Hack to deal with LxF in semi-discrete form
  * max speed is 3*a for the basic initial conditions (-1 <= u <= 2)
  * */
  if (r == PhysicsRiemann_Traffic_LxF) user->lxf_speed = 3*user->a;

  PetscFunctionReturn(0);
}




/* --------------------------------- Isothermal Gas Dynamics ----------------------------------- */

typedef struct {
  PetscReal acoustic_speed;
} IsoGasCtx;

static inline void IsoGasFlux(PetscReal c,const PetscScalar *u,PetscScalar *f)
{
  f[0] = u[1];
  f[1] = PetscSqr(u[1])/u[0] + c*c*u[0];
}

#undef __FUNCT__  
#define __FUNCT__ "PhysicsSample_IsoGas"
static PetscErrorCode PhysicsSample_IsoGas(void *vctx,PetscInt initial,FVBCType bctype,PetscReal xmin,PetscReal xmax,PetscReal t,PetscReal x,PetscReal *u)
{

  PetscFunctionBegin;
  if (t > 0) SETERRQ(1,"Exact solutions not implemented for t > 0");
  switch (initial) {
    case 0:
      u[0] = (x < 0) ? 1 : 0.5;
      u[1] = (x < 0) ? 1 : 0.7;
      break;
    case 1:
      u[0] = 1+0.5*sin(2*M_PI*x);
      u[1] = 1*u[0];
      break;
    default: SETERRQ(1,"unknown initial condition");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PhysicsRiemann_IsoGas_Roe"
static PetscErrorCode PhysicsRiemann_IsoGas_Roe(void *vctx,PetscInt m,const PetscScalar *uL,const PetscScalar *uR,PetscScalar *flux,PetscReal *maxspeed)
{
  IsoGasCtx *phys = (IsoGasCtx*)vctx;
  PetscReal c = phys->acoustic_speed;
  PetscScalar ubar,du[2],a[2],fL[2],fR[2],lam[2],ustar[2],R[2][2];
  PetscInt i;

  PetscFunctionBegin;
  ubar = (uL[1]/PetscSqrtScalar(uL[0]) + uR[1]/PetscSqrtScalar(uR[0])) / (PetscSqrtScalar(uL[0]) + PetscSqrtScalar(uR[0]));
  /* write fluxuations in characteristic basis */
  du[0] = uR[0] - uL[0];
  du[1] = uR[1] - uL[1];
  a[0] = (1/(2*c)) * ((ubar + c)*du[0] - du[1]);
  a[1] = (1/(2*c)) * ((-ubar + c)*du[0] + du[1]);
  /* wave speeds */
  lam[0] = ubar - c;
  lam[1] = ubar + c;
  /* Right eigenvectors */
  R[0][0] = 1; R[0][1] = ubar-c;
  R[1][0] = 1; R[1][1] = ubar+c;
  /* Compute state in star region (between the 1-wave and 2-wave) */
  for (i=0; i<2; i++) ustar[i] = uL[i] + a[0]*R[0][i];
  if (uL[1]/uL[0] < c && c < ustar[1]/ustar[0]) { /* 1-wave is sonic rarefaction */
    PetscScalar ufan[2];
    ufan[0] = uL[0]*PetscExpScalar(uL[1]/(uL[0]*c) - 1);
    ufan[1] = c*ufan[0];
    IsoGasFlux(c,ufan,flux);
  } else if (ustar[1]/ustar[0] < -c && -c < uR[1]/uR[0]) { /* 2-wave is sonic rarefaction */
    PetscScalar ufan[2];
    ufan[0] = uR[0]*PetscExpScalar(-uR[1]/(uR[0]*c) - 1);
    ufan[1] = -c*ufan[0];
    IsoGasFlux(c,ufan,flux);
  } else {                      /* Centered form */
    IsoGasFlux(c,uL,fL);
    IsoGasFlux(c,uR,fR);
    for (i=0; i<2; i++) {
      PetscScalar absdu = PetscAbsScalar(lam[0])*a[0]*R[0][i] + PetscAbsScalar(lam[1])*a[1]*R[1][i];
      flux[i] = 0.5*(fL[i]+fR[i]) - 0.5*absdu;
    }
  }
  *maxspeed = MaxAbs(lam[0],lam[1]);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PhysicsRiemann_IsoGas_Exact"
static PetscErrorCode PhysicsRiemann_IsoGas_Exact(void *vctx,PetscInt m,const PetscScalar *uL,const PetscScalar *uR,PetscScalar *flux,PetscReal *maxspeed)
{
  IsoGasCtx *phys = (IsoGasCtx*)vctx;
  PetscReal c = phys->acoustic_speed;
  PetscScalar ustar[2];
  struct {PetscScalar rho,u;} L = {uL[0],uL[1]/uL[0]},R = {uR[0],uR[1]/uR[0]},star;
  PetscInt i;

  PetscFunctionBegin;
  if (!(L.rho > 0 && R.rho > 0)) SETERRQ(1,"Reconstructed density is negative");
  {
    /* Solve for star state */
    PetscScalar res,tmp,rho = 0.5*(L.rho + R.rho); /* initial guess */
    for (i=0; i<20; i++) {
      PetscScalar fr,fl,dfr,dfl;
      fl = (L.rho < rho)
        ? (rho-L.rho)/PetscSqrtScalar(L.rho*rho)       /* shock */
        : PetscLogScalar(rho) - PetscLogScalar(L.rho); /* rarefaction */
      fr = (R.rho < rho)
        ? (rho-R.rho)/PetscSqrtScalar(R.rho*rho)       /* shock */
        : PetscLogScalar(rho) - PetscLogScalar(R.rho); /* rarefaction */
      res = R.u-L.u + c*(fr+fl);
      if (!isfinite(res)) SETERRQ1(1,"non-finite residual=%g",res);
      if (PetscAbsScalar(res) < 1e-10) {
        star.rho = rho;
        star.u   = L.u - c*fl;
        goto converged;
      }
      dfl = (L.rho < rho)
        ? 1/PetscSqrtScalar(L.rho*rho)*(1 - 0.5*(rho-L.rho)/rho)
        : 1/rho;
      dfr = (R.rho < rho)
        ? 1/PetscSqrtScalar(R.rho*rho)*(1 - 0.5*(rho-R.rho)/rho)
        : 1/rho;
      tmp = rho - res/(c*(dfr+dfl));
      if (tmp <= 0) rho /= 2;   /* Guard against Newton shooting off to a negative density */
      else rho = tmp;
      if (!((rho > 0) && isnormal(rho))) SETERRQ1(1,"non-normal iterate rho=%g",rho);
    }
    SETERRQ1(1,"Newton iteration for star.rho diverged after %d iterations",i);
  }
  converged:
  if (L.u-c < 0 && 0 < star.u-c) { /* 1-wave is sonic rarefaction */
    PetscScalar ufan[2];
    ufan[0] = L.rho*PetscExpScalar(L.u/c - 1);
    ufan[1] = c*ufan[0];
    IsoGasFlux(c,ufan,flux);
  } else if (star.u+c < 0 && 0 < R.u+c) { /* 2-wave is sonic rarefaction */
    PetscScalar ufan[2];
    ufan[0] = R.rho*PetscExpScalar(-R.u/c - 1);
    ufan[1] = -c*ufan[0];
    IsoGasFlux(c,ufan,flux);
  } else if ((L.rho >= star.rho && L.u-c >= 0)
             || (L.rho < star.rho && (star.rho*star.u-L.rho*L.u)/(star.rho-L.rho) > 0)) {
    /* 1-wave is supersonic rarefaction, or supersonic shock */
    IsoGasFlux(c,uL,flux);
  } else if ((star.rho <= R.rho && R.u+c <= 0)
             || (star.rho > R.rho && (R.rho*R.u-star.rho*star.u)/(R.rho-star.rho) < 0)) {
    /* 2-wave is supersonic rarefaction or supersonic shock */
    IsoGasFlux(c,uR,flux);
  } else {
    ustar[0] = star.rho;
    ustar[1] = star.rho*star.u;
    IsoGasFlux(c,ustar,flux);
  }
  *maxspeed = MaxAbs(MaxAbs(star.u-c,star.u+c),MaxAbs(L.u-c,R.u+c));
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PhysicsRiemann_IsoGas_Rusanov"
static PetscErrorCode PhysicsRiemann_IsoGas_Rusanov(void *vctx,PetscInt m,const PetscScalar *uL,const PetscScalar *uR,PetscScalar *flux,PetscReal *maxspeed)
{
  IsoGasCtx *phys = (IsoGasCtx*)vctx;
  PetscScalar c = phys->acoustic_speed,fL[2],fR[2],s;
  struct {PetscScalar rho,u;} L = {uL[0],uL[1]/uL[0]},R = {uR[0],uR[1]/uR[0]};

  PetscFunctionBegin;
  if (!(L.rho > 0 && R.rho > 0)) SETERRQ(1,"Reconstructed density is negative");
  IsoGasFlux(c,uL,fL);
  IsoGasFlux(c,uR,fR);
  s = PetscMax(PetscAbs(L.u),PetscAbs(R.u))+c;
  flux[0] = 0.5*(fL[0] + fR[0]) + 0.5*s*(uL[0] - uR[0]);
  flux[1] = 0.5*(fL[1] + fR[1]) + 0.5*s*(uL[1] - uR[1]);
  *maxspeed = s;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PhysicsCharacteristic_IsoGas"
static PetscErrorCode PhysicsCharacteristic_IsoGas(void *vctx,PetscInt m,const PetscScalar *u,PetscScalar *X,PetscScalar *Xi)
{
  IsoGasCtx *phys = (IsoGasCtx*)vctx;
  PetscReal c = phys->acoustic_speed;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  X[0*2+0] = 1;
  X[0*2+1] = u[1]/u[0] - c;
  X[1*2+0] = 1;
  X[1*2+1] = u[1]/u[0] + c;
  ierr = PetscMemcpy(Xi,X,4*sizeof(X[0]));CHKERRQ(ierr);
  ierr = Kernel_A_gets_inverse_A_2(Xi,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PhysicsCreate_IsoGas"
static PetscErrorCode PhysicsCreate_IsoGas(FVCtx *ctx)
{
  PetscErrorCode ierr;
  IsoGasCtx *user;
  PetscFList rlist = 0,rclist = 0;
  char rname[256] = "exact",rcname[256] = "characteristic";

  PetscFunctionBegin;
  ierr = PetscNew(*user,&user);CHKERRQ(ierr);
  ctx->physics.sample         = PhysicsSample_IsoGas;
  ctx->physics.destroy        = PhysicsDestroy_SimpleFree;
  ctx->physics.user           = user;
  ctx->physics.dof            = 2;
  ierr = PetscStrallocpy("density",&ctx->physics.fieldname[0]);CHKERRQ(ierr);
  ierr = PetscStrallocpy("momentum",&ctx->physics.fieldname[1]);CHKERRQ(ierr);
  user->acoustic_speed = 1;
  ierr = RiemannListAdd(&rlist,"exact",  PhysicsRiemann_IsoGas_Exact);CHKERRQ(ierr);
  ierr = RiemannListAdd(&rlist,"roe",    PhysicsRiemann_IsoGas_Roe);CHKERRQ(ierr);
  ierr = RiemannListAdd(&rlist,"rusanov",PhysicsRiemann_IsoGas_Rusanov);CHKERRQ(ierr);
  ierr = ReconstructListAdd(&rclist,"characteristic",PhysicsCharacteristic_IsoGas);CHKERRQ(ierr);
  ierr = ReconstructListAdd(&rclist,"conservative",PhysicsCharacteristic_Conservative);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(ctx->comm,ctx->prefix,"Options for IsoGas","");CHKERRQ(ierr);
  {
    ierr = PetscOptionsReal("-physics_isogas_acoustic_speed","Acoustic speed","",user->acoustic_speed,&user->acoustic_speed,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsList("-physics_isogas_riemann","Riemann solver","",rlist,rname,rname,sizeof rname,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsList("-physics_isogas_reconstruct","Reconstruction","",rclist,rcname,rcname,sizeof rcname,PETSC_NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  ierr = RiemannListFind(rlist,rname,&ctx->physics.riemann);CHKERRQ(ierr);
  ierr = ReconstructListFind(rclist,rcname,&ctx->physics.characteristic);CHKERRQ(ierr);
  ierr = PetscFListDestroy(&rlist);CHKERRQ(ierr);
  ierr = PetscFListDestroy(&rclist);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



/* --------------------------------- Shallow Water ----------------------------------- */

typedef struct {
  PetscReal gravity;
} ShallowCtx;

static inline void ShallowFlux(ShallowCtx *phys,const PetscScalar *u,PetscScalar *f)
{
  f[0] = u[1];
  f[1] = PetscSqr(u[1])/u[0] + 0.5*phys->gravity*PetscSqr(u[0]);
}

#undef __FUNCT__  
#define __FUNCT__ "PhysicsRiemann_Shallow_Exact"
static PetscErrorCode PhysicsRiemann_Shallow_Exact(void *vctx,PetscInt m,const PetscScalar *uL,const PetscScalar *uR,PetscScalar *flux,PetscReal *maxspeed)
{
  ShallowCtx *phys = (ShallowCtx*)vctx;
  PetscScalar g = phys->gravity,ustar[2],cL,cR,c,cstar;
  struct {PetscScalar h,u;} L = {uL[0],uL[1]/uL[0]},R = {uR[0],uR[1]/uR[0]},star;
  PetscInt i;

  PetscFunctionBegin;
  if (!(L.h > 0 && R.h > 0)) SETERRQ(1,"Reconstructed thickness is negative");
  cL = PetscSqrtScalar(g*L.h);
  cR = PetscSqrtScalar(g*R.h);
  c = PetscMax(cL,cR);
  {
    /* Solve for star state */
    const PetscInt maxits = 50;
    PetscScalar tmp,res,res0=0,h0,h = 0.5*(L.h + R.h); /* initial guess */
    h0 = h;
    for (i=0; i<maxits; i++) {
      PetscScalar fr,fl,dfr,dfl;
      fl = (L.h < h)
        ? PetscSqrtScalar(0.5*g*(h*h - L.h*L.h)*(1/L.h - 1/h)) /* shock */
        : 2*PetscSqrtScalar(g*h) - 2*PetscSqrtScalar(g*L.h);   /* rarefaction */
      fr = (R.h < h)
        ? PetscSqrtScalar(0.5*g*(h*h - R.h*R.h)*(1/R.h - 1/h)) /* shock */
        : 2*PetscSqrtScalar(g*h) - 2*PetscSqrtScalar(g*R.h);   /* rarefaction */
      res = R.u - L.u + fr + fl;
      if (!isfinite(res)) SETERRQ1(1,"non-finite residual=%g",res);
      //PetscPrintf(PETSC_COMM_WORLD,"h=%g, res[%d] = %g\n",h,i,res);
      if (PetscAbsScalar(res) < 1e-8 || (i > 0 && PetscAbsScalar(h-h0) < 1e-8)) {
        star.h = h;
        star.u = L.u - fl;
        goto converged;
      } else if (i > 0 && PetscAbsScalar(res) >= PetscAbsScalar(res0)) {        /* Line search */
        h = 0.8*h0 + 0.2*h;
        continue;
      }
      /* Accept the last step and take another */
      res0 = res;
      h0 = h;
      dfl = (L.h < h)
        ? 0.5/fl*0.5*g*(-L.h*L.h/(h*h) - 1 + 2*h/L.h)
        : PetscSqrtScalar(g/h);
      dfr = (R.h < h)
        ? 0.5/fr*0.5*g*(-R.h*R.h/(h*h) - 1 + 2*h/R.h)
        : PetscSqrtScalar(g/h);
      tmp = h - res/(dfr+dfl);
      if (tmp <= 0) h /= 2;   /* Guard against Newton shooting off to a negative thickness */
      else h = tmp;
      if (!((h > 0) && isnormal(h))) SETERRQ1(1,"non-normal iterate h=%g",h);
    }
    SETERRQ1(1,"Newton iteration for star.h diverged after %d iterations",i);
  }
  converged:
  cstar = PetscSqrtScalar(g*star.h);
  if (L.u-cL < 0 && 0 < star.u-cstar) { /* 1-wave is sonic rarefaction */
    PetscScalar ufan[2];
    ufan[0] = 1/g*PetscSqr(L.u/3 + 2./3*cL);
    ufan[1] = PetscSqrtScalar(g*ufan[0])*ufan[0];
    ShallowFlux(phys,ufan,flux);
  } else if (star.u+cstar < 0 && 0 < R.u+cR) { /* 2-wave is sonic rarefaction */
    PetscScalar ufan[2];
    ufan[0] = 1/g*PetscSqr(R.u/3 - 2./3*cR);
    ufan[1] = -PetscSqrtScalar(g*ufan[0])*ufan[0];
    ShallowFlux(phys,ufan,flux);
  } else if ((L.h >= star.h && L.u-c >= 0)
             || (L.h<star.h && (star.h*star.u-L.h*L.u)/(star.h-L.h) > 0)) {
    /* 1-wave is right-travelling shock (supersonic) */
    ShallowFlux(phys,uL,flux);
  } else if ((star.h <= R.h && R.u+c <= 0)
             || (star.h>R.h && (R.h*R.u-star.h*star.h)/(R.h-star.h) < 0)) {
    /* 2-wave is left-travelling shock (supersonic) */
    ShallowFlux(phys,uR,flux);
  } else {
    ustar[0] = star.h;
    ustar[1] = star.h*star.u;
    ShallowFlux(phys,ustar,flux);
  }
  *maxspeed = MaxAbs(MaxAbs(star.u-cstar,star.u+cstar),MaxAbs(L.u-cL,R.u+cR));
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PhysicsRiemann_Shallow_Rusanov"
static PetscErrorCode PhysicsRiemann_Shallow_Rusanov(void *vctx,PetscInt m,const PetscScalar *uL,const PetscScalar *uR,PetscScalar *flux,PetscReal *maxspeed)
{
  ShallowCtx *phys = (ShallowCtx*)vctx;
  PetscScalar g = phys->gravity,fL[2],fR[2],s;
  struct {PetscScalar h,u;} L = {uL[0],uL[1]/uL[0]},R = {uR[0],uR[1]/uR[0]};

  PetscFunctionBegin;
  if (!(L.h > 0 && R.h > 0)) SETERRQ(1,"Reconstructed thickness is negative");
  ShallowFlux(phys,uL,fL);
  ShallowFlux(phys,uR,fR);
  s = PetscMax(PetscAbs(L.u)+PetscSqrtScalar(g*L.h),PetscAbs(R.u)+PetscSqrtScalar(g*R.h));
  flux[0] = 0.5*(fL[0] + fR[0]) + 0.5*s*(uL[0] - uR[0]);
  flux[1] = 0.5*(fL[1] + fR[1]) + 0.5*s*(uL[1] - uR[1]);
  *maxspeed = s;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PhysicsCharacteristic_Shallow"
static PetscErrorCode PhysicsCharacteristic_Shallow(void *vctx,PetscInt m,const PetscScalar *u,PetscScalar *X,PetscScalar *Xi)
{
  ShallowCtx *phys = (ShallowCtx*)vctx;
  PetscReal c;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  c = PetscSqrtScalar(u[0]*phys->gravity);
  X[0*2+0] = 1;
  X[0*2+1] = u[1]/u[0] - c;
  X[1*2+0] = 1;
  X[1*2+1] = u[1]/u[0] + c;
  ierr = PetscMemcpy(Xi,X,4*sizeof(X[0]));CHKERRQ(ierr);
  ierr = Kernel_A_gets_inverse_A_2(Xi,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PhysicsCreate_Shallow"
static PetscErrorCode PhysicsCreate_Shallow(FVCtx *ctx)
{
  PetscErrorCode ierr;
  ShallowCtx *user;
  PetscFList rlist = 0,rclist = 0;
  char rname[256] = "exact",rcname[256] = "characteristic";

  PetscFunctionBegin;
  ierr = PetscNew(*user,&user);CHKERRQ(ierr);
  /* Shallow water and Isothermal Gas dynamics are similar so we reuse initial conditions for now */
  ctx->physics.sample         = PhysicsSample_IsoGas;
  ctx->physics.destroy        = PhysicsDestroy_SimpleFree;
  ctx->physics.user           = user;
  ctx->physics.dof            = 2;
  ierr = PetscStrallocpy("density",&ctx->physics.fieldname[0]);CHKERRQ(ierr);
  ierr = PetscStrallocpy("momentum",&ctx->physics.fieldname[1]);CHKERRQ(ierr);
  user->gravity = 1;
  ierr = RiemannListAdd(&rlist,"exact",  PhysicsRiemann_Shallow_Exact);CHKERRQ(ierr);
  ierr = RiemannListAdd(&rlist,"rusanov",PhysicsRiemann_Shallow_Rusanov);CHKERRQ(ierr);
  ierr = ReconstructListAdd(&rclist,"characteristic",PhysicsCharacteristic_Shallow);CHKERRQ(ierr);
  ierr = ReconstructListAdd(&rclist,"conservative",PhysicsCharacteristic_Conservative);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(ctx->comm,ctx->prefix,"Options for Shallow","");CHKERRQ(ierr);
  {
    ierr = PetscOptionsReal("-physics_shallow_gravity","Gravity","",user->gravity,&user->gravity,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsList("-physics_shallow_riemann","Riemann solver","",rlist,rname,rname,sizeof rname,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsList("-physics_shallow_reconstruct","Reconstruction","",rclist,rcname,rcname,sizeof rcname,PETSC_NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  ierr = RiemannListFind(rlist,rname,&ctx->physics.riemann);CHKERRQ(ierr);
  ierr = ReconstructListFind(rclist,rcname,&ctx->physics.characteristic);CHKERRQ(ierr);
  ierr = PetscFListDestroy(&rlist);CHKERRQ(ierr);
  ierr = PetscFListDestroy(&rclist);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



/* --------------------------------- Finite Volume Solver ----------------------------------- */

#undef __FUNCT__  
#define __FUNCT__ "FVRHSFunction"
static PetscErrorCode FVRHSFunction(TS ts,PetscReal time,Vec X,Vec F,void *vctx)
{
  FVCtx          *ctx = (FVCtx*)vctx;
  PetscErrorCode  ierr;
  PetscInt        i,j,k,Mx,dof,xs,xm;
  PetscReal       hx,cfl_idt = 0;
  PetscScalar    *x,*f,*slope;
  Vec             Xloc;

  PetscFunctionBegin;
  ierr = DAGetLocalVector(ctx->da,&Xloc);CHKERRQ(ierr);
  ierr = DAGetInfo(ctx->da,0, &Mx,0,0, 0,0,0, &dof,0,0,0);CHKERRQ(ierr);
  hx = (ctx->xmax - ctx->xmin)/Mx;
  ierr = DAGlobalToLocalBegin(ctx->da,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd  (ctx->da,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);

  ierr = VecZeroEntries(F);CHKERRQ(ierr);

  ierr = DAVecGetArray(ctx->da,Xloc,&x);CHKERRQ(ierr);
  ierr = DAVecGetArray(ctx->da,F,&f);CHKERRQ(ierr);
  ierr = DAGetArray(ctx->da,PETSC_TRUE,(void**)&slope);CHKERRQ(ierr);

  ierr = DAGetCorners(ctx->da,&xs,0,0,&xm,0,0);CHKERRQ(ierr);

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
    PetscScalar *cjmpL,*cjmpR;
    /* Determine the right eigenvectors R, where A = R \Lambda R^{-1} */
    ierr = (*ctx->physics.characteristic)(ctx->physics.user,dof,&x[i*dof],ctx->R,ctx->Rinv);CHKERRQ(ierr);
    /* Evaluate jumps across interfaces (i-1, i) and (i, i+1), put in characteristic basis */
    ierr = PetscMemzero(ctx->cjmpLR,2*dof*sizeof(ctx->cjmpLR[0]));CHKERRQ(ierr);
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
    info.m = dof;
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
    PetscReal maxspeed;
    PetscScalar *uL,*uR;
    uL = &ctx->uLR[0];
    uR = &ctx->uLR[dof];
    for (j=0; j<dof; j++) {
      uL[j] = x[(i-1)*dof+j] + slope[(i-1)*dof+j]*hx/2;
      uR[j] = x[(i-0)*dof+j] - slope[(i-0)*dof+j]*hx/2;
    }
    ierr = (*ctx->physics.riemann)(ctx->physics.user,dof,uL,uR,ctx->flux,&maxspeed);CHKERRQ(ierr);
    cfl_idt = PetscMax(cfl_idt,PetscAbsScalar(maxspeed/hx)); /* Max allowable value of 1/Delta t */

    if (i > xs) {
      for (j=0; j<dof; j++) f[(i-1)*dof+j] -= ctx->flux[j]/hx;
    }
    if (i < xs+xm) {
      for (j=0; j<dof; j++) f[i*dof+j] += ctx->flux[j]/hx;
    }
  }

  ierr = DAVecRestoreArray(ctx->da,Xloc,&x);CHKERRQ(ierr);
  ierr = DAVecRestoreArray(ctx->da,F,&f);CHKERRQ(ierr);
  ierr = DARestoreArray(ctx->da,PETSC_TRUE,(void**)&slope);CHKERRQ(ierr);
  ierr = DARestoreLocalVector(ctx->da,&Xloc);CHKERRQ(ierr);

  ierr = PetscGlobalMax(&cfl_idt,&ctx->cfl_idt,((PetscObject)ctx->da)->comm);CHKERRQ(ierr);
  if (0) {
    /* We need to a way to inform the TS of a CFL constraint, this is a debugging fragment */
    PetscReal dt,tnow;
    ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
    ierr = TSGetTime(ts,&tnow);CHKERRQ(ierr);
    if (dt > 0.5/ctx->cfl_idt) {
      if (1) {
        ierr = PetscPrintf(ctx->comm,"Stability constraint exceeded at t=%g, dt %g > %g\n",tnow,dt,0.5/ctx->cfl_idt);CHKERRQ(ierr);
      } else {
        SETERRQ2(1,"Stability constraint exceeded, %g > %g",dt,ctx->cfl/ctx->cfl_idt);
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "FVSample"
static PetscErrorCode FVSample(FVCtx *ctx,PetscReal time,Vec U)
{
  PetscErrorCode ierr;
  PetscScalar *u,*uj;
  PetscInt i,j,k,dof,xs,xm,Mx;

  PetscFunctionBegin;
  if (!ctx->physics.sample) SETERRQ(1,"Physics has not provided a sampling function");
  ierr = DAGetInfo(ctx->da,0, &Mx,0,0, 0,0,0, &dof,0,0,0);CHKERRQ(ierr);
  ierr = DAGetCorners(ctx->da,&xs,0,0,&xm,0,0);CHKERRQ(ierr);
  ierr = DAVecGetArray(ctx->da,U,&u);CHKERRQ(ierr);
  ierr = PetscMalloc(dof*sizeof uj[0],&uj);CHKERRQ(ierr);
  for (i=xs; i<xs+xm; i++) {
    const PetscReal h = (ctx->xmax-ctx->xmin)/Mx,xi = ctx->xmin+h/2+i*h;
    const PetscInt N = 200;
    /* Integrate over cell i using trapezoid rule with N points. */
    for (k=0; k<dof; k++) u[i*dof+k] = 0;
    for (j=0; j<N+1; j++) {
      PetscScalar xj = xi+h*(j-N/2)/(PetscReal)N;
      ierr = (*ctx->physics.sample)(ctx->physics.user,ctx->initial,ctx->bctype,ctx->xmin,ctx->xmax,time,xj,uj);CHKERRQ(ierr);
      for (k=0; k<dof; k++) u[i*dof+k] += ((j==0 || j==N)?0.5:1.0)*uj[k]/N;
    }
  }
  ierr = DAVecRestoreArray(ctx->da,U,&u);CHKERRQ(ierr);
  ierr = PetscFree(uj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SolutionStatsView"
static PetscErrorCode SolutionStatsView(DA da,Vec X,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscReal xmin,xmax;
  PetscScalar sum,*x,tvsum,tvgsum;
  PetscInt imin,imax,Mx,i,j,xs,xm,dof;
  Vec Xloc;
  PetscTruth iascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    /* PETSc lacks a function to compute total variation norm (difficult in multiple dimensions), we do it here */
    ierr = DAGetLocalVector(da,&Xloc);CHKERRQ(ierr);
    ierr = DAGlobalToLocalBegin(da,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
    ierr = DAGlobalToLocalEnd  (da,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
    ierr = DAVecGetArray(da,Xloc,&x);CHKERRQ(ierr);
    ierr = DAGetCorners(da,&xs,0,0,&xm,0,0);CHKERRQ(ierr);
    ierr = DAGetInfo(da,0, &Mx,0,0, 0,0,0, &dof,0,0,0);CHKERRQ(ierr);
    tvsum = 0;
    for (i=xs; i<xs+xm; i++) {
      for (j=0; j<dof; j++) tvsum += PetscAbsScalar(x[i*dof+j] - x[(i-1)*dof+j]);
    }
    ierr = PetscGlobalMax(&tvsum,&tvgsum,((PetscObject)da)->comm);CHKERRQ(ierr);
    ierr = DAVecRestoreArray(da,Xloc,&x);CHKERRQ(ierr);
    ierr = DARestoreLocalVector(da,&Xloc);CHKERRQ(ierr);

    ierr = VecMin(X,&imin,&xmin);CHKERRQ(ierr);
    ierr = VecMax(X,&imax,&xmax);CHKERRQ(ierr);
    ierr = VecSum(X,&sum);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Solution range [%8.5f,%8.5f] with extrema at %d and %d, mean %8.5f, ||x||_TV %8.5f\n",xmin,xmax,imin,imax,sum/Mx,tvgsum/Mx);CHKERRQ(ierr);
  } else {
    SETERRQ(1,"Viewer type not supported");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SolutionErrorNorms"
static PetscErrorCode SolutionErrorNorms(FVCtx *ctx,PetscReal t,Vec X,PetscReal *nrm1,PetscReal *nrmsup)
{
  PetscErrorCode ierr;
  Vec Y;
  PetscInt Mx;

  PetscFunctionBegin;
  ierr = VecGetSize(X,&Mx);CHKERRQ(ierr);
  ierr = VecDuplicate(X,&Y);CHKERRQ(ierr);
  ierr = FVSample(ctx,t,Y);CHKERRQ(ierr);
  ierr = VecAYPX(Y,-1,X);CHKERRQ(ierr);
  ierr = VecNorm(Y,NORM_1,nrm1);CHKERRQ(ierr);
  ierr = VecNorm(Y,NORM_INFINITY,nrmsup);CHKERRQ(ierr);
  *nrm1 /= Mx;
  ierr = VecDestroy(Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "main"
int main(int argc,char *argv[])
{
  char lname[256] = "mc",physname[256] = "advect",final_fname[256] = "solution.m";
  PetscFList limiters = 0,physics = 0;
  MPI_Comm comm;
  TS ts;
  Vec X,X0;
  FVCtx ctx;
  PetscInt i,dof,xs,xm,Mx,draw = 0;
  PetscTruth view_final = PETSC_FALSE;
  PetscReal ptime;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,0,help);CHKERRQ(ierr);
  comm = PETSC_COMM_WORLD;
  ierr = PetscMemzero(&ctx,sizeof(ctx));CHKERRQ(ierr);

  /* Register limiters to be available on the command line */
  ierr = PetscFListAdd(&limiters,"upwind"          ,"",(void(*)(void))Limit_Upwind);CHKERRQ(ierr);
  ierr = PetscFListAdd(&limiters,"lax-wendroff"    ,"",(void(*)(void))Limit_LaxWendroff);CHKERRQ(ierr);
  ierr = PetscFListAdd(&limiters,"beam-warming"    ,"",(void(*)(void))Limit_BeamWarming);CHKERRQ(ierr);
  ierr = PetscFListAdd(&limiters,"fromm"           ,"",(void(*)(void))Limit_Fromm);CHKERRQ(ierr);
  ierr = PetscFListAdd(&limiters,"minmod"          ,"",(void(*)(void))Limit_Minmod);CHKERRQ(ierr);
  ierr = PetscFListAdd(&limiters,"superbee"        ,"",(void(*)(void))Limit_Superbee);CHKERRQ(ierr);
  ierr = PetscFListAdd(&limiters,"mc"              ,"",(void(*)(void))Limit_MC);CHKERRQ(ierr);
  ierr = PetscFListAdd(&limiters,"vanleer"         ,"",(void(*)(void))Limit_VanLeer);CHKERRQ(ierr);
  ierr = PetscFListAdd(&limiters,"vanalbada"       ,"",(void(*)(void))Limit_VanAlbada);CHKERRQ(ierr);
  ierr = PetscFListAdd(&limiters,"vanalbadatvd"    ,"",(void(*)(void))Limit_VanAlbadaTVD);CHKERRQ(ierr);
  ierr = PetscFListAdd(&limiters,"koren"           ,"",(void(*)(void))Limit_Koren);CHKERRQ(ierr);
  ierr = PetscFListAdd(&limiters,"korensym"        ,"",(void(*)(void))Limit_KorenSym);CHKERRQ(ierr);
  ierr = PetscFListAdd(&limiters,"koren3"          ,"",(void(*)(void))Limit_Koren3);CHKERRQ(ierr);
  ierr = PetscFListAdd(&limiters,"cada-torrilhon2" ,"",(void(*)(void))Limit_CadaTorrilhon2);CHKERRQ(ierr);
  ierr = PetscFListAdd(&limiters,"cada-torrilhon3-r0p1","",(void(*)(void))Limit_CadaTorrilhon3R0p1);CHKERRQ(ierr);
  ierr = PetscFListAdd(&limiters,"cada-torrilhon3-r1"  ,"",(void(*)(void))Limit_CadaTorrilhon3R1);CHKERRQ(ierr);
  ierr = PetscFListAdd(&limiters,"cada-torrilhon3-r10" ,"",(void(*)(void))Limit_CadaTorrilhon3R10);CHKERRQ(ierr);
  ierr = PetscFListAdd(&limiters,"cada-torrilhon3-r100","",(void(*)(void))Limit_CadaTorrilhon3R100);CHKERRQ(ierr);

  /* Register physical models to be available on the command line */
  ierr = PetscFListAdd(&physics,"advect"          ,"",(void(*)(void))PhysicsCreate_Advect);CHKERRQ(ierr);
  ierr = PetscFListAdd(&physics,"burgers"         ,"",(void(*)(void))PhysicsCreate_Burgers);CHKERRQ(ierr);
  ierr = PetscFListAdd(&physics,"traffic"         ,"",(void(*)(void))PhysicsCreate_Traffic);CHKERRQ(ierr);
  ierr = PetscFListAdd(&physics,"isogas"          ,"",(void(*)(void))PhysicsCreate_IsoGas);CHKERRQ(ierr);
  ierr = PetscFListAdd(&physics,"shallow"         ,"",(void(*)(void))PhysicsCreate_Shallow);CHKERRQ(ierr);

  ctx.cfl = 0.9; ctx.bctype = FVBC_PERIODIC;
  ctx.xmin = 0; ctx.xmax = 1;
  ierr = PetscOptionsBegin(comm,PETSC_NULL,"Finite Volume solver options","");CHKERRQ(ierr);
  {
    ierr = PetscOptionsReal("-xmin","X min","",ctx.xmin,&ctx.xmin,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-xmax","X max","",ctx.xmax,&ctx.xmax,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsList("-limit","Name of flux limiter to use","",limiters,lname,lname,sizeof(lname),PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsList("-physics","Name of physics (Riemann solver and characteristics) to use","",physics,physname,physname,sizeof(physname),PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-draw","Draw solution vector, bitwise OR of (1=initial,2=final,4=final error)","",draw,&draw,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsString("-view_final","Write final solution in ASCII Matlab format to given file name","",final_fname,final_fname,sizeof final_fname,&view_final);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-initial","Initial condition (depends on the physics)","",ctx.initial,&ctx.initial,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-exact","Compare errors with exact solution","",ctx.exact,&ctx.exact,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-cfl","CFL number to time step at","",ctx.cfl,&ctx.cfl,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEnum("-bc_type","Boundary condition","",FVBCTypes,ctx.bctype,(PetscEnum*)&ctx.bctype,PETSC_NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* Choose the limiter from the list of registered limiters */
  ierr = PetscFListFind(limiters,comm,lname,(void(**)(void))&ctx.limit);CHKERRQ(ierr);
  if (!ctx.limit) SETERRQ1(1,"Limiter '%s' not found",lname);CHKERRQ(ierr);

  /* Choose the physics from the list of registered models */
  {
    PetscErrorCode (*r)(FVCtx*);
    ierr = PetscFListFind(physics,comm,physname,(void(**)(void))&r);CHKERRQ(ierr);
    if (!r) SETERRQ1(1,"Physics '%s' not found",physname);CHKERRQ(ierr);
    /* Create the physics, will set the number of fields and their names */
    ierr = (*r)(&ctx);CHKERRQ(ierr);
  }

  /* Create a DA to manage the parallel grid */
  ierr = DACreate1d(comm,DA_XPERIODIC,-50,ctx.physics.dof,2,PETSC_NULL,&ctx.da);CHKERRQ(ierr);
  /* Inform the DA of the field names provided by the physics. */
  /* The names will be shown in the title bars when run with -ts_monitor_solution */
  for (i=0; i<ctx.physics.dof; i++) {
    ierr = DASetFieldName(ctx.da,i,ctx.physics.fieldname[i]);CHKERRQ(ierr);
  }
  /* Allow customization of the DA at runtime, mostly to change problem size with -da_grid_x M */
  ierr = DASetFromOptions(ctx.da);CHKERRQ(ierr);
  ierr = DAGetInfo(ctx.da,0, &Mx,0,0, 0,0,0, &dof,0,0,0);CHKERRQ(ierr);
  ierr = DAGetCorners(ctx.da,&xs,0,0,&xm,0,0);CHKERRQ(ierr);

  /* Set coordinates of cell centers */
  ierr = DASetUniformCoordinates(ctx.da,ctx.xmin+0.5*(ctx.xmax-ctx.xmin)/Mx,ctx.xmax+0.5*(ctx.xmax-ctx.xmin)/Mx,0,0,0,0);CHKERRQ(ierr);

  /* Allocate work space for the Finite Volume solver (so it doesn't have to be reallocated on each function evaluation) */
  ierr = PetscMalloc4(dof*dof,PetscScalar,&ctx.R,dof*dof,PetscScalar,&ctx.Rinv,2*dof,PetscScalar,&ctx.cjmpLR,1*dof,PetscScalar,&ctx.cslope);CHKERRQ(ierr);
  ierr = PetscMalloc2(2*dof,PetscScalar,&ctx.uLR,dof,PetscScalar,&ctx.flux);CHKERRQ(ierr);

  /* Create a vector to store the solution and to save the initial state */
  ierr = DACreateGlobalVector(ctx.da,&X);CHKERRQ(ierr);
  ierr = VecDuplicate(X,&X0);CHKERRQ(ierr);

  /* Create a time-stepping object */
  ierr = TSCreate(comm,&ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,FVRHSFunction,&ctx);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSSSP);CHKERRQ(ierr);
  ierr = TSSetDuration(ts,1000,10);CHKERRQ(ierr);

  /* Compute initial conditions and starting time step */
  ierr = FVSample(&ctx,0,X0);CHKERRQ(ierr);
  ierr = FVRHSFunction(ts,0,X0,X,(void*)&ctx);CHKERRQ(ierr); /* Initial function evaluation, only used to determine max speed */
  ierr = VecCopy(X0,X);CHKERRQ(ierr);                        /* The function value was not used so we set X=X0 again */
  ierr = TSSetInitialTimeStep(ts,0,ctx.cfl/ctx.cfl_idt);CHKERRQ(ierr);

  ierr = TSSetSolution(ts,X);CHKERRQ(ierr);  /* The TS will use X for the solution, starting with it's current value as initial condition */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr); /* Take runtime options */

  ierr = SolutionStatsView(ctx.da,X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  {
    PetscReal nrm1,nrmsup;
    PetscInt steps;

    ierr = TSStep(ts,&steps,&ptime);CHKERRQ(ierr);

    ierr = PetscPrintf(comm,"Final time %8.5f, steps %d\n",ptime,steps);CHKERRQ(ierr);
    if (ctx.exact) {
      ierr = SolutionErrorNorms(&ctx,ptime,X,&nrm1,&nrmsup);CHKERRQ(ierr);
      ierr = PetscPrintf(comm,"Error ||x-x_e||_1 %8.4e  ||x-x_e||_sup %8.4e\n",nrm1,nrmsup);CHKERRQ(ierr);
    }
  }

  ierr = SolutionStatsView(ctx.da,X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  if (draw & 0x1) {ierr = VecView(X0,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}
  if (draw & 0x2) {ierr = VecView(X,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}
  if (draw & 0x4) {
    Vec Y;
    ierr = VecDuplicate(X,&Y);CHKERRQ(ierr);
    ierr = FVSample(&ctx,ptime,Y);CHKERRQ(ierr);
    ierr = VecAYPX(Y,-1,X);CHKERRQ(ierr);
    ierr = VecView(Y,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
    ierr = VecDestroy(Y);CHKERRQ(ierr);
  }

  if (view_final) {
    PetscViewer viewer;
    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD,final_fname,&viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(viewer,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
    ierr = VecView(X,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
  }

  /* Clean up */
  ierr = (*ctx.physics.destroy)(ctx.physics.user);CHKERRQ(ierr);
  for (i=0; i<ctx.physics.dof; i++) {ierr = PetscFree(ctx.physics.fieldname[i]);CHKERRQ(ierr);}
  ierr = PetscFree4(ctx.R,ctx.Rinv,ctx.cjmpLR,ctx.cslope);CHKERRQ(ierr);
  ierr = PetscFree2(ctx.uLR,ctx.flux);CHKERRQ(ierr);
  ierr = VecDestroy(X);CHKERRQ(ierr);
  ierr = VecDestroy(X0);CHKERRQ(ierr);
  ierr = DADestroy(ctx.da);CHKERRQ(ierr);
  ierr = TSDestroy(ts);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
