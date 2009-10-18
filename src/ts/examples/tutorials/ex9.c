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

static inline PetscReal MaxAbs(PetscReal a,PetscReal b) { return (PetscAbs(a) > PetscAbs(b)) ? a : b; }


/* ----------------------- Lots of limiters, these could go in a separate library ------------------------- */
static void Limit_Upwind(PetscInt m,const PetscScalar *t,PetscScalar *lmt)
{
  PetscInt i;
  for (i=0; i<m; i++) lmt[i] = 0;
}
static void Limit_LaxWendroff(PetscInt m,const PetscScalar *t,PetscScalar *lmt)
{
  PetscInt i;
  for (i=0; i<m; i++) lmt[i] = 1;
}
static void Limit_BeamWarming(PetscInt m,const PetscScalar *t,PetscScalar *lmt)
{
  PetscInt i;
  for (i=0; i<m; i++) lmt[i] = t[i];
}
static void Limit_Fromm(PetscInt m,const PetscScalar *t,PetscScalar *lmt)
{
  PetscInt i;
  for (i=0; i<m; i++) lmt[i] = 0.5*(1+t[i]);
}
static void Limit_Minmod(PetscInt m,const PetscScalar *t,PetscScalar *lmt)
{
  PetscInt i;
  for (i=0; i<m; i++) lmt[i] = PetscMax(0,PetscMin(1,t[i]));
}
static void Limit_Superbee(PetscInt m,const PetscScalar *t,PetscScalar *lmt)
{
  PetscInt i;
  for (i=0; i<m; i++) lmt[i] = PetscMax(0,PetscMax(PetscMin(1,2*t[i]),PetscMin(2,t[i])));
}
static void Limit_MC(PetscInt m,const PetscScalar *t,PetscScalar *lmt)
{
  PetscInt i;
  for (i=0; i<m; i++) lmt[i] = PetscMax(0,PetscMin((1+t[i])/2,PetscMin(2,2*t[i])));
}
static void Limit_VanLeer(PetscInt m,const PetscScalar *t,PetscScalar *lmt)
{
  PetscInt i;
  for (i=0; i<m; i++) lmt[i] = (t[i] + PetscAbsScalar(t[i])) / (1 + PetscAbsScalar(t[i]));
}
static void Limit_VanAlbada(PetscInt m,const PetscScalar *t,PetscScalar *lmt) /* differentiable */
{
  PetscInt i;
  for (i=0; i<m; i++) lmt[i] = (t[i] + PetscSqr(t[i])) / (1 + PetscSqr(t[i]));
}
static void Limit_Koren(PetscInt m,const PetscScalar *t,PetscScalar *lmt) /* differentiable and less negative */
{
  PetscInt i;
  for (i=0; i<m; i++) lmt[i] = (t[i] + 2*PetscSqr(t[i])) / (2 - t[i] + 2*PetscSqr(t[i]));
}
static void Limit_CadaTorrilhon(PetscInt m,const PetscScalar *t,PetscScalar *lmt) /* Cada-Torrilhon 2009 */
{
  PetscInt i;
  for (i=0; i<m; i++) lmt[i] = PetscMax(0,PetscMin((2+t[i])/3,
                                                   PetscMax(-0.5*t[i],
                                                            PetscMin(2*t[i],
                                                                     PetscMin((2+t[i])/3,1.6)))));
}


/* --------------------------------- Finite Volume data structures ----------------------------------- */

typedef struct {
  PetscErrorCode (*riemann)(void*,PetscInt,const PetscScalar*,const PetscScalar*,PetscScalar*,PetscReal*);
  PetscErrorCode (*characteristic)(void*,PetscInt,const PetscScalar*,PetscScalar*,PetscScalar*);
  PetscErrorCode (*destroy)(void*);
  void *user;
  PetscInt dof;
  char *fieldname[16];
} PhysicsCtx;

typedef struct {
  void (*limit)(PetscInt,const PetscScalar*,PetscScalar*);
  PhysicsCtx physics;

  MPI_Comm comm;
  char prefix[256];
  DA da;
  /* Local work arrays */
  PetscScalar *R,*Rinv;         /* Characteristic basis, and it's inverse.  COLUMN-MAJOR */
  PetscScalar *cjmpLR;          /* Jumps at left and right edge of cell, in characteristic basis, len=2*dof */
  PetscScalar *cslope;          /* Limited slope, written in characteristic basis */
  PetscScalar *theta;           /* Ratio of jumps in characteristic basis, for limiting */
  PetscScalar *lmt;             /* Limiter for each characteristic */
  PetscScalar *uLR;             /* Solution at left and right of interface, conservative variables, len=2*dof */
  PetscScalar *flux;            /* Flux across interface */

  PetscReal cfl_idt;            /* Max allowable value of 1/Delta t */
  PetscReal cfl;
  PetscInt initial;
  PetscInt exact;
} FVCtx;



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
#define __FUNCT__ "PhysicsDestroy_Null"  
static PetscErrorCode PhysicsDestroy_Null(void *vctx)
{
  PetscFunctionBegin;
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
#define __FUNCT__ "PhysicsCreate_Advect"
static PetscErrorCode PhysicsCreate_Advect(FVCtx *ctx)
{
  PetscErrorCode ierr;
  AdvectCtx *user;

  PetscFunctionBegin;
  ierr = PetscNew(*user,&user);CHKERRQ(ierr);
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
#define __FUNCT__ "PhysicsCreate_Burgers"
static PetscErrorCode PhysicsCreate_Burgers(FVCtx *ctx)
{
  PetscErrorCode ierr;
  char rname[256] = "exact";

  PetscFunctionBegin;
  ctx->physics.characteristic = PhysicsCharacteristic_Conservative;
  ctx->physics.destroy        = PhysicsDestroy_Null;
  ctx->physics.user           = 0;
  ctx->physics.dof            = 1;
  ierr = PetscStrallocpy("u",&ctx->physics.fieldname[0]);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(ctx->comm,ctx->prefix,"Options for advection","");CHKERRQ(ierr);
  {
    ierr = PetscOptionsString("-physics_burgers_riemann","Riemann solver to use (exact,roe)","",rname,rname,sizeof(rname),PETSC_NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (!strcmp(rname,"exact")) {
    ctx->physics.riemann = PhysicsRiemann_Burgers_Exact;
  } else if (!strcmp(rname,"roe")) {
    ctx->physics.riemann = PhysicsRiemann_Burgers_Roe;
  } else {
    SETERRQ1(1,"Riemann solver %s not available for Burgers equation",rname);
  }
  PetscFunctionReturn(0);
}



/* --------------------------------- Traffic ----------------------------------- */

typedef struct {
  PetscReal max_speed;
} TrafficCtx;

static inline PetscScalar TrafficFlux(PetscScalar maxspeed,PetscScalar u) { return maxspeed*u*(1-u); }

#undef __FUNCT__  
#define __FUNCT__ "PhysicsRiemann_Traffic_Exact"
static PetscErrorCode PhysicsRiemann_Traffic_Exact(void *vctx,PetscInt m,const PetscScalar *uL,const PetscScalar *uR,PetscScalar *flux,PetscReal *maxspeed)
{
  TrafficCtx *phys = (TrafficCtx*)vctx;

  PetscFunctionBegin;
#if 0
  flux[0] = PetscMin(TrafficFlux(phys->max_speed,PetscMax(uL[0],0.5)),
                     TrafficFlux(phys->max_speed,PetscMin(uR[0],0.5)));
#else
  if (uL[0] < uR[0]) {
    flux[0] = PetscMin(TrafficFlux(phys->max_speed,uL[0]),
                       TrafficFlux(phys->max_speed,uR[0]));
  } else {
    flux[0] = (uR[0] < 0.5 && 0.5 < uL[0])
      ? TrafficFlux(phys->max_speed,0.5)
      : PetscMax(TrafficFlux(phys->max_speed,uL[0]),
                 TrafficFlux(phys->max_speed,uR[0]));
  }
#endif
  *maxspeed = phys->max_speed;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PhysicsRiemann_Traffic_Roe"
static PetscErrorCode PhysicsRiemann_Traffic_Roe(void *vctx,PetscInt m,const PetscScalar *uL,const PetscScalar *uR,PetscScalar *flux,PetscReal *maxspeed)
{
  TrafficCtx *phys = (TrafficCtx*)vctx;
  PetscReal speed;

  PetscFunctionBegin;
  speed = phys->max_speed*(1. - 0.5*(uL[0] + uR[0]));
  flux[0] = PetscMax(0,speed)*uL[0] + PetscMin(0,speed)*uR[0];
  *maxspeed = speed;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PhysicsCreate_Traffic"
static PetscErrorCode PhysicsCreate_Traffic(FVCtx *ctx)
{
  PetscErrorCode ierr;
  TrafficCtx *user;
  char rname[256] = "exact";

  PetscFunctionBegin;
  ierr = PetscNew(*user,&user);CHKERRQ(ierr);
  ctx->physics.characteristic = PhysicsCharacteristic_Conservative;
  ctx->physics.destroy        = PhysicsDestroy_SimpleFree;
  ctx->physics.user           = user;
  ctx->physics.dof            = 1;
  ierr = PetscStrallocpy("density",&ctx->physics.fieldname[0]);CHKERRQ(ierr);
  user->max_speed = 1;
  ierr = PetscOptionsBegin(ctx->comm,ctx->prefix,"Options for Traffic","");CHKERRQ(ierr);
  {
    ierr = PetscOptionsReal("-physics_traffic_max_speed","Maximum speed","",user->max_speed,&user->max_speed,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsString("-physics_traffic_riemann","Riemann solver to use (exact,roe)","",rname,rname,sizeof(rname),PETSC_NULL);CHKERRQ(ierr);
  }
  if (!strcmp(rname,"exact")) {
    ctx->physics.riemann = PhysicsRiemann_Traffic_Exact;
  } else if (!strcmp(rname,"roe")) {
    ctx->physics.riemann = PhysicsRiemann_Traffic_Roe;
  } else {
    SETERRQ1(1,"Riemann solver %s not available for Traffic equation",rname);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
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
  if (L.u-c > 0) { /* right supersonic */
    IsoGasFlux(c,uL,flux);
    *maxspeed = L.u+c;
  } else if (R.u+c < 0) { /* left supersonic */
    IsoGasFlux(c,uR,flux);
    *maxspeed = R.u-c;
  } else {
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
        break;
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
    if (i == 20) SETERRQ(1,"Newton iteration for star.rho diverged after %d iterations");
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
    } else {
      ustar[0] = star.rho;
      ustar[1] = star.rho*star.u;
      IsoGasFlux(c,ustar,flux);
    }
    *maxspeed = MaxAbs(MaxAbs(star.u-c,star.u+c),MaxAbs(L.u-c,R.u+c));
  }
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
  char rname[256] = "exact",rcname[256] = "characteristic";

  PetscFunctionBegin;
  ierr = PetscNew(*user,&user);CHKERRQ(ierr);
  ctx->physics.destroy        = PhysicsDestroy_SimpleFree;
  ctx->physics.user           = user;
  ctx->physics.dof            = 2;
  ierr = PetscStrallocpy("density",&ctx->physics.fieldname[0]);CHKERRQ(ierr);
  ierr = PetscStrallocpy("momentum",&ctx->physics.fieldname[1]);CHKERRQ(ierr);
  user->acoustic_speed = 1;
  ierr = PetscOptionsBegin(ctx->comm,ctx->prefix,"Options for IsoGas","");CHKERRQ(ierr);
  {
    ierr = PetscOptionsReal("-physics_isogas_acoustic_speed","Acoustic speed","",user->acoustic_speed,&user->acoustic_speed,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsString("-physics_isogas_riemann","Riemann solver to use (exact,roe)","",rname,rname,sizeof(rname),PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsString("-physics_isogas_reconstruct","Reconstruction to use (characteristic,primal)","",rcname,rcname,sizeof(rcname),PETSC_NULL);CHKERRQ(ierr);
  }
  if (!strcmp(rname,"exact")) {
    ctx->physics.riemann = PhysicsRiemann_IsoGas_Exact;
  } else if (!strcmp(rname,"roe")) {
    ctx->physics.riemann = PhysicsRiemann_IsoGas_Roe;
  } else {
    SETERRQ1(1,"Riemann solver %s not available for IsoGas equation",rname);
  }
  if (!strcmp(rcname,"characteristic")) {
    ctx->physics.characteristic = PhysicsCharacteristic_IsoGas;
  } else if (!strcmp(rcname,"conservative")) {
    ctx->physics.characteristic = PhysicsCharacteristic_Conservative;
  } else {
    SETERRQ1(1,"Reconstruction %s not available for IsoGas equation",rcname);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
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
  PetscScalar g = phys->gravity,ustar[2],cL,cR;
  struct {PetscScalar h,u;} L = {uL[0],uL[1]/uL[0]},R = {uR[0],uR[1]/uR[0]},star;
  PetscInt i;

  PetscFunctionBegin;
  if (!(L.h > 0 && R.h > 0)) SETERRQ(1,"Reconstructed thickness is negative");
  cL = PetscSqrtScalar(g*L.h);
  cR = PetscSqrtScalar(g*R.h);
  if (L.u-cL > 0) { /* right supersonic */
    ShallowFlux(phys,uL,flux);
    *maxspeed = L.u+cL;
  } else if (R.u+cR < 0) { /* left supersonic */
    ShallowFlux(phys,uR,flux);
    *maxspeed = R.u-cR;
  } else {
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
      if (i > 0 && PetscAbsScalar(res) >= PetscAbsScalar(res0)) {        /* Line search */
        h = 0.8*h0 + 0.2*h;
        continue;
      }
      if (PetscAbsScalar(res) < 1e-8 || (i > 0 && PetscAbsScalar(h-h0) < 1e-8)) {
        star.h = h;
        star.u = L.u - fl;
        break;
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
    if (i == maxits) SETERRQ(1,"Newton iteration for star.h diverged after %d iterations");
    if (0 < star.u - PetscSqrtScalar(g*star.h)) { /* 1-wave is sonic rarefaction */
      PetscScalar ufan[2];
      ufan[0] = 1/g*PetscSqr(L.u/3 + 2./3*PetscSqrtScalar(g*L.h));
      ufan[1] = PetscSqrtScalar(g*ufan[0])*ufan[0];
      ShallowFlux(phys,ufan,flux);
      *maxspeed = MaxAbs(star.u+PetscSqrtScalar(g*star.h),R.u+PetscSqrtScalar(g*R.h));
    } else if (star.u + PetscSqrtScalar(g*star.h) < 0) { /* 2-wave is sonic rarefaction */
      PetscScalar ufan[2];
      ufan[0] = 1/g*PetscSqr(R.u/3 - 2./3*PetscSqrtScalar(g*R.h));
      ufan[1] = -PetscSqrtScalar(g*ufan[0])*ufan[0];
      ShallowFlux(phys,ufan,flux);
      *maxspeed = MaxAbs(star.u-PetscSqrtScalar(g*star.h),L.u-PetscSqrtScalar(g*L.h));
    } else {
      ustar[0] = star.h;
      ustar[1] = star.h*star.u;
      ShallowFlux(phys,ustar,flux);
      *maxspeed = MaxAbs(MaxAbs(star.u-PetscSqrtScalar(g*star.h),star.u-PetscSqrtScalar(g*star.h)),
                         MaxAbs(L.u-PetscSqrtScalar(g*L.h),R.u+PetscSqrtScalar(g*R.h)));
    }
  }
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
  char rname[256] = "exact",rcname[256] = "characteristic";

  PetscFunctionBegin;
  ierr = PetscNew(*user,&user);CHKERRQ(ierr);
  ctx->physics.destroy        = PhysicsDestroy_SimpleFree;
  ctx->physics.user           = user;
  ctx->physics.dof            = 2;
  ierr = PetscStrallocpy("density",&ctx->physics.fieldname[0]);CHKERRQ(ierr);
  ierr = PetscStrallocpy("momentum",&ctx->physics.fieldname[1]);CHKERRQ(ierr);
  user->gravity = 1;
  ierr = PetscOptionsBegin(ctx->comm,ctx->prefix,"Options for Shallow","");CHKERRQ(ierr);
  {
    ierr = PetscOptionsReal("-physics_shallow_gravity","Gravity","",user->gravity,&user->gravity,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsString("-physics_shallow_riemann","Riemann solver to use (exact,roe)","",rname,rname,sizeof(rname),PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsString("-physics_shallow_reconstruct","Reconstruction to use (characteristic,primal)","",rcname,rcname,sizeof(rcname),PETSC_NULL);CHKERRQ(ierr);
  }
  if (!strcmp(rname,"exact")) {
    ctx->physics.riemann = PhysicsRiemann_Shallow_Exact;
  } else {
    SETERRQ1(1,"Riemann solver %s not available for Shallow equation",rname);
  }
  if (!strcmp(rcname,"characteristic")) {
    ctx->physics.characteristic = PhysicsCharacteristic_Shallow;
  } else if (!strcmp(rcname,"conservative")) {
    ctx->physics.characteristic = PhysicsCharacteristic_Conservative;
  } else {
    SETERRQ1(1,"Reconstruction %s not available for Shallow equation",rcname);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
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
  hx = 1./Mx;
  ierr = DAGlobalToLocalBegin(ctx->da,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd  (ctx->da,X,INSERT_VALUES,Xloc);CHKERRQ(ierr);

  ierr = VecZeroEntries(F);CHKERRQ(ierr);

  ierr = DAVecGetArray(ctx->da,Xloc,&x);CHKERRQ(ierr);
  ierr = DAVecGetArray(ctx->da,F,&f);CHKERRQ(ierr);
  ierr = DAGetArray(ctx->da,PETSC_TRUE,(void**)&slope);CHKERRQ(ierr);

  ierr = DAGetCorners(ctx->da,&xs,0,0,&xm,0,0);CHKERRQ(ierr);

  for (i=xs-1; i<xs+xm+1; i++) {
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
    /* Get ratio of characteristic jumps, apply limiter */
    for (j=0; j<dof; j++) ctx->theta[j] = cjmpL[j]/cjmpR[j];
    (*ctx->limit)(dof,ctx->theta,ctx->lmt);
    for (j=0; j<dof; j++) ctx->cslope[j] = ctx->lmt[j] * cjmpR[j] / hx;
    Kernel_w_gets_A_times_v(dof,ctx->cslope,ctx->R,&slope[i*dof]);
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
#define __FUNCT__ "FVInitialSolution"
static PetscErrorCode FVInitialSolution(FVCtx *ctx,Vec X)
{
  PetscErrorCode ierr;
  PetscScalar *x;
  PetscInt i,xs,xm,Mx,dof;

  PetscFunctionBegin;
  ierr = DAGetInfo(ctx->da,0, &Mx,0,0, 0,0,0, &dof,0,0,0);CHKERRQ(ierr);
  ierr = DAGetCorners(ctx->da,&xs,0,0,&xm,0,0);CHKERRQ(ierr);
  ierr = DAVecGetArray(ctx->da,X,&x);CHKERRQ(ierr);
  for (i=xs*dof; i<(xs+xm)*dof; i++) {
    PetscReal xref = 1.0*i/dof/Mx;  /* Location of grid point on reference interval [0,1) */
    PetscScalar u0;
    switch (ctx->initial) {
      case 0:
        u0 = 0.6+0.2*sin(2*PETSC_PI*xref);
        switch (i%dof) {
          case 0: x[i] = u0; break;
          default: x[i] = u0*1;
        } break;
      case 1: x[i] = sin(2*PETSC_PI*xref); break;
      case 2: x[i] = (xref < 0.5) ? 1 : 0; break;
      case 3:
        u0 = (0.2 < xref && xref < 0.3) ? 0.9 : 0.4;
        switch (i%dof) {
          case 0: x[i] = u0; break;
          default: x[i] = u0*0.1;
        } break;
      case 4:
        switch (i%dof) {        /* symmetric rarefaction for isogas, shallow */
          case 0: x[i] = 1; break;
          case 1: x[i] = (xref<0.5) ? -2*xref : 2*(1-xref); if (xref == 0.5) x[i] = 0; break;
          default: x[i] = 0; break;
        } break;
      default:
        SETERRQ1(1,"Initial profile %d not recognized",ctx->initial);
    }
  }
  ierr = DAVecRestoreArray(ctx->da,X,&x);CHKERRQ(ierr);
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
/* Note: This is ugly (fragile and does not compose nicely), but we don't have exact solutions for an arbitrary equation
* and initial condition, so it's just hard-coded for advection right now. */
static PetscErrorCode SolutionErrorNorms(FVCtx *ctx,PetscReal t,Vec X,PetscReal *nrm1,PetscReal *nrmsup)
{
  PetscErrorCode ierr;
  Vec Y;
  PetscScalar *y;
  PetscInt i,xs,xm,Mx,dof;

  PetscFunctionBegin;
  ierr = VecDuplicate(X,&Y);CHKERRQ(ierr);
  ierr = DAGetInfo(ctx->da,0, &Mx,0,0, 0,0,0, &dof,0,0,0);CHKERRQ(ierr);
  ierr = DAGetCorners(ctx->da,&xs,0,0,&xm,0,0);CHKERRQ(ierr);
  ierr = DAVecGetArray(ctx->da,Y,&y);CHKERRQ(ierr);
  for (i=xs*dof; i<(xs+xm)*dof; i++) {
    PetscReal xref = 1.0*i/dof/Mx;  /* Location of cell center on reference interval [0,1) */
    switch (ctx->exact) {
      /* Case 0 indicates that there is no known solution */
      case 1: y[i] = sin(2*PETSC_PI*(xref - ((AdvectCtx*)ctx->physics.user)->a*t)); break;
      case 2: y[i] = fmod(1+fmod(xref - ((AdvectCtx*)ctx->physics.user)->a*t,1),1) < 0.5 ? 1 : 0; break;
      default: SETERRQ1(1,"Exact solution %d does not exist",ctx->exact);
    }
  }
  ierr = DAVecRestoreArray(ctx->da,Y,&y);CHKERRQ(ierr);
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
  char lname[256] = "koren",physname[256] = "advect";
  PetscFList limiters = 0,physics = 0;
  MPI_Comm comm;
  TS ts;
  Vec X,X0;
  FVCtx ctx;
  PetscInt i,dof,xs,xm,Mx;
  PetscInt draw = 0;
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
  ierr = PetscFListAdd(&limiters,"koren"           ,"",(void(*)(void))Limit_Koren);CHKERRQ(ierr);
  ierr = PetscFListAdd(&limiters,"cada-torrilhon"  ,"",(void(*)(void))Limit_CadaTorrilhon);CHKERRQ(ierr);

  /* Register physical models to be available on the command line */
  ierr = PetscFListAdd(&physics,"advect"          ,"",(void(*)(void))PhysicsCreate_Advect);CHKERRQ(ierr);
  ierr = PetscFListAdd(&physics,"burgers"         ,"",(void(*)(void))PhysicsCreate_Burgers);CHKERRQ(ierr);
  ierr = PetscFListAdd(&physics,"traffic"         ,"",(void(*)(void))PhysicsCreate_Traffic);CHKERRQ(ierr);
  ierr = PetscFListAdd(&physics,"isogas"          ,"",(void(*)(void))PhysicsCreate_IsoGas);CHKERRQ(ierr);
  ierr = PetscFListAdd(&physics,"shallow"         ,"",(void(*)(void))PhysicsCreate_Shallow);CHKERRQ(ierr);

  ctx.cfl = 0.9;
  ierr = PetscOptionsBegin(comm,PETSC_NULL,"Finite Volume solver options","");CHKERRQ(ierr);
  {
    ierr = PetscOptionsList("-limit","Name of flux limiter to use","",limiters,lname,lname,sizeof(lname),PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsList("-physics","Name of physics (Riemann solver and characteristics) to use","",physics,physname,physname,sizeof(physname),PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-draw","Draw solution vector at (1=initial,2=final,3=both)","",draw,&draw,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-initial","Initial condition (0=positive sine,1=sine,2=half square,3=narrow square,4=symmetric rarefaction)","",ctx.initial,&ctx.initial,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-exact","Exact solution for comparing errors, (1=sin+advect,2=half square+advect)","",ctx.exact,&ctx.exact,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-cfl","CFL number to time step at","",ctx.cfl,&ctx.cfl,PETSC_NULL);CHKERRQ(ierr);
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
  /* Informm the DA of the field names provided by the physics. */
  /* The names will be shown in the title bars when run with -ts_monitor_solution */
  for (i=0; i<ctx.physics.dof; i++) {
    ierr = DASetFieldName(ctx.da,i,ctx.physics.fieldname[i]);CHKERRQ(ierr);
  }
  /* Allow customization of the DA at runtime, mostly to change problem size with -da_grid_x M */
  ierr = DASetFromOptions(ctx.da);CHKERRQ(ierr);
  ierr = DAGetInfo(ctx.da,0, &Mx,0,0, 0,0,0, &dof,0,0,0);CHKERRQ(ierr);
  ierr = DAGetCorners(ctx.da,&xs,0,0,&xm,0,0);CHKERRQ(ierr);

  /* Allocate work space for th Finite Volume solver (so it doesn't have to be reallocated on each function evaluation) */
  ierr = PetscMalloc4(dof*dof,PetscScalar,&ctx.R,dof*dof,PetscScalar,&ctx.Rinv,2*dof,PetscScalar,&ctx.cjmpLR,1*dof,PetscScalar,&ctx.cslope);CHKERRQ(ierr);
  ierr = PetscMalloc4(dof,PetscScalar,&ctx.theta,dof,PetscScalar,&ctx.lmt,2*dof,PetscScalar,&ctx.uLR,dof,PetscScalar,&ctx.flux);CHKERRQ(ierr);

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
  ierr = FVInitialSolution(&ctx,X0);CHKERRQ(ierr);
  ierr = FVRHSFunction(ts,0,X0,X,(void*)&ctx);CHKERRQ(ierr); /* Initial function evaluation, only used to determine max speed */
  ierr = VecCopy(X0,X);CHKERRQ(ierr);                        /* The function value was not used so we set X=X0 again */
  ierr = TSSetInitialTimeStep(ts,0,ctx.cfl/ctx.cfl_idt);CHKERRQ(ierr);

  ierr = TSSetSolution(ts,X);CHKERRQ(ierr);  /* The TS will use X for the solution, starting with it's current value as initial condition */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr); /* Take runtime options */

  ierr = SolutionStatsView(ctx.da,X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  {
    PetscReal ptime,nrm1,nrmsup;
    PetscInt steps;

    ierr = TSStep(ts,&steps,&ptime);CHKERRQ(ierr);

    ierr = PetscPrintf(comm,"Final time %8.5f, steps %d\n",ptime,steps);CHKERRQ(ierr);
    if (ctx.exact) {
      ierr = SolutionErrorNorms(&ctx,ptime,X,&nrm1,&nrmsup);CHKERRQ(ierr);
      ierr = PetscPrintf(comm,"Error ||x-x_e||_1=%8.2g  ||x-x_e||_sup=%8.2g\n",nrm1,nrmsup);CHKERRQ(ierr);
    }
  }

  ierr = SolutionStatsView(ctx.da,X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  if (draw & 0x1) {ierr = VecView(X0,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}
  if (draw & 0x2) {ierr = VecView(X,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}

  /* Clean up */
  ierr = (*ctx.physics.destroy)(ctx.physics.user);CHKERRQ(ierr);
  for (i=0; i<ctx.physics.dof; i++) {ierr = PetscFree(ctx.physics.fieldname[i]);CHKERRQ(ierr);}
  ierr = PetscFree4(ctx.R,ctx.Rinv,ctx.cjmpLR,ctx.cslope);CHKERRQ(ierr);
  ierr = PetscFree4(ctx.theta,ctx.lmt,ctx.uLR,ctx.flux);CHKERRQ(ierr);
  ierr = VecDestroy(X);CHKERRQ(ierr);
  ierr = VecDestroy(X0);CHKERRQ(ierr);
  ierr = DADestroy(ctx.da);CHKERRQ(ierr);
  ierr = TSDestroy(ts);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
