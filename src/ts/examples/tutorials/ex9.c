static const char help[] = "1D periodic Finite Volume solver in flux-limiter form with semidiscrete time stepping.\n\n";

/* To get isfinite in math.h */
#define _XOPEN_SOURCE 600

#include <unistd.h>             /* usleep */
#include "petscts.h"
#include "petscda.h"

#include "../src/mat/blockinvert.h" /* For the Kernel_*_gets_* stuff for BAIJ */

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
static void Limit_CT(PetscInt m,const PetscScalar *t,PetscScalar *lmt) /* Cada-Torrilhon 2009 */
{
  PetscInt i;
  for (i=0; i<m; i++) lmt[i] = PetscMax(0,PetscMin((2+t[i])/3,
                                                   PetscMax(-0.5*t[i],
                                                            PetscMin(2*t[i],
                                                                     PetscMin((2+t[i])/3,1.6)))));
}


/* Various physics, we need create and destroy methods to make it enterprisey */
typedef struct {
  PetscReal a;                  /* advective velocity */
} AdvectCtx;
typedef struct {
  PetscReal max_speed;
} TrafficCtx;

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
#define __FUNCT__ "PhysicsRiemann_Burgers_Exact"
static PetscErrorCode PhysicsRiemann_Burgers_Exact(void *vctx,PetscInt m,const PetscScalar *uL,const PetscScalar *uR,PetscScalar *flux,PetscReal *maxspeed)
{
  PetscReal speed;

  PetscFunctionBegin;
  if (uL[0] < uR[0]) {          /* rarefaction */
    PetscReal sL = uL[0],sR = uR[0];
    if (sL > 0) flux[0] = 0.5*PetscSqr(uL[0]);      /* tail heading right */
    else if (sR < 0) flux[0] = 0.5*PetscSqr(uR[0]); /* tail heading left */
    else flux[0] = 0;                               /* transonic */
    *maxspeed = (PetscAbs(uL[0]) > PetscAbs(uR[0])) ? uL[0] : uR[0];
  } else {                      /* shock */
    speed = 0.5*(uL[0] + uR[0]);
    flux[0] = 0.5*PetscSqr(speed>0 ? uR[0] : uL[0]);
    *maxspeed = speed;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PhysicsRiemann_Burgers_Roe"
static PetscErrorCode PhysicsRiemann_Burgers_Roe(void *vctx,PetscInt m,const PetscScalar *uL,const PetscScalar *uR,PetscScalar *flux,PetscReal *maxspeed)
{
  PetscReal speed;

  PetscFunctionBegin;
  speed = 0.5*(uL[0] + uR[0]);
  flux[0] = 0.5*PetscSqr(speed>0?uL[0]:uR[0]) + speed*(uR[0]-uL[0]);
  *maxspeed = speed;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PhysicsRiemann_Traffic"
static PetscErrorCode PhysicsRiemann_Traffic(void *vctx,PetscInt m,const PetscScalar *uL,const PetscScalar *uR,PetscScalar *flux,PetscReal *maxspeed)
{
  TrafficCtx *phys = (TrafficCtx*)vctx;
  PetscReal speed;

  PetscFunctionBegin;
  speed = phys->max_speed*(1. - (uL[0] + uR[0]));
  flux[0] = PetscMax(0,speed)*uL[0] + PetscMin(0,speed)*uR[0];
  *maxspeed = speed;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PhysicsCharacteristic_Primitive"
static PetscErrorCode PhysicsCharacteristic_Primitive(void *vctx,PetscInt m,const PetscScalar *u,PetscScalar *X,PetscScalar *Xi)
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

typedef struct {
  PetscErrorCode (*riemann)(void*,PetscInt,const PetscScalar*,const PetscScalar*,PetscScalar*,PetscReal*);
  PetscErrorCode (*characteristic)(void*,PetscInt,const PetscScalar*,PetscScalar*,PetscScalar*);
  PetscErrorCode (*destroy)(void*);
  void *user;
  PetscInt dof;
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
  PetscScalar *uLR;             /* Solution at left and right of interface, primitive variables, len=2*dof */
  PetscScalar *flux;            /* Flux across interface */

  PetscReal cfl_idt;            /* Max allowable value of 1/Delta t */
  PetscReal cfl;
  PetscInt initial;
  PetscInt exact;
  PetscInt msleep;
} FVCtx;

#undef __FUNCT__  
#define __FUNCT__ "PhysicsCreate_Advect"
static PetscErrorCode PhysicsCreate_Advect(FVCtx *ctx)
{
  PetscErrorCode ierr;
  AdvectCtx *user;

  PetscFunctionBegin;
  ierr = PetscNew(*user,&user);CHKERRQ(ierr);
  ctx->physics.riemann        = PhysicsRiemann_Advect;
  ctx->physics.characteristic = PhysicsCharacteristic_Primitive;
  ctx->physics.destroy        = PhysicsDestroy_SimpleFree;
  ctx->physics.user           = user;
  ctx->physics.dof            = 1;
  user->a = 1;
  ierr = PetscOptionsBegin(ctx->comm,ctx->prefix,"Options for advection","");CHKERRQ(ierr);
  {
    ierr = PetscOptionsReal("-physics_advect_a","Speed","",user->a,&user->a,PETSC_NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PhysicsCreate_Burgers"
static PetscErrorCode PhysicsCreate_Burgers(FVCtx *ctx)
{
  PetscErrorCode ierr;
  char rname[256] = "exact";

  PetscFunctionBegin;
  ctx->physics.characteristic = PhysicsCharacteristic_Primitive;
  ctx->physics.destroy        = PhysicsDestroy_Null;
  ctx->physics.user           = 0;
  ctx->physics.dof            = 1;

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

#undef __FUNCT__  
#define __FUNCT__ "PhysicsCreate_Traffic"
static PetscErrorCode PhysicsCreate_Traffic(FVCtx *ctx)
{
  PetscErrorCode ierr;
  TrafficCtx *user;

  PetscFunctionBegin;
  ierr = PetscNew(*user,&user);CHKERRQ(ierr);
  ctx->physics.riemann        = PhysicsRiemann_Traffic;
  ctx->physics.characteristic = PhysicsCharacteristic_Primitive;
  ctx->physics.destroy        = PhysicsDestroy_SimpleFree;
  ctx->physics.user           = user;
  ctx->physics.dof            = 1;
  user->max_speed = 1;
  ierr = PetscOptionsBegin(ctx->comm,ctx->prefix,"Options for Traffic","");CHKERRQ(ierr);
  {
    ierr = PetscOptionsReal("-physics_traffic_max_speed","Maximum speed","",user->max_speed,&user->max_speed,PETSC_NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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
    // for (j=0; j<dof; j++) ctx->cslope[j] = ctx->lmt[j] * (cjmpL[j] + cjmpR[j]) / (2*hx);
    Kernel_w_gets_A_times_v(dof,ctx->R,ctx->cslope,&slope[i*dof]);
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
    ierr = (*ctx->physics.riemann)(ctx->physics.user,dof,uL,uR,ctx->flux,&maxspeed);
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

#if 0
  ierr = VecView(X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecView(F,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
#endif

  ierr = PetscGlobalMax(&cfl_idt,&ctx->cfl_idt,((PetscObject)ctx->da)->comm);CHKERRQ(ierr);
  if (1) {
    PetscReal dt,tnow;
    ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
    ierr = TSGetTime(ts,&tnow);CHKERRQ(ierr);
    if (dt > 0.5/ctx->cfl_idt) {
#if 1
      ierr = PetscPrintf(ctx->comm,"Stability constraint exceeded at t=%g, dt %g > %g\n",tnow,dt,0.5/ctx->cfl_idt);CHKERRQ(ierr);
#else
      SETERRQ2(1,"Stability constraint exceeded, %g > %g",dt,ctx->cfl/ctx->cfl_idt);
#endif
    }
  }
  ierr = usleep(ctx->msleep*1000);CHKERRQ(ierr);
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
    switch (ctx->initial) {
      case 0: x[i] = 0.5+0.2*sin(2*PETSC_PI*xref); break;
      case 1: x[i] = sin(2*PETSC_PI*xref); break;
      case 2: x[i] = (xref < 0.5) ? 1 : 0; break;
      case 3: x[i] = (0.2 < xref && xref < 0.3) ? 0.9 : 0.4; break;
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
  PetscInt dof,xs,xm,Mx;
  PetscInt draw = 0;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,0,help);CHKERRQ(ierr);
  comm = PETSC_COMM_WORLD;
  ierr = PetscMemzero(&ctx,sizeof(ctx));CHKERRQ(ierr);

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
  ierr = PetscFListAdd(&limiters,"ct"              ,"",(void(*)(void))Limit_CT);CHKERRQ(ierr);

  ierr = PetscFListAdd(&physics,"advect"          ,"",(void(*)(void))PhysicsCreate_Advect);CHKERRQ(ierr);
  ierr = PetscFListAdd(&physics,"burgers"         ,"",(void(*)(void))PhysicsCreate_Burgers);CHKERRQ(ierr);
  ierr = PetscFListAdd(&physics,"traffic"         ,"",(void(*)(void))PhysicsCreate_Traffic);CHKERRQ(ierr);

  ctx.cfl = 0.4;
  ierr = PetscOptionsBegin(comm,PETSC_NULL,"Finite Volume solver options","");CHKERRQ(ierr);
  {
    ierr = PetscOptionsList("-limit","Name of flux limiter to use","",limiters,lname,lname,sizeof(lname),PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsList("-physics","Name of physics (Riemann solver and characteristics) to use","",physics,physname,physname,sizeof(physname),PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-draw","Draw solution vector at (1=initial,2=final,3=both)","",draw,&draw,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-initial","Initial condition (0=positive sine,1=sine,2=half square,3=narrow square)","",ctx.initial,&ctx.initial,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-exact","Exact solution for comparing errors, (1=sin+advect,2=half square+advect)","",ctx.exact,&ctx.exact,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-msleep","How many milliseconds to sleep in each function evaluation","",ctx.msleep,&ctx.msleep,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-cfl","CFL number to time step at","",ctx.cfl,&ctx.cfl,PETSC_NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = PetscFListFind(limiters,comm,lname,(void(**)(void))&ctx.limit);CHKERRQ(ierr);
  if (!ctx.limit) SETERRQ1(1,"Limiter '%s' not found",lname);CHKERRQ(ierr);
  {
    PetscErrorCode (*r)(FVCtx*);
    ierr = PetscFListFind(physics,comm,physname,(void(**)(void))&r);CHKERRQ(ierr);
    if (!r) SETERRQ1(1,"Physics '%s' not found",physname);CHKERRQ(ierr);
    ierr = (*r)(&ctx);CHKERRQ(ierr);
  }

  ierr = DACreate1d(comm,DA_XPERIODIC,-5,ctx.physics.dof,2,PETSC_NULL,&ctx.da);CHKERRQ(ierr);
  ierr = DASetFromOptions(ctx.da);CHKERRQ(ierr);
  ierr = DAGetInfo(ctx.da,0, &Mx,0,0, 0,0,0, &dof,0,0,0);CHKERRQ(ierr);
  ierr = DAGetCorners(ctx.da,&xs,0,0,&xm,0,0);CHKERRQ(ierr);

  ierr = PetscMalloc4(dof*dof,PetscScalar,&ctx.R,dof*dof,PetscScalar,&ctx.Rinv,2*dof,PetscScalar,&ctx.cjmpLR,1*dof,PetscScalar,&ctx.cslope);CHKERRQ(ierr);
  ierr = PetscMalloc4(dof,PetscScalar,&ctx.theta,dof,PetscScalar,&ctx.lmt,2*dof,PetscScalar,&ctx.uLR,dof,PetscScalar,&ctx.flux);CHKERRQ(ierr);
  ierr = DAGetGlobalVector(ctx.da,&X);CHKERRQ(ierr);
  ierr = VecDuplicate(X,&X0);CHKERRQ(ierr);

  ierr = TSCreate(comm,&ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,FVRHSFunction,&ctx);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSEULER);CHKERRQ(ierr);
  ierr = TSSetDuration(ts,1000,10);CHKERRQ(ierr);

  ierr = TSSetInitialTimeStep(ts,0,0);CHKERRQ(ierr);
  ierr = FVInitialSolution(&ctx,X0);CHKERRQ(ierr);
  ierr = FVRHSFunction(ts,0,X0,X,(void*)&ctx);CHKERRQ(ierr); /* Initial function evaluation to determine max speed */
  ierr = VecCopy(X0,X);CHKERRQ(ierr);                        /* The function value is not used */
  ierr = TSSetInitialTimeStep(ts,0,ctx.cfl/ctx.cfl_idt);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,X);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

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

  ierr = (*ctx.physics.destroy)(ctx.physics.user);CHKERRQ(ierr);
  ierr = PetscFree4(ctx.R,ctx.Rinv,ctx.cjmpLR,ctx.cslope);CHKERRQ(ierr);
  ierr = PetscFree4(ctx.theta,ctx.lmt,ctx.uLR,ctx.flux);CHKERRQ(ierr);
  ierr = DADestroy(ctx.da);CHKERRQ(ierr);
  return 0;
}
