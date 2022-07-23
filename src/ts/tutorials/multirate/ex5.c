/*
  Note:
  To performance a convergence study, a reference solution can be generated with a small stepsize and stored into a binary file, e.g. -ts_type ssp -ts_dt 1e-5 -ts_max_steps 30000 -ts_view_solution binary:reference.bin
  Errors can be computed in the following runs with -simulation -f reference.bin

  Multirate RK options:
  -ts_rk_dtratio is the ratio between larger time step size and small time step size
  -ts_rk_multirate_type has three choices: none (for single step RK)
                                           combined (for multirate method and user just need to provide the same RHS with the single step RK)
                                           partitioned (for multiraet method and user need to provide two RHS, one is for fast components and the orther is for slow components
*/

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

#include "finitevolume1d.h"

static inline PetscReal RangeMod(PetscReal a,PetscReal xmin,PetscReal xmax) { PetscReal range = xmax-xmin; return xmin +PetscFmodReal(range+PetscFmodReal(a,range),range); }

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
  if (x==0 || x == xmin || x == xmax) flux[0] = PetscMax(0,(speed[0]+speed[1])/2.0)*uL[0] + PetscMin(0,(speed[0]+speed[1])/2.0)*uR[0]; /* if condition need to be changed base on different problem, '0' is the discontinuous point of a */
  else if (x<0) flux[0] = PetscMax(0,speed[0])*uL[0] + PetscMin(0,speed[0])*uR[0];  /* else if condition need to be changed based on diferent problem, 'x = 0' is discontinuous point of a */
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
  if (x<0) speeds[0] = ctx->a[0];  /* x is discontinuous point of a */
  else    speeds[0] = ctx->a[1];
  PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsSample_Advect(void *vctx,PetscInt initial,FVBCType bctype,PetscReal xmin,PetscReal xmax,PetscReal t,PetscReal x,PetscReal *u)
{
  AdvectCtx *ctx = (AdvectCtx*)vctx;
  PetscReal *a    = ctx->a,x0;

  PetscFunctionBeginUser;
  if (x<0){   /* x is cell center */
    switch (bctype) {
      case FVBC_OUTFLOW:  x0 = x-a[0]*t; break;
      case FVBC_PERIODIC: x0 = RangeMod(x-a[0]*t,xmin,xmax); break;
      default: SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"unknown BCType");
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
      default: SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"unknown initial condition");
    }
  }
  else{
    switch (bctype) {
      case FVBC_OUTFLOW:  x0 = x-a[1]*t; break;
      case FVBC_PERIODIC: x0 = RangeMod(x-a[1]*t,xmin,xmax); break;
      default: SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"unknown BCType");
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
      default: SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"unknown initial condition");
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PhysicsCreate_Advect(FVCtx *ctx)
{
  AdvectCtx      *user;

  PetscFunctionBeginUser;
  PetscCall(PetscNew(&user));
  ctx->physics.sample         = PhysicsSample_Advect;
  ctx->physics.riemann        = PhysicsRiemann_Advect;
  ctx->physics.characteristic = PhysicsCharacteristic_Advect;
  ctx->physics.destroy        = PhysicsDestroy_SimpleFree;
  ctx->physics.user           = user;
  ctx->physics.dof            = 1;
  PetscCall(PetscStrallocpy("u",&ctx->physics.fieldname[0]));
  user->a[0] = 1;
  user->a[1] = 1;
  PetscOptionsBegin(ctx->comm,ctx->prefix,"Options for advection","");
  {
    PetscCall(PetscOptionsReal("-physics_advect_a1","Speed1","",user->a[0],&user->a[0],NULL));
    PetscCall(PetscOptionsReal("-physics_advect_a2","Speed2","",user->a[1],&user->a[1],NULL));
  }
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

/* --------------------------------- Finite Volume Solver for slow parts ----------------------------------- */

PetscErrorCode FVRHSFunctionslow(TS ts,PetscReal time,Vec X,Vec F,void *vctx)
{
  FVCtx          *ctx = (FVCtx*)vctx;
  PetscInt       i,j,k,Mx,dof,xs,xm,len_slow;
  PetscReal      hx,cfl_idt = 0;
  PetscScalar    *x,*f,*slope;
  Vec            Xloc;
  DM             da;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts,&da));
  PetscCall(DMGetLocalVector(da,&Xloc));
  PetscCall(DMDAGetInfo(da,0, &Mx,0,0, 0,0,0, &dof,0,0,0,0,0));
  hx   = (ctx->xmax-ctx->xmin)/Mx;
  PetscCall(DMGlobalToLocalBegin(da,X,INSERT_VALUES,Xloc));
  PetscCall(DMGlobalToLocalEnd(da,X,INSERT_VALUES,Xloc));
  PetscCall(VecZeroEntries(F));
  PetscCall(DMDAVecGetArray(da,Xloc,&x));
  PetscCall(VecGetArray(F,&f));
  PetscCall(DMDAGetArray(da,PETSC_TRUE,&slope));
  PetscCall(DMDAGetCorners(da,&xs,0,0,&xm,0,0));
  PetscCall(ISGetSize(ctx->iss,&len_slow));

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
    if (i < len_slow+1) {
      /* Determine the right eigenvectors R, where A = R \Lambda R^{-1} */
      PetscCall((*ctx->physics.characteristic)(ctx->physics.user,dof,&x[i*dof],ctx->R,ctx->Rinv,ctx->speeds,ctx->xmin+hx*i));
      /* Evaluate jumps across interfaces (i-1, i) and (i, i+1), put in characteristic basis */
      PetscCall(PetscArrayzero(ctx->cjmpLR,2*dof));
      cjmpL = &ctx->cjmpLR[0];
      cjmpR = &ctx->cjmpLR[dof];
      for (j=0; j<dof; j++) {
        PetscScalar jmpL,jmpR;
        jmpL = x[(i+0)*dof+j]-x[(i-1)*dof+j];
        jmpR = x[(i+1)*dof+j]-x[(i+0)*dof+j];
        for (k=0; k<dof; k++) {
          cjmpL[k] += ctx->Rinv[k+j*dof]*jmpL;
          cjmpR[k] += ctx->Rinv[k+j*dof]*jmpR;
        }
      }
      /* Apply limiter to the left and right characteristic jumps */
      info.m  = dof;
      info.hx = hx;
      (*ctx->limit)(&info,cjmpL,cjmpR,ctx->cslope);
      for (j=0; j<dof; j++) ctx->cslope[j] /= hx; /* rescale to a slope */
      for (j=0; j<dof; j++) {
        PetscScalar tmp = 0;
        for (k=0; k<dof; k++) tmp += ctx->R[j+k*dof]*ctx->cslope[k];
        slope[i*dof+j] = tmp;
      }
    }
  }

  for (i=xs; i<xs+xm+1; i++) {
    PetscReal   maxspeed;
    PetscScalar *uL,*uR;
    if (i < len_slow) { /* slow parts can be changed based on a */
      uL = &ctx->uLR[0];
      uR = &ctx->uLR[dof];
      for (j=0; j<dof; j++) {
        uL[j] = x[(i-1)*dof+j]+slope[(i-1)*dof+j]*hx/2;
        uR[j] = x[(i-0)*dof+j]-slope[(i-0)*dof+j]*hx/2;
      }
      PetscCall((*ctx->physics.riemann)(ctx->physics.user,dof,uL,uR,ctx->flux,&maxspeed,ctx->xmin+hx*i,ctx->xmin,ctx->xmax));
      cfl_idt = PetscMax(cfl_idt,PetscAbsScalar(maxspeed/hx)); /* Max allowable value of 1/Delta t */
      if (i > xs) {
        for (j=0; j<dof; j++) f[(i-1)*dof+j] -= ctx->flux[j]/hx;
      }
      if (i < xs+xm) {
        for (j=0; j<dof; j++) f[i*dof+j] += ctx->flux[j]/hx;
      }
    }
    if (i == len_slow) { /* slow parts can be changed based on a */
      uL = &ctx->uLR[0];
      uR = &ctx->uLR[dof];
      for (j=0; j<dof; j++) {
        uL[j] = x[(i-1)*dof+j]+slope[(i-1)*dof+j]*hx/2;
        uR[j] = x[(i-0)*dof+j]-slope[(i-0)*dof+j]*hx/2;
      }
      PetscCall((*ctx->physics.riemann)(ctx->physics.user,dof,uL,uR,ctx->flux,&maxspeed,ctx->xmin+hx*i,ctx->xmin,ctx->xmax));
      cfl_idt = PetscMax(cfl_idt,PetscAbsScalar(maxspeed/hx)); /* Max allowable value of 1/Delta t */
      if (i > xs) {
        for (j=0; j<dof; j++) f[(i-1)*dof+j] -= ctx->flux[j]/hx;
      }
    }
  }
  PetscCall(DMDAVecRestoreArray(da,Xloc,&x));
  PetscCall(VecRestoreArray(F,&f));
  PetscCall(DMDARestoreArray(da,PETSC_TRUE,&slope));
  PetscCall(DMRestoreLocalVector(da,&Xloc));
  PetscFunctionReturn(0);
}

/* --------------------------------- Finite Volume Solver for fast  parts ----------------------------------- */

PetscErrorCode FVRHSFunctionfast(TS ts,PetscReal time,Vec X,Vec F,void *vctx)
{
  FVCtx          *ctx = (FVCtx*)vctx;
  PetscInt       i,j,k,Mx,dof,xs,xm,len_slow;
  PetscReal      hx,cfl_idt = 0;
  PetscScalar    *x,*f,*slope;
  Vec            Xloc;
  DM             da;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts,&da));
  PetscCall(DMGetLocalVector(da,&Xloc));
  PetscCall(DMDAGetInfo(da,0, &Mx,0,0, 0,0,0, &dof,0,0,0,0,0));
  hx   = (ctx->xmax-ctx->xmin)/Mx;
  PetscCall(DMGlobalToLocalBegin(da,X,INSERT_VALUES,Xloc));
  PetscCall(DMGlobalToLocalEnd(da,X,INSERT_VALUES,Xloc));
  PetscCall(VecZeroEntries(F));
  PetscCall(DMDAVecGetArray(da,Xloc,&x));
  PetscCall(VecGetArray(F,&f));
  PetscCall(DMDAGetArray(da,PETSC_TRUE,&slope));
  PetscCall(DMDAGetCorners(da,&xs,0,0,&xm,0,0));
  PetscCall(ISGetSize(ctx->iss,&len_slow));

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
    if (i > len_slow-2) {
      PetscCall((*ctx->physics.characteristic)(ctx->physics.user,dof,&x[i*dof],ctx->R,ctx->Rinv,ctx->speeds,ctx->xmin+hx*i));
      PetscCall(PetscArrayzero(ctx->cjmpLR,2*dof));
      cjmpL = &ctx->cjmpLR[0];
      cjmpR = &ctx->cjmpLR[dof];
      for (j=0; j<dof; j++) {
        PetscScalar jmpL,jmpR;
        jmpL = x[(i+0)*dof+j]-x[(i-1)*dof+j];
        jmpR = x[(i+1)*dof+j]-x[(i+0)*dof+j];
        for (k=0; k<dof; k++) {
          cjmpL[k] += ctx->Rinv[k+j*dof]*jmpL;
          cjmpR[k] += ctx->Rinv[k+j*dof]*jmpR;
        }
      }
      /* Apply limiter to the left and right characteristic jumps */
      info.m  = dof;
      info.hx = hx;
      (*ctx->limit)(&info,cjmpL,cjmpR,ctx->cslope);
      for (j=0; j<dof; j++) ctx->cslope[j] /= hx; /* rescale to a slope */
      for (j=0; j<dof; j++) {
        PetscScalar tmp = 0;
        for (k=0; k<dof; k++) tmp += ctx->R[j+k*dof]*ctx->cslope[k];
        slope[i*dof+j] = tmp;
      }
    }
  }

  for (i=xs; i<xs+xm+1; i++) {
    PetscReal   maxspeed;
    PetscScalar *uL,*uR;
    if (i > len_slow) {
      uL = &ctx->uLR[0];
      uR = &ctx->uLR[dof];
      for (j=0; j<dof; j++) {
        uL[j] = x[(i-1)*dof+j]+slope[(i-1)*dof+j]*hx/2;
        uR[j] = x[(i-0)*dof+j]-slope[(i-0)*dof+j]*hx/2;
      }
      PetscCall((*ctx->physics.riemann)(ctx->physics.user,dof,uL,uR,ctx->flux,&maxspeed,ctx->xmin+hx*i,ctx->xmin,ctx->xmax));
      cfl_idt = PetscMax(cfl_idt,PetscAbsScalar(maxspeed/hx)); /* Max allowable value of 1/Delta t */
      if (i > xs) {
        for (j=0; j<dof; j++) f[(i-len_slow-1)*dof+j] -= ctx->flux[j]/hx;
      }
      if (i < xs+xm) {
        for (j=0; j<dof; j++) f[(i-len_slow)*dof+j] += ctx->flux[j]/hx;
      }
    }
    if (i == len_slow) {
      uL = &ctx->uLR[0];
      uR = &ctx->uLR[dof];
      for (j=0; j<dof; j++) {
        uL[j] = x[(i-1)*dof+j]+slope[(i-1)*dof+j]*hx/2;
        uR[j] = x[(i-0)*dof+j]-slope[(i-0)*dof+j]*hx/2;
      }
      PetscCall((*ctx->physics.riemann)(ctx->physics.user,dof,uL,uR,ctx->flux,&maxspeed,ctx->xmin+hx*i,ctx->xmin,ctx->xmax));
      cfl_idt = PetscMax(cfl_idt,PetscAbsScalar(maxspeed/hx)); /* Max allowable value of 1/Delta t */
      if (i < xs+xm) {
        for (j=0; j<dof; j++) f[(i-len_slow)*dof+j] += ctx->flux[j]/hx;
      }
    }
  }
  PetscCall(DMDAVecRestoreArray(da,Xloc,&x));
  PetscCall(VecRestoreArray(F,&f));
  PetscCall(DMDARestoreArray(da,PETSC_TRUE,&slope));
  PetscCall(DMRestoreLocalVector(da,&Xloc));
  PetscFunctionReturn(0);
}

PetscErrorCode FVRHSFunctionslow2(TS ts,PetscReal time,Vec X,Vec F,void *vctx)
{
  FVCtx          *ctx = (FVCtx*)vctx;
  PetscInt       i,j,k,Mx,dof,xs,xm,len_slow1,len_slow2;
  PetscReal      hx,cfl_idt = 0;
  PetscScalar    *x,*f,*slope;
  Vec            Xloc;
  DM             da;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts,&da));
  PetscCall(DMGetLocalVector(da,&Xloc));
  PetscCall(DMDAGetInfo(da,0, &Mx,0,0, 0,0,0, &dof,0,0,0,0,0));
  hx   = (ctx->xmax-ctx->xmin)/Mx;
  PetscCall(DMGlobalToLocalBegin(da,X,INSERT_VALUES,Xloc));
  PetscCall(DMGlobalToLocalEnd(da,X,INSERT_VALUES,Xloc));
  PetscCall(VecZeroEntries(F));
  PetscCall(DMDAVecGetArray(da,Xloc,&x));
  PetscCall(VecGetArray(F,&f));
  PetscCall(DMDAGetArray(da,PETSC_TRUE,&slope));
  PetscCall(DMDAGetCorners(da,&xs,0,0,&xm,0,0));
  PetscCall(ISGetSize(ctx->iss,&len_slow1));
  PetscCall(ISGetSize(ctx->iss2,&len_slow2));
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
    if (i < len_slow1+len_slow2+1 && i > len_slow1-2) {
      PetscCall((*ctx->physics.characteristic)(ctx->physics.user,dof,&x[i*dof],ctx->R,ctx->Rinv,ctx->speeds,ctx->xmin+hx*i));
      PetscCall(PetscArrayzero(ctx->cjmpLR,2*dof));
      cjmpL = &ctx->cjmpLR[0];
      cjmpR = &ctx->cjmpLR[dof];
      for (j=0; j<dof; j++) {
        PetscScalar jmpL,jmpR;
        jmpL = x[(i+0)*dof+j]-x[(i-1)*dof+j];
        jmpR = x[(i+1)*dof+j]-x[(i+0)*dof+j];
        for (k=0; k<dof; k++) {
          cjmpL[k] += ctx->Rinv[k+j*dof]*jmpL;
          cjmpR[k] += ctx->Rinv[k+j*dof]*jmpR;
        }
      }
      /* Apply limiter to the left and right characteristic jumps */
      info.m  = dof;
      info.hx = hx;
      (*ctx->limit)(&info,cjmpL,cjmpR,ctx->cslope);
      for (j=0; j<dof; j++) ctx->cslope[j] /= hx; /* rescale to a slope */
      for (j=0; j<dof; j++) {
        PetscScalar tmp = 0;
        for (k=0; k<dof; k++) tmp += ctx->R[j+k*dof]*ctx->cslope[k];
        slope[i*dof+j] = tmp;
      }
    }
  }

  for (i=xs; i<xs+xm+1; i++) {
    PetscReal   maxspeed;
    PetscScalar *uL,*uR;
    if (i < len_slow1+len_slow2 && i > len_slow1) { /* slow parts can be changed based on a */
      uL = &ctx->uLR[0];
      uR = &ctx->uLR[dof];
      for (j=0; j<dof; j++) {
        uL[j] = x[(i-1)*dof+j]+slope[(i-1)*dof+j]*hx/2;
        uR[j] = x[(i-0)*dof+j]-slope[(i-0)*dof+j]*hx/2;
      }
      PetscCall((*ctx->physics.riemann)(ctx->physics.user,dof,uL,uR,ctx->flux,&maxspeed,ctx->xmin+hx*i,ctx->xmin,ctx->xmax));
      cfl_idt = PetscMax(cfl_idt,PetscAbsScalar(maxspeed/hx)); /* Max allowable value of 1/Delta t */
      if (i > xs) {
        for (j=0; j<dof; j++) f[(i-len_slow1-1)*dof+j] -= ctx->flux[j]/hx;
      }
      if (i < xs+xm) {
        for (j=0; j<dof; j++) f[(i-len_slow1)*dof+j] += ctx->flux[j]/hx;
      }
    }
    if (i == len_slow1+len_slow2) { /* slow parts can be changed based on a */
      uL = &ctx->uLR[0];
      uR = &ctx->uLR[dof];
      for (j=0; j<dof; j++) {
        uL[j] = x[(i-1)*dof+j]+slope[(i-1)*dof+j]*hx/2;
        uR[j] = x[(i-0)*dof+j]-slope[(i-0)*dof+j]*hx/2;
      }
      PetscCall((*ctx->physics.riemann)(ctx->physics.user,dof,uL,uR,ctx->flux,&maxspeed,ctx->xmin+hx*i,ctx->xmin,ctx->xmax));
      cfl_idt = PetscMax(cfl_idt,PetscAbsScalar(maxspeed/hx)); /* Max allowable value of 1/Delta t */
      if (i > xs) {
        for (j=0; j<dof; j++) f[(i-len_slow1-1)*dof+j] -= ctx->flux[j]/hx;
      }
    }
    if (i == len_slow1) { /* slow parts can be changed based on a */
      uL = &ctx->uLR[0];
      uR = &ctx->uLR[dof];
      for (j=0; j<dof; j++) {
        uL[j] = x[(i-1)*dof+j]+slope[(i-1)*dof+j]*hx/2;
        uR[j] = x[(i-0)*dof+j]-slope[(i-0)*dof+j]*hx/2;
      }
      PetscCall((*ctx->physics.riemann)(ctx->physics.user,dof,uL,uR,ctx->flux,&maxspeed,ctx->xmin+hx*i,ctx->xmin,ctx->xmax));
      cfl_idt = PetscMax(cfl_idt,PetscAbsScalar(maxspeed/hx)); /* Max allowable value of 1/Delta t */
      if (i < xs+xm) {
        for (j=0; j<dof; j++) f[(i-len_slow1)*dof+j] += ctx->flux[j]/hx;
      }
    }
  }

  PetscCall(DMDAVecRestoreArray(da,Xloc,&x));
  PetscCall(VecRestoreArray(F,&f));
  PetscCall(DMDARestoreArray(da,PETSC_TRUE,&slope));
  PetscCall(DMRestoreLocalVector(da,&Xloc));
  PetscFunctionReturn(0);
}

PetscErrorCode FVRHSFunctionfast2(TS ts,PetscReal time,Vec X,Vec F,void *vctx)
{
  FVCtx          *ctx = (FVCtx*)vctx;
  PetscInt       i,j,k,Mx,dof,xs,xm,len_slow1,len_slow2;
  PetscReal      hx,cfl_idt = 0;
  PetscScalar    *x,*f,*slope;
  Vec            Xloc;
  DM             da;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts,&da));
  PetscCall(DMGetLocalVector(da,&Xloc));
  PetscCall(DMDAGetInfo(da,0, &Mx,0,0, 0,0,0, &dof,0,0,0,0,0));
  hx   = (ctx->xmax-ctx->xmin)/Mx;
  PetscCall(DMGlobalToLocalBegin(da,X,INSERT_VALUES,Xloc));
  PetscCall(DMGlobalToLocalEnd(da,X,INSERT_VALUES,Xloc));
  PetscCall(VecZeroEntries(F));
  PetscCall(DMDAVecGetArray(da,Xloc,&x));
  PetscCall(VecGetArray(F,&f));
  PetscCall(DMDAGetArray(da,PETSC_TRUE,&slope));
  PetscCall(DMDAGetCorners(da,&xs,0,0,&xm,0,0));
  PetscCall(ISGetSize(ctx->iss,&len_slow1));
  PetscCall(ISGetSize(ctx->iss2,&len_slow2));

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
    if (i > len_slow1+len_slow2-2) {
      PetscCall((*ctx->physics.characteristic)(ctx->physics.user,dof,&x[i*dof],ctx->R,ctx->Rinv,ctx->speeds,ctx->xmin+hx*i));
      PetscCall(PetscArrayzero(ctx->cjmpLR,2*dof));
      cjmpL = &ctx->cjmpLR[0];
      cjmpR = &ctx->cjmpLR[dof];
      for (j=0; j<dof; j++) {
        PetscScalar jmpL,jmpR;
        jmpL = x[(i+0)*dof+j]-x[(i-1)*dof+j];
        jmpR = x[(i+1)*dof+j]-x[(i+0)*dof+j];
        for (k=0; k<dof; k++) {
          cjmpL[k] += ctx->Rinv[k+j*dof]*jmpL;
          cjmpR[k] += ctx->Rinv[k+j*dof]*jmpR;
        }
      }
      /* Apply limiter to the left and right characteristic jumps */
      info.m  = dof;
      info.hx = hx;
      (*ctx->limit)(&info,cjmpL,cjmpR,ctx->cslope);
      for (j=0; j<dof; j++) ctx->cslope[j] /= hx; /* rescale to a slope */
      for (j=0; j<dof; j++) {
        PetscScalar tmp = 0;
        for (k=0; k<dof; k++) tmp += ctx->R[j+k*dof]*ctx->cslope[k];
        slope[i*dof+j] = tmp;
      }
    }
  }

  for (i=xs; i<xs+xm+1; i++) {
    PetscReal   maxspeed;
    PetscScalar *uL,*uR;
    if (i > len_slow1+len_slow2) {
      uL = &ctx->uLR[0];
      uR = &ctx->uLR[dof];
      for (j=0; j<dof; j++) {
        uL[j] = x[(i-1)*dof+j]+slope[(i-1)*dof+j]*hx/2;
        uR[j] = x[(i-0)*dof+j]-slope[(i-0)*dof+j]*hx/2;
      }
      PetscCall((*ctx->physics.riemann)(ctx->physics.user,dof,uL,uR,ctx->flux,&maxspeed,ctx->xmin+hx*i,ctx->xmin,ctx->xmax));
      cfl_idt = PetscMax(cfl_idt,PetscAbsScalar(maxspeed/hx)); /* Max allowable value of 1/Delta t */
      if (i > xs) {
        for (j=0; j<dof; j++) f[(i-len_slow1-len_slow2-1)*dof+j] -= ctx->flux[j]/hx;
      }
      if (i < xs+xm) {
        for (j=0; j<dof; j++) f[(i-len_slow1-len_slow2)*dof+j] += ctx->flux[j]/hx;
      }
    }
    if (i == len_slow1+len_slow2) {
      uL = &ctx->uLR[0];
      uR = &ctx->uLR[dof];
      for (j=0; j<dof; j++) {
        uL[j] = x[(i-1)*dof+j]+slope[(i-1)*dof+j]*hx/2;
        uR[j] = x[(i-0)*dof+j]-slope[(i-0)*dof+j]*hx/2;
      }
      PetscCall((*ctx->physics.riemann)(ctx->physics.user,dof,uL,uR,ctx->flux,&maxspeed,ctx->xmin+hx*i,ctx->xmin,ctx->xmax));
      cfl_idt = PetscMax(cfl_idt,PetscAbsScalar(maxspeed/hx)); /* Max allowable value of 1/Delta t */
      if (i < xs+xm) {
        for (j=0; j<dof; j++) f[(i-len_slow1-len_slow2)*dof+j] += ctx->flux[j]/hx;
      }
    }
  }
  PetscCall(DMDAVecRestoreArray(da,Xloc,&x));
  PetscCall(VecRestoreArray(F,&f));
  PetscCall(DMDARestoreArray(da,PETSC_TRUE,&slope));
  PetscCall(DMRestoreLocalVector(da,&Xloc));
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
  PetscInt          i,k,dof,xs,xm,Mx,draw = 0,*index_slow,*index_fast,islow = 0,ifast = 0;
  PetscBool         view_final = PETSC_FALSE;
  PetscReal         ptime;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,0,help));
  comm = PETSC_COMM_WORLD;
  PetscCall(PetscMemzero(&ctx,sizeof(ctx)));

  /* Register limiters to be available on the command line */
  PetscCall(PetscFunctionListAdd(&limiters,"upwind"              ,Limit_Upwind));
  PetscCall(PetscFunctionListAdd(&limiters,"lax-wendroff"        ,Limit_LaxWendroff));
  PetscCall(PetscFunctionListAdd(&limiters,"beam-warming"        ,Limit_BeamWarming));
  PetscCall(PetscFunctionListAdd(&limiters,"fromm"               ,Limit_Fromm));
  PetscCall(PetscFunctionListAdd(&limiters,"minmod"              ,Limit_Minmod));
  PetscCall(PetscFunctionListAdd(&limiters,"superbee"            ,Limit_Superbee));
  PetscCall(PetscFunctionListAdd(&limiters,"mc"                  ,Limit_MC));
  PetscCall(PetscFunctionListAdd(&limiters,"vanleer"             ,Limit_VanLeer));
  PetscCall(PetscFunctionListAdd(&limiters,"vanalbada"           ,Limit_VanAlbada));
  PetscCall(PetscFunctionListAdd(&limiters,"vanalbadatvd"        ,Limit_VanAlbadaTVD));
  PetscCall(PetscFunctionListAdd(&limiters,"koren"               ,Limit_Koren));
  PetscCall(PetscFunctionListAdd(&limiters,"korensym"            ,Limit_KorenSym));
  PetscCall(PetscFunctionListAdd(&limiters,"koren3"              ,Limit_Koren3));
  PetscCall(PetscFunctionListAdd(&limiters,"cada-torrilhon2"     ,Limit_CadaTorrilhon2));
  PetscCall(PetscFunctionListAdd(&limiters,"cada-torrilhon3-r0p1",Limit_CadaTorrilhon3R0p1));
  PetscCall(PetscFunctionListAdd(&limiters,"cada-torrilhon3-r1"  ,Limit_CadaTorrilhon3R1));
  PetscCall(PetscFunctionListAdd(&limiters,"cada-torrilhon3-r10" ,Limit_CadaTorrilhon3R10));
  PetscCall(PetscFunctionListAdd(&limiters,"cada-torrilhon3-r100",Limit_CadaTorrilhon3R100));

  /* Register physical models to be available on the command line */
  PetscCall(PetscFunctionListAdd(&physics,"advect"          ,PhysicsCreate_Advect));

  ctx.comm = comm;
  ctx.cfl  = 0.9;
  ctx.bctype = FVBC_PERIODIC;
  ctx.xmin = -1.0;
  ctx.xmax = 1.0;
  PetscOptionsBegin(comm,NULL,"Finite Volume solver options","");
  PetscCall(PetscOptionsReal("-xmin","X min","",ctx.xmin,&ctx.xmin,NULL));
  PetscCall(PetscOptionsReal("-xmax","X max","",ctx.xmax,&ctx.xmax,NULL));
  PetscCall(PetscOptionsFList("-limit","Name of flux limiter to use","",limiters,lname,lname,sizeof(lname),NULL));
  PetscCall(PetscOptionsInt("-draw","Draw solution vector, bitwise OR of (1=initial,2=final,4=final error)","",draw,&draw,NULL));
  PetscCall(PetscOptionsString("-view_final","Write final solution in ASCII MATLAB format to given file name","",final_fname,final_fname,sizeof(final_fname),&view_final));
  PetscCall(PetscOptionsInt("-initial","Initial condition (depends on the physics)","",ctx.initial,&ctx.initial,NULL));
  PetscCall(PetscOptionsBool("-simulation","Compare errors with reference solution","",ctx.simulation,&ctx.simulation,NULL));
  PetscCall(PetscOptionsReal("-cfl","CFL number to time step at","",ctx.cfl,&ctx.cfl,NULL));
  PetscCall(PetscOptionsEnum("-bc_type","Boundary condition","",FVBCTypes,(PetscEnum)ctx.bctype,(PetscEnum*)&ctx.bctype,NULL));
  PetscCall(PetscOptionsBool("-recursive_split","Split the domain recursively","",ctx.recursive,&ctx.recursive,NULL));
  PetscOptionsEnd();

  /* Choose the limiter from the list of registered limiters */
  PetscCall(PetscFunctionListFind(limiters,lname,&ctx.limit));
  PetscCheck(ctx.limit,PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Limiter '%s' not found",lname);

  /* Choose the physics from the list of registered models */
  {
    PetscErrorCode (*r)(FVCtx*);
    PetscCall(PetscFunctionListFind(physics,physname,&r));
    PetscCheck(r,PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Physics '%s' not found",physname);
    /* Create the physics, will set the number of fields and their names */
    PetscCall((*r)(&ctx));
  }

  /* Create a DMDA to manage the parallel grid */
  PetscCall(DMDACreate1d(comm,DM_BOUNDARY_PERIODIC,50,ctx.physics.dof,2,NULL,&da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  /* Inform the DMDA of the field names provided by the physics. */
  /* The names will be shown in the title bars when run with -ts_monitor_draw_solution */
  for (i=0; i<ctx.physics.dof; i++) {
    PetscCall(DMDASetFieldName(da,i,ctx.physics.fieldname[i]));
  }
  PetscCall(DMDAGetInfo(da,0, &Mx,0,0, 0,0,0, &dof,0,0,0,0,0));
  PetscCall(DMDAGetCorners(da,&xs,0,0,&xm,0,0));

  /* Set coordinates of cell centers */
  PetscCall(DMDASetUniformCoordinates(da,ctx.xmin+0.5*(ctx.xmax-ctx.xmin)/Mx,ctx.xmax+0.5*(ctx.xmax-ctx.xmin)/Mx,0,0,0,0));

  /* Allocate work space for the Finite Volume solver (so it doesn't have to be reallocated on each function evaluation) */
  PetscCall(PetscMalloc4(dof*dof,&ctx.R,dof*dof,&ctx.Rinv,2*dof,&ctx.cjmpLR,1*dof,&ctx.cslope));
  PetscCall(PetscMalloc3(2*dof,&ctx.uLR,dof,&ctx.flux,dof,&ctx.speeds));

  /* Create a vector to store the solution and to save the initial state */
  PetscCall(DMCreateGlobalVector(da,&X));
  PetscCall(VecDuplicate(X,&X0));
  PetscCall(VecDuplicate(X,&R));

  /*-------------------------------- create index for slow parts and fast parts ----------------------------------------*/
  PetscCall(PetscMalloc1(xm*dof,&index_slow));
  PetscCall(PetscMalloc1(xm*dof,&index_fast));
  for (i=xs; i<xs+xm; i++) {
    if (ctx.xmin+i*(ctx.xmax-ctx.xmin)/(PetscReal)Mx+0.5*(ctx.xmax-ctx.xmin)/(PetscReal)Mx < 0)
      for (k=0; k<dof; k++) index_slow[islow++] = i*dof+k;
    else
      for (k=0; k<dof; k++) index_fast[ifast++] = i*dof+k;
  }  /* this step need to be changed based on discontinuous point of a */
  PetscCall(ISCreateGeneral(PETSC_COMM_WORLD,islow,index_slow,PETSC_COPY_VALUES,&ctx.iss));
  PetscCall(ISCreateGeneral(PETSC_COMM_WORLD,ifast,index_fast,PETSC_COPY_VALUES,&ctx.isf));

  /* Create a time-stepping object */
  PetscCall(TSCreate(comm,&ts));
  PetscCall(TSSetDM(ts,da));
  PetscCall(TSSetRHSFunction(ts,R,FVRHSFunction,&ctx));
  PetscCall(TSRHSSplitSetIS(ts,"slow",ctx.iss));
  PetscCall(TSRHSSplitSetIS(ts,"fast",ctx.isf));
  PetscCall(TSRHSSplitSetRHSFunction(ts,"slow",NULL,FVRHSFunctionslow,&ctx));
  PetscCall(TSRHSSplitSetRHSFunction(ts,"fast",NULL,FVRHSFunctionfast,&ctx));

  if (ctx.recursive) {
    TS subts;
    islow = 0;
    ifast = 0;
    for (i=xs; i<xs+xm; i++) {
      PetscReal coord = ctx.xmin+i*(ctx.xmax-ctx.xmin)/(PetscReal)Mx+0.5*(ctx.xmax-ctx.xmin)/(PetscReal)Mx;
      if (coord >= 0 && coord < ctx.xmin+(ctx.xmax-ctx.xmin)*3/4.)
        for (k=0; k<dof; k++) index_slow[islow++] = i*dof+k;
      if (coord >= ctx.xmin+(ctx.xmax-ctx.xmin)*3/4.)
        for (k=0; k<dof; k++) index_fast[ifast++] = i*dof+k;
    }  /* this step need to be changed based on discontinuous point of a */
    PetscCall(ISCreateGeneral(PETSC_COMM_WORLD,islow,index_slow,PETSC_COPY_VALUES,&ctx.iss2));
    PetscCall(ISCreateGeneral(PETSC_COMM_WORLD,ifast,index_fast,PETSC_COPY_VALUES,&ctx.isf2));

    PetscCall(TSRHSSplitGetSubTS(ts,"fast",&subts));
    PetscCall(TSRHSSplitSetIS(subts,"slow",ctx.iss2));
    PetscCall(TSRHSSplitSetIS(subts,"fast",ctx.isf2));
    PetscCall(TSRHSSplitSetRHSFunction(subts,"slow",NULL,FVRHSFunctionslow2,&ctx));
    PetscCall(TSRHSSplitSetRHSFunction(subts,"fast",NULL,FVRHSFunctionfast2,&ctx));
  }

  /*PetscCall(TSSetType(ts,TSSSP));*/
  PetscCall(TSSetType(ts,TSMPRK));
  PetscCall(TSSetMaxTime(ts,10));
  PetscCall(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER));

  /* Compute initial conditions and starting time step */
  PetscCall(FVSample(&ctx,da,0,X0));
  PetscCall(FVRHSFunction(ts,0,X0,X,(void*)&ctx)); /* Initial function evaluation, only used to determine max speed */
  PetscCall(VecCopy(X0,X));                        /* The function value was not used so we set X=X0 again */
  PetscCall(TSSetTimeStep(ts,ctx.cfl/ctx.cfl_idt));
  PetscCall(TSSetFromOptions(ts)); /* Take runtime options */
  PetscCall(SolutionStatsView(da,X,PETSC_VIEWER_STDOUT_WORLD));
  {
    PetscInt    steps;
    PetscScalar mass_initial,mass_final,mass_difference;

    PetscCall(TSSolve(ts,X));
    PetscCall(TSGetSolveTime(ts,&ptime));
    PetscCall(TSGetStepNumber(ts,&steps));
    PetscCall(PetscPrintf(comm,"Final time %g, steps %" PetscInt_FMT "\n",(double)ptime,steps));
    /* calculate the total mass at initial time and final time */
    mass_initial = 0.0;
    mass_final   = 0.0;
    PetscCall(VecSum(X0,&mass_initial));
    PetscCall(VecSum(X,&mass_final));
    mass_difference = (ctx.xmax-ctx.xmin)/(PetscScalar)Mx*(mass_final - mass_initial);
    PetscCall(PetscPrintf(comm,"Mass difference %g\n",(double)mass_difference));
    if (ctx.simulation) {
      PetscViewer  fd;
      char         filename[PETSC_MAX_PATH_LEN] = "binaryoutput";
      Vec          XR;
      PetscReal    nrm1,nrmsup;
      PetscBool    flg;

      PetscCall(PetscOptionsGetString(NULL,NULL,"-f",filename,sizeof(filename),&flg));
      PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER,"Must indicate binary file with the -f option");
      PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&fd));
      PetscCall(VecDuplicate(X0,&XR));
      PetscCall(VecLoad(XR,fd));
      PetscCall(PetscViewerDestroy(&fd));
      PetscCall(VecAYPX(XR,-1,X));
      PetscCall(VecNorm(XR,NORM_1,&nrm1));
      PetscCall(VecNorm(XR,NORM_INFINITY,&nrmsup));
      nrm1 /= Mx;
      PetscCall(PetscPrintf(comm,"Error ||x-x_e||_1 %g  ||x-x_e||_sup %g\n",(double)nrm1,(double)nrmsup));
      PetscCall(VecDestroy(&XR));
    }
  }

  PetscCall(SolutionStatsView(da,X,PETSC_VIEWER_STDOUT_WORLD));
  if (draw & 0x1) PetscCall(VecView(X0,PETSC_VIEWER_DRAW_WORLD));
  if (draw & 0x2) PetscCall(VecView(X,PETSC_VIEWER_DRAW_WORLD));
  if (draw & 0x4) {
    Vec Y;
    PetscCall(VecDuplicate(X,&Y));
    PetscCall(FVSample(&ctx,da,ptime,Y));
    PetscCall(VecAYPX(Y,-1,X));
    PetscCall(VecView(Y,PETSC_VIEWER_DRAW_WORLD));
    PetscCall(VecDestroy(&Y));
  }

  if (view_final) {
    PetscViewer viewer;
    PetscCall(PetscViewerASCIIOpen(PETSC_COMM_WORLD,final_fname,&viewer));
    PetscCall(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB));
    PetscCall(VecView(X,viewer));
    PetscCall(PetscViewerPopFormat(viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }

  /* Clean up */
  PetscCall((*ctx.physics.destroy)(ctx.physics.user));
  for (i=0; i<ctx.physics.dof; i++) PetscCall(PetscFree(ctx.physics.fieldname[i]));
  PetscCall(PetscFree4(ctx.R,ctx.Rinv,ctx.cjmpLR,ctx.cslope));
  PetscCall(PetscFree3(ctx.uLR,ctx.flux,ctx.speeds));
  PetscCall(ISDestroy(&ctx.iss));
  PetscCall(ISDestroy(&ctx.isf));
  PetscCall(ISDestroy(&ctx.iss2));
  PetscCall(ISDestroy(&ctx.isf2));
  PetscCall(VecDestroy(&X));
  PetscCall(VecDestroy(&X0));
  PetscCall(VecDestroy(&R));
  PetscCall(DMDestroy(&da));
  PetscCall(TSDestroy(&ts));
  PetscCall(PetscFree(index_slow));
  PetscCall(PetscFree(index_fast));
  PetscCall(PetscFunctionListDestroy(&limiters));
  PetscCall(PetscFunctionListDestroy(&physics));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
    build:
      requires: !complex
      depends: finitevolume1d.c

    test:
      args: -da_grid_x 60 -initial 1 -xmin -1 -xmax 1 -limit mc -physics_advect_a1 1 -physics_advect_a2 2 -ts_dt 0.025 -ts_max_steps 24 -ts_type rk -ts_rk_type 2a -ts_rk_dtratio 2 -ts_rk_multirate -ts_use_splitrhsfunction 0

    test:
      suffix: 2
      args: -da_grid_x 60 -initial 1 -xmin -1 -xmax 1 -limit mc -physics_advect_a1 1 -physics_advect_a2 2 -ts_dt 0.025 -ts_max_steps 24 -ts_type rk -ts_rk_type 2a -ts_rk_dtratio 2 -ts_rk_multirate -ts_use_splitrhsfunction 1
      output_file: output/ex5_1.out

    test:
      suffix: 3
      args: -da_grid_x 60 -initial 1 -xmin -1 -xmax 1 -limit mc -physics_advect_a1 1 -physics_advect_a2 2 -ts_dt 0.025 -ts_max_steps 24 -ts_type mprk -ts_mprk_type 2a22 -ts_use_splitrhsfunction 0

    test:
      suffix: 4
      args: -da_grid_x 60 -initial 1 -xmin -1 -xmax 1 -limit mc -physics_advect_a1 1 -physics_advect_a2 2 -ts_dt 0.025 -ts_max_steps 24 -ts_type mprk -ts_mprk_type 2a22 -ts_use_splitrhsfunction 1
      output_file: output/ex5_3.out

    test:
      suffix: 5
      args: -da_grid_x 60 -initial 1 -xmin -1 -xmax 1 -limit mc -physics_advect_a1 1 -physics_advect_a2 2 -ts_dt 0.025 -ts_max_steps 24 -ts_type rk -ts_rk_type 2a -ts_rk_dtratio 2 -ts_rk_multirate -ts_use_splitrhsfunction 0 -recursive_split

    test:
      suffix: 6
      args: -da_grid_x 60 -initial 1 -xmin -1 -xmax 1 -limit mc -physics_advect_a1 1 -physics_advect_a2 2 -ts_dt 0.025 -ts_max_steps 24 -ts_type rk -ts_rk_type 2a -ts_rk_dtratio 2 -ts_rk_multirate -ts_use_splitrhsfunction 1 -recursive_split
TEST*/
