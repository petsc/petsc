static char help[] = "Time-Dependent Reactive Flow example in 2D with Darcy Flow";

/*T
   Concepts: Solving a multicomponent time-dependent reactive flow system
   Concepts: DMDA with timestepping
   Processors: n
T*/

/*

This example solves the elementary chemical reaction:

SP_A + 2SP_B = SP_C

Subject to predetermined flow modeled as though it were in a porous media.
This flow is modeled as advection and diffusion of the three species as

v = porosity*saturation*grad(q)

and the time-dependent equation solved for each species as
advection + diffusion + reaction:

v dot grad SP_i + dSP_i / dt + dispersivity*div*grad(SP_i) + R(SP_i) = 0

The following options are available:

-length_x - The length of the domain in the direction of the flow
-length_y - The length of the domain in the direction orthogonal to the flow

-gradq_inflow - The inflow speed as if the saturation and porosity were 1.
-saturation - The saturation of the porous medium.
-porosity - The porosity of the medium.
-dispersivity - The dispersivity of the flow.
-rate_constant - The rate constant for the chemical reaction
-stoich - The stoichiometry matrix for the reaction
-sp_inflow - The species concentrations at the inflow
-sp_0 - The species concentrations at time 0

 */

#include <petscdm.h>
#include <petscdmda.h>
#include <petscsnes.h>
#include <petscts.h>

#define N_SPECIES 3
#define N_REACTIONS 1
#define DIM 2

#define stoich(i, j)  ctx->stoichiometry[N_SPECIES*i + j]

typedef struct {
  PetscScalar sp[N_SPECIES];
} Field;

typedef struct {

  Field     x_inflow;
  Field     x_0;
  PetscReal stoichiometry[N_SPECIES*N_REACTIONS];
  PetscReal porosity;
  PetscReal dispersivity;
  PetscReal saturation;
  PetscReal rate_constant[N_REACTIONS];
  PetscReal gradq_inflow;
  PetscReal length[DIM];
} AppCtx;

extern PetscErrorCode FormInitialGuess(DM da,AppCtx *ctx,Vec X);
extern PetscErrorCode FormIFunctionLocal(DMDALocalInfo*,PetscReal,Field**,Field**,Field**,AppCtx*);
extern PetscErrorCode FormIFunction(TS,PetscReal,Vec,Vec,Vec,void*);
extern PetscErrorCode ReactingFlowPostCheck(SNESLineSearch,Vec,Vec,Vec,PetscBool*,PetscBool*,void*);

PetscErrorCode SetFromOptions(AppCtx * ctx)
{
  PetscInt       i,j;

  PetscFunctionBeginUser;
  ctx->dispersivity     = 0.5;
  ctx->porosity         = 0.25;
  ctx->saturation       = 1.0;
  ctx->gradq_inflow     = 1.0;
  ctx->rate_constant[0] = 0.5;

  ctx->length[0] = 100.0;
  ctx->length[1] = 100.0;

  ctx->x_0.sp[0] = 0.0;
  ctx->x_0.sp[1] = 0.0;
  ctx->x_0.sp[2] = 0.0;

  ctx->x_inflow.sp[0] = 0.05;
  ctx->x_inflow.sp[1] = 0.05;
  ctx->x_inflow.sp[2] = 0.0;

  for (i = 0; i < N_REACTIONS; i++) {
    for (j = 0; j < N_SPECIES; j++) stoich(i, j) = 0.;
  }

  /* set up a pretty easy example */
  stoich(0, 0) = -1.;
  stoich(0, 1) = -2.;
  stoich(0, 2) = 1.;

  PetscInt as = N_SPECIES;
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-length_x",&ctx->length[0],NULL));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-length_y",&ctx->length[1],NULL));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-porosity",&ctx->porosity,NULL));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-saturation",&ctx->saturation,NULL));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-dispersivity",&ctx->dispersivity,NULL));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-gradq_inflow",&ctx->gradq_inflow,NULL));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-rate_constant",&ctx->rate_constant[0],NULL));
  PetscCall(PetscOptionsGetRealArray(NULL,NULL,"-sp_inflow",ctx->x_inflow.sp,&as,NULL));
  PetscCall(PetscOptionsGetRealArray(NULL,NULL,"-sp_0",ctx->x_0.sp,&as,NULL));
  as   = N_SPECIES;
  PetscCall(PetscOptionsGetRealArray(NULL,NULL,"-stoich",ctx->stoichiometry,&as,NULL));
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  TS             ts;
  SNES           snes;
  SNESLineSearch linesearch;
  Vec            x;
  AppCtx         ctx;
  DM             da;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(SetFromOptions(&ctx));
  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetType(ts,TSCN));
  PetscCall(TSSetProblemType(ts,TS_NONLINEAR));
  PetscCall(TSSetIFunction(ts, NULL, FormIFunction, &ctx));
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,-4,-4,PETSC_DECIDE,PETSC_DECIDE,N_SPECIES,1,NULL,NULL,&da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMDASetUniformCoordinates(da, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0));
  PetscCall(DMDASetFieldName(da,0,"species A"));
  PetscCall(DMDASetFieldName(da,1,"species B"));
  PetscCall(DMDASetFieldName(da,2,"species C"));
  PetscCall(DMSetApplicationContext(da,&ctx));
  PetscCall(DMCreateGlobalVector(da,&x));
  PetscCall(FormInitialGuess(da, &ctx, x));

  PetscCall(TSSetDM(ts, da));
  PetscCall(TSSetMaxTime(ts,1000.0));
  PetscCall(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER));
  PetscCall(TSSetTimeStep(ts,1.0));
  PetscCall(TSSetSolution(ts,x));
  PetscCall(TSSetFromOptions(ts));

  PetscCall(TSGetSNES(ts,&snes));
  PetscCall(SNESGetLineSearch(snes,&linesearch));
  PetscCall(SNESLineSearchSetPostCheck(linesearch, ReactingFlowPostCheck, (void*)&ctx));
  PetscCall(SNESSetFromOptions(snes));
  PetscCall(TSSolve(ts,x));

  PetscCall(VecDestroy(&x));
  PetscCall(TSDestroy(&ts));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */

PetscErrorCode FormInitialGuess(DM da,AppCtx *ctx,Vec X)
{
  PetscInt       i,j,l,Mx,My,xs,ys,xm,ym;
  PetscErrorCode ierr;
  Field          **x;

  PetscFunctionBeginUser;
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,
                     PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);

  PetscCall(DMDAVecGetArray(da,X,&x));
  PetscCall(DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL));

  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      for (l = 0; l < N_SPECIES; l++) {
        if (i == 0) {
          if (l == 0)      x[j][i].sp[l] = (ctx->x_inflow.sp[l]*((PetscScalar)j) / (My - 1));
          else if (l == 1) x[j][i].sp[l] = (ctx->x_inflow.sp[l]*(1. - ((PetscScalar)j) / (My - 1)));
          else             x[j][i].sp[l] = ctx->x_0.sp[l];
        }
      }
    }
  }
  PetscCall(DMDAVecRestoreArray(da,X,&x));
  PetscFunctionReturn(0);

}

PetscErrorCode FormIFunctionLocal(DMDALocalInfo *info,PetscScalar ptime,Field **x,Field **xt,Field **f,AppCtx *ctx)
{
  PetscInt       i,j,l,m;
  PetscReal      hx,hy,dhx,dhy,hxdhy,hydhx,scale;
  PetscScalar    u,uxx,uyy;
  PetscScalar    vx, vy,sxp,syp,sxm,sym,avx,vxp,vxm,avy,vyp,vym,f_advect;
  PetscScalar    rate;

  PetscFunctionBeginUser;
  hx = ctx->length[0]/((PetscReal)(info->mx-1));
  hy = ctx->length[1]/((PetscReal)(info->my-1));

  dhx   =     1. / hx;
  dhy   =     1. / hy;
  hxdhy =  hx/hy;
  hydhx =  hy/hx;
  scale =   hx*hy;

  for (j=info->ys; j<info->ys+info->ym; j++) {
    for (i=info->xs; i<info->xs+info->xm; i++) {
      vx  = ctx->gradq_inflow*ctx->porosity*ctx->saturation;
      vy  = 0.0*dhy;
      avx = PetscAbsScalar(vx);
      vxp = .5*(vx+avx); vxm = .5*(vx-avx);
      avy = PetscAbsScalar(vy);
      vyp = .5*(vy+avy); vym = .5*(vy-avy);
      /* chemical reactions */
      for (l = 0; l < N_SPECIES; l++) {
        /* determine the velocites as the gradients of the pressure */
        if (i == 0) {
          sxp = (x[j][i+1].sp[l] - x[j][i].sp[l])*dhx;
          sxm = sxp;
        } else if (i == info->mx - 1) {
          sxp = (x[j][i].sp[l] - x[j][i-1].sp[l])*dhx;
          sxm = sxp;
        } else {
          sxm = (x[j][i+1].sp[l] - x[j][i].sp[l])*dhx;
          sxp = (x[j][i].sp[l] - x[j][i-1].sp[l])*dhx;
        }
        if (j == 0) {
          syp = (x[j+1][i].sp[l] - x[j][i].sp[l])*dhy;
          sym = syp;
        } else if (j == info->my - 1) {
          syp = (x[j][i].sp[l] - x[j-1][i].sp[l])*dhy;
          sym = syp;
        } else {
          sym = (x[j+1][i].sp[l] - x[j][i].sp[l])*dhy;
          syp = (x[j][i].sp[l] - x[j-1][i].sp[l])*dhy;
        } /* 4 flops per species*point */

        if (i == 0) {
          if (l == 0)      f[j][i].sp[l] = (x[j][i].sp[l] - ctx->x_inflow.sp[l]*((PetscScalar)j) / (info->my - 1));
          else if (l == 1) f[j][i].sp[l] = (x[j][i].sp[l] - ctx->x_inflow.sp[l]*(1. - ((PetscScalar)j) / (info->my - 1)));
          else             f[j][i].sp[l] = x[j][i].sp[l];

        } else {
          f[j][i].sp[l] = xt[j][i].sp[l]*scale;
          u       = x[j][i].sp[l];
          if (j == 0) uyy = u - x[j+1][i].sp[l];
          else if (j == info->my - 1) uyy = u - x[j-1][i].sp[l];
          else                        uyy = (2.0*u - x[j-1][i].sp[l] - x[j+1][i].sp[l])*hxdhy;

          if (i != info->mx - 1) uxx = (2.0*u - x[j][i-1].sp[l] - x[j][i+1].sp[l])*hydhx;
          else                   uxx = u - x[j][i-1].sp[l];
          /* 10 flops per species*point */

          f_advect       = 0.;
          f_advect      += scale*(vxp*sxp + vxm*sxm);
          f_advect      += scale*(vyp*syp + vym*sym);
          f[j][i].sp[l] += f_advect + ctx->dispersivity*(uxx + uyy);
          /* 14 flops per species*point */
        }
      }
      /* reaction */
      if (i != 0) {
        for (m = 0; m < N_REACTIONS; m++) {
          rate = ctx->rate_constant[m];
          for (l = 0; l < N_SPECIES; l++) {
            if (stoich(m, l) < 0) {
              /* assume an elementary reaction */
              rate *= PetscPowScalar(x[j][i].sp[l], PetscAbsScalar(stoich(m, l)));
              /* ~10 flops per reaction per species per point */
            }
          }
          for (l = 0; l < N_SPECIES; l++) {
              f[j][i].sp[l] += -scale*stoich(m, l)*rate;  /* Reaction term */
              /* 3 flops per reaction per species per point */
          }
        }
      }
    }
  }
  PetscCall(PetscLogFlops((N_SPECIES*(28.0 + 13.0*N_REACTIONS))*info->ym*info->xm));
  PetscFunctionReturn(0);
}

PetscErrorCode ReactingFlowPostCheck(SNESLineSearch linesearch, Vec X, Vec Y, Vec W, PetscBool *changed_y, PetscBool *changed_w, void *vctx)
{
  PetscInt       i,j,l,Mx,My,xs,ys,xm,ym;
  PetscErrorCode ierr;
  Field          **x;
  SNES           snes;
  DM             da;
  PetscScalar    min;

  PetscFunctionBeginUser;
   *changed_w = PETSC_FALSE;
  PetscCall(VecMin(X,NULL,&min));
  if (min >= 0.) PetscFunctionReturn(0);

  *changed_w = PETSC_TRUE;
  PetscCall(SNESLineSearchGetSNES(linesearch, &snes));
  PetscCall(SNESGetDM(snes,&da));
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,
                     PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);
  PetscCall(DMDAVecGetArray(da,W,&x));
  PetscCall(DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL));
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      for (l = 0; l < N_SPECIES; l++) {
        if (x[j][i].sp[l] < 0.) x[j][i].sp[l] = 0.;
      }
    }
  }
  PetscCall(DMDAVecRestoreArray(da,W,&x));
  PetscFunctionReturn(0);
}

PetscErrorCode FormIFunction(TS ts,PetscReal ptime,Vec X,Vec Xdot,Vec F,void *user)
{
  DMDALocalInfo  info;
  Field          **u,**udot,**fu;
  Vec            localX;
  DM             da;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts,(DM*)&da));
  PetscCall(DMGetLocalVector(da,&localX));
  PetscCall(DMGlobalToLocalBegin(da,X,INSERT_VALUES,localX));
  PetscCall(DMGlobalToLocalEnd(da,X,INSERT_VALUES,localX));
  PetscCall(DMDAGetLocalInfo(da,&info));
  PetscCall(DMDAVecGetArrayRead(da,localX,&u));
  PetscCall(DMDAVecGetArrayRead(da,Xdot,&udot));
  PetscCall(DMDAVecGetArray(da,F,&fu));
  PetscCall(FormIFunctionLocal(&info,ptime,u,udot,fu,(AppCtx*)user));
  PetscCall(DMDAVecRestoreArrayRead(da,localX,&u));
  PetscCall(DMDAVecRestoreArrayRead(da,Xdot,&udot));
  PetscCall(DMDAVecRestoreArray(da,F,&fu));
  PetscCall(DMRestoreLocalVector(da,&localX));
  PetscFunctionReturn(0);
}
