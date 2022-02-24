static char help[] = "Time-Dependent Allan-Cahn example in 2D with Varying Coefficients";

/*
 This example is mainly here to show how to transfer coefficients between subdomains and levels in
 multigrid and domain decomposition.
 */

/*T
   Concepts: Alan-Cahn
   Concepts: DMDA with timestepping
   Concepts: Variable Coefficient
   Concepts: Nonlinear Domain Decomposition and Multigrid with Coefficients
   Processors: n
T*/

#include <petscdm.h>
#include <petscdmda.h>
#include <petscsnes.h>
#include <petscts.h>

typedef struct {
  PetscScalar epsilon;
  PetscScalar beta;
} Coeff;

typedef struct {
  PetscScalar u;
} Field;

extern PetscErrorCode FormInitialGuess(DM da,void *ctx,Vec X);
extern PetscErrorCode FormDiffusionCoefficient(DM da,void *ctx,Vec X);
extern PetscErrorCode FormIFunctionLocal(DMDALocalInfo*,PetscReal,Field**,Field**,Field**,void*);

/* hooks */

static PetscErrorCode CoefficientCoarsenHook(DM dm, DM dmc,void *ctx)
{
  Vec            c,cc,ccl;
  Mat            J;
  Vec            vscale;
  DM             cdm,cdmc;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectQuery((PetscObject)dm,"coefficientdm",(PetscObject*)&cdm));

  PetscCheck(cdm,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"The coefficient DM needs to be set up!");

  CHKERRQ(DMDACreateCompatibleDMDA(dmc,2,&cdmc));
  CHKERRQ(PetscObjectCompose((PetscObject)dmc,"coefficientdm",(PetscObject)cdmc));

  CHKERRQ(DMGetNamedGlobalVector(cdm,"coefficient",&c));
  CHKERRQ(DMGetNamedGlobalVector(cdmc,"coefficient",&cc));
  CHKERRQ(DMGetNamedLocalVector(cdmc,"coefficient",&ccl));

  CHKERRQ(DMCreateInterpolation(cdmc,cdm,&J,&vscale));
  CHKERRQ(MatRestrict(J,c,cc));
  CHKERRQ(VecPointwiseMult(cc,vscale,cc));

  CHKERRQ(MatDestroy(&J));
  CHKERRQ(VecDestroy(&vscale));

  CHKERRQ(DMGlobalToLocalBegin(cdmc,cc,INSERT_VALUES,ccl));
  CHKERRQ(DMGlobalToLocalEnd(cdmc,cc,INSERT_VALUES,ccl));

  CHKERRQ(DMRestoreNamedGlobalVector(cdm,"coefficient",&c));
  CHKERRQ(DMRestoreNamedGlobalVector(cdmc,"coefficient",&cc));
  CHKERRQ(DMRestoreNamedLocalVector(cdmc,"coefficient",&ccl));

  CHKERRQ(DMCoarsenHookAdd(dmc,CoefficientCoarsenHook,NULL,NULL));
  CHKERRQ(DMDestroy(&cdmc));
  PetscFunctionReturn(0);
}

/* This could restrict auxiliary information to the coarse level.
 */
static PetscErrorCode CoefficientSubDomainRestrictHook(DM dm,DM subdm,void *ctx)
{
  Vec            c,cc;
  DM             cdm,csubdm;
  VecScatter     *iscat,*oscat,*gscat;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectQuery((PetscObject)dm,"coefficientdm",(PetscObject*)&cdm));

  PetscCheck(cdm,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"The coefficient DM needs to be set up!");

  CHKERRQ(DMDACreateCompatibleDMDA(subdm,2,&csubdm));
  CHKERRQ(PetscObjectCompose((PetscObject)subdm,"coefficientdm",(PetscObject)csubdm));

  CHKERRQ(DMGetNamedGlobalVector(cdm,"coefficient",&c));
  CHKERRQ(DMGetNamedLocalVector(csubdm,"coefficient",&cc));

  CHKERRQ(DMCreateDomainDecompositionScatters(cdm,1,&csubdm,&iscat,&oscat,&gscat));

  CHKERRQ(VecScatterBegin(*gscat,c,cc,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(*gscat,c,cc,INSERT_VALUES,SCATTER_FORWARD));

  CHKERRQ(VecScatterDestroy(iscat));
  CHKERRQ(VecScatterDestroy(oscat));
  CHKERRQ(VecScatterDestroy(gscat));
  CHKERRQ(PetscFree(iscat));
  CHKERRQ(PetscFree(oscat));
  CHKERRQ(PetscFree(gscat));

  CHKERRQ(DMRestoreNamedGlobalVector(cdm,"coefficient",&c));
  CHKERRQ(DMRestoreNamedLocalVector(csubdm,"coefficient",&cc));

  CHKERRQ(DMDestroy(&csubdm));
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)

{
  TS             ts;
  Vec            x,c,clocal;
  PetscErrorCode ierr;
  DM             da,cda;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(TSCreate(PETSC_COMM_WORLD, &ts));
  CHKERRQ(TSSetType(ts,TSARKIMEX));
  CHKERRQ(TSSetProblemType(ts,TS_NONLINEAR));
  CHKERRQ(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,4,4,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&da));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));
  CHKERRQ(DMDASetUniformCoordinates(da, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0));

  CHKERRQ(DMDASetFieldName(da,0,"u"));
  CHKERRQ(DMCreateGlobalVector(da,&x));

  CHKERRQ(TSSetDM(ts, da));

  CHKERRQ(FormInitialGuess(da,NULL,x));
  CHKERRQ(DMDATSSetIFunctionLocal(da,INSERT_VALUES,(PetscErrorCode (*)(DMDALocalInfo*,PetscReal,void*,void*,void*,void*))FormIFunctionLocal,NULL));

  /* set up the coefficient */

  CHKERRQ(DMDACreateCompatibleDMDA(da,2,&cda));
  CHKERRQ(PetscObjectCompose((PetscObject)da,"coefficientdm",(PetscObject)cda));

  CHKERRQ(DMGetNamedGlobalVector(cda,"coefficient",&c));
  CHKERRQ(DMGetNamedLocalVector(cda,"coefficient",&clocal));

  CHKERRQ(FormDiffusionCoefficient(cda,NULL,c));

  CHKERRQ(DMGlobalToLocalBegin(cda,c,INSERT_VALUES,clocal));
  CHKERRQ(DMGlobalToLocalEnd(cda,c,INSERT_VALUES,clocal));

  CHKERRQ(DMRestoreNamedLocalVector(cda,"coefficient",&clocal));
  CHKERRQ(DMRestoreNamedGlobalVector(cda,"coefficient",&c));

  CHKERRQ(DMCoarsenHookAdd(da,CoefficientCoarsenHook,NULL,NULL));
  CHKERRQ(DMSubDomainHookAdd(da,CoefficientSubDomainRestrictHook,NULL,NULL));

  CHKERRQ(TSSetMaxSteps(ts,10000));
  CHKERRQ(TSSetMaxTime(ts,10000.0));
  CHKERRQ(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER));
  CHKERRQ(TSSetTimeStep(ts,0.05));
  CHKERRQ(TSSetSolution(ts,x));
  CHKERRQ(TSSetFromOptions(ts));

  CHKERRQ(TSSolve(ts,x));

  CHKERRQ(VecDestroy(&x));
  CHKERRQ(TSDestroy(&ts));
  CHKERRQ(DMDestroy(&da));
  CHKERRQ(DMDestroy(&cda));

  ierr = PetscFinalize();
  return ierr;
}

/* ------------------------------------------------------------------- */

PetscErrorCode FormInitialGuess(DM da,void *ctx,Vec X)
{
  PetscInt       i,j,Mx,My,xs,ys,xm,ym;
  Field          **x;
  PetscReal      x0,x1;

  PetscFunctionBeginUser;
  CHKERRQ(DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE));

  CHKERRQ(DMDAVecGetArray(da,X,&x));
  CHKERRQ(DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL));

  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      x0        = 10.0*(i - 0.5*(Mx-1)) / (Mx-1);
      x1        = 10.0*(j - 0.5*(Mx-1)) / (My-1);
      x[j][i].u = PetscCosReal(2.0*PetscSqrtReal(x1*x1 + x0*x0));
    }
  }

  CHKERRQ(DMDAVecRestoreArray(da,X,&x));
  PetscFunctionReturn(0);

}

PetscErrorCode FormDiffusionCoefficient(DM da,void *ctx,Vec X)
{
  PetscInt       i,j,Mx,My,xs,ys,xm,ym;
  Coeff          **x;
  PetscReal      x1,x0;

  PetscFunctionBeginUser;
  CHKERRQ(DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE));

  /*
  ierr = VecSetRandom(X,NULL);
  CHKERRQ(VecMin(X,NULL,&min));
   */

  CHKERRQ(DMDAVecGetArray(da,X,&x));
  CHKERRQ(DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL));

  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      x0 = 10.0*(i - 0.5*(Mx-1)) / (Mx-1);
      x1 = 10.0*(j - 0.5*(My-1)) / (My-1);

      x[j][i].epsilon = 0.0;
      x[j][i].beta    = 0.05+0.05*PetscSqrtReal(x0*x0+x1*x1);
    }
  }

  CHKERRQ(DMDAVecRestoreArray(da,X,&x));
  PetscFunctionReturn(0);

}

PetscErrorCode FormIFunctionLocal(DMDALocalInfo *info,PetscReal ptime,Field **x,Field **xt,Field **f,void *ctx)
{
  PetscInt       i,j;
  PetscReal      hx,hy,dhx,dhy,hxdhy,hydhx,scale;
  PetscScalar    u,uxx,uyy;
  PetscScalar    ux,uy,bx,by;
  Vec            C;
  Coeff          **c;
  DM             cdm;

  PetscFunctionBeginUser;
  CHKERRQ(PetscObjectQuery((PetscObject)info->da,"coefficientdm",(PetscObject*)&cdm));
  CHKERRQ(DMGetNamedLocalVector(cdm,"coefficient",&C));
  CHKERRQ(DMDAVecGetArray(cdm,C,&c));

  hx = 10.0/((PetscReal)(info->mx-1));
  hy = 10.0/((PetscReal)(info->my-1));

  dhx = 1. / hx;
  dhy = 1. / hy;

  hxdhy =  hx/hy;
  hydhx =  hy/hx;
  scale =   hx*hy;

  for (j=info->ys; j<info->ys+info->ym; j++) {
    for (i=info->xs; i<info->xs+info->xm; i++) {
      f[j][i].u = xt[j][i].u*scale;

      u = x[j][i].u;

      f[j][i].u += scale*(u*u - 1.)*u;

      if (i == 0)               f[j][i].u += (x[j][i].u - x[j][i+1].u)*dhx;
      else if (i == info->mx-1) f[j][i].u += (x[j][i].u - x[j][i-1].u)*dhx;
      else if (j == 0)          f[j][i].u += (x[j][i].u - x[j+1][i].u)*dhy;
      else if (j == info->my-1) f[j][i].u += (x[j][i].u - x[j-1][i].u)*dhy;
      else {
        uyy     = (2.0*u - x[j-1][i].u - x[j+1][i].u)*hxdhy;
        uxx     = (2.0*u - x[j][i-1].u - x[j][i+1].u)*hydhx;

        bx      = 0.5*(c[j][i+1].beta - c[j][i-1].beta)*dhx;
        by      = 0.5*(c[j+1][i].beta - c[j-1][i].beta)*dhy;

        ux      = 0.5*(x[j][i+1].u - x[j][i-1].u)*dhx;
        uy      = 0.5*(x[j+1][i].u - x[j-1][i].u)*dhy;

        f[j][i].u += c[j][i].beta*(uxx + uyy) + scale*(bx*ux + by*uy);
       }
    }
  }
  CHKERRQ(PetscLogFlops(11.*info->ym*info->xm));

  CHKERRQ(DMDAVecRestoreArray(cdm,C,&c));
  CHKERRQ(DMRestoreNamedLocalVector(cdm,"coefficient",&C));
  PetscFunctionReturn(0);
}

/*TEST

    test:
      args: -da_refine 4 -ts_max_steps 10 -ts_rtol 1e-3 -ts_atol 1e-3 -ts_type arkimex -ts_monitor -snes_monitor -snes_type ngmres  -npc_snes_type nasm -npc_snes_nasm_type restrict -da_overlap 4
      nsize: 16
      requires: !single
      output_file: output/ex29.out

TEST*/
