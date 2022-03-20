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
  PetscErrorCode ierr;
  Mat            J;
  Vec            vscale;
  DM             cdm,cdmc;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)dm,"coefficientdm",(PetscObject*)&cdm);CHKERRQ(ierr);

  PetscCheck(cdm,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"The coefficient DM needs to be set up!");

  ierr = DMDACreateCompatibleDMDA(dmc,2,&cdmc);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)dmc,"coefficientdm",(PetscObject)cdmc);CHKERRQ(ierr);

  ierr = DMGetNamedGlobalVector(cdm,"coefficient",&c);CHKERRQ(ierr);
  ierr = DMGetNamedGlobalVector(cdmc,"coefficient",&cc);CHKERRQ(ierr);
  ierr = DMGetNamedLocalVector(cdmc,"coefficient",&ccl);CHKERRQ(ierr);

  ierr = DMCreateInterpolation(cdmc,cdm,&J,&vscale);CHKERRQ(ierr);
  ierr = MatRestrict(J,c,cc);CHKERRQ(ierr);
  ierr = VecPointwiseMult(cc,vscale,cc);CHKERRQ(ierr);

  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = VecDestroy(&vscale);CHKERRQ(ierr);

  ierr = DMGlobalToLocalBegin(cdmc,cc,INSERT_VALUES,ccl);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(cdmc,cc,INSERT_VALUES,ccl);CHKERRQ(ierr);

  ierr = DMRestoreNamedGlobalVector(cdm,"coefficient",&c);CHKERRQ(ierr);
  ierr = DMRestoreNamedGlobalVector(cdmc,"coefficient",&cc);CHKERRQ(ierr);
  ierr = DMRestoreNamedLocalVector(cdmc,"coefficient",&ccl);CHKERRQ(ierr);

  ierr = DMCoarsenHookAdd(dmc,CoefficientCoarsenHook,NULL,NULL);CHKERRQ(ierr);
  ierr = DMDestroy(&cdmc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* This could restrict auxiliary information to the coarse level.
 */
static PetscErrorCode CoefficientSubDomainRestrictHook(DM dm,DM subdm,void *ctx)
{
  Vec            c,cc;
  DM             cdm,csubdm;
  PetscErrorCode ierr;
  VecScatter     *iscat,*oscat,*gscat;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)dm,"coefficientdm",(PetscObject*)&cdm);CHKERRQ(ierr);

  PetscCheck(cdm,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"The coefficient DM needs to be set up!");

  ierr = DMDACreateCompatibleDMDA(subdm,2,&csubdm);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)subdm,"coefficientdm",(PetscObject)csubdm);CHKERRQ(ierr);

  ierr = DMGetNamedGlobalVector(cdm,"coefficient",&c);CHKERRQ(ierr);
  ierr = DMGetNamedLocalVector(csubdm,"coefficient",&cc);CHKERRQ(ierr);

  ierr = DMCreateDomainDecompositionScatters(cdm,1,&csubdm,&iscat,&oscat,&gscat);CHKERRQ(ierr);

  ierr = VecScatterBegin(*gscat,c,cc,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(*gscat,c,cc,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  ierr = VecScatterDestroy(iscat);CHKERRQ(ierr);
  ierr = VecScatterDestroy(oscat);CHKERRQ(ierr);
  ierr = VecScatterDestroy(gscat);CHKERRQ(ierr);
  ierr = PetscFree(iscat);CHKERRQ(ierr);
  ierr = PetscFree(oscat);CHKERRQ(ierr);
  ierr = PetscFree(gscat);CHKERRQ(ierr);

  ierr = DMRestoreNamedGlobalVector(cdm,"coefficient",&c);CHKERRQ(ierr);
  ierr = DMRestoreNamedLocalVector(csubdm,"coefficient",&cc);CHKERRQ(ierr);

  ierr = DMDestroy(&csubdm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)

{
  TS             ts;
  Vec            x,c,clocal;
  PetscErrorCode ierr;
  DM             da,cda;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = TSCreate(PETSC_COMM_WORLD, &ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSARKIMEX);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,4,4,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(da, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);CHKERRQ(ierr);

  ierr = DMDASetFieldName(da,0,"u");CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&x);CHKERRQ(ierr);

  ierr = TSSetDM(ts, da);CHKERRQ(ierr);

  ierr = FormInitialGuess(da,NULL,x);CHKERRQ(ierr);
  ierr = DMDATSSetIFunctionLocal(da,INSERT_VALUES,(PetscErrorCode (*)(DMDALocalInfo*,PetscReal,void*,void*,void*,void*))FormIFunctionLocal,NULL);CHKERRQ(ierr);

  /* set up the coefficient */

  ierr = DMDACreateCompatibleDMDA(da,2,&cda);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)da,"coefficientdm",(PetscObject)cda);CHKERRQ(ierr);

  ierr = DMGetNamedGlobalVector(cda,"coefficient",&c);CHKERRQ(ierr);
  ierr = DMGetNamedLocalVector(cda,"coefficient",&clocal);CHKERRQ(ierr);

  ierr = FormDiffusionCoefficient(cda,NULL,c);CHKERRQ(ierr);

  ierr = DMGlobalToLocalBegin(cda,c,INSERT_VALUES,clocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(cda,c,INSERT_VALUES,clocal);CHKERRQ(ierr);

  ierr = DMRestoreNamedLocalVector(cda,"coefficient",&clocal);CHKERRQ(ierr);
  ierr = DMRestoreNamedGlobalVector(cda,"coefficient",&c);CHKERRQ(ierr);

  ierr = DMCoarsenHookAdd(da,CoefficientCoarsenHook,NULL,NULL);CHKERRQ(ierr);
  ierr = DMSubDomainHookAdd(da,CoefficientSubDomainRestrictHook,NULL,NULL);CHKERRQ(ierr);

  ierr = TSSetMaxSteps(ts,10000);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,10000.0);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,0.05);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,x);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  ierr = TSSolve(ts,x);CHKERRQ(ierr);

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = DMDestroy(&cda);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

/* ------------------------------------------------------------------- */

PetscErrorCode FormInitialGuess(DM da,void *ctx,Vec X)
{
  PetscInt       i,j,Mx,My,xs,ys,xm,ym;
  PetscErrorCode ierr;
  Field          **x;
  PetscReal      x0,x1;

  PetscFunctionBeginUser;
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);

  ierr = DMDAVecGetArray(da,X,&x);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);

  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      x0        = 10.0*(i - 0.5*(Mx-1)) / (Mx-1);
      x1        = 10.0*(j - 0.5*(Mx-1)) / (My-1);
      x[j][i].u = PetscCosReal(2.0*PetscSqrtReal(x1*x1 + x0*x0));
    }
  }

  ierr = DMDAVecRestoreArray(da,X,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);

}

PetscErrorCode FormDiffusionCoefficient(DM da,void *ctx,Vec X)
{
  PetscInt       i,j,Mx,My,xs,ys,xm,ym;
  PetscErrorCode ierr;
  Coeff          **x;
  PetscReal      x1,x0;

  PetscFunctionBeginUser;
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);

  /*
  ierr = VecSetRandom(X,NULL);
  ierr = VecMin(X,NULL,&min);CHKERRQ(ierr);
   */

  ierr = DMDAVecGetArray(da,X,&x);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);

  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      x0 = 10.0*(i - 0.5*(Mx-1)) / (Mx-1);
      x1 = 10.0*(j - 0.5*(My-1)) / (My-1);

      x[j][i].epsilon = 0.0;
      x[j][i].beta    = 0.05+0.05*PetscSqrtReal(x0*x0+x1*x1);
    }
  }

  ierr = DMDAVecRestoreArray(da,X,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);

}

PetscErrorCode FormIFunctionLocal(DMDALocalInfo *info,PetscReal ptime,Field **x,Field **xt,Field **f,void *ctx)
{
  PetscErrorCode ierr;
  PetscInt       i,j;
  PetscReal      hx,hy,dhx,dhy,hxdhy,hydhx,scale;
  PetscScalar    u,uxx,uyy;
  PetscScalar    ux,uy,bx,by;
  Vec            C;
  Coeff          **c;
  DM             cdm;

  PetscFunctionBeginUser;
  ierr = PetscObjectQuery((PetscObject)info->da,"coefficientdm",(PetscObject*)&cdm);CHKERRQ(ierr);
  ierr = DMGetNamedLocalVector(cdm,"coefficient",&C);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(cdm,C,&c);CHKERRQ(ierr);

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
  ierr = PetscLogFlops(11.*info->ym*info->xm);CHKERRQ(ierr);

  ierr = DMDAVecRestoreArray(cdm,C,&c);CHKERRQ(ierr);
  ierr = DMRestoreNamedLocalVector(cdm,"coefficient",&C);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*TEST

    test:
      args: -da_refine 4 -ts_max_steps 10 -ts_rtol 1e-3 -ts_atol 1e-3 -ts_type arkimex -ts_monitor -snes_monitor -snes_type ngmres  -npc_snes_type nasm -npc_snes_nasm_type restrict -da_overlap 4
      nsize: 16
      requires: !single
      output_file: output/ex29.out

TEST*/
