static char help[] = "Time-Dependent Heat Equation example in 2D with Varying Coefficients";

/*T
   Concepts: Heat Equation
   Concepts: DMDA with timestepping
   Concepts: Variable Coefficient
   Concepts: Nonlinear Domain Decomposition and Multigrid with Coefficients
   Processors: n
T*/

#include <petscdmda.h>
#include <petscsnes.h>
#include <petscts.h>

extern PetscErrorCode FormInitialGuess(DM da,void *ctx,Vec X);
extern PetscErrorCode FormDiffusionCoefficient(DM da,void *ctx,Vec X);
extern PetscErrorCode FormIFunctionLocal(DMDALocalInfo*,PetscReal,PetscScalar**,PetscScalar**,PetscScalar**,void*);

/* hooks */

#undef __FUNCT__
#define __FUNCT__ "CoefficientRestrictHook"
/* This could restrict auxiliary information to the coarse level.
 */
static PetscErrorCode CoefficientRestrictHook(DM dm,Mat restrct,Vec rscale,Mat Inject,DM dmc,void *ctx)
{
  Vec            c,cc;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = DMGetNamedGlobalVector(dm,"coefficient",&c);CHKERRQ(ierr);
  ierr = DMGetNamedGlobalVector(dmc,"coefficient",&cc);CHKERRQ(ierr);

  /* restrict the coefficient rather than injecting it */
  ierr = MatRestrict(restrct,c,cc);CHKERRQ(ierr);

  ierr = DMRestoreNamedGlobalVector(dm,"coefficient",&c);CHKERRQ(ierr);
  ierr = DMRestoreNamedGlobalVector(dmc,"coefficient",&cc);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "CoefficientSubDomainRestrictHook"
/* This could restrict auxiliary information to the coarse level.
 */
static PetscErrorCode CoefficientSubDomainRestrictHook(DM dm,VecScatter gscat,VecScatter lscat,DM subdm,void *ctx)
{
  Vec            c,cc;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = DMGetNamedGlobalVector(dm,"coefficient",&c);CHKERRQ(ierr);
  ierr = DMGetNamedGlobalVector(subdm,"coefficient",&cc);CHKERRQ(ierr);

  ierr = VecScatterBegin(gscat,c,cc,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(gscat,c,cc,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  ierr = DMRestoreNamedGlobalVector(dm,"coefficient",&c);CHKERRQ(ierr);
  ierr = DMRestoreNamedGlobalVector(subdm,"coefficient",&cc);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  TS                     ts;
  Vec                    x,c;
  PetscErrorCode         ierr;
  DM                     da;

  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = TSCreate(PETSC_COMM_WORLD, &ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSCN);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = DMDACreate2d(PETSC_COMM_WORLD, DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE,DMDA_STENCIL_STAR,-4,-4,PETSC_DECIDE,PETSC_DECIDE,1,1,PETSC_NULL,PETSC_NULL,&da);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(da, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);CHKERRQ(ierr);

  ierr = DMDASetFieldName(da,0,"Heat");CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&x);CHKERRQ(ierr);
  ierr = FormInitialGuess(da,PETSC_NULL,x);CHKERRQ(ierr);
  ierr = DMDATSSetIFunctionLocal(da,INSERT_VALUES,(PetscErrorCode (*)(DMDALocalInfo*,PetscReal,void*,void*,void*,void*))FormIFunctionLocal,PETSC_NULL);CHKERRQ(ierr);

  /* set up the coefficient */
  ierr = DMGetNamedGlobalVector(da,"coefficient",&c);CHKERRQ(ierr);
  ierr = FormDiffusionCoefficient(da,PETSC_NULL,c);CHKERRQ(ierr);
  ierr = DMRestoreNamedGlobalVector(da,"coefficient",&c);CHKERRQ(ierr);

  ierr = DMCoarsenHookAdd(da,PETSC_NULL,CoefficientRestrictHook,ts);CHKERRQ(ierr);
  ierr = DMSubDomainHookAdd(da,PETSC_NULL,CoefficientSubDomainRestrictHook,ts);CHKERRQ(ierr);

  ierr = TSSetDM(ts, da);CHKERRQ(ierr);
  ierr = TSSetDuration(ts,10000,1000.0);CHKERRQ(ierr);
  ierr = TSSetInitialTimeStep(ts,0.0,0.05);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,x);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  ierr = TSSolve(ts,x);CHKERRQ(ierr);

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscFinalize();

  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */

#undef __FUNCT__
#define __FUNCT__ "FormInitialGuess"
PetscErrorCode FormInitialGuess(DM da,void *ctx,Vec X)
{
  PetscInt       i,j,Mx,My,xs,ys,xm,ym;
  PetscErrorCode ierr;
  PetscScalar    **x;
  PetscReal      x0,x1;

  PetscFunctionBeginUser;
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,
		     PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);

  ierr = DMDAVecGetArray(da,X,&x);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&xs,&ys,PETSC_NULL,&xm,&ym,PETSC_NULL);CHKERRQ(ierr);

  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      x0 = 10.0*(i - 0.5*(Mx-1)) / (Mx-1);
      x1 = 10.0*(j - 0.5*(Mx-1)) / (My-1);
      if (PetscAbsScalar(x0) < 0.5 && PetscAbsScalar(x1) < 0.5) {
        x[j][i] = 1.;
      } else {
        x[j][i] = 0.0;
      }
    }
  }
  ierr = DMDAVecRestoreArray(da,X,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);

}

#undef __FUNCT__
#define __FUNCT__ "FormDiffusionCoefficient"
PetscErrorCode FormDiffusionCoefficient(DM da,void *ctx,Vec X)
{
  PetscInt       i,j,Mx,My,xs,ys,xm,ym;
  PetscErrorCode ierr;
  PetscScalar    **x;
  PetscReal      x1,min=0.1;

  PetscFunctionBeginUser;
  ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,&My,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,
		     PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);

  /*
  ierr = VecSetRandom(X,PETSC_NULL);
  ierr = VecMin(X,PETSC_NULL,&min);CHKERRQ(ierr);
   */

  ierr = DMDAVecGetArray(da,X,&x);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&xs,&ys,PETSC_NULL,&xm,&ym,PETSC_NULL);CHKERRQ(ierr);

  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      x1 = 10.0*(j - 0.5*(Mx-1)) / (My-1);
      x[j][i] += min + (1. - min)*PetscSqr(sin(2.*PETSC_PI*x1));
    }
  }
  ierr = DMDAVecRestoreArray(da,X,&x);CHKERRQ(ierr);

  PetscFunctionReturn(0);

}

#undef __FUNCT__
#define __FUNCT__ "FormIFunctionLocal"
PetscErrorCode FormIFunctionLocal(DMDALocalInfo *info,PetscScalar ptime,PetscScalar **x,PetscScalar **xt,PetscScalar **f,void *ctx)
{
  PetscErrorCode ierr;
  PetscInt       i,j;
  PetscReal      hx,hy,hxdhy,hydhx,scale;
  PetscScalar    u,uxx,uyy;
  Vec            C;
  PetscScalar    **c;

  PetscFunctionBeginUser;

  ierr = DMGetNamedGlobalVector(info->da,"coefficient",&C);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(info->da,C,&c);CHKERRQ(ierr);

  hx     = 10.0/((PetscReal)(info->mx-1));
  hy     = 10.0/((PetscReal)(info->my-1));

  /*
  dhx =     1. / hx;
  dhy =     1. / hy;
   */
  hxdhy  =  hx/hy;
  hydhx  =  hy/hx;
  scale =   hx*hy;

  for (j=info->ys; j<info->ys+info->ym; j++) {
    for (i=info->xs; i<info->xs+info->xm; i++) {
      f[j][i] = xt[j][i]*scale;
      if (i == 0) {
        /* f[j][i] += (x[j][i] - x[j][i+1])*dhx; */
      } else if (i == info->mx-1) {
        /* f[j][i] += (x[j][i] - x[j][i-1])*dhx; */
      } else if (j == 0) {
        /* f[j][i] += (x[j][i] - x[j+1][i])*dhy; */
      } else if (j == info->my-1) {
        /* f[j][i] += (x[j][i] - x[j-1][i])*dhy; */
      } else {
        u       = x[j][i];
        uyy     = (2.0*u - x[j-1][i] - x[j+1][i])*hxdhy;
        uxx     = (2.0*u - x[j][i-1] - x[j][i+1])*hydhx;
        f[j][i] += c[j][i]*(uxx + uyy);
      }
    }
  }
  ierr = PetscLogFlops(11.*info->ym*info->xm);CHKERRQ(ierr);

  ierr = DMDAVecRestoreArray(info->da,C,&c);CHKERRQ(ierr);
  ierr = DMRestoreNamedGlobalVector(info->da,"coefficient",&C);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
