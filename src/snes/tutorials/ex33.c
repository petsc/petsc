static char help[] = "Multiphase flow in a porous medium in 1d.\n\n";
#include <petscdm.h>
#include <petscdmda.h>
#include <petscsnes.h>

typedef struct {
  DM        cda;
  Vec       uold;
  Vec       Kappa;
  PetscReal phi;
  PetscReal kappaWet;
  PetscReal kappaNoWet;
  PetscReal dt;
  /* Boundary conditions */
  PetscReal sl, vl, pl;
} AppCtx;

typedef struct {
  PetscScalar s; /* The saturation on each cell */
  PetscScalar v; /* The velocity on each face */
  PetscScalar p; /* The pressure on each cell */
} Field;

/*
   FormPermeability - Forms permeability field.

   Input Parameters:
   user - user-defined application context

   Output Parameter:
   Kappa - vector
 */
PetscErrorCode FormPermeability(DM da, Vec Kappa, AppCtx *user)
{
  DM             cda;
  Vec            c;
  PetscScalar    *K;
  PetscScalar    *coords;
  PetscInt       xs, xm, i;

  PetscFunctionBeginUser;
  CHKERRQ(DMGetCoordinateDM(da, &cda));
  CHKERRQ(DMGetCoordinates(da, &c));
  CHKERRQ(DMDAGetCorners(da, &xs,NULL,NULL, &xm,NULL,NULL));
  CHKERRQ(DMDAVecGetArray(da, Kappa, &K));
  CHKERRQ(DMDAVecGetArray(cda, c, &coords));
  for (i = xs; i < xs+xm; ++i) {
#if 1
    K[i] = 1.0;
#else
    /* Notch */
    if (i == (xs+xm)/2) K[i] = 0.00000001;
    else K[i] = 1.0;
#endif
  }
  CHKERRQ(DMDAVecRestoreArray(da, Kappa, &K));
  CHKERRQ(DMDAVecRestoreArray(cda, c, &coords));
  PetscFunctionReturn(0);
}

/*
   FormFunctionLocal - Evaluates nonlinear residual, F(x) on local process patch
*/
PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, Field *u, Field *f, AppCtx *user)
{
  Vec            L;
  PetscReal      phi        = user->phi;
  PetscReal      dt         = user->dt;
  PetscReal      dx         = 1.0/(PetscReal)(info->mx-1);
  PetscReal      alpha      = 2.0;
  PetscReal      beta       = 2.0;
  PetscReal      kappaWet   = user->kappaWet;
  PetscReal      kappaNoWet = user->kappaNoWet;
  Field          *uold;
  PetscScalar    *Kappa;
  PetscInt       i;

  PetscFunctionBeginUser;
  CHKERRQ(DMGetGlobalVector(user->cda, &L));

  CHKERRQ(DMDAVecGetArray(info->da, user->uold,  &uold));
  CHKERRQ(DMDAVecGetArray(user->cda, user->Kappa, &Kappa));
  /* Compute residual over the locally owned part of the grid */
  for (i = info->xs; i < info->xs+info->xm; ++i) {
    if (i == 0) {
      f[i].s = u[i].s - user->sl;
      f[i].v = u[i].v - user->vl;
      f[i].p = u[i].p - user->pl;
    } else {
      PetscScalar K          = 2*dx/(dx/Kappa[i] + dx/Kappa[i-1]);
      PetscReal   lambdaWet  = kappaWet*PetscRealPart(PetscPowScalar(u[i].s, alpha));
      PetscReal   lambda     = lambdaWet + kappaNoWet*PetscRealPart(PetscPowScalar(1.-u[i].s, beta));
      PetscReal   lambdaWetL = kappaWet*PetscRealPart(PetscPowScalar(u[i-1].s, alpha));
      PetscReal   lambdaL    = lambdaWetL + kappaNoWet*PetscRealPart(PetscPowScalar(1.-u[i-1].s, beta));

      f[i].s = phi*(u[i].s - uold[i].s) + (dt/dx)*((lambdaWet/lambda)*u[i].v - (lambdaWetL/lambdaL)*u[i-1].v);

      f[i].v = u[i].v + K*lambda*(u[i].p - u[i-1].p)/dx;

      /*pxx     = (2.0*u[i].p - u[i-1].p - u[i+1].p)/dx;*/
      f[i].p = u[i].v - u[i-1].v;
    }
  }
  CHKERRQ(DMDAVecRestoreArray(info->da, user->uold, &uold));
  CHKERRQ(DMDAVecRestoreArray(user->cda, user->Kappa, &Kappa));
  /* CHKERRQ(PetscLogFlops(11.0*info->ym*info->xm)); */

  CHKERRQ(DMRestoreGlobalVector(user->cda, &L));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  SNES           snes;   /* nonlinear solver */
  DM             da;     /* grid */
  Vec            u;      /* solution vector */
  AppCtx         user;   /* user-defined work context */
  PetscReal      t = 0.0;/* time */
  PetscErrorCode ierr;
  PetscInt       n;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  /* Create solver */
  CHKERRQ(SNESCreate(PETSC_COMM_WORLD, &snes));
  /* Create mesh */
  CHKERRQ(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,4,3,1,NULL,&da));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));
  CHKERRQ(DMSetApplicationContext(da, &user));
  CHKERRQ(SNESSetDM(snes, da));
  /* Create coefficient */
  CHKERRQ(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,4,1,1,NULL,&user.cda));
  CHKERRQ(DMSetFromOptions(user.cda));
  CHKERRQ(DMSetUp(user.cda));
  CHKERRQ(DMDASetUniformCoordinates(user.cda, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0));
  CHKERRQ(DMGetGlobalVector(user.cda, &user.Kappa));
  CHKERRQ(FormPermeability(user.cda, user.Kappa, &user));
  /* Setup Problem */
  CHKERRQ(DMDASNESSetFunctionLocal(da,INSERT_VALUES,(PetscErrorCode (*)(DMDALocalInfo*,void*,void*,void*))FormFunctionLocal,&user));
  CHKERRQ(DMGetGlobalVector(da, &u));
  CHKERRQ(DMGetGlobalVector(da, &user.uold));

  user.sl  = 1.0;
  user.vl  = 0.1;
  user.pl  = 1.0;
  user.phi = 1.0;

  user.kappaWet   = 1.0;
  user.kappaNoWet = 0.3;

  /* Time Loop */
  user.dt = 0.1;
  for (n = 0; n < 100; ++n, t += user.dt) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "Starting time %g\n", (double)t));
    CHKERRQ(VecView(u, PETSC_VIEWER_DRAW_WORLD));
    /* Solve */
    CHKERRQ(SNESSetFromOptions(snes));
    CHKERRQ(SNESSolve(snes, NULL, u));
    /* Update */
    CHKERRQ(VecCopy(u, user.uold));

    CHKERRQ(VecView(u, PETSC_VIEWER_DRAW_WORLD));
  }
  /* Cleanup */
  CHKERRQ(DMRestoreGlobalVector(da, &u));
  CHKERRQ(DMRestoreGlobalVector(da, &user.uold));
  CHKERRQ(DMRestoreGlobalVector(user.cda, &user.Kappa));
  CHKERRQ(DMDestroy(&user.cda));
  CHKERRQ(DMDestroy(&da));
  CHKERRQ(SNESDestroy(&snes));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 0
    requires: !single
    args: -snes_converged_reason -snes_monitor_short

TEST*/
