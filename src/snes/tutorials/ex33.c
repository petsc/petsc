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
  PetscCall(DMGetCoordinateDM(da, &cda));
  PetscCall(DMGetCoordinates(da, &c));
  PetscCall(DMDAGetCorners(da, &xs,NULL,NULL, &xm,NULL,NULL));
  PetscCall(DMDAVecGetArray(da, Kappa, &K));
  PetscCall(DMDAVecGetArray(cda, c, &coords));
  for (i = xs; i < xs+xm; ++i) {
#if 1
    K[i] = 1.0;
#else
    /* Notch */
    if (i == (xs+xm)/2) K[i] = 0.00000001;
    else K[i] = 1.0;
#endif
  }
  PetscCall(DMDAVecRestoreArray(da, Kappa, &K));
  PetscCall(DMDAVecRestoreArray(cda, c, &coords));
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
  PetscCall(DMGetGlobalVector(user->cda, &L));

  PetscCall(DMDAVecGetArray(info->da, user->uold,  &uold));
  PetscCall(DMDAVecGetArray(user->cda, user->Kappa, &Kappa));
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
  PetscCall(DMDAVecRestoreArray(info->da, user->uold, &uold));
  PetscCall(DMDAVecRestoreArray(user->cda, user->Kappa, &Kappa));
  /* PetscCall(PetscLogFlops(11.0*info->ym*info->xm)); */

  PetscCall(DMRestoreGlobalVector(user->cda, &L));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  SNES           snes;   /* nonlinear solver */
  DM             da;     /* grid */
  Vec            u;      /* solution vector */
  AppCtx         user;   /* user-defined work context */
  PetscReal      t = 0.0;/* time */
  PetscInt       n;

  PetscCall(PetscInitialize(&argc, &argv, NULL,help));
  /* Create solver */
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  /* Create mesh */
  PetscCall(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,4,3,1,NULL,&da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMSetApplicationContext(da, &user));
  PetscCall(SNESSetDM(snes, da));
  /* Create coefficient */
  PetscCall(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,4,1,1,NULL,&user.cda));
  PetscCall(DMSetFromOptions(user.cda));
  PetscCall(DMSetUp(user.cda));
  PetscCall(DMDASetUniformCoordinates(user.cda, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0));
  PetscCall(DMGetGlobalVector(user.cda, &user.Kappa));
  PetscCall(FormPermeability(user.cda, user.Kappa, &user));
  /* Setup Problem */
  PetscCall(DMDASNESSetFunctionLocal(da,INSERT_VALUES,(PetscErrorCode (*)(DMDALocalInfo*,void*,void*,void*))FormFunctionLocal,&user));
  PetscCall(DMGetGlobalVector(da, &u));
  PetscCall(DMGetGlobalVector(da, &user.uold));

  user.sl  = 1.0;
  user.vl  = 0.1;
  user.pl  = 1.0;
  user.phi = 1.0;

  user.kappaWet   = 1.0;
  user.kappaNoWet = 0.3;

  /* Time Loop */
  user.dt = 0.1;
  for (n = 0; n < 100; ++n, t += user.dt) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Starting time %g\n", (double)t));
    PetscCall(VecView(u, PETSC_VIEWER_DRAW_WORLD));
    /* Solve */
    PetscCall(SNESSetFromOptions(snes));
    PetscCall(SNESSolve(snes, NULL, u));
    /* Update */
    PetscCall(VecCopy(u, user.uold));

    PetscCall(VecView(u, PETSC_VIEWER_DRAW_WORLD));
  }
  /* Cleanup */
  PetscCall(DMRestoreGlobalVector(da, &u));
  PetscCall(DMRestoreGlobalVector(da, &user.uold));
  PetscCall(DMRestoreGlobalVector(user.cda, &user.Kappa));
  PetscCall(DMDestroy(&user.cda));
  PetscCall(DMDestroy(&da));
  PetscCall(SNESDestroy(&snes));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    requires: !single
    args: -snes_converged_reason -snes_monitor_short

TEST*/
