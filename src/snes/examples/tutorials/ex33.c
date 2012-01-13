static char help[] = "Multiphase flow in a porous medium in 1d.\n\n";
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

#undef __FUNCT__
#define __FUNCT__ "FormPermeability"
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
  PetscScalar   *K;
  PetscScalar   *coords;
  PetscInt       xs, xm, i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMDAGetCoordinateDA(da, &cda);CHKERRQ(ierr);
  ierr = DMDAGetCoordinates(da, &c);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da, &xs,PETSC_NULL,PETSC_NULL, &xm,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, Kappa, &K);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(cda, c, &coords);CHKERRQ(ierr);
  for(i = xs; i < xs+xm; ++i) {
#if 1
    K[i] = 1.0;
#else
    /* Notch */
    if (i == (xs+xm)/2) {
      K[i] = 0.00000001;
    } else {
      K[i] = 1.0;
    }
#endif
  }
  ierr = DMDAVecRestoreArray(da, Kappa, &K);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(cda, c, &coords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormFunctionLocal"
/*
   FormFunctionLocal - Evaluates nonlinear residual, F(x) on local process patch
*/
PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, Field *u, Field *f, AppCtx *user)
{
  Vec L;
  PetscReal      phi = user->phi;
  PetscReal      dt  = user->dt;
  PetscReal      dx  = 1.0/(PetscReal)(info->mx-1);
  PetscReal      alpha = 2.0;
  PetscReal      beta  = 2.0;
  PetscReal      kappaWet   = user->kappaWet;
  PetscReal      kappaNoWet = user->kappaNoWet;
  Field         *uold;
  PetscScalar   *Kappa;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetGlobalVector(user->cda, &L);CHKERRQ(ierr);

  ierr = DMDAVecGetArray(info->da, user->uold,  &uold);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->cda, user->Kappa, &Kappa);CHKERRQ(ierr);
  /* Compute residual over the locally owned part of the grid */
  for(i = info->xs; i < info->xs+info->xm; ++i) {
    if (i == 0) {
      f[i].s = u[i].s - user->sl;
      f[i].v = u[i].v - user->vl;
      f[i].p = u[i].p - user->pl;
    } else {
      PetscScalar K          = 2*dx/(dx/Kappa[i] + dx/Kappa[i-1]);
      PetscReal   lambdaWet  = kappaWet*pow(u[i].s, alpha);
      PetscReal   lambda     = lambdaWet + kappaNoWet*pow(1-u[i].s, beta);
      PetscReal   lambdaWetL = kappaWet*pow(u[i-1].s, alpha);
      PetscReal   lambdaL    = lambdaWetL + kappaNoWet*pow(1-u[i-1].s, beta);

      f[i].s = phi*(u[i].s - uold[i].s) + (dt/dx)*((lambdaWet/lambda)*u[i].v - (lambdaWetL/lambdaL)*u[i-1].v);

      f[i].v = u[i].v + K*lambda*(u[i].p - u[i-1].p)/dx;

      //pxx     = (2.0*u[i].p - u[i-1].p - u[i+1].p)/dx;
      f[i].p = u[i].v - u[i-1].v;
    }
  }
  ierr = DMDAVecRestoreArray(info->da, user->uold, &uold);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->cda, user->Kappa, &Kappa);CHKERRQ(ierr);
  /* ierr = PetscLogFlops(11.0*info->ym*info->xm);CHKERRQ(ierr); */

  ierr = DMRestoreGlobalVector(user->cda, &L);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  SNES           snes;   /* nonlinear solver */
  DM             da;     /* grid */
  Vec            u;      /* solution vector */
  AppCtx         user;   /* user-defined work context */
  PetscReal      t;      /* time */
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, PETSC_NULL, help);CHKERRQ(ierr);
  /* Create solver */
  ierr = SNESCreate(PETSC_COMM_WORLD, &snes);CHKERRQ(ierr);
  /* Create mesh */
  ierr = DMDACreate1d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE,-4,3,1,PETSC_NULL,&da);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(da, &user);CHKERRQ(ierr);
  ierr = SNESSetDM(snes, da);CHKERRQ(ierr);
  /* Create coefficient */
  ierr = DMDACreate1d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE,-4,1,1,PETSC_NULL,&user.cda);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(user.cda, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(user.cda, &user.Kappa);CHKERRQ(ierr);
  ierr = FormPermeability(user.cda, user.Kappa, &user);CHKERRQ(ierr);
  /* Setup Problem */
  ierr = DMDASetLocalFunction(da, (DMDALocalFunction1) FormFunctionLocal);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(da, &u);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(da, &user.uold);CHKERRQ(ierr);
  user.sl  = 1.0;
  user.vl  = 0.1;
  user.pl  = 1.0;
  user.phi = 1.0;
  user.kappaWet   = 1.0;
  user.kappaNoWet = 0.3;
  /* Time Loop */
  user.dt = 0.1;
  for(PetscInt n = 0; n < 100; ++n, t += user.dt) {
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Starting time %g\n", t);CHKERRQ(ierr);
    ierr = VecView(u, PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
    /* Solve */
    ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
    ierr = SNESSolve(snes, PETSC_NULL, u);CHKERRQ(ierr);
    /* Update */
    ierr = VecCopy(u, user.uold);CHKERRQ(ierr);

    ierr = VecView(u, PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
  }
  /* Cleanup */
  ierr = DMRestoreGlobalVector(da, &u);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(da, &user.uold);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(user.cda, &user.Kappa);CHKERRQ(ierr);
  ierr = DMDestroy(&user.cda);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
