static const char help[] = "An elastic wave equation driven by Dieterich-Ruina friction\n";
/*
This whole derivation comes from Erickson, Birnir, and Lavallee [2010]. The model comes from the continuum limit in Carlson and Langer [1989],

  u_{tt}   = c^2 u_{xx} - \tilde\gamma^2 u - (\gamma^2 / \xi) (\theta + \ln(u_t + 1))
  \theta_t = -(u_t + 1) (\theta + (1 + \epsilon) \ln(u_t +1))

which can be reduced to a first order system,

  u_t      = v
  v_t      = c^2 u_{xx} - \tilde\gamma^2 u - (\gamma^2 / \xi)(\theta + ln(v + 1)))
  \theta_t = -(v + 1) (\theta + (1 + \epsilon) \ln(v+1))
*/

#include <petscdm.h>
#include <petscdmda.h>
#include <petscts.h>

typedef struct {
  PetscScalar u,v, th;
} Field;

typedef struct _User *User;
struct _User {
  PetscReal epsilon;    /* inverse of seismic ratio, B-A / A */
  PetscReal gamma;      /* wave frequency for interblock coupling */
  PetscReal gammaTilde; /* wave frequency for coupling to plate */
  PetscReal xi;         /* interblock spring constant */
  PetscReal c;          /* wavespeed */
};

static PetscErrorCode FormRHSFunction(TS ts, PetscReal t, Vec U, Vec F, void *ctx)
{
  User              user = (User) ctx;
  DM                dm, cdm;
  DMDALocalInfo     info;
  Vec               C;
  Field             *f;
  const Field       *u;
  const PetscScalar *x;
  PetscInt          i;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &C);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(dm, &info);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(dm,  U, (void*)&u);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(dm,  F, &f);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(cdm, C, (void*)&x);CHKERRQ(ierr);
  for (i = info.xs; i < info.xs+info.xm; ++i) {
    const PetscScalar hx = i+1 == info.xs+info.xm ? x[i] - x[i-1] : x[i+1] - x[i];

    f[i].u  =  hx*(u[i].v);
    f[i].v  = -hx*(PetscSqr(user->gammaTilde)*u[i].u + (PetscSqr(user->gamma) / user->xi)*(u[i].th + PetscLogScalar(u[i].v + 1)));
    f[i].th = -hx*(u[i].v + 1)*(u[i].th + (1 + user->epsilon)*PetscLogScalar(u[i].v + 1));
  }
  ierr = DMDAVecRestoreArrayRead(dm,  U, (void*)&u);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(dm,  F, &f);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(cdm, C, (void*)&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode FormIFunction(TS ts, PetscReal t, Vec U, Vec Udot, Vec F, void *ctx)
{
  User           user = (User) ctx;
  DM             dm, cdm;
  DMDALocalInfo  info;
  Vec            Uloc, C;
  Field         *u, *udot, *f;
  PetscScalar   *x;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(dm, &info);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &C);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &Uloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm, U, INSERT_VALUES, Uloc);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm, U, INSERT_VALUES, Uloc);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(dm,  Uloc, &u);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(dm,  Udot, &udot);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(dm,  F,    &f);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(cdm, C,    &x);CHKERRQ(ierr);
  for (i = info.xs; i < info.xs+info.xm; ++i) {
    if (i == 0) {
      const PetscScalar hx = x[i+1] - x[i];
      f[i].u  = hx * udot[i].u;
      f[i].v  = hx * udot[i].v - PetscSqr(user->c) * (u[i+1].u - u[i].u) / hx;
      f[i].th = hx * udot[i].th;
    } else if (i == info.mx-1) {
      const PetscScalar hx = x[i] - x[i-1];
      f[i].u  = hx * udot[i].u;
      f[i].v  = hx * udot[i].v - PetscSqr(user->c) * (u[i-1].u - u[i].u) / hx;
      f[i].th = hx * udot[i].th;
    } else {
      const PetscScalar hx = x[i+1] - x[i];
      f[i].u  = hx * udot[i].u;
      f[i].v  = hx * udot[i].v - PetscSqr(user->c) * (u[i-1].u - 2.*u[i].u + u[i+1].u) / hx;
      f[i].th = hx * udot[i].th;
    }
  }
  ierr = DMDAVecRestoreArrayRead(dm,  Uloc, &u);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(dm,  Udot, &udot);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(dm,  F,    &f);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(cdm, C,    &x);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &Uloc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* IJacobian - Compute IJacobian = dF/dU + a dF/dUdot */
PetscErrorCode FormIJacobian(TS ts, PetscReal t, Vec U, Vec Udot, PetscReal a, Mat J, Mat Jpre, void *ctx)
{
  User           user = (User) ctx;
  DM             dm, cdm;
  DMDALocalInfo  info;
  Vec            C;
  Field         *u, *udot;
  PetscScalar   *x;
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(dm, &info);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &C);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(dm,  U,    &u);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(dm,  Udot, &udot);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(cdm, C,    &x);CHKERRQ(ierr);
  for (i = info.xs; i < info.xs+info.xm; ++i) {
    if (i == 0) {
      const PetscScalar hx            = x[i+1] - x[i];
      const PetscInt    row           = i, col[] = {i,i+1};
      const PetscScalar dxx0          = PetscSqr(user->c)/hx,dxxR = -PetscSqr(user->c)/hx;
      const PetscScalar vals[3][2][3] = {{{a*hx,     0,0},{0,0,   0}},
                                         {{0,a*hx+dxx0,0},{0,dxxR,0}},
                                         {{0,0,     a*hx},{0,0,   0}}};

      ierr = MatSetValuesBlocked(Jpre, 1, &row, 2, col, &vals[0][0][0], INSERT_VALUES);CHKERRQ(ierr);
    } else if (i == info.mx-1) {
      const PetscScalar hx            = x[i+1] - x[i];
      const PetscInt    row           = i, col[] = {i-1,i};
      const PetscScalar dxxL          = -PetscSqr(user->c)/hx, dxx0 = PetscSqr(user->c)/hx;
      const PetscScalar vals[3][2][3] = {{{0,0,   0},{a*hx,     0,0}},
                                         {{0,dxxL,0},{0,a*hx+dxx0,0}},
                                         {{0,0,   0},{0,0,     a*hx}}};

      ierr = MatSetValuesBlocked(Jpre, 1, &row, 2, col, &vals[0][0][0], INSERT_VALUES);CHKERRQ(ierr);
    } else {
      const PetscScalar hx            = x[i+1] - x[i];
      const PetscInt    row           = i, col[] = {i-1,i,i+1};
      const PetscScalar dxxL          = -PetscSqr(user->c)/hx, dxx0 = 2.*PetscSqr(user->c)/hx,dxxR = -PetscSqr(user->c)/hx;
      const PetscScalar vals[3][3][3] = {{{0,0,   0},{a*hx,     0,0},{0,0,   0}},
                                         {{0,dxxL,0},{0,a*hx+dxx0,0},{0,dxxR,0}},
                                         {{0,0,   0},{0,0,     a*hx},{0,0,   0}}};

      ierr = MatSetValuesBlocked(Jpre, 1, &row, 3, col, &vals[0][0][0], INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = DMDAVecRestoreArrayRead(dm,  U,    &u);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(dm,  Udot, &udot);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(cdm, C,    &x);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(Jpre, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Jpre, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (J != Jpre) {
    ierr = MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode FormInitialSolution(TS ts, Vec U, void *ctx)
{
  /* User            user = (User) ctx; */
  DM              dm, cdm;
  DMDALocalInfo   info;
  Vec             C;
  Field          *u;
  PetscScalar    *x;
  const PetscReal sigma = 1.0;
  PetscInt        i;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  ierr = TSGetDM(ts, &dm);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &C);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(dm, &info);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(dm,  U, &u);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayRead(cdm, C, &x);CHKERRQ(ierr);
  for (i = info.xs; i < info.xs+info.xm; ++i) {
    u[i].u  = 1.5 * PetscExpScalar(-PetscSqr(x[i] - 10)/PetscSqr(sigma));
    u[i].v  = 0.0;
    u[i].th = 0.0;
  }
  ierr = DMDAVecRestoreArray(dm,  U, &u);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(cdm, C, &x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM                dm;
  TS                ts;
  Vec               X;
  Mat               J;
  PetscInt          steps, mx;
  PetscReal         ftime, hx, dt;
  TSConvergedReason reason;
  struct _User      user;
  PetscErrorCode    ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, 11, 3, 1, NULL, &dm);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMSetUp(dm);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(dm, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm, &X);CHKERRQ(ierr);

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Dynamic Friction Options", "");
  {
    user.epsilon    = 0.1;
    user.gamma      = 0.5;
    user.gammaTilde = 0.5;
    user.xi         = 0.5;
    user.c          = 0.5;
    ierr = PetscOptionsReal("-epsilon", "Inverse of seismic ratio", "", user.epsilon, &user.epsilon, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-gamma", "Wave frequency for interblock coupling", "", user.gamma, &user.gamma, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-gamma_tilde", "Wave frequency for plate coupling", "", user.gammaTilde, &user.gammaTilde, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-xi", "Interblock spring constant", "", user.xi, &user.xi, NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-c", "Wavespeed", "", user.c, &user.c, NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = TSCreate(PETSC_COMM_WORLD, &ts);CHKERRQ(ierr);
  ierr = TSSetDM(ts, dm);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts, NULL, FormRHSFunction, &user);CHKERRQ(ierr);
  ierr = TSSetIFunction(ts, NULL, FormIFunction, &user);CHKERRQ(ierr);
  ierr = DMSetMatType(dm, MATAIJ);CHKERRQ(ierr);
  ierr = DMCreateMatrix(dm, &J);CHKERRQ(ierr);
  ierr = TSSetIJacobian(ts, J, J, FormIJacobian, &user);CHKERRQ(ierr);

  ftime = 800.0;
  ierr = TSSetMaxTime(ts,ftime);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  ierr = FormInitialSolution(ts, X, &user);CHKERRQ(ierr);
  ierr = TSSetSolution(ts, X);CHKERRQ(ierr);
  ierr = VecGetSize(X, &mx);CHKERRQ(ierr);
  hx   = 20.0/(PetscReal)(mx-1);
  dt   = 0.4 * PetscSqr(hx) / PetscSqr(user.c); /* Diffusive stability limit */
  ierr = TSSetTimeStep(ts,dt);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  ierr = TSSolve(ts, X);CHKERRQ(ierr);
  ierr = TSGetSolveTime(ts, &ftime);CHKERRQ(ierr);
  ierr = TSGetStepNumber(ts, &steps);CHKERRQ(ierr);
  ierr = TSGetConvergedReason(ts, &reason);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "%s at time %g after %D steps\n", TSConvergedReasons[reason], (double)ftime, steps);CHKERRQ(ierr);

  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

    build:
      requires: !single  !complex c99

    test:
      TODO: broken, was not nightly tested, SNES solve eventually fails, -snes_test_jacobian indicates Jacobian is wrong, but even -snes_mf_operator fails

TEST*/
