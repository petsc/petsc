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

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &dm));
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMGetCoordinatesLocal(dm, &C));
  PetscCall(DMDAGetLocalInfo(dm, &info));
  PetscCall(DMDAVecGetArrayRead(dm,  U, (void*)&u));
  PetscCall(DMDAVecGetArray(dm,  F, &f));
  PetscCall(DMDAVecGetArrayRead(cdm, C, (void*)&x));
  for (i = info.xs; i < info.xs+info.xm; ++i) {
    const PetscScalar hx = i+1 == info.xs+info.xm ? x[i] - x[i-1] : x[i+1] - x[i];

    f[i].u  =  hx*(u[i].v);
    f[i].v  = -hx*(PetscSqr(user->gammaTilde)*u[i].u + (PetscSqr(user->gamma) / user->xi)*(u[i].th + PetscLogScalar(u[i].v + 1)));
    f[i].th = -hx*(u[i].v + 1)*(u[i].th + (1 + user->epsilon)*PetscLogScalar(u[i].v + 1));
  }
  PetscCall(DMDAVecRestoreArrayRead(dm,  U, (void*)&u));
  PetscCall(DMDAVecRestoreArray(dm,  F, &f));
  PetscCall(DMDAVecRestoreArrayRead(cdm, C, (void*)&x));
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

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &dm));
  PetscCall(DMDAGetLocalInfo(dm, &info));
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMGetCoordinatesLocal(dm, &C));
  PetscCall(DMGetLocalVector(dm, &Uloc));
  PetscCall(DMGlobalToLocalBegin(dm, U, INSERT_VALUES, Uloc));
  PetscCall(DMGlobalToLocalEnd(dm, U, INSERT_VALUES, Uloc));
  PetscCall(DMDAVecGetArrayRead(dm,  Uloc, &u));
  PetscCall(DMDAVecGetArrayRead(dm,  Udot, &udot));
  PetscCall(DMDAVecGetArray(dm,  F,    &f));
  PetscCall(DMDAVecGetArrayRead(cdm, C,    &x));
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
  PetscCall(DMDAVecRestoreArrayRead(dm,  Uloc, &u));
  PetscCall(DMDAVecRestoreArrayRead(dm,  Udot, &udot));
  PetscCall(DMDAVecRestoreArray(dm,  F,    &f));
  PetscCall(DMDAVecRestoreArrayRead(cdm, C,    &x));
  PetscCall(DMRestoreLocalVector(dm, &Uloc));
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

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &dm));
  PetscCall(DMDAGetLocalInfo(dm, &info));
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMGetCoordinatesLocal(dm, &C));
  PetscCall(DMDAVecGetArrayRead(dm,  U,    &u));
  PetscCall(DMDAVecGetArrayRead(dm,  Udot, &udot));
  PetscCall(DMDAVecGetArrayRead(cdm, C,    &x));
  for (i = info.xs; i < info.xs+info.xm; ++i) {
    if (i == 0) {
      const PetscScalar hx            = x[i+1] - x[i];
      const PetscInt    row           = i, col[] = {i,i+1};
      const PetscScalar dxx0          = PetscSqr(user->c)/hx,dxxR = -PetscSqr(user->c)/hx;
      const PetscScalar vals[3][2][3] = {{{a*hx,     0,0},{0,0,   0}},
                                         {{0,a*hx+dxx0,0},{0,dxxR,0}},
                                         {{0,0,     a*hx},{0,0,   0}}};

      PetscCall(MatSetValuesBlocked(Jpre, 1, &row, 2, col, &vals[0][0][0], INSERT_VALUES));
    } else if (i == info.mx-1) {
      const PetscScalar hx            = x[i+1] - x[i];
      const PetscInt    row           = i, col[] = {i-1,i};
      const PetscScalar dxxL          = -PetscSqr(user->c)/hx, dxx0 = PetscSqr(user->c)/hx;
      const PetscScalar vals[3][2][3] = {{{0,0,   0},{a*hx,     0,0}},
                                         {{0,dxxL,0},{0,a*hx+dxx0,0}},
                                         {{0,0,   0},{0,0,     a*hx}}};

      PetscCall(MatSetValuesBlocked(Jpre, 1, &row, 2, col, &vals[0][0][0], INSERT_VALUES));
    } else {
      const PetscScalar hx            = x[i+1] - x[i];
      const PetscInt    row           = i, col[] = {i-1,i,i+1};
      const PetscScalar dxxL          = -PetscSqr(user->c)/hx, dxx0 = 2.*PetscSqr(user->c)/hx,dxxR = -PetscSqr(user->c)/hx;
      const PetscScalar vals[3][3][3] = {{{0,0,   0},{a*hx,     0,0},{0,0,   0}},
                                         {{0,dxxL,0},{0,a*hx+dxx0,0},{0,dxxR,0}},
                                         {{0,0,   0},{0,0,     a*hx},{0,0,   0}}};

      PetscCall(MatSetValuesBlocked(Jpre, 1, &row, 3, col, &vals[0][0][0], INSERT_VALUES));
    }
  }
  PetscCall(DMDAVecRestoreArrayRead(dm,  U,    &u));
  PetscCall(DMDAVecRestoreArrayRead(dm,  Udot, &udot));
  PetscCall(DMDAVecRestoreArrayRead(cdm, C,    &x));
  PetscCall(MatAssemblyBegin(Jpre, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Jpre, MAT_FINAL_ASSEMBLY));
  if (J != Jpre) {
    PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
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

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &dm));
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMGetCoordinatesLocal(dm, &C));
  PetscCall(DMDAGetLocalInfo(dm, &info));
  PetscCall(DMDAVecGetArray(dm,  U, &u));
  PetscCall(DMDAVecGetArrayRead(cdm, C, &x));
  for (i = info.xs; i < info.xs+info.xm; ++i) {
    u[i].u  = 1.5 * PetscExpScalar(-PetscSqr(x[i] - 10)/PetscSqr(sigma));
    u[i].v  = 0.0;
    u[i].th = 0.0;
  }
  PetscCall(DMDAVecRestoreArray(dm,  U, &u));
  PetscCall(DMDAVecRestoreArrayRead(cdm, C, &x));
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

  PetscCall(PetscInitialize(&argc, &argv, NULL,help));
  PetscCall(DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, 11, 3, 1, NULL, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMDASetUniformCoordinates(dm, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0));
  PetscCall(DMCreateGlobalVector(dm, &X));

  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Dynamic Friction Options", "");
  {
    user.epsilon    = 0.1;
    user.gamma      = 0.5;
    user.gammaTilde = 0.5;
    user.xi         = 0.5;
    user.c          = 0.5;
    PetscCall(PetscOptionsReal("-epsilon", "Inverse of seismic ratio", "", user.epsilon, &user.epsilon, NULL));
    PetscCall(PetscOptionsReal("-gamma", "Wave frequency for interblock coupling", "", user.gamma, &user.gamma, NULL));
    PetscCall(PetscOptionsReal("-gamma_tilde", "Wave frequency for plate coupling", "", user.gammaTilde, &user.gammaTilde, NULL));
    PetscCall(PetscOptionsReal("-xi", "Interblock spring constant", "", user.xi, &user.xi, NULL));
    PetscCall(PetscOptionsReal("-c", "Wavespeed", "", user.c, &user.c, NULL));
  }
  PetscOptionsEnd();

  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetDM(ts, dm));
  PetscCall(TSSetRHSFunction(ts, NULL, FormRHSFunction, &user));
  PetscCall(TSSetIFunction(ts, NULL, FormIFunction, &user));
  PetscCall(DMSetMatType(dm, MATAIJ));
  PetscCall(DMCreateMatrix(dm, &J));
  PetscCall(TSSetIJacobian(ts, J, J, FormIJacobian, &user));

  ftime = 800.0;
  PetscCall(TSSetMaxTime(ts,ftime));
  PetscCall(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER));
  PetscCall(FormInitialSolution(ts, X, &user));
  PetscCall(TSSetSolution(ts, X));
  PetscCall(VecGetSize(X, &mx));
  hx   = 20.0/(PetscReal)(mx-1);
  dt   = 0.4 * PetscSqr(hx) / PetscSqr(user.c); /* Diffusive stability limit */
  PetscCall(TSSetTimeStep(ts,dt));
  PetscCall(TSSetFromOptions(ts));

  PetscCall(TSSolve(ts, X));
  PetscCall(TSGetSolveTime(ts, &ftime));
  PetscCall(TSGetStepNumber(ts, &steps));
  PetscCall(TSGetConvergedReason(ts, &reason));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%s at time %g after %" PetscInt_FMT " steps\n", TSConvergedReasons[reason], (double)ftime, steps));

  PetscCall(MatDestroy(&J));
  PetscCall(VecDestroy(&X));
  PetscCall(TSDestroy(&ts));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

    build:
      requires: !single  !complex

    test:
      TODO: broken, was not nightly tested, SNES solve eventually fails, -snes_test_jacobian indicates Jacobian is wrong, but even -snes_mf_operator fails

TEST*/
