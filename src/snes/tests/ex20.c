
static char help[] = "Nonlinear Radiative Transport PDE with multigrid in 3d.\n\
Uses 3-dimensional distributed arrays.\n\
A 3-dim simplified Radiative Transport test problem is used, with analytic Jacobian. \n\
\n\
  Solves the linear systems via multilevel methods \n\
\n\
The command line\n\
options are:\n\
  -tleft <tl>, where <tl> indicates the left Diriclet BC \n\
  -tright <tr>, where <tr> indicates the right Diriclet BC \n\
  -beta <beta>, where <beta> indicates the exponent in T \n\n";

/*

    This example models the partial differential equation

         - Div(alpha* T^beta (GRAD T)) = 0.

    where beta = 2.5 and alpha = 1.0

    BC: T_left = 1.0, T_right = 0.1, dT/dn_top = dTdn_bottom = dT/dn_up = dT/dn_down = 0.

    in the unit square, which is uniformly discretized in each of x and
    y in this simple encoding.  The degrees of freedom are cell centered.

    A finite volume approximation with the usual 7-point stencil
    is used to discretize the boundary value problem to obtain a
    nonlinear system of equations.

    This code was contributed by Nickolas Jovanovic based on ex18.c

*/

#include <petscsnes.h>
#include <petscdm.h>
#include <petscdmda.h>

/* User-defined application context */

typedef struct {
  PetscReal tleft, tright;   /* Dirichlet boundary conditions */
  PetscReal beta, bm1, coef; /* nonlinear diffusivity parameterizations */
} AppCtx;

#define POWFLOP 5 /* assume a pow() takes five flops */

extern PetscErrorCode FormInitialGuess(SNES, Vec, void *);
extern PetscErrorCode FormFunction(SNES, Vec, Vec, void *);
extern PetscErrorCode FormJacobian(SNES, Vec, Mat, Mat, void *);

int main(int argc, char **argv)
{
  SNES      snes;
  AppCtx    user;
  PetscInt  its, lits;
  PetscReal litspit;
  DM        da;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  /* set problem parameters */
  user.tleft  = 1.0;
  user.tright = 0.1;
  user.beta   = 2.5;
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-tleft", &user.tleft, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-tright", &user.tright, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-beta", &user.beta, NULL));
  user.bm1  = user.beta - 1.0;
  user.coef = user.beta / 2.0;

  /*
      Set the DMDA (grid structure) for the grids.
  */
  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, 5, 5, 5, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 1, 1, 0, 0, 0, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMSetApplicationContext(da, &user));

  /*
     Create the nonlinear solver
  */
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  PetscCall(SNESSetDM(snes, da));
  PetscCall(SNESSetFunction(snes, NULL, FormFunction, &user));
  PetscCall(SNESSetJacobian(snes, NULL, NULL, FormJacobian, &user));
  PetscCall(SNESSetFromOptions(snes));
  PetscCall(SNESSetComputeInitialGuess(snes, FormInitialGuess, NULL));

  PetscCall(SNESSolve(snes, NULL, NULL));
  PetscCall(SNESGetIterationNumber(snes, &its));
  PetscCall(SNESGetLinearSolveIterations(snes, &lits));
  litspit = ((PetscReal)lits) / ((PetscReal)its);
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Number of SNES iterations = %" PetscInt_FMT "\n", its));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Number of Linear iterations = %" PetscInt_FMT "\n", lits));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Average Linear its / SNES = %e\n", (double)litspit));

  PetscCall(SNESDestroy(&snes));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());
  return 0;
}
/* --------------------  Form initial approximation ----------------- */
PetscErrorCode FormInitialGuess(SNES snes, Vec X, void *ctx)
{
  AppCtx        *user;
  PetscInt       i, j, k, xs, ys, xm, ym, zs, zm;
  PetscScalar ***x;
  DM             da;

  PetscFunctionBeginUser;
  PetscCall(SNESGetDM(snes, &da));
  PetscCall(DMGetApplicationContext(da, &user));
  PetscCall(DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm));
  PetscCall(DMDAVecGetArray(da, X, &x));

  /* Compute initial guess */
  for (k = zs; k < zs + zm; k++) {
    for (j = ys; j < ys + ym; j++) {
      for (i = xs; i < xs + xm; i++) x[k][j][i] = user->tleft;
    }
  }
  PetscCall(DMDAVecRestoreArray(da, X, &x));
  PetscFunctionReturn(PETSC_SUCCESS);
}
/* --------------------  Evaluate Function F(x) --------------------- */
PetscErrorCode FormFunction(SNES snes, Vec X, Vec F, void *ptr)
{
  AppCtx        *user = (AppCtx *)ptr;
  PetscInt       i, j, k, mx, my, mz, xs, ys, zs, xm, ym, zm;
  PetscScalar    zero = 0.0, one = 1.0;
  PetscScalar    hx, hy, hz, hxhydhz, hyhzdhx, hzhxdhy;
  PetscScalar    t0, tn, ts, te, tw, an, as, ae, aw, dn, ds, de, dw, fn = 0.0, fs = 0.0, fe = 0.0, fw = 0.0;
  PetscScalar    tleft, tright, beta, td, ad, dd, fd = 0.0, tu, au, du = 0.0, fu = 0.0;
  PetscScalar ***x, ***f;
  Vec            localX;
  DM             da;

  PetscFunctionBeginUser;
  PetscCall(SNESGetDM(snes, &da));
  PetscCall(DMGetLocalVector(da, &localX));
  PetscCall(DMDAGetInfo(da, NULL, &mx, &my, &mz, 0, 0, 0, 0, 0, 0, 0, 0, 0));
  hx      = one / (PetscReal)(mx - 1);
  hy      = one / (PetscReal)(my - 1);
  hz      = one / (PetscReal)(mz - 1);
  hxhydhz = hx * hy / hz;
  hyhzdhx = hy * hz / hx;
  hzhxdhy = hz * hx / hy;
  tleft   = user->tleft;
  tright  = user->tright;
  beta    = user->beta;

  /* Get ghost points */
  PetscCall(DMGlobalToLocalBegin(da, X, INSERT_VALUES, localX));
  PetscCall(DMGlobalToLocalEnd(da, X, INSERT_VALUES, localX));
  PetscCall(DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm));
  PetscCall(DMDAVecGetArray(da, localX, &x));
  PetscCall(DMDAVecGetArray(da, F, &f));

  /* Evaluate function */
  for (k = zs; k < zs + zm; k++) {
    for (j = ys; j < ys + ym; j++) {
      for (i = xs; i < xs + xm; i++) {
        t0 = x[k][j][i];

        if (i > 0 && i < mx - 1 && j > 0 && j < my - 1 && k > 0 && k < mz - 1) {
          /* general interior volume */

          tw = x[k][j][i - 1];
          aw = 0.5 * (t0 + tw);
          dw = PetscPowScalar(aw, beta);
          fw = dw * (t0 - tw);

          te = x[k][j][i + 1];
          ae = 0.5 * (t0 + te);
          de = PetscPowScalar(ae, beta);
          fe = de * (te - t0);

          ts = x[k][j - 1][i];
          as = 0.5 * (t0 + ts);
          ds = PetscPowScalar(as, beta);
          fs = ds * (t0 - ts);

          tn = x[k][j + 1][i];
          an = 0.5 * (t0 + tn);
          dn = PetscPowScalar(an, beta);
          fn = dn * (tn - t0);

          td = x[k - 1][j][i];
          ad = 0.5 * (t0 + td);
          dd = PetscPowScalar(ad, beta);
          fd = dd * (t0 - td);

          tu = x[k + 1][j][i];
          au = 0.5 * (t0 + tu);
          du = PetscPowScalar(au, beta);
          fu = du * (tu - t0);

        } else if (i == 0) {
          /* left-hand (west) boundary */
          tw = tleft;
          aw = 0.5 * (t0 + tw);
          dw = PetscPowScalar(aw, beta);
          fw = dw * (t0 - tw);

          te = x[k][j][i + 1];
          ae = 0.5 * (t0 + te);
          de = PetscPowScalar(ae, beta);
          fe = de * (te - t0);

          if (j > 0) {
            ts = x[k][j - 1][i];
            as = 0.5 * (t0 + ts);
            ds = PetscPowScalar(as, beta);
            fs = ds * (t0 - ts);
          } else {
            fs = zero;
          }

          if (j < my - 1) {
            tn = x[k][j + 1][i];
            an = 0.5 * (t0 + tn);
            dn = PetscPowScalar(an, beta);
            fn = dn * (tn - t0);
          } else {
            fn = zero;
          }

          if (k > 0) {
            td = x[k - 1][j][i];
            ad = 0.5 * (t0 + td);
            dd = PetscPowScalar(ad, beta);
            fd = dd * (t0 - td);
          } else {
            fd = zero;
          }

          if (k < mz - 1) {
            tu = x[k + 1][j][i];
            au = 0.5 * (t0 + tu);
            du = PetscPowScalar(au, beta);
            fu = du * (tu - t0);
          } else {
            fu = zero;
          }

        } else if (i == mx - 1) {
          /* right-hand (east) boundary */
          tw = x[k][j][i - 1];
          aw = 0.5 * (t0 + tw);
          dw = PetscPowScalar(aw, beta);
          fw = dw * (t0 - tw);

          te = tright;
          ae = 0.5 * (t0 + te);
          de = PetscPowScalar(ae, beta);
          fe = de * (te - t0);

          if (j > 0) {
            ts = x[k][j - 1][i];
            as = 0.5 * (t0 + ts);
            ds = PetscPowScalar(as, beta);
            fs = ds * (t0 - ts);
          } else {
            fs = zero;
          }

          if (j < my - 1) {
            tn = x[k][j + 1][i];
            an = 0.5 * (t0 + tn);
            dn = PetscPowScalar(an, beta);
            fn = dn * (tn - t0);
          } else {
            fn = zero;
          }

          if (k > 0) {
            td = x[k - 1][j][i];
            ad = 0.5 * (t0 + td);
            dd = PetscPowScalar(ad, beta);
            fd = dd * (t0 - td);
          } else {
            fd = zero;
          }

          if (k < mz - 1) {
            tu = x[k + 1][j][i];
            au = 0.5 * (t0 + tu);
            du = PetscPowScalar(au, beta);
            fu = du * (tu - t0);
          } else {
            fu = zero;
          }

        } else if (j == 0) {
          /* bottom (south) boundary, and i <> 0 or mx-1 */
          tw = x[k][j][i - 1];
          aw = 0.5 * (t0 + tw);
          dw = PetscPowScalar(aw, beta);
          fw = dw * (t0 - tw);

          te = x[k][j][i + 1];
          ae = 0.5 * (t0 + te);
          de = PetscPowScalar(ae, beta);
          fe = de * (te - t0);

          fs = zero;

          tn = x[k][j + 1][i];
          an = 0.5 * (t0 + tn);
          dn = PetscPowScalar(an, beta);
          fn = dn * (tn - t0);

          if (k > 0) {
            td = x[k - 1][j][i];
            ad = 0.5 * (t0 + td);
            dd = PetscPowScalar(ad, beta);
            fd = dd * (t0 - td);
          } else {
            fd = zero;
          }

          if (k < mz - 1) {
            tu = x[k + 1][j][i];
            au = 0.5 * (t0 + tu);
            du = PetscPowScalar(au, beta);
            fu = du * (tu - t0);
          } else {
            fu = zero;
          }

        } else if (j == my - 1) {
          /* top (north) boundary, and i <> 0 or mx-1 */
          tw = x[k][j][i - 1];
          aw = 0.5 * (t0 + tw);
          dw = PetscPowScalar(aw, beta);
          fw = dw * (t0 - tw);

          te = x[k][j][i + 1];
          ae = 0.5 * (t0 + te);
          de = PetscPowScalar(ae, beta);
          fe = de * (te - t0);

          ts = x[k][j - 1][i];
          as = 0.5 * (t0 + ts);
          ds = PetscPowScalar(as, beta);
          fs = ds * (t0 - ts);

          fn = zero;

          if (k > 0) {
            td = x[k - 1][j][i];
            ad = 0.5 * (t0 + td);
            dd = PetscPowScalar(ad, beta);
            fd = dd * (t0 - td);
          } else {
            fd = zero;
          }

          if (k < mz - 1) {
            tu = x[k + 1][j][i];
            au = 0.5 * (t0 + tu);
            du = PetscPowScalar(au, beta);
            fu = du * (tu - t0);
          } else {
            fu = zero;
          }

        } else if (k == 0) {
          /* down boundary (interior only) */
          tw = x[k][j][i - 1];
          aw = 0.5 * (t0 + tw);
          dw = PetscPowScalar(aw, beta);
          fw = dw * (t0 - tw);

          te = x[k][j][i + 1];
          ae = 0.5 * (t0 + te);
          de = PetscPowScalar(ae, beta);
          fe = de * (te - t0);

          ts = x[k][j - 1][i];
          as = 0.5 * (t0 + ts);
          ds = PetscPowScalar(as, beta);
          fs = ds * (t0 - ts);

          tn = x[k][j + 1][i];
          an = 0.5 * (t0 + tn);
          dn = PetscPowScalar(an, beta);
          fn = dn * (tn - t0);

          fd = zero;

          tu = x[k + 1][j][i];
          au = 0.5 * (t0 + tu);
          du = PetscPowScalar(au, beta);
          fu = du * (tu - t0);

        } else if (k == mz - 1) {
          /* up boundary (interior only) */
          tw = x[k][j][i - 1];
          aw = 0.5 * (t0 + tw);
          dw = PetscPowScalar(aw, beta);
          fw = dw * (t0 - tw);

          te = x[k][j][i + 1];
          ae = 0.5 * (t0 + te);
          de = PetscPowScalar(ae, beta);
          fe = de * (te - t0);

          ts = x[k][j - 1][i];
          as = 0.5 * (t0 + ts);
          ds = PetscPowScalar(as, beta);
          fs = ds * (t0 - ts);

          tn = x[k][j + 1][i];
          an = 0.5 * (t0 + tn);
          dn = PetscPowScalar(an, beta);
          fn = dn * (tn - t0);

          td = x[k - 1][j][i];
          ad = 0.5 * (t0 + td);
          dd = PetscPowScalar(ad, beta);
          fd = dd * (t0 - td);

          fu = zero;
        }

        f[k][j][i] = -hyhzdhx * (fe - fw) - hzhxdhy * (fn - fs) - hxhydhz * (fu - fd);
      }
    }
  }
  PetscCall(DMDAVecRestoreArray(da, localX, &x));
  PetscCall(DMDAVecRestoreArray(da, F, &f));
  PetscCall(DMRestoreLocalVector(da, &localX));
  PetscCall(PetscLogFlops((22.0 + 4.0 * POWFLOP) * ym * xm));
  PetscFunctionReturn(PETSC_SUCCESS);
}
/* --------------------  Evaluate Jacobian F(x) --------------------- */
PetscErrorCode FormJacobian(SNES snes, Vec X, Mat J, Mat jac, void *ptr)
{
  AppCtx        *user = (AppCtx *)ptr;
  PetscInt       i, j, k, mx, my, mz, xs, ys, zs, xm, ym, zm;
  PetscScalar    one = 1.0;
  PetscScalar    hx, hy, hz, hxhydhz, hyhzdhx, hzhxdhy;
  PetscScalar    t0, tn, ts, te, tw, an, as, ae, aw, dn, ds, de, dw;
  PetscScalar    tleft, tright, beta, td, ad, dd, tu, au, du, v[7], bm1, coef;
  PetscScalar ***x, bn, bs, be, bw, bu, bd, gn, gs, ge, gw, gu, gd;
  Vec            localX;
  MatStencil     c[7], row;
  DM             da;

  PetscFunctionBeginUser;
  PetscCall(SNESGetDM(snes, &da));
  PetscCall(DMGetLocalVector(da, &localX));
  PetscCall(DMDAGetInfo(da, NULL, &mx, &my, &mz, 0, 0, 0, 0, 0, 0, 0, 0, 0));
  hx      = one / (PetscReal)(mx - 1);
  hy      = one / (PetscReal)(my - 1);
  hz      = one / (PetscReal)(mz - 1);
  hxhydhz = hx * hy / hz;
  hyhzdhx = hy * hz / hx;
  hzhxdhy = hz * hx / hy;
  tleft   = user->tleft;
  tright  = user->tright;
  beta    = user->beta;
  bm1     = user->bm1;
  coef    = user->coef;

  /* Get ghost points */
  PetscCall(DMGlobalToLocalBegin(da, X, INSERT_VALUES, localX));
  PetscCall(DMGlobalToLocalEnd(da, X, INSERT_VALUES, localX));
  PetscCall(DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm));
  PetscCall(DMDAVecGetArray(da, localX, &x));

  /* Evaluate Jacobian of function */
  for (k = zs; k < zs + zm; k++) {
    for (j = ys; j < ys + ym; j++) {
      for (i = xs; i < xs + xm; i++) {
        t0    = x[k][j][i];
        row.k = k;
        row.j = j;
        row.i = i;
        if (i > 0 && i < mx - 1 && j > 0 && j < my - 1 && k > 0 && k < mz - 1) {
          /* general interior volume */

          tw = x[k][j][i - 1];
          aw = 0.5 * (t0 + tw);
          bw = PetscPowScalar(aw, bm1);
          /* dw = bw * aw */
          dw = PetscPowScalar(aw, beta);
          gw = coef * bw * (t0 - tw);

          te = x[k][j][i + 1];
          ae = 0.5 * (t0 + te);
          be = PetscPowScalar(ae, bm1);
          /* de = be * ae; */
          de = PetscPowScalar(ae, beta);
          ge = coef * be * (te - t0);

          ts = x[k][j - 1][i];
          as = 0.5 * (t0 + ts);
          bs = PetscPowScalar(as, bm1);
          /* ds = bs * as; */
          ds = PetscPowScalar(as, beta);
          gs = coef * bs * (t0 - ts);

          tn = x[k][j + 1][i];
          an = 0.5 * (t0 + tn);
          bn = PetscPowScalar(an, bm1);
          /* dn = bn * an; */
          dn = PetscPowScalar(an, beta);
          gn = coef * bn * (tn - t0);

          td = x[k - 1][j][i];
          ad = 0.5 * (t0 + td);
          bd = PetscPowScalar(ad, bm1);
          /* dd = bd * ad; */
          dd = PetscPowScalar(ad, beta);
          gd = coef * bd * (t0 - td);

          tu = x[k + 1][j][i];
          au = 0.5 * (t0 + tu);
          bu = PetscPowScalar(au, bm1);
          /* du = bu * au; */
          du = PetscPowScalar(au, beta);
          gu = coef * bu * (tu - t0);

          c[0].k = k - 1;
          c[0].j = j;
          c[0].i = i;
          v[0]   = -hxhydhz * (dd - gd);
          c[1].k = k;
          c[1].j = j - 1;
          c[1].i = i;
          v[1]   = -hzhxdhy * (ds - gs);
          c[2].k = k;
          c[2].j = j;
          c[2].i = i - 1;
          v[2]   = -hyhzdhx * (dw - gw);
          c[3].k = k;
          c[3].j = j;
          c[3].i = i;
          v[3]   = hzhxdhy * (ds + dn + gs - gn) + hyhzdhx * (dw + de + gw - ge) + hxhydhz * (dd + du + gd - gu);
          c[4].k = k;
          c[4].j = j;
          c[4].i = i + 1;
          v[4]   = -hyhzdhx * (de + ge);
          c[5].k = k;
          c[5].j = j + 1;
          c[5].i = i;
          v[5]   = -hzhxdhy * (dn + gn);
          c[6].k = k + 1;
          c[6].j = j;
          c[6].i = i;
          v[6]   = -hxhydhz * (du + gu);
          PetscCall(MatSetValuesStencil(jac, 1, &row, 7, c, v, INSERT_VALUES));

        } else if (i == 0) {
          /* left-hand plane boundary */
          tw = tleft;
          aw = 0.5 * (t0 + tw);
          bw = PetscPowScalar(aw, bm1);
          /* dw = bw * aw */
          dw = PetscPowScalar(aw, beta);
          gw = coef * bw * (t0 - tw);

          te = x[k][j][i + 1];
          ae = 0.5 * (t0 + te);
          be = PetscPowScalar(ae, bm1);
          /* de = be * ae; */
          de = PetscPowScalar(ae, beta);
          ge = coef * be * (te - t0);

          /* left-hand bottom edge */
          if (j == 0) {
            tn = x[k][j + 1][i];
            an = 0.5 * (t0 + tn);
            bn = PetscPowScalar(an, bm1);
            /* dn = bn * an; */
            dn = PetscPowScalar(an, beta);
            gn = coef * bn * (tn - t0);

            /* left-hand bottom down corner */
            if (k == 0) {
              tu = x[k + 1][j][i];
              au = 0.5 * (t0 + tu);
              bu = PetscPowScalar(au, bm1);
              /* du = bu * au; */
              du = PetscPowScalar(au, beta);
              gu = coef * bu * (tu - t0);

              c[0].k = k;
              c[0].j = j;
              c[0].i = i;
              v[0]   = hzhxdhy * (dn - gn) + hyhzdhx * (dw + de + gw - ge) + hxhydhz * (du - gu);
              c[1].k = k;
              c[1].j = j;
              c[1].i = i + 1;
              v[1]   = -hyhzdhx * (de + ge);
              c[2].k = k;
              c[2].j = j + 1;
              c[2].i = i;
              v[2]   = -hzhxdhy * (dn + gn);
              c[3].k = k + 1;
              c[3].j = j;
              c[3].i = i;
              v[3]   = -hxhydhz * (du + gu);
              PetscCall(MatSetValuesStencil(jac, 1, &row, 4, c, v, INSERT_VALUES));

              /* left-hand bottom interior edge */
            } else if (k < mz - 1) {
              tu = x[k + 1][j][i];
              au = 0.5 * (t0 + tu);
              bu = PetscPowScalar(au, bm1);
              /* du = bu * au; */
              du = PetscPowScalar(au, beta);
              gu = coef * bu * (tu - t0);

              td = x[k - 1][j][i];
              ad = 0.5 * (t0 + td);
              bd = PetscPowScalar(ad, bm1);
              /* dd = bd * ad; */
              dd = PetscPowScalar(ad, beta);
              gd = coef * bd * (td - t0);

              c[0].k = k - 1;
              c[0].j = j;
              c[0].i = i;
              v[0]   = -hxhydhz * (dd - gd);
              c[1].k = k;
              c[1].j = j;
              c[1].i = i;
              v[1]   = hzhxdhy * (dn - gn) + hyhzdhx * (dw + de + gw - ge) + hxhydhz * (dd + du + gd - gu);
              c[2].k = k;
              c[2].j = j;
              c[2].i = i + 1;
              v[2]   = -hyhzdhx * (de + ge);
              c[3].k = k;
              c[3].j = j + 1;
              c[3].i = i;
              v[3]   = -hzhxdhy * (dn + gn);
              c[4].k = k + 1;
              c[4].j = j;
              c[4].i = i;
              v[4]   = -hxhydhz * (du + gu);
              PetscCall(MatSetValuesStencil(jac, 1, &row, 5, c, v, INSERT_VALUES));

              /* left-hand bottom up corner */
            } else {
              td = x[k - 1][j][i];
              ad = 0.5 * (t0 + td);
              bd = PetscPowScalar(ad, bm1);
              /* dd = bd * ad; */
              dd = PetscPowScalar(ad, beta);
              gd = coef * bd * (td - t0);

              c[0].k = k - 1;
              c[0].j = j;
              c[0].i = i;
              v[0]   = -hxhydhz * (dd - gd);
              c[1].k = k;
              c[1].j = j;
              c[1].i = i;
              v[1]   = hzhxdhy * (dn - gn) + hyhzdhx * (dw + de + gw - ge) + hxhydhz * (dd + gd);
              c[2].k = k;
              c[2].j = j;
              c[2].i = i + 1;
              v[2]   = -hyhzdhx * (de + ge);
              c[3].k = k;
              c[3].j = j + 1;
              c[3].i = i;
              v[3]   = -hzhxdhy * (dn + gn);
              PetscCall(MatSetValuesStencil(jac, 1, &row, 4, c, v, INSERT_VALUES));
            }

            /* left-hand top edge */
          } else if (j == my - 1) {
            ts = x[k][j - 1][i];
            as = 0.5 * (t0 + ts);
            bs = PetscPowScalar(as, bm1);
            /* ds = bs * as; */
            ds = PetscPowScalar(as, beta);
            gs = coef * bs * (ts - t0);

            /* left-hand top down corner */
            if (k == 0) {
              tu = x[k + 1][j][i];
              au = 0.5 * (t0 + tu);
              bu = PetscPowScalar(au, bm1);
              /* du = bu * au; */
              du = PetscPowScalar(au, beta);
              gu = coef * bu * (tu - t0);

              c[0].k = k;
              c[0].j = j - 1;
              c[0].i = i;
              v[0]   = -hzhxdhy * (ds - gs);
              c[1].k = k;
              c[1].j = j;
              c[1].i = i;
              v[1]   = hzhxdhy * (ds + gs) + hyhzdhx * (dw + de + gw - ge) + hxhydhz * (du - gu);
              c[2].k = k;
              c[2].j = j;
              c[2].i = i + 1;
              v[2]   = -hyhzdhx * (de + ge);
              c[3].k = k + 1;
              c[3].j = j;
              c[3].i = i;
              v[3]   = -hxhydhz * (du + gu);
              PetscCall(MatSetValuesStencil(jac, 1, &row, 4, c, v, INSERT_VALUES));

              /* left-hand top interior edge */
            } else if (k < mz - 1) {
              tu = x[k + 1][j][i];
              au = 0.5 * (t0 + tu);
              bu = PetscPowScalar(au, bm1);
              /* du = bu * au; */
              du = PetscPowScalar(au, beta);
              gu = coef * bu * (tu - t0);

              td = x[k - 1][j][i];
              ad = 0.5 * (t0 + td);
              bd = PetscPowScalar(ad, bm1);
              /* dd = bd * ad; */
              dd = PetscPowScalar(ad, beta);
              gd = coef * bd * (td - t0);

              c[0].k = k - 1;
              c[0].j = j;
              c[0].i = i;
              v[0]   = -hxhydhz * (dd - gd);
              c[1].k = k;
              c[1].j = j - 1;
              c[1].i = i;
              v[1]   = -hzhxdhy * (ds - gs);
              c[2].k = k;
              c[2].j = j;
              c[2].i = i;
              v[2]   = hzhxdhy * (ds + gs) + hyhzdhx * (dw + de + gw - ge) + hxhydhz * (dd + du + gd - gu);
              c[3].k = k;
              c[3].j = j;
              c[3].i = i + 1;
              v[3]   = -hyhzdhx * (de + ge);
              c[4].k = k + 1;
              c[4].j = j;
              c[4].i = i;
              v[4]   = -hxhydhz * (du + gu);
              PetscCall(MatSetValuesStencil(jac, 1, &row, 5, c, v, INSERT_VALUES));

              /* left-hand top up corner */
            } else {
              td = x[k - 1][j][i];
              ad = 0.5 * (t0 + td);
              bd = PetscPowScalar(ad, bm1);
              /* dd = bd * ad; */
              dd = PetscPowScalar(ad, beta);
              gd = coef * bd * (td - t0);

              c[0].k = k - 1;
              c[0].j = j;
              c[0].i = i;
              v[0]   = -hxhydhz * (dd - gd);
              c[1].k = k;
              c[1].j = j - 1;
              c[1].i = i;
              v[1]   = -hzhxdhy * (ds - gs);
              c[2].k = k;
              c[2].j = j;
              c[2].i = i;
              v[2]   = hzhxdhy * (ds + gs) + hyhzdhx * (dw + de + gw - ge) + hxhydhz * (dd + gd);
              c[3].k = k;
              c[3].j = j;
              c[3].i = i + 1;
              v[3]   = -hyhzdhx * (de + ge);
              PetscCall(MatSetValuesStencil(jac, 1, &row, 4, c, v, INSERT_VALUES));
            }

          } else {
            ts = x[k][j - 1][i];
            as = 0.5 * (t0 + ts);
            bs = PetscPowScalar(as, bm1);
            /* ds = bs * as; */
            ds = PetscPowScalar(as, beta);
            gs = coef * bs * (t0 - ts);

            tn = x[k][j + 1][i];
            an = 0.5 * (t0 + tn);
            bn = PetscPowScalar(an, bm1);
            /* dn = bn * an; */
            dn = PetscPowScalar(an, beta);
            gn = coef * bn * (tn - t0);

            /* left-hand down interior edge */
            if (k == 0) {
              tu = x[k + 1][j][i];
              au = 0.5 * (t0 + tu);
              bu = PetscPowScalar(au, bm1);
              /* du = bu * au; */
              du = PetscPowScalar(au, beta);
              gu = coef * bu * (tu - t0);

              c[0].k = k;
              c[0].j = j - 1;
              c[0].i = i;
              v[0]   = -hzhxdhy * (ds - gs);
              c[1].k = k;
              c[1].j = j;
              c[1].i = i;
              v[1]   = hzhxdhy * (ds + dn + gs - gn) + hyhzdhx * (dw + de + gw - ge) + hxhydhz * (du - gu);
              c[2].k = k;
              c[2].j = j;
              c[2].i = i + 1;
              v[2]   = -hyhzdhx * (de + ge);
              c[3].k = k;
              c[3].j = j + 1;
              c[3].i = i;
              v[3]   = -hzhxdhy * (dn + gn);
              c[4].k = k + 1;
              c[4].j = j;
              c[4].i = i;
              v[4]   = -hxhydhz * (du + gu);
              PetscCall(MatSetValuesStencil(jac, 1, &row, 5, c, v, INSERT_VALUES));

            } else if (k == mz - 1) { /* left-hand up interior edge */

              td = x[k - 1][j][i];
              ad = 0.5 * (t0 + td);
              bd = PetscPowScalar(ad, bm1);
              /* dd = bd * ad; */
              dd = PetscPowScalar(ad, beta);
              gd = coef * bd * (t0 - td);

              c[0].k = k - 1;
              c[0].j = j;
              c[0].i = i;
              v[0]   = -hxhydhz * (dd - gd);
              c[1].k = k;
              c[1].j = j - 1;
              c[1].i = i;
              v[1]   = -hzhxdhy * (ds - gs);
              c[2].k = k;
              c[2].j = j;
              c[2].i = i;
              v[2]   = hzhxdhy * (ds + dn + gs - gn) + hyhzdhx * (dw + de + gw - ge) + hxhydhz * (dd + gd);
              c[3].k = k;
              c[3].j = j;
              c[3].i = i + 1;
              v[3]   = -hyhzdhx * (de + ge);
              c[4].k = k;
              c[4].j = j + 1;
              c[4].i = i;
              v[4]   = -hzhxdhy * (dn + gn);
              PetscCall(MatSetValuesStencil(jac, 1, &row, 5, c, v, INSERT_VALUES));
            } else { /* left-hand interior plane */

              td = x[k - 1][j][i];
              ad = 0.5 * (t0 + td);
              bd = PetscPowScalar(ad, bm1);
              /* dd = bd * ad; */
              dd = PetscPowScalar(ad, beta);
              gd = coef * bd * (t0 - td);

              tu = x[k + 1][j][i];
              au = 0.5 * (t0 + tu);
              bu = PetscPowScalar(au, bm1);
              /* du = bu * au; */
              du = PetscPowScalar(au, beta);
              gu = coef * bu * (tu - t0);

              c[0].k = k - 1;
              c[0].j = j;
              c[0].i = i;
              v[0]   = -hxhydhz * (dd - gd);
              c[1].k = k;
              c[1].j = j - 1;
              c[1].i = i;
              v[1]   = -hzhxdhy * (ds - gs);
              c[2].k = k;
              c[2].j = j;
              c[2].i = i;
              v[2]   = hzhxdhy * (ds + dn + gs - gn) + hyhzdhx * (dw + de + gw - ge) + hxhydhz * (dd + du + gd - gu);
              c[3].k = k;
              c[3].j = j;
              c[3].i = i + 1;
              v[3]   = -hyhzdhx * (de + ge);
              c[4].k = k;
              c[4].j = j + 1;
              c[4].i = i;
              v[4]   = -hzhxdhy * (dn + gn);
              c[5].k = k + 1;
              c[5].j = j;
              c[5].i = i;
              v[5]   = -hxhydhz * (du + gu);
              PetscCall(MatSetValuesStencil(jac, 1, &row, 6, c, v, INSERT_VALUES));
            }
          }

        } else if (i == mx - 1) {
          /* right-hand plane boundary */
          tw = x[k][j][i - 1];
          aw = 0.5 * (t0 + tw);
          bw = PetscPowScalar(aw, bm1);
          /* dw = bw * aw */
          dw = PetscPowScalar(aw, beta);
          gw = coef * bw * (t0 - tw);

          te = tright;
          ae = 0.5 * (t0 + te);
          be = PetscPowScalar(ae, bm1);
          /* de = be * ae; */
          de = PetscPowScalar(ae, beta);
          ge = coef * be * (te - t0);

          /* right-hand bottom edge */
          if (j == 0) {
            tn = x[k][j + 1][i];
            an = 0.5 * (t0 + tn);
            bn = PetscPowScalar(an, bm1);
            /* dn = bn * an; */
            dn = PetscPowScalar(an, beta);
            gn = coef * bn * (tn - t0);

            /* right-hand bottom down corner */
            if (k == 0) {
              tu = x[k + 1][j][i];
              au = 0.5 * (t0 + tu);
              bu = PetscPowScalar(au, bm1);
              /* du = bu * au; */
              du = PetscPowScalar(au, beta);
              gu = coef * bu * (tu - t0);

              c[0].k = k;
              c[0].j = j;
              c[0].i = i - 1;
              v[0]   = -hyhzdhx * (dw - gw);
              c[1].k = k;
              c[1].j = j;
              c[1].i = i;
              v[1]   = hzhxdhy * (dn - gn) + hyhzdhx * (dw + de + gw - ge) + hxhydhz * (du - gu);
              c[2].k = k;
              c[2].j = j + 1;
              c[2].i = i;
              v[2]   = -hzhxdhy * (dn + gn);
              c[3].k = k + 1;
              c[3].j = j;
              c[3].i = i;
              v[3]   = -hxhydhz * (du + gu);
              PetscCall(MatSetValuesStencil(jac, 1, &row, 4, c, v, INSERT_VALUES));

              /* right-hand bottom interior edge */
            } else if (k < mz - 1) {
              tu = x[k + 1][j][i];
              au = 0.5 * (t0 + tu);
              bu = PetscPowScalar(au, bm1);
              /* du = bu * au; */
              du = PetscPowScalar(au, beta);
              gu = coef * bu * (tu - t0);

              td = x[k - 1][j][i];
              ad = 0.5 * (t0 + td);
              bd = PetscPowScalar(ad, bm1);
              /* dd = bd * ad; */
              dd = PetscPowScalar(ad, beta);
              gd = coef * bd * (td - t0);

              c[0].k = k - 1;
              c[0].j = j;
              c[0].i = i;
              v[0]   = -hxhydhz * (dd - gd);
              c[1].k = k;
              c[1].j = j;
              c[1].i = i - 1;
              v[1]   = -hyhzdhx * (dw - gw);
              c[2].k = k;
              c[2].j = j;
              c[2].i = i;
              v[2]   = hzhxdhy * (dn - gn) + hyhzdhx * (dw + de + gw - ge) + hxhydhz * (dd + du + gd - gu);
              c[3].k = k;
              c[3].j = j + 1;
              c[3].i = i;
              v[3]   = -hzhxdhy * (dn + gn);
              c[4].k = k + 1;
              c[4].j = j;
              c[4].i = i;
              v[4]   = -hxhydhz * (du + gu);
              PetscCall(MatSetValuesStencil(jac, 1, &row, 5, c, v, INSERT_VALUES));

              /* right-hand bottom up corner */
            } else {
              td = x[k - 1][j][i];
              ad = 0.5 * (t0 + td);
              bd = PetscPowScalar(ad, bm1);
              /* dd = bd * ad; */
              dd = PetscPowScalar(ad, beta);
              gd = coef * bd * (td - t0);

              c[0].k = k - 1;
              c[0].j = j;
              c[0].i = i;
              v[0]   = -hxhydhz * (dd - gd);
              c[1].k = k;
              c[1].j = j;
              c[1].i = i - 1;
              v[1]   = -hyhzdhx * (dw - gw);
              c[2].k = k;
              c[2].j = j;
              c[2].i = i;
              v[2]   = hzhxdhy * (dn - gn) + hyhzdhx * (dw + de + gw - ge) + hxhydhz * (dd + gd);
              c[3].k = k;
              c[3].j = j + 1;
              c[3].i = i;
              v[3]   = -hzhxdhy * (dn + gn);
              PetscCall(MatSetValuesStencil(jac, 1, &row, 4, c, v, INSERT_VALUES));
            }

            /* right-hand top edge */
          } else if (j == my - 1) {
            ts = x[k][j - 1][i];
            as = 0.5 * (t0 + ts);
            bs = PetscPowScalar(as, bm1);
            /* ds = bs * as; */
            ds = PetscPowScalar(as, beta);
            gs = coef * bs * (ts - t0);

            /* right-hand top down corner */
            if (k == 0) {
              tu = x[k + 1][j][i];
              au = 0.5 * (t0 + tu);
              bu = PetscPowScalar(au, bm1);
              /* du = bu * au; */
              du = PetscPowScalar(au, beta);
              gu = coef * bu * (tu - t0);

              c[0].k = k;
              c[0].j = j - 1;
              c[0].i = i;
              v[0]   = -hzhxdhy * (ds - gs);
              c[1].k = k;
              c[1].j = j;
              c[1].i = i - 1;
              v[1]   = -hyhzdhx * (dw - gw);
              c[2].k = k;
              c[2].j = j;
              c[2].i = i;
              v[2]   = hzhxdhy * (ds + gs) + hyhzdhx * (dw + de + gw - ge) + hxhydhz * (du - gu);
              c[3].k = k + 1;
              c[3].j = j;
              c[3].i = i;
              v[3]   = -hxhydhz * (du + gu);
              PetscCall(MatSetValuesStencil(jac, 1, &row, 4, c, v, INSERT_VALUES));

              /* right-hand top interior edge */
            } else if (k < mz - 1) {
              tu = x[k + 1][j][i];
              au = 0.5 * (t0 + tu);
              bu = PetscPowScalar(au, bm1);
              /* du = bu * au; */
              du = PetscPowScalar(au, beta);
              gu = coef * bu * (tu - t0);

              td = x[k - 1][j][i];
              ad = 0.5 * (t0 + td);
              bd = PetscPowScalar(ad, bm1);
              /* dd = bd * ad; */
              dd = PetscPowScalar(ad, beta);
              gd = coef * bd * (td - t0);

              c[0].k = k - 1;
              c[0].j = j;
              c[0].i = i;
              v[0]   = -hxhydhz * (dd - gd);
              c[1].k = k;
              c[1].j = j - 1;
              c[1].i = i;
              v[1]   = -hzhxdhy * (ds - gs);
              c[2].k = k;
              c[2].j = j;
              c[2].i = i - 1;
              v[2]   = -hyhzdhx * (dw - gw);
              c[3].k = k;
              c[3].j = j;
              c[3].i = i;
              v[3]   = hzhxdhy * (ds + gs) + hyhzdhx * (dw + de + gw - ge) + hxhydhz * (dd + du + gd - gu);
              c[4].k = k + 1;
              c[4].j = j;
              c[4].i = i;
              v[4]   = -hxhydhz * (du + gu);
              PetscCall(MatSetValuesStencil(jac, 1, &row, 5, c, v, INSERT_VALUES));

              /* right-hand top up corner */
            } else {
              td = x[k - 1][j][i];
              ad = 0.5 * (t0 + td);
              bd = PetscPowScalar(ad, bm1);
              /* dd = bd * ad; */
              dd = PetscPowScalar(ad, beta);
              gd = coef * bd * (td - t0);

              c[0].k = k - 1;
              c[0].j = j;
              c[0].i = i;
              v[0]   = -hxhydhz * (dd - gd);
              c[1].k = k;
              c[1].j = j - 1;
              c[1].i = i;
              v[1]   = -hzhxdhy * (ds - gs);
              c[2].k = k;
              c[2].j = j;
              c[2].i = i - 1;
              v[2]   = -hyhzdhx * (dw - gw);
              c[3].k = k;
              c[3].j = j;
              c[3].i = i;
              v[3]   = hzhxdhy * (ds + gs) + hyhzdhx * (dw + de + gw - ge) + hxhydhz * (dd + gd);
              PetscCall(MatSetValuesStencil(jac, 1, &row, 4, c, v, INSERT_VALUES));
            }

          } else {
            ts = x[k][j - 1][i];
            as = 0.5 * (t0 + ts);
            bs = PetscPowScalar(as, bm1);
            /* ds = bs * as; */
            ds = PetscPowScalar(as, beta);
            gs = coef * bs * (t0 - ts);

            tn = x[k][j + 1][i];
            an = 0.5 * (t0 + tn);
            bn = PetscPowScalar(an, bm1);
            /* dn = bn * an; */
            dn = PetscPowScalar(an, beta);
            gn = coef * bn * (tn - t0);

            /* right-hand down interior edge */
            if (k == 0) {
              tu = x[k + 1][j][i];
              au = 0.5 * (t0 + tu);
              bu = PetscPowScalar(au, bm1);
              /* du = bu * au; */
              du = PetscPowScalar(au, beta);
              gu = coef * bu * (tu - t0);

              c[0].k = k;
              c[0].j = j - 1;
              c[0].i = i;
              v[0]   = -hzhxdhy * (ds - gs);
              c[1].k = k;
              c[1].j = j;
              c[1].i = i - 1;
              v[1]   = -hyhzdhx * (dw - gw);
              c[2].k = k;
              c[2].j = j;
              c[2].i = i;
              v[2]   = hzhxdhy * (ds + dn + gs - gn) + hyhzdhx * (dw + de + gw - ge) + hxhydhz * (du - gu);
              c[3].k = k;
              c[3].j = j + 1;
              c[3].i = i;
              v[3]   = -hzhxdhy * (dn + gn);
              c[4].k = k + 1;
              c[4].j = j;
              c[4].i = i;
              v[4]   = -hxhydhz * (du + gu);
              PetscCall(MatSetValuesStencil(jac, 1, &row, 5, c, v, INSERT_VALUES));

            } else if (k == mz - 1) { /* right-hand up interior edge */

              td = x[k - 1][j][i];
              ad = 0.5 * (t0 + td);
              bd = PetscPowScalar(ad, bm1);
              /* dd = bd * ad; */
              dd = PetscPowScalar(ad, beta);
              gd = coef * bd * (t0 - td);

              c[0].k = k - 1;
              c[0].j = j;
              c[0].i = i;
              v[0]   = -hxhydhz * (dd - gd);
              c[1].k = k;
              c[1].j = j - 1;
              c[1].i = i;
              v[1]   = -hzhxdhy * (ds - gs);
              c[2].k = k;
              c[2].j = j;
              c[2].i = i - 1;
              v[2]   = -hyhzdhx * (dw - gw);
              c[3].k = k;
              c[3].j = j;
              c[3].i = i;
              v[3]   = hzhxdhy * (ds + dn + gs - gn) + hyhzdhx * (dw + de + gw - ge) + hxhydhz * (dd + gd);
              c[4].k = k;
              c[4].j = j + 1;
              c[4].i = i;
              v[4]   = -hzhxdhy * (dn + gn);
              PetscCall(MatSetValuesStencil(jac, 1, &row, 5, c, v, INSERT_VALUES));

            } else { /* right-hand interior plane */

              td = x[k - 1][j][i];
              ad = 0.5 * (t0 + td);
              bd = PetscPowScalar(ad, bm1);
              /* dd = bd * ad; */
              dd = PetscPowScalar(ad, beta);
              gd = coef * bd * (t0 - td);

              tu = x[k + 1][j][i];
              au = 0.5 * (t0 + tu);
              bu = PetscPowScalar(au, bm1);
              /* du = bu * au; */
              du = PetscPowScalar(au, beta);
              gu = coef * bu * (tu - t0);

              c[0].k = k - 1;
              c[0].j = j;
              c[0].i = i;
              v[0]   = -hxhydhz * (dd - gd);
              c[1].k = k;
              c[1].j = j - 1;
              c[1].i = i;
              v[1]   = -hzhxdhy * (ds - gs);
              c[2].k = k;
              c[2].j = j;
              c[2].i = i - 1;
              v[2]   = -hyhzdhx * (dw - gw);
              c[3].k = k;
              c[3].j = j;
              c[3].i = i;
              v[3]   = hzhxdhy * (ds + dn + gs - gn) + hyhzdhx * (dw + de + gw - ge) + hxhydhz * (dd + du + gd - gu);
              c[4].k = k;
              c[4].j = j + 1;
              c[4].i = i;
              v[4]   = -hzhxdhy * (dn + gn);
              c[5].k = k + 1;
              c[5].j = j;
              c[5].i = i;
              v[5]   = -hxhydhz * (du + gu);
              PetscCall(MatSetValuesStencil(jac, 1, &row, 6, c, v, INSERT_VALUES));
            }
          }

        } else if (j == 0) {
          tw = x[k][j][i - 1];
          aw = 0.5 * (t0 + tw);
          bw = PetscPowScalar(aw, bm1);
          /* dw = bw * aw */
          dw = PetscPowScalar(aw, beta);
          gw = coef * bw * (t0 - tw);

          te = x[k][j][i + 1];
          ae = 0.5 * (t0 + te);
          be = PetscPowScalar(ae, bm1);
          /* de = be * ae; */
          de = PetscPowScalar(ae, beta);
          ge = coef * be * (te - t0);

          tn = x[k][j + 1][i];
          an = 0.5 * (t0 + tn);
          bn = PetscPowScalar(an, bm1);
          /* dn = bn * an; */
          dn = PetscPowScalar(an, beta);
          gn = coef * bn * (tn - t0);

          /* bottom down interior edge */
          if (k == 0) {
            tu = x[k + 1][j][i];
            au = 0.5 * (t0 + tu);
            bu = PetscPowScalar(au, bm1);
            /* du = bu * au; */
            du = PetscPowScalar(au, beta);
            gu = coef * bu * (tu - t0);

            c[0].k = k;
            c[0].j = j;
            c[0].i = i - 1;
            v[0]   = -hyhzdhx * (dw - gw);
            c[1].k = k;
            c[1].j = j;
            c[1].i = i;
            v[1]   = hzhxdhy * (dn - gn) + hyhzdhx * (dw + de + gw - ge) + hxhydhz * (du - gu);
            c[2].k = k;
            c[2].j = j;
            c[2].i = i + 1;
            v[2]   = -hyhzdhx * (de + ge);
            c[3].k = k;
            c[3].j = j + 1;
            c[3].i = i;
            v[3]   = -hzhxdhy * (dn + gn);
            c[4].k = k + 1;
            c[4].j = j;
            c[4].i = i;
            v[4]   = -hxhydhz * (du + gu);
            PetscCall(MatSetValuesStencil(jac, 1, &row, 5, c, v, INSERT_VALUES));

          } else if (k == mz - 1) { /* bottom up interior edge */

            td = x[k - 1][j][i];
            ad = 0.5 * (t0 + td);
            bd = PetscPowScalar(ad, bm1);
            /* dd = bd * ad; */
            dd = PetscPowScalar(ad, beta);
            gd = coef * bd * (td - t0);

            c[0].k = k - 1;
            c[0].j = j;
            c[0].i = i;
            v[0]   = -hxhydhz * (dd - gd);
            c[1].k = k;
            c[1].j = j;
            c[1].i = i - 1;
            v[1]   = -hyhzdhx * (dw - gw);
            c[2].k = k;
            c[2].j = j;
            c[2].i = i;
            v[2]   = hzhxdhy * (dn - gn) + hyhzdhx * (dw + de + gw - ge) + hxhydhz * (dd + gd);
            c[3].k = k;
            c[3].j = j;
            c[3].i = i + 1;
            v[3]   = -hyhzdhx * (de + ge);
            c[4].k = k;
            c[4].j = j + 1;
            c[4].i = i;
            v[4]   = -hzhxdhy * (dn + gn);
            PetscCall(MatSetValuesStencil(jac, 1, &row, 5, c, v, INSERT_VALUES));

          } else { /* bottom interior plane */

            tu = x[k + 1][j][i];
            au = 0.5 * (t0 + tu);
            bu = PetscPowScalar(au, bm1);
            /* du = bu * au; */
            du = PetscPowScalar(au, beta);
            gu = coef * bu * (tu - t0);

            td = x[k - 1][j][i];
            ad = 0.5 * (t0 + td);
            bd = PetscPowScalar(ad, bm1);
            /* dd = bd * ad; */
            dd = PetscPowScalar(ad, beta);
            gd = coef * bd * (td - t0);

            c[0].k = k - 1;
            c[0].j = j;
            c[0].i = i;
            v[0]   = -hxhydhz * (dd - gd);
            c[1].k = k;
            c[1].j = j;
            c[1].i = i - 1;
            v[1]   = -hyhzdhx * (dw - gw);
            c[2].k = k;
            c[2].j = j;
            c[2].i = i;
            v[2]   = hzhxdhy * (dn - gn) + hyhzdhx * (dw + de + gw - ge) + hxhydhz * (dd + du + gd - gu);
            c[3].k = k;
            c[3].j = j;
            c[3].i = i + 1;
            v[3]   = -hyhzdhx * (de + ge);
            c[4].k = k;
            c[4].j = j + 1;
            c[4].i = i;
            v[4]   = -hzhxdhy * (dn + gn);
            c[5].k = k + 1;
            c[5].j = j;
            c[5].i = i;
            v[5]   = -hxhydhz * (du + gu);
            PetscCall(MatSetValuesStencil(jac, 1, &row, 6, c, v, INSERT_VALUES));
          }

        } else if (j == my - 1) {
          tw = x[k][j][i - 1];
          aw = 0.5 * (t0 + tw);
          bw = PetscPowScalar(aw, bm1);
          /* dw = bw * aw */
          dw = PetscPowScalar(aw, beta);
          gw = coef * bw * (t0 - tw);

          te = x[k][j][i + 1];
          ae = 0.5 * (t0 + te);
          be = PetscPowScalar(ae, bm1);
          /* de = be * ae; */
          de = PetscPowScalar(ae, beta);
          ge = coef * be * (te - t0);

          ts = x[k][j - 1][i];
          as = 0.5 * (t0 + ts);
          bs = PetscPowScalar(as, bm1);
          /* ds = bs * as; */
          ds = PetscPowScalar(as, beta);
          gs = coef * bs * (t0 - ts);

          /* top down interior edge */
          if (k == 0) {
            tu = x[k + 1][j][i];
            au = 0.5 * (t0 + tu);
            bu = PetscPowScalar(au, bm1);
            /* du = bu * au; */
            du = PetscPowScalar(au, beta);
            gu = coef * bu * (tu - t0);

            c[0].k = k;
            c[0].j = j - 1;
            c[0].i = i;
            v[0]   = -hzhxdhy * (ds - gs);
            c[1].k = k;
            c[1].j = j;
            c[1].i = i - 1;
            v[1]   = -hyhzdhx * (dw - gw);
            c[2].k = k;
            c[2].j = j;
            c[2].i = i;
            v[2]   = hzhxdhy * (ds + gs) + hyhzdhx * (dw + de + gw - ge) + hxhydhz * (du - gu);
            c[3].k = k;
            c[3].j = j;
            c[3].i = i + 1;
            v[3]   = -hyhzdhx * (de + ge);
            c[4].k = k + 1;
            c[4].j = j;
            c[4].i = i;
            v[4]   = -hxhydhz * (du + gu);
            PetscCall(MatSetValuesStencil(jac, 1, &row, 5, c, v, INSERT_VALUES));

          } else if (k == mz - 1) { /* top up interior edge */

            td = x[k - 1][j][i];
            ad = 0.5 * (t0 + td);
            bd = PetscPowScalar(ad, bm1);
            /* dd = bd * ad; */
            dd = PetscPowScalar(ad, beta);
            gd = coef * bd * (td - t0);

            c[0].k = k - 1;
            c[0].j = j;
            c[0].i = i;
            v[0]   = -hxhydhz * (dd - gd);
            c[1].k = k;
            c[1].j = j - 1;
            c[1].i = i;
            v[1]   = -hzhxdhy * (ds - gs);
            c[2].k = k;
            c[2].j = j;
            c[2].i = i - 1;
            v[2]   = -hyhzdhx * (dw - gw);
            c[3].k = k;
            c[3].j = j;
            c[3].i = i;
            v[3]   = hzhxdhy * (ds + gs) + hyhzdhx * (dw + de + gw - ge) + hxhydhz * (dd + gd);
            c[4].k = k;
            c[4].j = j;
            c[4].i = i + 1;
            v[4]   = -hyhzdhx * (de + ge);
            PetscCall(MatSetValuesStencil(jac, 1, &row, 5, c, v, INSERT_VALUES));

          } else { /* top interior plane */

            tu = x[k + 1][j][i];
            au = 0.5 * (t0 + tu);
            bu = PetscPowScalar(au, bm1);
            /* du = bu * au; */
            du = PetscPowScalar(au, beta);
            gu = coef * bu * (tu - t0);

            td = x[k - 1][j][i];
            ad = 0.5 * (t0 + td);
            bd = PetscPowScalar(ad, bm1);
            /* dd = bd * ad; */
            dd = PetscPowScalar(ad, beta);
            gd = coef * bd * (td - t0);

            c[0].k = k - 1;
            c[0].j = j;
            c[0].i = i;
            v[0]   = -hxhydhz * (dd - gd);
            c[1].k = k;
            c[1].j = j - 1;
            c[1].i = i;
            v[1]   = -hzhxdhy * (ds - gs);
            c[2].k = k;
            c[2].j = j;
            c[2].i = i - 1;
            v[2]   = -hyhzdhx * (dw - gw);
            c[3].k = k;
            c[3].j = j;
            c[3].i = i;
            v[3]   = hzhxdhy * (ds + gs) + hyhzdhx * (dw + de + gw - ge) + hxhydhz * (dd + du + gd - gu);
            c[4].k = k;
            c[4].j = j;
            c[4].i = i + 1;
            v[4]   = -hyhzdhx * (de + ge);
            c[5].k = k + 1;
            c[5].j = j;
            c[5].i = i;
            v[5]   = -hxhydhz * (du + gu);
            PetscCall(MatSetValuesStencil(jac, 1, &row, 6, c, v, INSERT_VALUES));
          }

        } else if (k == 0) {
          /* down interior plane */

          tw = x[k][j][i - 1];
          aw = 0.5 * (t0 + tw);
          bw = PetscPowScalar(aw, bm1);
          /* dw = bw * aw */
          dw = PetscPowScalar(aw, beta);
          gw = coef * bw * (t0 - tw);

          te = x[k][j][i + 1];
          ae = 0.5 * (t0 + te);
          be = PetscPowScalar(ae, bm1);
          /* de = be * ae; */
          de = PetscPowScalar(ae, beta);
          ge = coef * be * (te - t0);

          ts = x[k][j - 1][i];
          as = 0.5 * (t0 + ts);
          bs = PetscPowScalar(as, bm1);
          /* ds = bs * as; */
          ds = PetscPowScalar(as, beta);
          gs = coef * bs * (t0 - ts);

          tn = x[k][j + 1][i];
          an = 0.5 * (t0 + tn);
          bn = PetscPowScalar(an, bm1);
          /* dn = bn * an; */
          dn = PetscPowScalar(an, beta);
          gn = coef * bn * (tn - t0);

          tu = x[k + 1][j][i];
          au = 0.5 * (t0 + tu);
          bu = PetscPowScalar(au, bm1);
          /* du = bu * au; */
          du = PetscPowScalar(au, beta);
          gu = coef * bu * (tu - t0);

          c[0].k = k;
          c[0].j = j - 1;
          c[0].i = i;
          v[0]   = -hzhxdhy * (ds - gs);
          c[1].k = k;
          c[1].j = j;
          c[1].i = i - 1;
          v[1]   = -hyhzdhx * (dw - gw);
          c[2].k = k;
          c[2].j = j;
          c[2].i = i;
          v[2]   = hzhxdhy * (ds + dn + gs - gn) + hyhzdhx * (dw + de + gw - ge) + hxhydhz * (du - gu);
          c[3].k = k;
          c[3].j = j;
          c[3].i = i + 1;
          v[3]   = -hyhzdhx * (de + ge);
          c[4].k = k;
          c[4].j = j + 1;
          c[4].i = i;
          v[4]   = -hzhxdhy * (dn + gn);
          c[5].k = k + 1;
          c[5].j = j;
          c[5].i = i;
          v[5]   = -hxhydhz * (du + gu);
          PetscCall(MatSetValuesStencil(jac, 1, &row, 6, c, v, INSERT_VALUES));

        } else if (k == mz - 1) {
          /* up interior plane */

          tw = x[k][j][i - 1];
          aw = 0.5 * (t0 + tw);
          bw = PetscPowScalar(aw, bm1);
          /* dw = bw * aw */
          dw = PetscPowScalar(aw, beta);
          gw = coef * bw * (t0 - tw);

          te = x[k][j][i + 1];
          ae = 0.5 * (t0 + te);
          be = PetscPowScalar(ae, bm1);
          /* de = be * ae; */
          de = PetscPowScalar(ae, beta);
          ge = coef * be * (te - t0);

          ts = x[k][j - 1][i];
          as = 0.5 * (t0 + ts);
          bs = PetscPowScalar(as, bm1);
          /* ds = bs * as; */
          ds = PetscPowScalar(as, beta);
          gs = coef * bs * (t0 - ts);

          tn = x[k][j + 1][i];
          an = 0.5 * (t0 + tn);
          bn = PetscPowScalar(an, bm1);
          /* dn = bn * an; */
          dn = PetscPowScalar(an, beta);
          gn = coef * bn * (tn - t0);

          td = x[k - 1][j][i];
          ad = 0.5 * (t0 + td);
          bd = PetscPowScalar(ad, bm1);
          /* dd = bd * ad; */
          dd = PetscPowScalar(ad, beta);
          gd = coef * bd * (t0 - td);

          c[0].k = k - 1;
          c[0].j = j;
          c[0].i = i;
          v[0]   = -hxhydhz * (dd - gd);
          c[1].k = k;
          c[1].j = j - 1;
          c[1].i = i;
          v[1]   = -hzhxdhy * (ds - gs);
          c[2].k = k;
          c[2].j = j;
          c[2].i = i - 1;
          v[2]   = -hyhzdhx * (dw - gw);
          c[3].k = k;
          c[3].j = j;
          c[3].i = i;
          v[3]   = hzhxdhy * (ds + dn + gs - gn) + hyhzdhx * (dw + de + gw - ge) + hxhydhz * (dd + gd);
          c[4].k = k;
          c[4].j = j;
          c[4].i = i + 1;
          v[4]   = -hyhzdhx * (de + ge);
          c[5].k = k;
          c[5].j = j + 1;
          c[5].i = i;
          v[5]   = -hzhxdhy * (dn + gn);
          PetscCall(MatSetValuesStencil(jac, 1, &row, 6, c, v, INSERT_VALUES));
        }
      }
    }
  }
  PetscCall(MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY));
  if (jac != J) {
    PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
  }
  PetscCall(DMDAVecRestoreArray(da, localX, &x));
  PetscCall(DMRestoreLocalVector(da, &localX));

  PetscCall(PetscLogFlops((41.0 + 8.0 * POWFLOP) * xm * ym));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*TEST

   test:
      nsize: 4
      args: -snes_monitor_short -pc_mg_type full -ksp_type fgmres -pc_type mg -snes_view -pc_mg_levels 2 -pc_mg_galerkin pmat
      requires: !single

TEST*/
