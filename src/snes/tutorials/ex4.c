#include <petscsnes.h>
#include <petscdmda.h>

static const char help[] = "Minimum surface area problem in 2D using DMDA.\n\
It solves an unconstrained minimization problem. This example is based on a \n\
problem from the MINPACK-2 test suite. Given a rectangular 2-D domain and \n\
boundary values along the edges of the domain, the objective is to find the\n\
surface with the minimal area that satisfies the boundary conditions.\n\
\n\
The command line options are:\n\
  -da_grid_x <nx>, where <nx> = number of grid points in the 1st coordinate direction\n\
  -da_grid_y <ny>, where <ny> = number of grid points in the 2nd coordinate direction\n\
  \n";

/*
   User-defined application context - contains data needed by the
   application-provided call-back routines, FormJacobianLocal() and
   FormFunctionLocal().
*/

typedef enum {
  PROBLEM_ENNEPER,
  PROBLEM_SINS,
} ProblemType;
static const char *const ProblemTypes[] = {"ENNEPER", "SINS", "ProblemType", "PROBLEM_", 0};

typedef struct {
  PetscScalar *bottom, *top, *left, *right;
} AppCtx;

/* -------- User-defined Routines --------- */

static PetscErrorCode FormBoundaryConditions_Enneper(SNES, AppCtx **);
static PetscErrorCode FormBoundaryConditions_Sins(SNES, AppCtx **);
static PetscErrorCode DestroyBoundaryConditions(AppCtx **);
static PetscErrorCode FormObjectiveLocal(DMDALocalInfo *, PetscScalar **, PetscReal *, void *);
static PetscErrorCode FormFunctionLocal(DMDALocalInfo *, PetscScalar **, PetscScalar **, void *);
static PetscErrorCode FormJacobianLocal(DMDALocalInfo *, PetscScalar **, Mat, Mat, void *);

int main(int argc, char **argv)
{
  Vec         x;
  SNES        snes;
  DM          da;
  ProblemType ptype   = PROBLEM_ENNEPER;
  PetscBool   use_obj = PETSC_TRUE;
  PetscReal   bbox[4] = {0.};
  AppCtx     *user;
  PetscErrorCode (*form_bc)(SNES, AppCtx **) = NULL;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Minimal surface options", __FILE__);
  PetscCall(PetscOptionsEnum("-problem_type", "Problem type", NULL, ProblemTypes, (PetscEnum)ptype, (PetscEnum *)&ptype, NULL));
  PetscCall(PetscOptionsBool("-use_objective", "Use objective function", NULL, use_obj, &use_obj, NULL));
  PetscOptionsEnd();
  switch (ptype) {
  case PROBLEM_ENNEPER:
    bbox[0] = -0.5;
    bbox[1] = 0.5;
    bbox[2] = -0.5;
    bbox[3] = 0.5;
    form_bc = FormBoundaryConditions_Enneper;
    break;
  case PROBLEM_SINS:
    bbox[0] = 0.0;
    bbox[1] = 1.0;
    bbox[2] = 0.0;
    bbox[3] = 1.0;
    form_bc = FormBoundaryConditions_Sins;
    break;
  }

  /* Create distributed array to manage the 2d grid */
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, 4, 4, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMDASetUniformCoordinates(da, bbox[0], bbox[1], bbox[2], bbox[3], PETSC_DECIDE, PETSC_DECIDE));

  /* Extract global vectors from DMDA; */
  PetscCall(DMCreateGlobalVector(da, &x));

  /* Create nonlinear solver context */
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  PetscCall(SNESSetDM(snes, da));
  PetscCall((*form_bc)(snes, &user));
  PetscCall(SNESSetApplicationContext(snes, user));

  /*  Set local callbacks */
  if (use_obj) PetscCall(DMDASNESSetObjectiveLocal(da, (DMDASNESObjectiveFn *)FormObjectiveLocal, NULL));
  PetscCall(DMDASNESSetFunctionLocal(da, INSERT_VALUES, (DMDASNESFunctionFn *)FormFunctionLocal, NULL));
  PetscCall(DMDASNESSetJacobianLocal(da, (DMDASNESJacobianFn *)FormJacobianLocal, NULL));

  /* Customize from command line */
  PetscCall(SNESSetFromOptions(snes));

  /* Solve the application */
  PetscCall(SNESSolve(snes, NULL, x));

  /* Free user-created data structures */
  PetscCall(VecDestroy(&x));
  PetscCall(SNESDestroy(&snes));
  PetscCall(DMDestroy(&da));
  PetscCall(DestroyBoundaryConditions(&user));

  PetscCall(PetscFinalize());
  return 0;
}

/* Compute objective function over the locally owned part of the mesh */
PetscErrorCode FormObjectiveLocal(DMDALocalInfo *info, PetscScalar **x, PetscReal *v, void *ptr)
{
  AppCtx     *user = (AppCtx *)ptr;
  PetscInt    mx = info->mx, my = info->my;
  PetscInt    xs = info->xs, xm = info->xm, ys = info->ys, ym = info->ym;
  PetscInt    i, j;
  PetscScalar hx, hy;
  PetscScalar f2, f4, d1, d2, d3, d4, xc, xl, xr, xt, xb;
  PetscReal   ft = 0, area;

  PetscFunctionBeginUser;
  PetscCheck(user, PetscObjectComm((PetscObject)info->da), PETSC_ERR_PLIB, "Missing application context");
  hx   = 1.0 / (mx + 1);
  hy   = 1.0 / (my + 1);
  area = 0.5 * hx * hy;
  for (j = ys; j < ys + ym; j++) {
    for (i = xs; i < xs + xm; i++) {
      xc = x[j][i];
      xl = xr = xb = xt = xc;

      if (i == 0) { /* left side */
        xl = user->left[j + 1];
      } else xl = x[j][i - 1];

      if (j == 0) { /* bottom side */
        xb = user->bottom[i + 1];
      } else xb = x[j - 1][i];

      if (i + 1 == mx) { /* right side */
        xr = user->right[j + 1];
      } else xr = x[j][i + 1];

      if (j + 1 == 0 + my) { /* top side */
        xt = user->top[i + 1];
      } else xt = x[j + 1][i];

      d1 = (xc - xl);
      d2 = (xc - xr);
      d3 = (xc - xt);
      d4 = (xc - xb);

      d1 /= hx;
      d2 /= hx;
      d3 /= hy;
      d4 /= hy;

      f2 = PetscSqrtScalar(1.0 + d1 * d1 + d4 * d4);
      f4 = PetscSqrtScalar(1.0 + d3 * d3 + d2 * d2);

      ft += PetscRealPart(f2 + f4);
    }
  }

  /* Compute triangular areas along the border of the domain. */
  if (xs == 0) { /* left side */
    for (j = ys; j < ys + ym; j++) {
      d3 = (user->left[j + 1] - user->left[j + 2]) / hy;
      d2 = (user->left[j + 1] - x[j][0]) / hx;
      ft += PetscSqrtReal(1.0 + d3 * d3 + d2 * d2);
    }
  }
  if (ys == 0) { /* bottom side */
    for (i = xs; i < xs + xm; i++) {
      d2 = (user->bottom[i + 1] - user->bottom[i + 2]) / hx;
      d3 = (user->bottom[i + 1] - x[0][i]) / hy;
      ft += PetscSqrtReal(1.0 + d3 * d3 + d2 * d2);
    }
  }
  if (xs + xm == mx) { /* right side */
    for (j = ys; j < ys + ym; j++) {
      d1 = (x[j][mx - 1] - user->right[j + 1]) / hx;
      d4 = (user->right[j] - user->right[j + 1]) / hy;
      ft += PetscSqrtReal(1.0 + d1 * d1 + d4 * d4);
    }
  }
  if (ys + ym == my) { /* top side */
    for (i = xs; i < xs + xm; i++) {
      d1 = (x[my - 1][i] - user->top[i + 1]) / hy;
      d4 = (user->top[i + 1] - user->top[i]) / hx;
      ft += PetscSqrtReal(1.0 + d1 * d1 + d4 * d4);
    }
  }
  if (ys == 0 && xs == 0) {
    d1 = (user->left[0] - user->left[1]) / hy;
    d2 = (user->bottom[0] - user->bottom[1]) / hx;
    ft += PetscSqrtReal(1.0 + d1 * d1 + d2 * d2);
  }
  if (ys + ym == my && xs + xm == mx) {
    d1 = (user->right[ym + 1] - user->right[ym]) / hy;
    d2 = (user->top[xm + 1] - user->top[xm]) / hx;
    ft += PetscSqrtReal(1.0 + d1 * d1 + d2 * d2);
  }
  ft *= area;
  *v = ft;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Compute gradient over the locally owned part of the mesh */
PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, PetscScalar **x, PetscScalar **g, void *ptr)
{
  AppCtx     *user = (AppCtx *)ptr;
  PetscInt    mx = info->mx, my = info->my;
  PetscInt    xs = info->xs, xm = info->xm, ys = info->ys, ym = info->ym;
  PetscInt    i, j;
  PetscScalar hx, hy, hydhx, hxdhy;
  PetscScalar f1, f2, f3, f4, f5, f6, d1, d2, d3, d4, d5, d6, d7, d8, xc, xl, xr, xt, xb, xlt, xrb;
  PetscScalar df1dxc, df2dxc, df3dxc, df4dxc, df5dxc, df6dxc;

  PetscFunctionBeginUser;
  PetscCheck(user, PetscObjectComm((PetscObject)info->da), PETSC_ERR_PLIB, "Missing application context");
  hx    = 1.0 / (mx + 1);
  hy    = 1.0 / (my + 1);
  hydhx = hy / hx;
  hxdhy = hx / hy;

  for (j = ys; j < ys + ym; j++) {
    for (i = xs; i < xs + xm; i++) {
      xc  = x[j][i];
      xlt = xrb = xl = xr = xb = xt = xc;

      if (i == 0) { /* left side */
        xl  = user->left[j + 1];
        xlt = user->left[j + 2];
      } else xl = x[j][i - 1];

      if (j == 0) { /* bottom side */
        xb  = user->bottom[i + 1];
        xrb = user->bottom[i + 2];
      } else xb = x[j - 1][i];

      if (i + 1 == mx) { /* right side */
        xr  = user->right[j + 1];
        xrb = user->right[j];
      } else xr = x[j][i + 1];

      if (j + 1 == 0 + my) { /* top side */
        xt  = user->top[i + 1];
        xlt = user->top[i];
      } else xt = x[j + 1][i];

      if (i > 0 && j + 1 < my) xlt = x[j + 1][i - 1]; /* left top side */
      if (j > 0 && i + 1 < mx) xrb = x[j - 1][i + 1]; /* right bottom */

      d1 = (xc - xl);
      d2 = (xc - xr);
      d3 = (xc - xt);
      d4 = (xc - xb);
      d5 = (xr - xrb);
      d6 = (xrb - xb);
      d7 = (xlt - xl);
      d8 = (xt - xlt);

      df1dxc = d1 * hydhx;
      df2dxc = (d1 * hydhx + d4 * hxdhy);
      df3dxc = d3 * hxdhy;
      df4dxc = (d2 * hydhx + d3 * hxdhy);
      df5dxc = d2 * hydhx;
      df6dxc = d4 * hxdhy;

      d1 /= hx;
      d2 /= hx;
      d3 /= hy;
      d4 /= hy;
      d5 /= hy;
      d6 /= hx;
      d7 /= hy;
      d8 /= hx;

      f1 = PetscSqrtScalar(1.0 + d1 * d1 + d7 * d7);
      f2 = PetscSqrtScalar(1.0 + d1 * d1 + d4 * d4);
      f3 = PetscSqrtScalar(1.0 + d3 * d3 + d8 * d8);
      f4 = PetscSqrtScalar(1.0 + d3 * d3 + d2 * d2);
      f5 = PetscSqrtScalar(1.0 + d2 * d2 + d5 * d5);
      f6 = PetscSqrtScalar(1.0 + d4 * d4 + d6 * d6);

      df1dxc /= f1;
      df2dxc /= f2;
      df3dxc /= f3;
      df4dxc /= f4;
      df5dxc /= f5;
      df6dxc /= f6;

      g[j][i] = (df1dxc + df2dxc + df3dxc + df4dxc + df5dxc + df6dxc) / 2.0;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Compute Hessian over the locally owned part of the mesh */
PetscErrorCode FormJacobianLocal(DMDALocalInfo *info, PetscScalar **x, Mat H, Mat Hp, void *ptr)
{
  AppCtx     *user = (AppCtx *)ptr;
  PetscInt    mx = info->mx, my = info->my;
  PetscInt    xs = info->xs, xm = info->xm, ys = info->ys, ym = info->ym;
  PetscInt    i, j, k;
  MatStencil  row, col[7];
  PetscScalar hx, hy, hydhx, hxdhy;
  PetscScalar f1, f2, f3, f4, f5, f6, d1, d2, d3, d4, d5, d6, d7, d8, xc, xl, xr, xt, xb, xlt, xrb;
  PetscScalar hl, hr, ht, hb, hc, htl, hbr;
  PetscScalar v[7];

  PetscFunctionBeginUser;
  PetscCheck(user, PetscObjectComm((PetscObject)info->da), PETSC_ERR_PLIB, "Missing application context");
  hx    = 1.0 / (mx + 1);
  hy    = 1.0 / (my + 1);
  hydhx = hy / hx;
  hxdhy = hx / hy;

  for (j = ys; j < ys + ym; j++) {
    for (i = xs; i < xs + xm; i++) {
      xc  = x[j][i];
      xlt = xrb = xl = xr = xb = xt = xc;

      /* Left */
      if (i == 0) {
        xl  = user->left[j + 1];
        xlt = user->left[j + 2];
      } else xl = x[j][i - 1];

      /* Bottom */
      if (j == 0) {
        xb  = user->bottom[i + 1];
        xrb = user->bottom[i + 2];
      } else xb = x[j - 1][i];

      /* Right */
      if (i + 1 == mx) {
        xr  = user->right[j + 1];
        xrb = user->right[j];
      } else xr = x[j][i + 1];

      /* Top */
      if (j + 1 == my) {
        xt  = user->top[i + 1];
        xlt = user->top[i];
      } else xt = x[j + 1][i];

      /* Top left */
      if (i > 0 && j + 1 < my) xlt = x[j + 1][i - 1];

      /* Bottom right */
      if (j > 0 && i + 1 < mx) xrb = x[j - 1][i + 1];

      d1 = (xc - xl) / hx;
      d2 = (xc - xr) / hx;
      d3 = (xc - xt) / hy;
      d4 = (xc - xb) / hy;
      d5 = (xrb - xr) / hy;
      d6 = (xrb - xb) / hx;
      d7 = (xlt - xl) / hy;
      d8 = (xlt - xt) / hx;

      f1 = PetscSqrtScalar(1.0 + d1 * d1 + d7 * d7);
      f2 = PetscSqrtScalar(1.0 + d1 * d1 + d4 * d4);
      f3 = PetscSqrtScalar(1.0 + d3 * d3 + d8 * d8);
      f4 = PetscSqrtScalar(1.0 + d3 * d3 + d2 * d2);
      f5 = PetscSqrtScalar(1.0 + d2 * d2 + d5 * d5);
      f6 = PetscSqrtScalar(1.0 + d4 * d4 + d6 * d6);

      hl = (-hydhx * (1.0 + d7 * d7) + d1 * d7) / (f1 * f1 * f1) + (-hydhx * (1.0 + d4 * d4) + d1 * d4) / (f2 * f2 * f2);
      hr = (-hydhx * (1.0 + d5 * d5) + d2 * d5) / (f5 * f5 * f5) + (-hydhx * (1.0 + d3 * d3) + d2 * d3) / (f4 * f4 * f4);
      ht = (-hxdhy * (1.0 + d8 * d8) + d3 * d8) / (f3 * f3 * f3) + (-hxdhy * (1.0 + d2 * d2) + d2 * d3) / (f4 * f4 * f4);
      hb = (-hxdhy * (1.0 + d6 * d6) + d4 * d6) / (f6 * f6 * f6) + (-hxdhy * (1.0 + d1 * d1) + d1 * d4) / (f2 * f2 * f2);

      hbr = -d2 * d5 / (f5 * f5 * f5) - d4 * d6 / (f6 * f6 * f6);
      htl = -d1 * d7 / (f1 * f1 * f1) - d3 * d8 / (f3 * f3 * f3);

      hc = hydhx * (1.0 + d7 * d7) / (f1 * f1 * f1) + hxdhy * (1.0 + d8 * d8) / (f3 * f3 * f3) + hydhx * (1.0 + d5 * d5) / (f5 * f5 * f5) + hxdhy * (1.0 + d6 * d6) / (f6 * f6 * f6) + (hxdhy * (1.0 + d1 * d1) + hydhx * (1.0 + d4 * d4) - 2.0 * d1 * d4) / (f2 * f2 * f2) + (hxdhy * (1.0 + d2 * d2) + hydhx * (1.0 + d3 * d3) - 2.0 * d2 * d3) / (f4 * f4 * f4);

      hl /= 2.0;
      hr /= 2.0;
      ht /= 2.0;
      hb /= 2.0;
      hbr /= 2.0;
      htl /= 2.0;
      hc /= 2.0;

      k     = 0;
      row.i = i;
      row.j = j;
      /* Bottom */
      if (j > 0) {
        v[k]     = hb;
        col[k].i = i;
        col[k].j = j - 1;
        k++;
      }

      /* Bottom right */
      if (j > 0 && i < mx - 1) {
        v[k]     = hbr;
        col[k].i = i + 1;
        col[k].j = j - 1;
        k++;
      }

      /* left */
      if (i > 0) {
        v[k]     = hl;
        col[k].i = i - 1;
        col[k].j = j;
        k++;
      }

      /* Centre */
      v[k]     = hc;
      col[k].i = row.i;
      col[k].j = row.j;
      k++;

      /* Right */
      if (i < mx - 1) {
        v[k]     = hr;
        col[k].i = i + 1;
        col[k].j = j;
        k++;
      }

      /* Top left */
      if (i > 0 && j < my - 1) {
        v[k]     = htl;
        col[k].i = i - 1;
        col[k].j = j + 1;
        k++;
      }

      /* Top */
      if (j < my - 1) {
        v[k]     = ht;
        col[k].i = i;
        col[k].j = j + 1;
        k++;
      }

      PetscCall(MatSetValuesStencil(Hp, 1, &row, k, col, v, INSERT_VALUES));
    }
  }

  /* Assemble the matrix */
  PetscCall(MatAssemblyBegin(Hp, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Hp, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FormBoundaryConditions_Enneper(SNES snes, AppCtx **ouser)
{
  PetscInt     i, j, k, limit = 0, maxits = 5;
  PetscInt     mx, my;
  PetscInt     bsize = 0, lsize = 0, tsize = 0, rsize = 0;
  PetscScalar  one = 1.0, two = 2.0, three = 3.0;
  PetscScalar  det, hx, hy, xt = 0, yt = 0;
  PetscReal    fnorm, tol = 1e-10;
  PetscScalar  u1, u2, nf1, nf2, njac11, njac12, njac21, njac22;
  PetscScalar  b = -0.5, t = 0.5, l = -0.5, r = 0.5;
  PetscScalar *boundary;
  AppCtx      *user;
  DM           da;

  PetscFunctionBeginUser;
  PetscCall(SNESGetDM(snes, &da));
  PetscCall(PetscNew(&user));
  *ouser = user;
  PetscCall(DMDAGetInfo(da, PETSC_IGNORE, &mx, &my, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE));
  bsize = mx + 2;
  lsize = my + 2;
  rsize = my + 2;
  tsize = mx + 2;

  PetscCall(PetscMalloc1(bsize, &user->bottom));
  PetscCall(PetscMalloc1(tsize, &user->top));
  PetscCall(PetscMalloc1(lsize, &user->left));
  PetscCall(PetscMalloc1(rsize, &user->right));

  hx = 1.0 / (mx + 1.0);
  hy = 1.0 / (my + 1.0);

  for (j = 0; j < 4; j++) {
    if (j == 0) {
      yt       = b;
      xt       = l;
      limit    = bsize;
      boundary = user->bottom;
    } else if (j == 1) {
      yt       = t;
      xt       = l;
      limit    = tsize;
      boundary = user->top;
    } else if (j == 2) {
      yt       = b;
      xt       = l;
      limit    = lsize;
      boundary = user->left;
    } else { /* if  (j==3) */
      yt       = b;
      xt       = r;
      limit    = rsize;
      boundary = user->right;
    }

    for (i = 0; i < limit; i++) {
      u1 = xt;
      u2 = -yt;
      for (k = 0; k < maxits; k++) {
        nf1   = u1 + u1 * u2 * u2 - u1 * u1 * u1 / three - xt;
        nf2   = -u2 - u1 * u1 * u2 + u2 * u2 * u2 / three - yt;
        fnorm = PetscRealPart(PetscSqrtScalar(nf1 * nf1 + nf2 * nf2));
        if (fnorm <= tol) break;
        njac11 = one + u2 * u2 - u1 * u1;
        njac12 = two * u1 * u2;
        njac21 = -two * u1 * u2;
        njac22 = -one - u1 * u1 + u2 * u2;
        det    = njac11 * njac22 - njac21 * njac12;
        u1     = u1 - (njac22 * nf1 - njac12 * nf2) / det;
        u2     = u2 - (njac11 * nf2 - njac21 * nf1) / det;
      }

      boundary[i] = u1 * u1 - u2 * u2;
      if (j == 0 || j == 1) xt = xt + hx;
      else yt = yt + hy; /* if (j==2 || j==3) */
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DestroyBoundaryConditions(AppCtx **ouser)
{
  AppCtx *user = *ouser;

  PetscFunctionBeginUser;
  PetscCall(PetscFree(user->bottom));
  PetscCall(PetscFree(user->top));
  PetscCall(PetscFree(user->left));
  PetscCall(PetscFree(user->right));
  PetscCall(PetscFree(*ouser));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FormBoundaryConditions_Sins(SNES snes, AppCtx **ouser)
{
  PetscInt     i, j;
  PetscInt     mx, my;
  PetscInt     limit, bsize = 0, lsize = 0, tsize = 0, rsize = 0;
  PetscScalar  hx, hy, xt = 0, yt = 0;
  PetscScalar  b = 0.0, t = 1.0, l = 0.0, r = 1.0;
  PetscScalar *boundary;
  AppCtx      *user;
  DM           da;
  PetscReal    pi2 = 2 * PETSC_PI;

  PetscFunctionBeginUser;
  PetscCall(SNESGetDM(snes, &da));
  PetscCall(PetscNew(&user));
  *ouser = user;
  PetscCall(DMDAGetInfo(da, PETSC_IGNORE, &mx, &my, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE));
  bsize = mx + 2;
  lsize = my + 2;
  rsize = my + 2;
  tsize = mx + 2;

  PetscCall(PetscMalloc1(bsize, &user->bottom));
  PetscCall(PetscMalloc1(tsize, &user->top));
  PetscCall(PetscMalloc1(lsize, &user->left));
  PetscCall(PetscMalloc1(rsize, &user->right));

  hx = 1.0 / (mx + 1.0);
  hy = 1.0 / (my + 1.0);

  for (j = 0; j < 4; j++) {
    if (j == 0) {
      yt       = b;
      xt       = l;
      limit    = bsize;
      boundary = user->bottom;
    } else if (j == 1) {
      yt       = t;
      xt       = l;
      limit    = tsize;
      boundary = user->top;
    } else if (j == 2) {
      yt       = b;
      xt       = l;
      limit    = lsize;
      boundary = user->left;
    } else { /* if  (j==3) */
      yt       = b;
      xt       = r;
      limit    = rsize;
      boundary = user->right;
    }

    for (i = 0; i < limit; i++) {
      if (j == 0) { /* bottom */
        boundary[i] = -0.5 * PetscSinReal(pi2 * xt);
      } else if (j == 1) { /* top */
        boundary[i] = 0.5 * PetscSinReal(pi2 * xt);
      } else if (j == 2) { /* left */
        boundary[i] = -0.5 * PetscSinReal(pi2 * yt);
      } else { /* right */
        boundary[i] = 0.5 * PetscSinReal(pi2 * yt);
      }
      if (j == 0 || j == 1) xt = xt + hx;
      else yt = yt + hy;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*TEST

  build:
    requires: !complex

  test:
    requires: !single
    filter: sed -e "s/CONVERGED_FNORM_ABS/CONVERGED_FNORM_RELATIVE/g"
    suffix: qn_nasm
    args: -snes_type qn -snes_npc_side {{left right}separate output} -npc_snes_type nasm -snes_converged_reason -da_local_subdomains 4 -use_objective {{0 1}separate output}

TEST*/
