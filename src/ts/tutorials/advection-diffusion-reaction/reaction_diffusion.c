#include "reaction_diffusion.h"
#include <petscdm.h>
#include <petscdmda.h>

/*F
     This example is taken from the book, Numerical Solution of Time-Dependent Advection-Diffusion-Reaction Equations by
      W. Hundsdorf and J.G. Verwer,  Page 21, Pattern Formation with Reaction-Diffusion Equations
\begin{eqnarray*}
        u_t = D_1 (u_{xx} + u_{yy})  - u*v^2 + \gamma(1 -u)           \\
        v_t = D_2 (v_{xx} + v_{yy})  + u*v^2 - (\gamma + \kappa)v
\end{eqnarray*}
    Unlike in the book this uses periodic boundary conditions instead of Neumann
    (since they are easier for finite differences).
F*/

/*
   RHSFunction - Evaluates nonlinear function, F(x).

   Input Parameters:
.  ts - the TS context
.  X - input vector
.  ptr - optional user-defined context, as set by TSSetRHSFunction()

   Output Parameter:
.  F - function vector
 */
PetscErrorCode RHSFunction(TS ts, PetscReal ftime, Vec U, Vec F, void *ptr)
{
  AppCtx     *appctx = (AppCtx *)ptr;
  DM          da;
  PetscInt    i, j, Mx, My, xs, ys, xm, ym;
  PetscReal   hx, hy, sx, sy;
  PetscScalar uc, uxx, uyy, vc, vxx, vyy;
  Field     **u, **f;
  Vec         localU;

  PetscFunctionBegin;
  PetscCall(TSGetDM(ts, &da));
  PetscCall(DMGetLocalVector(da, &localU));
  PetscCall(DMDAGetInfo(da, PETSC_IGNORE, &Mx, &My, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE));
  hx = 2.50 / (PetscReal)Mx;
  sx = 1.0 / (hx * hx);
  hy = 2.50 / (PetscReal)My;
  sy = 1.0 / (hy * hy);

  /*
     Scatter ghost points to local vector,using the 2-step process
        DMGlobalToLocalBegin(),DMGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  PetscCall(DMGlobalToLocalBegin(da, U, INSERT_VALUES, localU));
  PetscCall(DMGlobalToLocalEnd(da, U, INSERT_VALUES, localU));

  /*
     Get pointers to vector data
  */
  PetscCall(DMDAVecGetArrayRead(da, localU, &u));
  PetscCall(DMDAVecGetArray(da, F, &f));

  /*
     Get local grid boundaries
  */
  PetscCall(DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL));

  /*
     Compute function over the locally owned part of the grid
  */
  for (j = ys; j < ys + ym; j++) {
    for (i = xs; i < xs + xm; i++) {
      uc        = u[j][i].u;
      uxx       = (-2.0 * uc + u[j][i - 1].u + u[j][i + 1].u) * sx;
      uyy       = (-2.0 * uc + u[j - 1][i].u + u[j + 1][i].u) * sy;
      vc        = u[j][i].v;
      vxx       = (-2.0 * vc + u[j][i - 1].v + u[j][i + 1].v) * sx;
      vyy       = (-2.0 * vc + u[j - 1][i].v + u[j + 1][i].v) * sy;
      f[j][i].u = appctx->D1 * (uxx + uyy) - uc * vc * vc + appctx->gamma * (1.0 - uc);
      f[j][i].v = appctx->D2 * (vxx + vyy) + uc * vc * vc - (appctx->gamma + appctx->kappa) * vc;
    }
  }
  PetscCall(PetscLogFlops(16.0 * xm * ym));

  /*
     Restore vectors
  */
  PetscCall(DMDAVecRestoreArrayRead(da, localU, &u));
  PetscCall(DMDAVecRestoreArray(da, F, &f));
  PetscCall(DMRestoreLocalVector(da, &localU));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RHSJacobian(TS ts, PetscReal t, Vec U, Mat A, Mat BB, void *ctx)
{
  AppCtx     *appctx = (AppCtx *)ctx; /* user-defined application context */
  DM          da;
  PetscInt    i, j, Mx, My, xs, ys, xm, ym;
  PetscReal   hx, hy, sx, sy;
  PetscScalar uc, vc;
  Field     **u;
  Vec         localU;
  MatStencil  stencil[6], rowstencil;
  PetscScalar entries[6];

  PetscFunctionBegin;
  PetscCall(TSGetDM(ts, &da));
  PetscCall(DMGetLocalVector(da, &localU));
  PetscCall(DMDAGetInfo(da, PETSC_IGNORE, &Mx, &My, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE));

  hx = 2.50 / (PetscReal)Mx;
  sx = 1.0 / (hx * hx);
  hy = 2.50 / (PetscReal)My;
  sy = 1.0 / (hy * hy);

  /*
     Scatter ghost points to local vector,using the 2-step process
        DMGlobalToLocalBegin(),DMGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  PetscCall(DMGlobalToLocalBegin(da, U, INSERT_VALUES, localU));
  PetscCall(DMGlobalToLocalEnd(da, U, INSERT_VALUES, localU));

  /*
     Get pointers to vector data
  */
  PetscCall(DMDAVecGetArrayRead(da, localU, &u));

  /*
     Get local grid boundaries
  */
  PetscCall(DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL));

  stencil[0].k = 0;
  stencil[1].k = 0;
  stencil[2].k = 0;
  stencil[3].k = 0;
  stencil[4].k = 0;
  stencil[5].k = 0;
  rowstencil.k = 0;
  /*
     Compute function over the locally owned part of the grid
  */
  for (j = ys; j < ys + ym; j++) {
    stencil[0].j = j - 1;
    stencil[1].j = j + 1;
    stencil[2].j = j;
    stencil[3].j = j;
    stencil[4].j = j;
    stencil[5].j = j;
    rowstencil.k = 0;
    rowstencil.j = j;
    for (i = xs; i < xs + xm; i++) {
      uc = u[j][i].u;
      vc = u[j][i].v;

      /*      uxx       = (-2.0*uc + u[j][i-1].u + u[j][i+1].u)*sx;
      uyy       = (-2.0*uc + u[j-1][i].u + u[j+1][i].u)*sy;

      vxx       = (-2.0*vc + u[j][i-1].v + u[j][i+1].v)*sx;
      vyy       = (-2.0*vc + u[j-1][i].v + u[j+1][i].v)*sy;
       f[j][i].u = appctx->D1*(uxx + uyy) - uc*vc*vc + appctx->gamma*(1.0 - uc);*/

      stencil[0].i = i;
      stencil[0].c = 0;
      entries[0]   = appctx->D1 * sy;
      stencil[1].i = i;
      stencil[1].c = 0;
      entries[1]   = appctx->D1 * sy;
      stencil[2].i = i - 1;
      stencil[2].c = 0;
      entries[2]   = appctx->D1 * sx;
      stencil[3].i = i + 1;
      stencil[3].c = 0;
      entries[3]   = appctx->D1 * sx;
      stencil[4].i = i;
      stencil[4].c = 0;
      entries[4]   = -2.0 * appctx->D1 * (sx + sy) - vc * vc - appctx->gamma;
      stencil[5].i = i;
      stencil[5].c = 1;
      entries[5]   = -2.0 * uc * vc;
      rowstencil.i = i;
      rowstencil.c = 0;

      PetscCall(MatSetValuesStencil(A, 1, &rowstencil, 6, stencil, entries, INSERT_VALUES));
      if (appctx->aijpc) PetscCall(MatSetValuesStencil(BB, 1, &rowstencil, 6, stencil, entries, INSERT_VALUES));
      stencil[0].c = 1;
      entries[0]   = appctx->D2 * sy;
      stencil[1].c = 1;
      entries[1]   = appctx->D2 * sy;
      stencil[2].c = 1;
      entries[2]   = appctx->D2 * sx;
      stencil[3].c = 1;
      entries[3]   = appctx->D2 * sx;
      stencil[4].c = 1;
      entries[4]   = -2.0 * appctx->D2 * (sx + sy) + 2.0 * uc * vc - appctx->gamma - appctx->kappa;
      stencil[5].c = 0;
      entries[5]   = vc * vc;
      rowstencil.c = 1;

      PetscCall(MatSetValuesStencil(A, 1, &rowstencil, 6, stencil, entries, INSERT_VALUES));
      if (appctx->aijpc) PetscCall(MatSetValuesStencil(BB, 1, &rowstencil, 6, stencil, entries, INSERT_VALUES));
      /* f[j][i].v = appctx->D2*(vxx + vyy) + uc*vc*vc - (appctx->gamma + appctx->kappa)*vc; */
    }
  }

  /*
     Restore vectors
  */
  PetscCall(PetscLogFlops(19.0 * xm * ym));
  PetscCall(DMDAVecRestoreArrayRead(da, localU, &u));
  PetscCall(DMRestoreLocalVector(da, &localU));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatSetOption(A, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE));
  if (appctx->aijpc) {
    PetscCall(MatAssemblyBegin(BB, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(BB, MAT_FINAL_ASSEMBLY));
    PetscCall(MatSetOption(BB, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   IFunction - Evaluates implicit nonlinear function, xdot - F(x).

   Input Parameters:
.  ts - the TS context
.  U - input vector
.  Udot - input vector
.  ptr - optional user-defined context, as set by TSSetRHSFunction()

   Output Parameter:
.  F - function vector
 */
PetscErrorCode IFunction(TS ts, PetscReal ftime, Vec U, Vec Udot, Vec F, void *ptr)
{
  AppCtx     *appctx = (AppCtx *)ptr;
  DM          da;
  PetscInt    i, j, Mx, My, xs, ys, xm, ym;
  PetscReal   hx, hy, sx, sy;
  PetscScalar uc, uxx, uyy, vc, vxx, vyy;
  Field     **u, **f, **udot;
  Vec         localU;

  PetscFunctionBegin;
  PetscCall(TSGetDM(ts, &da));
  PetscCall(DMGetLocalVector(da, &localU));
  PetscCall(DMDAGetInfo(da, PETSC_IGNORE, &Mx, &My, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE));
  hx = 2.50 / (PetscReal)Mx;
  sx = 1.0 / (hx * hx);
  hy = 2.50 / (PetscReal)My;
  sy = 1.0 / (hy * hy);

  /*
     Scatter ghost points to local vector,using the 2-step process
        DMGlobalToLocalBegin(),DMGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  PetscCall(DMGlobalToLocalBegin(da, U, INSERT_VALUES, localU));
  PetscCall(DMGlobalToLocalEnd(da, U, INSERT_VALUES, localU));

  /*
     Get pointers to vector data
  */
  PetscCall(DMDAVecGetArrayRead(da, localU, &u));
  PetscCall(DMDAVecGetArray(da, F, &f));
  PetscCall(DMDAVecGetArrayRead(da, Udot, &udot));

  /*
     Get local grid boundaries
  */
  PetscCall(DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL));

  /*
     Compute function over the locally owned part of the grid
  */
  for (j = ys; j < ys + ym; j++) {
    for (i = xs; i < xs + xm; i++) {
      uc        = u[j][i].u;
      uxx       = (-2.0 * uc + u[j][i - 1].u + u[j][i + 1].u) * sx;
      uyy       = (-2.0 * uc + u[j - 1][i].u + u[j + 1][i].u) * sy;
      vc        = u[j][i].v;
      vxx       = (-2.0 * vc + u[j][i - 1].v + u[j][i + 1].v) * sx;
      vyy       = (-2.0 * vc + u[j - 1][i].v + u[j + 1][i].v) * sy;
      f[j][i].u = udot[j][i].u - (appctx->D1 * (uxx + uyy) - uc * vc * vc + appctx->gamma * (1.0 - uc));
      f[j][i].v = udot[j][i].v - (appctx->D2 * (vxx + vyy) + uc * vc * vc - (appctx->gamma + appctx->kappa) * vc);
    }
  }
  PetscCall(PetscLogFlops(16.0 * xm * ym));

  /*
     Restore vectors
  */
  PetscCall(DMDAVecRestoreArrayRead(da, localU, &u));
  PetscCall(DMDAVecRestoreArray(da, F, &f));
  PetscCall(DMDAVecRestoreArrayRead(da, Udot, &udot));
  PetscCall(DMRestoreLocalVector(da, &localU));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode IJacobian(TS ts, PetscReal t, Vec U, Vec Udot, PetscReal a, Mat A, Mat BB, void *ctx)
{
  AppCtx     *appctx = (AppCtx *)ctx; /* user-defined application context */
  DM          da;
  PetscInt    i, j, Mx, My, xs, ys, xm, ym;
  PetscReal   hx, hy, sx, sy;
  PetscScalar uc, vc;
  Field     **u;
  Vec         localU;
  MatStencil  stencil[6], rowstencil;
  PetscScalar entries[6];

  PetscFunctionBegin;
  PetscCall(TSGetDM(ts, &da));
  PetscCall(DMGetLocalVector(da, &localU));
  PetscCall(DMDAGetInfo(da, PETSC_IGNORE, &Mx, &My, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE));

  hx = 2.50 / (PetscReal)Mx;
  sx = 1.0 / (hx * hx);
  hy = 2.50 / (PetscReal)My;
  sy = 1.0 / (hy * hy);

  /*
     Scatter ghost points to local vector,using the 2-step process
        DMGlobalToLocalBegin(),DMGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  PetscCall(DMGlobalToLocalBegin(da, U, INSERT_VALUES, localU));
  PetscCall(DMGlobalToLocalEnd(da, U, INSERT_VALUES, localU));

  /*
     Get pointers to vector data
  */
  PetscCall(DMDAVecGetArrayRead(da, localU, &u));

  /*
     Get local grid boundaries
  */
  PetscCall(DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL));

  stencil[0].k = 0;
  stencil[1].k = 0;
  stencil[2].k = 0;
  stencil[3].k = 0;
  stencil[4].k = 0;
  stencil[5].k = 0;
  rowstencil.k = 0;
  /*
     Compute function over the locally owned part of the grid
  */
  for (j = ys; j < ys + ym; j++) {
    stencil[0].j = j - 1;
    stencil[1].j = j + 1;
    stencil[2].j = j;
    stencil[3].j = j;
    stencil[4].j = j;
    stencil[5].j = j;
    rowstencil.k = 0;
    rowstencil.j = j;
    for (i = xs; i < xs + xm; i++) {
      uc = u[j][i].u;
      vc = u[j][i].v;

      /*      uxx       = (-2.0*uc + u[j][i-1].u + u[j][i+1].u)*sx;
      uyy       = (-2.0*uc + u[j-1][i].u + u[j+1][i].u)*sy;

      vxx       = (-2.0*vc + u[j][i-1].v + u[j][i+1].v)*sx;
      vyy       = (-2.0*vc + u[j-1][i].v + u[j+1][i].v)*sy;
       f[j][i].u = appctx->D1*(uxx + uyy) - uc*vc*vc + appctx->gamma*(1.0 - uc);*/

      stencil[0].i = i;
      stencil[0].c = 0;
      entries[0]   = -appctx->D1 * sy;
      stencil[1].i = i;
      stencil[1].c = 0;
      entries[1]   = -appctx->D1 * sy;
      stencil[2].i = i - 1;
      stencil[2].c = 0;
      entries[2]   = -appctx->D1 * sx;
      stencil[3].i = i + 1;
      stencil[3].c = 0;
      entries[3]   = -appctx->D1 * sx;
      stencil[4].i = i;
      stencil[4].c = 0;
      entries[4]   = 2.0 * appctx->D1 * (sx + sy) + vc * vc + appctx->gamma + a;
      stencil[5].i = i;
      stencil[5].c = 1;
      entries[5]   = 2.0 * uc * vc;
      rowstencil.i = i;
      rowstencil.c = 0;

      PetscCall(MatSetValuesStencil(A, 1, &rowstencil, 6, stencil, entries, INSERT_VALUES));
      if (appctx->aijpc) PetscCall(MatSetValuesStencil(BB, 1, &rowstencil, 6, stencil, entries, INSERT_VALUES));
      stencil[0].c = 1;
      entries[0]   = -appctx->D2 * sy;
      stencil[1].c = 1;
      entries[1]   = -appctx->D2 * sy;
      stencil[2].c = 1;
      entries[2]   = -appctx->D2 * sx;
      stencil[3].c = 1;
      entries[3]   = -appctx->D2 * sx;
      stencil[4].c = 1;
      entries[4]   = 2.0 * appctx->D2 * (sx + sy) - 2.0 * uc * vc + appctx->gamma + appctx->kappa + a;
      stencil[5].c = 0;
      entries[5]   = -vc * vc;
      rowstencil.c = 1;

      PetscCall(MatSetValuesStencil(A, 1, &rowstencil, 6, stencil, entries, INSERT_VALUES));
      if (appctx->aijpc) PetscCall(MatSetValuesStencil(BB, 1, &rowstencil, 6, stencil, entries, INSERT_VALUES));
      /* f[j][i].v = appctx->D2*(vxx + vyy) + uc*vc*vc - (appctx->gamma + appctx->kappa)*vc; */
    }
  }

  /*
     Restore vectors
  */
  PetscCall(PetscLogFlops(19.0 * xm * ym));
  PetscCall(DMDAVecRestoreArrayRead(da, localU, &u));
  PetscCall(DMRestoreLocalVector(da, &localU));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatSetOption(A, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE));
  if (appctx->aijpc) {
    PetscCall(MatAssemblyBegin(BB, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(BB, MAT_FINAL_ASSEMBLY));
    PetscCall(MatSetOption(BB, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
