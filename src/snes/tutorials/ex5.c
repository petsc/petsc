static char help[] = "Bratu nonlinear PDE in 2d.\n\
We solve the  Bratu (SFI - solid fuel ignition) problem in a 2D rectangular\n\
domain, using distributed arrays (DMDAs) to partition the parallel grid.\n\
The command line options include:\n\
  -par <parameter>, where <parameter> indicates the problem's nonlinearity\n\
     problem SFI:  <parameter> = Bratu parameter (0 <= par <= 6.81)\n\n\
  -m_par/n_par <parameter>, where <parameter> indicates an integer\n \
      that MMS3 will be evaluated with 2^m_par, 2^n_par";

/* ------------------------------------------------------------------------

    Solid Fuel Ignition (SFI) problem.  This problem is modeled by
    the partial differential equation

            -Laplacian u - lambda*exp(u) = 0,  0 < x,y < 1,

    with boundary conditions

             u = 0  for  x = 0, x = 1, y = 0, y = 1.

    A finite difference approximation with the usual 5-point stencil
    is used to discretize the boundary value problem to obtain a nonlinear
    system of equations.

      This example shows how geometric multigrid can be run transparently with a nonlinear solver so long
      as SNESSetDM() is provided. Example usage

      ./ex5 -pc_type mg -ksp_monitor  -snes_view -pc_mg_levels 3 -pc_mg_galerkin pmat -da_grid_x 17 -da_grid_y 17
             -mg_levels_ksp_monitor -snes_monitor -mg_levels_pc_type sor -pc_mg_type full

      or to run with grid sequencing on the nonlinear problem (note that you do not need to provide the number of
         multigrid levels, it will be determined automatically based on the number of refinements done)

      ./ex5 -pc_type mg -ksp_monitor  -snes_view -pc_mg_galerkin pmat -snes_grid_sequence 3
             -mg_levels_ksp_monitor -snes_monitor -mg_levels_pc_type sor -pc_mg_type full

  ------------------------------------------------------------------------- */

/*
   Include "petscdmda.h" so that we can use distributed arrays (DMDAs).
   Include "petscsnes.h" so that we can use SNES solvers.  Note that this
*/
#include <petscdm.h>
#include <petscdmda.h>
#include <petscsnes.h>
#include <petscmatlab.h>
#include <petsc/private/snesimpl.h> /* For SNES_Solve event */

/*
   User-defined application context - contains data needed by the
   application-provided call-back routines, FormJacobianLocal() and
   FormFunctionLocal().
*/
typedef struct AppCtx AppCtx;
struct AppCtx {
  PetscReal param; /* test problem parameter */
  PetscInt  m, n;  /* MMS3 parameters */
  PetscErrorCode (*mms_solution)(AppCtx *, const DMDACoor2d *, PetscScalar *);
  PetscErrorCode (*mms_forcing)(AppCtx *, const DMDACoor2d *, PetscScalar *);
};

/* ------------------------------------------------------------------- */
/*
   FormInitialGuess - Forms initial approximation.

   Input Parameters:
   da - The DM
   user - user-defined application context

   Output Parameter:
   X - vector
 */
static PetscErrorCode FormInitialGuess(DM da, AppCtx *user, Vec X)
{
  PetscInt      i, j, Mx, My, xs, ys, xm, ym;
  PetscReal     lambda, temp1, temp, hx, hy;
  PetscScalar **x;

  PetscFunctionBeginUser;
  PetscCall(DMDAGetInfo(da, PETSC_IGNORE, &Mx, &My, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE));

  lambda = user->param;
  hx     = 1.0 / (PetscReal)(Mx - 1);
  hy     = 1.0 / (PetscReal)(My - 1);
  temp1  = lambda / (lambda + 1.0);

  /*
     Get a pointer to vector data.
       - For default PETSc vectors, VecGetArray() returns a pointer to
         the data array.  Otherwise, the routine is implementation dependent.
       - You MUST call VecRestoreArray() when you no longer need access to
         the array.
  */
  PetscCall(DMDAVecGetArray(da, X, &x));

  /*
     Get local grid boundaries (for 2-dimensional DMDA):
       xs, ys   - starting grid indices (no ghost points)
       xm, ym   - widths of local grid (no ghost points)

  */
  PetscCall(DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL));

  /*
     Compute initial guess over the locally owned part of the grid
  */
  for (j = ys; j < ys + ym; j++) {
    temp = (PetscReal)(PetscMin(j, My - j - 1)) * hy;
    for (i = xs; i < xs + xm; i++) {
      if (i == 0 || j == 0 || i == Mx - 1 || j == My - 1) {
        /* boundary conditions are all zero Dirichlet */
        x[j][i] = 0.0;
      } else {
        x[j][i] = temp1 * PetscSqrtReal(PetscMin((PetscReal)(PetscMin(i, Mx - i - 1)) * hx, temp));
      }
    }
  }

  /*
     Restore vector
  */
  PetscCall(DMDAVecRestoreArray(da, X, &x));
  PetscFunctionReturn(0);
}

/*
  FormExactSolution - Forms MMS solution

  Input Parameters:
  da - The DM
  user - user-defined application context

  Output Parameter:
  X - vector
 */
static PetscErrorCode FormExactSolution(DM da, AppCtx *user, Vec U)
{
  DM            coordDA;
  Vec           coordinates;
  DMDACoor2d  **coords;
  PetscScalar **u;
  PetscInt      xs, ys, xm, ym, i, j;

  PetscFunctionBeginUser;
  PetscCall(DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL));
  PetscCall(DMGetCoordinateDM(da, &coordDA));
  PetscCall(DMGetCoordinates(da, &coordinates));
  PetscCall(DMDAVecGetArray(coordDA, coordinates, &coords));
  PetscCall(DMDAVecGetArray(da, U, &u));
  for (j = ys; j < ys + ym; ++j) {
    for (i = xs; i < xs + xm; ++i) user->mms_solution(user, &coords[j][i], &u[j][i]);
  }
  PetscCall(DMDAVecRestoreArray(da, U, &u));
  PetscCall(DMDAVecRestoreArray(coordDA, coordinates, &coords));
  PetscFunctionReturn(0);
}

static PetscErrorCode ZeroBCSolution(AppCtx *user, const DMDACoor2d *c, PetscScalar *u)
{
  u[0] = 0.;
  return 0;
}

/* The functions below evaluate the MMS solution u(x,y) and associated forcing

     f(x,y) = -u_xx - u_yy - lambda exp(u)

  such that u(x,y) is an exact solution with f(x,y) as the right hand side forcing term.
 */
static PetscErrorCode MMSSolution1(AppCtx *user, const DMDACoor2d *c, PetscScalar *u)
{
  PetscReal x = PetscRealPart(c->x), y = PetscRealPart(c->y);
  u[0] = x * (1 - x) * y * (1 - y);
  PetscLogFlops(5);
  return 0;
}
static PetscErrorCode MMSForcing1(AppCtx *user, const DMDACoor2d *c, PetscScalar *f)
{
  PetscReal x = PetscRealPart(c->x), y = PetscRealPart(c->y);
  f[0] = 2 * x * (1 - x) + 2 * y * (1 - y) - user->param * PetscExpReal(x * (1 - x) * y * (1 - y));
  return 0;
}

static PetscErrorCode MMSSolution2(AppCtx *user, const DMDACoor2d *c, PetscScalar *u)
{
  PetscReal x = PetscRealPart(c->x), y = PetscRealPart(c->y);
  u[0] = PetscSinReal(PETSC_PI * x) * PetscSinReal(PETSC_PI * y);
  PetscLogFlops(5);
  return 0;
}
static PetscErrorCode MMSForcing2(AppCtx *user, const DMDACoor2d *c, PetscScalar *f)
{
  PetscReal x = PetscRealPart(c->x), y = PetscRealPart(c->y);
  f[0] = 2 * PetscSqr(PETSC_PI) * PetscSinReal(PETSC_PI * x) * PetscSinReal(PETSC_PI * y) - user->param * PetscExpReal(PetscSinReal(PETSC_PI * x) * PetscSinReal(PETSC_PI * y));
  return 0;
}

static PetscErrorCode MMSSolution3(AppCtx *user, const DMDACoor2d *c, PetscScalar *u)
{
  PetscReal x = PetscRealPart(c->x), y = PetscRealPart(c->y);
  u[0] = PetscSinReal(user->m * PETSC_PI * x * (1 - y)) * PetscSinReal(user->n * PETSC_PI * y * (1 - x));
  PetscLogFlops(5);
  return 0;
}
static PetscErrorCode MMSForcing3(AppCtx *user, const DMDACoor2d *c, PetscScalar *f)
{
  PetscReal x = PetscRealPart(c->x), y = PetscRealPart(c->y);
  PetscReal m = user->m, n = user->n, lambda = user->param;
  f[0] = (-(PetscExpReal(PetscSinReal(m * PETSC_PI * x * (1 - y)) * PetscSinReal(n * PETSC_PI * (1 - x) * y)) * lambda) + PetscSqr(PETSC_PI) * (-2 * m * n * ((-1 + x) * x + (-1 + y) * y) * PetscCosReal(m * PETSC_PI * x * (-1 + y)) * PetscCosReal(n * PETSC_PI * (-1 + x) * y) + (PetscSqr(m) * (PetscSqr(x) + PetscSqr(-1 + y)) + PetscSqr(n) * (PetscSqr(-1 + x) + PetscSqr(y))) * PetscSinReal(m * PETSC_PI * x * (-1 + y)) * PetscSinReal(n * PETSC_PI * (-1 + x) * y)));
  return 0;
}

static PetscErrorCode MMSSolution4(AppCtx *user, const DMDACoor2d *c, PetscScalar *u)
{
  const PetscReal Lx = 1., Ly = 1.;
  PetscReal       x = PetscRealPart(c->x), y = PetscRealPart(c->y);
  u[0] = (PetscPowReal(x, 4) - PetscSqr(Lx) * PetscSqr(x)) * (PetscPowReal(y, 4) - PetscSqr(Ly) * PetscSqr(y));
  PetscLogFlops(9);
  return 0;
}
static PetscErrorCode MMSForcing4(AppCtx *user, const DMDACoor2d *c, PetscScalar *f)
{
  const PetscReal Lx = 1., Ly = 1.;
  PetscReal       x = PetscRealPart(c->x), y = PetscRealPart(c->y);
  f[0] = (2 * PetscSqr(x) * (PetscSqr(x) - PetscSqr(Lx)) * (PetscSqr(Ly) - 6 * PetscSqr(y)) + 2 * PetscSqr(y) * (PetscSqr(Lx) - 6 * PetscSqr(x)) * (PetscSqr(y) - PetscSqr(Ly)) - user->param * PetscExpReal((PetscPowReal(x, 4) - PetscSqr(Lx) * PetscSqr(x)) * (PetscPowReal(y, 4) - PetscSqr(Ly) * PetscSqr(y))));
  return 0;
}

/* ------------------------------------------------------------------- */
/*
   FormFunctionLocal - Evaluates nonlinear function, F(x) on local process patch

 */
static PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, PetscScalar **x, PetscScalar **f, AppCtx *user)
{
  PetscInt    i, j;
  PetscReal   lambda, hx, hy, hxdhy, hydhx;
  PetscScalar u, ue, uw, un, us, uxx, uyy, mms_solution, mms_forcing;
  DMDACoor2d  c;

  PetscFunctionBeginUser;
  lambda = user->param;
  hx     = 1.0 / (PetscReal)(info->mx - 1);
  hy     = 1.0 / (PetscReal)(info->my - 1);
  hxdhy  = hx / hy;
  hydhx  = hy / hx;
  /*
     Compute function over the locally owned part of the grid
  */
  for (j = info->ys; j < info->ys + info->ym; j++) {
    for (i = info->xs; i < info->xs + info->xm; i++) {
      if (i == 0 || j == 0 || i == info->mx - 1 || j == info->my - 1) {
        c.x = i * hx;
        c.y = j * hy;
        PetscCall(user->mms_solution(user, &c, &mms_solution));
        f[j][i] = 2.0 * (hydhx + hxdhy) * (x[j][i] - mms_solution);
      } else {
        u  = x[j][i];
        uw = x[j][i - 1];
        ue = x[j][i + 1];
        un = x[j - 1][i];
        us = x[j + 1][i];

        /* Enforce boundary conditions at neighboring points -- setting these values causes the Jacobian to be symmetric. */
        if (i - 1 == 0) {
          c.x = (i - 1) * hx;
          c.y = j * hy;
          PetscCall(user->mms_solution(user, &c, &uw));
        }
        if (i + 1 == info->mx - 1) {
          c.x = (i + 1) * hx;
          c.y = j * hy;
          PetscCall(user->mms_solution(user, &c, &ue));
        }
        if (j - 1 == 0) {
          c.x = i * hx;
          c.y = (j - 1) * hy;
          PetscCall(user->mms_solution(user, &c, &un));
        }
        if (j + 1 == info->my - 1) {
          c.x = i * hx;
          c.y = (j + 1) * hy;
          PetscCall(user->mms_solution(user, &c, &us));
        }

        uxx         = (2.0 * u - uw - ue) * hydhx;
        uyy         = (2.0 * u - un - us) * hxdhy;
        mms_forcing = 0;
        c.x         = i * hx;
        c.y         = j * hy;
        if (user->mms_forcing) PetscCall(user->mms_forcing(user, &c, &mms_forcing));
        f[j][i] = uxx + uyy - hx * hy * (lambda * PetscExpScalar(u) + mms_forcing);
      }
    }
  }
  PetscCall(PetscLogFlops(11.0 * info->ym * info->xm));
  PetscFunctionReturn(0);
}

/* FormObjectiveLocal - Evaluates nonlinear function, F(x) on local process patch */
static PetscErrorCode FormObjectiveLocal(DMDALocalInfo *info, PetscScalar **x, PetscReal *obj, AppCtx *user)
{
  PetscInt    i, j;
  PetscReal   lambda, hx, hy, hxdhy, hydhx, sc, lobj = 0;
  PetscScalar u, ue, uw, un, us, uxux, uyuy;
  MPI_Comm    comm;

  PetscFunctionBeginUser;
  *obj = 0;
  PetscCall(PetscObjectGetComm((PetscObject)info->da, &comm));
  lambda = user->param;
  hx     = 1.0 / (PetscReal)(info->mx - 1);
  hy     = 1.0 / (PetscReal)(info->my - 1);
  sc     = hx * hy * lambda;
  hxdhy  = hx / hy;
  hydhx  = hy / hx;
  /*
     Compute function over the locally owned part of the grid
  */
  for (j = info->ys; j < info->ys + info->ym; j++) {
    for (i = info->xs; i < info->xs + info->xm; i++) {
      if (i == 0 || j == 0 || i == info->mx - 1 || j == info->my - 1) {
        lobj += PetscRealPart((hydhx + hxdhy) * x[j][i] * x[j][i]);
      } else {
        u  = x[j][i];
        uw = x[j][i - 1];
        ue = x[j][i + 1];
        un = x[j - 1][i];
        us = x[j + 1][i];

        if (i - 1 == 0) uw = 0.;
        if (i + 1 == info->mx - 1) ue = 0.;
        if (j - 1 == 0) un = 0.;
        if (j + 1 == info->my - 1) us = 0.;

        /* F[u] = 1/2\int_{\omega}\nabla^2u(x)*u(x)*dx */

        uxux = u * (2. * u - ue - uw) * hydhx;
        uyuy = u * (2. * u - un - us) * hxdhy;

        lobj += PetscRealPart(0.5 * (uxux + uyuy) - sc * PetscExpScalar(u));
      }
    }
  }
  PetscCall(PetscLogFlops(12.0 * info->ym * info->xm));
  PetscCallMPI(MPI_Allreduce(&lobj, obj, 1, MPIU_REAL, MPIU_SUM, comm));
  PetscFunctionReturn(0);
}

/*
   FormJacobianLocal - Evaluates Jacobian matrix on local process patch
*/
static PetscErrorCode FormJacobianLocal(DMDALocalInfo *info, PetscScalar **x, Mat jac, Mat jacpre, AppCtx *user)
{
  PetscInt     i, j, k;
  MatStencil   col[5], row;
  PetscScalar  lambda, v[5], hx, hy, hxdhy, hydhx, sc;
  DM           coordDA;
  Vec          coordinates;
  DMDACoor2d **coords;

  PetscFunctionBeginUser;
  lambda = user->param;
  /* Extract coordinates */
  PetscCall(DMGetCoordinateDM(info->da, &coordDA));
  PetscCall(DMGetCoordinates(info->da, &coordinates));
  PetscCall(DMDAVecGetArray(coordDA, coordinates, &coords));
  hx = info->xm > 1 ? PetscRealPart(coords[info->ys][info->xs + 1].x) - PetscRealPart(coords[info->ys][info->xs].x) : 1.0;
  hy = info->ym > 1 ? PetscRealPart(coords[info->ys + 1][info->xs].y) - PetscRealPart(coords[info->ys][info->xs].y) : 1.0;
  PetscCall(DMDAVecRestoreArray(coordDA, coordinates, &coords));
  hxdhy = hx / hy;
  hydhx = hy / hx;
  sc    = hx * hy * lambda;

  /*
     Compute entries for the locally owned part of the Jacobian.
      - Currently, all PETSc parallel matrix formats are partitioned by
        contiguous chunks of rows across the processors.
      - Each processor needs to insert only elements that it owns
        locally (but any non-local elements will be sent to the
        appropriate processor during matrix assembly).
      - Here, we set all entries for a particular row at once.
      - We can set matrix entries either using either
        MatSetValuesLocal() or MatSetValues(), as discussed above.
  */
  for (j = info->ys; j < info->ys + info->ym; j++) {
    for (i = info->xs; i < info->xs + info->xm; i++) {
      row.j = j;
      row.i = i;
      /* boundary points */
      if (i == 0 || j == 0 || i == info->mx - 1 || j == info->my - 1) {
        v[0] = 2.0 * (hydhx + hxdhy);
        PetscCall(MatSetValuesStencil(jacpre, 1, &row, 1, &row, v, INSERT_VALUES));
      } else {
        k = 0;
        /* interior grid points */
        if (j - 1 != 0) {
          v[k]     = -hxdhy;
          col[k].j = j - 1;
          col[k].i = i;
          k++;
        }
        if (i - 1 != 0) {
          v[k]     = -hydhx;
          col[k].j = j;
          col[k].i = i - 1;
          k++;
        }

        v[k]     = 2.0 * (hydhx + hxdhy) - sc * PetscExpScalar(x[j][i]);
        col[k].j = row.j;
        col[k].i = row.i;
        k++;

        if (i + 1 != info->mx - 1) {
          v[k]     = -hydhx;
          col[k].j = j;
          col[k].i = i + 1;
          k++;
        }
        if (j + 1 != info->mx - 1) {
          v[k]     = -hxdhy;
          col[k].j = j + 1;
          col[k].i = i;
          k++;
        }
        PetscCall(MatSetValuesStencil(jacpre, 1, &row, k, col, v, INSERT_VALUES));
      }
    }
  }

  /*
     Assemble matrix, using the 2-step process:
       MatAssemblyBegin(), MatAssemblyEnd().
  */
  PetscCall(MatAssemblyBegin(jacpre, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(jacpre, MAT_FINAL_ASSEMBLY));
  /*
     Tell the matrix we will never add a new nonzero location to the
     matrix. If we do, it will generate an error.
  */
  PetscCall(MatSetOption(jac, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE));
  PetscFunctionReturn(0);
}

static PetscErrorCode FormFunctionMatlab(SNES snes, Vec X, Vec F, void *ptr)
{
#if PetscDefined(HAVE_MATLAB)
  AppCtx   *user = (AppCtx *)ptr;
  PetscInt  Mx, My;
  PetscReal lambda, hx, hy;
  Vec       localX, localF;
  MPI_Comm  comm;
  DM        da;

  PetscFunctionBeginUser;
  PetscCall(SNESGetDM(snes, &da));
  PetscCall(DMGetLocalVector(da, &localX));
  PetscCall(DMGetLocalVector(da, &localF));
  PetscCall(PetscObjectSetName((PetscObject)localX, "localX"));
  PetscCall(PetscObjectSetName((PetscObject)localF, "localF"));
  PetscCall(DMDAGetInfo(da, PETSC_IGNORE, &Mx, &My, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE));

  lambda = user->param;
  hx     = 1.0 / (PetscReal)(Mx - 1);
  hy     = 1.0 / (PetscReal)(My - 1);

  PetscCall(PetscObjectGetComm((PetscObject)snes, &comm));
  /*
     Scatter ghost points to local vector,using the 2-step process
        DMGlobalToLocalBegin(),DMGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  PetscCall(DMGlobalToLocalBegin(da, X, INSERT_VALUES, localX));
  PetscCall(DMGlobalToLocalEnd(da, X, INSERT_VALUES, localX));
  PetscCall(PetscMatlabEnginePut(PETSC_MATLAB_ENGINE_(comm), (PetscObject)localX));
  PetscCall(PetscMatlabEngineEvaluate(PETSC_MATLAB_ENGINE_(comm), "localF=ex5m(localX,%18.16e,%18.16e,%18.16e)", (double)hx, (double)hy, (double)lambda));
  PetscCall(PetscMatlabEngineGet(PETSC_MATLAB_ENGINE_(comm), (PetscObject)localF));

  /*
     Insert values into global vector
  */
  PetscCall(DMLocalToGlobalBegin(da, localF, INSERT_VALUES, F));
  PetscCall(DMLocalToGlobalEnd(da, localF, INSERT_VALUES, F));
  PetscCall(DMRestoreLocalVector(da, &localX));
  PetscCall(DMRestoreLocalVector(da, &localF));
  PetscFunctionReturn(0);
#else
  return 0; /* Never called */
#endif
}

/* ------------------------------------------------------------------- */
/*
      Applies some sweeps on nonlinear Gauss-Seidel on each process

 */
static PetscErrorCode NonlinearGS(SNES snes, Vec X, Vec B, void *ctx)
{
  PetscInt      i, j, k, Mx, My, xs, ys, xm, ym, its, tot_its, sweeps, l;
  PetscReal     lambda, hx, hy, hxdhy, hydhx, sc;
  PetscScalar **x, **b, bij, F, F0 = 0, J, u, un, us, ue, eu, uw, uxx, uyy, y;
  PetscReal     atol, rtol, stol;
  DM            da;
  AppCtx       *user;
  Vec           localX, localB;

  PetscFunctionBeginUser;
  tot_its = 0;
  PetscCall(SNESNGSGetSweeps(snes, &sweeps));
  PetscCall(SNESNGSGetTolerances(snes, &atol, &rtol, &stol, &its));
  PetscCall(SNESGetDM(snes, &da));
  PetscCall(DMGetApplicationContext(da, &user));

  PetscCall(DMDAGetInfo(da, PETSC_IGNORE, &Mx, &My, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE, PETSC_IGNORE));

  lambda = user->param;
  hx     = 1.0 / (PetscReal)(Mx - 1);
  hy     = 1.0 / (PetscReal)(My - 1);
  sc     = hx * hy * lambda;
  hxdhy  = hx / hy;
  hydhx  = hy / hx;

  PetscCall(DMGetLocalVector(da, &localX));
  if (B) PetscCall(DMGetLocalVector(da, &localB));
  for (l = 0; l < sweeps; l++) {
    PetscCall(DMGlobalToLocalBegin(da, X, INSERT_VALUES, localX));
    PetscCall(DMGlobalToLocalEnd(da, X, INSERT_VALUES, localX));
    if (B) {
      PetscCall(DMGlobalToLocalBegin(da, B, INSERT_VALUES, localB));
      PetscCall(DMGlobalToLocalEnd(da, B, INSERT_VALUES, localB));
    }
    /*
     Get a pointer to vector data.
     - For default PETSc vectors, VecGetArray() returns a pointer to
     the data array.  Otherwise, the routine is implementation dependent.
     - You MUST call VecRestoreArray() when you no longer need access to
     the array.
     */
    PetscCall(DMDAVecGetArray(da, localX, &x));
    if (B) PetscCall(DMDAVecGetArray(da, localB, &b));
    /*
     Get local grid boundaries (for 2-dimensional DMDA):
     xs, ys   - starting grid indices (no ghost points)
     xm, ym   - widths of local grid (no ghost points)
     */
    PetscCall(DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL));

    for (j = ys; j < ys + ym; j++) {
      for (i = xs; i < xs + xm; i++) {
        if (i == 0 || j == 0 || i == Mx - 1 || j == My - 1) {
          /* boundary conditions are all zero Dirichlet */
          x[j][i] = 0.0;
        } else {
          if (B) bij = b[j][i];
          else bij = 0.;

          u  = x[j][i];
          un = x[j - 1][i];
          us = x[j + 1][i];
          ue = x[j][i - 1];
          uw = x[j][i + 1];

          for (k = 0; k < its; k++) {
            eu  = PetscExpScalar(u);
            uxx = (2.0 * u - ue - uw) * hydhx;
            uyy = (2.0 * u - un - us) * hxdhy;
            F   = uxx + uyy - sc * eu - bij;
            if (k == 0) F0 = F;
            J = 2.0 * (hydhx + hxdhy) - sc * eu;
            y = F / J;
            u -= y;
            tot_its++;

            if (atol > PetscAbsReal(PetscRealPart(F)) || rtol * PetscAbsReal(PetscRealPart(F0)) > PetscAbsReal(PetscRealPart(F)) || stol * PetscAbsReal(PetscRealPart(u)) > PetscAbsReal(PetscRealPart(y))) break;
          }
          x[j][i] = u;
        }
      }
    }
    /*
     Restore vector
     */
    PetscCall(DMDAVecRestoreArray(da, localX, &x));
    PetscCall(DMLocalToGlobalBegin(da, localX, INSERT_VALUES, X));
    PetscCall(DMLocalToGlobalEnd(da, localX, INSERT_VALUES, X));
  }
  PetscCall(PetscLogFlops(tot_its * (21.0)));
  PetscCall(DMRestoreLocalVector(da, &localX));
  if (B) {
    PetscCall(DMDAVecRestoreArray(da, localB, &b));
    PetscCall(DMRestoreLocalVector(da, &localB));
  }
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  SNES      snes; /* nonlinear solver */
  Vec       x;    /* solution vector */
  AppCtx    user; /* user-defined work context */
  PetscInt  its;  /* iterations for convergence */
  PetscReal bratu_lambda_max = 6.81;
  PetscReal bratu_lambda_min = 0.;
  PetscInt  MMS              = 1;
  PetscBool flg              = PETSC_FALSE, setMMS;
  DM        da;
  Vec       r = NULL;
  KSP       ksp;
  PetscInt  lits, slits;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize problem parameters
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  user.param = 6.0;
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-par", &user.param, NULL));
  PetscCheck(user.param <= bratu_lambda_max && user.param >= bratu_lambda_min, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Lambda, %g, is out of range, [%g, %g]", (double)user.param, (double)bratu_lambda_min, (double)bratu_lambda_max);
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-mms", &MMS, &setMMS));
  if (MMS == 3) {
    PetscInt mPar = 2, nPar = 1;
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-m_par", &mPar, NULL));
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-n_par", &nPar, NULL));
    user.m = PetscPowInt(2, mPar);
    user.n = PetscPowInt(2, nPar);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create nonlinear solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  PetscCall(SNESSetCountersReset(snes, PETSC_FALSE));
  PetscCall(SNESSetNGS(snes, NonlinearGS, NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array (DMDA) to manage parallel grid and vectors
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, 4, 4, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMDASetUniformCoordinates(da, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0));
  PetscCall(DMSetApplicationContext(da, &user));
  PetscCall(SNESSetDM(snes, da));
  /*  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Extract global vectors from DMDA; then duplicate for remaining
     vectors that are the same types
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMCreateGlobalVector(da, &x));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set local function evaluation routine
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  switch (MMS) {
  case 0:
    user.mms_solution = ZeroBCSolution;
    user.mms_forcing  = NULL;
    break;
  case 1:
    user.mms_solution = MMSSolution1;
    user.mms_forcing  = MMSForcing1;
    break;
  case 2:
    user.mms_solution = MMSSolution2;
    user.mms_forcing  = MMSForcing2;
    break;
  case 3:
    user.mms_solution = MMSSolution3;
    user.mms_forcing  = MMSForcing3;
    break;
  case 4:
    user.mms_solution = MMSSolution4;
    user.mms_forcing  = MMSForcing4;
    break;
  default:
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_USER, "Unknown MMS type %" PetscInt_FMT, MMS);
  }
  PetscCall(DMDASNESSetFunctionLocal(da, INSERT_VALUES, (DMDASNESFunction)FormFunctionLocal, &user));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-fd", &flg, NULL));
  if (!flg) PetscCall(DMDASNESSetJacobianLocal(da, (DMDASNESJacobian)FormJacobianLocal, &user));

  PetscCall(PetscOptionsGetBool(NULL, NULL, "-obj", &flg, NULL));
  if (flg) PetscCall(DMDASNESSetObjectiveLocal(da, (DMDASNESObjective)FormObjectiveLocal, &user));

  if (PetscDefined(HAVE_MATLAB)) {
    PetscBool matlab_function = PETSC_FALSE;
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-matlab_function", &matlab_function, 0));
    if (matlab_function) {
      PetscCall(VecDuplicate(x, &r));
      PetscCall(SNESSetFunction(snes, r, FormFunctionMatlab, &user));
    }
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize nonlinear solver; set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(SNESSetFromOptions(snes));

  PetscCall(FormInitialGuess(da, &user, x));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(SNESSolve(snes, NULL, x));
  PetscCall(SNESGetIterationNumber(snes, &its));

  PetscCall(SNESGetLinearSolveIterations(snes, &slits));
  PetscCall(SNESGetKSP(snes, &ksp));
  PetscCall(KSPGetTotalIterations(ksp, &lits));
  PetscCheck(lits == slits, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Number of total linear iterations reported by SNES %" PetscInt_FMT " does not match reported by KSP %" PetscInt_FMT, slits, lits);
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     If using MMS, check the l_2 error
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  if (setMMS) {
    Vec       e;
    PetscReal errorl2, errorinf;
    PetscInt  N;

    PetscCall(VecDuplicate(x, &e));
    PetscCall(PetscObjectViewFromOptions((PetscObject)x, NULL, "-sol_view"));
    PetscCall(FormExactSolution(da, &user, e));
    PetscCall(PetscObjectViewFromOptions((PetscObject)e, NULL, "-exact_view"));
    PetscCall(VecAXPY(e, -1.0, x));
    PetscCall(PetscObjectViewFromOptions((PetscObject)e, NULL, "-error_view"));
    PetscCall(VecNorm(e, NORM_2, &errorl2));
    PetscCall(VecNorm(e, NORM_INFINITY, &errorinf));
    PetscCall(VecGetSize(e, &N));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "N: %" PetscInt_FMT " error L2 %g inf %g\n", N, (double)(errorl2 / PetscSqrtReal((PetscReal)N)), (double)errorinf));
    PetscCall(VecDestroy(&e));
    PetscCall(PetscLogEventSetDof(SNES_Solve, 0, N));
    PetscCall(PetscLogEventSetError(SNES_Solve, 0, errorl2 / PetscSqrtReal(N)));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecDestroy(&r));
  PetscCall(VecDestroy(&x));
  PetscCall(SNESDestroy(&snes));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
     suffix: asm_0
     requires: !single
     args: -mms 1 -par 0.0 -snes_monitor_short -snes_converged_reason -snes_view -ksp_rtol 1.0e-9 -ksp_monitor_short -ksp_type richardson -pc_type asm -pc_asm_blocks 2 -pc_asm_overlap 0 -pc_asm_local_type additive -sub_pc_type lu

   test:
     suffix: msm_0
     requires: !single
     args: -mms 1 -par 0.0 -snes_monitor_short -snes_converged_reason -snes_view -ksp_rtol 1.0e-9 -ksp_monitor_short -ksp_type richardson -pc_type asm -pc_asm_blocks 2 -pc_asm_overlap 0 -pc_asm_local_type multiplicative -sub_pc_type lu

   test:
     suffix: asm_1
     requires: !single
     args: -mms 1 -par 0.0 -snes_monitor_short -snes_converged_reason -snes_view -ksp_rtol 1.0e-9 -ksp_monitor_short -ksp_type richardson -pc_type asm -pc_asm_blocks 2 -pc_asm_overlap 0 -pc_asm_local_type additive -sub_pc_type lu -da_grid_x 8

   test:
     suffix: msm_1
     requires: !single
     args: -mms 1 -par 0.0 -snes_monitor_short -snes_converged_reason -snes_view -ksp_rtol 1.0e-9 -ksp_monitor_short -ksp_type richardson -pc_type asm -pc_asm_blocks 2 -pc_asm_overlap 0 -pc_asm_local_type multiplicative -sub_pc_type lu -da_grid_x 8

   test:
     suffix: asm_2
     requires: !single
     args: -mms 1 -par 0.0 -snes_monitor_short -snes_converged_reason -snes_view -ksp_rtol 1.0e-9 -ksp_monitor_short -ksp_type richardson -pc_type asm -pc_asm_blocks 3 -pc_asm_overlap 0 -pc_asm_local_type additive -sub_pc_type lu -da_grid_x 8

   test:
     suffix: msm_2
     requires: !single
     args: -mms 1 -par 0.0 -snes_monitor_short -snes_converged_reason -snes_view -ksp_rtol 1.0e-9 -ksp_monitor_short -ksp_type richardson -pc_type asm -pc_asm_blocks 3 -pc_asm_overlap 0 -pc_asm_local_type multiplicative -sub_pc_type lu -da_grid_x 8

   test:
     suffix: asm_3
     requires: !single
     args: -mms 1 -par 0.0 -snes_monitor_short -snes_converged_reason -snes_view -ksp_rtol 1.0e-9 -ksp_monitor_short -ksp_type richardson -pc_type asm -pc_asm_blocks 4 -pc_asm_overlap 0 -pc_asm_local_type additive -sub_pc_type lu -da_grid_x 8

   test:
     suffix: msm_3
     requires: !single
     args: -mms 1 -par 0.0 -snes_monitor_short -snes_converged_reason -snes_view -ksp_rtol 1.0e-9 -ksp_monitor_short -ksp_type richardson -pc_type asm -pc_asm_blocks 4 -pc_asm_overlap 0 -pc_asm_local_type multiplicative -sub_pc_type lu -da_grid_x 8

   test:
     suffix: asm_4
     requires: !single
     nsize: 2
     args: -mms 1 -par 0.0 -snes_monitor_short -snes_converged_reason -snes_view -ksp_rtol 1.0e-9 -ksp_monitor_short -ksp_type richardson -pc_type asm -pc_asm_blocks 2 -pc_asm_overlap 0 -pc_asm_local_type additive -sub_pc_type lu -da_grid_x 8

   test:
     suffix: msm_4
     requires: !single
     nsize: 2
     args: -mms 1 -par 0.0 -snes_monitor_short -snes_converged_reason -snes_view -ksp_rtol 1.0e-9 -ksp_monitor_short -ksp_type richardson -pc_type asm -pc_asm_blocks 2 -pc_asm_overlap 0 -pc_asm_local_type multiplicative -sub_pc_type lu -da_grid_x 8

   test:
     suffix: asm_5
     requires: !single
     nsize: 2
     args: -mms 1 -par 0.0 -snes_monitor_short -snes_converged_reason -snes_view -ksp_rtol 1.0e-9 -ksp_monitor_short -ksp_type richardson -pc_type asm -pc_asm_blocks 4 -pc_asm_overlap 0 -pc_asm_local_type additive -sub_pc_type lu -da_grid_x 8

   test:
     suffix: msm_5
     requires: !single
     nsize: 2
     args: -mms 1 -par 0.0 -snes_monitor_short -snes_converged_reason -snes_view -ksp_rtol 1.0e-9 -ksp_monitor_short -ksp_type richardson -pc_type asm -pc_asm_blocks 4 -pc_asm_overlap 0 -pc_asm_local_type multiplicative -sub_pc_type lu -da_grid_x 8

   test:
     args: -snes_rtol 1.e-5 -pc_type mg -ksp_monitor_short -snes_view -pc_mg_levels 3 -pc_mg_galerkin pmat -da_grid_x 17 -da_grid_y 17 -mg_levels_ksp_monitor_short -mg_levels_ksp_norm_type unpreconditioned -snes_monitor_short -mg_levels_ksp_chebyshev_esteig 0.5,1.1 -mg_levels_pc_type sor -pc_mg_type full
     requires: !single

   test:
     suffix: 2
     args: -pc_type mg -ksp_converged_reason -snes_view -pc_mg_galerkin pmat -snes_grid_sequence 3 -mg_levels_ksp_norm_type unpreconditioned -snes_monitor_short -mg_levels_ksp_chebyshev_esteig 0.5,1.1 -mg_levels_pc_type sor -pc_mg_type full -ksp_atol -1.

   test:
     suffix: 3
     nsize: 2
     args: -snes_grid_sequence 2 -snes_mf_operator -snes_converged_reason -snes_view -pc_type mg -snes_atol -1 -snes_rtol 1.e-2
     filter: grep -v "otal number of function evaluations"

   test:
     suffix: 4
     nsize: 2
     args: -snes_grid_sequence 2 -snes_monitor_short -ksp_converged_reason -snes_converged_reason -snes_view -pc_type mg -snes_atol -1 -ksp_atol -1

   test:
     suffix: 5_anderson
     args: -da_grid_x 81 -da_grid_y 81 -snes_monitor_short -snes_max_it 50 -par 6.0 -snes_type anderson

   test:
     suffix: 5_aspin
     nsize: 4
     args: -snes_monitor_short -ksp_monitor_short -snes_converged_reason -da_refine 4 -da_overlap 3 -snes_type aspin -snes_view

   test:
     suffix: 5_broyden
     args: -da_grid_x 81 -da_grid_y 81 -snes_monitor_short -snes_max_it 50 -par 6.0 -snes_type qn -snes_qn_type broyden -snes_qn_m 10

   test:
     suffix: 5_fas
     args: -fas_coarse_snes_max_it 1 -fas_coarse_pc_type lu -fas_coarse_ksp_type preonly -snes_monitor_short -snes_type fas -fas_coarse_ksp_type richardson -da_refine 6
     requires: !single

   test:
     suffix: 5_fas_additive
     args: -fas_coarse_snes_max_it 1 -fas_coarse_pc_type lu -fas_coarse_ksp_type preonly -snes_monitor_short -snes_type fas -fas_coarse_ksp_type richardson -da_refine 6 -snes_fas_type additive -snes_max_it 50

   test:
     suffix: 5_fas_monitor
     args: -da_refine 1 -snes_type fas -snes_fas_monitor
     requires: !single

   test:
     suffix: 5_ls
     args: -da_grid_x 81 -da_grid_y 81 -snes_monitor_short -snes_max_it 50 -par 6.0 -snes_type newtonls

   test:
     suffix: 5_ls_sell_sor
     args: -da_grid_x 81 -da_grid_y 81 -snes_monitor_short -snes_max_it 50 -par 6.0 -snes_type newtonls -dm_mat_type sell -pc_type sor
     output_file: output/ex5_5_ls.out

   test:
     suffix: 5_nasm
     nsize: 4
     args: -snes_monitor_short -snes_converged_reason -da_refine 4 -da_overlap 3 -snes_type nasm -snes_nasm_type restrict -snes_max_it 10

   test:
     suffix: 5_ncg
     args: -da_grid_x 81 -da_grid_y 81 -snes_monitor_short -snes_max_it 50 -par 6.0 -snes_type ncg -snes_ncg_type fr

   test:
     suffix: 5_newton_asm_dmda
     nsize: 4
     args: -snes_monitor_short -ksp_monitor_short -snes_converged_reason -da_refine 4 -da_overlap 3 -snes_type newtonls -pc_type asm -pc_asm_dm_subdomains -malloc_dump
     requires: !single

   test:
     suffix: 5_newton_gasm_dmda
     nsize: 4
     args: -snes_monitor_short -ksp_monitor_short -snes_converged_reason -da_refine 4 -da_overlap 3 -snes_type newtonls -pc_type gasm -malloc_dump
     requires: !single

   test:
     suffix: 5_ngmres
     args: -da_grid_x 81 -da_grid_y 81 -snes_monitor_short -snes_max_it 50 -par 6.0 -snes_type ngmres -snes_ngmres_m 10

   test:
     suffix: 5_ngmres_fas
     args: -snes_rtol 1.e-4 -snes_type ngmres -npc_fas_coarse_snes_max_it 1 -npc_fas_coarse_snes_type newtonls -npc_fas_coarse_pc_type lu -npc_fas_coarse_ksp_type preonly -snes_ngmres_m 10 -snes_monitor_short -npc_snes_max_it 1 -npc_snes_type fas -npc_fas_coarse_ksp_type richardson -da_refine 6

   test:
     suffix: 5_ngmres_ngs
     args: -da_grid_x 81 -da_grid_y 81 -snes_monitor_short -snes_max_it 50 -par 6.0 -snes_type ngmres -npc_snes_type ngs -npc_snes_max_it 1

   test:
     suffix: 5_ngmres_nrichardson
     args: -da_grid_x 81 -da_grid_y 81 -snes_monitor_short -snes_max_it 50 -par 6.0 -snes_type ngmres -snes_ngmres_m 10 -npc_snes_type nrichardson -npc_snes_max_it 3
     output_file: output/ex5_5_ngmres_richardson.out

   test:
     suffix: 5_nrichardson
     args: -da_grid_x 81 -da_grid_y 81 -snes_monitor_short -snes_max_it 50 -par 6.0 -snes_type nrichardson

   test:
     suffix: 5_qn
     args: -da_grid_x 81 -da_grid_y 81 -snes_monitor_short -snes_max_it 50 -par 6.0 -snes_type qn -snes_linesearch_type cp -snes_qn_m 10

   test:
     suffix: 6
     nsize: 4
     args: -snes_converged_reason -ksp_converged_reason -da_grid_x 129 -da_grid_y 129 -pc_type mg -pc_mg_levels 8 -mg_levels_ksp_type chebyshev -mg_levels_ksp_chebyshev_esteig 0,0.5,0,1.1 -mg_levels_ksp_max_it 2

   test:
     requires: complex !single
     suffix: complex
     args: -snes_mf_operator -mat_mffd_complex -snes_monitor

TEST*/
