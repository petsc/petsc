#include <petscdmda.h>
#include <petsctao.h>

static char help[] = "This example demonstrates use of the TAO package to \n\
solve an unconstrained minimization problem.  This example is based on a \n\
problem from the MINPACK-2 test suite.  Given a rectangular 2-D domain, \n\
boundary values along the edges of the domain, and a plate represented by \n\
lower boundary conditions, the objective is to find the\n\
surface with the minimal area that satisfies the boundary conditions.\n\
The command line options are:\n\
  -mx <xg>, where <xg> = number of grid points in the 1st coordinate direction\n\
  -my <yg>, where <yg> = number of grid points in the 2nd coordinate direction\n\
  -bmx <bxg>, where <bxg> = number of grid points under plate in 1st direction\n\
  -bmy <byg>, where <byg> = number of grid points under plate in 2nd direction\n\
  -bheight <ht>, where <ht> = height of the plate\n\
  -start <st>, where <st> =0 for zero vector, <st> >0 for random start, and <st> <0 \n\
               for an average of the boundary conditions\n\n";

/*
   User-defined application context - contains data needed by the
   application-provided call-back routines, FormFunctionGradient(),
   FormHessian().
*/
typedef struct {
  /* problem parameters */
  PetscReal bheight;                  /* Height of plate under the surface */
  PetscInt  mx, my;                   /* discretization in x, y directions */
  PetscInt  bmx, bmy;                 /* Size of plate under the surface */
  Vec       Bottom, Top, Left, Right; /* boundary values */

  /* Working space */
  Vec localX, localV; /* ghosted local vector */
  DM  dm;             /* distributed array data structure */
  Mat H;
} AppCtx;

/* -------- User-defined Routines --------- */

static PetscErrorCode MSA_BoundaryConditions(AppCtx *);
static PetscErrorCode MSA_InitialPoint(AppCtx *, Vec);
static PetscErrorCode MSA_Plate(Vec, Vec, void *);
PetscErrorCode        FormFunctionGradient(Tao, Vec, PetscReal *, Vec, void *);
PetscErrorCode        FormHessian(Tao, Vec, Mat, Mat, void *);

/* For testing matrix free submatrices */
PetscErrorCode MatrixFreeHessian(Tao, Vec, Mat, Mat, void *);
PetscErrorCode MyMatMult(Mat, Vec, Vec);

int main(int argc, char **argv)
{
  PetscInt               Nx, Ny;    /* number of processors in x- and y- directions */
  PetscInt               m, N;      /* number of local and global elements in vectors */
  Vec                    x, xl, xu; /* solution vector  and bounds*/
  PetscBool              flg;       /* A return variable when checking for user options */
  Tao                    tao;       /* Tao solver context */
  ISLocalToGlobalMapping isltog;    /* local-to-global mapping object */
  Mat                    H_shell;   /* to test matrix-free submatrices */
  AppCtx                 user;      /* user-defined work context */

  /* Initialize PETSc, TAO */
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));

  /* Specify default dimension of the problem */
  user.mx      = 10;
  user.my      = 10;
  user.bheight = 0.1;

  /* Check for any command line arguments that override defaults */
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-mx", &user.mx, &flg));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-my", &user.my, &flg));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-bheight", &user.bheight, &flg));

  user.bmx = user.mx / 2;
  user.bmy = user.my / 2;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-bmx", &user.bmx, &flg));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-bmy", &user.bmy, &flg));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n---- Minimum Surface Area With Plate Problem -----\n"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "mx:%" PetscInt_FMT ", my:%" PetscInt_FMT ", bmx:%" PetscInt_FMT ", bmy:%" PetscInt_FMT ", height:%g\n", user.mx, user.my, user.bmx, user.bmy, (double)user.bheight));

  /* Calculate any derived values from parameters */
  N = user.mx * user.my;

  /* Let Petsc determine the dimensions of the local vectors */
  Nx = PETSC_DECIDE;
  Ny = PETSC_DECIDE;

  /*
     A two dimensional distributed array will help define this problem,
     which derives from an elliptic PDE on two dimensional domain.  From
     the distributed array, Create the vectors.
  */
  PetscCall(DMDACreate2d(MPI_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, user.mx, user.my, Nx, Ny, 1, 1, NULL, NULL, &user.dm));
  PetscCall(DMSetFromOptions(user.dm));
  PetscCall(DMSetUp(user.dm));
  /*
     Extract global and local vectors from DM; The local vectors are
     used solely as work space for the evaluation of the function,
     gradient, and Hessian.  Duplicate for remaining vectors that are
     the same types.
  */
  PetscCall(DMCreateGlobalVector(user.dm, &x)); /* Solution */
  PetscCall(DMCreateLocalVector(user.dm, &user.localX));
  PetscCall(VecDuplicate(user.localX, &user.localV));

  PetscCall(VecDuplicate(x, &xl));
  PetscCall(VecDuplicate(x, &xu));

  /* The TAO code begins here */

  /*
     Create TAO solver and set desired solution method
     The method must either be TAOTRON or TAOBLMVM
     If TAOBLMVM is used, then hessian function is not called.
  */
  PetscCall(TaoCreate(PETSC_COMM_WORLD, &tao));
  PetscCall(TaoSetType(tao, TAOBLMVM));

  /* Set initial solution guess; */
  PetscCall(MSA_BoundaryConditions(&user));
  PetscCall(MSA_InitialPoint(&user, x));
  PetscCall(TaoSetSolution(tao, x));

  /* Set routines for function, gradient and hessian evaluation */
  PetscCall(TaoSetObjectiveAndGradient(tao, NULL, FormFunctionGradient, (void *)&user));

  PetscCall(VecGetLocalSize(x, &m));
  PetscCall(MatCreateAIJ(MPI_COMM_WORLD, m, m, N, N, 7, NULL, 3, NULL, &(user.H)));
  PetscCall(MatSetOption(user.H, MAT_SYMMETRIC, PETSC_TRUE));

  PetscCall(DMGetLocalToGlobalMapping(user.dm, &isltog));
  PetscCall(MatSetLocalToGlobalMapping(user.H, isltog, isltog));
  PetscCall(PetscOptionsHasName(NULL, NULL, "-matrixfree", &flg));
  if (flg) {
    PetscCall(MatCreateShell(PETSC_COMM_WORLD, m, m, N, N, (void *)&user, &H_shell));
    PetscCall(MatShellSetOperation(H_shell, MATOP_MULT, (void (*)(void))MyMatMult));
    PetscCall(MatSetOption(H_shell, MAT_SYMMETRIC, PETSC_TRUE));
    PetscCall(TaoSetHessian(tao, H_shell, H_shell, MatrixFreeHessian, (void *)&user));
  } else {
    PetscCall(TaoSetHessian(tao, user.H, user.H, FormHessian, (void *)&user));
  }

  /* Set Variable bounds */
  PetscCall(MSA_Plate(xl, xu, (void *)&user));
  PetscCall(TaoSetVariableBounds(tao, xl, xu));

  /* Check for any tao command line options */
  PetscCall(TaoSetFromOptions(tao));

  /* SOLVE THE APPLICATION */
  PetscCall(TaoSolve(tao));

  PetscCall(TaoView(tao, PETSC_VIEWER_STDOUT_WORLD));

  /* Free TAO data structures */
  PetscCall(TaoDestroy(&tao));

  /* Free PETSc data structures */
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&xl));
  PetscCall(VecDestroy(&xu));
  PetscCall(MatDestroy(&user.H));
  PetscCall(VecDestroy(&user.localX));
  PetscCall(VecDestroy(&user.localV));
  PetscCall(VecDestroy(&user.Bottom));
  PetscCall(VecDestroy(&user.Top));
  PetscCall(VecDestroy(&user.Left));
  PetscCall(VecDestroy(&user.Right));
  PetscCall(DMDestroy(&user.dm));
  if (flg) PetscCall(MatDestroy(&H_shell));
  PetscCall(PetscFinalize());
  return 0;
}

/*  FormFunctionGradient - Evaluates f(x) and gradient g(x).

    Input Parameters:
.   tao     - the Tao context
.   X      - input vector
.   userCtx - optional user-defined context, as set by TaoSetObjectiveAndGradient()

    Output Parameters:
.   fcn     - the function value
.   G      - vector containing the newly evaluated gradient

   Notes:
   In this case, we discretize the domain and Create triangles. The
   surface of each triangle is planar, whose surface area can be easily
   computed.  The total surface area is found by sweeping through the grid
   and computing the surface area of the two triangles that have their
   right angle at the grid point.  The diagonal line segments on the
   grid that define the triangles run from top left to lower right.
   The numbering of points starts at the lower left and runs left to
   right, then bottom to top.
*/
PetscErrorCode FormFunctionGradient(Tao tao, Vec X, PetscReal *fcn, Vec G, void *userCtx)
{
  AppCtx    *user = (AppCtx *)userCtx;
  PetscInt   i, j, row;
  PetscInt   mx = user->mx, my = user->my;
  PetscInt   xs, xm, gxs, gxm, ys, ym, gys, gym;
  PetscReal  ft   = 0;
  PetscReal  zero = 0.0;
  PetscReal  hx = 1.0 / (mx + 1), hy = 1.0 / (my + 1), hydhx = hy / hx, hxdhy = hx / hy, area = 0.5 * hx * hy;
  PetscReal  rhx = mx + 1, rhy = my + 1;
  PetscReal  f1, f2, f3, f4, f5, f6, d1, d2, d3, d4, d5, d6, d7, d8, xc, xl, xr, xt, xb, xlt, xrb;
  PetscReal  df1dxc, df2dxc, df3dxc, df4dxc, df5dxc, df6dxc;
  PetscReal *g, *x, *left, *right, *bottom, *top;
  Vec        localX = user->localX, localG = user->localV;

  /* Get local mesh boundaries */
  PetscCall(DMDAGetCorners(user->dm, &xs, &ys, NULL, &xm, &ym, NULL));
  PetscCall(DMDAGetGhostCorners(user->dm, &gxs, &gys, NULL, &gxm, &gym, NULL));

  /* Scatter ghost points to local vector */
  PetscCall(DMGlobalToLocalBegin(user->dm, X, INSERT_VALUES, localX));
  PetscCall(DMGlobalToLocalEnd(user->dm, X, INSERT_VALUES, localX));

  /* Initialize vector to zero */
  PetscCall(VecSet(localG, zero));

  /* Get pointers to vector data */
  PetscCall(VecGetArray(localX, &x));
  PetscCall(VecGetArray(localG, &g));
  PetscCall(VecGetArray(user->Top, &top));
  PetscCall(VecGetArray(user->Bottom, &bottom));
  PetscCall(VecGetArray(user->Left, &left));
  PetscCall(VecGetArray(user->Right, &right));

  /* Compute function over the locally owned part of the mesh */
  for (j = ys; j < ys + ym; j++) {
    for (i = xs; i < xs + xm; i++) {
      row = (j - gys) * gxm + (i - gxs);

      xc  = x[row];
      xlt = xrb = xl = xr = xb = xt = xc;

      if (i == 0) { /* left side */
        xl  = left[j - ys + 1];
        xlt = left[j - ys + 2];
      } else {
        xl = x[row - 1];
      }

      if (j == 0) { /* bottom side */
        xb  = bottom[i - xs + 1];
        xrb = bottom[i - xs + 2];
      } else {
        xb = x[row - gxm];
      }

      if (i + 1 == gxs + gxm) { /* right side */
        xr  = right[j - ys + 1];
        xrb = right[j - ys];
      } else {
        xr = x[row + 1];
      }

      if (j + 1 == gys + gym) { /* top side */
        xt  = top[i - xs + 1];
        xlt = top[i - xs];
      } else {
        xt = x[row + gxm];
      }

      if (i > gxs && j + 1 < gys + gym) xlt = x[row - 1 + gxm];
      if (j > gys && i + 1 < gxs + gxm) xrb = x[row + 1 - gxm];

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

      d1 *= rhx;
      d2 *= rhx;
      d3 *= rhy;
      d4 *= rhy;
      d5 *= rhy;
      d6 *= rhx;
      d7 *= rhy;
      d8 *= rhx;

      f1 = PetscSqrtScalar(1.0 + d1 * d1 + d7 * d7);
      f2 = PetscSqrtScalar(1.0 + d1 * d1 + d4 * d4);
      f3 = PetscSqrtScalar(1.0 + d3 * d3 + d8 * d8);
      f4 = PetscSqrtScalar(1.0 + d3 * d3 + d2 * d2);
      f5 = PetscSqrtScalar(1.0 + d2 * d2 + d5 * d5);
      f6 = PetscSqrtScalar(1.0 + d4 * d4 + d6 * d6);

      ft = ft + (f2 + f4);

      df1dxc /= f1;
      df2dxc /= f2;
      df3dxc /= f3;
      df4dxc /= f4;
      df5dxc /= f5;
      df6dxc /= f6;

      g[row] = (df1dxc + df2dxc + df3dxc + df4dxc + df5dxc + df6dxc) * 0.5;
    }
  }

  /* Compute triangular areas along the border of the domain. */
  if (xs == 0) { /* left side */
    for (j = ys; j < ys + ym; j++) {
      d3 = (left[j - ys + 1] - left[j - ys + 2]) * rhy;
      d2 = (left[j - ys + 1] - x[(j - gys) * gxm]) * rhx;
      ft = ft + PetscSqrtScalar(1.0 + d3 * d3 + d2 * d2);
    }
  }
  if (ys == 0) { /* bottom side */
    for (i = xs; i < xs + xm; i++) {
      d2 = (bottom[i + 1 - xs] - bottom[i - xs + 2]) * rhx;
      d3 = (bottom[i - xs + 1] - x[i - gxs]) * rhy;
      ft = ft + PetscSqrtScalar(1.0 + d3 * d3 + d2 * d2);
    }
  }

  if (xs + xm == mx) { /* right side */
    for (j = ys; j < ys + ym; j++) {
      d1 = (x[(j + 1 - gys) * gxm - 1] - right[j - ys + 1]) * rhx;
      d4 = (right[j - ys] - right[j - ys + 1]) * rhy;
      ft = ft + PetscSqrtScalar(1.0 + d1 * d1 + d4 * d4);
    }
  }
  if (ys + ym == my) { /* top side */
    for (i = xs; i < xs + xm; i++) {
      d1 = (x[(gym - 1) * gxm + i - gxs] - top[i - xs + 1]) * rhy;
      d4 = (top[i - xs + 1] - top[i - xs]) * rhx;
      ft = ft + PetscSqrtScalar(1.0 + d1 * d1 + d4 * d4);
    }
  }

  if (ys == 0 && xs == 0) {
    d1 = (left[0] - left[1]) * rhy;
    d2 = (bottom[0] - bottom[1]) * rhx;
    ft += PetscSqrtScalar(1.0 + d1 * d1 + d2 * d2);
  }
  if (ys + ym == my && xs + xm == mx) {
    d1 = (right[ym + 1] - right[ym]) * rhy;
    d2 = (top[xm + 1] - top[xm]) * rhx;
    ft += PetscSqrtScalar(1.0 + d1 * d1 + d2 * d2);
  }

  ft = ft * area;
  PetscCallMPI(MPI_Allreduce(&ft, fcn, 1, MPIU_REAL, MPIU_SUM, MPI_COMM_WORLD));

  /* Restore vectors */
  PetscCall(VecRestoreArray(localX, &x));
  PetscCall(VecRestoreArray(localG, &g));
  PetscCall(VecRestoreArray(user->Left, &left));
  PetscCall(VecRestoreArray(user->Top, &top));
  PetscCall(VecRestoreArray(user->Bottom, &bottom));
  PetscCall(VecRestoreArray(user->Right, &right));

  /* Scatter values to global vector */
  PetscCall(DMLocalToGlobalBegin(user->dm, localG, INSERT_VALUES, G));
  PetscCall(DMLocalToGlobalEnd(user->dm, localG, INSERT_VALUES, G));

  PetscCall(PetscLogFlops(70.0 * xm * ym));

  return 0;
}

/* ------------------------------------------------------------------- */
/*
   FormHessian - Evaluates Hessian matrix.

   Input Parameters:
.  tao  - the Tao context
.  x    - input vector
.  ptr  - optional user-defined context, as set by TaoSetHessian()

   Output Parameters:
.  A    - Hessian matrix
.  B    - optionally different preconditioning matrix

   Notes:
   Due to mesh point reordering with DMs, we must always work
   with the local mesh points, and then transform them to the new
   global numbering with the local-to-global mapping.  We cannot work
   directly with the global numbers for the original uniprocessor mesh!

   Two methods are available for imposing this transformation
   when setting matrix entries:
     (A) MatSetValuesLocal(), using the local ordering (including
         ghost points!)
         - Do the following two steps once, before calling TaoSolve()
           - Use DMGetISLocalToGlobalMapping() to extract the
             local-to-global map from the DM
           - Associate this map with the matrix by calling
             MatSetLocalToGlobalMapping()
         - Then set matrix entries using the local ordering
           by calling MatSetValuesLocal()
     (B) MatSetValues(), using the global ordering
         - Use DMGetGlobalIndices() to extract the local-to-global map
         - Then apply this map explicitly yourself
         - Set matrix entries using the global ordering by calling
           MatSetValues()
   Option (A) seems cleaner/easier in many cases, and is the procedure
   used in this example.
*/
PetscErrorCode FormHessian(Tao tao, Vec X, Mat Hptr, Mat Hessian, void *ptr)
{
  AppCtx    *user = (AppCtx *)ptr;
  PetscInt   i, j, k, row;
  PetscInt   mx = user->mx, my = user->my;
  PetscInt   xs, xm, gxs, gxm, ys, ym, gys, gym, col[7];
  PetscReal  hx = 1.0 / (mx + 1), hy = 1.0 / (my + 1), hydhx = hy / hx, hxdhy = hx / hy;
  PetscReal  rhx = mx + 1, rhy = my + 1;
  PetscReal  f1, f2, f3, f4, f5, f6, d1, d2, d3, d4, d5, d6, d7, d8, xc, xl, xr, xt, xb, xlt, xrb;
  PetscReal  hl, hr, ht, hb, hc, htl, hbr;
  PetscReal *x, *left, *right, *bottom, *top;
  PetscReal  v[7];
  Vec        localX = user->localX;
  PetscBool  assembled;

  /* Set various matrix options */
  PetscCall(MatSetOption(Hessian, MAT_IGNORE_OFF_PROC_ENTRIES, PETSC_TRUE));

  /* Initialize matrix entries to zero */
  PetscCall(MatAssembled(Hessian, &assembled));
  if (assembled) PetscCall(MatZeroEntries(Hessian));

  /* Get local mesh boundaries */
  PetscCall(DMDAGetCorners(user->dm, &xs, &ys, NULL, &xm, &ym, NULL));
  PetscCall(DMDAGetGhostCorners(user->dm, &gxs, &gys, NULL, &gxm, &gym, NULL));

  /* Scatter ghost points to local vector */
  PetscCall(DMGlobalToLocalBegin(user->dm, X, INSERT_VALUES, localX));
  PetscCall(DMGlobalToLocalEnd(user->dm, X, INSERT_VALUES, localX));

  /* Get pointers to vector data */
  PetscCall(VecGetArray(localX, &x));
  PetscCall(VecGetArray(user->Top, &top));
  PetscCall(VecGetArray(user->Bottom, &bottom));
  PetscCall(VecGetArray(user->Left, &left));
  PetscCall(VecGetArray(user->Right, &right));

  /* Compute Hessian over the locally owned part of the mesh */

  for (i = xs; i < xs + xm; i++) {
    for (j = ys; j < ys + ym; j++) {
      row = (j - gys) * gxm + (i - gxs);

      xc  = x[row];
      xlt = xrb = xl = xr = xb = xt = xc;

      /* Left side */
      if (i == gxs) {
        xl  = left[j - ys + 1];
        xlt = left[j - ys + 2];
      } else {
        xl = x[row - 1];
      }

      if (j == gys) {
        xb  = bottom[i - xs + 1];
        xrb = bottom[i - xs + 2];
      } else {
        xb = x[row - gxm];
      }

      if (i + 1 == gxs + gxm) {
        xr  = right[j - ys + 1];
        xrb = right[j - ys];
      } else {
        xr = x[row + 1];
      }

      if (j + 1 == gys + gym) {
        xt  = top[i - xs + 1];
        xlt = top[i - xs];
      } else {
        xt = x[row + gxm];
      }

      if (i > gxs && j + 1 < gys + gym) xlt = x[row - 1 + gxm];
      if (j > gys && i + 1 < gxs + gxm) xrb = x[row + 1 - gxm];

      d1 = (xc - xl) * rhx;
      d2 = (xc - xr) * rhx;
      d3 = (xc - xt) * rhy;
      d4 = (xc - xb) * rhy;
      d5 = (xrb - xr) * rhy;
      d6 = (xrb - xb) * rhx;
      d7 = (xlt - xl) * rhy;
      d8 = (xlt - xt) * rhx;

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

      hc = hydhx * (1.0 + d7 * d7) / (f1 * f1 * f1) + hxdhy * (1.0 + d8 * d8) / (f3 * f3 * f3) + hydhx * (1.0 + d5 * d5) / (f5 * f5 * f5) + hxdhy * (1.0 + d6 * d6) / (f6 * f6 * f6) + (hxdhy * (1.0 + d1 * d1) + hydhx * (1.0 + d4 * d4) - 2 * d1 * d4) / (f2 * f2 * f2) + (hxdhy * (1.0 + d2 * d2) + hydhx * (1.0 + d3 * d3) - 2 * d2 * d3) / (f4 * f4 * f4);

      hl *= 0.5;
      hr *= 0.5;
      ht *= 0.5;
      hb *= 0.5;
      hbr *= 0.5;
      htl *= 0.5;
      hc *= 0.5;

      k = 0;
      if (j > 0) {
        v[k]   = hb;
        col[k] = row - gxm;
        k++;
      }

      if (j > 0 && i < mx - 1) {
        v[k]   = hbr;
        col[k] = row - gxm + 1;
        k++;
      }

      if (i > 0) {
        v[k]   = hl;
        col[k] = row - 1;
        k++;
      }

      v[k]   = hc;
      col[k] = row;
      k++;

      if (i < mx - 1) {
        v[k]   = hr;
        col[k] = row + 1;
        k++;
      }

      if (i > 0 && j < my - 1) {
        v[k]   = htl;
        col[k] = row + gxm - 1;
        k++;
      }

      if (j < my - 1) {
        v[k]   = ht;
        col[k] = row + gxm;
        k++;
      }

      /*
         Set matrix values using local numbering, which was defined
         earlier, in the main routine.
      */
      PetscCall(MatSetValuesLocal(Hessian, 1, &row, k, col, v, INSERT_VALUES));
    }
  }

  /* Restore vectors */
  PetscCall(VecRestoreArray(localX, &x));
  PetscCall(VecRestoreArray(user->Left, &left));
  PetscCall(VecRestoreArray(user->Top, &top));
  PetscCall(VecRestoreArray(user->Bottom, &bottom));
  PetscCall(VecRestoreArray(user->Right, &right));

  /* Assemble the matrix */
  PetscCall(MatAssemblyBegin(Hessian, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Hessian, MAT_FINAL_ASSEMBLY));

  PetscCall(PetscLogFlops(199.0 * xm * ym));
  return 0;
}

/* ------------------------------------------------------------------- */
/*
   MSA_BoundaryConditions -  Calculates the boundary conditions for
   the region.

   Input Parameter:
.  user - user-defined application context

   Output Parameter:
.  user - user-defined application context
*/
static PetscErrorCode MSA_BoundaryConditions(AppCtx *user)
{
  PetscInt   i, j, k, maxits = 5, limit = 0;
  PetscInt   xs, ys, xm, ym, gxs, gys, gxm, gym;
  PetscInt   mx = user->mx, my = user->my;
  PetscInt   bsize = 0, lsize = 0, tsize = 0, rsize = 0;
  PetscReal  one = 1.0, two = 2.0, three = 3.0, scl = 1.0, tol = 1e-10;
  PetscReal  fnorm, det, hx, hy, xt = 0, yt = 0;
  PetscReal  u1, u2, nf1, nf2, njac11, njac12, njac21, njac22;
  PetscReal  b = -0.5, t = 0.5, l = -0.5, r = 0.5;
  PetscReal *boundary;
  PetscBool  flg;
  Vec        Bottom, Top, Right, Left;

  /* Get local mesh boundaries */
  PetscCall(DMDAGetCorners(user->dm, &xs, &ys, NULL, &xm, &ym, NULL));
  PetscCall(DMDAGetGhostCorners(user->dm, &gxs, &gys, NULL, &gxm, &gym, NULL));

  bsize = xm + 2;
  lsize = ym + 2;
  rsize = ym + 2;
  tsize = xm + 2;

  PetscCall(VecCreateMPI(MPI_COMM_WORLD, bsize, PETSC_DECIDE, &Bottom));
  PetscCall(VecCreateMPI(MPI_COMM_WORLD, tsize, PETSC_DECIDE, &Top));
  PetscCall(VecCreateMPI(MPI_COMM_WORLD, lsize, PETSC_DECIDE, &Left));
  PetscCall(VecCreateMPI(MPI_COMM_WORLD, rsize, PETSC_DECIDE, &Right));

  user->Top    = Top;
  user->Left   = Left;
  user->Bottom = Bottom;
  user->Right  = Right;

  hx = (r - l) / (mx + 1);
  hy = (t - b) / (my + 1);

  for (j = 0; j < 4; j++) {
    if (j == 0) {
      yt    = b;
      xt    = l + hx * xs;
      limit = bsize;
      VecGetArray(Bottom, &boundary);
    } else if (j == 1) {
      yt    = t;
      xt    = l + hx * xs;
      limit = tsize;
      VecGetArray(Top, &boundary);
    } else if (j == 2) {
      yt    = b + hy * ys;
      xt    = l;
      limit = lsize;
      VecGetArray(Left, &boundary);
    } else if (j == 3) {
      yt    = b + hy * ys;
      xt    = r;
      limit = rsize;
      VecGetArray(Right, &boundary);
    }

    for (i = 0; i < limit; i++) {
      u1 = xt;
      u2 = -yt;
      for (k = 0; k < maxits; k++) {
        nf1   = u1 + u1 * u2 * u2 - u1 * u1 * u1 / three - xt;
        nf2   = -u2 - u1 * u1 * u2 + u2 * u2 * u2 / three - yt;
        fnorm = PetscSqrtScalar(nf1 * nf1 + nf2 * nf2);
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
      if (j == 0 || j == 1) {
        xt = xt + hx;
      } else if (j == 2 || j == 3) {
        yt = yt + hy;
      }
    }
    if (j == 0) {
      PetscCall(VecRestoreArray(Bottom, &boundary));
    } else if (j == 1) {
      PetscCall(VecRestoreArray(Top, &boundary));
    } else if (j == 2) {
      PetscCall(VecRestoreArray(Left, &boundary));
    } else if (j == 3) {
      PetscCall(VecRestoreArray(Right, &boundary));
    }
  }

  /* Scale the boundary if desired */

  PetscCall(PetscOptionsGetReal(NULL, NULL, "-bottom", &scl, &flg));
  if (flg) PetscCall(VecScale(Bottom, scl));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-top", &scl, &flg));
  if (flg) PetscCall(VecScale(Top, scl));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-right", &scl, &flg));
  if (flg) PetscCall(VecScale(Right, scl));

  PetscCall(PetscOptionsGetReal(NULL, NULL, "-left", &scl, &flg));
  if (flg) PetscCall(VecScale(Left, scl));
  return 0;
}

/* ------------------------------------------------------------------- */
/*
   MSA_Plate -  Calculates an obstacle for surface to stretch over.

   Input Parameter:
.  user - user-defined application context

   Output Parameter:
.  user - user-defined application context
*/
static PetscErrorCode MSA_Plate(Vec XL, Vec XU, void *ctx)
{
  AppCtx    *user = (AppCtx *)ctx;
  PetscInt   i, j, row;
  PetscInt   xs, ys, xm, ym;
  PetscInt   mx = user->mx, my = user->my, bmy, bmx;
  PetscReal  t1, t2, t3;
  PetscReal *xl, lb = PETSC_NINFINITY, ub = PETSC_INFINITY;
  PetscBool  cylinder;

  user->bmy = PetscMax(0, user->bmy);
  user->bmy = PetscMin(my, user->bmy);
  user->bmx = PetscMax(0, user->bmx);
  user->bmx = PetscMin(mx, user->bmx);
  bmy       = user->bmy;
  bmx       = user->bmx;

  PetscCall(DMDAGetCorners(user->dm, &xs, &ys, NULL, &xm, &ym, NULL));

  PetscCall(VecSet(XL, lb));
  PetscCall(VecSet(XU, ub));

  PetscCall(VecGetArray(XL, &xl));

  PetscCall(PetscOptionsHasName(NULL, NULL, "-cylinder", &cylinder));
  /* Compute the optional lower box */
  if (cylinder) {
    for (i = xs; i < xs + xm; i++) {
      for (j = ys; j < ys + ym; j++) {
        row = (j - ys) * xm + (i - xs);
        t1  = (2.0 * i - mx) * bmy;
        t2  = (2.0 * j - my) * bmx;
        t3  = bmx * bmx * bmy * bmy;
        if (t1 * t1 + t2 * t2 <= t3) xl[row] = user->bheight;
      }
    }
  } else {
    /* Compute the optional lower box */
    for (i = xs; i < xs + xm; i++) {
      for (j = ys; j < ys + ym; j++) {
        row = (j - ys) * xm + (i - xs);
        if (i >= (mx - bmx) / 2 && i < mx - (mx - bmx) / 2 && j >= (my - bmy) / 2 && j < my - (my - bmy) / 2) xl[row] = user->bheight;
      }
    }
  }
  PetscCall(VecRestoreArray(XL, &xl));

  return 0;
}

/* ------------------------------------------------------------------- */
/*
   MSA_InitialPoint - Calculates the initial guess in one of three ways.

   Input Parameters:
.  user - user-defined application context
.  X - vector for initial guess

   Output Parameters:
.  X - newly computed initial guess
*/
static PetscErrorCode MSA_InitialPoint(AppCtx *user, Vec X)
{
  PetscInt  start = -1, i, j;
  PetscReal zero  = 0.0;
  PetscBool flg;

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-start", &start, &flg));
  if (flg && start == 0) { /* The zero vector is reasonable */
    PetscCall(VecSet(X, zero));
  } else if (flg && start > 0) { /* Try a random start between -0.5 and 0.5 */
    PetscRandom rctx;
    PetscReal   np5 = -0.5;

    PetscCall(PetscRandomCreate(MPI_COMM_WORLD, &rctx));
    for (i = 0; i < start; i++) PetscCall(VecSetRandom(X, rctx));
    PetscCall(PetscRandomDestroy(&rctx));
    PetscCall(VecShift(X, np5));

  } else { /* Take an average of the boundary conditions */

    PetscInt   row, xs, xm, gxs, gxm, ys, ym, gys, gym;
    PetscInt   mx = user->mx, my = user->my;
    PetscReal *x, *left, *right, *bottom, *top;
    Vec        localX = user->localX;

    /* Get local mesh boundaries */
    PetscCall(DMDAGetCorners(user->dm, &xs, &ys, NULL, &xm, &ym, NULL));
    PetscCall(DMDAGetGhostCorners(user->dm, &gxs, &gys, NULL, &gxm, &gym, NULL));

    /* Get pointers to vector data */
    PetscCall(VecGetArray(user->Top, &top));
    PetscCall(VecGetArray(user->Bottom, &bottom));
    PetscCall(VecGetArray(user->Left, &left));
    PetscCall(VecGetArray(user->Right, &right));

    PetscCall(VecGetArray(localX, &x));
    /* Perform local computations */
    for (j = ys; j < ys + ym; j++) {
      for (i = xs; i < xs + xm; i++) {
        row    = (j - gys) * gxm + (i - gxs);
        x[row] = ((j + 1) * bottom[i - xs + 1] / my + (my - j + 1) * top[i - xs + 1] / (my + 2) + (i + 1) * left[j - ys + 1] / mx + (mx - i + 1) * right[j - ys + 1] / (mx + 2)) / 2.0;
      }
    }

    /* Restore vectors */
    PetscCall(VecRestoreArray(localX, &x));

    PetscCall(VecRestoreArray(user->Left, &left));
    PetscCall(VecRestoreArray(user->Top, &top));
    PetscCall(VecRestoreArray(user->Bottom, &bottom));
    PetscCall(VecRestoreArray(user->Right, &right));

    /* Scatter values into global vector */
    PetscCall(DMLocalToGlobalBegin(user->dm, localX, INSERT_VALUES, X));
    PetscCall(DMLocalToGlobalEnd(user->dm, localX, INSERT_VALUES, X));
  }
  return 0;
}

/* For testing matrix free submatrices */
PetscErrorCode MatrixFreeHessian(Tao tao, Vec x, Mat H, Mat Hpre, void *ptr)
{
  AppCtx *user = (AppCtx *)ptr;
  PetscFunctionBegin;
  PetscCall(FormHessian(tao, x, user->H, user->H, ptr));
  PetscFunctionReturn(0);
}
PetscErrorCode MyMatMult(Mat H_shell, Vec X, Vec Y)
{
  void   *ptr;
  AppCtx *user;
  PetscFunctionBegin;
  PetscCall(MatShellGetContext(H_shell, &ptr));
  user = (AppCtx *)ptr;
  PetscCall(MatMult(user->H, X, Y));
  PetscFunctionReturn(0);
}

/*TEST

   build:
      requires: !complex

   test:
      args: -tao_smonitor -mx 8 -my 6 -bmx 3 -bmy 3 -bheight 0.2 -tao_type tron -tao_gatol 1.e-5
      requires: !single

   test:
      suffix: 2
      nsize: 2
      args: -tao_smonitor -mx 8 -my 8 -bmx 2 -bmy 5 -bheight 0.3 -tao_type blmvm -tao_gatol 1.e-4
      requires: !single

   test:
      suffix: 3
      nsize: 3
      args: -tao_smonitor -mx 8 -my 12 -bmx 4 -bmy 10 -bheight 0.1 -tao_type tron -tao_gatol 1.e-5
      requires: !single

   test:
      suffix: 4
      nsize: 3
      args: -tao_smonitor -mx 8 -my 12 -bmx 4 -bmy 10 -bheight 0.1 -tao_subset_type mask -tao_type tron -tao_gatol 1.e-5
      requires: !single

   test:
      suffix: 5
      nsize: 3
      args: -tao_smonitor -mx 8 -my 12 -bmx 4 -bmy 10 -bheight 0.1 -tao_subset_type matrixfree -matrixfree -pc_type none -tao_type tron -tao_gatol 1.e-5
      requires: !single

   test:
      suffix: 6
      nsize: 3
      args: -tao_smonitor -mx 8 -my 12 -bmx 4 -bmy 10 -bheight 0.1 -tao_subset_type matrixfree -matrixfree -tao_type blmvm -tao_gatol 1.e-4
      requires: !single

   test:
      suffix: 7
      nsize: 3
      args: -tao_smonitor -mx 8 -my 12 -bmx 4 -bmy 10 -bheight 0.1 -tao_subset_type matrixfree -pc_type none -tao_type gpcg -tao_gatol 1.e-5
      requires: !single

   test:
      suffix: 8
      args: -tao_smonitor -mx 8 -my 6 -bmx 3 -bmy 3 -bheight 0.2 -tao_type bncg -tao_bncg_type gd -tao_gatol 1e-4
      requires: !single

   test:
      suffix: 9
      args: -tao_smonitor -mx 8 -my 6 -bmx 3 -bmy 3 -bheight 0.2 -tao_type bncg -tao_gatol 1e-4
      requires: !single

   test:
      suffix: 10
      args: -tao_smonitor -mx 8 -my 6 -bmx 3 -bmy 3 -bheight 0.2 -tao_type bnls -tao_gatol 1e-5
      requires: !single

   test:
      suffix: 11
      args: -tao_smonitor -mx 8 -my 6 -bmx 3 -bmy 3 -bheight 0.2 -tao_type bntr -tao_gatol 1e-5
      requires: !single

   test:
      suffix: 12
      args: -tao_smonitor -mx 8 -my 6 -bmx 3 -bmy 3 -bheight 0.2 -tao_type bntl -tao_gatol 1e-5
      requires: !single

   test:
      suffix: 13
      args: -tao_smonitor -mx 8 -my 6 -bmx 3 -bmy 3 -bheight 0.2 -tao_type bnls -tao_gatol 1e-5 -tao_bnk_max_cg_its 3
      requires: !single

   test:
      suffix: 14
      args: -tao_smonitor -mx 8 -my 6 -bmx 3 -bmy 3 -bheight 0.2 -tao_type bntr -tao_gatol 1e-5 -tao_bnk_max_cg_its 3
      requires: !single

   test:
      suffix: 15
      args: -tao_smonitor -mx 8 -my 6 -bmx 3 -bmy 3 -bheight 0.2 -tao_type bntl -tao_gatol 1e-5 -tao_bnk_max_cg_its 3
      requires: !single

   test:
     suffix: 16
     args: -tao_smonitor -mx 8 -my 8 -bmx 2 -bmy 5 -bheight 0.3 -tao_gatol 1e-4 -tao_type bqnls
     requires: !single

   test:
     suffix: 17
     args: -tao_smonitor -mx 8 -my 8 -bmx 2 -bmy 5 -bheight 0.3 -tao_gatol 1e-4 -tao_type bqnkls -tao_bqnk_mat_type lmvmbfgs
     requires: !single

   test:
     suffix: 18
     args: -tao_smonitor -mx 8 -my 6 -bmx 3 -bmy 3 -bheight 0.2 -tao_type bnls -tao_gatol 1e-5 -tao_mf_hessian
     requires: !single

   test:
     suffix: 19
     args: -tao_smonitor -mx 8 -my 6 -bmx 3 -bmy 3 -bheight 0.2 -tao_type bntr -tao_gatol 1e-5 -tao_mf_hessian
     requires: !single

   test:
     suffix: 20
     args: -tao_smonitor -mx 8 -my 6 -bmx 3 -bmy 3 -bheight 0.2 -tao_type bntl -tao_gatol 1e-5 -tao_mf_hessian

TEST*/
