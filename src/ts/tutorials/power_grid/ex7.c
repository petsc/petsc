
static char help[] = "Time-dependent PDE in 2d for calculating joint PDF. \n";
/*
   p_t = -x_t*p_x -y_t*p_y + f(t)*p_yy
   xmin < x < xmax, ymin < y < ymax;

   Boundary conditions Neumman using mirror values

   Note that x_t and y_t in the above are given functions of x and y; they are not derivatives of x and y.
   x_t = (y - ws)  y_t = (ws/2H)*(Pm - Pmax*sin(x))

*/

#include <petscdm.h>
#include <petscdmda.h>
#include <petscts.h>

/*
   User-defined data structures and routines
*/
typedef struct {
  PetscScalar    ws;         /* Synchronous speed */
  PetscScalar    H;          /* Inertia constant */
  PetscScalar    D;          /* Damping constant */
  PetscScalar    Pmax;       /* Maximum power output of generator */
  PetscScalar    PM_min;     /* Mean mechanical power input */
  PetscScalar    lambda;     /* correlation time */
  PetscScalar    q;          /* noise strength */
  PetscScalar    mux;        /* Initial average angle */
  PetscScalar    sigmax;     /* Standard deviation of initial angle */
  PetscScalar    muy;        /* Average speed */
  PetscScalar    sigmay;     /* standard deviation of initial speed */
  PetscScalar    rho;        /* Cross-correlation coefficient */
  PetscScalar    xmin;       /* left boundary of angle */
  PetscScalar    xmax;       /* right boundary of angle */
  PetscScalar    ymin;       /* bottom boundary of speed */
  PetscScalar    ymax;       /* top boundary of speed */
  PetscScalar    dx;         /* x step size */
  PetscScalar    dy;         /* y step size */
  PetscScalar    disper_coe; /* Dispersion coefficient */
  DM             da;
  PetscInt       st_width; /* Stencil width */
  DMBoundaryType bx;       /* x boundary type */
  DMBoundaryType by;       /* y boundary type */
  PetscBool      nonoiseinitial;
} AppCtx;

PetscErrorCode Parameter_settings(AppCtx *);
PetscErrorCode ini_bou(Vec, AppCtx *);
PetscErrorCode IFunction(TS, PetscReal, Vec, Vec, Vec, void *);
PetscErrorCode IJacobian(TS, PetscReal, Vec, Vec, PetscReal, Mat, Mat, void *);
PetscErrorCode PostStep(TS);

int main(int argc, char **argv)
{
  Vec         x;    /* Solution vector */
  TS          ts;   /* Time-stepping context */
  AppCtx      user; /* Application context */
  PetscViewer viewer;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, "petscopt_ex7", help));

  /* Get physics and time parameters */
  PetscCall(Parameter_settings(&user));
  /* Create a 2D DA with dof = 1 */
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, user.bx, user.by, DMDA_STENCIL_STAR, 4, 4, PETSC_DECIDE, PETSC_DECIDE, 1, user.st_width, NULL, NULL, &user.da));
  PetscCall(DMSetFromOptions(user.da));
  PetscCall(DMSetUp(user.da));
  /* Set x and y coordinates */
  PetscCall(DMDASetUniformCoordinates(user.da, user.xmin, user.xmax, user.ymin, user.ymax, 0, 0));
  PetscCall(DMDASetCoordinateName(user.da, 0, "X - the angle"));
  PetscCall(DMDASetCoordinateName(user.da, 1, "Y - the speed"));

  /* Get global vector x from DM  */
  PetscCall(DMCreateGlobalVector(user.da, &x));

  PetscCall(ini_bou(x, &user));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, "ini_x", FILE_MODE_WRITE, &viewer));
  PetscCall(VecView(x, viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetDM(ts, user.da));
  PetscCall(TSSetProblemType(ts, TS_NONLINEAR));
  PetscCall(TSSetType(ts, TSARKIMEX));
  PetscCall(TSSetIFunction(ts, NULL, IFunction, &user));
  /*  PetscCall(TSSetIJacobian(ts,NULL,NULL,IJacobian,&user));  */
  PetscCall(TSSetApplicationContext(ts, &user));
  PetscCall(TSSetTimeStep(ts, .005));
  PetscCall(TSSetFromOptions(ts));
  PetscCall(TSSetPostStep(ts, PostStep));
  PetscCall(TSSolve(ts, x));

  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, "fin_x", FILE_MODE_WRITE, &viewer));
  PetscCall(VecView(x, viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscCall(VecDestroy(&x));
  PetscCall(DMDestroy(&user.da));
  PetscCall(TSDestroy(&ts));
  PetscCall(PetscFinalize());
  return 0;
}

PetscErrorCode PostStep(TS ts)
{
  Vec          X, gc;
  AppCtx      *user;
  PetscScalar  sum = 0, asum;
  PetscReal    t, **p;
  DMDACoor2d **coors;
  DM           cda;
  PetscInt     i, j, xs, ys, xm, ym;

  PetscFunctionBegin;
  PetscCall(TSGetApplicationContext(ts, &user));
  PetscCall(TSGetTime(ts, &t));
  PetscCall(TSGetSolution(ts, &X));

  PetscCall(DMGetCoordinateDM(user->da, &cda));
  PetscCall(DMDAGetCorners(cda, &xs, &ys, 0, &xm, &ym, 0));
  PetscCall(DMGetCoordinates(user->da, &gc));
  PetscCall(DMDAVecGetArrayRead(cda, gc, &coors));
  PetscCall(DMDAVecGetArrayRead(user->da, X, &p));
  for (i = xs; i < xs + xm; i++) {
    for (j = ys; j < ys + ym; j++) {
      if (coors[j][i].y < 5) sum += p[j][i];
    }
  }
  PetscCall(DMDAVecRestoreArrayRead(cda, gc, &coors));
  PetscCall(DMDAVecRestoreArrayRead(user->da, X, &p));
  PetscCall(MPIU_Allreduce(&sum, &asum, 1, MPIU_SCALAR, MPIU_SUM, PetscObjectComm((PetscObject)ts)));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "sum(p)*dw*dtheta at t = %f = %f\n", (double)t, (double)(asum)));
  if (sum < 1.0e-2) {
    PetscCall(TSSetConvergedReason(ts, TS_CONVERGED_USER));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Exiting TS as the integral of PDF is almost zero\n"));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode ini_bou(Vec X, AppCtx *user)
{
  DM            cda;
  DMDACoor2d  **coors;
  PetscScalar **p;
  Vec           gc;
  PetscInt      i, j;
  PetscInt      xs, ys, xm, ym, M, N;
  PetscScalar   xi, yi;
  PetscScalar   sigmax = user->sigmax, sigmay = user->sigmay;
  PetscScalar   rho = user->rho;
  PetscScalar   muy = user->muy, mux;
  PetscMPIInt   rank;
  PetscScalar   sum;

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCall(DMDAGetInfo(user->da, NULL, &M, &N, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
  user->dx = (user->xmax - user->xmin) / (M - 1);
  user->dy = (user->ymax - user->ymin) / (N - 1);
  PetscCall(DMGetCoordinateDM(user->da, &cda));
  PetscCall(DMGetCoordinates(user->da, &gc));
  PetscCall(DMDAVecGetArray(cda, gc, &coors));
  PetscCall(DMDAVecGetArray(user->da, X, &p));
  PetscCall(DMDAGetCorners(cda, &xs, &ys, 0, &xm, &ym, 0));

  /* mux and muy need to be grid points in the x and y-direction otherwise the solution goes unstable
     muy is set by choosing the y domain, no. of grid points along y-direction so that muy is a grid point
     in the y-direction. We only modify mux here
  */
  mux = user->mux = coors[0][M / 2 + 10].x; /* For -pi < x < pi, this should be some angle between 0 and pi/2 */
  if (user->nonoiseinitial) {
    for (i = xs; i < xs + xm; i++) {
      for (j = ys; j < ys + ym; j++) {
        xi = coors[j][i].x;
        yi = coors[j][i].y;
        if ((xi == mux) && (yi == muy)) p[j][i] = 1.0;
      }
    }
  } else {
    /* Change PM_min accordingly */
    user->PM_min = user->Pmax * PetscSinScalar(mux);
    for (i = xs; i < xs + xm; i++) {
      for (j = ys; j < ys + ym; j++) {
        xi = coors[j][i].x;
        yi = coors[j][i].y;
        p[j][i] = (0.5 / (PETSC_PI * sigmax * sigmay * PetscSqrtScalar(1.0 - rho * rho))) * PetscExpScalar(-0.5 / (1 - rho * rho) * (PetscPowScalar((xi - mux) / sigmax, 2) + PetscPowScalar((yi - muy) / sigmay, 2) - 2 * rho * (xi - mux) * (yi - muy) / (sigmax * sigmay)));
      }
    }
  }
  PetscCall(DMDAVecRestoreArray(cda, gc, &coors));
  PetscCall(DMDAVecRestoreArray(user->da, X, &p));
  PetscCall(VecSum(X, &sum));
  PetscCall(VecScale(X, 1.0 / sum));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* First advection term */
PetscErrorCode adv1(PetscScalar **p, PetscScalar y, PetscInt i, PetscInt j, PetscInt M, PetscScalar *p1, AppCtx *user)
{
  PetscScalar f, fpos, fneg;
  PetscFunctionBegin;
  f    = (y - user->ws);
  fpos = PetscMax(f, 0);
  fneg = PetscMin(f, 0);
  if (user->st_width == 1) {
    *p1 = fpos * (p[j][i] - p[j][i - 1]) / user->dx + fneg * (p[j][i + 1] - p[j][i]) / user->dx;
  } else if (user->st_width == 2) {
    *p1 = fpos * (3 * p[j][i] - 4 * p[j][i - 1] + p[j][i - 2]) / (2 * user->dx) + fneg * (-p[j][i + 2] + 4 * p[j][i + 1] - 3 * p[j][i]) / (2 * user->dx);
  } else if (user->st_width == 3) {
    *p1 = fpos * (2 * p[j][i + 1] + 3 * p[j][i] - 6 * p[j][i - 1] + p[j][i - 2]) / (6 * user->dx) + fneg * (-p[j][i + 2] + 6 * p[j][i + 1] - 3 * p[j][i] - 2 * p[j][i - 1]) / (6 * user->dx);
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No support for wider stencils");
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Second advection term */
PetscErrorCode adv2(PetscScalar **p, PetscScalar x, PetscInt i, PetscInt j, PetscInt N, PetscScalar *p2, AppCtx *user)
{
  PetscScalar f, fpos, fneg;
  PetscFunctionBegin;
  f    = (user->ws / (2 * user->H)) * (user->PM_min - user->Pmax * PetscSinScalar(x));
  fpos = PetscMax(f, 0);
  fneg = PetscMin(f, 0);
  if (user->st_width == 1) {
    *p2 = fpos * (p[j][i] - p[j - 1][i]) / user->dy + fneg * (p[j + 1][i] - p[j][i]) / user->dy;
  } else if (user->st_width == 2) {
    *p2 = fpos * (3 * p[j][i] - 4 * p[j - 1][i] + p[j - 2][i]) / (2 * user->dy) + fneg * (-p[j + 2][i] + 4 * p[j + 1][i] - 3 * p[j][i]) / (2 * user->dy);
  } else if (user->st_width == 3) {
    *p2 = fpos * (2 * p[j + 1][i] + 3 * p[j][i] - 6 * p[j - 1][i] + p[j - 2][i]) / (6 * user->dy) + fneg * (-p[j + 2][i] + 6 * p[j + 1][i] - 3 * p[j][i] - 2 * p[j - 1][i]) / (6 * user->dy);
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No support for wider stencils");
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Diffusion term */
PetscErrorCode diffuse(PetscScalar **p, PetscInt i, PetscInt j, PetscReal t, PetscScalar *p_diff, AppCtx *user)
{
  PetscFunctionBeginUser;
  if (user->st_width == 1) {
    *p_diff = user->disper_coe * ((p[j - 1][i] - 2 * p[j][i] + p[j + 1][i]) / (user->dy * user->dy));
  } else if (user->st_width == 2) {
    *p_diff = user->disper_coe * ((-p[j - 2][i] + 16 * p[j - 1][i] - 30 * p[j][i] + 16 * p[j + 1][i] - p[j + 2][i]) / (12.0 * user->dy * user->dy));
  } else if (user->st_width == 3) {
    *p_diff = user->disper_coe * ((2 * p[j - 3][i] - 27 * p[j - 2][i] + 270 * p[j - 1][i] - 490 * p[j][i] + 270 * p[j + 1][i] - 27 * p[j + 2][i] + 2 * p[j + 3][i]) / (180.0 * user->dy * user->dy));
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No support for wider stencils");
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode IFunction(TS ts, PetscReal t, Vec X, Vec Xdot, Vec F, void *ctx)
{
  AppCtx       *user = (AppCtx *)ctx;
  DM            cda;
  DMDACoor2d  **coors;
  PetscScalar **p, **f, **pdot;
  PetscInt      i, j;
  PetscInt      xs, ys, xm, ym, M, N;
  Vec           localX, gc, localXdot;
  PetscScalar   p_adv1 = 0.0, p_adv2 = 0.0, p_diff; /* initialize to prevent incorrect compiler warnings */

  PetscFunctionBeginUser;
  PetscCall(DMDAGetInfo(user->da, NULL, &M, &N, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
  PetscCall(DMGetCoordinateDM(user->da, &cda));
  PetscCall(DMDAGetCorners(cda, &xs, &ys, 0, &xm, &ym, 0));

  PetscCall(DMGetLocalVector(user->da, &localX));
  PetscCall(DMGetLocalVector(user->da, &localXdot));

  PetscCall(DMGlobalToLocalBegin(user->da, X, INSERT_VALUES, localX));
  PetscCall(DMGlobalToLocalEnd(user->da, X, INSERT_VALUES, localX));
  PetscCall(DMGlobalToLocalBegin(user->da, Xdot, INSERT_VALUES, localXdot));
  PetscCall(DMGlobalToLocalEnd(user->da, Xdot, INSERT_VALUES, localXdot));

  PetscCall(DMGetCoordinatesLocal(user->da, &gc));

  PetscCall(DMDAVecGetArrayRead(cda, gc, &coors));
  PetscCall(DMDAVecGetArrayRead(user->da, localX, &p));
  PetscCall(DMDAVecGetArrayRead(user->da, localXdot, &pdot));
  PetscCall(DMDAVecGetArray(user->da, F, &f));

  user->disper_coe = PetscPowScalar((user->lambda * user->ws) / (2 * user->H), 2) * user->q * (1.0 - PetscExpScalar(-t / user->lambda));
  for (i = xs; i < xs + xm; i++) {
    for (j = ys; j < ys + ym; j++) {
      PetscCall(adv1(p, coors[j][i].y, i, j, M, &p_adv1, user));
      PetscCall(adv2(p, coors[j][i].x, i, j, N, &p_adv2, user));
      PetscCall(diffuse(p, i, j, t, &p_diff, user));
      f[j][i] = -p_adv1 - p_adv2 + p_diff - pdot[j][i];
    }
  }
  PetscCall(DMDAVecRestoreArrayRead(user->da, localX, &p));
  PetscCall(DMDAVecRestoreArrayRead(user->da, localX, &pdot));
  PetscCall(DMRestoreLocalVector(user->da, &localX));
  PetscCall(DMRestoreLocalVector(user->da, &localXdot));
  PetscCall(DMDAVecRestoreArray(user->da, F, &f));
  PetscCall(DMDAVecRestoreArrayRead(cda, gc, &coors));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode IJacobian(TS ts, PetscReal t, Vec X, Vec Xdot, PetscReal a, Mat J, Mat Jpre, void *ctx)
{
  AppCtx      *user = (AppCtx *)ctx;
  DM           cda;
  DMDACoor2d **coors;
  PetscInt     i, j;
  PetscInt     xs, ys, xm, ym, M, N;
  Vec          gc;
  PetscScalar  val[5], xi, yi;
  MatStencil   row, col[5];
  PetscScalar  c1, c3, c5, c1pos, c1neg, c3pos, c3neg;

  PetscFunctionBeginUser;
  PetscCall(DMDAGetInfo(user->da, NULL, &M, &N, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
  PetscCall(DMGetCoordinateDM(user->da, &cda));
  PetscCall(DMDAGetCorners(cda, &xs, &ys, 0, &xm, &ym, 0));

  PetscCall(DMGetCoordinatesLocal(user->da, &gc));
  PetscCall(DMDAVecGetArrayRead(cda, gc, &coors));
  for (i = xs; i < xs + xm; i++) {
    for (j = ys; j < ys + ym; j++) {
      PetscInt nc = 0;
      xi          = coors[j][i].x;
      yi          = coors[j][i].y;
      row.i       = i;
      row.j       = j;
      c1          = (yi - user->ws) / user->dx;
      c1pos       = PetscMax(c1, 0);
      c1neg       = PetscMin(c1, 0);
      c3          = (user->ws / (2.0 * user->H)) * (user->PM_min - user->Pmax * PetscSinScalar(xi)) / user->dy;
      c3pos       = PetscMax(c3, 0);
      c3neg       = PetscMin(c3, 0);
      c5          = (PetscPowScalar((user->lambda * user->ws) / (2 * user->H), 2) * user->q * (1.0 - PetscExpScalar(-t / user->lambda))) / (user->dy * user->dy);
      col[nc].i   = i - 1;
      col[nc].j   = j;
      val[nc++]   = c1pos;
      col[nc].i   = i + 1;
      col[nc].j   = j;
      val[nc++]   = -c1neg;
      col[nc].i   = i;
      col[nc].j   = j - 1;
      val[nc++]   = c3pos + c5;
      col[nc].i   = i;
      col[nc].j   = j + 1;
      val[nc++]   = -c3neg + c5;
      col[nc].i   = i;
      col[nc].j   = j;
      val[nc++]   = -c1pos + c1neg - c3pos + c3neg - 2 * c5 - a;
      PetscCall(MatSetValuesStencil(Jpre, 1, &row, nc, col, val, INSERT_VALUES));
    }
  }
  PetscCall(DMDAVecRestoreArrayRead(cda, gc, &coors));

  PetscCall(MatAssemblyBegin(Jpre, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Jpre, MAT_FINAL_ASSEMBLY));
  if (J != Jpre) {
    PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Parameter_settings(AppCtx *user)
{
  PetscBool flg;

  PetscFunctionBeginUser;

  /* Set default parameters */
  user->ws             = 1.0;
  user->H              = 5.0;
  user->Pmax           = 2.1;
  user->PM_min         = 1.0;
  user->lambda         = 0.1;
  user->q              = 1.0;
  user->mux            = PetscAsinScalar(user->PM_min / user->Pmax);
  user->sigmax         = 0.1;
  user->sigmay         = 0.1;
  user->rho            = 0.0;
  user->xmin           = -PETSC_PI;
  user->xmax           = PETSC_PI;
  user->bx             = DM_BOUNDARY_PERIODIC;
  user->by             = DM_BOUNDARY_MIRROR;
  user->nonoiseinitial = PETSC_FALSE;

  /*
     ymin of -3 seems to let the unstable solution move up and leave a zero in its wake
     with an ymin of -1 the wake is never exactly zero
  */
  user->ymin     = -3.0;
  user->ymax     = 10.0;
  user->st_width = 1;

  PetscCall(PetscOptionsGetScalar(NULL, NULL, "-ws", &user->ws, &flg));
  PetscCall(PetscOptionsGetScalar(NULL, NULL, "-Inertia", &user->H, &flg));
  PetscCall(PetscOptionsGetScalar(NULL, NULL, "-Pmax", &user->Pmax, &flg));
  PetscCall(PetscOptionsGetScalar(NULL, NULL, "-PM_min", &user->PM_min, &flg));
  PetscCall(PetscOptionsGetScalar(NULL, NULL, "-lambda", &user->lambda, &flg));
  PetscCall(PetscOptionsGetScalar(NULL, NULL, "-q", &user->q, &flg));
  PetscCall(PetscOptionsGetScalar(NULL, NULL, "-mux", &user->mux, &flg));
  PetscCall(PetscOptionsGetScalar(NULL, NULL, "-sigmax", &user->sigmax, &flg));
  PetscCall(PetscOptionsGetScalar(NULL, NULL, "-muy", &user->muy, &flg));
  if (flg == 0) user->muy = user->ws;
  PetscCall(PetscOptionsGetScalar(NULL, NULL, "-sigmay", &user->sigmay, &flg));
  PetscCall(PetscOptionsGetScalar(NULL, NULL, "-rho", &user->rho, &flg));
  PetscCall(PetscOptionsGetScalar(NULL, NULL, "-xmin", &user->xmin, &flg));
  PetscCall(PetscOptionsGetScalar(NULL, NULL, "-xmax", &user->xmax, &flg));
  PetscCall(PetscOptionsGetScalar(NULL, NULL, "-ymin", &user->ymin, &flg));
  PetscCall(PetscOptionsGetScalar(NULL, NULL, "-ymax", &user->ymax, &flg));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-stencil_width", &user->st_width, &flg));
  PetscCall(PetscOptionsGetEnum(NULL, NULL, "-bx", DMBoundaryTypes, (PetscEnum *)&user->bx, &flg));
  PetscCall(PetscOptionsGetEnum(NULL, NULL, "-by", DMBoundaryTypes, (PetscEnum *)&user->by, &flg));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-nonoiseinitial", &user->nonoiseinitial, &flg));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/*TEST

   build:
      requires: !complex !single

   test:
      args: -ts_max_steps 2
      localrunfiles: petscopt_ex7

   test:
      suffix: 2
      args: -ts_max_steps 2 -snes_mf_operator
      output_file: output/ex7_1.out
      localrunfiles: petscopt_ex7
      timeoutfactor: 2

   test:
      suffix: 3
      args: -ts_max_steps 2 -snes_mf -pc_type none
      output_file: output/ex7_1.out
      localrunfiles: petscopt_ex7
      timeoutfactor: 2

TEST*/
