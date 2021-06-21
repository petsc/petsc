static char help[] = "Meinhard't activator-inhibitor model to test TS domain error feature.\n";

/*
   The activator-inhibitor on a line is described by the PDE:

   da/dt = \alpha a^2 / (1 + \beta h) + \rho_a - \mu_a a + D_a d^2 a/ dx^2
   dh/dt = \alpha a^2 + \rho_h - \mu_h h + D_h d^2 h/ dx^2

   The PDE part will be solve by finite-difference on the line of cells.
 */

#include <petscts.h>

typedef struct {
  PetscInt  nb_cells;
  PetscReal alpha;
  PetscReal beta;
  PetscReal rho_a;
  PetscReal rho_h;
  PetscReal mu_a;
  PetscReal mu_h;
  PetscReal D_a;
  PetscReal D_h;
} AppCtx;

PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec X, Vec DXDT, void* ptr)
{
  AppCtx*           user = (AppCtx*)ptr;
  PetscInt          nb_cells, i;
  PetscReal         alpha, beta;
  PetscReal         rho_a, mu_a, D_a;
  PetscReal         rho_h, mu_h, D_h;
  PetscReal         a, h, da, dh, d2a, d2h;
  PetscErrorCode    ierr;
  PetscScalar       *dxdt;
  const PetscScalar *x;

  PetscFunctionBegin;
  nb_cells = user->nb_cells;
  alpha    = user->alpha;
  beta     = user->beta;
  rho_a    = user->rho_a;
  mu_a     = user->mu_a;
  D_a      = user->D_a;
  rho_h    = user->rho_h;
  mu_h     = user->mu_h;
  D_h      = user->D_h;

  ierr = VecGetArrayRead(X, &x);CHKERRQ(ierr);
  ierr = VecGetArray(DXDT, &dxdt);CHKERRQ(ierr);

  for (i = 0 ; i < nb_cells ; i++) {
    a = x[2*i];
    h = x[2*i+1];
    // Reaction:
    da = alpha * a*a / (1. + beta * h) + rho_a - mu_a * a;
    dh = alpha * a*a + rho_h - mu_h*h;
    // Diffusion:
    d2a = d2h = 0.;
    if (i > 0) {
      d2a += (x[2*(i-1)] - a);
      d2h += (x[2*(i-1)+1] - h);
    }
    if (i < nb_cells-1) {
      d2a += (x[2*(i+1)] - a);
      d2h += (x[2*(i+1)+1] - h);
    }
    dxdt[2*i] = da + D_a*d2a;
    dxdt[2*i+1] = dh + D_h*d2h;
  }
  ierr = VecRestoreArray(DXDT, &dxdt);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X, &x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode RHSJacobian(TS ts, PetscReal t, Vec X, Mat J, Mat B, void *ptr)
{
  AppCtx            *user = (AppCtx*)ptr;
  PetscInt          nb_cells, i, idx;
  PetscReal         alpha, beta;
  PetscReal         mu_a, D_a;
  PetscReal         mu_h, D_h;
  PetscReal         a, h;
  const PetscScalar *x;
  PetscScalar       va[4], vh[4];
  PetscInt          ca[4], ch[4], rowa, rowh;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  nb_cells = user->nb_cells;
  alpha    = user->alpha;
  beta     = user->beta;
  mu_a     = user->mu_a;
  D_a      = user->D_a;
  mu_h     = user->mu_h;
  D_h      = user->D_h;

  ierr = VecGetArrayRead(X, &x);CHKERRQ(ierr);
  for (i = 0; i < nb_cells ; ++i) {
    rowa = 2*i;
    rowh = 2*i+1;
    a = x[2*i];
    h = x[2*i+1];
    ca[0] = ch[1] = 2*i;
    va[0] = 2*alpha*a / (1.+beta*h) - mu_a;
    vh[1] = 2*alpha*a;
    ca[1] = ch[0] = 2*i+1;
    va[1] = -beta*alpha*a*a / ((1.+beta*h)*(1.+beta*h));
    vh[0] = -mu_h;
    idx = 2;
    if (i > 0) {
      ca[idx] = 2*(i-1);
      ch[idx] = 2*(i-1)+1;
      va[idx] = D_a;
      vh[idx] = D_h;
      va[0] -= D_a;
      vh[0] -= D_h;
      idx++;
    }
    if (i < nb_cells-1) {
      ca[idx] = 2*(i+1);
      ch[idx] = 2*(i+1)+1;
      va[idx] = D_a;
      vh[idx] = D_h;
      va[0] -= D_a;
      vh[0] -= D_h;
      idx++;
    }
    ierr = MatSetValues(B, 1, &rowa, idx, ca, va, INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatSetValues(B, 1, &rowh, idx, ch, vh, INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecRestoreArrayRead(X, &x);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (J != B) {
    ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DomainErrorFunction(TS ts, PetscReal t, Vec Y, PetscBool *accept)
{
  AppCtx            *user;
  PetscReal         dt;
  PetscErrorCode    ierr;
  const PetscScalar *x;
  PetscInt          nb_cells, i;

  PetscFunctionBegin;
  ierr = TSGetApplicationContext(ts, &user);CHKERRQ(ierr);
  nb_cells = user->nb_cells;
  ierr = VecGetArrayRead(Y, &x);CHKERRQ(ierr);
  for (i = 0 ; i < 2*nb_cells ; ++i) {
    if (PetscRealPart(x[i]) < 0) {
      ierr = TSGetTimeStep(ts, &dt);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD, " ** Domain Error at time %g\n", (double)t);CHKERRQ(ierr);
      *accept = PETSC_FALSE;
      break;
    }
  }
  ierr = VecRestoreArrayRead(Y, &x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FormInitialState(Vec X, AppCtx* user)
{
  PetscErrorCode ierr;
  PetscRandom    R;

  PetscFunctionBegin;
  ierr = PetscRandomCreate(PETSC_COMM_WORLD, &R);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(R);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(R, 0., 10.);CHKERRQ(ierr);

  /*
   * Initialize the state vector
   */
  ierr = VecSetRandom(X, R);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&R);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PrintSolution(Vec X, AppCtx *user)
{
  PetscErrorCode    ierr;
  const PetscScalar *x;
  PetscInt          i;
  PetscInt          nb_cells = user->nb_cells;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(X, &x);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Activator,Inhibitor\n");CHKERRQ(ierr);
  for (i = 0 ; i < nb_cells ; i++) {
    ierr = PetscPrintf(PETSC_COMM_WORLD, "%5.6e,%5.6e\n", (double)x[2*i], (double)x[2*i+1]);CHKERRQ(ierr);
  }
  ierr = VecRestoreArrayRead(X, &x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  TS             ts;       /* time-stepping context */
  Vec            x;       /* State vector */
  Mat            J; /* Jacobian matrix */
  AppCtx         user; /* user-defined context */
  PetscErrorCode ierr;
  PetscReal      ftime;
  PetscInt       its;
  PetscMPIInt    size;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);CHKERRMPI(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "This is a uniprocessor example only");

  /*
   * Allow user to set the grid dimensions and the equations parameters
   */

  user.nb_cells = 50;
  user.alpha = 10.;
  user.beta = 1.;
  user.rho_a = 1.;
  user.rho_h = 2.;
  user.mu_a = 2.;
  user.mu_h = 3.;
  user.D_a = 0.;
  user.D_h = 30.;

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "Problem settings", "PROBLEM");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-nb_cells", "Number of cells", "ex42.c",user.nb_cells, &user.nb_cells,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-alpha", "Autocatalysis factor", "ex42.c",user.alpha, &user.alpha,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-beta", "Inhibition factor", "ex42.c",user.beta, &user.beta,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-rho_a", "Default production of the activator", "ex42.c",user.rho_a, &user.rho_a,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mu_a", "Degradation rate of the activator", "ex42.c",user.mu_a, &user.mu_a,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-D_a", "Diffusion rate of the activator", "ex42.c",user.D_a, &user.D_a,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-rho_h", "Default production of the inhibitor", "ex42.c",user.rho_h, &user.rho_h,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mu_h", "Degradation rate of the inhibitor", "ex42.c",user.mu_h, &user.mu_h,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-D_h", "Diffusion rate of the inhibitor", "ex42.c",user.D_h, &user.D_h,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD, "nb_cells: %D\n", user.nb_cells);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "alpha: %5.5g\n", (double)user.alpha);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "beta:  %5.5g\n", (double)user.beta);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "rho_a: %5.5g\n", (double)user.rho_a);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "mu_a:  %5.5g\n", (double)user.mu_a);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "D_a:   %5.5g\n", (double)user.D_a);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "rho_h: %5.5g\n", (double)user.rho_h);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "mu_h:  %5.5g\n", (double)user.mu_h);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "D_h:   %5.5g\n", (double)user.D_h);CHKERRQ(ierr);

  /*
   * Create vector to hold the solution
   */
  ierr = VecCreateSeq(PETSC_COMM_WORLD, 2*user.nb_cells, &x);CHKERRQ(ierr);

  /*
   * Create time-stepper context
   */
  ierr = TSCreate(PETSC_COMM_WORLD, &ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts, TS_NONLINEAR);CHKERRQ(ierr);

  /*
   * Tell the time-stepper context where to compute the solution
   */
  ierr = TSSetSolution(ts, x);CHKERRQ(ierr);

  /*
   * Allocate the jacobian matrix
   */
  ierr = MatCreateSeqAIJ(PETSC_COMM_WORLD, 2*user.nb_cells, 2*user.nb_cells, 4, 0, &J);CHKERRQ(ierr);

  /*
   * Provide the call-back for the non-linear function we are evaluating.
   */
  ierr = TSSetRHSFunction(ts, NULL, RHSFunction, &user);CHKERRQ(ierr);

  /*
   * Set the Jacobian matrix and the function user to compute Jacobians
   */
  ierr = TSSetRHSJacobian(ts, J, J, RHSJacobian, &user);CHKERRQ(ierr);

  /*
   * Set the function checking the domain
   */
  ierr = TSSetFunctionDomainError(ts, &DomainErrorFunction);CHKERRQ(ierr);

  /*
   * Initialize the problem with random values
   */
  ierr = FormInitialState(x, &user);CHKERRQ(ierr);

  /*
   * Read the solver type from options
   */
  ierr = TSSetType(ts, TSPSEUDO);CHKERRQ(ierr);

  /*
   * Set a large number of timesteps and final duration time to insure
   * convergenge to steady state
   */
  ierr = TSSetMaxSteps(ts, 2147483647);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts, 1.e12);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);

  /*
   * Set a larger number of potential errors
   */
  ierr = TSSetMaxStepRejections(ts, 50);CHKERRQ(ierr);

  /*
   * Also start with a very small dt
   */
  ierr = TSSetTimeStep(ts, 0.05);CHKERRQ(ierr);

  /*
   * Set a larger time step increment
   */
  ierr = TSPseudoSetTimeStepIncrement(ts, 1.5);CHKERRQ(ierr);

  /*
   * Let the user personalise TS
   */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /*
   * Set the context for the time stepper
   */
  ierr = TSSetApplicationContext(ts, &user);CHKERRQ(ierr);

  /*
   * Setup the time stepper, ready for evaluation
   */
  ierr = TSSetUp(ts);CHKERRQ(ierr);

  /*
   * Perform the solve.
   */
  ierr = TSSolve(ts, x);CHKERRQ(ierr);
  ierr = TSGetSolveTime(ts, &ftime);CHKERRQ(ierr);
  ierr = TSGetStepNumber(ts,&its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Number of time steps = %D, final time: %4.2e\nResult:\n\n", its, (double)ftime);CHKERRQ(ierr);
  ierr = PrintSolution(x, &user);CHKERRQ(ierr);

  /*
   * Free the data structures
   */
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST
    build:
      requires: !single !complex

    test:
      args: -ts_max_steps 8
      output_file: output/ex42.out

TEST*/
