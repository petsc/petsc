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

  PetscCall(VecGetArrayRead(X, &x));
  PetscCall(VecGetArray(DXDT, &dxdt));

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
  PetscCall(VecRestoreArray(DXDT, &dxdt));
  PetscCall(VecRestoreArrayRead(X, &x));
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

  PetscFunctionBegin;
  nb_cells = user->nb_cells;
  alpha    = user->alpha;
  beta     = user->beta;
  mu_a     = user->mu_a;
  D_a      = user->D_a;
  mu_h     = user->mu_h;
  D_h      = user->D_h;

  PetscCall(VecGetArrayRead(X, &x));
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
    PetscCall(MatSetValues(B, 1, &rowa, idx, ca, va, INSERT_VALUES));
    PetscCall(MatSetValues(B, 1, &rowh, idx, ch, vh, INSERT_VALUES));
  }
  PetscCall(VecRestoreArrayRead(X, &x));
  PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  if (J != B) {
    PetscCall(MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DomainErrorFunction(TS ts, PetscReal t, Vec Y, PetscBool *accept)
{
  AppCtx            *user;
  PetscReal         dt;
  const PetscScalar *x;
  PetscInt          nb_cells, i;

  PetscFunctionBegin;
  PetscCall(TSGetApplicationContext(ts, &user));
  nb_cells = user->nb_cells;
  PetscCall(VecGetArrayRead(Y, &x));
  for (i = 0 ; i < 2*nb_cells ; ++i) {
    if (PetscRealPart(x[i]) < 0) {
      PetscCall(TSGetTimeStep(ts, &dt));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, " ** Domain Error at time %g\n", (double)t));
      *accept = PETSC_FALSE;
      break;
    }
  }
  PetscCall(VecRestoreArrayRead(Y, &x));
  PetscFunctionReturn(0);
}

PetscErrorCode FormInitialState(Vec X, AppCtx* user)
{
  PetscRandom    R;

  PetscFunctionBegin;
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &R));
  PetscCall(PetscRandomSetFromOptions(R));
  PetscCall(PetscRandomSetInterval(R, 0., 10.));

  /*
   * Initialize the state vector
   */
  PetscCall(VecSetRandom(X, R));
  PetscCall(PetscRandomDestroy(&R));
  PetscFunctionReturn(0);
}

PetscErrorCode PrintSolution(Vec X, AppCtx *user)
{
  const PetscScalar *x;
  PetscInt          i;
  PetscInt          nb_cells = user->nb_cells;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(X, &x));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Activator,Inhibitor\n"));
  for (i = 0 ; i < nb_cells ; i++) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%5.6e,%5.6e\n", (double)x[2*i], (double)x[2*i+1]));
  }
  PetscCall(VecRestoreArrayRead(X, &x));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  TS             ts;       /* time-stepping context */
  Vec            x;       /* State vector */
  Mat            J; /* Jacobian matrix */
  AppCtx         user; /* user-defined context */
  PetscReal      ftime;
  PetscInt       its;
  PetscMPIInt    size;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1,PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This is a uniprocessor example only");

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

  PetscOptionsBegin(PETSC_COMM_WORLD, "", "Problem settings", "PROBLEM");
  PetscCall(PetscOptionsInt("-nb_cells", "Number of cells", "ex42.c",user.nb_cells, &user.nb_cells,NULL));
  PetscCall(PetscOptionsReal("-alpha", "Autocatalysis factor", "ex42.c",user.alpha, &user.alpha,NULL));
  PetscCall(PetscOptionsReal("-beta", "Inhibition factor", "ex42.c",user.beta, &user.beta,NULL));
  PetscCall(PetscOptionsReal("-rho_a", "Default production of the activator", "ex42.c",user.rho_a, &user.rho_a,NULL));
  PetscCall(PetscOptionsReal("-mu_a", "Degradation rate of the activator", "ex42.c",user.mu_a, &user.mu_a,NULL));
  PetscCall(PetscOptionsReal("-D_a", "Diffusion rate of the activator", "ex42.c",user.D_a, &user.D_a,NULL));
  PetscCall(PetscOptionsReal("-rho_h", "Default production of the inhibitor", "ex42.c",user.rho_h, &user.rho_h,NULL));
  PetscCall(PetscOptionsReal("-mu_h", "Degradation rate of the inhibitor", "ex42.c",user.mu_h, &user.mu_h,NULL));
  PetscCall(PetscOptionsReal("-D_h", "Diffusion rate of the inhibitor", "ex42.c",user.D_h, &user.D_h,NULL));
  PetscOptionsEnd();

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "nb_cells: %" PetscInt_FMT "\n", user.nb_cells));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "alpha: %5.5g\n", (double)user.alpha));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "beta:  %5.5g\n", (double)user.beta));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "rho_a: %5.5g\n", (double)user.rho_a));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "mu_a:  %5.5g\n", (double)user.mu_a));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "D_a:   %5.5g\n", (double)user.D_a));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "rho_h: %5.5g\n", (double)user.rho_h));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "mu_h:  %5.5g\n", (double)user.mu_h));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "D_h:   %5.5g\n", (double)user.D_h));

  /*
   * Create vector to hold the solution
   */
  PetscCall(VecCreateSeq(PETSC_COMM_WORLD, 2*user.nb_cells, &x));

  /*
   * Create time-stepper context
   */
  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetProblemType(ts, TS_NONLINEAR));

  /*
   * Tell the time-stepper context where to compute the solution
   */
  PetscCall(TSSetSolution(ts, x));

  /*
   * Allocate the jacobian matrix
   */
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_WORLD, 2*user.nb_cells, 2*user.nb_cells, 4, 0, &J));

  /*
   * Provide the call-back for the non-linear function we are evaluating.
   */
  PetscCall(TSSetRHSFunction(ts, NULL, RHSFunction, &user));

  /*
   * Set the Jacobian matrix and the function user to compute Jacobians
   */
  PetscCall(TSSetRHSJacobian(ts, J, J, RHSJacobian, &user));

  /*
   * Set the function checking the domain
   */
  PetscCall(TSSetFunctionDomainError(ts, &DomainErrorFunction));

  /*
   * Initialize the problem with random values
   */
  PetscCall(FormInitialState(x, &user));

  /*
   * Read the solver type from options
   */
  PetscCall(TSSetType(ts, TSPSEUDO));

  /*
   * Set a large number of timesteps and final duration time to insure
   * convergenge to steady state
   */
  PetscCall(TSSetMaxSteps(ts, 2147483647));
  PetscCall(TSSetMaxTime(ts, 1.e12));
  PetscCall(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER));

  /*
   * Set a larger number of potential errors
   */
  PetscCall(TSSetMaxStepRejections(ts, 50));

  /*
   * Also start with a very small dt
   */
  PetscCall(TSSetTimeStep(ts, 0.05));

  /*
   * Set a larger time step increment
   */
  PetscCall(TSPseudoSetTimeStepIncrement(ts, 1.5));

  /*
   * Let the user personalise TS
   */
  PetscCall(TSSetFromOptions(ts));

  /*
   * Set the context for the time stepper
   */
  PetscCall(TSSetApplicationContext(ts, &user));

  /*
   * Setup the time stepper, ready for evaluation
   */
  PetscCall(TSSetUp(ts));

  /*
   * Perform the solve.
   */
  PetscCall(TSSolve(ts, x));
  PetscCall(TSGetSolveTime(ts, &ftime));
  PetscCall(TSGetStepNumber(ts,&its));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Number of time steps = %" PetscInt_FMT ", final time: %4.2e\nResult:\n\n", its, (double)ftime));
  PetscCall(PrintSolution(x, &user));

  /*
   * Free the data structures
   */
  PetscCall(VecDestroy(&x));
  PetscCall(MatDestroy(&J));
  PetscCall(TSDestroy(&ts));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
    build:
      requires: !single !complex

    test:
      args: -ts_max_steps 8
      output_file: output/ex42.out

TEST*/
