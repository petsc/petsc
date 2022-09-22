
static char help[] = "Transistor amplifier (semi-explicit).\n";

/*F
  [I 0] [y'] = f(t,y,z)
  [0 0] [z'] = g(t,y,z)
  Useful options: -ts_monitor_lg_solution -ts_monitor_lg_timestep -lg_indicate_data_points 0
F*/

/*
   Include "petscts.h" so that we can use TS solvers.  Note that this
   file automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
     petscksp.h   - linear solvers
*/
#include <petscts.h>

FILE *gfilepointer_data, *gfilepointer_info;

/* Defines the source  */
PetscErrorCode Ue(PetscScalar t, PetscScalar *U)
{
  PetscFunctionBeginUser;
  U = 0.4 * sin(200 * pi * t);
  PetscFunctionReturn(0);
}
* /

  /*
     Defines the DAE passed to the time solver
*/
  static PetscErrorCode IFunctionSemiExplicit(TS ts, PetscReal t, Vec Y, Vec Ydot, Vec F, void *ctx)
{
  const PetscScalar *y, *ydot;
  PetscScalar       *f;

  PetscFunctionBeginUser;
  /*  The next three lines allow us to access the entries of the vectors directly */
  PetscCall(VecGetArrayRead(Y, &y));
  PetscCall(VecGetArrayRead(Ydot, &ydot));
  PetscCall(VecGetArray(F, &f));

  f[0] = -400 * PetscSinReal(200 * PETSC_PI * t) + 1000 * y[3] + ydot[0];
  f[1] = 0.5 - 1 / (2. * PetscExpReal((500 * (y[0] + y[1] - y[3])) / 13.)) + (500 * y[1]) / 9. + ydot[1];
  f[2] = -222.5522222222222 + 33 / (100. * PetscExpReal((500 * (y[0] + y[1] - y[3])) / 13.)) + (1000 * y[4]) / 27. + ydot[2];
  f[3] = 0.0006666766666666667 - 1 / (1.e8 * PetscExpReal((500 * (y[0] + y[1] - y[3])) / 13.)) + PetscSinReal(200 * PETSC_PI * t) / 2500. + y[0] / 4500. - (11 * y[3]) / 9000.;
  f[4] = 0.0006676566666666666 - 99 / (1.e8 * PetscExpReal((500 * (y[0] + y[1] - y[3])) / 13.)) + y[2] / 9000. - y[4] / 4500.;

  PetscCall(VecRestoreArrayRead(Y, &y));
  PetscCall(VecRestoreArrayRead(Ydot, &ydot));
  PetscCall(VecRestoreArray(F, &f));
  PetscFunctionReturn(0);
}

/*
     Defines the Jacobian of the ODE passed to the ODE solver. See TSSetIJacobian() for the meaning of a and the Jacobian.
*/
static PetscErrorCode IJacobianSemiExplicit(TS ts, PetscReal t, Vec Y, Vec Ydot, PetscReal a, Mat A, Mat B, void *ctx)
{
  PetscInt           rowcol[] = {0, 1, 2, 3, 4};
  const PetscScalar *y, *ydot;
  PetscScalar        J[5][5];

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(Y, &y));
  PetscCall(VecGetArrayRead(Ydot, &ydot));

  PetscCall(PetscMemzero(J, sizeof(J)));

  J[0][0] = a;
  J[0][3] = 1000;
  J[1][0] = 250 / (13. * PetscExpReal((500 * (y[0] + y[1] - y[3])) / 13.));
  J[1][1] = 55.55555555555556 + a + 250 / (13. * PetscExpReal((500 * (y[0] + y[1] - y[3])) / 13.));
  J[1][3] = -250 / (13. * PetscExpReal((500 * (y[0] + y[1] - y[3])) / 13.));
  J[2][0] = -165 / (13. * PetscExpReal((500 * (y[0] + y[1] - y[3])) / 13.));
  J[2][1] = -165 / (13. * PetscExpReal((500 * (y[0] + y[1] - y[3])) / 13.));
  J[2][2] = a;
  J[2][3] = 165 / (13. * PetscExpReal((500 * (y[0] + y[1] - y[3])) / 13.));
  J[2][4] = 37.03703703703704;
  J[3][0] = 0.00022222222222222223 + 1 / (2.6e6 * PetscExpReal((500 * (y[0] + y[1] - y[3])) / 13.));
  J[3][1] = 1 / (2.6e6 * PetscExpReal((500 * (y[0] + y[1] - y[3])) / 13.));
  J[3][3] = -0.0012222222222222222 - 1 / (2.6e6 * PetscExpReal((500 * (y[0] + y[1] - y[3])) / 13.));
  J[4][0] = 99 / (2.6e6 * PetscExpReal((500 * (y[0] + y[1] - y[3])) / 13.));
  J[4][1] = 99 / (2.6e6 * PetscExpReal((500 * (y[0] + y[1] - y[3])) / 13.));
  J[4][2] = 0.00011111111111111112;
  J[4][3] = -99 / (2.6e6 * PetscExpReal((500 * (y[0] + y[1] - y[3])) / 13.));
  J[4][4] = -0.00022222222222222223;

  PetscCall(MatSetValues(B, 5, rowcol, 5, rowcol, &J[0][0], INSERT_VALUES));

  PetscCall(VecRestoreArrayRead(Y, &y));
  PetscCall(VecRestoreArrayRead(Ydot, &ydot));

  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  if (A != B) {
    PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  TS           ts; /* ODE integrator */
  Vec          Y;  /* solution will be stored here */
  Mat          A;  /* Jacobian matrix */
  PetscMPIInt  size;
  PetscInt     n = 5;
  PetscScalar *y;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Only for sequential runs");

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necessary matrix and vectors
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, n, n, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));

  PetscCall(MatCreateVecs(A, &Y, NULL));

  PetscCall(VecGetArray(Y, &y));
  y[0] = -3.0;
  y[1] = 3.0;
  y[2] = 6.0;
  y[3] = 0.0;
  y[4] = 6.0;
  PetscCall(VecRestoreArray(Y, &y));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetProblemType(ts, TS_NONLINEAR));
  PetscCall(TSSetType(ts, TSARKIMEX));
  PetscCall(TSSetEquationType(ts, TS_EQ_DAE_IMPLICIT_INDEX1));
  PetscCall(TSARKIMEXSetFullyImplicit(ts, PETSC_TRUE));
  /*PetscCall(TSSetType(ts,TSROSW));*/
  PetscCall(TSSetIFunction(ts, NULL, IFunctionSemiExplicit, NULL));
  PetscCall(TSSetIJacobian(ts, A, A, IJacobianSemiExplicit, NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetSolution(ts, Y));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetMaxTime(ts, 0.15));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER));
  PetscCall(TSSetTimeStep(ts, .001));
  PetscCall(TSSetFromOptions(ts));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Do Time stepping
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSolve(ts, Y));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&Y));
  PetscCall(TSDestroy(&ts));
  PetscCall(PetscFinalize());
  return 0;
}
