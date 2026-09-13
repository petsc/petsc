static char help[] = "Tests TSARKIMEX adjoint parameter sensitivities when the parameter appears in the IFunction, the RHSFunction, or both.\n\n";

/*
  The scalar problem is

    u_t + alpha u = -beta u,   u(0) = 1

  split so that alpha u is the implicit part F(t,u,u_t) = u_t + alpha u and -beta u is the
  explicit part G(t,u). The single parameter p enters alpha, beta, or both depending on
  -param_dependence, so that exactly one of TSSetIJacobianP() and TSSetRHSJacobianP() is
  registered in the one-sided cases. With r the resulting decay rate, u(T) = exp(-r T) and
  the cost J = u(T) has the analytic gradient dJ/dp = -T (dr/dp) exp(-r T). The computed gradient
  is the gradient of the discrete cost, so it matches the analytic one only up to the time
  discretization error.
*/

#include <petscts.h>

typedef enum {
  PARAM_IMPLICIT,
  PARAM_EXPLICIT,
  PARAM_BOTH
} ParamDependence;
static const char *const ParamDependences[] = {"implicit", "explicit", "both", "ParamDependence", "PARAM_", NULL};

typedef struct {
  PetscReal       p;
  ParamDependence dep;
} AppCtx;

static PetscReal Alpha(AppCtx *user)
{
  return user->dep == PARAM_EXPLICIT ? 1.0 : user->p;
}

static PetscReal Beta(AppCtx *user)
{
  return user->dep == PARAM_IMPLICIT ? 1.0 : user->p;
}

/* G(t,u) = -beta u */
static PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec U, Vec G, void *ctx)
{
  AppCtx *user = (AppCtx *)ctx;

  PetscFunctionBeginUser;
  PetscCall(VecCopy(U, G));
  PetscCall(VecScale(G, -Beta(user)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RHSJacobian(TS ts, PetscReal t, Vec U, Mat J, Mat P, void *ctx)
{
  AppCtx *user = (AppCtx *)ctx;

  PetscFunctionBeginUser;
  PetscCall(MatSetValue(P, 0, 0, -Beta(user), INSERT_VALUES));
  PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* F(t,u,u_t) = u_t + alpha u */
static PetscErrorCode IFunction(TS ts, PetscReal t, Vec U, Vec Udot, Vec F, void *ctx)
{
  AppCtx *user = (AppCtx *)ctx;

  PetscFunctionBeginUser;
  PetscCall(VecCopy(U, F));
  PetscCall(VecScale(F, Alpha(user)));
  PetscCall(VecAXPY(F, 1.0, Udot));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode IJacobian(TS ts, PetscReal t, Vec U, Vec Udot, PetscReal shift, Mat J, Mat P, void *ctx)
{
  AppCtx *user = (AppCtx *)ctx;

  PetscFunctionBeginUser;
  PetscCall(MatSetValue(P, 0, 0, shift + Alpha(user), INSERT_VALUES));
  PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* dF/dp = u */
static PetscErrorCode IJacobianP(TS ts, PetscReal t, Vec U, Vec Udot, PetscReal shift, Mat Jacp, void *ctx)
{
  const PetscScalar *u;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(MatSetValue(Jacp, 0, 0, u[0], INSERT_VALUES));
  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(MatAssemblyBegin(Jacp, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Jacp, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* dG/dp = -u */
static PetscErrorCode RHSJacobianP(TS ts, PetscReal t, Vec U, Mat Jacp, void *ctx)
{
  const PetscScalar *u;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(MatSetValue(Jacp, 0, 0, -u[0], INSERT_VALUES));
  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(MatAssemblyBegin(Jacp, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Jacp, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  TS                 ts;
  Vec                U, lambda, mu;
  Mat                Jrhs, Ji, Jacp = NULL, Jacprhs = NULL;
  AppCtx             user;
  PetscReal          ftime = 0.1, rtol = 1e-2, rate, drate, analytic, gradient;
  PetscBool          nullijacobianp = PETSC_FALSE;
  const PetscScalar *m;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  user.p   = 2.0;
  user.dep = PARAM_IMPLICIT;
  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Adjoint parameter Jacobian options", "TS");
  PetscCall(PetscOptionsEnum("-param_dependence", "Which part of the IMEX splitting the parameter appears in", NULL, ParamDependences, (PetscEnum)user.dep, (PetscEnum *)&user.dep, NULL));
  PetscCall(PetscOptionsReal("-p", "Value of the parameter", NULL, user.p, &user.p, NULL));
  PetscCall(PetscOptionsBool("-null_ijacobianp", "Register the IJacobianP matrix without a callback, as PETSC_NULL_FUNCTION does from Fortran", NULL, nullijacobianp, &nullijacobianp, NULL));
  PetscOptionsEnd();

  PetscCall(VecCreateSeq(PETSC_COMM_SELF, 1, &U));
  PetscCall(VecSet(U, 1.0));
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, 1, 1, NULL, &Jrhs));
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, 1, 1, NULL, &Ji));

  PetscCall(TSCreate(PETSC_COMM_SELF, &ts));
  PetscCall(TSSetType(ts, TSARKIMEX));
  PetscCall(TSSetRHSFunction(ts, NULL, RHSFunction, &user));
  PetscCall(TSSetRHSJacobian(ts, Jrhs, Jrhs, RHSJacobian, &user));
  PetscCall(TSSetIFunction(ts, NULL, IFunction, &user));
  PetscCall(TSSetIJacobian(ts, Ji, Ji, IJacobian, &user));
  if (user.dep != PARAM_EXPLICIT) {
    PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, 1, 1, NULL, &Jacp));
    if (nullijacobianp) {
      /* seed the matrix so that a contribution taken from it, rather than from the absent callback, is unmistakable */
      PetscCall(MatSetValue(Jacp, 0, 0, 1e3, INSERT_VALUES));
      PetscCall(MatAssemblyBegin(Jacp, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(Jacp, MAT_FINAL_ASSEMBLY));
      PetscCall(TSSetIJacobianP(ts, Jacp, NULL, NULL));
    } else PetscCall(TSSetIJacobianP(ts, Jacp, IJacobianP, &user));
  }
  if (user.dep != PARAM_IMPLICIT) {
    PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, 1, 1, NULL, &Jacprhs));
    PetscCall(TSSetRHSJacobianP(ts, Jacprhs, RHSJacobianP, &user));
  }

  PetscCall(TSSetSaveTrajectory(ts));
  PetscCall(TSSetTime(ts, 0.0));
  PetscCall(TSSetTimeStep(ts, 0.01));
  PetscCall(TSSetMaxTime(ts, ftime));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetFromOptions(ts));
  PetscCall(TSSolve(ts, U));

  /* J = u(T), so lambda(T) = dJ/du(T) = 1 and mu(T) = 0 */
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, 1, &lambda));
  PetscCall(VecSet(lambda, 1.0));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, 1, &mu));
  PetscCall(VecSet(mu, 0.0));
  PetscCall(TSSetCostGradients(ts, 1, &lambda, &mu));
  PetscCall(TSAdjointSolve(ts));

  rate  = Alpha(&user) + Beta(&user);
  drate = user.dep == PARAM_BOTH ? 2.0 : 1.0;
  /* an unregistered callback contributes nothing, so its term drops out of the expected gradient as well */
  if (nullijacobianp && user.dep != PARAM_EXPLICIT) drate -= 1.0;
  analytic = -ftime * drate * PetscExpReal(-rate * ftime);
  PetscCall(VecGetArrayRead(mu, &m));
  gradient = PetscRealPart(m[0]);
  PetscCall(VecRestoreArrayRead(mu, &m));
  /* petscdiff masks floating point numbers unless DIFF_NUMBERS=1, so report the verdict as text */
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "parameter dependence = %s\n", ParamDependences[user.dep]));
  /* the agreement is the positive condition so that a NaN gradient, which compares false against everything, is reported as a failure */
  if (PetscAbsReal(gradient - analytic) <= rtol * PetscAbsReal(analytic)) PetscCall(PetscPrintf(PETSC_COMM_SELF, "  dJ/dp agrees with the analytic gradient to within the time discretization error\n"));
  else PetscCall(PetscPrintf(PETSC_COMM_SELF, "  dJ/dp = %g differs from the analytic gradient %g\n", (double)gradient, (double)analytic));

  PetscCall(VecDestroy(&U));
  PetscCall(VecDestroy(&lambda));
  PetscCall(VecDestroy(&mu));
  PetscCall(MatDestroy(&Jrhs));
  PetscCall(MatDestroy(&Ji));
  PetscCall(MatDestroy(&Jacp));
  PetscCall(MatDestroy(&Jacprhs));
  PetscCall(TSDestroy(&ts));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

    testset:
      requires: !single !complex
      args: -ts_trajectory_type memory

      test:
        suffix: implicit
        args: -param_dependence implicit

      test:
        suffix: explicit
        args: -param_dependence explicit

      test:
        suffix: both
        args: -param_dependence both

      # a matrix registered without a callback must contribute nothing, on the TSTHETA path
      # where TSComputeIJacobianP() is called with imex = PETSC_FALSE
      test:
        suffix: null_ijacobianp
        args: -param_dependence implicit -null_ijacobianp -ts_type beuler

TEST*/
