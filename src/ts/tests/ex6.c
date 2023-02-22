static char help[] = "Solves DAE with integrator only on non-algebraic terms \n";

#include <petscts.h>

/*
        \dot{U} = f(U,V)
        F(U,V)  = 0
*/

/*
   f(U,V) = U + V
*/
PetscErrorCode f(PetscReal t, Vec U, Vec V, Vec F)
{
  PetscFunctionBeginUser;
  PetscCall(VecWAXPY(F, 1.0, U, V));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   F(U,V) = U - V
*/
PetscErrorCode F(PetscReal t, Vec U, Vec V, Vec F)
{
  PetscFunctionBeginUser;
  PetscCall(VecWAXPY(F, -1.0, V, U));
  PetscFunctionReturn(PETSC_SUCCESS);
}

typedef struct {
  PetscReal t;
  SNES      snes;
  Vec       U, V;
  PetscErrorCode (*f)(PetscReal, Vec, Vec, Vec);
  PetscErrorCode (*F)(PetscReal, Vec, Vec, Vec);
} AppCtx;

extern PetscErrorCode TSFunction(TS, PetscReal, Vec, Vec, void *);
extern PetscErrorCode SNESFunction(SNES, Vec, Vec, void *);

int main(int argc, char **argv)
{
  AppCtx ctx;
  TS     ts;
  Vec    tsrhs, U;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetProblemType(ts, TS_NONLINEAR));
  PetscCall(TSSetType(ts, TSEULER));
  PetscCall(TSSetFromOptions(ts));
  PetscCall(VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, 1, &tsrhs));
  PetscCall(VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, 1, &U));
  PetscCall(TSSetRHSFunction(ts, tsrhs, TSFunction, &ctx));
  PetscCall(TSSetMaxTime(ts, 1.0));
  ctx.f = f;

  PetscCall(SNESCreate(PETSC_COMM_WORLD, &ctx.snes));
  PetscCall(SNESSetFromOptions(ctx.snes));
  PetscCall(SNESSetFunction(ctx.snes, NULL, SNESFunction, &ctx));
  PetscCall(SNESSetJacobian(ctx.snes, NULL, NULL, SNESComputeJacobianDefault, &ctx));
  ctx.F = F;
  PetscCall(VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, 1, &ctx.V));

  PetscCall(VecSet(U, 1.0));
  PetscCall(TSSolve(ts, U));

  PetscCall(VecDestroy(&ctx.V));
  PetscCall(VecDestroy(&tsrhs));
  PetscCall(VecDestroy(&U));
  PetscCall(SNESDestroy(&ctx.snes));
  PetscCall(TSDestroy(&ts));
  PetscCall(PetscFinalize());
  return 0;
}

/*
   Defines the RHS function that is passed to the time-integrator.

   Solves F(U,V) for V and then computes f(U,V)
*/
PetscErrorCode TSFunction(TS ts, PetscReal t, Vec U, Vec F, void *actx)
{
  AppCtx *ctx = (AppCtx *)actx;

  PetscFunctionBeginUser;
  ctx->t = t;
  ctx->U = U;
  PetscCall(SNESSolve(ctx->snes, NULL, ctx->V));
  PetscCall((*ctx->f)(t, U, ctx->V, F));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   Defines the nonlinear function that is passed to the nonlinear solver
*/
PetscErrorCode SNESFunction(SNES snes, Vec V, Vec F, void *actx)
{
  AppCtx *ctx = (AppCtx *)actx;

  PetscFunctionBeginUser;
  PetscCall((*ctx->F)(ctx->t, ctx->U, V, F));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*TEST

    test:
      args:  -ts_monitor -ts_view

TEST*/
