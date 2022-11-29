static char help[] = "Solves DAE with integrator only on non-algebraic terms \n";

#include <petscts.h>

/*
        \dot{U} = f(U,V)
        F(U,V)  = 0

    Same as ex6.c and ex7.c except calls the ARKIMEX integrator on the entire DAE
*/

/*
   f(U,V) = U + V

*/
PetscErrorCode f(PetscReal t, Vec U, Vec V, Vec F)
{
  PetscFunctionBeginUser;
  PetscCall(VecWAXPY(F, 1.0, U, V));
  PetscFunctionReturn(0);
}

/*
   F(U,V) = U - V

*/
PetscErrorCode F(PetscReal t, Vec U, Vec V, Vec F)
{
  PetscFunctionBeginUser;
  PetscCall(VecWAXPY(F, -1.0, V, U));
  PetscFunctionReturn(0);
}

typedef struct {
  Vec        U, V;
  Vec        UF, VF;
  VecScatter scatterU, scatterV;
  PetscErrorCode (*f)(PetscReal, Vec, Vec, Vec);
  PetscErrorCode (*F)(PetscReal, Vec, Vec, Vec);
} AppCtx;

extern PetscErrorCode TSFunctionRHS(TS, PetscReal, Vec, Vec, void *);
extern PetscErrorCode TSFunctionI(TS, PetscReal, Vec, Vec, Vec, void *);

int main(int argc, char **argv)
{
  AppCtx      ctx;
  TS          ts;
  Vec         tsrhs, UV;
  IS          is;
  PetscInt    I;
  PetscMPIInt rank;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetProblemType(ts, TS_NONLINEAR));
  PetscCall(TSSetType(ts, TSROSW));
  PetscCall(TSSetFromOptions(ts));
  PetscCall(VecCreateMPI(PETSC_COMM_WORLD, 2, PETSC_DETERMINE, &tsrhs));
  PetscCall(VecCreateMPI(PETSC_COMM_WORLD, 2, PETSC_DETERMINE, &UV));
  PetscCall(TSSetRHSFunction(ts, tsrhs, TSFunctionRHS, &ctx));
  PetscCall(TSSetIFunction(ts, NULL, TSFunctionI, &ctx));
  ctx.f = f;
  ctx.F = F;

  PetscCall(VecCreateMPI(PETSC_COMM_WORLD, 1, PETSC_DETERMINE, &ctx.U));
  PetscCall(VecCreateMPI(PETSC_COMM_WORLD, 1, PETSC_DETERMINE, &ctx.V));
  PetscCall(VecCreateMPI(PETSC_COMM_WORLD, 1, PETSC_DETERMINE, &ctx.UF));
  PetscCall(VecCreateMPI(PETSC_COMM_WORLD, 1, PETSC_DETERMINE, &ctx.VF));
  I = 2 * rank;
  PetscCall(ISCreateGeneral(PETSC_COMM_WORLD, 1, &I, PETSC_COPY_VALUES, &is));
  PetscCall(VecScatterCreate(ctx.U, NULL, UV, is, &ctx.scatterU));
  PetscCall(ISDestroy(&is));
  I = 2 * rank + 1;
  PetscCall(ISCreateGeneral(PETSC_COMM_WORLD, 1, &I, PETSC_COPY_VALUES, &is));
  PetscCall(VecScatterCreate(ctx.V, NULL, UV, is, &ctx.scatterV));
  PetscCall(ISDestroy(&is));

  PetscCall(VecSet(UV, 1.0));
  PetscCall(TSSolve(ts, UV));
  PetscCall(VecDestroy(&tsrhs));
  PetscCall(VecDestroy(&UV));
  PetscCall(VecDestroy(&ctx.U));
  PetscCall(VecDestroy(&ctx.V));
  PetscCall(VecDestroy(&ctx.UF));
  PetscCall(VecDestroy(&ctx.VF));
  PetscCall(VecScatterDestroy(&ctx.scatterU));
  PetscCall(VecScatterDestroy(&ctx.scatterV));
  PetscCall(TSDestroy(&ts));
  PetscCall(PetscFinalize());
  return 0;
}

/*
   Defines the RHS function that is passed to the time-integrator.

*/
PetscErrorCode TSFunctionRHS(TS ts, PetscReal t, Vec UV, Vec F, void *actx)
{
  AppCtx *ctx = (AppCtx *)actx;

  PetscFunctionBeginUser;
  PetscCall(VecSet(F, 0.0));
  PetscCall(VecScatterBegin(ctx->scatterU, UV, ctx->U, INSERT_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterEnd(ctx->scatterU, UV, ctx->U, INSERT_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterBegin(ctx->scatterV, UV, ctx->V, INSERT_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterEnd(ctx->scatterV, UV, ctx->V, INSERT_VALUES, SCATTER_REVERSE));
  PetscCall((*ctx->f)(t, ctx->U, ctx->V, ctx->UF));
  PetscCall(VecScatterBegin(ctx->scatterU, ctx->UF, F, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(ctx->scatterU, ctx->UF, F, INSERT_VALUES, SCATTER_FORWARD));
  PetscFunctionReturn(0);
}

/*
   Defines the nonlinear function that is passed to the time-integrator

*/
PetscErrorCode TSFunctionI(TS ts, PetscReal t, Vec UV, Vec UVdot, Vec F, void *actx)
{
  AppCtx *ctx = (AppCtx *)actx;

  PetscFunctionBeginUser;
  PetscCall(VecCopy(UVdot, F));
  PetscCall(VecScatterBegin(ctx->scatterU, UV, ctx->U, INSERT_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterEnd(ctx->scatterU, UV, ctx->U, INSERT_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterBegin(ctx->scatterV, UV, ctx->V, INSERT_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterEnd(ctx->scatterV, UV, ctx->V, INSERT_VALUES, SCATTER_REVERSE));
  PetscCall((*ctx->F)(t, ctx->U, ctx->V, ctx->VF));
  PetscCall(VecScatterBegin(ctx->scatterV, ctx->VF, F, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(ctx->scatterV, ctx->VF, F, INSERT_VALUES, SCATTER_FORWARD));
  PetscFunctionReturn(0);
}
