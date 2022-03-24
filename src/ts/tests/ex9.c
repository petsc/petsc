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
PetscErrorCode f(PetscReal t,Vec U,Vec V,Vec F)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  CHKERRQ(VecWAXPY(F,1.0,U,V));
  PetscFunctionReturn(0);
}

/*
   F(U,V) = U - V

*/
PetscErrorCode F(PetscReal t,Vec U,Vec V,Vec F)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  CHKERRQ(VecWAXPY(F,-1.0,V,U));
  PetscFunctionReturn(0);
}

typedef struct {
  Vec            U,V;
  Vec            UF,VF;
  VecScatter     scatterU,scatterV;
  PetscErrorCode (*f)(PetscReal,Vec,Vec,Vec);
  PetscErrorCode (*F)(PetscReal,Vec,Vec,Vec);
} AppCtx;

extern PetscErrorCode TSFunctionRHS(TS,PetscReal,Vec,Vec,void*);
extern PetscErrorCode TSFunctionI(TS,PetscReal,Vec,Vec,Vec,void*);

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  AppCtx         ctx;
  TS             ts;
  Vec            tsrhs,UV;
  IS             is;
  PetscInt       I;
  PetscMPIInt    rank;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRQ(TSCreate(PETSC_COMM_WORLD,&ts));
  CHKERRQ(TSSetProblemType(ts,TS_NONLINEAR));
  CHKERRQ(TSSetType(ts,TSROSW));
  CHKERRQ(TSSetFromOptions(ts));
  CHKERRQ(VecCreateMPI(PETSC_COMM_WORLD,2,PETSC_DETERMINE,&tsrhs));
  CHKERRQ(VecCreateMPI(PETSC_COMM_WORLD,2,PETSC_DETERMINE,&UV));
  CHKERRQ(TSSetRHSFunction(ts,tsrhs,TSFunctionRHS,&ctx));
  CHKERRQ(TSSetIFunction(ts,NULL,TSFunctionI,&ctx));
  ctx.f = f;
  ctx.F = F;

  CHKERRQ(VecCreateMPI(PETSC_COMM_WORLD,1,PETSC_DETERMINE,&ctx.U));
  CHKERRQ(VecCreateMPI(PETSC_COMM_WORLD,1,PETSC_DETERMINE,&ctx.V));
  CHKERRQ(VecCreateMPI(PETSC_COMM_WORLD,1,PETSC_DETERMINE,&ctx.UF));
  CHKERRQ(VecCreateMPI(PETSC_COMM_WORLD,1,PETSC_DETERMINE,&ctx.VF));
  I    = 2*rank;
  CHKERRQ(ISCreateGeneral(PETSC_COMM_WORLD,1,&I,PETSC_COPY_VALUES,&is));
  CHKERRQ(VecScatterCreate(ctx.U,NULL,UV,is,&ctx.scatterU));
  CHKERRQ(ISDestroy(&is));
  I    = 2*rank + 1;
  CHKERRQ(ISCreateGeneral(PETSC_COMM_WORLD,1,&I,PETSC_COPY_VALUES,&is));
  CHKERRQ(VecScatterCreate(ctx.V,NULL,UV,is,&ctx.scatterV));
  CHKERRQ(ISDestroy(&is));

  CHKERRQ(VecSet(UV,1.0));
  CHKERRQ(TSSolve(ts,UV));
  CHKERRQ(VecDestroy(&tsrhs));
  CHKERRQ(VecDestroy(&UV));
  CHKERRQ(VecDestroy(&ctx.U));
  CHKERRQ(VecDestroy(&ctx.V));
  CHKERRQ(VecDestroy(&ctx.UF));
  CHKERRQ(VecDestroy(&ctx.VF));
  CHKERRQ(VecScatterDestroy(&ctx.scatterU));
  CHKERRQ(VecScatterDestroy(&ctx.scatterV));
  CHKERRQ(TSDestroy(&ts));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*
   Defines the RHS function that is passed to the time-integrator.

*/
PetscErrorCode TSFunctionRHS(TS ts,PetscReal t,Vec UV,Vec F,void *actx)
{
  AppCtx         *ctx = (AppCtx*)actx;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  CHKERRQ(VecSet(F,0.0));
  CHKERRQ(VecScatterBegin(ctx->scatterU,UV,ctx->U,INSERT_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecScatterEnd(ctx->scatterU,UV,ctx->U,INSERT_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecScatterBegin(ctx->scatterV,UV,ctx->V,INSERT_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecScatterEnd(ctx->scatterV,UV,ctx->V,INSERT_VALUES,SCATTER_REVERSE));
  CHKERRQ((*ctx->f)(t,ctx->U,ctx->V,ctx->UF));
  CHKERRQ(VecScatterBegin(ctx->scatterU,ctx->UF,F,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(ctx->scatterU,ctx->UF,F,INSERT_VALUES,SCATTER_FORWARD));
  PetscFunctionReturn(0);
}

/*
   Defines the nonlinear function that is passed to the time-integrator

*/
PetscErrorCode TSFunctionI(TS ts,PetscReal t,Vec UV,Vec UVdot,Vec F,void *actx)
{
  AppCtx         *ctx = (AppCtx*)actx;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  CHKERRQ(VecCopy(UVdot,F));
  CHKERRQ(VecScatterBegin(ctx->scatterU,UV,ctx->U,INSERT_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecScatterEnd(ctx->scatterU,UV,ctx->U,INSERT_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecScatterBegin(ctx->scatterV,UV,ctx->V,INSERT_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecScatterEnd(ctx->scatterV,UV,ctx->V,INSERT_VALUES,SCATTER_REVERSE));
  CHKERRQ((*ctx->F)(t,ctx->U,ctx->V,ctx->VF));
  CHKERRQ(VecScatterBegin(ctx->scatterV,ctx->VF,F,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(ctx->scatterV,ctx->VF,F,INSERT_VALUES,SCATTER_FORWARD));
  PetscFunctionReturn(0);
}
