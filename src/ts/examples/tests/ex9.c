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
  ierr = VecWAXPY(F,1.0,U,V);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   F(U,V) = U - V

*/
PetscErrorCode F(PetscReal t,Vec U,Vec V,Vec F)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecWAXPY(F,-1.0,V,U);CHKERRQ(ierr);
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


  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSROSW);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = VecCreateMPI(PETSC_COMM_WORLD,2,PETSC_DETERMINE,&tsrhs);CHKERRQ(ierr);
  ierr = VecCreateMPI(PETSC_COMM_WORLD,2,PETSC_DETERMINE,&UV);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,tsrhs,TSFunctionRHS,&ctx);CHKERRQ(ierr);
  ierr = TSSetIFunction(ts,NULL,TSFunctionI,&ctx);CHKERRQ(ierr);
  ctx.f = f;
  ctx.F = F;

  ierr = VecCreateMPI(PETSC_COMM_WORLD,1,PETSC_DETERMINE,&ctx.U);CHKERRQ(ierr);
  ierr = VecCreateMPI(PETSC_COMM_WORLD,1,PETSC_DETERMINE,&ctx.V);CHKERRQ(ierr);
  ierr = VecCreateMPI(PETSC_COMM_WORLD,1,PETSC_DETERMINE,&ctx.UF);CHKERRQ(ierr);
  ierr = VecCreateMPI(PETSC_COMM_WORLD,1,PETSC_DETERMINE,&ctx.VF);CHKERRQ(ierr);
  I    = 2*rank;
  ierr = ISCreateGeneral(PETSC_COMM_WORLD,1,&I,PETSC_COPY_VALUES,&is);CHKERRQ(ierr);
  ierr = VecScatterCreate(ctx.U,NULL,UV,is,&ctx.scatterU);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  I    = 2*rank + 1;
  ierr = ISCreateGeneral(PETSC_COMM_WORLD,1,&I,PETSC_COPY_VALUES,&is);CHKERRQ(ierr);
  ierr = VecScatterCreate(ctx.V,NULL,UV,is,&ctx.scatterV);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);

  ierr = VecSet(UV,1.0);CHKERRQ(ierr);
  ierr = TSSolve(ts,UV);CHKERRQ(ierr);
  ierr = VecDestroy(&tsrhs);CHKERRQ(ierr);
  ierr = VecDestroy(&UV);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx.U);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx.V);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx.UF);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx.VF);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&ctx.scatterU);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&ctx.scatterV);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*
   Defines the RHS function that is passed to the time-integrator. 

*/
PetscErrorCode TSFunctionRHS(TS ts,PetscReal t,Vec UV,Vec F,void *actx)
{
  AppCtx         *ctx = (AppCtx*)actx;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecSet(F,0.0);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx->scatterU,UV,ctx->U,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx->scatterU,UV,ctx->U,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx->scatterV,UV,ctx->V,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx->scatterV,UV,ctx->V,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = (*ctx->f)(t,ctx->U,ctx->V,ctx->UF);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx->scatterU,ctx->UF,F,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx->scatterU,ctx->UF,F,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
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
  ierr = VecCopy(UVdot,F);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx->scatterU,UV,ctx->U,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx->scatterU,UV,ctx->U,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx->scatterV,UV,ctx->V,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx->scatterV,UV,ctx->V,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = (*ctx->F)(t,ctx->U,ctx->V,ctx->VF);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx->scatterV,ctx->VF,F,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx->scatterV,ctx->VF,F,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


