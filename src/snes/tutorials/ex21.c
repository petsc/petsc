
static const char help[] = "Solves PDE optimization problem using full-space method, treats state and adjoint variables separately.\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmredundant.h>
#include <petscdmcomposite.h>
#include <petscpf.h>
#include <petscsnes.h>

/*

       w - design variables (what we change to get an optimal solution)
       u - state variables (i.e. the PDE solution)
       lambda - the Lagrange multipliers

            U = (w u lambda)

       fu, fw, flambda contain the gradient of L(w,u,lambda)

            FU = (fw fu flambda)

       In this example the PDE is
                             Uxx = 2,
                            u(0) = w(0), thus this is the free parameter
                            u(1) = 0
       the function we wish to minimize is
                            \integral u^{2}

       The exact solution for u is given by u(x) = x*x - 1.25*x + .25

       Use the usual centered finite differences.

       Note we treat the problem as non-linear though it happens to be linear

       See ex22.c for the same code, but that interlaces the u and the lambda

*/

typedef struct {
  DM          red1,da1,da2;
  DM          packer;
  PetscViewer u_viewer,lambda_viewer;
  PetscViewer fu_viewer,flambda_viewer;
} UserCtx;

extern PetscErrorCode FormFunction(SNES,Vec,Vec,void*);
extern PetscErrorCode Monitor(SNES,PetscInt,PetscReal,void*);

int main(int argc,char **argv)
{
  PetscInt       its;
  Vec            U,FU;
  SNES           snes;
  UserCtx        user;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));

  /* Create a global vector that includes a single redundant array and two da arrays */
  CHKERRQ(DMCompositeCreate(PETSC_COMM_WORLD,&user.packer));
  CHKERRQ(DMRedundantCreate(PETSC_COMM_WORLD,0,1,&user.red1));
  CHKERRQ(DMCompositeAddDM(user.packer,user.red1));
  CHKERRQ(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,5,1,1,NULL,&user.da1));
  CHKERRQ(DMSetFromOptions(user.da1));
  CHKERRQ(DMSetUp(user.da1));
  CHKERRQ(DMCompositeAddDM(user.packer,user.da1));
  CHKERRQ(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,5,1,1,NULL,&user.da2));
  CHKERRQ(DMSetFromOptions(user.da2));
  CHKERRQ(DMSetUp(user.da2));
  CHKERRQ(DMDASetFieldName(user.da1,0,"u"));
  CHKERRQ(DMDASetFieldName(user.da2,0,"lambda"));
  CHKERRQ(DMCompositeAddDM(user.packer,user.da2));
  CHKERRQ(DMCreateGlobalVector(user.packer,&U));
  CHKERRQ(VecDuplicate(U,&FU));

  /* create graphics windows */
  CHKERRQ(PetscViewerDrawOpen(PETSC_COMM_WORLD,0,"u - state variables",-1,-1,-1,-1,&user.u_viewer));
  CHKERRQ(PetscViewerDrawOpen(PETSC_COMM_WORLD,0,"lambda - Lagrange multipliers",-1,-1,-1,-1,&user.lambda_viewer));
  CHKERRQ(PetscViewerDrawOpen(PETSC_COMM_WORLD,0,"fu - derivate w.r.t. state variables",-1,-1,-1,-1,&user.fu_viewer));
  CHKERRQ(PetscViewerDrawOpen(PETSC_COMM_WORLD,0,"flambda - derivate w.r.t. Lagrange multipliers",-1,-1,-1,-1,&user.flambda_viewer));

  /* create nonlinear solver */
  CHKERRQ(SNESCreate(PETSC_COMM_WORLD,&snes));
  CHKERRQ(SNESSetFunction(snes,FU,FormFunction,&user));
  CHKERRQ(SNESSetFromOptions(snes));
  CHKERRQ(SNESMonitorSet(snes,Monitor,&user,0));
  CHKERRQ(SNESSolve(snes,NULL,U));
  CHKERRQ(SNESGetIterationNumber(snes,&its));
  CHKERRQ(SNESDestroy(&snes));

  CHKERRQ(DMDestroy(&user.red1));
  CHKERRQ(DMDestroy(&user.da1));
  CHKERRQ(DMDestroy(&user.da2));
  CHKERRQ(DMDestroy(&user.packer));
  CHKERRQ(VecDestroy(&U));
  CHKERRQ(VecDestroy(&FU));
  CHKERRQ(PetscViewerDestroy(&user.u_viewer));
  CHKERRQ(PetscViewerDestroy(&user.lambda_viewer));
  CHKERRQ(PetscViewerDestroy(&user.fu_viewer));
  CHKERRQ(PetscViewerDestroy(&user.flambda_viewer));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*
      Evaluates FU = Gradiant(L(w,u,lambda))

*/
PetscErrorCode FormFunction(SNES snes,Vec U,Vec FU,void *dummy)
{
  UserCtx        *user = (UserCtx*)dummy;
  PetscInt       xs,xm,i,N;
  PetscScalar    *u,*lambda,*w,*fu,*fw,*flambda,d,h;
  Vec            vw,vu,vlambda,vfw,vfu,vflambda;

  PetscFunctionBeginUser;
  CHKERRQ(DMCompositeGetLocalVectors(user->packer,&vw,&vu,&vlambda));
  CHKERRQ(DMCompositeGetLocalVectors(user->packer,&vfw,&vfu,&vflambda));
  CHKERRQ(DMCompositeScatter(user->packer,U,vw,vu,vlambda));

  CHKERRQ(DMDAGetCorners(user->da1,&xs,NULL,NULL,&xm,NULL,NULL));
  CHKERRQ(DMDAGetInfo(user->da1,0,&N,0,0,0,0,0,0,0,0,0,0,0));
  CHKERRQ(VecGetArray(vw,&w));
  CHKERRQ(VecGetArray(vfw,&fw));
  CHKERRQ(DMDAVecGetArray(user->da1,vu,&u));
  CHKERRQ(DMDAVecGetArray(user->da1,vfu,&fu));
  CHKERRQ(DMDAVecGetArray(user->da1,vlambda,&lambda));
  CHKERRQ(DMDAVecGetArray(user->da1,vflambda,&flambda));
  d    = (N-1.0);
  h    = 1.0/d;

  /* derivative of L() w.r.t. w */
  if (xs == 0) { /* only first processor computes this */
    fw[0] = -2.*d*lambda[0];
  }

  /* derivative of L() w.r.t. u */
  for (i=xs; i<xs+xm; i++) {
    if      (i == 0)   flambda[0]   =    h*u[0]   + 2.*d*lambda[0]   - d*lambda[1];
    else if (i == 1)   flambda[1]   = 2.*h*u[1]   + 2.*d*lambda[1]   - d*lambda[2];
    else if (i == N-1) flambda[N-1] =    h*u[N-1] + 2.*d*lambda[N-1] - d*lambda[N-2];
    else if (i == N-2) flambda[N-2] = 2.*h*u[N-2] + 2.*d*lambda[N-2] - d*lambda[N-3];
    else               flambda[i]   = 2.*h*u[i]   - d*(lambda[i+1] - 2.0*lambda[i] + lambda[i-1]);
  }

  /* derivative of L() w.r.t. lambda */
  for (i=xs; i<xs+xm; i++) {
    if      (i == 0)   fu[0]   = 2.0*d*(u[0] - w[0]);
    else if (i == N-1) fu[N-1] = 2.0*d*u[N-1];
    else               fu[i]   = -(d*(u[i+1] - 2.0*u[i] + u[i-1]) - 2.0*h);
  }

  CHKERRQ(VecRestoreArray(vw,&w));
  CHKERRQ(VecRestoreArray(vfw,&fw));
  CHKERRQ(DMDAVecRestoreArray(user->da1,vu,&u));
  CHKERRQ(DMDAVecRestoreArray(user->da1,vfu,&fu));
  CHKERRQ(DMDAVecRestoreArray(user->da1,vlambda,&lambda));
  CHKERRQ(DMDAVecRestoreArray(user->da1,vflambda,&flambda));

  CHKERRQ(DMCompositeGather(user->packer,INSERT_VALUES,FU,vfw,vfu,vflambda));
  CHKERRQ(DMCompositeRestoreLocalVectors(user->packer,&vw,&vu,&vlambda));
  CHKERRQ(DMCompositeRestoreLocalVectors(user->packer,&vfw,&vfu,&vflambda));
  PetscFunctionReturn(0);
}

PetscErrorCode Monitor(SNES snes,PetscInt its,PetscReal rnorm,void *dummy)
{
  UserCtx        *user = (UserCtx*)dummy;
  Vec            w,u,lambda,U,F;

  PetscFunctionBeginUser;
  CHKERRQ(SNESGetSolution(snes,&U));
  CHKERRQ(DMCompositeGetAccess(user->packer,U,&w,&u,&lambda));
  CHKERRQ(VecView(u,user->u_viewer));
  CHKERRQ(VecView(lambda,user->lambda_viewer));
  CHKERRQ(DMCompositeRestoreAccess(user->packer,U,&w,&u,&lambda));

  CHKERRQ(SNESGetFunction(snes,&F,0,0));
  CHKERRQ(DMCompositeGetAccess(user->packer,F,&w,&u,&lambda));
  CHKERRQ(VecView(u,user->fu_viewer));
  CHKERRQ(VecView(lambda,user->flambda_viewer));
  CHKERRQ(DMCompositeRestoreAccess(user->packer,F,&w,&u,&lambda));
  PetscFunctionReturn(0);
}

/*TEST

   test:
      nsize: 4
      args: -snes_linesearch_monitor -snes_mf -pc_type none -snes_monitor_short -nox -ksp_monitor_short -snes_converged_reason
      requires: !single

TEST*/
