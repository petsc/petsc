
static const char help[] = "Solves PDE optimization problem using full-space method, treats state and adjoint variables separately.\n\n";

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
  DM           red1,da1,da2;
  DM           packer;
  PetscViewer  u_viewer,lambda_viewer;
  PetscViewer  fu_viewer,flambda_viewer;
} UserCtx;

extern PetscErrorCode FormFunction(SNES,Vec,Vec,void*);
extern PetscErrorCode Monitor(SNES,PetscInt,PetscReal,void*);


#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       its;
  Vec            U,FU;
  SNES           snes;
  UserCtx        user;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr);

  /* Create a global vector that includes a single redundant array and two da arrays */
  ierr = DMCompositeCreate(PETSC_COMM_WORLD,&user.packer);CHKERRQ(ierr);
  ierr = DMRedundantCreate(PETSC_COMM_WORLD,0,1,&user.red1);CHKERRQ(ierr);
  ierr = DMCompositeAddDM(user.packer,user.red1);CHKERRQ(ierr);
  ierr = DMDACreate1d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE,-5,1,1,PETSC_NULL,&user.da1);CHKERRQ(ierr);
  ierr = DMCompositeAddDM(user.packer,user.da1);CHKERRQ(ierr);
  ierr = DMDACreate1d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE,-5,1,1,PETSC_NULL,&user.da2);CHKERRQ(ierr);
  ierr = DMCompositeAddDM(user.packer,user.da2);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(user.packer,&U);CHKERRQ(ierr);
  ierr = VecDuplicate(U,&FU);CHKERRQ(ierr);

  /* create graphics windows */
  ierr = PetscViewerDrawOpen(PETSC_COMM_WORLD,0,"u - state variables",-1,-1,-1,-1,&user.u_viewer);CHKERRQ(ierr);
  ierr = PetscViewerDrawOpen(PETSC_COMM_WORLD,0,"lambda - Lagrange multipliers",-1,-1,-1,-1,&user.lambda_viewer);CHKERRQ(ierr);
  ierr = PetscViewerDrawOpen(PETSC_COMM_WORLD,0,"fu - derivate w.r.t. state variables",-1,-1,-1,-1,&user.fu_viewer);CHKERRQ(ierr);
  ierr = PetscViewerDrawOpen(PETSC_COMM_WORLD,0,"flambda - derivate w.r.t. Lagrange multipliers",-1,-1,-1,-1,&user.flambda_viewer);CHKERRQ(ierr);


  /* create nonlinear solver */
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
  ierr = SNESSetFunction(snes,FU,FormFunction,&user);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  ierr = SNESMonitorSet(snes,Monitor,&user,0);CHKERRQ(ierr);
  ierr = SNESSolve(snes,PETSC_NULL,U);CHKERRQ(ierr);
  ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);

  ierr = DMDestroy(&user.red1);CHKERRQ(ierr);
  ierr = DMDestroy(&user.da1);CHKERRQ(ierr);
  ierr = DMDestroy(&user.da2);CHKERRQ(ierr);
  ierr = DMDestroy(&user.packer);CHKERRQ(ierr);
  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = VecDestroy(&FU);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&user.u_viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&user.lambda_viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&user.fu_viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&user.flambda_viewer);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "FormFunction"
/*
      Evaluates FU = Gradiant(L(w,u,lambda))

*/
PetscErrorCode FormFunction(SNES snes,Vec U,Vec FU,void* dummy)
{
  UserCtx        *user = (UserCtx*)dummy;
  PetscErrorCode ierr;
  PetscInt       xs,xm,i,N;
  PetscScalar    *u,*lambda,*w,*fu,*fw,*flambda,d,h;
  Vec            vw,vu,vlambda,vfw,vfu,vflambda;

  PetscFunctionBegin;
  ierr = DMCompositeGetLocalVectors(user->packer,&vw,&vu,&vlambda);CHKERRQ(ierr);
  ierr = DMCompositeGetLocalVectors(user->packer,&vfw,&vfu,&vflambda);CHKERRQ(ierr);
  ierr = DMCompositeScatter(user->packer,U,vw,vu,vlambda);CHKERRQ(ierr);

  ierr = DMDAGetCorners(user->da1,&xs,PETSC_NULL,PETSC_NULL,&xm,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = DMDAGetInfo(user->da1,0,&N,0,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = VecGetArray(vw,&w);CHKERRQ(ierr);
  ierr = VecGetArray(vfw,&fw);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->da1,vu,&u);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->da1,vfu,&fu);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->da1,vlambda,&lambda);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->da1,vflambda,&flambda);CHKERRQ(ierr);
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

  ierr = VecRestoreArray(vw,&w);CHKERRQ(ierr);
  ierr = VecRestoreArray(vfw,&fw);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->da1,vu,&u);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->da1,vfu,&fu);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->da1,vlambda,&lambda);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->da1,vflambda,&flambda);CHKERRQ(ierr);

  ierr = DMCompositeGather(user->packer,FU,INSERT_VALUES,vfw,vfu,vflambda);CHKERRQ(ierr);
  ierr = DMCompositeRestoreLocalVectors(user->packer,&vw,&vu,&vlambda);CHKERRQ(ierr);
  ierr = DMCompositeRestoreLocalVectors(user->packer,&vfw,&vfu,&vflambda);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Monitor"
PetscErrorCode Monitor(SNES snes,PetscInt its,PetscReal rnorm,void *dummy)
{
  UserCtx        *user = (UserCtx*)dummy;
  PetscErrorCode ierr;
  Vec            w,u,lambda,U,F;

  PetscFunctionBegin;
  ierr = SNESGetSolution(snes,&U);CHKERRQ(ierr);
  ierr = DMCompositeGetAccess(user->packer,U,&w,&u,&lambda);CHKERRQ(ierr);
  ierr = VecView(u,user->u_viewer);CHKERRQ(ierr);
  ierr = VecView(lambda,user->lambda_viewer);CHKERRQ(ierr);
  ierr = DMCompositeRestoreAccess(user->packer,U,&w,&u,&lambda);CHKERRQ(ierr);

  ierr = SNESGetFunction(snes,&F,0,0);CHKERRQ(ierr);
  ierr = DMCompositeGetAccess(user->packer,F,&w,&u,&lambda);CHKERRQ(ierr);
  ierr = VecView(u,user->fu_viewer);CHKERRQ(ierr);
  ierr = VecView(lambda,user->flambda_viewer);CHKERRQ(ierr);
  ierr = DMCompositeRestoreAccess(user->packer,F,&w,&u,&lambda);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



