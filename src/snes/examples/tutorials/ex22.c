
static char help[] = "Solves PDE optimization problem.\n\n";

#include "petscda.h"
#include "petscpf.h"
#include "petscsnes.h"
#include "petscdmmg.h"

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

       See ex21.c for the same code, but that does NOT interlaces the u and the lambda

       The vectors u_lambda and fu_lambda contain the u and the lambda interlaced
*/

typedef struct {
  PetscViewer  u_lambda_viewer;
  PetscViewer  fu_lambda_viewer;
} UserCtx;

extern PetscErrorCode FormFunction(SNES,Vec,Vec,void*);
extern PetscErrorCode Monitor(SNES,PetscInt,PetscReal,void*);


#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  UserCtx        user;
  DA             da;
  DMMG           *dmmg;
  VecPack        packer;

  PetscInitialize(&argc,&argv,PETSC_NULL,help);

  /* Hardwire several options; can be changed at command line */
  ierr = PetscOptionsSetValue("-dmmg_grid_sequence",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsSetValue("-ksp_type","fgmres");CHKERRQ(ierr);
  ierr = PetscOptionsSetValue("-ksp_max_it","5");CHKERRQ(ierr);
  ierr = PetscOptionsSetValue("-pc_mg_type","full");CHKERRQ(ierr);
  ierr = PetscOptionsSetValue("-mg_coarse_ksp_type","gmres");CHKERRQ(ierr);
  ierr = PetscOptionsSetValue("-mg_levels_ksp_type","gmres");CHKERRQ(ierr);
  ierr = PetscOptionsSetValue("-mg_coarse_ksp_max_it","6");CHKERRQ(ierr);
  ierr = PetscOptionsSetValue("-mg_levels_ksp_max_it","3");CHKERRQ(ierr);
  ierr = PetscOptionsSetValue("-snes_mf_type","wp");CHKERRQ(ierr);
  ierr = PetscOptionsSetValue("-snes_mf_compute_norma","no");CHKERRQ(ierr);
  ierr = PetscOptionsSetValue("-snes_mf_compute_normu","no");CHKERRQ(ierr);
  ierr = PetscOptionsSetValue("-snes_ls","basic");CHKERRQ(ierr);
  ierr = PetscOptionsSetValue("-dmmg_jacobian_mf_fd",0);CHKERRQ(ierr);
  /* ierr = PetscOptionsSetValue("-snes_ls","basicnonorms");CHKERRQ(ierr); */
  ierr = PetscOptionsInsert(&argc,&argv,PETSC_NULL);CHKERRQ(ierr); 

  /* Create a global vector that includes a single redundant array and two da arrays */
  ierr = VecPackCreate(PETSC_COMM_WORLD,&packer);CHKERRQ(ierr);
  ierr = VecPackAddArray(packer,1);CHKERRQ(ierr);
  ierr = DACreate1d(PETSC_COMM_WORLD,DA_NONPERIODIC,-5,2,1,PETSC_NULL,&da);CHKERRQ(ierr);
  ierr = VecPackAddDA(packer,da);CHKERRQ(ierr);

  /* create graphics windows */
  ierr = PetscViewerDrawOpen(PETSC_COMM_WORLD,0,"u_lambda - state variables and Lagrange multipliers",-1,-1,-1,-1,&user.u_lambda_viewer);CHKERRQ(ierr);
  ierr = PetscViewerDrawOpen(PETSC_COMM_WORLD,0,"fu_lambda - derivate w.r.t. state variables and Lagrange multipliers",-1,-1,-1,-1,&user.fu_lambda_viewer);CHKERRQ(ierr);

  /* create nonlinear multi-level solver */
  ierr = DMMGCreate(PETSC_COMM_WORLD,2,&user,&dmmg);CHKERRQ(ierr);
  ierr = DMMGSetDM(dmmg,(DM)packer);CHKERRQ(ierr);
  ierr = DMMGSetSNES(dmmg,FormFunction,PETSC_NULL);CHKERRQ(ierr);
  /*
  for (i=0; i<DMMGGetLevels(dmmg); i++) {
    ierr = SNESSetMonitor(dmmg[i]->snes,Monitor,dmmg[i],0);CHKERRQ(ierr); 
  }*/
  ierr = DMMGSolve(dmmg);CHKERRQ(ierr);
  ierr = DMMGDestroy(dmmg);CHKERRQ(ierr);

  ierr = DADestroy(da);CHKERRQ(ierr);
  ierr = VecPackDestroy(packer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(user.u_lambda_viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(user.fu_lambda_viewer);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

typedef struct {
  PetscScalar u;
  PetscScalar lambda;
} ULambda;
 
/*
      Evaluates FU = Gradiant(L(w,u,lambda))

     This local function acts on the ghosted version of U (accessed via VecPackGetLocalVectors() and
   VecPackScatter()) BUT the global, nonghosted version of FU (via VecPackGetAccess()).

*/
PetscErrorCode FormFunction(SNES snes,Vec U,Vec FU,void* dummy)
{
  DMMG           dmmg = (DMMG)dummy;
  PetscErrorCode ierr;
  PetscInt       xs,xm,i,N,nredundant;
  ULambda        *u_lambda,*fu_lambda;
  PetscScalar    d,h,*w,*fw;
  Vec            vu_lambda,vfu_lambda;
  DA             da;
  VecPack        packer = (VecPack)dmmg->dm;

  PetscFunctionBegin;
  ierr = VecPackGetEntries(packer,&nredundant,&da);CHKERRQ(ierr);
  ierr = VecPackGetLocalVectors(packer,&w,&vu_lambda);CHKERRQ(ierr);
  ierr = VecPackScatter(packer,U,w,vu_lambda);CHKERRQ(ierr);
  ierr = VecPackGetAccess(packer,FU,&fw,&vfu_lambda);CHKERRQ(ierr);

  ierr = DAGetCorners(da,&xs,PETSC_NULL,PETSC_NULL,&xm,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = DAGetInfo(da,0,&N,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DAVecGetArray(da,vu_lambda,&u_lambda);CHKERRQ(ierr);
  ierr = DAVecGetArray(da,vfu_lambda,&fu_lambda);CHKERRQ(ierr);
  d    = N-1.0;
  h    = 1.0/d;

  /* derivative of L() w.r.t. w */
  if (xs == 0) { /* only first processor computes this */
    fw[0] = -2.0*d*u_lambda[0].lambda;
  }

  /* derivative of L() w.r.t. u */
  for (i=xs; i<xs+xm; i++) {
    if      (i == 0)   fu_lambda[0].lambda   =    h*u_lambda[0].u   + 2.*d*u_lambda[0].lambda   - d*u_lambda[1].lambda;
    else if (i == 1)   fu_lambda[1].lambda   = 2.*h*u_lambda[1].u   + 2.*d*u_lambda[1].lambda   - d*u_lambda[2].lambda;
    else if (i == N-1) fu_lambda[N-1].lambda =    h*u_lambda[N-1].u + 2.*d*u_lambda[N-1].lambda - d*u_lambda[N-2].lambda;
    else if (i == N-2) fu_lambda[N-2].lambda = 2.*h*u_lambda[N-2].u + 2.*d*u_lambda[N-2].lambda - d*u_lambda[N-3].lambda;
    else               fu_lambda[i].lambda   = 2.*h*u_lambda[i].u   - d*(u_lambda[i+1].lambda - 2.0*u_lambda[i].lambda + u_lambda[i-1].lambda);
  } 

  /* derivative of L() w.r.t. lambda */
  for (i=xs; i<xs+xm; i++) {
    if      (i == 0)   fu_lambda[0].u   = 2.0*d*(u_lambda[0].u - w[0]);
    else if (i == N-1) fu_lambda[N-1].u = 2.0*d*u_lambda[N-1].u;
    else               fu_lambda[i].u   = -(d*(u_lambda[i+1].u - 2.0*u_lambda[i].u + u_lambda[i-1].u) - 2.0*h);
  } 

  ierr = DAVecRestoreArray(da,vu_lambda,&u_lambda);CHKERRQ(ierr);
  ierr = DAVecRestoreArray(da,vfu_lambda,&fu_lambda);CHKERRQ(ierr);
  ierr = VecPackRestoreLocalVectors(packer,&w,&vu_lambda);CHKERRQ(ierr);
  ierr = VecPackRestoreAccess(packer,FU,&fw,&vfu_lambda);CHKERRQ(ierr);
  ierr = PetscLogFlops(13*N);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* 
    Computes the exact solution
*/
PetscErrorCode u_solution(void *dummy,PetscInt n,PetscScalar *x,PetscScalar *u)
{
  PetscInt i;
  PetscFunctionBegin;
  for (i=0; i<n; i++) {
    u[2*i] = x[i]*x[i] - 1.25*x[i] + .25;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode ExactSolution(VecPack packer,Vec U) 
{
  PF             pf;
  Vec            x,u_global;
  PetscScalar    *w;
  DA             da;
  PetscErrorCode ierr;
  PetscInt       m;

  PetscFunctionBegin;
  ierr = VecPackGetEntries(packer,&m,&da);CHKERRQ(ierr);

  ierr = PFCreate(PETSC_COMM_WORLD,1,1,&pf);CHKERRQ(ierr);
  ierr = PFSetType(pf,PFQUICK,(void*)u_solution);CHKERRQ(ierr);
  ierr = DAGetCoordinates(da,&x);CHKERRQ(ierr);
  if (!x) {
    ierr = DASetUniformCoordinates(da,0.0,1.0,0.0,1.0,0.0,1.0);CHKERRQ(ierr);
    ierr = DAGetCoordinates(da,&x);CHKERRQ(ierr);
  }
  ierr = VecPackGetAccess(packer,U,&w,&u_global,0);CHKERRQ(ierr);
  if (w) w[0] = .25;
  ierr = PFApplyVec(pf,x,u_global);CHKERRQ(ierr);
  ierr = PFDestroy(pf);CHKERRQ(ierr);
  ierr = VecPackRestoreAccess(packer,U,&w,&u_global,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode Monitor(SNES snes,PetscInt its,PetscReal rnorm,void *dummy)
{
  DMMG           dmmg = (DMMG)dummy;
  UserCtx        *user = (UserCtx*)dmmg->user;
  PetscErrorCode ierr;
  PetscInt       m,N;
  PetscScalar    mone = -1.0,*w,*dw;
  Vec            u_lambda,U,F,Uexact;
  VecPack        packer = (VecPack)dmmg->dm;
  PetscReal      norm;
  DA             da;

  PetscFunctionBegin;
  ierr = SNESGetSolution(snes,&U);CHKERRQ(ierr);
  ierr = VecPackGetAccess(packer,U,&w,&u_lambda);CHKERRQ(ierr);
  ierr = VecView(u_lambda,user->u_lambda_viewer); 
  ierr = VecPackRestoreAccess(packer,U,&w,&u_lambda);CHKERRQ(ierr);

  ierr = SNESGetFunction(snes,&F,0,0);CHKERRQ(ierr);
  ierr = VecPackGetAccess(packer,F,&w,&u_lambda);CHKERRQ(ierr);
  /* ierr = VecView(u_lambda,user->fu_lambda_viewer); */
  ierr = VecPackRestoreAccess(packer,U,&w,&u_lambda);CHKERRQ(ierr);

  ierr = VecPackGetEntries(packer,&m,&da);CHKERRQ(ierr);
  ierr = DAGetInfo(da,0,&N,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = VecDuplicate(U,&Uexact);CHKERRQ(ierr);
  ierr = ExactSolution(packer,Uexact);CHKERRQ(ierr);
  ierr = VecAXPY(&mone,U,Uexact);CHKERRQ(ierr);
  ierr = VecPackGetAccess(packer,Uexact,&dw,&u_lambda);CHKERRQ(ierr);
  ierr = VecStrideNorm(u_lambda,0,NORM_2,&norm);CHKERRQ(ierr);
  norm = norm/sqrt(N-1.);
  if (dw) ierr = PetscPrintf(dmmg->comm,"Norm of error %g Error at x = 0 %g\n",norm,PetscRealPart(dw[0]));CHKERRQ(ierr);
  ierr = VecView(u_lambda,user->fu_lambda_viewer);
  ierr = VecPackRestoreAccess(packer,Uexact,&dw,&u_lambda);CHKERRQ(ierr);
  ierr = VecDestroy(Uexact);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}








