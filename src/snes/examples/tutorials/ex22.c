/*$Id: ex22.c,v 1.8 2000/12/18 17:31:24 bsmith Exp bsmith $*/

static char help[] = "Solves PDE optimization problem\n\n";

#include "petscda.h"
#include "petscpf.h"
#include "petscsnes.h"

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
  Viewer  u_lambda_viewer;
  Viewer  fu_lambda_viewer;
} UserCtx;

extern int FormFunction(SNES,Vec,Vec,void*);
extern int Monitor(SNES,int,PetscReal,void*);


#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int     ierr,N = 5,i;
  UserCtx user;
  DA      da;
  DMMG    *dmmg;
  VecPack packer;

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = OptionsGetInt(PETSC_NULL,"-N",&N,PETSC_NULL);CHKERRQ(ierr);

  /* Hardwire several options; can be changed at command line */
  ierr = OptionsSetValue("-dmmg_grid_sequence",PETSC_NULL);CHKERRQ(ierr);
  ierr = OptionsSetValue("-ksp_type","fgmres");CHKERRQ(ierr);
  ierr = OptionsSetValue("-ksp_max_it","5");CHKERRQ(ierr);
  ierr = OptionsSetValue("-pc_mg_type","full");CHKERRQ(ierr);
  ierr = OptionsSetValue("-mg_coarse_ksp_type","cr");CHKERRQ(ierr);
  ierr = OptionsSetValue("-mg_levels_ksp_type","cr");CHKERRQ(ierr);
  ierr = OptionsSetValue("-mg_coarse_ksp_max_it","6");CHKERRQ(ierr);
  ierr = OptionsSetValue("-mg_levels_ksp_max_it","6");CHKERRQ(ierr);

  /* Create a global vector that includes a single redundant array and two da arrays */
  ierr = VecPackCreate(PETSC_COMM_WORLD,&packer);CHKERRQ(ierr);
  ierr = VecPackAddArray(packer,1);CHKERRQ(ierr);
  ierr = DACreate1d(PETSC_COMM_WORLD,DA_NONPERIODIC,N,2,1,PETSC_NULL,&da);CHKERRQ(ierr);
  ierr = VecPackAddDA(packer,da);CHKERRQ(ierr);

  /* create graphics windows */
  ierr = ViewerDrawOpen(PETSC_COMM_WORLD,0,"u_lambda - state variables and Lagrange multipliers",-1,-1,-1,-1,&user.u_lambda_viewer);CHKERRQ(ierr);
  ierr = ViewerDrawOpen(PETSC_COMM_WORLD,0,"fu_lambda - derivate w.r.t. state variables and Lagrange multipliers",-1,-1,-1,-1,&user.fu_lambda_viewer);CHKERRQ(ierr);

  /* create nonlinear multi-level solver */
  ierr = DMMGCreate(PETSC_COMM_WORLD,2,&user,&dmmg);CHKERRQ(ierr);
  ierr = DMMGSetUseMatrixFree(dmmg);CHKERRQ(ierr);
  ierr = DMMGSetDM(dmmg,(DM)packer);CHKERRQ(ierr);
  ierr = DMMGSetSNES(dmmg,FormFunction,PETSC_NULL);CHKERRQ(ierr);
  for (i=0; i<DMMGGetLevels(dmmg); i++) {
    /*    ierr = SNESSetMonitor(dmmg[i]->snes,Monitor,dmmg[i],0);CHKERRQ(ierr); */ ;
  }
  ierr = DMMGSolve(dmmg);CHKERRQ(ierr);
  ierr = DMMGDestroy(dmmg);CHKERRQ(ierr);

  ierr = DADestroy(da);CHKERRQ(ierr);
  ierr = VecPackDestroy(packer);CHKERRQ(ierr);
  ierr = ViewerDestroy(user.u_lambda_viewer);CHKERRQ(ierr);
  ierr = ViewerDestroy(user.fu_lambda_viewer);CHKERRQ(ierr);

  PetscFinalize();
  return 0;
}
 
/*
      Evaluates FU = Gradiant(L(w,u,lambda))

     This local function acts on the ghosted version of U (accessed via VecPackGetLocalVectors() and
   VecPackScatter()) BUT the global, nonghosted version of FU (via VecPackAccess()).

*/
int FormFunction(SNES snes,Vec U,Vec FU,void* dummy)
{
  DMMG    dmmg = (DMMG)dummy;
  int     ierr,xs,xm,i,N,nredundant;
  Scalar  **u_lambda,*w,*fw,**fu_lambda,d,h;
  Vec     vu_lambda,vfu_lambda;
  DA      da;
  VecPack packer = (VecPack)dmmg->dm;

  PetscFunctionBegin;
  ierr = VecPackGetEntries(packer,&nredundant,&da);CHKERRQ(ierr);
  ierr = VecPackGetLocalVectors(packer,&w,&vu_lambda);CHKERRQ(ierr);
  ierr = VecPackScatter(packer,U,w,vu_lambda);CHKERRQ(ierr);
  ierr = VecPackAccess(packer,FU,&fw,&vfu_lambda);CHKERRQ(ierr);

  ierr = DAGetCorners(da,&xs,PETSC_NULL,PETSC_NULL,&xm,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = DAGetInfo(da,0,&N,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DAVecGetArray(da,vu_lambda,(void**)&u_lambda);CHKERRQ(ierr);
  ierr = DAVecGetArray(da,vfu_lambda,(void**)&fu_lambda);CHKERRQ(ierr);
  d    = N-1.0;
  h    = 1.0/d;

#define u(i)        u_lambda[i][0]
#define lambda(i)   u_lambda[i][1]
#define fu(i)       fu_lambda[i][0]
#define flambda(i)  fu_lambda[i][1]

  /* derivative of L() w.r.t. w */
  if (xs == 0) { /* only first processor computes this */
    fw[0] = -2.0*d*lambda(0);
  }

  /* derivative of L() w.r.t. u */
  for (i=xs; i<xs+xm; i++) {
    if      (i == 0)   fu(0)   = 2.*h*u(0)   + 2.*d*lambda(0)   + d*lambda(1);
    else if (i == 1)   fu(1)   = 2.*h*u(1)   - 2.*d*lambda(1)   + d*lambda(2);
    else if (i == N-1) fu(N-1) = 2.*h*u(N-1) + 2.*d*lambda(N-1) + d*lambda(N-2);
    else if (i == N-2) fu(N-2) = 2.*h*u(N-2) - 2.*d*lambda(N-2) + d*lambda(N-3);
    else               fu(i)   = (2.*h*u(i)   + d*(lambda(i+1) - 2.0*lambda(i) + lambda(i-1)));
  } 

  /* derivative of L() w.r.t. lambda */
  for (i=xs; i<xs+xm; i++) {
    if      (i == 0)   flambda(0)   = 2.0*d*(u(0) - w[0]);
    else if (i == N-1) flambda(N-1) = 2.0*d*u(N-1);
    else               flambda(i)   = (d*(u(i+1) - 2.0*u(i) + u(i-1)) - 2.0*h);
  } 

  ierr = DAVecRestoreArray(da,vu_lambda,(void**)&u_lambda);CHKERRQ(ierr);
  ierr = DAVecRestoreArray(da,vfu_lambda,(void**)&fu_lambda);CHKERRQ(ierr);
  ierr = VecPackRestoreLocalVectors(packer,&w,&vu_lambda);CHKERRQ(ierr);
  PLogFlops(13*N);
  PetscFunctionReturn(0);
}

/* 
    Computes the exact solution
*/
int u_solution(void *dummy,int n,Scalar *x,Scalar *u)
{
  int i;
  PetscFunctionBegin;
  for (i=0; i<n; i++) {
    u[2*i] = x[i]*x[i] - 1.25*x[i] + .25;
  }
  PetscFunctionReturn(0);
}

int ExactSolution(VecPack packer,Vec U) 
{
  PF      pf;
  Vec     x;
  Vec     u_global;
  Scalar  *w;
  DA      da;
  int     m,ierr;

  PetscFunctionBegin;
  ierr = VecPackGetEntries(packer,&m,&da);CHKERRQ(ierr);

  ierr = PFCreate(PETSC_COMM_WORLD,1,1,&pf);CHKERRQ(ierr);
  ierr = PFSetType(pf,PFQUICK,(void*)u_solution);CHKERRQ(ierr);
  ierr = DAGetCoordinates(da,&x);CHKERRQ(ierr);
  if (!x) {
    ierr = DASetUniformCoordinates(da,0.0,1.0,0.0,1.0,0.0,1.0);CHKERRQ(ierr);
    ierr = DAGetCoordinates(da,&x);CHKERRQ(ierr);
  }
  ierr = VecPackAccess(packer,U,&w,&u_global,0);CHKERRQ(ierr);
  w[0] = .25;
  ierr = PFApplyVec(pf,x,u_global);CHKERRQ(ierr);
  ierr = PFDestroy(pf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


int Monitor(SNES snes,int its,PetscReal rnorm,void *dummy)
{
  DMMG      dmmg = (DMMG)dummy;
  UserCtx   *user = (UserCtx*)dmmg->user;
  int       ierr,m,N;
  Scalar    mone = -1.0,*w,*dw;
  Vec       u_lambda,U,F,Uexact;
  VecPack   packer = (VecPack)dmmg->dm;
  PetscReal norm;
  DA        da;

  PetscFunctionBegin;
  ierr = SNESGetSolution(snes,&U);CHKERRQ(ierr);
  ierr = VecPackAccess(packer,U,&w,&u_lambda);CHKERRQ(ierr);
  ierr = VecView(u_lambda,user->u_lambda_viewer); 

  ierr = SNESGetFunction(snes,&F,0,0);CHKERRQ(ierr);
  ierr = VecPackAccess(packer,F,&w,&u_lambda);CHKERRQ(ierr);
  /* ierr = VecView(u_lambda,user->fu_lambda_viewer); */

  ierr = VecPackGetEntries(packer,&m,&da);CHKERRQ(ierr);
  ierr = DAGetInfo(da,0,&N,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = VecDuplicate(U,&Uexact);CHKERRQ(ierr);
  ierr = ExactSolution(packer,Uexact);CHKERRQ(ierr);
  ierr = VecAXPY(&mone,U,Uexact);CHKERRQ(ierr);
  ierr = VecPackAccess(packer,Uexact,&dw,&u_lambda);CHKERRQ(ierr);
  ierr = VecStrideNorm(u_lambda,0,NORM_2,&norm);CHKERRQ(ierr);
  norm = norm/sqrt(N-1.);
  ierr = PetscPrintf(dmmg->comm,"Norm of error %g Error at x = 0 %g\n",norm,dw[0]);CHKERRQ(ierr);
  ierr = VecView(u_lambda,user->fu_lambda_viewer);
  ierr = VecDestroy(Uexact);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}








