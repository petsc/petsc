/*$Id: ex22.c,v 1.1 2000/12/04 21:36:36 bsmith Exp bsmith $*/

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
  DA      da;
  Vec     u_lambda,fu_lambda;
  Scalar  *w,*fw;
  int     nredundant;
  VecPack packer;
  Viewer  u_lambda_viewer;
  Viewer  fu_lambda_viewer;
} UserCtx;

extern int FormFunction(SNES,Vec,Vec,void*);
extern int Monitor(SNES,int,PetscReal,void*);


#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int     ierr,N = 5,its;
  Vec     U,FU;
  SNES    snes;
  UserCtx user;

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = OptionsGetInt(PETSC_NULL,"-N",&N,PETSC_NULL);CHKERRQ(ierr);

  /* Create a global vector that includes a single redundant array and two da arrays */
  ierr = VecPackCreate(PETSC_COMM_WORLD,&user.packer);CHKERRQ(ierr);
  user.nredundant = 1;
  ierr = VecPackAddArray(user.packer,user.nredundant);CHKERRQ(ierr);
  ierr = DACreate1d(PETSC_COMM_WORLD,DA_NONPERIODIC,N,2,1,PETSC_NULL,&user.da);CHKERRQ(ierr);
  ierr = VecPackAddDA(user.packer,user.da);CHKERRQ(ierr);
  ierr = VecPackCreateGlobalVector(user.packer,&U);CHKERRQ(ierr);
  ierr = VecDuplicate(U,&FU);CHKERRQ(ierr);

  /* create local work vectors (that include ghost points where appropriate */
  user.w = (Scalar*)PetscMalloc(user.nredundant*sizeof(Scalar));CHKPTRQ(user.w);
  ierr = DACreateLocalVector(user.da,&user.u_lambda);CHKERRQ(ierr);
  user.fw = (Scalar*)PetscMalloc(user.nredundant*sizeof(Scalar));CHKPTRQ(user.fw);
  ierr = DACreateLocalVector(user.da,&user.fu_lambda);CHKERRQ(ierr);

  /* create graphics windows */
  ierr = ViewerDrawOpen(PETSC_COMM_WORLD,0,"u_lambda - state variables and Lagrange multipliers",-1,-1,-1,-1,&user.u_lambda_viewer);CHKERRQ(ierr);
  ierr = ViewerDrawOpen(PETSC_COMM_WORLD,0,"fu_lambda - derivate w.r.t. state variables and Lagrange multipliers",-1,-1,-1,-1,&user.fu_lambda_viewer);CHKERRQ(ierr);


  /* create nonlinear solver */
  ierr = SNESCreate(PETSC_COMM_WORLD,SNES_NONLINEAR_EQUATIONS,&snes);CHKERRQ(ierr);
  ierr = SNESSetFunction(snes,FU,FormFunction,&user);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  ierr = SNESSetMonitor(snes,Monitor,&user,0);CHKERRQ(ierr);
  ierr = SNESSolve(snes,U,&its);CHKERRQ(ierr);
  ierr = SNESDestroy(snes);CHKERRQ(ierr);

  ierr = DADestroy(user.da);CHKERRQ(ierr);
  ierr = VecDestroy(user.u_lambda);CHKERRQ(ierr);
  ierr = VecDestroy(user.fu_lambda);CHKERRQ(ierr);
  ierr = PetscFree(user.w);CHKERRQ(ierr);
  ierr = PetscFree(user.fw);CHKERRQ(ierr);
  ierr = VecPackDestroy(user.packer);CHKERRQ(ierr);
  ierr = VecDestroy(U);CHKERRQ(ierr);
  ierr = VecDestroy(FU);CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}
 
/*
      Evaluates FU = Gradiant( L(w,u,lambda) )

*/
int FormFunction(SNES snes,Vec U,Vec FU,void* dummy)
{
  UserCtx *user = (UserCtx*)dummy;
  int     ierr,xs,xm,i,N;
  Scalar  **u_lambda,*w = user->w,*fw = user->fw,**fu_lambda,d;

  PetscFunctionBegin;
  ierr = VecPackScatter(user->packer,U,user->w,user->u_lambda);CHKERRQ(ierr);

  ierr = DAGetCorners(user->da,&xs,PETSC_NULL,PETSC_NULL,&xm,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = DAGetInfo(user->da,0,&N,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DAVecGetArray(user->da,user->u_lambda,(void**)&u_lambda);CHKERRQ(ierr);
  ierr = DAVecGetArray(user->da,user->fu_lambda,(void**)&fu_lambda);CHKERRQ(ierr);
  d    = (N-1.0)*(N-1.0);

#define u(i)        u_lambda[i][0]
#define lambda(i)   u_lambda[i][1]
#define fu(i)       fu_lambda[i][0]
#define flambda(i)  fu_lambda[i][1]

  /* derivative of L() w.r.t. w */
  if (xs == 0) { /* only first processor computes this */
    fw[0] = -d*lambda(0);
  }

  /* derivative of L() w.r.t. u */
  for (i=xs; i<xs+xm; i++) {
    if      (i == 0)   fu(0)   = 2.*u(0)   + d*lambda(0)   + d*lambda(1);
    else if (i == N-1) fu(N-1) = 2.*u(N-1) + d*lambda(N-1) + d*lambda(N-2);
    else               fu(i)   = 2.*u(i)   + d*(lambda(i+1) - 2.0*lambda(i) + lambda(i-1));
  } 

  /* derivative of L() w.r.t. lambda */
  for (i=xs; i<xs+xm; i++) {
    if      (i == 0)   flambda(0)   = d*u(0) - d*w[0];
    else if (i == N-1) flambda(N-1) = d*u(N-1);
    else               flambda(i)   = d*(u(i+1) - 2.0*u(i) + u(i-1)) - 2.0;
  } 

  ierr = DAVecRestoreArray(user->da,user->u_lambda,(void**)&u_lambda);CHKERRQ(ierr);
  ierr = DAVecRestoreArray(user->da,user->fu_lambda,(void**)&fu_lambda);CHKERRQ(ierr);
  ierr = VecPackGather(user->packer,FU,user->fw,user->fu_lambda);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

int Monitor(SNES snes,int its,PetscReal rnorm,void *dummy)
{
  UserCtx *user = (UserCtx*)dummy;
  int     ierr;
  Scalar  *w;
  Vec     u_lambda,U,F;

  PetscFunctionBegin;
  ierr = SNESGetSolution(snes,&U);CHKERRQ(ierr);
  ierr = VecPackAccess(user->packer,U,&w,&u_lambda);CHKERRQ(ierr);
  ierr = VecView(u_lambda,user->u_lambda_viewer);

  ierr = SNESGetFunction(snes,&F,0,0);CHKERRQ(ierr);
  ierr = VecPackAccess(user->packer,F,&w,&u_lambda);CHKERRQ(ierr);
  ierr = VecView(u_lambda,user->fu_lambda_viewer);
  PetscFunctionReturn(0);
}



