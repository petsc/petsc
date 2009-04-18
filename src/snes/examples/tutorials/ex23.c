
static char help[] = "Solves PDE problem from ex22.c\n\n";

#include "petscda.h"
#include "petscpf.h"
#include "petscsnes.h"
#include "petscdmmg.h"

/*

       In this example the PDE is 
                             Uxx + U^2 = 2, 
                            u(0) = .25
                            u(1) = 0

       The exact solution for u is given by u(x) = x*x - 1.25*x + .25

       Use the usual centered finite differences.


       See ex22.c for a design optimization problem

*/

typedef struct {
  PetscViewer  u_viewer;
  PetscViewer  fu_viewer;
} UserCtx;

extern PetscErrorCode FormFunction(SNES,Vec,Vec,void*);
extern PetscErrorCode FormFunctionLocali(DALocalInfo*,MatStencil*,PetscScalar*,PetscScalar*,void*);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  UserCtx        user;
  DA             da;
  DMMG           *dmmg;

  PetscInitialize(&argc,&argv,PETSC_NULL,help);

  /* Hardwire several options; can be changed at command line */
  ierr = PetscOptionsSetValue("-dmmg_grid_sequence",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsSetValue("-ksp_type","fgmres");CHKERRQ(ierr);
  ierr = PetscOptionsSetValue("-ksp_max_it","5");CHKERRQ(ierr);
  ierr = PetscOptionsSetValue("-pc_mg_type","full");CHKERRQ(ierr);
  ierr = PetscOptionsSetValue("-mg_coarse_ksp_type","gmres");CHKERRQ(ierr);
  ierr = PetscOptionsSetValue("-mg_levels_ksp_type","gmres");CHKERRQ(ierr);
  ierr = PetscOptionsSetValue("-mg_coarse_ksp_max_it","3");CHKERRQ(ierr);
  ierr = PetscOptionsSetValue("-mg_levels_ksp_max_it","3");CHKERRQ(ierr);
  /* ierr = PetscOptionsSetValue("-snes_mf_type","wp");CHKERRQ(ierr); */
  /* ierr = PetscOptionsSetValue("-snes_mf_compute_normu","no");CHKERRQ(ierr); */
  ierr = PetscOptionsSetValue("-snes_ls","basic");CHKERRQ(ierr);
  /*  ierr = PetscOptionsSetValue("-dmmg_jacobian_mf_fd",0);CHKERRQ(ierr); */
  /*  ierr = PetscOptionsSetValue("-snes_ls","basicnonorms");CHKERRQ(ierr); */
  ierr = PetscOptionsInsert(&argc,&argv,PETSC_NULL);CHKERRQ(ierr); 
  
  /* Create a global vector from a da arrays */
  ierr = DACreate1d(PETSC_COMM_WORLD,DA_NONPERIODIC,-5,1,1,PETSC_NULL,&da);CHKERRQ(ierr);

  /* create graphics windows */
  ierr = PetscViewerDrawOpen(PETSC_COMM_WORLD,0,"u - state variables",-1,-1,-1,-1,&user.u_viewer);CHKERRQ(ierr);
  ierr = PetscViewerDrawOpen(PETSC_COMM_WORLD,0,"fu - discretization of function",-1,-1,-1,-1,&user.fu_viewer);CHKERRQ(ierr);

  /* create nonlinear multi-level solver */
  ierr = DMMGCreate(PETSC_COMM_WORLD,2,&user,&dmmg);CHKERRQ(ierr);
  ierr = DMMGSetDM(dmmg,(DM)da);CHKERRQ(ierr);
  ierr = DMMGSetSNES(dmmg,FormFunction,PETSC_NULL);CHKERRQ(ierr);
  ierr = DMMGSetFromOptions(dmmg);CHKERRQ(ierr);
  ierr = DMMGSetSNESLocali(dmmg,FormFunctionLocali,0,0);CHKERRQ(ierr);
  ierr = DMMGSolve(dmmg);CHKERRQ(ierr);
  ierr = DMMGDestroy(dmmg);CHKERRQ(ierr);

  ierr = DADestroy(da);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(user.u_viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(user.fu_viewer);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
 
/*
     This local function acts on the ghosted version of U (accessed via DAGetLocalVector())
     BUT the global, nonghosted version of FU

*/
PetscErrorCode FormFunction(SNES snes,Vec U,Vec FU,void* dummy)
{
  DMMG           dmmg = (DMMG)dummy;
  PetscErrorCode ierr;
  PetscInt       xs,xm,i,N;
  PetscScalar    *u,*fu,d,h;
  Vec            vu;
  DA             da = (DA) dmmg->dm;

  PetscFunctionBegin;
  ierr = DAGetLocalVector(da,&vu);CHKERRQ(ierr);
  ierr = DAGlobalToLocalBegin(da,U,INSERT_VALUES,vu);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(da,U,INSERT_VALUES,vu);CHKERRQ(ierr);

  ierr = DAGetCorners(da,&xs,PETSC_NULL,PETSC_NULL,&xm,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = DAGetInfo(da,0,&N,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DAVecGetArray(da,vu,&u);CHKERRQ(ierr);
  ierr = DAVecGetArray(da,FU,&fu);CHKERRQ(ierr);
  d    = N-1.0;
  h    = 1.0/d;

  for (i=xs; i<xs+xm; i++) {
    if      (i == 0)   fu[0]   = 2.0*d*(u[0] - .25) + h*u[0]*u[0];
    else if (i == N-1) fu[N-1] = 2.0*d*u[N-1] + h*u[N-1]*u[N-1];
    else               fu[i]   = -(d*(u[i+1] - 2.0*u[i] + u[i-1]) - 2.0*h) + h*u[i]*u[i];
  } 

  ierr = DAVecRestoreArray(da,vu,&u);CHKERRQ(ierr);
  ierr = DAVecRestoreArray(da,FU,&fu);CHKERRQ(ierr);
  ierr = DARestoreLocalVector(da,&vu);CHKERRQ(ierr);
  ierr = PetscLogFlops(9.0*N);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FormFunctionLocali(DALocalInfo *info,MatStencil *pt,PetscScalar *u,PetscScalar *fu,void* dummy)
{
  PetscInt     i,N = info->mx;
  PetscScalar  d,h;

  PetscFunctionBegin;
  d    = N-1.0;
  h    = 1.0/d;

  i = pt->i;
  if      (i == 0)   *fu = 2.0*d*(u[0] - .25) + h*u[0]*u[0];
  else if (i == N-1) *fu = 2.0*d*u[N-1] + h*u[N-1]*u[N-1];
  else               *fu = -(d*(u[i+1] - 2.0*u[i] + u[i-1]) - 2.0*h) + h*u[i]*u[i];

  PetscFunctionReturn(0);
}
