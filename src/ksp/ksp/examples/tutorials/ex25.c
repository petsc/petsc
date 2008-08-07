
/*
 Partial differential equation

   d  (1 + e*sine(2*pi*k*x)) d u = 1, 0 < x < 1,
   --                        ---
   dx                        dx
with boundary conditions

   u = 0 for x = 0, x = 1

   This uses multigrid to solve the linear system

*/

static char help[] = "Solves 1D variable coefficient Laplacian using multigrid.\n\n";

#include "petscda.h"
#include "petscksp.h"
#include "petscdmmg.h"

extern PetscErrorCode ComputeMatrix(DMMG,Mat,Mat);
extern PetscErrorCode ComputeRHS(DMMG,Vec);

typedef struct {
  PetscInt    k;
  PetscScalar e;
} AppCtx;

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  DMMG           *dmmg;
  PetscReal      norm;
  DA             da;
  AppCtx         user;

  PetscInitialize(&argc,&argv,(char *)0,help);

  user.k = 1;
  user.e = .99;
  ierr = PetscOptionsGetInt(0,"-k",&user.k,0);CHKERRQ(ierr);
  ierr = PetscOptionsGetScalar(0,"-e",&user.e,0);CHKERRQ(ierr);

  ierr = DMMGCreate(PETSC_COMM_WORLD,3,&user,&dmmg);CHKERRQ(ierr);
  ierr = DACreate1d(PETSC_COMM_WORLD,DA_NONPERIODIC,-3,1,1,0,&da);CHKERRQ(ierr);  
  ierr = DMMGSetDM(dmmg,(DM)da);CHKERRQ(ierr);
  ierr = DADestroy(da);CHKERRQ(ierr);

  ierr = DMMGSetKSP(dmmg,ComputeRHS,ComputeMatrix);CHKERRQ(ierr);

  ierr = DMMGSolve(dmmg);CHKERRQ(ierr);

  ierr = MatMult(DMMGGetJ(dmmg),DMMGGetx(dmmg),DMMGGetr(dmmg));CHKERRQ(ierr);
  ierr = VecAXPY(DMMGGetr(dmmg),-1.0,DMMGGetRHS(dmmg));CHKERRQ(ierr);
  ierr = VecNorm(DMMGGetr(dmmg),NORM_2,&norm);CHKERRQ(ierr);
  /* ierr = PetscPrintf(PETSC_COMM_WORLD,"Residual norm %G\n",norm);CHKERRQ(ierr); */

  ierr = DMMGDestroy(dmmg);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);

  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "ComputeRHS"
PetscErrorCode ComputeRHS(DMMG dmmg,Vec b)
{
  PetscErrorCode ierr;
  PetscInt       mx,idx[2];
  PetscScalar    h,v[2];

  PetscFunctionBegin;
  ierr   = DAGetInfo((DA)dmmg->dm,0,&mx,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  h      = 1.0/((mx-1));
  ierr   = VecSet(b,h);CHKERRQ(ierr);
  idx[0] = 0; idx[1] = mx -1;
  v[0]   = v[1] = 0.0;
  ierr   = VecSetValues(b,2,idx,v,INSERT_VALUES);CHKERRQ(ierr);
  ierr   = VecAssemblyBegin(b);CHKERRQ(ierr);
  ierr   = VecAssemblyEnd(b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
    
#undef __FUNCT__
#define __FUNCT__ "ComputeMatrix"
PetscErrorCode ComputeMatrix(DMMG dmmg,Mat J,Mat jac)
{
  DA             da = (DA)dmmg->dm;
  PetscErrorCode ierr;
  PetscInt       i,mx,xm,xs;
  PetscScalar    v[3],h,xlow,xhigh;
  MatStencil     row,col[3];
  AppCtx         *user = (AppCtx*)dmmg->user;

  ierr = DAGetInfo(da,0,&mx,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);  
  ierr = DAGetCorners(da,&xs,0,0,&xm,0,0);CHKERRQ(ierr);
  h    = 1.0/(mx-1);

  for(i=xs; i<xs+xm; i++){
    row.i = i;
    if (i==0 || i==mx-1){
      v[0] = 2.0;
      ierr = MatSetValuesStencil(jac,1,&row,1,&row,v,INSERT_VALUES);CHKERRQ(ierr);
    } else {
       xlow  = h*(PetscReal)i - .5*h;
       xhigh = xlow + h;
       v[0] = (-1.0 - user->e*PetscSinScalar(2.0*PETSC_PI*user->k*xlow))/h;col[0].i = i-1;
       v[1] = (2.0 + user->e*PetscSinScalar(2.0*PETSC_PI*user->k*xlow) + user->e*PetscSinScalar(2.0*PETSC_PI*user->k*xhigh))/h;col[1].i = row.i;
       v[2] = (-1.0 - user->e*PetscSinScalar(2.0*PETSC_PI*user->k*xhigh))/h;col[2].i = i+1;
      ierr = MatSetValuesStencil(jac,1,&row,3,col,v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  return 0;
}


