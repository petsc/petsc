

static char help[] = "Solves 1D wave equation using multigrid.\n\n";

#include "petscda.h"
#include "petscksp.h"
#include "petscdmmg.h"

extern PetscErrorCode ComputeMatrix(DMMG,Mat,Mat);
extern PetscErrorCode ComputeRHS(DMMG,Vec);
extern PetscErrorCode ComputeInitialSolution(DMMG*);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       i;
  DMMG           *dmmg;
  PetscReal      norm;
  DA             da;

  PetscInitialize(&argc,&argv,(char *)0,help);

  ierr = DMMGCreate(PETSC_COMM_WORLD,3,PETSC_NULL,&dmmg);CHKERRQ(ierr);
  ierr = DACreate1d(PETSC_COMM_WORLD,DA_XPERIODIC,-3,2,1,0,&da);CHKERRQ(ierr);  
  ierr = DMMGSetDM(dmmg,(DM)da);CHKERRQ(ierr);
  ierr = DADestroy(da);CHKERRQ(ierr);

  ierr = DMMGSetKSP(dmmg,ComputeRHS,ComputeMatrix);CHKERRQ(ierr);

  ierr = ComputeInitialSolution(dmmg);CHKERRQ(ierr);

  VecView(DMMGGetx(dmmg),PETSC_VIEWER_DRAW_WORLD);
  for (i=0; i<1000; i++) {
    ierr = DMMGSolve(dmmg);CHKERRQ(ierr);
    ierr = VecView(DMMGGetx(dmmg),PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
  }
  ierr = MatMult(DMMGGetJ(dmmg),DMMGGetx(dmmg),DMMGGetr(dmmg));CHKERRQ(ierr);
  ierr = VecAXPY(DMMGGetr(dmmg),-1.0,DMMGGetRHS(dmmg));CHKERRQ(ierr);
  ierr = VecNorm(DMMGGetr(dmmg),NORM_2,&norm);CHKERRQ(ierr);
  /* ierr = PetscPrintf(PETSC_COMM_WORLD,"Residual norm %G\n",norm);CHKERRQ(ierr); */

  ierr = DMMGDestroy(dmmg);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);

  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "ComputeInitialSolution"
PetscErrorCode ComputeInitialSolution(DMMG *dmmg)
{
  PetscErrorCode ierr;
  PetscInt       mx,col[2],xs,xm,i;
  PetscScalar    Hx,val[2];
  Vec            x = DMMGGetx(dmmg);

  PetscFunctionBegin;
  ierr = DAGetInfo(DMMGGetDA(dmmg),0,&mx,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  Hx = 2.0*PETSC_PI / (PetscReal)(mx);
  ierr = DAGetCorners(DMMGGetDA(dmmg),&xs,0,0,&xm,0,0);CHKERRQ(ierr);
  
  for(i=xs; i<xs+xm; i++){
    col[0] = 2*i; col[1] = 2*i + 1;
    val[0] = val[1] = PetscSinScalar(((PetscScalar)i)*Hx);
    ierr = VecSetValues(x,2,col,val,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
    
#undef __FUNCT__
#define __FUNCT__ "ComputeRHS"
PetscErrorCode ComputeRHS(DMMG dmmg,Vec b)
{
  PetscErrorCode ierr;
  PetscInt       mx;
  PetscScalar    h;

  PetscFunctionBegin;
  ierr = DAGetInfo((DA)dmmg->dm,0,&mx,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  h    = 2.0*PETSC_PI/((mx));
  ierr = VecCopy(dmmg->x,b);CHKERRQ(ierr);
  ierr = VecScale(b,h);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ComputeMatrix"
PetscErrorCode ComputeMatrix(DMMG dmmg,Mat J,Mat jac)
{
  DA             da = (DA)dmmg->dm;
  PetscErrorCode ierr;
  PetscInt       i,mx,xm,xs;
  PetscScalar    v[7],Hx;
  MatStencil     row,col[7];
  PetscScalar    lambda;

  ierr = PetscMemzero(col,7*sizeof(MatStencil));CHKERRQ(ierr);
  ierr = DAGetInfo(da,0,&mx,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);  
  Hx = 2.0*PETSC_PI / (PetscReal)(mx);
  ierr = DAGetCorners(da,&xs,0,0,&xm,0,0);CHKERRQ(ierr);
  lambda = 2.0*Hx;
  for(i=xs; i<xs+xm; i++){
    row.i = i; row.j = 0; row.k = 0; row.c = 0;
    v[0] = Hx;     col[0].i = i;   col[0].c = 0;
    v[1] = lambda; col[1].i = i-1;   col[1].c = 1;
    v[2] = -lambda;col[2].i = i+1; col[2].c = 1;
    ierr = MatSetValuesStencil(jac,1,&row,3,col,v,INSERT_VALUES);CHKERRQ(ierr);

    row.i = i; row.j = 0; row.k = 0; row.c = 1;
    v[0] = lambda; col[0].i = i-1;   col[0].c = 0;
    v[1] = Hx;     col[1].i = i;   col[1].c = 1;
    v[2] = -lambda;col[2].i = i+1; col[2].c = 0;
    ierr = MatSetValuesStencil(jac,1,&row,3,col,v,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  MatView(jac,PETSC_VIEWER_BINARY_(PETSC_COMM_SELF));
  return 0;
}


