

static char help[] = "Solves 1D wave equation using multigrid.\n\n";

#include <petscdmda.h>
#include <petscksp.h>


extern PetscErrorCode ComputeMatrix(KSP,Mat,Mat,MatStructure*,void*);
extern PetscErrorCode ComputeRHS(KSP,Vec,void*);
extern PetscErrorCode ComputeInitialSolution(DM,Vec);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       i;
  KSP            ksp;
  DM             da;
  Vec            x;

  PetscInitialize(&argc,&argv,(char *)0,help);

  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = DMDACreate1d(PETSC_COMM_WORLD,DMDA_BOUNDARY_PERIODIC,-3,2,1,0,&da);CHKERRQ(ierr);
  ierr = KSPSetDM(ksp,da);CHKERRQ(ierr);
  ierr = KSPSetComputeRHS(ksp,ComputeRHS,PETSC_NULL);CHKERRQ(ierr);
  ierr = KSPSetComputeOperators(ksp,ComputeMatrix,PETSC_NULL);CHKERRQ(ierr);

  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&x);CHKERRQ(ierr);
  ierr = ComputeInitialSolution(da,x);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(da,x);CHKERRQ(ierr);
  ierr = KSPSetUp(ksp);CHKERRQ(ierr);
  ierr = VecView(x,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
  for (i=0; i<10; i++) {
    ierr = KSPSolve(ksp,PETSC_NULL,x);CHKERRQ(ierr);
    ierr = VecView(x,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "ComputeInitialSolution"
PetscErrorCode ComputeInitialSolution(DM da,Vec x)
{
  PetscErrorCode ierr;
  PetscInt       mx,col[2],xs,xm,i;
  PetscScalar    Hx,val[2];

  PetscFunctionBegin;
  ierr = DMDAGetInfo(da,0,&mx,0,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  Hx = 2.0*PETSC_PI / (PetscReal)(mx);
  ierr = DMDAGetCorners(da,&xs,0,0,&xm,0,0);CHKERRQ(ierr);

  for (i=xs; i<xs+xm; i++){
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
PetscErrorCode ComputeRHS(KSP ksp,Vec b,void *ctx)
{
  PetscErrorCode ierr;
  PetscInt       mx;
  PetscScalar    h;
  Vec            x;
  DM             da;

  PetscFunctionBegin;
  ierr = KSPGetDM(ksp,&da);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,0,&mx,0,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DMGetApplicationContext(da,&x);CHKERRQ(ierr);
  h    = 2.0*PETSC_PI/((mx));
  ierr = VecCopy(x,b);CHKERRQ(ierr);
  ierr = VecScale(b,h);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ComputeMatrix"
PetscErrorCode ComputeMatrix(KSP ksp,Mat J,Mat jac,MatStructure *str,void *ctx)
{
  PetscErrorCode ierr;
  PetscInt       i,mx,xm,xs;
  PetscScalar    v[7],Hx;
  MatStencil     row,col[7];
  PetscScalar    lambda;
  DM             da;

  PetscFunctionBegin;
  ierr = KSPGetDM(ksp,&da);CHKERRQ(ierr);
  ierr = PetscMemzero(col,7*sizeof(MatStencil));CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,0,&mx,0,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  Hx = 2.0*PETSC_PI / (PetscReal)(mx);
  ierr = DMDAGetCorners(da,&xs,0,0,&xm,0,0);CHKERRQ(ierr);
  lambda = 2.0*Hx;
  for (i=xs; i<xs+xm; i++){
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
  ierr = MatView(jac,PETSC_VIEWER_BINARY_(PETSC_COMM_SELF));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
