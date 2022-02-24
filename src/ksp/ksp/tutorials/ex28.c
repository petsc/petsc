
static char help[] = "Solves 1D wave equation using multigrid.\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscksp.h>

extern PetscErrorCode ComputeMatrix(KSP,Mat,Mat,void*);
extern PetscErrorCode ComputeRHS(KSP,Vec,void*);
extern PetscErrorCode ComputeInitialSolution(DM,Vec);

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       i;
  KSP            ksp;
  DM             da;
  Vec            x;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));
  CHKERRQ(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_PERIODIC,3,2,1,0,&da));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));
  CHKERRQ(KSPSetDM(ksp,da));
  CHKERRQ(KSPSetComputeRHS(ksp,ComputeRHS,NULL));
  CHKERRQ(KSPSetComputeOperators(ksp,ComputeMatrix,NULL));

  CHKERRQ(KSPSetFromOptions(ksp));
  CHKERRQ(DMCreateGlobalVector(da,&x));
  CHKERRQ(ComputeInitialSolution(da,x));
  CHKERRQ(DMSetApplicationContext(da,x));
  CHKERRQ(KSPSetUp(ksp));
  CHKERRQ(VecView(x,PETSC_VIEWER_DRAW_WORLD));
  for (i=0; i<10; i++) {
    CHKERRQ(KSPSolve(ksp,NULL,x));
    CHKERRQ(VecView(x,PETSC_VIEWER_DRAW_WORLD));
  }
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(KSPDestroy(&ksp));
  CHKERRQ(DMDestroy(&da));
  ierr = PetscFinalize();
  return ierr;
}

PetscErrorCode ComputeInitialSolution(DM da,Vec x)
{
  PetscInt       mx,col[2],xs,xm,i;
  PetscScalar    Hx,val[2];

  PetscFunctionBeginUser;
  CHKERRQ(DMDAGetInfo(da,0,&mx,0,0,0,0,0,0,0,0,0,0,0));
  Hx   = 2.0*PETSC_PI / (PetscReal)(mx);
  CHKERRQ(DMDAGetCorners(da,&xs,0,0,&xm,0,0));

  for (i=xs; i<xs+xm; i++) {
    col[0] = 2*i; col[1] = 2*i + 1;
    val[0] = val[1] = PetscSinScalar(((PetscScalar)i)*Hx);
    CHKERRQ(VecSetValues(x,2,col,val,INSERT_VALUES));
  }
  CHKERRQ(VecAssemblyBegin(x));
  CHKERRQ(VecAssemblyEnd(x));
  PetscFunctionReturn(0);
}

PetscErrorCode ComputeRHS(KSP ksp,Vec b,void *ctx)
{
  PetscInt       mx;
  PetscScalar    h;
  Vec            x;
  DM             da;

  PetscFunctionBeginUser;
  CHKERRQ(KSPGetDM(ksp,&da));
  CHKERRQ(DMDAGetInfo(da,0,&mx,0,0,0,0,0,0,0,0,0,0,0));
  CHKERRQ(DMGetApplicationContext(da,&x));
  h    = 2.0*PETSC_PI/((mx));
  CHKERRQ(VecCopy(x,b));
  CHKERRQ(VecScale(b,h));
  PetscFunctionReturn(0);
}

PetscErrorCode ComputeMatrix(KSP ksp,Mat J,Mat jac,void *ctx)
{
  PetscInt       i,mx,xm,xs;
  PetscScalar    v[7],Hx;
  MatStencil     row,col[7];
  PetscScalar    lambda;
  DM             da;

  PetscFunctionBeginUser;
  CHKERRQ(KSPGetDM(ksp,&da));
  CHKERRQ(PetscArrayzero(col,7));
  CHKERRQ(DMDAGetInfo(da,0,&mx,0,0,0,0,0,0,0,0,0,0,0));
  Hx     = 2.0*PETSC_PI / (PetscReal)(mx);
  CHKERRQ(DMDAGetCorners(da,&xs,0,0,&xm,0,0));
  lambda = 2.0*Hx;
  for (i=xs; i<xs+xm; i++) {
    row.i = i; row.j = 0; row.k = 0; row.c = 0;
    v[0]  = Hx;     col[0].i = i;   col[0].c = 0;
    v[1]  = lambda; col[1].i = i-1;   col[1].c = 1;
    v[2]  = -lambda;col[2].i = i+1; col[2].c = 1;
    CHKERRQ(MatSetValuesStencil(jac,1,&row,3,col,v,INSERT_VALUES));

    row.i = i; row.j = 0; row.k = 0; row.c = 1;
    v[0]  = lambda; col[0].i = i-1;   col[0].c = 0;
    v[1]  = Hx;     col[1].i = i;   col[1].c = 1;
    v[2]  = -lambda;col[2].i = i+1; col[2].c = 0;
    CHKERRQ(MatSetValuesStencil(jac,1,&row,3,col,v,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatView(jac,PETSC_VIEWER_BINARY_(PETSC_COMM_SELF)));
  PetscFunctionReturn(0);
}

/*TEST

   test:
      args: -ksp_monitor_short -pc_type mg -pc_mg_type full -ksp_type fgmres -da_refine 2 -mg_levels_ksp_type gmres -mg_levels_ksp_max_it 1 -mg_levels_pc_type ilu

TEST*/
