
static char help[] = "Solves 1D wave equation using multigrid.\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscksp.h>

extern PetscErrorCode ComputeMatrix(KSP,Mat,Mat,void*);
extern PetscErrorCode ComputeRHS(KSP,Vec,void*);
extern PetscErrorCode ComputeInitialSolution(DM,Vec);

int main(int argc,char **argv)
{
  PetscInt       i;
  KSP            ksp;
  DM             da;
  Vec            x;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
  PetscCall(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_PERIODIC,3,2,1,0,&da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(KSPSetDM(ksp,da));
  PetscCall(KSPSetComputeRHS(ksp,ComputeRHS,NULL));
  PetscCall(KSPSetComputeOperators(ksp,ComputeMatrix,NULL));

  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(DMCreateGlobalVector(da,&x));
  PetscCall(ComputeInitialSolution(da,x));
  PetscCall(DMSetApplicationContext(da,x));
  PetscCall(KSPSetUp(ksp));
  PetscCall(VecView(x,PETSC_VIEWER_DRAW_WORLD));
  for (i=0; i<10; i++) {
    PetscCall(KSPSolve(ksp,NULL,x));
    PetscCall(VecView(x,PETSC_VIEWER_DRAW_WORLD));
  }
  PetscCall(VecDestroy(&x));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());
  return 0;
}

PetscErrorCode ComputeInitialSolution(DM da,Vec x)
{
  PetscInt       mx,col[2],xs,xm,i;
  PetscScalar    Hx,val[2];

  PetscFunctionBeginUser;
  PetscCall(DMDAGetInfo(da,0,&mx,0,0,0,0,0,0,0,0,0,0,0));
  Hx   = 2.0*PETSC_PI / (PetscReal)(mx);
  PetscCall(DMDAGetCorners(da,&xs,0,0,&xm,0,0));

  for (i=xs; i<xs+xm; i++) {
    col[0] = 2*i; col[1] = 2*i + 1;
    val[0] = val[1] = PetscSinScalar(((PetscScalar)i)*Hx);
    PetscCall(VecSetValues(x,2,col,val,INSERT_VALUES));
  }
  PetscCall(VecAssemblyBegin(x));
  PetscCall(VecAssemblyEnd(x));
  PetscFunctionReturn(0);
}

PetscErrorCode ComputeRHS(KSP ksp,Vec b,void *ctx)
{
  PetscInt       mx;
  PetscScalar    h;
  Vec            x;
  DM             da;

  PetscFunctionBeginUser;
  PetscCall(KSPGetDM(ksp,&da));
  PetscCall(DMDAGetInfo(da,0,&mx,0,0,0,0,0,0,0,0,0,0,0));
  PetscCall(DMGetApplicationContext(da,&x));
  h    = 2.0*PETSC_PI/((mx));
  PetscCall(VecCopy(x,b));
  PetscCall(VecScale(b,h));
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
  PetscCall(KSPGetDM(ksp,&da));
  PetscCall(PetscArrayzero(col,7));
  PetscCall(DMDAGetInfo(da,0,&mx,0,0,0,0,0,0,0,0,0,0,0));
  Hx     = 2.0*PETSC_PI / (PetscReal)(mx);
  PetscCall(DMDAGetCorners(da,&xs,0,0,&xm,0,0));
  lambda = 2.0*Hx;
  for (i=xs; i<xs+xm; i++) {
    row.i = i; row.j = 0; row.k = 0; row.c = 0;
    v[0]  = Hx;     col[0].i = i;   col[0].c = 0;
    v[1]  = lambda; col[1].i = i-1;   col[1].c = 1;
    v[2]  = -lambda;col[2].i = i+1; col[2].c = 1;
    PetscCall(MatSetValuesStencil(jac,1,&row,3,col,v,INSERT_VALUES));

    row.i = i; row.j = 0; row.k = 0; row.c = 1;
    v[0]  = lambda; col[0].i = i-1;   col[0].c = 0;
    v[1]  = Hx;     col[1].i = i;   col[1].c = 1;
    v[2]  = -lambda;col[2].i = i+1; col[2].c = 0;
    PetscCall(MatSetValuesStencil(jac,1,&row,3,col,v,INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY));
  PetscCall(MatView(jac,PETSC_VIEWER_BINARY_(PETSC_COMM_SELF)));
  PetscFunctionReturn(0);
}

/*TEST

   test:
      args: -ksp_monitor_short -pc_type mg -pc_mg_type full -ksp_type fgmres -da_refine 2 -mg_levels_ksp_type gmres -mg_levels_ksp_max_it 1 -mg_levels_pc_type ilu

TEST*/
