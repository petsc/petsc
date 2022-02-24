
/*
 Partial differential equation

   d   d u = 1, 0 < x < 1,
   --   --
   dx   dx
with boundary conditions

   u = 0 for x = 0, x = 1

   This uses multigrid to solve the linear system

   Demonstrates how to build a DMSHELL for managing multigrid. The DMSHELL simply creates a
   DMDA1d to construct all the needed PETSc objects.

*/

static char help[] = "Solves 1D constant coefficient Laplacian using DMSHELL and multigrid.\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmshell.h>
#include <petscksp.h>

static PetscErrorCode ComputeMatrix(KSP,Mat,Mat,void*);
static PetscErrorCode ComputeRHS(KSP,Vec,void*);
static PetscErrorCode CreateMatrix(DM,Mat*);
static PetscErrorCode CreateGlobalVector(DM,Vec*);
static PetscErrorCode CreateLocalVector(DM,Vec*);
static PetscErrorCode Refine(DM,MPI_Comm,DM*);
static PetscErrorCode Coarsen(DM,MPI_Comm,DM*);
static PetscErrorCode CreateInterpolation(DM,DM,Mat*,Vec*);
static PetscErrorCode CreateRestriction(DM,DM,Mat*);

static PetscErrorCode MyDMShellCreate(MPI_Comm comm,DM da,DM *shell)
{

  CHKERRQ(DMShellCreate(comm,shell));
  CHKERRQ(DMShellSetContext(*shell,da));
  CHKERRQ(DMShellSetCreateMatrix(*shell,CreateMatrix));
  CHKERRQ(DMShellSetCreateGlobalVector(*shell,CreateGlobalVector));
  CHKERRQ(DMShellSetCreateLocalVector(*shell,CreateLocalVector));
  CHKERRQ(DMShellSetRefine(*shell,Refine));
  CHKERRQ(DMShellSetCoarsen(*shell,Coarsen));
  CHKERRQ(DMShellSetCreateInterpolation(*shell,CreateInterpolation));
  CHKERRQ(DMShellSetCreateRestriction(*shell,CreateRestriction));
  return 0;
}

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  KSP            ksp;
  DM             da,shell;
  PetscInt       levels;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));
  CHKERRQ(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,129,1,1,0,&da));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));
  CHKERRQ(MyDMShellCreate(PETSC_COMM_WORLD,da,&shell));
  /* these two lines are not needed but allow PCMG to automatically know how many multigrid levels the user wants */
  CHKERRQ(DMGetRefineLevel(da,&levels));
  CHKERRQ(DMSetRefineLevel(shell,levels));

  CHKERRQ(KSPSetDM(ksp,shell));
  CHKERRQ(KSPSetComputeRHS(ksp,ComputeRHS,NULL));
  CHKERRQ(KSPSetComputeOperators(ksp,ComputeMatrix,NULL));
  CHKERRQ(KSPSetFromOptions(ksp));
  CHKERRQ(KSPSolve(ksp,NULL,NULL));

  CHKERRQ(KSPDestroy(&ksp));
  CHKERRQ(DMDestroy(&shell));
  CHKERRQ(DMDestroy(&da));
  ierr = PetscFinalize();
  return ierr;
}

static PetscErrorCode CreateMatrix(DM shell,Mat *A)
{
  DM             da;

  CHKERRQ(DMShellGetContext(shell,&da));
  CHKERRQ(DMCreateMatrix(da,A));
  return 0;
}

static PetscErrorCode CreateInterpolation(DM dm1,DM dm2,Mat *mat,Vec *vec)
{
  DM             da1,da2;

  CHKERRQ(DMShellGetContext(dm1,&da1));
  CHKERRQ(DMShellGetContext(dm2,&da2));
  CHKERRQ(DMCreateInterpolation(da1,da2,mat,vec));
  return 0;
}

static PetscErrorCode CreateRestriction(DM dm1,DM dm2,Mat *mat)
{
  DM             da1,da2;
  Mat            tmat;

  CHKERRQ(DMShellGetContext(dm1,&da1));
  CHKERRQ(DMShellGetContext(dm2,&da2));
  CHKERRQ(DMCreateInterpolation(da1,da2,&tmat,NULL));
  CHKERRQ(MatTranspose(tmat,MAT_INITIAL_MATRIX,mat));
  CHKERRQ(MatDestroy(&tmat));
  return 0;
}

static PetscErrorCode CreateGlobalVector(DM shell,Vec *x)
{
  DM             da;

  CHKERRQ(DMShellGetContext(shell,&da));
  CHKERRQ(DMCreateGlobalVector(da,x));
  CHKERRQ(VecSetDM(*x,shell));
  return 0;
}

static PetscErrorCode CreateLocalVector(DM shell,Vec *x)
{
  DM             da;

  CHKERRQ(DMShellGetContext(shell,&da));
  CHKERRQ(DMCreateLocalVector(da,x));
  CHKERRQ(VecSetDM(*x,shell));
  return 0;
}

static PetscErrorCode Refine(DM shell,MPI_Comm comm,DM *dmnew)
{
  DM             da,dafine;

  CHKERRQ(DMShellGetContext(shell,&da));
  CHKERRQ(DMRefine(da,comm,&dafine));
  CHKERRQ(MyDMShellCreate(PetscObjectComm((PetscObject)shell),dafine,dmnew));
  return 0;
}

static PetscErrorCode Coarsen(DM shell,MPI_Comm comm,DM *dmnew)
{
  DM             da,dacoarse;

  CHKERRQ(DMShellGetContext(shell,&da));
  CHKERRQ(DMCoarsen(da,comm,&dacoarse));
  CHKERRQ(MyDMShellCreate(PetscObjectComm((PetscObject)shell),dacoarse,dmnew));
  /* discard an "extra" reference count to dacoarse */
  CHKERRQ(DMDestroy(&dacoarse));
  return 0;
}

static PetscErrorCode ComputeRHS(KSP ksp,Vec b,void *ctx)
{
  PetscInt       mx,idx[2];
  PetscScalar    h,v[2];
  DM             da,shell;

  PetscFunctionBeginUser;
  CHKERRQ(KSPGetDM(ksp,&shell));
  CHKERRQ(DMShellGetContext(shell,&da));
  CHKERRQ(DMDAGetInfo(da,0,&mx,0,0,0,0,0,0,0,0,0,0,0));
  h      = 1.0/((mx-1));
  CHKERRQ(VecSet(b,h));
  idx[0] = 0; idx[1] = mx -1;
  v[0]   = v[1] = 0.0;
  CHKERRQ(VecSetValues(b,2,idx,v,INSERT_VALUES));
  CHKERRQ(VecAssemblyBegin(b));
  CHKERRQ(VecAssemblyEnd(b));
  PetscFunctionReturn(0);
}

static PetscErrorCode ComputeMatrix(KSP ksp,Mat J,Mat jac,void *ctx)
{
  PetscInt       i,mx,xm,xs;
  PetscScalar    v[3],h;
  MatStencil     row,col[3];
  DM             da,shell;

  PetscFunctionBeginUser;
  CHKERRQ(KSPGetDM(ksp,&shell));
  CHKERRQ(DMShellGetContext(shell,&da));
  CHKERRQ(DMDAGetInfo(da,0,&mx,0,0,0,0,0,0,0,0,0,0,0));
  CHKERRQ(DMDAGetCorners(da,&xs,0,0,&xm,0,0));
  h    = 1.0/(mx-1);

  for (i=xs; i<xs+xm; i++) {
    row.i = i;
    if (i==0 || i==mx-1) {
      v[0] = 2.0/h;
      CHKERRQ(MatSetValuesStencil(jac,1,&row,1,&row,v,INSERT_VALUES));
    } else {
      v[0]  = (-1.0)/h;col[0].i = i-1;
      v[1]  = (2.0)/h;col[1].i = row.i;
      v[2]  = (-1.0)/h;col[2].i = i+1;
      CHKERRQ(MatSetValuesStencil(jac,1,&row,3,col,v,INSERT_VALUES));
    }
  }
  CHKERRQ(MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

/*TEST

   test:
      nsize: 4
      args: -ksp_monitor -pc_type mg -da_refine 3

TEST*/
