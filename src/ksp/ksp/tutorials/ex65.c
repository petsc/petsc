
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
  PetscErrorCode ierr;

  ierr = DMShellCreate(comm,shell);CHKERRQ(ierr);
  ierr = DMShellSetContext(*shell,da);CHKERRQ(ierr);
  ierr = DMShellSetCreateMatrix(*shell,CreateMatrix);CHKERRQ(ierr);
  ierr = DMShellSetCreateGlobalVector(*shell,CreateGlobalVector);CHKERRQ(ierr);
  ierr = DMShellSetCreateLocalVector(*shell,CreateLocalVector);CHKERRQ(ierr);
  ierr = DMShellSetRefine(*shell,Refine);CHKERRQ(ierr);
  ierr = DMShellSetCoarsen(*shell,Coarsen);CHKERRQ(ierr);
  ierr = DMShellSetCreateInterpolation(*shell,CreateInterpolation);CHKERRQ(ierr);
  ierr = DMShellSetCreateRestriction(*shell,CreateRestriction);CHKERRQ(ierr);
  return 0;
}

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  KSP            ksp;
  DM             da,shell;
  PetscInt       levels;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,129,1,1,0,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = MyDMShellCreate(PETSC_COMM_WORLD,da,&shell);CHKERRQ(ierr);
  /* these two lines are not needed but allow PCMG to automatically know how many multigrid levels the user wants */
  ierr = DMGetRefineLevel(da,&levels);CHKERRQ(ierr);
  ierr = DMSetRefineLevel(shell,levels);CHKERRQ(ierr);

  ierr = KSPSetDM(ksp,shell);CHKERRQ(ierr);
  ierr = KSPSetComputeRHS(ksp,ComputeRHS,NULL);CHKERRQ(ierr);
  ierr = KSPSetComputeOperators(ksp,ComputeMatrix,NULL);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,NULL,NULL);CHKERRQ(ierr);

  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = DMDestroy(&shell);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

static PetscErrorCode CreateMatrix(DM shell,Mat *A)
{
  PetscErrorCode ierr;
  DM             da;

  ierr = DMShellGetContext(shell,(void**)&da);CHKERRQ(ierr);
  ierr = DMCreateMatrix(da,A);CHKERRQ(ierr);
  return 0;
}

static PetscErrorCode CreateInterpolation(DM dm1,DM dm2,Mat *mat,Vec *vec)
{
  DM             da1,da2;
  PetscErrorCode ierr;

  ierr = DMShellGetContext(dm1,(void**)&da1);CHKERRQ(ierr);
  ierr = DMShellGetContext(dm2,(void**)&da2);CHKERRQ(ierr);
  ierr = DMCreateInterpolation(da1,da2,mat,vec);CHKERRQ(ierr);
  return 0;
}

static PetscErrorCode CreateRestriction(DM dm1,DM dm2,Mat *mat)
{
  DM             da1,da2;
  PetscErrorCode ierr;
  Mat            tmat;

  ierr = DMShellGetContext(dm1,(void**)&da1);CHKERRQ(ierr);
  ierr = DMShellGetContext(dm2,(void**)&da2);CHKERRQ(ierr);
  ierr = DMCreateInterpolation(da1,da2,&tmat,NULL);CHKERRQ(ierr);
  ierr = MatTranspose(tmat,MAT_INITIAL_MATRIX,mat);CHKERRQ(ierr);
  ierr = MatDestroy(&tmat);CHKERRQ(ierr);
  return 0;
}

static PetscErrorCode CreateGlobalVector(DM shell,Vec *x)
{
  PetscErrorCode ierr;
  DM             da;

  ierr = DMShellGetContext(shell,(void**)&da);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,x);CHKERRQ(ierr);
  ierr = VecSetDM(*x,shell);CHKERRQ(ierr);
  return 0;
}

static PetscErrorCode CreateLocalVector(DM shell,Vec *x)
{
  PetscErrorCode ierr;
  DM             da;

  ierr = DMShellGetContext(shell,(void**)&da);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(da,x);CHKERRQ(ierr);
  ierr = VecSetDM(*x,shell);CHKERRQ(ierr);
  return 0;
}

static PetscErrorCode Refine(DM shell,MPI_Comm comm,DM *dmnew)
{
  PetscErrorCode ierr;
  DM             da,dafine;

  ierr = DMShellGetContext(shell,(void**)&da);CHKERRQ(ierr);
  ierr = DMRefine(da,comm,&dafine);CHKERRQ(ierr);
  ierr = MyDMShellCreate(PetscObjectComm((PetscObject)shell),dafine,dmnew);CHKERRQ(ierr);
  return 0;
}

static PetscErrorCode Coarsen(DM shell,MPI_Comm comm,DM *dmnew)
{
  PetscErrorCode ierr;
  DM             da,dacoarse;

  ierr = DMShellGetContext(shell,(void**)&da);CHKERRQ(ierr);
  ierr = DMCoarsen(da,comm,&dacoarse);CHKERRQ(ierr);
  ierr = MyDMShellCreate(PetscObjectComm((PetscObject)shell),dacoarse,dmnew);CHKERRQ(ierr);
  /* discard an "extra" reference count to dacoarse */
  ierr = DMDestroy(&dacoarse);CHKERRQ(ierr);
  return 0;
}

static PetscErrorCode ComputeRHS(KSP ksp,Vec b,void *ctx)
{
  PetscErrorCode ierr;
  PetscInt       mx,idx[2];
  PetscScalar    h,v[2];
  DM             da,shell;

  PetscFunctionBeginUser;
  ierr   = KSPGetDM(ksp,&shell);CHKERRQ(ierr);
  ierr   = DMShellGetContext(shell,(void**)&da);CHKERRQ(ierr);
  ierr   = DMDAGetInfo(da,0,&mx,0,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  h      = 1.0/((mx-1));
  ierr   = VecSet(b,h);CHKERRQ(ierr);
  idx[0] = 0; idx[1] = mx -1;
  v[0]   = v[1] = 0.0;
  ierr   = VecSetValues(b,2,idx,v,INSERT_VALUES);CHKERRQ(ierr);
  ierr   = VecAssemblyBegin(b);CHKERRQ(ierr);
  ierr   = VecAssemblyEnd(b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ComputeMatrix(KSP ksp,Mat J,Mat jac,void *ctx)
{
  PetscErrorCode ierr;
  PetscInt       i,mx,xm,xs;
  PetscScalar    v[3],h;
  MatStencil     row,col[3];
  DM             da,shell;

  PetscFunctionBeginUser;
  ierr = KSPGetDM(ksp,&shell);CHKERRQ(ierr);
  ierr   = DMShellGetContext(shell,(void**)&da);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,0,&mx,0,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&xs,0,0,&xm,0,0);CHKERRQ(ierr);
  h    = 1.0/(mx-1);

  for (i=xs; i<xs+xm; i++) {
    row.i = i;
    if (i==0 || i==mx-1) {
      v[0] = 2.0/h;
      ierr = MatSetValuesStencil(jac,1,&row,1,&row,v,INSERT_VALUES);CHKERRQ(ierr);
    } else {
      v[0]  = (-1.0)/h;col[0].i = i-1;
      v[1]  = (2.0)/h;col[1].i = row.i;
      v[2]  = (-1.0)/h;col[2].i = i+1;
      ierr  = MatSetValuesStencil(jac,1,&row,3,col,v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*TEST

   test:
      nsize: 4
      args: -ksp_monitor -pc_type mg -da_refine 3

TEST*/
