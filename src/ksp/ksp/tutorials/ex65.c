
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

static PetscErrorCode ComputeMatrix(KSP, Mat, Mat, void *);
static PetscErrorCode ComputeRHS(KSP, Vec, void *);
static PetscErrorCode CreateMatrix(DM, Mat *);
static PetscErrorCode CreateGlobalVector(DM, Vec *);
static PetscErrorCode CreateLocalVector(DM, Vec *);
static PetscErrorCode Refine(DM, MPI_Comm, DM *);
static PetscErrorCode Coarsen(DM, MPI_Comm, DM *);
static PetscErrorCode CreateInterpolation(DM, DM, Mat *, Vec *);
static PetscErrorCode CreateRestriction(DM, DM, Mat *);
static PetscErrorCode Destroy(void *);

static PetscErrorCode MyDMShellCreate(MPI_Comm comm, DM da, DM *shell)
{
  PetscCall(DMShellCreate(comm, shell));
  PetscCall(DMShellSetContext(*shell, da));
  PetscCall(DMShellSetCreateMatrix(*shell, CreateMatrix));
  PetscCall(DMShellSetCreateGlobalVector(*shell, CreateGlobalVector));
  PetscCall(DMShellSetCreateLocalVector(*shell, CreateLocalVector));
  PetscCall(DMShellSetRefine(*shell, Refine));
  PetscCall(DMShellSetCoarsen(*shell, Coarsen));
  PetscCall(DMShellSetCreateInterpolation(*shell, CreateInterpolation));
  PetscCall(DMShellSetCreateRestriction(*shell, CreateRestriction));
  PetscCall(DMShellSetDestroyContext(*shell, Destroy));
  return 0;
}

int main(int argc, char **argv)
{
  KSP      ksp;
  DM       da, shell;
  PetscInt levels;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, 129, 1, 1, 0, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(MyDMShellCreate(PETSC_COMM_WORLD, da, &shell));
  /* these two lines are not needed but allow PCMG to automatically know how many multigrid levels the user wants */
  PetscCall(DMGetRefineLevel(da, &levels));
  PetscCall(DMSetRefineLevel(shell, levels));

  PetscCall(KSPSetDM(ksp, shell));
  PetscCall(KSPSetComputeRHS(ksp, ComputeRHS, NULL));
  PetscCall(KSPSetComputeOperators(ksp, ComputeMatrix, NULL));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp, NULL, NULL));

  PetscCall(KSPDestroy(&ksp));
  PetscCall(DMDestroy(&shell));
  PetscCall(PetscFinalize());
  return 0;
}

static PetscErrorCode Destroy(void *ctx)
{
  PetscCall(DMDestroy((DM *)&ctx));
  return 0;
}

static PetscErrorCode CreateMatrix(DM shell, Mat *A)
{
  DM da;

  PetscCall(DMShellGetContext(shell, &da));
  PetscCall(DMCreateMatrix(da, A));
  return 0;
}

static PetscErrorCode CreateInterpolation(DM dm1, DM dm2, Mat *mat, Vec *vec)
{
  DM da1, da2;

  PetscCall(DMShellGetContext(dm1, &da1));
  PetscCall(DMShellGetContext(dm2, &da2));
  PetscCall(DMCreateInterpolation(da1, da2, mat, vec));
  return 0;
}

static PetscErrorCode CreateRestriction(DM dm1, DM dm2, Mat *mat)
{
  DM  da1, da2;
  Mat tmat;

  PetscCall(DMShellGetContext(dm1, &da1));
  PetscCall(DMShellGetContext(dm2, &da2));
  PetscCall(DMCreateInterpolation(da1, da2, &tmat, NULL));
  PetscCall(MatTranspose(tmat, MAT_INITIAL_MATRIX, mat));
  PetscCall(MatDestroy(&tmat));
  return 0;
}

static PetscErrorCode CreateGlobalVector(DM shell, Vec *x)
{
  DM da;

  PetscCall(DMShellGetContext(shell, &da));
  PetscCall(DMCreateGlobalVector(da, x));
  PetscCall(VecSetDM(*x, shell));
  return 0;
}

static PetscErrorCode CreateLocalVector(DM shell, Vec *x)
{
  DM da;

  PetscCall(DMShellGetContext(shell, &da));
  PetscCall(DMCreateLocalVector(da, x));
  PetscCall(VecSetDM(*x, shell));
  return 0;
}

static PetscErrorCode Refine(DM shell, MPI_Comm comm, DM *dmnew)
{
  DM da, dafine;

  PetscCall(DMShellGetContext(shell, &da));
  PetscCall(DMRefine(da, comm, &dafine));
  PetscCall(MyDMShellCreate(PetscObjectComm((PetscObject)shell), dafine, dmnew));
  return 0;
}

static PetscErrorCode Coarsen(DM shell, MPI_Comm comm, DM *dmnew)
{
  DM da, dacoarse;

  PetscCall(DMShellGetContext(shell, &da));
  PetscCall(DMCoarsen(da, comm, &dacoarse));
  PetscCall(MyDMShellCreate(PetscObjectComm((PetscObject)shell), dacoarse, dmnew));
  return 0;
}

static PetscErrorCode ComputeRHS(KSP ksp, Vec b, void *ctx)
{
  PetscInt    mx, idx[2];
  PetscScalar h, v[2];
  DM          da, shell;

  PetscFunctionBeginUser;
  PetscCall(KSPGetDM(ksp, &shell));
  PetscCall(DMShellGetContext(shell, &da));
  PetscCall(DMDAGetInfo(da, 0, &mx, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
  h = 1.0 / ((mx - 1));
  PetscCall(VecSet(b, h));
  idx[0] = 0;
  idx[1] = mx - 1;
  v[0] = v[1] = 0.0;
  PetscCall(VecSetValues(b, 2, idx, v, INSERT_VALUES));
  PetscCall(VecAssemblyBegin(b));
  PetscCall(VecAssemblyEnd(b));
  PetscFunctionReturn(0);
}

static PetscErrorCode ComputeMatrix(KSP ksp, Mat J, Mat jac, void *ctx)
{
  PetscInt    i, mx, xm, xs;
  PetscScalar v[3], h;
  MatStencil  row, col[3];
  DM          da, shell;

  PetscFunctionBeginUser;
  PetscCall(KSPGetDM(ksp, &shell));
  PetscCall(DMShellGetContext(shell, &da));
  PetscCall(DMDAGetInfo(da, 0, &mx, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
  PetscCall(DMDAGetCorners(da, &xs, 0, 0, &xm, 0, 0));
  h = 1.0 / (mx - 1);

  for (i = xs; i < xs + xm; i++) {
    row.i = i;
    if (i == 0 || i == mx - 1) {
      v[0] = 2.0 / h;
      PetscCall(MatSetValuesStencil(jac, 1, &row, 1, &row, v, INSERT_VALUES));
    } else {
      v[0]     = (-1.0) / h;
      col[0].i = i - 1;
      v[1]     = (2.0) / h;
      col[1].i = row.i;
      v[2]     = (-1.0) / h;
      col[2].i = i + 1;
      PetscCall(MatSetValuesStencil(jac, 1, &row, 3, col, v, INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

/*TEST

   test:
      nsize: 4
      args: -ksp_monitor -pc_type mg -da_refine 3

TEST*/
