const char help[] = "Test whether KSPSolve() can be executed without zeroing output for KSPPREONLY";

#include <petscksp.h>

static PetscErrorCode PCApply_scale(PC pc, Vec x, Vec y)
{
  PetscFunctionBegin;
  PetscCall(VecCopy(x, y));
  PetscCall(VecScale(y, 0.5));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode VecSet_Error(Vec x, PetscReal _alpha)
{
  PetscFunctionBegin;
  SETERRQ(PetscObjectComm((PetscObject)x), PETSC_ERR_SUP, "Cannot zero this vector");
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  Mat      mat;
  PetscInt M = 10;
  MPI_Comm comm;
  Vec      x, y;
  KSP      ksp;
  PC       pc;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscCall(MatCreateConstantDiagonal(comm, PETSC_DETERMINE, PETSC_DETERMINE, M, M, 2.0, &mat));
  PetscCall(MatCreateVecs(mat, &x, &y));
  PetscCall(VecSet(x, 1.0));
  PetscCall(VecSet(y, 3.0));
  PetscCall(VecSetOperation(y, VECOP_SET, (void (*)(void))VecSet_Error));

  PetscCall(KSPCreate(comm, &ksp));
  PetscCall(KSPSetOperators(ksp, mat, mat));
  PetscCall(KSPSetType(ksp, KSPPREONLY));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCSHELL));

  PetscCall(PCShellSetApply(pc, PCApply_scale));
  PetscCall(KSPSolve(ksp, x, y));

  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&x));
  PetscCall(MatDestroy(&mat));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0

TEST*/
