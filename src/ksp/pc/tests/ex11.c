static const char help[] = "Tests PCMG setup with DMKSP{Create|Compute}Operators.\n\n";

#include <petscksp.h>
#include <petscdmda.h>

typedef struct {
  PetscBool same_operator;
  PetscInt  ncreate;
  PetscInt  ncompute;
} AppCtx;

static PetscErrorCode AssembleDiagonal(Mat A, PetscScalar diag)
{
  PetscInt rstart, rend;

  PetscFunctionBeginUser;
  PetscCall(MatZeroEntries(A));
  PetscCall(MatGetOwnershipRange(A, &rstart, &rend));
  for (PetscInt row = rstart; row < rend; row++) PetscCall(MatSetValue(A, row, row, diag, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateOperators(KSP ksp, Mat *A, Mat *P, void *ctx)
{
  AppCtx *user = (AppCtx *)ctx;
  DM      dm;

  PetscFunctionBeginUser;
  PetscCall(KSPGetDM(ksp, &dm));
  PetscCall(DMCreateMatrix(dm, A));
  if (!user->same_operator) PetscCall(DMCreateMatrix(dm, P));
  else *P = *A; /* No need to increment ref count of A */
  user->ncreate++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeOperators(KSP ksp, Mat A, Mat P, void *ctx)
{
  AppCtx *user = (AppCtx *)ctx;

  PetscFunctionBeginUser;
  if (!user->same_operator) {
    PetscCheck(A != P, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Expected distinct Amat and Pmat");
    PetscCall(AssembleDiagonal(A, 2.0));
    PetscCall(AssembleDiagonal(P, 3.0));
  } else {
    PetscCheck(A == P, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Expected Pmat-only operator");
    PetscCall(AssembleDiagonal(P, 3.0));
  }
  user->ncompute++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  AppCtx    user;
  DM        dm;
  KSP       ksp, cksp;
  PC        pc;
  Mat       A, P, cA, cP;
  PetscBool user_finest = PETSC_TRUE;

  PetscFunctionBeginUser;
  user.same_operator = PETSC_FALSE;
  user.ncreate       = 0;
  user.ncompute      = 0;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-user_finest", &user_finest, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-same_operator", &user.same_operator, NULL));

  PetscCall(DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, 9, 1, 1, NULL, &dm));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));

  PetscCall(DMKSPSetCreateOperators(dm, CreateOperators, &user));
  PetscCall(DMKSPSetComputeOperators(dm, ComputeOperators, &user));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetDM(ksp, dm));
  PetscCall(KSPSetDMActive(ksp, KSP_DMACTIVE_OPERATOR, (PetscBool)!user_finest));
  PetscCall(KSPSetType(ksp, KSPPREONLY));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCMG));
  PetscCall(PCMGSetLevels(pc, 2, NULL));
  if (user_finest) {
    PetscCall(DMCreateMatrix(dm, &A));
    PetscCall(AssembleDiagonal(A, 2.0));
    P = A;
    if (!user.same_operator) {
      PetscCall(DMCreateMatrix(dm, &P));
      PetscCall(AssembleDiagonal(P, 3.0));
    }
    PetscCall(KSPSetOperators(ksp, A, P));
    PetscCall(MatDestroy(&A));
    if (!user.same_operator) PetscCall(MatDestroy(&P));
  }
  PetscCall(KSPSetFromOptions(ksp));

  PetscCall(KSPSetUp(ksp));
  PetscCheck(user.ncreate == (user_finest ? 1 : 2), PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Expected PCMG to create at least one rediscretized level operator");
  PetscCheck(user.ncompute == (user_finest ? 1 : 2), PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Expected PCMG to compute at least one rediscretized level operator");

  PetscCall(PCMGGetCoarseSolve(pc, &cksp));
  PetscCall(KSPGetOperators(cksp, &cA, &cP));
  if (!user.same_operator) PetscCheck(cA != cP, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Coarse KSP operators were not distinct");
  else PetscCheck(cA == cP, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Coarse KSP operators were not Pmat-only");

  PetscCall(KSPDestroy(&ksp));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      args: -pc_mg_galerkin none -user_finest {{0 1}} -same_operator {{0 1}}
      output_file: output/empty.out

TEST*/
