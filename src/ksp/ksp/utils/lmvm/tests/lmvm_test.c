const char help[] = "Coverage and edge case test for LMVM";

#include <petscksp.h>
#include <petscmath.h>

int main(int argc, char **argv)
{
  PetscInt type = 0, n = 10;
  Mat      B;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, help, "KSP");
  /* LMVM Types. 0: LMVMDBFGS, 1: LMVMDDFP, 2: LMVMDQN */
  PetscCall(PetscOptionsInt("-type", "LMVM Type", __FILE__, type, &type, NULL));
  PetscOptionsEnd();
  if (type == 0) {
    PetscCall(MatCreateLMVMDBFGS(PETSC_COMM_WORLD, PETSC_DECIDE, n, &B));
  } else if (type == 1) {
    PetscCall(MatCreateLMVMDDFP(PETSC_COMM_WORLD, PETSC_DECIDE, n, &B));
  } else if (type == 2) {
    PetscCall(MatCreateLMVMDQN(PETSC_COMM_WORLD, PETSC_DECIDE, n, &B));
  } else {
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_INCOMP, "Incompatible LMVM Type.");
  }
  PetscCall(MatSetFromOptions(B));
  PetscCall(MatSetUp(B));
  PetscCall(MatLMVMDenseSetType(B, MAT_LMVM_DENSE_INPLACE));
  PetscCall(MatDestroy(&B));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    output_file: output/empty.out
    nsize: {{1 2}}
    args: -mat_lmvm_scale_type {{none scalar diagonal}} -type {{0 1 2}}

TEST*/
