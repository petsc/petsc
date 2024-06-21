static char help[] = "Losing nullspaces in PCFIELDSPLIT after zeroing rows.\n";

// Contributed by Jeremy Theler

#include <petscksp.h>

int main(int argc, char **args)
{
  KSP             ksp, *sub_ksp;
  Vec             x, b, rigid_mode[6];
  PetscViewer     viewer;
  PetscInt        rows, cols, size, bs, n_splits = 0;
  PetscBool       has_columns = PETSC_FALSE;
  Mat             A, K;
  MatNullSpace    nullsp, near_null_space;
  IS              is_thermal, is_mech;
  PC              pc, pc_thermal, pc_mech;
  const PetscInt *bc_thermal_indexes, *bc_mech_indexes;
  char            datafilespath[PETSC_MAX_PATH_LEN], datafile[PETSC_MAX_PATH_LEN];

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-datafilespath", datafilespath, sizeof(datafilespath), NULL));

  PetscCall(PetscStrcpy(datafile, datafilespath));
  PetscCall(PetscStrcat(datafile, "/lostnullspace/"));
  PetscCall(PetscStrcat(datafile, "A.bin"));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, datafile, FILE_MODE_READ, &viewer));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatLoad(A, viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscCall(MatGetNearNullSpace(A, &nullsp));
  PetscCall(MatGetSize(A, &rows, &cols));
  PetscCall(MatGetBlockSize(A, &bs));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "A has rows = %" PetscInt_FMT ", cols = %" PetscInt_FMT ", bs = %" PetscInt_FMT ", nearnullsp = %p\n", rows, cols, bs, nullsp));

  PetscCall(PetscStrcpy(datafile, datafilespath));
  PetscCall(PetscStrcat(datafile, "/lostnullspace/"));
  PetscCall(PetscStrcat(datafile, "is.bin"));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, datafile, FILE_MODE_READ, &viewer));

  PetscCall(ISCreate(PETSC_COMM_WORLD, &is_thermal));
  PetscCall(ISLoad(is_thermal, viewer));
  PetscCall(ISGetSize(is_thermal, &size));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "thermal field size = %" PetscInt_FMT " \n", size));

  PetscCall(ISCreate(PETSC_COMM_WORLD, &is_mech));
  PetscCall(ISLoad(is_mech, viewer));
  PetscCall(ISGetSize(is_mech, &size));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "mechanical field size = %" PetscInt_FMT " \n", size));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscCall(MatCreateVecs(A, &x, &b));
  PetscCall(VecZeroEntries(x));
  PetscCall(VecZeroEntries(b));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));

  PetscCall(KSPSetType(ksp, KSPPREONLY));

  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCFIELDSPLIT));
  PetscCall(PCFieldSplitSetIS(pc, "thermal", is_thermal));
  PetscCall(PCFieldSplitSetIS(pc, "mechanical", is_mech));
  PetscCall(PCSetUp(pc));
  PetscCall(PCFieldSplitGetSubKSP(pc, &n_splits, &sub_ksp));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "n_splits = %" PetscInt_FMT " \n", n_splits));
  PetscCall(KSPSetType(sub_ksp[0], KSPGMRES));

  PetscCall(KSPGetPC(sub_ksp[0], &pc_thermal));
  PetscCall(PCSetType(pc_thermal, PCJACOBI));
  PetscCall(KSPSetFromOptions(sub_ksp[0]));

  PetscCall(KSPSetType(sub_ksp[1], KSPCG));
  PetscCall(KSPGetPC(sub_ksp[1], &pc_mech));
  PetscCall(PCSetType(pc_mech, PCGAMG));
  PetscCall(KSPSetFromOptions(sub_ksp[1]));

  PetscCall(PetscStrcpy(datafile, datafilespath));
  PetscCall(PetscStrcat(datafile, "/lostnullspace/"));
  PetscCall(PetscStrcat(datafile, "rigid-modes.bin"));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, datafile, FILE_MODE_READ, &viewer));
  for (PetscInt i = 0; i < 6; i++) {
    PetscCall(VecCreate(PETSC_COMM_WORLD, &rigid_mode[i]));
    PetscCall(VecLoad(rigid_mode[i], viewer));
  }
  PetscCall(PetscViewerDestroy(&viewer));

  PetscCall(MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_FALSE, 6, rigid_mode, &near_null_space));
  PetscCall(KSPGetOperators(sub_ksp[1], &K, PETSC_NULLPTR));
  PetscCall(MatSetNearNullSpace(K, near_null_space));
  PetscCall(MatSetBlockSize(K, 3));

  PetscCall(MatGetSize(K, &rows, &cols));
  PetscCall(MatGetBlockSize(K, &bs));
  PetscCall(MatGetNearNullSpace(K, &nullsp));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "K has rows = %" PetscInt_FMT ", cols = %" PetscInt_FMT ", bs = %" PetscInt_FMT ", nearnullsp = %p\n", rows, cols, bs, nullsp));

  PetscCall(ISGetIndices(is_thermal, &bc_thermal_indexes));
  PetscCall(ISGetIndices(is_mech, &bc_mech_indexes));

  // check that MatZeroRows() without MAT_KEEP_NONZERO_PATTERN does not remove the near null spaces attached to the submatrices
  PetscCall(PetscOptionsHasName(PETSC_NULLPTR, PETSC_NULLPTR, "-columns", &has_columns));
  if (has_columns == PETSC_TRUE) {
    PetscCall(MatZeroRowsColumns(A, 3, bc_mech_indexes, 1, PETSC_NULLPTR, PETSC_NULLPTR));
    PetscCall(MatZeroRowsColumns(A, 1, bc_thermal_indexes, 1, PETSC_NULLPTR, PETSC_NULLPTR));
  } else {
    PetscCall(MatZeroRows(A, 3, bc_mech_indexes, 1, PETSC_NULLPTR, PETSC_NULLPTR));
    PetscCall(MatZeroRows(A, 1, bc_thermal_indexes, 1, PETSC_NULLPTR, PETSC_NULLPTR));
  }
  PetscCall(ISRestoreIndices(is_mech, &bc_mech_indexes));
  PetscCall(ISRestoreIndices(is_thermal, &bc_thermal_indexes));

  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSetUp(ksp));
  PetscCall(KSPSolve(ksp, b, x));

  PetscCall(MatDestroy(&A));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(ISDestroy(&is_mech));
  PetscCall(ISDestroy(&is_thermal));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(MatNullSpaceDestroy(&near_null_space));
  for (PetscInt i = 0; i < 6; i++) PetscCall(VecDestroy(&rigid_mode[i]));
  PetscCall(PetscFree(sub_ksp));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

    test:
      requires: datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES)
      args: -datafilespath ${DATAFILESPATH} -ksp_view
      filter: grep "near null"

TEST*/
