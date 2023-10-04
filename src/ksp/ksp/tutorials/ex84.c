#include <petscksp.h>

static char help[] = "Demonstrate PCFIELDSPLIT after MatZeroRowsColumns() inside PCREDISTRIBUTE";

int main(int argc, char **argv)
{
  PetscMPIInt rank, size;
  Mat         A;
  IS          field0, field1, zeroedrows;
  PetscInt    row;
  KSP         ksp, kspred;
  PC          pc;
  Vec         x, b;

  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCheck(size == 2, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Must be run with 2 MPI processes");

  // Set up a small problem with 2 dofs on rank 0 and 4 on rank 1
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, !rank ? 2 : 4, !rank ? 2 : 4, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatSetFromOptions(A));
  if (rank == 0) {
    PetscCall(MatSetValue(A, 0, 0, 2.0, INSERT_VALUES));
    PetscCall(MatSetValue(A, 0, 1, -1.0, INSERT_VALUES));
    PetscCall(MatSetValue(A, 1, 1, 3.0, INSERT_VALUES));
    PetscCall(MatSetValue(A, 1, 2, -1.0, INSERT_VALUES));
  } else if (rank == 1) {
    PetscCall(MatSetValue(A, 2, 2, 4.0, INSERT_VALUES));
    PetscCall(MatSetValue(A, 2, 3, -1.0, INSERT_VALUES));
    PetscCall(MatSetValue(A, 3, 3, 5.0, INSERT_VALUES));
    PetscCall(MatSetValue(A, 3, 4, -1.0, INSERT_VALUES));
    PetscCall(MatSetValue(A, 4, 4, 6.0, INSERT_VALUES));
    PetscCall(MatSetValue(A, 4, 5, -1.0, INSERT_VALUES));
    PetscCall(MatSetValue(A, 5, 5, 7.0, INSERT_VALUES));
    PetscCall(MatSetValue(A, 5, 4, -1, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatView(A, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(MatCreateVecs(A, &b, &x));
  PetscCall(VecSet(b, 1.0));

  // the two fields for PCFIELDSPLIT are initially (0,2,4) and (1,3,5)
  PetscCall(ISCreateStride(PETSC_COMM_WORLD, !rank ? 1 : 2, !rank ? 0 : 2, 2, &field0));
  PetscCall(ISView(field0, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(ISCreateStride(PETSC_COMM_WORLD, !rank ? 1 : 2, !rank ? 1 : 3, 2, &field1));
  PetscCall(ISView(field1, PETSC_VIEWER_STDOUT_WORLD));

  // these rows are being zeroed (0,3)
  row = (!rank ? 0 : 3);
  PetscCall(ISCreateGeneral(PETSC_COMM_WORLD, 1, &row, PETSC_COPY_VALUES, &zeroedrows));
  PetscCall(ISView(zeroedrows, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(MatZeroRowsColumnsIS(A, zeroedrows, 1.0, NULL, NULL));
  PetscCall(MatView(A, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCREDISTRIBUTE));
  /* note that one provides the indices for the fields on the original full system, not on the reduced system PCREDISTRIBUTE solves */
  PetscCall(PCFieldSplitSetIS(pc, NULL, field0));
  PetscCall(PCFieldSplitSetIS(pc, NULL, field1));
  PetscCall(KSPSetFromOptions(ksp));

  PetscCall(KSPSolve(ksp, b, x));

  PetscCall(KSPSetInitialGuessNonzero(ksp, PETSC_TRUE));
  PetscCall(PCRedistributeGetKSP(pc, &kspred));
  PetscCall(KSPSetInitialGuessNonzero(kspred, PETSC_TRUE));
  PetscCall(KSPSolve(ksp, b, x));

  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&x));
  PetscCall(MatDestroy(&A));
  PetscCall(ISDestroy(&field0));
  PetscCall(ISDestroy(&field1));
  PetscCall(ISDestroy(&zeroedrows));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
     nsize: 2
     args: -ksp_monitor -redistribute_ksp_monitor -ksp_view -redistribute_pc_type fieldsplit -ksp_type preonly

TEST*/
