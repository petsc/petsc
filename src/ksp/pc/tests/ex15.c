static char help[] = "Tests PCMGGetRScale with a square interpolation matrix and different row and column layouts.\n";

#include <petscksp.h>

int main(int argc, char **argv)
{
  const PetscScalar *values;
  Mat                P;
  PC                 pc;
  Vec                rscale;
  PetscInt           localRows, localColumns, start, end;
  PetscMPIInt        rank;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  localRows    = rank ? 3 : 1;
  localColumns = rank ? 1 : 3;
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, localRows, localColumns, 4, 4, 4, NULL, 4, NULL, &P));
  PetscCall(MatGetOwnershipRange(P, &start, &end));
  for (PetscInt row = start; row < end; ++row) {
    for (PetscInt col = 0; col < 4; ++col) PetscCall(MatSetValue(P, row, col, row + col + 1, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));

  PetscCall(PCCreate(PETSC_COMM_WORLD, &pc));
  PetscCall(PCSetType(pc, PCMG));
  PetscCall(PCMGSetLevels(pc, 2, NULL));
  PetscCall(PCMGSetInterpolation(pc, 1, P));
  PetscCall(PCMGGetRScale(pc, 1, &rscale));
  PetscCall(VecGetOwnershipRange(rscale, &start, &end));
  PetscCall(VecGetArrayRead(rscale, &values));
  for (PetscInt i = 0; i < end - start; ++i) {
    const PetscInt    col      = start + i;
    const PetscScalar expected = 1.0 / (PetscReal)(10 + 4 * col);

    PetscCheck(PetscAbsScalar(values[i] - expected) < 100.0 * PETSC_MACHINE_EPSILON, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incorrect restriction scaling at column %" PetscInt_FMT, col);
  }
  PetscCall(VecRestoreArrayRead(rscale, &values));

  PetscCall(PCDestroy(&pc));
  PetscCall(MatDestroy(&P));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    nsize: 2
    output_file: output/empty.out

TEST*/
