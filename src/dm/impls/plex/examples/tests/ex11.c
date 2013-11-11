static char help[] = "Tests for DMLabel\n\n";

#include <petscdmplex.h>

#undef __FUNCT__
#define __FUNCT__ "TestInsertion"
PetscErrorCode TestInsertion()
{
  DMLabel        label, label2;
  const PetscInt values[5] = {0, 3, 4, -1, 176}, N = 10000;
  PetscInt       i, v;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMLabelCreate("Test Label", &label);CHKERRQ(ierr);
  for (i = 0; i < N; ++i) {
    ierr = DMLabelSetValue(label, i, values[i%5]);CHKERRQ(ierr);
  }
  /* Test get in hash mode */
  for (i = 0; i < N; ++i) {
    PetscInt val;

    ierr = DMLabelGetValue(label, i, &val);CHKERRQ(ierr);
    if (val != values[i%5]) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Value %d for point %d should be %d", val, i, values[i%5]);
  }
  /* Test stratum */
  for (v = 0; v < 5; ++v) {
    IS              stratum;
    const PetscInt *points;
    PetscInt        n;

    ierr = DMLabelGetStratumIS(label, values[v], &stratum);CHKERRQ(ierr);
    ierr = ISGetIndices(stratum, &points);CHKERRQ(ierr);
    ierr = ISGetLocalSize(stratum, &n);CHKERRQ(ierr);
    for (i = 0; i < n; ++i) {
      if (points[i] != i*5+v) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Point %d should be %d", points[i], i*5+v);
    }
    ierr = ISRestoreIndices(stratum, &points);CHKERRQ(ierr);
    ierr = ISDestroy(&stratum);CHKERRQ(ierr);
  }
  /* Test get in array mode */
  for (i = 0; i < N; ++i) {
    PetscInt val;

    ierr = DMLabelGetValue(label, i, &val);CHKERRQ(ierr);
    if (val != values[i%5]) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Value %d should be %d", val, values[i%5]);
  }
  /* Test Duplicate */
  ierr = DMLabelDuplicate(label, &label2);CHKERRQ(ierr);
  for (i = 0; i < N; ++i) {
    PetscInt val;

    ierr = DMLabelGetValue(label2, i, &val);CHKERRQ(ierr);
    if (val != values[i%5]) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Value %d should be %d", val, values[i%5]);
  }
  ierr = DMLabelDestroy(&label2);CHKERRQ(ierr);
  ierr = DMLabelDestroy(&label);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  //AppCtx         user;                 /* user-defined work context */
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);CHKERRQ(ierr);
  //ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = TestInsertion();CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
