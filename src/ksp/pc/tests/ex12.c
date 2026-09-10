static const char help[] = "Tests PCASMGetSubKSP() ordering and reused submatrices across a change of operator.\n\n";

#include <petscksp.h>

int main(int argc, char **args)
{
  Mat       A;
  PC        pc;
  IS        is;
  Vec       x, b;
  KSP      *subksp;
  PetscInt  n            = 16, rstart, rend, nlocal;
  PetscBool before_setup = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  PetscCall(MatGetOwnershipRange(A, &rstart, &rend));
  for (PetscInt row = rstart; row < rend; row++) {
    PetscCall(MatSetValue(A, row, row, 2.0, INSERT_VALUES));
    if (row > 0) PetscCall(MatSetValue(A, row, row - 1, -1.0, INSERT_VALUES));
    if (row < n - 1) PetscCall(MatSetValue(A, row, row + 1, -1.0, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  PetscCall(MatCreateVecs(A, &x, &b));
  PetscCall(VecSet(b, 1.0));

  PetscCall(PCCreate(PETSC_COMM_WORLD, &pc));
  PetscCall(PCSetType(pc, PCASM));
  PetscCall(PCSetOperators(pc, A, A));

  PetscCall(ISCreateStride(PETSC_COMM_SELF, rend - rstart, rstart, 1, &is));
  PetscCall(PCASMSetLocalSubdomains(pc, 1, &is, NULL));

  /* Specifying the subdomains sets n_local_true, but the sub-KSPs are not
     allocated until PCSetUp(). Querying them here must raise an error rather
     than hand back a null array. */
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-before_setup", &before_setup, NULL));
  if (before_setup) PetscCall(PCASMGetSubKSP(pc, &nlocal, NULL, &subksp));

  /* After setup the sub-KSPs exist. */
  PetscCall(PCSetFromOptions(pc));
  PetscCall(PCSetUp(pc));
  PetscCall(PCASMGetSubKSP(pc, &nlocal, NULL, &subksp));
  PetscCheck(nlocal == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Expected 1 local subdomain, got %" PetscInt_FMT, nlocal);
  PetscCheck(subksp, PETSC_COMM_SELF, PETSC_ERR_PLIB, "PCASMGetSubKSP() returned a null array after PCSetUp()");
  PetscCall(PCApply(pc, b, x));

  /* A subsolver told to factor in place, e.g. -sub_pc_type ilu
     -sub_pc_factor_in_place, has overwritten its submatrix with the factors
     and left it flagged as factored. Changing the operator refills those
     submatrices with MAT_REUSE_MATRIX, which a matrix still flagged as
     factored rejects. */
  PetscCall(MatScale(A, 2.0));
  PetscCall(PCSetOperators(pc, A, A));
  PetscCall(PCSetUp(pc));
  PetscCall(PCApply(pc, b, x));

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(ISDestroy(&is));
  PetscCall(PCDestroy(&pc));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      nsize: {{1 2}}
      args: -sub_pc_type ilu -sub_pc_factor_in_place
      output_file: output/empty.out

   test:
      suffix: 2
      requires: !defined(PETSCTEST_VALGRIND) !defined(PETSC_HAVE_SANITIZER)
      args: -before_setup -petsc_ci_portable_error_output -error_output_stdout
      filter: grep -E "(PETSC ERROR)" | grep -E "(wrong order|Need to call|PCASMGetSubKSP\(\)|main\(\))"

TEST*/
