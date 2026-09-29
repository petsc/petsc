static char help[] = "Tests no-net-flux constraints combined with a supplied near-nullspace.\n\n";

#include <petsc/private/pcbddcimpl.h>

static PetscErrorCode CheckConstraint(Mat C, Vec v)
{
  Vec       c, projected;
  PetscReal error, norm;

  PetscFunctionBeginUser;
  PetscCall(MatCreateVecs(C, &projected, &c));
  PetscCall(MatMult(C, v, c));
  PetscCall(MatMultTranspose(C, c, projected));
  PetscCall(VecAXPY(projected, -1.0, v));
  PetscCall(VecNorm(projected, NORM_2, &error));
  PetscCall(VecNorm(v, NORM_2, &norm));
  PetscCheck(error < PETSC_SMALL * norm, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Constraint span does not contain the supplied vector: relative error %g", (double)(error / norm));
  PetscCall(VecDestroy(&projected));
  PetscCall(VecDestroy(&c));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **args)
{
  Mat                    A, B, local;
  MatNullSpace           nsp = NULL, attached;
  ISLocalToGlobalMapping map, pmap;
  KSP                    ksp = NULL;
  PC                     pc;
  PC_BDDC               *bddc;
  Vec                    modes[2], x, b, exact, v;
  PetscScalar           *values;
  PetscInt               indices[] = {0, 1, 2, 3, 4, 5};
  PetscInt               nlocal, npressure, pressure, start, end, nmodes = 2, nconstraints, expected = 0, offset = 0;
  PetscMPIInt            rank, size, active;
  PetscBool              constant = PETSC_FALSE, empty_rank = PETSC_FALSE, dependent = PETSC_FALSE, has_const = PETSC_FALSE, reset = PETSC_FALSE;
  PetscReal              error;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-user_modes", &nmodes, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-constant", &constant, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-empty_rank", &empty_rank, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-reset", &reset, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-dependent", &dependent, NULL));
  PetscCheck(nmodes >= 0 && nmodes <= 2, PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "Use zero, one, or two user modes");
  if (dependent) {
    nmodes   = 1;
    constant = PETSC_FALSE;
  }
  active    = size - (empty_rank ? 1 : 0);
  nlocal    = rank < active ? 6 : 0;
  npressure = rank < active ? 1 : 0;
  pressure  = rank;
  PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD, 1, nlocal, indices, PETSC_COPY_VALUES, &map));
  PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD, 1, npressure, &pressure, PETSC_COPY_VALUES, &pmap));
  PetscCall(MatCreateIS(PETSC_COMM_WORLD, 1, PETSC_DECIDE, PETSC_DECIDE, 6, 6, map, map, &A));
  PetscCall(MatISSetPreallocation(A, 6, NULL, 6, NULL));
  PetscCall(MatISGetLocalMat(A, &local));
  for (PetscInt i = 0; i < nlocal; i++)
    for (PetscInt j = 0; j < nlocal; j++) PetscCall(MatSetValue(local, i, j, i == j ? 2.0 : 0.1, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatSetOption(A, MAT_SPD, PETSC_TRUE));
  PetscCall(MatCreateIS(PETSC_COMM_WORLD, 1, npressure, PETSC_DECIDE, active, 6, pmap, map, &B));
  PetscCall(MatISSetPreallocation(B, 6, NULL, 6, NULL));
  PetscCall(MatISGetLocalMat(B, &local));
  for (PetscInt i = 0; i < nlocal; i++) PetscCall(MatSetValue(local, 0, i, (rank ? -1.0 : active - 1.0) * (i + 1), INSERT_VALUES));
  PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));

  PetscCall(MatCreateVecs(A, &modes[0], &b));
  PetscCall(VecDuplicate(modes[0], &modes[1]));
  PetscCall(VecDuplicate(modes[0], &x));
  PetscCall(VecDuplicate(modes[0], &exact));
  PetscCall(VecGetOwnershipRange(modes[0], &start, &end));
  for (PetscInt k = 0; k < 2; k++) {
    PetscCall(VecGetArray(modes[k], &values));
    for (PetscInt i = start; i < end; i++) values[i - start] = (i == 2 * k ? 1.0 : 0.0) - (i == 2 * k + 1 ? 1.0 : 0.0);
    if (dependent && !k)
      for (PetscInt i = start; i < end; i++) values[i - start] = i + 1;
    PetscCall(VecRestoreArray(modes[k], &values));
    PetscCall(VecNormalize(modes[k], NULL));
  }
  PetscCall(VecSet(exact, 1.0));
  PetscCall(VecAXPY(exact, 0.5, modes[1]));
  for (PetscInt step = 0; step < 5; step++) {
    // Rebuild the interface, replace the user modes, remove them, and reset or recreate the solver.
    if (step == 4 && reset) {
      PetscCall(PCReset(pc));
      PetscCall(PCBDDCSetDivergenceMat(pc, B, PETSC_FALSE, NULL));
    }
    if (step == 0 || (step == 4 && !reset)) {
      PetscCall(KSPDestroy(&ksp));
      PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
      PetscCall(KSPSetType(ksp, KSPCG));
      PetscCall(KSPSetOperators(ksp, A, A));
      PetscCall(KSPGetPC(ksp, &pc));
      PetscCall(PCSetType(pc, PCBDDC));
      PetscCall(PCBDDCSetDivergenceMat(pc, B, PETSC_FALSE, NULL));
      PetscCall(KSPSetFromOptions(ksp));
    }
    if (step != 1) {
      PetscCall(MatNullSpaceDestroy(&nsp));
      has_const = step == 2 || step == 3 ? PETSC_FALSE : constant;
      offset    = step == 2 ? 1 : 0;
      expected  = step == 2 ? 1 : step == 3 ? 0 : nmodes;
      if (has_const || expected) PetscCall(MatNullSpaceCreate(PETSC_COMM_WORLD, has_const, expected, modes + offset, &nsp));
      PetscCall(MatSetNearNullSpace(A, nsp));
    }
    bddc = (PC_BDDC *)pc->data;
    if (step == 1) bddc->recompute_topography = PETSC_TRUE;
    PetscCall(MatScale(A, 1.01));
    PetscCall(MatMult(A, exact, b));
    PetscCall(KSPSetOperators(ksp, A, A));
    PetscCall(KSPSolve(ksp, b, x));
    PetscCall(MatGetNearNullSpace(A, &attached));
    PetscCheck(attached == nsp, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "The supplied near-nullspace was replaced");
    PetscCall(VecAXPY(x, -1.0, exact));
    PetscCall(VecNorm(x, NORM_INFINITY, &error));
    PetscCheck(error < PETSC_SMALL, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Solution error %g", (double)error);

    PetscCall(MatGetSize(bddc->ConstraintMatrix, &nconstraints, NULL));
    PetscCheck(nconstraints == (active > 1 && nlocal ? expected + (has_const ? 1 : 0) + (dependent && step != 2 && step != 3 ? 0 : 1) : 0), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected number of constraints: %" PetscInt_FMT, nconstraints);
    if (active > 1 && nlocal) {
      PetscCall(VecCreateSeq(PETSC_COMM_SELF, nlocal, &v));
      PetscCall(VecGetArray(v, &values));
      for (PetscInt i = 0; i < nlocal; i++) values[i] = i + 1;
      PetscCall(VecRestoreArray(v, &values));
      PetscCall(CheckConstraint(bddc->ConstraintMatrix, v));
      if (has_const) {
        PetscCall(VecSet(v, 1.0));
        PetscCall(CheckConstraint(bddc->ConstraintMatrix, v));
      }
      for (PetscInt k = offset; k < offset + expected; k++) {
        PetscCall(VecGetArray(v, &values));
        for (PetscInt i = 0; i < nlocal; i++) values[i] = (i == 2 * k ? 1.0 : 0.0) - (i == 2 * k + 1 ? 1.0 : 0.0);
        if (dependent && !k)
          for (PetscInt i = 0; i < nlocal; i++) values[i] = i + 1;
        PetscCall(VecRestoreArray(v, &values));
        PetscCall(CheckConstraint(bddc->ConstraintMatrix, v));
      }
      PetscCall(VecDestroy(&v));
    }
  }

  // Exercise the constant flag of the flux space independently of its computation flag.
  PetscCall(MatNullSpaceDestroy(&bddc->nonetflux));
  PetscCall(MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, NULL, &bddc->nonetflux));
  for (PetscInt compute = 0; compute < 2; compute++) {
    bddc->compute_nonetflux = (PetscBool)compute;
    PetscCall(MatNullSpaceDestroy(&nsp));
    PetscCall(MatNullSpaceCreate(PETSC_COMM_WORLD, constant, nmodes, modes, &nsp));
    PetscCall(MatSetNearNullSpace(A, nsp));
    PetscCall(MatScale(A, 1.01));
    PetscCall(MatMult(A, exact, b));
    PetscCall(KSPSetOperators(ksp, A, A));
    PetscCall(KSPSolve(ksp, b, x));
    PetscCall(MatGetSize(bddc->ConstraintMatrix, &nconstraints, NULL));
    PetscCheck(nconstraints == (active > 1 && nlocal ? nmodes + 1 : 0), PETSC_COMM_SELF, PETSC_ERR_PLIB, "The constant from the flux space was lost: %" PetscInt_FMT " constraints", nconstraints);
    if (active > 1 && nlocal) {
      PetscCall(VecCreateSeq(PETSC_COMM_SELF, nlocal, &v));
      PetscCall(VecSet(v, 1.0));
      PetscCall(CheckConstraint(bddc->ConstraintMatrix, v));
      PetscCall(VecDestroy(&v));
    }
  }

  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatNullSpaceDestroy(&nsp));
  PetscCall(VecDestroy(&modes[0]));
  PetscCall(VecDestroy(&modes[1]));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&exact));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(ISLocalToGlobalMappingDestroy(&map));
  PetscCall(ISLocalToGlobalMappingDestroy(&pmap));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    requires: double
    output_file: output/empty.out
    args: -pc_bddc_use_change_of_basis 0 -pc_bddc_coarsening_ratio 1 -ksp_error_if_not_converged -ksp_rtol 1e-12
    test:
      suffix: modes
      nsize: {{1 2 3}}
      args: -user_modes {{0 1 2}} -constant {{0 1}}
    test:
      suffix: dependent
      nsize: 2
      args: -dependent
    test:
      suffix: empty
      nsize: 3
      args: -empty_rank -constant {{0 1}}
    test:
      suffix: reset
      nsize: {{1 2 3}}
      args: -reset -constant
    test:
      suffix: coarse_resize
      nsize: {{2 3}}
      args: -pc_bddc_coarsening_ratio 8 -reset {{0 1}}

TEST*/
