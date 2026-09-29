static char help[] = "Tests BDDC coarse solver reuse when the coarse MPI communicator changes.\n\n";

#include <petsc/private/pcbddcimpl.h>

int main(int argc, char **args)
{
  Mat                    A, local, coarse;
  MatNullSpace           nsp;
  ISLocalToGlobalMapping map;
  KSP                    ksp;
  PC                     pc, coarsepc;
  PC_BDDC               *bddc;
  Vec                    modes[3], x, b, exact;
  PetscScalar           *values;
  PetscInt               indices[] = {0, 1, 2, 3, 4, 5, 6, 7};
  PetscInt               counts[]  = {2, 2, 1, 1, 3, 3, 1, 2};
  PetscInt               nlocal, start, end, participating, previous = -1, empty_ranks = 0;
  PetscMPIInt            rank, size, active, comparison;
  PetscObjectId          id, previous_id = 0;
  PetscBool              reset = PETSC_FALSE, changed = PETSC_FALSE, redundant;
  PetscReal              error;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-empty_ranks", &empty_ranks, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-reset", &reset, NULL));
  PetscCall(PetscMPIIntCast(size - empty_ranks, &active));
  PetscCheck(active >= 2, PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "At least two nonempty subdomains are required");
  nlocal = rank < active ? 8 : 0;
  PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD, 1, nlocal, indices, PETSC_COPY_VALUES, &map));
  PetscCall(MatCreateIS(PETSC_COMM_WORLD, 1, PETSC_DECIDE, PETSC_DECIDE, 8, 8, map, map, &A));
  PetscCall(ISLocalToGlobalMappingDestroy(&map));
  PetscCall(MatISSetPreallocation(A, 8, NULL, 8, NULL));
  PetscCall(MatISGetLocalMat(A, &local));
  for (PetscInt i = 0; i < nlocal; i++)
    for (PetscInt j = 0; j < nlocal; j++) PetscCall(MatSetValue(local, i, j, i == j ? 2.0 : 0.1, INSERT_VALUES));
  PetscCall(MatISRestoreLocalMat(A, &local));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatSetOption(A, MAT_SPD, PETSC_TRUE));
  PetscCall(MatCreateVecs(A, &x, &b));
  PetscCall(VecDuplicate(x, &exact));
  PetscCall(VecGetOwnershipRange(x, &start, &end));
  for (PetscInt k = 0; k < 3; k++) {
    PetscCall(VecDuplicate(x, &modes[k]));
    PetscCall(VecGetArray(modes[k], &values));
    for (PetscInt i = start; i < end; i++) values[i - start] = (i == 2 * k ? 1.0 : 0.0) - (i == 2 * k + 1 ? 1.0 : 0.0);
    PetscCall(VecRestoreArray(modes[k], &values));
    PetscCall(VecNormalize(modes[k], NULL));
  }
  PetscCall(VecSet(exact, 1.0));
  PetscCall(VecAXPY(exact, 0.5, modes[1]));
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetType(ksp, KSPCG));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCBDDC));
  PetscCall(KSPSetFromOptions(ksp));
  bddc = (PC_BDDC *)pc->data;
  for (PetscInt step = 0; step < (PetscInt)PETSC_STATIC_ARRAY_LENGTH(counts); step++) {
    if (reset && step) PetscCall(PCReset(pc));
    PetscCall(MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_FALSE, counts[step], modes, &nsp));
    PetscCall(MatSetNearNullSpace(A, nsp));
    PetscCall(MatNullSpaceDestroy(&nsp));
    PetscCall(MatScale(A, 1.01));
    PetscCall(MatMult(A, exact, b));
    PetscCall(KSPSetOperators(ksp, A, A));
    PetscCall(KSPSolve(ksp, b, x));
    PetscCall(VecAXPY(x, -1.0, exact));
    PetscCall(VecNorm(x, NORM_INFINITY, &error));
    PetscCheck(error < PETSC_SMALL, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Solution error after changing the coarse space: %g", (double)error);
    PetscCheck(bddc->coarse_size == counts[step], PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Unexpected coarse size: %" PetscInt_FMT, bddc->coarse_size);
    participating = (PetscInt)!!bddc->coarse_ksp;
    PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &participating, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD));
    if (step && participating != previous) changed = PETSC_TRUE;
    previous = participating;
    if (bddc->coarse_ksp) {
      PetscCall(PetscObjectGetId((PetscObject)bddc->coarse_ksp, &id));
      PetscCheck(reset || !step || counts[step] != counts[step - 1] || id == previous_id, PETSC_COMM_SELF, PETSC_ERR_PLIB, "The coarse KSP was replaced despite unchanged participation");
      previous_id = id;
      PetscCall(KSPGetOperators(bddc->coarse_ksp, &coarse, NULL));
      PetscCallMPI(MPI_Comm_compare(PetscObjectComm((PetscObject)bddc->coarse_ksp), PetscObjectComm((PetscObject)coarse), &comparison));
      PetscCheck(comparison == MPI_IDENT || comparison == MPI_CONGRUENT, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Coarse KSP and matrix communicators differ");
      PetscCall(KSPGetPC(bddc->coarse_ksp, &coarsepc));
      PetscCall(PetscObjectTypeCompare((PetscObject)coarsepc, PCREDUNDANT, &redundant));
      PetscCheck(redundant, PETSC_COMM_SELF, PETSC_ERR_PLIB, "The terminal coarse PC must remain PCREDUNDANT");
    }
  }
  PetscCheck(changed, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "The test did not change coarse participation");
  PetscCall(KSPDestroy(&ksp));
  for (PetscInt k = 0; k < 3; k++) PetscCall(VecDestroy(&modes[k]));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&exact));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    requires: double
    output_file: output/empty.out
    args: -ksp_error_if_not_converged -ksp_rtol 1e-12
    test:
      suffix: resize
      nsize: {{2 3}}
      args: -pc_bddc_use_change_of_basis {{0 1}} -reset {{0 1}}
    test:
      suffix: empty
      nsize: 4
      args: -empty_ranks 2 -pc_bddc_coarse_eqs_per_proc 2 -pc_bddc_aggregator_0_mat_partitioning_type average -reset {{0 1}}
    test:
      suffix: deluxe
      nsize: 3
      args: -pc_bddc_use_deluxe_scaling -pc_bddc_use_change_of_basis {{0 1}}

TEST*/
