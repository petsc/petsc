static char help[] = "Tests BDDC reuse after PCReset() and KSPReset().\n\n";

#include <petsc/private/pcbddcimpl.h>

int main(int argc, char **args)
{
  Mat                    A, local;
  MatNullSpace           nsp;
  ISLocalToGlobalMapping map;
  KSP                    ksp;
  PC                     pc;
  PC_BDDC               *bddc;
  Vec                    modes[2], x, b, exact, scaling;
  PetscInt              *indices, *xadj, *adjncy;
  PetscInt               n, nlocal, start, end, nconstraints;
  PetscMPIInt            rank, size, active;
  PetscBool              empty_rank = PETSC_FALSE, user_graph = PETSC_FALSE, use_nnsp, change, deluxe, stiffness;
  PetscScalar            factor, weight;
  PetscScalar           *values;
  PetscReal              error;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-empty_rank", &empty_rank, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-user_graph", &user_graph, NULL));
  active = size - (empty_rank ? 1 : 0);
  PetscCheck(active > 0, PETSC_COMM_WORLD, PETSC_ERR_USER_INPUT, "At least one nonempty subdomain is required");
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetType(ksp, KSPCG));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCBDDC));
  PetscCall(PCBDDCSetCoarseningRatio(pc, 1));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(PCISSetSubdomainScalingFactor(pc, rank + 1.0));
  PetscCall(PCISSetUseStiffnessScaling(pc, PETSC_TRUE));
  bddc = (PC_BDDC *)pc->data;
  for (PetscInt step = 0; step < 3; step++) {
    use_nnsp  = bddc->use_nnsp;
    change    = bddc->use_change_of_basis;
    deluxe    = bddc->use_deluxe_scaling;
    stiffness = bddc->pcis.use_stiffness_scaling;
    factor    = bddc->pcis.scaling_factor;
    // Reset before the first setup and twice between systems of different sizes.
    PetscCall(PCReset(pc));
    PetscCall(KSPReset(ksp));
    PetscCheck(bddc->use_nnsp == use_nnsp && bddc->use_change_of_basis == change && bddc->use_deluxe_scaling == deluxe && bddc->coarsening_ratio == 1, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "BDDC options changed during reset");
    PetscCheck(bddc->pcis.use_stiffness_scaling == stiffness && bddc->pcis.scaling_factor == factor, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Scaling options changed during reset");

    n      = 6 + 2 * step;
    nlocal = rank < active ? n : 0;
    PetscCall(PetscMalloc1(nlocal, &indices));
    for (PetscInt i = 0; i < nlocal; i++) indices[i] = i;
    PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_WORLD, 1, nlocal, indices, PETSC_OWN_POINTER, &map));
    PetscCall(MatCreateIS(PETSC_COMM_WORLD, 1, PETSC_DECIDE, PETSC_DECIDE, n, n, map, map, &A));
    PetscCall(ISLocalToGlobalMappingDestroy(&map));
    PetscCall(MatISSetPreallocation(A, n, NULL, n, NULL));
    PetscCall(MatISGetLocalMat(A, &local));
    for (PetscInt i = 0; i < nlocal; i++)
      for (PetscInt j = 0; j < nlocal; j++) PetscCall(MatSetValue(local, i, j, (rank + 1.0) * (i == j ? 2.0 : 0.1), INSERT_VALUES));
    PetscCall(MatISRestoreLocalMat(A, &local));
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatSetOption(A, MAT_SPD, PETSC_TRUE));
    if (user_graph) {
      PetscCall(PetscMalloc1(nlocal + 1, &xadj));
      PetscCall(PetscMalloc1(nlocal * nlocal, &adjncy));
      for (PetscInt i = 0; i <= nlocal; i++) xadj[i] = i * nlocal;
      for (PetscInt i = 0; i < nlocal * nlocal; i++) adjncy[i] = i % nlocal;
      PetscCall(PCBDDCSetLocalAdjacencyGraph(pc, nlocal, xadj, adjncy, PETSC_OWN_POINTER));
    }
    PetscCall(PCISSetUseStiffnessScaling(pc, (PetscBool)(step == 0)));
    PetscCall(PCISSetSubdomainScalingFactor(pc, rank + step + 1.0));
    if (step == 2) {
      PetscCall(VecCreateSeq(PETSC_COMM_SELF, nlocal, &scaling));
      PetscCall(VecSet(scaling, rank + 5.0));
      PetscCall(PCISSetSubdomainDiagonalScaling(pc, scaling));
      PetscCall(VecDestroy(&scaling));
    }

    PetscCall(MatCreateVecs(A, &modes[0], &b));
    PetscCall(VecDuplicate(modes[0], &modes[1]));
    PetscCall(VecDuplicate(modes[0], &x));
    PetscCall(VecDuplicate(modes[0], &exact));
    PetscCall(VecGetOwnershipRange(modes[0], &start, &end));
    for (PetscInt k = 0; k < 2; k++) {
      PetscCall(VecGetArray(modes[k], &values));
      for (PetscInt i = start; i < end; i++) values[i - start] = (i == 2 * k ? 1.0 : 0.0) - (i == 2 * k + 1 ? 1.0 : 0.0);
      PetscCall(VecRestoreArray(modes[k], &values));
      PetscCall(VecNormalize(modes[k], NULL));
    }
    PetscCall(MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_FALSE, 2, modes, &nsp));
    PetscCall(MatSetNearNullSpace(A, nsp));
    PetscCall(MatNullSpaceDestroy(&nsp));
    PetscCall(VecSet(exact, 1.0));
    PetscCall(VecAXPY(exact, 0.5, modes[1]));
    PetscCall(MatMult(A, exact, b));
    PetscCall(KSPSetOperators(ksp, A, A));
    PetscCall(KSPSolve(ksp, b, x));
    PetscCall(VecAXPY(x, -1.0, exact));
    PetscCall(VecNorm(x, NORM_INFINITY, &error));
    PetscCheck(error < PETSC_SMALL, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Solution error after reset: %g", (double)error);
    PetscCall(MatGetSize(bddc->ConstraintMatrix, &nconstraints, NULL));
    PetscCheck(nconstraints == (nlocal && active > 1 ? (use_nnsp ? 2 : 1) : 0), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected number of constraints after reset: %" PetscInt_FMT, nconstraints);
    factor = step == 2 ? 5.0 : step + 1.0;
    weight = (rank + factor) / (active * (active + 2.0 * factor - 1.0) / 2.0);
    PetscCall(VecDuplicate(bddc->pcis.D, &scaling));
    PetscCall(VecCopy(bddc->pcis.D, scaling));
    PetscCall(VecShift(scaling, -weight));
    PetscCall(VecNorm(scaling, NORM_INFINITY, &error));
    PetscCheck(error < PETSC_SMALL, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Scaling error after reset: %g", (double)error);
    PetscCall(VecDestroy(&scaling));
    PetscCall(VecDestroy(&modes[0]));
    PetscCall(VecDestroy(&modes[1]));
    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroy(&b));
    PetscCall(VecDestroy(&exact));
    PetscCall(MatDestroy(&A));
  }
  // Changing type must remove the PCIS callbacks that access the BDDC context.
  PetscCall(PCSetType(pc, PCNONE));
  PetscCall(PCISSetSubdomainScalingFactor(pc, 2.0));
  PetscCall(PCISSetUseStiffnessScaling(pc, PETSC_TRUE));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    requires: double
    output_file: output/empty.out
    args: -ksp_error_if_not_converged -ksp_rtol 1e-12
    test:
      suffix: reset
      nsize: {{1 2 3}}
      args: -pc_bddc_use_change_of_basis {{0 1}} -user_graph {{0 1}}
    test:
      suffix: no_nnsp
      nsize: 2
      args: -pc_bddc_use_nnsp 0
    test:
      suffix: deluxe
      nsize: 2
      args: -pc_bddc_use_deluxe_scaling -pc_bddc_use_change_of_basis {{0 1}}
    test:
      suffix: empty
      nsize: 3
      args: -empty_rank -user_graph

TEST*/
