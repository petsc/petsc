static char help[] = "Tests repeated TSSolve() calls, optionally each followed by TSAdjointSolve(), with a saved TSTrajectory.\n\n";

/*
  The problem is

    u_t = -u,   u(0) = 1

  with one component of u on each MPI process. It is solved -num_solves times from t = 0 and step 0
  with the same TS, reusing the TSTrajectory created by TSSetSaveTrajectory(). Solve i runs to the final
  time T = 1 + i * max_time_increment, so a positive -max_time_increment makes each solve take more
  steps than the previous one. With -adjoint, each forward solve is followed by an adjoint solve for the
  cost J = sum_j u_j(T), whose gradient with respect to each u_j(0) is approximately exp(-T). With
  -adjoint_last, only the last one is. Every component of u(T) and of the gradient is the same, so the
  output does not depend on the number of processes or on the checkpointing schedule the trajectory uses.
*/

#include <petscts.h>

static PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec U, Vec F, void *ctx)
{
  PetscFunctionBeginUser;
  PetscCall(VecCopy(U, F));
  PetscCall(VecScale(F, -1.0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RHSJacobian(TS ts, PetscReal t, Vec U, Mat J, Mat P, void *ctx)
{
  PetscInt rstart;

  PetscFunctionBeginUser;
  PetscCall(MatGetOwnershipRange(P, &rstart, NULL));
  PetscCall(MatSetValue(P, rstart, rstart, -1.0, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  TS        ts;
  Vec       U, lambda;
  Mat       J;
  PetscInt  num_solves         = 2;
  PetscReal max_time_increment = 0.0;
  PetscBool adjoint = PETSC_FALSE, adjoint_last = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Repeated TSSolve() with trajectory options", NULL);
  PetscCall(PetscOptionsInt("-num_solves", "Number of forward solves", NULL, num_solves, &num_solves, NULL));
  PetscCall(PetscOptionsReal("-max_time_increment", "Increase of the final time from one solve to the next", NULL, max_time_increment, &max_time_increment, NULL));
  PetscCall(PetscOptionsBool("-adjoint", "Run TSAdjointSolve() after each TSSolve()", NULL, adjoint, &adjoint, NULL));
  PetscCall(PetscOptionsBool("-adjoint_last", "Run TSAdjointSolve() only after the last TSSolve()", NULL, adjoint_last, &adjoint_last, NULL));
  PetscOptionsEnd();

  PetscCall(VecCreateMPI(PETSC_COMM_WORLD, 1, PETSC_DETERMINE, &U));
  PetscCall(VecDuplicate(U, &lambda));
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, 1, 1, PETSC_DETERMINE, PETSC_DETERMINE, 1, NULL, 0, NULL, &J));

  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetType(ts, TSRK));
  PetscCall(TSSetRHSFunction(ts, NULL, RHSFunction, NULL));
  PetscCall(TSSetRHSJacobian(ts, J, J, RHSJacobian, NULL));
  PetscCall(TSSetMaxTime(ts, 1.0));
  PetscCall(TSSetTimeStep(ts, 0.1));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetSaveTrajectory(ts));
  if (adjoint || adjoint_last) PetscCall(TSSetCostGradients(ts, 1, &lambda, NULL));
  PetscCall(TSSetFromOptions(ts));

  for (PetscInt i = 0; i < num_solves; i++) {
    PetscReal u;

    PetscCall(TSSetTime(ts, 0.0));
    PetscCall(TSSetStepNumber(ts, 0));
    PetscCall(TSSetTimeStep(ts, 0.1));
    PetscCall(TSSetMaxTime(ts, 1.0 + i * max_time_increment));
    PetscCall(VecSet(U, 1.0));
    PetscCall(TSSolve(ts, U));
    PetscCall(VecMax(U, NULL, &u));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Solve %" PetscInt_FMT ": u(T) = %.6f\n", i, (double)u));
    if (adjoint || (adjoint_last && i == num_solves - 1)) {
      PetscReal l;

      PetscCall(VecSet(lambda, 1.0));
      PetscCall(TSAdjointSolve(ts));
      PetscCall(VecMax(lambda, NULL, &l));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Solve %" PetscInt_FMT ": dJ/du(0) = %.6f\n", i, (double)l));
    }
  }

  PetscCall(TSDestroy(&ts));
  PetscCall(MatDestroy(&J));
  PetscCall(VecDestroy(&lambda));
  PetscCall(VecDestroy(&U));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

    testset:
      requires: !single !complex
      nsize: {{1 2}}
      output_file: output/ex37.out

      test:
        suffix: memory
        args: -ts_trajectory_type memory

      test:
        suffix: basic
        args: -ts_trajectory_type basic

      # without TSHistory, the second solve reaches the checkpoints left on the memory stack by the first
      test:
        suffix: memory_no_history
        args: -ts_trajectory_type memory -ts_trajectory_use_history 0

    testset:
      requires: !single !complex
      nsize: {{1 2}}
      output_file: output/ex37_adjoint.out
      args: -adjoint

      test:
        suffix: memory_adjoint
        args: -ts_trajectory_type memory

      test:
        suffix: basic_adjoint
        args: -ts_trajectory_type basic

    # checkpointing schedules that store only some steps, which require a fixed time step
    testset:
      requires: !single !complex
      nsize: {{1 2}}
      output_file: output/ex37_schedule_adjoint.out
      args: -ts_trajectory_type memory -ts_adapt_type none -adjoint

      test:
        suffix: two_level_norevolve_adjoint
        args: -ts_trajectory_stride 5

      test:
        suffix: revolve_offline_adjoint
        requires: revolve
        args: -ts_trajectory_max_cps_ram 3 -ts_trajectory_memory_type revolve

      test:
        suffix: revolve_online_adjoint
        requires: revolve
        args: -ts_trajectory_max_cps_ram 4 -ts_trajectory_revolve_online

      test:
        suffix: revolve_multistage_adjoint
        requires: revolve
        args: -ts_trajectory_max_cps_ram 3 -ts_trajectory_max_cps_disk 8 -ts_trajectory_memory_type revolve

      test:
        suffix: two_level_revolve_adjoint
        requires: revolve
        args: -ts_trajectory_max_cps_ram 3 -ts_trajectory_stride 5

      test:
        suffix: two_level_two_revolve_adjoint
        requires: revolve
        args: -ts_trajectory_max_cps_ram 3 -ts_trajectory_max_cps_disk 8 -ts_trajectory_stride 5

      test:
        suffix: cams_adjoint
        requires: cams
        args: -ts_trajectory_max_units_ram 5 -ts_trajectory_memory_type cams

    # the second forward solve must discard the checkpoints and schedule of the first
    testset:
      requires: !single !complex
      nsize: {{1 2}}
      output_file: output/ex37_schedule_adjoint_last.out
      args: -ts_trajectory_type memory -ts_adapt_type none -adjoint_last

      test:
        suffix: two_level_norevolve_adjoint_last
        args: -ts_trajectory_stride 5

      test:
        suffix: revolve_offline_adjoint_last
        requires: revolve
        args: -ts_trajectory_max_cps_ram 3 -ts_trajectory_memory_type revolve -ts_trajectory_solution_only

      test:
        suffix: revolve_online_adjoint_last
        requires: revolve
        args: -ts_trajectory_max_cps_ram 4 -ts_trajectory_revolve_online

      test:
        suffix: revolve_multistage_adjoint_last
        requires: revolve
        args: -ts_trajectory_max_cps_ram 3 -ts_trajectory_max_cps_disk 8 -ts_trajectory_memory_type revolve

      test:
        suffix: two_level_revolve_adjoint_last
        requires: revolve
        args: -ts_trajectory_max_cps_ram 3 -ts_trajectory_stride 5

      test:
        suffix: two_level_two_revolve_adjoint_last
        requires: revolve
        args: -ts_trajectory_max_cps_ram 3 -ts_trajectory_max_cps_disk 8 -ts_trajectory_stride 5

      test:
        suffix: cams_adjoint_last
        requires: cams
        args: -ts_trajectory_max_units_ram 5 -ts_trajectory_memory_type cams

    # the second solve takes more steps than the first, so it needs a larger checkpoint stack
    testset:
      requires: !single !complex
      nsize: {{1 2}}
      args: -ts_trajectory_type memory -ts_adapt_type none -max_time_increment 1

      test:
        suffix: grow_adjoint
        output_file: output/ex37_grow_adjoint.out
        args: -adjoint

      test:
        suffix: grow_adjoint_last
        output_file: output/ex37_grow_adjoint_last.out
        args: -adjoint_last

TEST*/
