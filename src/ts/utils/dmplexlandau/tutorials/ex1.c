static char help[] = "Landau collision operator driver\n\n";

#include <petscts.h>
#include <petsclandau.h>

int main(int argc, char **argv)
{
  DM             dm;
  Vec            X,X_0;
  PetscErrorCode ierr;
  PetscInt       dim=2;
  TS             ts;
  Mat            J;
  SNES           snes;
  KSP            ksp;
  PC             pc;
  SNESLineSearch linesearch;
  PetscReal      time;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL, "-dim", &dim, NULL));
  /* Create a mesh */
  CHKERRQ(DMPlexLandauCreateVelocitySpace(PETSC_COMM_SELF, dim, "", &X, &J, &dm));
  CHKERRQ(DMSetUp(dm));
  CHKERRQ(VecDuplicate(X,&X_0));
  CHKERRQ(VecCopy(X,X_0));
  CHKERRQ(DMPlexLandauPrintNorms(X,0));
  CHKERRQ(DMSetOutputSequenceNumber(dm, 0, 0.0));
  CHKERRQ(DMViewFromOptions(dm,NULL,"-dm_view"));
  CHKERRQ(VecViewFromOptions(X,NULL,"-vec_view"));
  /* Create timestepping solver context */
  CHKERRQ(TSCreate(PETSC_COMM_SELF,&ts));
  CHKERRQ(TSSetDM(ts,dm));
  CHKERRQ(TSGetSNES(ts,&snes));
  CHKERRQ(SNESGetLineSearch(snes,&linesearch));
  CHKERRQ(SNESLineSearchSetType(linesearch,SNESLINESEARCHBASIC));
  CHKERRQ(TSSetIFunction(ts,NULL,DMPlexLandauIFunction,NULL));
  CHKERRQ(TSSetIJacobian(ts,J,J,DMPlexLandauIJacobian,NULL));
  CHKERRQ(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER));
  CHKERRQ(SNESGetKSP(snes,&ksp));
  CHKERRQ(KSPGetPC(ksp,&pc));
  CHKERRQ(TSSetFromOptions(ts));
  CHKERRQ(TSSetSolution(ts,X));
  CHKERRQ(TSSolve(ts,X));
  CHKERRQ(DMPlexLandauPrintNorms(X,1));
  CHKERRQ(TSGetTime(ts, &time));
  CHKERRQ(DMSetOutputSequenceNumber(dm, 1, time));
  CHKERRQ(VecViewFromOptions(X,NULL,"-vec_view"));
  CHKERRQ(VecAXPY(X,-1,X_0));
  /* clean up */
  CHKERRQ(DMPlexLandauDestroyVelocitySpace(&dm));
  CHKERRQ(TSDestroy(&ts));
  CHKERRQ(VecDestroy(&X));
  CHKERRQ(VecDestroy(&X_0));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST
  testset:
    requires: p4est !complex double
    output_file: output/ex1_0.out
    args: -dm_landau_num_species_grid 1,2 -petscspace_degree 3 -petscspace_poly_tensor 1 -dm_landau_type p4est -dm_landau_ion_masses 2,4 -dm_landau_ion_charges 1,18 -dm_landau_thermal_temps 5,5,.5 -dm_landau_n 1.00018,1,1e-5 -dm_landau_n_0 1e20 -ts_monitor -snes_rtol 1.e-14 -snes_stol 1.e-14 -snes_monitor -snes_converged_reason -ts_type arkimex -ts_arkimex_type 1bee -ts_max_snes_failures -1 -ts_rtol 1e-1 -ts_dt 1.e-1 -ts_max_time 1 -ts_adapt_clip .5,1.25 -ts_adapt_scale_solve_failed 0.75 -ts_adapt_time_step_increase_delay 5 -ts_max_steps 1 -pc_type lu -ksp_type preonly -dm_landau_amr_levels_max 2,1
    test:
      suffix: cpu
      args: -dm_landau_device_type cpu
    test:
      suffix: kokkos
      requires: kokkos_kernels
      args: -dm_landau_device_type kokkos -dm_mat_type aijkokkos -dm_vec_type kokkos
    test:
      suffix: cuda
      requires: cuda
      args: -dm_landau_device_type cuda -dm_mat_type aijcusparse -dm_vec_type cuda -mat_cusparse_use_cpu_solve

TEST*/
