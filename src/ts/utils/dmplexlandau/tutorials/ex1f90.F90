! test phase space (Maxwellian) mesh construction (serial)
!
!
!
! Contributed by Mark Adams
program DMPlexTestLandauInterface
  use petscts
  use petscdmplex
#include <petsc/finclude/petscts.h>
#include <petsc/finclude/petscdmplex.h>
  implicit none
  external DMPlexLandauIFunction
  external DMPlexLandauIJacobian
  DM             dm
  PetscInt       dim
  PetscInt       ii
  PetscErrorCode ierr
  TS             ts
  Vec            X,X_0
  Mat            J
  SNES           snes
  KSP            ksp
  PC             pc
  SNESLineSearch linesearch
  PetscReal      mone
  PetscScalar    scalar

  PetscCallA(PetscInitialize(ierr))

  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  !  Create mesh (DM), read in parameters, create and add f_0 (X)
  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  dim = 2
  PetscCallA(DMPlexLandauCreateVelocitySpace(PETSC_COMM_WORLD, dim, '', X, J, dm, ierr))
  PetscCallA(DMSetUp(dm,ierr))
  PetscCallA(VecDuplicate(X,X_0,ierr))
  PetscCallA(VecCopy(X,X_0,ierr))
  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  !  View
  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  ii = 0
  PetscCallA(DMPlexLandauPrintNorms(X,ii,ierr))
  mone = 0;
  PetscCallA(DMSetOutputSequenceNumber(dm, ii, mone, ierr))
  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  !    Create timestepping solver context
  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  PetscCallA(TSCreate(PETSC_COMM_SELF,ts,ierr))
  PetscCallA(TSSetOptionsPrefix(ts, 'ex1_', ierr)) ! should get this from the dm or give it to the dm
  PetscCallA(TSSetDM(ts,dm,ierr))
  PetscCallA(TSGetSNES(ts,snes,ierr))
  PetscCallA(SNESSetOptionsPrefix(snes, 'ex1_', ierr)) ! should get this from the dm or give it to the dm
  PetscCallA(SNESGetLineSearch(snes,linesearch,ierr))
  PetscCallA(SNESLineSearchSetType(linesearch,SNESLINESEARCHBASIC,ierr))
  PetscCallA(TSSetIFunction(ts,PETSC_NULL_VEC,DMPlexLandauIFunction,PETSC_NULL_VEC,ierr))
  PetscCallA(TSSetIJacobian(ts,J,J,DMPlexLandauIJacobian,PETSC_NULL_VEC,ierr))
  PetscCallA(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER,ierr))

  PetscCallA(SNESGetKSP(snes,ksp,ierr))
  PetscCallA(KSPSetOptionsPrefix(ksp, 'ex1_', ierr)) ! should get this from the dm or give it to the dm
  PetscCallA(KSPGetPC(ksp,pc,ierr))
  PetscCallA(PCSetOptionsPrefix(pc, 'ex1_', ierr)) ! should get this from the dm or give it to the dm

  PetscCallA(TSSetFromOptions(ts,ierr))
  PetscCallA(TSSetSolution(ts,X,ierr))
  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  !  Solve nonlinear system
  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  PetscCallA(TSSolve(ts,X,ierr))
  ii = 1
  PetscCallA(DMPlexLandauPrintNorms(X,ii,ierr))
  PetscCallA(TSGetTime(ts, mone, ierr))
  PetscCallA(DMSetOutputSequenceNumber(dm, ii, mone, ierr))
  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  !  remove f_0
  ! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  scalar = -1.
  PetscCallA(VecAXPY(X,scalar,X_0,ierr))
  PetscCallA(DMPlexLandauDestroyVelocitySpace(dm, ierr))
  PetscCallA(TSDestroy(ts, ierr))
  PetscCallA(VecDestroy(X, ierr))
  PetscCallA(VecDestroy(X_0, ierr))
  PetscCallA(PetscFinalize(ierr))
end program DMPlexTestLandauInterface

!/*TEST
!  build:
!    requires: defined(PETSC_USING_F90FREEFORM) defined(PETSC_USE_DMLANDAU_2D)
!
!  test:
!    suffix: 0
!    requires: p4est !complex  !kokkos_kernels !cuda
!    args: -dm_landau_num_species_grid 1,2 -petscspace_degree 3 -petscspace_poly_tensor 1 -dm_landau_type p4est -dm_landau_ion_masses 2,4 -dm_landau_ion_charges 1,18 -dm_landau_thermal_temps 5,5,.5 -dm_landau_n 1.00018,1,1e-5 -dm_landau_n_0 1e20 -ex1_ts_monitor -ex1_snes_rtol 1.e-14 -ex1_snes_stol 1.e-14 -ex1_snes_monitor -ex1_snes_converged_reason -ex1_ts_type arkimex -ex1_ts_arkimex_type 1bee -ex1_ts_max_snes_failures -1 -ex1_ts_rtol 1e-1 -ex1_ts_dt 1.e-1 -ex1_ts_max_time 1 -ex1_ts_adapt_clip .5,1.25 -ex1_ts_adapt_scale_solve_failed 0.75 -ex1_ts_adapt_time_step_increase_delay 5 -ex1_ts_max_steps 1 -ex1_pc_type lu -ex1_ksp_type preonly -dm_landau_amr_levels_max 2,1 -dm_landau_device_type cpu
!
!TEST*/
