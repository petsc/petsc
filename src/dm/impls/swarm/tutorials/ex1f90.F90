! Example program demonstrating projection between particle and finite element spaces
      program DMSwarmTestProjection
#include "petsc/finclude/petscdmplex.h"
#include "petsc/finclude/petscdmswarm.h"
#include "petsc/finclude/petscksp.h"
      use petscdmplex
      use petscdmswarm
      use petscdt
      use petscksp
      use petscsys
      implicit none

      DM ::          dm, sw
      PetscFE ::     fe
      KSP ::         ksp
      Mat ::         M_p, M
      Vec ::         f, rho, rhs
      PetscInt ::    dim, Nc = 1, degree = 1, timestep = 0
      PetscInt ::    Np = 100, p, field = 0, zero = 0, bs
      PetscReal ::   time = 0.0,  norm
      PetscBool ::   removePoints = PETSC_TRUE
      PetscDataType :: dtype
      PetscScalar, pointer :: coords(:)
      PetscScalar, pointer :: wq(:)
      PetscErrorCode :: ierr

      PetscCallA(PetscInitialize(PETSC_NULL_CHARACTER, ierr))
      PetscCallA(DMCreate(PETSC_COMM_WORLD, dm, ierr))
      PetscCallA(DMSetType(dm, DMPLEX, ierr))
      PetscCallA(DMSetFromOptions(dm, ierr))
      PetscCallA(DMGetDimension(dm, dim, ierr))
      PetscCallA(DMViewFromOptions(dm, PETSC_NULL_VEC, '-dm_view', ierr))

!     Create finite element space
      PetscCallA(PetscFECreateLagrange(PETSC_COMM_SELF, dim, Nc, PETSC_FALSE, degree, PETSC_DETERMINE, fe, ierr))
      PetscCallA(DMSetField(dm, field, PETSC_NULL_DMLABEL, fe, ierr))
      PetscCallA(DMCreateDS(dm, ierr))
      PetscCallA(PetscFEDestroy(fe, ierr))

!     Create particle swarm
      PetscCallA(DMCreate(PETSC_COMM_WORLD, sw, ierr))
      PetscCallA(DMSetType(sw, DMSWARM, ierr))
      PetscCallA(DMSetDimension(sw, dim, ierr))
      PetscCallA(DMSwarmSetType(sw, DMSWARM_PIC, ierr))
      PetscCallA(DMSwarmSetCellDM(sw, dm, ierr))
      PetscCallA(DMSwarmRegisterPetscDatatypeField(sw, 'w_q', Nc, PETSC_SCALAR, ierr))
      PetscCallA(DMSwarmFinalizeFieldRegister(sw, ierr))
      PetscCallA(DMSwarmSetLocalSizes(sw, Np, zero, ierr))
      PetscCallA(DMSetFromOptions(sw, ierr))
      PetscCallA(DMSwarmGetField(sw, 'w_q', bs, dtype, wq, ierr))
      PetscCallA(DMSwarmGetField(sw, 'DMSwarmPIC_coor', bs, dtype, coords, ierr))
      do p = 1, Np
        coords(p*2-1) = -cos(dble(p)/dble(Np+1) * PETSC_PI)
        coords(p*2-0) =  sin(dble(p)/dble(Np+1) * PETSC_PI)
        wq(p)         = 1.0
      end do
      PetscCallA(DMSwarmRestoreField(sw, 'DMSwarmPIC_coor', bs, dtype, coords, ierr))
      PetscCallA(DMSwarmRestoreField(sw, 'w_q', bs, dtype, wq, ierr))
      PetscCallA(DMSwarmMigrate(sw, removePoints, ierr))
      PetscCallA(DMViewFromOptions(sw, PETSC_NULL_VEC, '-swarm_view', ierr))

!     Project particles to field
!       This gives M f = \int_\Omega \phi f, which looks like a rhs for a PDE
      PetscCallA(DMCreateMassMatrix(sw, dm, M_p, ierr))
      PetscCallA(DMCreateGlobalVector(dm, rho, ierr))
      PetscCallA(DMSwarmCreateGlobalVectorFromField(sw, 'w_q', f, ierr))
      PetscCallA(MatMultTranspose(M_p, f, rho, ierr))

!     Visualize mesh field
      PetscCallA(DMSetOutputSequenceNumber(dm, timestep, time, ierr))
      PetscCallA(PetscObjectViewFromOptions(rho, PETSC_NULL_VEC, '-rho_view', ierr))

!     Project field to particles
!       This gives f_p = M_p^+ M f
      PetscCallA(DMCreateMassMatrix(dm, dm, M, ierr))
      PetscCallA(DMCreateGlobalVector(dm, rhs, ierr))
      if (.false.) then
         PetscCallA(MatMult(M, rho, rhs, ierr)) ! this is what you would do for and FE solve
      else
         PetscCallA(VecCopy(rho, rhs, ierr)) ! Identity: M^1 M rho
      end if
      PetscCallA(KSPCreate(PETSC_COMM_WORLD, ksp, ierr))
      PetscCallA(KSPSetOptionsPrefix(ksp, 'ftop_', ierr))
      PetscCallA(KSPSetFromOptions(ksp, ierr))
      PetscCallA(KSPSetOperators(ksp, M_p, M_p, ierr))
      PetscCallA(KSPSolveTranspose(ksp, rhs, f, ierr))
      PetscCallA(KSPDestroy(ksp, ierr))
      PetscCallA(VecDestroy(rhs, ierr))
      PetscCallA(MatDestroy(M_p, ierr))
      PetscCallA(MatDestroy(M, ierr))

!     Visualize particle field
      PetscCallA(DMSetOutputSequenceNumber(sw, timestep, time, ierr))
      PetscCallA(PetscObjectViewFromOptions(f, PETSC_NULL_VEC, '-weights_view', ierr))
      PetscCallA(VecNorm(f,NORM_1,norm,ierr))
      print *, 'Total number density = ', norm
!     Cleanup
      PetscCallA(DMSwarmDestroyGlobalVectorFromField(sw, 'w_q', f, ierr))
      PetscCallA(MatDestroy(M_p, ierr))
      PetscCallA(VecDestroy(rho, ierr))
      PetscCallA(DMDestroy(sw, ierr))
      PetscCallA(DMDestroy(dm, ierr))

      PetscCallA(PetscFinalize(ierr))
      end program DMSwarmTestProjection

!/*TEST
!  build:
!    requires: defined(PETSC_USING_F90FREEFORM) double !complex
!
!  test:
!    suffix: 0
!    requires: double
!    args: -dm_plex_simplex 0 -dm_plex_box_faces 40,20 -dm_plex_box_lower -2.0,0.0 -dm_plex_box_upper 2.0,2.0 -ftop_ksp_type lsqr -ftop_pc_type none -dm_view -swarm_view
!    filter: grep -v DM_ | grep -v atomic
!    filter_output: grep -v atomic
!
!TEST*/
