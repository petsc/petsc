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

      call PetscInitialize(PETSC_NULL_CHARACTER, ierr)
      if (ierr .ne. 0) then
        print*,'Unable to initialize PETSc'
        stop
      endif

      call DMCreate(PETSC_COMM_WORLD, dm, ierr);CHKERRA(ierr)
      call DMSetType(dm, DMPLEX, ierr);CHKERRA(ierr)
      call DMSetFromOptions(dm, ierr);CHKERRA(ierr)
      call DMGetDimension(dm, dim, ierr);CHKERRA(ierr)
      call DMViewFromOptions(dm, PETSC_NULL_VEC, '-dm_view', ierr);CHKERRA(ierr)

!     Create finite element space
      call PetscFECreateLagrange(PETSC_COMM_SELF, dim, Nc, PETSC_FALSE, degree, PETSC_DETERMINE, fe, ierr);CHKERRA(ierr)
      call DMSetField(dm, field, PETSC_NULL_DMLABEL, fe, ierr);CHKERRA(ierr)
      call DMCreateDS(dm, ierr);CHKERRA(ierr)
      call PetscFEDestroy(fe, ierr);CHKERRA(ierr)

!     Create particle swarm
      call DMCreate(PETSC_COMM_WORLD, sw, ierr);CHKERRA(ierr)
      call DMSetType(sw, DMSWARM, ierr);CHKERRA(ierr)
      call DMSetDimension(sw, dim, ierr);CHKERRA(ierr)
      call DMSwarmSetType(sw, DMSWARM_PIC, ierr);CHKERRA(ierr)
      call DMSwarmSetCellDM(sw, dm, ierr);CHKERRA(ierr)
      call DMSwarmRegisterPetscDatatypeField(sw, 'w_q', Nc, PETSC_SCALAR, ierr);CHKERRA(ierr)
      call DMSwarmFinalizeFieldRegister(sw, ierr);CHKERRA(ierr)
      call DMSwarmSetLocalSizes(sw, Np, zero, ierr);CHKERRA(ierr)
      call DMSetFromOptions(sw, ierr);CHKERRA(ierr)
      call DMSwarmGetField(sw, 'w_q', bs, dtype, wq, ierr);CHKERRA(ierr)
      call DMSwarmGetField(sw, 'DMSwarmPIC_coor', bs, dtype, coords, ierr);CHKERRA(ierr)
      do p = 1, Np
        coords(p*2-1) = -cos(dble(p)/dble(Np+1) * PETSC_PI)
        coords(p*2-0) =  sin(dble(p)/dble(Np+1) * PETSC_PI)
        wq(p)         = 1.0
      end do
      call DMSwarmRestoreField(sw, 'DMSwarmPIC_coor', bs, dtype, coords, ierr);CHKERRA(ierr)
      call DMSwarmRestoreField(sw, 'w_q', bs, dtype, wq, ierr);CHKERRA(ierr)
      call DMSwarmMigrate(sw, removePoints, ierr);CHKERRA(ierr)
      call DMViewFromOptions(sw, PETSC_NULL_VEC, '-swarm_view', ierr);CHKERRA(ierr)

!     Project particles to field
!       This gives M f = \int_\Omega \phi f, which looks like a rhs for a PDE
      call DMCreateMassMatrix(sw, dm, M_p, ierr);CHKERRA(ierr)
      call DMCreateGlobalVector(dm, rho, ierr);CHKERRA(ierr)
      call DMSwarmCreateGlobalVectorFromField(sw, 'w_q', f, ierr);CHKERRA(ierr)
      call MatMultTranspose(M_p, f, rho, ierr);CHKERRA(ierr)

!     Visualize mesh field
      call DMSetOutputSequenceNumber(dm, timestep, time, ierr);CHKERRA(ierr)
      call PetscObjectViewFromOptions(rho, PETSC_NULL_VEC, '-rho_view', ierr);CHKERRA(ierr)

!     Project field to particles
!       This gives f_p = M_p^+ M f
      call DMCreateMassMatrix(dm, dm, M, ierr);CHKERRA(ierr)
      call DMCreateGlobalVector(dm, rhs, ierr);CHKERRA(ierr)
      if (.false.) then
         call MatMult(M, rho, rhs, ierr);CHKERRA(ierr) ! this is what you would do for and FE solve
      else
         call VecCopy(rho, rhs, ierr);CHKERRA(ierr) ! Indentity: M^1 M rho
      end if
      call KSPCreate(PETSC_COMM_WORLD, ksp, ierr);CHKERRA(ierr)
      call KSPSetOptionsPrefix(ksp, 'ftop_', ierr);CHKERRA(ierr)
      call KSPSetFromOptions(ksp, ierr);CHKERRA(ierr)
      call KSPSetOperators(ksp, M_p, M_p, ierr);CHKERRA(ierr)
      call KSPSolveTranspose(ksp, rhs, f, ierr);CHKERRA(ierr)
      call KSPDestroy(ksp, ierr);CHKERRA(ierr)
      call VecDestroy(rhs, ierr);CHKERRA(ierr)
      call MatDestroy(M_p, ierr);CHKERRA(ierr)
      call MatDestroy(M, ierr);CHKERRA(ierr)

!     Visualize particle field
      call DMSetOutputSequenceNumber(sw, timestep, time, ierr);CHKERRA(ierr)
      call PetscObjectViewFromOptions(f, PETSC_NULL_VEC, '-weights_view', ierr);CHKERRA(ierr)
      call VecNorm(f,NORM_1,norm,ierr);CHKERRA(ierr)
      print *, 'Total number density = ', norm
!     Cleanup
      call DMSwarmDestroyGlobalVectorFromField(sw, 'w_q', f, ierr);CHKERRA(ierr)
      call MatDestroy(M_p, ierr);CHKERRA(ierr)
      call VecDestroy(rho, ierr);CHKERRA(ierr)
      call DMDestroy(sw, ierr);CHKERRA(ierr)
      call DMDestroy(dm, ierr);CHKERRA(ierr)

      call PetscFinalize(ierr)
      end program DMSwarmTestProjection

!/*TEST
!  build:
!    requires: define(PETSC_USING_F90FREEFORM) double !complex
!
!  test:
!    suffix: 0
!    requires: double
!    args: -dm_plex_simplex 0 -dm_plex_box_faces 40,20 -dm_plex_box_lower -2.0,0.0 -dm_plex_box_upper 2.0,2.0 -ftop_ksp_type lsqr -ftop_pc_type none -dm_view -swarm_view
!    filter: grep -v DM_ | grep -v atomic
!    filter_output: grep -v atomic
!
!TEST*/
