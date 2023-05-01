
!     Contributed by leonardo.mutti01@universitadipavia.it
      program main
#include <petsc/finclude/petscsys.h>
#include <petsc/finclude/petscmat.h>
#include <petsc/finclude/petscksp.h>
#include <petsc/finclude/petscpc.h>
      USE petscmat
      USE petscksp
      USE petscpc
      implicit none

      Mat   :: A
      PetscInt :: M, M2, NSubx, dof, overlap, NSub
      PetscInt :: I, J
      PetscMPIInt :: size
      PetscErrorCode :: ierr
      PetscScalar :: v
      KSP            :: ksp
      PC             :: pc
      IS :: subdomains_IS(20), inflated_IS(20)
      PetscViewer singleton

      PetscCallA(PetscInitialize(PETSC_NULL_CHARACTER, ierr))
      PetscCallMPIA(MPI_Comm_size(PETSC_COMM_WORLD, size, ierr))
      M = 16
      M2 = M*M
      PetscCallA(MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, M2, M2, PETSC_DEFAULT_INTEGER, PETSC_NULL_INTEGER, PETSC_DEFAULT_INTEGER, PETSC_NULL_INTEGER,A, ierr))
      DO I=1,M2
         DO J=1,M2
            v = I*J
            PetscCallA(MatSetValue(A, I-1, J-1, v, INSERT_VALUES, ierr))
         END DO
      END DO

      PetscCallA(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY, ierr))
      PetscCallA(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY, ierr))
      PetscCallA(KSPCreate(PETSC_COMM_WORLD, ksp, ierr))
      PetscCallA(KSPSetOperators(ksp, A, A, ierr))
      PetscCallA(KSPSetType(ksp, 'bcgs', ierr))
      PetscCallA(KSPGetPC(ksp, pc, ierr))
      PetscCallA(KSPSetUp(ksp, ierr))
      PetscCallA(PCSetType(pc, PCGASM, ierr))
      PetscCallA(PCSetUp(pc, ierr))

      NSubx = 4
      dof = 1
      overlap = 0

      PetscCallA(PCGASMCreateSubdomains2D(pc, M, M,NSubx, NSubx, dof, overlap, NSub, subdomains_IS, inflated_IS, ierr))
      PetscCallA(PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD, PETSC_COMM_SELF, singleton, ierr))
      PetscCallA(PetscViewerASCIIPrintf(singleton, 'GASM index sets from this MPI process\n', ierr))
      do i=1,Nsub
        PetscCallA(ISView(subdomains_IS(i), singleton, ierr))
        PetscCallA(ISDestroy(subdomains_IS(i), ierr))
        PetscCallA(ISDestroy(inflated_IS(i), ierr))
      end do
      PetscCallA(PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD, PETSC_COMM_SELF, singleton, ierr))
      PetscCallA(PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD, ierr))

      if (size == 1) then
        ! this routine only works on one rank
        PetscCallA(PCASMCreateSubdomains2D(M, M, NSubx, NSubx, dof, overlap, NSub, subdomains_IS, inflated_IS, ierr))
        do i=1,Nsub
          PetscCallA(ISView(subdomains_IS(i), PETSC_VIEWER_STDOUT_SELF, ierr))
          PetscCallA(ISDestroy(subdomains_IS(i), ierr))
          PetscCallA(ISDestroy(inflated_IS(i), ierr))
        end do
      endif

      PetscCallA(MatDestroy(A, ierr))
      PetscCallA(KSPDestroy(ksp, ierr))
      PetscCallA(PetscFinalize(ierr))
      end

!/*TEST
!
!   test:
!     suffix: 1
!
!   test:
!     suffix: 2
!     nsize: 2
!
!TEST*/
