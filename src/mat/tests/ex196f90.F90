!
!
!   This program demonstrates use of MatSeqAIJGetArrayF90()
!
      program main

#include <petsc/finclude/petscmat.h>
      use petscmat
      implicit none

      Mat                            A
      PetscErrorCode                 ierr
      PetscViewer                    v
      PetscScalar, pointer ::        aa(:)
      character*(PETSC_MAX_PATH_LEN) f
      PetscBool                      flg

      PetscCallA(PetscInitialize(ierr))

      PetscCallA(PetscOptionsGetString(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-f',f,flg,ierr))
      PetscCallA(PetscViewerBinaryOpen(PETSC_COMM_WORLD,f,FILE_MODE_READ,v,ierr))

      PetscCallA(MatCreate(PETSC_COMM_WORLD,A,ierr))
      PetscCallA(MatSetType(A, MATSEQAIJ,ierr))
      PetscCallA(MatLoad(A,v,ierr))

      PetscCallA(MatView(A,PETSC_VIEWER_STDOUT_WORLD,ierr))

      PetscCallA(MatSeqAIJGetArrayF90(A,aa,ierr))
      print*,aa(3)

      PetscCallA(MatDestroy(A,ierr))
      PetscCallA(PetscViewerDestroy(v,ierr))

      PetscCallA(PetscFinalize(ierr))
      end

!/*TEST
!
!   test:
!      args: -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/ns-real-int32-float64 -malloc_dump
!      requires: !complex double !defined(PETSC_USE_64BIT_INDICES)
!
!TEST*/
