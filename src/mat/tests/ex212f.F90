!
!  Program to test recently added F90 features for Mat
!
      program main

#include <petsc/finclude/petscmat.h>
       use petscmat
       implicit none

      PetscErrorCode  ierr
      Mat A,B
      Mat C,SC
      MatNullSpace sp,sp1
      PetscInt one,zero,rend
      PetscScalar sone
      Vec x,y

      zero = 0
      one  = 1
      sone = 1
      PetscCallA(PetscInitialize(ierr))

      PetscCallA(MatCreate(PETSC_COMM_WORLD,A,ierr))
      PetscCallA(MatCreate(PETSC_COMM_WORLD,B,ierr))

      PetscCallA(MatGetNullSpace(A,sp,ierr))
      if (sp .ne. PETSC_NULL_MATNULLSPACE) then; SETERRA(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Matrix null space should not exist"); endif

      PetscCallA(MatSetNullSpace(A,PETSC_NULL_MATNULLSPACE,ierr))
      PetscCallA(MatGetNullSpace(A,sp,ierr))
      if (sp .ne. PETSC_NULL_MATNULLSPACE) then; SETERRA(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Matrix null space should not exist"); endif

      PetscCallA(MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,zero,PETSC_NULL_VEC,sp,ierr))
      PetscCallA(MatSetNullSpace(A,sp,ierr))
      PetscCallA(MatGetNullSpace(A,sp1,ierr))
      if (sp1 .eq. PETSC_NULL_MATNULLSPACE) then; SETERRA(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Matrix null space should not be null"); endif
      PetscCallA(MatNullSpaceDestroy(sp,ierr))

      PetscCallA(MatCreateSeqDense(PETSC_COMM_WORLD,one,one,PETSC_NULL_SCALAR,C,ierr))
      PetscCallA(MatSetValues(C,one,zero,one,zero,sone,INSERT_VALUES,ierr))
      PetscCallA(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY,ierr))
      PetscCallA(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY,ierr))
      PetscCallA(MatCreateSchurComplement(C,C,C,C,PETSC_NULL_MAT,SC,ierr))
      PetscCallA(MatGetOwnershipRange(SC,PETSC_NULL_INTEGER,rend,ierr))
      PetscCallA(VecCreateSeq(PETSC_COMM_SELF,one,x,ierr))
      PetscCallA(VecDuplicate(x,y,ierr))
      PetscCallA(VecSetValues(x,one,zero,sone,INSERT_VALUES,ierr))
      PetscCallA(VecAssemblyBegin(x,ierr))
      PetscCallA(VecAssemblyEnd(x,ierr))
      PetscCallA(MatMult(SC,x,y,ierr))
      PetscCallA(VecView(y,PETSC_VIEWER_STDOUT_SELF,ierr))
      PetscCallA(VecSetRandom(x,PETSC_NULL_RANDOM,ierr))
      PetscCallA(VecView(x,PETSC_VIEWER_STDOUT_SELF,ierr))

      PetscCallA(MatDestroy(SC,ierr))
      PetscCallA(MatDestroy(C,ierr))
      PetscCallA(VecDestroy(x,ierr))
      PetscCallA(VecDestroy(y,ierr))
      PetscCallA(MatDestroy(A,ierr))
      PetscCallA(MatDestroy(B,ierr))
      PetscCallA(PetscFinalize(ierr))
      end

!/*TEST
!
!   test:
!      requires: !complex
!
!TEST*/
