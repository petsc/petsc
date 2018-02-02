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
      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then
         print*, 'Unable to begin PETSc program'
      endif

      call MatCreate(PETSC_COMM_WORLD,A,ierr)
      call MatCreate(PETSC_COMM_WORLD,B,ierr)

      call MatGetNullSpace(A,sp,ierr)
      if (sp .ne. PETSC_NULL_MATNULLSPACE) then; SETERRA(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Matrix null space should not exist"); endif

      call MatSetNullSpace(A,PETSC_NULL_MATNULLSPACE,ierr)
      call MatGetNullSpace(A,sp,ierr)
      if (sp .ne. PETSC_NULL_MATNULLSPACE) then; SETERRA(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Matrix null space should not exist"); endif

      call MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,zero,PETSC_NULL_VEC,sp,ierr)
      call MatSetNullSpace(A,sp,ierr)
      call MatGetNullSpace(A,sp1,ierr)
      if (sp1 .eq. PETSC_NULL_MATNULLSPACE) then; SETERRA(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Matrix null space should not be null"); endif
      call MatNullSpaceDestroy(sp,ierr)

      call MatCreateSeqDense(PETSC_COMM_WORLD,one,one,PETSC_NULL_SCALAR,C,ierr)
      call MatSetValues(C,one,zero,one,zero,sone,INSERT_VALUES,ierr)
      call MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY,ierr)
      call MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY,ierr)
      call MatCreateSchurComplement(C,C,C,C,PETSC_NULL_MAT,SC,ierr)
      call MatGetOwnershipRange(SC,PETSC_NULL_INTEGER,rend,ierr)
      call VecCreateSeq(PETSC_COMM_SELF,one,x,ierr)
      call VecDuplicate(x,y,ierr)
      call VecSetValues(x,one,zero,sone,INSERT_VALUES,ierr)
      call VecAssemblyBegin(x,ierr)
      call VecAssemblyEnd(x,ierr)
      call MatMult(SC,x,y,ierr)
      call VecView(y,PETSC_VIEWER_STDOUT_SELF,ierr)
      call VecSetRandom(x,PETSC_NULL_RANDOM,ierr)
      call VecView(x,PETSC_VIEWER_STDOUT_SELF,ierr)

      call MatDestroy(SC,ierr)
      call MatDestroy(C,ierr)
      call VecDestroy(x,ierr)
      call VecDestroy(y,ierr)
      call MatDestroy(A,ierr)
      call MatDestroy(B,ierr)
      call PetscFinalize(ierr)
      end

!/*TEST
!
!   test:
!      requires: !complex
!
!TEST*/
