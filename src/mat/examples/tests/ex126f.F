!
! This program is modified from a user's contribution.
! It illustrates how to call MUMPS's LU solver
!

      program main
      implicit none

#include <finclude/petscsys.h>
#include <finclude/petscvec.h>
#include <finclude/petscmat.h>

      Vec            x,b,u
      Mat            A, fact
      PetscInt       i,j,II,JJ,m
      PetscInt       Istart, Iend
      PetscInt       ione
      PetscBool      wmumps
      PetscBool      flg
      PetscScalar    one, v
      IS             perm,iperm
      PetscErrorCode ierr

      call PetscInitialize(PETSC_NULL_CHARACTER, ierr)
      m    = 10
      one  = 1.0
      ione = 1

      wmumps = PETSC_FALSE

      call PetscOptionsGetInt(PETSC_NULL_CHARACTER,'-m',m,flg, ierr)
      call PetscOptionsGetBool(PETSC_NULL_CHARACTER,'-use_mumps',                       &
     &                         wmumps,flg,ierr)

      call MatCreate(PETSC_COMM_WORLD, A, ierr)
      call MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, m*m, m*m, ierr)
      call MatSetType(A, MATAIJ, ierr)
      call MatSetFromOptions(A, ierr)
      call MatSeqAIJSetPreallocation(A,5, PETSC_NULL_INTEGER, ierr)
      call MatMPIAIJSetPreallocation(A,5,PETSC_NULL_INTEGER,5,                          &
     &     PETSC_NULL_INTEGER,ierr)

      call MatGetOwnershipRange(A,Istart,Iend,ierr)

      do 10, II=Istart,Iend - 1
        v = -1.0
        i = II/m
        j = II - i*m
        if (i.gt.0) then
          JJ = II - m
          call MatSetValues(A,ione,II,ione,JJ,v,INSERT_VALUES,ierr)
        endif
        if (i.lt.m-1) then
          JJ = II + m
          call MatSetValues(A,ione,II,ione,JJ,v,INSERT_VALUES,ierr)
        endif
        if (j.gt.0) then
          JJ = II - 1
          call MatSetValues(A,ione,II,ione,JJ,v,INSERT_VALUES,ierr)
        endif
        if (j.lt.m-1) then
          JJ = II + 1
          call MatSetValues(A,ione,II,ione,JJ,v,INSERT_VALUES,ierr)
        endif
        v = 4.0
        call  MatSetValues(A,ione,II,ione,II,v,INSERT_VALUES,ierr)
 10   continue

      call MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY, ierr)
      call MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY, ierr)

      call VecCreate(PETSC_COMM_WORLD, u, ierr)
      call VecSetSizes(u, PETSC_DECIDE, m*m, ierr)
      call VecSetFromOptions(u, ierr)
      call VecDuplicate(u,b,ierr)
      call VecDuplicate(b,x,ierr)
      call VecSet(u, one, ierr)
      call MatMult(A, u, b, ierr)

      if (wmumps) then
          write(*,*) 'use MUMPS LU...'
          call MatGetFactor(A, MATSOLVERMUMPS, MAT_FACTOR_LU,                             &
     &                      fact, ierr)
          call MatLUFactorSymbolic(fact, A, PETSC_NULL, PETSC_NULL,                       &
     &         PETSC_NULL, ierr)
      else
         write(*,*) 'use PETSc LU...'
         call MatGetOrdering(A,MATORDERINGNATURAL,perm,iperm,ierr)
         call MatGetFactor(A, MATSOLVERPETSC, MAT_FACTOR_LU,                              &
     &         fact, ierr)

         call MatLUFactorSymbolic(fact, A, perm, iperm,                                   &
     &         PETSC_NULL, ierr)
         call ISDestroy(perm,ierr)
         call ISDestroy(iperm,ierr)
      endif

      call MatLUFactorNumeric(fact, A, PETSC_NULL, ierr)
      call MatSolve(fact, b, x,ierr)
      call MatDestroy(fact, ierr)

      call MatDestroy(A, ierr)
      call VecDestroy(u, ierr)
      call VecDestroy(x, ierr)
      call VecDestroy(b, ierr)

      call PetscFinalize(ierr)
      end

