!
! This program is modified from a user's contribution.
! It illustrates how to call MUMPS's LU solver
!

      program main
#include <petsc/finclude/petscmat.h>
      use petscmat
      implicit none

      Vec            x,b,u
      Mat            A, fact
      PetscInt       i,j,II,JJ,m
      PetscInt       Istart, Iend
      PetscInt       ione, ifive
      PetscBool      wmumps
      PetscBool      flg
      PetscScalar    one, v
      IS             perm,iperm
      PetscErrorCode ierr
      PetscReal      info(MAT_FACTORINFO_SIZE)

      call PetscInitialize(PETSC_NULL_CHARACTER, ierr)
      if (ierr .ne. 0) then
        print*,'Unable to initialize PETSc'
        stop
      endif
      m    = 10
      one  = 1.0
      ione = 1
      ifive = 5

      wmumps = PETSC_FALSE

      call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,                   &
     &                        '-m',m,flg, ierr)
      call PetscOptionsGetBool(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,                  &
     &                         '-use_mumps',wmumps,flg,ierr)

      call MatCreate(PETSC_COMM_WORLD, A, ierr)
      call MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, m*m, m*m, ierr)
      call MatSetType(A, MATAIJ, ierr)
      call MatSetFromOptions(A, ierr)
      call MatSeqAIJSetPreallocation(A,ifive, PETSC_NULL_INTEGER, ierr)
      call MatMPIAIJSetPreallocation(A,ifive,PETSC_NULL_INTEGER,ifive,  &
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

      call MatFactorInfoInitialize(info,ierr)
      call MatGetOrdering(A,MATORDERINGNATURAL,perm,iperm,ierr)
      if (wmumps) then
          write(*,*) 'use MUMPS LU...'
          call MatGetFactor(A,MATSOLVERMUMPS,MAT_FACTOR_LU,fact,ierr)
      else
         write(*,*) 'use PETSc LU...'
         call MatGetFactor(A,MATSOLVERPETSC,MAT_FACTOR_LU,fact,ierr)
      endif
      call MatLUFactorSymbolic(fact, A, perm, iperm,                                   &
     &         info, ierr)
      call ISDestroy(perm,ierr)
      call ISDestroy(iperm,ierr)

      call MatLUFactorNumeric(fact, A, info, ierr)
      call MatSolve(fact, b, x,ierr)
      call MatDestroy(fact, ierr)

      call MatDestroy(A, ierr)
      call VecDestroy(u, ierr)
      call VecDestroy(x, ierr)
      call VecDestroy(b, ierr)

      call PetscFinalize(ierr)
      end

!/*TEST
!
!   test:
!
!   test:
!     suffix: 2
!     args: -use_mumps
!     requires: mumps
!
!TEST*/
