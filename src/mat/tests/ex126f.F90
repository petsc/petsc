!
! This program is modified from a user's contribution.
! It illustrates how to PetscCallA(MUMPS's LU solver
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

      PetscCallA(PetscInitialize(PETSC_NULL_CHARACTER, ierr))
      m    = 10
      one  = 1.0
      ione = 1
      ifive = 5

      wmumps = PETSC_FALSE

      PetscCallA(PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-m',m,flg, ierr))
      PetscCallA(PetscOptionsGetBool(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-use_mumps',wmumps,flg,ierr))

      PetscCallA(MatCreate(PETSC_COMM_WORLD, A, ierr))
      PetscCallA(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, m*m, m*m, ierr))
      PetscCallA(MatSetType(A, MATAIJ, ierr))
      PetscCallA(MatSetFromOptions(A, ierr))
      PetscCallA(MatSeqAIJSetPreallocation(A,ifive, PETSC_NULL_INTEGER, ierr))
      PetscCallA(MatMPIAIJSetPreallocation(A,ifive,PETSC_NULL_INTEGER,ifive,PETSC_NULL_INTEGER,ierr))

      PetscCallA(MatGetOwnershipRange(A,Istart,Iend,ierr))

      do 10, II=Istart,Iend - 1
        v = -1.0
        i = II/m
        j = II - i*m
        if (i.gt.0) then
          JJ = II - m
          PetscCallA(MatSetValues(A,ione,II,ione,JJ,v,INSERT_VALUES,ierr))
        endif
        if (i.lt.m-1) then
          JJ = II + m
          PetscCallA(MatSetValues(A,ione,II,ione,JJ,v,INSERT_VALUES,ierr))
        endif
        if (j.gt.0) then
          JJ = II - 1
          PetscCallA(MatSetValues(A,ione,II,ione,JJ,v,INSERT_VALUES,ierr))
        endif
        if (j.lt.m-1) then
          JJ = II + 1
          PetscCallA(MatSetValues(A,ione,II,ione,JJ,v,INSERT_VALUES,ierr))
        endif
        v = 4.0
        PetscCallA( MatSetValues(A,ione,II,ione,II,v,INSERT_VALUES,ierr))
 10   continue

      PetscCallA(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY, ierr))
      PetscCallA(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY, ierr))

      PetscCallA(VecCreate(PETSC_COMM_WORLD, u, ierr))
      PetscCallA(VecSetSizes(u, PETSC_DECIDE, m*m, ierr))
      PetscCallA(VecSetFromOptions(u, ierr))
      PetscCallA(VecDuplicate(u,b,ierr))
      PetscCallA(VecDuplicate(b,x,ierr))
      PetscCallA(VecSet(u, one, ierr))
      PetscCallA(MatMult(A, u, b, ierr))

      PetscCallA(MatFactorInfoInitialize(info,ierr))
      PetscCallA(MatGetOrdering(A,MATORDERINGNATURAL,perm,iperm,ierr))
      if (wmumps) then
          write(*,*) 'use MUMPS LU...'
          PetscCallA(MatGetFactor(A,MATSOLVERMUMPS,MAT_FACTOR_LU,fact,ierr))
      else
         write(*,*) 'use PETSc LU...'
         PetscCallA(MatGetFactor(A,MATSOLVERPETSC,MAT_FACTOR_LU,fact,ierr))
      endif
      PetscCallA(MatLUFactorSymbolic(fact, A, perm, iperm, info, ierr))
      PetscCallA(ISDestroy(perm,ierr))
      PetscCallA(ISDestroy(iperm,ierr))

      PetscCallA(MatLUFactorNumeric(fact, A, info, ierr))
      PetscCallA(MatSolve(fact, b, x,ierr))
      PetscCallA(MatDestroy(fact, ierr))

      PetscCallA(MatDestroy(A, ierr))
      PetscCallA(VecDestroy(u, ierr))
      PetscCallA(VecDestroy(x, ierr))
      PetscCallA(VecDestroy(b, ierr))

      PetscCallA(PetscFinalize(ierr))
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
