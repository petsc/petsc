!
!   Modified from ex15f.F for testing MUMPS
!   Solves a linear system in parallel with KSP.
!  -------------------------------------------------------------------------

      program main
#include <petsc/finclude/petscksp.h>
      use petscksp
      implicit none

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                   Variable declarations
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Vec              x,b,u
      Mat              A
      KSP              ksp
      PetscScalar      v,one,neg_one
      PetscReal        norm,tol
      PetscErrorCode   ierr
      PetscInt         i,j,II,JJ,Istart
      PetscInt         Iend,m,n,i1,its,five
      PetscMPIInt      rank
      PetscBool        flg
#if defined(PETSC_HAVE_MUMPS)
      PC               pc
      Mat              F
      PetscInt         ival,icntl,infog34
      PetscReal        cntl,rinfo12,rinfo13,val
#endif

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                 Beginning of program
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      PetscCallA(PetscInitialize(ierr))
      one     = 1.0
      neg_one = -1.0
      i1      = 1
      m       = 8
      n       = 7
      five    = 5
      PetscCallA(PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-m',m,flg,ierr))
      PetscCallA(PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-n',n,flg,ierr))
      PetscCallMPIA(MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr))

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!      Compute the matrix and right-hand-side vector that define
!      the linear system, Ax = b.
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      PetscCallA(MatCreate(PETSC_COMM_WORLD,A,ierr))
      PetscCallA(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n,ierr))
      PetscCallA(MatSetType(A, MATAIJ,ierr))
      PetscCallA(MatSetFromOptions(A,ierr))
      PetscCallA(MatMPIAIJSetPreallocation(A,five,PETSC_NULL_INTEGER,five,PETSC_NULL_INTEGER,ierr))
      PetscCallA(MatSeqAIJSetPreallocation(A,five,PETSC_NULL_INTEGER,ierr))

      PetscCallA(MatGetOwnershipRange(A,Istart,Iend,ierr))

!  Set matrix elements for the 2-D, five-point stencil in parallel.
!   - Each processor needs to insert only elements that it owns
!     locally (but any non-local elements will be sent to the
!     appropriate processor during matrix assembly).
!   - Always specify global row and columns of matrix entries.
!   - Note that MatSetValues() uses 0-based row and column numbers
!     in Fortran as well as in C.
      do 10, II=Istart,Iend-1
        v = -1.0
        i = II/n
        j = II - i*n
        if (i.gt.0) then
          JJ = II - n
          PetscCallA(MatSetValues(A,i1,II,i1,JJ,v,ADD_VALUES,ierr))
        endif
        if (i.lt.m-1) then
          JJ = II + n
          PetscCallA(MatSetValues(A,i1,II,i1,JJ,v,ADD_VALUES,ierr))
        endif
        if (j.gt.0) then
          JJ = II - 1
          PetscCallA(MatSetValues(A,i1,II,i1,JJ,v,ADD_VALUES,ierr))
        endif
        if (j.lt.n-1) then
          JJ = II + 1
          PetscCallA(MatSetValues(A,i1,II,i1,JJ,v,ADD_VALUES,ierr))
        endif
        v = 4.0
        PetscCallA( MatSetValues(A,i1,II,i1,II,v,ADD_VALUES,ierr))
 10   continue

!  Assemble matrix, using the 2-step process:
      PetscCallA(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY,ierr))
      PetscCallA(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY,ierr))

!  Create parallel vectors.
      PetscCallA(VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE,m*n,u,ierr))
      PetscCallA(VecDuplicate(u,b,ierr))
      PetscCallA(VecDuplicate(b,x,ierr))

!  Set exact solution; then compute right-hand-side vector.
      PetscCallA(VecSet(u,one,ierr))
      PetscCallA(MatMult(A,u,b,ierr))

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!         Create the linear solver and set various options
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      PetscCallA(KSPCreate(PETSC_COMM_WORLD,ksp,ierr))
      PetscCallA(KSPSetOperators(ksp,A,A,ierr))
      tol = 1.e-7
      PetscCallA(KSPSetTolerances(ksp,tol,PETSC_DEFAULT_REAL,PETSC_DEFAULT_REAL,PETSC_DEFAULT_INTEGER,ierr))

!  Test MUMPS
#if defined(PETSC_HAVE_MUMPS)
      PetscCallA(KSPSetType(ksp,KSPPREONLY,ierr))
      PetscCallA(KSPGetPC(ksp,pc,ierr))
      PetscCallA(PCSetType(pc,PCLU,ierr))
      PetscCallA(PCFactorSetMatSolverType(pc,MATSOLVERMUMPS,ierr))
      PetscCallA(PCFactorSetUpMatSolverType(pc,ierr))
      PetscCallA(PCFactorGetMatrix(pc,F,ierr))

!     sequential ordering
      icntl = 7
      ival  = 2
      PetscCallA(MatMumpsSetIcntl(F,icntl,ival,ierr))

!     threshold for row pivot detection
      icntl = 24
      ival  = 1
      PetscCallA(MatMumpsSetIcntl(F,icntl,ival,ierr))
      icntl = 3
      val = 1.e-6
      PetscCallA(MatMumpsSetCntl(F,icntl,val,ierr))

!     compute determinant of A
      icntl = 33
      ival  = 1
      PetscCallA(MatMumpsSetIcntl(F,icntl,ival,ierr))
#endif

      PetscCallA(KSPSetFromOptions(ksp,ierr))
      PetscCallA(KSPSetUp(ksp,ierr))
#if defined(PETSC_HAVE_MUMPS)
      icntl = 3;
      PetscCallA(MatMumpsGetCntl(F,icntl,cntl,ierr))
      icntl = 34
      PetscCallA(MatMumpsGetInfog(F,icntl,infog34,ierr))
      icntl = 12
      PetscCallA(MatMumpsGetRinfog(F,icntl,rinfo12,ierr))
      icntl = 13
      PetscCallA(MatMumpsGetRinfog(F,icntl,rinfo13,ierr))
      if (rank .eq. 0) then
         write(6,98) cntl
         write(6,99) rinfo12,rinfo13,infog34
      endif
 98   format('Mumps row pivot threshold = ',1pe11.2)
 99   format('Mumps determinant=(',1pe11.2,1pe11.2,')*2^',i3)
#endif

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                      Solve the linear system
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      PetscCallA(KSPSolve(ksp,b,x,ierr))

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                     Check solution and clean up
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      PetscCallA(VecAXPY(x,neg_one,u,ierr))
      PetscCallA(VecNorm(x,NORM_2,norm,ierr))
      PetscCallA(KSPGetIterationNumber(ksp,its,ierr))

      if (rank .eq. 0) then
        if (norm .gt. 1.e-12) then
           write(6,100) norm,its
        else
           write(6,110) its
        endif
      endif
  100 format('Norm of error ',1pe11.4,' iterations ',i5)
  110 format('Norm of error < 1.e-12,iterations ',i5)

!  Free work space.  All PETSc objects should be destroyed when they
!  are no longer needed.

      PetscCallA(KSPDestroy(ksp,ierr))
      PetscCallA(VecDestroy(u,ierr))
      PetscCallA(VecDestroy(x,ierr))
      PetscCallA(VecDestroy(b,ierr))
      PetscCallA(MatDestroy(A,ierr))

!  Always call PetscFinalize() before exiting a program.
      PetscCallA(PetscFinalize(ierr))
      end

!
!/*TEST
!
!   test:
!      suffix: mumps
!      nsize: 3
!      requires: mumps double
!      output_file: output/ex52f_1.out
!
!TEST*/
