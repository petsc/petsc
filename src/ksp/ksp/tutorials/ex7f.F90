! Block Jacobi preconditioner for solving a linear system in parallel with KSP
! The code indicates the procedures for setting the particular block sizes and
! for using different linear solvers on the individual blocks

! This example focuses on ways to customize the block Jacobi preconditioner.
! See ex1.c and ex2.c for more detailed comments on the basic usage of KSP
! (including working with matrices and vectors)

! Recall: The block Jacobi method is equivalent to the ASM preconditioner with zero overlap.

program main
#include <petsc/finclude/petscksp.h>
  use petscksp

  implicit none
  Vec             :: x, b, u      ! approx solution, RHS, exact solution
  Mat             :: A            ! linear system matrix
  KSP             :: ksp         ! KSP context
  PC              :: myPc           ! PC context
  PC              :: subpc        ! PC context for subdomain
  PetscReal       :: norm         ! norm of solution error
  PetscReal, parameter :: tol = 1.e-6
  PetscErrorCode  :: ierr
  PetscInt        :: i, j, Ii, JJ, n
  PetscInt        :: m
  PetscMPIInt     :: rank, size
  PetscInt        :: its, nlocal, first, Istart, Iend
  PetscScalar     :: v
  PetscScalar, parameter :: &
    myNone = -1.0, &
    sone = 1.0
  PetscBool       :: isbjacobi, flg
  KSP, pointer      ::   subksp(:) => null()
  PetscInt :: blks(4)
  character(len=PETSC_MAX_PATH_LEN) :: outputString
  PetscInt, parameter :: one = 1, five = 5

  PetscCallA(PetscInitialize(ierr))
  m = 4
  PetscCallA(PetscOptionsGetInt(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, '-m', m, flg, ierr))
  PetscCallMPIA(MPI_Comm_rank(PETSC_COMM_WORLD, rank, ierr))
  PetscCallMPIA(MPI_Comm_size(PETSC_COMM_WORLD, size, ierr))
  n = m + 2
  blks(1) = n
  blks(2) = n
  blks(3) = n
  blks(4) = n

  !-------------------------------------------------------------------
  ! Compute the matrix and right-hand-side vector that define
  ! the linear system, Ax = b.
  !---------------------------------------------------------------

  ! Create and assemble parallel matrix

  PetscCallA(MatCreate(PETSC_COMM_WORLD, A, ierr))
  PetscCallA(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, m*n, m*n, ierr))
  PetscCallA(MatSetFromOptions(A, ierr))
  PetscCallA(MatMPIAIJSetPreallocation(A, five, PETSC_NULL_INTEGER_ARRAY, five, PETSC_NULL_INTEGER_ARRAY, ierr))
  PetscCallA(MatSeqAIJSetPreallocation(A, five, PETSC_NULL_INTEGER_ARRAY, ierr))
  PetscCallA(MatGetOwnershipRange(A, Istart, Iend, ierr))

  do Ii = Istart, Iend - 1
    v = -1.0; i = Ii/n; j = Ii - i*n
    if (i > 0) then
      JJ = Ii - n
      PetscCallA(MatSetValues(A, one, [Ii], one, [JJ], [v], ADD_VALUES, ierr))
    end if

    if (i < m - 1) then
      JJ = Ii + n
      PetscCallA(MatSetValues(A, one, [Ii], one, [JJ], [v], ADD_VALUES, ierr))
    end if

    if (j > 0) then
      JJ = Ii - 1
      PetscCallA(MatSetValues(A, one, [Ii], one, [JJ], [v], ADD_VALUES, ierr))
    end if

    if (j < n - 1) then
      JJ = Ii + 1
      PetscCallA(MatSetValues(A, one, [Ii], one, [JJ], [v], ADD_VALUES, ierr))
    end if

    v = 4.0
    PetscCallA(MatSetValues(A, one, [Ii], one, [Ii], [v], ADD_VALUES, ierr))

  end do

  PetscCallA(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY, ierr))
  PetscCallA(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY, ierr))

  ! Create parallel vectors

  PetscCallA(VecCreate(PETSC_COMM_WORLD, u, ierr))
  PetscCallA(VecSetSizes(u, PETSC_DECIDE, m*n, ierr))
  PetscCallA(VecSetFromOptions(u, ierr))
  PetscCallA(VecDuplicate(u, b, ierr))
  PetscCallA(VecDuplicate(b, x, ierr))

  ! Set exact solution; then compute right-hand-side vector.

  PetscCallA(Vecset(u, sone, ierr))
  PetscCallA(MatMult(A, u, b, ierr))

  ! Create linear solver context

  PetscCallA(KSPCreate(PETSC_COMM_WORLD, ksp, ierr))

  ! Set operators. Here the matrix that defines the linear system
  ! also serves as the matrix used to construct the preconditioner.

  PetscCallA(KSPSetOperators(ksp, A, A, ierr))

  ! Set default preconditioner for this program to be block Jacobi.
  ! This choice can be overridden at runtime with the option
  ! -pc_type <type>

  PetscCallA(KSPGetPC(ksp, myPc, ierr))
  PetscCallA(PCSetType(myPc, PCBJACOBI, ierr))

  ! -----------------------------------------------------------------
  !            Define the problem decomposition
  !-------------------------------------------------------------------

  ! Call PCBJacobiSetTotalBlocks() to set individually the size of
  ! each block in the preconditioner.  This could also be done with
  ! the runtime option -pc_bjacobi_blocks <blocks>
  ! Also, see the command PCBJacobiSetLocalBlocks() to set the
  ! local blocks.

  ! Note: The default decomposition is 1 block per processor.

  PetscCallA(PCBJacobiSetTotalBlocks(myPc, m, blks, ierr))

  !-------------------------------------------------------------------
  !       Set the linear solvers for the subblocks
  !-------------------------------------------------------------------

  !  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  ! Basic method, should be sufficient for the needs of most users.
  !- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  ! By default, the block Jacobi method uses the same solver on each
  ! block of the problem.  To set the same solver options on all blocks,
  ! use the prefix -sub before the usual PC and KSP options, e.g.,
  ! -sub_pc_type <pc> -sub_ksp_type <ksp> -sub_ksp_rtol 1.e-4

  !  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  !  Advanced method, setting different solvers for various blocks.
  !- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  ! Note that each block's KSP context is completely independent of
  ! the others, and the full range of uniprocessor KSP options is
  ! available for each block. The following section of code is intended
  ! to be a simple illustration of setting different linear solvers for
  ! the individual blocks.  These choices are obviously not recommended
  ! for solving this particular problem.

  PetscCallA(PetscObjectTypeCompare(myPc, PCBJACOBI, isbjacobi, ierr))

  if (isbjacobi) then

    ! Call KSPSetUp() to set the block Jacobi data structures (including
    ! creation of an internal KSP context for each block).
    ! Note: KSPSetUp() MUST be called before PCBJacobiGetSubKSP()

    PetscCallA(KSPSetUp(ksp, ierr))

    ! Extract the array of KSP contexts for the local blocks
    PetscCallA(PCBJacobiGetSubKSP(myPc, nlocal, first, PETSC_NULL_KSP_POINTER, ierr))
    PetscCallA(PCBJacobiGetSubKSP(myPc, nlocal, first, subksp, ierr))

    ! Loop over the local blocks, setting various KSP options for each block

    do i = 0, nlocal - 1

      PetscCallA(KSPGetPC(subksp(i + 1), subpc, ierr))

      if (rank > 0) then

        if (mod(i, 2) == 1) then
          PetscCallA(PCSetType(subpc, PCILU, ierr))

        else
          PetscCallA(PCSetType(subpc, PCNONE, ierr))
          PetscCallA(KSPSetType(subksp(i + 1), KSPBCGS, ierr))
          PetscCallA(KSPSetTolerances(subksp(i + 1), tol, PETSC_CURRENT_REAL, PETSC_CURRENT_REAL, PETSC_CURRENT_INTEGER, ierr))
        end if

      else
        PetscCallA(PCSetType(subpc, PCJACOBI, ierr))
        PetscCallA(KSPSetType(subksp(i + 1), KSPGMRES, ierr))
        PetscCallA(KSPSetTolerances(subksp(i + 1), tol, PETSC_CURRENT_REAL, PETSC_CURRENT_REAL, PETSC_CURRENT_INTEGER, ierr))
      end if

    end do

  end if

  !----------------------------------------------------------------
  !                Solve the linear system
  !-----------------------------------------------------------------

  ! Set runtime options

  PetscCallA(KSPSetFromOptions(ksp, ierr))

  ! Solve the linear system

  PetscCallA(KSPSolve(ksp, b, x, ierr))

  !  -----------------------------------------------------------------
  !               Check solution and clean up
  !-------------------------------------------------------------------

  !  -----------------------------------------------------------------
  ! Check the error
  !  -----------------------------------------------------------------

  !PetscCallA(VecView(x,PETSC_VIEWER_STDOUT_WORLD,ierr))

  PetscCallA(VecAXPY(x, myNone, u, ierr))

  !PetscCallA(VecView(x,PETSC_VIEWER_STDOUT_WORLD,ierr))

  PetscCallA(VecNorm(x, NORM_2, norm, ierr))
  PetscCallA(KSPGetIterationNumber(ksp, its, ierr))
  write (outputString, *) 'Norm of error', real(norm), 'Iterations', its, '\n'         ! PETScScalar might be of complex type
  PetscCallA(PetscPrintf(PETSC_COMM_WORLD, outputString, ierr))

  ! Free work space.  All PETSc objects should be destroyed when they
  ! are no longer needed.
  PetscCallA(KSPDestroy(ksp, ierr))
  PetscCallA(VecDestroy(u, ierr))
  PetscCallA(VecDestroy(b, ierr))
  PetscCallA(MatDestroy(A, ierr))
  PetscCallA(VecDestroy(x, ierr))
  PetscCallA(PetscFinalize(ierr))

end program main

!/*TEST
!
!   test:
!      nsize: 2
!      args: -ksp_monitor_short -ksp_gmres_cgs_refinement_type refine_always
!
!   test:
!      suffix: 2
!      nsize: 2
!      args: -ksp_view ::ascii_info_detail
!
!TEST*/
