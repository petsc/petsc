!
!   Tests PCMGSetResidual() and PCMGSetResidualTranspose(), including on a MATSHELL with a Fortran MatMult()
!
! -----------------------------------------------------------------------
#include <petsc/finclude/petscksp.h>
module ex8fmodule
  use petscksp
  implicit none
  PetscInt :: nmult = 0, nresidual = 0, nresidualt = 0

contains
  subroutine MyMult(S, x, y, ierr)
    Mat S
    Vec x, y
    PetscErrorCode, intent(out) :: ierr
    nmult = nmult + 1
    PetscCall(VecCopy(x, y, ierr))
  end

  subroutine MyResidual(A, b, x, r, ierr)
    Mat A
    Vec b, x, r
    PetscErrorCode, intent(out) :: ierr
    PetscScalar, parameter :: mone = -1.0
    nresidual = nresidual + 1
    PetscCall(MatMult(A, x, r, ierr))
    PetscCall(VecAYPX(r, mone, b, ierr))
  end

  subroutine MyResidualTranspose(A, b, x, r, ierr)
    Mat A
    Vec b, x, r
    PetscErrorCode, intent(out) :: ierr
    PetscScalar, parameter :: mone = -1.0
    nresidualt = nresidualt + 1
    PetscCall(MatMultTranspose(A, x, r, ierr))
    PetscCall(VecAYPX(r, mone, b, ierr))
  end

end module ex8fmodule

program main
  use petscksp
  use ex8fmodule
  implicit none

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                   Variable declarations
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
!  Variables:
!     ksp     - linear solver context
!     x, b, u  - approx solution, right-hand side, exact solution vectors
!     A        - matrix that defines linear system
!     its      - iterations for convergence
!     norm     - norm of error in solution
!     rctx     - random number context
!

  Mat A, S, P
  Vec x, b, u
  PC pc
  PetscInt, parameter :: n = 6, dim = n**2, dimc = dim/2
  PetscInt i, j, jj, ii, istart, iend
  PetscErrorCode ierr
  PetscScalar v
  PetscScalar, parameter :: pfive = .5
  KSP ksp, sksp
  PC spc

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                 Beginning of program
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  PetscCallA(PetscInitialize(ierr))

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!      Compute the matrix and right-hand-side vector that define
!      the linear system, Ax = b.
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!  Create parallel matrix, specifying only its global dimensions.
!  When using MatCreate(), the matrix format can be specified at
!  runtime. Also, the parallel partitioning of the matrix is
!  determined by PETSc at runtime.

  PetscCallA(MatCreate(PETSC_COMM_WORLD, A, ierr))
  PetscCallA(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, dim, dim, ierr))
  PetscCallA(MatSetFromOptions(A, ierr))
  PetscCallA(MatSetUp(A, ierr))

!  Currently, all PETSc parallel matrix formats are partitioned by
!  contiguous chunks of rows across the processors.  Determine which
!  rows of the matrix are locally owned.

  PetscCallA(MatGetOwnershipRange(A, Istart, Iend, ierr))

!  Set matrix elements in parallel.
!   - Each processor needs to insert only elements that it owns
!     locally (but any non-local elements will be sent to the
!     appropriate processor during matrix assembly).
!   - Always specify global rows and columns of matrix entries.

  do II = Istart, Iend - 1
    v = -1.0
    i = II/n
    j = II - i*n
    if (i > 0) then
      JJ = II - n
      PetscCallA(MatSetValues(A, 1_PETSC_INT_KIND, [II], 1_PETSC_INT_KIND, [JJ], [v], ADD_VALUES, ierr))
    end if
    if (i < n - 1) then
      JJ = II + n
      PetscCallA(MatSetValues(A, 1_PETSC_INT_KIND, [II], 1_PETSC_INT_KIND, [JJ], [v], ADD_VALUES, ierr))
    end if
    if (j > 0) then
      JJ = II - 1
      PetscCallA(MatSetValues(A, 1_PETSC_INT_KIND, [II], 1_PETSC_INT_KIND, [JJ], [v], ADD_VALUES, ierr))
    end if
    if (j < n - 1) then
      JJ = II + 1
      PetscCallA(MatSetValues(A, 1_PETSC_INT_KIND, [II], 1_PETSC_INT_KIND, [JJ], [v], ADD_VALUES, ierr))
    end if
    v = 4.0
    PetscCallA(MatSetValues(A, 1_PETSC_INT_KIND, [II], 1_PETSC_INT_KIND, [II], [v], ADD_VALUES, ierr))
  end do

!  Assemble matrix, using the 2-step process:
!       MatAssemblyBegin(), MatAssemblyEnd()
!  Computations can be done while messages are in transition
!  by placing code between these two statements.

  PetscCallA(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY, ierr))
  PetscCallA(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY, ierr))

!  Create parallel vectors.
!   - Here, the parallel partitioning of the vector is determined by
!     PETSc at runtime.  We could also specify the local dimensions
!     if desired.
!   - Note: We form 1 vector from scratch and then duplicate as needed.

  PetscCallA(VecCreate(PETSC_COMM_WORLD, u, ierr))
  PetscCallA(VecSetSizes(u, PETSC_DECIDE, dim, ierr))
  PetscCallA(VecSetFromOptions(u, ierr))
  PetscCallA(VecDuplicate(u, b, ierr))
  PetscCallA(VecDuplicate(b, x, ierr))

!  Set exact solution; then compute right-hand-side vector.

  PetscCallA(VecSet(u, pfive, ierr))
  PetscCallA(MatMult(A, u, b, ierr))

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!         Create the linear solver and set various options
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

!  Create linear solver context

  PetscCallA(KSPCreate(PETSC_COMM_WORLD, ksp, ierr))
  PetscCallA(KSPGetPC(ksp, pc, ierr))
  PetscCallA(PCSetType(pc, PCMG, ierr))
  PetscCallA(PCMGSetLevels(pc, 2_PETSC_INT_KIND, PETSC_NULL_MPI_COMM, ierr))
  PetscCallA(PCMGSetGalerkin(pc, PC_MG_GALERKIN_BOTH, ierr))

!  Piecewise constant interpolation from pairs of fine points

  PetscCallA(MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, dim, dimc, 1_PETSC_INT_KIND, PETSC_NULL_INTEGER_ARRAY, 1_PETSC_INT_KIND, PETSC_NULL_INTEGER_ARRAY, P, ierr))
  PetscCallA(MatGetOwnershipRange(P, Istart, Iend, ierr))
  v = 1.0
  do II = Istart, Iend - 1
    JJ = II/2
    PetscCallA(MatSetValues(P, 1_PETSC_INT_KIND, [II], 1_PETSC_INT_KIND, [JJ], [v], INSERT_VALUES, ierr))
  end do
  PetscCallA(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY, ierr))
  PetscCallA(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY, ierr))
  PetscCallA(PCMGSetInterpolation(pc, 1_PETSC_INT_KIND, P, ierr))

!  The residual routines set on a MATSHELL must not replace its Fortran MatMult

  PetscCallA(MatCreateShell(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, dim, dim, PETSC_NULL_INTEGER, S, ierr))
  PetscCallA(MatShellSetOperation(S, MATOP_MULT, MyMult, ierr))
  PetscCallA(PCMGSetResidual(pc, 1_PETSC_INT_KIND, MyResidual, S, ierr))
  PetscCallA(PCMGSetResidualTranspose(pc, 1_PETSC_INT_KIND, MyResidualTranspose, S, ierr))
  PetscCallA(MatMult(S, u, x, ierr))
  PetscCheckA(nmult == 1, PETSC_COMM_WORLD, PETSC_ERR_PLIB, 'The MatMult of the MATSHELL was not called')

  PetscCallA(PCMGSetResidual(pc, 1_PETSC_INT_KIND, MyResidual, A, ierr))
  PetscCallA(PCMGSetResidualTranspose(pc, 1_PETSC_INT_KIND, MyResidualTranspose, A, ierr))

!  Smoothers whose transpose can be applied

  PetscCallA(PCMGGetSmoother(pc, 1_PETSC_INT_KIND, sksp, ierr))
  PetscCallA(KSPSetType(sksp, KSPRICHARDSON, ierr))
  PetscCallA(KSPGetPC(sksp, spc, ierr))
  PetscCallA(PCSetType(spc, PCJACOBI, ierr))

!  Set operators. Here the matrix that defines the linear system
!  also serves as the matrix used to construct the preconditioner.

  PetscCallA(KSPSetOperators(ksp, A, A, ierr))
  PetscCallA(KSPSetUp(ksp, ierr))

!  Both residual routines must be called by the cycles

  PetscCallA(PCApply(pc, b, x, ierr))
  PetscCheckA(nresidual > 0, PETSC_COMM_WORLD, PETSC_ERR_PLIB, 'The residual routine was not called')
  PetscCheckA(nresidualt == 0, PETSC_COMM_WORLD, PETSC_ERR_PLIB, 'The transposed residual routine was called by PCApply()')
  PetscCallA(PCApplyTranspose(pc, b, x, ierr))
  PetscCheckA(nresidualt > 0, PETSC_COMM_WORLD, PETSC_ERR_PLIB, 'The transposed residual routine was not called')

  PetscCallA(MatDestroy(P, ierr))
  PetscCallA(KSPDestroy(ksp, ierr))
  PetscCallA(VecDestroy(u, ierr))
  PetscCallA(VecDestroy(x, ierr))
  PetscCallA(VecDestroy(b, ierr))
  PetscCallA(MatDestroy(A, ierr))
  PetscCallA(MatDestroy(S, ierr))

  PetscCallA(PetscFinalize(ierr))
end

!/*TEST
!
!   test:
!      nsize: 1
!      output_file: output/empty.out
!TEST*/
