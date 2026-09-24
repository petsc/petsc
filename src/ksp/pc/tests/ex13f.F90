!
!   Tests PCASMWeightedGetScaling() from Fortran with an output pointer that starts disassociated,
!   and PCASMWeightedSetComputeScaling() with a Fortran callback and context
!
! -----------------------------------------------------------------------
#include <petsc/finclude/petscksp.h>

! Fills the weights of the only local subdomain with the value passed as context
subroutine FillScaling(pc, local, scaling, value, ierr)
  use petscksp
  implicit none

  PC pc
  PetscInt local
  Vec scaling
  PetscScalar value
  PetscErrorCode ierr

  PetscCheck(local == 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, 'Expected a single local subdomain')
  PetscCall(VecSet(scaling, value, ierr))
end subroutine

program main
  use petscksp
  implicit none

  PC pc
  Mat A
  Vec x, y, supplied(1)
  Vec, pointer :: scaling(:) => null()
  PetscInt n, m, i, istart, iend
  PetscInt, parameter :: nlocal = 4
  PetscReal norm
  PetscScalar total, three
  PetscScalar, parameter :: one = 1.0, two = 2.0
  PetscErrorCode ierr
  external FillScaling

  PetscCallA(PetscInitialize(ierr))
  PetscCallA(PCCreate(PETSC_COMM_WORLD, pc, ierr))
  PetscCallA(PCSetType(pc, PCASM, ierr))
  PetscCallA(PCASMSetType(pc, PC_ASM_WEIGHTED, ierr))

  ! No weights yet: the count is zero and the pointer stays disassociated
  PetscCallA(PCASMWeightedGetScaling(pc, n, scaling, ierr))
  PetscCheckA(n == 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, 'Expected no scaling vectors before PCASMWeightedSetScaling()')
  PetscCheckA(.not. associated(scaling), PETSC_COMM_SELF, PETSC_ERR_PLIB, 'Expected a disassociated pointer before PCASMWeightedSetScaling()')

  ! Identity operator with one subdomain per process and no overlap
  PetscCallA(MatCreateAIJ(PETSC_COMM_WORLD, nlocal, nlocal, PETSC_DETERMINE, PETSC_DETERMINE, 1_PETSC_INT_KIND, PETSC_NULL_INTEGER_ARRAY, 0_PETSC_INT_KIND, PETSC_NULL_INTEGER_ARRAY, A, ierr))
  PetscCallA(MatGetOwnershipRange(A, istart, iend, ierr))
  do i = istart, iend - 1
    PetscCallA(MatSetValue(A, i, i, one, INSERT_VALUES, ierr))
  end do
  PetscCallA(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY, ierr))
  PetscCallA(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY, ierr))
  PetscCallA(PCSetOperators(pc, A, A, ierr))
  PetscCallA(PCASMSetOverlap(pc, 0_PETSC_INT_KIND, ierr))
  PetscCallA(PCSetUp(pc, ierr))

  ! Supply one scaling vector; pc keeps its own reference
  PetscCallA(MatGetLocalSize(A, m, PETSC_NULL_INTEGER, ierr))
  PetscCallA(VecCreateSeq(PETSC_COMM_SELF, m, supplied(1), ierr))
  PetscCallA(VecSet(supplied(1), two, ierr))
  PetscCallA(PCASMWeightedSetScaling(pc, 1_PETSC_INT_KIND, supplied, ierr))
  PetscCallA(VecDestroy(supplied(1), ierr))

  ! The getter must associate a pointer that has never been associated
  PetscCallA(PCASMWeightedGetScaling(pc, n, scaling, ierr))
  PetscCheckA(n == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, 'Expected one scaling vector')
  PetscCheckA(associated(scaling), PETSC_COMM_SELF, PETSC_ERR_PLIB, 'PCASMWeightedGetScaling() left the output pointer disassociated')
  PetscCheckA(size(scaling) == 1 .and. lbound(scaling, 1) == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, 'Wrong scaling array bounds')
  PetscCallA(VecSum(scaling(1), total, ierr))
  PetscCheckA(abs(total - two*m) < PETSC_SMALL, PETSC_COMM_SELF, PETSC_ERR_PLIB, 'Wrong scaling vector contents')

  ! The count may be omitted
  PetscCallA(PCASMWeightedGetScaling(pc, PETSC_NULL_INTEGER, scaling, ierr))
  PetscCheckA(associated(scaling) .and. size(scaling) == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, 'PCASMWeightedGetScaling() failed with PETSC_NULL_INTEGER')

  ! With an identity operator and no overlap the preconditioner is the scaling itself
  PetscCallA(MatCreateVecs(A, x, y, ierr))
  PetscCallA(VecSet(x, one, ierr))
  PetscCallA(PCApply(pc, x, y, ierr))
  PetscCallA(VecAXPY(y, -two, x, ierr))
  PetscCallA(VecNorm(y, NORM_INFINITY, norm, ierr))
  PetscCheckA(norm < PETSC_SMALL, PETSC_COMM_SELF, PETSC_ERR_PLIB, 'PCApply() did not use the supplied weights')

  ! PCReset() releases the weights and the getter clears the associated pointer
  PetscCallA(PCReset(pc, ierr))
  PetscCallA(PCASMWeightedGetScaling(pc, n, scaling, ierr))
  PetscCheckA(n == 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, 'Expected no scaling vectors after PCReset()')
  PetscCheckA(.not. associated(scaling), PETSC_COMM_SELF, PETSC_ERR_PLIB, 'Expected a disassociated pointer after PCReset()')

  ! A callback registered before setup computes the weights, with its context passed through
  three = 3.0
  PetscCallA(PCASMWeightedSetComputeScaling(pc, FillScaling, three, ierr))
  PetscCallA(PCSetOperators(pc, A, A, ierr))
  PetscCallA(PCSetUp(pc, ierr))
  PetscCallA(PCASMWeightedGetScaling(pc, n, scaling, ierr))
  PetscCheckA(n == 1 .and. associated(scaling), PETSC_COMM_SELF, PETSC_ERR_PLIB, 'PCASMWeightedSetComputeScaling() did not create the weights')
  PetscCallA(PCApply(pc, x, y, ierr))
  PetscCallA(VecAXPY(y, -three, x, ierr))
  PetscCallA(VecNorm(y, NORM_INFINITY, norm, ierr))
  PetscCheckA(norm < PETSC_SMALL, PETSC_COMM_SELF, PETSC_ERR_PLIB, 'PCApply() did not use the computed weights')

  ! PETSC_NULL_FUNCTION disables the callback and keeps the computed weights
  PetscCallA(PCASMWeightedSetComputeScaling(pc, PETSC_NULL_FUNCTION, 0, ierr))
  PetscCallA(PCASMWeightedGetScaling(pc, n, scaling, ierr))
  PetscCheckA(n == 1 .and. associated(scaling), PETSC_COMM_SELF, PETSC_ERR_PLIB, 'Disabling the callback discarded the weights')

  PetscCallA(VecDestroy(x, ierr))
  PetscCallA(VecDestroy(y, ierr))
  PetscCallA(MatDestroy(A, ierr))
  PetscCallA(PCDestroy(pc, ierr))
  PetscCallA(PetscFinalize(ierr))
end

!/*TEST
!
!   test:
!      nsize: {{1 2}}
!      output_file: output/empty.out
!TEST*/
