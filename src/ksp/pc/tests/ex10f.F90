! Tests array outputs, omitted arguments and error returns in the PCASM Fortran bindings.
#include <petsc/finclude/petscksp.h>
program main
  use petscksp
  implicit none

  Mat A, saved_null_mat
  Mat, pointer :: saved_mat_pointer(:)
  IS, pointer :: subdomains(:) => null(), inner(:) => null(), saved_is_pointer(:)
  IS saved_null_is
  PC pc
  PetscInt i, nsub, n, imin, imax, saved_null_integer
  PetscInt, parameter :: nrows = 4
  PetscScalar, parameter :: one = 1
  PetscErrorCode ierr, expected_error, second_error
  character(len=16) :: test_case = 'create'

  PetscCallA(PetscInitialize(ierr))
  saved_mat_pointer => PETSC_NULL_MAT_POINTER
  saved_is_pointer => PETSC_NULL_IS_POINTER
  saved_null_mat = PETSC_NULL_MAT_ARRAY(1)
  saved_null_is = PETSC_NULL_IS_ARRAY(1)
  saved_null_integer = PETSC_NULL_INTEGER
  call CheckNullOutputs()
  PetscCallA(PetscOptionsGetString(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, '-case', test_case, PETSC_NULL_BOOL, ierr))
  PetscCallA(MatCreateSeqAIJ(PETSC_COMM_SELF, nrows, nrows, 1_PETSC_INT_KIND, PETSC_NULL_INTEGER_ARRAY, A, ierr))
  do i = 0, nrows - 1
    PetscCallA(MatSetValue(A, i, i, one, INSERT_VALUES, ierr))
  end do
  PetscCallA(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY, ierr))
  PetscCallA(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY, ierr))
  PetscCallA(PCCreate(PETSC_COMM_SELF, pc, ierr))
  PetscCallA(PCSetOperators(pc, A, A, ierr))
  PetscCallA(PCSetType(pc, PCASM, ierr))
  PetscCallA(PCASMSetOverlap(pc, 0_PETSC_INT_KIND, ierr))
  PetscCallA(PCSetUp(pc, ierr))
  nsub = -1
  select case (trim(test_case))
  case ('create')
    nsub = 1
    PetscCallA(PCASMCreateSubdomains(A, nsub, subdomains, ierr))
  case ('create_null')
    nsub = 1
    PetscCallA(PetscPushErrorHandler(ReturnError, PETSC_NULL_INTEGER, ierr))
    call PCASMCreateSubdomains(A, nsub, PETSC_NULL_IS_POINTER, expected_error)
    call PCASMDestroySubdomains(nsub, PETSC_NULL_IS_POINTER, PETSC_NULL_IS_POINTER, second_error)
    PetscCallA(PetscPopErrorHandler(ierr))
    PetscCheckA(expected_error == PETSC_ERR_ARG_NULL, PETSC_COMM_SELF, PETSC_ERR_PLIB, 'Creation accepted an omitted required output')
    PetscCheckA(second_error == PETSC_ERR_ARG_NULL, PETSC_COMM_SELF, PETSC_ERR_PLIB, 'Destruction accepted an omitted required array')
    call CheckNullOutputs()
    ! Creation returns no inner IS array, so omit it when destroying.
    PetscCallA(PCASMCreateSubdomains(A, nsub, subdomains, ierr))
    PetscCallA(PCASMDestroySubdomains(nsub, subdomains, PETSC_NULL_IS_POINTER, ierr))
    call CheckNullOutputs()
  case default
    SETERRA(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, 'Unknown -case value')
  end select
  PetscCheckA(nsub == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, 'Expected exactly one subdomain')
  if (test_case == 'create') then
    PetscCheckA(associated(subdomains), PETSC_COMM_SELF, PETSC_ERR_PLIB, 'PCASM left the subdomain output disassociated')
    PetscCheckA(size(subdomains) == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, 'Wrong subdomain array extent')
    PetscCallA(ISGetSize(subdomains(1), n, ierr))
    PetscCallA(ISGetMinMax(subdomains(1), imin, imax, ierr))
    PetscCheckA(n == nrows .and. imin == 0 .and. imax == nrows - 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, 'Wrong subdomain indices')
  end if
  if (test_case == 'create') then
    PetscCallA(PCASMDestroySubdomains(nsub, subdomains, inner, ierr))
  end if
  ! Getter outputs are borrowed from pc; only the create case owns its array.
  nullify (subdomains, inner)
  PetscCallA(PCDestroy(pc, ierr))
  PetscCallA(MatDestroy(A, ierr))
  call CheckNullOutputs()
  PetscCallA(PetscFinalize(ierr))

contains

  subroutine ReturnError(comm, line, fun, file, n, p, mess, ctx, ierr)
    MPIU_Comm comm
    integer line, p
    character(*) fun, file, mess
    PetscInt ctx
    PetscErrorCode n, ierr

    ierr = n
  end subroutine

  subroutine CheckNullOutputs()
    PetscCheckA(associated(PETSC_NULL_MAT_POINTER) .eqv. associated(saved_mat_pointer), PETSC_COMM_SELF, PETSC_ERR_PLIB, 'Changed null Mat association')
    if (associated(saved_mat_pointer)) then
      PetscCheckA(associated(PETSC_NULL_MAT_POINTER, saved_mat_pointer), PETSC_COMM_SELF, PETSC_ERR_PLIB, 'Changed null Mat target')
    end if
    PetscCheckA(associated(PETSC_NULL_IS_POINTER) .eqv. associated(saved_is_pointer), PETSC_COMM_SELF, PETSC_ERR_PLIB, 'Changed null IS association')
    if (associated(saved_is_pointer)) then
      PetscCheckA(associated(PETSC_NULL_IS_POINTER, saved_is_pointer), PETSC_COMM_SELF, PETSC_ERR_PLIB, 'Changed null IS target')
    end if
    PetscCheckA(PETSC_NULL_MAT_ARRAY(1) == saved_null_mat, PETSC_COMM_SELF, PETSC_ERR_PLIB, 'Changed null Mat target')
    PetscCheckA(PETSC_NULL_IS_ARRAY(1) == saved_null_is, PETSC_COMM_SELF, PETSC_ERR_PLIB, 'Changed null IS target')
    PetscCheckA(PETSC_NULL_INTEGER == saved_null_integer, PETSC_COMM_SELF, PETSC_ERR_PLIB, 'Changed null integer')
  end subroutine
end program

!/*TEST
!
!   testset:
!     nsize: 1
!     output_file: output/empty.out
!
!     test:
!       suffix: create
!       args: -case create
!     test:
!       suffix: create_null
!       args: -case create_null
!
!TEST*/
