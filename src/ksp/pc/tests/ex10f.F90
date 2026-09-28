! Tests array outputs, omitted arguments and error returns in the PCASM Fortran bindings.
#include <petsc/finclude/petscksp.h>
program main
  use petscksp
  implicit none

  Mat A, saved_null_mat
  Mat, target :: matrix_target(1)
  Mat, pointer :: submatrices(:) => null(), saved_mat_pointer(:)
  IS, pointer :: subdomains(:) => null(), inner(:) => null(), saved_is_pointer(:)
  IS saved_null_is
  IS, target :: explicit_outer(1), explicit_inner(1)
  KSP, pointer :: ksps(:) => null(), saved_ksp_pointer(:)
  KSP saved_null_ksp
  PC pc
  PetscInt i, nsub, first_sub, m, n, imin, imax, saved_null_integer
  PetscInt, parameter :: nrows = 4
  PetscScalar, parameter :: one = 1
  PetscScalar value(1)
  PetscErrorCode ierr, expected_error, second_error
  character(len=16) :: test_case = 'create'

  PetscCallA(PetscInitialize(ierr))
  saved_mat_pointer => PETSC_NULL_MAT_POINTER
  saved_is_pointer => PETSC_NULL_IS_POINTER
  saved_ksp_pointer => PETSC_NULL_KSP_POINTER
  saved_null_mat = PETSC_NULL_MAT_ARRAY(1)
  saved_null_is = PETSC_NULL_IS_ARRAY(1)
  saved_null_ksp = PETSC_NULL_KSP_ARRAY(1)
  saved_null_integer = PETSC_NULL_INTEGER
  call CheckNullOutputs()
  PetscCallA(PetscOptionsGetString(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, '-case', test_case, PETSC_NULL_BOOL, ierr))
  PetscCallA(MatCreateSeqAIJ(PETSC_COMM_SELF, nrows, nrows, 1_PETSC_INT_KIND, PETSC_NULL_INTEGER_ARRAY, A, ierr))
  do i = 0, nrows - 1
    PetscCallA(MatSetValue(A, i, i, one, INSERT_VALUES, ierr))
  end do
  PetscCallA(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY, ierr))
  PetscCallA(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY, ierr))
  matrix_target(1) = A
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
  case ('create2d')
    ! PCASMCreateSubdomains2D() always creates both arrays, so neither may be omitted.
    PetscCallA(PetscPushErrorHandler(ReturnError, PETSC_NULL_INTEGER, ierr))
    call PCASMCreateSubdomains2D(2_PETSC_INT_KIND, 2_PETSC_INT_KIND, 1_PETSC_INT_KIND, 1_PETSC_INT_KIND, 1_PETSC_INT_KIND, 0_PETSC_INT_KIND, nsub, subdomains, PETSC_NULL_IS_POINTER, expected_error)
    call PCASMCreateSubdomains2D(2_PETSC_INT_KIND, 2_PETSC_INT_KIND, 1_PETSC_INT_KIND, 1_PETSC_INT_KIND, 1_PETSC_INT_KIND, 0_PETSC_INT_KIND, nsub, PETSC_NULL_IS_POINTER, inner, second_error)
    PetscCallA(PetscPopErrorHandler(ierr))
    PetscCheckA(expected_error == PETSC_ERR_ARG_NULL .and. second_error == PETSC_ERR_ARG_NULL, PETSC_COMM_SELF, PETSC_ERR_PLIB, 'PCASMCreateSubdomains2D() accepted an omitted output')
    PetscCheckA(.not. associated(subdomains) .and. .not. associated(inner), PETSC_COMM_SELF, PETSC_ERR_PLIB, 'A rejected PCASMCreateSubdomains2D() associated its output')
    call CheckNullOutputs()
    PetscCallA(PCASMCreateSubdomains2D(2_PETSC_INT_KIND, 2_PETSC_INT_KIND, 1_PETSC_INT_KIND, 1_PETSC_INT_KIND, 1_PETSC_INT_KIND, 0_PETSC_INT_KIND, nsub, subdomains, inner, ierr))
    PetscCheckA(associated(inner), PETSC_COMM_SELF, PETSC_ERR_PLIB, 'PCASMCreateSubdomains2D() left the inner output disassociated')
    PetscCheckA(size(inner) == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, 'Wrong inner subdomain array extent')
  case ('subksp')
    first_sub = -1
    PetscCallA(PCASMGetSubKSP(pc, nsub, first_sub, PETSC_NULL_KSP_POINTER, ierr))
    call CheckNullOutputs()
    PetscCheckA(first_sub == 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, 'Omitting the KSP array lost the first block')
    PetscCallA(PCASMRestoreSubKSP(pc, nsub, first_sub, PETSC_NULL_KSP_POINTER, ierr))
    call CheckNullOutputs()
    PetscCallA(PCASMGetSubKSP(pc, PETSC_NULL_INTEGER, PETSC_NULL_INTEGER, ksps, ierr))
    PetscCheckA(associated(ksps), PETSC_COMM_SELF, PETSC_ERR_PLIB, 'PCASMGetSubKSP() left the output disassociated')
    PetscCheckA(size(ksps) == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, 'Wrong KSP array extent')
    PetscCallA(PCASMRestoreSubKSP(pc, PETSC_NULL_INTEGER, PETSC_NULL_INTEGER, ksps, ierr))
    call CheckNullOutputs()
  case ('subdomains')
    PetscCallA(PCASMGetLocalSubdomains(pc, nsub, subdomains, inner, ierr))
    PetscCheckA(.not. associated(inner), PETSC_COMM_SELF, PETSC_ERR_PLIB, 'Expected no inner subdomain array for one block')
  case ('submatrices')
    PetscCallA(PCASMGetLocalSubmatrices(pc, nsub, submatrices, ierr))
    PetscCheckA(associated(submatrices), PETSC_COMM_SELF, PETSC_ERR_PLIB, 'PCASMGetLocalSubmatrices() left the output disassociated')
    PetscCheckA(size(submatrices) == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, 'Wrong submatrix array extent')
    PetscCallA(MatGetSize(submatrices(1), m, n, ierr))
    PetscCheckA(m == nrows .and. n == nrows, PETSC_COMM_SELF, PETSC_ERR_PLIB, 'Wrong submatrix dimensions')
    PetscCallA(MatGetValues(submatrices(1), 1_PETSC_INT_KIND, [0_PETSC_INT_KIND], 1_PETSC_INT_KIND, [0_PETSC_INT_KIND], value, ierr))
    PetscCheckA(value(1) == one, PETSC_COMM_SELF, PETSC_ERR_PLIB, 'Wrong submatrix value')
  case default
    SETERRA(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, 'Unknown -case value')
  end select
  PetscCheckA(nsub == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, 'Expected exactly one subdomain')
  if (test_case == 'create' .or. test_case == 'create2d' .or. test_case == 'subdomains') then
    PetscCheckA(associated(subdomains), PETSC_COMM_SELF, PETSC_ERR_PLIB, 'PCASM left the subdomain output disassociated')
    PetscCheckA(size(subdomains) == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, 'Wrong subdomain array extent')
    PetscCallA(ISGetSize(subdomains(1), n, ierr))
    PetscCallA(ISGetMinMax(subdomains(1), imin, imax, ierr))
    PetscCheckA(n == nrows .and. imin == 0 .and. imax == nrows - 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, 'Wrong subdomain indices')
  end if
  select case (trim(test_case))
  case ('submatrices')
    nullify (submatrices)
    PetscCallA(PCASMGetLocalSubmatrices(pc, PETSC_NULL_INTEGER, submatrices, ierr))
    call CheckNullOutputs()
    PetscCheckA(associated(submatrices), PETSC_COMM_SELF, PETSC_ERR_PLIB, 'Omitting the count lost the submatrix output')
    PetscCheckA(size(submatrices) == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, 'Wrong submatrix extent with omitted count')
    nsub = -1
    PetscCallA(PCASMGetLocalSubmatrices(pc, nsub, PETSC_NULL_MAT_POINTER, ierr))
    call CheckNullOutputs()
    PetscCheckA(nsub == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, 'Omitting matrices lost the count')
    PetscCallA(PCASMGetLocalSubmatrices(pc, PETSC_NULL_INTEGER, PETSC_NULL_MAT_POINTER, ierr))
    call CheckNullOutputs()
    ! Before setup, both getters must return the C error and leave their outputs untouched.
    nullify (submatrices)
    PetscCallA(PCSetType(pc, PCNONE, ierr))
    PetscCallA(PCSetType(pc, PCASM, ierr))
    nsub = -1
    first_sub = -1
    PetscCallA(PetscPushErrorHandler(ReturnError, PETSC_NULL_INTEGER, ierr))
    call PCASMGetLocalSubmatrices(pc, nsub, submatrices, expected_error)
    call PCASMGetSubKSP(pc, nsub, first_sub, ksps, second_error)
    PetscCallA(PetscPopErrorHandler(ierr))
    PetscCheckA(expected_error == PETSC_ERR_ARG_WRONGSTATE, PETSC_COMM_SELF, PETSC_ERR_PLIB, 'PCASMGetLocalSubmatrices() lost the error before setup')
    PetscCheckA(second_error == PETSC_ERR_ORDER, PETSC_COMM_SELF, PETSC_ERR_PLIB, 'PCASMGetSubKSP() lost the error before setup')
    PetscCheckA(nsub == -1 .and. first_sub == -1, PETSC_COMM_SELF, PETSC_ERR_PLIB, 'A failed getter changed its counts')
    PetscCheckA(.not. associated(submatrices) .and. .not. associated(ksps), PETSC_COMM_SELF, PETSC_ERR_PLIB, 'A failed getter associated its output')
    call CheckNullOutputs()
    PetscCallA(PCSetType(pc, PCNONE, ierr))
    PetscCallA(PCSetUp(pc, ierr))
    submatrices => matrix_target
    PetscCallA(PCASMGetLocalSubmatrices(pc, nsub, submatrices, ierr))
    PetscCheckA(nsub == 0 .and. .not. associated(submatrices), PETSC_COMM_SELF, PETSC_ERR_PLIB, 'Empty matrix output retained an old association')
  case ('subdomains')
    inner => PETSC_NULL_IS_ARRAY
    PetscCallA(PCASMGetLocalSubdomains(pc, PETSC_NULL_INTEGER, PETSC_NULL_IS_POINTER, inner, ierr))
    call CheckNullOutputs()
    PetscCheckA(.not. associated(inner), PETSC_COMM_SELF, PETSC_ERR_PLIB, 'Absent inner IS array retained an old association')
    nullify (subdomains, inner)
    PetscCallA(PCSetType(pc, PCNONE, ierr))
    PetscCallA(PCSetType(pc, PCASM, ierr))
    ! Before setup, neither IS array exists. Ordinary pointers may alias the null arrays.
    subdomains => PETSC_NULL_IS_ARRAY
    inner => PETSC_NULL_IS_ARRAY
    PetscCallA(PCASMGetLocalSubdomains(pc, nsub, subdomains, inner, ierr))
    PetscCheckA(.not. associated(subdomains) .and. .not. associated(inner), PETSC_COMM_SELF, PETSC_ERR_PLIB, 'Absent IS outputs retained old associations')
    call CheckNullOutputs()
    PetscCallA(ISCreateStride(PETSC_COMM_SELF, nrows, 0_PETSC_INT_KIND, 1_PETSC_INT_KIND, explicit_outer(1), ierr))
    PetscCallA(ISCreateStride(PETSC_COMM_SELF, 2_PETSC_INT_KIND, 0_PETSC_INT_KIND, 1_PETSC_INT_KIND, explicit_inner(1), ierr))
    PetscCallA(PCASMSetLocalSubdomains(pc, 1_PETSC_INT_KIND, explicit_outer, explicit_inner, ierr))
    nsub = -1
    PetscCallA(PCASMGetLocalSubdomains(pc, nsub, PETSC_NULL_IS_POINTER, PETSC_NULL_IS_POINTER, ierr))
    call CheckNullOutputs()
    PetscCheckA(nsub == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, 'Omitting IS arrays lost the count')
    PetscCallA(PCASMGetLocalSubdomains(pc, PETSC_NULL_INTEGER, PETSC_NULL_IS_POINTER, PETSC_NULL_IS_POINTER, ierr))
    call CheckNullOutputs()
    ! Omit the count, then each array independently, using fresh output descriptors.
    do i = 1, 3
      nullify (subdomains, inner)
      if (i == 1) then
        PetscCallA(PCASMGetLocalSubdomains(pc, PETSC_NULL_INTEGER, subdomains, inner, ierr))
      else if (i == 2) then
        PetscCallA(PCASMGetLocalSubdomains(pc, PETSC_NULL_INTEGER, subdomains, PETSC_NULL_IS_POINTER, ierr))
      else
        PetscCallA(PCASMGetLocalSubdomains(pc, PETSC_NULL_INTEGER, PETSC_NULL_IS_POINTER, inner, ierr))
      end if
      call CheckNullOutputs()
      if (i /= 3) then
        PetscCheckA(associated(subdomains), PETSC_COMM_SELF, PETSC_ERR_PLIB, 'Omitted arguments lost the outer IS array')
        PetscCheckA(size(subdomains) == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, 'Wrong outer IS array extent')
        PetscCheckA(subdomains(1) == explicit_outer(1), PETSC_COMM_SELF, PETSC_ERR_PLIB, 'Wrong outer IS returned')
      end if
      if (i /= 2) then
        PetscCheckA(associated(inner), PETSC_COMM_SELF, PETSC_ERR_PLIB, 'Omitted arguments lost the inner IS array')
        PetscCheckA(size(inner) == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, 'Wrong inner IS array extent')
        PetscCheckA(inner(1) == explicit_inner(1), PETSC_COMM_SELF, PETSC_ERR_PLIB, 'Wrong inner IS returned')
      end if
    end do
    PetscCallA(ISDestroy(explicit_outer(1), ierr))
    PetscCallA(ISDestroy(explicit_inner(1), ierr))
  end select
  if (test_case == 'create' .or. test_case == 'create2d') then
    PetscCallA(PCASMDestroySubdomains(nsub, subdomains, inner, ierr))
  end if
  ! Getter outputs are borrowed from pc; only the create cases own their arrays.
  nullify (subdomains, inner, submatrices)
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
    PetscCheckA(associated(PETSC_NULL_KSP_POINTER) .eqv. associated(saved_ksp_pointer), PETSC_COMM_SELF, PETSC_ERR_PLIB, 'Changed null KSP association')
    if (associated(saved_ksp_pointer)) then
      PetscCheckA(associated(PETSC_NULL_KSP_POINTER, saved_ksp_pointer), PETSC_COMM_SELF, PETSC_ERR_PLIB, 'Changed null KSP target')
    end if
    PetscCheckA(PETSC_NULL_MAT_ARRAY(1) == saved_null_mat, PETSC_COMM_SELF, PETSC_ERR_PLIB, 'Changed null Mat target')
    PetscCheckA(PETSC_NULL_IS_ARRAY(1) == saved_null_is, PETSC_COMM_SELF, PETSC_ERR_PLIB, 'Changed null IS target')
    PetscCheckA(PETSC_NULL_KSP_ARRAY(1) == saved_null_ksp, PETSC_COMM_SELF, PETSC_ERR_PLIB, 'Changed null KSP target')
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
!     test:
!       suffix: create2d
!       args: -case create2d
!     test:
!       suffix: subksp
!       args: -case subksp
!     test:
!       suffix: submatrices
!       args: -case submatrices
!     test:
!       suffix: subdomains
!       args: -case subdomains
!
!TEST*/
