!
!     Prevents: Warning: Same actual argument associated with INTENT(IN)
!     argument 'errorcode' and INTENT(OUT) argument 'ierror' at (1)
!     when MPI_Abort() is called directly
!

#include <petsc/finclude/petscsys.h>
subroutine MPIU_Abort(comm, ierr)
#if defined(PETSC_DLL_EXPORTS)
!DEC$ ATTRIBUTES DLLEXPORT::MPIU_Abort
#endif
  use, intrinsic :: ISO_C_binding
  use petscmpi
  implicit none
  MPIU_Comm comm
  PetscMPIInt ierr, nierr, ciportable
  call PetscCIEnabledPortableErrorOutput(ciportable)
  if (ciportable == 1) then
    call MPI_Finalize(nierr)
    stop 0
  else
    call MPI_Abort(comm, ierr, nierr)
  end if
end

subroutine PetscFortranPrintToFileUnit(unit, str, ierr)
#if defined(PETSC_DLL_EXPORTS)
!DEC$ ATTRIBUTES DLLEXPORT::PetscFortranPrintToFileUnit
#endif
  use, intrinsic :: ISO_C_binding
  implicit none
  character(*) str
  integer4, intent(in) :: unit
  PetscErrorCode, intent(out) :: ierr
  write (unit=unit, fmt="(A)", advance='no') str
  ierr = 0
end

integer pure function PetscCommandArgumentCount()
  use, intrinsic :: ISO_C_binding
  implicit none
  PetscCommandArgumentCount = command_argument_count()
end

subroutine PetscGetCommandArgument(n, val)
  implicit none
  integer, intent(in) :: n
  character(*), intent(out) :: val
  call get_command_argument(n, val)
end
