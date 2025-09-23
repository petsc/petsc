!
!     Prevents: Warning: Same actual argument associated with INTENT(IN)
!     argument 'errorcode' and INTENT(OUT) argument 'ierror' at (1)
!     when MPI_Abort() is called directly
!

#include <petsc/finclude/petscsys.h>
      subroutine MPIU_Abort(comm, ierr)
        use, intrinsic :: ISO_C_binding
        implicit none
        MPI_Comm comm
        PetscMPIInt ierr, nierr, ciportable
        call PetscCIEnabledPortableErrorOutput(ciportable)
        if (ciportable == 1) then
          call MPI_Finalize(nierr)
          stop 0
        else
          call MPI_Abort(comm, ierr, nierr)
        end if
      end
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::MPIU_Abort
#endif

      subroutine PetscFortranPrintToFileUnit(unit, str, ierr)
        use, intrinsic :: ISO_C_binding
        implicit none
        character(*) str
        integer4 unit
        PetscErrorCode ierr
        write (unit=unit, fmt="(A)", advance='no') str
        ierr = 0
      end
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::PetscFortranPrintToFileUnit
#endif

!  This uses F2003 feature - and is the preferred mode for accessing command line arguments
      integer function PetscCommandArgumentCount()
        use, intrinsic :: ISO_C_binding
        implicit none
        PetscCommandArgumentCount = command_argument_count()
      end

      subroutine PetscGetCommandArgument(n, val)
        implicit none
        integer, intent(in) :: n
        character(*) val
        call get_command_argument(n, val)
      end
