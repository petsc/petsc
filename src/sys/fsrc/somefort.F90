!
!     Prevents: Warning: Same actual argument associated with INTENT(IN)
!     argument 'errorcode' and INTENT(OUT) argument 'ierror' at (1)
!     when MPI_Abort() is called directly
!

#include <petsc/finclude/petscsys.h>

      subroutine MPIU_Abort(comm,ierr)
      implicit none
      MPI_Comm comm
      PetscMPIInt ierr,nierr
      call MPI_Abort(comm,ierr,nierr)
      end
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::MPIU_Abort
#endif

      subroutine PetscFortranPrintToUnit(unit,str,ierr)
      implicit none
      character(*) str
      PetscInt unit
      PetscErrorCode ierr
      write(unit=unit, fmt="(A)", advance='no') str
      ierr = 0
      end
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::PetscFortranPrintToUnit
#endif

!  This uses F2003 feature - and is the preferred mode for accessing command line arguments
      integer function PetscCommandArgumentCount()
      implicit none
      PetscCommandArgumentCount= command_argument_count()
      end

      subroutine PetscGetCommandArgument(n,val)
      implicit none
      integer, intent(in) :: n
      character(*) val
      call get_command_argument(n,val)
      end
