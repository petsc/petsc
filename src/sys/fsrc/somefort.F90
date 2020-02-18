!
!     Prevents: Warning: Same actual argument associated with INTENT(IN)
!     argument 'errorcode' and INTENT(OUT) argument 'ierror' at (1)
!     when MPI_Abort() is called directly by CHKERRQ(ierr);
!

#include <petsc/finclude/petscsys.h>

      subroutine MPIU_Abort(comm,ierr)
      implicit none
      MPI_Comm comm
      PetscMPIInt ierr,nierr

      call MPI_Abort(comm,ierr,nierr)

      return
      end
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::MPIU_Abort
#endif

!  This uses F2003 feature - and is the preferred mode for accessing command line arguments
#if defined(PETSC_HAVE_FORTRAN_GET_COMMAND_ARGUMENT)
      integer function PetscCommandArgumentCount()
      implicit none
      PetscCommandArgumentCount= command_argument_count()
      return
      end

      subroutine PetscGetCommandArgument(n,val)
      implicit none
      integer n
      character(*) val
      call get_command_argument(n,val)
      return
      end
#endif
