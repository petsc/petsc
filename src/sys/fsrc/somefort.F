!
!     Prevents: Warning: Same actual argument associated with INTENT(IN)
!     argument 'errorcode' and INTENT(OUT) argument 'ierror' at (1)
!     when MPI_Abort() is called directly by CHKERRQ(ierr);
!

      subroutine MPIU_Abort(comm,ierr)
      implicit none

#include      <../src/sys/f90-mod/petscsys.h>

      integer comm,ierr,nierr

      call MPI_Abort(comm,ierr,nierr)

      return
      end

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
