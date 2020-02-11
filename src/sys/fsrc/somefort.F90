!
!     Prevents: Warning: Same actual argument associated with INTENT(IN)
!     argument 'errorcode' and INTENT(OUT) argument 'ierror' at (1)
!     when MPI_Abort() is called directly by CHKERRQ(ierr);
!

#include <petscconf.h>
#if defined(PETSC_HAVE_MPIUNI)
#include "petsc/mpiuni/mpiunifdef.h"
#endif

#if defined(PETSC_USE_FORTRANKIND)
#define integer4 integer(kind=selected_int_kind(5))
#else
#define nteger4 integer*4
#endif

      subroutine MPIU_Abort(comm,ierr)
      implicit none
      integer4 comm
      integer4 ierr,nierr

      call MPI_Abort(comm,ierr,nierr)

      return
      end
#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::MPIU_Abort
#endif

!  This is currently not used in PETSc
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
