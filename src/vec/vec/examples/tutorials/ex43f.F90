!
!

    module mymodule
#include <petsc/finclude/petscvec.h>
      use iso_c_binding
      interface
        subroutine fillupvector(vaddr,ierr) bind ( C, name = "fillupvector" )
        use iso_c_binding
          integer(c_long_long) vaddr
          integer(c_int) ierr
        end subroutine fillupvector
      end interface
    end module

#include <petsc/finclude/petscvec.h>
        use petscvec
        use mymodule
       implicit none
!
!  This routine demonstates how to call a C function from Fortran
       Vec            v
       PetscErrorCode ierr
       PetscInt five

       call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
       call VecCreate(PETSC_COMM_WORLD,v,ierr);CHKERRA(ierr)
       five = 5
       call VecSetSizes(v,PETSC_DECIDE,five,ierr);CHKERRA(ierr)
       call VecSetFromOptions(v,ierr);CHKERRA(ierr)
!
!  Now Call a Petsc Routine from Fortran
!
       call fillupvector(v%v,ierr);CHKERRA(ierr)

       call VecView(v,PETSC_VIEWER_STDOUT_WORLD,ierr);CHKERRA(ierr)
       call VecDestroy(v,ierr);CHKERRA(ierr)
       call PetscFinalize(ierr)
       end


!/*TEST
!
!   build:
!     depends: ex43.c
!
!   test:
!
!TEST*/
