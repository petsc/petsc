!
!

    module mymoduleex43f
#include <petsc/finclude/petscvec.h>
      use iso_c_binding
      interface
        subroutine fillupvector(vaddr,ierr) bind ( C, name = "fillupvector")
!
!     We need to use iso_c_binding variables or otherwise we get compiler warnings
!     Warning: Variable 'vaddr' at (1) is a dummy argument of the BIND(C)
!              procedure 'fillupvector' but may not be C interoperable
!
          use iso_c_binding
          integer(c_long_long) vaddr
          integer(c_int) ierr
        end subroutine fillupvector
      end interface
    end module

#include <petsc/finclude/petscvec.h>
        use iso_c_binding
        use petscvec
        use mymoduleex43f
       implicit none
!
!  This routine demonstates how to call a bind C function from Fortran
       Vec            v
       PetscErrorCode ierr
       PetscInt five
!
!     We need to use the same iso_c_binding variable types here or some compilers
!     will see a type mismatch in the call to fillupvector and thus not link
!
       integer(c_long_long) vaddr
       integer(c_int) err

       call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
       call VecCreate(PETSC_COMM_WORLD,v,ierr);CHKERRA(ierr)
       five = 5
       call VecSetSizes(v,PETSC_DECIDE,five,ierr);CHKERRA(ierr)
       call VecSetFromOptions(v,ierr);CHKERRA(ierr)
!
!     Now Call a Petsc Routine from Fortran
!
!
       vaddr = v%v
       call fillupvector(vaddr,err);CHKERRA(ierr)

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
