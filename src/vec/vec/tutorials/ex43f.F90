
    module ex43fmodule
#include <petsc/finclude/petscvec.h>
      use,intrinsic :: iso_c_binding
      interface
        subroutine fillupvector(vaddr,err) bind ( C, name = "fillupvector")
!
!     We need to use iso_c_binding variables or otherwise we get compiler warnings
!     Warning: Variable 'vaddr' at (1) is a dummy argument of the BIND(C)
!              procedure 'fillupvector' but may not be C interoperable
!
          use,intrinsic :: iso_c_binding
          integer(c_long_long) vaddr
          integer(c_int) err
        end subroutine fillupvector
      end interface
    end module

#include <petsc/finclude/petscvec.h>
        use,intrinsic :: iso_c_binding
        use petscvec
        use ex43fmodule
       implicit none
!
!  This routine demonstrates how to call a bind C function from Fortran
       Vec            v
       PetscErrorCode ierr
       PetscInt five
!
!     We need to use the same iso_c_binding variable types here or some compilers
!     will see a type mismatch in the call to fillupvector and thus not link
!
       integer(c_long_long) vaddr
       integer(c_int) err

       PetscCallA(PetscInitialize(ierr))
       PetscCallA(VecCreate(PETSC_COMM_WORLD,v,ierr))
       five = 5
       PetscCallA(VecSetSizes(v,PETSC_DECIDE,five,ierr))
       PetscCallA(VecSetFromOptions(v,ierr))
!
!     Now Call a Petsc Routine from Fortran
!
!
       vaddr = v%v
       call fillupvector(vaddr,err)

       PetscCallA(VecView(v,PETSC_VIEWER_STDOUT_WORLD,ierr))
       PetscCallA(VecDestroy(v,ierr))
       PetscCallA(PetscFinalize(ierr))
       end

!/*TEST
!
!   build:
!     depends: ex43.c
!
!   test:
!
!TEST*/
