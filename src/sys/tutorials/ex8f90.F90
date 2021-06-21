!
!   Example of getting an enum value from the options database in Fortran

#include "petsc/finclude/petsc.h"
      use petsc
      implicit none

      PetscErrorCode                            :: ierr
      Character(len=99) list1(6)
      PetscEnum                                 :: opt=-1
      PetscBool                                 :: set=PETSC_FALSE

      Call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then
        print*,'Unable to initialize PETSc'
        stop
      endif
      list1(1) = 'a123'
      list1(2) = 'b456'
      list1(3) = 'c789'
      list1(4) = 'list1'
      list1(5) = 'prefix_'
      list1(6) = ''

      write(*,20) list1(1)
20    format(A99)
      call PetscOptionsGetEnum(PETSC_NULL_OPTIONS,'joe_','-jeff',list1,opt,set,ierr);CHKERRA(ierr)
      write(*,*) 'opt is ', opt
      write(*,*) 'set is ', set

      Call PetscFinalize(ierr)
      end

!
!/*TEST
!
!   build:
!      requires: define(PETSC_USING_F2003) define(PETSC_USING_F90FREEFORM)
!
!   test:
!      args: -joe_jeff b456
!
!TEST*/
