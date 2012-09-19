

#include "finclude/petscdef.h"
      use petsc
      implicit none

      PetscErrorCode                            :: ierr
      Character(len=99) list1(6)
      PetscEnum                                 :: opt=-1
      PetscBool                                 :: set=PETSC_FALSE

      Call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      list1(1) = 'a123'
      list1(2) = 'b456'
      list1(3) = 'c789'
      list1(4) = 'list1'
      list1(5) = 'prefix_'
      list1(6) = ''

      print*, list1(1)
      call PetscOptionsGetEnum('joe_','-jeff',list1,opt,set,ierr)
      write(*,*) 'opt is ', opt
      write(*,*) 'set is ', set

      Call PetscFinalize(ierr)
      end




