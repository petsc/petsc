
        module petscsysdefdummy
#include <petscconf.h>
#if defined(PETSC_HAVE_MPIUNI)
        use mpiuni
#define PETSC_AVOID_MPIF_H
#elif defined(PETSC_HAVE_MPI_F90MODULE)
        use mpi
#define PETSC_AVOID_MPIF_H
#endif
#include <../src/sys/f90-mod/petscsys.h>
#include <../src/sys/f90-mod/petscdraw.h>
#include <../src/sys/f90-mod/petscviewer.h>
        end module

        module petscsysdef
        use petscsysdefdummy
        interface operator(.ne.)
          function petscviewernotequal(A,B)
            use petscsysdefdummy
            logical petscviewernotequal
            type(tPetscViewer), intent(in) :: A,B
          end function
        end interface operator (.ne.)
        interface operator(.eq.)
          function petscviewerequals(A,B)
            use petscsysdefdummy
            logical petscviewerequals
            type(tPetscViewer), intent(in) :: A,B
          end function
        end interface operator (.eq.)

        interface operator(.ne.)
        function petscrandomnotequal(A,B)
          use petscsysdefdummy
          logical petscrandomnotequal
          type(tPetscRandom), intent(in) :: A,B
        end function
        end interface operator (.ne.)
        interface operator(.eq.)
        function petscrandomequals(A,B)
          use petscsysdefdummy
          logical petscrandomequals
          type(tPetscRandom), intent(in) :: A,B
        end function
        end interface operator (.eq.)
        end module

        function petscviewernotequal(A,B)
          use petscsysdefdummy
          logical petscviewernotequal
          type(tPetscViewer), intent(in) :: A,B
          petscviewernotequal = (A%v .ne. B%v)
        end function
        function petscviewerequals(A,B)
          use petscsysdefdummy
          logical petscviewerequals
          type(tPetscViewer), intent(in) :: A,B
          petscviewerequals = (A%v .eq. B%v)
        end function

        function petscrandomnotequal(A,B)
          use petscsysdefdummy
          logical petscrandomnotequal
          type(tPetscRandom), intent(in) :: A,B
          petscrandomnotequal = (A%v .ne. B%v)
        end function
        function petscrandomequals(A,B)
          use petscsysdefdummy
          logical petscrandomequals
          type(tPetscRandom), intent(in) :: A,B
          petscrandomequals = (A%v .eq. B%v)
        end function

        module petscsys
        use iso_c_binding
        use petscsysdef
        MPI_Comm PETSC_COMM_SELF,PETSC_COMM_WORLD
#include <../src/sys/f90-mod/petscsys.h90>
        interface
#include <../src/sys/f90-mod/ftn-auto-interfaces/petscsys.h90>
        end interface

        end module

        subroutine PetscSetCOMM(c1,c2)
        use petscsys
        implicit none
        MPI_Comm c1,c2

        PETSC_COMM_WORLD    = c1
        PETSC_COMM_SELF     = c2
        return
        end

        subroutine PetscGetCOMM(c1)
        use petscsys
        implicit none
        MPI_Comm c1

        c1 = PETSC_COMM_WORLD
        return
        end

