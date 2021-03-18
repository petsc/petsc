
        module petscisdefdummy
        use petscsysdef
#include <../src/vec/f90-mod/petscis.h>
#include <../src/vec/f90-mod/petscao.h>
        end module petscisdefdummy

        module petscisdef
        use petscisdefdummy
        interface operator(.ne.)
          function isnotequal(A,B)
            import tIs
            logical isnotequal
            type(tIS), intent(in) :: A,B
          end function
          function petscsfnotequal(A,B)
            import tPetscSF
            logical petscsfnotequal
            type(tPetscSF), intent(in) :: A,B
          end function
        end interface operator (.ne.)
        interface operator(.eq.)
          function isequals(A,B)
            import tIs
            logical isequals
            type(tIS), intent(in) :: A,B
          end function
          function petscsfequals(A,B)
            import tPetscSF
            logical petscsfequals
            type(tPetscSF), intent(in) :: A,B
          end function
        end interface operator (.eq.)
        end module

        function isnotequal(A,B)
          use petscisdefdummy, only: tIS
          logical isnotequal
          type(tIS), intent(in) :: A,B
          isnotequal = (A%v .ne. B%v)
        end function

        function isequals(A,B)
          use petscisdefdummy, only: tIS
          logical isequals
          type(tIS), intent(in) :: A,B
          isequals = (A%v .eq. B%v)
        end function

        function petscsfnotequal(A,B)
          use petscisdefdummy, only: tPetscSF
          logical petscsfnotequal
          type(tPetscSF), intent(in) :: A,B
          petscsfnotequal = (A%v .ne. B%v)
        end function

        function petscsfequals(A,B)
          use petscisdefdummy, only: tPetscSF
          logical petscsfequals
          type(tPetscSF), intent(in) :: A,B
          petscsfequals = (A%v .eq. B%v)
        end function

        module  petscaodef
        use petscisdef
        use petscsysdef
#include <../src/vec/f90-mod/petscao.h>
        end module

        module petscvecdefdummy
        use petscisdef
        use petscaodef
#include <../src/vec/f90-mod/petscvec.h>
        end module

        module petscvecdef
        use petscvecdefdummy
        interface operator(.ne.)
          function vecnotequal(A,B)
            import tVec
            logical vecnotequal
            type(tVec), intent(in) :: A,B
          end function
          function vecscatternotequal(A,B)
            import tVecScatter
            logical vecscatternotequal
            type(tVecScatter), intent(in) :: A,B
          end function
        end interface operator (.ne.)
        interface operator(.eq.)
          function vecequals(A,B)
            import tVec
            logical vecequals
            type(tVec), intent(in) :: A,B
          end function
          function vecscatterequals(A,B)
            import tVecScatter
            logical vecscatterequals
            type(tVecScatter), intent(in) :: A,B
          end function
        end interface operator (.eq.)
        end module

        function vecnotequal(A,B)
          use petscvecdefdummy, only: tVec
          logical vecnotequal
          type(tVec), intent(in) :: A,B
          vecnotequal = (A%v .ne. B%v)
        end function

        function vecequals(A,B)
          use petscvecdefdummy, only: tVec
          logical vecequals
          type(tVec), intent(in) :: A,B
          vecequals = (A%v .eq. B%v)
        end function

        function vecscatternotequal(A,B)
          use petscvecdefdummy, only: tVecScatter
          logical vecscatternotequal
          type(tVecScatter), intent(in) :: A,B
          vecscatternotequal = (A%v .ne. B%v)
        end function

        function vecscatterequals(A,B)
          use petscvecdefdummy, only: tVecScatter
          logical vecscatterequals
          type(tVecScatter), intent(in) :: A,B
          vecscatterequals = (A%v .eq. B%v)
        end function

        module petscis
        use petscisdef
        use petscsys
#include <../src/vec/f90-mod/petscis.h90>
        interface
#include <../src/vec/f90-mod/ftn-auto-interfaces/petscis.h90>
        end interface
        end module

        module petscao
        use petscis
        use petscaodef
        end module

        module petscvec
        use petscvecdef
        use petscis
        use petscao
#include <../src/vec/f90-mod/petscvec.h90>
        interface
#include <../src/vec/f90-mod/ftn-auto-interfaces/petscvec.h90>
        end interface
        end module

      subroutine F90ArraySFNodeCreate(array,n,ptr)
      use petscis, only: PetscSFNode
      implicit none
      PetscInt n,array(2*n)
      type(PetscSFNode), pointer :: ptr(:)
      PetscInt i
      allocate(ptr(n))
      do i=1,n
        ptr(i)%rank  = array(2*i-1)
        ptr(i)%index = array(2*i)
      enddo

      end subroutine
