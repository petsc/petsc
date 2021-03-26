
        module petscpcdefdummy
        use petscdmdef
        use petscmatdef
#include <../src/ksp/f90-mod/petscpc.h>
        end module petscpcdefdummy

        module petscpcdef
        use petscpcdefdummy
        interface operator(.ne.)
          function pcnotequal(A,B)
            import tPC
            logical pcnotequal
            type(tPC), intent(in) :: A,B
          end function
        end interface operator (.ne.)
        interface operator(.eq.)
          function pcequals(A,B)
            import tPC
            logical pcequals
            type(tPC), intent(in) :: A,B
          end function
        end interface operator (.eq.)
        end module

        function pcnotequal(A,B)
          use petscpcdefdummy, only: tPC
          logical pcnotequal
          type(tPC), intent(in) :: A,B
          pcnotequal = (A%v .ne. B%v)
        end function

        function pcequals(A,B)
          use petscpcdefdummy, only: tPC
          logical pcequals
          type(tPC), intent(in) :: A,B
          pcequals = (A%v .eq. B%v)
        end function

        module petsckspdefdummy
        use petscpcdef
#include <../src/ksp/f90-mod/petscksp.h>
        end module

        module petsckspdef
        use petsckspdefdummy
        interface operator(.ne.)
          function kspnotequal(A,B)
            import tKSP
            logical kspnotequal
            type(tKSP), intent(in) :: A,B
          end function
        end interface operator (.ne.)
        interface operator(.eq.)
          function kspequals(A,B)
            import tKSP
            logical kspequals
            type(tKSP), intent(in) :: A,B
          end function
        end interface operator (.eq.)
        end module

        function kspnotequal(A,B)
          use petsckspdefdummy, only: tKSP
          logical kspnotequal
          type(tKSP), intent(in) :: A,B
          kspnotequal = (A%v .ne. B%v)
        end function

        function kspequals(A,B)
          use petsckspdefdummy, only: tKSP
          logical kspequals
          type(tKSP), intent(in) :: A,B
          kspequals = (A%v .eq. B%v)
        end function

