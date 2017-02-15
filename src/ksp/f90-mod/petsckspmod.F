
        module petscpcdefdummy
        use petscdmdef
        use petscmatdef
#include <../src/ksp/f90-mod/petscpc.h>
#include <../src/ksp/f90-mod/petscksp.h>
        end module

        module petscpcdef
        use petscpcdefdummy
        interface operator(.ne.)
          function pcnotequal(A,B)
            use petscpcdefdummy
            logical pcnotequal
            type(tPC), intent(in) :: A,B
          end function
        end interface operator (.ne.)
        interface operator(.eq.)
          function pcequals(A,B)
            use petscpcdefdummy
            logical pcequals
            type(tPC), intent(in) :: A,B
          end function
        end interface operator (.eq.)
        end module

        function pcnotequal(A,B)
          use petscpcdefdummy
          logical pcnotequal
          type(tPC), intent(in) :: A,B
          pcnotequal = (A%v .ne. B%v)
        end function

        function pcequals(A,B)
          use petscpcdefdummy
          logical pcequals
          type(tPC), intent(in) :: A,B
          pcequals = (A%v .eq. B%v)
        end function

        module petsckspdefdummy
        use petscpcdef
        end module

        module petsckspdef
        use petsckspdefdummy
        interface operator(.ne.)
          function kspnotequal(A,B)
            use petsckspdefdummy
            logical kspnotequal
            type(tKSP), intent(in) :: A,B
          end function
        end interface operator (.ne.)
        interface operator(.eq.)
          function kspequals(A,B)
            use petsckspdefdummy
            logical kspequals
            type(tKSP), intent(in) :: A,B
          end function
        end interface operator (.eq.)
        end module

        function kspnotequal(A,B)
          use petsckspdefdummy
          logical kspnotequal
          type(tKSP), intent(in) :: A,B
          kspnotequal = (A%v .ne. B%v)
        end function

        function kspequals(A,B)
          use petsckspdefdummy
          logical kspequals
          type(tKSP), intent(in) :: A,B
          kspequals = (A%v .eq. B%v)
        end function

        module petscpc
        use petsckspdef
        use petscdm
        use petscmat
#include <../src/ksp/f90-mod/petscpc.h90>
        interface
#include <../src/ksp/f90-mod/ftn-auto-interfaces/petscpc.h90>
        end interface
        end module

        module petscksp
        use petsckspdef
        use petscpc
#include <../src/ksp/f90-mod/petscksp.h90>
        interface
#include <../src/ksp/f90-mod/ftn-auto-interfaces/petscksp.h90>
        end interface
        end module


