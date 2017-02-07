
        module petscsnesdefdummy
        use petsckspdef
#include <../src/snes/f90-mod/petscsnes.h>
        end module

        module petscsnesdef
        use petscsnesdefdummy
        interface operator(.ne.)
          function snesnotequal(A,B)
            use petscsnesdefdummy
            logical snesnotequal
            type(tSNES), intent(in) :: A,B
          end function
        end interface operator (.ne.)
        interface operator(.eq.)
          function snesequals(A,B)
            use petscsnesdefdummy
            logical snesequals
            type(tSNES), intent(in) :: A,B
          end function
        end interface operator (.eq.)
        end module

        function snesnotequal(A,B)
          use petscsnesdefdummy
          logical snesnotequal
          type(tSNES), intent(in) :: A,B
          snesnotequal = (A%v .ne. B%v)
        end function

        function snesequals(A,B)
          use petscsnesdefdummy
          logical snesequals
          type(tSNES), intent(in) :: A,B
          snesequals = (A%v .eq. B%v)
        end function

        module petscsnes
        use petscsnesdef
        use petscksp
#include <../src/snes/f90-mod/petscsnes.h90>
        interface
#include <../src/snes/f90-mod/ftn-auto-interfaces/petscsnes.h90>
        end interface
        end module

