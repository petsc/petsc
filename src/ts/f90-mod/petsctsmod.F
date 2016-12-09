
        module petsctsdefdummy
        use petscsnesdef
#include <../src/ts/f90-mod/petscts.h>
        end module

        module petsctsdef
        use petsctsdefdummy
        interface operator(.ne.)
          function tsnotequal(A,B)
            use petsctsdefdummy
            logical tsnotequal
            type(tTS), intent(in) :: A,B
          end function
        end interface operator (.ne.)
        end module

        function tsnotequal(A,B)
          use petsctsdefdummy
          logical tsnotequal
          type(tTS), intent(in) :: A,B
          tsnotequal = (A%v .ne. B%v)
        end function

        module petscts
        use petsctsdef
        use petscsnes
#include <../src/ts/f90-mod/petscts.h90>
        interface
#include <../src/ts/f90-mod/ftn-auto-interfaces/petscts.h90>
        end interface
        end module

