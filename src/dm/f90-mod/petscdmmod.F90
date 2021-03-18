

        module petscdmdefdummy
        use petscmatdef
#include <../src/dm/f90-mod/petscdm.h>
        end module petscdmdefdummy

        module petscdmlabeldef
        use petscmatdef
#include <../src/dm/f90-mod/petscdmlabel.h>
        end module petscdmlabeldef

        module petscdmdef
        use petscdmdefdummy
        use petscdmlabeldef
        interface operator(.ne.)
          function dmnotequal(A,B)
            import tDM
            logical dmnotequal
            type(tDM), intent(in) :: A,B
          end function
        end interface operator (.ne.)
        interface operator(.eq.)
          function dmequals(A,B)
            import tDM
            logical dmequals
            type(tDM), intent(in) :: A,B
          end function
        end interface operator (.eq.)
        end module

        function dmnotequal(A,B)
          use petscdmdefdummy, only: tDM
          logical dmnotequal
          type(tDM), intent(in) :: A,B
          dmnotequal = (A%v .ne. B%v)
        end function

        function dmequals(A,B)
          use petscdmdefdummy, only: tDM
          logical dmequals
          type(tDM), intent(in) :: A,B
          dmequals = (A%v .eq. B%v)
        end function

        module petscdmpatchdef
        use petscdmdef
        end module

        module petscdmforestdef
        use petscdmdef
        end module


        module petscdmlabel
        use petscdmlabeldef
        use petscdmdef
#include <../src/dm/f90-mod/petscdmlabel.h90>
        interface
#include <../src/dm/f90-mod/ftn-auto-interfaces/petscdmlabel.h90>
        end interface
        end module

        module petscdm
        use petscdmdef
        use petscmat
#include <../src/dm/f90-mod/petscdm.h90>
        interface
#include <../src/dm/f90-mod/ftn-auto-interfaces/petscdm.h90>
        end interface
        end module

        module petscdmpatch
        use petscdmpatchdef
#include <../src/dm/f90-mod/petscdmpatch.h90>
        interface
#include <../src/dm/f90-mod/ftn-auto-interfaces/petscdmpatch.h90>
        end interface
        end module

        module petscdmforest
        use petscdmforestdef
#include <../src/dm/f90-mod/petscdmforest.h90>
        interface
#include <../src/dm/f90-mod/ftn-auto-interfaces/petscdmforest.h90>
        end interface
        end module


        module petscdt
        use petscdmdef
#include <../src/dm/f90-mod/petscdt.h90>
        interface
#include <../src/dm/f90-mod/ftn-auto-interfaces/petscdt.h90>
        end interface
        end module


