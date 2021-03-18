
        module petsctsdefdummy
        use petscsnesdef
#include <../src/ts/f90-mod/petscts.h>
        end module petsctsdefdummy

        module petsctsdef
        use petsctsdefdummy
        interface operator(.ne.)
          function tsnotequal(A,B)
            import tTS
            logical tsnotequal
            type(tTS), intent(in) :: A,B
          end function
          function tsadaptnotequal(A,B)
            import tTSAdapt
            logical tsadaptnotequal
            type(tTSAdapt), intent(in) :: A,B
          end function
          function tstrajectorynotequal(A,B)
            import tTSTrajectory
            logical tstrajectorynotequal
            type(tTSTrajectory), intent(in) :: A,B
          end function
        end interface operator (.ne.)
        interface operator(.eq.)
          function tsequals(A,B)
            import tTS
            logical tsequals
            type(tTS), intent(in) :: A,B
          end function
          function tsadaptequals(A,B)
            import tTSAdapt
            logical tsadaptequals
            type(tTSAdapt), intent(in) :: A,B
          end function
          function tstrajectoryequals(A,B)
            import tTSTrajectory
            logical tstrajectoryequals
            type(tTSTrajectory), intent(in) :: A,B
          end function
        end interface operator (.eq.)
        end module

        function tsnotequal(A,B)
          use petsctsdefdummy, only: tTS
          logical tsnotequal
          type(tTS), intent(in) :: A,B
          tsnotequal = (A%v .ne. B%v)
        end function

        function tsequals(A,B)
          use petsctsdefdummy, only: tTS
          logical tsequals
          type(tTS), intent(in) :: A,B
          tsequals = (A%v .eq. B%v)
        end function

        function tsadaptnotequal(A,B)
          use petsctsdefdummy, only: tTSAdapt
          logical tsadaptnotequal
          type(tTSAdapt), intent(in) :: A,B
          tsadaptnotequal = (A%v .ne. B%v)
        end function

        function tsadaptequals(A,B)
          use petsctsdefdummy, only: tTSAdapt
          logical tsadaptequals
          type(tTSAdapt), intent(in) :: A,B
          tsadaptequals = (A%v .eq. B%v)
        end function

        function tstrajectorynotequal(A,B)
          use petsctsdefdummy, only: tTSTrajectory
          logical tstrajectorynotequal
          type(tTSTrajectory), intent(in) :: A,B
          tstrajectorynotequal = (A%v .ne. B%v)
        end function

        function tstrajectoryequals(A,B)
          use petsctsdefdummy, only: tTSTrajectory
          logical tstrajectoryequals
          type(tTSTrajectory), intent(in) :: A,B
          tstrajectoryequals = (A%v .eq. B%v)
        end function

        module petscts
        use petsctsdef
        use petscsnes
#include <../src/ts/f90-mod/petscts.h90>
        interface
#include <../src/ts/f90-mod/ftn-auto-interfaces/petscts.h90>
#include <../src/ts/f90-mod/ftn-auto-interfaces/petscsensitivity.h90>
        end interface
        end module

