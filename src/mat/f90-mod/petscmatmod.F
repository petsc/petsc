
        module petscmatdefdummy
        use petscvecdef
#include <../src/mat/f90-mod/petscmat.h>
        end module

        module petscmatdef
        use petscmatdefdummy
        interface operator(.ne.)
          function matnotequal(A,B)
            use petscmatdefdummy
            logical matnotequal
            type(tMat), intent(in) :: A,B
          end function
          function matfdcoloringnotequal(A,B)
            use petscmatdefdummy
            logical matfdcoloringnotequal
            type(tMatFDColoring), intent(in) :: A,B
          end function
          function matnullspacenotequal(A,B)
            use petscmatdefdummy
            logical matnullspacenotequal
            type(tMatNullSpace), intent(in) :: A,B
            end function
      end interface operator (.ne.)
        interface operator(.eq.)
          function matequals(A,B)
            use petscmatdefdummy
            logical matequals
            type(tMat), intent(in) :: A,B
          end function
          function matfdcoloringequals(A,B)
            use petscmatdefdummy
            logical matfdcoloringequals
            type(tMatFDColoring), intent(in) :: A,B
          end function
           function matnullspaceequals(A,B)
            use petscmatdefdummy
            logical matnullspaceequals
            type(tMatNullSpace), intent(in) :: A,B
            end function
          end interface operator (.eq.)
        end module

        function matnotequal(A,B)
          use petscmatdefdummy
          logical matnotequal
          type(tMat), intent(in) :: A,B
          matnotequal = (A%v .ne. B%v)
        end function

       function matequals(A,B)
          use petscmatdefdummy
          logical matequals
          type(tMat), intent(in) :: A,B
          matequals = (A%v .eq. B%v)
        end function

        function matfdcoloringnotequal(A,B)
          use petscmatdefdummy
          logical matfdcoloringnotequal
          type(tMatFDColoring), intent(in) :: A,B
          matfdcoloringnotequal = (A%v .ne. B%v)
        end function

        function matfdcoloringequals(A,B)
          use petscmatdefdummy
          logical matfdcoloringequals
          type(tMatFDColoring), intent(in) :: A,B
          matfdcoloringequals = (A%v .eq. B%v)
        end function

        function matnullspacenotequal(A,B)
          use petscmatdefdummy
          logical matnullspacenotequal
          type(tMatNullSpace), intent(in) :: A,B
          matnullspacenotequal = (A%v .ne. B%v)
        end function

        function matnullspaceequals(A,B)
          use petscmatdefdummy
          logical matnullspaceequals
          type(tMatNullSpace), intent(in) :: A,B
          matnullspaceequals = (A%v .eq. B%v)
        end function

#if defined(_WIN32) && defined(PETSC_USE_SHARED_LIBRARIES)
!DEC$ ATTRIBUTES DLLEXPORT::matnotequal
!DEC$ ATTRIBUTES DLLEXPORT::matequals
!DEC$ ATTRIBUTES DLLEXPORT::matfdcoloringnotequal
!DEC$ ATTRIBUTES DLLEXPORT::matfdcoloringequals
!DEC$ ATTRIBUTES DLLEXPORT::matnullspacenotequal
!DEC$ ATTRIBUTES DLLEXPORT::matnullspaceequals
#endif
        module petscmat
        use petscmatdef
        use petscvec
#include <../src/mat/f90-mod/petscmat.h90>
        interface
#include <../src/mat/f90-mod/ftn-auto-interfaces/petscmat.h90>
        end interface
        end module

