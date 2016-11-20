#define PETSC_USE_FORTRAN_MODULES

        module petscmatdef
        use petscvecdef
#include <../src/mat/f90-mod/petscmat.h>
        interface operator(.ne.)
          procedure matnotequal
        end interface operator (.ne.)

        contains
          function matnotequal(A,B)
          logical matnotequal
          type(tMat), intent(in) :: A,B
          matnotequal = (A%v .ne. B%v)
          end function
        end module

        module petscmat
        use petscmatdef
        use petscvec
#include <../src/mat/f90-mod/petscmat.h90>
        interface
#include <../src/mat/f90-mod/ftn-auto-interfaces/petscmat.h90>
        end interface
        end module

