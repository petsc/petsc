#define PETSC_USE_FORTRAN_MODULES

        module petscmatdef
        use petscvecdef
#include <petsc/finclude/petscmat.h>
#include <petsc/finclude/ftn-custom/petscmatdef.h90>
        end module

        module petscmat
        use petscmatdef
        use petscvec
#include <petsc/finclude/petscmat.h90>
        end module

