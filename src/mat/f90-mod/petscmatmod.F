#define PETSC_USE_FORTRAN_MODULES

        module petscmatdef
        use petscvecdef
#include "finclude/petscmat.h"
#include "finclude/ftn-custom/petscmatdef.h90"
        end module

        module petscmat
        use petscmatdef
        use petscvec
#include "finclude/petscmat.h90"
        end module

