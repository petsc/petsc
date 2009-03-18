#define PETSC_USE_FORTRAN_MODULES
#include "finclude/petscalldef.h"

        module petscmatdef
        use petscvec
#include "finclude/petscmat.h"
#include "finclude/ftn-custom/petscmatdef.h90"
        end module

        module petscmat
        use petscmatdef
#include "finclude/petscmat.h90"
        end module

