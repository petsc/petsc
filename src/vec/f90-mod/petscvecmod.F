#define PETSC_USE_FORTRAN_MODULES

        module petscisdef
        use petscdef
#include "finclude/petscis.h"
#include "finclude/ftn-custom/petscisdef.h90"
        end module

        module petscis
        use petscisdef
        use petsc
#include "finclude/petscis.h90"
        end module

        module petscvecdef
        use petscisdef
#include "finclude/petscvec.h"
#include "finclude/ftn-custom/petscvecdef.h90"
        end module

        module petscvec
        use petscvecdef
        use petscis
#include "finclude/petscvec.h90"
        end module
