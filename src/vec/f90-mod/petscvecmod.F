#define PETSC_USE_FORTRAN_MODULES
#include "finclude/petscalldef.h"

        module petscisdef
        use petsc
#include "finclude/petscis.h"
#include "finclude/ftn-custom/petscisdef.h90"
        end module

        module petscis
        use petscisdef
#include "finclude/petscis.h90"
        end module

        module petscvecdef
        use petscis
#include "finclude/petscvec.h"
#include "finclude/ftn-custom/petscvecdef.h90"
        end module

        module petscvec
        use petscvecdef
#include "finclude/petscvec.h90"
        end module
