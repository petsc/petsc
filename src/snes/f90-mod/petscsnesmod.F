#define PETSC_USE_FORTRAN_MODULES

        module petscsnesdef
        use petscksp
#include "finclude/petscsnes.h"
#include "finclude/ftn-custom/petscsnesdef.h90"
        end module

        module petscsnes
        use petscsnesdef
#include "finclude/petscsnes.h90"
        end module

