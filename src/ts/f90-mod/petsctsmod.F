#define PETSC_USE_FORTRAN_MODULES

        module petsctsdef
        use petscsnes
#include "finclude/petscts.h"
#include "finclude/ftn-custom/petsctsdef.h90"
        end module

        module petscts
        use petsctsdef
#include "finclude/petscts.h90"
        end module
