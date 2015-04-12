#define PETSC_USE_FORTRAN_MODULES

        module petscsnesdef
        use petsckspdef
#include <petsc/finclude/petscsnes.h>
#include <petsc/finclude/ftn-custom/petscsnesdef.h90>
        end module

        module petscsnes
        use petscsnesdef
        use petscksp
#include <petsc/finclude/petscsnes.h90>
        end module

