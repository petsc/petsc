#define PETSC_USE_FORTRAN_MODULES

        module petsctsdef
        use petscsnesdef
#include <finclude/petscts.h>
#include <finclude/ftn-custom/petsctsdef.h90>
        end module

        module petscts
        use petsctsdef
        use petscsnes
#include <finclude/petscts.h90>
        end module


! The all encompassing petsc module

        module petscdef
        use petscdmdadef
        use petscdmmeshdef
        use petsctsdef
        end module

        module petsc
        use petscdmda
        use petscdmadda
        use petscdmcomposite
        use petscdmmesh
        use petscpcmg
        use petscts
        end module
