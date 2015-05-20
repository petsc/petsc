#define PETSC_USE_FORTRAN_MODULES

        module petsctsdef
        use petscsnesdef
#include <petsc/finclude/petscts.h>
#include <petsc/finclude/ftn-custom/petsctsdef.h90>
        end module

        module petscts
        use petsctsdef
        use petscsnes
#include <petsc/finclude/petscts.h90>
        end module


! The all encompassing petsc module

        module petscdef
        use petscdmdadef
        use petsctsdef
        end module

        module petsc
        use petscdmda
        use petscdmcomposite
        use petscts
        end module
