#define PETSC_USE_FORTRAN_MODULES

        module petscpcdef
        use petscdmdef
#include <petsc/finclude/petscpc.h>
#include <petsc/finclude/ftn-custom/petscpcdef.h90>
        end module

        module petsckspdef
        use petscpcdef
#include <petsc/finclude/petscksp.h>
#include <petsc/finclude/ftn-custom/petsckspdef.h90>
        end module

        module petscpc
        use petsckspdef
        use petscdm
#include <petsc/finclude/petscpc.h90>
        end module

        module petscksp
        use petsckspdef
        use petscpc
#include <petsc/finclude/petscksp.h90>
        end module


