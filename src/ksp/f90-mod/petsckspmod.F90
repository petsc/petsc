#include "petsc/finclude/petscpc.h"
#include "petsc/finclude/petscksp.h"
        module petscksp
        use petsckspdef
        use petscpc
#include <../src/ksp/f90-mod/petscksp.h90>
        interface
#include <../src/ksp/f90-mod/ftn-auto-interfaces/petscksp.h90>
        end interface
        end module petscksp


