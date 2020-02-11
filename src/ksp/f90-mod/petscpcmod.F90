#include "petsc/finclude/petscpc.h"
#include "petsc/finclude/petscksp.h"
        module petscpc
        use petsckspdef
        use petscdm
        use petscmat
#include <../src/ksp/f90-mod/petscpc.h90>
        interface
#include <../src/ksp/f90-mod/ftn-auto-interfaces/petscpc.h90>
        end interface
        end module

