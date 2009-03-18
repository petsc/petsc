#define PETSC_USE_FORTRAN_MODULES
#include "finclude/petscalldef.h"

        module petscpcdef
        use petscmat
#include "finclude/petscpc.h"
#include "finclude/ftn-custom/petscpcdef.h90"
        end module

        module petscpc
        use petscpcdef
#include "finclude/petscpc.h90"
        end module

        module petsckspdef
        use petscpc
#include "finclude/petscksp.h"
#include "finclude/ftn-custom/petsckspdef.h90"
        end module

        module petscksp
        use petsckspdef
#include "finclude/petscksp.h90"
        end module

        module petscmgdef
        use petscksp
#include "finclude/petscmg.h"
        end module

        module petscmg
        use petscmgdef
        end module

