#define PETSC_USE_FORTRAN_MODULES

        module  petscaodef
        use petscmatdef
#include "finclude/petscao.h"
!#include "finclude/ftn-custom/petscaodef.h90"
        end module

        module petscao
        use petscaodef
        use petscmat
!#include "finclude/petscao.h90"
        end module

        module petscdmdef
        use petscaodef
#include "finclude/petscdm.h"
#include "finclude/ftn-custom/petscdmdef.h90"
        end module

        module petscdm
        use petscdmdef
        use petscao
#include "finclude/petscdm.h90"
        end module

        module petscmeshdef
        use petscdmdef
#include "finclude/petscmesh.h"
#include "finclude/ftn-custom/petscmeshdef.h90"
        end module

        module petscmesh
        use petscmeshdef
        use petscdm
#include "finclude/petscmesh.h90"
        end module
