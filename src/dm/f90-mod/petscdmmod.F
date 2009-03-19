#define PETSC_USE_FORTRAN_MODULES

        module petscao
        use petscmat
#include "finclude/petscao.h"
        end module

        module petscdadef
#include "finclude/petscda.h"
#include "finclude/ftn-custom/petscdadef.h90"
        end module

        module petscda
        use petscdadef
#include "finclude/petscda.h90"
        end module

        module petscmeshdef
#include "finclude/petscmesh.h"
#include "finclude/ftn-custom/petscmeshdef.h90"
        end module

        module petscmesh
        use petscmeshdef
#include "finclude/petscmesh.h90"
        end module
