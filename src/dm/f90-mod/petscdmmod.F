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

        module petscdadef
        use petscaodef
#include "finclude/petscda.h"
#include "finclude/ftn-custom/petscdadef.h90"
        end module

        module petscda
        use petscdadef
        use petscao
#include "finclude/petscda.h90"
        end module

        module petscmeshdef
        use petscdadef
#include "finclude/petscmesh.h"
#include "finclude/ftn-custom/petscmeshdef.h90"
        end module

        module petscmesh
        use petscmeshdef
        use petscda
#include "finclude/petscmesh.h90"
        end module
