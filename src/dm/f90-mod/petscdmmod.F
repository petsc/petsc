#define PETSC_USE_FORTRAN_MODULES

        module petscao
        use petscmat
#include "finclude/petscao.h"
        end module

        module petscdadef
        use petscmatdef
#include "finclude/petscda.h"
#include "finclude/ftn-custom/petscdadef.h90"
        end module

        module petscda
        use petscdadef
#include "finclude/petscda.h90"
        end module
