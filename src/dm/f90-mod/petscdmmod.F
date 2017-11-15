#define PETSC_USE_FORTRAN_MODULES

        module  petscaodef
        use petscmatdef
#include <finclude/petscao.h>
!#include <finclude/ftn-custom/petscaodef.h90>
        end module

        module petscao
        use petscaodef
        use petscmat
!#include <finclude/petscao.h90>
        end module

        module petscdmdef
        use petscaodef
#include <finclude/petscdm.h>
#include <finclude/ftn-custom/petscdmdef.h90>
        end module

        module petscdm
        use petscdmdef
        use petscao
#include <finclude/petscdm.h90>
        end module

        module petscdmdadef
        use petscdmdef
#include <finclude/petscdmda.h>
!#include <finclude/ftn-custom/petscdmdadef.h90>
        end module

        module petscdmda
        use petscdmdadef
        use petscdm
#include <finclude/petscdmda.h90>
        end module

        module petscdmcomposite
        use petscdm
#include <finclude/petscdmcomposite.h90>
        end module

        module petscdmadda
        use petscdm
#include <finclude/petscdmadda.h90>
        end module

        module petscdmmeshdef
        use petscdmdef
#include <finclude/petscdmmesh.h>
#include <finclude/ftn-custom/petscdmmeshdef.h90>
        end module

        module petscdmmesh
        use petscdmmeshdef
        use petscdm
#include <finclude/petscdmmesh.h90>
        end module
