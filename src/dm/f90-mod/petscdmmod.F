#define PETSC_USE_FORTRAN_MODULES

        module  petscaodef
        use petscmatdef
#include <petsc/finclude/petscao.h>
!#include <petsc/finclude/ftn-custom/petscaodef.h90>
        end module

        module petscao
        use petscaodef
        use petscmat
!#include <petsc/finclude/petscao.h90>
        end module

        module petscdmdef
        use petscaodef
#include <petsc/finclude/petscdm.h>
#include <petsc/finclude/ftn-custom/petscdmdef.h90>
        end module

        module petscdm
        use petscdmdef
        use petscao
#include <petsc/finclude/petscdm.h90>
        end module

        module petscdmdadef
        use petscdmdef
#include <petsc/finclude/petscdmda.h>
!#include <petsc/finclude/ftn-custom/petscdmdadef.h90>
        end module

        module petscdmda
        use petscdmdadef
        use petscdm
#include <petsc/finclude/petscdmda.h90>
        end module

        module petscdmcomposite
        use petscdm
#include <petsc/finclude/petscdmcomposite.h90>
        end module

