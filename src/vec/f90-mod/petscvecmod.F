#define PETSC_USE_FORTRAN_MODULES

        module petscisdef
        use petscsysdef
#include <petsc/finclude/petscis.h>
#include <petsc/finclude/ftn-custom/petscisdef.h90>
        end module

        module petscis
        use petscisdef
        use petscsys
#include <petsc/finclude/petscis.h90>
        end module

        module petscvecdef
        use petscisdef
#include <petsc/finclude/petscvec.h>
#include <petsc/finclude/ftn-custom/petscvecdef.h90>
        end module

        module petscvec
        use petscvecdef
        use petscis
#include <petsc/finclude/petscvec.h90>
        end module
