
        module petscisdef
        use petscsysdef
#include <../src/vec/f90-mod/petscis.h>
        end module

        module  petscaodef
        use petscisdef
        use petscsysdef
#include <../src/vec/f90-mod/petscao.h>
        end module

        module petscvecdef
        use petscisdef
        use petscaodef
#include <../src/vec/f90-mod/petscvec.h>
        end module

        module petscis
        use petscisdef
        use petscsys
#include <../src/vec/f90-mod/petscis.h90>
        interface
#include <../src/vec/f90-mod/ftn-auto-interfaces/petscis.h90>
        end interface
        end module

        module petscao
        use petscis
        use petscaodef
        end module

        module petscvec
        use petscvecdef
        use petscis
        use petscao
#include <../src/vec/f90-mod/petscvec.h90>
        interface
#include <../src/vec/f90-mod/ftn-auto-interfaces/petscvec.h90>
        end interface
        end module
