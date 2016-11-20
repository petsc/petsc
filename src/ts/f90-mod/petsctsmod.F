
        module petsctsdef
        use petscsnesdef
#include <../src/ts/f90-mod/petscts.h>
        end module

        module petscts
        use petsctsdef
        use petscsnes
#include <../src/ts/f90-mod/petscts.h90>
        interface
#include <../src/ts/f90-mod/ftn-auto-interfaces/petscts.h90>
        end interface
        end module

