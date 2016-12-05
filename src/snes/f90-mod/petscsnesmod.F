
        module petscsnesdef
        use petsckspdef
#include <../src/snes/f90-mod/petscsnes.h>
        end module

        module petscsnes
        use petscsnesdef
        use petscksp
#include <../src/snes/f90-mod/petscsnes.h90>
        interface
#include <../src/snes/f90-mod/ftn-auto-interfaces/petscsnes.h90>
        end interface
        end module

