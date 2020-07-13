
        module petscdmswarmdef
        use petscdmdef
#include <../src/dm/f90-mod/petscdmswarm.h>
        end module

        module petscdmswarm
        use petscdmswarmdef
#include <../src/dm/f90-mod/petscdmswarm.h90>
        interface
#include <../src/dm/f90-mod/ftn-auto-interfaces/petscdmswarm.h90>
        end interface
        end module
