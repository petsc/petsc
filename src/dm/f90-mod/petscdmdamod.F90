
        module petscdmdadef
        use petscdmdef
#include <../src/dm/f90-mod/petscdmda.h>
        end module petscdmdadef

        module petscdmda
        use petscdmdadef
        use petscdm
#include <../src/dm/f90-mod/petscdmda.h90>
        interface
#include <../src/dm/f90-mod/ftn-auto-interfaces/petscdmda.h90>
        end interface
        end module petscdmda
