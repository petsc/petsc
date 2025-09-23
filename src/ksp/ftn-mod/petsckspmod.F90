module petsckspdef
  use, intrinsic :: ISO_C_binding
  use petscdmdef

#include <../ftn/ksp/petscall.h>
end module petsckspdef

!     ----------------------------------------------

module petscksp
  use petscdm
  use petsckspdef

#include <../src/ksp/ftn-mod/petscksp.h90>
#include <../ftn/ksp/petscall.h90>

contains

#include <../ftn/ksp/petscall.hf90>

end module petscksp

