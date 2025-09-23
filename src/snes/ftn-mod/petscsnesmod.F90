module petscsnesdef
  use, intrinsic :: ISO_C_binding
  use petsckspdef

#include <../ftn/snes/petscall.h>
#include "petsc/finclude/petscconvest.h"
#include <../ftn/snes/petscconvest.h>
end module petscsnesdef

module petscsnes
  use petscksp
  use petscsnesdef

#include <../src/snes/ftn-mod/petscsnes.h90>
#include <../ftn/snes/petscall.h90>
#include <../ftn/snes/petscconvest.h90>

!  Some PETSc Fortran functions that the user might pass as arguments
!
  external SNESCOMPUTEJACOBIANDEFAULT
  external MATMFFDCOMPUTEJACOBIAN
  external SNESCOMPUTEJACOBIANDEFAULTCOLOR

  external SNESCONVERGEDDEFAULT
  external SNESCONVERGEDSKIP

contains

#include <../ftn/snes/petscall.hf90>
#include <../ftn/snes/petscconvest.hf90>

end module petscsnes
