module petsctsdef
  use, intrinsic :: ISO_C_binding
  use petscsnesdef
#include "petsc/finclude/petscts.h"
#include <../ftn/ts/petscts.h>
end module petsctsdef

module petscts
  use petscsnes
  use petsctsdef

#include <../src/ts/ftn-mod/petscts.h90>
#include <../ftn/ts/petscts.h90>

!
!  Some PETSc Fortran functions that the user might pass as arguments
!
  external TSCOMPUTERHSFUNCTIONLINEAR
  external TSCOMPUTERHSJACOBIANCONSTANT
  external TSCOMPUTEIFUNCTIONLINEAR
  external TSCOMPUTEIJACOBIANCONSTANT

contains

#include <../ftn/ts/petscts.hf90>

end module petscts

!     ----------------------------------------------

module petsccharacteristic
  use petscvecdef
  use petscsys
#include <petsc/finclude/petsccharacteristic.h>
#include <../ftn/ts/petsccharacteristic.h>
#include <../ftn/ts/petsccharacteristic.h90>
contains
#include <../ftn/ts/petsccharacteristic.hf90>
end module petsccharacteristic
