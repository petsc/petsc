module petsctaodef
  use, intrinsic :: ISO_C_binding
  use petsckspdef

#include <../ftn/tao/petscall.h>
end module petsctaodef

module petsctao
  use petscts
  use petsctaodef

#include <../ftn/tao/petscall.h90>

contains

#include <../ftn/tao/petscall.hf90>

end module petsctao

! The all encompassing PETSc module

module petsc
  use petsctao
  use petscao
  use petscpf
  use petscdmplex
  use petscdmswarm
  use petscdmnetwork
  use petscdmda
  use petscdmcomposite
  use petscdmforest
  use petsccharacteristic
  use petscbag
end module petsc
