module petscregressordef
  use, intrinsic :: ISO_C_binding
  use petsctaodef
#include <petsc/finclude/petscregressor.h>
#include <../ftn/ml/petscregressor.h>

end module petscregressordef

module petscregressor
  use petscregressordef
  use petsctao

#include <../ftn/ml/petscregressor.h90>

contains

#include <../ftn/ml/petscregressor.hf90>

end module petscregressor

!     ----------------------------------------------

module petscmldef
  use petscregressordef
#include <petsc/finclude/petscml.h>
end module petscmldef

!     ----------------------------------------------

module petscml
  use petscregressor
  use petscmldef
end module petscml
