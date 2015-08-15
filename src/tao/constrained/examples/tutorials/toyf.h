#include "petsc/finclude/petscsys.h"
#include "petsc/finclude/petscvec.h"
#include "petsc/finclude/petscmat.h"
#include "petsc/finclude/petscksp.h"
#include "petsc/finclude/petscpc.h"
#include "petsc/finclude/petsctao.h"

      Vec x0,xl,xu
      Vec ce,ci,bl,bu
      Mat Ae,Ai,Hess
      PetscInt n,ne,ni

      common /userctx/ x0,xl,xu,ce,ci,bl,bu,Ae,Ai,Hess,n,ne,ni


