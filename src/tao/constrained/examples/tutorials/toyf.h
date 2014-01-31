#include "finclude/petscsys.h"
#include "finclude/petscvec.h"
#include "finclude/petscmat.h"
#include "finclude/petscksp.h"
#include "finclude/petscpc.h"
#include "finclude/petsctao.h"

      Vec x0,xl,xu
      Vec ce,ci,bl,bu
      Mat Ae,Ai,Hess
      PetscInt n,ne,ni

      common /userctx/ x0,xl,xu,ce,ci,bl,bu,Ae,Ai,Hess,n,ne,ni


