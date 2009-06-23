#include "mfqnls.h"

#undef __FUNCT__
#define __FUNCT__ "TaoSolverSolver_MFQNLS"
static PetscErrorCode TaoSolverSolve_MFQNLS(TaoSolver tao)
{
  TAO_MFQNLS *mfqP = (TAO_MFQNLS *)tao->data;
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

