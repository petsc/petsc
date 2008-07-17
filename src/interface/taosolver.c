#define TAOSOLVER_DLL

#include "include/private/taosolver_impl.h" 

PetscTruth TaoSolverRegisterAllCalled = PETSC_FALSE;
PetscFList TaoSolverList = PETSC_NULL;

PetscCookie TAOSOLVER_DLL TAOSOLVER_COOKIE;
PetscLogEvent TaoSolver_Solve, TaoSolver_FunctionEval, TaoSolver_GradientEval, TaoSolver_HessianEval, TaoSolver_JacobianEval;

#undef __FUNCT__
#define __FUNCT__ "TaoSolverSolve"
/*@ 
  TaoSolverSolve - Solves an optimization problem min F(x) s.t. l <= x <= u

  Collective on TaoSolver
  
  Input Parameters:
. tao - the TaoSolver context

  Notes:
  The user must set up the TaoSolver with calls to TaoSolverSetInitialVector(),
  TaoSolverSetObjective(),
  TaoSolverSetGradient(), and (if using 2nd order method) TaoSolverSetHessian().

  .seealso: TaoSolverCreate(), TaoSolverSetObjective(), TaoSolverSetGradient(), TaoSolverSetHessian()
  @*/
PetscErrorCode TAOSOLVER_DLL TaoSolverSolve(TaoSolver tao)
{
  PetscErrorCode info;
  TaoFunctionBegin;
  TaoValidHeaderSpecific(tao,TAOSOLVER_COOKIE,1);

  /*
  info = TaoGetSolution(tao,&xx);CHKERRQ(info);
  info = TaoSetUp(tao);CHKERRQ(info);
  info = TaoSetDefaultStatistics(tao); CHKERRQ(info);
  if (tao->solve){ info = (*(tao)->solve)(tao,tao->data);CHKERRQ(info); }
  if (tao->viewtao) { info = TaoView(tao);CHKERRQ(info); }
  if (tao->viewksptao) { info = TaoViewLinearSolver(tao);CHKERRQ(info); }
  */
  TaoFunctionReturn(0);

    
}
