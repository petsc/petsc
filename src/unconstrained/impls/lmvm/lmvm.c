/*$Id$*/

#include "lmvm.h"

#define LMM_BFGS                0
#define LMM_SCALED_GRADIENT     1
#define LMM_GRADIENT            2

#undef __FUNCT__  
#define __FUNCT__ "TaoSolve_LMVM"
static int TaoSolve_LMVM(TaoSolver tao, void *solver)
{

  /*
  TAO_LMVM *lm = (TAO_LMVM *)solver;
  TaoVec *X, *G = lm->G, *D = lm->D, *W = lm->W;
  TaoVec *Xold = lm->Xold, *Gold = lm->Gold;
  TaoLMVMMat *M = lm->M;

  
  TaoTerminateReason reason;
  TaoTruth success;

  double f, f_full, fold, gdx, gnorm;
  double step = 1.0;

  double delta;

  int info;
  TaoInt stepType;
  TaoInt iter = 0, status = 0;
  TaoInt bfgsUpdates = 0;

  TaoFunctionBegin;

  // Get vectors we will need
  info = TaoGetSolution(tao, &X); CHKERRQ(info);

  // Check convergence criteria
  info = TaoComputeFunctionGradient(tao, X, &f, G); CHKERRQ(info);
  info = G->Norm2(&gnorm); CHKERRQ(info);
  if (TaoInfOrNaN(f) || TaoInfOrNaN(gnorm)) {
    SETERRQ(1, "User provided compute function generated Inf or NaN");
  }

  info = TaoMonitor(tao, iter, f, gnorm, 0.0, step, &reason); CHKERRQ(info);
  if (reason != TAO_CONTINUE_ITERATING) {
    TaoFunctionReturn(0);
  }

  // Set initial scaling for the function
  if (f != 0.0) {
    delta = 2.0 * TaoAbsDouble(f) / (gnorm*gnorm);
  }
  else {
    delta = 2.0 / (gnorm*gnorm);
  }
  info = M->SetDelta(delta); CHKERRQ(info);

  // Set counter for gradient/reset steps
  lm->bfgs = 0;
  lm->sgrad = 0;
  lm->grad = 0;

  // Have not converged; continue with Newton method
  while (reason == TAO_CONTINUE_ITERATING) {
    // Compute direction
    info = M->Update(X, G); CHKERRQ(info);
    info = M->Solve(G, D, &success); CHKERRQ(info);
    ++bfgsUpdates;

    // Check for success (descent direction)
    info = D->Dot(G, &gdx); CHKERRQ(info);
    if ((gdx <= 0.0) || TaoInfOrNaN(gdx)) {
      // Step is not descent or direction produced not a number
      // We can assert bfgsUpdates > 1 in this case because
      // the first solve produces the scaled gradient direction,
      // which is guaranteed to be descent
      //
      // Use steepest descent direction (scaled)
      ++lm->grad;

      if (f != 0.0) {
        delta = 2.0 * TaoAbsDouble(f) / (gnorm*gnorm);
      }
      else {
        delta = 2.0 / (gnorm*gnorm);
      }
      info = M->SetDelta(delta); CHKERRQ(info);
      info = M->Reset(); CHKERRQ(info);
      info = M->Update(X, G); CHKERRQ(info);
      info = M->Solve(G, D, &success); CHKERRQ(info);

      // On a reset, the direction cannot be not a number; it is a 
      // scaled gradient step.  No need to check for this condition.
      // info = D->Norm2(&dnorm); CHKERRQ(info);
      // if (TaoInfOrNaN(dnorm)) {
      //   SETERRQ(1, "Direction generated Not-a-Number");
      // }

      bfgsUpdates = 1;
      ++lm->sgrad;
      stepType = LMM_SCALED_GRADIENT;
    }
    else {
      if (1 == bfgsUpdates) {
        // The first BFGS direction is always the scaled gradient
        ++lm->sgrad;
        stepType = LMM_SCALED_GRADIENT;
      }
      else {
        ++lm->bfgs;
        stepType = LMM_BFGS;
      }
    }
    info = D->Negate(); CHKERRQ(info);
    
    // Perform the linesearch
    fold = f;
    info = Xold->CopyFrom(X); CHKERRQ(info);
    info = Gold->CopyFrom(G); CHKERRQ(info);

    step = 1.0;
    info = TaoLineSearchApply(tao, X, G, D, W, &f, &f_full, &step, &status); CHKERRQ(info);

    while (status && stepType != LMM_GRADIENT) {
      // Linesearch failed
      // Reset factors and use scaled gradient step
      f = fold;
      info = X->CopyFrom(Xold); CHKERRQ(info);
      info = G->CopyFrom(Gold); CHKERRQ(info);
        
      switch(stepType) {
      case LMM_BFGS:
        // Failed to obtain acceptable iterate with BFGS step
        // Attempt to use the scaled gradient direction

        if (f != 0.0) {
          delta = 2.0 * TaoAbsDouble(f) / (gnorm*gnorm);
        }
        else {
          delta = 2.0 / (gnorm*gnorm);
        }
        info = M->SetDelta(delta); CHKERRQ(info);
        info = M->Reset(); CHKERRQ(info);
        info = M->Update(X, G); CHKERRQ(info);
        info = M->Solve(G, D, &success); CHKERRQ(info);

        // On a reset, the direction cannot be not a number; it is a 
        // scaled gradient step.  No need to check for this condition.
        // info = D->Norm2(&dnorm); CHKERRQ(info);
        // if (TaoInfOrNaN(dnorm)) {
        //   SETERRQ(1, "Direction generated Not-a-Number");
        // }
  
	bfgsUpdates = 1;
	++lm->sgrad;
	stepType = LMM_SCALED_GRADIENT;
	break;

      case LMM_SCALED_GRADIENT:
        // The scaled gradient step did not produce a new iterate;
	// attemp to use the gradient direction.
	// Need to make sure we are not using a different diagonal scaling
        info = M->SetDelta(1.0); CHKERRQ(info);
        info = M->Reset(); CHKERRQ(info);
        info = M->Update(X, G); CHKERRQ(info);
        info = M->Solve(G, D, &success); CHKERRQ(info);

        bfgsUpdates = 1;
        ++lm->grad;
        stepType = LMM_GRADIENT;
        break;
      }
      info = D->Negate(); CHKERRQ(info);
        
      // Perform the linesearch
      step = 1.0;
      info = TaoLineSearchApply(tao, X, G, D, W, &f, &f_full, &step, &status); CHKERRQ(info);
    }

    if (status) {
      // Failed to find an improving point
      f = fold;
      info = X->CopyFrom(Xold); CHKERRQ(info);
      info = G->CopyFrom(Gold); CHKERRQ(info);
      step = 0.0;
    }

    // Check for termination
    info = G->Norm2(&gnorm); CHKERRQ(info);
    if (TaoInfOrNaN(f) || TaoInfOrNaN(gnorm)) {
      SETERRQ(1, "User provided compute function generated Inf or NaN");
    }
    info = TaoMonitor(tao, ++iter, f, gnorm, 0.0, step, &reason); CHKERRQ(info);
  }
  TaoFunctionReturn(0);
  */
    return -1;
}

#undef __FUNCT__  
#define __FUNCT__ "TaoSetUp_LMVM"
static int TaoSetUp_LMVM(TaoSolver tao, void *solver)
{
    /*
  TAO_LMVM *lm = (TAO_LMVM *)solver;
  TaoVec *X;
  int info;

  TaoFunctionBegin;

  info = TaoGetSolution(tao, &X); CHKERRQ(info);
  info = X->Clone(&lm->G); CHKERRQ(info);
  info = X->Clone(&lm->D); CHKERRQ(info);
  info = X->Clone(&lm->W); CHKERRQ(info);

  // Create vectors we will need for linesearch
  info = X->Clone(&lm->Xold); CHKERRQ(info);
  info = X->Clone(&lm->Gold); CHKERRQ(info);

  info = TaoSetLagrangianGradientVector(tao, lm->G); CHKERRQ(info);
  info = TaoSetStepDirectionVector(tao, lm->D); CHKERRQ(info);

  // Create matrix for the limited memory approximation
  lm->M = new TaoLMVMMat(X);

  info = TaoCheckFG(tao); CHKERRQ(info);
  TaoFunctionReturn(0);
    */
    return -1;
}

/* ---------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "TaoSetDown_LMVM"
static int TaoSetDown_LMVM(TaoSolver tao, void *solver)
{
    /*
  TAO_LMVM *lm = (TAO_LMVM *)solver;
  int info;

  TaoFunctionBegin;
  info = TaoVecDestroy(lm->G); CHKERRQ(info);
  info = TaoVecDestroy(lm->D); CHKERRQ(info);
  info = TaoVecDestroy(lm->W); CHKERRQ(info);

  info = TaoVecDestroy(lm->Xold); CHKERRQ(info);
  info = TaoVecDestroy(lm->Gold); CHKERRQ(info);

  info = TaoMatDestroy(lm->M); CHKERRQ(info);

  info = TaoSetLagrangianGradientVector(tao, 0); CHKERRQ(info);
  info = TaoSetStepDirectionVector(tao, 0); CHKERRQ(info);
  TaoFunctionReturn(0); 
    */
  return -1;
}

/*------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "TaoSetOptions_LMVM"
static int TaoSetOptions_LMVM(TaoSolver tao, void *solver)
{
    /*
  int info;

  TaoFunctionBegin;
  info = TaoOptionsHead("Limited-memory variable-metric method for unconstrained optimization"); CHKERRQ(info);
  info = TaoLineSearchSetFromOptions(tao); CHKERRQ(info);
  info = TaoOptionsTail(); CHKERRQ(info);
  TaoFunctionReturn(0);
    */
    return -1;
}

/*------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "TaoView_LMVM"
static int TaoView_LMVM(TaoSolver tao, void *solver)
{
    /*
  TAO_LMVM *lm = (TAO_LMVM *)solver;
  int info;

  TaoFunctionBegin;
  info = TaoPrintInt(tao, "  Rejected matrix updates: %d\n", lm->M->GetRejects()); CHKERRQ(info);
  info = TaoPrintInt(tao, "  BFGS steps: %d\n", lm->bfgs); CHKERRQ(info);
  info = TaoPrintInt(tao, "  Scaled gradient steps: %d\n", lm->sgrad); CHKERRQ(info);
  info = TaoPrintInt(tao, "  Gradient steps: %d\n", lm->grad); CHKERRQ(info);
  info = TaoLineSearchView(tao); CHKERRQ(info);
  TaoFunctionReturn(0);
    */
    return -1;
}

/* ---------------------------------------------------------- */

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "TaoCreate_LMVM"
int TaoCreate_LMVM(TaoSolver tao)
{
    /*
  TAO_LMVM *lm;
  int info;

  TaoFunctionBegin;

  info = TaoNew(TAO_LMVM, &lm); CHKERRQ(info);
  info = PetscLogObjectMemory(tao, sizeof(TAO_LMVM)); CHKERRQ(info);

  info = TaoSetTaoSolveRoutine(tao, TaoSolve_LMVM, (void *)lm); CHKERRQ(info);
  info = TaoSetTaoSetUpDownRoutines(tao, TaoSetUp_LMVM, TaoSetDown_LMVM); CHKERRQ(info);
  info = TaoSetTaoOptionsRoutine(tao, TaoSetOptions_LMVM); CHKERRQ(info);
  info = TaoSetTaoViewRoutine(tao, TaoView_LMVM); CHKERRQ(info);

  info = TaoSetMaximumIterates(tao, 2000); CHKERRQ(info);
  info = TaoSetMaximumFunctionEvaluations(tao, 4000); CHKERRQ(info);
  info = TaoSetTolerances(tao, 1e-4, 1e-4, 0, 0); CHKERRQ(info);
  
  info = TaoCreateMoreThuenteLineSearch(tao, 0, 0); CHKERRQ(info);
  TaoFunctionReturn(0);
    */
    return -1;
}
EXTERN_C_END

// Todd: do not delete; they are needed for the component version
// of the code.

/* ---------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "TaoLMVMGetX0"
int TaoLMVMGetX0(TaoSolver tao, Vec x0)
{
    /*
  TAO_LMVM *lm;
  int info;

  TaoFunctionBegin;
  info=TaoGetSolverContext(tao, "tao_lmvm", (void **)&lm); CHKERRQ(info);
  if (lm && lm->M) {
    info=lm->M->GetX0(x0); CHKERRQ(info);
  }
  TaoFunctionReturn(0);
  */
      return -1;
}

/* ---------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "TaoInitializeLMVMmatrix"
int TaoInitializeLMVMmatrix(TaoSolver tao, Vec HV)
{
    /*
  TAO_LMVM *lm;
  int info;
  
  TaoFunctionBegin;
  info = TaoGetSolverContext(tao, "tao_lmvm", (void **)&lm); CHKERRQ(info);
  if (lm && lm->M) {
    info = lm->M->InitialApproximation(HV); CHKERRQ(info);
  }
  TaoFunctionReturn(0);
    */
    return -1;
}

