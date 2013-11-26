#include "SIDL.h"
#include "SIDL_BaseClass.h"
#include "Optimize_OptimizationModel.h"
#include "Solver_OptimizationSolver.h"
#include "Solver_ProjectState.h"

int main() 
{
  SIDL_BaseClass base;
  Optimize_OptimizationModel model;
  Solver_OptimizationSolver tao;
  Solver_ProjectState taoState;

  /* Create the SIDL objects and cast to appropriate interfaces */
  base = SIDL_Loader_createClass("Rosenbrock.RosenbrockModel");
  model = Optimize_OptimizationModel__cast(base);

  base = SIDL_Loader_createClass("TAO.Solver");
  tao = Solver_OptimizationSolver__cast(base);

  base = SIDL_Loader_createClass("TAO.Environment");
  taoState = Solver_ProjectState__cast(base);

  /* Initialize TAO and solve the application */
  Solver_ProjectState_InitializeNoArgs(taoState);
  Solver_OptimizationSolver_Create(tao,"tao_lmvm");
  Solver_OptimizationSolver_SolveApplication(tao,model);

  /* Output the solver results */
  Solver_OptimizationSolver_View(tao);

  /* Release memory */
  Solver_OptimizationSolver_Destroy(tao);
  Solver_ProjectState_Finalize(taoState);

  Optimize_OptimizationModel_deleteRef(model);
  Solver_OptimizationSolver_deleteRef(tao);
  Solver_ProjectState_deleteRef(taoState);

  return 0;
}

