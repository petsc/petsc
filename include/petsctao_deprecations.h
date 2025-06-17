#pragma once

/* MANSEC = Tao */

PETSC_DEPRECATED_FUNCTION(3, 17, 0, "TaoSetSolution()", ) static inline PetscErrorCode TaoSetInitialVector(Tao t, Vec v)
{
  return TaoSetSolution(t, v);
}
PETSC_DEPRECATED_FUNCTION(3, 17, 0, "TaoGetSolution()", ) static inline PetscErrorCode TaoGetInitialVector(Tao t, Vec *v)
{
  return TaoGetSolution(t, v);
}
PETSC_DEPRECATED_FUNCTION(3, 17, 0, "TaoSetObjective()", ) static inline PetscErrorCode TaoSetObjectiveRoutine(Tao t, PetscErrorCode (*f)(Tao, Vec, PetscReal *, void *), void *c)
{
  return TaoSetObjective(t, f, c);
}
PETSC_DEPRECATED_FUNCTION(3, 17, 0, "TaoGetGradient()", ) static inline PetscErrorCode TaoGetGradientVector(Tao t, Vec *v)
{
  return TaoGetGradient(t, v, PETSC_NULLPTR, PETSC_NULLPTR);
}
PETSC_DEPRECATED_FUNCTION(3, 17, 0, "TaoSetGradient()", ) static inline PetscErrorCode TaoSetGradientRoutine(Tao t, PetscErrorCode (*f)(Tao, Vec, Vec, void *), void *c)
{
  return TaoSetGradient(t, PETSC_NULLPTR, f, c);
}
PETSC_DEPRECATED_FUNCTION(3, 17, 0, "TaoSetObjectiveAndGradient()", ) static inline PetscErrorCode TaoSetObjectiveAndGradientRoutine(Tao t, PetscErrorCode (*f)(Tao, Vec, PetscReal *, Vec, void *), void *c)
{
  return TaoSetObjectiveAndGradient(t, PETSC_NULLPTR, f, c);
}
PETSC_DEPRECATED_FUNCTION(3, 17, 0, "TaoSetHessian()", ) static inline PetscErrorCode TaoSetHessianRoutine(Tao t, Mat H, Mat P, PetscErrorCode (*f)(Tao, Vec, Mat, Mat, void *), void *c)
{
  return TaoSetHessian(t, H, P, f, c);
}
PETSC_DEPRECATED_FUNCTION(3, 11, 0, "TaoSetResidualRoutine()", ) static inline PetscErrorCode TaoSetSeparableObjectiveRoutine(Tao tao, Vec res, PetscErrorCode (*func)(Tao, Vec, Vec, void *), void *ctx)
{
  return TaoSetResidualRoutine(tao, res, func, ctx);
}
PETSC_DEPRECATED_FUNCTION(3, 11, 0, "TaoSetResidualWeights()", ) static inline PetscErrorCode TaoSetSeparableObjectiveWeights(Tao tao, Vec sigma_v, PetscInt n, PetscInt *rows, PetscInt *cols, PetscReal *vals)
{
  return TaoSetResidualWeights(tao, sigma_v, n, rows, cols, vals);
}
PETSC_DEPRECATED_FUNCTION(3, 11, 0, "TaoComputeResidual()", ) static inline PetscErrorCode TaoComputeSeparableObjective(Tao tao, Vec X, Vec F)
{
  return TaoComputeResidual(tao, X, F);
}
PETSC_DEPRECATED_FUNCTION(3, 21, 0, "TaoMonitorSet()", ) static inline PetscErrorCode TaoSetMonitor(Tao tao, PetscErrorCode (*monitor)(Tao, void *), void *ctx, PetscCtxDestroyFn *destroy)
{
  return TaoMonitorSet(tao, monitor, ctx, destroy);
}
PETSC_DEPRECATED_FUNCTION(3, 21, 0, "TaoMonitorCancel()", ) static inline PetscErrorCode TaoCancelMonitors(Tao tao)
{
  return TaoMonitorCancel(tao);
}
PETSC_DEPRECATED_FUNCTION(3, 9, 0, "TaoMonitorDefault()", ) static inline PetscErrorCode TaoDefaultMonitor(Tao tao, void *ctx)
{
  return TaoMonitorDefault(tao, ctx);
}
PETSC_DEPRECATED_FUNCTION(3, 21, 0, "TaoMonitorGlobalization()", ) static inline PetscErrorCode TaoGMonitor(Tao tao, void *ctx)
{
  return TaoMonitorGlobalization(tao, ctx);
}
PETSC_DEPRECATED_FUNCTION(3, 21, 0, "TaoMonitorDefaultShort()", ) static inline PetscErrorCode TaoSMonitor(Tao tao, void *ctx)
{
  return TaoMonitorDefaultShort(tao, ctx);
}
PETSC_DEPRECATED_FUNCTION(3, 21, 0, "TaoMonitorConstraintNorm()", ) static inline PetscErrorCode TaoCMonitor(Tao tao, void *ctx)
{
  return TaoMonitorConstraintNorm(tao, ctx);
}
PETSC_DEPRECATED_FUNCTION(3, 21, 0, "TaoMonitorSolution()", ) static inline PetscErrorCode TaoSolutionMonitor(Tao tao, void *ctx)
{
  return TaoMonitorSolution(tao, ctx);
}
PETSC_DEPRECATED_FUNCTION(3, 21, 0, "TaoMonitorGradient()", ) static inline PetscErrorCode TaoGradientMonitor(Tao tao, void *ctx)
{
  return TaoMonitorGradient(tao, ctx);
}
PETSC_DEPRECATED_FUNCTION(3, 21, 0, "TaoMonitorResidual()", ) static inline PetscErrorCode TaoResidualMonitor(Tao tao, void *ctx)
{
  return TaoMonitorResidual(tao, ctx);
}
PETSC_DEPRECATED_FUNCTION(3, 21, 0, "TaoMonitorStep()", ) static inline PetscErrorCode TaoStepDirectionMonitor(Tao tao, void *ctx)
{
  return TaoMonitorStep(tao, ctx);
}
PETSC_DEPRECATED_FUNCTION(3, 21, 0, "TaoMonitorSolutionDraw()", ) static inline PetscErrorCode TaoDrawSolutionMonitor(Tao tao, void *ctx)
{
  return TaoMonitorGlobalization(tao, ctx);
}
PETSC_DEPRECATED_FUNCTION(3, 21, 0, "TaoMonitorGradientDraw()", ) static inline PetscErrorCode TaoDrawGradientMonitor(Tao tao, void *ctx)
{
  return TaoMonitorGlobalization(tao, ctx);
}
PETSC_DEPRECATED_FUNCTION(3, 21, 0, "TaoMonitorStepDraw()", ) static inline PetscErrorCode TaoDrawStepDirectionMonitor(Tao tao, void *ctx)
{
  return TaoMonitorGlobalization(tao, ctx);
}
