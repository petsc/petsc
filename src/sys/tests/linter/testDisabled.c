#include <petsc/private/petscimpl.h>

PetscErrorCode testDisabled(PetscRandom r)
{
  /* incorrect */
  PetscValidHeaderSpecific(r, PETSC_RANDOM_CLASSID, 2);

  /* correct by being disabled */
  PetscDisableStaticAnalyzerForExpressionUnderstandingThatThisIsDangerousAndBugprone(PetscValidHeaderSpecific(r, PETSC_RANDOM_CLASSID, 2));
  return 0;
}
