#include <petsc/private/petscimpl.h>

PetscErrorCode testDuplicatesWithChanges(PetscInt *a, PetscScalar *b)
{
  /* no remove */
  PetscValidPointer(a, 1);
  /* remove */
  PetscValidPointer(a, 1);
  /* no remove */
  PetscValidPointer(b, 5);
  /* ~should~ be removed but won't be */
  PetscValidPointer(b, 7);
  PetscValidPointer(b, 3);
  return 0;
}

PetscErrorCode testDuplicatesScoped(PetscInt *a, PetscScalar *b)
{
  /* no remove */
  PetscValidPointer(a, 1);
  PetscValidPointer(b, 2);
  /* remove */
  PetscValidPointer(a, 1);
  PetscValidPointer(b, 2);
  {
    /* remove */
    PetscValidPointer(a, 1);
    PetscValidPointer(b, 2);
  }
  return 0;
}

PetscErrorCode testDuplicatesDoubleScoped(PetscInt *a, PetscScalar *b)
{
  /* no remove */
  PetscValidPointer(a, 1);
  PetscValidPointer(b, 2);
  /* remove */
  PetscValidPointer(a, 1);
  PetscValidPointer(b, 2);
  {
    /* remove */
    PetscValidPointer(a, 1);
    PetscValidPointer(b, 2);
  }
  {
    /* remove */
    PetscValidPointer(a, 1);
    PetscValidPointer(b, 2);
  }
  return 0;
}

PetscErrorCode testNoDuplicatesSwitch(PetscInt *a, PetscScalar *b, PetscBool cond)
{
  switch (cond) {
  case PETSC_TRUE:
    /* no remove */
    PetscValidPointer(a, 1);
    PetscValidPointer(b, 2);
    break;
  case PETSC_FALSE:
    /* no remove */
    PetscValidPointer(a, 1);
    PetscValidPointer(b, 2);
    break;
  }
  return 0;
}

PetscErrorCode testDuplicatesNoChangesSwitch(PetscInt *a, PetscScalar *b, PetscBool cond)
{
  /* no remove */
  PetscValidPointer(a, 1);
  PetscValidPointer(b, 2);
  switch (cond) {
  case PETSC_TRUE:
    /* remove */
    PetscValidPointer(a, 1);
    PetscValidPointer(b, 2);
    break;
  case PETSC_FALSE:
    /* remove */
    PetscValidPointer(a, 1);
    PetscValidPointer(b, 2);
    break;
  }
  return 0;
}

PetscErrorCode testNoDuplicatesIfElse(PetscInt *a, PetscScalar *b, PetscBool cond)
{
  if (cond) {
    /* no remove */
    PetscValidPointer(a, 1);
    PetscValidPointer(b, 2);
  } else {
    /* no remove */
    PetscValidPointer(a, 1);
    PetscValidPointer(b, 2);
  }
  return 0;
}

PetscErrorCode testDuplicatesIfElse(PetscInt *a, PetscScalar *b, PetscBool cond)
{
  /* no remove */
  PetscValidPointer(a, 1);
  PetscValidPointer(b, 2);
  if (cond) {
    /* remove */
    PetscValidPointer(a, 1);
    PetscValidPointer(b, 2);
  } else {
    /* remove */
    PetscValidPointer(a, 1);
    PetscValidPointer(b, 2);
  }
  return 0;
}

PetscErrorCode testNoDuplicatesIfElseIfElse(PetscInt *a, PetscScalar *b, PetscBool cond)
{
  if (cond) {
    /* no remove */
    PetscValidPointer(a, 1);
    PetscValidPointer(b, 2);
  } else if (!cond) {
    /* no remove */
    PetscValidPointer(a, 1);
    PetscValidPointer(b, 2);
  } else {
    /* no remove */
    PetscValidPointer(a, 1);
    PetscValidPointer(b, 2);
  }
  return 0;
}
