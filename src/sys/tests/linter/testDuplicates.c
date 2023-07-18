#include <petsc/private/petscimpl.h>

PetscErrorCode testDuplicatesWithChanges(PetscInt *a, PetscScalar *b)
{
  /* no remove */
  PetscAssertPointer(a, 1);
  /* remove */
  PetscAssertPointer(a, 1);
  /* no remove */
  PetscAssertPointer(b, 5);
  /* ~should~ be removed but won't be */
  PetscAssertPointer(b, 7);
  PetscAssertPointer(b, 3);
  return 0;
}

PetscErrorCode testDuplicatesScoped(PetscInt *a, PetscScalar *b)
{
  /* no remove */
  PetscAssertPointer(a, 1);
  PetscAssertPointer(b, 2);
  /* remove */
  PetscAssertPointer(a, 1);
  PetscAssertPointer(b, 2);
  {
    /* remove */
    PetscAssertPointer(a, 1);
    PetscAssertPointer(b, 2);
  }
  return 0;
}

PetscErrorCode testDuplicatesDoubleScoped(PetscInt *a, PetscScalar *b)
{
  /* no remove */
  PetscAssertPointer(a, 1);
  PetscAssertPointer(b, 2);
  /* remove */
  PetscAssertPointer(a, 1);
  PetscAssertPointer(b, 2);
  {
    /* remove */
    PetscAssertPointer(a, 1);
    PetscAssertPointer(b, 2);
  }
  {
    /* remove */
    PetscAssertPointer(a, 1);
    PetscAssertPointer(b, 2);
  }
  return 0;
}

PetscErrorCode testNoDuplicatesSwitch(PetscInt *a, PetscScalar *b, PetscBool cond)
{
  switch (cond) {
  case PETSC_TRUE:
    /* no remove */
    PetscAssertPointer(a, 1);
    PetscAssertPointer(b, 2);
    break;
  case PETSC_FALSE:
    /* no remove */
    PetscAssertPointer(a, 1);
    PetscAssertPointer(b, 2);
    break;
  }
  return 0;
}

PetscErrorCode testDuplicatesNoChangesSwitch(PetscInt *a, PetscScalar *b, PetscBool cond)
{
  /* no remove */
  PetscAssertPointer(a, 1);
  PetscAssertPointer(b, 2);
  switch (cond) {
  case PETSC_TRUE:
    /* remove */
    PetscAssertPointer(a, 1);
    PetscAssertPointer(b, 2);
    break;
  case PETSC_FALSE:
    /* remove */
    PetscAssertPointer(a, 1);
    PetscAssertPointer(b, 2);
    break;
  }
  return 0;
}

PetscErrorCode testNoDuplicatesIfElse(PetscInt *a, PetscScalar *b, PetscBool cond)
{
  if (cond) {
    /* no remove */
    PetscAssertPointer(a, 1);
    PetscAssertPointer(b, 2);
  } else {
    /* no remove */
    PetscAssertPointer(a, 1);
    PetscAssertPointer(b, 2);
  }
  return 0;
}

PetscErrorCode testDuplicatesIfElse(PetscInt *a, PetscScalar *b, PetscBool cond)
{
  /* no remove */
  PetscAssertPointer(a, 1);
  PetscAssertPointer(b, 2);
  if (cond) {
    /* remove */
    PetscAssertPointer(a, 1);
    PetscAssertPointer(b, 2);
  } else {
    /* remove */
    PetscAssertPointer(a, 1);
    PetscAssertPointer(b, 2);
  }
  return 0;
}

PetscErrorCode testNoDuplicatesIfElseIfElse(PetscInt *a, PetscScalar *b, PetscBool cond)
{
  if (cond) {
    /* no remove */
    PetscAssertPointer(a, 1);
    PetscAssertPointer(b, 2);
  } else if (!cond) {
    /* no remove */
    PetscAssertPointer(a, 1);
    PetscAssertPointer(b, 2);
  } else {
    /* no remove */
    PetscAssertPointer(a, 1);
    PetscAssertPointer(b, 2);
  }
  return 0;
}
