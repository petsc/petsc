#include <petscsys.h>

void testDuplicatesWithChanges(PetscInt *a, PetscScalar *b)
{
  /* no remove */
  PetscValidIntPointer(a, 1);
  /* remove */
  PetscValidIntPointer(a, 1);
  /* no remove */
  PetscValidIntPointer(b, 5);
  /* ~should~ be removed but won't be */
  PetscValidScalarPointer(b, 7);
  PetscValidScalarPointer(b, 3);
  return;
}

void testDuplicatesScoped(PetscInt *a, PetscScalar *b)
{
  /* no remove */
  PetscValidIntPointer(a, 1);
  PetscValidScalarPointer(b, 2);
  /* remove */
  PetscValidIntPointer(a, 1);
  PetscValidScalarPointer(b, 2);
  {
    /* remove */
    PetscValidIntPointer(a, 1);
    PetscValidScalarPointer(b, 2);
  }
  return;
}

void testDuplicatesDoubleScoped(PetscInt *a, PetscScalar *b)
{
  /* no remove */
  PetscValidIntPointer(a, 1);
  PetscValidScalarPointer(b, 2);
  /* remove */
  PetscValidIntPointer(a, 1);
  PetscValidScalarPointer(b, 2);
  {
    /* remove */
    PetscValidIntPointer(a, 1);
    PetscValidScalarPointer(b, 2);
  }
  {
    /* remove */
    PetscValidIntPointer(a, 1);
    PetscValidScalarPointer(b, 2);
  }
  return;
}

void testNoDuplicatesSwitch(PetscInt *a, PetscScalar *b, PetscBool cond)
{
  switch (cond) {
  case PETSC_TRUE:
    /* no remove */
    PetscValidIntPointer(a, 1);
    PetscValidScalarPointer(b, 2);
    break;
  case PETSC_FALSE:
    /* no remove */
    PetscValidIntPointer(a, 1);
    PetscValidScalarPointer(b, 2);
    break;
  }
  return;
}

void testDuplicatesNoChangesSwitch(PetscInt *a, PetscScalar *b, PetscBool cond)
{
  /* no remove */
  PetscValidIntPointer(a, 1);
  PetscValidScalarPointer(b, 2);
  switch (cond) {
  case PETSC_TRUE:
    /* remove */
    PetscValidIntPointer(a, 1);
    PetscValidScalarPointer(b, 2);
    break;
  case PETSC_FALSE:
    /* remove */
    PetscValidIntPointer(a, 1);
    PetscValidScalarPointer(b, 2);
    break;
  }
  return;
}

void testNoDuplicatesIfElse(PetscInt *a, PetscScalar *b, PetscBool cond)
{
  if (cond) {
    /* no remove */
    PetscValidIntPointer(a, 1);
    PetscValidScalarPointer(b, 2);
  } else {
    /* no remove */
    PetscValidIntPointer(a, 1);
    PetscValidScalarPointer(b, 2);
  }
  return;
}

void testDuplicatesIfElse(PetscInt *a, PetscScalar *b, PetscBool cond)
{
  /* no remove */
  PetscValidIntPointer(a, 1);
  PetscValidScalarPointer(b, 2);
  if (cond) {
    /* remove */
    PetscValidIntPointer(a, 1);
    PetscValidScalarPointer(b, 2);
  } else {
    /* remove */
    PetscValidIntPointer(a, 1);
    PetscValidScalarPointer(b, 2);
  }
  return;
}

void testNoDuplicatesIfElseIfElse(PetscInt *a, PetscScalar *b, PetscBool cond)
{
  if (cond) {
    /* no remove */
    PetscValidIntPointer(a, 1);
    PetscValidScalarPointer(b, 2);
  } else if (!cond) {
    /* no remove */
    PetscValidIntPointer(a, 1);
    PetscValidScalarPointer(b, 2);
  } else {
    /* no remove */
    PetscValidIntPointer(a, 1);
    PetscValidScalarPointer(b, 2);
  }
  return;
}
