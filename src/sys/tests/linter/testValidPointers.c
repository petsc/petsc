#include <petsc/private/petscimpl.h>

PetscErrorCode testValidPointers(void *a, char *b, PetscInt *c, PetscMPIInt *d, PetscInt *e, PetscBool *f, PetscScalar *g, PetscReal *h)
{
  /* incorrect */
  PetscValidCharPointer(a, 2);
  PetscValidIntPointer(b, 3);
  PetscValidBoolPointer(c, 4);
  PetscValidRealPointer(d, 5);
  PetscValidScalarPointer(e, 6);
  PetscValidIntPointer(f, 7);
  PetscValidRealPointer(g, 8);
  PetscValidScalarPointer(h, 9);

  /* correct */
  PetscValidPointer(a, 1);
  PetscValidCharPointer(b, 2);
  PetscValidIntPointer(c, 3);
  PetscValidIntPointer(d, 4);
  PetscValidIntPointer(e, 5);
  PetscValidBoolPointer(f, 6);
  PetscValidScalarPointer(g, 7);
  PetscValidRealPointer(h, 8);
  return 0;
}

void testValidPointers2(void *a, char *b, PetscInt *c, PetscMPIInt *d, PetscInt *e, PetscBool *f, PetscScalar *g, PetscReal *h)
{
  /* incorrect */
  PetscValidCharPointer(a, 2);
  PetscValidPointer(b, 3);
  PetscValidPointer(c, 4);
  PetscValidPointer(d, 5);
  PetscValidPointer(e, 6);
  PetscValidPointer(f, 7);
  PetscValidPointer(g, 8);
  PetscValidPointer(h, 9);

  /* correct */
  PetscValidPointer(a, 1);
  PetscValidCharPointer(b, 2);
  PetscValidIntPointer(c, 3);
  PetscValidIntPointer(d, 4);
  PetscValidIntPointer(e, 5);
  PetscValidBoolPointer(f, 6);
  PetscValidScalarPointer(g, 7);
  PetscValidRealPointer(h, 8);
  return;
}

void testValidPointers3(void **a, char **b, PetscInt **c, PetscMPIInt **d, PetscInt **e, PetscBool **f, PetscScalar **g, PetscReal **h)
{
  /* incorrect */
  PetscValidCharPointer(a, 1);
  PetscValidCharPointer(b, 2);
  PetscValidIntPointer(c, 3);
  PetscValidIntPointer(d, 4);
  PetscValidIntPointer(e, 5);
  PetscValidBoolPointer(f, 6);
  PetscValidScalarPointer(g, 7);
  PetscValidRealPointer(h, 8);

  /* correct */
  PetscValidPointer(a, 1);
  PetscValidPointer(b, 2);
  PetscValidPointer(c, 3);
  PetscValidPointer(d, 4);
  PetscValidPointer(e, 5);
  PetscValidPointer(f, 6);
  PetscValidPointer(g, 7);
  PetscValidPointer(h, 8);
  return;
}
