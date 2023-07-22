#include <petsc/private/petscimpl.h>

PetscErrorCode testValidPointers(void *a, char *b, PetscInt *c, PetscMPIInt *d, PetscInt *e, PetscBool *f, PetscScalar *g, PetscReal *h)
{
  /* incorrect */
  PetscAssertPointer(a, 2);
  PetscAssertPointer(b, 3);
  PetscAssertPointer(c, 4);
  PetscAssertPointer(d, 5);
  PetscAssertPointer(e, 6);
  PetscAssertPointer(f, 7);
  PetscAssertPointer(g, 8);
  PetscAssertPointer(h, 9);

  /* correct */
  PetscAssertPointer(a, 1);
  PetscAssertPointer(b, 2);
  PetscAssertPointer(c, 3);
  PetscAssertPointer(d, 4);
  PetscAssertPointer(e, 5);
  PetscAssertPointer(f, 6);
  PetscAssertPointer(g, 7);
  PetscAssertPointer(h, 8);
  return 0;
}

void testValidPointers2(void *a, char *b, PetscInt *c, PetscMPIInt *d, PetscInt *e, PetscBool *f, PetscScalar *g, PetscReal *h)
{
  /* incorrect */
  PetscAssertPointer(a, 2);
  PetscAssertPointer(b, 3);
  PetscAssertPointer(c, 4);
  PetscAssertPointer(d, 5);
  PetscAssertPointer(e, 6);
  PetscAssertPointer(f, 7);
  PetscAssertPointer(g, 8);
  PetscAssertPointer(h, 9);

  /* correct */
  PetscAssertPointer(a, 1);
  PetscAssertPointer(b, 2);
  PetscAssertPointer(c, 3);
  PetscAssertPointer(d, 4);
  PetscAssertPointer(e, 5);
  PetscAssertPointer(f, 6);
  PetscAssertPointer(g, 7);
  PetscAssertPointer(h, 8);
  return;
}

void testValidPointers3(void **a, char **b, PetscInt **c, PetscMPIInt **d, PetscInt **e, PetscBool **f, PetscScalar **g, PetscReal **h)
{
  /* incorrect */
  PetscAssertPointer(a, 1);
  PetscAssertPointer(b, 2);
  PetscAssertPointer(c, 3);
  PetscAssertPointer(d, 4);
  PetscAssertPointer(e, 5);
  PetscAssertPointer(f, 6);
  PetscAssertPointer(g, 7);
  PetscAssertPointer(h, 8);

  /* correct */
  PetscAssertPointer(a, 1);
  PetscAssertPointer(b, 2);
  PetscAssertPointer(c, 3);
  PetscAssertPointer(d, 4);
  PetscAssertPointer(e, 5);
  PetscAssertPointer(f, 6);
  PetscAssertPointer(g, 7);
  PetscAssertPointer(h, 8);
  return;
}
