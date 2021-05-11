#include <petscsys.h>

void testValidPointers(void *a, char *b, PetscInt *c, PetscMPIInt *d, PetscInt64 *e, PetscBool *f, PetscScalar *g, PetscReal *h)
{
  /* incorrect */
  PetscValidCharPointer(a,2);
  PetscValidIntPointer(b,3);
  PetscValidBoolPointer(c,4);
  PetscValidRealPointer(d,5);
  PetscValidScalarPointer(e,6);
  PetscValidIntPointer(f,7);
  PetscValidRealPointer(g,8);
  PetscValidScalarPointer(h,9);

  /* correct */
  PetscValidPointer(a,1);
  PetscValidCharPointer(b,2);
  PetscValidIntPointer(c,3);
  PetscValidIntPointer(d,4);
  PetscValidIntPointer(e,5);
  PetscValidBoolPointer(f,6);
  PetscValidScalarPointer(g,7);
  PetscValidRealPointer(h,8);
  return;
}
