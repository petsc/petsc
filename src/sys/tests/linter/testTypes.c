#include <petscsys.h>

void testTypes(PetscRandom r, PetscViewer v, PetscObject o, PetscFunctionList f)
{
  /* incorrect */
  PetscValidType(r, -1);
  PetscCheckSameType(r, -1, v, -1);
  PetscCheckSameComm(o, -2, f, -2);
  PetscCheckSameTypeAndComm(r, -3, f, -3);

  /* correct */
  PetscValidType(r, 1);
  PetscCheckSameType(r, 1, v, 2);
  PetscCheckSameComm(o, 3, f, 4);
  PetscCheckSameTypeAndComm(r, 1, f, 4);
  return;
}
