#include <petscsys.h>

void testValidHeaders(PetscRandom r, PetscViewer v, PetscDraw d, PetscDrawAxis a)
{
  /* incorrect */
  PetscValidHeaderSpecificType(r, PETSC_VIEWER_CLASSID, 0, DMDA);
  PetscValidHeaderSpecificType(v, PETSC_DRAW_CLASSID, 0, DMDA);
  PetscValidHeaderSpecificType(d, PETSC_DRAWAXIS_CLASSID, 0, DMDA);
  PetscValidHeaderSpecificType(a, PETSC_RANDOM_CLASSID, 0, DMDA);

  /* correct */
  PetscValidHeaderSpecificType(r, PETSC_RANDOM_CLASSID, 1, DMDA);
  PetscValidHeaderSpecificType(v, PETSC_VIEWER_CLASSID, 2, DMDA);
  PetscValidHeaderSpecificType(d, PETSC_DRAW_CLASSID, 3, DMDA);
  PetscValidHeaderSpecificType(a, PETSC_DRAWAXIS_CLASSID, 4, DMDA);

  /* incorrect */
  PetscValidHeaderSpecific(r, PETSC_DRAW_CLASSID, 0);
  PetscValidHeaderSpecific(v, PETSC_DRAWAXIS_CLASSID, 0);
  PetscValidHeaderSpecific(d, PETSC_RANDOM_CLASSID, 0);
  PetscValidHeaderSpecific(a, PETSC_VIEWER_CLASSID, 0);

  /* correct */
  PetscValidHeaderSpecific(r, PETSC_RANDOM_CLASSID, 1);
  PetscValidHeaderSpecific(v, PETSC_VIEWER_CLASSID, 2);
  PetscValidHeaderSpecific(d, PETSC_DRAW_CLASSID, 3);
  PetscValidHeaderSpecific(a, PETSC_DRAWAXIS_CLASSID, 4);

  /* incorrect */
  PetscValidHeader(r, 55);
  PetscValidHeader(v, 56);
  PetscValidHeader(d, 57);
  PetscValidHeader(a, 58);

  /* correct */
  PetscValidHeader(r, 1);
  PetscValidHeader(v, 2);
  PetscValidHeader(d, 3);
  PetscValidHeader(a, 4);
  return;
}
