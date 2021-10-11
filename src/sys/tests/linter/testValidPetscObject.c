#include <petsc/private/petscimpl.h>

struct _p_PetscLinterDummyObj {
  void *data;
};

typedef struct _p_PetscLinterDummyObj *PetscLinterDummyObj;

PetscErrorCode ValidPetscObject(PetscObject obj, PetscLinterDummyObj dobj)
{
  /* incorrect */
  PetscValidHeader(obj, 2);
  PetscValidHeader(dobj, 600);

  /* correct */
  PetscValidHeader(obj, 1);
  return 0;
}
