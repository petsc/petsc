/* for access to private viewer members */
#include <petsc/private/viewerimpl.h>

/* forward declare */
extern void extractFunc(PetscViewer, void **);

PetscErrorCode testOutOfLineReference(PetscViewer v, PetscViewer v2)
{
  /* linter should be able to connect all of these to v */
  void  *foo  = v->data, *bar, *baz, *blop;
  void **blip = &v->data;

  bar  = v->data;
  blop = blip[0];
  extractFunc(v, &baz);

  /* incorrect */
  PetscAssertPointer(foo, -1);
  PetscAssertPointer(bar, -2);
  PetscAssertPointer(baz, -3);
  PetscAssertPointer((void *)v->data, -4);
  PetscAssertPointer(*blip, -5);
  PetscAssertPointer(blop, -6);

  /* correct */
  PetscAssertPointer(foo, 1);
  PetscAssertPointer(bar, 1);
  PetscAssertPointer(baz, 1);
  PetscAssertPointer((void *)v->data, 1);
  PetscAssertPointer(*blip, 1);
  PetscAssertPointer(blop, 1);
  return 0;
}
