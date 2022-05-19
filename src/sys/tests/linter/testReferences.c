/* for access to private vec members */
#include <petsc/private/viewerimpl.h>

/* forward declare */
void extractFunc(PetscViewer,void**);

void testOutOfLineReference(PetscViewer v, PetscViewer v2)
{
  /* linter should be able to connect all of these to v */
  void *foo = v->data,*bar,*baz,*blop;
  void **blip = &v->data;

  bar  = v->data;
  blop = blip[0];
  extractFunc(v,&baz);

  /* incorrect */
  PetscValidPointer(foo,-1);
  PetscValidPointer(bar,-2);
  PetscValidPointer(baz,-3);
  PetscValidPointer((void *)v->data,-4);
  PetscValidPointer(*blip,-5);
  PetscValidPointer(blop,-6);

  /* correct */
  PetscValidPointer(foo,1);
  PetscValidPointer(bar,1);
  PetscValidPointer(baz,1);
  PetscValidPointer((void *)v->data,1);
  PetscValidPointer(*blip,1);
  PetscValidPointer(blop,1);
  return;
}
