#ifndef lint
static char vcid[] = "$Id: view.c,v 1.1 1995/06/21 22:27:11 bsmith Exp bsmith $";
#endif

#include "ptscimpl.h"

/*@
       ViewerDestroy - Destroys a viewer.

  Input Parameters:
.   viewer - the viewer to be destroyed.

.seealso: ViewerMatlabOpen(), ViewerFileOpen(), ViewerFileOpenSync()
@*/
int ViewerDestroy(Viewer v)
{
  PetscObject o = (PetscObject) v;
  if (!v) SETERRQ(1,"ViewerDestroy: trying to destroy null viewer");
  return (*o->destroy)(o);
}
