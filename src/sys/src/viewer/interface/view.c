#ifndef lint
static char vcid[] = "$Id: view.c,v 1.2 1995/06/23 12:41:45 bsmith Exp curfman $";
#endif

#include "ptscimpl.h"

/*@
   ViewerDestroy - Destroys a viewer.

   Input Parameters:
.  viewer - the viewer to be destroyed.

.seealso: ViewerMatlabOpen(), ViewerFileOpen(), ViewerFileOpenSync()

.keywords: Viewer, destroy
@*/
int ViewerDestroy(Viewer v)
{
  PetscObject o = (PetscObject) v;
  if (!v) SETERRQ(1,"ViewerDestroy: trying to destroy null viewer");
  return (*o->destroy)(o);
}


