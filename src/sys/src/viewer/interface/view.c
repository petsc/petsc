#ifndef lint
static char vcid[] = "$Id: view.c,v 1.4 1995/08/07 22:01:26 bsmith Exp curfman $";
#endif

#include "petsc.h"

/*@
   ViewerDestroy - Destroys a viewer.

   Input Parameters:
.  viewer - the viewer to be destroyed.

.seealso: ViewerMatlabOpen(), ViewerFileOpen()

.keywords: Viewer, destroy
@*/
int ViewerDestroy(Viewer v)
{
  PetscObject o = (PetscObject) v;
  if (!v) SETERRQ(1,"ViewerDestroy: trying to destroy null viewer");
  return (*o->destroy)(o);
}


