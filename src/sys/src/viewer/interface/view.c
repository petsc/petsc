#ifndef lint
static char vcid[] = "$Id: view.c,v 1.5 1995/08/22 19:38:26 curfman Exp bsmith $";
#endif

#include "petsc.h"

/*@C
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


