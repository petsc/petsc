#ifndef lint
static char vcid[] = "$Id: view.c,v 1.6 1995/09/04 17:25:37 bsmith Exp bsmith $";
#endif

#include "petsc.h"

/*@C
   ViewerDestroy - Destroys a viewer.

   Input Parameters:
.  viewer - the viewer to be destroyed.

.seealso: ViewerMatlabOpen(), ViewerFileOpenASCII()

.keywords: Viewer, destroy
@*/
int ViewerDestroy(Viewer v)
{
  PetscObject o = (PetscObject) v;
  if (!v) SETERRQ(1,"ViewerDestroy: trying to destroy null viewer");
  return (*o->destroy)(o);
}


