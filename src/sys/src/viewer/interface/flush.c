#ifndef lint
static char vcid[] = "$Id: view.c,v 1.8 1995/10/01 21:53:22 bsmith Exp $";
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
  if (!v) SETERRQ(1,"ViewerDestroy:null viewer");
  return (*o->destroy)(o);
}


