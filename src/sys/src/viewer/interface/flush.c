#ifndef lint
static char vcid[] = "$Id: flush.c,v 1.3 1996/03/19 21:28:55 bsmith Exp curfman $";
#endif

#include "petsc.h"

struct _Viewer {
  PETSCHEADER
  int         (*flush)(Viewer);
};

/*@
   ViewerFlush - Flushes a viewer (i.e. tries to dump all the 
   data that has been printed through a viewer).

   Input Parameters:
.  viewer - the viewer to be flushed

.keywords: Viewer, flush

.seealso: ViewerMatlabOpen(), ViewerFileOpenASCII()
@*/
int ViewerFlush(Viewer v)
{
  PetscValidHeaderSpecific(v,VIEWER_COOKIE);
  if (v->flush) return (*v->flush)(v);
  return 0;
}


