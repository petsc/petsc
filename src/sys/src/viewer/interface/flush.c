#ifndef lint
static char vcid[] = "$Id: flush.c,v 1.1 1996/01/25 04:26:34 bsmith Exp bsmith $";
#endif

#include "petsc.h"

struct _Viewer {
  PETSCHEADER
  int         (*flush)(Viewer);
};

/*@
   ViewerFlush - Flushes a viewer (i.e. tries to dump all the 
                 data that has been printed through a viewer.

   Input Parameters:
.  viewer - the viewer to be flushed

.seealso: ViewerMatlabOpen(), ViewerFileOpenASCII()

.keywords: Viewer, flush
@*/
int ViewerFlush(Viewer v)
{
  PETSCVALIDHEADERSPECIFIC(v,VIEWER_COOKIE);
  if (v->flush) return (*v->flush)(v);
  return 0;
}


