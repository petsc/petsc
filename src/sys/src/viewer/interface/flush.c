#ifndef lint
static char vcid[] = "$Id: flush.c,v 1.7 1997/01/06 20:29:42 balay Exp bsmith $";
#endif

#include "petsc.h"  /*I "petsc.h" I*/

struct _Viewer {
  PETSCHEADER
  int         (*flush)(Viewer);
};

#undef __FUNC__  
#define __FUNC__ "ViewerFlush" /* ADIC Ignore */
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


