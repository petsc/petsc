#ifndef lint
static char vcid[] = "$Id: drawv.c,v 1.2 1996/03/08 05:48:11 bsmith Exp bsmith $";
#endif

#include "petsc.h"
#include "drawimpl.h"

int ViewerDestroy_Draw(PetscObject obj)
{
  int    ierr;
  Viewer v = (Viewer) obj;

  ierr = DrawDestroy(v->draw); CHKERRQ(ierr);
  PLogObjectDestroy(obj);
  PetscHeaderDestroy(obj);
  return 0;
}

int ViewerFlush_Draw(Viewer v)
{
  return DrawSyncFlush(v->draw);
}

/*@
    ViewerDrawGetDraw - Returns Draw object from Viewer object.
      This Draw object may then be used to perform graphics using 
      DrawXXX() commands.

  Input Parameter:
.   viewer - the viewer (created with ViewerDrawOpenX()

  Ouput Parameter:
.   draw - the draw object

@*/
int ViewerDrawGetDraw(Viewer v, Draw *draw)
{
  PetscValidHeaderSpecific(v, VIEWER_COOKIE);
  if (v->type != DRAW_VIEWER) SETERRQ(1,"ViewerDrawGetDraw:Must be draw");
  *draw = v->draw;
  return 0;
}





