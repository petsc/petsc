#ifndef lint
static char vcid[] = "$Id: drawv.c,v 1.8 1996/09/27 19:38:38 curfman Exp bsmith $";
#endif

#include "petsc.h"
#include "src/draw/drawimpl.h" /*I "draw.h" I*/

int ViewerDestroy_Draw(PetscObject obj)
{
  int    ierr;
  Viewer v = (Viewer) obj;

  ierr = DrawLGDestroy(v->drawlg); CHKERRQ(ierr);
  ierr = DrawDestroy(v->draw); CHKERRQ(ierr);
  PLogObjectDestroy(obj);
  PetscHeaderDestroy(obj);
  return 0;
}

int ViewerFlush_Draw(Viewer v)
{
  return DrawSyncFlush(v->draw);
}

/*@C
    ViewerDrawGetDraw - Returns Draw object from Viewer object.
    This Draw object may then be used to perform graphics using 
    DrawXXX() commands.

    Input Parameter:
.   viewer - the viewer (created with ViewerDrawOpenX()

    Ouput Parameter:
.   draw - the draw object

.keywords: viewer, draw, get

.seealso: ViewerDrawGetLG()
@*/
int ViewerDrawGetDraw(Viewer v, Draw *draw)
{
  PetscValidHeaderSpecific(v, VIEWER_COOKIE);
  if (v->type != DRAW_VIEWER) SETERRQ(1,"ViewerDrawGetDraw:Must be draw");
  *draw = v->draw;
  return 0;
}

/*@C
    ViewerDrawGetDrawLG - Returns DrawLG object from Viewer object.
    This DrawLG object may then be used to perform graphics using 
    DrawLGXXX() commands.

    Input Parameter:
.   viewer - the viewer (created with ViewerDrawOpenX()

    Ouput Parameter:
.   draw - the draw line graph object


.keywords: viewer, draw, get, line graph

.seealso: ViewerDrawGetDraw()
@*/
int ViewerDrawGetDrawLG(Viewer v, DrawLG *drawlg)
{
  PetscValidHeaderSpecific(v, VIEWER_COOKIE);
  if (v->type != DRAW_VIEWER) SETERRQ(1,"ViewerDrawGetDraw:Must be draw");
  *drawlg = v->drawlg;
  return 0;
}





