#ifndef lint
static char vcid[] = "$Id: drawv.c,v 1.12 1996/12/08 23:56:03 bsmith Exp balay $";
#endif

#include "petsc.h"
#include "src/draw/drawimpl.h" /*I "draw.h" I*/

#undef __FUNCTION__  
#define __FUNCTION__ "ViewerDestroy_Draw"
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

#undef __FUNCTION__  
#define __FUNCTION__ "ViewerFlush_Draw"
int ViewerFlush_Draw(Viewer v)
{
  return DrawSyncFlush(v->draw);
}

#undef __FUNCTION__  
#define __FUNCTION__ "ViewerDrawGetDraw"
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
  if (v->type != DRAW_VIEWER) {
    SETERRQ(PETSC_ERR_ARG_WRONG,"ViewerDrawGetDraw:Must be draw type viewer");
  }
  *draw = v->draw;
  return 0;
}

#undef __FUNCTION__  
#define __FUNCTION__ "ViewerDrawGetDrawLG"
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
  if (v->type != DRAW_VIEWER) {
    SETERRQ(PETSC_ERR_ARG_WRONG,"ViewerDrawGetDraw:Must be draw type viewer");
  }
  *drawlg = v->drawlg;
  return 0;
}





