#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: drawv.c,v 1.20 1997/08/22 15:15:58 bsmith Exp bsmith $";
#endif

#include "petsc.h"
#include "src/draw/drawimpl.h" /*I "draw.h" I*/

#undef __FUNC__  
#define __FUNC__ "ViewerDestroy_Draw" 
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

#undef __FUNC__  
#define __FUNC__ "ViewerFlush_Draw" 
int ViewerFlush_Draw(Viewer v)
{
  return DrawSynchronizedFlush(v->draw);
}

#undef __FUNC__  
#define __FUNC__ "ViewerDrawGetDraw" 
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
    SETERRQ(PETSC_ERR_ARG_WRONG,0,"Must be draw type viewer");
  }
  *draw = v->draw;
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "ViewerDrawGetDrawLG" 
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
    SETERRQ(PETSC_ERR_ARG_WRONG,0,"Must be draw type viewer");
  }
  *drawlg = v->drawlg;
  return 0;
}

