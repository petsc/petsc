#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: drawv.c,v 1.24 1998/04/13 17:46:34 bsmith Exp curfman $";
#endif

#include "petsc.h"
#include "src/draw/drawimpl.h" /*I "draw.h" I*/

#undef __FUNC__  
#define __FUNC__ "ViewerDestroy_Draw" 
int ViewerDestroy_Draw(Viewer v)
{
  int    ierr;

  PetscFunctionBegin;
  if (v->drawaxis) {ierr = DrawAxisDestroy(v->drawaxis); CHKERRQ(ierr);}
  if (v->drawlg)   {ierr = DrawLGDestroy(v->drawlg); CHKERRQ(ierr);}
  ierr = DrawDestroy(v->draw); CHKERRQ(ierr);
  PLogObjectDestroy((PetscObject)v);
  PetscHeaderDestroy((PetscObject)v);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewerFlush_Draw" 
int ViewerFlush_Draw(Viewer v)
{
  int ierr;
  PetscFunctionBegin;
  ierr = DrawSynchronizedFlush(v->draw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewerDrawGetDraw" 
/*@C
    ViewerDrawGetDraw - Returns Draw object from Viewer object.
    This Draw object may then be used to perform graphics using 
    DrawXXX() commands.

    Not collective (but Draw returned will be parallel object if Viewer is)

    Input Parameter:
.   viewer - the viewer (created with ViewerDrawOpenX()

    Ouput Parameter:
.   draw - the draw object

.keywords: viewer, draw, get

.seealso: ViewerDrawGetLG(), ViewerDrawGetAxis()
@*/
int ViewerDrawGetDraw(Viewer v, Draw *draw)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v, VIEWER_COOKIE);
  if (v->type != DRAW_VIEWER) {
    SETERRQ(PETSC_ERR_ARG_WRONG,0,"Must be draw type viewer");
  }
  *draw = v->draw;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewerDrawGetDrawLG" 
/*@C
    ViewerDrawGetDrawLG - Returns DrawLG object from Viewer object.
    This DrawLG object may then be used to perform graphics using 
    DrawLGXXX() commands.

    Not Collective (but DrawLG object will be parallel if Viewer is)

    Input Parameter:
.   viewer - the viewer (created with ViewerDrawOpenX()

    Ouput Parameter:
.   draw - the draw line graph object

.keywords: viewer, draw, get, line graph

.seealso: ViewerDrawGetDraw(), ViewerDrawGetAxis()
@*/
int ViewerDrawGetDrawLG(Viewer v, DrawLG *drawlg)
{
  int ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v, VIEWER_COOKIE);
  if (v->type != DRAW_VIEWER) {
    SETERRQ(PETSC_ERR_ARG_WRONG,0,"Must be draw type viewer");
  }
  if (!v->drawlg) {
    ierr = DrawLGCreate(v->draw,1,&v->drawlg);CHKERRQ(ierr);
    PLogObjectParent(v,v->drawlg);
  }
  *drawlg = v->drawlg;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "ViewerDrawGetDrawAxis" 
/*@C
    ViewerDrawGetDrawAxis - Returns DrawAxis object from Viewer object.
    This DrawAxis object may then be used to perform graphics using 
    DrawAxisXXX() commands.

    Not Collective (but DrawAxis object will be parallel if Viewer is)

    Input Parameter:
.   viewer - the viewer (created with ViewerDrawOpenX()

    Ouput Parameter:
.   drawaxis - the draw axis object

.keywords: viewer, draw, get, line graph

.seealso: ViewerDrawGetDraw(), ViewerDrawGetLG()
@*/
int ViewerDrawGetDrawAxis(Viewer v, DrawAxis *drawaxis)
{
  int ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v, VIEWER_COOKIE);
  if (v->type != DRAW_VIEWER) {
    SETERRQ(PETSC_ERR_ARG_WRONG,0,"Must be draw type viewer");
  }
  if (!v->drawaxis) {
    ierr = DrawAxisCreate(v->draw,&v->drawaxis);CHKERRQ(ierr);
    PLogObjectParent(v,v->drawaxis);
  }
  *drawaxis = v->drawaxis;
  PetscFunctionReturn(0);
}
