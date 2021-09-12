#include <petsc/private/drawimpl.h>                        /*I  "petscdraw.h" I*/

static PetscErrorCode PetscDrawCoordinateToPixel_Null(PetscDraw draw,PetscReal x,PetscReal y,int *i,int *j)
{
  PetscFunctionBegin;
  *i = *j = 0;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDrawPixelToCoordinate_Null(PetscDraw draw,int i,int j,PetscReal *x,PetscReal *y)
{
  PetscFunctionBegin;
  *x = *y = 0;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDrawPoint_Null(PetscDraw draw,PetscReal x,PetscReal y,int c)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDrawPointPixel_Null(PetscDraw draw,int x,int y,int c)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDrawLineGetWidth_Null(PetscDraw draw,PetscReal *width)
{
  PetscFunctionBegin;
  if (width) *width = 0.01;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDrawLine_Null(PetscDraw draw,PetscReal xl,PetscReal yl,PetscReal xr,PetscReal yr,int cl)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDrawArrow_Null(PetscDraw draw,PetscReal xl,PetscReal yl,PetscReal xr,PetscReal yr,int cl)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDrawRectangle_Null(PetscDraw draw,PetscReal xl,PetscReal yl,PetscReal xr,PetscReal yr,int c1,int c2,int c3,int c4)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDrawEllipse_Null(PetscDraw Win,PetscReal x,PetscReal y,PetscReal a,PetscReal b,int c)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDrawTriangle_Null(PetscDraw draw,PetscReal X1,PetscReal Y_1,PetscReal X2,PetscReal Y2,PetscReal X3,PetscReal Y3,int c1,int c2,int c3)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDrawStringGetSize_Null(PetscDraw draw,PetscReal *x,PetscReal  *y)
{
  PetscFunctionBegin;
  if (x) *x = 0.01;
  if (y) *y = 0.01;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDrawString_Null(PetscDraw draw,PetscReal x,PetscReal y,int c,const char chrs[])
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDrawStringVertical_Null(PetscDraw draw,PetscReal x,PetscReal y,int c,const char chrs[])
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDrawStringBoxed_Null(PetscDraw draw,PetscReal sxl,PetscReal syl,int sc,int bc,const char text[],PetscReal *w,PetscReal *h)
{
  PetscFunctionBegin;
  if (w) *w = 0.01;
  if (h) *h = 0.01;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDrawGetSingleton_Null(PetscDraw draw,PetscDraw *sdraw)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscDrawOpenNull(PETSC_COMM_SELF,sdraw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDrawRestoreSingleton_Null(PetscDraw draw,PetscDraw *sdraw)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscDrawDestroy(sdraw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static struct _PetscDrawOps DvOps = { NULL,/* PetscDrawSetDoubleBuffer_Null */
                                      NULL,/* PetscDrawFlush_Null */
                                      PetscDrawLine_Null,
                                      NULL,/* PetscDrawLineSetWidth_Null */
                                      PetscDrawLineGetWidth_Null,
                                      PetscDrawPoint_Null,
                                      NULL,/* PetscDrawPointSetSize_Null */
                                      PetscDrawString_Null,
                                      PetscDrawStringVertical_Null,
                                      NULL,/* PetscDrawStringSetSize_Null */
                                      PetscDrawStringGetSize_Null,
                                      NULL,/* PetscDrawSetViewport_Null */
                                      NULL,/* PetscDrawClear_Null */
                                      PetscDrawRectangle_Null,
                                      PetscDrawTriangle_Null,
                                      PetscDrawEllipse_Null,
                                      NULL,/* PetscDrawGetMouseButton_Null */
                                      NULL,/* PetscDrawPause_Null */
                                      NULL,/* PetscDrawBeginPage_Null */
                                      NULL,/* PetscDrawEndPage_Null */
                                      NULL,/* PetscDrawGetPopup_Null */
                                      NULL,/* PetscDrawSetTitle_Null */
                                      NULL,/* PetscDrawCheckResizedWindow_Null */
                                      NULL,/* PetscDrawResizeWindow_Null */
                                      NULL,/* PetscDrawDestroy_Null */
                                      NULL,/* PetscDrawView_Null */
                                      PetscDrawGetSingleton_Null,
                                      PetscDrawRestoreSingleton_Null,
                                      NULL,/* PetscDrawSave_Null */
                                      NULL,/* PetscDrawGetImage_Null */
                                      NULL,/* PetscDrawSetCoordinates_Null */
                                      PetscDrawArrow_Null,
                                      PetscDrawCoordinateToPixel_Null,
                                      PetscDrawPixelToCoordinate_Null,
                                      PetscDrawPointPixel_Null,
                                      PetscDrawStringBoxed_Null};

/*MC
     PETSC_DRAW_NULL - PETSc graphics device that ignores all draw commands

   Level: beginner

.seealso:  PetscDrawOpenNull(), PetscDrawIsNull()
M*/
PETSC_EXTERN PetscErrorCode PetscDrawCreate_Null(PetscDraw);

PETSC_EXTERN PetscErrorCode PetscDrawCreate_Null(PetscDraw draw)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  draw->pause   = 0;
  draw->coor_xl = 0; draw->coor_xr = 1;
  draw->coor_yl = 0; draw->coor_yr = 1;
  draw->port_xl = 0; draw->port_xr = 1;
  draw->port_yl = 0; draw->port_yr = 1;
  ierr = PetscDrawDestroy(&draw->popup);CHKERRQ(ierr);

  ierr = PetscMemcpy(draw->ops,&DvOps,sizeof(DvOps));CHKERRQ(ierr);
  draw->data = NULL;
  PetscFunctionReturn(0);
}

/*@
   PetscDrawOpenNull - Opens a null drawing context. All draw commands to
   it are ignored.

   Input Parameter:
.  comm - MPI communicator

   Output Parameter:
.  draw - the drawing context

   Level: advanced
@*/
PetscErrorCode  PetscDrawOpenNull(MPI_Comm comm,PetscDraw *win)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDrawCreate(comm,NULL,NULL,0,0,1,1,win);CHKERRQ(ierr);
  ierr = PetscDrawSetType(*win,PETSC_DRAW_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   PetscDrawIsNull - Returns PETSC_TRUE if draw is a null draw object.

   Not collective

   Input Parameter:
.  draw - the draw context

   Output Parameter:
.  yes - PETSC_TRUE if it is a null draw object; otherwise PETSC_FALSE

   Level: advanced
@*/
PetscErrorCode  PetscDrawIsNull(PetscDraw draw,PetscBool *yes)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  PetscValidBoolPointer(yes,2);
  ierr = PetscObjectTypeCompare((PetscObject)draw,PETSC_DRAW_NULL,yes);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
