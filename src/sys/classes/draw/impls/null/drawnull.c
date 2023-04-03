#include <petsc/private/drawimpl.h> /*I  "petscdraw.h" I*/

static PetscErrorCode PetscDrawCoordinateToPixel_Null(PetscDraw draw, PetscReal x, PetscReal y, int *i, int *j)
{
  PetscFunctionBegin;
  *i = *j = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDrawPixelToCoordinate_Null(PetscDraw draw, int i, int j, PetscReal *x, PetscReal *y)
{
  PetscFunctionBegin;
  *x = *y = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDrawPoint_Null(PetscDraw draw, PetscReal x, PetscReal y, int c)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDrawPointPixel_Null(PetscDraw draw, int x, int y, int c)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDrawLineGetWidth_Null(PetscDraw draw, PetscReal *width)
{
  PetscFunctionBegin;
  if (width) *width = 0.01;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDrawLine_Null(PetscDraw draw, PetscReal xl, PetscReal yl, PetscReal xr, PetscReal yr, int cl)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDrawArrow_Null(PetscDraw draw, PetscReal xl, PetscReal yl, PetscReal xr, PetscReal yr, int cl)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDrawRectangle_Null(PetscDraw draw, PetscReal xl, PetscReal yl, PetscReal xr, PetscReal yr, int c1, int c2, int c3, int c4)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDrawEllipse_Null(PetscDraw Win, PetscReal x, PetscReal y, PetscReal a, PetscReal b, int c)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDrawTriangle_Null(PetscDraw draw, PetscReal X1, PetscReal Y_1, PetscReal X2, PetscReal Y2, PetscReal X3, PetscReal Y3, int c1, int c2, int c3)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDrawStringGetSize_Null(PetscDraw draw, PetscReal *x, PetscReal *y)
{
  PetscFunctionBegin;
  if (x) *x = 0.01;
  if (y) *y = 0.01;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDrawString_Null(PetscDraw draw, PetscReal x, PetscReal y, int c, const char chrs[])
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDrawStringVertical_Null(PetscDraw draw, PetscReal x, PetscReal y, int c, const char chrs[])
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDrawStringBoxed_Null(PetscDraw draw, PetscReal sxl, PetscReal syl, int sc, int bc, const char text[], PetscReal *w, PetscReal *h)
{
  PetscFunctionBegin;
  if (w) *w = 0.01;
  if (h) *h = 0.01;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDrawGetSingleton_Null(PetscDraw draw, PetscDraw *sdraw)
{
  PetscFunctionBegin;
  PetscCall(PetscDrawOpenNull(PETSC_COMM_SELF, sdraw));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDrawRestoreSingleton_Null(PetscDraw draw, PetscDraw *sdraw)
{
  PetscFunctionBegin;
  PetscCall(PetscDrawDestroy(sdraw));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static struct _PetscDrawOps DvOps = {NULL, /* PetscDrawSetDoubleBuffer_Null */
                                     NULL, /* PetscDrawFlush_Null */
                                     PetscDrawLine_Null,
                                     NULL, /* PetscDrawLineSetWidth_Null */
                                     PetscDrawLineGetWidth_Null,
                                     PetscDrawPoint_Null,
                                     NULL, /* PetscDrawPointSetSize_Null */
                                     PetscDrawString_Null,
                                     PetscDrawStringVertical_Null,
                                     NULL, /* PetscDrawStringSetSize_Null */
                                     PetscDrawStringGetSize_Null,
                                     NULL, /* PetscDrawSetViewport_Null */
                                     NULL, /* PetscDrawClear_Null */
                                     PetscDrawRectangle_Null,
                                     PetscDrawTriangle_Null,
                                     PetscDrawEllipse_Null,
                                     NULL, /* PetscDrawGetMouseButton_Null */
                                     NULL, /* PetscDrawPause_Null */
                                     NULL, /* PetscDrawBeginPage_Null */
                                     NULL, /* PetscDrawEndPage_Null */
                                     NULL, /* PetscDrawGetPopup_Null */
                                     NULL, /* PetscDrawSetTitle_Null */
                                     NULL, /* PetscDrawCheckResizedWindow_Null */
                                     NULL, /* PetscDrawResizeWindow_Null */
                                     NULL, /* PetscDrawDestroy_Null */
                                     NULL, /* PetscDrawView_Null */
                                     PetscDrawGetSingleton_Null,
                                     PetscDrawRestoreSingleton_Null,
                                     NULL, /* PetscDrawSave_Null */
                                     NULL, /* PetscDrawGetImage_Null */
                                     NULL, /* PetscDrawSetCoordinates_Null */
                                     PetscDrawArrow_Null,
                                     PetscDrawCoordinateToPixel_Null,
                                     PetscDrawPixelToCoordinate_Null,
                                     PetscDrawPointPixel_Null,
                                     PetscDrawStringBoxed_Null,
                                     NULL /* PetscDrawSetVisible_Null */};

/*MC
     PETSC_DRAW_NULL - PETSc graphics device that ignores all draw commands

   Level: beginner

   Note:
    A `PETSC_DRAW_NULL` is useful in places where `PetscDraw` routines are called but no graphics window, for example, is available.

.seealso: `PetscDraw`, `PetscDrawOpenNull()`, `PETSC_DRAW_X`, `PetscDrawOpenNull()`, `PetscDrawIsNull()`
M*/
PETSC_EXTERN PetscErrorCode PetscDrawCreate_Null(PetscDraw);

PETSC_EXTERN PetscErrorCode PetscDrawCreate_Null(PetscDraw draw)
{
  PetscFunctionBegin;
  draw->pause   = 0;
  draw->coor_xl = 0;
  draw->coor_xr = 1;
  draw->coor_yl = 0;
  draw->coor_yr = 1;
  draw->port_xl = 0;
  draw->port_xr = 1;
  draw->port_yl = 0;
  draw->port_yr = 1;
  PetscCall(PetscDrawDestroy(&draw->popup));

  PetscCall(PetscMemcpy(draw->ops, &DvOps, sizeof(DvOps)));
  draw->data = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   PetscDrawOpenNull - Opens a null drawing context. All draw commands to
   it are ignored.

   Input Parameter:
.  comm - MPI communicator

   Output Parameter:
.  draw - the drawing context

   Level: advanced

.seealso: `PetscDraw`, `PetscDrawIsNull()`, `PETSC_DRAW_NULL`, `PetscDrawOpenX()`, `PetscDrawIsNull()`
@*/
PetscErrorCode PetscDrawOpenNull(MPI_Comm comm, PetscDraw *win)
{
  PetscFunctionBegin;
  PetscCall(PetscDrawCreate(comm, NULL, NULL, 0, 0, 1, 1, win));
  PetscCall(PetscDrawSetType(*win, PETSC_DRAW_NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   PetscDrawIsNull - Returns `PETSC_TRUE` if draw is a null draw object.

   Not Collective

   Input Parameter:
.  draw - the draw context

   Output Parameter:
.  yes - `PETSC_TRUE` if it is a null draw object; otherwise `PETSC_FALSE`

   Level: advanced

.seealso: `PetscDraw`, `PETSC_DRAW_NULL`, `PetscDrawOpenX()`, `PetscDrawIsNull()`
@*/
PetscErrorCode PetscDrawIsNull(PetscDraw draw, PetscBool *yes)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw, PETSC_DRAW_CLASSID, 1);
  PetscValidBoolPointer(yes, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)draw, PETSC_DRAW_NULL, yes));
  PetscFunctionReturn(PETSC_SUCCESS);
}
