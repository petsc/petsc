#include <petsc/private/drawimpl.h> /*I "petscdraw.h" I*/

PetscClassId PETSC_DRAW_CLASSID;

static PetscBool PetscDrawPackageInitialized = PETSC_FALSE;
/*@
  PetscDrawFinalizePackage - This function destroys everything in the PETSc interface to the `PetscDraw` package. It is
  called from `PetscFinalize()`.

  Level: developer

.seealso: `PetscDraw`, `PetscFinalize()`
@*/
PetscErrorCode PetscDrawFinalizePackage(void)
{
  PetscFunctionBegin;
  PetscCall(PetscFunctionListDestroy(&PetscDrawList));
  PetscDrawPackageInitialized = PETSC_FALSE;
  PetscDrawRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDrawInitializePackage - This function initializes everything in the `PetscDraw` package. It is called
  from PetscDLLibraryRegister_petsc() when using dynamic libraries, and on the call to `PetscInitialize()`
  when using shared or static libraries.

  Level: developer

.seealso: `PetscDraw`, `PetscInitialize()`
@*/
PetscErrorCode PetscDrawInitializePackage(void)
{
  char      logList[256];
  PetscBool opt, pkg;

  PetscFunctionBegin;
  if (PetscDrawPackageInitialized == PETSC_TRUE) PetscFunctionReturn(PETSC_SUCCESS);
  PetscDrawPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  PetscCall(PetscClassIdRegister("Draw", &PETSC_DRAW_CLASSID));
  PetscCall(PetscClassIdRegister("Draw Axis", &PETSC_DRAWAXIS_CLASSID));
  PetscCall(PetscClassIdRegister("Line Graph", &PETSC_DRAWLG_CLASSID));
  PetscCall(PetscClassIdRegister("Histogram", &PETSC_DRAWHG_CLASSID));
  PetscCall(PetscClassIdRegister("Bar Graph", &PETSC_DRAWBAR_CLASSID));
  PetscCall(PetscClassIdRegister("Scatter Plot", &PETSC_DRAWSP_CLASSID));
  /* Register Constructors */
  PetscCall(PetscDrawRegisterAll());
  /* Process Info */
  {
    PetscClassId classids[6];

    classids[0] = PETSC_DRAW_CLASSID;
    classids[1] = PETSC_DRAWAXIS_CLASSID;
    classids[2] = PETSC_DRAWLG_CLASSID;
    classids[3] = PETSC_DRAWHG_CLASSID;
    classids[4] = PETSC_DRAWBAR_CLASSID;
    classids[5] = PETSC_DRAWSP_CLASSID;
    PetscCall(PetscInfoProcessClass("draw", 6, classids));
  }
  /* Process summary exclusions */
  PetscCall(PetscOptionsGetString(NULL, NULL, "-log_exclude", logList, sizeof(logList), &opt));
  if (opt == PETSC_TRUE) {
    PetscCall(PetscStrInList("draw", logList, ',', &pkg));
    if (pkg == PETSC_TRUE) {
      PetscCall(PetscLogEventExcludeClass(PETSC_DRAW_CLASSID));
      PetscCall(PetscLogEventExcludeClass(PETSC_DRAWAXIS_CLASSID));
      PetscCall(PetscLogEventExcludeClass(PETSC_DRAWLG_CLASSID));
      PetscCall(PetscLogEventExcludeClass(PETSC_DRAWHG_CLASSID));
      PetscCall(PetscLogEventExcludeClass(PETSC_DRAWBAR_CLASSID));
      PetscCall(PetscLogEventExcludeClass(PETSC_DRAWSP_CLASSID));
    }
  }
  /* Register package finalizer */
  PetscCall(PetscRegisterFinalize(PetscDrawFinalizePackage));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDrawResizeWindow - Allows one to resize a window from a program.

  Collective

  Input Parameters:
+ draw - the window
. w    - the new width of the window
- h    - the new height of the window

  Level: intermediate

.seealso: `PetscDraw`, `PetscDrawCheckResizedWindow()`
@*/
PetscErrorCode PetscDrawResizeWindow(PetscDraw draw, int w, int h)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw, PETSC_DRAW_CLASSID, 1);
  PetscValidLogicalCollectiveInt(draw, w, 2);
  PetscValidLogicalCollectiveInt(draw, h, 3);
  PetscTryTypeMethod(draw, resizewindow, w, h);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDrawGetWindowSize - Gets the size of the window.

  Not Collective

  Input Parameter:
. draw - the window

  Output Parameters:
+ w - the window width
- h - the window height

  Level: intermediate

.seealso: `PetscDraw`, `PetscDrawResizeWindow()`, `PetscDrawCheckResizedWindow()`
@*/
PetscErrorCode PetscDrawGetWindowSize(PetscDraw draw, int *w, int *h)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw, PETSC_DRAW_CLASSID, 1);
  if (w != NULL) PetscAssertPointer(w, 2);
  if (h != NULL) PetscAssertPointer(h, 3);
  if (w != NULL) *w = draw->w;
  if (h != NULL) *h = draw->h;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDrawCheckResizedWindow - Checks if the user has resized the window.

  Collective

  Input Parameter:
. draw - the window

  Level: advanced

.seealso: `PetscDraw`, `PetscDrawResizeWindow()`
@*/
PetscErrorCode PetscDrawCheckResizedWindow(PetscDraw draw)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw, PETSC_DRAW_CLASSID, 1);
  PetscTryTypeMethod(draw, checkresizedwindow);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDrawGetTitle - Gets pointer to title of a `PetscDraw` context.

  Not Collective

  Input Parameter:
. draw - the graphics context

  Output Parameter:
. title - the title

  Level: intermediate

.seealso: `PetscDraw`, `PetscDrawSetTitle()`
@*/
PetscErrorCode PetscDrawGetTitle(PetscDraw draw, const char *title[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw, PETSC_DRAW_CLASSID, 1);
  PetscAssertPointer(title, 2);
  *title = draw->title;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDrawSetTitle - Sets the title of a `PetscDraw` context.

  Collective

  Input Parameters:
+ draw  - the graphics context
- title - the title

  Level: intermediate

  Notes:
  The title is positioned in the windowing system title bar for the window. Hence it will not be saved with -draw_save
  in the image.

  A copy of the string is made, so you may destroy the
  title string after calling this routine.

  You can use `PetscDrawAxisSetLabels()` to indicate a title within the window

.seealso: `PetscDraw`, `PetscDrawGetTitle()`, `PetscDrawAppendTitle()`
@*/
PetscErrorCode PetscDrawSetTitle(PetscDraw draw, const char title[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw, PETSC_DRAW_CLASSID, 1);
  PetscAssertPointer(title, 2);
  PetscCall(PetscFree(draw->title));
  PetscCall(PetscStrallocpy(title, &draw->title));
  PetscTryTypeMethod(draw, settitle, draw->title);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDrawAppendTitle - Appends to the title of a `PetscDraw` context.

  Collective

  Input Parameters:
+ draw  - the graphics context
- title - the title

  Level: advanced

  Note:
  A copy of the string is made, so you may destroy the
  title string after calling this routine.

.seealso: `PetscDraw`, `PetscDrawSetTitle()`, `PetscDrawGetTitle()`
@*/
PetscErrorCode PetscDrawAppendTitle(PetscDraw draw, const char title[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw, PETSC_DRAW_CLASSID, 1);
  if (title != NULL) PetscAssertPointer(title, 2);
  if (title == NULL || !title[0]) PetscFunctionReturn(PETSC_SUCCESS);

  if (draw->title != NULL) {
    size_t len1, len2, new_len;
    PetscCall(PetscStrlen(draw->title, &len1));
    PetscCall(PetscStrlen(title, &len2));
    new_len = len1 + len2 + 1;
    PetscCall(PetscRealloc(new_len * sizeof(*draw->title), &draw->title));
    PetscCall(PetscStrncpy(draw->title + len1, title, len2 + 1));
  } else {
    PetscCall(PetscStrallocpy(title, &draw->title));
  }
  PetscTryTypeMethod(draw, settitle, draw->title);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDrawDestroy_Private(PetscDraw draw)
{
  PetscFunctionBegin;
  if (draw->ops->save == NULL && draw->ops->getimage == NULL) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscDrawSaveMovie(draw));
  if (draw->savefinalfilename != NULL) {
    draw->savesinglefile = PETSC_TRUE;
    PetscCall(PetscDrawSetSave(draw, draw->savefinalfilename));
    PetscCall(PetscDrawSave(draw));
  }
  PetscCall(PetscBarrier((PetscObject)draw));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDrawDestroy - Deletes a draw context.

  Collective

  Input Parameter:
. draw - the drawing context

  Level: beginner

.seealso: `PetscDraw`, `PetscDrawCreate()`
@*/
PetscErrorCode PetscDrawDestroy(PetscDraw *draw)
{
  PetscFunctionBegin;
  if (*draw == NULL) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific(*draw, PETSC_DRAW_CLASSID, 1);
  if (--((PetscObject)*draw)->refct > 0) PetscFunctionReturn(PETSC_SUCCESS);

  if ((*draw)->pause == -2) {
    (*draw)->pause = -1;
    PetscCall(PetscDrawPause(*draw));
  }

  /* if memory was published then destroy it */
  PetscCall(PetscObjectSAWsViewOff((PetscObject)*draw));

  PetscCall(PetscDrawDestroy_Private(*draw));

  PetscTryTypeMethod(*draw, destroy);
  PetscCall(PetscDrawDestroy(&(*draw)->popup));
  PetscCall(PetscFree((*draw)->title));
  PetscCall(PetscFree((*draw)->display));
  PetscCall(PetscFree((*draw)->savefilename));
  PetscCall(PetscFree((*draw)->saveimageext));
  PetscCall(PetscFree((*draw)->savemovieext));
  PetscCall(PetscFree((*draw)->savefinalfilename));
  PetscCall(PetscHeaderDestroy(draw));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDrawGetPopup - Creates a popup window associated with a `PetscDraw` window.

  Collective

  Input Parameter:
. draw - the original window

  Output Parameter:
. popup - the new popup window

  Level: advanced

.seealso: `PetscDraw`, `PetscDrawScalePopup()`, `PetscDrawCreate()`
@*/
PetscErrorCode PetscDrawGetPopup(PetscDraw draw, PetscDraw *popup)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw, PETSC_DRAW_CLASSID, 1);
  PetscAssertPointer(popup, 2);

  if (draw->popup != NULL) *popup = draw->popup;
  else if (draw->ops->getpopup != NULL) {
    PetscUseTypeMethod(draw, getpopup, popup);
    if (*popup != NULL) {
      PetscCall(PetscObjectSetOptionsPrefix((PetscObject)*popup, "popup_"));
      (*popup)->pause = 0.0;
      PetscCall(PetscDrawSetFromOptions(*popup));
    }
  } else *popup = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDrawSetDisplay - Sets the display where a `PetscDraw` object will be displayed

  Input Parameters:
+ draw    - the drawing context
- display - the X windows display

  Level: advanced

.seealso: `PetscDraw`, `PetscDrawOpenX()`, `PetscDrawCreate()`
@*/
PetscErrorCode PetscDrawSetDisplay(PetscDraw draw, const char display[])
{
  PetscFunctionBegin;
  PetscCall(PetscFree(draw->display));
  PetscCall(PetscStrallocpy(display, &draw->display));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDrawSetDoubleBuffer - Sets a window to be double buffered.

  Logically Collective

  Input Parameter:
. draw - the drawing context

  Level: intermediate

.seealso: `PetscDraw`, `PetscDrawOpenX()`, `PetscDrawCreate()`
@*/
PetscErrorCode PetscDrawSetDoubleBuffer(PetscDraw draw)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw, PETSC_DRAW_CLASSID, 1);
  PetscTryTypeMethod(draw, setdoublebuffer);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDrawGetSingleton - Gain access to a `PetscDraw` object as if it were owned
  by the one process.

  Collective

  Input Parameter:
. draw - the original window

  Output Parameter:
. sdraw - the singleton window

  Level: advanced

.seealso: `PetscDraw`, `PetscDrawRestoreSingleton()`, `PetscViewerGetSingleton()`, `PetscViewerRestoreSingleton()`
@*/
PetscErrorCode PetscDrawGetSingleton(PetscDraw draw, PetscDraw *sdraw)
{
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw, PETSC_DRAW_CLASSID, 1);
  PetscAssertPointer(sdraw, 2);

  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)draw), &size));
  if (size == 1) {
    PetscCall(PetscObjectReference((PetscObject)draw));
    *sdraw = draw;
  } else PetscUseTypeMethod(draw, getsingleton, sdraw);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDrawRestoreSingleton - Remove access to a `PetscDraw` object obtained with `PetscDrawGetSingleton()`
  by the one process.

  Collective

  Input Parameters:
+ draw  - the original window
- sdraw - the singleton window

  Level: advanced

.seealso: `PetscDraw`, `PetscDrawGetSingleton()`, `PetscViewerGetSingleton()`, `PetscViewerRestoreSingleton()`
@*/
PetscErrorCode PetscDrawRestoreSingleton(PetscDraw draw, PetscDraw *sdraw)
{
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw, PETSC_DRAW_CLASSID, 1);
  PetscAssertPointer(sdraw, 2);
  PetscValidHeaderSpecific(*sdraw, PETSC_DRAW_CLASSID, 2);

  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)draw), &size));
  if (size == 1) {
    PetscCheck(draw == *sdraw, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot restore singleton, it is not the parent draw");
    PetscCall(PetscObjectDereference((PetscObject)draw));
    *sdraw = NULL;
  } else PetscUseTypeMethod(draw, restoresingleton, sdraw);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscDrawSetVisible - Sets if the drawing surface (the 'window') is visible on its display.

  Input Parameters:
+ draw    - the drawing window
- visible - if the surface should be visible

  Level: intermediate

.seealso: `PetscDraw`
@*/
PetscErrorCode PetscDrawSetVisible(PetscDraw draw, PetscBool visible)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw, PETSC_DRAW_CLASSID, 1);
  PetscTryTypeMethod(draw, setvisible, visible);
  PetscFunctionReturn(PETSC_SUCCESS);
}

// RGB values of the PETSC_DRAW_BASIC_COLORS named colors, for implementations that manage their own palette
const unsigned char PetscDrawBasicColors[PETSC_DRAW_BASIC_COLORS][3] = {
  {255, 255, 255}, // white
  {0,   0,   0  }, // black
  {255, 0,   0  }, // red
  {0,   255, 0  }, // green
  {0,   255, 255}, // cyan
  {0,   0,   255}, // blue
  {255, 0,   255}, // magenta
  {127, 255, 212}, // aquamarine
  {34,  139, 34 }, // forestgreen
  {255, 165, 0  }, // orange
  {238, 130, 238}, // violet
  {165, 42,  42 }, // brown
  {255, 192, 203}, // pink
  {255, 127, 80 }, // coral
  {190, 190, 190}, // gray
  {255, 255, 0  }, // yellow
  {255, 215, 0  }, // gold
  {255, 182, 193}, // lightpink
  {72,  209, 204}, // mediumturquoise
  {240, 230, 140}, // khaki
  {105, 105, 105}, // dimgray
  {154, 205, 50 }, // yellowgreen
  {135, 206, 235}, // skyblue
  {0,   100, 0  }, // darkgreen
  {0,   0,   128}, // navyblue
  {244, 164, 96 }, // sandybrown
  {95,  158, 160}, // cadetblue
  {176, 224, 230}, // powderblue
  {255, 20,  147}, // deeppink
  {216, 191, 216}, // thistle
  {50,  205, 50 }, // limegreen
  {255, 240, 245}, // lavenderblush
  {221, 160, 221}, // plum
};

/*
  PetscDrawSetUpWindowGeometry_Private - determine the location and size of a new window from the options database

  Input Parameters:
+ draw     - the `PetscDraw`, its `x`, `y`, `w`, `h` give the requested location and size
. xmax     - width of the display in pixels
. ymax     - height of the display in pixels
- dvirtual - the draw has no window, so `-geometry` and window placement do not apply

  Output Parameters:
+ x - horizontal location of the window
. y - vertical location of the window
. w - width of the window
- h - height of the window

  Note:
  Windows with `PETSC_DECIDE` location are tiled from the upper left corner of the display.
  The tiling state is shared by all `PetscDraw` implementations that call this function.
*/
PetscErrorCode PetscDrawSetUpWindowGeometry_Private(PetscDraw draw, int xmax, int ymax, PetscBool dvirtual, int *x, int *y, int *w, int *h)
{
  static int xavailable = 0, yavailable = 0, ybottom = 0;
  PetscBool  set;
  PetscInt   xywh[4], osize = 4, nsizes = 2;
  PetscReal  sizes[2] = {.3, .3};

  PetscFunctionBegin;
  *x = draw->x;
  *y = draw->y;
  *w = draw->w;
  *h = draw->h;
  // allow user to set size of drawable
  PetscCall(PetscOptionsGetRealArray(((PetscObject)draw)->options, ((PetscObject)draw)->prefix, "-draw_size", sizes, &nsizes, &set));
  if (set == PETSC_TRUE && nsizes == 1 && sizes[0] > 1.0) sizes[1] = sizes[0];
  if (set == PETSC_TRUE) {
    if (sizes[0] > 1.0) *w = (int)sizes[0];
    else if (sizes[0] == 1.0) *w = PETSC_DRAW_FULL_SIZE;
    else if (sizes[0] == .5) *w = PETSC_DRAW_HALF_SIZE;
    else if (PetscEqualReal(sizes[0], .3) == PETSC_TRUE) *w = PETSC_DRAW_THIRD_SIZE; // sizes == 0.3 will never be true
    else if (sizes[0] == .25) *w = PETSC_DRAW_QUARTER_SIZE;
    if (sizes[1] > 1.0) *h = (int)sizes[1];
    else if (sizes[1] == 1.0) *h = PETSC_DRAW_FULL_SIZE;
    else if (sizes[1] == .5) *h = PETSC_DRAW_HALF_SIZE;
    else if (PetscEqualReal(sizes[1], .3) == PETSC_TRUE) *h = PETSC_DRAW_THIRD_SIZE; // sizes == 0.3 will never be true
    else if (sizes[1] == .25) *h = PETSC_DRAW_QUARTER_SIZE;
  }
  if (*w == PETSC_DECIDE || *w == PETSC_DEFAULT) *w = draw->w = 300;
  if (*h == PETSC_DECIDE || *h == PETSC_DEFAULT) *h = draw->h = 300;
  switch (*w) {
  case PETSC_DRAW_FULL_SIZE:
    *w = draw->w = (xmax - 10);
    break;
  case PETSC_DRAW_HALF_SIZE:
    *w = draw->w = (xmax - 20) / 2;
    break;
  case PETSC_DRAW_THIRD_SIZE:
    *w = draw->w = (xmax - 30) / 3;
    break;
  case PETSC_DRAW_QUARTER_SIZE:
    *w = draw->w = (xmax - 40) / 4;
    break;
  }
  switch (*h) {
  case PETSC_DRAW_FULL_SIZE:
    *h = draw->h = (ymax - 10);
    break;
  case PETSC_DRAW_HALF_SIZE:
    *h = draw->h = (ymax - 20) / 2;
    break;
  case PETSC_DRAW_THIRD_SIZE:
    *h = draw->h = (ymax - 30) / 3;
    break;
  case PETSC_DRAW_QUARTER_SIZE:
    *h = draw->h = (ymax - 40) / 4;
    break;
  }
  if (dvirtual == PETSC_TRUE) PetscFunctionReturn(PETSC_SUCCESS);

  // allow user to set location and size of window
  xywh[0] = *x;
  xywh[1] = *y;
  xywh[2] = *w;
  xywh[3] = *h;
  PetscCall(PetscOptionsGetIntArray(((PetscObject)draw)->options, ((PetscObject)draw)->prefix, "-geometry", xywh, &osize, NULL));
  PetscCall(PetscCIntCast(xywh[0], x));
  PetscCall(PetscCIntCast(xywh[1], y));
  PetscCall(PetscCIntCast(xywh[2], w));
  PetscCall(PetscCIntCast(xywh[3], h));
  if (*w == PETSC_DECIDE || *w == PETSC_DEFAULT) *w = 300;
  if (*h == PETSC_DECIDE || *h == PETSC_DEFAULT) *h = 300;
  draw->x = *x;
  draw->y = *y;
  draw->w = *w;
  draw->h = *h;

  if (draw->x == PETSC_DECIDE || draw->y == PETSC_DECIDE) {
    /*
     PETSc tries to place windows starting in the upper left corner
      and moving across to the right.

     +0,0-------------------------------------------+
     |  Region used so far  +xavailable,yavailable  |
     |                      |                       |
     |                      |                       |
     +--------------------- +ybottom                |
     |                                              |
     |                                              |
     +----------------------------------------------+xmax,ymax

    */
    // First: can we add it to the right?
    if (xavailable + *w + 10 <= xmax) {
      *x      = xavailable;
      *y      = yavailable;
      ybottom = PetscMax(ybottom, *y + *h + 30);
    } else {
      // No, so add it below on the left
      xavailable = *x = 0;
      yavailable = *y = ybottom;
      ybottom         = ybottom + *h + 30;
    }
  }
  // update available region
  xavailable = PetscMax(xavailable, *x + *w + 10);
  if (xavailable >= xmax) {
    xavailable = 0;
    yavailable = yavailable + *h + 30;
    ybottom    = yavailable;
  }
  if (yavailable >= ymax) {
    *y         = 0;
    yavailable = 0;
    ybottom    = 0;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
