/*
  Interface to the PETSc graphics
*/
#pragma once
#include <petscdrawtypes.h>
#include <petscviewertypes.h>
#include <petscmath.h>

/* MANSEC = Sys */
/* SUBMANSEC = Draw */

PETSC_EXTERN PetscClassId PETSC_DRAW_CLASSID;

PETSC_EXTERN PetscFunctionList PetscDrawList;
PETSC_EXTERN PetscErrorCode    PetscDrawInitializePackage(void);
PETSC_EXTERN PetscErrorCode    PetscDrawFinalizePackage(void);
PETSC_EXTERN PetscErrorCode    PetscDrawRegister(const char[], PetscErrorCode (*)(PetscDraw));

PETSC_EXTERN PetscErrorCode PetscDrawGetType(PetscDraw, PetscDrawType *);
PETSC_EXTERN PetscErrorCode PetscDrawSetType(PetscDraw, PetscDrawType);
PETSC_EXTERN PetscErrorCode PetscDrawCreate(MPI_Comm, const char[], const char[], int, int, int, int, PetscDraw *);
PETSC_EXTERN PetscErrorCode PetscDrawSetOptionsPrefix(PetscDraw, const char[]);
PETSC_EXTERN PetscErrorCode PetscDrawSetFromOptions(PetscDraw);
PETSC_EXTERN PetscErrorCode PetscDrawSetSave(PetscDraw, const char[]);
PETSC_EXTERN PetscErrorCode PetscDrawSetSaveMovie(PetscDraw, const char[]);
PETSC_EXTERN PetscErrorCode PetscDrawSetSaveFinalImage(PetscDraw, const char[]);
PETSC_EXTERN PetscErrorCode PetscDrawView(PetscDraw, PetscViewer);
PETSC_EXTERN PetscErrorCode PetscDrawViewFromOptions(PetscDraw, PetscObject, const char[]);

/*MC
  PetscDrawRealToColor - Maps a real value within an interval to a color.
  The color is an integer value in the range [`PETSC_DRAW_BASIC_COLORS` to 255]
  that can be passed to various drawing routines.

  Synopsis:
  #include <petscdraw.h>
  int PetscDrawRealToColor(PetscReal value, PetscReal min, PetscReal max)

  Not Collective

  Input Parameters:
+ value - value to map within the interval [`min`, `max`]
. min   - lower end of interval
- max   - upper end of interval

  Returns:
  The result as integer

  Level: intermediate

  Note:
  Values outside the interval [`min`, `max`] are clipped.

.seealso: `PetscDraw`, `PetscDrawPointPixel()`, `PetscDrawPoint()`, `PetscDrawLine()`, `PetscDrawTriangle()`, `PetscDrawRectangle()`
M*/
static inline int PetscDrawRealToColor(PetscReal value, PetscReal min, PetscReal max)
{
  value = PetscClipInterval(value, min, max);
  return PETSC_DRAW_BASIC_COLORS + (int)((255 - PETSC_DRAW_BASIC_COLORS) * (value - min) / (max - min));
}

PETSC_EXTERN PetscErrorCode PetscDrawOpenX(MPI_Comm, const char[], const char[], int, int, int, int, PetscDraw *);

PETSC_EXTERN PetscErrorCode PetscDrawOpenImage(MPI_Comm, const char[], int, int, PetscDraw *);
PETSC_EXTERN PetscErrorCode PetscDrawOpenNull(MPI_Comm, PetscDraw *);
PETSC_EXTERN PetscErrorCode PetscDrawDestroy(PetscDraw *);
PETSC_EXTERN PetscErrorCode PetscDrawIsNull(PetscDraw, PetscBool *);

PETSC_EXTERN PetscErrorCode PetscDrawGetPopup(PetscDraw, PetscDraw *);
PETSC_EXTERN PetscErrorCode PetscDrawScalePopup(PetscDraw, PetscReal, PetscReal);

PETSC_EXTERN PetscErrorCode PetscDrawCheckResizedWindow(PetscDraw);
PETSC_EXTERN PetscErrorCode PetscDrawResizeWindow(PetscDraw, int, int);
PETSC_EXTERN PetscErrorCode PetscDrawGetWindowSize(PetscDraw, int *, int *);
PETSC_EXTERN PetscErrorCode PetscDrawPixelToCoordinate(PetscDraw, int, int, PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode PetscDrawCoordinateToPixel(PetscDraw, PetscReal, PetscReal, int *, int *);

PETSC_EXTERN PetscErrorCode PetscDrawIndicatorFunction(PetscDraw, PetscReal, PetscReal, PetscReal, PetscReal, int, PetscErrorCode (*)(void *, PetscReal, PetscReal, PetscBool *), void *);

PETSC_EXTERN PetscErrorCode PetscDrawLine(PetscDraw, PetscReal, PetscReal, PetscReal, PetscReal, int);
PETSC_EXTERN PetscErrorCode PetscDrawArrow(PetscDraw, PetscReal, PetscReal, PetscReal, PetscReal, int);
PETSC_EXTERN PetscErrorCode PetscDrawLineSetWidth(PetscDraw, PetscReal);
PETSC_EXTERN PetscErrorCode PetscDrawLineGetWidth(PetscDraw, PetscReal *);

PETSC_EXTERN PetscErrorCode PetscDrawMarker(PetscDraw, PetscReal, PetscReal, int);
PETSC_EXTERN PetscErrorCode PetscDrawSetMarkerType(PetscDraw, PetscDrawMarkerType);
PETSC_EXTERN PetscErrorCode PetscDrawGetMarkerType(PetscDraw, PetscDrawMarkerType *);

PETSC_EXTERN PetscErrorCode PetscDrawPoint(PetscDraw, PetscReal, PetscReal, int);
PETSC_EXTERN PetscErrorCode PetscDrawPointPixel(PetscDraw, int, int, int);
PETSC_EXTERN PetscErrorCode PetscDrawPointSetSize(PetscDraw, PetscReal);

PETSC_EXTERN PetscErrorCode PetscDrawRectangle(PetscDraw, PetscReal, PetscReal, PetscReal, PetscReal, int, int, int, int);
PETSC_EXTERN PetscErrorCode PetscDrawTriangle(PetscDraw, PetscReal, PetscReal, PetscReal, PetscReal, PetscReal, PetscReal, int, int, int);
PETSC_EXTERN PetscErrorCode PetscDrawEllipse(PetscDraw, PetscReal, PetscReal, PetscReal, PetscReal, int);
PETSC_EXTERN PetscErrorCode PetscDrawTensorContourPatch(PetscDraw, int, int, PetscReal *, PetscReal *, PetscReal, PetscReal, PetscReal *);
PETSC_EXTERN PetscErrorCode PetscDrawTensorContour(PetscDraw, int, int, const PetscReal[], const PetscReal[], PetscReal *);

PETSC_EXTERN PetscErrorCode PetscDrawString(PetscDraw, PetscReal, PetscReal, int, const char[]);
PETSC_EXTERN PetscErrorCode PetscDrawStringCentered(PetscDraw, PetscReal, PetscReal, int, const char[]);
PETSC_EXTERN PetscErrorCode PetscDrawStringBoxed(PetscDraw, PetscReal, PetscReal, int, int, const char[], PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode PetscDrawStringVertical(PetscDraw, PetscReal, PetscReal, int, const char[]);
PETSC_EXTERN PetscErrorCode PetscDrawStringSetSize(PetscDraw, PetscReal, PetscReal);
PETSC_EXTERN PetscErrorCode PetscDrawStringGetSize(PetscDraw, PetscReal *, PetscReal *);

PETSC_EXTERN PetscErrorCode PetscDrawSetViewPort(PetscDraw, PetscReal, PetscReal, PetscReal, PetscReal);
PETSC_EXTERN PetscErrorCode PetscDrawGetViewPort(PetscDraw, PetscReal *, PetscReal *, PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode PetscDrawSplitViewPort(PetscDraw);

PETSC_EXTERN PetscErrorCode PetscDrawSetCoordinates(PetscDraw, PetscReal, PetscReal, PetscReal, PetscReal);
PETSC_EXTERN PetscErrorCode PetscDrawGetCoordinates(PetscDraw, PetscReal *, PetscReal *, PetscReal *, PetscReal *);

PETSC_EXTERN PetscErrorCode PetscDrawSetTitle(PetscDraw, const char[]);
PETSC_EXTERN PetscErrorCode PetscDrawAppendTitle(PetscDraw, const char[]);
PETSC_EXTERN PetscErrorCode PetscDrawGetTitle(PetscDraw, const char *[]);

PETSC_EXTERN PetscErrorCode PetscDrawSetPause(PetscDraw, PetscReal);
PETSC_EXTERN PetscErrorCode PetscDrawGetPause(PetscDraw, PetscReal *);
PETSC_EXTERN PetscErrorCode PetscDrawPause(PetscDraw);
PETSC_EXTERN PetscErrorCode PetscDrawSetDoubleBuffer(PetscDraw);
PETSC_EXTERN PetscErrorCode PetscDrawClear(PetscDraw);
PETSC_EXTERN PetscErrorCode PetscDrawFlush(PetscDraw);
PETSC_EXTERN PetscErrorCode PetscDrawSave(PetscDraw);
PETSC_EXTERN PetscErrorCode PetscDrawSaveMovie(PetscDraw);
PETSC_EXTERN PetscErrorCode PetscDrawBOP(PetscDraw);
PETSC_EXTERN PetscErrorCode PetscDrawEOP(PetscDraw);

PETSC_EXTERN PetscErrorCode PetscDrawSetDisplay(PetscDraw, const char[]);
PETSC_EXTERN PetscErrorCode PetscDrawGetSingleton(PetscDraw, PetscDraw *);
PETSC_EXTERN PetscErrorCode PetscDrawRestoreSingleton(PetscDraw, PetscDraw *);

PETSC_EXTERN PetscErrorCode PetscDrawGetCurrentPoint(PetscDraw, PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode PetscDrawSetCurrentPoint(PetscDraw, PetscReal, PetscReal);
PETSC_EXTERN PetscErrorCode PetscDrawPushCurrentPoint(PetscDraw, PetscReal, PetscReal);
PETSC_EXTERN PetscErrorCode PetscDrawPopCurrentPoint(PetscDraw);
PETSC_EXTERN PetscErrorCode PetscDrawGetBoundingBox(PetscDraw, PetscReal *, PetscReal *, PetscReal *, PetscReal *);

PETSC_EXTERN PetscErrorCode PetscDrawSetVisible(PetscDraw, PetscBool);

PETSC_EXTERN PetscErrorCode PetscDrawGetMouseButton(PetscDraw, PetscDrawButton *, PetscReal *, PetscReal *, PetscReal *, PetscReal *);

PETSC_EXTERN PetscErrorCode PetscDrawZoom(PetscDraw, PetscErrorCode (*)(PetscDraw, void *), void *);

PETSC_EXTERN PetscErrorCode PetscDrawViewPortsCreate(PetscDraw, PetscInt, PetscDrawViewPorts **);
PETSC_EXTERN PetscErrorCode PetscDrawViewPortsCreateRect(PetscDraw, PetscInt, PetscInt, PetscDrawViewPorts **);
PETSC_EXTERN PetscErrorCode PetscDrawViewPortsDestroy(PetscDrawViewPorts *);
PETSC_EXTERN PetscErrorCode PetscDrawViewPortsSet(PetscDrawViewPorts *, PetscInt);

PETSC_EXTERN PetscClassId PETSC_DRAWAXIS_CLASSID;

PETSC_EXTERN PetscErrorCode PetscDrawAxisCreate(PetscDraw, PetscDrawAxis *);
PETSC_EXTERN PetscErrorCode PetscDrawAxisDestroy(PetscDrawAxis *);
PETSC_EXTERN PetscErrorCode PetscDrawAxisDraw(PetscDrawAxis);
PETSC_EXTERN PetscErrorCode PetscDrawAxisSetLimits(PetscDrawAxis, PetscReal, PetscReal, PetscReal, PetscReal);
PETSC_EXTERN PetscErrorCode PetscDrawAxisGetLimits(PetscDrawAxis, PetscReal *, PetscReal *, PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode PetscDrawAxisSetHoldLimits(PetscDrawAxis, PetscBool);
PETSC_EXTERN PetscErrorCode PetscDrawAxisSetColors(PetscDrawAxis, int, int, int);
PETSC_EXTERN PetscErrorCode PetscDrawAxisSetLabels(PetscDrawAxis, const char[], const char[], const char[]);

PETSC_EXTERN PetscClassId PETSC_DRAWLG_CLASSID;

PETSC_EXTERN PetscErrorCode PetscDrawLGCreate(PetscDraw, PetscInt, PetscDrawLG *);
PETSC_EXTERN PetscErrorCode PetscDrawLGDestroy(PetscDrawLG *);
PETSC_EXTERN PetscErrorCode PetscDrawLGAddPoint(PetscDrawLG, const PetscReal *, const PetscReal *);
PETSC_EXTERN PetscErrorCode PetscDrawLGAddCommonPoint(PetscDrawLG, const PetscReal, const PetscReal *);
PETSC_EXTERN PetscErrorCode PetscDrawLGAddPoints(PetscDrawLG, PetscInt, PetscReal **, PetscReal **);
PETSC_EXTERN PetscErrorCode PetscDrawLGDraw(PetscDrawLG);
PETSC_EXTERN PetscErrorCode PetscDrawLGSave(PetscDrawLG);
PETSC_EXTERN PetscErrorCode PetscDrawLGView(PetscDrawLG, PetscViewer);
PETSC_EXTERN PetscErrorCode PetscDrawLGReset(PetscDrawLG);
PETSC_EXTERN PetscErrorCode PetscDrawLGSetDimension(PetscDrawLG, PetscInt);
PETSC_EXTERN PetscErrorCode PetscDrawLGGetDimension(PetscDrawLG, PetscInt *);
PETSC_EXTERN PetscErrorCode PetscDrawLGSetLegend(PetscDrawLG, const char *const *);
PETSC_EXTERN PetscErrorCode PetscDrawLGGetAxis(PetscDrawLG, PetscDrawAxis *);
PETSC_EXTERN PetscErrorCode PetscDrawLGGetDraw(PetscDrawLG, PetscDraw *);
PETSC_EXTERN PetscErrorCode PetscDrawLGSetUseMarkers(PetscDrawLG, PetscBool);
PETSC_EXTERN PetscErrorCode PetscDrawLGSetLimits(PetscDrawLG, PetscReal, PetscReal, PetscReal, PetscReal);
PETSC_EXTERN PetscErrorCode PetscDrawLGSetColors(PetscDrawLG, const int[]);
PETSC_EXTERN PetscErrorCode PetscDrawLGSetOptionsPrefix(PetscDrawLG, const char[]);
PETSC_EXTERN PetscErrorCode PetscDrawLGSetFromOptions(PetscDrawLG);
PETSC_EXTERN PetscErrorCode PetscDrawLGGetData(PetscDrawLG, PetscInt *, PetscInt *, const PetscReal *[], const PetscReal *[]);

PETSC_EXTERN PetscClassId PETSC_DRAWSP_CLASSID;

PETSC_EXTERN PetscErrorCode PetscDrawSPCreate(PetscDraw, int, PetscDrawSP *);
PETSC_EXTERN PetscErrorCode PetscDrawSPDestroy(PetscDrawSP *);
PETSC_EXTERN PetscErrorCode PetscDrawSPAddPoint(PetscDrawSP, PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode PetscDrawSPAddPoints(PetscDrawSP, int, PetscReal **, PetscReal **);
PETSC_EXTERN PetscErrorCode PetscDrawSPDraw(PetscDrawSP, PetscBool);
PETSC_EXTERN PetscErrorCode PetscDrawSPSave(PetscDrawSP);
PETSC_EXTERN PetscErrorCode PetscDrawSPReset(PetscDrawSP);
PETSC_EXTERN PetscErrorCode PetscDrawSPGetDimension(PetscDrawSP, int *);
PETSC_EXTERN PetscErrorCode PetscDrawSPSetDimension(PetscDrawSP, int);
PETSC_EXTERN PetscErrorCode PetscDrawSPGetAxis(PetscDrawSP, PetscDrawAxis *);
PETSC_EXTERN PetscErrorCode PetscDrawSPGetDraw(PetscDrawSP, PetscDraw *);
PETSC_EXTERN PetscErrorCode PetscDrawSPSetLimits(PetscDrawSP, PetscReal, PetscReal, PetscReal, PetscReal);
PETSC_EXTERN PetscErrorCode PetscDrawLGSPDraw(PetscDrawLG, PetscDrawSP);
PETSC_EXTERN PetscErrorCode PetscDrawSPAddPointColorized(PetscDrawSP, PetscReal *, PetscReal *, PetscReal *);

PETSC_EXTERN PetscClassId PETSC_DRAWHG_CLASSID;

PETSC_EXTERN PetscErrorCode PetscDrawHGCreate(PetscDraw, int, PetscDrawHG *);
PETSC_EXTERN PetscErrorCode PetscDrawHGDestroy(PetscDrawHG *);
PETSC_EXTERN PetscErrorCode PetscDrawHGAddValue(PetscDrawHG, PetscReal);
PETSC_EXTERN PetscErrorCode PetscDrawHGAddWeightedValue(PetscDrawHG, PetscReal, PetscReal);
PETSC_EXTERN PetscErrorCode PetscDrawHGDraw(PetscDrawHG);
PETSC_EXTERN PetscErrorCode PetscDrawHGSave(PetscDrawHG);
PETSC_EXTERN PetscErrorCode PetscDrawHGView(PetscDrawHG, PetscViewer);
PETSC_EXTERN PetscErrorCode PetscDrawHGReset(PetscDrawHG);
PETSC_EXTERN PetscErrorCode PetscDrawHGGetAxis(PetscDrawHG, PetscDrawAxis *);
PETSC_EXTERN PetscErrorCode PetscDrawHGGetDraw(PetscDrawHG, PetscDraw *);
PETSC_EXTERN PetscErrorCode PetscDrawHGSetLimits(PetscDrawHG, PetscReal, PetscReal, int, int);
PETSC_EXTERN PetscErrorCode PetscDrawHGSetNumberBins(PetscDrawHG, int);
PETSC_EXTERN PetscErrorCode PetscDrawHGSetColor(PetscDrawHG, int);
PETSC_EXTERN PetscErrorCode PetscDrawHGCalcStats(PetscDrawHG, PetscBool);
PETSC_EXTERN PetscErrorCode PetscDrawHGIntegerBins(PetscDrawHG, PetscBool);

PETSC_EXTERN PetscClassId PETSC_DRAWBAR_CLASSID;

PETSC_EXTERN PetscErrorCode PetscDrawBarCreate(PetscDraw, PetscDrawBar *);
PETSC_EXTERN PetscErrorCode PetscDrawBarSetData(PetscDrawBar, PetscInt, const PetscReal[], const char *const *);
PETSC_EXTERN PetscErrorCode PetscDrawBarDestroy(PetscDrawBar *);
PETSC_EXTERN PetscErrorCode PetscDrawBarDraw(PetscDrawBar);
PETSC_EXTERN PetscErrorCode PetscDrawBarSave(PetscDrawBar);
PETSC_EXTERN PetscErrorCode PetscDrawBarSetColor(PetscDrawBar, int);
PETSC_EXTERN PetscErrorCode PetscDrawBarSetLimits(PetscDrawBar, PetscReal, PetscReal);
PETSC_EXTERN PetscErrorCode PetscDrawBarSort(PetscDrawBar, PetscBool, PetscReal);
PETSC_EXTERN PetscErrorCode PetscDrawBarSetFromOptions(PetscDrawBar);
PETSC_EXTERN PetscErrorCode PetscDrawBarGetAxis(PetscDrawBar, PetscDrawAxis *);
PETSC_EXTERN PetscErrorCode PetscDrawBarGetDraw(PetscDrawBar, PetscDraw *);

PETSC_EXTERN PetscErrorCode PetscDrawUtilitySetCmap(const char[], int, unsigned char[], unsigned char[], unsigned char[]);
PETSC_EXTERN PetscErrorCode PetscDrawUtilitySetGamma(PetscReal);

/*
    Handling of X11 I/O window resizing, window closing and errors in parallel
*/
#if PetscDefined(HAVE_X) && PetscDefined(HAVE_SETJMP_H)
  #include <setjmp.h>

PETSC_EXTERN jmp_buf PetscXIOErrorHandlerJumpBuf;
PETSC_EXTERN void    PetscXIOErrorHandlerJump(void *);
/*S
  PetscXIOErrorHandlerFn - Function type for the X11 I/O error handler installed by `PetscSetXIOErrorHandler()`, called when the X server connection is lost

  Calling Sequence:
. display - the X `Display *` whose connection has failed (passed as `void *` to avoid pulling in X headers)

  Level: developer

  Note:
  By default PETSc installs a handler that gracefully aborts the program rather than letting Xlib call `exit()`.

.seealso: `PetscDraw`, `PetscSetXIOErrorHandler()`
S*/
PETSC_EXTERN_TYPEDEF typedef void                    PetscXIOErrorHandlerFn(void *display);
PETSC_EXTERN_TYPEDEF typedef PetscXIOErrorHandlerFn *PetscXIOErrorHandler PETSC_DEPRECATED_TYPEDEF(3, 26, 0, "PetscXIOErrorHandlerFn*", );
PETSC_EXTERN PetscXIOErrorHandlerFn                 *PetscSetXIOErrorHandler(PetscXIOErrorHandlerFn *);

  /*MC
   PetscDrawCollectiveBegin - Begins a set of draw operations

   Collective

   Synopsis:
    #include <petscdraw.h>
    PetscErrorCode PetscDrawCollectiveBegin(PetscDraw draw)

   Collective

   Input Parameter:
.   draw - the draw object

   Level: advanced

   Notes:
   This is a macro that handles its own error checking, it does not return an error code.

   The set of operations needs to be ended by a call to `PetscDrawCollectiveEnd()`.

   X windows draw operations that are enclosed by these routines handle correctly resizing or closing of
   the window without crashing the program.

   Developer Note:
   This only applies to X windows and so should have a more specific name such as `PetscDrawXCollectiveBegin()`

.seealso: `PetscDraw`, `PetscDrawCollectiveEnd()`
M*/
  #define PetscDrawCollectiveBegin(draw) \
    do { \
      jmp_buf _Petsc_jmpbuf; \
      PetscXIOErrorHandlerFn *volatile _Petsc_xioerrhdl = PETSC_NULLPTR; \
      PetscBool _Petsc_isdrawx, _Petsc_xioerr = PETSC_FALSE; \
      PetscCall(PetscObjectTypeCompare((PetscObject)(draw), PETSC_DRAW_X, &_Petsc_isdrawx)); \
      if (_Petsc_isdrawx) { \
        PetscCall(PetscMemcpy(&_Petsc_jmpbuf, &PetscXIOErrorHandlerJumpBuf, sizeof(_Petsc_jmpbuf))); \
        _Petsc_xioerrhdl = PetscSetXIOErrorHandler(PetscXIOErrorHandlerJump); \
        if (setjmp(PetscXIOErrorHandlerJumpBuf)) { \
          _Petsc_xioerr = PETSC_TRUE; \
          do { \
            PetscDrawCollectiveEnd(draw); \
          } \
        } \
        do { \
      } while (0)

  /*MC
   PetscDrawCollectiveEnd - Ends a set of draw operations begun with `PetscDrawCollectiveBegin()`

   Collective

   Synopsis:
    #include <petscdraw.h>
    PetscErrorCode PetscDrawCollectiveEnd(PetscDraw draw)

   Collective

   Input Parameter:
.   draw - the draw object

   Level: advanced

   Notes:
   This is a macro that handles its own error checking, it does not return an error code.

   X windows draw operations that are enclosed by these routines handle correctly resizing or closing of
   the window without crashing the program.

   Developer Note:
   This only applies to X windows and so should have a more specific name such as `PetscDrawXCollectiveEnd()`

.seealso: `PetscDraw`, `PetscDrawCollectiveBegin()`
M*/
  #define PetscDrawCollectiveEnd(draw) \
    if (_Petsc_isdrawx) { \
      (void)PetscSetXIOErrorHandler(_Petsc_xioerrhdl); \
      PetscCall(PetscMemcpy(&PetscXIOErrorHandlerJumpBuf, &_Petsc_jmpbuf, sizeof(PetscXIOErrorHandlerJumpBuf))); \
      PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &_Petsc_xioerr, 1, MPI_C_BOOL, MPI_LOR, PetscObjectComm((PetscObject)(draw)))); \
      if (_Petsc_xioerr) { \
        PetscCall(PetscDrawSetType((draw), PETSC_DRAW_NULL)); \
        PetscFunctionReturn(PETSC_SUCCESS); \
      } \
    } \
    } \
    while (0)

#else
  #define PetscDrawCollectiveBegin(draw)
  #define PetscDrawCollectiveEnd(draw)
#endif /* PetscDefined(HAVE_X) && PetscDefined(HAVE_SETJMP_H) */

PETSC_EXTERN PetscErrorCode PetscViewerDrawGetDrawType(PetscViewer, PetscDrawType *);
PETSC_EXTERN PetscErrorCode PetscViewerDrawSetTitle(PetscViewer, const char[]);
PETSC_EXTERN PetscErrorCode PetscViewerDrawGetTitle(PetscViewer, const char *[]);
PETSC_EXTERN PetscErrorCode PetscViewerDrawGetDraw(PetscViewer, PetscInt, PetscDraw *);
PETSC_EXTERN PetscErrorCode PetscViewerDrawBaseAdd(PetscViewer, PetscInt);
PETSC_EXTERN PetscErrorCode PetscViewerDrawBaseSet(PetscViewer, PetscInt);
PETSC_EXTERN PetscErrorCode PetscViewerDrawGetDrawLG(PetscViewer, PetscInt, PetscDrawLG *);
PETSC_EXTERN PetscErrorCode PetscViewerDrawGetDrawAxis(PetscViewer, PetscInt, PetscDrawAxis *);
