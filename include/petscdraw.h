/*
  Interface to the PETSc graphics (currently only support for X-windows
*/
#if !defined(__PETSCDRAW_H)
#define __PETSCDRAW_H
#include "petscsys.h"
PETSC_EXTERN_CXX_BEGIN

extern PetscCookie PETSC_DRAW_COOKIE;

/*E
    PetscDrawType - String with the name of a PetscDraw 

   Level: beginner

.seealso: PetscDrawSetType(), PetscDraw, PetscViewer
E*/
#define PetscDrawType  char*
#define PETSC_DRAW_X     "x"
#define PETSC_DRAW_NULL  "null"
#define PETSC_DRAW_WIN32 "win32"
 
/*S
     PetscDraw - Abstract PETSc object for graphics

   Level: beginner

  Concepts: graphics

.seealso:  PetscDrawCreate(), PetscDrawSetType(), PetscDrawType
S*/
typedef struct _p_PetscDraw* PetscDraw;

extern PetscFList PetscDrawList;
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawRegisterAll(const char[]);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawInitializePackage(const char[]);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawRegisterDestroy(void);

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawRegister(const char*,const char*,const char*,PetscErrorCode(*)(PetscDraw));

/*MC
   PetscDrawRegisterDynamic - Adds a method to the Krylov subspace solver package.

   Synopsis:
   PetscErrorCode PetscDrawRegisterDynamic(const char *name_solver,const char *path,const char *name_create,PetscErrorCode (*routine_create)(PetscDraw))

   Not Collective

   Input Parameters:
+  name_solver - name of a new user-defined solver
.  path - path (either absolute or relative) the library containing this solver
.  name_create - name of routine to create method context
-  routine_create - routine to create method context

   Level: developer

   Notes:
   PetscDrawRegisterDynamic() may be called multiple times to add several user-defined solvers.

   If dynamic libraries are used, then the fourth input argument (routine_create)
   is ignored.

   Sample usage:
.vb
   PetscDrawRegisterDynamic("my_draw_type",/home/username/my_lib/lib/libO/solaris/mylib.a,
               "MyDrawCreate",MyDrawCreate);
.ve

   Then, your solver can be chosen with the procedural interface via
$     PetscDrawSetType(ksp,"my_draw_type")
   or at runtime via the option
$     -draw_type my_draw_type

   Concepts: graphics^registering new draw classes
   Concepts: PetscDraw^registering new draw classes

.seealso: PetscDrawRegisterAll(), PetscDrawRegisterDestroy()
M*/
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define PetscDrawRegisterDynamic(a,b,c,d) PetscDrawRegister(a,b,c,0)
#else
#define PetscDrawRegisterDynamic(a,b,c,d) PetscDrawRegister(a,b,c,d)
#endif

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawGetType(PetscDraw,const PetscDrawType*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawSetType(PetscDraw,const PetscDrawType);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawCreate(MPI_Comm,const char[],const char[],int,int,int,int,PetscDraw*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawSetFromOptions(PetscDraw);

/*
   Number of basic colors in the draw routines, the others are used
   for a uniform colormap.
*/
#define PETSC_DRAW_BASIC_COLORS 33

#define PETSC_DRAW_ROTATE          -1         /* will rotate through the colors, start with 2 */
#define PETSC_DRAW_WHITE            0
#define PETSC_DRAW_BLACK            1
#define PETSC_DRAW_RED              2
#define PETSC_DRAW_GREEN            3
#define PETSC_DRAW_CYAN             4
#define PETSC_DRAW_BLUE             5
#define PETSC_DRAW_MAGENTA          6
#define PETSC_DRAW_AQUAMARINE       7
#define PETSC_DRAW_FORESTGREEN      8
#define PETSC_DRAW_ORANGE           9
#define PETSC_DRAW_VIOLET          10
#define PETSC_DRAW_BROWN           11
#define PETSC_DRAW_PINK            12
#define PETSC_DRAW_CORAL           13
#define PETSC_DRAW_GRAY            14
#define PETSC_DRAW_YELLOW          15

#define PETSC_DRAW_GOLD            16
#define PETSC_DRAW_LIGHTPINK       17
#define PETSC_DRAW_MEDIUMTURQUOISE 18
#define PETSC_DRAW_KHAKI           19
#define PETSC_DRAW_DIMGRAY         20
#define PETSC_DRAW_YELLOWGREEN     21
#define PETSC_DRAW_SKYBLUE         22
#define PETSC_DRAW_DARKGREEN       23
#define PETSC_DRAW_NAVYBLUE        24
#define PETSC_DRAW_SANDYBROWN      25
#define PETSC_DRAW_CADETBLUE       26
#define PETSC_DRAW_POWDERBLUE      27
#define PETSC_DRAW_DEEPPINK        28
#define PETSC_DRAW_THISTLE         29
#define PETSC_DRAW_LIMEGREEN       30
#define PETSC_DRAW_LAVENDERBLUSH   31
#define PETSC_DRAW_PLUM            32

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawOpenX(MPI_Comm,const char[],const char[],int,int,int,int,PetscDraw*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawOpenPS(MPI_Comm,char *,PetscDraw *);
#define PETSC_DRAW_FULL_SIZE    -3
#define PETSC_DRAW_HALF_SIZE    -4
#define PETSC_DRAW_THIRD_SIZE   -5
#define PETSC_DRAW_QUARTER_SIZE -6

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawOpenNull(MPI_Comm,PetscDraw *);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawDestroy(PetscDraw);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawIsNull(PetscDraw,PetscTruth*);

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawGetPopup(PetscDraw,PetscDraw*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawCheckResizedWindow(PetscDraw);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawResizeWindow(PetscDraw,int,int);

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawScalePopup(PetscDraw,PetscReal,PetscReal); 

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawLine(PetscDraw,PetscReal,PetscReal,PetscReal,PetscReal,int);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawLineSetWidth(PetscDraw,PetscReal);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawLineGetWidth(PetscDraw,PetscReal*);

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawPoint(PetscDraw,PetscReal,PetscReal,int);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawPointSetSize(PetscDraw,PetscReal);

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawRectangle(PetscDraw,PetscReal,PetscReal,PetscReal,PetscReal,int,int,int,int);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawTriangle(PetscDraw,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal,int,int,int);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawEllipse(PetscDraw,PetscReal,PetscReal,PetscReal,PetscReal,int);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawTensorContourPatch(PetscDraw,int,int,PetscReal*,PetscReal*,PetscReal,PetscReal,PetscReal*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawTensorContour(PetscDraw,int,int,const PetscReal[],const PetscReal[],PetscReal *);

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawString(PetscDraw,PetscReal,PetscReal,int,const char[]);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawStringVertical(PetscDraw,PetscReal,PetscReal,int,const char[]);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawStringSetSize(PetscDraw,PetscReal,PetscReal);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawStringGetSize(PetscDraw,PetscReal*,PetscReal*);

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawSetViewPort(PetscDraw,PetscReal,PetscReal,PetscReal,PetscReal);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawSplitViewPort(PetscDraw);

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawSetCoordinates(PetscDraw,PetscReal,PetscReal,PetscReal,PetscReal);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawGetCoordinates(PetscDraw,PetscReal*,PetscReal*,PetscReal*,PetscReal*);

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawSetTitle(PetscDraw,const char[]);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawAppendTitle(PetscDraw,const char[]);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawGetTitle(PetscDraw,char **);

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawSetPause(PetscDraw,PetscReal);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawGetPause(PetscDraw,PetscReal*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawPause(PetscDraw);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawSetDoubleBuffer(PetscDraw);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawFlush(PetscDraw);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawSynchronizedFlush(PetscDraw);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawClear(PetscDraw);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawSynchronizedClear(PetscDraw);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawBOP(PetscDraw);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawEOP(PetscDraw);

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawSetDisplay(PetscDraw,char*);
#define PetscDrawSetFilename(a,b) PetscDrawSetDisplay(a,b)

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawGetSingleton(PetscDraw,PetscDraw*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawRestoreSingleton(PetscDraw,PetscDraw*);

/*E
    PetscDrawButton - Used to determine which button was pressed

   Level: intermediate

.seealso: PetscDrawGetMouseButton(), PetscDrawSynchronizedGetMouseButton()
E*/
typedef enum {BUTTON_NONE,BUTTON_LEFT,BUTTON_CENTER,BUTTON_RIGHT,BUTTON_LEFT_SHIFT,BUTTON_CENTER_SHIFT,BUTTON_RIGHT_SHIFT} PetscDrawButton;

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawGetMouseButton(PetscDraw,PetscDrawButton *,PetscReal*,PetscReal *,PetscReal *,PetscReal *);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawSynchronizedGetMouseButton(PetscDraw,PetscDrawButton *,PetscReal*,PetscReal *,PetscReal *,PetscReal *);

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawZoom(PetscDraw,PetscErrorCode (*)(PetscDraw,void *),void *);

/*S
     PetscDrawViewPorts - Subwindows in a PetscDraw object

   Level: intermediate

  Concepts: graphics

.seealso:  PetscDrawViewPortsCreate(), PetscDrawViewPortsSet()
S*/
typedef struct {
  PetscInt  nports;
  PetscReal *xl;
  PetscReal *xr;
  PetscReal *yl;
  PetscReal *yr;
  PetscDraw draw;
} PetscDrawViewPorts;
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawViewPortsCreate(PetscDraw,PetscInt,PetscDrawViewPorts**);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawViewPortsCreateRect(PetscDraw,PetscInt,PetscInt,PetscDrawViewPorts**);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawViewPortsDestroy(PetscDrawViewPorts*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawViewPortsSet(PetscDrawViewPorts*,int);

/*S
     PetscDrawAxis - Manages X-Y axis

   Level: advanced

  Concepts: graphics, axis

.seealso:  PetscDrawAxisCreate(), PetscDrawAxisSetLimits(), PetscDrawAxisSetColors(), PetscDrawAxisSetLabels()
S*/
typedef struct _p_DrawAxis* PetscDrawAxis;

extern PetscCookie DRAWAXIS_COOKIE;

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawAxisCreate(PetscDraw,PetscDrawAxis *);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawAxisDestroy(PetscDrawAxis);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawAxisDraw(PetscDrawAxis);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawAxisSetLimits(PetscDrawAxis,PetscReal,PetscReal,PetscReal,PetscReal);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawAxisSetHoldLimits(PetscDrawAxis,PetscTruth);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawAxisSetColors(PetscDrawAxis,int,int,int);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawAxisSetLabels(PetscDrawAxis,const char[],const char[],const char[]);

/*S
     PetscDrawLG - Manages drawing x-y plots

   Level: advanced

  Concepts: graphics, axis

.seealso:  PetscDrawAxisCreate(), PetscDrawLGCreate(), PetscDrawLGAddPoint()
S*/
typedef struct _p_DrawLG*   PetscDrawLG;

extern PetscCookie DRAWLG_COOKIE;

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawLGCreate(PetscDraw,int,PetscDrawLG *);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawLGDestroy(PetscDrawLG);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawLGAddPoint(PetscDrawLG,PetscReal*,PetscReal*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawLGAddPoints(PetscDrawLG,int,PetscReal**,PetscReal**);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawLGDraw(PetscDrawLG);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawLGPrint(PetscDrawLG);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawLGReset(PetscDrawLG);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawLGSetDimension(PetscDrawLG,int);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawLGGetAxis(PetscDrawLG,PetscDrawAxis *);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawLGGetDraw(PetscDrawLG,PetscDraw *);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawLGIndicateDataPoints(PetscDrawLG);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawLGSetLimits(PetscDrawLG,PetscReal,PetscReal,PetscReal,PetscReal); 

/*S
     PetscDrawSP - Manages drawing scatter plots

   Level: advanced

  Concepts: graphics, scatter plots

.seealso:  PetscDrawSPCreate()
S*/
typedef struct _p_DrawSP*   PetscDrawSP;

extern PetscCookie DRAWSP_COOKIE;

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawSPCreate(PetscDraw,int,PetscDrawSP *);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawSPDestroy(PetscDrawSP);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawSPAddPoint(PetscDrawSP,PetscReal*,PetscReal*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawSPAddPoints(PetscDrawSP,int,PetscReal**,PetscReal**);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawSPDraw(PetscDrawSP);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawSPReset(PetscDrawSP);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawSPSetDimension(PetscDrawSP,int);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawSPGetAxis(PetscDrawSP,PetscDrawAxis *);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawSPGetDraw(PetscDrawSP,PetscDraw *);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawSPSetLimits(PetscDrawSP,PetscReal,PetscReal,PetscReal,PetscReal); 
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawLGSPDraw(PetscDrawLG,PetscDrawSP);

/*S
     PetscDrawHG - Manages drawing histograms

   Level: advanced

  Concepts: graphics, histograms

.seealso:  PetscDrawHGCreate()
S*/
typedef struct _p_DrawHG*   PetscDrawHG;

extern PetscCookie DRAWHG_COOKIE;

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawHGCreate(PetscDraw,int,PetscDrawHG *);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawHGDestroy(PetscDrawHG);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawHGAddValue(PetscDrawHG,PetscReal);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawHGDraw(PetscDrawHG);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawHGPrint(PetscDrawHG);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawHGReset(PetscDrawHG);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawHGGetAxis(PetscDrawHG,PetscDrawAxis *);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawHGGetDraw(PetscDrawHG,PetscDraw *);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawHGSetLimits(PetscDrawHG,PetscReal,PetscReal,int,int);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawHGSetNumberBins(PetscDrawHG,int);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawHGSetColor(PetscDrawHG,int);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawHGCalcStats(PetscDrawHG, PetscTruth);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawHGIntegerBins(PetscDrawHG, PetscTruth);

/*
    PetscViewer routines that allow you to access underlying PetscDraw objects
*/
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerDrawGetDraw(PetscViewer,PetscInt,PetscDraw*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerDrawBaseAdd(PetscViewer,PetscInt);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerDrawBaseSet(PetscViewer,PetscInt);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerDrawGetDrawLG(PetscViewer,PetscInt,PetscDrawLG*);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscViewerDrawGetDrawAxis(PetscViewer,PetscInt,PetscDrawAxis*);

EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawUtilitySetCmapHue(unsigned char *,unsigned char *,unsigned char *,int);
EXTERN PetscErrorCode PETSC_DLLEXPORT PetscDrawUtilitySetGamma(PetscReal);

PETSC_EXTERN_CXX_END
#endif
