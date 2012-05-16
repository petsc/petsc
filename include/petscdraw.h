/*
  Interface to the PETSc graphics (currently only support for X-windows
*/
#if !defined(__PETSCDRAW_H)
#define __PETSCDRAW_H
#include "petscsys.h"

PETSC_EXTERN PetscClassId PETSC_DRAW_CLASSID;

/*J
    PetscDrawType - String with the name of a PetscDraw 

   Level: beginner

.seealso: PetscDrawSetType(), PetscDraw, PetscViewer
J*/
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

PETSC_EXTERN PetscFList PetscDrawList;
PETSC_EXTERN PetscErrorCode PetscDrawRegisterAll(const char[]);
PETSC_EXTERN PetscErrorCode PetscDrawInitializePackage(const char[]);
PETSC_EXTERN PetscErrorCode PetscDrawRegisterDestroy(void);

PETSC_EXTERN PetscErrorCode PetscDrawRegister(const char*,const char*,const char*,PetscErrorCode(*)(PetscDraw));

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

PETSC_EXTERN PetscErrorCode PetscDrawGetType(PetscDraw,const PetscDrawType*);
PETSC_EXTERN PetscErrorCode PetscDrawSetType(PetscDraw,const PetscDrawType);
PETSC_EXTERN PetscErrorCode PetscDrawCreate(MPI_Comm,const char[],const char[],int,int,int,int,PetscDraw*);
PETSC_EXTERN PetscErrorCode PetscDrawSetFromOptions(PetscDraw);
PETSC_EXTERN PetscErrorCode PetscDrawSetSave(PetscDraw,const char*,PetscBool);

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

PETSC_EXTERN PetscErrorCode PetscDrawOpenX(MPI_Comm,const char[],const char[],int,int,int,int,PetscDraw*);

#define PETSC_DRAW_FULL_SIZE    -3
#define PETSC_DRAW_HALF_SIZE    -4
#define PETSC_DRAW_THIRD_SIZE   -5
#define PETSC_DRAW_QUARTER_SIZE -6

PETSC_EXTERN PetscErrorCode PetscDrawOpenNull(MPI_Comm,PetscDraw *);
PETSC_EXTERN PetscErrorCode PetscDrawDestroy(PetscDraw*);
PETSC_EXTERN PetscErrorCode PetscDrawIsNull(PetscDraw,PetscBool *);

PETSC_EXTERN PetscErrorCode PetscDrawGetPopup(PetscDraw,PetscDraw*);
PETSC_EXTERN PetscErrorCode PetscDrawCheckResizedWindow(PetscDraw);
PETSC_EXTERN PetscErrorCode PetscDrawResizeWindow(PetscDraw,int,int);

PETSC_EXTERN PetscErrorCode PetscDrawScalePopup(PetscDraw,PetscReal,PetscReal);

PETSC_EXTERN PetscErrorCode PetscDrawLine(PetscDraw,PetscReal,PetscReal,PetscReal,PetscReal,int);
PETSC_EXTERN PetscErrorCode PetscDrawArrow(PetscDraw,PetscReal,PetscReal,PetscReal,PetscReal,int);
PETSC_EXTERN PetscErrorCode PetscDrawLineSetWidth(PetscDraw,PetscReal);
PETSC_EXTERN PetscErrorCode PetscDrawLineGetWidth(PetscDraw,PetscReal*);

PETSC_EXTERN PetscErrorCode PetscDrawPoint(PetscDraw,PetscReal,PetscReal,int);
PETSC_EXTERN PetscErrorCode PetscDrawPointSetSize(PetscDraw,PetscReal);

PETSC_EXTERN PetscErrorCode PetscDrawRectangle(PetscDraw,PetscReal,PetscReal,PetscReal,PetscReal,int,int,int,int);
PETSC_EXTERN PetscErrorCode PetscDrawTriangle(PetscDraw,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal,int,int,int);
PETSC_EXTERN PetscErrorCode PetscDrawEllipse(PetscDraw,PetscReal,PetscReal,PetscReal,PetscReal,int);
PETSC_EXTERN PetscErrorCode PetscDrawTensorContourPatch(PetscDraw,int,int,PetscReal*,PetscReal*,PetscReal,PetscReal,PetscReal*);
PETSC_EXTERN PetscErrorCode PetscDrawTensorContour(PetscDraw,int,int,const PetscReal[],const PetscReal[],PetscReal *);

PETSC_EXTERN PetscErrorCode PetscDrawString(PetscDraw,PetscReal,PetscReal,int,const char[]);
PETSC_EXTERN PetscErrorCode PetscDrawStringVertical(PetscDraw,PetscReal,PetscReal,int,const char[]);
PETSC_EXTERN PetscErrorCode PetscDrawStringSetSize(PetscDraw,PetscReal,PetscReal);
PETSC_EXTERN PetscErrorCode PetscDrawStringGetSize(PetscDraw,PetscReal*,PetscReal*);

PETSC_EXTERN PetscErrorCode PetscDrawSetViewPort(PetscDraw,PetscReal,PetscReal,PetscReal,PetscReal);
PETSC_EXTERN PetscErrorCode PetscDrawSplitViewPort(PetscDraw);

PETSC_EXTERN PetscErrorCode PetscDrawSetCoordinates(PetscDraw,PetscReal,PetscReal,PetscReal,PetscReal);
PETSC_EXTERN PetscErrorCode PetscDrawGetCoordinates(PetscDraw,PetscReal*,PetscReal*,PetscReal*,PetscReal*);

PETSC_EXTERN PetscErrorCode PetscDrawSetTitle(PetscDraw,const char[]);
PETSC_EXTERN PetscErrorCode PetscDrawAppendTitle(PetscDraw,const char[]);
PETSC_EXTERN PetscErrorCode PetscDrawGetTitle(PetscDraw,char **);

PETSC_EXTERN PetscErrorCode PetscDrawSetPause(PetscDraw,PetscReal);
PETSC_EXTERN PetscErrorCode PetscDrawGetPause(PetscDraw,PetscReal*);
PETSC_EXTERN PetscErrorCode PetscDrawPause(PetscDraw);
PETSC_EXTERN PetscErrorCode PetscDrawSetDoubleBuffer(PetscDraw);
PETSC_EXTERN PetscErrorCode PetscDrawFlush(PetscDraw);
PETSC_EXTERN PetscErrorCode PetscDrawSynchronizedFlush(PetscDraw);
PETSC_EXTERN PetscErrorCode PetscDrawClear(PetscDraw);
PETSC_EXTERN PetscErrorCode PetscDrawSave(PetscDraw);
PETSC_EXTERN PetscErrorCode PetscDrawSynchronizedClear(PetscDraw);
PETSC_EXTERN PetscErrorCode PetscDrawBOP(PetscDraw);
PETSC_EXTERN PetscErrorCode PetscDrawEOP(PetscDraw);

PETSC_EXTERN PetscErrorCode PetscDrawSetDisplay(PetscDraw,char*);
#define PetscDrawSetFilename(a,b) PetscDrawSetDisplay(a,b)

PETSC_EXTERN PetscErrorCode PetscDrawGetSingleton(PetscDraw,PetscDraw*);
PETSC_EXTERN PetscErrorCode PetscDrawRestoreSingleton(PetscDraw,PetscDraw*);

/*E
    PetscDrawButton - Used to determine which button was pressed

   Level: intermediate

.seealso: PetscDrawGetMouseButton(), PetscDrawSynchronizedGetMouseButton()
E*/
typedef enum {PETSC_BUTTON_NONE,PETSC_BUTTON_LEFT,PETSC_BUTTON_CENTER,PETSC_BUTTON_RIGHT,PETSC_BUTTON_LEFT_SHIFT,PETSC_BUTTON_CENTER_SHIFT,PETSC_BUTTON_RIGHT_SHIFT} PetscDrawButton;

PETSC_EXTERN PetscErrorCode PetscDrawGetMouseButton(PetscDraw,PetscDrawButton *,PetscReal*,PetscReal *,PetscReal *,PetscReal *);
PETSC_EXTERN PetscErrorCode PetscDrawSynchronizedGetMouseButton(PetscDraw,PetscDrawButton *,PetscReal*,PetscReal *,PetscReal *,PetscReal *);

PETSC_EXTERN PetscErrorCode PetscDrawZoom(PetscDraw,PetscErrorCode (*)(PetscDraw,void *),void *);

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
PETSC_EXTERN PetscErrorCode PetscDrawViewPortsCreate(PetscDraw,PetscInt,PetscDrawViewPorts**);
PETSC_EXTERN PetscErrorCode PetscDrawViewPortsCreateRect(PetscDraw,PetscInt,PetscInt,PetscDrawViewPorts**);
PETSC_EXTERN PetscErrorCode PetscDrawViewPortsDestroy(PetscDrawViewPorts*);
PETSC_EXTERN PetscErrorCode PetscDrawViewPortsSet(PetscDrawViewPorts*,PetscInt);

/*S
     PetscDrawAxis - Manages X-Y axis

   Level: advanced

  Concepts: graphics, axis

.seealso:  PetscDrawAxisCreate(), PetscDrawAxisSetLimits(), PetscDrawAxisSetColors(), PetscDrawAxisSetLabels()
S*/
typedef struct _p_PetscDrawAxis* PetscDrawAxis;

PETSC_EXTERN PetscClassId PETSC_DRAWAXIS_CLASSID;

PETSC_EXTERN PetscErrorCode PetscDrawAxisCreate(PetscDraw,PetscDrawAxis *);
PETSC_EXTERN PetscErrorCode PetscDrawAxisDestroy(PetscDrawAxis*);
PETSC_EXTERN PetscErrorCode PetscDrawAxisDraw(PetscDrawAxis);
PETSC_EXTERN PetscErrorCode PetscDrawAxisSetLimits(PetscDrawAxis,PetscReal,PetscReal,PetscReal,PetscReal);
PETSC_EXTERN PetscErrorCode PetscDrawAxisSetHoldLimits(PetscDrawAxis,PetscBool );
PETSC_EXTERN PetscErrorCode PetscDrawAxisSetColors(PetscDrawAxis,int,int,int);
PETSC_EXTERN PetscErrorCode PetscDrawAxisSetLabels(PetscDrawAxis,const char[],const char[],const char[]);

/*S
     PetscDrawLG - Manages drawing x-y plots

   Level: advanced

  Concepts: graphics, axis

.seealso:  PetscDrawAxisCreate(), PetscDrawLGCreate(), PetscDrawLGAddPoint()
S*/
typedef struct _p_PetscDrawLG*   PetscDrawLG;

PETSC_EXTERN PetscClassId PETSC_DRAWLG_CLASSID;

PETSC_EXTERN PetscErrorCode PetscDrawLGCreate(PetscDraw,int,PetscDrawLG *);
PETSC_EXTERN PetscErrorCode PetscDrawLGDestroy(PetscDrawLG*);
PETSC_EXTERN PetscErrorCode PetscDrawLGAddPoint(PetscDrawLG,PetscReal*,PetscReal*);
PETSC_EXTERN PetscErrorCode PetscDrawLGAddPoints(PetscDrawLG,int,PetscReal**,PetscReal**);
PETSC_EXTERN PetscErrorCode PetscDrawLGDraw(PetscDrawLG);
PETSC_EXTERN PetscErrorCode PetscDrawLGPrint(PetscDrawLG);
PETSC_EXTERN PetscErrorCode PetscDrawLGReset(PetscDrawLG);
PETSC_EXTERN PetscErrorCode PetscDrawLGSetDimension(PetscDrawLG,PetscInt);
PETSC_EXTERN PetscErrorCode PetscDrawLGSetLegend(PetscDrawLG,const char *const*);
PETSC_EXTERN PetscErrorCode PetscDrawLGGetAxis(PetscDrawLG,PetscDrawAxis *);
PETSC_EXTERN PetscErrorCode PetscDrawLGGetDraw(PetscDrawLG,PetscDraw *);
PETSC_EXTERN PetscErrorCode PetscDrawLGIndicateDataPoints(PetscDrawLG);
PETSC_EXTERN PetscErrorCode PetscDrawLGSetLimits(PetscDrawLG,PetscReal,PetscReal,PetscReal,PetscReal);
PETSC_EXTERN PetscErrorCode PetscDrawLGSetColors(PetscDrawLG,const int*);

/*S
     PetscDrawSP - Manages drawing scatter plots

   Level: advanced

  Concepts: graphics, scatter plots

.seealso:  PetscDrawSPCreate()
S*/
typedef struct _p_PetscDrawSP*   PetscDrawSP;

PETSC_EXTERN PetscClassId PETSC_DRAWSP_CLASSID;

PETSC_EXTERN PetscErrorCode PetscDrawSPCreate(PetscDraw,int,PetscDrawSP *);
PETSC_EXTERN PetscErrorCode PetscDrawSPDestroy(PetscDrawSP*);
PETSC_EXTERN PetscErrorCode PetscDrawSPAddPoint(PetscDrawSP,PetscReal*,PetscReal*);
PETSC_EXTERN PetscErrorCode PetscDrawSPAddPoints(PetscDrawSP,int,PetscReal**,PetscReal**);
PETSC_EXTERN PetscErrorCode PetscDrawSPDraw(PetscDrawSP);
PETSC_EXTERN PetscErrorCode PetscDrawSPReset(PetscDrawSP);
PETSC_EXTERN PetscErrorCode PetscDrawSPSetDimension(PetscDrawSP,int);
PETSC_EXTERN PetscErrorCode PetscDrawSPGetAxis(PetscDrawSP,PetscDrawAxis *);
PETSC_EXTERN PetscErrorCode PetscDrawSPGetDraw(PetscDrawSP,PetscDraw *);
PETSC_EXTERN PetscErrorCode PetscDrawSPSetLimits(PetscDrawSP,PetscReal,PetscReal,PetscReal,PetscReal);
PETSC_EXTERN PetscErrorCode PetscDrawLGSPDraw(PetscDrawLG,PetscDrawSP);

/*S
     PetscDrawHG - Manages drawing histograms

   Level: advanced

  Concepts: graphics, histograms

.seealso:  PetscDrawHGCreate()
S*/
typedef struct _p_PetscDrawHG*   PetscDrawHG;

PETSC_EXTERN PetscClassId PETSC_DRAWHG_CLASSID;

PETSC_EXTERN PetscErrorCode PetscDrawHGCreate(PetscDraw,int,PetscDrawHG *);
PETSC_EXTERN PetscErrorCode PetscDrawHGDestroy(PetscDrawHG*);
PETSC_EXTERN PetscErrorCode PetscDrawHGAddValue(PetscDrawHG,PetscReal);
PETSC_EXTERN PetscErrorCode PetscDrawHGDraw(PetscDrawHG);
PETSC_EXTERN PetscErrorCode PetscDrawHGPrint(PetscDrawHG);
PETSC_EXTERN PetscErrorCode PetscDrawHGReset(PetscDrawHG);
PETSC_EXTERN PetscErrorCode PetscDrawHGGetAxis(PetscDrawHG,PetscDrawAxis *);
PETSC_EXTERN PetscErrorCode PetscDrawHGGetDraw(PetscDrawHG,PetscDraw *);
PETSC_EXTERN PetscErrorCode PetscDrawHGSetLimits(PetscDrawHG,PetscReal,PetscReal,int,int);
PETSC_EXTERN PetscErrorCode PetscDrawHGSetNumberBins(PetscDrawHG,int);
PETSC_EXTERN PetscErrorCode PetscDrawHGSetColor(PetscDrawHG,int);
PETSC_EXTERN PetscErrorCode PetscDrawHGCalcStats(PetscDrawHG, PetscBool );
PETSC_EXTERN PetscErrorCode PetscDrawHGIntegerBins(PetscDrawHG, PetscBool );

/*
    PetscViewer routines that allow you to access underlying PetscDraw objects
*/
PETSC_EXTERN PetscErrorCode PetscViewerDrawGetDraw(PetscViewer,PetscInt,PetscDraw*);
PETSC_EXTERN PetscErrorCode PetscViewerDrawBaseAdd(PetscViewer,PetscInt);
PETSC_EXTERN PetscErrorCode PetscViewerDrawBaseSet(PetscViewer,PetscInt);
PETSC_EXTERN PetscErrorCode PetscViewerDrawGetDrawLG(PetscViewer,PetscInt,PetscDrawLG*);
PETSC_EXTERN PetscErrorCode PetscViewerDrawGetDrawAxis(PetscViewer,PetscInt,PetscDrawAxis*);

PETSC_EXTERN PetscErrorCode PetscDrawUtilitySetCmapHue(unsigned char *,unsigned char *,unsigned char *,int);
PETSC_EXTERN PetscErrorCode PetscDrawUtilitySetGamma(PetscReal);

#endif
