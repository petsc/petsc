/*
  Interface to the PETSc graphics (currently only support for X-windows
*/
#if !defined(__PETSCDRAW_H)
#define __PETSCDRAW_H
#include "petscsys.h"
PETSC_EXTERN_CXX_BEGIN

extern PetscClassId PETSC_DRAW_CLASSID;

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

extern PetscFList PetscDrawList;
extern PetscErrorCode  PetscDrawRegisterAll(const char[]);
extern PetscErrorCode  PetscDrawInitializePackage(const char[]);
extern PetscErrorCode  PetscDrawRegisterDestroy(void);

extern PetscErrorCode  PetscDrawRegister(const char*,const char*,const char*,PetscErrorCode(*)(PetscDraw));

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

extern PetscErrorCode  PetscDrawGetType(PetscDraw,const PetscDrawType*);
extern PetscErrorCode  PetscDrawSetType(PetscDraw,const PetscDrawType);
extern PetscErrorCode  PetscDrawCreate(MPI_Comm,const char[],const char[],int,int,int,int,PetscDraw*);
extern PetscErrorCode  PetscDrawSetFromOptions(PetscDraw);
extern PetscErrorCode  PetscDrawSetSave(PetscDraw,const char*,PetscBool);

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

extern PetscErrorCode  PetscDrawOpenX(MPI_Comm,const char[],const char[],int,int,int,int,PetscDraw*);

#define PETSC_DRAW_FULL_SIZE    -3
#define PETSC_DRAW_HALF_SIZE    -4
#define PETSC_DRAW_THIRD_SIZE   -5
#define PETSC_DRAW_QUARTER_SIZE -6

extern PetscErrorCode  PetscDrawOpenNull(MPI_Comm,PetscDraw *);
extern PetscErrorCode  PetscDrawDestroy(PetscDraw*);
extern PetscErrorCode  PetscDrawIsNull(PetscDraw,PetscBool *);

extern PetscErrorCode  PetscDrawGetPopup(PetscDraw,PetscDraw*);
extern PetscErrorCode  PetscDrawCheckResizedWindow(PetscDraw);
extern PetscErrorCode  PetscDrawResizeWindow(PetscDraw,int,int);

extern PetscErrorCode  PetscDrawScalePopup(PetscDraw,PetscReal,PetscReal);

extern PetscErrorCode  PetscDrawLine(PetscDraw,PetscReal,PetscReal,PetscReal,PetscReal,int);
extern PetscErrorCode  PetscDrawArrow(PetscDraw,PetscReal,PetscReal,PetscReal,PetscReal,int);
extern PetscErrorCode  PetscDrawLineSetWidth(PetscDraw,PetscReal);
extern PetscErrorCode  PetscDrawLineGetWidth(PetscDraw,PetscReal*);

extern PetscErrorCode  PetscDrawPoint(PetscDraw,PetscReal,PetscReal,int);
extern PetscErrorCode  PetscDrawPointSetSize(PetscDraw,PetscReal);

extern PetscErrorCode  PetscDrawRectangle(PetscDraw,PetscReal,PetscReal,PetscReal,PetscReal,int,int,int,int);
extern PetscErrorCode  PetscDrawTriangle(PetscDraw,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal,int,int,int);
extern PetscErrorCode  PetscDrawEllipse(PetscDraw,PetscReal,PetscReal,PetscReal,PetscReal,int);
extern PetscErrorCode  PetscDrawTensorContourPatch(PetscDraw,int,int,PetscReal*,PetscReal*,PetscReal,PetscReal,PetscReal*);
extern PetscErrorCode  PetscDrawTensorContour(PetscDraw,int,int,const PetscReal[],const PetscReal[],PetscReal *);

extern PetscErrorCode  PetscDrawString(PetscDraw,PetscReal,PetscReal,int,const char[]);
extern PetscErrorCode  PetscDrawStringVertical(PetscDraw,PetscReal,PetscReal,int,const char[]);
extern PetscErrorCode  PetscDrawStringSetSize(PetscDraw,PetscReal,PetscReal);
extern PetscErrorCode  PetscDrawStringGetSize(PetscDraw,PetscReal*,PetscReal*);

extern PetscErrorCode  PetscDrawSetViewPort(PetscDraw,PetscReal,PetscReal,PetscReal,PetscReal);
extern PetscErrorCode  PetscDrawSplitViewPort(PetscDraw);

extern PetscErrorCode  PetscDrawSetCoordinates(PetscDraw,PetscReal,PetscReal,PetscReal,PetscReal);
extern PetscErrorCode  PetscDrawGetCoordinates(PetscDraw,PetscReal*,PetscReal*,PetscReal*,PetscReal*);

extern PetscErrorCode  PetscDrawSetTitle(PetscDraw,const char[]);
extern PetscErrorCode  PetscDrawAppendTitle(PetscDraw,const char[]);
extern PetscErrorCode  PetscDrawGetTitle(PetscDraw,char **);

extern PetscErrorCode  PetscDrawSetPause(PetscDraw,PetscReal);
extern PetscErrorCode  PetscDrawGetPause(PetscDraw,PetscReal*);
extern PetscErrorCode  PetscDrawPause(PetscDraw);
extern PetscErrorCode  PetscDrawSetDoubleBuffer(PetscDraw);
extern PetscErrorCode  PetscDrawFlush(PetscDraw);
extern PetscErrorCode  PetscDrawSynchronizedFlush(PetscDraw);
extern PetscErrorCode  PetscDrawClear(PetscDraw);
extern PetscErrorCode  PetscDrawSave(PetscDraw);
extern PetscErrorCode  PetscDrawSynchronizedClear(PetscDraw);
extern PetscErrorCode  PetscDrawBOP(PetscDraw);
extern PetscErrorCode  PetscDrawEOP(PetscDraw);

extern PetscErrorCode  PetscDrawSetDisplay(PetscDraw,char*);
#define PetscDrawSetFilename(a,b) PetscDrawSetDisplay(a,b)

extern PetscErrorCode  PetscDrawGetSingleton(PetscDraw,PetscDraw*);
extern PetscErrorCode  PetscDrawRestoreSingleton(PetscDraw,PetscDraw*);

/*E
    PetscDrawButton - Used to determine which button was pressed

   Level: intermediate

.seealso: PetscDrawGetMouseButton(), PetscDrawSynchronizedGetMouseButton()
E*/
typedef enum {PETSC_BUTTON_NONE,PETSC_BUTTON_LEFT,PETSC_BUTTON_CENTER,PETSC_BUTTON_RIGHT,PETSC_BUTTON_LEFT_SHIFT,PETSC_BUTTON_CENTER_SHIFT,PETSC_BUTTON_RIGHT_SHIFT} PetscDrawButton;

extern PetscErrorCode  PetscDrawGetMouseButton(PetscDraw,PetscDrawButton *,PetscReal*,PetscReal *,PetscReal *,PetscReal *);
extern PetscErrorCode  PetscDrawSynchronizedGetMouseButton(PetscDraw,PetscDrawButton *,PetscReal*,PetscReal *,PetscReal *,PetscReal *);

extern PetscErrorCode  PetscDrawZoom(PetscDraw,PetscErrorCode (*)(PetscDraw,void *),void *);

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
extern PetscErrorCode  PetscDrawViewPortsCreate(PetscDraw,PetscInt,PetscDrawViewPorts**);
extern PetscErrorCode  PetscDrawViewPortsCreateRect(PetscDraw,PetscInt,PetscInt,PetscDrawViewPorts**);
extern PetscErrorCode  PetscDrawViewPortsDestroy(PetscDrawViewPorts*);
extern PetscErrorCode  PetscDrawViewPortsSet(PetscDrawViewPorts*,PetscInt);

/*S
     PetscDrawAxis - Manages X-Y axis

   Level: advanced

  Concepts: graphics, axis

.seealso:  PetscDrawAxisCreate(), PetscDrawAxisSetLimits(), PetscDrawAxisSetColors(), PetscDrawAxisSetLabels()
S*/
typedef struct _p_PetscDrawAxis* PetscDrawAxis;

extern PetscClassId PETSC_DRAWAXIS_CLASSID;

extern PetscErrorCode  PetscDrawAxisCreate(PetscDraw,PetscDrawAxis *);
extern PetscErrorCode  PetscDrawAxisDestroy(PetscDrawAxis*);
extern PetscErrorCode  PetscDrawAxisDraw(PetscDrawAxis);
extern PetscErrorCode  PetscDrawAxisSetLimits(PetscDrawAxis,PetscReal,PetscReal,PetscReal,PetscReal);
extern PetscErrorCode  PetscDrawAxisSetHoldLimits(PetscDrawAxis,PetscBool );
extern PetscErrorCode  PetscDrawAxisSetColors(PetscDrawAxis,int,int,int);
extern PetscErrorCode  PetscDrawAxisSetLabels(PetscDrawAxis,const char[],const char[],const char[]);

/*S
     PetscDrawLG - Manages drawing x-y plots

   Level: advanced

  Concepts: graphics, axis

.seealso:  PetscDrawAxisCreate(), PetscDrawLGCreate(), PetscDrawLGAddPoint()
S*/
typedef struct _p_PetscDrawLG*   PetscDrawLG;

extern PetscClassId PETSC_DRAWLG_CLASSID;

extern PetscErrorCode  PetscDrawLGCreate(PetscDraw,int,PetscDrawLG *);
extern PetscErrorCode  PetscDrawLGDestroy(PetscDrawLG*);
extern PetscErrorCode  PetscDrawLGAddPoint(PetscDrawLG,PetscReal*,PetscReal*);
extern PetscErrorCode  PetscDrawLGAddPoints(PetscDrawLG,int,PetscReal**,PetscReal**);
extern PetscErrorCode  PetscDrawLGDraw(PetscDrawLG);
extern PetscErrorCode  PetscDrawLGPrint(PetscDrawLG);
extern PetscErrorCode  PetscDrawLGReset(PetscDrawLG);
extern PetscErrorCode  PetscDrawLGSetDimension(PetscDrawLG,PetscInt);
extern PetscErrorCode  PetscDrawLGSetLegend(PetscDrawLG,const char *const*);
extern PetscErrorCode  PetscDrawLGGetAxis(PetscDrawLG,PetscDrawAxis *);
extern PetscErrorCode  PetscDrawLGGetDraw(PetscDrawLG,PetscDraw *);
extern PetscErrorCode  PetscDrawLGIndicateDataPoints(PetscDrawLG);
extern PetscErrorCode  PetscDrawLGSetLimits(PetscDrawLG,PetscReal,PetscReal,PetscReal,PetscReal);
extern PetscErrorCode  PetscDrawLGSetColors(PetscDrawLG,const int*);

/*S
     PetscDrawSP - Manages drawing scatter plots

   Level: advanced

  Concepts: graphics, scatter plots

.seealso:  PetscDrawSPCreate()
S*/
typedef struct _p_PetscDrawSP*   PetscDrawSP;

extern PetscClassId PETSC_DRAWSP_CLASSID;

extern PetscErrorCode  PetscDrawSPCreate(PetscDraw,int,PetscDrawSP *);
extern PetscErrorCode  PetscDrawSPDestroy(PetscDrawSP*);
extern PetscErrorCode  PetscDrawSPAddPoint(PetscDrawSP,PetscReal*,PetscReal*);
extern PetscErrorCode  PetscDrawSPAddPoints(PetscDrawSP,int,PetscReal**,PetscReal**);
extern PetscErrorCode  PetscDrawSPDraw(PetscDrawSP);
extern PetscErrorCode  PetscDrawSPReset(PetscDrawSP);
extern PetscErrorCode  PetscDrawSPSetDimension(PetscDrawSP,int);
extern PetscErrorCode  PetscDrawSPGetAxis(PetscDrawSP,PetscDrawAxis *);
extern PetscErrorCode  PetscDrawSPGetDraw(PetscDrawSP,PetscDraw *);
extern PetscErrorCode  PetscDrawSPSetLimits(PetscDrawSP,PetscReal,PetscReal,PetscReal,PetscReal);
extern PetscErrorCode  PetscDrawLGSPDraw(PetscDrawLG,PetscDrawSP);

/*S
     PetscDrawHG - Manages drawing histograms

   Level: advanced

  Concepts: graphics, histograms

.seealso:  PetscDrawHGCreate()
S*/
typedef struct _p_PetscDrawHG*   PetscDrawHG;

extern PetscClassId PETSC_DRAWHG_CLASSID;

extern PetscErrorCode  PetscDrawHGCreate(PetscDraw,int,PetscDrawHG *);
extern PetscErrorCode  PetscDrawHGDestroy(PetscDrawHG*);
extern PetscErrorCode  PetscDrawHGAddValue(PetscDrawHG,PetscReal);
extern PetscErrorCode  PetscDrawHGDraw(PetscDrawHG);
extern PetscErrorCode  PetscDrawHGPrint(PetscDrawHG);
extern PetscErrorCode  PetscDrawHGReset(PetscDrawHG);
extern PetscErrorCode  PetscDrawHGGetAxis(PetscDrawHG,PetscDrawAxis *);
extern PetscErrorCode  PetscDrawHGGetDraw(PetscDrawHG,PetscDraw *);
extern PetscErrorCode  PetscDrawHGSetLimits(PetscDrawHG,PetscReal,PetscReal,int,int);
extern PetscErrorCode  PetscDrawHGSetNumberBins(PetscDrawHG,int);
extern PetscErrorCode  PetscDrawHGSetColor(PetscDrawHG,int);
extern PetscErrorCode  PetscDrawHGCalcStats(PetscDrawHG, PetscBool );
extern PetscErrorCode  PetscDrawHGIntegerBins(PetscDrawHG, PetscBool );

/*
    PetscViewer routines that allow you to access underlying PetscDraw objects
*/
extern PetscErrorCode  PetscViewerDrawGetDraw(PetscViewer,PetscInt,PetscDraw*);
extern PetscErrorCode  PetscViewerDrawBaseAdd(PetscViewer,PetscInt);
extern PetscErrorCode  PetscViewerDrawBaseSet(PetscViewer,PetscInt);
extern PetscErrorCode  PetscViewerDrawGetDrawLG(PetscViewer,PetscInt,PetscDrawLG*);
extern PetscErrorCode  PetscViewerDrawGetDrawAxis(PetscViewer,PetscInt,PetscDrawAxis*);

extern PetscErrorCode  PetscDrawUtilitySetCmapHue(unsigned char *,unsigned char *,unsigned char *,int);
extern PetscErrorCode  PetscDrawUtilitySetGamma(PetscReal);

PETSC_EXTERN_CXX_END
#endif
