/*
  Interface to the PETSc graphics (currently only support for X-windows
*/
#if !defined(__PETSCDRAW_H)
#define __PETSCDRAW_H
#include "petscsys.h"
PETSC_EXTERN_CXX_BEGIN

extern PetscClassId PETSC_DRAW_CLASSID;

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
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawRegisterAll(const char[]);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawInitializePackage(const char[]);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawRegisterDestroy(void);

extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawRegister(const char*,const char*,const char*,PetscErrorCode(*)(PetscDraw));

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

extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawGetType(PetscDraw,const PetscDrawType*);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawSetType(PetscDraw,const PetscDrawType);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawCreate(MPI_Comm,const char[],const char[],int,int,int,int,PetscDraw*);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawSetFromOptions(PetscDraw);

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

extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawOpenX(MPI_Comm,const char[],const char[],int,int,int,int,PetscDraw*);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawOpenPS(MPI_Comm,char *,PetscDraw *);
#define PETSC_DRAW_FULL_SIZE    -3
#define PETSC_DRAW_HALF_SIZE    -4
#define PETSC_DRAW_THIRD_SIZE   -5
#define PETSC_DRAW_QUARTER_SIZE -6

extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawOpenNull(MPI_Comm,PetscDraw *);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawDestroy(PetscDraw);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawIsNull(PetscDraw,PetscBool *);

extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawGetPopup(PetscDraw,PetscDraw*);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawCheckResizedWindow(PetscDraw);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawResizeWindow(PetscDraw,int,int);

extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawScalePopup(PetscDraw,PetscReal,PetscReal);

extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawLine(PetscDraw,PetscReal,PetscReal,PetscReal,PetscReal,int);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawLineSetWidth(PetscDraw,PetscReal);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawLineGetWidth(PetscDraw,PetscReal*);

extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawPoint(PetscDraw,PetscReal,PetscReal,int);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawPointSetSize(PetscDraw,PetscReal);

extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawRectangle(PetscDraw,PetscReal,PetscReal,PetscReal,PetscReal,int,int,int,int);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawTriangle(PetscDraw,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal,int,int,int);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawEllipse(PetscDraw,PetscReal,PetscReal,PetscReal,PetscReal,int);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawTensorContourPatch(PetscDraw,int,int,PetscReal*,PetscReal*,PetscReal,PetscReal,PetscReal*);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawTensorContour(PetscDraw,int,int,const PetscReal[],const PetscReal[],PetscReal *);

extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawString(PetscDraw,PetscReal,PetscReal,int,const char[]);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawStringVertical(PetscDraw,PetscReal,PetscReal,int,const char[]);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawStringSetSize(PetscDraw,PetscReal,PetscReal);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawStringGetSize(PetscDraw,PetscReal*,PetscReal*);

extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawSetViewPort(PetscDraw,PetscReal,PetscReal,PetscReal,PetscReal);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawSplitViewPort(PetscDraw);

extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawSetCoordinates(PetscDraw,PetscReal,PetscReal,PetscReal,PetscReal);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawGetCoordinates(PetscDraw,PetscReal*,PetscReal*,PetscReal*,PetscReal*);

extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawSetTitle(PetscDraw,const char[]);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawAppendTitle(PetscDraw,const char[]);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawGetTitle(PetscDraw,char **);

extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawSetPause(PetscDraw,PetscReal);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawGetPause(PetscDraw,PetscReal*);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawPause(PetscDraw);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawSetDoubleBuffer(PetscDraw);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawFlush(PetscDraw);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawSynchronizedFlush(PetscDraw);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawClear(PetscDraw);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawSynchronizedClear(PetscDraw);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawBOP(PetscDraw);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawEOP(PetscDraw);

extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawSetDisplay(PetscDraw,char*);
#define PetscDrawSetFilename(a,b) PetscDrawSetDisplay(a,b)

extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawGetSingleton(PetscDraw,PetscDraw*);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawRestoreSingleton(PetscDraw,PetscDraw*);

/*E
    PetscDrawButton - Used to determine which button was pressed

   Level: intermediate

.seealso: PetscDrawGetMouseButton(), PetscDrawSynchronizedGetMouseButton()
E*/
typedef enum {BUTTON_NONE,BUTTON_LEFT,BUTTON_CENTER,BUTTON_RIGHT,BUTTON_LEFT_SHIFT,BUTTON_CENTER_SHIFT,BUTTON_RIGHT_SHIFT} PetscDrawButton;

extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawGetMouseButton(PetscDraw,PetscDrawButton *,PetscReal*,PetscReal *,PetscReal *,PetscReal *);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawSynchronizedGetMouseButton(PetscDraw,PetscDrawButton *,PetscReal*,PetscReal *,PetscReal *,PetscReal *);

extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawZoom(PetscDraw,PetscErrorCode (*)(PetscDraw,void *),void *);

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
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawViewPortsCreate(PetscDraw,PetscInt,PetscDrawViewPorts**);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawViewPortsCreateRect(PetscDraw,PetscInt,PetscInt,PetscDrawViewPorts**);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawViewPortsDestroy(PetscDrawViewPorts*);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawViewPortsSet(PetscDrawViewPorts*,PetscInt);

/*S
     PetscDrawAxis - Manages X-Y axis

   Level: advanced

  Concepts: graphics, axis

.seealso:  PetscDrawAxisCreate(), PetscDrawAxisSetLimits(), PetscDrawAxisSetColors(), PetscDrawAxisSetLabels()
S*/
typedef struct _p_DrawAxis* PetscDrawAxis;

extern PetscClassId DRAWAXIS_CLASSID;

extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawAxisCreate(PetscDraw,PetscDrawAxis *);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawAxisDestroy(PetscDrawAxis);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawAxisDraw(PetscDrawAxis);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawAxisSetLimits(PetscDrawAxis,PetscReal,PetscReal,PetscReal,PetscReal);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawAxisSetHoldLimits(PetscDrawAxis,PetscBool );
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawAxisSetColors(PetscDrawAxis,int,int,int);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawAxisSetLabels(PetscDrawAxis,const char[],const char[],const char[]);

/*S
     PetscDrawLG - Manages drawing x-y plots

   Level: advanced

  Concepts: graphics, axis

.seealso:  PetscDrawAxisCreate(), PetscDrawLGCreate(), PetscDrawLGAddPoint()
S*/
typedef struct _p_DrawLG*   PetscDrawLG;

extern PetscClassId DRAWLG_CLASSID;

extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawLGCreate(PetscDraw,int,PetscDrawLG *);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawLGDestroy(PetscDrawLG);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawLGAddPoint(PetscDrawLG,PetscReal*,PetscReal*);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawLGAddPoints(PetscDrawLG,int,PetscReal**,PetscReal**);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawLGDraw(PetscDrawLG);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawLGPrint(PetscDrawLG);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawLGReset(PetscDrawLG);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawLGSetDimension(PetscDrawLG,PetscInt);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawLGGetAxis(PetscDrawLG,PetscDrawAxis *);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawLGGetDraw(PetscDrawLG,PetscDraw *);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawLGIndicateDataPoints(PetscDrawLG);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawLGSetLimits(PetscDrawLG,PetscReal,PetscReal,PetscReal,PetscReal);

/*S
     PetscDrawSP - Manages drawing scatter plots

   Level: advanced

  Concepts: graphics, scatter plots

.seealso:  PetscDrawSPCreate()
S*/
typedef struct _p_DrawSP*   PetscDrawSP;

extern PetscClassId DRAWSP_CLASSID;

extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawSPCreate(PetscDraw,int,PetscDrawSP *);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawSPDestroy(PetscDrawSP);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawSPAddPoint(PetscDrawSP,PetscReal*,PetscReal*);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawSPAddPoints(PetscDrawSP,int,PetscReal**,PetscReal**);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawSPDraw(PetscDrawSP);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawSPReset(PetscDrawSP);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawSPSetDimension(PetscDrawSP,int);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawSPGetAxis(PetscDrawSP,PetscDrawAxis *);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawSPGetDraw(PetscDrawSP,PetscDraw *);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawSPSetLimits(PetscDrawSP,PetscReal,PetscReal,PetscReal,PetscReal);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawLGSPDraw(PetscDrawLG,PetscDrawSP);

/*S
     PetscDrawHG - Manages drawing histograms

   Level: advanced

  Concepts: graphics, histograms

.seealso:  PetscDrawHGCreate()
S*/
typedef struct _p_DrawHG*   PetscDrawHG;

extern PetscClassId DRAWHG_CLASSID;

extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawHGCreate(PetscDraw,int,PetscDrawHG *);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawHGDestroy(PetscDrawHG);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawHGAddValue(PetscDrawHG,PetscReal);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawHGDraw(PetscDrawHG);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawHGPrint(PetscDrawHG);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawHGReset(PetscDrawHG);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawHGGetAxis(PetscDrawHG,PetscDrawAxis *);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawHGGetDraw(PetscDrawHG,PetscDraw *);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawHGSetLimits(PetscDrawHG,PetscReal,PetscReal,int,int);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawHGSetNumberBins(PetscDrawHG,int);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawHGSetColor(PetscDrawHG,int);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawHGCalcStats(PetscDrawHG, PetscBool );
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawHGIntegerBins(PetscDrawHG, PetscBool );

/*
    PetscViewer routines that allow you to access underlying PetscDraw objects
*/
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscViewerDrawGetDraw(PetscViewer,PetscInt,PetscDraw*);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscViewerDrawBaseAdd(PetscViewer,PetscInt);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscViewerDrawBaseSet(PetscViewer,PetscInt);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscViewerDrawGetDrawLG(PetscViewer,PetscInt,PetscDrawLG*);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscViewerDrawGetDrawAxis(PetscViewer,PetscInt,PetscDrawAxis*);

extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawUtilitySetCmapHue(unsigned char *,unsigned char *,unsigned char *,int);
extern PetscErrorCode PETSCSYS_DLLEXPORT PetscDrawUtilitySetGamma(PetscReal);

PETSC_EXTERN_CXX_END
#endif
