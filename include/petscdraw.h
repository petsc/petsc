/*
  Interface to the PETSc graphics (currently only support for X-windows
*/
#if !defined(__PETSCDRAW_H)
#define __PETSCDRAW_H
#include "petsc.h"
PETSC_EXTERN_CXX_BEGIN

extern int PETSC_DRAW_COOKIE;

/*E
    PetscDrawType - String with the name of a PetscDraw 

   Level: beginner

.seealso: PetscDrawSetType(), PetscDraw, PetscViewer
E*/
#define PetscDrawType char*
#define PETSC_DRAW_X     "x"
#define PETSC_DRAW_NULL  "null"
#define PETSC_DRAW_PS    "ps"
#define PETSC_DRAW_WIN32 "win32"
 
/*S
     PetscDraw - Abstract PETSc object for graphics

   Level: beginner

  Concepts: graphics

.seealso:  PetscDrawCreate(), PetscDrawSetType(), PetscDrawType
S*/
typedef struct _p_PetscDraw* PetscDraw;

extern PetscFList PetscDrawList;
EXTERN PetscErrorCode PetscDrawRegisterAll(const char *);
EXTERN PetscErrorCode PetscDrawRegisterDestroy(void);

EXTERN PetscErrorCode PetscDrawRegister(const char*,const char*,const char*,int(*)(PetscDraw));

/*MC
   PetscDrawRegisterDynamic - Adds a method to the Krylov subspace solver package.

   Synopsis:
   int PetscDrawRegisterDynamic(char *name_solver,char *path,char *name_create,int (*routine_create)(PetscDraw))

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

EXTERN PetscErrorCode PetscDrawGetType(PetscDraw,PetscDrawType*);
EXTERN PetscErrorCode PetscDrawSetType(PetscDraw,const PetscDrawType);
EXTERN PetscErrorCode PetscDrawCreate(MPI_Comm,const char[],const char[],int,int,int,int,PetscDraw*);
EXTERN PetscErrorCode PetscDrawSetFromOptions(PetscDraw);

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

EXTERN PetscErrorCode PetscDrawOpenX(MPI_Comm,const char[],const char[],int,int,int,int,PetscDraw*);
EXTERN PetscErrorCode PetscDrawOpenPS(MPI_Comm,char *,PetscDraw *);
#define PETSC_DRAW_FULL_SIZE    -3
#define PETSC_DRAW_HALF_SIZE    -4
#define PETSC_DRAW_THIRD_SIZE   -5
#define PETSC_DRAW_QUARTER_SIZE -6

EXTERN PetscErrorCode PetscDrawOpenNull(MPI_Comm,PetscDraw *);
EXTERN PetscErrorCode PetscDrawDestroy(PetscDraw);
EXTERN PetscErrorCode PetscDrawIsNull(PetscDraw,PetscTruth*);

EXTERN PetscErrorCode PetscDrawGetPopup(PetscDraw,PetscDraw*);
EXTERN PetscErrorCode PetscDrawCheckResizedWindow(PetscDraw);
EXTERN PetscErrorCode PetscDrawResizeWindow(PetscDraw,int,int);

EXTERN PetscErrorCode PetscDrawScalePopup(PetscDraw,PetscReal min,PetscReal max); 

EXTERN PetscErrorCode PetscDrawLine(PetscDraw,PetscReal,PetscReal,PetscReal,PetscReal,int);
EXTERN PetscErrorCode PetscDrawLineSetWidth(PetscDraw,PetscReal);
EXTERN PetscErrorCode PetscDrawLineGetWidth(PetscDraw,PetscReal*);

EXTERN PetscErrorCode PetscDrawPoint(PetscDraw,PetscReal,PetscReal,int);
EXTERN PetscErrorCode PetscDrawPointSetSize(PetscDraw,PetscReal);

EXTERN PetscErrorCode PetscDrawRectangle(PetscDraw,PetscReal,PetscReal,PetscReal,PetscReal,int,int,int,int);
EXTERN PetscErrorCode PetscDrawTriangle(PetscDraw,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal,int,int,int);
EXTERN PetscErrorCode PetscDrawEllipse(PetscDraw,PetscReal,PetscReal,PetscReal,PetscReal,int);
EXTERN PetscErrorCode PetscDrawTensorContourPatch(PetscDraw,int,int,PetscReal*,PetscReal*,PetscReal,PetscReal,PetscReal*);
EXTERN PetscErrorCode PetscDrawTensorContour(PetscDraw,int,int,const PetscReal[],const PetscReal[],PetscReal *);

EXTERN PetscErrorCode PetscDrawString(PetscDraw,PetscReal,PetscReal,int,const char[]);
EXTERN PetscErrorCode PetscDrawStringVertical(PetscDraw,PetscReal,PetscReal,int,const char[]);
EXTERN PetscErrorCode PetscDrawStringSetSize(PetscDraw,PetscReal,PetscReal);
EXTERN PetscErrorCode PetscDrawStringGetSize(PetscDraw,PetscReal*,PetscReal*);

EXTERN PetscErrorCode PetscDrawSetViewPort(PetscDraw,PetscReal,PetscReal,PetscReal,PetscReal);
EXTERN PetscErrorCode PetscDrawSplitViewPort(PetscDraw);

EXTERN PetscErrorCode PetscDrawSetCoordinates(PetscDraw,PetscReal,PetscReal,PetscReal,PetscReal);
EXTERN PetscErrorCode PetscDrawGetCoordinates(PetscDraw,PetscReal*,PetscReal*,PetscReal*,PetscReal*);

EXTERN PetscErrorCode PetscDrawSetTitle(PetscDraw,const char[]);
EXTERN PetscErrorCode PetscDrawAppendTitle(PetscDraw,const char[]);
EXTERN PetscErrorCode PetscDrawGetTitle(PetscDraw,char **);

EXTERN PetscErrorCode PetscDrawSetPause(PetscDraw,int);
EXTERN PetscErrorCode PetscDrawGetPause(PetscDraw,int*);
EXTERN PetscErrorCode PetscDrawPause(PetscDraw);
EXTERN PetscErrorCode PetscDrawSetDoubleBuffer(PetscDraw);
EXTERN PetscErrorCode PetscDrawFlush(PetscDraw);
EXTERN PetscErrorCode PetscDrawSynchronizedFlush(PetscDraw);
EXTERN PetscErrorCode PetscDrawClear(PetscDraw);
EXTERN PetscErrorCode PetscDrawSynchronizedClear(PetscDraw);
EXTERN PetscErrorCode PetscDrawBOP(PetscDraw);
EXTERN PetscErrorCode PetscDrawEOP(PetscDraw);

EXTERN PetscErrorCode PetscDrawSetDisplay(PetscDraw,char*);
#define PetscDrawSetFilename(a,b) PetscDrawSetDisplay(a,b)

EXTERN PetscErrorCode PetscDrawGetSingleton(PetscDraw,PetscDraw*);
EXTERN PetscErrorCode PetscDrawRestoreSingleton(PetscDraw,PetscDraw*);

/*E
    PetscDrawButton - Used to determine which button was pressed

   Level: intermediate

.seealso: PetscDrawGetMouseButton(), PetscDrawSynchronizedGetMouseButton()
E*/
typedef enum {BUTTON_NONE,BUTTON_LEFT,BUTTON_CENTER,BUTTON_RIGHT,BUTTON_LEFT_SHIFT,BUTTON_CENTER_SHIFT,BUTTON_RIGHT_SHIFT} PetscDrawButton;

EXTERN PetscErrorCode PetscDrawGetMouseButton(PetscDraw,PetscDrawButton *,PetscReal*,PetscReal *,PetscReal *,PetscReal *);
EXTERN PetscErrorCode PetscDrawSynchronizedGetMouseButton(PetscDraw,PetscDrawButton *,PetscReal*,PetscReal *,PetscReal *,PetscReal *);

EXTERN PetscErrorCode PetscDrawZoom(PetscDraw,int (*)(PetscDraw,void *),void *);

/*S
     PetscDrawViewPorts - Subwindows in a PetscDraw object

   Level: intermediate

  Concepts: graphics

.seealso:  PetscDrawViewPortsCreate(), PetscDrawViewPortsSet()
S*/
typedef struct {
  int       nports;
  PetscReal    *xl,*xr,*yl,*yr;
  PetscDraw draw;
} PetscDrawViewPorts;
EXTERN PetscErrorCode PetscDrawViewPortsCreate(PetscDraw,int,PetscDrawViewPorts**);
EXTERN PetscErrorCode PetscDrawViewPortsDestroy(PetscDrawViewPorts*);
EXTERN PetscErrorCode PetscDrawViewPortsSet(PetscDrawViewPorts*,int);

/*S
     PetscDrawAxis - Manages X-Y axis

   Level: advanced

  Concepts: graphics, axis

.seealso:  PetscDrawAxisCreate(), PetscDrawAxisSetLimits(), PetscDrawAxisSetColors(), PetscDrawAxisSetLabels()
S*/
typedef struct _p_DrawAxis* PetscDrawAxis;

extern int DRAWAXIS_COOKIE;

EXTERN PetscErrorCode PetscDrawAxisCreate(PetscDraw,PetscDrawAxis *);
EXTERN PetscErrorCode PetscDrawAxisDestroy(PetscDrawAxis);
EXTERN PetscErrorCode PetscDrawAxisDraw(PetscDrawAxis);
EXTERN PetscErrorCode PetscDrawAxisSetLimits(PetscDrawAxis,PetscReal,PetscReal,PetscReal,PetscReal);
EXTERN PetscErrorCode PetscDrawAxisSetHoldLimits(PetscDrawAxis,PetscTruth);
EXTERN PetscErrorCode PetscDrawAxisSetColors(PetscDrawAxis,int,int,int);
EXTERN PetscErrorCode PetscDrawAxisSetLabels(PetscDrawAxis,const char[],const char[],const char[]);

/*S
     PetscDrawLG - Manages drawing x-y plots

   Level: advanced

  Concepts: graphics, axis

.seealso:  PetscDrawAxisCreate(), PetscDrawLGCreate(), PetscDrawLGAddPoint()
S*/
typedef struct _p_DrawLG*   PetscDrawLG;

extern int DRAWLG_COOKIE;

EXTERN PetscErrorCode PetscDrawLGCreate(PetscDraw,int,PetscDrawLG *);
EXTERN PetscErrorCode PetscDrawLGDestroy(PetscDrawLG);
EXTERN PetscErrorCode PetscDrawLGAddPoint(PetscDrawLG,PetscReal*,PetscReal*);
EXTERN PetscErrorCode PetscDrawLGAddPoints(PetscDrawLG,int,PetscReal**,PetscReal**);
EXTERN PetscErrorCode PetscDrawLGDraw(PetscDrawLG);
EXTERN PetscErrorCode PetscDrawLGPrint(PetscDrawLG);
EXTERN PetscErrorCode PetscDrawLGReset(PetscDrawLG);
EXTERN PetscErrorCode PetscDrawLGSetDimension(PetscDrawLG,int);
EXTERN PetscErrorCode PetscDrawLGGetAxis(PetscDrawLG,PetscDrawAxis *);
EXTERN PetscErrorCode PetscDrawLGGetDraw(PetscDrawLG,PetscDraw *);
EXTERN PetscErrorCode PetscDrawLGIndicateDataPoints(PetscDrawLG);
EXTERN PetscErrorCode PetscDrawLGSetLimits(PetscDrawLG,PetscReal,PetscReal,PetscReal,PetscReal); 

/*S
     PetscDrawSP - Manages drawing scatter plots

   Level: advanced

  Concepts: graphics, scatter plots

.seealso:  PetscDrawSPCreate()
S*/
typedef struct _p_DrawSP*   PetscDrawSP;

extern int DRAWSP_COOKIE;

EXTERN PetscErrorCode PetscDrawSPCreate(PetscDraw,int,PetscDrawSP *);
EXTERN PetscErrorCode PetscDrawSPDestroy(PetscDrawSP);
EXTERN PetscErrorCode PetscDrawSPAddPoint(PetscDrawSP,PetscReal*,PetscReal*);
EXTERN PetscErrorCode PetscDrawSPAddPoints(PetscDrawSP,int,PetscReal**,PetscReal**);
EXTERN PetscErrorCode PetscDrawSPDraw(PetscDrawSP);
EXTERN PetscErrorCode PetscDrawSPReset(PetscDrawSP);
EXTERN PetscErrorCode PetscDrawSPSetDimension(PetscDrawSP,int);
EXTERN PetscErrorCode PetscDrawSPGetAxis(PetscDrawSP,PetscDrawAxis *);
EXTERN PetscErrorCode PetscDrawSPGetDraw(PetscDrawSP,PetscDraw *);
EXTERN PetscErrorCode PetscDrawSPSetLimits(PetscDrawSP,PetscReal,PetscReal,PetscReal,PetscReal); 
EXTERN PetscErrorCode PetscDrawLGSPDraw(PetscDrawLG,PetscDrawSP);

/*S
     PetscDrawHG - Manages drawing histograms

   Level: advanced

  Concepts: graphics, histograms

.seealso:  PetscDrawHGCreate()
S*/
typedef struct _p_DrawHG*   PetscDrawHG;

extern int DRAWHG_COOKIE;

EXTERN PetscErrorCode PetscDrawHGCreate(PetscDraw,int,PetscDrawHG *);
EXTERN PetscErrorCode PetscDrawHGDestroy(PetscDrawHG);
EXTERN PetscErrorCode PetscDrawHGAddValue(PetscDrawHG,PetscReal);
EXTERN PetscErrorCode PetscDrawHGDraw(PetscDrawHG);
EXTERN PetscErrorCode PetscDrawHGPrint(PetscDrawHG);
EXTERN PetscErrorCode PetscDrawHGReset(PetscDrawHG);
EXTERN PetscErrorCode PetscDrawHGGetAxis(PetscDrawHG,PetscDrawAxis *);
EXTERN PetscErrorCode PetscDrawHGGetDraw(PetscDrawHG,PetscDraw *);
EXTERN PetscErrorCode PetscDrawHGSetLimits(PetscDrawHG,PetscReal,PetscReal,int,int);
EXTERN PetscErrorCode PetscDrawHGSetNumberBins(PetscDrawHG,int);
EXTERN PetscErrorCode PetscDrawHGSetColor(PetscDrawHG,int);
EXTERN PetscErrorCode PetscDrawHGCalcStats(PetscDrawHG, PetscTruth);
EXTERN PetscErrorCode PetscDrawHGIntegerBins(PetscDrawHG, PetscTruth);

/*
    PetscViewer routines that allow you to access underlying PetscDraw objects
*/
EXTERN PetscErrorCode PetscViewerDrawGetDraw(PetscViewer,int,PetscDraw*);
EXTERN PetscErrorCode PetscViewerDrawGetDrawLG(PetscViewer,int,PetscDrawLG*);
EXTERN PetscErrorCode PetscViewerDrawGetDrawAxis(PetscViewer,int,PetscDrawAxis*);

EXTERN PetscErrorCode PetscDrawUtilitySetCmapHue(unsigned char *,unsigned char *,unsigned char *,int);
EXTERN PetscErrorCode PetscDrawUtilitySetGamma(PetscReal);

/* Mesh management routines */
typedef struct _p_DrawMesh* PetscDrawMesh;
PetscErrorCode PetscDrawMeshCreate(PetscDrawMesh *,PetscReal *,PetscReal *,PetscReal *,
		        int,int,int,int,int,int,int,int,int,int,int,int,int,PetscReal *,int);
PetscErrorCode PetscDrawMeshCreateSimple(PetscDrawMesh *,PetscReal *,PetscReal *,PetscReal *,int,int,int,int,PetscReal *,int);
PetscErrorCode PetscDrawMeshDestroy(PetscDrawMesh *);

PETSC_EXTERN_CXX_END
#endif
