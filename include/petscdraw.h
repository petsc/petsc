/* $Id: petscdraw.h,v 1.79 2001/09/07 20:07:55 bsmith Exp $ */
/*
  Interface to the PETSc graphics (currently only support for X-windows
*/
#if !defined(__PETSCDRAW_H)
#define __PETSCDRAW_H
#include "petsc.h"

extern int PETSC_DRAW_COOKIE;

/*E
    PetscDrawType - String with the name of a PetscDraw 

   Level: beginner

.seealso: PetscDrawSetType(), PetscDraw, PetscViewer
E*/
typedef char* PetscDrawType;
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
EXTERN int PetscDrawRegisterAll(char *);
EXTERN int PetscDrawRegisterDestroy(void);

EXTERN int PetscDrawRegister(char*,char*,char*,int(*)(PetscDraw));

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

EXTERN int PetscDrawGetType(PetscDraw,PetscDrawType*);
EXTERN int PetscDrawSetType(PetscDraw,PetscDrawType);
EXTERN int PetscDrawCreate(MPI_Comm,const char[],const char[],int,int,int,int,PetscDraw*);
EXTERN int PetscDrawSetFromOptions(PetscDraw);

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

EXTERN int PetscDrawOpenX(MPI_Comm,const char[],const char[],int,int,int,int,PetscDraw*);
EXTERN int PetscDrawOpenPS(MPI_Comm,char *,PetscDraw *);
#define PETSC_DRAW_FULL_SIZE    -3
#define PETSC_DRAW_HALF_SIZE    -4
#define PETSC_DRAW_THIRD_SIZE   -5
#define PETSC_DRAW_QUARTER_SIZE -6

EXTERN int PetscDrawOpenNull(MPI_Comm,PetscDraw *);
EXTERN int PetscDrawDestroy(PetscDraw);
EXTERN int PetscDrawIsNull(PetscDraw,PetscTruth*);

EXTERN int PetscDrawGetPopup(PetscDraw,PetscDraw*);
EXTERN int PetscDrawCheckResizedWindow(PetscDraw);
EXTERN int PetscDrawResizeWindow(PetscDraw,int,int);

EXTERN int PetscDrawScalePopup(PetscDraw,PetscReal min,PetscReal max); 

EXTERN int PetscDrawLine(PetscDraw,PetscReal,PetscReal,PetscReal,PetscReal,int);
EXTERN int PetscDrawLineSetWidth(PetscDraw,PetscReal);
EXTERN int PetscDrawLineGetWidth(PetscDraw,PetscReal*);

EXTERN int PetscDrawPoint(PetscDraw,PetscReal,PetscReal,int);
EXTERN int PetscDrawPointSetSize(PetscDraw,PetscReal);

EXTERN int PetscDrawRectangle(PetscDraw,PetscReal,PetscReal,PetscReal,PetscReal,int,int,int,int);
EXTERN int PetscDrawTriangle(PetscDraw,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal,PetscReal,int,int,int);
EXTERN int PetscDrawEllipse(PetscDraw,PetscReal,PetscReal,PetscReal,PetscReal,int);
EXTERN int PetscDrawTensorContourPatch(PetscDraw,int,int,PetscReal*,PetscReal*,PetscReal,PetscReal,PetscReal*);
EXTERN int PetscDrawTensorContour(PetscDraw,int,int,const PetscReal[],const PetscReal[],PetscReal *);

EXTERN int PetscDrawString(PetscDraw,PetscReal,PetscReal,int,char*);
EXTERN int PetscDrawStringVertical(PetscDraw,PetscReal,PetscReal,int,char*);
EXTERN int PetscDrawStringSetSize(PetscDraw,PetscReal,PetscReal);
EXTERN int PetscDrawStringGetSize(PetscDraw,PetscReal*,PetscReal*);

EXTERN int PetscDrawSetViewPort(PetscDraw,PetscReal,PetscReal,PetscReal,PetscReal);
EXTERN int PetscDrawSplitViewPort(PetscDraw);

EXTERN int PetscDrawSetCoordinates(PetscDraw,PetscReal,PetscReal,PetscReal,PetscReal);
EXTERN int PetscDrawGetCoordinates(PetscDraw,PetscReal*,PetscReal*,PetscReal*,PetscReal*);

EXTERN int PetscDrawSetTitle(PetscDraw,char *);
EXTERN int PetscDrawAppendTitle(PetscDraw,char *);
EXTERN int PetscDrawGetTitle(PetscDraw,char **);

EXTERN int PetscDrawSetPause(PetscDraw,int);
EXTERN int PetscDrawGetPause(PetscDraw,int*);
EXTERN int PetscDrawPause(PetscDraw);
EXTERN int PetscDrawSetDoubleBuffer(PetscDraw);
EXTERN int PetscDrawFlush(PetscDraw);
EXTERN int PetscDrawSynchronizedFlush(PetscDraw);
EXTERN int PetscDrawClear(PetscDraw);
EXTERN int PetscDrawSynchronizedClear(PetscDraw);
EXTERN int PetscDrawBOP(PetscDraw);
EXTERN int PetscDrawEOP(PetscDraw);

EXTERN int PetscDrawSetDisplay(PetscDraw,char*);
#define PetscDrawSetFilename(a,b) PetscDrawSetDisplay(a,b)

EXTERN int PetscDrawGetSingleton(PetscDraw,PetscDraw*);
EXTERN int PetscDrawRestoreSingleton(PetscDraw,PetscDraw*);

/*E
    PetscDrawButton - Used to determine which button was pressed

   Level: intermediate

.seealso: PetscDrawGetMouseButton(), PetscDrawSynchronizedGetMouseButton()
E*/
typedef enum {BUTTON_NONE,BUTTON_LEFT,BUTTON_CENTER,BUTTON_RIGHT,BUTTON_LEFT_SHIFT,BUTTON_CENTER_SHIFT,BUTTON_RIGHT_SHIFT} PetscDrawButton;

EXTERN int PetscDrawGetMouseButton(PetscDraw,PetscDrawButton *,PetscReal*,PetscReal *,PetscReal *,PetscReal *);
EXTERN int PetscDrawSynchronizedGetMouseButton(PetscDraw,PetscDrawButton *,PetscReal*,PetscReal *,PetscReal *,PetscReal *);

EXTERN int PetscDrawZoom(PetscDraw,int (*)(PetscDraw,void *),void *);

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
EXTERN int PetscDrawViewPortsCreate(PetscDraw,int,PetscDrawViewPorts**);
EXTERN int PetscDrawViewPortsDestroy(PetscDrawViewPorts*);
EXTERN int PetscDrawViewPortsSet(PetscDrawViewPorts*,int);

/*S
     PetscDrawAxis - Manages X-Y axis

   Level: advanced

  Concepts: graphics, axis

.seealso:  PetscDrawAxisCreate(), PetscDrawAxisSetLimits(), PetscDrawAxisSetColors(), PetscDrawAxisSetLabels()
S*/
typedef struct _p_DrawAxis* PetscDrawAxis;

extern int DRAWAXIS_COOKIE;

EXTERN int PetscDrawAxisCreate(PetscDraw,PetscDrawAxis *);
EXTERN int PetscDrawAxisDestroy(PetscDrawAxis);
EXTERN int PetscDrawAxisDraw(PetscDrawAxis);
EXTERN int PetscDrawAxisSetLimits(PetscDrawAxis,PetscReal,PetscReal,PetscReal,PetscReal);
EXTERN int PetscDrawAxisSetHoldLimits(PetscDrawAxis,PetscTruth);
EXTERN int PetscDrawAxisSetColors(PetscDrawAxis,int,int,int);
EXTERN int PetscDrawAxisSetLabels(PetscDrawAxis,char*,char*,char*);

/*S
     PetscDrawLG - Manages drawing x-y plots

   Level: advanced

  Concepts: graphics, axis

.seealso:  PetscDrawAxisCreate(), PetscDrawLGCreate(), PetscDrawLGAddPoint()
S*/
typedef struct _p_DrawLG*   PetscDrawLG;

extern int DRAWLG_COOKIE;

EXTERN int PetscDrawLGCreate(PetscDraw,int,PetscDrawLG *);
EXTERN int PetscDrawLGDestroy(PetscDrawLG);
EXTERN int PetscDrawLGAddPoint(PetscDrawLG,PetscReal*,PetscReal*);
EXTERN int PetscDrawLGAddPoints(PetscDrawLG,int,PetscReal**,PetscReal**);
EXTERN int PetscDrawLGDraw(PetscDrawLG);
EXTERN int PetscDrawLGPrint(PetscDrawLG);
EXTERN int PetscDrawLGReset(PetscDrawLG);
EXTERN int PetscDrawLGSetDimension(PetscDrawLG,int);
EXTERN int PetscDrawLGGetAxis(PetscDrawLG,PetscDrawAxis *);
EXTERN int PetscDrawLGGetDraw(PetscDrawLG,PetscDraw *);
EXTERN int PetscDrawLGIndicateDataPoints(PetscDrawLG);
EXTERN int PetscDrawLGSetLimits(PetscDrawLG,PetscReal,PetscReal,PetscReal,PetscReal); 

/*S
     PetscDrawSP - Manages drawing scatter plots

   Level: advanced

  Concepts: graphics, scatter plots

.seealso:  PetscDrawSPCreate()
S*/
typedef struct _p_DrawSP*   PetscDrawSP;

extern int DRAWSP_COOKIE;

EXTERN int PetscDrawSPCreate(PetscDraw,int,PetscDrawSP *);
EXTERN int PetscDrawSPDestroy(PetscDrawSP);
EXTERN int PetscDrawSPAddPoint(PetscDrawSP,PetscReal*,PetscReal*);
EXTERN int PetscDrawSPAddPoints(PetscDrawSP,int,PetscReal**,PetscReal**);
EXTERN int PetscDrawSPDraw(PetscDrawSP);
EXTERN int PetscDrawSPReset(PetscDrawSP);
EXTERN int PetscDrawSPSetDimension(PetscDrawSP,int);
EXTERN int PetscDrawSPGetAxis(PetscDrawSP,PetscDrawAxis *);
EXTERN int PetscDrawSPGetDraw(PetscDrawSP,PetscDraw *);
EXTERN int PetscDrawSPSetLimits(PetscDrawSP,PetscReal,PetscReal,PetscReal,PetscReal); 

/*S
     PetscDrawHG - Manages drawing histograms

   Level: advanced

  Concepts: graphics, histograms

.seealso:  PetscDrawHGCreate()
S*/
typedef struct _p_DrawHG*   PetscDrawHG;

extern int DRAWHG_COOKIE;

EXTERN int PetscDrawHGCreate(PetscDraw,int,PetscDrawHG *);
EXTERN int PetscDrawHGDestroy(PetscDrawHG);
EXTERN int PetscDrawHGAddValue(PetscDrawHG,PetscReal);
EXTERN int PetscDrawHGDraw(PetscDrawHG);
EXTERN int PetscDrawHGPrint(PetscDrawHG);
EXTERN int PetscDrawHGReset(PetscDrawHG);
EXTERN int PetscDrawHGGetAxis(PetscDrawHG,PetscDrawAxis *);
EXTERN int PetscDrawHGGetDraw(PetscDrawHG,PetscDraw *);
EXTERN int PetscDrawHGSetLimits(PetscDrawHG,PetscReal,PetscReal,int,int);
EXTERN int PetscDrawHGSetNumberBins(PetscDrawHG,int);
EXTERN int PetscDrawHGSetColor(PetscDrawHG,int);
EXTERN int PetscDrawHGCalcStats(PetscDrawHG, PetscTruth);
EXTERN int PetscDrawHGIntegerBins(PetscDrawHG, PetscTruth);

/*
    PetscViewer routines that allow you to access underlying PetscDraw objects
*/
EXTERN int PetscViewerDrawGetDraw(PetscViewer,int,PetscDraw*);
EXTERN int PetscViewerDrawGetDrawLG(PetscViewer,int,PetscDrawLG*);
EXTERN int PetscViewerDrawGetDrawAxis(PetscViewer,int,PetscDrawAxis*);

EXTERN int PetscDrawUtilitySetCmapHue(unsigned char *,unsigned char *,unsigned char *,int);
EXTERN int PetscDrawUtilitySetGamma(PetscReal);

/* Mesh management routines */
typedef struct _p_DrawMesh* PetscDrawMesh;
int PetscDrawMeshCreate(PetscDrawMesh *,PetscReal *,PetscReal *,PetscReal *,
		        int,int,int,int,int,int,int,int,int,int,int,int,int,PetscReal *,int);
int PetscDrawMeshCreateSimple(PetscDrawMesh *,PetscReal *,PetscReal *,PetscReal *,int,int,int,int,PetscReal *,int);
int PetscDrawMeshDestroy(PetscDrawMesh *);




#endif





