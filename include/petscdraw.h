/* $Id: petscdraw.h,v 1.75 2001/01/15 21:43:21 bsmith Exp bsmith $ */
/*
  Interface to the PETSc graphics (currently only support for X-windows
*/
#if !defined(__PETSCDRAW_H)
#define __PETSCDRAW_H
#include "petsc.h"

#define PETSC_DRAW_COOKIE PETSC_COOKIE+6

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

EXTERN int PetscDrawScalePopup(PetscDraw,double min,double max); 

EXTERN int PetscDrawLine(PetscDraw,double,double,double,double,int);
EXTERN int PetscDrawLineSetWidth(PetscDraw,double);
EXTERN int PetscDrawLineGetWidth(PetscDraw,double*);

EXTERN int PetscDrawPoint(PetscDraw,double,double,int);
EXTERN int PetscDrawPointSetSize(PetscDraw,double);

EXTERN int PetscDrawRectangle(PetscDraw,double,double,double,double,int,int,int,int);
EXTERN int PetscDrawTriangle(PetscDraw,double,double,double,double,double,double,int,int,int);
EXTERN int PetscDrawTensorContourPatch(PetscDraw,int,int,double*,double*,double,double,Scalar*);
EXTERN int PetscDrawTensorContour(PetscDraw,int,int,const double[],const double[],Scalar *);

EXTERN int PetscDrawString(PetscDraw,double,double,int,char*);
EXTERN int PetscDrawStringVertical(PetscDraw,double,double,int,char*);
EXTERN int PetscDrawStringSetSize(PetscDraw,double,double);
EXTERN int PetscDrawStringGetSize(PetscDraw,double*,double*);

EXTERN int PetscDrawSetViewPort(PetscDraw,double,double,double,double);
EXTERN int PetscDrawSplitViewPort(PetscDraw);

EXTERN int PetscDrawSetCoordinates(PetscDraw,double,double,double,double);
EXTERN int PetscDrawGetCoordinates(PetscDraw,double*,double*,double*,double*);

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

EXTERN int PetscDrawGetSingleton(PetscDraw,PetscDraw*);
EXTERN int PetscDrawRestoreSingleton(PetscDraw,PetscDraw*);

/*E
    PetscDrawButton - Used to determine which button was pressed

   Level: intermediate

.seealso: PetscDrawGetMouseButton(), PetscDrawSynchronizedGetMouseButton()
E*/
typedef enum {BUTTON_NONE,BUTTON_LEFT,BUTTON_CENTER,BUTTON_RIGHT } PetscDrawButton;

EXTERN int PetscDrawGetMouseButton(PetscDraw,PetscDrawButton *,double*,double *,double *,double *);
EXTERN int PetscDrawSynchronizedGetMouseButton(PetscDraw,PetscDrawButton *,double*,double *,double *,double *);

EXTERN int PetscDrawZoom(PetscDraw,int (*)(PetscDraw,void *),void *);

/*S
     PetscDrawViewPorts - Subwindows in a PetscDraw object

   Level: intermediate

  Concepts: graphics

.seealso:  PetscDrawViewPortsCreate(), PetscDrawViewPortsSet()
S*/
typedef struct {
  int       nports;
  double    *xl,*xr,*yl,*yr;
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

#define DRAWAXIS_COOKIE PETSC_COOKIE+16
EXTERN int PetscDrawAxisCreate(PetscDraw,PetscDrawAxis *);
EXTERN int PetscDrawAxisDestroy(PetscDrawAxis);
EXTERN int PetscDrawAxisDraw(PetscDrawAxis);
EXTERN int PetscDrawAxisSetLimits(PetscDrawAxis,double,double,double,double);
EXTERN int PetscDrawAxisSetColors(PetscDrawAxis,int,int,int);
EXTERN int PetscDrawAxisSetLabels(PetscDrawAxis,char*,char*,char*);

/*S
     PetscDrawLG - Manages drawing x-y plots

   Level: advanced

  Concepts: graphics, axis

.seealso:  PetscDrawAxisCreate(), PetscDrawLGCreate(), PetscDrawLGAddPoint()
S*/
typedef struct _p_DrawLG*   PetscDrawLG;

#define DRAWLG_COOKIE PETSC_COOKIE+7
EXTERN int PetscDrawLGCreate(PetscDraw,int,PetscDrawLG *);
EXTERN int PetscDrawLGDestroy(PetscDrawLG);
EXTERN int PetscDrawLGAddPoint(PetscDrawLG,double*,double*);
EXTERN int PetscDrawLGAddPoints(PetscDrawLG,int,double**,double**);
EXTERN int PetscDrawLGDraw(PetscDrawLG);
EXTERN int PetscDrawLGReset(PetscDrawLG);
EXTERN int PetscDrawLGSetDimension(PetscDrawLG,int);
EXTERN int PetscDrawLGGetAxis(PetscDrawLG,PetscDrawAxis *);
EXTERN int PetscDrawLGGetDraw(PetscDrawLG,PetscDraw *);
EXTERN int PetscDrawLGIndicateDataPoints(PetscDrawLG);
EXTERN int PetscDrawLGSetLimits(PetscDrawLG,double,double,double,double); 

/*S
     PetscDrawSP - Manages drawing scatter plots

   Level: advanced

  Concepts: graphics, scatter plots

.seealso:  PetscDrawSPCreate()
S*/
typedef struct _p_DrawSP*   PetscDrawSP;

#define DRAWSP_COOKIE PETSC_COOKIE+27
EXTERN int PetscDrawSPCreate(PetscDraw,int,PetscDrawSP *);
EXTERN int PetscDrawSPDestroy(PetscDrawSP);
EXTERN int PetscDrawSPAddPoint(PetscDrawSP,double*,double*);
EXTERN int PetscDrawSPAddPoints(PetscDrawSP,int,double**,double**);
EXTERN int PetscDrawSPDraw(PetscDrawSP);
EXTERN int PetscDrawSPReset(PetscDrawSP);
EXTERN int PetscDrawSPSetDimension(PetscDrawSP,int);
EXTERN int PetscDrawSPGetAxis(PetscDrawSP,PetscDrawAxis *);
EXTERN int PetscDrawSPGetDraw(PetscDrawSP,PetscDraw *);
EXTERN int PetscDrawSPSetLimits(PetscDrawSP,double,double,double,double); 

/*S
     PetscDrawHG - Manages drawing histograms

   Level: advanced

  Concepts: graphics, histograms

.seealso:  PetscDrawHGCreate()
S*/
typedef struct _p_DrawHG*   PetscDrawHG;

#define DRAWHG_COOKIE PETSC_COOKIE+15
EXTERN int PetscDrawHGCreate(PetscDraw,int,PetscDrawHG *);
EXTERN int PetscDrawHGDestroy(PetscDrawHG);
EXTERN int PetscDrawHGAddValue(PetscDrawHG,double);
EXTERN int PetscDrawHGDraw(PetscDrawHG);
EXTERN int PetscDrawHGReset(PetscDrawHG);
EXTERN int PetscDrawHGGetAxis(PetscDrawHG,PetscDrawAxis *);
EXTERN int PetscDrawHGGetDraw(PetscDrawHG,PetscDraw *);
EXTERN int PetscDrawHGSetLimits(PetscDrawHG,double,double,int,int);
EXTERN int PetscDrawHGSetNumberBins(PetscDrawHG,int);
EXTERN int PetscDrawHGSetColor(PetscDrawHG,int);

/*
    PetscViewer routines that allow you to access underlying PetscDraw objects
*/
EXTERN int PetscViewerDrawGetDraw(PetscViewer,int,PetscDraw*);
EXTERN int PetscViewerDrawGetDrawLG(PetscViewer,int,PetscDrawLG*);
EXTERN int PetscViewerDrawGetDrawAxis(PetscViewer,int,PetscDrawAxis*);

EXTERN int PetscDrawUtilitySetCmapHue(unsigned char *,unsigned char *,unsigned char *,int);
EXTERN int PetscDrawUtilitySetGamma(double);

/* Mesh management routines */
typedef struct _p_DrawMesh* PetscDrawMesh;
int PetscDrawMeshCreate(PetscDrawMesh *,double *,double *,double *,
		        int,int,int,int,int,int,int,int,int,int,int,int,int,double *,int);
int PetscDrawMeshCreateSimple(PetscDrawMesh *,double *,double *,double *,int,int,int,int,double *,int);
int PetscDrawMeshDestroy(PetscDrawMesh *);




#endif





