/* $Id: petscdraw.h,v 1.71 2000/05/08 15:09:50 balay Exp bsmith $ */
/*
  Interface to the PETSc graphics (currently only support for X-windows
*/
#if !defined(__PETSCDRAW_H)
#define __PETSCDRAW_H
#include "petsc.h"

#define DRAW_COOKIE PETSC_COOKIE+6

/* types of draw contexts */
#define DRAW_X    "x"
#define DRAW_NULL "null"
#define DRAW_PS   "ps"
 
typedef struct _p_Draw* Draw;

typedef char* DrawType;
extern FList DrawList;
EXTERN int DrawRegisterAll(char *);
EXTERN int DrawRegisterDestroy(void);

EXTERN int DrawRegister(char*,char*,char*,int(*)(Draw));
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define DrawRegisterDynamic(a,b,c,d) DrawRegister(a,b,c,0)
#else
#define DrawRegisterDynamic(a,b,c,d) DrawRegister(a,b,c,d)
#endif
EXTERN int DrawGetType(Draw,DrawType*);
EXTERN int DrawSetType(Draw,DrawType);
EXTERN int DrawCreate(MPI_Comm,const char[],const char[],int,int,int,int,Draw*);
EXTERN int DrawSetFromOptions(Draw);

/*
   Number of basic colors in the draw routines, the others are used
   for a uniform colormap.
*/
#define DRAW_BASIC_COLORS 32

#define DRAW_ROTATE          -1         /* will rotate through the colors, start with 2 */
#define DRAW_WHITE            0
#define DRAW_BLACK            1
#define DRAW_RED              2
#define DRAW_GREEN            3
#define DRAW_CYAN             4
#define DRAW_BLUE             5
#define DRAW_MAGENTA          6
#define DRAW_AQUAMARINE       7
#define DRAW_FORESTGREEN      8
#define DRAW_ORANGE           9
#define DRAW_VIOLET          10
#define DRAW_BROWN           11
#define DRAW_PINK            12
#define DRAW_CORAL           13
#define DRAW_GRAY            14
#define DRAW_YELLOW          15

#define DRAW_GOLD            16
#define DRAW_LIGHTPINK       17
#define DRAW_MEDIUMTURQUOISE 18
#define DRAW_KHAKI           19
#define DRAW_DIMGRAY         20
#define DRAW_YELLOWGREEN     21
#define DRAW_SKYBLUE         22
#define DRAW_DARKGREEN       23
#define DRAW_NAVYBLUE        24
#define DRAW_SANDYBROWN      25
#define DRAW_CADETBLUE       26
#define DRAW_POWDERBLUE      27
#define DRAW_DEEPPINK        28
#define DRAW_THISTLE         29
#define DRAW_LIMEGREEN       30
#define DRAW_LAVENDERBLUSH   31


EXTERN int DrawOpenX(MPI_Comm,const char[],const char[],int,int,int,int,Draw*);
EXTERN int DrawOpenPS(MPI_Comm,char *,Draw *);
#define DRAW_FULL_SIZE    -3
#define DRAW_HALF_SIZE    -4
#define DRAW_THIRD_SIZE   -5
#define DRAW_QUARTER_SIZE -6

EXTERN int DrawOpenNull(MPI_Comm,Draw *);
EXTERN int DrawDestroy(Draw);
EXTERN int DrawIsNull(Draw,PetscTruth*);

EXTERN int DrawGetPopup(Draw,Draw*);
EXTERN int DrawCheckResizedWindow(Draw);
EXTERN int DrawResizeWindow(Draw,int,int);

EXTERN int DrawScalePopup(Draw,double min,double max); 

EXTERN int DrawLine(Draw,double,double,double,double,int);
EXTERN int DrawLineSetWidth(Draw,double);
EXTERN int DrawLineGetWidth(Draw,double*);

EXTERN int DrawPoint(Draw,double,double,int);
EXTERN int DrawPointSetSize(Draw,double);

EXTERN int DrawRectangle(Draw,double,double,double,double,int,int,int,int);
EXTERN int DrawTriangle(Draw,double,double,double,double,double,double,int,int,int);
EXTERN int DrawTensorContourPatch(Draw,int,int,double*,double*,double,double,Scalar*);
EXTERN int DrawTensorContour(Draw,int,int,const double[],const double[],Scalar *);

EXTERN int DrawString(Draw,double,double,int,char*);
EXTERN int DrawStringVertical(Draw,double,double,int,char*);
EXTERN int DrawStringSetSize(Draw,double,double);
EXTERN int DrawStringGetSize(Draw,double*,double*);

EXTERN int DrawSetViewPort(Draw,double,double,double,double);
EXTERN int DrawSplitViewPort(Draw);

EXTERN int DrawSetCoordinates(Draw,double,double,double,double);
EXTERN int DrawGetCoordinates(Draw,double*,double*,double*,double*);

EXTERN int DrawSetTitle(Draw,char *);
EXTERN int DrawAppendTitle(Draw,char *);
EXTERN int DrawGetTitle(Draw,char **);

EXTERN int DrawSetPause(Draw,int);
EXTERN int DrawGetPause(Draw,int*);
EXTERN int DrawPause(Draw);
EXTERN int DrawSetDoubleBuffer(Draw);
EXTERN int DrawFlush(Draw);
EXTERN int DrawSynchronizedFlush(Draw);
EXTERN int DrawClear(Draw);
EXTERN int DrawSynchronizedClear(Draw);
EXTERN int DrawBOP(Draw);
EXTERN int DrawEOP(Draw);

EXTERN int DrawGetSingleton(Draw,Draw*);
EXTERN int DrawRestoreSingleton(Draw,Draw*);

typedef enum {BUTTON_NONE,BUTTON_LEFT,BUTTON_CENTER,BUTTON_RIGHT } DrawButton;
EXTERN int DrawGetMouseButton(Draw,DrawButton *,double*,double *,double *,double *);
EXTERN int DrawSynchronizedGetMouseButton(Draw,DrawButton *,double*,double *,double *,double *);

EXTERN int DrawZoom(Draw,int (*)(Draw,void *),void *);

/*   Allows one to maintain a subset of viewports for a single window */
typedef struct {
  int    nports;
  double *xl,*xr,*yl,*yr;
  Draw   draw;
} DrawViewPorts;
EXTERN int DrawViewPortsCreate(Draw,int,DrawViewPorts**);
EXTERN int DrawViewPortsDestroy(DrawViewPorts*);
EXTERN int DrawViewPortsSet(DrawViewPorts*,int);

/*
    Routines for drawing X-Y axises in a Draw object
*/
typedef struct _p_DrawAxis* DrawAxis;
#define DRAWAXIS_COOKIE PETSC_COOKIE+16
EXTERN int DrawAxisCreate(Draw,DrawAxis *);
EXTERN int DrawAxisDestroy(DrawAxis);
EXTERN int DrawAxisDraw(DrawAxis);
EXTERN int DrawAxisSetLimits(DrawAxis,double,double,double,double);
EXTERN int DrawAxisSetColors(DrawAxis,int,int,int);
EXTERN int DrawAxisSetLabels(DrawAxis,char*,char*,char*);

/*
    Routines to draw line curves in X-Y space
*/
typedef struct _p_DrawLG*   DrawLG;
#define DRAWLG_COOKIE PETSC_COOKIE+7
EXTERN int DrawLGCreate(Draw,int,DrawLG *);
EXTERN int DrawLGDestroy(DrawLG);
EXTERN int DrawLGAddPoint(DrawLG,double*,double*);
EXTERN int DrawLGAddPoints(DrawLG,int,double**,double**);
EXTERN int DrawLGDraw(DrawLG);
EXTERN int DrawLGReset(DrawLG);
EXTERN int DrawLGSetDimension(DrawLG,int);
EXTERN int DrawLGGetAxis(DrawLG,DrawAxis *);
EXTERN int DrawLGGetDraw(DrawLG,Draw *);
EXTERN int DrawLGIndicateDataPoints(DrawLG);
EXTERN int DrawLGSetLimits(DrawLG,double,double,double,double); 

/*
    Routines to draw scatter plots in complex space
*/
typedef struct _p_DrawSP*   DrawSP;
#define DRAWSP_COOKIE PETSC_COOKIE+27
EXTERN int DrawSPCreate(Draw,int,DrawSP *);
EXTERN int DrawSPDestroy(DrawSP);
EXTERN int DrawSPAddPoint(DrawSP,double*,double*);
EXTERN int DrawSPAddPoints(DrawSP,int,double**,double**);
EXTERN int DrawSPDraw(DrawSP);
EXTERN int DrawSPReset(DrawSP);
EXTERN int DrawSPSetDimension(DrawSP,int);
EXTERN int DrawSPGetAxis(DrawSP,DrawAxis *);
EXTERN int DrawSPGetDraw(DrawSP,Draw *);
EXTERN int DrawSPSetLimits(DrawSP,double,double,double,double); 

/*
    Routines to draw histograms
*/
typedef struct _p_DrawHG*   DrawHG;
#define DRAWHG_COOKIE PETSC_COOKIE+15
EXTERN int DrawHGCreate(Draw,int,DrawHG *);
EXTERN int DrawHGDestroy(DrawHG);
EXTERN int DrawHGAddValue(DrawHG,double);
EXTERN int DrawHGDraw(DrawHG);
EXTERN int DrawHGReset(DrawHG);
EXTERN int DrawHGGetAxis(DrawHG,DrawAxis *);
EXTERN int DrawHGGetDraw(DrawHG,Draw *);
EXTERN int DrawHGSetLimits(DrawHG,double,double,int,int);
EXTERN int DrawHGSetNumberBins(DrawHG,int);
EXTERN int DrawHGSetColor(DrawHG,int);

/*
    Viewer routines that allow you to access underlying Draw objects
*/
EXTERN int ViewerDrawGetDraw(Viewer,int,Draw*);
EXTERN int ViewerDrawGetDrawLG(Viewer,int,DrawLG*);
EXTERN int ViewerDrawGetDrawAxis(Viewer,int,DrawAxis*);

EXTERN int DrawUtilitySetCmapHue(unsigned char *,unsigned char *,unsigned char *,int);
EXTERN int DrawUtilitySetGamma(double);

/* Mesh management routines */
typedef struct _p_DrawMesh* DrawMesh;
int DrawMeshCreate(DrawMesh *,
		    double *,double *,double *,
		    int,int,int,int,int,int,int,int,int,
		    int,int,int,int,double *,int);
int DrawMeshCreateSimple(DrawMesh *,double *,double *,double *,
			  int,int,int,int,double *,int);
int DrawMeshDestroy(DrawMesh *);




#endif





