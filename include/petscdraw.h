/* $Id: draw.h,v 1.66 1999/10/13 20:39:18 bsmith Exp bsmith $ */
/*
  Interface to the PETSc graphics (currently only support for X-windows
*/
#if !defined(__DRAW_H)
#define __DRAW_H
#include "petsc.h"

#define DRAW_COOKIE PETSC_COOKIE+6

/* types of draw contexts */
#define DRAW_X    "x"
#define DRAW_NULL "null"
 
typedef struct _p_Draw* Draw;

typedef char* DrawType;
extern FList DrawList;
extern int DrawRegisterAll(char *);
extern int DrawRegisterDestroy(void);

extern int DrawRegister_Private(char*,char*,char*,int(*)(Draw));
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#define DrawRegister(a,b,c,d) DrawRegister_Private(a,b,c,0)
#else
#define DrawRegister(a,b,c,d) DrawRegister_Private(a,b,c,d)
#endif
extern int DrawGetType(Draw,DrawType*);
extern int DrawSetType(Draw,DrawType);
extern int DrawCreate(MPI_Comm,const char[],const char[],int,int,int,int,Draw*);
extern int DrawSetFromOptions(Draw);

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


extern int DrawOpenX(MPI_Comm,const char[],const char[],int,int,int,int,Draw*);

extern int DrawOpenNull(MPI_Comm,Draw *);
extern int DrawDestroy(Draw);
extern int DrawIsNull(Draw,PetscTruth*);

extern int DrawGetPopup(Draw,Draw*);
extern int DrawCheckResizedWindow(Draw);
extern int DrawResizeWindow(Draw,int,int);

extern int DrawScalePopup(Draw,double min,double max); 

extern int DrawLine(Draw,double,double,double,double,int);
extern int DrawLineSetWidth(Draw,double);
extern int DrawLineGetWidth(Draw,double*);

extern int DrawPoint(Draw,double,double,int);
extern int DrawPointSetSize(Draw,double);

extern int DrawRectangle(Draw,double,double,double,double,int,int,int,int);
extern int DrawTriangle(Draw,double,double,double,double,double,double,int,int,int);
extern int DrawTensorContourPatch(Draw,int,int,double*,double*,double,double,Scalar*);
extern int DrawTensorContour(Draw,int,int,const double[],const double[],Scalar *);

extern int DrawString(Draw,double,double,int,char*);
extern int DrawStringVertical(Draw,double,double,int,char*);
extern int DrawStringSetSize(Draw,double,double);
extern int DrawStringGetSize(Draw,double*,double*);

extern int DrawSetViewPort(Draw,double,double,double,double);
extern int DrawSplitViewPort(Draw);

extern int DrawSetCoordinates(Draw,double,double,double,double);
extern int DrawGetCoordinates(Draw,double*,double*,double*,double*);

extern int DrawSetTitle(Draw,char *);
extern int DrawAppendTitle(Draw,char *);
extern int DrawGetTitle(Draw,char **);

extern int DrawSetPause(Draw,int);
extern int DrawGetPause(Draw,int*);
extern int DrawPause(Draw);
extern int DrawSetDoubleBuffer(Draw);
extern int DrawFlush(Draw);
extern int DrawSynchronizedFlush(Draw);
extern int DrawClear(Draw);
extern int DrawSynchronizedClear(Draw);
extern int DrawBOP(Draw);
extern int DrawEOP(Draw);

extern int DrawGetSingleton(Draw,Draw*);
extern int DrawRestoreSingleton(Draw,Draw*);

typedef enum {BUTTON_NONE, BUTTON_LEFT, BUTTON_CENTER, BUTTON_RIGHT } DrawButton;
extern int DrawGetMouseButton(Draw,DrawButton *,double*,double *,double *,double *);
extern int DrawSynchronizedGetMouseButton(Draw,DrawButton *,double*,double *,double *,double *);

extern int DrawZoom(Draw,int (*)(Draw,void *),void *);
/*
    Routines for drawing X-Y axises in a Draw object
*/
typedef struct _p_DrawAxis* DrawAxis;
#define DRAWAXIS_COOKIE PETSC_COOKIE+16
extern int DrawAxisCreate(Draw,DrawAxis *);
extern int DrawAxisDestroy(DrawAxis);
extern int DrawAxisDraw(DrawAxis);
extern int DrawAxisSetLimits(DrawAxis,double,double,double,double);
extern int DrawAxisSetColors(DrawAxis,int,int,int);
extern int DrawAxisSetLabels(DrawAxis,char*,char*,char*);

/*
    Routines to draw line curves in X-Y space
*/
typedef struct _p_DrawLG*   DrawLG;
#define DRAWLG_COOKIE PETSC_COOKIE+7
extern int DrawLGCreate(Draw,int,DrawLG *);
extern int DrawLGDestroy(DrawLG);
extern int DrawLGAddPoint(DrawLG,double*,double*);
extern int DrawLGAddPoints(DrawLG,int,double**,double**);
extern int DrawLGDraw(DrawLG);
extern int DrawLGReset(DrawLG);
extern int DrawLGSetDimension(DrawLG,int);
extern int DrawLGGetAxis(DrawLG,DrawAxis *);
extern int DrawLGGetDraw(DrawLG,Draw *);
extern int DrawLGIndicateDataPoints(DrawLG);
extern int DrawLGSetLimits(DrawLG,double,double,double,double); 

/*
    Routines to draw scatter plots in complex space
*/
typedef struct _p_DrawSP*   DrawSP;
#define DRAWSP_COOKIE PETSC_COOKIE+27
extern int DrawSPCreate(Draw,int,DrawSP *);
extern int DrawSPDestroy(DrawSP);
extern int DrawSPAddPoint(DrawSP,double*,double*);
extern int DrawSPAddPoints(DrawSP,int,double**,double**);
extern int DrawSPDraw(DrawSP);
extern int DrawSPReset(DrawSP);
extern int DrawSPSetDimension(DrawSP,int);
extern int DrawSPGetAxis(DrawSP,DrawAxis *);
extern int DrawSPGetDraw(DrawSP,Draw *);
extern int DrawSPSetLimits(DrawSP,double,double,double,double); 

/*
    Routines to draw histograms
*/
typedef struct _p_DrawHist*   DrawHist;
#define DRAWHIST_COOKIE PETSC_COOKIE+15
extern int DrawHistCreate(Draw, int, DrawHist *);
extern int DrawHistDestroy(DrawHist);
extern int DrawHistAddValue(DrawHist, double);
extern int DrawHistDraw(DrawHist);
extern int DrawHistReset(DrawHist);
extern int DrawHistGetAxis(DrawHist, DrawAxis *);
extern int DrawHistGetDraw(DrawHist, Draw *);
extern int DrawHistSetLimits(DrawHist, double, double, int, int);
extern int DrawHistSetNumberBins(DrawHist, int);
extern int DrawHistSetColor(DrawHist,int);

/*
    Viewer routines that allow you to access underlying Draw objects
*/
extern int ViewerDrawGetDraw(Viewer,int, Draw*);
extern int ViewerDrawGetDrawLG(Viewer,int, DrawLG*);
extern int ViewerDrawGetDrawAxis(Viewer,int, DrawAxis*);

/* Mesh management routines */
typedef struct _p_DrawMesh* DrawMesh;
int DrawMeshCreate( DrawMesh *, 
		    double *, double *, double *,
		    int, int, int, int, int, int, int, int, int,
		    int, int, int, int, double *, int );
int DrawMeshCreateSimple( DrawMesh *, double *, double *, double *,
			  int, int, int, int, double *, int );
int DrawMeshDestroy( DrawMesh * );




#endif





