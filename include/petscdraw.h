/* $Id: draw.h,v 1.27 1996/03/08 05:49:06 bsmith Exp bsmith $ */
/*
  Public include file for all of the PETSc graphics routines
*/
#if !defined(__DRAW_PACKAGE)
#define __DRAW_PACKAGE
#include "petsc.h"

#define DRAW_COOKIE PETSC_COOKIE+6

/* types of draw contexts */
#define XWINDOW    0
#define NULLWINDOW 1
 
typedef struct _Draw* Draw;

#define DRAW_WHITE       0
#define DRAW_BLACK       1
#define DRAW_RED         2
#define DRAW_GREEN       3
#define DRAW_CYAN        4
#define DRAW_BLUE        5
#define DRAW_MAGENTA     6
#define DRAW_AQUAMARINE  7
#define DRAW_FORESTGREEN 8
#define DRAW_ORANGE      9
#define DRAW_VIOLET      10
#define DRAW_BROWN       11
#define DRAW_PINK        12
#define DRAW_CORAL       13
#define DRAW_GRAY        14
#define DRAW_YELLOW      15

extern int DrawOpenX(MPI_Comm,char *,char *,int,int,int,int,Draw*);
extern int DrawOpenNull(MPI_Comm,Draw *);
extern int DrawDestroy(Draw);
extern int DrawIsNull(Draw,PetscTruth*);

extern int ViewerDrawGetDraw(Viewer, Draw*);

extern int DrawLine(Draw,double,double,double,double,int);
extern int DrawLineSetWidth(Draw,double);
extern int DrawLineGetWidth(Draw,double*);

extern int DrawPoint(Draw,double,double,int);
extern int DrawPointSetSize(Draw,double);

extern int DrawRectangle(Draw,double,double,double,double,int,int,int,int);
extern int DrawTriangle(Draw,double,double,double,double,double,double,int,int,int);

extern int DrawText(Draw,double,double,int,char*);
extern int DrawTextVertical(Draw,double,double,int,char*);
extern int DrawTextSetSize(Draw,double,double);
extern int DrawTextGetSize(Draw,double*,double*);

extern int DrawSetViewPort(Draw,double,double,double,double);
extern int DrawSetCoordinates(Draw,double,double,double,double);
extern int DrawGetCoordinates(Draw,double*,double*,double*,double*);

extern int DrawSetPause(Draw,int);
extern int DrawSetDoubleBuffer(Draw);
extern int DrawFlush(Draw);
extern int DrawSyncFlush(Draw);
extern int DrawClear(Draw);
extern int DrawSyncClear(Draw);
extern int DrawPause(Draw);
extern int DrawGetPause(Draw,int*);

typedef enum {BUTTON_NONE, BUTTON_LEFT, BUTTON_CENTER, BUTTON_RIGHT } DrawButton;
extern int DrawGetMouseButton(Draw,DrawButton *,double*,double *,double *,double *);

typedef struct _DrawAxis* DrawAxis;
#define AXIS_COOKIE PETSC_COOKIE+16
extern int DrawAxisCreate(Draw,DrawAxis *);
extern int DrawAxisDestroy(DrawAxis);
extern int DrawAxisDraw(DrawAxis);
extern int DrawAxisSetLimits(DrawAxis,double,double,double,double);
extern int DrawAxisSetColors(DrawAxis,int,int,int);
extern int DrawAxisSetLabels(DrawAxis,char*,char*,char*);

typedef struct _DrawLG*   DrawLG;
#define LG_COOKIE PETSC_COOKIE+7
extern int DrawLGCreate(Draw,int,DrawLG *);
extern int DrawLGDestroy(DrawLG);
extern int DrawLGAddPoint(DrawLG,double*,double*);
extern int DrawLGAddPoints(DrawLG,int,double**,double**);
extern int DrawLGDraw(DrawLG);
extern int DrawLGReset(DrawLG);
extern int DrawLGGetAxis(DrawLG,DrawAxis *);
extern int DrawLGGetDraw(DrawLG,Draw *);
extern int DrawLGIndicateDataPoints(DrawLG);
extern int DrawLGSetLimits(DrawLG,double,double,double,double); 

#if defined(__VEC_PACKAGE)
int DrawTensorContour(Draw,int,int,double*,double*,Vec);
#endif

#endif
