/*
  Public include file for all of the PETSc graphics routines
*/
#if !defined(__DRAW_PACKAGE)
#define __DRAW_PACKAGE
#include "petsc.h"

#define DRAW_COOKIE PETSC_COOKIE+6

/* types of draw contexts */
#define XWINDOW 0
 
typedef struct _DrawCtx* DrawCtx;

#define DRAW_WHITE  0
#define DRAW_BLACK  1
#define DRAW_RED    2
#define DRAW_YELLOW 3
#define DRAW_GREEN  4
#define DRAW_CYAN   5
#define DRAW_BLUE   6

extern int DrawOpenX(MPI_Comm,char *,char *,int,int,int,int,DrawCtx*);
extern int DrawDestroy(DrawCtx);

extern int DrawLine(DrawCtx,double,double,double,double,int,int);
extern int DrawLineSetWidth(DrawCtx,double);

extern int DrawPoint(DrawCtx,double,double,int);
extern int DrawPointSetSize(DrawCtx,double);

extern int DrawRectangle(DrawCtx,double,double,double,double,int,int,int,int);
extern int DrawTriangle(DrawCtx,double,double,double,double,double,double,int,int,int);

extern int DrawText(DrawCtx,double,double,int,char*);
extern int DrawTextVertical(DrawCtx,double,double,int,char*);
extern int DrawTextSetSize(DrawCtx,double,double);
extern int DrawTextGetSize(DrawCtx,double*,double*);

extern int DrawSetViewPort(DrawCtx,double,double,double,double);
extern int DrawSetCoordinates(DrawCtx,double,double,double,double);
extern int DrawGetCoordinates(DrawCtx,double*,double*,double*,double*);

extern int DrawSetPause(DrawCtx,int);
extern int DrawSetDoubleBuffer(DrawCtx);
extern int DrawFlush(DrawCtx);
extern int DrawSyncFlush(DrawCtx);
extern int DrawClear(DrawCtx);

/* routines related to drawing Axis and line graphs */

typedef struct _DrawAxisCtx* DrawAxisCtx;
typedef struct _DrawLGCtx*   DrawLGCtx;

extern int DrawAxisCreate(DrawCtx,DrawAxisCtx *);
extern int DrawAxisDestroy(DrawAxisCtx);
extern int DrawAxis(DrawAxisCtx);
extern int DrawAxisSetLimits(DrawAxisCtx,double,double,double,double);
extern int DrawAxisSetColors(DrawAxisCtx,int,int,int);
extern int DrawAxisSetLabels(DrawAxisCtx,char*,char*,char*);

#define LG_COOKIE PETSC_COOKIE+7
extern int DrawLGCreate(DrawCtx,int,DrawLGCtx *);
extern int DrawLGDestroy(DrawLGCtx);
extern int DrawLGAddPoint(DrawLGCtx,double*,double*);
extern int DrawLGAddPoints(DrawLGCtx,int,double**,double**);
extern int DrawLG(DrawLGCtx);
extern int DrawLGReset(DrawLGCtx);
extern int DrawLGGetAxisCtx(DrawLGCtx,DrawAxisCtx *);
extern int DrawLGGetDrawCtx(DrawLGCtx,DrawCtx *);

int DrawTensorContour(DrawCtx,int,int,double*,double*,Vec);

#endif
