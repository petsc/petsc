/*
  Public include file for all of the PETSc graphics routines
*/
#if !defined(__DRAW_H)
#define __DRAW_H

typedef struct _DrawCtx* DrawCtx;

#define DRAW_WHITE  0
#define DRAW_BLACK  1
#define DRAW_RED    2
#define DRAW_YELLOW 3
#define DRAW_GREEN  4
#define DRAW_CYAN   5
#define DRAW_BLUE   6

int DrawOpenX(char*,char *,int,int,int,int,DrawCtx*);
int DrawDestroy(DrawCtx);

int DrawLine(DrawCtx,double,double,double,double,int,int);
int DrawLineSetWidth(DrawCtx,double);

int DrawPoint(DrawCtx,double,double,int);
int DrawPointSetSize(DrawCtx,double);

int DrawText(DrawCtx,double,double,int,char*);
int DrawTextVertical(DrawCtx,double,double,int,char*);
int DrawTextSetSize(DrawCtx,double,double);
int DrawTextGetSize(DrawCtx,double*,double*);

int DrawSetViewPort(DrawCtx,double,double,double,double);
int DrawSetCoordinates(DrawCtx,double,double,double,double);
int DrawSetDoubleBuffer(DrawCtx);
int DrawFlush(DrawCtx);
int DrawClear(DrawCtx);

/* routines related to drawing Axis and line graphs */

typedef struct _DrawAxisCtx* DrawAxisCtx;
typedef struct _DrawLGCtx*   DrawLGCtx;

int DrawAxisCreate(DrawCtx,DrawAxisCtx *);
int DrawAxisDestroy(DrawAxisCtx);
int DrawAxis(DrawAxisCtx);
int DrawAxisSetLimits(DrawAxisCtx,double,double,double,double);
int DrawAxisSetColors(DrawAxisCtx,int,int,int);
int DrawAxisSetLabels(DrawAxisCtx,char*,char*,char*);

int DrawLGCreate(DrawCtx,int,DrawLGCtx *);
int DrawLGDestroy(DrawLGCtx);
int DrawLGAddPoint(DrawLGCtx,double*,double*);
int DrawLG(DrawLGCtx);
int DrawLGGetAxisCtx(DrawLGCtx,DrawAxisCtx *);

#endif
