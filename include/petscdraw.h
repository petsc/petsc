/*
  Public include file for all of the PETSc graphics routines
*/
#if !defined(__DRAW_H)
#define __DRAW_H

typedef struct _DrawCtx* DrawCtx;

int DrawOpenX(char*,char *,int,int,int,int,DrawCtx*);
int DrawDestroy(DrawCtx);

int DrawLine(DrawCtx,double,double,double,double,int,int);
int DrawLineSetWidth(DrawCtx,double,double,double,double,int,int);

int DrawPoint(DrawCtx,double,double,int);

int DrawText(DrawCtx,double,double,int,char*);

int DrawSetViewPort(DrawCtx,double,double,double,double);
int DrawSetCoordinates(DrawCtx,double,double,double,double);
int DrawSetDoubleBuffer(DrawCtx);
int DrawFlush(DrawCtx);

#endif
