/*
  Public include file for all of the PETSc graphics routines
*/
#if !defined(__DRAW_H)
#define __DRAW_H

typedef struct _DrawCtx* DrawCtx;

int DrawOpenX(char*,int,int,int,int,DrawCtx*);
int DrawDestroy(DrawCtx);

int Draw2dLine(DrawCtx,double,double,double,double,int,int);
int Draw2dLineSetWidth(DrawCtx,double,double,double,double,int,int);
int Draw2dPoint(DrawCtx,double,double,int);

int Draw2dText(DrawCtx,double,double,int,char*);

#endif
