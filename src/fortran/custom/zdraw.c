
#ifndef lint
static char vcid[] = "$Id: zdraw.c,v 1.6 1996/01/30 00:40:19 bsmith Exp bsmith $";
#endif

#include "zpetsc.h"
#include "draw.h"
#include "pinclude/petscfix.h"

#ifdef HAVE_FORTRAN_CAPS
#define drawaxisdestroy_   DRAWAXISDESTROY
#define drawaxiscreate_    DRAWAXISCREATE
#define drawaxissetlabels_ DRAWAXISSETLABELS
#define drawlgcreate_      DRAWLGCREATE
#define drawlgdestroy_     DRAWLGDESTROY
#define drawlggetaxis_     DRAWLGGETAXIS
#define drawlggetdraw_     DRAWLGGETDRAW
#define drawopenx_         DRAWOPENX
#define drawtext_          DRAWTEXT
#define drawtextvertical_  DRAWTEXTVERTICAL
#define drawdestroy_       DRAWDESTROY
#elif !defined(HAVE_FORTRAN_UNDERSCORE)
#define drawaxisdestroy_   drawaxisdestroy
#define drawaxiscreate_    drawaxiscreate
#define drawaxissetlabels_ drawaxissetlabels
#define drawlgcreate_      drawlgcreate
#define drawlgdestroy_     drawlgdestroy
#define drawlggetaxis_     drawlggetaxis
#define drawlggetdraw_     drawlggetdraw
#define drawopenx_         drawopenx
#define drawtext_          drawtext
#define drawtextvertical_  drawtextvertical
#define drawdestroy_       drawdestroy
#endif

#if defined(__cplusplus)
extern "C" {
#endif

void drawtext_(Draw ctx,double* xl,double* yl,int* cl,CHAR text,
               int *__ierr, int len){
  char *t;
  FIXCHAR(text,len,t);
  *__ierr = DrawText((Draw)MPIR_ToPointer( *(int*)(ctx) ),*xl,*yl,*cl,t);
  FREECHAR(text,t);
}
void drawtextvertical_(Draw ctx,double *xl,double *yl,int *cl,CHAR text, 
                       int *__ierr,int len ){
  char *t;
  FIXCHAR(text,len,t);
  *__ierr = DrawTextVertical(
	(Draw)MPIR_ToPointer( *(int*)(ctx) ),*xl,*yl,*cl,t);
  FREECHAR(text,t);
}

void drawdestroy_(Draw ctx, int *__ierr ){
  *__ierr = DrawDestroy((Draw)MPIR_ToPointer( *(int*)(ctx) ));
  MPIR_RmPointer(*(int*)(ctx) );
}

void drawopenx_(MPI_Comm comm,CHAR display,CHAR title,int *x,int *y,
                int *w,int *h,Draw* inctx, int *__ierr,int len1,int len2 )
{
  Draw a;
  char *t1,*t2;

  FIXCHAR(display,len1,t1);
  FIXCHAR(title,len2,t2);
  *__ierr = DrawOpenX((MPI_Comm)MPIR_ToPointer_Comm( *(int*)(comm)),t1,t2,
                       *x,*y,*w,*h,&a);
  *(int*)inctx = MPIR_FromPointer(a);
  FREECHAR(display,t1);
  FREECHAR(title,t2);
}

void drawlggetaxis_(DrawLG lg,DrawAxis *axis, int *__ierr )
{
  DrawAxis a;
  *__ierr = DrawLGGetAxis(
	(DrawLG)MPIR_ToPointer( *(int*)(lg) ),&a);
  *(int*)axis = MPIR_FromPointer(a);
}
void drawlggetDraw_(DrawLG lg,Draw *win, int *__ierr )
{
  Draw a;
  *__ierr = DrawLGGetDraw(
	(DrawLG)MPIR_ToPointer( *(int*)(lg) ),&a);
  *(int*)win = MPIR_FromPointer(a);
}

void drawlgdestroy_(DrawLG lg, int *__ierr )
{
  *__ierr = DrawLGDestroy((DrawLG)MPIR_ToPointer( *(int*)(lg) ));
  MPIR_RmPointer(*(int*)(lg));
}

void drawlgcreate_(Draw win,int *dim,DrawLG *outctx, int *__ierr )
{
  DrawLG lg;
  *__ierr = DrawLGCreate(
	(Draw)MPIR_ToPointer( *(int*)(win) ),*dim,&lg);
  *(int*)outctx = MPIR_FromPointer(lg);
}

void drawaxissetlabels_(DrawAxis axis,CHAR top,CHAR xlabel,CHAR ylabel,
                        int *__ierr,int len1,int len2,int len3 )
{
  char *t1,*t2,*t3;
 
  FIXCHAR(top,len1,t1);
  FIXCHAR(xlabel,len2,t2);
  FIXCHAR(ylabel,len3,t3);
  *__ierr = DrawAxisSetLabels(
	 (DrawAxis)MPIR_ToPointer( *(int*)(axis) ),t1,t2,t3);
  FREECHAR(top,t1);
  FREECHAR(xlabel,t2);
  FREECHAR(ylabel,t3);
}

void drawaxisdestroy_(DrawAxis axis, int *__ierr )
{
  *__ierr = DrawAxisDestroy((DrawAxis)MPIR_ToPointer(*(int*)(axis)));
  MPIR_RmPointer(*(int*)(axis));
}

void drawaxiscreate_(Draw win,DrawAxis *ctx, int *__ierr )
{
  DrawAxis tmp;
  *__ierr = DrawAxisCreate((Draw)MPIR_ToPointer( *(int*)(win) ),&tmp);
  *(int*)ctx = MPIR_FromPointer(tmp);
}

#if defined(__cplusplus)
}
#endif














