
#ifndef lint
static char vcid[] = "$Id: zdraw.c,v 1.2 1995/09/04 17:18:58 bsmith Exp bsmith $";
#endif

#include "zpetsc.h"
#include "draw.h"
#include "pinclude/petscfix.h"

#ifdef FORTRANCAPS
#define drawaxisdestroy_   DRAWAXISDESTROY
#define drawaxiscreate_    DRAWAXISCREATE
#define drawaxissetlabels_ DRAWAXISSETLABELS
#define drawlgcreate_      DRAWLGCREATE
#define drawlgdestroy_     DRAWLGDESTROY
#define drawlggetaxisctx_  DRAWLGGETAXISCTX
#define drawlggetdrawctx_  DRAWLGGETDRAWCTX
#define drawopenx_         DRAWOPENX
#define drawtext_          DRAWTEXT
#define drawtextvertical_  DRAWTEXTVERTICAL
#define drawdestroy_       DRAWDESTROY
#elif !defined(FORTRANUNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define drawaxisdestroy_   drawaxisdestroy
#define drawaxiscreate_    drawaxiscreate
#define drawaxissetlabels_ drawaxissetlabels
#define drawlgcreate_      drawlgcreate
#define drawlgdestroy_     drawlgdestroy
#define drawlggetaxisctx_  drawlggetaxisctx
#define drawlggetdrawctx_  drawlggetdrawctx
#define drawopenx_         drawopenx
#define drawtext_          drawtext
#define drawtextvertical_  drawtextvertical
#define drawdestroy_       drawdestroy
#endif

void drawtext_(DrawCtx ctx,double* xl,double* yl,int* cl,char* text,
               int *__ierr, int len){
  char *t;
  if (text[len] != 0) {
    t = (char *) PETSCMALLOC( (len+1)*sizeof(char) ); 
    PetscStrncpy(t,text,len);
    t[len] = 0;
  }
  else t = text;
  *__ierr = DrawText(
	(DrawCtx)MPIR_ToPointer( *(int*)(ctx) ),*xl,*yl,*cl,t);
  if (t != text) PETSCFREE(t);
}
void drawtextvertical_(DrawCtx ctx,double *xl,double *yl,int *cl,char *text, 
                       int *__ierr,int len ){
  char *t;
  if (text[len] != 0) {
    t = (char *) PETSCMALLOC( (len+1)*sizeof(char) ); 
    PetscStrncpy(t,text,len);
    t[len] = 0;
  }
  else t = text;
  *__ierr = DrawTextVertical(
	(DrawCtx)MPIR_ToPointer( *(int*)(ctx) ),*xl,*yl,*cl,t);
  if (t != text) PETSCFREE(t);
}

void drawdestroy_(DrawCtx ctx, int *__ierr ){
  *__ierr = DrawDestroy((DrawCtx)MPIR_ToPointer( *(int*)(ctx) ));
  MPIR_RmPointer(*(int*)(ctx) );
}

void drawopenx_(MPI_Comm comm,char* display,char *title,int *x,int *y,
                int *w,int *h,DrawCtx* inctx, int *__ierr,int len1,int len2 )
{
  DrawCtx a;
  char    *t1,*t2;
  if (display[0] == ' ') {t1 = 0; display = 0;}
  else {
    if (display[len1] != 0) {
      t1 = (char *) PETSCMALLOC( (len1+1)*sizeof(char) ); 
      PetscStrncpy(t1,display,len1);
      t1[len1] = 0;
    }
    else t1 = display;
  }
  if (title[0] == ' ')   {title = 0; t2 = 0;}
  else {
    if (title[len2] != 0) {
      t2 = (char *) PETSCMALLOC( (len2+1)*sizeof(char) ); 
      PetscStrncpy(t2,title,len2);
      t2[len2] = 0;
    }
    else t2 = display;
  }  
  *__ierr = DrawOpenX((MPI_Comm)MPIR_ToPointer( *(int*)(comm)),t1,t2,
                       *x,*y,*w,*h,&a);
  *(int*)inctx = MPIR_FromPointer(a);
  if (t1 != display) PETSCFREE(t1);
  if (t2 != title) PETSCFREE(t2);
}

void drawlggetaxisctx_(DrawLGCtx lg,DrawAxisCtx *axis, int *__ierr )
{
  DrawAxisCtx a;
  *__ierr = DrawLGGetAxisCtx(
	(DrawLGCtx)MPIR_ToPointer( *(int*)(lg) ),&a);
  *(int*)axis = MPIR_FromPointer(a);
}
void drawlggetdrawctx_(DrawLGCtx lg,DrawCtx *win, int *__ierr )
{
  DrawCtx a;
  *__ierr = DrawLGGetDrawCtx(
	(DrawLGCtx)MPIR_ToPointer( *(int*)(lg) ),&a);
  *(int*)win = MPIR_FromPointer(a);
}

void drawlgdestroy_(DrawLGCtx lg, int *__ierr )
{
  *__ierr = DrawLGDestroy((DrawLGCtx)MPIR_ToPointer( *(int*)(lg) ));
  MPIR_RmPointer(*(int*)(lg));
}

void drawlgcreate_(DrawCtx win,int *dim,DrawLGCtx *outctx, int *__ierr )
{
  DrawLGCtx lg;
  *__ierr = DrawLGCreate(
	(DrawCtx)MPIR_ToPointer( *(int*)(win) ),*dim,&lg);
  *(int*)outctx = MPIR_FromPointer(lg);
}

void drawaxissetlabels_(DrawAxisCtx axis,char* top,char *xlabel,char *ylabel,
                        int *__ierr,int len1,int len2,int len3 )
{
  char *t1,*t2,*t3;
  if (top[len1] != 0) {
    t1 = (char *) PETSCMALLOC((len1+1)*sizeof(char)); if (!t1) *__ierr = 1;
    PetscStrncpy(t1,top,len1);
    t1[len1] = 0;
  }
  else t1 = top;
  if (xlabel[len2] != 0) {
    t2 = (char *) PETSCMALLOC((len2+1)*sizeof(char));if (!t2) *__ierr = 1; 
    PetscStrncpy(t2,xlabel,len2);
    t2[len2] = 0;
  }
  else t2 = xlabel;
  if (ylabel[len3] != 0) {
    t3 = (char *) PETSCMALLOC((len3+1)*sizeof(char));if (!t3) *__ierr = 1; 
    PetscStrncpy(t3,ylabel,len3);
    t3[len3] = 0;
  }
  else t3 = ylabel;

  *__ierr = DrawAxisSetLabels(
	 (DrawAxisCtx)MPIR_ToPointer( *(int*)(axis) ),t1,t2,t3);
  if (t1 != top) PETSCFREE(t1);
  if (t2 != xlabel) PETSCFREE(t2);
  if (t3 != ylabel) PETSCFREE(t3);
}

void drawaxisdestroy_(DrawAxisCtx axis, int *__ierr )
{
  *__ierr = DrawAxisDestroy((DrawAxisCtx)MPIR_ToPointer(*(int*)(axis)));
  MPIR_RmPointer(*(int*)(axis));
}

void drawaxiscreate_(DrawCtx win,DrawAxisCtx *ctx, int *__ierr )
{
  DrawAxisCtx tmp;
  *__ierr = DrawAxisCreate((DrawCtx)MPIR_ToPointer( *(int*)(win) ),&tmp);
  *(int*)ctx = MPIR_FromPointer(tmp);
}
