
#ifndef lint
static char vcid[] = "$Id: zdraw.c,v 1.5 1995/11/23 04:15:38 bsmith Exp bsmith $";
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

void drawtext_(Draw ctx,double* xl,double* yl,int* cl,char* text,
               int *__ierr, int len){
  char *t;
  if (text[len] != 0) {
    t = (char *) PetscMalloc( (len+1)*sizeof(char) ); 
    PetscStrncpy(t,text,len);
    t[len] = 0;
  }
  else t = text;
  *__ierr = DrawText(
	(Draw)MPIR_ToPointer( *(int*)(ctx) ),*xl,*yl,*cl,t);
  if (t != text) PetscFree(t);
}
void drawtextvertical_(Draw ctx,double *xl,double *yl,int *cl,char *text, 
                       int *__ierr,int len ){
  char *t;
  if (text[len] != 0) {
    t = (char *) PetscMalloc( (len+1)*sizeof(char) ); 
    PetscStrncpy(t,text,len);
    t[len] = 0;
  }
  else t = text;
  *__ierr = DrawTextVertical(
	(Draw)MPIR_ToPointer( *(int*)(ctx) ),*xl,*yl,*cl,t);
  if (t != text) PetscFree(t);
}

void drawdestroy_(Draw ctx, int *__ierr ){
  *__ierr = DrawDestroy((Draw)MPIR_ToPointer( *(int*)(ctx) ));
  MPIR_RmPointer(*(int*)(ctx) );
}

void drawopenx_(MPI_Comm comm,char* display,char *title,int *x,int *y,
                int *w,int *h,Draw* inctx, int *__ierr,int len1,int len2 )
{
  Draw a;
  char    *t1,*t2;
  if (display == PETSC_NULL_Fortran) {
    t1 = 0; display = 0; len2 = len1;
  }
  else {
    if (display[len1] != 0) {
      t1 = (char *) PetscMalloc( (len1+1)*sizeof(char) ); 
      PetscStrncpy(t1,display,len1);
      t1[len1] = 0;
    }
    else t1 = display;
  }
  if (title == PETSC_NULL_Fortran) {title = 0; t2 = 0;}
  else {
    if (title[len2] != 0) {
      t2 = (char *) PetscMalloc( (len2+1)*sizeof(char) ); 
      PetscStrncpy(t2,title,len2);
      t2[len2] = 0;
    }
    else t2 = title;
  }  
  *__ierr = DrawOpenX((MPI_Comm)MPIR_ToPointer( *(int*)(comm)),t1,t2,
                       *x,*y,*w,*h,&a);
  *(int*)inctx = MPIR_FromPointer(a);
  if (t1 != display) PetscFree(t1);
  if (t2 != title) PetscFree(t2);
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

void drawaxissetlabels_(DrawAxis axis,char* top,char *xlabel,char *ylabel,
                        int *__ierr,int len1,int len2,int len3 )
{
  char *t1,*t2,*t3;
  if (top[len1] != 0) {
    t1 = (char *) PetscMalloc((len1+1)*sizeof(char)); if (!t1) *__ierr = 1;
    PetscStrncpy(t1,top,len1);
    t1[len1] = 0;
  }
  else t1 = top;
  if (xlabel[len2] != 0) {
    t2 = (char *) PetscMalloc((len2+1)*sizeof(char));if (!t2) *__ierr = 1; 
    PetscStrncpy(t2,xlabel,len2);
    t2[len2] = 0;
  }
  else t2 = xlabel;
  if (ylabel[len3] != 0) {
    t3 = (char *) PetscMalloc((len3+1)*sizeof(char));if (!t3) *__ierr = 1; 
    PetscStrncpy(t3,ylabel,len3);
    t3[len3] = 0;
  }
  else t3 = ylabel;

  *__ierr = DrawAxisSetLabels(
	 (DrawAxis)MPIR_ToPointer( *(int*)(axis) ),t1,t2,t3);
  if (t1 != top) PetscFree(t1);
  if (t2 != xlabel) PetscFree(t2);
  if (t3 != ylabel) PetscFree(t3);
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
