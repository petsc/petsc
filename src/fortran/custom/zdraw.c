/*$Id: zdraw.c,v 1.32 1999/10/24 14:04:19 bsmith Exp bsmith $*/

#include "src/fortran/custom/zpetsc.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define drawaxisdestroy_     DRAWAXISDESTROY
#define drawaxiscreate_      DRAWAXISCREATE
#define drawaxissetlabels_   DRAWAXISSETLABELS
#define drawlgcreate_        DRAWLGCREATE
#define drawlgdestroy_       DRAWLGDESTROY
#define drawlggetaxis_       DRAWLGGETAXIS
#define drawlggetdraw_       DRAWLGGETDRAW
#define drawopenx_           DRAWOPENX
#define drawstring_          DRAWSTRING
#define drawstringvertical_  DRAWSTRINGVERTICAL
#define drawdestroy_         DRAWDESTROY
#define viewerdrawgetdraw_   VIEWERDRAWGETDRAW
#define viewerdrawgetdrawlg_ VIEWERDRAWGETDRAWLG
#define drawgettitle_        DRAWGETTITLE
#define drawsettitle_        DRAWSETTITLE
#define drawappendtitle_     DRAWAPPENDTITLE
#define drawgetpopup_        DRAWGETPOPUP
#define drawzoom_            DRAWZOOM
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define drawzoom_            drawzoom
#define drawaxisdestroy_     drawaxisdestroy
#define drawaxiscreate_      drawaxiscreate
#define drawaxissetlabels_   drawaxissetlabels
#define drawlgcreate_        drawlgcreate
#define drawlgdestroy_       drawlgdestroy
#define drawlggetaxis_       drawlggetaxis
#define drawlggetdraw_       drawlggetdraw
#define drawopenx_           drawopenx
#define drawstring_          drawstring
#define drawstringvertical_  drawstringvertical
#define drawdestroy_         drawdestroy
#define viewerdrawgetdraw_   viewerdrawgetdraw
#define viewerdrawgetdrawlg_ viewerdrawgetdrawlg
#define drawgettitle_        drawgettitle
#define drawsettitle_        drawsettitle
#define drawappendtitle_     drawappendtitle
#define drawgetpopup_        drawgetpopup
#endif

EXTERN_C_BEGIN

static void (*f1)(Draw *,void *,int *);
static int ourdrawzoom(Draw draw,void *ctx)
{
  int ierr = 0;

  (*f1)(&draw,ctx,&ierr);CHKERRQ(ierr);
  return 0;
}

void PETSC_STDCALL drawzoom_(Draw *draw,void (*f)(Draw *,void *,int *),void *ctx,int *__ierr)
{
  f1      = f;
  *__ierr = DrawZoom(*draw,ourdrawzoom,ctx);
}

void PETSC_STDCALL viewerdrawgetdraw_(Viewer *vin,int *win,Draw *draw, int *__ierr )
{
  Viewer v;
  PetscPatchDefaultViewers_Fortran(vin,v);
  *__ierr = ViewerDrawGetDraw(v,*win,draw);
}

void PETSC_STDCALL viewerdrawgetdrawlg_(Viewer *vin,int *win,DrawLG *drawlg, int *__ierr )
{
  Viewer v;
  PetscPatchDefaultViewers_Fortran(vin,v);
  *__ierr = ViewerDrawGetDrawLG(v,*win,drawlg);
}

void PETSC_STDCALL drawstring_(Draw *ctx,double* xl,double* yl,int* cl,CHAR text PETSC_MIXED_LEN(len),
               int *__ierr PETSC_END_LEN(len)){
  char *t;
  FIXCHAR(text,len,t);
  *__ierr = DrawString(*ctx,*xl,*yl,*cl,t);
  FREECHAR(text,t);
}
void PETSC_STDCALL drawstringvertical_(Draw *ctx,double *xl,double *yl,int *cl,
                   CHAR text PETSC_MIXED_LEN(len),int *__ierr PETSC_END_LEN(len))
{
  char *t;
  FIXCHAR(text,len,t);
  *__ierr = DrawStringVertical(*ctx,*xl,*yl,*cl,t);
  FREECHAR(text,t);
}

void PETSC_STDCALL drawdestroy_(Draw *ctx, int *__ierr )
{
  *__ierr = DrawDestroy(*ctx);
}

void PETSC_STDCALL drawopenx_(MPI_Comm *comm,CHAR display PETSC_MIXED_LEN(len1),
                    CHAR title PETSC_MIXED_LEN(len2),int *x,int *y,int *w,int *h,Draw* inctx, 
                    int *__ierr PETSC_END_LEN(len1) PETSC_END_LEN(len2))
{
  char *t1,*t2;

  FIXCHAR(display,len1,t1);
  FIXCHAR(title,len2,t2);
  *__ierr = DrawOpenX((MPI_Comm)PetscToPointerComm( *comm),t1,t2,*x,*y,*w,*h,inctx);
  FREECHAR(display,t1);
  FREECHAR(title,t2);
}

void PETSC_STDCALL drawlggetaxis_(DrawLG *lg,DrawAxis *axis, int *__ierr )
{
  *__ierr = DrawLGGetAxis(*lg,axis);
}

void PETSC_STDCALL drawlggetdraw_(DrawLG *lg,Draw *win, int *__ierr )
{
  *__ierr = DrawLGGetDraw(*lg,win);
}

void PETSC_STDCALL drawlgdestroy_(DrawLG *lg, int *__ierr )
{
  *__ierr = DrawLGDestroy(*lg);
}

void PETSC_STDCALL drawlgcreate_(Draw *win,int *dim,DrawLG *outctx, int *__ierr )
{
  *__ierr = DrawLGCreate(*win,*dim,outctx);
}

void PETSC_STDCALL drawaxissetlabels_(DrawAxis *axis,CHAR top PETSC_MIXED_LEN(len1),
                    CHAR xlabel PETSC_MIXED_LEN(len2),CHAR ylabel PETSC_MIXED_LEN(len3),
                    int *__ierr PETSC_END_LEN(len1) PETSC_END_LEN(len2) PETSC_END_LEN(len3))
{
  char *t1,*t2,*t3;
 
  FIXCHAR(top,len1,t1);
  FIXCHAR(xlabel,len2,t2);
  FIXCHAR(ylabel,len3,t3);
  *__ierr = DrawAxisSetLabels(*axis,t1,t2,t3);
  FREECHAR(top,t1);
  FREECHAR(xlabel,t2);
  FREECHAR(ylabel,t3);
}

void PETSC_STDCALL drawaxisdestroy_(DrawAxis *axis, int *__ierr )
{
  *__ierr = DrawAxisDestroy(*axis);
}

void PETSC_STDCALL drawaxiscreate_(Draw *win,DrawAxis *ctx, int *__ierr )
{
  *__ierr = DrawAxisCreate(*win,ctx);
}

void PETSC_STDCALL drawgettitle_(Draw *draw,CHAR title PETSC_MIXED_LEN(len),
                                 int *__ierr PETSC_END_LEN(len) )
{
  char *c3,*t;
  int  len3;
#if defined(PETSC_USES_CPTOFCD)
    c3   = _fcdtocp(title);
    len3 = _fcdlen(title) - 1;
#else
    c3   = title;
    len3 = len - 1;
#endif
  *__ierr = DrawGetTitle(*draw,&t);
  *__ierr = PetscStrncpy(c3,t,len3);
}

void PETSC_STDCALL drawsettitle_(Draw *draw,CHAR title PETSC_MIXED_LEN(len),
                                 int *__ierr PETSC_END_LEN(len) )
{
  char *t1;
  FIXCHAR(title,len,t1);
  *__ierr = DrawSetTitle(*draw,t1);
  FREECHAR(title,t1);
}

void PETSC_STDCALL drawappendtitle_(Draw *draw,CHAR title PETSC_MIXED_LEN(len), 
                                    int *__ierr PETSC_END_LEN(len) )
{
  char *t1;
  FIXCHAR(title,len,t1);
  *__ierr = DrawAppendTitle(*draw,t1);
  FREECHAR(title,t1);
}

void PETSC_STDCALL drawgetpopup_(Draw *draw,Draw *popup, int *__ierr )
{
  *__ierr = DrawGetPopup(*draw,popup);
}

EXTERN_C_END














