#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: zdraw.c,v 1.27 1999/04/06 18:13:32 balay Exp balay $";
#endif

#include "src/fortran/custom/zpetsc.h"

#ifdef HAVE_FORTRAN_CAPS
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
#elif !defined(HAVE_FORTRAN_UNDERSCORE)
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

void viewerdrawgetdraw_(Viewer *vin,int *win,Draw *draw, int *__ierr )
{
  Viewer v;
  PetscPatchDefaultViewers_Fortran(vin,v);
  *__ierr = ViewerDrawGetDraw(v,*win,draw);
}

void viewerdrawgetdrawlg_(Viewer *vin,int *win,DrawLG *drawlg, int *__ierr )
{
  Viewer v;
  PetscPatchDefaultViewers_Fortran(vin,v);
  *__ierr = ViewerDrawGetDrawLG(v,*win,drawlg);
}

void drawstring_(Draw *ctx,double* xl,double* yl,int* cl,CHAR text,
               int *__ierr, int len){
  char *t;
  FIXCHAR(text,len,t);
  *__ierr = DrawString(*ctx,*xl,*yl,*cl,t);
  FREECHAR(text,t);
}
void drawstringvertical_(Draw *ctx,double *xl,double *yl,int *cl,CHAR text, 
                       int *__ierr,int len )
{
  char *t;
  FIXCHAR(text,len,t);
  *__ierr = DrawStringVertical(*ctx,*xl,*yl,*cl,t);
  FREECHAR(text,t);
}

void drawdestroy_(Draw *ctx, int *__ierr )
{
  *__ierr = DrawDestroy(*ctx);
}

void drawopenx_(MPI_Comm *comm,CHAR display,CHAR title,int *x,int *y,
                int *w,int *h,Draw* inctx, int *__ierr,int len1,int len2 )
{
  char *t1,*t2;

  FIXCHAR(display,len1,t1);
  FIXCHAR(title,len2,t2);
  *__ierr = DrawOpenX((MPI_Comm)PetscToPointerComm( *comm),t1,t2,*x,*y,*w,*h,inctx);
  FREECHAR(display,t1);
  FREECHAR(title,t2);
}

void drawlggetaxis_(DrawLG *lg,DrawAxis *axis, int *__ierr )
{
  *__ierr = DrawLGGetAxis(*lg,axis);
}

void drawlggetdraw_(DrawLG *lg,Draw *win, int *__ierr )
{
  *__ierr = DrawLGGetDraw(*lg,win);
}

void drawlgdestroy_(DrawLG *lg, int *__ierr )
{
  *__ierr = DrawLGDestroy(*lg);
}

void drawlgcreate_(Draw *win,int *dim,DrawLG *outctx, int *__ierr )
{
  *__ierr = DrawLGCreate(*win,*dim,outctx);
}

void drawaxissetlabels_(DrawAxis *axis,CHAR top,CHAR xlabel,CHAR ylabel,
                        int *__ierr,int len1,int len2,int len3 )
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

void drawaxisdestroy_(DrawAxis *axis, int *__ierr )
{
  *__ierr = DrawAxisDestroy(*axis);
}

void drawaxiscreate_(Draw *win,DrawAxis *ctx, int *__ierr )
{
  *__ierr = DrawAxisCreate(*win,ctx);
}

void drawgettitle_(Draw *draw,CHAR title, int *__ierr,int len )
{
  char *c3,*t;
  int  len3;
#if defined(USES_CPTOFCD)
    c3   = _fcdtocp(title);
    len3 = _fcdlen(title) - 1;
#else
    c3   = title;
    len3 = len - 1;
#endif
  *__ierr = DrawGetTitle(*draw,&t);
  *__ierr = PetscStrncpy(c3,t,len3);
}

void drawsettitle_(Draw *draw,CHAR title, int *__ierr,int len )
{
  char *t1;
  FIXCHAR(title,len,t1);
  *__ierr = DrawSetTitle(*draw,t1);
  FREECHAR(title,t1);
}

void drawappendtitle_(Draw *draw,CHAR title, int *__ierr,int len )
{
  char *t1;
  FIXCHAR(title,len,t1);
  *__ierr = DrawAppendTitle(*draw,t1);
  FREECHAR(title,t1);
}

void drawgetpopup_(Draw *draw,Draw *popup, int *__ierr )
{
  *__ierr = DrawGetPopup(*draw,popup);
}

EXTERN_C_END














