/*$Id: zdraw.c,v 1.37 2000/09/26 19:11:19 balay Exp bsmith $*/

#include "src/fortran/custom/zpetsc.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define drawsettype_         DRAWSETTYPE
#define drawcreate_          DRAWCREATE
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
#define viewerdrawgetdraw_   PETSC_VIEWERDRAWGETDRAW
#define viewerdrawgetdrawlg_ PETSC_VIEWERDRAWGETDRAWLG
#define drawgettitle_        DRAWGETTITLE
#define drawsettitle_        DRAWSETTITLE
#define drawappendtitle_     DRAWAPPENDTITLE
#define drawgetpopup_        DRAWGETPOPUP
#define drawzoom_            DRAWZOOM
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define drawcreate_          drawcreate
#define drawsettype_         drawsettype
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

static void (PETSC_STDCALL *f1)(PetscDraw *,void *,int *);
static int ourdrawzoom(PetscDraw draw,void *ctx)
{
  int ierr = 0;

  (*f1)(&draw,ctx,&ierr);CHKERRQ(ierr);
  return 0;
}

void PETSC_STDCALL drawzoom_(PetscDraw *draw,void (PETSC_STDCALL *f)(PetscDraw *,void *,int *),void *ctx,int *ierr)
{
  f1      = f;
  *ierr = PetscDrawZoom(*draw,ourdrawzoom,ctx);
}

void PETSC_STDCALL viewerdrawgetdraw_(PetscViewer *vin,int *win,PetscDraw *draw,int *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(vin,v);
  *ierr = PetscViewerDrawGetDraw(v,*win,draw);
}

void PETSC_STDCALL viewerdrawgetdrawlg_(PetscViewer *vin,int *win,PetscDrawLG *drawlg,int *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(vin,v);
  *ierr = PetscViewerDrawGetDrawLG(v,*win,drawlg);
}

void PETSC_STDCALL drawsettype_(PetscDraw *ctx,CHAR text PETSC_MIXED_LEN(len),
               int *ierr PETSC_END_LEN(len)){
  char *t;
  FIXCHAR(text,len,t);
  *ierr = PetscDrawSetType(*ctx,t);
  FREECHAR(text,t);
}

void PETSC_STDCALL drawstring_(PetscDraw *ctx,double* xl,double* yl,int* cl,CHAR text PETSC_MIXED_LEN(len),
               int *ierr PETSC_END_LEN(len)){
  char *t;
  FIXCHAR(text,len,t);
  *ierr = PetscDrawString(*ctx,*xl,*yl,*cl,t);
  FREECHAR(text,t);
}
void PETSC_STDCALL drawstringvertical_(PetscDraw *ctx,double *xl,double *yl,int *cl,
                   CHAR text PETSC_MIXED_LEN(len),int *ierr PETSC_END_LEN(len))
{
  char *t;
  FIXCHAR(text,len,t);
  *ierr = PetscDrawStringVertical(*ctx,*xl,*yl,*cl,t);
  FREECHAR(text,t);
}

void PETSC_STDCALL drawdestroy_(PetscDraw *ctx,int *ierr)
{
  *ierr = PetscDrawDestroy(*ctx);
}

void PETSC_STDCALL drawcreate_(MPI_Comm *comm,CHAR display PETSC_MIXED_LEN(len1),
                    CHAR title PETSC_MIXED_LEN(len2),int *x,int *y,int *w,int *h,PetscDraw* inctx,
                    int *ierr PETSC_END_LEN(len1) PETSC_END_LEN(len2))
{
  char *t1,*t2;

  FIXCHAR(display,len1,t1);
  FIXCHAR(title,len2,t2);
  *ierr = PetscDrawCreate((MPI_Comm)PetscToPointerComm(*comm),t1,t2,*x,*y,*w,*h,inctx);
  FREECHAR(display,t1);
  FREECHAR(title,t2);
}

#if defined(PETSC_HAVE_X11)
void PETSC_STDCALL drawopenx_(MPI_Comm *comm,CHAR display PETSC_MIXED_LEN(len1),
                    CHAR title PETSC_MIXED_LEN(len2),int *x,int *y,int *w,int *h,PetscDraw* inctx,
                    int *ierr PETSC_END_LEN(len1) PETSC_END_LEN(len2))
{
  char *t1,*t2;

  FIXCHAR(display,len1,t1);
  FIXCHAR(title,len2,t2);
  *ierr = PetscDrawOpenX((MPI_Comm)PetscToPointerComm(*comm),t1,t2,*x,*y,*w,*h,inctx);
  FREECHAR(display,t1);
  FREECHAR(title,t2);
}
#endif

void PETSC_STDCALL drawlggetaxis_(PetscDrawLG *lg,PetscDrawAxis *axis,int *ierr)
{
  *ierr = PetscDrawLGGetAxis(*lg,axis);
}

void PETSC_STDCALL drawlggetdraw_(PetscDrawLG *lg,PetscDraw *win,int *ierr)
{
  *ierr = PetscDrawLGGetDraw(*lg,win);
}

void PETSC_STDCALL drawlgdestroy_(PetscDrawLG *lg,int *ierr)
{
  *ierr = PetscDrawLGDestroy(*lg);
}

void PETSC_STDCALL drawlgcreate_(PetscDraw *win,int *dim,PetscDrawLG *outctx,int *ierr)
{
  *ierr = PetscDrawLGCreate(*win,*dim,outctx);
}

void PETSC_STDCALL drawaxissetlabels_(PetscDrawAxis *axis,CHAR top PETSC_MIXED_LEN(len1),
                    CHAR xlabel PETSC_MIXED_LEN(len2),CHAR ylabel PETSC_MIXED_LEN(len3),
                    int *ierr PETSC_END_LEN(len1) PETSC_END_LEN(len2) PETSC_END_LEN(len3))
{
  char *t1,*t2,*t3;
 
  FIXCHAR(top,len1,t1);
  FIXCHAR(xlabel,len2,t2);
  FIXCHAR(ylabel,len3,t3);
  *ierr = PetscDrawAxisSetLabels(*axis,t1,t2,t3);
  FREECHAR(top,t1);
  FREECHAR(xlabel,t2);
  FREECHAR(ylabel,t3);
}

void PETSC_STDCALL drawaxisdestroy_(PetscDrawAxis *axis,int *ierr)
{
  *ierr = PetscDrawAxisDestroy(*axis);
}

void PETSC_STDCALL drawaxiscreate_(PetscDraw *win,PetscDrawAxis *ctx,int *ierr)
{
  *ierr = PetscDrawAxisCreate(*win,ctx);
}

void PETSC_STDCALL drawgettitle_(PetscDraw *draw,CHAR title PETSC_MIXED_LEN(len),
                                 int *ierr PETSC_END_LEN(len))
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
  *ierr = PetscDrawGetTitle(*draw,&t);
  *ierr = PetscStrncpy(c3,t,len3);
}

void PETSC_STDCALL drawsettitle_(PetscDraw *draw,CHAR title PETSC_MIXED_LEN(len),
                                 int *ierr PETSC_END_LEN(len))
{
  char *t1;
  FIXCHAR(title,len,t1);
  *ierr = PetscDrawSetTitle(*draw,t1);
  FREECHAR(title,t1);
}

void PETSC_STDCALL drawappendtitle_(PetscDraw *draw,CHAR title PETSC_MIXED_LEN(len),
                                    int *ierr PETSC_END_LEN(len))
{
  char *t1;
  FIXCHAR(title,len,t1);
  *ierr = PetscDrawAppendTitle(*draw,t1);
  FREECHAR(title,t1);
}

void PETSC_STDCALL drawgetpopup_(PetscDraw *draw,PetscDraw *popup,int *ierr)
{
  *ierr = PetscDrawGetPopup(*draw,popup);
}

EXTERN_C_END














