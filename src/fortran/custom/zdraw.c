#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: zdraw.c,v 1.18 1997/11/28 16:17:05 bsmith Exp balay $";
#endif

#include "src/fortran/custom/zpetsc.h"
#include "vec.h"
#include "pinclude/petscfix.h"

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
#define drawtensorcontour_   DRAWTENSORCONTOUR
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
#define drawtensorcontour_   drawtensorcontour
#define drawgettitle_        drawgettitle
#define drawsettitle_        drawsettitle
#define drawappendtitle_     drawappendtitle
#define drawgetpopup_        drawgetpopup
#endif

#if defined(__cplusplus)
extern "C" {
#endif

void drawtensorcontour_(Draw win,int *m,int *n,double *x,double *y,Vec V, int *__ierr )
{
  double *xx,*yy;
  if ((void*)x == PETSC_NULL_Fortran) xx = PETSC_NULL; else xx = x;
  if ((void*)y == PETSC_NULL_Fortran) yy = PETSC_NULL; else yy = y;

  *__ierr = DrawTensorContour((Draw)PetscToPointer(win),*m,*n,xx,yy,
	                      (Vec)PetscToPointer(V));
}

void viewerdrawgetdraw_(Viewer v,Draw *draw, int *__ierr )
{
  Draw d;
  PetscPatchDefaultViewers_Fortran(v);
  *__ierr = ViewerDrawGetDraw(v,&d);
  *(PetscFortranAddr*) draw = PetscFromPointer(d);
}

void viewerdrawgetdrawlg_(Viewer v,DrawLG *drawlg, int *__ierr )
{
  DrawLG d;
  PetscPatchDefaultViewers_Fortran(v);
  *__ierr = ViewerDrawGetDrawLG(v,&d);
  *(PetscFortranAddr*) drawlg = PetscFromPointer(d);
}

void drawstring_(Draw ctx,double* xl,double* yl,int* cl,CHAR text,
               int *__ierr, int len){
  char *t;
  FIXCHAR(text,len,t);
  *__ierr = DrawString((Draw)PetscToPointer(ctx),*xl,*yl,*cl,t);
  FREECHAR(text,t);
}
void drawstringvertical_(Draw ctx,double *xl,double *yl,int *cl,CHAR text, 
                       int *__ierr,int len ){
  char *t;
  FIXCHAR(text,len,t);
  *__ierr = DrawStringVertical(
	(Draw)PetscToPointer(ctx),*xl,*yl,*cl,t);
  FREECHAR(text,t);
}

void drawdestroy_(Draw ctx, int *__ierr ){
  *__ierr = DrawDestroy((Draw)PetscToPointer(ctx));
  PetscRmPointer(ctx);
}

void drawopenx_(MPI_Comm *comm,CHAR display,CHAR title,int *x,int *y,
                int *w,int *h,Draw* inctx, int *__ierr,int len1,int len2 )
{
  Draw a;
  char *t1,*t2;

  FIXCHAR(display,len1,t1);
  FIXCHAR(title,len2,t2);
  *__ierr = DrawOpenX((MPI_Comm)PetscToPointerComm( *comm),t1,t2,
                       *x,*y,*w,*h,&a);
  *(PetscFortranAddr*)inctx = PetscFromPointer(a);
  FREECHAR(display,t1);
  FREECHAR(title,t2);
}

void drawlggetaxis_(DrawLG lg,DrawAxis *axis, int *__ierr )
{
  DrawAxis a;
  *__ierr = DrawLGGetAxis(
	(DrawLG)PetscToPointer(lg),&a);
  *(PetscFortranAddr*)axis = PetscFromPointer(a);
}
void drawlggetdraw_(DrawLG lg,Draw *win, int *__ierr )
{
  Draw a;
  *__ierr = DrawLGGetDraw(
	(DrawLG)PetscToPointer(lg),&a);
  *(PetscFortranAddr*)win = PetscFromPointer(a);
}

void drawlgdestroy_(DrawLG lg, int *__ierr )
{
  *__ierr = DrawLGDestroy((DrawLG)PetscToPointer(lg));
  PetscRmPointer(lg);
}

void drawlgcreate_(Draw win,int *dim,DrawLG *outctx, int *__ierr )
{
  DrawLG lg;
  *__ierr = DrawLGCreate(
	(Draw)PetscToPointer(win),*dim,&lg);
  *(PetscFortranAddr*)outctx = PetscFromPointer(lg);
}

void drawaxissetlabels_(DrawAxis axis,CHAR top,CHAR xlabel,CHAR ylabel,
                        int *__ierr,int len1,int len2,int len3 )
{
  char *t1,*t2,*t3;
 
  FIXCHAR(top,len1,t1);
  FIXCHAR(xlabel,len2,t2);
  FIXCHAR(ylabel,len3,t3);
  *__ierr = DrawAxisSetLabels(
	 (DrawAxis)PetscToPointer(axis),t1,t2,t3);
  FREECHAR(top,t1);
  FREECHAR(xlabel,t2);
  FREECHAR(ylabel,t3);
}

void drawaxisdestroy_(DrawAxis axis, int *__ierr )
{
  *__ierr = DrawAxisDestroy((DrawAxis)PetscToPointer(axis));
  PetscRmPointer(axis);
}

void drawaxiscreate_(Draw win,DrawAxis *ctx, int *__ierr )
{
  DrawAxis tmp;
  *__ierr = DrawAxisCreate((Draw)PetscToPointer(win),&tmp);
  *(PetscFortranAddr*)ctx = PetscFromPointer(tmp);
}

void drawgettitle_(Draw draw,CHAR title, int *__ierr,int len )
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
  *__ierr = DrawGetTitle((Draw)PetscToPointer(draw),&t);
  PetscStrncpy(c3,t,len3);
}

void drawsettitle_(Draw draw,CHAR title, int *__ierr,int len )
{
  char *t1;
  FIXCHAR(title,len,t1);
  *__ierr = DrawSetTitle((Draw)PetscToPointer(draw),t1);
  FREECHAR(title,t1);
}

void drawappendtitle_(Draw draw,CHAR title, int *__ierr,int len )
{
  char *t1;
  FIXCHAR(title,len,t1);
  *__ierr = DrawAppendTitle((Draw)PetscToPointer(draw),t1);
  FREECHAR(title,t1);
}

void drawgetpopup_(Draw draw,Draw *popup, int *__ierr )
{
  Draw tmp;
  *__ierr = DrawGetPopup((Draw)PetscToPointer(draw),&tmp);
  *(PetscFortranAddr*) popup = PetscFromPointer(tmp);
}


#if defined(__cplusplus)
}
#endif














