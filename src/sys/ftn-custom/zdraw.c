
#include "zpetsc.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define petscdrawsettype_         PETSCDRAWSETTYPE
#define petscdrawcreate_          PETSCDRAWCREATE
#define petscdrawaxissetlabels_   PETSCDRAWAXISSETLABELS
#define petscdrawopenx_           PETSCDRAWOPENX
#define petscdrawstring_          PETSCDRAWSTRING
#define petscdrawstringvertical_  PETSCDRAWSTRINGVERTICAL
#define petscdrawdestroy_         PETSCDRAWDESTROY
#define petscviewerdrawgetdrawlg_ PETSCVIEWERDRAWGETDRAWLG
#define petscdrawgettitle_        PETSCDRAWGETTITLE
#define petscdrawsettitle_        PETSCDRAWSETTITLE
#define petscdrawappendtitle_     PETSCDRAWAPPENDTITLE
#define petscdrawzoom_            PETSCDRAWZOOM
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscdrawcreate_          petscdrawcreate
#define petscdrawsettype_         petscdrawsettype
#define petscdrawzoom_            petscdrawzoom
#define petscdrawaxissetlabels_   petscdrawaxissetlabels
#define petscdrawopenx_           petscdrawopenx
#define petscdrawstring_          petscdrawstring
#define petscdrawstringvertical_  petscdrawstringvertical
#define petscviewerdrawgetdraw_   petscviewerdrawgetdraw
#define petscviewerdrawgetdrawlg_ petscviewerdrawgetdrawlg
#define petscdrawgettitle_        petscdrawgettitle
#define petscdrawsettitle_        petscdrawsettitle
#define petscdrawappendtitle_     petscdrawappendtitle
#endif

typedef void (PETSC_STDCALL *FCN)(PetscDraw*,void*,PetscErrorCode*); /* force argument to next function to not be extern C*/
static FCN f1;

static PetscErrorCode ourdrawzoom(PetscDraw draw,void *ctx)
{
  PetscErrorCode ierr = 0;

  (*f1)(&draw,ctx,&ierr);CHKERRQ(ierr);
  return 0;
}
EXTERN_C_BEGIN


void PETSC_STDCALL petscdrawzoom_(PetscDraw *draw,FCN f,void *ctx,PetscErrorCode *ierr)
{
  f1      = f;
  *ierr = PetscDrawZoom(*draw,ourdrawzoom,ctx);
}

void PETSC_STDCALL petscviewerdrawgetdraw_(PetscViewer *vin,int *win,PetscDraw *draw,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(vin,v);
  *ierr = PetscViewerDrawGetDraw(v,*win,draw);
}

void PETSC_STDCALL petscviewerdrawgetdrawlg_(PetscViewer *vin,int *win,PetscDrawLG *drawlg,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(vin,v);
  *ierr = PetscViewerDrawGetDrawLG(v,*win,drawlg);
}

void PETSC_STDCALL petscdrawsettype_(PetscDraw *ctx,CHAR text PETSC_MIXED_LEN(len),
               PetscErrorCode *ierr PETSC_END_LEN(len)){
  char *t;
  FIXCHAR(text,len,t);
  *ierr = PetscDrawSetType(*ctx,t);
  FREECHAR(text,t);
}

void PETSC_STDCALL petscdrawstring_(PetscDraw *ctx,double* xl,double* yl,int* cl,CHAR text PETSC_MIXED_LEN(len),
               PetscErrorCode *ierr PETSC_END_LEN(len)){
  char *t;
  FIXCHAR(text,len,t);
  *ierr = PetscDrawString(*ctx,*xl,*yl,*cl,t);
  FREECHAR(text,t);
}
void PETSC_STDCALL petscdrawstringvertical_(PetscDraw *ctx,double *xl,double *yl,int *cl,
                   CHAR text PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;
  FIXCHAR(text,len,t);
  *ierr = PetscDrawStringVertical(*ctx,*xl,*yl,*cl,t);
  FREECHAR(text,t);
}

void PETSC_STDCALL petscdrawcreate_(MPI_Comm *comm,CHAR display PETSC_MIXED_LEN(len1),
                    CHAR title PETSC_MIXED_LEN(len2),int *x,int *y,int *w,int *h,PetscDraw* inctx,
                    PetscErrorCode *ierr PETSC_END_LEN(len1) PETSC_END_LEN(len2))
{
  char *t1,*t2;

  FIXCHAR(display,len1,t1);
  FIXCHAR(title,len2,t2);
  *ierr = PetscDrawCreate((MPI_Comm)PetscToPointerComm(*comm),t1,t2,*x,*y,*w,*h,inctx);
  FREECHAR(display,t1);
  FREECHAR(title,t2);
}

#if defined(PETSC_HAVE_X11)
void PETSC_STDCALL petscdrawopenx_(MPI_Comm *comm,CHAR display PETSC_MIXED_LEN(len1),
                    CHAR title PETSC_MIXED_LEN(len2),int *x,int *y,int *w,int *h,PetscDraw* inctx,
                    PetscErrorCode *ierr PETSC_END_LEN(len1) PETSC_END_LEN(len2))
{
  char *t1,*t2;

  FIXCHAR(display,len1,t1);
  FIXCHAR(title,len2,t2);
  *ierr = PetscDrawOpenX((MPI_Comm)PetscToPointerComm(*comm),t1,t2,*x,*y,*w,*h,inctx);
  FREECHAR(display,t1);
  FREECHAR(title,t2);
}
#endif

void PETSC_STDCALL petscdrawaxissetlabels_(PetscDrawAxis *axis,CHAR top PETSC_MIXED_LEN(len1),
                    CHAR xlabel PETSC_MIXED_LEN(len2),CHAR ylabel PETSC_MIXED_LEN(len3),
                    PetscErrorCode *ierr PETSC_END_LEN(len1) PETSC_END_LEN(len2) PETSC_END_LEN(len3))
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

void PETSC_STDCALL petscdrawgettitle_(PetscDraw *draw,CHAR title PETSC_MIXED_LEN(len),
                                 PetscErrorCode *ierr PETSC_END_LEN(len))
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

void PETSC_STDCALL petscdrawsettitle_(PetscDraw *draw,CHAR title PETSC_MIXED_LEN(len),
                                 PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t1;
  FIXCHAR(title,len,t1);
  *ierr = PetscDrawSetTitle(*draw,t1);
  FREECHAR(title,t1);
}

void PETSC_STDCALL petscdrawappendtitle_(PetscDraw *draw,CHAR title PETSC_MIXED_LEN(len),
                                    PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t1;
  FIXCHAR(title,len,t1);
  *ierr = PetscDrawAppendTitle(*draw,t1);
  FREECHAR(title,t1);
}

EXTERN_C_END
