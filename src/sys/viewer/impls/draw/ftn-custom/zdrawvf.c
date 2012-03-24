#include <petsc-private/fortranimpl.h>
#include <petscdraw.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscviewerdrawgetdraw_   PETSCVIEWERDRAWGETDRAW
#define petscviewerdrawgetdrawlg_ PETSCVIEWERDRAWGETDRAWLG
#define petscviewerdrawopen_       PETSCVIEWERDRAWOPEN
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscviewerdrawgetdraw_   petscviewerdrawgetdraw
#define petscviewerdrawgetdrawlg_ petscviewerdrawgetdrawlg
#define petscviewerdrawopen_       petscviewerdrawopen
#endif

EXTERN_C_BEGIN

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

void PETSC_STDCALL petscviewerdrawopen_(MPI_Comm *comm,CHAR display PETSC_MIXED_LEN(len1),
                   CHAR title PETSC_MIXED_LEN(len2),int *x,int*y,int*w,int*h,PetscViewer *v,
                   PetscErrorCode *ierr PETSC_END_LEN(len1) PETSC_END_LEN(len2))
{
  char   *c1,*c2;

  FIXCHAR(display,len1,c1);
  FIXCHAR(title,len2,c2);
  *ierr = PetscViewerDrawOpen(MPI_Comm_f2c(*(MPI_Fint *)&*comm),c1,c2,*x,*y,*w,*h,v);
  FREECHAR(display,c1);
  FREECHAR(title,c2);
}

EXTERN_C_END
