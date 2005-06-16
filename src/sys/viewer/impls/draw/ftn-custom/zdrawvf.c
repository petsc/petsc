#include "zpetsc.h"
#include "petscsys.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscviewerdrawgetdraw_   PETSCVIEWERDRAWGETDRAW
#define petscviewerdrawgetdrawlg_ PETSCVIEWERDRAWGETDRAWLG
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscviewerdrawgetdraw_   petscviewerdrawgetdraw
#define petscviewerdrawgetdrawlg_ petscviewerdrawgetdrawlg
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


EXTERN_C_END
