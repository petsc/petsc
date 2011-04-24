#include <private/fortranimpl.h>
#include <petscdm.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dmview_                      DMVIEW
#define dmgetcoloring_               DMGETCOLORING
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dmview_                      dmview
#define dmgetcoloring_               dmgetcoloring
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL  dmgetcoloring_(DM *dm,ISColoringType *ctype, CHAR mtype PETSC_MIXED_LEN(len),ISColoring *coloring, int *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(mtype,len,t);
  *ierr = DMGetColoring(*dm,*ctype,t,coloring);
  FREECHAR(mtype,t);
}
EXTERN_C_END

EXTERN_C_BEGIN
void PETSC_STDCALL dmview_(DM *da,PetscViewer *vin,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(vin,v);
  *ierr = DMView(*da,v);
}
EXTERN_C_END
