#include <petsc-private/fortranimpl.h>
#include <petscdm.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dmview_                      DMVIEW
#define dmcreatecoloring_            DMCREATECOLORING
#define dmcreatematrix_              DMCREATEMATRIX
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dmview_                      dmview
#define dmcreatecoloring_            dmcreatecoloring
#define dmcreatematrix_              dmcreatematrix
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL  dmcreatecoloring_(DM *dm,ISColoringType *ctype, CHAR mtype PETSC_MIXED_LEN(len),ISColoring *coloring, int *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(mtype,len,t);
  *ierr = DMCreateColoring(*dm,*ctype,t,coloring);
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

EXTERN_C_BEGIN
void PETSC_STDCALL dmcreatematrix_(DM *dm,CHAR mat_type PETSC_MIXED_LEN(len),Mat *J,PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;
  FIXCHAR(mat_type,len,t);
  *ierr = DMCreateMatrix(*dm,t,J);
  FREECHAR(mat_type,t);
}
EXTERN_C_END
