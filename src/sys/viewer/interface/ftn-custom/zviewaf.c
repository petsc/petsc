#include "zpetsc.h"
#include "petsc.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscviewersetformat_      PETSCVIEWERSETFORMAT
#define petscviewerpushformat_     PETSCVIEWERPUSHFORMAT
#define petscviewerpopformat_      PETSCVIEWERPOPFORMAT
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscviewersetformat_      petscviewersetformat
#define petscviewerpushformat_     petscviewerpushformat
#define petscviewerpopformat_      petscviewerpopformat
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL petscviewersetformat_(PetscViewer *vin,PetscViewerFormat *format,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(vin,v);
  *ierr = PetscViewerSetFormat(v,*format);
}

void PETSC_STDCALL petscviewerpushformat_(PetscViewer *vin,PetscViewerFormat *format,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(vin,v);
  *ierr = PetscViewerPushFormat(v,*format);
}

void PETSC_STDCALL petscviewerpopformat_(PetscViewer *vin,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(vin,v);
  *ierr = PetscViewerPopFormat(v);
}

EXTERN_C_END
