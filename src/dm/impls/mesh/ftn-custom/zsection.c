#include <private/fortranimpl.h>
#include <petscdmmesh.h>

#ifdef PETSC_USE_POINTER_CONVERSION
#if defined(__cplusplus)
extern "C" {
#endif
extern void *PetscToPointer(void*);
extern int PetscFromPointer(void *);
extern void PetscRmPointer(void*);
#if defined(__cplusplus)
}
#endif

#else

#define PetscToPointer(a) (*(long *)(a))
#define PetscFromPointer(a) (long)(a)
#define PetscRmPointer(a)
#endif

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define sectionrealview_       SECTIONREALVIEW
#define sectionintview_        SECTIONINTVIEW
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define sectionrealview_       sectionrealview
#define sectionintview_        sectionintview
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL sectionrealview_(SectionReal *x,PetscViewer *vin,PetscErrorCode *ierr)
{
  PetscViewer v;

  PetscPatchDefaultViewers_Fortran(vin,v);
  *ierr = SectionRealView(*x,v);
}
void PETSC_STDCALL sectionintview_(SectionInt *x,PetscViewer *vin,PetscErrorCode *ierr)
{
  PetscViewer v;

  PetscPatchDefaultViewers_Fortran(vin,v);
  *ierr = SectionIntView(*x,v);
}
EXTERN_C_END
