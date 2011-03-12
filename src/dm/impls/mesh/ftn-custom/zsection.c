#include "private/fortranimpl.h"
#include "petscmesh.h"

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
#define sectionrealdistribute_ SECTIONREALDISTRIBUTE
#define sectionintdistribute_  SECTIONINTDISTRIBUTE
#define sectionrealgetfibration_ SECTIONREALGETFIBRATION
#define sectionintgetfibration_  SECTIONINTGETFIBRATION
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define sectionrealview_       sectionrealview
#define sectionintview_        sectionintview
#define sectionrealdistribute_ sectionrealdistribute
#define sectionintdistribute_  sectionintdistribute
#define sectionrealgetfibration_ sectionrealgetfibration
#define sectionintgetfibration_  sectionintgetfibration
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
void PETSC_STDCALL sectionrealdistribute(SectionReal *serialSection, Mesh parallelMesh, SectionReal *parallelSection, PetscErrorCode *ierr)
{
  *ierr = SectionRealDistribute(*serialSection, (Mesh) PetscToPointer(parallelMesh), parallelSection);
}
void PETSC_STDCALL sectionintdistribute(SectionInt *serialSection, Mesh parallelMesh, SectionInt *parallelSection, PetscErrorCode *ierr)
{
  *ierr = SectionIntDistribute(*serialSection, (Mesh) PetscToPointer(parallelMesh), parallelSection);
}
void PETSC_STDCALL sectionrealgetfibration_(SectionReal *section, PetscInt *field,SectionReal *subsection, int *__ierr ){
  *__ierr = SectionRealGetFibration(*section,*field,subsection);
}
void PETSC_STDCALL sectionintgetfibration_(SectionInt *section, PetscInt *field,SectionInt *subsection, int *__ierr ){
  *__ierr = SectionIntGetFibration(*section,*field,subsection);
}
EXTERN_C_END
