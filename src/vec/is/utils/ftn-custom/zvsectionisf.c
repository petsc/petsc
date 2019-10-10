#include <petsc/private/fortranimpl.h>
#include <petscis.h>
#include <petscsection.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscsectiongetpointsyms_          PETSCSECTIONGETPOINTSYMS
#define petscsectionrestorepointsyms_      PETSCSECTIONRESTOREPOINTSYMS
#define petscsectiongetfieldpointsyms_     PETSCSECTIONGETFIELDPOINTSYMS
#define petscsectionrestorefieldpointsyms_ PETSCSECTIONRESTOREFIELDPOINTSYMS
#define petscsectionview_                  PETSCSECTIONVIEW
#define petscsectiongetfieldname_          PETSCSECTIONGETFIELDNAME
#define petscsectionsetfieldname_          PETSCSECTIONSETFIELDNAME
#define petscsfdistributesection_          PETSCSFDISTRIBUTESECTION
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscsectiongetpointsyms_          petscsectiongetpointsyms
#define petscsectionrestorepointsyms_      petscsectionrestorepointsyms
#define petscsectiongetfieldpointsyms_     petscsectiongetfieldpointsyms
#define petscsectionrestorefieldpointsyms_ petscsectionrestorefieldpointsyms
#define petscsectionview_                  petscsectionview
#define petscsectiongetfieldname_          petscsectiongetfieldname
#define petscsectionsetfieldname_          petscsectionsetfieldname
#define petscsfdistributesection_          petscsfdistributesection
#endif

PETSC_EXTERN void PETSC_STDCALL  petscsectiongetpointsyms_(PetscSection section,PetscInt *numPoints, PetscInt *points, PetscInt ***perms, PetscScalar ***rots, int *__ierr ){
*__ierr = PetscSectionGetPointSyms(section,*numPoints,points,(const PetscInt ***)perms,(const PetscScalar ***)rots);
}
PETSC_EXTERN void PETSC_STDCALL  petscsectionrestorepointsyms_(PetscSection section,PetscInt *numPoints, PetscInt *points, PetscInt ***perms, PetscScalar ***rots, int *__ierr ){
*__ierr = PetscSectionRestorePointSyms(section,*numPoints,points,(const PetscInt ***)perms,(const PetscScalar ***)rots);
}
PETSC_EXTERN void PETSC_STDCALL  petscsectiongetfieldpointsyms_(PetscSection section,PetscInt *field,PetscInt *numPoints, PetscInt *points, PetscInt ***perms, PetscScalar ***rots, int *__ierr ){
*__ierr = PetscSectionGetFieldPointSyms(section,*field,*numPoints,points,(const PetscInt ***)perms,(const PetscScalar ***)rots);
}
PETSC_EXTERN void PETSC_STDCALL  petscsectionrestorefieldpointsyms_(PetscSection section,PetscInt *field,PetscInt *numPoints, PetscInt *points, PetscInt ***perms, PetscScalar ***rots, int *__ierr ){
*__ierr = PetscSectionRestoreFieldPointSyms(section,*field,*numPoints,points,(const PetscInt ***)perms,(const PetscScalar ***)rots);
}

PETSC_EXTERN void PETSC_STDCALL petscsectionview_(PetscSection *s, PetscViewer *vin, PetscErrorCode *ierr)
{
  PetscViewer v;

  PetscPatchDefaultViewers_Fortran(vin, v);
  *ierr = PetscSectionView(*s, v);
}

PETSC_EXTERN void PETSC_STDCALL petscsectiongetfieldname_(PetscSection *s, PetscInt *field, char* name PETSC_MIXED_LEN(len), PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *fname;

  *ierr = PetscSectionGetFieldName(*s, *field, &fname);if (*ierr) return;
  *ierr = PetscStrncpy(name, fname, len);
  FIXRETURNCHAR(PETSC_TRUE,name,len);
}

PETSC_EXTERN void PETSC_STDCALL petscsectionsetfieldname_(PetscSection *s, PetscInt *field, char* name PETSC_MIXED_LEN(len), PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *f;

  FIXCHAR(name, len, f);
  *ierr = PetscSectionSetFieldName(*s, *field, f);if (*ierr) return;
  FREECHAR(name, f);
}

PETSC_EXTERN void PETSC_STDCALL  petscsfdistributesection_(PetscSF sf,PetscSection rootSection,PetscInt **remoteOffsets,PetscSection leafSection, int *__ierr ){
  if (remoteOffsets != PETSC_NULL_INTEGER_Fortran) {
    PetscError(PETSC_COMM_SELF, __LINE__, "PetscSFDistributeSection_Fortran", __FILE__, PETSC_ERR_SUP, PETSC_ERROR_INITIAL,
               "The remoteOffsets argument must be PETSC_NULL_INTEGER in Fortran");
    *__ierr = 1;
    return;
  }
  *__ierr = PetscSFDistributeSection(sf,rootSection,NULL,leafSection);
}
