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
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscsectiongetpointsyms_          petscsectiongetpointsyms
#define petscsectionrestorepointsyms_      petscsectionrestorepointsyms
#define petscsectiongetfieldpointsyms_     petscsectiongetfieldpointsyms
#define petscsectionrestorefieldpointsyms_ petscsectionrestorefieldpointsyms
#define petscsectionview_                  petscsectionview
#define petscsectiongetfieldname_          petscsectiongetfieldname
#define petscsectionsetfieldname_          petscsectionsetfieldname
#endif

PETSC_EXTERN void  petscsectiongetpointsyms_(PetscSection section,PetscInt *numPoints, PetscInt *points, PetscInt ***perms, PetscScalar ***rots, int *__ierr){
*__ierr = PetscSectionGetPointSyms(section,*numPoints,points,(const PetscInt ***)perms,(const PetscScalar ***)rots);
}
PETSC_EXTERN void  petscsectionrestorepointsyms_(PetscSection section,PetscInt *numPoints, PetscInt *points, PetscInt ***perms, PetscScalar ***rots, int *__ierr){
*__ierr = PetscSectionRestorePointSyms(section,*numPoints,points,(const PetscInt ***)perms,(const PetscScalar ***)rots);
}
PETSC_EXTERN void  petscsectiongetfieldpointsyms_(PetscSection section,PetscInt *field,PetscInt *numPoints, PetscInt *points, PetscInt ***perms, PetscScalar ***rots, int *__ierr){
*__ierr = PetscSectionGetFieldPointSyms(section,*field,*numPoints,points,(const PetscInt ***)perms,(const PetscScalar ***)rots);
}
PETSC_EXTERN void  petscsectionrestorefieldpointsyms_(PetscSection section,PetscInt *field,PetscInt *numPoints, PetscInt *points, PetscInt ***perms, PetscScalar ***rots, int *__ierr){
*__ierr = PetscSectionRestoreFieldPointSyms(section,*field,*numPoints,points,(const PetscInt ***)perms,(const PetscScalar ***)rots);
}

PETSC_EXTERN void petscsectionview_(PetscSection *s, PetscViewer *vin, PetscErrorCode *ierr)
{
  PetscViewer v;

  PetscPatchDefaultViewers_Fortran(vin, v);
  *ierr = PetscSectionView(*s, v);
}

PETSC_EXTERN void petscsectiongetfieldname_(PetscSection *s, PetscInt *field, char* name, PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  const char *fname;

  *ierr = PetscSectionGetFieldName(*s, *field, &fname);if (*ierr) return;
  *ierr = PetscStrncpy(name, fname, len);
  FIXRETURNCHAR(PETSC_TRUE,name,len);
}

PETSC_EXTERN void petscsectionsetfieldname_(PetscSection *s, PetscInt *field, char* name, PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *f;

  FIXCHAR(name, len, f);
  *ierr = PetscSectionSetFieldName(*s, *field, f);if (*ierr) return;
  FREECHAR(name, f);
}
