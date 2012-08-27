#include <petsc-private/fortranimpl.h>
#include <petscvec.h>
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscsectionview_            PETSCSECTIONVIEW
#define petscsectiongetfieldname_    PETSCSECTIONGETFIELDNAME
#define petscsectionsetfieldname_    PETSCSECTIONSETFIELDNAME
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscsectionview_            petscsectionview
#define petscsectiongetfieldname_    petscsectiongetfieldname
#define petscsectionsetfieldname_    petscsectionsetfieldname
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL petscsectionview_(PetscSection *s, PetscViewer *vin, PetscErrorCode *ierr)
{
  PetscViewer v;

  PetscPatchDefaultViewers_Fortran(vin, v);
  *ierr = PetscSectionView(*s, v);
}

void PETSC_STDCALL petscsectiongetfieldname_(PetscSection *s, PetscInt *field, CHAR name PETSC_MIXED_LEN(len), PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *fname;

  *ierr = PetscSectionGetFieldName(*s, *field, &fname);if (*ierr) return;
  *ierr = PetscStrncpy(name, fname, len);
}

void PETSC_STDCALL petscsectionsetfieldname_(PetscSection *s, PetscInt *field, CHAR name PETSC_MIXED_LEN(len), PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *f;

  FIXCHAR(name, len, f);
  *ierr = PetscSectionSetFieldName(*s, *field, f);
  FREECHAR(name, f);
}

EXTERN_C_END
