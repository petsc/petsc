#include <petsc-private/fortranimpl.h>
#include <petscdm.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dmview_                      DMVIEW
#define dmsetoptionsprefix_          DMSETOPTIONSPREFIX
#define dmsetmattype_                DMSETMATTYPE
#define dmsetvectype_                DMSETVECTYPE
#define dmgetmattype_                DMGETMATTYPE
#define dmgetvectype_                DMGETVECTYPE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dmview_                      dmview
#define dmsetoptionsprefix_          dmsetoptionsprefix
#define dmsetmattype_                dmsetmattype
#define dmsetvectype_                dmsetvectype
#define dmgetmattype_                dmgetmattype
#define dmgetvectype_                dmgetvectype
#endif

PETSC_EXTERN void PETSC_STDCALL dmgetmattype_(DM *mm,CHAR name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *tname;

  *ierr = DMGetMatType(*mm,&tname);if (*ierr) return;
  if (name != PETSC_NULL_CHARACTER_Fortran) {
    *ierr = PetscStrncpy(name,tname,len);if (*ierr) return;
  }
  FIXRETURNCHAR(PETSC_TRUE,name,len);
}

PETSC_EXTERN void PETSC_STDCALL dmgetvectype_(DM *mm,CHAR name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *tname;

  *ierr = DMGetVecType(*mm,&tname);if (*ierr) return;
  if (name != PETSC_NULL_CHARACTER_Fortran) {
    *ierr = PetscStrncpy(name,tname,len);if (*ierr) return;
  }
  FIXRETURNCHAR(PETSC_TRUE,name,len);
}

PETSC_EXTERN void PETSC_STDCALL dmview_(DM *da,PetscViewer *vin,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(vin,v);
  *ierr = DMView(*da,v);
}

PETSC_EXTERN void PETSC_STDCALL dmsetoptionsprefix_(DM *dm,CHAR prefix PETSC_MIXED_LEN(len), PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = DMSetOptionsPrefix(*dm,t);
  FREECHAR(prefix,t);
}

PETSC_EXTERN void PETSC_STDCALL dmsetmattype_(DM *dm,CHAR prefix PETSC_MIXED_LEN(len), PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = DMSetMatType(*dm,t);
  FREECHAR(prefix,t);
}


PETSC_EXTERN void PETSC_STDCALL dmsetvectype_(DM *dm,CHAR prefix PETSC_MIXED_LEN(len), PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = DMSetVecType(*dm,t);
  FREECHAR(prefix,t);
}
