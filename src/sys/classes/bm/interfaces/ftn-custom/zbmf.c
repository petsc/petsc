#include <petsc/private/fortranimpl.h>
#include <petscbm.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petscbmsettype_          PETSCBMSETTYPE
  #define petscbmgettype_          PETSCBMGETTYPE
  #define petscbmsetoptionsprefix_ PETSCBMSETOPTIONSPREFIX
  #define petscbmviewfromoptions_  PETSCBMVIEWFROMOPTIONS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define petscbmsettype_          petscbmsettype
  #define petscbmgettype_          petscbmgettype
  #define petscbmsetoptionsprefix_ petscbmsetoptionsprefix
  #define petscbmviewfromoptions_  petscbmviewfromoptions
#endif

PETSC_EXTERN void petscbmsettype_(PetscBench *ctx, char *text, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;
  FIXCHAR(text, len, t);
  *ierr = PetscBenchSetType(*ctx, t);
  if (*ierr) return;
  FREECHAR(text, t);
}

PETSC_EXTERN void petscbmgettype_(PetscBench *bm, char *name, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  const char *tname;

  *ierr = PetscBenchGetType(*bm, &tname);
  if (*ierr) return;
  *ierr = PetscStrncpy(name, tname, len);
  FIXRETURNCHAR(PETSC_TRUE, name, len);
}

PETSC_EXTERN void petscbmsetoptionsprefix_(PetscBench *ctx, char *text, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;
  FIXCHAR(text, len, t);
  *ierr = PetscBenchSetOptionsPrefix(*ctx, t);
  if (*ierr) return;
  FREECHAR(text, t);
}

PETSC_EXTERN void petscbmviewfromoptions_(PetscBench *bm, PetscObject obj, char *type, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type, len, t);
  CHKFORTRANNULLOBJECT(obj);
  *ierr = PetscBenchViewFromOptions(*bm, obj, t);
  if (*ierr) return;
  FREECHAR(type, t);
}
