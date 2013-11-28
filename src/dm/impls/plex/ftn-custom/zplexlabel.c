#include <petsc-private/fortranimpl.h>
#include <petscdmplex.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dmplexhaslabel_            DMPLEXHASLABEL
#define dmplexgetlabelvalue_       DMPLEXGETLABELVALUE
#define dmplexsetlabelvalue_       DMPLEXSETLABELVALUE
#define dmplexgetlabelsize_        DMPLEXGETLABELSIZE
#define dmplexgetlabelidis_        DMPLEXGETLABELIDIS
#define dmplexgetstratumsize_      DMPLEXGETSTRATUMSIZE
#define dmplexgetstratumis_        DMPLEXGETSTRATUMIS
#define dmplexgetlabel_            DMPLEXGETLABEL
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define dmplexhaslabel_            dmplexhaslabel
#define dmplexgetlabelvalue_       dmplexgetlabelvalue
#define dmplexsetlabelvalue_       dmplexsetlabelvalue
#define dmplexgetlabelsize_        dmplexlabelsize
#define dmplexgetlabelidis_        dmplexlabelidis
#define dmplexgetstratumsize_      dmplexgetstratumsize
#define dmplexgetstratumis_        dmplexgetstratumis
#define dmplexgetlabel_            dmplexgetlabel
#endif

/* Definitions of Fortran Wrapper routines */

PETSC_EXTERN void PETSC_STDCALL dmplexhaslabel_(DM *dm, CHAR name PETSC_MIXED_LEN(lenN), PetscBool *hasLabel, int *ierr PETSC_END_LEN(lenN))
{
  char *lname;

  FIXCHAR(name, lenN, lname);
  *ierr = DMPlexHasLabel(*dm, lname, hasLabel);
  FREECHAR(name, lname);
}

PETSC_EXTERN void PETSC_STDCALL dmplexgetlabelvalue_(DM *dm, CHAR name PETSC_MIXED_LEN(lenN), PetscInt *point, PetscInt *value, int *ierr PETSC_END_LEN(lenN))
{
  char *lname;

  FIXCHAR(name, lenN, lname);
  *ierr = DMPlexGetLabelValue(*dm, lname, *point, value);
  FREECHAR(name, lname);
}

PETSC_EXTERN void PETSC_STDCALL dmplexsetlabelvalue_(DM *dm, CHAR name PETSC_MIXED_LEN(lenN), PetscInt *point, PetscInt *value, int *ierr PETSC_END_LEN(lenN))
{
  char *lname;

  FIXCHAR(name, lenN, lname);
  *ierr = DMPlexSetLabelValue(*dm, lname, *point, *value);
  FREECHAR(name, lname);
}

PETSC_EXTERN void PETSC_STDCALL dmplexgetlabelsize_(DM *dm, CHAR name PETSC_MIXED_LEN(lenN), PetscInt *size, int *ierr PETSC_END_LEN(lenN))
{
  char *lname;

  FIXCHAR(name, lenN, lname);
  *ierr = DMPlexGetLabelSize(*dm, lname, size);
  FREECHAR(name, lname);
}

PETSC_EXTERN void PETSC_STDCALL dmplexgetlabelidis_(DM *dm, CHAR name PETSC_MIXED_LEN(lenN), IS *ids, int *ierr PETSC_END_LEN(lenN))
{
  char *lname;

  FIXCHAR(name, lenN, lname);
  *ierr = DMPlexGetLabelIdIS(*dm, lname, ids);
  FREECHAR(name, lname);
}

PETSC_EXTERN void PETSC_STDCALL dmplexgetstratumsize_(DM *dm, CHAR name PETSC_MIXED_LEN(lenN), PetscInt *value, PetscInt *size, int *ierr PETSC_END_LEN(lenN))
{
  char *lname;

  FIXCHAR(name, lenN, lname);
  *ierr = DMPlexGetStratumSize(*dm, lname, *value, size);
  FREECHAR(name, lname);
}

PETSC_EXTERN void PETSC_STDCALL dmplexgetstratumis_(DM *dm, CHAR name PETSC_MIXED_LEN(lenN), PetscInt *value, IS *is, int *ierr PETSC_END_LEN(lenN))
{
  char *lname;

  FIXCHAR(name, lenN, lname);
  *ierr = DMPlexGetStratumIS(*dm, lname, *value, is);
  FREECHAR(name, lname);
}

PETSC_EXTERN void PETSC_STDCALL dmplexgetlabel_(DM *dm, CHAR name PETSC_MIXED_LEN(lenN), DMLabel *label, int *ierr PETSC_END_LEN(lenN))
{
  char *lname;

  FIXCHAR(name, lenN, lname);
  *ierr = DMPlexGetLabel(*dm, lname, label);
  FREECHAR(name, lname);
}
