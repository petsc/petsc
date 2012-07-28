#include <petsc-private/fortranimpl.h>
#include <petscdmcomplex.h>

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define dmcomplexdistribute_          DMCOMPLEXDISTRIBUTE
#define dmcomplexhaslabel_            DMCOMPLEXHASLABEL
#define dmcomplexgetlabelvalue_       DMCOMPLEXGETLABELVALUE
#define dmcomplexsetlabelvalue_       DMCOMPLEXSETLABELVALUE
#define dmcomplexgetlabelsize_        DMCOMPLEXGETLABELSIZE
#define dmcomplexgetlabelidis_        DMCOMPLEXGETLABELIDIS
#define dmcomplexgetstratumsize_      DMCOMPLEXGETSTRATUMSIZE
#define dmcomplexgetstratumis_        DMCOMPLEXGETSTRATUMIS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define dmcomplexdistribute_          dmcomplexdistribute
#define dmcomplexhaslabel_            dmcomplexhaslabel
#define dmcomplexgetlabelvalue_       dmcomplexgetlabelvalue
#define dmcomplexsetlabelvalue_       dmcomplexsetlabelvalue
#define dmcomplexgetlabelsize_        dmcomplexlabelsize
#define dmcomplexgetlabelidis_        dmcomplexlabelidis
#define dmcomplexgetstratumsize_      dmcomplexgetstratumsize
#define dmcomplexgetstratumis_        dmcomplexgetstratumis
#endif

/* Definitions of Fortran Wrapper routines */
EXTERN_C_BEGIN

void PETSC_STDCALL dmcomplexdistribute_(DM *dm, CHAR name PETSC_MIXED_LEN(lenN), DM *dmParallel, int *ierr PETSC_END_LEN(lenN)) {
  char *partitioner;

  FIXCHAR(name, lenN, partitioner);
  *ierr = DMComplexDistribute(*dm, partitioner, dmParallel);
  FREECHAR(name, partitioner);
}

void PETSC_STDCALL dmcomplexhaslabel_(DM *dm, CHAR name PETSC_MIXED_LEN(lenN), PetscBool *hasLabel, int *ierr PETSC_END_LEN(lenN)) {
  char *lname;

  FIXCHAR(name, lenN, lname);
  *ierr = DMComplexHasLabel(*dm, lname, hasLabel);
  FREECHAR(name, lname);
}

void PETSC_STDCALL dmcomplexgetlabelvalue_(DM *dm, CHAR name PETSC_MIXED_LEN(lenN), PetscInt *point, PetscInt *value, int *ierr PETSC_END_LEN(lenN)) {
  char *lname;

  FIXCHAR(name, lenN, lname);
  *ierr = DMComplexGetLabelValue(*dm, lname, *point, value);
  FREECHAR(name, lname);
}

void PETSC_STDCALL dmcomplexsetlabelvalue_(DM *dm, CHAR name PETSC_MIXED_LEN(lenN), PetscInt *point, PetscInt *value, int *ierr PETSC_END_LEN(lenN)) {
  char *lname;

  FIXCHAR(name, lenN, lname);
  *ierr = DMComplexSetLabelValue(*dm, lname, *point, *value);
  FREECHAR(name, lname);
}

void PETSC_STDCALL dmcomplexgetlabelsize_(DM *dm, CHAR name PETSC_MIXED_LEN(lenN), PetscInt *size, int *ierr PETSC_END_LEN(lenN)) {
  char *lname;

  FIXCHAR(name, lenN, lname);
  *ierr = DMComplexGetLabelSize(*dm, lname, size);
  FREECHAR(name, lname);
}

void PETSC_STDCALL dmcomplexgetlabelidis_(DM *dm, CHAR name PETSC_MIXED_LEN(lenN), IS *ids, int *ierr PETSC_END_LEN(lenN)) {
  char *lname;

  FIXCHAR(name, lenN, lname);
  *ierr = DMComplexGetLabelIdIS(*dm, lname, ids);
  FREECHAR(name, lname);
}

void PETSC_STDCALL dmcomplexgetstratumsize_(DM *dm, CHAR name PETSC_MIXED_LEN(lenN), PetscInt *value, PetscInt *size, int *ierr PETSC_END_LEN(lenN)) {
  char *lname;

  FIXCHAR(name, lenN, lname);
  *ierr = DMComplexGetStratumSize(*dm, lname, *value, size);
  FREECHAR(name, lname);
}

void PETSC_STDCALL dmcomplexgetstratumis_(DM *dm, CHAR name PETSC_MIXED_LEN(lenN), PetscInt *value, IS *is, int *ierr PETSC_END_LEN(lenN)) {
  char *lname;

  FIXCHAR(name, lenN, lname);
  *ierr = DMComplexGetStratumIS(*dm, lname, *value, is);
  FREECHAR(name, lname);
}

EXTERN_C_END
