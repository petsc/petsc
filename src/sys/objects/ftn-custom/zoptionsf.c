/*
  This file contains Fortran stubs for Options routines.
  These are not generated automatically since they require passing strings
  between Fortran and C.
*/

#include <petsc/private/ftnimpl.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petscoptionsbegin_               PETSCOPTIONSBEGIN
  #define petscoptionsend_                 PETSCOPTIONSEND
  #define petscoptionsbool_                PETSCOPTIONSBOOL
  #define petscoptionsbool3_               PETSCOPTIONSBOOL3
  #define petscoptionsboolarray_           PETSCOPTIONSBOOLARRAY
  #define petscoptionsenumprivate_         PETSCOPTIONSENUMPRIVATE
  #define petscoptionsint_                 PETSCOPTIONSINT
  #define petscoptionsintarray_            PETSCOPTIONSINTARRAY
  #define petscoptionsreal_                PETSCOPTIONSREAL
  #define petscoptionsrealarray_           PETSCOPTIONSREALARRAY
  #define petscoptionsscalar_              PETSCOPTIONSSCALAR
  #define petscoptionsscalararray_         PETSCOPTIONSSCALARARRAY
  #define petscoptionsstring_              PETSCOPTIONSSTRING
  #define petscsubcommgetparent_           PETSCSUBCOMMGETPARENT
  #define petscsubcommgetcontiguousparent_ PETSCSUBCOMMGETCONTIGUOUSPARENT
  #define petscsubcommgetchild_            PETSCSUBCOMMGETCHILD
  #define petscoptionsallused_             PETSCOPTIONSALLUSED
  #define petscoptionsgetenumprivate_      PETSCOPTIONSGETENUMPRIVATE
  #define petscoptionsgetstring_           PETSCOPTIONSGETSTRING
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define petscoptionsbegin_               petscoptionsbegin
  #define petscoptionsend_                 petscoptionsend
  #define petscoptionsbool_                petscoptionsbool
  #define petscoptionsbool3_               petscoptionsbool3
  #define petscoptionsboolarray_           petscoptionsboolarray
  #define petscoptionsenumprivate_         petscoptionsenumprivate
  #define petscoptionsint_                 petscoptionsint
  #define petscoptionsintarray_            petscoptionsintarray
  #define petscoptionsreal_                petscoptionsreal
  #define petscoptionsrealarray_           petscoptionsrealarray
  #define petscoptionsscalar_              petscoptionsscalar
  #define petscoptionsscalararray_         petscoptionsscalararray
  #define petscoptionsstring_              petscoptionsstring
  #define petscsubcommgetparent_           petscsubcommgetparent
  #define petscsubcommgetcontiguousparent_ petscsubcommgetcontiguousparent
  #define petscsubcommgetchild_            petscsubcommgetchild
  #define petscoptionsallused_             petscoptionsallused
  #define petscoptionsgetenumprivate_      petscoptionsgetenumprivate
  #define petscoptionsgetstring_           petscoptionsgetstring
#endif

static struct _n_PetscOptionItems PetscOptionsObjectBase;
static PetscOptionItems           PetscOptionsObject = NULL;

PETSC_EXTERN void petscoptionsbegin_(MPI_Fint *fcomm, char *prefix, char *mess, char *sec, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T lenprefix, PETSC_FORTRAN_CHARLEN_T lenmess, PETSC_FORTRAN_CHARLEN_T lensec)
{
  MPI_Comm comm = MPI_Comm_f2c(*fcomm);
  char    *cprefix, *cmess, *csec;

  FIXCHAR(prefix, lenprefix, cprefix);
  FIXCHAR(mess, lenmess, cmess);
  FIXCHAR(sec, lensec, csec);
  if (PetscOptionsObject) {
    *ierr = PETSC_ERR_ARG_WRONGSTATE;
    return;
  }
  PetscOptionsObject = &PetscOptionsObjectBase;
  *ierr              = PetscMemzero(PetscOptionsObject, sizeof(*PetscOptionsObject));
  if (*ierr) return;
  PetscOptionsObject->count = 1;
  *ierr                     = PetscOptionsBegin_Private(PetscOptionsObject, comm, cprefix, cmess, csec);
  if (*ierr) return;
  FREECHAR(prefix, cprefix);
  FREECHAR(mess, cmess);
  FREECHAR(sec, csec);
}

PETSC_EXTERN void petscoptionsend_(PetscErrorCode *ierr)
{
  if (!PetscOptionsObject) {
    *ierr = PETSC_ERR_ARG_WRONGSTATE;
    return;
  }
  PetscOptionsObject->count = 1;
  *ierr                     = PetscOptionsEnd_Private(PetscOptionsObject);
  PetscOptionsObject        = NULL;
}

PETSC_EXTERN void petscoptionsbool_(char *opt, char *text, char *man, PetscBool *currentvalue, PetscBool *value, PetscBool *set, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T lenopt, PETSC_FORTRAN_CHARLEN_T lentext, PETSC_FORTRAN_CHARLEN_T lenman)
{
  char *copt, *ctext, *cman;

  FIXCHAR(opt, lenopt, copt);
  FIXCHAR(text, lentext, ctext);
  FIXCHAR(man, lenman, cman);
  if (!PetscOptionsObject) {
    *ierr = PETSC_ERR_ARG_WRONGSTATE;
    return;
  }
  PetscOptionsObject->count = 1;
  *ierr                     = PetscOptionsBool_Private(PetscOptionsObject, copt, ctext, cman, *currentvalue, value, set);
  if (*ierr) return;
  FREECHAR(opt, copt);
  FREECHAR(text, ctext);
  FREECHAR(man, cman);
}

PETSC_EXTERN void petscoptionsbool3_(char *opt, char *text, char *man, PetscBool3 *currentvalue, PetscBool3 *value, PetscBool *set, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T lenopt, PETSC_FORTRAN_CHARLEN_T lentext, PETSC_FORTRAN_CHARLEN_T lenman)
{
  char *copt, *ctext, *cman;

  FIXCHAR(opt, lenopt, copt);
  FIXCHAR(text, lentext, ctext);
  FIXCHAR(man, lenman, cman);
  if (!PetscOptionsObject) {
    *ierr = PETSC_ERR_ARG_WRONGSTATE;
    return;
  }
  PetscOptionsObject->count = 1;
  *ierr                     = PetscOptionsBool3_Private(PetscOptionsObject, copt, ctext, cman, *currentvalue, value, set);
  if (*ierr) return;
  FREECHAR(opt, copt);
  FREECHAR(text, ctext);
  FREECHAR(man, cman);
}

PETSC_EXTERN void petscoptionsboolarray_(char *opt, char *text, char *man, PetscBool *dvalue, PetscInt *nmax, PetscBool *flg, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T lenopt, PETSC_FORTRAN_CHARLEN_T lentext, PETSC_FORTRAN_CHARLEN_T lenman)
{
  char     *copt, *ctext, *cman;
  PetscBool flag;

  FIXCHAR(opt, lenopt, copt);
  FIXCHAR(text, lentext, ctext);
  FIXCHAR(man, lenman, cman);
  if (!PetscOptionsObject) {
    *ierr = PETSC_ERR_ARG_WRONGSTATE;
    return;
  }
  PetscOptionsObject->count = 1;
  *ierr                     = PetscOptionsBoolArray_Private(PetscOptionsObject, copt, ctext, cman, dvalue, nmax, &flag);
  if (*ierr) return;
  if (!FORTRANNULLBOOL(flg)) *flg = flag;
  FREECHAR(opt, copt);
  FREECHAR(text, ctext);
  FREECHAR(man, cman);
}

PETSC_EXTERN void petscoptionsenumprivate_(char *opt, char *text, char *man, const char *const *list, PetscEnum *currentvalue, PetscEnum *ivalue, PetscBool *flg, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T lenopt, PETSC_FORTRAN_CHARLEN_T lentext, PETSC_FORTRAN_CHARLEN_T lenman)
{
  char     *copt, *ctext, *cman;
  PetscBool flag;

  FIXCHAR(opt, lenopt, copt);
  FIXCHAR(text, lentext, ctext);
  FIXCHAR(man, lenman, cman);
  if (!PetscOptionsObject) {
    *ierr = PETSC_ERR_ARG_WRONGSTATE;
    return;
  }
  PetscOptionsObject->count = 1;
  *ierr                     = PetscOptionsEnum_Private(PetscOptionsObject, copt, ctext, cman, list, *currentvalue, ivalue, &flag);
  if (*ierr) return;
  if (!FORTRANNULLBOOL(flg)) *flg = flag;
  FREECHAR(opt, copt);
  FREECHAR(text, ctext);
  FREECHAR(man, cman);
}

PETSC_EXTERN void petscoptionsint_(char *opt, char *text, char *man, PetscInt *currentvalue, PetscInt *value, PetscBool *set, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T lenopt, PETSC_FORTRAN_CHARLEN_T lentext, PETSC_FORTRAN_CHARLEN_T lenman)
{
  char *copt, *ctext, *cman;

  FIXCHAR(opt, lenopt, copt);
  FIXCHAR(text, lentext, ctext);
  FIXCHAR(man, lenman, cman);
  if (!PetscOptionsObject) {
    *ierr = PETSC_ERR_ARG_WRONGSTATE;
    return;
  }
  PetscOptionsObject->count = 1;
  *ierr                     = PetscOptionsInt_Private(PetscOptionsObject, copt, ctext, cman, *currentvalue, value, set, PETSC_INT_MIN, PETSC_INT_MAX);
  if (*ierr) return;
  FREECHAR(opt, copt);
  FREECHAR(text, ctext);
  FREECHAR(man, cman);
}

PETSC_EXTERN void petscoptionsintarray_(char *opt, char *text, char *man, PetscInt *currentvalue, PetscInt *n, PetscBool *set, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T lenopt, PETSC_FORTRAN_CHARLEN_T lentext, PETSC_FORTRAN_CHARLEN_T lenman)
{
  char *copt, *ctext, *cman;

  FIXCHAR(opt, lenopt, copt);
  FIXCHAR(text, lentext, ctext);
  FIXCHAR(man, lenman, cman);
  if (!PetscOptionsObject) {
    *ierr = PETSC_ERR_ARG_WRONGSTATE;
    return;
  }
  PetscOptionsObject->count = 1;
  *ierr                     = PetscOptionsIntArray_Private(PetscOptionsObject, copt, ctext, cman, currentvalue, n, set);
  if (*ierr) return;
  FREECHAR(opt, copt);
  FREECHAR(text, ctext);
  FREECHAR(man, cman);
}

PETSC_EXTERN void petscoptionsreal_(char *opt, char *text, char *man, PetscReal *currentvalue, PetscReal *value, PetscBool *set, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T lenopt, PETSC_FORTRAN_CHARLEN_T lentext, PETSC_FORTRAN_CHARLEN_T lenman)
{
  char *copt, *ctext, *cman;

  FIXCHAR(opt, lenopt, copt);
  FIXCHAR(text, lentext, ctext);
  FIXCHAR(man, lenman, cman);
  if (!PetscOptionsObject) {
    *ierr = PETSC_ERR_ARG_WRONGSTATE;
    return;
  }
  PetscOptionsObject->count = 1;
  *ierr                     = PetscOptionsReal_Private(PetscOptionsObject, copt, ctext, cman, *currentvalue, value, set, PETSC_MIN_REAL, PETSC_MAX_REAL);
  if (*ierr) return;
  FREECHAR(opt, copt);
  FREECHAR(text, ctext);
  FREECHAR(man, cman);
}

PETSC_EXTERN void petscoptionsrealarray_(char *opt, char *text, char *man, PetscReal *currentvalue, PetscInt *n, PetscBool *set, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T lenopt, PETSC_FORTRAN_CHARLEN_T lentext, PETSC_FORTRAN_CHARLEN_T lenman)
{
  char *copt, *ctext, *cman;

  FIXCHAR(opt, lenopt, copt);
  FIXCHAR(text, lentext, ctext);
  FIXCHAR(man, lenman, cman);
  if (!PetscOptionsObject) {
    *ierr = PETSC_ERR_ARG_WRONGSTATE;
    return;
  }
  PetscOptionsObject->count = 1;
  *ierr                     = PetscOptionsRealArray_Private(PetscOptionsObject, copt, ctext, cman, currentvalue, n, set);
  if (*ierr) return;
  FREECHAR(opt, copt);
  FREECHAR(text, ctext);
  FREECHAR(man, cman);
}

PETSC_EXTERN void petscoptionsscalar_(char *opt, char *text, char *man, PetscScalar *currentvalue, PetscScalar *value, PetscBool *set, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T lenopt, PETSC_FORTRAN_CHARLEN_T lentext, PETSC_FORTRAN_CHARLEN_T lenman)
{
  char *copt, *ctext, *cman;

  FIXCHAR(opt, lenopt, copt);
  FIXCHAR(text, lentext, ctext);
  FIXCHAR(man, lenman, cman);
  if (!PetscOptionsObject) {
    *ierr = PETSC_ERR_ARG_WRONGSTATE;
    return;
  }
  PetscOptionsObject->count = 1;
  *ierr                     = PetscOptionsScalar_Private(PetscOptionsObject, copt, ctext, cman, *currentvalue, value, set);
  if (*ierr) return;
  FREECHAR(opt, copt);
  FREECHAR(text, ctext);
  FREECHAR(man, cman);
}

PETSC_EXTERN void petscoptionsscalararray_(char *opt, char *text, char *man, PetscScalar *currentvalue, PetscInt *n, PetscBool *set, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T lenopt, PETSC_FORTRAN_CHARLEN_T lentext, PETSC_FORTRAN_CHARLEN_T lenman)
{
  char *copt, *ctext, *cman;

  FIXCHAR(opt, lenopt, copt);
  FIXCHAR(text, lentext, ctext);
  FIXCHAR(man, lenman, cman);
  if (!PetscOptionsObject) {
    *ierr = PETSC_ERR_ARG_WRONGSTATE;
    return;
  }
  PetscOptionsObject->count = 1;
  *ierr                     = PetscOptionsScalarArray_Private(PetscOptionsObject, copt, ctext, cman, currentvalue, n, set);
  if (*ierr) return;
  FREECHAR(opt, copt);
  FREECHAR(text, ctext);
  FREECHAR(man, cman);
}

PETSC_EXTERN void petscoptionsstring_(char *opt, char *text, char *man, char *currentvalue, char *value, PetscBool *flg, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T lenopt, PETSC_FORTRAN_CHARLEN_T lentext, PETSC_FORTRAN_CHARLEN_T lenman, PETSC_FORTRAN_CHARLEN_T lencurrent, PETSC_FORTRAN_CHARLEN_T lenvalue)
{
  char     *copt, *ctext, *cman, *ccurrent;
  PetscBool flag;

  FIXCHAR(opt, lenopt, copt);
  FIXCHAR(text, lentext, ctext);
  FIXCHAR(man, lenman, cman);
  FIXCHAR(currentvalue, lencurrent, ccurrent);

  if (!PetscOptionsObject) {
    *ierr = PETSC_ERR_ARG_WRONGSTATE;
    return;
  }
  PetscOptionsObject->count = 1;

  *ierr = PetscOptionsString_Private(PetscOptionsObject, copt, ctext, cman, ccurrent, value, lenvalue - 1, &flag);
  if (*ierr) return;
  if (!FORTRANNULLBOOL(flg)) *flg = flag;
  FREECHAR(opt, copt);
  FREECHAR(text, ctext);
  FREECHAR(man, cman);
  FREECHAR(currentvalue, ccurrent);
  FIXRETURNCHAR(flag, value, lenvalue);
}

PETSC_EXTERN void petscoptionsgetenumprivate_(PetscOptions *opt, char *pre, char *name, const char *const *list, PetscEnum *ivalue, PetscBool *flg, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len1, PETSC_FORTRAN_CHARLEN_T len2)
{
  char     *c1, *c2;
  PetscBool flag;

  FIXCHAR(pre, len1, c1);
  FIXCHAR(name, len2, c2);
  *ierr = PetscOptionsGetEnum(*opt, c1, c2, list, ivalue, &flag);
  if (*ierr) return;
  if (!FORTRANNULLBOOL(flg)) *flg = flag;
  FREECHAR(pre, c1);
  FREECHAR(name, c2);
}

PETSC_EXTERN void petscoptionsgetstring_(PetscOptions *options, char *pre, char *name, char *string, PetscBool *flg, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len1, PETSC_FORTRAN_CHARLEN_T len2, PETSC_FORTRAN_CHARLEN_T len)
{
  char     *c1, *c2, *c3;
  size_t    len3;
  PetscBool flag;

  FIXCHAR(pre, len1, c1);
  FIXCHAR(name, len2, c2);
  c3   = string;
  len3 = len - 1;

  *ierr = PetscOptionsGetString(*options, c1, c2, c3, len3, &flag);
  if (*ierr) return;
  if (!FORTRANNULLBOOL(flg)) *flg = flag;
  FREECHAR(pre, c1);
  FREECHAR(name, c2);
  FIXRETURNCHAR(flag, string, len);
}
PETSC_EXTERN void petscsubcommgetparent_(PetscSubcomm *scomm, MPI_Fint *pcomm, int *ierr)
{
  MPI_Comm tcomm;

  *ierr  = PetscSubcommGetParent(*scomm, &tcomm);
  *pcomm = MPI_Comm_c2f(tcomm);
}

PETSC_EXTERN void petscsubcommgetcontiguousparent_(PetscSubcomm *scomm, MPI_Fint *pcomm, int *ierr)
{
  MPI_Comm tcomm;

  *ierr  = PetscSubcommGetContiguousParent(*scomm, &tcomm);
  *pcomm = MPI_Comm_c2f(tcomm);
}

PETSC_EXTERN void petscsubcommgetchild_(PetscSubcomm *scomm, MPI_Fint *ccomm, int *ierr)
{
  MPI_Comm tcomm;

  *ierr  = PetscSubcommGetChild(*scomm, &tcomm);
  *ccomm = MPI_Comm_c2f(tcomm);
}
