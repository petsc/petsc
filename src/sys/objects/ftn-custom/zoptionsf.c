/*
  This file contains Fortran stubs for Options routines.
  These are not generated automatically since they require passing strings
  between Fortran and C.
*/

#include <petsc/private/fortranimpl.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petscoptionsbegin_               PETSCOPTIONSBEGIN
  #define petscoptionsend_                 PETSCOPTIONSEND
  #define petscoptionsbool_                PETSCOPTIONSBOOL
  #define petscoptionsboolarray_           PETSCOPTIONSBOOLARRAY
  #define petscoptionsenumprivate_         PETSCOPTIONSENUMPRIVATE
  #define petscoptionsint_                 PETSCOPTIONSINT
  #define petscoptionsintarray_            PETSCOPTIONSINTARRAY
  #define petscoptionsreal_                PETSCOPTIONSREAL
  #define petscoptionsrealarray_           PETSCOPTIONSREALARRAY
  #define petscoptionsscalar_              PETSCOPTIONSSCALAR
  #define petscoptionsscalararray_         PETSCOPTIONSSCALARARRAY
  #define petscoptionsstring_              PETSCOPTIONSSTRING
  #define petscsubcommview_                PETSCSUBCOMMVIEW
  #define petscsubcommgetparent_           PETSCSUBCOMMGETPARENT
  #define petscsubcommgetcontiguousparent_ PETSCSUBCOMMGETCONTIGUOUSPARENT
  #define petscsubcommgetchild_            PETSCSUBCOMMGETCHILD
  #define petscoptionsallused_             PETSCOPTIONSALLUSED
  #define petscoptionsgetenumprivate_      PETSCOPTIONSGETENUMPRIVATE
  #define petscoptionsgetbool_             PETSCOPTIONSGETBOOL
  #define petscoptionsgetboolarray_        PETSCOPTIONSGETBOOLARRAY
  #define petscoptionsgetintarray_         PETSCOPTIONSGETINTARRAY
  #define petscoptionssetvalue_            PETSCOPTIONSSETVALUE
  #define petscoptionsclearvalue_          PETSCOPTIONSCLEARVALUE
  #define petscoptionshasname_             PETSCOPTIONSHASNAME
  #define petscoptionsgetint_              PETSCOPTIONSGETINT
  #define petscoptionsgetreal_             PETSCOPTIONSGETREAL
  #define petscoptionsgetscalar_           PETSCOPTIONSGETSCALAR
  #define petscoptionsgetscalararray_      PETSCOPTIONSGETSCALARARRAY
  #define petscoptionsgetrealarray_        PETSCOPTIONSGETREALARRAY
  #define petscoptionsgetstring_           PETSCOPTIONSGETSTRING
  #define petscgetprogramname              PETSCGETPROGRAMNAME
  #define petscoptionsinsertfile_          PETSCOPTIONSINSERTFILE
  #define petscoptionsclear_               PETSCOPTIONSCLEAR
  #define petscoptionsinsertstring_        PETSCOPTIONSINSERTSTRING
  #define petscoptionsview_                PETSCOPTIONSVIEW
  #define petscoptionsleft_                PETSCOPTIONSLEFT
  #define petscobjectviewfromoptions_      PETSCOBJECTVIEWFROMOPTIONS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define petscoptionsbegin_               petscoptionsbegin
  #define petscoptionsend_                 petscoptionsend
  #define petscoptionsbool_                petscoptionsbool
  #define petscoptionsboolarray_           petscoptionsboolarray
  #define petscoptionsenumprivate_         petscoptionsenumprivate
  #define petscoptionsint_                 petscoptionsint
  #define petscoptionsintarray_            petscoptionsintarray
  #define petscoptionsreal_                petscoptionsreal
  #define petscoptionsrealarray_           petscoptionsrealarray
  #define petscoptionsscalar_              petscoptionsscalar
  #define petscoptionsscalararray_         petscoptionsscalararray
  #define petscoptionsstring_              petscoptionsstring
  #define petscsubcommview_                petscsubcommview
  #define petscsubcommgetparent_           petscsubcommgetparent
  #define petscsubcommgetcontiguousparent_ petscsubcommgetcontiguousparent
  #define petscsubcommgetchild_            petscsubcommgetchild
  #define petscoptionsallused_             petscoptionsallused
  #define petscoptionsgetenumprivate_      petscoptionsgetenumprivate
  #define petscoptionsgetbool_             petscoptionsgetbool
  #define petscoptionsgetboolarray_        petscoptionsgetboolarray
  #define petscoptionssetvalue_            petscoptionssetvalue
  #define petscoptionsclearvalue_          petscoptionsclearvalue
  #define petscoptionshasname_             petscoptionshasname
  #define petscoptionsgetint_              petscoptionsgetint
  #define petscoptionsgetreal_             petscoptionsgetreal
  #define petscoptionsgetscalar_           petscoptionsgetscalar
  #define petscoptionsgetscalararray_      petscoptionsgetscalararray
  #define petscoptionsgetrealarray_        petscoptionsgetrealarray
  #define petscoptionsgetstring_           petscoptionsgetstring
  #define petscoptionsgetintarray_         petscoptionsgetintarray
  #define petscgetprogramname_             petscgetprogramname
  #define petscoptionsinsertfile_          petscoptionsinsertfile
  #define petscoptionsclear_               petscoptionsclear
  #define petscoptionsinsertstring_        petscoptionsinsertstring
  #define petscoptionsview_                petscoptionsview
  #define petscoptionsleft_                petscoptionsleft
  #define petscobjectviewfromoptions_      petscobjectviewfromoptions
#endif

static PetscOptionItems PetscOptionsObjectBase, *PetscOptionsObject = NULL;

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
  *ierr                     = PetscOptionsInt_Private(PetscOptionsObject, copt, ctext, cman, *currentvalue, value, set, PETSC_MIN_INT, PETSC_MAX_INT);
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

PETSC_EXTERN void petscoptionsinsertstring_(PetscOptions *options, char *file, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *c1;

  FIXCHAR(file, len, c1);
  *ierr = PetscOptionsInsertString(*options, c1);
  if (*ierr) return;
  FREECHAR(file, c1);
}

PETSC_EXTERN void petscoptionsinsertfile_(MPI_Fint *comm, PetscOptions *options, char *file, PetscBool *require, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *c1;

  FIXCHAR(file, len, c1);
  *ierr = PetscOptionsInsertFile(MPI_Comm_f2c(*comm), *options, c1, *require);
  if (*ierr) return;
  FREECHAR(file, c1);
}

PETSC_EXTERN void petscoptionssetvalue_(PetscOptions *options, char *name, char *value, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len1, PETSC_FORTRAN_CHARLEN_T len2)
{
  char *c1, *c2;

  FIXCHAR(name, len1, c1);
  FIXCHAR(value, len2, c2);
  *ierr = PetscOptionsSetValue(*options, c1, c2);
  if (*ierr) return;
  FREECHAR(name, c1);
  FREECHAR(value, c2);
}

PETSC_EXTERN void petscoptionsclear_(PetscOptions *options, PetscErrorCode *ierr)
{
  *ierr = PetscOptionsClear(*options);
}

PETSC_EXTERN void petscoptionsclearvalue_(PetscOptions *options, char *name, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *c1;

  FIXCHAR(name, len, c1);
  *ierr = PetscOptionsClearValue(*options, c1);
  if (*ierr) return;
  FREECHAR(name, c1);
}

PETSC_EXTERN void petscoptionshasname_(PetscOptions *options, char *pre, char *name, PetscBool *flg, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len1, PETSC_FORTRAN_CHARLEN_T len2)
{
  char *c1, *c2;

  FIXCHAR(pre, len1, c1);
  FIXCHAR(name, len2, c2);
  *ierr = PetscOptionsHasName(*options, c1, c2, flg);
  if (*ierr) return;
  FREECHAR(pre, c1);
  FREECHAR(name, c2);
}

PETSC_EXTERN void petscoptionsgetint_(PetscOptions *opt, char *pre, char *name, PetscInt *ivalue, PetscBool *flg, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len1, PETSC_FORTRAN_CHARLEN_T len2)
{
  char     *c1, *c2;
  PetscBool flag;

  FIXCHAR(pre, len1, c1);
  FIXCHAR(name, len2, c2);
  *ierr = PetscOptionsGetInt(*opt, c1, c2, ivalue, &flag);
  if (*ierr) return;
  if (!FORTRANNULLBOOL(flg)) *flg = flag;
  FREECHAR(pre, c1);
  FREECHAR(name, c2);
}

PETSC_EXTERN void petscoptionsgetenumprivate_(PetscOptions *options, char *pre, char *name, const char *const *list, PetscEnum *ivalue, PetscBool *flg, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len1, PETSC_FORTRAN_CHARLEN_T len2)
{
  char     *c1, *c2;
  PetscBool flag;

  FIXCHAR(pre, len1, c1);
  FIXCHAR(name, len2, c2);
  *ierr = PetscOptionsGetEnum(*options, c1, c2, list, ivalue, &flag);
  if (*ierr) return;
  if (!FORTRANNULLBOOL(flg)) *flg = flag;
  FREECHAR(pre, c1);
  FREECHAR(name, c2);
}

PETSC_EXTERN void petscoptionsgetbool_(PetscOptions *options, char *pre, char *name, PetscBool *ivalue, PetscBool *flg, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len1, PETSC_FORTRAN_CHARLEN_T len2)
{
  char     *c1, *c2;
  PetscBool flag;

  FIXCHAR(pre, len1, c1);
  FIXCHAR(name, len2, c2);
  *ierr = PetscOptionsGetBool(*options, c1, c2, ivalue, &flag);
  if (*ierr) return;
  if (!FORTRANNULLBOOL(flg)) *flg = flag;
  FREECHAR(pre, c1);
  FREECHAR(name, c2);
}

PETSC_EXTERN void petscoptionsgetboolarray_(PetscOptions *options, char *pre, char *name, PetscBool *dvalue, PetscInt *nmax, PetscBool *flg, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len1, PETSC_FORTRAN_CHARLEN_T len2)
{
  char     *c1, *c2;
  PetscBool flag;

  FIXCHAR(pre, len1, c1);
  FIXCHAR(name, len2, c2);
  *ierr = PetscOptionsGetBoolArray(*options, c1, c2, dvalue, nmax, &flag);
  if (*ierr) return;
  if (!FORTRANNULLBOOL(flg)) *flg = flag;
  FREECHAR(pre, c1);
  FREECHAR(name, c2);
}

PETSC_EXTERN void petscoptionsgetreal_(PetscOptions *options, char *pre, char *name, PetscReal *dvalue, PetscBool *flg, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len1, PETSC_FORTRAN_CHARLEN_T len2)
{
  char     *c1, *c2;
  PetscBool flag;

  FIXCHAR(pre, len1, c1);
  FIXCHAR(name, len2, c2);
  *ierr = PetscOptionsGetReal(*options, c1, c2, dvalue, &flag);
  if (*ierr) return;
  if (!FORTRANNULLBOOL(flg)) *flg = flag;
  FREECHAR(pre, c1);
  FREECHAR(name, c2);
}

PETSC_EXTERN void petscoptionsgetscalar_(PetscOptions *options, char *pre, char *name, PetscScalar *dvalue, PetscBool *flg, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len1, PETSC_FORTRAN_CHARLEN_T len2)
{
  char     *c1, *c2;
  PetscBool flag;

  FIXCHAR(pre, len1, c1);
  FIXCHAR(name, len2, c2);
  *ierr = PetscOptionsGetScalar(*options, c1, c2, dvalue, &flag);
  if (*ierr) return;
  if (!FORTRANNULLBOOL(flg)) *flg = flag;
  FREECHAR(pre, c1);
  FREECHAR(name, c2);
}

PETSC_EXTERN void petscoptionsgetscalararray_(PetscOptions *options, char *pre, char *name, PetscScalar *dvalue, PetscInt *nmax, PetscBool *flg, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len1, PETSC_FORTRAN_CHARLEN_T len2)
{
  char     *c1, *c2;
  PetscBool flag;

  FIXCHAR(pre, len1, c1);
  FIXCHAR(name, len2, c2);
  *ierr = PetscOptionsGetScalarArray(*options, c1, c2, dvalue, nmax, &flag);
  if (*ierr) return;
  if (!FORTRANNULLBOOL(flg)) *flg = flag;
  FREECHAR(pre, c1);
  FREECHAR(name, c2);
}

PETSC_EXTERN void petscoptionsgetrealarray_(PetscOptions *options, char *pre, char *name, PetscReal *dvalue, PetscInt *nmax, PetscBool *flg, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len1, PETSC_FORTRAN_CHARLEN_T len2)
{
  char     *c1, *c2;
  PetscBool flag;

  FIXCHAR(pre, len1, c1);
  FIXCHAR(name, len2, c2);
  *ierr = PetscOptionsGetRealArray(*options, c1, c2, dvalue, nmax, &flag);
  if (*ierr) return;
  if (!FORTRANNULLBOOL(flg)) *flg = flag;
  FREECHAR(pre, c1);
  FREECHAR(name, c2);
}

PETSC_EXTERN void petscoptionsgetintarray_(PetscOptions *options, char *pre, char *name, PetscInt *dvalue, PetscInt *nmax, PetscBool *flg, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len1, PETSC_FORTRAN_CHARLEN_T len2)
{
  char     *c1, *c2;
  PetscBool flag;

  FIXCHAR(pre, len1, c1);
  FIXCHAR(name, len2, c2);
  *ierr = PetscOptionsGetIntArray(*options, c1, c2, dvalue, nmax, &flag);
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

PETSC_EXTERN void petscgetprogramname_(char *name, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len_in)
{
  char  *tmp;
  size_t len;
  tmp   = name;
  len   = len_in - 1;
  *ierr = PetscGetProgramName(tmp, len);
  FIXRETURNCHAR(PETSC_TRUE, name, len_in);
}

PETSC_EXTERN void petscoptionsview_(PetscOptions *options, PetscViewer *vin, PetscErrorCode *ierr)
{
  PetscViewer v;

  PetscPatchDefaultViewers_Fortran(vin, v);
  *ierr = PetscOptionsView(*options, v);
}

PETSC_EXTERN void petscobjectviewfromoptions_(PetscObject *obj, PetscObject *bobj, char *option, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T loption)
{
  char *o;

  FIXCHAR(option, loption, o);
  CHKFORTRANNULLOBJECT(obj);
  *ierr = PetscObjectViewFromOptions(*obj, *bobj, o);
  if (*ierr) return;
  FREECHAR(option, o);
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

PETSC_EXTERN void petscsubcommview_(PetscSubcomm *psubcomm, PetscViewer *viewer, int *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer, v);
  *ierr = PetscSubcommView(*psubcomm, v);
}
