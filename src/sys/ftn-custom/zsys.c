#include <petsc/private/ftnimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define chkmemfortran_                     CHKMEMFORTRAN
  #define petscoffsetfortran_                PETSCOFFSETFORTRAN
  #define petscobjectstateincrease_          PETSCOBJECTSTATEINCREASE
  #define petsccienabledportableerroroutput_ PETSCCIENABLEDPORTABLEERROROUTPUT
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define chkmemfortran_                     chkmemfortran
  #define petscoffsetfortran_                petscoffsetfortran
  #define petscobjectstateincrease_          petscobjectstateincrease
  #define petsccienabledportableerroroutput_ petsccienabledportableerroroutput
#endif

PETSC_EXTERN void petsccienabledportableerroroutput_(PetscMPIInt *cienabled)
{
  *cienabled = PetscCIEnabledPortableErrorOutput ? 1 : 0;
}

PETSC_EXTERN void petscobjectstateincrease_(PetscObject *obj, PetscErrorCode *ierr)
{
  *ierr = PetscObjectStateIncrease(*obj);
}

PETSC_EXTERN void petscoffsetfortran_(PetscScalar *x, PetscScalar *y, size_t *shift, PetscErrorCode *ierr)
{
  *ierr  = PETSC_SUCCESS;
  *shift = y - x;
}

/* ---------------------------------------------------------------------------------*/
/*
        This version does not do a malloc
*/
static char FIXCHARSTRING[1024];

#define FIXCHARNOMALLOC(a, n, b) \
  do { \
    if (a == PETSC_NULL_CHARACTER_Fortran) { \
      b = a = NULL; \
    } else { \
      while ((n > 0) && (a[n - 1] == ' ')) n--; \
      if (a[n] != 0) { \
        b     = FIXCHARSTRING; \
        *ierr = PetscStrncpy(b, a, n + 1); \
        if (*ierr) return; \
      } else b = a; \
    } \
  } while (0)

PETSC_EXTERN void chkmemfortran_(int *line, char *file, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *c1;

  FIXCHARNOMALLOC(file, len, c1);
  *ierr = PetscMallocValidate(*line, "Userfunction", c1);
}
