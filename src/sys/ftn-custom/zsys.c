
#include <petsc/private/fortranimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define chkmemfortran_              CHKMEMFORTRAN
#define petscoffsetfortran_         PETSCOFFSETFORTRAN
#define petscobjectstateincrease_   PETSCOBJECTSTATEINCREASE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscoffsetfortran_         petscoffsetfortran
#define chkmemfortran_              chkmemfortran
#define flush__                     flush_
#define petscobjectstateincrease_   petscobjectstateincrease
#endif

PETSC_EXTERN void petscobjectstateincrease_(PetscObject *obj, PetscErrorCode *ierr)
{
  *ierr = PetscObjectStateIncrease(*obj);
}

#if defined(PETSC_MISSING_FORTRAN_FLUSH_)
void flush__(int unit)
{
}
#endif


PETSC_EXTERN void petscoffsetfortran_(PetscScalar *x,PetscScalar *y,size_t *shift,PetscErrorCode *ierr)
{
  *ierr  = 0;
  *shift = y - x;
}

/* ---------------------------------------------------------------------------------*/
/*
        This version does not do a malloc
*/
static char FIXCHARSTRING[1024];

#define FIXCHARNOMALLOC(a,n,b) \
{\
  if (a == PETSC_NULL_CHARACTER_Fortran) { \
    b = a = 0; \
  } else { \
    while ((n > 0) && (a[n-1] == ' ')) n--; \
    if (a[n] != 0) { \
      b = FIXCHARSTRING; \
      *ierr = PetscStrncpy(b,a,n+1); \
      if (*ierr) return; \
    } else b = a;\
  } \
}

PETSC_EXTERN void chkmemfortran_(int *line,char* file,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *c1;

  FIXCHARNOMALLOC(file,len,c1);
  *ierr = PetscMallocValidate(*line,"Userfunction",c1);
}



