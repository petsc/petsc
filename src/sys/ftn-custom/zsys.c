
#include "private/fortranimpl.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define chkmemfortran_             CHKMEMFORTRAN
#define petscoffsetfortran_        PETSCOFFSETFORTRAN
#define petscobjectstateincrease_  PETSCOBJECTSTATEINCREASE
#define petscobjectstatedecrease_  PETSCOBJECTSTATEDECREASE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscoffsetfortran_        petscoffsetfortran     
#define chkmemfortran_             chkmemfortran
#define flush__                    flush_
#define petscobjectstateincrease_  petscobjectstateincrease
#define petscobjectstatedecrease_  petscobjectstatedecrease
#endif


EXTERN_C_BEGIN

void PETSC_STDCALL  petscobjectstateincrease_(PetscObject *obj, PetscErrorCode *ierr )
{
  *ierr = PetscObjectStateIncrease(*obj);
}
void PETSC_STDCALL  petscobjectstatedecrease_(PetscObject *obj, PetscErrorCode *ierr ){
  *ierr = PetscObjectStateDecrease(*obj);
}


#if defined(PETSC_MISSING_FORTRAN_FLUSH_)
void flush__(int unit)
{
}
#endif


void PETSC_STDCALL petscoffsetfortran_(PetscScalar *x,PetscScalar *y,size_t *shift,PetscErrorCode *ierr)
{
  *ierr = 0;
  *shift = y - x;
}

/* ---------------------------------------------------------------------------------*/
/*
        This version does not do a malloc 
*/
static char FIXCHARSTRING[1024];

#define CHAR char*
#define FIXCHARNOMALLOC(a,n,b) \
{\
  if (a == PETSC_NULL_CHARACTER_Fortran) { \
    b = a = 0; \
  } else { \
    while((n > 0) && (a[n-1] == ' ')) n--; \
    if (a[n] != 0) { \
      b = FIXCHARSTRING; \
      *ierr = PetscStrncpy(b,a,n); \
      if (*ierr) return; \
      b[n] = 0; \
    } else b = a;\
  } \
}

void PETSC_STDCALL chkmemfortran_(int *line,CHAR file PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *c1;

  FIXCHARNOMALLOC(file,len,c1);
  *ierr = PetscMallocValidate(*line,"Userfunction",c1," ");
}


EXTERN_C_END


