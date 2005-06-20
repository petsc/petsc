
#include "zpetsc.h"
#include "petscsys.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define chkmemfortran_             CHKMEMFORTRAN
#define petscoffsetfortran_        PETSCOFFSETFORTRAN
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscoffsetfortran_        petscoffsetfortran     
#define chkmemfortran_             chkmemfortran
#endif

EXTERN_C_BEGIN

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
#if defined(PETSC_USES_CPTOFCD)
#include <fortran.h>

#define CHAR _fcd
#define FIXCHARNOMALLOC(a,n,b) \
{ \
  b = _fcdtocp(a); \
  n = _fcdlen (a); \
  if (b == PETSC_NULL_CHARACTER_Fortran) { \
      b = 0; \
  } else {  \
    while((n > 0) && (b[n-1] == ' ')) n--; \
    b = FIXCHARSTRING; \
    *ierr = PetscStrncpy(b,_fcdtocp(a),n); \
    if (*ierr) return; \
    b[n] = 0; \
  } \
}

#else

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

#endif

void PETSC_STDCALL chkmemfortran_(int *line,CHAR file PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *c1;

  FIXCHARNOMALLOC(file,len,c1);
  *ierr = PetscMallocValidate(*line,"Userfunction",c1," ");
}


EXTERN_C_END


