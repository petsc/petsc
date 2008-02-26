#include "private/fortranimpl.h"
#include "petscmat.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define matpartitioningpartysetglobal_   MATPARTITIONINGPARTYSETGLOBAL
#define matpartitioningpartysetlocal_    MATPARTITIONINGPARTYSETLOCAL
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define matpartitioningpartysetglobal_   matpartitioningpartysetglobal
#define matpartitioningpartysetlocal_    matpartitioningpartysetlocal
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL matpartitioningpartysetglobal_(MatPartitioning *part,CHAR method PETSC_MIXED_LEN(len),
                                           PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;
  FIXCHAR(method,len,t);
  *ierr = MatPartitioningPartySetGlobal(*part,t);
  FREECHAR(method,t);
}

void PETSC_STDCALL matpartitioningpartysetlocal_(MatPartitioning *part,CHAR method PETSC_MIXED_LEN(len),
                                           PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;
  FIXCHAR(method,len,t);
  *ierr = MatPartitioningPartySetLocal(*part,t);
  FREECHAR(method,t);
}

EXTERN_C_END
