#include <petsc/private/fortranimpl.h>
#include <petscmat.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define matpartitioningpartysetglobal_   MATPARTITIONINGPARTYSETGLOBAL
#define matpartitioningpartysetlocal_    MATPARTITIONINGPARTYSETLOCAL
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define matpartitioningpartysetglobal_   matpartitioningpartysetglobal
#define matpartitioningpartysetlocal_    matpartitioningpartysetlocal
#endif

PETSC_EXTERN void PETSC_STDCALL matpartitioningpartysetglobal_(MatPartitioning *part,char* method PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;
  FIXCHAR(method,len,t);
  *ierr = MatPartitioningPartySetGlobal(*part,t);
  FREECHAR(method,t);
}

PETSC_EXTERN void PETSC_STDCALL matpartitioningpartysetlocal_(MatPartitioning *part,char* method PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;
  FIXCHAR(method,len,t);
  *ierr = MatPartitioningPartySetLocal(*part,t);
  FREECHAR(method,t);
}

