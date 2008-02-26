#include "private/fortranimpl.h"
#include "petscmat.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define matpartitioningscotchsetstrategy_ MATPARTITIONINGSCOTCHSETSTRATEGY
#define matpartitioningscotchsetarch_    MATPARTITIONINGSCOTCHSETARCH
#define matpartitioningscotchsethostlist_ MATPARTITIONINGSCOTCHSETHOSTLIST
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define matpartitioningscotchsetstrategy_ matpartitioningscotchsetstrategy
#define matpartitioningscotchsetarch_    matpartitioningscotchsetarch
#define matpartitioningscotchsethostlist_ matpartitioningscotchsethostlist
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL matpartitioningscotchsetstrategy_(MatPartitioning *part,CHAR strategy PETSC_MIXED_LEN(len),
                                           PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;
  FIXCHAR(strategy,len,t);
  *ierr = MatPartitioningScotchSetStrategy(*part,t);
  FREECHAR(strategy,t);
}

void PETSC_STDCALL matpartitioningscotchsetarch_(MatPartitioning *part,CHAR filename PETSC_MIXED_LEN(len),
                                           PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;
  FIXCHAR(filename,len,t);
  *ierr = MatPartitioningScotchSetArch(*part,t);
  FREECHAR(filename,t);
}
void PETSC_STDCALL matpartitioningscotchsethostlist_(MatPartitioning *part,CHAR filename PETSC_MIXED_LEN(len),
                                           PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;
  FIXCHAR(filename,len,t);
  *ierr = MatPartitioningScotchSetHostList(*part,t);
  FREECHAR(filename,t);
}
EXTERN_C_END
