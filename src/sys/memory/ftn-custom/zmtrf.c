#include "zpetsc.h"
#include "petsc.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscmallocdump_           PETSCMALLOCDUMP
#define petscmallocdumplog_        PETSCMALLOCDUMPLOG
#define petscmallocvalidate_       PETSCMALLOCVALIDATE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscmallocdump_           petscmallocdump
#define petscmallocdumplog_        petscmallocdumplog
#define petscmallocvalidate_       petscmallocvalidate
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL  petscmallocdump_(PetscErrorCode *ierr)
{
  *ierr = PetscMallocDump(stdout);
}
void PETSC_STDCALL petscmallocdumplog_(PetscErrorCode *ierr)
{
  *ierr = PetscMallocDumpLog(stdout);
}

void PETSC_STDCALL petscmallocvalidate_(PetscErrorCode *ierr)
{
  *ierr = PetscMallocValidate(0,"Unknown Fortran",0,0);
}

EXTERN_C_END
