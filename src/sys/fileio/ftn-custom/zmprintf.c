#include "zpetsc.h"
#include "petsc.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscfprintf_              PETSCFPRINTF
#define petscprintf_               PETSCPRINTF
#define petscsynchronizedfprintf_  PETSCSYNCHRONIZEDFPRINTF
#define petscsynchronizedprintf_   PETSCSYNCHRONIZEDPRINTF
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscfprintf_                 petscfprintf
#define petscprintf_                  petscprintf
#define petscsynchronizedfprintf_     petscsynchronizedfprintf
#define petscsynchronizedprintf_      petscsynchronizedprintf
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL petscfprintf_(MPI_Comm *comm,FILE **file,CHAR fname PETSC_MIXED_LEN(len1),PetscErrorCode *ierr PETSC_END_LEN(len1))
{
  char *c1;

  FIXCHAR(fname,len1,c1);
  *ierr = PetscFPrintf((MPI_Comm)PetscToPointerComm(*comm),*file,c1);
  FREECHAR(fname,c1);
}

void PETSC_STDCALL petscprintf_(MPI_Comm *comm,CHAR fname PETSC_MIXED_LEN(len1),PetscErrorCode *ierr PETSC_END_LEN(len1))
{
  char *c1;

  FIXCHAR(fname,len1,c1);
  *ierr = PetscPrintf((MPI_Comm)PetscToPointerComm(*comm),c1);
  FREECHAR(fname,c1);
}

void PETSC_STDCALL petscsynchronizedfprintf_(MPI_Comm *comm,FILE **file,CHAR fname PETSC_MIXED_LEN(len1),PetscErrorCode *ierr PETSC_END_LEN(len1))
{
  char *c1;

  FIXCHAR(fname,len1,c1);
  *ierr = PetscSynchronizedFPrintf((MPI_Comm)PetscToPointerComm(*comm),*file,c1);
  FREECHAR(fname,c1);
}

void PETSC_STDCALL petscsynchronizedprintf_(MPI_Comm *comm,CHAR fname PETSC_MIXED_LEN(len1),PetscErrorCode *ierr PETSC_END_LEN(len1))
{
  char *c1;

  FIXCHAR(fname,len1,c1);
  *ierr = PetscSynchronizedPrintf((MPI_Comm)PetscToPointerComm(*comm),c1);
  FREECHAR(fname,c1);
}

EXTERN_C_END
