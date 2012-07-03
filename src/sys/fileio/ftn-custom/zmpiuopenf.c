#include <petsc-private/fortranimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscfopen_                PETSCFOPEN
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscfopen_                   petscfopen
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL petscfopen_(MPI_Comm *comm,CHAR fname PETSC_MIXED_LEN(len1),CHAR fmode PETSC_MIXED_LEN(len2),
                               FILE **file,PetscErrorCode *ierr PETSC_END_LEN(len1) PETSC_END_LEN(len2))
{
  char *c1,*c2;

  FIXCHAR(fname,len1,c1);
  FIXCHAR(fmode,len2,c2);
  *ierr = PetscFOpen(MPI_Comm_f2c(*(MPI_Fint *)&*comm),c1,c2,file);
  FREECHAR(fname,c1);
  FREECHAR(fmode,c2);
}

EXTERN_C_END
