
#include "private/fortranimpl.h"
#include "petscda.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dasetfieldname_              DASETFIELDNAME
#define dagetfieldname_              DAGETFIELDNAME
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dasetfieldname_              dasetfieldname
#define dagetfieldname_              dagetfieldname
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL dasetfieldname_(DA *da,PetscInt *nf,CHAR name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;
  FIXCHAR(name,len,t);
  *ierr = DASetFieldName(*da,*nf,t);
  FREECHAR(name,t);
}

void PETSC_STDCALL dagetfieldname_(DA *da,PetscInt *nf,CHAR name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *tname;

  *ierr = DAGetFieldName(*da,*nf,&tname);
  *ierr = PetscStrncpy(name,tname,len);
}

EXTERN_C_END
