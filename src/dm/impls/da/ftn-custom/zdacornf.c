
#include <petsc-private/fortranimpl.h>
#include <petscdmda.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dmdasetfieldname_              DMDASETFIELDNAME
#define dmdagetfieldname_              DMDAGETFIELDNAME
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dmdasetfieldname_              dmdasetfieldname
#define dmdagetfieldname_              dmdagetfieldname
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL dmdasetfieldname_(DM *da,PetscInt *nf,CHAR name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;
  FIXCHAR(name,len,t);
  *ierr = DMDASetFieldName(*da,*nf,t);
  FREECHAR(name,t);
}

void PETSC_STDCALL dmdagetfieldname_(DM *da,PetscInt *nf,CHAR name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *tname;

  *ierr = DMDAGetFieldName(*da,*nf,&tname);
  *ierr = PetscStrncpy(name,tname,len);
}

EXTERN_C_END
