#include <petsc/private/fortranimpl.h>
#include <petscksp.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define pcgamggettype_                PCGAMGGETTYPE
#define pcgamgsettype_                PCGAMGSETTYPE
#define pcgamgsetesteigksptype_       PCGAMGSETESTEIGKSPTYPE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define pcgamggettype_                pcgamggettype
#define pcgamgsettype_                pcgamgsettype
#define pcgamgsetesteigksptype_       pcgamgsetesteigksptype
#endif

PETSC_EXTERN void PETSC_STDCALL pcgamggettype_(PC *pc,char* name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *tname;

  *ierr = PCGAMGGetType(*pc,&tname);if (*ierr) return;
  *ierr = PetscStrncpy(name,tname,len);
  FIXRETURNCHAR(PETSC_TRUE,name,len);

}

PETSC_EXTERN void PETSC_STDCALL pcgamgsettype_(PC *pc,char* type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = PCGAMGSetType(*pc,t);if (*ierr) return;
  FREECHAR(type,t);
}

PETSC_EXTERN void PETSC_STDCALL pcgamgsetesteigksptype_(PC *pc,char* type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = PCGAMGSetEstEigKSPType(*pc,t);if (*ierr) return;
  FREECHAR(type,t);
}



