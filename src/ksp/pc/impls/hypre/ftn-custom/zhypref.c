#include <petsc/private/fortranimpl.h>
#include <petscpc.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define pchypresettype_            PCHYPRESETTYPE
#define pchypregettype_            PCHYPREGETTYPE
#define pcmggalerkinsetmatproductalgorithm_ PCMGGALERKINSETMATPRODUCTALGORITHM
#define pcmggalerkingetmatproductalgorithm_ PCMGGALERKINGETMATPRODUCTALGORITHM
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define pchypresettype_            pchypresettype
#define pchypregettype_            pchypregettype
#define pcmggalerkinsetmatproductalgorithm_ pcmggalerkinsetmatproductalgorithm
#define pcmggalerkingetmatproductalgorithm_ pcmggalerkingetmatproductalgorithm
#endif

PETSC_EXTERN void pchypresettype_(PC *pc, char* name,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(name,len,t);
  *ierr = PCHYPRESetType(*pc,t);if (*ierr) return;
  FREECHAR(name,t);
}

PETSC_EXTERN void pchypregettype_(PC *pc,char* name,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  const char *tname;

  *ierr = PCHYPREGetType(*pc,&tname);if (*ierr) return;
  *ierr = PetscStrncpy(name,tname,len);if (*ierr) return;
  FIXRETURNCHAR(PETSC_TRUE,name,len);
}

PETSC_EXTERN void pcmggalerkinsetmatproductalgorithm_(PC *pc, char* name,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(name,len,t);
  *ierr = PCMGGalerkinSetMatProductAlgorithm(*pc,t);if (*ierr) return;
  FREECHAR(name,t);
}

PETSC_EXTERN void pcmggalerkingetmatproductalgorithm_(PC *pc,char* name,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  const char *tname;

  *ierr = PCMGGalerkinGetMatProductAlgorithm(*pc,&tname);if (*ierr) return;
  *ierr = PetscStrncpy(name,tname,len);if (*ierr) return;
  FIXRETURNCHAR(PETSC_TRUE,name,len);
}

