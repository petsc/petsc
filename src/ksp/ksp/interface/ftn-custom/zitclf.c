#include "zpetsc.h"
#include "petscksp.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define kspgetoptionsprefix_       KSPGETOPTIONSPREFIX
#define kspappendoptionsprefix_    KSPAPPENDOPTIONSPREFIX
#define kspsetoptionsprefix_       KSPSETOPTIONSPREFIX
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define kspgetoptionsprefix_       kspgetoptionsprefix
#define kspappendoptionsprefix_    kspappendoptionsprefix
#define kspsetoptionsprefix_       kspsetoptionsprefix
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL kspgetoptionsprefix_(KSP *ksp,CHAR prefix PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *tname;

  *ierr = KSPGetOptionsPrefix(*ksp,&tname);
#if defined(PETSC_USES_CPTOFCD)
  {
    char *t = _fcdtocp(prefix); int len1 = _fcdlen(prefix);
    *ierr = PetscStrncpy(t,tname,len1); if (*ierr) return;
  }
#else
  *ierr = PetscStrncpy(prefix,tname,len); if (*ierr) return;
#endif
}
void PETSC_STDCALL kspappendoptionsprefix_(KSP *ksp,CHAR prefix PETSC_MIXED_LEN(len),
                                           PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = KSPAppendOptionsPrefix(*ksp,t);
  FREECHAR(prefix,t);
}

void PETSC_STDCALL kspsetoptionsprefix_(KSP *ksp,CHAR prefix PETSC_MIXED_LEN(len),
                                        PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = KSPSetOptionsPrefix(*ksp,t);
  FREECHAR(prefix,t);
}


EXTERN_C_END
