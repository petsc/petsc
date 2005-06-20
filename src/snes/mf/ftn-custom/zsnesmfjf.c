#include "zpetsc.h"
#include "petscsnes.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define matsnesmfsetfunction_            MATSNESMFSETFUNCTION
#define matsnesmfsettype_                MATSNESMFSETTYPE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define matsnesmfsetfunction_            matsnesmfsetfunction
#define matsnesmfsettype_                matsnesmfsettype
#endif

EXTERN_C_BEGIN
static void (PETSC_STDCALL *f11)(SNES*,Vec*,Vec*,void*,PetscErrorCode*);
EXTERN_C_END

static PetscErrorCode ourmatsnesmffunction(SNES snes,Vec x,Vec f,void *ctx)
{
  PetscErrorCode ierr = 0;
  (*f11)(&snes,&x,&f,ctx,&ierr);CHKERRQ(ierr);
  return 0;
}

EXTERN_C_BEGIN
void PETSC_STDCALL matsnesmfsetfunction_(Mat *mat,Vec *r,void (PETSC_STDCALL *func)(SNES*,Vec*,Vec*,void*,PetscErrorCode*),
                      void *ctx,PetscErrorCode *ierr){
  f11 = func;
  CHKFORTRANNULLOBJECT(ctx);
  *ierr = MatSNESMFSetFunction(*mat,*r,ourmatsnesmffunction,ctx);
}

void PETSC_STDCALL matsnesmfsettype_(Mat *mat,CHAR ftype PETSC_MIXED_LEN(len),
                                     PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;
  FIXCHAR(ftype,len,t);
  *ierr = MatSNESMFSetType(*mat,t);
  FREECHAR(ftype,t);
}

EXTERN_C_END
