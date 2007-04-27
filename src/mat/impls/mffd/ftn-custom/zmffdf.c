#include "zpetsc.h"
#include "petscsnes.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define matmffdsetfunction_            MATMFFDSETFUNCTION
#define matmffdsettype_                MATMFFDSETTYPE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define matmffdsetfunction_            matmffdsetfunction
#define matmffdsettype_                matmffdsettype
#endif

EXTERN_C_BEGIN
static void (PETSC_STDCALL *f11)(void*,Vec*,Vec*,PetscErrorCode*);
EXTERN_C_END

static PetscErrorCode ourmatmffdfunction(void *ctx,Vec x,Vec f)
{
  PetscErrorCode ierr = 0;
  (*f11)(ctx,&x,&f,&ierr);CHKERRQ(ierr);
  return 0;
}

EXTERN_C_BEGIN
void PETSC_STDCALL matmffdsetfunction_(Mat *mat,void (PETSC_STDCALL *func)(void*,Vec*,Vec*,PetscErrorCode*),
                      void *ctx,PetscErrorCode *ierr)
{
  f11 = func;
  CHKFORTRANNULLOBJECT(ctx);
  *ierr = MatMFFDSetFunction(*mat,ourmatmffdfunction,ctx);
}

void PETSC_STDCALL matmffdsettype_(Mat *mat,CHAR ftype PETSC_MIXED_LEN(len),
                                     PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;
  FIXCHAR(ftype,len,t);
  *ierr = MatMFFDSetType(*mat,t);
  FREECHAR(ftype,t);
}

EXTERN_C_END
