#include "private/fortranimpl.h"
#include "tao.h"

extern PetscBool TaoBeganPetsc;
extern PetscBool   TaoInitializeCalled;

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define taoinitialize_ TAOINITIALIZE
#define petscinitialize_ PETSCINITIALIZE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define taoinitialize_ taoinitialize
#define petscinitialize_ petscinitialize
#endif

EXTERN_C_BEGIN
#if defined(PETSC_USE_FORTRAN_MIXED_STR_ARG)
extern void petscinitialize_(CHAR,int,int*);
#else
extern void petscinitialize_(CHAR,int*,int);
#endif

void taoinitialize_(CHAR filename PETSC_MIXED_LEN(len), PetscErrorCode *ierr PETSC_END_LEN(len))
{
  *ierr = 1;
  if (TaoInitializeCalled) {*ierr = 0; return;}
  if (!PetscInitializeCalled) {
#if defined(PETSC_USE_FORTRAN_MIXED_STR)
    petscinitialize_(filename,len,ierr);
#else
    petscinitialize_(filename,ierr,len);
#endif
    if (*ierr) return;
    TaoBeganPetsc = PETSC_TRUE;
  }
  *ierr = TaoInitialize_DynamicLibraries();
  TaoInitializeCalled = PETSC_TRUE;
  *ierr = PetscInfo(0,"TAO succesfully started from Fortran\n");
  return;
}
