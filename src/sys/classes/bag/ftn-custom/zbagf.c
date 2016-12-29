

#include <petsc/private/fortranimpl.h>
#include <petscbag.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscbagregisterenumprivate_        PETSCBAGREGISTERENUMPRIVATE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscbagregisterenumprivate_        petscbagregisterenumprivate
#endif

/* ---------------------------------------------------------------------*/


PETSC_EXTERN void PETSC_STDCALL petscbagregisterenumprivate_(PetscBag *bag,void *addr,const char *const*list,
            PetscEnum *def,char* name PETSC_MIXED_LEN(len1),char* help PETSC_MIXED_LEN(len2),PetscErrorCode *ierr PETSC_END_LEN(len1) PETSC_END_LEN(len2))
{
  char *c1,*c2;

  FIXCHAR(name,len1,c1);
  FIXCHAR(help,len2,c2);
  *ierr = PetscBagRegisterEnum(*bag,addr,list,*def,c1,c2);
  FREECHAR(name,c1);
  FREECHAR(help,c2);
}


