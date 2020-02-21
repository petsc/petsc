

#include <petsc/private/fortranimpl.h>
#include <petscbag.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscbagregisterenumprivate_        PETSCBAGREGISTERENUMPRIVATE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscbagregisterenumprivate_        petscbagregisterenumprivate
#endif

/* ---------------------------------------------------------------------*/


PETSC_EXTERN void petscbagregisterenumprivate_(PetscBag *bag,void *addr,const char *const*list,
            PetscEnum *def,char* name,char* help,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len1,PETSC_FORTRAN_CHARLEN_T len2)
{
  char *c1,*c2;

  FIXCHAR(name,len1,c1);
  FIXCHAR(help,len2,c2);
  *ierr = PetscBagRegisterEnum(*bag,addr,list,*def,c1,c2);if (*ierr) return;
  FREECHAR(name,c1);
  FREECHAR(help,c2);
}


