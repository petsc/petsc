/*$Id: pmainf.c,v 1.1 1999/11/14 00:58:48 bsmith Exp bsmith $*/
/*
   Provides a simple main program that initializes PETSc and then
   calls the FORTRAN routine PetscMain()

   Currently not supported
*/
#include "petsc.h"        /*I  "petsc.h"   I*/

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define petscmain_ PETSCMAIN
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscmain_ petscmain
#endif
extern void petscmain_(int*);

#undef __FUNC__  
#define __FUNC__ "main"
int main(int argc,char **args)
{
  int ierr;

  PetscInitialize(&argc,&args,PETSC_NULL,PETSC_NULL);
  petscmain_(&ierr);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
