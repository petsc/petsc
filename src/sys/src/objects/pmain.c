/*$Id: pmain.c,v 1.1 1999/11/14 00:42:58 bsmith Exp bsmith $*/
/*
   Provides a simple main program that initializes PETSc and then
   calls PetscMain()

   CURRENTLY NOT SUPPORTED 
*/

#include "petscconfig.h"
#include "petsc.h"        /*I  "petsc.h"   I*/

extern int PetscMain(void);

#undef __FUNC__  
#define __FUNC__ "main"
int main(int argc,char **args)
{
  int ierr;

  PetscInitialize(&argc,&args,PETSC_NULL,PETSC_NULL);
  ierr = PetscMain();CHKERRA(ierr);
  PetscFinalize();
  return 0;
}

