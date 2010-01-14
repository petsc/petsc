#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dlregis.c,v 1.1 2000/01/10 03:10:37 knepley Exp $";
#endif

#include "bilinear.h"

#ifdef PETSC_USE_DYNAMIC_LIBRARIES
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PetscDLLibraryRegister"
/*
  PetscDLLibraryRegister - This function is called when the dynamic library it is in is opened.

  This one registers all the solvers in the Bilinear library.

  Input Parameter:
  path - library path
*/
int PetscDLLibraryRegister(char *path) {
  int ierr;

  ierr = PetscInitializeNoArguments();
  if (ierr) return(1);
  /*
      If we got here then PETSc was properly loaded
  */
  ierr = BilinearInitializePackage(path);                                                                 CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

/* --------------------------------------------------------------------------*/
static char *contents = "PETSc Bilinear Operator library";

static char *authors = "Matt Knepley    knepley@cs.purdue.edu\n\
  http://www.cs.purdue.edu/homes/knepley/comp_fluid";
static char *version = PETSC_VERSION_NUMBER;

/* --------------------------------------------------------------------------*/
EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "DLLibraryInfo"
int DLLibraryInfo(char *path,char *type,char **mess) 
{ 
  PetscTruth iscontents, isauthors, isversion;
  int        ierr;

  ierr = PetscStrcmp(type, "Contents", &iscontents);                                                     CHKERRQ(ierr);
  ierr = PetscStrcmp(type, "Authors",  &isauthors);                                                      CHKERRQ(ierr);
  ierr = PetscStrcmp(type, "Version",  &isversion);                                                      CHKERRQ(ierr);
  if      (iscontents == PETSC_TRUE) *mess = contents;
  else if (isauthors  == PETSC_TRUE) *mess = authors;
  else if (isversion  == PETSC_TRUE) *mess = version;
  else                               *mess = PETSC_NULL;

  return(0);
}
EXTERN_C_END
#endif /* PETSC_USE_DYNAMIC_LIBRARIES */
