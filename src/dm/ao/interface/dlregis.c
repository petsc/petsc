#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dlregis.c,v 1.1 2000/01/10 06:34:46 knepley Exp $";
#endif

#include "petscao.h"
#include "petscda.h"

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PetscDLLibraryRegister"
/*
  PetscDLLibraryRegister - This function is called when the dynamic library it is in is opened.

  This one registers all the mesh generators and partitioners that are in
  the basic DM library.

  Input Parameter:
  path - library path
*/
int PetscDLLibraryRegister(char *path)
{
  int ierr;

  ierr = PetscInitializeNoArguments();
  if (ierr) return(1);

  /*
      If we got here then PETSc was properly loaded
  */
#if PETSC_USE_NEW_LOGGING
  ierr = PetscLogClassRegister(&AO_COOKIE,     "Application Order");                                      CHKERRQ(ierr);
  ierr = PetscLogClassRegister(&AODATA_COOKIE, "Application Data");                                       CHKERRQ(ierr);
  ierr = PetscLogClassRegister(&DA_COOKIE,     "Distributed array");                                      CHKERRQ(ierr);
#endif
  ierr = AOSerializeRegisterAll(path);                                                                    CHKERRQ(ierr);
  return(0);
}
EXTERN_C_END

/* --------------------------------------------------------------------------*/
static char *contents = "PETSc Distributed Structures library, includes\n\
Application Orderings, Application Data, and Distributed Arrays";

static char *authors = PETSC_AUTHOR_INFO;
static char *version = PETSC_VERSION_NUMBER;

/* --------------------------------------------------------------------------*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PetscDLLibraryInfo"
int PetscDLLibraryInfo(char *path,char *type,char **mess) 
{ 
  PetscTruth iscontents, isauthors, isversion;
  int        ierr;

  ierr = PetscStrcmp(type, "Contents", &iscontents);                                                      CHKERRQ(ierr);
  ierr = PetscStrcmp(type, "Authors",  &isauthors);                                                       CHKERRQ(ierr);
  ierr = PetscStrcmp(type, "Version",  &isversion);                                                       CHKERRQ(ierr);
  if      (iscontents == PETSC_TRUE) *mess = contents;
  else if (isauthors  == PETSC_TRUE) *mess = authors;
  else if (isversion  == PETSC_TRUE) *mess = version;
  else                               *mess = PETSC_NULL;

  return(0);
}
EXTERN_C_END


