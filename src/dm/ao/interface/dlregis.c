#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dlregis.c,v 1.1 2000/01/10 06:34:46 knepley Exp $";
#endif

#include "petscao.h"
#include "petscda.h"

#undef __FUNCT__  
#define __FUNCT__ "DMInitializePackage"
/*@C
  DMInitializePackage - This function initializes everything in the DM package. It is called
  from PetscDLLibraryRegister() when using dynamic libraries, and on the first call to AOCreate()
  or DACreate() when using static libraries.

  Input Parameter:
  path - The dynamic library path, or PETSC_NULL

  Level: developer

.keywords: AO, DA, initialize, package
.seealso: PetscInitialize()
@*/
int DMInitializePackage(char *path) {
  static PetscTruth initialized = PETSC_FALSE;
  char              logList[256];
  char             *className;
  PetscTruth        opt;
  int               ierr;

  PetscFunctionBegin;
  if (initialized == PETSC_TRUE) PetscFunctionReturn(0);
  initialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscLogClassRegister(&AO_COOKIE,     "Application Order");                                      CHKERRQ(ierr);
  ierr = PetscLogClassRegister(&AODATA_COOKIE, "Application Data");                                       CHKERRQ(ierr);
  ierr = PetscLogClassRegister(&DA_COOKIE,     "Distributed array");                                      CHKERRQ(ierr);
  /* Register Constructors and Serializers */
  ierr = AOSerializeRegisterAll(path);                                                                    CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister(&AOEvents[AO_PetscToApplication], "AOPetscToApplication", AO_COOKIE);      CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&AOEvents[AO_ApplicationToPetsc], "AOApplicationToPetsc", AO_COOKIE);      CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&DAEvents[DA_GlobalToLocal],      "DAGlobalToLocal",      DA_COOKIE);      CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&DAEvents[DA_LocalToGlobal],      "DALocalToGlobal",      DA_COOKIE);      CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&DAEvents[DA_LocalADFunction],    "DALocalADFunc",        DA_COOKIE);      CHKERRQ(ierr);
  /* Process info exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-log_info_exclude", logList, 256, &opt);                      CHKERRQ(ierr);
  if (opt == PETSC_TRUE) {
    ierr = PetscStrstr(logList, "ao", &className);                                                        CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogInfoDeactivateClass(AO_COOKIE);                                                      CHKERRQ(ierr);
    }
    ierr = PetscStrstr(logList, "da", &className);                                                        CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogInfoDeactivateClass(DA_COOKIE);                                                      CHKERRQ(ierr);
    }
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-log_summary_exclude", logList, 256, &opt);                   CHKERRQ(ierr);
  if (opt == PETSC_TRUE) {
    ierr = PetscStrstr(logList, "ao", &className);                                                        CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(AO_COOKIE);                                                     CHKERRQ(ierr);
    }
    ierr = PetscStrstr(logList, "da", &className);                                                        CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(DA_COOKIE);                                                     CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#ifdef PETSC_USE_DYNAMIC_LIBRARIES
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
  ierr = DMInitializePackage(path);                                                                       CHKERRQ(ierr);
  return(0);
}
EXTERN_C_END

/* --------------------------------------------------------------------------*/
static char *contents = "PETSc Distributed Structures library, includes\n\
Application Orderings, Application Data, and Distributed Arrays";
static char *authors  = PETSC_AUTHOR_INFO;

#include "src/sys/src/utils/dlregis.h"

/* --------------------------------------------------------------------------*/

#endif /* PETSC_USE_DYNAMIC_LIBRARIES */
