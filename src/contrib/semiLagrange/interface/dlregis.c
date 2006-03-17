#include "src/contrib/semiLagrange/characteristicimpl.h"

#undef __FUNCT__  
#define __FUNCT__ "CharacteristicInitializePackage"
/*@C
  CharacteristicInitializePackage - This function initializes everything in the Characteristic package. It is called
  from PetscDLLibraryRegister() when using dynamic libraries, and on the first call to CharacteristicCreate()
  when using static libraries.

  Input Parameter:
  path - The dynamic library path, or PETSC_NULL

  Level: developer

.keywords: Characteristic, initialize, package
.seealso: PetscInitialize()
@*/
PetscErrorCode CharacteristicInitializePackage(const char path[]) {
  static PetscTruth initialized = PETSC_FALSE;
  char              logList[256];
  char             *className;
  PetscTruth        opt;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (initialized == PETSC_TRUE) PetscFunctionReturn(0);
  initialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscLogClassRegister(&CHARACTERISTIC_COOKIE,  "Method of Characteristics");CHKERRQ(ierr);
  /* Register Constructors */
#if 0
  ierr = CharacteristicRegisterAll(path);CHKERRQ(ierr);
#endif
  /* Register Events */
  ierr = PetscLogEventRegister(&CHARACTERISTIC_SetUp,            "MOCSetUp",         CHARACTERISTIC_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&CHARACTERISTIC_Solve,            "MOCSolve",         CHARACTERISTIC_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&CHARACTERISTIC_QueueSetup,       "MOCQueueSetup",    CHARACTERISTIC_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&CHARACTERISTIC_DAUpdate,         "MOCDAUpdate",      CHARACTERISTIC_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&CHARACTERISTIC_HalfTimeLocal,    "MOCHalfTimeLocal", CHARACTERISTIC_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&CHARACTERISTIC_HalfTimeRemote,   "MOCHalfTimeRemot", CHARACTERISTIC_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&CHARACTERISTIC_HalfTimeExchange, "MOCHalfTimeExchg", CHARACTERISTIC_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&CHARACTERISTIC_FullTimeLocal,    "MOCFullTimeLocal", CHARACTERISTIC_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&CHARACTERISTIC_FullTimeRemote,   "MOCFullTimeRemot", CHARACTERISTIC_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&CHARACTERISTIC_FullTimeExchange, "MOCFullTimeExchg", CHARACTERISTIC_COOKIE);CHKERRQ(ierr);
  /* Process info exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-log_info_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt == PETSC_TRUE) {
    ierr = PetscStrstr(logList, "characteristic", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscInfoDeactivateClass(CHARACTERISTIC_COOKIE);CHKERRQ(ierr);
    }
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-log_summary_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt == PETSC_TRUE) {
    ierr = PetscStrstr(logList, "characteristic", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(CHARACTERISTIC_COOKIE);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#ifdef PETSC_USE_DYNAMIC_LIBRARIES
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PetscDLLibraryRegister_characteristic"
/*
  PetscDLLibraryRegister - This function is called when the dynamic library it is in is opened.

  This one registers all the KSP and PC methods that are in the basic PETSc libpetscksp
  library.

  Input Parameter:
  path - library path
 */
PetscErrorCode PetscDLLibraryRegister_petsccontrib(char *path)
{
  PetscErrorCode ierr;

  ierr = PetscInitializeNoArguments(); if (ierr) return 1;

  PetscFunctionBegin;
  /*
      If we got here then PETSc was properly loaded
  */
  ierr = CharacteristicInitializePackage(path);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

/* --------------------------------------------------------------------------*/
static char *contents = "PETSc Method of Characteristics library.\n";
static char *authors  = "Richard Katz and Matthew G. Knepley\n";

/* $Id: dlregis.h,v 1.8 2001/03/23 23:20:45 balay Exp $ */
/*
   This file is included by all the dlregis.c files to provide common information
   on the PETSC team.
*/

static char *version = "???";

EXTERN_C_BEGIN
/* --------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "PetscDLLibraryInfo"
int PetscDLLibraryInfo(char *path,char *type,char **mess) 
{
  PetscTruth iscon,isaut,isver;
  int        ierr;

  PetscFunctionBegin; 

  ierr = PetscStrcmp(type,"Contents",&iscon);CHKERRQ(ierr);
  ierr = PetscStrcmp(type,"Authors",&isaut);CHKERRQ(ierr);
  ierr = PetscStrcmp(type,"Version",&isver);CHKERRQ(ierr);
  if (iscon)      *mess = contents;
  else if (isaut) *mess = authors;
  else if (isver) *mess = version;
  else            *mess = 0;

  PetscFunctionReturn(0);
}
EXTERN_C_END

#endif /* PETSC_USE_DYNAMIC_LIBRARIES */
