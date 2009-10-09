#include "../src/semiLagrange/characteristicimpl.h"

static PetscTruth CharacteristicPackageInitialized = PETSC_FALSE;
#undef __FUNCT__  
#define __FUNCT__ "CharacteristicFinalizePackage"
/*@C
  CharacteristicFinalizePackage - This function destroys everything in the Petsc interface to the charactoristics package. It is
  called from PetscFinalize().

  Level: developer

.keywords: Petsc, destroy, package, mathematica
.seealso: PetscFinalize()
@*/
PetscErrorCode PETSC_DLLEXPORT CharacteristicFinalizePackage(void) 
{
  PetscFunctionBegin;
  CharacteristicPackageInitialized = PETSC_FALSE;
  CharacteristicRegisterAllCalled = PETSC_FALSE;
  CharacteristicList              = PETSC_NULL;
  PetscFunctionReturn(0);
}

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
PetscErrorCode CharacteristicInitializePackage(const char path[]) 
{
  char              logList[256];
  char             *className;
  PetscTruth        opt;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (CharacteristicPackageInitialized) PetscFunctionReturn(0);
  CharacteristicPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscCookieRegister("Method of Characteristics",&CHARACTERISTIC_COOKIE);CHKERRQ(ierr);
  /* Register Constructors */
  ierr = CharacteristicRegisterAll(path);CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister("MOCSetUp",         CHARACTERISTIC_COOKIE,&CHARACTERISTIC_SetUp);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MOCSolve",         CHARACTERISTIC_COOKIE,&CHARACTERISTIC_Solve);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MOCQueueSetup",    CHARACTERISTIC_COOKIE,&CHARACTERISTIC_QueueSetup);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MOCDAUpdate",      CHARACTERISTIC_COOKIE,&CHARACTERISTIC_DAUpdate);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MOCHalfTimeLocal", CHARACTERISTIC_COOKIE,&CHARACTERISTIC_HalfTimeLocal);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MOCHalfTimeRemot", CHARACTERISTIC_COOKIE,&CHARACTERISTIC_HalfTimeRemote);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MOCHalfTimeExchg", CHARACTERISTIC_COOKIE,&CHARACTERISTIC_HalfTimeExchange);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MOCFullTimeLocal", CHARACTERISTIC_COOKIE,&CHARACTERISTIC_FullTimeLocal);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MOCFullTimeRemot", CHARACTERISTIC_COOKIE,&CHARACTERISTIC_FullTimeRemote);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("MOCFullTimeExchg", CHARACTERISTIC_COOKIE,&CHARACTERISTIC_FullTimeExchange);CHKERRQ(ierr);
  /* Process info exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-log_info_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "characteristic", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscInfoDeactivateClass(CHARACTERISTIC_COOKIE);CHKERRQ(ierr);
    }
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-log_summary_exclude", logList, 256, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrstr(logList, "characteristic", &className);CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(CHARACTERISTIC_COOKIE);CHKERRQ(ierr);
    }
  }
  ierr = PetscRegisterFinalize(CharacteristicFinalizePackage);CHKERRQ(ierr);
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
PetscErrorCode PetscDLLibraryRegister_petsccontrib(const char path[])
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
static const char *contents = "PETSc Method of Characteristics library.\n";
static const char *authors  = "Richard Katz and Matthew G. Knepley\n";

/* $Id: dlregis.h,v 1.8 2001/03/23 23:20:45 balay Exp $ */
/*
   This file is included by all the dlregis.c files to provide common information
   on the PETSC team.
*/

static const char *version = "???";

EXTERN_C_BEGIN
/* --------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "PetscDLLibraryInfo"
int PetscDLLibraryInfo(const char *path,const char *type,const char **mess) 
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
