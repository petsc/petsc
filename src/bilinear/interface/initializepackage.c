#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: initializepackage.c,v 1.1 2000/01/10 03:10:37 knepley Exp $";
#endif

#include "bilinear.h"

EXTERN_C_BEGIN
extern int BilinearCreate_Dense_Seq(Bilinear);

extern int BilinearSerialize_Dense_Seq(MPI_Comm, Bilinear *, PetscViewer, PetscTruth);
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "BilinearInitializePackage"
/*@C
  BilinearInitializePackage - This function initializes everything in the Bilinear package. It is called
  from PetscDLLibraryRegister() when using dynamic libraries, and on the first call to BilienarCreate()
  when using static libraries.

  Input Parameter:
  path - The dynamic library path, or PETSC_NULL

  Level: developer

.keywords: Bilinear, initialize, package
.seealso: PetscInitialize()
@*/
int BilinearInitializePackage(char *path) {
  static PetscTruth initialized = PETSC_FALSE;
  char              logList[256];
  char             *className;
  PetscTruth        opt;
  int               ierr;

  PetscFunctionBegin;
  if (initialized == PETSC_TRUE) PetscFunctionReturn(0);
  initialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscLogClassRegister(&BILINEAR_COOKIE, "Bilinear");                                             CHKERRQ(ierr);
  /* Register Constructors and Serializers */
  ierr = BilinearSerializeRegisterAll(path);                                                              CHKERRQ(ierr);
  /* Register Events */
  ierr = PetscLogEventRegister(&BILINEAR_Copy,           "BilinearCopy",     PETSC_NULL, BILINEAR_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&BILINEAR_Convert,        "BilinearConvert",  PETSC_NULL, BILINEAR_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&BILINEAR_SetValues,      "BilinSetValues",   PETSC_NULL, BILINEAR_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&BILINEAR_AssemblyBegin,  "BilinAssemBegin",  PETSC_NULL, BILINEAR_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&BILINEAR_AssemblyEnd,    "BilinAssemEnd",    PETSC_NULL, BILINEAR_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&BILINEAR_ZeroEntries,    "BilinZeroEntries", PETSC_NULL, BILINEAR_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&BILINEAR_Mult,           "BilinearMult",     PETSC_NULL, BILINEAR_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&BILINEAR_FullMult,       "BilinearFullMult", PETSC_NULL, BILINEAR_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&BILINEAR_Diamond,        "BilinearDiamond",  PETSC_NULL, BILINEAR_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&BILINEAR_LUFactor,       "BilinearLUFactor", PETSC_NULL, BILINEAR_COOKIE);CHKERRQ(ierr);
  ierr = PetscLogEventRegister(&BILINEAR_CholeskyFactor, "BilinearCholFact", PETSC_NULL, BILINEAR_COOKIE);CHKERRQ(ierr);
  /* Process info exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-log_info_exclude", logList, 256, &opt);                      CHKERRQ(ierr);
  if (opt == PETSC_TRUE) {
    ierr = PetscStrstr(logList, "bilinear", &className);                                                  CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogInfoDeactivateClass(BILINEAR_COOKIE);                                                CHKERRQ(ierr);
    }
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(PETSC_NULL, "-log_summary_exclude", logList, 256, &opt);                   CHKERRQ(ierr);
  if (opt == PETSC_TRUE) {
    ierr = PetscStrstr(logList, "bilinear", &className);                                                  CHKERRQ(ierr);
    if (className) {
      ierr = PetscLogEventDeactivateClass(BILINEAR_COOKIE);                                               CHKERRQ(ierr);
    }
  }
  /* Add options checkers */
  return(0);
}
