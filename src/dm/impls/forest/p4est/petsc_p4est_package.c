#include <petscdmforest.h>
#include <petsc/private/petscimpl.h>
#include "petsc_p4est_package.h"

static const char*const SCLogTypes[] = {"DEFAULT","ALWAYS","TRACE","DEBUG","VERBOSE","INFO","STATISTICS","PRODUCTION","ESSENTIAL","ERROR","SILENT","SCLogTypes","SC_LP_", NULL};

static PetscBool    PetscP4estInitialized = PETSC_FALSE;
static PetscBool    PetscBeganSc          = PETSC_FALSE;
static PetscClassId P4ESTLOGGING_CLASSID;

PetscObject P4estLoggingObject; /* Just a vehicle for its classid */

static void PetscScLogHandler(FILE *log_stream, const char *filename, int lineno,int package, int category,int priority, const char *msg)
{
  PetscInfo_Private(filename,P4estLoggingObject,":%d{%s} %s",lineno,package == sc_package_id ? "sc" : package == p4est_package_id ? "p4est" : "",msg);
}

/* p4est tries to abort: if possible, use setjmp to enable at least a little unwinding */
#if defined(PETSC_HAVE_SETJMP_H) && defined(PETSC_USE_DEBUG)
#include <setjmp.h>
PETSC_VISIBILITY_INTERNAL jmp_buf PetscScJumpBuf;
PETSC_INTERN void PetscScAbort_longjmp(void)
{
  PetscError(PETSC_COMM_SELF,-1,"p4est function","p4est file",PETSC_ERR_LIB,PETSC_ERROR_INITIAL,"Error in p4est stack call\n");
  longjmp(PetscScJumpBuf,1);
  return;
}

#define PetscScAbort PetscScAbort_longjmp
#else
#define PetscScAbort NULL
#endif

static PetscErrorCode PetscP4estFinalize(void)
{
  PetscFunctionBegin;
  if (PetscBeganSc) {
    /* We do not want libsc to abort on a mismatched allocation and prevent further Petsc unwinding */
    PetscStackCallP4est(sc_package_set_abort_alloc_mismatch,(sc_package_id,0));
    PetscStackCallP4est(sc_package_set_abort_alloc_mismatch,(p4est_package_id,0));
    PetscStackCallP4est(sc_package_set_abort_alloc_mismatch,(-1,0));
    PetscStackCallP4est(sc_finalize,());
  }
  CHKERRQ(PetscHeaderDestroy(&P4estLoggingObject));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscP4estInitialize(void)
{
  PetscBool psc_catch_signals    = PETSC_FALSE;
  PetscBool psc_print_backtrace  = PETSC_TRUE;
  int       psc_log_threshold    = SC_LP_DEFAULT;
  int       pp4est_log_threshold = SC_LP_DEFAULT;
  char      logList[256];
  PetscBool opt,pkg;

  PetscFunctionBegin;
  if (PetscP4estInitialized) PetscFunctionReturn(0);
  PetscP4estInitialized = PETSC_TRUE;

  /* Register Classes */
  CHKERRQ(PetscClassIdRegister("p4est logging",&P4ESTLOGGING_CLASSID));
  /* Process Info */
  {
    PetscClassId  classids[1];

    classids[0] = P4ESTLOGGING_CLASSID;
    CHKERRQ(PetscInfoProcessClass("p4est", 1, classids));
  }
  /* Process summary exclusions */
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt));
  if (opt) {
    CHKERRQ(PetscStrInList("p4est",logList,',',&pkg));
    if (pkg) CHKERRQ(PetscLogEventExcludeClass(P4ESTLOGGING_CLASSID));
  }
  CHKERRQ(PetscHeaderCreate(P4estLoggingObject,P4ESTLOGGING_CLASSID,"p4est","p4est logging","DM",PETSC_COMM_WORLD,NULL,PetscObjectView));
  if (sc_package_id == -1) {
    int       log_threshold_shifted = psc_log_threshold + 1;
    PetscBool set;

    PetscBeganSc = PETSC_TRUE;
    CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-petsc_sc_catch_signals",&psc_catch_signals,NULL));
    CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-petsc_sc_print_backtrace",&psc_print_backtrace,NULL));
    CHKERRQ(PetscOptionsGetEnum(NULL,NULL,"-petsc_sc_log_threshold",SCLogTypes,(PetscEnum*)&log_threshold_shifted,&set));
    if (set) psc_log_threshold = log_threshold_shifted - 1;
    sc_init(PETSC_COMM_WORLD,(int)psc_catch_signals,(int)psc_print_backtrace,PetscScLogHandler,psc_log_threshold);
    PetscCheckFalse(sc_package_id == -1,PETSC_COMM_WORLD,PETSC_ERR_LIB,"Could not initialize libsc package used by p4est");
    sc_set_abort_handler(PetscScAbort);
  }
  if (p4est_package_id == -1) {
    int       log_threshold_shifted = pp4est_log_threshold + 1;
    PetscBool set;

    CHKERRQ(PetscOptionsGetEnum(NULL,NULL,"-petsc_p4est_log_threshold",SCLogTypes,(PetscEnum*)&log_threshold_shifted,&set));
    if (set) pp4est_log_threshold = log_threshold_shifted - 1;
    PetscStackCallP4est(p4est_init,(PetscScLogHandler,pp4est_log_threshold));
    PetscCheckFalse(p4est_package_id == -1,PETSC_COMM_WORLD,PETSC_ERR_LIB,"Could not initialize p4est");
  }
  CHKERRQ(DMForestRegisterType(DMP4EST));
  CHKERRQ(DMForestRegisterType(DMP8EST));
  CHKERRQ(PetscRegisterFinalize(PetscP4estFinalize));
  PetscFunctionReturn(0);
}
