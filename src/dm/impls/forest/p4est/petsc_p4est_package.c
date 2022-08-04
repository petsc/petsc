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
    PetscCallP4est(sc_package_set_abort_alloc_mismatch,(sc_package_id,0));
    PetscCallP4est(sc_package_set_abort_alloc_mismatch,(p4est_package_id,0));
    PetscCallP4est(sc_package_set_abort_alloc_mismatch,(-1,0));
    PetscCallP4est(sc_finalize,());
  }
  PetscCall(PetscHeaderDestroy(&P4estLoggingObject));
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
  PetscCall(PetscClassIdRegister("p4est logging",&P4ESTLOGGING_CLASSID));
  /* Process Info */
  {
    PetscClassId  classids[1];

    classids[0] = P4ESTLOGGING_CLASSID;
    PetscCall(PetscInfoProcessClass("p4est", 1, classids));
  }
  /* Process summary exclusions */
  PetscCall(PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt));
  if (opt) {
    PetscCall(PetscStrInList("p4est",logList,',',&pkg));
    if (pkg) PetscCall(PetscLogEventExcludeClass(P4ESTLOGGING_CLASSID));
  }
  PetscCall(PetscHeaderCreate(P4estLoggingObject,P4ESTLOGGING_CLASSID,"p4est","p4est logging","DM",PETSC_COMM_WORLD,NULL,PetscObjectView));
  if (sc_package_id == -1) {
    int       log_threshold_shifted = psc_log_threshold + 1;
    PetscBool set;
#if defined(PETSC_HAVE_MPIUNI)
    sc_MPI_Comm comm_world = sc_MPI_COMM_WORLD;
#else
    MPI_Comm comm_world = PETSC_COMM_WORLD;
#endif

    PetscBeganSc = PETSC_TRUE;
    PetscCall(PetscOptionsGetBool(NULL,NULL,"-petsc_sc_catch_signals",&psc_catch_signals,NULL));
    PetscCall(PetscOptionsGetBool(NULL,NULL,"-petsc_sc_print_backtrace",&psc_print_backtrace,NULL));
    PetscCall(PetscOptionsGetEnum(NULL,NULL,"-petsc_sc_log_threshold",SCLogTypes,(PetscEnum*)&log_threshold_shifted,&set));
    if (set) psc_log_threshold = log_threshold_shifted - 1;
    sc_init(comm_world,(int)psc_catch_signals,(int)psc_print_backtrace,PetscScLogHandler,psc_log_threshold);
    PetscCheck(sc_package_id != -1,PETSC_COMM_WORLD,PETSC_ERR_LIB,"Could not initialize libsc package used by p4est");
    sc_set_abort_handler(PetscScAbort);
  }
  if (p4est_package_id == -1) {
    int       log_threshold_shifted = pp4est_log_threshold + 1;
    PetscBool set;

    PetscCall(PetscOptionsGetEnum(NULL,NULL,"-petsc_p4est_log_threshold",SCLogTypes,(PetscEnum*)&log_threshold_shifted,&set));
    if (set) pp4est_log_threshold = log_threshold_shifted - 1;
    PetscCallP4est(p4est_init,(PetscScLogHandler,pp4est_log_threshold));
    PetscCheck(p4est_package_id != -1,PETSC_COMM_WORLD,PETSC_ERR_LIB,"Could not initialize p4est");
  }
  PetscCall(DMForestRegisterType(DMP4EST));
  PetscCall(DMForestRegisterType(DMP8EST));
  PetscCall(PetscRegisterFinalize(PetscP4estFinalize));
  PetscFunctionReturn(0);
}
