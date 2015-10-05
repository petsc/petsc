#include "petsc_p4est_package.h"
#include <p4est_base.h>

static const char *const SCLogTypes[] = {"DEFAULT","ALWAYS","TRACE","DEBUG","VERBOSE","INFO","STATISTICS","PRODUCTION","ESSENTIAL","ERROR","SILENT","SCLogTypes","SC_LP_",0};

static PetscBool PetscP4estInitialized = PETSC_FALSE;
static PetscBool PetscBeganSc          = PETSC_FALSE;

#undef __FUNCT__
#define __FUNCT__ "PetscScLogHandler"
static void PetscScLogHandler (FILE *log_stream, const char *filename, int lineno,
                               int package, int category,
                               int priority, const char *msg)
{
  PetscInfo_Private(filename,NULL,":%d{%s} %s",lineno,package == sc_package_id ? "sc" : package == p4est_package_id ? "p4est" : "",msg);
}


/* p4est tries to abort: if possible, use setjmp to enable at least a little unwinding */
#if defined(PETSC_HAVE_SETJMP_H) && defined(PETSC_USE_ERRORCHECKING)
#include <setjmp.h>
PETSC_VISIBILITY_INTERNAL jmp_buf PetscScJumpBuf;
PETSC_INTERN void PetscScAbort_longjmp(void)
{
  longjmp(PetscScJumpBuf,1);
  return;
}

#define PetscScAbort PetscScAbort_longjmp
#else
#define PetscScAbort NULL
#endif

#undef __FUNCT__
#define __FUNCT__ "PetscP4estFinalize"
static PetscErrorCode PetscP4estFinalize()
{
  PetscFunctionBegin;
  if (PetscBeganSc) {
    PetscStackCallP4est(sc_finalize,());
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscP4estInitialize"
PetscErrorCode PetscP4estInitialize()
{
  PetscBool psc_catch_signals      = PETSC_TRUE;
  PetscBool psc_print_backtrace    = PETSC_TRUE;
  int       psc_log_threshold      = SC_LP_DEFAULT;
  int       pp4est_log_threshold   = SC_LP_DEFAULT;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscP4estInitialized) PetscFunctionReturn(0);
  PetscP4estInitialized = PETSC_TRUE;
  if (sc_package_id == -1) {
    int log_threshold_shifted = psc_log_threshold + 1;
    PetscBool set;

    PetscBeganSc = PETSC_TRUE;
    ierr = PetscOptionsGetBool(NULL,"-petsc_sc_catch_signals",&psc_catch_signals,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetBool(NULL,"-petsc_sc_print_backtrace",&psc_print_backtrace,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetEnum(NULL,"-petsc_sc_log_threshold",SCLogTypes,(PetscEnum*)&log_threshold_shifted,&set);CHKERRQ(ierr);
    if (set) {
      psc_log_threshold = log_threshold_shifted - 1;
    }
    sc_init(PETSC_COMM_WORLD,psc_catch_signals,psc_print_backtrace,PetscScLogHandler,psc_log_threshold);
    if (sc_package_id == -1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_LIB,"Could not initialize libsc package used by p4est");
    sc_set_abort_handler(PetscScAbort);
  }
  if (p4est_package_id == -1) {
    int log_threshold_shifted = pp4est_log_threshold + 1;
    PetscBool set;

    ierr = PetscOptionsGetEnum(NULL,"-petsc_p4est_log_threshold",SCLogTypes,(PetscEnum*)&log_threshold_shifted,&set);CHKERRQ(ierr);
    if (set) {
      pp4est_log_threshold = log_threshold_shifted - 1;
    }
    PetscStackCallP4est(p4est_init,(PetscScLogHandler,pp4est_log_threshold));
    if (p4est_package_id == -1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_LIB,"Could not initialize p4est");
  }
  ierr = PetscRegisterFinalize(PetscP4estFinalize);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
