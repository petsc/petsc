/*
   IEEE error handler for all machines. Since each OS has
   enough slight differences we have completely separate codes for each one.
*/

/*
  This feature test macro provides FE_NOMASK_ENV on GNU.  It must be defined
  at the top of the file because other headers may pull in fenv.h even when
  not strictly necessary.  Strictly speaking, we could include ONLY petscconf.h,
  check PETSC_HAVE_FENV_H, and only define _GNU_SOURCE in that case, but such
  shenanigans ought to be unnecessary.
*/
#if !defined(_GNU_SOURCE)
  #define _GNU_SOURCE
#endif

#include <petsc/private/petscimpl.h> /*I  "petscsys.h"  I*/
#include <signal.h>
#if PetscDefined(HAVE_XMMINTRIN_H)
  #include <xmmintrin.h>
#endif

struct PetscFPTrapLink {
  PetscFPTrap             trapmode;
  struct PetscFPTrapLink *next;
};
static PetscFPTrap             _trapmode = PETSC_FP_TRAP_OFF; /* Current trapping mode; see PetscDetermineInitialFPTrap() */
static struct PetscFPTrapLink *_trapstack;                    /* Any pushed states of _trapmode */

/*@
  PetscFPTrapPush - push a floating point trapping mode, restored using `PetscFPTrapPop()`

  Not Collective

  Input Parameter:
. trap - `PETSC_FP_TRAP_ON` or `PETSC_FP_TRAP_OFF` or any of the values passable to `PetscSetFPTrap()`

  Level: advanced

  Notes:
  This only changes the trapping if the new mode is different than the current mode.

  This routine is called to turn off trapping for certain LAPACK routines that assume that dividing
  by zero is acceptable. In particular the routine ieeeck().

  Most systems by default have all trapping turned off, but certain Fortran compilers have
  link flags that turn on trapping before the program begins.
.vb
       gfortran -ffpe-trap=invalid,zero,overflow,underflow,denormal
       ifort -fpe0
.ve

.seealso: `PetscFPTrapPop()`, `PetscSetFPTrap()`, `PetscDetermineInitialFPTrap()`
@*/
PetscErrorCode PetscFPTrapPush(PetscFPTrap trap)
{
  struct PetscFPTrapLink *link;

  PetscFunctionBegin;
  PetscCall(PetscNew(&link));
#if PetscDefined(HAVE_THREADSAFETY) && PetscDefined(HAVE_OPENMP)
  PetscPragmaOMP(critical)
#endif
  {
    link->trapmode = _trapmode;
    link->next     = _trapstack;
    _trapstack     = link;
  }
  if (trap != _trapmode) PetscCall(PetscSetFPTrap(trap));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscFPTrapPop - push a floating point trapping mode, to be restored using `PetscFPTrapPop()`

  Not Collective

  Level: advanced

.seealso: `PetscFPTrapPush()`, `PetscSetFPTrap()`, `PetscDetermineInitialFPTrap()`
@*/
PetscErrorCode PetscFPTrapPop(void)
{
  struct PetscFPTrapLink *link;

  PetscFunctionBegin;
  if (_trapstack->trapmode != _trapmode) PetscCall(PetscSetFPTrap(_trapstack->trapmode));
#if PetscDefined(HAVE_THREADSAFETY) && PetscDefined(HAVE_OPENMP)
  PetscPragmaOMP(critical)
#endif
  {
    link       = _trapstack;
    _trapstack = _trapstack->next;
  }
  PetscCall(PetscFree(link));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if PetscDefined(HAVE_WINDOWS_COMPILERS)
  #include <float.h>
void PetscDefaultFPTrap(int sig)
{
  PetscFunctionBegin;
  (void)(*PetscErrorPrintf)("*** floating point error occurred ***\n");
  (void)PetscError(PETSC_COMM_SELF, 0, NULL, NULL, PETSC_ERR_FP, PETSC_ERROR_REPEAT, "floating point error");
  PETSCABORT(MPI_COMM_WORLD, PETSC_ERR_FP);
}

/*@
   PetscSetFPTrap - Enables traps/exceptions on common floating point errors. This option may not work on certain systems or only a
   subset of exceptions may be trapable.

   Not Collective

   Input Parameter:
.  flag - values are
.vb
    PETSC_FP_TRAP_OFF   - do not trap any exceptions
    PETSC_FP_TRAP_ON - all exceptions that are possible on the system except underflow
    PETSC_FP_TRAP_INDIV - integer divide by zero
    PETSC_FP_TRAP_FLTOPERR - improper argument to function, for example with real numbers, the square root of a negative number
    PETSC_FP_TRAP_FLTOVF - overflow
    PETSC_FP_TRAP_FLTUND - underflow - not trapped by default on most systems
    PETSC_FP_TRAP_FLTDIV - floating point divide by zero
    PETSC_FP_TRAP_FLTINEX - inexact floating point result
.ve

   Options Database Key:
.  -fp_trap (true|false) - turn on or off trapping of floating point exceptions

   Level: advanced

   Notes:
   Preferred usage is `PetscFPTrapPush()` and `PetscFPTrapPop()` instead of this routine

   Currently only `PETSC_FP_TRAP_OFF` and `PETSC_FP_TRAP_ON` are handled. All others are treated as `PETSC_FP_TRAP_ON`.

   The values are bit values and may be |ed together in the function call

   On systems that support it this routine causes floating point
   overflow, divide-by-zero, and invalid-operand (e.g., a NaN), but not underflow, to
   cause a message to be printed and the program to exit.

   On many common systems, the floating
   point exception state is not preserved from the location where the trap
   occurred through to the signal handler.  In this case, the signal handler
   will just say that an unknown floating point exception occurred and which
   function it occurred in.  If you run with -fp_trap in a debugger, it will
   break on the line where the error occurred.  On systems that support C99
   floating point exception handling You can check which
   exception occurred using fetestexcept(FE_ALL_EXCEPT).  See fenv.h
   (usually at /usr/include/bits/fenv.h) for the enum values on your system.

.seealso: `PetscFPTrapPush()`, `PetscFPTrapPop()`, `PetscDetermineInitialFPTrap()`
@*/
PetscErrorCode PetscSetFPTrap(PetscFPTrap flag)
{
  unsigned int cw;

  PetscFunctionBegin;
  if (flag == PETSC_FP_TRAP_ON) {
    /* cw = _EM_INVALID | _EM_ZERODIVIDE | _EM_OVERFLOW | _EM_UNDERFLOW; */
    cw = _EM_INVALID | _EM_ZERODIVIDE | _EM_OVERFLOW;
    PetscCheck(SIG_ERR != signal(SIGFPE, PetscDefaultFPTrap), PETSC_COMM_SELF, PETSC_ERR_LIB, "Can't set floating point handler");
  } else {
    cw = 0;
    PetscCheck(SIG_ERR != signal(SIGFPE, SIG_DFL), PETSC_COMM_SELF, PETSC_ERR_LIB, "Can't clear floating point handler");
  }
  (void)_controlfp(0, cw);
  _trapmode = flag;
  PetscCall(PetscInfo(NULL, "Using PETSC_HAVE_WINDOWS_COMPILERS FPTRAP\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   PetscDetermineInitialFPTrap - Attempts to determine the floating point trapping that exists when `PetscInitialize()` is called

   Not Collective

   Note:
   Currently only supported on Linux and macOS. Checks if divide by zero is enable and if so declares that trapping is on.

   Level: advanced

.seealso: `PetscFPTrapPush()`, `PetscFPTrapPop()`, `PetscDetermineInitialFPTrap()`
@*/
PetscErrorCode PetscDetermineInitialFPTrap(void)
{
  PetscFunctionBegin;
  PetscCall(PetscInfo(NULL, "Unable to determine initial floating point trapping. Assuming it is off\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#elif PetscDefined(HAVE_FENV_H) && !defined(__cplusplus)
  /*
   C99 style floating point environment.

   Note that C99 merely specifies how to save, restore, and clear the floating
   point environment as well as defining an enumeration of exception codes.  In
   particular, C99 does not specify how to make floating point exceptions raise
   a signal.  Glibc offers this capability through FE_NOMASK_ENV (or with finer
   granularity, feenableexcept()), xmmintrin.h offers _MM_SET_EXCEPTION_MASK().
*/
  #include <fenv.h>
  #if PetscDefined(HAVE_FE_VALUES)
typedef struct {
  int         code;
  const char *name;
} FPNode;
static const FPNode error_codes[] = {
  {FE_DIVBYZERO, "divide by zero"                                 },
  {FE_INEXACT,   "inexact floating point result"                  },
  {FE_INVALID,   "invalid floating point arguments (domain error)"},
  {FE_OVERFLOW,  "floating point overflow"                        },
  {FE_UNDERFLOW, "floating point underflow"                       },
  {0,            "unknown error"                                  }
};
  #endif

void PetscDefaultFPTrap(int sig)
{
  #if PetscDefined(HAVE_FE_VALUES)
  const FPNode *node;
  int           code;
  PetscBool     matched = PETSC_FALSE;
  #endif

  PetscFunctionBegin;
  /* Note: While it is possible for the exception state to be preserved by the
   * kernel, this seems to be rare which makes the following flag testing almost
   * useless.  But on a system where the flags can be preserved, it would provide
   * more detail.
   */
  #if PetscDefined(HAVE_FE_VALUES)
  code = fetestexcept(FE_ALL_EXCEPT);
  for (node = &error_codes[0]; node->code; node++) {
    if (code & node->code) {
      matched = PETSC_TRUE;
      (void)(*PetscErrorPrintf)("*** floating point error \"%s\" occurred ***\n", node->name);
      code &= ~node->code; /* Unset this flag since it has been processed */
    }
  }
  if (!matched || code) { /* If any remaining flags are set, or we didn't process any flags */
    (void)(*PetscErrorPrintf)("*** unknown floating point error occurred ***\n");
    (void)(*PetscErrorPrintf)("The specific exception can be determined by running in a debugger.  When the\n");
    (void)(*PetscErrorPrintf)("debugger traps the signal, the exception can be found with fetestexcept(0x%x)\n", FE_ALL_EXCEPT);
    (void)(*PetscErrorPrintf)("where the result is a bitwise OR of the following flags:\n");
    (void)(*PetscErrorPrintf)("FE_INVALID=0x%x FE_DIVBYZERO=0x%x FE_OVERFLOW=0x%x FE_UNDERFLOW=0x%x FE_INEXACT=0x%x\n", FE_INVALID, FE_DIVBYZERO, FE_OVERFLOW, FE_UNDERFLOW, FE_INEXACT);
  }
  #else
  (void)(*PetscErrorPrintf)("*** unknown floating point error occurred ***\n");
  #endif

  (void)(*PetscErrorPrintf)("Try option -start_in_debugger\n");
  if (PetscDefined(USE_DEBUG)) {
    (void)(*PetscErrorPrintf)("likely location of problem given in stack below\n");
    (void)(*PetscErrorPrintf)("---------------------  Stack Frames ------------------------------------\n");
    (void)PetscStackView(PETSC_STDOUT);
  } else {
    (void)(*PetscErrorPrintf)("configure using --with-debugging=yes, recompile, link, and run \n");
    (void)(*PetscErrorPrintf)("with -start_in_debugger to get more information on the crash.\n");
  }
  (void)PetscError(PETSC_COMM_SELF, 0, NULL, NULL, PETSC_ERR_FP, PETSC_ERROR_INITIAL, "trapped floating point error");
  PETSCABORT(MPI_COMM_WORLD, PETSC_ERR_FP);
}

PetscErrorCode PetscSetFPTrap(PetscFPTrap flag)
{
  PetscFunctionBegin;
  if (flag == PETSC_FP_TRAP_ON) {
    /* Clear any flags that are currently set so that activating trapping will not immediately call the signal handler. */
    PetscCheck(!feclearexcept(FE_ALL_EXCEPT), PETSC_COMM_SELF, PETSC_ERR_LIB, "Cannot clear floating point exception flags");
  #if defined(FE_NOMASK_ENV) && PetscDefined(HAVE_FE_VALUES)
    /* Could use fesetenv(FE_NOMASK_ENV), but that causes spurious exceptions (like gettimeofday() -> PetscLogDouble). */
    /* PetscCheck(feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW | FE_UNDERFLOW) != -1,PETSC_COMM_SELF,PETSC_ERR_LIB,"Cannot activate floating point exceptions"); */
    /* Doesn't work on AArch64 targets. There's a known hardware limitation. Need to detect hardware at configure time? */
    PetscCheck(feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW) != -1, PETSC_COMM_SELF, PETSC_ERR_LIB, "Cannot activate floating point exceptions");
    PetscCall(PetscInfo(NULL, "Using PETSC_HAVE_FENV_H FPTRAP with FE_NOMASK_ENV\n"));
  #elif defined PETSC_HAVE_XMMINTRIN_H
    _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~_MM_MASK_DIV_ZERO);
    /* _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~_MM_MASK_UNDERFLOW); */
    _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~_MM_MASK_OVERFLOW);
    _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~_MM_MASK_INVALID);
    PetscCall(PetscInfo(NULL, "Using PETSC_HAVE_FENV_H FPTRAP with PETSC_HAVE_XMMINTRIN_H\n"));
  #else
    /* C99 does not provide a way to modify the environment so there is no portable way to activate trapping. */
    PetscCall(PetscInfo(NULL, "Using PETSC_HAVE_FENV_H FPTRAP\n"));
  #endif
    PetscCheck(SIG_ERR != signal(SIGFPE, PetscDefaultFPTrap), PETSC_COMM_SELF, PETSC_ERR_LIB, "Can't set floating point handler");
  } else {
    PetscCheck(!fesetenv(FE_DFL_ENV), PETSC_COMM_SELF, PETSC_ERR_LIB, "Cannot disable floating point exceptions");
    /* can use _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() | _MM_MASK_UNDERFLOW); if PETSC_HAVE_XMMINTRIN_H exists */
    PetscCheck(SIG_ERR != signal(SIGFPE, SIG_DFL), PETSC_COMM_SELF, PETSC_ERR_LIB, "Can't clear floating point handler");
  }
  _trapmode = flag;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDetermineInitialFPTrap(void)
{
  #if defined(FE_NOMASK_ENV) || defined PETSC_HAVE_XMMINTRIN_H
  unsigned int flags;
  #endif

  PetscFunctionBegin;
  #if defined(FE_NOMASK_ENV)
  flags = fegetexcept();
  if (flags & FE_DIVBYZERO) {
  #elif defined PETSC_HAVE_XMMINTRIN_H
  flags = _MM_GET_EXCEPTION_MASK();
  if (!(flags & _MM_MASK_DIV_ZERO)) {
  #else
  PetscCall(PetscInfo(NULL, "Floating point trapping unknown, assuming off\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
  #endif
  #if defined(FE_NOMASK_ENV) || defined PETSC_HAVE_XMMINTRIN_H
    _trapmode = PETSC_FP_TRAP_ON;
    PetscCall(PetscInfo(NULL, "Floating point trapping is on by default %d\n", flags));
  } else {
    _trapmode = PETSC_FP_TRAP_OFF;
    PetscCall(PetscInfo(NULL, "Floating point trapping is off by default %d\n", flags));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
  #endif
}

#elif PetscDefined(HAVE_IEEEFP_H)
  #include <ieeefp.h>
void PetscDefaultFPTrap(int sig)
{
  PetscFunctionBegin;
  (void)(*PetscErrorPrintf)("*** floating point error occurred ***\n");
  (void)PetscError(PETSC_COMM_SELF, 0, NULL, NULL, PETSC_ERR_FP, PETSC_ERROR_REPEAT, "floating point error");
  PETSCABORT(MPI_COMM_WORLD, PETSC_ERR_FP);
}

PetscErrorCode PetscSetFPTrap(PetscFPTrap flag)
{
  PetscFunctionBegin;
  if (flag == PETSC_FP_TRAP_ON) {
  #if PetscDefined(HAVE_FPRESETSTICKY)
    fpresetsticky(fpgetsticky());
  #elif PetscDefined(HAVE_FPSETSTICKY)
    fpsetsticky(fpgetsticky());
  #endif
    fpsetmask(FP_X_INV | FP_X_DZ | FP_X_OFL | FP_X_OFL);
    PetscCheck(SIG_ERR != signal(SIGFPE, PetscDefaultFPTrap), PETSC_COMM_SELF, PETSC_ERR_LIB, "Can't set floating point handler");
  } else {
  #if PetscDefined(HAVE_FPRESETSTICKY)
    fpresetsticky(fpgetsticky());
  #elif PetscDefined(HAVE_FPSETSTICKY)
    fpsetsticky(fpgetsticky());
  #endif
    fpsetmask(0);
    PetscCheck(SIG_ERR != signal(SIGFPE, SIG_DFL), PETSC_COMM_SELF, PETSC_ERR_LIB, "Can't clear floating point handler");
  }
  _trapmode = flag;
  PetscCall(PetscInfo(NULL, "Using PETSC_HAVE_IEEEFP_H FPTRAP\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDetermineInitialFPTrap(void)
{
  PetscFunctionBegin;
  PetscCall(PetscInfo(NULL, "Unable to determine initial floating point trapping. Assuming it is off\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Default */
#else

static void PetscDefaultFPTrap(int sig)
{
  PetscFunctionBegin;
  (void)(*PetscErrorPrintf)("*** floating point error occurred ***\n");
  (void)PetscError(PETSC_COMM_SELF, 0, NULL, NULL, PETSC_ERR_FP, PETSC_ERROR_REPEAT, "floating point error");
  PETSCABORT(MPI_COMM_WORLD, PETSC_ERR_FP);
}

PetscErrorCode PetscSetFPTrap(PetscFPTrap flag)
{
  PetscFunctionBegin;
  if (flag == PETSC_FP_TRAP_ON) {
    if (SIG_ERR == signal(SIGFPE, PetscDefaultFPTrap)) PetscCall((*PetscErrorPrintf)("Can't set floatingpoint handler\n"));
  } else if (SIG_ERR == signal(SIGFPE, SIG_DFL)) PetscCall((*PetscErrorPrintf)("Can't clear floatingpoint handler\n"));

  _trapmode = flag;
  PetscCall(PetscInfo(NULL, "Using default FPTRAP\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscDetermineInitialFPTrap(void)
{
  PetscFunctionBegin;
  PetscCall(PetscInfo(NULL, "Unable to determine initial floating point trapping. Assuming it is off\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif
