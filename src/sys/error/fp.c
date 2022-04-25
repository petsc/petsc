
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

#include <petsc/private/petscimpl.h>           /*I  "petscsys.h"  I*/
#include <signal.h>

struct PetscFPTrapLink {
  PetscFPTrap            trapmode;
  struct PetscFPTrapLink *next;
};
static PetscFPTrap            _trapmode = PETSC_FP_TRAP_OFF; /* Current trapping mode; see PetscDetermineInitialFPTrap() */
static struct PetscFPTrapLink *_trapstack;                   /* Any pushed states of _trapmode */

/*@
   PetscFPTrapPush - push a floating point trapping mode, restored using PetscFPTrapPop()

   Not Collective

   Input Parameter:
.    trap - PETSC_FP_TRAP_ON or PETSC_FP_TRAP_OFF

   Level: advanced

   Notes:
     This only changes the trapping if the new mode is different than the current mode.

     This routine is called to turn off trapping for certain LAPACK routines that assume that dividing
     by zero is acceptable. In particular the routine ieeeck().

     Most systems by default have all trapping turned off, but certain Fortran compilers have
     link flags that turn on trapping before the program begins.
$       gfortran -ffpe-trap=invalid,zero,overflow,underflow,denormal
$       ifort -fpe0

.seealso: `PetscFPTrapPop()`, `PetscSetFPTrap()`, `PetscDetermineInitialFPTrap()`
@*/
PetscErrorCode PetscFPTrapPush(PetscFPTrap trap)
{
  struct PetscFPTrapLink *link;

  PetscFunctionBegin;
  PetscCall(PetscNew(&link));
  link->trapmode = _trapmode;
  link->next     = _trapstack;
  _trapstack     = link;
  if (trap != _trapmode) PetscCall(PetscSetFPTrap(trap));
  PetscFunctionReturn(0);
}

/*@
   PetscFPTrapPop - push a floating point trapping mode, to be restored using PetscFPTrapPop()

   Not Collective

   Level: advanced

.seealso: `PetscFPTrapPush()`, `PetscSetFPTrap()`, `PetscDetermineInitialFPTrap()`
@*/
PetscErrorCode PetscFPTrapPop(void)
{
  struct PetscFPTrapLink *link;

  PetscFunctionBegin;
  if (_trapstack->trapmode != _trapmode) PetscCall(PetscSetFPTrap(_trapstack->trapmode));
  link       = _trapstack;
  _trapstack = _trapstack->next;
  PetscCall(PetscFree(link));
  PetscFunctionReturn(0);
}

/*--------------------------------------- ---------------------------------------------------*/
#if defined(PETSC_HAVE_SUN4_STYLE_FPTRAP)
#include <floatingpoint.h>

PETSC_EXTERN PetscErrorCode ieee_flags(char*,char*,char*,char**);
PETSC_EXTERN PetscErrorCode ieee_handler(char*,char*,sigfpe_handler_type(int,int,struct sigcontext*,char*));

static struct { int code_no; char *name; } error_codes[] = {
  { FPE_INTDIV_TRAP    ,"integer divide" },
  { FPE_FLTOPERR_TRAP  ,"IEEE operand error" },
  { FPE_FLTOVF_TRAP    ,"floating point overflow" },
  { FPE_FLTUND_TRAP    ,"floating point underflow" },
  { FPE_FLTDIV_TRAP    ,"floating pointing divide" },
  { FPE_FLTINEX_TRAP   ,"inexact floating point result" },
  { 0                  ,"unknown error" }
};
#define SIGPC(scp) (scp->sc_pc)

sigfpe_handler_type PetscDefaultFPTrap(int sig,int code,struct sigcontext *scp,char *addr)
{
  int err_ind = -1;

  PetscFunctionBegin;
  for (int j = 0; error_codes[j].code_no; j++) {
    if (error_codes[j].code_no == code) err_ind = j;
  }

  if (err_ind >= 0) (*PetscErrorPrintf)("*** %s occurred at pc=%X ***\n",error_codes[err_ind].name,SIGPC(scp));
  else              (*PetscErrorPrintf)("*** floating point error 0x%x occurred at pc=%X ***\n",code,SIGPC(scp));

  (void)PetscError(PETSC_COMM_SELF,PETSC_ERR_FP,"User provided function","Unknown file",PETSC_ERR_FP,PETSC_ERROR_REPEAT,"floating point error");
  PETSCABORT(MPI_COMM_WORLD,PETSC_ERR_FP);
  PetscFunctionReturn(0);
}

/*@
   PetscSetFPTrap - Enables traps/exceptions on common floating point errors.
                    This option may not work on certain systems.

   Not Collective

   Input Parameters:
.  flag - PETSC_FP_TRAP_ON, PETSC_FP_TRAP_OFF.

   Options Database Keys:
.  -fp_trap - Activates floating point trapping

   Level: advanced

   Description:
   On systems that support it, when called with PETSC_FP_TRAP_ON this routine causes floating point
   underflow, overflow, divide-by-zero, and invalid-operand (e.g., a NaN) to
   cause a message to be printed and the program to exit.

   Note:
   On many common systems, the floating
   point exception state is not preserved from the location where the trap
   occurred through to the signal handler.  In this case, the signal handler
   will just say that an unknown floating point exception occurred and which
   function it occurred in.  If you run with -fp_trap in a debugger, it will
   break on the line where the error occurred.  On systems that support C99
   floating point exception handling You can check which
   exception occurred using fetestexcept(FE_ALL_EXCEPT).  See fenv.h
   (usually at /usr/include/bits/fenv.h) for the enum values on your system.

   Caution:
   On certain machines, in particular the IBM PowerPC, floating point
   trapping may be VERY slow!

.seealso: `PetscFPTrapPush()`, `PetscFPTrapPop()`, `PetscDetermineInitialFPTrap()`
@*/
PetscErrorCode PetscSetFPTrap(PetscFPTrap flag)
{
  char *out;

  PetscFunctionBegin;
  /* Clear accumulated exceptions.  Used to suppress meaningless messages from f77 programs */
  (void) ieee_flags("clear","exception","all",&out);
  if (flag == PETSC_FP_TRAP_ON) {
    /*
      To trap more fp exceptions, including underflow, change the line below to
      if (ieee_handler("set","all",PetscDefaultFPTrap)) {
    */
    if (ieee_handler("set","common",PetscDefaultFPTrap))        (*PetscErrorPrintf)("Can't set floatingpoint handler\n");
  } else if (ieee_handler("clear","common",PetscDefaultFPTrap)) (*PetscErrorPrintf)("Can't clear floatingpoint handler\n");

  _trapmode = flag;
  PetscFunctionReturn(0);
}

/*@
   PetscDetermineInitialFPTrap - Attempts to determine the floating point trapping that exists when PetscInitialize() is called

   Not Collective

   Notes:
      Currently only supported on Linux and MacOS. Checks if divide by zero is enable and if so declares that trapping is on.

   Level: advanced

.seealso: `PetscFPTrapPush()`, `PetscFPTrapPop()`, `PetscDetermineInitialFPTrap()`
@*/
PetscErrorCode  PetscDetermineInitialFPTrap(void)
{
  PetscFunctionBegin;
  PetscCall(PetscInfo(NULL,"Unable to determine initial floating point trapping. Assuming it is off\n"));
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------------------*/
#elif defined(PETSC_HAVE_SOLARIS_STYLE_FPTRAP)
#include <sunmath.h>
#include <floatingpoint.h>
#include <siginfo.h>
#include <ucontext.h>

static struct { int code_no; char *name; } error_codes[] = {
  { FPE_FLTINV,"invalid floating point operand"},
  { FPE_FLTRES,"inexact floating point result"},
  { FPE_FLTDIV,"division-by-zero"},
  { FPE_FLTUND,"floating point underflow"},
  { FPE_FLTOVF,"floating point overflow"},
  { 0,         "unknown error"}
};
#define SIGPC(scp) (scp->si_addr)

void PetscDefaultFPTrap(int sig,siginfo_t *scp,ucontext_t *uap)
{
  int err_ind = -1,code = scp->si_code;

  PetscFunctionBegin;
  for (int j = 0; error_codes[j].code_no; j++) {
    if (error_codes[j].code_no == code) err_ind = j;
  }

  if (err_ind >= 0) (*PetscErrorPrintf)("*** %s occurred at pc=%X ***\n",error_codes[err_ind].name,SIGPC(scp));
  else              (*PetscErrorPrintf)("*** floating point error 0x%x occurred at pc=%X ***\n",code,SIGPC(scp));

  (void)PetscError(PETSC_COMM_SELF,0,"User provided function","Unknown file",PETSC_ERR_FP,PETSC_ERROR_REPEAT,"floating point error");
  PETSCABORT(MPI_COMM_WORLD,PETSC_ERR_FP);
}

PetscErrorCode PetscSetFPTrap(PetscFPTrap flag)
{
  char *out;

  PetscFunctionBegin;
  /* Clear accumulated exceptions.  Used to suppress meaningless messages from f77 programs */
  (void) ieee_flags("clear","exception","all",&out);
  if (flag == PETSC_FP_TRAP_ON) {
    if (ieee_handler("set","common",(sigfpe_handler_type)PetscDefaultFPTrap))        (*PetscErrorPrintf)("Can't set floating point handler\n");
  } else {
    if (ieee_handler("clear","common",(sigfpe_handler_type)PetscDefaultFPTrap)) (*PetscErrorPrintf)("Can't clear floatingpoint handler\n");
  }
  _trapmode = flag;
  PetscFunctionReturn(0);
}

PetscErrorCode  PetscDetermineInitialFPTrap(void)
{
  PetscFunctionBegin;
  PetscCall(PetscInfo(NULL,"Unable to determine initial floating point trapping. Assuming it is off\n"));
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------------------------*/
#elif defined(PETSC_HAVE_IRIX_STYLE_FPTRAP)
#include <sigfpe.h>
static struct { int code_no; char *name; } error_codes[] = {
  { _INVALID   ,"IEEE operand error" },
  { _OVERFL    ,"floating point overflow" },
  { _UNDERFL   ,"floating point underflow" },
  { _DIVZERO   ,"floating point divide" },
  { 0          ,"unknown error" }
} ;
void PetscDefaultFPTrap(unsigned exception[],int val[])
{
  int err_ind = -1,code = exception[0];

  PetscFunctionBegin;
  for (int j = 0; error_codes[j].code_no; j++) {
    if (error_codes[j].code_no == code) err_ind = j;
  }
  if (err_ind >= 0) (*PetscErrorPrintf)("*** %s occurred ***\n",error_codes[err_ind].name);
  else              (*PetscErrorPrintf)("*** floating point error 0x%x occurred ***\n",code);

  (void)PetscError(PETSC_COMM_SELF,0,"User provided function","Unknown file",PETSC_ERR_FP,PETSC_ERROR_REPEAT,"floating point error");
  PETSCABORT(MPI_COMM_WORLD,PETSC_ERR_FP);
}

PetscErrorCode PetscSetFPTrap(PetscFPTrap flag)
{
  PetscFunctionBegin;
  if (flag == PETSC_FP_TRAP_ON) handle_sigfpes(_ON,,_EN_UNDERFL|_EN_OVERFL|_EN_DIVZERO|_EN_INVALID,PetscDefaultFPTrap,_ABORT_ON_ERROR,0);
  else                          handle_sigfpes(_OFF,_EN_UNDERFL|_EN_OVERFL|_EN_DIVZERO|_EN_INVALID,0,_ABORT_ON_ERROR,0);
  _trapmode = flag;
  PetscFunctionReturn(0);
}

PetscErrorCode  PetscDetermineInitialFPTrap(void)
{
  PetscFunctionBegin;
  PetscCall(PetscInfo(NULL,"Unable to determine initial floating point trapping. Assuming it is off\n"));
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------------------*/
#elif defined(PETSC_HAVE_SOLARIS_STYLE_FPTRAP)
#include <sunmath.h>
#include <floatingpoint.h>
#include <siginfo.h>
#include <ucontext.h>

static struct { int code_no; char *name; } error_codes[] = {
  { FPE_FLTINV,"invalid floating point operand"},
  { FPE_FLTRES,"inexact floating point result"},
  { FPE_FLTDIV,"division-by-zero"},
  { FPE_FLTUND,"floating point underflow"},
  { FPE_FLTOVF,"floating point overflow"},
  { 0,         "unknown error"}
};
#define SIGPC(scp) (scp->si_addr)

void PetscDefaultFPTrap(int sig,siginfo_t *scp,ucontext_t *uap)
{
  int err_ind = -1,code = scp->si_code;

  PetscFunctionBegin;
  err_ind = -1;
  for (int j = 0; error_codes[j].code_no; j++) {
    if (error_codes[j].code_no == code) err_ind = j;
  }

  if (err_ind >= 0) (*PetscErrorPrintf)("*** %s occurred at pc=%X ***\n",error_codes[err_ind].name,SIGPC(scp));
  else              (*PetscErrorPrintf)("*** floating point error 0x%x occurred at pc=%X ***\n",code,SIGPC(scp));

  (void)PetscError(PETSC_COMM_SELF,0,"User provided function","Unknown file",PETSC_ERR_FP,PETSC_ERROR_REPEAT,"floating point error");
  PETSCABORT(MPI_COMM_WORLD,PETSC_ERR_FP);
}

PetscErrorCode PetscSetFPTrap(PetscFPTrap flag)
{
  char *out;

  PetscFunctionBegin;
  /* Clear accumulated exceptions.  Used to suppress meaningless messages from f77 programs */
  (void) ieee_flags("clear","exception","all",&out);
  if (flag == PETSC_FP_TRAP_ON) {
    if (ieee_handler("set","common",(sigfpe_handler_type)PetscDefaultFPTrap))        (*PetscErrorPrintf)("Can't set floating point handler\n");
  } else {
    if (ieee_handler("clear","common",(sigfpe_handler_type)PetscDefaultFPTrap)) (*PetscErrorPrintf)("Can't clear floatingpoint handler\n");
  }
  _trapmode = flag;
  PetscFunctionReturn(0);
}

PetscErrorCode  PetscDetermineInitialFPTrap(void)
{
  PetscFunctionBegin;
  PetscCall(PetscInfo(NULL,"Unable to determine initial floating point trapping. Assuming it is off\n"));
  PetscFunctionReturn(0);
}

/*----------------------------------------------- --------------------------------------------*/
#elif defined(PETSC_HAVE_RS6000_STYLE_FPTRAP)
/* In "fast" mode, floating point traps are imprecise and ignored.
   This is the reason for the fptrap(FP_TRAP_SYNC) call */
struct sigcontext;
#include <fpxcp.h>
#include <fptrap.h>
#define FPE_FLTOPERR_TRAP (fptrap_t)(0x20000000)
#define FPE_FLTOVF_TRAP   (fptrap_t)(0x10000000)
#define FPE_FLTUND_TRAP   (fptrap_t)(0x08000000)
#define FPE_FLTDIV_TRAP   (fptrap_t)(0x04000000)
#define FPE_FLTINEX_TRAP  (fptrap_t)(0x02000000)

static struct { int code_no; char *name; } error_codes[] = {
  {FPE_FLTOPERR_TRAP   ,"IEEE operand error" },
  { FPE_FLTOVF_TRAP    ,"floating point overflow" },
  { FPE_FLTUND_TRAP    ,"floating point underflow" },
  { FPE_FLTDIV_TRAP    ,"floating point divide" },
  { FPE_FLTINEX_TRAP   ,"inexact floating point result" },
  { 0                  ,"unknown error" }
} ;
#define SIGPC(scp) (0) /* Info MIGHT be in scp->sc_jmpbuf.jmp_context.iar */
/*
   For some reason, scp->sc_jmpbuf does not work on the RS6000, even though
   it looks like it should from the include definitions.  It is probably
   some strange interaction with the "POSIX_SOURCE" that we require.
*/

void PetscDefaultFPTrap(int sig,int code,struct sigcontext *scp)
{
  int      err_ind,j;
  fp_ctx_t flt_context;

  PetscFunctionBegin;
  fp_sh_trap_info(scp,&flt_context);

  err_ind = -1;
  for (j = 0; error_codes[j].code_no; j++) {
    if (error_codes[j].code_no == flt_context.trap) err_ind = j;
  }

  if (err_ind >= 0) (*PetscErrorPrintf)("*** %s occurred ***\n",error_codes[err_ind].name);
  else              (*PetscErrorPrintf)("*** floating point error 0x%x occurred ***\n",flt_context.trap);

  (void)PetscError(PETSC_COMM_SELF,0,"User provided function","Unknown file",PETSC_ERR_FP,PETSC_ERROR_REPEAT,"floating point error");
  PETSCABORT(MPI_COMM_WORLD,PETSC_ERR_FP);
}

PetscErrorCode PetscSetFPTrap(PetscFPTrap on)
{
  PetscFunctionBegin;
  if (on == PETSC_FP_TRAP_ON) {
    signal(SIGFPE,(void (*)(int))PetscDefaultFPTrap);
    fp_trap(FP_TRAP_SYNC);
    fp_enable(TRP_INVALID | TRP_DIV_BY_ZERO | TRP_OVERFLOW | TRP_UNDERFLOW);
    /* fp_enable(mask) for individual traps.  Values are:
       TRP_INVALID
       TRP_DIV_BY_ZERO
       TRP_OVERFLOW
       TRP_UNDERFLOW
       TRP_INEXACT
       Can OR then together.
       fp_enable_all(); for all traps.
    */
  } else {
    signal(SIGFPE,SIG_DFL);
    fp_disable(TRP_INVALID | TRP_DIV_BY_ZERO | TRP_OVERFLOW | TRP_UNDERFLOW);
    fp_trap(FP_TRAP_OFF);
  }
  _trapmode = on;
  PetscFunctionReturn(0);
}

PetscErrorCode  PetscDetermineInitialFPTrap(void)
{
  PetscFunctionBegin;
  PetscCall(PetscInfo(NULL,"Unable to determine initial floating point trapping. Assuming it is off\n"));
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------*/
#elif defined(PETSC_HAVE_WINDOWS_COMPILERS)
#include <float.h>
void PetscDefaultFPTrap(int sig)
{
  PetscFunctionBegin;
  (*PetscErrorPrintf)("*** floating point error occurred ***\n");
  PetscError(PETSC_COMM_SELF,0,"User provided function","Unknown file",PETSC_ERR_FP,PETSC_ERROR_REPEAT,"floating point error");
  PETSCABORT(MPI_COMM_WORLD,PETSC_ERR_FP);
}

PetscErrorCode  PetscSetFPTrap(PetscFPTrap on)
{
  unsigned int cw;

  PetscFunctionBegin;
  if (on == PETSC_FP_TRAP_ON) {
    cw = _EM_INVALID | _EM_ZERODIVIDE | _EM_OVERFLOW | _EM_UNDERFLOW;
    PetscCheck(SIG_ERR != signal(SIGFPE,PetscDefaultFPTrap),PETSC_COMM_SELF,PETSC_ERR_LIB,"Can't set floating point handler");
  } else {
    cw = 0;
    PetscCheck(SIG_ERR != signal(SIGFPE,SIG_DFL),PETSC_COMM_SELF,PETSC_ERR_LIB,"Can't clear floating point handler");
  }
  (void)_controlfp(0, cw);
  _trapmode = on;
  PetscFunctionReturn(0);
}

PetscErrorCode  PetscDetermineInitialFPTrap(void)
{
  PetscFunctionBegin;
  PetscCall(PetscInfo(NULL,"Unable to determine initial floating point trapping. Assuming it is off\n"));
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------*/
#elif defined(PETSC_HAVE_FENV_H) && !defined(__cplusplus)
/*
   C99 style floating point environment.

   Note that C99 merely specifies how to save, restore, and clear the floating
   point environment as well as defining an enumeration of exception codes.  In
   particular, C99 does not specify how to make floating point exceptions raise
   a signal.  Glibc offers this capability through FE_NOMASK_ENV (or with finer
   granularity, feenableexcept()), xmmintrin.h offers _MM_SET_EXCEPTION_MASK().
*/
#include <fenv.h>
typedef struct {int code; const char *name;} FPNode;
static const FPNode error_codes[] = {
  {FE_DIVBYZERO,"divide by zero"},
  {FE_INEXACT,  "inexact floating point result"},
  {FE_INVALID,  "invalid floating point arguments (domain error)"},
  {FE_OVERFLOW, "floating point overflow"},
  {FE_UNDERFLOW,"floating point underflow"},
  {0           ,"unknown error"}
};

void PetscDefaultFPTrap(int sig)
{
  const FPNode *node;
  int          code;
  PetscBool    matched = PETSC_FALSE;

  PetscFunctionBegin;
  /* Note: While it is possible for the exception state to be preserved by the
   * kernel, this seems to be rare which makes the following flag testing almost
   * useless.  But on a system where the flags can be preserved, it would provide
   * more detail.
   */
  code = fetestexcept(FE_ALL_EXCEPT);
  for (node=&error_codes[0]; node->code; node++) {
    if (code & node->code) {
      matched = PETSC_TRUE;
      (*PetscErrorPrintf)("*** floating point error \"%s\" occurred ***\n",node->name);
      code &= ~node->code; /* Unset this flag since it has been processed */
    }
  }
  if (!matched || code) { /* If any remaining flags are set, or we didn't process any flags */
    (*PetscErrorPrintf)("*** unknown floating point error occurred ***\n");
    (*PetscErrorPrintf)("The specific exception can be determined by running in a debugger.  When the\n");
    (*PetscErrorPrintf)("debugger traps the signal, the exception can be found with fetestexcept(0x%x)\n",FE_ALL_EXCEPT);
    (*PetscErrorPrintf)("where the result is a bitwise OR of the following flags:\n");
    (*PetscErrorPrintf)("FE_INVALID=0x%x FE_DIVBYZERO=0x%x FE_OVERFLOW=0x%x FE_UNDERFLOW=0x%x FE_INEXACT=0x%x\n",FE_INVALID,FE_DIVBYZERO,FE_OVERFLOW,FE_UNDERFLOW,FE_INEXACT);
  }

  (*PetscErrorPrintf)("Try option -start_in_debugger\n");
#if PetscDefined(USE_DEBUG)
  (*PetscErrorPrintf)("likely location of problem given in stack below\n");
  (*PetscErrorPrintf)("---------------------  Stack Frames ------------------------------------\n");
  PetscStackView(PETSC_STDOUT);
#else
  (*PetscErrorPrintf)("configure using --with-debugging=yes, recompile, link, and run \n");
  (*PetscErrorPrintf)("with -start_in_debugger to get more information on the crash.\n");
#endif
  PetscError(PETSC_COMM_SELF,0,"User provided function","Unknown file",PETSC_ERR_FP,PETSC_ERROR_INITIAL,"trapped floating point error");
  PETSCABORT(MPI_COMM_WORLD,PETSC_ERR_FP);
}

PetscErrorCode  PetscSetFPTrap(PetscFPTrap on)
{
  PetscFunctionBegin;
  if (on == PETSC_FP_TRAP_ON) {
    /* Clear any flags that are currently set so that activating trapping will not immediately call the signal handler. */
    PetscCheckFalse(feclearexcept(FE_ALL_EXCEPT),PETSC_COMM_SELF,PETSC_ERR_LIB,"Cannot clear floating point exception flags");
#if defined(FE_NOMASK_ENV)
    /* Could use fesetenv(FE_NOMASK_ENV), but that causes spurious exceptions (like gettimeofday() -> PetscLogDouble). */
    PetscCheck(feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW | FE_UNDERFLOW) != -1,PETSC_COMM_SELF,PETSC_ERR_LIB,"Cannot activate floating point exceptions");
#elif defined PETSC_HAVE_XMMINTRIN_H
   _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~_MM_MASK_DIV_ZERO);
   _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~_MM_MASK_UNDERFLOW);
   _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~_MM_MASK_OVERFLOW);
   _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~_MM_MASK_INVALID);
#else
    /* C99 does not provide a way to modify the environment so there is no portable way to activate trapping. */
#endif
    PetscCheck(SIG_ERR != signal(SIGFPE,PetscDefaultFPTrap),PETSC_COMM_SELF,PETSC_ERR_LIB,"Can't set floating point handler");
  } else {
    PetscCheckFalse(fesetenv(FE_DFL_ENV),PETSC_COMM_SELF,PETSC_ERR_LIB,"Cannot disable floating point exceptions");
    /* can use _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() | _MM_MASK_UNDERFLOW); if PETSC_HAVE_XMMINTRIN_H exists */
    PetscCheck(SIG_ERR != signal(SIGFPE,SIG_DFL),PETSC_COMM_SELF,PETSC_ERR_LIB,"Can't clear floating point handler");
  }
  _trapmode = on;
  PetscFunctionReturn(0);
}

PetscErrorCode  PetscDetermineInitialFPTrap(void)
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
  PetscCall(PetscInfo(NULL,"Floating point trapping unknown, assuming off\n"));
  PetscFunctionReturn(0);
#endif
#if defined(FE_NOMASK_ENV) || defined PETSC_HAVE_XMMINTRIN_H
    _trapmode = PETSC_FP_TRAP_ON;
    PetscCall(PetscInfo(NULL,"Floating point trapping is on by default %d\n",flags));
  } else {
    _trapmode = PETSC_FP_TRAP_OFF;
    PetscCall(PetscInfo(NULL,"Floating point trapping is off by default %d\n",flags));
  }
  PetscFunctionReturn(0);
#endif
}

/* ------------------------------------------------------------*/
#elif defined(PETSC_HAVE_IEEEFP_H)
#include <ieeefp.h>
void PetscDefaultFPTrap(int sig)
{
  PetscFunctionBegin;
  (*PetscErrorPrintf)("*** floating point error occurred ***\n");
  PetscError(PETSC_COMM_SELF,0,"User provided function","Unknown file",PETSC_ERR_FP,PETSC_ERROR_REPEAT,"floating point error");
  PETSCABORT(MPI_COMM_WORLD,PETSC_ERR_FP);
}

PetscErrorCode  PetscSetFPTrap(PetscFPTrap on)
{
  PetscFunctionBegin;
  if (on == PETSC_FP_TRAP_ON) {
#if defined(PETSC_HAVE_FPPRESETSTICKY)
    fpresetsticky(fpgetsticky());
#elif defined(PETSC_HAVE_FPSETSTICKY)
    fpsetsticky(fpgetsticky());
#endif
    fpsetmask(FP_X_INV | FP_X_DZ | FP_X_OFL |  FP_X_OFL);
    PetscCheck(SIG_ERR != signal(SIGFPE,PetscDefaultFPTrap),PETSC_COMM_SELF,PETSC_ERR_LIB,"Can't set floating point handler");
  } else {
#if defined(PETSC_HAVE_FPPRESETSTICKY)
    fpresetsticky(fpgetsticky());
#elif defined(PETSC_HAVE_FPSETSTICKY)
    fpsetsticky(fpgetsticky());
#endif
    fpsetmask(0);
    PetscCheck(SIG_ERR != signal(SIGFPE,SIG_DFL),PETSC_COMM_SELF,PETSC_ERR_LIB,"Can't clear floating point handler");
  }
  _trapmode = on;
  PetscFunctionReturn(0);
}

PetscErrorCode  PetscDetermineInitialFPTrap(void)
{
  PetscFunctionBegin;
  PetscCall(PetscInfo(NULL,"Unable to determine initial floating point trapping. Assuming it is off\n"));
  PetscFunctionReturn(0);
}

/* -------------------------Default -----------------------------------*/
#else

void PetscDefaultFPTrap(int sig)
{
  PetscFunctionBegin;
  (*PetscErrorPrintf)("*** floating point error occurred ***\n");
  PetscError(PETSC_COMM_SELF,0,"User provided function","Unknown file",PETSC_ERR_FP,PETSC_ERROR_REPEAT,"floating point error");
  PETSCABORT(MPI_COMM_WORLD,PETSC_ERR_FP);
}

PetscErrorCode  PetscSetFPTrap(PetscFPTrap on)
{
  PetscFunctionBegin;
  if (on == PETSC_FP_TRAP_ON) {
    if (SIG_ERR == signal(SIGFPE,PetscDefaultFPTrap)) (*PetscErrorPrintf)("Can't set floatingpoint handler\n");
  } else if (SIG_ERR == signal(SIGFPE,SIG_DFL))       (*PetscErrorPrintf)("Can't clear floatingpoint handler\n");

  _trapmode = on;
  PetscFunctionReturn(0);
}

PetscErrorCode  PetscDetermineInitialFPTrap(void)
{
  PetscFunctionBegin;
  PetscCall(PetscInfo(NULL,"Unable to determine initial floating point trapping. Assuming it is off\n"));
  PetscFunctionReturn(0);
}
#endif
