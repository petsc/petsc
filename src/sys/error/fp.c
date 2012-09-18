
/*
*	IEEE error handler for all machines. Since each machine has
*   enough slight differences we have completely separate codes for each one.
*
*/

/*
  This feature test macro provides FE_NOMASK_ENV on GNU.  It must be defined
  at the top of the file because other headers may pull in fenv.h even when
  not strictly necessary.  Strictly speaking, we could include ONLY petscconf.h,
  check PETSC_HAVE_FENV_H, and only define _GNU_SOURCE in that case, but such
  shenanigans ought to be unnecessary.
*/
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <petscsys.h>           /*I  "petscsys.h"  I*/
#include <signal.h>
#if defined(PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif

struct PetscFPTrapLink {
  PetscFPTrap trapmode;
  struct PetscFPTrapLink *next;
};
static PetscFPTrap _trapmode = PETSC_FP_TRAP_OFF; /* Current trapping mode */
static struct PetscFPTrapLink *_trapstack;        /* Any pushed states of _trapmode */

#undef __FUNCT__
#define __FUNCT__ "PetscFPTrapPush"
/*@
   PetscFPTrapPush - push a floating point trapping mode, to be restored using PetscFPTrapPop()

   Not Collective

   Input Arguments:
. trap - PETSC_FP_TRAP_ON or PETSC_FP_TRAP_OFF

   Level: advanced

.seealso: PetscFPTrapPop(), PetscSetFPTrap()
@*/
PetscErrorCode PetscFPTrapPush(PetscFPTrap trap)
{
  PetscErrorCode ierr;
  struct PetscFPTrapLink *link;

  PetscFunctionBegin;
  ierr = PetscMalloc(sizeof(*link),&link);CHKERRQ(ierr);
  link->trapmode = _trapmode;
  link->next = _trapstack;
  _trapstack = link;
  if (trap != _trapmode) {ierr = PetscSetFPTrap(trap);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscFPTrapPop"
/*@
   PetscFPTrapPop - push a floating point trapping mode, to be restored using PetscFPTrapPop()

   Not Collective

   Level: advanced

.seealso: PetscFPTrapPush(), PetscSetFPTrap()
@*/
PetscErrorCode PetscFPTrapPop(void)
{
  PetscErrorCode ierr;
  struct PetscFPTrapLink *link;

  PetscFunctionBegin;
  if (_trapstack->trapmode != _trapmode) {ierr = PetscSetFPTrap(_trapstack->trapmode);CHKERRQ(ierr);}
  link = _trapstack;
  _trapstack = _trapstack->next;
  ierr = PetscFree(link);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*--------------------------------------- ---------------------------------------------------*/
#if defined(PETSC_HAVE_SUN4_STYLE_FPTRAP)
#include <floatingpoint.h>

EXTERN_C_BEGIN
PetscErrorCode ieee_flags(char*,char*,char*,char**);
PetscErrorCode ieee_handler(char *,char *,sigfpe_handler_type(int,int,struct sigcontext*,char *));
EXTERN_C_END

static struct { int code_no; char *name; } error_codes[] = {
           { FPE_INTDIV_TRAP	,"integer divide" },
	   { FPE_FLTOPERR_TRAP	,"IEEE operand error" },
	   { FPE_FLTOVF_TRAP	,"floating point overflow" },
	   { FPE_FLTUND_TRAP	,"floating point underflow" },
	   { FPE_FLTDIV_TRAP	,"floating pointing divide" },
	   { FPE_FLTINEX_TRAP	,"inexact floating point result" },
	   { 0			,"unknown error" }
} ;
#define SIGPC(scp) (scp->sc_pc)

#undef __FUNCT__
#define __FUNCT__ "PetscDefaultFPTrap"
sigfpe_handler_type PetscDefaultFPTrap(int sig,int code,struct sigcontext *scp,char *addr)
{
  PetscErrorCode ierr;
  int err_ind = -1,j;

  PetscFunctionBegin;
  for (j = 0 ; error_codes[j].code_no ; j++) {
    if (error_codes[j].code_no == code) err_ind = j;
  }

  if (err_ind >= 0) {
    (*PetscErrorPrintf)("*** %s occurred at pc=%X ***\n",error_codes[err_ind].name,SIGPC(scp));
  } else {
    (*PetscErrorPrintf)("*** floating point error 0x%x occurred at pc=%X ***\n",code,SIGPC(scp));
  }
  ierr = PetscError(PETSC_COMM_SELF,PETSC_ERR_FP,"User provided function","Unknown file","Unknown directory",PETSC_ERR_FP,PETSC_ERROR_REPEAT,"floating point error");
  MPI_Abort(PETSC_COMM_WORLD,0);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSetFPTrap"
/*@
   PetscSetFPTrap - Enables traps/exceptions on common floating point errors.
                    This option may not work on certain machines.

   Not Collective

   Input Parameters:
.  flag - PETSC_FP_TRAP_ON, PETSC_FP_TRAP_OFF.

   Options Database Keys:
.  -fp_trap - Activates floating point trapping

   Level: advanced

   Description:
   On systems that support it, this routine causes floating point
   overflow, divide-by-zero, and invalid-operand (e.g., a NaN) to
   cause a message to be printed and the program to exit.

   Note:
   On many common systems including x86 and x86-64 Linux, the floating
   point exception state is not preserved from the location where the trap
   occurred through to the signal handler.  In this case, the signal handler
   will just say that an unknown floating point exception occurred and which
   function it occurred in.  If you run with -fp_trap in a debugger, it will
   break on the line where the error occurred.  You can check which
   exception occurred using fetestexcept(FE_ALL_EXCEPT).  See fenv.h
   (usually at /usr/include/bits/fenv.h) for the enum values on your system.

   Caution:
   On certain machines, in particular the IBM rs6000, floating point
   trapping is VERY slow!

   Concepts: floating point exceptions^trapping
   Concepts: divide by zero

.seealso: PetscFPTrapPush(), PetscFPTrapPop()
@*/
PetscErrorCode PetscSetFPTrap(PetscFPTrap flag)
{
  char *out;

  PetscFunctionBegin;
  /* Clear accumulated exceptions.  Used to suppress meaningless messages from f77 programs */
  (void) ieee_flags("clear","exception","all",&out);
  if (flag == PETSC_FP_TRAP_ON) {
    if (ieee_handler("set","common",PetscDefaultFPTrap)) {
      /*
        To trap more fp exceptions, including undrflow, change the above line to
        if (ieee_handler("set","all",PetscDefaultFPTrap)) {
      */
      (*PetscErrorPrintf)("Can't set floatingpoint handler\n");
    }
  } else {
    if (ieee_handler("clear","common",PetscDefaultFPTrap)) {
      (*PetscErrorPrintf)("Can't clear floatingpoint handler\n");
    }
  }
  _trapmode = flag;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------------------*/
#elif defined(PETSC_HAVE_SOLARIS_STYLE_FPTRAP)
#include <sunmath.h>
#include <floatingpoint.h>
#include <siginfo.h>
#include <ucontext.h>

static struct { int code_no; char *name; } error_codes[] = {
  {  FPE_FLTINV,"invalid floating point operand"},
  {  FPE_FLTRES,"inexact floating point result"},
  {  FPE_FLTDIV,"division-by-zero"},
  {  FPE_FLTUND,"floating point underflow"},
  {  FPE_FLTOVF,"floating point overflow"},
  {  0,         "unknown error"}
};
#define SIGPC(scp) (scp->si_addr)

#undef __FUNCT__
#define __FUNCT__ "PetscDefaultFPTrap"
void PetscDefaultFPTrap(int sig,siginfo_t *scp,ucontext_t *uap)
{
  int err_ind,j,code = scp->si_code;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  err_ind = -1 ;
  for (j = 0 ; error_codes[j].code_no ; j++) {
    if (error_codes[j].code_no == code) err_ind = j;
  }

  if (err_ind >= 0) {
    (*PetscErrorPrintf)("*** %s occurred at pc=%X ***\n",error_codes[err_ind].name,SIGPC(scp));
  } else {
    (*PetscErrorPrintf)("*** floating point error 0x%x occurred at pc=%X ***\n",code,SIGPC(scp));
  }
  ierr = PetscError(PETSC_COMM_SELF,0,"User provided function","Unknown file","Unknown directory",PETSC_ERR_FP,PETSC_ERROR_REPEAT,"floating point error");
  MPI_Abort(PETSC_COMM_WORLD,0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSetFPTrap"
PetscErrorCode PetscSetFPTrap(PetscFPTrap flag)
{
  char *out;

  PetscFunctionBegin;
  /* Clear accumulated exceptions.  Used to suppress meaningless messages from f77 programs */
  (void) ieee_flags("clear","exception","all",&out);
  if (flag == PETSC_FP_TRAP_ON) {
    if (ieee_handler("set","common",(sigfpe_handler_type)PetscDefaultFPTrap)) {
      (*PetscErrorPrintf)("Can't set floating point handler\n");
    }
  } else {
    if (ieee_handler("clear","common",(sigfpe_handler_type)PetscDefaultFPTrap)) {
     (*PetscErrorPrintf)("Can't clear floatingpoint handler\n");
    }
  }
  _trapmode = flag;
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------------------------*/

#elif defined (PETSC_HAVE_IRIX_STYLE_FPTRAP)
#include <sigfpe.h>
static struct { int code_no; char *name; } error_codes[] = {
       { _INVALID   ,"IEEE operand error" },
       { _OVERFL    ,"floating point overflow" },
       { _UNDERFL   ,"floating point underflow" },
       { _DIVZERO   ,"floating point divide" },
       { 0          ,"unknown error" }
} ;
#undef __FUNCT__
#define __FUNCT__ "PetscDefaultFPTrap"
void PetscDefaultFPTrap(unsigned exception[],int val[])
{
  int err_ind,j,code;

  PetscFunctionBegin;
  code = exception[0];
  err_ind = -1 ;
  for (j = 0 ; error_codes[j].code_no ; j++){
    if (error_codes[j].code_no == code) err_ind = j;
  }
  if (err_ind >= 0){
    (*PetscErrorPrintf)("*** %s occurred ***\n",error_codes[err_ind].name);
  } else{
    (*PetscErrorPrintf)("*** floating point error 0x%x occurred ***\n",code);
  }
  PetscError(PETSC_COMM_SELF,0,"User provided function","Unknown file","Unknown directory",PETSC_ERR_FP,PETSC_ERROR_REPEAT,"floating point error");
  MPI_Abort(PETSC_COMM_WORLD,0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSetFPTrap"
PetscErrorCode PetscSetFPTrap(PetscFPTrap flag)
{
  PetscFunctionBegin;
  if (flag == PETSC_FP_TRAP_ON) {
    handle_sigfpes(_ON,_EN_OVERFL|_EN_DIVZERO|_EN_INVALID,PetscDefaultFPTrap,_ABORT_ON_ERROR,0);
  } else {
    handle_sigfpes(_OFF,_EN_OVERFL|_EN_DIVZERO|_EN_INVALID,0,_ABORT_ON_ERROR,0);
  }
  _trapmode = flag;
  PetscFunctionReturn(0);
}
/*----------------------------------------------- --------------------------------------------*/
/* In "fast" mode, floating point traps are imprecise and ignored.
   This is the reason for the fptrap(FP_TRAP_SYNC) call */
#elif defined(PETSC_HAVE_RS6000_STYLE_FPTRAP)
struct sigcontext;
#include <fpxcp.h>
#include <fptrap.h>
#include <stdlib.h>
#define FPE_FLTOPERR_TRAP (fptrap_t)(0x20000000)
#define FPE_FLTOVF_TRAP   (fptrap_t)(0x10000000)
#define FPE_FLTUND_TRAP   (fptrap_t)(0x08000000)
#define FPE_FLTDIV_TRAP   (fptrap_t)(0x04000000)
#define FPE_FLTINEX_TRAP  (fptrap_t)(0x02000000)

static struct { int code_no; char *name; } error_codes[] = {
           {FPE_FLTOPERR_TRAP	,"IEEE operand error" },
	   { FPE_FLTOVF_TRAP	,"floating point overflow" },
	   { FPE_FLTUND_TRAP	,"floating point underflow" },
	   { FPE_FLTDIV_TRAP	,"floating point divide" },
	   { FPE_FLTINEX_TRAP	,"inexact floating point result" },
	   { 0			,"unknown error" }
} ;
#define SIGPC(scp) (0) /* Info MIGHT be in scp->sc_jmpbuf.jmp_context.iar */
/*
   For some reason, scp->sc_jmpbuf does not work on the RS6000, even though
   it looks like it should from the include definitions.  It is probably
   some strange interaction with the "POSIX_SOURCE" that we require.
*/

#undef __FUNCT__
#define __FUNCT__ "PetscDefaultFPTrap"
void PetscDefaultFPTrap(int sig,int code,struct sigcontext *scp)
{
  PetscErrorCode ierr;
  int      err_ind,j;
  fp_ctx_t flt_context;

  PetscFunctionBegin;
  fp_sh_trap_info(scp,&flt_context);

  err_ind = -1 ;
  for (j = 0 ; error_codes[j].code_no ; j++) {
    if (error_codes[j].code_no == flt_context.trap) err_ind = j;
  }

  if (err_ind >= 0){
    (*PetscErrorPrintf)("*** %s occurred ***\n",error_codes[err_ind].name);
  } else{
    (*PetscErrorPrintf)("*** floating point error 0x%x occurred ***\n",flt_context.trap);
  }
  ierr = PetscError(PETSC_COMM_SELF,0,"User provided function","Unknown file","Unknown directory",PETSC_ERR_FP,PETSC_ERROR_REPEAT,"floating point error");
  MPI_Abort(PETSC_COMM_WORLD,0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSetFPTrap"
PetscErrorCode PetscSetFPTrap(PetscFPTrap on)
{
  PetscFunctionBegin;
  if (on == PETSC_FP_TRAP_ON) {
    signal(SIGFPE,(void (*)(int))PetscDefaultFPTrap);
    fp_trap(FP_TRAP_SYNC);
    fp_enable(TRP_INVALID | TRP_DIV_BY_ZERO | TRP_OVERFLOW);
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
    fp_disable(TRP_INVALID | TRP_DIV_BY_ZERO | TRP_OVERFLOW);
    fp_trap(FP_TRAP_OFF);
  }
  _trapmode = on;
  PetscFunctionReturn(0);
}

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
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PetscDefaultFPTrap"
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
#if defined(PETSC_USE_DEBUG)
  if (!PetscStackActive) {
    (*PetscErrorPrintf)("  or try option -log_stack\n");
  } else {
    (*PetscErrorPrintf)("likely location of problem given in stack below\n");
    (*PetscErrorPrintf)("---------------------  Stack Frames ------------------------------------\n");
    PetscStackView(PETSC_VIEWER_STDOUT_SELF);
  }
#endif
#if !defined(PETSC_USE_DEBUG)
  (*PetscErrorPrintf)("configure using --with-debugging=yes, recompile, link, and run \n");
  (*PetscErrorPrintf)("with -start_in_debugger to get more information on the crash.\n");
#endif
  PetscError(PETSC_COMM_SELF,0,"User provided function","Unknown file","Unknown directory",PETSC_ERR_FP,PETSC_ERROR_INITIAL,"trapped floating point error");
  MPI_Abort(PETSC_COMM_WORLD,0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "PetscSetFPTrap"
PetscErrorCode  PetscSetFPTrap(PetscFPTrap on)
{
  PetscFunctionBegin;
  if (on == PETSC_FP_TRAP_ON) {
    /* Clear any flags that are currently set so that activating trapping will not immediately call the signal handler. */
    if (feclearexcept(FE_ALL_EXCEPT)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Cannot clear floating point exception flags\n");
#if defined FE_NOMASK_ENV
    /* We could use fesetenv(FE_NOMASK_ENV), but that causes spurious exceptions (like gettimeofday() -> PetscLogDouble). */
    if (feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW | FE_UNDERFLOW) == -1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Cannot activate floating point exceptions\n");
#elif defined PETSC_HAVE_XMMINTRIN_H
    _MM_SET_EXCEPTION_MASK(_MM_MASK_INEXACT);
#else
    /* C99 does not provide a way to modify the environment so there is no portable way to activate trapping. */
#endif
    if (SIG_ERR == signal(SIGFPE,PetscDefaultFPTrap)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Can't set floating point handler\n");
  } else {
    if (fesetenv(FE_DFL_ENV)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Cannot disable floating point exceptions");
    if (SIG_ERR == signal(SIGFPE,SIG_DFL)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Can't clear floating point handler\n");
  }
  _trapmode = on;
  PetscFunctionReturn(0);
}

/* -------------------------Default -----------------------------------*/
#else
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PetscDefaultFPTrap"
void PetscDefaultFPTrap(int sig)
{
  PetscFunctionBegin;
  (*PetscErrorPrintf)("*** floating point error occurred ***\n");
  PetscError(PETSC_COMM_SELF,0,"User provided function","Unknown file","Unknown directory",PETSC_ERR_FP,PETSC_ERROR_REPEAT,"floating point error");
  MPI_Abort(PETSC_COMM_WORLD,0);
}
EXTERN_C_END
#undef __FUNCT__
#define __FUNCT__ "PetscSetFPTrap"
PetscErrorCode  PetscSetFPTrap(PetscFPTrap on)
{
  PetscFunctionBegin;
  if (on == PETSC_FP_TRAP_ON) {
    if (SIG_ERR == signal(SIGFPE,PetscDefaultFPTrap)) {
      (*PetscErrorPrintf)("Can't set floatingpoint handler\n");
    }
  } else {
    if (SIG_ERR == signal(SIGFPE,SIG_DFL)) {
      (*PetscErrorPrintf)("Can't clear floatingpoint handler\n");
    }
  }
  _trapmode = on;
  PetscFunctionReturn(0);
}
#endif



