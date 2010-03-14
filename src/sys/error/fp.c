#define PETSC_DLL
/*
*	IEEE error handler for all machines. Since each machine has 
*   enough slight differences we have completely separate codes for each one.
*
*/
#include "petscsys.h"           /*I  "petscsys.h"  I*/
#include <signal.h>
#if defined(PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif

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
  ierr = PetscError(PETSC_ERR_FP,"User provided function","Unknown file","Unknown directory",PETSC_ERR_FP,1,"floating point error");
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

   Caution:
   On certain machines, in particular the IBM rs6000, floating point 
   trapping is VERY slow!

   Concepts: floating point exceptions^trapping
   Concepts: divide by zero

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
  ierr = PetscError(0,"User provided function","Unknown file","Unknown directory",PETSC_ERR_FP,1,"floating point error");
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
  PetscError(0,"User provided function","Unknown file","Unknown directory",PETSC_ERR_FP,1,"floating point error");
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
  ierr = PetscError(0,"User provided function","Unknown file","Unknown directory",PETSC_ERR_FP,1,"floating point error");
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
  PetscError(0,"User provided function","Unknown file","Unknown directory",PETSC_ERR_FP,1,"floating point error");
  MPI_Abort(PETSC_COMM_WORLD,0);
}
EXTERN_C_END
#undef __FUNCT__  
#define __FUNCT__ "PetscSetFPTrap"
PetscErrorCode PETSC_DLLEXPORT PetscSetFPTrap(PetscFPTrap on)
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
  PetscFunctionReturn(0);
}
#endif



