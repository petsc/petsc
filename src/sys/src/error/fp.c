#ifndef lint
static char vcid[] = "$Id: pbvec.c,v 1.7 1995/03/06 03:56:21 bsmith Exp bsmith $";
#endif
/*
*	IEEE error handler for all machines. Since each machine has 
*   enough slight differences we have completely separate codes for each one.
*
*   This means there is a different one for sun4 4.1.3, Solaris and Meiko
*  not one incomprehensible one for all three, Bill.
*/
#include <signal.h>
#include <stdio.h>
#include "petsc.h"
#include "sys.h"

/*----------------IEEE error handler for Sun SparcStations.--------------*/
#if defined(PARCH_sun4) 
#include <floatingpoint.h>
int ieee_handler();
struct { int code_no; char *name; } error_codes[] = {
       { FPE_INTDIV_TRAP	, "integer divide" } ,
	   { FPE_FLTOPERR_TRAP	, "IEEE operand error" } ,
	   { FPE_FLTOVF_TRAP	, "floating overflow" } ,
	   { FPE_FLTUND_TRAP	, "floating underflow" } ,
	   { FPE_FLTDIV_TRAP	, "floating divide" } ,
	   { FPE_FLTINEX_TRAP	, "inexact floating result" } ,
	   { 0			, "unknown error" } 
} ;
#define SIGPC(scp) (scp->sc_pc)

#if defined(PARCH_sun4) && defined(__cplusplus)
extern "C" {
int exit(int);
};
#endif

sigfpe_handler_type SYsample_handler(int sig,int code,struct sigcontext *scp,
                                     char *addr)
{
  int err_ind, j,ierr;

  err_ind = -1 ;
  for ( j = 0 ; error_codes[j].code_no ; j++ )
    {if ( error_codes[j].code_no == code ) err_ind = j ;}

    if ( err_ind >= 0 )
      fprintf(stderr, "*** %s occurred at pc=%X ***\n",
			error_codes[err_ind].name, SIGPC(scp));
    else
      fprintf(stderr,
              "*** floating point error 0x%x occurred at pc=%X ***\n",
              code, SIGPC(scp));
    ierr = PetscError(0,0,"Unknown","floating point error",1);
    exit(ierr);
}

/*@
   PetscSetFPTrap - Enable traps/exceptions on common floating point errors

   Description:
   This routine, on systems that support it, causes floating point
   overflow, divide-by-zero, and invalid-operand (e.g., a NaN) to
   cause a message to be printed and the program to exit.

  Input Parameters:
.  flag - 1 to enable trapping, 0 to disable.

   Notes on specific implementations:
.   IBMRs6000  - Using this may slow down fp by a factor of 10.
.   Sun4       - Clears any pre-existing exceptions.
.   freebsd    - Seems to be ignored, reqular signal handler traps fp.
@*/
int PetscSetFPTrap(int flag)
{
  char *out; 
  /* Clear accumulated exceptions.  Used to try and suppress
	   meaningless messages from f77 programs */
  (void) ieee_flags( "clear", "exception", "all", &out );
  if (flag) {
     if (ieee_handler("set","common",SYsample_handler))
		fprintf(stderr, "Can't set floatingpoint handler\n");
  }
  else {
     if (ieee_handler("clear","common",SYsample_handler))
		fprintf(stderr,"Can't clear floatingpoint handler\n");
  }
  return 0;
}

/* ------------------------ IRIX --------------------------------------*/
#elif defined(PARCH_IRIX)
#include <sigfpe.h>
struct { int code_no; char *name; } error_codes[] = {
       { _INVALID   , "IEEE operand error" } ,
       { _OVERFL    , "floating overflow" } ,
       { _UNDERFL   , "floating underflow" } ,
       { _DIVZERO   , "floating divide" } ,
       { 0      , "unknown error" }
} ;
void SYsample_handler( unsigned exception[],int val[] )
{
    int err_ind, j, code,ierr;
        code = exception[0];
    err_ind = -1 ;
    for ( j = 0 ; error_codes[j].code_no ; j++ )
        if ( error_codes[j].code_no == code ) err_ind = j ;
    if ( err_ind >= 0 )
        fprintf(stderr, "*** %s occurred ***\n",
            error_codes[err_ind].name );
    else
        fprintf(stderr,
            "*** floating point error 0x%x occurred ***\n",
            code);  
    ierr = PetscError(0,0,"Unknown","floating point error",1);
    exit(ierr);
}
int PetscSetFPTrap(int flag)
{
  if (flag) {
    sigfpe_[_EN_OVERFL].abort = 1;
    sigfpe_[_EN_DIVZERO].abort = 1;
    sigfpe_[_EN_INVALID].abort = 1;
    handle_sigfpes(_ON,_EN_OVERFL | _EN_DIVZERO | _EN_INVALID, 
                    SYsample_handler,_ABORT_ON_ERROR,0);
  }
  else {
  handle_sigfpes(_OFF,_EN_OVERFL | _EN_DIVZERO | _EN_INVALID, 
                 0,_ABORT_ON_ERROR,0);
  }
}
/* ------------------------intelnx-------------------------------------*/
#elif defined(PARCH_intelnx)
/* You have to compile YOUR code with -Knoieee to catch divide-by-zero (and 
   perhaps others)
   
   We have some additional code from Intel, but it is stamped confidential
   so I'm not prepared to release it.  ANL people (already covered) can
   find it in ~gropp/tmp/fir.c .
 */

#include <ieeefp.h>
struct { int code_no; char *name; } error_codes[] = {
       { FP_X_OFL    , "floating overflow" } ,
       { FP_X_DZ  , "floating divide" } ,
       { FP_X_INV   , "invalide operand" } ,
       { 0      , "unknown error" }
} ;
#define SIGPC(scp)  0   /* maybe we could figure this out ? */
void SYsample_handler(int sig,int code,struct sigcontext *scp)
{
  int err_ind, j,ierr;
  /*
	   Sample user-written sigfpe code handler.
	   Prints a message and continues.
	   struct sigcontext is defined in <signal.h>.
   */

  err_ind = -1 ;
  for ( j = 0 ; error_codes[j].code_no ; j++ )
    {if ( error_codes[j].code_no == code ) err_ind = j ;}

    if ( err_ind >= 0 )
      fprintf(stderr, "*** %s occurred at pc=%X ***\n",
			error_codes[err_ind].name, SIGPC(scp));
    else
      fprintf(stderr,
              "*** floating point error 0x%x occurred at pc=%X ***\n",
              code, SIGPC(scp));
    ierr = PetscError(0,0,"Unknown","floating point error",1);
    exit(ierr);
}
int PetscSetFPTrap(int on)
{
  int flag;
  if (on) {
    fpsetmask( FP_X_OFL | FP_X_DZ | FP_X_INV );
    flag = (int) 	signal(SIGFPE,SYsample_handler);
    if (flag == -1) fprintf(stderr, "Can't set floatingpoint handler\n");
  }
  else {
    fpsetmask(0);
    flag = (int) 	signal(SIGFPE,SYsample_handler);
    if (flag == -1) fprintf(stderr,"Can't clear floatingpoint handler\n");
  }
  return 0;
}
/*----------------IEEE error handler for IBM RS6000.--------------*/
/* In "fast" mode, floating point traps are imprecise and ignored.
   This is the reason for the fptrap( FP_TRAP_SYNC ) call */
/* Also, this needs to be AIX 3.2 or later */
#elif defined(PARCH_rs6000) 
#include <fpxcp.h>
#include <fptrap.h>
#include <stdlib.h>
#define FPE_FLTOPERR_TRAP (fptrap_t)(0x20000000)
#define FPE_FLTOVF_TRAP   (fptrap_t)(0x10000000)
#define FPE_FLTUND_TRAP   (fptrap_t)(0x08000000)
#define FPE_FLTDIV_TRAP   (fptrap_t)(0x04000000)
#define FPE_FLTINEX_TRAP  (fptrap_t)(0x02000000)

struct { int code_no; char *name; } error_codes[] = {
       {FPE_FLTOPERR_TRAP	, "IEEE operand error" } ,
	   { FPE_FLTOVF_TRAP	, "floating overflow" } ,
	   { FPE_FLTUND_TRAP	, "floating underflow" } ,
	   { FPE_FLTDIV_TRAP	, "floating divide" } ,
	   { FPE_FLTINEX_TRAP	, "inexact floating result" } ,
	   { 0			, "unknown error" } 
} ;
#define SIGPC(scp) (0) /* Info MIGHT be in scp->sc_jmpbuf.jmp_context.iar */
/* 
   For some reason, scp->sc_jmpbuf doesn't work on the RS6000, even though
   it looks like it should from the include definitions.  It is probably
   some strange interaction with the "POSIX_SOURCE" that we require.
 */

void SYsample_handler(int sig,int code,struct sigcontext *scp )
{
  int ierr,err_ind, j;
  fp_ctx_t flt_context;

  fp_sh_trap_info( scp, &flt_context );
    
  /*
	   Sample user-written sigfpe code handler.
	   Prints a message and continues.
	   struct sigcontext is defined in <signal.h>.
   */

  err_ind = -1 ;
  for ( j = 0 ; error_codes[j].code_no ; j++ )
    {if ( error_codes[j].code_no == flt_context.trap ) err_ind = j ;}

    if ( err_ind >= 0 )
      fprintf(stderr, "*** %s occurred ***\n",
			error_codes[err_ind].name );
    else
      fprintf(stderr,
              "*** floating point error 0x%x occurred ***\n",
              flt_context.trap );
    ierr = PetscError(0,0,"Unknown","floating point error",1);
    exit(ierr);
}

int PetscSetFPTrap(int on)
{
  int flag;
  if (on) {
    signal( SIGFPE, (void (*)(int))SYsample_handler );
    fp_trap( FP_TRAP_SYNC );
    fp_enable( TRP_INVALID | TRP_DIV_BY_ZERO | TRP_OVERFLOW );
    /* fp_enable(mask) for individual traps.  Values are:
       TRP_INVALID
       TRP_DIV_BY_ZERO
       TRP_OVERFLOW
       TRP_UNDERFLOW
       TRP_INEXACT
       Can OR then together.
       fp_enable_all(); for all traps.
    */
  }
  else {
    signal(SIGFPE,SIG_DFL);
    fp_disable( TRP_INVALID | TRP_DIV_BY_ZERO | TRP_OVERFLOW );
    fp_trap( FP_TRAP_OFF );
  }
  return 0;
}

/* -------------------------Default -----------------------------------*/
#else 
struct { int code_no; char *name; } error_codes[] = {
	   { 0		, "unknown error\0" } 
} ;
/*ARGSUSED*/
void SYsample_handler(int sig)
{
  int ierr;
  fprintf(stderr, "*** floating point error occurred ***\n" );
  ierr = PetscError(0,0,"Unknown","floating point error",1);
  exit(ierr);
}
int PetscSetFPTrap(int on)
{
  int flag;
  if (on) {
    flag = (int) signal(SIGFPE,SYsample_handler);
    if (flag == -1) fprintf(stderr, "Can't set floatingpoint handler\n");
  }
  else {
    flag = (int) signal(SIGFPE,SIG_DFL);
    if (flag == -1) fprintf(stderr,"Can't clear floatingpoint handler\n");
  }
  return 0;
}
#endif

/***************************************************************************
  This code enables and clears "benign" underflows.  This is for code that
  does NOT care how underflows are handled, just that they be handled "fast".
  Note that for general-purpose code, this can be very, very dangerous.
  But for many PDE calculations, it may even be provable that underflows
  are benign.

  Not all machines need worry or care about this, but for those that do,
  we provide routines to change to/from a benign mode.
 ***************************************************************************/
#if defined(PARCH_intelnx)
int PetscSetBenignUnderflows()
{
  /* This needs the following assembly-language program:
  .globl     _set_fsr
  .globl     _set_fsr_

  _set_fsr:
  _set_fsr_:
      st.c   r16,fsr
      bri    r1
      nop
 */

  /*         set_fsr( 0x21 );  */
  return 0;
}
#elif defined(PARCH_rs6000)
int PetscSetBenignUnderflows()
{
  /* abrupt_underflow seems to have disappeared! */
  /* abrupt_underflow(); */
  return 0;
}
#else
int PetscSetBenignUnderflows()
{
  return 0;
}
#endif



