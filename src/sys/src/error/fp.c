/*
*	IEEE error handler for all machines. Since each machine has 
*   enough slight differences we have completely separate codes for each one.
*
*/
#include <signal.h>
#include <stdio.h>
#include "tools.h"
#include "system/system.h"

/*----------------IEEE error handler for Sun SparcStations.--------------*/
#if defined(sun4) 
#include <floatingpoint.h>
#if defined(SOLARIS) || defined(solaris)

#ifdef MEIKO_SOLARIS
#include <ieeefp.h>
struct { int code_no; char *name; } error_codes[] = {
           { FP_X_INV   , "IEEE invalid operation error" } ,
           { FP_X_OFL   , "floating overflow" } ,
           { FP_X_UFL   , "floating underflow" } ,
           { FP_X_DZ    , "floating divide" } ,
           { FP_X_IMP   , "inexact floating result" } ,
           { 0                  , "unknown error" }
} ;

#define SIGPC(scp) (0)
#else 
#include <siginfo.h>
#include <ucontext.h>
struct { int code_no; char *name; } error_codes[] = {
       { 0 /* ??? */ , "integer divide" } ,
       { fp_invalid, "IEEE operand error" } ,
       { fp_overflow, "floating overflow" } ,
       { fp_underflow, "floating underflow" } ,
       { fp_division, "floating divide" } ,
       { fp_inexact, "inexact floating result" } ,
       { 0          , "unknown error" }
};
sigfpe_handler_type SYsample_handler( sig, sip, uap)
int        sig;               /* sig == SIGFPE always */
siginfo_t  *sip;
ucontext_t *uap;
{
  int err_ind, j;
  /*
       Sample user-written sigfpe code handler.
       Prints a message and continues.
       struct sigcontext is defined in <signal.h>.
   */

    SYExit("floating point error",err_ind);
}
#endif
#else
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

sigfpe_handler_type SYsample_handler( sig, code, scp, addr)
int sig ;               /* sig == SIGFPE always */
int code ;
struct sigcontext *scp ;
char *addr ;
{
  int err_ind, j;
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
    SYExit("floating point error",err_ind);
}
#endif

/*@
   SYSetFPTraps - Enable traps/exceptions on common floating point errors

   Description:
   This routine, on systems that support it, causes floating point
   overflow, divide-by-zero, and invalid-operand (e.g., a NaN) to
   cause a message to be printed and the program to exit.

   Use SYClearFPTraps() to prevent catching FP exceptions.

   Notes on specific implementations:
.   IBMRs6000  - Using this may slow down fp by a factor of 10.
.   Sun4       - Clears any pre-existing exceptions.
@*/
void SYSetFPTraps()
{
char *out; 
        /* Clear accumulated exceptions.  Used to try and suppress
	   meaningless messages from f77 programs */
        (void) ieee_flags( "clear", "exception", "all", &out );
	if (ieee_handler("set","common",SYsample_handler))
		fprintf(stderr, "Can't set floatingpoint handler\n");
}

/*@
   SYClearFPTraps - Disable traps/exceptions on common floating point errors

   Description:
   This routine, on systems that support it, causes floating point
   overflow, divide-by-zero, and invalid-operand (e.g., a NaN) to
   behave with the default IEEE semantics --- no exception or message is
   generated, and computation continues.

   Use SYSetFPTraps() to catch FP exceptions.
@*/
void SYClearFPTraps()
{
	if (ieee_handler("clear","common",SYsample_handler))
		fprintf(stderr,"Can't clear floatingpoint handler\n");
}
/* ------------------------Alliant fx2800------------------------------*/
#elif defined(fx2800)
/* There seems to be no include file with this info */
#define FP_OPERR   0x2000  /*operand error*/
#define FP_OVFL    0x1000  /*overflow*/
#define FP_UNFL    0x0800  /*underflow*/
#define FP_DZ      0x0400  /*divide by zero*/
#define FP_INEX2   0x0200  /*inexact result*/
#include <machine/sr.h>

struct { int code_no; char *name; } error_codes[] = {
       { FP_OPERR   , "IEEE operand error" } ,
       { FP_OVFL    , "floating overflow" } ,
       { FP_UNFL    , "floating underflow" } ,
       { FP_DZ  , "floating divide" } ,
       { FP_INEX2   , "inexact floating result" } ,
       { 0      , "unknown error" }
} ;
#define SIGPC(scp) (0)
void SYsample_handler( sig, code, scp, addr)
int sig ;               /* sig == SIGFPE always */
int code ;
struct sigcontext *scp ;
char *addr ;
{
  int err_ind, j;
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
    SYExit("floating point error",err_ind);
}
void SYSetFPTraps()
{
  int flag;
  /* set_fcontrol( FP_OVFL | FP_DZ | FP_OPERR ); */
  i860_access_fsr( FSR_FTE, 0 );
  flag = (int) 	signal(SIGFPE,SYsample_handler);
  if (flag == -1) fprintf(stderr, "Can't set floatingpoint handler\n");
}
void SYClearFPTraps()
{
  int flag;
  /* set_fcontrol(0); */
  i860_access_fsr( 0, FSR_FTE );
  flag = (int) 	signal(SIGFPE,SYsample_handler);
  if (flag == -1) fprintf(stderr,"Can't clear floatingpoint handler\n");
}
/* ------------------------ IRIX --------------------------------------*/
#elif defined(IRIX)
#include <sigfpe.h>
struct { int code_no; char *name; } error_codes[] = {
       { _INVALID   , "IEEE operand error" } ,
       { _OVERFL    , "floating overflow" } ,
       { _UNDERFL   , "floating underflow" } ,
       { _DIVZERO   , "floating divide" } ,
       { 0      , "unknown error" }
} ;
void SYsample_handler( exception, val )
unsigned exception[5];
int      val[2];
{
    int err_ind, j, code;
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
  SYExit("floating point error",0);
}
void SYClearFPTraps()
{
  handle_sigfpes(_OFF,_EN_OVERFL | _EN_DIVZERO | _EN_INVALID, 
                 0,_ABORT_ON_ERROR,0);
}
void SYSetFPTraps()
{
  SYClearFPTraps();   /* clear anything that was set */
  sigfpe_[_EN_OVERFL].abort = 1;
  sigfpe_[_EN_DIVZERO].abort = 1;
  sigfpe_[_EN_INVALID].abort = 1;
  handle_sigfpes(_ON,_EN_OVERFL | _EN_DIVZERO | _EN_INVALID, 
                  SYsample_handler,_ABORT_ON_ERROR,0);
}
/* ------------------------tc2000 -------------------------------------*/
#elif defined(tc2000)
#include <ieeefp.h>
struct { int code_no; char *name; } error_codes[] = {
       { FP_OPERR   , "IEEE operand error" } ,
       { FP_OVFL    , "floating overflow" } ,
       { FP_UNFL    , "floating underflow" } ,
       { FP_DZ  , "floating divide" } ,
       { FP_INEX2   , "inexact floating result" } ,
       { 0      , "unknown error" }
} ;
#define SIGPC(scp) (scp->sc_pc)
int SYsample_handler( sig, code, scp)
int sig ;               /* sig == SIGFPE always */
int code ;
struct sigcontext *scp ;
{
  int err_ind, j;
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
    SYExit("floating point error",err_ind);
}
void SYSetFPTraps()
{
  int flag;
  fpsetmask( FP_OVFL | FP_DZ | FP_OPERR );
  flag = (int) 	signal(SIGFPE,SYsample_handler);
  if (flag == -1) fprintf(stderr, "Can't set floatingpoint handler\n");
}
void SYClearFPTraps()
{
  int flag;
  fpsetmask(0);
  flag = (int) 	signal(SIGFPE,SYsample_handler);
  if (flag == -1) fprintf(stderr,"Can't clear floatingpoint handler\n");
}

/* ------------------------intelnx-------------------------------------*/
#elif defined(intelnx)
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
void SYsample_handler( sig, code, scp)
int sig ;               /* sig == SIGFPE always */
int code ;
struct sigcontext *scp ;
{
  int err_ind, j;
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
    SYExit("floating point error",err_ind);
}
void SYSetFPTraps()
{
  int flag;
  fpsetmask( FP_X_OFL | FP_X_DZ | FP_X_INV );
  flag = (int) 	signal(SIGFPE,SYsample_handler);
  if (flag == -1) fprintf(stderr, "Can't set floatingpoint handler\n");
}
void SYClearFPTraps()
{
  int flag;
  fpsetmask(0);
  flag = (int) 	signal(SIGFPE,SYsample_handler);
  if (flag == -1) fprintf(stderr,"Can't clear floatingpoint handler\n");
}

/* --------------------Intel/NX (Hypercubes and Delta)----------------------*/
#elif defined(intelnxFOO)
/*
   Enableing the FTE bit isn't enough; we also need the trap handeler to
   call our signal routine.  Sigh...
 */
/* We can get the status (sort of) from the fsr, using bits 9 (mult underflow),
  10 (mult overflow), 11 (mult inexact), 13 (add underflow), 14 (add overflow)

  Also, by setting bit FZ (bit 1), underflows are replaced with zero. 

  Finally, all of these things are rather dangerous, since the i860 does
  not have very fine-grained control.
 */
#define FSR_FTE 0x20

struct { int code_no; char *name; } error_codes[] = {
       { 0      , "unknown error" }
} ;
#define SIGPC(scp) (0)
void SYsample_handler( sig, code, scp)
int sig ;               /* sig == SIGFPE always */
int code ;
struct sigcontext *scp ;
{
  int err_ind, j;
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
    SYExit("floating point error",err_ind);
}
void SYSetFPTraps()
{
  int flag;
  SYNX860( FSR_FTE, 1 );
  flag = (int) 	signal(SIGFPE,SYsample_handler);
  if (flag == -1) fprintf(stderr, "Can't set floatingpoint handler\n");
}
void SYClearFPTraps()
{
  int flag;
  SYNX860( FSR_FTE, 0 );
  flag = (int) 	signal(SIGFPE,SYsample_handler);
  if (flag == -1) fprintf(stderr,"Can't clear floatingpoint handler\n");
}

/*----------------IEEE error handler for IBM RS6000.--------------*/
/* In "fast" mode, floating point traps are imprecise and ignored.
   This is the reason for the fptrap( FP_TRAP_SYNC ) call */
/* Also, this needs to be AIX 3.2 or later */
#elif defined(rs6000) 
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

void SYsample_handler( sig, code, scp )
int               sig;                /* sig == SIGFPE always */
int               code;
struct sigcontext *scp ;
{
  int err_ind, j;
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
    SYExit("floating point error",err_ind);
}

void SYSetFPTraps()
{
int flag;

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

void SYClearFPTraps()
{
int flag;

signal(SIGFPE,SIG_DFL);
fp_disable( TRP_INVALID | TRP_DIV_BY_ZERO | TRP_OVERFLOW );
fp_trap( FP_TRAP_OFF );
/* fp_disable_all(); */
/* fp_disable(mask) for individual traps */
}

/* -------------------------Default -----------------------------------*/
#else 
struct { int code_no; char *name; } error_codes[] = {
	   { 0		, "unknown error\0" } 
} ;
/*ARGSUSED*/
void SYsample_handler( sig)
int sig;
{
    fprintf(stderr, "*** floating point error occurred ***\n" );
    SYExit("floating point error",0);    
}
void SYSetFPTraps()
{
  int flag;
  flag = (int) signal(SIGFPE,SYsample_handler);
  if (flag == -1) fprintf(stderr, "Can't set floatingpoint handler\n");
}
void SYClearFPTraps()
{
  int flag;
  flag = (int) signal(SIGFPE,SIG_DFL);
  if (flag == -1) fprintf(stderr,"Can't clear floatingpoint handler\n");
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
#if defined(intelnx)
void SYSetBenignUnderflows()
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
}
void SYClrBenignUnderflows()
{
/* Not sure how to disable it */
}

#elif defined(rs6000)
void SYSetBenignUnderflows()
{
/* abrupt_underflow seems to have disappeared! */
/* abrupt_underflow(); */
}
void SYClrBenignUnderflows()
{
}

#else
/* This is the default do-nothing code */

void SYSetBenignUnderflows()
{
}
void SYClrBenignUnderflows()
{
}
#endif
