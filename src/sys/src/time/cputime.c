#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: cputime.c,v 1.24 1998/12/17 21:56:44 balay Exp bsmith $";
#endif

/*
  This is to allow one to measure CPU time usage of their job, 
  NOT real time usage.
*/

#include "petsc.h"                     /*I "petsc.h" I*/
#include "sys.h"
#include "pinclude/ptime.h"
#if defined(HAVE_PWD_H)
#include <pwd.h>
#endif
#include <ctype.h>
#include <sys/types.h>
#include <sys/stat.h>
#if defined(HAVE_UNISTD_H)
#include <unistd.h>
#endif
#if defined(HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#if !defined(PARCH_win32)
#include <sys/utsname.h>
#endif
#if defined(PARCH_win32)
#include <windows.h>
#include <io.h>
#include <direct.h>
#endif
#if defined (PARCH_win32_gnu)
#include <windows.h>
#endif
#include <fcntl.h>
#include <time.h>  
#if defined(HAVE_SYS_SYSTEMINFO_H)
#include <sys/systeminfo.h>
#endif
#include "pinclude/petscfix.h"

#undef __FUNC__
#define __FUNC__ "PetscGetCPUTime"

#if defined (PARCH_solaris)

#include <sys/times.h>
#include <limits.h>
int PetscGetCPUTime(PLogDouble *t)
{
  struct tms temp;

  PetscFunctionBegin;
  times(&temp);
  *t = ((double) temp.tms_utime)/((double) CLK_TCK);
  PetscFunctionReturn(0);
}

#elif defined(HAVE_CLOCK)

#include <time.h>
#include <sys/types.h>

int PetscGetCPUTime(PLogDouble *t)
{
  PetscFunctionBegin;
  *t = ((double)clock()) / ((double)CLOCKS_PER_SEC);
  PetscFunctionReturn(0);
}  

#else

#include <sys/types.h>
#include <sys/time.h>
#include <sys/resource.h>

/*@
    PetscGetCPUTime - Returns the CPU time in seconds used by the process.

    Not Collective

    Output Parameter:
.   t - Time in seconds charged to the process.

    Example:
.vb
    #include "petsc.h"
    ...
    PLogDouble t1, t2;
 
    ierr = PetscGetCPUTime(&t1); CHKERRA(ierr);
    ... code to time ...
    ierr = PetscGetCPUTime(&t2); CHKERRA(ierr);
    printf( "Code took %f CPU seconds\n", t2-t1);
.ve

    Level: intermediate

    Notes:
    One should use PetscGetTime() or the -log_summary option of 
    PETSc for profiling. The CPU time is NOT a realistic number to
    use since it does not include the time for message passing etc.
    Also on many systems the accuracy is only on the order of microseconds.
@*/
int PetscGetCPUTime(PLogDouble *t)
{
  static struct rusage temp;
  double foo, foo1;

  PetscFunctionBegin;
  getrusage(RUSAGE_SELF,&temp);
  foo     = temp.ru_utime.tv_sec;     /* seconds */
  foo1    = temp.ru_utime.tv_usec;    /* uSecs */
  *t      = foo + foo1 * 1.0e-6;
  PetscFunctionReturn(0);
}

#endif
