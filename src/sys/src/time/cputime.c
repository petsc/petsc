#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: cputime.c,v 1.14 1998/03/23 21:18:59 bsmith Exp balay $";
#endif

/*
  This is to allow one to measure CPU time usage of their job, 
  NOT real time usage.
*/

#include "petsc.h"                     /*I "petsc.h" I*/

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

#elif defined(PARCH_hpux)  

#include "src/sys/src/files.h"
#include <time.h>
#include <sys/types.h>
int PetscGetCPUTime(PLogDouble *t)
{
  PetscFunctionBegin;
  *t = ((double)clock()) / ((double)CLOCKS_PER_SEC);
  PetscFunctionReturn(0);
}  

#elif defined(PARCH_t3d) || defined (PARCH_nt)  

#include "src/sys/src/files.h"
#include <sys/types.h>
int PetscGetCPUTime(PLogDouble *t)
{
  PetscFunctionBegin;
  *t = ((double)clock()) / ((double)CLOCKS_PER_SEC);
  PetscFunctionReturn(0);
}  

#else

#include "src/sys/src/files.h"
#include <sys/types.h>
#include <sys/time.h>
#include <sys/resource.h>
#if defined(__cplusplus)
extern "C" {
#endif
/*
   On later versions of Linux you should remove the next line:
 if you the error message 
   cputime.c:13: declaration of C function 'int getrusage(int, struct
             rusage*)' conflicts with
  /usr/include/sys/resource.h:45: previous declaration 'int
  getrusage(enum_rusage_who, struct rusage*)' here

*/
extern int getrusage(int,struct rusage*);
#if defined(__cplusplus)
}
#endif

/*@
    PetscGetCPUTime - Returns the CPU time in seconds used by the process.
         One should use PetscGetTime() or the -log_summary option of 
         PETSc for profiling. The CPU time is not a realistic number to
         use since it does not include the time for message passing etc.
         Also on many systems the accuracy is only on the order of 
         microseconds.

    Returns:
    Time in seconds charged to the process.

    Example:
$   #include "petsc.h"
$   ...
$   PLogDouble t1, t2;
$
$   ierr = PetscGetCPUTime(&t1); CHKERRA(ierr);
$   ... code to time ...
$   ierr = PetscGetCPUTime(&t2); CHKERRA(ierr);
$   printf( "Code took %f CPU seconds\n", t2-t1);
$
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
