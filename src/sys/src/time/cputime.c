#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: cputime.c,v 1.8 1997/07/23 02:17:38 bsmith Exp bsmith $";
#endif

/*
  This is to allow one to measure CPU time usage of their job, 
  NOT real time usage.
*/

#include "petsc.h"                     /*I "petsc.h" I*/

#if defined(PARCH_t3d)

PLogDouble PetscGetCPUTime()
{
  fprintf(stderr,"CPUTime unavailable on Cray T3D/E\n");
  fprintf(stderr,"PetscGetCPUTime() returning 0\n");
  return 0.0;
}

#else

#include "src/sys/src/files.h"
#if defined(PARCH_hpux)
#include <time.h>
#elif defined(PARCH_solaris)
#include <sys/times.h>
#include <limits.h>
#else
#include <sys/types.h>
#if !defined(PARCH_nt)
#include <sys/time.h>
#include <sys/resource.h>
#endif
#if defined(__cplusplus)
extern "C" {
#endif
extern int getrusage(int,struct rusage*);
#if defined(__cplusplus)
}
#endif
#endif

/*@C
    PetscGetCPUTime - Returns the CPU time in seconds used by the process.
         One should use PetscGetTime() or the -log_summary option of 
         PETSc for profiling. The CPU time is not a realistic number to
         use since it does not include the time for message passing etc.
         Also on many systems the accuracy is only on the order of 
         microseconds.

    Returns:
    Time in seconds charged to the process.

    Example:
$   #include "system/system.h"
$   ...
$   double t1, t2;
$
$   t1 = PetscGetCPUTime();
$   ... code to time ...
$   t2 = PetscGetCPUTime() - t1;
$   printf( "Code took %f CPU seconds\n", t2 );
$
@*/
PLogDouble PetscGetCPUTime()
{
#if defined(PARCH_solaris)
  struct tms temp;
  times(&temp);
  return  ((double) temp.tms_utime)/((double) CLK_TCK);
#elif defined(PARCH_hpux) || defined (PARCH_nt)
return  ((double)clock()) / ((double)CLOCKS_PER_SEC);
#else
  static struct rusage temp;
  double foo, foo1;

  getrusage(RUSAGE_SELF,&temp);
  foo     = temp.ru_utime.tv_sec;     /* seconds */
  foo1    = temp.ru_utime.tv_usec;    /* uSecs */
  return(foo + foo1 * 1.0e-6);
#endif
}

#endif




