#ifndef lint
static char vcid[] = "$Id: cputime.c,v 1.2 1997/04/01 20:33:29 bsmith Exp bsmith $";
#endif

/*
              This file is not currently used. It is to allow one
     to measure CPU time usage of their job, NOT real time usage.
*/

#include "petsc.h"

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
#include <sys/time.h>
#include <sys/resource.h>
#if defined(__cplusplus)
extern "C" {
#endif
extern int getrusage(int,struct rusage*);
#if defined(__cplusplus)
}
#endif
#endif

/*@
    PetscGetCPUTime - Returns the time in seconds used by the process.

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
#elif defined(PARCH_hpux)
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




