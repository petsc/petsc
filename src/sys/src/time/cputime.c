#ifndef lint
static char vcid[] = "$Id: cputime.c,v 1.1 1994/03/18 00:22:04 gropp Exp $";
#endif

#include "tools.h"
#include <stdio.h>
#if defined(cray) || defined(__MSDOS__) || defined(HPUX)
#include <time.h>
#else
#if defined(solaris)
#include <sys/times.h>
#include <limits.h>
#else
#include <sys/types.h>
#include <sys/time.h>
#include <sys/resource.h>
#endif
#ifdef SOLARIS
#include <sys/rusage.h>
#endif
#endif

/*@
    SYGetCPUTime - Returns the time in seconds used by the process.

    Returns:
    Time in seconds charged to the process.

    Example:
$   #include "system/system.h"
$   ...
$   double t1, t2;
$
$   t1 = SYGetCPUTime();
$   ... code to time ...
$   t2 = SYGetCPUTime() - t1;
$   printf( "Code took %f CPU seconds\n", t2 );
$
@*/
double SYGetCPUTime()
{
#if defined(titan)
  return(1.0e-6*((double) clock()));

#elif defined(intelnx)
  double dclock();
  return dclock();

#elif defined(cm5)
static int not_ready = 1;
extern double CMMD_node_timer_busy();
double val;
#define CM5TOOLSTIMER1 1
if (not_ready) {
    CMMD_node_timer_clear( CM5TOOLSTIMER1 );
    CMMD_node_timer_start( CM5TOOLSTIMER1 );
    not_ready = 0;
    }

CMMD_node_timer_stop( CM5TOOLSTIMER1 );
val = CMMD_node_timer_busy( CM5TOOLSTIMER1 );
CMMD_node_timer_start( CM5TOOLSTIMER1 );
return val;

#elif defined(solaris)
  struct tms temp;
  times(&temp);
  return  ((double) temp.tms_utime)/((double) CLK_TCK);

#elif defined(cray)
/* PROBLEM - this is user + system on behalf of user.  times(2) may be used
   to get the individual elements of the time */
return ((double)clock()) / ((double)CLOCKS_PER_SEC);

#elif defined(__MSDOS__) || defined(HPUX)
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





