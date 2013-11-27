#include <sys/time.h>

/* 
This function is used to determine user time in seconds.
Timing of a code segment requires calls to determine the 
initial and final times.

This code courtesy of Satish Balay.
*/

#ifndef rs6000
#define wallclock wallclock_
#endif

void wallclock(double * time) {
  static struct timeval _tp;
  gettimeofday(&_tp,(struct timezone *)0);
  (*time)=((double)_tp.tv_sec)+(1.0e-6)*(_tp.tv_usec);
}
