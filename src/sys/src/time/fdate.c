#ifndef lint
static char vcid[] = "$Id: fdate.c,v 1.4 1996/08/08 14:41:26 bsmith Exp bsmith $";
#endif

#include "src/sys/src/files.h"
/*
     Some versions of the Gnu g++ compiler require a prototype for gettimeofday()
  on the IBM rs6000.

extern "C" {
extern int gettimeofday(struct timeval *, struct timezone *);
}
*/
   
char *PetscGetDate()
{
  struct timeval tp;
  gettimeofday( &tp, (struct timezone *)0 );
  return asctime(localtime((time_t *) &tp.tv_sec));
}
