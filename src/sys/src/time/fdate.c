#ifndef lint
static char vcid[] = "$Id: fdate.c,v 1.5 1996/10/24 19:26:02 bsmith Exp balay $";
#endif

#include "src/sys/src/files.h"
/*
     Some versions of the Gnu g++ compiler require a prototype for gettimeofday()
  on the IBM rs6000.

extern "C" {
extern int gettimeofday(struct timeval *, struct timezone *);
}
*/
   
#undef __FUNCTION__  
#define __FUNCTION__ "PetscGetDate"
char *PetscGetDate()
{
  struct timeval tp;
  gettimeofday( &tp, (struct timezone *)0 );
  return asctime(localtime((time_t *) &tp.tv_sec));
}
