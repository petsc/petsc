#ifndef lint
static char vcid[] = "$Id: fdate.c,v 1.7 1997/01/06 20:22:55 balay Exp bsmith $";
#endif

#include "src/sys/src/files.h"
/*
     Some versions of the Gnu g++ compiler require a prototype for gettimeofday()
  on the IBM rs6000.

extern "C" {
extern int gettimeofday(struct timeval *, struct timezone *);
}
*/
   
#undef __FUNC__  
#define __FUNC__ "PetscGetDate" /* ADIC Ignore */
char *PetscGetDate()
{
  struct timeval tp;
  gettimeofday( &tp, (struct timezone *)0 );
  return asctime(localtime((time_t *) &tp.tv_sec));
}
