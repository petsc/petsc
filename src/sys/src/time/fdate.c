#ifndef lint
static char vcid[] = "$Id: fdate.c,v 1.9 1997/02/27 00:39:01 balay Exp balay $";
#endif

#include "src/sys/src/files.h"
#if defined (PARCH_IRIX64)
#include <sys/resource.h>
#endif
/*
     Some versions of the Gnu g++ compiler require a prototype for gettimeofday()
  on the IBM rs6000. Also CC on some IRIX64 machines

extern "C" {
extern int gettimeofday(struct timeval *, struct timezone *);
}
*/
   
#undef __FUNC__  
#define __FUNC__ "PetscGetDate" /* ADIC Ignore */
char *PetscGetDate()
{
#if defined (PARCH_nt)
  time_t aclock;
  time( &aclock);
  return asctime(localtime(&aclock));
#else
  struct timeval tp;
  gettimeofday( &tp, (struct timezone *)0 );
  return asctime(localtime((time_t *) &tp.tv_sec));
#endif
}
