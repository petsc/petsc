#ifndef lint
static char vcid[] = "$Id: fdate.c,v 1.8 1997/02/22 02:23:29 bsmith Exp balay $";
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
