#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: fdate.c,v 1.19 1998/03/23 21:18:59 bsmith Exp balay $";
#endif

#include "src/sys/src/files.h"
#if defined (NEEDS_GETTIMEOFDAY_PROTO)
#include <sys/resource.h>
#if defined(__cplusplus)
extern "C" {
#if (defined (PARCH_IRIX64) ||  defined (PARCH_IRIX) || defined PARCH_IRIX5)
extern int gettimeofday(struct timeval *,...);
#else
extern int gettimeofday(struct timeval *, struct timezone *);
#endif
}
#endif
#endif
/*
     Some versions of the Gnu g++ compiler require a prototype for gettimeofday()
  on the IBM rs6000. Also CC on some IRIX64 machines

#if defined(__cplusplus)
extern "C" {
extern int gettimeofday(struct timeval *, struct timezone *);
}
#endif

*/
   
#undef __FUNC__  
#define __FUNC__ "PetscGetDate"
char *PetscGetDate(void)
{
#if defined (PARCH_nt)
  time_t aclock;

  PetscFunctionBegin;
  time( &aclock);
  PetscFunctionReturn(asctime(localtime(&aclock)));
#else
  struct timeval tp;

  PetscFunctionBegin;
  gettimeofday( &tp, (struct timezone *)0 );
  PetscFunctionReturn(asctime(localtime((time_t *) &tp.tv_sec)));
#endif
}
