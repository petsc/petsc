#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: fdate.c,v 1.18 1997/10/19 03:23:45 bsmith Exp bsmith $";
#endif

#include "src/sys/src/files.h"
#if defined (NEEDS_GETTIMEOFDAY_PROTO)
#include <sys/resource.h>
#if defined(__cplusplus)
extern "C" {
extern int gettimeofday(struct timeval *, struct timezone *);
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
