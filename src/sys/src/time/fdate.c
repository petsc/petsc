#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: fdate.c,v 1.23 1998/07/15 15:16:09 balay Exp balay $";
#endif

#include "petsc.h"
#include "sys.h"
#include "pinclude/ptime.h"
#if defined(HAVE_PWD_H)
#include <pwd.h>
#endif
#include <ctype.h>
#include <sys/types.h>
#include <sys/stat.h>
#if defined(HAVE_UNISTD_H)
#include <unistd.h>
#endif
#if defined(HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#if !defined(PARCH_win32)
#include <sys/utsname.h>
#endif
#if defined(PARCH_win32)
#include <windows.h>
#include <io.h>
#include <direct.h>
#endif
#if defined (PARCH_win32_gnu)
#include <windows.h>
#endif
#include <fcntl.h>
#include <time.h>  
#if defined(HAVE_SYS_SYSTEMINFO_H)
#include <sys/systeminfo.h>
#endif
#include "pinclude/petscfix.h"

#ifndef MAXPATHLEN
#define MAXPATHLEN 1024
#endif

#if defined (NEEDS_GETTIMEOFDAY_PROTO)
#include <sys/resource.h>
#if defined(__cplusplus)
extern "C" {
#if (defined (PARCH_IRIX64) ||  defined (PARCH_IRIX) || defined (PARCH_IRIX5))
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
#if defined (PARCH_win32)
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
