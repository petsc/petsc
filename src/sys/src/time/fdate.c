#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: fdate.c,v 1.25 1999/03/09 21:17:20 balay Exp balay $";
#endif

#include "petsc.h"
#include "sys.h"
#include "pinclude/ptime.h"
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

static char starttime[64];
   
/*
  This function is called once during the initialize stage.
  It stashes the timestamp, and uses it when needed.
*/
#undef __FUNC__  
#define __FUNC__ "PetscGetDate"
int PetscGetDate(char name[],int len)
{
  char *str=0;
#if defined (PARCH_win32)
  time_t aclock;

  PetscFunctionBegin;
  time( &aclock);
  PetscStrncpy(name,asctime(localtime(&aclock)),len);
#else
  struct timeval tp;

  PetscFunctionBegin;
  gettimeofday( &tp, (struct timezone *)0 );
  PetscStrncpy(name,asctime(localtime((time_t *) &tp.tv_sec)),len);
#endif
  /* now strip out the new-line chars at the end of the string */
  str = PetscStrstr(name,"\n");
  if (str) str[0] = 0;
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "PetscGetInitialDate"
int PetscSetInitialDate(void)
{
  PetscFunctionBegin;
  PetscGetDate(starttime,64);
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "PetscGetInitialDate"
int PetscGetInitialDate(char name[],int len)
{
  PetscFunctionBegin;
  PetscStrncpy(name,starttime,len);
  PetscFunctionReturn(0);
}
