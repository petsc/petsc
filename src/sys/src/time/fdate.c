/*$Id: fdate.c,v 1.31 1999/09/21 15:10:23 bsmith Exp bsmith $*/

#include "petsc.h"
#include "sys.h"
#include "pinclude/ptime.h"
#include <ctype.h>
#include <sys/types.h>
#include <sys/stat.h>
#if defined(PETSC_HAVE_UNISTD_H)
#include <unistd.h>
#endif
#if defined(PETSC_HAVE_STDLIB_H)
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
#if defined(PETSC_HAVE_SYS_SYSTEMINFO_H)
#include <sys/systeminfo.h>
#endif
#ifndef MAXPATHLEN
#define MAXPATHLEN 1024
#endif

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
  int    ierr;
#else
  struct timeval tp;
  int            ierr;
#endif

  PetscFunctionBegin;
#if defined (PARCH_win32)
  time( &aclock);
  ierr = PetscStrncpy(name,asctime(localtime(&aclock)),len);CHKERRQ(ierr);
#else
  gettimeofday( &tp, (struct timezone *)0 );
  ierr = PetscStrncpy(name,asctime(localtime((time_t *) &tp.tv_sec)),len);CHKERRQ(ierr);
#endif
  /* now strip out the new-line chars at the end of the string */
  ierr = PetscStrstr(name,"\n",&str);CHKERRQ(ierr);
  if (str) str[0] = 0;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscSetInitialDate"
int PetscSetInitialDate(void)
{
  int ierr;
  PetscFunctionBegin;
  ierr = PetscGetDate(starttime,64);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "PetscGetInitialDate"
int PetscGetInitialDate(char name[],int len)
{
  int ierr;

  PetscFunctionBegin;
  ierr = PetscStrncpy(name,starttime,len);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
