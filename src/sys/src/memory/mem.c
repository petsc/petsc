#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: mem.c,v 1.16 1997/07/09 20:51:14 balay Exp bsmith $";
#endif

#include "petsc.h"           /*I "petsc.h" I*/
#include "src/sys/src/files.h"

#if !defined (PARCH_t3d) && !defined(PARCH_nt)
#include <sys/resource.h>
#if defined(__cplusplus)
extern "C" {
#endif
extern int getrusage(int,struct rusage*);
#if !defined(PARCH_linux) && !defined(PARCH_nt_gnu)
extern int getpagesize();
#endif
#if defined(__cplusplus)
}
#endif
#endif
#if defined(PARCH_solaris)
#include <sys/procfs.h>
#include <fcntl.h>
#endif

#undef __FUNC__  
#define __FUNC__ "PetscGetResidentSetSize"
/*@
   PetscGetResidentSetSize - Returns the maximum resident set size (memory used)
   for the program.

   Output Parameter:
   mem - memory usage in bytes

   Options Database Key:
.    -trmalloc_log

   Notes:
   The memory usage reported here includes all Fortran arrays 
   (that may be used in application-defined sections of code).
   This routine thus provides a more complete picture of memory
   usage than PetscTrSpace() for codes that employ Fortran with
   hardwired arrays.

.seealso: PetscTrSpace()

.keywords: get, resident, set, size
@*/
int PetscGetResidentSetSize(PLogDouble *foo)
{
#if !defined(PARCH_solaris) && !defined(PARCH_hpux) && !defined(PARCH_t3d) \
  && !defined (PARCH_nt) && !defined (PARCH_nt_gnu)
  static struct rusage temp;
  getrusage(RUSAGE_SELF,&temp);
#if defined(PARCH_rs6000)
  /* RS6000 always reports sizes in k instead of pages */
  *foo = 1024.0 * ((double) temp.ru_maxrss);
#else
  *foo = ( (double) getpagesize())*( (double) temp.ru_maxrss );
#endif
#elif defined(PARCH_solaris)
  int             fd;
  char            proc[1024];
  prpsinfo_t      prusage;
  sprintf(proc,"/proc/%d", getpid());
  if ((fd = open(proc,O_RDONLY)) == -1) SETERRQ(1,1,"Unable to access system files");
  if (ioctl(fd, PIOCPSINFO,&prusage) == -1) SETERRQ(1,1,"Unable to access system files"); 
  *foo = (double) prusage.pr_byrssize;
  close(fd);
#else
  *foo = 0.0;
#endif
  return 0;
}
