#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: mem.c,v 1.28 1998/04/26 03:14:59 bsmith Exp balay $";
#endif

#include "petsc.h"           /*I "petsc.h" I*/
#include "src/sys/src/files.h"

#if !defined (PARCH_t3d) && !defined(PARCH_nt)
#include <sys/resource.h>
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

   Not Collective

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
#if defined(PARCH_solaris)
  int             fd;
  char            proc[1024];
  prpsinfo_t      prusage;

  PetscFunctionBegin;
  sprintf(proc,"/proc/%d", (int)getpid());
  if ((fd = open(proc,O_RDONLY)) == -1) {
    SETERRQ(PETSC_ERR_FILE_OPEN,1,"Unable to access system file to get memory usage data");
  }
  if (ioctl(fd, PIOCPSINFO,&prusage) == -1) {
    SETERRQ(PETSC_ERR_FILE_READ,1,"Unable to access system file  to get memory usage data"); 
  }
  *foo = (double) prusage.pr_byrssize;
  close(fd);
#elif defined(PARCH_hpux) || defined(PARCH_t3d) || defined(PARCH_nt)
  PetscFunctionBegin;
  *foo = 0.0;
#else
  static struct rusage temp;

  PetscFunctionBegin;
  getrusage(RUSAGE_SELF,&temp);
#if defined(PARCH_rs6000) || defined(PARCH_IRIX) || defined(PARCH_IRIX64) || defined(PARCH_IRIX5)
  *foo = 1024.0 * ((double) temp.ru_maxrss);
#else
  *foo = ( (double) getpagesize())*( (double) temp.ru_maxrss );
#endif
#endif
  PetscFunctionReturn(0);
}
