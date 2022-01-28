#define PETSC_DESIRE_FEATURE_TEST_MACROS /* for getpagesize() with c89 */
#include <petscsys.h>           /*I "petscsys.h" I*/
#if defined(PETSC_HAVE_PWD_H)
#include <pwd.h>
#endif
#include <ctype.h>
#include <sys/stat.h>
#if defined(PETSC_HAVE_UNISTD_H)
#include <unistd.h>
#endif
#if defined(PETSC_HAVE_SYS_UTSNAME_H)
#include <sys/utsname.h>
#endif
#include <fcntl.h>
#include <time.h>
#if defined(PETSC_HAVE_SYS_SYSTEMINFO_H)
#include <sys/systeminfo.h>
#endif

#if defined(PETSC_HAVE_SYS_RESOURCE_H)
#include <sys/resource.h>
#endif
#if defined(PETSC_HAVE_SYS_PROCFS_H)
/* #include <sys/int_types.h> Required if using gcc on solaris 2.6 */
#include <sys/procfs.h>
#endif
#if defined(PETSC_HAVE_FCNTL_H)
#include <fcntl.h>
#endif

/*@
   PetscMemoryGetCurrentUsage - Returns the current resident set size (memory used)
   for the program.

   Not Collective

   Output Parameter:
.   mem - memory usage in bytes

   Options Database Key:
+  -memory_view - Print memory usage at end of run
-  -malloc_log - Activate logging of memory usage

   Level: intermediate

   Notes:
   The memory usage reported here includes all Fortran arrays
   (that may be used in application-defined sections of code).
   This routine thus provides a more complete picture of memory
   usage than PetscMallocGetCurrentUsage() for codes that employ Fortran with
   hardwired arrays.

.seealso: PetscMallocGetMaximumUsage(), PetscMemoryGetMaximumUsage(), PetscMallocGetCurrentUsage(), PetscMemorySetGetMaximumUsage(), PetscMemoryView()

@*/
PetscErrorCode  PetscMemoryGetCurrentUsage(PetscLogDouble *mem)
{
#if defined(PETSC_USE_PROCFS_FOR_SIZE)
  FILE       *file;
  int        fd;
  char       proc[PETSC_MAX_PATH_LEN];
  prpsinfo_t prusage;
#elif defined(PETSC_USE_SBREAK_FOR_SIZE)
  long       *ii = sbreak(0);
  int        fd  = ii - (long*)0;
#elif defined(PETSC_USE_PROC_FOR_SIZE) && defined(PETSC_HAVE_GETPAGESIZE)
  FILE       *file;
  char       proc[PETSC_MAX_PATH_LEN];
  int        mm,rss,err;
#elif defined(PETSC_HAVE_GETRUSAGE)
  static struct rusage temp;
#endif

  PetscFunctionBegin;
#if defined(PETSC_USE_PROCFS_FOR_SIZE)

  sprintf(proc,"/proc/%d",(int)getpid());
  PetscAssertFalse((fd = open(proc,O_RDONLY)) == -1,PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to access system file %s to get memory usage data",file);
  PetscAssertFalse(ioctl(fd,PIOCPSINFO,&prusage) == -1,PETSC_COMM_SELF,PETSC_ERR_FILE_READ,"Unable to access system file %s to get memory usage data",file);
  *mem = (PetscLogDouble)prusage.pr_byrssize;
  close(fd);

#elif defined(PETSC_USE_SBREAK_FOR_SIZE)

  *mem = (PetscLogDouble)(8*fd - 4294967296); /* 2^32 - upper bits */

#elif defined(PETSC_USE_PROC_FOR_SIZE) && defined(PETSC_HAVE_GETPAGESIZE)
  sprintf(proc,"/proc/%d/statm",(int)getpid());
  PetscAssertFalse(!(file = fopen(proc,"r")),PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to access system file %s to get memory usage data",proc);
  PetscAssertFalse(fscanf(file,"%d %d",&mm,&rss) != 2,PETSC_COMM_SELF,PETSC_ERR_SYS,"Failed to read two integers (mm and rss) from %s",proc);
  *mem = ((PetscLogDouble)rss) * ((PetscLogDouble)getpagesize());
  err  = fclose(file);
  PetscAssertFalse(err,PETSC_COMM_SELF,PETSC_ERR_SYS,"fclose() failed on file");

#elif defined(PETSC_HAVE_GETRUSAGE)
  getrusage(RUSAGE_SELF,&temp);
#if defined(PETSC_USE_KBYTES_FOR_SIZE)
  *mem = 1024.0 * ((PetscLogDouble)temp.ru_maxrss);
#elif defined(PETSC_USE_PAGES_FOR_SIZE) && defined(PETSC_HAVE_GETPAGESIZE)
  *mem = ((PetscLogDouble)getpagesize())*((PetscLogDouble)temp.ru_maxrss);
#else
  *mem = temp.ru_maxrss;
#endif

#else
  *mem = 0.0;
#endif
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscBool      PetscMemoryCollectMaximumUsage;
PETSC_INTERN PetscLogDouble PetscMemoryMaximumUsage;

PetscBool      PetscMemoryCollectMaximumUsage = PETSC_FALSE;
PetscLogDouble PetscMemoryMaximumUsage        = 0;

/*@
   PetscMemoryGetMaximumUsage - Returns the maximum resident set size (memory used)
   for the program.

   Not Collective

   Output Parameter:
.   mem - memory usage in bytes

   Options Database Key:
+  -memory_view - Print memory usage at end of run
-  -malloc_log - Activate logging of memory usage

   Level: intermediate

   Notes:
   The memory usage reported here includes all Fortran arrays
   (that may be used in application-defined sections of code).
   This routine thus provides a more complete picture of memory
   usage than PetscMallocGetCurrentUsage() for codes that employ Fortran with
   hardwired arrays.

.seealso: PetscMallocGetMaximumUsage(), PetscMemoryGetCurrentUsage(), PetscMallocGetCurrentUsage(),
          PetscMemorySetGetMaximumUsage()

@*/
PetscErrorCode  PetscMemoryGetMaximumUsage(PetscLogDouble *mem)
{
  PetscFunctionBegin;
  PetscAssertFalse(!PetscMemoryCollectMaximumUsage,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"To use this function you must first call PetscMemorySetGetMaximumUsage()");
  *mem = PetscMemoryMaximumUsage;
  PetscFunctionReturn(0);
}

/*@
   PetscMemorySetGetMaximumUsage - Tells PETSc to monitor the maximum memory usage so that
       PetscMemoryGetMaximumUsage() will work.

   Not Collective

   Options Database Key:
+  -memory_view - Print memory usage at end of run
-  -malloc_log - Activate logging of memory usage

   Level: intermediate

.seealso: PetscMallocGetMaximumUsage(), PetscMemoryGetCurrentUsage(), PetscMallocGetCurrentUsage(),
          PetscMemoryGetMaximumUsage()

@*/
PetscErrorCode  PetscMemorySetGetMaximumUsage(void)
{
  PetscFunctionBegin;
  PetscMemoryCollectMaximumUsage = PETSC_TRUE;
  PetscFunctionReturn(0);
}
