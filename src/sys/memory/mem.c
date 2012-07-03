
#include <petscsys.h>           /*I "petscsys.h" I*/
#if defined(PETSC_HAVE_PWD_H)
#include <pwd.h>
#endif
#include <ctype.h>
#include <sys/types.h>
#include <sys/stat.h>
#if defined(PETSC_HAVE_UNISTD_H)
#include <unistd.h>
#endif
#if defined(PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#if defined(PETSC_HAVE_SYS_UTSNAME_H)
#include <sys/utsname.h>
#endif
#include <fcntl.h>
#include <time.h>  
#if defined(PETSC_HAVE_SYS_SYSTEMINFO_H)
#include <sys/systeminfo.h>
#endif

/* task_info seems to be buggy plus pgcc doesn't like including this file
#if defined(PETSC_HAVE_TASK_INFO)
#include <mach/mach.h>
#endif
*/

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

#undef __FUNCT__  
#define __FUNCT__ "PetscMemoryGetCurrentUsage"
/*@
   PetscMemoryGetCurrentUsage - Returns the current resident set size (memory used)
   for the program.

   Not Collective

   Output Parameter:
.   mem - memory usage in bytes

   Options Database Key:
.  -memory_info - Print memory usage at end of run
.  -malloc_log - Activate logging of memory usage

   Level: intermediate

   Notes:
   The memory usage reported here includes all Fortran arrays 
   (that may be used in application-defined sections of code).
   This routine thus provides a more complete picture of memory
   usage than PetscMallocGetCurrentUsage() for codes that employ Fortran with
   hardwired arrays.

.seealso: PetscMallocGetMaximumUsage(), PetscMemoryGetMaximumUsage(), PetscMallocGetCurrentUsage()

   Concepts: resident set size
   Concepts: memory usage

@*/
PetscErrorCode  PetscMemoryGetCurrentUsage(PetscLogDouble *mem)
{
#if defined(PETSC_USE_PROCFS_FOR_SIZE)
  FILE                   *file;
  int                    fd;
  char                   proc[PETSC_MAX_PATH_LEN];
  prpsinfo_t             prusage;
#elif defined(PETSC_USE_SBREAK_FOR_SIZE)
  long                   *ii = sbreak(0); 
  int                    fd = ii - (long*)0; 
#elif defined(PETSC_USE_PROC_FOR_SIZE) && defined(PETSC_HAVE_GETPAGESIZE)
  FILE                   *file;
  char                   proc[PETSC_MAX_PATH_LEN];
  int                    mm,rss,err;
#elif defined(PETSC_HAVE_TASK_INFO)
  /*  task_basic_info_data_t ti;
      unsigned int           count; */
  /* 
     The next line defined variables that are not used; but if they 
     are not included the code crashes. Something must be wrong
     with either the task_info() command or compiler corrupting the 
     stack.
  */
  /* kern_return_t          kerr; */
#elif defined(PETSC_HAVE_GETRUSAGE)
  static struct rusage   temp;
#endif

  PetscFunctionBegin;
#if defined(PETSC_USE_PROCFS_FOR_SIZE)

  sprintf(proc,"/proc/%d",(int)getpid());
  if ((fd = open(proc,O_RDONLY)) == -1) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to access system file %s to get memory usage data",file);
  }
  if (ioctl(fd,PIOCPSINFO,&prusage) == -1) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_READ,"Unable to access system file %s to get memory usage data",file); 
  }
  *mem = (PetscLogDouble)prusage.pr_byrssize;
  close(fd);

#elif defined(PETSC_USE_SBREAK_FOR_SIZE)

  *mem = (PetscLogDouble)(8*fd - 4294967296); /* 2^32 - upper bits */

#elif defined(PETSC_USE_PROC_FOR_SIZE) && defined(PETSC_HAVE_GETPAGESIZE)
  sprintf(proc,"/proc/%d/statm",(int)getpid());
  if (!(file = fopen(proc,"r"))) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to access system file %s to get memory usage data",proc);
  }
  if (fscanf(file,"%d %d",&mm,&rss) != 2) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SYS,"Failed to read two integers (mm and rss) from %s",proc);
  *mem = ((PetscLogDouble)rss) * ((PetscLogDouble)getpagesize());
  err = fclose(file);
  if (err) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SYS,"fclose() failed on file");    

#elif defined(PETSC_HAVE_TASK_INFO)
  *mem = 0;
  /* if ((kerr = task_info(mach_task_self(), TASK_BASIC_INFO, (task_info_t)&ti,&count)) != KERN_SUCCESS) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Mach system call failed: kern_return_t ",kerr);
   *mem = (PetscLogDouble) ti.resident_size; */
  
#elif defined(PETSC_HAVE_GETRUSAGE)
  getrusage(RUSAGE_SELF,&temp);
#if defined(PETSC_USE_KBYTES_FOR_SIZE)
  *mem = 1024.0 * ((PetscLogDouble)temp.ru_maxrss);
#elif defined(PETSC_HAVE_GETPAGESIZE)
  *mem = ((PetscLogDouble)getpagesize())*((PetscLogDouble)temp.ru_maxrss);
#else
  *mem = 0.0;
#endif

#else
  *mem = 0.0;
#endif
  PetscFunctionReturn(0);
}

PetscBool      PetscMemoryCollectMaximumUsage = PETSC_FALSE;
PetscLogDouble PetscMemoryMaximumUsage = 0;

#undef __FUNCT__  
#define __FUNCT__ "PetscMemoryGetMaximumUsage"
/*@
   PetscMemoryGetMaximumUsage - Returns the maximum resident set size (memory used)
   for the program.

   Not Collective

   Output Parameter:
.   mem - memory usage in bytes

   Options Database Key:
.  -memory_info - Print memory usage at end of run
.  -malloc_log - Activate logging of memory usage

   Level: intermediate

   Notes:
   The memory usage reported here includes all Fortran arrays 
   (that may be used in application-defined sections of code).
   This routine thus provides a more complete picture of memory
   usage than PetscMallocGetCurrentUsage() for codes that employ Fortran with
   hardwired arrays.

.seealso: PetscMallocGetMaximumUsage(), PetscMemoryGetCurrentUsage(), PetscMallocGetCurrentUsage(),
          PetscMemorySetGetMaximumUsage()

   Concepts: resident set size
   Concepts: memory usage

@*/
PetscErrorCode  PetscMemoryGetMaximumUsage(PetscLogDouble *mem)
{
  PetscFunctionBegin;
  if (!PetscMemoryCollectMaximumUsage) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"To use this function you must first call PetscMemorySetGetMaximumUsage()");
  *mem = PetscMemoryMaximumUsage;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscMemorySetGetMaximumUsage"
/*@C
   PetscMemorySetGetMaximumUsage - Tells PETSc to monitor the maximum memory usage so that
       PetscMemoryGetMaximumUsage() will work.

   Not Collective

   Options Database Key:
.  -memory_info - Print memory usage at end of run
.  -malloc_log - Activate logging of memory usage

   Level: intermediate

.seealso: PetscMallocGetMaximumUsage(), PetscMemoryGetCurrentUsage(), PetscMallocGetCurrentUsage(),
          PetscMemoryGetMaximumUsage()

   Concepts: resident set size
   Concepts: memory usage

@*/
PetscErrorCode  PetscMemorySetGetMaximumUsage(void)
{
  PetscFunctionBegin;
  PetscMemoryCollectMaximumUsage = PETSC_TRUE;
  PetscFunctionReturn(0);
}
