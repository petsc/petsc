
/*
      PetscInfo() is contained in a different file from the other profiling to
   allow it to be replaced at link time by an alternative routine.
*/
#include <petscsys.h>        /*I    "petscsys.h"   I*/
#include <stdarg.h>
#include <sys/types.h>
#if defined(PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#if defined(PETSC_HAVE_MALLOC_H)
#include <malloc.h>
#endif

/*
  The next three variables determine which, if any, PetscInfo() calls are used.
  If PetscLogPrintInfo is zero, no info messages are printed.
  If PetscLogPrintInfoNull is zero, no info messages associated with a null object are printed.

  If PetscInfoFlags[OBJECT_CLASSID - PETSC_SMALLEST_CLASSID] is zero, no messages related
  to that object are printed. OBJECT_CLASSID is, for example, MAT_CLASSID.
*/
PetscBool   PetscLogPrintInfo     = PETSC_FALSE;
PetscBool   PetscLogPrintInfoNull = PETSC_FALSE;
int        PetscInfoFlags[]   = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                                    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                                    1,1,1,1,1,1,1,1,1,1,1,1};
FILE      *PetscInfoFile      = PETSC_NULL;

#undef __FUNCT__
#define __FUNCT__ "PetscInfoAllow"
/*@C
    PetscInfoAllow - Causes PetscInfo() messages to be printed to standard output.

    Not Collective, each processor may call this separately, but printing is only
    turned on if the lowest processor number associated with the PetscObject associated
    with the call to PetscInfo() has called this routine.

    Input Parameter:
+   flag - PETSC_TRUE or PETSC_FALSE
-   filename - optional name of file to write output to (defaults to stdout)

    Options Database Key:
.   -info [optional filename] - Activates PetscInfoAllow()

    Level: advanced

   Concepts: debugging^detailed runtime information
   Concepts: dumping detailed runtime information

.seealso: PetscInfo()
@*/
PetscErrorCode  PetscInfoAllow(PetscBool  flag, const char filename[])
{
  char           fname[PETSC_MAX_PATH_LEN], tname[5];
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (flag && filename) {
    ierr = PetscFixFilename(filename, fname);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRQ(ierr);
    sprintf(tname, ".%d", rank);
    ierr = PetscStrcat(fname, tname);CHKERRQ(ierr);
    ierr = PetscFOpen(MPI_COMM_SELF, fname, "w", &PetscInfoFile);CHKERRQ(ierr);
    if (!PetscInfoFile) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN, "Cannot open requested file for writing: %s",fname);
  } else if (flag) {
    PetscInfoFile = PETSC_STDOUT;
  }
  PetscLogPrintInfo     = flag;
  PetscLogPrintInfoNull = flag;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscInfoDeactivateClass"
/*@
  PetscInfoDeactivateClass - Deactivates PlogInfo() messages for a PETSc object class.

  Not Collective

  Input Parameter:
. objclass - The object class,  e.g., MAT_CLASSID, SNES_CLASSID, etc.

  Notes:
  One can pass 0 to deactivate all messages that are not associated with an object.

  Level: developer

.keywords: allow, information, printing, monitoring
.seealso: PetscInfoActivateClass(), PetscInfo(), PetscInfoAllow()
@*/
PetscErrorCode  PetscInfoDeactivateClass(int objclass)
{
  PetscFunctionBegin;
  if (!objclass) {
    PetscLogPrintInfoNull = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
  PetscInfoFlags[objclass - PETSC_SMALLEST_CLASSID - 1] = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscInfoActivateClass"
/*@
  PetscInfoActivateClass - Activates PlogInfo() messages for a PETSc object class.

  Not Collective

  Input Parameter:
. objclass - The object class, e.g., MAT_CLASSID, SNES_CLASSID, etc.

  Notes:
  One can pass 0 to activate all messages that are not associated with an object.

  Level: developer

.keywords: allow, information, printing, monitoring
.seealso: PetscInfoDeactivateClass(), PetscInfo(), PetscInfoAllow()
@*/
PetscErrorCode  PetscInfoActivateClass(int objclass)
{
  PetscFunctionBegin;
  if (!objclass) {
    PetscLogPrintInfoNull = PETSC_TRUE;
  } else {
    PetscInfoFlags[objclass - PETSC_SMALLEST_CLASSID - 1] = 1;
  }
  PetscFunctionReturn(0);
}

/*
   If the option -history was used, then all printed PetscInfo()
  messages are also printed to the history file, called by default
  .petschistory in ones home directory.
*/
extern FILE *petsc_history;

#undef __FUNCT__
#define __FUNCT__ "PetscInfo_Private"
/*MC
    PetscInfo - Logs informative data, which is printed to standard output
    or a file when the option -info <file> is specified.

   Synopsis:
       PetscErrorCode PetscInfo(void *vobj, const char message[])
       PetscErrorCode PetscInfo1(void *vobj, const char formatmessage[],arg1)
       PetscErrorCode PetscInfo2(void *vobj, const char formatmessage[],arg1,arg2)
       etc

    Collective over PetscObject argument

    Input Parameter:
+   vobj - object most closely associated with the logging statement or PETSC_NULL
.   message - logging message
-   formatmessage - logging message using standard "printf" format

    Options Database Key:
$    -info : activates printing of PetscInfo() messages

    Level: intermediate

    Fortran Note: This function does not take the vobj argument, there is only the PetscInfo()
     version, not PetscInfo1() etc.

    Example of Usage:
$
$     Mat A
$     double alpha
$     PetscInfo1(A,"Matrix uses parameter alpha=%g\n",alpha);
$

   Concepts: runtime information

.seealso: PetscInfoAllow()
M*/
PetscErrorCode  PetscInfo_Private(const char func[],void *vobj, const char message[], ...)
{
  va_list        Argp;
  PetscMPIInt    rank,urank;
  size_t         len;
  PetscObject    obj = (PetscObject)vobj;
  char           string[8*1024];
  PetscErrorCode ierr;
  size_t         fullLength;
  int            err;

  PetscFunctionBegin;
  if (obj) PetscValidHeader(obj,1);
  PetscValidCharPointer(message,2);
  if (!PetscLogPrintInfo) PetscFunctionReturn(0);
  if ((!PetscLogPrintInfoNull) && !vobj) PetscFunctionReturn(0);
  if (obj && !PetscInfoFlags[obj->classid - PETSC_SMALLEST_CLASSID - 1]) PetscFunctionReturn(0);
  if (!obj) {
    rank = 0;
  } else {
    ierr = MPI_Comm_rank(obj->comm, &rank);CHKERRQ(ierr);
  }
  if (rank) PetscFunctionReturn(0);

  ierr = MPI_Comm_rank(MPI_COMM_WORLD, &urank);CHKERRQ(ierr);
  va_start(Argp, message);
  sprintf(string, "[%d] %s(): ", urank,func);
  ierr = PetscStrlen(string, &len);CHKERRQ(ierr);
  ierr = PetscVSNPrintf(string+len, 8*1024-len,message,&fullLength, Argp);
  ierr = PetscFPrintf(PETSC_COMM_SELF,PetscInfoFile, "%s", string);CHKERRQ(ierr);
  err = fflush(PetscInfoFile);
  if (err) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SYS,"fflush() failed on file");
  if (petsc_history) {
    va_start(Argp, message);
    (*PetscVFPrintf)(petsc_history, message, Argp);CHKERRQ(ierr);
  }
  va_end(Argp);
  PetscFunctionReturn(0);
}
