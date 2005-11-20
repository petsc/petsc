#define PETSC_DLL
/*
      PetscVerboseInfo() is contained in a different file from the other profiling to 
   allow it to be replaced at link time by an alternative routine.
*/
#include "petsc.h"        /*I    "petsc.h"   I*/
#include <stdarg.h>
#include <sys/types.h>
#include "petscsys.h"
#if defined(PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#if defined(PETSC_HAVE_MALLOC_H)
#include <malloc.h>
#endif
#include "petscfix.h"

/*
  The next three variables determine which, if any, PetscVerboseInfo() calls are used.
  If PetscLogPrintInfo is zero, no info messages are printed. 
  If PetscLogPrintInfoNull is zero, no info messages associated with a null object are printed.

  If PetscVerboseInfoFlags[OBJECT_COOKIE - PETSC_COOKIE] is zero, no messages related
  to that object are printed. OBJECT_COOKIE is, for example, MAT_COOKIE.
*/
PetscTruth PETSC_DLLEXPORT PetscLogPrintInfo     = PETSC_FALSE;
PetscTruth PETSC_DLLEXPORT PetscLogPrintInfoNull = PETSC_FALSE;
int        PetscVerboseInfoFlags[]   = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                                    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                                    1,1,1,1,1,1,1,1,1,1,1,1};
FILE      *PetscVerboseInfoFile      = PETSC_NULL;

#undef __FUNCT__  
#define __FUNCT__ "PetscVerboseInfoAllow"
/*@C
    PetscVerboseInfoAllow - Causes PetscVerboseInfo() messages to be printed to standard output.

    Not Collective, each processor may call this separately, but printing is only
    turned on if the lowest processor number associated with the PetscObject associated
    with the call to PetscVerboseInfo() has called this routine.

    Input Parameter:
+   flag - PETSC_TRUE or PETSC_FALSE
-   filename - optional name of file to write output to (defaults to stdout)

    Options Database Key:
.   -verbose_info [optional filename] - Activates PetscVerboseInfoAllow()

    Level: advanced

   Concepts: debugging^detailed runtime information
   Concepts: dumping detailed runtime information

.seealso: PetscVerboseInfo()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscVerboseInfoAllow(PetscTruth flag, const char filename[])
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
    ierr = PetscFOpen(MPI_COMM_SELF, fname, "w", &PetscVerboseInfoFile);CHKERRQ(ierr);
    if (!PetscVerboseInfoFile) SETERRQ1(PETSC_ERR_FILE_OPEN, "Cannot open requested file for writing: %s",fname);
  } else if (flag) {
    PetscVerboseInfoFile = stdout;
  }
  PetscLogPrintInfo     = flag;
  PetscLogPrintInfoNull = flag;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscVerboseInfoDeactivateClass"
/*@
  PetscVerboseInfoDeactivateClass - Deactivates PlogInfo() messages for a PETSc object class.

  Not Collective

  Input Parameter:
. objclass - The object class,  e.g., MAT_COOKIE, SNES_COOKIE, etc.

  Notes:
  One can pass 0 to deactivate all messages that are not associated with an object.

  Level: developer

.keywords: allow, information, printing, monitoring
.seealso: PetscVerboseInfoActivateClass(), PetscVerboseInfo(), PetscVerboseInfoAllow()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscVerboseInfoDeactivateClass(int objclass)
{
  PetscFunctionBegin;
  if (!objclass) {
    PetscLogPrintInfoNull = PETSC_FALSE;
    PetscFunctionReturn(0); 
  }
  PetscVerboseInfoFlags[objclass - PETSC_COOKIE - 1] = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscVerboseInfoActivateClass"
/*@
  PetscVerboseInfoActivateClass - Activates PlogInfo() messages for a PETSc object class.

  Not Collective

  Input Parameter:
. objclass - The object class, e.g., MAT_COOKIE, SNES_COOKIE, etc.

  Notes:
  One can pass 0 to activate all messages that are not associated with an object.

  Level: developer

.keywords: allow, information, printing, monitoring
.seealso: PetscVerboseInfoDeactivateClass(), PetscVerboseInfo(), PetscVerboseInfoAllow()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscVerboseInfoActivateClass(int objclass)
{
  PetscFunctionBegin;
  if (!objclass) {
    PetscLogPrintInfoNull = PETSC_TRUE;
  } else {
    PetscVerboseInfoFlags[objclass - PETSC_COOKIE - 1] = 1;
  }
  PetscFunctionReturn(0);
}

/*
   If the option -log_history was used, then all printed PetscVerboseInfo() 
  messages are also printed to the history file, called by default
  .petschistory in ones home directory.
*/
extern FILE *petsc_history;

#undef __FUNCT__  
#define __FUNCT__ "PetscVerboseInfo"
/*@C
    PetscVerboseInfo - Logs informative data, which is printed to standard output
    or a file when the option -verbose_info <file> is specified.

    Collective over PetscObject argument

   Synopsis:
       PetscErrorCode PetscVerboseInfo((void *vobj, const char message[], ...))  

    Input Parameter:
+   vobj - object most closely associated with the logging statement
-   message - logging message, using standard "printf" format

    Options Database Key:
$    -verbose_info : activates printing of PetscVerboseInfo() messages 

    Level: intermediate

    Note: Since this is a macro you must wrap the arguments in TWO sets of (())

    Fortran Note: This routine is not supported in Fortran.

    Example of Usage:
$
$     Mat A
$     double alpha
$     PetscVerboseInfo((A,"Matrix uses parameter alpha=%g\n",alpha));
$

   Concepts: runtime information

.seealso: PetscVerboseInfoAllow()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscVerboseInfo_Private(void *vobj, const char message[], ...)  
{
  va_list        Argp;
  PetscMPIInt    rank,urank;
  size_t         len;
  PetscObject    obj = (PetscObject)vobj;
  char           string[8*1024];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (obj) PetscValidHeader(obj,1);
  PetscValidCharPointer(message,2);
  if (!PetscLogPrintInfo) PetscFunctionReturn(0);
  if ((!PetscLogPrintInfoNull) && !vobj) PetscFunctionReturn(0);
  if (obj && !PetscVerboseInfoFlags[obj->cookie - PETSC_COOKIE - 1]) PetscFunctionReturn(0);
  if (!obj) {
    rank = 0;
  } else {
    ierr = MPI_Comm_rank(obj->comm, &rank);CHKERRQ(ierr);
  }
  if (rank) PetscFunctionReturn(0);

  ierr = MPI_Comm_rank(MPI_COMM_WORLD, &urank);CHKERRQ(ierr);
  va_start(Argp, message);
  sprintf(string, "[%d]", urank); 
  ierr = PetscStrlen(string, &len);CHKERRQ(ierr);
  ierr = PetscVSNPrintf(string+len, 8*1024-len,message, Argp);
  ierr = PetscFPrintf(PETSC_COMM_SELF,PetscVerboseInfoFile, "%s", string);CHKERRQ(ierr);
  fflush(PetscVerboseInfoFile);
  if (petsc_history) {
    PetscVFPrintf(petsc_history, message, Argp);CHKERRQ(ierr);
  }
  va_end(Argp);
  PetscFunctionReturn(0);
}
