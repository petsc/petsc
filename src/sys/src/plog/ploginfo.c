/*$Id: ploginfo.c,v 1.22 2001/03/23 23:20:50 balay Exp $*/
/*
      PetscLogInfo() is contained in a different file from the other profiling to 
   allow it to be replaced at link time by an alternative routine.
*/
#include "petscconfig.h"
#include "petsc.h"        /*I    "petsc.h"   I*/
#include <stdarg.h>
#include <sys/types.h>
#include "petscsys.h"
#if defined(PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#if defined(PETSC_HAVE_MALLOC_H) && !defined(__cplusplus)
#include <malloc.h>
#endif
#include "petscfix.h"

/*
  The next three variables determine which, if any, PetscLogInfo() calls are used.
  If PetscLogPrintInfo is zero, no info messages are printed. 
  If PetscLogPrintInfoNull is zero, no info messages associated with a null object are printed.

  If PetscLogInfoFlags[OBJECT_COOKIE - PETSC_COOKIE] is zero, no messages related
  to that object are printed. OBJECT_COOKIE is, for example, MAT_COOKIE.
*/
PetscTruth PetscLogPrintInfo     = PETSC_FALSE;
PetscTruth PetscLogPrintInfoNull = PETSC_FALSE;
int        PetscLogInfoFlags[]   = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                                    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                                    1,1,1,1,1,1,1,1,1,1,1,1};
FILE      *PetscLogInfoFile      = PETSC_NULL;

#undef __FUNCT__  
#define __FUNCT__ "PetscLogInfoAllow"
/*@C
    PetscLogInfoAllow - Causes PetscLogInfo() messages to be printed to standard output.

    Not Collective, each processor may call this seperately, but printing is only
    turned on if the lowest processor number associated with the PetscObject associated
    with the call to PetscLogInfo() has called this routine.

    Input Parameter:
+   flag - PETSC_TRUE or PETSC_FALSE
-   filename - optional name of file to write output to (defaults to stdout)

    Options Database Key:
.   -log_info - Activates PetscLogInfoAllow()

    Level: advanced

   Concepts: debugging^detailed runtime information
   Concepts: dumping detailed runtime information

.seealso: PetscLogInfo()
@*/
int PetscLogInfoAllow(PetscTruth flag, char *filename)
{
  char fname[256], tname[5];
  int  rank;
  int  ierr;

  PetscFunctionBegin;
  PetscLogPrintInfo     = flag;
  PetscLogPrintInfoNull = flag;
  if (flag && filename) {
    ierr = PetscFixFilename(filename, fname);                                                             CHKERRQ(ierr);
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);                                                        CHKERRQ(ierr);
    sprintf(tname, ".%d", rank);
    ierr = PetscStrcat(fname, tname);                                                                     CHKERRQ(ierr);
    ierr = PetscFOpen(PETSC_COMM_SELF, fname, "w", &PetscLogInfoFile);                                    CHKERRQ(ierr);
    if (PetscLogInfoFile == PETSC_NULL) SETERRQ1(PETSC_ERR_FILE_OPEN, "Cannot open requested file for writing: %s",fname);
  } else if (flag) {
    PetscLogInfoFile = stdout;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscLogInfoDeactivateClass"
/*@
  PetscLogInfoDeactivateClass - Deactivates PlogInfo() messages for a PETSc object class.

  Not Collective

  Input Parameter:
. objclass - The object class,  e.g., MAT_COOKIE, SNES_COOKIE, etc.

  Notes:
  One can pass 0 to deactivate all messages that are not associated with an object.

  Level: developer

.keywords: allow, information, printing, monitoring
.seealso: PetscLogInfoActivateClass(), PetscLogInfo(), PetscLogInfoAllow()
@*/
int PetscLogInfoDeactivateClass(int objclass)
{
  PetscFunctionBegin;
  if (objclass == 0) {
    PetscLogPrintInfoNull = PETSC_FALSE;
    PetscFunctionReturn(0); 
  }
  PetscLogInfoFlags[objclass - PETSC_COOKIE - 1] = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscLogInfoActivateClass"
/*@
  PetscLogInfoActivateClass - Activates PlogInfo() messages for a PETSc object class.

  Not Collective

  Input Parameter:
. objclass - The object class, e.g., MAT_COOKIE, SNES_COOKIE, etc.

  Notes:
  One can pass 0 to activate all messages that are not associated with an object.

  Level: developer

.keywords: allow, information, printing, monitoring
.seealso: PetscLogInfoDeactivateClass(), PetscLogInfo(), PetscLogInfoAllow()
@*/
int PetscLogInfoActivateClass(int objclass)
{
  PetscFunctionBegin;
  if (objclass == 0) {
    PetscLogPrintInfoNull = PETSC_TRUE;
  } else {
    PetscLogInfoFlags[objclass - PETSC_COOKIE - 1] = 1;
  }
  PetscFunctionReturn(0);
}

/*
   If the option -log_history was used, then all printed PetscLogInfo() 
  messages are also printed to the history file, called by default
  .petschistory in ones home directory.
*/
extern FILE *petsc_history;

#undef __FUNCT__  
#define __FUNCT__ "PetscLogInfo"
/*@C
    PetscLogInfo - Logs informative data, which is printed to standard output
    or a file when the option -log_info <file> is specified.

    Collective over PetscObject argument

    Input Parameter:
+   vobj - object most closely associated with the logging statement
-   message - logging message, using standard "printf" format

    Options Database Key:
$    -log_info : activates printing of PetscLogInfo() messages 

    Level: intermediate

    Fortran Note:
    This routine is not supported in Fortran.

    Example of Usage:
$
$     Mat A
$     double alpha
$     PetscLogInfo(A,"Matrix uses parameter alpha=%g\n",alpha);
$

   Concepts: runtime information

.seealso: PetscLogInfoAllow()
@*/
int PetscLogInfo(void *vobj, const char message[], ...)  
{
  va_list     Argp;
  int         rank,urank,len;
  PetscObject obj = (PetscObject)vobj;
  char        string[8*1024];
  int         ierr;

  PetscFunctionBegin;
  if (obj) PetscValidHeader(obj);
  if (PetscLogPrintInfo == PETSC_FALSE) PetscFunctionReturn(0);
  if ((PetscLogPrintInfoNull == PETSC_FALSE) && !vobj) PetscFunctionReturn(0);
  if (obj && !PetscLogInfoFlags[obj->cookie - PETSC_COOKIE - 1]) PetscFunctionReturn(0);
  if (!obj) {
    rank = 0;
  } else {
    ierr = MPI_Comm_rank(obj->comm, &rank);                                                               CHKERRQ(ierr);
  }
  if (rank) PetscFunctionReturn(0);

  ierr = MPI_Comm_rank(MPI_COMM_WORLD, &urank);                                                           CHKERRQ(ierr);
  va_start(Argp, message);
  sprintf(string, "[%d]", urank); 
  ierr = PetscStrlen(string, &len);                                                                       CHKERRQ(ierr);
#if defined(PETSC_HAVE_VPRINTF_CHAR)
  vsprintf(string+len, message, (char *) Argp);
#else
  vsprintf(string+len, message, Argp);
#endif
  fprintf(PetscLogInfoFile, "%s", string);
  fflush(PetscLogInfoFile);
  if (petsc_history) {
#if defined(PETSC_HAVE_VPRINTF_CHAR)
    vfprintf(petsc_history, message, (char *) Argp);
#else
    vfprintf(petsc_history, message, Argp);
#endif
  }
  va_end(Argp);
  PetscFunctionReturn(0);
}
