/*$Id: mpiuopen.c,v 1.22 1999/12/22 03:31:33 bsmith Exp bsmith $*/
/*
      Some PETSc utilites routines to add simple parallel IO capability
*/
#include "petsc.h"
#include "sys.h"
#include <stdarg.h>
#if defined(PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#include "petscfix.h"

#undef __FUNC__  
#define __FUNC__ "PetscFOpen"
/*@C
    PetscFOpen - Has the first process in the communicator open a file;
    all others do nothing.

    Collective on MPI_Comm

    Input Parameters:
+   comm - the communicator
.   name - the filename
-   mode - the mode for fopen(), usually "w"

    Output Parameter:
.   fp - the file pointer

    Level: developer

    Notes:
       PETSC_NULL (0), "stderr" or "stdout" may be passed in as the filename
  
    Fortran Note:
    This routine is not supported in Fortran.

.keywords: file, open

.seealso: PetscFClose()
@*/
int PetscFOpen(MPI_Comm comm,const char name[],const char mode[],FILE **fp)
{
  int  rank,ierr;
  FILE *fd;
  char fname[256];

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    PetscTruth isstdout,isstderr;
    ierr = PetscStrcmp(name,"stdout",&isstdout);CHKERRQ(ierr);
    ierr = PetscStrcmp(name,"stderr",&isstderr);CHKERRQ(ierr);
    if (isstdout || !name) {
      fd = stdout;
    } else if (isstderr) {
      fd = stderr;
    } else {
      ierr = PetscFixFilename(name,fname);CHKERRQ(ierr);
      fd = fopen(fname,mode);
    }
  } else fd = 0;
  *fp = fd;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscFClose"
/*@C
    PetscFClose - Has the first processor in the communicator close a 
    file; all others do nothing.

    Collective on MPI_Comm

    Input Parameters:
+   comm - the communicator
-   fd - the file, opened with PetscFOpen()

   Level: developer

    Fortran Note:
    This routine is not supported in Fortran.

.keywords: file, close

.seealso: PetscFOpen()
@*/
int PetscFClose(MPI_Comm comm,FILE *fd)
{
  int  rank,ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank && fd != stdout && fd != stderr) fclose(fd);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscPClose"
int PetscPClose(MPI_Comm comm,FILE *fd)
{
  int  rank,ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) fclose(fd);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscPOpen"
/*@C
      PetscPOpen - Runs a program on processor zero and sends either its input or output to 
          a file.

     Not Collective

   Input Parameters:
+   comm - MPI communicator, only processor zero runs the program
.   machine - machine to run command on or PETSC_NULL
.   program - name of program to run
-   mode - either r or w

   Output Parameter:
.   fp - the file pointer where program input or output may be read

   Level: intermediate

   Notes:
       Does not work under Windows

       The program string may contain $DISPLAY, $HOMEDIRECTORY or $WORKINGDIRECTORY; these
    will be replaced with relevent values.

.seealso: PetscFOpen(), PetscFClose(), PetscPClose()

@*/
int PetscPOpen(MPI_Comm comm,char *machine,char *program,const char mode[],FILE **fp)
{
  int  ierr,rank;
  FILE *fd;
  char commandt[1024],command[1024];
  char *s[] = {"$DISPLAY","$HOMEDIRECTORY","$WORKINGDIRECTORY",0},*r[4];

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {

    if (machine) {
      ierr = PetscStrcpy(command,"rsh ");CHKERRQ(ierr);
      ierr = PetscStrcat(command,machine);CHKERRQ(ierr);
      ierr = PetscStrcat(command," ");CHKERRQ(ierr);
      ierr = PetscStrcat(command,program);CHKERRQ(ierr);
    } else {
      ierr = PetscStrcpy(command,program);CHKERRQ(ierr);
    }

    /* get values for replaced variables */
    r[0] = (char*)PetscMalloc(256*sizeof(char));CHKERRQ(ierr);
    r[1] = (char*)PetscMalloc(256*sizeof(char));CHKERRQ(ierr);
    r[2] = (char*)PetscMalloc(256*sizeof(char));CHKERRQ(ierr);
    ierr = PetscGetDisplay(r[0],256);CHKERRQ(ierr);
    ierr = PetscGetHomeDirectory(r[1],256);CHKERRQ(ierr);
    ierr = PetscGetWorkingDirectory(r[2],256);CHKERRQ(ierr);

    ierr = PetscStrreplace(command,commandt,1024,s,r);CHKERRQ(ierr);
    
    ierr = PetscFree(r[0]);CHKERRQ(ierr);
    ierr = PetscFree(r[1]);CHKERRQ(ierr);
    ierr = PetscFree(r[2]);CHKERRQ(ierr);
    
    PLogInfo(0,"Running command :%s\n",commandt);

#if defined (PARCH_win32)
    SETERRQ(1,1,"Cannot run programs on NT");
#else 
    if (!(fd = popen(commandt,mode))) {
       SETERRQ1(1,1,"Cannot run command %s",commandt);
    }
#endif
    if (fp) *fp = fd;
  }
  PetscFunctionReturn(0);
}
