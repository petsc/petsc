
#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: fretrieve.c,v 1.8 1999/03/17 23:21:32 bsmith Exp bsmith $";
#endif
/*
      Code for opening and closing files.
*/
#include "petsc.h"
#include "sys.h"
#include "pinclude/ptime.h"
#if defined(HAVE_PWD_H)
#include <pwd.h>
#endif
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

extern int Petsc_DelTag(MPI_Comm,int,void*,void*);

#undef __FUNC__  
#define __FUNC__ "PetscSharedTmp"
/*@C
   PetscSharedTmp _ Determines if all processors in a communicator share a
         /tmp or have different ones.

   Collective on MPI_Comm

   Input Parameters:
.  comm - MPI_Communicator that may share /tmp

   Output Parameters:
.  shared - PETSC_TRUE or PETSC_FALSE

   Level: developer

   Notes:
   Stores the status as a MPI attribute so it does not have
    to be redetermined each time.

      Assumes that all processors in a communicator either
       1) have a common /tmp or
       2) each has a seperate /tmp
      eventually we can write a fancier one that determines which processors
      share a common /tmp.

   This will be very slow on runs with a large number of files since
   it requires O(p*p) file opens.

   If the environmental variable PETSC_TMP is set it will use this directory
  as the "/tmp" directory.

@*/
int PetscSharedTmp(MPI_Comm comm,PetscTruth *shared)
{
  int        ierr,size,rank,iflag,*tagvalp,sum,cnt,i;
  PetscTruth flag;
  FILE       *fd;
  static int Petsc_Tmp_keyval = MPI_KEYVAL_INVALID;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size == 1) {
    *shared = PETSC_TRUE;
    PetscFunctionReturn(0);
  }

  if (Petsc_Tmp_keyval == MPI_KEYVAL_INVALID) {
    ierr = MPI_Keyval_create(MPI_NULL_COPY_FN,Petsc_DelTag,&Petsc_Tmp_keyval,0);CHKERRQ(ierr);
  }

  ierr = MPI_Attr_get(comm,Petsc_Tmp_keyval,(void**)&tagvalp,&iflag);CHKERRQ(ierr);
  if (!iflag) {
    char       filename[256];
    /* This communicator does not yet have a shared tmp attribute */
    tagvalp    = (int *) PetscMalloc( sizeof(int) ); CHKPTRQ(tagvalp);
    ierr       = MPI_Attr_put(comm,Petsc_Tmp_keyval, tagvalp);CHKERRQ(ierr);

    ierr = OptionsGetenv(comm,"PETSC_TMP",filename,238,&flag);CHKERRQ(ierr);
    if (!flag) {
      ierr = PetscStrcpy(filename,"/tmp");CHKERRQ(ierr);
    }
    ierr = PetscStrcat(filename,"/petsctestshared");CHKERRQ(ierr);
    ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
    
    /* each processor creates a /tmp file and all the later ones check */
    /* this makes sure no subset of processors is shared */
    *shared = PETSC_FALSE;
    for ( i=0; i<size-1; i++ ) {
      if (rank == i) {
        fd = fopen(filename,"w");
        if (!fd) {
          SETERRQ(1,1,"Unable to open test file in /tmp directory");
        }
        fclose(fd);
      }
      ierr = MPI_Barrier(comm);CHKERRQ(ierr);
      if (rank >= i) {
        fd = fopen(filename,"r");
        if (fd) cnt = 1; else cnt = 0;
        if (fd) {
          fclose(fd);
        }
      } else {
        cnt = 0;
      }
      ierr = MPI_Allreduce(&cnt,&sum,1,MPI_INT,MPI_SUM,comm);CHKERRQ(ierr);
      if (rank == i) {
        unlink(filename);
      }

      if (sum == size) {
        *shared = PETSC_TRUE;
        break;
      } else if (sum != 1) {
        SETERRQ(1,1,"Subset of processes share /tmp cannot load remote or compressed file");
      }
    }
    *tagvalp = (int) *shared;
  } else {
    *shared = (PetscTruth) *tagvalp;
  }
  PLogInfo(0,"PetscSharedTmp:1 indicates detected shared /tmp, 0 not shared %d\n",(int) *shared);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscFileRetrieve"
/*@C
    PetscFileRetrieve - Obtains a library from a URL or compressed 
        and copies into local disk space as uncompressed.

    Collective on MPI_Comm

    Input Parameter:
+   comm     - processors accessing the library
.   libname  - name of library, including entire URL (with or without .gz)
-   llen     - length of llibname

    Output Parameter:
+   llibname - name of local copy of library
-   found - if found and retrieve the file

    Level: developer

@*/
int PetscFileRetrieve(MPI_Comm comm,const char *libname,char *llibname,int llen,PetscTruth *found)
{
  char       *par,buf[1024];
  FILE       *fp;
  int        i,rank,ierr;
  PetscTruth sharedtmp;

  PetscFunctionBegin;
  *found = PETSC_FALSE;

  /* if file does not have an ftp:// or http:// or .gz then need not process file */
  if (PetscStrncmp(libname,"ftp://",6) && PetscStrncmp(libname,"http://",7) &&
      (!(par = PetscStrstr(libname,".gz")) || PetscStrlen(par) != 3)) {
    ierr = PetscStrncpy(llibname,libname,llen);CHKERRQ(ierr);
    ierr = PetscTestFile(libname,'r',found);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  /* Determine if all processors share a common /tmp */
  ierr = PetscSharedTmp(comm,&sharedtmp);CHKERRQ(ierr);

  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank || !sharedtmp) {
  
    /* Construct the Python script to get URL file */
    par = (char *) PetscMalloc(1024*sizeof(char));CHKPTRQ(par);
    ierr = PetscStrcpy(par,"python1.5 ");
    ierr = PetscStrcat(par,PETSC_DIR);
    ierr = PetscStrcat(par,"/bin/urlget.py ");
    ierr = PetscStrcat(par,libname);
    ierr = PetscStrcat(par," 2>&1 ");

    PLogInfo(0,"PetscFileRetrieve: Running python script:%s\n",par);
#if defined (PARCH_win32)
  SETERRQ(1,1,"Cannot use PetscFileRetrieve on NT");
#else 
    if ((fp = popen(par,"r")) == 0) {
      SETERRQ(1,1,"Cannot Execute python1.5 on ${PETSC_DIR}/bin/urlget.py\n\
        Check if python1.5 is in your path");
    }
#endif
    if (fgets(buf,1024,fp) == 0) {
      SETERRQ1(1,1,"No output from ${PETSC_DIR}/bin/urlget.py in getting file %s",libname);
    }
    /* Check for \n and make it 0 */
    for ( i=0; i<1024; i++ ) {
      if ( buf[i] == '\n') {
        buf[i] = 0;
        break;
      }
    }
    if (!PetscStrncmp(buf,"Error",5) ||!PetscStrncmp(buf,"Traceback",9)) {
      PLogInfo(0,"PetscFileRetrieve:Did not find file %s",libname);
    } else {
      *found = PETSC_TRUE;
    }
    ierr = PetscStrncpy(llibname,buf,llen);CHKERRQ(ierr);
    PetscFree(par);
  }
  if (sharedtmp) { /* send library name to all processors */
    MPI_Bcast(llibname,llen,MPI_CHAR,0,comm);
    MPI_Bcast(found,1,MPI_INT,0,comm);
  }

  PetscFunctionReturn(0);
}
