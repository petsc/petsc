/*$Id: fretrieve.c,v 1.21 1999/11/10 03:17:57 bsmith Exp bsmith $*/
/*
      Code for opening and closing files.
*/
#include "petsc.h"
#include "sys.h"
#include "pinclude/ptime.h"
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
#include "pinclude/petscfix.h"

EXTERN_C_BEGIN
extern int Petsc_DelTag(MPI_Comm,int,void*,void*);
EXTERN_C_END

#undef __FUNC__  
#define __FUNC__ "PetscSharedTmp"
/*@C
   PetscSharedTmp - Determines if all processors in a communicator share a
         /tmp or have different ones.

   Collective on MPI_Comm

   Input Parameters:
.  comm - MPI_Communicator that may share /tmp

   Output Parameters:
.  shared - PETSC_TRUE or PETSC_FALSE

   Options Database Keys:
+    -petsc_shared_tmp 
.    -petsc_not_shared_tmp
-    -petsc_tmp tmpdir

   Environmental Variables:
+     PETSC_SHARED_TMP
.     PETSC_NOT_SHARED_TMP
-     PETSC_TMP

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
  int        ierr,size,rank,*tagvalp,sum,cnt,i;
  PetscTruth flg,iflg;
  FILE       *fd;
  static int Petsc_Tmp_keyval = MPI_KEYVAL_INVALID;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size == 1) {
    *shared = PETSC_TRUE;
    PetscFunctionReturn(0);
  }

  ierr = OptionsHasName(PETSC_NULL,"-petsc_shared_tmp",&iflg);CHKERRQ(ierr);
  if (iflg) {
    *shared = PETSC_TRUE;
    PetscFunctionReturn(0);
  }

  ierr = OptionsGetenv(comm,"PETSC_SHARED_TMP",PETSC_NULL,0,&flg);CHKERRQ(ierr);
  if (flg) {
    *shared = PETSC_TRUE;
    PetscFunctionReturn(0);
  }

  ierr = OptionsHasName(PETSC_NULL,"-petsc_not_shared_tmp",&iflg);CHKERRQ(ierr);
  if (iflg) {
    *shared = PETSC_FALSE;
    PetscFunctionReturn(0);
  }

  ierr = OptionsGetenv(comm,"PETSC_NOT_SHARED_TMP",PETSC_NULL,0,&flg);CHKERRQ(ierr);
  if (flg) {
    *shared = PETSC_FALSE;
    PetscFunctionReturn(0);
  }

  if (Petsc_Tmp_keyval == MPI_KEYVAL_INVALID) {
    ierr = MPI_Keyval_create(MPI_NULL_COPY_FN,Petsc_DelTag,&Petsc_Tmp_keyval,0);CHKERRQ(ierr);
  }

  ierr = MPI_Attr_get(comm,Petsc_Tmp_keyval,(void**)&tagvalp,(int*)&iflg);CHKERRQ(ierr);
  if (!iflg) {
    char       filename[256];

    /* This communicator does not yet have a shared tmp attribute */
    tagvalp    = (int *) PetscMalloc( sizeof(int) );CHKPTRQ(tagvalp);
    ierr       = MPI_Attr_put(comm,Petsc_Tmp_keyval, tagvalp);CHKERRQ(ierr);

    ierr = OptionsGetString(PETSC_NULL,"-petsc_tmp",filename,238,&iflg);CHKERRQ(ierr);
    if (!iflg) {
      ierr = OptionsGetenv(comm,"PETSC_TMP",filename,238,&iflg);CHKERRQ(ierr);
      if (!iflg) {
        ierr = PetscStrcpy(filename,"/tmp");CHKERRQ(ierr);
      }
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
          SETERRQ1(1,1,"Unable to open test file %s",filename);
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
  char              *par,buf[1024],tmpdir[256];
  FILE              *fp;
  int               i,rank,ierr,len = 0;
  PetscTruth        flg1,flg2,sharedtmp;

  PetscFunctionBegin;
  *found = PETSC_FALSE;

  /* if file does not have an ftp:// or http:// or .gz then need not process file */
  ierr = PetscStrstr(libname,".gz",&par);CHKERRQ(ierr);
  if (par) {ierr = PetscStrlen(par,&len);CHKERRQ(ierr);}

  ierr = PetscStrncmp(libname,"ftp://",6,&flg1);CHKERRQ(ierr);
  ierr = PetscStrncmp(libname,"http://",7,&flg2);CHKERRQ(ierr);
  if (!flg1 && !flg2 && (!par || len != 3)) {
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
    ierr = PetscStrcpy(par,"python1.5 ");CHKERRQ(ierr);
    ierr = PetscStrcat(par,PETSC_DIR);CHKERRQ(ierr);
    ierr = PetscStrcat(par,"/bin/urlget.py ");CHKERRQ(ierr);

    /* are we using an alternative /tmp? */
    ierr = OptionsGetString(PETSC_NULL,"-petsc_tmp",tmpdir,256,&flg1);CHKERRQ(ierr);
    if (!flg1) {
      ierr = OptionsGetenv(comm,"PETSC_TMP",tmpdir,256,&flg1);CHKERRQ(ierr);
    }
    if (flg1) {
      ierr = PetscStrcat(par,"-tmp ");CHKERRQ(ierr);
      ierr = PetscStrcat(par,tmpdir);CHKERRQ(ierr);
      ierr = PetscStrcat(par," ");CHKERRQ(ierr);
    }

    ierr = PetscStrcat(par,libname);CHKERRQ(ierr);
    ierr = PetscStrcat(par," 2>&1 ");CHKERRQ(ierr);

    PLogInfo(0,"PetscFileRetrieve: Running python script:%s\n",par);
#if defined (PARCH_win32)
  SETERRQ(1,1,"Cannot use PetscFileRetrieve on NT");
#else 
    if (!(fp = popen(par,"r"))) {
      SETERRQ(1,1,"Cannot Execute python1.5 on ${PETSC_DIR}/bin/urlget.py\n\
        Check if python1.5 is in your path");
    }
    if (!fgets(buf,1024,fp)) {
      SETERRQ1(1,1,"No output from ${PETSC_DIR}/bin/urlget.py in getting file %s",libname);
    }
#endif
    /* Check for \n and make it 0 */
    for ( i=0; i<1024; i++ ) {
      if ( buf[i] == '\n') {
        buf[i] = 0;
        break;
      }
    }
    ierr = PetscStrncmp(buf,"Error",5,&flg1);CHKERRQ(ierr);
    ierr = PetscStrncmp(buf,"Traceback",9,&flg2);CHKERRQ(ierr);
    if (flg1 || flg2) {
      PLogInfo(0,"PetscFileRetrieve:Did not find file %s",libname);
    } else {
      *found = PETSC_TRUE;
    }
    ierr = PetscStrncpy(llibname,buf,llen);CHKERRQ(ierr);
    ierr = PetscFree(par);CHKERRQ(ierr);
  }
  if (sharedtmp) { /* send library name to all processors */
    ierr = MPI_Bcast(llibname,llen,MPI_CHAR,0,comm);CHKERRQ(ierr);
    ierr = MPI_Bcast(found,1,MPI_INT,0,comm);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}
