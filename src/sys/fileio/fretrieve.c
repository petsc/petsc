#define PETSC_DLL
/*
      Code for opening and closing files.
*/
#include "petscsys.h"
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

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "Petsc_DelTmpShared"
/*
   Private routine to delete tmp/shared storage

   This is called by MPI, not by users.

   Note: this is declared extern "C" because it is passed to MPI_Keyval_create()

*/
PetscMPIInt PETSC_DLLEXPORT MPIAPI Petsc_DelTmpShared(MPI_Comm comm,PetscMPIInt keyval,void *count_val,void *extra_state)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInfo1(0,"Deleting tmp/shared data in an MPI_Comm %ld\n",(long)comm);if (ierr) PetscFunctionReturn((PetscMPIInt)ierr);
  ierr = PetscFree(count_val);if (ierr) PetscFunctionReturn((PetscMPIInt)ierr);
  PetscFunctionReturn(MPI_SUCCESS);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "PetscGetTmp"
/*@C
   PetscGetTmp - Gets the name of the tmp directory

   Collective on MPI_Comm

   Input Parameters:
+  comm - MPI_Communicator that may share /tmp
-  len - length of string to hold name

   Output Parameters:
.  dir - directory name

   Options Database Keys:
+    -shared_tmp 
.    -not_shared_tmp
-    -tmp tmpdir

   Environmental Variables:
+     PETSC_SHARED_TMP
.     PETSC_NOT_SHARED_TMP
-     PETSC_TMP

   Level: developer

   
   If the environmental variable PETSC_TMP is set it will use this directory
  as the "/tmp" directory.

@*/
PetscErrorCode PETSC_DLLEXPORT PetscGetTmp(MPI_Comm comm,char dir[],size_t len)
{
  PetscErrorCode ierr;
  PetscTruth     flg;

  PetscFunctionBegin;
  ierr = PetscOptionsGetenv(comm,"PETSC_TMP",dir,len,&flg);CHKERRQ(ierr);
  if (!flg) {
    ierr = PetscStrncpy(dir,"/tmp",len);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscSharedTmp"
/*@C
   PetscSharedTmp - Determines if all processors in a communicator share a
         /tmp or have different ones.

   Collective on MPI_Comm

   Input Parameters:
.  comm - MPI_Communicator that may share /tmp

   Output Parameters:
.  shared - PETSC_TRUE or PETSC_FALSE

   Options Database Keys:
+    -shared_tmp 
.    -not_shared_tmp
-    -tmp tmpdir

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
       2) each has a separate /tmp
      eventually we can write a fancier one that determines which processors
      share a common /tmp.

   This will be very slow on runs with a large number of processors since
   it requires O(p*p) file opens.

   If the environmental variable PETSC_TMP is set it will use this directory
  as the "/tmp" directory.

@*/
PetscErrorCode PETSC_DLLEXPORT PetscSharedTmp(MPI_Comm comm,PetscTruth *shared)
{
  PetscErrorCode     ierr;
  PetscMPIInt        size,rank,*tagvalp,sum,cnt,i;
  PetscTruth         flg,iflg;
  FILE               *fd;
  static PetscMPIInt Petsc_Tmp_keyval = MPI_KEYVAL_INVALID;
  int                err;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size == 1) {
    *shared = PETSC_TRUE;
    PetscFunctionReturn(0);
  }

  ierr = PetscOptionsGetenv(comm,"PETSC_SHARED_TMP",PETSC_NULL,0,&flg);CHKERRQ(ierr);
  if (flg) {
    *shared = PETSC_TRUE;
    PetscFunctionReturn(0);
  }

  ierr = PetscOptionsGetenv(comm,"PETSC_NOT_SHARED_TMP",PETSC_NULL,0,&flg);CHKERRQ(ierr);
  if (flg) {
    *shared = PETSC_FALSE;
    PetscFunctionReturn(0);
  }

  if (Petsc_Tmp_keyval == MPI_KEYVAL_INVALID) {
    ierr = MPI_Keyval_create(MPI_NULL_COPY_FN,Petsc_DelTmpShared,&Petsc_Tmp_keyval,0);CHKERRQ(ierr);
  }

  ierr = MPI_Attr_get(comm,Petsc_Tmp_keyval,(void**)&tagvalp,(int*)&iflg);CHKERRQ(ierr);
  if (!iflg) {
    char       filename[PETSC_MAX_PATH_LEN],tmpname[PETSC_MAX_PATH_LEN];

    /* This communicator does not yet have a shared tmp attribute */
    ierr = PetscMalloc(sizeof(PetscMPIInt),&tagvalp);CHKERRQ(ierr);
    ierr = MPI_Attr_put(comm,Petsc_Tmp_keyval,tagvalp);CHKERRQ(ierr);

    ierr = PetscOptionsGetenv(comm,"PETSC_TMP",tmpname,238,&iflg);CHKERRQ(ierr);
    if (!iflg) {
      ierr = PetscStrcpy(filename,"/tmp");CHKERRQ(ierr);
    } else {
      ierr = PetscStrcpy(filename,tmpname);CHKERRQ(ierr);
    }

    ierr = PetscStrcat(filename,"/petsctestshared");CHKERRQ(ierr);
    ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
    
    /* each processor creates a /tmp file and all the later ones check */
    /* this makes sure no subset of processors is shared */
    *shared = PETSC_FALSE;
    for (i=0; i<size-1; i++) {
      if (rank == i) {
        fd = fopen(filename,"w");
        if (!fd) {
          SETERRQ1(PETSC_ERR_FILE_OPEN,"Unable to open test file %s",filename);
        }
        err = fclose(fd);
        if (err) SETERRQ(PETSC_ERR_SYS,"fclose() failed on file");    
      }
      ierr = MPI_Barrier(comm);CHKERRQ(ierr);
      if (rank >= i) {
        fd = fopen(filename,"r");
        if (fd) cnt = 1; else cnt = 0;
        if (fd) {
          err = fclose(fd);
          if (err) SETERRQ(PETSC_ERR_SYS,"fclose() failed on file");    
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
        SETERRQ(PETSC_ERR_SUP_SYS,"Subset of processes share /tmp ");
      }
    }
    *tagvalp = (int)*shared;
    ierr = PetscInfo2(0,"processors %s %s\n",(*shared) ? "share":"do NOT share",(iflg ? tmpname:"/tmp"));CHKERRQ(ierr);
  } else {
    *shared = (PetscTruth) *tagvalp;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscSharedWorkingDirectory"
/*@C
   PetscSharedWorkingDirectory - Determines if all processors in a communicator share a
         working directory or have different ones.

   Collective on MPI_Comm

   Input Parameters:
.  comm - MPI_Communicator that may share working directory

   Output Parameters:
.  shared - PETSC_TRUE or PETSC_FALSE

   Options Database Keys:
+    -shared_working_directory 
.    -not_shared_working_directory

   Environmental Variables:
+     PETSC_SHARED_WORKING_DIRECTORY
.     PETSC_NOT_SHARED_WORKING_DIRECTORY

   Level: developer

   Notes:
   Stores the status as a MPI attribute so it does not have
    to be redetermined each time.

      Assumes that all processors in a communicator either
       1) have a common working directory or
       2) each has a separate working directory
      eventually we can write a fancier one that determines which processors
      share a common working directory.

   This will be very slow on runs with a large number of processors since
   it requires O(p*p) file opens.

@*/
PetscErrorCode PETSC_DLLEXPORT PetscSharedWorkingDirectory(MPI_Comm comm,PetscTruth *shared)
{
  PetscErrorCode     ierr;
  PetscMPIInt        size,rank,*tagvalp,sum,cnt,i;
  PetscTruth         flg,iflg;
  FILE               *fd;
  static PetscMPIInt Petsc_WD_keyval = MPI_KEYVAL_INVALID;
  int                err;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size == 1) {
    *shared = PETSC_TRUE;
    PetscFunctionReturn(0);
  }

  ierr = PetscOptionsGetenv(comm,"PETSC_SHARED_WORKING_DIRECTORY",PETSC_NULL,0,&flg);CHKERRQ(ierr);
  if (flg) {
    *shared = PETSC_TRUE;
    PetscFunctionReturn(0);
  }

  ierr = PetscOptionsGetenv(comm,"PETSC_NOT_SHARED_WORKING_DIRECTORY",PETSC_NULL,0,&flg);CHKERRQ(ierr);
  if (flg) {
    *shared = PETSC_FALSE;
    PetscFunctionReturn(0);
  }

  if (Petsc_WD_keyval == MPI_KEYVAL_INVALID) {
    ierr = MPI_Keyval_create(MPI_NULL_COPY_FN,Petsc_DelTmpShared,&Petsc_WD_keyval,0);CHKERRQ(ierr);
  }

  ierr = MPI_Attr_get(comm,Petsc_WD_keyval,(void**)&tagvalp,(int*)&iflg);CHKERRQ(ierr);
  if (!iflg) {
    char       filename[PETSC_MAX_PATH_LEN];

    /* This communicator does not yet have a shared  attribute */
    ierr = PetscMalloc(sizeof(PetscMPIInt),&tagvalp);CHKERRQ(ierr);
    ierr = MPI_Attr_put(comm,Petsc_WD_keyval,tagvalp);CHKERRQ(ierr);

    ierr = PetscGetWorkingDirectory(filename,240);CHKERRQ(ierr);
    ierr = PetscStrcat(filename,"/petsctestshared");CHKERRQ(ierr);
    ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
    
    /* each processor creates a  file and all the later ones check */
    /* this makes sure no subset of processors is shared */
    *shared = PETSC_FALSE;
    for (i=0; i<size-1; i++) {
      if (rank == i) {
        fd = fopen(filename,"w");
        if (!fd) SETERRQ1(PETSC_ERR_FILE_OPEN,"Unable to open test file %s",filename);
        err = fclose(fd);
        if (err) SETERRQ(PETSC_ERR_SYS,"fclose() failed on file");    
      }
      ierr = MPI_Barrier(comm);CHKERRQ(ierr);
      if (rank >= i) {
        fd = fopen(filename,"r");
        if (fd) cnt = 1; else cnt = 0;
        if (fd) {
          err = fclose(fd);
          if (err) SETERRQ(PETSC_ERR_SYS,"fclose() failed on file");    
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
        SETERRQ(PETSC_ERR_SUP_SYS,"Subset of processes share working directory");
      }
    }
    *tagvalp = (int)*shared;
  } else {
    *shared = (PetscTruth) *tagvalp;
  }
  ierr = PetscInfo1(0,"processors %s working directory\n",(*shared) ? "shared" : "do NOT share");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "PetscFileRetrieve"
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
-   found - if found and retrieved the file

    Level: developer

@*/
PetscErrorCode PETSC_DLLEXPORT PetscFileRetrieve(MPI_Comm comm,const char libname[],char llibname[],size_t llen,PetscTruth *found)
{
  char              buf[1024],tmpdir[PETSC_MAX_PATH_LEN],urlget[PETSC_MAX_PATH_LEN],*par;
  const char        *pdir;
  FILE              *fp;
  PetscErrorCode    ierr;
  int               i;
  PetscMPIInt       rank;
  size_t            len = 0;
  PetscTruth        flg1,flg2,flg3,sharedtmp,exists;

  PetscFunctionBegin;
  *found = PETSC_FALSE;

  /* if file does not have an ftp:// or http:// or .gz then need not process file */
  ierr = PetscStrstr(libname,".gz",&par);CHKERRQ(ierr);
  if (par) {ierr = PetscStrlen(par,&len);CHKERRQ(ierr);}

  ierr = PetscStrncmp(libname,"ftp://",6,&flg1);CHKERRQ(ierr);
  ierr = PetscStrncmp(libname,"http://",7,&flg2);CHKERRQ(ierr);
  ierr = PetscStrncmp(libname,"file://",7,&flg3);CHKERRQ(ierr);
  if (!flg1 && !flg2 && !flg3 && (!par || len != 3)) {
    ierr = PetscStrncpy(llibname,libname,llen);CHKERRQ(ierr);
    ierr = PetscTestFile(libname,'r',found);CHKERRQ(ierr);
    if (*found) {
      ierr = PetscInfo1(PETSC_NULL,"Found file %s\n",libname);
    } else {
      ierr = PetscInfo1(PETSC_NULL,"Did not find file %s\n",libname);
    }
    PetscFunctionReturn(0);
  }

  /* Determine if all processors share a common /tmp */
  ierr = PetscSharedTmp(comm,&sharedtmp);CHKERRQ(ierr);
  ierr = PetscOptionsGetenv(comm,"PETSC_TMP",tmpdir,PETSC_MAX_PATH_LEN,&flg1);CHKERRQ(ierr);

  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank || !sharedtmp) {
  
    /* Construct the script to get URL file */
    ierr = PetscGetPetscDir(&pdir);CHKERRQ(ierr);
    ierr = PetscStrcpy(urlget,pdir);CHKERRQ(ierr);
    ierr = PetscStrcat(urlget,"/bin/urlget");CHKERRQ(ierr);
    ierr = PetscTestFile(urlget,'r',&exists);CHKERRQ(ierr);
    if (!exists) {
      ierr = PetscTestFile("urlget",'r',&exists);CHKERRQ(ierr);
      if (!exists) {
        SETERRQ1(PETSC_ERR_PLIB,"Cannot locate PETSc script urlget in %s or current directory",urlget);
      }
      ierr = PetscStrcpy(urlget,"urlget");CHKERRQ(ierr);
    }
    ierr = PetscStrcat(urlget," ");CHKERRQ(ierr);

    /* are we using an alternative /tmp? */
    if (flg1) {
      ierr = PetscStrcat(urlget,"-tmp ");CHKERRQ(ierr);
      ierr = PetscStrcat(urlget,tmpdir);CHKERRQ(ierr);
      ierr = PetscStrcat(urlget," ");CHKERRQ(ierr);
    }

    ierr = PetscStrcat(urlget,libname);CHKERRQ(ierr);
    ierr = PetscStrcat(urlget," 2>&1 ");CHKERRQ(ierr);

#if defined(PETSC_HAVE_POPEN)
    ierr = PetscPOpen(PETSC_COMM_SELF,PETSC_NULL,urlget,"r",&fp);CHKERRQ(ierr);
#else
    SETERRQ(PETSC_ERR_SUP_SYS,"Cannot run external programs on this machine");
#endif
    if (!fgets(buf,1024,fp)) {
      SETERRQ1(PETSC_ERR_PLIB,"No output from ${PETSC_DIR}/bin/urlget in getting file %s",libname);
    }
    ierr = PetscInfo1(0,"Message back from urlget: %s\n",buf);CHKERRQ(ierr);

    ierr = PetscStrncmp(buf,"Error",5,&flg1);CHKERRQ(ierr);
    ierr = PetscStrncmp(buf,"Traceback",9,&flg2);CHKERRQ(ierr);
#if defined(PETSC_HAVE_POPEN)
    ierr = PetscPClose(PETSC_COMM_SELF,fp);CHKERRQ(ierr);
#endif
    if (flg1 || flg2) {
      *found = PETSC_FALSE;
    } else {
      *found = PETSC_TRUE;
  
      /* Check for \n and make it 0 */
      for (i=0; i<1024; i++) {
        if (buf[i] == '\n') {
          buf[i] = 0;
          break;
        }
      }
      ierr = PetscStrncpy(llibname,buf,llen);CHKERRQ(ierr);
    }
  }
  if (sharedtmp) { /* send library name to all processors */
    ierr = MPI_Bcast(found,1,MPI_INT,0,comm);CHKERRQ(ierr);
    if (*found) {
      ierr = MPI_Bcast(llibname,llen,MPI_CHAR,0,comm);CHKERRQ(ierr);
      ierr = MPI_Bcast(found,1,MPI_INT,0,comm);CHKERRQ(ierr);
    }
  }

  PetscFunctionReturn(0);
}
