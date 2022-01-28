
/*
      Code for opening and closing files.
*/
#include <petscsys.h>
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

/*
   Private routine to delete tmp/shared storage

   This is called by MPI, not by users.

   Note: this is declared extern "C" because it is passed to MPI_Comm_create_keyval()

*/
PETSC_EXTERN PetscMPIInt MPIAPI Petsc_DelTmpShared(MPI_Comm comm,PetscMPIInt keyval,void *count_val,void *extra_state)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInfo(NULL,"Deleting tmp/shared data in an MPI_Comm %ld\n",(long)comm);CHKERRMPI(ierr);
  ierr = PetscFree(count_val);CHKERRMPI(ierr);
  PetscFunctionReturn(MPI_SUCCESS);
}

/*@C
   PetscGetTmp - Gets the name of the tmp directory

   Collective

   Input Parameters:
+  comm - MPI_Communicator that may share /tmp
-  len - length of string to hold name

   Output Parameter:
.  dir - directory name

   Options Database Keys:
+    -shared_tmp  - indicates the directory is shared among the MPI ranks
.    -not_shared_tmp - indicates the directory is not shared among the MPI ranks
-    -tmp tmpdir - name of the directory you wish to use as /tmp

   Environmental Variables:
+     PETSC_SHARED_TMP - indicates the directory is shared among the MPI ranks
.     PETSC_NOT_SHARED_TMP - indicates the directory is not shared among the MPI ranks
-     PETSC_TMP - name of the directory you wish to use as /tmp

   Level: developer

@*/
PetscErrorCode  PetscGetTmp(MPI_Comm comm,char dir[],size_t len)
{
  PetscErrorCode ierr;
  PetscBool      flg;

  PetscFunctionBegin;
  ierr = PetscOptionsGetenv(comm,"PETSC_TMP",dir,len,&flg);CHKERRQ(ierr);
  if (!flg) {
    ierr = PetscStrncpy(dir,"/tmp",len);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
   PetscSharedTmp - Determines if all processors in a communicator share a
         /tmp or have different ones.

   Collective

   Input Parameters:
.  comm - MPI_Communicator that may share /tmp

   Output Parameters:
.  shared - PETSC_TRUE or PETSC_FALSE

   Options Database Keys:
+    -shared_tmp  - indicates the directory is shared among the MPI ranks
.    -not_shared_tmp - indicates the directory is not shared among the MPI ranks
-    -tmp tmpdir - name of the directory you wish to use as /tmp

   Environmental Variables:
+     PETSC_SHARED_TMP  - indicates the directory is shared among the MPI ranks
.     PETSC_NOT_SHARED_TMP - indicates the directory is not shared among the MPI ranks
-     PETSC_TMP - name of the directory you wish to use as /tmp

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
PetscErrorCode  PetscSharedTmp(MPI_Comm comm,PetscBool  *shared)
{
  PetscErrorCode     ierr;
  PetscMPIInt        size,rank,*tagvalp,sum,cnt,i;
  PetscBool          flg,iflg;
  FILE               *fd;
  static PetscMPIInt Petsc_Tmp_keyval = MPI_KEYVAL_INVALID;
  int                err;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  if (size == 1) {
    *shared = PETSC_TRUE;
    PetscFunctionReturn(0);
  }

  ierr = PetscOptionsGetenv(comm,"PETSC_SHARED_TMP",NULL,0,&flg);CHKERRQ(ierr);
  if (flg) {
    *shared = PETSC_TRUE;
    PetscFunctionReturn(0);
  }

  ierr = PetscOptionsGetenv(comm,"PETSC_NOT_SHARED_TMP",NULL,0,&flg);CHKERRQ(ierr);
  if (flg) {
    *shared = PETSC_FALSE;
    PetscFunctionReturn(0);
  }

  if (Petsc_Tmp_keyval == MPI_KEYVAL_INVALID) {
    ierr = MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN,Petsc_DelTmpShared,&Petsc_Tmp_keyval,NULL);CHKERRMPI(ierr);
  }

  ierr = MPI_Comm_get_attr(comm,Petsc_Tmp_keyval,(void**)&tagvalp,(int*)&iflg);CHKERRMPI(ierr);
  if (!iflg) {
    char filename[PETSC_MAX_PATH_LEN],tmpname[PETSC_MAX_PATH_LEN];

    /* This communicator does not yet have a shared tmp attribute */
    ierr = PetscMalloc1(1,&tagvalp);CHKERRQ(ierr);
    ierr = MPI_Comm_set_attr(comm,Petsc_Tmp_keyval,tagvalp);CHKERRMPI(ierr);

    ierr = PetscOptionsGetenv(comm,"PETSC_TMP",tmpname,238,&iflg);CHKERRQ(ierr);
    if (!iflg) {
      ierr = PetscStrcpy(filename,"/tmp");CHKERRQ(ierr);
    } else {
      ierr = PetscStrcpy(filename,tmpname);CHKERRQ(ierr);
    }

    ierr = PetscStrcat(filename,"/petsctestshared");CHKERRQ(ierr);
    ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);

    /* each processor creates a /tmp file and all the later ones check */
    /* this makes sure no subset of processors is shared */
    *shared = PETSC_FALSE;
    for (i=0; i<size-1; i++) {
      if (rank == i) {
        fd = fopen(filename,"w");
        PetscAssertFalse(!fd,PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to open test file %s",filename);
        err = fclose(fd);
        PetscAssertFalse(err,PETSC_COMM_SELF,PETSC_ERR_SYS,"fclose() failed on file");
      }
      ierr = MPI_Barrier(comm);CHKERRMPI(ierr);
      if (rank >= i) {
        fd = fopen(filename,"r");
        if (fd) cnt = 1;
        else cnt = 0;
        if (fd) {
          err = fclose(fd);
          PetscAssertFalse(err,PETSC_COMM_SELF,PETSC_ERR_SYS,"fclose() failed on file");
        }
      } else cnt = 0;

      ierr = MPIU_Allreduce(&cnt,&sum,1,MPI_INT,MPI_SUM,comm);CHKERRMPI(ierr);
      if (rank == i) unlink(filename);

      if (sum == size) {
        *shared = PETSC_TRUE;
        break;
      } else PetscAssertFalse(sum != 1,PETSC_COMM_SELF,PETSC_ERR_SUP_SYS,"Subset of processes share /tmp ");
    }
    *tagvalp = (int)*shared;
    ierr = PetscInfo(NULL,"processors %s %s\n",(*shared) ? "share":"do NOT share",(iflg ? tmpname:"/tmp"));CHKERRQ(ierr);
  } else *shared = (PetscBool) *tagvalp;
  PetscFunctionReturn(0);
}

/*@C
   PetscSharedWorkingDirectory - Determines if all processors in a communicator share a
         working directory or have different ones.

   Collective

   Input Parameter:
.  comm - MPI_Communicator that may share working directory

   Output Parameter:
.  shared - PETSC_TRUE or PETSC_FALSE

   Options Database Keys:
+    -shared_working_directory - indicates the directory is shared among the MPI ranks
-    -not_shared_working_directory - indicates the directory is shared among the MPI ranks

   Environmental Variables:
+     PETSC_SHARED_WORKING_DIRECTORY - indicates the directory is shared among the MPI ranks
-     PETSC_NOT_SHARED_WORKING_DIRECTORY - indicates the directory is shared among the MPI ranks

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
PetscErrorCode  PetscSharedWorkingDirectory(MPI_Comm comm,PetscBool  *shared)
{
  PetscErrorCode     ierr;
  PetscMPIInt        size,rank,*tagvalp,sum,cnt,i;
  PetscBool          flg,iflg;
  FILE               *fd;
  static PetscMPIInt Petsc_WD_keyval = MPI_KEYVAL_INVALID;
  int                err;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  if (size == 1) {
    *shared = PETSC_TRUE;
    PetscFunctionReturn(0);
  }

  ierr = PetscOptionsGetenv(comm,"PETSC_SHARED_WORKING_DIRECTORY",NULL,0,&flg);CHKERRQ(ierr);
  if (flg) {
    *shared = PETSC_TRUE;
    PetscFunctionReturn(0);
  }

  ierr = PetscOptionsGetenv(comm,"PETSC_NOT_SHARED_WORKING_DIRECTORY",NULL,0,&flg);CHKERRQ(ierr);
  if (flg) {
    *shared = PETSC_FALSE;
    PetscFunctionReturn(0);
  }

  if (Petsc_WD_keyval == MPI_KEYVAL_INVALID) {
    ierr = MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN,Petsc_DelTmpShared,&Petsc_WD_keyval,NULL);CHKERRMPI(ierr);
  }

  ierr = MPI_Comm_get_attr(comm,Petsc_WD_keyval,(void**)&tagvalp,(int*)&iflg);CHKERRMPI(ierr);
  if (!iflg) {
    char filename[PETSC_MAX_PATH_LEN];

    /* This communicator does not yet have a shared  attribute */
    ierr = PetscMalloc1(1,&tagvalp);CHKERRQ(ierr);
    ierr = MPI_Comm_set_attr(comm,Petsc_WD_keyval,tagvalp);CHKERRMPI(ierr);

    ierr = PetscGetWorkingDirectory(filename,240);CHKERRQ(ierr);
    ierr = PetscStrcat(filename,"/petsctestshared");CHKERRQ(ierr);
    ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);

    /* each processor creates a  file and all the later ones check */
    /* this makes sure no subset of processors is shared */
    *shared = PETSC_FALSE;
    for (i=0; i<size-1; i++) {
      if (rank == i) {
        fd = fopen(filename,"w");
        PetscAssertFalse(!fd,PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to open test file %s",filename);
        err = fclose(fd);
        PetscAssertFalse(err,PETSC_COMM_SELF,PETSC_ERR_SYS,"fclose() failed on file");
      }
      ierr = MPI_Barrier(comm);CHKERRMPI(ierr);
      if (rank >= i) {
        fd = fopen(filename,"r");
        if (fd) cnt = 1;
        else cnt = 0;
        if (fd) {
          err = fclose(fd);
          PetscAssertFalse(err,PETSC_COMM_SELF,PETSC_ERR_SYS,"fclose() failed on file");
        }
      } else cnt = 0;

      ierr = MPIU_Allreduce(&cnt,&sum,1,MPI_INT,MPI_SUM,comm);CHKERRMPI(ierr);
      if (rank == i) unlink(filename);

      if (sum == size) {
        *shared = PETSC_TRUE;
        break;
      } else PetscAssertFalse(sum != 1,PETSC_COMM_SELF,PETSC_ERR_SUP_SYS,"Subset of processes share working directory");
    }
    *tagvalp = (int)*shared;
  } else *shared = (PetscBool) *tagvalp;
  ierr = PetscInfo(NULL,"processors %s working directory\n",(*shared) ? "shared" : "do NOT share");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
    PetscFileRetrieve - Obtains a file from a URL or compressed
        and copies into local disk space as uncompressed.

    Collective

    Input Parameters:
+   comm     - processors accessing the file
.   url      - name of file, including entire URL (with or without .gz)
-   llen     - length of localname

    Output Parameters:
+   localname - name of local copy of file - valid on only process zero
-   found - if found or retrieved the file - valid on all processes

    Notes:
    if the file already exists local this function just returns without downloading it.

    Level: intermediate
@*/
PetscErrorCode  PetscFileRetrieve(MPI_Comm comm,const char url[],char localname[],size_t llen,PetscBool  *found)
{
  char           buffer[PETSC_MAX_PATH_LEN],*par,*tlocalname,name[PETSC_MAX_PATH_LEN];
  FILE           *fp;
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  size_t         len = 0;
  PetscBool      flg1,flg2,flg3,flg4,download,compressed = PETSC_FALSE;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
  if (rank == 0) {
    *found = PETSC_FALSE;

    ierr = PetscStrstr(url,".gz",&par);CHKERRQ(ierr);
    if (par) {
      ierr = PetscStrlen(par,&len);CHKERRQ(ierr);
      if (len == 3) compressed = PETSC_TRUE;
    }

    ierr = PetscStrncmp(url,"ftp://",6,&flg1);CHKERRQ(ierr);
    ierr = PetscStrncmp(url,"http://",7,&flg2);CHKERRQ(ierr);
    ierr = PetscStrncmp(url,"file://",7,&flg3);CHKERRQ(ierr);
    ierr = PetscStrncmp(url,"https://",8,&flg4);CHKERRQ(ierr);
    download = (PetscBool) (flg1 || flg2 || flg3 || flg4);

    if (!download && !compressed) {
      ierr = PetscStrncpy(localname,url,llen);CHKERRQ(ierr);
      ierr = PetscTestFile(url,'r',found);CHKERRQ(ierr);
      if (*found) {
        ierr = PetscInfo(NULL,"Found file %s\n",url);CHKERRQ(ierr);
      } else {
        ierr = PetscInfo(NULL,"Did not find file %s\n",url);CHKERRQ(ierr);
      }
      goto done;
    }

    /* look for uncompressed file in requested directory */
    if (compressed) {
      ierr = PetscStrncpy(localname,url,llen);CHKERRQ(ierr);
      ierr = PetscStrstr(localname,".gz",&par);CHKERRQ(ierr);
      *par = 0; /* remove .gz extension */
      ierr = PetscTestFile(localname,'r',found);CHKERRQ(ierr);
      if (*found) goto done;
    }

    /* look for file in current directory */
    ierr = PetscStrrchr(url,'/',&tlocalname);CHKERRQ(ierr);
    ierr = PetscStrncpy(localname,tlocalname,llen);CHKERRQ(ierr);
    if (compressed) {
      ierr = PetscStrstr(localname,".gz",&par);CHKERRQ(ierr);
      *par = 0; /* remove .gz extension */
    }
    ierr = PetscTestFile(localname,'r',found);CHKERRQ(ierr);
    if (*found) goto done;

    if (download) {
      /* local file is not already here so use curl to get it */
      ierr = PetscStrncpy(localname,tlocalname,llen);CHKERRQ(ierr);
      ierr = PetscStrcpy(buffer,"curl --fail --silent --show-error ");CHKERRQ(ierr);
      ierr = PetscStrcat(buffer,url);CHKERRQ(ierr);
      ierr = PetscStrcat(buffer," > ");CHKERRQ(ierr);
      ierr = PetscStrcat(buffer,localname);CHKERRQ(ierr);
#if defined(PETSC_HAVE_POPEN)
      ierr = PetscPOpen(PETSC_COMM_SELF,NULL,buffer,"r",&fp);CHKERRQ(ierr);
      ierr = PetscPClose(PETSC_COMM_SELF,fp);CHKERRQ(ierr);
#else
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP_SYS,"Cannot run external programs on this machine");
#endif
      ierr = PetscTestFile(localname,'r',found);CHKERRQ(ierr);
      if (*found) {
        FILE      *fd;
        char      buf[1024],*str,*substring;

        /* check if the file didn't exist so it downloaded an HTML message instead */
        fd = fopen(localname,"r");
        PetscAssertFalse(!fd,PETSC_COMM_SELF,PETSC_ERR_PLIB,"PetscTestFile() indicates %s exists but fopen() cannot open it",localname);
        str = fgets(buf,sizeof(buf)-1,fd);
        while (str) {
          ierr = PetscStrstr(buf,"<!DOCTYPE html>",&substring);CHKERRQ(ierr);
          PetscAssertFalse(substring,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unable to download %s it does not appear to exist at this URL, dummy HTML file was downloaded",url);
          ierr = PetscStrstr(buf,"Not Found",&substring);CHKERRQ(ierr);
          PetscAssertFalse(substring,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unable to download %s it does not appear to exist at this URL, dummy HTML file was downloaded",url);
          str = fgets(buf,sizeof(buf)-1,fd);
        }
        fclose(fd);
      }
    } else if (compressed) {
      ierr = PetscTestFile(url,'r',found);CHKERRQ(ierr);
      if (!*found) goto done;
      ierr = PetscStrncpy(localname,url,llen);CHKERRQ(ierr);
    }
    if (compressed) {
      ierr = PetscStrrchr(localname,'/',&tlocalname);CHKERRQ(ierr);
      ierr = PetscStrncpy(name,tlocalname,PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
      ierr = PetscStrstr(name,".gz",&par);CHKERRQ(ierr);
      *par = 0; /* remove .gz extension */
      /* uncompress file */
      ierr = PetscStrcpy(buffer,"gzip -c -d ");CHKERRQ(ierr);
      ierr = PetscStrcat(buffer,localname);CHKERRQ(ierr);
      ierr = PetscStrcat(buffer," > ");CHKERRQ(ierr);
      ierr = PetscStrcat(buffer,name);CHKERRQ(ierr);
#if defined(PETSC_HAVE_POPEN)
      ierr = PetscPOpen(PETSC_COMM_SELF,NULL,buffer,"r",&fp);CHKERRQ(ierr);
      ierr = PetscPClose(PETSC_COMM_SELF,fp);CHKERRQ(ierr);
#else
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP_SYS,"Cannot run external programs on this machine");
#endif
      ierr = PetscStrncpy(localname,name,llen);CHKERRQ(ierr);
      ierr = PetscTestFile(localname,'r',found);CHKERRQ(ierr);
    }
  }
  done:
  ierr = MPI_Bcast(found,1,MPIU_BOOL,0,comm);CHKERRMPI(ierr);
  ierr = MPI_Bcast(localname, llen, MPI_CHAR, 0, comm);CHKERRMPI(ierr);
  PetscFunctionReturn(0);
}
