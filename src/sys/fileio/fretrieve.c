
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
  PetscFunctionBegin;
  CHKERRMPI(PetscInfo(NULL,"Deleting tmp/shared data in an MPI_Comm %ld\n",(long)comm));
  CHKERRMPI(PetscFree(count_val));
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
  PetscBool      flg;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsGetenv(comm,"PETSC_TMP",dir,len,&flg));
  if (!flg) {
    CHKERRQ(PetscStrncpy(dir,"/tmp",len));
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
  PetscMPIInt        size,rank,*tagvalp,sum,cnt,i;
  PetscBool          flg,iflg;
  FILE               *fd;
  static PetscMPIInt Petsc_Tmp_keyval = MPI_KEYVAL_INVALID;
  int                err;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_size(comm,&size));
  if (size == 1) {
    *shared = PETSC_TRUE;
    PetscFunctionReturn(0);
  }

  CHKERRQ(PetscOptionsGetenv(comm,"PETSC_SHARED_TMP",NULL,0,&flg));
  if (flg) {
    *shared = PETSC_TRUE;
    PetscFunctionReturn(0);
  }

  CHKERRQ(PetscOptionsGetenv(comm,"PETSC_NOT_SHARED_TMP",NULL,0,&flg));
  if (flg) {
    *shared = PETSC_FALSE;
    PetscFunctionReturn(0);
  }

  if (Petsc_Tmp_keyval == MPI_KEYVAL_INVALID) {
    CHKERRMPI(MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN,Petsc_DelTmpShared,&Petsc_Tmp_keyval,NULL));
  }

  CHKERRMPI(MPI_Comm_get_attr(comm,Petsc_Tmp_keyval,(void**)&tagvalp,(int*)&iflg));
  if (!iflg) {
    char filename[PETSC_MAX_PATH_LEN],tmpname[PETSC_MAX_PATH_LEN];

    /* This communicator does not yet have a shared tmp attribute */
    CHKERRQ(PetscMalloc1(1,&tagvalp));
    CHKERRMPI(MPI_Comm_set_attr(comm,Petsc_Tmp_keyval,tagvalp));

    CHKERRQ(PetscOptionsGetenv(comm,"PETSC_TMP",tmpname,238,&iflg));
    if (!iflg) {
      CHKERRQ(PetscStrcpy(filename,"/tmp"));
    } else {
      CHKERRQ(PetscStrcpy(filename,tmpname));
    }

    CHKERRQ(PetscStrcat(filename,"/petsctestshared"));
    CHKERRMPI(MPI_Comm_rank(comm,&rank));

    /* each processor creates a /tmp file and all the later ones check */
    /* this makes sure no subset of processors is shared */
    *shared = PETSC_FALSE;
    for (i=0; i<size-1; i++) {
      if (rank == i) {
        fd = fopen(filename,"w");
        PetscCheckFalse(!fd,PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to open test file %s",filename);
        err = fclose(fd);
        PetscCheckFalse(err,PETSC_COMM_SELF,PETSC_ERR_SYS,"fclose() failed on file");
      }
      CHKERRMPI(MPI_Barrier(comm));
      if (rank >= i) {
        fd = fopen(filename,"r");
        if (fd) cnt = 1;
        else cnt = 0;
        if (fd) {
          err = fclose(fd);
          PetscCheckFalse(err,PETSC_COMM_SELF,PETSC_ERR_SYS,"fclose() failed on file");
        }
      } else cnt = 0;

      CHKERRMPI(MPIU_Allreduce(&cnt,&sum,1,MPI_INT,MPI_SUM,comm));
      if (rank == i) unlink(filename);

      if (sum == size) {
        *shared = PETSC_TRUE;
        break;
      } else PetscCheckFalse(sum != 1,PETSC_COMM_SELF,PETSC_ERR_SUP_SYS,"Subset of processes share /tmp ");
    }
    *tagvalp = (int)*shared;
    CHKERRQ(PetscInfo(NULL,"processors %s %s\n",(*shared) ? "share":"do NOT share",(iflg ? tmpname:"/tmp")));
  } else *shared = (PetscBool) *tagvalp;
  PetscFunctionReturn(0);
}

/*@C
  PetscSharedWorkingDirectory - Determines if all processors in a communicator share a working directory or have different ones.

  Collective

  Input Parameter:
. comm - MPI_Communicator that may share working directory

  Output Parameter:
. shared - PETSC_TRUE or PETSC_FALSE

  Options Database Keys:
+ -shared_working_directory - indicates the directory is shared among the MPI ranks
- -not_shared_working_directory - indicates the directory is shared among the MPI ranks

  Environmental Variables:
+ PETSC_SHARED_WORKING_DIRECTORY - indicates the directory is shared among the MPI ranks
- PETSC_NOT_SHARED_WORKING_DIRECTORY - indicates the directory is shared among the MPI ranks

  Level: developer

  Notes:
  Stores the status as a MPI attribute so it does not have to be redetermined each time.

  Assumes that all processors in a communicator either
$   1) have a common working directory or
$   2) each has a separate working directory
  eventually we can write a fancier one that determines which processors share a common working directory.

  This will be very slow on runs with a large number of processors since it requires O(p*p) file opens.
@*/
PetscErrorCode PetscSharedWorkingDirectory(MPI_Comm comm, PetscBool *shared)
{
  PetscMPIInt        size,rank,*tagvalp,sum,cnt,i;
  PetscBool          flg,iflg;
  FILE               *fd;
  static PetscMPIInt Petsc_WD_keyval = MPI_KEYVAL_INVALID;
  int                err;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_size(comm,&size));
  if (size == 1) {
    *shared = PETSC_TRUE;
    PetscFunctionReturn(0);
  }

  CHKERRQ(PetscOptionsGetenv(comm,"PETSC_SHARED_WORKING_DIRECTORY",NULL,0,&flg));
  if (flg) {
    *shared = PETSC_TRUE;
    PetscFunctionReturn(0);
  }

  CHKERRQ(PetscOptionsGetenv(comm,"PETSC_NOT_SHARED_WORKING_DIRECTORY",NULL,0,&flg));
  if (flg) {
    *shared = PETSC_FALSE;
    PetscFunctionReturn(0);
  }

  if (Petsc_WD_keyval == MPI_KEYVAL_INVALID) {
    CHKERRMPI(MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN,Petsc_DelTmpShared,&Petsc_WD_keyval,NULL));
  }

  CHKERRMPI(MPI_Comm_get_attr(comm,Petsc_WD_keyval,(void**)&tagvalp,(int*)&iflg));
  if (!iflg) {
    char filename[PETSC_MAX_PATH_LEN];

    /* This communicator does not yet have a shared  attribute */
    CHKERRQ(PetscMalloc1(1,&tagvalp));
    CHKERRMPI(MPI_Comm_set_attr(comm,Petsc_WD_keyval,tagvalp));

    CHKERRQ(PetscGetWorkingDirectory(filename,240));
    CHKERRQ(PetscStrcat(filename,"/petsctestshared"));
    CHKERRMPI(MPI_Comm_rank(comm,&rank));

    /* each processor creates a  file and all the later ones check */
    /* this makes sure no subset of processors is shared */
    *shared = PETSC_FALSE;
    for (i=0; i<size-1; i++) {
      if (rank == i) {
        fd = fopen(filename,"w");
        PetscCheckFalse(!fd,PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to open test file %s",filename);
        err = fclose(fd);
        PetscCheckFalse(err,PETSC_COMM_SELF,PETSC_ERR_SYS,"fclose() failed on file");
      }
      CHKERRMPI(MPI_Barrier(comm));
      if (rank >= i) {
        fd = fopen(filename,"r");
        if (fd) cnt = 1;
        else cnt = 0;
        if (fd) {
          err = fclose(fd);
          PetscCheckFalse(err,PETSC_COMM_SELF,PETSC_ERR_SYS,"fclose() failed on file");
        }
      } else cnt = 0;

      CHKERRMPI(MPIU_Allreduce(&cnt,&sum,1,MPI_INT,MPI_SUM,comm));
      if (rank == i) unlink(filename);

      if (sum == size) {
        *shared = PETSC_TRUE;
        break;
      } else PetscCheckFalse(sum != 1,PETSC_COMM_SELF,PETSC_ERR_SUP_SYS,"Subset of processes share working directory");
    }
    *tagvalp = (int)*shared;
  } else *shared = (PetscBool) *tagvalp;
  CHKERRQ(PetscInfo(NULL,"processors %s working directory\n",(*shared) ? "shared" : "do NOT share"));
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
  PetscMPIInt    rank;
  size_t         len = 0;
  PetscBool      flg1,flg2,flg3,flg4,download,compressed = PETSC_FALSE;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_rank(comm,&rank));
  if (rank == 0) {
    *found = PETSC_FALSE;

    CHKERRQ(PetscStrstr(url,".gz",&par));
    if (par) {
      CHKERRQ(PetscStrlen(par,&len));
      if (len == 3) compressed = PETSC_TRUE;
    }

    CHKERRQ(PetscStrncmp(url,"ftp://",6,&flg1));
    CHKERRQ(PetscStrncmp(url,"http://",7,&flg2));
    CHKERRQ(PetscStrncmp(url,"file://",7,&flg3));
    CHKERRQ(PetscStrncmp(url,"https://",8,&flg4));
    download = (PetscBool) (flg1 || flg2 || flg3 || flg4);

    if (!download && !compressed) {
      CHKERRQ(PetscStrncpy(localname,url,llen));
      CHKERRQ(PetscTestFile(url,'r',found));
      if (*found) {
        CHKERRQ(PetscInfo(NULL,"Found file %s\n",url));
      } else {
        CHKERRQ(PetscInfo(NULL,"Did not find file %s\n",url));
      }
      goto done;
    }

    /* look for uncompressed file in requested directory */
    if (compressed) {
      CHKERRQ(PetscStrncpy(localname,url,llen));
      CHKERRQ(PetscStrstr(localname,".gz",&par));
      *par = 0; /* remove .gz extension */
      CHKERRQ(PetscTestFile(localname,'r',found));
      if (*found) goto done;
    }

    /* look for file in current directory */
    CHKERRQ(PetscStrrchr(url,'/',&tlocalname));
    CHKERRQ(PetscStrncpy(localname,tlocalname,llen));
    if (compressed) {
      CHKERRQ(PetscStrstr(localname,".gz",&par));
      *par = 0; /* remove .gz extension */
    }
    CHKERRQ(PetscTestFile(localname,'r',found));
    if (*found) goto done;

    if (download) {
      /* local file is not already here so use curl to get it */
      CHKERRQ(PetscStrncpy(localname,tlocalname,llen));
      CHKERRQ(PetscStrcpy(buffer,"curl --fail --silent --show-error "));
      CHKERRQ(PetscStrcat(buffer,url));
      CHKERRQ(PetscStrcat(buffer," > "));
      CHKERRQ(PetscStrcat(buffer,localname));
#if defined(PETSC_HAVE_POPEN)
      CHKERRQ(PetscPOpen(PETSC_COMM_SELF,NULL,buffer,"r",&fp));
      CHKERRQ(PetscPClose(PETSC_COMM_SELF,fp));
#else
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP_SYS,"Cannot run external programs on this machine");
#endif
      CHKERRQ(PetscTestFile(localname,'r',found));
      if (*found) {
        FILE      *fd;
        char      buf[1024],*str,*substring;

        /* check if the file didn't exist so it downloaded an HTML message instead */
        fd = fopen(localname,"r");
        PetscCheckFalse(!fd,PETSC_COMM_SELF,PETSC_ERR_PLIB,"PetscTestFile() indicates %s exists but fopen() cannot open it",localname);
        str = fgets(buf,sizeof(buf)-1,fd);
        while (str) {
          CHKERRQ(PetscStrstr(buf,"<!DOCTYPE html>",&substring));
          PetscCheckFalse(substring,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unable to download %s it does not appear to exist at this URL, dummy HTML file was downloaded",url);
          CHKERRQ(PetscStrstr(buf,"Not Found",&substring));
          PetscCheckFalse(substring,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unable to download %s it does not appear to exist at this URL, dummy HTML file was downloaded",url);
          str = fgets(buf,sizeof(buf)-1,fd);
        }
        fclose(fd);
      }
    } else if (compressed) {
      CHKERRQ(PetscTestFile(url,'r',found));
      if (!*found) goto done;
      CHKERRQ(PetscStrncpy(localname,url,llen));
    }
    if (compressed) {
      CHKERRQ(PetscStrrchr(localname,'/',&tlocalname));
      CHKERRQ(PetscStrncpy(name,tlocalname,PETSC_MAX_PATH_LEN));
      CHKERRQ(PetscStrstr(name,".gz",&par));
      *par = 0; /* remove .gz extension */
      /* uncompress file */
      CHKERRQ(PetscStrcpy(buffer,"gzip -c -d "));
      CHKERRQ(PetscStrcat(buffer,localname));
      CHKERRQ(PetscStrcat(buffer," > "));
      CHKERRQ(PetscStrcat(buffer,name));
#if defined(PETSC_HAVE_POPEN)
      CHKERRQ(PetscPOpen(PETSC_COMM_SELF,NULL,buffer,"r",&fp));
      CHKERRQ(PetscPClose(PETSC_COMM_SELF,fp));
#else
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP_SYS,"Cannot run external programs on this machine");
#endif
      CHKERRQ(PetscStrncpy(localname,name,llen));
      CHKERRQ(PetscTestFile(localname,'r',found));
    }
  }
  done:
  CHKERRMPI(MPI_Bcast(found,1,MPIU_BOOL,0,comm));
  CHKERRMPI(MPI_Bcast(localname, llen, MPI_CHAR, 0, comm));
  PetscFunctionReturn(0);
}
