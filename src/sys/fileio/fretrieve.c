
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

   Note: this is declared extern "C" because it is passed to MPI_Keyval_create()

*/
PETSC_EXTERN PetscMPIInt MPIAPI Petsc_DelTmpShared(MPI_Comm comm,PetscMPIInt keyval,void *count_val,void *extra_state)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInfo1(0,"Deleting tmp/shared data in an MPI_Comm %ld\n",(long)comm);if (ierr) PetscFunctionReturn((PetscMPIInt)ierr);
  ierr = PetscFree(count_val);if (ierr) PetscFunctionReturn((PetscMPIInt)ierr);
  PetscFunctionReturn(MPI_SUCCESS);
}

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
PetscErrorCode  PetscSharedTmp(MPI_Comm comm,PetscBool  *shared)
{
  PetscErrorCode     ierr;
  PetscMPIInt        size,rank,*tagvalp,sum,cnt,i;
  PetscBool          flg,iflg;
  FILE               *fd;
  static PetscMPIInt Petsc_Tmp_keyval = MPI_KEYVAL_INVALID;
  int                err;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
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
    ierr = MPI_Keyval_create(MPI_NULL_COPY_FN,Petsc_DelTmpShared,&Petsc_Tmp_keyval,0);CHKERRQ(ierr);
  }

  ierr = MPI_Attr_get(comm,Petsc_Tmp_keyval,(void**)&tagvalp,(int*)&iflg);CHKERRQ(ierr);
  if (!iflg) {
    char filename[PETSC_MAX_PATH_LEN],tmpname[PETSC_MAX_PATH_LEN];

    /* This communicator does not yet have a shared tmp attribute */
    ierr = PetscMalloc1(1,&tagvalp);CHKERRQ(ierr);
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
        if (!fd) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to open test file %s",filename);
        err = fclose(fd);
        if (err) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SYS,"fclose() failed on file");
      }
      ierr = MPI_Barrier(comm);CHKERRQ(ierr);
      if (rank >= i) {
        fd = fopen(filename,"r");
        if (fd) cnt = 1;
        else cnt = 0;
        if (fd) {
          err = fclose(fd);
          if (err) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SYS,"fclose() failed on file");
        }
      } else cnt = 0;

      ierr = MPIU_Allreduce(&cnt,&sum,1,MPI_INT,MPI_SUM,comm);CHKERRQ(ierr);
      if (rank == i) unlink(filename);

      if (sum == size) {
        *shared = PETSC_TRUE;
        break;
      } else if (sum != 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP_SYS,"Subset of processes share /tmp ");
    }
    *tagvalp = (int)*shared;
    ierr = PetscInfo2(0,"processors %s %s\n",(*shared) ? "share":"do NOT share",(iflg ? tmpname:"/tmp"));CHKERRQ(ierr);
  } else *shared = (PetscBool) *tagvalp;
  PetscFunctionReturn(0);
}

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
PetscErrorCode  PetscSharedWorkingDirectory(MPI_Comm comm,PetscBool  *shared)
{
  PetscErrorCode     ierr;
  PetscMPIInt        size,rank,*tagvalp,sum,cnt,i;
  PetscBool          flg,iflg;
  FILE               *fd;
  static PetscMPIInt Petsc_WD_keyval = MPI_KEYVAL_INVALID;
  int                err;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
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
    ierr = MPI_Keyval_create(MPI_NULL_COPY_FN,Petsc_DelTmpShared,&Petsc_WD_keyval,0);CHKERRQ(ierr);
  }

  ierr = MPI_Attr_get(comm,Petsc_WD_keyval,(void**)&tagvalp,(int*)&iflg);CHKERRQ(ierr);
  if (!iflg) {
    char filename[PETSC_MAX_PATH_LEN];

    /* This communicator does not yet have a shared  attribute */
    ierr = PetscMalloc1(1,&tagvalp);CHKERRQ(ierr);
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
        if (!fd) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to open test file %s",filename);
        err = fclose(fd);
        if (err) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SYS,"fclose() failed on file");
      }
      ierr = MPI_Barrier(comm);CHKERRQ(ierr);
      if (rank >= i) {
        fd = fopen(filename,"r");
        if (fd) cnt = 1;
        else cnt = 0;
        if (fd) {
          err = fclose(fd);
          if (err) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SYS,"fclose() failed on file");
        }
      } else cnt = 0;

      ierr = MPIU_Allreduce(&cnt,&sum,1,MPI_INT,MPI_SUM,comm);CHKERRQ(ierr);
      if (rank == i) unlink(filename);

      if (sum == size) {
        *shared = PETSC_TRUE;
        break;
      } else if (sum != 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP_SYS,"Subset of processes share working directory");
    }
    *tagvalp = (int)*shared;
  } else *shared = (PetscBool) *tagvalp;
  ierr = PetscInfo1(0,"processors %s working directory\n",(*shared) ? "shared" : "do NOT share");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*@C
    PetscFileRetrieve - Obtains a file from a URL or compressed
        and copies into local disk space as uncompressed.

    Collective on MPI_Comm

    Input Parameter:
+   comm     - processors accessing the file
.   url      - name of file, including entire URL (with or without .gz)
-   llen     - length of localname

    Output Parameter:
+   localname - name of local copy of file
-   found - if found and retrieved the file

    Notes: if the file already exists local this function just returns without downloading it.

    Level: intermediate
@*/
PetscErrorCode  PetscFileRetrieve(MPI_Comm comm,const char url[],char localname[],size_t llen,PetscBool  *found)
{
  char           urlget[PETSC_MAX_PATH_LEN],*par,*tlocalname;
  FILE           *fp;
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  size_t         len = 0;
  PetscBool      flg1,flg2,flg3,flg4;
#if defined(PETSC_HAVE_POPEN)
  int            rval;
#endif

  PetscFunctionBegin;
  *found = PETSC_FALSE;

  /* if file does not have an ftp:// or http:// or .gz then need not process file */
  ierr = PetscStrstr(url,".gz",&par);CHKERRQ(ierr);
  if (par) {ierr = PetscStrlen(par,&len);CHKERRQ(ierr);}

  ierr = PetscStrncmp(url,"ftp://",6,&flg1);CHKERRQ(ierr);
  ierr = PetscStrncmp(url,"http://",7,&flg2);CHKERRQ(ierr);
  ierr = PetscStrncmp(url,"https://",8,&flg4);CHKERRQ(ierr);
  ierr = PetscStrncmp(url,"file://",7,&flg3);CHKERRQ(ierr);
  if (!flg1 && !flg2 && !flg3 && !flg4 && (!par || len != 3)) {
    ierr = PetscStrncpy(localname,url,llen);CHKERRQ(ierr);
    ierr = PetscTestFile(url,'r',found);CHKERRQ(ierr);
    if (*found) {
      ierr = PetscInfo1(NULL,"Found file %s\n",url);CHKERRQ(ierr);
    } else {
      ierr = PetscInfo1(NULL,"Did not find file %s\n",url);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
  }

  if (par && len == 3){
    size_t llen;
    ierr = PetscStrlen(url,&llen);CHKERRQ(ierr);
    ierr = PetscStrncpy(localname,url,llen);CHKERRQ(ierr);
    localname[llen-len] = 0;
    ierr = PetscTestFile(localname,'r',found);CHKERRQ(ierr);
    if (*found) {
      ierr = PetscInfo1(NULL,"Found uncompressed version of file %s\n",localname);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    } else {
      ierr = PetscInfo1(NULL,"Did not find uncompressed version of file %s\n",url);CHKERRQ(ierr);
    }
  }

  ierr = PetscStrrchr(url,'/',&tlocalname);CHKERRQ(ierr);
  ierr = PetscStrncpy(localname,tlocalname,llen);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    ierr = PetscTestFile(localname,'r',found);CHKERRQ(ierr);
    if (!*found) { /* local file is not already here so use curl to get it */
      ierr = PetscStrcpy(urlget,"curl ");CHKERRQ(ierr);
      ierr = PetscStrcat(urlget,url);CHKERRQ(ierr);
      ierr = PetscStrcat(urlget," > ");CHKERRQ(ierr);
      ierr = PetscStrcat(urlget,localname);CHKERRQ(ierr);

#if defined(PETSC_HAVE_POPEN)
      ierr = PetscPOpen(PETSC_COMM_SELF,NULL,urlget,"r",&fp);CHKERRQ(ierr);
#else
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP_SYS,"Cannot run external programs on this machine");
#endif
#if defined(PETSC_HAVE_POPEN)
      ierr = PetscPClose(PETSC_COMM_SELF,fp,&rval);CHKERRQ(ierr);
#endif
      ierr = PetscTestFile(localname,'r',found);CHKERRQ(ierr);
    }
  }
  ierr = MPI_Bcast(found,1,MPIU_BOOL,0,comm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
