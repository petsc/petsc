
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
#include <petsc/private/petscimpl.h>

/*
   Private routine to delete tmp/shared storage

   This is called by MPI, not by users.

   Note: this is declared extern "C" because it is passed to MPI_Comm_create_keyval()

*/
PETSC_EXTERN PetscMPIInt MPIAPI Petsc_DelTmpShared(MPI_Comm comm, PetscMPIInt keyval, void *count_val, void *extra_state)
{
  PetscFunctionBegin;
  PetscCallMPI(PetscInfo(NULL, "Deleting tmp/shared data in an MPI_Comm %ld\n", (long)comm));
  PetscCallMPI(PetscFree(count_val));
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
+     `PETSC_SHARED_TMP` - indicates the directory is shared among the MPI ranks
.     `PETSC_NOT_SHARED_TMP` - indicates the directory is not shared among the MPI ranks
-     `PETSC_TMP` - name of the directory you wish to use as /tmp

   Level: developer

.seealso: `PetscSharedTmp()`, `PetscSharedWorkingDirectory()`, `PetscGetWorkingDirectory()`, `PetscGetHomeDirectory()`
@*/
PetscErrorCode PetscGetTmp(MPI_Comm comm, char dir[], size_t len)
{
  PetscBool flg;

  PetscFunctionBegin;
  PetscCall(PetscOptionsGetenv(comm, "PETSC_TMP", dir, len, &flg));
  if (!flg) PetscCall(PetscStrncpy(dir, "/tmp", len));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscSharedTmp - Determines if all processors in a communicator share a
         /tmp or have different ones.

   Collective

   Input Parameter:
.  comm - MPI_Communicator that may share /tmp

   Output Parameter:
.  shared - `PETSC_TRUE` or `PETSC_FALSE`

   Options Database Keys:
+    -shared_tmp  - indicates the directory is shared among the MPI ranks
.    -not_shared_tmp - indicates the directory is not shared among the MPI ranks
-    -tmp tmpdir - name of the directory you wish to use as /tmp

   Environmental Variables:
+     `PETSC_SHARED_TMP`  - indicates the directory is shared among the MPI ranks
.     `PETSC_NOT_SHARED_TMP` - indicates the directory is not shared among the MPI ranks
-     `PETSC_TMP` - name of the directory you wish to use as /tmp

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

   If the environmental variable `PETSC_TMP` is set it will use this directory
  as the "/tmp" directory.

.seealso: `PetscGetTmp()`, `PetscSharedWorkingDirectory()`, `PetscGetWorkingDirectory()`, `PetscGetHomeDirectory()`
@*/
PetscErrorCode PetscSharedTmp(MPI_Comm comm, PetscBool *shared)
{
  PetscMPIInt size, rank, *tagvalp, sum, cnt, i;
  PetscBool   flg, iflg;
  FILE       *fd;
  int         err;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  if (size == 1) {
    *shared = PETSC_TRUE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(PetscOptionsGetenv(comm, "PETSC_SHARED_TMP", NULL, 0, &flg));
  if (flg) {
    *shared = PETSC_TRUE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(PetscOptionsGetenv(comm, "PETSC_NOT_SHARED_TMP", NULL, 0, &flg));
  if (flg) {
    *shared = PETSC_FALSE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  if (Petsc_SharedTmp_keyval == MPI_KEYVAL_INVALID) PetscCallMPI(MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN, Petsc_DelTmpShared, &Petsc_SharedTmp_keyval, NULL));

  PetscCallMPI(MPI_Comm_get_attr(comm, Petsc_SharedTmp_keyval, (void **)&tagvalp, (int *)&iflg));
  if (!iflg) {
    char filename[PETSC_MAX_PATH_LEN], tmpname[PETSC_MAX_PATH_LEN];

    /* This communicator does not yet have a shared tmp attribute */
    PetscCall(PetscMalloc1(1, &tagvalp));
    PetscCallMPI(MPI_Comm_set_attr(comm, Petsc_SharedTmp_keyval, tagvalp));

    PetscCall(PetscOptionsGetenv(comm, "PETSC_TMP", tmpname, 238, &iflg));
    if (!iflg) {
      PetscCall(PetscStrncpy(filename, "/tmp", sizeof(filename)));
    } else {
      PetscCall(PetscStrncpy(filename, tmpname, sizeof(filename)));
    }

    PetscCall(PetscStrlcat(filename, "/petsctestshared", sizeof(filename)));
    PetscCallMPI(MPI_Comm_rank(comm, &rank));

    /* each processor creates a /tmp file and all the later ones check */
    /* this makes sure no subset of processors is shared */
    *shared = PETSC_FALSE;
    for (i = 0; i < size - 1; i++) {
      if (rank == i) {
        fd = fopen(filename, "w");
        PetscCheck(fd, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Unable to open test file %s", filename);
        err = fclose(fd);
        PetscCheck(!err, PETSC_COMM_SELF, PETSC_ERR_SYS, "fclose() failed on file");
      }
      PetscCallMPI(MPI_Barrier(comm));
      if (rank >= i) {
        fd = fopen(filename, "r");
        if (fd) cnt = 1;
        else cnt = 0;
        if (fd) {
          err = fclose(fd);
          PetscCheck(!err, PETSC_COMM_SELF, PETSC_ERR_SYS, "fclose() failed on file");
        }
      } else cnt = 0;

      PetscCall(MPIU_Allreduce(&cnt, &sum, 1, MPI_INT, MPI_SUM, comm));
      if (rank == i) unlink(filename);

      if (sum == size) {
        *shared = PETSC_TRUE;
        break;
      } else PetscCheck(sum == 1, PETSC_COMM_SELF, PETSC_ERR_SUP_SYS, "Subset of processes share /tmp ");
    }
    *tagvalp = (int)*shared;
    PetscCall(PetscInfo(NULL, "processors %s %s\n", (*shared) ? "share" : "do NOT share", (iflg ? tmpname : "/tmp")));
  } else *shared = (PetscBool)*tagvalp;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscSharedWorkingDirectory - Determines if all processors in a communicator share a working directory or have different ones.

  Collective

  Input Parameter:
. comm - MPI_Communicator that may share working directory

  Output Parameter:
. shared - `PETSC_TRUE` or `PETSC_FALSE`

  Options Database Keys:
+ -shared_working_directory - indicates the directory is shared among the MPI ranks
- -not_shared_working_directory - indicates the directory is shared among the MPI ranks

  Environmental Variables:
+ `PETSC_SHARED_WORKING_DIRECTORY` - indicates the directory is shared among the MPI ranks
- `PETSC_NOT_SHARED_WORKING_DIRECTORY` - indicates the directory is shared among the MPI ranks

  Level: developer

  Notes:
  Stores the status as a MPI attribute so it does not have to be redetermined each time.

  Assumes that all processors in a communicator either
.vb
   1) have a common working directory or
   2) each has a separate working directory
.ve
  eventually we can write a fancier one that determines which processors share a common working directory.

  This will be very slow on runs with a large number of processors since it requires O(p*p) file opens.

.seealso: `PetscGetTmp()`, `PetscSharedTmp()`, `PetscGetWorkingDirectory()`, `PetscGetHomeDirectory()`
@*/
PetscErrorCode PetscSharedWorkingDirectory(MPI_Comm comm, PetscBool *shared)
{
  PetscMPIInt size, rank, *tagvalp, sum, cnt, i;
  PetscBool   flg, iflg;
  FILE       *fd;
  int         err;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  if (size == 1) {
    *shared = PETSC_TRUE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(PetscOptionsGetenv(comm, "PETSC_SHARED_WORKING_DIRECTORY", NULL, 0, &flg));
  if (flg) {
    *shared = PETSC_TRUE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(PetscOptionsGetenv(comm, "PETSC_NOT_SHARED_WORKING_DIRECTORY", NULL, 0, &flg));
  if (flg) {
    *shared = PETSC_FALSE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  if (Petsc_SharedWD_keyval == MPI_KEYVAL_INVALID) PetscCallMPI(MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN, Petsc_DelTmpShared, &Petsc_SharedWD_keyval, NULL));

  PetscCallMPI(MPI_Comm_get_attr(comm, Petsc_SharedWD_keyval, (void **)&tagvalp, (int *)&iflg));
  if (!iflg) {
    char filename[PETSC_MAX_PATH_LEN];

    /* This communicator does not yet have a shared  attribute */
    PetscCall(PetscMalloc1(1, &tagvalp));
    PetscCallMPI(MPI_Comm_set_attr(comm, Petsc_SharedWD_keyval, tagvalp));

    PetscCall(PetscGetWorkingDirectory(filename, 240));
    PetscCall(PetscStrlcat(filename, "/petsctestshared", sizeof(filename)));
    PetscCallMPI(MPI_Comm_rank(comm, &rank));

    /* each processor creates a  file and all the later ones check */
    /* this makes sure no subset of processors is shared */
    *shared = PETSC_FALSE;
    for (i = 0; i < size - 1; i++) {
      if (rank == i) {
        fd = fopen(filename, "w");
        PetscCheck(fd, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Unable to open test file %s", filename);
        err = fclose(fd);
        PetscCheck(!err, PETSC_COMM_SELF, PETSC_ERR_SYS, "fclose() failed on file");
      }
      PetscCallMPI(MPI_Barrier(comm));
      if (rank >= i) {
        fd = fopen(filename, "r");
        if (fd) cnt = 1;
        else cnt = 0;
        if (fd) {
          err = fclose(fd);
          PetscCheck(!err, PETSC_COMM_SELF, PETSC_ERR_SYS, "fclose() failed on file");
        }
      } else cnt = 0;

      PetscCall(MPIU_Allreduce(&cnt, &sum, 1, MPI_INT, MPI_SUM, comm));
      if (rank == i) unlink(filename);

      if (sum == size) {
        *shared = PETSC_TRUE;
        break;
      } else PetscCheck(sum == 1, PETSC_COMM_SELF, PETSC_ERR_SUP_SYS, "Subset of processes share working directory");
    }
    *tagvalp = (int)*shared;
  } else *shared = (PetscBool)*tagvalp;
  PetscCall(PetscInfo(NULL, "processors %s working directory\n", (*shared) ? "shared" : "do NOT share"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
    PetscFileRetrieve - Obtains a file from a URL or a compressed file
        and copies into local disk space as uncompressed.

    Collective

    Input Parameters:
+   comm     - processors accessing the file
.   url      - name of file, including entire URL (with or without .gz)
-   llen     - length of localname

    Output Parameters:
+   localname - name of local copy of file - valid on only process zero
-   found - if found or retrieved the file - valid on all processes

    Level: intermediate

    Note:
    if the file already exists local this function just returns without downloading it.

@*/
PetscErrorCode PetscFileRetrieve(MPI_Comm comm, const char url[], char localname[], size_t llen, PetscBool *found)
{
  char        buffer[PETSC_MAX_PATH_LEN], *par = NULL, *tlocalname = NULL, name[PETSC_MAX_PATH_LEN];
  FILE       *fp;
  PetscMPIInt rank;
  size_t      len = 0;
  PetscBool   flg1, flg2, flg3, flg4, download, compressed = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  if (rank == 0) {
    *found = PETSC_FALSE;

    PetscCall(PetscStrstr(url, ".gz", &par));
    if (par) {
      PetscCall(PetscStrlen(par, &len));
      if (len == 3) compressed = PETSC_TRUE;
    }

    PetscCall(PetscStrncmp(url, "ftp://", 6, &flg1));
    PetscCall(PetscStrncmp(url, "http://", 7, &flg2));
    PetscCall(PetscStrncmp(url, "file://", 7, &flg3));
    PetscCall(PetscStrncmp(url, "https://", 8, &flg4));
    download = (PetscBool)(flg1 || flg2 || flg3 || flg4);

    if (!download && !compressed) {
      PetscCall(PetscStrncpy(localname, url, llen));
      PetscCall(PetscTestFile(url, 'r', found));
      if (*found) {
        PetscCall(PetscInfo(NULL, "Found file %s\n", url));
      } else {
        PetscCall(PetscInfo(NULL, "Did not find file %s\n", url));
      }
      goto done;
    }

    /* look for uncompressed file in requested directory */
    if (compressed) {
      PetscCall(PetscStrncpy(localname, url, llen));
      PetscCall(PetscStrstr(localname, ".gz", &par));
      *par = 0; /* remove .gz extension */
      PetscCall(PetscTestFile(localname, 'r', found));
      if (*found) goto done;
    }

    /* look for file in current directory */
    PetscCall(PetscStrrchr(url, '/', &tlocalname));
    PetscCall(PetscStrncpy(localname, tlocalname, llen));
    if (compressed) {
      PetscCall(PetscStrstr(localname, ".gz", &par));
      *par = 0; /* remove .gz extension */
    }
    PetscCall(PetscTestFile(localname, 'r', found));
    if (*found) goto done;

    if (download) {
      /* local file is not already here so use curl to get it */
      PetscCall(PetscStrncpy(localname, tlocalname, llen));
      PetscCall(PetscStrncpy(buffer, "curl --fail --silent --show-error ", sizeof(buffer)));
      PetscCall(PetscStrlcat(buffer, url, sizeof(buffer)));
      PetscCall(PetscStrlcat(buffer, " > ", sizeof(buffer)));
      PetscCall(PetscStrlcat(buffer, localname, sizeof(buffer)));
#if defined(PETSC_HAVE_POPEN)
      PetscCall(PetscPOpen(PETSC_COMM_SELF, NULL, buffer, "r", &fp));
      PetscCall(PetscPClose(PETSC_COMM_SELF, fp));
#else
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP_SYS, "Cannot run external programs on this machine");
#endif
      PetscCall(PetscTestFile(localname, 'r', found));
      if (*found) {
        FILE *fd;
        char  buf[1024], *str, *substring;

        /* check if the file didn't exist so it downloaded an HTML message instead */
        fd = fopen(localname, "r");
        PetscCheck(fd, PETSC_COMM_SELF, PETSC_ERR_PLIB, "PetscTestFile() indicates %s exists but fopen() cannot open it", localname);
        str = fgets(buf, sizeof(buf) - 1, fd);
        while (str) {
          PetscCall(PetscStrstr(buf, "<!DOCTYPE html>", &substring));
          PetscCheck(!substring, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unable to download %s it does not appear to exist at this URL, dummy HTML file was downloaded", url);
          PetscCall(PetscStrstr(buf, "Not Found", &substring));
          PetscCheck(!substring, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unable to download %s it does not appear to exist at this URL, dummy HTML file was downloaded", url);
          str = fgets(buf, sizeof(buf) - 1, fd);
        }
        fclose(fd);
      }
    } else if (compressed) {
      PetscCall(PetscTestFile(url, 'r', found));
      if (!*found) goto done;
      PetscCall(PetscStrncpy(localname, url, llen));
    }
    if (compressed) {
      PetscCall(PetscStrrchr(localname, '/', &tlocalname));
      PetscCall(PetscStrncpy(name, tlocalname, PETSC_MAX_PATH_LEN));
      PetscCall(PetscStrstr(name, ".gz", &par));
      *par = 0; /* remove .gz extension */
      /* uncompress file */
      PetscCall(PetscStrncpy(buffer, "gzip -c -d ", sizeof(buffer)));
      PetscCall(PetscStrlcat(buffer, localname, sizeof(buffer)));
      PetscCall(PetscStrlcat(buffer, " > ", sizeof(buffer)));
      PetscCall(PetscStrlcat(buffer, name, sizeof(buffer)));
#if defined(PETSC_HAVE_POPEN)
      PetscCall(PetscPOpen(PETSC_COMM_SELF, NULL, buffer, "r", &fp));
      PetscCall(PetscPClose(PETSC_COMM_SELF, fp));
#else
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP_SYS, "Cannot run external programs on this machine");
#endif
      PetscCall(PetscStrncpy(localname, name, llen));
      PetscCall(PetscTestFile(localname, 'r', found));
    }
  }
done:
  PetscCallMPI(MPI_Bcast(found, 1, MPIU_BOOL, 0, comm));
  PetscCallMPI(MPI_Bcast(localname, llen, MPI_CHAR, 0, comm));
  PetscFunctionReturn(PETSC_SUCCESS);
}
