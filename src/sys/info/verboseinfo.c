/*
      PetscInfo() is contained in a different file from the other profiling to
   allow it to be replaced at link time by an alternative routine.
*/
#include <petsc/private/petscimpl.h>        /*I    "petscsys.h"   I*/

/*
  The next set of variables determine which, if any, PetscInfo() calls are used.
  If PetscLogPrintInfo is false, no info messages are printed.

  If PetscInfoFlags[OBJECT_CLASSID - PETSC_SMALLEST_CLASSID] is zero, no messages related
  to that object are printed. OBJECT_CLASSID is, for example, MAT_CLASSID.
  Note for developers: the PetscInfoFlags array is currently 160 entries large, to ensure headroom. Perhaps it is worth
  dynamically allocating this array intelligently rather than just some big number.

  PetscInfoFilename determines where PetscInfo() output is piped.
  PetscInfoClassnames holds a char array of classes which are filtered out/for in PetscInfo() calls.
*/
const char * const        PetscInfoCommFlags[] = {"all", "no_self", "only_self", "PetscInfoCommFlag", "PETSC_INFO_COMM_", 0};
static PetscBool          PetscInfoClassesLocked = PETSC_FALSE, PetscInfoInvertClasses = PETSC_FALSE, PetscInfoClassesSet = PETSC_FALSE;
static char               **PetscInfoClassnames = NULL;
static char               *PetscInfoFilename = NULL;
static PetscInt           PetscInfoNumClasses = -1;
static PetscInfoCommFlag  PetscInfoCommFilter = PETSC_INFO_COMM_ALL;
static int                PetscInfoFlags[]  = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                                               1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                                               1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                                               1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                                               1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                                               1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                                               1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                                               1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                                               1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
                                               1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
PetscBool                 PetscLogPrintInfo = PETSC_FALSE;
FILE                      *PetscInfoFile = NULL;

/*@
    PetscInfoEnabled - Checks whether a given OBJECT_CLASSID is allowed to print using PetscInfo()

    Not Collective

    Input Parameters:
.   classid - PetscClassid retrieved from a PetscObject e.g. VEC_CLASSID

    Output Parameter:
.   enabled - PetscBool indicating whether this classid is allowed to print

    Notes:
    Use PETSC_SMALLEST_CLASSID to check if "sys" PetscInfo() calls are enabled. When PETSc is configured with debugging
    support this function checks if classid >= PETSC_SMALLEST_CLASSID, otherwise it assumes valid classid.

    Level: advanced

.seealso: PetscInfo(), PetscInfoAllow(), PetscInfoGetInfo(), PetscObjectGetClassid()
@*/
PetscErrorCode PetscInfoEnabled(PetscClassId classid, PetscBool *enabled)
{
  PetscFunctionBegin;
#if defined(PETSC_USE_DEBUG)
  if (classid < PETSC_SMALLEST_CLASSID) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Classid (current: %d) must be equal to or greater than PETSC_SMALLEST_CLASSID", classid);
#endif
  *enabled = (PetscBool) (PetscLogPrintInfo && PetscInfoFlags[classid - PETSC_SMALLEST_CLASSID]);
  PetscFunctionReturn(0);
}

/*@
    PetscInfoAllow - Enables/disables PetscInfo() messages

    Not Collective

    Input Parameter:
.   flag - PETSC_TRUE or PETSC_FALSE

    Level: advanced

.seealso: PetscInfo(), PetscInfoEnabled(), PetscInfoGetInfo(), PetscInfoSetFromOptions()
@*/
PetscErrorCode PetscInfoAllow(PetscBool flag)
{
  PetscFunctionBegin;
  PetscLogPrintInfo = flag;
  PetscFunctionReturn(0);
}

/*@C
    PetscInfoSetFile - Sets the printing destination for all PetscInfo() calls

    Not Collective

    Input Parameter:
+   filename - Name of the file where PetscInfo() will print to
-   mode - Write mode passed to PetscFOpen()

    Notes:
    Use filename=NULL to set PetscInfo() to write to PETSC_STDOUT.

    Level: advanced

.seealso: PetscInfo(), PetscInfoSetFile(), PetscInfoSetFromOptions(), PetscFOpen()
@*/
PetscErrorCode PetscInfoSetFile(const char filename[], const char mode[])
{
  char            fname[PETSC_MAX_PATH_LEN], tname[11];
  PetscMPIInt     rank;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (!PetscInfoFile) PetscInfoFile = PETSC_STDOUT;
  ierr = PetscFree(PetscInfoFilename);CHKERRQ(ierr);
  if (filename) {
    PetscBool  oldflag;
    PetscValidCharPointer(filename, 1);
    ierr = PetscFixFilename(filename, fname);CHKERRQ(ierr);
    ierr = PetscStrallocpy(fname, &PetscInfoFilename);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &rank);CHKERRQ(ierr);
    sprintf(tname, ".%d", rank);
    ierr = PetscStrcat(fname, tname);CHKERRQ(ierr);
    oldflag = PetscLogPrintInfo; PetscLogPrintInfo = PETSC_FALSE;
    ierr = PetscFOpen(MPI_COMM_SELF, fname, mode, &PetscInfoFile);CHKERRQ(ierr);
    PetscLogPrintInfo = oldflag;
    /* PetscFOpen will write to PETSC_STDOUT and not PetscInfoFile here, so we disable the PetscInfo call inside it, and
     call it afterwards so that it actually writes to file */
    ierr = PetscInfo1(NULL, "Opened PetscInfo file %s\n", fname);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
    PetscInfoGetFile - Gets the name and FILE pointer of the file where PetscInfo() prints to

    Not Collective

    Output Parameters:
+   filename - The name of the output file
-   InfoFile - The FILE pointer for the output file

    Level: advanced

    Note:
    This routine allocates and copies the filename so that the filename survives PetscInfoDestroy(). The user is
    therefore responsible for freeing the allocated filename pointer afterwards.

    Fortran Note:
    This routine is not supported in Fortran.

.seealso: PetscInfo(), PetscInfoSetFile(), PetscInfoSetFromOptions(), PetscInfoDestroy()
@*/
PetscErrorCode PetscInfoGetFile(char **filename, FILE **InfoFile)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidPointer(filename, 1);
  PetscValidPointer(InfoFile, 2);
  ierr = PetscStrallocpy(PetscInfoFilename, filename);CHKERRQ(ierr);
  *InfoFile = PetscInfoFile;
  PetscFunctionReturn(0);
}

/*@C
    PetscInfoSetClasses - Sets the classes which PetscInfo() is filtered for/against

    Not Collective

    Input Parameters:
+   exclude - Whether or not to invert the filter, i.e. if exclude is true, PetscInfo() will print from every class that
    is NOT one of the classes specified
.   N - Number of classes to filter for (size of classnames)
-   classnames - String array containing the names of classes to filter for, e.g. "vec"

    Notes:
    Not for use in Fortran

    This function CANNOT be called after PetscInfoGetClass() or PetscInfoProcessClass() has been called.

    Names in the classnames list should correspond to the names returned by PetscObjectGetClassName().

    This function only sets the list of class names.
    The actual filtering is deferred to PetscInfoProcessClass(), except of sys which is processed right away.
    The reason for this is that we need to set the list of included/excluded classes before their classids are known.
    Typically the classid is assigned and PetscInfoProcessClass() called in <Class>InitializePackage() (e.g. VecInitializePackage()).

    Level: developer

.seealso: PetscInfo(), PetscInfoGetClass(), PetscInfoProcessClass(), PetscInfoSetFromOptions(), PetscStrToArray(), PetscObjectGetName()
@*/
PetscErrorCode PetscInfoSetClasses(PetscBool exclude, PetscInt N, const char *const *classnames)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (PetscInfoClassesLocked) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "PetscInfoSetClasses() cannot be called after PetscInfoGetClass() or PetscInfoProcessClass()");
  ierr = PetscStrNArrayDestroy(PetscInfoNumClasses, &PetscInfoClassnames);CHKERRQ(ierr);
  ierr = PetscStrNArrayallocpy(N, classnames, &PetscInfoClassnames);CHKERRQ(ierr);
  PetscInfoNumClasses = N;
  PetscInfoInvertClasses = exclude;
  {
    /* Process sys class right away */
    PetscClassId  sysclassid = PETSC_SMALLEST_CLASSID;
    ierr = PetscInfoProcessClass("sys", 1, &sysclassid);CHKERRQ(ierr);
  }
  PetscInfoClassesSet = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@C
    PetscInfoGetClass - Indicates whether the provided classname is marked as a filter in PetscInfo() as set by PetscInfoSetClasses()

    Not Collective

    Input Paramater:
.   classname - Name of the class to search for

    Output Parameter:
.   found - PetscBool indicating whether the classname was found

    Notes:
    Use PetscObjectGetName() to retrieve an appropriate classname

    Level: developer

.seealso: PetscInfo(), PetscInfoSetClasses(), PetscInfoSetFromOptions(), PetscObjectGetName()
@*/
PetscErrorCode PetscInfoGetClass(const char *classname, PetscBool *found)
{
  PetscInt        idx;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidCharPointer(classname,1);
  ierr = PetscEListFind(PetscInfoNumClasses, (const char *const *) PetscInfoClassnames, classname ? classname : "sys", &idx, found);CHKERRQ(ierr);
  PetscInfoClassesLocked = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@
    PetscInfoGetInfo - Returns the current state of several important flags for PetscInfo()

    Not Collective

    Output Parameters:
+   infoEnabled - PETSC_TRUE if PetscInfoAllow(PETSC_TRUE) has been called
.   classesSet - PETSC_TRUE if the list of classes to filter for has been set
.   exclude - PETSC_TRUE if the class filtering for PetscInfo() is inverted
.   locked - PETSC_TRUE if the list of classes to filter for has been locked
-   commSelfFlag - Enum indicating whether PetscInfo() will print for communicators of size 1, any size != 1, or all
    communicators

    Notes:
    Initially commSelfFlag = PETSC_INFO_COMM_ALL

    Level: developer

.seealso: PetscInfo(), PetscInfoAllow(), PetscInfoSetFilterCommSelf, PetscInfoSetFromOptions()
@*/
PetscErrorCode PetscInfoGetInfo(PetscBool *infoEnabled, PetscBool *classesSet, PetscBool *exclude, PetscBool *locked, PetscInfoCommFlag *commSelfFlag)
{
  PetscFunctionBegin;
  if (infoEnabled)  *infoEnabled  = PetscLogPrintInfo;
  if (classesSet)   *classesSet   = PetscInfoClassesSet;
  if (exclude)      *exclude      = PetscInfoInvertClasses;
  if (locked)       *locked       = PetscInfoClassesLocked;
  if (commSelfFlag) *commSelfFlag = PetscInfoCommFilter;
  PetscFunctionReturn(0);
}

/*@C
    PetscInfoProcessClass - Activates or deactivates a class based on the filtering status of PetscInfo()

    Not Collective

    Input Parameters:
+   classname - Name of the class to activate/deactivate PetscInfo() for
.   numClassID - Number of entries in classIDs
-   classIDs - Array containing all of the PetscClassids associated with classname

    Level: developer

.seealso: PetscInfo(), PetscInfoActivateClass(), PetscInfoDeactivateClass(), PetscInfoSetFromOptions()
@*/
PetscErrorCode PetscInfoProcessClass(const char classname[], PetscInt numClassID, PetscClassId classIDs[])
{
  PetscInt        i;
  PetscBool       enabled, exclude, found, opt, pkg;
  char            logList[256];
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidCharPointer(classname, 1);
  ierr = PetscInfoGetInfo(&enabled, NULL, &exclude, NULL, NULL);CHKERRQ(ierr);
  /* -info_exclude is DEPRECATED */
  ierr = PetscOptionsGetString(NULL,NULL,"-info_exclude",logList,sizeof(logList),&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrInList(classname,logList,',',&pkg);CHKERRQ(ierr);
    if (pkg) {
      for (i = 0; i < numClassID; ++i) {
        ierr = PetscInfoDeactivateClass(classIDs[i]);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscInfoGetClass(classname, &found);CHKERRQ(ierr);
  if ((found && exclude) || (!found && !exclude)) {
    if (PetscInfoNumClasses > 0) {
      /* Check if -info was called empty */
      for (i = 0; i < numClassID; ++i) {
        ierr = PetscInfoDeactivateClass(classIDs[i]);CHKERRQ(ierr);
      }
    }
  } else {
    for (i = 0; i < numClassID; ++i) {
      ierr = PetscInfoActivateClass(classIDs[i]);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*@
    PetscInfoSetFilterCommSelf - Sets PetscInfoCommFlag enum to determine communicator filtering for PetscInfo()

    Not Collective

    Input Parameter:
.   commSelfFlag - Enum value indicating method with which to filter PetscInfo() based on the size of the communicator of the object calling PetscInfo()

    Level: advanced

.seealso: PetscInfo(), PetscInfoGetInfo()
@*/
PetscErrorCode PetscInfoSetFilterCommSelf(PetscInfoCommFlag commSelfFlag)
{
  PetscFunctionBegin;
  PetscInfoCommFilter = commSelfFlag;
  PetscFunctionReturn(0);
}

/*@
    PetscInfoSetFromOptions - Configure PetscInfo() using command line options, enabling or disabling various calls to PetscInfo()

    Not Collective

    Input Parameter:
.   options - Options database, use NULL for default global database

    Options Database Keys:
.   -info [filename][:[~]<list,of,classnames>[:[~]self]] - specify which informative messages are printed, See PetscInfo().

    Notes:
    This function is called automatically during PetscInitialize() so users usually do not need to call it themselves.

    Level: advanced

.seealso: PetscInfo(), PetscInfoAllow(), PetscInfoSetFile(), PetscInfoSetClasses(), PetscInfoSetFilterCommSelf(), PetscInfoDestroy()
@*/
PetscErrorCode PetscInfoSetFromOptions(PetscOptions options)
{
  char               optstring[PETSC_MAX_PATH_LEN], *loc0_ = NULL, *loc1_ = NULL, *loc2_ = NULL;
  char               **loc1_array = NULL;
  PetscBool          set, loc1_invert = PETSC_FALSE, loc2_invert = PETSC_FALSE, foundSelf = PETSC_FALSE;
  size_t             size_loc0_ = 0, size_loc1_ = 0, size_loc2_ = 0;
  int                nLoc1_ = 0;
  PetscInfoCommFlag  commSelfFlag = PETSC_INFO_COMM_ALL;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsDeprecated_Private(NULL,"-info_exclude", NULL, "3.13", "Use -info instead");CHKERRQ(ierr);
  ierr = PetscOptionsGetString(options, NULL, "-info", optstring, PETSC_MAX_PATH_LEN, &set);CHKERRQ(ierr);
  if (set) {
    PetscInfoClassesSet = PETSC_TRUE;
    ierr = PetscInfoAllow(PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscStrallocpy(optstring,&loc0_);CHKERRQ(ierr);
    ierr = PetscStrchr(loc0_,':',&loc1_);CHKERRQ(ierr);
    if (loc1_) {
      *loc1_++ = 0;
      if (*loc1_ == '~') {
        loc1_invert = PETSC_TRUE;
        ++loc1_;
      }
      ierr = PetscStrchr(loc1_,':',&loc2_);CHKERRQ(ierr);
    }
    if (loc2_) {
      *loc2_++ = 0;
      if (*loc2_ == '~') {
        loc2_invert = PETSC_TRUE;
        ++loc2_;
      }
    }
    ierr = PetscStrlen(loc0_, &size_loc0_);CHKERRQ(ierr);
    ierr = PetscStrlen(loc1_, &size_loc1_);CHKERRQ(ierr);
    ierr = PetscStrlen(loc2_, &size_loc2_);CHKERRQ(ierr);
    if (size_loc1_) {
      ierr = PetscStrtolower(loc1_);CHKERRQ(ierr);
      ierr = PetscStrToArray(loc1_, ',', &nLoc1_, &loc1_array);CHKERRQ(ierr);
    }
    if (size_loc2_) {
      ierr = PetscStrtolower(loc2_);CHKERRQ(ierr);
      ierr = PetscStrcmp("self", loc2_, &foundSelf);CHKERRQ(ierr);
      if (foundSelf) {
        if (loc2_invert) {
          commSelfFlag = PETSC_INFO_COMM_NO_SELF;
        } else {
          commSelfFlag = PETSC_INFO_COMM_ONLY_SELF;
        }
      }
    }
    ierr = PetscInfoSetFile(size_loc0_ ? loc0_ : NULL, "w");CHKERRQ(ierr);
    ierr = PetscInfoSetClasses(loc1_invert, (PetscInt) nLoc1_, (const char *const *) loc1_array);CHKERRQ(ierr);
    ierr = PetscInfoSetFilterCommSelf(commSelfFlag);CHKERRQ(ierr);
    ierr = PetscStrToArrayDestroy(nLoc1_, loc1_array);CHKERRQ(ierr);
    ierr = PetscFree(loc0_);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
  PetscInfoDestroy - Destroys and resets internal PetscInfo() data structures.

  Not Collective

  Notes:
  This is automatically called in PetscFinalize(). Useful for changing filters mid-program, or culling subsequent
  PetscInfo() calls down the line.

  Level: developer

.seealso: PetscInfo(), PetscInfoSetFromOptions()
@*/
PetscErrorCode PetscInfoDestroy(void)
{
  PetscErrorCode  ierr;
  int             err;
  size_t          i;

  PetscFunctionBegin;
  ierr = PetscInfoAllow(PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscStrNArrayDestroy(PetscInfoNumClasses, &PetscInfoClassnames);CHKERRQ(ierr);
  err  = fflush(PetscInfoFile);
  if (err) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SYS,"fflush() failed on file");
  if (PetscInfoFilename) {
    ierr  = PetscFClose(MPI_COMM_SELF, PetscInfoFile);CHKERRQ(ierr);
  }
  ierr = PetscFree(PetscInfoFilename);CHKERRQ(ierr);
  for (i=0; i<sizeof(PetscInfoFlags)/sizeof(PetscInfoFlags[0]); i++) PetscInfoFlags[i] = 1;
  PetscInfoClassesLocked = PETSC_FALSE;
  PetscInfoInvertClasses = PETSC_FALSE;
  PetscInfoClassesSet = PETSC_FALSE;
  PetscInfoNumClasses = -1;
  PetscInfoCommFilter = PETSC_INFO_COMM_ALL;
  PetscFunctionReturn(0);
}

/*@
  PetscInfoDeactivateClass - Deactivates PetscInfo() messages for a PETSc object class.

  Not Collective

  Input Parameter:
. classid - The object class,  e.g., MAT_CLASSID, SNES_CLASSID, etc.

  Notes:
  One can pass 0 to deactivate all messages that are not associated with an object.

  Level: developer

.seealso: PetscInfoActivateClass(), PetscInfo(), PetscInfoAllow(), PetscInfoSetFromOptions()
@*/
PetscErrorCode  PetscInfoDeactivateClass(PetscClassId classid)
{
  PetscFunctionBegin;
  if (!classid) classid = PETSC_SMALLEST_CLASSID;
  PetscInfoFlags[classid - PETSC_SMALLEST_CLASSID] = 0;
  PetscFunctionReturn(0);
}

/*@
  PetscInfoActivateClass - Activates PetscInfo() messages for a PETSc object class.

  Not Collective

  Input Parameter:
. classid - The object class, e.g., MAT_CLASSID, SNES_CLASSID, etc.

  Notes:
  One can pass 0 to activate all messages that are not associated with an object.

  Level: developer

.seealso: PetscInfoDeactivateClass(), PetscInfo(), PetscInfoAllow(), PetscInfoSetFromOptions()
@*/
PetscErrorCode  PetscInfoActivateClass(PetscClassId classid)
{
  PetscFunctionBegin;
  if (!classid) classid = PETSC_SMALLEST_CLASSID;
  PetscInfoFlags[classid - PETSC_SMALLEST_CLASSID] = 1;
  PetscFunctionReturn(0);
}

/*
   If the option -history was used, then all printed PetscInfo()
  messages are also printed to the history file, called by default
  .petschistory in ones home directory.
*/
PETSC_INTERN FILE *petsc_history;

/*MC
    PetscInfo - Logs informative data

   Synopsis:
       #include <petscsys.h>
       PetscErrorCode PetscInfo(PetscObject obj, const char message[])
       PetscErrorCode PetscInfo1(PetscObject obj, const char formatmessage[],arg1)
       PetscErrorCode PetscInfo2(PetscObject obj, const char formatmessage[],arg1,arg2)
       ...

    Collective on obj

    Input Parameter:
+   obj - object most closely associated with the logging statement or NULL
.   message - logging message
.   formatmessage - logging message using standard "printf" format
-   arg1, arg2, ... - arguments of the format

    Notes:
    PetscInfo() prints only from the first processor in the communicator of obj.
    If obj is NULL, the PETSC_COMM_SELF communicator is used, i.e. every rank of PETSC_COMM_WORLD prints the message.

    Extent of the printed messages can be controlled using the option database key -info as follows.

$   -info [filename][:[~]<list,of,classnames>[:[~]self]]

    No filename means standard output PETSC_STDOUT is used.

    The optional <list,of,classnames> is a comma separated list of enabled classes, e.g. vec,mat,ksp.
    If this list is not specified, all classes are enabled.
    Prepending the list with ~ means inverted selection, i.e. all classes except the listed are enabled.
    A special classname sys relates to PetscInfo() with obj being NULL. 

    The optional self keyword specifies that PetscInfo() is enabled only for communicator size = 1 (e.g. PETSC_COMM_SELF), i.e. only PetscInfo() calls which print from every rank of PETSC_COMM_WORLD are enabled.
    By contrast, ~self means that PetscInfo() is enabled only for communicator size > 1 (e.g. PETSC_COMM_WORLD), i.e. those PetscInfo() calls which print from every rank of PETSC_COMM_WORLD are disabled.

    All classname/self matching is case insensitive. Filename is case sensitive.

    Example of Usage:
$     Mat A;
$     PetscInt alpha;
$     ...
$     PetscInfo1(A,"Matrix uses parameter alpha=%D\n",alpha);

    Options Examples:
    Each call of the form
$     PetscInfo(obj, msg);
$     PetscInfo1(obj, msg, arg1);
$     PetscInfo2(obj, msg, arg1, arg2);
    is evaluated as follows.
$     -info or -info :: prints msg to PETSC_STDOUT, for any obj regardless class or communicator
$     -info :mat:self prints msg to PETSC_STDOUT only if class of obj is Mat, and its communicator has size = 1
$     -info myInfoFileName:~vec:~self prints msg to file named myInfoFileName, only if the obj's class is NULL or other than Vec, and obj's communicator has size > 1
$     -info :sys prints to PETSC_STDOUT only if obj is NULL
    Note that
$     -info :sys:~self
    deactivates all info messages because sys means obj = NULL which implies PETSC_COMM_SELF but ~self filters out everything on PETSC_COMM_SELF.

    Fortran Note:
    This function does not take the obj argument, there is only the PetscInfo()
     version, not PetscInfo1() etc.

    Level: intermediate

.seealso: PetscInfoAllow(), PetscInfoSetFromOptions()
M*/
PetscErrorCode  PetscInfo_Private(const char func[],PetscObject obj, const char message[], ...)
{
  va_list        Argp;
  PetscMPIInt    rank = 0,urank,size = 1;
  PetscClassId   classid;
  PetscBool      enabled = PETSC_FALSE, oldflag;
  char           string[8*1024];
  PetscErrorCode ierr;
  size_t         fullLength,len;
  int            err;

  PetscFunctionBegin;
  if (obj) PetscValidHeader(obj,2);
  classid = obj ? obj->classid : PETSC_SMALLEST_CLASSID;
  ierr = PetscInfoEnabled(classid, &enabled);CHKERRQ(ierr);
  if (!enabled) PetscFunctionReturn(0);
  PetscValidCharPointer(message,3);
  if (obj) {
    ierr = MPI_Comm_rank(obj->comm, &rank);CHKERRQ(ierr);
    ierr = MPI_Comm_size(obj->comm, &size);CHKERRQ(ierr);
  }
  /* rank > 0 always jumps out */
  if (rank) PetscFunctionReturn(0);
  if (!PetscInfoCommFilter && (size < 2)) {
    /* If no self printing is allowed, and size too small get out */
    PetscFunctionReturn(0);
  } else if ((PetscInfoCommFilter == PETSC_INFO_COMM_ONLY_SELF) && (size > 1)) {
    /* If ONLY self printing, and size too big, get out */
    PetscFunctionReturn(0);
  }
  /* Mute info messages within this function */
  oldflag = PetscLogPrintInfo; PetscLogPrintInfo = PETSC_FALSE;
  ierr = MPI_Comm_rank(MPI_COMM_WORLD, &urank);CHKERRQ(ierr);
  va_start(Argp, message);
  sprintf(string, "[%d] %s(): ",urank,func);
  ierr = PetscStrlen(string, &len);CHKERRQ(ierr);
  ierr = PetscVSNPrintf(string+len, 8*1024-len,message,&fullLength, Argp);CHKERRQ(ierr);
  ierr = PetscFPrintf(PETSC_COMM_SELF,PetscInfoFile, "%s", string);CHKERRQ(ierr);
  err  = fflush(PetscInfoFile);
  if (err) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SYS,"fflush() failed on file");
  if (petsc_history) {
    va_start(Argp, message);
    ierr = (*PetscVFPrintf)(petsc_history, message, Argp);CHKERRQ(ierr);
  }
  va_end(Argp);
  PetscLogPrintInfo = oldflag;
  PetscFunctionReturn(0);
}
