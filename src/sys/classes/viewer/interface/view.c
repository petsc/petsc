
#include <petsc/private/viewerimpl.h>  /*I "petscviewer.h" I*/

PetscClassId PETSC_VIEWER_CLASSID;

static PetscBool PetscViewerPackageInitialized = PETSC_FALSE;
/*@C
  PetscViewerFinalizePackage - This function destroys any global objects created in the Petsc viewers. It is
  called from PetscFinalize().

  Level: developer

.seealso: PetscFinalize()
@*/
PetscErrorCode  PetscViewerFinalizePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (Petsc_Viewer_keyval != MPI_KEYVAL_INVALID) {
    ierr = MPI_Comm_free_keyval(&Petsc_Viewer_keyval);CHKERRQ(ierr);
  }
  if (Petsc_Viewer_Stdout_keyval != MPI_KEYVAL_INVALID) {
    ierr = MPI_Comm_free_keyval(&Petsc_Viewer_Stdout_keyval);CHKERRQ(ierr);
  }
  if (Petsc_Viewer_Stderr_keyval != MPI_KEYVAL_INVALID) {
    ierr = MPI_Comm_free_keyval(&Petsc_Viewer_Stderr_keyval);CHKERRQ(ierr);
  }
  if (Petsc_Viewer_Binary_keyval != MPI_KEYVAL_INVALID) {
    ierr = MPI_Comm_free_keyval(&Petsc_Viewer_Binary_keyval);CHKERRQ(ierr);
  }
  if (Petsc_Viewer_Draw_keyval != MPI_KEYVAL_INVALID) {
    ierr = MPI_Comm_free_keyval(&Petsc_Viewer_Draw_keyval);CHKERRQ(ierr);
  }
#if defined(PETSC_HAVE_HDF5)
  if (Petsc_Viewer_HDF5_keyval != MPI_KEYVAL_INVALID) {
    ierr = MPI_Comm_free_keyval(&Petsc_Viewer_HDF5_keyval);CHKERRQ(ierr);
  }
#endif
#if defined(PETSC_USE_SOCKETVIEWER)
  if (Petsc_Viewer_Socket_keyval != MPI_KEYVAL_INVALID) {
    ierr = MPI_Comm_free_keyval(&Petsc_Viewer_Socket_keyval);CHKERRQ(ierr);
  }
#endif
  ierr = PetscFunctionListDestroy(&PetscViewerList);CHKERRQ(ierr);
  PetscViewerPackageInitialized = PETSC_FALSE;
  PetscViewerRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@C
  PetscViewerInitializePackage - This function initializes everything in the main PetscViewer package.

  Level: developer

.seealso: PetscInitialize()
@*/
PetscErrorCode  PetscViewerInitializePackage(void)
{
  char           logList[256];
  PetscBool      opt,pkg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscViewerPackageInitialized) PetscFunctionReturn(0);
  PetscViewerPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  ierr = PetscClassIdRegister("Viewer",&PETSC_VIEWER_CLASSID);CHKERRQ(ierr);
  /* Register Constructors */
  ierr = PetscViewerRegisterAll();CHKERRQ(ierr);
  /* Process Info */
  {
    PetscClassId  classids[1];

    classids[0] = PETSC_VIEWER_CLASSID;
    ierr = PetscInfoProcessClass("viewer", 1, classids);CHKERRQ(ierr);
  }
  /* Process summary exclusions */
  ierr = PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscStrInList("viewer",logList,',',&pkg);CHKERRQ(ierr);
    if (pkg) {ierr = PetscLogEventExcludeClass(PETSC_VIEWER_CLASSID);CHKERRQ(ierr);}
  }
#if defined(PETSC_HAVE_MATHEMATICA)
  ierr = PetscViewerMathematicaInitializePackage();CHKERRQ(ierr);
#endif
  /* Register package finalizer */
  ierr = PetscRegisterFinalize(PetscViewerFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   PetscViewerDestroy - Destroys a PetscViewer.

   Collective on PetscViewer

   Input Parameters:
.  viewer - the PetscViewer to be destroyed.

   Level: beginner

.seealso: PetscViewerSocketOpen(), PetscViewerASCIIOpen(), PetscViewerCreate(), PetscViewerDrawOpen()

@*/
PetscErrorCode  PetscViewerDestroy(PetscViewer *viewer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*viewer) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*viewer,PETSC_VIEWER_CLASSID,1);

  ierr = PetscViewerFlush(*viewer);CHKERRQ(ierr);
  if (--((PetscObject)(*viewer))->refct > 0) {*viewer = NULL; PetscFunctionReturn(0);}

  ierr = PetscObjectSAWsViewOff((PetscObject)*viewer);CHKERRQ(ierr);
  if ((*viewer)->ops->destroy) {
    ierr = (*(*viewer)->ops->destroy)(*viewer);CHKERRQ(ierr);
  }
  ierr = PetscHeaderDestroy(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscViewerAndFormatCreate - Creates a PetscViewerAndFormat struct.

   Collective on PetscViewer

   Input Parameters:
+  viewer - the viewer
-  format - the format 

   Output Parameter:
.   vf - viewer and format object

   Notes:
    This increases the reference count of the viewer so you can destroy the viewer object after this call
   Level: developer

   This is used as the context variable for many of the TS, SNES, and KSP monitor functions

.seealso: PetscViewerSocketOpen(), PetscViewerASCIIOpen(), PetscViewerCreate(), PetscViewerDrawOpen(), PetscViewerAndFormatDestroy()

@*/
PetscErrorCode  PetscViewerAndFormatCreate(PetscViewer viewer, PetscViewerFormat format,PetscViewerAndFormat **vf)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectReference((PetscObject)viewer);CHKERRQ(ierr);
  ierr = PetscNew(vf);CHKERRQ(ierr);
  (*vf)->viewer = viewer;
  (*vf)->format = format;
  PetscFunctionReturn(0);
}


/*@C
   PetscViewerAndFormatDestroy - Destroys a PetscViewerAndFormat struct.

   Collective on PetscViewer

   Input Parameters:
.  viewer - the PetscViewerAndFormat to be destroyed.

   Level: developer

.seealso: PetscViewerSocketOpen(), PetscViewerASCIIOpen(), PetscViewerCreate(), PetscViewerDrawOpen(), PetscViewerAndFormatCreate()

@*/
PetscErrorCode  PetscViewerAndFormatDestroy(PetscViewerAndFormat **vf)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerDestroy(&(*vf)->viewer);CHKERRQ(ierr);
  ierr = PetscFree(*vf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscViewerGetType - Returns the type of a PetscViewer.

   Not Collective

   Input Parameter:
.   viewer - the PetscViewer

   Output Parameter:
.  type - PetscViewer type (see below)

   Available Types Include:
+  PETSCVIEWERSOCKET - Socket PetscViewer
.  PETSCVIEWERASCII - ASCII PetscViewer
.  PETSCVIEWERBINARY - binary file PetscViewer
.  PETSCVIEWERSTRING - string PetscViewer
-  PETSCVIEWERDRAW - drawing PetscViewer

   Level: intermediate

   Note:
   See include/petscviewer.h for a complete list of PetscViewers.

   PetscViewerType is actually a string

.seealso: PetscViewerCreate(), PetscViewerSetType(), PetscViewerType

@*/
PetscErrorCode  PetscViewerGetType(PetscViewer viewer,PetscViewerType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  PetscValidPointer(type,2);
  *type = ((PetscObject)viewer)->type_name;
  PetscFunctionReturn(0);
}

/*@C
   PetscViewerSetOptionsPrefix - Sets the prefix used for searching for all
   PetscViewer options in the database.

   Logically Collective on PetscViewer

   Input Parameter:
+  viewer - the PetscViewer context
-  prefix - the prefix to prepend to all option names

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

   Level: advanced

.seealso: PetscViewerSetFromOptions()
@*/
PetscErrorCode  PetscViewerSetOptionsPrefix(PetscViewer viewer,const char prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)viewer,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscViewerAppendOptionsPrefix - Appends to the prefix used for searching for all
   PetscViewer options in the database.

   Logically Collective on PetscViewer

   Input Parameters:
+  viewer - the PetscViewer context
-  prefix - the prefix to prepend to all option names

   Notes:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

   Level: advanced

.seealso: PetscViewerGetOptionsPrefix()
@*/
PetscErrorCode  PetscViewerAppendOptionsPrefix(PetscViewer viewer,const char prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  ierr = PetscObjectAppendOptionsPrefix((PetscObject)viewer,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscViewerGetOptionsPrefix - Sets the prefix used for searching for all
   PetscViewer options in the database.

   Not Collective

   Input Parameter:
.  viewer - the PetscViewer context

   Output Parameter:
.  prefix - pointer to the prefix string used

   Notes:
    On the fortran side, the user should pass in a string 'prefix' of
   sufficient length to hold the prefix.

   Level: advanced

.seealso: PetscViewerAppendOptionsPrefix()
@*/
PetscErrorCode  PetscViewerGetOptionsPrefix(PetscViewer viewer,const char *prefix[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  ierr = PetscObjectGetOptionsPrefix((PetscObject)viewer,prefix);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   PetscViewerSetUp - Sets up the internal viewer data structures for the later use.

   Collective on PetscViewer

   Input Parameters:
.  viewer - the PetscViewer context

   Notes:
   For basic use of the PetscViewer classes the user need not explicitly call
   PetscViewerSetUp(), since these actions will happen automatically.

   Level: advanced

.seealso: PetscViewerCreate(), PetscViewerDestroy()
@*/
PetscErrorCode  PetscViewerSetUp(PetscViewer viewer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  if (viewer->setupcalled) PetscFunctionReturn(0);
  if (viewer->ops->setup) {
    ierr = (*viewer->ops->setup)(viewer);CHKERRQ(ierr);
  }
  viewer->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@C
   PetscViewerViewFromOptions - View from Options

   Collective on PetscViewer

   Input Parameters:
+  A - the PetscViewer context
.  obj - Optional object
-  name - command line option

   Level: intermediate
.seealso:  PetscViewer, PetscViewerView, PetscObjectViewFromOptions(), PetscViewerCreate()
@*/
PetscErrorCode  PetscViewerViewFromOptions(PetscViewer A,PetscObject obj,const char name[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,PETSC_VIEWER_CLASSID,1);
  ierr = PetscObjectViewFromOptions((PetscObject)A,obj,name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscViewerView - Visualizes a viewer object.

   Collective on PetscViewer

   Input Parameters:
+  v - the viewer to be viewed
-  viewer - visualization context

  Notes:
  The available visualization contexts include
+    PETSC_VIEWER_STDOUT_SELF - standard output (default)
.    PETSC_VIEWER_STDOUT_WORLD - synchronized standard
        output where only the first processor opens
        the file.  All other processors send their
        data to the first processor to print.
-     PETSC_VIEWER_DRAW_WORLD - graphical display of nonzero structure

   Level: beginner

.seealso: PetscViewerPushFormat(), PetscViewerASCIIOpen(), PetscViewerDrawOpen(),
          PetscViewerSocketOpen(), PetscViewerBinaryOpen(), PetscViewerLoad()
@*/
PetscErrorCode  PetscViewerView(PetscViewer v,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscBool         iascii;
  PetscViewerFormat format;
#if defined(PETSC_HAVE_SAWS)
  PetscBool         issaws;
#endif

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,PETSC_VIEWER_CLASSID,1);
  PetscValidType(v,1);
  if (!viewer) {
    ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)v),&viewer);CHKERRQ(ierr);
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(v,1,viewer,2);

  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
#if defined(PETSC_HAVE_SAWS)
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERSAWS,&issaws);CHKERRQ(ierr);
#endif
  if (iascii) {
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    ierr = PetscObjectPrintClassNamePrefixType((PetscObject)v,viewer);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_DEFAULT || format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      if (v->format) {
        ierr = PetscViewerASCIIPrintf(viewer,"  Viewer format = %s\n",PetscViewerFormats[v->format]);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      if (v->ops->view) {
        ierr = (*v->ops->view)(v,viewer);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
#if defined(PETSC_HAVE_SAWS)
  } else if (issaws) {
    if (!((PetscObject)v)->amsmem) {
      ierr = PetscObjectViewSAWs((PetscObject)v,viewer);CHKERRQ(ierr);
      if (v->ops->view) {
        ierr = (*v->ops->view)(v,viewer);CHKERRQ(ierr);
      }
    }
#endif
  }
  PetscFunctionReturn(0);
}

/*@C
   PetscViewerRead - Reads data from a PetscViewer

   Collective

   Input Parameters:
+  viewer   - The viewer
.  data     - Location to write the data
.  num      - Number of items of data to read
-  datatype - Type of data to read

   Output Parameters:
.  count - number of items of data actually read, or NULL

   Notes:
   If datatype is PETSC_STRING and num is negative, reads until a newline character is found,
   until a maximum of (-num - 1) chars.

   Level: beginner

.seealso: PetscViewerASCIIOpen(), PetscViewerPushFormat(), PetscViewerDestroy(),
          VecView(), MatView(), VecLoad(), MatLoad(), PetscViewerBinaryGetDescriptor(),
          PetscViewerBinaryGetInfoPointer(), PetscFileMode, PetscViewer
@*/
PetscErrorCode  PetscViewerRead(PetscViewer viewer, void *data, PetscInt num, PetscInt *count, PetscDataType dtype)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  if (dtype == PETSC_STRING) {
    PetscInt c, i = 0, cnt;
    char *s = (char *)data;
    if (num >= 0) {
      for (c = 0; c < num; c++) {
        /* Skip leading whitespaces */
        do {ierr = (*viewer->ops->read)(viewer, &(s[i]), 1, &cnt, PETSC_CHAR);CHKERRQ(ierr); if (!cnt) break;}
        while (s[i]=='\n' || s[i]=='\t' || s[i]==' ' || s[i]=='\0' || s[i]=='\v' || s[i]=='\f' || s[i]=='\r');
        i++;
        /* Read strings one char at a time */
        do {ierr = (*viewer->ops->read)(viewer, &(s[i++]), 1, &cnt, PETSC_CHAR);CHKERRQ(ierr); if (!cnt) break;}
        while (s[i-1]!='\n' && s[i-1]!='\t' && s[i-1]!=' ' && s[i-1]!='\0' && s[i-1]!='\v' && s[i-1]!='\f' && s[i-1]!='\r');
        /* Terminate final string */
        if (c == num-1) s[i-1] = '\0';
      }
    } else {
      /* Read until a \n is encountered (-num is the max size allowed) */
      do {ierr = (*viewer->ops->read)(viewer, &(s[i++]), 1, &cnt, PETSC_CHAR);CHKERRQ(ierr); if (i == -num || !cnt) break;}
      while (s[i-1]!='\n');
      /* Terminate final string */
      s[i-1] = '\0';
      c      = i;
    }
    if (count) *count = c;
    else if (c < num) SETERRQ2(PetscObjectComm((PetscObject) viewer), PETSC_ERR_FILE_READ, "Insufficient data, only read %D < %D strings", c, num);
  } else {
    ierr = (*viewer->ops->read)(viewer, data, num, count, dtype);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
   PetscViewerReadable - Return a flag whether the viewer can be read from

   Not Collective

   Input Parameters:
.  viewer - the PetscViewer context

   Output Parameters:
.  flg - PETSC_TRUE if the viewer is readable, PETSC_FALSE otherwise

   Notes:
   PETSC_TRUE means that viewer's PetscViewerType supports reading (this holds e.g. for PETSCVIEWERBINARY)
   and viewer is in a mode allowing reading, i.e. PetscViewerFileGetMode()
   returns one of FILE_MODE_READ, FILE_MODE_UPDATE, FILE_MODE_APPEND_UPDATE.

   Level: intermediate

.seealso: PetscViewerWritable(), PetscViewerCheckReadable(), PetscViewerCreate(), PetscViewerFileSetMode(), PetscViewerFileSetType()
@*/
PetscErrorCode  PetscViewerReadable(PetscViewer viewer, PetscBool *flg)
{
  PetscErrorCode    ierr;
  PetscFileMode     mode;
  PetscErrorCode    (*f)(PetscViewer,PetscFileMode*) = NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  PetscValidBoolPointer(flg,2);
  ierr = PetscObjectQueryFunction((PetscObject)viewer, "PetscViewerFileGetMode_C", &f);CHKERRQ(ierr);
  *flg = PETSC_FALSE;
  if (!f) PetscFunctionReturn(0);
  ierr = (*f)(viewer, &mode);CHKERRQ(ierr);
  switch (mode) {
    case FILE_MODE_READ:
    case FILE_MODE_UPDATE:
    case FILE_MODE_APPEND_UPDATE:
      *flg = PETSC_TRUE;
    default: break;
  }
  PetscFunctionReturn(0);
}

/*@
   PetscViewerWritable - Return a flag whether the viewer can be written to

   Not Collective

   Input Parameters:
.  viewer - the PetscViewer context

   Output Parameters:
.  flg - PETSC_TRUE if the viewer is writable, PETSC_FALSE otherwise

   Notes:
   PETSC_TRUE means viewer is in a mode allowing writing, i.e. PetscViewerFileGetMode()
   returns one of FILE_MODE_WRITE, FILE_MODE_APPEND, FILE_MODE_UPDATE, FILE_MODE_APPEND_UPDATE.

   Level: intermediate

.seealso: PetscViewerReadable(), PetscViewerCheckWritable(), PetscViewerCreate(), PetscViewerFileSetMode(), PetscViewerFileSetType()
@*/
PetscErrorCode  PetscViewerWritable(PetscViewer viewer, PetscBool *flg)
{
  PetscErrorCode    ierr;
  PetscFileMode     mode;
  PetscErrorCode    (*f)(PetscViewer,PetscFileMode*) = NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  PetscValidBoolPointer(flg,2);
  ierr = PetscObjectQueryFunction((PetscObject)viewer, "PetscViewerFileGetMode_C", &f);CHKERRQ(ierr);
  *flg = PETSC_TRUE;
  if (!f) PetscFunctionReturn(0);
  ierr = (*f)(viewer, &mode);CHKERRQ(ierr);
  if (mode == FILE_MODE_READ) *flg = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@
   PetscViewerCheckReadable - Check whether the viewer can be read from

   Collective

   Input Parameters:
.  viewer - the PetscViewer context

   Level: intermediate

.seealso: PetscViewerReadable(), PetscViewerCheckWritable(), PetscViewerCreate(), PetscViewerFileSetMode(), PetscViewerFileSetType()
@*/
PetscErrorCode  PetscViewerCheckReadable(PetscViewer viewer)
{
  PetscBool         flg;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  ierr = PetscViewerReadable(viewer, &flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PetscObjectComm((PetscObject)viewer), PETSC_ERR_SUP, "Viewer doesn't support reading, or is not in reading mode (FILE_MODE_READ, FILE_MODE_UPDATE, FILE_MODE_APPEND_UPDATE)");
  PetscFunctionReturn(0);
}

/*@
   PetscViewerCheckWritable - Check whether the viewer can be written to

   Collective

   Input Parameters:
.  viewer - the PetscViewer context

   Level: intermediate

.seealso: PetscViewerWritable(), PetscViewerCheckReadable(), PetscViewerCreate(), PetscViewerFileSetMode(), PetscViewerFileSetType()
@*/
PetscErrorCode  PetscViewerCheckWritable(PetscViewer viewer)
{
  PetscBool         flg;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  ierr = PetscViewerWritable(viewer, &flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PetscObjectComm((PetscObject)viewer), PETSC_ERR_SUP, "Viewer doesn't support writing, or is in FILE_MODE_READ mode");
  PetscFunctionReturn(0);
}
