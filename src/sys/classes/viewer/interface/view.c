
#include <petsc/private/viewerimpl.h>  /*I "petscviewer.h" I*/
#include <petscdraw.h>

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
  PetscFunctionBegin;
  if (Petsc_Viewer_keyval != MPI_KEYVAL_INVALID) {
    CHKERRMPI(MPI_Comm_free_keyval(&Petsc_Viewer_keyval));
  }
  if (Petsc_Viewer_Stdout_keyval != MPI_KEYVAL_INVALID) {
    CHKERRMPI(MPI_Comm_free_keyval(&Petsc_Viewer_Stdout_keyval));
  }
  if (Petsc_Viewer_Stderr_keyval != MPI_KEYVAL_INVALID) {
    CHKERRMPI(MPI_Comm_free_keyval(&Petsc_Viewer_Stderr_keyval));
  }
  if (Petsc_Viewer_Binary_keyval != MPI_KEYVAL_INVALID) {
    CHKERRMPI(MPI_Comm_free_keyval(&Petsc_Viewer_Binary_keyval));
  }
  if (Petsc_Viewer_Draw_keyval != MPI_KEYVAL_INVALID) {
    CHKERRMPI(MPI_Comm_free_keyval(&Petsc_Viewer_Draw_keyval));
  }
#if defined(PETSC_HAVE_HDF5)
  if (Petsc_Viewer_HDF5_keyval != MPI_KEYVAL_INVALID) {
    CHKERRMPI(MPI_Comm_free_keyval(&Petsc_Viewer_HDF5_keyval));
  }
#endif
#if defined(PETSC_USE_SOCKETVIEWER)
  if (Petsc_Viewer_Socket_keyval != MPI_KEYVAL_INVALID) {
    CHKERRMPI(MPI_Comm_free_keyval(&Petsc_Viewer_Socket_keyval));
  }
#endif
  CHKERRQ(PetscFunctionListDestroy(&PetscViewerList));
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

  PetscFunctionBegin;
  if (PetscViewerPackageInitialized) PetscFunctionReturn(0);
  PetscViewerPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  CHKERRQ(PetscClassIdRegister("Viewer",&PETSC_VIEWER_CLASSID));
  /* Register Constructors */
  CHKERRQ(PetscViewerRegisterAll());
  /* Process Info */
  {
    PetscClassId  classids[1];

    classids[0] = PETSC_VIEWER_CLASSID;
    CHKERRQ(PetscInfoProcessClass("viewer", 1, classids));
  }
  /* Process summary exclusions */
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-log_exclude",logList,sizeof(logList),&opt));
  if (opt) {
    CHKERRQ(PetscStrInList("viewer",logList,',',&pkg));
    if (pkg) CHKERRQ(PetscLogEventExcludeClass(PETSC_VIEWER_CLASSID));
  }
#if defined(PETSC_HAVE_MATHEMATICA)
  CHKERRQ(PetscViewerMathematicaInitializePackage());
#endif
  /* Register package finalizer */
  CHKERRQ(PetscRegisterFinalize(PetscViewerFinalizePackage));
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
  PetscFunctionBegin;
  if (!*viewer) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*viewer,PETSC_VIEWER_CLASSID,1);

  CHKERRQ(PetscViewerFlush(*viewer));
  if (--((PetscObject)(*viewer))->refct > 0) {*viewer = NULL; PetscFunctionReturn(0);}

  CHKERRQ(PetscObjectSAWsViewOff((PetscObject)*viewer));
  if ((*viewer)->ops->destroy) {
    CHKERRQ((*(*viewer)->ops->destroy)(*viewer));
  }
  CHKERRQ(PetscHeaderDestroy(viewer));
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
PetscErrorCode PetscViewerAndFormatCreate(PetscViewer viewer, PetscViewerFormat format, PetscViewerAndFormat **vf)
{
  PetscFunctionBegin;
  CHKERRQ(PetscObjectReference((PetscObject)viewer));
  CHKERRQ(PetscNew(vf));
  (*vf)->viewer = viewer;
  (*vf)->format = format;
  (*vf)->lg     = NULL;
  (*vf)->data   = NULL;
  PetscFunctionReturn(0);
}

/*@C
   PetscViewerAndFormatDestroy - Destroys a PetscViewerAndFormat struct.

   Collective on PetscViewer

   Input Parameters:
.  vf - the PetscViewerAndFormat to be destroyed.

   Level: developer

.seealso: PetscViewerSocketOpen(), PetscViewerASCIIOpen(), PetscViewerCreate(), PetscViewerDrawOpen(), PetscViewerAndFormatCreate()
@*/
PetscErrorCode PetscViewerAndFormatDestroy(PetscViewerAndFormat **vf)
{
  PetscFunctionBegin;
  CHKERRQ(PetscViewerDestroy(&(*vf)->viewer));
  CHKERRQ(PetscDrawLGDestroy(&(*vf)->lg));
  CHKERRQ(PetscFree(*vf));
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

   Input Parameters:
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject)viewer,prefix));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  CHKERRQ(PetscObjectAppendOptionsPrefix((PetscObject)viewer,prefix));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  CHKERRQ(PetscObjectGetOptionsPrefix((PetscObject)viewer,prefix));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  if (viewer->setupcalled) PetscFunctionReturn(0);
  if (viewer->ops->setup) {
    CHKERRQ((*viewer->ops->setup)(viewer));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,PETSC_VIEWER_CLASSID,1);
  CHKERRQ(PetscObjectViewFromOptions((PetscObject)A,obj,name));
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
  PetscBool         iascii;
  PetscViewerFormat format;
#if defined(PETSC_HAVE_SAWS)
  PetscBool         issaws;
#endif

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,PETSC_VIEWER_CLASSID,1);
  PetscValidType(v,1);
  if (!viewer) {
    CHKERRQ(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)v),&viewer));
  }
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(v,1,viewer,2);

  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
#if defined(PETSC_HAVE_SAWS)
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERSAWS,&issaws));
#endif
  if (iascii) {
    CHKERRQ(PetscViewerGetFormat(viewer,&format));
    CHKERRQ(PetscObjectPrintClassNamePrefixType((PetscObject)v,viewer));
    if (format == PETSC_VIEWER_DEFAULT || format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      if (v->format) {
        CHKERRQ(PetscViewerASCIIPrintf(viewer,"  Viewer format = %s\n",PetscViewerFormats[v->format]));
      }
      CHKERRQ(PetscViewerASCIIPushTab(viewer));
      if (v->ops->view) {
        CHKERRQ((*v->ops->view)(v,viewer));
      }
      CHKERRQ(PetscViewerASCIIPopTab(viewer));
    }
#if defined(PETSC_HAVE_SAWS)
  } else if (issaws) {
    if (!((PetscObject)v)->amsmem) {
      CHKERRQ(PetscObjectViewSAWs((PetscObject)v,viewer));
      if (v->ops->view) {
        CHKERRQ((*v->ops->view)(v,viewer));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  if (dtype == PETSC_STRING) {
    PetscInt c, i = 0, cnt;
    char *s = (char *)data;
    if (num >= 0) {
      for (c = 0; c < num; c++) {
        /* Skip leading whitespaces */
        do {CHKERRQ((*viewer->ops->read)(viewer, &(s[i]), 1, &cnt, PETSC_CHAR)); if (!cnt) break;}
        while (s[i]=='\n' || s[i]=='\t' || s[i]==' ' || s[i]=='\0' || s[i]=='\v' || s[i]=='\f' || s[i]=='\r');
        i++;
        /* Read strings one char at a time */
        do {CHKERRQ((*viewer->ops->read)(viewer, &(s[i++]), 1, &cnt, PETSC_CHAR)); if (!cnt) break;}
        while (s[i-1]!='\n' && s[i-1]!='\t' && s[i-1]!=' ' && s[i-1]!='\0' && s[i-1]!='\v' && s[i-1]!='\f' && s[i-1]!='\r');
        /* Terminate final string */
        if (c == num-1) s[i-1] = '\0';
      }
    } else {
      /* Read until a \n is encountered (-num is the max size allowed) */
      do {CHKERRQ((*viewer->ops->read)(viewer, &(s[i++]), 1, &cnt, PETSC_CHAR)); if (i == -num || !cnt) break;}
      while (s[i-1]!='\n');
      /* Terminate final string */
      s[i-1] = '\0';
      c      = i;
    }
    if (count) *count = c;
    else PetscCheckFalse(c < num,PetscObjectComm((PetscObject) viewer), PETSC_ERR_FILE_READ, "Insufficient data, only read %" PetscInt_FMT " < %" PetscInt_FMT " strings", c, num);
  } else {
    CHKERRQ((*viewer->ops->read)(viewer, data, num, count, dtype));
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
  PetscFileMode     mode;
  PetscErrorCode    (*f)(PetscViewer,PetscFileMode*) = NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  PetscValidBoolPointer(flg,2);
  CHKERRQ(PetscObjectQueryFunction((PetscObject)viewer, "PetscViewerFileGetMode_C", &f));
  *flg = PETSC_FALSE;
  if (!f) PetscFunctionReturn(0);
  CHKERRQ((*f)(viewer, &mode));
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
  PetscFileMode     mode;
  PetscErrorCode    (*f)(PetscViewer,PetscFileMode*) = NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  PetscValidBoolPointer(flg,2);
  CHKERRQ(PetscObjectQueryFunction((PetscObject)viewer, "PetscViewerFileGetMode_C", &f));
  *flg = PETSC_TRUE;
  if (!f) PetscFunctionReturn(0);
  CHKERRQ((*f)(viewer, &mode));
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  CHKERRQ(PetscViewerReadable(viewer, &flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)viewer), PETSC_ERR_SUP, "Viewer doesn't support reading, or is not in reading mode (FILE_MODE_READ, FILE_MODE_UPDATE, FILE_MODE_APPEND_UPDATE)");
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  CHKERRQ(PetscViewerWritable(viewer, &flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)viewer), PETSC_ERR_SUP, "Viewer doesn't support writing, or is in FILE_MODE_READ mode");
  PetscFunctionReturn(0);
}
