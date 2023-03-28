
#include <petsc/private/viewerimpl.h> /*I "petscviewer.h" I*/
#include <petscdraw.h>

PetscClassId PETSC_VIEWER_CLASSID;

static PetscBool PetscViewerPackageInitialized = PETSC_FALSE;
/*@C
  PetscViewerFinalizePackage - This function destroys any global objects created in the Petsc viewers. It is
  called from `PetscFinalize()`.

  Level: developer

.seealso: [](sec_viewers), `PetscViewer`, `PetscFinalize()`, `PetscViewerInitializePackage()`
@*/
PetscErrorCode PetscViewerFinalizePackage(void)
{
  PetscFunctionBegin;
  if (Petsc_Viewer_keyval != MPI_KEYVAL_INVALID) PetscCallMPI(MPI_Comm_free_keyval(&Petsc_Viewer_keyval));
  if (Petsc_Viewer_Stdout_keyval != MPI_KEYVAL_INVALID) PetscCallMPI(MPI_Comm_free_keyval(&Petsc_Viewer_Stdout_keyval));
  if (Petsc_Viewer_Stderr_keyval != MPI_KEYVAL_INVALID) PetscCallMPI(MPI_Comm_free_keyval(&Petsc_Viewer_Stderr_keyval));
  if (Petsc_Viewer_Binary_keyval != MPI_KEYVAL_INVALID) PetscCallMPI(MPI_Comm_free_keyval(&Petsc_Viewer_Binary_keyval));
  if (Petsc_Viewer_Draw_keyval != MPI_KEYVAL_INVALID) PetscCallMPI(MPI_Comm_free_keyval(&Petsc_Viewer_Draw_keyval));
#if defined(PETSC_HAVE_HDF5)
  if (Petsc_Viewer_HDF5_keyval != MPI_KEYVAL_INVALID) PetscCallMPI(MPI_Comm_free_keyval(&Petsc_Viewer_HDF5_keyval));
#endif
#if defined(PETSC_USE_SOCKETVIEWER)
  if (Petsc_Viewer_Socket_keyval != MPI_KEYVAL_INVALID) PetscCallMPI(MPI_Comm_free_keyval(&Petsc_Viewer_Socket_keyval));
#endif
  PetscCall(PetscFunctionListDestroy(&PetscViewerList));
  PetscViewerPackageInitialized = PETSC_FALSE;
  PetscViewerRegisterAllCalled  = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscViewerInitializePackage - This function initializes everything in the `PetscViewer` package.

  Level: developer

.seealso: [](sec_viewers), `PetscViewer`, `PetscInitialize()`, `PetscViewerFinalizePackage()`
@*/
PetscErrorCode PetscViewerInitializePackage(void)
{
  char      logList[256];
  PetscBool opt, pkg;

  PetscFunctionBegin;
  if (PetscViewerPackageInitialized) PetscFunctionReturn(PETSC_SUCCESS);
  PetscViewerPackageInitialized = PETSC_TRUE;
  /* Register Classes */
  PetscCall(PetscClassIdRegister("Viewer", &PETSC_VIEWER_CLASSID));
  /* Register Constructors */
  PetscCall(PetscViewerRegisterAll());
  /* Process Info */
  {
    PetscClassId classids[1];

    classids[0] = PETSC_VIEWER_CLASSID;
    PetscCall(PetscInfoProcessClass("viewer", 1, classids));
  }
  /* Process summary exclusions */
  PetscCall(PetscOptionsGetString(NULL, NULL, "-log_exclude", logList, sizeof(logList), &opt));
  if (opt) {
    PetscCall(PetscStrInList("viewer", logList, ',', &pkg));
    if (pkg) PetscCall(PetscLogEventExcludeClass(PETSC_VIEWER_CLASSID));
  }
#if defined(PETSC_HAVE_MATHEMATICA)
  PetscCall(PetscViewerMathematicaInitializePackage());
#endif
  /* Register package finalizer */
  PetscCall(PetscRegisterFinalize(PetscViewerFinalizePackage));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   PetscViewerDestroy - Destroys a `PetscViewer`.

   Collective

   Input Parameter:
.  viewer - the `PetscViewer` to be destroyed.

   Level: beginner

.seealso: [](sec_viewers), `PetscViewer`, `PetscViewerSocketOpen()`, `PetscViewerASCIIOpen()`, `PetscViewerCreate()`, `PetscViewerDrawOpen()`
@*/
PetscErrorCode PetscViewerDestroy(PetscViewer *viewer)
{
  PetscFunctionBegin;
  if (!*viewer) PetscFunctionReturn(PETSC_SUCCESS);
  PetscValidHeaderSpecific(*viewer, PETSC_VIEWER_CLASSID, 1);

  PetscCall(PetscViewerFlush(*viewer));
  if (--((PetscObject)(*viewer))->refct > 0) {
    *viewer = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(PetscObjectSAWsViewOff((PetscObject)*viewer));
  if ((*viewer)->ops->destroy) PetscCall((*(*viewer)->ops->destroy)(*viewer));
  PetscCall(PetscHeaderDestroy(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscViewerAndFormatCreate - Creates a `PetscViewerAndFormat` struct.

   Collective

   Input Parameters:
+  viewer - the viewer
-  format - the format

   Output Parameter:
.   vf - viewer and format object

   Level: developer

   Notes:
   This increases the reference count of the viewer.

   Use `PetscViewerAndFormatDestroy()` to free the struct

   This is used as the context variable for many of the `TS`, `SNES`, and `KSP` monitor functions

   This construct exists because it allows one to keep track of the use of a `PetscViewerFormat` without requiring the
   format in the viewer to be permanently changed.

.seealso: [](sec_viewers), `PetscViewerFormat`, `PetscViewerSocketOpen()`, `PetscViewerASCIIOpen()`, `PetscViewerCreate()`,
          `PetscViewerDrawOpen()`, `PetscViewerAndFormatDestroy()`
@*/
PetscErrorCode PetscViewerAndFormatCreate(PetscViewer viewer, PetscViewerFormat format, PetscViewerAndFormat **vf)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectReference((PetscObject)viewer));
  PetscCall(PetscNew(vf));
  (*vf)->viewer = viewer;
  (*vf)->format = format;
  (*vf)->lg     = NULL;
  (*vf)->data   = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscViewerAndFormatDestroy - Destroys a `PetscViewerAndFormat` struct created with `PetscViewerAndFormatCreate()`

   Collective

   Input Parameter:
.  vf - the `PetscViewerAndFormat` to be destroyed.

   Level: developer

.seealso: [](sec_viewers), `PetscViewerAndFormatCreate()`, `PetscViewerSocketOpen()`, `PetscViewerASCIIOpen()`, `PetscViewerCreate()`,
          `PetscViewerDrawOpen()`, `PetscViewerAndFormatDestroy()`
@*/
PetscErrorCode PetscViewerAndFormatDestroy(PetscViewerAndFormat **vf)
{
  PetscFunctionBegin;
  PetscCall(PetscViewerDestroy(&(*vf)->viewer));
  PetscCall(PetscDrawLGDestroy(&(*vf)->lg));
  PetscCall(PetscFree(*vf));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscViewerGetType - Returns the type of a `PetscViewer`.

   Not Collective

   Input Parameter:
.   viewer - the `PetscViewer`

   Output Parameter:
.  type - `PetscViewerType`

   Available Types Include:
+  `PETSCVIEWERSOCKET` - Socket PetscViewer
.  `PETSCVIEWERASCII` - ASCII PetscViewer
.  `PETSCVIEWERBINARY` - binary file PetscViewer
.  `PETSCVIEWERSTRING` - string PetscViewer
-  `PETSCVIEWERDRAW` - drawing PetscViewer

   Level: intermediate

   Note:
   `PetscViewerType` is actually a string

.seealso: [](sec_viewers), `PetscViewerType`, `PetscViewer`, `PetscViewerCreate()`, `PetscViewerSetType()`, `PetscViewerType`
@*/
PetscErrorCode PetscViewerGetType(PetscViewer viewer, PetscViewerType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscValidPointer(type, 2);
  *type = ((PetscObject)viewer)->type_name;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscViewerSetOptionsPrefix - Sets the prefix used for searching for all
   `PetscViewer` options in the database.

   Logically Collective

   Input Parameters:
+  viewer - the `PetscViewer` context
-  prefix - the prefix to prepend to all option names

   Note:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

   Level: advanced

.seealso: [](sec_viewers), `PetscViewer`, `PetscViewerSetFromOptions()`
@*/
PetscErrorCode PetscViewerSetOptionsPrefix(PetscViewer viewer, const char prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)viewer, prefix));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscViewerAppendOptionsPrefix - Appends to the prefix used for searching for all
   `PetscViewer` options in the database.

   Logically Collective

   Input Parameters:
+  viewer - the `PetscViewer` context
-  prefix - the prefix to prepend to all option names

   Level: advanced

   Note:
   A hyphen (-) must NOT be given at the beginning of the prefix name.
   The first character of all runtime options is AUTOMATICALLY the hyphen.

.seealso: [](sec_viewers), `PetscViewer`, `PetscViewerGetOptionsPrefix()`
@*/
PetscErrorCode PetscViewerAppendOptionsPrefix(PetscViewer viewer, const char prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscCall(PetscObjectAppendOptionsPrefix((PetscObject)viewer, prefix));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscViewerGetOptionsPrefix - Sets the prefix used for searching for all
   `PetscViewer` options in the database.

   Not Collective

   Input Parameter:
.  viewer - the `PetscViewer` context

   Output Parameter:
.  prefix - pointer to the prefix string used

   Level: advanced

   Fortran Note:
   The user should pass in a string 'prefix' of sufficient length to hold the prefix.

.seealso: [](sec_viewers), `PetscViewer`, `PetscViewerAppendOptionsPrefix()`
@*/
PetscErrorCode PetscViewerGetOptionsPrefix(PetscViewer viewer, const char *prefix[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)viewer, prefix));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   PetscViewerSetUp - Sets up the internal viewer data structures for the later use.

   Collective

   Input Parameter:
.  viewer - the `PetscViewer` context

   Level: advanced

   Note:
   For basic use of the `PetscViewer` classes the user need not explicitly call
   `PetscViewerSetUp()`, since these actions will happen automatically.

.seealso: [](sec_viewers), `PetscViewer`, `PetscViewerCreate()`, `PetscViewerDestroy()`
@*/
PetscErrorCode PetscViewerSetUp(PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  if (viewer->setupcalled) PetscFunctionReturn(PETSC_SUCCESS);
  PetscTryTypeMethod(viewer, setup);
  viewer->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscViewerViewFromOptions - View from the viewer based on options in the options database

   Collective

   Input Parameters:
+  A - the `PetscViewer` context
.  obj - Optional object that provides the prefix for the option names
-  name - command line option

   Level: intermediate

   Note:
   See `PetscObjectViewFromOptions()` for details on the viewers and formats support via this interface

.seealso: [](sec_viewers), `PetscViewer`, `PetscViewerView`, `PetscObjectViewFromOptions()`, `PetscViewerCreate()`
@*/
PetscErrorCode PetscViewerViewFromOptions(PetscViewer A, PetscObject obj, const char name[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, PETSC_VIEWER_CLASSID, 1);
  PetscCall(PetscObjectViewFromOptions((PetscObject)A, obj, name));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscViewerView - Visualizes a viewer object.

   Collective

   Input Parameters:
+  v - the viewer to be viewed
-  viewer - visualization context

   Level: beginner

.seealso: [](sec_viewers), `PetscViewer`, `PetscViewerPushFormat()`, `PetscViewerASCIIOpen()`, `PetscViewerDrawOpen()`,
          `PetscViewerSocketOpen()`, `PetscViewerBinaryOpen()`, `PetscViewerLoad()`
@*/
PetscErrorCode PetscViewerView(PetscViewer v, PetscViewer viewer)
{
  PetscBool         iascii;
  PetscViewerFormat format;
#if defined(PETSC_HAVE_SAWS)
  PetscBool issaws;
#endif

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v, PETSC_VIEWER_CLASSID, 1);
  PetscValidType(v, 1);
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)v), &viewer));
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCheckSameComm(v, 1, viewer, 2);

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
#if defined(PETSC_HAVE_SAWS)
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERSAWS, &issaws));
#endif
  if (iascii) {
    PetscCall(PetscViewerGetFormat(viewer, &format));
    PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)v, viewer));
    if (format == PETSC_VIEWER_DEFAULT || format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      if (v->format) PetscCall(PetscViewerASCIIPrintf(viewer, "  Viewer format = %s\n", PetscViewerFormats[v->format]));
      PetscCall(PetscViewerASCIIPushTab(viewer));
      PetscTryTypeMethod(v, view, viewer);
      PetscCall(PetscViewerASCIIPopTab(viewer));
    }
#if defined(PETSC_HAVE_SAWS)
  } else if (issaws) {
    if (!((PetscObject)v)->amsmem) {
      PetscCall(PetscObjectViewSAWs((PetscObject)v, viewer));
      PetscTryTypeMethod(v, view, viewer);
    }
#endif
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscViewerRead - Reads data from a `PetscViewer`

   Collective

   Input Parameters:
+  viewer   - The viewer
.  data     - Location to write the data
.  num      - Number of items of data to read
-  datatype - Type of data to read

   Output Parameter:
.  count - number of items of data actually read, or `NULL`

   Level: beginner

   Notes:
   If datatype is `PETSC_STRING` and num is negative, reads until a newline character is found,
   until a maximum of (-num - 1) chars.

   Only certain viewers, such as `PETSCVIEWERBINARY` can be read from, see `PetscViewerReadable()`

.seealso: [](sec_viewers), `PetscViewer`, `PetscViewerASCIIOpen()`, `PetscViewerPushFormat()`, `PetscViewerDestroy()`,
          `PetscViewerReadable()`, `PetscViewerBinaryGetDescriptor()`,
          `PetscViewerBinaryGetInfoPointer()`, `PetscFileMode`, `PetscViewer`
@*/
PetscErrorCode PetscViewerRead(PetscViewer viewer, void *data, PetscInt num, PetscInt *count, PetscDataType dtype)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  if (dtype == PETSC_STRING) {
    PetscInt c, i = 0, cnt;
    char    *s = (char *)data;
    if (num >= 0) {
      for (c = 0; c < num; c++) {
        /* Skip leading whitespaces */
        do {
          PetscCall((*viewer->ops->read)(viewer, &(s[i]), 1, &cnt, PETSC_CHAR));
          if (!cnt) break;
        } while (s[i] == '\n' || s[i] == '\t' || s[i] == ' ' || s[i] == '\0' || s[i] == '\v' || s[i] == '\f' || s[i] == '\r');
        i++;
        /* Read strings one char at a time */
        do {
          PetscCall((*viewer->ops->read)(viewer, &(s[i++]), 1, &cnt, PETSC_CHAR));
          if (!cnt) break;
        } while (s[i - 1] != '\n' && s[i - 1] != '\t' && s[i - 1] != ' ' && s[i - 1] != '\0' && s[i - 1] != '\v' && s[i - 1] != '\f' && s[i - 1] != '\r');
        /* Terminate final string */
        if (c == num - 1) s[i - 1] = '\0';
      }
    } else {
      /* Read until a \n is encountered (-num is the max size allowed) */
      do {
        PetscCall((*viewer->ops->read)(viewer, &(s[i++]), 1, &cnt, PETSC_CHAR));
        if (i == -num || !cnt) break;
      } while (s[i - 1] != '\n');
      /* Terminate final string */
      s[i - 1] = '\0';
      c        = i;
    }
    if (count) *count = c;
    else PetscCheck(c >= num, PetscObjectComm((PetscObject)viewer), PETSC_ERR_FILE_READ, "Insufficient data, only read %" PetscInt_FMT " < %" PetscInt_FMT " strings", c, num);
  } else PetscUseTypeMethod(viewer, read, data, num, count, dtype);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   PetscViewerReadable - Return a flag whether the viewer can be read from

   Not Collective

   Input Parameter:
.  viewer - the `PetscViewer` context

   Output Parameter:
.  flg - `PETSC_TRUE` if the viewer is readable, `PETSC_FALSE` otherwise

   Level: intermediate

   Note:
   `PETSC_TRUE` means that viewer's `PetscViewerType` supports reading (this holds e.g. for `PETSCVIEWERBINARY`)
   and viewer is in a mode allowing reading, i.e. `PetscViewerFileGetMode()`
   returns one of `FILE_MODE_READ`, `FILE_MODE_UPDATE`, `FILE_MODE_APPEND_UPDATE`.

.seealso: [](sec_viewers), `PetscViewerRead()`, `PetscViewer`, `PetscViewerWritable()`, `PetscViewerCheckReadable()`, `PetscViewerCreate()`, `PetscViewerFileSetMode()`, `PetscViewerFileSetType()`
@*/
PetscErrorCode PetscViewerReadable(PetscViewer viewer, PetscBool *flg)
{
  PetscFileMode mode;
  PetscErrorCode (*f)(PetscViewer, PetscFileMode *) = NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscValidBoolPointer(flg, 2);
  PetscCall(PetscObjectQueryFunction((PetscObject)viewer, "PetscViewerFileGetMode_C", &f));
  *flg = PETSC_FALSE;
  if (!f) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall((*f)(viewer, &mode));
  switch (mode) {
  case FILE_MODE_READ:
  case FILE_MODE_UPDATE:
  case FILE_MODE_APPEND_UPDATE:
    *flg = PETSC_TRUE;
  default:
    break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   PetscViewerWritable - Return a flag whether the viewer can be written to

   Not Collective

   Input Parameter:
.  viewer - the `PetscViewer` context

   Output Parameter:
.  flg - `PETSC_TRUE` if the viewer is writable, `PETSC_FALSE` otherwise

   Level: intermediate

   Note:
   `PETSC_TRUE` means viewer is in a mode allowing writing, i.e. `PetscViewerFileGetMode()`
   returns one of `FILE_MODE_WRITE`, `FILE_MODE_APPEND`, `FILE_MODE_UPDATE`, `FILE_MODE_APPEND_UPDATE`.

.seealso: [](sec_viewers), `PetscViewer`, `PetscViewerReadable()`, `PetscViewerCheckWritable()`, `PetscViewerCreate()`, `PetscViewerFileSetMode()`, `PetscViewerFileSetType()`
@*/
PetscErrorCode PetscViewerWritable(PetscViewer viewer, PetscBool *flg)
{
  PetscFileMode mode;
  PetscErrorCode (*f)(PetscViewer, PetscFileMode *) = NULL;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscValidBoolPointer(flg, 2);
  PetscCall(PetscObjectQueryFunction((PetscObject)viewer, "PetscViewerFileGetMode_C", &f));
  *flg = PETSC_TRUE;
  if (!f) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall((*f)(viewer, &mode));
  if (mode == FILE_MODE_READ) *flg = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   PetscViewerCheckReadable - Check whether the viewer can be read from, generates an error if not

   Collective

   Input Parameter:
.  viewer - the `PetscViewer` context

   Level: intermediate

.seealso: [](sec_viewers), `PetscViewer`, `PetscViewerReadable()`, `PetscViewerCheckWritable()`, `PetscViewerCreate()`, `PetscViewerFileSetMode()`, `PetscViewerFileSetType()`
@*/
PetscErrorCode PetscViewerCheckReadable(PetscViewer viewer)
{
  PetscBool flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscCall(PetscViewerReadable(viewer, &flg));
  PetscCheck(flg, PetscObjectComm((PetscObject)viewer), PETSC_ERR_SUP, "Viewer doesn't support reading, or is not in reading mode (FILE_MODE_READ, FILE_MODE_UPDATE, FILE_MODE_APPEND_UPDATE)");
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   PetscViewerCheckWritable - Check whether the viewer can be written to, generates an error if not

   Collective

   Input Parameter:
.  viewer - the `PetscViewer` context

   Level: intermediate

.seealso: [](sec_viewers), `PetscViewer`, `PetscViewerWritable()`, `PetscViewerCheckReadable()`, `PetscViewerCreate()`, `PetscViewerFileSetMode()`, `PetscViewerFileSetType()`
@*/
PetscErrorCode PetscViewerCheckWritable(PetscViewer viewer)
{
  PetscBool flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscCall(PetscViewerWritable(viewer, &flg));
  PetscCheck(flg, PetscObjectComm((PetscObject)viewer), PETSC_ERR_SUP, "Viewer doesn't support writing, or is in FILE_MODE_READ mode");
  PetscFunctionReturn(PETSC_SUCCESS);
}
