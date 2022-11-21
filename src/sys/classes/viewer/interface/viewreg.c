
#include <petsc/private/viewerimpl.h> /*I "petscviewer.h" I*/
#include <petsc/private/hashtable.h>
#if defined(PETSC_HAVE_SAWS)
  #include <petscviewersaws.h>
#endif

PetscFunctionList PetscViewerList = NULL;

PetscOptionsHelpPrinted PetscOptionsHelpPrintedSingleton = NULL;
KHASH_SET_INIT_STR(HTPrinted)
struct _n_PetscOptionsHelpPrinted {
  khash_t(HTPrinted) *printed;
  PetscSegBuffer      strings;
};

PetscErrorCode PetscOptionsHelpPrintedDestroy(PetscOptionsHelpPrinted *hp)
{
  PetscFunctionBegin;
  if (!*hp) PetscFunctionReturn(0);
  kh_destroy(HTPrinted, (*hp)->printed);
  PetscCall(PetscSegBufferDestroy(&(*hp)->strings));
  PetscCall(PetscFree(*hp));
  PetscFunctionReturn(0);
}

/*@C
      PetscOptionsHelpPrintedCreate - Creates an object used to manage tracking which help messages have
         been printed so they will not be printed again.

     Not collective

    Level: developer

.seealso: `PetscOptionsHelpPrintedCheck()`, `PetscOptionsHelpPrintChecked()`
@*/
PetscErrorCode PetscOptionsHelpPrintedCreate(PetscOptionsHelpPrinted *hp)
{
  PetscFunctionBegin;
  PetscCall(PetscNew(hp));
  (*hp)->printed = kh_init(HTPrinted);
  PetscCall(PetscSegBufferCreate(sizeof(char), 10000, &(*hp)->strings));
  PetscFunctionReturn(0);
}

/*@C
      PetscOptionsHelpPrintedCheck - Checks if a particular pre, name pair has previous been entered (meaning the help message was printed)

     Not collective

    Input Parameters:
+     hp - the object used to manage tracking what help messages have been printed
.     pre - the prefix part of the string, many be NULL
-     name - the string to look for (cannot be NULL)

    Output Parameter:
.     found - PETSC_TRUE if the string was already set

    Level: intermediate

.seealso: `PetscOptionsHelpPrintedCreate()`
@*/
PetscErrorCode PetscOptionsHelpPrintedCheck(PetscOptionsHelpPrinted hp, const char *pre, const char *name, PetscBool *found)
{
  size_t l1, l2;
#if !defined(PETSC_HAVE_THREADSAFETY)
  char *both;
  int   newitem;
#endif

  PetscFunctionBegin;
  PetscCall(PetscStrlen(pre, &l1));
  PetscCall(PetscStrlen(name, &l2));
  if (l1 + l2 == 0) {
    *found = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
#if !defined(PETSC_HAVE_THREADSAFETY)
  PetscCall(PetscSegBufferGet(hp->strings, l1 + l2 + 1, &both));
  PetscCall(PetscStrcpy(both, pre));
  PetscCall(PetscStrcat(both, name));
  kh_put(HTPrinted, hp->printed, both, &newitem);
  if (!newitem) PetscCall(PetscSegBufferUnuse(hp->strings, l1 + l2 + 1));
  *found = newitem ? PETSC_FALSE : PETSC_TRUE;
#else
  *found = PETSC_FALSE;
#endif
  PetscFunctionReturn(0);
}

static PetscBool noviewer = PETSC_FALSE;
static PetscBool noviewers[PETSCVIEWERGETVIEWEROFFPUSHESMAX];
static PetscInt  inoviewers = 0;

/*@
  PetscOptionsPushGetViewerOff - sets if a `PetscOptionsGetViewer()` returns a viewer.

  Logically Collective

  Input Parameter:
. flg - `PETSC_TRUE` to turn off viewer creation, `PETSC_FALSE` to turn it on.

  Level: developer

  Note:
    Calling XXXViewFromOptions in an inner loop can be very expensive.  This can appear, for example, when using
   many small subsolves.  Call this function to control viewer creation in `PetscOptionsGetViewer()`, thus removing the expensive XXXViewFromOptions calls.

.seealso: `PetscOptionsGetViewer()`, `PetscOptionsPopGetViewerOff()`
@*/
PetscErrorCode PetscOptionsPushGetViewerOff(PetscBool flg)
{
  PetscFunctionBegin;
  PetscCheck(inoviewers < PETSCVIEWERGETVIEWEROFFPUSHESMAX, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Too many PetscOptionsPushGetViewerOff(), perhaps you forgot PetscOptionsPopGetViewerOff()?");

  noviewers[inoviewers++] = noviewer;
  noviewer                = flg;
  PetscFunctionReturn(0);
}

/*@
  PetscOptionsPopGetViewerOff - reset whether `PetscOptionsGetViewer()` returns a viewer.

  Logically Collective

  Level: developer

  Note:
    Calling XXXViewFromOptions in an inner loop can be very expensive.  This can appear, for example, when using
   many small subsolves.  Call this function to control viewer creation in `PetscOptionsGetViewer()`, thus removing the expensive XXXViewFromOptions calls.

.seealso: `PetscOptionsGetViewer()`, `PetscOptionsPushGetViewerOff()`
@*/
PetscErrorCode PetscOptionsPopGetViewerOff(void)
{
  PetscFunctionBegin;
  PetscCheck(inoviewers, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Too many PetscOptionsPopGetViewerOff(), perhaps you forgot PetscOptionsPushGetViewerOff()?");
  noviewer = noviewers[--inoviewers];
  PetscFunctionReturn(0);
}

/*@
  PetscOptionsGetViewerOff - does `PetscOptionsGetViewer()` return a viewer?

  Logically Collective

  Output Parameter:
. flg - whether viewers are returned.

  Level: developer

  Note:
    Calling XXXViewFromOptions in an inner loop can be very expensive.  This can appear, for example, when using
   many small subsolves.

.seealso: `PetscOptionsGetViewer()`, `PetscOptionsPushGetViewerOff()`, `PetscOptionsPopGetViewerOff()`
@*/
PetscErrorCode PetscOptionsGetViewerOff(PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidBoolPointer(flg, 1);
  *flg = noviewer;
  PetscFunctionReturn(0);
}

/*@C
   PetscOptionsGetViewer - Gets a viewer appropriate for the type indicated by the user

   Collective

   Input Parameters:
+  comm - the communicator to own the viewer
.  options - options database, use NULL for default global database
.  pre - the string to prepend to the name or NULL
-  name - the option one is seeking

   Output Parameters:
+  viewer - the viewer, pass NULL if not needed
.  format - the `PetscViewerFormat` requested by the user, pass NULL if not needed
-  set - `PETSC_TRUE` if found, else `PETSC_FALSE`

   Level: intermediate

   Notes:
    If no value is provided ascii:stdout is used
+       ascii[:[filename][:[format][:append]]]  -  defaults to stdout - format can be one of ascii_info, ascii_info_detail, or ascii_matlab,
                                                  for example ascii::ascii_info prints just the information about the object not all details
                                                  unless :append is given filename opens in write mode, overwriting what was already there
.       binary[:[filename][:[format][:append]]] -  defaults to the file binaryoutput
.       draw[:drawtype[:filename]]              -  for example, draw:tikz, draw:tikz:figure.tex  or draw:x
.       socket[:port]                           -  defaults to the standard output port
-       saws[:communicatorname]                 -   publishes object to the Scientific Application Webserver (SAWs)

   Use `PetscViewerDestroy()` after using the viewer, otherwise a memory leak will occur

   You can control whether calls to this function create a viewer (or return early with *set of `PETSC_FALSE`) with
   `PetscOptionsPushGetViewerOff()`.  This is useful if calling many small subsolves, in which case XXXViewFromOptions can take
   an appreciable fraction of the runtime.

   If PETSc is configured with --with-viewfromoptions=0 this function always returns with *set of `PETSC_FALSE`

.seealso: `PetscOptionsGetReal()`, `PetscOptionsHasName()`, `PetscOptionsGetString()`,
          `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsBool()`
          `PetscOptionsInt()`, `PetscOptionsString()`, `PetscOptionsReal()`, `PetscOptionsBool()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`, `PetscOptionsPushGetViewerOff()`, `PetscOptionsPopGetViewerOff()`,
          `PetscOptionsGetViewerOff()`
@*/
PetscErrorCode PetscOptionsGetViewer(MPI_Comm comm, PetscOptions options, const char pre[], const char name[], PetscViewer *viewer, PetscViewerFormat *format, PetscBool *set)
{
  const char *value;
  PetscBool   flag, hashelp;

  PetscFunctionBegin;
  PetscValidCharPointer(name, 4);

  if (viewer) *viewer = NULL;
  if (format) *format = PETSC_VIEWER_DEFAULT;
  if (set) *set = PETSC_FALSE;
  PetscCall(PetscOptionsGetViewerOff(&flag));
  if (flag) PetscFunctionReturn(0);

  PetscCall(PetscOptionsHasHelp(NULL, &hashelp));
  if (hashelp) {
    PetscBool found;

    if (!PetscOptionsHelpPrintedSingleton) PetscCall(PetscOptionsHelpPrintedCreate(&PetscOptionsHelpPrintedSingleton));
    PetscCall(PetscOptionsHelpPrintedCheck(PetscOptionsHelpPrintedSingleton, pre, name, &found));
    if (!found && viewer) {
      PetscCall((*PetscHelpPrintf)(comm, "----------------------------------------\nViewer (-%s%s) options:\n", pre ? pre : "", name + 1));
      PetscCall((*PetscHelpPrintf)(comm, "  -%s%s ascii[:[filename][:[format][:append]]]: %s (%s)\n", pre ? pre : "", name + 1, "Prints object to stdout or ASCII file", "PetscOptionsGetViewer"));
      PetscCall((*PetscHelpPrintf)(comm, "  -%s%s binary[:[filename][:[format][:append]]]: %s (%s)\n", pre ? pre : "", name + 1, "Saves object to a binary file", "PetscOptionsGetViewer"));
      PetscCall((*PetscHelpPrintf)(comm, "  -%s%s draw[:[drawtype][:filename|format]] %s (%s)\n", pre ? pre : "", name + 1, "Draws object", "PetscOptionsGetViewer"));
      PetscCall((*PetscHelpPrintf)(comm, "  -%s%s socket[:port]: %s (%s)\n", pre ? pre : "", name + 1, "Pushes object to a Unix socket", "PetscOptionsGetViewer"));
      PetscCall((*PetscHelpPrintf)(comm, "  -%s%s saws[:communicatorname]: %s (%s)\n", pre ? pre : "", name + 1, "Publishes object to SAWs", "PetscOptionsGetViewer"));
    }
  }

  if (format) *format = PETSC_VIEWER_DEFAULT;
  PetscCall(PetscOptionsFindPair(options, pre, name, &value, &flag));
  if (flag) {
    if (set) *set = PETSC_TRUE;
    if (!value) {
      if (viewer) {
        PetscCall(PetscViewerASCIIGetStdout(comm, viewer));
        PetscCall(PetscObjectReference((PetscObject)*viewer));
      }
    } else {
      char       *loc0_vtype, *loc1_fname, *loc2_fmt = NULL, *loc3_fmode = NULL;
      PetscInt    cnt;
      const char *viewers[] = {PETSCVIEWERASCII, PETSCVIEWERBINARY, PETSCVIEWERDRAW, PETSCVIEWERSOCKET, PETSCVIEWERMATLAB, PETSCVIEWERSAWS, PETSCVIEWERVTK, PETSCVIEWERHDF5, PETSCVIEWERGLVIS, PETSCVIEWEREXODUSII, NULL};

      PetscCall(PetscStrallocpy(value, &loc0_vtype));
      PetscCall(PetscStrchr(loc0_vtype, ':', &loc1_fname));
      if (loc1_fname) {
        *loc1_fname++ = 0;
        PetscCall(PetscStrchr(loc1_fname, ':', &loc2_fmt));
      }
      if (loc2_fmt) {
        *loc2_fmt++ = 0;
        PetscCall(PetscStrchr(loc2_fmt, ':', &loc3_fmode));
      }
      if (loc3_fmode) *loc3_fmode++ = 0;
      PetscCall(PetscStrendswithwhich(*loc0_vtype ? loc0_vtype : "ascii", viewers, &cnt));
      PetscCheck(cnt <= (PetscInt)sizeof(viewers) - 1, comm, PETSC_ERR_ARG_OUTOFRANGE, "Unknown viewer type: %s", loc0_vtype);
      if (viewer) {
        if (!loc1_fname) {
          switch (cnt) {
          case 0:
            PetscCall(PetscViewerASCIIGetStdout(comm, viewer));
            break;
          case 1:
            if (!(*viewer = PETSC_VIEWER_BINARY_(comm))) PetscCall(PETSC_ERR_PLIB);
            break;
          case 2:
            if (!(*viewer = PETSC_VIEWER_DRAW_(comm))) PetscCall(PETSC_ERR_PLIB);
            break;
#if defined(PETSC_USE_SOCKET_VIEWER)
          case 3:
            if (!(*viewer = PETSC_VIEWER_SOCKET_(comm))) PetscCall(PETSC_ERR_PLIB);
            break;
#endif
#if defined(PETSC_HAVE_MATLAB)
          case 4:
            if (!(*viewer = PETSC_VIEWER_MATLAB_(comm))) PetscCall(PETSC_ERR_PLIB);
            break;
#endif
#if defined(PETSC_HAVE_SAWS)
          case 5:
            if (!(*viewer = PETSC_VIEWER_SAWS_(comm))) PetscCall(PETSC_ERR_PLIB);
            break;
#endif
#if defined(PETSC_HAVE_HDF5)
          case 7:
            if (!(*viewer = PETSC_VIEWER_HDF5_(comm))) PetscCall(PETSC_ERR_PLIB);
            break;
#endif
          case 8:
            if (!(*viewer = PETSC_VIEWER_GLVIS_(comm))) PetscCall(PETSC_ERR_PLIB);
            break;
#if defined(PETSC_HAVE_EXODUSII)
          case 9:
            if (!(*viewer = PETSC_VIEWER_EXODUSII_(comm))) PetscCall(PETSC_ERR_PLIB);
            break;
#endif
          default:
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported viewer %s", loc0_vtype);
          }
          PetscCall(PetscObjectReference((PetscObject)*viewer));
        } else {
          if (loc2_fmt && !*loc1_fname && (cnt == 0)) { /* ASCII format without file name */
            PetscCall(PetscViewerASCIIGetStdout(comm, viewer));
            PetscCall(PetscObjectReference((PetscObject)*viewer));
          } else {
            PetscFileMode fmode;
            PetscCall(PetscViewerCreate(comm, viewer));
            PetscCall(PetscViewerSetType(*viewer, *loc0_vtype ? loc0_vtype : "ascii"));
            fmode = FILE_MODE_WRITE;
            if (loc3_fmode && *loc3_fmode) { /* Has non-empty file mode ("write" or "append") */
              PetscCall(PetscEnumFind(PetscFileModes, loc3_fmode, (PetscEnum *)&fmode, &flag));
              PetscCheck(flag, comm, PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown file mode: %s", loc3_fmode);
            }
            if (loc2_fmt) {
              PetscBool tk, im;
              PetscCall(PetscStrcmp(loc1_fname, "tikz", &tk));
              PetscCall(PetscStrcmp(loc1_fname, "image", &im));
              if (tk || im) {
                PetscCall(PetscViewerDrawSetInfo(*viewer, NULL, loc2_fmt, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE));
                *loc2_fmt = 0;
              }
            }
            PetscCall(PetscViewerFileSetMode(*viewer, flag ? fmode : FILE_MODE_WRITE));
            PetscCall(PetscViewerFileSetName(*viewer, loc1_fname));
            if (*loc1_fname) PetscCall(PetscViewerDrawSetDrawType(*viewer, loc1_fname));
            PetscCall(PetscViewerSetFromOptions(*viewer));
          }
        }
      }
      if (viewer) PetscCall(PetscViewerSetUp(*viewer));
      if (loc2_fmt && *loc2_fmt) {
        PetscViewerFormat tfmt;

        PetscCall(PetscEnumFind(PetscViewerFormats, loc2_fmt, (PetscEnum *)&tfmt, &flag));
        if (format) *format = tfmt;
        PetscCheck(flag, PETSC_COMM_SELF, PETSC_ERR_SUP, "Unknown viewer format %s", loc2_fmt);
      } else if (viewer && (cnt == 6) && format) { /* Get format from VTK viewer */
        PetscCall(PetscViewerGetFormat(*viewer, format));
      }
      PetscCall(PetscFree(loc0_vtype));
    }
  }
  PetscFunctionReturn(0);
}

/*@
   PetscViewerCreate - Creates a viewing context. A `PetscViewer` represents a file, a graphical window, a Unix socket or a variety of other ways of viewing a PETSc object

   Collective

   Input Parameter:
.  comm - MPI communicator

   Output Parameter:
.  inviewer - location to put the `PetscViewer` context

   Level: advanced

.seealso: `PetscViewer`, `PetscViewerDestroy()`, `PetscViewerSetType()`, `PetscViewerType`
@*/
PetscErrorCode PetscViewerCreate(MPI_Comm comm, PetscViewer *inviewer)
{
  PetscViewer viewer;

  PetscFunctionBegin;
  *inviewer = NULL;
  PetscCall(PetscViewerInitializePackage());
  PetscCall(PetscHeaderCreate(viewer, PETSC_VIEWER_CLASSID, "PetscViewer", "PetscViewer", "Viewer", comm, PetscViewerDestroy, PetscViewerView));
  *inviewer    = viewer;
  viewer->data = NULL;
  PetscFunctionReturn(0);
}

/*@C
   PetscViewerSetType - Builds `PetscViewer` for a particular implementation.

   Collective on viewer

   Input Parameters:
+  viewer      - the `PetscViewer` context obtained with `PetscViewerCreate()`
-  type        - for example, `PETSCVIEWERASCII`

   Options Database Key:
.  -viewer_type  <type> - Sets the type; use -help for a list of available methods (for instance, ascii)

   Level: advanced

   Note:
   See "include/petscviewer.h" for available methods (for instance,
   `PETSCVIEWERSOCKET`)

.seealso: `PetscViewer`, `PetscViewerCreate()`, `PetscViewerGetType()`, `PetscViewerType`, `PetscViewerPushFormat()`
@*/
PetscErrorCode PetscViewerSetType(PetscViewer viewer, PetscViewerType type)
{
  PetscBool match;
  PetscErrorCode (*r)(PetscViewer);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscValidCharPointer(type, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, type, &match));
  if (match) PetscFunctionReturn(0);

  /* cleanup any old type that may be there */
  PetscTryTypeMethod(viewer, destroy);
  viewer->ops->destroy = NULL;
  viewer->data         = NULL;

  PetscCall(PetscMemzero(viewer->ops, sizeof(struct _PetscViewerOps)));

  PetscCall(PetscFunctionListFind(PetscViewerList, type, &r));
  PetscCheck(r, PETSC_COMM_SELF, PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown PetscViewer type given: %s", type);

  PetscCall(PetscObjectChangeTypeName((PetscObject)viewer, type));
  PetscCall((*r)(viewer));
  PetscFunctionReturn(0);
}

/*@C
   PetscViewerRegister - Adds a viewer to those available for use

   Not Collective

   Input Parameters:
+  name_solver - name of a new user-defined viewer
-  routine_create - routine to create method context

   Level: developer

   Note:
   `PetscViewerRegister()` may be called multiple times to add several user-defined viewers.

   Sample usage:
.vb
   PetscViewerRegister("my_viewer_type",MyViewerCreate);
.ve

   Then, your solver can be chosen with the procedural interface via
$     PetscViewerSetType(viewer,"my_viewer_type")
   or at runtime via the option
$     -viewer_type my_viewer_type

.seealso: `PetscViewerRegisterAll()`
 @*/
PetscErrorCode PetscViewerRegister(const char *sname, PetscErrorCode (*function)(PetscViewer))
{
  PetscFunctionBegin;
  PetscCall(PetscViewerInitializePackage());
  PetscCall(PetscFunctionListAdd(&PetscViewerList, sname, function));
  PetscFunctionReturn(0);
}

/*@C
   PetscViewerSetFromOptions - Sets various options for a viewer from the options database.

   Collective on viewer

   Input Parameter:
.     viewer - the viewer context

   Level: intermediate

   Note:
    Must be called after PetscViewerCreate() before the PetscViewer is used.

.seealso: `PetscViewer`, `PetscViewerCreate()`, `PetscViewerSetType()`, `PetscViewerType`
@*/
PetscErrorCode PetscViewerSetFromOptions(PetscViewer viewer)
{
  char      vtype[256];
  PetscBool flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);

  if (!PetscViewerList) PetscCall(PetscViewerRegisterAll());
  PetscObjectOptionsBegin((PetscObject)viewer);
  PetscCall(PetscOptionsFList("-viewer_type", "Type of PetscViewer", "None", PetscViewerList, (char *)(((PetscObject)viewer)->type_name ? ((PetscObject)viewer)->type_name : PETSCVIEWERASCII), vtype, 256, &flg));
  if (flg) PetscCall(PetscViewerSetType(viewer, vtype));
  /* type has not been set? */
  if (!((PetscObject)viewer)->type_name) PetscCall(PetscViewerSetType(viewer, PETSCVIEWERASCII));
  PetscTryTypeMethod(viewer, setfromoptions, PetscOptionsObject);

  /* process any options handlers added with PetscObjectAddOptionsHandler() */
  PetscCall(PetscObjectProcessOptionsHandlers((PetscObject)viewer, PetscOptionsObject));
  PetscCall(PetscViewerViewFromOptions(viewer, NULL, "-viewer_view"));
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

PetscErrorCode PetscViewerFlowControlStart(PetscViewer viewer, PetscInt *mcnt, PetscInt *cnt)
{
  PetscFunctionBegin;
  PetscCall(PetscViewerBinaryGetFlowControl(viewer, mcnt));
  PetscCall(PetscViewerBinaryGetFlowControl(viewer, cnt));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscViewerFlowControlStepMain(PetscViewer viewer, PetscInt i, PetscInt *mcnt, PetscInt cnt)
{
  MPI_Comm comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)viewer, &comm));
  if (i >= *mcnt) {
    *mcnt += cnt;
    PetscCallMPI(MPI_Bcast(mcnt, 1, MPIU_INT, 0, comm));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscViewerFlowControlEndMain(PetscViewer viewer, PetscInt *mcnt)
{
  MPI_Comm comm;
  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)viewer, &comm));
  *mcnt = 0;
  PetscCallMPI(MPI_Bcast(mcnt, 1, MPIU_INT, 0, comm));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscViewerFlowControlStepWorker(PetscViewer viewer, PetscMPIInt rank, PetscInt *mcnt)
{
  MPI_Comm comm;
  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)viewer, &comm));
  while (PETSC_TRUE) {
    if (rank < *mcnt) break;
    PetscCallMPI(MPI_Bcast(mcnt, 1, MPIU_INT, 0, comm));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscViewerFlowControlEndWorker(PetscViewer viewer, PetscInt *mcnt)
{
  MPI_Comm comm;
  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)viewer, &comm));
  while (PETSC_TRUE) {
    PetscCallMPI(MPI_Bcast(mcnt, 1, MPIU_INT, 0, comm));
    if (!*mcnt) break;
  }
  PetscFunctionReturn(0);
}
