#include <petsc/private/viewerimpl.h> /*I "petscviewer.h" I*/
#include <petsc/private/hashtable.h>
#if PetscDefined(HAVE_SAWS)
  #include <petscviewersaws.h>
#endif

PetscFunctionList PetscViewerList = NULL;

PetscOptionsHelpPrinted PetscOptionsHelpPrintedSingleton = NULL;
KHASH_SET_INIT_STR(HTPrinted)
struct _n_PetscOptionsHelpPrinted {
  khash_t(HTPrinted) *printed;
  PetscSegBuffer      strings;
};

/*@
  PetscOptionsHelpPrintedDestroy - Destroys the object used to track which help messages have already been printed

  Not Collective

  Input Parameter:
. hp - pointer to the `PetscOptionsHelpPrinted` object to destroy; set to `NULL` on return

  Level: developer

.seealso: `PetscOptionsHelpPrintedCreate()`, `PetscOptionsHelpPrintedCheck()`
@*/
PetscErrorCode PetscOptionsHelpPrintedDestroy(PetscOptionsHelpPrinted *hp)
{
  PetscFunctionBegin;
  if (!*hp) PetscFunctionReturn(PETSC_SUCCESS);
  kh_destroy(HTPrinted, (*hp)->printed);
  PetscCall(PetscSegBufferDestroy(&(*hp)->strings));
  PetscCall(PetscFree(*hp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscOptionsHelpPrintedCreate - Creates an object used to manage tracking which help messages have
  been printed so they will not be printed again.

  Output Parameter:
. hp - the created object

  Not Collective

  Level: developer

.seealso: `PetscOptionsHelpPrintedCheck()`, `PetscOptionsHelpPrintChecked()`
@*/
PetscErrorCode PetscOptionsHelpPrintedCreate(PetscOptionsHelpPrinted *hp)
{
  PetscFunctionBegin;
  PetscCall(PetscNew(hp));
  (*hp)->printed = kh_init(HTPrinted);
  PetscCall(PetscSegBufferCreate(sizeof(char), 10000, &(*hp)->strings));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscOptionsHelpPrintedCheck - Checks if a particular pre, name pair has previous been entered (meaning the help message was printed)

  Not Collective

  Input Parameters:
+ hp   - the object used to manage tracking what help messages have been printed
. pre  - the prefix part of the string, many be `NULL`
- name - the string to look for (cannot be `NULL`)

  Output Parameter:
. found - `PETSC_TRUE` if the string was already set

  Level: intermediate

.seealso: `PetscOptionsHelpPrintedCreate()`
@*/
PetscErrorCode PetscOptionsHelpPrintedCheck(PetscOptionsHelpPrinted hp, const char *pre, const char *name, PetscBool *found)
{
  size_t l1, l2;
#if !PetscDefined(HAVE_THREADSAFETY)
  char *both;
  int   newitem;
#endif

  PetscFunctionBegin;
  PetscCall(PetscStrlen(pre, &l1));
  PetscCall(PetscStrlen(name, &l2));
  if (l1 + l2 == 0) {
    *found = PETSC_FALSE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
#if !PetscDefined(HAVE_THREADSAFETY)
  size_t lboth = l1 + l2 + 1;
  PetscCall(PetscSegBufferGet(hp->strings, lboth, &both));
  PetscCall(PetscStrncpy(both, pre, lboth));
  PetscCall(PetscStrncpy(both + l1, name, l2 + 1));
  kh_put(HTPrinted, hp->printed, both, &newitem);
  if (!newitem) PetscCall(PetscSegBufferUnuse(hp->strings, lboth));
  *found = newitem ? PETSC_FALSE : PETSC_TRUE;
#else
  *found = PETSC_FALSE;
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscBool noviewer = PETSC_FALSE;
static PetscBool noviewers[PETSCVIEWERCREATEVIEWEROFFPUSHESMAX];
static PetscInt  inoviewers = 0;

/*@
  PetscOptionsPushCreateViewerOff - sets if `PetscOptionsCreateViewer()`, `PetscOptionsViewer()`, and `PetscOptionsCreateViewers()` return viewers.

  Logically Collective

  Input Parameter:
. flg - `PETSC_TRUE` to turn off viewer creation, `PETSC_FALSE` to turn it on.

  Options Database Key:
. -viewfromoptions (on|off) - Enable or disable `PetscOptionsCreateViewer()` and `XXXViewFromOptions()` calls, for applications with many small solves turn this off

  Level: developer

  Note:
  Calling `XXXViewFromOptions` in an inner loop can be expensive.  This can appear, for example, when using
  many small subsolves.  Call this function to control viewer creation in `PetscOptionsCreateViewer()`, thus removing the expensive `XXXViewFromOptions` calls.

  Developer Notes:
  Instead of using this approach, the calls to `PetscOptionsCreateViewer()` can be moved into `XXXSetFromOptions()`

.seealso: [](sec_viewers), `PetscOptionsCreateViewer()`, `PetscOptionsPopCreateViewerOff()`
@*/
PetscErrorCode PetscOptionsPushCreateViewerOff(PetscBool flg)
{
  PetscFunctionBegin;
  PetscCheck(inoviewers < PETSCVIEWERCREATEVIEWEROFFPUSHESMAX, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Too many PetscOptionsPushCreateViewerOff(), perhaps you forgot PetscOptionsPopCreateViewerOff()?");

  noviewers[inoviewers++] = noviewer;
  noviewer                = flg;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscOptionsPopCreateViewerOff - reset whether `PetscOptionsCreateViewer()` returns a viewer.

  Logically Collective

  Level: developer

  Note:
  See `PetscOptionsPushCreateViewerOff()`

.seealso: [](sec_viewers), `PetscOptionsCreateViewer()`, `PetscOptionsPushCreateViewerOff()`
@*/
PetscErrorCode PetscOptionsPopCreateViewerOff(void)
{
  PetscFunctionBegin;
  PetscCheck(inoviewers, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Too many PetscOptionsPopCreateViewerOff(), perhaps you forgot PetscOptionsPushCreateViewerOff()?");
  noviewer = noviewers[--inoviewers];
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscOptionsGetCreateViewerOff - do `PetscOptionsCreateViewer()`, `PetscOptionsViewer()`, and `PetscOptionsCreateViewers()` return viewers

  Logically Collective

  Output Parameter:
. flg - whether viewers are returned.

  Level: developer

.seealso: [](sec_viewers), `PetscOptionsCreateViewer()`, `PetscOptionsPushCreateViewerOff()`, `PetscOptionsPopCreateViewerOff()`
@*/
PetscErrorCode PetscOptionsGetCreateViewerOff(PetscBool *flg)
{
  PetscFunctionBegin;
  PetscAssertPointer(flg, 1);
  *flg = noviewer;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscOptionsCreateViewers_Single(MPI_Comm comm, const char value[], PetscViewer *viewer, PetscViewerFormat *format)
{
  char    *loc0_vtype = NULL, *loc1_fname = NULL, *loc2_fmt = NULL, *loc3_fmode = NULL;
  PetscInt cnt;
  size_t   viewer_string_length;
  const char *viewers[] = {PETSCVIEWERASCII, PETSCVIEWERBINARY, PETSCVIEWERDRAW, PETSCVIEWERSOCKET, PETSCVIEWERMATLAB, PETSCVIEWERSAWS, PETSCVIEWERVTK, PETSCVIEWERHDF5, PETSCVIEWERGLVIS, PETSCVIEWEREXODUSII, PETSCVIEWERPYTHON, PETSCVIEWERPYVISTA, NULL}; /* list should be automatically generated from PetscViewersList */

  PetscFunctionBegin;
  PetscCall(PetscStrlen(value, &viewer_string_length));
  if (!viewer_string_length) {
    if (format) *format = PETSC_VIEWER_DEFAULT;
    if (viewer) {
      PetscCall(PetscViewerASCIIGetStdout(comm, viewer));
      PetscCall(PetscObjectReference((PetscObject)*viewer));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(PetscStrallocpy(value, &loc0_vtype));
  PetscCall(PetscStrchr(loc0_vtype, ':', &loc1_fname));
  if (loc1_fname) {
    PetscBool is_daos;
    *loc1_fname++ = 0;
    // When using DAOS, the filename will have the form "daos:/path/to/file.h5", so capture the rest of it.
    PetscCall(PetscStrncmp(loc1_fname, "daos:", 5, &is_daos));
    PetscCall(PetscStrchr(loc1_fname + (is_daos == PETSC_TRUE ? 5 : 0), ':', &loc2_fmt));
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
        PetscCall(PetscObjectReference((PetscObject)*viewer));
        break;
      case 1:
        if (!(*viewer = PETSC_VIEWER_BINARY_(comm))) PetscCall(PETSC_ERR_PLIB);
        PetscCall(PetscObjectReference((PetscObject)*viewer));
        break;
      case 2:
        if (!(*viewer = PETSC_VIEWER_DRAW_(comm))) PetscCall(PETSC_ERR_PLIB);
        PetscCall(PetscObjectReference((PetscObject)*viewer));
        break;
#if PetscDefined(USE_SOCKET_VIEWER)
      case 3:
        if (!(*viewer = PETSC_VIEWER_SOCKET_(comm))) PetscCall(PETSC_ERR_PLIB);
        PetscCall(PetscObjectReference((PetscObject)*viewer));
        break;
#endif
#if PetscDefined(HAVE_MATLAB)
      case 4:
        if (!(*viewer = PETSC_VIEWER_MATLAB_(comm))) PetscCall(PETSC_ERR_PLIB);
        PetscCall(PetscObjectReference((PetscObject)*viewer));
        break;
#endif
#if PetscDefined(HAVE_SAWS)
      case 5:
        if (!(*viewer = PETSC_VIEWER_SAWS_(comm))) PetscCall(PETSC_ERR_PLIB);
        PetscCall(PetscObjectReference((PetscObject)*viewer));
        break;
#endif
#if PetscDefined(HAVE_HDF5)
      case 7:
        if (!(*viewer = PETSC_VIEWER_HDF5_(comm))) PetscCall(PETSC_ERR_PLIB);
        PetscCall(PetscObjectReference((PetscObject)*viewer));
        break;
#endif
      case 8:
        if (!(*viewer = PETSC_VIEWER_GLVIS_(comm))) PetscCall(PETSC_ERR_PLIB);
        PetscCall(PetscObjectReference((PetscObject)*viewer));
        break;
#if PetscDefined(HAVE_EXODUSII)
      case 9:
        if (!(*viewer = PETSC_VIEWER_EXODUSII_(comm))) PetscCall(PETSC_ERR_PLIB);
        PetscCall(PetscObjectReference((PetscObject)*viewer));
        break;
#endif
      case 10:
        if (!(*viewer = PETSC_VIEWER_PYTHON_(comm))) PetscCall(PETSC_ERR_PLIB);
        PetscCall(PetscObjectReference((PetscObject)*viewer));
        break;
      case 11:
        if (!(*viewer = PETSC_VIEWER_PYVISTA_(comm))) PetscCall(PETSC_ERR_PLIB);
        PetscCall(PetscObjectReference((PetscObject)*viewer));
        break;
      default:
        SETERRQ(comm, PETSC_ERR_SUP, "Unsupported viewer %s", loc0_vtype);
      }
    } else {
      if (loc2_fmt && !*loc1_fname && (cnt == 0)) { /* ASCII format without file name */
        PetscCall(PetscViewerASCIIGetStdout(comm, viewer));
        PetscCall(PetscObjectReference((PetscObject)*viewer));
      } else {
        PetscFileMode fmode;
        PetscBool     flag = PETSC_FALSE;

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
    PetscBool         flag;

    PetscCall(PetscEnumFind(PetscViewerFormats, loc2_fmt, (PetscEnum *)&tfmt, &flag));
    if (format) *format = tfmt;
    PetscCheck(flag, comm, PETSC_ERR_SUP, "Unknown viewer format %s", loc2_fmt);
  } else if (viewer && (cnt == 6) && format) { /* Get format from VTK viewer */
    PetscCall(PetscViewerGetFormat(*viewer, format));
  }
  PetscCall(PetscFree(loc0_vtype));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscOptionsCreateViewers_Internal(MPI_Comm comm, PetscOptions options, const char pre[], const char name[], PetscInt *n_max_p, PetscViewer viewer[], PetscViewerFormat format[], PetscBool *set, const char func_name[], PetscBool allow_multiple)
{
  const char *value;
  PetscBool   flag, hashelp;
  PetscInt    n_max;

  PetscFunctionBegin;
  PetscAssertPointer(name, 4);
  PetscAssertPointer(n_max_p, 5);
  n_max = *n_max_p;
  PetscCheck(n_max >= 0, comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid size %" PetscInt_FMT " of passed arrays", *n_max_p);
  *n_max_p = 0;

  if (set) *set = PETSC_FALSE;
  PetscCall(PetscOptionsGetCreateViewerOff(&flag));
  if (flag) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscOptionsHasHelp(NULL, &hashelp));
  if (hashelp) {
    PetscBool found;

    if (!PetscOptionsHelpPrintedSingleton) PetscCall(PetscOptionsHelpPrintedCreate(&PetscOptionsHelpPrintedSingleton));
    PetscCall(PetscOptionsHelpPrintedCheck(PetscOptionsHelpPrintedSingleton, pre, name, &found));
    if (!found && viewer) {
      PetscCall((*PetscHelpPrintf)(comm, "----------------------------------------\nViewer (-%s%s) options:\n", pre ? pre : "", name + 1));
      PetscCall((*PetscHelpPrintf)(comm, "  -%s%s ascii[:[filename][:[format][:filemode]]]: %s (%s)\n", pre ? pre : "", name + 1, "Prints object to stdout or ASCII file", func_name));
      PetscCall((*PetscHelpPrintf)(comm, "  -%s%s binary[:[filename][:[format][:filemode]]]: %s (%s)\n", pre ? pre : "", name + 1, "Saves object to a binary file", func_name));
      PetscCall((*PetscHelpPrintf)(comm, "  -%s%s draw[:[drawtype][:filename|format]] %s (%s)\n", pre ? pre : "", name + 1, "Draws object", func_name));
      PetscCall((*PetscHelpPrintf)(comm, "  -%s%s socket[:port]: %s (%s)\n", pre ? pre : "", name + 1, "Pushes object to a Unix socket", func_name));
      PetscCall((*PetscHelpPrintf)(comm, "  -%s%s saws[:communicatorname]: %s (%s)\n", pre ? pre : "", name + 1, "Publishes object to SAWs", func_name));
      if (allow_multiple) PetscCall((*PetscHelpPrintf)(comm, "  -%s%s v1[,v2,...]: %s (%s)\n", pre ? pre : "", name + 1, "Multiple viewers", func_name));
    }
  }

  PetscCall(PetscOptionsFindPair(options, pre, name, &value, &flag));
  if (flag) {
    if (set) *set = PETSC_TRUE;
    if (!value) {
      PetscCheck(n_max > 0, comm, PETSC_ERR_ARG_SIZ, "More viewers (1) than max available (0)");
      if (format) *format = PETSC_VIEWER_DEFAULT;
      if (viewer) {
        PetscCall(PetscViewerASCIIGetStdout(comm, viewer));
        PetscCall(PetscObjectReference((PetscObject)*viewer));
      }
      *n_max_p = 1;
    } else {
      char  *loc0_viewer_string = NULL, *this_viewer_string = NULL;
      size_t viewer_string_length;

      PetscCall(PetscStrallocpy(value, &loc0_viewer_string));
      PetscCall(PetscStrlen(loc0_viewer_string, &viewer_string_length));
      this_viewer_string = loc0_viewer_string;

      do {
        PetscViewer       *this_viewer;
        PetscViewerFormat *this_viewer_format;
        char              *next_viewer_string = NULL;
        char              *comma_separator    = NULL;
        PetscInt           n                  = *n_max_p;

        PetscCheck(n < n_max, comm, PETSC_ERR_PLIB, "More viewers than max available (%" PetscInt_FMT ")", n_max);

        PetscCall(PetscStrchr(this_viewer_string, ',', &comma_separator));
        if (comma_separator) {
          PetscCheck(allow_multiple, comm, PETSC_ERR_ARG_OUTOFRANGE, "Trying to pass multiple viewers to %s: only one allowed.  Use PetscOptionsCreateViewers() instead", func_name);
          *comma_separator   = 0;
          next_viewer_string = comma_separator + 1;
        }
        this_viewer = PetscSafePointerPlusOffset(viewer, n);
        if (this_viewer) *this_viewer = NULL;
        this_viewer_format = PetscSafePointerPlusOffset(format, n);
        if (this_viewer_format) *this_viewer_format = PETSC_VIEWER_DEFAULT;
        PetscCall(PetscOptionsCreateViewers_Single(comm, this_viewer_string, this_viewer, this_viewer_format));
        this_viewer_string = next_viewer_string;
        (*n_max_p)++;
      } while (this_viewer_string);
      PetscCall(PetscFree(loc0_viewer_string));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscOptionsCreateViewer - Creates a `PetscViewer` and `PetscViewerFormat` based on a viewer specification in the options database

  Collective

  Input Parameters:
+ comm    - the communicator to own the viewer
. options - options database, use `NULL` for default global database
. prefix  - the string to prepend to the name (may be `NULL`)
- name    - the options database name that will be checked for

  Output Parameters:
+ viewer - the viewer, pass `NULL` if not needed
. format - the `PetscViewerFormat` requested by the user, pass `NULL` if not needed
- set    - `PETSC_TRUE` if found, else `PETSC_FALSE`

  Level: intermediate

  Notes:
  The Viewer specification has the following form
.vb
  ascii[:[filename][:[format][:filemode]]]   - filename defaults to stdout
  binary[:[filename][:[format][:filemode]]]  - defaults to the filename of binaryoutput
  hdf5[:[filename][:[format][:filemode]]]    - HDF5 input and output, PETSCVIEWERHDF5
  pyvista[:[filename][:[format][:filemode]]] - display the object with PyVista, PETSCVIEWERPYVISTA
  draw[:x]                                   - draw the object to X Windows
  draw[:tikz[:filename]]                     - draw the object to a TikZ file
  draw[:image[:dirname]]                     - draw the object to an image in memory that gets saved to files in a directory
  socket[:port]                              - defaults to the standard socket output port of 5005, see PetscViewerSocketOpen()
  saws[:communicatorname]                    - publishes object to the Scientific Application Webserver (SAWs)
  vtk:filename.vts                           - VTK output, PETSCVIEWERVTK
.ve

  See `PetscViewerType` for a list of all available viewer types (the string before the first `:`).

  See `PetscViewerFormat` for the possible values of `format`.

  See `PetscFileMode` for the possible values of `filemode`.

  If no viewer type is indicated before the first `:`, then `ascii` is used.

  Unless `filemode` is `append` or `append_update`, files opened in write mode overwrite any previous file.

  You can control whether calls to this function return immediately with a value of `set` of `PETSC_FALSE` using `PetscOptionsPushCreateViewerOff()`.
  This is useful if calling many small subsolves, in which case `XXXViewFromOptions()` calls can take an appreciable fraction of the runtime.

  This routine is thread-safe for accessing predefined `PetscViewer`s like `PETSC_VIEWER_STDOUT_SELF` but not for accessing
  files by name.

  This routine is used by `KSPMonitorSetFromOptions()`, `SNESMonitorSetFromOptions()`, `TSMonitorSetFromOptions()`, `TaoMonitorSetFromOptions()`, and `DMMonitorSetFromOptions()`,
  as well as `PetscObjectViewFromOptions()` and all functions, such as `VecViewFromOptions()` that call it.

  Example Usage:
.vb
  ascii:mesh.tex:ascii_latex - View a `DMPLEX` in LaTeX/TikZ
  draw:tikz:figure.tex       - View an object in the file `figure.tex` using TikZ
.ve

.seealso: [](sec_viewers), `PetscViewerFormat`, `PetscViewerDestroy()`, `PetscOptionsGetReal()`, `PetscOptionsHasName()`, `PetscOptionsGetString()`,
          `PetscOptionsGetIntArray()`, `PetscOptionsGetRealArray()`, `PetscOptionsBool()`,
          `PetscOptionsInt()`, `PetscOptionsString()`, `PetscOptionsReal()`,
          `PetscOptionsName()`, `PetscOptionsBegin()`, `PetscOptionsEnd()`, `PetscOptionsHeadBegin()`,
          `PetscOptionsStringArray()`, `PetscOptionsRealArray()`, `PetscOptionsScalar()`,
          `PetscOptionsBoolGroupBegin()`, `PetscOptionsBoolGroup()`, `PetscOptionsBoolGroupEnd()`,
          `PetscOptionsFList()`, `PetscOptionsEList()`, `PetscOptionsPushCreateViewerOff()`, `PetscOptionsPopCreateViewerOff()`,
          `KSPMonitorSetFromOptions()`, `SNESMonitorSetFromOptions()`, `TSMonitorSetFromOptions()`,
          `TaoMonitorSetFromOptions()`, `DMMonitorSetFromOptions()`, `PetscObjectViewFromOptions()`, `VecViewFromOptions()`
@*/
PetscErrorCode PetscOptionsCreateViewer(MPI_Comm comm, PetscOptions options, const char prefix[], const char name[], PetscViewer *viewer, PetscViewerFormat *format, PetscBool *set)
{
  PetscInt  n_max = 1;
  PetscBool set_internal;

  PetscFunctionBegin;
  if (viewer) *viewer = NULL;
  if (format) *format = PETSC_VIEWER_DEFAULT;
  PetscCall(PetscOptionsCreateViewers_Internal(comm, options, prefix, name, &n_max, viewer, format, &set_internal, PETSC_FUNCTION_NAME, PETSC_FALSE));
  if (set_internal) PetscAssert(n_max == 1, comm, PETSC_ERR_PLIB, "Unexpected: %" PetscInt_FMT " != 1 viewers set", n_max);
  if (set) *set = set_internal;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscOptionsCreateViewers - Create multiple viewers from a comma-separated list of viewer specifications in the options database

  Collective

  Input Parameters:
+ comm    - the communicator to own the viewers
. options - options database, use `NULL` for default global database
. prefix  - the string to prepend to the name (may be `NULL`)
. name    - the options database name that will be checked for
- n_max   - on input: the maximum number of viewers; on output: the number of viewers found in the comma-separated list

  Output Parameters:
+ viewers - an array to hold at least `n_max` `PetscViewer`s, or `NULL` if not needed; on output: if not `NULL`, the
            first `n_max` entries are initialized `PetscViewer`s
. formats - an array to hold at least `n_max` `PetscViewerFormat`s, or `NULL` if not needed; on output: if not
            `NULL`, the first `n_max` entries are valid `PetscViewewFormat`s
- set     - `PETSC_TRUE` if found, else `PETSC_FALSE`

  Level: intermediate

  Notes:
  See `PetscOptionsCreateViewer()` for how the viewer specifications are interpreted.

  Use `PetscViewerDestroy()` on each viewer.

.seealso: [](sec_viewers), `PetscOptionsCreateViewer()`
@*/
PetscErrorCode PetscOptionsCreateViewers(MPI_Comm comm, PetscOptions options, const char prefix[], const char name[], PetscInt *n_max, PetscViewer viewers[], PetscViewerFormat formats[], PetscBool *set)
{
  PetscFunctionBegin;
  PetscCall(PetscOptionsCreateViewers_Internal(comm, options, prefix, name, n_max, viewers, formats, set, PETSC_FUNCTION_NAME, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscViewerCreate - Creates a viewing context. A `PetscViewer` represents a file, a graphical window, a Unix socket or a variety of other ways
  of viewing a PETSc object

  Collective

  Input Parameter:
. comm - MPI communicator

  Output Parameter:
. inviewer - location to put the `PetscViewer` context

  Level: advanced

.seealso: [](sec_viewers), `PetscViewer`, `PetscViewerDestroy()`, `PetscViewerSetType()`, `PetscViewerType`
@*/
PetscErrorCode PetscViewerCreate(MPI_Comm comm, PetscViewer *inviewer)
{
  PetscViewer viewer;

  PetscFunctionBegin;
  PetscAssertPointer(inviewer, 2);
  PetscCall(PetscViewerInitializePackage());
  PetscCall(PetscHeaderCreate(viewer, PETSC_VIEWER_CLASSID, "PetscViewer", "PetscViewer", "Viewer", comm, PetscViewerDestroy, PetscViewerView));
  *inviewer    = viewer;
  viewer->data = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscViewerSetType - Builds `PetscViewer` for a particular implementation.

  Collective

  Input Parameters:
+ viewer - the `PetscViewer` context obtained with `PetscViewerCreate()`
- type   - for example, `PETSCVIEWERASCII`

  Options Database Key:
. -viewer_type  type - Sets the type; use -help for a list of available methods (for instance, ascii)

  Level: advanced

  Note:
  See `PetscViewerType` for possible values

.seealso: [](sec_viewers), `PetscViewer`, `PetscViewerCreate()`, `PetscViewerGetType()`, `PetscViewerType`, `PetscViewerPushFormat()`
@*/
PetscErrorCode PetscViewerSetType(PetscViewer viewer, PetscViewerType type)
{
  PetscBool match;
  PetscErrorCode (*r)(PetscViewer);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscAssertPointer(type, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, type, &match));
  if (match) PetscFunctionReturn(PETSC_SUCCESS);

  /* cleanup any old type that may be there */
  PetscTryTypeMethod(viewer, destroy);
  viewer->ops->destroy = NULL;
  viewer->data         = NULL;

  PetscCall(PetscMemzero(viewer->ops, sizeof(struct _PetscViewerOps)));

  PetscCall(PetscFunctionListFind(PetscViewerList, type, &r));
  PetscCheck(r, PetscObjectComm((PetscObject)viewer), PETSC_ERR_ARG_UNKNOWN_TYPE, "Unknown PetscViewer type given: %s", type);

  PetscCall(PetscObjectChangeTypeName((PetscObject)viewer, type));
  PetscCall((*r)(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscViewerRegister - Adds a viewer to those available for use with `PetscViewerSetType()`

  Not Collective, No Fortran Support

  Input Parameters:
+ sname    - name of a new user-defined viewer
- function - routine to create method context

  Level: developer

  Note:
  `PetscViewerRegister()` may be called multiple times to add several user-defined viewers.

  Example Usage:
.vb
   PetscViewerRegister("my_viewer_type", MyViewerCreate);
.ve

  Then, your solver can be chosen with the procedural interface via
.vb
  PetscViewerSetType(viewer, "my_viewer_type")
.ve
  or at runtime via the option
.vb
  -viewer_type my_viewer_type
.ve

.seealso: [](sec_viewers), `PetscViewerRegisterAll()`
 @*/
PetscErrorCode PetscViewerRegister(const char *sname, PetscErrorCode (*function)(PetscViewer))
{
  PetscFunctionBegin;
  PetscCall(PetscViewerInitializePackage());
  PetscCall(PetscFunctionListAdd(&PetscViewerList, sname, function));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscViewerSetFromOptions - Sets various options for a viewer based on values in the options database.

  Collective

  Input Parameter:
. viewer - the viewer context

  Level: intermediate

  Note:
  Must be called after `PetscViewerCreate()` but before the `PetscViewer` is used.

.seealso: [](sec_viewers), `PetscViewer`, `PetscViewerCreate()`, `PetscViewerSetType()`, `PetscViewerType`
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
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscViewerFlowControlStart - Begin a flow-controlled viewer operation on the main MPI process

  Collective

  Input Parameter:
. viewer - the binary viewer

  Output Parameters:
+ mcnt - the current flow-control counter on the main MPI process
- cnt  - the flow-control window size (also read from the viewer)

  Level: developer

  Note:
  Used together with `PetscViewerFlowControlStepMain()` and `PetscViewerFlowControlEndMain()` on the main process (rank 0),
  and with `PetscViewerFlowControlStepWorker()` and `PetscViewerFlowControlEndWorker()` on the other processes, to serialize
  I/O work through a bounded window so that all processes do not simultaneously flood the main process with data.

.seealso: `PetscViewer`, `PetscViewerFlowControlStepMain()`, `PetscViewerFlowControlEndMain()`, `PetscViewerFlowControlStepWorker()`,
          `PetscViewerFlowControlEndWorker()`, `PetscViewerBinaryGetFlowControl()`
@*/
PetscErrorCode PetscViewerFlowControlStart(PetscViewer viewer, PetscInt *mcnt, PetscInt *cnt)
{
  PetscFunctionBegin;
  PetscCall(PetscViewerBinaryGetFlowControl(viewer, mcnt));
  PetscCall(PetscViewerBinaryGetFlowControl(viewer, cnt));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscViewerFlowControlStepMain - Advance the flow-control window on the main MPI process during a viewer operation

  Collective

  Input Parameters:
+ viewer - the binary viewer
. i      - the current MPI rank being served
- cnt    - the flow-control window size returned by `PetscViewerFlowControlStart()`

  Input/Output Parameter:
. mcnt - the running flow-control counter; incremented and broadcast when `i` reaches it

  Level: developer

  Note:
  Called on the main MPI process (rank 0) once per worker rank in a loop; when the current rank has caught up to `mcnt`
  the window is advanced by `cnt` and broadcast so waiting workers can proceed.

.seealso: `PetscViewer`, `PetscViewerFlowControlStart()`, `PetscViewerFlowControlEndMain()`, `PetscViewerFlowControlStepWorker()`,
          `PetscViewerFlowControlEndWorker()`
@*/
PetscErrorCode PetscViewerFlowControlStepMain(PetscViewer viewer, PetscInt i, PetscInt *mcnt, PetscInt cnt)
{
  MPI_Comm comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)viewer, &comm));
  if (i >= *mcnt) {
    *mcnt += cnt;
    PetscCallMPI(MPI_Bcast(mcnt, 1, MPIU_INT, 0, comm));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscViewerFlowControlEndMain - Finish a flow-controlled viewer operation on the main MPI process by signalling completion to the workers

  Collective

  Input Parameter:
. viewer - the binary viewer

  Input/Output Parameter:
. mcnt - the flow-control counter; reset to 0 and broadcast to signal completion

  Level: developer

  Note:
  Broadcasting `mcnt = 0` releases any worker MPI processes still waiting inside `PetscViewerFlowControlEndWorker()`.

.seealso: `PetscViewer`, `PetscViewerFlowControlStart()`, `PetscViewerFlowControlStepMain()`, `PetscViewerFlowControlStepWorker()`,
          `PetscViewerFlowControlEndWorker()`
@*/
PetscErrorCode PetscViewerFlowControlEndMain(PetscViewer viewer, PetscInt *mcnt)
{
  MPI_Comm comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)viewer, &comm));
  *mcnt = 0;
  PetscCallMPI(MPI_Bcast(mcnt, 1, MPIU_INT, 0, comm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscViewerFlowControlStepWorker - Wait on a worker MPI process until the flow-control window includes this rank

  Collective

  Input Parameters:
+ viewer - the binary viewer
- rank   - the calling MPI process rank

  Input/Output Parameter:
. mcnt - the flow-control counter; updated with values broadcast from the main MPI process until it exceeds `rank`

  Level: developer

  Note:
  Blocks in a loop of `MPI_Bcast()` until the main MPI process (through `PetscViewerFlowControlStepMain()`) advances the
  window past this rank, giving the worker permission to perform its I/O.

.seealso: `PetscViewer`, `PetscViewerFlowControlStart()`, `PetscViewerFlowControlStepMain()`, `PetscViewerFlowControlEndMain()`,
          `PetscViewerFlowControlEndWorker()`
@*/
PetscErrorCode PetscViewerFlowControlStepWorker(PetscViewer viewer, PetscMPIInt rank, PetscInt *mcnt)
{
  MPI_Comm comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)viewer, &comm));
  while (PETSC_TRUE) {
    if (rank < *mcnt) break;
    PetscCallMPI(MPI_Bcast(mcnt, 1, MPIU_INT, 0, comm));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscViewerFlowControlEndWorker - Wait on a worker MPI process for the main MPI process to signal completion of a flow-controlled viewer operation

  Collective

  Input Parameter:
. viewer - the binary viewer

  Input/Output Parameter:
. mcnt - the flow-control counter; updated with values broadcast from the main MPI process until it becomes 0

  Level: developer

  Note:
  Blocks in a loop of `MPI_Bcast()` until `PetscViewerFlowControlEndMain()` sends a `mcnt = 0` completion signal.

.seealso: `PetscViewer`, `PetscViewerFlowControlStart()`, `PetscViewerFlowControlStepMain()`, `PetscViewerFlowControlEndMain()`,
          `PetscViewerFlowControlStepWorker()`
@*/
PetscErrorCode PetscViewerFlowControlEndWorker(PetscViewer viewer, PetscInt *mcnt)
{
  MPI_Comm comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)viewer, &comm));
  while (PETSC_TRUE) {
    PetscCallMPI(MPI_Bcast(mcnt, 1, MPIU_INT, 0, comm));
    if (!*mcnt) break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
