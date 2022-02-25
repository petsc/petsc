
#include <petsc/private/viewerimpl.h>  /*I "petscviewer.h" I*/
#include <petsc/private/hashtable.h>
#if defined(PETSC_HAVE_SAWS)
#include <petscviewersaws.h>
#endif

PetscFunctionList PetscViewerList = NULL;

PetscOptionsHelpPrinted PetscOptionsHelpPrintedSingleton = NULL;
KHASH_SET_INIT_STR(HTPrinted)
struct  _n_PetscOptionsHelpPrinted{
  khash_t(HTPrinted) *printed;
  PetscSegBuffer     strings;
};

PetscErrorCode PetscOptionsHelpPrintedDestroy(PetscOptionsHelpPrinted *hp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*hp) PetscFunctionReturn(0);
  kh_destroy(HTPrinted,(*hp)->printed);
  ierr = PetscSegBufferDestroy(&(*hp)->strings);CHKERRQ(ierr);
  ierr = PetscFree(*hp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
      PetscOptionsHelpPrintedCreate - Creates an object used to manage tracking which help messages have
         been printed so they will not be printed again.

     Not collective

    Level: developer

.seealso: PetscOptionsHelpPrintedCheck(), PetscOptionsHelpPrintChecked()
@*/
PetscErrorCode PetscOptionsHelpPrintedCreate(PetscOptionsHelpPrinted *hp)
{
  PetscErrorCode             ierr;

  PetscFunctionBegin;
  ierr = PetscNew(hp);CHKERRQ(ierr);
  (*hp)->printed = kh_init(HTPrinted);
  ierr = PetscSegBufferCreate(sizeof(char),10000,&(*hp)->strings);CHKERRQ(ierr);
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

.seealso: PetscOptionsHelpPrintedCreate()
@*/
PetscErrorCode PetscOptionsHelpPrintedCheck(PetscOptionsHelpPrinted hp,const char *pre,const char* name,PetscBool *found)
{
  size_t          l1,l2;
#if !defined(PETSC_HAVE_THREADSAFETY)
  char            *both;
  int             newitem;
#endif
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscStrlen(pre,&l1);CHKERRQ(ierr);
  ierr = PetscStrlen(name,&l2);CHKERRQ(ierr);
  if (l1+l2 == 0) {
    *found = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
#if !defined(PETSC_HAVE_THREADSAFETY)
  ierr = PetscSegBufferGet(hp->strings,l1+l2+1,&both);CHKERRQ(ierr);
  ierr = PetscStrcpy(both,pre);CHKERRQ(ierr);
  ierr = PetscStrcat(both,name);CHKERRQ(ierr);
  kh_put(HTPrinted,hp->printed,both,&newitem);
  if (!newitem) {
    ierr = PetscSegBufferUnuse(hp->strings,l1+l2+1);CHKERRQ(ierr);
  }
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
  PetscOptionsPushGetViewerOff - control whether PetscOptionsGetViewer returns a viewer.

  Logically Collective

  Input Parameter:
. flg - PETSC_TRUE to turn off viewer creation, PETSC_FALSE to turn it on.

  Level: developer

  Notes:
    Calling XXXViewFromOptions in an inner loop can be very expensive.  This can appear, for example, when using
   many small subsolves.  Call this function to control viewer creation in PetscOptionsGetViewer, thus removing the expensive XXXViewFromOptions calls.

.seealso: PetscOptionsGetViewer(), PetscOptionsPopGetViewerOff()
@*/
PetscErrorCode  PetscOptionsPushGetViewerOff(PetscBool flg)
{
  PetscFunctionBegin;
  PetscCheckFalse(inoviewers > PETSCVIEWERGETVIEWEROFFPUSHESMAX - 1,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Too many PetscOptionsPushGetViewerOff(), perhaps you forgot PetscOptionsPopGetViewerOff()?");

  noviewers[inoviewers++] = noviewer;
  noviewer = flg;
  PetscFunctionReturn(0);
}

/*@
  PetscOptionsPopGetViewerOff - reset whether PetscOptionsGetViewer returns a viewer.

  Logically Collective

  Level: developer

  Notes:
    Calling XXXViewFromOptions in an inner loop can be very expensive.  This can appear, for example, when using
   many small subsolves.  Call this function to control viewer creation in PetscOptionsGetViewer, thus removing the expensive XXXViewFromOptions calls.

.seealso: PetscOptionsGetViewer(), PetscOptionsPushGetViewerOff()
@*/
PetscErrorCode  PetscOptionsPopGetViewerOff(void)
{
  PetscFunctionBegin;
  PetscCheckFalse(!inoviewers,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Too many PetscOptionsPopGetViewerOff(), perhaps you forgot PetscOptionsPushGetViewerOff()?");
  noviewer = noviewers[--inoviewers];
  PetscFunctionReturn(0);
}

/*@
  PetscOptionsGetViewerOff - does PetscOptionsGetViewer return a viewer?

  Logically Collective

  Output Parameter:
. flg - whether viewers are returned.

  Level: developer

  Notes:
    Calling XXXViewFromOptions in an inner loop can be very expensive.  This can appear, for example, when using
   many small subsolves.

.seealso: PetscOptionsGetViewer(), PetscOptionsPushGetViewerOff(), PetscOptionsPopGetViewerOff()
@*/
PetscErrorCode  PetscOptionsGetViewerOff(PetscBool *flg)
{
  PetscFunctionBegin;
  PetscValidBoolPointer(flg,1);
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
.  format - the PetscViewerFormat requested by the user, pass NULL if not needed
-  set - PETSC_TRUE if found, else PETSC_FALSE

   Level: intermediate

   Notes:
    If no value is provided ascii:stdout is used
$       ascii[:[filename][:[format][:append]]]    defaults to stdout - format can be one of ascii_info, ascii_info_detail, or ascii_matlab,
                                                  for example ascii::ascii_info prints just the information about the object not all details
                                                  unless :append is given filename opens in write mode, overwriting what was already there
$       binary[:[filename][:[format][:append]]]   defaults to the file binaryoutput
$       draw[:drawtype[:filename]]                for example, draw:tikz, draw:tikz:figure.tex  or draw:x
$       socket[:port]                             defaults to the standard output port
$       saws[:communicatorname]                    publishes object to the Scientific Application Webserver (SAWs)

   Use PetscViewerDestroy() after using the viewer, otherwise a memory leak will occur

   You can control whether calls to this function create a viewer (or return early with *set of PETSC_FALSE) with
   PetscOptionsPushGetViewerOff.  This is useful if calling many small subsolves, in which case XXXViewFromOptions can take
   an appreciable fraction of the runtime.

   If PETSc is configured with --with-viewfromoptions=0 this function always returns with *set of PETSC_FALSE

.seealso: PetscOptionsGetReal(), PetscOptionsHasName(), PetscOptionsGetString(),
          PetscOptionsGetIntArray(), PetscOptionsGetRealArray(), PetscOptionsBool()
          PetscOptionsInt(), PetscOptionsString(), PetscOptionsReal(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList(), PetscOptionsPushGetViewerOff(), PetscOptionsPopGetViewerOff(),
          PetscOptionsGetViewerOff()
@*/
PetscErrorCode  PetscOptionsGetViewer(MPI_Comm comm,PetscOptions options,const char pre[],const char name[],PetscViewer *viewer,PetscViewerFormat *format,PetscBool  *set)
{
  const char                     *value;
  PetscErrorCode                 ierr;
  PetscBool                      flag,hashelp;

  PetscFunctionBegin;
  PetscValidCharPointer(name,4);

  if (viewer) *viewer = NULL;
  if (format) *format = PETSC_VIEWER_DEFAULT;
  if (set)    *set    = PETSC_FALSE;
  ierr = PetscOptionsGetViewerOff(&flag);CHKERRQ(ierr);
  if (flag) PetscFunctionReturn(0);

  ierr = PetscOptionsHasHelp(NULL,&hashelp);CHKERRQ(ierr);
  if (hashelp) {
    PetscBool found;

    if (!PetscOptionsHelpPrintedSingleton) {
      ierr = PetscOptionsHelpPrintedCreate(&PetscOptionsHelpPrintedSingleton);CHKERRQ(ierr);
    }
    ierr = PetscOptionsHelpPrintedCheck(PetscOptionsHelpPrintedSingleton,pre,name,&found);CHKERRQ(ierr);
    if (!found && viewer) {
      ierr = (*PetscHelpPrintf)(comm,"----------------------------------------\nViewer (-%s%s) options:\n",pre ? pre : "",name+1);CHKERRQ(ierr);
      ierr = (*PetscHelpPrintf)(comm,"  -%s%s ascii[:[filename][:[format][:append]]]: %s (%s)\n",pre ? pre : "",name+1,"Prints object to stdout or ASCII file","PetscOptionsGetViewer");CHKERRQ(ierr);
      ierr = (*PetscHelpPrintf)(comm,"  -%s%s binary[:[filename][:[format][:append]]]: %s (%s)\n",pre ? pre : "",name+1,"Saves object to a binary file","PetscOptionsGetViewer");CHKERRQ(ierr);
      ierr = (*PetscHelpPrintf)(comm,"  -%s%s draw[:[drawtype][:filename|format]] %s (%s)\n",pre ? pre : "",name+1,"Draws object","PetscOptionsGetViewer");CHKERRQ(ierr);
      ierr = (*PetscHelpPrintf)(comm,"  -%s%s socket[:port]: %s (%s)\n",pre ? pre : "",name+1,"Pushes object to a Unix socket","PetscOptionsGetViewer");CHKERRQ(ierr);
      ierr = (*PetscHelpPrintf)(comm,"  -%s%s saws[:communicatorname]: %s (%s)\n",pre ? pre : "",name+1,"Publishes object to SAWs","PetscOptionsGetViewer");CHKERRQ(ierr);
    }
  }

  if (format) *format = PETSC_VIEWER_DEFAULT;
  ierr = PetscOptionsFindPair(options,pre,name,&value,&flag);CHKERRQ(ierr);
  if (flag) {
    if (set) *set = PETSC_TRUE;
    if (!value) {
      if (viewer) {
        ierr = PetscViewerASCIIGetStdout(comm,viewer);CHKERRQ(ierr);
        ierr = PetscObjectReference((PetscObject)*viewer);CHKERRQ(ierr);
      }
    } else {
      char       *loc0_vtype,*loc1_fname,*loc2_fmt = NULL,*loc3_fmode = NULL;
      PetscInt   cnt;
      const char *viewers[] = {PETSCVIEWERASCII,PETSCVIEWERBINARY,PETSCVIEWERDRAW,PETSCVIEWERSOCKET,PETSCVIEWERMATLAB,PETSCVIEWERSAWS,PETSCVIEWERVTK,PETSCVIEWERHDF5,PETSCVIEWERGLVIS,PETSCVIEWEREXODUSII,NULL};

      ierr = PetscStrallocpy(value,&loc0_vtype);CHKERRQ(ierr);
      ierr = PetscStrchr(loc0_vtype,':',&loc1_fname);CHKERRQ(ierr);
      if (loc1_fname) {
        *loc1_fname++ = 0;
        ierr = PetscStrchr(loc1_fname,':',&loc2_fmt);CHKERRQ(ierr);
      }
      if (loc2_fmt) {
        *loc2_fmt++ = 0;
        ierr = PetscStrchr(loc2_fmt,':',&loc3_fmode);CHKERRQ(ierr);
      }
      if (loc3_fmode) *loc3_fmode++ = 0;
      ierr = PetscStrendswithwhich(*loc0_vtype ? loc0_vtype : "ascii",viewers,&cnt);CHKERRQ(ierr);
      PetscCheckFalse(cnt > (PetscInt) sizeof(viewers)-1,comm,PETSC_ERR_ARG_OUTOFRANGE,"Unknown viewer type: %s",loc0_vtype);
      if (viewer) {
        if (!loc1_fname) {
          switch (cnt) {
          case 0:
            ierr = PetscViewerASCIIGetStdout(comm,viewer);CHKERRQ(ierr);
            break;
          case 1:
            if (!(*viewer = PETSC_VIEWER_BINARY_(comm))) CHKERRQ(PETSC_ERR_PLIB);
            break;
          case 2:
            if (!(*viewer = PETSC_VIEWER_DRAW_(comm))) CHKERRQ(PETSC_ERR_PLIB);
            break;
#if defined(PETSC_USE_SOCKET_VIEWER)
          case 3:
            if (!(*viewer = PETSC_VIEWER_SOCKET_(comm))) CHKERRQ(PETSC_ERR_PLIB);
            break;
#endif
#if defined(PETSC_HAVE_MATLAB_ENGINE)
          case 4:
            if (!(*viewer = PETSC_VIEWER_MATLAB_(comm))) CHKERRQ(PETSC_ERR_PLIB);
            break;
#endif
#if defined(PETSC_HAVE_SAWS)
          case 5:
            if (!(*viewer = PETSC_VIEWER_SAWS_(comm))) CHKERRQ(PETSC_ERR_PLIB);
            break;
#endif
#if defined(PETSC_HAVE_HDF5)
          case 7:
            if (!(*viewer = PETSC_VIEWER_HDF5_(comm))) CHKERRQ(PETSC_ERR_PLIB);
            break;
#endif
          case 8:
            if (!(*viewer = PETSC_VIEWER_GLVIS_(comm))) CHKERRQ(PETSC_ERR_PLIB);
            break;
#if defined(PETSC_HAVE_EXODUSII)
          case 9:
            if (!(*viewer = PETSC_VIEWER_EXODUSII_(comm))) CHKERRQ(PETSC_ERR_PLIB);
            break;
#endif
          default: SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Unsupported viewer %s",loc0_vtype);
          }
          ierr = PetscObjectReference((PetscObject)*viewer);CHKERRQ(ierr);
        } else {
          if (loc2_fmt && !*loc1_fname && (cnt == 0)) { /* ASCII format without file name */
            ierr = PetscViewerASCIIGetStdout(comm,viewer);CHKERRQ(ierr);
            ierr = PetscObjectReference((PetscObject)*viewer);CHKERRQ(ierr);
          } else {
            PetscFileMode fmode;
            ierr = PetscViewerCreate(comm,viewer);CHKERRQ(ierr);
            ierr = PetscViewerSetType(*viewer,*loc0_vtype ? loc0_vtype : "ascii");CHKERRQ(ierr);
            fmode = FILE_MODE_WRITE;
            if (loc3_fmode && *loc3_fmode) { /* Has non-empty file mode ("write" or "append") */
              ierr = PetscEnumFind(PetscFileModes,loc3_fmode,(PetscEnum*)&fmode,&flag);CHKERRQ(ierr);
              PetscCheckFalse(!flag,comm,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown file mode: %s",loc3_fmode);
            }
            if (loc2_fmt) {
              PetscBool tk,im;
              ierr = PetscStrcmp(loc1_fname,"tikz",&tk);CHKERRQ(ierr);
              ierr = PetscStrcmp(loc1_fname,"image",&im);CHKERRQ(ierr);
              if (tk || im) {
                ierr = PetscViewerDrawSetInfo(*viewer,NULL,loc2_fmt,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
                *loc2_fmt = 0;
              }
            }
            ierr = PetscViewerFileSetMode(*viewer,flag?fmode:FILE_MODE_WRITE);CHKERRQ(ierr);
            ierr = PetscViewerFileSetName(*viewer,loc1_fname);CHKERRQ(ierr);
            if (*loc1_fname) {
              ierr = PetscViewerDrawSetDrawType(*viewer,loc1_fname);CHKERRQ(ierr);
            }
            ierr = PetscViewerSetFromOptions(*viewer);CHKERRQ(ierr);
          }
        }
      }
      if (viewer) {
        ierr = PetscViewerSetUp(*viewer);CHKERRQ(ierr);
      }
      if (loc2_fmt && *loc2_fmt) {
        PetscViewerFormat tfmt;

        ierr = PetscEnumFind(PetscViewerFormats,loc2_fmt,(PetscEnum*)&tfmt,&flag);CHKERRQ(ierr);
        if (format) *format = tfmt;
        PetscCheckFalse(!flag,PETSC_COMM_SELF,PETSC_ERR_SUP,"Unknown viewer format %s",loc2_fmt);
      } else if (viewer && (cnt == 6) && format) { /* Get format from VTK viewer */
        ierr = PetscViewerGetFormat(*viewer,format);CHKERRQ(ierr);
      }
      ierr = PetscFree(loc0_vtype);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*@
   PetscViewerCreate - Creates a viewing context

   Collective

   Input Parameter:
.  comm - MPI communicator

   Output Parameter:
.  inviewer - location to put the PetscViewer context

   Level: advanced

.seealso: PetscViewerDestroy(), PetscViewerSetType(), PetscViewerType

@*/
PetscErrorCode  PetscViewerCreate(MPI_Comm comm,PetscViewer *inviewer)
{
  PetscViewer    viewer;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *inviewer = NULL;
  ierr = PetscViewerInitializePackage();CHKERRQ(ierr);
  ierr         = PetscHeaderCreate(viewer,PETSC_VIEWER_CLASSID,"PetscViewer","PetscViewer","Viewer",comm,PetscViewerDestroy,PetscViewerView);CHKERRQ(ierr);
  *inviewer    = viewer;
  viewer->data = NULL;
  PetscFunctionReturn(0);
}

/*@C
   PetscViewerSetType - Builds PetscViewer for a particular implementation.

   Collective on PetscViewer

   Input Parameters:
+  viewer      - the PetscViewer context
-  type        - for example, PETSCVIEWERASCII

   Options Database Command:
.  -viewer_type  <type> - Sets the type; use -help for a list
    of available methods (for instance, ascii)

   Level: advanced

   Notes:
   See "include/petscviewer.h" for available methods (for instance,
   PETSCVIEWERSOCKET)

.seealso: PetscViewerCreate(), PetscViewerGetType(), PetscViewerType, PetscViewerPushFormat()
@*/
PetscErrorCode  PetscViewerSetType(PetscViewer viewer,PetscViewerType type)
{
  PetscErrorCode ierr,(*r)(PetscViewer);
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  PetscValidCharPointer(type,2);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  /* cleanup any old type that may be there */
  if (viewer->data) {
    ierr         = (*viewer->ops->destroy)(viewer);CHKERRQ(ierr);

    viewer->ops->destroy = NULL;
    viewer->data         = NULL;
  }
  ierr = PetscMemzero(viewer->ops,sizeof(struct _PetscViewerOps));CHKERRQ(ierr);

  ierr =  PetscFunctionListFind(PetscViewerList,type,&r);CHKERRQ(ierr);
  PetscCheckFalse(!r,PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown PetscViewer type given: %s",type);

  ierr = PetscObjectChangeTypeName((PetscObject)viewer,type);CHKERRQ(ierr);
  ierr = (*r)(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscViewerRegister - Adds a viewer

   Not Collective

   Input Parameters:
+  name_solver - name of a new user-defined viewer
-  routine_create - routine to create method context

   Level: developer
   Notes:
   PetscViewerRegister() may be called multiple times to add several user-defined viewers.

   Sample usage:
.vb
   PetscViewerRegister("my_viewer_type",MyViewerCreate);
.ve

   Then, your solver can be chosen with the procedural interface via
$     PetscViewerSetType(viewer,"my_viewer_type")
   or at runtime via the option
$     -viewer_type my_viewer_type

.seealso: PetscViewerRegisterAll()
 @*/
PetscErrorCode  PetscViewerRegister(const char *sname,PetscErrorCode (*function)(PetscViewer))
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerInitializePackage();CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&PetscViewerList,sname,function);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscViewerSetFromOptions - Sets the graphics type from the options database.
      Defaults to a PETSc X windows graphics.

   Collective on PetscViewer

   Input Parameter:
.     PetscViewer - the graphics context

   Level: intermediate

   Notes:
    Must be called after PetscViewerCreate() before the PetscViewer is used.

.seealso: PetscViewerCreate(), PetscViewerSetType(), PetscViewerType

@*/
PetscErrorCode  PetscViewerSetFromOptions(PetscViewer viewer)
{
  PetscErrorCode    ierr;
  char              vtype[256];
  PetscBool         flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);

  if (!PetscViewerList) {
    ierr = PetscViewerRegisterAll();CHKERRQ(ierr);
  }
  ierr = PetscObjectOptionsBegin((PetscObject)viewer);CHKERRQ(ierr);
  ierr = PetscOptionsFList("-viewer_type","Type of PetscViewer","None",PetscViewerList,(char*)(((PetscObject)viewer)->type_name ? ((PetscObject)viewer)->type_name : PETSCVIEWERASCII),vtype,256,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscViewerSetType(viewer,vtype);CHKERRQ(ierr);
  }
  /* type has not been set? */
  if (!((PetscObject)viewer)->type_name) {
    ierr = PetscViewerSetType(viewer,PETSCVIEWERASCII);CHKERRQ(ierr);
  }
  if (viewer->ops->setfromoptions) {
    ierr = (*viewer->ops->setfromoptions)(PetscOptionsObject,viewer);CHKERRQ(ierr);
  }

  /* process any options handlers added with PetscObjectAddOptionsHandler() */
  ierr = PetscObjectProcessOptionsHandlers(PetscOptionsObject,(PetscObject)viewer);CHKERRQ(ierr);
  ierr = PetscViewerViewFromOptions(viewer,NULL,"-viewer_view");CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscViewerFlowControlStart(PetscViewer viewer,PetscInt *mcnt,PetscInt *cnt)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscViewerBinaryGetFlowControl(viewer,mcnt);CHKERRQ(ierr);
  ierr = PetscViewerBinaryGetFlowControl(viewer,cnt);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscViewerFlowControlStepMain(PetscViewer viewer,PetscInt i,PetscInt *mcnt,PetscInt cnt)
{
  PetscErrorCode ierr;
  MPI_Comm       comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
  if (i >= *mcnt) {
    *mcnt += cnt;
    ierr = MPI_Bcast(mcnt,1,MPIU_INT,0,comm);CHKERRMPI(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscViewerFlowControlEndMain(PetscViewer viewer,PetscInt *mcnt)
{
  PetscErrorCode ierr;
  MPI_Comm       comm;
  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
  *mcnt = 0;
  ierr = MPI_Bcast(mcnt,1,MPIU_INT,0,comm);CHKERRMPI(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscViewerFlowControlStepWorker(PetscViewer viewer,PetscMPIInt rank,PetscInt *mcnt)
{
  PetscErrorCode ierr;
  MPI_Comm       comm;
  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
  while (PETSC_TRUE) {
    if (rank < *mcnt) break;
    ierr = MPI_Bcast(mcnt,1,MPIU_INT,0,comm);CHKERRMPI(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscViewerFlowControlEndWorker(PetscViewer viewer,PetscInt *mcnt)
{
  PetscErrorCode ierr;
  MPI_Comm       comm;
  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
  while (PETSC_TRUE) {
    ierr = MPI_Bcast(mcnt,1,MPIU_INT,0,comm);CHKERRMPI(ierr);
    if (!*mcnt) break;
  }
  PetscFunctionReturn(0);
}
