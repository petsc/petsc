#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: mathematica.c,v 1.9 2000/01/26 15:46:22 baggag Exp $";
#endif

/* 
        Written by Matt Knepley, knepley@cs.purdue.edu 7/23/97
        Major overhall for interactivity               11/14/97
        Reorganized                                    11/8/98
*/
#include "src/sys/src/viewer/viewerimpl.h"   /* "petsc.h" */
#include "src/ksp/pc/pcimpl.h"
#include "src/mat/impls/aij/seq/aij.h"
#include "mathematica.h"
#include "petscfix.h"

#if defined (PETSC_HAVE__SNPRINTF)
#define snprintf _snprintf
#endif

PetscViewer  PETSC_VIEWER_MATHEMATICA_WORLD_PRIVATE = PETSC_NULL;
#ifdef PETSC_HAVE_MATHEMATICA
static void *mathematicaEnv                   = PETSC_NULL;
#endif

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerMathematicaInitializePackage"
/*@C
  PetscViewerMathematicaInitializePackage - This function initializes everything in the Petsc interface to Mathematica. It is
  called from PetscDLLibraryRegister() when using dynamic libraries, and on the call to PetscInitialize()
  when using static libraries.

  Input Parameter:
  path - The dynamic library path, or PETSC_NULL

  Level: developer

.keywords: Petsc, initialize, package, PLAPACK
.seealso: PetscInitializePackage(), PetscInitialize()
@*/
int PetscViewerMathematicaInitializePackage(char *path) {
  static PetscTruth initialized = PETSC_FALSE;

  PetscFunctionBegin;
  if (initialized == PETSC_TRUE) PetscFunctionReturn(0);
  initialized = PETSC_TRUE;
#ifdef PETSC_HAVE_MATHEMATICA
  mathematicaEnv = (void *) MLInitialize(0);
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerMathematicaDestroyPackage"
/*@C
  PetscViewerMathematicaDestroyPackage - This function destroys everything in the Petsc interface to Mathematica. It is
  called from PetscFinalize().

  Level: developer

.keywords: Petsc, destroy, package, mathematica
.seealso: PetscFinalize()
@*/
int PetscViewerMathematicaFinalizePackage(void) {
  PetscFunctionBegin;
#ifdef PETSC_HAVE_MATHEMATICA
  if (mathematicaEnv != PETSC_NULL) MLDeinitialize((MLEnvironment) mathematicaEnv);
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerInitializeMathematicaWorld_Private"
int PetscViewerInitializeMathematicaWorld_Private()
{
  int ierr;

  PetscFunctionBegin;
  if (PETSC_VIEWER_MATHEMATICA_WORLD_PRIVATE) PetscFunctionReturn(0);
  ierr = PetscViewerMathematicaOpen(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_NULL, PETSC_NULL, &PETSC_VIEWER_MATHEMATICA_WORLD_PRIVATE);
  CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerDestroy_Mathematica"
static int PetscViewerDestroy_Mathematica(PetscViewer viewer)
{
  PetscViewer_Mathematica *vmath = (PetscViewer_Mathematica *) viewer->data; 
  int                 ierr;

  PetscFunctionBegin;
#ifdef PETSC_HAVE_MATHEMATICA
  MLClose(vmath->link);
#endif
  if (vmath->linkname != PETSC_NULL) {
    ierr = PetscFree(vmath->linkname);                                                                    CHKERRQ(ierr);
  }
  if (vmath->linkhost != PETSC_NULL) {
    ierr = PetscFree(vmath->linkhost);                                                                    CHKERRQ(ierr);
  }
  ierr = PetscFree(vmath);                                                                                CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerDestroyMathematica_Private"
int PetscViewerDestroyMathematica_Private(void)
{
  int ierr;

  PetscFunctionBegin;
  if (PETSC_VIEWER_MATHEMATICA_WORLD_PRIVATE) {
    ierr = PetscViewerDestroy(PETSC_VIEWER_MATHEMATICA_WORLD_PRIVATE);                                    CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerMathematicaSetupConnection_Private"
int PetscViewerMathematicaSetupConnection_Private(PetscViewer v) {
#ifdef PETSC_HAVE_MATHEMATICA
  PetscViewer_Mathematica *vmath = (PetscViewer_Mathematica *) v->data;
#ifdef MATHEMATICA_3_0
  int                      argc = 6;
  char                    *argv[6];
#else
  int                      argc = 5;
  char                    *argv[5];
#endif
  char                     hostname[256];
  long                     lerr;
  int                      ierr;
#endif

  PetscFunctionBegin;
#ifdef PETSC_HAVE_MATHEMATICA
  /* Link name */
  argv[0] = "-linkname";
  if (vmath->linkname == PETSC_NULL) {
    argv[1] = "math -mathlink";
  } else {
    argv[1] = vmath->linkname;
  }

  /* Link host */
  argv[2] = "-linkhost";
  if (vmath->linkhost == PETSC_NULL) {
    ierr = PetscGetHostName(hostname, 255);                                                               CHKERRQ(ierr);
    argv[3] = hostname;
  } else {
    argv[3] = vmath->linkhost;
  }

  /* Link mode */
#ifdef MATHEMATICA_3_0
  argv[4] = "-linkmode";
  switch(vmath->linkmode) {
  case MATHEMATICA_LINK_CREATE:
    argv[5] = "Create";
    break;
  case MATHEMATICA_LINK_CONNECT:
    argv[5] = "Connect";
    break;
  case MATHEMATICA_LINK_LAUNCH:
    argv[5] = "Launch";
    break;
  }
#else
  switch(vmath->linkmode) {
  case MATHEMATICA_LINK_CREATE:
    argv[4] = "-linkcreate";
    break;
  case MATHEMATICA_LINK_CONNECT:
    argv[4] = "-linkconnect";
    break;
  case MATHEMATICA_LINK_LAUNCH:
    argv[4] = "-linklaunch";
    break;
  }
#endif

  vmath->link = MLOpenInEnv(mathematicaEnv, argc, argv, &lerr);
#endif

  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PetscViewerCreate_Mathematica"
int PetscViewerCreate_Mathematica(PetscViewer v) {
  PetscViewer_Mathematica *vmath;
  int                      ierr;

  PetscFunctionBegin;

  ierr = PetscNew(PetscViewer_Mathematica, &vmath);                                                       CHKERRQ(ierr);
  v->data         = (void *) vmath;
  v->ops->destroy = PetscViewerDestroy_Mathematica;
  v->ops->flush   = 0;
  ierr = PetscStrallocpy(PETSC_VIEWER_MATHEMATICA, &v->type_name);                                        CHKERRQ(ierr);

  vmath->linkname         = PETSC_NULL;
  vmath->linkhost         = PETSC_NULL;
  vmath->linkmode         = MATHEMATICA_LINK_CONNECT;
  vmath->graphicsType     = GRAPHICS_MOTIF;
  vmath->plotType         = MATHEMATICA_TRIANGULATION_PLOT;
  vmath->objName          = PETSC_NULL;

  ierr = PetscViewerMathematicaSetFromOptions(v);                                                         CHKERRQ(ierr);
  ierr = PetscViewerMathematicaSetupConnection_Private(v);                                                CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerMathematicaParseLinkMode_Private"
int PetscViewerMathematicaParseLinkMode_Private(char *modename, LinkMode *mode) {
  PetscTruth isCreate, isConnect, isLaunch;
  int        ierr;

  PetscFunctionBegin;
  ierr = PetscStrcasecmp(modename, "Create",  &isCreate);                                                 CHKERRQ(ierr);
  ierr = PetscStrcasecmp(modename, "Connect", &isConnect);                                                CHKERRQ(ierr);
  ierr = PetscStrcasecmp(modename, "Launch",  &isLaunch);                                                 CHKERRQ(ierr);
  if (isCreate == PETSC_TRUE) {
    *mode = MATHEMATICA_LINK_CREATE;
  } else if (isConnect == PETSC_TRUE) {
    *mode = MATHEMATICA_LINK_CONNECT;
  } else if (isLaunch == PETSC_TRUE) {
    *mode = MATHEMATICA_LINK_LAUNCH;
  } else {
    SETERRQ1(PETSC_ERR_ARG_WRONG, "Invalid Mathematica link mode: %s", modename);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerMathematicaSetFromOptions"
int PetscViewerMathematicaSetFromOptions(PetscViewer v) {
  PetscViewer_Mathematica *vmath = (PetscViewer_Mathematica *) v->data;
  char                     linkname[256];
  char                     modename[256];
  char                     hostname[256];
  char                     type[256];
  int                      numPorts;
  int                     *ports;
  int                      numHosts;
  char                   **hosts;
  int                      size, rank;
  int                      h;
  PetscTruth               opt;
  int                      ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(v->comm, &size);                                                                   CHKERRQ(ierr);
  ierr = MPI_Comm_rank(v->comm, &rank);                                                                   CHKERRQ(ierr);

  /* Get link name */
  ierr = PetscOptionsGetString("viewer_", "-math_linkname", linkname, 255, &opt);                         CHKERRQ(ierr);
  if (opt == PETSC_TRUE) {
    ierr = PetscViewerMathematicaSetLinkName(v, linkname);                                                CHKERRQ(ierr);
  }
  /* Get link port */
  numPorts = size;
  ierr = PetscMalloc(size * sizeof(int), &ports);                                                         CHKERRQ(ierr);
  ierr = PetscOptionsGetIntArray("viewer_", "-math_linkport", ports, &numPorts, &opt);                    CHKERRQ(ierr);
  if (opt == PETSC_TRUE) {
    if (numPorts > rank) {
      snprintf(linkname, 255, "%6d", ports[rank]);
    } else {
      snprintf(linkname, 255, "%6d", ports[0]);
    }
    ierr = PetscViewerMathematicaSetLinkName(v, linkname);                                                CHKERRQ(ierr);
  }
  ierr = PetscFree(ports);                                                                                CHKERRQ(ierr);
  /* Get link host */
  numHosts = size;
  ierr = PetscMalloc(size * sizeof(char *), &hosts);                                                      CHKERRQ(ierr);
  ierr = PetscOptionsGetStringArray("viewer_", "-math_linkhost", hosts, &numHosts, &opt);                 CHKERRQ(ierr);
  if (opt == PETSC_TRUE) {
    if (numHosts > rank) {
      ierr = PetscStrncpy(hostname, hosts[rank], 255);                                                    CHKERRQ(ierr);
    } else {
      ierr = PetscStrncpy(hostname, hosts[0], 255);                                                       CHKERRQ(ierr);
    }
    ierr = PetscViewerMathematicaSetLinkHost(v, hostname);                                                CHKERRQ(ierr);
  }
  for(h = 0; h < numHosts; h++) {
    ierr = PetscFree(hosts[h]);                                                                           CHKERRQ(ierr);
  }
  ierr = PetscFree(hosts);                                                                                CHKERRQ(ierr);
  /* Get link mode */
  ierr = PetscOptionsGetString("viewer_", "-math_linkmode", modename, 255, &opt);                         CHKERRQ(ierr);
  if (opt == PETSC_TRUE) {
    LinkMode mode;

    ierr = PetscViewerMathematicaParseLinkMode_Private(modename, &mode);                                  CHKERRQ(ierr);
    ierr = PetscViewerMathematicaSetLinkMode(v, mode);                                                    CHKERRQ(ierr);
  }
  /* Get graphics type */
  ierr = PetscOptionsGetString("viewer_", "-math_graphics", type, 255, &opt);                             CHKERRQ(ierr);
  if (opt == PETSC_TRUE) {
    PetscTruth isMotif, isPS, isPSFile;

    ierr = PetscStrcasecmp(type, "Motif",  &isMotif);                                                     CHKERRQ(ierr);
    ierr = PetscStrcasecmp(type, "PS",     &isPS);                                                        CHKERRQ(ierr);
    ierr = PetscStrcasecmp(type, "PSFile", &isPSFile);                                                    CHKERRQ(ierr);
    if (isMotif == PETSC_TRUE) {
      vmath->graphicsType = GRAPHICS_MOTIF;
    } else if (isPS == PETSC_TRUE) {
      vmath->graphicsType = GRAPHICS_PS_STDOUT;
    } else if (isPSFile == PETSC_TRUE) {
      vmath->graphicsType = GRAPHICS_PS_FILE;
    }
  }
  /* Get plot type */
  ierr = PetscOptionsGetString("viewer_", "-math_type", type, 255, &opt);                                 CHKERRQ(ierr);
  if (opt == PETSC_TRUE) {
    PetscTruth isTri, isVecTri, isVec, isSurface;

    ierr = PetscStrcasecmp(type, "Triangulation",       &isTri);                                          CHKERRQ(ierr);
    ierr = PetscStrcasecmp(type, "VectorTriangulation", &isVecTri);                                       CHKERRQ(ierr);
    ierr = PetscStrcasecmp(type, "Vector",              &isVec);                                          CHKERRQ(ierr);
    ierr = PetscStrcasecmp(type, "Surface",             &isSurface);                                      CHKERRQ(ierr);
    if (isTri == PETSC_TRUE) {
      vmath->plotType     = MATHEMATICA_TRIANGULATION_PLOT;
    } else if (isVecTri == PETSC_TRUE) {
      vmath->plotType     = MATHEMATICA_VECTOR_TRIANGULATION_PLOT;
    } else if (isVec == PETSC_TRUE) {
      vmath->plotType     = MATHEMATICA_VECTOR_PLOT;
    } else if (isSurface == PETSC_TRUE) {
      vmath->plotType     = MATHEMATICA_SURFACE_PLOT;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerMathematicaSetLinkName"
int PetscViewerMathematicaSetLinkName(PetscViewer v, const char *name) {
  PetscViewer_Mathematica *vmath = (PetscViewer_Mathematica *) v->data;
  int                      ierr;

  PetscFunctionBegin;
  PetscValidCharPointer(name);
  ierr = PetscStrallocpy(name, &vmath->linkname);                                                         CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerMathematicaLinkPort"
int PetscViewerMathematicaSetLinkPort(PetscViewer v, int port) {
  char name[16];
  int  ierr;

  PetscFunctionBegin;
  snprintf(name, 16, "%6d", port);
  ierr = PetscViewerMathematicaSetLinkName(v, name);                                                      CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerMathematicaSetLinkHost"
int PetscViewerMathematicaSetLinkHost(PetscViewer v, const char *host) {
  PetscViewer_Mathematica *vmath = (PetscViewer_Mathematica *) v->data;
  int                      ierr;

  PetscFunctionBegin;
  PetscValidCharPointer(host);
  ierr = PetscStrallocpy(host, &vmath->linkhost);                                                         CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerMathematicaSetLinkHost"
int PetscViewerMathematicaSetLinkMode(PetscViewer v, LinkMode mode) {
  PetscViewer_Mathematica *vmath = (PetscViewer_Mathematica *) v->data;

  PetscFunctionBegin;
  vmath->linkmode = mode;
  PetscFunctionReturn(0);
}

/*----------------------------------------- Public Functions --------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "PetscViewerMathematicaOpen"
/*@C
  PetscViewerMathematicaOpen - Communicates with Mathemtica using MathLink.

  Collective on comm

  Input Parameters:
+ comm    - The MPI communicator
. port    - [optional] The port to connect on, or PETSC_DECIDE
. machine - [optional] The machine to run Mathematica on, or PETSC_NULL
- mode    - [optional] The connection mode, or PETSC_NULL

  Output Parameter:
. viewer  - The Mathematica viewer

  Level: intermediate

  Notes:
  Most users should employ the following commands to access the 
  Mathematica viewers
$
$    PetscViewerMathematicaOpen(MPI_Comm comm, int port, char *machine, char *mode, PetscViewer &viewer)
$    MatView(Mat matrix, PetscViewer viewer)
$
$                or
$
$    PetscViewerMathematicaOpen(MPI_Comm comm, int port, char *machine, char *mode, PetscViewer &viewer)
$    VecView(Vec vector, PetscViewer viewer)

   Options Database Keys:
$    -viewer_math_linkhost <machine> - The host machine for the kernel
$    -viewer_math_linkname <name>    - The full link name for the connection
$    -viewer_math_linkport <port>    - The port for the connection
$    -viewer_math_mode <mode>        - The mode, e.g. Launch, Connect
$    -viewer_math_type <type>        - The plot type, e.g. Triangulation, Vector
$    -viewer_math_graphics <output>  - The output type, e.g. Motif, PS, PSFile

.keywords: PetscViewer, Mathematica, open

.seealso: MatView(), VecView()
@*/
int PetscViewerMathematicaOpen(MPI_Comm comm, int port, const char machine[], const char mode[], PetscViewer *v) {
  int      ierr;

  PetscFunctionBegin;
  ierr = PetscViewerCreate(comm, v);                                                                      CHKERRQ(ierr);
#if 0
  LinkMode linkmode;
  ierr = PetscViewerMathematicaSetLinkPort(*v, port);                                                     CHKERRQ(ierr);
  ierr = PetscViewerMathematicaSetLinkHost(*v, machine);                                                  CHKERRQ(ierr);
  ierr = PetscViewerMathematicaParseLinkMode_Private(mode, &linkmode);                                    CHKERRQ(ierr);
  ierr = PetscViewerMathematicaSetLinkMode(*v, linkmode);                                                 CHKERRQ(ierr);
#endif
  ierr = PetscViewerSetType(*v, PETSC_VIEWER_MATHEMATICA);                                                CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#ifdef PETSC_HAVE_MATHEMATICA
#undef __FUNCT__  
#define __FUNCT__ "PetscViewerMathematicaGetLink"
/*@C
  PetscViewerMathematicaGetLink - Returns the link to Mathematica

  Input Parameters:
. viewer - The Mathematica viewer
. link   - The link to Mathematica

  Level: intermediate

.keywords PetscViewer, Mathematica, link
.seealso PetscViewerMathematicaOpen()
@*/
int PetscViewerMathematicaGetLink(PetscViewer viewer, MLINK *link)
{
  PetscViewer_Mathematica *vmath = (PetscViewer_Mathematica *) viewer->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_COOKIE);
  *link = vmath->link;
  PetscFunctionReturn(0);
}
#endif

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerMathematicaSkipPackets"
/*@C
  PetscViewerMathematicaSkipPackets - Discard packets sent by Mathematica until a certain packet type is received

  Input Parameters:
. viewer - The Mathematica viewer
. type   - The packet type to search for, e.g RETURNPKT

  Level: advanced

.keywords PetscViewer, Mathematica, packets
.seealso PetscViewerMathematicaSetName(), PetscViewerMathematicaGetVector()
@*/
int PetscViewerMathematicaSkipPackets(PetscViewer viewer, int type)
{
#ifdef PETSC_HAVE_MATHEMATICA
  PetscViewer_Mathematica *vmath = (PetscViewer_Mathematica *) viewer->data;
  MLINK               link  = vmath->link; /* The link to Mathematica */
  int                 pkt;                 /* The packet type */

  PetscFunctionBegin;
  while((pkt = MLNextPacket(link)) && (pkt != type))
    MLNewPacket(link);
  if (!pkt) {
    MLClearError(link);
    SETERRQ(PETSC_ERR_LIB, (char *) MLErrorMessage(link));
  }
  PetscFunctionReturn(0);
#endif
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerMathematicaLoadGraphics"
/*@C
  PetscViewerMathematicaLoadGraphics - Change the type of graphics output for Mathematica

  Input Parameters:
. viewer - The Mathematica viewer
. type   - The type of graphics, e.g. GRAPHICS_MOTIF, GRAPHICS_PS_FILE, GRAPHICS_PS_STDOUT

  Level: intermediate

.keywords PetscViewer, Mathematica, packets
.seealso PetscViewerMathematicaSetName(), PetscViewerMathematicaSkipPackets()
@*/
int PetscViewerMathematicaLoadGraphics(PetscViewer viewer, GraphicsType type)
{
#ifdef PETSC_HAVE_MATHEMATICA
  PetscViewer_Mathematica *vmath = (PetscViewer_Mathematica *) viewer->data;
  MLINK               link  = vmath->link; /* The link to Mathematica */
  char                programName[256];
  int                 ierr;

  PetscFunctionBegin;
  /* Load graphics package */
  MLPutFunction(link, "EvaluatePacket", 1);
    switch(type)
    {
    case GRAPHICS_MOTIF: 
        MLPutFunction(link, "Get", 1);
          MLPutString(link, "Motif.m");
      break;
    case GRAPHICS_PS_FILE:
    MLPutFunction(link, "CompoundExpression", 4);
      MLPutFunction(link, "Set", 2);
        MLPutSymbol(link, "PetscGraphicsCounter");
        MLPutInteger(link, 0);
      MLPutFunction(link, "SetDelayed", 2);
        MLPutSymbol(link, "$Display");
        MLPutSymbol(link, "$FileDisplay");
      MLPutFunction(link, "Set", 2);
        MLPutSymbol(link, "$FileDisplay");
        MLPutFunction(link, "OpenWrite", 1);
          MLPutFunction(link, "StringJoin", 3);
          if (!PetscGetProgramName(programName, 255))
            MLPutString(link, programName);
          else
            MLPutString(link, "GVec");
            MLPutFunction(link, "ToString", 1);
              MLPutSymbol(link, "PetscGraphicsCounter");
            MLPutString(link, ".ps");
      MLPutFunction(link, "Set", 2);
        MLPutSymbol(link, "$DisplayFunction");
        MLPutFunction(link, "Function", 1);
          MLPutFunction(link, "CompoundExpression", 2);
            MLPutFunction(link, "Display", 3);
              MLPutSymbol(link, "$Display");
              MLPutFunction(link, "Slot", 1);
                MLPutInteger(link, 1);
              MLPutString(link, "EPS");
            MLPutFunction(link, "Increment", 1);
              MLPutSymbol(link, "PetscGraphicsCounter");
      break;
    case GRAPHICS_PS_STDOUT:
      MLPutFunction(link, "Get", 1);
        MLPutString(link, "PSDirect.m");
      break;
    default:
      SETERRQ1(PETSC_ERR_ARG_WRONG, "Invalid graphics type: %d", type);
    }
  MLEndPacket(link);

  /* Skip packets until ReturnPacket */
  ierr = PetscViewerMathematicaSkipPackets(viewer, RETURNPKT);                                                CHKERRQ(ierr);
  /* Skip ReturnPacket */
  MLNewPacket(link);

  /* Load PlotField.m for vector plots */
  MLPutFunction(link, "EvaluatePacket", 1);
    MLPutFunction(link, "Get", 1);
      MLPutString(link, "Graphics/PlotField.m");
  MLEndPacket(link);

  /* Skip packets until ReturnPacket */
  ierr = PetscViewerMathematicaSkipPackets(viewer, RETURNPKT);                                                CHKERRQ(ierr);
  /* Skip ReturnPacket */
  MLNewPacket(link);

  PetscFunctionReturn(0);
#else
  PetscFunctionBegin;
  switch(type)
  {
  case GRAPHICS_MOTIF: 
  case GRAPHICS_PS_FILE:
  case GRAPHICS_PS_STDOUT:
    break;
  default:
    SETERRQ1(PETSC_ERR_ARG_WRONG, "Invalid graphics type: %d", type);
  }
  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerMathematicaGetName"
/*@C
  PetscViewerMathematicaGetName - Retrieve the default name for objects communicated to Mathematica

  Input Parameter:
. viewer - The Mathematica viewer

  Output Parameter:
. name   - The name for new objects created in Mathematica

  Level: intermediate

.keywords PetscViewer, Mathematica, name
.seealso PetscViewerMathematicaSetName(), PetscViewerMathematicaClearName()
@*/
int PetscViewerMathematicaGetName(PetscViewer viewer, const char **name)
{
  PetscViewer_Mathematica *vmath = (PetscViewer_Mathematica *) viewer->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_COOKIE);
  PetscValidPointer(name);
  *name = vmath->objName;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerMathematicaSetName"
/*@C
  PetscViewerMathematicaSetName - Override the default name for objects communicated to Mathematica

  Input Parameters:
. viewer - The Mathematica viewer
. name   - The name for new objects created in Mathematica

  Level: intermediate

.keywords PetscViewer, Mathematica, name
.seealso PetscViewerMathematicaSetName(), PetscViewerMathematicaClearName()
@*/
int PetscViewerMathematicaSetName(PetscViewer viewer, const char name[])
{
  PetscViewer_Mathematica *vmath = (PetscViewer_Mathematica *) viewer->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_COOKIE);
  PetscValidPointer(name);
  vmath->objName = name;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerMathematicaClearName"
/*@C
  PetscViewerMathematicaClearName - Use the default name for objects communicated to Mathematica

  Input Parameter:
. viewer - The Mathematica viewer

  Level: intermediate

.keywords PetscViewer, Mathematica, name
.seealso PetscViewerMathematicaGetName(), PetscViewerMathematicaSetName()
@*/
int PetscViewerMathematicaClearName(PetscViewer viewer)
{
  PetscViewer_Mathematica *vmath = (PetscViewer_Mathematica *) viewer->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_COOKIE);
  vmath->objName = PETSC_NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerMathematicaGetVector"
/*@C
  PetscViewerMathematicaGetVector - Retrieve a vector from Mathematica

  Input Parameter:
. viewer - The Mathematica viewer

  Output Parameter:
. v      - The vector

  Level: intermediate

.keywords PetscViewer, Mathematica, vector
.seealso VecView(), PetscViewerMathematicaPutVector()
@*/
int PetscViewerMathematicaGetVector(PetscViewer viewer, Vec v) {
#ifdef PETSC_HAVE_MATHEMATICA
  PetscViewer_Mathematica *vmath = (PetscViewer_Mathematica *) viewer->data;
  MLINK               link;   /* The link to Mathematica */
  char               *name;
  PetscScalar        *mArray;
  PetscScalar        *array;
  long                mSize;
  int                 size;
  int                 ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_COOKIE);
  PetscValidHeaderSpecific(v,      VEC_COOKIE);

  /* Determine the object name */
  if (vmath->objName == PETSC_NULL) {
    name = "vec";
  } else {
    name = (char *) vmath->objName;
  }

  link = vmath->link;
  ierr = VecGetLocalSize(v, &size);                                                                      CHKERRQ(ierr);
  ierr = VecGetArray(v, &array);                                                                         CHKERRQ(ierr);
  MLPutFunction(link, "EvaluatePacket", 1);
    MLPutSymbol(link, name);
  MLEndPacket(link);
  ierr = PetscViewerMathematicaSkipPackets(viewer, RETURNPKT);                                                CHKERRQ(ierr);
  MLGetRealList(link, &mArray, &mSize);
  if (size != mSize) SETERRQ(PETSC_ERR_ARG_WRONG, "Incompatible vector sizes");
  ierr = PetscMemcpy(array, mArray, mSize * sizeof(double));                                             CHKERRQ(ierr);
  MLDisownRealList(link, mArray, mSize);
  ierr = VecRestoreArray(v, &array);                                                                     CHKERRQ(ierr);

  PetscFunctionReturn(0);
#endif
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerMathematicaPutVector"
/*@C
  PetscViewerMathematicaPutVector - Send a vector to Mathematica

  Input Parameters:
+ viewer - The Mathematica viewer
- v      - The vector

  Level: intermediate

.keywords PetscViewer, Mathematica, vector
.seealso VecView(), PetscViewerMathematicaGetVector()
@*/
int PetscViewerMathematicaPutVector(PetscViewer viewer, Vec v) {
#ifdef PETSC_HAVE_MATHEMATICA
  PetscViewer_Mathematica *vmath = (PetscViewer_Mathematica *) viewer->data;
  MLINK               link  = vmath->link; /* The link to Mathematica */
  char               *name;
  PetscScalar        *array;
  int                 size;
  int                 ierr;

  PetscFunctionBegin;
  /* Determine the object name */
  if (vmath->objName == PETSC_NULL) {
    name = "vec";
  } else {
    name = (char *) vmath->objName;
  }
  ierr = VecGetLocalSize(v, &size);                                                                       CHKERRQ(ierr);
  ierr = VecGetArray(v, &array);                                                                          CHKERRQ(ierr);

  /* Send the Vector object */
  MLPutFunction(link, "EvaluatePacket", 1);
    MLPutFunction(link, "Set", 2);
      MLPutSymbol(link, name);
      MLPutRealList(link, array, size);
  MLEndPacket(link);
  /* Skip packets until ReturnPacket */
  ierr = PetscViewerMathematicaSkipPackets(viewer, RETURNPKT);                                            CHKERRQ(ierr);
  /* Skip ReturnPacket */
  MLNewPacket(link);

  ierr = VecRestoreArray(v, &array);                                                                      CHKERRQ(ierr);
  PetscFunctionReturn(0);
#endif
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

int PetscViewerMathematicaPutMatrix(PetscViewer viewer, int m, int n, PetscReal *a) {
#ifdef PETSC_HAVE_MATHEMATICA
  PetscViewer_Mathematica *vmath = (PetscViewer_Mathematica *) viewer->data;
  MLINK               link  = vmath->link; /* The link to Mathematica */
  char               *name;
  int                 ierr;

  PetscFunctionBegin;
  /* Determine the object name */
  if (vmath->objName == PETSC_NULL) {
    name = "mat";
  } else {
    name = (char *) vmath->objName;
  }

  /* Send the dense matrix object */
  MLPutFunction(link, "EvaluatePacket", 1);
    MLPutFunction(link, "Set", 2);
      MLPutSymbol(link, name);
      MLPutFunction(link, "Transpose", 1);
        MLPutFunction(link, "Partition", 2);
          MLPutRealList(link, a, m*n);
          MLPutInteger(link, m);
  MLEndPacket(link);
  /* Skip packets until ReturnPacket */
  ierr = PetscViewerMathematicaSkipPackets(viewer, RETURNPKT);                                                CHKERRQ(ierr);
  /* Skip ReturnPacket */
  MLNewPacket(link);

  PetscFunctionReturn(0);
#endif
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

int PetscViewerMathematicaPutCSRMatrix(PetscViewer viewer, int m, int n, int *i, int *j, PetscReal *a) {
#ifdef PETSC_HAVE_MATHEMATICA
  PetscViewer_Mathematica *vmath = (PetscViewer_Mathematica *) viewer->data;
  MLINK               link  = vmath->link; /* The link to Mathematica */
  const char         *symbol;
  char               *name;
  PetscTruth          match;
  int                 ierr;

  PetscFunctionBegin;
  /* Determine the object name */
  if (vmath->objName == PETSC_NULL) {
    name = "mat";
  } else {
    name = (char *) vmath->objName;
  }

  /* Make sure Mathematica recognizes sparse matrices */
  MLPutFunction(link, "EvaluatePacket", 1);
    MLPutFunction(link, "Needs", 1);
      MLPutString(link, "LinearAlgebra`CSRMatrix`");
  MLEndPacket(link);
  /* Skip packets until ReturnPacket */
  ierr = PetscViewerMathematicaSkipPackets(viewer, RETURNPKT);                                                CHKERRQ(ierr);
  /* Skip ReturnPacket */
  MLNewPacket(link);

  /* Send the CSRMatrix object */
  MLPutFunction(link, "EvaluatePacket", 1);
    MLPutFunction(link, "Set", 2);
      MLPutSymbol(link, name);
      MLPutFunction(link, "CSRMatrix", 5);
        MLPutInteger(link, m);
        MLPutInteger(link, n);
        MLPutFunction(link, "Plus", 2);
          MLPutIntegerList(link, i, m+1);
          MLPutInteger(link, 1);
        MLPutFunction(link, "Plus", 2);
          MLPutIntegerList(link, j, i[m]);
          MLPutInteger(link, 1);
        MLPutRealList(link, a, i[m]);
  MLEndPacket(link);
  /* Skip packets until ReturnPacket */
  ierr = PetscViewerMathematicaSkipPackets(viewer, RETURNPKT);                                                CHKERRQ(ierr);
  /* Skip ReturnPacket */
  MLNewPacket(link);

  /* Check that matrix is valid */
  MLPutFunction(link, "EvaluatePacket", 1);
    MLPutFunction(link, "ValidQ", 1);
      MLPutSymbol(link, name);
  MLEndPacket(link);
  ierr = PetscViewerMathematicaSkipPackets(viewer, RETURNPKT);                                                CHKERRQ(ierr);
  MLGetSymbol(link, &symbol);
  ierr = PetscStrcmp("True", (char *) symbol, &match);                                                   CHKERRQ(ierr);
  if (match == PETSC_FALSE) {
    MLDisownSymbol(link, symbol);
    SETERRQ(PETSC_ERR_PLIB, "Invalid CSR matrix in Mathematica");
  }
  MLDisownSymbol(link, symbol);
  /* Skip ReturnPacket */
  MLNewPacket(link);

  PetscFunctionReturn(0);
#endif
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/*------------------------------------------- ML Functions ----------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "PetscViewerMathematicaPartitionMesh"
int PetscViewerMathematicaPartitionMesh(PetscViewer viewer, int numElements, int numVertices, double *vertices, int **mesh,
                                   int *numParts, int *colPartition)
{
#ifdef PETSC_HAVE_MATHEMATICA
  PetscViewer_Mathematica *vmath = (PetscViewer_Mathematica *) viewer->data;
  MLINK               link;   /* The link to Mathematica */
  const char         *symbol;
  int                 numOptions;
  int                 partSize;
  int                *part;
  long                numCols;
  int                 col;
  PetscTruth          match, opt;
  int                 ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_COOKIE);
  link = vmath->link;

  /* Make sure that the reduce.m package is loaded */
  MLPutFunction(link, "EvaluatePacket", 1);
    MLPutFunction(link, "Get", 1);
      MLPutFunction(link, "StringJoin", 2);
        MLPutFunction(link, "Environment", 1);
          MLPutString(link, "PETSC_DIR");
        MLPutString(link, "/src/pc/impls/ml/reduce.m");
  MLEndPacket(link);
  ierr = PetscViewerMathematicaSkipPackets(viewer, RETURNPKT);                                            CHKERRQ(ierr);
  MLGetSymbol(link, &symbol);
  ierr = PetscStrcmp("$Failed", (char *) symbol, &match);                                                 CHKERRQ(ierr);
  if (match == PETSC_TRUE) {
    MLDisownSymbol(link, symbol);
    SETERRQ(PETSC_ERR_FILE_OPEN, "Unable to load package reduce.m");
  }
  MLDisownSymbol(link, symbol);

  /* Partition the mesh */
  numOptions = 0;
  partSize   = 0;
  ierr = PetscOptionsGetInt(PETSC_NULL, "-pc_ml_partition_size", &partSize, &opt);                        CHKERRQ(ierr);
  if ((opt == PETSC_TRUE) && (partSize > 0))
    numOptions++;
  MLPutFunction(link, "EvaluatePacket", 1);
    MLPutFunction(link, "PartitionMesh", 1 + numOptions);
      MLPutFunction(link, "MeshData", 5);
        MLPutInteger(link, numElements);
        MLPutInteger(link, numVertices);
        MLPutInteger(link, numVertices);
        MLPutFunction(link, "Partition", 2);
          MLPutRealList(link, vertices, numVertices*2);
          MLPutInteger(link, 2);
        MLPutFunction(link, "Partition", 2);
          MLPutIntegerList(link, mesh[MESH_ELEM], numElements*3);
          MLPutInteger(link, 3);
      if (partSize > 0)
      {
        MLPutFunction(link, "Rule", 2);
          MLPutSymbol(link, "PartitionSize");
          MLPutInteger(link, partSize);
      }
  MLEndPacket(link);
  /* Skip packets until ReturnPacket */
  ierr = PetscViewerMathematicaSkipPackets(viewer, RETURNPKT);                                                CHKERRQ(ierr);

  /* Get the vertex partiton */
  MLGetIntegerList(link, &part, &numCols);
  if (numCols != numVertices) SETERRQ(PETSC_ERR_PLIB, "Invalid partition");
  for(col = 0, *numParts = 0; col < numCols; col++) {
    colPartition[col] = part[col]-1;
    *numParts = PetscMax(*numParts, part[col]);
  }
  MLDisownIntegerList(link, part, numCols);

  PetscFunctionReturn(0);
#endif
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerMathematicaReduce"
int PetscViewerMathematicaReduce(PetscViewer viewer, PC pc, int thresh)
{
#ifdef PETSC_HAVE_MATHEMATICA
  PetscViewer_Mathematica *vmath = (PetscViewer_Mathematica *) viewer->data;
  MLINK               link;   /* The link to Mathematica */
  PC_Multilevel      *ml;
  int                *range;
  long                numRange;
  int                *null;
  long                numNull;
  const char         *symbol;
  int                 numOptions;
  int                 partSize;
  int                 row, col;
  PetscTruth          match, opt;
  int                 ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_COOKIE);
  PetscValidHeaderSpecific(pc,     PC_COOKIE);
  link = vmath->link;
  ml   = (PC_Multilevel *) pc->data;

  /* Make sure that the reduce.m package is loaded */
  MLPutFunction(link, "EvaluatePacket", 1);
    MLPutFunction(link, "Get", 1);
      MLPutFunction(link, "StringJoin", 2);
        MLPutFunction(link, "Environment", 1);
          MLPutString(link, "PETSC_DIR");
        MLPutString(link, "/src/pc/impls/ml/reduce.m");
  MLEndPacket(link);
  ierr = PetscViewerMathematicaSkipPackets(viewer, RETURNPKT);                                                 CHKERRQ(ierr);
  MLGetSymbol(link, &symbol);
  ierr = PetscStrcmp("$Failed", (char *) symbol, &match);                                                 CHKERRQ(ierr);
  if (match == PETSC_TRUE) {
    MLDisownSymbol(link, symbol);
    SETERRQ(PETSC_ERR_FILE_OPEN, "Unable to load package reduce.m");
  }
  MLDisownSymbol(link, symbol);

  /* mesh = MeshData[] */
  MLPutFunction(link, "EvaluatePacket", 1);
    MLPutFunction(link, "Set", 2);
      MLPutSymbol(link, "mesh");
      MLPutFunction(link, "MeshData", 5);
        MLPutInteger(link, ml->numElements[0]);
        MLPutInteger(link, ml->numVertices[0]);
        MLPutInteger(link, ml->numVertices[0]);
        MLPutFunction(link, "Partition", 2);
          MLPutRealList(link, ml->vertices, ml->numVertices[0]*2);
          MLPutInteger(link, 2);
        MLPutFunction(link, "Partition", 2);
          MLPutIntegerList(link, ml->meshes[0][MESH_ELEM], ml->numElements[0]*3);
          MLPutInteger(link, 3);
  MLEndPacket(link);
  /* Skip packets until ReturnPacket */
  ierr = PetscViewerMathematicaSkipPackets(viewer, RETURNPKT);                                                 CHKERRQ(ierr);
  /* Skip ReturnPacket */
  MLNewPacket(link);
  /* Check that mesh is valid */
  MLPutFunction(link, "EvaluatePacket", 1);
    MLPutFunction(link, "ValidQ", 1);
      MLPutSymbol(link, "mesh");
  MLEndPacket(link);
  ierr = PetscViewerMathematicaSkipPackets(viewer, RETURNPKT);                                                 CHKERRQ(ierr);
  MLGetSymbol(link, &symbol);
  ierr = PetscStrcmp("True", (char *) symbol, &match);                                                    CHKERRQ(ierr);
  if (match == PETSC_FALSE) {
    MLDisownSymbol(link, symbol);
    SETERRQ(PETSC_ERR_PLIB, "Invalid mesh in Mathematica");
  }
  MLDisownSymbol(link, symbol);

  /* mat = gradient matrix */
  ierr = MatView(ml->locB, viewer);                                                                       CHKERRQ(ierr);

  /* mattML = ReduceMatrix[mat,ml->minNodes] */
  numOptions = 0;
  partSize   = 0;
  ierr = PetscOptionsGetInt(PETSC_NULL, "-pc_ml_partition_size", &partSize, &opt);                        CHKERRQ(ierr);
  if ((opt == PETSC_TRUE) && (partSize > 0))
    numOptions++;
  MLPutFunction(link, "EvaluatePacket", 1);
    MLPutFunction(link, "Set", 2);
      MLPutSymbol(link, "mattML");
      MLPutFunction(link, "ReduceMatrix", 3 + numOptions);
        MLPutSymbol(link, "mesh");
        MLPutSymbol(link, "mat");
        MLPutInteger(link, thresh);
        if (partSize > 0) {
          MLPutFunction(link, "Rule", 2);
            MLPutSymbol(link, "PartitionSize");
            MLPutInteger(link, partSize);
        }
  MLEndPacket(link);
  /* Skip packets until ReturnPacket */
  ierr = PetscViewerMathematicaSkipPackets(viewer, RETURNPKT);                                                 CHKERRQ(ierr);
  /* Skip ReturnPacket */
  MLNewPacket(link);
  /* Check that mattML is valid */
  MLPutFunction(link, "EvaluatePacket", 1);
    MLPutFunction(link, "ValidQ", 1);
      MLPutSymbol(link, "mattML");
  MLEndPacket(link);
  ierr = PetscViewerMathematicaSkipPackets(viewer, RETURNPKT);                                                 CHKERRQ(ierr);
  MLGetSymbol(link, &symbol);
  ierr = PetscStrcmp("True", (char *) symbol, &match);                                                    CHKERRQ(ierr);
  if (match == PETSC_FALSE) {
    MLDisownSymbol(link, symbol);
    SETERRQ(PETSC_ERR_PLIB, "Invalid MLData in Mathematica");
  }
  MLDisownSymbol(link, symbol);

  /* Copy information to the preconditioner */
  MLPutFunction(link, "EvaluatePacket", 1);
    MLPutFunction(link, "Part", 2);
      MLPutSymbol(link, "mattML");
      MLPutInteger(link, 3);
  MLEndPacket(link);
  ierr = PetscViewerMathematicaSkipPackets(viewer, RETURNPKT);                                                 CHKERRQ(ierr);
  MLGetInteger(link, &ml->numLevels);

  /* Create lists of the range and nullspace columns */
  MLPutFunction(link, "EvaluatePacket", 1);
    MLPutFunction(link, "GetRange", 1);
      MLPutSymbol(link, "mattML");
  MLEndPacket(link);
  ierr = PetscViewerMathematicaSkipPackets(viewer, RETURNPKT);                                                 CHKERRQ(ierr);
  MLGetIntegerList(link, &range, &numRange);
  if (numRange > ml->sOrder->numLocVars) SETERRQ(PETSC_ERR_PLIB, "Invalid size for range space");
  ml->rank       = numRange;
  ml->globalRank = ml->rank;
  if (ml->rank > 0) {
    ierr = PetscMalloc(numRange * sizeof(int), &ml->range);                                               CHKERRQ(ierr);
    for(row = 0; row < numRange; row++)
      ml->range[row] = range[row]-1;
  }
  MLDisownIntegerList(link, range, numRange);

  MLPutFunction(link, "EvaluatePacket", 1);
    MLPutFunction(link, "GetNullColumns", 1);
      MLPutSymbol(link, "mattML");
  MLEndPacket(link);
  ierr = PetscViewerMathematicaSkipPackets(viewer, RETURNPKT);                                                 CHKERRQ(ierr);
  MLGetIntegerList(link, &null, &numNull);
  if (numRange + numNull != ml->sOrder->numLocVars) SETERRQ(PETSC_ERR_PLIB, "Invalid size for range and null spaces");
  ml->numLocNullCols = numNull;
  if (numNull > 0)
  {
    ierr = PetscMalloc(numNull * sizeof(int), &ml->nullCols);                                             CHKERRQ(ierr);
    for(col = 0; col < numNull; col++)
      ml->nullCols[col] = null[col] - 1;
  }
  MLDisownIntegerList(link, null, numNull);

  PetscFunctionReturn(0);
#endif
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerMathematicaMultiLevelConvert"
int PetscViewerMathematicaMultiLevelConvert(PetscViewer viewer, PC pc)
{
#ifdef PETSC_HAVE_MATHEMATICA
  PetscViewer_Mathematica *vmath = (PetscViewer_Mathematica *) viewer->data;
  MLINK               link;   /* The link to Mathematica */
  PC_Multilevel      *ml;
  Mat_SeqAIJ         *grad;
  int                *numPartitions;
  int                *numPartitionCols, *cols;
  int                *numPartitionRows, *rows;
  double             *U, *singVals, *V;
  long               *Udims, *Vdims;
  char              **Uheads, **Vheads;
  int                *nnz;
  int                *offsets;
  double             *vals;
  long                numLevels, numParts, numCols, numRows, Udepth, numSingVals, Vdepth, len;
  int                 numBdRows, numBdCols;
  int                 mesh, level, part, col, row;
  int                 ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_COOKIE);
  PetscValidHeaderSpecific(pc,     PC_COOKIE);
  link = vmath->link;
  ml   = (PC_Multilevel *) pc->data;

  /* ml->numLevels = ml[[3]] */
  MLPutFunction(link, "EvaluatePacket", 1);
    MLPutFunction(link, "Part", 2);
      MLPutSymbol(link, "mattML");
      MLPutInteger(link, 3);
  MLEndPacket(link);
  ierr = PetscViewerMathematicaSkipPackets(viewer, RETURNPKT);                                                 CHKERRQ(ierr);
  MLGetInteger(link, &ml->numLevels);

  /* ml->numMeshes = Length[ml[[4]]] */
  MLPutFunction(link, "EvaluatePacket", 1);
    MLPutFunction(link, "Length", 1);
      MLPutFunction(link, "Part", 2);
        MLPutSymbol(link, "mattML");
        MLPutInteger(link, 4);
  MLEndPacket(link);
  ierr = PetscViewerMathematicaSkipPackets(viewer, RETURNPKT);                                                 CHKERRQ(ierr);
  MLGetInteger(link, &ml->numMeshes);

  /* ml->numElements[level] = ml[[4,level,1]] */
  ierr = PetscMalloc(ml->numMeshes * sizeof(int), &ml->numElements);                                      CHKERRQ(ierr);

  /* ml->numVertices[level] = ml[[4,level,2]] */
  ierr = PetscMalloc(ml->numMeshes * sizeof(int), &ml->numVertices);                                      CHKERRQ(ierr);

  /* ml->meshes = ml[[4]] */
  ierr = PetscMalloc(ml->numMeshes * sizeof(int **), &ml->meshes);                                        CHKERRQ(ierr);
  for(mesh = 0; mesh < ml->numMeshes; mesh++) {
    ierr = PetscMalloc(NUM_MESH_DIV * sizeof(int *), &ml->meshes[mesh]);                                  CHKERRQ(ierr);
    /* Here we should get meshes */
    ierr = PetscMalloc(1            * sizeof(int),   &ml->meshes[mesh][MESH_OFFSETS]);                    CHKERRQ(ierr);
    ierr = PetscMalloc(1            * sizeof(int),   &ml->meshes[mesh][MESH_ADJ]);                        CHKERRQ(ierr);
    ierr = PetscMalloc(1            * sizeof(int),   &ml->meshes[mesh][MESH_ELEM]);                       CHKERRQ(ierr);
  }

  /* ml->numPartitions = Map[Length,Drop[ml[[5]],-1]] */
  MLPutFunction(link, "EvaluatePacket", 1);
    MLPutFunction(link, "Map", 2);
      MLPutSymbol(link, "Length");
      MLPutFunction(link, "Drop", 2);
        MLPutFunction(link, "Part", 2);
          MLPutSymbol(link, "mattML");
          MLPutInteger(link, 5);
        MLPutInteger(link, -1);
  MLEndPacket(link);
  ierr = PetscViewerMathematicaSkipPackets(viewer, RETURNPKT);                                                 CHKERRQ(ierr);
  MLGetIntegerList(link, &numPartitions, &numLevels);
  if (numLevels != ml->numLevels) SETERRQ(PETSC_ERR_PLIB, "Invalid node partition in MLData object");
  if (numLevels > 0) {
    ierr = PetscMalloc(numLevels * sizeof(int), &ml->numPartitions);                                      CHKERRQ(ierr);
    PetscMemcpy(ml->numPartitions, numPartitions, numLevels * sizeof(int));
  }
  MLDisownIntegerList(link, numPartitions, numLevels);

  if (ml->numLevels > 0) {
    /* ml->numPartitionCols = Map[Length,ml[[5,level]]] */
    ierr = PetscMalloc(ml->numLevels * sizeof(int *), &ml->numPartitionCols);                             CHKERRQ(ierr);
    for(level = 0; level < ml->numLevels; level++) {
      MLPutFunction(link, "EvaluatePacket", 1);
        MLPutFunction(link, "Map", 2);
          MLPutSymbol(link, "Length");
          MLPutFunction(link, "Part", 3);
            MLPutSymbol(link, "mattML");
            MLPutInteger(link, 5);
            MLPutInteger(link, level+1);
      MLEndPacket(link);
      ierr = PetscViewerMathematicaSkipPackets(viewer, RETURNPKT);                                             CHKERRQ(ierr);
      MLGetIntegerList(link, &numPartitionCols, &numParts);
      if (numParts != ml->numPartitions[level]) SETERRQ(PETSC_ERR_PLIB, "Invalid node partition in MLData object");
      if (numParts > 0) {
        ierr = PetscMalloc(numParts * sizeof(int), &ml->numPartitionCols[level]);                         CHKERRQ(ierr);
        PetscMemcpy(ml->numPartitionCols[level], numPartitionCols, numParts * sizeof(int));
      }
      MLDisownIntegerList(link, numPartitionCols, numParts);
    }

    /* ml->colPartition[level][part] = ml[[5,level,part]] */
    ierr = PetscMalloc(ml->numLevels * sizeof(int **), &ml->colPartition);                                CHKERRQ(ierr);
    for(level = 0; level < ml->numLevels; level++) {
      if (ml->numPartitions[level] == 0) continue;
      ierr = PetscMalloc(ml->numPartitions[level] * sizeof(int *), &ml->colPartition[level]);             CHKERRQ(ierr);
      for(part = 0; part < ml->numPartitions[level]; part++) {
        MLPutFunction(link, "EvaluatePacket", 1);
          MLPutFunction(link, "Part", 4);
            MLPutSymbol(link, "mattML");
            MLPutInteger(link, 5);
            MLPutInteger(link, level+1);
            MLPutInteger(link, part+1);
        MLEndPacket(link);
        ierr = PetscViewerMathematicaSkipPackets(viewer, RETURNPKT);                                           CHKERRQ(ierr);
        MLGetIntegerList(link, &cols, &numCols);
        if (numCols != ml->numPartitionCols[level][part]) SETERRQ(PETSC_ERR_PLIB, "Invalid node partition in MLData object");
        if (numCols > 0) {
          ierr = PetscMalloc(numCols * sizeof(int), &ml->colPartition[level][part]);                      CHKERRQ(ierr);
          /* Convert to zero-based indices */
          for(col = 0; col < numCols; col++) ml->colPartition[level][part][col] = cols[col] - 1;
        }
        MLDisownIntegerList(link, cols, numCols);
      }
    }

    /* ml->numPartitionRows = Map[Length,FlattenAt[ml[[6,level]],1]] */
    ierr = PetscMalloc(ml->numLevels * sizeof(int *), &ml->numPartitionRows);                             CHKERRQ(ierr);
    for(level = 0; level < ml->numLevels; level++) {
      MLPutFunction(link, "EvaluatePacket", 1);
        MLPutFunction(link, "Map", 2);
          MLPutSymbol(link, "Length");
          MLPutFunction(link, "FlattenAt", 2);
            MLPutFunction(link, "Part", 3);
              MLPutSymbol(link, "mattML");
              MLPutInteger(link, 6);
              MLPutInteger(link, level+1);
            MLPutInteger(link, 1);
      MLEndPacket(link);
      ierr = PetscViewerMathematicaSkipPackets(viewer, RETURNPKT);                                            CHKERRQ(ierr);
      MLGetIntegerList(link, &numPartitionRows, &numParts);
      if (numParts != ml->numPartitions[level] + NUM_PART_ROW_DIV - 1) {
        SETERRQ(PETSC_ERR_PLIB, "Invalid edge partition in MLData object");
      }
      ierr = PetscMalloc(numParts * sizeof(int), &ml->numPartitionRows[level]);                           CHKERRQ(ierr);
      PetscMemcpy(ml->numPartitionRows[level], numPartitionRows, numParts * sizeof(int));
      MLDisownIntegerList(link, numPartitionRows, numParts);
    }

    /* ml->rowPartition[level][PART_ROW_INT][part] = ml[[6,level,1,part]]
       ml->rowPartition[level][PART_ROW_BD]        = ml[[6,level,2]]
       ml->rowPartition[level][PART_ROW_RES]       = ml[[6,level,3]] */
    ierr = PetscMalloc(ml->numLevels * sizeof(int ***), &ml->rowPartition);                               CHKERRQ(ierr);
    for(level = 0; level < ml->numLevels; level++) {
      ierr = PetscMalloc(NUM_PART_ROW_DIV * sizeof(int **), &ml->rowPartition[level]);                    CHKERRQ(ierr);
      /* Interior rows */
      if (ml->numPartitions[level] > 0) {
        ierr = PetscMalloc(ml->numPartitions[level] * sizeof(int *), &ml->rowPartition[level][PART_ROW_INT]); CHKERRQ(ierr);
        for(part = 0; part < ml->numPartitions[level]; part++) {
          MLPutFunction(link, "EvaluatePacket", 1);
            MLPutFunction(link, "Part", 5);
              MLPutSymbol(link, "mattML");
              MLPutInteger(link, 6);
              MLPutInteger(link, level+1);
              MLPutInteger(link, 1);
              MLPutInteger(link, part+1);
          MLEndPacket(link);
          ierr = PetscViewerMathematicaSkipPackets(viewer, RETURNPKT);                                         CHKERRQ(ierr);
          MLGetIntegerList(link, &rows, &numRows);
          if (numRows != ml->numPartitionRows[level][part]) {
            SETERRQ(PETSC_ERR_PLIB, "Invalid edge partition in MLData object");
          }
          if (numRows > 0) {
            ierr = PetscMalloc(numRows * sizeof(int), &ml->rowPartition[level][PART_ROW_INT][part]);      CHKERRQ(ierr);
            /* Convert to zero-based indices */
            for(row = 0; row < numRows; row++) {
              ml->rowPartition[level][PART_ROW_INT][part][row] = rows[row] - 1;
            }
          }
          MLDisownIntegerList(link, rows, numRows);
        }
      }
      /* Boundary rows */
      ierr = PetscMalloc(1 * sizeof(int *), &ml->rowPartition[level][PART_ROW_BD]);                       CHKERRQ(ierr);
      MLPutFunction(link, "EvaluatePacket", 1);
        MLPutFunction(link, "Part", 4);
          MLPutSymbol(link, "mattML");
          MLPutInteger(link, 6);
          MLPutInteger(link, level+1);
          MLPutInteger(link, 2);
      MLEndPacket(link);
      ierr = PetscViewerMathematicaSkipPackets(viewer, RETURNPKT);                                            CHKERRQ(ierr);
      MLGetIntegerList(link, &rows, &numRows);
      if (numRows != ml->numPartitionRows[level][ml->numPartitions[level]]) {
        SETERRQ(PETSC_ERR_PLIB, "Invalid edge partition in MLData object");
      }
      if (numRows > 0) {
        ierr = PetscMalloc(numRows * sizeof(int), &ml->rowPartition[level][PART_ROW_BD][0]);              CHKERRQ(ierr);
        /* Convert to zero-based indices */
        for(row = 0; row < numRows; row++) {
          ml->rowPartition[level][PART_ROW_BD][0][row] = rows[row] - 1;
        }
      }
      MLDisownIntegerList(link, rows, numRows);
      /* Residual rows*/
      ierr = PetscMalloc(1 * sizeof(int *), &ml->rowPartition[level][PART_ROW_RES]);                      CHKERRQ(ierr);
      MLPutFunction(link, "EvaluatePacket", 1);
        MLPutFunction(link, "Part", 4);
          MLPutSymbol(link, "mattML");
          MLPutInteger(link, 6);
          MLPutInteger(link, level+1);
          MLPutInteger(link, 3);
      MLEndPacket(link);
      ierr = PetscViewerMathematicaSkipPackets(viewer, RETURNPKT);                                            CHKERRQ(ierr);
      MLGetIntegerList(link, &rows, &numRows);
      if (numRows != ml->numPartitionRows[level][ml->numPartitions[level]+1]) {
        SETERRQ(PETSC_ERR_PLIB, "Invalid edge partition in MLData object");
      }
      if (numRows > 0) {
        ierr = PetscMalloc(numRows * sizeof(int), &ml->rowPartition[level][PART_ROW_RES][0]);             CHKERRQ(ierr);
        /* Convert to zero-based indices */
        for(row = 0; row < numRows; row++) {
          ml->rowPartition[level][PART_ROW_RES][0][row] = rows[row] - 1;
        }
      }
      MLDisownIntegerList(link, rows, numRows);
    }
  } else {
    ierr = PetscMalloc(1 * sizeof(int),     &ml->numPartitions);                                          CHKERRQ(ierr);
    ierr = PetscMalloc(1 * sizeof(int *),   &ml->numPartitionCols);                                       CHKERRQ(ierr);
    ierr = PetscMalloc(1 * sizeof(int **),  &ml->colPartition);                                           CHKERRQ(ierr);
    ierr = PetscMalloc(1 * sizeof(int *),   &ml->numPartitionRows);                                       CHKERRQ(ierr);
    ierr = PetscMalloc(1 * sizeof(int ***), &ml->rowPartition);                                           CHKERRQ(ierr);
  }

  /* ml->numRows = ml[[1]] */
  MLPutFunction(link, "EvaluatePacket", 1);
    MLPutFunction(link, "Part", 2);
      MLPutSymbol(link, "mattML");
      MLPutInteger(link, 1);
  MLEndPacket(link);
  ierr = PetscViewerMathematicaSkipPackets(viewer, RETURNPKT);                                                 CHKERRQ(ierr);
  MLGetInteger(link, &ml->numRows);

  /* ml->numCols = ml[[2]] */
  MLPutFunction(link, "EvaluatePacket", 1);
    MLPutFunction(link, "Part", 2);
      MLPutSymbol(link, "mattML");
      MLPutInteger(link, 2);
  MLEndPacket(link);
  ierr = PetscViewerMathematicaSkipPackets(viewer, RETURNPKT);                                                 CHKERRQ(ierr);
  MLGetInteger(link, &ml->numCols);

  /* ml->zeroTol = ml[[9]] */
  MLPutFunction(link, "EvaluatePacket", 1);
    MLPutFunction(link, "N", 1);
      MLPutFunction(link, "Part", 2);
        MLPutSymbol(link, "mattML");
        MLPutInteger(link, 9);
  MLEndPacket(link);
  ierr = PetscViewerMathematicaSkipPackets(viewer, RETURNPKT);                                                 CHKERRQ(ierr);
  MLGetReal(link, &ml->zeroTol);

  if (ml->numLevels > 0) {
    /* ml->factors[level][part][FACT_U]    = ml[[7,level,part,1]]
       ml->factors[level][part][FACT_DINV] = Divide[1,Select[ml[[7,level,part,2]],(#>ml[[9]])&]]
       ml->factors[level][part][FACT_V]    = ml[[7,level,part,3]] */
    ierr = PetscMalloc(ml->numLevels * sizeof(double ***), &ml->factors);                                 CHKERRQ(ierr);
    for(level = 0; level < ml->numLevels; level++) {
      if (ml->numPartitions[level] == 0) continue;
      ierr = PetscMalloc(ml->numPartitions[level] * sizeof(double **), &ml->factors[level]);              CHKERRQ(ierr);
      for(part = 0; part < ml->numPartitions[level]; part++) {
        ierr = PetscMalloc(NUM_FACT_DIV * sizeof(double *), &ml->factors[level][part]);                   CHKERRQ(ierr);
        /* U, or U^T in LAPACK terms */
        MLPutFunction(link, "EvaluatePacket", 1);
          MLPutFunction(link, "Part", 5);
            MLPutSymbol(link, "mattML");
            MLPutInteger(link, 7);
            MLPutInteger(link, level+1);
            MLPutInteger(link, part+1);
            MLPutInteger(link, 1);
        MLEndPacket(link);
        ierr = PetscViewerMathematicaSkipPackets(viewer, RETURNPKT);                                          CHKERRQ(ierr);
        MLGetRealArray(link, &U, &Udims, &Uheads, &Udepth);
        if (Udepth > 1) {
          if (Udepth != 2) SETERRQ(PETSC_ERR_PLIB, "Invalid U matrix");
          if ((Udims[0] != ml->numPartitionRows[level][part]) || (Udims[0] != Udims[1])) {
            SETERRQ(PETSC_ERR_PLIB, "Incompatible dimensions for U matrix");
          }
          ierr = PetscMalloc(Udims[0]*Udims[0] * sizeof(double), &ml->factors[level][part][FACT_U]);      CHKERRQ(ierr);
          /* Notice that LAPACK will think that this is U^T, or U in LAPACK terms */
          PetscMemcpy(ml->factors[level][part][FACT_U], U, Udims[0]*Udims[0] * sizeof(double));
        } else if (ml->numPartitionRows[level][part] != 0) {
          SETERRQ(PETSC_ERR_PLIB, "Missing U matrix");
        }
        MLDisownRealArray(link, U, Udims, Uheads, Udepth);
        /* D^{-1} */
        MLPutFunction(link, "EvaluatePacket", 1);
          MLPutFunction(link, "Divide", 2);
            MLPutReal(link, 1.0);
            MLPutFunction(link, "Select", 2);
              MLPutFunction(link, "Part", 5);
                MLPutSymbol(link, "mattML");
                MLPutInteger(link, 7);
                MLPutInteger(link, level+1);
                MLPutInteger(link, part+1);
                MLPutInteger(link, 2);
              MLPutFunction(link, "Function", 2);
                MLPutSymbol(link, "x");
                MLPutFunction(link, "Greater", 2);
                  MLPutSymbol(link, "x");
                  MLPutFunction(link, "Part", 2);
                    MLPutSymbol(link, "mattML");
                    MLPutInteger(link, 9);
        MLEndPacket(link);
        ierr = PetscViewerMathematicaSkipPackets(viewer, RETURNPKT);                                          CHKERRQ(ierr);
        MLGetRealList(link, &singVals, &numSingVals);
        if (numSingVals > ml->numPartitionCols[level][part]) {
          SETERRQ(PETSC_ERR_PLIB, "Invalid factor list in MLData object");
        }
        if (ml->numPartitionCols[level][part] > 0) {
          ierr = PetscMalloc(ml->numPartitionCols[level][part] * sizeof(double), &ml->factors[level][part][FACT_DINV]); CHKERRQ(ierr);
          PetscMemzero(ml->factors[level][part][FACT_DINV], ml->numPartitionCols[level][part] * sizeof(double));
          PetscMemcpy(ml->factors[level][part][FACT_DINV], singVals, numSingVals * sizeof(double));
        }
        MLDisownRealList(link, singVals, numSingVals);
        /* V^T, but V in LAPACK terms */
        MLPutFunction(link, "EvaluatePacket", 1);
          MLPutFunction(link, "Transpose", 1);
            MLPutFunction(link, "Part", 5);
              MLPutSymbol(link, "mattML");
              MLPutInteger(link, 7);
              MLPutInteger(link, level+1);
              MLPutInteger(link, part+1);
              MLPutInteger(link, 3);
        MLEndPacket(link);
        ierr = PetscViewerMathematicaSkipPackets(viewer, RETURNPKT);                                          CHKERRQ(ierr);
        MLGetRealArray(link, &V, &Vdims, &Vheads, &Vdepth);
        if (Vdepth > 1) {
          if (Vdepth != 2) SETERRQ(PETSC_ERR_PLIB, "Invalid V matrix");
          if ((Vdims[0] != ml->numPartitionCols[level][part]) || (Vdims[0] != Vdims[1])) {
            SETERRQ(PETSC_ERR_PLIB, "Incompatible dimensions for U matrix");
          }
          ierr = PetscMalloc(Vdims[0]*Vdims[0] * sizeof(double), &ml->factors[level][part][FACT_V]);      CHKERRQ(ierr);
          /* Notice that LAPACK will think that this is V, or V^T in LAPACK terms */
          PetscMemcpy(ml->factors[level][part][FACT_V], V, Vdims[0]*Vdims[0] * sizeof(double));
        } else if (ml->numPartitionCols[level][part] != 0) {
          SETERRQ(PETSC_ERR_PLIB, "Missing V matrix");
        }
        MLDisownRealArray(link, V, Vdims, Vheads, Vdepth);
      }
    }

    /* ml->grads = ml[[8]] */
    ierr = PetscMalloc(ml->numLevels * sizeof(Mat), &ml->grads);                                          CHKERRQ(ierr);
    ierr = PetscMalloc(ml->numLevels * sizeof(Vec), &ml->bdReduceVecs);                                   CHKERRQ(ierr);
    ierr = PetscMalloc(ml->numLevels * sizeof(Vec), &ml->colReduceVecs);                                  CHKERRQ(ierr);
    ierr = PetscMalloc(ml->numLevels * sizeof(Vec), &ml->colReduceVecs2);                                 CHKERRQ(ierr);
    for(level = 0; level < ml->numLevels; level++) {
      if (ml->numPartitions[level] == 0) continue;
      /* numRows = ml[[8,level,1]] */
      MLPutFunction(link, "EvaluatePacket", 1);
        MLPutFunction(link, "Part", 4);
          MLPutSymbol(link, "mattML");
          MLPutInteger(link, 8);
          MLPutInteger(link, level+1);
          MLPutInteger(link, 1);
      MLEndPacket(link);
      ierr = PetscViewerMathematicaSkipPackets(viewer, RETURNPKT);                                            CHKERRQ(ierr);
      MLGetInteger(link, &numBdRows);
      ierr = VecCreateSeq(PETSC_COMM_SELF, numBdRows, &ml->bdReduceVecs[level]);                         CHKERRQ(ierr);
      /* numCols = ml[[8,level,2]] */
      MLPutFunction(link, "EvaluatePacket", 1);
        MLPutFunction(link, "Part", 4);
          MLPutSymbol(link, "mattML");
          MLPutInteger(link, 8);
          MLPutInteger(link, level+1);
          MLPutInteger(link, 2);
      MLEndPacket(link);
      ierr = PetscViewerMathematicaSkipPackets(viewer, RETURNPKT);                                            CHKERRQ(ierr);
      MLGetInteger(link, &numBdCols);
      ierr = VecCreateSeq(PETSC_COMM_SELF, numBdCols, &ml->colReduceVecs[level]);                        CHKERRQ(ierr);
      ierr = VecDuplicate(ml->colReduceVecs[level], &ml->colReduceVecs2[level]);                         CHKERRQ(ierr);
      /* nnz = Map[Apply[Subtract,Sort[#,Greater]]&, Partition[ml[[8,level,3]],2,1]] */
      MLPutFunction(link, "EvaluatePacket", 1);
        MLPutFunction(link, "Map", 2);
          MLPutFunction(link, "Function", 2);
            MLPutSymbol(link, "x");
            MLPutFunction(link, "Apply", 2);
              MLPutSymbol(link, "Subtract");
              MLPutFunction(link, "Sort", 2);
                MLPutSymbol(link, "x");
                MLPutSymbol(link, "Greater");
          MLPutFunction(link, "Partition", 3);
            MLPutFunction(link, "Part", 4);
              MLPutSymbol(link, "mattML");
              MLPutInteger(link, 8);
              MLPutInteger(link, level+1);
              MLPutInteger(link, 3);
            MLPutInteger(link, 2);
            MLPutInteger(link, 1);
      MLEndPacket(link);
      ierr = PetscViewerMathematicaSkipPackets(viewer, RETURNPKT);                                            CHKERRQ(ierr);
      MLGetIntegerList(link, &nnz, &len);
      if (len != numBdRows) SETERRQ(PETSC_ERR_PLIB, "Invalid boundary gradient matrix");
      ierr = MatCreate(PETSC_COMM_SELF,numBdRows,numBdCols,numBdRows,numBdCols,&ml->grads[level]);CHKERRQ(ierr);
      ierr = MatSetType(ml->grads[level],MATSEQAIJ);CHKERRQ(ierr);
      ierr = MatSeqAIJSetPreallocation(ml->grads[level],PETSC_DEFAULT,nnz);CHKERRQ(ierr);
      grad = (Mat_SeqAIJ *) ml->grads[level]->data;
      PetscMemcpy(grad->ilen, nnz, numBdRows * sizeof(int));
      MLDisownIntegerList(link, nnz, len);
      /* ml->grads[level]->i = ml[[8,level,3]] */
      MLPutFunction(link, "EvaluatePacket", 1);
        MLPutFunction(link, "Part", 4);
          MLPutSymbol(link, "mattML");
          MLPutInteger(link, 8);
          MLPutInteger(link, level+1);
          MLPutInteger(link, 3);
      MLEndPacket(link);
      ierr = PetscViewerMathematicaSkipPackets(viewer, RETURNPKT);                                            CHKERRQ(ierr);
      MLGetIntegerList(link, &offsets, &len);
      if (len != numBdRows+1) SETERRQ(PETSC_ERR_PLIB, "Invalid boundary gradient matrix");
      for(row = 0; row <= numBdRows; row++)
        grad->i[row] = offsets[row]-1;
      MLDisownIntegerList(link, offsets, len);
      /* ml->grads[level]->j = ml[[8,level,4]] */
      MLPutFunction(link, "EvaluatePacket", 1);
        MLPutFunction(link, "Part", 4);
          MLPutSymbol(link, "mattML");
          MLPutInteger(link, 8);
          MLPutInteger(link, level+1);
          MLPutInteger(link, 4);
      MLEndPacket(link);
      ierr = PetscViewerMathematicaSkipPackets(viewer, RETURNPKT);                                            CHKERRQ(ierr);
      MLGetIntegerList(link, &cols, &len);
      if (len != grad->i[numBdRows]) SETERRQ(PETSC_ERR_PLIB, "Invalid boundary gradient matrix");
      for(col = 0; col < len; col++)
        grad->j[col] = cols[col]-1;
      MLDisownIntegerList(link, cols, len);
      /* ml->grads[level]->i = ml[[8,level,5]] */
      MLPutFunction(link, "EvaluatePacket", 1);
        MLPutFunction(link, "Part", 4);
          MLPutSymbol(link, "mattML");
          MLPutInteger(link, 8);
          MLPutInteger(link, level+1);
          MLPutInteger(link, 5);
      MLEndPacket(link);
      ierr = PetscViewerMathematicaSkipPackets(viewer, RETURNPKT);                                            CHKERRQ(ierr);
      MLGetRealList(link, &vals, &len);
      if (len != grad->i[numBdRows]) SETERRQ(PETSC_ERR_PLIB, "Invalid boundary gradient matrix");
      PetscMemcpy(grad->a, vals, len * sizeof(double));
      MLDisownRealList(link, vals, len);
      /* Fix up all the members */
      grad->nz += len;
      ierr = MatAssemblyBegin(ml->grads[level], MAT_FINAL_ASSEMBLY);                                      CHKERRQ(ierr);
      ierr = MatAssemblyEnd(ml->grads[level], MAT_FINAL_ASSEMBLY);                                        CHKERRQ(ierr);
    }
  } else {
    ierr = PetscMalloc(1 * sizeof(double ***), &ml->factors);                                             CHKERRQ(ierr);
    ierr = PetscMalloc(1 * sizeof(Mat), &ml->grads);                                                      CHKERRQ(ierr);
    ierr = PetscMalloc(1 * sizeof(Vec), &ml->bdReduceVecs);                                               CHKERRQ(ierr);
    ierr = PetscMalloc(1 * sizeof(Vec), &ml->colReduceVecs);                                              CHKERRQ(ierr);
    ierr = PetscMalloc(1 * sizeof(Vec), &ml->colReduceVecs2);                                             CHKERRQ(ierr);
  }

  ml->interiorWorkLen = 1;
  for(level = 0; level < ml->numLevels; level++) {
    for(part = 0; part < ml->numPartitions[level]; part++)
      ml->interiorWorkLen = PetscMax(ml->interiorWorkLen, ml->numPartitionRows[level][part]);
  }
  ierr = PetscMalloc(ml->interiorWorkLen * sizeof(double), &ml->interiorWork);                            CHKERRQ(ierr);
  ierr = PetscMalloc(ml->interiorWorkLen * sizeof(double), &ml->interiorWork2);                           CHKERRQ(ierr);

  PetscFunctionReturn(0);
#endif
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#if 0
/*------------------------------ Functions for Triangular 2d Meshes -------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "PetscViewerMathematicaCreateSamplePoints_Triangular_2D"
int PetscViewerMathematicaCreateSamplePoints_Triangular_2D(PetscViewer viewer, GVec v)
{
#ifdef PETSC_HAVE_MATHEMATICA
  PetscViewer_Mathematica *vmath = (PetscViewer_Mathematica *) viewer->data;
  MLINK               link  = vmath->link; /* The link to Mathematica */
  Grid                grid;
  int                 numNodes;
  int                *classes;
  int                *offsets;
  double             *array;
  int                *localStart;
  int                 localOffset, comp;
  int                 node, nclass;
  int                 ierr;

  PetscFunctionBegin;
  ierr       = GVecGetGrid(v, &grid);                                                                    CHKERRQ(ierr);
  numNodes   = grid->mesh->numNodes;
  comp       = grid->viewComp;
  offsets    = grid->order->offsets;
  localStart = grid->order->localStart[grid->viewField];
  classes    = grid->cm->classes;

  /* Attach a value to each point in the mesh -- Extra values are put in for fields not
     defined on some nodes, but these values are never used */
  ierr = VecGetArray(v, &array);                                                                         CHKERRQ(ierr);
  MLPutFunction(link, "ReplaceAll", 2);
    MLPutFunction(link, "Thread", 1);
      MLPutFunction(link, "f", 2);
        MLPutSymbol(link, "nodes");
        MLPutFunction(link, "List", numNodes);
        for(node = 0; node < numNodes; node++)
        {
          nclass      = classes[node];
          localOffset = localStart[nclass] + comp;
          MLPutReal(link, array[offsets[node] + localOffset]);
        }
    MLPutFunction(link, "Rule", 2);
      MLPutSymbol(link, "f");
      MLPutSymbol(link, "Append");
  ierr = VecRestoreArray(v, &array);                                                                     CHKERRQ(ierr);

  PetscFunctionReturn(0);
#endif
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerMathematicaCreateVectorSamplePoints_Triangular_2D"
int PetscViewerMathematicaCreateVectorSamplePoints_Triangular_2D(PetscViewer viewer, GVec v)
{
#ifdef PETSC_HAVE_MATHEMATICA
  PetscViewer_Mathematica *vmath = (PetscViewer_Mathematica *) viewer->data;
  MLINK               link  = vmath->link; /* The link to Mathematica */
  Grid                grid;
  int                 numNodes;
  int                *classes;
  int                *offsets;
  int                *fieldClasses;
  double             *array;
  int                *localStart;
  int                 localOffset;
  int                 node, nclass;
  int                 ierr;

  PetscFunctionBegin;
  ierr         = GVecGetGrid(v, &grid);                                                                  CHKERRQ(ierr);
  numNodes     = grid->mesh->numNodes;
  fieldClasses = grid->cm->fieldClasses[grid->viewField];
  offsets      = grid->order->offsets;
  localStart   = grid->order->localStart[grid->viewField];
  classes      = grid->cm->classes;

  /* Make a list {{{x_0,y_0},{f^0_x,f^0_y}},...} */
  ierr = VecGetArray(v, &array);                                                                         CHKERRQ(ierr);
  MLPutFunction(link, "ReplaceAll", 2);
    MLPutFunction(link, "Thread", 1);
      MLPutFunction(link, "f", 2);
        MLPutSymbol(link, "nodes");
        MLPutFunction(link, "List", numNodes);
        for(node = 0; node < numNodes; node++)
        {
          nclass = classes[node];
          if (fieldClasses[nclass])
          {
            localOffset = localStart[nclass];
            MLPutFunction(link, "List", 2);
              MLPutReal(link, array[offsets[node] + localOffset]);
              MLPutReal(link, array[offsets[node] + localOffset + 1]);
          }
          else
          {
            /* Vectors of length zero are ignored */
            MLPutFunction(link, "List", 2);
              MLPutReal(link, 0.0);
              MLPutReal(link, 0.0);
          }
        }
    MLPutFunction(link, "Rule", 2);
      MLPutSymbol(link, "f");
      MLPutSymbol(link, "List");
  ierr = VecRestoreArray(v, &array);                                                                     CHKERRQ(ierr);

  PetscFunctionReturn(0);
#endif
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerMathematicaCreateInterpolatedSamplePoints_Triangular_2D"
int PetscViewerMathematicaCreateInterpolatedSamplePoints_Triangular_2D(PetscViewer viewer, GVec v, int vComp)
{
#ifdef PETSC_HAVE_MATHEMATICA
#if 0
  PetscViewer_Mathematica  *vmath = (PetscViewer_Mathematica *) viewer->data;
  MLINK                link  = vmath->link; /* The link to Mathematica */
  InterpolationContext iCtx;
  Grid                 grid;
  Mesh                 mesh;
  double              *x, *y, *values;
  long                 m, n;
  double               startX, endX, incX;
  double               startY, endY, incY;
  int                  comp;
  int                  proc, row, col;
  PetscTruth           opt;
  int                  ierr;
#endif

  PetscFunctionBegin;
#if 0
  ierr  = GVecGetGrid(v, &grid);                                                                         CHKERRQ(ierr);
  ierr  = GridGetMesh(grid, &mesh);                                                                      CHKERRQ(ierr);
  comp  = grid->comp[grid->viewField];

  /* This sucks, but I will fix it later (It is for GridReduceInterpolationElementVec_Triangular_2D) */
  grid->classesOld = grid->classes;

  /* Setup InterpolationContext */
  iCtx.vec         = v;
  iCtx.mesh        = mesh;
  iCtx.order       = grid->order;
  iCtx.ghostVec    = grid->ghostVec;
  iCtx.field       = grid->viewField;
  iCtx.numProcs    = mesh->part->numProcs;
  iCtx.rank        = mesh->part->rank;
  ierr = PetscMalloc(iCtx.numProcs   * sizeof(int),      &iCtx.activeProcs);                              CHKERRQ(ierr);
  ierr = PetscMalloc(iCtx.numProcs   * sizeof(int),      &iCtx.calcProcs);                                CHKERRQ(ierr);
  ierr = PetscMalloc(iCtx.numProcs*3 * sizeof(PetscScalar),   &iCtx.coords);                              CHKERRQ(ierr);
  ierr = PetscMalloc(iCtx.numProcs   * sizeof(PetscScalar *), &iCtx.values);                              CHKERRQ(ierr);
  for(proc = 0; proc < iCtx.numProcs; proc++) {
    ierr = PetscMalloc(comp * sizeof(PetscScalar), &iCtx.values[proc]);                                   CHKERRQ(ierr);
  }

  /* Setup domain */
  startX = 0.0;
  startY = 0.0;
  endX   = 1.0;
  endY   = 1.0;
  ierr   = PetscOptionsGetDouble("viewer_", "-math_start_x", &startX, &opt);                              CHKERRQ(ierr);
  ierr   = PetscOptionsGetDouble("viewer_", "-math_start_y", &startY, &opt);                              CHKERRQ(ierr);
  ierr   = PetscOptionsGetDouble("viewer_", "-math_end_x",   &endX,   &opt);                              CHKERRQ(ierr);
  ierr   = PetscOptionsGetDouble("viewer_", "-math_end_y",   &endY,   &opt);                              CHKERRQ(ierr);
  ierr   = PetscOptionsGetInt("viewer_", "-math_div_x", (int *) &n, &opt);                                CHKERRQ(ierr);
  ierr   = PetscOptionsGetInt("viewer_", "-math_div_y", (int *) &m, &opt);                                CHKERRQ(ierr);
  ierr   = PetscMalloc((n+1)      * sizeof(double), &x);                                                  CHKERRQ(ierr);
  ierr   = PetscMalloc((n+1)      * sizeof(double), &y);                                                  CHKERRQ(ierr);
  ierr   = PetscMalloc((n+1)*comp * sizeof(double), &values);                                             CHKERRQ(ierr);
  incX   = (endX - startX)/n;
  incY   = (endY - startY)/m;

  x[0] = startX;
  for(col = 1; col <= n; col++)
    x[col] = x[col-1] + incX;

  MLPutFunction(link, "List", m+1);
    for(row = 0; row <= m; row++)
    {
      ierr = PetscMemzero(values, (n+1)*comp * sizeof(double));                                          CHKERRQ(ierr);
      for(col = 0; col <= n; col++)
        y[col] = startY + incY*row;
      ierr = PointFunctionInterpolateField(n+1, comp, x, y, x, values, &iCtx);                           CHKERRQ(ierr);
      if (vComp >= 0)
      {
        for(col = 0; col <= n; col++)
          values[col] = values[col*comp+vComp];
        MLPutRealList(link, values, n+1);
      }
      else
      {
        MLPutFunction(link, "Transpose", 1);
        MLPutFunction(link, "List", 2);
          MLPutFunction(link, "Transpose", 1);
            MLPutFunction(link, "List", 2);
              MLPutRealList(link, x, n+1);
              MLPutRealList(link, y, n+1);
          MLPutFunction(link, "Partition", 2);
            MLPutRealList(link, values, (n+1)*comp);
            MLPutInteger(link, comp);
      }
    }

  /* This sucks, but I will fix it later (It is for GridReduceInterpolationElementVec_Triangular_2D) */
  grid->classesOld = PETSC_NULL;

  /* Cleanup */
  ierr = PetscFree(x);                                                                                   CHKERRQ(ierr);
  ierr = PetscFree(y);                                                                                   CHKERRQ(ierr);
  ierr = PetscFree(values);                                                                              CHKERRQ(ierr);
  ierr = PetscFree(iCtx.activeProcs);                                                                    CHKERRQ(ierr);
  ierr = PetscFree(iCtx.calcProcs);                                                                      CHKERRQ(ierr);
  ierr = PetscFree(iCtx.coords);                                                                         CHKERRQ(ierr);
  for(proc = 0; proc < iCtx.numProcs; proc++) {
    ierr = PetscFree(iCtx.values[proc]);                                                                 CHKERRQ(ierr);
  }
  ierr = PetscFree(iCtx.values);                                                                         CHKERRQ(ierr);
#endif

  PetscFunctionReturn(0);
#endif
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerMathematica_Mesh_Triangular_2D"
int PetscViewerMathematica_Mesh_Triangular_2D(PetscViewer viewer, Mesh mesh)
{
#ifdef PETSC_HAVE_MATHEMATICA
  PetscViewer_Mathematica  *vmath = (PetscViewer_Mathematica *) viewer->data;
  MLINK                link  = vmath->link;
  Mesh_Triangular     *tri   = (Mesh_Triangular *) mesh->data;
  int                  numCorners = mesh->numCorners;
  int                  numFaces   = mesh->numFaces;
  int                 *faces      = tri->faces;
  int                  numNodes   = mesh->numNodes;
  double              *nodes      = tri->nodes;
  int                  node, face, corner;
  int                  ierr;

  PetscFunctionBegin;
  /* Load package to view graphics in a window */
  ierr = PetscViewerMathematicaLoadGraphics(viewer, vmath->graphicsType);                                     CHKERRQ(ierr);

  /* Send the node coordinates */
  MLPutFunction(link, "EvaluatePacket", 1);
    MLPutFunction(link, "Set", 2);
      MLPutSymbol(link, "nodes");
      MLPutFunction(link, "List", numNodes);
      for(node = 0; node < numNodes; node++) {
        MLPutFunction(link, "List", 2);
          MLPutReal(link, nodes[node*2]);
          MLPutReal(link, nodes[node*2+1]);
      }
  MLEndPacket(link);
  /* Skip packets until ReturnPacket */
  ierr = PetscViewerMathematicaSkipPackets(viewer, RETURNPKT);                                                CHKERRQ(ierr);
  /* Skip ReturnPacket */
  MLNewPacket(link);

  /* Send the faces */
  MLPutFunction(link, "EvaluatePacket", 1);
    MLPutFunction(link, "Set", 2);
      MLPutSymbol(link, "faces");
      MLPutFunction(link, "List", numFaces);
      for(face = 0; face < numFaces; face++) {
        MLPutFunction(link, "List", numCorners);
        for(corner = 0; corner < numCorners; corner++) {
          MLPutReal(link, faces[face*numCorners+corner]);
        }
      }
  MLEndPacket(link);
  /* Skip packets until ReturnPacket */
  ierr = PetscViewerMathematicaSkipPackets(viewer, RETURNPKT);                                                CHKERRQ(ierr);
  /* Skip ReturnPacket */
  MLNewPacket(link);

  PetscFunctionReturn(0);
#endif
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerMathematicaCheckMesh_Triangular_2D"
int PetscViewerMathematicaCheckMesh_Triangular_2D(PetscViewer viewer, Mesh mesh)
{
#ifdef PETSC_HAVE_MATHEMATICA
  PetscViewer_Mathematica  *vmath = (PetscViewer_Mathematica *) viewer->data;
  MLINK                link  = vmath->link;
  int                  numCorners = mesh->numCorners;
  int                  numFaces   = mesh->numFaces;
  int                  numNodes   = mesh->numNodes;
  const char          *symbol;
  long                 args;
  int                  dim, type;
  PetscTruth           match;
  int                  ierr;

  PetscFunctionBegin;
  /* Check that nodes are defined */
  MLPutFunction(link, "EvaluatePacket", 1);
    MLPutFunction(link, "ValueQ", 1);
      MLPutSymbol(link, "nodes");
  MLEndPacket(link);
  ierr = PetscViewerMathematicaSkipPackets(viewer, RETURNPKT);                                                CHKERRQ(ierr);
  MLGetSymbol(link, &symbol);
  ierr = PetscStrcmp("True", (char *) symbol, &match);                                                   CHKERRQ(ierr);
  if (match == PETSC_FALSE) {
    MLDisownSymbol(link, symbol);
    goto redefineMesh;
  }
  MLDisownSymbol(link, symbol);

  /* Check the dimensions */
  MLPutFunction(link, "EvaluatePacket", 1);
    MLPutFunction(link, "MatrixQ", 2);
      MLPutSymbol(link, "nodes");
      MLPutSymbol(link, "NumberQ");
  MLEndPacket(link);
  ierr = PetscViewerMathematicaSkipPackets(viewer, RETURNPKT);                                                CHKERRQ(ierr);
  MLGetSymbol(link, &symbol);
  ierr = PetscStrcmp("True", (char *) symbol, &match);                                                   CHKERRQ(ierr);
  if (match == PETSC_FALSE) {
    MLDisownSymbol(link, symbol);
    goto redefineMesh;
  }
  MLDisownSymbol(link, symbol);
  MLPutFunction(link, "EvaluatePacket", 1);
    MLPutFunction(link, "Dimensions", 1);
      MLPutSymbol(link, "nodes");
  MLEndPacket(link);
  ierr = PetscViewerMathematicaSkipPackets(viewer, RETURNPKT);                                                CHKERRQ(ierr);
  args = 0;
  type = MLGetNext(link);
  MLGetArgCount(link, &args);
  if (args != 2) {
    MLNewPacket(link);
    goto redefineMesh;
  }
  MLGetSymbol(link, &symbol);
  MLDisownSymbol(link, symbol);
  MLGetInteger(link, &dim);
  if (dim != numNodes) {
    MLNewPacket(link);
    goto redefineMesh;
  }
  MLGetInteger(link, &dim);
  if (dim != mesh->dim) {
    MLNewPacket(link);
    goto redefineMesh;
  }

  /* Check that faces are defined */
  MLPutFunction(link, "EvaluatePacket", 1);
    MLPutFunction(link, "ValueQ", 1);
      MLPutSymbol(link, "faces");
  MLEndPacket(link);
  ierr = PetscViewerMathematicaSkipPackets(viewer, RETURNPKT);                                                CHKERRQ(ierr);
  MLGetSymbol(link, &symbol);
  ierr = PetscStrcmp("True", (char *) symbol, &match);                                                   CHKERRQ(ierr);
  if (match == PETSC_FALSE) {
    MLDisownSymbol(link, symbol);
    goto redefineMesh;
  }
  MLDisownSymbol(link, symbol);

  /* Check the dimensions */
  MLPutFunction(link, "EvaluatePacket", 1);
    MLPutFunction(link, "MatrixQ", 2);
      MLPutSymbol(link, "faces");
      MLPutSymbol(link, "NumberQ");
  MLEndPacket(link);
  ierr = PetscViewerMathematicaSkipPackets(viewer, RETURNPKT);                                                CHKERRQ(ierr);
  MLGetSymbol(link, &symbol);
  ierr = PetscStrcmp("True", (char *) symbol, &match);                                                   CHKERRQ(ierr);
  if (match == PETSC_FALSE) {
    MLDisownSymbol(link, symbol);
    goto redefineMesh;
  }
  MLDisownSymbol(link, symbol);
  MLPutFunction(link, "EvaluatePacket", 1);
    MLPutFunction(link, "Dimensions", 1);
      MLPutSymbol(link, "faces");
  MLEndPacket(link);
  ierr = PetscViewerMathematicaSkipPackets(viewer, RETURNPKT);                                                CHKERRQ(ierr);
  args = 0;
  type = MLGetNext(link);
  MLGetArgCount(link, &args);
  if (args != 2) {
    MLNewPacket(link);
    goto redefineMesh;
  }
  MLGetSymbol(link, &symbol);
  MLDisownSymbol(link, symbol);
  MLGetInteger(link, &dim);
  if (dim != numFaces) {
    MLNewPacket(link);
    goto redefineMesh;
  }
  MLGetInteger(link, &dim);
  if (dim != numCorners) {
    MLNewPacket(link);
    goto redefineMesh;
  }

  PetscFunctionReturn(0);

 redefineMesh:
  ierr = MeshView(mesh, viewer);                                                                         CHKERRQ(ierr);
  PetscFunctionReturn(0);
#endif
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerMathematicaTriangulationPlot_Triangular_2D"
int PetscViewerMathematicaTriangulationPlot_Triangular_2D(PetscViewer viewer, GVec v)
{
#ifdef PETSC_HAVE_MATHEMATICA
  PetscViewer_Mathematica  *vmath = (PetscViewer_Mathematica *) viewer->data;
  MLINK                link  = vmath->link; /* The link to Mathematica */
  Mesh                 mesh;
  Grid                 grid;
  int                  numCorners;
  int                  ierr;

  PetscFunctionBegin;
  ierr       = GVecGetGrid(v, &grid);                                                                    CHKERRQ(ierr);
  mesh       = grid->mesh;
  numCorners = mesh->numCorners;

  MLPutFunction(link, "Show", 2);
    MLPutFunction(link, "Graphics3D", 1);
      MLPutFunction(link, "Flatten", 1);
        MLPutFunction(link, "Map", 2);
          MLPutFunction(link, "Function", 1);
          if ((numCorners == 6) && ((grid->cm->numClasses == 1) || (grid->cm->fieldClasses[grid->viewField][1])))
          {
            MLPutFunction(link, "List", 4);
            /* Triangle 0--5--4 */
            MLPutFunction(link, "Polygon", 1);
              MLPutFunction(link, "Part", 2);
                MLPutSymbol(link, "points");
                MLPutFunction(link, "Plus", 2);
                  MLPutFunction(link, "Part", 2);
                    MLPutFunction(link, "Slot", 1);
                      MLPutInteger(link, 1);
                    MLPutFunction(link, "List", 3);
                      MLPutInteger(link, 1);
                      MLPutInteger(link, 6);
                      MLPutInteger(link, 5);
                  MLPutInteger(link, 1);
            /* Triangle 1--3--5 */
            MLPutFunction(link, "Polygon", 1);
              MLPutFunction(link, "Part", 2);
                MLPutSymbol(link, "points");
                MLPutFunction(link, "Plus", 2);
                  MLPutFunction(link, "Part", 2);
                    MLPutFunction(link, "Slot", 1);
                      MLPutInteger(link, 1);
                    MLPutFunction(link, "List", 3);
                      MLPutInteger(link, 2);
                      MLPutInteger(link, 4);
                      MLPutInteger(link, 6);
                  MLPutInteger(link, 1);
            /* Triangle 2--4--3 */
            MLPutFunction(link, "Polygon", 1);
              MLPutFunction(link, "Part", 2);
                MLPutSymbol(link, "points");
                MLPutFunction(link, "Plus", 2);
                  MLPutFunction(link, "Part", 2);
                    MLPutFunction(link, "Slot", 1);
                      MLPutInteger(link, 1);
                    MLPutFunction(link, "List", 3);
                      MLPutInteger(link, 3);
                      MLPutInteger(link, 5);
                      MLPutInteger(link, 4);
                  MLPutInteger(link, 1);
            /* Triangle 3--4--5 */
            MLPutFunction(link, "Polygon", 1);
              MLPutFunction(link, "Part", 2);
                MLPutSymbol(link, "points");
                MLPutFunction(link, "Plus", 2);
                  MLPutFunction(link, "Part", 2);
                    MLPutFunction(link, "Slot", 1);
                      MLPutInteger(link, 1);
                    MLPutFunction(link, "List", 3);
                      MLPutInteger(link, 4);
                      MLPutInteger(link, 5);
                      MLPutInteger(link, 6);
                  MLPutInteger(link, 1);
          }
          else if ((numCorners == 3) || (!grid->cm->fieldClasses[grid->viewField][1]))
          {
            /* Triangle 0--1--2 */
            MLPutFunction(link, "Polygon", 1);
              MLPutFunction(link, "Part", 2);
                MLPutSymbol(link, "points");
                MLPutFunction(link, "Plus", 2);
                  MLPutFunction(link, "Part", 2);
                    MLPutFunction(link, "Slot", 1);
                      MLPutInteger(link, 1);
                    MLPutFunction(link, "List", 3);
                      MLPutInteger(link, 1);
                      MLPutInteger(link, 2);
                      MLPutInteger(link, 3);
                  MLPutInteger(link, 1);
          } else {
            SETERRQ(PETSC_ERR_ARG_WRONG, "Invalid number of local nodes");
          }
          MLPutSymbol(link, "faces");
    MLPutFunction(link, "Rule", 2);
      MLPutSymbol(link, "AspectRatio");
      MLPutReal(link, 1.0);
  PetscFunctionReturn(0);
#endif
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerMathematicaVectorPlot_Triangular_2D"
int PetscViewerMathematicaVectorPlot_Triangular_2D(PetscViewer viewer, GVec v)
{
#ifdef PETSC_HAVE_MATHEMATICA
  PetscViewer_Mathematica  *vmath = (PetscViewer_Mathematica *) viewer->data;
  MLINK                link  = vmath->link; /* The link to Mathematica */
  Grid                 grid;
  Mesh                 mesh;
  PetscReal            scale;
  PetscTruth           opt;
  int                  ierr;

  PetscFunctionBegin;
  ierr = GVecGetGrid(v, &grid);                                                                           CHKERRQ(ierr);
  ierr = GridGetMesh(grid, &mesh);                                                                        CHKERRQ(ierr);

  MLPutFunction(link, "ListPlotVectorField", 3);
    MLPutSymbol(link, "points");
    MLPutFunction(link, "Rule", 2);
      MLPutSymbol(link, "AspectRatio");
      MLPutReal(link, mesh->sizeY/mesh->sizeX);
    MLPutFunction(link, "Rule", 2);
      MLPutSymbol(link, "ScaleFactor");
      ierr = PetscOptionsGetReal("viewer_", "-math_scale", &scale, &opt);                                 CHKERRQ(ierr);
      if (opt == PETSC_TRUE) {
        MLPutReal(link, scale);
      } else {
        MLPutSymbol(link, "None");
      }
  PetscFunctionReturn(0);
#endif
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerMathematicaSurfacePlot_Triangular_2D"
int PetscViewerMathematicaSurfacePlot_Triangular_2D(PetscViewer viewer, GVec v)
{
#ifdef PETSC_HAVE_MATHEMATICA
  PetscViewer_Mathematica  *vmath = (PetscViewer_Mathematica *) viewer->data;
  MLINK                link  = vmath->link; /* The link to Mathematica */
  PetscReal            startX, endX;
  PetscReal            startY, endY;
  PetscTruth           opt;
  int                  ierr;

  PetscFunctionBegin;
  /* Setup domain */
  startX = 0.0;
  startY = 0.0;
  endX   = 1.0;
  endY   = 1.0;
  ierr   = PetscOptionsGetReal("viewer_", "-math_start_x", &startX, &opt);                                CHKERRQ(ierr);
  ierr   = PetscOptionsGetReal("viewer_", "-math_start_y", &startY, &opt);                                CHKERRQ(ierr);
  ierr   = PetscOptionsGetReal("viewer_", "-math_end_x",   &endX,   &opt);                                CHKERRQ(ierr);
  ierr   = PetscOptionsGetReal("viewer_", "-math_end_y",   &endY,   &opt);                                CHKERRQ(ierr);

  MLPutFunction(link, "Show", 1);
    MLPutFunction(link, "SurfaceGraphics", 6);
      MLPutSymbol(link, "points");
      MLPutFunction(link, "Rule", 2);
        MLPutSymbol(link, "ClipFill");
        MLPutSymbol(link, "None");
      MLPutFunction(link, "Rule", 2);
        MLPutSymbol(link, "Axes");
        MLPutSymbol(link, "True");
      MLPutFunction(link, "Rule", 2);
        MLPutSymbol(link, "PlotLabel");
        MLPutString(link, vmath->objName);
      MLPutFunction(link, "Rule", 2);
        MLPutSymbol(link, "MeshRange");
        MLPutFunction(link, "List", 2);
          MLPutFunction(link, "List", 2);
            MLPutReal(link, startX);
            MLPutReal(link, endX);
          MLPutFunction(link, "List", 2);
            MLPutReal(link, startY);
            MLPutReal(link, endY);
      MLPutFunction(link, "Rule", 2);
        MLPutSymbol(link, "AspectRatio");
        MLPutReal(link, (endY - startY)/(endX - startX));
  PetscFunctionReturn(0);
#endif
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerMathematica_GVec_Triangular_2D"
int PetscViewerMathematica_GVec_Triangular_2D(PetscViewer viewer, GVec v)
{
  PetscViewer_Mathematica  *vmath = (PetscViewer_Mathematica *) viewer->data;
#ifdef PETSC_HAVE_MATHEMATICA
  MLINK                link  = vmath->link; /* The link to Mathematica */
  Mesh                 mesh;
  Grid                 grid;
  Mat                  P;
  GVec                 w;
  int                  numCorners;
  int                  ierr;

  PetscFunctionBegin;
  ierr       = GVecGetGrid(v, &grid);                                                                    CHKERRQ(ierr);
  mesh       = grid->mesh;
  numCorners = mesh->numCorners;

  /* Check that a field component has been specified */
  if ((grid->viewField < 0) || (grid->viewField >= grid->numFields)) PetscFunctionReturn(0);

  if (grid->isConstrained) {
    ierr = GVecCreate(grid, &w);                                                                         CHKERRQ(ierr);
    ierr = GridGetConstraintMatrix(grid, &P);                                                            CHKERRQ(ierr);
    ierr = MatMult(P, v, w);                                                                             CHKERRQ(ierr);
  } else {
    w = v;
  }

  /* Check that the mesh is defined correctly */
  ierr = PetscViewerMathematicaCheckMesh_Triangular_2D(viewer, mesh);                                         CHKERRQ(ierr);

  /* Send the first component of the first field */
  MLPutFunction(link, "EvaluatePacket", 1);
  switch(vmath->plotType)
  {
  case MATHEMATICA_TRIANGULATION_PLOT:
    MLPutFunction(link, "Module", 2);
      /* Make temporary points with each value of the field component in the vector */
      MLPutFunction(link, "List", 2);
        MLPutSymbol(link, "f");
        MLPutFunction(link, "Set", 2);
          MLPutSymbol(link, "points");
          ierr = PetscViewerMathematicaCreateSamplePoints_Triangular_2D(viewer, w);                           CHKERRQ(ierr);
      /* Display the picture */
      ierr = PetscViewerMathematicaTriangulationPlot_Triangular_2D(viewer, w);                                CHKERRQ(ierr);
      break;
  case MATHEMATICA_VECTOR_TRIANGULATION_PLOT:
    if (grid->fields[grid->viewField].numComp != 2) {
      SETERRQ(PETSC_ERR_ARG_WRONG, "Field must be a 2D vector field for this plot type");
    }
    MLPutFunction(link, "Module", 2);
      /* Make temporary list with points and 2D vector field values */
      MLPutFunction(link, "List", 2);
        MLPutSymbol(link, "f");
        MLPutFunction(link, "Set", 2);
          MLPutSymbol(link, "points");
          ierr = PetscViewerMathematicaCreateVectorSamplePoints_Triangular_2D(viewer, w);                     CHKERRQ(ierr);
      /* Display the picture */
      ierr = PetscViewerMathematicaVectorPlot_Triangular_2D(viewer, w);                                       CHKERRQ(ierr);
      break;
  case MATHEMATICA_VECTOR_PLOT:
    if (grid->fields[grid->viewField].numComp != 2) {
      SETERRQ(PETSC_ERR_ARG_WRONG, "Field must be a 2D vector field for this plot type");
    }
    MLPutFunction(link, "Module", 2);
      /* Make temporary list with points and 2D vector field values */
      MLPutFunction(link, "List", 2);
        MLPutSymbol(link, "f");
        MLPutFunction(link, "Set", 2);
          MLPutSymbol(link, "points");
          ierr = PetscViewerMathematicaCreateInterpolatedSamplePoints_Triangular_2D(viewer, w, -1);           CHKERRQ(ierr);
      /* Display the picture */
      ierr = PetscViewerMathematicaVectorPlot_Triangular_2D(viewer, w);                                       CHKERRQ(ierr);
      break;
  case MATHEMATICA_SURFACE_PLOT:
    if (grid->fields[grid->viewField].numComp != 2) {
      SETERRQ(PETSC_ERR_ARG_WRONG, "Field must be a 2D vector field for this plot type");
    }
    MLPutFunction(link, "Module", 2);
      /* Make temporary list with interpolated field values on a square mesh */
      MLPutFunction(link, "List", 1);
        MLPutFunction(link, "Set", 2);
          MLPutSymbol(link, "points");
          ierr = PetscViewerMathematicaCreateInterpolatedSamplePoints_Triangular_2D(viewer, w, grid->viewComp); CHKERRQ(ierr);
      /* Display the picture */
      ierr = PetscViewerMathematicaSurfacePlot_Triangular_2D(viewer, w);                                      CHKERRQ(ierr);
      break;
  default:
    SETERRQ(PETSC_ERR_ARG_WRONG, "Invalid plot type");
  }
  MLEndPacket(link);
  /* Skip packets until ReturnPacket */
  ierr = PetscViewerMathematicaSkipPackets(viewer, RETURNPKT);                                                CHKERRQ(ierr);
  /* Skip ReturnPacket */
  MLNewPacket(link);

  if (grid->isConstrained) {
    ierr = VecDestroy(w);                                                                                CHKERRQ(ierr);
  }
#else
  PetscFunctionBegin;
  switch(vmath->plotType)
  {
  case MATHEMATICA_TRIANGULATION_PLOT:
      break;
  case MATHEMATICA_VECTOR_TRIANGULATION_PLOT:
      break;
  case MATHEMATICA_VECTOR_PLOT:
      break;
  case MATHEMATICA_SURFACE_PLOT:
      break;
  default:
    SETERRQ(PETSC_ERR_ARG_WRONG, "Invalid plot type");
  }
#endif
  PetscFunctionReturn(0);
}
#endif
