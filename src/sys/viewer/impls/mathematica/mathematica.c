#define PETSC_DLL

#include "private/viewerimpl.h"   /* "petscsys.h" */
#include "private/pcimpl.h"
#include "../src/mat/impls/aij/seq/aij.h"
#include "mathematica.h"

#if defined (PETSC_HAVE__SNPRINTF) && !defined(PETSC_HAVE_SNPRINTF)
#define snprintf _snprintf
#endif

PetscViewer  PETSC_VIEWER_MATHEMATICA_WORLD_PRIVATE = PETSC_NULL;
static void *mathematicaEnv                   = PETSC_NULL;

static PetscTruth PetscViewerMathematicaPackageInitialized = PETSC_FALSE;
#undef __FUNCT__  
#define __FUNCT__ "PetscViewerMathematicaFinalizePackage"
/*@C
  PetscViewerMathematicaFinalizePackage - This function destroys everything in the Petsc interface to Mathematica. It is
  called from PetscFinalize().

  Level: developer

.keywords: Petsc, destroy, package, mathematica
.seealso: PetscFinalize()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscViewerMathematicaFinalizePackage(void) 
{
  PetscFunctionBegin;
  if (mathematicaEnv) MLDeinitialize((MLEnvironment) mathematicaEnv);
  PetscViewerMathematicaPackageInitialized = PETSC_TRUE;
  PetscFunctionReturn(0);
}

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
PetscErrorCode PETSC_DLLEXPORT PetscViewerMathematicaInitializePackage(const char path[])
{
  PetscError ierr;

  PetscFunctionBegin;
  if (PetscViewerMathematicaPackageInitialized) PetscFunctionReturn(0);
  PetscViewerMathematicaPackageInitialized = PETSC_TRUE;
  mathematicaEnv = (void*) MLInitialize(0);
  ierr = PetscRegisterFinalize(PetscViewerMathematicaFinalizePackage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "PetscViewerInitializeMathematicaWorld_Private"
PetscErrorCode PetscViewerInitializeMathematicaWorld_Private()
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PETSC_VIEWER_MATHEMATICA_WORLD_PRIVATE) PetscFunctionReturn(0);
  ierr = PetscViewerMathematicaOpen(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_NULL, PETSC_NULL, &PETSC_VIEWER_MATHEMATICA_WORLD_PRIVATE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerDestroy_Mathematica"
static PetscErrorCode PetscViewerDestroy_Mathematica(PetscViewer viewer)
{
  PetscViewer_Mathematica *vmath = (PetscViewer_Mathematica *) viewer->data; 
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  MLClose(vmath->link);
  ierr = PetscFree(vmath->linkname);CHKERRQ(ierr);
  ierr = PetscFree(vmath->linkhost);CHKERRQ(ierr);
  ierr = PetscFree(vmath);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerDestroyMathematica_Private"
PetscErrorCode PetscViewerDestroyMathematica_Private(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PETSC_VIEWER_MATHEMATICA_WORLD_PRIVATE) {
    ierr = PetscViewerDestroy(PETSC_VIEWER_MATHEMATICA_WORLD_PRIVATE);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerMathematicaSetupConnection_Private"
PetscErrorCode PetscViewerMathematicaSetupConnection_Private(PetscViewer v) 
{
  PetscViewer_Mathematica *vmath = (PetscViewer_Mathematica *) v->data;
#ifdef MATHEMATICA_3_0
  int                     argc = 6;
  char                    *argv[6];
#else
  int                     argc = 5;
  char                    *argv[5];
#endif
  char                    hostname[256];
  long                    lerr;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  /* Link name */
  argv[0] = "-linkname";
  if (!vmath->linkname) {
    argv[1] = "math -mathlink";
  } else {
    argv[1] = vmath->linkname;
  }

  /* Link host */
  argv[2] = "-linkhost";
  if (!vmath->linkhost) {
    ierr = PetscGetHostName(hostname, 255);CHKERRQ(ierr);
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
PetscErrorCode PETSC_DLLEXPORT PetscViewerCreate_Mathematica(PetscViewer v) 
{
  PetscViewer_Mathematica *vmath;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = PetscViewerMathematicaInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif

  ierr = PetscNewLog(v,PetscViewer_Mathematica, &vmath);CHKERRQ(ierr);
  v->data         = (void*) vmath;
  v->ops->destroy = PetscViewerDestroy_Mathematica;
  v->ops->flush   = 0;
  ierr = PetscStrallocpy(PETSC_VIEWER_MATHEMATICA, &((PetscObject)v)->type_name);CHKERRQ(ierr);

  vmath->linkname         = PETSC_NULL;
  vmath->linkhost         = PETSC_NULL;
  vmath->linkmode         = MATHEMATICA_LINK_CONNECT;
  vmath->graphicsType     = GRAPHICS_MOTIF;
  vmath->plotType         = MATHEMATICA_TRIANGULATION_PLOT;
  vmath->objName          = PETSC_NULL;

  ierr = PetscViewerMathematicaSetFromOptions(v);CHKERRQ(ierr);
  ierr = PetscViewerMathematicaSetupConnection_Private(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerMathematicaParseLinkMode_Private"
PetscErrorCode PetscViewerMathematicaParseLinkMode_Private(char *modename, LinkMode *mode) {
  PetscTruth     isCreate, isConnect, isLaunch;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStrcasecmp(modename, "Create",  &isCreate);CHKERRQ(ierr);
  ierr = PetscStrcasecmp(modename, "Connect", &isConnect);CHKERRQ(ierr);
  ierr = PetscStrcasecmp(modename, "Launch",  &isLaunch);CHKERRQ(ierr);
  if (isCreate) {
    *mode = MATHEMATICA_LINK_CREATE;
  } else if (isConnect) {
    *mode = MATHEMATICA_LINK_CONNECT;
  } else if (isLaunch) {
    *mode = MATHEMATICA_LINK_LAUNCH;
  } else {
    SETERRQ1(PETSC_ERR_ARG_WRONG, "Invalid Mathematica link mode: %s", modename);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerMathematicaSetFromOptions"
PetscErrorCode PETSC_DLLEXPORT PetscViewerMathematicaSetFromOptions(PetscViewer v) 
{
  PetscViewer_Mathematica  *vmath = (PetscViewer_Mathematica *) v->data;
  char                     linkname[256];
  char                     modename[256];
  char                     hostname[256];
  char                     type[256];
  PetscInt                 numPorts;
  PetscInt                 *ports;
  PetscInt                 numHosts;
  int                      h;
  char                     **hosts;
  PetscMPIInt              size, rank;
  PetscTruth               opt;
  PetscErrorCode           ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(((PetscObject)v)->comm, &size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(((PetscObject)v)->comm, &rank);CHKERRQ(ierr);

  /* Get link name */
  ierr = PetscOptionsGetString("viewer_", "-math_linkname", linkname, 255, &opt);CHKERRQ(ierr);
  if (opt) {
    ierr = PetscViewerMathematicaSetLinkName(v, linkname);CHKERRQ(ierr);
  }
  /* Get link port */
  numPorts = size;
  ierr = PetscMalloc(size*sizeof(int), &ports);CHKERRQ(ierr);
  ierr = PetscOptionsGetIntArray("viewer_", "-math_linkport", ports, &numPorts, &opt);CHKERRQ(ierr);
  if (opt) {
    if (numPorts > rank) {
      snprintf(linkname, 255, "%6d", ports[rank]);
    } else {
      snprintf(linkname, 255, "%6d", ports[0]);
    }
    ierr = PetscViewerMathematicaSetLinkName(v, linkname);CHKERRQ(ierr);
  }
  ierr = PetscFree(ports);CHKERRQ(ierr);
  /* Get link host */
  numHosts = size;
  ierr = PetscMalloc(size*sizeof(char *), &hosts);CHKERRQ(ierr);
  ierr = PetscOptionsGetStringArray("viewer_", "-math_linkhost", hosts, &numHosts, &opt);CHKERRQ(ierr);
  if (opt) {
    if (numHosts > rank) {
      ierr = PetscStrncpy(hostname, hosts[rank], 255);CHKERRQ(ierr);
    } else {
      ierr = PetscStrncpy(hostname, hosts[0], 255);CHKERRQ(ierr);
    }
    ierr = PetscViewerMathematicaSetLinkHost(v, hostname);CHKERRQ(ierr);
  }
  for(h = 0; h < numHosts; h++) {
    ierr = PetscFree(hosts[h]);CHKERRQ(ierr);
  }
  ierr = PetscFree(hosts);CHKERRQ(ierr);
  /* Get link mode */
  ierr = PetscOptionsGetString("viewer_", "-math_linkmode", modename, 255, &opt);CHKERRQ(ierr);
  if (opt) {
    LinkMode mode;

    ierr = PetscViewerMathematicaParseLinkMode_Private(modename, &mode);CHKERRQ(ierr);
    ierr = PetscViewerMathematicaSetLinkMode(v, mode);CHKERRQ(ierr);
  }
  /* Get graphics type */
  ierr = PetscOptionsGetString("viewer_", "-math_graphics", type, 255, &opt);CHKERRQ(ierr);
  if (opt) {
    PetscTruth isMotif, isPS, isPSFile;

    ierr = PetscStrcasecmp(type, "Motif",  &isMotif);CHKERRQ(ierr);
    ierr = PetscStrcasecmp(type, "PS",     &isPS);CHKERRQ(ierr);
    ierr = PetscStrcasecmp(type, "PSFile", &isPSFile);CHKERRQ(ierr);
    if (isMotif) {
      vmath->graphicsType = GRAPHICS_MOTIF;
    } else if (isPS) {
      vmath->graphicsType = GRAPHICS_PS_STDOUT;
    } else if (isPSFile) {
      vmath->graphicsType = GRAPHICS_PS_FILE;
    }
  }
  /* Get plot type */
  ierr = PetscOptionsGetString("viewer_", "-math_type", type, 255, &opt);CHKERRQ(ierr);
  if (opt) {
    PetscTruth isTri, isVecTri, isVec, isSurface;

    ierr = PetscStrcasecmp(type, "Triangulation",       &isTri);CHKERRQ(ierr);
    ierr = PetscStrcasecmp(type, "VectorTriangulation", &isVecTri);CHKERRQ(ierr);
    ierr = PetscStrcasecmp(type, "Vector",              &isVec);CHKERRQ(ierr);
    ierr = PetscStrcasecmp(type, "Surface",             &isSurface);CHKERRQ(ierr);
    if (isTri) {
      vmath->plotType     = MATHEMATICA_TRIANGULATION_PLOT;
    } else if (isVecTri) {
      vmath->plotType     = MATHEMATICA_VECTOR_TRIANGULATION_PLOT;
    } else if (isVec) {
      vmath->plotType     = MATHEMATICA_VECTOR_PLOT;
    } else if (isSurface) {
      vmath->plotType     = MATHEMATICA_SURFACE_PLOT;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerMathematicaSetLinkName"
PetscErrorCode PETSC_DLLEXPORT PetscViewerMathematicaSetLinkName(PetscViewer v, const char *name) {
  PetscViewer_Mathematica *vmath = (PetscViewer_Mathematica *) v->data;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,PETSC_VIEWER_COOKIE,1);
  PetscValidCharPointer(name,2);
  ierr = PetscStrallocpy(name, &vmath->linkname);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerMathematicaLinkPort"
PetscErrorCode PETSC_DLLEXPORT PetscViewerMathematicaSetLinkPort(PetscViewer v, int port) {
  char           name[16];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  snprintf(name, 16, "%6d", port);
  ierr = PetscViewerMathematicaSetLinkName(v, name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerMathematicaSetLinkHost"
PetscErrorCode PETSC_DLLEXPORT PetscViewerMathematicaSetLinkHost(PetscViewer v, const char *host) {
  PetscViewer_Mathematica *vmath = (PetscViewer_Mathematica *) v->data;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,PETSC_VIEWER_COOKIE,1);
  PetscValidCharPointer(host,2);
  ierr = PetscStrallocpy(host, &vmath->linkhost);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscViewerMathematicaSetLinkHost"
PetscErrorCode PETSC_DLLEXPORT PetscViewerMathematicaSetLinkMode(PetscViewer v, LinkMode mode) {
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
PetscErrorCode PETSC_DLLEXPORT PetscViewerMathematicaOpen(MPI_Comm comm, int port, const char machine[], const char mode[], PetscViewer *v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerCreate(comm, v);CHKERRQ(ierr);
#if 0
  LinkMode linkmode;
  ierr = PetscViewerMathematicaSetLinkPort(*v, port);CHKERRQ(ierr);
  ierr = PetscViewerMathematicaSetLinkHost(*v, machine);CHKERRQ(ierr);
  ierr = PetscViewerMathematicaParseLinkMode_Private(mode, &linkmode);CHKERRQ(ierr);
  ierr = PetscViewerMathematicaSetLinkMode(*v, linkmode);CHKERRQ(ierr);
#endif
  ierr = PetscViewerSetType(*v, PETSC_VIEWER_MATHEMATICA);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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
PetscErrorCode PETSC_DLLEXPORT PetscViewerMathematicaGetLink(PetscViewer viewer, MLINK *link)
{
  PetscViewer_Mathematica *vmath = (PetscViewer_Mathematica *) viewer->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_COOKIE,1);
  *link = vmath->link;
  PetscFunctionReturn(0);
}

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
PetscErrorCode PETSC_DLLEXPORT PetscViewerMathematicaSkipPackets(PetscViewer viewer, int type)
{
  PetscViewer_Mathematica *vmath = (PetscViewer_Mathematica *) viewer->data;
  MLINK                   link  = vmath->link; /* The link to Mathematica */
  int                     pkt;                 /* The packet type */

  PetscFunctionBegin;
  while((pkt = MLNextPacket(link)) && (pkt != type))
    MLNewPacket(link);
  if (!pkt) {
    MLClearError(link);
    SETERRQ(PETSC_ERR_LIB, (char *) MLErrorMessage(link));
  }
  PetscFunctionReturn(0);
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
PetscErrorCode PETSC_DLLEXPORT PetscViewerMathematicaGetName(PetscViewer viewer, const char **name)
{
  PetscViewer_Mathematica *vmath = (PetscViewer_Mathematica *) viewer->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_COOKIE,1);
  PetscValidPointer(name,2);
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
PetscErrorCode PETSC_DLLEXPORT PetscViewerMathematicaSetName(PetscViewer viewer, const char name[])
{
  PetscViewer_Mathematica *vmath = (PetscViewer_Mathematica *) viewer->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_COOKIE,1);
  PetscValidPointer(name,2);
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
PetscErrorCode PETSC_DLLEXPORT PetscViewerMathematicaClearName(PetscViewer viewer)
{
  PetscViewer_Mathematica *vmath = (PetscViewer_Mathematica *) viewer->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_COOKIE,1);
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
PetscErrorCode PETSC_DLLEXPORT PetscViewerMathematicaGetVector(PetscViewer viewer, Vec v) {
  PetscViewer_Mathematica *vmath = (PetscViewer_Mathematica *) viewer->data;
  MLINK                   link;   /* The link to Mathematica */
  char                    *name;
  PetscScalar             *mArray,*array;
  long                    mSize;
  int                     n;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_COOKIE,1);
  PetscValidHeaderSpecific(v,      VEC_COOKIE,2);

  /* Determine the object name */
  if (!vmath->objName) {
    name = "vec";
  } else {
    name = (char *) vmath->objName;
  }

  link = vmath->link;
  ierr = VecGetLocalSize(v, &n);CHKERRQ(ierr);
  ierr = VecGetArray(v, &array);CHKERRQ(ierr);
  MLPutFunction(link, "EvaluatePacket", 1);
    MLPutSymbol(link, name);
  MLEndPacket(link);
  ierr = PetscViewerMathematicaSkipPackets(viewer, RETURNPKT);CHKERRQ(ierr);
  MLGetRealList(link, &mArray, &mSize);
  if (n != mSize) SETERRQ2(PETSC_ERR_ARG_WRONG, "Incompatible vector sizes %d %d",n,mSize);
  ierr = PetscMemcpy(array, mArray, mSize * sizeof(double));CHKERRQ(ierr);
  MLDisownRealList(link, mArray, mSize);
  ierr = VecRestoreArray(v, &array);CHKERRQ(ierr);

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
PetscErrorCode PETSC_DLLEXPORT PetscViewerMathematicaPutVector(PetscViewer viewer, Vec v) 
{
  PetscViewer_Mathematica *vmath = (PetscViewer_Mathematica *) viewer->data;
  MLINK                   link  = vmath->link; /* The link to Mathematica */
  char                    *name;
  PetscScalar             *array;
  int                     n;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  /* Determine the object name */
  if (!vmath->objName) {
    name = "vec";
  } else {
    name = (char *) vmath->objName;
  }
  ierr = VecGetLocalSize(v, &n);CHKERRQ(ierr);
  ierr = VecGetArray(v, &array);CHKERRQ(ierr);

  /* Send the Vector object */
  MLPutFunction(link, "EvaluatePacket", 1);
    MLPutFunction(link, "Set", 2);
      MLPutSymbol(link, name);
      MLPutRealList(link, array, n);
  MLEndPacket(link);
  /* Skip packets until ReturnPacket */
  ierr = PetscViewerMathematicaSkipPackets(viewer, RETURNPKT);CHKERRQ(ierr);
  /* Skip ReturnPacket */
  MLNewPacket(link);

  ierr = VecRestoreArray(v, &array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PETSC_DLLEXPORT PetscViewerMathematicaPutMatrix(PetscViewer viewer, int m, int n, PetscReal *a) 
{
  PetscViewer_Mathematica *vmath = (PetscViewer_Mathematica *) viewer->data;
  MLINK                   link  = vmath->link; /* The link to Mathematica */
  char                    *name;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  /* Determine the object name */
  if (!vmath->objName) {
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
  ierr = PetscViewerMathematicaSkipPackets(viewer, RETURNPKT);CHKERRQ(ierr);
  /* Skip ReturnPacket */
  MLNewPacket(link);

  PetscFunctionReturn(0);
}

PetscErrorCode PETSC_DLLEXPORT PetscViewerMathematicaPutCSRMatrix(PetscViewer viewer, int m, int n, int *i, int *j, PetscReal *a) 
{
  PetscViewer_Mathematica *vmath = (PetscViewer_Mathematica *) viewer->data;
  MLINK                   link  = vmath->link; /* The link to Mathematica */
  const char              *symbol;
  char                    *name;
  PetscTruth              match;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  /* Determine the object name */
  if (!vmath->objName) {
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
  ierr = PetscViewerMathematicaSkipPackets(viewer, RETURNPKT);CHKERRQ(ierr);
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
  ierr = PetscViewerMathematicaSkipPackets(viewer, RETURNPKT);CHKERRQ(ierr);
  /* Skip ReturnPacket */
  MLNewPacket(link);

  /* Check that matrix is valid */
  MLPutFunction(link, "EvaluatePacket", 1);
    MLPutFunction(link, "ValidQ", 1);
      MLPutSymbol(link, name);
  MLEndPacket(link);
  ierr = PetscViewerMathematicaSkipPackets(viewer, RETURNPKT);CHKERRQ(ierr);
  MLGetSymbol(link, &symbol);
  ierr = PetscStrcmp("True", (char *) symbol, &match);CHKERRQ(ierr);
  if (!match) {
    MLDisownSymbol(link, symbol);
    SETERRQ(PETSC_ERR_PLIB, "Invalid CSR matrix in Mathematica");
  }
  MLDisownSymbol(link, symbol);
  /* Skip ReturnPacket */
  MLNewPacket(link);

  PetscFunctionReturn(0);
}

