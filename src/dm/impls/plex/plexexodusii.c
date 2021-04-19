#define PETSCDM_DLL
#include <petsc/private/dmpleximpl.h>    /*I   "petscdmplex.h"   I*/

#if defined(PETSC_HAVE_EXODUSII)
#include <netcdf.h>
#include <exodusII.h>
#endif

#include <petsc/private/viewerimpl.h>
#include <petsc/private/viewerexodusiiimpl.h>
#if defined(PETSC_HAVE_EXODUSII)
/*
  PETSC_VIEWER_EXODUSII_ - Creates an ExodusII PetscViewer shared by all processors in a communicator.

  Collective

  Input Parameter:
. comm - the MPI communicator to share the ExodusII PetscViewer

  Level: intermediate

  Notes:
    misses Fortran bindings

  Notes:
  Unlike almost all other PETSc routines, PETSC_VIEWER_EXODUSII_ does not return
  an error code.  The GLVIS PetscViewer is usually used in the form
$       XXXView(XXX object, PETSC_VIEWER_EXODUSII_(comm));

.seealso: PetscViewerExodusIIOpen(), PetscViewerType, PetscViewerCreate(), PetscViewerDestroy()
*/
PetscViewer PETSC_VIEWER_EXODUSII_(MPI_Comm comm)
{
  PetscViewer    viewer;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerExodusIIOpen(comm, "mesh.exo", FILE_MODE_WRITE, &viewer);
  if (ierr) {PetscError(PETSC_COMM_SELF,__LINE__,"PETSC_VIEWER_EXODUSII_",__FILE__,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL," ");PetscFunctionReturn(0);}
  ierr = PetscObjectRegisterDestroy((PetscObject) viewer);
  if (ierr) {PetscError(PETSC_COMM_SELF,__LINE__,"PETSC_VIEWER_EXODUSII_",__FILE__,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL," ");PetscFunctionReturn(0);}
  PetscFunctionReturn(viewer);
}

static PetscErrorCode PetscViewerView_ExodusII(PetscViewer v, PetscViewer viewer)
{
  PetscViewer_ExodusII *exo = (PetscViewer_ExodusII *) v->data;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  if (exo->filename) {ierr = PetscViewerASCIIPrintf(viewer, "Filename:    %s\n", exo->filename);CHKERRQ(ierr);}
  if (exo->exoid)    {ierr = PetscViewerASCIIPrintf(viewer, "exoid:       %d\n", exo->exoid)   ;CHKERRQ(ierr);}
  if (exo->btype)    {ierr = PetscViewerASCIIPrintf(viewer, "IO Mode:     %d\n", exo->btype)   ;CHKERRQ(ierr);}
  if (exo->order)    {ierr = PetscViewerASCIIPrintf(viewer, "Mesh order:  %d\n", exo->order)   ;CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerSetFromOptions_ExodusII(PetscOptionItems *PetscOptionsObject, PetscViewer v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject, "ExodusII PetscViewer Options");CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerSetUp_ExodusII(PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerDestroy_ExodusII(PetscViewer viewer)
{
  PetscViewer_ExodusII *exo = (PetscViewer_ExodusII *) viewer->data;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  if (exo->exoid >= 0) {PetscStackCallStandard(ex_close,(exo->exoid));}
  ierr = PetscFree(exo->filename);CHKERRQ(ierr);
  ierr = PetscFree(exo);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)viewer,"PetscViewerFileSetName_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)viewer,"PetscViewerFileGetName_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)viewer,"PetscViewerFileSetMode_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)viewer,"PetscViewerGetId_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)viewer,"PetscViewerGetOrder_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)viewer,"PetscViewerSetOrder_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerFileSetName_ExodusII(PetscViewer viewer, const char name[])
{
  PetscViewer_ExodusII *exo = (PetscViewer_ExodusII *) viewer->data;
  PetscMPIInt           rank;
  int                   CPU_word_size, IO_word_size, EXO_mode;
  PetscErrorCode        ierr;
  MPI_Info              mpi_info = MPI_INFO_NULL;
  float                 EXO_version;

  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject) viewer), &rank);CHKERRMPI(ierr);
  CPU_word_size = sizeof(PetscReal);
  IO_word_size  = sizeof(PetscReal);

  PetscFunctionBegin;
  if (exo->exoid >= 0) {
    PetscStackCallStandard(ex_close,(exo->exoid));
    exo->exoid = -1;
  }
  if (exo->filename) {ierr = PetscFree(exo->filename);CHKERRQ(ierr);}
  ierr = PetscStrallocpy(name, &exo->filename);CHKERRQ(ierr);
  switch (exo->btype) {
  case FILE_MODE_READ:
    EXO_mode = EX_READ;
    break;
  case FILE_MODE_APPEND:
  case FILE_MODE_UPDATE:
  case FILE_MODE_APPEND_UPDATE:
    /* Will fail if the file does not already exist */
    EXO_mode = EX_WRITE;
    break;
  case FILE_MODE_WRITE:
    /*
      exodus only allows writing geometry upon file creation, so we will let DMView create the file.
    */
    PetscFunctionReturn(0);
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER, "Must call PetscViewerFileSetMode() before PetscViewerFileSetName()");
  }
  #if defined(PETSC_USE_64BIT_INDICES)
  EXO_mode += EX_ALL_INT64_API;
  #endif
  exo->exoid = ex_open_par(name,EXO_mode,&CPU_word_size,&IO_word_size,&EXO_version,PETSC_COMM_WORLD,mpi_info);
  if (exo->exoid < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_LIB, "ex_open_par failed for %s", name);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerFileGetName_ExodusII(PetscViewer viewer, const char **name)
{
  PetscViewer_ExodusII *exo = (PetscViewer_ExodusII *) viewer->data;

  PetscFunctionBegin;
  *name = exo->filename;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerFileSetMode_ExodusII(PetscViewer viewer, PetscFileMode type)
{
  PetscViewer_ExodusII *exo = (PetscViewer_ExodusII *) viewer->data;

  PetscFunctionBegin;
  exo->btype = type;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerFileGetMode_ExodusII(PetscViewer viewer, PetscFileMode *type)
{
  PetscViewer_ExodusII *exo = (PetscViewer_ExodusII *) viewer->data;

  PetscFunctionBegin;
  *type = exo->btype;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerExodusIIGetId_ExodusII(PetscViewer viewer, int *exoid)
{
  PetscViewer_ExodusII *exo = (PetscViewer_ExodusII *) viewer->data;

  PetscFunctionBegin;
  *exoid = exo->exoid;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerExodusIIGetOrder_ExodusII(PetscViewer viewer, PetscInt *order)
{
  PetscViewer_ExodusII *exo = (PetscViewer_ExodusII *) viewer->data;

  PetscFunctionBegin;
  *order = exo->order;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerExodusIISetOrder_ExodusII(PetscViewer viewer, PetscInt order)
{
  PetscViewer_ExodusII *exo = (PetscViewer_ExodusII *) viewer->data;

  PetscFunctionBegin;
  exo->order = order;
  PetscFunctionReturn(0);
}

/*MC
   PETSCVIEWEREXODUSII - A viewer that writes to an Exodus II file


.seealso:  PetscViewerExodusIIOpen(), PetscViewerCreate(), PETSCVIEWERBINARY, PETSCVIEWERHDF5, DMView(),
           PetscViewerFileSetName(), PetscViewerFileSetMode(), PetscViewerFormat, PetscViewerType, PetscViewerSetType()

  Level: beginner
M*/

PETSC_EXTERN PetscErrorCode PetscViewerCreate_ExodusII(PetscViewer v)
{
  PetscViewer_ExodusII *exo;
  PetscErrorCode        ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(v,&exo);CHKERRQ(ierr);

  v->data                = (void*) exo;
  v->ops->destroy        = PetscViewerDestroy_ExodusII;
  v->ops->setfromoptions = PetscViewerSetFromOptions_ExodusII;
  v->ops->setup          = PetscViewerSetUp_ExodusII;
  v->ops->view           = PetscViewerView_ExodusII;
  v->ops->flush          = 0;
  exo->btype             = (PetscFileMode) -1;
  exo->filename          = 0;
  exo->exoid             = -1;

  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerFileSetName_C",PetscViewerFileSetName_ExodusII);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerFileGetName_C",PetscViewerFileGetName_ExodusII);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerFileSetMode_C",PetscViewerFileSetMode_ExodusII);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerFileGetMode_C",PetscViewerFileGetMode_ExodusII);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerGetId_C",PetscViewerExodusIIGetId_ExodusII);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerSetOrder_C",PetscViewerExodusIISetOrder_ExodusII);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)v,"PetscViewerGetOrder_C",PetscViewerExodusIIGetOrder_ExodusII);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  EXOGetVarIndex - Locate a result in an exodus file based on its name

  Collective

  Input Parameters:
+ exoid    - the exodus id of a file (obtained from ex_open or ex_create for instance)
. obj_type - the type of entity for instance EX_NODAL, EX_ELEM_BLOCK
- name     - the name of the result

  Output Parameters:
. varIndex - the location in the exodus file of the result

  Notes:
  The exodus variable index is obtained by comparing name and the
  names of zonal variables declared in the exodus file. For instance if name is "V"
  the location in the exodus file will be the first match of "V", "V_X", "V_XX", "V_1", or "V_11"
  amongst all variables of type obj_type.

  Level: beginner

.seealso: EXOGetVarIndex(),DMPlexView_ExodusII_Internal(),VecViewPlex_ExodusII_Nodal_Internal(),VecLoadNodal_PlexEXO(),VecLoadZonal_PlexEXO()
*/
PetscErrorCode EXOGetVarIndex_Internal(int exoid, ex_entity_type obj_type, const char name[], int *varIndex)
{
  int            num_vars, i, j;
  char           ext_name[MAX_STR_LENGTH+1], var_name[MAX_STR_LENGTH+1];
  const int      num_suffix = 5;
  char          *suffix[5];
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  suffix[0] = (char *) "";
  suffix[1] = (char *) "_X";
  suffix[2] = (char *) "_XX";
  suffix[3] = (char *) "_1";
  suffix[4] = (char *) "_11";

  *varIndex = -1;
  PetscStackCallStandard(ex_get_variable_param,(exoid, obj_type, &num_vars));
  for (i = 0; i < num_vars; ++i) {
    PetscStackCallStandard(ex_get_variable_name,(exoid, obj_type, i+1, var_name));
    for (j = 0; j < num_suffix; ++j){
      ierr = PetscStrncpy(ext_name, name, MAX_STR_LENGTH);CHKERRQ(ierr);
      ierr = PetscStrlcat(ext_name, suffix[j], MAX_STR_LENGTH);CHKERRQ(ierr);
      ierr = PetscStrcasecmp(ext_name, var_name, &flg);
      if (flg) {
        *varIndex = i+1;
      }
    }
  }
  PetscFunctionReturn(0);
}

/*
  DMView_PlexExodusII - Write a DM to disk in exodus format

  Collective on dm

  Input Parameters:
+ dm  - The dm to be written
. viewer - an exodusII viewer

  Notes:
  Not all DM can be written to disk this way. For instance, exodus assume that element blocks (mapped to "Cell sets" labels)
  consists of sequentially numbered cells. If this is not the case, the exodus file will be corrupted.

  If the dm has been distributed, only the part of the DM on MPI rank 0 (including "ghost" cells and vertices)
  will be written.

  DMPlex only represents geometry while most post-processing software expect that a mesh also provides information
  on the discretization space. This function assumes that the file represents Lagrange finite elements of order 1 or 2.
  The order of the mesh shall be set using PetscViewerExodusIISetOrder
  It should be extended to use PetscFE objects.

  This function will only handle TRI, TET, QUAD, and HEX cells.
  Level: beginner

.seealso:
*/
PetscErrorCode DMView_PlexExodusII(DM dm, PetscViewer viewer)
{
  enum ElemType {TRI, QUAD, TET, HEX};
  MPI_Comm        comm;
  PetscInt        degree; /* the order of the mesh */
  /* Connectivity Variables */
  PetscInt        cellsNotInConnectivity;
  /* Cell Sets */
  DMLabel         csLabel;
  IS              csIS;
  const PetscInt *csIdx;
  PetscInt        num_cs, cs;
  enum ElemType  *type;
  PetscBool       hasLabel;
  /* Coordinate Variables */
  DM              cdm;
  PetscSection    coordSection;
  Vec             coord;
  PetscInt      **nodes;
  PetscInt        depth, d, dim, skipCells = 0;
  PetscInt        pStart, pEnd, p, cStart, cEnd, numCells, vStart, vEnd, numVertices, eStart, eEnd, numEdges, fStart, fEnd, numFaces, numNodes;
  PetscInt        num_vs, num_fs;
  PetscMPIInt     rank, size;
  const char     *dmName;
  PetscInt        nodesTriP1[4]  = {3,0,0,0};
  PetscInt        nodesTriP2[4]  = {3,3,0,0};
  PetscInt        nodesQuadP1[4] = {4,0,0,0};
  PetscInt        nodesQuadP2[4] = {4,4,0,1};
  PetscInt        nodesTetP1[4]  = {4,0,0,0};
  PetscInt        nodesTetP2[4]  = {4,6,0,0};
  PetscInt        nodesHexP1[4]  = {8,0,0,0};
  PetscInt        nodesHexP2[4]  = {8,12,6,1};
  PetscErrorCode  ierr;

  int             CPU_word_size, IO_word_size, EXO_mode;
  float           EXO_version;

  PetscViewer_ExodusII *exo = (PetscViewer_ExodusII *) viewer->data;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRMPI(ierr);

  /*
    Creating coordSection is a collective operation so we do it somewhat out of sequence
  */
  ierr = PetscSectionCreate(comm, &coordSection);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocalSetUp(dm);CHKERRQ(ierr);
  if (!rank) {
    switch (exo->btype) {
    case FILE_MODE_READ:
    case FILE_MODE_APPEND:
    case FILE_MODE_UPDATE:
    case FILE_MODE_APPEND_UPDATE:
      /* exodusII does not allow writing geometry to an existing file */
      SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_LIB, "cannot add geometry to existing file %s", exo->filename);
    case FILE_MODE_WRITE:
      /* Create an empty file if one already exists*/
      EXO_mode = EX_CLOBBER;
#if defined(PETSC_USE_64BIT_INDICES)
      EXO_mode += EX_ALL_INT64_API;
#endif
        CPU_word_size = sizeof(PetscReal);
        IO_word_size  = sizeof(PetscReal);
        exo->exoid = ex_create(exo->filename, EXO_mode, &CPU_word_size, &IO_word_size);
        if (exo->exoid < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_LIB, "ex_create failed for %s", exo->filename);

      break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER, "Must call PetscViewerFileSetMode() before PetscViewerFileSetName()");
    }

    /* --- Get DM info --- */
    ierr = PetscObjectGetName((PetscObject) dm, &dmName);CHKERRQ(ierr);
    ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
    ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
    ierr = DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd);CHKERRQ(ierr);
    ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
    numCells    = cEnd - cStart;
    numEdges    = eEnd - eStart;
    numVertices = vEnd - vStart;
    if (depth == 3) {numFaces = fEnd - fStart;}
    else            {numFaces = 0;}
    ierr = DMGetLabelSize(dm, "Cell Sets", &num_cs);CHKERRQ(ierr);
    ierr = DMGetLabelSize(dm, "Vertex Sets", &num_vs);CHKERRQ(ierr);
    ierr = DMGetLabelSize(dm, "Face Sets", &num_fs);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(dm, &coord);CHKERRQ(ierr);
    ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
    if (num_cs > 0) {
      ierr = DMGetLabel(dm, "Cell Sets", &csLabel);CHKERRQ(ierr);
      ierr = DMLabelGetValueIS(csLabel, &csIS);CHKERRQ(ierr);
      ierr = ISGetIndices(csIS, &csIdx);CHKERRQ(ierr);
    }
    ierr = PetscMalloc1(num_cs, &nodes);CHKERRQ(ierr);
    /* Set element type for each block and compute total number of nodes */
    ierr = PetscMalloc1(num_cs, &type);CHKERRQ(ierr);
    numNodes = numVertices;

    ierr = PetscViewerExodusIIGetOrder(viewer, &degree);CHKERRQ(ierr);
    if (degree == 2) {numNodes += numEdges;}
    cellsNotInConnectivity = numCells;
    for (cs = 0; cs < num_cs; ++cs) {
      IS              stratumIS;
      const PetscInt *cells;
      PetscScalar    *xyz = NULL;
      PetscInt        csSize, closureSize;

      ierr = DMLabelGetStratumIS(csLabel, csIdx[cs], &stratumIS);CHKERRQ(ierr);
      ierr = ISGetIndices(stratumIS, &cells);CHKERRQ(ierr);
      ierr = ISGetSize(stratumIS, &csSize);CHKERRQ(ierr);
      ierr = DMPlexVecGetClosure(cdm, NULL, coord, cells[0], &closureSize, &xyz);CHKERRQ(ierr);
      switch (dim) {
        case 2:
          if      (closureSize == 3*dim) {type[cs] = TRI;}
          else if (closureSize == 4*dim) {type[cs] = QUAD;}
          else SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Number of vertices %D in dimension %D has no ExodusII type", closureSize/dim, dim);
          break;
        case 3:
          if      (closureSize == 4*dim) {type[cs] = TET;}
          else if (closureSize == 8*dim) {type[cs] = HEX;}
          else SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Number of vertices %D in dimension %D has no ExodusII type", closureSize/dim, dim);
          break;
        default: SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Dimension %D not handled by ExodusII viewer", dim);
      }
      if ((degree == 2) && (type[cs] == QUAD)) {numNodes += csSize;}
      if ((degree == 2) && (type[cs] == HEX))  {numNodes += csSize; numNodes += numFaces;}
      ierr = DMPlexVecRestoreClosure(cdm, NULL, coord, cells[0], &closureSize, &xyz);CHKERRQ(ierr);
      /* Set nodes and Element type */
      if (type[cs] == TRI) {
        if      (degree == 1) nodes[cs] = nodesTriP1;
        else if (degree == 2) nodes[cs] = nodesTriP2;
      } else if (type[cs] == QUAD) {
        if      (degree == 1) nodes[cs] = nodesQuadP1;
        else if (degree == 2) nodes[cs] = nodesQuadP2;
      } else if (type[cs] == TET) {
        if      (degree == 1) nodes[cs] = nodesTetP1;
        else if (degree == 2) nodes[cs] = nodesTetP2;
      } else if (type[cs] == HEX) {
        if      (degree == 1) nodes[cs] = nodesHexP1;
        else if (degree == 2) nodes[cs] = nodesHexP2;
      }
      /* Compute the number of cells not in the connectivity table */
      cellsNotInConnectivity -= nodes[cs][3]*csSize;

      ierr = ISRestoreIndices(stratumIS, &cells);CHKERRQ(ierr);
      ierr = ISDestroy(&stratumIS);CHKERRQ(ierr);
    }
    if (num_cs > 0) {PetscStackCallStandard(ex_put_init,(exo->exoid, dmName, dim, numNodes, numCells, num_cs, num_vs, num_fs));}
    /* --- Connectivity --- */
    for (cs = 0; cs < num_cs; ++cs) {
      IS              stratumIS;
      const PetscInt *cells;
      PetscInt       *connect, off = 0;
      PetscInt        edgesInClosure = 0, facesInClosure = 0, verticesInClosure = 0;
      PetscInt        csSize, c, connectSize, closureSize;
      char           *elem_type = NULL;
      char            elem_type_tri3[]  = "TRI3",  elem_type_quad4[] = "QUAD4";
      char            elem_type_tri6[]  = "TRI6",  elem_type_quad9[] = "QUAD9";
      char            elem_type_tet4[]  = "TET4",  elem_type_hex8[]  = "HEX8";
      char            elem_type_tet10[] = "TET10", elem_type_hex27[] = "HEX27";

      ierr = DMLabelGetStratumIS(csLabel, csIdx[cs], &stratumIS);CHKERRQ(ierr);
      ierr = ISGetIndices(stratumIS, &cells);CHKERRQ(ierr);
      ierr = ISGetSize(stratumIS, &csSize);CHKERRQ(ierr);
      /* Set Element type */
      if (type[cs] == TRI) {
        if      (degree == 1) elem_type = elem_type_tri3;
        else if (degree == 2) elem_type = elem_type_tri6;
      } else if (type[cs] == QUAD) {
        if      (degree == 1) elem_type = elem_type_quad4;
        else if (degree == 2) elem_type = elem_type_quad9;
      } else if (type[cs] == TET) {
        if      (degree == 1) elem_type = elem_type_tet4;
        else if (degree == 2) elem_type = elem_type_tet10;
      } else if (type[cs] == HEX) {
        if      (degree == 1) elem_type = elem_type_hex8;
        else if (degree == 2) elem_type = elem_type_hex27;
      }
      connectSize = nodes[cs][0] + nodes[cs][1] + nodes[cs][2] + nodes[cs][3];
      ierr = PetscMalloc1(PetscMax(27,connectSize)*csSize, &connect);CHKERRQ(ierr);
      PetscStackCallStandard(ex_put_block,(exo->exoid, EX_ELEM_BLOCK, csIdx[cs], elem_type, csSize, connectSize, 0, 0, 1));
      /* Find number of vertices, edges, and faces in the closure */
      verticesInClosure = nodes[cs][0];
      if (depth > 1) {
        if (dim == 2) {
          ierr = DMPlexGetConeSize(dm, cells[0], &edgesInClosure);CHKERRQ(ierr);
        } else if (dim == 3) {
          PetscInt *closure = NULL;

          ierr = DMPlexGetConeSize(dm, cells[0], &facesInClosure);CHKERRQ(ierr);
          ierr = DMPlexGetTransitiveClosure(dm, cells[0], PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
          edgesInClosure = closureSize - facesInClosure - 1 - verticesInClosure;
          ierr = DMPlexRestoreTransitiveClosure(dm, cells[0], PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
        }
      }
      /* Get connectivity for each cell */
      for (c = 0; c < csSize; ++c) {
        PetscInt *closure = NULL;
        PetscInt  temp, i;

        ierr = DMPlexGetTransitiveClosure(dm, cells[c], PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
        for (i = 0; i < connectSize; ++i) {
          if (i < nodes[cs][0]) {/* Vertices */
            connect[i+off] = closure[(i+edgesInClosure+facesInClosure+1)*2] + 1;
            connect[i+off] -= cellsNotInConnectivity;
          } else if (i < nodes[cs][0]+nodes[cs][1]) { /* Edges */
            connect[i+off] = closure[(i-verticesInClosure+facesInClosure+1)*2] + 1;
            if (nodes[cs][2] == 0) connect[i+off] -= numFaces;
            connect[i+off] -= cellsNotInConnectivity;
          } else if (i < nodes[cs][0]+nodes[cs][1]+nodes[cs][3]){ /* Cells */
            connect[i+off] = closure[0] + 1;
            connect[i+off] -= skipCells;
          } else if (i < nodes[cs][0]+nodes[cs][1]+nodes[cs][3]+nodes[cs][2]){ /* Faces */
            connect[i+off] = closure[(i-edgesInClosure-verticesInClosure)*2] + 1;
            connect[i+off] -= cellsNotInConnectivity;
          } else {
            connect[i+off] = -1;
          }
        }
        /* Tetrahedra are inverted */
        if (type[cs] == TET) {
          temp = connect[0+off]; connect[0+off] = connect[1+off]; connect[1+off] = temp;
          if (degree == 2) {
            temp = connect[5+off]; connect[5+off] = connect[6+off]; connect[6+off] = temp;
            temp = connect[7+off]; connect[7+off] = connect[8+off]; connect[8+off] = temp;
          }
        }
        /* Hexahedra are inverted */
        if (type[cs] == HEX) {
          temp = connect[1+off]; connect[1+off] = connect[3+off]; connect[3+off] = temp;
          if (degree == 2) {
            temp = connect[8+off];  connect[8+off]  = connect[11+off]; connect[11+off] = temp;
            temp = connect[9+off];  connect[9+off]  = connect[10+off]; connect[10+off] = temp;
            temp = connect[16+off]; connect[16+off] = connect[17+off]; connect[17+off] = temp;
            temp = connect[18+off]; connect[18+off] = connect[19+off]; connect[19+off] = temp;

            temp = connect[12+off]; connect[12+off] = connect[16+off]; connect[16+off] = temp;
            temp = connect[13+off]; connect[13+off] = connect[17+off]; connect[17+off] = temp;
            temp = connect[14+off]; connect[14+off] = connect[18+off]; connect[18+off] = temp;
            temp = connect[15+off]; connect[15+off] = connect[19+off]; connect[19+off] = temp;

            temp = connect[23+off]; connect[23+off] = connect[26+off]; connect[26+off] = temp;
            temp = connect[24+off]; connect[24+off] = connect[25+off]; connect[25+off] = temp;
            temp = connect[25+off]; connect[25+off] = connect[26+off]; connect[26+off] = temp;
          }
        }
        off += connectSize;
        ierr = DMPlexRestoreTransitiveClosure(dm, cells[c], PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      }
      PetscStackCallStandard(ex_put_conn,(exo->exoid, EX_ELEM_BLOCK, csIdx[cs], connect, 0, 0));
      skipCells += (nodes[cs][3] == 0)*csSize;
      ierr = PetscFree(connect);CHKERRQ(ierr);
      ierr = ISRestoreIndices(stratumIS, &cells);CHKERRQ(ierr);
      ierr = ISDestroy(&stratumIS);CHKERRQ(ierr);
    }
    ierr = PetscFree(type);CHKERRQ(ierr);
    /* --- Coordinates --- */
    ierr = PetscSectionSetChart(coordSection, pStart, pEnd);CHKERRQ(ierr);
    if (num_cs) {
      for (d = 0; d < depth; ++d) {
        ierr = DMPlexGetDepthStratum(dm, d, &pStart, &pEnd);CHKERRQ(ierr);
        for (p = pStart; p < pEnd; ++p) {
          ierr = PetscSectionSetDof(coordSection, p, nodes[0][d] > 0);CHKERRQ(ierr);
        }
      }
    }
    for (cs = 0; cs < num_cs; ++cs) {
      IS              stratumIS;
      const PetscInt *cells;
      PetscInt        csSize, c;

      ierr = DMLabelGetStratumIS(csLabel, csIdx[cs], &stratumIS);CHKERRQ(ierr);
      ierr = ISGetIndices(stratumIS, &cells);CHKERRQ(ierr);
      ierr = ISGetSize(stratumIS, &csSize);CHKERRQ(ierr);
      for (c = 0; c < csSize; ++c) {
        ierr = PetscSectionSetDof(coordSection, cells[c], nodes[cs][3] > 0);CHKERRQ(ierr);
      }
      ierr = ISRestoreIndices(stratumIS, &cells);CHKERRQ(ierr);
      ierr = ISDestroy(&stratumIS);CHKERRQ(ierr);
    }
    if (num_cs > 0) {
      ierr = ISRestoreIndices(csIS, &csIdx);CHKERRQ(ierr);
      ierr = ISDestroy(&csIS);CHKERRQ(ierr);
    }
    ierr = PetscFree(nodes);CHKERRQ(ierr);
    ierr = PetscSectionSetUp(coordSection);CHKERRQ(ierr);
    if (numNodes > 0) {
      const char  *coordNames[3] = {"x", "y", "z"};
      PetscScalar *coords, *closure;
      PetscReal   *cval;
      PetscInt     hasDof, n = 0;

      /* There can't be more than 24 values in the closure of a point for the coord coordSection */
      ierr = PetscCalloc3(numNodes*3, &coords, dim, &cval, 24, &closure);CHKERRQ(ierr);
      ierr = DMGetCoordinatesLocalNoncollective(dm, &coord);CHKERRQ(ierr);
      ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
      for (p = pStart; p < pEnd; ++p) {
        ierr = PetscSectionGetDof(coordSection, p, &hasDof);
        if (hasDof) {
          PetscInt closureSize = 24, j;

          ierr = DMPlexVecGetClosure(cdm, NULL, coord, p, &closureSize, &closure);CHKERRQ(ierr);
          for (d = 0; d < dim; ++d) {
            cval[d] = 0.0;
            for (j = 0; j < closureSize/dim; j++) cval[d] += closure[j*dim+d];
            coords[d*numNodes+n] = cval[d] * dim / closureSize;
          }
          ++n;
        }
      }
      PetscStackCallStandard(ex_put_coord,(exo->exoid, &coords[0*numNodes], &coords[1*numNodes], &coords[2*numNodes]));
      ierr = PetscFree3(coords, cval, closure);CHKERRQ(ierr);
      PetscStackCallStandard(ex_put_coord_names,(exo->exoid, (char **) coordNames));
    }

    /* --- Node Sets/Vertex Sets --- */
    DMHasLabel(dm, "Vertex Sets", &hasLabel);
    if (hasLabel) {
      PetscInt        i, vs, vsSize;
      const PetscInt *vsIdx, *vertices;
      PetscInt       *nodeList;
      IS              vsIS, stratumIS;
      DMLabel         vsLabel;
      ierr = DMGetLabel(dm, "Vertex Sets", &vsLabel);CHKERRQ(ierr);
      ierr = DMLabelGetValueIS(vsLabel, &vsIS);CHKERRQ(ierr);
      ierr = ISGetIndices(vsIS, &vsIdx);CHKERRQ(ierr);
      for (vs=0; vs<num_vs; ++vs) {
        ierr = DMLabelGetStratumIS(vsLabel, vsIdx[vs], &stratumIS);CHKERRQ(ierr);
        ierr = ISGetIndices(stratumIS, &vertices);CHKERRQ(ierr);
        ierr = ISGetSize(stratumIS, &vsSize);CHKERRQ(ierr);
        ierr = PetscMalloc1(vsSize, &nodeList);
        for (i=0; i<vsSize; ++i) {
          nodeList[i] = vertices[i] - skipCells + 1;
        }
        PetscStackCallStandard(ex_put_set_param,(exo->exoid, EX_NODE_SET, vsIdx[vs], vsSize, 0));
        PetscStackCallStandard(ex_put_set,(exo->exoid, EX_NODE_SET, vsIdx[vs], nodeList, NULL));
        ierr = ISRestoreIndices(stratumIS, &vertices);CHKERRQ(ierr);
        ierr = ISDestroy(&stratumIS);CHKERRQ(ierr);
        ierr = PetscFree(nodeList);CHKERRQ(ierr);
      }
      ierr = ISRestoreIndices(vsIS, &vsIdx);CHKERRQ(ierr);
      ierr = ISDestroy(&vsIS);CHKERRQ(ierr);
    }
    /* --- Side Sets/Face Sets --- */
    ierr = DMHasLabel(dm, "Face Sets", &hasLabel);CHKERRQ(ierr);
    if (hasLabel) {
      PetscInt        i, j, fs, fsSize;
      const PetscInt *fsIdx, *faces;
      IS              fsIS, stratumIS;
      DMLabel         fsLabel;
      PetscInt        numPoints, *points;
      PetscInt        elem_list_size = 0;
      PetscInt       *elem_list, *elem_ind, *side_list;

      ierr = DMGetLabel(dm, "Face Sets", &fsLabel);CHKERRQ(ierr);
      /* Compute size of Node List and Element List */
      ierr = DMLabelGetValueIS(fsLabel, &fsIS);CHKERRQ(ierr);
      ierr = ISGetIndices(fsIS, &fsIdx);CHKERRQ(ierr);
      for (fs=0; fs<num_fs; ++fs) {
        ierr = DMLabelGetStratumIS(fsLabel, fsIdx[fs], &stratumIS);CHKERRQ(ierr);
        ierr = ISGetSize(stratumIS, &fsSize);CHKERRQ(ierr);
        elem_list_size += fsSize;
        ierr = ISDestroy(&stratumIS);CHKERRQ(ierr);
      }
      ierr = PetscMalloc3(num_fs, &elem_ind,
                          elem_list_size, &elem_list,
                          elem_list_size, &side_list);CHKERRQ(ierr);
      elem_ind[0] = 0;
      for (fs=0; fs<num_fs; ++fs) {
        ierr = DMLabelGetStratumIS(fsLabel, fsIdx[fs], &stratumIS);CHKERRQ(ierr);
        ierr = ISGetIndices(stratumIS, &faces);CHKERRQ(ierr);
        ierr = ISGetSize(stratumIS, &fsSize);CHKERRQ(ierr);
        /* Set Parameters */
        PetscStackCallStandard(ex_put_set_param,(exo->exoid, EX_SIDE_SET, fsIdx[fs], fsSize, 0));
        /* Indices */
        if (fs<num_fs-1) {
          elem_ind[fs+1] = elem_ind[fs] + fsSize;
        }

        for (i=0; i<fsSize; ++i) {
          /* Element List */
          points = NULL;
          ierr = DMPlexGetTransitiveClosure(dm, faces[i], PETSC_FALSE, &numPoints, &points);CHKERRQ(ierr);
          elem_list[elem_ind[fs] + i] = points[2] +1;
          ierr = DMPlexRestoreTransitiveClosure(dm, faces[i], PETSC_FALSE, &numPoints, &points);CHKERRQ(ierr);

          /* Side List */
          points = NULL;
          ierr = DMPlexGetTransitiveClosure(dm, elem_list[elem_ind[fs] + i]-1, PETSC_TRUE, &numPoints, &points);CHKERRQ(ierr);
          for (j=1; j<numPoints; ++j) {
            if (points[j*2]==faces[i]) {break;}
          }
          /* Convert HEX sides */
          if (numPoints == 27) {
            if      (j == 1) {j = 5;}
            else if (j == 2) {j = 6;}
            else if (j == 3) {j = 1;}
            else if (j == 4) {j = 3;}
            else if (j == 5) {j = 2;}
            else if (j == 6) {j = 4;}
          }
          /* Convert TET sides */
          if (numPoints == 15) {
            --j;
            if (j == 0) {j = 4;}
          }
          side_list[elem_ind[fs] + i] = j;
          ierr = DMPlexRestoreTransitiveClosure(dm, elem_list[elem_ind[fs] + i]-1, PETSC_TRUE, &numPoints, &points);CHKERRQ(ierr);

        }
        ierr = ISRestoreIndices(stratumIS, &faces);CHKERRQ(ierr);
        ierr = ISDestroy(&stratumIS);CHKERRQ(ierr);
      }
      ierr = ISRestoreIndices(fsIS, &fsIdx);CHKERRQ(ierr);
      ierr = ISDestroy(&fsIS);CHKERRQ(ierr);

      /* Put side sets */
      for (fs=0; fs<num_fs; ++fs) {
        PetscStackCallStandard(ex_put_set,(exo->exoid, EX_SIDE_SET, fsIdx[fs], &elem_list[elem_ind[fs]], &side_list[elem_ind[fs]]));
      }
      ierr = PetscFree3(elem_ind,elem_list,side_list);CHKERRQ(ierr);
    }
    /*
      close the exodus file
    */
    ex_close(exo->exoid);
    exo->exoid = -1;
  }
  ierr = PetscSectionDestroy(&coordSection);CHKERRQ(ierr);

  /*
    reopen the file in parallel
  */
  EXO_mode = EX_WRITE;
#if defined(PETSC_USE_64BIT_INDICES)
  EXO_mode += EX_ALL_INT64_API;
#endif
  CPU_word_size = sizeof(PetscReal);
  IO_word_size  = sizeof(PetscReal);
  exo->exoid = ex_open_par(exo->filename,EXO_mode,&CPU_word_size,&IO_word_size,&EXO_version,comm,MPI_INFO_NULL);
  if (exo->exoid < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_LIB, "ex_open_par failed for %s", exo->filename);
  PetscFunctionReturn(0);
}

/*
  VecView_PlexExodusII_Internal - Write a Vec corresponding in an exodus file

  Collective on v

  Input Parameters:
+ v  - The vector to be written
- viewer - The PetscViewerExodusII viewer associate to an exodus file

  Notes:
  The exodus result variable index is obtained by comparing the Vec name and the
  names of variables declared in the exodus file. For instance for a Vec named "V"
  the location in the exodus file will be the first match of "V", "V_X", "V_XX", "V_1", or "V_11"
  amongst all variables.
  In the event where a nodal and zonal variable both match, the function will return an error instead of
  possibly corrupting the file

  Level: beginner

.seealso: EXOGetVarIndex_Internal(),DMPlexView_ExodusII(),VecView_PlexExodusII()
@*/
PetscErrorCode VecView_PlexExodusII_Internal(Vec v, PetscViewer viewer)
{
  DM                 dm;
  MPI_Comm           comm;
  PetscMPIInt        rank;


  int                exoid,offsetN = 0, offsetZ = 0;
  const char        *vecname;
  PetscInt           step;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) v, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
  ierr = PetscViewerExodusIIGetId(viewer,&exoid);CHKERRQ(ierr);
  ierr = VecGetDM(v, &dm);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject) v, &vecname);CHKERRQ(ierr);

  ierr = DMGetOutputSequenceNumber(dm,&step,NULL);CHKERRQ(ierr);
  ierr = EXOGetVarIndex_Internal(exoid,EX_NODAL,vecname,&offsetN);CHKERRQ(ierr);
  ierr = EXOGetVarIndex_Internal(exoid,EX_ELEM_BLOCK,vecname,&offsetZ);CHKERRQ(ierr);
  if (offsetN <= 0 && offsetZ <= 0) SETERRQ1(comm, PETSC_ERR_FILE_UNEXPECTED, "Found both nodal and zonal variable %s in exodus file. \n", vecname);
  if (offsetN > 0) {
    ierr = VecViewPlex_ExodusII_Nodal_Internal(v,exoid,(int) step+1,offsetN);CHKERRQ(ierr);
  } else if (offsetZ > 0) {
    ierr = VecViewPlex_ExodusII_Zonal_Internal(v,exoid,(int) step+1,offsetZ);CHKERRQ(ierr);
  } else SETERRQ1(comm, PETSC_ERR_FILE_UNEXPECTED, "Could not find nodal or zonal variable %s in exodus file. \n", vecname);
  PetscFunctionReturn(0);
}

/*
  VecLoad_PlexExodusII_Internal - Write a Vec corresponding in an exodus file

  Collective on v

  Input Parameters:
+ v  - The vector to be written
- viewer - The PetscViewerExodusII viewer associate to an exodus file

  Notes:
  The exodus result variable index is obtained by comparing the Vec name and the
  names of variables declared in the exodus file. For instance for a Vec named "V"
  the location in the exodus file will be the first match of "V", "V_X", "V_XX", "V_1", or "V_11"
  amongst all variables.
  In the event where a nodal and zonal variable both match, the function will return an error instead of
  possibly corrupting the file

  Level: beginner

.seealso: EXOGetVarIndex_Internal(),DMPlexView_ExodusII(),VecView_PlexExodusII()
@*/
PetscErrorCode VecLoad_PlexExodusII_Internal(Vec v, PetscViewer viewer)
{
  DM                 dm;
  MPI_Comm           comm;
  PetscMPIInt        rank;


  int                exoid,offsetN = 0, offsetZ = 0;
  const char        *vecname;
  PetscInt           step;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) v, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
  ierr = PetscViewerExodusIIGetId(viewer,&exoid);CHKERRQ(ierr);
  ierr = VecGetDM(v, &dm);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject) v, &vecname);CHKERRQ(ierr);

  ierr = DMGetOutputSequenceNumber(dm,&step,NULL);CHKERRQ(ierr);
  ierr = EXOGetVarIndex_Internal(exoid,EX_NODAL,vecname,&offsetN);CHKERRQ(ierr);
  ierr = EXOGetVarIndex_Internal(exoid,EX_ELEM_BLOCK,vecname,&offsetZ);CHKERRQ(ierr);
  if (offsetN <= 0 && offsetZ <= 0) SETERRQ1(comm, PETSC_ERR_FILE_UNEXPECTED, "Found both nodal and zonal variable %s in exodus file. \n", vecname);
  if (offsetN > 0) {
    ierr = VecLoadPlex_ExodusII_Nodal_Internal(v,exoid,(int) step+1,offsetN);CHKERRQ(ierr);
  } else if (offsetZ > 0) {
    ierr = VecLoadPlex_ExodusII_Zonal_Internal(v,exoid,(int) step+1,offsetZ);CHKERRQ(ierr);
  } else SETERRQ1(comm, PETSC_ERR_FILE_UNEXPECTED, "Could not find nodal or zonal variable %s in exodus file. \n", vecname);
  PetscFunctionReturn(0);
}

/*
  VecViewPlex_ExodusII_Nodal_Internal - Write a Vec corresponding to a nodal field to an exodus file

  Collective on v

  Input Parameters:
+ v  - The vector to be written
. exoid - the exodus id of a file (obtained from ex_open or ex_create for instance)
. step - the time step to write at (exodus steps are numbered starting from 1)
- offset - the location of the variable in the file

  Notes:
  The exodus result nodal variable index is obtained by comparing the Vec name and the
  names of nodal variables declared in the exodus file. For instance for a Vec named "V"
  the location in the exodus file will be the first match of "V", "V_X", "V_XX", "V_1", or "V_11"
  amongst all nodal variables.

  Level: beginner

.seealso: EXOGetVarIndex_Internal(),DMPlexView_ExodusII_Internal(),VecLoadNodal_PlexEXO(),VecViewZonal_PlexEXO(),VecLoadZonal_PlexEXO()
@*/
PetscErrorCode VecViewPlex_ExodusII_Nodal_Internal(Vec v, int exoid, int step, int offset)
{
  MPI_Comm           comm;
  PetscMPIInt        size;
  DM                 dm;
  Vec                vNatural, vComp;
  const PetscScalar *varray;
  PetscInt           xs, xe, bs;
  PetscBool          useNatural;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) v, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRMPI(ierr);
  ierr = VecGetDM(v, &dm);CHKERRQ(ierr);
  ierr = DMGetUseNatural(dm, &useNatural);CHKERRQ(ierr);
  useNatural = useNatural && size > 1 ? PETSC_TRUE : PETSC_FALSE;
  if (useNatural) {
    ierr = DMGetGlobalVector(dm, &vNatural);CHKERRQ(ierr);
    ierr = DMPlexGlobalToNaturalBegin(dm, v, vNatural);CHKERRQ(ierr);
    ierr = DMPlexGlobalToNaturalEnd(dm, v, vNatural);CHKERRQ(ierr);
  } else {
    vNatural = v;
  }

  /* Write local chunk of the result in the exodus file
     exodus stores each component of a vector-valued field as a separate variable.
     We assume that they are stored sequentially */
  ierr = VecGetOwnershipRange(vNatural, &xs, &xe);CHKERRQ(ierr);
  ierr = VecGetBlockSize(vNatural, &bs);CHKERRQ(ierr);
  if (bs == 1) {
    ierr = VecGetArrayRead(vNatural, &varray);CHKERRQ(ierr);
    PetscStackCallStandard(ex_put_partial_var,(exoid, step, EX_NODAL, offset, 1, xs+1, xe-xs, varray));
    ierr = VecRestoreArrayRead(vNatural, &varray);CHKERRQ(ierr);
  } else {
    IS       compIS;
    PetscInt c;

    ierr = ISCreateStride(comm, (xe-xs)/bs, xs, bs, &compIS);CHKERRQ(ierr);
    for (c = 0; c < bs; ++c) {
      ierr = ISStrideSetStride(compIS, (xe-xs)/bs, xs+c, bs);CHKERRQ(ierr);
      ierr = VecGetSubVector(vNatural, compIS, &vComp);CHKERRQ(ierr);
      ierr = VecGetArrayRead(vComp, &varray);CHKERRQ(ierr);
      PetscStackCallStandard(ex_put_partial_var,(exoid, step, EX_NODAL, offset+c, 1, xs/bs+1, (xe-xs)/bs, varray));
      ierr = VecRestoreArrayRead(vComp, &varray);CHKERRQ(ierr);
      ierr = VecRestoreSubVector(vNatural, compIS, &vComp);CHKERRQ(ierr);
    }
    ierr = ISDestroy(&compIS);CHKERRQ(ierr);
  }
  if (useNatural) {ierr = DMRestoreGlobalVector(dm, &vNatural);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*
  VecLoadPlex_ExodusII_Nodal_Internal - Read a Vec corresponding to a nodal field from an exodus file

  Collective on v

  Input Parameters:
+ v  - The vector to be written
. exoid - the exodus id of a file (obtained from ex_open or ex_create for instance)
. step - the time step to read at (exodus steps are numbered starting from 1)
- offset - the location of the variable in the file

  Notes:
  The exodus result nodal variable index is obtained by comparing the Vec name and the
  names of nodal variables declared in the exodus file. For instance for a Vec named "V"
  the location in the exodus file will be the first match of "V", "V_X", "V_XX", "V_1", or "V_11"
  amongst all nodal variables.

  Level: beginner

.seealso: EXOGetVarIndex_Internal(), DMPlexView_ExodusII_Internal(), VecViewPlex_ExodusII_Nodal_Internal(), VecViewZonal_PlexEXO(), VecLoadZonal_PlexEXO()
*/
PetscErrorCode VecLoadPlex_ExodusII_Nodal_Internal(Vec v, int exoid, int step, int offset)
{
  MPI_Comm       comm;
  PetscMPIInt    size;
  DM             dm;
  Vec            vNatural, vComp;
  PetscScalar   *varray;
  PetscInt       xs, xe, bs;
  PetscBool      useNatural;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) v, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRMPI(ierr);
  ierr = VecGetDM(v,&dm);CHKERRQ(ierr);
  ierr = DMGetUseNatural(dm, &useNatural);CHKERRQ(ierr);
  useNatural = useNatural && size > 1 ? PETSC_TRUE : PETSC_FALSE;
  if (useNatural) {ierr = DMGetGlobalVector(dm,&vNatural);CHKERRQ(ierr);}
  else            {vNatural = v;}

  /* Read local chunk from the file */
  ierr = VecGetOwnershipRange(vNatural, &xs, &xe);CHKERRQ(ierr);
  ierr = VecGetBlockSize(vNatural, &bs);CHKERRQ(ierr);
  if (bs == 1) {
    ierr = VecGetArray(vNatural, &varray);CHKERRQ(ierr);
    PetscStackCallStandard(ex_get_partial_var,(exoid, step, EX_NODAL, offset, 1, xs+1, xe-xs, varray));
    ierr = VecRestoreArray(vNatural, &varray);CHKERRQ(ierr);
  } else {
    IS       compIS;
    PetscInt c;

    ierr = ISCreateStride(comm, (xe-xs)/bs, xs, bs, &compIS);CHKERRQ(ierr);
    for (c = 0; c < bs; ++c) {
      ierr = ISStrideSetStride(compIS, (xe-xs)/bs, xs+c, bs);CHKERRQ(ierr);
      ierr = VecGetSubVector(vNatural, compIS, &vComp);CHKERRQ(ierr);
      ierr = VecGetArray(vComp, &varray);CHKERRQ(ierr);
      PetscStackCallStandard(ex_get_partial_var,(exoid, step, EX_NODAL, offset+c, 1, xs/bs+1, (xe-xs)/bs, varray));
      ierr = VecRestoreArray(vComp, &varray);CHKERRQ(ierr);
      ierr = VecRestoreSubVector(vNatural, compIS, &vComp);CHKERRQ(ierr);
    }
    ierr = ISDestroy(&compIS);CHKERRQ(ierr);
  }
  if (useNatural) {
    ierr = DMPlexNaturalToGlobalBegin(dm, vNatural, v);CHKERRQ(ierr);
    ierr = DMPlexNaturalToGlobalEnd(dm, vNatural, v);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dm, &vNatural);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
  VecViewPlex_ExodusII_Zonal_Internal - Write a Vec corresponding to a zonal (cell based) field to an exodus file

  Collective on v

  Input Parameters:
+ v  - The vector to be written
. exoid - the exodus id of a file (obtained from ex_open or ex_create for instance)
. step - the time step to write at (exodus steps are numbered starting from 1)
- offset - the location of the variable in the file

  Notes:
  The exodus result zonal variable index is obtained by comparing the Vec name and the
  names of zonal variables declared in the exodus file. For instance for a Vec named "V"
  the location in the exodus file will be the first match of "V", "V_X", "V_XX", "V_1", or "V_11"
  amongst all zonal variables.

  Level: beginner

.seealso: EXOGetVarIndex_Internal(),DMPlexView_ExodusII_Internal(),VecViewPlex_ExodusII_Nodal_Internal(),VecLoadPlex_ExodusII_Nodal_Internal(),VecLoadPlex_ExodusII_Zonal_Internal()
*/
PetscErrorCode VecViewPlex_ExodusII_Zonal_Internal(Vec v, int exoid, int step, int offset)
{
  MPI_Comm          comm;
  PetscMPIInt       size;
  DM                dm;
  Vec               vNatural, vComp;
  const PetscScalar *varray;
  PetscInt          xs, xe, bs;
  PetscBool         useNatural;
  IS                compIS;
  PetscInt         *csSize, *csID;
  PetscInt          numCS, set, csxs = 0;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)v, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRMPI(ierr);
  ierr = VecGetDM(v, &dm);CHKERRQ(ierr);
  ierr = DMGetUseNatural(dm, &useNatural);CHKERRQ(ierr);
  useNatural = useNatural && size > 1 ? PETSC_TRUE : PETSC_FALSE;
  if (useNatural) {
    ierr = DMGetGlobalVector(dm, &vNatural);CHKERRQ(ierr);
    ierr = DMPlexGlobalToNaturalBegin(dm, v, vNatural);CHKERRQ(ierr);
    ierr = DMPlexGlobalToNaturalEnd(dm, v, vNatural);CHKERRQ(ierr);
  } else {
    vNatural = v;
  }

  /* Write local chunk of the result in the exodus file
     exodus stores each component of a vector-valued field as a separate variable.
     We assume that they are stored sequentially
     Zonal variables are accessed one element block at a time, so we loop through the cell sets,
     but once the vector has been reordered to natural size, we cannot use the label informations
     to figure out what to save where. */
  numCS = ex_inquire_int(exoid, EX_INQ_ELEM_BLK);
  ierr = PetscMalloc2(numCS, &csID, numCS, &csSize);CHKERRQ(ierr);
  PetscStackCallStandard(ex_get_ids,(exoid, EX_ELEM_BLOCK, csID));
  for (set = 0; set < numCS; ++set) {
    ex_block block;

    block.id   = csID[set];
    block.type = EX_ELEM_BLOCK;
    PetscStackCallStandard(ex_get_block_param,(exoid, &block));
    csSize[set] = block.num_entry;
  }
  ierr = VecGetOwnershipRange(vNatural, &xs, &xe);CHKERRQ(ierr);
  ierr = VecGetBlockSize(vNatural, &bs);CHKERRQ(ierr);
  if (bs > 1) {ierr = ISCreateStride(comm, (xe-xs)/bs, xs, bs, &compIS);CHKERRQ(ierr);}
  for (set = 0; set < numCS; set++) {
    PetscInt csLocalSize, c;

    /* range of indices for set setID[set]: csxs:csxs + csSize[set]-1
       local slice of zonal values:         xs/bs,xm/bs-1
       intersection:                        max(xs/bs,csxs),min(xm/bs-1,csxs + csSize[set]-1) */
    csLocalSize = PetscMax(0, PetscMin(xe/bs, csxs+csSize[set]) - PetscMax(xs/bs, csxs));
    if (bs == 1) {
      ierr = VecGetArrayRead(vNatural, &varray);CHKERRQ(ierr);
      PetscStackCallStandard(ex_put_partial_var,(exoid, step, EX_ELEM_BLOCK, offset, csID[set], PetscMax(xs-csxs, 0)+1, csLocalSize, &varray[PetscMax(0, csxs-xs)]));
      ierr = VecRestoreArrayRead(vNatural, &varray);CHKERRQ(ierr);
    } else {
      for (c = 0; c < bs; ++c) {
        ierr = ISStrideSetStride(compIS, (xe-xs)/bs, xs+c, bs);CHKERRQ(ierr);
        ierr = VecGetSubVector(vNatural, compIS, &vComp);CHKERRQ(ierr);
        ierr = VecGetArrayRead(vComp, &varray);CHKERRQ(ierr);
        PetscStackCallStandard(ex_put_partial_var,(exoid, step, EX_ELEM_BLOCK, offset+c, csID[set], PetscMax(xs/bs-csxs, 0)+1, csLocalSize, &varray[PetscMax(0, csxs-xs/bs)]));
        ierr = VecRestoreArrayRead(vComp, &varray);CHKERRQ(ierr);
        ierr = VecRestoreSubVector(vNatural, compIS, &vComp);CHKERRQ(ierr);
      }
    }
    csxs += csSize[set];
  }
  ierr = PetscFree2(csID, csSize);CHKERRQ(ierr);
  if (bs > 1) {ierr = ISDestroy(&compIS);CHKERRQ(ierr);}
  if (useNatural) {ierr = DMRestoreGlobalVector(dm,&vNatural);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*
  VecLoadPlex_ExodusII_Zonal_Internal - Read a Vec corresponding to a zonal (cell based) field from an exodus file

  Collective on v

  Input Parameters:
+ v  - The vector to be written
. exoid - the exodus id of a file (obtained from ex_open or ex_create for instance)
. step - the time step to read at (exodus steps are numbered starting from 1)
- offset - the location of the variable in the file

  Notes:
  The exodus result zonal variable index is obtained by comparing the Vec name and the
  names of zonal variables declared in the exodus file. For instance for a Vec named "V"
  the location in the exodus file will be the first match of "V", "V_X", "V_XX", "V_1", or "V_11"
  amongst all zonal variables.

  Level: beginner

.seealso: EXOGetVarIndex_Internal(), DMPlexView_ExodusII_Internal(), VecViewPlex_ExodusII_Nodal_Internal(), VecLoadPlex_ExodusII_Nodal_Internal(), VecLoadPlex_ExodusII_Zonal_Internal()
*/
PetscErrorCode VecLoadPlex_ExodusII_Zonal_Internal(Vec v, int exoid, int step, int offset)
{
  MPI_Comm          comm;
  PetscMPIInt       size;
  DM                dm;
  Vec               vNatural, vComp;
  PetscScalar      *varray;
  PetscInt          xs, xe, bs;
  PetscBool         useNatural;
  IS                compIS;
  PetscInt         *csSize, *csID;
  PetscInt          numCS, set, csxs = 0;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)v,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRMPI(ierr);
  ierr = VecGetDM(v, &dm);CHKERRQ(ierr);
  ierr = DMGetUseNatural(dm, &useNatural);CHKERRQ(ierr);
  useNatural = useNatural && size > 1 ? PETSC_TRUE : PETSC_FALSE;
  if (useNatural) {ierr = DMGetGlobalVector(dm,&vNatural);CHKERRQ(ierr);}
  else            {vNatural = v;}

  /* Read local chunk of the result in the exodus file
     exodus stores each component of a vector-valued field as a separate variable.
     We assume that they are stored sequentially
     Zonal variables are accessed one element block at a time, so we loop through the cell sets,
     but once the vector has been reordered to natural size, we cannot use the label informations
     to figure out what to save where. */
  numCS = ex_inquire_int(exoid, EX_INQ_ELEM_BLK);
  ierr = PetscMalloc2(numCS, &csID, numCS, &csSize);CHKERRQ(ierr);
  PetscStackCallStandard(ex_get_ids,(exoid, EX_ELEM_BLOCK, csID));
  for (set = 0; set < numCS; ++set) {
    ex_block block;

    block.id   = csID[set];
    block.type = EX_ELEM_BLOCK;
    PetscStackCallStandard(ex_get_block_param,(exoid, &block));
    csSize[set] = block.num_entry;
  }
  ierr = VecGetOwnershipRange(vNatural, &xs, &xe);CHKERRQ(ierr);
  ierr = VecGetBlockSize(vNatural, &bs);CHKERRQ(ierr);
  if (bs > 1) {ierr = ISCreateStride(comm, (xe-xs)/bs, xs, bs, &compIS);CHKERRQ(ierr);}
  for (set = 0; set < numCS; ++set) {
    PetscInt csLocalSize, c;

    /* range of indices for set setID[set]: csxs:csxs + csSize[set]-1
       local slice of zonal values:         xs/bs,xm/bs-1
       intersection:                        max(xs/bs,csxs),min(xm/bs-1,csxs + csSize[set]-1) */
    csLocalSize = PetscMax(0, PetscMin(xe/bs, csxs+csSize[set]) - PetscMax(xs/bs, csxs));
    if (bs == 1) {
      ierr = VecGetArray(vNatural, &varray);CHKERRQ(ierr);
      PetscStackCallStandard(ex_get_partial_var,(exoid, step, EX_ELEM_BLOCK, offset, csID[set], PetscMax(xs-csxs, 0)+1, csLocalSize, &varray[PetscMax(0, csxs-xs)]));
      ierr = VecRestoreArray(vNatural, &varray);CHKERRQ(ierr);
    } else {
      for (c = 0; c < bs; ++c) {
        ierr = ISStrideSetStride(compIS, (xe-xs)/bs, xs+c, bs);CHKERRQ(ierr);
        ierr = VecGetSubVector(vNatural, compIS, &vComp);CHKERRQ(ierr);
        ierr = VecGetArray(vComp, &varray);CHKERRQ(ierr);
        PetscStackCallStandard(ex_get_partial_var,(exoid, step, EX_ELEM_BLOCK, offset+c, csID[set], PetscMax(xs/bs-csxs, 0)+1, csLocalSize, &varray[PetscMax(0, csxs-xs/bs)]));
        ierr = VecRestoreArray(vComp, &varray);CHKERRQ(ierr);
        ierr = VecRestoreSubVector(vNatural, compIS, &vComp);CHKERRQ(ierr);
      }
    }
    csxs += csSize[set];
  }
  ierr = PetscFree2(csID, csSize);CHKERRQ(ierr);
  if (bs > 1) {ierr = ISDestroy(&compIS);CHKERRQ(ierr);}
  if (useNatural) {
    ierr = DMPlexNaturalToGlobalBegin(dm, vNatural, v);CHKERRQ(ierr);
    ierr = DMPlexNaturalToGlobalEnd(dm, vNatural, v);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dm, &vNatural);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
#endif

/*@
  PetscViewerExodusIIGetId - Get the file id of the ExodusII file

  Logically Collective on PetscViewer

  Input Parameter:
.  viewer - the PetscViewer

  Output Parameter:
-  exoid - The ExodusII file id

  Level: intermediate

.seealso: PetscViewerFileSetMode(), PetscViewerCreate(), PetscViewerSetType(), PetscViewerBinaryOpen()
@*/
PetscErrorCode PetscViewerExodusIIGetId(PetscViewer viewer, int *exoid)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  ierr = PetscTryMethod(viewer, "PetscViewerGetId_C",(PetscViewer,int*),(viewer,exoid));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



/*@
   PetscViewerExodusIISetOrder - Set the elements order in the exodusII file.

   Collective

   Input Parameters:
+  viewer - the viewer
-  order - elements order

   Output Parameter:

   Level: beginner

   Note:


.seealso: PetscViewerExodusIIGetId(), PetscViewerExodusIIGetOrder(), PetscViewerExodusIISetOrder()
@*/
PetscErrorCode PetscViewerExodusIISetOrder(PetscViewer viewer, PetscInt order)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  ierr = PetscTryMethod(viewer, "PetscViewerSetOrder_C",(PetscViewer,PetscInt),(viewer,order));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   PetscViewerExodusIIGetOrder - Get the elements order in the exodusII file.

   Collective

   Input Parameters:
+  viewer - the viewer
-  order - elements order

   Output Parameter:

   Level: beginner

   Note:


.seealso: PetscViewerExodusIIGetId(), PetscViewerExodusIIGetOrder(), PetscViewerExodusIISetOrder()
@*/
PetscErrorCode PetscViewerExodusIIGetOrder(PetscViewer viewer, PetscInt *order)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  ierr = PetscTryMethod(viewer, "PetscViewerGetOrder_C",(PetscViewer,PetscInt*),(viewer,order));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscViewerExodusIIOpen - Opens a file for ExodusII input/output.

   Collective

   Input Parameters:
+  comm - MPI communicator
.  name - name of file
-  type - type of file
$    FILE_MODE_WRITE - create new file for binary output
$    FILE_MODE_READ - open existing file for binary input
$    FILE_MODE_APPEND - open existing file for binary output

   Output Parameter:
.  exo - PetscViewer for Exodus II input/output to use with the specified file

   Level: beginner

   Note:
   This PetscViewer should be destroyed with PetscViewerDestroy().


.seealso: PetscViewerPushFormat(), PetscViewerDestroy(),
          DMLoad(), PetscFileMode, PetscViewer, PetscViewerSetType(), PetscViewerFileSetMode(), PetscViewerFileSetName()
@*/
PetscErrorCode PetscViewerExodusIIOpen(MPI_Comm comm, const char name[], PetscFileMode type, PetscViewer *exo)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerCreate(comm, exo);CHKERRQ(ierr);
  ierr = PetscViewerSetType(*exo, PETSCVIEWEREXODUSII);CHKERRQ(ierr);
  ierr = PetscViewerFileSetMode(*exo, type);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(*exo, name);CHKERRQ(ierr);
  ierr = PetscViewerSetFromOptions(*exo);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMPlexCreateExodusFromFile - Create a DMPlex mesh from an ExodusII file.

  Collective

  Input Parameters:
+ comm  - The MPI communicator
. filename - The name of the ExodusII file
- interpolate - Create faces and edges in the mesh

  Output Parameter:
. dm  - The DM object representing the mesh

  Level: beginner

.seealso: DMPLEX, DMCreate(), DMPlexCreateExodus()
@*/
PetscErrorCode DMPlexCreateExodusFromFile(MPI_Comm comm, const char filename[], PetscBool interpolate, DM *dm)
{
  PetscMPIInt    rank;
  PetscErrorCode ierr;
#if defined(PETSC_HAVE_EXODUSII)
  int   CPU_word_size = sizeof(PetscReal), IO_word_size = 0, exoid = -1;
  float version;
#endif

  PetscFunctionBegin;
  PetscValidCharPointer(filename, 2);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
#if defined(PETSC_HAVE_EXODUSII)
  if (!rank) {
    exoid = ex_open(filename, EX_READ, &CPU_word_size, &IO_word_size, &version);
    if (exoid <= 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_LIB, "ex_open(\"%s\",...) did not return a valid file ID", filename);
  }
  ierr = DMPlexCreateExodus(comm, exoid, interpolate, dm);CHKERRQ(ierr);
  if (!rank) {PetscStackCallStandard(ex_close,(exoid));}
  PetscFunctionReturn(0);
#else
  SETERRQ(comm, PETSC_ERR_SUP, "This method requires ExodusII support. Reconfigure using --download-exodusii");
#endif
}

#if defined(PETSC_HAVE_EXODUSII)
static PetscErrorCode ExodusGetCellType_Internal(const char *elem_type, DMPolytopeType *ct)
{
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *ct = DM_POLYTOPE_UNKNOWN;
  ierr = PetscStrcmp(elem_type, "TRI", &flg);CHKERRQ(ierr);
  if (flg) {*ct = DM_POLYTOPE_TRIANGLE; goto done;}
  ierr = PetscStrcmp(elem_type, "TRI3", &flg);CHKERRQ(ierr);
  if (flg) {*ct = DM_POLYTOPE_TRIANGLE; goto done;}
  ierr = PetscStrcmp(elem_type, "QUAD", &flg);CHKERRQ(ierr);
  if (flg) {*ct = DM_POLYTOPE_QUADRILATERAL; goto done;}
  ierr = PetscStrcmp(elem_type, "QUAD4", &flg);CHKERRQ(ierr);
  if (flg) {*ct = DM_POLYTOPE_QUADRILATERAL; goto done;}
  ierr = PetscStrcmp(elem_type, "SHELL4", &flg);CHKERRQ(ierr);
  if (flg) {*ct = DM_POLYTOPE_QUADRILATERAL; goto done;}
  ierr = PetscStrcmp(elem_type, "TETRA", &flg);CHKERRQ(ierr);
  if (flg) {*ct = DM_POLYTOPE_TETRAHEDRON; goto done;}
  ierr = PetscStrcmp(elem_type, "TET4", &flg);CHKERRQ(ierr);
  if (flg) {*ct = DM_POLYTOPE_TETRAHEDRON; goto done;}
  ierr = PetscStrcmp(elem_type, "WEDGE", &flg);CHKERRQ(ierr);
  if (flg) {*ct = DM_POLYTOPE_TRI_PRISM; goto done;}
  ierr = PetscStrcmp(elem_type, "HEX", &flg);CHKERRQ(ierr);
  if (flg) {*ct = DM_POLYTOPE_HEXAHEDRON; goto done;}
  ierr = PetscStrcmp(elem_type, "HEX8", &flg);CHKERRQ(ierr);
  if (flg) {*ct = DM_POLYTOPE_HEXAHEDRON; goto done;}
  ierr = PetscStrcmp(elem_type, "HEXAHEDRON", &flg);CHKERRQ(ierr);
  if (flg) {*ct = DM_POLYTOPE_HEXAHEDRON; goto done;}
  SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unrecognized element type %s", elem_type);
  done:
  PetscFunctionReturn(0);
}
#endif

/*@
  DMPlexCreateExodus - Create a DMPlex mesh from an ExodusII file ID.

  Collective

  Input Parameters:
+ comm  - The MPI communicator
. exoid - The ExodusII id associated with a exodus file and obtained using ex_open
- interpolate - Create faces and edges in the mesh

  Output Parameter:
. dm  - The DM object representing the mesh

  Level: beginner

.seealso: DMPLEX, DMCreate()
@*/
PetscErrorCode DMPlexCreateExodus(MPI_Comm comm, PetscInt exoid, PetscBool interpolate, DM *dm)
{
#if defined(PETSC_HAVE_EXODUSII)
  PetscMPIInt    num_proc, rank;
  PetscSection   coordSection;
  Vec            coordinates;
  PetscScalar    *coords;
  PetscInt       coordSize, v;
  PetscErrorCode ierr;
  /* Read from ex_get_init() */
  char title[PETSC_MAX_PATH_LEN+1];
  int  dim    = 0, dimEmbed = 0, numVertices = 0, numCells = 0, numHybridCells = 0;
  int  num_cs = 0, num_vs = 0, num_fs = 0;
#endif

  PetscFunctionBegin;
#if defined(PETSC_HAVE_EXODUSII)
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(comm, &num_proc);CHKERRMPI(ierr);
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
  /* Open EXODUS II file and read basic informations on rank 0, then broadcast to all processors */
  if (!rank) {
    ierr = PetscMemzero(title,PETSC_MAX_PATH_LEN+1);CHKERRQ(ierr);
    PetscStackCallStandard(ex_get_init,(exoid, title, &dimEmbed, &numVertices, &numCells, &num_cs, &num_vs, &num_fs));
    if (!num_cs) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Exodus file does not contain any cell set\n");
  }
  ierr = MPI_Bcast(title, PETSC_MAX_PATH_LEN+1, MPI_CHAR, 0, comm);CHKERRMPI(ierr);
  ierr = MPI_Bcast(&dim, 1, MPI_INT, 0, comm);CHKERRMPI(ierr);
  ierr = PetscObjectSetName((PetscObject) *dm, title);CHKERRQ(ierr);
  ierr = DMPlexSetChart(*dm, 0, numCells+numVertices);CHKERRQ(ierr);
  /*   We do not want this label automatically computed, instead we compute it here */
  ierr = DMCreateLabel(*dm, "celltype");CHKERRQ(ierr);

  /* Read cell sets information */
  if (!rank) {
    PetscInt *cone;
    int      c, cs, ncs, c_loc, v, v_loc;
    /* Read from ex_get_elem_blk_ids() */
    int *cs_id, *cs_order;
    /* Read from ex_get_elem_block() */
    char buffer[PETSC_MAX_PATH_LEN+1];
    int  num_cell_in_set, num_vertex_per_cell, num_hybrid, num_attr;
    /* Read from ex_get_elem_conn() */
    int *cs_connect;

    /* Get cell sets IDs */
    ierr = PetscMalloc2(num_cs, &cs_id, num_cs, &cs_order);CHKERRQ(ierr);
    PetscStackCallStandard(ex_get_ids,(exoid, EX_ELEM_BLOCK, cs_id));
    /* Read the cell set connectivity table and build mesh topology
       EXO standard requires that cells in cell sets be numbered sequentially and be pairwise disjoint. */
    /* Check for a hybrid mesh */
    for (cs = 0, num_hybrid = 0; cs < num_cs; ++cs) {
      DMPolytopeType ct;
      char           elem_type[PETSC_MAX_PATH_LEN];

      ierr = PetscArrayzero(elem_type, sizeof(elem_type));CHKERRQ(ierr);
      PetscStackCallStandard(ex_get_elem_type,(exoid, cs_id[cs], elem_type));
      ierr = ExodusGetCellType_Internal(elem_type, &ct);CHKERRQ(ierr);
      dim  = PetscMax(dim, DMPolytopeTypeGetDim(ct));
      PetscStackCallStandard(ex_get_block,(exoid, EX_ELEM_BLOCK, cs_id[cs], buffer, &num_cell_in_set,&num_vertex_per_cell, 0, 0, &num_attr));
      switch (ct) {
        case DM_POLYTOPE_TRI_PRISM:
          cs_order[cs] = cs;
          numHybridCells += num_cell_in_set;
          ++num_hybrid;
          break;
        default:
          for (c = cs; c > cs-num_hybrid; --c) cs_order[c] = cs_order[c-1];
          cs_order[cs-num_hybrid] = cs;
      }
    }
    /* First set sizes */
    for (ncs = 0, c = 0; ncs < num_cs; ++ncs) {
      DMPolytopeType ct;
      char           elem_type[PETSC_MAX_PATH_LEN];
      const PetscInt cs = cs_order[ncs];

      ierr = PetscArrayzero(elem_type, sizeof(elem_type));CHKERRQ(ierr);
      PetscStackCallStandard(ex_get_elem_type,(exoid, cs_id[cs], elem_type));
      ierr = ExodusGetCellType_Internal(elem_type, &ct);CHKERRQ(ierr);
      PetscStackCallStandard(ex_get_block,(exoid, EX_ELEM_BLOCK, cs_id[cs], buffer, &num_cell_in_set,&num_vertex_per_cell, 0, 0, &num_attr));
      for (c_loc = 0; c_loc < num_cell_in_set; ++c_loc, ++c) {
        ierr = DMPlexSetConeSize(*dm, c, num_vertex_per_cell);CHKERRQ(ierr);
        ierr = DMPlexSetCellType(*dm, c, ct);CHKERRQ(ierr);
      }
    }
    for (v = numCells; v < numCells+numVertices; ++v) {ierr = DMPlexSetCellType(*dm, v, DM_POLYTOPE_POINT);CHKERRQ(ierr);}
    ierr = DMSetUp(*dm);CHKERRQ(ierr);
    for (ncs = 0, c = 0; ncs < num_cs; ++ncs) {
      const PetscInt cs = cs_order[ncs];
      PetscStackCallStandard(ex_get_block,(exoid, EX_ELEM_BLOCK, cs_id[cs], buffer, &num_cell_in_set, &num_vertex_per_cell, 0, 0, &num_attr));
      ierr = PetscMalloc2(num_vertex_per_cell*num_cell_in_set,&cs_connect,num_vertex_per_cell,&cone);CHKERRQ(ierr);
      PetscStackCallStandard(ex_get_conn,(exoid, EX_ELEM_BLOCK, cs_id[cs], cs_connect,NULL,NULL));
      /* EXO uses Fortran-based indexing, DMPlex uses C-style and numbers cell first then vertices. */
      for (c_loc = 0, v = 0; c_loc < num_cell_in_set; ++c_loc, ++c) {
        DMPolytopeType ct;

        for (v_loc = 0; v_loc < num_vertex_per_cell; ++v_loc, ++v) {
          cone[v_loc] = cs_connect[v]+numCells-1;
        }
        ierr = DMPlexGetCellType(*dm, c, &ct);CHKERRQ(ierr);
        ierr = DMPlexInvertCell(ct, cone);CHKERRQ(ierr);
        ierr = DMPlexSetCone(*dm, c, cone);CHKERRQ(ierr);
        ierr = DMSetLabelValue(*dm, "Cell Sets", c, cs_id[cs]);CHKERRQ(ierr);
      }
      ierr = PetscFree2(cs_connect,cone);CHKERRQ(ierr);
    }
    ierr = PetscFree2(cs_id, cs_order);CHKERRQ(ierr);
  }
  {
    PetscInt ints[] = {dim, dimEmbed};

    ierr = MPI_Bcast(ints, 2, MPIU_INT, 0, comm);CHKERRMPI(ierr);
    ierr = DMSetDimension(*dm, ints[0]);CHKERRQ(ierr);
    ierr = DMSetCoordinateDim(*dm, ints[1]);CHKERRQ(ierr);
    dim      = ints[0];
    dimEmbed = ints[1];
  }
  ierr = DMPlexSymmetrize(*dm);CHKERRQ(ierr);
  ierr = DMPlexStratify(*dm);CHKERRQ(ierr);
  if (interpolate) {
    DM idm;

    ierr = DMPlexInterpolate(*dm, &idm);CHKERRQ(ierr);
    ierr = DMDestroy(dm);CHKERRQ(ierr);
    *dm  = idm;
  }

  /* Create vertex set label */
  if (!rank && (num_vs > 0)) {
    int vs, v;
    /* Read from ex_get_node_set_ids() */
    int *vs_id;
    /* Read from ex_get_node_set_param() */
    int num_vertex_in_set;
    /* Read from ex_get_node_set() */
    int *vs_vertex_list;

    /* Get vertex set ids */
    ierr = PetscMalloc1(num_vs, &vs_id);CHKERRQ(ierr);
    PetscStackCallStandard(ex_get_ids,(exoid, EX_NODE_SET, vs_id));
    for (vs = 0; vs < num_vs; ++vs) {
      PetscStackCallStandard(ex_get_set_param,(exoid, EX_NODE_SET, vs_id[vs], &num_vertex_in_set, NULL));
      ierr = PetscMalloc1(num_vertex_in_set, &vs_vertex_list);CHKERRQ(ierr);
      PetscStackCallStandard(ex_get_set,(exoid, EX_NODE_SET, vs_id[vs], vs_vertex_list, NULL));
      for (v = 0; v < num_vertex_in_set; ++v) {
        ierr = DMSetLabelValue(*dm, "Vertex Sets", vs_vertex_list[v]+numCells-1, vs_id[vs]);CHKERRQ(ierr);
      }
      ierr = PetscFree(vs_vertex_list);CHKERRQ(ierr);
    }
    ierr = PetscFree(vs_id);CHKERRQ(ierr);
  }
  /* Read coordinates */
  ierr = DMGetCoordinateSection(*dm, &coordSection);CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(coordSection, 1);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(coordSection, 0, dimEmbed);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(coordSection, numCells, numCells + numVertices);CHKERRQ(ierr);
  for (v = numCells; v < numCells+numVertices; ++v) {
    ierr = PetscSectionSetDof(coordSection, v, dimEmbed);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldDof(coordSection, v, 0, dimEmbed);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(coordSection);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(coordSection, &coordSize);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_SELF, &coordinates);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) coordinates, "coordinates");CHKERRQ(ierr);
  ierr = VecSetSizes(coordinates, coordSize, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetBlockSize(coordinates, dimEmbed);CHKERRQ(ierr);
  ierr = VecSetType(coordinates,VECSTANDARD);CHKERRQ(ierr);
  ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
  if (!rank) {
    PetscReal *x, *y, *z;

    ierr = PetscMalloc3(numVertices,&x,numVertices,&y,numVertices,&z);CHKERRQ(ierr);
    PetscStackCallStandard(ex_get_coord,(exoid, x, y, z));
    if (dimEmbed > 0) {
      for (v = 0; v < numVertices; ++v) coords[v*dimEmbed+0] = x[v];
    }
    if (dimEmbed > 1) {
      for (v = 0; v < numVertices; ++v) coords[v*dimEmbed+1] = y[v];
    }
    if (dimEmbed > 2) {
      for (v = 0; v < numVertices; ++v) coords[v*dimEmbed+2] = z[v];
    }
    ierr = PetscFree3(x,y,z);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(*dm, coordinates);CHKERRQ(ierr);
  ierr = VecDestroy(&coordinates);CHKERRQ(ierr);

  /* Create side set label */
  if (!rank && interpolate && (num_fs > 0)) {
    int fs, f, voff;
    /* Read from ex_get_side_set_ids() */
    int *fs_id;
    /* Read from ex_get_side_set_param() */
    int num_side_in_set;
    /* Read from ex_get_side_set_node_list() */
    int *fs_vertex_count_list, *fs_vertex_list;
    /* Read side set labels */
    char fs_name[MAX_STR_LENGTH+1];

    /* Get side set ids */
    ierr = PetscMalloc1(num_fs, &fs_id);CHKERRQ(ierr);
    PetscStackCallStandard(ex_get_ids,(exoid, EX_SIDE_SET, fs_id));
    for (fs = 0; fs < num_fs; ++fs) {
      PetscStackCallStandard(ex_get_set_param,(exoid, EX_SIDE_SET, fs_id[fs], &num_side_in_set, NULL));
      ierr = PetscMalloc2(num_side_in_set,&fs_vertex_count_list,num_side_in_set*4,&fs_vertex_list);CHKERRQ(ierr);
      PetscStackCallStandard(ex_get_side_set_node_list,(exoid, fs_id[fs], fs_vertex_count_list, fs_vertex_list));
      /* Get the specific name associated with this side set ID. */
      int fs_name_err = ex_get_name(exoid, EX_SIDE_SET, fs_id[fs], fs_name);
      for (f = 0, voff = 0; f < num_side_in_set; ++f) {
        const PetscInt *faces   = NULL;
        PetscInt       faceSize = fs_vertex_count_list[f], numFaces;
        PetscInt       faceVertices[4], v;

        if (faceSize > 4) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "ExodusII side cannot have %d > 4 vertices", faceSize);
        for (v = 0; v < faceSize; ++v, ++voff) {
          faceVertices[v] = fs_vertex_list[voff]+numCells-1;
        }
        ierr = DMPlexGetFullJoin(*dm, faceSize, faceVertices, &numFaces, &faces);CHKERRQ(ierr);
        if (numFaces != 1) SETERRQ3(comm, PETSC_ERR_ARG_WRONG, "Invalid ExodusII side %d in set %d maps to %d faces", f, fs, numFaces);
        ierr = DMSetLabelValue(*dm, "Face Sets", faces[0], fs_id[fs]);CHKERRQ(ierr);
        /* Only add the label if one has been detected for this side set. */
        if (!fs_name_err) {
          ierr = DMSetLabelValue(*dm, fs_name, faces[0], fs_id[fs]);CHKERRQ(ierr);
        }
        ierr = DMPlexRestoreJoin(*dm, faceSize, faceVertices, &numFaces, &faces);CHKERRQ(ierr);
      }
      ierr = PetscFree2(fs_vertex_count_list,fs_vertex_list);CHKERRQ(ierr);
    }
    ierr = PetscFree(fs_id);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
#else
  SETERRQ(comm, PETSC_ERR_SUP, "This method requires ExodusII support. Reconfigure using --download-exodusii");
#endif
}
