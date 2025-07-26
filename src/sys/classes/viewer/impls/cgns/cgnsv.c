#include <petsc/private/viewercgnsimpl.h> /*I "petscviewer.h" I*/
#include <petsc/private/dmpleximpl.h>     /*I   "petscdmplex.h"   I*/
#if defined(PETSC_HDF5_HAVE_PARALLEL)
  #include <pcgnslib.h>
#else
  #include <cgnslib.h>
#endif
#include <cgns_io.h>
#include <ctype.h>

PetscLogEvent PETSC_VIEWER_CGNS_Open, PETSC_VIEWER_CGNS_Close, PETSC_VIEWER_CGNS_ReadMeta, PETSC_VIEWER_CGNS_WriteMeta, PETSC_VIEWER_CGNS_ReadData, PETSC_VIEWER_CGNS_WriteData;

PetscErrorCode PetscViewerCGNSRegisterLogEvents_Internal()
{
  static PetscBool is_initialized = PETSC_FALSE;

  PetscFunctionBeginUser;
  if (is_initialized) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscLogEventRegister("CGNSOpen", PETSC_VIEWER_CLASSID, &PETSC_VIEWER_CGNS_Open));
  PetscCall(PetscLogEventRegister("CGNSClose", PETSC_VIEWER_CLASSID, &PETSC_VIEWER_CGNS_Close));
  PetscCall(PetscLogEventRegister("CGNSReadMeta", PETSC_VIEWER_CLASSID, &PETSC_VIEWER_CGNS_ReadMeta));
  PetscCall(PetscLogEventRegister("CGNSReadData", PETSC_VIEWER_CLASSID, &PETSC_VIEWER_CGNS_ReadData));
  PetscCall(PetscLogEventRegister("CGNSWriteMeta", PETSC_VIEWER_CLASSID, &PETSC_VIEWER_CGNS_WriteMeta));
  PetscCall(PetscLogEventRegister("CGNSWriteData", PETSC_VIEWER_CLASSID, &PETSC_VIEWER_CGNS_WriteData));
  is_initialized = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerSetFromOptions_CGNS(PetscViewer v, PetscOptionItems PetscOptionsObject)
{
  PetscViewer_CGNS *cgv = (PetscViewer_CGNS *)v->data;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "CGNS Viewer Options");
  PetscCall(PetscOptionsInt("-viewer_cgns_batch_size", "Max number of steps to store in single file when using a template cgns:name-\%d.cgns", "", cgv->batch_size, &cgv->batch_size, NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerView_CGNS(PetscViewer v, PetscViewer viewer)
{
  PetscViewer_CGNS *cgv = (PetscViewer_CGNS *)v->data;

  PetscFunctionBegin;
  if (cgv->filename) PetscCall(PetscViewerASCIIPrintf(viewer, "Filename: %s\n", cgv->filename));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerFileClose_CGNS(PetscViewer viewer)
{
  PetscViewer_CGNS *cgv = (PetscViewer_CGNS *)viewer->data;

  PetscFunctionBegin;
  if (cgv->output_times) {
    PetscCount size, width = 32, *steps;
    char      *solnames;
    PetscReal *times;
    cgsize_t   num_times;
    PetscCall(PetscSegBufferGetSize(cgv->output_times, &size));
    PetscCall(PetscSegBufferExtractInPlace(cgv->output_times, &times));
    PetscCall(PetscSegBufferExtractInPlace(cgv->output_steps, &steps));
    num_times = size;
    PetscCallCGNSWrite(cg_biter_write(cgv->file_num, cgv->base, "TimeIterValues", num_times), viewer, 0);
    PetscCallCGNS(cg_goto(cgv->file_num, cgv->base, "BaseIterativeData_t", 1, NULL));
    PetscCallCGNSWrite(cg_array_write("TimeValues", CGNS_ENUMV(RealDouble), 1, &num_times, times), viewer, 0);
    { // Cast output_steps to long for writing into file
      int *steps_int;
      PetscCall(PetscMalloc1(size, &steps_int));
      for (PetscCount i = 0; i < size; i++) PetscCall(PetscCIntCast(steps[i], &steps_int[i]));
      PetscCallCGNSWrite(cg_array_write("IterationValues", CGNS_ENUMV(Integer), 1, &num_times, steps_int), viewer, 0);
      PetscCall(PetscFree(steps_int));
    }
    PetscCall(PetscSegBufferDestroy(&cgv->output_times));
    PetscCallCGNSWrite(cg_ziter_write(cgv->file_num, cgv->base, cgv->zone, "ZoneIterativeData"), viewer, 0);
    PetscCallCGNS(cg_goto(cgv->file_num, cgv->base, "Zone_t", cgv->zone, "ZoneIterativeData_t", 1, NULL));
    PetscCall(PetscMalloc(size * width + 1, &solnames));
    for (PetscCount i = 0; i < size; i++) PetscCall(PetscSNPrintf(&solnames[i * width], width + 1, "FlowSolution%-20zu", (size_t)steps[i]));
    PetscCall(PetscSegBufferDestroy(&cgv->output_steps));
    cgsize_t shape[2] = {(cgsize_t)width, (cgsize_t)size};
    PetscCallCGNSWrite(cg_array_write("FlowSolutionPointers", CGNS_ENUMV(Character), 2, shape, solnames), viewer, 0);
    // The VTK reader looks for names like FlowSolution*Pointers.
    for (PetscCount i = 0; i < size; i++) PetscCall(PetscSNPrintf(&solnames[i * width], width + 1, "%-32s", "CellInfo"));
    PetscCallCGNSWrite(cg_array_write("FlowSolutionCellInfoPointers", CGNS_ENUMV(Character), 2, shape, solnames), viewer, 0);
    PetscCall(PetscFree(solnames));

    PetscCallCGNSWrite(cg_simulation_type_write(cgv->file_num, cgv->base, CGNS_ENUMV(TimeAccurate)), viewer, 0);
  }
  PetscCall(PetscFree(cgv->filename));
#if defined(PETSC_HDF5_HAVE_PARALLEL)
  if (cgv->file_num) PetscCallCGNSClose(cgp_close(cgv->file_num), viewer, 0);
#else
  if (cgv->file_num) PetscCallCGNSClose(cg_close(cgv->file_num), viewer, 0);
#endif
  cgv->file_num = 0;
  PetscCall(PetscFree(cgv->node_l2g));
  PetscCall(PetscFree(cgv->nodal_field));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscViewerCGNSFileOpen_Internal(PetscViewer viewer, PetscInt sequence_number)
{
  PetscViewer_CGNS *cgv          = (PetscViewer_CGNS *)viewer->data;
  int               cg_file_mode = -1;

  PetscFunctionBegin;
  PetscCheck((cgv->filename == NULL) ^ (sequence_number < 0), PetscObjectComm((PetscObject)viewer), PETSC_ERR_ARG_INCOMP, "Expect either a template filename or non-negative sequence number");
  if (!cgv->filename) {
    char filename_numbered[PETSC_MAX_PATH_LEN];
    // Cast sequence_number so %d can be used also when PetscInt is 64-bit. We could upgrade the format string if users
    // run more than 2B time steps.
    PetscCall(PetscSNPrintf(filename_numbered, sizeof filename_numbered, cgv->filename_template, (int)sequence_number));
    PetscCall(PetscStrallocpy(filename_numbered, &cgv->filename));
  }
  switch (cgv->btype) {
  case FILE_MODE_READ:
    cg_file_mode = CG_MODE_READ;
    break;
  case FILE_MODE_WRITE:
    cg_file_mode = CG_MODE_WRITE;
    break;
  case FILE_MODE_UNDEFINED:
    SETERRQ(PetscObjectComm((PetscObject)viewer), PETSC_ERR_ORDER, "Must call PetscViewerFileSetMode() before PetscViewerFileSetName()");
  default:
    SETERRQ(PetscObjectComm((PetscObject)viewer), PETSC_ERR_SUP, "Unsupported file mode %s", PetscFileModes[cgv->btype]);
  }
#if defined(PETSC_HDF5_HAVE_PARALLEL)
  PetscCallCGNS(cgp_mpi_comm(PetscObjectComm((PetscObject)viewer)));
  PetscCallCGNSOpen(cgp_open(cgv->filename, cg_file_mode, &cgv->file_num), viewer, 0);
#else
  PetscCallCGNSOpen(cg_open(filename, cg_file_mode, &cgv->file_num), viewer, 0);
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscViewerCGNSCheckBatch_Internal(PetscViewer viewer)
{
  PetscViewer_CGNS *cgv = (PetscViewer_CGNS *)viewer->data;
  PetscCount        num_steps;

  PetscFunctionBegin;
  if (!cgv->filename_template) PetscFunctionReturn(PETSC_SUCCESS); // Batches are closed when viewer is destroyed
  PetscCall(PetscSegBufferGetSize(cgv->output_times, &num_steps));
  if (num_steps >= (PetscCount)cgv->batch_size) PetscCall(PetscViewerFileClose_CGNS(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerDestroy_CGNS(PetscViewer viewer)
{
  PetscViewer_CGNS *cgv = (PetscViewer_CGNS *)viewer->data;

  PetscFunctionBegin;
  PetscCall(PetscViewerFileClose_CGNS(viewer));
  PetscCall(PetscFree(cgv->solution_name));
  PetscCall(PetscFree(cgv->filename_template));
  PetscCall(PetscFree(cgv));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerFileSetName_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerFileGetName_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerFileSetMode_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer, "PetscViewerFileGetMode_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerFileSetMode_CGNS(PetscViewer viewer, PetscFileMode type)
{
  PetscViewer_CGNS *cgv = (PetscViewer_CGNS *)viewer->data;

  PetscFunctionBegin;
  cgv->btype = type;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerFileGetMode_CGNS(PetscViewer viewer, PetscFileMode *type)
{
  PetscViewer_CGNS *cgv = (PetscViewer_CGNS *)viewer->data;

  PetscFunctionBegin;
  *type = cgv->btype;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerFileSetName_CGNS(PetscViewer viewer, const char *filename)
{
  PetscViewer_CGNS *cgv = (PetscViewer_CGNS *)viewer->data;
  char             *has_pattern;

  PetscFunctionBegin;
#if defined(PETSC_HDF5_HAVE_PARALLEL)
  if (cgv->file_num) PetscCallCGNSClose(cgp_close(cgv->file_num), viewer, 0);
#else
  if (cgv->file_num) PetscCallCGNSClose(cg_close(cgv->file_num), viewer, 0);
#endif
  PetscCall(PetscFree(cgv->filename));
  PetscCall(PetscFree(cgv->filename_template));
  PetscCall(PetscStrchr(filename, '%', &has_pattern));
  if (has_pattern) {
    PetscCall(PetscStrallocpy(filename, &cgv->filename_template));
    // Templated file names open lazily, once we know DMGetOutputSequenceNumber()
  } else {
    PetscCall(PetscStrallocpy(filename, &cgv->filename));
    PetscCall(PetscViewerCGNSFileOpen_Internal(viewer, -1));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerFileGetName_CGNS(PetscViewer viewer, const char **filename)
{
  PetscViewer_CGNS *cgv = (PetscViewer_CGNS *)viewer->data;

  PetscFunctionBegin;
  *filename = cgv->filename;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   PETSCVIEWERCGNS - A viewer for CGNS files

  Level: beginner

  Options Database Key:
. -viewer_cgns_batch_size SIZE - set max number of output sequence times to write per batch

  Note:
  If the filename contains an integer format character, the CGNS viewer will created a batched output sequence. For
  example, one could use `-ts_monitor_solution cgns:flow-%d.cgns`. This is desirable if one wants to limit file sizes or
  if the job might crash/be killed by a resource manager before exiting cleanly.

.seealso: [](sec_viewers), `PetscViewer`, `PetscViewerCreate()`, `VecView()`, `DMView()`, `PetscViewerFileSetName()`, `PetscViewerFileSetMode()`, `TSSetFromOptions()`
M*/
PetscErrorCode PetscViewerCreate_CGNS(PetscViewer v)
{
  PetscViewer_CGNS *cgv;

  PetscFunctionBegin;
  PetscCall(PetscViewerCGNSRegisterLogEvents_Internal());
  PetscCall(PetscNew(&cgv));

  v->data                   = cgv;
  v->ops->destroy           = PetscViewerDestroy_CGNS;
  v->ops->setfromoptions    = PetscViewerSetFromOptions_CGNS;
  v->ops->view              = PetscViewerView_CGNS;
  cgv->btype                = FILE_MODE_UNDEFINED;
  cgv->filename             = NULL;
  cgv->batch_size           = 20;
  cgv->solution_index       = -1; // Default to use the "last" solution
  cgv->base                 = 1;
  cgv->zone                 = 1;
  cgv->previous_output_step = -1; // No previous output step

  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerFileSetName_C", PetscViewerFileSetName_CGNS));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerFileGetName_C", PetscViewerFileGetName_CGNS));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerFileSetMode_C", PetscViewerFileSetMode_CGNS));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerFileGetMode_C", PetscViewerFileGetMode_CGNS));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Find DataArray_t node under the current node (determined by `cg_goto` and friends) that matches `name`
// Return the index of that array and (optionally) other data about the array
static inline PetscErrorCode CGNS_Find_Array(MPI_Comm comm, const char name[], int *A_index, CGNS_ENUMT(DataType_t) * data_type, int *dim, cgsize_t size[], PetscBool *found)
{
  int  narrays; // number of arrays under the current node
  char array_name[CGIO_MAX_NAME_LENGTH + 1];
  CGNS_ENUMT(DataType_t) data_type_local;
  int       _dim;
  cgsize_t  _size[12];
  PetscBool matches_name = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCallCGNSRead(cg_narrays(&narrays), 0, 0);
  for (int i = 1; i <= narrays; i++) {
    PetscCallCGNSRead(cg_array_info(i, array_name, &data_type_local, &_dim, _size), 0, 0);
    PetscCall(PetscStrcmp(name, array_name, &matches_name));
    if (matches_name) {
      *A_index = i;
      if (data_type) *data_type = data_type_local;
      if (dim) *dim = _dim;
      if (size) PetscArraycpy(size, _size, _dim);
      if (found) *found = PETSC_TRUE;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }
  if (found) *found = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscViewerCGNSOpen - Opens a file for CGNS input/output.

  Collective

  Input Parameters:
+ comm - MPI communicator
. name - name of file
- type - type of file
.vb
    FILE_MODE_WRITE - create new file for binary output
    FILE_MODE_READ - open existing file for binary input
    FILE_MODE_APPEND - open existing file for binary output
.ve

  Output Parameter:
. viewer - `PETSCVIEWERCGNS` `PetscViewer` for CGNS input/output to use with the specified file

  Level: beginner

.seealso: `PETSCVIEWERCGNS`, `PetscViewer`, `PetscViewerPushFormat()`, `PetscViewerDestroy()`,
          `DMLoad()`, `PetscFileMode`, `PetscViewerSetType()`, `PetscViewerFileSetMode()`, `PetscViewerFileSetName()`
@*/
PetscErrorCode PetscViewerCGNSOpen(MPI_Comm comm, const char name[], PetscFileMode type, PetscViewer *viewer)
{
  PetscFunctionBegin;
  PetscAssertPointer(name, 2);
  PetscAssertPointer(viewer, 4);
  PetscCall(PetscViewerCreate(comm, viewer));
  PetscCall(PetscViewerSetType(*viewer, PETSCVIEWERCGNS));
  PetscCall(PetscViewerFileSetMode(*viewer, type));
  PetscCall(PetscViewerFileSetName(*viewer, name));
  PetscCall(PetscViewerSetFromOptions(*viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscViewerCGNSSetSolutionIndex - Set index of solution

  Not Collective

  Input Parameters:
+ viewer      - `PETSCVIEWERCGNS` `PetscViewer` for CGNS input/output to use with the specified file
- solution_id - Index of the solution id, or `-1` for the last solution on the file

  Level: intermediate

  Notes:
  By default, `solution_id` is set to `-1` to mean the last solution available in the file.
  If the file contains a FlowSolutionPointers node, then that array is indexed to determine which FlowSolution_t node to read from.
  Otherwise, `solution_id` indexes the total available FlowSolution_t nodes in the file.

  This solution index is used by `VecLoad()` to determine which solution to load from the file

.seealso: `PETSCVIEWERCGNS`, `PetscViewerCGNSGetSolutionIndex()`, `PetscViewerCGNSGetSolutionInfo()`

@*/
PetscErrorCode PetscViewerCGNSSetSolutionIndex(PetscViewer viewer, PetscInt solution_id)
{
  PetscViewer_CGNS *cgv = (PetscViewer_CGNS *)viewer->data;

  PetscFunctionBeginUser;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscValidLogicalCollectiveInt(viewer, solution_id, 2);
  PetscCheck((solution_id != 0) && (solution_id > -2), PetscObjectComm((PetscObject)viewer), PETSC_ERR_USER_INPUT, "Solution index must be either -1 or greater than 0, not %" PetscInt_FMT, solution_id);
  cgv->solution_index      = solution_id;
  cgv->solution_file_index = 0; // Reset sol_index when solution_id changes (0 is invalid)
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscViewerCGNSGetSolutionIndex - Get index of solution

  Not Collective

  Input Parameter:
. viewer - `PETSCVIEWERCGNS` `PetscViewer` for CGNS input/output to use with the specified file

  Output Parameter:
. solution_id - Index of the solution id

  Level: intermediate

  Notes:
  By default, solution_id is set to `-1` to mean the last solution available in the file

.seealso: `PETSCVIEWERCGNS`, `PetscViewerCGNSSetSolutionIndex()`, `PetscViewerCGNSGetSolutionInfo()`

@*/
PetscErrorCode PetscViewerCGNSGetSolutionIndex(PetscViewer viewer, PetscInt *solution_id)
{
  PetscViewer_CGNS *cgv = (PetscViewer_CGNS *)viewer->data;

  PetscFunctionBeginUser;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscAssertPointer(solution_id, 2);
  *solution_id = cgv->solution_index;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Gets the index for the solution in this CGNS file
PetscErrorCode PetscViewerCGNSGetSolutionFileIndex_Internal(PetscViewer viewer, int *sol_index)
{
  PetscViewer_CGNS *cgv  = (PetscViewer_CGNS *)viewer->data;
  MPI_Comm          comm = PetscObjectComm((PetscObject)viewer);
  int               nsols, cgns_ier;
  char              buffer[CGIO_MAX_NAME_LENGTH + 1];
  CGNS_ENUMT(GridLocation_t) gridloc; // Throwaway

  PetscFunctionBeginUser;
  if (cgv->solution_file_index > 0) {
    *sol_index = cgv->solution_file_index;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCallCGNSRead(cg_nsols(cgv->file_num, cgv->base, cgv->zone, &nsols), viewer, 0);
  cgns_ier = cg_goto(cgv->file_num, cgv->base, "Zone_t", cgv->zone, "ZoneIterativeData_t", 1, "FlowSolutionPointers", 0, NULL);
  if (cgns_ier == CG_NODE_NOT_FOUND) {
    // If FlowSolutionPointers does not exist, then just index off of nsols (which can include non-solution data)
    PetscCheck(cgv->solution_index == -1 || cgv->solution_index <= nsols, comm, PETSC_ERR_ARG_OUTOFRANGE, "CGNS Solution index (%" PetscInt_FMT ") not in [1, %d]", cgv->solution_index, nsols);

    cgv->solution_file_index = cgv->solution_index == -1 ? nsols : cgv->solution_index;
  } else {
    // If FlowSolutionPointers exists, then solution_id should index that array of FlowSolutions
    char     *pointer_id_name;
    PetscBool matches_name = PETSC_FALSE;
    int       sol_id;

    PetscCheck(cgns_ier == CG_OK, PETSC_COMM_SELF, PETSC_ERR_LIB, "CGNS error %d %s", cgns_ier, cg_get_error());
    cgns_ier = cg_goto(cgv->file_num, cgv->base, "Zone_t", cgv->zone, "ZoneIterativeData_t", 1, NULL);

    { // Get FlowSolutionPointer name corresponding to solution_id
      cgsize_t  size[12];
      int       dim, A_index;
      char     *pointer_names, *pointer_id_name_ref;
      PetscBool found_array;

      PetscCall(CGNS_Find_Array(comm, "FlowSolutionPointers", &A_index, NULL, &dim, size, &found_array));
      PetscCheck(found_array, comm, PETSC_ERR_SUP, "Cannot find FlowSolutionPointers array under current CGNS node");
      PetscCheck(cgv->solution_index == -1 || cgv->solution_index <= size[1], comm, PETSC_ERR_ARG_OUTOFRANGE, "CGNS Solution index (%" PetscInt_FMT ") not in range of FlowSolutionPointers [1, %" PRIdCGSIZE "]", cgv->solution_index, size[1]);
      PetscCall(PetscCalloc1(size[0] * size[1] + 1, &pointer_names)); // Need the +1 for (possibly) setting \0 for the last pointer name if it's full
      PetscCallCGNSRead(cg_array_read_as(1, CGNS_ENUMV(Character), pointer_names), viewer, 0);
      cgv->solution_file_pointer_index = cgv->solution_index == -1 ? size[1] : cgv->solution_index;
      pointer_id_name_ref              = &pointer_names[size[0] * (cgv->solution_file_pointer_index - 1)];
      { // Set last non-whitespace character of the pointer name to \0 (CGNS pads with spaces)
        int str_idx;
        for (str_idx = size[0] - 1; str_idx > 0; str_idx--) {
          if (!isspace((unsigned char)pointer_id_name_ref[str_idx])) break;
        }
        pointer_id_name_ref[str_idx + 1] = '\0';
      }
      PetscCall(PetscStrallocpy(pointer_id_name_ref, &pointer_id_name));
      PetscCall(PetscFree(pointer_names));
    }

    // Find FlowSolution_t node that matches pointer_id_name
    for (sol_id = 1; sol_id <= nsols; sol_id++) {
      PetscCallCGNSRead(cg_sol_info(cgv->file_num, cgv->base, cgv->zone, sol_id, buffer, &gridloc), viewer, 0);
      PetscCall(PetscStrcmp(pointer_id_name, buffer, &matches_name));
      if (matches_name) break;
    }
    PetscCheck(matches_name, comm, PETSC_ERR_LIB, "Cannot find FlowSolution_t node %s in file", pointer_id_name);
    cgv->solution_file_index = sol_id;
    PetscCall(PetscFree(pointer_id_name));
  }

  *sol_index = cgv->solution_file_index;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscViewerCGNSGetSolutionTime - Gets the solution time for the FlowSolution of the viewer

  Collective

  Input Parameter:
. viewer - `PETSCVIEWERCGNS` `PetscViewer` for CGNS input/output to use with the specified file

  Output Parameters:
+ time - Solution time of the FlowSolution_t node
- set  - Whether the time data is in the file

  Level: intermediate

  Notes:
  Reads data from a DataArray named `TimeValues` under a `BaseIterativeData_t` node

.seealso: `PETSCVIEWERCGNS`, `PetscViewer`, `PetscViewerCGNSGetSolutionIteration()`, `PetscViewerCGNSSetSolutionIndex()`, `PetscViewerCGNSGetSolutionIndex()`, `PetscViewerCGNSGetSolutionName()`
@*/
PetscErrorCode PetscViewerCGNSGetSolutionTime(PetscViewer viewer, PetscReal *time, PetscBool *set)
{
  PetscViewer_CGNS *cgv = (PetscViewer_CGNS *)viewer->data;
  int               cgns_ier, A_index = 0, sol_id;
  PetscReal        *times;
  cgsize_t          size[12];

  PetscFunctionBeginUser;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscAssertPointer(time, 2);
  PetscAssertPointer(set, 3);
  cgns_ier = cg_goto(cgv->file_num, cgv->base, "BaseIterativeData_t", 1, NULL);
  if (cgns_ier == CG_NODE_NOT_FOUND) {
    *set = PETSC_FALSE;
    PetscFunctionReturn(PETSC_SUCCESS);
  } else PetscCallCGNS(cgns_ier);
  PetscCall(CGNS_Find_Array(PetscObjectComm((PetscObject)viewer), "TimeValues", &A_index, NULL, NULL, size, set));
  if (!*set) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscMalloc1(size[0], &times));
  PetscCallCGNSRead(cg_array_read_as(A_index, CGNS_ENUMV(RealDouble), times), viewer, 0);
  PetscCall(PetscViewerCGNSGetSolutionFileIndex_Internal(viewer, &sol_id)); // Call to set file pointer index
  *time = times[cgv->solution_file_pointer_index - 1];
  PetscCall(PetscFree(times));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscViewerCGNSGetSolutionIteration - Gets the solution iteration for the FlowSolution of the viewer

  Collective

  Input Parameter:
. viewer - `PETSCVIEWERCGNS` `PetscViewer` for CGNS input/output to use with the specified file

  Output Parameters:
+ iteration - Solution iteration of the FlowSolution_t node
- set       - Whether the time data is in the file

  Level: intermediate

  Notes:
  Reads data from a DataArray named `IterationValues` under a `BaseIterativeData_t` node

.seealso: `PETSCVIEWERCGNS`, `PetscViewer`, `PetscViewerCGNSGetSolutionTime()`, `PetscViewerCGNSSetSolutionIndex()`, `PetscViewerCGNSGetSolutionIndex()`, `PetscViewerCGNSGetSolutionName()`
@*/
PetscErrorCode PetscViewerCGNSGetSolutionIteration(PetscViewer viewer, PetscInt *iteration, PetscBool *set)
{
  PetscViewer_CGNS *cgv = (PetscViewer_CGNS *)viewer->data;
  int               cgns_ier, A_index = 0, sol_id;
  int              *steps;
  cgsize_t          size[12];

  PetscFunctionBeginUser;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscAssertPointer(iteration, 2);
  PetscAssertPointer(set, 3);
  cgns_ier = cg_goto(cgv->file_num, cgv->base, "BaseIterativeData_t", 1, NULL);
  if (cgns_ier == CG_NODE_NOT_FOUND) {
    *set = PETSC_FALSE;
    PetscFunctionReturn(PETSC_SUCCESS);
  } else PetscCallCGNS(cgns_ier);
  PetscCall(CGNS_Find_Array(PetscObjectComm((PetscObject)viewer), "IterationValues", &A_index, NULL, NULL, size, set));
  if (!*set) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscMalloc1(size[0], &steps));
  PetscCallCGNSRead(cg_array_read_as(A_index, CGNS_ENUMV(Integer), steps), viewer, 0);
  PetscCall(PetscViewerCGNSGetSolutionFileIndex_Internal(viewer, &sol_id)); // Call to set file pointer index
  *iteration = (PetscInt)steps[cgv->solution_file_pointer_index - 1];
  PetscCall(PetscFree(steps));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  PetscViewerCGNSGetSolutionName - Gets name of FlowSolution of the viewer

  Collective

  Input Parameter:
. viewer - `PETSCVIEWERCGNS` `PetscViewer` for CGNS input/output to use with the specified file

  Output Parameter:
. name - Name of the FlowSolution_t node corresponding to the solution index

  Level: intermediate

  Notes:
  Currently assumes there is only one Zone in the CGNS file

.seealso: `PETSCVIEWERCGNS`, `PetscViewer`, `PetscViewerCGNSSetSolutionIndex()`, `PetscViewerCGNSGetSolutionIndex()`, `PetscViewerCGNSGetSolutionTime()`
@*/
PetscErrorCode PetscViewerCGNSGetSolutionName(PetscViewer viewer, const char *name[])
{
  PetscViewer_CGNS *cgv = (PetscViewer_CGNS *)viewer->data;
  int               sol_id;
  char              buffer[CGIO_MAX_NAME_LENGTH + 1];
  CGNS_ENUMT(GridLocation_t) gridloc; // Throwaway

  PetscFunctionBeginUser;
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 1);
  PetscAssertPointer(name, 2);
  PetscCall(PetscViewerCGNSGetSolutionFileIndex_Internal(viewer, &sol_id));

  PetscCallCGNSRead(cg_sol_info(cgv->file_num, cgv->base, cgv->zone, sol_id, buffer, &gridloc), viewer, 0);
  if (cgv->solution_name) PetscCall(PetscFree(cgv->solution_name));
  PetscCall(PetscStrallocpy(buffer, &cgv->solution_name));
  *name = cgv->solution_name;
  PetscFunctionReturn(PETSC_SUCCESS);
}
