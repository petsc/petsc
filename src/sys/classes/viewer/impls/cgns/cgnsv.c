#include <petsc/private/viewercgnsimpl.h> /*I "petscviewer.h" I*/
#if defined(PETSC_HDF5_HAVE_PARALLEL)
  #include <pcgnslib.h>
#else
  #include <cgnslib.h>
#endif

static PetscErrorCode PetscViewerSetFromOptions_CGNS(PetscViewer v, PetscOptionItems *PetscOptionsObject)
{
  PetscViewer_CGNS *cgv = (PetscViewer_CGNS *)v->data;
  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "CGNS Viewer Options");
  PetscOptionsInt("-viewer_cgns_batch_size", "Max number of steps to store in single file when using a template cgns:name-%d.cgns", "", cgv->batch_size, &cgv->batch_size, NULL);
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
    size_t     size, width = 32, *steps;
    char      *solnames;
    PetscReal *times;
    cgsize_t   num_times;
    PetscCall(PetscSegBufferGetSize(cgv->output_times, &size));
    PetscCall(PetscSegBufferExtractInPlace(cgv->output_times, &times));
    num_times = size;
    PetscCallCGNS(cg_biter_write(cgv->file_num, cgv->base, "TimeIterValues", num_times));
    PetscCallCGNS(cg_goto(cgv->file_num, cgv->base, "BaseIterativeData_t", 1, NULL));
    PetscCallCGNS(cg_array_write("TimeValues", CGNS_ENUMV(RealDouble), 1, &num_times, times));
    PetscCall(PetscSegBufferDestroy(&cgv->output_times));
    PetscCallCGNS(cg_ziter_write(cgv->file_num, cgv->base, cgv->zone, "ZoneIterativeData"));
    PetscCallCGNS(cg_goto(cgv->file_num, cgv->base, "Zone_t", cgv->zone, "ZoneIterativeData_t", 1, NULL));
    PetscCall(PetscMalloc(size * width + 1, &solnames));
    PetscCall(PetscSegBufferExtractInPlace(cgv->output_steps, &steps));
    for (size_t i = 0; i < size; i++) PetscCall(PetscSNPrintf(&solnames[i * width], width + 1, "FlowSolution%-20zu", steps[i]));
    PetscCall(PetscSegBufferDestroy(&cgv->output_steps));
    cgsize_t shape[2] = {(cgsize_t)width, (cgsize_t)size};
    PetscCallCGNS(cg_array_write("FlowSolutionPointers", CGNS_ENUMV(Character), 2, shape, solnames));
    // The VTK reader looks for names like FlowSolution*Pointers.
    for (size_t i = 0; i < size; i++) PetscCall(PetscSNPrintf(&solnames[i * width], width + 1, "%-32s", "CellInfo"));
    PetscCallCGNS(cg_array_write("FlowSolutionCellInfoPointers", CGNS_ENUMV(Character), 2, shape, solnames));
    PetscCall(PetscFree(solnames));

    PetscCallCGNS(cg_simulation_type_write(cgv->file_num, cgv->base, CGNS_ENUMV(TimeAccurate)));
  }
  PetscCall(PetscFree(cgv->filename));
#if defined(PETSC_HDF5_HAVE_PARALLEL)
  if (cgv->file_num) PetscCallCGNS(cgp_close(cgv->file_num));
#else
  if (cgv->file_num) PetscCallCGNS(cg_close(cgv->file_num));
#endif
  cgv->file_num = 0;
  PetscCall(PetscFree(cgv->node_l2g));
  PetscCall(PetscFree(cgv->nodal_field));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscViewerCGNSFileOpen_Internal(PetscViewer viewer, PetscInt sequence_number)
{
  PetscViewer_CGNS *cgv = (PetscViewer_CGNS *)viewer->data;
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
    SETERRQ(PetscObjectComm((PetscObject)viewer), PETSC_ERR_SUP, "FILE_MODE_READ not yet implemented");
    break;
  case FILE_MODE_WRITE:
#if defined(PETSC_HDF5_HAVE_PARALLEL)
    PetscCallCGNS(cgp_mpi_comm(PetscObjectComm((PetscObject)viewer)));
    PetscCallCGNS(cgp_open(cgv->filename, CG_MODE_WRITE, &cgv->file_num));
#else
    PetscCallCGNS(cg_open(filename, CG_MODE_WRITE, &cgv->file_num));
#endif
    break;
  case FILE_MODE_UNDEFINED:
    SETERRQ(PetscObjectComm((PetscObject)viewer), PETSC_ERR_ORDER, "Must call PetscViewerFileSetMode() before PetscViewerFileSetName()");
  default:
    SETERRQ(PetscObjectComm((PetscObject)viewer), PETSC_ERR_SUP, "Unsupported file mode %s", PetscFileModes[cgv->btype]);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscViewerCGNSCheckBatch_Internal(PetscViewer viewer)
{
  PetscViewer_CGNS *cgv = (PetscViewer_CGNS *)viewer->data;
  size_t            num_steps;

  PetscFunctionBegin;
  if (!cgv->filename_template) PetscFunctionReturn(PETSC_SUCCESS); // Batches are closed when viewer is destroyed
  PetscCall(PetscSegBufferGetSize(cgv->output_times, &num_steps));
  if (num_steps >= cgv->batch_size) PetscCall(PetscViewerFileClose_CGNS(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscViewerDestroy_CGNS(PetscViewer viewer)
{
  PetscViewer_CGNS *cgv = (PetscViewer_CGNS *)viewer->data;

  PetscFunctionBegin;
  PetscCall(PetscViewerFileClose_CGNS(viewer));
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
  if (cgv->file_num) PetscCallCGNS(cg_close(cgv->file_num));
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

PETSC_EXTERN PetscErrorCode PetscViewerCreate_CGNS(PetscViewer v)
{
  PetscViewer_CGNS *cgv;

  PetscFunctionBegin;
  PetscCall(PetscNew(&cgv));

  v->data                = cgv;
  v->ops->destroy        = PetscViewerDestroy_CGNS;
  v->ops->setfromoptions = PetscViewerSetFromOptions_CGNS;
  v->ops->view           = PetscViewerView_CGNS;
  cgv->btype             = FILE_MODE_UNDEFINED;
  cgv->filename          = NULL;
  cgv->batch_size        = 20;

  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerFileSetName_C", PetscViewerFileSetName_CGNS));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerFileGetName_C", PetscViewerFileGetName_CGNS));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerFileSetMode_C", PetscViewerFileSetMode_CGNS));
  PetscCall(PetscObjectComposeFunction((PetscObject)v, "PetscViewerFileGetMode_C", PetscViewerFileGetMode_CGNS));
  PetscFunctionReturn(PETSC_SUCCESS);
}
