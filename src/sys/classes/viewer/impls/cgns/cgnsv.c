#include <petsc/private/viewercgnsimpl.h> /*I "petscviewer.h" I*/
#if defined(PETSC_HDF5_HAVE_PARALLEL)
#include <pcgnslib.h>
#else
#include <cgnslib.h>
#endif

static PetscErrorCode PetscViewerSetFromOptions_CGNS(PetscOptionItems *PetscOptionsObject,PetscViewer v)
{
  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject,"CGNS Viewer Options");
  PetscOptionsHeadEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerView_CGNS(PetscViewer v,PetscViewer viewer)
{
  PetscViewer_CGNS  *cgv = (PetscViewer_CGNS*)v->data;

  PetscFunctionBegin;
  if (cgv->filename) {
    PetscCall(PetscViewerASCIIPrintf(viewer,"Filename: %s\n", cgv->filename));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerFileClose_CGNS(PetscViewer viewer)
{
  PetscViewer_CGNS *cgv = (PetscViewer_CGNS*)viewer->data;

  PetscFunctionBegin;
  if (cgv->output_times) {
    size_t size, width = 32;
    char *solnames;
    PetscReal *times;
    cgsize_t num_times;
    PetscCall(PetscSegBufferGetSize(cgv->output_times, &size));
    PetscCall(PetscSegBufferExtractInPlace(cgv->output_times, &times));
    num_times = size;
    PetscCallCGNS(cg_biter_write(cgv->file_num, cgv->base, "TimeIterValues", num_times));
    PetscCallCGNS(cg_goto(cgv->file_num, cgv->base, "BaseIterativeData_t", 1, NULL));
    PetscCallCGNS(cg_array_write("TimeValues", CGNS_ENUMV(RealDouble), 1, &num_times, times));
    PetscCall(PetscSegBufferDestroy(&cgv->output_times));
    PetscCallCGNS(cg_ziter_write(cgv->file_num, cgv->base, cgv->zone, "ZoneIterativeData"));
    PetscCallCGNS(cg_goto(cgv->file_num, cgv->base, "Zone_t", cgv->zone, "ZoneIterativeData_t", 1, NULL));
    PetscCall(PetscMalloc(size*width+1, &solnames));
    for (size_t i=0; i<size; i++) PetscCall(PetscSNPrintf(&solnames[i*width], width+1, "FlowSolution%-20zu", i));
    cgsize_t shape[2] = {(cgsize_t)width, (cgsize_t)size};
    PetscCallCGNS(cg_array_write("FlowSolutionPointers", CGNS_ENUMV(Character), 2, shape, solnames));
    // The VTK reader looks for names like FlowSolution*Pointers.
    for (size_t i=0; i<size; i++) PetscCall(PetscSNPrintf(&solnames[i*width], width+1, "%-32s", "CellInfo"));
    PetscCallCGNS(cg_array_write("FlowSolutionCellInfoPointers", CGNS_ENUMV(Character), 2, shape, solnames));
    PetscCall(PetscFree(solnames));

    PetscCallCGNS(cg_simulation_type_write(cgv->file_num, cgv->base, CGNS_ENUMV(TimeAccurate)));
  }
  PetscCall(PetscFree(cgv->filename));
#if defined(PETSC_HDF5_HAVE_PARALLEL)
  PetscCallCGNS(cgp_close(cgv->file_num));
#else
  if (cgv->file_num) PetscCallCGNS(cg_close(cgv->file_num));
#endif
  cgv->file_num = 0;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerDestroy_CGNS(PetscViewer viewer)
{
  PetscViewer_CGNS *cgv = (PetscViewer_CGNS*)viewer->data;

  PetscFunctionBegin;
  PetscCall(PetscViewerFileClose_CGNS(viewer));
  PetscCall(PetscFree(cgv->node_l2g));
  PetscCall(PetscFree(cgv->nodal_field));
  PetscCall(PetscFree(cgv));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer,"PetscViewerFileSetName_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer,"PetscViewerFileGetName_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer,"PetscViewerFileSetMode_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)viewer,"PetscViewerFileGetMode_C",NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerFileSetMode_CGNS(PetscViewer viewer, PetscFileMode type)
{
  PetscViewer_CGNS *cgv = (PetscViewer_CGNS*)viewer->data;

  PetscFunctionBegin;
  cgv->btype = type;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerFileGetMode_CGNS(PetscViewer viewer, PetscFileMode *type)
{
  PetscViewer_CGNS *cgv = (PetscViewer_CGNS*)viewer->data;

  PetscFunctionBegin;
  *type = cgv->btype;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerFileSetName_CGNS(PetscViewer viewer, const char *filename)
{
  PetscViewer_CGNS *cgv = (PetscViewer_CGNS*)viewer->data;

  PetscFunctionBegin;
  if (cgv->file_num) PetscCallCGNS(cg_close(cgv->file_num));
  PetscCall(PetscFree(cgv->filename));
  PetscCall(PetscStrallocpy(filename, &cgv->filename));

  switch (cgv->btype) {
  case FILE_MODE_READ:
    SETERRQ(PetscObjectComm((PetscObject)viewer),PETSC_ERR_SUP,"FILE_MODE_READ not yet implemented");
    break;
  case FILE_MODE_WRITE:
#if defined(PETSC_HDF5_HAVE_PARALLEL)
    PetscCallCGNS(cgp_mpi_comm(PetscObjectComm((PetscObject)viewer)));
    PetscCallCGNS(cgp_open(filename, CG_MODE_WRITE, &cgv->file_num));
#else
    PetscCallCGNS(cg_open(filename, CG_MODE_WRITE, &cgv->file_num));
#endif
    break;
  case FILE_MODE_UNDEFINED:
    SETERRQ(PetscObjectComm((PetscObject)viewer),PETSC_ERR_ORDER, "Must call PetscViewerFileSetMode() before PetscViewerFileSetName()");
  default:
    SETERRQ(PetscObjectComm((PetscObject)viewer),PETSC_ERR_SUP, "Unsupported file mode %s",PetscFileModes[cgv->btype]);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscViewerFileGetName_CGNS(PetscViewer viewer, const char **filename)
{
  PetscViewer_CGNS *cgv = (PetscViewer_CGNS*)viewer->data;

  PetscFunctionBegin;
  *filename = cgv->filename;
  PetscFunctionReturn(0);
}

/*MC
   PETSCVIEWERCGNS - A viewer for CGNS files

.seealso: `PetscViewerCreate()`, `VecView()`, `DMView()`, `PetscViewerFileSetName()`, `PetscViewerFileSetMode()`

  Level: beginner
M*/

PETSC_EXTERN PetscErrorCode PetscViewerCreate_CGNS(PetscViewer v)
{
  PetscViewer_CGNS *cgv;

  PetscFunctionBegin;
  PetscCall(PetscNewLog(v,&cgv));

  v->data                = cgv;
  v->ops->destroy        = PetscViewerDestroy_CGNS;
  v->ops->setfromoptions = PetscViewerSetFromOptions_CGNS;
  v->ops->view           = PetscViewerView_CGNS;
  cgv->btype            = FILE_MODE_UNDEFINED;
  cgv->filename         = NULL;

  PetscCall(PetscObjectComposeFunction((PetscObject)v,"PetscViewerFileSetName_C",PetscViewerFileSetName_CGNS));
  PetscCall(PetscObjectComposeFunction((PetscObject)v,"PetscViewerFileGetName_C",PetscViewerFileGetName_CGNS));
  PetscCall(PetscObjectComposeFunction((PetscObject)v,"PetscViewerFileSetMode_C",PetscViewerFileSetMode_CGNS));
  PetscCall(PetscObjectComposeFunction((PetscObject)v,"PetscViewerFileGetMode_C",PetscViewerFileGetMode_CGNS));
  PetscFunctionReturn(0);
}
