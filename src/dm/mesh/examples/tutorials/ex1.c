/*T
   Concepts: Mesh^loading a mesh
   Concepts: Mesh^partitioning a mesh
   Concepts: Mesh^viewing a mesh
   Processors: n
T*/

/*
  Read in a mesh using the PCICE format:

  connectivity file:
  ------------------
  NumCells
  Cell #   v_0 v_1 ... v_d
  .
  .
  .

  coordinate file:
  ----------------
  NumVertices
  Vertex #  x_0 x_1 ... x_{d-1}
  .
  .
  .

Partition the mesh and distribute it to each process.

Output the mesh in VTK format with a scalar field indicating
the rank of the process owning each cell.
*/

static char help[] = "Reads, partitions, and outputs an unstructured mesh.\n\n";

#include <Distribution.hh>
#include "petscmesh.h"
#include "petscviewer.h"
#include "src/dm/mesh/meshpcice.h"
#include "src/dm/mesh/meshpylith.h"
#include <stdlib.h>
#include <string.h>

using ALE::Obj;

typedef enum {PCICE, PYLITH} FileType;

typedef struct {
  int            debug;              // The debugging level
  PetscInt       dim;                // The topological mesh dimension
  PetscTruth     useZeroBase;        // Use zero-based indexing
  FileType       inputFileType;      // The input file type, e.g. PCICE
  FileType       outputFileType;     // The output file type, e.g. PCICE
  char           baseFilename[2048]; // The base filename for mesh files
  PetscTruth     output;             // Output the mesh
  PetscTruth     outputLocal;        // Output the local form of the mesh
  PetscTruth     outputVTK;          // Output the mesh in VTK
  PetscTruth     distribute;         // Distribute the mesh among processes
  char           partitioner[2048];  // The partitioner name
  PetscTruth     interpolate;        // Construct missing elements of the mesh
  PetscTruth     doPartition;        // Construct field over cells indicating process number
  SectionReal    partition;          // Section with partition number in each cell
  PetscTruth     doMaterial;         // Construct field over cells indicating material type
  SectionReal    material;           // Section with material number in each cell
  SectionPair    split;              // Section with split node values in fault cells
  SectionReal    traction;           // Section with tractions in boundary cells
  PetscTruth     doOdd;              // Construct field over odd cells indicating process number
  SectionReal    odd;                // Section with cell number in each odd cell
} Options;

EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshView_Sieve(const Obj<ALE::Mesh>&, PetscViewer);

#undef __FUNCT__
#define __FUNCT__ "OutputVTK"
PetscErrorCode OutputVTK(Mesh mesh, Options *options)
{
  MPI_Comm       comm;
  PetscViewer    viewer;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (options->outputVTK) {
    ALE::LogStage stage = ALE::LogStageRegister("VTKOutput");
    ALE::LogStagePush(stage);
    ierr = PetscObjectGetComm((PetscObject) mesh, &comm);CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "Creating VTK mesh file\n");CHKERRQ(ierr);
    ierr = PetscViewerCreate(comm, &viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer, "testMesh.vtk");CHKERRQ(ierr);
    ierr = MeshView(mesh, viewer);CHKERRQ(ierr);
    if (options->doPartition) {
      ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_VTK_CELL);CHKERRQ(ierr);
      ierr = SectionRealView(options->partition, viewer);CHKERRQ(ierr);
      ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    }
    if (options->doMaterial) {
      ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_VTK_CELL);CHKERRQ(ierr);
      ierr = SectionRealView(options->material, viewer);CHKERRQ(ierr);
      ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    }
    if (options->odd) {
      ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_VTK_CELL);CHKERRQ(ierr);
      ierr = SectionRealView(options->odd, viewer);CHKERRQ(ierr);
      ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    }
    ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
    ALE::LogStagePop(stage);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "OutputMesh"
PetscErrorCode OutputMesh(Mesh mesh, Options *options)
{
  MPI_Comm       comm;
  PetscViewer    viewer;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (options->output) {
    ALE::LogStage stage = ALE::LogStageRegister("MeshOutput");
    ALE::LogStagePush(stage);
    ierr = PetscObjectGetComm((PetscObject) mesh, &comm);CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "Creating original format mesh file\n");CHKERRQ(ierr);
    ierr = PetscViewerCreate(comm, &viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
    if (options->outputFileType == PCICE) {
      ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_PCICE);CHKERRQ(ierr);
      ierr = PetscViewerFileSetName(viewer, "testMesh.lcon");CHKERRQ(ierr);
    } else if (options->outputFileType == PYLITH) {
      if (options->outputLocal) {
        ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_PYLITH_LOCAL);CHKERRQ(ierr);
        ierr = PetscViewerFileSetMode(viewer, FILE_MODE_READ);CHKERRQ(ierr);
        ierr = PetscExceptionTry1(PetscViewerFileSetName(viewer, "testMesh"), PETSC_ERR_FILE_OPEN);
        if (PetscExceptionValue(ierr)) {
          /* this means that a caller above me has also tryed this exception so I don't handle it here, pass it up */
        } else if (PetscExceptionCaught(ierr, PETSC_ERR_FILE_OPEN)) {
          ierr = 0;
        } 
        CHKERRQ(ierr);
      } else {
        ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_PYLITH);CHKERRQ(ierr);
        ierr = PetscViewerFileSetMode(viewer, FILE_MODE_READ);CHKERRQ(ierr);
        ierr = PetscExceptionTry1(PetscViewerFileSetName(viewer, "testMesh"), PETSC_ERR_FILE_OPEN);
        if (PetscExceptionValue(ierr)) {
          /* this means that a caller above me has also tryed this exception so I don't handle it here, pass it up */
        } else if (PetscExceptionCaught(ierr, PETSC_ERR_FILE_OPEN)) {
          ierr = 0;
        } 
        CHKERRQ(ierr);
      }
    }
    ierr = MeshView(mesh, viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
    ALE::LogStagePop(stage);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreatePartition"
// Creates a field whose value is the processor rank on each element
PetscErrorCode CreatePartition(Mesh mesh, SectionReal *partition)
{
  Obj<ALE::Mesh::section_type> section;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ALE_LOG_EVENT_BEGIN;
  ierr = MeshGetCellSectionReal(mesh, 1, partition);CHKERRQ(ierr);
  ierr = SectionRealGetSection(*partition, section);CHKERRQ(ierr);
  const ALE::Mesh::section_type::patch_type                patch    = 0;
  const Obj<ALE::Mesh::topology_type>&                     topology = section->getTopology();
  const Obj<ALE::Mesh::topology_type::label_sequence>&     cells    = topology->heightStratum(patch, 0);
  const ALE::Mesh::topology_type::label_sequence::iterator end      = cells->end();
  const ALE::Mesh::section_type::value_type                rank     = section->commRank();

  for(ALE::Mesh::topology_type::label_sequence::iterator c_iter = cells->begin(); c_iter != end; ++c_iter) {
    section->updatePoint(patch, *c_iter, &rank);
  }
  ALE_LOG_EVENT_END;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DistributeMesh"
PetscErrorCode DistributeMesh(Mesh mesh, Options *options)
{
  Obj<ALE::Mesh> m;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  if (options->distribute) {
    ALE::LogStage stage = ALE::LogStageRegister("MeshDistribution");
    ALE::LogStagePush(stage);
    ierr = PetscPrintf(m->comm(), "Distributing mesh\n");CHKERRQ(ierr);
    m    = ALE::New::Distribution<ALE::Mesh::topology_type>::redistributeMesh(m, std::string(options->partitioner));
    ALE::LogStagePop(stage);
    ierr = MeshSetMesh(mesh, m);CHKERRQ(ierr);
  }
  if (options->doMaterial) {
    SectionReal parallelMaterial;

    ierr = SectionRealDistribute(options->material, mesh, &parallelMaterial);CHKERRQ(ierr);
    ierr = SectionRealDestroy(options->material);CHKERRQ(ierr);
    options->material = parallelMaterial;
  }
  if (options->split) {
    SectionPair parallelSplit;

    ierr = SectionPairDistribute(options->split, mesh, &parallelSplit);CHKERRQ(ierr);
    ierr = SectionPairDestroy(options->split);CHKERRQ(ierr);
    options->split = parallelSplit;
    Obj<ALE::Mesh::pair_section_type> section;
    ierr = SectionPairGetSection(parallelSplit, section);CHKERRQ(ierr);
    section->view("Parallel split section");
  }
  if (options->traction) {
    SectionReal parallelTraction;

    ierr = SectionRealDistribute(options->traction, mesh, &parallelTraction);CHKERRQ(ierr);
    ierr = SectionRealDestroy(options->traction);CHKERRQ(ierr);
    options->traction = parallelTraction;
    Obj<ALE::Mesh::section_type> section;
    ierr = SectionRealGetSection(parallelTraction, section);CHKERRQ(ierr);
    section->view("Parallel traction section");
  }
  if (options->doPartition) {
    ierr = CreatePartition(mesh, &options->partition);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateOdd"
// Creates a field whose value is the element number on each element with an odd number
PetscErrorCode CreateOdd(Mesh mesh, SectionReal *odd)
{
  Obj<ALE::Mesh> m;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ALE_LOG_EVENT_BEGIN;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = SectionRealCreate(m->comm(), odd);CHKERRQ(ierr);
  const Obj<ALE::Mesh::int_section_type>&              section = new ALE::Mesh::int_section_type(m->getTopology());
  const ALE::Mesh::section_type::patch_type            patch   = 0;
  const Obj<ALE::Mesh::topology_type::label_sequence>& cells   = m->getTopology()->heightStratum(patch, 0);
  ALE::Mesh::topology_type::label_sequence::iterator   begin   = cells->begin();
  ALE::Mesh::topology_type::label_sequence::iterator   end     = cells->end();

  for(ALE::Mesh::topology_type::label_sequence::iterator c_iter = begin; c_iter != end; ++c_iter) {
    const int num = *c_iter;

    if (num%2) {
      section->setFiberDimension(patch, num, num%3+1);
    }
  }
  section->allocate();
  for(ALE::Mesh::topology_type::label_sequence::iterator c_iter = begin; c_iter != end; ++c_iter) {
    const int num = *c_iter;
    int       val[3];

    if (num%2) {
      for(int n = 0; n <= num%3; n++) val[n] = num+n;
      section->updatePoint(patch, num, val);
    }
  }
  //FIX ierr = SectionRealSetSection(*odd, section);CHKERRQ(ierr);
  ALE_LOG_EVENT_END;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateMesh"
PetscErrorCode CreateMesh(MPI_Comm comm, Options *options, Mesh *mesh)
{
  Obj<ALE::Mesh> m;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ALE::LogStage stage = ALE::LogStageRegister("MeshCreation");
  ALE::LogStagePush(stage);
  ierr = PetscPrintf(comm, "Creating mesh\n");CHKERRQ(ierr);
  ierr = MeshCreate(comm, mesh);CHKERRQ(ierr);
  if (options->inputFileType == PCICE) {
    m = ALE::PCICE::Builder::readMesh(comm, options->dim, options->baseFilename, options->useZeroBase, options->interpolate, options->debug);
  } else if (options->inputFileType == PYLITH) {
    Obj<ALE::Mesh::section_type> material = new ALE::Mesh::section_type(comm, options->debug);

    m    = ALE::PyLith::Builder::readMesh(material, options->dim, options->baseFilename, options->useZeroBase, options->interpolate);
    ierr = SectionRealCreate(comm, &options->material);CHKERRQ(ierr);
    ierr = SectionRealSetSection(options->material, material);CHKERRQ(ierr);
    Obj<ALE::Mesh::pair_section_type> split = ALE::PyLith::Builder::createSplit(m, options->baseFilename, options->useZeroBase);

    if (!split.isNull()) {
      ierr = SectionPairCreate(comm, &options->split);CHKERRQ(ierr);
      ierr = SectionPairSetSection(options->split, split);CHKERRQ(ierr);
      split->view("Split section");
    }
    Obj<ALE::Mesh::section_type> traction = ALE::PyLith::Builder::createTraction(m, options->baseFilename, options->useZeroBase);
    if (!traction.isNull()) {
      ierr = SectionRealCreate(comm, &options->traction);CHKERRQ(ierr);
      ierr = SectionRealSetSection(options->traction, traction);CHKERRQ(ierr);
      traction->view("Traction section");
    }
  } else {
    SETERRQ1(PETSC_ERR_ARG_WRONG, "Invalid mesh input type: %d", options->inputFileType);
  }
  ierr = MeshSetMesh(*mesh, m);CHKERRQ(ierr);
  ALE::LogStagePop(stage);
  Obj<ALE::Mesh::topology_type> topology = m->getTopology();
  ierr = PetscPrintf(comm, "  Read %d elements\n", topology->heightStratum(0, 0)->size());CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "  Read %d vertices\n", topology->depthStratum(0, 0)->size());CHKERRQ(ierr);
  if (options->debug) {
    topology->view("Serial topology");
  }
  if (options->doOdd) {
    ierr = CreateOdd(*mesh, &options->odd);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, Options *options)
{
  const char    *fileTypes[2] = {"pcice", "pylith"};
  PetscInt       inputFt, outputFt;
  PetscTruth     setOutputType;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->debug          = 0;
  options->dim            = 2;
  options->useZeroBase    = PETSC_TRUE;
  options->inputFileType  = PCICE;
  options->outputFileType = PCICE;
  ierr = PetscStrcpy(options->baseFilename, "data/ex1_2d");CHKERRQ(ierr);
  options->output         = PETSC_TRUE;
  options->outputLocal    = PETSC_FALSE;
  options->outputVTK      = PETSC_TRUE;
  options->distribute     = PETSC_TRUE;
  ierr = PetscStrcpy(options->partitioner, "chaco");CHKERRQ(ierr);
  options->interpolate    = PETSC_TRUE;
  options->doPartition    = PETSC_TRUE;
  options->partition      = PETSC_NULL;
  options->doMaterial     = PETSC_TRUE;
  options->material       = PETSC_NULL;
  options->split          = PETSC_NULL;
  options->traction       = PETSC_NULL;
  options->doOdd          = PETSC_FALSE;
  options->odd            = PETSC_NULL;

  inputFt  = (PetscInt) options->inputFileType;
  outputFt = (PetscInt) options->outputFileType;

  ierr = PetscOptionsBegin(comm, "", "Options for mesh loading", "DMMG");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-debug", "The debugging level", "ex1.c", options->debug, &options->debug, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex1.c", options->dim, &options->dim, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-use_zero_base", "Use zero-based indexing", "ex1.c", options->useZeroBase, &options->useZeroBase, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEList("-file_type", "Type of input files", "ex1.c", fileTypes, 2, fileTypes[0], &inputFt, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEList("-output_file_type", "Type of output files", "ex1.c", fileTypes, 2, fileTypes[0], &outputFt, &setOutputType);CHKERRQ(ierr);
    ierr = PetscOptionsString("-base_file", "The base filename for mesh files", "ex1.c", options->baseFilename, options->baseFilename, 2048, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-output", "Output the mesh", "ex1.c", options->output, &options->output, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-output_local", "Output the local form of the mesh", "ex1.c", options->outputLocal, &options->outputLocal, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-output_vtk", "Output the mesh in VTK", "ex1.c", options->outputVTK, &options->outputVTK, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-distribute", "Distribute the mesh among processes", "ex1.c", options->distribute, &options->distribute, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsString("-partitioner", "The partitioner name", "ex1.c", options->partitioner, options->partitioner, 2048, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-interpolate", "Construct missing elements of the mesh", "ex1.c", options->interpolate, &options->interpolate, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-partition", "Create the partition field", "ex1.c", options->doPartition, &options->doPartition, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-material", "Create the material field", "ex1.c", options->doMaterial, &options->doMaterial, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-odd", "Create the odd field", "ex1.c", options->doOdd, &options->doOdd, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  options->inputFileType = (FileType) inputFt;
  if (setOutputType) {
    options->outputFileType = (FileType) outputFt;
  } else {
    options->outputFileType = options->inputFileType;
  }
  if (options->doMaterial && options->inputFileType != PYLITH) {
    options->doMaterial = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  MPI_Comm       comm;
  Options        options;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr);
  comm = PETSC_COMM_WORLD;

  try {
    Mesh mesh;

    ierr = ProcessOptions(comm, &options);CHKERRQ(ierr);
    ierr = CreateMesh(comm, &options, &mesh);CHKERRQ(ierr);
    ierr = DistributeMesh(mesh, &options);CHKERRQ(ierr);
    ierr = OutputVTK(mesh, &options);CHKERRQ(ierr);
    ierr = OutputMesh(mesh, &options);CHKERRQ(ierr);
  } catch (ALE::Exception e) {
    std::cout << e << std::endl;
    MPI_Abort(comm, 1);
  }
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
