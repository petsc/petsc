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
  PetscTruth     partition;          // Construct field over cells indicating process number
  PetscTruth     material;           // Construct field over cells indicating material type
  PetscTruth     odd;                // Construct field over odd cells indicating process number
} Options;

EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshView_Sieve(const Obj<ALE::Mesh>&, PetscViewer);
EXTERN PetscErrorCode PETSCDM_DLLEXPORT FieldView_Sieve(const Obj<ALE::Mesh>&, const std::string&, PetscViewer);
PetscErrorCode ProcessOptions(MPI_Comm, Options *);
PetscErrorCode CreateMesh(MPI_Comm, Obj<ALE::Mesh>&, Options *);
PetscErrorCode CreatePartition(const Obj<ALE::Mesh>&);
PetscErrorCode CreateOdd(const Obj<ALE::Mesh>&);
PetscErrorCode DistributeMesh(Obj<ALE::Mesh>&, Options *);
PetscErrorCode OutputVTK(const Obj<ALE::Mesh>&, Options *);
PetscErrorCode OutputMesh(const Obj<ALE::Mesh>&, Options *);

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
    Obj<ALE::Mesh> mesh;

    ierr = ProcessOptions(comm, &options);CHKERRQ(ierr);
    ierr = CreateMesh(comm, mesh, &options);CHKERRQ(ierr);
    ierr = DistributeMesh(mesh, &options);CHKERRQ(ierr);
    ierr = OutputVTK(mesh, &options);CHKERRQ(ierr);
    ierr = OutputMesh(mesh, &options);CHKERRQ(ierr);
  } catch (ALE::Exception e) {
    std::cout << e << std::endl;
  }
  ierr = PetscFinalize();CHKERRQ(ierr);
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
  options->partition      = PETSC_TRUE;
  options->material       = PETSC_FALSE;
  options->odd            = PETSC_FALSE;

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
    ierr = PetscOptionsTruth("-partition", "Create the partition field", "ex1.c", options->partition, &options->partition, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-material", "Create the material field", "ex1.c", options->material, &options->material, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-odd", "Create the odd field", "ex1.c", options->odd, &options->odd, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  options->inputFileType = (FileType) inputFt;
  if (setOutputType) {
    options->outputFileType = (FileType) outputFt;
  } else {
    options->outputFileType = options->inputFileType;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateMesh"
PetscErrorCode CreateMesh(MPI_Comm comm, Obj<ALE::Mesh>& mesh, Options *options)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ALE::LogStage stage = ALE::LogStageRegister("MeshCreation");
  ALE::LogStagePush(stage);
  ierr = PetscPrintf(comm, "Creating mesh\n");CHKERRQ(ierr);
  if (options->inputFileType == PCICE) {
    mesh = ALE::PCICE::Builder::readMesh(comm, options->dim, options->baseFilename, options->useZeroBase, options->interpolate, options->debug);
  } else if (options->inputFileType == PYLITH) {
    mesh = ALE::PyLith::Builder::readMesh(comm, options->dim, options->baseFilename, options->useZeroBase, options->interpolate, options->debug);
  } else {
    SETERRQ1(PETSC_ERR_ARG_WRONG, "Invalid mesh input type: %d", options->inputFileType);
  }
  ALE::LogStagePop(stage);
  Obj<ALE::Mesh::topology_type> topology = mesh->getTopologyNew();
  ierr = PetscPrintf(comm, "  Read %d elements\n", topology->heightStratum(0, 0)->size());CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "  Read %d vertices\n", topology->depthStratum(0, 0)->size());CHKERRQ(ierr);
  if (options->debug) {
    topology->view("Serial topology");
  }
  if (options->odd) {
    ierr = CreateOdd(mesh);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DistributeMesh"
PetscErrorCode DistributeMesh(Obj<ALE::Mesh>& mesh, Options *options)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (options->distribute) {
    ALE::LogStage stage = ALE::LogStageRegister("MeshDistribution");
    ALE::LogStagePush(stage);
    ierr = PetscPrintf(mesh->comm(), "Distributing mesh\n");CHKERRQ(ierr);
    mesh = ALE::New::Distribution<ALE::Mesh::topology_type>::redistributeMesh(mesh, std::string(options->partitioner));
    ALE::LogStagePop(stage);
  }
  if (options->partition) {
    ierr = CreatePartition(mesh);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "OutputVTK"
PetscErrorCode OutputVTK(const Obj<ALE::Mesh>& mesh, Options *options)
{
  PetscViewer    viewer;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (options->outputVTK) {
    ALE::LogStage stage = ALE::LogStageRegister("VTKOutput");
    ALE::LogStagePush(stage);
    ierr = PetscPrintf(mesh->comm(), "Creating VTK mesh file\n");CHKERRQ(ierr);
    ierr = PetscViewerCreate(mesh->comm(), &viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer, "testMesh.vtk");CHKERRQ(ierr);
    ierr = MeshView_Sieve(mesh, viewer);CHKERRQ(ierr);
    if (options->partition) {
      ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_VTK_CELL);CHKERRQ(ierr);
      ierr = FieldView_Sieve(mesh, "partition", viewer);CHKERRQ(ierr);
      ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    }
    if (options->material) {
      ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_VTK_CELL);CHKERRQ(ierr);
      ierr = FieldView_Sieve(mesh, "material", viewer);CHKERRQ(ierr);
      ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    }
    if (options->odd) {
      mesh->getBCSection("odd")->view("Odd cells");
/*       ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_VTK_CELL);CHKERRQ(ierr); */
/*       ierr = FieldView_Sieve(mesh, "odd", viewer);CHKERRQ(ierr); */
/*       ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr); */
    }
    ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
    ALE::LogStagePop(stage);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "OutputMesh"
PetscErrorCode OutputMesh(const Obj<ALE::Mesh>& mesh, Options *options)
{
  PetscViewer    viewer;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (options->output) {
    ALE::LogStage stage = ALE::LogStageRegister("MeshOutput");
    ALE::LogStagePush(stage);
    ierr = PetscPrintf(mesh->comm(), "Creating original format mesh file\n");CHKERRQ(ierr);
    ierr = PetscViewerCreate(mesh->comm(), &viewer);CHKERRQ(ierr);
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
    ierr = MeshView_Sieve(mesh, viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
    ALE::LogStagePop(stage);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreatePartition"
/*
  Creates a field whose value is the processor rank on each element
*/
PetscErrorCode CreatePartition(const Obj<ALE::Mesh>& mesh)
{
  const Obj<ALE::Mesh::section_type>&       partition = mesh->getSection("partition");
  const ALE::Mesh::section_type::patch_type patch     = 0;
  const ALE::Mesh::section_type::value_type rank      = mesh->commRank();

  PetscFunctionBegin;
  ALE_LOG_EVENT_BEGIN;
  partition->setFiberDimensionByHeight(patch, 0, 1);
  partition->allocate();
  const Obj<ALE::Mesh::topology_type::label_sequence>& cells = partition->getTopology()->heightStratum(patch, 0);
  ALE::Mesh::topology_type::label_sequence::iterator   end   = cells->end();

  for(ALE::Mesh::topology_type::label_sequence::iterator c_iter = cells->begin(); c_iter != end; ++c_iter) {
    partition->updatePoint(patch, *c_iter, &rank);
  }
  ALE_LOG_EVENT_END;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateOdd"
/*
  Creates a field whose value is the element number on each element with an odd number
*/
PetscErrorCode CreateOdd(const Obj<ALE::Mesh>& mesh)
{
  const Obj<ALE::Mesh::bc_section_type>&    odd   = mesh->getBCSection("odd");
  const ALE::Mesh::section_type::patch_type patch = 0;

  PetscFunctionBegin;
  ALE_LOG_EVENT_BEGIN;
  const Obj<ALE::Mesh::topology_type::label_sequence>& cells = odd->getTopology()->heightStratum(patch, 0);
  ALE::Mesh::topology_type::label_sequence::iterator   begin = cells->begin();
  ALE::Mesh::topology_type::label_sequence::iterator   end   = cells->end();

  for(ALE::Mesh::topology_type::label_sequence::iterator c_iter = begin; c_iter != end; ++c_iter) {
    const int num = *c_iter;

    if (num%2) {
      odd->setFiberDimension(patch, num, num%3+1);
    }
  }
  odd->allocate();
  for(ALE::Mesh::topology_type::label_sequence::iterator c_iter = begin; c_iter != end; ++c_iter) {
    const int num = *c_iter;
    int       val[3];

    if (num%2) {
      for(int n = 0; n <= num%3; n++) val[n] = num+n;
      odd->updatePoint(patch, num, val);
    }
  }
  ALE_LOG_EVENT_END;
  PetscFunctionReturn(0);
}
