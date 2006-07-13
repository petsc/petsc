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

#include <Mesh.hh>
#include "petscmesh.h"
#include "petscviewer.h"
#include <stdlib.h>
#include <string.h>

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
  PetscTruth     interpolate;        // Construct missing elements of the mesh

  Vec            partition;          // Field over cells indicating process number
  Vec            material;           // Field over cells indicating material type
} Options;

EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshView_Sieve_Newer(ALE::Obj<ALE::Mesh> mesh, PetscViewer viewer);
PetscErrorCode ProcessOptions(MPI_Comm, Options *);
PetscErrorCode CreateMesh(MPI_Comm, ALE::Obj<ALE::Mesh>&, Options *);
PetscErrorCode CreatePartitionVector(ALE::Obj<ALE::Mesh>, Vec *);
PetscErrorCode CreateFieldVector(ALE::Obj<ALE::Mesh>, const char[], Vec *);
PetscErrorCode DistributeMesh(ALE::Obj<ALE::Mesh>&, Options *);
PetscErrorCode OutputVTK(const ALE::Obj<ALE::Mesh>&, Options *);
PetscErrorCode OutputMesh(const ALE::Obj<ALE::Mesh>&, Options *);

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
    ALE::Obj<ALE::Mesh> mesh;

    ierr = ProcessOptions(comm, &options);CHKERRQ(ierr);
    ierr = CreateMesh(comm, mesh, &options);CHKERRQ(ierr);
#if 0
    ALE::LogStage stage = ALE::LogStageRegister("MeshCreation");
    ALE::LogStagePush(stage);
    ierr = PetscPrintf(comm, "Creating mesh\n");CHKERRQ(ierr);
    if (options.inputFileType == PCICE) {
      mesh = ALE::PCICEBuilder::createNew(comm, options.baseFilename, options.dim, options.useZeroBase, options.interpolate, options.debug);
    } else if (options.inputFileType == PYLITH) {
      mesh = ALE::PyLithBuilder::createNew(comm, options.baseFilename, options.interpolate, options.debug);
    }
    ALE::LogStagePop(stage);
    ALE::Obj<ALE::Mesh::sieve_type> topology = mesh->getTopology();
    ierr = PetscPrintf(comm, "  Read %d elements\n", topology->heightStratum(0)->size());CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "  Read %d vertices\n", topology->depthStratum(0)->size());CHKERRQ(ierr);
    if (options.debug) {
      mesh->getTopology()->view("Serial topology");
    }
#endif
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
  options->useZeroBase    = PETSC_FALSE;
  options->inputFileType  = PCICE;
  options->outputFileType = PCICE;
  ierr = PetscStrcpy(options->baseFilename, "data/ex1_2d");CHKERRQ(ierr);
  options->distribute     = PETSC_TRUE;
  options->output         = PETSC_TRUE;
  options->outputLocal    = PETSC_FALSE;
  options->outputVTK      = PETSC_TRUE;
  options->distribute     = PETSC_TRUE;
  options->interpolate    = PETSC_TRUE;
  options->partition      = PETSC_NULL;
  options->material       = PETSC_NULL;

  inputFt  = (PetscInt) options->inputFileType;
  outputFt = (PetscInt) options->outputFileType;

  ierr = PetscOptionsBegin(comm, "", "Options for mesh loading", "DMMG");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-debug", "The debugging level", "ex1.c", 0, &options->debug, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex1.c", 2, &options->dim, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-use_zero_base", "Use zero-based indexing", "ex1.c", PETSC_FALSE, &options->useZeroBase, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEList("-file_type", "Type of input files", "ex1.c", fileTypes, 2, fileTypes[0], &inputFt, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEList("-output_file_type", "Type of output files", "ex1.c", fileTypes, 2, fileTypes[0], &outputFt, &setOutputType);CHKERRQ(ierr);
    ierr = PetscOptionsString("-base_file", "The base filename for mesh files", "ex33.c", "ex1", options->baseFilename, 2048, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-output", "Output the mesh", "ex1.c", PETSC_TRUE, &options->output, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-output_local", "Output the local form of the mesh", "ex1.c", PETSC_FALSE, &options->outputLocal, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-output_vtk", "Output the mesh in VTK", "ex1.c", PETSC_TRUE, &options->outputVTK, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-distribute", "Distribute the mesh among processes", "ex1.c", PETSC_TRUE, &options->distribute, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-interpolate", "Construct missing elements of the mesh", "ex1.c", PETSC_TRUE, &options->interpolate, PETSC_NULL);CHKERRQ(ierr);
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
PetscErrorCode CreateMesh(MPI_Comm comm, ALE::Obj<ALE::Mesh>& mesh, Options *options)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ALE::LogStage stage = ALE::LogStageRegister("MeshCreation");
  ALE::LogStagePush(stage);
  ierr = PetscPrintf(comm, "Creating mesh\n");CHKERRQ(ierr);
  if (options->inputFileType == PCICE) {
    //mesh = ALE::PCICEBuilder::createNew(comm, options->baseFilename, options->dim, options->useZeroBase, options->interpolate, options->debug);
    mesh = new ALE::Mesh(comm, 1, options->debug);
  } else if (options->inputFileType == PYLITH) {
    mesh = ALE::PyLithBuilder::createNew(comm, options->baseFilename, options->interpolate, options->debug);
  }
  ALE::LogStagePop(stage);
  ALE::Obj<ALE::Mesh::sieve_type> topology = mesh->getTopology();
  ierr = PetscPrintf(comm, "  Read %d elements\n", topology->heightStratum(0)->size());CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "  Read %d vertices\n", topology->depthStratum(0)->size());CHKERRQ(ierr);
  if (options->debug) {
    mesh->getTopology()->view("Serial topology");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DistributeMesh"
PetscErrorCode DistributeMesh(ALE::Obj<ALE::Mesh>& mesh, Options *options)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (options->distribute) {
    ALE::LogStage stage = ALE::LogStageRegister("MeshDistribution");
    ALE::LogStagePush(stage);
    ierr = PetscPrintf(mesh->comm(), "Distributing mesh\n");CHKERRQ(ierr);
    mesh = mesh->distribute();
    ierr = CreatePartitionVector(mesh, &options->partition);CHKERRQ(ierr);
    ierr = CreateFieldVector(mesh, "material", &options->material);CHKERRQ(ierr);
    ALE::LogStagePop(stage);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "OutputVTK"
PetscErrorCode OutputVTK(const ALE::Obj<ALE::Mesh>& mesh, Options *options)
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
    ierr = MeshView_Sieve_Newer(mesh, viewer);CHKERRQ(ierr);
    if (options->partition) {
      ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_VTK_CELL);CHKERRQ(ierr);
      ierr = VecView(options->partition, viewer);CHKERRQ(ierr);
      ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    }
    if (options->material) {
      ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_VTK_CELL);CHKERRQ(ierr);
      ierr = VecView(options->material, viewer);CHKERRQ(ierr);
      ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    }
    ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
    ALE::LogStagePop(stage);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "OutputMesh"
PetscErrorCode OutputMesh(const ALE::Obj<ALE::Mesh>& mesh, Options *options)
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
    ierr = MeshView_Sieve_Newer(mesh, viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
    ALE::LogStagePop(stage);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreatePartitionVector"
/*
  Creates a vector whose value is the processor rank on each element
*/
PetscErrorCode CreatePartitionVector(ALE::Obj<ALE::Mesh> mesh, Vec *partition)
{
  PetscScalar   *array;
  int            rank = mesh->commRank();
  PetscInt       n, i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ALE_LOG_EVENT_BEGIN;
  ierr = MeshCreateVector(mesh, mesh->getBundle(mesh->getTopology()->depth()), partition);CHKERRQ(ierr);
  ierr = VecSetBlockSize(*partition, 1);CHKERRQ(ierr);
  ierr = VecGetLocalSize(*partition, &n);CHKERRQ(ierr);
  ierr = VecGetArray(*partition, &array);CHKERRQ(ierr);
  for(i = 0; i < n; i++) {
    array[i] = rank;
  }
  ierr = VecRestoreArray(*partition, &array);CHKERRQ(ierr);
  ALE_LOG_EVENT_END;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateFieldVector"
/*
  Creates a vector whose value is the field value on each element
*/
PetscErrorCode CreateFieldVector(ALE::Obj<ALE::Mesh> mesh, const char fieldName[], Vec *fieldVec)
{
  if (!mesh->hasField(fieldName)) {
    *fieldVec = PETSC_NULL;
    return(0);
  }
  ALE::Obj<ALE::Mesh::field_type> field = mesh->getField(fieldName);
  ALE::Mesh::field_type::patch_type patch;
  VecScatter     injection;
  Vec            locField;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ALE_LOG_EVENT_BEGIN;
  ierr = MeshCreateVector(mesh, mesh->getBundle(mesh->getTopology()->depth()), fieldVec);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *fieldVec, fieldName);CHKERRQ(ierr);
  ierr = MeshGetGlobalScatter(mesh, fieldName, *fieldVec, &injection); CHKERRQ(ierr);

  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, field->getSize(patch), field->restrict(patch), &locField);CHKERRQ(ierr);
  ierr = VecScatterBegin(locField, *fieldVec, INSERT_VALUES, SCATTER_FORWARD, injection);CHKERRQ(ierr);
  ierr = VecScatterEnd(locField, *fieldVec, INSERT_VALUES, SCATTER_FORWARD, injection);CHKERRQ(ierr);
  ierr = VecDestroy(locField);CHKERRQ(ierr);
  ALE_LOG_EVENT_END;
  PetscFunctionReturn(0);
}
