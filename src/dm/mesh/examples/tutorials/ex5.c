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
#include <values.h>

EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshView_Sieve_Newer(ALE::Obj<ALE::Mesh> mesh, PetscViewer viewer);
PetscErrorCode CreatePartitionVector(ALE::Obj<ALE::Mesh>, Vec *);
PetscErrorCode CreateFieldVector(ALE::Obj<ALE::Mesh>, const char[], int depth, Vec *);
PetscErrorCode CreateSpacingFunction(ALE::Obj<ALE::Mesh>);

typedef enum {PCICE, PYLITH} FileType;

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  MPI_Comm       comm;
  PetscViewer    viewer;
  Vec            partition, material, spacing;
  char           baseFilename[2048];
  PetscTruth     useZeroBase;
  const char    *fileTypes[2] = {"pcice", "pylith"};
  FileType       fileType, outputFileType;
  PetscTruth     distribute, interpolate, outputLocal, outputVTK, setOutputType;
  PetscInt       dim, ft, outputFt;
  int            verbosity;
  int            debug;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(comm, "", "Options for mesh loading", "DMMG");
    debug = 0;
    ierr = PetscOptionsInt("-debug", "The debugging flag", "ex1.c", 0, &debug, PETSC_NULL);CHKERRQ(ierr);
    dim  = 2;
    ierr = PetscOptionsInt("-dim", "The mesh dimension", "ex1.c", 2, &dim, PETSC_NULL);CHKERRQ(ierr);
    useZeroBase = PETSC_FALSE;
    ierr = PetscOptionsTruth("-use_zero_base", "Use zero-based indexing", "ex1.c", PETSC_FALSE, &useZeroBase, PETSC_NULL);CHKERRQ(ierr);
    ft   = (PetscInt) PCICE;
    ierr = PetscOptionsEList("-file_type", "Type of input files", "ex1.c", fileTypes, 2, fileTypes[0], &ft, PETSC_NULL);CHKERRQ(ierr);
    fileType = (FileType) ft;
    outputFt = (PetscInt) PCICE;
    ierr = PetscOptionsEList("-output_file_type", "Type of output files", "ex1.c", fileTypes, 2, fileTypes[0], &outputFt, &setOutputType);CHKERRQ(ierr);
    if (setOutputType) {
      outputFileType = (FileType) outputFt;
    } else {
      outputFileType = fileType;
    }
    ierr = PetscStrcpy(baseFilename, "data/ex1_2d");CHKERRQ(ierr);
    ierr = PetscOptionsString("-base_file", "The base filename for mesh files", "ex33.c", "ex1", baseFilename, 2048, PETSC_NULL);CHKERRQ(ierr);
    distribute = PETSC_TRUE;
    ierr = PetscOptionsTruth("-distribute", "Distribute the mesh among processes", "ex1.c", PETSC_TRUE, &distribute, PETSC_NULL);CHKERRQ(ierr);
    outputLocal = PETSC_FALSE;
    ierr = PetscOptionsTruth("-output_local", "Output the local form of the mesh", "ex1.c", PETSC_FALSE, &outputLocal, PETSC_NULL);CHKERRQ(ierr);
    outputVTK = PETSC_TRUE;
    ierr = PetscOptionsTruth("-output_vtk", "Output the mesh in VTK", "ex1.c", PETSC_TRUE, &outputVTK, PETSC_NULL);CHKERRQ(ierr);
    interpolate = PETSC_TRUE;
    ierr = PetscOptionsTruth("-interpolate", "Construct missing elements of the mesh", "ex1.c", PETSC_TRUE, &interpolate, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  ierr = PetscOptionsBegin(comm, "", "Debugging options", "ALE");
    verbosity = 0;
    ierr = PetscOptionsInt("-verbosity", "Verbosity level", "ex1.c", 0, &verbosity, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); 
  comm = PETSC_COMM_WORLD;

  ALE::Obj<ALE::Mesh> mesh;

  try {
    ALE::LogStage stage = ALE::LogStageRegister("MeshCreation");
    ALE::LogStagePush(stage);
    ierr = PetscPrintf(comm, "Creating mesh\n");CHKERRQ(ierr);
    if (fileType == PCICE) {
      mesh = ALE::PCICEBuilder::createNew(comm, baseFilename, dim, useZeroBase, interpolate, debug);
    } else if (fileType == PYLITH) {
      mesh = ALE::PyLithBuilder::createNew(comm, baseFilename, interpolate, debug);
    }
    ALE::LogStagePop(stage);
    ALE::Obj<ALE::Mesh::sieve_type> topology = mesh->getTopology();
    ierr = PetscPrintf(comm, "  Read %d elements\n", topology->heightStratum(0)->size());CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "  Read %d vertices\n", topology->depthStratum(0)->size());CHKERRQ(ierr);
    if (debug) {mesh->getTopology()->view("Serial topology");}

    stage = ALE::LogStageRegister("Mesh Spacing");
    ALE::LogStagePush(stage);
    ierr = CreateSpacingFunction(mesh);CHKERRQ(ierr);
    mesh->getField("spacing")->view("Mesh spacing");
    ALE::LogStagePop(stage);

    if (distribute) {
      stage = ALE::LogStageRegister("MeshDistribution");
      ALE::LogStagePush(stage);
      ierr = PetscPrintf(comm, "Distributing mesh\n");CHKERRQ(ierr);
      mesh = mesh->distribute();
      ierr = CreatePartitionVector(mesh, &partition);CHKERRQ(ierr);
      ierr = CreateFieldVector(mesh, "material", mesh->getTopology()->depth(), &material);CHKERRQ(ierr);
      ierr = CreateFieldVector(mesh, "spacing", 0, &spacing);CHKERRQ(ierr);
      ALE::LogStagePop(stage);
    }

    stage = ALE::LogStageRegister("MeshOutput");
    ALE::LogStagePush(stage);
    if (outputVTK) {
      ierr = PetscPrintf(comm, "Creating VTK mesh file\n");CHKERRQ(ierr);
      ierr = PetscViewerCreate(comm, &viewer);CHKERRQ(ierr);
      ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
      ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
      ierr = PetscViewerFileSetName(viewer, "testMesh.vtk");CHKERRQ(ierr);
      ierr = MeshView_Sieve_Newer(mesh, viewer);CHKERRQ(ierr);
      if (spacing) {
        ierr = VecView(spacing, viewer);CHKERRQ(ierr);
      }
      ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK_CELL);CHKERRQ(ierr);
      ierr = VecView(partition, viewer);CHKERRQ(ierr);
      if (material) {
        ierr = VecView(material, viewer);CHKERRQ(ierr);
      }
      ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
    }

    ierr = PetscPrintf(comm, "Creating original format mesh file\n");CHKERRQ(ierr);
    ierr = PetscViewerCreate(comm, &viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
    if (outputFileType == PCICE) {
      ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_PCICE);CHKERRQ(ierr);
      ierr = PetscViewerFileSetName(viewer, "testMesh.lcon");CHKERRQ(ierr);
    } else if (outputFileType == PYLITH) {
      if (outputLocal) {
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
  } catch (ALE::Exception e) {
    std::cout << e << std::endl;
  }
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateSpacing Function"
/*
  Creates the initial nearest neighbor spacing function
*/
PetscErrorCode CreateSpacingFunction(ALE::Obj<ALE::Mesh> mesh)
{
  ALE::Obj<ALE::Mesh::field_type> spacing  = mesh->getField("spacing");
  ALE::Obj<ALE::Mesh::field_type> coords   = mesh->getCoordinates();
  ALE::Obj<ALE::Mesh::sieve_type> topology = mesh->getTopology();
  ALE::Mesh::field_type::patch_type patch;

  PetscFunctionBegin;
  ALE::Obj<ALE::Mesh::sieve_type::traits::depthSequence> vertices = topology->depthStratum(0);
  int                                                    dim      = coords->getFiberDimension(patch, *vertices->begin());

  spacing->setPatch(vertices, patch);
  spacing->setFiberDimensionByDepth(patch, 0, 1);
  spacing->orderPatches();
  for(ALE::Mesh::sieve_type::traits::depthSequence::iterator v_iter = vertices->begin(); v_iter != vertices->end(); ++v_iter) {
    ALE::Obj<ALE::Mesh::sieve_type::traits::supportSequence> support = topology->support(*v_iter);
    const double *vCoords = coords->restrict(patch, *v_iter);
    double        minDist = MAXDOUBLE;

    for(ALE::Mesh::sieve_type::traits::supportSequence::iterator s_iter = support->begin(); s_iter != support->end(); ++s_iter) {
      ALE::Obj<ALE::Mesh::sieve_type::traits::coneSequence> neighbors = topology->cone(*s_iter);

      for(ALE::Mesh::sieve_type::traits::coneSequence::iterator n_iter = neighbors->begin(); n_iter != neighbors->end(); ++n_iter) {
        if (*v_iter != *n_iter) {
          const double *nCoords = coords->restrict(patch, *n_iter);
          double        dist    = 0.0;

          for(int d = 0; d < dim; d++) {
            dist += (vCoords[d] - nCoords[d])*(vCoords[d] - nCoords[d]);
          }
          if (dist < minDist) minDist = dist;
        }
      }
    }
    spacing->update(patch, *v_iter, &minDist);
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
PetscErrorCode CreateFieldVector(ALE::Obj<ALE::Mesh> mesh, const char fieldName[], int depth, Vec *fieldVec)
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
  ierr = MeshCreateVector(mesh, mesh->getBundle(depth), fieldVec);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *fieldVec, fieldName);CHKERRQ(ierr);
  ierr = MeshGetGlobalScatter(mesh, fieldName, *fieldVec, &injection);CHKERRQ(ierr);

  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, field->getSize(patch), field->restrict(patch), &locField);CHKERRQ(ierr);
  ierr = VecScatterBegin(injection, locField, *fieldVec, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(injection, locField, *fieldVec, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecDestroy(locField);CHKERRQ(ierr);
  ALE_LOG_EVENT_END;
  PetscFunctionReturn(0);
}
