/*T
   Concepts: Mesh^generating a mesh
   Concepts: Mesh^partitioning a mesh
   Concepts: Mesh^viewing a mesh
   Concepts: Applications^radially-symmetric ion channel 
   Processors: n
T*/

/*
  Generate a 2D triangular mesh of a radially-symmetric slide of a schematic ion channel using the builtin mesh generator.

  Partition the mesh and distribute it to each process.

  Output the mesh in VTK format with a scalar field indicating
  the rank of the process owning each cell.
*/

static char help[] = "Generates, partitions, and outputs an unstructured 2D mesh of a radially-symmetric simple ion channel.\n\n";

#include <Mesh.hh>
#include "petscmesh.h"
#include "petscviewer.h"

EXTERN PetscErrorCode PETSCDM_DLLEXPORT MeshView_Sieve_Newer(ALE::Obj<ALE::Two::Mesh> mesh, PetscViewer viewer);
PetscErrorCode CreateMeshBoundary(ALE::Obj<ALE::Two::Mesh>);
PetscErrorCode CreatePartitionVector(ALE::Obj<ALE::Two::Mesh>, Vec *);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  MPI_Comm       comm;
  Vec            partition;
  PetscViewer    viewer;
  PetscInt       dim, debug;
  PetscReal      refinementLimit;
  PetscTruth     interpolate;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr);
  comm = PETSC_COMM_WORLD;
  ierr = PetscOptionsBegin(comm, "", "Options for channel mesh generation", "DMMG");CHKERRQ(ierr);
    debug = 0;
    ierr = PetscOptionsInt("-debug", "The debugging flag", "ex4.c", 0, &debug, PETSC_NULL);CHKERRQ(ierr);
    interpolate = PETSC_TRUE;
    ierr = PetscOptionsTruth("-interpolate", "Construct missing elements of the mesh", "ex4.c", PETSC_TRUE, &interpolate, PETSC_NULL);CHKERRQ(ierr);
    refinementLimit = 0.0;
    ierr = PetscOptionsReal("-refinement_limit", "The area of the largest triangle in the mesh", "ex4.c", 1.0, &refinementLimit, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  /* We are dealing with a 2D problem */
  dim  = 2;
  ALE::Obj<ALE::Two::Mesh> meshBoundary = ALE::Two::Mesh(comm, dim-1, debug);
  ALE::Obj<ALE::Two::Mesh> mesh;

  try {
    ALE::LogStage stage = ALE::LogStageRegister("MeshCreation");
    ALE::LogStagePush(stage);
    ierr = PetscPrintf(comm, "Generating mesh\n");CHKERRQ(ierr);
    /* Generate the boundary */
    ierr = CreateMeshBoundary(meshBoundary);CHKERRQ(ierr);
    if (debug) {
      ierr = PetscViewerCreate(comm, &viewer);CHKERRQ(ierr);
      ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
      ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
      ierr = PetscViewerFileSetName(viewer, "channelBoundary2D.vtk");CHKERRQ(ierr);
      ierr = MeshView_Sieve_Newer(meshBoundary, viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
    }
    ierr = PetscPrintf(comm, "  Generated %d boundary edges\n",    meshBoundary->getTopology()->depthStratum(1)->size());CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "  Generated %d boundary vertices\n", meshBoundary->getTopology()->depthStratum(0)->size());CHKERRQ(ierr);

    /* Generate the interior from the boundary */
    mesh = ALE::Two::Generator::generate(meshBoundary, interpolate);
    ALE::Obj<ALE::Two::Mesh::sieve_type> topology = mesh->getTopology();
    ierr = PetscPrintf(comm, "  Generated %d elements\n", topology->heightStratum(0)->size());CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "  Generated %d vertices\n", topology->depthStratum(0)->size());CHKERRQ(ierr);

    /* Distribute the mesh */
    stage = ALE::LogStageRegister("MeshDistribution");
    ALE::LogStagePush(stage);
    ierr = PetscPrintf(comm, "Distributing mesh\n");CHKERRQ(ierr);
    mesh = mesh->distribute();
    ALE::LogStagePop(stage);

    /* Refine the mesh */
    if (refinementLimit > 0.0) {
      stage = ALE::LogStageRegister("MeshRefine");
      ALE::LogStagePush(stage);
      ierr = PetscPrintf(comm, "Refining mesh\n");CHKERRQ(ierr);
      mesh = ALE::Two::Generator::refine(mesh, refinementLimit, interpolate);
      ALE::LogStagePop(stage);
      ierr = PetscPrintf(comm, "  Generated %d elements\n", mesh->getTopology()->heightStratum(0)->size());CHKERRQ(ierr);
      ierr = PetscPrintf(comm, "  Generated %d vertices\n", mesh->getTopology()->depthStratum(0)->size());CHKERRQ(ierr);
    }
    /* Output the mesh */
    stage = ALE::LogStageRegister("MeshOutput");
    ALE::LogStagePush(stage);
    ierr = CreatePartitionVector(mesh, &partition);CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "Creating VTK channel mesh file\n");CHKERRQ(ierr);
    ierr = PetscViewerCreate(comm, &viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer, "channelMesh2D.vtk");CHKERRQ(ierr);
    ierr = MeshView_Sieve_Newer(mesh, viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK_CELL);CHKERRQ(ierr);
    ierr = VecView(partition, viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
    ALE::LogStagePop(stage);
  } catch (ALE::Exception e) {
    std::cout << e.msg() << std::endl;
  }
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateMeshBoundary"
/*
  2D radially-symmetric channel boundary:
             28                         29      35         43
              V                          V      V          V 
    2----------------------------------3-----4----12--------------------------------13
    |                                  |     |     |                                 |
    |                                  |     |     |                                 |
    |                               34>|     |     | <36                             |
    |                                  |     |     |                                 |
 27>|                                  |  30>|     |                                 |
    |                                  8     |     11                            42> |
    |                               33>\     |     / <37                             |
    |                                   7 31 | 39 10                                 |
    |                                32> \ V | V / <38                               |
    |                                     6--5--9      41                            |
    |                                        |<40      V                             |
    1----------------------------------------O--------------------------------------14
    |          ^                             |<50                                    |
    |         57                         25-20--19                                   |
    |                               56>  / ^ | ^ \ <48                               |
    |                                   24 51| 49 18                                 |
    |                              55> /     |     \ <47                             |
    | <58                              23    |    17                             44> |
    |                                  | 52> |     |                                 |
    |                                  |     |     |                                 |
    |                              54> |     |     |<46                              |
    |       59(XX)                     | 53  |     |        45                       |
    |        V                         |  V  |     |        V                        |
    26(X)-----------------------------22----21-----16-------------------------------15

    (X) denotes the last vertex, (XX) denotes the last edge
*/
PetscErrorCode CreateMeshBoundary(ALE::Obj<ALE::Two::Mesh> mesh)
{
  ALE::Obj<ALE::Two::Mesh::sieve_type> topology = mesh->getTopology();
  PetscScalar       coords[54] =  {/*O*/      0.0,      0.0, 
                                   /*1*/   -112.5,      0.0, 
                                   /*2*/   -112.5,     50.0, 
                                   /*3*/    -12.5,     50.0,
                                   /*4*/      0.0,     50.0,
                                   /*5*/      0.0,      3.0,
                                   /*6*/     -2.5,      3.0,
                                   /*7*/   -35.0/6.0,   1.0,
                                   /*8*/    -12.5,     15.0,
                                   /*9*/      2.5,      3.0, 
                                   /*10*/   35.0/6.0,   1.0, 
                                   /*11*/    12.5,     15.0,
                                   /*12*/    12.5,     50.0,
                                   /*13*/   112.5,     50.0, 
                                   /*14*/   112.5,      0.0, 
                                   /*15*/   112.5,    -50.0, 
                                   /*16*/    12.5,    -50.0,
                                   /*17*/    12.5,    -15.0, 
                                   /*18*/   35.0/6.0,  -1.0,  
                                   /*19*/     2.5,     -3.0, 
                                   /*20*/     0.0,     -3.0,
                                   /*21*/     0.0,    -50.0,
                                   /*22*/   -12.5,    -50.0,
                                   /*23*/   -12.5,    -15.0,
                                   /*24*/  -35.0/6.0,  -1.0,
                                   /*25*/    -2.5,     -3.0,
                                   /*26*/  -112.5,    -50.0};
  PetscInt    connectivity[66] = {1, 2,
                                  2, 3,
                                  3, 4,
                                  4, 5,
                                  5, 6,
                                  6, 7,
                                  7, 8,
                                  8, 3,
                                  4, 12,
                                  11,12,
                                  10,11,
                                  9, 10,
                                  5,  9,
                                  0,  5,
                                  0, 14,
                                  13,14,
                                  12,13,
                                  14,15,
                                  15,16,
                                  16,17,
                                  17,18,
                                  18,19,
                                  19,20,
                                  0, 20,
                                  20,25,
                                  20,21,
                                  21,22,
                                  22,23,
                                  23,24,
                                  24,25,
                                   0, 1,
                                   1,26,
                                  22,26};
  ALE::Two::Mesh::point_type vertices[27];

  PetscFunctionBegin;
  PetscInt order = 0;
  if (mesh->commRank() == 0) {
    ALE::Two::Mesh::point_type edge;

    /* Create topology and ordering */
    for(int v = 0; v < 27; v++) {
      vertices[v] = ALE::Two::Mesh::point_type(0, v);
    }
    for(int e = 27; e < 60; e++) {
      int ee = e - 27;
      edge = ALE::Two::Mesh::point_type(0, e);
      topology->addArrow(vertices[connectivity[2*ee]],  edge, order++);
      topology->addArrow(vertices[connectivity[2*ee]], edge, order++);
    }
  }
  topology->stratify();
  mesh->createVertexBundle(33, connectivity);
  mesh->createSerialCoordinates(2, 0, coords);
  /* Create boundary conditions */
  /* Use ex2 as template */
  PetscFunctionReturn(0);
}



#undef __FUNCT__
#define __FUNCT__ "CreatePartitionVector"
/*
  Creates a vector whose value is the processor rank on each element
*/
PetscErrorCode CreatePartitionVector(ALE::Obj<ALE::Two::Mesh> mesh, Vec *partition)
{
  PetscScalar   *array;
  int            rank = mesh->commRank();
  PetscInt       n, i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ALE_LOG_EVENT_BEGIN;
  ierr = MeshCreateVector(mesh, mesh->getBundle(mesh->getDimension()), partition);CHKERRQ(ierr);
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
