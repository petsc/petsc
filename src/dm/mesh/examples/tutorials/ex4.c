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
PetscErrorCode CreateDielectricVector(ALE::Obj<ALE::Two::Mesh>, Vec *);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  MPI_Comm       comm;
  Vec            partition, dielectric;
  PetscViewer    viewer;
  PetscInt       dim, debug;
  PetscReal      refinementLimit;
  PetscTruth     interpolate, viewDielectric;
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
    viewDielectric = PETSC_FALSE;
    ierr = PetscOptionsTruth("-view_dielectric", "View the dielectric constant as a field", "ex4.c", PETSC_FALSE, &viewDielectric, PETSC_NULL);CHKERRQ(ierr);
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
    ierr = CreateDielectricVector(mesh, &dielectric);CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "Creating VTK channel mesh file\n");CHKERRQ(ierr);
    ierr = PetscViewerCreate(comm, &viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer, "channelMesh2D.vtk");CHKERRQ(ierr);
    ierr = MeshView_Sieve_Newer(mesh, viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK_CELL);CHKERRQ(ierr);
    if (viewDielectric) {
      ierr = VecView(dielectric, viewer);CHKERRQ(ierr);
    } else {
      ierr = VecView(partition, viewer);CHKERRQ(ierr);
    }
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
                    29                   30    31                 32
                     V                    V     V                  V
    2----------------------------------3-----4----12--------------------------------13
    |                                  |     |     |                                 |
    |                                  |     |     |                                 |
    |                               39>|     |     |<46                              |
    |                                  |     |     |                                 |
 28>|                                  |  59>|     |                                 |<33
    |                                  8     |     11                                |
    |                               40>\     |     /<45                              |
    |                                   7 42 |43 10                                  |
    |                                 41>\ V | V /<44                                |
    |               55                    6--5--9                 57                 |
    |                V                       |<56                  V                 |
    1----------------------------------------O--------------------------------------14
    |                                        |<58                                    |
    |                                    25-20--19                                   |
    |                                 49>/ ^ | ^ \<52                                |
    |                                   24 50| 51 18                                 |
    |                               48>/     |     \<53                              |
 27>|                                  23    |    17                                 |<34
    |                                  |  60>|     |                                 |
    |                                  |     |     |                                 |
    |                               47>|     |     |<54                              |
    |               38                 | 37  | 36  |              35                 |
    |                V                 |  V  |  V  |               V                 |
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
                                   /*7*/   -35.0/6.0,  10.0,
                                   /*8*/    -12.5,     15.0,
                                   /*9*/      2.5,      3.0, 
                                   /*10*/   35.0/6.0,  10.0, 
                                   /*11*/    12.5,     15.0,
                                   /*12*/    12.5,     50.0,
                                   /*13*/   112.5,     50.0, 
                                   /*14*/   112.5,      0.0, 
                                   /*15*/   112.5,    -50.0, 
                                   /*16*/    12.5,    -50.0,
                                   /*17*/    12.5,    -15.0, 
                                   /*18*/   35.0/6.0, -10.0,  
                                   /*19*/     2.5,     -3.0, 
                                   /*20*/     0.0,     -3.0,
                                   /*21*/     0.0,    -50.0,
                                   /*22*/   -12.5,    -50.0,
                                   /*23*/   -12.5,    -15.0,
                                   /*24*/  -35.0/6.0, -10.0,
                                   /*25*/    -2.5,     -3.0,
                                   /*26*/  -112.5,    -50.0};
  PetscInt    connectivity[68] = {26, 1, /* 1: phi = 0 */
                                  1, 2,  /* 1: phi = 0 */
                                  2, 3,  /* 2: grad phi = 0 */
                                  3, 4,  /* 2: grad phi = 0 */
                                  4, 12, /* 2: grad phi = 0 */
                                  12,13, /* 2: grad phi = 0 */
                                  13,14, /* 3: phi = V */
                                  14,15, /* 3: phi = V */
                                  15,16, /* 4: grad phi = 0 */
                                  16,21, /* 4: grad phi = 0 */
                                  21,22, /* 4: grad phi = 0 */
                                  22,26, /* 4: grad phi = 0 */
                                  3, 8,  /* 5: top lipid boundary */
                                  8, 7,  /* 5: top lipid boundary */
                                  7, 6,  /* 5: top lipid boundary */
                                  6, 5,  /* 5: top lipid boundary */
                                  5,  9, /* 5: top lipid boundary */
                                  9, 10, /* 5: top lipid boundary */
                                  10,11, /* 5: top lipid boundary */
                                  11,12, /* 5: top lipid boundary */
                                  22,23, /* 6: bottom lipid boundary */
                                  23,24, /* 6: bottom lipid boundary */
                                  24,25, /* 6: bottom lipid boundary */
                                  25,20, /* 6: bottom lipid boundary */
                                  20,19, /* 6: bottom lipid boundary */
                                  19,18, /* 6: bottom lipid boundary */
                                  18,17, /* 6: bottom lipid boundary */
                                  17,16, /* 6: bottom lipid boundary */
                                  0, 1,  /* 7: symmetry preservation */
                                  0, 5,  /* 7: symmetry preservation */
                                  0, 14, /* 7: symmetry preservation */
                                  0, 20, /* 7: symmetry preservation */
                                  4, 5,  /* 7: symmetry preservation */
                                  21,20  /* 7: symmetry preservation */
                                  };
  ALE::Two::Mesh::point_type vertices[27];

  PetscFunctionBegin;
  PetscInt order = 0;
  if (mesh->commRank() == 0) {
    ALE::Two::Mesh::point_type edge;

    /* Create topology and ordering */
    for(int v = 0; v < 27; v++) {
      vertices[v] = ALE::Two::Mesh::point_type(0, v);
    }
    for(int e = 27; e < 61; e++) {
      int ee = e - 27;
      edge = ALE::Two::Mesh::point_type(0, e);
      topology->addArrow(vertices[connectivity[2*ee]],   edge, order++);
      topology->addArrow(vertices[connectivity[2*ee+1]], edge, order++);
    }
  }
  topology->stratify();
  mesh->createVertexBundle(34, connectivity, 27);
  mesh->createSerialCoordinates(2, 0, coords);
  /* Create boundary conditions */
  if (mesh->commRank() == 0) {
    for(int e = 27; e < 29; e++) {
      topology->setMarker(ALE::Two::Mesh::point_type(0, e), 1);
    }
    for(int e = 29; e < 33; e++) {
      topology->setMarker(ALE::Two::Mesh::point_type(0, e), 2);
    }
    for(int e = 33; e < 35; e++) {
      topology->setMarker(ALE::Two::Mesh::point_type(0, e), 3);
    }
    for(int e = 35; e < 39; e++) {
      topology->setMarker(ALE::Two::Mesh::point_type(0, e), 4);
    }
    for(int e = 39; e < 47; e++) {
      topology->setMarker(ALE::Two::Mesh::point_type(0, e), 5);
    }
    for(int e = 47; e < 55; e++) {
      topology->setMarker(ALE::Two::Mesh::point_type(0, e), 6);
    }
  }
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

PetscErrorCode ComputeDielectric(double x, double y, double *epsilon) {
  double water   = 80.0;
  double lipid   = 40.0;
  double channel = 0.0;

  PetscFunctionBegin;
  *epsilon = -80;
  if ((x >= -112.5) && (x <= -12.5)) {
    // Left water bath
    *epsilon = water;
  } else if ((x >= 12.5) && (x <= 112.5)) {
    // Right water bath
    *epsilon = water;
  } else {
    if ((y >= 15.0) && (y <= 50.0)) {
      // Top lipid
      *epsilon = lipid;
    } else if ((y <= -15.0) && (y >= -50.0)) {
      // Bottom lipid
      *epsilon = lipid;
    } else {
      if ((x >= -12.5) && (x <= -2.5)) {
        // Left lipid or water
        if (x <= -35.0/6.0) {
          // Left parallelogram
          if (y >= 0.0) {
            // Top half
            double slope = (15.0 - 10.0)/(-12.5 + 35.0/6.0);

            if (y <= 15.0 + slope*(x + 12.5)) {
              // Middle water
              *epsilon = water;
            } else {
              // Middle lipid
              *epsilon = lipid;
            }
          } else {
            // Bottom half
            double slope = (-15.0 + 10.0)/(-12.5 + 35.0/6.0);

            if (y >= -15.0 + slope*(x + 12.5)) {
              // Middle water
              *epsilon = water;
            } else {
              // Middle lipid
              *epsilon = lipid;
            }
          }
        } else {
          // Right parallelogram
          if (y >= 0.0) {
            // Top half
            double slope = (10.0 - 3.0)/(-35.0/6.0 + 2.5);

            if (y <= 10.0 + slope*(x + 35.0/6.0)) {
              // Middle water
              *epsilon = water;
            } else {
              // Middle lipid
              *epsilon = lipid;
            }
          } else {
            // Bottom half
            double slope = (-10.0 + 3.0)/(-35.0/6.0 + 2.5);

            if (y >= -10.0 + slope*(x + 35.0/6.0)) {
              // Middle water
              *epsilon = water;
            } else {
              // Middle lipid
              *epsilon = lipid;
            }
          }
        }
      } else if ((x >= 2.5) && (x <= 12.5)) {
        // Right lipid or water
        if (x >= 35.0/6.0) {
          // Right parallelogram
          if (y >= 0.0) {
            // Top half
            double slope = (15.0 - 10.0)/(12.5 - 35.0/6.0);

            if (y <= 15.0 + slope*(x - 12.5)) {
              // Middle water
              *epsilon = water;
            } else {
              // Middle lipid
              *epsilon = lipid;
            }
          } else {
            // Bottom half
            double slope = (-15.0 + 10.0)/(12.5 - 35.0/6.0);

            if (y >= -15.0 + slope*(x - 12.5)) {
              // Middle water
              *epsilon = water;
            } else {
              // Middle lipid
              *epsilon = lipid;
            }
          }
        } else {
          // Left parallelogram
          if (y >= 0.0) {
            // Top half
            double slope = (10.0 - 3.0)/(35.0/6.0 - 2.5);

            if (y <= 10.0 + slope*(x - 35.0/6.0)) {
              // Middle water
              *epsilon = water;
            } else {
              // Middle lipid
              *epsilon = lipid;
            }
          } else {
            // Bottom half
            double slope = (-10.0 + 3.0)/(35.0/6.0 - 2.5);

            if (y >= -10.0 + slope*(x - 35.0/6.0)) {
              // Middle water
              *epsilon = water;
            } else {
              // Middle lipid
              *epsilon = lipid;
            }
          }
        }
      } else {
        if ((y <= 3.0) && (y >= -3.0)) {
          // Channel
          *epsilon = channel;
        } else {
          // Central lipid
          *epsilon = lipid;
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateDielectricVector"
/*
  Creates a vector whose value is the dielectric constant on each element
*/
PetscErrorCode CreateDielectricVector(ALE::Obj<ALE::Two::Mesh> mesh, Vec *dielectric)
{
  ALE::Obj<ALE::Two::Mesh::sieve_type> topology = mesh->getTopology();
  ALE::Obj<ALE::Two::Mesh::sieve_type::traits::heightSequence> elements = topology->heightStratum(0);
  ALE::Obj<ALE::Two::Mesh::field_type> coordinates = mesh->getCoordinates();
  ALE::Obj<ALE::Two::Mesh::field_type> epsilon = mesh->getField("epsilon");
  ALE::Two::Mesh::field_type::patch_type patch;
  std::string orderName("element");
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ALE_LOG_EVENT_BEGIN;
  epsilon->setPatch(topology->leaves(), patch);
  epsilon->setFiberDimensionByHeight(patch, 0, 1);
  epsilon->orderPatches();
  epsilon->createGlobalOrder();

  for(ALE::Two::Mesh::sieve_type::traits::heightSequence::iterator e_itor = elements->begin(); e_itor != elements->end(); ++e_itor) {
    const double *coords = coordinates->restrict(orderName, *e_itor);
    double centroidX = (coords[0]+coords[2]+coords[4])/3.0;
    double centroidY = (coords[1]+coords[3]+coords[5])/3.0;

    double eps;
    ierr = ComputeDielectric(centroidX, centroidY, &eps);CHKERRQ(ierr);
    epsilon->update(patch, *e_itor, &eps);
  }

  VecScatter injection;
  ierr = MeshCreateVector(mesh, mesh->getBundle(mesh->getDimension()), dielectric);CHKERRQ(ierr);
  ierr = MeshGetGlobalScatter(mesh, "epsilon", *dielectric, &injection); CHKERRQ(ierr);

  Vec locEpsilon;
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, epsilon->getSize(patch), epsilon->restrict(patch), &locEpsilon);CHKERRQ(ierr);
  ierr = VecScatterBegin(locEpsilon, *dielectric, INSERT_VALUES, SCATTER_FORWARD, injection);CHKERRQ(ierr);
  ierr = VecScatterEnd(locEpsilon, *dielectric, INSERT_VALUES, SCATTER_FORWARD, injection);CHKERRQ(ierr);
  ierr = VecDestroy(locEpsilon);CHKERRQ(ierr);
  ALE_LOG_EVENT_END;
  PetscFunctionReturn(0);
}
