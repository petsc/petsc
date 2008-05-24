/*T
   Concepts: Mesh^generating a mesh
   Concepts: Mesh^partitioning a mesh
   Concepts: Mesh^viewing a mesh
   Concepts: Applications^radially-symmetric ion channel 
   Processors: n
T*/
static char help[] = "Generates, partitions, and outputs an unstructured 2D mesh of a radially-symmetric simple ion channel.\n\n";

#include <petscda.h>
#include <petscmesh.h>

using ALE::Obj;

/*
  Generate a 2D triangular mesh of a radially-symmetric slide of a schematic ion channel using the builtin mesh generator.

  Partition the mesh and distribute it to each process.

  Output the mesh in VTK format with a scalar field indicating
  the rank of the process owning each cell.
*/

typedef struct {
  PetscInt   debug;           // The debugging level
  PetscInt   dim;             // The topological mesh dimension
  PetscTruth interpolate;     // Construct intermediate mesh elements
  PetscReal  refinementLimit; // The area of the largest triangle in the mesh
  PetscReal  refinementExp;   // The exponent of the radius for refinement
  PetscTruth viewDielectric;  // View the dielectric constant as a field
} Options;

double refineLimit(const double [], void *);

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->debug           = 0;
  options->dim             = 2;
  options->interpolate     = PETSC_TRUE;
  options->refinementLimit = 0.0;
  options->refinementExp   = 1.0;
  options->viewDielectric  = PETSC_FALSE;

  ierr = PetscOptionsBegin(comm, "", "PFLOTRAN Options", "DMMG");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-debug", "The debugging level", "ionChannel.cxx", options->debug, &options->debug, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-interpolate", "Construct missing elements of the mesh", "ionChannel.cxx", options->interpolate, &options->interpolate, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-refinement_limit", "The area of the largest triangle in the mesh", "ionChannel.cxx", options->refinementLimit, &options->refinementLimit, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-refinement_exp", "The exponent of the radius for refinement", "ionChannel.cxx", options->refinementExp, &options->refinementExp, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-view_dielectric", "View the dielectric constant as a field", "ionChannel.cxx", options->viewDielectric, &options->viewDielectric, PETSC_NULL);CHKERRQ(ierr);

  ierr = PetscOptionsEnd();

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreatePartition"
// Creates a field whose value is the processor rank on each element
PetscErrorCode CreatePartition(Mesh mesh, SectionInt *partition)
{
  Obj<ALE::Mesh> m;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = MeshGetCellSectionInt(mesh, 1, partition);CHKERRQ(ierr);
  const Obj<ALE::Mesh::label_sequence>&     cells = m->heightStratum(0);
  const ALE::Mesh::label_sequence::iterator end   = cells->end();
  const int                                 rank  = m->commRank();

  for(ALE::Mesh::label_sequence::iterator c_iter = cells->begin(); c_iter != end; ++c_iter) {
    ierr = SectionIntUpdate(*partition, *c_iter, &rank);
  }
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
#define __FUNCT__ "CreateDielectric"
// Creates a vector whose value is the dielectric constant on each element
PetscErrorCode CreateDielectric(Mesh mesh, SectionReal *dielectric)
{
  Obj<ALE::Mesh> m;
  Obj<ALE::Mesh::real_section_type> s;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = MeshGetCellSectionReal(mesh, 1, dielectric);CHKERRQ(ierr);
  ierr = SectionRealGetSection(*dielectric, s);CHKERRQ(ierr);
  const Obj<ALE::Mesh::label_sequence>&     cells       = m->heightStratum(0);
  const ALE::Mesh::label_sequence::iterator end         = cells->end();
  const Obj<ALE::Mesh::real_section_type>   coordinates = m->getRealSection("coordinates");

  s->setName("epsilon");
  for(ALE::Mesh::label_sequence::iterator c_iter = cells->begin(); c_iter != end; ++c_iter) {
    const double *coords = m->restrictClosure(coordinates, *c_iter);
    double centroidX = (coords[0]+coords[2]+coords[4])/3.0;
    double centroidY = (coords[1]+coords[3]+coords[5])/3.0;
    double eps;

    ierr = ComputeDielectric(centroidX, centroidY, &eps);CHKERRQ(ierr);
    ierr = SectionRealUpdate(*dielectric, *c_iter, &eps);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ViewMesh"
PetscErrorCode ViewMesh(Mesh mesh, const char filename[], Options *options)
{
  MPI_Comm       comm;
  SectionInt     partition;
  PetscViewer    viewer;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) mesh, &comm);CHKERRQ(ierr);
  ierr = PetscViewerCreate(comm, &viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
  ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(viewer, filename);CHKERRQ(ierr);
  ierr = MeshView(mesh, viewer);CHKERRQ(ierr);
  ierr = CreatePartition(mesh, &partition);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_VTK_CELL);CHKERRQ(ierr);
  ierr = SectionIntView(partition, viewer);CHKERRQ(ierr);
  ierr = SectionIntDestroy(partition);CHKERRQ(ierr);
  if (options->viewDielectric) {
    SectionReal dielectric;

    ierr = CreateDielectric(mesh, &dielectric);CHKERRQ(ierr);
    ierr = SectionRealView(dielectric, viewer);CHKERRQ(ierr);
    ierr = SectionRealDestroy(dielectric);CHKERRQ(ierr);
  }
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
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
PetscErrorCode CreateMeshBoundary(MPI_Comm comm, Mesh *boundary, Options *options)
{
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
  const Obj<ALE::Mesh>             m           = new ALE::Mesh(comm, 1, options->debug);
  const Obj<ALE::Mesh::sieve_type> sieve       = new ALE::Mesh::sieve_type(m->comm(), m->debug());
  const int                        numVertices = 27;
  const int                        numEdges    = 34;
  PetscErrorCode                   ierr;

  PetscFunctionBegin;
  ierr = MeshCreate(comm, boundary);CHKERRQ(ierr);
  ierr = MeshSetMesh(*boundary, m);CHKERRQ(ierr);
  m->setSieve(sieve);
  const Obj<ALE::Mesh::label_type>& markers = m->createLabel("marker");
  if (m->commRank() == 0) {
    ALE::Mesh::point_type vertices[27];

    /* Create sieve and ordering */
    for(int v = numEdges; v < numEdges+numVertices; ++v) {
      vertices[v-numEdges] = ALE::Mesh::point_type(v);
    }
    for(int e = 0; e < numEdges; ++e) {
      ALE::Mesh::point_type edge = ALE::Mesh::point_type(e);
      PetscInt              c    = -1;

      sieve->addArrow(vertices[connectivity[2*e]],   edge, ++c);
      sieve->addArrow(vertices[connectivity[2*e+1]], edge, ++c);
    }
  }
  m->stratify();
  ALE::SieveBuilder<ALE::Mesh>::buildCoordinates(m, m->getDimension()+1, coords);
  /* Create boundary conditions */
  if (m->commRank() == 0) {
    for(int e = 0; e < 2; e++) {
      m->setValue(markers, ALE::Mesh::point_type(e), 1);
    }
    for(int e = 2; e < 6; e++) {
      m->setValue(markers, ALE::Mesh::point_type(e), 2);
    }
    for(int e = 6; e < 8; e++) {
      m->setValue(markers, ALE::Mesh::point_type(e), 3);
    }
    for(int e = 8; e < 12; e++) {
      m->setValue(markers, ALE::Mesh::point_type(e), 4);
    }
    for(int e = 12; e < 20; e++) {
      m->setValue(markers, ALE::Mesh::point_type(e), 5);
    }
    for(int e = 20; e < 28; e++) {
      m->setValue(markers, ALE::Mesh::point_type(e), 6);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateMesh"
PetscErrorCode CreateMesh(MPI_Comm comm, Mesh *mesh, Options *options)
{
  Mesh           boundary;
  PetscTruth     view;
  PetscMPIInt    size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ALE::LogStage stage = ALE::LogStageRegister("MeshCreation");
  ALE::LogStagePush(stage);
  ierr = PetscPrintf(comm, "Generating mesh\n");CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr = CreateMeshBoundary(comm, &boundary, options);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL, "-boundary_view_vtk", &view);CHKERRQ(ierr);
  if (view) {ierr = ViewMesh(boundary, "ionChannelBoundary.vtk", options);CHKERRQ(ierr);}
  ierr = MeshGenerate(boundary, options->interpolate, mesh);CHKERRQ(ierr);
  ierr = MeshDestroy(boundary);CHKERRQ(ierr);
  if (options->refinementLimit > 0.0) {
    Mesh refinedMesh;

    // Need to put in nonuniform refinment
    //mesh = ALE::Two::Generator::refine(mesh, refineLimit, (void *) &refCtx, interpolate);
    ierr = MeshRefine(*mesh, options->refinementLimit, options->interpolate, &refinedMesh);CHKERRQ(ierr);
    ierr = MeshDestroy(*mesh);CHKERRQ(ierr);
    *mesh = refinedMesh;
  }
  if (size > 1) {
    Mesh parallelMesh;

    ierr = MeshDistribute(*mesh, PETSC_NULL, &parallelMesh);CHKERRQ(ierr);
    ierr = MeshDestroy(*mesh);CHKERRQ(ierr);
    *mesh = parallelMesh;
  }
  {
    Obj<ALE::Mesh> m;
    ierr = MeshGetMesh(*mesh, m);CHKERRQ(ierr);
    m->markBoundaryCells("marker");
  }
  ierr = PetscOptionsHasName(PETSC_NULL, "-mesh_view", &view);CHKERRQ(ierr);
  if (view) {
    Obj<ALE::Mesh> m;
    ierr = MeshGetMesh(*mesh, m);CHKERRQ(ierr);
    m->view("Mesh");
  }
  ierr = PetscOptionsHasName(PETSC_NULL, "-mesh_view_vtk", &view);CHKERRQ(ierr);
  if (view) {ierr = ViewMesh(*mesh, "ionChannel.vtk", options);CHKERRQ(ierr);}
  ALE::LogStagePop(stage);
  PetscFunctionReturn(0);
}

double refineLimit(const double centroid[], double refinementLimit, double refinementExp) {
  double r2 = centroid[0]*centroid[0] + centroid[1]*centroid[1];

  return refinementLimit*pow(r2, refinementExp*0.5);
}

#undef __FUNCT__
#define __FUNCT__ "CreateField"
PetscErrorCode CreateField(DM dm, Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  Mesh        mesh = (Mesh) dm;
  SectionReal f;
  Obj<ALE::Mesh> m;
  Obj<ALE::Mesh::real_section_type> s;
  
  ierr = MeshGetSectionReal(mesh, "u", &f);CHKERRQ(ierr);
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = SectionRealGetSection(f, s);CHKERRQ(ierr);
  const Obj<ALE::Mesh::label_sequence>& cells = m->heightStratum(0);
  const Obj<ALE::Discretization>&       disc  = m->getDiscretization();
  
  disc->setNumDof(m->depth(), 1);
  s->setDebug(options->debug);
  m->setupField(s);
  // Loop over elements (quadrilaterals)
  for(ALE::Mesh::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
    const double value = (double) *c_iter;
    
    ierr = SectionRealUpdate(f, *c_iter, &value);CHKERRQ(ierr);
  }
  ierr = SectionRealView(f, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = SectionRealDestroy(f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  MPI_Comm       comm;
  Options        options;
  Mesh           mesh;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr);
  try {
    comm = PETSC_COMM_WORLD;
    ierr = ProcessOptions(comm, &options);CHKERRQ(ierr);
    ierr = CreateMesh(comm, &mesh, &options);CHKERRQ(ierr);
    ierr = MeshDestroy(mesh);CHKERRQ(ierr);
  } catch(ALE::Exception e) {
    std::cout << "ERROR: " << e.msg() << std::endl;
  }
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
