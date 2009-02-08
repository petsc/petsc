/*T
   Concepts: KSP^solving a system of linear equations
   Concepts: KSP^Laplacian, 2d
   Processors: n
T*/

/*
Added at the request of Bob Eisenberg.

Inhomogeneous Laplacian in a 2D ion channel. Modeled by the partial differential equation

   div \epsilon grad u = f,  on \Omega,

with forcing function

   f = \sum_i q_i \delta(x - x_i)

with Dirichlet boundary conditions

   u = 0 for x = -L
   u = V for x =  L

and Neumman boundary conditions

   \hat n \cdot \grad u = 0 for y = -W, W

This uses multigrid to solve the linear system on a 2D radially-symmetric channel boundary:

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
    |                              54> |     |(XX) |<46                              |
    |       59                         | 53  | 60  |        45                       |
    |        V                         |  V  |  V  |        V                        |
    26(X)-----------------------------22----21-----16-------------------------------15

    (X) denotes the last vertex, (XX) denotes the last edge
*/

static char help[] = "Solves 2D inhomogeneous Laplacian using multigrid in an ion channel.\n\n";

#include "petscmesh.h"
#include "petscksp.h"
#include "petscmg.h"
#include "petscdmmg.h"

PetscErrorCode MeshView_Sieve_Newer(ALE::Obj<ALE::Mesh>, PetscViewer);
PetscErrorCode CreateMeshBoundary(ALE::Obj<ALE::Mesh>);
PetscErrorCode updateOperator(Mat, ALE::Obj<ALE::Mesh::field_type>, const ALE::Mesh::point_type&, PetscScalar [], InsertMode);

extern PetscErrorCode CheckElementGeometry(ALE::Obj<ALE::Mesh>);
extern PetscErrorCode ComputeRHS(DMMG,Vec);
extern PetscErrorCode ComputeMatrix(DMMG,Mat,Mat);
extern PetscErrorCode CreateDielectricField(ALE::Obj<ALE::Mesh>, ALE::Obj<ALE::Mesh::field_type>);
extern PetscErrorCode CreateEnergyDensity(ALE::Obj<ALE::Mesh>, ALE::Obj<ALE::Mesh::field_type>, ALE::Obj<ALE::Mesh::field_type>, Vec *);
double refineLimit(const double [], void *);

typedef enum {DIRICHLET, NEUMANN} BCType;

typedef struct {
  ALE::Obj<ALE::Mesh::field_type> epsilon;
  PetscScalar voltage;
  VecScatter  injection;
  PetscReal   refinementLimit;
  PetscReal   refinementExp;

  PetscInt    numCharges;
  PetscReal  *charge;
  PetscReal  *chargeLocation;
} UserContext;

PetscInt debug;

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  MPI_Comm       comm;
  DMMG          *dmmg;
  UserContext    user;
  PetscViewer    viewer;
  PetscReal      norm;
  PetscInt       dim, l, meshDebug;
  PetscTruth     viewEnergy;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char *)0,help);CHKERRQ(ierr);
  comm = PETSC_COMM_WORLD;
  ierr = PetscOptionsBegin(comm, "", "Options for the inhomogeneous Poisson equation", "DMMG");CHKERRQ(ierr);
    debug = 0;
    ierr = PetscOptionsInt("-debug", "The debugging flag", "ex37.c", 0, &debug, PETSC_NULL);CHKERRQ(ierr);
    meshDebug = 0;
    ierr = PetscOptionsInt("-mesh_debug", "The mesh debugging flag", "ex37.c", 0, &meshDebug, PETSC_NULL);CHKERRQ(ierr);
    dim  = 2;
    ierr = PetscOptionsInt("-dim", "The mesh dimension", "ex37.c", 2, &dim, PETSC_NULL);CHKERRQ(ierr);
    user.refinementLimit = 0.0;
    ierr = PetscOptionsReal("-refinement_limit", "The area of the largest triangle in the mesh", "ex37.c", 1.0, &user.refinementLimit, PETSC_NULL);CHKERRQ(ierr);
    user.refinementExp = 0.0;
    ierr = PetscOptionsReal("-refinement_exp", "The exponent of the radius for refinement", "ex37.c", 1.0, &user.refinementExp, PETSC_NULL);CHKERRQ(ierr);
    user.voltage = 1.0;
    ierr = PetscOptionsScalar("-voltage", "The voltage of the clamp", "ex37.c", 1.0, &user.voltage, PETSC_NULL);CHKERRQ(ierr);
    viewEnergy = PETSC_FALSE;
    ierr = PetscOptionsTruth("-view_energy", "View the energy density as a field", "ex37.c", PETSC_FALSE, &viewEnergy, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  ALE::Obj<ALE::Mesh> meshBoundary = ALE::Mesh(comm, dim-1, meshDebug);
  ALE::Obj<ALE::Mesh> mesh;

  try {
    ALE::LogStage stage = ALE::LogStageRegister("MeshCreation");
    ALE::LogStagePush(stage);
    ierr = PetscPrintf(comm, "Generating mesh\n");CHKERRQ(ierr);
    ierr = CreateMeshBoundary(meshBoundary);CHKERRQ(ierr);
    mesh = ALE::Generator::generate(meshBoundary);
    ALE::Obj<ALE::Mesh::sieve_type> topology = mesh->getTopology();
    ierr = PetscPrintf(comm, "  Read %d elements\n", topology->heightStratum(0)->size());CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "  Read %d vertices\n", topology->depthStratum(0)->size());CHKERRQ(ierr);
    ALE::LogStagePop(stage);
    mesh->getBoundary()->view("Initial Mesh Boundary");

    stage = ALE::LogStageRegister("MeshDistribution");
    ALE::LogStagePush(stage);
    ierr = PetscPrintf(comm, "Distributing mesh\n");CHKERRQ(ierr);
    mesh = mesh->distribute();
    ALE::LogStagePop(stage);
    mesh->getBoundary()->view("Distributed Mesh Boundary");

    if (user.refinementLimit > 0.0) {
      stage = ALE::LogStageRegister("MeshRefine");
      ALE::LogStagePush(stage);
      ierr = PetscPrintf(comm, "Refining mesh\n");CHKERRQ(ierr);
      if (user.refinementExp == 0.0) {
        mesh = ALE::Generator::refine(mesh, user.refinementLimit, true);
      } else {
        mesh = ALE::Generator::refine(mesh, refineLimit, (void *) &user, true);
      }
      ALE::LogStagePop(stage);
      ierr = PetscSynchronizedPrintf(comm, "  [%d]Generated %d local elements\n", mesh->commRank(), mesh->getTopology()->heightStratum(0)->size());CHKERRQ(ierr);
      ierr = PetscSynchronizedPrintf(comm, "  [%d]Generated %d local vertices\n", mesh->commRank(), mesh->getTopology()->depthStratum(0)->size());CHKERRQ(ierr);
      ierr = PetscSynchronizedFlush(comm);CHKERRQ(ierr);
    }
    topology = mesh->getTopology();

    stage = ALE::LogStageRegister("BndValues");
    ALE::LogStagePush(stage);
    ierr = PetscPrintf(comm, "Calculating boundary values\n");CHKERRQ(ierr);
    ALE::Obj<ALE::Mesh::field_type> boundary = mesh->getBoundary();
    ALE::Obj<ALE::Mesh::sieve_type::traits::depthSequence> vertices = topology->depthStratum(0);
    ALE::Mesh::field_type::patch_type groundPatch(0, 1);
    ALE::Mesh::field_type::patch_type voltagePatch(0, 3);
    ALE::Mesh::field_type::patch_type patch;

    for(ALE::Mesh::sieve_type::traits::depthSequence::iterator v_iter = vertices->begin(); v_iter != vertices->end(); ++v_iter) {
      if (boundary->getIndex(groundPatch, *v_iter).index > 0) {
        double values[1] = {0.0};

        boundary->update(groundPatch, *v_iter, values);
      } else if (boundary->getIndex(voltagePatch, *v_iter).index > 0) {
        double values[1] = {user.voltage};

        boundary->update(voltagePatch, *v_iter, values);
      }
    }
    if (debug) {boundary->view("Mesh Boundary");}
    ALE::LogStagePop(stage);

    user.numCharges = 1;
    ierr = PetscMalloc2(user.numCharges,PetscReal,&user.charge,user.numCharges*2,PetscReal,&user.chargeLocation);CHKERRQ(ierr);
    user.charge[0] = 1.0;
    user.chargeLocation[0] = 0.0;
    user.chargeLocation[1] = 0.0;

    ALE::Obj<ALE::Mesh::field_type> u = mesh->getField("u");
    ALE::Obj<ALE::Mesh::field_type> b = mesh->getField("b");
    u->setPatch(topology->leaves(), ALE::Mesh::field_type::patch_type());
    u->setFiberDimensionByDepth(patch, 0, 1);
    u->orderPatches();
    if (debug) {u->view("u");}
    u->createGlobalOrder();
    b->setPatch(topology->leaves(), ALE::Mesh::field_type::patch_type());
    b->setFiberDimensionByDepth(patch, 0, 1);
    b->orderPatches();
    if (debug) {b->view("b");}
    b->createGlobalOrder();
    ALE::Obj<ALE::Mesh::sieve_type::traits::heightSequence> elements = topology->heightStratum(0);
    ALE::Obj<ALE::Mesh::bundle_type> vertexBundle = mesh->getBundle(0);
    std::string orderName("element");

    for(ALE::Mesh::sieve_type::traits::heightSequence::iterator e_iter = elements->begin(); e_iter != elements->end(); e_iter++) {
      // setFiberDimensionByDepth() does not work here since we only want it to apply to the patch cone
      //   What we really need is the depthStratum relative to the patch
      ALE::Obj<ALE::Mesh::bundle_type::order_type::coneSequence> cone = vertexBundle->getPatch(orderName, *e_iter);

      u->setPatch(orderName, cone, *e_iter);
      b->setPatch(orderName, cone, *e_iter);
      for(ALE::Mesh::bundle_type::order_type::coneSequence::iterator c_iter = cone->begin(); c_iter != cone->end(); ++c_iter) {
        u->setFiberDimension(orderName, *e_iter, *c_iter, 1);
        b->setFiberDimension(orderName, *e_iter, *c_iter, 1);
      }
    }
    u->orderPatches(orderName);
    b->orderPatches(orderName);
    CheckElementGeometry(mesh);

    ierr = CreateDielectricField(mesh, mesh->getField("epsilon"));CHKERRQ(ierr);
    user.epsilon = mesh->getField("epsilon");

    Mesh petscMesh;
    ierr = MeshCreate(comm, &petscMesh);CHKERRQ(ierr);
    ierr = MeshSetMesh(petscMesh, mesh);CHKERRQ(ierr);
    ierr = DMMGCreate(comm,3,PETSC_NULL,&dmmg);CHKERRQ(ierr);
    ierr = DMMGSetDM(dmmg, (DM) petscMesh);CHKERRQ(ierr);
    ierr = MeshDestroy(petscMesh);CHKERRQ(ierr);
    for (l = 0; l < DMMGGetLevels(dmmg); l++) {
      ierr = DMMGSetUser(dmmg,l,&user);CHKERRQ(ierr);
    }

    ierr = DMMGSetKSP(dmmg,ComputeRHS,ComputeMatrix);CHKERRQ(ierr);
    ierr = MeshGetGlobalScatter(mesh, "u", DMMGGetx(dmmg), &user.injection);CHKERRQ(ierr);

    ierr = DMMGSolve(dmmg);CHKERRQ(ierr);

    if (debug) {
      ierr = PetscPrintf(mesh->comm(), "Solution vector:");CHKERRQ(ierr);
      ierr = VecView(DMMGGetx(dmmg), PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }
    //ierr = ComputeError(dmmg[DMMGGetLevels(dmmg)-1], DMMGGetx(dmmg), &error);CHKERRQ(ierr);
    //ierr = PetscPrintf(comm,"Error norm %g\n",error);CHKERRQ(ierr);

    ierr = MatMult(DMMGGetJ(dmmg),DMMGGetx(dmmg),DMMGGetr(dmmg));CHKERRQ(ierr);
    ierr = VecAXPY(DMMGGetr(dmmg),-1.0,DMMGGetRHS(dmmg));CHKERRQ(ierr);
    ierr = VecNorm(DMMGGetr(dmmg),NORM_2,&norm);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"Residual norm %g\n",norm);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(DMMGGetx(dmmg));CHKERRQ(ierr);
    ierr = VecAssemblyEnd(DMMGGetx(dmmg));CHKERRQ(ierr);

    stage = ALE::LogStageRegister("MeshOutput");
    ALE::LogStagePush(stage);
    ierr = PetscPrintf(comm, "Creating VTK mesh file\n");CHKERRQ(ierr);
    ierr = PetscViewerCreate(comm, &viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer, "channel.vtk");CHKERRQ(ierr);
    ierr = MeshView_Sieve_Newer(mesh, viewer);CHKERRQ(ierr);
    //ierr = VecView(DMMGGetRHS(dmmg), viewer);CHKERRQ(ierr);
    if (viewEnergy) {
      ALE::Obj<ALE::Mesh::field_type> u = mesh->getField("u");
      Vec energy, locU;

      ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, u->getSize(patch), u->restrict(patch), &locU);CHKERRQ(ierr);
      ierr = VecScatterBegin(user.injection, DMMGGetx(dmmg), locU, INSERT_VALUES, SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecScatterEnd(user.injection, DMMGGetx(dmmg), locU, INSERT_VALUES, SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecDestroy(locU);CHKERRQ(ierr);

      ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_VTK_CELL);CHKERRQ(ierr);
      ierr = CreateEnergyDensity(mesh, u, mesh->getField("epsilon"), &energy);CHKERRQ(ierr);
      ierr = VecView(energy, viewer);CHKERRQ(ierr);
      ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    } else {
      ierr = VecView(DMMGGetx(dmmg), viewer);CHKERRQ(ierr);
    }
    ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
    ALE::LogStagePop(stage);

    ierr = PetscFree2(user.charge,user.chargeLocation);CHKERRQ(ierr);
    ierr = DMMGDestroy(dmmg);CHKERRQ(ierr);
  } catch (ALE::Exception e) {
    std::cout << e << std::endl;
  }
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

double refineLimit(const double centroid[], void *ctx) {
  UserContext *user = (UserContext *) ctx;
  double       r2   = centroid[0]*centroid[0] + centroid[1]*centroid[1];

  return user->refinementLimit*pow(r2, user->refinementExp*0.5);
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
PetscErrorCode CreateMeshBoundary(ALE::Obj<ALE::Mesh> mesh)
{
  ALE::Obj<ALE::Mesh::sieve_type> topology = mesh->getTopology();
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
  ALE::Mesh::point_type vertices[27];

  PetscFunctionBegin;
  PetscInt order = 0;
  if (mesh->commRank() == 0) {
    ALE::Mesh::point_type edge;

    /* Create topology and ordering */
    for(int v = 0; v < 27; v++) {
      vertices[v] = ALE::Mesh::point_type(0, v);
    }
    for(int e = 27; e < 61; e++) {
      int ee = e - 27;
      edge = ALE::Mesh::point_type(0, e);
      topology->addArrow(vertices[connectivity[2*ee]],   edge, order++);
      topology->addArrow(vertices[connectivity[2*ee+1]], edge, order++);
    }
  }
  topology->stratify();
  mesh->createVertexBundle(34, connectivity, 27);
  mesh->createSerialCoordinates(2, 0, coords);
  /* Create boundary conditions */
  if (mesh->commRank() == 0) {
    for(int e = 47; e < 55; e++) {
      int ee = e - 27;

      topology->setMarker(ALE::Mesh::point_type(0, e), 6);
      topology->setMarker(ALE::Mesh::point_type(0, connectivity[2*ee+0]), 6);
      topology->setMarker(ALE::Mesh::point_type(0, connectivity[2*ee+1]), 6);
    }
    for(int e = 39; e < 47; e++) {
      int ee = e - 27;

      topology->setMarker(ALE::Mesh::point_type(0, e), 5);
      topology->setMarker(ALE::Mesh::point_type(0, connectivity[2*ee+0]), 5);
      topology->setMarker(ALE::Mesh::point_type(0, connectivity[2*ee+1]), 5);
    }
    for(int e = 35; e < 39; e++) {
      int ee = e - 27;

      topology->setMarker(ALE::Mesh::point_type(0, e), 4);
      topology->setMarker(ALE::Mesh::point_type(0, connectivity[2*ee+0]), 4);
      topology->setMarker(ALE::Mesh::point_type(0, connectivity[2*ee+1]), 4);
    }
    for(int e = 29; e < 33; e++) {
      int ee = e - 27;

      topology->setMarker(ALE::Mesh::point_type(0, e), 2);
      topology->setMarker(ALE::Mesh::point_type(0, connectivity[2*ee+0]), 2);
      topology->setMarker(ALE::Mesh::point_type(0, connectivity[2*ee+1]), 2);
    }
    for(int e = 33; e < 35; e++) {
      int ee = e - 27;

      topology->setMarker(ALE::Mesh::point_type(0, e), 3);
      topology->setMarker(ALE::Mesh::point_type(0, connectivity[2*ee+0]), 3);
      topology->setMarker(ALE::Mesh::point_type(0, connectivity[2*ee+1]), 3);
    }
    for(int e = 27; e < 29; e++) {
      int ee = e - 27;

      topology->setMarker(ALE::Mesh::point_type(0, e), 1);
      topology->setMarker(ALE::Mesh::point_type(0, connectivity[2*ee+0]), 1);
      topology->setMarker(ALE::Mesh::point_type(0, connectivity[2*ee+1]), 1);
    }
  }
  PetscFunctionReturn(0);
}

#define NUM_QUADRATURE_POINTS 9

/* Quadrature points */
static double points[18] = {
  -0.794564690381,
  -0.822824080975,
  -0.866891864322,
  -0.181066271119,
  -0.952137735426,
  0.575318923522,
  -0.0885879595127,
  -0.822824080975,
  -0.409466864441,
  -0.181066271119,
  -0.787659461761,
  0.575318923522,
  0.617388771355,
  -0.822824080975,
  0.0479581354402,
  -0.181066271119,
  -0.623181188096,
  0.575318923522};

/* Quadrature weights */
static double weights[9] = {
  0.223257681932,
  0.2547123404,
  0.0775855332238,
  0.357212291091,
  0.407539744639,
  0.124136853158,
  0.223257681932,
  0.2547123404,
  0.0775855332238};

#define NUM_BASIS_FUNCTIONS 3

/* Nodal basis function evaluations */
static double Basis[27] = {
  0.808694385678,
  0.10271765481,
  0.0885879595127,
  0.52397906772,
  0.0665540678392,
  0.409466864441,
  0.188409405952,
  0.0239311322871,
  0.787659461761,
  0.455706020244,
  0.455706020244,
  0.0885879595127,
  0.29526656778,
  0.29526656778,
  0.409466864441,
  0.10617026912,
  0.10617026912,
  0.787659461761,
  0.10271765481,
  0.808694385678,
  0.0885879595127,
  0.0665540678392,
  0.52397906772,
  0.409466864441,
  0.0239311322871,
  0.188409405952,
  0.787659461761};

/* Nodal basis function derivative evaluations */
static double BasisDerivatives[54] = {
  -0.5,
  -0.5,
  0.5,
  4.74937635818e-17,
  0.0,
  0.5,
  -0.5,
  -0.5,
  0.5,
  4.74937635818e-17,
  0.0,
  0.5,
  -0.5,
  -0.5,
  0.5,
  4.74937635818e-17,
  0.0,
  0.5,
  -0.5,
  -0.5,
  0.5,
  4.74937635818e-17,
  0.0,
  0.5,
  -0.5,
  -0.5,
  0.5,
  4.74937635818e-17,
  0.0,
  0.5,
  -0.5,
  -0.5,
  0.5,
  4.74937635818e-17,
  0.0,
  0.5,
  -0.5,
  -0.5,
  0.5,
  4.74937635818e-17,
  0.0,
  0.5,
  -0.5,
  -0.5,
  0.5,
  4.74937635818e-17,
  0.0,
  0.5,
  -0.5,
  -0.5,
  0.5,
  4.74937635818e-17,
  0.0,
  0.5};

#undef __FUNCT__
#define __FUNCT__ "ElementGeometry"
PetscErrorCode ElementGeometry(ALE::Obj<ALE::Mesh> mesh, const ALE::Mesh::point_type& e, PetscReal v0[], PetscReal J[], PetscReal invJ[], PetscReal *detJ)
{
  const double  *coords = mesh->getCoordinates()->restrict(std::string("element"), e);
  int            dim = mesh->getDimension();
  PetscReal      det, invDet;

  PetscFunctionBegin;
  if (debug) {
    MPI_Comm comm = mesh->comm();
    int      rank = mesh->commRank();

    PetscSynchronizedPrintf(comm, "[%d]Element (%d, %d)\n", rank, e.prefix, e.index);
    PetscSynchronizedPrintf(comm, "[%d]Coordinates:\n[%d]  ", rank, rank);
    for(int f = 0; f <= dim; f++) {
      PetscSynchronizedPrintf(comm, " (");
      for(int d = 0; d < dim; d++) {
        if (d > 0) PetscSynchronizedPrintf(comm, ", ");
        PetscSynchronizedPrintf(comm, "%g", coords[f*dim+d]);
      }
      PetscSynchronizedPrintf(comm, ")");
    }
    PetscSynchronizedPrintf(comm, "\n");
  }
  if (v0) {
    for(int d = 0; d < dim; d++) {
      v0[d] = coords[d];
    }
  }
  if (J) {
    for(int d = 0; d < dim; d++) {
      for(int f = 0; f < dim; f++) {
        J[d*dim+f] = 0.5*(coords[(f+1)*dim+d] - coords[0*dim+d]);
      }
    }
    if (debug) {
      MPI_Comm comm = mesh->comm();
      int      rank = mesh->commRank();

      for(int d = 0; d < dim; d++) {
        if (d == 0) {
          PetscSynchronizedPrintf(comm, "[%d]J = /", rank);
        } else if (d == dim-1) {
          PetscSynchronizedPrintf(comm, "[%d]    \\", rank);
        } else {
          PetscSynchronizedPrintf(comm, "[%d]    |", rank);
        }
        for(int e = 0; e < dim; e++) {
          PetscSynchronizedPrintf(comm, " %g", J[d*dim+e]);
        }
        if (d == 0) {
          PetscSynchronizedPrintf(comm, " \\\n");
        } else if (d == dim-1) {
          PetscSynchronizedPrintf(comm, " /\n");
        } else {
          PetscSynchronizedPrintf(comm, " |\n");
        }
      }
    }
    if (dim == 2) {
      det = J[0]*J[3] - J[1]*J[2];
    } else if (dim == 3) {
      det = J[0*3+0]*(J[1*3+1]*J[2*3+2] - J[1*3+2]*J[2*3+1]) +
            J[0*3+1]*(J[1*3+2]*J[2*3+0] - J[1*3+0]*J[2*3+2]) +
            J[0*3+2]*(J[1*3+0]*J[2*3+1] - J[1*3+1]*J[2*3+0]);
    }
    invDet = 1.0/det;
    if (detJ) {
      if (det < 0) {SETERRQ(PETSC_ERR_ARG_WRONG, "Negative Matrix determinant");}
      *detJ = det;
    }
    if (invJ) {
      if (dim == 2) {
        invJ[0] =  invDet*J[3];
        invJ[1] = -invDet*J[1];
        invJ[2] = -invDet*J[2];
        invJ[3] =  invDet*J[0];
      } else if (dim == 3) {
        // FIX: This may be wrong
        invJ[0*3+0] = invDet*(J[1*3+1]*J[2*3+2] - J[1*3+2]*J[2*3+1]);
        invJ[0*3+1] = invDet*(J[1*3+2]*J[2*3+0] - J[1*3+0]*J[2*3+2]);
        invJ[0*3+2] = invDet*(J[1*3+0]*J[2*3+1] - J[1*3+1]*J[2*3+0]);
        invJ[1*3+0] = invDet*(J[0*3+1]*J[2*3+2] - J[0*3+2]*J[2*3+1]);
        invJ[1*3+1] = invDet*(J[0*3+2]*J[2*3+0] - J[0*3+0]*J[2*3+2]);
        invJ[1*3+2] = invDet*(J[0*3+0]*J[2*3+1] - J[0*3+1]*J[2*3+0]);
        invJ[2*3+0] = invDet*(J[0*3+1]*J[1*3+2] - J[0*3+2]*J[1*3+1]);
        invJ[2*3+1] = invDet*(J[0*3+2]*J[1*3+0] - J[0*3+0]*J[1*3+2]);
        invJ[2*3+2] = invDet*(J[0*3+0]*J[1*3+1] - J[0*3+1]*J[1*3+0]);
      }
      if (debug) {
        MPI_Comm comm = mesh->comm();
        int      rank = mesh->commRank();

        for(int d = 0; d < dim; d++) {
          if (d == 0) {
            PetscSynchronizedPrintf(comm, "[%d]Jinv = /", rank);
          } else if (d == dim-1) {
            PetscSynchronizedPrintf(comm, "[%d]       \\", rank);
          } else {
            PetscSynchronizedPrintf(comm, "[%d]       |", rank);
          }
          for(int e = 0; e < dim; e++) {
            PetscSynchronizedPrintf(comm, " %g", invJ[d*dim+e]);
          }
          if (d == 0) {
            PetscSynchronizedPrintf(comm, " \\\n");
          } else if (d == dim-1) {
            PetscSynchronizedPrintf(comm, " /\n");
          } else {
            PetscSynchronizedPrintf(comm, " |\n");
          }
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ElementContains"
PetscErrorCode ElementContains(ALE::Obj<ALE::Mesh> mesh, const ALE::Mesh::point_type& e, double point[], PetscTruth *contains)
{
  const double  *coords = mesh->getCoordinates()->restrict(std::string("element"), e);
  int            dim = mesh->getDimension();

  PetscFunctionBegin;
  *contains = PETSC_TRUE;
  for(int p = 0; p < 3; p++) {
    double side[2] = {coords[((p+1)%3)*dim+0] - coords[p*dim+0], coords[((p+1)%3)*dim+1] - coords[p*dim+1]};
    double ray[2]  = {point[0]                - coords[p*dim+0], point[1]                - coords[p*dim+1]};
    double cross   = side[0]*ray[1] - side[1]*ray[0];

    if (cross <= 0.0) {
      *contains = PETSC_FALSE;
      break;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CheckElementGeometry"
PetscErrorCode CheckElementGeometry(ALE::Obj<ALE::Mesh> mesh)
{
  ALE::Obj<ALE::Mesh::sieve_type::traits::heightSequence> elements = mesh->getTopology()->heightStratum(0);
  PetscInt       dim = mesh->getDimension();
  PetscReal     *v0, *Jac;
  PetscReal      detJ;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc2(dim,PetscReal,&v0,dim*dim,PetscReal,&Jac);CHKERRQ(ierr);
  for(ALE::Mesh::sieve_type::traits::heightSequence::iterator e_iter = elements->begin(); e_iter != elements->end(); ++e_iter) {
    ierr = ElementGeometry(mesh, *e_iter, v0, Jac, PETSC_NULL, &detJ);
  }
  ierr = PetscSynchronizedFlush(mesh->comm());CHKERRQ(ierr);
  ierr = PetscFree2(v0,Jac);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ComputeChargeDensity"
PetscErrorCode ComputeChargeDensity(ALE::Obj<ALE::Mesh> mesh, double points[], double jac[], double v0[], double detJ, double basis[], ALE::Obj<ALE::Mesh::field_type> field, UserContext *user)
{
  ALE::Obj<ALE::Mesh::sieve_type::traits::heightSequence> elements = mesh->getTopology()->heightStratum(0);
  PetscReal      elementVec[NUM_BASIS_FUNCTIONS];
  PetscTruth     contains;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for(int q = 0; q < user->numCharges; q++) {
    double x = user->chargeLocation[q*2+0];
    double y = user->chargeLocation[q*2+1];

    for(ALE::Mesh::sieve_type::traits::heightSequence::iterator e_itor = elements->begin(); e_itor != elements->end(); e_itor++) {
      ierr = ElementContains(mesh, *e_itor, user->chargeLocation, &contains);CHKERRQ(ierr);
      if (contains) {
        double     minDist = 1.0e30;
        int        minP    = -1;
        PetscReal *v0, *Jac, detJ;

        ierr = ElementGeometry(mesh, *e_itor, v0, Jac, PETSC_NULL, &detJ);CHKERRQ(ierr);
        for(int p = 0; p < NUM_QUADRATURE_POINTS; p++) {
          double xi, eta, x_p, y_p, dist;

          xi = points[p*2+0] + 1.0;
          eta = points[p*2+1] + 1.0;
          x_p = jac[0]*xi + jac[1]*eta + v0[0];
          y_p = jac[2]*xi + jac[3]*eta + v0[1];
          dist = (x - x_p)*(x - x_p) + (y - y_p)*(y - y_p);
          if (dist < minDist) {
            minDist = dist;
            minP    = p;
          }
        }
        for(int f = 0; f < NUM_BASIS_FUNCTIONS; f++) {
          elementVec[f] += basis[minP*NUM_BASIS_FUNCTIONS+f]*user->charge[q]*detJ;
        }
        /* Assembly */
        field->updateAdd("element", *e_itor, elementVec);
        break;
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ComputeRHS"
PetscErrorCode ComputeRHS(DMMG dmmg, Vec b)
{
  ALE::Obj<ALE::Mesh> m;
  Mesh                mesh = (Mesh) dmmg->dm;
  UserContext        *user = (UserContext *) dmmg->user;
  MPI_Comm            comm;
  PetscReal           elementVec[NUM_BASIS_FUNCTIONS];
  PetscReal           *v0, *Jac;
  PetscReal           xi, eta, x_q, y_q, detJ, funcValue;
  PetscInt            dim;
  PetscInt            f, q;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) mesh, &comm);CHKERRQ(ierr);
  ierr = MeshGetMesh(mesh, &m);CHKERRQ(ierr);
  dim  = m->getDimension();
  ierr = PetscMalloc(dim * sizeof(PetscReal), &v0);CHKERRQ(ierr);
  ierr = PetscMalloc(dim*dim * sizeof(PetscReal), &Jac);CHKERRQ(ierr);
  ALE::Obj<ALE::Mesh::field_type> field = m->getField("b");
  ALE::Obj<ALE::Mesh::sieve_type::traits::heightSequence> elements = m->getTopology()->heightStratum(0);
  ALE::Mesh::field_type::patch_type patch;
  for(ALE::Mesh::sieve_type::traits::heightSequence::iterator e_itor = elements->begin(); e_itor != elements->end(); e_itor++) {
    ierr = ElementGeometry(m, *e_itor, v0, Jac, PETSC_NULL, &detJ);CHKERRQ(ierr);
    /* Element integral */
    ierr = PetscMemzero(elementVec, NUM_BASIS_FUNCTIONS*sizeof(PetscScalar));CHKERRQ(ierr);
    for(q = 0; q < NUM_QUADRATURE_POINTS; q++) {
      xi = points[q*2+0] + 1.0;
      eta = points[q*2+1] + 1.0;
      x_q = Jac[0]*xi + Jac[1]*eta + v0[0];
      y_q = Jac[2]*xi + Jac[3]*eta + v0[1];
      funcValue = 0.0;
      for(f = 0; f < NUM_BASIS_FUNCTIONS; f++) {
        elementVec[f] += Basis[q*NUM_BASIS_FUNCTIONS+f]*funcValue*weights[q]*detJ;
      }
    }
    if (debug) {PetscSynchronizedPrintf(comm, "elementVec = [%g %g %g]\n", elementVec[0], elementVec[1], elementVec[2]);CHKERRQ(ierr);}
    /* Assembly */
    field->updateAdd("element", *e_itor, elementVec);
    if (debug) {ierr = PetscSynchronizedFlush(comm);CHKERRQ(ierr);}
  }
  ierr = ComputeChargeDensity(m, points, v0, Jac, detJ, Basis, field, user);CHKERRQ(ierr);
  ierr = PetscFree(v0);CHKERRQ(ierr);
  ierr = PetscFree(Jac);CHKERRQ(ierr);

  Vec locB;
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, field->getSize(patch), field->restrict(patch), &locB);CHKERRQ(ierr);
  ierr = VecScatterBegin(user.injection, locB, b, ADD_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(user.injection, locB, b, ADD_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecDestroy(locB);CHKERRQ(ierr);

  {
    /* Zero out BC rows */
    ALE::Mesh::field_type::patch_type patch;
    ALE::Mesh::field_type::patch_type groundPatch(0, 1);
    ALE::Mesh::field_type::patch_type voltagePatch(0, 3);
    ALE::Obj<ALE::Mesh::field_type> boundary = m->getBoundary();
    ALE::Obj<ALE::Mesh::field_type::order_type::coneSequence> groundCone = boundary->getPatch(groundPatch);
    ALE::Obj<ALE::Mesh::field_type::order_type::coneSequence> voltageCone = boundary->getPatch(voltagePatch);
    PetscScalar *boundaryValues;
    PetscInt    *boundaryIndices;
    PetscInt     numBoundaryIndices = 0;
    PetscInt     k = 0;

    for(ALE::Mesh::field_type::order_type::coneSequence::iterator p = groundCone->begin(); p != groundCone->end(); ++p) {
      numBoundaryIndices += field->getGlobalOrder()->getIndex(patch, *p).index;
    }
    for(ALE::Mesh::field_type::order_type::coneSequence::iterator p = voltageCone->begin(); p != voltageCone->end(); ++p) {
      numBoundaryIndices += field->getGlobalOrder()->getIndex(patch, *p).index;
    }
    ierr = PetscMalloc2(numBoundaryIndices,PetscInt,&boundaryIndices,numBoundaryIndices,PetscScalar,&boundaryValues);CHKERRQ(ierr);
    for(ALE::Mesh::field_type::order_type::coneSequence::iterator p = groundCone->begin(); p != groundCone->end(); ++p) {
      const ALE::Mesh::field_type::index_type& idx = field->getGlobalOrder()->getIndex(patch, *p);
      const double *data = boundary->restrict(groundPatch, *p);

      for(int i = 0; i < idx.index; i++) {
        boundaryIndices[k] = idx.prefix + i;
        boundaryValues[k] = data[i];
        k++;
      }
    }
    for(ALE::Mesh::field_type::order_type::coneSequence::iterator p = voltageCone->begin(); p != voltageCone->end(); ++p) {
      const ALE::Mesh::field_type::index_type& idx = field->getGlobalOrder()->getIndex(patch, *p);
      const double *data = boundary->restrict(voltagePatch, *p);

      for(int i = 0; i < idx.index; i++) {
        boundaryIndices[k] = idx.prefix + i;
        boundaryValues[k] = data[i];
        k++;
      }
    }
    if (debug) {
      boundary->view("Boundary for rhs conditions");
      for(int i = 0; i < numBoundaryIndices; i++) {
        ierr = PetscSynchronizedPrintf(comm, "[%d]boundaryIndices[%d] = %d\n", m->commRank(), i, boundaryIndices[i]);CHKERRQ(ierr);
      }
    }
    ierr = PetscSynchronizedFlush(comm);
    ierr = VecSetValues(b, numBoundaryIndices, boundaryIndices, boundaryValues, INSERT_VALUES);CHKERRQ(ierr);
    ierr = PetscFree2(boundaryIndices, boundaryValues);CHKERRQ(ierr);
  }
  if (debug) {
    ierr = PetscPrintf(comm, "Rhs vector:");CHKERRQ(ierr);
    ierr = VecView(b, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ComputeMatrix"
PetscErrorCode ComputeMatrix(DMMG dmmg, Mat J, Mat jac)
{
  ALE::Obj<ALE::Mesh> m;
  Mesh              mesh = (Mesh) dmmg->dm;
  UserContext      *user = (UserContext *) dmmg->user;
  MPI_Comm          comm;
  PetscReal         elementMat[NUM_BASIS_FUNCTIONS*NUM_BASIS_FUNCTIONS];
  PetscReal        *v0, *Jac, *Jinv, *t_der, *b_der;
  PetscReal         xi, eta, x_q, y_q, detJ;
  PetscInt          dim;
  PetscInt          f, g, q;
  PetscMPIInt       rank;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) mesh, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MeshGetMesh(mesh, &m);CHKERRQ(ierr);
  dim  = m->getDimension();
  ierr = PetscMalloc(dim * sizeof(PetscReal), &v0);CHKERRQ(ierr);
  ierr = PetscMalloc(dim * sizeof(PetscReal), &t_der);CHKERRQ(ierr);
  ierr = PetscMalloc(dim * sizeof(PetscReal), &b_der);CHKERRQ(ierr);
  ierr = PetscMalloc(dim*dim * sizeof(PetscReal), &Jac);CHKERRQ(ierr);
  ierr = PetscMalloc(dim*dim * sizeof(PetscReal), &Jinv);CHKERRQ(ierr);
  ALE::Obj<ALE::Mesh::field_type> field = m->getField("u");
  ALE::Obj<ALE::Mesh::sieve_type::traits::heightSequence> elements = m->getTopology()->heightStratum(0);
  for(ALE::Mesh::sieve_type::traits::heightSequence::iterator e_itor = elements->begin(); e_itor != elements->end(); e_itor++) {
    ALE::Mesh::field_type::patch_type patch;
    double eps = user->epsilon->restrict(patch, *e_itor)[0];

    CHKMEMQ;
    ierr = ElementGeometry(m, *e_itor, v0, Jac, Jinv, &detJ);CHKERRQ(ierr);
    /* Element integral */
    ierr = PetscMemzero(elementMat, NUM_BASIS_FUNCTIONS*NUM_BASIS_FUNCTIONS*sizeof(PetscScalar));CHKERRQ(ierr);
    for(q = 0; q < NUM_QUADRATURE_POINTS; q++) {
      xi = points[q*2+0] + 1.0;
      eta = points[q*2+1] + 1.0;
      x_q = Jac[0]*xi + Jac[1]*eta + v0[0];
      y_q = Jac[2]*xi + Jac[3]*eta + v0[1];
      for(f = 0; f < NUM_BASIS_FUNCTIONS; f++) {
        t_der[0] = Jinv[0]*BasisDerivatives[(q*NUM_BASIS_FUNCTIONS+f)*2+0] + Jinv[2]*BasisDerivatives[(q*NUM_BASIS_FUNCTIONS+f)*2+1];
        t_der[1] = Jinv[1]*BasisDerivatives[(q*NUM_BASIS_FUNCTIONS+f)*2+0] + Jinv[3]*BasisDerivatives[(q*NUM_BASIS_FUNCTIONS+f)*2+1];
        for(g = 0; g < NUM_BASIS_FUNCTIONS; g++) {
          b_der[0] = Jinv[0]*BasisDerivatives[(q*NUM_BASIS_FUNCTIONS+g)*2+0] + Jinv[2]*BasisDerivatives[(q*NUM_BASIS_FUNCTIONS+g)*2+1];
          b_der[1] = Jinv[1]*BasisDerivatives[(q*NUM_BASIS_FUNCTIONS+g)*2+0] + Jinv[3]*BasisDerivatives[(q*NUM_BASIS_FUNCTIONS+g)*2+1];
          elementMat[f*NUM_BASIS_FUNCTIONS+g] += eps*(t_der[0]*b_der[0] + t_der[1]*b_der[1])*weights[q]*detJ;
        }
      }
    }
    if (debug) {
      ierr = PetscSynchronizedPrintf(comm, "[%d]elementMat = [%g %g %g]\n                [%g %g %g]\n                [%g %g %g]\n",
                                     rank, elementMat[0], elementMat[1], elementMat[2], elementMat[3], elementMat[4],
                                     elementMat[5], elementMat[6], elementMat[7], elementMat[8]);CHKERRQ(ierr);
    }
    /* Assembly */
    ierr = updateOperator(jac, field, *e_itor, elementMat, ADD_VALUES);CHKERRQ(ierr);
    if (debug) {ierr = PetscSynchronizedFlush(comm);CHKERRQ(ierr);}
  }
  ierr = PetscFree(v0);CHKERRQ(ierr);
  ierr = PetscFree(t_der);CHKERRQ(ierr);
  ierr = PetscFree(b_der);CHKERRQ(ierr);
  ierr = PetscFree(Jac);CHKERRQ(ierr);
  ierr = PetscFree(Jinv);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  {
    /* Zero out BC rows */
    ALE::Mesh::field_type::patch_type patch;
    ALE::Mesh::field_type::patch_type groundPatch(0, 1);
    ALE::Mesh::field_type::patch_type voltagePatch(0, 3);
    ALE::Obj<ALE::Mesh::field_type> boundary = m->getBoundary();
    ALE::Obj<ALE::Mesh::field_type::order_type::coneSequence> groundCone = boundary->getPatch(groundPatch);
    ALE::Obj<ALE::Mesh::field_type::order_type::coneSequence> voltageCone = boundary->getPatch(voltagePatch);
    PetscInt *boundaryIndices;
    PetscInt  numBoundaryIndices = 0;
    PetscInt  k = 0;

    for(ALE::Mesh::field_type::order_type::coneSequence::iterator p = groundCone->begin(); p != groundCone->end(); ++p) {
      numBoundaryIndices += field->getGlobalOrder()->getIndex(patch, *p).index;
    }
    for(ALE::Mesh::field_type::order_type::coneSequence::iterator p = voltageCone->begin(); p != voltageCone->end(); ++p) {
      numBoundaryIndices += field->getGlobalOrder()->getIndex(patch, *p).index;
    }
    ierr = PetscMalloc(numBoundaryIndices * sizeof(PetscInt), &boundaryIndices);CHKERRQ(ierr);
    for(ALE::Mesh::field_type::order_type::coneSequence::iterator p = groundCone->begin(); p != groundCone->end(); ++p) {
      const ALE::Mesh::field_type::index_type& idx = field->getGlobalOrder()->getIndex(patch, *p);

      for(int i = 0; i < idx.index; i++) {
        boundaryIndices[k++] = idx.prefix + i;
      }
    }
    for(ALE::Mesh::field_type::order_type::coneSequence::iterator p = voltageCone->begin(); p != voltageCone->end(); ++p) {
      const ALE::Mesh::field_type::index_type& idx = field->getGlobalOrder()->getIndex(patch, *p);

      for(int i = 0; i < idx.index; i++) {
        boundaryIndices[k++] = idx.prefix + i;
      }
    }
    if (debug) {
      for(int i = 0; i < numBoundaryIndices; i++) {
        ierr = PetscSynchronizedPrintf(comm, "[%d]boundaryIndices[%d] = %d\n", rank, i, boundaryIndices[i]);CHKERRQ(ierr);
      }
    }
    ierr = PetscSynchronizedFlush(comm);
    ierr = MatZeroRows(jac, numBoundaryIndices, boundaryIndices, 1.0);CHKERRQ(ierr);
    ierr = PetscFree(boundaryIndices);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ComputeDielectric"
PetscErrorCode ComputeDielectric(double x, double y, double *epsilon) {
  double water   = 80.0;
  double lipid   = 2.0;
  double channel = 10.0;

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
#define __FUNCT__ "CreateDielectricField"
/*
  Creates a vector whose value is the dielectric constant on each element
*/
PetscErrorCode CreateDielectricField(ALE::Obj<ALE::Mesh> mesh, ALE::Obj<ALE::Mesh::field_type> epsilon)
{
  ALE::Obj<ALE::Mesh::sieve_type> topology = mesh->getTopology();
  ALE::Obj<ALE::Mesh::sieve_type::traits::heightSequence> elements = topology->heightStratum(0);
  ALE::Obj<ALE::Mesh::field_type> coordinates = mesh->getCoordinates();
  ALE::Mesh::field_type::patch_type patch;
  std::string orderName("element");
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ALE_LOG_EVENT_BEGIN;
  epsilon->setPatch(topology->leaves(), patch);
  epsilon->setFiberDimensionByHeight(patch, 0, 1);
  epsilon->orderPatches();
  epsilon->createGlobalOrder();

  for(ALE::Mesh::sieve_type::traits::heightSequence::iterator e_itor = elements->begin(); e_itor != elements->end(); ++e_itor) {
    const double *coords = coordinates->restrict(orderName, *e_itor);
    double centroidX = (coords[0]+coords[2]+coords[4])/3.0;
    double centroidY = (coords[1]+coords[3]+coords[5])/3.0;

    double eps;
    ierr = ComputeDielectric(centroidX, centroidY, &eps);CHKERRQ(ierr);
    epsilon->update(patch, *e_itor, &eps);
  }
  ALE_LOG_EVENT_END;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateEnergyDensity"
/*
  Creates a vector whose value is the energy on each element
*/
PetscErrorCode CreateEnergyDensity(ALE::Obj<ALE::Mesh> mesh, ALE::Obj<ALE::Mesh::field_type> field, ALE::Obj<ALE::Mesh::field_type> epsilon, Vec *energy)
{
  ALE::Obj<ALE::Mesh::sieve_type> topology = mesh->getTopology();
  ALE::Obj<ALE::Mesh::sieve_type::traits::heightSequence> elements = topology->heightStratum(0);
  ALE::Obj<ALE::Mesh::field_type> coordinates = mesh->getCoordinates();
  ALE::Obj<ALE::Mesh::field_type> e2 = mesh->getField("energy");
  ALE::Mesh::field_type::patch_type patch;
  std::string orderName("element");
  PetscInt       dim = mesh->getDimension();
  PetscReal      elementMat[NUM_BASIS_FUNCTIONS*NUM_BASIS_FUNCTIONS];
  PetscReal     *v0, *Jac, *Jinv, *t_der, *b_der;
  PetscReal      xi, eta, x_q, y_q, detJ;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ALE_LOG_EVENT_BEGIN;
  e2->setPatch(mesh->getTopology()->leaves(), patch);
  e2->setFiberDimensionByHeight(patch, 0, 1);
  e2->orderPatches();
  e2->createGlobalOrder();

  ierr = PetscMalloc(dim * sizeof(PetscReal), &v0);CHKERRQ(ierr);
  ierr = PetscMalloc(dim * sizeof(PetscReal), &t_der);CHKERRQ(ierr);
  ierr = PetscMalloc(dim * sizeof(PetscReal), &b_der);CHKERRQ(ierr);
  ierr = PetscMalloc(dim*dim * sizeof(PetscReal), &Jac);CHKERRQ(ierr);
  ierr = PetscMalloc(dim*dim * sizeof(PetscReal), &Jinv);CHKERRQ(ierr);
  for(ALE::Mesh::sieve_type::traits::heightSequence::iterator e_itor = elements->begin(); e_itor != elements->end(); ++e_itor) {
    const double *phi = field->restrict(orderName, *e_itor);
    double eps = epsilon->restrict(patch, *e_itor)[0];
    double elementEnergy = 0.0;

    ierr = ElementGeometry(mesh, *e_itor, v0, Jac, Jinv, &detJ);CHKERRQ(ierr);
    /* Element integral */
    ierr = PetscMemzero(elementMat, NUM_BASIS_FUNCTIONS*NUM_BASIS_FUNCTIONS*sizeof(PetscScalar));CHKERRQ(ierr);
    for(int q = 0; q < NUM_QUADRATURE_POINTS; q++) {
      xi = points[q*2+0] + 1.0;
      eta = points[q*2+1] + 1.0;
      x_q = Jac[0]*xi + Jac[1]*eta + v0[0];
      y_q = Jac[2]*xi + Jac[3]*eta + v0[1];
      for(int f = 0; f < NUM_BASIS_FUNCTIONS; f++) {
        t_der[0] = Jinv[0]*BasisDerivatives[(q*NUM_BASIS_FUNCTIONS+f)*2+0] + Jinv[2]*BasisDerivatives[(q*NUM_BASIS_FUNCTIONS+f)*2+1];
        t_der[1] = Jinv[1]*BasisDerivatives[(q*NUM_BASIS_FUNCTIONS+f)*2+0] + Jinv[3]*BasisDerivatives[(q*NUM_BASIS_FUNCTIONS+f)*2+1];
        for(int g = 0; g < NUM_BASIS_FUNCTIONS; g++) {
          b_der[0] = Jinv[0]*BasisDerivatives[(q*NUM_BASIS_FUNCTIONS+g)*2+0] + Jinv[2]*BasisDerivatives[(q*NUM_BASIS_FUNCTIONS+g)*2+1];
          b_der[1] = Jinv[1]*BasisDerivatives[(q*NUM_BASIS_FUNCTIONS+g)*2+0] + Jinv[3]*BasisDerivatives[(q*NUM_BASIS_FUNCTIONS+g)*2+1];
          elementMat[f*NUM_BASIS_FUNCTIONS+g] += eps*(t_der[0]*b_der[0] + t_der[1]*b_der[1])*weights[q]*detJ;
        }
      }
    }
    for(int f = 0; f < NUM_BASIS_FUNCTIONS; f++) {
      for(int g = 0; g < NUM_BASIS_FUNCTIONS; g++) {
        elementEnergy += phi[f]*elementMat[f*NUM_BASIS_FUNCTIONS+g]*phi[g];
      }
    }
    e2->update(patch, *e_itor, &elementEnergy);
  }
  ierr = PetscFree(v0);CHKERRQ(ierr);
  ierr = PetscFree(t_der);CHKERRQ(ierr);
  ierr = PetscFree(b_der);CHKERRQ(ierr);
  ierr = PetscFree(Jac);CHKERRQ(ierr);
  ierr = PetscFree(Jinv);CHKERRQ(ierr);

  VecScatter injection;
  ierr = MeshCreateVector(mesh, mesh->getBundle(mesh->getDimension()), energy);CHKERRQ(ierr);
  ierr = MeshGetGlobalScatter(mesh, "energy", *energy, &injection);CHKERRQ(ierr);

  Vec locEnergy;
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, e2->getSize(patch), e2->restrict(patch), &locEnergy);CHKERRQ(ierr);
  ierr = VecScatterBegin(injection, locEnergy, *energy, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(injection, locEnergy, *energy, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecDestroy(locEnergy);CHKERRQ(ierr);
  ALE_LOG_EVENT_END;
  PetscFunctionReturn(0);
}
