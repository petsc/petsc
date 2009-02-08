// This example will solve the Bratu problem eventually
static char help[] = "This example solves the Bratu problem.\n\n";

#include <petscmesh.hh>
#include <petscmesh_viewers.hh>
#include <petscdmmg.h>

#include <Selection.hh>

using ALE::Obj;
typedef enum {RUN_FULL, RUN_MESH} RunType;
typedef enum {NEUMANN, DIRICHLET} BCType;
typedef enum {ASSEMBLY_FULL, ASSEMBLY_STORED, ASSEMBLY_CALCULATED} AssemblyType;
typedef union {SectionReal section; Vec vec;} ExactSolType;

typedef struct {
  PetscInt      debug;                       // The debugging level
  RunType       run;                         // The run type
  PetscInt      dim;                         // The topological mesh dimension
  PetscTruth    structured;                  // Use a structured mesh
  PetscTruth    generateMesh;                // Generate the unstructure mesh
  PetscTruth    interpolate;                 // Generate intermediate mesh elements
  PetscReal     refinementLimit;             // The largest allowable cell volume
  PetscReal     particleRadius;              // The radius of the charged particle
  PetscInt      particleEdges;               // The number of edges along the particle
  char          baseFilename[2048];          // The base filename for mesh files
  PetscScalar (*func)(const double []);      // The function to project
  BCType        bcType;                      // The type of boundary conditions
  PetscScalar (*exactFunc)(const double []); // The exact solution function
  ExactSolType  exactSol;                    // The discrete exact solution
  AssemblyType  operatorAssembly;            // The type of operator assembly 
  double        lambda;                      // The parameter controlling nonlinearity
  // Boundary
  Mesh          bdMesh;
  // Cell geometry data
  double       *v0;                          // The coordinates of the first vertex in a cell
  double       *J;                           // The linear transform from the reference cell to a cell (Jacobian)
  double       *invJ;                        // The inverse of J
  double        detJ;                        // The determinant of J
  double       *coords;                      // The coordinates of a quadrature point
  // Face geometry data
  double       *fInvJ;                       // The inverse of J
  double        fDetJ;                       // The determinant of J
  double       *normal;                      // The normal to the active face
  double       *tangent;                     // The tangent to the active face
  // Physical Data
  double        waterEpsilon;                // The dielectric constant for the water
  double        particleEpsilon;             // The dielectric constant for the water
} Options;

#include "electrostatic_quadrature.h"

PetscScalar lambda  = 0.0;
PetscScalar radius  = 0.0;
PetscScalar epsilon = 0.0;
PetscScalar E_0     = 1.0;

PetscScalar zero(const double x[]) {
  return 0.0;
}

PetscScalar constant(const double x[]) {
  return -4.0;
}

// This only works for E_0 = 1 and e_w = 1
PetscScalar constantField_2d(const double x[]) {
  const double r = sqrt((x[0] - 0.5)*(x[0] - 0.5) + (x[1] - 0.5)*(x[1] - 0.5));

  if (r >= radius) {
    const double ratio = radius/r;

    return -(x[0] - 0.5) + ((epsilon - 1.0)/(epsilon + 2.0))*ratio*ratio*ratio*(x[0] - 0.5);
  }
  return (-3.0/(epsilon + 2.0))*(x[0] - 0.5);
}

PetscScalar nonlinear_2d(const double x[]) {
  return -4.0 - lambda*PetscExpScalar(x[0]*x[0] + x[1]*x[1]);
}

PetscScalar linear_2d(const double x[]) {
  return -6.0*(x[0] - 0.5) - 6.0*(x[1] - 0.5);
}

PetscScalar quadratic_2d(const double x[]) {
  return x[0]*x[0] + x[1]*x[1];
}

PetscScalar cubic_2d(const double x[]) {
  return x[0]*x[0]*x[0] - 1.5*x[0]*x[0] + x[1]*x[1]*x[1] - 1.5*x[1]*x[1] + 0.5;
}

// This only works for E_0 = 1 and e_w = 1
PetscScalar constantField_3d(const double x[]) {
  const double r = sqrt((x[0] - 0.5)*(x[0] - 0.5) + (x[1] - 0.5)*(x[1] - 0.5) + (x[2] - 0.5)*(x[2] - 0.5));

  if (r >= radius) {
    const double ratio = radius/r;

    return -(x[0] - 0.5) + ((epsilon - 1.0)/(epsilon + 2.0))*ratio*ratio*ratio*(x[0] - 0.5);
  }
  return (-3.0/(epsilon + 2.0))*(x[0] - 0.5);
}

PetscScalar nonlinear_3d(const double x[]) {
  return -4.0 - lambda*PetscExpScalar((2.0/3.0)*(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]));
}

PetscScalar linear_3d(const double x[]) {
  return -6.0*(x[0] - 0.5) - 6.0*(x[1] - 0.5) - 6.0*(x[2] - 0.5);
}

PetscScalar quadratic_3d(const double x[]) {
  return (2.0/3.0)*(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
}

PetscScalar cubic_3d(const double x[]) {
  return x[0]*x[0]*x[0] - 1.5*x[0]*x[0] + x[1]*x[1]*x[1] - 1.5*x[1]*x[1] + x[2]*x[2]*x[2] - 1.5*x[2]*x[2] + 0.75;
}

PetscScalar cos_x(const double x[]) {
  return cos(2.0*PETSC_PI*x[0]);
}

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, Options *options)
{
  const char    *runTypes[3] = {"full", "test", "mesh"};
  const char    *bcTypes[2]  = {"neumann", "dirichlet"};
  const char    *asTypes[4]  = {"full", "stored", "calculated"};
  ostringstream  filename;
  PetscInt       run, bc, as;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->debug            = 0;
  options->run              = RUN_FULL;
  options->dim              = 2;
  options->structured       = PETSC_FALSE;
  options->generateMesh     = PETSC_TRUE;
  options->interpolate      = PETSC_TRUE;
  options->refinementLimit  = 0.0;
  options->particleRadius   = 0.3;
  options->particleEdges    = 10;
  options->bcType           = DIRICHLET;
  options->operatorAssembly = ASSEMBLY_FULL;
  options->lambda           = 0.0;
  options->waterEpsilon     = 1.0;
  options->particleEpsilon  = 1.0;

  ierr = PetscOptionsBegin(comm, "", "Bratu Problem Options", "DMMG");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-debug", "The debugging level", "bratu.cxx", options->debug, &options->debug, PETSC_NULL);CHKERRQ(ierr);
    run = options->run;
    ierr = PetscOptionsEList("-run", "The run type", "bratu.cxx", runTypes, 3, runTypes[options->run], &run, PETSC_NULL);CHKERRQ(ierr);
    options->run = (RunType) run;
    ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "bratu.cxx", options->dim, &options->dim, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-structured", "Use a structured mesh", "bratu.cxx", options->structured, &options->structured, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-generate", "Generate the unstructured mesh", "bratu.cxx", options->generateMesh, &options->generateMesh, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-interpolate", "Generate intermediate mesh elements", "bratu.cxx", options->interpolate, &options->interpolate, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-refinement_limit", "The largest allowable cell volume", "bratu.cxx", options->refinementLimit, &options->refinementLimit, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-particle_radius", "The radius of the charged particle", "bratu.cxx", options->particleRadius, &options->particleRadius, PETSC_NULL);CHKERRQ(ierr);
    radius = options->particleRadius;
    ierr = PetscOptionsInt("-particle_edges", "The number of edges around charged particle", "bratu.cxx", options->particleEdges, &options->particleEdges, PETSC_NULL);CHKERRQ(ierr);
    filename << "data/bratu_" << options->dim <<"d";
    ierr = PetscStrcpy(options->baseFilename, filename.str().c_str());CHKERRQ(ierr);
    ierr = PetscOptionsString("-base_filename", "The base filename for mesh files", "bratu.cxx", options->baseFilename, options->baseFilename, 2048, PETSC_NULL);CHKERRQ(ierr);
    bc = options->bcType;
    ierr = PetscOptionsEList("-bc_type","Type of boundary condition","bratu.cxx",bcTypes,2,bcTypes[options->bcType],&bc,PETSC_NULL);CHKERRQ(ierr);
    options->bcType = (BCType) bc;
    as = options->operatorAssembly;
    ierr = PetscOptionsEList("-assembly_type","Type of operator assembly","bratu.cxx",asTypes,3,asTypes[options->operatorAssembly],&as,PETSC_NULL);CHKERRQ(ierr);
    options->operatorAssembly = (AssemblyType) as;
    ierr = PetscOptionsReal("-lambda", "The parameter controlling nonlinearity", "bratu.cxx", options->lambda, &options->lambda, PETSC_NULL);CHKERRQ(ierr);
    lambda = options->lambda;
    ierr = PetscOptionsReal("-water_epsilon", "The dielectric constant of water", "bratu.cxx", options->waterEpsilon, &options->waterEpsilon, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-particle_epsilon", "The dielectric constant of the charged particle", "bratu.cxx", options->particleEpsilon, &options->particleEpsilon, PETSC_NULL);CHKERRQ(ierr);
    epsilon = options->particleEpsilon;
    ierr = PetscOptionsReal("-background_field", "The background field", "bratu.cxx", E_0, &E_0, PETSC_NULL);CHKERRQ(ierr);
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


PetscErrorCode CreateParticleLabel(Mesh mesh, Options *options);

#undef __FUNCT__
#define __FUNCT__ "CreateParticle"
PetscErrorCode CreateParticle(Mesh mesh, SectionInt *particle)
{
  Obj<ALE::Mesh> m;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = MeshGetCellSectionInt(mesh, 1, particle);CHKERRQ(ierr);
  const Obj<ALE::Mesh::label_sequence>&     cells = m->heightStratum(0);
  const ALE::Mesh::label_sequence::iterator begin = cells->begin();
  const ALE::Mesh::label_sequence::iterator end   = cells->end();
  const Obj<ALE::Mesh::label_type>&         label = m->getLabel("particle");

  for(ALE::Mesh::label_sequence::iterator c_iter = begin; c_iter != end; ++c_iter) {
    const int p = m->getValue(label, *c_iter);

    ierr = SectionIntUpdate(*particle, *c_iter, &p);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ViewSection"
PetscErrorCode ViewSection(Mesh mesh, SectionReal section, const char filename[])
{
  MPI_Comm       comm;
  SectionInt     partition, particle;
  PetscViewer    viewer;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) mesh, &comm);CHKERRQ(ierr);
  ierr = PetscViewerCreate(comm, &viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
  ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(viewer, filename);CHKERRQ(ierr);
  ierr = MeshView(mesh, viewer);CHKERRQ(ierr);
  ierr = SectionRealView(section, viewer);CHKERRQ(ierr);
  ierr = CreatePartition(mesh, &partition);CHKERRQ(ierr);
  ierr = CreateParticle(mesh, &particle);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_VTK_CELL);CHKERRQ(ierr);
  ierr = SectionIntView(partition, viewer);CHKERRQ(ierr);
  ierr = SectionIntView(particle,  viewer);CHKERRQ(ierr);
  ierr = SectionIntDestroy(partition);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ViewMesh"
PetscErrorCode ViewMesh(Mesh mesh, const char filename[])
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
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  ierr = SectionIntDestroy(partition);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateMesh"
PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm, Options *options)
{
  const int      dim = options->dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (options->structured) {
    DA       da;
    PetscInt dof = 1;

    if (dim == 2) {
      ierr = DACreate2d(comm, DA_NONPERIODIC, DA_STENCIL_BOX, -3, -3, PETSC_DECIDE, PETSC_DECIDE, dof, 1, PETSC_NULL, PETSC_NULL, &da);CHKERRQ(ierr);
    } else if (dim == 3) {
      ierr = DACreate3d(comm, DA_NONPERIODIC, DA_STENCIL_BOX, -3, -3, -3, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, dof, 1, PETSC_NULL, PETSC_NULL, PETSC_NULL, &da);CHKERRQ(ierr);
    } else {
      SETERRQ1(PETSC_ERR_SUP, "Dimension not supported: %d", dim);
    }
    ierr = DASetUniformCoordinates(da, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);CHKERRQ(ierr);
    *dm = (DM) da;
  } else {
    Mesh        mesh;
    PetscTruth  view;
    PetscMPIInt size;

    if (options->generateMesh) {
      ierr = MeshCreate(comm, &options->bdMesh);CHKERRQ(ierr);
      if (dim == 2) {
        double lower[2] = {0.0, 0.0};
        double upper[2] = {1.0, 1.0};
        int    edges[2] = {1, 1};

        Obj<ALE::Mesh> mB = ALE::MeshBuilder::createParticleInSquareBoundary(comm, lower, upper, edges, options->particleRadius, options->particleEdges, options->debug);
        ierr = MeshSetMesh(options->bdMesh, mB);CHKERRQ(ierr);
      } else if (dim == 3) {
        double lower[3] = {0.0, 0.0, 0.0};
        double upper[3] = {1.0, 1.0, 1.0};
        int    faces[3] = {1, 1, 1};

        Obj<ALE::Mesh> mB = ALE::MeshBuilder::createParticleInCubeBoundary(comm, lower, upper, faces, options->particleRadius, options->particleEdges, options->particleEdges, options->debug);
        ierr = MeshSetMesh(options->bdMesh, mB);CHKERRQ(ierr);
      } else {
        SETERRQ1(PETSC_ERR_SUP, "Dimension not supported: %d", dim);
      }
      ierr = MeshGenerate(options->bdMesh, options->interpolate, &mesh);CHKERRQ(ierr);
    } else {
      std::string baseFilename(options->baseFilename);
      std::string coordFile = baseFilename+".nodes";
      std::string adjFile   = baseFilename+".lcon";

      ierr = MeshCreatePCICE(comm, dim, coordFile.c_str(), adjFile.c_str(), options->interpolate, PETSC_NULL, &mesh);CHKERRQ(ierr);
    }
    ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
    if (size > 1) {
      Mesh parallelMesh;

      ierr = MeshDistribute(mesh, PETSC_NULL, &parallelMesh);CHKERRQ(ierr);
      ierr = MeshDestroy(mesh);CHKERRQ(ierr);
      mesh = parallelMesh;
    }
    if (options->refinementLimit > 0.0) {
      Mesh refinedMesh;

      ierr = MeshRefine(mesh, options->refinementLimit, options->interpolate, &refinedMesh);CHKERRQ(ierr);
      ierr = MeshDestroy(mesh);CHKERRQ(ierr);
      mesh = refinedMesh;
    }
    if (options->bcType == DIRICHLET) {
      Obj<ALE::Mesh> m;

      ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
      m->markBoundaryCells("marker");
    }
    ierr = PetscOptionsHasName(PETSC_NULL, "-mesh_view_vtk", &view);CHKERRQ(ierr);
    if (view) {ierr = ViewMesh(mesh, "electrostatic.vtk");CHKERRQ(ierr);}
    ierr = PetscOptionsHasName(PETSC_NULL, "-mesh_view", &view);CHKERRQ(ierr);
    if (view) {
      Obj<ALE::Mesh> m;
      ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
      m->view("Mesh");
    }
    ierr = PetscMalloc7(dim,double,&options->coords,dim,double,&options->v0,dim*dim,double,&options->J,dim*dim,double,&options->invJ,(dim-1)*(dim-1),double,&options->fInvJ,dim,double,&options->normal,dim*(dim-1),double,&options->tangent);CHKERRQ(ierr);
    *dm = (DM) mesh;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DestroyMesh"
PetscErrorCode DestroyMesh(DM dm, Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (options->structured) {
    ierr = DADestroy((DA) dm);CHKERRQ(ierr);
  } else {
    ierr = MeshDestroy(options->bdMesh);CHKERRQ(ierr);
    ierr = MeshDestroy((Mesh) dm);CHKERRQ(ierr);
    ierr = PetscFree7(options->coords, options->v0, options->J, options->invJ, options->fInvJ, options->normal, options->tangent);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DestroyExactSolution"
PetscErrorCode DestroyExactSolution(ExactSolType sol, Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (options->structured) {
    ierr = VecDestroy(sol.vec);CHKERRQ(ierr);
  } else {
    ierr = SectionRealDestroy(sol.section);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Function_Structured_2d"
PetscErrorCode Function_Structured_2d(DALocalInfo *info, PetscScalar *x[], PetscScalar *f[], void *ctx)
{
  Options       *options = (Options *) ctx;
  PetscScalar  (*func)(const double *) = options->func;
  DA             coordDA;
  Vec            coordinates;
  DACoor2d     **coords;
  PetscInt       i, j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DAGetCoordinateDA(info->da, &coordDA);CHKERRQ(ierr);
  ierr = DAGetCoordinates(info->da, &coordinates);CHKERRQ(ierr);
  ierr = DAVecGetArray(coordDA, coordinates, &coords);CHKERRQ(ierr);
  for(j = info->ys; j < info->ys+info->ym; j++) {
    for(i = info->xs; i < info->xs+info->xm; i++) {
      f[j][i] = func((PetscReal *) &coords[j][i]);
    }
  }
  ierr = DAVecRestoreArray(coordDA, coordinates, &coords);CHKERRQ(ierr);
  PetscFunctionReturn(0); 
}

#undef __FUNCT__
#define __FUNCT__ "Function_Structured_3d"
PetscErrorCode Function_Structured_3d(DALocalInfo *info, PetscScalar **x[], PetscScalar **f[], void *ctx)
{
  Options       *options = (Options *) ctx;
  PetscScalar  (*func)(const double *) = options->func;
  DA             coordDA;
  Vec            coordinates;
  DACoor3d    ***coords;
  PetscInt       i, j, k;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DAGetCoordinateDA(info->da, &coordDA);CHKERRQ(ierr);
  ierr = DAGetCoordinates(info->da, &coordinates);CHKERRQ(ierr);
  ierr = DAVecGetArray(coordDA, coordinates, &coords);CHKERRQ(ierr);
  for(k = info->zs; k < info->zs+info->zm; k++) {
    for(j = info->ys; j < info->ys+info->ym; j++) {
      for(i = info->xs; i < info->xs+info->xm; i++) {
        f[k][j][i] = func((PetscReal *) &coords[k][j][i]);
      }
    }
  }
  ierr = DAVecRestoreArray(coordDA, coordinates, &coords);CHKERRQ(ierr);
  PetscFunctionReturn(0); 
}

#undef __FUNCT__
#define __FUNCT__ "Function_Unstructured"
PetscErrorCode Function_Unstructured(Mesh mesh, SectionReal section, void *ctx)
{
  Options       *options = (Options *) ctx;
  PetscScalar  (*func)(const double *) = options->func;
  Obj<ALE::Mesh> m;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  const Obj<ALE::Mesh::real_section_type>& coordinates = m->getRealSection("coordinates");
  const Obj<ALE::Mesh::label_sequence>&    vertices    = m->depthStratum(0);

  for(ALE::Mesh::label_sequence::iterator v_iter = vertices->begin(); v_iter != vertices->end(); ++v_iter) {
    const ALE::Mesh::real_section_type::value_type *coords = coordinates->restrictPoint(*v_iter);
    const PetscScalar                               value  = (*func)(coords);

    ierr = SectionRealUpdate(section, *v_iter, &value);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Rhs_Structured_2d_FD"
PetscErrorCode Rhs_Structured_2d_FD(DALocalInfo *info, PetscScalar *x[], PetscScalar *f[], void *ctx)
{
  Options       *options = (Options *) ctx;
  PetscScalar  (*func)(const double *)   = options->func;
  PetscScalar  (*bcFunc)(const double *) = options->exactFunc;
  const double   lambda                  = options->lambda;
  DA             coordDA;
  Vec            coordinates;
  DACoor2d     **coords;
  PetscReal      hxa, hxb, hx, hya, hyb, hy;
  PetscInt       ie = info->xs+info->xm;
  PetscInt       je = info->ys+info->ym;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DAGetCoordinateDA(info->da, &coordDA);CHKERRQ(ierr);
  ierr = DAGetGhostedCoordinates(info->da, &coordinates);CHKERRQ(ierr);
  ierr = DAVecGetArray(coordDA, coordinates, &coords);CHKERRQ(ierr);
  // Loop over stencils
  for(int j = info->ys; j < je; j++) {
    for(int i = info->xs; i < ie; i++) {
      if (i == 0 || j == 0 || i == info->mx-1 || j == info->my-1) {
        f[j][i] = x[j][i] - bcFunc((PetscReal *) &coords[j][i]);
      } else {
        hya = coords[j+1][i].y - coords[j][i].y;
        hyb = coords[j][i].y   - coords[j-1][i].y;
        hxa = coords[j][i+1].x - coords[j][i].x;
        hxb = coords[j][i].x   - coords[j][i-1].x;
        hy  = 0.5*(hya+hyb);
        hx  = 0.5*(hxa+hxb);
        f[j][i] = -func((const double *) &coords[j][i])*hx*hy -
          ((x[j][i+1] - x[j][i])/hxa - (x[j][i] - x[j][i-1])/hxb)*hy -
          ((x[j+1][i] - x[j][i])/hya - (x[j][i] - x[j-1][i])/hyb)*hx -
          lambda*hx*hy*PetscExpScalar(x[j][i]);
      }
    }
  }
  ierr = DAVecRestoreArray(coordDA, coordinates, &coords);CHKERRQ(ierr);
  PetscFunctionReturn(0); 
}

#undef __FUNCT__
#define __FUNCT__ "Rhs_Structured_3d_FD"
PetscErrorCode Rhs_Structured_3d_FD(DALocalInfo *info, PetscScalar **x[], PetscScalar **f[], void *ctx)
{
  Options       *options = (Options *) ctx;
  PetscScalar  (*func)(const double *)   = options->func;
  PetscScalar  (*bcFunc)(const double *) = options->exactFunc;
  const double   lambda                  = options->lambda;
  DA             coordDA;
  Vec            coordinates;
  DACoor3d    ***coords;
  PetscReal      hxa, hxb, hx, hya, hyb, hy, hza, hzb, hz;
  PetscInt       ie = info->xs+info->xm;
  PetscInt       je = info->ys+info->ym;
  PetscInt       ke = info->zs+info->zm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DAGetCoordinateDA(info->da, &coordDA);CHKERRQ(ierr);
  ierr = DAGetGhostedCoordinates(info->da, &coordinates);CHKERRQ(ierr);
  ierr = DAVecGetArray(coordDA, coordinates, &coords);CHKERRQ(ierr);
  // Loop over stencils
  for(int k = info->zs; k < ke; k++) {
    for(int j = info->ys; j < je; j++) {
      for(int i = info->xs; i < ie; i++) {
        if (i == 0 || j == 0 || k == 0 || i == info->mx-1 || j == info->my-1 || k == info->mz-1) {
          f[k][j][i] = x[k][j][i] - bcFunc((PetscReal *) &coords[k][j][i]);
        } else {
          hza = coords[k+1][j][i].z - coords[k][j][i].z;
          hzb = coords[k][j][i].z   - coords[k-1][j][i].z;
          hya = coords[k][j+1][i].y - coords[k][j][i].y;
          hyb = coords[k][j][i].y   - coords[k][j-1][i].y;
          hxa = coords[k][j][i+1].x - coords[k][j][i].x;
          hxb = coords[k][j][i].x   - coords[k][j][i-1].x;
          hz  = 0.5*(hza+hzb);
          hy  = 0.5*(hya+hyb);
          hx  = 0.5*(hxa+hxb);
        f[k][j][i] = -func((const double *) &coords[k][j][i])*hx*hy*hz -
          ((x[k][j][i+1] - x[k][j][i])/hxa - (x[k][j][i] - x[k][j][i-1])/hxb)*hy*hz -
          ((x[k][j+1][i] - x[k][j][i])/hya - (x[k][j][i] - x[k][j-1][i])/hyb)*hx*hz - 
          ((x[k+1][j][i] - x[k][j][i])/hza - (x[k][j][i] - x[k-1][j][i])/hzb)*hx*hy -
          lambda*hx*hy*hz*PetscExpScalar(x[k][j][i]);
        }
      }
    }
  }
  ierr = DAVecRestoreArray(coordDA, coordinates, &coords);CHKERRQ(ierr);
  PetscFunctionReturn(0); 
}

#undef __FUNCT__
#define __FUNCT__ "computeFaceGeometry"
PetscErrorCode computeFaceGeometry(const Obj<ALE::Mesh>& mesh, const ALE::Mesh::point_type cell, const ALE::Mesh::point_type face, const int f, const Obj<ALE::Mesh::arrow_section_type>& orientation, double invJ[], double& detJ, double normal[], double tangent[]) {
  const ALE::Mesh::arrow_section_type::point_type arrow(cell, face);
  const bool   reversed = (orientation->restrictPoint(arrow)[0] == -2);
  const int    dim      = mesh->getDimension();
  PetscScalar  norm     = 0.0;
  double      *vec      = tangent;

  PetscFunctionBegin;
  if (f == 0) {
    vec[0] = 0.0;        vec[1] = -1.0;
  } else if (f == 1) {
    vec[0] = 0.70710678; vec[1] = 0.70710678;
  } else if (f == 2) {
    vec[0] = -1.0;       vec[1] = 0.0;
  }
  for(int d = 0; d < dim; ++d) {
    normal[d] = 0.0;
    for(int e = 0; e < dim; ++e) normal[d] += invJ[e*dim+d]*vec[e];
    if (reversed) normal[d] = -normal[d];
    norm += normal[d]*normal[d];
  }
  norm = std::sqrt(norm);
  for(int d = 0; d < dim; ++d) {
    normal[d] /= norm;
  }
  // 2D only right now
  tangent[0] =  normal[1];
  tangent[1] = -normal[0];
  if (mesh->debug()) {
    std::cout << "Cell: " << cell << " Face: " << face << "("<<f<<")" << std::endl;
    for(int d = 0; d < dim; ++d) {
      std::cout << "Normal["<<d<<"]: " << normal[d] << " Tangent["<<d<<"]: " << tangent[d] << std::endl;
    }
  }
  // Now get 1D Jacobian info
  invJ[0] = invJ[0]*invJ[3] - invJ[1]*invJ[2];
  detJ = 1.0/invJ[0];
  PetscFunctionReturn(0);
}

PetscErrorCode cellResidual(const Obj<ALE::Discretization>& disc, PetscScalar x[], PetscScalar t_der[], PetscScalar b_der[], PetscScalar elemVec[], PetscScalar elemMat[], double epsilon, Options *options) {
  const int      dim    = options->dim;
  double        *v0     = options->v0;
  double        *J      = options->J;
  double        *invJ   = options->invJ;
  double        *coords = options->coords;
  double         detJ   = options->detJ;
  const int      numQuadPoints = disc->getQuadratureSize();
  const double  *quadPoints    = disc->getQuadraturePoints();
  const double  *quadWeights   = disc->getQuadratureWeights();
  const int      numBasisFuncs = disc->getBasisSize();
  const double  *basis         = disc->getBasis();
  const double  *basisDer      = disc->getBasisDerivatives();
  PetscScalar  (*func)(const double *) = options->func;
  const double   lambda                = options->lambda;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMemzero(elemVec, numBasisFuncs * sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMemzero(elemMat, numBasisFuncs*numBasisFuncs * sizeof(PetscScalar));CHKERRQ(ierr);
  // Loop over quadrature points
  for(int q = 0; q < numQuadPoints; ++q) {
    // Transform to real space coordinates
    for(int d = 0; d < dim; d++) {
      coords[d] = v0[d];
      for(int e = 0; e < dim; e++) {
        coords[d] += J[d*dim+e]*(quadPoints[q*dim+e] + 1.0);
      }
    }
    const PetscScalar funcVal  = (*func)(coords);
    PetscScalar       fieldVal = 0.0;

    for(int f = 0; f < numBasisFuncs; ++f) {
      fieldVal += x[f]*basis[q*numBasisFuncs+f];
    }
    // Loop over trial functions
    for(int f = 0; f < numBasisFuncs; ++f) {
      // Constant part
      elemVec[f] -= basis[q*numBasisFuncs+f]*funcVal*quadWeights[q]*detJ;
      // Linear part
      for(int d = 0; d < dim; ++d) {
        t_der[d] = 0.0;
        for(int e = 0; e < dim; ++e) t_der[d] += invJ[e*dim+d]*basisDer[(q*numBasisFuncs+f)*dim+e];
      }
      // Loop over basis functions
      for(int g = 0; g < numBasisFuncs; ++g) {
        // Linear part
        for(int d = 0; d < dim; ++d) {
          b_der[d] = 0.0;
          for(int e = 0; e < dim; ++e) b_der[d] += invJ[e*dim+d]*basisDer[(q*numBasisFuncs+g)*dim+e];
        }
        PetscScalar product = 0.0;
        for(int d = 0; d < dim; ++d) product += t_der[d]*b_der[d];
        elemMat[f*numBasisFuncs+g] += epsilon*product*quadWeights[q]*detJ;
      }
      // Nonlinear part
      elemVec[f] -= basis[q*numBasisFuncs+f]*lambda*PetscExpScalar(fieldVal)*quadWeights[q]*detJ;
    }
  }    
  // Add linear contribution
  for(int f = 0; f < numBasisFuncs; ++f) {
    for(int g = 0; g < numBasisFuncs; ++g) {
      elemVec[f] += elemMat[f*numBasisFuncs+g]*x[g];
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode faceResidual(const Obj<ALE::Discretization>& cellDisc, const Obj<ALE::Discretization>& disc, const PetscScalar x[], const PetscScalar cellBasisDer[], PetscScalar b_der[], PetscScalar elemVec[], PetscScalar elemMat[], double epsilon, Options *options) {
  const int      dim     = options->dim;
  double        *invJ    = options->invJ;
  double         detJ    = options->fDetJ;
  double        *normal  = options->normal;
  double        *tangent = options->tangent;
  const int      numQuadPoints = disc->getQuadratureSize();
  const double  *quadWeights   = disc->getQuadratureWeights();
  const int      numCellBasisFuncs = cellDisc->getBasisSize();
  const int      numBasisFuncs = disc->getBasisSize();
  const double  *basis         = disc->getBasis();
  //const double  *basisDer      = disc->getBasisDerivatives();
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (options->debug) {
    std::cout << ALE::Mesh::printMatrix(std::string("  Cell x"), numCellBasisFuncs, 1, x);
    std::cout << ALE::Mesh::printMatrix(std::string("  Cell invJ"), dim, dim, invJ);
    std::cout << "  Face detJ: " << detJ << std::endl;
  }
  ierr = PetscMemzero(elemVec, numBasisFuncs * sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMemzero(elemMat, numBasisFuncs*numCellBasisFuncs * sizeof(PetscScalar));CHKERRQ(ierr);
  // Loop over quadrature points
  for(int q = 0; q < numQuadPoints; ++q) {
    // Loop over trial functions
    for(int f = 0; f < numBasisFuncs; ++f) {
      // Linear part
      // Loop over basis functions
      for(int g = 0; g < numCellBasisFuncs; ++g) {
        // Linear part
        for(int d = 0; d < dim; ++d) {
          b_der[d] = 0.0;
          for(int e = 0; e < dim; ++e) b_der[d] += invJ[e*dim+d]*cellBasisDer[(q*numCellBasisFuncs+g)*dim+e];
        }
        PetscScalar productN = 0.0, productT = 0.0;
        for(int d = 0; d < dim; ++d) {
          productN += normal[d]*b_der[d];
          for(int e = 0; e < dim-1; ++e) {
            productT += tangent[e*dim+d]*b_der[d];
          }
        }
        elemMat[f*numCellBasisFuncs+g] += basis[q*numBasisFuncs+f]*(epsilon*productN + productT)*quadWeights[q]*detJ;
      }
    }
  }    
  if (options->debug) {
    std::cout << ALE::Mesh::printMatrix(std::string("  Face elemMat"), numBasisFuncs, numCellBasisFuncs, elemMat);
  }
  // Add linear contribution
  for(int f = 0; f < numBasisFuncs; ++f) {
    for(int g = 0; g < numCellBasisFuncs; ++g) {
      elemVec[f] += elemMat[f*numCellBasisFuncs+g]*x[g];
    }
  }
  PetscFunctionReturn(0);
}

static PetscScalar cellBasisDer2D[36] = {
    -0.5, -0.5, 0.5, 0.0, 0.0, 0.5,
    -0.5, -0.5, 0.5, 0.0, 0.0, 0.5,
    -0.5, -0.5, 0.5, 0.0, 0.0, 0.5,
    -0.5, -0.5, 0.5, 0.0, 0.0, 0.5,
    -0.5, -0.5, 0.5, 0.0, 0.0, 0.5,
    -0.5, -0.5, 0.5, 0.0, 0.0, 0.5
  };
static PetscScalar cellBasisDer3D[192] = {
    -0.5, -0.5, -0.5, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5,
    -0.5, -0.5, -0.5, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5,
    -0.5, -0.5, -0.5, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5,
    -0.5, -0.5, -0.5, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5,
    -0.5, -0.5, -0.5, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5,
    -0.5, -0.5, -0.5, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5,
    -0.5, -0.5, -0.5, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5,
    -0.5, -0.5, -0.5, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5,
    -0.5, -0.5, -0.5, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5,
    -0.5, -0.5, -0.5, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5,
    -0.5, -0.5, -0.5, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5,
    -0.5, -0.5, -0.5, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5,
    -0.5, -0.5, -0.5, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5,
    -0.5, -0.5, -0.5, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5,
    -0.5, -0.5, -0.5, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5,
    -0.5, -0.5, -0.5, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5
  };

#undef __FUNCT__
#define __FUNCT__ "Rhs_Unstructured"
PetscErrorCode Rhs_Unstructured(Mesh mesh, SectionReal X, SectionReal section, void *ctx)
{
  Options       *options = (Options *) ctx;
  Obj<ALE::Mesh> m;
  Obj<ALE::Mesh> bdM;
  PetscScalar   *cellBasisDer;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = MeshGetMesh(options->bdMesh, bdM);CHKERRQ(ierr);
  const Obj<ALE::Mesh::sieve_type>&         sieve         = m->getSieve();
  const Obj<ALE::Discretization>&           disc          = m->getDiscretization("u");
  const Obj<ALE::Discretization>&           bdDisc        = bdM->getDiscretization("u");
  const int                                 numBasisFuncs = disc->getBasisSize();
  const Obj<ALE::Mesh::real_section_type>&  coordinates   = m->getRealSection("coordinates");
  const int                                 dim           = m->getDimension();
  const int                                 debug         = options->debug;
  const int                                 offset        = bdDisc->getQuadratureSize()*disc->getBasisSize()*dim;
  PetscScalar *t_der, *b_der, *elemVec, *elemMat;

  if (dim == 2) {
    cellBasisDer = cellBasisDer2D;
  } else if (dim == 3) {
    cellBasisDer = cellBasisDer3D;
  }
  ierr = SectionRealZero(section);CHKERRQ(ierr);
  ierr = PetscMalloc4(dim,PetscScalar,&t_der,dim,PetscScalar,&b_der,numBasisFuncs,PetscScalar,&elemVec,numBasisFuncs*numBasisFuncs,PetscScalar,&elemMat);CHKERRQ(ierr);
  // Must create the label on coarser meshes if it is missing
  if (!m->hasLabel("particle")) {ierr = CreateParticleLabel(mesh, options);CHKERRQ(ierr);}
  // Loop over water cells
  const Obj<ALE::Mesh::label_sequence>&     waterCells = m->getLabelStratum("particle", 1);
  const ALE::Mesh::label_sequence::iterator wBegin     = waterCells->begin();
  const ALE::Mesh::label_sequence::iterator wEnd       = waterCells->end();

  for(ALE::Mesh::label_sequence::iterator c_iter = wBegin; c_iter != wEnd; ++c_iter) {
    PetscScalar *x;

    m->computeElementGeometry(coordinates, *c_iter, options->v0, options->J, options->invJ, options->detJ);
    if (options->detJ < 0) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %d", options->detJ, *c_iter);
    ierr = SectionRealRestrict(X, *c_iter, &x);CHKERRQ(ierr);
    ierr = cellResidual(disc, x, t_der, b_der, elemVec, elemMat, options->waterEpsilon, options);CHKERRQ(ierr);
    ierr = SectionRealUpdateAdd(section, *c_iter, elemVec);CHKERRQ(ierr);
    if (debug) {
      ostringstream title;
      title << "Water cell " << *c_iter << " elemVec";
      std::cout << m->printMatrix(title.str(), numBasisFuncs, 1, elemVec);
    }
  }
  if (debug) {ierr = SectionRealView(section, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);}
  // Loop over particle cells
  const Obj<ALE::Mesh::label_sequence>&     particleCells = m->getLabelStratum("particle", 2);
  const ALE::Mesh::label_sequence::iterator pBegin        = particleCells->begin();
  const ALE::Mesh::label_sequence::iterator pEnd          = particleCells->end();

  for(ALE::Mesh::label_sequence::iterator c_iter = pBegin; c_iter != pEnd; ++c_iter) {
    PetscScalar *x;

    m->computeElementGeometry(coordinates, *c_iter, options->v0, options->J, options->invJ, options->detJ);
    if (options->detJ < 0) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %d", options->detJ, *c_iter);
    ierr = SectionRealRestrict(X, *c_iter, &x);CHKERRQ(ierr);
    ierr = cellResidual(disc, x, t_der, b_der, elemVec, elemMat, options->particleEpsilon, options);CHKERRQ(ierr);
    ierr = SectionRealUpdateAdd(section, *c_iter, elemVec);CHKERRQ(ierr);
    if (debug) {
      ostringstream title;
      title << "Particle cell " << *c_iter << " elemVec";
      std::cout << m->printMatrix(title.str(), numBasisFuncs, 1, elemVec);
    }
  }
  if (debug) {ierr = SectionRealView(section, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);}
  // Loop over water boundary
  const Obj<ALE::Mesh::label_type>&         particleBd   = m->getLabel("particleBd");
  const Obj<ALE::Mesh::label_sequence>&     waterBdCells = m->getLabelStratum("particleBd", 3);
  const ALE::Mesh::label_sequence::iterator wBdBegin     = waterBdCells->begin();
  const ALE::Mesh::label_sequence::iterator wBdEnd       = waterBdCells->end();

  for(ALE::Mesh::label_sequence::iterator c_iter = wBdBegin; c_iter != wBdEnd; ++c_iter) {
    const Obj<ALE::Mesh::sieve_type::traits::coneSequence>&     cone   = sieve->cone(*c_iter);
    const ALE::Mesh::sieve_type::traits::coneSequence::iterator cBegin = cone->begin();
    const ALE::Mesh::sieve_type::traits::coneSequence::iterator cEnd   = cone->end();
    int                                                         f      = 0;

    m->computeElementGeometry(coordinates, *c_iter, options->v0, options->J, options->invJ, options->detJ);
    if (options->detJ < 0) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for face %d", options->detJ, *c_iter);
    for(ALE::Mesh::sieve_type::traits::coneSequence::iterator f_iter = cBegin; f_iter != cEnd; ++f_iter, ++f) {
      if (m->getValue(particleBd, *f_iter)) {
        PetscScalar *x;

        m->computeFaceGeometry(*c_iter, *f_iter, f, options->invJ, options->fInvJ, options->fDetJ, options->normal, options->tangent);
        ierr = SectionRealRestrict(X, *c_iter, &x);CHKERRQ(ierr);
        ierr = faceResidual(disc, bdDisc, x, &cellBasisDer[f*offset], b_der, elemVec, elemMat, options->waterEpsilon, options);CHKERRQ(ierr);
        ierr = SectionRealUpdateAdd(section, *f_iter, elemVec);CHKERRQ(ierr);
        if (debug) {
          ostringstream title;
          title << "Water face " << *f_iter << " elemVec";
          std::cout << m->printMatrix(title.str(), bdDisc->getBasisSize(), 1, elemVec);
        }
      }
    }
  }
  if (debug) {ierr = SectionRealView(section, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);}
  // Loop over particle boundary
  const Obj<ALE::Mesh::label_sequence>&     particleBdCells = m->getLabelStratum("particleBd", 2);
  const ALE::Mesh::label_sequence::iterator pBdBegin        = particleBdCells->begin();
  const ALE::Mesh::label_sequence::iterator pBdEnd          = particleBdCells->end();

  for(ALE::Mesh::label_sequence::iterator c_iter = pBdBegin; c_iter != pBdEnd; ++c_iter) {
    const Obj<ALE::Mesh::sieve_type::traits::coneSequence>&     cone   = sieve->cone(*c_iter);
    const ALE::Mesh::sieve_type::traits::coneSequence::iterator cBegin = cone->begin();
    const ALE::Mesh::sieve_type::traits::coneSequence::iterator cEnd   = cone->end();
    int                                                         f      = 0;
    PetscScalar *x;

    m->computeElementGeometry(coordinates, *c_iter, options->v0, options->J, options->invJ, options->detJ);
    if (options->detJ < 0) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for face %d", options->detJ, *c_iter);
    for(ALE::Mesh::sieve_type::traits::coneSequence::iterator f_iter = cBegin; f_iter != cEnd; ++f_iter, ++f) {
      if (m->getValue(particleBd, *f_iter)) {
        m->computeFaceGeometry(*c_iter, *f_iter, f, options->invJ, options->fInvJ, options->fDetJ, options->normal, options->tangent);
        ierr = SectionRealRestrict(X, *c_iter, &x);CHKERRQ(ierr);
        for(int e = 1; e < dim-1; ++e) {
          for(int d = 0; d < dim; ++d) options->tangent[e*dim+d] *= -1.0;
        }
        ierr = faceResidual(disc, bdDisc, x, &cellBasisDer[f*offset], b_der, elemVec, elemMat, options->particleEpsilon, options);CHKERRQ(ierr);
        ierr = SectionRealUpdateAdd(section, *f_iter, elemVec);CHKERRQ(ierr);
        if (debug) {
          ostringstream title;
          title << "Particle face " << *f_iter << " elemVec";
          std::cout << m->printMatrix(title.str(), bdDisc->getBasisSize(), 1, elemVec);
        }
      }
    }
  }
  if (debug) {ierr = SectionRealView(section, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);}
  // Cleanup
  ierr = PetscFree4(t_der,b_der,elemVec,elemMat);CHKERRQ(ierr);
  // Exchange neighbors
  ierr = SectionRealComplete(section);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CalculateError"
PetscErrorCode CalculateError(Mesh mesh, SectionReal X, double *error, void *ctx)
{
  Options       *options = (Options *) ctx;
  PetscScalar  (*func)(const double *) = options->exactFunc;
  Obj<ALE::Mesh> m;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  const Obj<ALE::Discretization>&          disc          = m->getDiscretization("u");
  const int                                numQuadPoints = disc->getQuadratureSize();
  const double                            *quadPoints    = disc->getQuadraturePoints();
  const double                            *quadWeights   = disc->getQuadratureWeights();
  const int                                numBasisFuncs = disc->getBasisSize();
  const double                            *basis         = disc->getBasis();
  const Obj<ALE::Mesh::real_section_type>& coordinates   = m->getRealSection("coordinates");
  const Obj<ALE::Mesh::label_sequence>&    cells         = m->heightStratum(0);
  const int                                dim           = m->getDimension();
  double *coords, *v0, *J, *invJ, detJ;
  double  localError = 0.0;

  ierr = PetscMalloc4(dim,double,&coords,dim,double,&v0,dim*dim,double,&J,dim*dim,double,&invJ);CHKERRQ(ierr);
  // Loop over cells
  for(ALE::Mesh::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
    PetscScalar *x;
    double       elemError = 0.0;

    m->computeElementGeometry(coordinates, *c_iter, v0, J, invJ, detJ);
    ierr = SectionRealRestrict(X, *c_iter, &x);CHKERRQ(ierr);
    // Loop over quadrature points
    for(int q = 0; q < numQuadPoints; ++q) {
      for(int d = 0; d < dim; d++) {
        coords[d] = v0[d];
        for(int e = 0; e < dim; e++) {
          coords[d] += J[d*dim+e]*(quadPoints[q*dim+e] + 1.0);
        }
      }
      const PetscScalar funcVal = (*func)(coords);

      double interpolant = 0.0;
      for(int f = 0; f < numBasisFuncs; ++f) {
        interpolant += x[f]*basis[q*numBasisFuncs+f];
      }
      elemError += (interpolant - funcVal)*(interpolant - funcVal)*quadWeights[q];
    }    
    if (options->debug) {
      std::cout << "Element " << *c_iter << " error: " << elemError << std::endl;
    }
    localError += elemError;
  }
  ierr = MPI_Allreduce(&localError, error, 1, MPI_DOUBLE, MPI_SUM, m->comm());CHKERRQ(ierr);
  ierr = PetscFree4(coords,v0,J,invJ);CHKERRQ(ierr);
  *error = sqrt(*error);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Jac_Structured_2d_FD"
PetscErrorCode Jac_Structured_2d_FD(DALocalInfo *info, PetscScalar *x[], Mat J, void *ctx)
{
  Options       *options = (Options *) ctx;
  const double   lambda  = options->lambda;
  DA             coordDA;
  Vec            coordinates;
  DACoor2d     **coords;
  MatStencil     row, col[5];
  PetscScalar    v[5];
  PetscReal      hxa, hxb, hx, hya, hyb, hy;
  PetscInt       ie = info->xs+info->xm;
  PetscInt       je = info->ys+info->ym;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DAGetCoordinateDA(info->da, &coordDA);CHKERRQ(ierr);
  ierr = DAGetGhostedCoordinates(info->da, &coordinates);CHKERRQ(ierr);
  ierr = DAVecGetArray(coordDA, coordinates, &coords);CHKERRQ(ierr);
  // Loop over stencils
  for(int j = info->ys; j < je; j++) {
    for(int i = info->xs; i < ie; i++) {
      row.j = j; row.i = i;
      if (i == 0 || j == 0 || i == info->mx-1 || j == info->my-1) {
        v[0] = 1.0;
        ierr = MatSetValuesStencil(J, 1, &row, 1, &row, v, INSERT_VALUES);CHKERRQ(ierr);
      } else {
        hya = coords[j+1][i].y - coords[j][i].y;
        hyb = coords[j][i].y   - coords[j-1][i].y;
        hxa = coords[j][i+1].x - coords[j][i].x;
        hxb = coords[j][i].x   - coords[j][i-1].x;
        hy  = 0.5*(hya+hyb);
        hx  = 0.5*(hxa+hxb);
        v[0] = -hx/hyb;                                          col[0].j = j - 1; col[0].i = i;
        v[1] = -hy/hxb;                                          col[1].j = j;     col[1].i = i-1;
        v[2] = (hy/hxa + hy/hxb + hx/hya + hx/hyb);              col[2].j = row.j; col[2].i = row.i;
        v[3] = -hy/hxa;                                          col[3].j = j;     col[3].i = i+1;
        v[4] = -hx/hya;                                          col[4].j = j + 1; col[4].i = i;
        v[2] -= lambda*hx*hy*PetscExpScalar(x[j][i]);
        ierr = MatSetValuesStencil(J, 1, &row, 5, col, v, INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = DAVecRestoreArray(coordDA, coordinates, &coords);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0); 
}

#undef __FUNCT__
#define __FUNCT__ "Jac_Structured_3d_FD"
PetscErrorCode Jac_Structured_3d_FD(DALocalInfo *info, PetscScalar **x[], Mat J, void *ctx)
{
  Options       *options = (Options *) ctx;
  const double   lambda  = options->lambda;
  DA             coordDA;
  Vec            coordinates;
  DACoor3d    ***coords;
  MatStencil     row, col[7];
  PetscScalar    v[7];
  PetscReal      hxa, hxb, hx, hya, hyb, hy, hza, hzb, hz;
  PetscInt       ie = info->xs+info->xm;
  PetscInt       je = info->ys+info->ym;
  PetscInt       ke = info->zs+info->zm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DAGetCoordinateDA(info->da, &coordDA);CHKERRQ(ierr);
  ierr = DAGetGhostedCoordinates(info->da, &coordinates);CHKERRQ(ierr);
  ierr = DAVecGetArray(coordDA, coordinates, &coords);CHKERRQ(ierr);
  // Loop over stencils
  for(int k = info->zs; k < ke; k++) {
    for(int j = info->ys; j < je; j++) {
      for(int i = info->xs; i < ie; i++) {
        row.k = k; row.j = j; row.i = i;
        if (i == 0 || j == 0 || k == 0 || i == info->mx-1 || j == info->my-1 || k == info->mz-1) {
          v[0] = 1.0;
          ierr = MatSetValuesStencil(J, 1, &row, 1, &row, v, INSERT_VALUES);CHKERRQ(ierr);
        } else {
          hza = coords[k+1][j][i].z - coords[k][j][i].z;
          hzb = coords[k][j][i].z   - coords[k-1][j][i].z;
          hya = coords[k][j+1][i].y - coords[k][j][i].y;
          hyb = coords[k][j][i].y   - coords[k][j-1][i].y;
          hxa = coords[k][j][i+1].x - coords[k][j][i].x;
          hxb = coords[k][j][i].x   - coords[k][j][i-1].x;
          hz  = 0.5*(hza+hzb);
          hy  = 0.5*(hya+hyb);
          hx  = 0.5*(hxa+hxb);
          v[0] = -hx*hy/hzb;                                       col[0].k = k - 1; col[0].j = j;     col[0].i = i;
          v[1] = -hx*hz/hyb;                                       col[1].k = k;     col[1].j = j - 1; col[1].i = i;
          v[2] = -hy*hz/hxb;                                       col[2].k = k;     col[2].j = j;     col[2].i = i - 1;
          v[3] = (hy*hz/hxa + hy*hz/hxb + hx*hz/hya + hx*hz/hyb + hx*hy/hza + hx*hy/hzb); col[3].k = row.k; col[3].j = row.j; col[3].i = row.i;
          v[4] = -hx*hy/hxa;                                       col[4].k = k + 1; col[4].j = j;     col[4].i = i;
          v[5] = -hx*hz/hya;                                       col[5].k = k;     col[5].j = j + 1; col[5].i = i;
          v[6] = -hy*hz/hxa;                                       col[6].k = k;     col[6].j = j;     col[6].i = i + 1;
          v[3] -= lambda*hx*hy*hz*PetscExpScalar(x[k][j][i]);
          ierr = MatSetValuesStencil(J, 1, &row, 7, col, v, INSERT_VALUES);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = DAVecRestoreArray(coordDA, coordinates, &coords);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0); 
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "Laplacian_2D_MF"
PetscErrorCode Laplacian_2D_MF(Mat A, Vec x, Vec y)
{
  Mesh             mesh;
  Obj<ALE::Mesh>   m;
  SectionReal      X, Y;
  PetscQuadrature *q;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject) A, "mesh", (PetscObject *) &mesh);CHKERRQ(ierr);
  ierr = MatShellGetContext(A, (void **) &q);CHKERRQ(ierr);
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);

  ierr = MeshGetSectionReal(mesh, "work1", &X);CHKERRQ(ierr);
  ierr = MeshGetSectionReal(mesh, "work2", &Y);CHKERRQ(ierr);
  ierr = SectionRealToVec(X, mesh, SCATTER_REVERSE, x);CHKERRQ(ierr);

  const Obj<ALE::Mesh::real_section_type>& coordinates = m->getRealSection("coordinates");
  const Obj<ALE::Mesh::label_sequence>&    cells       = m->heightStratum(0);
  const int     numQuadPoints = q->numQuadPoints;
  const int     numBasisFuncs = q->numBasisFuncs;
  const double *quadWeights   = q->quadWeights;
  const double *basisDer      = q->basisDer;
  const int     dim           = m->getDimension();
  double       *t_der, *b_der, *v0, *J, *invJ, detJ;
  PetscScalar  *elemMat, *elemVec;

  ierr = PetscMalloc2(numBasisFuncs,PetscScalar,&elemVec,numBasisFuncs*numBasisFuncs,PetscScalar,&elemMat);CHKERRQ(ierr);
  ierr = PetscMalloc5(dim,double,&t_der,dim,double,&b_der,dim,double,&v0,dim*dim,double,&J,dim*dim,double,&invJ);CHKERRQ(ierr);
  // Loop over cells
  ierr = SectionRealZero(Y);CHKERRQ(ierr);
  for(ALE::Mesh::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
    ierr = PetscMemzero(elemMat, numBasisFuncs*numBasisFuncs * sizeof(PetscScalar));CHKERRQ(ierr);
    m->computeElementGeometry(coordinates, *c_iter, v0, J, invJ, detJ);
    // Loop over quadrature points
    for(int q = 0; q < numQuadPoints; ++q) {
      // Loop over trial functions
      for(int f = 0; f < numBasisFuncs; ++f) {
        for(int d = 0; d < dim; ++d) {
          t_der[d] = 0.0;
          for(int e = 0; e < dim; ++e) t_der[d] += invJ[e*dim+d]*basisDer[(q*numBasisFuncs+f)*dim+e];
        }
        // Loop over basis functions
        for(int g = 0; g < numBasisFuncs; ++g) {
          for(int d = 0; d < dim; ++d) {
            b_der[d] = 0.0;
            for(int e = 0; e < dim; ++e) b_der[d] += invJ[e*dim+d]*basisDer[(q*numBasisFuncs+g)*dim+e];
          }
          PetscScalar product = 0.0;
          for(int d = 0; d < dim; ++d) product += t_der[d]*b_der[d];
          elemMat[f*numBasisFuncs+g] += product*quadWeights[q]*detJ;
        }
      }
    }
    PetscScalar *ev;

    ierr = SectionRealRestrict(X, *c_iter, &ev);CHKERRQ(ierr);
    // Do local matvec
    for(int f = 0; f < numBasisFuncs; ++f) {
      elemVec[f] = 0.0;
      for(int g = 0; g < numBasisFuncs; ++g) {
        elemVec[f] += elemMat[f*numBasisFuncs+g]*ev[g];
      }
    }
    ierr = SectionRealUpdateAdd(Y, *c_iter, elemVec);CHKERRQ(ierr);
  }
  ierr = PetscFree2(elemVec,elemMat);CHKERRQ(ierr);
  ierr = PetscFree5(t_der,b_der,v0,J,invJ);CHKERRQ(ierr);
  ierr = SectionRealComplete(Y);CHKERRQ(ierr);

  ierr = SectionRealToVec(Y, mesh, SCATTER_FORWARD, y);CHKERRQ(ierr);
  ierr = SectionRealDestroy(X);CHKERRQ(ierr);
  ierr = SectionRealDestroy(Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "Jac_Unstructured_Calculated"
PetscErrorCode Jac_Unstructured_Calculated(Mesh mesh, SectionReal section, Mat A, void *ctx)
{
  Obj<ALE::Mesh>   m;
  PetscQuadrature *q;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = MatShellSetOperation(A, MATOP_MULT, (void(*)(void)) Laplacian_2D_MF);CHKERRQ(ierr);
  ierr = PetscMalloc(sizeof(PetscQuadrature), &q);CHKERRQ(ierr);
  ierr = MatShellSetContext(A, (void *) q);CHKERRQ(ierr);
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  const Obj<ALE::Discretization>&          disc  = m->getDiscretization("u");
  const Obj<ALE::Mesh::real_section_type>& def   = m->getRealSection("default");
  const Obj<ALE::Mesh::real_section_type>& work1 = m->getRealSection("work1");
  const Obj<ALE::Mesh::real_section_type>& work2 = m->getRealSection("work2");
  q->numQuadPoints = disc->getQuadratureSize();
  q->quadPoints    = disc->getQuadraturePoints();
  q->quadWeights   = disc->getQuadratureWeights();
  q->numBasisFuncs = disc->getBasisSize();
  q->basis         = disc->getBasis();
  q->basisDer      = disc->getBasisDerivatives();
  work1->setAtlas(def->getAtlas());
  work1->allocateStorage();
  work2->setAtlas(def->getAtlas());
  work2->allocateStorage();
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "Laplacian_2D_MF2"
PetscErrorCode Laplacian_2D_MF2(Mat A, Vec x, Vec y)
{
  Mesh           mesh;
  Obj<ALE::Mesh> m;
  Obj<ALE::Mesh::real_section_type> s;
  SectionReal    op, X, Y;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject) A, "mesh", (PetscObject *) &mesh);CHKERRQ(ierr);
  ierr = MatShellGetContext(A, (void **) &op);CHKERRQ(ierr);
  ierr = SectionRealGetSection(op, s);CHKERRQ(ierr);
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);

  ierr = MeshGetSectionReal(mesh, "work1", &X);CHKERRQ(ierr);
  ierr = MeshGetSectionReal(mesh, "work2", &Y);CHKERRQ(ierr);
  ierr = SectionRealToVec(X, mesh, SCATTER_REVERSE, x);CHKERRQ(ierr);

  const Obj<ALE::Mesh::label_sequence>& cells         = m->heightStratum(0);
  int                                   numBasisFuncs = m->getDiscretization("u")->getBasisSize();
  PetscScalar                          *elemVec;

  ierr = PetscMalloc(numBasisFuncs *sizeof(PetscScalar), &elemVec);CHKERRQ(ierr);
  // Loop over cells
  ierr = SectionRealZero(Y);CHKERRQ(ierr);
  for(ALE::Mesh::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
    const ALE::Mesh::real_section_type::value_type *elemMat = s->restrictPoint(*c_iter);
    PetscScalar *ev;

    ierr = SectionRealRestrict(X,  *c_iter, &ev);CHKERRQ(ierr);
    // Do local matvec
    for(int f = 0; f < numBasisFuncs; ++f) {
      elemVec[f] = 0.0;
      for(int g = 0; g < numBasisFuncs; ++g) {
        elemVec[f] += elemMat[f*numBasisFuncs+g]*ev[g];
      }
    }
    ierr = SectionRealUpdateAdd(Y, *c_iter, elemVec);CHKERRQ(ierr);
  }
  ierr = PetscFree(elemVec);CHKERRQ(ierr);
  ierr = SectionRealComplete(Y);CHKERRQ(ierr);

  ierr = SectionRealToVec(Y, mesh, SCATTER_FORWARD, y);CHKERRQ(ierr);
  ierr = SectionRealDestroy(X);CHKERRQ(ierr);
  ierr = SectionRealDestroy(Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "Jac_Unstructured_Stored"
PetscErrorCode Jac_Unstructured_Stored(Mesh mesh, SectionReal section, Mat A, void *ctx)
{
  SectionReal    op;
  Obj<ALE::Mesh> m;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = MatShellSetOperation(A, MATOP_MULT, (void(*)(void)) Laplacian_2D_MF2);CHKERRQ(ierr);
  const Obj<ALE::Discretization>&          disc          = m->getDiscretization("u");
  const int                                numQuadPoints = disc->getQuadratureSize();
  const double                            *quadWeights   = disc->getQuadratureWeights();
  const int                                numBasisFuncs = disc->getBasisSize();
  const double                            *basisDer      = disc->getBasisDerivatives();
  const Obj<ALE::Mesh::real_section_type>& coordinates   = m->getRealSection("coordinates");
  const Obj<ALE::Mesh::label_sequence>&    cells         = m->heightStratum(0);
  const int dim = m->getDimension();
  double      *t_der, *b_der, *v0, *J, *invJ, detJ;
  PetscScalar *elemMat;

  ierr = MeshGetCellSectionReal(mesh, numBasisFuncs*numBasisFuncs, &op);CHKERRQ(ierr);
  ierr = MatShellSetContext(A, (void *) op);CHKERRQ(ierr);
  ierr = PetscMalloc(numBasisFuncs*numBasisFuncs * sizeof(PetscScalar), &elemMat);CHKERRQ(ierr);
  ierr = PetscMalloc5(dim,double,&t_der,dim,double,&b_der,dim,double,&v0,dim*dim,double,&J,dim*dim,double,&invJ);CHKERRQ(ierr);
  // Loop over cells
  for(ALE::Mesh::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
    ierr = PetscMemzero(elemMat, numBasisFuncs*numBasisFuncs * sizeof(PetscScalar));CHKERRQ(ierr);
    m->computeElementGeometry(coordinates, *c_iter, v0, J, invJ, detJ);
    // Loop over quadrature points
    for(int q = 0; q < numQuadPoints; ++q) {
      // Loop over trial functions
      for(int f = 0; f < numBasisFuncs; ++f) {
        for(int d = 0; d < dim; ++d) {
          t_der[d] = 0.0;
          for(int e = 0; e < dim; ++e) t_der[d] += invJ[e*dim+d]*basisDer[(q*numBasisFuncs+f)*dim+e];
        }
        // Loop over basis functions
        for(int g = 0; g < numBasisFuncs; ++g) {
          for(int d = 0; d < dim; ++d) {
            b_der[d] = 0.0;
            for(int e = 0; e < dim; ++e) b_der[d] += invJ[e*dim+d]*basisDer[(q*numBasisFuncs+g)*dim+e];
          }
          PetscScalar product = 0.0;
          for(int d = 0; d < dim; ++d) product += t_der[d]*b_der[d];
          elemMat[f*numBasisFuncs+g] += product*quadWeights[q]*detJ;
        }
      }
    }
    ierr = SectionRealUpdate(op, *c_iter, elemMat);CHKERRQ(ierr);
  }
  ierr = PetscFree(elemMat);CHKERRQ(ierr);
  ierr = PetscFree5(t_der,b_der,v0,J,invJ);CHKERRQ(ierr);

  const Obj<ALE::Mesh::real_section_type>& def   = m->getRealSection("default");
  const Obj<ALE::Mesh::real_section_type>& work1 = m->getRealSection("work1");
  const Obj<ALE::Mesh::real_section_type>& work2 = m->getRealSection("work2");
  work1->setAtlas(def->getAtlas());
  work1->allocateStorage();
  work2->setAtlas(def->getAtlas());
  work2->allocateStorage();
  PetscFunctionReturn(0);
}

PetscErrorCode cellJacobian(const Obj<ALE::Discretization>& disc, PetscScalar x[], PetscScalar t_der[], PetscScalar b_der[], PetscScalar elemMat[], double epsilon, Options *options) {
  const int      dim    = options->dim;
  double        *invJ   = options->invJ;
  double         detJ   = options->detJ;
  const int      numQuadPoints = disc->getQuadratureSize();
  const double  *quadWeights   = disc->getQuadratureWeights();
  const int      numBasisFuncs = disc->getBasisSize();
  const double  *basis         = disc->getBasis();
  const double  *basisDer      = disc->getBasisDerivatives();
  const double   lambda        = options->lambda;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMemzero(elemMat, numBasisFuncs*numBasisFuncs * sizeof(PetscScalar));CHKERRQ(ierr);
  // Loop over quadrature points
  for(int q = 0; q < numQuadPoints; ++q) {
    PetscScalar fieldVal = 0.0;

    for(int f = 0; f < numBasisFuncs; ++f) {
      fieldVal += x[f]*basis[q*numBasisFuncs+f];
    }
    // Loop over trial functions
    for(int f = 0; f < numBasisFuncs; ++f) {
      for(int d = 0; d < dim; ++d) {
        t_der[d] = 0.0;
        for(int e = 0; e < dim; ++e) t_der[d] += invJ[e*dim+d]*basisDer[(q*numBasisFuncs+f)*dim+e];
      }
      // Loop over basis functions
      for(int g = 0; g < numBasisFuncs; ++g) {
        for(int d = 0; d < dim; ++d) {
          b_der[d] = 0.0;
          for(int e = 0; e < dim; ++e) b_der[d] += invJ[e*dim+d]*basisDer[(q*numBasisFuncs+g)*dim+e];
        }
        PetscScalar product = 0.0;
        for(int d = 0; d < dim; ++d) product += t_der[d]*b_der[d];
        elemMat[f*numBasisFuncs+g] += epsilon*product*quadWeights[q]*detJ;
        // Nonlinear part
        elemMat[f*numBasisFuncs+g] -= basis[q*numBasisFuncs+f]*basis[q*numBasisFuncs+g]*lambda*PetscExpScalar(fieldVal)*quadWeights[q]*detJ;
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode faceJacobian(const Obj<ALE::Discretization>& cellDisc, const Obj<ALE::Discretization>& disc, const PetscScalar cellBasisDer[], PetscScalar b_der[], PetscScalar elemMat[], double epsilon, Options *options) {
  const int      dim     = options->dim;
  double        *invJ    = options->invJ;
  double         detJ    = options->fDetJ;
  double        *normal  = options->normal;
  double        *tangent = options->tangent;
  const int      numQuadPoints = disc->getQuadratureSize();
  const double  *quadWeights   = disc->getQuadratureWeights();
  const int      numCellBasisFuncs = cellDisc->getBasisSize();
  const int      numBasisFuncs = disc->getBasisSize();
  const double  *basis         = disc->getBasis();
  //const double  *basisDer      = disc->getBasisDerivatives();
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMemzero(elemMat, numBasisFuncs*numCellBasisFuncs * sizeof(PetscScalar));CHKERRQ(ierr);
  // Loop over quadrature points
  for(int q = 0; q < numQuadPoints; ++q) {
    // Loop over trial functions
    for(int f = 0; f < numBasisFuncs; ++f) {
      // Loop over basis functions
      for(int g = 0; g < numCellBasisFuncs; ++g) {
        // Linear part
        for(int d = 0; d < dim; ++d) {
          b_der[d] = 0.0;
          for(int e = 0; e < dim; ++e) b_der[d] += invJ[e*dim+d]*cellBasisDer[(q*numCellBasisFuncs+g)*dim+e];
        }
        PetscScalar productN = 0.0, productT = 0.0;
        for(int d = 0; d < dim; ++d) {
          productN += normal[d]*b_der[d];
          for(int e = 0; e < dim-1; ++e) {
            productT += tangent[e*dim+d]*b_der[d];
          }
        }
        elemMat[f*numCellBasisFuncs+g] += basis[q*numBasisFuncs+f]*(epsilon*productN + productT)*quadWeights[q]*detJ;
      }
    }
  }    
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Jac_Unstructured"
PetscErrorCode Jac_Unstructured(Mesh mesh, SectionReal section, Mat A, void *ctx)
{
  Options       *options = (Options *) ctx;
  Obj<ALE::Mesh::real_section_type> s;
  Obj<ALE::Mesh> m;
  Obj<ALE::Mesh> bdM;
  PetscScalar   *cellBasisDer;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = MeshGetMesh(options->bdMesh, bdM);CHKERRQ(ierr);
  ierr = SectionRealGetSection(section, s);CHKERRQ(ierr);
  const Obj<ALE::Mesh::sieve_type>&         sieve         = m->getSieve();
  const Obj<ALE::Discretization>&           disc          = m->getDiscretization("u");
  const Obj<ALE::Discretization>&           bdDisc        = bdM->getDiscretization("u");
  const int                                 numBasisFuncs = disc->getBasisSize();
  const Obj<ALE::Mesh::real_section_type>&  coordinates   = m->getRealSection("coordinates");
  const Obj<ALE::Mesh::order_type>&         order         = m->getFactory()->getGlobalOrder(m, "default", s);
  const int                                 dim           = m->getDimension();
  const int                                 offset        = bdDisc->getQuadratureSize()*disc->getBasisSize()*dim;
  PetscScalar *t_der, *b_der, *elemMat;

  if (dim == 2) {
    cellBasisDer = cellBasisDer2D;
  } else if (dim == 3) {
    cellBasisDer = cellBasisDer3D;
  }
  ierr = MatZeroEntries(A);CHKERRQ(ierr);
  ierr = PetscMalloc3(dim,PetscScalar,&t_der,dim,PetscScalar,&b_der,numBasisFuncs*numBasisFuncs,PetscScalar,&elemMat);CHKERRQ(ierr);
  // Must create the label on coarser meshes if it is missing
  if (!m->hasLabel("particle")) {ierr = CreateParticleLabel(mesh, options);CHKERRQ(ierr);}
  // Loop over water cells
  const Obj<ALE::Mesh::label_sequence>&     waterCells = m->getLabelStratum("particle", 1);
  const ALE::Mesh::label_sequence::iterator wBegin     = waterCells->begin();
  const ALE::Mesh::label_sequence::iterator wEnd       = waterCells->end();

  for(ALE::Mesh::label_sequence::iterator c_iter = wBegin; c_iter != wEnd; ++c_iter) {
    PetscScalar *x;

    m->computeElementGeometry(coordinates, *c_iter, options->v0, options->J, options->invJ, options->detJ);
    if (options->detJ < 0) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %d", options->detJ, *c_iter);
    ierr = SectionRealRestrict(section, *c_iter, &x);CHKERRQ(ierr);
    ierr = cellJacobian(disc, x, t_der, b_der, elemMat, options->waterEpsilon, options);CHKERRQ(ierr);
    ierr = updateOperator(A, m, s, order, *c_iter, elemMat, ADD_VALUES);CHKERRQ(ierr);
  }
  // Loop over particle cells
  const Obj<ALE::Mesh::label_sequence>&     particleCells = m->getLabelStratum("particle", 2);
  const ALE::Mesh::label_sequence::iterator pBegin        = particleCells->begin();
  const ALE::Mesh::label_sequence::iterator pEnd          = particleCells->end();

  for(ALE::Mesh::label_sequence::iterator c_iter = pBegin; c_iter != pEnd; ++c_iter) {
    PetscScalar *x;

    m->computeElementGeometry(coordinates, *c_iter, options->v0, options->J, options->invJ, options->detJ);
    if (options->detJ < 0) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %d", options->detJ, *c_iter);
    ierr = SectionRealRestrict(section, *c_iter, &x);CHKERRQ(ierr);
    ierr = cellJacobian(disc, x, t_der, b_der, elemMat, options->particleEpsilon, options);CHKERRQ(ierr);
    ierr = updateOperator(A, m, s, order, *c_iter, elemMat, ADD_VALUES);CHKERRQ(ierr);
  }
  // Loop over water boundary
  const Obj<ALE::Mesh::label_type>&         particleBd   = m->getLabel("particleBd");
  const Obj<ALE::Mesh::label_sequence>&     waterBdCells = m->getLabelStratum("particleBd", 3);
  const ALE::Mesh::label_sequence::iterator wBdBegin     = waterBdCells->begin();
  const ALE::Mesh::label_sequence::iterator wBdEnd       = waterBdCells->end();

  for(ALE::Mesh::label_sequence::iterator c_iter = wBdBegin; c_iter != wBdEnd; ++c_iter) {
    const Obj<ALE::Mesh::sieve_type::traits::coneSequence>&     cone   = sieve->cone(*c_iter);
    const ALE::Mesh::sieve_type::traits::coneSequence::iterator cBegin = cone->begin();
    const ALE::Mesh::sieve_type::traits::coneSequence::iterator cEnd   = cone->end();
    int                                                         f      = 0;

    m->computeElementGeometry(coordinates, *c_iter, options->v0, options->J, options->invJ, options->detJ);
    if (options->detJ < 0) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for face %d", options->detJ, *c_iter);
    for(ALE::Mesh::sieve_type::traits::coneSequence::iterator f_iter = cBegin; f_iter != cEnd; ++f_iter, ++f) {
      if (m->getValue(particleBd, *f_iter)) {
        m->computeFaceGeometry(*c_iter, *f_iter, f, options->invJ, options->fInvJ, options->fDetJ, options->normal, options->tangent);
        ierr = faceJacobian(disc, bdDisc, &cellBasisDer[f*offset], b_der, elemMat, options->waterEpsilon, options);CHKERRQ(ierr);
        ierr = updateOperatorGeneral(A, m, s, order, *f_iter, m, s, order, *c_iter, elemMat, ADD_VALUES);CHKERRQ(ierr);
      }
    }
  }
  // Loop over particle boundary
  const Obj<ALE::Mesh::label_sequence>&     particleBdCells = m->getLabelStratum("particleBd", 2);
  const ALE::Mesh::label_sequence::iterator pBdBegin        = particleBdCells->begin();
  const ALE::Mesh::label_sequence::iterator pBdEnd          = particleBdCells->end();

  for(ALE::Mesh::label_sequence::iterator c_iter = pBdBegin; c_iter != pBdEnd; ++c_iter) {
    const Obj<ALE::Mesh::sieve_type::traits::coneSequence>&     cone   = sieve->cone(*c_iter);
    const ALE::Mesh::sieve_type::traits::coneSequence::iterator cBegin = cone->begin();
    const ALE::Mesh::sieve_type::traits::coneSequence::iterator cEnd   = cone->end();
    int                                                         f      = 0;

    m->computeElementGeometry(coordinates, *c_iter, options->v0, options->J, options->invJ, options->detJ);
    if (options->detJ < 0) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for face %d", options->detJ, *c_iter);
    for(ALE::Mesh::sieve_type::traits::coneSequence::iterator f_iter = cBegin; f_iter != cEnd; ++f_iter, ++f) {
      if (m->getValue(particleBd, *f_iter)) {
        m->computeFaceGeometry(*c_iter, *f_iter, f, options->invJ, options->fInvJ, options->fDetJ, options->normal, options->tangent);
        for(int e = 1; e < dim-1; ++e) {
          for(int d = 0; d < dim; ++d) options->tangent[e*dim+d] *= -1.0;
        }
        ierr = faceJacobian(disc, bdDisc, &cellBasisDer[f*offset], b_der, elemMat, options->particleEpsilon, options);CHKERRQ(ierr);
        ierr = updateOperatorGeneral(A, m, s, order, *f_iter, m, s, order, *c_iter, elemMat, ADD_VALUES);CHKERRQ(ierr);
      }
    }
  }
  // Cleanup
  ierr = PetscFree3(t_der,b_der,elemMat);CHKERRQ(ierr);
  // Exchange neighbors
  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateParticleLabel"
PetscErrorCode CreateParticleLabel(Mesh mesh, Options *options)
{
  Obj<ALE::Mesh> m;
  SectionReal    coordinates;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = MeshGetSectionReal(mesh, "coordinates", &coordinates);CHKERRQ(ierr);
  const Obj<ALE::Mesh::label_type>&         particleLabel = m->createLabel("particle");
  const Obj<ALE::Mesh::label_sequence>&     cells         = m->heightStratum(0);
  const ALE::Mesh::label_sequence::iterator end           = cells->end();
  const int dim      = m->getDimension();
  const int corners  = m->getSieve()->nCone(*cells->begin(), m->depth())->size();
  double   *centroid = new double[dim];

  // WARNING: This assumes I have a round particle and can mislabel elements if they are
  //          too refined compared to the particle surface. I should either put in a check
  //          or do the correct check with triangles.
  for(ALE::Mesh::label_sequence::iterator c_iter = cells->begin(); c_iter != end; ++c_iter) {
    double *coords;
    double  sqRadius = 0.0;

    ierr = SectionRealRestrict(coordinates, *c_iter, &coords);CHKERRQ(ierr);
    for(int d = 0; d < dim; ++d) {
      centroid[d] = 0.0;
      for(int c = 0; c < corners; ++c) {
        centroid[d] += coords[c*dim+d];
      }
      centroid[d] /= corners;
      sqRadius    += (centroid[d] - 0.5)*(centroid[d] - 0.5);
    }
    if (sqRadius <= options->particleRadius*options->particleRadius + 1.0e-9) {
      m->setValue(particleLabel, *c_iter, 2);
    } else {
      m->setValue(particleLabel, *c_iter, 1);
    }
  }
  delete [] centroid;
  ierr = SectionRealDestroy(coordinates);CHKERRQ(ierr);
  // Label:
  //   particles faces:   1
  //   particle bd cells: 2
  //   water bd cells:    3
  const Obj<ALE::Mesh::sieve_type>&         sieve           = m->getSieve();
  const Obj<ALE::Mesh::label_type>&         particleBdLabel = m->createLabel("particleBd");
  const Obj<ALE::Mesh::label_sequence>&     faces           = m->heightStratum(1);
  const ALE::Mesh::label_sequence::iterator fEnd            = faces->end();

  for(ALE::Mesh::label_sequence::iterator f_iter = faces->begin(); f_iter != fEnd; ++f_iter) {
    const Obj<ALE::Mesh::sieve_type::traits::supportSequence>& support = sieve->support(*f_iter);

    if (support->size() == 2) {
      const ALE::Mesh::point_type cellA = *support->begin();
      const ALE::Mesh::point_type cellB = *(++support->begin());
      const int                   valA  = m->getValue(particleLabel, cellA);
      const int                   valB  = m->getValue(particleLabel, cellB);

      if (valA != valB) {
        m->setValue(particleBdLabel, *f_iter, 1);
        if (valA == 2) {
          m->setValue(particleBdLabel, cellA, 2);
          m->setValue(particleBdLabel, cellB, 3);
        } else {
          m->setValue(particleBdLabel, cellA, 3);
          m->setValue(particleBdLabel, cellB, 2);
        }
      }
    }
  }
  if (options->debug) {particleBdLabel->view("particleBdLabel");}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateProblem"
PetscErrorCode CreateProblem(DM dm, Options *options)
{
  PetscFunctionBegin;
  if (options->dim == 2) {
    if (options->bcType == DIRICHLET) {
      if (options->lambda > 0.0) {
        options->func    = nonlinear_2d;
      } else {
        options->func    = zero;
      }
      options->exactFunc = constantField_2d;
    } else {
      options->func      = linear_2d;
      options->exactFunc = cubic_2d;
    }
  } else if (options->dim == 3) {
    if (options->bcType == DIRICHLET) {
      if (options->lambda > 0.0) {
        options->func    = nonlinear_3d;
      } else {
        options->func    = zero;
      }
      options->exactFunc = constantField_3d;
    } else {
      options->func      = linear_3d;
      options->exactFunc = cubic_3d;
    }
  } else {
    SETERRQ1(PETSC_ERR_SUP, "Dimension not supported: %d", options->dim);
  }
  if (options->structured) {
    // The DA defines most of the problem during creation
  } else {
    Mesh           mesh = (Mesh) dm;
    Obj<ALE::Mesh> m;
    int            numBC = (options->bcType == DIRICHLET) ? 1 : 0;
    int            markers[1]  = {1};
    double       (*funcs[1])(const double *coords) = {options->exactFunc};
    PetscErrorCode ierr;

    ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
    if (options->dim == 1) {
      ierr = CreateProblem_gen_0(dm, "u", numBC, markers, funcs, options->exactFunc);CHKERRQ(ierr);
    } else if (options->dim == 2) {
      ierr = CreateProblem_gen_1(dm, "u", numBC, markers, funcs, options->exactFunc);CHKERRQ(ierr);
      ierr = CreateProblem_gen_0((DM) options->bdMesh, "u", 0, PETSC_NULL, PETSC_NULL, PETSC_NULL);CHKERRQ(ierr);
    } else if (options->dim == 3) {
      ierr = CreateProblem_gen_2(dm, "u", numBC, markers, funcs, options->exactFunc);CHKERRQ(ierr);
      ierr = CreateProblem_gen_1((DM) options->bdMesh, "u", 0, PETSC_NULL, PETSC_NULL, PETSC_NULL);CHKERRQ(ierr);
    } else {
      SETERRQ1(PETSC_ERR_SUP, "Dimension not supported: %d", options->dim);
    }
    ierr = CreateParticleLabel(mesh, options);CHKERRQ(ierr);
    radius  = options->particleRadius;
    epsilon = options->particleEpsilon;
#if 0
    {
      const Obj<ALE::Mesh::int_section_type>&   groupField = new ALE::Mesh::int_section_type(m->comm(), m->debug());
      const Obj<ALE::Mesh::label_sequence>&     faces      = m->getLabelStratum("particleBd", 1);
      const ALE::Mesh::label_sequence::iterator fEnd       = faces->end();
      Obj<ALE::Mesh>                            bdMesh     = PETSC_NULL;

      for(ALE::Mesh::label_sequence::iterator f_iter = faces->begin(); f_iter != fEnd; ++f_iter) {
        const Obj<ALE::Mesh::sieve_type::traits::coneSequence>& cone = m->getSieve()->cone(*f_iter);

        for(ALE::Mesh::sieve_type::traits::coneSequence::iterator v_iter = cone->begin(); v_iter != cone->end(); ++v_iter) {
          groupField->setFiberDimension(*v_iter, 1);
        }
      }
      groupField->allocatePoint();
      if (options->debug) {groupField->view("Boundary vertices");}
      ALE::MySelection::create(m, bdMesh, groupField);
      m->view("Cohesive Mesh");
    }
#endif
    const Obj<ALE::Mesh::real_section_type> s = m->getRealSection("default");
    s->setDebug(options->debug);
    m->setupField(s);
    if (options->debug) {s->view("Default field");}
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateExactSolution"
PetscErrorCode CreateExactSolution(DM dm, Options *options)
{
  const int      dim = options->dim;
  PetscTruth     flag;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (options->structured) {
    DA  da = (DA) dm;
    PetscScalar (*func)(const double *) = options->func;
    Vec X, U;

    ierr = DAGetGlobalVector(da, &X);CHKERRQ(ierr);
    ierr = DACreateGlobalVector(da, &options->exactSol.vec);CHKERRQ(ierr);
    options->func = options->exactFunc;
    U             = options->exactSol.vec;
    if (dim == 2) {
      ierr = DAFormFunctionLocal(da, (DALocalFunction1) Function_Structured_2d, X, U, (void *) options);CHKERRQ(ierr);
    } else if (dim == 3) {
      ierr = DAFormFunctionLocal(da, (DALocalFunction1) Function_Structured_3d, X, U, (void *) options);CHKERRQ(ierr);
    } else {
      SETERRQ1(PETSC_ERR_SUP, "Dimension not supported: %d", dim);
    }
    ierr = DARestoreGlobalVector(da, &X);CHKERRQ(ierr);
    ierr = PetscOptionsHasName(PETSC_NULL, "-vec_view", &flag);CHKERRQ(ierr);
    if (flag) {ierr = VecView(U, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);}
    ierr = PetscOptionsHasName(PETSC_NULL, "-vec_view_draw", &flag);CHKERRQ(ierr);
    if (flag) {ierr = VecView(U, PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}
    options->func = func;
  } else {
    Mesh mesh = (Mesh) dm;

    Obj<ALE::Mesh> m;
    Obj<ALE::Mesh::real_section_type> s;

    ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
    ierr = MeshGetSectionReal(mesh, "exactSolution", &options->exactSol.section);CHKERRQ(ierr);
    ierr = SectionRealGetSection(options->exactSol.section, s);CHKERRQ(ierr);
    m->setupField(s);
    const Obj<ALE::Discretization>&           disc        = m->getDiscretization("u");
    const Obj<ALE::BoundaryCondition>&        bc          = disc->getExactSolution();
    const Obj<ALE::Mesh::label_sequence>&     cells       = m->heightStratum(0);
    const Obj<ALE::Mesh::real_section_type>&  coordinates = m->getRealSection("coordinates");
    const int                                 localDof    = m->sizeWithBC(s, *cells->begin());
    ALE::Mesh::real_section_type::value_type *values      = new ALE::Mesh::real_section_type::value_type[localDof];
    double                                   *v0          = new double[dim];
    double                                   *J           = new double[dim*dim];
    double                                    detJ;

    for(ALE::Mesh::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
      const Obj<ALE::Mesh::coneArray>      closure = ALE::SieveAlg<ALE::Mesh>::closure(m, *c_iter);
      const ALE::Mesh::coneArray::iterator end     = closure->end();
      int                                  v       = 0;

      m->computeElementGeometry(coordinates, *c_iter, v0, J, PETSC_NULL, detJ);
      for(ALE::Mesh::coneArray::iterator cl_iter = closure->begin(); cl_iter != end; ++cl_iter) {
        const int pointDim = s->getFiberDimension(*cl_iter);

        if (pointDim) {
          for(int d = 0; d < pointDim; ++d, ++v) {
            values[v] = (*bc->getDualIntegrator())(v0, J, v, bc->getFunction());
          }
        }
      }
      m->updateAll(s, *c_iter, values);
    }
    delete [] values;
    delete [] v0;
    delete [] J;
    ierr = PetscOptionsHasName(PETSC_NULL, "-vec_view", &flag);CHKERRQ(ierr);
    if (flag) {s->view("Exact Solution");}
    ierr = PetscOptionsHasName(PETSC_NULL, "-vec_view_vtk", &flag);CHKERRQ(ierr);
    if (flag) {ierr = ViewSection(mesh, options->exactSol.section, "exact_sol.vtk");CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CheckError"
PetscErrorCode CheckError(DM dm, ExactSolType sol, Options *options)
{
  MPI_Comm       comm;
  const char    *name;
  PetscScalar    norm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  if (options->structured) {
    DA  da = (DA) dm;
    Vec error;

    ierr = DAGetGlobalVector(da, &error);CHKERRQ(ierr);
    ierr = VecCopy(sol.vec, error);CHKERRQ(ierr);
    ierr = VecAXPY(error, -1.0, options->exactSol.vec);CHKERRQ(ierr);
    ierr = VecNorm(error, NORM_2, &norm);CHKERRQ(ierr);
    ierr = DARestoreGlobalVector(da, &error);CHKERRQ(ierr);
    ierr = PetscObjectGetName((PetscObject) sol.vec, &name);CHKERRQ(ierr);
  } else {
    Mesh mesh = (Mesh) dm;

    ierr = CalculateError(mesh, sol.section, &norm, options);CHKERRQ(ierr);
    ierr = PetscObjectGetName((PetscObject) sol.section, &name);CHKERRQ(ierr);
  }
  PetscPrintf(comm, "Error for trial solution %s: %g\n", name, norm);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CheckResidual"
PetscErrorCode CheckResidual(DM dm, ExactSolType sol, Options *options)
{
  MPI_Comm       comm;
  const char    *name;
  PetscScalar    norm;
  PetscTruth     flag;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHasName(PETSC_NULL, "-vec_view", &flag);CHKERRQ(ierr);
  if (options->structured) {
    DA  da = (DA) dm;
    Vec residual;

    ierr = DAGetGlobalVector(da, &residual);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) residual, "residual");CHKERRQ(ierr);
    if (options->dim == 2) {
      ierr = DAFormFunctionLocal(da, (DALocalFunction1) Rhs_Structured_2d_FD, sol.vec, residual, (void *) options);CHKERRQ(ierr);
    } else if (options->dim == 3) {
      ierr = DAFormFunctionLocal(da, (DALocalFunction1) Rhs_Structured_3d_FD, sol.vec, residual, (void *) options);CHKERRQ(ierr);
    } else {
      SETERRQ1(PETSC_ERR_SUP, "Dimension not supported: %d", options->dim);
    }
    ierr = VecNorm(residual, NORM_2, &norm);CHKERRQ(ierr);
    if (flag) {ierr = VecView(residual, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);}
    ierr = DARestoreGlobalVector(da, &residual);CHKERRQ(ierr);
    ierr = PetscObjectGetName((PetscObject) sol.vec, &name);CHKERRQ(ierr);
  } else {
    Mesh        mesh = (Mesh) dm;
    SectionReal residual;

    ierr = SectionRealDuplicate(sol.section, &residual);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) residual, "residual");CHKERRQ(ierr);
    ierr = Rhs_Unstructured(mesh, sol.section, residual, options);CHKERRQ(ierr);
    if (flag) {ierr = SectionRealView(residual, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);}
    ierr = SectionRealNorm(residual, mesh, NORM_2, &norm);CHKERRQ(ierr);
    ierr = SectionRealDestroy(residual);CHKERRQ(ierr);
    ierr = PetscObjectGetName((PetscObject) sol.section, &name);CHKERRQ(ierr);
  }
  PetscPrintf(comm, "Residual for trial solution %s: %g\n", name, norm);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CheckJacobian"
PetscErrorCode CheckJacobian(DM dm, ExactSolType sol, Options *options)
{
  MPI_Comm       comm;
  const char    *name;
  PetscScalar    norm;
  PetscTruth     flag;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHasName(PETSC_NULL, "-mat_view", &flag);CHKERRQ(ierr);
  Mesh        mesh = (Mesh) dm;
  Mat         J;
  SectionReal Y, L, M;
  Vec         x, y;

  ierr = SectionRealDuplicate(sol.section, &Y);CHKERRQ(ierr);
  ierr = MeshCreateMatrix(mesh, sol.section, MATAIJ, &J);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) J, "Jacobian");CHKERRQ(ierr);
  ierr = Jac_Unstructured(mesh, sol.section, J, options);CHKERRQ(ierr);
  if (flag) {ierr = MatView(J, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);}
  ierr = MeshCreateGlobalVector(mesh, &x);CHKERRQ(ierr);
  ierr = VecDuplicate(x, &y);CHKERRQ(ierr);
  ierr = SectionRealToVec(sol.section, mesh, SCATTER_FORWARD, x);CHKERRQ(ierr);
  ierr = MatMult(J, x, y);CHKERRQ(ierr);
  ierr = SectionRealToVec(Y, mesh, SCATTER_REVERSE, y);CHKERRQ(ierr);
  ierr = VecDestroy(x);CHKERRQ(ierr);
  ierr = VecDestroy(y);CHKERRQ(ierr);
  ierr = SectionRealDuplicate(sol.section, &L);CHKERRQ(ierr);
  ierr = SectionRealDuplicate(sol.section, &M);CHKERRQ(ierr);
  ierr = Rhs_Unstructured(mesh, M, L, options);CHKERRQ(ierr);
  ierr = SectionRealAXPY(Y, mesh, 1.0, L);CHKERRQ(ierr);
  ierr = SectionRealNorm(Y, mesh, NORM_2, &norm);CHKERRQ(ierr);
  ierr = SectionRealDestroy(Y);CHKERRQ(ierr);
  ierr = SectionRealDestroy(L);CHKERRQ(ierr);
  ierr = SectionRealDestroy(M);CHKERRQ(ierr);
  ierr = MatDestroy(J);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject) sol.section, &name);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject) sol.section, &comm);CHKERRQ(ierr);
  PetscPrintf(comm, "Error for linear residual for trial solution %s: %g\n", name, norm);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateSolver"
PetscErrorCode CreateSolver(DM dm, DMMG **dmmg, Options *options)
{
  MPI_Comm       comm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = DMMGCreate(comm, 1, options, dmmg);CHKERRQ(ierr);
  ierr = DMMGSetDM(*dmmg, dm);CHKERRQ(ierr);
  if (options->structured) {
    // Needed if using finite elements
    // ierr = PetscOptionsSetValue("-dmmg_form_function_ghost", PETSC_NULL);CHKERRQ(ierr);
    if (options->dim == 2) {
      ierr = DMMGSetSNESLocal(*dmmg, Rhs_Structured_2d_FD, Jac_Structured_2d_FD, 0, 0);CHKERRQ(ierr);
    } else if (options->dim == 3) {
      ierr = DMMGSetSNESLocal(*dmmg, Rhs_Structured_3d_FD, Jac_Structured_3d_FD, 0, 0);CHKERRQ(ierr);
    } else {
      SETERRQ1(PETSC_ERR_SUP, "Dimension not supported: %d", options->dim);
    }
    ierr = DMMGSetFromOptions(*dmmg);CHKERRQ(ierr);
    for(int l = 0; l < DMMGGetLevels(*dmmg); l++) {
      ierr = DASetUniformCoordinates((DA) (*dmmg)[l]->dm, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0);CHKERRQ(ierr);
    }
  } else {
    if (options->operatorAssembly == ASSEMBLY_FULL) {
      ierr = DMMGSetSNESLocal(*dmmg, Rhs_Unstructured, Jac_Unstructured, 0, 0);CHKERRQ(ierr);
    } else if (options->operatorAssembly == ASSEMBLY_CALCULATED) {
      ierr = DMMGSetMatType(*dmmg, MATSHELL);CHKERRQ(ierr);
      ierr = DMMGSetSNESLocal(*dmmg, Rhs_Unstructured, Jac_Unstructured_Calculated, 0, 0);CHKERRQ(ierr);
    } else if (options->operatorAssembly == ASSEMBLY_STORED) {
      ierr = DMMGSetMatType(*dmmg, MATSHELL);CHKERRQ(ierr);
      ierr = DMMGSetSNESLocal(*dmmg, Rhs_Unstructured, Jac_Unstructured_Stored, 0, 0);CHKERRQ(ierr);
    } else {
      SETERRQ1(PETSC_ERR_ARG_WRONG, "Assembly type not supported: %d", options->operatorAssembly);
    }
    ierr = DMMGSetFromOptions(*dmmg);CHKERRQ(ierr);
  }
  if (options->bcType == NEUMANN) {
    // With Neumann conditions, we tell DMMG that constants are in the null space of the operator
    ierr = DMMGSetNullSpace(*dmmg, PETSC_TRUE, 0, PETSC_NULL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Solve"
PetscErrorCode Solve(DMMG *dmmg, Options *options)
{
  SNES                snes;
  MPI_Comm            comm;
  PetscInt            its;
  PetscTruth          flag;
  SNESConvergedReason reason;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = DMMGSolve(dmmg);CHKERRQ(ierr);
  snes = DMMGGetSNES(dmmg);
  ierr = SNESGetIterationNumber(snes, &its);CHKERRQ(ierr);
  ierr = SNESGetConvergedReason(snes, &reason);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject) snes, &comm);CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "Number of Newton iterations = %D\n", its);CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "Reason for solver termination: %s\n", SNESConvergedReasons[reason]);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL, "-vec_view", &flag);CHKERRQ(ierr);
  if (flag) {ierr = VecView(DMMGGetx(dmmg), PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);}
  ierr = PetscOptionsHasName(PETSC_NULL, "-vec_view_draw", &flag);CHKERRQ(ierr);
  if (flag && options->dim == 2) {ierr = VecView(DMMGGetx(dmmg), PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}
  if (options->structured) {
    ExactSolType sol;

    sol.vec = DMMGGetx(dmmg);
    ierr = CheckError(DMMGGetDM(dmmg), sol, options);CHKERRQ(ierr);
  } else {
    Mesh        mesh = (Mesh) DMMGGetDM(dmmg);
    SectionReal solution;
    Obj<ALE::Mesh::real_section_type> sol;
    double      error;

    ierr = MeshGetSectionReal(mesh, "default", &solution);CHKERRQ(ierr);
    ierr = SectionRealGetSection(solution, sol);CHKERRQ(ierr);
    ierr = SectionRealToVec(solution, mesh, SCATTER_REVERSE, DMMGGetx(dmmg));CHKERRQ(ierr);
    ierr = CalculateError(mesh, solution, &error, options);CHKERRQ(ierr);
    ierr = PetscPrintf(sol->comm(), "Total error: %g\n", error);CHKERRQ(ierr);
    ierr = PetscOptionsHasName(PETSC_NULL, "-vec_view_vtk", &flag);CHKERRQ(ierr);
    if (flag) {ierr = ViewSection(mesh, solution, "sol.vtk");CHKERRQ(ierr);}
    ierr = PetscOptionsHasName(PETSC_NULL, "-vec_view", &flag);CHKERRQ(ierr);
    if (flag) {sol->view("Solution");}
    ierr = PetscOptionsHasName(PETSC_NULL, "-hierarchy_vtk", &flag);CHKERRQ(ierr);
    if (flag) {
      PetscViewer    viewer;
      ierr = PetscViewerCreate(sol->comm(), &viewer);CHKERRQ(ierr);
      ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
      ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
      ierr = PetscViewerFileSetName(viewer, "mesh_hierarchy.vtk");CHKERRQ(ierr);
      double offset[3] = {1.5, 0.0, 0.0};
      ierr = VTKViewer::writeHeader(viewer);CHKERRQ(ierr);
      ierr = VTKViewer::writeHierarchyVertices(dmmg, viewer, offset);CHKERRQ(ierr);
      ierr = VTKViewer::writeHierarchyElements(dmmg, viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
    }
    ierr = SectionRealDestroy(solution);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  MPI_Comm       comm;
  Options        options;
  DM             dm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr);
  comm = PETSC_COMM_WORLD;
  ierr = ProcessOptions(comm, &options);CHKERRQ(ierr);
  try {
    ierr = CreateMesh(comm, &dm, &options);CHKERRQ(ierr);
    ierr = CreateProblem(dm, &options);CHKERRQ(ierr);
    if (options.run == RUN_FULL) {
      DMMG *dmmg;

      ierr = CreateExactSolution(dm, &options);CHKERRQ(ierr);
      ierr = CheckError(dm, options.exactSol, &options);CHKERRQ(ierr);
      ierr = CheckResidual(dm, options.exactSol, &options);CHKERRQ(ierr);
      ierr = CheckJacobian(dm, options.exactSol, &options);CHKERRQ(ierr);
      ierr = CreateSolver(dm, &dmmg, &options);CHKERRQ(ierr);
      ierr = Solve(dmmg, &options);CHKERRQ(ierr);
      ierr = DMMGDestroy(dmmg);CHKERRQ(ierr);
      ierr = DestroyExactSolution(options.exactSol, &options);CHKERRQ(ierr);
    }
    ierr = DestroyMesh(dm, &options);CHKERRQ(ierr);
  } catch(ALE::Exception e) {
    std::cerr << e << std::endl;
  }
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
