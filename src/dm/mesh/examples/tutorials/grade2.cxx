/* ---------------------------------------------------------------------------------------------------------------------
 * File: grade2.cxx
 *
 * Author: Andy R Terrel
 * Email: aterrel@uchicago.edu
 * Written: October 2007
 */

static char help[] = "This example uses a Grade 2 Fluid model on a journal bearing.\n\n";

/*
 *  The model grade 2 fluid model:
 *                         -\mu\Delta u + z\times u + \nabla p = f
 *                                              \nabla \cdot u = 0
 *     \mu z + \alpha u\cdot\nabla z - \alpha z \cdot \nabla u = \mu \nabla \times u
 *
 *  The journal bearing consists of a journal which we use the unit circle, and a bearing inside
 *  which we give a radius (r) and center(X) with u = 0 on outer boundary and u \cdot t = 1 on bearing
 *
 *
 *                                         (0,1)
 *                  -------------
 *             ----/             \----
 *               -/               \-
 *              -/                 \-
 *             -/     -------       \-
 *			   /    -/       \-      \
 *			  /     /      r  \       \
 *			 /      |   X-----|        \
 *			/       \         /         \
 *			|       -\       /-         |
 * (-1,0)-->|         -------           |<---(1,0)
 *			|      u \cdot t = 1        |
 *			\                           /
 *			 \     	                   /
 *			  \                       /
 *			   \                     /
 *             -\                   /-
 *              -\                 /-
 *               -\               /-
 *             ----\     u = 0   /----
 *                  -------------
 *                                         (0,-1)
 *
 *
 *
 *  To solve the system there are three basic steps. 
 *
 *  STEP 1:
 *  --------------------------------------------------------------------
 *    Solve the Stokes like equations, with z either 0 or set by previous iteration:
 *  
 *                   -\mu\Delta u + z\times u + \nabla p = f
 *                                        \nabla \cdot u = 0
 *
 *  STEP 2:
 *  --------------------------------------------------------------------
 *    Solve the transport equation:
 *
 *  \mu z + \alpha u\cdot\nabla z - \alpha z \cdot \nabla u = \mu \nabla \times u
 *   
 *    
 *  STEP 3:
 *  --------------------------------------------------------------------
 *    Check the stopping criteria:
 *
 *     z\cdot\nabla\cdot u < tolerance
 *
 */

// ---------------------------------------------------------------------------------------------------------------------
//  Includes and Namespace
#include <petscda.h>
#include <petscmesh.h>
#include <petscdmmg.h>


using ALE::Obj;


// ---------------------------------------------------------------------------------------------------------------------
// Top level data definitions
typedef struct {
  PetscInt      debug;                                        // The debugging level
  PetscTruth    generateMesh;                                 // Generate the unstructure mesh
  PetscTruth    interpolate;                                  // Generate intermediate mesh elements
  PetscReal     refinementLimit;                              // The largest allowable cell volume
  char          baseFilename[2048];                           // The base filename for mesh files
  double        (*funcs[4])(const double []);                 // The function to project
  PetscReal     r,rho;                                        // IP parameters
  PetscReal     mu,alpha;                                     // Transport parameters
} Options;

double zero(const double x[]) {
  return 0.0;
}

double constant(const double x[]) {
  return -3.0;
}

#include "grade2_quadrature.h"

// ---------------------------------------------------------------------------------------------------------------------
// Function Prototypes

PetscErrorCode ProcessOptions(MPI_Comm comm, Options *options);
PetscErrorCode CreatePartition(Mesh mesh, SectionInt *partition);
PetscErrorCode ViewMesh(Mesh mesh, const char filename[]);
PetscErrorCode ViewSection(Mesh mesh, SectionReal section, const char filename[]);
PetscErrorCode CreateMesh(MPI_Comm comm, DM *stokesDM, DM *transportDM, Options *options);
PetscErrorCode DestroyMesh(DM stokesDM, DM transportDM, Options *options);
PetscErrorCode CreateProblem(DM stokesDM, DM transportDM, Options *options);
PetscErrorCode CreateSolver(DM dm, DMMG **dmmg, Options *options);

PetscErrorCode SolveStokes(DMMG *dmmg, Options *options);
PetscErrorCode Stokes_Rhs_Unstructured(Mesh mesh, SectionReal X, SectionReal section, void *ctx);
PetscErrorCode Stokes_Jac_Unstructured(Mesh mesh, SectionReal section, Mat A, void *ctx);
PetscErrorCode IterateStokes(DMMG *dmmg, Options *options);
PetscErrorCode CheckStokesConvergence(DMMG *dmmg, PetscTruth *iterate, Options *options);

PetscErrorCode SolveTransport(DM dm, Options *options);
PetscErrorCode Transport_Rhs_Unstructured(Mesh mesh, SectionReal X, SectionReal section, void *ctx);
PetscErrorCode Transport_Jac_Unstructured(Mesh mesh, SectionReal section, Mat A, void *ctx);
PetscErrorCode CheckStoppingCriteria(DM dm, PetscTruth *iterate, Options *options);

PetscErrorCode WriteSolution(DM dm, Options *options);

// ---------------------------------------------------------------------------------------------------------------------
// Main Procedure

/* ______________________________________________________________________ */
// Main
/*!
  \param[in] argc Size of command line array
  \param[in] **argv command line array

  Processes command line options.

  \returns
    PetscErrorCode
*/
/* ______________________________________________________________________ */
#undef __FUNCT__

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  MPI_Comm       comm;
  Options        options;
  DM             stokesDM, transportDM;
  PetscErrorCode ierr;
  PetscTruth     iterate = PETSC_TRUE;
  PetscInt       iter=0,max_iter=2;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr);
  comm = PETSC_COMM_WORLD;
  ierr = ProcessOptions(comm, &options);CHKERRQ(ierr);
  try {
    DMMG *stokes;

    ierr = CreateMesh(comm, &stokesDM, &transportDM, &options);CHKERRQ(ierr);
    ierr = CreateProblem(stokesDM, transportDM, &options);CHKERRQ(ierr);
    ierr = CreateSolver(stokesDM, &stokes, &options);CHKERRQ(ierr)

    while (iterate and max_iter >= ++iter){
      ierr = SolveStokes(stokes, &options);CHKERRQ(ierr);
      ierr = SolveTransport(transportDM, &options);CHKERRQ(ierr);
      ierr = CheckStoppingCriteria(stokesDM, &iterate, &options);CHKERRQ(ierr);
    }
    ierr = WriteSolution(stokesDM, &options);CHKERRQ(ierr);
    ierr = DMMGDestroy(stokes);CHKERRQ(ierr);
    ierr = DestroyMesh(stokesDM, transportDM, &options);CHKERRQ(ierr);
  } catch(ALE::Exception e) {
    std::cerr << e << std::endl;
  }
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// ---------------------------------------------------------------------------------------------------------------------
// Function Definitions

/* 
 *  compute:
 *       -\mu\Delta u + z\times u + \nabla p = f
 *                            \nabla \cdot u = 0
 *
 *  Using Iterated Penalty Method, becomes:
 * 
 *  while \nabla \cdot u > tol
 *       a(u^n,v)+r(\nabla\cdot u^n, \nabla\cdot v)+(\nabla\cdot v, \nabla\cdot w^n) = F(v)
 *       w^{n+1} = w^n + rho * u^n
 *
 */
#undef __FUNCT__
#define __FUNCT__ "SolveStokes"
PetscErrorCode SolveStokes(DMMG *dmmg, Options *options)
{
  Mesh           mesh     = (Mesh) DMMGGetDM(dmmg);
  PetscTruth     iterate  = PETSC_TRUE;
  PetscInt       iter     = 0;
  PetscInt       max_iter = 3;
  MPI_Comm       comm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) mesh, &comm);CHKERRQ(ierr);
  while(iterate and max_iter >= ++iter){
    SNES                snes;
    SNESConvergedReason reason;
    PetscInt            its;
    PetscTruth          flag;

    //ierr = SectionRealToVec(options->exactSol.section, mesh, SCATTER_FORWARD, DMMGGetx(dmmg));CHKERRQ(ierr);
    ierr = DMMGSolve(dmmg);CHKERRQ(ierr);
    snes = DMMGGetSNES(dmmg);
    ierr = SNESGetIterationNumber(snes, &its);CHKERRQ(ierr);
    ierr = SNESGetConvergedReason(snes, &reason);CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "Number of Newton iterations = %D\n", its);CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "Reason for solver termination: %s\n", SNESConvergedReasons[reason]);CHKERRQ(ierr);
    ierr = PetscOptionsHasName(PETSC_NULL, "-vec_view", &flag);CHKERRQ(ierr);
    if (flag) {ierr = VecView(DMMGGetx(dmmg), PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);}
    ierr = PetscOptionsHasName(PETSC_NULL, "-vec_view_draw", &flag);CHKERRQ(ierr);
    if (flag) {ierr = VecView(DMMGGetx(dmmg), PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}

    ierr = IterateStokes(dmmg, options);CHKERRQ(ierr);
    ierr = CheckStokesConvergence(dmmg, &iterate, options);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


/* 
 * compute w^{n+1} = w^{n} + rho u^{n}
 */
#undef __FUNCT__
#define __FUNCT__ "IterateStokes"
PetscErrorCode IterateStokes(DMMG *dmmg, Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Do nothing so far */
  PetscFunctionReturn(0);
}

/* 
 * check div(u) < tol
 */
#undef __FUNCT__
#define __FUNCT__ "CheckStokesConvergence"
PetscErrorCode CheckStokesConvergence(DMMG *dmmg, PetscTruth *iterate, Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Do nothing so far */
  PetscFunctionReturn(0);
}


/* 
 *  compute z from:
 *  \mu z + \alpha u\cdot\nabla z - \alpha z \cdot \nabla u = \mu \nabla \times u
 */
#undef __FUNCT__
#define __FUNCT__ "SolveTransport"
PetscErrorCode SolveTransport(DM dm, Options *options)
{
  MPI_Comm       comm;
  PetscErrorCode ierr;
  DMMG *dmmg;

  PetscFunctionBegin;

  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = DMMGCreate(comm, 1, options, &dmmg);CHKERRQ(ierr);
  ierr = DMMGSetDM(dmmg, dm);CHKERRQ(ierr);
  ierr = DMMGSetSNESLocal(dmmg, Transport_Rhs_Unstructured, Transport_Jac_Unstructured, 0, 0);CHKERRQ(ierr);
  //   if (options->bcType == NEUMANN) {
  //     // With Neumann conditions, we tell DMMG that constants are in the null space of the operator
  //     ierr = DMMGSetNullSpace(*dmmg, PETSC_TRUE, 0, PETSC_NULL);CHKERRQ(ierr);
  //   }
 
  //  Mesh                mesh = (Mesh) DMMGGetDM(dmmg);
  SNES                snes;
  PetscInt            its;
  PetscTruth          flag;
  SNESConvergedReason reason;

  //ierr = SectionRealToVec(options->exactSol.section, mesh, SCATTER_FORWARD, DMMGGetx(dmmg));CHKERRQ(ierr);
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
  if (flag) {ierr = VecView(DMMGGetx(dmmg), PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}

  PetscFunctionReturn(0);
}

/*
 *    Check the stopping criteria:
 *
 *     z\cdot\nabla\cdot u < tolerance
 */ 
#undef __FUNCT__
#define __FUNCT__ "CheckStoppingCriteria"
PetscErrorCode CheckStoppingCriteria(DM dm, PetscTruth *iterate, Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Do nothing so far */

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "WriteSolution"
PetscErrorCode WriteSolution(DM dm, Options *options)
{
  PetscErrorCode ierr;
  PetscTruth flag;
  Mesh mesh = (Mesh) dm;
  SectionReal solution;
  Obj<ALE::Mesh::real_section_type> sol;

  PetscFunctionBegin;
  
  ierr = MeshGetSectionReal(mesh, "default", &solution);CHKERRQ(ierr);
  ierr = SectionRealGetSection(solution, sol);CHKERRQ(ierr);
  //ierr = SectionRealToVec(solution, mesh, SCATTER_REVERSE, DMMGGetx(dmmg));CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL, "-vec_view_vtk", &flag);CHKERRQ(ierr);
  if (flag) {ierr = ViewSection(mesh, solution, "sol.vtk");CHKERRQ(ierr);}
  ierr = PetscOptionsHasName(PETSC_NULL, "-vec_view", &flag);CHKERRQ(ierr);
  if (flag) {sol->view("Solution");}
  ierr = PetscOptionsHasName(PETSC_NULL, "-vec_view_fibrated", &flag);CHKERRQ(ierr);
  if (flag) {
    Obj<ALE::Mesh::real_section_type> velocityX = sol->getFibration(1);
    Obj<ALE::Mesh::real_section_type> velocityY = sol->getFibration(2);

    velocityX->view("X-Velocity Solution");
    velocityY->view("Y-Velocity Solution");
  }
  ierr = SectionRealDestroy(solution);CHKERRQ(ierr);
  PetscFunctionReturn(0);
  PetscFunctionReturn(0);
}


/* ______________________________________________________________________ */
// ProcessOptions
/*!
  \param[in] comm  The MPI communicator
  \param[out] options The options table

  Processes command line options.

  \returns
    PetscErrorCode
*/
/* ______________________________________________________________________ */
#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, Options *options)
{
  ostringstream  filename;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->debug            = 0;
  options->generateMesh     = PETSC_TRUE;
  options->interpolate      = PETSC_TRUE;
  options->refinementLimit  = 0.0;
  options->r                = 0.5;
  options->rho              = -1e+3;
  options->mu               = 1;
  options->alpha            = 1;


  ierr = PetscOptionsBegin(comm, "", "Grade 2 journal bearing Options", "DMMG");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-debug", "The debugging level", "grade2.cxx", options->debug, &options->debug, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTruth("-generate", "Generate the unstructured mesh", "grade2.cxx", options->generateMesh, &options->generateMesh, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTruth("-interpolate", "Generate intermediate mesh elements", "grade2.cxx", options->interpolate, &options->interpolate, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-refinement_limit", "The largest allowable cell volume", "grade2.cxx", options->refinementLimit, &options->refinementLimit, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-r", "The IP parameter r", "grade2.cxx", options->r, &options->r, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-rho", "The IP parameter rho", "grade2.cxx", options->rho, &options->rho, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mu", "The transport parameter mu", "grade2.cxx", options->mu, &options->mu, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-alpha", "The transport parameter alpha", "grade2.cxx", options->alpha, &options->alpha, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-refinement_limit", "The largest allowable cell volume", "grade2.cxx", options->refinementLimit, &options->refinementLimit, PETSC_NULL);CHKERRQ(ierr);
  filename << "data/journal_bearing";
  ierr = PetscStrcpy(options->baseFilename, filename.str().c_str());CHKERRQ(ierr);
  ierr = PetscOptionsString("-base_filename", "The base filename for mesh files", "grade2.cxx", options->baseFilename, options->baseFilename, 2048, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  PetscFunctionReturn(0);
}

/* ______________________________________________________________________ */
// CreatePartition
/*!
  \param[in] mesh  The MPI communicator
  \param[out] partition Section whose value is the processor rank on each element.
                     

  Creates a field whose value is the processor rank on each element

  \returns
    PetscErrorCode
*/
/* ______________________________________________________________________ */
#undef __FUNCT__
#define __FUNCT__ "CreatePartition"
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

/* ______________________________________________________________________ */
// ViewMesh
/*!
  \param[in] mesh  The MPI communicator
  \param[in] filename The filename for writing the mesh to file.

  Writes mesh to file using the vtk format.

  \returns
    PetscErrorCode
*/
/* ______________________________________________________________________ */
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
#define __FUNCT__ "ViewSection"
PetscErrorCode ViewSection(Mesh mesh, SectionReal section, const char filename[])
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
  ierr = SectionRealView(section, viewer);CHKERRQ(ierr);
  ierr = CreatePartition(mesh, &partition);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_VTK_CELL);CHKERRQ(ierr);
  ierr = SectionIntView(partition, viewer);CHKERRQ(ierr);
  ierr = SectionIntDestroy(partition);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ______________________________________________________________________ */
// CreateMesh
/*!
  \param[in] comm  The MPI communicator
  \param[out] stokesDM  The Stokes DM object
  \param[out] transportDM  The Transport DM object
  \param[in] options The options table

  Creates the mesh and stores in the DM object.

  \returns
    PetscErrorCode
*/
/* ______________________________________________________________________ */
#undef __FUNCT__
#define __FUNCT__ "CreateMesh"
PetscErrorCode CreateMesh(MPI_Comm comm, DM *stokesDM, DM *transportDM, Options *options)
{
  Mesh           stokesMesh, transportMesh;
  PetscTruth     view;
  PetscMPIInt    size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (options->generateMesh) {
    double centers[4] = {0.0, 0.0, 0.0, 0.0};
    double radii[2]   = {1.0, options->r};

    Obj<ALE::Mesh> mB = ALE::MeshBuilder::createAnnularBoundary(comm, 10, centers, radii, options->debug);
    Obj<ALE::Mesh> sM = ALE::Generator::generateMesh(mB, options->interpolate);
    ierr = MeshCreate(sM->comm(), &stokesMesh);CHKERRQ(ierr);
    ierr = MeshSetMesh(stokesMesh, sM);CHKERRQ(ierr);
  } else {
    throw ALE::Exception("Mesh Reader currently removed");
  }
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  if (size > 1) {
    Mesh parallelMesh;

    ierr = MeshDistribute(stokesMesh, PETSC_NULL, &parallelMesh);CHKERRQ(ierr);
    ierr = MeshDestroy(stokesMesh);CHKERRQ(ierr);
    stokesMesh = parallelMesh;
  }
  if (options->refinementLimit > 0.0) {
    Mesh refinedMesh;

    ierr = MeshRefine(stokesMesh, options->refinementLimit, options->interpolate, &refinedMesh);CHKERRQ(ierr);
    ierr = MeshDestroy(stokesMesh);CHKERRQ(ierr);
    stokesMesh = refinedMesh;
  }
  Obj<ALE::Mesh> sM;
  ierr = MeshGetMesh(stokesMesh, sM);CHKERRQ(ierr);
  Obj<ALE::Mesh> tM = ALE::Mesh(comm, 2, sM->debug());
  tM->setSieve(sM->getSieve());
  tM->setLabel("height", sM->getLabel("height"));
  tM->setLabel("depth", sM->getLabel("depth"));
  tM->setLabel("marker", sM->getLabel("marker"));
  tM->setRealSection("coordinates", sM->getRealSection("coordinates"));
  ierr = MeshCreate(tM->comm(), &transportMesh);CHKERRQ(ierr);
  ierr = MeshSetMesh(transportMesh, tM);CHKERRQ(ierr);

  /*
   *  Mark the boundary so that we can apply Dirichelet boundary conditions.
   */
  Obj<ALE::Mesh> m;
  ierr = MeshGetMesh(stokesMesh, m);CHKERRQ(ierr);
  m->markBoundaryCells("marker");
  ierr = MeshGetMesh(transportMesh, m);CHKERRQ(ierr);
  m->markBoundaryCells("marker");
  
  /*
   *  Check to see if we want to view the mesh, and add appropriate calls if necessary
   */
  ierr = PetscOptionsHasName(PETSC_NULL, "-mesh_view_vtk", &view);CHKERRQ(ierr);
  if (view) {ierr = ViewMesh(stokesMesh, "grade2.vtk");CHKERRQ(ierr);}
  ierr = PetscOptionsHasName(PETSC_NULL, "-mesh_view", &view);CHKERRQ(ierr);
  if (view) {
    Obj<ALE::Mesh> m;
    ierr = MeshGetMesh(stokesMesh, m);CHKERRQ(ierr);
    m->view("Mesh");
  }
  *stokesDM    = (DM) stokesMesh;
  *transportDM = (DM) transportMesh;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DestroyMesh"
PetscErrorCode DestroyMesh(DM stokesDM, DM transportDM, Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshDestroy((Mesh) stokesDM);CHKERRQ(ierr);
  ierr = MeshDestroy((Mesh) transportDM);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/* ______________________________________________________________________ */
// CreateProblem
/*!
  \param[out] stokesDM  The Stokes DM object
  \param[out] transportDM  The Transport DM object
  \param[in] options The options table

  Sets up the problem to be solved in the DM object

  \returns
    PetscErrorCode
*/
/* ______________________________________________________________________ */
#undef __FUNCT__
#define __FUNCT__ "CreateProblem"
PetscErrorCode CreateProblem(DM stokesDM, DM transportDM, Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  int      velMarkers[1] = {1};
  double (*velFuncs[1])(const double *coords);
  // Create the Stokes problem (assumes 2D)
  velFuncs[0] = constant;
  ierr = CreateProblem_gen_1(stokesDM, "u0", 1, velMarkers, velFuncs,   PETSC_NULL);CHKERRQ(ierr);
  velFuncs[0] = zero;
  ierr = CreateProblem_gen_1(stokesDM, "u1", 1, velMarkers, velFuncs,   PETSC_NULL);CHKERRQ(ierr);
  ierr = CreateProblem_gen_1(stokesDM, "w0", 0, PETSC_NULL, PETSC_NULL, PETSC_NULL);CHKERRQ(ierr);
  ierr = CreateProblem_gen_1(stokesDM, "w1", 0, PETSC_NULL, PETSC_NULL, PETSC_NULL);CHKERRQ(ierr);
  // Create the default Stokes section
  Obj<ALE::Mesh> m;

  ierr = MeshGetMesh((Mesh) stokesDM, m);CHKERRQ(ierr);
  const ALE::Obj<ALE::Mesh::real_section_type> s = m->getRealSection("default");
  s->setDebug(options->debug);
  m->calculateIndices();
  m->setupField(s, 2);
  if (options->debug) {s->view("Default Stokes field");}
  // Create the Transport problem (assumes 2D)
  ierr = CreateProblem_gen_1(transportDM, "z0", 0, PETSC_NULL, PETSC_NULL, PETSC_NULL);CHKERRQ(ierr);
  ierr = CreateProblem_gen_1(transportDM, "z1", 0, PETSC_NULL, PETSC_NULL, PETSC_NULL);CHKERRQ(ierr);
  // Create the default Transport section
  ierr = MeshGetMesh((Mesh) transportDM, m);CHKERRQ(ierr);
  const ALE::Obj<ALE::Mesh::real_section_type> t = m->getRealSection("default");
  t->setDebug(options->debug);
  m->calculateIndices();
  m->setupField(t, 2);
  if (options->debug) {t->view("Default Transport field");}
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
  ierr = DMMGSetSNESLocal(*dmmg, Stokes_Rhs_Unstructured, Stokes_Jac_Unstructured, 0, 0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/* ______________________________________________________________________ */
// Stoke_Rhs_Unstructured
/*!
  \param[out] mesh
  \param[out]  X
  \param[out] section
  \param[out] ctx


  \returns
    PetscErrorCode
*/
/* ______________________________________________________________________ */
#undef __FUNCT__
#define __FUNCT__ "Stokes_Rhs_Unstructured"
PetscErrorCode Stokes_Rhs_Unstructured(Mesh mesh, SectionReal X, SectionReal section, void *ctx)
{
  Options       *options = (Options *) ctx;
  double      (**funcs)(const double *) = options->funcs;
  Obj<ALE::Mesh> m;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  const Obj<ALE::Mesh::real_section_type>& coordinates = m->getRealSection("coordinates");
  const Obj<ALE::Mesh::label_sequence>&    cells       = m->heightStratum(0);
  const int                                dim         = m->getDimension();
  const Obj<std::set<std::string> >&       discs       = m->getDiscretizations();
  int          totBasisFuncs = 0;
  double      *t_der, *b_der, *coords, *v0, *J, *invJ, detJ;
  PetscScalar *elemVec, *elemMat;

  ierr = SectionRealZero(section);CHKERRQ(ierr);
  for(std::set<std::string>::const_iterator f_iter = discs->begin(); f_iter != discs->end(); ++f_iter) {
    totBasisFuncs += m->getDiscretization(*f_iter)->getBasisSize();
  }
  ierr = PetscMalloc2(totBasisFuncs,PetscScalar,&elemVec,totBasisFuncs*totBasisFuncs,PetscScalar,&elemMat);CHKERRQ(ierr);
  ierr = PetscMalloc6(dim,double,&t_der,dim,double,&b_der,dim,double,&coords,dim,double,&v0,dim*dim,double,&J,dim*dim,double,&invJ);CHKERRQ(ierr);
  // Loop over cells
  for(ALE::Mesh::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
    PetscScalar *x;
    int          field = 0;

    m->computeElementGeometry(coordinates, *c_iter, v0, J, invJ, detJ);
    //ierr = SectionRealRestrict(X, *c_iter, &x);CHKERRQ(ierr);
    {
      Obj<ALE::Mesh::real_section_type> sX;

      ierr = SectionRealGetSection(X, sX);CHKERRQ(ierr);
      x = (PetscScalar *) m->restrictNew(sX, *c_iter);
    }
    if (detJ < 0) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %d", detJ, *c_iter);
    ierr = PetscMemzero(elemVec, totBasisFuncs * sizeof(PetscScalar));CHKERRQ(ierr);
    for(std::set<std::string>::const_iterator f_iter = discs->begin(); f_iter != discs->end(); ++f_iter, ++field) {
      const Obj<ALE::Discretization>& disc          = m->getDiscretization(*f_iter);
      const int                       numQuadPoints = disc->getQuadratureSize();
      const double                   *quadPoints    = disc->getQuadraturePoints();
      const double                   *quadWeights   = disc->getQuadratureWeights();
      const int                       numBasisFuncs = disc->getBasisSize();
      const double                   *basis         = disc->getBasis();
      const double                   *basisDer      = disc->getBasisDerivatives();
      const int                      *indices       = disc->getIndices();

      ierr = PetscMemzero(elemMat, numBasisFuncs*totBasisFuncs * sizeof(PetscScalar));CHKERRQ(ierr);
      // Loop over quadrature points
      for(int q = 0; q < numQuadPoints; ++q) {
        for(int d = 0; d < dim; d++) {
          coords[d] = v0[d];
          for(int e = 0; e < dim; e++) {
            coords[d] += J[d*dim+e]*(quadPoints[q*dim+e] + 1.0);
          }
        }
        PetscScalar funcVal = (*funcs[field])(coords);

        // Loop over trial functions
        for(int f = 0; f < numBasisFuncs; ++f) {
          // Constant part
          elemVec[indices[f]] -= basis[q*numBasisFuncs+f]*funcVal*quadWeights[q]*detJ;
          // Linear part
          //if (*f_iter == "pressure") {
          if (field == 0) {
            // Divergence of u
            const Obj<ALE::Discretization>& u         = m->getDiscretization("u0");
            const int                       numUFuncs = u->getBasisSize();
            const double                   *uBasisDer = u->getBasisDerivatives();
            const int                      *uIndices  = u->getIndices();

            for(int g = 0; g < numUFuncs; ++g) {
              PetscScalar uDiv = 0.0;

              for(int e = 0; e < dim; ++e) uDiv += invJ[e*dim+0]*uBasisDer[(q*numUFuncs+g)*dim+e];
              elemMat[f*totBasisFuncs+uIndices[g]] += basis[q*numBasisFuncs+f]*uDiv*quadWeights[q]*detJ;
            }
            // Divergence of v
            const Obj<ALE::Discretization>& v         = m->getDiscretization("u1");
            const int                       numVFuncs = v->getBasisSize();
            const double                   *vBasisDer = v->getBasisDerivatives();
            const int                      *vIndices  = v->getIndices();

            for(int g = 0; g < numVFuncs; ++g) {
              PetscScalar vDiv = 0.0;

              for(int e = 0; e < dim; ++e) vDiv += invJ[e*dim+1]*vBasisDer[(q*numVFuncs+g)*dim+e];
              elemMat[f*totBasisFuncs+vIndices[g]] += basis[q*numBasisFuncs+f]*vDiv*quadWeights[q]*detJ;
            }
          } else {
            // Laplacian of u or v
            for(int d = 0; d < dim; ++d) {
              t_der[d] = 0.0;
              for(int e = 0; e < dim; ++e) t_der[d] += invJ[e*dim+d]*basisDer[(q*numBasisFuncs+f)*dim+e];
            }
            for(int g = 0; g < numBasisFuncs; ++g) {
              for(int d = 0; d < dim; ++d) {
                b_der[d] = 0.0;
                for(int e = 0; e < dim; ++e) b_der[d] += invJ[e*dim+d]*basisDer[(q*numBasisFuncs+g)*dim+e];
              }
              PetscScalar product = 0.0;

              for(int d = 0; d < dim; ++d) product += t_der[d]*b_der[d];
              elemMat[f*totBasisFuncs+indices[g]] += product*quadWeights[q]*detJ;
            }
            // Gradient of pressure
            const Obj<ALE::Discretization>& pres         = m->getDiscretization("p");
            const int                       numPresFuncs = pres->getBasisSize();
            const double                   *presBasisDer = pres->getBasisDerivatives();
            const int                      *presIndices  = pres->getIndices();

            for(int g = 0; g < numPresFuncs; ++g) {
              PetscScalar presGrad = 0.0;
              const int   d        = field-1;

              for(int e = 0; e < dim; ++e) presGrad -= invJ[e*dim+d]*presBasisDer[(q*numPresFuncs+g)*dim+e];
              elemMat[f*totBasisFuncs+presIndices[g]] += basis[q*numBasisFuncs+f]*presGrad*quadWeights[q]*detJ;
            }
          }
        }
      }
      if (options->debug) {
        std::cout << "Constant element vector for field " << *f_iter << ":" << std::endl;
        for(int f = 0; f < numBasisFuncs; ++f) {
          std::cout << "  " << elemVec[indices[f]] << std::endl;
        }
      }
      // Add linear contribution
      for(int f = 0; f < numBasisFuncs; ++f) {
        for(int g = 0; g < totBasisFuncs; ++g) {
          elemVec[indices[f]] += elemMat[f*totBasisFuncs+g]*x[g];
        }
      }
      if (options->debug) {
        ostringstream label; label << "Element Matrix for field " << *f_iter;
        std::cout << ALE::Mesh::printMatrix(label.str(), numBasisFuncs, totBasisFuncs, elemMat, m->commRank()) << std::endl;
        std::cout << "Linear element vector for field " << *f_iter << ":" << std::endl;
        for(int f = 0; f < numBasisFuncs; ++f) {
          std::cout << "  " << elemVec[indices[f]] << std::endl;
        }
      }
    }
    if (options->debug) {
      std::cout << "Element vector:" << std::endl;
      for(int f = 0; f < totBasisFuncs; ++f) {
        std::cout << "  " << elemVec[f] << std::endl;
      }
    }
    ierr = SectionRealUpdateAdd(section, *c_iter, elemVec);CHKERRQ(ierr);
    if (options->debug) {
      ierr = SectionRealView(section, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree2(elemVec,elemMat);CHKERRQ(ierr);
  ierr = PetscFree6(t_der,b_der,coords,v0,J,invJ);CHKERRQ(ierr);
  // Exchange neighbors
  ierr = SectionRealComplete(section);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/* ______________________________________________________________________ */
// StokeJac_Unstructured
/*!
  \param[out] mesh The Mesh Object
  \param[out] section The section to solve the problem on
  \param[out] A The operator matrix
  \param[in] *ctx The current context


  \returns
    PetscErrorCode
*/
/* ______________________________________________________________________ */
#undef __FUNCT__
#define __FUNCT__ "Stokes_Jac_Unstructured"
PetscErrorCode Stokes_Jac_Unstructured(Mesh mesh, SectionReal section, Mat A, void *ctx)
{
  Options       *options = (Options *) ctx;
  Obj<ALE::Mesh::real_section_type> s;
  Obj<ALE::Mesh> m;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = SectionRealGetSection(section, s);CHKERRQ(ierr);
  const Obj<ALE::Mesh::real_section_type>& coordinates = m->getRealSection("coordinates");
  const Obj<ALE::Mesh::label_sequence>&    cells       = m->heightStratum(0);
  const Obj<ALE::Mesh::order_type>&        order       = m->getFactory()->getGlobalOrder(m, "default", s);
  const int                                dim         = m->getDimension();
  const Obj<std::set<std::string> >&       discs       = m->getDiscretizations();
  int          totBasisFuncs = 0;
  double      *t_der, *b_der, *v0, *J, *invJ, detJ;
  PetscScalar *elemMat;

  ierr = MatZeroEntries(A);CHKERRQ(ierr);
  for(std::set<std::string>::const_iterator f_iter = discs->begin(); f_iter != discs->end(); ++f_iter) {
    totBasisFuncs += m->getDiscretization(*f_iter)->getBasisSize();
  }
  ierr = PetscMalloc(totBasisFuncs*totBasisFuncs * sizeof(PetscScalar), &elemMat);CHKERRQ(ierr);
  ierr = PetscMalloc5(dim,double,&t_der,dim,double,&b_der,dim,double,&v0,dim*dim,double,&J,dim*dim,double,&invJ);CHKERRQ(ierr);
  // Loop over cells
  for(ALE::Mesh::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
    PetscScalar *x;
    int          field = 0;

    x = (PetscScalar *) m->restrictNew(s, *c_iter);
    m->computeElementGeometry(coordinates, *c_iter, v0, J, invJ, detJ);
    if (detJ < 0) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %d", detJ, *c_iter);
    ierr = PetscMemzero(elemMat, totBasisFuncs*totBasisFuncs * sizeof(PetscScalar));CHKERRQ(ierr);
    for(std::set<std::string>::const_iterator f_iter = discs->begin(); f_iter != discs->end(); ++f_iter, ++field) {
      const Obj<ALE::Discretization>& disc          = m->getDiscretization(*f_iter);
      const int                       numQuadPoints = disc->getQuadratureSize();
      const double                   *quadPoints    = disc->getQuadraturePoints();
      const double                   *quadWeights   = disc->getQuadratureWeights();
      const int                       numBasisFuncs = disc->getBasisSize();
      const double                   *basis         = disc->getBasis();
      const double                   *basisDer      = disc->getBasisDerivatives();
      const int                      *indices       = disc->getIndices();

      // Loop over quadrature points
      for(int q = 0; q < numQuadPoints; ++q) {
        // Loop over trial functions
        for(int f = 0; f < numBasisFuncs; ++f) {
          //if (*f_iter == "pressure") {
          if (field == 0) {
            // Divergence of u
            const Obj<ALE::Discretization>& u         = m->getDiscretization("u0");
            const int                       numUFuncs = u->getBasisSize();
            const double                   *uBasisDer = u->getBasisDerivatives();
            const int                      *uIndices  = u->getIndices();

            for(int g = 0; g < numUFuncs; ++g) {
              PetscScalar uDiv = 0.0;

              for(int e = 0; e < dim; ++e) uDiv += invJ[e*dim+0]*uBasisDer[(q*numUFuncs+g)*dim+e];
              elemMat[indices[f]*totBasisFuncs+uIndices[g]] += basis[q*numBasisFuncs+f]*uDiv*quadWeights[q]*detJ;
            }
            // Divergence of v
            const Obj<ALE::Discretization>& v         = m->getDiscretization("u1");
            const int                       numVFuncs = v->getBasisSize();
            const double                   *vBasisDer = v->getBasisDerivatives();
            const int                      *vIndices  = v->getIndices();

            for(int g = 0; g < numVFuncs; ++g) {
              PetscScalar vDiv = 0.0;

              for(int e = 0; e < dim; ++e) vDiv += invJ[e*dim+1]*vBasisDer[(q*numVFuncs+g)*dim+e];
              elemMat[indices[f]*totBasisFuncs+vIndices[g]] += basis[q*numBasisFuncs+f]*vDiv*quadWeights[q]*detJ;
            }
          } else {
            // Laplacian of u or v
            for(int d = 0; d < dim; ++d) {
              t_der[d] = 0.0;
              for(int e = 0; e < dim; ++e) t_der[d] += invJ[e*dim+d]*basisDer[(q*numBasisFuncs+f)*dim+e];
            }
            for(int g = 0; g < numBasisFuncs; ++g) {
              for(int d = 0; d < dim; ++d) {
                b_der[d] = 0.0;
                for(int e = 0; e < dim; ++e) b_der[d] += invJ[e*dim+d]*basisDer[(q*numBasisFuncs+g)*dim+e];
              }
              PetscScalar product = 0.0;

              for(int d = 0; d < dim; ++d) product += t_der[d]*b_der[d];
              elemMat[indices[f]*totBasisFuncs+indices[g]] += product*quadWeights[q]*detJ;
            }
            // Gradient of pressure
            const Obj<ALE::Discretization>& pres         = m->getDiscretization("p");
            const int                       numPresFuncs = pres->getBasisSize();
            const double                   *presBasisDer = pres->getBasisDerivatives();
            const int                      *presIndices  = pres->getIndices();

            for(int g = 0; g < numPresFuncs; ++g) {
              PetscScalar presGrad = 0.0;
              const int   d        = field-1;

              for(int e = 0; e < dim; ++e) presGrad -= invJ[e*dim+d]*presBasisDer[(q*numPresFuncs+g)*dim+e];
              elemMat[indices[f]*totBasisFuncs+presIndices[g]] += basis[q*numBasisFuncs+f]*presGrad*quadWeights[q]*detJ;
            }
          }
        }
      }
    }
    ierr = updateOperator(A, m, s, order, *c_iter, elemMat, ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = PetscFree(elemMat);CHKERRQ(ierr);
  ierr = PetscFree5(t_der,b_der,v0,J,invJ);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "Transport_Rhs_Unstructured"
PetscErrorCode Transport_Rhs_Unstructured(Mesh mesh, SectionReal X, SectionReal section, void *ctx)
{
  Options       *options = (Options *) ctx;
  double      (**funcs)(const double *) = options->funcs;
  Obj<ALE::Mesh> m;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Transport_Jac_Unstructured"
PetscErrorCode Transport_Jac_Unstructured(Mesh mesh, SectionReal section, Mat A, void *ctx)
{
  Options       *options = (Options *) ctx;
  Obj<ALE::Mesh::real_section_type> s;
  Obj<ALE::Mesh> m;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
// ---------------------------------------------------------------------------------------------------------------------
// End of file grade2.cxx
