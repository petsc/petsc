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
 *                      (0,1)
 *                  -------------
 *             ----/             \----
 *               -/               \-
 *              -/                 \-
 *             -/     -------       \-
 *             /    -/       \-      \
 *            /     /      r  \       \
 *           /      |   X-----|        \
 *          /       \         /         \
 *          |       -\       /-         |
 * (-1,0)-->|         -------           |<---(1,0)
 *          |      u \cdot t = 1        |
 *          \      u \cdot n = 0        /
 *           \                         /
 *            \                       /
 *             \                     /
 *             -\                   /-
 *              -\                 /-
 *               -\               /-
 *             ----\     u = 0   /----
 *                  -------------
 *                      (0,-1)
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
 * However, we would like the iterated penalty formulation
 *
 *    <\nabla v, \nabla u^n> + r <\nabla\cdot v, \nabla\cdot u^n> = <v, f> - <\nabla\cdot v, \nabla\cdot w^n>
 *        w^{n+1} = w^n + \rho u^n
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

typedef enum {VISC_CONSTANT, VISC_VARIABLE} ViscosityType;

// ---------------------------------------------------------------------------------------------------------------------
// Top level data definitions
typedef struct {
  PetscInt      debug;                                        // The debugging level
  PetscInt      dim;                                          // The dimension
  PetscTruth    generateMesh;                                 // Generate the unstructure mesh
  PetscTruth    square;                                       // Use the square mesh test problem
  PetscTruth    interpolate;                                  // Generate intermediate mesh elements
  PetscReal     refinementLimit;                              // The largest allowable cell volume
  char          baseFilename[2048];                           // The base filename for mesh files
  PetscReal     radius;                                       // The inner radius
  double        (*funcs[2])(const double []);                 // The function to project
  PetscReal     r,rho;                                        // IP parameters
  PetscReal     mu,alpha;                                     // Transport parameters
  DM            paramDM;                                      // Parameter DM which holds w
  std::string   paramName;                                    // Name of the parameter section
  SectionReal   exactU;                                       // Discrete exact Stokes velocity solution
  SectionReal   exactW;                                       // Discrete exact Stokes pressure potential solution
  ViscosityType viscosityModel;                               // The viscosity model
  double        (*viscosity)(const double []);                // A variable viscosity function
} Options;

double zero(const double x[]) {
  return 0.0;
}

double one(const double x[]) {
  return 1.0;
}

double linearViscosity(const double x[]) {
  return x[0]+0.01;
}

double constant(const double x[]) {
  return -3.0;
}

double radius = 0.0;

// Assuming center (0.0,0.0)
double uAnnulus(const double x[]) {
  const double r = sqrt(x[0]*x[0] + x[1]*x[1]);

  if (r <= 1.000001*radius) {
    return x[1];
  }
  return 0.0;
}

// Assuming center (0.0,0.0)
double vAnnulus(const double x[]) {
  const double r = sqrt(x[0]*x[0] + x[1]*x[1]);

  if (r <= 1.000001*radius) {
    return -x[0];
  }
  return 0.0;
}

double quadratic_2d_u(const double x[]) {
  return x[0]*x[0] - 2.0*x[0]*x[1];
}

double quadratic_2d_v(const double x[]) {
  return x[1]*x[1] - 2.0*x[0]*x[1];
}

// \nabla\cdot w = p
double quadratic_2d_w0(const double x[]) {
  return 0.5*(x[0]*x[0] - x[0]);
}

double quadratic_2d_w1(const double x[]) {
  return 0.5*(x[1]*x[1] - x[1]);
}

double quadratic_2d_p(const double x[]) {
  return x[0] + x[1] - 1.0;
}

#include "grade2_quadrature.h"

// ---------------------------------------------------------------------------------------------------------------------
// Function Prototypes

PetscErrorCode ProcessOptions(MPI_Comm comm, Options *options);
PetscErrorCode CreatePartition(Mesh mesh, SectionInt *partition);
PetscErrorCode ViewMesh(Mesh mesh, const char filename[]);
PetscErrorCode ViewSection(Mesh mesh, SectionReal section, const char filename[]);
PetscErrorCode CreateMesh(MPI_Comm comm, DM *stokesDM, DM *paramDM, DM *transportDM, Options *options);
PetscErrorCode DestroyMesh(DM stokesDM, DM transportDM, Options *options);
PetscErrorCode CreateProblem(DM stokesDM, DM paramDM, DM transportDM, Options *options);
PetscErrorCode CreateSolver(DM dm, DMMG **dmmg, Options *options);
PetscErrorCode CreateExactSolution(DM dm, SectionReal *sol, Options *options);
PetscErrorCode CheckError(DM dm, SectionReal sol, Options *options);
PetscErrorCode CheckStokesResidual(DM dm, SectionReal sol, const std::string& paramName, Options *options);

PetscErrorCode SolveStokes(DMMG *dmmg, Options *options);
PetscErrorCode Stokes_Rhs_Unstructured(Mesh mesh, SectionReal X, SectionReal section, void *ctx);
PetscErrorCode Stokes_Jac_Unstructured(Mesh mesh, SectionReal section, Mat A, void *ctx);
PetscErrorCode IterateStokes(DMMG *dmmg, Options *options);
PetscErrorCode CheckStokesConvergence(DMMG *dmmg, PetscTruth *iterate, Options *options);
PetscErrorCode DivNorm_L2(Mesh mesh, SectionReal X, PetscReal *norm, Options *options);

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
  DM             stokesDM, paramDM, transportDM;
  PetscErrorCode ierr;
  PetscTruth     iterate = PETSC_TRUE;
  PetscInt       iter = 0, max_iter = 1;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr);
  comm = PETSC_COMM_WORLD;
  ierr = ProcessOptions(comm, &options);CHKERRQ(ierr);
  try {
    DMMG *stokes;
    PetscReal error;

    ierr = CreateMesh(comm, &stokesDM, &paramDM, &transportDM, &options);CHKERRQ(ierr);
    ierr = CreateProblem(stokesDM, paramDM, transportDM, &options);CHKERRQ(ierr);
    ierr = CreateExactSolution(stokesDM, &options.exactU, &options);CHKERRQ(ierr);
    ierr = CreateExactSolution(paramDM,  &options.exactW, &options);CHKERRQ(ierr);
    ierr = CheckError(stokesDM, options.exactU, &options);CHKERRQ(ierr);
    ierr = CheckError(paramDM, options.exactW, &options);CHKERRQ(ierr);
    ierr = DivNorm_L2((Mesh) stokesDM, options.exactU, &error, &options);CHKERRQ(ierr);
    ierr = CheckStokesResidual(stokesDM, options.exactU, "exactSolution", &options);CHKERRQ(ierr);
    ierr = CreateSolver(stokesDM, &stokes, &options);CHKERRQ(ierr)

    while (iterate and max_iter >= ++iter){
      ierr = SolveStokes(stokes, &options);CHKERRQ(ierr);
      //ierr = SolveTransport(transportDM, &options);CHKERRQ(ierr);
      //ierr = CheckStoppingCriteria(stokesDM, &iterate, &options);CHKERRQ(ierr);
    }
    PetscPrintf(comm, "Writing stokesDM\n");
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
    // CHECK options->paramName = "exactSolution";
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
    SectionReal sol;

    ierr = MeshGetSectionReal(mesh, "default", &sol);CHKERRQ(ierr);
    if (options->debug) {ierr = SectionRealView(sol, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);}
    ierr = CheckError((DM) mesh, sol, options);CHKERRQ(ierr);
    ierr = SectionRealDestroy(sol);CHKERRQ(ierr);
    ierr = IterateStokes(dmmg, options);CHKERRQ(ierr);
    ierr = MeshGetSectionReal((Mesh) options->paramDM, "default", &sol);CHKERRQ(ierr);
    if (options->debug) {ierr = SectionRealView(sol, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);}
    ierr = CheckError(options->paramDM, sol, options);CHKERRQ(ierr);
    ierr = SectionRealDestroy(sol);CHKERRQ(ierr);
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
  Obj<ALE::Mesh> m;
  Obj<ALE::Mesh> pM;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh((Mesh) DMMGGetFine(dmmg)->dm, m);CHKERRQ(ierr);
  ierr = MeshGetMesh((Mesh) options->paramDM, pM);CHKERRQ(ierr);
  const Obj<ALE::Mesh::real_section_type>& u = m->getRealSection("default");
  const Obj<ALE::Mesh::real_section_type>& w = pM->getRealSection(options->paramName);

  if (m->debug()) {w->view("w before");}
  w->axpy(options->rho, u);
  if (m->debug()) {w->view("w after");}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DivNorm_L2"
PetscErrorCode DivNorm_L2(Mesh mesh, SectionReal X, PetscReal *norm, Options *options)
{
  Obj<ALE::Mesh> m;
  Obj<ALE::Mesh::real_section_type> sX;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = SectionRealGetSection(X, sX);CHKERRQ(ierr);
  const Obj<ALE::Mesh::real_section_type>& coordinates   = m->getRealSection("coordinates");
  const Obj<ALE::Mesh::label_sequence>&    cells         = m->heightStratum(0);
  const int                                dim           = m->getDimension();
  const Obj<std::set<std::string> >&       discs         = m->getDiscretizations();
  const int                                numQuadPoints = m->getDiscretization(*discs->begin())->getQuadratureSize();
  const double                            *quadWeights   = m->getDiscretization(*discs->begin())->getQuadratureWeights();
  double                                   localNorm     = 0.0;
  double *v0, *J, *invJ, detJ;

  ierr = PetscMalloc3(dim,double,&v0,dim*dim,double,&J,dim*dim,double,&invJ);CHKERRQ(ierr);
  // Loop over cells
  for(ALE::Mesh::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
    m->computeElementGeometry(coordinates, *c_iter, v0, J, invJ, detJ);
    const PetscScalar *x = m->restrictNew(sX, *c_iter);
    double elemNorm = 0.0;

    if (detJ < 0) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %d", detJ, *c_iter);
    // Loop over quadrature points
    for(int q = 0; q < numQuadPoints; ++q) {
      PetscScalar divU  = 0.0;
      int         field = 0;

      for(std::set<std::string>::const_iterator f_iter = discs->begin(); f_iter != discs->end(); ++f_iter, ++field) {
        const Obj<ALE::Discretization>& disc          = m->getDiscretization(*f_iter);
        const int                       numBasisFuncs = disc->getBasisSize();
        const double                   *basisDer      = disc->getBasisDerivatives();
        const int                      *indices       = disc->getIndices();

        for(int f = 0; f < numBasisFuncs; ++f) {
          PetscScalar deriv = 0.0;
          for(int e = 0; e < dim; ++e) deriv += invJ[e*dim+field]*basisDer[(q*numBasisFuncs+f)*dim+e];
          divU += x[indices[f]]*deriv;
        }
      }
      elemNorm += divU*divU*quadWeights[q];
    }    
    elemNorm *= detJ;
    if (m->debug()) {
      std::cout << "Element " << *c_iter << " norm^2: " << elemNorm << std::endl;
    }
    localNorm += elemNorm;
  }
  ierr = MPI_Allreduce(&localNorm, norm, 1, MPI_DOUBLE, MPI_SUM, m->comm());CHKERRQ(ierr);
  ierr = PetscFree3(v0,J,invJ);CHKERRQ(ierr);
  *norm = sqrt(*norm);
  PetscFunctionReturn(0);
}

/* 
 * check div(u) < tol
 */
#undef __FUNCT__
#define __FUNCT__ "CheckStokesConvergence"
PetscErrorCode CheckStokesConvergence(DMMG *dmmg, PetscTruth *iterate, Options *options)
{
  Mesh            mesh = (Mesh) DMMGGetFine(dmmg)->dm;
  const PetscReal tol  = 1.0e-5;
  SectionReal     u;
  PetscReal       error;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = MeshGetSectionReal(mesh, "default", &u);CHKERRQ(ierr);
  ierr = DivNorm_L2(mesh, u, &error, options);CHKERRQ(ierr);
  ierr = SectionRealDestroy(u);CHKERRQ(ierr);
  PetscPrintf(dmmg[0]->comm, "Checking Stokes convergence: div_error = %g\n", error);
  if (error < tol) {
    *iterate = PETSC_FALSE;
  } else {
    *iterate = PETSC_TRUE;
  }
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
  ierr = DMMGSetFromOptions(dmmg);CHKERRQ(ierr);
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
  Obj<ALE::Mesh> m;
  PetscReal      error;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh((Mesh) dm, m);CHKERRQ(ierr);
  const Obj<ALE::Mesh::real_section_type>& coordinates = m->getRealSection("coordinates");
  const Obj<ALE::Mesh::label_sequence>&    cells       = m->heightStratum(0);
  const int                                dim         = m->getDimension();
  const Obj<std::set<std::string> >&       discs       = m->getDiscretizations();
  const Obj<ALE::Mesh::real_section_type>& X = m->getRealSection("default");

  double *coords, *v0, *J, *invJ, detJ;
  double  localError = 0.0;

  ierr = PetscMalloc4(dim,double,&coords,dim,double,&v0,dim*dim,double,&J,dim*dim,double,&invJ);CHKERRQ(ierr);
  // Loop over cells
  for(ALE::Mesh::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
    m->computeElementGeometry(coordinates, *c_iter, v0, J, invJ, detJ);
    const PetscScalar *x = m->restrictNew(X, *c_iter);
    double elemError = 0.0;

    for(std::set<std::string>::const_iterator f_iter = discs->begin(); f_iter != discs->end(); ++f_iter) {
      const Obj<ALE::Discretization>&    disc          = m->getDiscretization(*f_iter);
      const int                          numQuadPoints = disc->getQuadratureSize();
      const double                      *quadPoints    = disc->getQuadraturePoints();
      const double                      *quadWeights   = disc->getQuadratureWeights();
      const int                          numBasisFuncs = disc->getBasisSize();
      const double                      *basisDer      = disc->getBasisDerivatives();
      const int                         *indices       = disc->getIndices();

      // Loop over quadrature points
      for(int q = 0; q < numQuadPoints; ++q) {
        for(int d = 0; d < dim; d++) {
          coords[d] = v0[d];
          for(int e = 0; e < dim; e++) {
            coords[d] += J[d*dim+e]*(quadPoints[q*dim+e] + 1.0);
          }
        }
        PetscScalar interpolant = 0.0;

        for(int f = 0; f < numBasisFuncs; ++f) {
          interpolant += x[indices[f]]*basisDer[q*numBasisFuncs+f];
        }
        elemError += interpolant*interpolant*quadWeights[q];
      }
    }    
    if (m->debug()) {
      std::cout << "Element " << *c_iter << " error: " << elemError << std::endl;
    }
    localError += elemError;
  }
  ierr = MPI_Allreduce(&localError, &error, 1, MPI_DOUBLE, MPI_SUM, m->comm());CHKERRQ(ierr);
  ierr = PetscFree4(coords,v0,J,invJ);CHKERRQ(ierr);
  error = sqrt(error);
  printf("Checking grade 2 convergence: div_error = %f\n",error);
  if (error < 1e-5)
    *iterate = PETSC_FALSE;
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
    Obj<ALE::Mesh::real_section_type> velocityX = sol->getFibration(0);
    Obj<ALE::Mesh::real_section_type> velocityY = sol->getFibration(1);

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
  const char    *viscTypes[2] = {"constant", "variable"};
  ostringstream  filename;
  PetscInt       visc;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->debug           = 0;
  options->dim             = 2;
  options->generateMesh    = PETSC_TRUE;
  options->interpolate     = PETSC_TRUE;
  options->refinementLimit = 0.0;
  options->radius          = 0.5;
  options->r               = -1.0e5;
  options->rho             =  1.0e5;
  options->mu              = 1;
  options->alpha           = 1;
  options->square          = PETSC_FALSE;
  options->viscosityModel  = VISC_CONSTANT;

  ierr = PetscOptionsBegin(comm, "", "Grade 2 journal bearing Options", "DMMG");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-debug", "The debugging level", "grade2.cxx", options->debug, &options->debug, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTruth("-generate", "Generate the unstructured mesh", "grade2.cxx", options->generateMesh, &options->generateMesh, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTruth("-square", "Use the unit square test problem", "grade2.cxx", options->square, &options->square, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTruth("-interpolate", "Generate intermediate mesh elements", "grade2.cxx", options->interpolate, &options->interpolate, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-refinement_limit", "The largest allowable cell volume", "grade2.cxx", options->refinementLimit, &options->refinementLimit, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-radius", "The inner radius ", "grade2.cxx", options->radius, &options->radius, PETSC_NULL);CHKERRQ(ierr);
  radius = options->radius;
  ierr = PetscOptionsReal("-r", "The IP parameter r", "grade2.cxx", options->r, &options->r, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-rho", "The IP parameter rho", "grade2.cxx", options->rho, &options->rho, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mu", "The transport parameter mu", "grade2.cxx", options->mu, &options->mu, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-alpha", "The transport parameter alpha", "grade2.cxx", options->alpha, &options->alpha, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-refinement_limit", "The largest allowable cell volume", "grade2.cxx", options->refinementLimit, &options->refinementLimit, PETSC_NULL);CHKERRQ(ierr);
  filename << "data/journal_bearing";
  ierr = PetscStrcpy(options->baseFilename, filename.str().c_str());CHKERRQ(ierr);
  ierr = PetscOptionsString("-base_filename", "The base filename for mesh files", "grade2.cxx", options->baseFilename, options->baseFilename, 2048, PETSC_NULL);CHKERRQ(ierr);
  visc = options->viscosityModel;
  ierr = PetscOptionsEList("-viscosity_model", "The viscosity model", "grade2.cxx", viscTypes, 2, viscTypes[options->viscosityModel], &visc, PETSC_NULL);CHKERRQ(ierr);
  options->viscosityModel = (ViscosityType) visc;
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
PetscErrorCode CreateMesh(MPI_Comm comm, DM *stokesDM, DM *paramDM, DM *transportDM, Options *options)
{
  Mesh           stokesMesh, paramMesh, transportMesh;
  PetscTruth     view;
  PetscMPIInt    size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (options->generateMesh) {
    if (options->square){
      double lower[2] = {0.0, 0.0};
      double upper[2] = {1.0, 1.0};
      int    edges[2] = {2, 2};
            
      Obj<ALE::Mesh> mB = ALE::MeshBuilder::createSquareBoundary(comm, lower, upper, edges, options->debug);
      Obj<ALE::Mesh> sM = ALE::Generator::generateMesh(mB, options->interpolate);
      ierr = MeshCreate(sM->comm(), &stokesMesh);CHKERRQ(ierr);
      ierr = MeshSetMesh(stokesMesh, sM);CHKERRQ(ierr);
    } else {
      double centers[4] = {0.0, 0.0, 0.0, 0.0};
      double radii[2]   = {1.0, options->radius};
      
      Obj<ALE::Mesh> mB = ALE::MeshBuilder::createAnnularBoundary(comm, 10, centers, radii, options->debug);
      Obj<ALE::Mesh> sM = ALE::Generator::generateMesh(mB, options->interpolate);
      ierr = MeshCreate(sM->comm(), &stokesMesh);CHKERRQ(ierr);
      ierr = MeshSetMesh(stokesMesh, sM);CHKERRQ(ierr);
    }
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
  Obj<ALE::Mesh> pM = ALE::Mesh(sM->comm(), sM->getDimension(), sM->debug());
  pM->copy(sM);
  ierr = MeshCreate(pM->comm(), &paramMesh);CHKERRQ(ierr);
  ierr = MeshSetMesh(paramMesh, pM);CHKERRQ(ierr);
  Obj<ALE::Mesh> tM = ALE::Mesh(sM->comm(), sM->getDimension(), sM->debug());
  tM->copy(sM);
  ierr = MeshCreate(tM->comm(), &transportMesh);CHKERRQ(ierr);
  ierr = MeshSetMesh(transportMesh, tM);CHKERRQ(ierr);

  /*
   *  Mark the boundary so that we can apply Dirichelet boundary conditions.
   */
  Obj<ALE::Mesh> m;
  ierr = MeshGetMesh(stokesMesh, m);CHKERRQ(ierr);
  m->markBoundaryCells("marker");
  ierr = MeshGetMesh(paramMesh, m);CHKERRQ(ierr);
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
  *paramDM     = (DM) paramMesh;
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
  \param[out] paramDM  The Parameter DM object
  \param[out] transportDM  The Transport DM object
  \param[in] options The options table

  Sets up the problem to be solved in the DM object

  \returns
    PetscErrorCode
*/
/* ______________________________________________________________________ */
#undef __FUNCT__
#define __FUNCT__ "CreateProblem"
PetscErrorCode CreateProblem(DM stokesDM, DM paramDM, DM transportDM, Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  int      velMarkers[1] = {1};
  double (*velFuncs[1])(const double *coords);
  // Create the Stokes problem (assumes 2D)
  if (options->square) {
    velFuncs[0] = quadratic_2d_u;
  } else {
    velFuncs[0] = uAnnulus;
  }
  ierr = CreateProblem_gen_1(stokesDM, "u0", 1, velMarkers, velFuncs, velFuncs[0]);CHKERRQ(ierr);
  if (options->square) {
    velFuncs[0] = quadratic_2d_v;
  } else {
    velFuncs[0] = vAnnulus;
  }
  ierr = CreateProblem_gen_1(stokesDM, "u1", 1, velMarkers, velFuncs, velFuncs[0]);CHKERRQ(ierr);
  options->funcs[0] = constant;
  options->funcs[1] = constant;
  // Create viscosity model
  if (options->viscosityModel == VISC_CONSTANT) {
    options->viscosity = one;
  } else if (options->viscosityModel == VISC_VARIABLE) {
    options->viscosity = linearViscosity;
  } else {
    SETERRQ(PETSC_ERR_ARG_WRONG, "Unrecognized viscosity model");
  }
  // Create the default Stokes section
  Obj<ALE::Mesh> m;

  ierr = MeshGetMesh((Mesh) stokesDM, m);CHKERRQ(ierr);
  const ALE::Obj<ALE::Mesh::real_section_type> s = m->getRealSection("default");
  s->setDebug(options->debug);
  m->calculateIndices();
  m->setupField(s, 2);
  if (options->debug) {s->view("Default Stokes velocity");}
  // Create the w parameter field
  ierr = CreateProblem_gen_1(paramDM, "w0", 0, PETSC_NULL, PETSC_NULL, quadratic_2d_w0);CHKERRQ(ierr);
  ierr = CreateProblem_gen_1(paramDM, "w1", 0, PETSC_NULL, PETSC_NULL, quadratic_2d_w1);CHKERRQ(ierr);
  // Create the default parameter section
  ierr = MeshGetMesh((Mesh) paramDM, m);CHKERRQ(ierr);
  const ALE::Obj<ALE::Mesh::real_section_type> w = m->getRealSection("default");
  w->setDebug(options->debug);
  m->calculateIndices();
  m->setupField(w, 2);
  options->paramDM   = paramDM;
  options->paramName = "default";
  if (options->debug) {w->view("Default Stokes pressure potential");}
  // Create the Transport problem (assumes 2D)
  ierr = CreateProblem_gen_1(transportDM, "z0", 0, PETSC_NULL, PETSC_NULL, PETSC_NULL);CHKERRQ(ierr);
  ierr = CreateProblem_gen_1(transportDM, "z1", 0, PETSC_NULL, PETSC_NULL, PETSC_NULL);CHKERRQ(ierr);
  // Create the default Transport section
  ierr = MeshGetMesh((Mesh) transportDM, m);CHKERRQ(ierr);
  const ALE::Obj<ALE::Mesh::real_section_type> z = m->getRealSection("default");
  z->setDebug(options->debug);
  m->calculateIndices();
  m->setupField(z, 2);
  if (options->debug) {z->view("Default Transport field");}
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
  ierr = DMMGSetFromOptions(*dmmg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateExactSolution"
PetscErrorCode CreateExactSolution(DM dm, SectionReal *exactSol, Options *options)
{
  Mesh           mesh = (Mesh) dm;
  const int      dim  = options->dim;
  Obj<ALE::Mesh> m;
  Obj<ALE::Mesh::real_section_type> s;
  PetscTruth     flag;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = MeshGetSectionReal(mesh, "exactSolution", exactSol);CHKERRQ(ierr);
  ierr = SectionRealGetSection(*exactSol, s);CHKERRQ(ierr);
  m->setupField(s, 2);
  const Obj<ALE::Mesh::label_sequence>&     cells       = m->heightStratum(0);
  const Obj<ALE::Mesh::real_section_type>&  coordinates = m->getRealSection("coordinates");
  const int                                 localDof    = m->sizeWithBC(s, *cells->begin());
  ALE::Mesh::real_section_type::value_type *values      = new ALE::Mesh::real_section_type::value_type[localDof];
  double                                   *v0          = new double[dim];
  double                                   *J           = new double[dim*dim];
  double                                    detJ;
  const Obj<std::set<std::string> >&        discs       = m->getDiscretizations();
  const int                                 numFields   = discs->size();
  int                                      *v           = new int[numFields];

  for(ALE::Mesh::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
    const Obj<ALE::Mesh::coneArray>      closure = ALE::SieveAlg<ALE::Mesh>::closure(m, *c_iter);
    const ALE::Mesh::coneArray::iterator end     = closure->end();

    m->computeElementGeometry(coordinates, *c_iter, v0, J, PETSC_NULL, detJ);
    for(int f = 0; f < numFields; ++f) v[f] = 0;
    for(ALE::Mesh::coneArray::iterator cl_iter = closure->begin(); cl_iter != end; ++cl_iter) {
      int f = 0;

      for(std::set<std::string>::const_iterator f_iter = discs->begin(); f_iter != discs->end(); ++f_iter, ++f) {
        const Obj<ALE::Discretization>&    disc     = m->getDiscretization(*f_iter);
        const Obj<ALE::BoundaryCondition>& bc       = disc->getExactSolution();
        const int                          pointDim = disc->getNumDof(m->depth(*cl_iter));
        const int                         *indices  = disc->getIndices();

        for(int d = 0; d < pointDim; ++d, ++v[f]) {
          values[indices[v[f]]] = (*bc->getDualIntegrator())(v0, J, v[f], bc->getFunction());
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
  if (flag) {ierr = ViewSection(mesh, *exactSol, "exact_sol.vtk");CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CalculateError"
PetscErrorCode CalculateError(Mesh mesh, SectionReal X, double *error, void *ctx)
{
  Obj<ALE::Mesh> m;
  Obj<ALE::Mesh::real_section_type> sX;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = SectionRealGetSection(X, sX);CHKERRQ(ierr);
  const Obj<ALE::Mesh::real_section_type>& coordinates = m->getRealSection("coordinates");
  const Obj<ALE::Mesh::label_sequence>&    cells       = m->heightStratum(0);
  const int                                dim         = m->getDimension();
  const Obj<std::set<std::string> >&       discs       = m->getDiscretizations();
  double *coords, *v0, *J, *invJ, detJ;
  double  localError = 0.0;

  ierr = PetscMalloc4(dim,double,&coords,dim,double,&v0,dim*dim,double,&J,dim*dim,double,&invJ);CHKERRQ(ierr);
  // Loop over cells
  for(ALE::Mesh::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
    m->computeElementGeometry(coordinates, *c_iter, v0, J, invJ, detJ);
    const PetscScalar *x = m->restrictNew(sX, *c_iter);
    double elemError = 0.0;

    for(std::set<std::string>::const_iterator f_iter = discs->begin(); f_iter != discs->end(); ++f_iter) {
      const Obj<ALE::Discretization>&    disc          = m->getDiscretization(*f_iter);
      const Obj<ALE::BoundaryCondition>& bc            = disc->getExactSolution();
      const int                          numQuadPoints = disc->getQuadratureSize();
      const double                      *quadPoints    = disc->getQuadraturePoints();
      const double                      *quadWeights   = disc->getQuadratureWeights();
      const int                          numBasisFuncs = disc->getBasisSize();
      const double                      *basis         = disc->getBasis();
      const int                         *indices       = disc->getIndices();

      // Loop over quadrature points
      for(int q = 0; q < numQuadPoints; ++q) {
        for(int d = 0; d < dim; d++) {
          coords[d] = v0[d];
          for(int e = 0; e < dim; e++) {
            coords[d] += J[d*dim+e]*(quadPoints[q*dim+e] + 1.0);
          }
        }
        PetscScalar funcVal     = (*bc->getFunction())(coords);
        PetscScalar interpolant = 0.0;

        for(int f = 0; f < numBasisFuncs; ++f) {
          interpolant += x[indices[f]]*basis[q*numBasisFuncs+f];
        }
        elemError += (interpolant - funcVal)*(interpolant - funcVal)*quadWeights[q];
      }
    }    
    if (m->debug()) {
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
#define __FUNCT__ "CheckError"
PetscErrorCode CheckError(DM dm, SectionReal sol, Options *options)
{
  MPI_Comm       comm;
  const char    *name;
  PetscScalar    norm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  Mesh mesh = (Mesh) dm;

  ierr = CalculateError(mesh, sol, &norm, options);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject) sol, &name);CHKERRQ(ierr);
  PetscPrintf(comm, "Error for trial solution %s: %g\n", name, norm);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CheckStokesResidual"
PetscErrorCode CheckStokesResidual(DM dm, SectionReal sol, const std::string& paramName, Options *options)
{
  MPI_Comm       comm;
  const char    *name;
  PetscScalar    norm;
  PetscTruth     flag;
  std::string    oldParamName = options->paramName;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHasName(PETSC_NULL, "-vec_view", &flag);CHKERRQ(ierr);
  Mesh        mesh = (Mesh) dm;
  SectionReal residual;

  ierr = SectionRealDuplicate(sol, &residual);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) residual, "residual");CHKERRQ(ierr);
  options->paramName = paramName;
  ierr = Stokes_Rhs_Unstructured(mesh, sol, residual, options);CHKERRQ(ierr);
  options->paramName = oldParamName;
  if (flag) {ierr = SectionRealView(residual, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);}
  ierr = SectionRealNorm(residual, mesh, NORM_2, &norm);CHKERRQ(ierr);
  ierr = SectionRealDestroy(residual);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject) sol, &name);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject) sol, &comm);CHKERRQ(ierr);
  PetscPrintf(comm, "Residual for trial solution %s: %g\n", name, norm);
  PetscFunctionReturn(0);
}

/* ______________________________________________________________________ */
// Stoke_Rhs_Unstructured
/*!
  \param[out] mesh
  \param[out]  X
  \param[out] section
  \param[out] ctx

  <\nabla v, \nabla u^n> + r <\nabla\cdot v, \nabla\cdot u^n> = <v, f> - <\nabla\cdot v, \nabla\cdot w^n>

  \returns
    PetscErrorCode
*/
/* ______________________________________________________________________ */
#undef __FUNCT__
#define __FUNCT__ "Stokes_Rhs_Unstructured"
PetscErrorCode Stokes_Rhs_Unstructured(Mesh mesh, SectionReal X, SectionReal section, void *ctx)
{
  Options       *options   = (Options *) ctx;
  Mesh           paramMesh = (Mesh) options->paramDM;
  double      (**funcs)(const double *)    = options->funcs;
  double      (*viscosity)(const double *) = options->viscosity;
  Obj<ALE::Mesh> m;
  Obj<ALE::Mesh> pM;
  Obj<ALE::Mesh::real_section_type> sX;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = MeshGetMesh(paramMesh, pM);CHKERRQ(ierr);
  ierr = SectionRealGetSection(X, sX);CHKERRQ(ierr);
  const Obj<ALE::Mesh::real_section_type>&  sW            = pM->getRealSection(options->paramName);
  const Obj<ALE::Mesh::real_section_type>&  coordinates   = m->getRealSection("coordinates");
  const Obj<ALE::Mesh::label_sequence>&     cells         = m->heightStratum(0);
  const int                                 dim           = m->getDimension();
  const Obj<std::set<std::string> >&        discs         = m->getDiscretizations();
  const double                              r             = options->r;
  const int                                 debug         = options->debug;
  const int                                 localDof      = m->sizeWithBC(sW, *cells->begin());
  ALE::Mesh::real_section_type::value_type *values        = new ALE::Mesh::real_section_type::value_type[localDof];
  int                                       totBasisFuncs = 0;
  double      *t_der, *b_der, *coords, *v0, *J, *invJ, detJ;
  PetscScalar *elemVec, *elemMat, *div_elemMat;

  ierr = SectionRealZero(section);CHKERRQ(ierr);
  for(std::set<std::string>::const_iterator f_iter = discs->begin(); f_iter != discs->end(); ++f_iter) {
    totBasisFuncs += m->getDiscretization(*f_iter)->getBasisSize();
  }
  ierr = PetscMalloc3(totBasisFuncs,PetscScalar,&elemVec,totBasisFuncs*totBasisFuncs,PetscScalar,&elemMat,totBasisFuncs*totBasisFuncs,PetscScalar,&div_elemMat);CHKERRQ(ierr);
  ierr = PetscMalloc6(dim,double,&t_der,dim,double,&b_der,dim,double,&coords,dim,double,&v0,dim*dim,double,&J,dim*dim,double,&invJ);CHKERRQ(ierr);
  // Loop over cells
  for(ALE::Mesh::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
    if (debug) {std::cout << "Cell " << *c_iter << std::endl;}
    m->computeElementGeometry(coordinates, *c_iter, v0, J, invJ, detJ);
    const PetscScalar *x     = m->restrictNew(sX, *c_iter);
    const PetscScalar *w     = pM->restrictNew(sW, *c_iter, values, localDof);
    int                field = 0;

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

      ierr = PetscMemzero(elemMat,     numBasisFuncs*totBasisFuncs * sizeof(PetscScalar));CHKERRQ(ierr);
      ierr = PetscMemzero(div_elemMat, numBasisFuncs*totBasisFuncs * sizeof(PetscScalar));CHKERRQ(ierr);
      // Loop over quadrature points
      for(int q = 0; q < numQuadPoints; ++q) {
        for(int d = 0; d < dim; d++) {
          coords[d] = v0[d];
          for(int e = 0; e < dim; e++) {
            coords[d] += J[d*dim+e]*(quadPoints[q*dim+e] + 1.0);
          }
        }
        PetscScalar funcVal = (*funcs[field])(coords);
        PetscScalar visc    = (*viscosity)(coords);

        // Loop over trial functions
        for(int f = 0; f < numBasisFuncs; ++f) {
          // Constant part
          elemVec[indices[f]] -= basis[q*numBasisFuncs+f]*funcVal*quadWeights[q]*detJ;
          // Linear part
          // The div-div term
          PetscScalar tDiv = 0.0;

          for(int e = 0; e < dim; ++e) {
            tDiv += invJ[e*dim+field]*basisDer[(q*numBasisFuncs+f)*dim+e];
          }
          for(int g = 0; g < numBasisFuncs; ++g) {
            PetscScalar bDiv = 0.0;

            for(int d = 0; d < dim; ++d) {
              for(int e = 0; e < dim; ++e) {
                bDiv += invJ[e*dim+d]*basisDer[(q*numBasisFuncs+g)*dim+e];
              }
            }
            elemMat[f*totBasisFuncs+indices[g]]     += r*tDiv*visc*bDiv*quadWeights[q]*detJ;
            div_elemMat[f*totBasisFuncs+indices[g]] +=   tDiv*bDiv*quadWeights[q]*detJ;
          }
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

            for(int d = 0; d < dim; ++d) product += t_der[d]*visc*b_der[d];
            elemMat[f*totBasisFuncs+indices[g]] += product*quadWeights[q]*detJ;
          }
        }
      }
      if (debug) {
        std::cout << "Constant element vector for field " << *f_iter << ":" << std::endl;
        for(int f = 0; f < numBasisFuncs; ++f) {
          std::cout << "  " << elemVec[indices[f]] << std::endl;
        }
      }
      // Add linear contribution
      for(int f = 0; f < numBasisFuncs; ++f) {
        for(int g = 0; g < totBasisFuncs; ++g) {
          elemVec[indices[f]] += elemMat[f*totBasisFuncs+g]*x[g] + div_elemMat[f*totBasisFuncs+g]*w[g];
        }
      }
      if (debug) {
        ostringstream label; label << "Element Matrix for field " << *f_iter;
        std::cout << ALE::Mesh::printMatrix(label.str(), numBasisFuncs, totBasisFuncs, elemMat, m->commRank()) << std::endl;
        ostringstream label2; label2 << "Div-Element Matrix for field " << *f_iter;
        std::cout << ALE::Mesh::printMatrix(label2.str(), numBasisFuncs, totBasisFuncs, div_elemMat, m->commRank()) << std::endl;
        std::cout << "Linear element vector for field " << *f_iter << ":" << std::endl;
        for(int f = 0; f < numBasisFuncs; ++f) {
          std::cout << "  " << elemVec[indices[f]] << std::endl;
        }
      }
    }
    if (debug) {
      std::cout << "Element vector:" << std::endl;
      for(int f = 0; f < totBasisFuncs; ++f) {
        std::cout << "  " << elemVec[f] << std::endl;
      }
    }
    ierr = SectionRealUpdateAdd(section, *c_iter, elemVec);CHKERRQ(ierr);
    if (debug) {
      ierr = SectionRealView(section, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }
  }
  delete [] values;
  ierr = PetscFree3(elemVec,elemMat,div_elemMat);CHKERRQ(ierr);
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

  <\nabla v, \nabla u^n> + r <\nabla\cdot v, \nabla\cdot u^n> = <v, f> - <\nabla\cdot v, \nabla\cdot w^n>

  \returns
    PetscErrorCode
*/
/* ______________________________________________________________________ */
#undef __FUNCT__
#define __FUNCT__ "Stokes_Jac_Unstructured"
PetscErrorCode Stokes_Jac_Unstructured(Mesh mesh, SectionReal X, Mat A, void *ctx)
{
  Options       *options = (Options *) ctx;
  double       (*viscosity)(const double *) = options->viscosity;
  Obj<ALE::Mesh> m;
  Obj<ALE::Mesh::real_section_type> sX;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = SectionRealGetSection(X, sX);CHKERRQ(ierr);
  const Obj<ALE::Mesh::real_section_type>& coordinates   = m->getRealSection("coordinates");
  const Obj<ALE::Mesh::label_sequence>&    cells         = m->heightStratum(0);
  const Obj<ALE::Mesh::order_type>&        order         = m->getFactory()->getGlobalOrder(m, "default", sX);
  const int                                dim           = m->getDimension();
  const Obj<std::set<std::string> >&       discs         = m->getDiscretizations();
  const double                             r             = options->r;
  int                                      totBasisFuncs = 0;
  double      *t_der, *b_der, *coords, *v0, *J, *invJ, detJ;
  PetscScalar *elemMat;

  ierr = MatZeroEntries(A);CHKERRQ(ierr);
  for(std::set<std::string>::const_iterator f_iter = discs->begin(); f_iter != discs->end(); ++f_iter) {
    totBasisFuncs += m->getDiscretization(*f_iter)->getBasisSize();
  }
  ierr = PetscMalloc(totBasisFuncs*totBasisFuncs * sizeof(PetscScalar), &elemMat);CHKERRQ(ierr);
  ierr = PetscMalloc6(dim,double,&t_der,dim,double,&b_der,dim,double,&coords,dim,double,&v0,dim*dim,double,&J,dim*dim,double,&invJ);CHKERRQ(ierr);
  // Loop over cells
  for(ALE::Mesh::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
    int field = 0;

    m->computeElementGeometry(coordinates, *c_iter, v0, J, invJ, detJ);
    if (detJ < 0) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %d", detJ, *c_iter);
    ierr = PetscMemzero(elemMat, totBasisFuncs*totBasisFuncs * sizeof(PetscScalar));CHKERRQ(ierr);
    for(std::set<std::string>::const_iterator f_iter = discs->begin(); f_iter != discs->end(); ++f_iter, ++field) {
      const Obj<ALE::Discretization>& disc          = m->getDiscretization(*f_iter);
      const int                       numQuadPoints = disc->getQuadratureSize();
      const double                   *quadPoints    = disc->getQuadraturePoints();
      const double                   *quadWeights   = disc->getQuadratureWeights();
      const int                       numBasisFuncs = disc->getBasisSize();
      const double                   *basisDer      = disc->getBasisDerivatives();
      const int                      *indices       = disc->getIndices();

      // Loop over quadrature points
      for(int q = 0; q < numQuadPoints; ++q) {
        for(int d = 0; d < dim; d++) {
          coords[d] = v0[d];
          for(int e = 0; e < dim; e++) {
            coords[d] += J[d*dim+e]*(quadPoints[q*dim+e] + 1.0);
          }
        }
        PetscScalar visc = (*viscosity)(coords);

        // Loop over trial functions
        for(int f = 0; f < numBasisFuncs; ++f) {
          // The div-div term for u or v
          PetscScalar tDiv = 0.0;

          for(int e = 0; e < dim; ++e) {
            tDiv += invJ[e*dim+field]*basisDer[(q*numBasisFuncs+f)*dim+e];
          }
          for(int g = 0; g < numBasisFuncs; ++g) {
            PetscScalar bDiv = 0.0;

            for(int d = 0; d < dim; ++d) {
              for(int e = 0; e < dim; ++e) {
                bDiv += invJ[e*dim+d]*basisDer[(q*numBasisFuncs+g)*dim+e];
              }
            }
            elemMat[indices[f]*totBasisFuncs+indices[g]] += r*tDiv*visc*bDiv*quadWeights[q]*detJ;
          }
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
            
            for(int d = 0; d < dim; ++d) product += t_der[d]*visc*b_der[d];
            elemMat[indices[f]*totBasisFuncs+indices[g]] += product*quadWeights[q]*detJ;
          }
        }
      }
    }
    ierr = updateOperator(A, m, sX, order, *c_iter, elemMat, ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = PetscFree(elemMat);CHKERRQ(ierr);
  ierr = PetscFree6(t_der,b_der,coords,v0,J,invJ);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Transport_Rhs_Unstructured"
PetscErrorCode Transport_Rhs_Unstructured(Mesh mesh, SectionReal X, SectionReal section, void *ctx)
{
  //Options       *options = (Options *) ctx;
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
  //Options       *options = (Options *) ctx;
  Obj<ALE::Mesh::real_section_type> s;
  Obj<ALE::Mesh> m;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
// ---------------------------------------------------------------------------------------------------------------------
// End of file grade2.cxx
