static char help[] = "This example solves the Mixed Poisson problem.\n\n";

/*
     sigma + grad(u) = 0
          div(sigma) = f

 The corresponding weak (variational problem)

     <tau, sigma> - <div(tau), u> = 0       for all tau
                  <w, div(sigma)> = <w, f>  for all w
*/


// ---------------------------------------------------------------------------------------------------------------------
// Top level Includes and Data

#include <petscda.h>
#include <petscmesh.h>
#include <petscdmmg.h>

using ALE::Obj;
typedef enum {RUN_FULL, RUN_TEST, RUN_MESH} RunType;
typedef enum {NEUMANN, DIRICHLET} BCType;
typedef enum {SOLN_SIMPLE, SOLN_RIDGE} SolutionType;
typedef enum {ASSEMBLY_FULL, ASSEMBLY_STORED, ASSEMBLY_CALCULATED} AssemblyType;
typedef union {SectionReal section; Vec vec;} ExactSolType;

typedef struct {
  PetscInt      debug;                                        // The debugging level
  RunType       run;                                          // The run type
  SolutionType  solnType;                                     // The type of exact solution
  PetscInt      dim;                                          // The topological mesh dimension
  PetscTruth    generateMesh;                                 // Generate the unstructure mesh
  PetscTruth    interpolate;                                  // Generate intermediate mesh elements
  PetscReal     refinementLimit;                              // The largest allowable cell volume
  char          baseFilename[2048];                           // The base filename for mesh files
  double      (*funcs[4])(const double []);                   // The function to project
  BCType        bcType;                                       // The type of boundary conditions
  ExactSolType  exactSol;                                     // The discrete exact solution
  AssemblyType  operatorAssembly;                             // The type of operator assembly 
} Options;

#include "mixedpoisson_quadrature.h"

// ---------------------------------------------------------------------------------------------------------------------
// Some simple convenience functions

double zero(const double x[]) {
  return 0.0;
}

double constant(const double x[]) {
  return -3.0;
}

double quadratic_2d_u(const double x[]) {
  return x[0]*x[0] + x[1]*x[1];
}

double linear_2d_sigma_x(const double x[]) {
  return x[0];
}

double linear_2d_sigma_y(const double x[]) {
  return x[1];
}

double quadratic_3d_u(const double x[]) {
  return x[0]*x[0] + x[1]*x[1] + x[2]*x[2];
}

double linear_3d_sigma_x(const double x[]) {
  return x[0];
}

double linear_3d_sigma_y(const double x[]) {
  return x[1];
}

double linear_3d_sigma_z(const double x[]) {
  return x[2];
}

// ---------------------------------------------------------------------------------------------------------------------
// Function Definitions

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, Options *options)
{
  const char    *runTypes[3]  = {"full", "test", "mesh"};
  const char    *solnTypes[2] = {"simple", "ridge"};
  const char    *bcTypes[2]   = {"neumann", "dirichlet"};
  const char    *asTypes[4]   = {"full", "stored", "calculated"};
  ostringstream  filename;
  PetscInt       run, soln, bc, as;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->debug            = 0;
  options->run              = RUN_FULL;
  options->solnType         = SOLN_SIMPLE;
  options->dim              = 2;
  options->generateMesh     = PETSC_TRUE;
  options->interpolate      = PETSC_TRUE;
  options->refinementLimit  = 0.0;
  options->bcType           = DIRICHLET;
  options->operatorAssembly = ASSEMBLY_FULL;

  ierr = PetscOptionsBegin(comm, "", "Mixed Poisson Problem Options", "DMMG");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-debug", "The debugging level", "mixedpoisson.cxx", options->debug, &options->debug, PETSC_NULL);CHKERRQ(ierr);
  run = options->run;
  ierr = PetscOptionsEList("-run", "The run type", "mixedpoisson.cxx", runTypes, 3, runTypes[options->run], &run, PETSC_NULL);CHKERRQ(ierr);
  options->run = (RunType) run;
  ierr = PetscOptionsEList("-solution", "The solution type", "mixedpoisson.cxx", solnTypes, 2, solnTypes[options->solnType], &soln, PETSC_NULL);CHKERRQ(ierr);
  options->solnType = (SolutionType) soln;
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "mixedpoisson.cxx", options->dim, &options->dim, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTruth("-generate", "Generate the unstructured mesh", "mixedpoisson.cxx", options->generateMesh, &options->generateMesh, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTruth("-interpolate", "Generate intermediate mesh elements", "mixedpoisson.cxx", options->interpolate, &options->interpolate, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-refinement_limit", "The largest allowable cell volume", "mixedpoisson.cxx", options->refinementLimit, &options->refinementLimit, PETSC_NULL);CHKERRQ(ierr);
  filename << "data/mixedpoisson_" << options->dim <<"d";
  ierr = PetscStrcpy(options->baseFilename, filename.str().c_str());CHKERRQ(ierr);
  ierr = PetscOptionsString("-base_filename", "The base filename for mesh files", "mixedpoisson.cxx", options->baseFilename, options->baseFilename, 2048, PETSC_NULL);CHKERRQ(ierr);
  bc = options->bcType;
  ierr = PetscOptionsEList("-bc_type","Type of boundary condition","mixedpoisson.cxx",bcTypes,2,bcTypes[options->bcType],&bc,PETSC_NULL);CHKERRQ(ierr);
  options->bcType = (BCType) bc;
  as = options->operatorAssembly;
  ierr = PetscOptionsEList("-assembly_type","Type of operator assembly","mixedpoisson.cxx",asTypes,3,asTypes[options->operatorAssembly],&as,PETSC_NULL);CHKERRQ(ierr);
  options->operatorAssembly = (AssemblyType) as;
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  Mesh        mesh;
  PetscTruth  view;
  PetscMPIInt size;

  if (options->generateMesh) {
    Mesh boundary;

    ierr = MeshCreate(comm, &boundary);CHKERRQ(ierr);
    if (options->dim == 2) {
      double lower[2] = {0.0, 0.0};
      double upper[2] = {1.0, 1.0};
      int    edges[2] = {2, 2};

      Obj<ALE::Mesh> mB = ALE::MeshBuilder::createSquareBoundary(comm, lower, upper, edges, options->debug);
      ierr = MeshSetMesh(boundary, mB);CHKERRQ(ierr);
    } else if (options->dim == 3) {
      double lower[3] = {0.0, 0.0, 0.0};
      double upper[3] = {1.0, 1.0, 1.0};
      int    faces[3] = {1, 1, 1};

      Obj<ALE::Mesh> mB = ALE::MeshBuilder::createCubeBoundary(comm, lower, upper, faces, options->debug);
      ierr = MeshSetMesh(boundary, mB);CHKERRQ(ierr);
    } else {
      SETERRQ1(PETSC_ERR_SUP, "Dimension not supported: %d", options->dim);
    }
    ierr = MeshGenerate(boundary, options->interpolate, &mesh);CHKERRQ(ierr);
    ierr = MeshDestroy(boundary);CHKERRQ(ierr);
  } else {
    std::string baseFilename(options->baseFilename);
    std::string coordFile = baseFilename+".nodes";
    std::string adjFile   = baseFilename+".lcon";

    ierr = MeshCreatePCICE(comm, options->dim, coordFile.c_str(), adjFile.c_str(), options->interpolate, PETSC_NULL, &mesh);CHKERRQ(ierr);
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
  if (view) {ierr = ViewMesh(mesh, "mixedpoisson.vtk");CHKERRQ(ierr);}
  ierr = PetscOptionsHasName(PETSC_NULL, "-mesh_view", &view);CHKERRQ(ierr);
  if (view) {
    Obj<ALE::Mesh> m;
    ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
    m->view("Mesh");
  }
  *dm = (DM) mesh;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DestroyMesh"
PetscErrorCode DestroyMesh(DM dm, Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshDestroy((Mesh) dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DestroyExactSolution"
PetscErrorCode DestroyExactSolution(ExactSolType sol, Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SectionRealDestroy(sol.section);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Rhs_Unstructured"
PetscErrorCode Rhs_Unstructured(Mesh mesh, SectionReal X, SectionReal section, void *ctx)
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
  int         totBasisFuncs = 0;
  double      *coords, *v0, *J, *invJ, detJ;
  PetscScalar *elemVec, *elemMat;

  ierr = SectionRealZero(section);CHKERRQ(ierr);
  for(std::set<std::string>::const_iterator f_iter = discs->begin(); f_iter != discs->end(); ++f_iter) {
    totBasisFuncs += m->getDiscretization(*f_iter)->getBasisSize();
  }
  ierr = PetscMalloc2(totBasisFuncs,PetscScalar,&elemVec,totBasisFuncs*totBasisFuncs,PetscScalar,&elemMat);CHKERRQ(ierr);
  ierr = PetscMalloc4(dim,double,&coords,dim,double,&v0,dim*dim,double,&J,dim*dim,double,&invJ);CHKERRQ(ierr);
  // Loop over cells
  for (ALE::Mesh::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
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
          if (field == 0) {
            // < w, div(sigma_x) > 
            const Obj<ALE::Discretization>& sx         = m->getDiscretization("sigma_x");
            const int                       numSxFuncs = sx->getBasisSize();
            const double                   *sxBasisDer = sx->getBasisDerivatives();
            const int                      *sxIndices  = sx->getIndices();

            for(int g = 0; g < numSxFuncs; ++g) {
              PetscScalar sxDiv = 0.0;

              for(int e = 0; e < dim; ++e) sxDiv += invJ[e*dim+0]*sxBasisDer[(q*numSxFuncs+g)*dim+e];
              elemMat[f*totBasisFuncs+sxIndices[g]] += basis[q*numBasisFuncs+f]*sxDiv*quadWeights[q]*detJ;
            }
            // < w, div(sigma_y)>
            const Obj<ALE::Discretization>& sy         = m->getDiscretization("sigma_y");
            const int                       numSyFuncs = sy->getBasisSize();
            const double                   *syBasisDer = sy->getBasisDerivatives();
            const int                      *syIndices  = sy->getIndices();

            for(int g = 0; g < numSyFuncs; ++g) {
              PetscScalar syDiv = 0.0;

              for(int e = 0; e < dim; ++e) syDiv += invJ[e*dim+1]*syBasisDer[(q*numSyFuncs+g)*dim+e];
              elemMat[f*totBasisFuncs+syIndices[g]] += basis[q*numBasisFuncs+f]*syDiv*quadWeights[q]*detJ;
            }
          } else {
            // <sigma_x,tau_x> when field = 1
	    // <sigma_y,tau_y> when field = 2
            for(int g = 0; g < numBasisFuncs; ++g) {
              elemMat[f*totBasisFuncs+indices[g]] += basis[q*numBasisFuncs+f] * basis[q*numBasisFuncs+g]*quadWeights[q]*detJ;
            }
            // <div(tau_x),u> when field = 1
	    // <div(tau_y),u> when field = 2
            const Obj<ALE::Discretization>& u         = m->getDiscretization("u");
            const int                       numUFuncs = u->getBasisSize();
            const double                   *uBasisDer = u->getBasisDerivatives();
            const int                      *uIndices  = u->getIndices();

            for(int g = 0; g < numUFuncs; ++g) {
              PetscScalar tauDiv   = 0.0;
              const int   d        = field-1;

              for(int e = 0; e < dim; ++e) tauDiv -= invJ[e*dim+d]*uBasisDer[(q*numUFuncs+g)*dim+e];
              elemMat[f*totBasisFuncs+uIndices[g]] += basis[q*numBasisFuncs+f]*tauDiv*quadWeights[q]*detJ;
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
  ierr = PetscFree4(coords,v0,J,invJ);CHKERRQ(ierr);
  // Exchange neighbors
  ierr = SectionRealComplete(section);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CalculateError"
PetscErrorCode CalculateError(Mesh mesh, SectionReal X, double *error, void *ctx)
{
  Obj<ALE::Mesh> m;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  const Obj<ALE::Mesh::real_section_type>& coordinates = m->getRealSection("coordinates");
  const Obj<ALE::Mesh::label_sequence>&    cells       = m->heightStratum(0);
  const int                                dim         = m->getDimension();
  const Obj<std::set<std::string> >&       discs       = m->getDiscretizations();
  double *coords, *v0, *J, *invJ, detJ;
  double  localError = 0.0;

  ierr = PetscMalloc4(dim,double,&coords,dim,double,&v0,dim*dim,double,&J,dim*dim,double,&invJ);CHKERRQ(ierr);
  // Loop over cells
  for(ALE::Mesh::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
    PetscScalar *x;
    double       elemError = 0.0;

    m->computeElementGeometry(coordinates, *c_iter, v0, J, invJ, detJ);
    //ierr = SectionRealRestrict(X, *c_iter, &x);CHKERRQ(ierr);
    {
      Obj<ALE::Mesh::real_section_type> sX;

      ierr = SectionRealGetSection(X, sX);CHKERRQ(ierr);
      x = (PetscScalar *) m->restrictNew(sX, *c_iter);
    }
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
#define __FUNCT__ "Jac_Unstructured"
PetscErrorCode Jac_Unstructured(Mesh mesh, SectionReal section, Mat A, void *ctx)
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
  double       *v0, *J, *invJ, detJ;
  PetscScalar *elemMat;

  ierr = MatZeroEntries(A);CHKERRQ(ierr);
  for(std::set<std::string>::const_iterator f_iter = discs->begin(); f_iter != discs->end(); ++f_iter) {
    totBasisFuncs += m->getDiscretization(*f_iter)->getBasisSize();
  }
  ierr = PetscMalloc(totBasisFuncs*totBasisFuncs * sizeof(PetscScalar), &elemMat);CHKERRQ(ierr);
  ierr = PetscMalloc3(dim,double,&v0,dim*dim,double,&J,dim*dim,double,&invJ);CHKERRQ(ierr);
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
      const double                   *quadWeights   = disc->getQuadratureWeights();
      const int                       numBasisFuncs = disc->getBasisSize();
      const double                   *basis         = disc->getBasis();
      const int                      *indices       = disc->getIndices();

      // Loop over quadrature points
      for(int q = 0; q < numQuadPoints; ++q) {
        // Loop over trial functions
        for(int f = 0; f < numBasisFuncs; ++f) {
          //if (*f_iter == "pressure") {
          if (field == 0) {
            // < w, div(sigma_x) > 
            const Obj<ALE::Discretization>& sx         = m->getDiscretization("sigma_x");
            const int                       numSxFuncs = sx->getBasisSize();
            const double                   *sxBasisDer = sx->getBasisDerivatives();
            const int                      *sxIndices  = sx->getIndices();

            for(int g = 0; g < numSxFuncs; ++g) {
              PetscScalar sxDiv = 0.0;

              for(int e = 0; e < dim; ++e) sxDiv += invJ[e*dim+0]*sxBasisDer[(q*numSxFuncs+g)*dim+e];
              elemMat[f*totBasisFuncs+sxIndices[g]] += basis[q*numBasisFuncs+f]*sxDiv*quadWeights[q]*detJ;
            }
            // < w, div(sigma_y)>
            const Obj<ALE::Discretization>& sy         = m->getDiscretization("sigma_y");
            const int                       numSyFuncs = sy->getBasisSize();
            const double                   *syBasisDer = sy->getBasisDerivatives();
            const int                      *syIndices  = sy->getIndices();

            for(int g = 0; g < numSyFuncs; ++g) {
              PetscScalar syDiv = 0.0;

              for(int e = 0; e < dim; ++e) syDiv += invJ[e*dim+1]*syBasisDer[(q*numSyFuncs+g)*dim+e];
              elemMat[f*totBasisFuncs+syIndices[g]] += basis[q*numBasisFuncs+f]*syDiv*quadWeights[q]*detJ;
            }
	  } else {
           // <sigma_x,tau_x> when field = 1
	    // <sigma_y,tau_y> when field = 2
            for(int g = 0; g < numBasisFuncs; ++g) {
              elemMat[f*totBasisFuncs+indices[g]] += basis[q*numBasisFuncs+f] * basis[q*numBasisFuncs+g]*quadWeights[q]*detJ;
            }
            // <div(tau_x),u> when field = 1
	    // <div(tau_y),u> when field = 2
            const Obj<ALE::Discretization>& u         = m->getDiscretization("u");
            const int                       numUFuncs = u->getBasisSize();
            const double                   *uBasisDer = u->getBasisDerivatives();
            const int                      *uIndices  = u->getIndices();

            for(int g = 0; g < numUFuncs; ++g) {
              PetscScalar tauDiv   = 0.0;
              const int   d        = field-1;

              for(int e = 0; e < dim; ++e) tauDiv -= invJ[e*dim+d]*uBasisDer[(q*numUFuncs+g)*dim+e];
              elemMat[f*totBasisFuncs+uIndices[g]] += basis[q*numBasisFuncs+f]*tauDiv*quadWeights[q]*detJ;
            }
          }
        }
      }
    }
    ierr = updateOperator(A, m, s, order, *c_iter, elemMat, ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = PetscFree(elemMat);CHKERRQ(ierr);
  ierr = PetscFree3(v0,J,invJ);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateProblem"
PetscErrorCode CreateProblem(DM dm, Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (options->dim == 2) {
        options->funcs[0]  = constant;
        options->funcs[1]  = zero;
        options->funcs[2]  = zero;
  } else if (options->dim == 3) {
      options->funcs[0]  = constant;
      options->funcs[1]  = zero;
      options->funcs[2]  = zero;
      options->funcs[3]  = zero;
  } else {
    SETERRQ1(PETSC_ERR_SUP, "Dimension not supported: %d", options->dim);
  }
  Mesh mesh = (Mesh) dm;
  Obj<ALE::Mesh> m;

  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  if (options->dim == 1) {
    ierr = CreateProblem_gen_0(dm, "u", 0, PETSC_NULL, PETSC_NULL, quadratic_2d_u);CHKERRQ(ierr);
    ierr = CreateProblem_gen_1(dm, "sigma_x", 0, PETSC_NULL, PETSC_NULL, linear_2d_sigma_x);CHKERRQ(ierr);
  } else if (options->dim == 2) {
    ierr = CreateProblem_gen_2(dm, "u", 0, PETSC_NULL, PETSC_NULL, quadratic_2d_u);CHKERRQ(ierr);
    ierr = CreateProblem_gen_3(dm, "sigma_x", 0, PETSC_NULL, PETSC_NULL, linear_2d_sigma_x);CHKERRQ(ierr);
    ierr = CreateProblem_gen_3(dm, "sigma_y", 0, PETSC_NULL, PETSC_NULL, linear_2d_sigma_y);CHKERRQ(ierr);
  } else if (options->dim == 3) {
    ierr = CreateProblem_gen_4(dm, "u", 0, PETSC_NULL, PETSC_NULL, quadratic_3d_u);CHKERRQ(ierr);
    ierr = CreateProblem_gen_5(dm, "sigma_x", 0, PETSC_NULL, PETSC_NULL, linear_3d_sigma_x);CHKERRQ(ierr);
    ierr = CreateProblem_gen_5(dm, "sigma_y", 0, PETSC_NULL, PETSC_NULL, linear_3d_sigma_y);CHKERRQ(ierr);
    ierr = CreateProblem_gen_5(dm, "sigma_z", 0, PETSC_NULL, PETSC_NULL, linear_3d_sigma_z);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_ERR_SUP, "Dimension not supported: %d", options->dim);
  }
  const ALE::Obj<ALE::Mesh::real_section_type> s = m->getRealSection("default");
  s->setDebug(options->debug);
  m->calculateIndices();
  m->setupField(s, 2);
  if (options->debug) {s->view("Default field");}
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
  Mesh mesh = (Mesh) dm;

  Obj<ALE::Mesh> m;
  Obj<ALE::Mesh::real_section_type> s;

  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = MeshGetSectionReal(mesh, "exactSolution", &options->exactSol.section);CHKERRQ(ierr);
  ierr = SectionRealGetSection(options->exactSol.section, s);CHKERRQ(ierr);
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
  if (flag) {ierr = ViewSection(mesh, options->exactSol.section, "exact_sol.vtk");CHKERRQ(ierr);}
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
  Mesh mesh = (Mesh) dm;

  ierr = CalculateError(mesh, sol.section, &norm, options);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject) sol.section, &name);CHKERRQ(ierr);
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
  Mesh        mesh = (Mesh) dm;
  SectionReal residual;

  ierr = SectionRealDuplicate(sol.section, &residual);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) residual, "residual");CHKERRQ(ierr);
  ierr = Rhs_Unstructured(mesh, sol.section, residual, options);CHKERRQ(ierr);
  if (flag) {ierr = SectionRealView(residual, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);}
  ierr = SectionRealNorm(residual, mesh, NORM_2, &norm);CHKERRQ(ierr);
  ierr = SectionRealDestroy(residual);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject) sol.section, &name);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject) sol.section, &comm);CHKERRQ(ierr);
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
  ierr = DMMGSetSNESLocal(*dmmg, Rhs_Unstructured, Jac_Unstructured, 0, 0);CHKERRQ(ierr);
  ierr = DMMGSetFromOptions(*dmmg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Solve"
PetscErrorCode Solve(DMMG *dmmg, Options *options)
{
  Mesh                mesh = (Mesh) DMMGGetDM(dmmg);
  SNES                snes;
  MPI_Comm            comm;
  PetscInt            its;
  PetscTruth          flag;
  SNESConvergedReason reason;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
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
  if (flag && options->dim == 2) {ierr = VecView(DMMGGetx(dmmg), PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);}
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
  ierr = PetscOptionsHasName(PETSC_NULL, "-vec_view_fibrated", &flag);CHKERRQ(ierr);
  if (flag) {
    Obj<ALE::Mesh::real_section_type> u  = sol->getFibration(0);
    Obj<ALE::Mesh::real_section_type> sigmaX = sol->getFibration(1);
    Obj<ALE::Mesh::real_section_type> sigmaY = sol->getFibration(2);

    u->view("Solution");
    sigmaX->view("X-Sigma Solution");
    sigmaY->view("Y-Sigma Solution");
  }
  ierr = SectionRealDestroy(solution);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


// ---------------------------------------------------------------------------------------------------------------------
// Main Procedure

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
