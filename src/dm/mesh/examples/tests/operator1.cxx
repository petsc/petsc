static char help[] = "Operator Tests.\n\n";

#include <petscmesh_viewers.hh>

#include "../tutorials/bratu_quadrature.h"

using ALE::Obj;
extern PetscLogEvent MAT_Mult;
typedef ALE::MinimalArrow<ALE::Mesh::point_type, ALE::Mesh::point_type> arrow_type;
typedef ALE::UniformSection<arrow_type, double>                         matrix_section_type;

typedef struct {
  int        debug;           // The debugging level
  int        dim;             // The topological mesh dimension
  PetscTruth interpolate;     // Construct missing elements of the mesh
  PetscReal  refinementLimit; // The largest allowable cell volume
  PetscLogEvent assemblyEvent;
} Options;

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

  ierr = PetscOptionsBegin(comm, "", "Options for mesh test", "Mesh");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-debug", "The debugging level",            "operator1.cxx", options->debug, &options->debug, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-dim",   "The topological mesh dimension", "operator1.cxx", options->dim, &options->dim,   PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-interpolate", "Construct missing elements of the mesh", "operator1.cxx", options->interpolate, &options->interpolate, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-refinement_limit", "The largest allowable cell volume", "operator1.cxx", options->refinementLimit, &options->refinementLimit, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  ierr = PetscLogEventRegister("Assembly", MAT_COOKIE,&options->assemblyEvent);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreatePartition"
// Creates a field whose value is the processor rank on each element
PetscErrorCode CreatePartition(const Obj<ALE::Mesh>& m, const Obj<ALE::Mesh::int_section_type>& s)
{
  PetscFunctionBegin;
  const Obj<ALE::Mesh::label_sequence>&         cells = m->heightStratum(0);
  const ALE::Mesh::int_section_type::value_type rank  = s->commRank();

  s->setFiberDimension(m->heightStratum(0), 1);
  m->allocate(s);
  for(ALE::Mesh::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
    s->updatePoint(*c_iter, &rank);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ViewMesh"
PetscErrorCode ViewMesh(const Obj<ALE::Mesh>& m, const char filename[])
{
  const Obj<ALE::Mesh::int_section_type>& partition = m->getIntSection("partition");
  PetscViewer                        viewer;
  PetscErrorCode                     ierr;

  PetscFunctionBegin;
  ierr = PetscViewerCreate(m->comm(), &viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(viewer, PETSC_VIEWER_ASCII);CHKERRQ(ierr);
  ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(viewer, filename);CHKERRQ(ierr);
  ierr = VTKViewer::writeHeader(viewer);CHKERRQ(ierr);
  ierr = VTKViewer::writeVertices(m, viewer);CHKERRQ(ierr);
  ierr = VTKViewer::writeElements(m, viewer);CHKERRQ(ierr);
  ierr = CreatePartition(m, partition);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_VTK_CELL);CHKERRQ(ierr);
  ierr = VTKViewer::writeField(partition, partition->getName(), 1, m->getFactory()->getNumbering(m, m->depth()), viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateMesh"
PetscErrorCode CreateMesh(MPI_Comm comm, Obj<ALE::Mesh>& m, Options *options)
{
  Obj<ALE::Mesh> mB;
  PetscTruth     view;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  ierr = PetscOptionsHasName(PETSC_NULL, "-mesh_view", &view);CHKERRQ(ierr);
  if (options->dim == 2) {
    double lower[2] = {0.0, 0.0};
    double upper[2] = {1.0, 1.0};
    int    edges[2] = {2, 2};

    mB = ALE::MeshBuilder::createSquareBoundary(comm, lower, upper, edges, options->debug);
  } else if (options->dim == 3) {
    double lower[3] = {0.0, 0.0, 0.0};
    double upper[3] = {1.0, 1.0, 1.0};
    int    faces[3] = {1, 1, 1};

    mB = ALE::MeshBuilder::createCubeBoundary(comm, lower, upper, faces, options->debug);
  } else {
    SETERRQ1(PETSC_ERR_SUP, "Dimension not supported: %d", options->dim);
  }
  if (view) {mB->view("Boundary");}
  m = ALE::Generator::generateMesh(mB, options->interpolate);
  if (view) {m->view("Mesh");}
  if (options->refinementLimit > 0.0) {
    Obj<ALE::Mesh> refMesh = ALE::Generator::refineMesh(m, options->refinementLimit, options->interpolate);
    if (view) {refMesh->view("Refined Mesh");}
    m = refMesh;
  } else if (m->commSize() > 1) {
    Obj<ALE::Mesh> newMesh = ALE::Distribution<ALE::Mesh>::distributeMesh(m);
    if (view) {newMesh->view("Parallel Mesh");}
    m = newMesh;
  }
  ierr = PetscOptionsHasName(PETSC_NULL, "-mesh_view_vtk", &view);CHKERRQ(ierr);
  if (view) {ierr = ViewMesh(m, "mesh.vtk");CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateProblem"
PetscErrorCode CreateProblem(const Obj<ALE::Mesh>& m, Options *options)
{
  Mesh           mesh;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshCreate(m->comm(), &mesh);CHKERRQ(ierr);
  ierr = MeshSetMesh(mesh, m);CHKERRQ(ierr);
  if (options->dim == 1) {
    ierr = CreateProblem_gen_0((DM) mesh, "u", 0, PETSC_NULL, PETSC_NULL, PETSC_NULL);CHKERRQ(ierr);
  } else if (options->dim == 2) {
    ierr = CreateProblem_gen_1((DM) mesh, "u", 0, PETSC_NULL, PETSC_NULL, PETSC_NULL);CHKERRQ(ierr);
  } else if (options->dim == 3) {
    ierr = CreateProblem_gen_2((DM) mesh, "u", 0, PETSC_NULL, PETSC_NULL, PETSC_NULL);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_ERR_SUP, "Dimension not supported: %d", options->dim);
  }
  ierr = MeshDestroy(mesh);CHKERRQ(ierr);
  const ALE::Obj<ALE::Mesh::real_section_type> s = m->getRealSection("default");
  s->setDebug(options->debug);
  m->setupField(s);
  if (options->debug) {s->view("Default field");}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateMatrix"
PetscErrorCode CreateMatrix(Mesh mesh, MatType mtype, Mat *A)
{
  Obj<ALE::Mesh> m;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  if (!m->hasRealSection("default")) SETERRQ(PETSC_ERR_ARG_WRONGSTATE, "Must set default section");
  ierr = MeshCreateMatrix(m, m->getRealSection("default"), mtype, A);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject) *A, "mesh", (PetscObject) mesh);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "AssembleMatrix_Full"
PetscErrorCode AssembleMatrix_Full(const Obj<ALE::Mesh>& m, const Obj<ALE::Mesh::real_section_type>& s, Mat A, void *ctx)
{
  Options       *options = (Options *) ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatZeroEntries(A);CHKERRQ(ierr);
  const Obj<ALE::Discretization>&          disc          = m->getDiscretization("u");
  const int                                numQuadPoints = disc->getQuadratureSize();
  const double                            *quadWeights   = disc->getQuadratureWeights();
  const int                                numBasisFuncs = disc->getBasisSize();
  const double                            *basisDer      = disc->getBasisDerivatives();
  const Obj<ALE::Mesh::real_section_type>& coordinates   = m->getRealSection("coordinates");
  const Obj<ALE::Mesh::label_sequence>&    cells         = m->heightStratum(0);
  const Obj<ALE::Mesh::order_type>&        order         = m->getFactory()->getGlobalOrder(m, "default", s);
  const int                                dim           = m->getDimension();
  double      *t_der, *b_der, *v0, *J, *invJ, detJ;
  PetscScalar *elemMat;

  ierr = PetscLogEventBegin(options->assemblyEvent,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscMalloc(numBasisFuncs*numBasisFuncs * sizeof(PetscScalar), &elemMat);CHKERRQ(ierr);
  ierr = PetscMalloc5(dim,double,&t_der,dim,double,&b_der,dim,double,&v0,dim*dim,double,&J,dim*dim,double,&invJ);CHKERRQ(ierr);
  // Loop over cells
  const ALE::Mesh::label_sequence::iterator cBegin = cells->begin();
  const ALE::Mesh::label_sequence::iterator cEnd   = cells->end();

  for(ALE::Mesh::label_sequence::iterator c_iter = cBegin; c_iter != cEnd; ++c_iter) {
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
    ierr = updateOperator(A, m, s, order, *c_iter, elemMat, ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = PetscLogFlops(((dim*dim*2.0 + dim*2.0 + 3.0)*numBasisFuncs + dim*dim*2.0)*numBasisFuncs*numQuadPoints*cells->size());CHKERRQ(ierr);
  ierr = PetscFree(elemMat);CHKERRQ(ierr);
  ierr = PetscFree5(t_der,b_der,v0,J,invJ);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(options->assemblyEvent,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FullAssemblyTest"
PetscErrorCode FullAssemblyTest(const Obj<ALE::Mesh>& m, Options *options)
{
  Mesh           mesh;
  Mat            A;
  Vec            x, y;
  PetscLogStage  stage;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogStageRegister("Full Assembly",&stage);CHKERRQ(ierr);
  ierr = PetscLogStagePush(stage);CHKERRQ(ierr);
  ierr = MeshCreate(m->comm(), &mesh);CHKERRQ(ierr);
  ierr = MeshSetMesh(mesh, m);CHKERRQ(ierr);
  ierr = CreateMatrix(mesh, MATAIJ, &A);CHKERRQ(ierr);
  ierr = MeshCreateGlobalVector(mesh, &x);CHKERRQ(ierr);
  ierr = VecSet(x, 1.0);CHKERRQ(ierr);
  ierr = VecDuplicate(x, &y);CHKERRQ(ierr);
  ierr = AssembleMatrix_Full(m, m->getRealSection("default"), A, options);CHKERRQ(ierr);
  ierr = MatMult(A, x, y);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(y);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(y);CHKERRQ(ierr);
  ierr = VecDestroy(x);CHKERRQ(ierr);
  ierr = VecDestroy(y);CHKERRQ(ierr);
  ierr = MatDestroy(A);CHKERRQ(ierr);
  ierr = MeshDestroy(mesh);CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "AssembleMatrix_None"
PetscErrorCode AssembleMatrix_None(const Obj<ALE::Mesh>& m, const Obj<ALE::Mesh::real_section_type>& s, const Obj<ALE::Mesh::real_section_type>& t, void *ctx)
{
  Options       *options = (Options *) ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  t->zero();
  const Obj<ALE::Discretization>&          disc          = m->getDiscretization("u");
  const int                                numQuadPoints = disc->getQuadratureSize();
  const double                            *quadWeights   = disc->getQuadratureWeights();
  const int                                numBasisFuncs = disc->getBasisSize();
  const double                            *basisDer      = disc->getBasisDerivatives();
  const Obj<ALE::Mesh::real_section_type>& coordinates   = m->getRealSection("coordinates");
  const Obj<ALE::Mesh::label_sequence>&    cells         = m->heightStratum(0);
  const int                                dim           = m->getDimension();
  double      *t_der, *b_der, *v0, *J, *invJ, detJ;
  PetscScalar *elemVec, *elemMat;

  ierr = PetscLogEventBegin(options->assemblyEvent,0,0,0,0);CHKERRQ(ierr);
  ierr = PetscMalloc2(numBasisFuncs,PetscScalar,&elemVec,numBasisFuncs*numBasisFuncs,PetscScalar,&elemMat);CHKERRQ(ierr);
  ierr = PetscMalloc5(dim,double,&t_der,dim,double,&b_der,dim,double,&v0,dim*dim,double,&J,dim*dim,double,&invJ);CHKERRQ(ierr);
  // Loop over cells
  const ALE::Mesh::label_sequence::iterator cBegin = cells->begin();
  const ALE::Mesh::label_sequence::iterator cEnd   = cells->end();

  for(ALE::Mesh::label_sequence::iterator c_iter = cBegin; c_iter != cEnd; ++c_iter) {
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
    const ALE::Mesh::real_section_type::value_type *ev = m->restrictClosure(s, *c_iter);

    // Do local matvec
    for(int f = 0; f < numBasisFuncs; ++f) {
      elemVec[f] = 0.0;
      for(int g = 0; g < numBasisFuncs; ++g) {
        elemVec[f] += elemMat[f*numBasisFuncs+g]*ev[g];
      }
    }
    m->updateAdd(t, *c_iter, elemVec);
  }
  ierr = PetscLogFlops(((dim*dim*2.0 + dim*2.0 + 3.0 + 2.0)*numBasisFuncs + dim*dim*2.0)*numBasisFuncs*numQuadPoints*cells->size());CHKERRQ(ierr);
  ierr = PetscFree2(elemVec,elemMat);CHKERRQ(ierr);
  ierr = PetscFree5(t_der,b_der,v0,J,invJ);CHKERRQ(ierr);
  ALE::Distribution<ALE::Mesh>::completeSection(m, t);
  ierr = PetscLogEventEnd(options->assemblyEvent,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NoAssemblyTest"
PetscErrorCode NoAssemblyTest(const Obj<ALE::Mesh>& m, Options *options)
{
  const Obj<ALE::Mesh::real_section_type>& s = m->getRealSection("default");
  Obj<ALE::Mesh::real_section_type>        t = new ALE::Mesh::real_section_type(s->comm(), s->debug());
  PetscLogStage  stage;
  PetscTruth     view;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogStageRegister("No Assembly",&stage);CHKERRQ(ierr);
  ierr = PetscLogStagePush(stage);CHKERRQ(ierr);
  t->setAtlas(s->getAtlas());
  t->allocateStorage();
  t->copyBC(s);
  s->set(1.0);
  ierr = AssembleMatrix_None(m, s, t, options);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL, "-vec_view", &view);CHKERRQ(ierr);
  if (view) {t->view("");}
  ierr = PetscLogStagePop();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "AssembleMatrix_Stored"
PetscErrorCode AssembleMatrix_Stored(const Obj<ALE::Mesh>& m, const Obj<ALE::Mesh::real_section_type>& s, const Obj<ALE::Mesh::real_section_type>& o, void *ctx)
{
  Options       *options = (Options *) ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  const Obj<ALE::Discretization>&          disc          = m->getDiscretization("u");
  const int                                numQuadPoints = disc->getQuadratureSize();
  const double                            *quadWeights   = disc->getQuadratureWeights();
  const int                                numBasisFuncs = disc->getBasisSize();
  const double                            *basisDer      = disc->getBasisDerivatives();
  const Obj<ALE::Mesh::real_section_type>& coordinates   = m->getRealSection("coordinates");
  const Obj<ALE::Mesh::label_sequence>&    cells         = m->heightStratum(0);
  const int                                dim           = m->getDimension();
  double      *t_der, *b_der, *v0, *J, *invJ, detJ;
  PetscScalar *elemMat;

  ierr = PetscLogEventBegin(options->assemblyEvent,0,0,0,0);CHKERRQ(ierr);
  o->setFiberDimension(cells, numBasisFuncs*numBasisFuncs);
  o->allocatePoint();
  ierr = PetscMalloc(numBasisFuncs*numBasisFuncs * sizeof(PetscScalar), &elemMat);CHKERRQ(ierr);
  ierr = PetscMalloc5(dim,double,&t_der,dim,double,&b_der,dim,double,&v0,dim*dim,double,&J,dim*dim,double,&invJ);CHKERRQ(ierr);
  // Loop over cells
  const ALE::Mesh::label_sequence::iterator cBegin = cells->begin();
  const ALE::Mesh::label_sequence::iterator cEnd   = cells->end();

  for(ALE::Mesh::label_sequence::iterator c_iter = cBegin; c_iter != cEnd; ++c_iter) {
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
    o->updatePoint(*c_iter, elemMat);
  }
  ierr = PetscLogFlops(((dim*dim*2.0 + dim*2.0 + 3.0)*numBasisFuncs + dim*dim*2.0)*numBasisFuncs*numQuadPoints*cells->size());CHKERRQ(ierr);
  ierr = PetscFree(elemMat);CHKERRQ(ierr);
  ierr = PetscFree5(t_der,b_der,v0,J,invJ);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(options->assemblyEvent,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ApplyMatrix_Stored"
PetscErrorCode ApplyMatrix_Stored(const Obj<ALE::Mesh>& m, const Obj<ALE::Mesh::real_section_type>& o, const Obj<ALE::Mesh::real_section_type>& s, const Obj<ALE::Mesh::real_section_type>& t, void *ctx)
{
  const Obj<ALE::Mesh::label_sequence>& cells         = m->heightStratum(0);
  int                                   numBasisFuncs = m->getDiscretization("u")->getBasisSize();
  PetscScalar                          *elemVec;
  PetscErrorCode                        ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(MAT_Mult,0,0,0,0);CHKERRQ(ierr);
  t->zero();
  ierr = PetscMalloc(numBasisFuncs *sizeof(PetscScalar), &elemVec);CHKERRQ(ierr);
  // Loop over cells
  const ALE::Mesh::label_sequence::iterator cBegin = cells->begin();
  const ALE::Mesh::label_sequence::iterator cEnd   = cells->end();

  for(ALE::Mesh::label_sequence::iterator c_iter = cBegin; c_iter != cEnd; ++c_iter) {
    const ALE::Mesh::real_section_type::value_type *elemMat = o->restrictPoint(*c_iter);
    const ALE::Mesh::real_section_type::value_type *ev      = m->restrictClosure(s, *c_iter);

    // Do local matvec
    for(int f = 0; f < numBasisFuncs; ++f) {
      elemVec[f] = 0.0;
      for(int g = 0; g < numBasisFuncs; ++g) {
        elemVec[f] += elemMat[f*numBasisFuncs+g]*ev[g];
      }
    }
    m->updateAdd(t, *c_iter, elemVec);
  }
  ierr = PetscLogFlops((numBasisFuncs*numBasisFuncs*2.0)*cells->size());CHKERRQ(ierr);
  ierr = PetscFree(elemVec);CHKERRQ(ierr);
  ALE::Distribution<ALE::Mesh>::completeSection(m, t);
  ierr = PetscLogEventEnd(MAT_Mult,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "StoredAssemblyTest"
PetscErrorCode StoredAssemblyTest(const Obj<ALE::Mesh>& m, Options *options)
{
  const Obj<ALE::Mesh::real_section_type>& s = m->getRealSection("default");
  const Obj<ALE::Mesh::real_section_type>& o = m->getRealSection("operator");
  Obj<ALE::Mesh::real_section_type>        t = new ALE::Mesh::real_section_type(s->comm(), s->debug());
  PetscLogStage  stage;
  PetscTruth     view;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogStageRegister("Stored Assembly",&stage);CHKERRQ(ierr);
  ierr = PetscLogStagePush(stage);CHKERRQ(ierr);
  t->setAtlas(s->getAtlas());
  t->allocateStorage();
  t->copyBC(s);
  s->set(1.0);
  ierr = AssembleMatrix_Stored(m, s, o, options);CHKERRQ(ierr);
  ierr = ApplyMatrix_Stored(m, o, s, t, options);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL, "-vec_view", &view);CHKERRQ(ierr);
  if (view) {t->view("");}
  ierr = PetscLogStagePop();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "AssembleMatrix_Partial"
PetscErrorCode AssembleMatrix_Partial(const Obj<ALE::Mesh>& m, const Obj<ALE::Mesh::real_section_type>& s, const Obj<matrix_section_type>& o, void *ctx)
{
  Options       *options = (Options *) ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  const Obj<ALE::Mesh::sieve_type>&        sieve         = m->getSieve();
  const Obj<ALE::Discretization>&          disc          = m->getDiscretization("u");
  const int                                numQuadPoints = disc->getQuadratureSize();
  const double                            *quadWeights   = disc->getQuadratureWeights();
  const int                                numBasisFuncs = disc->getBasisSize();
  const double                            *basisDer      = disc->getBasisDerivatives();
  const Obj<ALE::Mesh::real_section_type>& coordinates   = m->getRealSection("coordinates");
  const int                                dim           = m->getDimension();
  double      *t_der, *b_der, *v0, *J, *invJ, detJ;
  PetscScalar *elemMat;

  ierr = PetscLogEventBegin(options->assemblyEvent,0,0,0,0);CHKERRQ(ierr);
  // Preallocate
  const Obj<ALE::Mesh::label_sequence>&     vertices = m->depthStratum(0);
  const ALE::Mesh::label_sequence::iterator vBegin   = vertices->begin();
  const ALE::Mesh::label_sequence::iterator vEnd     = vertices->end();

  for(ALE::Mesh::label_sequence::iterator v_iter = vBegin; v_iter != vEnd; ++v_iter) {
    const Obj<ALE::Mesh::sieve_type::coneSet>&     adj  = sieve->cone(sieve->support(*v_iter));
    const ALE::Mesh::sieve_type::coneSet::iterator end  = adj->end();

    for(ALE::Mesh::sieve_type::coneSet::iterator n_iter = adj->begin(); n_iter != end; ++n_iter) {
      o->setFiberDimension(arrow_type(*v_iter, *n_iter), 1);
    }
  }
  ierr = PetscMalloc(numBasisFuncs*numBasisFuncs * sizeof(PetscScalar), &elemMat);CHKERRQ(ierr);
  ierr = PetscMalloc5(dim,double,&t_der,dim,double,&b_der,dim,double,&v0,dim*dim,double,&J,dim*dim,double,&invJ);CHKERRQ(ierr);
  // Loop over cells
  const Obj<ALE::Mesh::label_sequence>&     cells  = m->heightStratum(0);
  const ALE::Mesh::label_sequence::iterator cBegin = cells->begin();
  const ALE::Mesh::label_sequence::iterator cEnd   = cells->end();

  for(ALE::Mesh::label_sequence::iterator c_iter = cBegin; c_iter != cEnd; ++c_iter) {
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
    // Update each vertex
    const Obj<ALE::Mesh::sieve_type::traits::coneSequence>&     cone   = sieve->cone(*c_iter);
    const ALE::Mesh::sieve_type::traits::coneSequence::iterator nBegin = cone->begin();
    const ALE::Mesh::sieve_type::traits::coneSequence::iterator nEnd   = cone->end();
    int                                                         n      = 0;

    for(ALE::Mesh::sieve_type::traits::coneSequence::iterator n_iter = nBegin; n_iter != nEnd; ++n_iter, ++n) {
      int m = 0;
      for(ALE::Mesh::sieve_type::traits::coneSequence::iterator m_iter = nBegin; m_iter != nEnd; ++m_iter, ++m) {
        o->updateAddPoint(arrow_type(*n_iter, *m_iter), &elemMat[n*numBasisFuncs+m]);
      }
    }
  }
  if (m->debug()) {o->view("Operator");}
  ierr = PetscLogFlops(((dim*dim*2.0 + dim*2.0 + 3.0)*numBasisFuncs + dim*dim*2.0)*numBasisFuncs*numQuadPoints*cells->size());CHKERRQ(ierr);
  ierr = PetscFree(elemMat);CHKERRQ(ierr);
  ierr = PetscFree5(t_der,b_der,v0,J,invJ);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(options->assemblyEvent,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ApplyMatrix_Partial"
PetscErrorCode ApplyMatrix_Partial(const Obj<ALE::Mesh>& m, const Obj<matrix_section_type>& o, const Obj<ALE::Mesh::real_section_type>& s, const Obj<ALE::Mesh::real_section_type>& t, void *ctx)
{
  const Obj<ALE::Mesh::sieve_type>&     sieve    = m->getSieve();
  const Obj<ALE::Mesh::label_sequence>& vertices = m->depthStratum(0);
  PetscErrorCode                        ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(MAT_Mult,0,0,0,0);CHKERRQ(ierr);
  t->zero();
  // Loop over vertices
  const ALE::Mesh::label_sequence::iterator vBegin = vertices->begin();
  const ALE::Mesh::label_sequence::iterator vEnd   = vertices->end();

  for(ALE::Mesh::label_sequence::iterator v_iter = vBegin; v_iter != vEnd; ++v_iter) {
    const Obj<ALE::Mesh::sieve_type::coneSet>&     adj  = sieve->cone(sieve->support(*v_iter));
    const ALE::Mesh::sieve_type::coneSet::iterator end  = adj->end();
    ALE::Mesh::real_section_type::value_type       sum  = 0.0;

    // Do local dot product
    for(ALE::Mesh::sieve_type::coneSet::iterator n_iter = adj->begin(); n_iter != end; ++n_iter) {
      sum += o->restrictPoint(arrow_type(*v_iter, *n_iter))[0]*s->restrictPoint(*n_iter)[0];
    }
    t->updatePoint(*v_iter, &sum);
    ierr = PetscLogFlops(adj->size()*2.0);CHKERRQ(ierr);
  }
  ALE::Distribution<ALE::Mesh>::completeSection(m, t);
  ierr = PetscLogEventEnd(MAT_Mult,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PartialAssemblyTest"
PetscErrorCode PartialAssemblyTest(const Obj<ALE::Mesh>& m, Options *options)
{
  const Obj<ALE::Mesh::real_section_type>& s = m->getRealSection("default");
  const Obj<matrix_section_type>&          o = new matrix_section_type(s->comm(), s->debug());
  Obj<ALE::Mesh::real_section_type>        t = new ALE::Mesh::real_section_type(s->comm(), s->debug());
  PetscLogStage  stage;
  PetscTruth     view;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogStageRegister("Partial Assembly",&stage);CHKERRQ(ierr);
  ierr = PetscLogStagePush(stage);CHKERRQ(ierr);
  t->setAtlas(s->getAtlas());
  t->allocateStorage();
  t->copyBC(s);
  s->set(1.0);
  ierr = AssembleMatrix_Partial(m, s, o, options);CHKERRQ(ierr);
  ierr = ApplyMatrix_Partial(m, o, s, t, options);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL, "-vec_view", &view);CHKERRQ(ierr);
  if (view) {t->view("");}
  ierr = PetscLogStagePop();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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
  ierr = ProcessOptions(comm, &options);CHKERRQ(ierr);
  try {
    Obj<ALE::Mesh> m;

    ierr = CreateMesh(comm, m, &options);CHKERRQ(ierr);
    ierr = CreateProblem(m, &options);CHKERRQ(ierr);
    ierr = FullAssemblyTest(m, &options);CHKERRQ(ierr);
    ierr = NoAssemblyTest(m, &options);CHKERRQ(ierr);
    ierr = StoredAssemblyTest(m, &options);CHKERRQ(ierr);
    ierr = PartialAssemblyTest(m, &options);CHKERRQ(ierr);
  } catch (ALE::Exception e) {
    std::cout << e << std::endl;
  }
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
