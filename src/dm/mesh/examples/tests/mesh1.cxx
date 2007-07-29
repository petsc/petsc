static char help[] = "Mesh Tests.\n\n";

#include "petscmesh_formats.hh"
#include "meshTest.hh"

#include <LabelSifter.hh>

using ALE::Obj;
typedef ALE::Mesh::sieve_type        sieve_type;
typedef ALE::Mesh::real_section_type section_type;
typedef ALE::Mesh::label_type        label_type;
typedef ALE::LabelSifter<int,sieve_type::point_type> new_label_type;

typedef struct {
  int        debug;           // The debugging level
  int        test;            // The testing level
  int        dim;             // The topological mesh dimension
  PetscTruth interpolate;     // Construct missing elements of the mesh
  PetscReal  refinementLimit; // The largest allowable cell volume
} Options;

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->debug           = 0;
  options->test            = 0;
  options->dim             = 2;
  options->interpolate     = PETSC_TRUE;
  options->refinementLimit = 0.0;

  ierr = PetscOptionsBegin(comm, "", "Options for Mesh test", "Mesh");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-debug", "The debugging level", "mesh1.cxx", options->debug, &options->debug, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-test", "The testing level", "mesh1.cxx", options->test, &options->test, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-dim",   "The topological mesh dimension", "mesh1.cxx", options->dim, &options->dim, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-interpolate", "Construct missing elements of the mesh", "mesh1.cxx", options->interpolate, &options->interpolate, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-refinement_limit", "The largest allowable cell volume", "mesh1.cxx", options->refinementLimit, &options->refinementLimit, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

PetscErrorCode PrintMatrix(MPI_Comm comm, int rank, const std::string& name, const int rows, const int cols, const section_type::value_type matrix[])
{
  PetscFunctionBegin;
  PetscSynchronizedPrintf(comm, "%s", ALE::Mesh::printMatrix(name, rows, cols, matrix, rank).c_str());
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "GeometryTest"
PetscErrorCode GeometryTest(const Obj<ALE::Mesh>& mesh, const Obj<section_type>& coordinates, Options *options)
{
  const Obj<ALE::Mesh::label_sequence>&     cells  = mesh->heightStratum(0);
  const ALE::Mesh::label_sequence::iterator cBegin = cells->begin();
  const ALE::Mesh::label_sequence::iterator cEnd   = cells->end();
  const int                                 dim    = mesh->getDimension();
  const MPI_Comm                            comm   = mesh->comm();
  const int                                 rank   = mesh->commRank();
  const int                                 debug  = mesh->debug();
  section_type::value_type *v0   = new section_type::value_type[dim];
  section_type::value_type *J    = new section_type::value_type[dim*dim];
  section_type::value_type *invJ = new section_type::value_type[dim*dim];
  section_type::value_type  detJ;
  PetscErrorCode            ierr;

  PetscFunctionBegin;
  for(ALE::Mesh::label_sequence::iterator c_iter = cBegin; c_iter != cEnd; ++c_iter) {
    const sieve_type::point_type& e = *c_iter;

    if (debug) {
      const std::string elem = ALE::Test::MeshProcessor::printElement(e, dim, mesh->restrict(coordinates, e), rank);
      ierr = PetscSynchronizedPrintf(comm, "%s", elem.c_str());CHKERRQ(ierr);
    }
    mesh->computeElementGeometry(coordinates, e, v0, J, invJ, detJ);
    if (debug) {
      ierr = PrintMatrix(comm, rank, "J",    dim, dim, J);CHKERRQ(ierr);
      ierr = PrintMatrix(comm, rank, "invJ", dim, dim, invJ);CHKERRQ(ierr);
    }
    if (detJ < 0) {SETERRQ(PETSC_ERR_ARG_WRONG, "Negative Jacobian determinant");}
  }
  delete [] v0;
  delete [] J;
  delete [] invJ;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "LabelTest"
PetscErrorCode LabelTest(const Obj<ALE::Mesh>& mesh, const Obj<label_type>& label, Options *options)
{
  const Obj<ALE::Mesh::label_sequence>&     cells  = mesh->heightStratum(0);
  const ALE::Mesh::label_sequence::iterator cBegin = cells->begin();
  const ALE::Mesh::label_sequence::iterator cEnd   = cells->end();

  PetscFunctionBegin;
  for(ALE::Mesh::label_sequence::iterator c_iter = cBegin; c_iter != cEnd; ++c_iter) {
    const sieve_type::point_type& e = *c_iter;

    if (options->test > 4) {
      mesh->setValue(label, e, 1);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NewLabelTest"
PetscErrorCode NewLabelTest(const Obj<ALE::Mesh>& mesh, const Obj<new_label_type>& label, Options *options)
{
  const Obj<ALE::Mesh::label_sequence>&     cells  = mesh->heightStratum(0);
  const ALE::Mesh::label_sequence::iterator cBegin = cells->begin();
  const ALE::Mesh::label_sequence::iterator cEnd   = cells->end();

  PetscFunctionBegin;
  for(ALE::Mesh::label_sequence::iterator c_iter = cBegin; c_iter != cEnd; ++c_iter) {
    const sieve_type::point_type& e = *c_iter;

    if (options->test > 4) {
      label->setCone(1, e);
    }
  }
  PetscFunctionReturn(0);
}

#if 0
#undef __FUNCT__
#define __FUNCT__ "AllocationTest"
PetscErrorCode AllocationTest(const Obj<ALE::Mesh>& mesh, Options *options)
{
  std::string                          name("coordinates");
  const Obj<ALE::Mesh::topology_type>& topology = mesh->getTopology();
  const Obj<ALE::Mesh::section_type>&  section  = mesh->getSection(name);
  const Obj<ALE::Mesh::order_type>&    order    = ALE::Mesh::NumberingFactory::singleton(mesh->debug)->getGlobalOrder(topology, 0, name, section->getAtlas());
  Mat                                 A;
  PetscInt                            numLocalRows, firstRow, lastRow;
  PetscInt                           *dnz, *onz;
  PetscErrorCode                      ierr;

  PetscFunctionBegin;
  ierr = MatCreate(mesh->comm(), &A);CHKERRQ(ierr);
  ierr = MatSetSizes(A, order->getLocalSize(), order->getLocalSize(), order->getGlobalSize(), order->getGlobalSize());CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  //ierr = preallocateMatrix(mesh.objPtr, section, order, A);CHKERRQ(ierr);
  const Obj<ALE::Mesh::sieve_type>    adjGraph    = new ALE::Mesh::sieve_type(mesh->comm(), mesh->debug);
  const Obj<ALE::Mesh::topology_type> adjTopology = new ALE::Mesh::topology_type(mesh->comm(), mesh->debug);
  const ALE::Mesh::section_type::patch_type patch = 0;

  adjTopology->setPatch(patch, adjGraph);
  ierr = MatGetLocalSize(A, &numLocalRows, PETSC_NULL);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A, &firstRow, &lastRow);CHKERRQ(ierr);
  ierr = PetscMalloc2(numLocalRows, PetscInt, &dnz, numLocalRows, PetscInt, &onz);CHKERRQ(ierr);
  // Create local adjacency graph
  //   In general, we need to get FIAT info that attaches dual basis vectors to sieve points
  const ALE::Mesh::section_type::atlas_type::chart_type& chart = section->getAtlas()->getPatch(patch);
  const Obj<ALE::Mesh::sieve_type>&                      sieve = mesh->getTopology()->getPatch(patch);

  for(ALE::Mesh::section_type::atlas_type::chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
    adjGraph->addCone(sieve->cone(sieve->support(*c_iter)), *c_iter);
  }
  // Distribute adjacency graph
  const Obj<ALE::Mesh::numbering_type>&    vNumbering        = ALE::Mesh::NumberingFactory::singleton(mesh->debug)->getNumbering(topology, 0, 0);
  const Obj<ALE::Mesh::send_overlap_type>& vertexSendOverlap = topology->getSendOverlap();
  const Obj<ALE::Mesh::recv_overlap_type>& vertexRecvOverlap = topology->getRecvOverlap();
  const Obj<ALE::Mesh::send_overlap_type>  nbrSendOverlap    = new ALE::Mesh::send_overlap_type(mesh->comm(), mesh->debug);
  const Obj<ALE::Mesh::recv_overlap_type>  nbrRecvOverlap    = new ALE::Mesh::recv_overlap_type(mesh->comm(), mesh->debug);
  const Obj<ALE::Mesh::send_section_type>  sendSection       = new ALE::Mesh::send_section_type(mesh->comm(), mesh->debug);
  const Obj<ALE::Mesh::recv_section_type>  recvSection       = new ALE::Mesh::recv_section_type(mesh->comm(), sendSection->getTag(), mesh->debug);

  ALE::New::Distribution<ALE::Mesh::topology_type>::coneCompletion(vertexSendOverlap, vertexRecvOverlap, adjTopology, sendSection, recvSection);
  if (mesh->debug) {
    adjTopology->view("Adjacency topology");
    vNumbering->view("Global vertex numbering");
    order->view("Global vertex order");
  }
  // Distribute indices for new points
  ALE::New::Distribution<ALE::Mesh::topology_type>::updateOverlap(sendSection, recvSection, nbrSendOverlap, nbrRecvOverlap);
  ALE::Mesh::NumberingFactory::singleton(mesh->debug)->completeOrder(order, nbrSendOverlap, nbrRecvOverlap, patch, true);
  if (mesh->debug) {
    nbrSendOverlap->view("Neighbor Send Overlap");
    nbrRecvOverlap->view("Neighbor Receive Overlap");
    order->view("Global vertex order after completion");
  }
  // Read out adjacency graph
  const ALE::Obj<ALE::Mesh::sieve_type> graph = adjTopology->getPatch(patch);

  ierr = PetscMemzero(dnz, numLocalRows * sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemzero(onz, numLocalRows * sizeof(PetscInt));CHKERRQ(ierr);
  for(ALE::Mesh::section_type::atlas_type::chart_type::const_iterator c_iter = chart.begin(); c_iter != chart.end(); ++c_iter) {
    const ALE::Mesh::section_type::atlas_type::point_type& point = *c_iter;

    if (order->isLocal(point)) {
      const Obj<ALE::Mesh::sieve_type::traits::coneSequence>& adj   = graph->cone(point);
      const ALE::Mesh::order_type::value_type&                rIdx  = order->restrictPoint(patch, point)[0];
      const int                                               row   = rIdx.prefix;
      const int                                               rSize = rIdx.index;

      for(ALE::Mesh::sieve_type::traits::coneSequence::iterator v_iter = adj->begin(); v_iter != adj->end(); ++v_iter) {
        const ALE::Mesh::atlas_type::point_type& neighbor = *v_iter;
        const ALE::Mesh::order_type::value_type& cIdx     = order->restrictPoint(patch, neighbor)[0];
        const int&                               cSize    = cIdx.index;
        
        if (order->isLocal(neighbor)) {
          for(int r = 0; r < rSize; ++r) {dnz[row - firstRow + r] += cSize;}
        } else {
          for(int r = 0; r < rSize; ++r) {onz[row - firstRow + r] += cSize;}
        }
      }
    }
  }
  if (mesh->debug) {
    int rank = mesh->commRank();
    for(int r = 0; r < numLocalRows; r++) {
      std::cout << "["<<rank<<"]: dnz["<<r<<"]: " << dnz[r] << " onz["<<r<<"]: " << onz[r] << std::endl;
    }
  }
  ierr = MatSeqAIJSetPreallocation(A, 0, dnz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(A, 0, dnz, 0, onz);CHKERRQ(ierr);
  ierr = PetscFree2(dnz, onz);CHKERRQ(ierr);
  ierr = MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE);CHKERRQ(ierr);
  // Fill matrix
  const Obj<ALE::Mesh::topology_type::label_sequence>& elements = mesh->getTopology()->heightStratum(0, 0);
  int          size       = options->dim*options->dim*9;
  PetscScalar *elementMat = new PetscScalar[size];

  for(int i = 0; i < size; ++i) elementMat[i] = 1.0;
  for(ALE::Mesh::topology_type::label_sequence::iterator e_iter = elements->begin(); e_iter != elements->end(); ++e_iter) {
    ierr = updateOperator(A, section, order, *e_iter, elementMat, ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

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
    Obj<ALE::Mesh> boundary;

    if (options.dim == 2) {
      double lower[2] = {0.0, 0.0};
      double upper[2] = {1.0, 1.0};
      int    edges[2] = {2, 2};

      boundary = ALE::MeshBuilder::createSquareBoundary(comm, lower, upper, edges, options.debug);
    } else if (options.dim == 3) {
      double lower[3] = {0.0, 0.0, 0.0};
      double upper[3] = {1.0, 1.0, 1.0};
      int    faces[3] = {1, 1, 1};

      boundary = ALE::MeshBuilder::createCubeBoundary(comm, lower, upper, faces, options.debug);
    } else {
      SETERRQ1(PETSC_ERR_SUP, "Dimension not supported: %d", options.dim);
    }
    Obj<ALE::Mesh> mesh = ALE::Generator::generateMesh(boundary, options.interpolate);

    if (mesh->commSize() > 1) {
      mesh = ALE::Distribution<ALE::Mesh>::distributeMesh(mesh);
    }
    if (options.refinementLimit > 0.0) {
      mesh = ALE::Generator::refineMesh(mesh, options.refinementLimit, options.interpolate);
    }
    if (options.debug) {
      mesh->view("Mesh");
    }
    if (options.test > 0) {
      ierr = GeometryTest(mesh, mesh->getRealSection("coordinates"), &options);CHKERRQ(ierr);
    }
    if (options.test > 1) {
      ierr = LabelTest(mesh, mesh->getLabel("marker"), &options);CHKERRQ(ierr);
    }
    if (options.test > 3) {
      Obj<new_label_type> label = new new_label_type(mesh->comm(), mesh->debug());
      ierr = NewLabelTest(mesh, label, &options);CHKERRQ(ierr);
    }
    //ierr = AllocationTest(mesh, &options);CHKERRQ(ierr);
  } catch (ALE::Exception e) {
    std::cout << e << std::endl;
  }
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
