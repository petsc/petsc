static char help[] = "Creates and outputs a structured mesh.\n\n";

#include <petscmesh.h>
#include <CartesianSieve.hh>
#include <Distribution.hh>

using ALE::Obj;
typedef ALE::CartesianMesh MeshType;

typedef struct {
  int       debug;      // The debugging level
  PetscInt  dim;        // The topological mesh dimension
  PetscInt *numCells;   // The number of cells in each dimension
  PetscInt *partitions; // The number of divisions in each dimension
} Options;

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, Options *options)
{
  PetscInt       n;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->debug = 0;
  options->dim   = 2;

  ierr = PetscOptionsBegin(comm, "", "Options for mesh loading", "DMMG");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-debug", "The debugging level", "ex11.cxx", options->debug, &options->debug, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex11.cxx", options->dim, &options->dim, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscMalloc(options->dim * sizeof(PetscInt), &options->numCells);CHKERRQ(ierr);
    for(int d = 0; d < options->dim; ++d) options->numCells[d] = 1;
    n = options->dim;
    ierr = PetscOptionsIntArray("-num_cells", "The number of cells in each dimension", "ex11.cxx", options->numCells, &n, PETSC_NULL);
    ierr = PetscMalloc(options->dim * sizeof(PetscInt), &options->partitions);CHKERRQ(ierr);
    for(int d = 0; d < options->dim; ++d) options->partitions[d] = 1;
    n = options->dim;
    ierr = PetscOptionsIntArray("-partitions", "The number of divisions in each dimension", "ex11.cxx", options->partitions, &n, PETSC_NULL);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateMesh"
PetscErrorCode CreateMesh(MPI_Comm comm, Options *options, Obj<MeshType>& mesh)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ALE::LogStage stage = ALE::LogStageRegister("MeshCreation");
  ALE::LogStagePush(stage);
  ierr = PetscPrintf(comm, "Creating mesh\n");CHKERRQ(ierr);
  mesh = ALE::CartesianMeshBuilder::createCartesianMesh(comm, options->dim, options->numCells, options->partitions, options->debug);
  mesh->view("Mesh");
  ALE::LogStagePop(stage);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TraverseCells"
PetscErrorCode TraverseCells(const Obj<MeshType>& m, Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  const int                                 rank        = m->commRank();
  //const Obj<MeshType::real_section_type>& coordinates = m->getRealSection("coordinates");
  const Obj<MeshType::sieve_type>&          sieve       = m->getSieve();
    
  // Loop over cells
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Each cell, on each process\n");CHKERRQ(ierr);
  const Obj<MeshType::label_sequence>& cells = m->heightStratum(0);
  for(MeshType::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[%d]Cell %d\n", rank, *c_iter);CHKERRQ(ierr);
    const Obj<MeshType::sieve_type::coneSequence>&     vertices = sieve->cone(*c_iter);
    const MeshType::sieve_type::coneSequence::iterator end      = vertices->end();

    for(MeshType::sieve_type::coneSequence::iterator v_iter = vertices->begin(); v_iter != end; ++v_iter) {
      ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "      vertex %d, with coordinates ", *v_iter);CHKERRQ(ierr);
#if 0
      const MeshType::real_section_type::value_type *array = coordinates->restrictPoint(*v_iter);

      ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, " %d (", *v_iter);CHKERRQ(ierr);
      for(int d = 0; d < m->getDimension(); ++d) {
        if (d > 0) {ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, ", ");CHKERRQ(ierr);}
        ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%g", array[d]);CHKERRQ(ierr);
      }
#endif
      ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, ")\n");CHKERRQ(ierr);
    }
  }
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateField"
PetscErrorCode CreateField(const Obj<MeshType>& m, Options *options)
{
  PetscFunctionBegin;
  const Obj<MeshType::real_section_type>& s     = m->getRealSection("u");
  const Obj<MeshType::label_sequence>&    cells = m->heightStratum(0);
  const Obj<ALE::Discretization>&         disc  = m->getDiscretization();
  
  disc->setNumDof(m->depth(), 1);
  s->setDebug(options->debug);
  m->setupField(s);
  for(MeshType::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
    const double value = (double) *c_iter;

    s->updatePoint(*c_iter, &value);
  }
  s->view("");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "UpdateGhosts"
PetscErrorCode UpdateGhosts(const Obj<MeshType>& m, Options *options)
{
  PetscFunctionBegin;
  const Obj<MeshType::real_section_type>& s     = m->getRealSection("v");
  const Obj<MeshType::sieve_type>&        sieve = m->getSieve();
  const Obj<MeshType::label_sequence>&    cells = m->heightStratum(0);
  const Obj<ALE::Discretization>&         disc  = m->getDiscretization();

  
  disc->setNumDof(0, 1);
  disc->setNumDof(1, 0);
  s->setDebug(options->debug);
  m->setupField(s);
  s->zero();
  for(MeshType::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
    const Obj<MeshType::sieve_type::coneSequence>&     vertices = sieve->cone(*c_iter);
    const MeshType::sieve_type::coneSequence::iterator end      = vertices->end();
    const MeshType::real_section_type::value_type      value    = 1.0;
    
    for(MeshType::sieve_type::coneSequence::iterator v_iter = vertices->begin(); v_iter != end; ++v_iter) {
      s->updateAddPoint(*v_iter, &value);
    }
  }
  s->view("Uncompleted section");
  ALE::Distribution<MeshType>::completeSection(m, s);
  s->view("Completed section");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RunTests"
PetscErrorCode RunTests(const Obj<MeshType>& mesh, Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = TraverseCells(mesh, options);CHKERRQ(ierr);
  ierr = CreateField(mesh, options);CHKERRQ(ierr);
  ierr = UpdateGhosts(mesh, options);CHKERRQ(ierr);
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
  try {
    Obj<MeshType> mesh;

    ierr = ProcessOptions(comm, &options);CHKERRQ(ierr);
    ierr = CreateMesh(comm, &options, mesh);CHKERRQ(ierr);
    ierr = RunTests(mesh, &options);CHKERRQ(ierr);
  } catch (ALE::Exception e) {
    std::cout << e << std::endl;
  }
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
