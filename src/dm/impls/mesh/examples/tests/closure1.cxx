static char help[] = "Traverses the closure of an element.\n\n";

#include "petscmesh.h"
#include <Field.hh>

using ALE::Obj;

typedef struct {
  int        debug;       // The debugging level
  PetscInt   dim;         // The topological mesh dimension
  PetscBool  interpolate; // Generate intermediate mesh elements
} Options;

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->debug       = 0;
  options->dim         = 2;
  options->interpolate = PETSC_TRUE;

  ierr = PetscOptionsBegin(comm, "", "PFLOTRAN Options", "SNES");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-debug", "The debugging level", "pflotran.cxx", options->debug, &options->debug, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex11.cxx", options->dim, &options->dim, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-interpolate", "Generate intermediate mesh elements", "bratu.cxx", options->interpolate, &options->interpolate, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateMesh"
PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm, Options *options)
{
  Mesh           boundary, mesh;
  PetscBool      view;
  PetscMPIInt    size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr = MeshCreate(comm, &boundary);CHKERRQ(ierr);
  if (options->dim == 2) {
    double lower[2] = {0.0, 0.0};
    double upper[2] = {1.0, 1.0};
    int    edges[2] = {2, 2};

    ALE::Obj<ALE::Mesh> mB = ALE::MeshBuilder::createSquareBoundary(comm, lower, upper, edges, options->debug);
    ierr = MeshSetMesh(boundary, mB);CHKERRQ(ierr);
  } else if (options->dim == 3) {
    double lower[3] = {0.0, 0.0, 0.0};
    double upper[3] = {1.0, 1.0, 1.0};
    int    faces[3] = {1, 1, 1};

    ALE::Obj<ALE::Mesh> mB = ALE::MeshBuilder::createCubeBoundary(comm, lower, upper, faces, options->debug);
    ierr = MeshSetMesh(boundary, mB);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP, "Dimension not supported: %d", options->dim);
  }
  ierr = MeshGenerate(boundary, options->interpolate, &mesh);CHKERRQ(ierr);
  if (size > 1) {
    Mesh parallelMesh;

    ierr = MeshDistribute(mesh, PETSC_NULL, &parallelMesh);CHKERRQ(ierr);
    ierr = MeshDestroy(mesh);CHKERRQ(ierr);
    mesh = parallelMesh;
  }
  ierr = PetscOptionsHasName(PETSC_NULL, "-mesh_view", &view);CHKERRQ(ierr);
  if (view) {
    ALE::Obj<ALE::Mesh> m;
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
#define __FUNCT__ "TraverseCells"
PetscErrorCode TraverseCells(DM dm, Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  Mesh mesh = (Mesh) dm;
  ALE::Obj<ALE::Mesh> m;
  
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  const int                                     rank        = m->commRank();
  const ALE::Obj<ALE::Mesh::real_section_type>& coordinates = m->getRealSection("coordinates");
  const ALE::Obj<ALE::Mesh::sieve_type>&        sieve       = m->getSieve();
    
  // Loop over cells
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Each cell (including ghosts), on each process\n");CHKERRQ(ierr);
  const ALE::Obj<ALE::Mesh::label_sequence>& cells = m->heightStratum(0);
  for(ALE::Mesh::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {

    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[%d]Cell %d\n", rank, *c_iter);CHKERRQ(ierr);
    const ALE::Obj<ALE::Mesh::sieve_type::traits::coneSequence>& faces = sieve->cone(*c_iter);
    const ALE::Mesh::sieve_type::traits::coneSequence::iterator  end  = faces->end();

    // Loop over faces owned by this process on the given cell    
    for(ALE::Mesh::sieve_type::traits::coneSequence::iterator f_iter = faces->begin(); f_iter != end; ++f_iter) {
      ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "      Face %d, with coordinates ", *f_iter);CHKERRQ(ierr);
      const ALE::Obj<ALE::Mesh::sieve_type::coneArray>& vertices = sieve->nCone(*f_iter, m->depth(*f_iter));
      
      // Loop over vertices of the given face
      for(ALE::Mesh::sieve_type::coneArray::iterator v_iter = vertices->begin(); v_iter != vertices->end(); ++v_iter) {
        const ALE::Mesh::real_section_type::value_type *array = coordinates->restrictPoint(*v_iter);
	
        ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, " %d (", *v_iter);CHKERRQ(ierr);
        for(int d = 0; d < m->getDimension(); ++d) {
          if (d > 0) {ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, ", ");CHKERRQ(ierr);}
          ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%g", array[d]);CHKERRQ(ierr);
        }
        ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, ")");CHKERRQ(ierr);
      }
      ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "\n");CHKERRQ(ierr);
    }

    // Loop over closure
    typedef ALE::SieveAlg<ALE::Mesh> sieve_alg_type;
    const Obj<sieve_alg_type::coneArray> closure = sieve_alg_type::closure(m, *c_iter);
    sieve_alg_type::coneArray::iterator  cEnd    = closure->end();

    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "      Closure of %d\n      ", *c_iter);CHKERRQ(ierr);
    for(sieve_alg_type::coneArray::iterator p_iter = closure->begin(); p_iter != cEnd; ++p_iter) {
      ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, " %d", *p_iter);CHKERRQ(ierr);
    }
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "\n");CHKERRQ(ierr);

    // Loop over oriented closure
    const Obj<sieve_alg_type::orientedConeArray> oClosure = sieve_alg_type::orientedClosure(m, *c_iter);
    sieve_alg_type::orientedConeArray::iterator  ocEnd    = oClosure->end();

    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "      Oriented Closure of %d\n      ", *c_iter);CHKERRQ(ierr);
    for(sieve_alg_type::orientedConeArray::iterator p_iter = oClosure->begin(); p_iter != ocEnd; ++p_iter) {
      ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, " (%d, %d)", p_iter->first, p_iter->second);CHKERRQ(ierr);
    }
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "\n");CHKERRQ(ierr);
  }
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD);CHKERRQ(ierr);
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
  try {
    comm = PETSC_COMM_WORLD;
    ierr = ProcessOptions(comm, &options);CHKERRQ(ierr);
    ierr = CreateMesh(comm, &dm, &options);CHKERRQ(ierr);
    ierr = TraverseCells(dm, &options);CHKERRQ(ierr);
    ierr = DestroyMesh(dm, &options);CHKERRQ(ierr);
  } catch(ALE::Exception e) {
    std::cout << "ERROR: " << e.msg() << std::endl;
  }
  ierr = PetscFinalize();
  PetscFunctionReturn(0);
}
