static char help[] = "Sieve Distribution Tests.\n\n";

#include <petsc.h>
#include "overlapTest.hh"
#include "meshTest.hh"
#include <Completion.hh>

extern PetscErrorCode PetscCommSynchonizeTags(MPI_Comm);

using ALE::Obj;
typedef ALE::Test::sieve_type sieve_type;
typedef ALE::Test::OverlapTest::dsieve_type       dsieve_type;
typedef ALE::Test::OverlapTest::send_overlap_type send_overlap_type;
typedef ALE::Test::OverlapTest::send_section_type send_section_type;
typedef ALE::Test::OverlapTest::recv_overlap_type recv_overlap_type;
typedef ALE::Test::OverlapTest::recv_section_type recv_section_type;

typedef struct {
  int        debug;              // The debugging level
  int        dim;                // The topological mesh dimension
  char       baseFilename[2048]; // The base filename for mesh files
  PetscTruth useZeroBase;        // Use zero-based indexing
  PetscTruth interpolate;        // Construct missing elements of the mesh
} Options;

PetscErrorCode SendDistribution(const Obj<ALE::Mesh>& mesh, const Obj<ALE::Mesh>& meshNew, Options *options)
{
  PetscFunctionBegin;
  ALE::New::Completion::sendDistribution(mesh->getTopologyNew(), mesh->getDimension(), meshNew->getTopologyNew());
  PetscFunctionReturn(0);
}

PetscErrorCode ReceiveDistribution(const Obj<ALE::Mesh>& mesh, const Obj<ALE::Mesh>& meshNew, Options *options)
{
  PetscFunctionBegin;
  ALE::New::Completion::receiveDistribution(mesh->getTopologyNew(), meshNew->getTopologyNew());
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DistributionTest"
// This example does distribution from a central source
PetscErrorCode DistributionTest(const Obj<ALE::Mesh>& mesh, const Obj<ALE::Mesh>& meshNew, Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  const Obj<ALE::Mesh::topology_type>& topology = new ALE::Mesh::topology_type(mesh->comm(), mesh->debug);
  const Obj<ALE::Mesh::sieve_type>&    sieve    = new ALE::Mesh::sieve_type(mesh->comm(), mesh->debug);

  topology->setPatch(0, sieve);
  meshNew->setTopologyNew(topology);
  if (mesh->commRank() == 0) {
    ierr = SendDistribution(mesh, meshNew, options);CHKERRQ(ierr);
  } else {
    ierr = ReceiveDistribution(mesh, meshNew, options);CHKERRQ(ierr);
  }
  // This is necessary since we create types (like PartitionSection) on a subset of processors
  ierr = PetscCommSynchonizeTags(PETSC_COMM_WORLD); CHKERRQ(ierr);
  sieve->view("Distributed sieve");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->debug       = 0;
  options->dim         = 2;
  ierr = PetscStrcpy(options->baseFilename, "../tutorials/data/ex1_2d");CHKERRQ(ierr);
  options->useZeroBase = PETSC_TRUE;
  options->interpolate = PETSC_TRUE;

  ierr = PetscOptionsBegin(comm, "", "Options for sifter stress test", "Sieve");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-debug", "The debugging level",            "section1.c", options->debug, &options->debug, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-dim",   "The topological mesh dimension", "section1.c", options->dim, &options->dim,   PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsString("-base_file", "The base filename for mesh files", "section1.c", options->baseFilename, options->baseFilename, 2048, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-use_zero_base", "Use zero-based indexing", "section1.c", options->useZeroBase, &options->useZeroBase, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-interpolate", "Construct missing elements of the mesh", "ex1.c", options->interpolate, &options->interpolate, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
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
    Obj<ALE::Mesh> mesh = ALE::PCICE::Builder::readMesh(comm, options.dim, options.baseFilename, options.useZeroBase, options.interpolate, options.debug);
    Obj<ALE::Mesh> meshNew = new ALE::Mesh(comm, options.debug);

    if (options.debug) {
      mesh->getTopologyNew()->view("Mesh");
    }
    ierr = DistributionTest(mesh, meshNew, &options);CHKERRQ(ierr);
  } catch (ALE::Exception e) {
    std::cout << e << std::endl;
  }
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
