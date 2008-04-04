static char help[] = "Sieve Package Parallel Correctness Tests.\n\n";

#define ALE_HAVE_CXX_ABI
#define ALE_MEM_LOGGING
#include <petscmesh.hh>
#include <petscmesh_viewers.hh>

#include <IField.hh>
#include <ParallelMapping.hh>

using ALE::Obj;

typedef struct {
  PetscInt   debug;
  MPI_Comm   comm;
  PetscInt   rank;
  PetscInt   size;
  // Classes
  PetscTruth section;     // Run the Section tests
  PetscTruth isection;    // Run the ISection tests
  PetscTruth partition;   // Run the Partition tests
  // Run flags
  PetscInt   number;      // Number of each class to create
  // Mesh flags
  PetscInt   numCells;    // If possible, set the total number of cells
  PetscTruth interpolate; // Interpolate the mesh
  // Section flags
  PetscInt   components;  // Number of section components
} Options;

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, Options *options)
{
  //ALE::MemoryLogger& logger = ALE::MemoryLogger::singleton();
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  options->debug       = 0;
  options->section     = PETSC_FALSE;
  options->isection    = PETSC_FALSE;
  options->partition   = PETSC_FALSE;
  options->number      = 0;
  options->numCells    = 8;
  options->interpolate = PETSC_FALSE;
  options->components  = 3;

  ierr = PetscOptionsBegin(comm, "", "Options for the Sieve package tests", "Sieve");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-debug", "Debugging flag", "distTests", options->debug, &options->debug, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-section", "Run Section tests", "distTests", options->section, &options->section, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-isection", "Run ISection tests", "distTests", options->isection, &options->isection, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-partition", "Run Partition tests", "distTests", options->partition, &options->partition, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-num", "Number of each class to create", "distTests", options->number, &options->number, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-numCells", "Number of mesh cells", "distTests", options->numCells, &options->numCells, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-interpolate", "Interpolate the flag", "distTests", options->interpolate, &options->interpolate, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-components", "Number of section components", "distTests", options->components, &options->components, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  options->comm = comm;
  ierr = MPI_Comm_rank(comm, &options->rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &options->size);CHKERRQ(ierr);
  //logger.setDebug(options->debug);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateScatterOverlap"
template<typename SendOverlap, typename RecvOverlap>
PetscErrorCode CreateScatterOverlap(const Obj<SendOverlap>& sendOverlap, const Obj<RecvOverlap>& recvOverlap, const bool localNumbering, const Options *options)
{
  const PetscInt rank     = options->rank;
  const PetscInt size     = options->size;
  const PetscInt numCells = options->numCells;
  const PetscInt block    = numCells/size;

  PetscFunctionBegin;
  if (!rank) {
    for(PetscInt r = 1; r < size; ++r) {
      const PetscInt rStart = r*block     + PetscMin(r, numCells%size);
      const PetscInt rEnd   = (r+1)*block + PetscMin(r+1, numCells%size);

      for(PetscInt c = rStart; c < rEnd; ++c) {
        if (localNumbering) {
          sendOverlap->addArrow(c, r, c - rStart);
        } else {
          sendOverlap->addArrow(c, r, c);
        }
      }
    }
  } else {
    const PetscInt start = rank*block     + PetscMin(rank, numCells%size);
    const PetscInt end   = (rank+1)*block + PetscMin(rank+1, numCells%size);

    for(PetscInt c = start; c < end; ++c) {
      if (localNumbering) {
        recvOverlap->addArrow(0, c - start, c);
      } else {
        recvOverlap->addArrow(0, c, c);
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ConstantSectionTest"
PetscErrorCode ConstantSectionTest(const Options *options)
{
  typedef int point_type;
  typedef ALE::Sifter<int,point_type,point_type> send_overlap_type;
  typedef ALE::Sifter<point_type,int,point_type> recv_overlap_type;
  typedef ALE::ConstantSection<point_type, double> section;
  Obj<send_overlap_type> sendOverlap     = new send_overlap_type(options->comm);
  Obj<recv_overlap_type> recvOverlap     = new recv_overlap_type(options->comm);
  Obj<section>           serialSection   = new section(options->comm, options->debug);
  Obj<section>           parallelSection = new section(options->comm, options->debug);
  section::value_type    value           = 7.0;
  
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = CreateScatterOverlap(sendOverlap, recvOverlap, true, options);CHKERRQ(ierr);
  serialSection->addPoint(sendOverlap->cap());
  if (!options->rank) {
    serialSection->update(0, &value);
    parallelSection->update(0, &value);
  }
  ALE::Completion::completeSection(sendOverlap, recvOverlap, serialSection, parallelSection);
  if (options->debug) {parallelSection->view("Parallel ConstantSection");}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "UniformSectionTest"
PetscErrorCode UniformSectionTest(const Options *options)
{
  typedef int point_type;
  typedef ALE::Sifter<int,point_type,point_type> send_overlap_type;
  typedef ALE::Sifter<point_type,int,point_type> recv_overlap_type;
  typedef ALE::UniformSection<point_type, double, 4> section;
  Obj<send_overlap_type> sendOverlap     = new send_overlap_type(options->comm);
  Obj<recv_overlap_type> recvOverlap     = new recv_overlap_type(options->comm);
  Obj<section>           serialSection   = new section(options->comm, options->debug);
  Obj<section>           parallelSection = new section(options->comm, options->debug);
  section::value_type    value[4];
  
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = CreateScatterOverlap(sendOverlap, recvOverlap, true, options);CHKERRQ(ierr);
  if (!options->rank) {
    for(int c = 0; c < options->numCells; ++c) {
      for(PetscInt comp = 0; comp < 4; ++comp) {value[comp] = (c+1)*(comp+1);}
      serialSection->setFiberDimension(c, 4);
      serialSection->updatePoint(c, value);
    }
    const PetscInt rStart = 0;
    const PetscInt rEnd   = options->numCells/options->size + PetscMin(1, options->numCells%options->size);

    for(PetscInt c = rStart, locC = 0; c < rEnd; ++c, ++locC) {
      parallelSection->setFiberDimension(locC, 4);
      parallelSection->updatePoint(locC, serialSection->restrictPoint(c));
    }
  }
  ALE::Completion::completeSection(sendOverlap, recvOverlap, serialSection, parallelSection);
  if (options->debug) {parallelSection->view("Parallel UniformSection");}
  PetscFunctionReturn(0);
}

// Describe the completion process as the composition of two pullbacks:
//
//   copy(): unifies pullback to the overlap (restrict) and copy to neighboring process
//   fuse(): unifies pullback across the overlap, then to the big section (update), and then fusion
//
// For parallel completion, we must first copy the section, restricted to the
// overlap, to the other process, then we can proceed as above. The copy
// process can be broken down as in the distribution paper:
//
//   1) Copy atlas (perhaps empty, signaling a single datum per point)
//   2) Copy data
//
// The atlas copy is recursive, since it is itself a section.
//
// Thus, we have a software plan
//
// - Write parallel section copy (currently in ALE::Field)
//
// - Write pullback as a generic algorithm
//
// - Write fuse as a generic algorithm
//   - Implement add and replace fusers
//
// Now we move back to
//   We do not need the initial pullback
//   copy: Copy the vector and leave it in the old domain
//   fuse: fuse the copy, pulled back to the new domain, with the existing whole section
//
// Also, we need a way to update overlaps based on a renumbering
//
// We need to wrap distribution around completion
//   Create local section
//   Complete, with fuser which adds in
#undef __FUNCT__
#define __FUNCT__ "SectionTest"
PetscErrorCode SectionTest(const Options *options)
{
  typedef int point_type;
  typedef ALE::Sifter<int,point_type,point_type> send_overlap_type;
  typedef ALE::Sifter<point_type,int,point_type> recv_overlap_type;
  typedef ALE::Section<point_type, double> section;
  Obj<send_overlap_type> sendOverlap     = new send_overlap_type(options->comm);
  Obj<recv_overlap_type> recvOverlap     = new recv_overlap_type(options->comm);
  Obj<section>           serialSection   = new section(options->comm, options->debug);
  Obj<section>           parallelSection = new section(options->comm, options->debug);
  section::value_type   *value;
  
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc(options->components * sizeof(double), &value);CHKERRQ(ierr);
  ierr = CreateScatterOverlap(sendOverlap, recvOverlap, true, options);CHKERRQ(ierr);
  if (!options->rank) {
    for(PetscInt c = 0; c < options->numCells; ++c) {
      serialSection->setFiberDimension(c, options->components);
    }
  }
  serialSection->allocatePoint();
  if (!options->rank) {
    for(PetscInt c = 0; c < options->numCells; ++c) {
      for(PetscInt comp = 0; comp < options->components; ++comp) {value[comp] = (c+1)*(comp+1);}
      serialSection->updatePoint(c, value);
    }
  }
  ierr = PetscFree(value);CHKERRQ(ierr);
  ALE::Completion::completeSection(sendOverlap, recvOverlap, serialSection, parallelSection);
  if (!options->rank) {
    const PetscInt rStart = 0;
    const PetscInt rEnd   = options->numCells/options->size + PetscMin(1, options->numCells%options->size);

    for(PetscInt c = rStart, locC = 0; c < rEnd; ++c, ++locC) {
      parallelSection->setFiberDimension(locC, serialSection->getFiberDimension(c));
    }
    parallelSection->allocatePoint();
    for(PetscInt c = rStart, locC = 0; c < rEnd; ++c, ++locC) {
      parallelSection->updatePoint(locC, serialSection->restrictPoint(c));
    }
  }
  if (options->debug) {parallelSection->view("Parallel Section");}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SectionTests"
PetscErrorCode SectionTests(const Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = ConstantSectionTest(options);CHKERRQ(ierr);
  ierr = UniformSectionTest(options);CHKERRQ(ierr);
  ierr = SectionTest(options);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SectionToISectionTest"
PetscErrorCode SectionToISectionTest(const Options *options)
{
  typedef int point_type;
  typedef ALE::Sifter<point_type,int,point_type> send_overlap_type;
  typedef ALE::Sifter<int,point_type,point_type> recv_overlap_type;
  typedef ALE::Section<point_type, double>  section;
  typedef ALE::ISection<point_type, double> isection;
  Obj<send_overlap_type> sendOverlap     = new send_overlap_type(options->comm);
  Obj<recv_overlap_type> recvOverlap     = new recv_overlap_type(options->comm);
  Obj<section>           serialSection   = new section(options->comm, options->debug);
  section::value_type   *value;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc(options->components * sizeof(double), &value);CHKERRQ(ierr);
  ierr = CreateScatterOverlap(sendOverlap, recvOverlap, true, options);CHKERRQ(ierr);
  const int     localSize       = options->rank ? recvOverlap->base()->size() : options->numCells/options->size + PetscMin(1, options->numCells%options->size);
  Obj<isection> parallelSection = new isection(options->comm, 0, localSize, options->debug);

  if (!options->rank) {
    for(PetscInt c = 0; c < options->numCells; ++c) {
      serialSection->setFiberDimension(c, options->components);
    }
  }
  serialSection->allocatePoint();
  if (!options->rank) {
    for(PetscInt c = 0; c < options->numCells; ++c) {
      for(PetscInt comp = 0; comp < options->components; ++comp) {value[comp] = (c+1)*(comp+1);}
      serialSection->updatePoint(c, value);
    }
  }
  ierr = PetscFree(value);CHKERRQ(ierr);
  ALE::Completion::completeSection(sendOverlap, recvOverlap, serialSection, parallelSection);
  if (!options->rank) {
    const PetscInt rStart = 0;
    const PetscInt rEnd   = options->numCells/options->size + PetscMin(1, options->numCells%options->size);

    for(PetscInt c = rStart, locC = 0; c < rEnd; ++c, ++locC) {
      parallelSection->setFiberDimension(locC, serialSection->getFiberDimension(c));
    }
    parallelSection->allocatePoint();
    for(PetscInt c = rStart, locC = 0; c < rEnd; ++c, ++locC) {
      parallelSection->updatePoint(locC, serialSection->restrictPoint(c));
    }
  }
  if (options->debug) {parallelSection->view("Parallel ISection");}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ISectionTests"
PetscErrorCode ISectionTests(const Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SectionToISectionTest(options);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ViewMesh"
template<typename Partition>
PetscErrorCode ViewMesh(const Obj<ALE::Mesh>& mesh, const Obj<Partition>& partition, const Options *options)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SieveISectionPartitionTest"
PetscErrorCode SieveISectionPartitionTest(const Options *options)
{
  typedef ALE::Mesh::point_type                        point_type;
  typedef ALE::Partitioner<>::part_type                rank_type;
  typedef ALE::Sifter<point_type,rank_type,point_type> mesh_send_overlap_type;
  typedef ALE::Sifter<rank_type,point_type,point_type> mesh_recv_overlap_type;
  typedef ALE::DistributionNew<ALE::Mesh>              distribution_type;
  typedef distribution_type::partition_type            partition_type;
  double                            lower[2]        = {0.0, 0.0};
  double                            upper[2]        = {1.0, 1.0};
  int                               edges[2]        = {2, 2};
  const Obj<ALE::Mesh>              mB              = ALE::MeshBuilder::createSquareBoundary(PETSC_COMM_WORLD, lower, upper, edges, 0);
  const Obj<ALE::Mesh>              mesh            = ALE::Generator::generateMesh(mB, options->interpolate);
  Obj<ALE::Mesh>                    parallelMesh    = new ALE::Mesh(options->comm, mesh->getDimension(), options->debug);
  Obj<ALE::Mesh::sieve_type>        parallelSieve   = new ALE::Mesh::sieve_type(options->comm, options->debug);
  const Obj<mesh_send_overlap_type> sendMeshOverlap = new mesh_send_overlap_type(mesh->comm(), mesh->debug());
  const Obj<mesh_recv_overlap_type> recvMeshOverlap = new mesh_recv_overlap_type(mesh->comm(), mesh->debug());
  const int                         height          = 0;
  std::map<point_type,point_type>   renumbering;

  PetscFunctionBegin;
  mesh->setDebug(options->debug);
  parallelMesh->setSieve(parallelSieve);
  if (options->debug) {mesh->view("Serial Mesh");}
  Obj<partition_type> partition = distribution_type::distributeMesh(mesh, parallelMesh, renumbering, sendMeshOverlap, recvMeshOverlap, height);
  if (options->debug) {parallelMesh->view("Parallel Mesh");}
  // Distribute the coordinates
  typedef ALE::ISection<point_type, double> real_section_type;
  const Obj<ALE::Mesh::real_section_type>& coordinates    = mesh->getRealSection("coordinates");
  const int                                firstVertex    = parallelMesh->heightStratum(0)->size();
  const int                                lastVertex     = firstVertex+parallelMesh->depthStratum(0)->size();
  const Obj<real_section_type>             newCoordinates = new real_section_type(parallelMesh->comm(), firstVertex, lastVertex, parallelMesh->debug());

  distribution_type::distributeSection(coordinates, partition, renumbering, sendMeshOverlap, recvMeshOverlap, newCoordinates);
  if (options->debug) {newCoordinates->view("Parallel Coordinates");}
  // Create the parallel overlap
  Obj<mesh_send_overlap_type> sendParallelMeshOverlap = new mesh_send_overlap_type(options->comm);
  Obj<mesh_recv_overlap_type> recvParallelMeshOverlap = new mesh_recv_overlap_type(options->comm);
  //   Can I figure this out in a nicer way?
  ALE::SetFromMap<std::map<point_type,point_type> > globalPoints(renumbering);

  ALE::OverlapBuilder<>::constructOverlap(globalPoints, renumbering, sendParallelMeshOverlap, recvParallelMeshOverlap);
  sendParallelMeshOverlap->view("Send parallel mesh overlap");
  recvParallelMeshOverlap->view("Receive parallel mesh overlap");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ISieveISectionPartitionTest"
PetscErrorCode ISieveISectionPartitionTest(const Options *options)
{
  typedef ALE::IMesh                                   mesh_type;
  typedef ALE::Mesh::point_type                        point_type;
  typedef ALE::Partitioner<>::part_type                rank_type;
  typedef ALE::Sifter<point_type,rank_type,point_type> mesh_send_overlap_type;
  typedef ALE::Sifter<rank_type,point_type,point_type> mesh_recv_overlap_type;
  typedef ALE::DistributionNew<mesh_type>              distribution_type;
  typedef distribution_type::partition_type            partition_type;
  double                            lower[2]        = {0.0, 0.0};
  double                            upper[2]        = {1.0, 1.0};
  int                               edges[2]        = {2, 2};
  const Obj<ALE::Mesh>              mB              = ALE::MeshBuilder::createSquareBoundary(PETSC_COMM_WORLD, lower, upper, edges, 0);
  const Obj<ALE::Mesh>              m               = ALE::Generator::generateMesh(mB, options->interpolate);
  Obj<mesh_type>                    mesh            = new mesh_type(options->comm, m->getDimension(), options->debug);
  Obj<mesh_type::sieve_type>        sieve           = new mesh_type::sieve_type(options->comm, options->debug);
  Obj<mesh_type>                    parallelMesh    = new mesh_type(options->comm, m->getDimension(), options->debug);
  Obj<mesh_type::sieve_type>        parallelSieve   = new mesh_type::sieve_type(options->comm, options->debug);
  const Obj<mesh_send_overlap_type> sendMeshOverlap = new mesh_send_overlap_type(mesh->comm(), mesh->debug());
  const Obj<mesh_recv_overlap_type> recvMeshOverlap = new mesh_recv_overlap_type(mesh->comm(), mesh->debug());
  const int                         height          = 0;
  std::map<point_type,point_type>   renumbering;

  PetscFunctionBegin;
  mesh->setSieve(sieve);
  ALE::ISieveConverter::convertMesh(*m, *mesh, renumbering);
  renumbering.clear();
  parallelMesh->setSieve(parallelSieve);
  if (options->debug) {mesh->view("Serial Mesh");}
  Obj<partition_type> partition = distribution_type::distributeMeshV(mesh, parallelMesh, renumbering, sendMeshOverlap, recvMeshOverlap, height);
  if (options->debug) {parallelMesh->view("Parallel Mesh");}
  // Distribute the coordinates
  typedef mesh_type::real_section_type real_section_type;
  const Obj<real_section_type>& coordinates         = mesh->getRealSection("coordinates");
  const Obj<real_section_type>& parallelCoordinates = parallelMesh->getRealSection("coordinates");

  if (options->debug) {coordinates->view("Serial Coordinates");}
  parallelMesh->setupCoordinates(parallelCoordinates);
  distribution_type::distributeSection(coordinates, partition, renumbering, sendMeshOverlap, recvMeshOverlap, parallelCoordinates);
  if (options->debug) {parallelCoordinates->view("Parallel Coordinates");}
  // Create the parallel overlap
  Obj<mesh_send_overlap_type> sendParallelMeshOverlap = new mesh_send_overlap_type(options->comm);
  Obj<mesh_recv_overlap_type> recvParallelMeshOverlap = new mesh_recv_overlap_type(options->comm);
  //   Can I figure this out in a nicer way?
  ALE::SetFromMap<std::map<point_type,point_type> > globalPoints(renumbering);

  ALE::OverlapBuilder<>::constructOverlap(globalPoints, renumbering, sendParallelMeshOverlap, recvParallelMeshOverlap);
  sendParallelMeshOverlap->view("Send parallel mesh overlap");
  recvParallelMeshOverlap->view("Receive parallel mesh overlap");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PartitionTests"
PetscErrorCode PartitionTests(const Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = SieveISectionPartitionTest(options);CHKERRQ(ierr);
  ierr = ISieveISectionPartitionTest(options);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RunUnitTests"
PetscErrorCode RunUnitTests(const Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (options->section)   {ierr = SectionTests(options);CHKERRQ(ierr);}
  if (options->isection)  {ierr = ISectionTests(options);CHKERRQ(ierr);}
  if (options->partition) {ierr = PartitionTests(options);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  Options        options;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr);
  ierr = PetscLogBegin();CHKERRQ(ierr);
  ierr = ProcessOptions(PETSC_COMM_WORLD, &options);CHKERRQ(ierr);
  try {
    ierr = RunUnitTests(&options);CHKERRQ(ierr);
  } catch (ALE::Exception e) {
    std::cerr << e << std::endl;
  }
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
