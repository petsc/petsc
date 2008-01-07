static char help[] = "Sieve Package Memory Tests.\n\n";

#include <aleAlloc.hh>

#include <petscmesh.hh>
#include <set>

using ALE::Obj;

typedef struct {
  // Classes
  PetscTruth set;         // Run the set tests
  PetscTruth sifter;      // Run the Sifter tests
  PetscTruth label;       // Run the label tests
  PetscTruth sieve;       // Run the Sieve tests
  PetscTruth mesh;        // Run the Mesh tests
  PetscTruth section;     // Run the Section tests
  // Run flags
  PetscInt   number;      // Number of each class to create
  // Mesh flags
  PetscInt   numCells;    // If possible, set the total number of cells
  PetscTruth interpolate; // Interpolate the mesh
  PetscReal  refine;      // The refinement limit
  // Section flags
  PetscInt   components;  // Number of section components
  PetscTruth shareAtlas;  // Share the atlas among the sections
  PetscTruth distribute;  // Distribute the section
} Options;

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->set         = PETSC_FALSE;
  options->sifter      = PETSC_FALSE;
  options->label       = PETSC_FALSE;
  options->sieve       = PETSC_FALSE;
  options->mesh        = PETSC_FALSE;
  options->section     = PETSC_FALSE;
  options->number      = 0;
  options->numCells    = 8;
  options->interpolate = PETSC_FALSE;
  options->refine      = 0.0;
  options->components  = 3;
  options->shareAtlas  = PETSC_FALSE;
  options->distribute  = PETSC_FALSE;

  ierr = PetscOptionsBegin(comm, "", "Options for the Sieve package tests", "Sieve");CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-set", "Run set tests", "memTests", options->set, &options->set, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-sifter", "Run Sifter tests", "memTests", options->sifter, &options->sifter, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-label", "Run Label tests", "memTests", options->label, &options->label, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-sieve", "Run Sieve tests", "memTests", options->sieve, &options->sieve, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-mesh", "Run Mesh tests", "memTests", options->mesh, &options->mesh, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-section", "Run Section tests", "memTests", options->section, &options->section, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-num", "Number of each class to create", "memTests", options->number, &options->number, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-numCells", "Number of mesh cells", "memTests", options->numCells, &options->numCells, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-interpolate", "Interpolate the mesh", "memTests", options->interpolate, &options->interpolate, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-refine", "The refinement limit", "memTests", options->refine, &options->refine, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-components", "Number of section components", "memTests", options->components, &options->components, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-share_atlas", "Share section atlases", "memTests", options->shareAtlas, &options->shareAtlas, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-distribute", "Distribute the section", "memTests", options->distribute, &options->distribute, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateSquareMesh"
PetscErrorCode CreateSquareMesh(Obj<ALE::Mesh>& mesh, const Options *options)
{
  double lower[2] = {0.0, 0.0};
  double upper[2] = {1.0, 1.0};
  int    edges[2] = {2, 2};

  PetscFunctionBegin;
  const ALE::Obj<ALE::Mesh> mB = ALE::MeshBuilder::createSquareBoundary(PETSC_COMM_WORLD, lower, upper, edges, 0);
  mesh = ALE::Generator::generateMesh(mB, options->interpolate);
  if (options->refine > 0.0) {
    mesh = ALE::Generator::refineMesh(mesh, options->refine, options->interpolate);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SetTest"
PetscErrorCode SetTest(const Options *options)
{
  ALE::MemoryLogger& logger = ALE::MemoryLogger::singleton();
  const PetscInt     num    = options->number;

  PetscFunctionBegin;
  logger.stagePush("Set");
  for(PetscInt i = 0; i < num; ++i) {
    std::set<int, std::less<int>, ALE::malloc_allocator<int> > s;

    for(PetscInt c = 0; c < options->numCells; ++c) {
      s.insert(c);
    }
  }
  logger.stagePop();
  if (logger.getNumAllocations("Set") != options->numCells*num) {
    SETERRQ2(PETSC_ERR_PLIB, "Invalid number of allocations %d should be %d", logger.getNumAllocations("Set"), options->numCells*num);
  }
  if (logger.getNumDeallocations("Set") != options->numCells*num) {
    SETERRQ2(PETSC_ERR_PLIB, "Invalid number of deallocations %d should be %d", logger.getNumDeallocations("Set"), options->numCells*num);
  }
  if (logger.getAllocationTotal("Set") != 20*options->numCells*num) {
    SETERRQ2(PETSC_ERR_PLIB, "Invalid number of bytes allocated %d should be %d", logger.getAllocationTotal("Set"), 20*options->numCells*num);
  }
  if (logger.getDeallocationTotal("Set") != 20*options->numCells*num) {
    SETERRQ2(PETSC_ERR_PLIB, "Invalid number of bytes deallocated %d should be %d", logger.getDeallocationTotal("Set"), 20*options->numCells*num);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "LabelTest"
PetscErrorCode LabelTest(const Options *options)
{
  ALE::MemoryLogger& logger = ALE::MemoryLogger::singleton();
  const PetscInt     num    = options->number;

  PetscFunctionBegin;
  logger.stagePush("Label");
  for(PetscInt i = 0; i < num; ++i) {
    ALE::LabelSifter<int, int, ALE::malloc_allocator<ALE::NewSifterDef::ArrowContainer<int, int>::traits::arrow_type> > label;

    for(PetscInt c = 0; c < options->numCells; ++c) {
      label.setCone(1, c);
    }
  }
  logger.stagePop();
  if (logger.getNumAllocations("Label") != (options->numCells+1)*num) {
    SETERRQ2(PETSC_ERR_PLIB, "Invalid number of allocations %d should be %d", logger.getNumAllocations("Label"), (options->numCells+1)*num);
  }
  if (logger.getNumDeallocations("Label") != (options->numCells+1)*num) {
    SETERRQ2(PETSC_ERR_PLIB, "Invalid number of deallocations %d should be %d", logger.getNumDeallocations("Label"), (options->numCells+1)*num);
  }
  if (logger.getAllocationTotal("Label") != 40*(options->numCells+1)*num) {
    SETERRQ2(PETSC_ERR_PLIB, "Invalid number of bytes allocated %d should be %d", logger.getAllocationTotal("Label"), 40*(options->numCells+1)*num);
  }
  if (logger.getDeallocationTotal("Label") != 40*(options->numCells+1)*num) {
    SETERRQ2(PETSC_ERR_PLIB, "Invalid number of bytes deallocated %d should be %d", logger.getDeallocationTotal("Label"), 40*(options->numCells+1)*num);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SectionTest"
PetscErrorCode SectionTest(const Options *options)
{
  ALE::MemoryLogger& logger   = ALE::MemoryLogger::singleton();
  const PetscInt     num      = options->number;
  // Allocs:
  //   Atlas
  //   Atlas Obj
  //     Atlas Obj
  //   BC Obj
  //     Atlas Obj
  //       Atlas Obj
  //   Data
  const PetscInt     numAlloc = 7*options->number;
  const PetscInt     numBytes = (84+4*5+8*options->components*options->numCells)*options->number;
  double            *values;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  logger.setDebug(1);
  logger.stagePush("Section");
  ierr = PetscMalloc(options->components * sizeof(double), &values);CHKERRQ(ierr);
  for(PetscInt c = 0; c < options->components; ++c) {values[c] = 1.0;}
  for(PetscInt i = 0; i < num; ++i) {
    ALE::GeneralSection<int, double, ALE::malloc_allocator<double> > section(PETSC_COMM_SELF);

    for(PetscInt c = 0; c < options->numCells; ++c) {
      section.setFiberDimension(c, options->components);
    }
    section.allocatePoint();
    for(PetscInt c = 0; c < options->numCells; ++c) {
      section.updatePoint(c, values);
    }
  }
  ierr = PetscFree(values);CHKERRQ(ierr);
  logger.stagePop();
  if (logger.getNumAllocations("Section") != numAlloc) {
    SETERRQ2(PETSC_ERR_PLIB, "Invalid number of allocations %d should be %d", logger.getNumAllocations("Section"), numAlloc);
  }
  if (logger.getNumDeallocations("Section") != numAlloc) {
    SETERRQ2(PETSC_ERR_PLIB, "Invalid number of deallocations %d should be %d", logger.getNumDeallocations("Section"), numAlloc);
  }
  if (logger.getAllocationTotal("Section") != numBytes) {
    SETERRQ2(PETSC_ERR_PLIB, "Invalid number of bytes allocated %d should be %d", logger.getAllocationTotal("Section"), numBytes);
  }
  if (logger.getDeallocationTotal("Section") != numBytes) {
    SETERRQ2(PETSC_ERR_PLIB, "Invalid number of bytes deallocated %d should be %d", logger.getDeallocationTotal("Section"), numBytes);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MeshTest"
PetscErrorCode MeshTest(const Options *options)
{
  Obj<ALE::Mesh> mesh;
  const PetscInt num = options->number;
  PetscInt       numPoints = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for(PetscInt i = 0; i < num; ++i) {
    ierr = CreateSquareMesh(mesh, options);CHKERRQ(ierr);
    for(int d = 0; d <= 2; ++d) {
      numPoints += mesh->depthStratum(d)->size();
    }
    std::cout << "Mesh points: " << numPoints << std::endl;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SectionTest2"
PetscErrorCode SectionTest2(const Options *options)
{
  Obj<ALE::Mesh> mesh;
  Obj<ALE::Mesh> parallelMesh;
  Obj<ALE::Mesh::real_section_type::atlas_type> firstAtlas;
  const PetscInt num = options->number;
  double        *values;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = CreateSquareMesh(mesh, options);CHKERRQ(ierr);
  if (options->distribute) {
    parallelMesh = ALE::Distribution<ALE::Mesh>::distributeMesh(mesh);
  }
  ierr = PetscMalloc(options->components * sizeof(double), &values);CHKERRQ(ierr);
  for(PetscInt c = 0; c < options->components; ++c) {values[c] = 1.0;}
  for(PetscInt i = 0; i < num; ++i) {
    ostringstream                             name;
    name << "section: " << i;
    const Obj<ALE::Mesh::real_section_type>&  section = mesh->getRealSection(name.str());
    const Obj<ALE::Mesh::label_sequence>&     cells   = mesh->heightStratum(0);
    const ALE::Mesh::label_sequence::iterator end     = cells->end();

    if ((i > 0) && options->shareAtlas) {
      section->setAtlas(firstAtlas);
    } else {
      section->setFiberDimension(cells, options->components);
    }
    section->allocatePoint();
    for(ALE::Mesh::label_sequence::iterator c_iter = cells->begin(); c_iter != end; ++c_iter) {
      section->updatePoint(*c_iter, values);
    }
    if (i == 0) {firstAtlas = section->getAtlas();}
    if (options->distribute) {
      Obj<ALE::Mesh::real_section_type> distSection;

      distSection = ALE::Distribution<ALE::Mesh>::distributeSection(section, parallelMesh, parallelMesh->getDistSendOverlap(), parallelMesh->getDistRecvOverlap());
    }
  }
  ierr = PetscFree(values);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RunUnitTests"
PetscErrorCode RunUnitTests(const Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (options->set)     {ierr = SetTest(options);CHKERRQ(ierr);}
  if (options->label)   {ierr = LabelTest(options);CHKERRQ(ierr);}
  if (options->section) {ierr = SectionTest(options);CHKERRQ(ierr);}
  if (options->mesh)    {ierr = MeshTest(options);CHKERRQ(ierr);}
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
  ierr = RunUnitTests(&options);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
