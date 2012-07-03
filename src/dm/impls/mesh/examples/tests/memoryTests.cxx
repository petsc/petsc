static char help[] = "Sieve Package Memory Tests.\n\n";

#define ALE_HAVE_CXX_ABI
#define ALE_MEM_LOGGING
#include <petscmesh.hh>
#include <set>

#include <IField.hh>

using ALE::Obj;

typedef struct {
  PetscInt   debug;
  PetscInt   rank;
  PetscInt   size;
  // Classes
  PetscBool  sifter;      // Run the Sifter tests
  PetscBool  label;       // Run the label tests
  PetscBool  sieve;       // Run the Sieve tests
  PetscBool  mesh;        // Run the Mesh tests
  PetscBool  section;     // Run the Section tests
  PetscBool  isection;    // Run the ISection tests
  PetscBool  sectionDist; // Run the Section distribution tests
  // Run flags
  PetscInt   number;      // Number of each class to create
  // Mesh flags
  PetscInt   numCells;    // If possible, set the total number of cells
  PetscBool  interpolate; // Interpolate the mesh
  PetscReal  refine;      // The refinement limit
  // Section flags
  PetscInt   components;  // Number of section components
} Options;

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, Options *options)
{
  ALE::MemoryLogger& logger = ALE::MemoryLogger::singleton();
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  options->debug       = 0;
  options->sifter      = PETSC_FALSE;
  options->label       = PETSC_FALSE;
  options->sieve       = PETSC_FALSE;
  options->mesh        = PETSC_FALSE;
  options->section     = PETSC_FALSE;
  options->isection    = PETSC_FALSE;
  options->sectionDist = PETSC_FALSE;
  options->number      = 0;
  options->numCells    = 8;
  options->interpolate = PETSC_FALSE;
  options->refine      = 0.0;
  options->components  = 3;

  ierr = PetscOptionsBegin(comm, "", "Options for the Sieve package tests", "Sieve");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-debug", "Debugging flag", "memTests", options->debug, &options->debug, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-sifter", "Run Sifter tests", "memTests", options->sifter, &options->sifter, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-label", "Run Label tests", "memTests", options->label, &options->label, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-sieve", "Run Sieve tests", "memTests", options->sieve, &options->sieve, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-mesh", "Run Mesh tests", "memTests", options->mesh, &options->mesh, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-section", "Run Section tests", "memTests", options->section, &options->section, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-isection", "Run ISection tests", "memTests", options->isection, &options->isection, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-sectionDist", "Run Section distribution tests", "memTests", options->sectionDist, &options->sectionDist, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-num", "Number of each class to create", "memTests", options->number, &options->number, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-numCells", "Number of mesh cells", "memTests", options->numCells, &options->numCells, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-interpolate", "Interpolate the mesh", "memTests", options->interpolate, &options->interpolate, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-refine", "The refinement limit", "memTests", options->refine, &options->refine, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-components", "Number of section components", "memTests", options->components, &options->components, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  ierr = MPI_Comm_rank(comm, &options->rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &options->size);CHKERRQ(ierr);
  logger.setDebug(options->debug);
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
  const ALE::Obj<ALE::Mesh> mB = ALE::MeshBuilder<ALE::Mesh>::createSquareBoundary(PETSC_COMM_WORLD, lower, upper, edges, 0);
  mesh = ALE::Generator<ALE::Mesh>::generateMesh(mB, options->interpolate);
  if (options->refine > 0.0) {
    mesh = ALE::Generator<ALE::Mesh>::refineMesh(mesh, options->refine, options->interpolate);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "LabelTest"
PetscErrorCode LabelTest(const Options *options)
{
  ALE::MemoryLogger& logger = ALE::MemoryLogger::singleton();
  const PetscInt     num    = options->number;
  // Allocs:
  //  coneSeq Obj
  //  supportSeq Obj
  //  arrows
  const PetscInt     numAlloc = (2 + options->numCells+1)*options->number;
  const PetscInt     numBytes = (4*2+40*(options->numCells+1))*options->number;

  PetscFunctionBegin;
  logger.stagePush("Label");
  for(PetscInt i = 0; i < num; ++i) {
    ALE::LabelSifter<int, int, ALE::malloc_allocator<ALE::NewSifterDef::ArrowContainer<int, int>::traits::arrow_type> > label;

    for(PetscInt c = 0; c < options->numCells; ++c) {
      label.setCone(1, c);
    }
  }
  logger.stagePop();
  if (logger.getNumAllocations("Label") != numAlloc) {
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB, "Invalid number of allocations %d should be %d", logger.getNumAllocations("Label"), numAlloc);
  }
  if (logger.getNumDeallocations("Label") != numAlloc) {
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB, "Invalid number of deallocations %d should be %d", logger.getNumDeallocations("Label"), numAlloc);
  }
  if (logger.getAllocationTotal("Label") != numBytes) {
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB, "Invalid number of bytes allocated %d should be %d", logger.getAllocationTotal("Label"), numBytes);
  }
  if (logger.getDeallocationTotal("Label") != numBytes) {
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB, "Invalid number of bytes deallocated %d should be %d", logger.getDeallocationTotal("Label"), numBytes);
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
  //   Atlas (UniformSection) + Obj
  //     Atlas (ConstantSection) + Obj
  //       Atlas (points)
  //     Data (sizes)
  //   BC (Section) + Obj
  //     Atlas (UniformSection) + Obj
  //       Atlas (ConstantSection) + Obj
  //     Data
  //   Data
  const PetscInt     numAlloc = (12 + 2*options->numCells)*options->number;
  const PetscInt     numBytes = ((100+4)+(68+4)+20*options->numCells+28*options->numCells+(88+4)+(100+4)+(68+4)+8*options->components*options->numCells)*options->number;
  double            *values;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  logger.stagePush("Section");
  ierr = PetscMalloc(options->components * sizeof(double), &values);CHKERRQ(ierr);
  for(PetscInt c = 0; c < options->components; ++c) {values[c] = 1.0;}
  for(PetscInt i = 0; i < num; ++i) {
    ALE::GeneralSection<int, double, ALE::malloc_allocator<double> > section(PETSC_COMM_WORLD);

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
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB, "Invalid number of allocations %d should be %d", logger.getNumAllocations("Section"), numAlloc);
  }
  if (logger.getNumDeallocations("Section") != numAlloc) {
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB, "Invalid number of deallocations %d should be %d", logger.getNumDeallocations("Section"), numAlloc);
  }
  if (logger.getAllocationTotal("Section") != numBytes) {
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB, "Invalid number of bytes allocated %d should be %d", logger.getAllocationTotal("Section"), numBytes);
  }
  if (logger.getDeallocationTotal("Section") != numBytes) {
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB, "Invalid number of bytes deallocated %d should be %d", logger.getDeallocationTotal("Section"), numBytes);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ISectionTest"
PetscErrorCode ISectionTest(const Options *options)
{
  ALE::MemoryLogger& logger   = ALE::MemoryLogger::singleton();
  const PetscInt     num      = options->number;
  // Allocs:
  //   Atlas (IUniformSection) + Obj
  //     Atlas (IConstantSection) + Obj
  //       Atlas (interval)
  //     Data (sizes)
  //   BC Atlas (ISection) + Obj
  //     Atlas (IUniformSection) + Obj
  //       Atlas (IConstantSection) + Obj
  //         Atlas (interval)
  //       Data (bc sizes)
  //     Data
  //   Data
  const PetscInt     numAlloc = (14)*options->number;
  const PetscInt     numBytes = ((72+4+8*options->numCells)+(44+4)+(88+4+0)+(72+4+8*options->numCells)+(44+4)+8*options->components*options->numCells)*options->number;
  double            *values;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  logger.stagePush("ISection");
  ierr = PetscMalloc(options->components * sizeof(double), &values);CHKERRQ(ierr);
  for(PetscInt c = 0; c < options->components; ++c) {values[c] = 1.0;}
  for(PetscInt i = 0; i < num; ++i) {
    ALE::IGeneralSection<int, double, ALE::malloc_allocator<double> > section(PETSC_COMM_WORLD, 0, options->numCells);

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
  if (logger.getNumAllocations("ISection") != numAlloc) {
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB, "Invalid number of allocations %d should be %d", logger.getNumAllocations("ISection"), numAlloc);
  }
  if (logger.getNumDeallocations("ISection") != numAlloc) {
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB, "Invalid number of deallocations %d should be %d", logger.getNumDeallocations("ISection"), numAlloc);
  }
  if (logger.getAllocationTotal("ISection") != numBytes) {
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB, "Invalid number of bytes allocated %d should be %d", logger.getAllocationTotal("ISection"), numBytes);
  }
  if (logger.getDeallocationTotal("ISection") != numBytes) {
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB, "Invalid number of bytes deallocated %d should be %d", logger.getDeallocationTotal("ISection"), numBytes);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SectionDistributionTest"
PetscErrorCode SectionDistributionTest(const Options *options)
{
  typedef ALE::GeneralSection<int, double, ALE::malloc_allocator<double> > TestSection;
  ALE::MemoryLogger& logger   = ALE::MemoryLogger::singleton();
  const PetscInt     num      = options->number;
  const PetscInt     block    = options->numCells/options->size;
  const PetscInt     rank     = options->rank;
  const PetscInt     start    = rank*block     + PetscMin(rank, options->numCells%options->size);
  const PetscInt     end      = (rank+1)*block + PetscMin(rank+1, options->numCells%options->size);
  const PetscInt     numCells = options->numCells;
  const PetscInt     locCells = end - start;
  const PetscInt     remCells = numCells - locCells;
  const PetscInt     remRanks = options->size-1;
  const PetscInt     rotRanks = 1; 
 // Allocs:
  //   Atlas (UniformSection) + Obj
  //     Atlas (ConstantSection) + Obj
  //       Atlas (points)
  //     Data (sizes)
  //   BC (Section) + Obj
  //     Atlas (UniformSection) + Obj
  //       Atlas (ConstantSection) + Obj
  //     Data
  //   Data
  const PetscInt     numAlloc = (12 + 2*numCells)*options->number;
  const PetscInt     numBytes = ((100+4)+(68+4)+20*numCells+28*numCells+(88+4)+(100+4)+(68+4)+8*options->components*numCells)*options->number;
  PetscInt           numDistAlloc;
  PetscInt           numDistBytes;
  double            *values;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (!rank) {
    // There are 3 main sources of savings:
    //   1) Get rid of labels in Topology
    //   2) Get rid of base/cap in Overlap
    //   3) Share domain information between UniformSection and ConstantSection
    // We may be able to
    //   1) Share domain information between Topology and Overlap
    //   2) Share domain information between sizer and section
    // Synopsis:
    //   remRanks: 44
    //   locCells: (60+84+84)+(20+28+8*comp) = 228 (Sieve) + 48+8*comp (Section)
    //   remCells: (60+24+24)+(20+28+8*comp)+(20+28+4)+(20) = 108 (Overlap) + 48+8*comp (OSection) + 52 (Sizer) + 20 (Topology)
    numDistAlloc = (1+15+8+3*locCells+6+3*remCells+6+13+2*locCells+7+2*remCells+1+1+7+2*remCells+1+2+1+2+32+1*remCells+//50+9*remCells+
                    1*remRanks+2+23+2+2+2)*options->number+1;
    numDistBytes = (4+
                    4*3+(60+4)+(24+4)+(24+4)+(60+4)+(24+4)+(24+4)+
                    (60+4+60*locCells)+(84+4+84*locCells)+(84+4+84*locCells)+4+4+
                    (60+4+60*remCells)+(24+4+24*remCells)+(24+4+24*remCells)+(60+4)+(24+4)+(24+4)+
                    4+(100+4+28*locCells)+(68+4+20*locCells)+(88+4+0)+(100+4)+(68+4)+8*options->components*locCells+
                    4+4+(100+4+28*remCells)+(68+4+20*remCells)+8*options->components*remCells+
                    4+
                    4+
                    4+4+(100+4+28*remCells)+(68+4+20*remCells)+4*remCells+
                    4+
                    4+4+
                    4+
                    (8+4)+4+(148+60+4)+(60+24+24)+(24+4)+(24+4)+(148+60+4)+(60+24+24)+(24+4)+(24+4)+(32+4)+4+(32+4+20*remCells)+(20+4+0)+(20+4+0)+44*remRanks+
                    (8+4)+4+(148+60+4)+(60+24+24)+(24+4)+(24+4)+(148+60+4)+(60+24+24)+(24+4)+(24+4)+(32+4)+
                    (8+4)+(8+4)+(8+4))*options->number+4;
  } else {
    numDistAlloc = (1+15+8+3*locCells+6+6+3*locCells+17+4*locCells+1+7+2*locCells+1+1+7+2*locCells+1+2+2+23+2+32+1*locCells+//50+9*locCells+
                    1*rotRanks+2+2+2)*options->number+1;
    numDistBytes = (4+
                    4*3+(60+4)+(24+4)+(24+4)+(60+4)+(24+4)+(24+4)+
                    (60+4+60*locCells)+(84+4+84*locCells)+(84+4+84*locCells)+4+4+
                    (60+4)+(24+4)+(24+4)+(60+4+60*locCells)+(24+4+24*locCells)+(24+4+24*locCells)+
                    4+(100+4+28*locCells)+(68+4+20*locCells)+(88+4+0)+(100+4)+(68+4)+8*options->components*locCells+4+28*locCells+(68+4+20*locCells)+8*options->components*locCells+
                    4+
                    4+4+(100+4+28*locCells)+(68+4+20*locCells)+8*options->components*locCells+
                    4+
                    4+
                    4+4+(100+4+28*locCells)+(68+4+20*locCells)+4*locCells+
                    4+
                    4+4+
                    (8+4)+4+(148+60+4)+(60+24+24)+(24+4)+(24+4)+(148+60+4)+(60+24+24)+(24+4)+(24+4)+(32+4)+
                    (8+4)+4+(148+60+4)+(60+24+24)+(24+4)+(24+4)+(148+60+4)+(60+24+24)+(24+4)+(24+4)+(32+4)+4+(32+4+20*locCells)+(20+4+0)+(20+4+0)+44*rotRanks+
                    (8+4)+(8+4)+(8+4))*options->number+4;
  }
  ierr = PetscMalloc(options->components * sizeof(double), &values);CHKERRQ(ierr);
  for(PetscInt c = 0; c < options->components; ++c) {values[c] = 1.0;}
  logger.stagePush("SectionII");
  for(PetscInt i = 0; i < num; ++i) {
    TestSection section(PETSC_COMM_WORLD);

    for(PetscInt c = 0; c < numCells; ++c) {
      section.setFiberDimension(c, options->components);
    }
    section.allocatePoint();
    for(PetscInt c = 0; c < numCells; ++c) {
      section.updatePoint(c, values);
    }
    logger.stagePush("Distribution");
    {
      // Allocs:
      //   Section Obj
      //   Mesh Obj
      //     indexArray Obj
      //     modifiedPoints Obj
      //     numberingFactory Obj (only once)
      //     Send Overlap + Obj
      //       Base + Obj
      //       Cap  + Obj
      //     Recv Overlap + Obj
      //       Base + Obj
      //       Cap  + Obj
      //   Sieve + Obj
      //     Base (just one cell) + Obj
      //     Cap (points) + Obj
      //     Markers Obj
      //     ConeSet Obj
      //     Data (arrows)
      //   Send Overlap + Obj
      //     Base + Obj
      //       Data (remote ranks)
      //     Cap  + Obj
      //       Data (remote points)
      //     Data (arrows)
      //   Recv Overlap + Obj
      //     Base + Obj
      //       Data (remote points)
      //     Cap  + Obj
      //       Data (remote ranks)
      //     Data (arrows)
      //   Parallel Section Obj
      //     Atlas (UniformSection) + Obj
      //       Atlas (ConstantSection) + Obj
      //         Atlas (points)
      //       Data (sizes)
      //     BC (Section) + Obj
      //       Atlas (UniformSection) + Obj
      //         Atlas (ConstantSection) + Obj
      //       Data
      //     Data
      //     NewAtlas Obj only for p_k
      //       Atlas (ConstantSection) + Obj
      //         Data (points)
      //       Data (sizes)
      //     NewData only for p_k
      //   Send Section Obj
      //     Section Obj only p_0
      //       Atlas (UniformSection) + Obj
      //         Atlas (ConstantSection) + Obj
      //           Data (point)
      //         Data (sizes)
      //       Data (8*comp*points)
      //   Recv Section Obj
      //     Section Obj only p_k
      //       Atlas (UniformSection) + Obj
      //         Atlas (ConstantSection) + Obj
      //           Data (point)
      //         Data (sizes)
      //       Data (8*comp*points)
      //   Sizer Obj (sizerFiller)
      //   Send Sizer Obj
      //     Section Obj only p_0
      //       Atlas (UniformSection) + Obj
      //         Atlas (ConstantSection) + Obj
      //           Data (point)
      //         Data (sizes)
      //       Data (4*points)
      //   Recv Sizer Obj
      //     Section Obj only p_k
      //       Atlas (UniformSection) + Obj
      //         Atlas (ConstantSection) + Obj
      //           Data (point)
      //         Data (sizes)
      //       Data (4*points)
      //   Const Send Sizer Obj
      //     Section Obj only p_0
      //   Const Recv Sizer Obj
      //     Section Obj only p_k
      //   BaseSequence + Obj
      //   SendTopology Obj
      //     Send Overlap + Obj (148+60)
      //       LOOKS LIKE AN ARROW IS INSERTED THEN DELETED TO START
      //       Base + Obj
      //       Cap  + Obj
      //     Recv Overlap + Obj (148+60)
      //       LOOKS LIKE AN ARROW IS INSERTED THEN DELETED TO START
      //       Base + Obj
      //       Cap  + Obj
      //     modifiedPoints (32) + Obj
      //       Data (points) 20
      //     DiscreteSieve Obj only p_0
      //       Domain (points) 32 + Obj
      //         Data (sendPoints) 20
      //       EmptySeq 20 + Obj
      //         Data (0)
      //       ReturnSeq 20 + Obj
      //         Data (0)
      //     Data (sieves) 44
      //     50+44+24 for height label
      //     Height Label + Obj only p_0
      //       Base + Obj
      //         Data (height)
      //       Cap  + Obj
      //         Data (points)
      //       Data (arrows)
      //     50+44+24 for depth label
      //     Depth Label + Obj only p_0
      //       Base + Obj
      //         Data (height)
      //       Cap  + Obj
      //         Data (points)
      //       Data (arrows)
      //   CapSequence + Obj
      //   RecvTopology Obj
      //     Send Overlap + Obj (148+60)
      //       LOOKS LIKE AN ARROW IS INSERTED THEN DELETED TO START
      //       Base + Obj
      //       Cap  + Obj
      //     Recv Overlap + Obj (148+60)
      //       LOOKS LIKE AN ARROW IS INSERTED THEN DELETED TO START
      //       Base + Obj
      //       Cap  + Obj
      //     modifiedPoints (32) + Obj
      //       Data (points) 20
      //     DiscreteSieve Obj only p_k
      //       Domain (points) 32 + Obj
      //         Data (sendPoints) 20
      //       EmptySeq 20 + Obj
      //         Data (0)
      //       ReturnSeq 20 + Obj
      //         Data (0)
      //     Data (sieves) 44 only p_k
      //     50+44+24 for height label
      //     Height Label + Obj only p_k
      //       Base + Obj
      //         Data (height)
      //       Cap  + Obj
      //         Data (points)
      //       Data (arrows)
      //     50+44+24 for depth label
      //     Depth Label + Obj only p_k
      //       Base + Obj
      //         Data (height)
      //       Cap  + Obj
      //         Data (points)
      //       Data (arrows)
      //   CapSequence + Obj (for setupReceive() sizes)
      //   CapSequence + Obj (for setupReceive() data)
      //   BaseSequence + Obj (for updateRemote())
      Obj<TestSection>                  objSection(&section);
      objSection.addRef();
      Obj<ALE::Mesh>                    parallelMesh = new ALE::Mesh(PETSC_COMM_WORLD, 1);
      Obj<ALE::Mesh::sieve_type>        sieve        = new ALE::Mesh::sieve_type(PETSC_COMM_WORLD);
      Obj<ALE::Mesh::send_overlap_type> sendOverlap  = new ALE::Mesh::send_overlap_type(PETSC_COMM_WORLD);
      Obj<ALE::Mesh::recv_overlap_type> recvOverlap  = new ALE::Mesh::recv_overlap_type(PETSC_COMM_WORLD);

      parallelMesh->setSieve(sieve);
      if (!rank) {
        for(PetscInt r = 1; r < section.commSize(); ++r) {
          const PetscInt rStart = r*block     + PetscMin(r, numCells%options->size);
          const PetscInt rEnd   = (r+1)*block + PetscMin(r+1, numCells%options->size);

          for(PetscInt c = rStart; c < rEnd; ++c) {
            sendOverlap->addArrow(c, r, c);
          }
        }
      }
      for(PetscInt c = start; c < end; ++c) {
        sieve->addCone(c, -(rank+1));
        if (rank) {recvOverlap->addArrow(0, c, c);}
      }
      // This implies that distribution should be templated over all the arguments
      //   and not take so much from the Bundle
      Obj<TestSection> distSection = ALE::Distribution<ALE::Mesh>::distributeSection(objSection, parallelMesh, sendOverlap, recvOverlap);
    }
    logger.stagePop();
  }
  logger.stagePop();
  ierr = PetscFree(values);CHKERRQ(ierr);
  if (logger.getNumAllocations("SectionII") != numAlloc) {
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB, "Invalid number of allocations %d should be %d", logger.getNumAllocations("SectionII"), numAlloc);
  }
  if (logger.getNumDeallocations("SectionII") != numAlloc) {
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB, "Invalid number of deallocations %d should be %d", logger.getNumDeallocations("SectionII"), numAlloc);
  }
  if (logger.getAllocationTotal("SectionII") != numBytes) {
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB, "Invalid number of bytes allocated %d should be %d", logger.getAllocationTotal("SectionII"), numBytes);
  }
  if (logger.getDeallocationTotal("SectionII") != numBytes) {
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB, "Invalid number of bytes deallocated %d should be %d", logger.getDeallocationTotal("SectionII"), numBytes);
  }
  if (logger.getNumAllocations("Distribution") != numDistAlloc) {
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB, "Invalid number of allocations %d should be %d", logger.getNumAllocations("Distribution"), numDistAlloc);
  }
  if (logger.getNumDeallocations("Distribution") != numDistAlloc-options->number-1) {
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB, "Invalid number of deallocations %d should be %d", logger.getNumDeallocations("Distribution"), numDistAlloc);
  }
  if (logger.getAllocationTotal("Distribution") != numDistBytes) {
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB, "Invalid number of bytes allocated %d should be %d", logger.getAllocationTotal("Distribution"), numDistBytes);
  }
  if (logger.getDeallocationTotal("Distribution") != numDistBytes-4*options->number-4) {
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB, "Invalid number of bytes deallocated %d should be %d", logger.getDeallocationTotal("Distribution"), numDistBytes);
  }
  PetscFunctionReturn(0);
}

#if 0
#undef __FUNCT__
#define __FUNCT__ "ISectionDistributionTest"
PetscErrorCode ISectionDistributionTest(const Options *options)
{
  typedef ALE::IUniformSection<int, double, 1, ALE::malloc_allocator<double> > TestSection;
  ALE::MemoryLogger& logger   = ALE::MemoryLogger::singleton();
  const PetscInt     num      = options->number;
  const PetscInt     block    = options->numCells/options->size;
  const PetscInt     rank     = options->rank;
  const PetscInt     start    = rank*block     + PetscMin(rank, options->numCells%options->size);
  const PetscInt     end      = (rank+1)*block + PetscMin(rank+1, options->numCells%options->size);
  const PetscInt     numCells = options->numCells;
  const PetscInt     locCells = end - start;
  const PetscInt     remCells = numCells - locCells;
  const PetscInt     remRanks = options->size-1;
  const PetscInt     rotRanks = 1; 
  PetscInt           numAlloc;
  PetscInt           numBytes;
  double            *values;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (!rank) {
    // There are 3 main sources of savings:
    //   1) Get rid of labels in Topology
    //   2) Get rid of base/cap in Overlap
    //   3) Share domain information between UniformSection and ConstantSection
    // We may be able to
    //   1) Share domain information between Topology and Overlap
    //   2) Share domain information between sizer and section
    // Synopsis:
    //   remRanks: 44
    //   locCells: (60+84+84)+(20+28+8*comp) = 228 (Sieve) + 48+8*comp (Section)
    //   remCells: (60+24+24)+(20+28+8*comp)+(20+28+4)+(20) = 108 (Overlap) + 48+8*comp (OSection) + 52 (Sizer) + 20 (Topology)
    numAlloc = (1+15+8+3*locCells+6+3*remCells+6+13+2*locCells+7+2*remCells+1+1+7+2*remCells+1+2+1+2+32+1*remCells+//50+9*remCells+
                1*remRanks+2+23+2+2+2)*options->number+1;
    numBytes = (4+
                4*3+(60+4)+(24+4)+(24+4)+(60+4)+(24+4)+(24+4)+
                (60+4+60*locCells)+(84+4+84*locCells)+(84+4+84*locCells)+4+4+
                (60+4+60*remCells)+(24+4+24*remCells)+(24+4+24*remCells)+(60+4)+(24+4)+(24+4)+
                4+(100+4+28*locCells)+(68+4+20*locCells)+(88+4+0)+(100+4)+(68+4)+8*options->components*locCells+
                4+4+(100+4+28*remCells)+(68+4+20*remCells)+8*options->components*remCells+
                4+
                4+
                4+4+(100+4+28*remCells)+(68+4+20*remCells)+4*remCells+
                4+
                4+4+
                4+
                (8+4)+4+(148+60+4)+(60+24+24)+(24+4)+(24+4)+(148+60+4)+(60+24+24)+(24+4)+(24+4)+(32+4)+4+(32+4+20*remCells)+(20+4+0)+(20+4+0)+44*remRanks+
                (8+4)+4+(148+60+4)+(60+24+24)+(24+4)+(24+4)+(148+60+4)+(60+24+24)+(24+4)+(24+4)+(32+4)+
                (8+4)+(8+4)+(8+4))*options->number+4;
  } else {
    numAlloc = (1+15+8+3*locCells+6+6+3*locCells+17+4*locCells+1+7+2*locCells+1+1+7+2*locCells+1+2+2+23+2+32+1*locCells+//50+9*locCells+
                1*rotRanks+2+2+2)*options->number+1;
    numBytes = (4+
                4*3+(60+4)+(24+4)+(24+4)+(60+4)+(24+4)+(24+4)+
                (60+4+60*locCells)+(84+4+84*locCells)+(84+4+84*locCells)+4+4+
                (60+4)+(24+4)+(24+4)+(60+4+60*locCells)+(24+4+24*locCells)+(24+4+24*locCells)+
                4+(100+4+28*locCells)+(68+4+20*locCells)+(88+4+0)+(100+4)+(68+4)+8*options->components*locCells+4+28*locCells+(68+4+20*locCells)+8*options->components*locCells+
                4+
                4+4+(100+4+28*locCells)+(68+4+20*locCells)+8*options->components*locCells+
                4+
                4+
                4+4+(100+4+28*locCells)+(68+4+20*locCells)+4*locCells+
                4+
                4+4+
                (8+4)+4+(148+60+4)+(60+24+24)+(24+4)+(24+4)+(148+60+4)+(60+24+24)+(24+4)+(24+4)+(32+4)+
                (8+4)+4+(148+60+4)+(60+24+24)+(24+4)+(24+4)+(148+60+4)+(60+24+24)+(24+4)+(24+4)+(32+4)+4+(32+4+20*locCells)+(20+4+0)+(20+4+0)+44*rotRanks+
                (8+4)+(8+4)+(8+4))*options->number+4;
  }
  ierr = PetscMalloc(options->components * sizeof(double), &values);CHKERRQ(ierr);
  for(PetscInt c = 0; c < options->components; ++c) {values[c] = 1.0;}
  logger.stagePush("SectionII");
  for(PetscInt i = 0; i < num; ++i) {
    TestSection section(PETSC_COMM_WORLD, 0, options->numCells);

    for(PetscInt c = 0; c < numCells; ++c) {
      section.setFiberDimension(c, options->components);
    }
    section.allocatePoint();
    for(PetscInt c = 0; c < numCells; ++c) {
      section.updatePoint(c, values);
    }
    logger.stagePush("Distribution");
    {
      // Allocs:
      //   Section Obj
      //   Mesh Obj
      //     indexArray Obj
      //     modifiedPoints Obj
      //     numberingFactory Obj (only once)
      //     Send Overlap + Obj
      //       Base + Obj
      //       Cap  + Obj
      //     Recv Overlap + Obj
      //       Base + Obj
      //       Cap  + Obj
      //   Sieve + Obj
      //     Base (just one cell) + Obj
      //     Cap (points) + Obj
      //     Markers Obj
      //     ConeSet Obj
      //     Data (arrows)
      //   Send Overlap + Obj
      //     Base + Obj
      //       Data (remote ranks)
      //     Cap  + Obj
      //       Data (remote points)
      //     Data (arrows)
      //   Recv Overlap + Obj
      //     Base + Obj
      //       Data (remote points)
      //     Cap  + Obj
      //       Data (remote ranks)
      //     Data (arrows)
      //   Parallel Section Obj
      //     Atlas (UniformSection) + Obj
      //       Atlas (ConstantSection) + Obj
      //         Atlas (points)
      //       Data (sizes)
      //     BC (Section) + Obj
      //       Atlas (UniformSection) + Obj
      //         Atlas (ConstantSection) + Obj
      //       Data
      //     Data
      //     NewAtlas Obj only for p_k
      //       Atlas (ConstantSection) + Obj
      //         Data (points)
      //       Data (sizes)
      //     NewData only for p_k
      //   Send Section Obj
      //     Section Obj only p_0
      //       Atlas (UniformSection) + Obj
      //         Atlas (ConstantSection) + Obj
      //           Data (point)
      //         Data (sizes)
      //       Data (8*comp*points)
      //   Recv Section Obj
      //     Section Obj only p_k
      //       Atlas (UniformSection) + Obj
      //         Atlas (ConstantSection) + Obj
      //           Data (point)
      //         Data (sizes)
      //       Data (8*comp*points)
      //   Sizer Obj (sizerFiller)
      //   Send Sizer Obj
      //     Section Obj only p_0
      //       Atlas (UniformSection) + Obj
      //         Atlas (ConstantSection) + Obj
      //           Data (point)
      //         Data (sizes)
      //       Data (4*points)
      //   Recv Sizer Obj
      //     Section Obj only p_k
      //       Atlas (UniformSection) + Obj
      //         Atlas (ConstantSection) + Obj
      //           Data (point)
      //         Data (sizes)
      //       Data (4*points)
      //   Const Send Sizer Obj
      //     Section Obj only p_0
      //   Const Recv Sizer Obj
      //     Section Obj only p_k
      //   BaseSequence + Obj
      //   SendTopology Obj
      //     Send Overlap + Obj (148+60)
      //       LOOKS LIKE AN ARROW IS INSERTED THEN DELETED TO START
      //       Base + Obj
      //       Cap  + Obj
      //     Recv Overlap + Obj (148+60)
      //       LOOKS LIKE AN ARROW IS INSERTED THEN DELETED TO START
      //       Base + Obj
      //       Cap  + Obj
      //     modifiedPoints (32) + Obj
      //       Data (points) 20
      //     DiscreteSieve Obj only p_0
      //       Domain (points) 32 + Obj
      //         Data (sendPoints) 20
      //       EmptySeq 20 + Obj
      //         Data (0)
      //       ReturnSeq 20 + Obj
      //         Data (0)
      //     Data (sieves) 44
      //     50+44+24 for height label
      //     Height Label + Obj only p_0
      //       Base + Obj
      //         Data (height)
      //       Cap  + Obj
      //         Data (points)
      //       Data (arrows)
      //     50+44+24 for depth label
      //     Depth Label + Obj only p_0
      //       Base + Obj
      //         Data (height)
      //       Cap  + Obj
      //         Data (points)
      //       Data (arrows)
      //   CapSequence + Obj
      //   RecvTopology Obj
      //     Send Overlap + Obj (148+60)
      //       LOOKS LIKE AN ARROW IS INSERTED THEN DELETED TO START
      //       Base + Obj
      //       Cap  + Obj
      //     Recv Overlap + Obj (148+60)
      //       LOOKS LIKE AN ARROW IS INSERTED THEN DELETED TO START
      //       Base + Obj
      //       Cap  + Obj
      //     modifiedPoints (32) + Obj
      //       Data (points) 20
      //     DiscreteSieve Obj only p_k
      //       Domain (points) 32 + Obj
      //         Data (sendPoints) 20
      //       EmptySeq 20 + Obj
      //         Data (0)
      //       ReturnSeq 20 + Obj
      //         Data (0)
      //     Data (sieves) 44 only p_k
      //     50+44+24 for height label
      //     Height Label + Obj only p_k
      //       Base + Obj
      //         Data (height)
      //       Cap  + Obj
      //         Data (points)
      //       Data (arrows)
      //     50+44+24 for depth label
      //     Depth Label + Obj only p_k
      //       Base + Obj
      //         Data (height)
      //       Cap  + Obj
      //         Data (points)
      //       Data (arrows)
      //   CapSequence + Obj (for setupReceive() sizes)
      //   CapSequence + Obj (for setupReceive() data)
      //   BaseSequence + Obj (for updateRemote())
      Obj<TestSection>                  objSection(&section);
      objSection.addRef();
      Obj<ALE::Mesh>                    parallelMesh = new ALE::Mesh(PETSC_COMM_WORLD, 1);
      Obj<ALE::Mesh::sieve_type>        sieve        = new ALE::Mesh::sieve_type(PETSC_COMM_WORLD);
      Obj<ALE::Mesh::send_overlap_type> sendOverlap  = new ALE::Mesh::send_overlap_type(PETSC_COMM_WORLD);
      Obj<ALE::Mesh::recv_overlap_type> recvOverlap  = new ALE::Mesh::recv_overlap_type(PETSC_COMM_WORLD);

      parallelMesh->setSieve(sieve);
      if (!rank) {
        for(PetscInt r = 1; r < section.commSize(); ++r) {
          const PetscInt rStart = r*block     + PetscMin(r, numCells%options->size);
          const PetscInt rEnd   = (r+1)*block + PetscMin(r+1, numCells%options->size);

          for(PetscInt c = rStart; c < rEnd; ++c) {
            sendOverlap->addArrow(c, r, c);
          }
        }
      }
      for(PetscInt c = start; c < end; ++c) {
        sieve->addCone(c, -(rank+1));
        if (rank) {recvOverlap->addArrow(0, c, c);}
      }
      // This implies that distribution should be templated over all the arguments
      //   and not take so much from the Bundle
      Obj<TestSection> distSection = ALE::Distribution<ALE::Mesh>::distributeSection(objSection, parallelMesh, sendOverlap, recvOverlap);
    }
    logger.stagePop();
  }
  logger.stagePop();
  ierr = PetscFree(values);CHKERRQ(ierr);
  if (logger.getNumAllocations("Distribution") != numAlloc) {
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB, "Invalid number of allocations %d should be %d", logger.getNumAllocations("Distribution"), numAlloc);
  }
  if (logger.getNumDeallocations("Distribution") != numAlloc) {
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB, "Invalid number of deallocations %d should be %d", logger.getNumDeallocations("Distribution"), numAlloc);
  }
  if (logger.getAllocationTotal("Distribution") != numBytes) {
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB, "Invalid number of bytes allocated %d should be %d", logger.getAllocationTotal("Distribution"), numBytes);
  }
  if (logger.getDeallocationTotal("Distribution") != numBytes) {
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB, "Invalid number of bytes deallocated %d should be %d", logger.getDeallocationTotal("Distribution"), numBytes);
  }
  PetscFunctionReturn(0);
}
#endif

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
#define __FUNCT__ "RunUnitTests"
PetscErrorCode RunUnitTests(const Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (options->label)       {ierr = LabelTest(options);CHKERRQ(ierr);}
  if (options->section)     {ierr = SectionTest(options);CHKERRQ(ierr);}
  if (options->isection)    {ierr = ISectionTest(options);CHKERRQ(ierr);}
  if (options->sectionDist) {ierr = SectionDistributionTest(options);CHKERRQ(ierr);}
  if (options->mesh)        {ierr = MeshTest(options);CHKERRQ(ierr);}
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
  ierr = PetscFinalize();
  PetscFunctionReturn(0);
}
