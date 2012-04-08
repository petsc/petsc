/*T
   Concepts: Mesh^loading a mesh
   Concepts: Mesh^partitioning a mesh
   Concepts: Mesh^viewing a mesh
   Processors: n
T*/

/*
  Read in a mesh using the PCICE format:

  connectivity file:
  ------------------
  NumCells
  Cell #   v_0 v_1 ... v_d
  .
  .
  .

  coordinate file:
  ----------------
  NumVertices
  Vertex #  x_0 x_1 ... x_{d-1}
  .
  .
  .

Partition the mesh and distribute it to each process.

Output the mesh in VTK format with a scalar field indicating
the rank of the process owning each cell.
*/

static char help[] = "Reads, partitions, and outputs an unstructured mesh.\n\n";

#include <petscdmmesh.hh>
//#include <petscmesh_formats.hh>

typedef enum {PCICE} FileType;

typedef struct {
  int            debug;              // The debugging level
  PetscInt       dim;                // The topological mesh dimension
  PetscBool      useZeroBase;        // Use zero-based indexing
  FileType       inputFileType;      // The input file type, e.g. PCICE
  FileType       outputFileType;     // The output file type, e.g. PCICE
  char           baseFilename[2048]; // The base filename for mesh files
  PetscBool      output;             // Output the mesh
  PetscBool      outputLocal;        // Output the local form of the mesh
  PetscBool      outputVTK;          // Output the mesh in VTK
  PetscBool      distribute;         // Distribute the mesh among processes
  char           partitioner[2048];  // The partitioner name
  PetscBool      interpolate;        // Construct missing elements of the mesh
  PetscBool      doPartition;        // Construct field over cells indicating process number
  SectionInt     partition;          // Section with partition number in each cell
  PetscBool      doOdd;              // Construct field over odd cells indicating process number
  SectionInt     odd;                // Section with cell number in each odd cell
} Options;

#undef __FUNCT__
#define __FUNCT__ "OutputVTK"
PetscErrorCode OutputVTK(DM mesh, Options *options)
{
  MPI_Comm       comm;
  PetscViewer    viewer;
  PetscLogStage  stage;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (options->outputVTK) {
    ierr = PetscLogStageRegister("VTKOutput", &stage);CHKERRQ(ierr);
    ierr = PetscLogStagePush(stage);CHKERRQ(ierr);
    ierr = PetscObjectGetComm((PetscObject) mesh, &comm);CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "Creating VTK mesh file\n");CHKERRQ(ierr);
    ierr = PetscViewerCreate(comm, &viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer, PETSCVIEWERASCII);CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer, "testMesh.vtk");CHKERRQ(ierr);
    ierr = DMView(mesh, viewer);CHKERRQ(ierr);
    if (options->doPartition) {
      ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_VTK_CELL);CHKERRQ(ierr);
      ierr = SectionIntView(options->partition, viewer);CHKERRQ(ierr);
      ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    }
    if (options->odd) {
      ierr = PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_VTK_CELL);CHKERRQ(ierr);
      ierr = SectionIntView(options->odd, viewer);CHKERRQ(ierr);
      ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    }
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = PetscLogStagePop();CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "OutputMesh"
PetscErrorCode OutputMesh(DM mesh, Options *options)
{
  MPI_Comm       comm;
  PetscViewer    viewer;
  PetscLogStage  stage;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (options->output) {
    ierr = PetscLogStageRegister("MeshOutput", &stage);CHKERRQ(ierr);
    ierr = PetscLogStagePush(stage);CHKERRQ(ierr);
    ierr = PetscObjectGetComm((PetscObject) mesh, &comm);CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "Creating original format mesh file\n");CHKERRQ(ierr);
    ierr = PetscViewerCreate(comm, &viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer, PETSCVIEWERASCII);CHKERRQ(ierr);
    if (options->outputFileType == PCICE) {
      ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_PCICE);CHKERRQ(ierr);
      ierr = PetscViewerFileSetName(viewer, "testMesh.lcon");CHKERRQ(ierr);
    }
    ierr = DMView(mesh, viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = PetscLogStagePop();CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreatePartition"
/* Creates a field whose value is the processor rank on each element */
PetscErrorCode CreatePartition(DM mesh, SectionInt *partition)
{
  PetscInt       cStart, cEnd;
  PetscLogEvent  event;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(((PetscObject) mesh)->comm, &rank);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("CreatePartition", PETSC_OBJECT_CLASSID, &event);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(event,0,0,0,0);CHKERRQ(ierr);
  ierr = DMMeshGetCellSectionInt(mesh, "partition", 1, partition);CHKERRQ(ierr);
  ierr = DMMeshGetHeightStratum(mesh, 0, &cStart, &cEnd);CHKERRQ(ierr);
  for(PetscInt c = cStart; c < cEnd; ++c) {
    ierr = SectionIntUpdate(*partition, c, &rank, INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(event,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DistributeMesh"
PetscErrorCode DistributeMesh(Options *options, DM *mesh)
{
  PetscMPIInt    size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(((PetscObject) *mesh)->comm, &size);CHKERRQ(ierr);
  if (options->distribute && size > 1) {
    PetscLogStage stage;
    DM            parallelMesh;

    ierr = PetscLogStageRegister("MeshDistribution", &stage);CHKERRQ(ierr);
    ierr = PetscLogStagePush(stage);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Distributing mesh\n");CHKERRQ(ierr);
    ierr = DMMeshDistribute(*mesh, options->partitioner, &parallelMesh);CHKERRQ(ierr);
    ierr = PetscLogStagePop();CHKERRQ(ierr);
    ierr = DMDestroy(mesh);CHKERRQ(ierr);
    *mesh = parallelMesh;
  }
  if (options->doPartition) {
    ierr = CreatePartition(*mesh, &options->partition);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateOdd"
/* Creates a field whose value is the element number on each element with an odd number */
PetscErrorCode CreateOdd(DM mesh, Options *options, SectionInt *odd)
{
  PetscInt       cStart, cEnd;
  PetscLogEvent  event;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventRegister("CreateOdd", PETSC_OBJECT_CLASSID, &event);CHKERRQ(ierr);
  ierr = PetscLogEventBegin(event,0,0,0,0);CHKERRQ(ierr);
  ierr = SectionIntCreate(((PetscObject) mesh)->comm, odd);CHKERRQ(ierr);
  const Obj<PETSC_MESH_TYPE::int_section_type>& section = new PETSC_MESH_TYPE::int_section_type(((PetscObject) mesh)->comm, options->debug);

  ierr = DMMeshGetHeightStratum(mesh, 0, &cStart, &cEnd);CHKERRQ(ierr);
  for(PetscInt c = cStart; c < cEnd; ++c) {
    if (c%2) {
      section->setFiberDimension(c, c%3+1);
    }
  }
  section->allocatePoint();
  for(PetscInt c = cStart; c < cEnd; ++c) {
    int val[3];

    if (c%2) {
      for(int n = 0; n <= c%3; n++) val[n] = c+n;
      section->updatePoint(c, val);
    }
  }
  ierr = SectionIntSetSection(*odd, section);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(event,0,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateMesh"
PetscErrorCode CreateMesh(MPI_Comm comm, Options *options, DM *mesh)
{
  PetscLogStage  stage;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLogStageRegister("MeshCreation", &stage);CHKERRQ(ierr);
  ierr = PetscLogStagePush(stage);CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "Creating mesh\n");CHKERRQ(ierr);
  ierr = DMCreate(comm, mesh);CHKERRQ(ierr);
  if (options->inputFileType == PCICE) {
    char coordFile[2048];
    char adjFile[2048];

    ierr = PetscStrcpy(coordFile, options->baseFilename);CHKERRQ(ierr);
    ierr = PetscStrcat(coordFile, ".nodes");CHKERRQ(ierr);
    ierr = PetscStrcpy(adjFile,   options->baseFilename);CHKERRQ(ierr);
    ierr = PetscStrcat(adjFile,   ".lcon");CHKERRQ(ierr);
    ierr = DMMeshCreatePCICE(comm, options->dim, coordFile, adjFile, options->interpolate, PETSC_NULL, mesh);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid mesh input type: %d", options->inputFileType);
  }
  ierr = PetscLogStagePop();CHKERRQ(ierr);
  ierr = DMView(*mesh, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  if (options->debug) {
    ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_ASCII_INFO_DETAIL);CHKERRQ(ierr);
    ierr = DMView(*mesh, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  if (options->doOdd) {
    ierr = CreateOdd(*mesh, options, &options->odd);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "CreateMeshBoundary"
PetscErrorCode CreateMeshBoundary(DM mesh, Options *options)
{
  typedef ALE::ISieveVisitor::PointRetriever<PETSC_MESH_TYPE::sieve_type> Retriever;
  ALE::Obj<PETSC_MESH_TYPE> m;
  MPI_Comm       comm   = ((PetscObject) mesh)->comm;
  PetscInt       dim    = options->dim;
  PetscInt       debug  = options->debug;
  PetscInt       numCells, numBC, bc;
  char           bndfilename[2048];
  FILE          *fp;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (dim != 3) {PetscFunctionReturn(0);};
  ierr = DMMeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = DMMeshGetHeightStratum(mesh, 0, PETSC_NULL, &numCells);CHKERRQ(ierr);
  ierr = PetscStrcpy(bndfilename, options->baseFilename);CHKERRQ(ierr);
  ierr = PetscStrcat(bndfilename, ".bnd");CHKERRQ(ierr);
  ierr = PetscFOpen(comm, bndfilename, "r", &fp);
  if (ierr == PETSC_ERR_FILE_OPEN) {
    PetscFunctionReturn(0);
  } else {CHKERRQ(ierr);}
  ierr = PetscPrintf(comm, "Creating mesh boundary on CPU 0\n");CHKERRQ(ierr);
  const Obj<PETSC_MESH_TYPE::real_section_type>& coordinates = m->getRealSection("coordinates");

  if (!m->commRank()) {ierr = fscanf(fp, "%d\n", &numBC);CHKERRQ(!ierr);}
  ierr = MPI_Bcast(&numBC, 1, MPIU_INT, 0, comm);CHKERRQ(ierr);
  for(bc = 0; bc < numBC; ++bc) {
    Obj<PETSC_MESH_TYPE::label_type> label;
    char     bdName[2048];
    size_t   len      = 0;
    PetscInt bcType   = 0;
    PetscInt numFaces = 0, f;

    if (!m->commRank()) {
      ierr = fscanf(fp, "\n%s\n", bdName);CHKERRQ(!ierr);
      ierr = PetscStrlen(bdName, &len);CHKERRQ(ierr);
    }
    ierr = MPI_Bcast(&len, 1, MPIU_INT, 0, comm);CHKERRQ(ierr);
    ierr = MPI_Bcast(bdName, len+1, MPI_CHAR, 0, comm);CHKERRQ(ierr);
    if (m->hasLabel(bdName)) {
      label = m->getLabel(bdName);
    } else {
      label = m->createLabel(bdName);
    }

    if (!m->commRank()) {ierr = fscanf(fp, "%d %d\n", &bcType, &numFaces);CHKERRQ(!ierr);}

    for(f = 0; f < numFaces; ++f) {
      Retriever visitor(1);
      PetscInt  numCorners, c;

      ierr = fscanf(fp, "%d", &numCorners);CHKERRQ(!ierr);
      PetscInt face[numCorners];

      if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "    Face %d with %d corners\n", f, numCorners);CHKERRQ(ierr);}
      for(c = 0; c < numCorners; ++c) {
        ierr = fscanf(fp, " %d", &face[c]);CHKERRQ(!ierr);

        // Must transform from vertex numbering to sieve point numbering
        face[c] += numCells;
        // Output vertex coordinates
        const double *coords   = coordinates->restrictPoint(face[c]);
        const int     fiberDim = coordinates->getFiberDimension(face[c]);

        if (debug) {
          ierr = PetscPrintf(PETSC_COMM_SELF, "      (");CHKERRQ(ierr);
          for(PetscInt d = 0; d < fiberDim; ++d) {
            ierr = PetscPrintf(PETSC_COMM_SELF, "%g ", coords[d]);CHKERRQ(ierr);
          }
          ierr = PetscPrintf(PETSC_COMM_SELF, ") %d\n", face[c]);CHKERRQ(ierr);
        }
      }
      m->getSieve()->nJoin(&face[0], &face[numCorners], dim-1, visitor);
      if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "      found %d faces %d using join\n", visitor.getSize(), visitor.getPoints()[0]);CHKERRQ(ierr);}
      if (visitor.getSize() != 1) {
        SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Mesh did not have a unique face for vertices, had %d faces", visitor.getSize());
      }
      m->setValue(label, visitor.getPoints()[0], bcType);
      visitor.clear();
    }
    label->view((const char *) bdName);
  }
  ierr = PetscFClose(comm, fp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, Options *options)
{
  const char    *fileTypes[2] = {"pcice", "pylith"};
  PetscInt       inputFt, outputFt;
  PetscBool      setOutputType;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->debug          = 0;
  options->dim            = 2;
  options->useZeroBase    = PETSC_TRUE;
  options->inputFileType  = PCICE;
  options->outputFileType = PCICE;
  ierr = PetscStrcpy(options->baseFilename, "src/dm/impls/mesh/examples/tutorials/data/ex1_2d");CHKERRQ(ierr);
  options->output         = PETSC_FALSE;
  options->outputLocal    = PETSC_FALSE;
  options->outputVTK      = PETSC_TRUE;
  options->distribute     = PETSC_TRUE;
  ierr = PetscStrcpy(options->partitioner, "chaco");CHKERRQ(ierr);
  options->interpolate    = PETSC_TRUE;
  options->doPartition    = PETSC_TRUE;
  options->partition      = PETSC_NULL;
  options->doOdd          = PETSC_FALSE;
  options->odd            = PETSC_NULL;

  inputFt  = (PetscInt) options->inputFileType;
  outputFt = (PetscInt) options->outputFileType;

  ierr = PetscOptionsBegin(comm, "", "Options for mesh loading", "Options");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-debug", "The debugging level", "ex1.c", options->debug, &options->debug, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex1.c", options->dim, &options->dim, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-use_zero_base", "Use zero-based indexing", "ex1.c", options->useZeroBase, &options->useZeroBase, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEList("-file_type", "Type of input files", "ex1.c", fileTypes, 2, fileTypes[0], &inputFt, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsEList("-output_file_type", "Type of output files", "ex1.c", fileTypes, 2, fileTypes[0], &outputFt, &setOutputType);CHKERRQ(ierr);
    ierr = PetscOptionsString("-base_file", "The base filename for mesh files", "ex1.c", options->baseFilename, options->baseFilename, 2048, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-output", "Output the mesh", "ex1.c", options->output, &options->output, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-output_local", "Output the local form of the mesh", "ex1.c", options->outputLocal, &options->outputLocal, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-output_vtk", "Output the mesh in VTK", "ex1.c", options->outputVTK, &options->outputVTK, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-distribute", "Distribute the mesh among processes", "ex1.c", options->distribute, &options->distribute, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsString("-partitioner", "The partitioner name", "ex1.c", options->partitioner, options->partitioner, 2048, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-interpolate", "Construct missing elements of the mesh", "ex1.c", options->interpolate, &options->interpolate, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-partition", "Create the partition field", "ex1.c", options->doPartition, &options->doPartition, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-odd", "Create the odd field", "ex1.c", options->doOdd, &options->doOdd, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  options->inputFileType = (FileType) inputFt;
  if (setOutputType) {
    options->outputFileType = (FileType) outputFt;
  } else {
    options->outputFileType = options->inputFileType;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  MPI_Comm       comm;
  DM             mesh;
  Options        options;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr);
  comm = PETSC_COMM_WORLD;
  ierr = ProcessOptions(comm, &options);CHKERRQ(ierr);
  ierr = CreateMesh(comm, &options, &mesh);CHKERRQ(ierr);
  ierr = CreateMeshBoundary(mesh, &options);CHKERRQ(ierr);
  ierr = DistributeMesh(&options, &mesh);CHKERRQ(ierr);
  ierr = OutputVTK(mesh, &options);CHKERRQ(ierr);
  ierr = OutputMesh(mesh, &options);CHKERRQ(ierr);
  if (options.doPartition) {
    ierr = SectionIntDestroy(&options.partition);CHKERRQ(ierr);
  }
  ierr = DMDestroy(&mesh);CHKERRQ(ierr);
  ierr = PetscFinalize();
  PetscFunctionReturn(0);
}
