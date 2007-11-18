static char help[] = "This example demonstrates partitioning/distributed hexes by face.\n\n";

#include <petscda.h>
#include <petscmesh.h>

using ALE::Obj;

/*
    * Each FACE is assigned to (owned by) a unique process. (Faces are NOT ghosted)
    * Cells are ghosted and live on all processes that own any of its faces
    * For the linear algebra, each cell is assigned to (owned by) a unique process
      a VecScatter is created that maps values from the owned cells to the ghost cells

*/
 
typedef struct {
  PetscInt   debug;              // The debugging level
  PetscInt   test;               // The testing level
  PetscTruth postponeGhosts;     // Number ghost variables last
  char       baseFilename[2048]; // The base filename for mesh files
  char       partitioner[2048];  // The graph partitioner

  VecScatter scatter;            // maps from global (non-ghosted) to local (ghosted) 
  Vec        local,global;       // template global and local vectors
} Options;

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->debug          = 0;
  options->test           = 0;
  options->postponeGhosts = PETSC_FALSE;

  ierr = PetscOptionsBegin(comm, "", "PFLOTRAN Options", "DMMG");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-debug", "The debugging level", "pflotran.cxx", options->debug, &options->debug, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-test", "The testing level", "pflotran.cxx", options->test, &options->test, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-postpone_ghosts", "Number ghost variables last", "pflotran.cxx", options->postponeGhosts, &options->postponeGhosts, PETSC_NULL);CHKERRQ(ierr);

    ierr = PetscStrcpy(options->baseFilename, "data/ex10");CHKERRQ(ierr);
    ierr = PetscOptionsString("-base_filename", "The base filename for mesh files", "pflotran.cxx", options->baseFilename, options->baseFilename, 2048, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscStrcpy(options->partitioner, "parmetis");CHKERRQ(ierr);
    ierr = PetscOptionsString("-partitioner", "The graph partitioner", "pflotran.cxx", options->partitioner, options->partitioner, 2048, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreatePartition"
// Creates a field whose value is the processor rank on each element
PetscErrorCode CreatePartition(Mesh mesh, SectionInt *partition)
{
  Obj<ALE::Mesh> m;
  Obj<ALE::Mesh::int_section_type> section;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = MeshGetCellSectionInt(mesh, 1, partition);CHKERRQ(ierr);
  ierr = SectionIntGetSection(*partition, section);CHKERRQ(ierr);
  const Obj<ALE::Mesh::label_sequence>&         cells = m->heightStratum(0);
  const ALE::Mesh::label_sequence::iterator     end   = cells->end();
  const ALE::Mesh::int_section_type::value_type rank  = section->commRank();

  for(ALE::Mesh::label_sequence::iterator c_iter = cells->begin(); c_iter != end; ++c_iter) {
    section->updatePoint(*c_iter, &rank);
  }
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
  std::string baseFilename(options->baseFilename);
  std::string coordFile = baseFilename+".nodes";
  std::string adjFile   = baseFilename+".lcon";
  PetscTruth  view;
  PetscMPIInt size;

  ierr = MeshCreatePCICE(comm, 3, coordFile.c_str(), adjFile.c_str(), PETSC_TRUE, PETSC_NULL, &mesh);CHKERRQ(ierr);

  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  if (size > 1) {
    Mesh parallelMesh;

    ierr = MeshDistributeByFace(mesh, options->partitioner, &parallelMesh);CHKERRQ(ierr);
    ierr = MeshDestroy(mesh);CHKERRQ(ierr);
    mesh = parallelMesh;
  }
  ierr = PetscOptionsHasName(PETSC_NULL, "-mesh_view", &view);CHKERRQ(ierr);
  if (view) {
    Obj<ALE::Mesh> m;
    ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
    m->view("Mesh");
  }
  ierr = PetscOptionsHasName(PETSC_NULL, "-mesh_view_vtk", &view);CHKERRQ(ierr);
  if (view) {ierr = ViewMesh(mesh, "pflotran.vtk");CHKERRQ(ierr);}
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
  typedef ALE::SieveAlg<ALE::Mesh> sieve_alg;
  Mesh           mesh = (Mesh) dm;
  Obj<ALE::Mesh> m;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  const int                                rank        = m->commRank();
  const Obj<ALE::Mesh::real_section_type>& coordinates = m->getRealSection("coordinates");
  const Obj<ALE::Mesh::sieve_type>&        sieve       = m->getSieve();
    
  // Loop over cells
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Each cell (including ghosts), on each process\n");CHKERRQ(ierr);
  const Obj<ALE::Mesh::label_sequence>& cells = m->heightStratum(0);
  for(ALE::Mesh::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {

    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[%d]Cell %d\n", rank, *c_iter);CHKERRQ(ierr);
    const Obj<ALE::Mesh::sieve_type::traits::coneSequence>& faces = sieve->cone(*c_iter);
    const ALE::Mesh::sieve_type::traits::coneSequence::iterator  end  = faces->end();

    // Loop over faces owned by this process on the given cell    
    for(ALE::Mesh::sieve_type::traits::coneSequence::iterator f_iter = faces->begin(); f_iter != end; ++f_iter) {
      ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "      Face %d, with coordinates ", *f_iter);CHKERRQ(ierr);
      const Obj<sieve_alg::coneArray>& vertices = sieve_alg::nCone(m, *f_iter, m->depth(*f_iter));
      
      // Loop over vertices of the given face
      for(sieve_alg::coneArray::iterator v_iter = vertices->begin(); v_iter != vertices->end(); ++v_iter) {
        const ALE::Mesh::real_section_type::value_type *array = coordinates->restrictPoint(*v_iter);
	
        ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, " %d (%g,%g,%g)",*v_iter,array[0],array[1],array[2]);CHKERRQ(ierr);
      }
      ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "\n");CHKERRQ(ierr);
    }
  }
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TraverseFaces"
PetscErrorCode TraverseFaces(DM dm, Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  Mesh mesh = (Mesh) dm;
  Obj<ALE::Mesh> m;
  
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  const int                                rank  = m->commRank();
  const Obj<ALE::Mesh::sieve_type>& sieve = m->getSieve();
    
  // Loop over cells
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Each face (they are not ghosted), on each process\n");CHKERRQ(ierr);
  const Obj<ALE::Mesh::label_sequence>& faces = m->heightStratum(1);
  for(ALE::Mesh::label_sequence::iterator f_iter = faces->begin(); f_iter != faces->end(); ++f_iter) {

    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[%d]Face %d\n", rank, *f_iter);CHKERRQ(ierr);
    const Obj<ALE::Mesh::sieve_type::traits::supportSequence>& cells = sieve->support(*f_iter);
    const ALE::Mesh::sieve_type::traits::supportSequence::iterator  end   = cells->end();

    // Loop over cells (including ghosts) for the given face
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "       Cells ");CHKERRQ(ierr);
    for(ALE::Mesh::sieve_type::traits::supportSequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
      ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "       %d ", *c_iter);CHKERRQ(ierr);
    }
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "\n");CHKERRQ(ierr);
  }
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateGlobalVector"
PetscErrorCode CreateGlobalVector(DM dm, Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  Mesh        mesh = (Mesh) dm;
  SectionReal f;
  Obj<ALE::Mesh> m;
  Obj<ALE::Mesh::real_section_type> s;
  
  ierr = MeshGetSectionReal(mesh, "default", &f);CHKERRQ(ierr);
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = SectionRealGetSection(f, s);CHKERRQ(ierr);
  const Obj<ALE::Discretization>& disc = m->getDiscretization();
  
  disc->setNumDof(m->depth(), 2);
  s->setDebug(options->debug);
  m->setupField(s);

  VecScatter scatter;

  ierr = MeshCreateGlobalScatter(mesh, f, &scatter);CHKERRQ(ierr);
  ierr = VecScatterDestroy(scatter);CHKERRQ(ierr);

  const Obj<ALE::Mesh::order_type>& order = m->getFactory()->getGlobalOrder(m, "default", m->getRealSection("default"));

  ierr = VecCreate(m->comm(), &options->global);CHKERRQ(ierr);
  ierr = VecSetSizes(options->global, order->getLocalSize(), order->getGlobalSize());CHKERRQ(ierr);
  ierr = VecSetFromOptions(options->global);CHKERRQ(ierr);
  ierr = SectionRealDestroy(f);CHKERRQ(ierr);

  ierr = MeshGetSectionReal(mesh, "u", &f);CHKERRQ(ierr);
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = SectionRealGetSection(f, s);CHKERRQ(ierr);
  
  disc->setNumDof(m->depth(), 2);
  s->setDebug(options->debug);
  m->setupField(s);

  ierr = VecCreateSeq(PETSC_COMM_SELF, s->size(), &options->local);CHKERRQ(ierr);
  ierr = SectionRealDestroy(f);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateField"
PetscErrorCode CreateField(DM dm, Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  Mesh        mesh = (Mesh) dm;
  SectionReal f;
  Obj<ALE::Mesh> m;
  Obj<ALE::Mesh::real_section_type> s;
  
  ierr = MeshGetSectionReal(mesh, "u", &f);CHKERRQ(ierr);
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = SectionRealGetSection(f, s);CHKERRQ(ierr);
  const Obj<ALE::Mesh::label_sequence>& cells = m->heightStratum(0);
  const Obj<ALE::Discretization>&       disc  = m->getDiscretization();
  
  disc->setNumDof(m->depth(), 1);
  s->setDebug(options->debug);
  m->setupField(s, options->postponeGhosts);
  // Loop over elements (quadrilaterals)
  for(ALE::Mesh::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
    const double value = (double) *c_iter;
    
    ierr = SectionRealUpdate(f, *c_iter, &value);CHKERRQ(ierr);
  }
  ierr = SectionRealView(f, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = SectionRealDestroy(f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "UpdateGhosts"
PetscErrorCode UpdateGhosts(DM dm, Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  Mesh        mesh = (Mesh) dm;
  SectionReal f;
  Obj<ALE::Mesh> m;
  Obj<ALE::Mesh::real_section_type> s;
  
  ierr = MeshGetSectionReal(mesh, "u", &f);CHKERRQ(ierr);
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = SectionRealGetSection(f, s);CHKERRQ(ierr);
  const Obj<ALE::Mesh::sieve_type>&     sieve = m->getSieve();
  const Obj<ALE::Mesh::label_sequence>& faces = m->heightStratum(1);
  
  ierr = SectionRealZero(f);CHKERRQ(ierr);
  for(ALE::Mesh::label_sequence::iterator f_iter = faces->begin(); f_iter != faces->end(); ++f_iter) {
    const Obj<ALE::Mesh::sieve_type::traits::supportSequence>& neighbors = sieve->support(*f_iter);
    const ALE::Mesh::sieve_type::traits::supportSequence::iterator  end       = neighbors->end();
    const double value = 1.0;
    
    for(ALE::Mesh::sieve_type::traits::supportSequence::iterator c_iter = neighbors->begin(); c_iter != end; ++c_iter) {
      ierr = SectionRealUpdateAdd(f, *c_iter, &value);CHKERRQ(ierr);
    }
  }
  ierr = SectionRealComplete(f);CHKERRQ(ierr);
  ierr = SectionRealView(f, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = SectionRealDestroy(f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RunTests"
PetscErrorCode RunTests(DM dm, Options *options)
{
  PetscErrorCode ierr;
  PetscScalar    *l;
  PetscInt       i,n;

  PetscFunctionBegin;
  if (options->test) {
    ierr = TraverseCells(dm, options);CHKERRQ(ierr);
    ierr = TraverseFaces(dm, options);CHKERRQ(ierr);
    ierr = CreateGlobalVector(dm, options);CHKERRQ(ierr);
    ierr = VecGetArray(options->local,&l);CHKERRQ(ierr);
    ierr = VecGetLocalSize(options->local,&n);CHKERRQ(ierr);
    for (i=0; i<n; i++) l[i] = i;
    ierr = VecRestoreArray(options->local,&l);CHKERRQ(ierr);
    ierr = VecView(options->local,0);CHKERRQ(ierr);
    ierr = CreateField(dm, options);CHKERRQ(ierr);
    ierr = UpdateGhosts(dm, options);CHKERRQ(ierr);
  }
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
    ierr = RunTests(dm, &options);CHKERRQ(ierr);
    ierr = DestroyMesh(dm, &options);CHKERRQ(ierr);
  } catch(ALE::Exception e) {
    std::cout << "ERROR: " << e.msg() << std::endl;
  }
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
