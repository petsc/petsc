// This example will implement PFLOTRAN eventually
static char help[] = "This example runs PFLOTRAN.\n\n";

#include <petscda.h>
#include <petscmesh.h>

#include <Distribution.hh>

typedef struct {
  PetscInt   debug;              // The debugging level
  PetscInt   test;               // The testing level
  PetscTruth interpolate;        // Generate missing intermediate elements
  PetscReal  refinementLimit;    // The largest allowable cell volume
  char       baseFilename[2048]; // The base filename for mesh files
  char       partitioner[2048];  // The graph partitioner
} Options;

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->debug           = 0;
  options->test            = 1;
  options->interpolate     = PETSC_TRUE;
  options->refinementLimit = 0.0;

  ierr = PetscOptionsBegin(comm, "", "PFLOTRAN Options", "DMMG");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-debug", "The debugging level", "pflotran.cxx", options->debug, &options->debug, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-test", "The testing level", "pflotran.cxx", options->test, &options->test, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsTruth("-interpolate", "Generate missing intermediate elements", "pflotran.cxx", options->interpolate, &options->interpolate, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-refinement_limit", "The largest allowable cell volume", "pflotran.cxx", options->refinementLimit, &options->refinementLimit, PETSC_NULL);CHKERRQ(ierr);

    ierr = PetscStrcpy(options->baseFilename, "data/ex10");CHKERRQ(ierr);
    ierr = PetscOptionsString("-base_filename", "The base filename for mesh files", "pflotran.cxx", options->baseFilename, options->baseFilename, 2048, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscStrcpy(options->partitioner, "parmetis");CHKERRQ(ierr);
    ierr = PetscOptionsString("-partitioner", "The graph partitioner", "pflotran.cxx", options->partitioner, options->partitioner, 2048, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  if (!options->interpolate) {
    SETERRQ(PETSC_ERR_ARG_WRONG, "Cannot partition faces of a non-interpolated mesh");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreatePartition"
// Creates a field whose value is the processor rank on each element
PetscErrorCode CreatePartition(Mesh mesh, SectionInt *partition)
{
  ALE::Obj<ALE::Mesh::int_section_type> section;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MeshGetCellSectionInt(mesh, 1, partition);CHKERRQ(ierr);
  ierr = SectionIntGetSection(*partition, section);CHKERRQ(ierr);
  const ALE::Mesh::int_section_type::patch_type             patch    = 0;
  const ALE::Obj<ALE::Mesh::topology_type>&                 topology = section->getTopology();
  const ALE::Obj<ALE::Mesh::topology_type::label_sequence>& cells    = topology->heightStratum(patch, 0);
  const ALE::Mesh::topology_type::label_sequence::iterator  end      = cells->end();
  const ALE::Mesh::int_section_type::value_type             rank     = section->commRank();

  for(ALE::Mesh::topology_type::label_sequence::iterator c_iter = cells->begin(); c_iter != end; ++c_iter) {
    section->updatePoint(patch, *c_iter, &rank);
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

  ierr = MeshCreatePCICE(comm, 3, coordFile.c_str(), adjFile.c_str(), options->interpolate, PETSC_NULL, 0, 0, &mesh);CHKERRQ(ierr);

  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  if (size > 1) {
    Mesh parallelMesh;

    ierr = MeshDistributeByFace(mesh, options->partitioner, &parallelMesh);CHKERRQ(ierr);
    ierr = MeshDestroy(mesh);CHKERRQ(ierr);
    mesh = parallelMesh;
  }
  ierr = PetscOptionsHasName(PETSC_NULL, "-mesh_view", &view);CHKERRQ(ierr);
  if (view) {
    ALE::Obj<ALE::Mesh> m;
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
#define __FUNCT__ "TraverseFaces"
PetscErrorCode TraverseFaces(DM dm, Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  Mesh mesh = (Mesh) dm;
  ALE::Obj<ALE::Mesh> m;
  
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  const int                                      rank        = m->commRank();
  const ALE::Mesh::real_section_type::patch_type patch       = 0;
  const ALE::Obj<ALE::Mesh::real_section_type>&  coordinates = m->getRealSection("coordinates");
  const ALE::Obj<ALE::Mesh::topology_type>&      topology    = m->getTopology();
  const ALE::Obj<ALE::Mesh::sieve_type>&         sieve       = topology->getPatch(patch);
  const ALE::Obj<ALE::Mesh::topology_type::label_sequence>& cells = topology->heightStratum(patch, 0);
  const int dim = m->getDimension();
  
  // Loop over elements (quadrilaterals)
  for(ALE::Mesh::topology_type::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[%d]Processing cell %d\n", rank, *c_iter);CHKERRQ(ierr);
    if (!options->interpolate) continue;
    const ALE::Obj<ALE::Mesh::sieve_type::traits::coneSequence>& cone = sieve->cone(*c_iter);
    const ALE::Mesh::sieve_type::traits::coneSequence::iterator  end  = cone->end();
    
    for(ALE::Mesh::sieve_type::traits::coneSequence::iterator f_iter = cone->begin(); f_iter != end; ++f_iter) {
      ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[%d]  Processing face %d, with coordinates\n", rank, *f_iter);CHKERRQ(ierr);
      const ALE::Obj<ALE::Mesh::sieve_type::coneArray>& vertices = sieve->nCone(*f_iter, topology->depth(patch, *f_iter));
      const ALE::Mesh::sieve_type::coneArray::iterator  vEnd     = vertices->end();
      
      ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "    ");CHKERRQ(ierr);
      for(ALE::Mesh::sieve_type::coneArray::iterator v_iter = vertices->begin(); v_iter != vEnd; ++v_iter) {
	const ALE::Mesh::real_section_type::value_type *array = coordinates->restrict(patch, *v_iter);
	
	ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "(");CHKERRQ(ierr);
	for(int d = 0; d < dim; ++d) {
	  if (d > 0) {
	    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, ", ");CHKERRQ(ierr);
	  }
	  ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "%g", array[d]);CHKERRQ(ierr);
	}
	ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, ") ");CHKERRQ(ierr);
      }
      ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "\n");CHKERRQ(ierr);
    }
  }
  ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD);CHKERRQ(ierr);
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
  ALE::Obj<ALE::Mesh> m;
  ALE::Obj<ALE::Mesh::real_section_type> s;
  
  ierr = MeshGetSectionReal(mesh, "u", &f);CHKERRQ(ierr);
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = SectionRealGetSection(f, s);CHKERRQ(ierr);
  const ALE::Mesh::real_section_type::patch_type            patch    = 0;
  const ALE::Obj<ALE::Mesh::topology_type>&                 topology = m->getTopology();
  const ALE::Obj<ALE::Mesh::topology_type::label_sequence>& cells    = topology->heightStratum(patch, 0);
  const ALE::Obj<ALE::Discretization>&                      disc     = m->getDiscretization();
  
  disc->setNumDof(topology->depth(), 1);
  m->setupField(s);
  s->setDebug(options->debug);
  // Loop over elements (quadrilaterals)
  for(ALE::Mesh::topology_type::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
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
  ALE::Obj<ALE::Mesh> m;
  ALE::Obj<ALE::Mesh::real_section_type> s;
  
  ierr = MeshGetSectionReal(mesh, "u", &f);CHKERRQ(ierr);
  ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
  ierr = SectionRealGetSection(f, s);CHKERRQ(ierr);
  const ALE::Mesh::real_section_type::patch_type            patch    = 0;
  const ALE::Obj<ALE::Mesh::topology_type>&                 topology = m->getTopology();
  const ALE::Obj<ALE::Mesh::sieve_type>&                    sieve    = topology->getPatch(patch);
  const ALE::Obj<ALE::Mesh::topology_type::label_sequence>& faces    = topology->heightStratum(patch, 1);
  
  ierr = SectionRealZero(f);CHKERRQ(ierr);
  for(ALE::Mesh::topology_type::label_sequence::iterator f_iter = faces->begin(); f_iter != faces->end(); ++f_iter) {
    const ALE::Obj<ALE::Mesh::sieve_type::traits::supportSequence>& neighbors = sieve->support(*f_iter);
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

  PetscFunctionBegin;
  if (options->test) {
    ierr = TraverseFaces(dm, options);CHKERRQ(ierr);
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
