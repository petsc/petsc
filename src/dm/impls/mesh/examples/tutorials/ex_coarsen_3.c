//Use Hierarchy.hh to implement the creation of an entire array of topologies for use in multigrid methods.
//Use Hierarchy.hh to implement the creation of an entire array of topologies for use in multigrid methods.



#include <petscmesh.hh>
#include <petscmesh_viewers.hh>
#include <petscmesh_formats.hh>
//TEST compile without triangle and tetgen
//#undef PETSC_HAVE_TRIANGLE
//#undef PETSC_HAVE_TETGEN
#include "Generator.hh"
#include "Hierarchy.hh"
#include "Hierarchy_New.hh"


using ALE::Obj;


typedef struct {
  int        dim;                // The mesh dimension
  int        debug;              // The debugging level
  PetscBool  useZeroBase;        // Use zero-based indexing
  char       baseFilename[2048]; // The base filename for mesh files
  PetscInt   levels;             // The number of levels in the hierarchy
  PetscReal  coarseFactor;       // The maximum coarsening factor
  PetscReal  zScale;             // The relative spread of levels for visualization
  PetscBool  outputVTK;          // Output the mesh in VTK
  PetscBool  generate;           // Generate the mesh rather than reading it in
  PetscReal  curvatureCutoff;     // the cutoff for the curvature
  PetscReal  refinementLimit;    // the maximum cell volume used in the finest mesh 
  PetscBool  refinementGrading;  //grade the L-shaped and Fichera corner meshes as C0r^-2 \leq h \leq C1r^-2
  PetscBool  interpolate;        //construct the subdimensional elements of the mesh
} Options;

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsBegin(comm, "", "Options for mesh coarsening", "Options");CHKERRQ(ierr);
  options->dim          = 2;
  ierr = PetscOptionsInt("-dim", "The mesh dimension", "ex_coarsen_3.c", options->dim, &options->dim, PETSC_NULL);    
  options->debug        = 0;
  options->useZeroBase  = PETSC_TRUE;
  ierr = PetscStrcpy(options->baseFilename, "data/coarsen_mesh");CHKERRQ(ierr);
  options->levels       = 3;
  options->coarseFactor = 1.45;
  options->zScale       = 1.0;
  options->outputVTK    = PETSC_TRUE;
  options->curvatureCutoff = 1.5;
  options->generate = PETSC_TRUE;
  options->interpolate = PETSC_TRUE;  

  if (options->dim == 2) {
    options->refinementLimit = 0.001;
  } else {
    options->refinementLimit = 0.0001;
  }
  ierr = PetscOptionsBool("-interpolate", "construct additional elements of the sieve", "ex_coarsen.c", options->interpolate, &options->interpolate, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-refinementlimit", "The maximum cell volume in the finest mesh", "ex_coarsen.c", options->refinementLimit, &options->refinementLimit, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-debug", "The debugging level", "ex_coarsen_3", options->debug, &options->debug, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-use_zero_base", "Use zero-based indexing", "ex_coarsen_3.c", options->useZeroBase, &options->useZeroBase, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-base_file", "The base filename for mesh files", "ex_coarsen_3.c", options->baseFilename, options->baseFilename, 2048, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-levels", "The number of coarse levels", "ex_coarsen_3.c", options->levels, &options->levels, PETSC_NULL);    
  ierr = PetscOptionsReal("-coarsen", "The maximum coarsening factor", "ex_coarsen_3.c", options->coarseFactor, &options->coarseFactor, PETSC_NULL);   
  ierr = PetscOptionsReal("-curvature", "The automatic inclusion threshhold for the curvature", "ex_coarsen_3.c", options->curvatureCutoff, &options->curvatureCutoff, PETSC_NULL); 
  ierr = PetscOptionsBool("-generate", "Generate the mesh rather than reading it in.", "ex_coarsen.c", options->generate, &options->generate, PETSC_NULL);
  ierr = PetscOptionsReal("-z_scale", "The relative spread of levels for visualization", "ex_coarsen_3.c", options->zScale, &options->zScale, PETSC_NULL);    
  ierr = PetscOptionsBool("-output_vtk", "Output the mesh in VTK", "ex_coarsen_3.c", options->outputVTK, &options->outputVTK, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "Coarsen_RefineSingularity_Fichera"
PetscErrorCode  MeshRefineSingularity_Fichera(ALE::Obj<ALE::Mesh> mesh, MPI_Comm comm, double * singularity, double factor, ALE::Obj<ALE::Mesh> * refinedMesh, PetscBool  interpolate = PETSC_FALSE)
{
  ALE::Obj<ALE::Mesh> oldMesh = mesh;
  double              oldLimit;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  //ierr = MeshGetMesh(mesh, oldMesh);CHKERRQ(ierr);
  //ierr = MeshCreate(comm, refinedMesh);CHKERRQ(ierr);
  int dim = oldMesh->getDimension();
  oldLimit = oldMesh->getMaxVolume();
  //double oldLimInv = 1./oldLimit;
  double curLimit, tmpLimit;
  double minLimit = oldLimit/16384.;             //arbitrary;
  const ALE::Obj<ALE::Mesh::real_section_type>& coordinates = oldMesh->getRealSection("coordinates");
  const ALE::Obj<ALE::Mesh::real_section_type>& volume_limits = oldMesh->getRealSection("volume_limits");
  volume_limits->setFiberDimension(oldMesh->heightStratum(0), 1);
  oldMesh->allocate(volume_limits);
  const ALE::Obj<ALE::Mesh::label_sequence>& cells = oldMesh->heightStratum(0);
  ALE::Mesh::label_sequence::iterator c_iter = cells->begin();
  ALE::Mesh::label_sequence::iterator c_iter_end = cells->end();
  double centerCoords[dim];
  while (c_iter != c_iter_end) {
    const double * coords = oldMesh->restrictClosure(coordinates, *c_iter);
    for (int i = 0; i < dim; i++) {
      centerCoords[i] = 0;
      for (int j = 0; j < dim+1; j++) {
        centerCoords[i] += coords[j*dim+i];
      }
      centerCoords[i] = centerCoords[i]/(dim+1);
      //PetscPrintf(oldMesh->comm(), "%f, ", centerCoords[i]);
    }
    //PetscPrintf(oldMesh->comm(), "\n");
    double dist = 0.;
    double cornerdist = 0.;
    //HERE'S THE DIFFERENCE: if centercoords is less than the singularity coordinate for each direction, include that direction in the distance
    /*
    for (int i = 0; i < dim; i++) {
      if (centerCoords[i] <= singularity[i]) {
        dist += (centerCoords[i] - singularity[i])*(centerCoords[i] - singularity[i]);
      }
    }
    */
    //determine: the per-dimension distance: cases
    if (dim > 2) {
      for (int i = 0; i < dim; i++) {
	cornerdist = 0.;
	if (centerCoords[i] > singularity[i]) {
	  for (int j = 0; j < dim; j++) {
	    if (j != i) cornerdist += (centerCoords[j] - singularity[j])*(centerCoords[j] - singularity[j]);
	  }
	  if (cornerdist < dist || dist == 0.) dist = cornerdist; 
	}
      }
    }
    //patch up AROUND the corner by minimizing between the distance from the relevant axis and the singular vertex
    double singdist = 0.;
    for (int i = 0; i < dim; i++) {
      singdist += (centerCoords[i] - singularity[i])*(centerCoords[i] - singularity[i]);
    }
    if (singdist < dist || dist == 0.) dist = singdist;
    if (dist > 0.) {
      dist = sqrt(dist);
      double mu = pow(dist, factor);
      //PetscPrintf(oldMesh->comm(), "%f, %f\n", mu, dist);
      tmpLimit = oldLimit*pow(mu, dim);
      if (tmpLimit > minLimit) {
        curLimit = tmpLimit;
      } else curLimit = minLimit;
    } else curLimit = minLimit;
    //PetscPrintf(oldMesh->comm(), "%f, %f\n", dist, tmpLimit);
    volume_limits->updatePoint(*c_iter, &curLimit);
    c_iter++;
  }

  ALE::Obj<ALE::Mesh> newMesh = ALE::Generator::refineMesh(oldMesh, volume_limits, interpolate);
  //ierr = MeshSetMesh(*refinedMesh, newMesh);CHKERRQ(ierr);
  *refinedMesh = newMesh;
  const ALE::Obj<ALE::Mesh::real_section_type>& s = newMesh->getRealSection("default");
  const Obj<std::set<std::string> >& discs = oldMesh->getDiscretizations();

  for(std::set<std::string>::const_iterator f_iter = discs->begin(); f_iter != discs->end(); ++f_iter) {
    newMesh->setDiscretization(*f_iter, oldMesh->getDiscretization(*f_iter));
  }
  newMesh->setupField(s);
  //  PetscPrintf(newMesh->comm(), "refined\n");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateMesh"
PetscErrorCode CreateMesh(MPI_Comm comm, Obj<ALE::Mesh>& mesh, Options *options)
{
  ALE::Obj<ALE::Mesh> mesh2;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ALE::LogStage stage = ALE::LogStageRegister("MeshCreation");
  ALE::LogStagePush(stage);
  ierr = PetscPrintf(comm, "Creating mesh\n");CHKERRQ(ierr);
  if (options->generate) {
    if (options->dim == 3) {
      double lower[3] = {0.0, 0.0, 0.0};
      double upper[3] = {1.0, 1.0, 1.0};
      double offset[3] = {0.5, 0.5, 0.5};
      ALE::Obj<ALE::Mesh> mb = ALE::MeshBuilder::createFicheraCornerBoundary(comm, lower, upper, offset);
      mesh2 = ALE::Generator::refineMesh(ALE::Generator::generateMesh(mb, options->interpolate), options->refinementLimit, options->interpolate);
      ierr = MeshRefineSingularity_Fichera(mesh2, comm, offset, 0.75, &mesh);CHKERRQ(ierr);
    } else if (options->dim == 2) {
      double lower[2] = {0.0, 0.0};
      double upper[2] = {1.0, 1.0};
      double offset[2] = {0.0, 0.0};
      //ALE::Obj<ALE::Mesh> mb = ALE::MeshBuilder::createReentrantBoundary(comm, lower, upper, offset);
      ALE::Obj<ALE::Mesh> mb = ALE::MeshBuilder::createCircularReentrantBoundary(comm, 100, 1.0, 0.9);
      mesh2 = ALE::Generator::refineMesh(ALE::Generator::generateMesh(mb, options->interpolate), options->refinementLimit, options->interpolate);
      ierr = MeshRefineSingularity_Fichera(mesh2, comm, offset, 0.9, &mesh, PETSC_TRUE);CHKERRQ(ierr);
      //mesh = ALE::Generator::generateMesh(mb, options->debug);
    }
  } else {
    mesh = ALE::PCICE::Builder::readMesh(comm, options->dim, options->baseFilename, options->useZeroBase, options->interpolate, options->debug);
  }
  //ALE::Coarsener::IdentifyBoundary(mesh, 2);
  //ALE::Coarsener::make_coarsest_boundary(mesh, 2, options->levels + 1);
  ALE::LogStagePop(stage);
  ierr = PetscPrintf(comm, "  Read %d elements\n", mesh->heightStratum(0)->size());CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "  Read %d edges/faces\n", mesh->heightStratum(1)->size());CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "  Read %d vertices\n", mesh->depthStratum(0)->size());CHKERRQ(ierr);
  if (options->debug) {
   // topology->view("Serial topology");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "OutputVTK"
PetscErrorCode OutputVTK(ALE::Obj<ALE::Mesh> mesh, Options *options, std::string outname)
{
  PetscViewer    viewer;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (options->outputVTK) {
    ALE::LogStage stage = ALE::LogStageRegister("VTKOutput");
    ALE::LogStagePush(stage);
    ierr = PetscPrintf(mesh->comm(), "Creating VTK mesh files\n");CHKERRQ(ierr);
    ierr = PetscViewerCreate(mesh->comm(), &viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetType(viewer, PETSCVIEWERASCII);CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
    ierr = PetscViewerFileSetName(viewer, outname.c_str());CHKERRQ(ierr);
    ierr = VTKViewer::writeHeader(viewer);CHKERRQ(ierr);
    ierr = VTKViewer::writeVertices(mesh, viewer);
    ierr = VTKViewer::writeElements(mesh, viewer);
//    ierr = VTKViewer::writeHierarchyVertices(mesh, viewer, options->zScale);CHKERRQ(ierr);
//    ierr = VTKViewer::writeHierarchyElements(mesh, viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    //const ALE::Mesh::topology_type::sheaf_type& patches = mesh->getTopology()->getPatches();
    ALE::LogStagePop(stage);
  }
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
  ierr = PetscInitialize(&argc, &argv, (char *) 0, NULL);CHKERRQ(ierr);
  comm = PETSC_COMM_WORLD;

  try {
    ierr = ProcessOptions(comm, &options);CHKERRQ(ierr);
    Obj<ALE::Mesh> mesh;
    Mesh mesh_set[options.levels];
    for (int i = 0; i < options.levels; i++) {
      ierr = MeshCreate(comm, &mesh_set[i]);CHKERRQ(ierr);
    };
    ierr = CreateMesh(comm, mesh, &options);CHKERRQ(ierr);
    
    MeshSetMesh(mesh_set[0], mesh);
    ierr = MeshIDBoundary(mesh_set[0]);
    mesh->markBoundaryCells("marker");
    ierr = PetscPrintf(mesh->comm(), "%d boundary vertices, %d boundary cells\n", mesh->getLabelStratum("marker", 1)->size(), mesh->getLabelStratum("marker", 2)->size());
    ierr = MeshSpacingFunction(mesh_set[0]);
    //ierr = MeshIDBoundary(mesh_set[0]);
//    mesh->createLabel("marker");
//    mesh->markBoundaryCells("marker", 1, 2, false);
    MeshCreateHierarchyLabel_Link(mesh_set[0], options.coarseFactor, options.levels, &mesh_set[1],NULL,options.curvatureCutoff );
    Obj<ALE::Mesh> ale_meshes[options.levels];
    for (int i = 0; i < options.levels; i++) {
      MeshGetMesh(mesh_set[i], ale_meshes[i]);
    }
    Hierarchy_qualityInfo(ale_meshes, options.levels);
    //ierr = MeshCoarsenMesh(m, pow(options.coarseFactor, 2), &n);
    //ierr = MeshGetMesh(n, mesh);
    //ierr = MeshLocateInMesh(m, n);
   // Obj<ALE::Mesh::sieve_type> sieve = new ALE::Mesh::sieve_type(mesh->comm(), 0);
   // mesh->getTopology()->setPatch(options.levels, sieve);
   // mesh->getTopology()->stratify();
    char vtkfilename[128];
    for (int i = 0; i < options.levels; i++) {
      sprintf(vtkfilename, "testMesh%d.vtk", i);
      ierr = OutputVTK(ale_meshes[i], &options, vtkfilename);CHKERRQ(ierr);
    }
  } catch (ALE::Exception e) {
    std::cout << e << std::endl;
  }
  ierr = PetscFinalize();
  PetscFunctionReturn(0);
}

