static char help[] = "This example reads in a FVM mesh for PFLOTRAN.\n\n";

#include <petscmesh.hh>

using ALE::Obj;

typedef struct {
  PetscInt      debug;                       // The debugging level
  PetscInt      dim;                         // The topological mesh dimension
} Options;

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, Options *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->debug = 0;
  options->dim   = 2;

  ierr = PetscOptionsBegin(comm, "", "Bratu Problem Options", "DMMG");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-debug", "The debugging level", "fvTest.cxx", options->debug, &options->debug, PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "fvTest.cxx", options->dim, &options->dim, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ReadMesh"
PetscErrorCode ReadMesh(MPI_Comm comm, PetscInt *numCells, int *cellIds, double *cellX, double *cellY, double *cellZ, double *cellVols, PetscInt *numFaces, double *faceAreas, int *downCells, int *upCells, double *downX, double *downY, double *downZ, double *upX, double *upY, double *upZ, Options *options)
{
  PetscViewer    viewer;
  hid_t          file_id, dataset_id, dataspace_id;
  herr_t         status;
  hsize_t        dims[2];
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if defined(PETSC_HAVE_HDF5)
  ierr = PetscViewerCreate(comm, &viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(viewer, PETSC_VIEWER_HDF5);CHKERRQ(ierr);
  ierr = PetscViewerFileSetMode(viewer, FILE_MODE_READ);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(viewer, "sieve.h5");CHKERRQ(ierr);
  ierr = PetscViewerHDF5GetFileId(viewer, &file_id);CHKERRQ(ierr);
  /* Open an existing dataset. */
  dataset_id = H5Dopen(file_id, "/Cells/Natural IDs");
  /* Retrieve the dataspace. */
  dataspace_id = H5Dget_space(dataset_id);
  /* Allocate array for data */
  status = H5Sget_simple_extent_dims(dataspace_id, dims, NULL);
  /* Close the dataspace. */
  status = H5Sclose(dataspace_id);
  ierr = PetscMalloc5(dims[0],int,&cellIds,dims[0],double,&cellVols,dims[0],double,&cellX,dims[0],double,&cellY,dims[0],double,&cellZ);CHKERRQ(ierr);
  /* Read the data. */
  status = H5Dread(dataset_id, H5T_STD_I32LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, cellIds);
  /* Close the dataset. */
  status = H5Dclose(dataset_id);
  /* Repeat */
  dataset_id = H5Dopen(file_id, "/Cells/Volumes");
  status = H5Dread(dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, cellVols);
  status = H5Dclose(dataset_id);
  dataset_id = H5Dopen(file_id, "/Cells/X-Coordinates");
  status = H5Dread(dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, cellX);
  status = H5Dclose(dataset_id);
  dataset_id = H5Dopen(file_id, "/Cells/Y-Coordinates");
  status = H5Dread(dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, cellY);
  status = H5Dclose(dataset_id);
  dataset_id = H5Dopen(file_id, "/Cells/Z-Coordinates");
  status = H5Dread(dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, cellZ);
  status = H5Dclose(dataset_id);
  /* Get Connections */
  dataset_id = H5Dopen(file_id, "/Connections/Areas");
  dataspace_id = H5Dget_space(dataset_id);
  status = H5Sget_simple_extent_dims(dataspace_id, &dims[1], NULL);
  status = H5Sclose(dataspace_id);
  ierr = PetscMalloc5(dims[1],double,&faceAreas,dims[1],int,&downCells,dims[1],double,&downX,dims[1],double,&downY,dims[1],double,&downZ);CHKERRQ(ierr);
  status = H5Dread(dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, faceAreas);
  status = H5Dclose(dataset_id);
  dataset_id = H5Dopen(file_id, "/Connections/Downwind Cell IDs");
  status = H5Dread(dataset_id, H5T_STD_I32LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, downCells);
  status = H5Dclose(dataset_id);
  dataset_id = H5Dopen(file_id, "/Connections/Downwind Distance X");
  status = H5Dread(dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, downX);
  status = H5Dclose(dataset_id);
  dataset_id = H5Dopen(file_id, "/Connections/Downwind Distance Y");
  status = H5Dread(dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, downY);
  status = H5Dclose(dataset_id);
  dataset_id = H5Dopen(file_id, "/Connections/Downwind Distance Z");
  status = H5Dread(dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, downZ);
  status = H5Dclose(dataset_id);
  ierr = PetscMalloc4(dims[1],int,&upCells,dims[1],double,&upX,dims[1],double,&upY,dims[1],double,&upZ);CHKERRQ(ierr);
  dataset_id = H5Dopen(file_id, "/Connections/Upwind Cell IDs");
  status = H5Dread(dataset_id, H5T_STD_I32LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, downCells);
  status = H5Dclose(dataset_id);
  dataset_id = H5Dopen(file_id, "/Connections/Upwind Distance X");
  status = H5Dread(dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, downX);
  status = H5Dclose(dataset_id);
  dataset_id = H5Dopen(file_id, "/Connections/Upwind Distance Y");
  status = H5Dread(dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, downY);
  status = H5Dclose(dataset_id);
  dataset_id = H5Dopen(file_id, "/Connections/Upwind Distance Z");
  status = H5Dread(dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, downZ);
  status = H5Dclose(dataset_id);
  /* Cleanup */
  ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
  *numCells = dims[0];
  *numFaces = dims[1];
  for(int c = 0; c < *numCells; ++c) {
    std::cout << "Cell: " << cellIds[c] << " Center: " << cellX[c]<<","<<cellY[c]<<","<<cellZ[c] << " Vol: " << cellVols[c] << std::endl;
  }
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CreateMesh"
// We will represent cells and faces in the sieve
//   In addition, we will have default sections for:
//     cell volume
//     cell centroid
//     face area
//     face centroid
PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm, Options *options)
{
  Mesh           mesh;
  PetscTruth     view;
  PetscMPIInt    size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  Obj<ALE::Mesh>             m     = new ALE::Mesh(comm, options->dim, options->debug);
  Obj<ALE::Mesh::sieve_type> sieve = new ALE::Mesh::sieve_type(comm, options->debug);
#if 1
  PetscInt                   numCells  = 17;
  PetscInt                   numFaces  = 32;
  PetscInt                   connect[64] = { 1,  2,  2,  3,  4,  9,  4, 12,  4, 15,  9, 10, 12, 13,
                                            15, 16, 10, 11, 13, 14, 16, 17, 11,  5, 14,  5, 17,  5,
                                             6,  7,  7,  8,  1,  4,  4,  6,  2,  9,  2, 10,  2, 11,
                                             9, 12, 10, 13, 11, 14, 12, 15, 13, 16, 14, 17, 15,  7,
                                          16,  7, 17,  7,  3,  5,  5,  8};
  PetscReal                  cellCenters[34] = {0.5,   0.5,   1.5,   0.5, 2.5, 0.5, 0.5,   1.5,   2.5,   1.5,  
                                                0.5,   2.5,   1.5,   2.5, 2.5, 2.5, 1.167, 1.167, 1.5,   1.167,
                                                1.833, 1.167, 1.167, 1.5, 1.5, 1.5, 1.833, 1.5,   1.167, 1.833,
                                                1.5,   1.833, 1.833, 1.833};
  PetscReal                  cellVolumes[17] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.111, 0.111, 0.111,
                                                0.111, 0.111, 0.111, 0.111, 0.111, 0.111};
  PetscReal                  faceCenters[64] = {1.0,     0.5,   2.0,    0.5,   1.0,    1.167, 1.0,   1.5,
                                                1.0,     1.833, 1.333,  1.167, 1.333,  1.5,   1.333, 1.833,
                                                1.667,   1.167, 1.667,  1.5,   1.667,  1.833, 2.0,   1.167,
                                                2.0,     1.5,   2.0,    1.833, 1.0,    2.5,   2.0,   2.5,  
                                                0.5,     1.0,   0.5,    2.0,   1.167,  1.0,   1.5,   1.0,
                                                1.833,   1.0,   1.167,  1.333, 1.5,    1.333, 1.833, 1.333,
                                                1.167,   1.667, 1.5,    1.667, 1.833,  1.667, 1.167, 2.0,
                                                1.5,     2.0,   1.833,  2.0,   2.5,    1.0,   2.5,   2.0};
  PetscReal                  faceVolumes[32] = {1.0, 1.0, 0.333, 0.333, 0.333, 0.333, 0.333, 0.333, 0.333,
                                                0.333, 0.333, 0.333, 0.333, 0.333, 1.0, 1.0, 1.0, 1.0, 0.333,
                                                0.333, 0.333, 0.333, 0.333, 0.333, 0.333, 0.333, 0.333, 0.333,
                                                0.333, 0.333, 1.0, 1.0};
#endif
  //PetscInt numCells;
  //PetscInt numFaces;
  int     *cellIds, *downCells, *upCells;
  double  *cellX, *cellY, *cellZ, *cellVols;
  double  *faceAreas, *downX, *downY, *downZ, *upX, *upY, *upZ;

  //ierr = ReadMesh(comm, &numCells, cellIds, cellX, cellY, cellZ, cellVols, &numFaces, faceAreas, downCells, upCells, downX, downY, downZ, upX, upY, upZ, options);CHKERRQ(ierr);
  if (!m->commRank()) {
    for(int f = 0; f < numFaces; ++f) {
      sieve->addCone(f+numCells+1, connect[f*2+0]);
      sieve->addCone(f+numCells+1, connect[f*2+1]);
    }
  }
  m->setSieve(sieve);
  m->stratify();
  const Obj<ALE::Mesh::real_section_type>& cellVol = m->getRealSection("cell volumes");
  const Obj<ALE::Mesh::real_section_type>& cellCen = m->getRealSection("cell centers");
  const Obj<ALE::Mesh::label_sequence>&    cells   = m->heightStratum(0);

  cellVol->setFiberDimension(cells, 1);
  cellCen->setFiberDimension(cells, options->dim);
  m->allocate(cellVol);
  m->allocate(cellCen);
  for(ALE::Mesh::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
    cellVol->updatePoint(*c_iter, &cellVolumes[*c_iter-1]);
    cellCen->updatePoint(*c_iter, &cellCenters[(*c_iter-1)*options->dim]);
  }
  const Obj<ALE::Mesh::real_section_type>& faceVol = m->getRealSection("face volumes");
  const Obj<ALE::Mesh::real_section_type>& faceCen = m->getRealSection("face centers");
  const Obj<ALE::Mesh::label_sequence>&    faces   = m->depthStratum(0);

  faceVol->setFiberDimension(faces, 1);
  faceCen->setFiberDimension(faces, options->dim);
  m->allocate(faceVol);
  m->allocate(faceCen);
  for(ALE::Mesh::label_sequence::iterator f_iter = faces->begin(); f_iter != faces->end(); ++f_iter) {
    faceVol->updatePoint(*f_iter, &faceVolumes[*f_iter - numCells-1]);
    faceCen->updatePoint(*f_iter, &faceCenters[(*f_iter - numCells-1)*options->dim]);
  }
  //ierr = PetscFree5(cellIds,cellVols,cellX,cellY,cellZ);CHKERRQ(ierr);
  //ierr = PetscFree5(faceAreas,downCells,downX,downY,downZ);CHKERRQ(ierr);
  //ierr = PetscFree4(upCells,upX,upY,upZ);CHKERRQ(ierr);

  ierr = MeshCreate(comm, &mesh);CHKERRQ(ierr);
  ierr = MeshSetMesh(mesh, m);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  if (size > 1) {
    Mesh parallelMesh;

    // TODO: Need different partitioning
    ierr = MeshDistribute(mesh, PETSC_NULL, &parallelMesh);CHKERRQ(ierr);
    ierr = MeshDestroy(mesh);CHKERRQ(ierr);
    mesh = parallelMesh;
  }
  ierr = PetscOptionsHasName(PETSC_NULL, "-mesh_view", &view);CHKERRQ(ierr);
  if (view) {
    Obj<ALE::Mesh> m;
    ierr = MeshGetMesh(mesh, m);CHKERRQ(ierr);
    m->view("Mesh");
  }
  ierr = PetscOptionsHasName(PETSC_NULL, "-mesh_view_simple", &view);CHKERRQ(ierr);
  if (view) {ierr = MeshView(mesh, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);}
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
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  MPI_Comm       comm;
  Options        options;
  DM             dm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr);
  comm = PETSC_COMM_WORLD;
  ierr = ProcessOptions(comm, &options);CHKERRQ(ierr);
  try {
    ierr = CreateMesh(comm, &dm, &options);CHKERRQ(ierr);
    ierr = DestroyMesh(dm, &options);CHKERRQ(ierr);
  } catch(ALE::Exception e) {
    std::cerr << e << std::endl;
  }
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
