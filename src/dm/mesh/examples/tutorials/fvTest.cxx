static char help[] = "This example reads in a FVM mesh for PFLOTRAN.\n\n";

#include <petscmesh.hh>

/*
Barry and Matt,

Attached is a sample hdf5 file for reading into Sieve.  The file
contains sufficient data to read in either by explicit (duals) or
implicit (cells with shared vertices) connection.  The grid size is
nx=5, ny=4, nz=3 with grid spacing:

 PetscReal dx[5] = {10.,11.,12.,13.,14.};
 PetscReal dy[4] = {13.,12.,11.,10.};
 PetscReal dz[3] = {15.,20.,25.};

Anyone interested in perusing the file and commenting on potential
changes can do so between now and the telecon tomorrow using the hdf5
viewer:

http://hdf.ncsa.uiuc.edu/hdf-java-html/hdfview/

It runs with java and is cake to install.

I will try to provide something with inactive cells and local grid
refinement in the future in order to provide a bit more complexity, but
this should due for a start.

Note that all data sets are stored in 1D.  Doing so enables very fast
reads on large problems.

--------------------------------------------
Explanation of file contents:

Cells - group containing all cell-based data sets
 Natural IDs - zero-based natural id of each cell
(** see attached pdf for layout of vertices in HEX)
 Vertex IDs 0 - first vertex in standard hex 
                element/cell layout
 Vertex IDs 1 - second vertex in standard hex 
                element/cell layout
 ... 
 Vertex IDs 7 - last vertex in standard hex 
                element/cell layout
 Volumes - volumes of grid cells
 X-Coordinates - cell x-coords
 Y-Coordinates - cell y-coords
 Z-Coordinates - cell z-coords

Connections - group containing all connection-based 
             data sets
 Area - connection interfacial areas
 Upwind Cell IDs - zero-based ids of upwind cells 
                   for each connection
 Upwind Distance X - x-compoonent of distance vector
                     between upwidn cell center and
                     center of interface between cells
 Upwind Distance Y - y-compoonent of distance vector
                     between upwidn cell center and
                     center of interface between cells
 Upwind Distance Z - z-compoonent of distance vector
                     between upwidn cell center and
                     center of interface between cells
 All the same for downwind data sets

Vertices - cell containing all vertex-based (cell corners) data sets
 Natural IDs - zero-based natural id of each vertex
 X-Coordinates - vertex x-coords
 Y-Coordinates - vertex y-coords
 Z-Coordinates - vertex z-coords
--------------------------------------------*/


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

typedef struct {
  int     numCells;
  int    *cellIds; /* Id of each cell */
  double *cellX;   /* X of cell center */
  double *cellY;   /* Y of cell center */
  double *cellZ;   /* Z of cell center */
  double *cellVols;
  int     numFaces;
  double *faceAreas;
  int    *downCells;
  int    *upCells;
  double *downX;
  double *downY;
  double *downZ;
  double *upX;
  double *upY;
  double *upZ;
} PFLOTRANMesh;

#undef __FUNCT__
#define __FUNCT__ "ReadMesh"
PetscErrorCode ReadMesh(MPI_Comm comm, PFLOTRANMesh *data, Options *options)
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
  ierr = PetscMalloc5(dims[0],int,&data->cellIds,dims[0],double,&data->cellVols,dims[0],double,&data->cellX,dims[0],double,&data->cellY,dims[0],double,&data->cellZ);CHKERRQ(ierr);
  /* Read the data. */
  status = H5Dread(dataset_id, H5T_STD_I32LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data->cellIds);
  /* Close the dataset. */
  status = H5Dclose(dataset_id);
  /* Repeat */
  dataset_id = H5Dopen(file_id, "/Cells/Volumes");
  status = H5Dread(dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data->cellVols);
  status = H5Dclose(dataset_id);
  dataset_id = H5Dopen(file_id, "/Cells/X-Coordinates");
  status = H5Dread(dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data->cellX);
  status = H5Dclose(dataset_id);
  dataset_id = H5Dopen(file_id, "/Cells/Y-Coordinates");
  status = H5Dread(dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data->cellY);
  status = H5Dclose(dataset_id);
  dataset_id = H5Dopen(file_id, "/Cells/Z-Coordinates");
  status = H5Dread(dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data->cellZ);
  status = H5Dclose(dataset_id);
  /* Get Connections */
  dataset_id = H5Dopen(file_id, "/Connections/Areas");
  dataspace_id = H5Dget_space(dataset_id);
  status = H5Sget_simple_extent_dims(dataspace_id, &dims[1], NULL);
  status = H5Sclose(dataspace_id);
  ierr = PetscMalloc5(dims[1],double,&data->faceAreas,dims[1],int,&data->downCells,dims[1],double,&data->downX,dims[1],double,&data->downY,dims[1],double,&data->downZ);CHKERRQ(ierr);
  status = H5Dread(dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data->faceAreas);
  status = H5Dclose(dataset_id);
  dataset_id = H5Dopen(file_id, "/Connections/Downwind Cell IDs");
  status = H5Dread(dataset_id, H5T_STD_I32LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data->downCells);
  status = H5Dclose(dataset_id);
  dataset_id = H5Dopen(file_id, "/Connections/Downwind Distance X");
  status = H5Dread(dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data->downX);
  status = H5Dclose(dataset_id);
  dataset_id = H5Dopen(file_id, "/Connections/Downwind Distance Y");
  status = H5Dread(dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data->downY);
  status = H5Dclose(dataset_id);
  dataset_id = H5Dopen(file_id, "/Connections/Downwind Distance Z");
  status = H5Dread(dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data->downZ);
  status = H5Dclose(dataset_id);
  ierr = PetscMalloc4(dims[1],int,&data->upCells,dims[1],double,&data->upX,dims[1],double,&data->upY,dims[1],double,&data->upZ);CHKERRQ(ierr);
  dataset_id = H5Dopen(file_id, "/Connections/Upwind Cell IDs");
  status = H5Dread(dataset_id, H5T_STD_I32LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data->upCells);
  status = H5Dclose(dataset_id);
  dataset_id = H5Dopen(file_id, "/Connections/Upwind Distance X");
  status = H5Dread(dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data->upX);
  status = H5Dclose(dataset_id);
  dataset_id = H5Dopen(file_id, "/Connections/Upwind Distance Y");
  status = H5Dread(dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data->upY);
  status = H5Dclose(dataset_id);
  dataset_id = H5Dopen(file_id, "/Connections/Upwind Distance Z");
  status = H5Dread(dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data->upZ);
  status = H5Dclose(dataset_id);
  /* Cleanup */
  ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
  data->numCells = dims[0];
  data->numFaces = dims[1];
  for(int c = 0; c < data->numCells; ++c) {
    std::cout << "Cell: " << data->cellIds[c] << " Center: " << data->cellX[c]<<","<<data->cellY[c]<<","<<data->cellZ[c] << " Vol: " << data->cellVols[c] << std::endl;
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
#if 0
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
  PFLOTRANMesh data;

  ierr = ReadMesh(comm, &data, options);CHKERRQ(ierr);
  if (!m->commRank()) {
    for(int f = 0; f < data.numFaces; ++f) {
      sieve->addCone(f+data.numCells, data.downCells[f]);
      sieve->addCone(f+data.numCells, data.upCells[f]);
    }
  }
  m->setSieve(sieve);
  m->stratify();
  const Obj<ALE::Mesh::real_section_type>& sCellVols = m->getRealSection("cell volumes");
  const Obj<ALE::Mesh::real_section_type>& sCellX    = m->getRealSection("cell centers X");
  const Obj<ALE::Mesh::real_section_type>& sCellY    = m->getRealSection("cell centers Y");
  const Obj<ALE::Mesh::real_section_type>& sCellZ    = m->getRealSection("cell centers Z");
  const Obj<ALE::Mesh::label_sequence>&    cells     = m->heightStratum(0);

  sCellVols->setFiberDimension(cells, 1);
  sCellX->setFiberDimension(cells, 1);
  sCellY->setFiberDimension(cells, 1);
  sCellZ->setFiberDimension(cells, 1);
  m->allocate(sCellVols);
  m->allocate(sCellX);
  m->allocate(sCellY);
  m->allocate(sCellZ);
  for(ALE::Mesh::label_sequence::iterator c_iter = cells->begin(); c_iter != cells->end(); ++c_iter) {
    sCellVols->updatePoint(*c_iter, &data.cellVols[*c_iter]);
    sCellX->updatePoint(*c_iter, &data.cellX[*c_iter]);
    sCellY->updatePoint(*c_iter, &data.cellY[*c_iter]);
    sCellZ->updatePoint(*c_iter, &data.cellZ[*c_iter]);
  }
  const Obj<ALE::Mesh::real_section_type>& faceArea = m->getRealSection("face areas");
  // magnitude, upwind fraction, unit_x, unit_y, unit_z
  const Obj<ALE::Mesh::real_section_type>& faceInfo = m->getRealSection("face info");
  const Obj<ALE::Mesh::label_sequence>&    faces    = m->depthStratum(0);

  faceArea->setFiberDimension(faces, 1);
  faceInfo->setFiberDimension(faces, 5);
  m->allocate(faceArea);
  m->allocate(faceInfo);
  for(ALE::Mesh::label_sequence::iterator f_iter = faces->begin(); f_iter != faces->end(); ++f_iter) {
    double dirX = data.upX[*f_iter - data.numCells] + data.downX[*f_iter - data.numCells];
    double dirY = data.upY[*f_iter - data.numCells] + data.downY[*f_iter - data.numCells];
    double dirZ = data.upZ[*f_iter - data.numCells] + data.downZ[*f_iter - data.numCells];
    double mag  = sqrt(dirX*dirX + dirY*dirY + dirZ*dirZ);
    double info[5];

    faceArea->updatePoint(*f_iter, &data.faceAreas[*f_iter - data.numCells]);
    info[0] = mag;
    info[1] = (data.upX[*f_iter - data.numCells]+data.upY[*f_iter - data.numCells]+data.upZ[*f_iter - data.numCells])/(dirX+dirY+dirZ);
    info[2] = dirX/mag;
    info[3] = dirY/mag;
    info[4] = dirZ/mag;
    faceInfo->updatePoint(*f_iter, info);
  }
  ierr = PetscFree5(data.cellIds,data.cellVols,data.cellX,data.cellY,data.cellZ);CHKERRQ(ierr);
  ierr = PetscFree5(data.faceAreas,data.downCells,data.downX,data.downY,data.downZ);CHKERRQ(ierr);
  ierr = PetscFree4(data.upCells,data.upX,data.upY,data.upZ);CHKERRQ(ierr);

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
