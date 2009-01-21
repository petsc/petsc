static char help[] = "Reads in a finite volume mesh for Flotran and saves it as PETSc data files.\n\n";

#include "petscmat.h"

/*
Barry and Matt,

Attached is a sample hdf5 file for reading into PETSc.  The file
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
PetscErrorCode ReadMesh(MPI_Comm comm, PFLOTRANMesh *data)
{
  PetscViewer    viewer;
  hid_t          file_id, dataset_id, dataspace_id;
  herr_t         status;
  hsize_t        dims[2];
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  data->numCells = 0;
  data->numFaces = 0;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = PetscViewerCreate(comm, &viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(viewer, PETSC_VIEWER_HDF5);CHKERRQ(ierr);
  ierr = PetscViewerFileSetMode(viewer, FILE_MODE_READ);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(viewer, "mesh.h5");CHKERRQ(ierr);
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
  data->numCells = dims[0];
  data->numFaces = dims[1];
  /* Cleanup */
  ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF, "Number of cells %D Number of faces %D \n",data->numCells,data->numFaces);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *args[])
{
  PFLOTRANMesh    data;
  Mat             Adj;       /* The adjacency matrix of the mesh */
  PetscInt        bs = 3;
  PetscScalar     values[9],*cc;
  PetscMPIInt     size;
  PetscInt        i;
  PetscErrorCode  ierr;
  PetscViewer     viewer;
  Vec             cellCenters;
  
  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &args, (char *) 0, help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PETSC_ERR_SUP,"This preprocessor runs only on one process");

  // Read in the matrix from Glenn's file
  ierr = ReadMesh(PETSC_COMM_WORLD, &data);CHKERRQ(ierr);
  // Create adjacency matrix (there are 9 pieces of data per edge)
  ierr = MatCreate(PETSC_COMM_WORLD, &Adj);CHKERRQ(ierr);
  ierr = MatSetSizes(Adj, data.numCells*bs, data.numCells*bs, PETSC_DECIDE, PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Adj);CHKERRQ(ierr);
  ierr = MatSetType(Adj,MATSEQBAIJ);CHKERRQ(ierr);
  ierr = MatSeqBAIJSetPreallocation(Adj, bs, 6,PETSC_NULL);CHKERRQ(ierr);
  for(i = 0; i < data.numFaces; ++i) {
    values[0] = data.faceAreas[i];
    values[1] = data.downCells[i];
    values[2] = data.downX[i];
    values[3] = data.downY[i];
    values[4] = data.downZ[i];
    values[5] = data.upCells[i];
    values[6] = data.upX[i];
    values[7] = data.upY[i];
    values[8] = data.upZ[i];
    ierr = MatSetValuesBlocked(Adj, 1, &data.downCells[i], 1, &data.upCells[i], values, INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatSetValuesBlocked(Adj, 1, &data.upCells[i], 1, &data.downCells[i], values, INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(Adj, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Adj, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscFree5(data.faceAreas, data.downCells, data.downX, data.downY, data.downZ);CHKERRQ(ierr);
  ierr = PetscFree4(data.upCells, data.upX, data.upY, data.upZ);CHKERRQ(ierr);

  ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF,"mesh.petsc", FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = MatView(Adj, viewer);CHKERRQ(ierr);
  ierr = MatDestroy(Adj);CHKERRQ(ierr);

  ierr = VecCreateSeq(PETSC_COMM_SELF,3*data.numCells,&cellCenters);CHKERRQ(ierr);
  ierr = VecSetBlockSize(cellCenters,3);CHKERRQ(ierr);
  ierr = VecGetArray(cellCenters,&cc);CHKERRQ(ierr);
  for (i=0; i<data.numCells; i++) {
    cc[3*i] = data.cellX[i];
    cc[3*i+1] = data.cellY[i];
    cc[3*i+2] = data.cellZ[i];
  }
  ierr = VecRestoreArray(cellCenters,&cc);CHKERRQ(ierr);
  ierr = PetscFree5(data.cellIds, data.cellVols, data.cellX, data.cellY, data.cellZ);CHKERRQ(ierr);
  ierr = VecView(cellCenters,viewer);CHKERRQ(ierr);
  ierr = VecDestroy(cellCenters);
  ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
