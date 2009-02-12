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
  hsize_t numCells;
  int     *cellIds; /* Id of each cell */
  double  *cellX;   /* X of cell center */
  double  *cellY;   /* Y of cell center */
  double  *cellZ;   /* Z of cell center */
  double  *cellVols;
  hsize_t numFaces;
  double  *faceAreas;
  int     *downCells;
  int     *upCells;
  double  *downX;
  double  *downY;
  double  *downZ;
  double  *upX;
  double  *upY;
  double  *upZ;
} PFLOTRANMesh;


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
  PetscViewer     binaryviewer;
  Vec             cellCenters;
  PetscViewer    hdf5viewer;
  hid_t          file_id, dataset_id, dataspace_id;
  herr_t         status;
  
  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &args, (char *) 0, help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PETSC_ERR_SUP,"This preprocessor runs only on one process");

  /* Open Glenn's file */
  ierr = PetscViewerCreate(PETSC_COMM_SELF, &hdf5viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(hdf5viewer, PETSC_VIEWER_HDF5);CHKERRQ(ierr);
  ierr = PetscViewerFileSetMode(hdf5viewer, FILE_MODE_READ);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(hdf5viewer, "mesh.h5");CHKERRQ(ierr);
  ierr = PetscViewerHDF5GetFileId(hdf5viewer, &file_id);CHKERRQ(ierr);

  /* get number of cells and then number of edges */
  dataset_id = H5Dopen(file_id, "/Cells/Natural IDs");
  dataspace_id = H5Dget_space(dataset_id);
  status = H5Sget_simple_extent_dims(dataspace_id, &data.numCells, NULL);if (status < 0) SETERRQ(PETSC_ERR_LIB,"Bad dimension");
  status = H5Sclose(dataspace_id);CHKERRQ(status);
  status = H5Dclose(dataset_id);CHKERRQ(status);
  dataset_id = H5Dopen(file_id, "/Connections/Areas");
  dataspace_id = H5Dget_space(dataset_id);
  status = H5Sget_simple_extent_dims(dataspace_id, &data.numFaces, NULL);if (status < 0) SETERRQ(PETSC_ERR_LIB,"Bad dimension");
  status = H5Sclose(dataspace_id);CHKERRQ(status);
  status = H5Dclose(dataset_id);CHKERRQ(status);
  ierr = PetscPrintf(PETSC_COMM_SELF, "Number of cells %D Number of faces %D \n",(PetscInt)data.numCells,(PetscInt)data.numFaces);CHKERRQ(ierr);

  /* read face data */
  ierr = PetscMalloc5(data.numFaces,double,&data.faceAreas,data.numFaces,int,&data.downCells,data.numFaces,double,&data.downX,data.numFaces,double,&data.downY,data.numFaces,double,&data.downZ);CHKERRQ(ierr);
  dataset_id = H5Dopen(file_id, "/Connections/Areas");
  status = H5Dread(dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.faceAreas);CHKERRQ(status);
  status = H5Dclose(dataset_id);CHKERRQ(status);
  dataset_id = H5Dopen(file_id, "/Connections/Downwind Cell IDs");
  status = H5Dread(dataset_id, H5T_STD_I32LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.downCells);CHKERRQ(status);
  status = H5Dclose(dataset_id);CHKERRQ(status);
  dataset_id = H5Dopen(file_id, "/Connections/Downwind Distance X");
  status = H5Dread(dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.downX);CHKERRQ(status);
  status = H5Dclose(dataset_id);CHKERRQ(status);
  dataset_id = H5Dopen(file_id, "/Connections/Downwind Distance Y");
  status = H5Dread(dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.downY);CHKERRQ(status);
  status = H5Dclose(dataset_id);CHKERRQ(status);
  dataset_id = H5Dopen(file_id, "/Connections/Downwind Distance Z");
  status = H5Dread(dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.downZ);CHKERRQ(status);
  status = H5Dclose(dataset_id);CHKERRQ(status);
  ierr = PetscMalloc4(data.numFaces,int,&data.upCells,data.numFaces,double,&data.upX,data.numFaces,double,&data.upY,data.numFaces,double,&data.upZ);CHKERRQ(ierr);
  dataset_id = H5Dopen(file_id, "/Connections/Upwind Cell IDs");
  status = H5Dread(dataset_id, H5T_STD_I32LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.upCells);CHKERRQ(status);
  status = H5Dclose(dataset_id);CHKERRQ(status);
  dataset_id = H5Dopen(file_id, "/Connections/Upwind Distance X");
  status = H5Dread(dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.upX);CHKERRQ(status);
  status = H5Dclose(dataset_id);CHKERRQ(status);
  dataset_id = H5Dopen(file_id, "/Connections/Upwind Distance Y");
  status = H5Dread(dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.upY);CHKERRQ(status);
  status = H5Dclose(dataset_id);CHKERRQ(status);
  dataset_id = H5Dopen(file_id, "/Connections/Upwind Distance Z");
  status = H5Dread(dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.upZ);CHKERRQ(status);
  status = H5Dclose(dataset_id);CHKERRQ(status);


  // Put face data into matrix 
  ierr = MatCreate(PETSC_COMM_WORLD, &Adj);CHKERRQ(ierr);
  ierr = MatSetSizes(Adj, data.numCells*bs, data.numCells*bs, PETSC_DECIDE, PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Adj);CHKERRQ(ierr);
  ierr = MatSetType(Adj,MATSEQBAIJ);CHKERRQ(ierr);
  ierr = MatSeqBAIJSetPreallocation(Adj, bs, 6,PETSC_NULL);CHKERRQ(ierr);
  //ierr = MatSetType(Adj,MATSEQAIJ);CHKERRQ(ierr);
  //ierr = MatSeqAIJSetPreallocation(Adj, 6,PETSC_NULL);CHKERRQ(ierr);
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
    //ierr = MatSetValues(Adj, 1, &data.downCells[i], 1, &data.upCells[i], values, INSERT_VALUES);CHKERRQ(ierr);
    //ierr = MatSetValues(Adj, 1, &data.upCells[i], 1, &data.downCells[i], values, INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(Adj, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Adj, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscFree5(data.faceAreas, data.downCells, data.downX, data.downY, data.downZ);CHKERRQ(ierr);
  ierr = PetscFree4(data.upCells, data.upX, data.upY, data.upZ);CHKERRQ(ierr);

  ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF,"mesh.petsc", FILE_MODE_WRITE,&binaryviewer);CHKERRQ(ierr);
  ierr = MatView(Adj, binaryviewer);CHKERRQ(ierr);
  ierr = MatDestroy(Adj);CHKERRQ(ierr);

  /* read cell information */
  ierr = PetscMalloc5(data.numCells,int,&data.cellIds,data.numCells,double,&data.cellVols,data.numCells,double,&data.cellX,data.numCells,double,&data.cellY,data.numCells,double,&data.cellZ);CHKERRQ(ierr);
  dataset_id = H5Dopen(file_id, "/Cells/Natural IDs");
  status = H5Dread(dataset_id, H5T_STD_I32LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.cellIds);CHKERRQ(status);
  status = H5Dclose(dataset_id);CHKERRQ(status);
  dataset_id = H5Dopen(file_id, "/Cells/Volumes");
  status = H5Dread(dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.cellVols);CHKERRQ(status);
  status = H5Dclose(dataset_id);CHKERRQ(status);
  dataset_id = H5Dopen(file_id, "/Cells/X-Coordinates");
  status = H5Dread(dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.cellX);CHKERRQ(status);
  status = H5Dclose(dataset_id);CHKERRQ(status);
  dataset_id = H5Dopen(file_id, "/Cells/Y-Coordinates");
  status = H5Dread(dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.cellY);CHKERRQ(status);
  status = H5Dclose(dataset_id);CHKERRQ(status);
  dataset_id = H5Dopen(file_id, "/Cells/Z-Coordinates");
  status = H5Dread(dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.cellZ);CHKERRQ(status);
  status = H5Dclose(dataset_id);CHKERRQ(status);
  ierr = PetscViewerDestroy(hdf5viewer);CHKERRQ(ierr);

  /* put cell information into vectors */
  ierr = VecCreateSeq(PETSC_COMM_SELF,3*data.numCells,&cellCenters);CHKERRQ(ierr);
  ierr = VecSetBlockSize(cellCenters,3);CHKERRQ(ierr);
  ierr = VecGetArray(cellCenters,&cc);CHKERRQ(ierr);
  for (i=0; i<data.numCells; i++) {
    cc[3*i]   = data.cellX[i];
    cc[3*i+1] = data.cellY[i];
    cc[3*i+2] = data.cellZ[i];
  }
  ierr = VecRestoreArray(cellCenters,&cc);CHKERRQ(ierr);
  ierr = VecView(cellCenters,binaryviewer);CHKERRQ(ierr);

  ierr = VecGetArray(cellCenters,&cc);CHKERRQ(ierr);
  for (i=0; i<data.numCells; i++) {
    cc[3*i]   = data.cellIds[i];
    cc[3*i+1] = data.cellVols[i];
    cc[3*i+2] = 0.0;
  }
  ierr = VecRestoreArray(cellCenters,&cc);CHKERRQ(ierr);
  ierr = VecView(cellCenters,binaryviewer);CHKERRQ(ierr);
  ierr = PetscFree5(data.cellIds, data.cellVols, data.cellX, data.cellY, data.cellZ);CHKERRQ(ierr);
  ierr = VecDestroy(cellCenters);
  ierr = PetscViewerDestroy(binaryviewer);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
