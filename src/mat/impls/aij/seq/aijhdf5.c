
#include <../src/mat/impls/aij/seq/aij.h>
#include <petscviewerhdf5.h>

#if defined(PETSC_HAVE_HDF5)
PetscErrorCode MatLoad_AIJ_HDF5(Mat mat, PetscViewer viewer)
{
  hid_t           file_id,group_matrix_id,dset_id,dspace_id,mspace_id,dtype;
  hsize_t         h5_count_i,h5_offset_i,h5_count_data,h5_offset_data,h5_dims[4];

  PetscInt        count_i,offset_i,count_data,offset_data,offset_i_end;
  PetscInt        *i = NULL,*j = NULL;
  PetscReal       *a = NULL;
  const char      *a_name,*i_name,*j_name,*mat_name,*c_name;
  int             rdim;
  PetscInt        p,m,M,N;
  PetscInt        bs = mat->rmap->bs;
  PetscBool       flg;

  PetscErrorCode  ierr;
  MPI_Comm        comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)mat,&comm);CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject)mat,&mat_name);CHKERRQ(ierr);
  ierr = PetscViewerHDF5GetAIJNames(viewer,&i_name,&j_name,&a_name,&c_name);CHKERRQ(ierr);

  ierr = PetscOptionsBegin(comm,NULL,"Options for loading matrix from HDF5","Mat");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-matload_block_size","Set the blocksize used to store the matrix","MatLoad",bs,&bs,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (flg) {
    ierr = MatSetBlockSize(mat, bs);CHKERRQ(ierr);
  }

  ierr = PetscViewerHDF5PushGroup(viewer,mat_name);CHKERRQ(ierr);
  ierr = PetscViewerHDF5OpenGroup(viewer,&file_id,&group_matrix_id);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);

  ierr = PetscViewerHDF5ReadAttribute(viewer,mat_name,c_name,PETSC_INT,&N);CHKERRQ(ierr);

  PetscStackCallHDF5Return(dset_id,H5Dopen,(group_matrix_id,i_name,H5P_DEFAULT));
  PetscStackCallHDF5Return(dspace_id,H5Dget_space,(dset_id));
  PetscStackCallHDF5Return(rdim,H5Sget_simple_extent_dims,(dspace_id,h5_dims,NULL));
  M = (PetscInt) h5_dims[rdim-1] - 1;
  PetscStackCallHDF5(H5Sclose,(dspace_id));
  PetscStackCallHDF5(H5Dclose,(dset_id));

  m = PETSC_DECIDE;
  ierr = PetscSplitOwnershipBlock(comm,bs,&m,&M);CHKERRQ(ierr);
  ierr = MPI_Scan(&m,&offset_i_end,1,MPIU_INT,MPI_SUM,comm);CHKERRQ(ierr);

  if (m > 0) {
    /* Read array i (array of row indices) */
    PetscStackCallHDF5Return(dset_id,H5Dopen,(group_matrix_id,i_name,H5P_DEFAULT));
    PetscStackCallHDF5Return(dspace_id,H5Dget_space,(dset_id));

    ierr = PetscMalloc((m + 1) * sizeof(PetscInt),&i);CHKERRQ(ierr);

    /* Determine offset and count of elements for reading local part of array i */
    offset_i = offset_i_end - m;
    ierr = PetscHDF5IntCast(offset_i,&h5_offset_i);CHKERRQ(ierr);
    count_i = m + 1;
    ierr = PetscHDF5IntCast(count_i,&h5_count_i);CHKERRQ(ierr);

    /* Read local part of array i */
    PetscStackCallHDF5Return(mspace_id,H5Screate_simple,(1,&h5_count_i,NULL));
    PetscStackCallHDF5(H5Sselect_hyperslab,(dspace_id,H5S_SELECT_SET,&h5_offset_i,NULL,&h5_count_i,NULL));
    ierr = PetscDataTypeToHDF5DataType(PETSC_INT,&dtype);CHKERRQ(ierr);
    PetscStackCallHDF5(H5Dread,(dset_id,dtype,mspace_id,dspace_id,H5P_DEFAULT,i));

    PetscStackCallHDF5(H5Sclose,(mspace_id));
    PetscStackCallHDF5(H5Sclose,(dspace_id));
    PetscStackCallHDF5(H5Dclose,(dset_id));

    /* Determine offset and count of elements for reading local part of array data (a, j) */
    offset_data = i[0];
    ierr = PetscHDF5IntCast(offset_data,&h5_offset_data);CHKERRQ(ierr);
    count_data = i[count_i - 1] - i[0];
    ierr = PetscHDF5IntCast(count_data,&h5_count_data);CHKERRQ(ierr);

    /* Read array j (array of column indices) */
    PetscStackCallHDF5Return(dset_id,H5Dopen,(group_matrix_id,j_name,H5P_DEFAULT));
    PetscStackCallHDF5Return(dspace_id,H5Dget_space,(dset_id));

    ierr = PetscMalloc(count_data * sizeof(PetscInt),&j);CHKERRQ(ierr);

    /* Read local part of array j */
    PetscStackCallHDF5Return(mspace_id,H5Screate_simple,(1,&h5_count_data,NULL));
    PetscStackCallHDF5(H5Sselect_hyperslab,(dspace_id,H5S_SELECT_SET,&h5_offset_data,NULL,&h5_count_data,NULL));
    ierr = PetscDataTypeToHDF5DataType(PETSC_INT,&dtype);CHKERRQ(ierr);
    PetscStackCallHDF5(H5Dread,(dset_id,dtype,mspace_id,dspace_id,H5P_DEFAULT,j));

    PetscStackCallHDF5(H5Sclose,(mspace_id));
    PetscStackCallHDF5(H5Sclose,(dspace_id));
    PetscStackCallHDF5(H5Dclose,(dset_id));

    /* Read array a (array of values) */
    PetscStackCallHDF5Return(dset_id,H5Dopen,(group_matrix_id,a_name,H5P_DEFAULT));
    PetscStackCallHDF5Return(dspace_id,H5Dget_space,(dset_id));

    ierr = PetscMalloc(count_data * sizeof(PetscReal),&a);CHKERRQ(ierr);

    /* Read local part of array a */
    PetscStackCallHDF5Return(mspace_id,H5Screate_simple,(1,&h5_count_data,NULL));
    PetscStackCallHDF5(H5Sselect_hyperslab,(dspace_id,H5S_SELECT_SET,&h5_offset_data,NULL,&h5_count_data,NULL));
    ierr = PetscDataTypeToHDF5DataType(PETSC_REAL,&dtype);CHKERRQ(ierr);
    PetscStackCallHDF5(H5Dread,(dset_id,dtype,mspace_id,dspace_id,H5P_DEFAULT,a));

    PetscStackCallHDF5(H5Sclose,(mspace_id));
    PetscStackCallHDF5(H5Sclose,(dspace_id));
    PetscStackCallHDF5(H5Dclose,(dset_id));
  } else {
    a = NULL;
    j = NULL;
    ierr = PetscMalloc(sizeof(PetscInt),&i);CHKERRQ(ierr);
    i[0] = 0;
  }

  /* close group */
  PetscStackCallHDF5(H5Gclose,(group_matrix_id));

  /* Converting global to local indexing of rows */
  for (p=1; p<count_i; ++p) i[p] -= i[0];
  i[0] = 0;

  /* create matrix */
  ierr = MatSetSizes(mat,m,PETSC_DETERMINE,M,N);CHKERRQ(ierr);
  ierr = MatSetBlockSize(mat,bs);CHKERRQ(ierr);
  if (!((PetscObject)mat)->type_name) {
    ierr = MatSetType(mat,MATAIJ);CHKERRQ(ierr);
  }
  ierr = MatSeqAIJSetPreallocationCSR(mat,i,j,a);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocationCSR(mat,i,j,a);CHKERRQ(ierr);
  /*
  ierr = MatSeqBAIJSetPreallocationCSR(mat,bs,i,j,a);CHKERRQ(ierr);
  ierr = MatMPIBAIJSetPreallocationCSR(mat,bs,i,j,a);CHKERRQ(ierr);
  */

  ierr = PetscFree(i);CHKERRQ(ierr);
  ierr = PetscFree(j);CHKERRQ(ierr);
  ierr = PetscFree(a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

