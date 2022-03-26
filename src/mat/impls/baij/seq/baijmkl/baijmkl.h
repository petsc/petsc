#define MKL_Complex16 PetscScalar
#define MKL_Complex8 PetscScalar
#define PetscCallMKL(stat) do {if (stat != SPARSE_STATUS_SUCCESS){PetscFunctionReturn(PETSC_ERR_LIB);}}while (0)
#if !defined(PETSC_USE_COMPLEX)
# if defined(PETSC_USE_REAL_SINGLE)
#   define mkl_sparse_x_create_bsr(A,indexing,block_layout,rows,cols,block_size,rows_start,rows_end,col_indx,values) mkl_sparse_s_create_bsr(A,indexing,block_layout,rows,cols,block_size,rows_start,rows_end,col_indx,values)
# elif defined(PETSC_USE_REAL_DOUBLE)
#   define mkl_sparse_x_create_bsr(A,indexing,block_layout,rows,cols,block_size,rows_start,rows_end,col_indx,values) mkl_sparse_d_create_bsr(A,indexing,block_layout,rows,cols,block_size,rows_start,rows_end,col_indx,values)
# endif
#else
# if defined(PETSC_USE_REAL_SINGLE)
#   define mkl_sparse_x_create_bsr(A,indexing,block_layout,rows,cols,block_size,rows_start,rows_end,col_indx,values) mkl_sparse_c_create_bsr(A,indexing,block_layout,rows,cols,block_size,rows_start,rows_end,col_indx,values)
# elif defined(PETSC_USE_REAL_DOUBLE)
#   define mkl_sparse_x_create_bsr(A,indexing,block_layout,rows,cols,block_size,rows_start,rows_end,col_indx,values) mkl_sparse_z_create_bsr(A,indexing,block_layout,rows,cols,block_size,rows_start,rows_end,col_indx,values)
# endif
#endif

#if !defined(PETSC_USE_COMPLEX)
# if defined(PETSC_USE_REAL_SINGLE)
#   define mkl_sparse_x_mv(operation,alpha,A,descr,x,beta,y) mkl_sparse_s_mv(operation,alpha,A,descr,x,beta,y)
# elif defined(PETSC_USE_REAL_DOUBLE)
#   define mkl_sparse_x_mv(operation,alpha,A,descr,x,beta,y) mkl_sparse_d_mv(operation,alpha,A,descr,x,beta,y)
# endif
#else
# if defined(PETSC_USE_REAL_SINGLE)
#   define mkl_sparse_x_mv(operation,alpha,A,descr,x,beta,y) mkl_sparse_c_mv(operation,alpha,A,descr,x,beta,y)
# elif defined(PETSC_USE_REAL_DOUBLE)
#   define mkl_sparse_x_mv(operation,alpha,A,descr,x,beta,y) mkl_sparse_z_mv(operation,alpha,A,descr,x,beta,y)
# endif
#endif
