/*
  Wrappers for mkl_cspblas_ routines.
  A more elegant way to do this would be to use an approach like that used in petsclbaslapack_mangle.h,
  but since the MKL sparse BLAS routines are not going to be as widely used, and because
  we don't have to worry about Fortran name mangling, this seems OK for now.
*/

/* Have to redefine MKL_Complex16 and MKL_Complex8 as PetscScalar for the complex number cases.
 * This works fine with a C99 compiler -- still need to verify that this works with C89.
 * Note: These definitions need to occur BEFORE including MKL headers. */
#define MKL_Complex16 PetscScalar
#define MKL_Complex8 PetscScalar

#if !defined(PETSC_USE_COMPLEX)
# if defined(PETSC_USE_REAL_SINGLE)
#   define mkl_cspblas_xcsrgemv(transa,m,a,ia,ja,x,y) mkl_cspblas_scsrgemv(transa,m,a,ia,ja,x,y)
# elif defined(PETSC_USE_REAL_DOUBLE)
#   define mkl_cspblas_xcsrgemv(transa,m,a,ia,ja,x,y) mkl_cspblas_dcsrgemv(transa,m,a,ia,ja,x,y)
# endif
#else
# if defined(PETSC_USE_REAL_SINGLE)
#   define mkl_cspblas_xcsrgemv(transa,m,a,ia,ja,x,y) mkl_cspblas_ccsrgemv(transa,m,a,ia,ja,x,y)
# elif defined(PETSC_USE_REAL_DOUBLE)
#   define mkl_cspblas_xcsrgemv(transa,m,a,ia,ja,x,y) mkl_cspblas_zcsrgemv(transa,m,a,ia,ja,x,y)
# endif
#endif

/* Note: MKL releases prior to the end of 2014 do not have a const-correct interface -> ugly casts necessary.
         Does not apply to mkl_sparse_x_*()-routines, because these have been introduced later. */
#if !defined(PETSC_USE_COMPLEX)
# if defined(PETSC_USE_REAL_SINGLE)
#   define mkl_xcsrmv(transa,m,k,alpha,matdescra,val,indx,pntrb,pntre,x,beta,y) mkl_scsrmv(transa,m,k,alpha,matdescra,(MatScalar*)val,(PetscInt*)indx,(PetscInt*)pntrb,(PetscInt*)pntre,(PetscScalar*)x,beta,y)
# elif defined(PETSC_USE_REAL_DOUBLE)
#   define mkl_xcsrmv(transa,m,k,alpha,matdescra,val,indx,pntrb,pntre,x,beta,y) mkl_dcsrmv(transa,m,k,alpha,matdescra,(MatScalar*)val,(PetscInt*)indx,(PetscInt*)pntrb,(PetscInt*)pntre,(PetscScalar*)x,beta,y)
# endif
#else
# if defined(PETSC_USE_REAL_SINGLE)
#   define mkl_xcsrmv(transa,m,k,alpha,matdescra,val,indx,pntrb,pntre,x,beta,y) mkl_ccsrmv(transa,m,k,alpha,matdescra,(MatScalar*)val,(PetscInt*)indx,(PetscInt*)pntrb,(PetscInt*)pntre,(PetscScalar*)x,beta,y)
# elif defined(PETSC_USE_REAL_DOUBLE)
#   define mkl_xcsrmv(transa,m,k,alpha,matdescra,val,indx,pntrb,pntre,x,beta,y) mkl_zcsrmv(transa,m,k,alpha,matdescra,(MatScalar*)val,(PetscInt*)indx,(PetscInt*)pntrb,(PetscInt*)pntre,(PetscScalar*)x,beta,y)
# endif
#endif

#if !defined(PETSC_USE_COMPLEX)
# if defined(PETSC_USE_REAL_SINGLE)
#   define mkl_sparse_x_create_csr(A,indexing,rows,cols,rows_start,rows_end,col_indx,values) mkl_sparse_s_create_csr(A,indexing,rows,cols,rows_start,rows_end,col_indx,values)
# elif defined(PETSC_USE_REAL_DOUBLE)
#   define mkl_sparse_x_create_csr(A,indexing,rows,cols,rows_start,rows_end,col_indx,values) mkl_sparse_d_create_csr(A,indexing,rows,cols,rows_start,rows_end,col_indx,values)
# endif
#else
# if defined(PETSC_USE_REAL_SINGLE)
#   define mkl_sparse_x_create_csr(A,indexing,rows,cols,rows_start,rows_end,col_indx,values) mkl_sparse_c_create_csr(A,indexing,rows,cols,rows_start,rows_end,col_indx,values)
# elif defined(PETSC_USE_REAL_DOUBLE)
#   define mkl_sparse_x_create_csr(A,indexing,rows,cols,rows_start,rows_end,col_indx,values) mkl_sparse_z_create_csr(A,indexing,rows,cols,rows_start,rows_end,col_indx,values)
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

#if !defined(PETSC_USE_COMPLEX)
# if defined(PETSC_USE_REAL_SINGLE)
#   define mkl_sparse_x_export_csr(source,indexing,rows,cols,rows_start,rows_end,col_indx,values) mkl_sparse_s_export_csr(source,indexing,rows,cols,rows_start,rows_end,col_indx,values)
# elif defined(PETSC_USE_REAL_DOUBLE)
#   define mkl_sparse_x_export_csr(source,indexing,rows,cols,rows_start,rows_end,col_indx,values) mkl_sparse_d_export_csr(source,indexing,rows,cols,rows_start,rows_end,col_indx,values)
# endif
#else
# if defined(PETSC_USE_REAL_SINGLE)
#   define mkl_sparse_x_export_csr(source,indexing,rows,cols,rows_start,rows_end,col_indx,values) mkl_sparse_c_export_csr(source,indexing,rows,cols,rows_start,rows_end,col_indx,values)
# elif defined(PETSC_USE_REAL_DOUBLE)
#   define mkl_sparse_x_export_csr(source,indexing,rows,cols,rows_start,rows_end,col_indx,values) mkl_sparse_z_export_csr(source,indexing,rows,cols,rows_start,rows_end,col_indx,values)
# endif
#endif
