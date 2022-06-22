/*
   Define type spbas_matrix: sparse matrices using pointers

   Global matrix information
      nrows, ncols: dimensions
      nnz         : number of nonzeros (in whole matrix)
      col_idx_type: storage scheme for column numbers
                    SPBAS_COLUMN_NUMBERS:
                        array icol contains column indices:
                           A(i,icol[i][j]) = values[i][j]
                    SPBAS_DIAGONAL_OFFSETS:
                        array icol contains diagonal offsets:
                           A(i,i+icol[i][j]) = values[i][j]
                    SPBAS_OFFSET_ARRAY:
                        array icol contains offsets wrt array
                        icol0:
                           A(i,icol0[i]+icol[i][j]) = values[i][j]

   Information about each row
      row_nnz     : number of nonzeros for each row
      icol0       : column index offset (when needed, otherwise NULL)
      icols       : array of diagonal offsets for each row, as descibed
                    for col_idx_type, above
      values      : array of matrix entries for each row
                    when values == NULL, this matrix is really
                    a sparseness pattern, not a matrix

   The other fields describe the way in which the data are stored
   in memory.

      block_data  : The pointers icols[i] all point to places in a
                    single allocated array. Only for icols[0] was
                    malloc called. Freeing icols[0] will free
                    all other icols=arrays as well.
                    Same for arrays values[i]
*/

#define SPBAS_COLUMN_NUMBERS   (0)
#define SPBAS_DIAGONAL_OFFSETS (1)
#define SPBAS_OFFSET_ARRAY     (2)

#define NEGATIVE_DIAGONAL (-42)

typedef struct {
  PetscInt nrows;
  PetscInt ncols;
  PetscInt nnz;
  PetscInt col_idx_type;

  PetscInt    *row_nnz;
  PetscInt    *icol0;
  PetscInt    **icols;
  PetscScalar **values;

  PetscBool   block_data;
  PetscInt    n_alloc_icol;
  PetscInt    n_alloc_val;
  PetscInt    *alloc_icol;
  PetscScalar *alloc_val;
} spbas_matrix;

/*
  spbas_compress_pattern:
     calculate a compressed sparseness pattern for a sparseness pattern
     given in compressed row storage. The compressed sparseness pattern may
     require (much) less memory.

  spbas_memory_requirement:
     Calculate the number of bytes needed to store tha matrix

  spbas_incomplete_cholesky:
     Incomplete Cholesky decomposition

  spbas_delete:
     de-allocate the arrays owned by this matrix

  spbas_matrix_to_crs:
     Convert an spbas_matrix to compessed row storage

  spbas_dump:
     Print the matrix in i,j,val-format

  spbas_transpose:
     Return the transpose of a matrix

  spbas_pattern_only:
     Return the sparseness pattern (matrix without values) of a
     compressed row storage
*/
PetscErrorCode spbas_compress_pattern(PetscInt*,PetscInt*,PetscInt,PetscInt,PetscInt,spbas_matrix*,PetscReal*);
size_t         spbas_memory_requirement(spbas_matrix);
PetscErrorCode spbas_delete(spbas_matrix);
PetscErrorCode spbas_incomplete_cholesky(Mat,const PetscInt*,const PetscInt*,spbas_matrix,PetscReal,PetscReal,spbas_matrix*,PetscBool*);
PetscErrorCode spbas_matrix_to_crs(spbas_matrix, MatScalar **,PetscInt **,PetscInt**);
PetscErrorCode spbas_dump(const char*,spbas_matrix);
PetscErrorCode spbas_transpose(spbas_matrix,spbas_matrix*);
PetscErrorCode spbas_apply_reordering(spbas_matrix*, const PetscInt*, const PetscInt*);
PetscErrorCode spbas_pattern_only(PetscInt, PetscInt, PetscInt*, PetscInt*, spbas_matrix*);
PetscErrorCode spbas_power (spbas_matrix, PetscInt, spbas_matrix*);
PetscErrorCode spbas_keep_upper(spbas_matrix*);
