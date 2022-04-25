#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/seq/bas/spbas.h>

/*MC
  MATSOLVERBAS -  Provides ICC(k) with drop tolerance

  Works with MATAIJ  matrices

  Options Database Keys:
+ -pc_factor_levels <l> - number of levels of fill
- -pc_factor_drop_tolerance - is not currently hooked up to do anything

  Level: intermediate

   Contributed by: Bas van 't Hof

   Notes:
    Since this currently hooked up to use drop tolerance it should produce the same factors and hence convergence as the PETSc ICC, for higher
     levels of fill it does not. This needs to be investigated. Unless you are interested in drop tolerance ICC and willing to work through the code
     we recommend not using this functionality.

.seealso: `PCFactorSetMatSolverType()`, `MatSolverType`, `PCFactorSetLevels()`, `PCFactorSetDropTolerance()`

M*/

/*
  spbas_memory_requirement:
    Calculate the number of bytes needed to store tha matrix
*/
size_t spbas_memory_requirement(spbas_matrix matrix)
{
  size_t memreq = 6 * sizeof(PetscInt)  + /* nrows, ncols, nnz, n_alloc_icol, n_alloc_val, col_idx_type */
                    sizeof(PetscBool)               + /* block_data */
                    sizeof(PetscScalar**)           + /* values */
                    sizeof(PetscScalar*)            + /* alloc_val */
                    2 * sizeof(PetscInt**)          + /* icols, icols0 */
                    2 * sizeof(PetscInt*)           + /* row_nnz, alloc_icol */
                    matrix.nrows * sizeof(PetscInt) + /* row_nnz[*] */
                    matrix.nrows * sizeof(PetscInt*); /* icols[*] */

  /* icol0[*] */
  if (matrix.col_idx_type == SPBAS_OFFSET_ARRAY) memreq += matrix.nrows * sizeof(PetscInt);

  /* icols[*][*] */
  if (matrix.block_data) memreq += matrix.n_alloc_icol * sizeof(PetscInt);
  else memreq += matrix.nnz * sizeof(PetscInt);

  if (matrix.values) {
    memreq += matrix.nrows * sizeof(PetscScalar*); /* values[*] */
    /* values[*][*] */
    if (matrix.block_data) memreq += matrix.n_alloc_val  * sizeof(PetscScalar);
    else memreq += matrix.nnz * sizeof(PetscScalar);
  }
  return memreq;
}

/*
  spbas_allocate_pattern:
    allocate the pattern arrays row_nnz, icols and optionally values
*/
PetscErrorCode spbas_allocate_pattern(spbas_matrix * result, PetscBool do_values)
{
  PetscInt nrows        = result->nrows;
  PetscInt col_idx_type = result->col_idx_type;

  PetscFunctionBegin;
  /* Allocate sparseness pattern */
  PetscCall(PetscMalloc1(nrows,&result->row_nnz));
  PetscCall(PetscMalloc1(nrows,&result->icols));

  /* If offsets are given wrt an array, create array */
  if (col_idx_type == SPBAS_OFFSET_ARRAY) {
    PetscCall(PetscMalloc1(nrows,&result->icol0));
  } else  {
    result->icol0 = NULL;
  }

  /* If values are given, allocate values array */
  if (do_values)  {
    PetscCall(PetscMalloc1(nrows,&result->values));
  } else {
    result->values = NULL;
  }
  PetscFunctionReturn(0);
}

/*
spbas_allocate_data:
   in case of block_data:
       Allocate the data arrays alloc_icol and optionally alloc_val,
       set appropriate pointers from icols and values;
   in case of !block_data:
       Allocate the arrays icols[i] and optionally values[i]
*/
PetscErrorCode spbas_allocate_data(spbas_matrix * result)
{
  PetscInt  i;
  PetscInt  nnz        = result->nnz;
  PetscInt  nrows      = result->nrows;
  PetscInt  r_nnz;
  PetscBool do_values  = (result->values) ? PETSC_TRUE : PETSC_FALSE;
  PetscBool block_data = result->block_data;

  PetscFunctionBegin;
  if (block_data) {
    /* Allocate the column number array and point to it */
    result->n_alloc_icol = nnz;

    PetscCall(PetscMalloc1(nnz, &result->alloc_icol));

    result->icols[0] = result->alloc_icol;
    for (i=1; i<nrows; i++)  {
      result->icols[i] = result->icols[i-1] + result->row_nnz[i-1];
    }

    /* Allocate the value array and point to it */
    if (do_values) {
      result->n_alloc_val = nnz;

      PetscCall(PetscMalloc1(nnz, &result->alloc_val));

      result->values[0] = result->alloc_val;
      for (i=1; i<nrows; i++) {
        result->values[i] = result->values[i-1] + result->row_nnz[i-1];
      }
    }
  } else {
    for (i=0; i<nrows; i++)   {
      r_nnz = result->row_nnz[i];
      PetscCall(PetscMalloc1(r_nnz, &result->icols[i]));
    }
    if (do_values) {
      for (i=0; i<nrows; i++)  {
        r_nnz = result->row_nnz[i];
        PetscCall(PetscMalloc1(r_nnz, &result->values[i]));
      }
    }
  }
  PetscFunctionReturn(0);
}

/*
  spbas_row_order_icol
     determine if row i1 should come
       + before row i2 in the sorted rows (return -1),
       + after (return 1)
       + is identical (return 0).
*/
int spbas_row_order_icol(PetscInt i1, PetscInt i2, PetscInt *irow_in, PetscInt *icol_in,PetscInt col_idx_type)
{
  PetscInt j;
  PetscInt nnz1    = irow_in[i1+1] - irow_in[i1];
  PetscInt nnz2    = irow_in[i2+1] - irow_in[i2];
  PetscInt * icol1 = &icol_in[irow_in[i1]];
  PetscInt * icol2 = &icol_in[irow_in[i2]];

  if (nnz1<nnz2) return -1;
  if (nnz1>nnz2) return 1;

  if (col_idx_type == SPBAS_COLUMN_NUMBERS) {
    for (j=0; j<nnz1; j++) {
      if (icol1[j]< icol2[j]) return -1;
      if (icol1[j]> icol2[j]) return 1;
    }
  } else if (col_idx_type == SPBAS_DIAGONAL_OFFSETS) {
    for (j=0; j<nnz1; j++) {
      if (icol1[j]-i1< icol2[j]-i2) return -1;
      if (icol1[j]-i1> icol2[j]-i2) return 1;
    }
  } else if (col_idx_type == SPBAS_OFFSET_ARRAY) {
    for (j=1; j<nnz1; j++) {
      if (icol1[j]-icol1[0] < icol2[j]-icol2[0]) return -1;
      if (icol1[j]-icol1[0] > icol2[j]-icol2[0]) return 1;
    }
  }
  return 0;
}

/*
  spbas_mergesort_icols:
    return a sorting of the rows in which identical sparseness patterns are
    next to each other
*/
PetscErrorCode spbas_mergesort_icols(PetscInt nrows, PetscInt * irow_in, PetscInt * icol_in,PetscInt col_idx_type, PetscInt *isort)
{
  PetscInt       istep;       /* Chunk-sizes of already sorted parts of arrays */
  PetscInt       i, i1, i2;   /* Loop counters for (partly) sorted arrays */
  PetscInt       istart, i1end, i2end; /* start of newly sorted array part, end of both  parts */
  PetscInt       *ialloc;     /* Allocated arrays */
  PetscInt       *iswap;      /* auxiliary pointers for swapping */
  PetscInt       *ihlp1;      /* Pointers to new version of arrays, */
  PetscInt       *ihlp2;      /* Pointers to previous version of arrays, */

  PetscFunctionBegin;
  PetscCall(PetscMalloc1(nrows,&ialloc));

  ihlp1 = ialloc;
  ihlp2 = isort;

  /* Sorted array chunks are first 1 long, and increase until they are the complete array */
  for (istep=1; istep<nrows; istep*=2) {
    /*
      Combine sorted parts
          istart:istart+istep-1 and istart+istep-1:istart+2*istep-1
      of ihlp2 and vhlp2

      into one sorted part
          istart:istart+2*istep-1
      of ihlp1 and vhlp1
    */
    for (istart=0; istart<nrows; istart+=2*istep) {
      /* Set counters and bound array part endings */
      i1=istart;        i1end = i1+istep;  if (i1end>nrows) i1end=nrows;
      i2=istart+istep;  i2end = i2+istep;  if (i2end>nrows) i2end=nrows;

      /* Merge the two array parts */
      for (i=istart; i<i2end; i++)  {
        if (i1<i1end && i2<i2end && spbas_row_order_icol(ihlp2[i1], ihlp2[i2], irow_in, icol_in, col_idx_type) < 0) {
          ihlp1[i] = ihlp2[i1];
          i1++;
        } else if (i2<i2end) {
          ihlp1[i] = ihlp2[i2];
          i2++;
        } else {
          ihlp1[i] = ihlp2[i1];
          i1++;
        }
      }
    }

    /* Swap the two array sets */
    iswap = ihlp2; ihlp2 = ihlp1; ihlp1 = iswap;
  }

  /* Copy one more time in case the sorted arrays are the temporary ones */
  if (ihlp2 != isort) {
    for (i=0; i<nrows; i++) isort[i] = ihlp2[i];
  }
  PetscCall(PetscFree(ialloc));
  PetscFunctionReturn(0);
}

/*
  spbas_compress_pattern:
     calculate a compressed sparseness pattern for a sparseness pattern
     given in compressed row storage. The compressed sparseness pattern may
     require (much) less memory.
*/
PetscErrorCode spbas_compress_pattern(PetscInt *irow_in, PetscInt *icol_in, PetscInt nrows, PetscInt ncols, PetscInt col_idx_type, spbas_matrix *B,PetscReal *mem_reduction)
{
  PetscInt        nnz      = irow_in[nrows];
  size_t          mem_orig = (nrows + nnz) * sizeof(PetscInt);
  size_t          mem_compressed;
  PetscInt        *isort;
  PetscInt        *icols;
  PetscInt        row_nnz;
  PetscInt        *ipoint;
  PetscBool       *used;
  PetscInt        ptr;
  PetscInt        i,j;
  const PetscBool no_values = PETSC_FALSE;

  PetscFunctionBegin;
  /* Allocate the structure of the new matrix */
  B->nrows        = nrows;
  B->ncols        = ncols;
  B->nnz          = nnz;
  B->col_idx_type = col_idx_type;
  B->block_data   = PETSC_TRUE;

  PetscCall(spbas_allocate_pattern(B, no_values));

  /* When using an offset array, set it */
  if (col_idx_type==SPBAS_OFFSET_ARRAY)  {
    for (i=0; i<nrows; i++) B->icol0[i] = icol_in[irow_in[i]];
  }

  /* Allocate the ordering for the rows */
  PetscCall(PetscMalloc1(nrows,&isort));
  PetscCall(PetscMalloc1(nrows,&ipoint));
  PetscCall(PetscCalloc1(nrows,&used));

  for (i = 0; i<nrows; i++)  {
    B->row_nnz[i] = irow_in[i+1]-irow_in[i];
    isort[i]      = i;
    ipoint[i]     = i;
  }

  /* Sort the rows so that identical columns will be next to each other */
  PetscCall(spbas_mergesort_icols(nrows, irow_in, icol_in, col_idx_type, isort));
  PetscCall(PetscInfo(NULL,"Rows have been sorted for patterns\n"));

  /* Replace identical rows with the first one in the list */
  for (i=1; i<nrows; i++) {
    if (spbas_row_order_icol(isort[i-1], isort[i], irow_in, icol_in, col_idx_type) == 0) {
      ipoint[isort[i]] = ipoint[isort[i-1]];
    }
  }

  /* Collect the rows which are used*/
  for (i=0; i<nrows; i++) used[ipoint[i]] = PETSC_TRUE;

  /* Calculate needed memory */
  B->n_alloc_icol = 0;
  for (i=0; i<nrows; i++)  {
    if (used[i]) B->n_alloc_icol += B->row_nnz[i];
  }
  PetscCall(PetscMalloc1(B->n_alloc_icol,&B->alloc_icol));

  /* Fill in the diagonal offsets for the rows which store their own data */
  ptr = 0;
  for (i=0; i<B->nrows; i++) {
    if (used[i]) {
      B->icols[i] = &B->alloc_icol[ptr];
      icols = &icol_in[irow_in[i]];
      row_nnz = B->row_nnz[i];
      if (col_idx_type == SPBAS_COLUMN_NUMBERS) {
        for (j=0; j<row_nnz; j++) {
          B->icols[i][j] = icols[j];
        }
      } else if (col_idx_type == SPBAS_DIAGONAL_OFFSETS) {
        for (j=0; j<row_nnz; j++) {
          B->icols[i][j] = icols[j]-i;
        }
      } else if (col_idx_type == SPBAS_OFFSET_ARRAY) {
        for (j=0; j<row_nnz; j++) {
          B->icols[i][j] = icols[j]-icols[0];
        }
      }
      ptr += B->row_nnz[i];
    }
  }

  /* Point to the right places for all data */
  for (i=0; i<nrows; i++) {
    B->icols[i] = B->icols[ipoint[i]];
  }
  PetscCall(PetscInfo(NULL,"Row patterns have been compressed\n"));
  PetscCall(PetscInfo(NULL,"         (%g nonzeros per row)\n", (double) ((PetscReal) nnz / (PetscReal) nrows)));

  PetscCall(PetscFree(isort));
  PetscCall(PetscFree(used));
  PetscCall(PetscFree(ipoint));

  mem_compressed = spbas_memory_requirement(*B);
  *mem_reduction = 100.0 * (PetscReal)(mem_orig-mem_compressed)/ (PetscReal) mem_orig;
  PetscFunctionReturn(0);
}

/*
   spbas_incomplete_cholesky
       Incomplete Cholesky decomposition
*/
#include <../src/mat/impls/aij/seq/bas/spbas_cholesky.h>

/*
  spbas_delete : de-allocate the arrays owned by this matrix
*/
PetscErrorCode spbas_delete(spbas_matrix matrix)
{
  PetscInt       i;

  PetscFunctionBegin;
  if (matrix.block_data) {
    PetscCall(PetscFree(matrix.alloc_icol));
    if (matrix.values) PetscCall(PetscFree(matrix.alloc_val));
  } else {
    for (i=0; i<matrix.nrows; i++) PetscCall(PetscFree(matrix.icols[i]));
    PetscCall(PetscFree(matrix.icols));
    if (matrix.values) {
      for (i=0; i<matrix.nrows; i++) PetscCall(PetscFree(matrix.values[i]));
    }
  }

  PetscCall(PetscFree(matrix.row_nnz));
  PetscCall(PetscFree(matrix.icols));
  if (matrix.col_idx_type == SPBAS_OFFSET_ARRAY) PetscCall(PetscFree(matrix.icol0));
  PetscCall(PetscFree(matrix.values));
  PetscFunctionReturn(0);
}

/*
spbas_matrix_to_crs:
   Convert an spbas_matrix to compessed row storage
*/
PetscErrorCode spbas_matrix_to_crs(spbas_matrix matrix_A,MatScalar **val_out, PetscInt **irow_out, PetscInt **icol_out)
{
  PetscInt       nrows = matrix_A.nrows;
  PetscInt       nnz   = matrix_A.nnz;
  PetscInt       i,j,r_nnz,i0;
  PetscInt       *irow;
  PetscInt       *icol;
  PetscInt       *icol_A;
  MatScalar      *val;
  PetscScalar    *val_A;
  PetscInt       col_idx_type = matrix_A.col_idx_type;
  PetscBool      do_values    = matrix_A.values ? PETSC_TRUE : PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(PetscMalloc1(nrows+1, &irow));
  PetscCall(PetscMalloc1(nnz, &icol));
  *icol_out = icol;
  *irow_out = irow;
  if (do_values) {
    PetscCall(PetscMalloc1(nnz, &val));
    *val_out = val; *icol_out = icol; *irow_out=irow;
  }

  irow[0]=0;
  for (i=0; i<nrows; i++) {
    r_nnz     = matrix_A.row_nnz[i];
    i0        = irow[i];
    irow[i+1] = i0 + r_nnz;
    icol_A    = matrix_A.icols[i];

    if (do_values) {
      val_A = matrix_A.values[i];
      for (j=0; j<r_nnz; j++) {
        icol[i0+j] = icol_A[j];
        val[i0+j]  = val_A[j];
      }
    } else {
      for (j=0; j<r_nnz; j++) icol[i0+j] = icol_A[j];
    }

    if (col_idx_type == SPBAS_DIAGONAL_OFFSETS) {
      for (j=0; j<r_nnz; j++) icol[i0+j] += i;
    } else if (col_idx_type == SPBAS_OFFSET_ARRAY) {
      i0 = matrix_A.icol0[i];
      for (j=0; j<r_nnz; j++) icol[i0+j] += i0;
    }
  }
  PetscFunctionReturn(0);
}

/*
    spbas_transpose
       return the transpose of a matrix
*/
PetscErrorCode spbas_transpose(spbas_matrix in_matrix, spbas_matrix * result)
{
  PetscInt       col_idx_type = in_matrix.col_idx_type;
  PetscInt       nnz          = in_matrix.nnz;
  PetscInt       ncols        = in_matrix.nrows;
  PetscInt       nrows        = in_matrix.ncols;
  PetscInt       i,j,k;
  PetscInt       r_nnz;
  PetscInt       *irow;
  PetscInt       icol0 = 0;
  PetscScalar    * val;

  PetscFunctionBegin;
  /* Copy input values */
  result->nrows        = nrows;
  result->ncols        = ncols;
  result->nnz          = nnz;
  result->col_idx_type = SPBAS_COLUMN_NUMBERS;
  result->block_data   = PETSC_TRUE;

  /* Allocate sparseness pattern */
  PetscCall(spbas_allocate_pattern(result, in_matrix.values ? PETSC_TRUE : PETSC_FALSE));

  /*  Count the number of nonzeros in each row */
  for (i = 0; i<nrows; i++) result->row_nnz[i] = 0;

  for (i=0; i<ncols; i++) {
    r_nnz = in_matrix.row_nnz[i];
    irow  = in_matrix.icols[i];
    if (col_idx_type == SPBAS_COLUMN_NUMBERS)  {
      for (j=0; j<r_nnz; j++) result->row_nnz[irow[j]]++;
    } else if (col_idx_type == SPBAS_DIAGONAL_OFFSETS)  {
      for (j=0; j<r_nnz; j++) result->row_nnz[i+irow[j]]++;
    } else if (col_idx_type == SPBAS_OFFSET_ARRAY) {
      icol0=in_matrix.icol0[i];
      for (j=0; j<r_nnz; j++) result->row_nnz[icol0+irow[j]]++;
    }
  }

  /* Set the pointers to the data */
  PetscCall(spbas_allocate_data(result));

  /* Reset the number of nonzeros in each row */
  for (i = 0; i<nrows; i++) result->row_nnz[i] = 0;

  /* Fill the data arrays */
  if (in_matrix.values) {
    for (i=0; i<ncols; i++) {
      r_nnz = in_matrix.row_nnz[i];
      irow  = in_matrix.icols[i];
      val   = in_matrix.values[i];

      if      (col_idx_type == SPBAS_COLUMN_NUMBERS)   icol0 = 0;
      else if (col_idx_type == SPBAS_DIAGONAL_OFFSETS) icol0 = i;
      else if (col_idx_type == SPBAS_OFFSET_ARRAY)     icol0 = in_matrix.icol0[i];
      for (j=0; j<r_nnz; j++)  {
        k = icol0 + irow[j];
        result->icols[k][result->row_nnz[k]]  = i;
        result->values[k][result->row_nnz[k]] = val[j];
        result->row_nnz[k]++;
      }
    }
  } else {
    for (i=0; i<ncols; i++) {
      r_nnz = in_matrix.row_nnz[i];
      irow  = in_matrix.icols[i];

      if      (col_idx_type == SPBAS_COLUMN_NUMBERS)   icol0=0;
      else if (col_idx_type == SPBAS_DIAGONAL_OFFSETS) icol0=i;
      else if (col_idx_type == SPBAS_OFFSET_ARRAY)     icol0=in_matrix.icol0[i];

      for (j=0; j<r_nnz; j++) {
        k = icol0 + irow[j];
        result->icols[k][result->row_nnz[k]] = i;
        result->row_nnz[k]++;
      }
    }
  }
  PetscFunctionReturn(0);
}

/*
   spbas_mergesort

      mergesort for an array of integers and an array of associated
      reals

      on output, icol[0..nnz-1] is increasing;
                  val[0..nnz-1] has undergone the same permutation as icol

      NB: val may be NULL: in that case, only the integers are sorted

*/
PetscErrorCode spbas_mergesort(PetscInt nnz, PetscInt *icol, PetscScalar *val)
{
  PetscInt       istep;       /* Chunk-sizes of already sorted parts of arrays */
  PetscInt       i, i1, i2;   /* Loop counters for (partly) sorted arrays */
  PetscInt       istart, i1end, i2end; /* start of newly sorted array part, end of both parts */
  PetscInt       *ialloc;     /* Allocated arrays */
  PetscScalar    *valloc=NULL;
  PetscInt       *iswap;      /* auxiliary pointers for swapping */
  PetscScalar    *vswap;
  PetscInt       *ihlp1;      /* Pointers to new version of arrays, */
  PetscScalar    *vhlp1=NULL;  /* (arrays under construction) */
  PetscInt       *ihlp2;      /* Pointers to previous version of arrays, */
  PetscScalar    *vhlp2=NULL;

  PetscFunctionBegin;
  PetscCall(PetscMalloc1(nnz,&ialloc));
  ihlp1 = ialloc;
  ihlp2 = icol;

  if (val) {
    PetscCall(PetscMalloc1(nnz,&valloc));
    vhlp1 = valloc;
    vhlp2 = val;
  }

  /* Sorted array chunks are first 1 long, and increase until they are the complete array */
  for (istep=1; istep<nnz; istep*=2) {
    /*
      Combine sorted parts
          istart:istart+istep-1 and istart+istep-1:istart+2*istep-1
      of ihlp2 and vhlp2

      into one sorted part
          istart:istart+2*istep-1
      of ihlp1 and vhlp1
    */
    for (istart=0; istart<nnz; istart+=2*istep) {
      /* Set counters and bound array part endings */
      i1=istart;        i1end = i1+istep;  if (i1end>nnz) i1end=nnz;
      i2=istart+istep;  i2end = i2+istep;  if (i2end>nnz) i2end=nnz;

      /* Merge the two array parts */
      if (val) {
        for (i=istart; i<i2end; i++) {
          if (i1<i1end && i2<i2end && ihlp2[i1] < ihlp2[i2]) {
            ihlp1[i] = ihlp2[i1];
            vhlp1[i] = vhlp2[i1];
            i1++;
          } else if (i2<i2end) {
            ihlp1[i] = ihlp2[i2];
            vhlp1[i] = vhlp2[i2];
            i2++;
          } else {
            ihlp1[i] = ihlp2[i1];
            vhlp1[i] = vhlp2[i1];
            i1++;
          }
        }
      } else {
        for (i=istart; i<i2end; i++) {
          if (i1<i1end && i2<i2end && ihlp2[i1] < ihlp2[i2]) {
            ihlp1[i] = ihlp2[i1];
            i1++;
          } else if (i2<i2end) {
            ihlp1[i] = ihlp2[i2];
            i2++;
          } else {
            ihlp1[i] = ihlp2[i1];
            i1++;
          }
        }
      }
    }

    /* Swap the two array sets */
    iswap = ihlp2; ihlp2 = ihlp1; ihlp1 = iswap;
    vswap = vhlp2; vhlp2 = vhlp1; vhlp1 = vswap;
  }

  /* Copy one more time in case the sorted arrays are the temporary ones */
  if (ihlp2 != icol) {
    for (i=0; i<nnz; i++) icol[i] = ihlp2[i];
    if (val) {
      for (i=0; i<nnz; i++) val[i] = vhlp2[i];
    }
  }

  PetscCall(PetscFree(ialloc));
  if (val) PetscCall(PetscFree(valloc));
  PetscFunctionReturn(0);
}

/*
  spbas_apply_reordering_rows:
    apply the given reordering to the rows:  matrix_A = matrix_A(perm,:);
*/
PetscErrorCode spbas_apply_reordering_rows(spbas_matrix *matrix_A, const PetscInt *permutation)
{
  PetscInt       i,j,ip;
  PetscInt       nrows=matrix_A->nrows;
  PetscInt       * row_nnz;
  PetscInt       **icols;
  PetscBool      do_values = matrix_A->values ? PETSC_TRUE : PETSC_FALSE;
  PetscScalar    **vals    = NULL;

  PetscFunctionBegin;
  PetscCheck(matrix_A->col_idx_type == SPBAS_DIAGONAL_OFFSETS,PETSC_COMM_SELF, PETSC_ERR_SUP_SYS,"must have diagonal offsets in pattern");

  if (do_values) {
    PetscCall(PetscMalloc1(nrows, &vals));
  }
  PetscCall(PetscMalloc1(nrows, &row_nnz));
  PetscCall(PetscMalloc1(nrows, &icols));

  for (i=0; i<nrows; i++) {
    ip = permutation[i];
    if (do_values) vals[i] = matrix_A->values[ip];
    icols[i]   = matrix_A->icols[ip];
    row_nnz[i] = matrix_A->row_nnz[ip];
    for (j=0; j<row_nnz[i]; j++) icols[i][j] += ip-i;
  }

  if (do_values) PetscCall(PetscFree(matrix_A->values));
  PetscCall(PetscFree(matrix_A->icols));
  PetscCall(PetscFree(matrix_A->row_nnz));

  if (do_values) matrix_A->values = vals;
  matrix_A->icols   = icols;
  matrix_A->row_nnz = row_nnz;
  PetscFunctionReturn(0);
}

/*
  spbas_apply_reordering_cols:
    apply the given reordering to the columns:  matrix_A(:,perm) = matrix_A;
*/
PetscErrorCode spbas_apply_reordering_cols(spbas_matrix *matrix_A,const PetscInt *permutation)
{
  PetscInt       i,j;
  PetscInt       nrows=matrix_A->nrows;
  PetscInt       row_nnz;
  PetscInt       *icols;
  PetscBool      do_values = matrix_A->values ? PETSC_TRUE : PETSC_FALSE;
  PetscScalar    *vals     = NULL;

  PetscFunctionBegin;
  PetscCheck(matrix_A->col_idx_type == SPBAS_DIAGONAL_OFFSETS,PETSC_COMM_SELF, PETSC_ERR_SUP_SYS, "must have diagonal offsets in pattern");

  for (i=0; i<nrows; i++) {
    icols   = matrix_A->icols[i];
    row_nnz = matrix_A->row_nnz[i];
    if (do_values) vals = matrix_A->values[i];

    for (j=0; j<row_nnz; j++) {
      icols[j] = permutation[i+icols[j]]-i;
    }
    PetscCall(spbas_mergesort(row_nnz, icols, vals));
  }
  PetscFunctionReturn(0);
}

/*
  spbas_apply_reordering:
    apply the given reordering:  matrix_A(perm,perm) = matrix_A;
*/
PetscErrorCode spbas_apply_reordering(spbas_matrix *matrix_A, const PetscInt *permutation, const PetscInt * inv_perm)
{
  PetscFunctionBegin;
  PetscCall(spbas_apply_reordering_rows(matrix_A, inv_perm));
  PetscCall(spbas_apply_reordering_cols(matrix_A, permutation));
  PetscFunctionReturn(0);
}

PetscErrorCode spbas_pattern_only(PetscInt nrows, PetscInt ncols, PetscInt *ai, PetscInt *aj, spbas_matrix * result)
{
  spbas_matrix   retval;
  PetscInt       i, j, i0, r_nnz;

  PetscFunctionBegin;
  /* Copy input values */
  retval.nrows = nrows;
  retval.ncols = ncols;
  retval.nnz   = ai[nrows];

  retval.block_data   = PETSC_TRUE;
  retval.col_idx_type = SPBAS_DIAGONAL_OFFSETS;

  /* Allocate output matrix */
  PetscCall(spbas_allocate_pattern(&retval, PETSC_FALSE));
  for (i=0; i<nrows; i++) retval.row_nnz[i] = ai[i+1]-ai[i];
  PetscCall(spbas_allocate_data(&retval));
  /* Copy the structure */
  for (i = 0; i<retval.nrows; i++)  {
    i0    = ai[i];
    r_nnz = ai[i+1]-i0;

    for (j=0; j<r_nnz; j++) {
      retval.icols[i][j] = aj[i0+j]-i;
    }
  }
  *result = retval;
  PetscFunctionReturn(0);
}

/*
   spbas_mark_row_power:
      Mark the columns in row 'row' which are nonzero in
          matrix^2log(marker).
*/
PetscErrorCode spbas_mark_row_power(PetscInt *iwork,             /* marker-vector */
                                    PetscInt row,                /* row for which the columns are marked */
                                    spbas_matrix * in_matrix,    /* matrix for which the power is being  calculated */
                                    PetscInt marker,             /* marker-value: 2^power */
                                    PetscInt minmrk,             /* lower bound for marked points */
                                    PetscInt maxmrk)             /* upper bound for marked points */
{
  PetscInt       i,j, nnz;

  PetscFunctionBegin;
  nnz = in_matrix->row_nnz[row];

  /* For higher powers, call this function recursively */
  if (marker>1) {
    for (i=0; i<nnz; i++) {
      j = row + in_matrix->icols[row][i];
      if (minmrk<=j && j<maxmrk && iwork[j] < marker) {
        PetscCall(spbas_mark_row_power(iwork, row + in_matrix->icols[row][i],in_matrix, marker/2,minmrk,maxmrk));
        iwork[j] |= marker;
      }
    }
  } else {
    /*  Mark the columns reached */
    for (i=0; i<nnz; i++)  {
      j = row + in_matrix->icols[row][i];
      if (minmrk<=j && j<maxmrk) iwork[j] |= 1;
    }
  }
  PetscFunctionReturn(0);
}

/*
   spbas_power
      Calculate sparseness patterns for incomplete Cholesky decompositions
      of a given order: (almost) all nonzeros of the matrix^(order+1) which
      are inside the band width are found and stored in the output sparseness
      pattern.
*/
PetscErrorCode spbas_power(spbas_matrix in_matrix,PetscInt power, spbas_matrix * result)
{
  spbas_matrix   retval;
  PetscInt       nrows = in_matrix.nrows;
  PetscInt       ncols = in_matrix.ncols;
  PetscInt       i, j, kend;
  PetscInt       nnz, inz;
  PetscInt       *iwork;
  PetscInt       marker;
  PetscInt       maxmrk=0;

  PetscFunctionBegin;
  PetscCheck(in_matrix.col_idx_type == SPBAS_DIAGONAL_OFFSETS,PETSC_COMM_SELF, PETSC_ERR_SUP_SYS,"must have diagonal offsets in pattern");
  PetscCheck(ncols == nrows,PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Dimension error");
  PetscCheckFalse(in_matrix.values,PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Input array must be sparseness pattern (no values)");
  PetscCheck(power>0,PETSC_COMM_SELF, PETSC_ERR_SUP_SYS, "Power must be 1 or up");

  /* Copy input values*/
  retval.nrows        = ncols;
  retval.ncols        = nrows;
  retval.nnz          = 0;
  retval.col_idx_type = SPBAS_DIAGONAL_OFFSETS;
  retval.block_data   = PETSC_FALSE;

  /* Allocate sparseness pattern */
  PetscCall(spbas_allocate_pattern(&retval, in_matrix.values ? PETSC_TRUE : PETSC_FALSE));

  /* Allocate marker array: note sure the max needed so use the max of the two */
  PetscCall(PetscCalloc1(PetscMax(ncols,nrows), &iwork));

  /* Calculate marker values */
  marker = 1; for (i=1; i<power; i++) marker*=2;

  for (i=0; i<nrows; i++)  {
    /* Calculate the pattern for each row */

    nnz  = in_matrix.row_nnz[i];
    kend = i+in_matrix.icols[i][nnz-1];
    if (maxmrk<=kend) maxmrk=kend+1;
    PetscCall(spbas_mark_row_power(iwork, i, &in_matrix, marker, i, maxmrk));

    /* Count the columns*/
    nnz = 0;
    for (j=i; j<maxmrk; j++) nnz+= (iwork[j]!=0);

    /* Allocate the column indices */
    retval.row_nnz[i] = nnz;
    PetscCall(PetscMalloc1(nnz,&retval.icols[i]));

    /* Administrate the column indices */
    inz = 0;
    for (j=i; j<maxmrk; j++) {
      if (iwork[j]) {
        retval.icols[i][inz] = j-i;
        inz++;
        iwork[j] = 0;
      }
    }
    retval.nnz += nnz;
  };
  PetscCall(PetscFree(iwork));
  *result = retval;
  PetscFunctionReturn(0);
}

/*
   spbas_keep_upper:
      remove the lower part of the matrix: keep the upper part
*/
PetscErrorCode spbas_keep_upper(spbas_matrix * inout_matrix)
{
  PetscInt i, j;
  PetscInt jstart;

  PetscFunctionBegin;
  PetscCheck(!inout_matrix->block_data,PETSC_COMM_SELF, PETSC_ERR_SUP_SYS, "Not yet for block data matrices");
  for (i=0; i<inout_matrix->nrows; i++)  {
    for (jstart=0; (jstart<inout_matrix->row_nnz[i]) && (inout_matrix->icols[i][jstart]<0); jstart++) {}
    if (jstart>0) {
      for (j=0; j<inout_matrix->row_nnz[i]-jstart; j++) {
        inout_matrix->icols[i][j] = inout_matrix->icols[i][j+jstart];
      }

      if (inout_matrix->values) {
        for (j=0; j<inout_matrix->row_nnz[i]-jstart; j++) {
          inout_matrix->values[i][j] = inout_matrix->values[i][j+jstart];
        }
      }

      inout_matrix->row_nnz[i] -= jstart;

      inout_matrix->icols[i] = (PetscInt*) realloc((void*) inout_matrix->icols[i], inout_matrix->row_nnz[i]*sizeof(PetscInt));

      if (inout_matrix->values) {
        inout_matrix->values[i] = (PetscScalar*) realloc((void*) inout_matrix->values[i], inout_matrix->row_nnz[i]*sizeof(PetscScalar));
      }
      inout_matrix->nnz -= jstart;
    }
  }
  PetscFunctionReturn(0);
}
