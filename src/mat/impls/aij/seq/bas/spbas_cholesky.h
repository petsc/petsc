
/*
   spbas_cholesky_row_alloc:
      in the data arrays, find a place where another row may be stored.
      Return PETSC_ERR_MEM if there is insufficient space to store the
      row, so garbage collection and/or re-allocation may be done.
*/
PetscErrorCode spbas_cholesky_row_alloc(spbas_matrix retval, PetscInt k, PetscInt r_nnz,PetscInt * n_alloc_used)
{
  PetscFunctionBegin;
  retval.icols[k]  = &retval.alloc_icol[*n_alloc_used];
  retval.values[k] = &retval.alloc_val[*n_alloc_used];
  *n_alloc_used   += r_nnz;
  if (*n_alloc_used > retval.n_alloc_icol) PetscFunctionReturn(PETSC_ERR_MEM);
  PetscFunctionReturn(0);
}

/*
  spbas_cholesky_garbage_collect:
     move the rows which have been calculated so far, as well as
     those currently under construction, to the front of the
     array, while putting them in the proper order.
     When it seems necessary, enlarge the current arrays.

     NB: re-allocation is being simulated using
         PetscMalloc, memcpy, PetscFree, because
         PetscRealloc does not seem to exist.

*/
PetscErrorCode spbas_cholesky_garbage_collect(spbas_matrix *result,         /* I/O: the Cholesky factor matrix being constructed.
                                                                                    Only the storage, not the contents of this matrix is changed in this function */
                                              PetscInt     i_row,           /* I  : Number of rows for which the final contents are known */
                                              PetscInt     *n_row_alloc_ok, /* I/O: Number of rows which are already in their final
                                                                                    places in the arrays: they need not be moved any more */
                                              PetscInt     *n_alloc_used,   /* I/O:  */
                                              PetscInt     *max_row_nnz)    /* I  : Over-estimate of the number of nonzeros needed to store each row */
{
/* PSEUDO-CODE:
  1. Choose the appropriate size for the arrays
  2. Rescue the arrays which would be lost during garbage collection
  3. Reallocate and correct administration
  4. Move all arrays so that they are in proper order */

  PetscInt        i,j;
  PetscInt        nrows         = result->nrows;
  PetscInt        n_alloc_ok    =0;
  PetscInt        n_alloc_ok_max=0;
  PetscErrorCode  ierr;
  PetscInt        need_already  = 0;
  PetscInt        n_rows_ahead  =0;
  PetscInt        max_need_extra= 0;
  PetscInt        n_alloc_max, n_alloc_est, n_alloc;
  PetscInt        n_alloc_now     = result->n_alloc_icol;
  PetscInt        *alloc_icol_old = result->alloc_icol;
  PetscScalar     *alloc_val_old  = result->alloc_val;
  PetscInt        *icol_rescue;
  PetscScalar     *val_rescue;
  PetscInt        n_rescue;
  PetscInt        n_row_rescue;
  PetscInt        i_here, i_last, n_copy;
  const PetscReal xtra_perc = 20;

  PetscFunctionBegin;
  /*********************************************************
  1. Choose appropriate array size
  Count number of rows and memory usage which is already final */
  for (i=0; i<i_row; i++)  {
    n_alloc_ok     += result->row_nnz[i];
    n_alloc_ok_max += max_row_nnz[i];
  }

  /* Count currently needed memory usage and future memory requirements
    (max, predicted)*/
  for (i=i_row; i<nrows; i++) {
    if (!result->row_nnz[i]) {
      max_need_extra += max_row_nnz[i];
    } else {
      need_already += max_row_nnz[i];
      n_rows_ahead++;
    }
  }

  /* Make maximal and realistic memory requirement estimates */
  n_alloc_max = n_alloc_ok + need_already + max_need_extra;
  n_alloc_est = n_alloc_ok + need_already + (int) (((PetscReal) max_need_extra) *  ((PetscReal) n_alloc_ok) /((PetscReal) n_alloc_ok_max));

  /* Choose array sizes */
  if (n_alloc_max == n_alloc_est) n_alloc = n_alloc_max;
  else if (n_alloc_now >= n_alloc_est) n_alloc = n_alloc_now;
  else if (n_alloc_max < n_alloc_est * (1+xtra_perc/100.0)) n_alloc = n_alloc_max;
  else n_alloc = (int) (n_alloc_est * (1+xtra_perc/100.0));

  /* If new estimate is less than what we already have,
    don't reallocate, just garbage-collect */
  if (n_alloc_max != n_alloc_est && n_alloc < result->n_alloc_icol) {
    n_alloc = result->n_alloc_icol;
  }

  /* Motivate dimension choice */
  ierr = PetscInfo1(NULL,"   Allocating %d nonzeros: ",n_alloc);CHKERRQ(ierr);
  if (n_alloc_max == n_alloc_est) {
    ierr = PetscInfo(NULL,"this is the correct size\n");CHKERRQ(ierr);
  } else if (n_alloc_now >= n_alloc_est) {
    ierr = PetscInfo(NULL,"the current size, which seems enough\n");CHKERRQ(ierr);
  } else if (n_alloc_max < n_alloc_est * (1+xtra_perc/100.0)) {
    ierr = PetscInfo(NULL,"the maximum estimate\n");CHKERRQ(ierr);
  } else {
    ierr = PetscInfo1(NULL,"%6.2f %% more than the estimate\n",xtra_perc);CHKERRQ(ierr);
  }

  /**********************************************************
  2. Rescue arrays which would be lost
  Count how many rows and nonzeros will have to be rescued
  when moving all arrays in place */
  n_row_rescue = 0; n_rescue = 0;
  if (*n_row_alloc_ok==0) *n_alloc_used = 0;
  else {
    i = *n_row_alloc_ok - 1;

    *n_alloc_used = (result->icols[i]-result->alloc_icol) +  result->row_nnz[i];
  }

  for (i=*n_row_alloc_ok; i<nrows; i++) {
    i_here = result->icols[i]-result->alloc_icol;
    i_last = i_here + result->row_nnz[i];
    if (result->row_nnz[i]>0) {
      if (*n_alloc_used > i_here || i_last > n_alloc) {
        n_rescue += result->row_nnz[i];
        n_row_rescue++;
      }

      if (i<i_row) *n_alloc_used += result->row_nnz[i];
      else         *n_alloc_used += max_row_nnz[i];
    }
  }

  /* Allocate rescue arrays */
  ierr = PetscMalloc1(n_rescue, &icol_rescue);CHKERRQ(ierr);
  ierr = PetscMalloc1(n_rescue, &val_rescue);CHKERRQ(ierr);

  /* Rescue the arrays which need rescuing */
  n_row_rescue = 0; n_rescue = 0;
  if (*n_row_alloc_ok==0) *n_alloc_used = 0;
  else {
    i             = *n_row_alloc_ok - 1;
    *n_alloc_used = (result->icols[i]-result->alloc_icol) +  result->row_nnz[i];
  }

  for (i=*n_row_alloc_ok; i<nrows; i++) {
    i_here = result->icols[i]-result->alloc_icol;
    i_last = i_here + result->row_nnz[i];
    if (result->row_nnz[i]>0) {
      if (*n_alloc_used > i_here || i_last > n_alloc) {
        ierr = PetscArraycpy(&icol_rescue[n_rescue], result->icols[i], result->row_nnz[i]);CHKERRQ(ierr);
        ierr = PetscArraycpy(&val_rescue[n_rescue], result->values[i], result->row_nnz[i]);CHKERRQ(ierr);
        n_rescue += result->row_nnz[i];
        n_row_rescue++;
      }

      if (i<i_row) *n_alloc_used += result->row_nnz[i];
      else         *n_alloc_used += max_row_nnz[i];
    }
  }

  /**********************************************************
  3. Reallocate and correct administration */

  if (n_alloc != result->n_alloc_icol) {
    n_copy = PetscMin(n_alloc,result->n_alloc_icol);

    /* PETSC knows no REALLOC, so we'll REALLOC ourselves.

        Allocate new icol-data, copy old contents */
    ierr = PetscMalloc1(n_alloc, &result->alloc_icol);CHKERRQ(ierr);
    ierr = PetscArraycpy(result->alloc_icol, alloc_icol_old, n_copy);CHKERRQ(ierr);

    /* Update administration, Reset pointers to new arrays  */
    result->n_alloc_icol = n_alloc;
    for (i=0; i<nrows; i++) {
      result->icols[i]  =  result->alloc_icol + (result->icols[i]  - alloc_icol_old);
      result->values[i] =  result->alloc_val  + (result->icols[i]  - result->alloc_icol);
    }

    /* Delete old array */
    ierr = PetscFree(alloc_icol_old);CHKERRQ(ierr);

    /* Allocate new value-data, copy old contents */
    ierr = PetscMalloc1(n_alloc, &result->alloc_val);CHKERRQ(ierr);
    ierr = PetscArraycpy(result->alloc_val, alloc_val_old, n_copy);CHKERRQ(ierr);

    /* Update administration, Reset pointers to new arrays  */
    result->n_alloc_val = n_alloc;
    for (i=0; i<nrows; i++) {
      result->values[i] =  result->alloc_val + (result->icols[i]  - result->alloc_icol);
    }

    /* Delete old array */
    ierr = PetscFree(alloc_val_old);CHKERRQ(ierr);
  }

  /*********************************************************
  4. Copy all the arrays to their proper places */
  n_row_rescue = 0; n_rescue = 0;
  if (*n_row_alloc_ok==0) *n_alloc_used = 0;
  else {
    i = *n_row_alloc_ok - 1;

    *n_alloc_used = (result->icols[i]-result->alloc_icol) +  result->row_nnz[i];
  }

  for (i=*n_row_alloc_ok; i<nrows; i++) {
    i_here = result->icols[i]-result->alloc_icol;
    i_last = i_here + result->row_nnz[i];

    result->icols[i] = result->alloc_icol + *n_alloc_used;
    result->values[i]= result->alloc_val  + *n_alloc_used;

    if (result->row_nnz[i]>0) {
      if (*n_alloc_used > i_here || i_last > n_alloc) {
        ierr = PetscArraycpy(result->icols[i], &icol_rescue[n_rescue], result->row_nnz[i]);CHKERRQ(ierr);
        ierr = PetscArraycpy(result->values[i],&val_rescue[n_rescue],result->row_nnz[i]);CHKERRQ(ierr);

        n_rescue += result->row_nnz[i];
        n_row_rescue++;
      } else {
        for (j=0; j<result->row_nnz[i]; j++) {
          result->icols[i][j]  = result->alloc_icol[i_here+j];
          result->values[i][j] = result->alloc_val[i_here+j];
        }
      }
      if (i<i_row) *n_alloc_used += result->row_nnz[i];
      else         *n_alloc_used += max_row_nnz[i];
    }
  }

  /* Delete the rescue arrays */
  ierr = PetscFree(icol_rescue);CHKERRQ(ierr);
  ierr = PetscFree(val_rescue);CHKERRQ(ierr);

  *n_row_alloc_ok = i_row;
  PetscFunctionReturn(0);
}

/*
  spbas_incomplete_cholesky:
     incomplete Cholesky decomposition of a square, symmetric,
     positive definite matrix.

     In case negative diagonals are encountered, function returns
     NEGATIVE_DIAGONAL. When this happens, call this function again
     with a larger epsdiag_in, a less sparse pattern, and/or a smaller
     droptol
*/
PetscErrorCode spbas_incomplete_cholesky(Mat A, const PetscInt *rip, const PetscInt *riip, spbas_matrix pattern, PetscReal droptol, PetscReal epsdiag_in, spbas_matrix * matrix_L)
{
  PetscInt        jL;
  Mat_SeqAIJ      *a =(Mat_SeqAIJ*)A->data;
  PetscInt        *ai=a->i,*aj=a->j;
  MatScalar       *aa=a->a;
  PetscInt        nrows, ncols;
  PetscInt        *max_row_nnz;
  PetscErrorCode  ierr;
  spbas_matrix    retval;
  PetscScalar     *diag;
  PetscScalar     *val;
  PetscScalar     *lvec;
  PetscScalar     epsdiag;
  PetscInt        i,j,k;
  const PetscBool do_values = PETSC_TRUE;
  PetscInt        *r1_icol;
  PetscScalar     *r1_val;
  PetscInt        *r_icol;
  PetscInt        r_nnz;
  PetscScalar     *r_val;
  PetscInt        *A_icol;
  PetscInt        A_nnz;
  PetscScalar     *A_val;
  PetscInt        *p_icol;
  PetscInt        p_nnz;
  PetscInt        n_row_alloc_ok = 0;   /* number of rows which have been stored   correctly in the matrix */
  PetscInt        n_alloc_used   = 0;   /* part of result->icols and result->values   which is currently being used */

  PetscFunctionBegin;
  /* Convert the Manteuffel shift from 'fraction of average diagonal' to   dimensioned value */
  ierr = MatGetSize(A, &nrows, &ncols);CHKERRQ(ierr);
  ierr = MatGetTrace(A, &epsdiag);CHKERRQ(ierr);

  epsdiag *= epsdiag_in / nrows;

  ierr = PetscInfo2(NULL,"   Dimensioned Manteuffel shift %g Drop tolerance %g\n", (double)PetscRealPart(epsdiag),(double)droptol);CHKERRQ(ierr);

  if (droptol<1e-10) droptol=1e-10;

  if ((nrows != pattern.nrows) || (ncols != pattern.ncols) || (ncols != nrows)) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP,"Dimension error in spbas_incomplete_cholesky\n");

  retval.nrows        = nrows;
  retval.ncols        = nrows;
  retval.nnz          = pattern.nnz/10;
  retval.col_idx_type = SPBAS_COLUMN_NUMBERS;
  retval.block_data   = PETSC_TRUE;

  ierr       = spbas_allocate_pattern(&retval, do_values);CHKERRQ(ierr);
  ierr       = PetscArrayzero(retval.row_nnz, nrows);CHKERRQ(ierr);
  ierr       = spbas_allocate_data(&retval);CHKERRQ(ierr);
  retval.nnz = 0;

  ierr = PetscMalloc1(nrows, &diag);CHKERRQ(ierr);
  ierr = PetscCalloc1(nrows, &val);CHKERRQ(ierr);
  ierr = PetscCalloc1(nrows, &lvec);CHKERRQ(ierr);
  ierr = PetscCalloc1(nrows, &max_row_nnz);CHKERRQ(ierr);

  /* Check correct format of sparseness pattern */
  if (pattern.col_idx_type != SPBAS_DIAGONAL_OFFSETS) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP_SYS, "Error in spbas_incomplete_cholesky: must have diagonal offsets in pattern\n");

  /* Count the nonzeros on transpose of pattern */
  for (i = 0; i<nrows; i++)  {
    p_nnz  = pattern.row_nnz[i];
    p_icol = pattern.icols[i];
    for (j=0; j<p_nnz; j++)  {
      max_row_nnz[i+p_icol[j]]++;
    }
  }

  /* Calculate rows of L */
  for (i = 0; i<nrows; i++)  {
    p_nnz  = pattern.row_nnz[i];
    p_icol = pattern.icols[i];

    r_nnz  = retval.row_nnz[i];
    r_icol = retval.icols[i];
    r_val  = retval.values[i];
    A_nnz  = ai[rip[i]+1] - ai[rip[i]];
    A_icol = &aj[ai[rip[i]]];
    A_val  = &aa[ai[rip[i]]];

    /* Calculate  val += A(i,i:n)'; */
    for (j=0; j<A_nnz; j++) {
      k = riip[A_icol[j]];
      if (k>=i) val[k] = A_val[j];
    }

    /*  Add regularization */
    val[i] += epsdiag;

    /* Calculate lvec   = diag(D(0:i-1)) * L(0:i-1,i);
        val(i) = A(i,i) - L(0:i-1,i)' * lvec */
    for (j=0; j<r_nnz; j++)  {
      k       = r_icol[j];
      lvec[k] = diag[k] * r_val[j];
      val[i] -= r_val[j] * lvec[k];
    }

    /* Calculate the new diagonal */
    diag[i] = val[i];
    if (PetscRealPart(diag[i])<droptol) {
      ierr = PetscInfo(NULL,"Error in spbas_incomplete_cholesky:\n");CHKERRQ(ierr);
      ierr = PetscInfo1(NULL,"Negative diagonal in row %d\n",i+1);CHKERRQ(ierr);

      /* Delete the whole matrix at once. */
      ierr = spbas_delete(retval);CHKERRQ(ierr);
      return NEGATIVE_DIAGONAL;
    }

    /* If necessary, allocate arrays */
    if (r_nnz==0) {
      ierr = spbas_cholesky_row_alloc(retval, i, 1, &n_alloc_used);
      if (ierr == PETSC_ERR_MEM) {
        ierr = spbas_cholesky_garbage_collect(&retval,  i, &n_row_alloc_ok, &n_alloc_used, max_row_nnz);CHKERRQ(ierr);
        ierr = spbas_cholesky_row_alloc(retval, i, 1, &n_alloc_used);CHKERRQ(ierr);
      } else CHKERRQ(ierr);
      r_icol = retval.icols[i];
      r_val  = retval.values[i];
    }

    /* Now, fill in */
    r_icol[r_nnz] = i;
    r_val [r_nnz] = 1.0;
    r_nnz++;
    retval.row_nnz[i]++;

    retval.nnz += r_nnz;

    /* Calculate
        val(i+1:n) = (A(i,i+1:n)- L(0:i-1,i+1:n)' * lvec)/diag(i) */
    for (j=1; j<p_nnz; j++)  {
      k       = i+p_icol[j];
      r1_icol = retval.icols[k];
      r1_val  = retval.values[k];
      for (jL=0; jL<retval.row_nnz[k]; jL++) {
        val[k] -= r1_val[jL] * lvec[r1_icol[jL]];
      }
      val[k] /= diag[i];

      if (PetscAbsScalar(val[k]) > droptol || PetscAbsScalar(val[k])< -droptol) {
        /* If necessary, allocate arrays */
        if (retval.row_nnz[k]==0) {
          ierr = spbas_cholesky_row_alloc(retval, k, max_row_nnz[k], &n_alloc_used);
          if (ierr == PETSC_ERR_MEM) {
            ierr   = spbas_cholesky_garbage_collect(&retval,  i, &n_row_alloc_ok, &n_alloc_used, max_row_nnz);CHKERRQ(ierr);
            ierr   = spbas_cholesky_row_alloc(retval, k, max_row_nnz[k], &n_alloc_used);CHKERRQ(ierr);
            r_icol = retval.icols[i];
          } else CHKERRQ(ierr);
        }

        retval.icols[k][retval.row_nnz[k]]  = i;
        retval.values[k][retval.row_nnz[k]] = val[k];
        retval.row_nnz[k]++;
      }
      val[k] = 0;
    }

    /* Erase the values used in the work arrays */
    for (j=0; j<r_nnz; j++) lvec[r_icol[j]] = 0;
  }

  ierr=PetscFree(lvec);CHKERRQ(ierr);
  ierr=PetscFree(val);CHKERRQ(ierr);

  ierr = spbas_cholesky_garbage_collect(&retval, nrows, &n_row_alloc_ok, &n_alloc_used, max_row_nnz);CHKERRQ(ierr);
  ierr = PetscFree(max_row_nnz);CHKERRQ(ierr);

  /* Place the inverse of the diagonals in the matrix */
  for (i=0; i<nrows; i++) {
    r_nnz = retval.row_nnz[i];

    retval.values[i][r_nnz-1] = 1.0 / diag[i];
    for (j=0; j<r_nnz-1; j++) {
      retval.values[i][j] *= -1;
    }
  }
  ierr      = PetscFree(diag);CHKERRQ(ierr);
  *matrix_L = retval;
  PetscFunctionReturn(0);
}
