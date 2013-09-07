
#include <../src/mat/impls/aij/seq/aij.h>

#undef __FUNCT__
#define __FUNCT__ "MatFDColoringCreate_SeqAIJ_den2sp"
/*
    This routine optimizes MatFDColoringCreate_SeqAIJ() by using den2sp array
*/
PetscErrorCode MatFDColoringCreate_SeqAIJ_den2sp(Mat mat,ISColoring iscoloring,MatFDColoring c)
{
  PetscErrorCode ierr;
  PetscInt       i,n,nrows,N,j,k,m,ncols,col;
  const PetscInt *is,*rows,*ci,*cj;
  PetscInt       nis=iscoloring->n,*rowhit,*columnsforrow,bs=1,*spidx,*spidxhit,*den2sp,*den2sp_i;
  IS             *isa;
  PetscBool      flg1;     
  Mat_SeqAIJ     *csp = (Mat_SeqAIJ*)mat->data;
  PetscInt       nz_den=0;

  PetscFunctionBegin;
  ierr = ISColoringGetIS(iscoloring,PETSC_IGNORE,&isa);CHKERRQ(ierr);
  /* this is ugly way to get blocksize but cannot call MatGetBlockSize() because AIJ can have bs > 1.
     SeqBAIJ calls this routine, thus check it */
  ierr = PetscObjectTypeCompare((PetscObject)mat,MATSEQBAIJ,&flg1);CHKERRQ(ierr);
  if (flg1) {
    ierr = MatGetBlockSize(mat,&bs);CHKERRQ(ierr);
  }

  N         = mat->cmap->N/bs;
  c->M      = mat->rmap->N/bs;   /* set total rows, columns and local rows */
  c->N      = mat->cmap->N/bs;
  c->m      = mat->rmap->N/bs;
  c->rstart = 0;

  c->ncolors = nis;
  ierr       = PetscMalloc(nis*sizeof(PetscInt),&c->ncolumns);CHKERRQ(ierr);
  ierr       = PetscMalloc(nis*sizeof(PetscInt*),&c->columns);CHKERRQ(ierr);
  ierr       = PetscMalloc(nis*sizeof(PetscInt),&c->nrows);CHKERRQ(ierr);
  ierr       = PetscMalloc(nis*sizeof(PetscInt*),&c->rows);CHKERRQ(ierr);
  ierr       = PetscMalloc(nis*sizeof(PetscInt*),&c->columnsforrow);CHKERRQ(ierr);
  ierr       = PetscMalloc((csp->nz+1)*sizeof(PetscInt),&den2sp);CHKERRQ(ierr);
  den2sp_i   = den2sp;
  ierr       = PetscMalloc2(csp->nz,PetscInt,&c->colorforrow,csp->nz,PetscInt,&c->colorforcolumn);CHKERRQ(ierr);
  ierr       = PetscMalloc(3*csp->nz*sizeof(PetscInt*),&c->rowcolden2sp3);CHKERRQ(ierr);

  ierr = MatGetColumnIJ_SeqAIJ_Color(mat,0,PETSC_FALSE,PETSC_FALSE,&ncols,&ci,&cj,&spidx,NULL);CHKERRQ(ierr);
 
  ierr = PetscMalloc3(c->m,PetscInt,&rowhit,N,PetscInt,&columnsforrow,c->m,PetscInt,&spidxhit);CHKERRQ(ierr);
  ierr = PetscMemzero(rowhit,c->m*sizeof(PetscInt));CHKERRQ(ierr);

  for (i=0; i<nis; i++) { /* loop over colors */
    ierr = ISGetLocalSize(isa[i],&n);CHKERRQ(ierr);
    ierr = ISGetIndices(isa[i],&is);CHKERRQ(ierr);

    c->ncolumns[i] = n;
    if (n) {
      ierr = PetscMalloc(n*sizeof(PetscInt),&c->columns[i]);CHKERRQ(ierr);
      ierr = PetscMemcpy(c->columns[i],is,n*sizeof(PetscInt));CHKERRQ(ierr);
    } else {
      c->columns[i] = 0;
    }

    /* fast, crude version requires O(N*N) work */
    nrows = 0;
    for (j=0; j<n; j++) {  /* loop over columns */
      col  = is[j];
      rows = cj + ci[col];
      m    = ci[col+1] - ci[col];  
      for (k=0; k<m; k++) {  /* loop over columns marking them in rowhit */
        rowhit[*rows]     = col + 1;
        spidxhit[*rows++] = spidx[ci[col] + k];
        nrows++;
      }
    }
    c->nrows[i] = nrows; /* total num of rows for this color */
   
    ierr = PetscMalloc((nrows+1)*sizeof(PetscInt),&c->rows[i]);CHKERRQ(ierr);
    ierr = PetscMalloc((nrows+1)*sizeof(PetscInt),&c->columnsforrow[i]);CHKERRQ(ierr);
    nrows = 0;
    for (j=0; j<N; j++) { /* loop over rows */
      if (rowhit[j]) {
        c->rows[i][nrows]          = j;             /* row index */
        c->columnsforrow[i][nrows] = rowhit[j] - 1; /* column index */
        den2sp_i[nrows++]          = spidxhit[j];

        c->colorforrow[nz_den]    = j;             /* row index */
        c->colorforcolumn[nz_den] = rowhit[j] - 1; /* column index */

        c->rowcolden2sp3[3*nz_den]   = j;             /* row index */
        c->rowcolden2sp3[3*nz_den+1] = rowhit[j] - 1; /* column index */
        c->rowcolden2sp3[3*nz_den+2] = spidxhit[j];

        nz_den++;
        rowhit[j] = 0.0; /* zero rowhit for reuse */
      }
    } 
    den2sp_i += nrows;
    ierr = ISRestoreIndices(isa[i],&is);CHKERRQ(ierr);
  }
  printf("nz_den %d, csp->nz %d\n",nz_den,csp->nz);

  ierr = MatRestoreColumnIJ_SeqAIJ_Color(mat,0,PETSC_FALSE,PETSC_FALSE,&ncols,&ci,&cj,&spidx,NULL);CHKERRQ(ierr);
  ierr = PetscFree3(rowhit,columnsforrow,spidxhit);CHKERRQ(ierr);
  ierr = ISColoringRestoreIS(iscoloring,&isa);CHKERRQ(ierr);

  c->den2sp                 = den2sp;
  c->ctype                  = IS_COLORING_GHOSTED;
  mat->ops->fdcoloringapply = MatFDColoringApply_SeqAIJ;
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------- */

#undef __FUNCT__
#define __FUNCT__ "MatFDColoringCreate_SeqAIJ"
/*
    This routine is shared by AIJ and BAIJ matrices, since it operators only on the nonzero structure of the elements or blocks.
    This is why it has the ugly code with the MatGetBlockSize()
*/
PetscErrorCode MatFDColoringCreate_SeqAIJ(Mat mat,ISColoring iscoloring,MatFDColoring c)
{
  PetscErrorCode ierr;
  PetscInt       i,n,nrows,N,j,k,m,ncols,col;
  const PetscInt *is,*rows,*ci,*cj;
  PetscInt       nis = iscoloring->n,*rowhit,*columnsforrow,l,bs = 1;
  IS             *isa;
  PetscBool      done,flg = PETSC_FALSE;
  PetscBool      flg1;
  PetscBool      new_impl=PETSC_FALSE;

  PetscFunctionBegin;
  ierr = PetscOptionsName("-new","using new impls","",&new_impl);CHKERRQ(ierr);
  if (new_impl){
    ierr =  MatFDColoringCreate_SeqAIJ_den2sp(mat,iscoloring,c);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (!mat->assembled) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Matrix must be assembled by calls to MatAssemblyBegin/End();");

  ierr = ISColoringGetIS(iscoloring,PETSC_IGNORE,&isa);CHKERRQ(ierr);
  /* this is ugly way to get blocksize but cannot call MatGetBlockSize() because AIJ can have bs > 1.
     SeqBAIJ calls this routine, thus check it */
  ierr = PetscObjectTypeCompare((PetscObject)mat,MATSEQBAIJ,&flg1);CHKERRQ(ierr);
  if (flg1) {
    ierr = MatGetBlockSize(mat,&bs);CHKERRQ(ierr);
  }

  N         = mat->cmap->N/bs;
  c->M      = mat->rmap->N/bs;   /* set total rows, columns and local rows */
  c->N      = mat->cmap->N/bs;
  c->m      = mat->rmap->N/bs;
  c->rstart = 0;

  c->ncolors = nis;
  ierr       = PetscMalloc(nis*sizeof(PetscInt),&c->ncolumns);CHKERRQ(ierr);
  ierr       = PetscMalloc(nis*sizeof(PetscInt*),&c->columns);CHKERRQ(ierr);
  ierr       = PetscMalloc(nis*sizeof(PetscInt),&c->nrows);CHKERRQ(ierr);
  ierr       = PetscMalloc(nis*sizeof(PetscInt*),&c->rows);CHKERRQ(ierr);
  ierr       = PetscMalloc(nis*sizeof(PetscInt*),&c->columnsforrow);CHKERRQ(ierr);

  ierr = MatGetColumnIJ(mat,0,PETSC_FALSE,PETSC_FALSE,&ncols,&ci,&cj,&done);CHKERRQ(ierr);
  if (!done) SETERRQ1(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"MatGetColumnIJ() not supported for matrix type %s",((PetscObject)mat)->type_name);

  /*
     Temporary option to allow for debugging/testing
  */
  ierr = PetscOptionsGetBool(NULL,"-matfdcoloring_slow",&flg,NULL);CHKERRQ(ierr);

  ierr = PetscMalloc((N+1)*sizeof(PetscInt),&rowhit);CHKERRQ(ierr);
  ierr = PetscMalloc((N+1)*sizeof(PetscInt),&columnsforrow);CHKERRQ(ierr);

  for (i=0; i<nis; i++) {
    ierr = ISGetLocalSize(isa[i],&n);CHKERRQ(ierr);
    ierr = ISGetIndices(isa[i],&is);CHKERRQ(ierr);

    c->ncolumns[i] = n;
    if (n) {
      ierr = PetscMalloc(n*sizeof(PetscInt),&c->columns[i]);CHKERRQ(ierr);
      ierr = PetscMemcpy(c->columns[i],is,n*sizeof(PetscInt));CHKERRQ(ierr);
    } else {
      c->columns[i] = 0;
    }

    if (!flg) { /* ------------------------------------------------------------------------------*/
      /* fast, crude version requires O(N*N) work */
      ierr = PetscMemzero(rowhit,N*sizeof(PetscInt));CHKERRQ(ierr);
      /* loop over columns*/
      for (j=0; j<n; j++) {
        col  = is[j];
        rows = cj + ci[col];
        m    = ci[col+1] - ci[col];
        /* loop over columns marking them in rowhit */
        for (k=0; k<m; k++) {
          rowhit[*rows++] = col + 1;
        }
      }
      /* count the number of hits */
      nrows = 0;
      for (j=0; j<N; j++) {
        if (rowhit[j]) nrows++;
      }
      c->nrows[i] = nrows;
      ierr        = PetscMalloc((nrows+1)*sizeof(PetscInt),&c->rows[i]);CHKERRQ(ierr);
      ierr        = PetscMalloc((nrows+1)*sizeof(PetscInt),&c->columnsforrow[i]);CHKERRQ(ierr);
      nrows       = 0;
      for (j=0; j<N; j++) {
        if (rowhit[j]) {
          c->rows[i][nrows]          = j;
          c->columnsforrow[i][nrows] = rowhit[j] - 1;
          nrows++;
        }
      }
    } else {  /*-------------------------------------------------------------------------------*/
      /* slow version, using rowhit as a linked list */
      PetscInt currentcol,fm,mfm;
      rowhit[N] = N;
      nrows     = 0;
      /* loop over columns */
      for (j=0; j<n; j++) {
        col  = is[j];
        rows = cj + ci[col];
        m    = ci[col+1] - ci[col];
        /* loop over columns marking them in rowhit */
        fm = N;    /* fm points to first entry in linked list */
        for (k=0; k<m; k++) {
          currentcol = *rows++;
          /* is it already in the list? */
          do {
            mfm = fm;
            fm  = rowhit[fm];
          } while (fm < currentcol);
          /* not in list so add it */
          if (fm != currentcol) {
            nrows++;
            columnsforrow[currentcol] = col;
            /* next three lines insert new entry into linked list */
            rowhit[mfm]        = currentcol;
            rowhit[currentcol] = fm;
            fm                 = currentcol;
            /* fm points to present position in list since we know the columns are sorted */
          } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Detected invalid coloring");
        }
      }
      c->nrows[i] = nrows;
      ierr        = PetscMalloc((nrows+1)*sizeof(PetscInt),&c->rows[i]);CHKERRQ(ierr);
      ierr        = PetscMalloc((nrows+1)*sizeof(PetscInt),&c->columnsforrow[i]);CHKERRQ(ierr);
      /* now store the linked list of rows into c->rows[i] */
      nrows = 0;
      fm    = rowhit[N];
      do {
        c->rows[i][nrows]            = fm;
        c->columnsforrow[i][nrows++] = columnsforrow[fm];
        fm                           = rowhit[fm];
      } while (fm < N);
    } /* ---------------------------------------------------------------------------------------*/
    ierr = ISRestoreIndices(isa[i],&is);CHKERRQ(ierr);
  }
  ierr = MatRestoreColumnIJ(mat,0,PETSC_FALSE,PETSC_FALSE,&ncols,&ci,&cj,&done);CHKERRQ(ierr);

  ierr = PetscFree(rowhit);CHKERRQ(ierr);
  ierr = PetscFree(columnsforrow);CHKERRQ(ierr);

  /* Optimize by adding the vscale, and scaleforrow[][] fields */
  /*
       see the version for MPIAIJ
  */
  ierr = VecCreateGhost(PetscObjectComm((PetscObject)mat),mat->rmap->n,PETSC_DETERMINE,0,NULL,&c->vscale);CHKERRQ(ierr);
  ierr = PetscMalloc(c->ncolors*sizeof(PetscInt*),&c->vscaleforrow);CHKERRQ(ierr);
  for (k=0; k<c->ncolors; k++) {
    ierr = PetscMalloc((c->nrows[k]+1)*sizeof(PetscInt),&c->vscaleforrow[k]);CHKERRQ(ierr);
    for (l=0; l<c->nrows[k]; l++) {
      col = c->columnsforrow[k][l];

      c->vscaleforrow[k][l] = col;
    }
  }
  ierr = ISColoringRestoreIS(iscoloring,&isa);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

