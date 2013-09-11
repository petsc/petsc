
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/baij/seq/baij.h>

#undef __FUNCT__
#define __FUNCT__ "MatFDColoringApply_SeqBAIJ"
PetscErrorCode  MatFDColoringApply_SeqBAIJ(Mat J,MatFDColoring coloring,Vec x1,MatStructure *flag,void *sctx)
{
  PetscErrorCode (*f)(void*,Vec,Vec,void*)=(PetscErrorCode (*)(void*,Vec,Vec,void*))coloring->f;
  PetscErrorCode ierr;
  PetscInt       bs=J->rmap->bs,bs2=bs*bs,i,j,k,l,row,col,N=J->cmap->n,spidx,nz;
  PetscScalar    dx,*dy_i,*xx,*w3_array,*dy=coloring->dy; 
  PetscScalar    *vscale_array;
  PetscReal      epsilon = coloring->error_rel,umin = coloring->umin,unorm;
  Vec            w1      = coloring->w1,w2=coloring->w2,w3;
  void           *fctx   = coloring->fctx;
  PetscBool      flg     = PETSC_FALSE;
  Mat_SeqBAIJ    *csp=(Mat_SeqBAIJ*)J->data;
  PetscScalar    *ca = csp->a,*ca_l;

  PetscFunctionBegin;
  ierr = MatSetUnfactored(J);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,"-mat_fd_coloring_dont_rezero",&flg,NULL);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscInfo(coloring,"Not calling MatZeroEntries()\n");CHKERRQ(ierr);
  } else {
    PetscBool assembled;
    ierr = MatAssembled(J,&assembled);CHKERRQ(ierr);
    if (assembled) {
      ierr = MatZeroEntries(J);CHKERRQ(ierr);
    }
  }
  if (!coloring->vscale) {
    ierr = VecDuplicate(x1,&coloring->vscale);CHKERRQ(ierr);
  }

  /* Set w1 = F(x1) */
  if (!coloring->fset) {
    ierr = PetscLogEventBegin(MAT_FDColoringFunction,0,0,0,0);CHKERRQ(ierr);
    ierr = (*f)(sctx,x1,w1,fctx);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(MAT_FDColoringFunction,0,0,0,0);CHKERRQ(ierr);
  } else {
    coloring->fset = PETSC_FALSE;
  }

  if (!coloring->w3) {
    ierr = VecDuplicate(x1,&coloring->w3);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)coloring,(PetscObject)coloring->w3);CHKERRQ(ierr);
  }
  w3 = coloring->w3;

  /* Compute local scale factors: vscale = dx */
  if (coloring->htype[0] == 'w') {
    /* tacky test; need to make systematic if we add other approaches to computing h*/
    ierr = VecNorm(x1,NORM_2,&unorm);CHKERRQ(ierr);
    dx = (1.0 + unorm)*epsilon;
    ierr = VecSet(coloring->vscale,dx);CHKERRQ(ierr);
  } else {
    ierr = VecGetArray(coloring->vscale,&vscale_array);CHKERRQ(ierr);
    ierr = VecGetArray(x1,&xx);CHKERRQ(ierr);
    for (col=0; col<N; col++) {
      dx = xx[col];
      if (dx == 0.0) dx = 1.0;
      if (PetscAbsScalar(dx) < umin && PetscRealPart(dx) >= 0.0)     dx = umin;
      else if (PetscRealPart(dx) < 0.0 && PetscAbsScalar(dx) < umin) dx = -umin;
      dx               *= epsilon;
      vscale_array[col] = dx;
    }
    ierr = VecRestoreArray(coloring->vscale,&vscale_array);CHKERRQ(ierr);
    ierr = VecRestoreArray(x1,&xx);CHKERRQ(ierr);
  }

  nz = 0;
  ierr = VecGetArray(coloring->vscale,&vscale_array);CHKERRQ(ierr);
  for (k=0; k<coloring->ncolors; k++) { /*  Loop over each color */
    coloring->currentcolor = k;

    /* Compute w3 = x1 + dx */
    ierr = VecCopy(x1,w3);CHKERRQ(ierr);
    dy_i = dy;
    for (i=0; i<bs; i++) {              /* Loop over a block of columns */
      ierr = VecGetArray(w3,&w3_array);CHKERRQ(ierr);
      for (l=0; l<coloring->ncolumns[k]; l++) {
        col            = i + bs*coloring->columns[k][l];  
        w3_array[col] += vscale_array[col];
        if (i) {
          w3_array[col-1] -= vscale_array[col-1]; /* resume original w3[col-1] */
        }
      }
      ierr = VecRestoreArray(w3,&w3_array);CHKERRQ(ierr);
    
      /* Evaluate function w2 = F(w3) - F(x1) */
      ierr = PetscLogEventBegin(MAT_FDColoringFunction,0,0,0,0);CHKERRQ(ierr);
      ierr = VecPlaceArray(w2,dy_i);CHKERRQ(ierr); /* place w2 to the array dy_i */
      ierr = (*f)(sctx,w3,w2,fctx);CHKERRQ(ierr); 
      ierr = PetscLogEventEnd(MAT_FDColoringFunction,0,0,0,0);CHKERRQ(ierr);
      ierr = VecAXPY(w2,-1.0,w1);CHKERRQ(ierr);
      ierr = VecResetArray(w2);CHKERRQ(ierr);
      dy_i += N; /* points to dy+i*N */
    }

    /* Loop over row blocks, putting dy/dx into Jacobian matrix */
    for (l=0; l<coloring->nrows[k]; l++) {  
      row   = bs*coloring->rowcolden2sp3[nz++];
      col   = bs*coloring->rowcolden2sp3[nz++];
      ca_l  = ca + bs2*coloring->rowcolden2sp3[nz++];           
      spidx = 0;
      dy_i  = dy;
      for (i=0; i<bs; i++) {   /* column of the block */
        for (j=0; j<bs; j++) { /* row of the block */
          ca_l[spidx++] = dy_i[row+j]/vscale_array[col+i];
        }
        dy_i += N; /* points to dy+i*N */
      }
    }
  } /* endof for each color */
  ierr = VecRestoreArray(coloring->vscale,&vscale_array);CHKERRQ(ierr);

  coloring->currentcolor = -1;
  PetscFunctionReturn(0);
}
                                 
/* Optimize MatFDColoringApply_AIJ() by using array den2sp to skip calling MatSetValues() */
/* #define JACOBIANCOLOROPT */
#if defined(JACOBIANCOLOROPT)
#include <petsctime.h>
#endif
#undef __FUNCT__
#define __FUNCT__ "MatFDColoringApply_SeqAIJ"
PetscErrorCode MatFDColoringApply_SeqAIJ(Mat J,MatFDColoring coloring,Vec x1,MatStructure *flag,void *sctx)
{
  PetscErrorCode (*f)(void*,Vec,Vec,void*) = (PetscErrorCode (*)(void*,Vec,Vec,void*))coloring->f;
  PetscErrorCode ierr;
  PetscInt       k,l,row,col,N,bs=J->rmap->bs,bs2=bs*bs;
  PetscScalar    dx,*y,*xx,*w3_array;
  PetscScalar    *vscale_array;
  PetscReal      epsilon=coloring->error_rel,umin = coloring->umin,unorm;
  Vec            w1=coloring->w1,w2=coloring->w2,w3;
  void           *fctx=coloring->fctx;
  PetscBool      flg=PETSC_FALSE,isBAIJ=PETSC_FALSE;
  PetscScalar    *ca;
  const PetscInt ncolors=coloring->ncolors,*ncolumns=coloring->ncolumns,*nrows=coloring->nrows;
  PetscInt       **columns=coloring->columns,ncolumns_k,nrows_k,nz,spidx;
#if defined(JACOBIANCOLOROPT)
  PetscLogDouble t0,t1,t_init=0.0,t_setvals=0.0,t_func=0.0,t_dx=0.0,t_kl=0.0,t00,t11;
#endif

  PetscFunctionBegin;
#if defined(JACOBIANCOLOROPT)
  ierr = PetscTime(&t0);CHKERRQ(ierr);
#endif
  ierr = PetscObjectTypeCompare((PetscObject)J,MATSEQBAIJ,&isBAIJ);CHKERRQ(ierr);
  if (isBAIJ) { 
    Mat_SeqBAIJ     *csp=(Mat_SeqBAIJ*)J->data;
    ca = csp->a;
  } else {
    Mat_SeqAIJ     *csp=(Mat_SeqAIJ*)J->data;
    ca = csp->a;
    bs = 1; bs2 = 1;
  }

  ierr = MatSetUnfactored(J);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,"-mat_fd_coloring_dont_rezero",&flg,NULL);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscInfo(coloring,"Not calling MatZeroEntries()\n");CHKERRQ(ierr);
  } else {
    PetscBool assembled;
    ierr = MatAssembled(J,&assembled);CHKERRQ(ierr);
    if (assembled) {
      ierr = MatZeroEntries(J);CHKERRQ(ierr);
    }
  }

  if (!coloring->vscale) {
    ierr = VecDuplicate(x1,&coloring->vscale);CHKERRQ(ierr);
  }

  if (coloring->htype[0] == 'w') { /* tacky test; need to make systematic if we add other approaches to computing h*/
    ierr = VecNorm(x1,NORM_2,&unorm);CHKERRQ(ierr);
  }
#if defined(JACOBIANCOLOROPT)
    ierr = PetscTime(&t1);CHKERRQ(ierr);
    t_init += t1 - t0;
#endif

  /* Set w1 = F(x1) */
#if defined(JACOBIANCOLOROPT)
    ierr = PetscTime(&t0);CHKERRQ(ierr);
#endif
  if (!coloring->fset) {
    ierr = PetscLogEventBegin(MAT_FDColoringFunction,0,0,0,0);CHKERRQ(ierr);
    ierr = (*f)(sctx,x1,w1,fctx);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(MAT_FDColoringFunction,0,0,0,0);CHKERRQ(ierr);
  } else {
    coloring->fset = PETSC_FALSE;
  }
#if defined(JACOBIANCOLOROPT)
    ierr = PetscTime(&t1);CHKERRQ(ierr);
    t_setvals += t1 - t0;
#endif

#if defined(JACOBIANCOLOROPT)
    ierr = PetscTime(&t0);CHKERRQ(ierr);
#endif
  if (!coloring->w3) {
    ierr = VecDuplicate(x1,&coloring->w3);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)coloring,(PetscObject)coloring->w3);CHKERRQ(ierr);
  } 
  w3 = coloring->w3;
  
  /* Compute scale factors: vscale = 1./dx = 1./(epsilon*xx) */
  ierr = VecGetArray(x1,&xx);CHKERRQ(ierr);
  ierr = VecGetArray(coloring->vscale,&vscale_array);CHKERRQ(ierr);
  ierr = VecGetSize(x1,&N);CHKERRQ(ierr);
  for (col=0; col<N; col++) {
    if (coloring->htype[0] == 'w') {
      dx = 1.0 + unorm;
    } else {
      dx = xx[col];
    }
    if (dx == (PetscScalar)0.0) dx = 1.0;
    if (PetscAbsScalar(dx) < umin && PetscRealPart(dx) >= 0.0)     dx = umin;
    else if (PetscRealPart(dx) < 0.0 && PetscAbsScalar(dx) < umin) dx = -umin;
    dx               *= epsilon;
    vscale_array[col] = (PetscScalar)dx;
  }
#if defined(JACOBIANCOLOROPT)
    ierr = PetscTime(&t1);CHKERRQ(ierr);
    t_init += t1 - t0;
#endif
 
  nz  = 0;
  for (k=0; k<ncolors; k++) { /* loop over colors */
#if defined(JACOBIANCOLOROPT)
    ierr = PetscTime(&t0);CHKERRQ(ierr);
#endif
    coloring->currentcolor = k;

    /*
      Loop over each column associated with color
      adding the perturbation to the vector w3 = x1 + dx.
    */
    ierr = VecCopy(x1,w3);CHKERRQ(ierr);
    ierr = VecGetArray(w3,&w3_array);CHKERRQ(ierr);
    ncolumns_k = ncolumns[k];
    for (l=0; l<ncolumns_k; l++) { /* loop over columns */
      col = columns[k][l];   
      w3_array[col] += vscale_array[col]; 
    }
    ierr = VecRestoreArray(w3,&w3_array);CHKERRQ(ierr);
#if defined(JACOBIANCOLOROPT)
    ierr = PetscTime(&t1);CHKERRQ(ierr);
    t_dx += t1 - t0;
#endif

    /*
      Evaluate function at w3 = x1 + dx (here dx is a vector of perturbations)
                           w2 = F(x1 + dx) - F(x1)
    */
#if defined(JACOBIANCOLOROPT)
    ierr = PetscTime(&t0);CHKERRQ(ierr);
#endif
    ierr = PetscLogEventBegin(MAT_FDColoringFunction,0,0,0,0);CHKERRQ(ierr);
    ierr = (*f)(sctx,w3,w2,fctx);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(MAT_FDColoringFunction,0,0,0,0);CHKERRQ(ierr);
#if defined(JACOBIANCOLOROPT)
    ierr = PetscTime(&t1);CHKERRQ(ierr);
    t_func += t1 - t0;
#endif

#if defined(JACOBIANCOLOROPT)
    ierr = PetscTime(&t0);CHKERRQ(ierr);
#endif
    ierr = VecAXPY(w2,-1.0,w1);CHKERRQ(ierr);

    /*
      Loop over rows of vector, putting w2/dx into Jacobian matrix
    */
    ierr = VecGetArray(w2,&y);CHKERRQ(ierr);
    nrows_k = nrows[k];
    for (l=0; l<nrows_k; l++) { /* loop over rows */
#if defined(JACOBIANCOLOROPT)
    ierr = PetscTime(&t00);CHKERRQ(ierr);
#endif
      row   = coloring->rowcolden2sp3[nz++];
      col   = coloring->rowcolden2sp3[nz++];
      spidx = coloring->rowcolden2sp3[nz++];
#if defined(JACOBIANCOLOROPT)
      ierr = PetscTime(&t11);CHKERRQ(ierr);
      t_kl += t11 - t00;
#endif
      //printf(" row col val
      ca[spidx*bs2] = y[row*bs]/vscale_array[col*bs];
      //printf(" (%d, %d, %g)\n", row*bs,col*bs,ca[spidx*bs2]);
    }
    ierr = VecRestoreArray(w2,&y);CHKERRQ(ierr);

#if defined(JACOBIANCOLOROPT)
    ierr = PetscTime(&t1);CHKERRQ(ierr);
    t_setvals += t1 - t0;
#endif
  } /* endof for each color */
#if defined(JACOBIANCOLOROPT)
  printf("     FDColorApply time: init %g + func %g + setvalues %g + dx %g= %g\n",t_init,t_func,t_setvals,t_dx,t_init+t_func+t_setvals+t_dx);
  printf("     FDColorApply time: kl %g\n",t_kl);
#endif
  ierr = VecRestoreArray(coloring->vscale,&vscale_array);CHKERRQ(ierr);
  ierr = VecRestoreArray(x1,&xx);CHKERRQ(ierr);

  coloring->currentcolor = -1;
  PetscFunctionReturn(0);
}

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
  PetscInt       nis=iscoloring->n,*rowhit,*columnsforrow,bs,*spidx,*spidxhit,nz;
  IS             *isa;
  PetscBool      isBAIJ;     
  Mat_SeqAIJ     *csp = (Mat_SeqAIJ*)mat->data;

  PetscFunctionBegin;
  ierr = ISColoringGetIS(iscoloring,PETSC_IGNORE,&isa);CHKERRQ(ierr);

  /* this is ugly way to get blocksize but cannot call MatGetBlockSize() because AIJ can have bs > 1.
     SeqBAIJ calls this routine, thus check it */
  ierr = MatGetBlockSize(mat,&bs);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)mat,MATSEQBAIJ,&isBAIJ);CHKERRQ(ierr);
  if (!isBAIJ) {
    bs = 1; /* only bs=1 is supported for non SEQBAIJ matrix */
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
  ierr       = PetscMalloc(3*csp->nz*sizeof(PetscInt*),&c->rowcolden2sp3);CHKERRQ(ierr);

  if (isBAIJ) {
    ierr = MatGetColumnIJ_SeqBAIJ_Color(mat,0,PETSC_FALSE,PETSC_FALSE,&ncols,&ci,&cj,&spidx,NULL);CHKERRQ(ierr);
  } else {
    ierr = MatGetColumnIJ_SeqAIJ_Color(mat,0,PETSC_FALSE,PETSC_FALSE,&ncols,&ci,&cj,&spidx,NULL);CHKERRQ(ierr);
  }
 
  ierr = PetscMalloc3(c->m,PetscInt,&rowhit,N,PetscInt,&columnsforrow,c->m,PetscInt,&spidxhit);CHKERRQ(ierr);
  ierr = PetscMemzero(rowhit,c->m*sizeof(PetscInt));CHKERRQ(ierr);

  nz = 0;
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
   
    nrows = 0;
    for (j=0; j<N; j++) { /* loop over rows */
      if (rowhit[j]) {
        c->rowcolden2sp3[nz++]   = j;             /* row index */
        c->rowcolden2sp3[nz++] = rowhit[j] - 1;   /* column index */
        c->rowcolden2sp3[nz++] = spidxhit[j];     /* den2sp index */
        rowhit[j] = 0.0;                          /* zero rowhit for reuse */
      }
    } 
    ierr = ISRestoreIndices(isa[i],&is);CHKERRQ(ierr);
  }
  if (nz/3 != csp->nz) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"nz %d != mat->nz %d\n",nz/3,csp->nz); 

  if (isBAIJ) {
    ierr = MatRestoreColumnIJ_SeqBAIJ_Color(mat,0,PETSC_FALSE,PETSC_FALSE,&ncols,&ci,&cj,&spidx,NULL);CHKERRQ(ierr);
  } else {
    ierr = MatRestoreColumnIJ_SeqAIJ_Color(mat,0,PETSC_FALSE,PETSC_FALSE,&ncols,&ci,&cj,&spidx,NULL);CHKERRQ(ierr);
  }
  ierr = PetscFree3(rowhit,columnsforrow,spidxhit);CHKERRQ(ierr);
  ierr = ISColoringRestoreIS(iscoloring,&isa);CHKERRQ(ierr);

  c->ctype                  = IS_COLORING_GHOSTED;
  if (isBAIJ) {
    mat->ops->fdcoloringapply = MatFDColoringApply_SeqBAIJ;
    ierr = PetscMalloc(bs*mat->cmap->N*sizeof(PetscScalar),&c->dy);CHKERRQ(ierr);
  } else {
    mat->ops->fdcoloringapply = MatFDColoringApply_SeqAIJ;
  }
  ierr = VecCreateGhost(PetscObjectComm((PetscObject)mat),mat->rmap->n,PETSC_DETERMINE,0,NULL,&c->vscale);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}

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

