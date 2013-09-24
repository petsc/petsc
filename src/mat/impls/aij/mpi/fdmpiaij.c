
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <../src/mat/impls/baij/mpi/mpibaij.h>

#undef __FUNCT__
#define __FUNCT__ "MatFDColoringApply_BAIJ_new"
PetscErrorCode  MatFDColoringApply_BAIJ_new(Mat J,MatFDColoring coloring,Vec x1,MatStructure *flag,void *sctx)
{
  PetscErrorCode (*f)(void*,Vec,Vec,void*) = (PetscErrorCode (*)(void*,Vec,Vec,void*))coloring->f;
  PetscErrorCode ierr;
  PetscInt       k,cstart,cend,l,row,col,nz,spidx,i,j;
  PetscScalar    dx=0.0,*y,*xx,*w3_array,*dy_i,*dy=coloring->dy;
  PetscScalar    *vscale_array;
  PetscReal      epsilon=coloring->error_rel,umin=coloring->umin,unorm;
  Vec            w1=coloring->w1,w2=coloring->w2,w3,vscale=coloring->vscale;
  void           *fctx=coloring->fctx;
  PetscBool      flg=PETSC_FALSE;
  PetscInt       ctype=coloring->ctype,nxloc,nrows_k;
  Mat_MPIBAIJ    *aij=(Mat_MPIBAIJ*)J->data;
  PetscScalar    *valaddr;
  MatEntry       *Jentry=coloring->matentry;
  const PetscInt ncolors=coloring->ncolors,*ncolumns=coloring->ncolumns,*nrows=coloring->nrows;
  PetscInt       bs=J->rmap->bs;

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

  /* create vscale for storing dx */
  if (!vscale) {
    if (ctype == IS_COLORING_GLOBAL && coloring->htype[0] == 'd') {
      ierr = VecCreateGhost(PetscObjectComm((PetscObject)J),J->cmap->n,PETSC_DETERMINE,aij->B->cmap->n,aij->garray,&vscale);CHKERRQ(ierr);
      
    } else if (ctype == IS_COLORING_GHOSTED) {
      ierr = VecDuplicate(x1,&vscale);CHKERRQ(ierr);
    }
    coloring->vscale = vscale;
  }
  
  /* (1) Set w1 = F(x1) */
  if (!coloring->fset) {
    ierr = PetscLogEventBegin(MAT_FDColoringFunction,0,0,0,0);CHKERRQ(ierr);
    ierr = (*f)(sctx,x1,w1,fctx);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(MAT_FDColoringFunction,0,0,0,0);CHKERRQ(ierr);
  } else {
    coloring->fset = PETSC_FALSE;
  }
  
  /* (2) Compute vscale = 1./dx - the local scale factors, including ghost points */
  ierr = VecGetLocalSize(x1,&nxloc);CHKERRQ(ierr);
  //PetscMPIInt rank;
  //ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  //printf("[%d] nxloc %d\n",rank,nxloc);
  if (coloring->htype[0] == 'w') { 
    /* vscale = dx is a constant scalar */
    ierr = VecNorm(x1,NORM_2,&unorm);CHKERRQ(ierr);
    dx = 1.0/(PetscSqrtReal(1.0 + unorm)*epsilon); 
  } else { 
    ierr = VecGetArray(x1,&xx);CHKERRQ(ierr);
    ierr = VecGetArray(vscale,&vscale_array);CHKERRQ(ierr);
    for (col=0; col<nxloc; col++) {
      dx = xx[col];
      if (PetscAbsScalar(dx) < umin) {
        if (PetscRealPart(dx) >= 0.0)      dx = umin;
        else if (PetscRealPart(dx) < 0.0 ) dx = -umin;
      }
      dx               *= epsilon;
      vscale_array[col] = 1.0/dx;
    }
    ierr = VecRestoreArray(x1,&xx);CHKERRQ(ierr);
    ierr = VecRestoreArray(vscale,&vscale_array);CHKERRQ(ierr);
  }
  if (ctype == IS_COLORING_GLOBAL && coloring->htype[0] != 'w') {
    ierr = VecGhostUpdateBegin(vscale,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecGhostUpdateEnd(vscale,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  }

  /* (3) Loop over each color */
  if (!coloring->w3) {
    ierr = VecDuplicate(x1,&coloring->w3);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)coloring,(PetscObject)coloring->w3);CHKERRQ(ierr);
  }
  w3 = coloring->w3;
  
  ierr = VecGetOwnershipRange(x1,&cstart,&cend);CHKERRQ(ierr); /* used by ghosted vscale */
  if (vscale) {
    ierr = VecGetArray(vscale,&vscale_array);CHKERRQ(ierr);
  }
  nz   = 0;
  for (k=0; k<ncolors; k++) {
    coloring->currentcolor = k;

    /*
      (3-1) Loop over each column associated with color
      adding the perturbation to the vector w3 = x1 + dx.
    */
    ierr = VecCopy(x1,w3);CHKERRQ(ierr);
    dy_i = dy;
    for (i=0; i<bs; i++) {     /* Loop over a block of columns */ 
      //------------------------------------------------
      ierr = VecGetArray(w3,&w3_array);CHKERRQ(ierr);
      if (ctype == IS_COLORING_GLOBAL) w3_array -= cstart; /* shift pointer so global index can be used */
      if (coloring->htype[0] == 'w') {
        for (l=0; l<ncolumns[k]; l++) {
          col            = i + bs*coloring->columns[k][l];  /* local column (in global index!) of the matrix we are probing for */
          w3_array[col] += 1.0/dx;

          if (i) {
            w3_array[col-1] -= 1.0/dx; /* resume original w3[col-1] */
          }

        } 
      } else { /* htype == 'ds' */
        vscale_array -= cstart; /* shift pointer so global index can be used */
        for (l=0; l<ncolumns[k]; l++) {
          col = coloring->columns[k][l]; /* local column (in global index!) of the matrix we are probing for */
          w3_array[col] += 1.0/vscale_array[col];

          if (i) {
            w3_array[col-1] -=  1.0/vscale_array[col-1]; /* resume original w3[col-1] */
          }

        }
        vscale_array += cstart;
      }
      if (ctype == IS_COLORING_GLOBAL) w3_array += cstart;
      ierr = VecRestoreArray(w3,&w3_array);CHKERRQ(ierr);
    
      /*
       (3-2) Evaluate function at w3 = x1 + dx (here dx is a vector of perturbations)
                           w2 = F(x1 + dx) - F(x1)
       */
      ierr = PetscLogEventBegin(MAT_FDColoringFunction,0,0,0,0);CHKERRQ(ierr);
      ierr = VecPlaceArray(w2,dy_i);CHKERRQ(ierr); /* place w2 to the array dy_i */
      ierr = (*f)(sctx,w3,w2,fctx);CHKERRQ(ierr);
      ierr = PetscLogEventEnd(MAT_FDColoringFunction,0,0,0,0);CHKERRQ(ierr);
      ierr = VecAXPY(w2,-1.0,w1);CHKERRQ(ierr);
      //---------------------------------------------
      ierr = VecResetArray(w2);CHKERRQ(ierr);
      dy_i += nxloc; /* points to dy+i*nxloc */
    }

    /* 
     (3-3) Loop over rows of vector, putting results into Jacobian matrix 
    */
    nrows_k = nrows[k];
    ierr = VecGetArray(w2,&y);CHKERRQ(ierr);
    if (coloring->htype[0] == 'w') {
      for (l=0; l<nrows_k; l++) { 
        row     = bs*Jentry[nz].row;   /* local row index */
        valaddr = Jentry[nz].valaddr;
        nz++;
        spidx = 0;
        dy_i  = dy;
        for (i=0; i<bs; i++) {   /* column of the block */
          for (j=0; j<bs; j++) { /* row of the block */
            valaddr[spidx++] = dy_i[row+j]*dx;
          }
          dy_i += nxloc; /* points to dy+i*N */
        }
      }
    } else { /* htype == 'ds' */
      for (l=0; l<nrows_k; l++) { 
        row                   = bs*Jentry[nz].row;   /* local row index */
        col                   = bs*Jentry[nz].col;   /* local column index */
        //*(Jentry[nz].valaddr) = y[row]*vscale_array[col];
        nz++;
      }
    }
    ierr = VecRestoreArray(w2,&y);CHKERRQ(ierr);
  } 
  ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (vscale) {
    ierr = VecRestoreArray(vscale,&vscale_array);CHKERRQ(ierr);
  }

  coloring->currentcolor = -1;
  PetscFunctionReturn(0);
}

extern PetscErrorCode MatCreateColmap_MPIAIJ_Private(Mat);
#undef __FUNCT__
#define __FUNCT__ "MatFDColoringApply_AIJ_new"
PetscErrorCode  MatFDColoringApply_AIJ_new(Mat J,MatFDColoring coloring,Vec x1,MatStructure *flag,void *sctx)
{
  PetscErrorCode (*f)(void*,Vec,Vec,void*) = (PetscErrorCode (*)(void*,Vec,Vec,void*))coloring->f;
  PetscErrorCode ierr;
  PetscInt       k,cstart,cend,l,row,col,nz;
  PetscScalar    dx=0.0,*y,*xx,*w3_array;
  PetscScalar    *vscale_array;
  PetscReal      epsilon=coloring->error_rel,umin=coloring->umin,unorm;
  Vec            w1=coloring->w1,w2=coloring->w2,w3,vscale=coloring->vscale;
  void           *fctx=coloring->fctx;
  PetscBool      flg=PETSC_FALSE;
  PetscInt       ctype=coloring->ctype,nxloc,nrows_k;
  Mat_MPIAIJ     *aij=(Mat_MPIAIJ*)J->data;
  MatEntry       *Jentry=coloring->matentry;
  const PetscInt ncolors=coloring->ncolors,*ncolumns=coloring->ncolumns,*nrows=coloring->nrows;
  
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

  /* create vscale for storing dx */
  if (!vscale) {
    if (ctype == IS_COLORING_GLOBAL && coloring->htype[0] == 'd') {
      ierr = VecCreateGhost(PetscObjectComm((PetscObject)J),J->cmap->n,PETSC_DETERMINE,aij->B->cmap->n,aij->garray,&vscale);CHKERRQ(ierr);
    } else if (ctype == IS_COLORING_GHOSTED) {
      ierr = VecDuplicate(x1,&vscale);CHKERRQ(ierr);
    }
    coloring->vscale = vscale;
  }
  
  /* (1) Set w1 = F(x1) */
  if (!coloring->fset) {
    ierr = PetscLogEventBegin(MAT_FDColoringFunction,0,0,0,0);CHKERRQ(ierr);
    ierr = (*f)(sctx,x1,w1,fctx);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(MAT_FDColoringFunction,0,0,0,0);CHKERRQ(ierr);
  } else {
    coloring->fset = PETSC_FALSE;
  }
  
  /* (2) Compute vscale = 1./dx - the local scale factors, including ghost points */
  if (coloring->htype[0] == 'w') { 
    /* vscale = dx is a constant scalar */
    ierr = VecNorm(x1,NORM_2,&unorm);CHKERRQ(ierr);
    dx = 1.0/(PetscSqrtReal(1.0 + unorm)*epsilon); 
  } else { 
    ierr = VecGetLocalSize(x1,&nxloc);CHKERRQ(ierr);
    ierr = VecGetArray(x1,&xx);CHKERRQ(ierr);
    ierr = VecGetArray(vscale,&vscale_array);CHKERRQ(ierr);
    for (col=0; col<nxloc; col++) {
      dx = xx[col];
      if (PetscAbsScalar(dx) < umin) {
        if (PetscRealPart(dx) >= 0.0)      dx = umin;
        else if (PetscRealPart(dx) < 0.0 ) dx = -umin;
      }
      dx               *= epsilon;
      vscale_array[col] = 1.0/dx;
    }
    ierr = VecRestoreArray(x1,&xx);CHKERRQ(ierr);
    ierr = VecRestoreArray(vscale,&vscale_array);CHKERRQ(ierr);
  }
  if (ctype == IS_COLORING_GLOBAL && coloring->htype[0] != 'w') {
    ierr = VecGhostUpdateBegin(vscale,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecGhostUpdateEnd(vscale,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  }

  /* (3) Loop over each color */
  if (!coloring->w3) {
    ierr = VecDuplicate(x1,&coloring->w3);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)coloring,(PetscObject)coloring->w3);CHKERRQ(ierr);
  }
  w3 = coloring->w3;
  
  ierr = VecGetOwnershipRange(x1,&cstart,&cend);CHKERRQ(ierr); /* used by ghosted vscale */
  if (vscale) {
    ierr = VecGetArray(vscale,&vscale_array);CHKERRQ(ierr);
  }
  nz   = 0;
  for (k=0; k<ncolors; k++) {
    coloring->currentcolor = k;

    /*
      (3-1) Loop over each column associated with color
      adding the perturbation to the vector w3 = x1 + dx.
    */
    ierr = VecCopy(x1,w3);CHKERRQ(ierr);
    ierr = VecGetArray(w3,&w3_array);CHKERRQ(ierr);
    if (ctype == IS_COLORING_GLOBAL) w3_array -= cstart; /* shift pointer so global index can be used */
    if (coloring->htype[0] == 'w') {
      for (l=0; l<ncolumns[k]; l++) {
        col = coloring->columns[k][l]; /* local column (in global index!) of the matrix we are probing for */
        w3_array[col] += 1.0/dx;
      } 
    } else { /* htype == 'ds' */
      vscale_array -= cstart; /* shift pointer so global index can be used */
      for (l=0; l<ncolumns[k]; l++) {
        col = coloring->columns[k][l]; /* local column (in global index!) of the matrix we are probing for */
        w3_array[col] += 1.0/vscale_array[col];
      }
      vscale_array += cstart;
    }
    if (ctype == IS_COLORING_GLOBAL) w3_array += cstart;
    ierr = VecRestoreArray(w3,&w3_array);CHKERRQ(ierr);

    /*
      (3-2) Evaluate function at w3 = x1 + dx (here dx is a vector of perturbations)
                           w2 = F(x1 + dx) - F(x1)
    */
    ierr = PetscLogEventBegin(MAT_FDColoringFunction,0,0,0,0);CHKERRQ(ierr);
    ierr = (*f)(sctx,w3,w2,fctx);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(MAT_FDColoringFunction,0,0,0,0);CHKERRQ(ierr);
    ierr = VecAXPY(w2,-1.0,w1);CHKERRQ(ierr);

    /* 
     (3-3) Loop over rows of vector, putting results into Jacobian matrix 
    */
    nrows_k = nrows[k];
    ierr = VecGetArray(w2,&y);CHKERRQ(ierr);
    if (coloring->htype[0] == 'w') {
      for (l=0; l<nrows_k; l++) { 
        row                     = Jentry[nz].row;   /* local row index */
        *(Jentry[nz++].valaddr) = y[row]*dx;
      }
    } else { /* htype == 'ds' */
      for (l=0; l<nrows_k; l++) { 
        row                   = Jentry[nz].row;   /* local row index */
        *(Jentry[nz].valaddr) = y[row]*vscale_array[Jentry[nz].col];
        nz++;
      }
    }
    ierr = VecRestoreArray(w2,&y);CHKERRQ(ierr);
  } 
  ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (vscale) {
    ierr = VecRestoreArray(vscale,&vscale_array);CHKERRQ(ierr);
  }

  coloring->currentcolor = -1;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatFDColoringCreate_MPIAIJ_new"
PetscErrorCode MatFDColoringCreate_MPIAIJ_new(Mat mat,ISColoring iscoloring,MatFDColoring c)
{
  //Mat_MPIAIJ             *aij=(Mat_MPIAIJ*)mat->data;
  PetscErrorCode         ierr;
  PetscMPIInt            size,*ncolsonproc,*disp,nn;
  PetscInt               i,n,nrows,nrows_i,j,k,m,ncols,col;
  const PetscInt         *is,*A_ci,*A_cj,*B_ci,*B_cj,*row = NULL,*ltog=NULL;
  PetscInt               nis=iscoloring->n,nctot,*cols;
  PetscInt               *rowhit,cstart,cend,colb;
  IS                     *isa;
  ISLocalToGlobalMapping map=mat->cmap->mapping;
  PetscInt               ctype=c->ctype;
  Mat                    A,B;
  //Mat                    A=aij->A,B=aij->B;
  //Mat_SeqAIJ             *spA=(Mat_SeqAIJ*)A->data,*spB=(Mat_SeqAIJ*)B->data;
  //PetscScalar            *A_val=spA->a,*B_val=spB->a;
  PetscScalar            *A_val,*B_val;
  PetscInt               spidx;
  PetscInt               *spidxA,*spidxB,nz,bs,bs2;
  PetscScalar            **valaddrhit;
  MatEntry               *Jentry;
  PetscBool              isBAIJ; 
#if defined(PETSC_USE_CTABLE)
  PetscTable             colmap=NULL;
#else
  PetscInt               *colmap=NULL;     /* local col number of off-diag col */
#endif

  PetscFunctionBegin;
  if (ctype == IS_COLORING_GHOSTED) {
    if (!map) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_INCOMP,"When using ghosted differencing matrix must have local to global mapping provided with MatSetLocalToGlobalMapping");
    ierr = ISLocalToGlobalMappingGetIndices(map,&ltog);CHKERRQ(ierr);
  }

  ierr = MatGetBlockSize(mat,&bs);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)mat,MATMPIBAIJ,&isBAIJ);CHKERRQ(ierr);
  if (!isBAIJ) {
    bs = 1; /* only bs=1 is supported for non MPIBAIJ matrix */
    Mat_MPIAIJ             *aij=(Mat_MPIAIJ*)mat->data;
    A=aij->A; B=aij->B;
    Mat_SeqAIJ             *spA=(Mat_SeqAIJ*)A->data,*spB=(Mat_SeqAIJ*)B->data;
    A_val=spA->a; B_val=spB->a;
    nz         = spA->nz + spB->nz; /* total nonzero entries of mat */  
    if (!aij->colmap) {
      ierr = MatCreateColmap_MPIAIJ_Private(mat);CHKERRQ(ierr);
      colmap = aij->colmap;
    }
    ierr = MatGetColumnIJ_SeqAIJ_Color(A,0,PETSC_FALSE,PETSC_FALSE,&ncols,&A_ci,&A_cj,&spidxA,NULL);CHKERRQ(ierr);
    ierr = MatGetColumnIJ_SeqAIJ_Color(B,0,PETSC_FALSE,PETSC_FALSE,&ncols,&B_ci,&B_cj,&spidxB,NULL);CHKERRQ(ierr);
  } else {
    Mat_MPIBAIJ             *aij=(Mat_MPIBAIJ*)mat->data;
    A=aij->A; B=aij->B;
    Mat_SeqBAIJ             *spA=(Mat_SeqBAIJ*)A->data,*spB=(Mat_SeqBAIJ*)B->data;
    A_val=spA->a; B_val=spB->a;
    nz         = spA->nz + spB->nz; /* total nonzero entries of mat */  
    if (!aij->colmap) {
      ierr = MatCreateColmap_MPIBAIJ_Private(mat);CHKERRQ(ierr);
      colmap = aij->colmap;
    }
    ierr = MatGetColumnIJ_SeqBAIJ_Color(A,0,PETSC_FALSE,PETSC_FALSE,&ncols,&A_ci,&A_cj,&spidxA,NULL);CHKERRQ(ierr);
    ierr = MatGetColumnIJ_SeqBAIJ_Color(B,0,PETSC_FALSE,PETSC_FALSE,&ncols,&B_ci,&B_cj,&spidxB,NULL);CHKERRQ(ierr);
  }

  m         = mat->rmap->n/bs;
  cstart    = mat->cmap->rstart/bs;
  cend      = mat->cmap->rend/bs;
  c->M      = mat->rmap->N/bs;         /* set the global rows and columns and local rows */
  c->N      = mat->cmap->N/bs;
  c->m      = m; 
  c->rstart = mat->rmap->rstart/bs;

  c->ncolors = nis;
  ierr       = PetscMalloc(nis*sizeof(PetscInt),&c->ncolumns);CHKERRQ(ierr);
  ierr       = PetscMalloc(nis*sizeof(PetscInt*),&c->columns);CHKERRQ(ierr);
  ierr       = PetscMalloc(nis*sizeof(PetscInt),&c->nrows);CHKERRQ(ierr);
  ierr       = PetscLogObjectMemory((PetscObject)c,3*nis*sizeof(PetscInt));CHKERRQ(ierr);

  //nz         = spA->nz + spB->nz; /* total nonzero entries of mat */                                  
  ierr       = PetscMalloc(nz*sizeof(MatEntry),&Jentry);CHKERRQ(ierr);
  ierr       = PetscLogObjectMemory((PetscObject)c,nz*sizeof(MatEntry));CHKERRQ(ierr);
  c->matentry = Jentry;

  /* Allow access to data structures of local part of matrix 
   - creates aij->colmap which maps global column number to local number in part B */
  //if (!aij->colmap) {
  //  ierr = MatCreateColmap_MPIAIJ_Private(mat);CHKERRQ(ierr);
  //}
  //ierr = MatGetColumnIJ_SeqAIJ_Color(aij->A,0,PETSC_FALSE,PETSC_FALSE,&ncols,&A_ci,&A_cj,&spidxA,NULL);CHKERRQ(ierr);
  //ierr = MatGetColumnIJ_SeqAIJ_Color(aij->B,0,PETSC_FALSE,PETSC_FALSE,&ncols,&B_ci,&B_cj,&spidxB,NULL);CHKERRQ(ierr);

  ierr = PetscMalloc2(m+1,PetscInt,&rowhit,m+1,PetscScalar*,&valaddrhit);CHKERRQ(ierr);
  nz = 0;
  ierr = ISColoringGetIS(iscoloring,PETSC_IGNORE,&isa);CHKERRQ(ierr);
  for (i=0; i<nis; i++) { /* for each local color */
    ierr = ISGetLocalSize(isa[i],&n);CHKERRQ(ierr);
    ierr = ISGetIndices(isa[i],&is);CHKERRQ(ierr);

    c->ncolumns[i] = n; /* local number of columns of this color on this process */
    if (n) {
      ierr = PetscMalloc(n*sizeof(PetscInt),&c->columns[i]);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory((PetscObject)c,n*sizeof(PetscInt));CHKERRQ(ierr);
      ierr = PetscMemcpy(c->columns[i],is,n*sizeof(PetscInt));CHKERRQ(ierr);
      /* convert global column indices to local ones! */

    } else {
      c->columns[i] = 0;
    }

    if (ctype == IS_COLORING_GLOBAL) {
      /* Determine nctot, the total (parallel) number of columns of this color */
      ierr = MPI_Comm_size(PetscObjectComm((PetscObject)mat),&size);CHKERRQ(ierr);
      ierr = PetscMalloc2(size,PetscMPIInt,&ncolsonproc,size,PetscMPIInt,&disp);CHKERRQ(ierr);

      /* ncolsonproc[j]: local ncolumns on proc[j] of this color */
      ierr  = PetscMPIIntCast(n,&nn);CHKERRQ(ierr);
      ierr  = MPI_Allgather(&nn,1,MPI_INT,ncolsonproc,1,MPI_INT,PetscObjectComm((PetscObject)mat));CHKERRQ(ierr);
      nctot = 0; for (j=0; j<size; j++) nctot += ncolsonproc[j];
      if (!nctot) {
        ierr = PetscInfo(mat,"Coloring of matrix has some unneeded colors with no corresponding rows\n");CHKERRQ(ierr);
      }

      disp[0] = 0;
      for (j=1; j<size; j++) {
        disp[j] = disp[j-1] + ncolsonproc[j-1];
      }

      /* Get cols, the complete list of columns for this color on each process */
      ierr = PetscMalloc((nctot+1)*sizeof(PetscInt),&cols);CHKERRQ(ierr);
      ierr = MPI_Allgatherv((void*)is,n,MPIU_INT,cols,ncolsonproc,disp,MPIU_INT,PetscObjectComm((PetscObject)mat));CHKERRQ(ierr);
      ierr = PetscFree2(ncolsonproc,disp);CHKERRQ(ierr);
    } else if (ctype == IS_COLORING_GHOSTED) {
      /* Determine local number of columns of this color on this process, including ghost points */
      nctot = n;
      ierr  = PetscMalloc((nctot+1)*sizeof(PetscInt),&cols);CHKERRQ(ierr);
      ierr  = PetscMemcpy(cols,is,n*sizeof(PetscInt));CHKERRQ(ierr);
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not provided for this MatFDColoring type");

    /* Mark all rows affect by these columns */
    ierr = PetscMemzero(rowhit,m*sizeof(PetscInt));CHKERRQ(ierr);
    bs2     = bs*bs;
    nrows_i = 0;
    for (j=0; j<nctot; j++) { /* loop over columns*/
      if (ctype == IS_COLORING_GHOSTED) {
        col = ltog[cols[j]];
      } else {
        col = cols[j];
      }
      if (col >= cstart && col < cend) { /* column is in diagonal block of matrix A */
        row      = A_cj + A_ci[col-cstart];
        nrows    = A_ci[col-cstart+1] - A_ci[col-cstart];
        nrows_i += nrows;
        /* loop over columns of A marking them in rowhit */
        for (k=0; k<nrows; k++) {
          /* set valaddrhit for part A */
          spidx            = spidxA[A_ci[col-cstart] + k];
          valaddrhit[*row] = &A_val[bs2*spidx]; 
          rowhit[*row++]   = col - cstart + 1; /* local column index */
        }
      } else { /* column is in off-diagonal block of matrix B */
#if defined(PETSC_USE_CTABLE)
        //ierr = PetscTableFind(aij->colmap,col+1,&colb);CHKERRQ(ierr);
        ierr = PetscTableFind(colmap,col+1,&colb);CHKERRQ(ierr);
        colb--;
#else
        //colb = aij->colmap[col] - 1; /* local column index */
        colb = colmap[col] - 1; /* local column index */
#endif
        if (colb == -1) {
          nrows = 0;
        } else {
          colb  = colb/bs;
          row   = B_cj + B_ci[colb];
          nrows = B_ci[colb+1] - B_ci[colb];
        }
        nrows_i += nrows;
        /* loop over columns of B marking them in rowhit */
        for (k=0; k<nrows; k++) {
          /* set valaddrhit for part B */
          spidx            = spidxB[B_ci[colb] + k];
          valaddrhit[*row] = &B_val[bs2*spidx]; 
          rowhit[*row++]   = colb + 1 + cend - cstart; /* local column index */
        }
      }
    }
    c->nrows[i] = nrows_i;
    
    for (j=0; j<m; j++) {
      if (rowhit[j]) {
        Jentry[nz].row     = j;              /* local row index */
        Jentry[nz].col     = rowhit[j] - 1;  /* local column index */
        Jentry[nz].valaddr = valaddrhit[j];  /* address of mat value for this entry */ 
        nz++;
      }
    }
    ierr = PetscFree(cols);CHKERRQ(ierr);
  }
  ierr = ISColoringRestoreIS(iscoloring,&isa);CHKERRQ(ierr);
  //if (nz != spA->nz + spB->nz) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"nz %d != mat->nz %d\n",nz,spA->nz+spB->nz); 

  ierr = PetscFree2(rowhit,valaddrhit);CHKERRQ(ierr);
  if (isBAIJ) {
    ierr = MatRestoreColumnIJ_SeqBAIJ_Color(A,0,PETSC_FALSE,PETSC_FALSE,&ncols,&A_ci,&A_cj,&spidxA,NULL);CHKERRQ(ierr);
    ierr = MatRestoreColumnIJ_SeqBAIJ_Color(B,0,PETSC_FALSE,PETSC_FALSE,&ncols,&B_ci,&B_cj,&spidxB,NULL);CHKERRQ(ierr);
    ierr = PetscMalloc(bs*mat->cmap->n*sizeof(PetscScalar),&c->dy);CHKERRQ(ierr);
    mat->ops->fdcoloringapply = MatFDColoringApply_BAIJ_new;
  } else {
    ierr = MatRestoreColumnIJ_SeqAIJ_Color(A,0,PETSC_FALSE,PETSC_FALSE,&ncols,&A_ci,&A_cj,&spidxA,NULL);CHKERRQ(ierr);
    ierr = MatRestoreColumnIJ_SeqAIJ_Color(B,0,PETSC_FALSE,PETSC_FALSE,&ncols,&B_ci,&B_cj,&spidxB,NULL);CHKERRQ(ierr);
    mat->ops->fdcoloringapply = MatFDColoringApply_AIJ_new;
  }

  if (ctype == IS_COLORING_GHOSTED) {
    ierr = ISLocalToGlobalMappingRestoreIndices(map,&ltog);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "MatFDColoringCreate_MPIAIJ"
PetscErrorCode MatFDColoringCreate_MPIAIJ(Mat mat,ISColoring iscoloring,MatFDColoring c)
{
  Mat_MPIAIJ             *aij = (Mat_MPIAIJ*)mat->data;
  PetscErrorCode         ierr;
  PetscMPIInt            size,*ncolsonproc,*disp,nn;
  PetscInt               i,n,nrows,j,k,m,ncols,col;
  const PetscInt         *is,*A_ci,*A_cj,*B_ci,*B_cj,*rows = 0,*ltog;
  PetscInt               nis = iscoloring->n,nctot,*cols;
  PetscInt               *rowhit,M,cstart,cend,colb;
  PetscInt               *columnsforrow,l;
  IS                     *isa;
  PetscBool              done,flg;
  ISLocalToGlobalMapping map   = mat->cmap->mapping;
  PetscInt               ctype=c->ctype;
  PetscBool              new_impl=PETSC_FALSE;

  PetscFunctionBegin;
  ierr = PetscOptionsName("-new","using new impls","",&new_impl);CHKERRQ(ierr);
  if (new_impl){
    ierr =  MatFDColoringCreate_MPIAIJ_new(mat,iscoloring,c);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (ctype == IS_COLORING_GHOSTED && !map) SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_INCOMP,"When using ghosted differencing matrix must have local to global mapping provided with MatSetLocalToGlobalMapping");

  if (map) {ierr = ISLocalToGlobalMappingGetIndices(map,&ltog);CHKERRQ(ierr);}
  else     ltog = NULL;
  ierr = ISColoringGetIS(iscoloring,PETSC_IGNORE,&isa);CHKERRQ(ierr);

  M         = mat->rmap->n;
  cstart    = mat->cmap->rstart;
  cend      = mat->cmap->rend;
  c->M      = mat->rmap->N;         /* set the global rows and columns and local rows */
  c->N      = mat->cmap->N;
  c->m      = mat->rmap->n;
  c->rstart = mat->rmap->rstart;

  c->ncolors = nis;
  ierr       = PetscMalloc(nis*sizeof(PetscInt),&c->ncolumns);CHKERRQ(ierr);
  ierr       = PetscMalloc(nis*sizeof(PetscInt*),&c->columns);CHKERRQ(ierr);
  ierr       = PetscMalloc(nis*sizeof(PetscInt),&c->nrows);CHKERRQ(ierr);
  ierr       = PetscMalloc(nis*sizeof(PetscInt*),&c->rows);CHKERRQ(ierr);
  ierr       = PetscMalloc(nis*sizeof(PetscInt*),&c->columnsforrow);CHKERRQ(ierr);
  ierr       = PetscLogObjectMemory((PetscObject)c,5*nis*sizeof(PetscInt));CHKERRQ(ierr);

  /* Allow access to data structures of local part of matrix */
  if (!aij->colmap) {
    ierr = MatCreateColmap_MPIAIJ_Private(mat);CHKERRQ(ierr);
  }
  ierr = MatGetColumnIJ(aij->A,0,PETSC_FALSE,PETSC_FALSE,&ncols,&A_ci,&A_cj,&done);CHKERRQ(ierr);
  ierr = MatGetColumnIJ(aij->B,0,PETSC_FALSE,PETSC_FALSE,&ncols,&B_ci,&B_cj,&done);CHKERRQ(ierr);

  ierr = PetscMalloc((M+1)*sizeof(PetscInt),&rowhit);CHKERRQ(ierr);
  ierr = PetscMalloc((M+1)*sizeof(PetscInt),&columnsforrow);CHKERRQ(ierr);

  for (i=0; i<nis; i++) {
    ierr = ISGetLocalSize(isa[i],&n);CHKERRQ(ierr);
    ierr = ISGetIndices(isa[i],&is);CHKERRQ(ierr);

    c->ncolumns[i] = n; /* local number of columns of this color on this process */
    if (n) {
      ierr = PetscMalloc(n*sizeof(PetscInt),&c->columns[i]);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory((PetscObject)c,n*sizeof(PetscInt));CHKERRQ(ierr);
      ierr = PetscMemcpy(c->columns[i],is,n*sizeof(PetscInt));CHKERRQ(ierr);
    } else {
      c->columns[i] = 0;
    }

    if (ctype == IS_COLORING_GLOBAL) {
      /* Determine the total (parallel) number of columns of this color */
      ierr = MPI_Comm_size(PetscObjectComm((PetscObject)mat),&size);CHKERRQ(ierr);
      ierr = PetscMalloc2(size,PetscMPIInt,&ncolsonproc,size,PetscMPIInt,&disp);CHKERRQ(ierr);

      ierr  = PetscMPIIntCast(n,&nn);CHKERRQ(ierr);
      ierr  = MPI_Allgather(&nn,1,MPI_INT,ncolsonproc,1,MPI_INT,PetscObjectComm((PetscObject)mat));CHKERRQ(ierr);
      nctot = 0; for (j=0; j<size; j++) nctot += ncolsonproc[j];
      if (!nctot) {
        ierr = PetscInfo(mat,"Coloring of matrix has some unneeded colors with no corresponding rows\n");CHKERRQ(ierr);
      }

      disp[0] = 0;
      for (j=1; j<size; j++) {
        disp[j] = disp[j-1] + ncolsonproc[j-1];
      }

      /* Get complete list of columns for color on each processor */
      ierr = PetscMalloc((nctot+1)*sizeof(PetscInt),&cols);CHKERRQ(ierr);
      ierr = MPI_Allgatherv((void*)is,n,MPIU_INT,cols,ncolsonproc,disp,MPIU_INT,PetscObjectComm((PetscObject)mat));CHKERRQ(ierr);
      ierr = PetscFree2(ncolsonproc,disp);CHKERRQ(ierr);
    } else if (ctype == IS_COLORING_GHOSTED) {
      /* Determine local number of columns of this color on this process, including ghost points */
      nctot = n;
      ierr  = PetscMalloc((nctot+1)*sizeof(PetscInt),&cols);CHKERRQ(ierr);
      ierr  = PetscMemcpy(cols,is,n*sizeof(PetscInt));CHKERRQ(ierr);
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not provided for this MatFDColoring type");

    /*
       Mark all rows affect by these columns
    */
    /* Temporary option to allow for debugging/testing */
    flg  = PETSC_FALSE;
    ierr = PetscOptionsGetBool(NULL,"-matfdcoloring_slow",&flg,NULL);CHKERRQ(ierr);
    if (!flg) { /*-----------------------------------------------------------------------------*/
      /* crude, fast version */
      ierr = PetscMemzero(rowhit,M*sizeof(PetscInt));CHKERRQ(ierr);
      /* loop over columns*/
      for (j=0; j<nctot; j++) {
        if (ctype == IS_COLORING_GHOSTED) {
          col = ltog[cols[j]];
        } else {
          col = cols[j];
        }
        if (col >= cstart && col < cend) {
          /* column is in diagonal block of matrix */
          rows = A_cj + A_ci[col-cstart];
          m    = A_ci[col-cstart+1] - A_ci[col-cstart];
        } else {
#if defined(PETSC_USE_CTABLE)
          ierr = PetscTableFind(aij->colmap,col+1,&colb);CHKERRQ(ierr);
          colb--;
#else
          colb = aij->colmap[col] - 1;
#endif
          if (colb == -1) {
            m = 0;
          } else {
            rows = B_cj + B_ci[colb];
            m    = B_ci[colb+1] - B_ci[colb];
          }
        }
        /* loop over columns marking them in rowhit */
        for (k=0; k<m; k++) {
          rowhit[*rows++] = col + 1;
        }
      }

      /* count the number of hits */
      nrows = 0;
      for (j=0; j<M; j++) {
        if (rowhit[j]) nrows++;
      }
      c->nrows[i] = nrows;
      ierr        = PetscMalloc((nrows+1)*sizeof(PetscInt),&c->rows[i]);CHKERRQ(ierr);
      ierr        = PetscMalloc((nrows+1)*sizeof(PetscInt),&c->columnsforrow[i]);CHKERRQ(ierr);
      ierr        = PetscLogObjectMemory((PetscObject)c,2*(nrows+1)*sizeof(PetscInt));CHKERRQ(ierr);
      nrows       = 0;
      for (j=0; j<M; j++) {
        if (rowhit[j]) {
          c->rows[i][nrows]          = j;              /* local row index */
          c->columnsforrow[i][nrows] = rowhit[j] - 1;  /* global column index */
          nrows++;
        }
      }
    } else { /*-------------------------------------------------------------------------------*/
      /* slow version, using rowhit as a linked list */
      PetscInt currentcol,fm,mfm;
      rowhit[M] = M;
      nrows     = 0;
      /* loop over columns*/
      for (j=0; j<nctot; j++) {
        if (ctype == IS_COLORING_GHOSTED) {
          col = ltog[cols[j]];
        } else {
          col = cols[j];
        }
        if (col >= cstart && col < cend) {
          /* column is in diagonal block of matrix */
          rows = A_cj + A_ci[col-cstart];
          m    = A_ci[col-cstart+1] - A_ci[col-cstart];
        } else {
#if defined(PETSC_USE_CTABLE)
          ierr = PetscTableFind(aij->colmap,col+1,&colb);CHKERRQ(ierr);
          colb--;
#else
          colb = aij->colmap[col] - 1;
#endif
          if (colb == -1) {
            m = 0;
          } else {
            rows = B_cj + B_ci[colb];
            m    = B_ci[colb+1] - B_ci[colb];
          }
        }

        /* loop over columns marking them in rowhit */
        fm = M;    /* fm points to first entry in linked list */
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
          } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Invalid coloring of matrix detected");
        }
      }
      c->nrows[i] = nrows;

      ierr = PetscMalloc((nrows+1)*sizeof(PetscInt),&c->rows[i]);CHKERRQ(ierr);
      ierr = PetscMalloc((nrows+1)*sizeof(PetscInt),&c->columnsforrow[i]);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory((PetscObject)c,(nrows+1)*sizeof(PetscInt));CHKERRQ(ierr);
      /* now store the linked list of rows into c->rows[i] */
      nrows = 0;
      fm    = rowhit[M];
      do {
        c->rows[i][nrows]            = fm;
        c->columnsforrow[i][nrows++] = columnsforrow[fm];
        fm                           = rowhit[fm];
      } while (fm < M);
    } /* ---------------------------------------------------------------------------------------*/
    ierr = PetscFree(cols);CHKERRQ(ierr);
  }

  /* Optimize by adding the vscale, and scaleforrow[][] fields */
  /*
       vscale will contain the "diagonal" on processor scalings followed by the off processor
  */
  if (ctype == IS_COLORING_GLOBAL) {
    ierr = VecCreateGhost(PetscObjectComm((PetscObject)mat),aij->A->rmap->n,PETSC_DETERMINE,aij->B->cmap->n,aij->garray,&c->vscale);CHKERRQ(ierr);
    ierr = PetscMalloc(c->ncolors*sizeof(PetscInt*),&c->vscaleforrow);CHKERRQ(ierr);
    for (k=0; k<c->ncolors; k++) {
      ierr = PetscMalloc((c->nrows[k]+1)*sizeof(PetscInt),&c->vscaleforrow[k]);CHKERRQ(ierr);
      for (l=0; l<c->nrows[k]; l++) {
        col = c->columnsforrow[k][l];
        if (col >= cstart && col < cend) {
          /* column is in diagonal block of matrix */
          colb = col - cstart;
        } else {
          /* column  is in "off-processor" part */
#if defined(PETSC_USE_CTABLE)
          ierr = PetscTableFind(aij->colmap,col+1,&colb);CHKERRQ(ierr);
          colb--;
#else
          colb = aij->colmap[col] - 1;
#endif
          colb += cend - cstart;
        }
        c->vscaleforrow[k][l] = colb;
      }
    }
  } else if (ctype == IS_COLORING_GHOSTED) {
    /* Get gtol mapping */
    PetscInt N = mat->cmap->N,nlocal,*gtol;
    ierr = PetscMalloc((N+1)*sizeof(PetscInt),&gtol);CHKERRQ(ierr);
    for (i=0; i<N; i++) gtol[i] = -1;
    ierr = ISLocalToGlobalMappingGetSize(map,&nlocal);CHKERRQ(ierr);
    for (i=0; i<nlocal; i++) gtol[ltog[i]] = i;

    c->vscale = 0; /* will be created in MatFDColoringApply() */
    ierr      = PetscMalloc(c->ncolors*sizeof(PetscInt*),&c->vscaleforrow);CHKERRQ(ierr);
    for (k=0; k<c->ncolors; k++) {
      ierr = PetscMalloc((c->nrows[k]+1)*sizeof(PetscInt),&c->vscaleforrow[k]);CHKERRQ(ierr);
      for (l=0; l<c->nrows[k]; l++) {
        col = c->columnsforrow[k][l];      /* global column index */
        c->vscaleforrow[k][l] = gtol[col]; /* local column index */
      }
    }
    ierr = PetscFree(gtol);CHKERRQ(ierr);
  }
  ierr = ISColoringRestoreIS(iscoloring,&isa);CHKERRQ(ierr);

  ierr = PetscFree(rowhit);CHKERRQ(ierr);
  ierr = PetscFree(columnsforrow);CHKERRQ(ierr);
  ierr = MatRestoreColumnIJ(aij->A,0,PETSC_FALSE,PETSC_FALSE,&ncols,&A_ci,&A_cj,&done);CHKERRQ(ierr);
  ierr = MatRestoreColumnIJ(aij->B,0,PETSC_FALSE,PETSC_FALSE,&ncols,&B_ci,&B_cj,&done);CHKERRQ(ierr);
  if (map) {ierr = ISLocalToGlobalMappingRestoreIndices(map,&ltog);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}






