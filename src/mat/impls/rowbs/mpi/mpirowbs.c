/* $Id: mpirowbs.c,v 2.9 2001/08/10 03:30:53 bsmith Exp $*/

#include "src/mat/impls/rowbs/mpi/mpirowbs.h"
#include "src/vec/vecimpl.h"

#define CHUNCKSIZE_LOCAL   10

#undef __FUNCT__  
#define __FUNCT__ "MatFreeRowbs_Private"
static int MatFreeRowbs_Private(Mat A,int n,int *i,PetscScalar *v)
{
  int ierr;

  PetscFunctionBegin;
  if (v) {
#if defined(PETSC_USE_LOG)
    int len = -n*(sizeof(int)+sizeof(PetscScalar));
#endif
    ierr = PetscFree(v);CHKERRQ(ierr);
    PetscLogObjectMemory(A,len);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMallocRowbs_Private"
static int MatMallocRowbs_Private(Mat A,int n,int **i,PetscScalar **v)
{
  int len,ierr;

  PetscFunctionBegin;
  if (!n) {
    *i = 0; *v = 0;
  } else {
    len = n*(sizeof(int) + sizeof(PetscScalar));
    ierr = PetscMalloc(len,v);CHKERRQ(ierr);
    PetscLogObjectMemory(A,len);
    *i = (int *)(*v + n);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatScale_MPIRowbs"
int MatScale_MPIRowbs(PetscScalar *alphain,Mat inA)
{
  Mat_MPIRowbs *a = (Mat_MPIRowbs*)inA->data;
  BSspmat      *A = a->A;
  BSsprow      *vs;
  PetscScalar  *ap,alpha = *alphain;
  int          i,m = inA->m,nrow,j;

  PetscFunctionBegin;
  for (i=0; i<m; i++) {
    vs   = A->rows[i];
    nrow = vs->length;
    ap   = vs->nz;
    for (j=0; j<nrow; j++) {
      ap[j] *= alpha;
    }
  }
  PetscLogFlops(a->nz);
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "MatCreateMPIRowbs_local"
static int MatCreateMPIRowbs_local(Mat A,int nz,int *nnz)
{
  Mat_MPIRowbs *bsif = (Mat_MPIRowbs*)A->data;
  int          ierr,i,len,nzalloc = 0,m = A->m;
  BSspmat      *bsmat;
  BSsprow      *vs;

  PetscFunctionBegin;
  if (!nnz) {
    if (nz == PETSC_DEFAULT || nz == PETSC_DECIDE) nz = 5;
    if (nz <= 0)             nz = 1;
    nzalloc = 1;
    ierr = PetscMalloc((m+1)*sizeof(int),&nnz);CHKERRQ(ierr);
    for (i=0; i<m; i++) nnz[i] = nz;
    nz      = nz*m;
  } else {
    nz = 0;
    for (i=0; i<m; i++) {
      if (nnz[i] <= 0) nnz[i] = 1;
      nz += nnz[i];
    }
  }

  /* Allocate BlockSolve matrix context */
  ierr  = PetscNew(BSspmat,&bsif->A);CHKERRQ(ierr);
  bsmat = bsif->A;
  BSset_mat_icc_storage(bsmat,PETSC_FALSE);
  BSset_mat_symmetric(bsmat,PETSC_FALSE);
  len                    = m*(sizeof(BSsprow*)+ sizeof(BSsprow)) + 1;
  ierr                   = PetscMalloc(len,&bsmat->rows);CHKERRQ(ierr);
  bsmat->num_rows        = m;
  bsmat->global_num_rows = A->M;
  bsmat->map             = bsif->bsmap;
  vs                     = (BSsprow*)(bsmat->rows + m);
  for (i=0; i<m; i++) {
    bsmat->rows[i]  = vs;
    bsif->imax[i]   = nnz[i];
    vs->diag_ind    = -1;
    if (nnz[i] > 0) {
      ierr = MatMallocRowbs_Private(A,nnz[i],&(vs->col),&(vs->nz));CHKERRQ(ierr);
    } else {
      vs->col = 0; vs->nz = 0;
    }
    /* put zero on diagonal */
    /*vs->length	    = 1;
    vs->col[0]      = i + bsif->rstart;
    vs->nz[0]       = 0.0;*/
    vs->length = 0;
    vs++; 
  }
  PetscLogObjectMemory(A,sizeof(BSspmat) + len);
  bsif->nz               = 0;
  bsif->maxnz            = nz;
  bsif->sorted           = 0;
  bsif->roworiented      = PETSC_TRUE;
  bsif->nonew            = 0;
  bsif->bs_color_single  = 0;

  if (nzalloc) {ierr = PetscFree(nnz);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetValues_MPIRowbs_local"
static int MatSetValues_MPIRowbs_local(Mat AA,int m,int *im,int n,int *in,PetscScalar *v,InsertMode addv)
{
  Mat_MPIRowbs *mat = (Mat_MPIRowbs*)AA->data;
  BSspmat      *A = mat->A;
  BSsprow      *vs;
  int          *rp,k,a,b,t,ii,row,nrow,i,col,l,rmax,ierr;
  int          *imax = mat->imax,nonew = mat->nonew,sorted = mat->sorted;
  PetscScalar  *ap,value;

  PetscFunctionBegin;
  for (k=0; k<m; k++) { /* loop over added rows */
    row = im[k];
    if (row < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Negative row");
    if (row >= AA->m) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Row too large");
    vs   = A->rows[row];
    ap   = vs->nz; rp = vs->col;
    rmax = imax[row]; nrow = vs->length;
    a    = 0;
    for (l=0; l<n; l++) { /* loop over added columns */
      if (in[l] < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Negative col");
      if (in[l] >= AA->N) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Column too large");
      col = in[l]; value = *v++;
      if (!sorted) a = 0; b = nrow;
      while (b-a > 5) {
        t = (b+a)/2;
        if (rp[t] > col) b = t;
        else             a = t;
      }
      for (i=a; i<b; i++) {
        if (rp[i] > col) break;
        if (rp[i] == col) {
          if (addv == ADD_VALUES) ap[i] += value;
          else                    ap[i] = value;
          goto noinsert;
        }
      }
      if (nonew) goto noinsert;
      if (nrow >= rmax) {
        /* there is no extra room in row, therefore enlarge */
        int    *itemp,*iout,*iin = vs->col;
        PetscScalar *vout,*vin = vs->nz,*vtemp;

        /* malloc new storage space */
        imax[row] += CHUNCKSIZE_LOCAL;
        ierr = MatMallocRowbs_Private(AA,imax[row],&itemp,&vtemp);CHKERRQ(ierr);
        vout = vtemp; iout = itemp;
        for (ii=0; ii<i; ii++) {
          vout[ii] = vin[ii];
          iout[ii] = iin[ii];
        }
        vout[i] = value;
        iout[i] = col;
        for (ii=i+1; ii<=nrow; ii++) {
          vout[ii] = vin[ii-1];
          iout[ii] = iin[ii-1];
        }
        /* free old row storage */
        if (rmax > 0) {
          ierr = MatFreeRowbs_Private(AA,rmax,vs->col,vs->nz);CHKERRQ(ierr);
        }
        vs->col           =  iout; vs->nz = vout;
        rmax              =  imax[row];
        mat->maxnz        += CHUNCKSIZE_LOCAL;
        mat->reallocs++;
      } else {
        /* shift higher columns over to make room for newie */
        for (ii=nrow-1; ii>=i; ii--) {
          rp[ii+1] = rp[ii];
          ap[ii+1] = ap[ii];
        }
        rp[i] = col;
        ap[i] = value;
      }
      nrow++;
      mat->nz++;
      AA->same_nonzero = PETSC_FALSE;
      noinsert:;
      a = i + 1;
    }
    vs->length = nrow;
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatAssemblyBegin_MPIRowbs_local"
static int MatAssemblyBegin_MPIRowbs_local(Mat A,MatAssemblyType mode)
{ 
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatAssemblyEnd_MPIRowbs_local"
static int MatAssemblyEnd_MPIRowbs_local(Mat AA,MatAssemblyType mode)
{
  Mat_MPIRowbs *a = (Mat_MPIRowbs*)AA->data;
  BSspmat      *A = a->A;
  BSsprow      *vs;
  int          i,j,rstart = a->rstart;

  PetscFunctionBegin;
  if (mode == MAT_FLUSH_ASSEMBLY) PetscFunctionReturn(0);

  /* Mark location of diagonal */
  for (i=0; i<AA->m; i++) {
    vs = A->rows[i];
    for (j=0; j<vs->length; j++) {
      if (vs->col[j] == i + rstart) {
        vs->diag_ind = j;
        break;
      }
    }
    if (vs->diag_ind == -1) { 
      SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"no diagonal entry");
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatZeroRows_MPIRowbs_local"
static int MatZeroRows_MPIRowbs_local(Mat A,IS is,PetscScalar *diag)
{
  Mat_MPIRowbs *a = (Mat_MPIRowbs*)A->data;
  BSspmat      *l = a->A;
  int          i,ierr,N,*rz,m = A->m - 1;

  PetscFunctionBegin;
  ierr = ISGetLocalSize(is,&N);CHKERRQ(ierr);
  ierr = ISGetIndices(is,&rz);CHKERRQ(ierr);
  if (diag) {
    for (i=0; i<N; i++) {
      if (rz[i] < 0 || rz[i] > m) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Out of range");
      if (l->rows[rz[i]]->length > 0) { /* in case row was completely empty */
        l->rows[rz[i]]->length = 1;
        l->rows[rz[i]]->nz[0]  = *diag;
        l->rows[rz[i]]->col[0] = a->rstart + rz[i];
      } else {
        ierr = MatSetValues(A,1,&rz[i],1,&rz[i],diag,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  } else {
    for (i=0; i<N; i++) {
      if (rz[i] < 0 || rz[i] > m) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Out of range");
      l->rows[rz[i]]->length = 0;
    }
  }
  A->same_nonzero = PETSC_FALSE;
  ISRestoreIndices(is,&rz);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatNorm_MPIRowbs_local"
static int MatNorm_MPIRowbs_local(Mat A,NormType type,PetscReal *norm)
{
  Mat_MPIRowbs *mat = (Mat_MPIRowbs*)A->data;
  BSsprow      *vs,**rs;
  PetscScalar  *xv;
  PetscReal    sum = 0.0;
  int          *xi,nz,i,j,ierr;

  PetscFunctionBegin;
  rs = mat->A->rows;
  if (type == NORM_FROBENIUS) {
    for (i=0; i<A->m; i++) {
      vs = *rs++;
      nz = vs->length;
      xv = vs->nz;
      while (nz--) {
#if defined(PETSC_USE_COMPLEX)
        sum += PetscRealPart(PetscConj(*xv)*(*xv)); xv++;
#else
        sum += (*xv)*(*xv); xv++;
#endif
      }
    }
    *norm = sqrt(sum);
  } else if (type == NORM_1) { /* max column norm */
    PetscReal *tmp;
    ierr  = PetscMalloc(A->n*sizeof(PetscReal),&tmp);CHKERRQ(ierr);
    ierr  = PetscMemzero(tmp,A->n*sizeof(PetscReal));CHKERRQ(ierr);
    *norm = 0.0;
    for (i=0; i<A->m; i++) {
      vs = *rs++;
      nz = vs->length;
      xi = vs->col;
      xv = vs->nz;
      while (nz--) {
        tmp[*xi] += PetscAbsScalar(*xv); 
        xi++; xv++;
      }
    }
    for (j=0; j<A->n; j++) {
      if (tmp[j] > *norm) *norm = tmp[j];
    }
    ierr = PetscFree(tmp);CHKERRQ(ierr);
  } else if (type == NORM_INFINITY) { /* max row norm */
    *norm = 0.0;
    for (i=0; i<A->m; i++) {
      vs = *rs++;
      nz = vs->length;
      xv = vs->nz;
      sum = 0.0;
      while (nz--) {
        sum += PetscAbsScalar(*xv); xv++;
      }
      if (sum > *norm) *norm = sum;
    }
  } else {
    SETERRQ(PETSC_ERR_SUP,"No support for the two norm");
  }
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------- */

#undef __FUNCT__  
#define __FUNCT__ "MatSetValues_MPIRowbs"
int MatSetValues_MPIRowbs(Mat mat,int m,int *im,int n,int *in,PetscScalar *v,InsertMode av)
{
  Mat_MPIRowbs *a = (Mat_MPIRowbs*)mat->data;
  int          ierr,i,j,row,col,rstart = a->rstart,rend = a->rend;
  PetscTruth   roworiented = a->roworiented;

  PetscFunctionBegin;
  /* Note:  There's no need to "unscale" the matrix, since scaling is
     confined to a->pA, and we're working with a->A here */
  for (i=0; i<m; i++) {
    if (im[i] < 0) continue;
    if (im[i] >= mat->M) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Row too large");
    if (im[i] >= rstart && im[i] < rend) {
      row = im[i] - rstart;
      for (j=0; j<n; j++) {
        if (in[j] < 0) continue;
        if (in[j] >= mat->N) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Column too large");
        if (in[j] >= 0 && in[j] < mat->N){
          col = in[j];
          if (roworiented) {
            ierr = MatSetValues_MPIRowbs_local(mat,1,&row,1,&col,v+i*n+j,av);CHKERRQ(ierr);
          } else {
            ierr = MatSetValues_MPIRowbs_local(mat,1,&row,1,&col,v+i+j*m,av);CHKERRQ(ierr);
          }
        } else {SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Invalid column");}
      }
    } else {
      if (!a->donotstash) {
        if (roworiented) {
          ierr = MatStashValuesRow_Private(&mat->stash,im[i],n,in,v+i*n);CHKERRQ(ierr);
        } else {
          ierr = MatStashValuesCol_Private(&mat->stash,im[i],n,in,v+i,m);CHKERRQ(ierr);
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatAssemblyBegin_MPIRowbs"
int MatAssemblyBegin_MPIRowbs(Mat mat,MatAssemblyType mode)
{ 
  Mat_MPIRowbs  *a = (Mat_MPIRowbs*)mat->data;
  MPI_Comm      comm = mat->comm;
  int           ierr,nstash,reallocs;
  InsertMode    addv;

  PetscFunctionBegin;
  /* Note:  There's no need to "unscale" the matrix, since scaling is
            confined to a->pA, and we're working with a->A here */

  /* make sure all processors are either in INSERTMODE or ADDMODE */
  ierr = MPI_Allreduce(&mat->insertmode,&addv,1,MPI_INT,MPI_BOR,comm);CHKERRQ(ierr);
  if (addv == (ADD_VALUES|INSERT_VALUES)) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Some procs inserted; others added");
  }
  mat->insertmode = addv; /* in case this processor had no cache */

  ierr = MatStashScatterBegin_Private(&mat->stash,a->rowners);CHKERRQ(ierr);
  ierr = MatStashGetInfo_Private(&mat->stash,&nstash,&reallocs);CHKERRQ(ierr);
  PetscLogInfo(0,"MatAssemblyBegin_MPIRowbs:Block-Stash has %d entries, uses %d mallocs.\n",nstash,reallocs);
  PetscFunctionReturn(0);
}

#include "petscviewer.h"

#undef __FUNCT__  
#define __FUNCT__ "MatView_MPIRowbs_ASCII"
static int MatView_MPIRowbs_ASCII(Mat mat,PetscViewer viewer)
{
  Mat_MPIRowbs      *a = (Mat_MPIRowbs*)mat->data;
  int               ierr,i,j;
  PetscTruth        isascii;
  BSspmat           *A = a->A;
  BSsprow           **rs = A->rows;
  PetscViewerFormat format;

  PetscFunctionBegin;
  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);

  if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
    int ind_l,ind_g,clq_l,clq_g,color;
    ind_l = BSlocal_num_inodes(a->pA);CHKERRBS(0);
    ind_g = BSglobal_num_inodes(a->pA);CHKERRBS(0);
    clq_l = BSlocal_num_cliques(a->pA);CHKERRBS(0);
    clq_g = BSglobal_num_cliques(a->pA);CHKERRBS(0);
    color = BSnum_colors(a->pA);CHKERRBS(0);
    ierr = PetscViewerASCIIPrintf(viewer,"  %d global inode(s), %d global clique(s), %d color(s)\n",ind_g,clq_g,color);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"    [%d] %d local inode(s), %d local clique(s)\n",a->rank,ind_l,clq_l);
  } else  if (format == PETSC_VIEWER_ASCII_COMMON) {
    for (i=0; i<A->num_rows; i++) {
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"row %d:",i+a->rstart);CHKERRQ(ierr);
      for (j=0; j<rs[i]->length; j++) {
        if (rs[i]->nz[j]) {ierr = PetscViewerASCIISynchronizedPrintf(viewer," %d %g ",rs[i]->col[j],rs[i]->nz[j]);CHKERRQ(ierr);}
      }
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"\n");CHKERRQ(ierr);
    }
  } else if (format == PETSC_VIEWER_ASCII_MATLAB) {
    SETERRQ(PETSC_ERR_SUP,"Matlab format not supported");
  } else {
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_NO);CHKERRQ(ierr);
    for (i=0; i<A->num_rows; i++) {
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"row %d:",i+a->rstart);CHKERRQ(ierr);
      for (j=0; j<rs[i]->length; j++) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer," %d %g ",rs[i]->col[j],rs[i]->nz[j]);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_YES);CHKERRQ(ierr);
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatView_MPIRowbs_Binary"
static int MatView_MPIRowbs_Binary(Mat mat,PetscViewer viewer)
{
  Mat_MPIRowbs *a = (Mat_MPIRowbs*)mat->data;
  int          ierr,i,M,m,rank,size,*sbuff,*rowlengths;
  int          *recvcts,*recvdisp,fd,*cols,maxnz,nz,j;
  BSspmat      *A = a->A;
  BSsprow      **rs = A->rows;
  MPI_Comm     comm = mat->comm;
  MPI_Status   status;
  PetscScalar  *vals;
  MatInfo      info;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  M = mat->M; m = mat->m;
  /* First gather together on the first processor the lengths of 
     each row, and write them out to the file */
  ierr = PetscMalloc(m*sizeof(int),&sbuff);CHKERRQ(ierr);
  for (i=0; i<A->num_rows; i++) {
    sbuff[i] = rs[i]->length;
  }
  ierr = MatGetInfo(mat,MAT_GLOBAL_SUM,&info);CHKERRQ(ierr);
  if (!rank) {
    ierr = PetscViewerBinaryGetDescriptor(viewer,&fd);CHKERRQ(ierr);
    ierr = PetscMalloc((4+M)*sizeof(int),&rowlengths);CHKERRQ(ierr);
    ierr = PetscMalloc(size*sizeof(int),&recvcts);CHKERRQ(ierr);
    recvdisp = a->rowners;
    for (i=0; i<size; i++) {
      recvcts[i] = recvdisp[i+1] - recvdisp[i];
    }
    /* first four elements of rowlength are the header */
    rowlengths[0] = mat->cookie;
    rowlengths[1] = mat->M;
    rowlengths[2] = mat->N;
    rowlengths[3] = (int)info.nz_used;
    ierr = MPI_Gatherv(sbuff,m,MPI_INT,rowlengths+4,recvcts,recvdisp,MPI_INT,0,comm);CHKERRQ(ierr);
    ierr = PetscFree(sbuff);CHKERRQ(ierr);
    ierr = PetscBinaryWrite(fd,rowlengths,4+M,PETSC_INT,0);CHKERRQ(ierr);
    /* count the number of nonzeros on each processor */
    ierr = PetscMemzero(recvcts,size*sizeof(int));CHKERRQ(ierr);
    for (i=0; i<size; i++) {
      for (j=recvdisp[i]; j<recvdisp[i+1]; j++) {
        recvcts[i] += rowlengths[j+3];
      }
    }
    /* allocate buffer long enough to hold largest one */
    maxnz = 0;
    for (i=0; i<size; i++) {
      maxnz = PetscMax(maxnz,recvcts[i]);
    }
    ierr = PetscFree(rowlengths);CHKERRQ(ierr);
    ierr = PetscFree(recvcts);CHKERRQ(ierr);
    ierr = PetscMalloc(maxnz*sizeof(int),&cols);CHKERRQ(ierr);

    /* binary store column indices for 0th processor */
    nz = 0;
    for (i=0; i<A->num_rows; i++) {
      for (j=0; j<rs[i]->length; j++) {
        cols[nz++] = rs[i]->col[j];
      }
    }
    ierr = PetscBinaryWrite(fd,cols,nz,PETSC_INT,0);CHKERRQ(ierr);

    /* receive and store column indices for all other processors */
    for (i=1; i<size; i++) {
      /* should tell processor that I am now ready and to begin the send */
      ierr = MPI_Recv(cols,maxnz,MPI_INT,i,mat->tag,comm,&status);CHKERRQ(ierr);
      ierr = MPI_Get_count(&status,MPI_INT,&nz);CHKERRQ(ierr);
      ierr = PetscBinaryWrite(fd,cols,nz,PETSC_INT,0);CHKERRQ(ierr);
    }
    ierr = PetscFree(cols);CHKERRQ(ierr);
    ierr = PetscMalloc(maxnz*sizeof(PetscScalar),&vals);CHKERRQ(ierr);

    /* binary store values for 0th processor */
    nz = 0;
    for (i=0; i<A->num_rows; i++) {
      for (j=0; j<rs[i]->length; j++) {
        vals[nz++] = rs[i]->nz[j];
      }
    }
    ierr = PetscBinaryWrite(fd,vals,nz,PETSC_SCALAR,0);CHKERRQ(ierr);

    /* receive and store nonzeros for all other processors */
    for (i=1; i<size; i++) {
      /* should tell processor that I am now ready and to begin the send */
      ierr = MPI_Recv(vals,maxnz,MPIU_SCALAR,i,mat->tag,comm,&status);CHKERRQ(ierr);
      ierr = MPI_Get_count(&status,MPIU_SCALAR,&nz);CHKERRQ(ierr);
      ierr = PetscBinaryWrite(fd,vals,nz,PETSC_SCALAR,0);CHKERRQ(ierr);
    }
    ierr = PetscFree(vals);CHKERRQ(ierr);
  } else {
    ierr = MPI_Gatherv(sbuff,m,MPI_INT,0,0,0,MPI_INT,0,comm);CHKERRQ(ierr);
    ierr = PetscFree(sbuff);CHKERRQ(ierr);

    /* count local nonzeros */
    nz = 0;
    for (i=0; i<A->num_rows; i++) {
      for (j=0; j<rs[i]->length; j++) {
        nz++;
      }
    }
    /* copy into buffer column indices */
    ierr = PetscMalloc(nz*sizeof(int),&cols);CHKERRQ(ierr);
    nz = 0;
    for (i=0; i<A->num_rows; i++) {
      for (j=0; j<rs[i]->length; j++) {
        cols[nz++] = rs[i]->col[j];
      }
    }
    /* send */  /* should wait until processor zero tells me to go */
    ierr = MPI_Send(cols,nz,MPI_INT,0,mat->tag,comm);CHKERRQ(ierr);
    ierr = PetscFree(cols);CHKERRQ(ierr);

    /* copy into buffer column values */
    ierr = PetscMalloc(nz*sizeof(PetscScalar),&vals);CHKERRQ(ierr);
    nz   = 0;
    for (i=0; i<A->num_rows; i++) {
      for (j=0; j<rs[i]->length; j++) {
        vals[nz++] = rs[i]->nz[j];
      }
    }
    /* send */  /* should wait until processor zero tells me to go */
    ierr = MPI_Send(vals,nz,MPIU_SCALAR,0,mat->tag,comm);CHKERRQ(ierr);
    ierr = PetscFree(vals);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatView_MPIRowbs"
int MatView_MPIRowbs(Mat mat,PetscViewer viewer)
{
  Mat_MPIRowbs *bsif = (Mat_MPIRowbs*)mat->data;
  int          ierr;
  PetscTruth   isascii,isbinary;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_BINARY,&isbinary);CHKERRQ(ierr);
  if (!bsif->blocksolveassembly) {
    ierr = MatAssemblyEnd_MPIRowbs_ForBlockSolve(mat);CHKERRQ(ierr);
  }
  if (isascii) {
    ierr = MatView_MPIRowbs_ASCII(mat,viewer);CHKERRQ(ierr);
  } else if (isbinary) {
    ierr = MatView_MPIRowbs_Binary(mat,viewer);CHKERRQ(ierr);
  } else {
    SETERRQ1(1,"Viewer type %s not supported by MPIRowbs matrices",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}
  
#undef __FUNCT__  
#define __FUNCT__ "MatAssemblyEnd_MPIRowbs_MakeSymmetric"
static int MatAssemblyEnd_MPIRowbs_MakeSymmetric(Mat mat)
{
  Mat_MPIRowbs *a = (Mat_MPIRowbs*)mat->data;
  BSspmat      *A = a->A;
  BSsprow      *vs;
  int          size,rank,M,rstart,tag,i,j,*rtable,*w1,*w3,*w4,len,proc,nrqs;
  int          msz,*pa,bsz,nrqr,**rbuf1,**sbuf1,**ptr,*tmp,*ctr,col,index,row;
  int          ctr_j,*sbuf1_j,k,ierr;
  PetscScalar  val=0.0;
  MPI_Comm     comm;
  MPI_Request  *s_waits1,*r_waits1;
  MPI_Status   *s_status,*r_status;

  PetscFunctionBegin;
  comm   = mat->comm;
  tag    = mat->tag;
  size   = a->size;
  rank   = a->rank;
  M      = mat->M;
  rstart = a->rstart;

  ierr = PetscMalloc(M*sizeof(int),&rtable);CHKERRQ(ierr);
  /* Create hash table for the mapping :row -> proc */
  for (i=0,j=0; i<size; i++) {
    len = a->rowners[i+1];  
    for (; j<len; j++) {
      rtable[j] = i;
    }
  }

  /* Evaluate communication - mesg to whom, length of mesg, and buffer space
     required. Based on this, buffers are allocated, and data copied into them. */
  ierr = PetscMalloc(size*4*sizeof(int),&w1);CHKERRQ(ierr);/*  mesg size */
  w3   = w1 + 2*size;       /* no of IS that needs to be sent to proc i */
  w4   = w3 + size;       /* temp work space used in determining w1,  w3 */
  ierr = PetscMemzero(w1,size*3*sizeof(int));CHKERRQ(ierr); /* initialize work vector */

  for (i=0;  i<mat->m; i++) { 
    ierr = PetscMemzero(w4,size*sizeof(int));CHKERRQ(ierr); /* initialize work vector */
    vs = A->rows[i];
    for (j=0; j<vs->length; j++) {
      proc = rtable[vs->col[j]];
      w4[proc]++;
    }
    for (j=0; j<size; j++) { 
      if (w4[j]) { w1[2*j] += w4[j]; w3[j]++;} 
    }
  }
  
  nrqs       = 0;              /* number of outgoing messages */
  msz        = 0;              /* total mesg length (for all proc */
  w1[2*rank] = 0;              /* no mesg sent to itself */
  w3[rank]   = 0;
  for (i=0; i<size; i++) {
    if (w1[2*i])  {w1[2*i+1] = 1; nrqs++;} /* there exists a message to proc i */
  }
  /* pa - is list of processors to communicate with */
  ierr = PetscMalloc((nrqs+1)*sizeof(int),&pa);CHKERRQ(ierr);
  for (i=0,j=0; i<size; i++) {
    if (w1[2*i]) {pa[j] = i; j++;}
  } 

  /* Each message would have a header = 1 + 2*(no of ROWS) + data */
  for (i=0; i<nrqs; i++) {
    j       = pa[i];
    w1[2*j] += w1[2*j+1] + 2*w3[j];   
    msz     += w1[2*j];  
  }
  
  /* Do a global reduction to determine how many messages to expect */
  ierr = PetscMaxSum(comm,w1,&bsz,&nrqr);CHKERRQ(ierr);

  /* Allocate memory for recv buffers . Prob none if nrqr = 0 ???? */
  len      = (nrqr+1)*sizeof(int*) + nrqr*bsz*sizeof(int);
  ierr     = PetscMalloc(len,&rbuf1);CHKERRQ(ierr);
  rbuf1[0] = (int*)(rbuf1 + nrqr);
  for (i=1; i<nrqr; ++i) rbuf1[i] = rbuf1[i-1] + bsz;

  /* Post the receives */
  ierr = PetscMalloc((nrqr+1)*sizeof(MPI_Request),&r_waits1);CHKERRQ(ierr);
  for (i=0; i<nrqr; ++i){
    ierr = MPI_Irecv(rbuf1[i],bsz,MPI_INT,MPI_ANY_SOURCE,tag,comm,r_waits1+i);CHKERRQ(ierr);
  }
  
  /* Allocate Memory for outgoing messages */
  len   = 2*size*sizeof(int*) + (size+msz)*sizeof(int);
  ierr  = PetscMalloc(len,&sbuf1);CHKERRQ(ierr);
  ptr   = sbuf1 + size;     /* Pointers to the data in outgoing buffers */
  ierr  = PetscMemzero(sbuf1,2*size*sizeof(int*));CHKERRQ(ierr);
  tmp   = (int*)(sbuf1 + 2*size);
  ctr   = tmp + msz;

  {
    int *iptr = tmp,ict  = 0;
    for (i=0; i<nrqs; i++) {
      j        = pa[i];
      iptr    += ict;
      sbuf1[j] = iptr;
      ict      = w1[2*j];
    }
  }

  /* Form the outgoing messages */
  /* Clean up the header space */
  for (i=0; i<nrqs; i++) {
    j           = pa[i];
    sbuf1[j][0] = 0;
    ierr        = PetscMemzero(sbuf1[j]+1,2*w3[j]*sizeof(int));CHKERRQ(ierr);
    ptr[j]      = sbuf1[j] + 2*w3[j] + 1;
  }

  /* Parse the matrix and copy the data into sbuf1 */
  for (i=0; i<mat->m; i++) {
    ierr = PetscMemzero(ctr,size*sizeof(int));CHKERRQ(ierr);
    vs = A->rows[i];
    for (j=0; j<vs->length; j++) {
      col  = vs->col[j];
      proc = rtable[col];
      if (proc != rank) { /* copy to the outgoing buffer */
        ctr[proc]++;
          *ptr[proc] = col;
          ptr[proc]++;
      } else {
        row = col - rstart;
        col = i + rstart;
        ierr = MatSetValues_MPIRowbs_local(mat,1,&row,1,&col,&val,ADD_VALUES);CHKERRQ(ierr);
      }
    }
    /* Update the headers for the current row */
    for (j=0; j<size; j++) { /* Can Optimise this loop by using pa[] */
      if ((ctr_j = ctr[j])) {
        sbuf1_j        = sbuf1[j];
        k               = ++sbuf1_j[0];
        sbuf1_j[2*k]   = ctr_j;
        sbuf1_j[2*k-1] = i + rstart;
      }
    }
  }
   /* Check Validity of the outgoing messages */
  {
    int sum;
    for (i=0 ; i<nrqs ; i++) {
      j = pa[i];
      if (w3[j] != sbuf1[j][0]) {SETERRQ(PETSC_ERR_PLIB,"Blew it! Header[1] mismatch!\n"); }
    }

    for (i=0 ; i<nrqs ; i++) {
      j = pa[i];
      sum = 1;
      for (k = 1; k <= w3[j]; k++) sum += sbuf1[j][2*k]+2;
      if (sum != w1[2*j]) { SETERRQ(PETSC_ERR_PLIB,"Blew it! Header[2-n] mismatch!\n"); }
    }
  }
 
  /* Now post the sends */
  ierr = PetscMalloc((nrqs+1)*sizeof(MPI_Request),&s_waits1);CHKERRQ(ierr);
  for (i=0; i<nrqs; ++i) {
    j    = pa[i];
    ierr = MPI_Isend(sbuf1[j],w1[2*j],MPI_INT,j,tag,comm,s_waits1+i);CHKERRQ(ierr);
  }
   
  /* Receive messages*/
  ierr = PetscMalloc((nrqr+1)*sizeof(MPI_Status),&r_status);CHKERRQ(ierr);
  for (i=0; i<nrqr; ++i) {
    ierr = MPI_Waitany(nrqr,r_waits1,&index,r_status+i);CHKERRQ(ierr);
    /* Process the Message */
    {
      int    *rbuf1_i,n_row,ct1;

      rbuf1_i = rbuf1[index];
      n_row   = rbuf1_i[0];
      ct1     = 2*n_row+1;
      val     = 0.0;
      /* Optimise this later */
      for (j=1; j<=n_row; j++) {
        col = rbuf1_i[2*j-1];
        for (k=0; k<rbuf1_i[2*j]; k++,ct1++) {
          row = rbuf1_i[ct1] - rstart;
          ierr = MatSetValues_MPIRowbs_local(mat,1,&row,1,&col,&val,ADD_VALUES);CHKERRQ(ierr);
        }
      }
    }
  }

  ierr = PetscMalloc((nrqs+1)*sizeof(MPI_Status),&s_status);CHKERRQ(ierr);
  ierr = MPI_Waitall(nrqs,s_waits1,s_status);CHKERRQ(ierr);

  ierr = PetscFree(rtable);CHKERRQ(ierr);
  ierr = PetscFree(w1);CHKERRQ(ierr);
  ierr = PetscFree(pa);CHKERRQ(ierr);
  ierr = PetscFree(rbuf1);CHKERRQ(ierr);
  ierr = PetscFree(sbuf1);CHKERRQ(ierr);
  ierr = PetscFree(r_waits1);CHKERRQ(ierr);
  ierr = PetscFree(s_waits1);CHKERRQ(ierr);
  ierr = PetscFree(r_status);CHKERRQ(ierr);
  ierr = PetscFree(s_status);CHKERRQ(ierr);
  PetscFunctionReturn(0);    
}

/*
     This does the BlockSolve portion of the matrix assembly.
   It is provided in a seperate routine so that users can
   operate on the matrix (using MatScale(), MatShift() etc.) after 
   the matrix has been assembled but before BlockSolve has sucked it
   in and devoured it.
*/
#undef __FUNCT__  
#define __FUNCT__ "MatAssemblyEnd_MPIRowbs_ForBlockSolve"
int MatAssemblyEnd_MPIRowbs_ForBlockSolve(Mat mat)
{ 
  Mat_MPIRowbs *a = (Mat_MPIRowbs*)mat->data;
  int          ierr,ldim,low,high,i;
  PetscScalar  *diag;

  PetscFunctionBegin;
  if ((mat->was_assembled) && (!mat->same_nonzero)) {  /* Free the old info */
    if (a->pA)       {BSfree_par_mat(a->pA);CHKERRBS(0);}
    if (a->comm_pA)  {BSfree_comm(a->comm_pA);CHKERRBS(0);} 
  }

  if ((!mat->same_nonzero) || (!mat->was_assembled)) {
    /* Indicates bypassing cliques in coloring */
    if (a->bs_color_single) {
      BSctx_set_si(a->procinfo,100);
    }
    /* Form permuted matrix for efficient parallel execution */
    a->pA = BSmain_perm(a->procinfo,a->A);CHKERRBS(0);
    /* Set up the communication */
    a->comm_pA = BSsetup_forward(a->pA,a->procinfo);CHKERRBS(0);
  } else {
    /* Repermute the matrix */
    BSmain_reperm(a->procinfo,a->A,a->pA);CHKERRBS(0);
  }

  /* Symmetrically scale the matrix by the diagonal */
  BSscale_diag(a->pA,a->pA->diag,a->procinfo);CHKERRBS(0);

  /* Store inverse of square root of permuted diagonal scaling matrix */
  ierr = VecGetLocalSize(a->diag,&ldim);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(a->diag,&low,&high);CHKERRQ(ierr);
  ierr = VecGetArray(a->diag,&diag);CHKERRQ(ierr);
  for (i=0; i<ldim; i++) {
    if (a->pA->scale_diag[i] != 0.0) {
      diag[i] = 1.0/sqrt(PetscAbsScalar(a->pA->scale_diag[i]));
    } else {
      diag[i] = 1.0;
    }   
  }
  ierr = VecRestoreArray(a->diag,&diag);CHKERRQ(ierr);
  a->blocksolveassembly = 1;
  mat->was_assembled    = PETSC_TRUE;
  mat->same_nonzero     = PETSC_TRUE;
  PetscLogInfo(mat,"MatAssemblyEnd_MPIRowbs_ForBlockSolve:Completed BlockSolve95 matrix assembly\n");
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatAssemblyEnd_MPIRowbs"
int MatAssemblyEnd_MPIRowbs(Mat mat,MatAssemblyType mode)
{ 
  Mat_MPIRowbs *a = (Mat_MPIRowbs*)mat->data;
  int          i,n,row,col,*rows,*cols,ierr,rstart,nzcount,flg,j,ncols;
  PetscScalar  *vals,val;
  InsertMode   addv = mat->insertmode;

  PetscFunctionBegin;
  while (1) {
    ierr = MatStashScatterGetMesg_Private(&mat->stash,&n,&rows,&cols,&vals,&flg);CHKERRQ(ierr);
    if (!flg) break;
    
    for (i=0; i<n;) {
      /* Now identify the consecutive vals belonging to the same row */
      for (j=i,rstart=rows[j]; j<n; j++) { if (rows[j] != rstart) break; }
      if (j < n) ncols = j-i;
      else       ncols = n-i;
      /* Now assemble all these values with a single function call */
      ierr = MatSetValues_MPIRowbs(mat,1,rows+i,ncols,cols+i,vals+i,addv);CHKERRQ(ierr);
      i = j;
    }
  }
  ierr = MatStashScatterEnd_Private(&mat->stash);CHKERRQ(ierr);

  rstart = a->rstart;
  nzcount = a->nz; /* This is the number of nonzeros entered by the user */
  /* BlockSolve requires that the matrix is structurally symmetric */
  if (mode == MAT_FINAL_ASSEMBLY && !mat->structurally_symmetric) {
    ierr = MatAssemblyEnd_MPIRowbs_MakeSymmetric(mat);CHKERRQ(ierr);
  }
  
  /* BlockSolve requires that all the diagonal elements are set */
  val  = 0.0;
  for (i=0; i<mat->m; i++) {
    row = i; col = i + rstart;
    ierr = MatSetValues_MPIRowbs_local(mat,1,&row,1,&col,&val,ADD_VALUES);CHKERRQ(ierr);
  }
  
  ierr = MatAssemblyBegin_MPIRowbs_local(mat,mode);CHKERRQ(ierr);
  ierr = MatAssemblyEnd_MPIRowbs_local(mat,mode);CHKERRQ(ierr);
  
  a->blocksolveassembly = 0;
  PetscLogInfo(mat,"MatAssemblyEnd_MPIRowbs:Matrix size: %d X %d; storage space: %d unneeded,%d used\n",mat->m,mat->n,a->maxnz-a->nz,a->nz);
  PetscLogInfo(mat,"MatAssemblyEnd_MPIRowbs: User entered %d nonzeros, PETSc added %d\n",nzcount,a->nz-nzcount);    
  PetscLogInfo(mat,"MatAssemblyEnd_MPIRowbs:Number of mallocs during MatSetValues is %d\n",a->reallocs);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatZeroEntries_MPIRowbs"
int MatZeroEntries_MPIRowbs(Mat mat)
{
  Mat_MPIRowbs *l = (Mat_MPIRowbs*)mat->data;
  BSspmat      *A = l->A;
  BSsprow      *vs;
  int          i,j;

  PetscFunctionBegin;
  for (i=0; i <mat->m; i++) {
    vs = A->rows[i];
    for (j=0; j< vs->length; j++) vs->nz[j] = 0.0;
  }
  PetscFunctionReturn(0);
}

/* the code does not do the diagonal entries correctly unless the 
   matrix is square and the column and row owerships are identical.
   This is a BUG.
*/

#undef __FUNCT__  
#define __FUNCT__ "MatZeroRows_MPIRowbs"
int MatZeroRows_MPIRowbs(Mat A,IS is,PetscScalar *diag)
{
  Mat_MPIRowbs   *l = (Mat_MPIRowbs*)A->data;
  int            i,ierr,N,*rows,*owners = l->rowners,size = l->size;
  int            *nprocs,j,idx,nsends;
  int            nmax,*svalues,*starts,*owner,nrecvs,rank = l->rank;
  int            *rvalues,tag = A->tag,count,base,slen,n,*source;
  int            *lens,imdex,*lrows,*values;
  MPI_Comm       comm = A->comm;
  MPI_Request    *send_waits,*recv_waits;
  MPI_Status     recv_status,*send_status;
  IS             istmp;
  PetscTruth     found;

  PetscFunctionBegin;
  ierr = ISGetLocalSize(is,&N);CHKERRQ(ierr);
  ierr = ISGetIndices(is,&rows);CHKERRQ(ierr);

  /*  first count number of contributors to each processor */
  ierr   = PetscMalloc(2*size*sizeof(int),&nprocs);CHKERRQ(ierr);
  ierr   = PetscMemzero(nprocs,2*size*sizeof(int));CHKERRQ(ierr);
  ierr   = PetscMalloc((N+1)*sizeof(int),&owner);CHKERRQ(ierr); /* see note*/
  for (i=0; i<N; i++) {
    idx = rows[i];
    found = PETSC_FALSE;
    for (j=0; j<size; j++) {
      if (idx >= owners[j] && idx < owners[j+1]) {
        nprocs[2*j]++; nprocs[2*j] = 1; owner[i] = j; found = PETSC_TRUE; break;
      }
    }
    if (!found) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Row out of range");
  }
  nsends = 0;  for (i=0; i<size; i++) {nsends += nprocs[2*i+1];} 

  /* inform other processors of number of messages and max length*/
  ierr = PetscMaxSum(comm,nprocs,&nmax,&nrecvs);CHKERRQ(ierr);

  /* post receives:   */
  ierr = PetscMalloc((nrecvs+1)*(nmax+1)*sizeof(int),&rvalues);CHKERRQ(ierr);
  ierr = PetscMalloc((nrecvs+1)*sizeof(MPI_Request),&recv_waits);CHKERRQ(ierr);
  for (i=0; i<nrecvs; i++) {
    ierr = MPI_Irecv(rvalues+nmax*i,nmax,MPI_INT,MPI_ANY_SOURCE,tag,comm,recv_waits+i);CHKERRQ(ierr);
  }

  /* do sends:
      1) starts[i] gives the starting index in svalues for stuff going to 
         the ith processor
  */
  ierr = PetscMalloc((N+1)*sizeof(int),&svalues);CHKERRQ(ierr);
  ierr = PetscMalloc((nsends+1)*sizeof(MPI_Request),&send_waits);CHKERRQ(ierr);
  ierr = PetscMalloc((size+1)*sizeof(int),&starts);CHKERRQ(ierr);
  starts[0] = 0; 
  for (i=1; i<size; i++) { starts[i] = starts[i-1] + nprocs[2*i-2];} 
  for (i=0; i<N; i++) {
    svalues[starts[owner[i]]++] = rows[i];
  }
  ierr = ISRestoreIndices(is,&rows);CHKERRQ(ierr);

  starts[0] = 0;
  for (i=1; i<size+1; i++) { starts[i] = starts[i-1] + nprocs[2*i-2];} 
  count = 0;
  for (i=0; i<size; i++) {
    if (nprocs[2*i+1]) {
      ierr = MPI_Isend(svalues+starts[i],nprocs[2*i],MPI_INT,i,tag,comm,send_waits+count++);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(starts);CHKERRQ(ierr);

  base = owners[rank];

  /*  wait on receives */
  ierr = PetscMalloc(2*(nrecvs+1)*sizeof(int),&lens);CHKERRQ(ierr);
  source = lens + nrecvs;
  count = nrecvs; slen = 0;
  while (count) {
    ierr = MPI_Waitany(nrecvs,recv_waits,&imdex,&recv_status);CHKERRQ(ierr);
    /* unpack receives into our local space */
    ierr = MPI_Get_count(&recv_status,MPI_INT,&n);CHKERRQ(ierr);
    source[imdex]  = recv_status.MPI_SOURCE;
    lens[imdex]    = n;
    slen           += n;
    count--;
  }
  ierr = PetscFree(recv_waits);CHKERRQ(ierr);
  
  /* move the data into the send scatter */
  ierr = PetscMalloc((slen+1)*sizeof(int),&lrows);CHKERRQ(ierr);
  count = 0;
  for (i=0; i<nrecvs; i++) {
    values = rvalues + i*nmax;
    for (j=0; j<lens[i]; j++) {
      lrows[count++] = values[j] - base;
    }
  }
  ierr = PetscFree(rvalues);CHKERRQ(ierr);
  ierr = PetscFree(lens);CHKERRQ(ierr);
  ierr = PetscFree(owner);CHKERRQ(ierr);
  ierr = PetscFree(nprocs);CHKERRQ(ierr);
    
  /* actually zap the local rows */
  ierr = ISCreateGeneral(PETSC_COMM_SELF,slen,lrows,&istmp);CHKERRQ(ierr);  
  PetscLogObjectParent(A,istmp);
  ierr = PetscFree(lrows);CHKERRQ(ierr);
  ierr = MatZeroRows_MPIRowbs_local(A,istmp,diag);CHKERRQ(ierr);
  ierr = ISDestroy(istmp);CHKERRQ(ierr);

  /* wait on sends */
  if (nsends) {
    ierr = PetscMalloc(nsends*sizeof(MPI_Status),&send_status);CHKERRQ(ierr);
    ierr = MPI_Waitall(nsends,send_waits,send_status);CHKERRQ(ierr);
    ierr = PetscFree(send_status);CHKERRQ(ierr);
  }
  ierr = PetscFree(send_waits);
  ierr = PetscFree(svalues);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatNorm_MPIRowbs"
int MatNorm_MPIRowbs(Mat mat,NormType type,PetscReal *norm)
{
  Mat_MPIRowbs *a = (Mat_MPIRowbs*)mat->data;
  BSsprow      *vs,**rs;
  PetscScalar  *xv;
  PetscReal    sum = 0.0;
  int          *xi,nz,i,j,ierr;

  PetscFunctionBegin;
  if (a->size == 1) {
    ierr = MatNorm_MPIRowbs_local(mat,type,norm);CHKERRQ(ierr);
  } else {
    rs = a->A->rows;
    if (type == NORM_FROBENIUS) {
      for (i=0; i<mat->m; i++) {
        vs = *rs++;
        nz = vs->length;
        xv = vs->nz;
        while (nz--) {
#if defined(PETSC_USE_COMPLEX)
          sum += PetscRealPart(PetscConj(*xv)*(*xv)); xv++;
#else
          sum += (*xv)*(*xv); xv++;
#endif
        }
      }
      ierr  = MPI_Allreduce(&sum,norm,1,MPIU_REAL,MPI_SUM,mat->comm);CHKERRQ(ierr);
      *norm = sqrt(*norm);
    } else if (type == NORM_1) { /* max column norm */
      PetscReal *tmp,*tmp2;
      ierr  = PetscMalloc(mat->n*sizeof(PetscReal),&tmp);CHKERRQ(ierr);
      ierr  = PetscMalloc(mat->n*sizeof(PetscReal),&tmp2);CHKERRQ(ierr);
      ierr  = PetscMemzero(tmp,mat->n*sizeof(PetscReal));CHKERRQ(ierr);
      *norm = 0.0;
      for (i=0; i<mat->m; i++) {
        vs = *rs++;
        nz = vs->length;
        xi = vs->col;
        xv = vs->nz;
        while (nz--) {
          tmp[*xi] += PetscAbsScalar(*xv); 
          xi++; xv++;
        }
      }
      ierr = MPI_Allreduce(tmp,tmp2,mat->N,MPIU_REAL,MPI_SUM,mat->comm);CHKERRQ(ierr);
      for (j=0; j<mat->n; j++) {
        if (tmp2[j] > *norm) *norm = tmp2[j];
      }
      ierr = PetscFree(tmp);CHKERRQ(ierr);
      ierr = PetscFree(tmp2);CHKERRQ(ierr);
    } else if (type == NORM_INFINITY) { /* max row norm */
      PetscReal ntemp = 0.0;
      for (i=0; i<mat->m; i++) {
        vs = *rs++;
        nz = vs->length;
        xv = vs->nz;
        sum = 0.0;
        while (nz--) {
          sum += PetscAbsScalar(*xv); xv++;
        }
        if (sum > ntemp) ntemp = sum;
      }
      ierr = MPI_Allreduce(&ntemp,norm,1,MPIU_REAL,MPI_MAX,mat->comm);CHKERRQ(ierr);
    } else {
      SETERRQ(PETSC_ERR_SUP,"No support for two norm");
    }
  }
  PetscFunctionReturn(0); 
}

#undef __FUNCT__  
#define __FUNCT__ "MatMult_MPIRowbs"
int MatMult_MPIRowbs(Mat mat,Vec xx,Vec yy)
{
  Mat_MPIRowbs *bsif = (Mat_MPIRowbs*)mat->data;
  BSprocinfo   *bspinfo = bsif->procinfo;
  PetscScalar  *xxa,*xworka,*yya;
  int          ierr;

  PetscFunctionBegin;
  if (!bsif->blocksolveassembly) {
    ierr = MatAssemblyEnd_MPIRowbs_ForBlockSolve(mat);CHKERRQ(ierr);
  }

  /* Permute and apply diagonal scaling:  [ xwork = D^{1/2} * x ] */
  if (!bsif->vecs_permscale) {
    ierr = VecGetArray(bsif->xwork,&xworka);CHKERRQ(ierr);
    ierr = VecGetArray(xx,&xxa);CHKERRQ(ierr);
    BSperm_dvec(xxa,xworka,bsif->pA->perm);CHKERRBS(0);
    ierr = VecRestoreArray(bsif->xwork,&xworka);CHKERRQ(ierr);
    ierr = VecRestoreArray(xx,&xxa);CHKERRQ(ierr);
    ierr = VecPointwiseDivide(bsif->xwork,bsif->diag,xx);CHKERRQ(ierr);
  } 

  ierr = VecGetArray(xx,&xxa);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&yya);CHKERRQ(ierr);
  /* Do lower triangular multiplication:  [ y = L * xwork ] */
  if (bspinfo->single) {
    BSforward1(bsif->pA,xxa,yya,bsif->comm_pA,bspinfo);CHKERRBS(0);
  }  else {
    BSforward(bsif->pA,xxa,yya,bsif->comm_pA,bspinfo);CHKERRBS(0);
  }
  
  /* Do upper triangular multiplication:  [ y = y + L^{T} * xwork ] */
  if (mat->symmetric) {
    if (bspinfo->single){
      BSbackward1(bsif->pA,xxa,yya,bsif->comm_pA,bspinfo);CHKERRBS(0);
    } else {
      BSbackward(bsif->pA,xxa,yya,bsif->comm_pA,bspinfo);CHKERRBS(0);
    }
  }
  /* not needed for ILU version since forward does it all */
  ierr = VecRestoreArray(xx,&xxa);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&yya);CHKERRQ(ierr);

  /* Apply diagonal scaling to vector:  [  y = D^{1/2} * y ] */
  if (!bsif->vecs_permscale) {
    ierr = VecGetArray(bsif->xwork,&xworka);CHKERRQ(ierr);
    ierr = VecGetArray(xx,&xxa);CHKERRQ(ierr);
    BSiperm_dvec(xworka,xxa,bsif->pA->perm);CHKERRBS(0);
    ierr = VecRestoreArray(bsif->xwork,&xworka);CHKERRQ(ierr);
    ierr = VecRestoreArray(xx,&xxa);CHKERRQ(ierr);
    ierr = VecPointwiseDivide(yy,bsif->diag,bsif->xwork);CHKERRQ(ierr);
    ierr = VecGetArray(bsif->xwork,&xworka);CHKERRQ(ierr);
    ierr = VecGetArray(yy,&yya);CHKERRQ(ierr);
    BSiperm_dvec(xworka,yya,bsif->pA->perm);CHKERRBS(0);
    ierr = VecRestoreArray(bsif->xwork,&xworka);CHKERRQ(ierr);
    ierr = VecRestoreArray(yy,&yya);CHKERRQ(ierr);
  }
  PetscLogFlops(2*bsif->nz - mat->m);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_MPIRowbs"
int MatMultAdd_MPIRowbs(Mat mat,Vec xx,Vec yy,Vec zz)
{
  int          ierr;
  PetscScalar  one = 1.0;

  PetscFunctionBegin;
  ierr = (*mat->ops->mult)(mat,xx,zz);CHKERRQ(ierr);
  ierr = VecAXPY(&one,yy,zz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetInfo_MPIRowbs"
int MatGetInfo_MPIRowbs(Mat A,MatInfoType flag,MatInfo *info)
{
  Mat_MPIRowbs *mat = (Mat_MPIRowbs*)A->data;
  PetscReal    isend[5],irecv[5];
  int          ierr;

  PetscFunctionBegin;
  info->rows_global    = (double)A->M;
  info->columns_global = (double)A->N;
  info->rows_local     = (double)A->m;
  info->columns_local  = (double)A->N;
  info->block_size     = 1.0;
  info->mallocs        = (double)mat->reallocs;
  isend[0] = mat->nz; isend[1] = mat->maxnz; isend[2] =  mat->maxnz -  mat->nz;
  isend[3] = A->mem;  isend[4] = info->mallocs;

  if (flag == MAT_LOCAL) {
    info->nz_used      = isend[0];
    info->nz_allocated = isend[1];
    info->nz_unneeded  = isend[2];
    info->memory       = isend[3];
    info->mallocs      = isend[4];
  } else if (flag == MAT_GLOBAL_MAX) {
    ierr = MPI_Allreduce(isend,irecv,3,MPIU_REAL,MPI_MAX,A->comm);CHKERRQ(ierr);
    info->nz_used      = irecv[0];
    info->nz_allocated = irecv[1];
    info->nz_unneeded  = irecv[2];
    info->memory       = irecv[3];
    info->mallocs      = irecv[4];
  } else if (flag == MAT_GLOBAL_SUM) {
    ierr = MPI_Allreduce(isend,irecv,3,MPIU_REAL,MPI_SUM,A->comm);CHKERRQ(ierr);
    info->nz_used      = irecv[0];
    info->nz_allocated = irecv[1];
    info->nz_unneeded  = irecv[2];
    info->memory       = irecv[3];
    info->mallocs      = irecv[4];
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetDiagonal_MPIRowbs"
int MatGetDiagonal_MPIRowbs(Mat mat,Vec v)
{
  Mat_MPIRowbs *a = (Mat_MPIRowbs*)mat->data;
  BSsprow      **rs = a->A->rows;
  int          i,n,ierr;
  PetscScalar  *x,zero = 0.0;

  PetscFunctionBegin;
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");  
  if (!a->blocksolveassembly) {
    ierr = MatAssemblyEnd_MPIRowbs_ForBlockSolve(mat);CHKERRQ(ierr);
  }

  ierr = VecSet(&zero,v);CHKERRQ(ierr);
  ierr = VecGetLocalSize(v,&n);CHKERRQ(ierr);
  if (n != mat->m) SETERRQ(PETSC_ERR_ARG_SIZ,"Nonconforming mat and vec");
  ierr = VecGetArray(v,&x);CHKERRQ(ierr); 
  for (i=0; i<mat->m; i++) {
    x[i] = rs[i]->nz[rs[i]->diag_ind]; 
  }
  ierr = VecRestoreArray(v,&x);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_MPIRowbs"
int MatDestroy_MPIRowbs(Mat mat)
{
  Mat_MPIRowbs *a = (Mat_MPIRowbs*)mat->data;
  BSspmat      *A = a->A;
  BSsprow      *vs;
  int          i,ierr;

  PetscFunctionBegin;
#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject)mat,"Rows=%d, Cols=%d",mat->M,mat->N);
#endif
  ierr = PetscFree(a->rowners);CHKERRQ(ierr);
  ierr = MatStashDestroy_Private(&mat->stash);CHKERRQ(ierr);
  if (a->bsmap) {
    if (a->bsmap->vlocal2global) {ierr = PetscFree(a->bsmap->vlocal2global);CHKERRQ(ierr);}
    if (a->bsmap->vglobal2local) {ierr = PetscFree(a->bsmap->vglobal2local);CHKERRQ(ierr);}
    if (a->bsmap->vglobal2proc)  (*a->bsmap->free_g2p)(a->bsmap->vglobal2proc);
    ierr = PetscFree(a->bsmap);CHKERRQ(ierr);
  } 

  if (A) {
    for (i=0; i<mat->m; i++) {
      vs = A->rows[i];
      ierr = MatFreeRowbs_Private(mat,vs->length,vs->col,vs->nz);CHKERRQ(ierr);
    }
    /* Note: A->map = a->bsmap is freed above */
    ierr = PetscFree(A->rows);CHKERRQ(ierr);
    ierr = PetscFree(A);CHKERRQ(ierr);
  }
  if (a->procinfo) {BSfree_ctx(a->procinfo);CHKERRBS(0);}
  if (a->diag)     {ierr = VecDestroy(a->diag);CHKERRQ(ierr);}
  if (a->xwork)    {ierr = VecDestroy(a->xwork);CHKERRQ(ierr);}
  if (a->pA)       {BSfree_par_mat(a->pA);CHKERRBS(0);}
  if (a->fpA)      {BSfree_copy_par_mat(a->fpA);CHKERRBS(0);}
  if (a->comm_pA)  {BSfree_comm(a->comm_pA);CHKERRBS(0);}
  if (a->comm_fpA) {BSfree_comm(a->comm_fpA);CHKERRBS(0);}
  if (a->imax)     {ierr = PetscFree(a->imax);CHKERRQ(ierr);}
  ierr = PetscFree(a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetOption_MPIRowbs"
int MatSetOption_MPIRowbs(Mat A,MatOption op)
{
  Mat_MPIRowbs *a = (Mat_MPIRowbs*)A->data;

  PetscFunctionBegin;
  switch (op) {
  case MAT_ROW_ORIENTED:
    a->roworiented = PETSC_TRUE;
    break;
  case MAT_COLUMN_ORIENTED:
    a->roworiented = PETSC_FALSE; 
    break;
  case MAT_COLUMNS_SORTED:
    a->sorted      = 1;
    break;
  case MAT_COLUMNS_UNSORTED:
    a->sorted      = 0;
    break;
  case MAT_NO_NEW_NONZERO_LOCATIONS:
    a->nonew       = 1;
    break;
  case MAT_YES_NEW_NONZERO_LOCATIONS:
    a->nonew       = 0;
    break;
  case MAT_DO_NOT_USE_INODES:
    a->bs_color_single = 1;
    break;
  case MAT_YES_NEW_DIAGONALS:
  case MAT_ROWS_SORTED: 
  case MAT_NEW_NONZERO_LOCATION_ERR:
  case MAT_NEW_NONZERO_ALLOCATION_ERR:
  case MAT_ROWS_UNSORTED:
  case MAT_USE_HASH_TABLE:
    PetscLogInfo(A,"MatSetOption_MPIRowbs:Option ignored\n");
    break;
  case MAT_IGNORE_OFF_PROC_ENTRIES:
    a->donotstash = PETSC_TRUE;
    break;
  case MAT_NO_NEW_DIAGONALS:
    SETERRQ(PETSC_ERR_SUP,"MAT_NO_NEW_DIAGONALS");
    break;
  case MAT_KEEP_ZEROED_ROWS:
    SETERRQ(PETSC_ERR_SUP,"MAT_KEEP_ZEROED_ROWS");
    break;
  default:
    SETERRQ(PETSC_ERR_SUP,"unknown option");
    break;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetRow_MPIRowbs"
int MatGetRow_MPIRowbs(Mat AA,int row,int *nz,int **idx,PetscScalar **v)
{
  Mat_MPIRowbs *mat = (Mat_MPIRowbs*)AA->data;
  BSspmat      *A = mat->A;
  BSsprow      *rs;
   
  PetscFunctionBegin;
  if (row < mat->rstart || row >= mat->rend) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Only local rows");

  rs  = A->rows[row - mat->rstart];
  *nz = rs->length;
  if (v)   *v   = rs->nz;
  if (idx) *idx = rs->col;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatRestoreRow_MPIRowbs"
int MatRestoreRow_MPIRowbs(Mat A,int row,int *nz,int **idx,PetscScalar **v)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------ */

#undef __FUNCT__  
#define __FUNCT__ "MatPrintHelp_MPIRowbs"
int MatPrintHelp_MPIRowbs(Mat A)
{
  static PetscTruth called = PETSC_FALSE; 
  MPI_Comm          comm = A->comm;
  int               ierr;

  PetscFunctionBegin;
  if (called) {PetscFunctionReturn(0);} else called = PETSC_TRUE;
  ierr = (*PetscHelpPrintf)(comm," Options for MATMPIROWBS matrix format (needed for BlockSolve):\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(comm,"  -mat_rowbs_no_inode  - Do not use inodes\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetUpPreallocation_MPIRowbs"
int MatSetUpPreallocation_MPIRowbs(Mat A)
{
  int        ierr;

  PetscFunctionBegin;
  ierr =  MatMPIRowbsSetPreallocation(A,PETSC_DEFAULT,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------*/
EXTERN int MatCholeskyFactorNumeric_MPIRowbs(Mat,Mat*);
EXTERN int MatIncompleteCholeskyFactorSymbolic_MPIRowbs(Mat,IS,MatFactorInfo*,Mat *);
EXTERN int MatLUFactorNumeric_MPIRowbs(Mat,Mat*);
EXTERN int MatILUFactorSymbolic_MPIRowbs(Mat,IS,IS,MatFactorInfo*,Mat *);
EXTERN int MatSolve_MPIRowbs(Mat,Vec,Vec);
EXTERN int MatForwardSolve_MPIRowbs(Mat,Vec,Vec);
EXTERN int MatBackwardSolve_MPIRowbs(Mat,Vec,Vec);
EXTERN int MatScaleSystem_MPIRowbs(Mat,Vec,Vec);
EXTERN int MatUnScaleSystem_MPIRowbs(Mat,Vec,Vec);
EXTERN int MatUseScaledForm_MPIRowbs(Mat,PetscTruth);

static struct _MatOps MatOps_Values = {MatSetValues_MPIRowbs,
       MatGetRow_MPIRowbs,
       MatRestoreRow_MPIRowbs,
       MatMult_MPIRowbs,
       MatMultAdd_MPIRowbs,
       MatMult_MPIRowbs,
       MatMultAdd_MPIRowbs,
       MatSolve_MPIRowbs,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       MatGetInfo_MPIRowbs,
       0,
       MatGetDiagonal_MPIRowbs,
       0,MatNorm_MPIRowbs,
       MatAssemblyBegin_MPIRowbs,
       MatAssemblyEnd_MPIRowbs,
       0,
       MatSetOption_MPIRowbs,
       MatZeroEntries_MPIRowbs,
       MatZeroRows_MPIRowbs,
       0,
       MatLUFactorNumeric_MPIRowbs,
       0,
       MatCholeskyFactorNumeric_MPIRowbs,
       MatSetUpPreallocation_MPIRowbs,
       MatILUFactorSymbolic_MPIRowbs,
       MatIncompleteCholeskyFactorSymbolic_MPIRowbs,
       0,
       0,
       0,
       MatForwardSolve_MPIRowbs,
       MatBackwardSolve_MPIRowbs,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       MatPrintHelp_MPIRowbs,
       MatScale_MPIRowbs,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       MatDestroy_MPIRowbs,
       MatView_MPIRowbs,
       MatGetPetscMaps_Petsc,
       MatUseScaledForm_MPIRowbs,
       MatScaleSystem_MPIRowbs,
       MatUnScaleSystem_MPIRowbs};

/* ------------------------------------------------------------------- */


EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatCreate_MPIRowbs"
int MatCreate_MPIRowbs(Mat A)
{
  Mat_MPIRowbs *a;
  BSmapping    *bsmap;
  BSoff_map    *bsoff;
  int          i,ierr,*offset,m,M;
  PetscTruth   flg1,flg2,flg3;
  BSprocinfo   *bspinfo;
  MPI_Comm     comm;
  
  PetscFunctionBegin;
  comm = A->comm;
  m    = A->m;
  M    = A->M;

  ierr                  = PetscNew(Mat_MPIRowbs,&a);CHKERRQ(ierr);
  A->data               = (void*)a;
  ierr                  = PetscMemcpy(A->ops,&MatOps_Values,sizeof(struct _MatOps));CHKERRQ(ierr);
  A->factor             = 0;
  A->mapping            = 0;
  a->vecs_permscale     = PETSC_FALSE;
  A->insertmode         = NOT_SET_VALUES;
  a->blocksolveassembly = 0;
  ierr = MPI_Comm_rank(comm,&a->rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&a->size);CHKERRQ(ierr);

  ierr = PetscSplitOwnership(comm,&m,&M);CHKERRQ(ierr);

  A->N = M;
  A->M = M;
  A->m = m;
  A->n = A->N;  /* each row stores all columns */
  ierr                             = PetscMalloc((A->m+1)*sizeof(int),&a->imax);CHKERRQ(ierr);
  a->reallocs                      = 0;

  /* the information in the maps duplicates the information computed below, eventually 
     we should remove the duplicate information that is not contained in the maps */
  ierr = PetscMapCreateMPI(comm,m,M,&A->rmap);CHKERRQ(ierr);
  ierr = PetscMapCreateMPI(comm,m,M,&A->cmap);CHKERRQ(ierr);

  /* build local table of row ownerships */
  ierr          = PetscMalloc((a->size+2)*sizeof(int),&a->rowners);CHKERRQ(ierr);
  ierr          = MPI_Allgather(&m,1,MPI_INT,a->rowners+1,1,MPI_INT,comm);CHKERRQ(ierr);
  a->rowners[0] = 0;
  for (i=2; i<=a->size; i++) {
    a->rowners[i] += a->rowners[i-1];
  }
  a->rstart = a->rowners[a->rank]; 
  a->rend   = a->rowners[a->rank+1]; 
  PetscLogObjectMemory(A,(A->m+a->size+3)*sizeof(int));

  /* build cache for off array entries formed */
  ierr = MatStashCreate_Private(A->comm,1,&A->stash);CHKERRQ(ierr);
  a->donotstash = PETSC_FALSE;

  /* Initialize BlockSolve information */
  a->A	      = 0;
  a->pA	      = 0;
  a->comm_pA  = 0;
  a->fpA      = 0;
  a->comm_fpA = 0;
  a->alpha    = 1.0;
  a->ierr     = 0;
  a->failures = 0;
  ierr = VecCreateMPI(A->comm,A->m,A->M,&(a->diag));CHKERRQ(ierr);
  ierr = VecDuplicate(a->diag,&(a->xwork));CHKERRQ(ierr);
  PetscLogObjectParent(A,a->diag);  PetscLogObjectParent(A,a->xwork);
  PetscLogObjectMemory(A,(A->m+1)*sizeof(PetscScalar));
  bspinfo = BScreate_ctx();CHKERRBS(0);
  a->procinfo = bspinfo;
  BSctx_set_id(bspinfo,a->rank);CHKERRBS(0);
  BSctx_set_np(bspinfo,a->size);CHKERRBS(0);
  BSctx_set_ps(bspinfo,comm);CHKERRBS(0);
  BSctx_set_cs(bspinfo,INT_MAX);CHKERRBS(0);
  BSctx_set_is(bspinfo,INT_MAX);CHKERRBS(0);
  BSctx_set_ct(bspinfo,IDO);CHKERRBS(0);
#if defined(PETSC_USE_DEBUG)
  BSctx_set_err(bspinfo,1);CHKERRBS(0);  /* BS error checking */
#endif
  BSctx_set_rt(bspinfo,1);CHKERRBS(0);
  ierr = PetscOptionsHasName(PETSC_NULL,"-log_info",&flg1);CHKERRQ(ierr);
  if (flg1) {
    BSctx_set_pr(bspinfo,1);CHKERRBS(0);
  }
  ierr = PetscOptionsHasName(PETSC_NULL,"-pc_ilu_factorpointwise",&flg1);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-pc_icc_factorpointwise",&flg2);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-mat_rowbs_no_inode",&flg3);CHKERRQ(ierr);
  if (flg1 || flg2 || flg3) {
    BSctx_set_si(bspinfo,1);CHKERRBS(0);
  } else {
    BSctx_set_si(bspinfo,0);CHKERRBS(0);
  }
#if defined(PETSC_USE_LOG)
  MLOG_INIT();  /* Initialize logging */
#endif

  /* Compute global offsets */
  offset = &a->rstart;

  ierr = PetscNew(BSmapping,&a->bsmap);CHKERRQ(ierr);
  PetscLogObjectMemory(A,sizeof(BSmapping));
  bsmap = a->bsmap;
  ierr                           = PetscMalloc(sizeof(int),&bsmap->vlocal2global);CHKERRQ(ierr);
  *((int *)bsmap->vlocal2global) = (*offset);
  bsmap->flocal2global	         = BSloc2glob;
  bsmap->free_l2g                = 0;
  ierr                           = PetscMalloc(sizeof(int),&bsmap->vglobal2local);CHKERRQ(ierr);
  *((int *)bsmap->vglobal2local) = (*offset);
  bsmap->fglobal2local	         = BSglob2loc;
  bsmap->free_g2l	         = 0;
  bsoff                          = BSmake_off_map(*offset,bspinfo,A->M);
  bsmap->vglobal2proc	         = (void *)bsoff;
  bsmap->fglobal2proc	         = BSglob2proc;
  bsmap->free_g2p                = (void(*)(void*)) BSfree_off_map;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatMPIRowbsSetPreallocation"
/* @
  MatMPIRowbsSetPreallocation - Sets the number of expected nonzeros 
  per row in the matrix.

  Input Parameter:
+  mat - matrix
.  nz - maximum expected for any row
-  nzz - number expected in each row

  Note:
  This routine is valid only for matrices stored in the MATMPIROWBS
  format.
@ */
int MatMPIRowbsSetPreallocation(Mat mat,int nz,int *nnz)
{
  PetscTruth ismpirowbs;
  int        ierr;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)mat,MATMPIROWBS,&ismpirowbs);
  if (!ismpirowbs) SETERRQ(PETSC_ERR_ARG_WRONG,"For MATMPIROWBS matrix type");
  mat->preallocated = PETSC_TRUE;
  ierr = MatCreateMPIRowbs_local(mat,nz,nnz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



/* --------------- extra BlockSolve-specific routines -------------- */
#undef __FUNCT__  
#define __FUNCT__ "MatGetBSProcinfo"
/* @
  MatGetBSProcinfo - Gets the BlockSolve BSprocinfo context, which the
  user can then manipulate to alter the default parameters.

  Input Parameter:
  mat - matrix

  Output Parameter:
  procinfo - processor information context

  Note:
  This routine is valid only for matrices stored in the MATMPIROWBS
  format.
@ */
int MatGetBSProcinfo(Mat mat,BSprocinfo *procinfo)
{
  Mat_MPIRowbs *a = (Mat_MPIRowbs*)mat->data;
  PetscTruth   ismpirowbs;
  int          ierr;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)mat,MATMPIROWBS,&ismpirowbs);CHKERRQ(ierr);
  if (!ismpirowbs) SETERRQ(PETSC_ERR_ARG_WRONG,"For MATMPIROWBS matrix type");
  procinfo = a->procinfo;
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatLoad_MPIRowbs"
int MatLoad_MPIRowbs(PetscViewer viewer,MatType type,Mat *newmat)
{
  Mat_MPIRowbs *a;
  BSspmat      *A;
  BSsprow      **rs;
  Mat          mat;
  int          i,nz,ierr,j,rstart,rend,fd,*ourlens,*sndcounts = 0,*procsnz;
  int          header[4],rank,size,*rowlengths = 0,M,m,*rowners,maxnz,*cols;
  PetscScalar  *vals;
  MPI_Comm     comm = ((PetscObject)viewer)->comm;
  MPI_Status   status;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr); 
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    ierr = PetscViewerBinaryGetDescriptor(viewer,&fd);CHKERRQ(ierr);
    ierr = PetscBinaryRead(fd,(char *)header,4,PETSC_INT);CHKERRQ(ierr);
    if (header[0] != MAT_FILE_COOKIE) SETERRQ(PETSC_ERR_FILE_UNEXPECTED,"Not matrix object");
    if (header[3] < 0) {
      SETERRQ(PETSC_ERR_FILE_UNEXPECTED,"Matrix stored in special format,cannot load as MPIRowbs");
    }
  }

  ierr = MPI_Bcast(header+1,3,MPI_INT,0,comm);CHKERRQ(ierr);
  M = header[1]; 
  /* determine ownership of all rows */
  m          = M/size + ((M % size) > rank);
  ierr       = PetscMalloc((size+2)*sizeof(int),&rowners);CHKERRQ(ierr);
  ierr       = MPI_Allgather(&m,1,MPI_INT,rowners+1,1,MPI_INT,comm);
  rowners[0] = 0;
  for (i=2; i<=size; i++) {
    rowners[i] += rowners[i-1];
  }
  rstart = rowners[rank]; 
  rend   = rowners[rank+1]; 

  /* distribute row lengths to all processors */
  ierr = PetscMalloc((rend-rstart)*sizeof(int),&ourlens);CHKERRQ(ierr);
  if (!rank) {
    ierr = PetscMalloc(M*sizeof(int),&rowlengths);CHKERRQ(ierr);
    ierr = PetscBinaryRead(fd,rowlengths,M,PETSC_INT);CHKERRQ(ierr);
    ierr = PetscMalloc(size*sizeof(int),&sndcounts);CHKERRQ(ierr);
    for (i=0; i<size; i++) sndcounts[i] = rowners[i+1] - rowners[i];
    ierr = MPI_Scatterv(rowlengths,sndcounts,rowners,MPI_INT,ourlens,rend-rstart,MPI_INT,0,comm);CHKERRQ(ierr);
    ierr = PetscFree(sndcounts);CHKERRQ(ierr);
  } else {
    ierr = MPI_Scatterv(0,0,0,MPI_INT,ourlens,rend-rstart,MPI_INT,0,comm);CHKERRQ(ierr);
  }

  /* create our matrix */
  ierr = MatCreateMPIRowbs(comm,m,M,0,ourlens,newmat);CHKERRQ(ierr);
  mat = *newmat;
  ierr = PetscFree(ourlens);CHKERRQ(ierr);

  a = (Mat_MPIRowbs*)mat->data;
  A = a->A;
  rs = A->rows;

  if (!rank) {
    /* calculate the number of nonzeros on each processor */
    ierr = PetscMalloc(size*sizeof(int),&procsnz);CHKERRQ(ierr);
    ierr = PetscMemzero(procsnz,size*sizeof(int));CHKERRQ(ierr);
    for (i=0; i<size; i++) {
      for (j=rowners[i]; j< rowners[i+1]; j++) {
        procsnz[i] += rowlengths[j];
      }
    }
    ierr = PetscFree(rowlengths);CHKERRQ(ierr);

    /* determine max buffer needed and allocate it */
    maxnz = 0;
    for (i=0; i<size; i++) {
      maxnz = PetscMax(maxnz,procsnz[i]);
    }
    ierr = PetscMalloc(maxnz*sizeof(int),&cols);CHKERRQ(ierr);

    /* read in my part of the matrix column indices  */
    nz = procsnz[0];
    ierr = PetscBinaryRead(fd,cols,nz,PETSC_INT);CHKERRQ(ierr);
    
    /* insert it into my part of matrix */
    nz = 0;
    for (i=0; i<A->num_rows; i++) {
      for (j=0; j<a->imax[i]; j++) {
        rs[i]->col[j] = cols[nz++];
      }
      rs[i]->length = a->imax[i];
    }
    /* read in parts for all other processors */
    for (i=1; i<size; i++) {
      nz   = procsnz[i];
      ierr = PetscBinaryRead(fd,cols,nz,PETSC_INT);CHKERRQ(ierr);
      ierr = MPI_Send(cols,nz,MPI_INT,i,mat->tag,comm);CHKERRQ(ierr);
    }
    ierr = PetscFree(cols);CHKERRQ(ierr);
    ierr = PetscMalloc(maxnz*sizeof(PetscScalar),&vals);CHKERRQ(ierr);

    /* read in my part of the matrix numerical values  */
    nz   = procsnz[0];
    ierr = PetscBinaryRead(fd,vals,nz,PETSC_SCALAR);CHKERRQ(ierr);
    
    /* insert it into my part of matrix */
    nz = 0;
    for (i=0; i<A->num_rows; i++) {
      for (j=0; j<a->imax[i]; j++) {
        rs[i]->nz[j] = vals[nz++];
      }
    }
    /* read in parts for all other processors */
    for (i=1; i<size; i++) {
      nz   = procsnz[i];
      ierr = PetscBinaryRead(fd,vals,nz,PETSC_SCALAR);CHKERRQ(ierr);
      ierr = MPI_Send(vals,nz,MPIU_SCALAR,i,mat->tag,comm);CHKERRQ(ierr);
    }
    ierr = PetscFree(vals);CHKERRQ(ierr);
    ierr = PetscFree(procsnz);CHKERRQ(ierr);
  } else {
    /* determine buffer space needed for message */
    nz = 0;
    for (i=0; i<A->num_rows; i++) {
      nz += a->imax[i];
    }
    ierr = PetscMalloc(nz*sizeof(int),&cols);CHKERRQ(ierr);

    /* receive message of column indices*/
    ierr = MPI_Recv(cols,nz,MPI_INT,0,mat->tag,comm,&status);CHKERRQ(ierr);
    ierr = MPI_Get_count(&status,MPI_INT,&maxnz);CHKERRQ(ierr);
    if (maxnz != nz) SETERRQ(PETSC_ERR_FILE_UNEXPECTED,"something is wrong");

    /* insert it into my part of matrix */
    nz = 0;
    for (i=0; i<A->num_rows; i++) {
      for (j=0; j<a->imax[i]; j++) {
        rs[i]->col[j] = cols[nz++];
      }
      rs[i]->length = a->imax[i];
    }
    ierr = PetscFree(cols);CHKERRQ(ierr);
    ierr = PetscMalloc(nz*sizeof(PetscScalar),&vals);CHKERRQ(ierr);

    /* receive message of values*/
    ierr = MPI_Recv(vals,nz,MPIU_SCALAR,0,mat->tag,comm,&status);CHKERRQ(ierr);
    ierr = MPI_Get_count(&status,MPIU_SCALAR,&maxnz);CHKERRQ(ierr);
    if (maxnz != nz) SETERRQ(PETSC_ERR_FILE_UNEXPECTED,"something is wrong");

    /* insert it into my part of matrix */
    nz = 0;
    for (i=0; i<A->num_rows; i++) {
      for (j=0; j<a->imax[i]; j++) {
        rs[i]->nz[j] = vals[nz++];
      }
      rs[i]->length = a->imax[i];
    }
    ierr = PetscFree(vals);CHKERRQ(ierr);
  }
  ierr = PetscFree(rowners);CHKERRQ(ierr);
  a->nz = a->maxnz;
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

/* 
    Special destroy and view routines for factored matrices 
*/
#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_MPIRowbs_Factored"
static int MatDestroy_MPIRowbs_Factored(Mat mat)
{
  PetscFunctionBegin;
#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject)mat,"Rows=%d, Cols=%d",mat->M,mat->N);
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatView_MPIRowbs_Factored"
static int MatView_MPIRowbs_Factored(Mat mat,PetscViewer viewer)
{
  int ierr;

  PetscFunctionBegin;
  ierr = MatView((Mat) mat->data,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatIncompleteCholeskyFactorSymbolic_MPIRowbs"
int MatIncompleteCholeskyFactorSymbolic_MPIRowbs(Mat mat,IS isrow,MatFactorInfo *info,Mat *newfact)
{
  /* Note:  f is not currently used in BlockSolve */
  Mat          newmat;
  Mat_MPIRowbs *mbs = (Mat_MPIRowbs*)mat->data;
  int          ierr;
  PetscTruth   idn;

  PetscFunctionBegin;
  if (isrow) {
    ierr = ISIdentity(isrow,&idn);CHKERRQ(ierr);
    if (!idn) SETERRQ(PETSC_ERR_SUP,"Only identity row permutation supported");
  }

  if (!mat->symmetric) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"To use incomplete Cholesky \n\
        preconditioning with a MATMPIROWBS matrix you must declare it to be \n\
        symmetric using the option MatSetOption(A,MAT_SYMMETRIC)");
  }

  if (!mbs->blocksolveassembly) {
    BSset_mat_icc_storage(mbs->A,PETSC_TRUE);CHKERRBS(0);
    BSset_mat_symmetric(mbs->A,PETSC_TRUE);CHKERRBS(0);
    ierr = MatAssemblyEnd_MPIRowbs_ForBlockSolve(mat);CHKERRQ(ierr);
  }

  /* Copy permuted matrix */
  if (mbs->fpA) {BSfree_copy_par_mat(mbs->fpA);CHKERRBS(0);}
  mbs->fpA = BScopy_par_mat(mbs->pA);CHKERRBS(0);

  /* Set up the communication for factorization */
  if (mbs->comm_fpA) {BSfree_comm(mbs->comm_fpA);CHKERRBS(0);}
  mbs->comm_fpA = BSsetup_factor(mbs->fpA,mbs->procinfo);CHKERRBS(0);

  /* 
      Create a new Mat structure to hold the "factored" matrix, 
    not this merely contains a pointer to the original matrix, since
    the original matrix contains the factor information.
  */
  PetscHeaderCreate(newmat,_p_Mat,struct _MatOps,MAT_COOKIE,-1,"Mat",mat->comm,MatDestroy,MatView);
  PetscLogObjectCreate(newmat);
  PetscLogObjectMemory(newmat,sizeof(struct _p_Mat));

  newmat->data         = (void*)mat;
  ierr                 = PetscMemcpy(newmat->ops,&MatOps_Values,sizeof(struct _MatOps));CHKERRQ(ierr);
  newmat->ops->destroy = MatDestroy_MPIRowbs_Factored;
  newmat->ops->view    = MatView_MPIRowbs_Factored;
  newmat->factor       = 1;
  newmat->preallocated = PETSC_TRUE;
  newmat->M            = mat->M;
  newmat->N            = mat->N;
  newmat->m            = mat->m;
  newmat->n            = mat->n;
  ierr = PetscStrallocpy(MATMPIROWBS,&newmat->type_name);CHKERRQ(ierr);

  *newfact = newmat; 
  PetscFunctionReturn(0); 
}

#undef __FUNCT__  
#define __FUNCT__ "MatILUFactorSymbolic_MPIRowbs"
int MatILUFactorSymbolic_MPIRowbs(Mat mat,IS isrow,IS iscol,MatFactorInfo* info,Mat *newfact)
{
  Mat          newmat;
  Mat_MPIRowbs *mbs = (Mat_MPIRowbs*)mat->data;
  int          ierr;
  PetscTruth   idn;
  PetscFunctionBegin;

  if (info->levels != 0) SETERRQ(1,"Blocksolve ILU only supports 0 fill");
  if (isrow) {
    ierr = ISIdentity(isrow,&idn);CHKERRQ(ierr);
    if (!idn) SETERRQ(PETSC_ERR_SUP,"Only identity row permutation supported");
  }
  if (iscol) {
    ierr = ISIdentity(iscol,&idn);CHKERRQ(ierr);
    if (!idn) SETERRQ(PETSC_ERR_SUP,"Only identity column permutation supported");
  }

  if (!mbs->blocksolveassembly) {
    ierr = MatAssemblyEnd_MPIRowbs_ForBlockSolve(mat);CHKERRQ(ierr);
  }
 
  if (mat->symmetric) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"To use ILU preconditioner with \n\
        MatCreateMPIRowbs() matrix you CANNOT declare it to be a symmetric matrix\n\
        using the option MatSetOption(A,MAT_SYMMETRIC)");
  }

  /* Copy permuted matrix */
  if (mbs->fpA) {BSfree_copy_par_mat(mbs->fpA);CHKERRBS(0);}
  mbs->fpA = BScopy_par_mat(mbs->pA);CHKERRBS(0); 

  /* Set up the communication for factorization */
  if (mbs->comm_fpA) {BSfree_comm(mbs->comm_fpA);CHKERRBS(0);}
  mbs->comm_fpA = BSsetup_factor(mbs->fpA,mbs->procinfo);CHKERRBS(0);

  /* 
      Create a new Mat structure to hold the "factored" matrix,
    not this merely contains a pointer to the original matrix, since
    the original matrix contains the factor information.
  */
  PetscHeaderCreate(newmat,_p_Mat,struct _MatOps,MAT_COOKIE,-1,"Mat",mat->comm,MatDestroy,MatView);
  PetscLogObjectCreate(newmat);
  PetscLogObjectMemory(newmat,sizeof(struct _p_Mat));

  newmat->data         = (void*)mat;
  ierr                 = PetscMemcpy(newmat->ops,&MatOps_Values,sizeof(struct _MatOps));CHKERRQ(ierr);
  newmat->ops->destroy = MatDestroy_MPIRowbs_Factored;
  newmat->ops->view    = MatView_MPIRowbs_Factored;
  newmat->factor       = 1;
  newmat->preallocated = PETSC_TRUE;
  newmat->M            = mat->M;
  newmat->N            = mat->N;
  newmat->m            = mat->m;
  newmat->n            = mat->n;
  ierr = PetscStrallocpy(MATMPIROWBS,&newmat->type_name);CHKERRQ(ierr);

  *newfact = newmat; 
  PetscFunctionReturn(0); 
}

#undef __FUNCT__  
#define __FUNCT__ "MatMPIRowbsGetColor"
int MatMPIRowbsGetColor(Mat mat,ISColoring *coloring)
{
  int          ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  ierr = ISColoringCreate(mat->comm,mat->m,0,coloring);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCreateMPIRowbs"
/*@C
   MatCreateMPIRowbs - Creates a sparse parallel matrix in the MATMPIROWBS
   format.  This format is intended primarily as an interface for BlockSolve95.

   Collective on MPI_Comm

   Input Parameters:
+  comm - MPI communicator
.  m - number of local rows (or PETSC_DECIDE to have calculated)
.  M - number of global rows (or PETSC_DECIDE to have calculated)
.  nz - number of nonzeros per row (same for all local rows)
-  nnz - number of nonzeros per row (possibly different for each row).

   Output Parameter:
.  newA - the matrix 

   Notes:
   If PETSC_DECIDE or  PETSC_DETERMINE is used for a particular argument on one processor
   than it must be used on all processors that share the object for that argument.

   The user MUST specify either the local or global matrix dimensions
   (possibly both).

   Specify the preallocated storage with either nz or nnz (not both).  Set 
   nz=PETSC_DEFAULT and nnz=PETSC_NULL for PETSc to control dynamic memory 
   allocation.

   Notes:
   By default, the matrix is assumed to be nonsymmetric; the user can
   take advantage of special optimizations for symmetric matrices by calling
$     MatSetOption(mat,MAT_SYMMETRIC)
   BEFORE calling the routine MatAssemblyBegin().

   Internally, the MATMPIROWBS format inserts zero elements to the
   matrix if necessary, so that nonsymmetric matrices are considered
   to be symmetric in terms of their sparsity structure; this format
   is required for use of the parallel communication routines within
   BlockSolve95. In particular, if the matrix element A[i,j] exists,
   then PETSc will internally allocate a 0 value for the element
   A[j,i] during MatAssemblyEnd() if the user has not already set
   a value for the matrix element A[j,i].

   Options Database Keys:
.  -mat_rowbs_no_inode - Do not use inodes.

   Level: intermediate
  
.keywords: matrix, row, symmetric, sparse, parallel, BlockSolve

.seealso: MatCreate(), MatSetValues()
@*/
int MatCreateMPIRowbs(MPI_Comm comm,int m,int M,int nz,int *nnz,Mat *newA)
{
  int ierr;
  
  PetscFunctionBegin;
  ierr = MatCreate(comm,m,m,M,M,newA);CHKERRQ(ierr);
  ierr = MatSetType(*newA,MATMPIROWBS);CHKERRQ(ierr);
  ierr = MatMPIRowbsSetPreallocation(*newA,nz,nnz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}




