#ifndef lint
static char vcid[] = "$Id: aij.c,v 1.90 1995/09/21 20:10:56 bsmith Exp bsmith $";
#endif

#include "aij.h"
#include "vec/vecimpl.h"
#include "inline/spops.h"

extern int MatToSymmetricIJ_SeqAIJ(Mat_SeqAIJ*,int**,int**);

static int MatGetReordering_SeqAIJ(Mat mat,MatOrdering type,IS *rperm, IS *cperm)
{
  Mat_SeqAIJ *aij = (Mat_SeqAIJ *) mat->data;
  int     ierr, *ia, *ja;

  if (!aij->assembled) 
    SETERRQ(1,"MatGetReordering_SeqAIJ:Cannot reorder unassembled matrix");

  ierr = MatToSymmetricIJ_SeqAIJ( aij, &ia, &ja ); CHKERRQ(ierr);
  ierr = MatGetReordering_IJ(aij->n,ia,ja,type,rperm,cperm); CHKERRQ(ierr);
  PETSCFREE(ia); PETSCFREE(ja);
  return 0; 
}

#define CHUNCKSIZE   10

/* This version has row oriented v  */
static int MatSetValues_SeqAIJ(Mat matin,int m,int *idxm,int n,
                            int *idxn,Scalar *v,InsertMode  addv)
{
  Mat_SeqAIJ *mat = (Mat_SeqAIJ *) matin->data;
  int    *rp,k,a,b,t,ii,row,nrow,i,col,l,rmax, N, sorted = mat->sorted;
  int    *imax = mat->imax, *ai = mat->i, *ailen = mat->ilen;
  int    *aj = mat->j, nonew = mat->nonew;
  Scalar *ap,value, *aa = mat->a;
   int shift = mat->indexshift;

  for ( k=0; k<m; k++ ) { /* loop over added rows */
    row  = idxm[k];   
    if (row < 0) SETERRQ(1,"MatSetValues_SeqAIJ:Negative row");
    if (row >= mat->m) SETERRQ(1,"MatSetValues_SeqAIJ:Row too large");
    rp   = aj + ai[row] + shift; ap = aa + ai[row] + shift;
    rmax = imax[row]; nrow = ailen[row]; 
    a = 0;
    for ( l=0; l<n; l++ ) { /* loop over added columns */
      if (idxn[l] < 0) SETERRQ(1,"MatSetValues_SeqAIJ:Negative column");
      if (idxn[l] >= mat->n) 
        SETERRQ(1,"MatSetValues_SeqAIJ:Column too large");
      col = idxn[l] - shift; value = *v++; 
      if (!sorted) a = 0; b = nrow;
      while (b-a > 5) {
        t = (b+a)/2;
        if (rp[t] > col) b = t;
        else             a = t;
      }
      for ( i=a; i<b; i++ ) {
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
        int    new_nz = ai[mat->m] + CHUNCKSIZE,len,*new_i,*new_j;
        Scalar *new_a;

        /* malloc new storage space */
        len     = new_nz*(sizeof(int)+sizeof(Scalar))+(mat->m+1)*sizeof(int);
        new_a  = (Scalar *) PETSCMALLOC( len ); CHKPTRQ(new_a);
        new_j  = (int *) (new_a + new_nz);
        new_i  = new_j + new_nz;

        /* copy over old data into new slots */
        for ( ii=0; ii<row+1; ii++ ) {new_i[ii] = ai[ii];}
        for ( ii=row+1; ii<mat->m+1; ii++ ) {new_i[ii] = ai[ii]+CHUNCKSIZE;}
        PETSCMEMCPY(new_j,aj,(ai[row]+nrow+shift)*sizeof(int));
        len = (new_nz - CHUNCKSIZE - ai[row] - nrow - shift);
        PETSCMEMCPY(new_j+ai[row]+shift+nrow+CHUNCKSIZE,aj+ai[row]+shift+nrow,
                                                           len*sizeof(int));
        PETSCMEMCPY(new_a,aa,(ai[row]+nrow+shift)*sizeof(Scalar));
        PETSCMEMCPY(new_a+ai[row]+shift+nrow+CHUNCKSIZE,aa+ai[row]+shift+nrow,
                                                         len*sizeof(Scalar)); 
        /* free up old matrix storage */
        PETSCFREE(mat->a); if (!mat->singlemalloc) {PETSCFREE(mat->i);PETSCFREE(mat->j);}
        aa = mat->a = new_a; ai = mat->i = new_i; aj = mat->j = new_j; 
        mat->singlemalloc = 1;

        rp   = aj + ai[row] + shift; ap = aa + ai[row] + shift;
        rmax = imax[row] = imax[row] + CHUNCKSIZE;
        PLogObjectMemory(matin,CHUNCKSIZE*(sizeof(int) + sizeof(Scalar)));
        mat->maxnz += CHUNCKSIZE;
      }
      N = nrow++ - 1; mat->nz++;
      /* this has too many shifts here; but alternative was slower*/
      for ( ii=N; ii>=i; ii-- ) {/* shift values up*/
        rp[ii+1] = rp[ii];
        ap[ii+1] = ap[ii];
      }
      rp[i] = col; 
      ap[i] = value; 
      noinsert:;
      a = i + 1;
    }
    ailen[row] = nrow;
  }
  return 0;
} 

#include "draw.h"
#include "pinclude/pviewer.h"

static int MatView_SeqAIJ(PetscObject obj,Viewer ptr)
{
  Mat         aijin = (Mat) obj;
  Mat_SeqAIJ     *aij = (Mat_SeqAIJ *) aijin->data;
  int         ierr, i,j, m = aij->m;
  PetscObject vobj = (PetscObject) ptr;
  double      xl,yl,xr,yr,w,h;
   int shift = aij->indexshift;

  if (!aij->assembled) 
    SETERRQ(1,"MatView_SeqAIJ:Cannot view unassembled matrix");
  if (!ptr) { /* so that viewers may be used from debuggers */
    ptr = STDOUT_VIEWER_SELF; vobj = (PetscObject) ptr;
  }
  if (vobj->cookie == DRAW_COOKIE && vobj->type == NULLWINDOW) return 0;
  if (vobj && vobj->cookie == VIEWER_COOKIE && vobj->type == MATLAB_VIEWER) {
    return ViewerMatlabPutSparse_Private(ptr,aij->m,aij->n,aij->nz,aij->a,
                                 aij->i,aij->j); 
  }
  if (vobj && vobj->cookie == DRAW_COOKIE) {
    DrawCtx draw = (DrawCtx) ptr;
    xr = aij->n; yr = aij->m; h = yr/10.0; w = xr/10.0; 
    xr += w; yr += h; xl = -w; yl = -h;
    ierr = DrawSetCoordinates(draw,xl,yl,xr,yr); CHKERRQ(ierr);
    /* loop over matrix elements drawing boxes */
    for ( i=0; i<m; i++ ) {
      yl = m - i - 1.0; yr = yl + 1.0;
      for ( j=aij->i[i]+shift; j<aij->i[i+1]+shift; j++ ) {
        xl = aij->j[j] + shift; xr = xl + 1.0;
        DrawRectangle(draw,xl,yl,xr,yr,DRAW_BLACK,DRAW_BLACK,DRAW_BLACK,
                                                             DRAW_BLACK);
      } 
    }
    DrawFlush(draw); 
    return 0;
  }
  else {
    FILE *fd;
    char *outputname;
    int format;

    ierr = ViewerFileGetPointer_Private(ptr,&fd); CHKERRQ(ierr);
    ierr = ViewerFileGetOutputname_Private(ptr,&outputname); CHKERRQ(ierr);
    ierr = ViewerFileGetFormat_Private(ptr,&format);
    if (format == FILE_FORMAT_INFO) {
      /* do nothing for now */
    }
    else if (format == FILE_FORMAT_MATLAB) {
      int nz, nzalloc, mem;
      MatGetInfo(aijin,MAT_LOCAL,&nz,&nzalloc,&mem);
      fprintf(fd,"%% Size = %d %d \n",m,aij->n);
      fprintf(fd,"%% Nonzeros = %d \n",nz);
      fprintf(fd,"zzz = zeros(%d,3);\n",nz);
      fprintf(fd,"zzz = [\n");

      for (i=0; i<m; i++) {
        for ( j=aij->i[i]+shift; j<aij->i[i+1]+shift; j++ ) {
#if defined(PETSC_COMPLEX)
          fprintf(fd,"%d %d  %18.16e  %18.16e \n",
               i+1,aij->j[j],real(aij->a[j]),imag(aij->a[j]));
#else
          fprintf(fd,"%d %d  %18.16e\n", i+1, aij->j[j], aij->a[j]);
#endif
        }
      }
      fprintf(fd,"];\n %s = spconvert(zzz);\n",outputname);
    } 
    else {
      for ( i=0; i<m; i++ ) {
        fprintf(fd,"row %d:",i);
        for ( j=aij->i[i]+shift; j<aij->i[i+1]+shift; j++ ) {
#if defined(PETSC_COMPLEX)
          if (imag(aij->a[j]) != 0.0) {
            fprintf(fd," %d %g + %g i",
              aij->j[j]+shift,real(aij->a[j]),imag(aij->a[j]));
          }
          else {
            fprintf(fd," %d %g ",aij->j[j]+shift,real(aij->a[j]));
          }
#else
          fprintf(fd," %d %g ",aij->j[j]+shift,aij->a[j]);
#endif
        }
        fprintf(fd,"\n");
      }
    }
    fflush(fd);
  }
  return 0;
}

static int MatAssemblyEnd_SeqAIJ(Mat aijin,MatAssemblyType mode)
{
  Mat_SeqAIJ *aij = (Mat_SeqAIJ *) aijin->data;
  int    fshift = 0,i,j,*ai = aij->i, *aj = aij->j, *imax = aij->imax;
  int    m = aij->m, *ip, N, *ailen = aij->ilen;
  Scalar *a = aij->a, *ap;
   int shift = aij->indexshift;

  if (mode == FLUSH_ASSEMBLY) return 0;

  for ( i=1; i<m; i++ ) {
    fshift += imax[i-1] - ailen[i-1];
    if (fshift) {
      ip = aj + ai[i] + shift; ap = a + ai[i] + shift;
      N = ailen[i];
      for ( j=0; j<N; j++ ) {
        ip[j-fshift] = ip[j];
        ap[j-fshift] = ap[j]; 
      }
    } 
    ai[i] = ai[i-1] + ailen[i-1];
  }
  if (m) {
    fshift += imax[m-1] - ailen[m-1];
    ai[m] = ai[m-1] + ailen[m-1];
  }
  /* reset ilen and imax for each row */
  for ( i=0; i<m; i++ ) {
    ailen[i] = imax[i] = ai[i+1] - ai[i];
  }
  aij->nz = ai[m] + shift; 

  /* diagonals may have moved, so kill the diagonal pointers */
  if (fshift && aij->diag) {
    PETSCFREE(aij->diag);
    PLogObjectMemory(aijin,-(m+1)*sizeof(int));
    aij->diag = 0;
  } 
  aij->assembled = 1;
  return 0;
}

static int MatZeroEntries_SeqAIJ(Mat mat)
{
  Mat_SeqAIJ *aij = (Mat_SeqAIJ *) mat->data; 
   int shift = aij->indexshift;
  Scalar  *a = aij->a;
  int     i,n = aij->i[aij->m]+shift;

  for ( i=0; i<n; i++ ) a[i] = 0.0;
  return 0;

}
int MatDestroy_SeqAIJ(PetscObject obj)
{
  Mat mat         = (Mat) obj;
  Mat_SeqAIJ *aij = (Mat_SeqAIJ *) mat->data;
#if defined(PETSC_LOG)
  PLogObjectState(obj,"Rows=%d, Cols=%d, NZ=%d",aij->m,aij->n,aij->nz);
#endif
  PETSCFREE(aij->a); 
  if (!aij->singlemalloc) { PETSCFREE(aij->i); PETSCFREE(aij->j);}
  if (aij->diag) PETSCFREE(aij->diag);
  if (aij->ilen) PETSCFREE(aij->ilen);
  if (aij->imax) PETSCFREE(aij->imax);
  if (aij->solve_work) PETSCFREE(aij->solve_work);
  PETSCFREE(aij); 
  PLogObjectDestroy(mat);
  PETSCHEADERDESTROY(mat);
  return 0;
}

static int MatCompress_SeqAIJ(Mat aijin)
{
  return 0;
}

static int MatSetOption_SeqAIJ(Mat aijin,MatOption op)
{
  Mat_SeqAIJ *aij = (Mat_SeqAIJ *) aijin->data;
  if      (op == ROW_ORIENTED)              aij->roworiented = 1;
  else if (op == COLUMN_ORIENTED)           aij->roworiented = 0;
  else if (op == COLUMNS_SORTED)            aij->sorted      = 1;
  /* doesn't care about sorted rows */
  else if (op == NO_NEW_NONZERO_LOCATIONS)  aij->nonew       = 1;
  else if (op == YES_NEW_NONZERO_LOCATIONS) aij->nonew       = 0;

  if (op == COLUMN_ORIENTED) 
    SETERRQ(1,"MatSetOption_SeqAIJ:Column oriented input not supported");
  return 0;
}

static int MatGetDiagonal_SeqAIJ(Mat aijin,Vec v)
{
  Mat_SeqAIJ *aij = (Mat_SeqAIJ *) aijin->data;
  int    i,j, n;
  Scalar *x, zero = 0.0;
   int shift = aij->indexshift;

  if (!aij->assembled) SETERRQ(1,
    "MatGetDiagonal_SeqAIJ:Cannot get diagonal of unassembled matrix");
  VecSet(&zero,v);
  VecGetArray(v,&x); VecGetLocalSize(v,&n);
  if (n != aij->m) 
    SETERRQ(1,"MatGetDiagonal_SeqAIJ:Nonconforming matrix and vector");
  for ( i=0; i<aij->m; i++ ) {
    for ( j=aij->i[i]+shift; j<aij->i[i+1]+shift; j++ ) {
      if (aij->j[j]+shift == i) {
        x[i] = aij->a[j];
        break;
      }
    }
  }
  return 0;
}

/* -------------------------------------------------------*/
/* Should check that shapes of vectors and matrices match */
/* -------------------------------------------------------*/
static int MatMultTrans_SeqAIJ(Mat aijin,Vec xx,Vec yy)
{
  Mat_SeqAIJ *aij = (Mat_SeqAIJ *) aijin->data;
  Scalar  *x, *y, *v, alpha;
  int     m = aij->m, n, i, *idx;
   int shift = aij->indexshift;

  if (!aij->assembled) 
    SETERRQ(1,"MatMultTrans_SeqAIJ:Cannot multiply unassembled matrix");
  VecGetArray(xx,&x); VecGetArray(yy,&y);
  PETSCMEMSET(y,0,aij->n*sizeof(Scalar));
  y = y + shift; /* shift for Fortran start by 1 indexing */
  for ( i=0; i<m; i++ ) {
    idx   = aij->j + aij->i[i] + shift;
    v     = aij->a + aij->i[i] + shift;
    n     = aij->i[i+1] - aij->i[i];
    alpha = x[i];
    /* should inline */
    while (n-->0) {y[*idx++] += alpha * *v++;}
  }
  PLogFlops(2*aij->nz - aij->n);
  return 0;
}
static int MatMultTransAdd_SeqAIJ(Mat aijin,Vec xx,Vec zz,Vec yy)
{
  Mat_SeqAIJ *aij = (Mat_SeqAIJ *) aijin->data;
  Scalar  *x, *y, *v, alpha;
  int     m = aij->m, n, i, *idx;
   int shift = aij->indexshift;

  if (!aij->assembled) 
    SETERRQ(1,"MatMultTransAdd_SeqAIJ:Cannot multiply unassembled matrix");
  VecGetArray(xx,&x); VecGetArray(yy,&y);
  if (zz != yy) VecCopy(zz,yy);
  y = y + shift; /* shift for Fortran start by 1 indexing */
  for ( i=0; i<m; i++ ) {
    idx   = aij->j + aij->i[i] + shift;
    v     = aij->a + aij->i[i] + shift;
    n     = aij->i[i+1] - aij->i[i];
    alpha = x[i];
    /* should inline */
    while (n-->0) {y[*idx++] += alpha * *v++;}
  }
  return 0;
}

static int MatMult_SeqAIJ(Mat aijin,Vec xx,Vec yy)
{
  Mat_SeqAIJ *aij = (Mat_SeqAIJ *) aijin->data;
  Scalar     *x, *y, *v, sum;
  int        m = aij->m, n, i, *idx;
  int        shift = aij->indexshift,ii;

  if (!aij->assembled) 
    SETERRQ(1,"MatMult_SeqAIJ:Cannot multiply unassembled matrix");
  VecGetArray(xx,&x); VecGetArray(yy,&y);
  x = x + shift; /* shift for Fortran start by 1 indexing */
  for ( i=0; i<m; i++ ) {
    idx  = aij->j + aij->i[i] + shift;
    v    = aij->a + aij->i[i] + shift;
    n    = aij->i[i+1] - aij->i[i];
    sum  = 0.0;
    /* SPARSEDENSEDOT(sum,x,v,idx,n);  */
    while (n--) sum += *v++ * x[*idx++];
    /* for ( j=n-1; j>-1; j--) sum += v[j]*x[idx[j]]; */
    y[i] = sum;
  }
  PLogFlops(2*aij->nz - m);
  return 0;
}

static int MatMultAdd_SeqAIJ(Mat aijin,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqAIJ *aij = (Mat_SeqAIJ *) aijin->data;
  Scalar  *x, *y, *z, *v, sum;
  int     m = aij->m, n, i, *idx;
   int shift = aij->indexshift;

  if (!aij->assembled) 
    SETERRQ(1,"MatMultAdd_SeqAIJ:Cannot multiply unassembled matrix");
  VecGetArray(xx,&x); VecGetArray(yy,&y); VecGetArray(zz,&z); 
  x = x + shift; /* shift for Fortran start by 1 indexing */
  for ( i=0; i<m; i++ ) {
    idx  = aij->j + aij->i[i] + shift;
    v    = aij->a + aij->i[i] + shift;
    n    = aij->i[i+1] - aij->i[i];
    sum  = y[i];
    SPARSEDENSEDOT(sum,x,v,idx,n); 
    z[i] = sum;
  }
  PLogFlops(2*aij->nz);
  return 0;
}

/*
     Adds diagonal pointers to sparse matrix structure.
*/

int MatMarkDiag_SeqAIJ(Mat mat)
{
   Mat_SeqAIJ *aij = (Mat_SeqAIJ *) mat->data; 
  int    i,j, *diag, m = aij->m;
   int shift = aij->indexshift;

  if (!aij->assembled) SETERRQ(1,"MatMarkDiag_SeqAIJ:unassembled matrix");
  diag = (int *) PETSCMALLOC( (m+1)*sizeof(int)); CHKPTRQ(diag);
  PLogObjectMemory(mat,(m+1)*sizeof(int));
  for ( i=0; i<aij->m; i++ ) {
    for ( j=aij->i[i]+shift; j<aij->i[i+1]+shift; j++ ) {
      if (aij->j[j]+shift == i) {
        diag[i] = j - shift;
        break;
      }
    }
  }
  aij->diag = diag;
  return 0;
}

static int MatRelax_SeqAIJ(Mat matin,Vec bb,double omega,MatSORType flag,
                        double fshift,int its,Vec xx)
{
  Mat_SeqAIJ *mat = (Mat_SeqAIJ *) matin->data;
  Scalar *x, *b, *bs,  d, *xs, sum, *v = mat->a,*t,scale,*ts, *xb;
  int    ierr, *idx, *diag;
  int    n = mat->n, m = mat->m, i;
   int shift = mat->indexshift;

  VecGetArray(xx,&x); VecGetArray(bb,&b);
  if (!mat->diag) {if ((ierr = MatMarkDiag_SeqAIJ(matin))) return ierr;}
  diag = mat->diag;
  xs = x + shift; /* shifted by one for index start of a or mat->j*/
  if (flag == SOR_APPLY_UPPER) {
   /* apply ( U + D/omega) to the vector */
    bs = b + shift;
    for ( i=0; i<m; i++ ) {
        d    = fshift + mat->a[diag[i] + shift];
        n    = mat->i[i+1] - diag[i] - 1;
        idx  = mat->j + diag[i] + (!shift);
        v    = mat->a + diag[i] + (!shift);
        sum  = b[i]*d/omega;
        SPARSEDENSEDOT(sum,bs,v,idx,n); 
        x[i] = sum;
    }
    return 0;
  }
  if (flag == SOR_APPLY_LOWER) {
    SETERRQ(1,"MatRelax_SeqAIJ:SOR_APPLY_LOWER is not done");
  }
  if (flag & SOR_EISENSTAT) {
    /* Let  A = L + U + D; where L is lower trianglar,
    U is upper triangular, E is diagonal; This routine applies

            (L + E)^{-1} A (U + E)^{-1}

    to a vector efficiently using Eisenstat's trick. This is for
    the case of SSOR preconditioner, so E is D/omega where omega
    is the relaxation factor.
    */
    t = (Scalar *) PETSCMALLOC( m*sizeof(Scalar) ); CHKPTRQ(t);
    scale = (2.0/omega) - 1.0;

    /*  x = (E + U)^{-1} b */
    for ( i=m-1; i>=0; i-- ) {
      d    = fshift + mat->a[diag[i] + shift];
      n    = mat->i[i+1] - diag[i] - 1;
      idx  = mat->j + diag[i] + (!shift);
      v    = mat->a + diag[i] + (!shift);
      sum  = b[i];
      SPARSEDENSEMDOT(sum,xs,v,idx,n); 
      x[i] = omega*(sum/d);
    }

    /*  t = b - (2*E - D)x */
    v = mat->a;
    for ( i=0; i<m; i++ ) { t[i] = b[i] - scale*(v[*diag++ + shift])*x[i]; }

    /*  t = (E + L)^{-1}t */
    ts = t + shift; /* shifted by one for index start of a or mat->j*/
    diag = mat->diag;
    for ( i=0; i<m; i++ ) {
      d    = fshift + mat->a[diag[i]+shift];
      n    = diag[i] - mat->i[i];
      idx  = mat->j + mat->i[i] + shift;
      v    = mat->a + mat->i[i] + shift;
      sum  = t[i];
      SPARSEDENSEMDOT(sum,ts,v,idx,n); 
      t[i] = omega*(sum/d);
    }

    /*  x = x + t */
    for ( i=0; i<m; i++ ) { x[i] += t[i]; }
    PETSCFREE(t);
    return 0;
  }
  if (flag & SOR_ZERO_INITIAL_GUESS) {
    if (flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP){
      for ( i=0; i<m; i++ ) {
        d    = fshift + mat->a[diag[i]+shift];
        n    = diag[i] - mat->i[i];
        idx  = mat->j + mat->i[i] + shift;
        v    = mat->a + mat->i[i] + shift;
        sum  = b[i];
        SPARSEDENSEMDOT(sum,xs,v,idx,n); 
        x[i] = omega*(sum/d);
      }
      xb = x;
    }
    else xb = b;
    if ((flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP) && 
        (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP)) {
      for ( i=0; i<m; i++ ) {
        x[i] *= mat->a[diag[i]+shift];
      }
    }
    if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP){
      for ( i=m-1; i>=0; i-- ) {
        d    = fshift + mat->a[diag[i] + shift];
        n    = mat->i[i+1] - diag[i] - 1;
        idx  = mat->j + diag[i] + (!shift);
        v    = mat->a + diag[i] + (!shift);
        sum  = xb[i];
        SPARSEDENSEMDOT(sum,xs,v,idx,n); 
        x[i] = omega*(sum/d);
      }
    }
    its--;
  }
  while (its--) {
    if (flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP){
      for ( i=0; i<m; i++ ) {
        d    = fshift + mat->a[diag[i]+shift];
        n    = mat->i[i+1] - mat->i[i]; 
        idx  = mat->j + mat->i[i] + shift;
        v    = mat->a + mat->i[i] + shift;
        sum  = b[i];
        SPARSEDENSEMDOT(sum,xs,v,idx,n); 
        x[i] = (1. - omega)*x[i] + omega*(sum/d + x[i]);
      }
    }
    if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP){
      for ( i=m-1; i>=0; i-- ) {
        d    = fshift + mat->a[diag[i] + shift];
        n    = mat->i[i+1] - mat->i[i]; 
        idx  = mat->j + mat->i[i] + shift;
        v    = mat->a + mat->i[i] + shift;
        sum  = b[i];
        SPARSEDENSEMDOT(sum,xs,v,idx,n); 
        x[i] = (1. - omega)*x[i] + omega*(sum/d + x[i]);
      }
    }
  }
  return 0;
} 

static int MatGetInfo_SeqAIJ(Mat matin,MatInfoType flag,int *nz,
                          int *nzalloc,int *mem)
{
  Mat_SeqAIJ *mat = (Mat_SeqAIJ *) matin->data;
  *nz      = mat->nz;
  *nzalloc = mat->maxnz;
  *mem     = (int)matin->mem;
  return 0;
}

extern int MatLUFactorSymbolic_SeqAIJ(Mat,IS,IS,double,Mat*);
extern int MatLUFactorNumeric_SeqAIJ(Mat,Mat*);
extern int MatLUFactor_SeqAIJ(Mat,IS,IS,double);
extern int MatSolve_SeqAIJ(Mat,Vec,Vec);
extern int MatSolveAdd_SeqAIJ(Mat,Vec,Vec,Vec);
extern int MatSolveTrans_SeqAIJ(Mat,Vec,Vec);
extern int MatSolveTransAdd_SeqAIJ(Mat,Vec,Vec,Vec);

static int MatZeroRows_SeqAIJ(Mat A,IS is,Scalar *diag)
{
  Mat_SeqAIJ *l = (Mat_SeqAIJ *) A->data;
  int     i,ierr,N, *rows,m = l->m - 1;
   int shift = l->indexshift;

  ierr = ISGetLocalSize(is,&N); CHKERRQ(ierr);
  ierr = ISGetIndices(is,&rows); CHKERRQ(ierr);
  if (diag) {
    for ( i=0; i<N; i++ ) {
      if (rows[i] < 0 || rows[i] > m) 
        SETERRQ(1,"MatZeroRows_SeqAIJ:row out of range");
      if (l->ilen[rows[i]] > 0) { /* in case row was completely empty */
        l->ilen[rows[i]] = 1; 
        l->a[l->i[rows[i]]+shift] = *diag;
        l->j[l->i[rows[i]]+shift] = rows[i]+shift;
      }
      else {
        ierr = MatSetValues_SeqAIJ(A,1,&rows[i],1,&rows[i],diag,INSERT_VALUES);
        CHKERRQ(ierr);
      }
    }
  }
  else {
    for ( i=0; i<N; i++ ) {
      if (rows[i] < 0 || rows[i] > m) 
        SETERRQ(1,"MatZeroRows_SeqAIJ:row out of range");
      l->ilen[rows[i]] = 0; 
    }
  }
  ISRestoreIndices(is,&rows);
  ierr = MatAssemblyBegin(A,FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,FINAL_ASSEMBLY); CHKERRQ(ierr);
  return 0;
}

static int MatGetSize_SeqAIJ(Mat matin,int *m,int *n)
{
  Mat_SeqAIJ *mat = (Mat_SeqAIJ *) matin->data;
  *m = mat->m; *n = mat->n;
  return 0;
}

static int MatGetOwnershipRange_SeqAIJ(Mat matin,int *m,int *n)
{
  Mat_SeqAIJ *mat = (Mat_SeqAIJ *) matin->data;
  *m = 0; *n = mat->m;
  return 0;
}
static int MatGetRow_SeqAIJ(Mat matin,int row,int *nz,int **idx,Scalar **v)
{
  Mat_SeqAIJ *mat = (Mat_SeqAIJ *) matin->data;
  int     *itmp,i,ierr;
   int shift = mat->indexshift;

  if (row < 0 || row >= mat->m) SETERRQ(1,"MatGetRow_SeqAIJ:Row out of range");

  if (!mat->assembled) {
    ierr = MatAssemblyBegin(matin,FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(matin,FINAL_ASSEMBLY); CHKERRQ(ierr);
  }
  *nz = mat->i[row+1] - mat->i[row];
  if (v) *v = mat->a + mat->i[row] + shift;
  if (idx) {
    if (*nz) {
      itmp = mat->j + mat->i[row] + shift;
      *idx = (int *) PETSCMALLOC( (*nz)*sizeof(int) ); CHKPTRQ(*idx);
      for ( i=0; i<(*nz); i++ ) {(*idx)[i] = itmp[i] + shift;}
    }
    else *idx = 0;
  }
  return 0;
}

static int MatRestoreRow_SeqAIJ(Mat matin,int row,int *nz,int **idx,Scalar **v)
{
  if (idx) {if (*idx) PETSCFREE(*idx);}
  return 0;
}

static int MatNorm_SeqAIJ(Mat matin,MatNormType type,double *norm)
{
  Mat_SeqAIJ *mat = (Mat_SeqAIJ *) matin->data;
  Scalar  *v = mat->a;
  double  sum = 0.0;
  int     i, j;
   int shift = mat->indexshift;

  if (!mat->assembled) 
    SETERRQ(1,"MatNorm_SeqAIJ:Cannot compute norm of unassembled matrix");
  if (type == NORM_FROBENIUS) {
    for (i=0; i<mat->nz; i++ ) {
#if defined(PETSC_COMPLEX)
      sum += real(conj(*v)*(*v)); v++;
#else
      sum += (*v)*(*v); v++;
#endif
    }
    *norm = sqrt(sum);
  }
  else if (type == NORM_1) {
    double *tmp;
    int    *jj = mat->j;
    tmp = (double *) PETSCMALLOC( mat->n*sizeof(double) ); CHKPTRQ(tmp);
    PETSCMEMSET(tmp,0,mat->n*sizeof(double));
    *norm = 0.0;
    for ( j=0; j<mat->nz; j++ ) {
#if defined(PETSC_COMPLEX)
        tmp[*jj++ + shift] += abs(*v++); 
#else
        tmp[*jj++ + shift] += fabs(*v++); 
#endif
    }
    for ( j=0; j<mat->n; j++ ) {
      if (tmp[j] > *norm) *norm = tmp[j];
    }
    PETSCFREE(tmp);
  }
  else if (type == NORM_INFINITY) {
    *norm = 0.0;
    for ( j=0; j<mat->m; j++ ) {
      v = mat->a + mat->i[j] + shift;
      sum = 0.0;
      for ( i=0; i<mat->i[j+1]-mat->i[j]; i++ ) {
#if defined(PETSC_COMPLEX)
        sum += abs(*v); v++;
#else
        sum += fabs(*v); v++;
#endif
      }
      if (sum > *norm) *norm = sum;
    }
  }
  else {
    SETERRQ(1,"MatNorm_SeqAIJ:No support for the two norm yet");
  }
  return 0;
}

static int MatTranspose_SeqAIJ(Mat A,Mat *matout)
{ 
  Mat_SeqAIJ *amat = (Mat_SeqAIJ *) A->data;
  Mat     tmat;
  int     i, ierr, *aj = amat->j, *ai = amat->i, m = amat->m, len, *col;
  Scalar  *array = amat->a;
   int shift = amat->indexshift;

  if (!matout && m != amat->n) SETERRQ(1,
    "MatTranspose_SeqAIJ: Cannot transpose rectangular matrix in place");
  col = (int *) PETSCMALLOC((1+amat->n)*sizeof(int)); CHKPTRQ(col);
  PETSCMEMSET(col,0,(1+amat->n)*sizeof(int));
  if (shift) {
    for ( i=0; i<ai[m]-1; i++ ) aj[i] -= 1;
  }
  for ( i=0; i<ai[m]+shift; i++ ) col[aj[i]] += 1;
  ierr = MatCreateSeqAIJ(A->comm,amat->n,m,0,col,&tmat); CHKERRQ(ierr);
  PETSCFREE(col);
  for ( i=0; i<m; i++ ) {
    len = ai[i+1]-ai[i];
    ierr = MatSetValues(tmat,len,aj,1,&i,array,INSERT_VALUES); CHKERRQ(ierr);
    array += len; aj += len;
  }
  if (shift) { 
    for ( i=0; i<ai[m]-1; i++ ) aj[i] += 1;
  }

  ierr = MatAssemblyBegin(tmat,FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(tmat,FINAL_ASSEMBLY); CHKERRQ(ierr);

  if (matout) {
    *matout = tmat;
  } else {
    /* This isn't really an in-place transpose ... but free data structures from amat */
    PETSCFREE(amat->a); 
    if (!amat->singlemalloc) {PETSCFREE(amat->i); PETSCFREE(amat->j);}
    if (amat->diag) PETSCFREE(amat->diag);
    if (amat->ilen) PETSCFREE(amat->ilen);
    if (amat->imax) PETSCFREE(amat->imax);
    if (amat->solve_work) PETSCFREE(amat->solve_work);
    PETSCFREE(amat); 
    PETSCMEMCPY(A,tmat,sizeof(struct _Mat)); 
    PETSCHEADERDESTROY(tmat);
  }
  return 0;
}

static int MatScale_SeqAIJ(Mat matin,Vec ll,Vec rr)
{
  Mat_SeqAIJ *mat = (Mat_SeqAIJ *) matin->data;
  Scalar  *l,*r,x,*v;
  int     i,j,m = mat->m, n = mat->n, M, nz = mat->nz, *jj;
   int shift = mat->indexshift;

  if (!mat->assembled) 
    SETERRQ(1,"MatScale_SeqAIJ:Cannot scale unassembled matrix");
  if (ll) {
    VecGetArray(ll,&l); VecGetSize(ll,&m);
    if (m != mat->m) 
      SETERRQ(1,"MatScale_SeqAIJ:Left scaling vector wrong length");
    v = mat->a;
    for ( i=0; i<m; i++ ) {
      x = l[i];
      M = mat->i[i+1] - mat->i[i];
      for ( j=0; j<M; j++ ) { (*v++) *= x;} 
    }
  }
  if (rr) {
    VecGetArray(rr,&r); VecGetSize(rr,&n);
    if (n != mat->n) 
      SETERRQ(1,"MatScale_SeqAIJ:Right scaling vector wrong length");
    v = mat->a; jj = mat->j;
    for ( i=0; i<nz; i++ ) {
      (*v++) *= r[*jj++ + shift]; 
    }
  }
  return 0;
}

static int MatGetSubMatrix_SeqAIJ(Mat matin,IS isrow,IS iscol,Mat *submat)
{
  Mat_SeqAIJ *mat = (Mat_SeqAIJ *) matin->data;
  int        nznew, *smap, i, k, kstart, kend, ierr, oldcols = mat->n;
  int        *irow, *icol, nrows, ncols, *cwork;
  Scalar     *vwork;
  Mat        newmat;
  int        shift = mat->indexshift;

  if (!mat->assembled) SETERRQ(1,
    "MatGetSubMatrix_SeqAIJ:Cannot extract submatrix from unassembled matrix");  
  ierr = ISGetIndices(isrow,&irow); CHKERRQ(ierr);
  ierr = ISGetIndices(iscol,&icol); CHKERRQ(ierr);
  ierr = ISGetSize(isrow,&nrows); CHKERRQ(ierr);
  ierr = ISGetSize(iscol,&ncols); CHKERRQ(ierr);

  smap = (int *) PETSCMALLOC((1+oldcols)*sizeof(int)); CHKPTRQ(smap);
  cwork = (int *) PETSCMALLOC((1+ncols)*sizeof(int)); CHKPTRQ(cwork);
  vwork = (Scalar *) PETSCMALLOC((1+ncols)*sizeof(Scalar)); CHKPTRQ(vwork);
  PETSCMEMSET(smap,0,oldcols*sizeof(int));
  for ( i=0; i<ncols; i++ ) smap[icol[i]] = i+1;

  /* Create and fill new matrix */
  ierr = MatCreateSeqAIJ(matin->comm,nrows,ncols,0,0,&newmat);
         CHKERRQ(ierr);
  for (i=0; i<nrows; i++) {
    nznew = 0;
    kstart = mat->i[irow[i]]+shift; 
    kend = kstart + mat->ilen[irow[i]];
    for ( k=kstart; k<kend; k++ ) {
      if (smap[mat->j[k]+shift]) {
        cwork[nznew]   = smap[mat->j[k]+shift] - 1;
        vwork[nznew++] = mat->a[k];
      }
    }
    ierr = MatSetValues(newmat,1,&i,nznew,cwork,vwork,INSERT_VALUES); 
           CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(newmat,FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(newmat,FINAL_ASSEMBLY); CHKERRQ(ierr);

  /* Free work space */
  PETSCFREE(smap); PETSCFREE(cwork); PETSCFREE(vwork);
  ierr = ISRestoreIndices(isrow,&irow); CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol,&icol); CHKERRQ(ierr);
  *submat = newmat;
  return 0;
}

static int MatGetSubMatrixInPlace_SeqAIJ(Mat matin,IS rows,IS cols)
{
  /* Mat_SeqAIJ *mat = (Mat_SeqAIJ *) matin->data; */
 
  SETERRQ(1,"MatGetSubMatrixInPlace_SeqAIJ:not finished");
}

/* -------------------------------------------------------------------*/
extern int MatILUFactorSymbolic_SeqAIJ(Mat,IS,IS,double,int,Mat *);
extern int MatConvert_SeqAIJ(Mat,MatType,Mat *);
static int MatCopyPrivate_SeqAIJ(Mat,Mat *);

static struct _MatOps MatOps = {MatSetValues_SeqAIJ,
       MatGetRow_SeqAIJ,MatRestoreRow_SeqAIJ,
       MatMult_SeqAIJ,MatMultAdd_SeqAIJ,MatMultTrans_SeqAIJ,MatMultTransAdd_SeqAIJ,
       MatSolve_SeqAIJ,MatSolveAdd_SeqAIJ,
       MatSolveTrans_SeqAIJ,MatSolveTransAdd_SeqAIJ,
       MatLUFactor_SeqAIJ,0,
       MatRelax_SeqAIJ,
       MatTranspose_SeqAIJ,
       MatGetInfo_SeqAIJ,0,
       MatGetDiagonal_SeqAIJ,MatScale_SeqAIJ,MatNorm_SeqAIJ,
       0,MatAssemblyEnd_SeqAIJ,
       MatCompress_SeqAIJ,
       MatSetOption_SeqAIJ,MatZeroEntries_SeqAIJ,MatZeroRows_SeqAIJ,
       MatGetReordering_SeqAIJ,
       MatLUFactorSymbolic_SeqAIJ,MatLUFactorNumeric_SeqAIJ,0,0,
       MatGetSize_SeqAIJ,MatGetSize_SeqAIJ,MatGetOwnershipRange_SeqAIJ,
       MatILUFactorSymbolic_SeqAIJ,0,
       0,0,MatConvert_SeqAIJ,
       MatGetSubMatrix_SeqAIJ,MatGetSubMatrixInPlace_SeqAIJ,
       MatCopyPrivate_SeqAIJ};

extern int MatUseSuperLU_SeqAIJ(Mat);
extern int MatUseEssl_SeqAIJ(Mat);
extern int MatUseDXML_SeqAIJ(Mat);

/*@C
   MatCreateSeqAIJ - Creates a sparse matrix in AIJ format
   (the default uniprocessor PETSc format).

   Input Parameters:
.  comm - MPI communicator, set to MPI_COMM_SELF
.  m - number of rows
.  n - number of columns
.  nz - number of nonzeros per row (same for all rows)
.  nzz - number of nonzeros per row or null (possibly different for each row)

   Output Parameter:
.  newmat - the matrix 

   Notes:
   The AIJ format (also called the Yale sparse matrix format or
   compressed row storage), is fully compatible with standard Fortran 77
   storage.  That is, the stored row and column indices begin at 
   one, not zero.

   Specify the preallocated storage with either nz or nnz (not both).
   Set both nz and nnz to zero for PETSc to control dynamic memory 
   allocation.

.keywords: matrix, aij, compressed row, sparse

.seealso: MatCreate(), MatCreateMPIAIJ(), MatSetValues()
@*/
int MatCreateSeqAIJ(MPI_Comm comm,int m,int n,int nz,
                           int *nnz, Mat *newmat)
{
  Mat        mat;
  Mat_SeqAIJ *aij;
  int        i,len,ierr;
  *newmat      = 0;
  PETSCHEADERCREATE(mat,_Mat,MAT_COOKIE,MATSEQAIJ,comm);
  PLogObjectCreate(mat);
  mat->data             = (void *) (aij = PETSCNEW(Mat_SeqAIJ)); CHKPTRQ(aij);
  PETSCMEMCPY(&mat->ops,&MatOps,sizeof(struct _MatOps));
  mat->destroy          = MatDestroy_SeqAIJ;
  mat->view             = MatView_SeqAIJ;
  mat->factor           = 0;
  mat->lupivotthreshold = 1.0;
  OptionsGetDouble(0,"-mat_lu_pivotthreshold",&mat->lupivotthreshold);
  aij->row              = 0;
  aij->col              = 0;
  aij->indexshift       = 0;
  if (OptionsHasName(0,"-mat_aij_oneindex")) aij->indexshift = -1;

  aij->m       = m;
  aij->n       = n;
  aij->imax    = (int *) PETSCMALLOC( (m+1)*sizeof(int) ); CHKPTRQ(aij->imax);
  if (!nnz) {
    if (nz <= 0) nz = 1;
    for ( i=0; i<m; i++ ) aij->imax[i] = nz;
    nz = nz*m;
  }
  else {
    nz = 0;
    for ( i=0; i<m; i++ ) {aij->imax[i] = nnz[i]; nz += nnz[i];}
  }

  /* allocate the matrix space */
  len     = nz*(sizeof(int) + sizeof(Scalar)) + (aij->m+1)*sizeof(int);
  aij->a  = (Scalar *) PETSCMALLOC( len ); CHKPTRQ(aij->a);
  aij->j  = (int *) (aij->a + nz);
  PETSCMEMSET(aij->j,0,nz*sizeof(int));
  aij->i  = aij->j + nz;
  aij->singlemalloc = 1;

  aij->i[0] = -aij->indexshift;
  for (i=1; i<m+1; i++) {
    aij->i[i] = aij->i[i-1] + aij->imax[i-1];
  }

  /* aij->ilen will count nonzeros in each row so far. */
  aij->ilen = (int *) PETSCMALLOC((m+1)*sizeof(int)); 
  PLogObjectMemory(mat,len + 2*(m+1)*sizeof(int) + sizeof(struct _Mat)
                       + sizeof(Mat_SeqAIJ));
  for ( i=0; i<aij->m; i++ ) { aij->ilen[i] = 0;}

  aij->nz          = 0;
  aij->maxnz       = nz;
  aij->sorted      = 0;
  aij->roworiented = 1;
  aij->nonew       = 0;
  aij->diag        = 0;
  aij->assembled   = 0;
  aij->solve_work  = 0;

  *newmat = mat;
  if (OptionsHasName(0,"-mat_aij_superlu")) {
    ierr = MatUseSuperLU_SeqAIJ(mat); CHKERRQ(ierr);
  }
  if (OptionsHasName(0,"-mat_aij_essl")) {
    ierr = MatUseEssl_SeqAIJ(mat); CHKERRQ(ierr);
  }
  if (OptionsHasName(0,"-mat_aij_dxml")) {
    if (!aij->indexshift)
      SETERRQ(1,"MatCreateSeqAIJ: must use -mat_aij_oneindex with -mat_aij_dxml");
    ierr = MatUseDXML_SeqAIJ(mat); CHKERRQ(ierr);
  }

  return 0;
}

static int MatCopyPrivate_SeqAIJ(Mat matin,Mat *newmat)
{
  Mat     mat;
  Mat_SeqAIJ *aij,*oldmat = (Mat_SeqAIJ *) matin->data;
  int     i,len, m = oldmat->m;
  int     shift = oldmat->indexshift;
  *newmat      = 0;

  if (!oldmat->assembled) 
    SETERRQ(1,"MatCopyPrivate_SeqAIJ:Cannot copy unassembled matrix");
  PETSCHEADERCREATE(mat,_Mat,MAT_COOKIE,MATSEQAIJ,matin->comm);
  PLogObjectCreate(mat);
  mat->data       = (void *) (aij = PETSCNEW(Mat_SeqAIJ)); CHKPTRQ(aij);
  PETSCMEMCPY(&mat->ops,&MatOps,sizeof(struct _MatOps));
  mat->destroy    = MatDestroy_SeqAIJ;
  mat->view       = MatView_SeqAIJ;
  mat->factor     = matin->factor;
  aij->row        = 0;
  aij->col        = 0;
  aij->indexshift = shift;

  aij->m          = oldmat->m;
  aij->n          = oldmat->n;

  aij->imax       = (int *) PETSCMALLOC((m+1)*sizeof(int)); CHKPTRQ(aij->imax);
  aij->ilen       = (int *) PETSCMALLOC((m+1)*sizeof(int)); CHKPTRQ(aij->ilen);
  for ( i=0; i<m; i++ ) {
    aij->imax[i] = oldmat->imax[i];
    aij->ilen[i] = oldmat->ilen[i]; 
  }

  /* allocate the matrix space */
  aij->singlemalloc = 1;
  len     = (m+1)*sizeof(int)+(oldmat->i[m])*(sizeof(Scalar)+sizeof(int));
  aij->a  = (Scalar *) PETSCMALLOC( len ); CHKPTRQ(aij->a);
  aij->j  = (int *) (aij->a + oldmat->i[m] + shift);
  aij->i  = aij->j + oldmat->i[m] + shift;
  PETSCMEMCPY(aij->i,oldmat->i,(m+1)*sizeof(int));
  if (m > 0) {
    PETSCMEMCPY(aij->j,oldmat->j,(oldmat->i[m]+shift)*sizeof(int));
    PETSCMEMCPY(aij->a,oldmat->a,(oldmat->i[m]+shift)*sizeof(Scalar));
  }

  PLogObjectMemory(mat,len + 2*(m+1)*sizeof(int) + sizeof(struct _Mat)
                       + sizeof(Mat_SeqAIJ));  
  aij->sorted      = oldmat->sorted;
  aij->roworiented = oldmat->roworiented;
  aij->nonew       = oldmat->nonew;

  if (oldmat->diag) {
    aij->diag = (int *) PETSCMALLOC( (m+1)*sizeof(int) ); CHKPTRQ(aij->diag);
    PLogObjectMemory(mat,(m+1)*sizeof(int));
    for ( i=0; i<m; i++ ) {
      aij->diag[i] = oldmat->diag[i];
    }
  }
  else aij->diag        = 0;
  aij->assembled        = 1;
  aij->nz               = oldmat->nz;
  aij->maxnz            = oldmat->maxnz;
  aij->solve_work       = 0;
  *newmat = mat;
  return 0;
}

#include "sysio.h"

int MatLoad_SeqAIJ(Viewer bview,MatType type,Mat *newmat)
{
  Mat_SeqAIJ   *aij;
  Mat          mat;
  int          i, nz, ierr;
  int          fd;
  PetscObject  vobj = (PetscObject) bview;
  MPI_Comm     comm = vobj->comm;
  int          header[4],numtid,*rowlengths = 0,M,N;
  int          shift;

  MPI_Comm_size(comm,&numtid);
  if (numtid > 1) SETERRQ(1,"MatLoad_SeqAIJ: view must have one processor");
  ierr = ViewerFileGetDescriptor_Private(bview,&fd); CHKERRQ(ierr);
  ierr = SYRead(fd,(char *)header,4*sizeof(int),SYINT); CHKERRQ(ierr);
  if (header[0] != MAT_COOKIE) SETERRQ(1,"MatLoad_SeqAIJ: not matrix object");
  M = header[1]; N = header[2]; nz = header[3];

  /* read in row lengths */
  rowlengths = (int*) PETSCMALLOC( M*sizeof(int) ); CHKPTRQ(rowlengths);
  ierr = SYRead(fd,(char *)rowlengths,M*sizeof(int),SYINT); CHKERRQ(ierr);

  /* create our matrix */
  ierr = MatCreateSeqAIJ(comm,M,N,0,rowlengths,newmat); CHKERRQ(ierr);
  mat = *newmat;
  aij = (Mat_SeqAIJ *) mat->data;
  shift = aij->indexshift;

  /* read in column indices and adjust for Fortran indexing*/
  ierr = SYRead(fd,(char *)aij->j,nz*sizeof(int),SYINT); CHKERRQ(ierr);
  if (shift) {
    for ( i=0; i<nz; i++ ) {
      aij->j[i] += 1;
    }
  }

  /* read in nonzero values */
  ierr = SYRead(fd,(char *)aij->a,nz*sizeof(Scalar),SYSCALAR); CHKERRQ(ierr);

  /* set matrix "i" values */
  aij->i[0] = -shift;
  for ( i=1; i<= M; i++ ) {
    aij->i[i]      = aij->i[i-1] + rowlengths[i-1];
    aij->ilen[i-1] = rowlengths[i-1];
  }
  PETSCFREE(rowlengths);   

  ierr = MatAssemblyBegin(mat,FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,FINAL_ASSEMBLY); CHKERRQ(ierr);
  return 0;
}



