#ifndef lint
static char vcid[] = "$Id: aij.c,v 1.91 1995/09/22 03:13:08 bsmith Exp bsmith $";
#endif

#include "aij.h"
#include "vec/vecimpl.h"
#include "inline/spops.h"

extern int MatToSymmetricIJ_SeqAIJ(Mat_SeqAIJ*,int**,int**);

static int MatGetReordering_SeqAIJ(Mat A,MatOrdering type,IS *rperm, IS *cperm)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  int        ierr, *ia, *ja;

  if (!a->assembled) SETERRQ(1,"MatGetReordering_SeqAIJ:Not for unassembled matrix");

  ierr = MatToSymmetricIJ_SeqAIJ( a, &ia, &ja ); CHKERRQ(ierr);
  ierr = MatGetReordering_IJ(a->n,ia,ja,type,rperm,cperm); CHKERRQ(ierr);
  PETSCFREE(ia); PETSCFREE(ja);
  return 0; 
}

#define CHUNKSIZE   10

/* This version has row oriented v  */
static int MatSetValues_SeqAIJ(Mat A,int m,int *im,int n,int *in,Scalar *v,InsertMode is)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  int        *rp,k,low,high,t,ii,row,nrow,i,col,l,rmax, N, sorted = a->sorted;
  int        *imax = a->imax, *ai = a->i, *ailen = a->ilen;
  int        *aj = a->j, nonew = a->nonew;
  Scalar     *ap,value, *aa = a->a;
  int        shift = a->indexshift;

  for ( k=0; k<m; k++ ) { /* loop over added rows */
    row  = im[k];   
    if (row < 0) SETERRQ(1,"MatSetValues_SeqAIJ:Negative row");
    if (row >= a->m) SETERRQ(1,"MatSetValues_SeqAIJ:Row too large");
    rp   = aj + ai[row] + shift; ap = aa + ai[row] + shift;
    rmax = imax[row]; nrow = ailen[row]; 
    low = 0;
    for ( l=0; l<n; l++ ) { /* loop over added columns */
      if (in[l] < 0) SETERRQ(1,"MatSetValues_SeqAIJ:Negative column");
      if (in[l] >= a->n) SETERRQ(1,"MatSetValues_SeqAIJ:Column too large");
      col = in[l] - shift; value = *v++; 
      if (!sorted) low = 0; high = nrow;
      while (high-low > 5) {
        t = (low+high)/2;
        if (rp[t] > col) high = t;
        else             low  = t;
      }
      for ( i=low; i<high; i++ ) {
        if (rp[i] > col) break;
        if (rp[i] == col) {
          if (is == ADD_VALUES) ap[i] += value;  
          else                  ap[i] = value;
          goto noinsert;
        }
      } 
      if (nonew) goto noinsert;
      if (nrow >= rmax) {
        /* there is no extra room in row, therefore enlarge */
        int    new_nz = ai[a->m] + CHUNKSIZE,len,*new_i,*new_j;
        Scalar *new_a;

        /* malloc new storage space */
        len     = new_nz*(sizeof(int)+sizeof(Scalar))+(a->m+1)*sizeof(int);
        new_a   = (Scalar *) PETSCMALLOC( len ); CHKPTRQ(new_a);
        new_j   = (int *) (new_a + new_nz);
        new_i   = new_j + new_nz;

        /* copy over old data into new slots */
        for ( ii=0; ii<row+1; ii++ ) {new_i[ii] = ai[ii];}
        for ( ii=row+1; ii<a->m+1; ii++ ) {new_i[ii] = ai[ii]+CHUNKSIZE;}
        PetscMemcpy(new_j,aj,(ai[row]+nrow+shift)*sizeof(int));
        len = (new_nz - CHUNKSIZE - ai[row] - nrow - shift);
        PetscMemcpy(new_j+ai[row]+shift+nrow+CHUNKSIZE,aj+ai[row]+shift+nrow,
                                                           len*sizeof(int));
        PetscMemcpy(new_a,aa,(ai[row]+nrow+shift)*sizeof(Scalar));
        PetscMemcpy(new_a+ai[row]+shift+nrow+CHUNKSIZE,aa+ai[row]+shift+nrow,
                                                           len*sizeof(Scalar)); 
        /* free up old matrix storage */
        PETSCFREE(a->a); 
        if (!a->singlemalloc) {PETSCFREE(a->i);PETSCFREE(a->j);}
        aa = a->a = new_a; ai = a->i = new_i; aj = a->j = new_j; 
        a->singlemalloc = 1;

        rp   = aj + ai[row] + shift; ap = aa + ai[row] + shift;
        rmax = imax[row] = imax[row] + CHUNKSIZE;
        PLogObjectMemory(A,CHUNKSIZE*(sizeof(int) + sizeof(Scalar)));
        a->maxnz += CHUNKSIZE;
      }
      N = nrow++ - 1; a->nz++;
      /* shift up all the later entries in this row */
      for ( ii=N; ii>=i; ii-- ) {
        rp[ii+1] = rp[ii];
        ap[ii+1] = ap[ii];
      }
      rp[i] = col; 
      ap[i] = value; 
      noinsert:;
      low = i + 1;
    }
    ailen[row] = nrow;
  }
  return 0;
} 

#include "draw.h"
#include "pinclude/pviewer.h"
#include "sysio.h"

static int MatView_SeqAIJ_Binary(Mat A,Viewer viewer)
{
  Mat_SeqAIJ  *a = (Mat_SeqAIJ *) A->data;
  int         i,fd,*col_lens,ierr;

  ierr = ViewerFileGetDescriptor_Private(viewer,&fd); CHKERRQ(ierr);

  col_lens    = (int *) PETSCMALLOC( (4+a->nz)*sizeof(int) ); CHKPTRQ(col_lens);
  col_lens[0] = MAT_COOKIE;
  col_lens[1] = a->m;
  col_lens[2] = a->n;
  col_lens[3] = a->nz;

  /* store lengths of each row and write (including header) to file */
  for ( i=0; i<a->m; i++ ) {
    col_lens[4+i] = a->i[i+1] - a->i[i];
  }
  ierr = SYWrite(fd,col_lens,4+a->m,SYINT,1); CHKERRQ(ierr);
  PETSCFREE(col_lens);

  /* store column indices (zero start index) */
  if (a->indexshift) {
    for ( i=0; i<a->nz; i++ ) a->j[i]--;
  }
  ierr = SYWrite(fd,a->j,a->nz,SYINT,0); CHKERRQ(ierr);
  if (a->indexshift) {
    for ( i=0; i<a->nz; i++ ) a->j[i]++;
  }

  /* store nonzero values */
  ierr = SYWrite(fd,a->a,a->nz,SYSCALAR,0); CHKERRQ(ierr);
  return 0;
}

static int MatView_SeqAIJ_ASCII(Mat A,Viewer viewer)
{
  Mat_SeqAIJ  *a = (Mat_SeqAIJ *) A->data;
  int         ierr, i,j, m = a->m, shift = a->indexshift,format;
  FILE        *fd;
  char        *outputname;

  ierr = ViewerFileGetPointer_Private(viewer,&fd); CHKERRQ(ierr);
  ierr = ViewerFileGetOutputname_Private(viewer,&outputname); CHKERRQ(ierr);
  ierr = ViewerFileGetFormat_Private(viewer,&format);
  if (format == FILE_FORMAT_INFO) {
    ; /* do nothing for now */
  } 
  else if (format == FILE_FORMAT_MATLAB) {
    int nz, nzalloc, mem;
    MatGetInfo(A,MAT_LOCAL,&nz,&nzalloc,&mem);
    fprintf(fd,"%% Size = %d %d \n",m,a->n);
    fprintf(fd,"%% Nonzeros = %d \n",nz);
    fprintf(fd,"zzz = zeros(%d,3);\n",nz);
    fprintf(fd,"zzz = [\n");

    for (i=0; i<m; i++) {
      for ( j=a->i[i]+shift; j<a->i[i+1]+shift; j++ ) {
#if defined(PETSC_COMPLEX)
        fprintf(fd,"%d %d  %18.16e  %18.16e \n",i+1,a->j[j],real(a->a[j]),
                   imag(a->a[j]));
#else
        fprintf(fd,"%d %d  %18.16e\n", i+1, a->j[j], a->a[j]);
#endif
      }
    }
    fprintf(fd,"];\n %s = spconvert(zzz);\n",outputname);
  } 
  else {
    for ( i=0; i<m; i++ ) {
      fprintf(fd,"row %d:",i);
      for ( j=a->i[i]+shift; j<a->i[i+1]+shift; j++ ) {
#if defined(PETSC_COMPLEX)
        if (imag(a->a[j]) != 0.0) {
          fprintf(fd," %d %g + %g i",a->j[j]+shift,real(a->a[j]),imag(a->a[j]));
        }
        else {
          fprintf(fd," %d %g ",a->j[j]+shift,real(a->a[j]));
        }
#else
        fprintf(fd," %d %g ",a->j[j]+shift,a->a[j]);
#endif
      }
      fprintf(fd,"\n");
    }
  }
  fflush(fd);
  return 0;
}

static int MatView_SeqAIJ_Draw(Mat A,Viewer viewer)
{
  Mat_SeqAIJ  *a = (Mat_SeqAIJ *) A->data;
  int         ierr, i,j, m = a->m, shift = a->indexshift;
  double      xl,yl,xr,yr,w,h;
  DrawCtx draw = (DrawCtx) viewer;
  xr  = a->n; yr = a->m; h = yr/10.0; w = xr/10.0; 
  xr += w;    yr += h;  xl = -w;     yl = -h;
  ierr = DrawSetCoordinates(draw,xl,yl,xr,yr); CHKERRQ(ierr);
  /* loop over matrix elements drawing boxes */
  for ( i=0; i<m; i++ ) {
    yl = m - i - 1.0; yr = yl + 1.0;
    for ( j=a->i[i]+shift; j<a->i[i+1]+shift; j++ ) {
      xl = a->j[j] + shift; xr = xl + 1.0;
      DrawRectangle(draw,xl,yl,xr,yr,DRAW_BLACK,DRAW_BLACK,DRAW_BLACK,DRAW_BLACK);
    } 
  }
  DrawFlush(draw); 
  return 0;
}

static int MatView_SeqAIJ(PetscObject obj,Viewer viewer)
{
  Mat         A = (Mat) obj;
  Mat_SeqAIJ  *a = (Mat_SeqAIJ*) A->data;
  PetscObject vobj = (PetscObject) viewer;

  if (!a->assembled) SETERRQ(1,"MatView_SeqAIJ:Not for unassembled matrix");
  if (!viewer) { 
    viewer = STDOUT_VIEWER_SELF; vobj = (PetscObject) viewer;
  }
  if (vobj->cookie == VIEWER_COOKIE) {
    if (vobj->type == MATLAB_VIEWER) {
      return ViewerMatlabPutSparse_Private(viewer,a->m,a->n,a->nz,a->a,a->i,a->j); 
    }
    else if (vobj->type == ASCII_FILE_VIEWER || vobj->type == ASCII_FILES_VIEWER){
      return MatView_SeqAIJ_ASCII(A,viewer);
    }
    else if (vobj->type == BINARY_FILE_VIEWER) {
      return MatView_SeqAIJ_Binary(A,viewer);
    }
  }
  else if (vobj->cookie == DRAW_COOKIE) {
    if (vobj->type == NULLWINDOW) return 0;
    else return MatView_SeqAIJ_Draw(A,viewer);
  }
  return 0;
}

static int MatAssemblyEnd_SeqAIJ(Mat A,MatAssemblyType mode)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  int        fshift = 0,i,j,*ai = a->i, *aj = a->j, *imax = a->imax;
  int        m = a->m, *ip, N, *ailen = a->ilen,shift = a->indexshift;
  Scalar     *aa = a->a, *ap;

  if (mode == FLUSH_ASSEMBLY) return 0;

  for ( i=1; i<m; i++ ) {
    /* move each row back by the amount of empty slots (fshift) before it*/
    fshift += imax[i-1] - ailen[i-1];
    if (fshift) {
      ip = aj + ai[i] + shift; ap = aa + ai[i] + shift;
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
  a->nz = ai[m] + shift; 

  /* diagonals may have moved, so kill the diagonal pointers */
  if (fshift && a->diag) {
    PETSCFREE(a->diag);
    PLogObjectMemory(A,-(m+1)*sizeof(int));
    a->diag = 0;
  } 
  a->assembled = 1;
  return 0;
}

static int MatZeroEntries_SeqAIJ(Mat A)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data; 
  PetscZero(a->a,(a->i[a->m]+a->indexshift)*sizeof(Scalar));
  return 0;
}

int MatDestroy_SeqAIJ(PetscObject obj)
{
  Mat        A  = (Mat) obj;
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
#if defined(PETSC_LOG)
  PLogObjectState(obj,"Rows=%d, Cols=%d, NZ=%d",a->m,a->n,a->nz);
#endif
  PETSCFREE(a->a); 
  if (!a->singlemalloc) { PETSCFREE(a->i); PETSCFREE(a->j);}
  if (a->diag) PETSCFREE(a->diag);
  if (a->ilen) PETSCFREE(a->ilen);
  if (a->imax) PETSCFREE(a->imax);
  if (a->solve_work) PETSCFREE(a->solve_work);
  PETSCFREE(a); 
  PLogObjectDestroy(A);
  PETSCHEADERDESTROY(A);
  return 0;
}

static int MatCompress_SeqAIJ(Mat A)
{
  return 0;
}

static int MatSetOption_SeqAIJ(Mat A,MatOption op)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  if      (op == ROW_ORIENTED)              a->roworiented = 1;
  else if (op == COLUMN_ORIENTED)           a->roworiented = 0;
  else if (op == COLUMNS_SORTED)            a->sorted      = 1;
  /* doesn't care about sorted rows */
  else if (op == NO_NEW_NONZERO_LOCATIONS)  a->nonew       = 1;
  else if (op == YES_NEW_NONZERO_LOCATIONS) a->nonew       = 0;

  if (op == COLUMN_ORIENTED) SETERRQ(PETSC_ERR_SUP,"MatSetOption_SeqAIJ:Column input not supported");
  return 0;
}

static int MatGetDiagonal_SeqAIJ(Mat A,Vec v)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  int        i,j, n,shift = a->indexshift;
  Scalar     *x, zero = 0.0;

  if (!a->assembled) SETERRQ(1,"MatGetDiagonal_SeqAIJ:Not for unassembled matrix");
  VecSet(&zero,v);
  VecGetArray(v,&x); VecGetLocalSize(v,&n);
  if (n != a->m) SETERRQ(1,"MatGetDiagonal_SeqAIJ:Nonconforming matrix and vector");
  for ( i=0; i<a->m; i++ ) {
    for ( j=a->i[i]+shift; j<a->i[i+1]+shift; j++ ) {
      if (a->j[j]+shift == i) {
        x[i] = a->a[j];
        break;
      }
    }
  }
  return 0;
}

/* -------------------------------------------------------*/
/* Should check that shapes of vectors and matrices match */
/* -------------------------------------------------------*/
static int MatMultTrans_SeqAIJ(Mat A,Vec xx,Vec yy)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  Scalar     *x, *y, *v, alpha;
  int        m = a->m, n, i, *idx, shift = a->indexshift;

  if (!a->assembled) SETERRQ(1,"MatMultTrans_SeqAIJ:Not for unassembled matrix");
  VecGetArray(xx,&x); VecGetArray(yy,&y);
  PetscZero(y,a->n*sizeof(Scalar));
  y = y + shift; /* shift for Fortran start by 1 indexing */
  for ( i=0; i<m; i++ ) {
    idx   = a->j + a->i[i] + shift;
    v     = a->a + a->i[i] + shift;
    n     = a->i[i+1] - a->i[i];
    alpha = x[i];
    while (n-->0) {y[*idx++] += alpha * *v++;}
  }
  PLogFlops(2*a->nz - a->n);
  return 0;
}
static int MatMultTransAdd_SeqAIJ(Mat A,Vec xx,Vec zz,Vec yy)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  Scalar     *x, *y, *v, alpha;
  int        m = a->m, n, i, *idx,shift = a->indexshift;

  if (!a->assembled) SETERRQ(1,"MatMultTransAdd_SeqAIJ:Not for unassembled matrix");
  VecGetArray(xx,&x); VecGetArray(yy,&y);
  if (zz != yy) VecCopy(zz,yy);
  y = y + shift; /* shift for Fortran start by 1 indexing */
  for ( i=0; i<m; i++ ) {
    idx   = a->j + a->i[i] + shift;
    v     = a->a + a->i[i] + shift;
    n     = a->i[i+1] - a->i[i];
    alpha = x[i];
    while (n-->0) {y[*idx++] += alpha * *v++;}
  }
  return 0;
}

static int MatMult_SeqAIJ(Mat A,Vec xx,Vec yy)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  Scalar     *x, *y, *v, sum;
  int        m = a->m, n, i, *idx, shift = a->indexshift,*ii;

  if (!a->assembled) SETERRQ(1,"MatMult_SeqAIJ:Not for unassembled matrix");
  VecGetArray(xx,&x); VecGetArray(yy,&y);
  x = x + shift; /* shift for Fortran start by 1 indexing */
  idx  = a->j;
  v    = a->a;
  ii   = a->i;
#if defined(PARCH_rs6000) 
#pragma disjoint (*x,*v,*y)
#endif
  for ( i=0; i<m; i++ ) {
    n    = ii[1] - ii[0]; ii++;
    sum  = 0.0;
    /* SPARSEDENSEDOT(sum,x,v,idx,n);  */
    /* for ( j=n-1; j>-1; j--) sum += v[j]*x[idx[j]]; */
    while (n--) sum += *v++ * x[*idx++];
    y[i] = sum;
  }
  PLogFlops(2*a->nz - m);
  return 0;
}

static int MatMultAdd_SeqAIJ(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  Scalar     *x, *y, *z, *v, sum;
  int        m = a->m, n, i, *idx, shift = a->indexshift;

  if (!a->assembled) SETERRQ(1,"MatMultAdd_SeqAIJ:Cannot multiply unassembled matrix");
  VecGetArray(xx,&x); VecGetArray(yy,&y); VecGetArray(zz,&z); 
  x = x + shift; /* shift for Fortran start by 1 indexing */
  for ( i=0; i<m; i++ ) {
    idx  = a->j + a->i[i] + shift;
    v    = a->a + a->i[i] + shift;
    n    = a->i[i+1] - a->i[i];
    sum  = y[i];
    SPARSEDENSEDOT(sum,x,v,idx,n); 
    z[i] = sum;
  }
  PLogFlops(2*a->nz);
  return 0;
}

/*
     Adds diagonal pointers to sparse matrix structure.
*/

int MatMarkDiag_SeqAIJ(Mat A)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data; 
  int        i,j, *diag, m = a->m,shift = a->indexshift;

  if (!a->assembled) SETERRQ(1,"MatMarkDiag_SeqAIJ:unassembled matrix");
  diag = (int *) PETSCMALLOC( (m+1)*sizeof(int)); CHKPTRQ(diag);
  PLogObjectMemory(A,(m+1)*sizeof(int));
  for ( i=0; i<a->m; i++ ) {
    for ( j=a->i[i]+shift; j<a->i[i+1]+shift; j++ ) {
      if (a->j[j]+shift == i) {
        diag[i] = j - shift;
        break;
      }
    }
  }
  a->diag = diag;
  return 0;
}

static int MatRelax_SeqAIJ(Mat A,Vec bb,double omega,MatSORType flag,
                           double fshift,int its,Vec xx)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  Scalar     *x, *b, *bs,  d, *xs, sum, *v = a->a,*t,scale,*ts, *xb;
  int        ierr, *idx, *diag,n = a->n, m = a->m, i;
  int        shift = a->indexshift;

  VecGetArray(xx,&x); VecGetArray(bb,&b);
  if (!a->diag) {if ((ierr = MatMarkDiag_SeqAIJ(A))) return ierr;}
  diag = a->diag;
  xs   = x + shift; /* shifted by one for index start of a or a->j*/
  if (flag == SOR_APPLY_UPPER) {
   /* apply ( U + D/omega) to the vector */
    bs = b + shift;
    for ( i=0; i<m; i++ ) {
        d    = fshift + a->a[diag[i] + shift];
        n    = a->i[i+1] - diag[i] - 1;
        idx  = a->j + diag[i] + (!shift);
        v    = a->a + diag[i] + (!shift);
        sum  = b[i]*d/omega;
        SPARSEDENSEDOT(sum,bs,v,idx,n); 
        x[i] = sum;
    }
    return 0;
  }
  if (flag == SOR_APPLY_LOWER) {
    SETERRQ(1,"MatRelax_SeqAIJ:SOR_APPLY_LOWER is not done");
  }
  else if (flag & SOR_EISENSTAT) {
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
      d    = fshift + a->a[diag[i] + shift];
      n    = a->i[i+1] - diag[i] - 1;
      idx  = a->j + diag[i] + (!shift);
      v    = a->a + diag[i] + (!shift);
      sum  = b[i];
      SPARSEDENSEMDOT(sum,xs,v,idx,n); 
      x[i] = omega*(sum/d);
    }

    /*  t = b - (2*E - D)x */
    v = a->a;
    for ( i=0; i<m; i++ ) { t[i] = b[i] - scale*(v[*diag++ + shift])*x[i]; }

    /*  t = (E + L)^{-1}t */
    ts = t + shift; /* shifted by one for index start of a or a->j*/
    diag = a->diag;
    for ( i=0; i<m; i++ ) {
      d    = fshift + a->a[diag[i]+shift];
      n    = diag[i] - a->i[i];
      idx  = a->j + a->i[i] + shift;
      v    = a->a + a->i[i] + shift;
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
        d    = fshift + a->a[diag[i]+shift];
        n    = diag[i] - a->i[i];
        idx  = a->j + a->i[i] + shift;
        v    = a->a + a->i[i] + shift;
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
        x[i] *= a->a[diag[i]+shift];
      }
    }
    if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP){
      for ( i=m-1; i>=0; i-- ) {
        d    = fshift + a->a[diag[i] + shift];
        n    = a->i[i+1] - diag[i] - 1;
        idx  = a->j + diag[i] + (!shift);
        v    = a->a + diag[i] + (!shift);
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
        d    = fshift + a->a[diag[i]+shift];
        n    = a->i[i+1] - a->i[i]; 
        idx  = a->j + a->i[i] + shift;
        v    = a->a + a->i[i] + shift;
        sum  = b[i];
        SPARSEDENSEMDOT(sum,xs,v,idx,n); 
        x[i] = (1. - omega)*x[i] + omega*(sum/d + x[i]);
      }
    }
    if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP){
      for ( i=m-1; i>=0; i-- ) {
        d    = fshift + a->a[diag[i] + shift];
        n    = a->i[i+1] - a->i[i]; 
        idx  = a->j + a->i[i] + shift;
        v    = a->a + a->i[i] + shift;
        sum  = b[i];
        SPARSEDENSEMDOT(sum,xs,v,idx,n); 
        x[i] = (1. - omega)*x[i] + omega*(sum/d + x[i]);
      }
    }
  }
  return 0;
} 

static int MatGetInfo_SeqAIJ(Mat A,MatInfoType flag,int *nz,
                             int *nzalloc,int *mem)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  *nz      = a->nz;
  *nzalloc = a->maxnz;
  *mem     = (int)A->mem;
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
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  int         i,ierr,N, *rows,m = a->m - 1,shift = a->indexshift;

  ierr = ISGetLocalSize(is,&N); CHKERRQ(ierr);
  ierr = ISGetIndices(is,&rows); CHKERRQ(ierr);
  if (diag) {
    for ( i=0; i<N; i++ ) {
      if (rows[i] < 0 || rows[i] > m) SETERRQ(1,"MatZeroRows_SeqAIJ:row out of range");
      if (a->ilen[rows[i]] > 0) { /* in case row was completely empty */
        a->ilen[rows[i]] = 1; 
        a->a[a->i[rows[i]]+shift] = *diag;
        a->j[a->i[rows[i]]+shift] = rows[i]+shift;
      }
      else {
        ierr = MatSetValues_SeqAIJ(A,1,&rows[i],1,&rows[i],diag,INSERT_VALUES);
        CHKERRQ(ierr);
      }
    }
  }
  else {
    for ( i=0; i<N; i++ ) {
      if (rows[i] < 0 || rows[i] > m) SETERRQ(1,"MatZeroRows_SeqAIJ:row out of range");
      a->ilen[rows[i]] = 0; 
    }
  }
  ISRestoreIndices(is,&rows);
  ierr = MatAssemblyBegin(A,FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,FINAL_ASSEMBLY); CHKERRQ(ierr);
  return 0;
}

static int MatGetSize_SeqAIJ(Mat A,int *m,int *n)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  *m = a->m; *n = a->n;
  return 0;
}

static int MatGetOwnershipRange_SeqAIJ(Mat A,int *m,int *n)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  *m = 0; *n = a->m;
  return 0;
}
static int MatGetRow_SeqAIJ(Mat A,int row,int *nz,int **idx,Scalar **v)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  int        *itmp,i,ierr,shift = a->indexshift;

  if (row < 0 || row >= a->m) SETERRQ(1,"MatGetRow_SeqAIJ:Row out of range");

  if (!a->assembled) {
    ierr = MatAssemblyBegin(A,FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,FINAL_ASSEMBLY); CHKERRQ(ierr);
  }
  *nz = a->i[row+1] - a->i[row];
  if (v) *v = a->a + a->i[row] + shift;
  if (idx) {
    if (*nz) {
      itmp = a->j + a->i[row] + shift;
      *idx = (int *) PETSCMALLOC( (*nz)*sizeof(int) ); CHKPTRQ(*idx);
      for ( i=0; i<(*nz); i++ ) {(*idx)[i] = itmp[i] + shift;}
    }
    else *idx = 0;
  }
  return 0;
}

static int MatRestoreRow_SeqAIJ(Mat A,int row,int *nz,int **idx,Scalar **v)
{
  if (idx) {if (*idx) PETSCFREE(*idx);}
  return 0;
}

static int MatNorm_SeqAIJ(Mat A,MatNormType type,double *norm)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  Scalar     *v = a->a;
  double     sum = 0.0;
  int        i, j,shift = a->indexshift;

  if (!a->assembled) SETERRQ(1,"MatNorm_SeqAIJ:Not for unassembled matrix");
  if (type == NORM_FROBENIUS) {
    for (i=0; i<a->nz; i++ ) {
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
    int    *jj = a->j;
    tmp = (double *) PETSCMALLOC( a->n*sizeof(double) ); CHKPTRQ(tmp);
    PetscZero(tmp,a->n*sizeof(double));
    *norm = 0.0;
    for ( j=0; j<a->nz; j++ ) {
#if defined(PETSC_COMPLEX)
        tmp[*jj++ + shift] += abs(*v++); 
#else
        tmp[*jj++ + shift] += fabs(*v++); 
#endif
    }
    for ( j=0; j<a->n; j++ ) {
      if (tmp[j] > *norm) *norm = tmp[j];
    }
    PETSCFREE(tmp);
  }
  else if (type == NORM_INFINITY) {
    *norm = 0.0;
    for ( j=0; j<a->m; j++ ) {
      v = a->a + a->i[j] + shift;
      sum = 0.0;
      for ( i=0; i<a->i[j+1]-a->i[j]; i++ ) {
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

static int MatTranspose_SeqAIJ(Mat A,Mat *B)
{ 
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  Mat        C;
  int        i, ierr, *aj = a->j, *ai = a->i, m = a->m, len, *col;
  Scalar     *array = a->a;
  int        shift = a->indexshift;

  if (!B && m != a->n) SETERRQ(1,"MatTranspose_SeqAIJ:Not for rectangular mat in place");
  col = (int *) PETSCMALLOC((1+a->n)*sizeof(int)); CHKPTRQ(col);
  PetscZero(col,(1+a->n)*sizeof(int));
  if (shift) {
    for ( i=0; i<ai[m]-1; i++ ) aj[i] -= 1;
  }
  for ( i=0; i<ai[m]+shift; i++ ) col[aj[i]] += 1;
  ierr = MatCreateSeqAIJ(A->comm,a->n,m,0,col,&C); CHKERRQ(ierr);
  PETSCFREE(col);
  for ( i=0; i<m; i++ ) {
    len = ai[i+1]-ai[i];
    ierr = MatSetValues(C,len,aj,1,&i,array,INSERT_VALUES); CHKERRQ(ierr);
    array += len; aj += len;
  }
  if (shift) { 
    for ( i=0; i<ai[m]-1; i++ ) aj[i] += 1;
  }

  ierr = MatAssemblyBegin(C,FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,FINAL_ASSEMBLY); CHKERRQ(ierr);

  if (B) {
    *B = C;
  } else {
    /* This isn't really an in-place transpose */
    PETSCFREE(a->a); 
    if (!a->singlemalloc) {PETSCFREE(a->i); PETSCFREE(a->j);}
    if (a->diag) PETSCFREE(a->diag);
    if (a->ilen) PETSCFREE(a->ilen);
    if (a->imax) PETSCFREE(a->imax);
    if (a->solve_work) PETSCFREE(a->solve_work);
    PETSCFREE(a); 
    PetscMemcpy(A,C,sizeof(struct _Mat)); 
    PETSCHEADERDESTROY(C);
  }
  return 0;
}

static int MatScale_SeqAIJ(Mat A,Vec ll,Vec rr)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  Scalar     *l,*r,x,*v;
  int        i,j,m = a->m, n = a->n, M, nz = a->nz, *jj;
  int        shift = a->indexshift;

  if (!a->assembled) SETERRQ(1,"MatScale_SeqAIJ:Cannot scale unassembled matrix");
  if (ll) {
    VecGetArray(ll,&l); VecGetSize(ll,&m);
    if (m != a->m) SETERRQ(1,"MatScale_SeqAIJ:Left scaling vector wrong length");
    v = a->a;
    for ( i=0; i<m; i++ ) {
      x = l[i];
      M = a->i[i+1] - a->i[i];
      for ( j=0; j<M; j++ ) { (*v++) *= x;} 
    }
  }
  if (rr) {
    VecGetArray(rr,&r); VecGetSize(rr,&n);
    if (n != a->n) SETERRQ(1,"MatScale_SeqAIJ:Right scaling vector wrong length");
    v = a->a; jj = a->j;
    for ( i=0; i<nz; i++ ) {
      (*v++) *= r[*jj++ + shift]; 
    }
  }
  return 0;
}

static int MatGetSubMatrix_SeqAIJ(Mat A,IS isrow,IS iscol,Mat *B)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  int        nznew, *smap, i, k, kstart, kend, ierr, oldcols = a->n;
  int        *irow, *icol, nrows, ncols, *cwork, shift = a->indexshift;
  Scalar     *vwork;
  Mat        C;

  if (!a->assembled) SETERRQ(1,"MatGetSubMatrix_SeqAIJ:Not for unassembled matrix");  
  ierr = ISGetIndices(isrow,&irow); CHKERRQ(ierr);
  ierr = ISGetIndices(iscol,&icol); CHKERRQ(ierr);
  ierr = ISGetSize(isrow,&nrows); CHKERRQ(ierr);
  ierr = ISGetSize(iscol,&ncols); CHKERRQ(ierr);

  smap  = (int *) PETSCMALLOC((1+oldcols)*sizeof(int)); CHKPTRQ(smap);
  cwork = (int *) PETSCMALLOC((1+ncols)*sizeof(int)); CHKPTRQ(cwork);
  vwork = (Scalar *) PETSCMALLOC((1+ncols)*sizeof(Scalar)); CHKPTRQ(vwork);
  PetscZero(smap,oldcols*sizeof(int));
  for ( i=0; i<ncols; i++ ) smap[icol[i]] = i+1;

  /* Create and fill new matrix */
  ierr = MatCreateSeqAIJ(A->comm,nrows,ncols,0,0,&C);CHKERRQ(ierr);
  for (i=0; i<nrows; i++) {
    nznew  = 0;
    kstart = a->i[irow[i]]+shift; 
    kend   = kstart + a->ilen[irow[i]];
    for ( k=kstart; k<kend; k++ ) {
      if (smap[a->j[k]+shift]) {
        cwork[nznew]   = smap[a->j[k]+shift] - 1;
        vwork[nznew++] = a->a[k];
      }
    }
    ierr = MatSetValues(C,1,&i,nznew,cwork,vwork,INSERT_VALUES); CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(C,FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,FINAL_ASSEMBLY); CHKERRQ(ierr);

  /* Free work space */
  PETSCFREE(smap); PETSCFREE(cwork); PETSCFREE(vwork);
  ierr = ISRestoreIndices(isrow,&irow); CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol,&icol); CHKERRQ(ierr);
  *B = C;
  return 0;
}

/* -------------------------------------------------------------------*/
extern int MatILUFactorSymbolic_SeqAIJ(Mat,IS,IS,double,int,Mat *);
extern int MatConvert_SeqAIJ(Mat,MatType,Mat *);
static int MatCopyPrivate_SeqAIJ(Mat,Mat *);

static struct _MatOps MatOps = {MatSetValues_SeqAIJ,
       MatGetRow_SeqAIJ,MatRestoreRow_SeqAIJ,
       MatMult_SeqAIJ,MatMultAdd_SeqAIJ,
       MatMultTrans_SeqAIJ,MatMultTransAdd_SeqAIJ,
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
       MatGetSubMatrix_SeqAIJ,0,
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
.  A - the matrix 

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
int MatCreateSeqAIJ(MPI_Comm comm,int m,int n,int nz,int *nnz, Mat *A)
{
  Mat        B;
  Mat_SeqAIJ *b;
  int        i,len,ierr;
  *A      = 0;
  PETSCHEADERCREATE(B,_Mat,MAT_COOKIE,MATSEQAIJ,comm);
  PLogObjectCreate(B);
  B->data               = (void *) (b = PETSCNEW(Mat_SeqAIJ)); CHKPTRQ(b);
  PetscMemcpy(&B->ops,&MatOps,sizeof(struct _MatOps));
  B->destroy          = MatDestroy_SeqAIJ;
  B->view             = MatView_SeqAIJ;
  B->factor           = 0;
  B->lupivotthreshold = 1.0;
  OptionsGetDouble(0,"-mat_lu_pivotthreshold",&B->lupivotthreshold);
  b->row              = 0;
  b->col              = 0;
  b->indexshift       = 0;
  if (OptionsHasName(0,"-mat_aij_oneindex")) b->indexshift = -1;

  b->m       = m;
  b->n       = n;
  b->imax    = (int *) PETSCMALLOC( (m+1)*sizeof(int) ); CHKPTRQ(b->imax);
  if (!nnz) {
    if (nz <= 0) nz = 1;
    for ( i=0; i<m; i++ ) b->imax[i] = nz;
    nz = nz*m;
  }
  else {
    nz = 0;
    for ( i=0; i<m; i++ ) {b->imax[i] = nnz[i]; nz += nnz[i];}
  }

  /* allocate the matrix space */
  len     = nz*(sizeof(int) + sizeof(Scalar)) + (b->m+1)*sizeof(int);
  b->a  = (Scalar *) PETSCMALLOC( len ); CHKPTRQ(b->a);
  b->j  = (int *) (b->a + nz);
  PetscZero(b->j,nz*sizeof(int));
  b->i  = b->j + nz;
  b->singlemalloc = 1;

  b->i[0] = -b->indexshift;
  for (i=1; i<m+1; i++) {
    b->i[i] = b->i[i-1] + b->imax[i-1];
  }

  /* b->ilen will count nonzeros in each row so far. */
  b->ilen = (int *) PETSCMALLOC((m+1)*sizeof(int)); 
  PLogObjectMemory(B,len+2*(m+1)*sizeof(int)+sizeof(struct _Mat)+sizeof(Mat_SeqAIJ));
  for ( i=0; i<b->m; i++ ) { b->ilen[i] = 0;}

  b->nz          = 0;
  b->maxnz       = nz;
  b->sorted      = 0;
  b->roworiented = 1;
  b->nonew       = 0;
  b->diag        = 0;
  b->assembled   = 0;
  b->solve_work  = 0;

  *A = B;
  if (OptionsHasName(0,"-mat_aij_superlu")) {
    ierr = MatUseSuperLU_SeqAIJ(B); CHKERRQ(ierr);
  }
  if (OptionsHasName(0,"-mat_aij_essl")) {
    ierr = MatUseEssl_SeqAIJ(B); CHKERRQ(ierr);
  }
  if (OptionsHasName(0,"-mat_aij_dxml")) {
    if (!b->indexshift) SETERRQ(1,"MatCreateSeqAIJ:need -mat_aij_oneindex with -mat_aij_dxml");
    ierr = MatUseDXML_SeqAIJ(B); CHKERRQ(ierr);
  }

  return 0;
}

static int MatCopyPrivate_SeqAIJ(Mat A,Mat *B)
{
  Mat        C;
  Mat_SeqAIJ *c,*a = (Mat_SeqAIJ *) A->data;
  int        i,len, m = a->m;
  int        shift = a->indexshift;
  *B      = 0;

  if (!a->assembled) SETERRQ(1,"MatCopyPrivate_SeqAIJ:Cannot copy unassembled matrix");
  PETSCHEADERCREATE(C,_Mat,MAT_COOKIE,MATSEQAIJ,A->comm);
  PLogObjectCreate(C);
  C->data       = (void *) (c = PETSCNEW(Mat_SeqAIJ)); CHKPTRQ(c);
  PetscMemcpy(&C->ops,&MatOps,sizeof(struct _MatOps));
  C->destroy    = MatDestroy_SeqAIJ;
  C->view       = MatView_SeqAIJ;
  C->factor     = A->factor;
  c->row        = 0;
  c->col        = 0;
  c->indexshift = shift;

  c->m          = a->m;
  c->n          = a->n;

  c->imax       = (int *) PETSCMALLOC((m+1)*sizeof(int)); CHKPTRQ(c->imax);
  c->ilen       = (int *) PETSCMALLOC((m+1)*sizeof(int)); CHKPTRQ(c->ilen);
  for ( i=0; i<m; i++ ) {
    c->imax[i] = a->imax[i];
    c->ilen[i] = a->ilen[i]; 
  }

  /* allocate the matrix space */
  c->singlemalloc = 1;
  len     = (m+1)*sizeof(int)+(a->i[m])*(sizeof(Scalar)+sizeof(int));
  c->a  = (Scalar *) PETSCMALLOC( len ); CHKPTRQ(c->a);
  c->j  = (int *) (c->a + a->i[m] + shift);
  c->i  = c->j + a->i[m] + shift;
  PetscMemcpy(c->i,a->i,(m+1)*sizeof(int));
  if (m > 0) {
    PetscMemcpy(c->j,a->j,(a->i[m]+shift)*sizeof(int));
    PetscMemcpy(c->a,a->a,(a->i[m]+shift)*sizeof(Scalar));
  }

  PLogObjectMemory(C,len+2*(m+1)*sizeof(int)+sizeof(struct _Mat)+sizeof(Mat_SeqAIJ));  
  c->sorted      = a->sorted;
  c->roworiented = a->roworiented;
  c->nonew       = a->nonew;

  if (a->diag) {
    c->diag = (int *) PETSCMALLOC( (m+1)*sizeof(int) ); CHKPTRQ(c->diag);
    PLogObjectMemory(C,(m+1)*sizeof(int));
    for ( i=0; i<m; i++ ) {
      c->diag[i] = a->diag[i];
    }
  }
  else c->diag        = 0;
  c->assembled        = 1;
  c->nz               = a->nz;
  c->maxnz            = a->maxnz;
  c->solve_work       = 0;
  *B = C;
  return 0;
}

int MatLoad_SeqAIJ(Viewer bview,MatType type,Mat *A)
{
  Mat_SeqAIJ   *a;
  Mat          B;
  int          i, nz, ierr, fd, header[4],numtid,*rowlengths = 0,M,N,shift;
  PetscObject  vobj = (PetscObject) bview;
  MPI_Comm     comm = vobj->comm;

  MPI_Comm_size(comm,&numtid);
  if (numtid > 1) SETERRQ(1,"MatLoad_SeqAIJ: view must have one processor");
  ierr = ViewerFileGetDescriptor_Private(bview,&fd); CHKERRQ(ierr);
  ierr = SYRead(fd,header,4,SYINT); CHKERRQ(ierr);
  if (header[0] != MAT_COOKIE) SETERRQ(1,"MatLoad_SeqAIJ: not matrix object");
  M = header[1]; N = header[2]; nz = header[3];

  /* read in row lengths */
  rowlengths = (int*) PETSCMALLOC( M*sizeof(int) ); CHKPTRQ(rowlengths);
  ierr = SYRead(fd,rowlengths,M,SYINT); CHKERRQ(ierr);

  /* create our matrix */
  ierr = MatCreateSeqAIJ(comm,M,N,0,rowlengths,A); CHKERRQ(ierr);
  B = *A;
  a = (Mat_SeqAIJ *) B->data;
  shift = a->indexshift;

  /* read in column indices and adjust for Fortran indexing*/
  ierr = SYRead(fd,a->j,nz,SYINT); CHKERRQ(ierr);
  if (shift) {
    for ( i=0; i<nz; i++ ) {
      a->j[i] += 1;
    }
  }

  /* read in nonzero values */
  ierr = SYRead(fd,a->a,nz,SYSCALAR); CHKERRQ(ierr);

  /* set matrix "i" values */
  a->i[0] = -shift;
  for ( i=1; i<= M; i++ ) {
    a->i[i]      = a->i[i-1] + rowlengths[i-1];
    a->ilen[i-1] = rowlengths[i-1];
  }
  PETSCFREE(rowlengths);   

  ierr = MatAssemblyBegin(B,FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,FINAL_ASSEMBLY); CHKERRQ(ierr);
  return 0;
}



