
#ifndef lint
static char vcid[] = "$Id: aij.c,v 1.169 1996/04/07 22:45:46 curfman Exp balay $";
#endif

/*
    Defines the basic matrix operations for the AIJ (compressed row)
  matrix storage format.
*/
#include "aij.h"
#include "src/vec/vecimpl.h"
#include "src/inline/spops.h"
#include "petsc.h"
#include "src/inline/bitarray.h"

extern int MatToSymmetricIJ_SeqAIJ(int,int*,int*,int,int,int**,int**);

static int MatGetReordering_SeqAIJ(Mat A,MatOrdering type,IS *rperm, IS *cperm)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  int        ierr, *ia, *ja,n,*idx,i,oshift,ishift;

  /* 
     this is tacky: In the future when we have written special factorization
     and solve routines for the identity permutation we should use a 
     stride index set instead of the general one.
  */
  if (type  == ORDER_NATURAL) {
    n = a->n;
    idx = (int *) PetscMalloc( n*sizeof(int) ); CHKPTRQ(idx);
    for ( i=0; i<n; i++ ) idx[i] = i;
    ierr = ISCreateSeq(MPI_COMM_SELF,n,idx,rperm); CHKERRQ(ierr);
    ierr = ISCreateSeq(MPI_COMM_SELF,n,idx,cperm); CHKERRQ(ierr);
    PetscFree(idx);
    ISSetPermutation(*rperm);
    ISSetPermutation(*cperm);
    ISSetIdentity(*rperm);
    ISSetIdentity(*cperm);
    return 0;
  }

  MatReorderingRegisterAll();
  ishift = a->indexshift;
  oshift = -MatReorderingIndexShift[(int)type];
  if (MatReorderingRequiresSymmetric[(int)type]) {
    ierr = MatToSymmetricIJ_SeqAIJ(a->n,a->i,a->j,ishift,oshift,&ia,&ja);
    CHKERRQ(ierr);
    ierr = MatGetReordering_IJ(a->n,ia,ja,type,rperm,cperm); CHKERRQ(ierr);
    PetscFree(ia); PetscFree(ja);
  } else {
    if (ishift == oshift) {
      ierr = MatGetReordering_IJ(a->n,a->i,a->j,type,rperm,cperm); CHKERRQ(ierr);
    }
    else if (ishift == -1) {
      /* temporarily subtract 1 from i and j indices */
      int nz = a->i[a->n] - 1; 
      for ( i=0; i<nz; i++ ) a->j[i]--;
      for ( i=0; i<a->n+1; i++ ) a->i[i]--;
      ierr = MatGetReordering_IJ(a->n,a->i,a->j,type,rperm,cperm); CHKERRQ(ierr);
      for ( i=0; i<nz; i++ ) a->j[i]++;
      for ( i=0; i<a->n+1; i++ ) a->i[i]++;
    } else {
      /* temporarily add 1 to i and j indices */
      int nz = a->i[a->n] - 1; 
      for ( i=0; i<nz; i++ ) a->j[i]++;
      for ( i=0; i<a->n+1; i++ ) a->i[i]++;
      ierr = MatGetReordering_IJ(a->n,a->i,a->j,type,rperm,cperm); CHKERRQ(ierr);
      for ( i=0; i<nz; i++ ) a->j[i]--;
      for ( i=0; i<a->n+1; i++ ) a->i[i]--;
    }
  }
  return 0; 
}

#define CHUNKSIZE   15

/* This version has row oriented v  */
static int MatSetValues_SeqAIJ(Mat A,int m,int *im,int n,int *in,Scalar *v,InsertMode is)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  int        *rp,k,low,high,t,ii,row,nrow,i,col,l,rmax, N, sorted = a->sorted;
  int        *imax = a->imax, *ai = a->i, *ailen = a->ilen,roworiented = a->roworiented;
  int        *aj = a->j, nonew = a->nonew,shift = a->indexshift;
  Scalar     *ap,value, *aa = a->a;

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
      col = in[l] - shift;
      if (roworiented) {
        value = *v++; 
      }
      else {
        value = v[k + l*m];
      }
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
        new_a   = (Scalar *) PetscMalloc( len ); CHKPTRQ(new_a);
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
        PetscFree(a->a); 
        if (!a->singlemalloc) {PetscFree(a->i);PetscFree(a->j);}
        aa = a->a = new_a; ai = a->i = new_i; aj = a->j = new_j; 
        a->singlemalloc = 1;

        rp   = aj + ai[row] + shift; ap = aa + ai[row] + shift;
        rmax = imax[row] = imax[row] + CHUNKSIZE;
        PLogObjectMemory(A,CHUNKSIZE*(sizeof(int) + sizeof(Scalar)));
        a->maxnz += CHUNKSIZE;
        a->reallocs++;
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

static int MatGetValues_SeqAIJ(Mat A,int m,int *im,int n,int *in,Scalar *v)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  int        *rp, k, low, high, t, row, nrow, i, col, l, *aj = a->j;
  int        *ai = a->i, *ailen = a->ilen, shift = a->indexshift;
  Scalar     *ap, *aa = a->a, zero = 0.0;

  for ( k=0; k<m; k++ ) { /* loop over rows */
    row  = im[k];   
    if (row < 0) SETERRQ(1,"MatGetValues_SeqAIJ:Negative row");
    if (row >= a->m) SETERRQ(1,"MatGetValues_SeqAIJ:Row too large");
    rp   = aj + ai[row] + shift; ap = aa + ai[row] + shift;
    nrow = ailen[row]; 
    for ( l=0; l<n; l++ ) { /* loop over columns */
      if (in[l] < 0) SETERRQ(1,"MatGetValues_SeqAIJ:Negative column");
      if (in[l] >= a->n) SETERRQ(1,"MatGetValues_SeqAIJ:Column too large");
      col = in[l] - shift;
      high = nrow; low = 0; /* assume unsorted */
      while (high-low > 5) {
        t = (low+high)/2;
        if (rp[t] > col) high = t;
        else             low  = t;
      }
      for ( i=low; i<high; i++ ) {
        if (rp[i] > col) break;
        if (rp[i] == col) {
          *v++ = ap[i];
          goto finished;
        }
      } 
      *v++ = zero;
      finished:;
    }
  }
  return 0;
} 

#include "draw.h"
#include "pinclude/pviewer.h"
#include "sys.h"

static int MatView_SeqAIJ_Binary(Mat A,Viewer viewer)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  int        i, fd, *col_lens, ierr;

  ierr = ViewerBinaryGetDescriptor(viewer,&fd); CHKERRQ(ierr);
  col_lens = (int *) PetscMalloc( (4+a->m)*sizeof(int) ); CHKPTRQ(col_lens);
  col_lens[0] = MAT_COOKIE;
  col_lens[1] = a->m;
  col_lens[2] = a->n;
  col_lens[3] = a->nz;

  /* store lengths of each row and write (including header) to file */
  for ( i=0; i<a->m; i++ ) {
    col_lens[4+i] = a->i[i+1] - a->i[i];
  }
  ierr = PetscBinaryWrite(fd,col_lens,4+a->m,BINARY_INT,1); CHKERRQ(ierr);
  PetscFree(col_lens);

  /* store column indices (zero start index) */
  if (a->indexshift) {
    for ( i=0; i<a->nz; i++ ) a->j[i]--;
  }
  ierr = PetscBinaryWrite(fd,a->j,a->nz,BINARY_INT,0); CHKERRQ(ierr);
  if (a->indexshift) {
    for ( i=0; i<a->nz; i++ ) a->j[i]++;
  }

  /* store nonzero values */
  ierr = PetscBinaryWrite(fd,a->a,a->nz,BINARY_SCALAR,0); CHKERRQ(ierr);
  return 0;
}

static int MatView_SeqAIJ_ASCII(Mat A,Viewer viewer)
{
  Mat_SeqAIJ  *a = (Mat_SeqAIJ *) A->data;
  int         ierr, i,j, m = a->m, shift = a->indexshift, format, flg;
  FILE        *fd;
  char        *outputname;

  ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
  ierr = ViewerFileGetOutputname_Private(viewer,&outputname); CHKERRQ(ierr);
  ierr = ViewerGetFormat(viewer,&format);
  if (format == ASCII_FORMAT_INFO) {
    return 0;
  } 
  else if (format == ASCII_FORMAT_INFO_DETAILED) {
    ierr = OptionsHasName(PETSC_NULL,"-mat_aij_no_inode",&flg); CHKERRQ(ierr);
    if (flg) fprintf(fd,"  not using I-node routines\n");
    else     fprintf(fd,"  using I-node routines: found %d nodes, limit used is %d\n",
        a->inode.node_count,a->inode.limit);
  }
  else if (format == ASCII_FORMAT_MATLAB) {
    int nz, nzalloc, mem;
    MatGetInfo(A,MAT_LOCAL,&nz,&nzalloc,&mem);
    fprintf(fd,"%% Size = %d %d \n",m,a->n);
    fprintf(fd,"%% Nonzeros = %d \n",nz);
    fprintf(fd,"zzz = zeros(%d,3);\n",nz);
    fprintf(fd,"zzz = [\n");

    for (i=0; i<m; i++) {
      for ( j=a->i[i]+shift; j<a->i[i+1]+shift; j++ ) {
#if defined(PETSC_COMPLEX)
        fprintf(fd,"%d %d  %18.16e  %18.16e \n",i+1,a->j[j]+!shift,real(a->a[j]),
                   imag(a->a[j]));
#else
        fprintf(fd,"%d %d  %18.16e\n", i+1, a->j[j]+!shift, a->a[j]);
#endif
      }
    }
    fprintf(fd,"];\n %s = spconvert(zzz);\n",outputname);
  } 
  else if (format == ASCII_FORMAT_COMMON) {
    for ( i=0; i<m; i++ ) {
      fprintf(fd,"row %d:",i);
      for ( j=a->i[i]+shift; j<a->i[i+1]+shift; j++ ) {
#if defined(PETSC_COMPLEX)
        if (imag(a->a[j]) != 0.0 && real(a->a[j]) != 0.0)
          fprintf(fd," %d %g + %g i",a->j[j]+shift,real(a->a[j]),imag(a->a[j]));
        else if (real(a->a[j]) != 0.0)
          fprintf(fd," %d %g ",a->j[j]+shift,real(a->a[j]));
#else
        if (a->a[j] != 0.0) fprintf(fd," %d %g ",a->j[j]+shift,a->a[j]);
#endif
      }
      fprintf(fd,"\n");
    }
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
  int         ierr, i,j, m = a->m, shift = a->indexshift,pause,color;
  double      xl,yl,xr,yr,w,h,xc,yc,scale = 1.0,x_l,x_r,y_l,y_r;
  Draw        draw;
  DrawButton  button;
  PetscTruth  isnull;

  ViewerDrawGetDraw(viewer,&draw);
  ierr = DrawIsNull(draw,&isnull); CHKERRQ(ierr); if (isnull) return 0;

  xr  = a->n; yr = a->m; h = yr/10.0; w = xr/10.0; 
  xr += w;    yr += h;  xl = -w;     yl = -h;
  ierr = DrawSetCoordinates(draw,xl,yl,xr,yr); CHKERRQ(ierr);
  /* loop over matrix elements drawing boxes */
  color = DRAW_BLUE;
  for ( i=0; i<m; i++ ) {
    y_l = m - i - 1.0; y_r = y_l + 1.0;
    for ( j=a->i[i]+shift; j<a->i[i+1]+shift; j++ ) {
      x_l = a->j[j] + shift; x_r = x_l + 1.0;
#if defined(PETSC_COMPLEX)
      if (real(a->a[j]) >=  0.) continue;
#else
      if (a->a[j] >=  0.) continue;
#endif
      DrawRectangle(draw,x_l,y_l,x_r,y_r,color,color,color,color);
    } 
  }
  color = DRAW_CYAN;
  for ( i=0; i<m; i++ ) {
    y_l = m - i - 1.0; y_r = y_l + 1.0;
    for ( j=a->i[i]+shift; j<a->i[i+1]+shift; j++ ) {
      x_l = a->j[j] + shift; x_r = x_l + 1.0;
      if (a->a[j] !=  0.) continue;
      DrawRectangle(draw,x_l,y_l,x_r,y_r,color,color,color,color);
    } 
  }
  color = DRAW_RED;
  for ( i=0; i<m; i++ ) {
    y_l = m - i - 1.0; y_r = y_l + 1.0;
    for ( j=a->i[i]+shift; j<a->i[i+1]+shift; j++ ) {
      x_l = a->j[j] + shift; x_r = x_l + 1.0;
#if defined(PETSC_COMPLEX)
      if (real(a->a[j]) <=  0.) continue;
#else
      if (a->a[j] <=  0.) continue;
#endif
      DrawRectangle(draw,x_l,y_l,x_r,y_r,color,color,color,color);
    } 
  }
  DrawFlush(draw); 
  DrawGetPause(draw,&pause);
  if (pause >= 0) { PetscSleep(pause); return 0;}

  /* allow the matrix to zoom or shrink */
  ierr = DrawGetMouseButton(draw,&button,&xc,&yc,0,0); 
  while (button != BUTTON_RIGHT) {
    DrawClear(draw);
    if (button == BUTTON_LEFT) scale = .5;
    else if (button == BUTTON_CENTER) scale = 2.;
    xl = scale*(xl + w - xc) + xc - w*scale;
    xr = scale*(xr - w - xc) + xc + w*scale;
    yl = scale*(yl + h - yc) + yc - h*scale;
    yr = scale*(yr - h - yc) + yc + h*scale;
    w *= scale; h *= scale;
    ierr = DrawSetCoordinates(draw,xl,yl,xr,yr); CHKERRQ(ierr);
    color = DRAW_BLUE;
    for ( i=0; i<m; i++ ) {
      y_l = m - i - 1.0; y_r = y_l + 1.0;
      for ( j=a->i[i]+shift; j<a->i[i+1]+shift; j++ ) {
        x_l = a->j[j] + shift; x_r = x_l + 1.0;
#if defined(PETSC_COMPLEX)
        if (real(a->a[j]) >=  0.) continue;
#else
        if (a->a[j] >=  0.) continue;
#endif
        DrawRectangle(draw,x_l,y_l,x_r,y_r,color,color,color,color);
      } 
    }
    color = DRAW_CYAN;
    for ( i=0; i<m; i++ ) {
      y_l = m - i - 1.0; y_r = y_l + 1.0;
      for ( j=a->i[i]+shift; j<a->i[i+1]+shift; j++ ) {
        x_l = a->j[j] + shift; x_r = x_l + 1.0;
        if (a->a[j] !=  0.) continue;
        DrawRectangle(draw,x_l,y_l,x_r,y_r,color,color,color,color);
      } 
    }
    color = DRAW_RED;
    for ( i=0; i<m; i++ ) {
      y_l = m - i - 1.0; y_r = y_l + 1.0;
      for ( j=a->i[i]+shift; j<a->i[i+1]+shift; j++ ) {
        x_l = a->j[j] + shift; x_r = x_l + 1.0;
#if defined(PETSC_COMPLEX)
        if (real(a->a[j]) <=  0.) continue;
#else
        if (a->a[j] <=  0.) continue;
#endif
        DrawRectangle(draw,x_l,y_l,x_r,y_r,color,color,color,color);
      } 
    }
    ierr = DrawGetMouseButton(draw,&button,&xc,&yc,0,0); 
  }
  return 0;
}

static int MatView_SeqAIJ(PetscObject obj,Viewer viewer)
{
  Mat         A = (Mat) obj;
  Mat_SeqAIJ  *a = (Mat_SeqAIJ*) A->data;
  ViewerType  vtype;
  int         ierr;

  if (!viewer) { 
    viewer = STDOUT_VIEWER_SELF; 
  }
  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (vtype == MATLAB_VIEWER) {
    return ViewerMatlabPutSparse_Private(viewer,a->m,a->n,a->nz,a->a,a->i,a->j); 
  }
  else if (vtype == ASCII_FILE_VIEWER || vtype == ASCII_FILES_VIEWER){
    return MatView_SeqAIJ_ASCII(A,viewer);
  }
  else if (vtype == BINARY_FILE_VIEWER) {
    return MatView_SeqAIJ_Binary(A,viewer);
  }
  else if (vtype == DRAW_VIEWER) {
    return MatView_SeqAIJ_Draw(A,viewer);
  }
  return 0;
}

extern int Mat_AIJ_CheckInode(Mat);
static int MatAssemblyEnd_SeqAIJ(Mat A,MatAssemblyType mode)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  int        fshift = 0,i,j,*ai = a->i, *aj = a->j, *imax = a->imax,ierr;
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
    PetscFree(a->diag);
    PLogObjectMemory(A,-(m+1)*sizeof(int));
    a->diag = 0;
  } 
  PLogInfo(A,"MatAssemblyEnd_SeqAIJ:Unneeded storage space %d used %d rows %d\n",
           fshift,a->nz,m);
  PLogInfo(A,"MatAssemblyEnd_SeqAIJ:Number of mallocs during MatSetValues %d\n",
           a->reallocs);
  /* check out for identical nodes. If found, use inode functions */
  ierr = Mat_AIJ_CheckInode(A); CHKERRQ(ierr);
  return 0;
}

static int MatZeroEntries_SeqAIJ(Mat A)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data; 
  PetscMemzero(a->a,(a->i[a->m]+a->indexshift)*sizeof(Scalar));
  return 0;
}

int MatDestroy_SeqAIJ(PetscObject obj)
{
  Mat        A  = (Mat) obj;
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;

#if defined(PETSC_LOG)
  PLogObjectState(obj,"Rows=%d, Cols=%d, NZ=%d",a->m,a->n,a->nz);
#endif
  PetscFree(a->a); 
  if (!a->singlemalloc) { PetscFree(a->i); PetscFree(a->j);}
  if (a->diag) PetscFree(a->diag);
  if (a->ilen) PetscFree(a->ilen);
  if (a->imax) PetscFree(a->imax);
  if (a->solve_work) PetscFree(a->solve_work);
  if (a->inode.size) PetscFree(a->inode.size);
  PetscFree(a); 
  PLogObjectDestroy(A);
  PetscHeaderDestroy(A);
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
  else if (op == NO_NEW_NONZERO_LOCATIONS)  a->nonew       = 1;
  else if (op == YES_NEW_NONZERO_LOCATIONS) a->nonew       = 0;
  else if (op == ROWS_SORTED || 
           op == SYMMETRIC_MATRIX ||
           op == STRUCTURALLY_SYMMETRIC_MATRIX ||
           op == YES_NEW_DIAGONALS)
    PLogInfo(A,"Info:MatSetOption_SeqAIJ:Option ignored\n");
  else if (op == NO_NEW_DIAGONALS)
    {SETERRQ(PETSC_ERR_SUP,"MatSetOption_SeqAIJ:NO_NEW_DIAGONALS");}
  else if (op == INODE_LIMIT_1)            a->inode.limit  = 1;
  else if (op == INODE_LIMIT_2)            a->inode.limit  = 2;
  else if (op == INODE_LIMIT_3)            a->inode.limit  = 3;
  else if (op == INODE_LIMIT_4)            a->inode.limit  = 4;
  else if (op == INODE_LIMIT_5)            a->inode.limit  = 5;
  else 
    {SETERRQ(PETSC_ERR_SUP,"MatSetOption_SeqAIJ:unknown option");}
  return 0;
}

static int MatGetDiagonal_SeqAIJ(Mat A,Vec v)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  int        i,j, n,shift = a->indexshift;
  Scalar     *x, zero = 0.0;

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
int MatMultTrans_SeqAIJ(Mat A,Vec xx,Vec yy)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  Scalar     *x, *y, *v, alpha;
  int        m = a->m, n, i, *idx, shift = a->indexshift;

  VecGetArray(xx,&x); VecGetArray(yy,&y);
  PetscMemzero(y,a->n*sizeof(Scalar));
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

int MatMultTransAdd_SeqAIJ(Mat A,Vec xx,Vec zz,Vec yy)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  Scalar     *x, *y, *v, alpha;
  int        m = a->m, n, i, *idx,shift = a->indexshift;

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

int MatMult_SeqAIJ(Mat A,Vec xx,Vec yy)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  Scalar     *x, *y, *v, sum;
  int        m = a->m, n, i, *idx, shift = a->indexshift,*ii;

  VecGetArray(xx,&x); VecGetArray(yy,&y);
  x    = x + shift; /* shift for Fortran start by 1 indexing */
  idx  = a->j;
  v    = a->a;
  ii   = a->i;
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

int MatMultAdd_SeqAIJ(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  Scalar     *x, *y, *z, *v, sum;
  int        m = a->m, n, i, *idx, shift = a->indexshift,*ii;

  VecGetArray(xx,&x); VecGetArray(yy,&y); VecGetArray(zz,&z); 
  x    = x + shift; /* shift for Fortran start by 1 indexing */
  idx  = a->j;
  v    = a->a;
  ii   = a->i;
  for ( i=0; i<m; i++ ) {
    n    = ii[1] - ii[0]; ii++;
    sum  = y[i];
    /* SPARSEDENSEDOT(sum,x,v,idx,n);  */
    while (n--) sum += *v++ * x[*idx++];
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

  diag = (int *) PetscMalloc( (m+1)*sizeof(int)); CHKPTRQ(diag);
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

int MatRelax_SeqAIJ(Mat A,Vec bb,double omega,MatSORType flag,
                           double fshift,int its,Vec xx)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  Scalar     *x, *b, *bs,  d, *xs, sum, *v = a->a,*t,scale,*ts, *xb;
  int        ierr, *idx, *diag,n = a->n, m = a->m, i, shift = a->indexshift;

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
    t = (Scalar *) PetscMalloc( m*sizeof(Scalar) ); CHKPTRQ(t);
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
    PetscFree(t);
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

static int MatGetInfo_SeqAIJ(Mat A,MatInfoType flag,int *nz,int *nzalloc,int *mem)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  if (nz)      *nz      = a->nz;
  if (nzalloc) *nzalloc = a->maxnz;
  if (mem)     *mem     = (int)A->mem;
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

  ierr = ISGetSize(is,&N); CHKERRQ(ierr);
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
int MatGetRow_SeqAIJ(Mat A,int row,int *nz,int **idx,Scalar **v)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  int        *itmp,i,shift = a->indexshift;

  if (row < 0 || row >= a->m) SETERRQ(1,"MatGetRow_SeqAIJ:Row out of range");

  *nz = a->i[row+1] - a->i[row];
  if (v) *v = a->a + a->i[row] + shift;
  if (idx) {
    itmp = a->j + a->i[row] + shift;
    if (*nz && shift) {
      *idx = (int *) PetscMalloc( (*nz)*sizeof(int) ); CHKPTRQ(*idx);
      for ( i=0; i<(*nz); i++ ) {(*idx)[i] = itmp[i] + shift;}
    } else if (*nz) {
      *idx = itmp;
    }
    else *idx = 0;
  }
  return 0;
}

int MatRestoreRow_SeqAIJ(Mat A,int row,int *nz,int **idx,Scalar **v)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  if (idx) {if (*idx && a->indexshift) PetscFree(*idx);}
  return 0;
}

static int MatNorm_SeqAIJ(Mat A,NormType type,double *norm)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  Scalar     *v = a->a;
  double     sum = 0.0;
  int        i, j,shift = a->indexshift;

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
    tmp = (double *) PetscMalloc( a->n*sizeof(double) ); CHKPTRQ(tmp);
    PetscMemzero(tmp,a->n*sizeof(double));
    *norm = 0.0;
    for ( j=0; j<a->nz; j++ ) {
        tmp[*jj++ + shift] += PetscAbsScalar(*v);  v++;
    }
    for ( j=0; j<a->n; j++ ) {
      if (tmp[j] > *norm) *norm = tmp[j];
    }
    PetscFree(tmp);
  }
  else if (type == NORM_INFINITY) {
    *norm = 0.0;
    for ( j=0; j<a->m; j++ ) {
      v = a->a + a->i[j] + shift;
      sum = 0.0;
      for ( i=0; i<a->i[j+1]-a->i[j]; i++ ) {
        sum += PetscAbsScalar(*v); v++;
      }
      if (sum > *norm) *norm = sum;
    }
  }
  else {
    SETERRQ(1,"MatNorm_SeqAIJ:No support for two norm yet");
  }
  return 0;
}

static int MatTranspose_SeqAIJ(Mat A,Mat *B)
{ 
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  Mat        C;
  int        i, ierr, *aj = a->j, *ai = a->i, m = a->m, len, *col;
  int        shift = a->indexshift;
  Scalar     *array = a->a;

  if (B == PETSC_NULL && m != a->n)
    SETERRQ(1,"MatTranspose_SeqAIJ:Square matrix only for in-place");
  col = (int *) PetscMalloc((1+a->n)*sizeof(int)); CHKPTRQ(col);
  PetscMemzero(col,(1+a->n)*sizeof(int));
  if (shift) {
    for ( i=0; i<ai[m]-1; i++ ) aj[i] -= 1;
  }
  for ( i=0; i<ai[m]+shift; i++ ) col[aj[i]] += 1;
  ierr = MatCreateSeqAIJ(A->comm,a->n,m,0,col,&C); CHKERRQ(ierr);
  PetscFree(col);
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

  if (B != PETSC_NULL) {
    *B = C;
  } else {
    /* This isn't really an in-place transpose */
    PetscFree(a->a); 
    if (!a->singlemalloc) {PetscFree(a->i); PetscFree(a->j);}
    if (a->diag) PetscFree(a->diag);
    if (a->ilen) PetscFree(a->ilen);
    if (a->imax) PetscFree(a->imax);
    if (a->solve_work) PetscFree(a->solve_work);
    if (a->inode.size) PetscFree(a->inode.size);
    PetscFree(a); 
    PetscMemcpy(A,C,sizeof(struct _Mat)); 
    PetscHeaderDestroy(C);
  }
  return 0;
}

static int MatDiagonalScale_SeqAIJ(Mat A,Vec ll,Vec rr)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  Scalar     *l,*r,x,*v;
  int        i,j,m = a->m, n = a->n, M, nz = a->nz, *jj,shift = a->indexshift;

  if (ll) {
    VecGetArray(ll,&l); VecGetSize(ll,&m);
    if (m != a->m) SETERRQ(1,"MatDiagonalScale_SeqAIJ:Left scaling vector wrong length");
    v = a->a;
    for ( i=0; i<m; i++ ) {
      x = l[i];
      M = a->i[i+1] - a->i[i];
      for ( j=0; j<M; j++ ) { (*v++) *= x;} 
    }
    PLogFlops(nz);
  }
  if (rr) {
    VecGetArray(rr,&r); VecGetSize(rr,&n);
    if (n != a->n) SETERRQ(1,"MatDiagonalScale_SeqAIJ:Right scaling vector wrong length");
    v = a->a; jj = a->j;
    for ( i=0; i<nz; i++ ) {
      (*v++) *= r[*jj++ + shift]; 
    }
    PLogFlops(nz);
  }
  return 0;
}

static int MatGetSubMatrix_SeqAIJ(Mat A,IS isrow,IS iscol,MatGetSubMatrixCall scall,Mat *B)
{
  Mat_SeqAIJ   *a = (Mat_SeqAIJ *) A->data,*c;
  int          nznew, *smap, i, k, kstart, kend, ierr, oldcols = a->n,*lens;
  int          row,mat_i,*mat_j,tcol,first,step,*mat_ilen;
  register int sum,lensi;
  int          *irow, *icol, nrows, ncols, shift = a->indexshift,*ssmap;
  int          *starts,*j_new,*i_new,*aj = a->j, *ai = a->i,ii,*ailen = a->ilen;
  Scalar       *a_new,*mat_a;
  Mat          C;

  ierr = ISSorted(iscol,(PetscTruth*)&i);
  if (!i) SETERRQ(1,"MatGetSubmatrices_SeqAIJ:IS is not sorted");

  ierr = ISGetIndices(isrow,&irow); CHKERRQ(ierr);
  ierr = ISGetSize(isrow,&nrows); CHKERRQ(ierr);
  ierr = ISGetSize(iscol,&ncols); CHKERRQ(ierr);

  if (ISStrideGetInfo(iscol,&first,&step) && step == 1) { /* no need to sort */
    /* special case of contiguous rows */
    lens   = (int *) PetscMalloc((ncols+nrows+1)*sizeof(int)); CHKPTRQ(lens);
    starts = lens + ncols;
    /* loop over new rows determining lens and starting points */
    for (i=0; i<nrows; i++) {
      kstart  = ai[irow[i]]+shift; 
      kend    = kstart + ailen[irow[i]];
      for ( k=kstart; k<kend; k++ ) {
        if (aj[k]+shift >= first) {
          starts[i] = k;
          break;
	}
      }
      sum = 0;
      while (k < kend) {
        if (aj[k++]+shift >= first+ncols) break;
        sum++;
      }
      lens[i] = sum;
    }
    /* create submatrix */
    if (scall == MAT_REUSE_MATRIX) {
      int n_cols,n_rows;
      ierr = MatGetSize(*B,&n_rows,&n_cols); CHKERRQ(ierr);
      if (n_rows != nrows || n_cols != ncols) SETERRQ(1,"MatGetSubMatrix_SeqAIJ:");
      ierr = MatZeroEntries(*B); CHKERRQ(ierr);
      C = *B;
    }
    else {  
      ierr = MatCreateSeqAIJ(A->comm,nrows,ncols,0,lens,&C);CHKERRQ(ierr);
    }
    c = (Mat_SeqAIJ*) C->data;

    /* loop over rows inserting into submatrix */
    a_new    = c->a;
    j_new    = c->j;
    i_new    = c->i;
    i_new[0] = -shift;
    for (i=0; i<nrows; i++) {
      ii    = starts[i];
      lensi = lens[i];
      for ( k=0; k<lensi; k++ ) {
        *j_new++ = aj[ii+k] - first;
      }
      PetscMemcpy(a_new,a->a + starts[i],lensi*sizeof(Scalar));
      a_new      += lensi;
      i_new[i+1]  = i_new[i] + lensi;
      c->ilen[i]  = lensi;
    }
    PetscFree(lens);
  }
  else {
    ierr = ISGetIndices(iscol,&icol); CHKERRQ(ierr);
    smap  = (int *) PetscMalloc((1+oldcols)*sizeof(int)); CHKPTRQ(smap);
    ssmap = smap + shift;
    lens  = (int *) PetscMalloc((1+nrows)*sizeof(int)); CHKPTRQ(lens);
    PetscMemzero(smap,oldcols*sizeof(int));
    for ( i=0; i<ncols; i++ ) smap[icol[i]] = i+1;
    /* determine lens of each row */
    for (i=0; i<nrows; i++) {
      kstart  = ai[irow[i]]+shift; 
      kend    = kstart + a->ilen[irow[i]];
      lens[i] = 0;
      for ( k=kstart; k<kend; k++ ) {
        if (ssmap[aj[k]]) {
          lens[i]++;
        }
      }
    }
    /* Create and fill new matrix */
    if (scall == MAT_REUSE_MATRIX) {
      c = (Mat_SeqAIJ *)((*B)->data);

      if (c->m  != nrows || c->n != ncols) SETERRQ(1,"MatGetSubMatrix_SeqAIJ:");
      if (PetscMemcmp(c->ilen,lens, c->m *sizeof(int))) {
        SETERRQ(1,"MatGetSubmatrices_SeqAIJ:Cannot reuse matrix. wrong no of nonzeros");
      }
      PetscMemzero(c->ilen,c->m*sizeof(int));
      C = *B;
    }
    else {  
      ierr = MatCreateSeqAIJ(A->comm,nrows,ncols,0,lens,&C);CHKERRQ(ierr);
    }
    c = (Mat_SeqAIJ *)(C->data);
    for (i=0; i<nrows; i++) {
      row    = irow[i];
      nznew  = 0;
      kstart = ai[row]+shift; 
      kend   = kstart + a->ilen[row];
      mat_i  = c->i[i]+shift;
      mat_j  = c->j + mat_i; 
      mat_a  = c->a + mat_i;
      mat_ilen = c->ilen + i;
      for ( k=kstart; k<kend; k++ ) {
        if ((tcol=ssmap[a->j[k]])) {
          *mat_j++ = tcol - (!shift);
          *mat_a++ = a->a[k];
          (*mat_ilen)++;

        }
      }
    }
    /* Free work space */
    ierr = ISRestoreIndices(iscol,&icol); CHKERRQ(ierr);
    PetscFree(smap); PetscFree(lens);
  }
  ierr = MatAssemblyBegin(C,FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,FINAL_ASSEMBLY); CHKERRQ(ierr);

  ierr = ISRestoreIndices(isrow,&irow); CHKERRQ(ierr);
  *B = C;
  return 0;
}

/*
     note: This can only work for identity for row and col. It would 
   be good to check this and otherwise generate an error.
*/
static int MatILUFactor_SeqAIJ(Mat inA,IS row,IS col,double efill,int fill)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) inA->data;
  int        ierr;
  Mat        outA;

  if (fill != 0) SETERRQ(1,"MatILUFactor_SeqAIJ:Only fill=0 supported");

  outA          = inA; 
  inA->factor   = FACTOR_LU;
  a->row        = row;
  a->col        = col;

  a->solve_work = (Scalar *) PetscMalloc( (a->m+1)*sizeof(Scalar)); CHKPTRQ(a->solve_work);

  if (!a->diag) {
    ierr = MatMarkDiag_SeqAIJ(inA); CHKERRQ(ierr);
  }
  ierr = MatLUFactorNumeric_SeqAIJ(inA,&outA); CHKERRQ(ierr);
  return 0;
}

#include "pinclude/plapack.h"
static int MatScale_SeqAIJ(Scalar *alpha,Mat inA)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) inA->data;
  int        one = 1;
  BLscal_( &a->nz, alpha, a->a, &one );
  PLogFlops(a->nz);
  return 0;
}

static int MatGetSubMatrices_SeqAIJ(Mat A,int n, IS *irow,IS *icol,MatGetSubMatrixCall scall,
                                    Mat **B)
{
  int ierr,i;

  if (scall == MAT_INITIAL_MATRIX) {
    *B = (Mat *) PetscMalloc( (n+1)*sizeof(Mat) ); CHKPTRQ(*B);
  }

  for ( i=0; i<n; i++ ) {
    ierr = MatGetSubMatrix(A,irow[i],icol[i],scall,&(*B)[i]); CHKERRQ(ierr);
  }
  return 0;
}

static int MatIncreaseOverlap_SeqAIJ(Mat A, int is_max, IS *is, int ov)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;
  int        shift, row, i,j,k,l,m,n, *idx,ierr, *nidx, isz, val;
  int        start, end, *ai, *aj;
  char       *table;
  shift = a->indexshift;
  m     = a->m;
  ai    = a->i;
  aj    = a->j+shift;

  if (ov < 0)  SETERRQ(1,"MatIncreaseOverlap_SeqAIJ: illegal overlap value used");

  table = (char *) PetscMalloc((m/BITSPERBYTE +1)*sizeof(char)); CHKPTRQ(table); 
  nidx  = (int *) PetscMalloc((m+1)*sizeof(int)); CHKPTRQ(nidx); 

  for ( i=0; i<is_max; i++ ) {
    /* Initialise the two local arrays */
    isz  = 0;
    PetscMemzero(table,(m/BITSPERBYTE +1)*sizeof(char));
                 
                /* Extract the indices, assume there can be duplicate entries */
    ierr = ISGetIndices(is[i],&idx);  CHKERRQ(ierr);
    ierr = ISGetSize(is[i],&n);  CHKERRQ(ierr);
    
    /* Enter these into the temp arrays i.e mark table[row], enter row into new index */
    for ( j=0; j<n ; ++j){
      if(!BT_LOOKUP(table, idx[j])) { nidx[isz++] = idx[j];}
    }
    ierr = ISRestoreIndices(is[i],&idx);  CHKERRQ(ierr);
    ierr = ISDestroy(is[i]); CHKERRQ(ierr);
    
    k = 0;
    for ( j=0; j<ov; j++){ /* for each overlap*/
      n = isz;
      for ( ; k<n ; k++){ /* do only those rows in nidx[k], which are not done yet */
        row   = nidx[k];
        start = ai[row];
        end   = ai[row+1];
        for ( l = start; l<end ; l++){
          val = aj[l] + shift;
          if (!BT_LOOKUP(table,val)) {nidx[isz++] = val;}
        }
      }
    }
    ierr = ISCreateSeq(MPI_COMM_SELF, isz, nidx, (is+i)); CHKERRQ(ierr);
  }
  PetscFree(table);
  PetscFree(nidx);
  return 0;
}

int MatPrintHelp_SeqAIJ(Mat A)
{
  static int called = 0; 
  MPI_Comm   comm = A->comm;

  if (called) return 0; else called = 1;
  PetscPrintf(comm," Options for MATSEQAIJ and MATMPIAIJ matrix formats (the defaults):\n");
  PetscPrintf(comm,"  -mat_lu_pivotthreshold <threshold>\n");
  PetscPrintf(comm,"  -mat_aij_oneindex - internal indices begin at 1 instead of the default 0.\n");
  PetscPrintf(comm,"  -mat_aij_no_inode  - Do not use inodes\n");
  PetscPrintf(comm,"  -mat_aij_inode_limit <limit> - Set inode limit (max limit=5)\n");
#if defined(HAVE_ESSL)
  PetscPrintf(comm,"  -mat_aij_essl  - Use IBM sparse LU factorization and solve.\n");
#endif
  return 0;
}
static int MatEqual_SeqAIJ(Mat A,Mat B, PetscTruth* flg);
/* -------------------------------------------------------------------*/
static struct _MatOps MatOps = {MatSetValues_SeqAIJ,
       MatGetRow_SeqAIJ,MatRestoreRow_SeqAIJ,
       MatMult_SeqAIJ,MatMultAdd_SeqAIJ,
       MatMultTrans_SeqAIJ,MatMultTransAdd_SeqAIJ,
       MatSolve_SeqAIJ,MatSolveAdd_SeqAIJ,
       MatSolveTrans_SeqAIJ,MatSolveTransAdd_SeqAIJ,
       MatLUFactor_SeqAIJ,0,
       MatRelax_SeqAIJ,
       MatTranspose_SeqAIJ,
       MatGetInfo_SeqAIJ,MatEqual_SeqAIJ,
       MatGetDiagonal_SeqAIJ,MatDiagonalScale_SeqAIJ,MatNorm_SeqAIJ,
       0,MatAssemblyEnd_SeqAIJ,
       MatCompress_SeqAIJ,
       MatSetOption_SeqAIJ,MatZeroEntries_SeqAIJ,MatZeroRows_SeqAIJ,
       MatGetReordering_SeqAIJ,
       MatLUFactorSymbolic_SeqAIJ,MatLUFactorNumeric_SeqAIJ,0,0,
       MatGetSize_SeqAIJ,MatGetSize_SeqAIJ,MatGetOwnershipRange_SeqAIJ,
       MatILUFactorSymbolic_SeqAIJ,0,
       0,0,MatConvert_SeqAIJ,
       MatGetSubMatrix_SeqAIJ,0,
       MatConvertSameType_SeqAIJ,0,0,
       MatILUFactor_SeqAIJ,0,0,
       MatGetSubMatrices_SeqAIJ,MatIncreaseOverlap_SeqAIJ,
       MatGetValues_SeqAIJ,0,
       MatPrintHelp_SeqAIJ,
       MatScale_SeqAIJ};

extern int MatUseSuperLU_SeqAIJ(Mat);
extern int MatUseEssl_SeqAIJ(Mat);
extern int MatUseDXML_SeqAIJ(Mat);

/*@C
   MatCreateSeqAIJ - Creates a sparse matrix in AIJ (compressed row) format
   (the default parallel PETSc format).  For good matrix assembly performance
   the user should preallocate the matrix storage by setting the parameter nz
   (or nzz).  By setting these parameters accurately, performance can be
   increased by more than a factor of 50.

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
   storage.  That is, the stored row and column indices can begin at
   either one (as in Fortran) or zero.  See the users' manual for details.

   Specify the preallocated storage with either nz or nnz (not both).
   Set nz=PETSC_DEFAULT and nnz=PETSC_NULL for PETSc to control dynamic memory 
   allocation.  For additional details, see the users' manual chapter on
   matrices and the file $(PETSC_DIR)/Performance.

   By default, this format uses inodes (identical nodes) when possible, to 
   improve numerical efficiency of Matrix vector products and solves. We 
   search for consecutive rows with the same nonzero structure, thereby
   reusing matrix information to achieve increased efficiency.

   Options Database Keys:
$    -mat_aij_no_inode  - Do not use inodes
$    -mat_aij_inode_limit <limit> - Set inode limit.
$        (max limit=5)
$    -mat_aij_oneindex - Internally use indexing starting at 1
$        rather than 0.  Note: When calling MatSetValues(),
$        the user still MUST index entries starting at 0!

.seealso: MatCreate(), MatCreateMPIAIJ(), MatSetValues()
@*/
int MatCreateSeqAIJ(MPI_Comm comm,int m,int n,int nz,int *nnz, Mat *A)
{
  Mat        B;
  Mat_SeqAIJ *b;
  int        i, len, ierr, flg;

  *A                  = 0;
  PetscHeaderCreate(B,_Mat,MAT_COOKIE,MATSEQAIJ,comm);
  PLogObjectCreate(B);
  B->data             = (void *) (b = PetscNew(Mat_SeqAIJ)); CHKPTRQ(b);
  PetscMemzero(b,sizeof(Mat_SeqAIJ));
  PetscMemcpy(&B->ops,&MatOps,sizeof(struct _MatOps));
  B->destroy          = MatDestroy_SeqAIJ;
  B->view             = MatView_SeqAIJ;
  B->factor           = 0;
  B->lupivotthreshold = 1.0;
  ierr = OptionsGetDouble(PETSC_NULL,"-mat_lu_pivotthreshold",&B->lupivotthreshold,
                          &flg); CHKERRQ(ierr);
  b->ilu_preserve_row_sums = PETSC_FALSE;
  ierr = OptionsHasName(PETSC_NULL,"-pc_ilu_preserve_row_sums",
                        (int*) &b->ilu_preserve_row_sums); CHKERRQ(ierr);
  b->row              = 0;
  b->col              = 0;
  b->indexshift       = 0;
  b->reallocs         = 0;
  ierr = OptionsHasName(PETSC_NULL,"-mat_aij_oneindex", &flg); CHKERRQ(ierr);
  if (flg) b->indexshift = -1;
  
  b->m = m; B->m = m; B->M = m;
  b->n = n; B->n = n; B->N = n;
  b->imax = (int *) PetscMalloc( (m+1)*sizeof(int) ); CHKPTRQ(b->imax);
  if (nnz == PETSC_NULL) {
    if (nz == PETSC_DEFAULT) nz = 10;
    else if (nz <= 0)        nz = 1;
    for ( i=0; i<m; i++ ) b->imax[i] = nz;
    nz = nz*m;
  }
  else {
    nz = 0;
    for ( i=0; i<m; i++ ) {b->imax[i] = nnz[i]; nz += nnz[i];}
  }

  /* allocate the matrix space */
  len     = nz*(sizeof(int) + sizeof(Scalar)) + (b->m+1)*sizeof(int);
  b->a  = (Scalar *) PetscMalloc( len ); CHKPTRQ(b->a);
  b->j  = (int *) (b->a + nz);
  PetscMemzero(b->j,nz*sizeof(int));
  b->i  = b->j + nz;
  b->singlemalloc = 1;

  b->i[0] = -b->indexshift;
  for (i=1; i<m+1; i++) {
    b->i[i] = b->i[i-1] + b->imax[i-1];
  }

  /* b->ilen will count nonzeros in each row so far. */
  b->ilen = (int *) PetscMalloc((m+1)*sizeof(int)); 
  PLogObjectMemory(B,len+2*(m+1)*sizeof(int)+sizeof(struct _Mat)+sizeof(Mat_SeqAIJ));
  for ( i=0; i<b->m; i++ ) { b->ilen[i] = 0;}

  b->nz               = 0;
  b->maxnz            = nz;
  b->sorted           = 0;
  b->roworiented      = 1;
  b->nonew            = 0;
  b->diag             = 0;
  b->solve_work       = 0;
  b->spptr            = 0;
  b->inode.node_count = 0;
  b->inode.size       = 0;
  b->inode.limit      = 5;
  b->inode.max_limit  = 5;

  *A = B;
  /*  SuperLU is not currently supported through PETSc */
#if defined(HAVE_SUPERLU)
  ierr = OptionsHasName(PETSC_NULL,"-mat_aij_superlu", &flg); CHKERRQ(ierr);
  if (flg) { ierr = MatUseSuperLU_SeqAIJ(B); CHKERRQ(ierr); }
#endif
  ierr = OptionsHasName(PETSC_NULL,"-mat_aij_essl", &flg); CHKERRQ(ierr);
  if (flg) { ierr = MatUseEssl_SeqAIJ(B); CHKERRQ(ierr); }
  ierr = OptionsHasName(PETSC_NULL,"-mat_aij_dxml", &flg); CHKERRQ(ierr);
  if (flg) {
    if (!b->indexshift) SETERRQ(1,"MatCreateSeqAIJ:need -mat_aij_oneindex with -mat_aij_dxml");
    ierr = MatUseDXML_SeqAIJ(B); CHKERRQ(ierr);
  }
  ierr = OptionsHasName(PETSC_NULL,"-help", &flg); CHKERRQ(ierr);
  if (flg) {ierr = MatPrintHelp(B); CHKERRQ(ierr); }
  return 0;
}

int MatConvertSameType_SeqAIJ(Mat A,Mat *B,int cpvalues)
{
  Mat        C;
  Mat_SeqAIJ *c,*a = (Mat_SeqAIJ *) A->data;
  int        i,len, m = a->m,shift = a->indexshift;

  *B = 0;
  PetscHeaderCreate(C,_Mat,MAT_COOKIE,MATSEQAIJ,A->comm);
  PLogObjectCreate(C);
  C->data       = (void *) (c = PetscNew(Mat_SeqAIJ)); CHKPTRQ(c);
  PetscMemcpy(&C->ops,&A->ops,sizeof(struct _MatOps));
  C->destroy    = MatDestroy_SeqAIJ;
  C->view       = MatView_SeqAIJ;
  C->factor     = A->factor;
  c->row        = 0;
  c->col        = 0;
  c->indexshift = shift;
  C->assembled  = PETSC_TRUE;

  c->m = C->m   = a->m;
  c->n = C->n   = a->n;
  C->M          = a->m;
  C->N          = a->n;

  c->imax       = (int *) PetscMalloc((m+1)*sizeof(int)); CHKPTRQ(c->imax);
  c->ilen       = (int *) PetscMalloc((m+1)*sizeof(int)); CHKPTRQ(c->ilen);
  for ( i=0; i<m; i++ ) {
    c->imax[i] = a->imax[i];
    c->ilen[i] = a->ilen[i]; 
  }

  /* allocate the matrix space */
  c->singlemalloc = 1;
  len     = (m+1)*sizeof(int)+(a->i[m])*(sizeof(Scalar)+sizeof(int));
  c->a  = (Scalar *) PetscMalloc( len ); CHKPTRQ(c->a);
  c->j  = (int *) (c->a + a->i[m] + shift);
  c->i  = c->j + a->i[m] + shift;
  PetscMemcpy(c->i,a->i,(m+1)*sizeof(int));
  if (m > 0) {
    PetscMemcpy(c->j,a->j,(a->i[m]+shift)*sizeof(int));
    if (cpvalues == COPY_VALUES) {
      PetscMemcpy(c->a,a->a,(a->i[m]+shift)*sizeof(Scalar));
    }
  }

  PLogObjectMemory(C,len+2*(m+1)*sizeof(int)+sizeof(struct _Mat)+sizeof(Mat_SeqAIJ));  
  c->sorted      = a->sorted;
  c->roworiented = a->roworiented;
  c->nonew       = a->nonew;
  c->ilu_preserve_row_sums = a->ilu_preserve_row_sums;

  if (a->diag) {
    c->diag = (int *) PetscMalloc( (m+1)*sizeof(int) ); CHKPTRQ(c->diag);
    PLogObjectMemory(C,(m+1)*sizeof(int));
    for ( i=0; i<m; i++ ) {
      c->diag[i] = a->diag[i];
    }
  }
  else c->diag          = 0;
  c->inode.limit        = a->inode.limit;
  c->inode.max_limit    = a->inode.max_limit;
  if (a->inode.size){
    c->inode.size       = (int *) PetscMalloc( m *sizeof(int) ); CHKPTRQ(c->inode.size);
    c->inode.node_count = a->inode.node_count;
    PetscMemcpy( c->inode.size, a->inode.size, m*sizeof(int));
  } else {
    c->inode.size       = 0;
    c->inode.node_count = 0;
  }
  c->nz                 = a->nz;
  c->maxnz              = a->maxnz;
  c->solve_work         = 0;
  c->spptr              = 0;      /* Dangerous -I'm throwing away a->spptr */

  *B = C;
  return 0;
}

int MatLoad_SeqAIJ(Viewer viewer,MatType type,Mat *A)
{
  Mat_SeqAIJ   *a;
  Mat          B;
  int          i, nz, ierr, fd, header[4],size,*rowlengths = 0,M,N,shift;
  MPI_Comm     comm;
  
  PetscObjectGetComm((PetscObject) viewer,&comm);
  MPI_Comm_size(comm,&size);
  if (size > 1) SETERRQ(1,"MatLoad_SeqAIJ:view must have one processor");
  ierr = ViewerBinaryGetDescriptor(viewer,&fd); CHKERRQ(ierr);
  ierr = PetscBinaryRead(fd,header,4,BINARY_INT); CHKERRQ(ierr);
  if (header[0] != MAT_COOKIE) SETERRQ(1,"MatLoad_SeqAIJ:not matrix object in file");
  M = header[1]; N = header[2]; nz = header[3];

  /* read in row lengths */
  rowlengths = (int*) PetscMalloc( M*sizeof(int) ); CHKPTRQ(rowlengths);
  ierr = PetscBinaryRead(fd,rowlengths,M,BINARY_INT); CHKERRQ(ierr);

  /* create our matrix */
  ierr = MatCreateSeqAIJ(comm,M,N,0,rowlengths,A); CHKERRQ(ierr);
  B = *A;
  a = (Mat_SeqAIJ *) B->data;
  shift = a->indexshift;

  /* read in column indices and adjust for Fortran indexing*/
  ierr = PetscBinaryRead(fd,a->j,nz,BINARY_INT); CHKERRQ(ierr);
  if (shift) {
    for ( i=0; i<nz; i++ ) {
      a->j[i] += 1;
    }
  }

  /* read in nonzero values */
  ierr = PetscBinaryRead(fd,a->a,nz,BINARY_SCALAR); CHKERRQ(ierr);

  /* set matrix "i" values */
  a->i[0] = -shift;
  for ( i=1; i<= M; i++ ) {
    a->i[i]      = a->i[i-1] + rowlengths[i-1];
    a->ilen[i-1] = rowlengths[i-1];
  }
  PetscFree(rowlengths);   

  ierr = MatAssemblyBegin(B,FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,FINAL_ASSEMBLY); CHKERRQ(ierr);
  return 0;
}

static int MatEqual_SeqAIJ(Mat A,Mat B, PetscTruth* flg)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *)A->data, *b = (Mat_SeqAIJ *)B->data;

  if (B->type !=MATSEQAIJ)SETERRQ(1,"MatEqual_SeqAIJ:Matrices must be same type");

  /* If the  matrix dimensions are not equal, or no of nonzeros or shift */
  if ((a->m != b->m ) || (a->n !=b->n) ||( a->nz != b->nz)|| 
      (a->indexshift != b->indexshift)) {
    *flg = PETSC_FALSE; return 0; 
  }
  
  /* if the a->i are the same */
  if (PetscMemcmp(a->i,b->i, (a->n+1)*sizeof(int))) { 
    *flg = PETSC_FALSE; return 0;
  }
  
  /* if a->j are the same */
  if (PetscMemcmp(a->j, b->j, (a->nz)*sizeof(int))) { 
    *flg = PETSC_FALSE; return 0;
  }
  
  /* if a->a are the same */
  if (PetscMemcmp(a->a, b->a, (a->nz)*sizeof(Scalar))) {
    *flg = PETSC_FALSE; return 0;
  }
  *flg = PETSC_TRUE; 
  return 0;
  
}
