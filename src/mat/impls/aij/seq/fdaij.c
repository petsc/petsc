/*$Id: fdaij.c,v 1.40 2001/06/21 21:16:21 bsmith Exp $*/

#include "src/mat/impls/aij/seq/aij.h"
#include "src/vec/vecimpl.h"

EXTERN int MatGetColumnIJ_SeqAIJ(Mat,int,PetscTruth,int*,int**,int**,PetscTruth*);
EXTERN int MatRestoreColumnIJ_SeqAIJ(Mat,int,PetscTruth,int*,int**,int**,PetscTruth*);

#undef __FUNCT__  
#define __FUNCT__ "MatFDColoringCreate_SeqAIJ"
int MatFDColoringCreate_SeqAIJ(Mat mat,ISColoring iscoloring,MatFDColoring c)
{
  int        i,*is,n,nrows,N = mat->N,j,k,m,*rows,ierr,*ci,*cj,ncols,col;
  int        nis = iscoloring->n,*rowhit,*columnsforrow,l;
  IS         *isa;
  PetscTruth done,flg;

  PetscFunctionBegin;
  if (!mat->assembled) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Matrix must be assembled by calls to MatAssemblyBegin/End();");
  }

  ierr = ISColoringGetIS(iscoloring,PETSC_IGNORE,&isa);CHKERRQ(ierr);
  c->M       = mat->M;  /* set total rows, columns and local rows */
  c->N       = mat->N;
  c->m       = mat->M;
  c->rstart  = 0;

  c->ncolors = nis;
  ierr       = PetscMalloc(nis*sizeof(int),&c->ncolumns);CHKERRQ(ierr);
  ierr       = PetscMalloc(nis*sizeof(int*),&c->columns);CHKERRQ(ierr); 
  ierr       = PetscMalloc(nis*sizeof(int),&c->nrows);CHKERRQ(ierr);
  ierr       = PetscMalloc(nis*sizeof(int*),&c->rows);CHKERRQ(ierr);
  ierr       = PetscMalloc(nis*sizeof(int*),&c->columnsforrow);CHKERRQ(ierr);

  /*
      Calls the _SeqAIJ() version of these routines to make sure it does not 
     get the reduced (by inodes) version of I and J
  */
  ierr = MatGetColumnIJ_SeqAIJ(mat,0,PETSC_FALSE,&ncols,&ci,&cj,&done);CHKERRQ(ierr);

  /*
     Temporary option to allow for debugging/testing
  */
  ierr = PetscOptionsHasName(PETSC_NULL,"-matfdcoloring_slow",&flg);CHKERRQ(ierr);

  ierr = PetscMalloc((N+1)*sizeof(int),&rowhit);CHKERRQ(ierr);
  ierr = PetscMalloc((N+1)*sizeof(int),&columnsforrow);CHKERRQ(ierr);

  for (i=0; i<nis; i++) {
    ierr = ISGetLocalSize(isa[i],&n);CHKERRQ(ierr);
    ierr = ISGetIndices(isa[i],&is);CHKERRQ(ierr);
    c->ncolumns[i] = n;
    if (n) {
      ierr = PetscMalloc(n*sizeof(int),&c->columns[i]);CHKERRQ(ierr);
      ierr = PetscMemcpy(c->columns[i],is,n*sizeof(int));CHKERRQ(ierr);
    } else {
      c->columns[i]  = 0;
    }

    if (!flg) { /* ------------------------------------------------------------------------------*/
      /* fast, crude version requires O(N*N) work */
      ierr = PetscMemzero(rowhit,N*sizeof(int));CHKERRQ(ierr);
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
      ierr        = PetscMalloc((nrows+1)*sizeof(int),&c->rows[i]);CHKERRQ(ierr);
      ierr        = PetscMalloc((nrows+1)*sizeof(int),&c->columnsforrow[i]);CHKERRQ(ierr);
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
      int currentcol,fm,mfm;
      rowhit[N] = N;
      nrows     = 0;
      /* loop over columns */
      for (j=0; j<n; j++) {
        col   = is[j];
        rows  = cj + ci[col]; 
        m     = ci[col+1] - ci[col];
        /* loop over columns marking them in rowhit */
        fm    = N; /* fm points to first entry in linked list */
        for (k=0; k<m; k++) {
          currentcol = *rows++;
	  /* is it already in the list? */
          do {
            mfm  = fm;
            fm   = rowhit[fm];
          } while (fm < currentcol);
          /* not in list so add it */
          if (fm != currentcol) {
            nrows++;
            columnsforrow[currentcol] = col;
            /* next three lines insert new entry into linked list */
            rowhit[mfm]               = currentcol;
            rowhit[currentcol]        = fm;
            fm                        = currentcol; 
            /* fm points to present position in list since we know the columns are sorted */
          } else {
            SETERRQ(PETSC_ERR_PLIB,"Detected invalid coloring");
          }

        }
      }
      c->nrows[i] = nrows;
      ierr        = PetscMalloc((nrows+1)*sizeof(int),&c->rows[i]);CHKERRQ(ierr);
      ierr        = PetscMalloc((nrows+1)*sizeof(int),&c->columnsforrow[i]);CHKERRQ(ierr);
      /* now store the linked list of rows into c->rows[i] */
      nrows       = 0;
      fm          = rowhit[N];
      do {
        c->rows[i][nrows]            = fm;
        c->columnsforrow[i][nrows++] = columnsforrow[fm];
        fm                           = rowhit[fm];
      } while (fm < N);
    } /* ---------------------------------------------------------------------------------------*/
    ierr = ISRestoreIndices(isa[i],&is);CHKERRQ(ierr);  
  }
  ierr = MatRestoreColumnIJ_SeqAIJ(mat,0,PETSC_FALSE,&ncols,&ci,&cj,&done);CHKERRQ(ierr);

  ierr = PetscFree(rowhit);CHKERRQ(ierr);
  ierr = PetscFree(columnsforrow);CHKERRQ(ierr);

  /* Optimize by adding the vscale, and scaleforrow[][] fields */
  /*
       see the version for MPIAIJ
  */
  ierr = VecCreateGhost(mat->comm,mat->m,PETSC_DETERMINE,0,PETSC_NULL,&c->vscale);CHKERRQ(ierr)
  ierr = PetscMalloc(c->ncolors*sizeof(int*),&c->vscaleforrow);CHKERRQ(ierr);
  for (k=0; k<c->ncolors; k++) { 
    ierr = PetscMalloc((c->nrows[k]+1)*sizeof(int),&c->vscaleforrow[k]);CHKERRQ(ierr);
    for (l=0; l<c->nrows[k]; l++) {
      col = c->columnsforrow[k][l];
      c->vscaleforrow[k][l] = col;
    }
  }
  ierr = ISColoringRestoreIS(iscoloring,&isa);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
     Makes a longer coloring[] array and calls the usual code with that
*/
#undef __FUNCT__  
#define __FUNCT__ "MatColoringPatch_SeqAIJ_Inode"
int MatColoringPatch_SeqAIJ_Inode(Mat mat,int nin,int ncolors,const ISColoringValue coloring[],ISColoring *iscoloring)
{
  Mat_SeqAIJ      *a = (Mat_SeqAIJ*)mat->data;
  int             n = mat->n,ierr,m = a->inode.node_count,j,*ns = a->inode.size,row;
  int             *colorused,i;
  ISColoringValue *newcolor;

  PetscFunctionBegin;
  ierr = PetscMalloc((n+1)*sizeof(int),&newcolor);CHKERRQ(ierr);
  /* loop over inodes, marking a color for each column*/
  row = 0;
  for (i=0; i<m; i++){
    for (j=0; j<ns[i]; j++) {
      newcolor[row++] = coloring[i] + j*ncolors;
    }
  }

  /* eliminate unneeded colors */
  ierr = PetscMalloc(5*ncolors*sizeof(int),&colorused);CHKERRQ(ierr);
  ierr = PetscMemzero(colorused,5*ncolors*sizeof(int));CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    colorused[newcolor[i]] = 1;
  }

  for (i=1; i<5*ncolors; i++) {
    colorused[i] += colorused[i-1];
  }
  ncolors = colorused[5*ncolors-1];
  for (i=0; i<n; i++) {
    newcolor[i] = colorused[newcolor[i]];
  }
  ierr = PetscFree(colorused);CHKERRQ(ierr);
  ierr = ISColoringCreate(mat->comm,n,newcolor,iscoloring);CHKERRQ(ierr);
  ierr = PetscFree((void*)coloring);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}






