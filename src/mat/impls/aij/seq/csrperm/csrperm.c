#define PETSCMAT_DLL

/*
  Defines basic operations for the MATSEQCSRPERM matrix class.
  This class is derived from the MATSEQAIJ class and retains the 
  compressed row storage (aka Yale sparse matrix format) but augments 
  it with some permutation information that enables some operations 
  to be more vectorizable.  A physically rearranged copy of the matrix 
  may be stored if the user desires.

  Eventually a variety of permutations may be supported.
*/

#include "../src/mat/impls/aij/seq/aij.h"

#define NDIM 512
    /* NDIM specifies how many rows at a time we should work with when 
     * performing the vectorized mat-vec.  This depends on various factors 
     * such as vector register length, etc., and I really need to add a 
     * way for the user (or the library) to tune this.  I'm setting it to  
     * 512 for now since that is what Ed D'Azevedo was using in his Fortran 
     * routines. */

typedef struct {
  PetscInt  ngroup;
  PetscInt *xgroup;
    /* Denotes where groups of rows with same number of nonzeros 
     * begin and end, i.e., xgroup[i] gives us the position in iperm[] 
     * where the ith group begins. */
  PetscInt *nzgroup; /*  how many nonzeros each row that is a member of group i has. */
  PetscInt *iperm;  /* The permutation vector. */

  /* Flag that indicates whether we need to clean up permutation 
   * information during the MatDestroy. */
  PetscTruth CleanUpCSRPERM;

  /* Some of this stuff is for Ed's recursive triangular solve.
   * I'm not sure what I need yet. */
  PetscInt blocksize;
  PetscInt nstep;
  PetscInt *jstart_list;
  PetscInt *jend_list;
  PetscInt *action_list;
  PetscInt *ngroup_list;
  PetscInt **ipointer_list;
  PetscInt **xgroup_list;
  PetscInt **nzgroup_list;
  PetscInt **iperm_list;
} Mat_SeqCSRPERM;

extern PetscErrorCode MatAssemblyEnd_SeqAIJ(Mat,MatAssemblyType);

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatConvert_SeqCSRPERM_SeqAIJ"
PetscErrorCode PETSCMAT_DLLEXPORT MatConvert_SeqCSRPERM_SeqAIJ(Mat A,MatType type,MatReuse reuse,Mat *newmat)
{
  /* This routine is only called to convert a MATCSRPERM to its base PETSc type, */
  /* so we will ignore 'MatType type'. */
  PetscErrorCode ierr;
  Mat            B = *newmat;
  Mat_SeqCSRPERM *csrperm=(Mat_SeqCSRPERM*)A->spptr;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatDuplicate(A,MAT_COPY_VALUES,&B);CHKERRQ(ierr);
  }

  /* Reset the original function pointers. */
  B->ops->assemblyend = MatAssemblyEnd_SeqAIJ;
  B->ops->destroy     = MatDestroy_SeqAIJ;
  B->ops->duplicate   = MatDuplicate_SeqAIJ;

  /* Free everything in the Mat_SeqCSRPERM data structure. 
   * We don't free the Mat_SeqCSRPERM struct itself, as this will 
   * cause problems later when MatDestroy() tries to free it. */
  if(csrperm->CleanUpCSRPERM) {
    ierr = PetscFree(csrperm->xgroup);CHKERRQ(ierr);
    ierr = PetscFree(csrperm->nzgroup);CHKERRQ(ierr);
    ierr = PetscFree(csrperm->iperm);CHKERRQ(ierr);
  }

  /* Change the type of B to MATSEQAIJ. */
  ierr = PetscObjectChangeTypeName( (PetscObject)B, MATSEQAIJ);CHKERRQ(ierr);
  
  *newmat = B;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_SeqCSRPERM"
PetscErrorCode MatDestroy_SeqCSRPERM(Mat A)
{
  PetscErrorCode ierr;
  Mat_SeqCSRPERM *csrperm = (Mat_SeqCSRPERM *) A->spptr;

  PetscFunctionBegin;
  /* Free everything in the Mat_SeqCSRPERM data structure. 
   * Note that we don't need to free the Mat_SeqCSRPERM struct 
   * itself, as MatDestroy() will do so. */
  if(csrperm->CleanUpCSRPERM) {
    ierr = PetscFree(csrperm->xgroup);CHKERRQ(ierr);
    ierr = PetscFree(csrperm->nzgroup);CHKERRQ(ierr);
    ierr = PetscFree(csrperm->iperm);CHKERRQ(ierr);
  }

  /* Change the type of A back to SEQAIJ and use MatDestroy_SeqAIJ() 
   * to destroy everything that remains. */
  ierr = PetscObjectChangeTypeName( (PetscObject)A, MATSEQAIJ);CHKERRQ(ierr);
  /* Note that I don't call MatSetType().  I believe this is because that 
   * is only to be called when *building* a matrix.  I could be wrong, but 
   * that is how things work for the SuperLU matrix class. */
  ierr = MatDestroy_SeqAIJ(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatDuplicate_SeqCSRPERM(Mat A, MatDuplicateOption op, Mat *M) 
{
  PetscErrorCode ierr;
  Mat_SeqCSRPERM *csrperm = (Mat_SeqCSRPERM *) A->spptr;
  Mat_SeqCSRPERM *csrperm_dest = (Mat_SeqCSRPERM *) (*M)->spptr;

  PetscFunctionBegin;
  ierr = MatDuplicate_SeqAIJ(A,op,M);CHKERRQ(ierr);
  ierr = PetscMemcpy((*M)->spptr,csrperm,sizeof(Mat_SeqCSRPERM));CHKERRQ(ierr);
  /* Allocate space for, and copy the grouping and permutation info. 
   * I note that when the groups are initially determined in 
   * SeqCSRPERM_create_perm, xgroup and nzgroup may be sized larger than 
   * necessary.  But at this point, we know how large they need to be, and 
   * allocate only the necessary amount of memory.  So the duplicated matrix 
   * may actually use slightly less storage than the original! */
  ierr = PetscMalloc(A->rmap->n*sizeof(PetscInt), csrperm_dest->iperm);CHKERRQ(ierr);
  ierr = PetscMalloc((csrperm->ngroup+1)*sizeof(PetscInt), csrperm_dest->xgroup);CHKERRQ(ierr);
  ierr = PetscMalloc((csrperm->ngroup)*sizeof(PetscInt), csrperm_dest->nzgroup);CHKERRQ(ierr);
  ierr = PetscMemcpy(csrperm_dest->iperm,csrperm->iperm,sizeof(PetscInt)*A->rmap->n);CHKERRQ(ierr);
  ierr = PetscMemcpy(csrperm_dest->xgroup,csrperm->xgroup,sizeof(PetscInt)*(csrperm->ngroup+1));CHKERRQ(ierr);
  ierr = PetscMemcpy(csrperm_dest->nzgroup,csrperm->nzgroup,sizeof(PetscInt)*csrperm->ngroup);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "SeqCSRPERM_create_perm"
PetscErrorCode SeqCSRPERM_create_perm(Mat A)
{
  PetscInt        m;  /* Number of rows in the matrix. */
  Mat_SeqAIJ      *a = (Mat_SeqAIJ *)(A)->data;
  PetscInt        *ia;  /* From the CSR representation; points to the beginning  of each row. */
  PetscInt        maxnz; /* Maximum number of nonzeros in any row. */
  PetscInt        *rows_in_bucket;
    /* To construct the permutation, we sort each row into one of maxnz 
     * buckets based on how many nonzeros are in the row. */
  PetscInt        nz;
  PetscInt        *nz_in_row;  /* the number of nonzero elements in row k. */
  PetscInt        *ipnz;
    /* When constructing the iperm permutation vector, 
     * ipnz[nz] is used to point to the next place in the permutation vector 
     * that a row with nz nonzero elements should be placed.*/
  Mat_SeqCSRPERM *csrperm = (Mat_SeqCSRPERM*) A->spptr;
    /* Points to the MATSEQCSRPERM-specific data in the matrix B. */
  PetscErrorCode ierr;
  PetscInt       i, ngroup, istart, ipos;

  /* I really ought to put something in here to check if B is of 
   * type MATSEQCSRPERM and return an error code if it is not.
   * Come back and do this! */
  PetscFunctionBegin;
  m  = A->rmap->n;
  ia = a->i;
   
  /* Allocate the arrays that will hold the permutation vector. */
  ierr = PetscMalloc(m*sizeof(PetscInt), &csrperm->iperm);CHKERRQ(ierr);

  /* Allocate some temporary work arrays that will be used in 
   * calculating the permuation vector and groupings. */
  ierr = PetscMalloc((m+1)*sizeof(PetscInt), &rows_in_bucket);CHKERRQ(ierr);
  ierr = PetscMalloc((m+1)*sizeof(PetscInt), &ipnz);CHKERRQ(ierr);
  ierr = PetscMalloc(m*sizeof(PetscInt), &nz_in_row);CHKERRQ(ierr); 

  /* Now actually figure out the permutation and grouping. */

  /* First pass: Determine number of nonzeros in each row, maximum 
   * number of nonzeros in any row, and how many rows fall into each  
   * "bucket" of rows with same number of nonzeros. */
  maxnz = 0;
  for (i=0; i<m; i++) {
    nz_in_row[i] = ia[i+1]-ia[i];
    if (nz_in_row[i] > maxnz) maxnz = nz_in_row[i];
  }

  for (i=0; i<=maxnz; i++) {
    rows_in_bucket[i] = 0;
  }
  for (i=0; i<m; i++) {
    nz = nz_in_row[i];
    rows_in_bucket[nz]++;
  }

  /* Allocate space for the grouping info.  There will be at most (maxnz + 1) 
   * groups.  (It is maxnz + 1 instead of simply maxnz because there may be 
   * rows with no nonzero elements.)  If there are (maxnz + 1) groups, 
   * then xgroup[] must consist of (maxnz + 2) elements, since the last 
   * element of xgroup will tell us where the (maxnz + 1)th group ends.
   * We allocate space for the maximum number of groups; 
   * that is potentially a little wasteful, but not too much so.  
   * Perhaps I should fix it later. */
  ierr = PetscMalloc((maxnz+2)*sizeof(PetscInt), &csrperm->xgroup);CHKERRQ(ierr);
  ierr = PetscMalloc((maxnz+1)*sizeof(PetscInt), &csrperm->nzgroup);CHKERRQ(ierr);

  /* Second pass.  Look at what is in the buckets and create the groupings.
   * Note that it is OK to have a group of rows with no non-zero values. */
  ngroup = 0;
  istart = 0;
  for (i=0; i<=maxnz; i++) {
    if (rows_in_bucket[i] > 0) {
      csrperm->nzgroup[ngroup] = i;
      csrperm->xgroup[ngroup] = istart;
      ngroup++;
      istart += rows_in_bucket[i];
    }
  }

  csrperm->xgroup[ngroup] = istart;
  csrperm->ngroup = ngroup;

  /* Now fill in the permutation vector iperm. */
  ipnz[0] = 0;
  for (i=0; i<maxnz; i++) {
    ipnz[i+1] = ipnz[i] + rows_in_bucket[i];
  }

  for (i=0; i<m; i++) {
    nz = nz_in_row[i];
    ipos = ipnz[nz];
    csrperm->iperm[ipos] = i;
    ipnz[nz]++;
  }

  /* Clean up temporary work arrays. */
  ierr = PetscFree(rows_in_bucket);CHKERRQ(ierr);
  ierr = PetscFree(ipnz);CHKERRQ(ierr);
  ierr = PetscFree(nz_in_row);CHKERRQ(ierr);

  /* Since we've allocated some memory to hold permutation info, 
   * flip the CleanUpCSRPERM flag to true. */
  csrperm->CleanUpCSRPERM = PETSC_TRUE;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatAssemblyEnd_SeqCSRPERM"
PetscErrorCode MatAssemblyEnd_SeqCSRPERM(Mat A, MatAssemblyType mode)
{
  PetscErrorCode ierr;
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;

  PetscFunctionBegin;
  if (mode == MAT_FLUSH_ASSEMBLY) PetscFunctionReturn(0);
  
  /* Since a MATSEQCSRPERM matrix is really just a MATSEQAIJ with some 
   * extra information, call the AssemblyEnd routine for a MATSEQAIJ. 
   * I'm not sure if this is the best way to do this, but it avoids 
   * a lot of code duplication.
   * I also note that currently MATSEQCSRPERM doesn't know anything about 
   * the Mat_CompressedRow data structure that SeqAIJ now uses when there 
   * are many zero rows.  If the SeqAIJ assembly end routine decides to use 
   * this, this may break things.  (Don't know... haven't looked at it.) */
  a->inode.use = PETSC_FALSE;
  ierr = MatAssemblyEnd_SeqAIJ(A, mode);

  /* Now calculate the permutation and grouping information. */
  ierr = SeqCSRPERM_create_perm(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqCSRPERM"
PetscErrorCode MatMult_SeqCSRPERM(Mat A,Vec xx,Vec yy)
{
  Mat_SeqAIJ      *a = (Mat_SeqAIJ*)A->data;
  PetscScalar     *x,*y;
  const MatScalar *aa;
  PetscErrorCode  ierr;
  PetscInt        *aj,*ai;
#if !(defined(PETSC_USE_FORTRAN_KERNEL_MULTCSRPERM) && defined(notworking))
  PetscInt        i,j;
#endif

  /* Variables that don't appear in MatMult_SeqAIJ. */
  Mat_SeqCSRPERM *csrperm = (Mat_SeqCSRPERM *) A->spptr;
  PetscInt       *iperm;  /* Points to the permutation vector. */
  PetscInt       *xgroup;
    /* Denotes where groups of rows with same number of nonzeros 
     * begin and end in iperm. */
  PetscInt       *nzgroup;
  PetscInt       ngroup;
  PetscInt       igroup;
  PetscInt       jstart,jend;
    /* jstart is used in loops to denote the position in iperm where a 
     * group starts; jend denotes the position where it ends.
     * (jend + 1 is where the next group starts.) */
  PetscInt       iold,nz;
  PetscInt       istart,iend,isize;
  PetscInt       ipos;
  PetscScalar    yp[NDIM];
  PetscInt       ip[NDIM]; /* yp[] and ip[] are treated as vector "registers" for performing the mat-vec. */

#if defined(PETSC_HAVE_PRAGMA_DISJOINT)
#pragma disjoint(*x,*y,*aa)
#endif

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  aj  = a->j;    /* aj[k] gives column index for element aa[k]. */
  aa  = a->a;  /* Nonzero elements stored row-by-row. */
  ai  = a->i;   /* ai[k] is the position in aa and aj where row k starts. */

  /* Get the info we need about the permutations and groupings. */
  iperm  = csrperm->iperm;
  ngroup = csrperm->ngroup;
  xgroup = csrperm->xgroup;
  nzgroup = csrperm->nzgroup;
  
#if defined(PETSC_USE_FORTRAN_KERNEL_MULTCSRPERM) && defined(notworking)
  fortranmultcsrperm_(&m,x,ii,aj,aa,y);
#else

  for (igroup=0; igroup<ngroup; igroup++) {
    jstart = xgroup[igroup];
    jend   = xgroup[igroup+1] - 1;
    nz     = nzgroup[igroup];

    /* Handle the special cases where the number of nonzeros per row 
     * in the group is either 0 or 1. */
    if (nz == 0) {
      for (i=jstart; i<=jend; i++) {
        y[iperm[i]] = 0.0;
      }
    } else if (nz == 1) {
      for (i=jstart; i<=jend; i++) {
        iold = iperm[i];
        ipos = ai[iold];
        y[iold] = aa[ipos] * x[aj[ipos]];
      }
    } else {
    
      /* We work our way through the current group in chunks of NDIM rows 
       * at a time. */

      for (istart=jstart; istart<=jend; istart+=NDIM) {
        /* Figure out where the chunk of 'isize' rows ends in iperm.
         * 'isize may of course be less than NDIM for the last chunk. */
        iend = istart + (NDIM - 1);
        if (iend > jend) { iend = jend; }
        isize = iend - istart + 1;

        /* Initialize the yp[] array that will be used to hold part of 
         * the permuted results vector, and figure out where in aa each 
         * row of the chunk will begin. */
        for (i=0; i<isize; i++) {
          iold = iperm[istart + i];
            /* iold is a row number from the matrix A *before* reordering. */
          ip[i] = ai[iold];
            /* ip[i] tells us where the ith row of the chunk begins in aa. */
          yp[i] = (PetscScalar) 0.0;
        }

        /* If the number of zeros per row exceeds the number of rows in 
         * the chunk, we should vectorize along nz, that is, perform the 
         * mat-vec one row at a time as in the usual CSR case. */
        if (nz > isize) {
#if defined(PETSC_HAVE_CRAYC)
#pragma _CRI preferstream
#endif
          for (i=0; i<isize; i++) {
#if defined(PETSC_HAVE_CRAYC)
#pragma _CRI prefervector
#endif
            for (j=0; j<nz; j++) {
              ipos = ip[i] + j;
              yp[i] += aa[ipos] * x[aj[ipos]];
            }
          }
        } else {
        /* Otherwise, there are enough rows in the chunk to make it 
         * worthwhile to vectorize across the rows, that is, to do the 
         * matvec by operating with "columns" of the chunk. */
          for (j=0; j<nz; j++) {
            for(i=0; i<isize; i++) {
              ipos = ip[i] + j;
              yp[i] += aa[ipos] * x[aj[ipos]];
            }
          }
        }

#if defined(PETSC_HAVE_CRAYC)
#pragma _CRI ivdep
#endif
        /* Put results from yp[] into non-permuted result vector y. */
        for (i=0; i<isize; i++) {
          y[iperm[istart+i]] = yp[i];
        }
      } /* End processing chunk of isize rows of a group. */
    } /* End handling matvec for chunk with nz > 1. */
  } /* End loop over igroup. */
#endif
  ierr = PetscLogFlops(2.0*a->nz - A->rmap->n);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/* MatMultAdd_SeqCSRPERM() calculates yy = ww + A * xx.
 * Note that the names I used to designate the vectors differs from that 
 * used in MatMultAdd_SeqAIJ().  I did this to keep my notation consistent 
 * with the MatMult_SeqCSRPERM() routine, which is very similar to this one. */
/*
    I hate having virtually identical code for the mult and the multadd!!!
*/
#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqCSRPERM"
PetscErrorCode MatMultAdd_SeqCSRPERM(Mat A,Vec xx,Vec ww,Vec yy)
{
  Mat_SeqAIJ      *a = (Mat_SeqAIJ*)A->data;
  PetscScalar     *x,*y,*w;
  const MatScalar *aa;
  PetscErrorCode  ierr;
  PetscInt        *aj,*ai;
#if !defined(PETSC_USE_FORTRAN_KERNEL_MULTADDCSRPERM)
  PetscInt        i,j;
#endif

  /* Variables that don't appear in MatMultAdd_SeqAIJ. */
  Mat_SeqCSRPERM *csrperm;
  PetscInt       *iperm;  /* Points to the permutation vector. */
  PetscInt       *xgroup;
    /* Denotes where groups of rows with same number of nonzeros 
     * begin and end in iperm. */
  PetscInt       *nzgroup;
  PetscInt       ngroup;
  PetscInt       igroup;
  PetscInt       jstart,jend;
    /* jstart is used in loops to denote the position in iperm where a 
     * group starts; jend denotes the position where it ends.
     * (jend + 1 is where the next group starts.) */
  PetscInt       iold,nz;
  PetscInt       istart,iend,isize;
  PetscInt       ipos;
  PetscScalar    yp[NDIM];
  PetscInt       ip[NDIM];
    /* yp[] and ip[] are treated as vector "registers" for performing 
     * the mat-vec. */

#if defined(PETSC_HAVE_PRAGMA_DISJOINT)
#pragma disjoint(*x,*y,*aa)
#endif

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  if (yy != ww) {
    ierr = VecGetArray(ww,&w);CHKERRQ(ierr);
  } else {
    w = y;
  }

  aj  = a->j;  /* aj[k] gives column index for element aa[k]. */
  aa  = a->a;  /* Nonzero elements stored row-by-row. */
  ai  = a->i;  /* ai[k] is the position in aa and aj where row k starts. */

  /* Get the info we need about the permutations and groupings. */
  csrperm = (Mat_SeqCSRPERM *) A->spptr;
  iperm = csrperm->iperm;
  ngroup = csrperm->ngroup;
  xgroup = csrperm->xgroup;
  nzgroup = csrperm->nzgroup;
  
#if defined(PETSC_USE_FORTRAN_KERNEL_MULTADDCSRPERM)
  fortranmultaddcsrperm_(&m,x,ii,aj,aa,y,w);
#else

  for(igroup=0; igroup<ngroup; igroup++) {
    jstart = xgroup[igroup];
    jend = xgroup[igroup+1] - 1;

    nz = nzgroup[igroup];

    /* Handle the special cases where the number of nonzeros per row 
     * in the group is either 0 or 1. */
    if(nz == 0) {
      for(i=jstart; i<=jend; i++) {
        iold = iperm[i];
        y[iold] = w[iold];
      }
    }
    else if(nz == 1) {
      for(i=jstart; i<=jend; i++) {
        iold = iperm[i];
        ipos = ai[iold];
        y[iold] = w[iold] + aa[ipos] * x[aj[ipos]];
      }
    }
    /* For the general case: */
    else {
    
      /* We work our way through the current group in chunks of NDIM rows 
       * at a time. */

      for(istart=jstart; istart<=jend; istart+=NDIM) {
        /* Figure out where the chunk of 'isize' rows ends in iperm.
         * 'isize may of course be less than NDIM for the last chunk. */
        iend = istart + (NDIM - 1);
        if(iend > jend) { iend = jend; }
        isize = iend - istart + 1;

        /* Initialize the yp[] array that will be used to hold part of 
         * the permuted results vector, and figure out where in aa each 
         * row of the chunk will begin. */
        for(i=0; i<isize; i++) {
          iold = iperm[istart + i];
            /* iold is a row number from the matrix A *before* reordering. */
          ip[i] = ai[iold];
            /* ip[i] tells us where the ith row of the chunk begins in aa. */
          yp[i] = w[iold];
        }

        /* If the number of zeros per row exceeds the number of rows in 
         * the chunk, we should vectorize along nz, that is, perform the 
         * mat-vec one row at a time as in the usual CSR case. */
        if(nz > isize) {
#if defined(PETSC_HAVE_CRAYC)
#pragma _CRI preferstream
#endif
          for(i=0; i<isize; i++) {
#if defined(PETSC_HAVE_CRAYC)
#pragma _CRI prefervector
#endif
            for(j=0; j<nz; j++) {
              ipos = ip[i] + j;
              yp[i] += aa[ipos] * x[aj[ipos]];
            }
          }
        }
        /* Otherwise, there are enough rows in the chunk to make it 
         * worthwhile to vectorize across the rows, that is, to do the 
         * matvec by operating with "columns" of the chunk. */
        else {
          for(j=0; j<nz; j++) {
            for(i=0; i<isize; i++) {
              ipos = ip[i] + j;
              yp[i] += aa[ipos] * x[aj[ipos]];
            }
          }
        }

#if defined(PETSC_HAVE_CRAYC)
#pragma _CRI ivdep
#endif
        /* Put results from yp[] into non-permuted result vector y. */
        for(i=0; i<isize; i++) {
          y[iperm[istart+i]] = yp[i];
        }
      } /* End processing chunk of isize rows of a group. */
      
    } /* End handling matvec for chunk with nz > 1. */
  } /* End loop over igroup. */

#endif
  ierr = PetscLogFlops(2.0*a->nz);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  if (yy != ww) {
    ierr = VecRestoreArray(ww,&w);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


/* MatConvert_SeqAIJ_SeqCSRPERM converts a SeqAIJ matrix into a 
 * SeqCSRPERM matrix.  This routine is called by the MatCreate_SeqCSRPERM() 
 * routine, but can also be used to convert an assembled SeqAIJ matrix 
 * into a SeqCSRPERM one. */
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatConvert_SeqAIJ_SeqCSRPERM"
PetscErrorCode PETSCMAT_DLLEXPORT MatConvert_SeqAIJ_SeqCSRPERM(Mat A,const MatType type,MatReuse reuse,Mat *newmat)
{
  PetscErrorCode ierr;
  Mat            B = *newmat;
  Mat_SeqCSRPERM *csrperm;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatDuplicate(A,MAT_COPY_VALUES,&B);CHKERRQ(ierr);
  }

  ierr = PetscNewLog(B,Mat_SeqCSRPERM,&csrperm);CHKERRQ(ierr);
  B->spptr = (void *) csrperm;

  /* Set function pointers for methods that we inherit from AIJ but override. */
  B->ops->duplicate   = MatDuplicate_SeqCSRPERM;
  B->ops->assemblyend = MatAssemblyEnd_SeqCSRPERM;
  B->ops->destroy     = MatDestroy_SeqCSRPERM;
  B->ops->mult        = MatMult_SeqCSRPERM;
  B->ops->multadd     = MatMultAdd_SeqCSRPERM;

  /* If A has already been assembled, compute the permutation. */
  if (A->assembled) {
    ierr = SeqCSRPERM_create_perm(B);CHKERRQ(ierr);
  }
 
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatConvert_seqcsrperm_seqaij_C",
                                           "MatConvert_SeqCSRPERM_SeqAIJ",MatConvert_SeqCSRPERM_SeqAIJ);CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)B,MATSEQCSRPERM);CHKERRQ(ierr);
  *newmat = B;
  PetscFunctionReturn(0);
}
EXTERN_C_END


#undef __FUNCT__
#define __FUNCT__ "MatCreateSeqCSRPERM"
/*@C
   MatCreateSeqCSRPERM - Creates a sparse matrix of type SEQCSRPERM.
   This type inherits from AIJ, but calculates some additional permutation 
   information that is used to allow better vectorization of some 
   operations.  At the cost of increased storage, the AIJ formatted 
   matrix can be copied to a format in which pieces of the matrix are 
   stored in ELLPACK format, allowing the vectorized matrix multiply 
   routine to use stride-1 memory accesses.  As with the AIJ type, it is 
   important to preallocate matrix storage in order to get good assembly 
   performance.
   
   Collective on MPI_Comm

   Input Parameters:
+  comm - MPI communicator, set to PETSC_COMM_SELF
.  m - number of rows
.  n - number of columns
.  nz - number of nonzeros per row (same for all rows)
-  nnz - array containing the number of nonzeros in the various rows 
         (possibly different for each row) or PETSC_NULL

   Output Parameter:
.  A - the matrix 

   Notes:
   If nnz is given then nz is ignored

   Level: intermediate

.keywords: matrix, cray, sparse, parallel

.seealso: MatCreate(), MatCreateMPICSRPERM(), MatSetValues()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatCreateSeqCSRPERM(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt nz,const PetscInt nnz[],Mat *A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,m,n);CHKERRQ(ierr);
  ierr = MatSetType(*A,MATSEQCSRPERM);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation_SeqAIJ(*A,nz,(PetscInt*)nnz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatCreate_SeqCSRPERM"
PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_SeqCSRPERM(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSetType(A,MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatConvert_SeqAIJ_SeqCSRPERM(A,MATSEQCSRPERM,MAT_REUSE_MATRIX,&A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

