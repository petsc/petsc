#include "petscsles.h"
#include "samgfunc.h"

/* ------------------------------------------------------------------- */
#undef __FUNC__
#define __FUNC__ "SamgGetCoarseMat"
/*..SamgGetCoarseMat - This routine gets the coarse level matrix on the 
    level specified at input. 
    WARNING: This routine does not work to get the fine level matrix, 
             i.e. the value of k at input should be at least 2 
    INPUT:  level:    current grid level 
            ia_shift: shift to apply on ia_cg elements 
            ja_shift: shift to apply on ja_cg elements 
    OUTPUT: coarsemat: coarse level matrix  
..*/

int SamgGetCoarseMat(int level, int ia_shift, int ja_shift, 
                     Mat* coarsemat, void* ctx)
{
   int      nnu_cg, nna_cg; /* size and non-zeros of coarse grid matrix */ 
   int      *ia_cg, *ja_cg; /* coarse grid matrix in skyline format     */
   double   *a_cg; 
   int      *nnz_per_row; /* integer vector to hold the number of nonzeros */
                          /* of each row. This vector will be used to      */
                          /* allocate memory for the matrix, and to store  */
                          /* elements in the matrix                        */
   int      I, ierr; 

   /*..Get size (nnu_cg) and number of non-zeros (nna_cg) of coarse 
     grid matrix..*/
   samggetdimmat_(&level, &nnu_cg, &nna_cg);

   /*..Now that nnu_cg and nna_cg are known, we can allocate memory for 
     coarse level matrix in compresses skyline format..*/ 
   a_cg =  (double *) PetscMalloc(nna_cg  * sizeof(double)); 
           CHKPTRQ(a_cg);
   ia_cg = (int*)     PetscMalloc((nnu_cg+1)* sizeof(int));
           CHKPTRQ(ia_cg); 
   ja_cg = (int*)     PetscMalloc(nna_cg * sizeof(int));
           CHKPTRQ(ja_cg);

   /*..Get coarse grid matrix in skyline format..*/ 
   samggetmat_(&level, a_cg, ia_cg, ja_cg); 

   /*..Apply shift on each of the ia_cg and ja_cg elements..*/
   apply_shift(ia_cg, nnu_cg, ia_shift, ja_cg, nna_cg, ja_shift);    

   nnz_per_row = (int *) PetscMalloc(nnu_cg * sizeof(int) ); 
                 CHKPTRQ(nnz_per_row);

   /*..The numbero f nonzeros entries in row I can be calculated as      
       ia[I+1] - 1 - ia[I] + 1 = ia[I+1] - ia[I]                         ..*/
   for (I=0;I<nnu_cg;I++)
       nnz_per_row[I] = ia_cg[I+1] - ia_cg[I]; 

   /*..Allocate (create) matrix  for use within PETSc..*/
   ierr =  MatCreateSeqAIJ(PETSC_COMM_WORLD,nnu_cg,nnu_cg,PETSC_NULL,
           nnz_per_row,coarsemat); CHKERRA(ierr);

   /*..Store coarse grid matrix in Petsc Mat object..*/
   for (I=0;I<nnu_cg;I++){
      ierr = MatSetValues(*coarsemat, 
               1,              /* number of rows */
               &I,             /* pointer to global row number */
               nnz_per_row[I], /* number of collums = number of nonzero ... */
                               /* entries in row I                          */
               &(ja_cg[ ia_cg[I] ]), 
                              /* vector global column indices */
               (Scalar *) &(a_cg[ ia_cg[I] ]),
                              /* vector of coefficients */
	INSERT_VALUES); CHKERRA(ierr);
   }   

   ierr = MatAssemblyBegin(*coarsemat,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);
   ierr = MatAssemblyEnd(*coarsemat,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);

   /*..Free memory required by storage in skyline format..*/ 
   PetscFree(a_cg); 
   PetscFree(ia_cg); 
   PetscFree(ja_cg); 
   /*..Free memory for auxilary array..*/ 
   PetscFree(nnz_per_row); 
 
   return 0;
}
/* ------------------------------------------------------------------- */
#undef __FUNC__
#define __FUNC__ "SamgGetInterpolation"
/*..SamgGetInterpolation - Get interpolation operator that interpolates 
    from level k+1 (coarse grid) to k (fine grid), where the input level 
    equals k ..*/ 
/*..WARNING: This routine assumes that the input value for level is strictly 
    smaller than the number of levels created..*/  
/*..Implementation notes
    o) The interplotion is a rectangular matrix with 
       number of rows equal to fine grid dim 
                 cols          coarse. 
..*/  
int SamgGetInterpolation(int level, int iw_shift, int jw_shift,
                         Mat* interpolation, void* ctx) 
{
   int     rows_weights, cols_weights, nna_weights; 
   int     *iweights, *jweights; 
   double  *weights; 
   int      *nnz_per_row; /* integer vector to hold the number of nonzeros */
                          /* of each row. This vector will be used to      */
                          /* allocate memory for the matrix, and to store  */
                          /* elements in the matrix                        */
   int      I, ierr, coarser_level=level+1, dummy; 

   /*..Get number of rows and number of nonzeros of interpolation operator..*/
   samggetdimint_(&level, &rows_weights, &nna_weights);

   /*..Get number of cols of interpolation operator. NOTE: The number of 
       collums of the interpolation on level k equals the size of 
       the coarse grid matrix on the next coarsest grid.  samggetdimint_ 
       does not allow to get the number of collumns of next to coarsest 
       grid..*/
   samggetdimmat_(&coarser_level, &cols_weights, &dummy);   

   /*..Now that nnu_weights and nna_weights are known, we can allocate 
       memory for interpolation operator in compresses skyline format..*/ 
   weights  = (double *) PetscMalloc(nna_weights  * sizeof(double)); 
               CHKPTRQ(weights);
   iweights = (int*)     PetscMalloc((rows_weights+1)* sizeof(int));
              CHKPTRQ(iweights); 
   jweights = (int*)     PetscMalloc(nna_weights * sizeof(int));
              CHKPTRQ(jweights); 

   /*..Get interpolation operator in compressed skyline format..*/
   samggetint_(&level, weights, iweights, jweights); 

   /*..Apply shift on each of the ia_cg and ja_cg elements..*/
   apply_shift(iweights, rows_weights, iw_shift, jweights, nna_weights,
               jw_shift);

   nnz_per_row = (int *) PetscMalloc(rows_weights * sizeof(int) ); 
                 CHKPTRQ(nnz_per_row);

   /*..The numbero f nonzeros entries in row I can be calculated as      
       ia[I+1] - 1 - ia[I] + 1 = ia[I+1] - ia[I]                         ..*/
   for (I=0;I<rows_weights;I++)
       nnz_per_row[I] = iweights[I+1] - iweights[I]; 

   /*..Allocate (create) matrix  for use within PETSc..*/
   ierr =  MatCreateSeqAIJ(PETSC_COMM_WORLD,rows_weights,cols_weights, 
           PETSC_NULL, nnz_per_row,interpolation); CHKERRA(ierr);

   /*..Store coarse grid matrix in Petsc Mat object..*/
   for (I=0;I<rows_weights;I++){
      ierr = MatSetValues(*interpolation, 
               1,              /* number of rows */
               &I,             /* pointer to global row number */
               nnz_per_row[I], /* number of collums = number of nonzero ... */
                               /* entries in row I                          */
               &(jweights[ iweights[I] ]), 
                              /* vector global column indices */
               (Scalar *) &(weights[ iweights[I] ]),
                              /* vector of coefficients */
	INSERT_VALUES); CHKERRA(ierr);
   }   

   ierr = MatAssemblyBegin(*interpolation,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);
   ierr = MatAssemblyEnd(*interpolation,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);

   /*..Free memory required by storage in skyline format..*/ 
   PetscFree(weights); 
   PetscFree(iweights); 
   PetscFree(jweights); 
   /*..Free memory for auxilary array..*/ 
   PetscFree(nnz_per_row); 

   return 0;
} 




