#define PETSCKSP_DLL

#include "global.h"
#include "petscksp.h"
#include "samgfunc.h"
#include "petscfunc.h"
#include "externc.h"

EXTERN_C_BEGIN
void USER_coo(int * i,int * ndim, double * x, double * y, double * z)
{
  printf("in user_coo");
}
EXTERN_C_END

static  double Machine_Precision_Eps = 2.e-16;
/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "SamgGetGrid"
/*..SamgGetGrid - This routine gets an array of grids
    INPUT:  levels: number of levels created by SAMG 
            numnodes: number of nodes on finest grid 
            numnonzeros: number of nonzeros on coarsest grid 
    OUTPUT: grid  : array of grids                   ..*/   
PetscErrorCode PETSCKSP_DLLEXPORT SamgGetGrid(int levels, int numnodes, int numnonzero, 
                GridCtx* grid, void* ctx)
{
   int      k; 
   int      ia_shift[MAX_LEVELS], ja_shift[MAX_LEVELS], nnu_cg, nna_cg;
   int      iw_shift[MAX_LEVELS], jw_shift[MAX_LEVELS], rows_weights, 
            nna_weights, dummy;
   PetscErrorCode ierr; 
   MatInfo  info;

   /*..Get coarse grid operators..*/ 
   /*....Initialize ia_shift, ja_shift, nnu_cg and nna_cg....*/ 
   ia_shift[1] = 1; 
   ja_shift[1] = 1; 
   nnu_cg = numnodes; 
   nna_cg = numnonzero; 

   for (k=2;k<=levels;k++){ /*....We do not get the finest level matrix....*/ 
       /*....Update ia_shift and ja_shift values with nna_cg and nnu_cg 
             from previous loop....*/ 
       ia_shift[k] = ia_shift[k-1] + nna_cg ; 
       ja_shift[k] = ja_shift[k-1] + nnu_cg ; 

       /*....Get coarse grid matrix on level k....*/ 
       ierr = SamgGetCoarseMat(k, ia_shift[k], ja_shift[k], &(grid[k].A), 
                               PETSC_NULL); 

       /*....Get size and number of nonzeros of coarse grid matrix on
             level k, i.e. get new nna_cg and nnu_cg values....*/ 
       ierr = MatGetSize(grid[k].A, &nnu_cg, &nnu_cg);CHKERRQ(ierr);
       ierr = MatGetInfo(grid[k].A, MAT_LOCAL, &info);CHKERRQ(ierr); 
       nna_cg = int(info.nz_used);
   }  
 
   /*..Get interpolation operators..*/ 
   /*....Initialize iw_shift, jw_shift and nna_weights....*/ 
   iw_shift[0] = 1; 
   jw_shift[0] = 1; 
   nna_weights = 0;
   rows_weights = numnodes;

   for (k=1;k<=levels-1;k++){/*....There's NO interpolation operator 
                                   associated to the coarsest level....*/ 
       /*....Update iw_shift with nna_weights value from 
             previous loop....*/ 
       iw_shift[k] = iw_shift[k-1] + nna_weights ; 
       /*....Update jw_shift with rows_weights value from 
             current loop....*/ 
       jw_shift[k] = jw_shift[k-1] + rows_weights ; 
         
       /*....Get interpolation from level k+1 to level k....*/
       ierr = SamgGetInterpolation(k, iw_shift[k], jw_shift[k],
                                   &(grid[k].Interp), PETSC_NULL) ; 

       /*....Get number of collumns and number of nonzeros of 
             interpolation associated to level k. NOTE: The 
             number of collums at this loop equals the number of 
             rows at the next loop...*/
       ierr = MatGetSize(grid[k].Interp, &dummy, &rows_weights);CHKERRQ(ierr);
       ierr = MatGetInfo(grid[k].Interp, MAT_LOCAL, &info);CHKERRQ(ierr); 
       nna_weights = int(info.nz_used);
   }

   return 0;
}
/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "SamgGetCoarseMat"
/*..SamgGetCoarseMat - This routine gets the coarse level matrix on the 
    level specified at input. 
    WARNING: This routine does not work to get the fine level matrix, 
             i.e. the value of k at input should be at least 2 
    INPUT:  level:    current grid level 
            ia_shift: shift to apply on ia_cg elements 
            ja_shift: shift to apply on ja_cg elements 
    OUTPUT: coarsemat: coarse level matrix  
..*/

PetscErrorCode PETSCKSP_DLLEXPORT SamgGetCoarseMat(int level, int ia_shift, int ja_shift, 
                     Mat* coarsemat, void* ctx)
{
   int      nnu_k, nna_k; /* size and non-zeros of operator on level k     */ 
   int      *ia_k, *ja_k; /* coarse grid matrix in skyline format          */
   double   *a_k; 
   int      *nnz_per_row; /* integer vector to hold the number of nonzeros */
                          /* of each row. This vector will be used to      */
                          /* allocate memory for the matrix, and to store  */
                          /* elements in the matrix                        */
  PetscErrorCode ierr;
   int      I; 

   /*..Get size (nnu_k) and number of non-zeros (nna_k) of operator 
     on level k..*/
   SAMGPETSC_get_dim_operator(&level, &nnu_k, &nna_k);

   /*..Now that nnu_cg and nna_cg are known, we can allocate memory for 
     coarse level matrix in compresses skyline format..*/ 
   ierr = PetscMalloc(nna_k     * sizeof(double),&a_k);CHKERRQ(ierr);    
   ierr = PetscMalloc((nnu_k+1) * sizeof(int),&ia_k);CHKERRQ(ierr);    
   ierr = PetscMalloc(nna_k     * sizeof(int),&ja_k);CHKERRQ(ierr);    

   /*..Get coarse grid matrix in skyline format..*/ 
   SAMGPETSC_get_operator(&level, a_k, ia_k, ja_k); 

   /*..Apply shift on each of the ia_cg and ja_cg elements..*/
   SAMGPETSC_apply_shift(ia_k, &nnu_k, &ia_shift, 
                         ja_k, &nna_k, &ja_shift);    

   ierr = PetscMalloc(nnu_k * sizeof(int),&nnz_per_row);CHKERRQ(ierr);    

   /*..The numbero f nonzeros entries in row I can be calculated as      
       ia[I+1] - 1 - ia[I] + 1 = ia[I+1] - ia[I]                         ..*/
   for (I=0;I<nnu_k;I++)
       nnz_per_row[I] = ia_k[I+1] - ia_k[I]; 

   /*..Allocate (create) SeqAIJ matrix  for use within PETSc..*/
   ierr = MatCreate(PETSC_COMM_WORLD,nnu_k,nnu_k,nnu_k,nnu_k,coarsemat);CHKERRQ(ierr);
   ierr = MatSetType(*coarsemat,MATSEQAIJ);CHKERRQ(ierr);
   ierr = MatSeqAIJSetPreallocation(*coarsemat,0,nnz_per_row);CHKERRQ(ierr);

   /*..Store coarse grid matrix in Petsc Mat object..*/
   for (I=0;I<nnu_k;I++){
      ierr = MatSetValues(*coarsemat, 
               1,              /* number of rows */
               &I,             /* pointer to global row number */
               nnz_per_row[I], /* number of collums = number of nonzero ... */
                               /* entries in row I                          */
               &(ja_k[ ia_k[I] ]), 
                              /* vector global column indices */
               (PetscScalar *) &(a_k[ ia_k[I] ]),
                              /* vector of coefficients */
	INSERT_VALUES);CHKERRQ(ierr);
   }   

   ierr = MatAssemblyBegin(*coarsemat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
   ierr = MatAssemblyEnd(*coarsemat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

   /*..Free memory required by storage in skyline format..*/ 
   PetscFree(a_k); 
   PetscFree(ia_k); 
   PetscFree(ja_k); 
   /*..Free memory for auxilary array..*/ 
   PetscFree(nnz_per_row); 
 
   return 0;
}
/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "SamgGetInterpolation"
/*..SamgGetInterpolation - Get interpolation operator that interpolates 
    from level k+1 (coarse grid) to k (fine grid), where the input level 
    equals k ..*/ 
/*..WARNING: This routine assumes that the input value for level is strictly 
    smaller than the number of levels created..*/  
/*..Implementation notes
    o) The interpolation is a rectangular matrix with 
       number of rows equal to fine grid dim 
                 cols          coarse. 
..*/  
PetscErrorCode PETSCKSP_DLLEXPORT SamgGetInterpolation(int level, int iw_shift, int jw_shift,
                         Mat* interpolation, void* ctx) 
{
   int     rows_weights, cols_weights, nna_weights; 
   int     *iweights, *jweights; 
   double  *weights; 
   int      *nnz_per_row; /* integer vector to hold the number of nonzeros */
                          /* of each row. This vector will be used to      */
                          /* allocate memory for the matrix, and to store  */
                          /* elements in the matrix                        */
  PetscErrorCode ierr;
   int      I,  coarser_level=level+1, dummy; 

   /*..Get number of rows and number of nonzeros of interpolation operator..*/
   SAMGPETSC_get_dim_interpol(&level, &rows_weights, &nna_weights);

   /*..Get number of cols of interpolation operator. NOTE: The number of 
       collums of the interpolation on level k equals the size of 
       the coarse grid matrix on the next coarsest grid.  
       SAMGPETSC_get_dim_interpol does not allow to get the number of 
       collumns of next to coarsest grid..*/
   SAMGPETSC_get_dim_operator(&coarser_level, &cols_weights, &dummy);   

   /*..Now that nnu_weights and nna_weights are known, we can allocate 
       memory for interpolation operator in compresses skyline format..*/ 
   ierr = PetscMalloc(nna_weights  * sizeof(double),&weights);
          CHKERRQ(ierr);    
   ierr = PetscMalloc((rows_weights+1) * sizeof(int),&iweights);
          CHKERRQ(ierr);    
   ierr = PetscMalloc(nna_weights  * sizeof(int),&jweights);
          CHKERRQ(ierr);    

   /*..Get interpolation operator in compressed skyline format..*/
   SAMGPETSC_get_interpol(&level, weights, iweights, jweights); 

   /*..Apply shift on each of the ia_cg and ja_cg elements..*/
   SAMGPETSC_apply_shift(iweights, &rows_weights, &iw_shift, 
                         jweights, &nna_weights, &jw_shift);

   ierr = PetscMalloc(rows_weights * sizeof(int),&nnz_per_row);
          CHKERRQ(ierr);    

   /*..The numbero f nonzeros entries in row I can be calculated as      
       ia[I+1] - 1 - ia[I] + 1 = ia[I+1] - ia[I]                         ..*/
   for (I=0;I<rows_weights;I++)
       nnz_per_row[I] = iweights[I+1] - iweights[I]; 

   /*..Allocate (create) SeqAIJ matrix  for use within PETSc..*/
   ierr = MatCreate(PETSC_COMM_WORLD,rows_weights,cols_weights,rows_weights,cols_weights,interpolation);CHKERRQ(ierr);
   ierr = MatSetType(*interpolation,MATSEQAIJ);CHKERRQ(ierr);
   ierr = MatSeqAIJSetPreallocation(*interpolation,0,nnz_per_row);CHKERRQ(ierr);

   /*..Store coarse grid matrix in Petsc Mat object..*/
   for (I=0;I<rows_weights;I++){
      ierr = MatSetValues(*interpolation, 
               1,              /* number of rows */
               &I,             /* pointer to global row number */
               nnz_per_row[I], /* number of collums = number of nonzero ... */
                               /* entries in row I                          */
               &(jweights[ iweights[I] ]), 
                              /* vector global column indices */
               (PetscScalar *) &(weights[ iweights[I] ]),
                              /* vector of coefficients */
	INSERT_VALUES);CHKERRQ(ierr);
   }   

   ierr = MatAssemblyBegin(*interpolation,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
   ierr = MatAssemblyEnd(*interpolation,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

   /*..Free memory required by storage in skyline format..*/ 
   PetscFree(weights); 
   PetscFree(iweights); 
   PetscFree(jweights); 
   /*..Free memory for auxilary array..*/ 
   PetscFree(nnz_per_row); 

   return 0;
} 
/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "SamgPetscWriteOperator"
/*..Write coarser grid operators constructed by SAMG to ASCII file..*/

PetscErrorCode PETSCKSP_DLLEXPORT SamgPetscWriteOperator(const int numnodes, const double* Asky, 
                           const int* ia, const int* ja, int extension)
{

     static char filename[80]; 
     int         I,j,j1,j2; 
     FILE        *output;
     
     /*..Switch arrays iacopy and jacopy to C conventions..*/ 
     //     for (j=0;j<=numnodes;j++)
     //         iacopy[j]--;
     //     for (j=0;j<ia[numnodes];j++)
     //         jacopy[j]--;
	   
     /*....Write matrix to file....*/
      sprintf(filename,"coarsemat.%02u", extension);
      output=fopen(filename,"w");
      fprintf(output, "%% \n"); 
      fprintf(output, "%% %d %d \n", numnodes, ia[numnodes] ); 

      for (I=0;I<numnodes;I++){
           j1 = ia[I]; 
           j2 = ia[I+1] - 1; 
           for (j=j1;j<=j2;j++){
               fprintf(output, "%d %d %22.18e\n", I+1, ja[j]+1, 
                                Asky[j] ); 
	       //               printf("%d %d %e \n", I+1, ja[j]+1, 
	       //                              Asky[j] );
           }
      }
      fclose(output); 
      
      return 0;
}
/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "SamgPetscWriteInterpol"
/*..Write interpolation operators constructed by SAMG to ASCII file..*/

PetscErrorCode PETSCKSP_DLLEXPORT SamgPetscWriteInterpol(const int numrows, const double* weights, 
                  const int* iweights, const int* jweights, int extension)
{

     static char filename[80]; 
     int         I,j,j1,j2,numcols,numnonzero; 
     FILE        *output;
  
     /*..Set number of nonzeros..*/ 
     numnonzero = iweights[numrows];

     /*..Determine numcols as the maximum ja value +1..*/ 
     numcols = jweights[0]; 
     for (j=0;j<numnonzero;j++){
       if (jweights[j] > numcols) numcols = jweights[j];
     }
     numcols++; 

     /*..Write interpolation operator from grid k+1 (coarse grid) grid to k 
         (finer grid) to file..*/
      sprintf(filename,"interpol.%02u%02u", 
                        extension+1, extension);
      output=fopen(filename,"w");
      fprintf(output, "%% \n%% %d %d %d \n", numrows, numcols, iweights[numrows] ); 
      for (I=0;I<numrows;I++){
           j1 = iweights[I]; 
           j2 = iweights[I+1] - 1; 
           for (j=j1;j<=j2;j++){
               fprintf(output, "%d %d %22.18e\n", I+1, jweights[j]+1, 
                                weights[j] ); 
	       //               printf("%d %d %e \n", I+1, jweights[j]+1, 
	       //                                weights[j] );
           }
      }
      fclose(output); 

return 0;
}
/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "SamgCheckGalerkin"
/*..SamgCheckGalerkin - This routine offers a check on the correctness 
    of how SAMG interpolation and coarse grid operators are parsed to 
    PETSc. This routine computes I^H_h A^h I^h_H by PETSc matrix - matrix 
    multiplications, and compares this product with A^H..*/ 
 
PetscErrorCode PETSCKSP_DLLEXPORT SamgCheckGalerkin(int levels, Mat A, GridCtx* grid, 
                      void* ctx)
{
   Mat     FineLevelMatrix, Restriction, HalfGalerkin, Galerkin, Diff; 
   double  normdiff; 
   PetscErrorCode ierr;
   int  k; 

   for (k=1;k<=levels-1;k++){ 
      if (k==1)
          FineLevelMatrix = A; 
          else
          FineLevelMatrix = grid[k].A; 
      /*....Compute A^h I^h_H....*/ 
      ierr = MatMatMult(FineLevelMatrix, grid[k].Interp, &HalfGalerkin); 
      /*....Get I^h_H....*/ 
      ierr = MatTranspose(grid[k].Interp,&Restriction);
      /*....Compute I^H_h A^h I^h_H....*/ 
      ierr = MatMatMult(Restriction, HalfGalerkin, &Galerkin);
      /*....Compute A^H - I^H_h A^h I^h_H....*/ 
      ierr = MatSubstract(grid[k+1].A, Galerkin, &Diff); 
     /*....Compute || A^H - I^H_h A^h I^h_H||_{\infty}....*/ 
      ierr = MatNorm(Diff,NORM_INFINITY,&normdiff);CHKERRQ(ierr);

      printf("SamgCheckGalerkin :: || A^H - I^H_h A^h I^h_H||_{infty} on level %8d = %e\n", 
	      k+1, normdiff); 

      ierr = MatDestroy(Restriction);CHKERRQ(ierr); 
      ierr = MatDestroy(HalfGalerkin);CHKERRQ(ierr); 
      ierr = MatDestroy(Galerkin);CHKERRQ(ierr); 
      ierr = MatDestroy(Diff);CHKERRQ(ierr); 
   }
   return 0;
} 
/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "MatSubstract"
/*..MatSubstract - Computes the difference Term1 - Term2 
    INPUT:  Term1, Term2 : The input matrices 
    OUTPUT: Prod:          the difference 
  NOTE: Memory needed by difference has to freed outside this routine! 
..*/

PetscErrorCode MatSubstract(Mat Term1, Mat Term2, Mat* Diff)
{
   Vec          col_vec1, col_vec2, diff_vec; 
  PetscErrorCode ierr;
   int          rows1, cols1, rows2, cols2, col, row;
   static PetscScalar dminusone = -1.; 
   PetscScalar  matrix_element ;
   PetscScalar  *vec_getvalues ;
   double       inf_norm_diff_vec = 0.0 ;
   double       Zero_Element_Treshold = 0.0 ;

   /*..Get sizes of terms..*/ 
   ierr = MatGetSize(Term1, &rows1, &cols1);CHKERRQ(ierr); 
   ierr = MatGetSize(Term2, &rows2, &cols2);CHKERRQ(ierr); 

   /*..Check input..*/ 
   if ( (cols1 != rows1) || (cols2 != rows2) ){
      SETERRQ(PETSC_ERR_ARG_SIZ,"Error in MatMatMult: cols1 <> rows1 or cols1 <> rows1"); 
   }

   /*..Create difference of 2 SeqAIJ matrices..*/ 
   ierr = MatCreate(PETSC_COMM_WORLD,rows1,cols1,rows1,cols1,Diff);CHKERRQ(ierr);
   ierr = MatSetType(*Diff,MATSEQAIJ);CHKERRQ(ierr);
   ierr = MatSeqAIJSetPreallocation(*Diff,0,PETSC_NULL);CHKERRQ(ierr);

   /*..Create vectors..*/ 
   ierr = VecCreate(MPI_COMM_WORLD,&col_vec1);CHKERRQ(ierr);
   ierr = VecSetSizes(col_vec1,PETSC_DECIDE,rows1);CHKERRQ(ierr);
   ierr = VecSetType(col_vec1,VECSEQ);CHKERRQ(ierr);
   ierr = VecDuplicate(col_vec1, &col_vec2);CHKERRQ(ierr);
   ierr = VecDuplicate(col_vec1, &diff_vec);CHKERRQ(ierr);

   for (col=0;col<cols1;col++){
       /*..Get collumns..*/ 
      ierr = MatGetColumnVector(Term1,col_vec1,col);CHKERRQ(ierr); 
      ierr = MatGetColumnVector(Term2,col_vec2,col);CHKERRQ(ierr); 
      /*..Substract collumns..*/ 
      ierr = VecWAXPY(&dminusone, col_vec2, col_vec1, diff_vec); 
             CHKERRQ(ierr);  
 
      /*..Compute norm..*/ 
      ierr = VecNorm( diff_vec, NORM_INFINITY, &inf_norm_diff_vec ); 
             CHKERRQ(ierr);
      /*..Set threshold..*/ 
      Zero_Element_Treshold = inf_norm_diff_vec * Machine_Precision_Eps ;

      /*..Get Term1(:,col) -  Term2(:,col) values..*/       
      ierr = VecGetArray(diff_vec, &vec_getvalues);CHKERRQ(ierr);

      for (row=0;row<rows1;row++){
          matrix_element = vec_getvalues[row];
	   if ( PetscAbsScalar( matrix_element ) >= Zero_Element_Treshold ) {
   	      MatSetValue(*Diff, row, col,matrix_element, INSERT_VALUES ); 
	   }
      }
      ierr = VecRestoreArray(diff_vec, &vec_getvalues);CHKERRQ(ierr);
   }

   ierr = MatAssemblyBegin(*Diff,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
   ierr = MatAssemblyEnd(*Diff,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

   ierr = VecDestroy(col_vec1);CHKERRQ(ierr);
   ierr = VecDestroy(col_vec2);CHKERRQ(ierr);
   ierr = VecDestroy(diff_vec);CHKERRQ(ierr);

   return 0; 
} 
/* ------------------------------------------------------------------- */
