#include "global.h"

static  double Machine_Precision_Eps = 2.e-16; 

/* ------------------------------------------------------------------- */
#undef __FUNC__
#define __FUNC__ "MatMatMult"
/*..MatMatMult - Computes product Fact1 * Fact2
    INPUT: Fact1, Fact2: the matrices to be multiplied 
    OUPUT: Prod:         the result                        
  NOTE: Memory needed by product has to freed outside this routine! 
..*/

int MatMatMult(Mat Fact1, Mat Fact2, Mat* Prod)
{ 

   Vec     col_vec, mat_col_vec; 
   Scalar  *vec_getvalues; 
   Scalar  matrix_element ;
   int     rows1, cols1, rows2, cols2, col, row, ierr;
   double  inf_norm_mat_col_vec = 0.0 ;
   double  Zero_Element_Treshold = 0.0 ;

   /*..Get sizes of factors..*/ 
   ierr = MatGetSize(Fact1, &rows1, &cols1); CHKERRA(ierr); 
   ierr = MatGetSize(Fact2, &rows2, &cols2); CHKERRA(ierr); 

   /*..Check input..*/ 
   if (cols1 != rows2){
      Die("\n Error in MatMatMult: cols1 <> rows2 \n\n"); 
   }
 
   /*..Create product matrix..*/ 
   ierr =  MatCreateSeqAIJ(PETSC_COMM_WORLD,rows1,cols2,PETSC_NULL,
           PETSC_NULL,Prod); CHKERRA(ierr);

   /*..Create vector to hold Fact2(:,i)..*/ 
   ierr = VecCreate(MPI_COMM_WORLD,rows2,rows2,&col_vec); CHKERRA(ierr);
   ierr = VecSetType(col_vec,VEC_SEQ); CHKERRA(ierr);

   /*..Create vector to hold Fact1 * Fact2(:,i)..*/ 
   ierr = VecCreate(MPI_COMM_WORLD,rows1,rows1,&mat_col_vec); CHKERRA(ierr);
   ierr = VecSetType(mat_col_vec,VEC_SEQ); CHKERRA(ierr);

   for (col=0;col<cols2;col++){

       /*..Get Fact2(:,col)..*/ 
      ierr = MatGetColumnVector(Fact2,col_vec,col); CHKERRA(ierr); 
      
      /*..Conpute Fact1 * Fact2(:,col)..*/
      ierr = MatMult(Fact1, col_vec, mat_col_vec); CHKERRA(ierr);
      /*..Compute norm..*/ 
      ierr = VecNorm( mat_col_vec, NORM_INFINITY, &inf_norm_mat_col_vec ); 
             CHKERRA(ierr);
      /*..Set threshold..*/ 
      Zero_Element_Treshold = inf_norm_mat_col_vec * Machine_Precision_Eps ;

      /*..Get Fact1 * Fact2(:,col) values..*/       
      ierr = VecGetArray(mat_col_vec, &vec_getvalues); CHKERRA(ierr);

      for (row=0;row<rows1;row++){
          matrix_element = vec_getvalues[row];
	   if ( PetscAbsScalar( matrix_element ) >= Zero_Element_Treshold ) {
   	      MatSetValue(*Prod, row, col,matrix_element, INSERT_VALUES ); 
	   }
      }
      ierr = VecRestoreArray(mat_col_vec, &vec_getvalues); CHKERRA(ierr);
   }

   ierr = MatAssemblyBegin(*Prod,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);
   ierr = MatAssemblyEnd(*Prod,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);

   ierr = VecDestroy(col_vec); CHKERRA(ierr);
   ierr = VecDestroy(mat_col_vec); CHKERRA(ierr);

   return 0; 
}
/* ------------------------------------------------------------------- */
#undef __FUNC__
#define __FUNC__ "MatSubstract"
/*..MatSubstract - Computes the difference Term1 - Term2 
    INPUT:  Term1, Term2 : The input matrices 
    OUTPUT: Prod:          the difference 
  NOTE: Memory needed by difference has to freed outside this routine! 
..*/

int MatSubstract(Mat Term1, Mat Term2, Mat* Diff)
{
   Vec     col_vec1, col_vec2, diff_vec; 
   int     rows1, cols1, rows2, cols2, col, row, ierr;
   static Scalar dminusone = -1.; 
   Scalar  matrix_element ;
   Scalar  *vec_getvalues ;
   double  inf_norm_diff_vec = 0.0 ;
   double  Zero_Element_Treshold = 0.0 ;

   /*..Get sizes of terms..*/ 
   ierr = MatGetSize(Term1, &rows1, &cols1); CHKERRA(ierr); 
   ierr = MatGetSize(Term2, &rows2, &cols2); CHKERRA(ierr); 

   /*..Check input..*/ 
   if ( (cols1 != rows1) || (cols2 != rows2) ){
      Die("\n Error in MatMatMult: cols1 <> rows1 or cols1 <> rows1 \n\n"); 
   }

   /*..Create difference matrix..*/ 
   ierr =  MatCreateSeqAIJ(PETSC_COMM_WORLD,rows1,cols1,PETSC_NULL,
           PETSC_NULL,Diff); CHKERRA(ierr);

   /*..Create vectors..*/ 
   ierr = VecCreate(MPI_COMM_WORLD,rows1,rows1,&col_vec1); CHKERRA(ierr);
   ierr = VecSetType(col_vec1,VEC_SEQ); CHKERRA(ierr);
   ierr = VecDuplicate(col_vec1, &col_vec2); CHKERRA(ierr);
   ierr = VecDuplicate(col_vec1, &diff_vec); CHKERRA(ierr);

   for (col=0;col<cols1;col++){
       /*..Get collumns..*/ 
      ierr = MatGetColumnVector(Term1,col_vec1,col); CHKERRA(ierr); 
      ierr = MatGetColumnVector(Term2,col_vec2,col); CHKERRA(ierr); 
      /*..Subtract columns..*/ 
      ierr = VecWAXPY(diff_vec, dminusone, col_vec2, col_vec1); 
             CHKERRA(ierr);  
 
      /*..Compute norm..*/ 
      ierr = VecNorm( diff_vec, NORM_INFINITY, &inf_norm_diff_vec ); 
             CHKERRA(ierr);
      /*..Set threshold..*/ 
      Zero_Element_Treshold = inf_norm_diff_vec * Machine_Precision_Eps ;

      /*..Get Term1(:,col) -  Term2(:,col) values..*/       
      ierr = VecGetArray(diff_vec, &vec_getvalues); CHKERRA(ierr);

      for (row=0;row<rows1;row++){
          matrix_element = vec_getvalues[row];
	   if ( PetscAbsScalar( matrix_element ) >= Zero_Element_Treshold ) {
   	      MatSetValue(*Diff, row, col,matrix_element, INSERT_VALUES ); 
	   }
      }
      ierr = VecRestoreArray(diff_vec, &vec_getvalues); CHKERRA(ierr);
   }

   ierr = MatAssemblyBegin(*Diff,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);
   ierr = MatAssemblyEnd(*Diff,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);

   ierr = VecDestroy(col_vec1); CHKERRA(ierr);
   ierr = VecDestroy(col_vec2); CHKERRA(ierr);
   ierr = VecDestroy(diff_vec); CHKERRA(ierr);

   return 0; 
} 
