/*$Id: ex78.c,v 1.3 2000/11/13 17:11:24 hzhang Exp hzhang $*/

static char help[] =
"Reads in a matrix in ASCII Matlab format (I,J,A), read in vectors rhs and exact_solu in ASCII format, then writes them using the PETSc sparse format.\n\
Note: I and J start at 1, not 0!\n\
Input parameters are:\n\
  -Ain  <filename> : input matrix in ascii format\n\
  -bin  <filename> : input rhs in ascii format\n\
  -uin  <filename> : input true solution in ascii format\n\
Run this program: ex33h -Ain Ain -bin bin -uin uin\n\n";

#include "petscmat.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Mat    A;
  Vec    b,u,u_tmp;
  char   Ain[128],bin[128],uin[128]; 
  int    i,m,n,nz,ierr,*ib,col_i,row_i;
  Scalar val_i,*work,mone=-1.0;
  double *col,*row,res_norm,*val,*bval,*uval;
  FILE   *Afile,*bfile,*ufile;
  Viewer view;
  PetscTruth flg_A,flg_b,flg_u;

  PetscInitialize(&argc,&args,(char *)0,help);

  /* Read in matrix, rhs and exact solution from an ascii file */
  ierr = OptionsGetString(PETSC_NULL,"-Ain",Ain,127,&flg_A);CHKERRA(ierr);
  if (flg_A){
    ierr = PetscPrintf(PETSC_COMM_SELF,"\n Read matrix in ascii format ...\n");CHKERRA(ierr);
    ierr = PetscFOpen(PETSC_COMM_SELF,Ain,"r",&Afile);CHKERRA(ierr); 
    fscanf(Afile,"%d %d %d\n",&m,&n,&nz);
    printf("m: %d, n: %d, nz: %d \n", m,n,nz);
    if (m != n) SETERRQ(PETSC_ERR_ARG_SIZ, "Number of rows, cols must be same for SBAIJ format\n");

    ierr = MatCreateSeqBAIJ(PETSC_COMM_SELF,1,n,n,20,0,&A);CHKERRA(ierr);  
    ierr = VecCreateSeq(PETSC_COMM_SELF,n,&b);CHKERRA(ierr);
    ierr = VecCreateSeq(PETSC_COMM_SELF,n,&u);CHKERRA(ierr);

    val = (double*)PetscMalloc(nz*sizeof(double));CHKPTRA(val);
    row = (double*)PetscMalloc(nz*sizeof(double));CHKPTRA(row);
    col = (double*)PetscMalloc(nz*sizeof(double));CHKPTRA(col);
    for (i=0; i<nz; i++) {
      fscanf(Afile,"%le %le %le\n",row+i,col+i,val+i); /* modify according to data file! */
      row[i]--; col[i]--;  /* set index set starts at 0 */
    }
    printf("row[0]: %g, col[0]: %g, val: %g\n",row[0],col[0],val[0]);
    printf("row[last]: %g, col: %g, val: %g\n",row[nz-1],col[nz-1],val[nz-1]);
    fclose(Afile);
  }

  ierr = OptionsGetString(PETSC_NULL,"-bin",bin,127,&flg_b);CHKERRA(ierr);
  if (flg_b){
    ierr = PetscPrintf(PETSC_COMM_SELF,"\n Read rhs in ascii format ...\n");CHKERRA(ierr);
    ierr = PetscFOpen(PETSC_COMM_SELF,bin,"r",&bfile);CHKERRA(ierr); 
    bval = (double*)PetscMalloc(n*sizeof(double));CHKPTRA(bval);
    work = (Scalar*)PetscMalloc(n*sizeof(Scalar));CHKPTRA(work);
    ib   = (int*)PetscMalloc(n*sizeof(int));CHKPTRA(ib);
    for (i=0; i<n; i++) {
      /* fscanf(bfile,"%d %le\n",ib+i,bval+i); ib[i]--;  */  /* modify according to data file! */
      fscanf(bfile,"%le\n",bval+i); ib[i] = i;         /* modify according to data file! */
    }
    printf("bval[0]: %g, bval[%d]: %g\n",bval[0],ib[n-1],bval[n-1]);
    fclose(bfile);
  }

  ierr = OptionsGetString(PETSC_NULL,"-uin",uin,127,&flg_u);CHKERRA(ierr);
  if (flg_u){
    ierr = PetscPrintf(PETSC_COMM_SELF,"\n Read exact solution in ascii format ...\n");CHKERRA(ierr);
    ierr = PetscFOpen(PETSC_COMM_SELF,uin,"r",&ufile);CHKERRA(ierr); 
    uval = (double*)PetscMalloc(n*sizeof(double));CHKPTRA(bval);
    for (i=0; i<n; i++) {
      fscanf(ufile,"  %le\n",uval+i);  /* modify according to data file! */
    }
    printf("uval[0]: %g, uval[%d]: %g\n",uval[0], n-1, uval[n-1]);
    fclose(ufile);
  }

  if(flg_A){
    /*
    for (i=0; i<n; i++){ 
      ierr = MatSetValues(A,1,&i,1,&i,&zero,INSERT_VALUES);CHKERRA(ierr); 
    }
    */
    for (i=0; i<nz; i++) {
      row_i =(int)row[i]; col_i =(int)col[i]; val_i = (Scalar)val[i];
      ierr = MatSetValues(A,1,&row_i,1,&col_i,&val_i,INSERT_VALUES);CHKERRA(ierr);
    }
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
    ierr = PetscFree(col);CHKERRA(ierr);
    ierr = PetscFree(val);CHKERRA(ierr);
    ierr = PetscFree(row);CHKERRA(ierr);
  }
  if(flg_b){
    for (i=0; i<n; i++) work[i]=(Scalar)bval[i];
    ierr = VecSetValues(b,n,ib,work,INSERT_VALUES);CHKERRA(ierr);
    ierr = VecAssemblyBegin(b);CHKERRA(ierr);
    ierr = VecAssemblyEnd(b);CHKERRA(ierr);
    /* printf("b: \n"); ierr = VecView(b,VIEWER_STDOUT_SELF); */
    ierr = PetscFree(bval);CHKERRA(ierr);
  }

  if(flg_u & flg_b){
    for (i=0; i<n; i++) work[i]=(Scalar)uval[i];
    ierr = VecSetValues(u,n,ib,work,INSERT_VALUES);CHKERRA(ierr);
    ierr = VecAssemblyBegin(u);CHKERRA(ierr);
    ierr = VecAssemblyEnd(u);CHKERRA(ierr);
    /* printf("u: \n"); ierr = VecView(u,VIEWER_STDOUT_SELF); */
    ierr = PetscFree(uval);CHKERRA(ierr);                        
  }
  
  if(flg_b) {
    ierr = PetscFree(ib);CHKERRA(ierr);
    ierr = PetscFree(work);CHKERRA(ierr)
  }
  /* Check accuracy of the data */
  if (flg_A & flg_b & flg_u){
    ierr = VecDuplicate(u,&u_tmp);CHKERRA(ierr); 
    ierr = MatMult(A,u,u_tmp);CHKERRA(ierr);
    ierr = VecAXPY(&mone,b,u_tmp);CHKERRA(ierr);
    ierr = VecNorm(u_tmp,NORM_2,&res_norm);CHKERRA(ierr);
    printf("\n Accuracy of the reading data: | b - A*u |_2 : %g \n",res_norm); 

  /* Write the matrix, rhs and exact solution in Petsc binary file */
    ierr = PetscPrintf(PETSC_COMM_SELF,"\n Write matrix and rhs in binary to 'matrix.dat' ...\n");CHKERRA(ierr);
    ierr = ViewerBinaryOpen(PETSC_COMM_SELF,"matrix.dat",BINARY_CREATE,&view);CHKERRA(ierr);
    ierr = MatView(A,view);CHKERRA(ierr);
    ierr = VecView(b,view);CHKERRA(ierr);
    ierr = VecView(u,view);CHKERRA(ierr);
    ierr = ViewerDestroy(view);CHKERRA(ierr);

    ierr = VecDestroy(b);CHKERRA(ierr);
    ierr = VecDestroy(u);CHKERRA(ierr);
    ierr = VecDestroy(u_tmp);CHKERRA(ierr);
    
    ierr = MatDestroy(A);CHKERRA(ierr);
  }

  PetscFinalize();
  return 0;
}

