/*$Id: ex78.c,v 1.14 2001/08/07 21:30:08 bsmith Exp $*/

static char help[] = "Reads in a matrix in ASCII Matlab format (I,J,A), read in vectors rhs and exact_solu in ASCII format.\n\
Writes them using the PETSc sparse format.\n\
Note: I and J start at 1, not 0!\n\
Input parameters are:\n\
  -Ain  <filename> : input matrix in ascii format\n\
  -bin  <filename> : input rhs in ascii format\n\
  -uin  <filename> : input true solution in ascii format\n\
Run this program: ex33h -Ain Ain -bin bin -uin uin\n\n";

#include "petscmat.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat         A;
  Vec         b,u,u_tmp;
  char        Ain[128],bin[128],uin[128]; 
  int         i,m,n,nz,ierr,*ib=0,col_i,row_i;
  PetscScalar val_i,*work=0,mone=-1.0;
  PetscReal   *col=0,*row=0,res_norm,*val=0,*bval=0,*uval=0;
  FILE        *Afile,*bfile,*ufile;
  PetscViewer view;
  PetscTruth  flg_A,flg_b,flg_u;

  PetscInitialize(&argc,&args,(char *)0,help);

  /* Read in matrix, rhs and exact solution from an ascii file */
  ierr = PetscOptionsGetString(PETSC_NULL,"-Ain",Ain,127,&flg_A);CHKERRQ(ierr);
  if (flg_A){
    ierr = PetscPrintf(PETSC_COMM_SELF,"\n Read matrix in ascii format ...\n");CHKERRQ(ierr);
    ierr = PetscFOpen(PETSC_COMM_SELF,Ain,"r",&Afile);CHKERRQ(ierr); 
    fscanf(Afile,"%d %d %d\n",&m,&n,&nz);
    printf("m: %d, n: %d, nz: %d \n", m,n,nz);
    if (m != n) SETERRQ(PETSC_ERR_ARG_SIZ, "Number of rows, cols must be same for SBAIJ format\n");

    ierr = MatCreateSeqBAIJ(PETSC_COMM_SELF,1,n,n,20,0,&A);CHKERRQ(ierr);  
    ierr = VecCreateSeq(PETSC_COMM_SELF,n,&b);CHKERRQ(ierr);
    ierr = VecCreateSeq(PETSC_COMM_SELF,n,&u);CHKERRQ(ierr);

    ierr = PetscMalloc(nz*sizeof(PetscReal),&val);CHKERRQ(ierr);
    ierr = PetscMalloc(nz*sizeof(PetscReal),&row);CHKERRQ(ierr);
    ierr = PetscMalloc(nz*sizeof(PetscReal),&col);CHKERRQ(ierr);
    for (i=0; i<nz; i++) {
      fscanf(Afile,"%le %le %le\n",row+i,col+i,val+i); /* modify according to data file! */
      row[i]--; col[i]--;  /* set index set starts at 0 */
    }
    printf("row[0]: %g, col[0]: %g, val: %g\n",row[0],col[0],val[0]);
    printf("row[last]: %g, col: %g, val: %g\n",row[nz-1],col[nz-1],val[nz-1]);
    fclose(Afile);
  }

  ierr = PetscOptionsGetString(PETSC_NULL,"-bin",bin,127,&flg_b);CHKERRQ(ierr);
  if (flg_b){
    ierr = PetscPrintf(PETSC_COMM_SELF,"\n Read rhs in ascii format ...\n");CHKERRQ(ierr);
    ierr = PetscFOpen(PETSC_COMM_SELF,bin,"r",&bfile);CHKERRQ(ierr); 
    ierr = PetscMalloc(n*sizeof(PetscReal),&bval);CHKERRQ(ierr);
    ierr = PetscMalloc(n*sizeof(PetscScalar),&work);CHKERRQ(ierr);
    ierr = PetscMalloc(n*sizeof(int),&ib);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      /* fscanf(bfile,"%d %le\n",ib+i,bval+i); ib[i]--;  */  /* modify according to data file! */
      fscanf(bfile,"%le\n",bval+i); ib[i] = i;         /* modify according to data file! */
    }
    printf("bval[0]: %g, bval[%d]: %g\n",bval[0],ib[n-1],bval[n-1]);
    fclose(bfile);
  }

  ierr = PetscOptionsGetString(PETSC_NULL,"-uin",uin,127,&flg_u);CHKERRQ(ierr);
  if (flg_u){
    ierr = PetscPrintf(PETSC_COMM_SELF,"\n Read exact solution in ascii format ...\n");CHKERRQ(ierr);
    ierr = PetscFOpen(PETSC_COMM_SELF,uin,"r",&ufile);CHKERRQ(ierr); 
    ierr = PetscMalloc(n*sizeof(PetscReal),&uval);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      fscanf(ufile,"  %le\n",uval+i);  /* modify according to data file! */
    }
    printf("uval[0]: %g, uval[%d]: %g\n",uval[0], n-1, uval[n-1]);
    fclose(ufile);
  }

  if(flg_A){
    /*
    for (i=0; i<n; i++){ 
      ierr = MatSetValues(A,1,&i,1,&i,&zero,INSERT_VALUES);CHKERRQ(ierr); 
    }
    */
    for (i=0; i<nz; i++) {
      row_i =(int)row[i]; col_i =(int)col[i]; val_i = (PetscScalar)val[i];
      ierr = MatSetValues(A,1,&row_i,1,&col_i,&val_i,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = PetscFree(col);CHKERRQ(ierr);
    ierr = PetscFree(val);CHKERRQ(ierr);
    ierr = PetscFree(row);CHKERRQ(ierr);
  }
  if(flg_b){
    for (i=0; i<n; i++) work[i]=(PetscScalar)bval[i];
    ierr = VecSetValues(b,n,ib,work,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(b);CHKERRQ(ierr);
    /* printf("b: \n"); ierr = VecView(b,PETSC_VIEWER_STDOUT_SELF); */
    ierr = PetscFree(bval);CHKERRQ(ierr);
  }

  if(flg_u & flg_b){
    for (i=0; i<n; i++) work[i]=(PetscScalar)uval[i];
    ierr = VecSetValues(u,n,ib,work,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(u);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(u);CHKERRQ(ierr);
    /* printf("u: \n"); ierr = VecView(u,PETSC_VIEWER_STDOUT_SELF); */
    ierr = PetscFree(uval);CHKERRQ(ierr);                        
  }
  
  if(flg_b) {
    ierr = PetscFree(ib);CHKERRQ(ierr);
    ierr = PetscFree(work);CHKERRQ(ierr);
  }
  /* Check accuracy of the data */
  if (flg_A & flg_b & flg_u){
    ierr = VecDuplicate(u,&u_tmp);CHKERRQ(ierr); 
    ierr = MatMult(A,u,u_tmp);CHKERRQ(ierr);
    ierr = VecAXPY(&mone,b,u_tmp);CHKERRQ(ierr);
    ierr = VecNorm(u_tmp,NORM_2,&res_norm);CHKERRQ(ierr);
    printf("\n Accuracy of the reading data: | b - A*u |_2 : %g \n",res_norm); 

  /* Write the matrix, rhs and exact solution in Petsc binary file */
    ierr = PetscPrintf(PETSC_COMM_SELF,"\n Write matrix and rhs in binary to 'matrix.dat' ...\n");CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF,"matrix.dat",PETSC_FILE_CREATE,&view);CHKERRQ(ierr);
    ierr = MatView(A,view);CHKERRQ(ierr);
    ierr = VecView(b,view);CHKERRQ(ierr);
    ierr = VecView(u,view);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(view);CHKERRQ(ierr);

    ierr = VecDestroy(b);CHKERRQ(ierr);
    ierr = VecDestroy(u);CHKERRQ(ierr);
    ierr = VecDestroy(u_tmp);CHKERRQ(ierr);
    
    ierr = MatDestroy(A);CHKERRQ(ierr);
  }

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

