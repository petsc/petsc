
static char help[] = "Reads in a matrix in ASCII Matlab format (I,J,A), read in vectors rhs and exact_solu in ASCII format.\n\
Writes them using the PETSc sparse format.\n\
Note: I and J start at 1, not 0, use -noshift if indices in file start with zero!\n\
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
  Mat            A;
  Vec            b,u,u_tmp;
  char           Ain[PETSC_MAX_PATH_LEN],bin[PETSC_MAX_PATH_LEN],uin[PETSC_MAX_PATH_LEN]; 
  PetscErrorCode ierr;
  int            m,n,nz,dummy,*col=0,*row=0; /* these are fscaned so kept as int */
  PetscInt       i,col_i,row_i,*nnz,*ib,shift = 1,sizes[3],nsizes;
  PetscScalar    val_i,*work=0;
  PetscReal      res_norm,*val=0,*bval=0,*uval=0;
  FILE           *Afile,*bfile,*ufile;
  PetscViewer    view;
  PetscTruth     flg_A,flg_b,flg_u,flg;

  PetscInitialize(&argc,&args,(char *)0,help);

  /* Read in matrix, rhs and exact solution from an ascii file */
  ierr = PetscOptionsGetString(PETSC_NULL,"-Ain",Ain,PETSC_MAX_PATH_LEN-1,&flg_A);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-noshift",&flg);CHKERRQ(ierr);
  if (flg) shift = 0;
  if (flg_A){
    ierr = PetscPrintf(PETSC_COMM_SELF,"\n Read matrix in ascii format ...\n");CHKERRQ(ierr);
    ierr = PetscFOpen(PETSC_COMM_SELF,Ain,"r",&Afile);CHKERRQ(ierr); 
    nsizes = 3;
    ierr = PetscOptionsGetIntArray(PETSC_NULL,"-nosizesinfile",sizes,&nsizes,&flg);CHKERRQ(ierr);
    if (flg) {
      if (nsizes != 3) SETERRQ(1,"Must pass in three m,n,nz as arguments for -nosizesinfile");
      m = sizes[0];
      n = sizes[1];
      nz = sizes[2];
    } else {
      fscanf(Afile,"%d %d %d\n",&m,&n,&nz);
    }
    printf("m: %d, n: %d, nz: %d \n", m,n,nz);
    if (m != n) SETERRQ(PETSC_ERR_ARG_SIZ, "Number of rows, cols must be same for SBAIJ format\n");

    ierr = VecCreateSeq(PETSC_COMM_SELF,n,&b);CHKERRQ(ierr);
    ierr = VecCreateSeq(PETSC_COMM_SELF,n,&u);CHKERRQ(ierr);

    ierr = PetscMalloc(m*sizeof(PetscInt),&nnz);CHKERRQ(ierr);
    ierr = PetscMemzero(nnz,m*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscMalloc(nz*sizeof(PetscReal),&val);CHKERRQ(ierr);
    ierr = PetscMalloc(nz*sizeof(PetscInt),&row);CHKERRQ(ierr);
    ierr = PetscMalloc(nz*sizeof(PetscInt),&col);CHKERRQ(ierr);
    for (i=0; i<nz; i++) {
      fscanf(Afile,"%d %d %le\n",row+i,col+i,(double*)val+i); /* modify according to data file! */
      row[i] -= shift; col[i] -= shift;  /* set index set starts at 0 */
      nnz[row[i]]++;
    }
    printf("row[0]: %d, col[0]: %d, val: %g\n",row[0],col[0],val[0]);
    printf("row[last]: %d, col: %d, val: %g\n",row[nz-1],col[nz-1],val[nz-1]);
    fflush(stdout);
    fclose(Afile);
    ierr = MatCreateSeqBAIJ(PETSC_COMM_SELF,1,n,n,0,nnz,&A);CHKERRQ(ierr);  
    ierr = PetscFree(nnz);CHKERRQ(ierr);
  }

  ierr = PetscOptionsGetString(PETSC_NULL,"-bin",bin,PETSC_MAX_PATH_LEN-1,&flg_b);CHKERRQ(ierr);
  if (flg_b){
    ierr = PetscPrintf(PETSC_COMM_SELF,"\n Read rhs in ascii format ...\n");CHKERRQ(ierr);
    ierr = PetscFOpen(PETSC_COMM_SELF,bin,"r",&bfile);CHKERRQ(ierr); 
    ierr = PetscMalloc(n*sizeof(PetscReal),&bval);CHKERRQ(ierr);
    ierr = PetscMalloc(n*sizeof(PetscScalar),&work);CHKERRQ(ierr);
    ierr = PetscMalloc(n*sizeof(PetscInt),&ib);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      /* fscanf(bfile,"%d %le\n",ib+i,bval+i); ib[i]--;  */  /* modify according to data file! */
      fscanf(bfile,"%d %le\n",&dummy,(double*)(bval+i)); ib[i] = i;         /* modify according to data file! */
    }
    printf("bval[0]: %g, bval[%d]: %g\n",bval[0],(int)ib[n-1],bval[n-1]);
    fflush(stdout);
    fclose(bfile);
  }

  ierr = PetscOptionsGetString(PETSC_NULL,"-uin",uin,PETSC_MAX_PATH_LEN-1,&flg_u);CHKERRQ(ierr);
  if (flg_u){
    ierr = PetscPrintf(PETSC_COMM_SELF,"\n Read exact solution in ascii format ...\n");CHKERRQ(ierr);
    ierr = PetscFOpen(PETSC_COMM_SELF,uin,"r",&ufile);CHKERRQ(ierr); 
    ierr = PetscMalloc(n*sizeof(PetscReal),&uval);CHKERRQ(ierr);
    for (i=0; i<n; i++) {
      fscanf(ufile,"%d  %le\n",&dummy,(double*)(uval+i));  /* modify according to data file! */
    }
    printf("uval[0]: %g, uval[%d]: %g\n",uval[0], n-1, uval[n-1]);
    fflush(stdout);
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
    ierr = VecAXPY(u_tmp,-1.0,b);CHKERRQ(ierr);
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

