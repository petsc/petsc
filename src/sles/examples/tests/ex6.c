
#include "draw.h"
#include "mat.h"
#include "sles.h"
#include "petsc.h"
#include <stdio.h>
#include <stdlib.h>

int FiletoMat(char *,Mat *,Vec *);

int main(int argc,char **args)
{
  int        ierr, its;
  double     time, norm;
  Scalar     zero = 0.0, none = -1.0;
  Vec        x, b, u;
  Mat        A;
  SLES       sles;
/*  DrawCtx    draw; */
  char       file[128];

  PetscInitialize(&argc,&args,0,0);

/*   Read in matrix and RHS   */
  OptionsGetString(0,"-f",file,127);
  FiletoMat(file,&A,&b);
  printf("Reading matrix completes.\n");

MatView(A,STDOUT_VIEWER_WORLD);
/*
  ierr = DrawOpenX(MPI_COMM_WORLD,0,"Test1.matrix1",0,0,600,500,&draw);

  CHKERRA(ierr);
  ierr = MatView(A, (Viewer) draw); CHKERRA(ierr);
*/
/*
  ierr = DrawDestroy(draw); CHKERRA(ierr);

  ierr = MatView(A,SYNC_STDOUT_VIEWER); CHKERRA(ierr);
  ierr = VecView(b,SYNC_STDOUT_VIEWER); CHKERRA(ierr);
*/

/*   Set up solution   */
  ierr = VecDuplicate(b,&x); CHKERRA(ierr);
  ierr = VecDuplicate(b,&u); CHKERRA(ierr);
  ierr = VecSet(&zero,x); CHKERRA(ierr);

/*   Solve solution    */
  ierr = SLESCreate(MPI_COMM_WORLD,&sles); CHKERRA(ierr);
  ierr = SLESSetOperators(sles,A,A, ALLMAT_DIFFERENT_NONZERO_PATTERN);
  CHKERRA(ierr);
  ierr = SLESSetFromOptions(sles); CHKERRA(ierr);
  time = MPI_Wtime();
  ierr = SLESSolve(sles,b,x,&its); CHKERRA(ierr);
  time = MPI_Wtime()-time;

/*   Show result   */
  ierr = MatMult(A,x,u);
  ierr = VecAXPY(&none,b,u); CHKERRA(ierr);
  ierr = VecNorm(u,&norm); CHKERRA(ierr);
  MPIU_printf(MPI_COMM_WORLD,"Number of iterations = %3d\n",its);
  MPIU_printf(MPI_COMM_WORLD,"Residual norm = %10.4e\n",norm);
  MPIU_printf(MPI_COMM_WORLD,"Time for solve = %5.2f\n",time);

/*   Dumping   */
  ierr = SLESDestroy(sles); CHKERRA(ierr);
  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(b); CHKERRA(ierr);
  ierr = VecDestroy(u); CHKERRA(ierr);
  ierr = MatDestroy(A); CHKERRA(ierr);

  PetscFinalize();
  return 0;
}

int FiletoMat(char *filename, Mat *mat, Vec *rhs)
{
  int    i, j, m, n, nnz;
  int    ierr, mytid, numtids, start, end;
  int    *col, *row, *brow, length;
  Scalar *val, *bval;
  char   buf[30];
  FILE   *file;
  Mat    A;
  Vec    b;

  MPI_Comm_rank(MPI_COMM_WORLD,&mytid);
  MPI_Comm_size(MPI_COMM_WORLD,&numtids);

/*  if (mytid == 0) { */
    if ((file = fopen(filename,"r")) == NULL) {
      printf("cannot open file\n");
      exit(0);
    }
    fscanf(file,"  NUNKNS =%d  NCOEFF =%d\n",&n,&nnz);
    fscanf(file,"  JA POINTER IN SLAPSV\n");
/*  } */

  ierr = MatCreateMPIRowbs(MPI_COMM_WORLD,PETSC_DECIDE,n,10,0,0,&A);
  CHKERRA(ierr);
  ierr = VecCreateMPI(MPI_COMM_WORLD,PETSC_DECIDE,n,&b); CHKERRA(ierr);

/*  if (mytid == 0) { */
    col = (int *) PETSCMALLOC((n+1)*sizeof(int));
    for (i=0; i<n+1; i++)
      fscanf(file,"     I=%d%d\n",&j,&col[i]);
    fscanf(file,"  EOD JA\n");

    val = (Scalar *) PETSCMALLOC(nnz*sizeof(Scalar));
    row = (int *) PETSCMALLOC(nnz*sizeof(int));
    fscanf(file,"  COEFFICIENT MATRIX IN SLAPSV: I, IA, A\n");
    for (i=0; i<nnz; i++) {
      fscanf(file,"    %d%d%s\n",&j,&row[i],buf);
      val[i] = atof(buf);
      row[i]--;
    }
    fscanf(file,"  EOD IA\n");

    bval = (Scalar *) PETSCMALLOC(n*sizeof(Scalar));
    brow = (int *) PETSCMALLOC(n*sizeof(int));
    fscanf(file,"  RESIDUAL IN SLAPSV ;IRHS=%d\n",&j);
    for (i=0; i<n; i++) {
      fscanf(file,"      %d%s%d\n",&j,buf,&j);
      bval[i] = atof(buf);
      brow[i] = i;
    }
    fscanf(file,"  EOD RESIDUAL");

    fclose(file);
/*  } */

  PLogDestroy();
  PLogBegin();
  m = n/numtids+1;
  start = mytid*m;
  end = (mytid+1)*m; if (end > n) end = n;
  for (j=start; j<end; j++) {
    length = col[j+1]-col[j];
    MatSetValues(A,length,&row[col[j]-1],1,&j,&val[col[j]-1],INSERTVALUES);
  }
  ierr = MatAssemblyBegin(A,FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatAssemblyEnd(A,FINAL_ASSEMBLY); CHKERRA(ierr);

  ierr = VecGetOwnershipRange(b,&start,&end); CHKERRQ(ierr);
  ierr = VecSetValues(b,end-start,brow+start,bval+start,INSERTVALUES); 
  CHKERRQ(ierr);
  ierr = VecAssemblyBegin(b); CHKERRA(ierr);
  ierr = VecAssemblyEnd(b); CHKERRA(ierr);

  *mat = A;
  *rhs = b;

  PETSCFREE(col); PETSCFREE(val); PETSCFREE(row);
  PETSCFREE(bval); PETSCFREE(brow);

  return 0;
}
