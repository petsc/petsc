
#include "petscfunc.h"
#include "ramgfunc.h"
#include "petscksp.h"

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "KSPMonitorWriteConvHist"
/*
   KSPMonitorWriteConvHist - Write convergence history to external ASCII file. 

   Input Parameters:
     ksp   - iterative context
     n     - iteration number
     rnorm - 2-norm (preconditioned) residual value (may be estimated)
     dummy - optional user-defined monitor context (unused here)
*/
/*
  Sample usage: 
  ierr = KSPSetMonitor(ksp, KSPMonitorWriteConvHist,PETSC_NULL, 
                       PETSC_NULL); CHKERRQ(ierr); 
   
  Note: the tolerance file can viewed using gnuplot, e.g. 
  gnuplot plotpetsctol 

*/
int KSPMonitorWriteConvHist(KSP ksp,int n,double rnorm,void* ctx)
{
  char     filename[161];
  FILE     *ftol;
  /* 
  CONVHIST *convhist;

  convhist = (CONVHIST*)(ctx); 
  bnrm2 =    (*convhist).BNRM2;
  */

  sprintf(filename,"petsctol"); 

  if (n == 0){
     PetscFOpen(MPI_COMM_WORLD,filename,"w",&ftol);
     /*  PetscFPrintf(MPI_COMM_WORLD,ftol,"%14.12e \n",rnorm/bnrm2); */
     PetscFPrintf(MPI_COMM_WORLD,ftol,"%14.12e \n",rnorm); 
     PetscFClose(MPI_COMM_WORLD,ftol);
  }
  else if (n > 0) {
     PetscFOpen(MPI_COMM_WORLD,filename,"a",&ftol);
     /* PetscFPrintf(MPI_COMM_WORLD,ftol,"%14.12e \n",rnorm/bnrm2);  */
     PetscFPrintf(MPI_COMM_WORLD,ftol,"%14.12e \n",rnorm); 
     PetscFClose(MPI_COMM_WORLD,ftol);
  }
  return 0;
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "KSPMonitorAmg"
/*
   KSPMonitorWriteConvHist - Write convergence history to AMG-PETSc 
   interface external ASCII file. This routine differs from the previous one 
   by the fact that the index of each iteration is put in front of each 
   residual. 
    
   Input Parameters:
     ksp   - iterative context
     n     - iteration number
     rnorm - 2-norm (preconditioned) residual value (may be estimated)
     dummy - optional user-defined monitor context (unused here)
*/
int KSPMonitorAmg(KSP ksp,int n,double rnorm,void* ctx)
{
  char     filename[161];
  FILE     *ftol;
  /*
  CONVHIST *convhist;

  convhist = (CONVHIST*)(ctx); 
  bnrm2 =    (*convhist).BNRM2;
  */

  sprintf(filename,"petsctol"); 

  if (n == 0){
     PetscFOpen(MPI_COMM_WORLD,filename,"w",&ftol);
     /*    PetscFPrintf(MPI_COMM_WORLD,ftol,"%14.12e \n",rnorm/bnrm2); */
     PetscFPrintf(MPI_COMM_WORLD,ftol,"%d %14.12e \n",n, rnorm); 
     PetscFClose(MPI_COMM_WORLD,ftol);
  }
  else if (n > 0) {
     PetscFOpen(MPI_COMM_WORLD,filename,"a",&ftol);
     /*     PetscFPrintf(MPI_COMM_WORLD,ftol,"%14.12e \n",rnorm/bnrm2); */
     PetscFPrintf(MPI_COMM_WORLD,ftol,"%d %14.12e \n",n, rnorm); 
     PetscFClose(MPI_COMM_WORLD,ftol);
  }
  return 0;
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "KSPMonitorWriteResVecs"
/*
    KSPMonitorWriteResVecs - Write residual vectors to file. 
*/ 
int KSPMonitorWriteResVecs(KSP ksp,int n,double rnorm,void* ctx)
{
  PetscScalar *values; 
  Vec         t, v, V; 
  char        filename[161];
  int         ierr, i, numnodes; 
  CONVHIST    *convhist;
  FILE        *fout; 

  convhist = (CONVHIST*)(ctx); 
  numnodes = convhist->NUMNODES;

  sprintf(filename,"/home/domenico/Antas/Output/residual.%u",n); 
  ierr = VecCreate(MPI_COMM_WORLD,&t); CHKERRQ(ierr);
  ierr = VecSetSizes(t,numnodes,numnodes); CHKERRQ(ierr);
  ierr = VecSetType(t,VECSEQ); CHKERRQ(ierr);
  ierr = VecDuplicate(t,&v); CHKERRQ(ierr); 

  ierr = KSPBuildResidual(ksp, t, v, &V); CHKERRQ(ierr); 
  
  /*  ierr = PetscViewerFileOpenASCII(MPI_COMM_WORLD,filename,&viewer); CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(viewer,PETSC_VIEWER_ASCII_MATLAB); 
           CHKERRQ(ierr);
    ierr = VecView(V, viewer); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(viewer); CHKERRQ(ierr); */
  ierr = VecGetArray(V,&values); CHKERRQ(ierr); 
  PetscFOpen(MPI_COMM_WORLD,filename,"w",&fout);
  for (i=0;i<numnodes;i++)
      PetscFPrintf(MPI_COMM_WORLD,fout,"%14.12e \n", values[i]); 
  PetscFClose(MPI_COMM_WORLD,fout);

  ierr = VecRestoreArray(V,&values); CHKERRQ(ierr);

  return 0;
}


/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "ConvhistDestroy"
int ConvhistCtxDestroy(CONVHIST *convhist)
{
   PetscFree(convhist);
   return 0; 
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "MyConvTest"
int MyConvTest(KSP ksp,int n, double rnorm, KSPConvergedReason *reason, 
               void* ctx)
{
  int ierr; 
  double   bnrm2, rtol; 
  CONVHIST *convhist = (CONVHIST*) ctx;
 
  bnrm2 =    convhist->BNRM2;
  ierr = KSPGetTolerances(ksp, &rtol, PETSC_NULL, PETSC_NULL, PETSC_NULL); 
         CHKERRQ(ierr);
  if (rnorm/bnrm2 < rtol){ 
    PetscPrintf(MPI_COMM_WORLD,"[test] %d %g \r",n,rnorm/bnrm2);
    return 1; }
  else{
    PetscPrintf(MPI_COMM_WORLD,"[test] %d %g \r",n,rnorm/bnrm2);
    return 0;}
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "ReorderSubmatrices"
int ReorderSubmatrices(PC pc,int nsub,IS *row,IS *col,Mat *submat,void *dummy)
{
  int               i, ierr;
  IS                isrow,iscol;      /* row and column permutations */
  MatOrderingType   rtype = MATORDERING_RCM;

  for (i=0; i<nsub; i++) {
     ierr = MatGetOrdering(submat[i], rtype, &isrow, &iscol); CHKERRQ(ierr);
  }

  return 0;
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "PrintSubMatrices"
int PrintSubMatrices(PC pc,int nsub,IS *row,IS *col,Mat *submat,void *dummy)
{
  int    i, j, ierr, nloc, *glo_row_ind;

  PetscPrintf(PETSC_COMM_WORLD,"***  Overzicht van opdeling matrix *** \n");
  for (i=0; i<nsub; i++) {
    PetscPrintf(PETSC_COMM_WORLD,"\n** Jacobi Blok %d \n",i);
    ierr = ISGetSize(row[i],&nloc); CHKERRQ(ierr); 
    ierr = ISGetIndices(row[i], &glo_row_ind); CHKERRQ(ierr);
    for (j=0; j< nloc; j++) {
       PetscPrintf(PETSC_COMM_WORLD,"[%d] global row %d\n",j,glo_row_ind[j]); 
    }
  ierr = ISRestoreIndices(row[i],&glo_row_ind); CHKERRQ(ierr);
  }

  return 0;
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "ViewSubMatrices"
int ViewSubMatrices(PC pc,int nsub,IS *row,IS *col,Mat *submat,void *dummy)
{
  int         i, ierr;
  PetscViewer viewer; 
  PetscDraw   draw; 

  for (i=0; i<nsub; i++) {
     /* ierr = MatView(submat[i],PETSC_NULL); CHKERRQ(ierr); */
     ierr = PetscViewerDrawOpen(MPI_COMM_WORLD,PETSC_NULL, PETSC_NULL, 
            0, 0, 500,500,&viewer); 
     ierr = PetscViewerDrawGetDraw(viewer, 0, &draw); CHKERRQ(ierr);
     ierr = MatView(submat[i], viewer); CHKERRQ(ierr);
     ierr = PetscDrawPause(draw); CHKERRQ(ierr);
     ierr = PetscViewerDestroy(viewer); CHKERRQ(ierr);
  }

  return 0;
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "MyMatView"
int MyMatView(Mat mat,void *dummy)
{
  int         ierr;
  PetscViewer viewer; 
  PetscDraw   draw; 

  ierr = PetscViewerDrawOpen(MPI_COMM_WORLD,PETSC_NULL, PETSC_NULL, 0, 0, 500,500,&viewer); 
  ierr = PetscViewerDrawGetDraw(viewer, 0, &draw); CHKERRQ(ierr);
  ierr = MatView(mat, viewer); CHKERRQ(ierr);
  ierr = PetscDrawSetPause(draw, 5); CHKERRQ(ierr);
  ierr = PetscDrawPause(draw); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(viewer); CHKERRQ(ierr);

  return 0;
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "PrintMatrix"
int PrintMatrix(Mat mat, char* path, char* base)
{
   int         ierr,numrows, numcols, numnonzero, I, j, ncols_getrow, *cols_getrow;
   PetscViewer viewer; 
   char        filename[80]; 
   PetscScalar *vals_getrow; 
   MatInfo     info;

   /*..Get size and number of unknowns of matrix..*/ 
   ierr = MatGetSize(mat, &numrows, &numcols); CHKERRQ(ierr);
   ierr = MatGetInfo(mat,MAT_LOCAL,&info); CHKERRQ(ierr); 
   numnonzero = (int)info.nz_used;

   /*..Set file and open file for reading..*/ 
   sprintf(filename, "%s%s", path, base);
   printf("   [PrintMatrix]::Generating file %s ...\n", filename); 
   ierr = PetscViewerASCIIOpen(MPI_COMM_WORLD,filename,&viewer); 
          CHKERRQ(ierr);
 
   /*..Print file header..*/
   ierr = PetscViewerASCIIPrintf(viewer,"%% \n"); 
   if (numrows==numcols)    /*..square matrix..*/  
     ierr = PetscViewerASCIIPrintf(viewer,"%% %d %d \n", numrows, numnonzero); 
   else                     /*..rectangular matrix..*/ 
     ierr = PetscViewerASCIIPrintf(viewer,"%% %d %d %d \n", numrows, numcols,  
                                                    numnonzero); 
 
   /*..Print matrix to rowwise file..*/ 
   for (I=0;I<numrows;I++){
     /*....Get row I of matrix....*/
     ierr = MatGetRow(mat,I,&ncols_getrow,&cols_getrow,&vals_getrow); 
            CHKERRQ(ierr); 
     /*....Print row I of matrix....*/ 
     for (j=0;j<ncols_getrow;j++){
       #if defined(PETSC_USE_COMPLEX)
         ierr = PetscViewerASCIIPrintf( viewer,"%d %d %22.18e %22.18e\n", 
                I+1, cols_getrow[j]+1, real(vals_getrow[j]), 
                imag(vals_getrow[j])); 
       #else
         ierr = PetscViewerASCIIPrintf( viewer,"%d %d %22.18e \n", 
                I+1, cols_getrow[j]+1, vals_getrow[j]); 
       #endif 
     }
   }

   /*..Close file..*/ 
   ierr = PetscViewerDestroy(viewer); CHKERRQ(ierr);
   printf("   [PrintMatrix]::Done Generating file ! %s\n", filename);       
   return 0; 
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "PrintVector"
int PrintVector(Vec vec, char* path, char* base) 
{
   int         ierr,size,i;
   PetscViewer viewer; 
   char        filename[80]; 
   PetscScalar *values; 

   sprintf(filename, "%s%s%s", path, base, ".m");
   printf("   [PrintVector]::Generating file %s ...\n", filename); 
   ierr = VecGetSize(vec, &size); CHKERRQ(ierr);
   ierr = PetscMalloc(size * sizeof(PetscScalar),&values);CHKERRQ(ierr); 
   ierr = VecGetArray(vec, &values); CHKERRQ(ierr);
   ierr = PetscViewerASCIIOpen(MPI_COMM_WORLD,filename,&viewer);CHKERRQ(ierr);
   ierr = PetscViewerASCIIPrintf(viewer,"function [z] = %s()\n",base);CHKERRQ(ierr);
   ierr = PetscViewerASCIIPrintf(viewer,"z = [\n");CHKERRQ(ierr);
   for (i=0;i<size;i++){
     #if defined(PETSC_USE_COMPLEX)
       ierr = PetscViewerASCIIPrintf(viewer, "%22.18e %22.18e \n",
                                real( values[i] ), imag( values[i]);CHKERRQ(ierr);
     #else 
       ierr = PetscViewerASCIIPrintf(viewer, "%22.18e \n", values[i]);CHKERRQ(ierr);
     #endif 
   }
   ierr = PetscViewerASCIIPrintf(viewer,"]; \n");CHKERRQ(ierr);
   ierr = PetscViewerDestroy(viewer); CHKERRQ(ierr);
   ierr = VecRestoreArray(vec, &values); CHKERRQ(ierr);
   printf("   [PrintVector]::Done Generating file ! %s\n", filename);    
   return 0; 
}






