static char help[] = "Benchmark for MatMatMult() of AIJ matrices using different 2d finite-difference stencils.\n\n";

#include <petscmat.h>

/* Converts 3d grid coordinates (i,j,k) for a grid of size m \times n to global indexing. Pass k = 0 for a 2d grid. */
int global_index(PetscInt i,PetscInt j,PetscInt k, PetscInt m, PetscInt n) { return i + j * m + k * m * n; }

int main(int argc,char **argv)
{
  Mat            A,B,C,PtAP,PtAP_copy,PtAP_squared;
  PetscInt       i,M,N,Istart,Iend,n=7,j,J,Ii,m=8,k,o=1;
  PetscScalar    v;
  PetscErrorCode ierr;
  PetscBool      equal=PETSC_FALSE,mat_view=PETSC_FALSE;
  char           stencil[PETSC_MAX_PATH_LEN];
#if defined(PETSC_USE_LOG)
  PetscLogStage  fullMatMatMultStage;
#endif

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-o",&o,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL,"-result_view",&mat_view);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-stencil",stencil,sizeof(stencil),NULL);CHKERRQ(ierr);

  /* Create a aij matrix A */
  M    = N = m*n*o;
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,M,N);CHKERRQ(ierr);
  ierr = MatSetType(A,MATAIJ);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);

  /* Consistency checks */
  if (o < 1 || m < 1 || n < 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Dimensions need to be larger than zero!");

  /************ 2D stencils ***************/
  ierr = PetscStrcmp(stencil, "2d5point", &equal);CHKERRQ(ierr);
  if (equal) {   /* 5-point stencil, 2D */
    ierr = MatMPIAIJSetPreallocation(A,5,NULL,5,NULL);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(A,5,NULL);CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);
    for (Ii=Istart; Ii<Iend; Ii++) {
      v = -1.0; k = Ii / (m*n); j = (Ii - k * m * n) / m; i = (Ii - k * m * n - j * m);
      if (i>0)   {J = global_index(i-1,j,k,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (i<m-1) {J = global_index(i+1,j,k,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (j>0)   {J = global_index(i,j-1,k,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (j<n-1) {J = global_index(i,j+1,k,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      v = 4.0; ierr = MatSetValues(A,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = PetscStrcmp(stencil, "2d9point", &equal);CHKERRQ(ierr);
  if (equal) {      /* 9-point stencil, 2D */
    ierr = MatMPIAIJSetPreallocation(A,9,NULL,9,NULL);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(A,9,NULL);CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);
    for (Ii=Istart; Ii<Iend; Ii++) {
      v = -1.0; k = Ii / (m*n); j = (Ii - k * m * n) / m; i = (Ii - k * m * n - j * m);
      if (i>0)            {J = global_index(i-1,j,  k,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (i>0 && j>0)   {J = global_index(i-1,j-1,k,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (j>0)            {J = global_index(i,  j-1,k,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (i<m-1 && j>0)   {J = global_index(i+1,j-1,k,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (i<m-1)          {J = global_index(i+1,j,  k,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (i<m-1 && j<n-1) {J = global_index(i+1,j+1,k,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (j<n-1)          {J = global_index(i,  j+1,k,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (i>0 && j<n-1) {J = global_index(i-1,j+1,k,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      v = 8.0; ierr = MatSetValues(A,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = PetscStrcmp(stencil, "2d9point2", &equal);CHKERRQ(ierr);
  if (equal) {      /* 9-point Cartesian stencil (width 2 per coordinate), 2D */
    ierr = MatMPIAIJSetPreallocation(A,9,NULL,9,NULL);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(A,9,NULL);CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);
    for (Ii=Istart; Ii<Iend; Ii++) {
      v = -1.0; k = Ii / (m*n); j = (Ii - k * m * n) / m; i = (Ii - k * m * n - j * m);
      if (i>0)   {J = global_index(i-1,j,k,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (i>1)   {J = global_index(i-2,j,k,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (i<m-1) {J = global_index(i+1,j,k,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (i<m-2) {J = global_index(i+2,j,k,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (j>0)   {J = global_index(i,j-1,k,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (j>1)   {J = global_index(i,j-2,k,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (j<n-1) {J = global_index(i,j+1,k,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (j<n-2) {J = global_index(i,j+2,k,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      v = 8.0; ierr = MatSetValues(A,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = PetscStrcmp(stencil, "2d13point", &equal);CHKERRQ(ierr);
  if (equal) {      /* 13-point Cartesian stencil (width 3 per coordinate), 2D */
    ierr = MatMPIAIJSetPreallocation(A,13,NULL,13,NULL);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(A,13,NULL);CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);
    for (Ii=Istart; Ii<Iend; Ii++) {
      v = -1.0; k = Ii / (m*n); j = (Ii - k * m * n) / m; i = (Ii - k * m * n - j * m);
      if (i>0)   {J = global_index(i-1,j,k,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (i>1)   {J = global_index(i-2,j,k,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (i>2)   {J = global_index(i-3,j,k,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (i<m-1) {J = global_index(i+1,j,k,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (i<m-2) {J = global_index(i+2,j,k,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (i<m-3) {J = global_index(i+3,j,k,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (j>0)   {J = global_index(i,j-1,k,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (j>1)   {J = global_index(i,j-2,k,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (j>2)   {J = global_index(i,j-3,k,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (j<n-1) {J = global_index(i,j+1,k,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (j<n-2) {J = global_index(i,j+2,k,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (j<n-3) {J = global_index(i,j+3,k,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      v = 12.0; ierr = MatSetValues(A,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  /************ 3D stencils ***************/
  ierr = PetscStrcmp(stencil, "3d7point", &equal);CHKERRQ(ierr);
  if (equal) {      /* 7-point stencil, 3D */
    ierr = MatMPIAIJSetPreallocation(A,7,NULL,7,NULL);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(A,7,NULL);CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);
    for (Ii=Istart; Ii<Iend; Ii++) {
      v = -1.0; k = Ii / (m*n); j = (Ii - k * m * n) / m; i = (Ii - k * m * n - j * m);
      if (i>0)   {J = global_index(i-1,j,k,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (i<m-1) {J = global_index(i+1,j,k,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (j>0)   {J = global_index(i,j-1,k,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (j<n-1) {J = global_index(i,j+1,k,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (k>0)   {J = global_index(i,j,k-1,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (k<o-1) {J = global_index(i,j,k+1,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      v = 6.0; ierr = MatSetValues(A,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = PetscStrcmp(stencil, "3d13point", &equal);CHKERRQ(ierr);
  if (equal) {      /* 13-point stencil, 3D */
    ierr = MatMPIAIJSetPreallocation(A,13,NULL,13,NULL);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(A,13,NULL);CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);
    for (Ii=Istart; Ii<Iend; Ii++) {
      v = -1.0; k = Ii / (m*n); j = (Ii - k * m * n) / m; i = (Ii - k * m * n - j * m);
      if (i>0)   {J = global_index(i-1,j,k,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (i>1)   {J = global_index(i-2,j,k,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (i<m-1) {J = global_index(i+1,j,k,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (i<m-2) {J = global_index(i+2,j,k,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (j>0)   {J = global_index(i,j-1,k,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (j>1)   {J = global_index(i,j-2,k,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (j<n-1) {J = global_index(i,j+1,k,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (j<n-2) {J = global_index(i,j+2,k,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (k>0)   {J = global_index(i,j,k-1,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (k>1)   {J = global_index(i,j,k-2,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (k<o-1) {J = global_index(i,j,k+1,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (k<o-2) {J = global_index(i,j,k+2,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      v = 12.0; ierr = MatSetValues(A,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = PetscStrcmp(stencil, "3d19point", &equal);CHKERRQ(ierr);
  if (equal) {      /* 19-point stencil, 3D */
    ierr = MatMPIAIJSetPreallocation(A,19,NULL,19,NULL);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(A,19,NULL);CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);
    for (Ii=Istart; Ii<Iend; Ii++) {
      v = -1.0; k = Ii / (m*n); j = (Ii - k * m * n) / m; i = (Ii - k * m * n - j * m);
      /* one hop */
      if (i>0)   {J = global_index(i-1,j,k,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (i<m-1) {J = global_index(i+1,j,k,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (j>0)   {J = global_index(i,j-1,k,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (j<n-1) {J = global_index(i,j+1,k,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (k>0)   {J = global_index(i,j,k-1,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (k<o-1) {J = global_index(i,j,k+1,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      /* two hops */
      if (i>0   && j>0)   {J = global_index(i-1,j-1,k  ,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (i>0   && k>0)   {J = global_index(i-1,j,  k-1,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (i>0   && j<n-1) {J = global_index(i-1,j+1,k  ,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (i>0   && k<o-1) {J = global_index(i-1,j,  k+1,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (i<m-1 && j>0)   {J = global_index(i+1,j-1,k  ,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (i<m-1 && k>0)   {J = global_index(i+1,j,  k-1,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (i<m-1 && j<n-1) {J = global_index(i+1,j+1,k  ,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (i<m-1 && k<o-1) {J = global_index(i+1,j,  k+1,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (j>0   && k>0)   {J = global_index(i,  j-1,k-1,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (j>0   && k<o-1) {J = global_index(i,  j-1,k+1,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (j<n-1 && k>0)   {J = global_index(i,  j+1,k-1,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (j<n-1 && k<o-1) {J = global_index(i,  j+1,k+1,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      v = 18.0; ierr = MatSetValues(A,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = PetscStrcmp(stencil, "3d27point", &equal);CHKERRQ(ierr);
  if (equal) {      /* 27-point stencil, 3D */
    ierr = MatMPIAIJSetPreallocation(A,27,NULL,27,NULL);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(A,27,NULL);CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);
    for (Ii=Istart; Ii<Iend; Ii++) {
      v = -1.0; k = Ii / (m*n); j = (Ii - k * m * n) / m; i = (Ii - k * m * n - j * m);
      if (k>0) {
        if (j>0) {
          if (i>0)   {J = global_index(i-1,j-1,k-1,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
                      J = global_index(i,  j-1,k-1,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);
          if (i<m-1) {J = global_index(i+1,j-1,k-1,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
        }
        {
          if (i>0)   {J = global_index(i-1,j,  k-1,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
                      J = global_index(i,  j,  k-1,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);
          if (i<m-1) {J = global_index(i+1,j,  k-1,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
        }
        if (j<n-1) {
          if (i>0)   {J = global_index(i-1,j+1,k-1,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
                      J = global_index(i,  j+1,k-1,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);
          if (i<m-1) {J = global_index(i+1,j+1,k-1,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
        }
      }
      {
        if (j>0) {
          if (i>0)   {J = global_index(i-1,j-1,k  ,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
                      J = global_index(i,  j-1,k  ,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);
          if (i<m-1) {J = global_index(i+1,j-1,k  ,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
        }
        {
          if (i>0)   {J = global_index(i-1,j,  k  ,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
                      J = global_index(i,  j,  k  ,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);
          if (i<m-1) {J = global_index(i+1,j,  k  ,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
        }
        if (j<n-1) {
          if (i>0)   {J = global_index(i-1,j+1,k  ,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
                      J = global_index(i,  j+1,k  ,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);
          if (i<m-1) {J = global_index(i+1,j+1,k  ,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
        }
      }
      if (k<o-1) {
        if (j>0) {
          if (i>0)   {J = global_index(i-1,j-1,k+1,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
                      J = global_index(i,  j-1,k+1,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);
          if (i<m-1) {J = global_index(i+1,j-1,k+1,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
        }
        {
          if (i>0)   {J = global_index(i-1,j,  k+1,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
                      J = global_index(i,  j,  k+1,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);
          if (i<m-1) {J = global_index(i+1,j,  k+1,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
        }
        if (j<n-1) {
          if (i>0)   {J = global_index(i-1,j+1,k+1,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
                      J = global_index(i,  j+1,k+1,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);
          if (i<m-1) {J = global_index(i+1,j+1,k+1,m,n); ierr = MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
        }
      }
      v = 26.0; ierr = MatSetValues(A,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Copy A into B in order to have a more representative benchmark (A*A has more cache hits than A*B) */
  ierr = MatDuplicate(A,MAT_COPY_VALUES,&B);CHKERRQ(ierr);

  ierr = PetscLogStageRegister("Full MatMatMult",&fullMatMatMultStage);CHKERRQ(ierr);

  /* Test C = A*B */
  ierr = PetscLogStagePush(fullMatMatMultStage);CHKERRQ(ierr);
  ierr = MatMatMult(A,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&C);CHKERRQ(ierr);

  /* Test PtAP_squared = PtAP(C,C)*PtAP(C,C)  */
  ierr = MatPtAP(C,C,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&PtAP);CHKERRQ(ierr);
  ierr = MatDuplicate(PtAP,MAT_COPY_VALUES,&PtAP_copy);CHKERRQ(ierr);
  ierr = MatMatMult(PtAP,PtAP_copy,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&PtAP_squared);CHKERRQ(ierr);

  ierr = MatView(C,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MatView(PtAP_squared,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = MatDestroy(&PtAP_squared);CHKERRQ(ierr);
  ierr = MatDestroy(&PtAP_copy);CHKERRQ(ierr);
  ierr = MatDestroy(&PtAP);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

 test:
      suffix: 1
      nsize: 1
      args: -m 8 -n 8 -stencil 2d5point -matmatmult_via sorted

 test:
       suffix: 2
       nsize: 1
       args: -m 5 -n 5 -o 5 -stencil 3d27point -matmatmult_via rowmerge

 test:
      suffix: 3
      nsize: 4
      args: -m 6 -n 6 -stencil 2d5point -matmatmult_via seqmpi

TEST*/
