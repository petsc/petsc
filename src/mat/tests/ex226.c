static char help[] = "Benchmark for MatMatMult() of AIJ matrices using different 2d finite-difference stencils.\n\n";

#include <petscmat.h>

/* Converts 3d grid coordinates (i,j,k) for a grid of size m \times n to global indexing. Pass k = 0 for a 2d grid. */
int global_index(PetscInt i,PetscInt j,PetscInt k, PetscInt m, PetscInt n) { return i + j * m + k * m * n; }

int main(int argc,char **argv)
{
  Mat            A,B,C,PtAP,PtAP_copy,PtAP_squared;
  PetscInt       i,M,N,Istart,Iend,n=7,j,J,Ii,m=8,k,o=1;
  PetscScalar    v;
  PetscBool      equal=PETSC_FALSE,mat_view=PETSC_FALSE;
  char           stencil[PETSC_MAX_PATH_LEN];
#if defined(PETSC_USE_LOG)
  PetscLogStage  fullMatMatMultStage;
#endif

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-o",&o,NULL));
  PetscCall(PetscOptionsHasName(NULL,NULL,"-result_view",&mat_view));
  PetscCall(PetscOptionsGetString(NULL,NULL,"-stencil",stencil,sizeof(stencil),NULL));

  /* Create a aij matrix A */
  M    = N = m*n*o;
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,M,N));
  PetscCall(MatSetType(A,MATAIJ));
  PetscCall(MatSetFromOptions(A));

  /* Consistency checks */
  PetscCheck(o >= 1 && m > 1 && n >= 1,PETSC_COMM_WORLD,PETSC_ERR_USER,"Dimensions need to be larger than zero!");

  /************ 2D stencils ***************/
  PetscCall(PetscStrcmp(stencil, "2d5point", &equal));
  if (equal) {   /* 5-point stencil, 2D */
    PetscCall(MatMPIAIJSetPreallocation(A,5,NULL,5,NULL));
    PetscCall(MatSeqAIJSetPreallocation(A,5,NULL));
    PetscCall(MatGetOwnershipRange(A,&Istart,&Iend));
    for (Ii=Istart; Ii<Iend; Ii++) {
      v = -1.0; k = Ii / (m*n); j = (Ii - k * m * n) / m; i = (Ii - k * m * n - j * m);
      if (i>0)   {J = global_index(i-1,j,k,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (i<m-1) {J = global_index(i+1,j,k,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (j>0)   {J = global_index(i,j-1,k,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (j<n-1) {J = global_index(i,j+1,k,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      v = 4.0; PetscCall(MatSetValues(A,1,&Ii,1,&Ii,&v,INSERT_VALUES));
    }
  }
  PetscCall(PetscStrcmp(stencil, "2d9point", &equal));
  if (equal) {      /* 9-point stencil, 2D */
    PetscCall(MatMPIAIJSetPreallocation(A,9,NULL,9,NULL));
    PetscCall(MatSeqAIJSetPreallocation(A,9,NULL));
    PetscCall(MatGetOwnershipRange(A,&Istart,&Iend));
    for (Ii=Istart; Ii<Iend; Ii++) {
      v = -1.0; k = Ii / (m*n); j = (Ii - k * m * n) / m; i = (Ii - k * m * n - j * m);
      if (i>0)            {J = global_index(i-1,j,  k,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (i>0 && j>0)   {J = global_index(i-1,j-1,k,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (j>0)            {J = global_index(i,  j-1,k,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (i<m-1 && j>0)   {J = global_index(i+1,j-1,k,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (i<m-1)          {J = global_index(i+1,j,  k,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (i<m-1 && j<n-1) {J = global_index(i+1,j+1,k,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (j<n-1)          {J = global_index(i,  j+1,k,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (i>0 && j<n-1) {J = global_index(i-1,j+1,k,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      v = 8.0; PetscCall(MatSetValues(A,1,&Ii,1,&Ii,&v,INSERT_VALUES));
    }
  }
  PetscCall(PetscStrcmp(stencil, "2d9point2", &equal));
  if (equal) {      /* 9-point Cartesian stencil (width 2 per coordinate), 2D */
    PetscCall(MatMPIAIJSetPreallocation(A,9,NULL,9,NULL));
    PetscCall(MatSeqAIJSetPreallocation(A,9,NULL));
    PetscCall(MatGetOwnershipRange(A,&Istart,&Iend));
    for (Ii=Istart; Ii<Iend; Ii++) {
      v = -1.0; k = Ii / (m*n); j = (Ii - k * m * n) / m; i = (Ii - k * m * n - j * m);
      if (i>0)   {J = global_index(i-1,j,k,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (i>1)   {J = global_index(i-2,j,k,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (i<m-1) {J = global_index(i+1,j,k,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (i<m-2) {J = global_index(i+2,j,k,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (j>0)   {J = global_index(i,j-1,k,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (j>1)   {J = global_index(i,j-2,k,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (j<n-1) {J = global_index(i,j+1,k,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (j<n-2) {J = global_index(i,j+2,k,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      v = 8.0; PetscCall(MatSetValues(A,1,&Ii,1,&Ii,&v,INSERT_VALUES));
    }
  }
  PetscCall(PetscStrcmp(stencil, "2d13point", &equal));
  if (equal) {      /* 13-point Cartesian stencil (width 3 per coordinate), 2D */
    PetscCall(MatMPIAIJSetPreallocation(A,13,NULL,13,NULL));
    PetscCall(MatSeqAIJSetPreallocation(A,13,NULL));
    PetscCall(MatGetOwnershipRange(A,&Istart,&Iend));
    for (Ii=Istart; Ii<Iend; Ii++) {
      v = -1.0; k = Ii / (m*n); j = (Ii - k * m * n) / m; i = (Ii - k * m * n - j * m);
      if (i>0)   {J = global_index(i-1,j,k,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (i>1)   {J = global_index(i-2,j,k,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (i>2)   {J = global_index(i-3,j,k,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (i<m-1) {J = global_index(i+1,j,k,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (i<m-2) {J = global_index(i+2,j,k,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (i<m-3) {J = global_index(i+3,j,k,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (j>0)   {J = global_index(i,j-1,k,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (j>1)   {J = global_index(i,j-2,k,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (j>2)   {J = global_index(i,j-3,k,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (j<n-1) {J = global_index(i,j+1,k,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (j<n-2) {J = global_index(i,j+2,k,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (j<n-3) {J = global_index(i,j+3,k,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      v = 12.0; PetscCall(MatSetValues(A,1,&Ii,1,&Ii,&v,INSERT_VALUES));
    }
  }
  /************ 3D stencils ***************/
  PetscCall(PetscStrcmp(stencil, "3d7point", &equal));
  if (equal) {      /* 7-point stencil, 3D */
    PetscCall(MatMPIAIJSetPreallocation(A,7,NULL,7,NULL));
    PetscCall(MatSeqAIJSetPreallocation(A,7,NULL));
    PetscCall(MatGetOwnershipRange(A,&Istart,&Iend));
    for (Ii=Istart; Ii<Iend; Ii++) {
      v = -1.0; k = Ii / (m*n); j = (Ii - k * m * n) / m; i = (Ii - k * m * n - j * m);
      if (i>0)   {J = global_index(i-1,j,k,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (i<m-1) {J = global_index(i+1,j,k,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (j>0)   {J = global_index(i,j-1,k,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (j<n-1) {J = global_index(i,j+1,k,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (k>0)   {J = global_index(i,j,k-1,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (k<o-1) {J = global_index(i,j,k+1,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      v = 6.0; PetscCall(MatSetValues(A,1,&Ii,1,&Ii,&v,INSERT_VALUES));
    }
  }
  PetscCall(PetscStrcmp(stencil, "3d13point", &equal));
  if (equal) {      /* 13-point stencil, 3D */
    PetscCall(MatMPIAIJSetPreallocation(A,13,NULL,13,NULL));
    PetscCall(MatSeqAIJSetPreallocation(A,13,NULL));
    PetscCall(MatGetOwnershipRange(A,&Istart,&Iend));
    for (Ii=Istart; Ii<Iend; Ii++) {
      v = -1.0; k = Ii / (m*n); j = (Ii - k * m * n) / m; i = (Ii - k * m * n - j * m);
      if (i>0)   {J = global_index(i-1,j,k,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (i>1)   {J = global_index(i-2,j,k,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (i<m-1) {J = global_index(i+1,j,k,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (i<m-2) {J = global_index(i+2,j,k,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (j>0)   {J = global_index(i,j-1,k,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (j>1)   {J = global_index(i,j-2,k,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (j<n-1) {J = global_index(i,j+1,k,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (j<n-2) {J = global_index(i,j+2,k,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (k>0)   {J = global_index(i,j,k-1,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (k>1)   {J = global_index(i,j,k-2,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (k<o-1) {J = global_index(i,j,k+1,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (k<o-2) {J = global_index(i,j,k+2,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      v = 12.0; PetscCall(MatSetValues(A,1,&Ii,1,&Ii,&v,INSERT_VALUES));
    }
  }
  PetscCall(PetscStrcmp(stencil, "3d19point", &equal));
  if (equal) {      /* 19-point stencil, 3D */
    PetscCall(MatMPIAIJSetPreallocation(A,19,NULL,19,NULL));
    PetscCall(MatSeqAIJSetPreallocation(A,19,NULL));
    PetscCall(MatGetOwnershipRange(A,&Istart,&Iend));
    for (Ii=Istart; Ii<Iend; Ii++) {
      v = -1.0; k = Ii / (m*n); j = (Ii - k * m * n) / m; i = (Ii - k * m * n - j * m);
      /* one hop */
      if (i>0)   {J = global_index(i-1,j,k,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (i<m-1) {J = global_index(i+1,j,k,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (j>0)   {J = global_index(i,j-1,k,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (j<n-1) {J = global_index(i,j+1,k,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (k>0)   {J = global_index(i,j,k-1,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (k<o-1) {J = global_index(i,j,k+1,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      /* two hops */
      if (i>0   && j>0)   {J = global_index(i-1,j-1,k  ,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (i>0   && k>0)   {J = global_index(i-1,j,  k-1,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (i>0   && j<n-1) {J = global_index(i-1,j+1,k  ,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (i>0   && k<o-1) {J = global_index(i-1,j,  k+1,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (i<m-1 && j>0)   {J = global_index(i+1,j-1,k  ,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (i<m-1 && k>0)   {J = global_index(i+1,j,  k-1,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (i<m-1 && j<n-1) {J = global_index(i+1,j+1,k  ,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (i<m-1 && k<o-1) {J = global_index(i+1,j,  k+1,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (j>0   && k>0)   {J = global_index(i,  j-1,k-1,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (j>0   && k<o-1) {J = global_index(i,  j-1,k+1,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (j<n-1 && k>0)   {J = global_index(i,  j+1,k-1,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (j<n-1 && k<o-1) {J = global_index(i,  j+1,k+1,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
      v = 18.0; PetscCall(MatSetValues(A,1,&Ii,1,&Ii,&v,INSERT_VALUES));
    }
  }
  PetscCall(PetscStrcmp(stencil, "3d27point", &equal));
  if (equal) {      /* 27-point stencil, 3D */
    PetscCall(MatMPIAIJSetPreallocation(A,27,NULL,27,NULL));
    PetscCall(MatSeqAIJSetPreallocation(A,27,NULL));
    PetscCall(MatGetOwnershipRange(A,&Istart,&Iend));
    for (Ii=Istart; Ii<Iend; Ii++) {
      v = -1.0; k = Ii / (m*n); j = (Ii - k * m * n) / m; i = (Ii - k * m * n - j * m);
      if (k>0) {
        if (j>0) {
          if (i>0)   {J = global_index(i-1,j-1,k-1,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
                      J = global_index(i,  j-1,k-1,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));
          if (i<m-1) {J = global_index(i+1,j-1,k-1,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
        }
        {
          if (i>0)   {J = global_index(i-1,j,  k-1,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
                      J = global_index(i,  j,  k-1,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));
          if (i<m-1) {J = global_index(i+1,j,  k-1,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
        }
        if (j<n-1) {
          if (i>0)   {J = global_index(i-1,j+1,k-1,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
                      J = global_index(i,  j+1,k-1,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));
          if (i<m-1) {J = global_index(i+1,j+1,k-1,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
        }
      }
      {
        if (j>0) {
          if (i>0)   {J = global_index(i-1,j-1,k  ,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
                      J = global_index(i,  j-1,k  ,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));
          if (i<m-1) {J = global_index(i+1,j-1,k  ,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
        }
        {
          if (i>0)   {J = global_index(i-1,j,  k  ,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
                      J = global_index(i,  j,  k  ,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));
          if (i<m-1) {J = global_index(i+1,j,  k  ,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
        }
        if (j<n-1) {
          if (i>0)   {J = global_index(i-1,j+1,k  ,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
                      J = global_index(i,  j+1,k  ,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));
          if (i<m-1) {J = global_index(i+1,j+1,k  ,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
        }
      }
      if (k<o-1) {
        if (j>0) {
          if (i>0)   {J = global_index(i-1,j-1,k+1,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
                      J = global_index(i,  j-1,k+1,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));
          if (i<m-1) {J = global_index(i+1,j-1,k+1,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
        }
        {
          if (i>0)   {J = global_index(i-1,j,  k+1,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
                      J = global_index(i,  j,  k+1,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));
          if (i<m-1) {J = global_index(i+1,j,  k+1,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
        }
        if (j<n-1) {
          if (i>0)   {J = global_index(i-1,j+1,k+1,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
                      J = global_index(i,  j+1,k+1,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));
          if (i<m-1) {J = global_index(i+1,j+1,k+1,m,n); PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,INSERT_VALUES));}
        }
      }
      v = 26.0; PetscCall(MatSetValues(A,1,&Ii,1,&Ii,&v,INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* Copy A into B in order to have a more representative benchmark (A*A has more cache hits than A*B) */
  PetscCall(MatDuplicate(A,MAT_COPY_VALUES,&B));

  PetscCall(PetscLogStageRegister("Full MatMatMult",&fullMatMatMultStage));

  /* Test C = A*B */
  PetscCall(PetscLogStagePush(fullMatMatMultStage));
  PetscCall(MatMatMult(A,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&C));

  /* Test PtAP_squared = PtAP(C,C)*PtAP(C,C)  */
  PetscCall(MatPtAP(C,C,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&PtAP));
  PetscCall(MatDuplicate(PtAP,MAT_COPY_VALUES,&PtAP_copy));
  PetscCall(MatMatMult(PtAP,PtAP_copy,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&PtAP_squared));

  PetscCall(MatView(C,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(MatView(PtAP_squared,PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(MatDestroy(&PtAP_squared));
  PetscCall(MatDestroy(&PtAP_copy));
  PetscCall(MatDestroy(&PtAP));
  PetscCall(MatDestroy(&C));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
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
