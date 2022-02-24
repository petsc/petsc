
static char help[] = "Tests reusing MPI parallel matrices and MatGetValues().\n\
To test the parallel matrix assembly, this example intentionally lays out\n\
the matrix across processors differently from the way it is assembled.\n\
This example uses bilinear elements on the unit square.  Input arguments are:\n\
  -m <size> : problem size\n\n";

#include <petscmat.h>

int FormElementStiffness(PetscReal H,PetscScalar *Ke)
{
  PetscFunctionBegin;
  Ke[0]  = H/6.0;    Ke[1]  = -.125*H; Ke[2]  = H/12.0;   Ke[3]  = -.125*H;
  Ke[4]  = -.125*H;  Ke[5]  = H/6.0;   Ke[6]  = -.125*H;  Ke[7]  = H/12.0;
  Ke[8]  = H/12.0;   Ke[9]  = -.125*H; Ke[10] = H/6.0;    Ke[11] = -.125*H;
  Ke[12] = -.125*H;  Ke[13] = H/12.0;  Ke[14] = -.125*H;  Ke[15] = H/6.0;
  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
  Mat            C;
  Vec            u,b;
  PetscErrorCode ierr;
  PetscMPIInt    size,rank;
  PetscInt       i,m = 5,N,start,end,M,idx[4];
  PetscInt       j,nrsub,ncsub,*rsub,*csub,mystart,myend;
  PetscBool      flg;
  PetscScalar    one = 1.0,Ke[16],*vals;
  PetscReal      h,norm;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));

  N    = (m+1)*(m+1); /* dimension of matrix */
  M    = m*m;      /* number of elements */
  h    = 1.0/m;    /* mesh width */
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  /* Create stiffness matrix */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&C));
  CHKERRQ(MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,N,N));
  CHKERRQ(MatSetFromOptions(C));
  CHKERRQ(MatSetUp(C));

  start = rank*(M/size) + ((M%size) < rank ? (M%size) : rank);
  end   = start + M/size + ((M%size) > rank);

  /* Form the element stiffness for the Laplacian */
  CHKERRQ(FormElementStiffness(h*h,Ke));
  for (i=start; i<end; i++) {
    /* location of lower left corner of element */
    /* node numbers for the four corners of element */
    idx[0] = (m+1)*(i/m) + (i % m);
    idx[1] = idx[0]+1; idx[2] = idx[1] + m + 1; idx[3] = idx[2] - 1;
    CHKERRQ(MatSetValues(C,4,idx,4,idx,Ke,ADD_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));

  /* Assemble the matrix again */
  CHKERRQ(MatZeroEntries(C));

  for (i=start; i<end; i++) {
    /* location of lower left corner of element */
    /* node numbers for the four corners of element */
    idx[0] = (m+1)*(i/m) + (i % m);
    idx[1] = idx[0]+1; idx[2] = idx[1] + m + 1; idx[3] = idx[2] - 1;
    CHKERRQ(MatSetValues(C,4,idx,4,idx,Ke,ADD_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));

  /* Create test vectors */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&u));
  CHKERRQ(VecSetSizes(u,PETSC_DECIDE,N));
  CHKERRQ(VecSetFromOptions(u));
  CHKERRQ(VecDuplicate(u,&b));
  CHKERRQ(VecSet(u,one));

  /* Check error */
  CHKERRQ(MatMult(C,u,b));
  CHKERRQ(VecNorm(b,NORM_2,&norm));
  if (norm > PETSC_SQRT_MACHINE_EPSILON) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Norm of error b %g should be near 0\n",(double)norm));
  }

  /* Now test MatGetValues() */
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-get_values",&flg));
  if (flg) {
    CHKERRQ(MatGetOwnershipRange(C,&mystart,&myend));
    nrsub = myend - mystart; ncsub = 4;
    CHKERRQ(PetscMalloc1(nrsub*ncsub,&vals));
    CHKERRQ(PetscMalloc1(nrsub,&rsub));
    CHKERRQ(PetscMalloc1(ncsub,&csub));
    for (i=myend-1; i>=mystart; i--) rsub[myend-i-1] = i;
    for (i=0; i<ncsub; i++) csub[i] = 2*(ncsub-i) + mystart;
    CHKERRQ(MatGetValues(C,nrsub,rsub,ncsub,csub,vals));
    CHKERRQ(MatView(C,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"processor number %d: start=%" PetscInt_FMT ", end=%" PetscInt_FMT ", mystart=%" PetscInt_FMT ", myend=%" PetscInt_FMT "\n",rank,start,end,mystart,myend));
    for (i=0; i<nrsub; i++) {
      for (j=0; j<ncsub; j++) {
        if (PetscImaginaryPart(vals[i*ncsub+j]) != 0.0) {
          CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"  C[%" PetscInt_FMT ", %" PetscInt_FMT "] = %g + %g i\n",rsub[i],csub[j],(double)PetscRealPart(vals[i*ncsub+j]),(double)PetscImaginaryPart(vals[i*ncsub+j])));
        } else {
          CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"  C[%" PetscInt_FMT ", %" PetscInt_FMT "] = %g\n",rsub[i],csub[j],(double)PetscRealPart(vals[i*ncsub+j])));
        }
      }
    }
    CHKERRQ(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT));
    CHKERRQ(PetscFree(rsub));
    CHKERRQ(PetscFree(csub));
    CHKERRQ(PetscFree(vals));
  }

  /* Free data structures */
  CHKERRQ(VecDestroy(&u));
  CHKERRQ(VecDestroy(&b));
  CHKERRQ(MatDestroy(&C));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: 4

TEST*/
