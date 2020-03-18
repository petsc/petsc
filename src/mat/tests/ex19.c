
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
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);

  N    = (m+1)*(m+1); /* dimension of matrix */
  M    = m*m;      /* number of elements */
  h    = 1.0/m;    /* mesh width */
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);

  /* Create stiffness matrix */
  ierr = MatCreate(PETSC_COMM_WORLD,&C);CHKERRQ(ierr);
  ierr = MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,N,N);CHKERRQ(ierr);
  ierr = MatSetFromOptions(C);CHKERRQ(ierr);
  ierr = MatSetUp(C);CHKERRQ(ierr);

  start = rank*(M/size) + ((M%size) < rank ? (M%size) : rank);
  end   = start + M/size + ((M%size) > rank);

  /* Form the element stiffness for the Laplacian */
  ierr = FormElementStiffness(h*h,Ke);CHKERRQ(ierr);
  for (i=start; i<end; i++) {
    /* location of lower left corner of element */
    /* node numbers for the four corners of element */
    idx[0] = (m+1)*(i/m) + (i % m);
    idx[1] = idx[0]+1; idx[2] = idx[1] + m + 1; idx[3] = idx[2] - 1;
    ierr   = MatSetValues(C,4,idx,4,idx,Ke,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Assemble the matrix again */
  ierr = MatZeroEntries(C);CHKERRQ(ierr);

  for (i=start; i<end; i++) {
    /* location of lower left corner of element */
    /* node numbers for the four corners of element */
    idx[0] = (m+1)*(i/m) + (i % m);
    idx[1] = idx[0]+1; idx[2] = idx[1] + m + 1; idx[3] = idx[2] - 1;
    ierr   = MatSetValues(C,4,idx,4,idx,Ke,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Create test vectors */
  ierr = VecCreate(PETSC_COMM_WORLD,&u);CHKERRQ(ierr);
  ierr = VecSetSizes(u,PETSC_DECIDE,N);CHKERRQ(ierr);
  ierr = VecSetFromOptions(u);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&b);CHKERRQ(ierr);
  ierr = VecSet(u,one);CHKERRQ(ierr);

  /* Check error */
  ierr = MatMult(C,u,b);CHKERRQ(ierr);
  ierr = VecNorm(b,NORM_2,&norm);CHKERRQ(ierr);
  if (norm > PETSC_SQRT_MACHINE_EPSILON) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error b %g should be near 0\n",(double)norm);CHKERRQ(ierr);
  }

  /* Now test MatGetValues() */
  ierr = PetscOptionsHasName(NULL,NULL,"-get_values",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr  = MatGetOwnershipRange(C,&mystart,&myend);CHKERRQ(ierr);
    nrsub = myend - mystart; ncsub = 4;
    ierr  = PetscMalloc1(nrsub*ncsub,&vals);CHKERRQ(ierr);
    ierr  = PetscMalloc1(nrsub,&rsub);CHKERRQ(ierr);
    ierr  = PetscMalloc1(ncsub,&csub);CHKERRQ(ierr);
    for (i=myend-1; i>=mystart; i--) rsub[myend-i-1] = i;
    for (i=0; i<ncsub; i++) csub[i] = 2*(ncsub-i) + mystart;
    ierr = MatGetValues(C,nrsub,rsub,ncsub,csub,vals);CHKERRQ(ierr);
    ierr = MatView(C,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"processor number %d: start=%D, end=%D, mystart=%D, myend=%D\n",rank,start,end,mystart,myend);CHKERRQ(ierr);
    for (i=0; i<nrsub; i++) {
      for (j=0; j<ncsub; j++) {
        if (PetscImaginaryPart(vals[i*ncsub+j]) != 0.0) {
          ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"  C[%D, %D] = %g + %g i\n",rsub[i],csub[j],(double)PetscRealPart(vals[i*ncsub+j]),(double)PetscImaginaryPart(vals[i*ncsub+j]));CHKERRQ(ierr);
        } else {
          ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"  C[%D, %D] = %g\n",rsub[i],csub[j],(double)PetscRealPart(vals[i*ncsub+j]));CHKERRQ(ierr);
        }
      }
    }
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);CHKERRQ(ierr);
    ierr = PetscFree(rsub);CHKERRQ(ierr);
    ierr = PetscFree(csub);CHKERRQ(ierr);
    ierr = PetscFree(vals);CHKERRQ(ierr);
  }

  /* Free data structures */
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}




/*TEST

   test:
      nsize: 4

TEST*/
