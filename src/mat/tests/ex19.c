
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
  PetscMPIInt    size,rank;
  PetscInt       i,m = 5,N,start,end,M,idx[4];
  PetscInt       j,nrsub,ncsub,*rsub,*csub,mystart,myend;
  PetscBool      flg;
  PetscScalar    one = 1.0,Ke[16],*vals;
  PetscReal      h,norm;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));

  N    = (m+1)*(m+1); /* dimension of matrix */
  M    = m*m;      /* number of elements */
  h    = 1.0/m;    /* mesh width */
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  /* Create stiffness matrix */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&C));
  PetscCall(MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,N,N));
  PetscCall(MatSetFromOptions(C));
  PetscCall(MatSetUp(C));

  start = rank*(M/size) + ((M%size) < rank ? (M%size) : rank);
  end   = start + M/size + ((M%size) > rank);

  /* Form the element stiffness for the Laplacian */
  PetscCall(FormElementStiffness(h*h,Ke));
  for (i=start; i<end; i++) {
    /* location of lower left corner of element */
    /* node numbers for the four corners of element */
    idx[0] = (m+1)*(i/m) + (i % m);
    idx[1] = idx[0]+1; idx[2] = idx[1] + m + 1; idx[3] = idx[2] - 1;
    PetscCall(MatSetValues(C,4,idx,4,idx,Ke,ADD_VALUES));
  }
  PetscCall(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));

  /* Assemble the matrix again */
  PetscCall(MatZeroEntries(C));

  for (i=start; i<end; i++) {
    /* location of lower left corner of element */
    /* node numbers for the four corners of element */
    idx[0] = (m+1)*(i/m) + (i % m);
    idx[1] = idx[0]+1; idx[2] = idx[1] + m + 1; idx[3] = idx[2] - 1;
    PetscCall(MatSetValues(C,4,idx,4,idx,Ke,ADD_VALUES));
  }
  PetscCall(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));

  /* Create test vectors */
  PetscCall(VecCreate(PETSC_COMM_WORLD,&u));
  PetscCall(VecSetSizes(u,PETSC_DECIDE,N));
  PetscCall(VecSetFromOptions(u));
  PetscCall(VecDuplicate(u,&b));
  PetscCall(VecSet(u,one));

  /* Check error */
  PetscCall(MatMult(C,u,b));
  PetscCall(VecNorm(b,NORM_2,&norm));
  if (norm > PETSC_SQRT_MACHINE_EPSILON) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Norm of error b %g should be near 0\n",(double)norm));
  }

  /* Now test MatGetValues() */
  PetscCall(PetscOptionsHasName(NULL,NULL,"-get_values",&flg));
  if (flg) {
    PetscCall(MatGetOwnershipRange(C,&mystart,&myend));
    nrsub = myend - mystart; ncsub = 4;
    PetscCall(PetscMalloc1(nrsub*ncsub,&vals));
    PetscCall(PetscMalloc1(nrsub,&rsub));
    PetscCall(PetscMalloc1(ncsub,&csub));
    for (i=myend-1; i>=mystart; i--) rsub[myend-i-1] = i;
    for (i=0; i<ncsub; i++) csub[i] = 2*(ncsub-i) + mystart;
    PetscCall(MatGetValues(C,nrsub,rsub,ncsub,csub,vals));
    PetscCall(MatView(C,PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"processor number %d: start=%" PetscInt_FMT ", end=%" PetscInt_FMT ", mystart=%" PetscInt_FMT ", myend=%" PetscInt_FMT "\n",rank,start,end,mystart,myend));
    for (i=0; i<nrsub; i++) {
      for (j=0; j<ncsub; j++) {
        if (PetscImaginaryPart(vals[i*ncsub+j]) != 0.0) {
          PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"  C[%" PetscInt_FMT ", %" PetscInt_FMT "] = %g + %g i\n",rsub[i],csub[j],(double)PetscRealPart(vals[i*ncsub+j]),(double)PetscImaginaryPart(vals[i*ncsub+j])));
        } else {
          PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"  C[%" PetscInt_FMT ", %" PetscInt_FMT "] = %g\n",rsub[i],csub[j],(double)PetscRealPart(vals[i*ncsub+j])));
        }
      }
    }
    PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT));
    PetscCall(PetscFree(rsub));
    PetscCall(PetscFree(csub));
    PetscCall(PetscFree(vals));
  }

  /* Free data structures */
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&C));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 4

TEST*/
