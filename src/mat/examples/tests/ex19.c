/*$Id: ex19.c,v 1.20 2000/05/04 03:19:45 bsmith Exp balay $*/

static char help[] = "Tests reusing MPI parallel matrices and MatGetValues().\n\
To test the parallel matrix assembly, this example intentionally lays out\n\
the matrix across processors differently from the way it is assembled.\n\
This example uses bilinear elements on the unit square.  Input arguments are:\n\
  -m <size> : problem size\n\n";

#include "petscmat.h"

#undef __FUNC__
#define __FUNC__ "FormElementStiffness"
int FormElementStiffness(double H,Scalar *Ke)
{
  PetscFunctionBegin;
  Ke[0]  = H/6.0;    Ke[1]  = -.125*H; Ke[2]  = H/12.0;   Ke[3]  = -.125*H;
  Ke[4]  = -.125*H;  Ke[5]  = H/6.0;   Ke[6]  = -.125*H;  Ke[7]  = H/12.0;
  Ke[8]  = H/12.0;   Ke[9]  = -.125*H; Ke[10] = H/6.0;    Ke[11] = -.125*H;
  Ke[12] = -.125*H;  Ke[13] = H/12.0;  Ke[14] = -.125*H;  Ke[15] = H/6.0;
  PetscFunctionReturn(0);
}

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Mat        C; 
  Vec        u,b;
  int        i,m = 5,rank,size,N,start,end,M,ierr,idx[4];
  int        j,nrsub,ncsub,*rsub,*csub,mystart,myend;
  PetscTruth flg;
  Scalar     one = 1.0,Ke[16],*vals;
  double     h,norm;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = OptionsGetInt(PETSC_NULL,"-m",&m,PETSC_NULL);CHKERRA(ierr);

  N = (m+1)*(m+1); /* dimension of matrix */
  M = m*m;         /* number of elements */
  h = 1.0/m;       /* mesh width */
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRA(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRA(ierr);

  /* Create stiffness matrix */
  ierr = MatCreate(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,N,N,&C);CHKERRA(ierr);

  start = rank*(M/size) + ((M%size) < rank ? (M%size) : rank);
  end   = start + M/size + ((M%size) > rank); 

  /* Form the element stiffness for the Laplacian */
  ierr = FormElementStiffness(h*h,Ke);
  for (i=start; i<end; i++) {
     /* location of lower left corner of element */
     /* node numbers for the four corners of element */
     idx[0] = (m+1)*(i/m) + (i % m);
     idx[1] = idx[0]+1; idx[2] = idx[1] + m + 1; idx[3] = idx[2] - 1;
     ierr = MatSetValues(C,4,idx,4,idx,Ke,ADD_VALUES);CHKERRA(ierr);
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);

  /* Assemble the matrix again */
  ierr = MatZeroEntries(C);CHKERRA(ierr);

  for (i=start; i<end; i++) {
     /* location of lower left corner of element */
     /* node numbers for the four corners of element */
     idx[0] = (m+1)*(i/m) + (i % m);
     idx[1] = idx[0]+1; idx[2] = idx[1] + m + 1; idx[3] = idx[2] - 1;
     ierr = MatSetValues(C,4,idx,4,idx,Ke,ADD_VALUES);CHKERRA(ierr);
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);

  /* Create test vectors */
  ierr = VecCreate(PETSC_COMM_WORLD,PETSC_DECIDE,N,&u);CHKERRA(ierr); 
  ierr = VecSetFromOptions(u);CHKERRA(ierr);
  ierr = VecDuplicate(u,&b);CHKERRA(ierr);
  ierr = VecSet(&one,u);CHKERRA(ierr);

  /* Check error */
  ierr = MatMult(C,u,b);CHKERRA(ierr);
  ierr = VecNorm(b,NORM_2,&norm);CHKERRA(ierr);
  if (norm > 1.e-10 || norm < -1.e-10) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error b %g should be near 0\n",norm);CHKERRA(ierr);
  }

  /* Now test MatGetValues() */
  ierr = OptionsHasName(PETSC_NULL,"-get_values",&flg);CHKERRA(ierr);
  if (flg) {
    ierr = MatGetOwnershipRange(C,&mystart,&myend);CHKERRA(ierr);
    nrsub = myend - mystart; ncsub = 4;
    vals = (Scalar*)PetscMalloc(nrsub*ncsub*sizeof(Scalar));CHKPTRA(vals);
    rsub = (int*)PetscMalloc(nrsub*sizeof(int));CHKPTRA(rsub);
    csub = (int*)PetscMalloc(ncsub*sizeof(int));CHKPTRA(csub);
    for (i=myend-1; i>=mystart; i--) rsub[myend-i-1] = i;
    for (i=0; i<ncsub; i++) csub[i] = 2*(ncsub-i) + mystart;
    ierr = MatGetValues(C,nrsub,rsub,ncsub,csub,vals);CHKERRA(ierr);
    ierr = MatView(C,VIEWER_STDOUT_WORLD);CHKERRA(ierr);
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"processor number %d: start=%d, end=%d, mystart=%d, myend=%d\n",
            rank,start,end,mystart,myend);CHKERRQ(ierr);
    for (i=0; i<nrsub; i++) {
      for (j=0; j<ncsub; j++) {
#if defined(PETSC_USE_COMPLEX)
	if (PetscImaginaryPart(vals[i*ncsub+j]) != 0.0) {
           ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"  C[%d, %d] = %g + %g i\n",rsub[i],csub[j],PetscRealPart(vals[i*ncsub+j]),
                                       PetscImaginaryPart(vals[i*ncsub+j]));CHKERRQ(ierr);
	} else {
           ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"  C[%d, %d] = %g\n",rsub[i],csub[j],PetscRealPart(vals[i*ncsub+j]));CHKERRQ(ierr);
        }
#else
         ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"  C[%d, %d] = %g\n",rsub[i],csub[j],vals[i*ncsub+j]);CHKERRQ(ierr);
#endif
      }
    }
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD);CHKERRA(ierr);
    ierr = PetscFree(rsub);CHKERRA(ierr);
    ierr = PetscFree(csub);CHKERRA(ierr);
    ierr = PetscFree(vals);CHKERRA(ierr);
  }

  /* Free data structures */
  ierr = VecDestroy(u);CHKERRA(ierr);
  ierr = VecDestroy(b);CHKERRA(ierr);
  ierr = MatDestroy(C);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}


