#include "src/mat/impls/feti/feti.h"
#include "src/sles/pc/impls/is/feti/fetipc.h"

/*#include "mpi.h"*/
/* src/sles/pc/impls/is/feti/ex1.c */

/* Example File for FETI; assembles stiffness matrices */
/* make BOPT=g_c++ ex1 
   mpirun -np 5 ./ex1  */

#define PRINT(A) PetscPrintf(PETSC_COMM_WORLD,"debug:%d\n",(A))

static char help[]="";


/* Was to be a fast hack for a local stiffness matrix for FETI, but will not be used for now
   since I prefer to read the matrices in from a file */

int build_K(MPI_Comm comm, const int n, const int N, const int domain_num)
{ /* a fast hack... builds a stiffness-matrix for the laplacian using trilinear elements in a cube */
    int ierr;
    PetscReal H=1./N;
    PetscReal h=H/(n-1);
    

    int n_elem=(n-1)*(n-1)*(n-1);
    int n_dofs=n*n*n;

    int * dof_table;
    
    PetscFunctionBegin;  /* traceback stack push */

    PetscMalloc(n_dofs*8*sizeof(int),&dof_table); /* does logging, alignment and malloc */

    for(int i=1;i<n;i++)
    {
	for(int j=1;j<n;j++)
	{
	    for(int k=1;k<n;k++)
	    {
		int base=((i-1)*(n-1)*(n-1) + (j-1)*(n-1) + (k-1))*8 + 1;
		dof_table[base]  =(i-1)*n*n+(j-1)*n+k  ;    /* bottom left  1 */
		dof_table[base+1]=(i-1)*n*n+(j-1)*n+k+1;    /* bottom right 2 */
		dof_table[base+2]=(i-1)*n*n+    j*n+k  ;    /*              4 */ 
		dof_table[base+3]=(i-1)*n*n+    j*n+k+1;    /*              5 */
		dof_table[base+4]=    i*n*n+(j-1)*n+k  ;    /*             10 */
		dof_table[base+5]=    i*n*n+(j-1)*n+k+1;    /*             11 */
		dof_table[base+6]=    i*n*n+    j*n+k  ;    /*             13 */
		dof_table[base+7]=    i*n*n+    j*n+k+1;    /*             14 */
	    }
	}
    }

    PetscPrintf(PETSC_COMM_WORLD,"size:%d\n",sizeof(int));

    double t[9][9]={{}}; /* integration table hard coded */ 


    t[1][1]=t[2][2]=t[3][3]=t[4][4]=t[5][5]=t[6][6]=t[7][7]=t[8][8]=(1./3)*h;       /* self */
    t[1][8]=t[2][7]=t[3][6]=t[4][5]=t[8][1]=t[7][2]=t[6][3]=t[5][4]=-(1./12)*h;     /* diagonal */
    t[1][4]=t[1][6]=t[1][7]=t[2][3]=t[2][5]=t[2][8]= 
	t[3][2]=t[3][5]=t[3][8]=t[4][1]=t[4][6]=t[4][7]= 
	t[5][2]=t[5][3]=t[5][8]=t[6][1]=t[6][4]=t[6][7]= 
	t[7][1]=t[7][4]=t[7][6]=t[8][2]=t[8][3]=t[8][5]= -(1./12)*h;                /* on face diagonal*/

    
    Mat K;
    MatCreateSeqAIJ(PETSC_COMM_SELF,n_dofs,n_dofs,5,0,&K);  /* on one processor only */
    Vec f;
    VecCreateSeq(PETSC_COMM_SELF,n_dofs,&f);  /* VecGetArray() */ /* PETSC_COMM_SELF */

    for(int i=1;i<=n_elem;i++)   
    {
	for(int node=1;node<=8;node++)
	{
	    for(int node_to=1;node_to<=8;node_to++)
	    {

		/* MatSetValue(K,1,1,.5,ADD_VALUES); */

		MatSetValue(K, dof_table[(i-1)*8+node]-1, dof_table[(i-1)*8+node_to]-1, t[node][node_to], ADD_VALUES);
		/*  MatSetValues would be faster; PetSc matrices start at 0,0  */
	    }

            /* rhs */
            VecSetValue(f,dof_table[(i-1)*8+node-1],h*h*h/8,ADD_VALUES);

	}   
    }
    MatAssemblyBegin(K,MAT_FINAL_ASSEMBLY);  /* presumably compress */
    MatAssemblyEnd(K,MAT_FINAL_ASSEMBLY);

    // MatView(K,PETSC_VIEWER_STDOUT_SELF);
    // MatView(K,PETSC_VIEWER_DRAW_WORLD);

    PetscViewer viewer;

    PetscViewerDrawOpen(PETSC_COMM_SELF,0,"K",PETSC_DECIDE,PETSC_DECIDE,500,500,&viewer); /* PETSC_DRAW_HALF_SIZE */
    MatView(K,viewer); /* PETSC_DRAW_WORLD: first processors draws everything coming from all *


    // VecView(f,PETSC_VIEWER_STDOUT_SELF);

    PetscFree(dof_table);
    PetscFunctionReturn(0);  /* traceback stack pop */


    /* int MatGetSubMatrices(Mat mat,int n,IS *irow,IS *icol,MatReuse scall,Mat **submat) */
    /* int MatGetSubMatrix(Mat mat,IS isrow,IS iscol,int csize,MatReuse cll,Mat *newmat)  */

}



  /* Build the data-structure for B */ 
  /* In x,y,z one there can be a different number of domains */
  /* Every processor connects the domains but does only logging
     to add to the length of lambda for the domains that do not belong to it.
     Only when the domains belongs to the processor (identified by MPI-rank) 
     then B is actually built */
  /* Discontinued in favor of reading the matrices from a file */

int Prepare_B(const int M, const int N, const int O)   /* should be called on all processors */
{
    PetscFunctionBegin;
    for(int i=0; i<M; i++)
    {
	for(int j=0;j<N;j++)
	{
	    for(int k=0;k<O;k++)
	    {
		/*
		 */
	    }
	}
    }   

    PetscFunctionReturn(0);
}



Mat_Feti A;
PC_Feti P;


int main(int argc, char ** argv)
{
    int rank, ierr, ch, its;
    PetscScalar zero=0,one=1;

    PetscFunctionBegin;

    PetscInitialize(&argc,&argv, (char *)0 , help);
    PetscPrintf(PETSC_COMM_WORLD,"FETI example program\nbuilt %s using %d bit arithmetics.\n",__DATE__, sizeof(PetscReal));
    // PetscOptionsGetInt 

    ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank); CHKERRQ(ierr); 
    PetscPrintf(PETSC_COMM_WORLD,"My rank: %d\n",rank); // PetscPrintf only prints on the first processor 


    /*
    build_K(PETSC_COMM_WORLD,10,1,rank);  // now obsolete 
    printf("build_K:");                   // uses PetscDrawOpen
    scanf("%d",&ch);
    */


    /* Test MatLoad */
    
/*
    printf("Test MatLoad...");
    Mat B;
    PetscViewer viewer;
    PetscViewerBinaryOpen(PETSC_COMM_SELF, "FetiDP_Krr_1.mat", PETSC_BINARY_RDONLY, &viewer); // needs comm 
    MatLoad(viewer,MATSEQAIJ,&B); 
   
    MatView(B,PETSC_VIEWER_STDOUT_SELF);  // MatView 
    
    printf("...Testend\n");
    scanf("%d",&ch);
*/  
    
    /* End Test */


    /*
       What can we use?
       2 3 => 8 domains   with  27 dofs  |  2 Prozessoren mit je 4 Gebieten  (27 dof pro Gebiet)

       3 3 =>27 domains   with  27 dofs  |  3 Prozessoren mit je 9 Gebieten  (27 dof pro Gebiet)
    */

    /* ------------------------------------------------------- */
    Vec lambda;
    //VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE,100,&lambda);
    Mat A;
    Vec dr_Scc;

    PetscPrintf(MPI_COMM_WORLD,"Creating Mat_Feti...\n");
    MatCreateFeti(PETSC_COMM_WORLD,9,&A); // creates 1 Mat_Feti per processor containing 3 FetiDomains each 

                                          // so for tests we will look at 27 subdomains; 3 per proc => 9 proc 
                                          // mpirun -nP 2 
    PetscPrintf(MPI_COMM_WORLD,"Loading Mat_Feti...\n");
    MatLoad_Feti(A,&lambda);  // MatLoad also sets lambda;
    PetscPrintf(MPI_COMM_WORLD,"...finished\n");

    //MatSetUp_Feti(A);  // MatLoad_Feti does this already
    //MatCalculateRHS_Feti(A,dr_Scc);

#if 0
    PetscViewer dummy;
    printf("Viewing Mat_Feti...\n");
    MatView_Feti(A, dummy );
#endif

    SLES sles;
    SLESCreate(PETSC_COMM_WORLD,&sles);

    KSP ksp;
    PC pc;

    SLESGetKSP(sles,&ksp);
    SLESGetPC(sles,&pc);

    KSPSetType(ksp,KSPCG); // the matrix type does not have to be registered
    PCSetType(pc,PCNONE);  // sufficient to set ops->mult to the right function
    
    //KSPSetInitialGuessNonzero(ksp,PETSC_TRUE);

    MatCalculateRHS_Feti(A, &dr_Scc);

    SLESSetOperators(sles,A,A,DIFFERENT_NONZERO_PATTERN);

    SLESSolve(sles,dr_Scc,lambda,&its);

    PetscPrintf(PETSC_COMM_WORLD,"Iterations: %d\n",its);
    // Mat
    

    PetscPrintf(MPI_COMM_WORLD,"...finished program (%d It.)\n",its);
    MatWrite_lambda(A);

    
    //(((Mat_Feti*)A->data)->domains[0].Krr);
    MatlabInspect_Feti(A);

    scanf("%d",ch);

    // Scattertest
    Vec ltest,ltest2;
    int llen;
    VecDuplicate(lambda,&ltest);
#if 1
    VecSet(&one,ltest);
#else
    VecGetSize(ltest,&llen);
    for(int i=0;i<llen;i++)
    {
	VecSetValue(ltest,i,i,INSERT_VALUES);
    }
    VecAssemblyBegin(ltest);
    VecAssemblyEnd  (ltest);
#endif


    //MatScatterTest_Feti(A,ltest);

    VecDuplicate(ltest,&ltest2);
    MatMult_Feti(A,ltest,ltest2);
    MatlabInspectVec(ltest2);

// Probiere LU mit MPI-Matrix
    Mat S_tmp;  // Mat
    MatDuplicate( ((Mat_Feti*)(A->data))->Scc_ass ,MAT_COPY_VALUES,&S_tmp);
    SLES solver;

    SLESCreate(PETSC_COMM_WORLD,&solver);

    //KSP ksp;
    //PC pc;

    SLESGetKSP(solver,&ksp);
    SLESGetPC(solver,&pc);

    KSPSetType(ksp,KSPCG);
    PCSetType(pc,PCJACOBI);  // PCLU only 2^k, no PCILU, no PCSOR; only PCJACOBI und PCNONE
    KSPSetTolerances(ksp,1e+16,1e-15,PETSC_DEFAULT,5000); // could set one to infinity
                                                          // because max is the stopping criterium


    Vec v,w;
    VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE,S_tmp->M,&v);
    VecSetRandom(PETSC_NULL,v);
    VecDuplicate(v,&w);

    SLESSetOperators(solver,S_tmp,S_tmp,DIFFERENT_NONZERO_PATTERN);

    SLESSolve(solver,v,w,&its);
    int reason;
    KSPGetConvergedReason(ksp,(KSPConvergedReason*)&reason);
    PetscPrintf(PETSC_COMM_WORLD,"Scc_ass its: %d,%d",its,reason); // no PCLU (only 2^k), No PCILU, No PCSOR
    VecDestroy(v);
    VecDestroy(w);
// Ende



//Probiere BcT-Scatter
    //Mat *Bc=((Mat_Feti*)(A->data))->domains[0].Bc;
    {
    Mat_Feti* matfeti=(Mat_Feti*)(A->data);
    Vec v;
    int low,high;
    VecDuplicate(matfeti->fc_ass,&v);
    VecGetOwnershipRange(v,&low,&high);
    for(int i=0;i<v->n;i++)  // only local part
    {
	//VecSetValue(v,i+low,i+low+1,INSERT_VALUES);
	VecSetValue(v,i+low,1,INSERT_VALUES);
    }	
    VecAssemblyBegin(v);
    VecAssemblyEnd(v);
    //MatlabInspectVec(v);
    MatScatterBc_Feti(A,v,FETI_SCATTER_REVERSE_INSERT);
    MatlabInspectVecs(matfeti->domains[0].uc,matfeti->domains[0].domain_num);  // works
    //   I suppose the other way also works; just scatter back
    //   write a sum in Matlab
    MatScatterBc_Feti(A,v,FETI_SCATTER_FORWARD_ADD);
    MatlabInspectVec(v); // backward
    }
    


//Ende

    MatDestroy_Feti(A); 

    PetscFinalize();
    PetscFunctionReturn(0);
}
