/* Example file for Feti-DP */
/* src/sles/pc/impls/is/feti/examples/ex1.c */

#include "src/mat/impls/feti/feti.h"
#include "src/sles/pc/impls/is/feti/fetipc.h"

#include "src/mat/impls/feti/feti.c"
#include "src/sles/pc/impls/is/feti/fetipc.c"

#define PRINT(A) PetscPrintf(PETSC_COMM_WORLD,"debug:%d\n",(A))

static char help[]="\t\t\tFetiDP\nSample usage:\n mpirun -np 1 ex1 -N 27\nSome meaningful options (first is default):\n\t\t-local_pc_type\t\tlu|none|ilu|jacobi|asm\n\t\t-local_ksp_type\t\tpreonly|cg|gmres\n\t\t-ksp_type\t\tcg|gmres\n\t\t-pc_type\t\tpcfeti|none\n\t\t-pcfeti_schur_ksp_type\tpreonly|cg|gmres\n\t\t-pcfeti_schur_pc_type\tlu|ilu\n\t\t-pcfeti_lumped\t\toff|on\n\t\t-local_ksp_rtol\t\t1e-7\n\t\t-ksp_rtol\t\t1e-7\nConvergence:\n\t\t-ksp_monitor_draw\t\t-ksp_monitor\t\t-ksp_monitor_true_residual\n\t\t-compute_explicit_operator\n\nUsing:\n";

PetscReal cond(PetscReal *r, PetscReal *c, int neig, PetscReal *min,PetscReal *max)
{
    *min=1e50;
    *max=0;
    PetscReal l;
    for(int i=0;i<neig;i++)
    {

	l=sqrt(r[i]*r[i]+c[i]*c[i]);
        *min=l<*min?l:*min;
        *max=l>*max?l:*max;
    }
    return *max/ *min;
}

int main(int argc, char ** argv)
{
    int rank, size, ierr, ch, its=4711;
    int reason=0;
    const PetscScalar zero=0,one=1;
    int N=0; 

    Mat A;
    Vec lambda;         
    Vec dr_Scc;

    SLES sles;
    KSP ksp;
    PC pc;

    PetscLogDouble start_time, end_time;
    char arch[64]={};
    char str[256]={};

    PetscFunctionBegin;

    PetscInitialize(&argc,&argv, (char *)0 , help);
    PetscGetArchType(arch,64);

    PetscTruth flag,gflag;
    PetscSSEIsEnabled(PETSC_COMM_WORLD,&flag,&gflag);
    if(flag) sprintf(str,", SSE");

    PetscPrintf(PETSC_COMM_WORLD,"\t\t\tFetiDP example program\nBuilt %s (%s). Machine using %d bit arithmetics (%s%s).\n",__DATE__,__TIME__, sizeof(PetscReal),arch,str);

    PetscMemzero(str,5*sizeof(char));

    PetscOptionsGetInt(PETSC_NULL,"-N",&N,PETSC_NULL);
    if(!N)
    {
      PetscPrintf(MPI_COMM_WORLD,"Nothing to do (Specify -N <number of domains>).\n");
      PetscFunctionReturn(1);
    }
    PetscOptionsHasName(PETSC_NULL,"-ksp_compute_condition",&flag);

    ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank); CHKERRQ(ierr); 

    ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size); CHKERRQ(ierr);

    if(N%size)
	sprintf(str,"-%d",N/size+1);

    PetscPrintf(PETSC_COMM_WORLD,"Started on %d processors, %d domains: Processors will hold %d%s domains each.\n",size,N,N/size,str);
    PetscGetCPUTime(&start_time);

    /*
       What can we use as the smallest reasonable example?
       2 3 => 8 domains   with  27 dofs  |  2 processors with 4 domains each  (and e.g. 27 dof per domain)

       3 3 =>27 domains   with  27 dofs  |  3 processors with 9 domains each  (and e.g. 27 dof pro Gebiet)
    */

    /* ------------------------------------------------------- */

    SLESCreate(PETSC_COMM_WORLD,&sles);

    SLESGetKSP(sles,&ksp);
    SLESGetPC(sles,&pc);

    if(flag)
	KSPSetComputeEigenvalues(ksp,PETSC_TRUE);

    KSPSetTolerances(ksp,1e-7,1e-50,1e+5,500); 

    {
	PetscReal a,b,c;
	int n;
	KSPGetTolerances(ksp,&a,&b,&c,&n); 
	PetscPrintf(PETSC_COMM_WORLD,"Tolerances: %.0e %.0e %.0e %d\n",a,b,c,n);
    }

    KSPSetType(ksp,KSPCG); 

    PCRegister("pcfeti",0,"PCCreate_Feti",PCCreate_Feti);
    PCSetType(pc,"pcfeti");  

    PetscPrintf(MPI_COMM_WORLD,"Creating Mat_Feti... ");
    MatCreateFeti(PETSC_COMM_WORLD,N,&A); 
    PetscPrintf(MPI_COMM_WORLD,"...Creating finished. ");

    PetscPrintf(MPI_COMM_WORLD,"Loading Mat_Feti... ");
    MatLoad_Feti(A,&lambda);  

    SLESSetOperators(sles,A,A,DIFFERENT_NONZERO_PATTERN); 

    PCLoad_Feti(pc);          

    PetscPrintf(MPI_COMM_WORLD,"...Loading finished. ");

    SLESSetFromOptions(sles);  

    MatFetiCalculateRHS(A, &dr_Scc);

    PetscPrintf(PETSC_COMM_WORLD,"Starting iterations... ");
    SLESSolve(sles,dr_Scc,lambda,&its); 

    KSPGetConvergedReason(ksp,(KSPConvergedReason*)&reason);
    PetscPrintf(PETSC_COMM_WORLD,"ConvergedReason (2 is rtol): %d ",reason); 

    {
	Mat mat;
	PetscTruth flag;
	PetscOptionsHasName(PETSC_NULL,"-ksp_compute_explicit_operator",&flag);
	if(flag)
	{
	    PetscPrintf(PETSC_COMM_WORLD,"Computing explicit operator\n");
	    KSPComputeExplicitOperator(ksp,&mat);
	    MatlabInspect(mat,"explicit_operator",4711); 

	    MatDestroy(mat);

	}
    }

    VecDestroy(dr_Scc);

    PetscPrintf(MPI_COMM_WORLD,"...writing lambda to file",its);
    MatlabWrite_lambda(A);
    PetscPrintf(MPI_COMM_WORLD,". ",its);    

    PetscPrintf(PETSC_COMM_WORLD,"...Iterations: %d \n",its);

    PetscGetCPUTime(&end_time);

    {
	PetscLogDouble flops,gflops;
	PetscGetFlops(&flops);
	PetscGlobalSum(&flops,&gflops,PETSC_COMM_WORLD);  
	if(flops<1e6)
	    PetscSynchronizedPrintf(PETSC_COMM_WORLD,"       Proc. %d: %.0f flops",rank,flops);  
	else
	    PetscSynchronizedPrintf(PETSC_COMM_WORLD,"       Proc. %d: %6g Mflops",rank,flops/1e6);  
	if(!((rank+1)%3))
	    PetscSynchronizedPrintf(PETSC_COMM_WORLD,"\n");  
	PetscSynchronizedFlush (PETSC_COMM_WORLD); /* PetscSynchronizedPrintf needs flush */
    }

    MatDestroy(A); 

    if(flag)       
    {
	int maxsize=10000,n_eig=0;
	PetscScalar *real,*im;
	PetscReal max,min;
	PetscMalloc(maxsize*sizeof(PetscScalar),&real);
	PetscMalloc(maxsize*sizeof(PetscScalar),&im);
	KSPComputeEigenvalues(ksp,maxsize,real,im,&n_eig);
	PetscPrintf(PETSC_COMM_WORLD,"\nEstimated condition:%G %.3g %.3g",cond(real,im,n_eig,&max,&min),max,min);
    }

    SLESDestroy(sles); 

    PetscPrintf(MPI_COMM_WORLD,"\n...finished program (%d it).\n",its);

    VecDestroy(lambda);

    PetscOptionsHasName("","-get_total_flops",(PetscTruth*)&its); 
    PetscOptionsLeft();
    PetscFinalize();
    PetscFunctionReturn(0);

}

