/* feti.h is a matrix type that stores the Feti system matrix */ 
/* /src/mat/impls/feti/feti.h */

/* For the future:
   1. Q
   2. kspcg does not have full orthogonalisation and 
   3. cannot build the lanczos-matrix for eigenvalue-estimates
*/


#if !defined(__feti_h)
#define __feti_h

#define MATFETI "MatFeti"
#define FETIDP_PREFIX "FetiDP_"
#define FETIDP_KCC "Kcc"  /* only globally assembled needed; all processors read this */
#define FETIDP_KRC "Krc"
#define FETIDP_KRR "Krr"
#define FETIDP_BRT "BrT"
#define FETIDP_FR  "fr"
#define FETIDP_FC  "fc"   /* globally assembled needed */
#define FETIDP_RHO "rho"  /* all processors read this */
#define FETIDP_BC "Bc"


#define FETIDP_VIEWER_XDISPLAY "localhost:0.0"


#include <stdlib.h>
#define ASSERT(A,S) if(!(A)) SETERRQ1(4711,"Assertion Failed. %s\n",(S));
#define ASSERT2(A,S,x,y) if(!(A)) SETERRQ3(4711,"Assertion Failed. %s  %d  %d\n",(S),(x),(y));
#define WARN_IF(A,S) if((A)) { PetscSynchronizedPrintf(PETSC_COMM_SELF,"Warning. %s\n",(S)); PetscSynchronizedFlush(PETSC_COMM_SELF); }
// it is PetscPrintf(PETSC_COMM_WORLD,...) to let the first processor print only but it is
//       PetscSynchronizedPrintf(PETSC_COMM_SELF,...) to let all processors print to the first screen

#define Wait(A) {int d; PetscSynchronizedPrintf(PETSC_COMM_WORLD,"%s Press key.\n",(A)); PetscSleep(1); }
//PetscSequentialPhaseEnd(PETSC_COMM_WORLD,1);
//note: PetscDrawPause, 

/* Where will I put the global stuff? */

#include "src/mat/matimpl.h"  /* includes also petscmat.h */
#include "petscsles.h"
#include "src/vec/vecimpl.h"  /* otherwise complains at first use of vec->N aso. (typedef, but declaration) */ 
#include <string.h>
//#include <mpi.h>


MPI_Comm PETSC_COMM_TOWN;
//#define PETSC_COMM_TOWN PETSC_COMM_WORLD
/* Group for Scc; */

typedef enum { FETI_SCATTER_FORWARD_ADD, FETI_SCATTER_REVERSE_INSERT } FetiScatterMode;

/* -------------------------------------------  FetiDomain  -------------------------------------------- */
typedef struct { /* all sequential */

    /* ------ User provided input ----- */
    int domain_num;    /* global number of the domain */ 

          /* all sequential aij*/
    Mat Krr;           /* local stiffness matrix; */
    Mat Krc;  /* is it more efficient to store KrcT? one small reason: columns of Krc needed */ 

    Vec fr;


    PetscReal my_rho;  /* a representative for the local coefficient
                         that will be used for scaling */
                      /* for now, we have only one per domain; even though 
			 the coefficient may vary within the domain  */



    Mat BrT;      /* temporary space for BrT as matrices; read transposed so that
		     columns are more easily accessible using MatGetRow            */
                  /* Faster alternative: MatGetArray and MatGetRowIJ  == find(BrT) */



    Mat Bc;        /* global; */
                   /* the preconditioner is made aware of the Matrix type by SLESSetOperators
		      which solely calls PCSetOperators */


    /* ------ End user privided input ----- */

    SLES Krrinv;

    Vec ur;  // 1. Only temporary space for solver; but 2. primal variable which IS of interest
    PetscScalar * ur_array; // need to put the array somewhere; this is truely temporary 

    Vec ur_tmp1;  /* space for storing temporary results, same dimensions as ur */
    Vec ur_tmp2;  /* space for storing temporary results, same dimensions as ur */

//    Vec ucl_tmp;  /* uc instead; space for storing temporary results, local c */

    Vec ucg_tmp1; /* space for storing temporary results, global c = c_ass */
    Vec ucg_tmp2; /* space for storing temporary results, global c = c_ass */

    Vec uc;   /* space for storing temporary results, but with a prominent name for later use, local c, */
    PetscScalar * uc_array;  

} FetiDomain;        /* possible to get from rank and index in domains */

// MatSetType
//   calls MatRegisterAll it not called yet (calls MatRegisterDynamic
//   finds contructor for new matrix
//   destroys old
//   calls found constructor

/* -------------------------------------------  Mat_Feti  -------------------------------------------- */
typedef struct {
    // Note that PETSCHEADER(struct_ MatOps); in struct _p_Mat introduces the virtual function table Mat->ops

     /* ---- MPI-Matrices ---- */
    Mat Kcc_ass;      // Kcc_1 not needed, Kcc global is enough; all use the same (bad?)
    Vec fc_ass;       // fc_1 also not needed, only global fc

    Mat Scc_ass;      // Scc before consolidation to all processors; internal
    Vec fScc_ass;     // fScc before consolidation to all processors;internal
                      // both could be made local variables to the respective function
    Vec uc_ass_tmp;   // used by BcT-Scatter; MPI-Vector like fc_ass
    Vec uc_tmp1;   // used by BcT-Scatter; sequential like Scc
    Vec uc_tmp2;   // used by BcT-Scatter; sequential like Scc

    /* ---- Seq-Matrices ---- */
    Mat Scc;          // Scc after consolidation; now here
    SLES Sccinv;      // now here; use this 
    Vec fScc;         // also now here; use this
    


    int n_dom;        /* number of domains on this processor; equals length of domains[] */
    FetiDomain * domains;

    Vec contrib;      /* temporary space where blown up u's from the domains will be stored */
                      /* can also be seen as the local result of B*u */

    /* First Stage Scatter u ---> contrib */
    int * scatter_src;                /* first stage local scatter to Mat_Feti */
    int * scatter_src_domain;         /* first stage local scatter to Mat_Feti */
    PetscScalar * scatter_scale;      /* use:  dest[i]=scatter_scale[i]*src[scatter_src[i]] */
    // PetscScalar *scatter_scale_pc; /* for use with pc */
    int scatter_len;  /* length of scatter_* and contrib */

    /* Second Stage Scatter contriv ---> lambda */
    VecScatter Br_scatter; /* single scatter for contrib */

#if 1
    Vec BcT_contrib;      
    /* First Stage Scatter uc ---> uc_ass */
    int * BcT_scatter_src;                /* first stage local scatter to Mat_Feti */
    int * BcT_scatter_src_domain;         /* first stage local scatter to Mat_Feti */
    PetscScalar * BcT_scatter_scale;      /* always equals one */
    // PetscScalar *scatter_scale_pc; /* for use with pc */
    int BcT_scatter_len;  /* length of scatter_* and contrib */

    /* Second Stage Scatter contriv ---> lambda */
    VecScatter BcT_scatter; /* single scatter for contrib */
#endif
    /* ---------- */


    //IS global; // not stored any more after creation of Br_scatter
    //IS local;

    /* Vec lambda: I wanted to put lambda here since I found it was an integral part of the solver
        - at least the partitioning of lambda to the processor is - , since it is needed in
       the scatter; nevertheless this is not feasible since the ksp of Petsc shall be
       used which will use a "virtual MatMultiply-Method". So the creation of the vector is 
       in the responsibility of the user: An MPI-Vector over PETSC_COMM_WORLD.
       Afterwards, the creation of the Mat_IS needs this Vector!! 
       The other possibility would be having to give the vector matis->lambda to the ksp
       when solving... not a very natural thing to do */

    Vec lambda_copy; /* this stores a copy of the user contributed lambda for access
			to the partitioning information needed by the VecScatter */
                     /* should be a const object, or a rather a pointer to const in this case
			but this is not feasible in C (only way: typedef _p_Vec const* ConstVec; */
                     

} Mat_Feti; /* FetiPartition */


/* Important note about MPIMatrices: manual page 53; if PETSC_DECIDE is not used in the creation
   of the solution vector, then one must ensure manually that the partition is the same
   as the one used by the MPI-matrix 
   Memory layout is also found there. */


/* Some Forward Declarations */
int FetiDomainLoad(FetiDomain *, char const * const);
int MatCreateScatter_Feti(Mat A);
int MatSetUpTemporarySpace_Feti(Mat A);
int MatCreate_Feti(Mat A); /* internal, if this is called, some setup still has to be made by hands afterwards */
int MatSetUp_Feti(Mat A);  // setup coarse, solve local, setup scatter;
int MatMult_Feti(Mat A, Vec src_lambda, Vec dst_lambda);
int MatDestroy_Feti(Mat a);
int MatView_Feti(Mat A,PetscViewer);
int MatlabInspect(Mat, char const * const, int const);
int MatlabInspectVec(Vec);
int MatlabInspectVecs(Vec v, int dom);
int CreatePetscCommTown();  // 2 Processors for now

/* internal use */
/* has to read K_1.mat; f_1.mat; Kcc_1; Krr_1; Krc_1; BrT_1; Bc_1; rho; Future Q */
int FetiDomainLoadMatSeq(char const * const prefix, char const * const name, char const * const postfix, Mat* A)
{
    char fname[256]={};
    PetscViewer viewer;
    PetscFunctionBegin;
    strcat(fname,prefix);
    strcat(fname,name);
    strcat(fname,postfix);
    PetscViewerBinaryOpen(PETSC_COMM_SELF, fname,PETSC_BINARY_RDONLY,&viewer); /* needs comm */
    MatLoad(viewer,MATSEQAIJ,A);  // MATSEQAIJ    
    /* MATSEQAIJ must get a communicator with one processor only; so PETSC_COMM_SELF */
    PetscFunctionReturn(0);
}

int FetiDomainLoadMatMPI(char const * const prefix, char const * const name, char const * const postfix, Mat* A)
{
    PetscFunctionBegin;
    char fname[256]={0};
    PetscViewer viewer;
    strcat(fname,prefix);
    strcat(fname,name);
    strcat(fname,postfix);
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, fname,PETSC_BINARY_RDONLY,&viewer); /* needs comm */
    MatLoad(viewer,MATMPIAIJ,A);  // MATSEQAIJ    
    PetscFunctionReturn(0);
}  


/* internal use */
int FetiDomainLoadVecSeq(char const * const prefix, char const * const name, char const * const postfix, Vec* v)
{
    char fname[256]={};
    PetscViewer viewer;
    PetscFunctionBegin;
    strcat(fname,prefix);
    strcat(fname,name);
    strcat(fname,postfix);
    PetscViewerBinaryOpen(PETSC_COMM_SELF, fname,PETSC_BINARY_RDONLY,&viewer);
    VecLoad(viewer,v);    
    PetscFunctionReturn(0);
}

/* internal use */
int FetiDomainLoadVecMPI(char const * const prefix, char const * const name, char const * const postfix, Vec* v)
{
    PetscFunctionBegin;
    char fname[256]={};
    PetscViewer viewer;
    strcat(fname,prefix);
    strcat(fname,name);
    strcat(fname,postfix);
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, fname,PETSC_BINARY_RDONLY,&viewer);
    VecLoad(viewer,v);    
    PetscFunctionReturn(0);
}



/* Load an entire Mat_Feti */
/*   The parameter lambda is meant as a help to the user 
     who will typically not know the size that
     lambda will have; MatCreateFeti sets
     it for him; if the NullPointer 
     is given he must VecCreateMPI lambda himself   */
/* List of members that must be initialised by any function that replaces MatLoad_Feti:
   Kcc, fc, BrT_, lambda, Krr_, Krc_ */
int MatLoad_Feti(Mat A, Vec* lambda) /* Mat_Feti => therefore MatLoad_Feti */
{
    int llen;
    Mat_Feti * const matfeti=(Mat_Feti*)A->data;  /* that's actually me in data */
    PetscFunctionBegin;

    FetiDomainLoadMatMPI(FETIDP_PREFIX,FETIDP_KCC,".mat",&(matfeti->Kcc_ass)); // cave ! MPI-Matrix
    FetiDomainLoadVecMPI(FETIDP_PREFIX,FETIDP_FC,".mat",&(matfeti->fc_ass));


    /* take the length of lambda from the length of domains[0]->BrT */
    FetiDomainLoad(matfeti->domains,FETIDP_PREFIX);
    
    llen=matfeti->domains->BrT->N;
    
    for(int i=1;i<matfeti->n_dom;i++)
    {
	FetiDomainLoad(matfeti->domains+i,FETIDP_PREFIX);
    }

    if(matfeti->lambda_copy==0)  // was zeroed by MatCreate_Feti
    {
	VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE,llen,lambda);
	/* This is a major point for optimization; the partitioning of lambda
	   should minimize the communication and be influenced somewhat
	   by the scatters BrT */
	matfeti->lambda_copy=*lambda; /* save a copy for creation of the scatter */
    }

    MatSetUp_Feti(A);  // setup coarse, solve local, setup scatter;

    int localsize;
    VecGetLocalSize(*lambda,&localsize);

    A->M=llen; // MatMult checks for this...
    A->N=llen; //
    A->n=localsize; // again just to appease MatMult; does not apply here
    A->m=localsize; // also to appease MatMult; MatCreate stored dom_per_proc here during Creation
    // background story:
    //  The way MatMult is supposed to work with MATMPIAIG is that the matrix is distributed to the processors by rows
    //  The result vector therefore must have the same distribution (local sizes) as the matrix 
    //  whereas the distribution of the source vector does not matter (handled by communication/scatters),
    //  but has to be made known to the matrix in form of A->n; so when iterating over a vector
    //  the most convienient way would be A->m = A->n = src->n = dst->n (and of course A->M = src->N = dst->N = A-> N)
    //  (Vectors only have N's, not M's as you might think)
    PetscFunctionReturn(0);
}                     
/* it is a bit nonnatural that "Mat A" but "Vec * lambda"... */


/* Creates Mat_Feti assigning #dom_per_proc domains to each processor 
   Generally one could have different numbers of domains per processor */

/* This Function can be used by the user */
/* The generic version through MatCreate() is not yet supported */

int FetiDomainLoad(FetiDomain * domain, const char * const prefix) /* Get all matrices from files */
{
    //char prefix[]="FetiDP_";
    char num[8]={0};
    Vec rho;
    PetscScalar *rho_array;
    sprintf(num,"_%d.mat",domain->domain_num); /* domain_num should start at 1 */ 
    FetiDomainLoadMatSeq(prefix,FETIDP_KRR,num,&(domain->Krr));
    FetiDomainLoadMatSeq(prefix,FETIDP_KRC,num,&(domain->Krc));
    FetiDomainLoadVecSeq(prefix,FETIDP_FR,num,&(domain->fr));

    FetiDomainLoadMatSeq(prefix,FETIDP_BRT,num,&(domain->BrT));

    FetiDomainLoadMatSeq(prefix,FETIDP_BC,num,&(domain->Bc)); /* MatConvert to dense later*/
    

    /**/
    FetiDomainLoadVecSeq(prefix,FETIDP_RHO,".mat",&rho);
    
    VecGetArray(rho,&rho_array);
    
    domain->my_rho=rho_array[domain->domain_num-1]; /* set my own rho */
      /* *o* forgot the information about neighboring rho's; fix later */
      /* solution: will be done by analysing B; -> scatter forward or backward... */
      /* domains start at 1 */

    VecRestoreArray(rho,&rho_array);
    /* set ur to the right size */
    VecCreateSeq(PETSC_COMM_SELF,domain->Krr->M,&domain->ur); // Krr->M should be the right choice
    VecCreateSeq(PETSC_COMM_SELF,domain->Krc->N,&domain->uc); 
}



/* internal use; called by MatLoad */
/* assumes BrT to contain the connectivity information */
/* assumes that the partitioning information for the scatter can be taken from lambda_copy */
int MatCreateScatter_Feti(Mat A)         /* name may be misleading */
{
    Mat_Feti * const matfeti=(Mat_Feti*)A->data;
    /* storage for MatGetRow */
    int n_cols, *cols; /* with cubes max. 6; in the general case more */
    PetscScalar *vals; 
    MatInfo info;

    int is_len=0;        /* size of IS */
    int *is_idx_local;   /* index array for IS local  */
    int *is_idx_global;  /* index array for IS global */
    IS ISlocal;
    IS ISglobal;

    int pos=0;


    /* length of IS has to be calculated beforehand */

    for(int i=0;i<matfeti->n_dom;i++)
    {
	const FetiDomain * const domain = &(matfeti->domains[i]); /* shortcut in the absence of references in C */
	const Mat * const BrT=&(domain->BrT);
	MatGetInfo(*BrT, MAT_LOCAL, &info);
	is_len+=(int)info.nz_used;                          /* cast from PetscLogDouble (double) */
    }
    
    
    PetscMalloc(is_len*sizeof(int),&matfeti->scatter_src_domain); /* source domain number */
    PetscMalloc(is_len*sizeof(int),&matfeti->scatter_src);        /* source index */
    PetscMalloc(is_len*sizeof(PetscScalar),&matfeti->scatter_scale); /* scaling */

    PetscMalloc(is_len*sizeof(int),&is_idx_local);      /* for scatter to lambda */
    PetscMalloc(is_len*sizeof(int),&is_idx_global);     /* for scatter to lambda */

    VecCreateSeq(PETSC_COMM_SELF, is_len, &matfeti->contrib);


    /* in case by any reason nz_used is not correct, flag the unused portion */
    for(int i=0;i<is_len;i++) matfeti->scatter_src_domain[i]=-1;  /* to be respected by MatApplyScatter_Feti */
    
    for(int i=0;i<matfeti->n_dom;i++)  /* for all domains on processor */
    {
	const FetiDomain * domain = &(matfeti->domains[i]); /* shortcut in the absence of references in C */
	const Mat * BrT=&(domain->BrT);

	
	for(int row=0;row<(*BrT)->M;row++)  /* (Br_1)^T; MatLoadSeq_AIJ stores nrows in M */
	{
	    MatGetRow(*BrT,row,&n_cols,&cols,&vals);  /* loop through all rows only; rows of Br would be many more */
	    
	    for(int k=0;k<n_cols;k++)
	    {
		matfeti->scatter_src_domain[pos]=i;                 /* its ony a little waste of memory... */
		matfeti->scatter_src[pos]=row;
		matfeti->scatter_scale[pos]=vals[k];                /* only +1 -1 for now, *o* fix later */

		is_idx_local[pos]=pos;                /* obvious, nevertheless necessary for ISCreateGeneral */
		is_idx_global[pos]=cols[k];
		pos++;
	    }  // could also use a VecScatter (per domain) (Seq to Seq) plus a scale-Vector out of this
	    
	    MatRestoreRow(*BrT,row,&n_cols,&cols,&vals); 
	}

    }
    matfeti->scatter_len=pos;  /* test with < */
    ASSERT2(pos==is_len,"pos==is_len",pos,is_len);
    
    //VecCreateSeq(PETSC_COMM_SELF,pos,&matfeti->contrib);
    ISCreateGeneral(PETSC_COMM_SELF ,is_len,is_idx_local ,&ISlocal);     /* this is local */
    ISCreateGeneral(PETSC_COMM_SELF ,is_len,is_idx_global,&ISglobal);    /* this is going to be distributed */
	
    VecScatterCreate(matfeti->contrib,
		     ISlocal,
		     matfeti->lambda_copy,
		     ISglobal,
		     &matfeti->Br_scatter); /* ibi */
    /* note: both index sets are local (PETSC_COMM_SELF) to each processor,
       even though lambda is shared/distributed (PETSC_COMM_WORLD).
       make sure you understand the difference */


    PetscFree(is_idx_local); /* not needed any more */
    PetscFree(is_idx_global);

}





#if 1  // must be called AFTER Scc ist calculated
// Maybe this should be two VecScatters instead (more elegant)... no time now change that later
// bad design: took this literally from MatCreateScatter_Feti... fix later
int MatCreateScatterBc_Feti(Mat A)         
{
    Mat_Feti * const matfeti=(Mat_Feti*)A->data;
    /* storage for MatGetRow */
    int n_cols, *cols; /* with cubes max. 6; in the general case more */
    PetscScalar *vals; 
    MatInfo info;

    int is_len=0;        /* size of IS */
    int *is_idx_local;   /* index array for IS local  */
    int *is_idx_global;  /* index array for IS global */
    IS ISlocal;
    IS ISglobal;

    int pos=0;


    /* length of IS has to be calculated beforehand */

    for(int i=0;i<matfeti->n_dom;i++)
    {
	const FetiDomain * const domain = &(matfeti->domains[i]); /* shortcut in the absence of references in C */
	const Mat * const Bc=&(domain->Bc);
	MatGetInfo(*Bc, MAT_LOCAL, &info);
	is_len+=(int)info.nz_used;                          /* cast from PetscLogDouble (double) */
    }
    
    
    PetscMalloc(is_len*sizeof(int),&matfeti->BcT_scatter_src_domain); /* source domain number */
    PetscMalloc(is_len*sizeof(int),&matfeti->BcT_scatter_src);        /* source index */
    PetscMalloc(is_len*sizeof(PetscScalar),&matfeti->BcT_scatter_scale); /* scaling */

    PetscMalloc(is_len*sizeof(int),&is_idx_local);      /* for scatter to lambda */
    PetscMalloc(is_len*sizeof(int),&is_idx_global);     /* for scatter to lambda */

    VecCreateSeq(PETSC_COMM_SELF, is_len, &matfeti->BcT_contrib);


    /* in case by any reason nz_used is not correct, flag the unused portion */
    for(int i=0;i<is_len;i++) 
   	matfeti->BcT_scatter_src_domain[i]=-1;  /* to be respected by MatApplyScatter_Feti */
   
    for(int i=0;i<matfeti->n_dom;i++)  /* for all domains on processor */
    {
	const FetiDomain * domain = &(matfeti->domains[i]); /* shortcut in the absence of references in C */
	const Mat * Bc=&(domain->Bc);

	
	for(int row=0;row<(*Bc)->M;row++)  /* (Bc_1); MatLoadSeq_AIJ stores nrows in M */
	{
	    MatGetRow(*Bc,row,&n_cols,&cols,&vals);  /* loop through all rows only; rows of Bc would be many more */
	    
	    for(int k=0;k<n_cols;k++)
	    {
		matfeti->BcT_scatter_src_domain[pos]=i;                 /* its ony a little waste of memory... */
		matfeti->BcT_scatter_src[pos]=row;
		matfeti->BcT_scatter_scale[pos]=vals[k];                /* only +1 -1 for now, *o* fix later */

		is_idx_local[pos]=pos;                /* obvious, nevertheless necessary for ISCreateGeneral */
		is_idx_global[pos]=cols[k];
		pos++;
	    }  // could also use a VecScatter (per domain) (Seq to Seq) plus a scale-Vector out of this
	    
	    MatRestoreRow(*Bc,row,&n_cols,&cols,&vals); 
	}

    }
    matfeti->BcT_scatter_len=pos;  /* test with < */
    ASSERT2(pos==is_len,"pos==is_len",pos,is_len);
    
    //VecCreateSeq(PETSC_COMM_SELF,pos,&matfeti->contrib);
    ISCreateGeneral(PETSC_COMM_SELF ,is_len,is_idx_local ,&ISlocal);     /* this is local */
    ISCreateGeneral(PETSC_COMM_SELF ,is_len,is_idx_global,&ISglobal);    /* this is going to be distributed */

/*
    VecCreateMPI(matfeti->Kcc_ass->comm,
		 matfeti->Kcc_ass->m,  // same partitioning as Scc
		 matfeti->Kcc_ass->M,
		 &matfeti->uc_ass_copy);  // uses the communicator of Scc so that it can be a subset of world
    VecDuplicate(matfeti->fc_ass,&matfeti->uc_ass_copy); // neither
*/



    VecScatterCreate(matfeti->BcT_contrib,
		     ISlocal,
		     matfeti->fc_ass, // !!! maybe fc_ass would be enough; no uc_ass_copy
		     ISglobal,
		     &matfeti->BcT_scatter); /* ibi */
    /* note: both index sets are local (PETSC_COMM_SELF) to each processor,
       even though lambda is shared/distributed (PETSC_COMM_WORLD).
       make sure you understand the difference */


    PetscFree(is_idx_local); /* not needed any more */
    PetscFree(is_idx_global);

}
#endif



int MatCreate_Feti(Mat A); /* internal, if this is called, some setup still has to be made by hands afterwards */
                           /* i.e. make sure number of domains per processors is set as you want it (M)
			      and Mat_IS.lamba_copy must get the information about the partitioning
			      of lambda to be used with MatMult */

/* for use by user */
/* Mat_Feti is not registered for generic MatSetType (normally you do MatCreate and then MatSetType)
   (which does MatRegisterAll to fill the function pointer list with the constructors for the matrices) */
int MatCreateFeti(MPI_Comm comm, const int dom_per_proc, Mat* A)  
{   /* needs lambda to know the distribution over the processors, needed by the VecScatter */

    MatCreate(comm,dom_per_proc,0,0,0,A);        /* dom_per_proc is stored in A->m */
    MatCreate_Feti(*A);

    PetscObjectChangeTypeName((PetscObject)*A,MATFETI);  // solely changes the typename
    (*A)->assembled=PETSC_TRUE; // this Matrix type stays unassembled; MatMult checks for this flag

}

int CreatePetscCommTown()  // 2 Processors for now
{
    
    int world_size,town_size=2;   // if PCLU is to be used, then town_size=2^k
    int *ranks;
    MPI_Group town_group, world_group; 
    MPI_Comm_group(PETSC_COMM_WORLD, &world_group);  // extract group from comm
    MPI_Comm_size(PETSC_COMM_WORLD,&world_size);
    
    town_size=(world_size>town_size)?town_size:world_size;

    PetscMalloc(town_size*sizeof(int),&ranks);
    for(int i=0;i<town_size;i++) ranks[i]=i;

    MPI_Group_incl(world_group,town_size,ranks,&town_group);
    MPI_Comm_create(PETSC_COMM_WORLD,town_group,&PETSC_COMM_WORLD); // create now comm

    PetscFree(ranks);
}

//MatMult

/* For comparison:
   Mat_IS has to be used by first creating the Matrix with CreateMat()
   and then MatSetFromOptions probably calls MatCreate_IS();
   no MatCreateIS() for the user is provided */


/* internal function: supposes that a generic Mat has been created by CreateMat() */
/* since it may go into the function table, it can only 
   have one argument */
int MatCreate_Feti(Mat A)  /* Constructor for Mat_Feti() */
{                          /* has to set things up for simulating inheritance */
    int ierr;
    int rank,size;
    Mat_Feti * matfeti;
    int dom_per_proc;
    PetscFunctionBegin;
    dom_per_proc=A->m;  /* MatCreateFeti stored it there */

    ierr=PetscNew(Mat_Feti, &matfeti);CHKERRQ(ierr);
    A->data=(void*)matfeti;  /* that's actually me in data */
    ierr = PetscMemzero(matfeti,sizeof(Mat_Feti));CHKERRQ(ierr);    /* just for safety, everything to zero */
    ierr = PetscMemzero(A->ops,sizeof(struct _MatOps));CHKERRQ(ierr); /* MatCreate was already called */

    PetscMalloc(dom_per_proc*sizeof(FetiDomain),&matfeti->domains);
    PetscMemzero(matfeti->domains,dom_per_proc*sizeof(FetiDomain)); /* just for safety */

    A->ops->mult                    = MatMult_Feti; //MatMult_IS;
    A->ops->destroy                 = MatDestroy_Feti; //MatDestroy_IS;
    A->ops->setlocaltoglobalmapping = 0; //MatSetLocalToGlobalMapping_IS;
    A->ops->setvalueslocal          = 0; //MatSetValuesLocal_IS;
    A->ops->zerorowslocal           = 0; //MatZeroRowsLocal_IS;
    A->ops->assemblybegin           = 0; //MatAssemblyBegin_IS;  // could make this MatSetup_Feti
    A->ops->assemblyend             = 0; //MatAssemblyEnd_IS;
    A->ops->view                    = MatView_Feti; //MatView_IS;

    /* MPI_Comm_size(A->comm, &size); */
    MPI_Comm_rank(A->comm, &rank);       // comm is hidden in PETSCHEADER (!) in struct _p_Mat

    for(int i=0;i<dom_per_proc;i++)
    {
	(matfeti->domains[i]).domain_num=rank*dom_per_proc+i+1; /* okay, domain-numbers start at 1 */
    }                                                           /* even though rank starts at 0 */
                                                                /* but domains[] starts at 0 of course */

    matfeti->n_dom=dom_per_proc;
    PetscFunctionReturn(0);

    
}


/* Krr_1 usw. are sequential Matrices loaded using PETSC_COMM_SELF; 
   obviously MatView must use the same communicator as the one used 
   by the Matrix (Matrix seems only to be aware of its own processor */

int MatView_Feti(Mat A, PetscViewer)
{
    Mat_Feti * const matfeti=(Mat_Feti*)A->data;  /* that's actually me in data */
    FetiDomain * const domain=matfeti->domains;

    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD,&rank); 
    PetscSynchronizedPrintf(PETSC_COMM_WORLD,"MatView_Feti on processor %d\n",rank);

    PetscViewer view;
    PetscViewerDrawOpen(matfeti->Kcc_ass->comm,FETIDP_VIEWER_XDISPLAY,"K",PETSC_DECIDE,PETSC_DECIDE,400,400,&view);  
    MatView(matfeti->Kcc_ass,view);   // its now here 
    Wait("Kcc_ass");


  PetscSequentialPhaseBegin(PETSC_COMM_WORLD,1); // needed? 

    PetscViewer viewer;
    PetscViewerDrawOpen(PETSC_COMM_SELF,FETIDP_VIEWER_XDISPLAY,"K",PETSC_DECIDE,PETSC_DECIDE,500,500,&viewer);  

    for(int i=0;i<matfeti->n_dom;i++)
    {
	PetscSynchronizedPrintf(PETSC_COMM_WORLD,"Domain %d (%d)\n",domain[i].domain_num,i);
	FetiDomainLoad(matfeti->domains+i,FETIDP_PREFIX);

	MatView(domain[i].Krr,viewer); 
	Wait("Krr");

	MatView(domain[i].Krc,viewer); 
	Wait("Krc");
        //MatView(domain[i].Krr,PETSC_VIEWER_DRAW_WORLD); /* does not work: view must have one processor! */
                /* first processors draws everything coming from all; does not work; see above */

	MatView(domain[i].BrT,viewer); 
	Wait("BrT");

	MatView(domain[i].Bc,viewer); 
	Wait("Bc");

    }

    PetscViewerDestroy(viewer);

  PetscSequentialPhaseEnd(PETSC_COMM_WORLD,1);
    
    //ISView(matfeti->local, PETSC_VIEWER_STDOUT_SELF);
    //ISView(matfeti->global,PETSC_VIEWER_STDOUT_SELF);
    //Instead:

//    VecScatterView(matfeti->Br_scatter,PETSC_VIEWER_STDOUT_SELF);
//    Wait("Br_scatter");

    PetscSynchronizedFlush(PETSC_COMM_WORLD); // PetscSynchronizedPrintf needs flush

    PetscViewerDestroy(view);

    /* Alternative to PETSC_DRAW_WORLD: PetscViewerDrawOpen opens X on given X-display
       or presumable if given 0, on the every local machines  */
      /* PetscViewerDrawOpen(PETSC_COMM_SELF,0,"K",PETSC_DECIDE,PETSC_DECIDE,500,500,&viewer); */ 
      /* PETSC_DRAW_HALF_SIZE */

}

int MatSolveLocal_Feti(Mat A) /* of course could also take a Mat_Feti * */
{
    Mat_Feti * const matfeti=(Mat_Feti*)A->data;
    PC pc;
    KSP ksp;

    for(int i=0;i<matfeti->n_dom;i++)                   /* solve each domain */
    {
	SLES * sles=&matfeti->domains[i].Krrinv;
	SLESCreate(PETSC_COMM_SELF,sles);
	SLESSetOperators(*sles,
			 matfeti->domains[i].Krr,
			 matfeti->domains[i].Krr,
			 DIFFERENT_NONZERO_PATTERN); 	/* flag is not relevant */
	SLESGetKSP(*sles,&ksp);
	SLESGetPC(*sles,&pc);

	KSPSetType(ksp,KSPPREONLY); 
	PCSetType(pc,PCLU);         /* PCCholesky */
	//PCLUSetUseInPlace(pc);      /* destroys Krr */
	PCLUSetMatOrdering(pc,MATORDERING_NATURAL); /* MATORDERING_RCM, MATORDERING_QMD */
	/* normally SLES now factors upon first use of SLESSolve(), which is not what we want; 
	   therefore we call SLES_Setup explicitly
	   Alternatively with possible preallocation: MatLUFactor(), MatGetOrdering(), 
 	   MatLUFactorSymbolic(), MatLUFactorNumeric(), MatSolve(), see page 160ff in manual
	   but this way also different local solvers could be tested */
	/* SLES_SetUp(sles,b,x) needs x,b: calls PCSetVector(pc,b): b to use as
	   a template for temporary variables (pc->vec); not used by PCLU */ /* PCSetup? */
	/*
	PC_SetUp(pc); // is level developper and should be called by SLES_SetUp, but... 
	sles->setupcalled=1; 	
	*/
	// okay, okay, I decided this is too much of a hack, so I will rely on standard behavior
    }  
}


// like  VecConvertMPIToSeqAll 
int MatConvertMPI2SeqAll(const Mat src, Mat * const dst) // not used any more
{
    int M=src->M,N=src->N;
    IS is_row, is_col;
    Mat * seq_tmp;

    ISCreateStride(PETSC_COMM_WORLD,M,0,1,&is_row);
    ISCreateStride(PETSC_COMM_WORLD,N,0,1,&is_col); 
          /* side note: for some reason is_col must be the same on all processors (see manpage) */
    

    /* suggested reading: MatLUFactorNumeric_MPIAIJ_SuperLU_DIST */
    /* note NOT (!) MatGetSubMatrix or MatGetSubMatrix_MPIAIJ which are obviously meant to create MPI-submatrices;
       they actually call MatGetSubMatrices but then transform it into an MPI-Matrix using the row-distribution 
       given by is_row (it uses MatSetValues); that is also why all is_col have to be equal */
    
    /* MatGetSubMatrices(...) will only forward the call; so we call to 
       MatGetSubMattrices_MPIAIJ; but this one is not in the standard header files */
    MatGetSubMatrices(src,
		      1, // n in manpage of MatGetSubMatrices; MatGetSubMatrices_MPIAIJ calls this ismax (!)
		      &is_row,
		      &is_col,
		      MAT_INITIAL_MATRIX,
		      &seq_tmp); /* creates an Seq-Matrix, takes (Mat**) */
    /* note: calls MatGetSubMatrix_MPIAIJ_All (this one is spelled MatriX again); uses then MPIAllreduce   */

    ISDestroy(is_col);
    ISDestroy(is_row);
    
    *dst=*seq_tmp;

    /* Just for the files: */
    /* MatCreateSeqAIJ(PETSC_COMM_SELF,M,N,PetscNull,PetscDecide,dst);
            does:MatCreate; sets M,N
                 MatCreate_SeqAIJ; creates (MatSeqAIJ*) data 
                 MatSeqAIJSetPreallocation; allocates the array i.e. data->a
		                            in Mat only B->info.nz_unneeded and B->preallocated are changed
					    
       Also there exists
	         MatCreateSeqAIJ(PETSC_COMM_SELF,M,N,SKIP_ALLOCATION,PetscDecide,dst);
		 see here for significance of SKIP_ALLOCATION 

    */

    
}




/* New strategy: sum BcT (KrcT (Krrinv Krc)) Bc */

int MatSetUpCoarse_Feti(Mat A) // only internal use; anyway takes a generic Mat A 
{
    Mat_Feti * const matfeti=(Mat_Feti*)A->data;
    
    int its;

    PetscSynchronizedPrintf(PETSC_COMM_WORLD,"Entering Coarse Setup...");


#if 0
    MatlabInspect(matfeti->Kcc_ass,"Kcc_ass",0);  // MPI-Matrix; it seems only the first processor actually writes
#endif
    

    MatDuplicate(matfeti->Kcc_ass,MAT_COPY_VALUES,&matfeti->Scc_ass);

    for(int i=0;i<matfeti->n_dom;i++)
    {
	FetiDomain * domain=&matfeti->domains[i];

	
	int n_c=domain->Krc->N;  // small 
	int n_r=domain->Krc->M;
	
	Vec * vecs1;
	Vec * vecs2;

	PetscScalar ** vecs1_array;

	PetscMalloc(n_c*sizeof(Vec),&vecs1);
	PetscMalloc(n_c*sizeof(Vec),&vecs2);
	PetscMalloc(n_c*sizeof(PetscScalar *),&vecs1_array);


	
	for(int j=0;j<n_c;j++)
	{
	    VecCreateSeq(PETSC_COMM_SELF,n_r,&vecs1[j]); 
	    VecCreateSeq(PETSC_COMM_SELF,n_r,&vecs2[j]); 

	    MatGetColumnVector(domain->Krc,
			       vecs1[j],
			       j);

	    SLESSolve         (domain->Krrinv,
			       vecs1[j],
			       vecs2[j],&its);

	    VecDestroy(vecs1[j]);
	}
	

	for(int j=0;j<n_c;j++)
	{
	    VecCreateSeq(PETSC_COMM_SELF,n_c,&vecs1[j]);
	    
	    MatMultTranspose(domain->Krc,
			     vecs2[j],
			     vecs1[j]);

	    VecDestroy(vecs2[j]);
	}


	/* Now, distribute all this to the right places by using BcT and Bc */

	int * col_idx;
	PetscScalar * values;

	PetscMalloc(n_c*sizeof(int),&col_idx);
	PetscMalloc(n_c*sizeof(PetscScalar),&values);
	
	for(int j=0;j<n_c;j++)
	{
	    VecGetArray(vecs1[j],&vecs1_array[j]);
	}


	for(int j=0;j<n_c;j++)          // --- traverse all columns of BcT; 
	{                               // determines the target row and source vector
	    int n_cols;
	    int * cols;
	    PetscScalar * vals;

	    int n_col;
	    int col;
	    PetscScalar val;

	    int pos=0;


	    MatGetRow(domain->Bc,j,&n_cols,&cols,&vals);     // n_cols is either zero or one, in fact
	    ASSERT(n_cols==0||n_cols==1,"Bc corrupt: n_cols not 0 or 1");
	    n_col=n_cols;
	    if(n_col)
	    {
		col=*cols;
		val=*vals;
	    }
	    MatRestoreRow(domain->Krc,j,&n_cols,&cols,&vals); // only one call to MatGetRow can be pending
	    

	    for(int k=0;k<n_col;k++)        // --- if found s.th. (could be an if also)
	    {                               // this is a loop for psychological reasons

		for(int l=0;l<n_c;l++)  // --- traverse all rows of Bc; also all rows of vecs1[]
		{                       // determines the target column and the vectors component
		    MatGetRow(domain->Bc,l,&n_cols,&cols,&vals); // n_cols either zero or one
		
		    for(int m=0;m<n_cols;m++)   // --- if found s.th. 
		    {
			col_idx[pos]=cols[m];
			values[pos]= - vecs1_array[j][l];  // don't forget the minus!!
			pos++;
		    }

		}

	    }

	    if(n_col)
	    {
		ASSERT(pos==n_c,"Bc corrupt: pos!=n_c when building Scc");
		/* row oriented by default; so I stuck with that: this inserts a row */
		int * row_idx=&col;
		MatSetValues(matfeti->Scc_ass,  // not collective
			     1,   row_idx,
			     pos, col_idx,
			     values,
			     ADD_VALUES);
		
	    }
	}
	
	
	for(int j=0;j<n_c;j++)
	{
	    VecRestoreArray(vecs1[j],&vecs1_array[j]);
	    VecDestroy(vecs1[j]);
	}

	PetscFree(vecs1);  // all memory freed ?
	PetscFree(vecs2);

    }

    MatAssemblyBegin(matfeti->Scc_ass,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (matfeti->Scc_ass,MAT_FINAL_ASSEMBLY);
    
    
    /* convert from parallel to sequential matrix */

    
    /* MatConvert can only convert over the same communicator */
    //MatConvert(matfeti->Scc_ass,MATSEQAIJ,&matfeti->Scc);// there is no parallel LU => mpiaij.c line 1096 
                                                           // of course again, this could be solved iteratively
                                                           // without a conversion; what is cheaper?
    // if so: throw this conversion out and just change preconditioner, solver and PETSC_COMM_SELF below; should work fine

    // he is back: il est de retour: er ist zurueck: 
    MatConvertMPI2SeqAll(matfeti->Scc_ass, &matfeti->Scc);

    //MatlabInspect(matfeti->Scc,"Scc",4711);


    // --------------  Now we have Scc  ---------------
    
    ///// take it out for now...
    //MatDestroy(matfeti->Scc_ass); /* Kcc_ass could also be destroyed, but it is small */


    /*        ---------                                                                           */

    
    /* Invert Scc; could also be done iteratively */
    SLES * sles=&matfeti->Sccinv;

    SLESCreate(matfeti->Scc->comm,sles);  /**??? hier **/

    SLESSetOperators(*sles,
		     matfeti->Scc,
		     matfeti->Scc,
		     DIFFERENT_NONZERO_PATTERN); 	// flag is not relevant 

    KSP ksp;
    PC pc;

    SLESGetKSP(*sles,&ksp);
    SLESGetPC(*sles,&pc);

    KSPSetType(ksp,KSPPREONLY); 
    PCSetType(pc,PCLU);  

    //KSPSetType(ksp,KSPCG); 
    //PCSetType(pc,PCJACOBI);  

	// calculate fScc => see MatCalculateRHS_Feti

    PetscSynchronizedPrintf(PETSC_COMM_WORLD,"...Coarse Setup finished");    
}
/* 1. full matrices for Scc would use too much memory;
   2. cross-processor assembly of the results has to be done by SetValues anyway (no matrix scatter!)
   3. SLES does not take matrices as argument 
*/



/* internal use; called by MatMult and MatCalculateRHS_Feti */
/* scatters from matfeti->domains[i].ur to lambda (FETI_SCATTER_FORWARD_ADD)
   and back (FETI_SCATTER_REVERSE_INSERT) */
/* SCATTER_FORWARD, ADD_VALUES */  /* ibi */
/* takes local sequential vectors ur (Cave! - several per processor)
   and scatters them to lambda 
   using indices and the VectorScatter;
   I decided to put the local vectors in FetiDomain rather than having a src of Vec*
   which would have to be managed seperately; justification? its internal data to the CG-solver.
   this once again poses the question of putting lambda in Mat_Feti; but this 
   can not be done with the SLES-interface, since the user provides the solver with lambda...
*/
int MatScatter_Feti(Mat A, Vec lambda, FetiScatterMode mode) /* src the local seq. Vector */
{
    PetscFunctionBegin;
    Mat_Feti * const matfeti=(Mat_Feti*)A->data;
    PetscScalar * contrib_array;

    if(mode==FETI_SCATTER_FORWARD_ADD)
    {   /*  u ---> lambda  */
	VecGetArray(matfeti->contrib,&contrib_array); /* length must have been set */

	for(int i=0;i<matfeti->n_dom;i++)
	{
	    VecGetArray(matfeti->domains[i].ur,&matfeti->domains[i].ur_array);
	}

	for(int j=0;j<matfeti->scatter_len && matfeti->scatter_src[j]!=-1;j++)
	{  
	    WARN_IF(matfeti->scatter_src[j]==-1,"There seem to be unnecessary (zero?) entries in BrT.");
	    PetscScalar result;
	    result=matfeti->domains[matfeti->scatter_src_domain[j]].ur_array[matfeti->scatter_src[j]];
	    result*=matfeti->scatter_scale[j];            // this is why I could not use a VecScatter
	    contrib_array[j]=result;

	}	    

		    /* VecScatter */
	VecRestoreArray(matfeti->contrib,&contrib_array);
	VecScatterBegin(matfeti->contrib,lambda,ADD_VALUES,SCATTER_FORWARD,matfeti->Br_scatter); 
	VecScatterEnd  (matfeti->contrib,lambda,ADD_VALUES,SCATTER_FORWARD,matfeti->Br_scatter); 

    }
    else
    {
	if(mode==FETI_SCATTER_REVERSE_INSERT)
	{   /*  lambda ---> u  */
	    
	    PetscScalar const zero=0;
	    for(int i=0;i<matfeti->n_dom;i++)
	    {
		VecSet(&zero,matfeti->domains[i].ur);
		VecGetArray(matfeti->domains[i].ur,&matfeti->domains[i].ur_array);
	    }

	            /* VecScatter */
	    //VecScatterBegin(matfeti->contrib,lambda,INSERT_VALUES,SCATTER_REVERSE,matfeti->Br_scatter); 
	    //VecScatterEnd(matfeti->contrib,lambda,INSERT_VALUES,SCATTER_REVERSE,matfeti->Br_scatter); 
	    //why is the Petsc documentation always so laconic?
	    VecScatterBegin(lambda,matfeti->contrib,INSERT_VALUES,SCATTER_REVERSE,matfeti->Br_scatter); 
	    VecScatterEnd  (lambda,matfeti->contrib,INSERT_VALUES,SCATTER_REVERSE,matfeti->Br_scatter); 
	    

	    VecGetArray(matfeti->contrib,&contrib_array);
	    
	    for(int j=0;j<matfeti->scatter_len && matfeti->scatter_src[j]!=-1;j++)
	    {
		PetscScalar result;
		result=contrib_array[j];
		result*=matfeti->scatter_scale[j];  /* is this correct ?*/
		matfeti->domains[matfeti->scatter_src_domain[j]].ur_array[matfeti->scatter_src[j]]+=result ;
	    }	    

	    VecRestoreArray(matfeti->contrib,&contrib_array);
	    
	}
	else
	{
	    /* error */
	    ASSERT(0,"Unknown mode in MatScatter_Feti");

	}
    }

    for(int i=0;i<matfeti->n_dom;i++)
    {
	VecRestoreArray(matfeti->domains[i].ur,&matfeti->domains[i].ur_array);
    }
    PetscFunctionReturn(0);
}

#if 1
/* bad design: took this word by word from MatScatter_Feti... fix later */
int MatScatterBc_Feti(Mat A, Vec ucg, FetiScatterMode mode)  // uc_ass/ucg here; (same comm as fc_ass)
{
    PetscFunctionBegin;

    Mat_Feti * const matfeti=(Mat_Feti*)A->data;
    PetscScalar * contrib_array;

    if(mode==FETI_SCATTER_FORWARD_ADD)  // uc-->uc_ass  (domains[0].uc-->matfeti->uc_ass)
    {   
	VecGetArray(matfeti->BcT_contrib,&contrib_array); /* length must have been set */

	for(int i=0;i<matfeti->n_dom;i++)
	{
	    VecGetArray(matfeti->domains[i].uc,&matfeti->domains[i].uc_array);
	}

	for(int j=0;j<matfeti->BcT_scatter_len && matfeti->BcT_scatter_src[j]!=-1;j++)
	{  
	    WARN_IF(matfeti->BcT_scatter_src[j]==-1,"There seem to be unnecessary (zero?) entries in Bc.");
	    PetscScalar result;
	    result=matfeti->domains[matfeti->BcT_scatter_src_domain[j]].uc_array[matfeti->BcT_scatter_src[j]];
	    result*=matfeti->BcT_scatter_scale[j];            // this is why I could not use a VecScatter
	    contrib_array[j]=result;

	}	    

		    /* VecScatter */
	VecRestoreArray(matfeti->BcT_contrib,&contrib_array);
	VecScatterBegin(matfeti->BcT_contrib,ucg,ADD_VALUES,SCATTER_FORWARD,matfeti->BcT_scatter); 
	VecScatterEnd  (matfeti->BcT_contrib,ucg,ADD_VALUES,SCATTER_FORWARD,matfeti->BcT_scatter); 

    }
    else
    {
	if(mode==FETI_SCATTER_REVERSE_INSERT) // uc_ass-->uc
	{   
	    
	    PetscScalar const zero=0;
	    for(int i=0;i<matfeti->n_dom;i++)
	    {
		VecSet(&zero,matfeti->domains[i].uc);
		VecGetArray(matfeti->domains[i].uc,&matfeti->domains[i].uc_array);
	    }

	            /* VecScatter */
	    //VecScatterBegin(matfeti->contrib,lambda,INSERT_VALUES,SCATTER_REVERSE,matfeti->Br_scatter); 
	    //VecScatterEnd(matfeti->contrib,lambda,INSERT_VALUES,SCATTER_REVERSE,matfeti->Br_scatter); 
	    //why is the Petsc documentation always so laconic?
	    VecScatterBegin(ucg,matfeti->BcT_contrib,INSERT_VALUES,SCATTER_REVERSE,matfeti->BcT_scatter); 
	    VecScatterEnd  (ucg,matfeti->BcT_contrib,INSERT_VALUES,SCATTER_REVERSE,matfeti->BcT_scatter); 
	    

	    VecGetArray(matfeti->BcT_contrib,&contrib_array);
	    
	    for(int j=0;j<matfeti->BcT_scatter_len && matfeti->BcT_scatter_src[j]!=-1;j++)
	    {
		PetscScalar result;
		result=contrib_array[j];
		result*=matfeti->BcT_scatter_scale[j];  /* is this correct ?*/
		matfeti->domains[matfeti->BcT_scatter_src_domain[j]].uc_array[matfeti->BcT_scatter_src[j]]+=result ;
	    }	    

	    VecRestoreArray(matfeti->BcT_contrib,&contrib_array);
	    
	}
	else
	{
	    /* error */
	    ASSERT(0,"Unknown mode in MatScatterBc_Feti");

	}
    }

    for(int i=0;i<matfeti->n_dom;i++)
    {
	VecRestoreArray(matfeti->domains[i].uc,&matfeti->domains[i].uc_array);
    }
    PetscFunctionReturn(0);
}
#endif

/*  internal use; called by MatLoad        */
/*  space for coarse-solve aso. in MatMult */
int MatSetUpTemporarySpace_Feti(Mat A)
{

    Mat_Feti * const matfeti=(Mat_Feti*)A->data;    
    for(int i=0;i<matfeti->n_dom;i++)
    {
	FetiDomain * const domain=&matfeti->domains[i];
	VecCreateSeq(PETSC_COMM_SELF,domain->Krr->M     ,&domain->ur_tmp1);
	VecCreateSeq(PETSC_COMM_SELF,domain->Krr->M     ,&domain->ur_tmp2);
//	VecCreateSeq(PETSC_COMM_SELF,domain->Krc->N     ,&domain->ucl_tmp);  ////!! weg
	VecCreateSeq(PETSC_COMM_SELF,matfeti->Kcc_ass->M,&domain->ucg_tmp1);  // cave Kcc_ass->M !!
	VecCreateSeq(PETSC_COMM_SELF,matfeti->Kcc_ass->M,&domain->ucg_tmp2);
    }
    VecDuplicate(matfeti->fc_ass,&matfeti->uc_ass_tmp);  // MPI
    VecCreateSeq(PETSC_COMM_SELF,matfeti->Kcc_ass->M,&matfeti->uc_tmp1); // Seq
    VecCreateSeq(PETSC_COMM_SELF,matfeti->Kcc_ass->M,&matfeti->uc_tmp2);
}

int MatDestroyTemporarySpace_Feti(Mat A)
{
    Mat_Feti * const matfeti=(Mat_Feti*)A->data;    
    for(int i=0;i<matfeti->n_dom;i++)
    {
	FetiDomain * const domain=&matfeti->domains[i];
	VecDestroy(domain->ur_tmp1);
	VecDestroy(domain->ur_tmp2);
	VecDestroy(domain->ucg_tmp1);  
	VecDestroy(domain->ucg_tmp2);
    }
    VecDestroy(matfeti->uc_ass_tmp);
    VecDestroy(matfeti->uc_tmp1);
    VecDestroy(matfeti->uc_tmp2);
}

/* Feti-DP system matrix */

/* sum ( Br Krrinv BrT  +   Br Krrinv Krc   Sccinv  BcT KrcT Krrinv BrT ) lambda */


int MatMult_Feti(Mat A, Vec src_lambda, Vec dst_lambda)
{
    Mat_Feti * const matfeti=(Mat_Feti*)A->data;
    int its;
    
    PetscFunctionBegin;

    ASSERT(src_lambda!=dst_lambda,"Source and destination vectors can't be the same in MatMult_Feti.");
    
    PetscScalar zero=0;
    VecSet(&zero,dst_lambda);


    MatScatter_Feti  (A,                               /* apply all Br */
		      src_lambda,                      /* src_lambda ---> all ur */
		      FETI_SCATTER_REVERSE_INSERT);  


    for(int i=0;i<matfeti->n_dom;i++)
    {
    	FetiDomain * domain=&matfeti->domains[i];


    /* Standard Feti */
    
	SLESSolve        (domain->Krrinv,
			  domain->ur,              /* not needed any more until scatter reverse */
			  domain->ur_tmp1,&its);   /* this is important; Krrinv*Br*lambda; keep it */


	ASSERT(its>0,"SLES error.");

    /* DP Coarse Problem */

	MatMultTranspose (domain->Krc,
			  domain->ur_tmp1,
			  domain->uc);

//	MatMultTranspose (domain->Bc,
//			  domain->ucl_tmp,
//			  domain->ucg_tmp1);      // there has to be a sum here! 

    }

    MatScatterBc_Feti(A,                         // taken out of the loop
		      matfeti->uc_ass_tmp,
		      FETI_SCATTER_FORWARD_ADD);  

    VecConvertMPIToSeqAll(matfeti->uc_ass_tmp,&matfeti->uc_tmp1);

    SLESSolve        (matfeti->Sccinv,     
		      matfeti->uc_tmp1,
		      matfeti->uc_tmp2,&its); 

    ASSERT(its>0,"SLES error.");       

    for(int i=0;i<matfeti->n_dom;i++)
    {
	FetiDomain * const domain=&matfeti->domains[i];

    
	MatMult          (domain->Bc,           // no scatter this time; just take matmult
			  matfeti->uc_tmp2,    
			  domain->uc);

//	MatScatterBc_Feti(A,                    // if use this then take it out of the loop
//			  matfeti->uc_tmp2,     // scatter would result want an MPI-Vector (not there)
//			  FETI_SCATTER_REVERSE_INSERT);


	MatMult          (domain->Krc,
			  domain->uc,
			  domain->ur);
    
	SLESSolve        (domain->Krrinv,       // more efficient: evaluate Krrinv*Krc beforehand
			  domain->ur,
			  domain->ur_tmp2,&its);

	ASSERT(its>0,"SLES error.");
	
	PetscScalar const unity=1;
	VecWAXPY         (&unity,             /* ur = 1*ur_tmp + ur_tmp2 */
			  domain->ur_tmp1,
			  domain->ur_tmp2,
			  domain->ur);

    }
    

    MatScatter_Feti(A,                              /* apply BrT */
		    dst_lambda,                     /* src_lambda ---> all ur */
		    FETI_SCATTER_FORWARD_ADD);     

    PetscFunctionReturn(0);
}


/* the RHS is of the size of lambda and should therefore have the same parallel
   layout as lambda:
   In the KSP the initial residual is r=d_rhs-A*lambda, so it should
   have the same layout as lambda; the user must take care of this              */

/* fScc = fc - \sum BcT KrcT Krrinv fr   must be assembled over all processors!! */

int MatCalculateRHS_Feti(Mat A, Vec * dr_Scc)  /* dr with additional correction from Scc */
{
    int its;
    Mat_Feti * const matfeti=(Mat_Feti*)A->data;      
    PetscScalar const zero=0, minus_one=-1;

    PetscSynchronizedPrintf(PETSC_COMM_WORLD,"Calculating RHS...");

    ASSERT(matfeti->lambda_copy,"Need MatLoad first: lambda_copy==0");
    VecDuplicate(matfeti->lambda_copy,dr_Scc);  /* give dr_Scc the same layout as lambda; does not copy values */
    VecSet(&zero, *dr_Scc);                     // VecDuplicateMPI_AIJ

    VecDuplicate(matfeti->fc_ass,&matfeti->fScc_ass);
    VecCopy     (matfeti->fc_ass, matfeti->fScc_ass);

   
    // First we have to calculate fScc
    for(int i=0;i<matfeti->n_dom;i++)
    {
	FetiDomain * const domain=&matfeti->domains[i];    
	
	SLESSolve        (domain->Krrinv,
			  domain->fr,
			  domain->ur_tmp1,&its);

	MatMultTranspose (domain->Krc,
			  domain->ur_tmp1,
			  domain->uc);

	VecScale(&minus_one,domain->uc);
    }

#if 1  // both alternatives work; maybe this is a smarter way to do the scatter? Bc is not very big...
    MatScatterBc_Feti(A,
		      matfeti->fScc_ass,   //uc_ass_tmp1,
		      FETI_SCATTER_FORWARD_ADD);

#else
    for(int i=0;i<matfeti->n_dom;i++)
    {
	FetiDomain * const domain=&matfeti->domains[i];    

	MatMultTranspose (domain->Bc,        // this is the slow but the safe way: actually performing Bc*ucl
			  domain->uc,
			  domain->ucg_tmp1); // okay, this is 8 times linear and it could be 8, but hey... 

 
        PetscScalar * ucg_tmp1_array;
        int n_cl;  // length of fc_ass

        VecGetArray(domain->ucg_tmp1,&ucg_tmp1_array);
        VecGetSize (domain->ucg_tmp1,&n_cl); // Vec have property N for n_rows; unlike Matrices (M for rows)
        for(int j=0;j<n_cl;j++)
        {
            if(ucg_tmp1_array[j])  // if worth doing s.th.
            {
                VecSetValue(matfeti->fScc_ass,
			    j,
			    ucg_tmp1_array[j],   // minus (!)
			    ADD_VALUES);
            }   // VecSetValues could be slightly faster
        }
	VecRestoreArray(domain->ucg_tmp1,&ucg_tmp1_array);
    }
    VecAssemblyBegin       (matfeti->fScc_ass);  /* No communicator? Matrix has one ! */
    VecAssemblyEnd         (matfeti->fScc_ass);
#endif

    VecConvertMPIToSeqAll  (matfeti->fScc_ass,
			    &matfeti->fScc);

    // MatlabInspectVec(matfeti->fScc); // fScc is now correct; also second check; all the same


    // --- okay, we've got fScc now; what we really want ist dr_Scc, so we continue ---

    // first calculate dr; dr and dr_Scc have the same size as lambda
    //  I factored out Br and also Krrinv so I get
    //  dr_Scc=Br Krrinv (fr - Krc Bc Sccinv fScc)


    SLESSolve      (matfeti->Sccinv,
		    matfeti->fScc,
		    matfeti->uc_tmp1,&its);


    for(int i=0;i<matfeti->n_dom;i++)
    {
	FetiDomain * const domain=&matfeti->domains[i];
/*
	SLESSolve      (matfeti->Sccinv,
			matfeti->fScc,
			domain->ucg_tmp1,&its);
*/

	MatMult        (domain->Bc,          // efficient enough   
			matfeti->uc_tmp1,       
			domain->uc);     /*hier*/

//	MatScatterBc_Feti(A,
//			  matfeti->uc_ass_tmp1,
//			  FETI_SCATTER_REVERSE_INSERT);


	VecScale       (&minus_one,          // put the minus here, this vector is short
 	                domain->uc);

	MatMultAdd     (domain->Krc,    
			domain->uc,
			domain->fr,
			domain->ur_tmp1);  // ur_tmp1 = A uc + fr

	SLESSolve      (domain->Krrinv,
			domain->ur_tmp1,
			domain->ur,&its);	

	//MatlabInspectVecs(domain->ur,i); // never used this; since dr_Scc turned out to be right
    }


    // note that MatScatter_Feti can only be called when all domain
    // contributions are calculated


    MatScatter_Feti (A,                           /* apply Br */
		     *dr_Scc,                     /* ur --> dr_Scc */
		     FETI_SCATTER_FORWARD_ADD);   /* note: *dr_Scc is an mpi-vector */


    MatlabInspectVec(*dr_Scc); // *dr_Scc seems to be correct now also

    PetscSynchronizedPrintf(PETSC_COMM_WORLD,"...RHS finished");
}

/* Make Kcc a MPI-Matrix and put the values there by MatSetValues (only nonzero)
   then convert it to Seq-Matrix (P. 58 manual)*/



int MatSetUp_Feti(Mat A)   // MatLoad_Feti calls this
{
    static PetscTruth matsetup_called=PETSC_FALSE;

    ASSERT(!matsetup_called,"MatSetUp_Feti was already called.");
    matsetup_called=PETSC_TRUE;

    Mat_Feti * const matfeti=(Mat_Feti*)A->data;
    PetscSynchronizedPrintf(PETSC_COMM_WORLD,"Entering Setup...");


    MatCreateScatter_Feti(A);      /* transform BrT into the two scatters */
    MatCreateScatterBc_Feti(A);   /* transform Bc into the two scatters */


    MatSetUpTemporarySpace_Feti(A);

    MatSolveLocal_Feti(A);    // setups the SLES
    MatSetUpCoarse_Feti(A);   // /*hier*/ 
    PetscSynchronizedPrintf(PETSC_COMM_WORLD,"...Setup finished");

}

// destroy domains and scatters
int MatDestroy_Feti(Mat A)
{
    Mat_Feti * const matfeti=(Mat_Feti*)A->data;

    /* domains */
    for(int i=0;i<matfeti->n_dom;i++)
    {
	FetiDomain * const domain = &(matfeti->domains[i]); 

	MatlabInspect(domain->BrT,"BrT",i);  // last minute inspection :)

	MatDestroy(domain->Krr);
	MatDestroy(domain->Krc);
	MatDestroy(domain->BrT);
	MatDestroy(domain->Bc);
	SLESDestroy(domain->Krrinv);
	
	VecDestroy(domain->ur);
	VecDestroy(domain->uc);
	VecDestroy(domain->fr);
    }

    MatDestroyTemporarySpace_Feti(A);

    /* global */

    MatDestroy(matfeti->Kcc_ass);
    VecDestroy(matfeti->fc_ass);
    
    MatDestroy(matfeti->Scc_ass); // used to be destroyed in MatSetUpCoarse_Feti
    VecDestroy(matfeti->fScc_ass);

    MatDestroy(matfeti->Scc); 
    VecDestroy(matfeti->fScc);


    /* Scatter */
    VecDestroy(matfeti->contrib);
    PetscFree(matfeti->scatter_src_domain);
    PetscFree(matfeti->scatter_src);
    PetscFree(matfeti->scatter_scale);
    VecScatterDestroy(matfeti->Br_scatter);


    PetscFree(matfeti->domains);
    
    PetscFree(matfeti);
    //?? PetscFree(A->ops);?
    
}

int MatWrite_lambda(Mat A) 
{
    Mat_Feti * const matfeti=(Mat_Feti*)A->data;
    
    PetscViewer file;
    PetscViewerASCIIOpen(PETSC_COMM_WORLD,"Petsc_lambda",&file); // only first processor writes
    PetscViewerSetFormat(file,PETSC_VIEWER_ASCII_MATLAB);
    PetscObjectSetName((PetscObject)matfeti->lambda_copy,"Petsc_lambda"); // name for matlab
                                                               // PETSC_COMM_WORLD: each proc. writes
    VecView(matfeti->lambda_copy,file);
    PetscViewerDestroy(file);
}

int MatlabInspect(Mat mat, char const * const name, int const domain_num)
{
    char fname[256]={};

    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);  // not mat->comm; all processors would write to the same file
                                             // if mat was sequential; now if mat is mpi, only the first processor 
                                             // will have a nonempty file
                                             
    PetscViewer viewer;

    sprintf(fname,"Petsc_%s_%d__%d.m",name,domain_num,rank);
    PetscSynchronizedPrintf(PETSC_COMM_WORLD,"Writing %s (%d)\n",fname,rank); 
                                               // omitting rank also works...
                                               // wonder how Petsc does it... (last one wins?)
    PetscViewerASCIIOpen(mat->comm,fname,&viewer);
    PetscViewerSetFormat(viewer,PETSC_VIEWER_ASCII_MATLAB);
    
    char oname[256]={};
    sprintf(oname,"Petsc_%s_%d__%d",name,domain_num,rank);
    PetscObjectSetName((PetscObject)mat,oname);
    
    MatView(mat,viewer);

    PetscSynchronizedFlush(PETSC_COMM_WORLD); // PetscSynchronizedPrintf needs flush    

}


int MatlabInspect_Feti(Mat A)   /* write all the stuff to files */
{
    Mat_Feti * const matfeti=(Mat_Feti*)A->data;    
    char fname[256]={};
    int rank;
    

    MPI_Comm_rank(A->comm, &rank);  // A->comm ist PETSC_COMM_WORLD anyway

#if 1
    PetscViewer viewer;

    //viewer=PETSC_VIEWER_STDOUT_WORLD;
    //PetscViewerASCIIOpen(PETSC_COMM_WORLD,"inspect.mat",&viewer);
    //PetscViewerSocketOpen(PETSC_COMM_SELF,PETSC_NULL,PETSC_DEFAULT,&viewer); --

    char name[]="Scc";
    Mat mat=matfeti->Scc;
    
    sprintf(fname,"Petsc_%s__%d.m",name,rank);
    PetscSynchronizedPrintf(PETSC_COMM_WORLD,"Writing %s (%d)\n",fname,rank); 
                                               // omitting rank also works...
                                               // wonder how Petsc does it... (last one wins?)
    PetscViewerASCIIOpen(PETSC_COMM_SELF,fname,&viewer);
    PetscViewerSetFormat(viewer,PETSC_VIEWER_ASCII_MATLAB);
    
    char oname[256]={};
    sprintf(oname,"Petsc_%s__%d",name,rank);
    PetscObjectSetName((PetscObject)mat,oname);
    
    MatView(mat,viewer);

#else

    for(int i=0;i<matfeti->n_dom;i++)
    {
	FetiDomain * const domain=&matfeti->domains[i];    

	PetscViewer viewer;

	char name[]="Bc";
	Mat mat=domain->Bc;

	sprintf(fname,"Petsc_%s_%d.m",name,domain->domain_num);
	PetscSynchronizedPrintf(PETSC_COMM_WORLD,"Writing %s\n",fname);

	PetscViewerASCIIOpen(PETSC_COMM_SELF,fname,&viewer);
	PetscViewerSetFormat(viewer,PETSC_VIEWER_ASCII_MATLAB);

	char oname[256]={};
	sprintf(oname,"Petsc_%s_%d",name,domain->domain_num);
	PetscObjectSetName((PetscObject)mat,oname);
	
	MatView(mat,viewer);
    }
#endif
    PetscSynchronizedFlush(PETSC_COMM_WORLD); // PetscSynchronizedPrintf needs flush    
}

int MatlabInspectVec(Vec v)
{

    MatlabInspectVecs(v,-1);

}

int MatlabInspectVecs(Vec v, int dom)
{

    char fname[256]={};
    int rank;

    //VecCreate(PETSC_COMM_WORLD,&v);

    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    char name[]="vecinspect";
    char domnum[256]={};
    if(dom>=0)
	sprintf(domnum,"%d_",dom);
    
    sprintf(fname,"Petsc_%s_%s_%d.m",name,domnum,rank);

    PetscViewer viewer;
    //viewer=PETSC_VIEWER_STDOUT_WORLD;
    PetscViewerASCIIOpen(v->comm,fname,&viewer);
    PetscViewerSetFormat(viewer,PETSC_VIEWER_ASCII_MATLAB);

    char oname[256]={};
    sprintf(oname,"Petsc_%s_%s_%d",name,domnum,rank);
    PetscObjectSetName((PetscObject)v,oname);

    VecView(v,viewer);


}


int MatlabPut(PetscObject A)
    /**/
    {
    //PetscMatlabEngine *matlab;
    //PetscMatlabEngineCreate(PETSC_COMM_WORLD,PETSC_NULL,matlab);
    //PetscMatlabEnginePut(PETSC_MATLAB_ENGINE_WORLD,A);
    //PETSC_MATLAB_ENGINE_WORLD
    //PetscMatlabEngine PETSC_MATLAB_ENGINE_(MPI_Comm comm)
    //int PetscMatlabEngineEvaluate(PetscMatlabEngine mengine,char *string,...)
    //PetscMatlabEngineDestroy(matlab);
    }
    /**/

int MatScatterTest_Feti(Mat A,Vec lambda)  // calculates lambda=lambda+Br*BrT*lambda
{
    

    MatScatter_Feti  (A,                           /* apply all Br */
		      lambda,                      /* src_lambda ---> all ur */
		      FETI_SCATTER_REVERSE_INSERT);  


    MatScatter_Feti  (A,                              /* apply BrT */
		      lambda,                     /* src_lambda ---> all ur */
		      FETI_SCATTER_FORWARD_ADD);     
}


#endif

/* This will have a member function MatMult_FetiPart 
   The function  MatCreate_FetiPart(Mat A)  will set
      A->ops->mult=MatMult_FetiPart
   so later on by SLESSetOperators(...A...)
   KSP will access it by calling
   KSP_MatMult(ksp,A,x,y) which is defined as
   MatMult(A,x,y) which does
   (*mat->ops->mult)(mat,x,y);


/* The Fetipreconditioner will habe PCApply_FETI 
   pc->ops->apply will be set to it 
   to be more accurate
   The preconditioner will provide
   PCApply_Feti(PC,Vec,Vec)
   to which in PCCreate_Feti will be set to
   pc->ops->apply=PCApply_Feti
   which will be called in KSPSolve_CG in the form
   KSP_PCApply(ksp,ksp->B,x,y)
   which is defined as
   PCApply(ksp->B,x,y) 
   which is in fact
   (*pc->ops->apply)(pc,x,y)

   KSP (to be more exact _P_KSP)
   has a PC as a substructure...
   So KSP has access to all that is in KSP.


   lambda is provided by the user upon solution of the system;
   uc is used internally for the coarse grid problem;
   only the scatters (Br...) have to be set up before.

*/

