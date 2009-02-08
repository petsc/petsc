#include "feti.h"

           /* Creates Mat_Feti assigning the N domains to the processors using a function MatFetiBalance
              Generally one could have different numbers of domains per processor */
int MatCreateFeti(MPI_Comm comm, const int N, Mat* A)  
{   /* needs lambda to know the distribution over the processors, needed by the VecScatter */

    PetscFunctionBegin;

    MatCreate(comm,A);                /* N is stored in A->m */
    MatSetSizes(*A,N,0,0,0)
    MatCreate_Feti(*A);

    PetscObjectChangeTypeName((PetscObject)*A,MATFETI);                   /* solely changes the typename */
    (*A)->assembled=PETSC_TRUE;      /* this Matrix type stays unassembled; MatMult checks for this flag */

    PetscFunctionReturn(0);
}

           /*  internal function: supposes that a generic Mat has been created by CreateMat()  */

           /* if this function is used by user, some setup still has to be made by hands afterwards  */
           /* i.e. make sure number of domains per processors is set as you want it (M) and Mat_IS.lamba_copy 
              must get the information about the partitioning of lambda to be used with MatMult */

           /* since it may go into the function table, it can only have one argument */
int MatCreate_Feti(Mat A)    /* Constructor for Mat_Feti() */
{                            /* has to set things up for simulating inheritance */
    int ierr;
    int rank,size;
    Mat_Feti * matfeti;
    int N,start,end;

    PetscFunctionBegin;
    N=A->m;                  /* MatCreateFeti stored it there */

    ierr=PetscNewLog(A,Mat_Feti, &matfeti);CHKERRQ(ierr);
    A->data=(void*)matfeti;  /* that's actually me in data */
    ierr = PetscMemzero(matfeti,sizeof(Mat_Feti));CHKERRQ(ierr);      /* just for safety, everything to zero */

    PetscMalloc(N*sizeof(FetiDomain),&matfeti->domains);
    PetscMemzero(matfeti->domains,N*sizeof(FetiDomain));/* just for safety */

    A->ops->mult                    = MatMult_Feti; 
    A->ops->destroy                 = MatDestroy_Feti; 
    A->ops->assemblybegin           = 0;                /* could make this MatSetup_Feti */
    A->ops->assemblyend             = 0; 
    A->ops->view                    = MatView_Feti;     

#if 1
    MatFetiBalance(((PetscObject)A)->comm,N,&start,&end);              /* determines local domains by a simple formula */

    matfeti->n_dom=end-start+1;
    for(int i=0;i<matfeti->n_dom;i++)
    {
	(matfeti->domains[i]).domain_num=start+i;       /* domain-numbers start at 1 */
    }
#else
    MPI_Comm_rank(((PetscObject)A)->comm, &rank);                      /* comm is hidden in PETSCHEADER (!) in struct _p_Mat */

    for(int i=0;i<9;i++)
    {
        (matfeti->domains[i]).domain_num=rank*9+i+1;    /* okay, domain-numbers start at 1      */
    }                                                           /* even though rank starts at 0         */
                                                                /* but domains[] starts at 0, of course */
    matfeti->n_dom=9;
#endif

    PetscFunctionReturn(0);
}  

           /*   Load an entire Mat_Feti  */
           /*   The parameter lambda is meant as a help to the user who will typically not know
                the size that lambda will have; MatCreateFeti sets it for him; 
                if the NullPointer is given he must VecCreateMPI lambda himself                       */

           /*   List of members that must be initialised by any function that replaces MatLoad_Feti:
                Kcc, fc, BrT_, lambda, Krr_, Krc_                                                     */

int MatFetiBalance(MPI_Comm comm, int N, int * start, int * end)
{
   PetscFunctionBegin;

   int mpi_rank, mpi_size;
   int size;

   MPI_Comm_rank(comm,&mpi_rank); 
   MPI_Comm_size(comm,&mpi_size);

   ASSERT(N>=mpi_size,"More Processors than domains! Please start on less processors.");

   /* 41/5 => 8+1 8 8 8 8 */
   *start           = mpi_rank  *  (N/mpi_size) + ((mpi_rank<(N%mpi_size))?mpi_rank:(N%mpi_size))  +  1;
   size             = N / mpi_size + ((mpi_rank < (N % mpi_size))?1:0); 
   *end             = *start + size - 1 ;

   PetscSynchronizedPrintf(PETSC_COMM_WORLD,"%d-%d ",*start,*end);
   PetscSynchronizedFlush (PETSC_COMM_WORLD);

   PetscFunctionReturn(0);
}

int MatLoad_Feti(Mat A, Vec* lambda) 
{
    int llen;
    PetscScalar const zero=0;
    Mat_Feti * const matfeti=(Mat_Feti*)A->data;                

    PetscFunctionBegin;

    FetiLoadMatMPI(FETIDP_PREFIX,FETIDP_KCC,".mat",&(matfeti->Kcc_ass)); /* cave ! MPI-Matrix */
    FetiLoadVecMPI(FETIDP_PREFIX,FETIDP_FC,".mat",&(matfeti->fc_ass));

    /* take the length of lambda from the length of domains[0]->BrT */
    FetiDomainLoad(matfeti->domains,FETIDP_PREFIX);

    llen=matfeti->domains->BrT->N;

    for(int i=1;i<matfeti->n_dom;i++)
    {
	FetiDomainLoad(matfeti->domains+i,FETIDP_PREFIX);
    }

    if(matfeti->lambda_copy==0 && lambda!=0)  /* lambda_copy was zeroed by MatCreate_Feti */
    {
	VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE,llen,lambda);
	VecSet(*lambda,zero);
	/* There is room for optimization; the partitioning of lambda should minimize 
	   the communication and be influenced somewhat by the scatters BrT      */
	matfeti->lambda_copy=*lambda; /* save a copy for creation of the scatter */
    }

    MatSetUp_Feti(A);  /* setup coarse, solve local, setup scatter; */

    int localsize;
    VecGetLocalSize(*lambda,&localsize);

    A->M=llen;         /* the generic MatMult checks for this... */
    A->N=llen; 
    A->n=localsize;    /* again just to please MatMult; does not apply here                          */
    A->m=localsize;    /* also to please MatMult; MatCreate stored N here during creation */

    if(matfeti->domains->use_Q)
    {
	VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE,matfeti->domains->Q->N,&matfeti->mu);
	VecCreateSeq(PETSC_COMM_WORLD,matfeti->domains->Q->N,&matfeti->mu_seq);
    }

    PetscFunctionReturn(0);
}                     

           /* -------------------- Feti-DP   --   System   --    Matrix ------------------- */
           /* sum ( Br Krrinv BrT  +   Br Krrinv Krc   Sccinv  BcT KrcT Krrinv BrT ) lambda */

int FetiVecAppend(Vec a, Vec b, Vec c)  /* c=[a;b]; c cannot be a or b */
{
    PetscScalar *a_array, *b_array, *c_array;

    PetscFunctionBegin;

    ASSERT(a->N+b->N==c->N,"Wrong sizes in FetiVecAppend\n");

    VecGetArray(a,&a_array);
    VecGetArray(b,&b_array);
    VecGetArray(c,&c_array);

    for(int i=0;i<a->N;i++)
	c_array[i]=a_array[i];

    for(int i=0;i<b->N;i++)
	c_array[a->N+i]=b_array[i];

    VecRestoreArray(a,&a_array);
    VecRestoreArray(b,&b_array);
    VecRestoreArray(c,&c_array);

    PetscFunctionReturn(0);
}

int FetiVecSplit(Vec c, Vec a, Vec b)   /* c cannot be a or b */
{
    PetscScalar *a_array, *b_array, *c_array;

    PetscFunctionBegin;

    ASSERT(a->N+b->N==c->N,"Wrong sizes in FetiVecAppend\n");

    VecGetArray(a,&a_array);
    VecGetArray(b,&b_array);
    VecGetArray(c,&c_array);

    for(int i=0;i<a->N;i++)
	a_array[i]=c_array[i];

    for(int i=0;i<b->N;i++)
	b_array[i]=c_array[a->N+i];

    VecRestoreArray(a,&a_array);
    VecRestoreArray(b,&b_array);
    VecRestoreArray(c,&c_array);

    PetscFunctionReturn(0);
}

int FetiVecAddValues(Vec a, Vec b)  /* meant for "a" Seq-Vec and "b" MPI-Vec */
{
    PetscScalar *a_array;
    int a_array_len;

    PetscFunctionBegin;

    ASSERT(a->N==b->N,"Sizes don't match in FetiVecAddValues\n");

    VecGetArray(a,&a_array);
    VecGetSize (a,&a_array_len); 

    for(int j=0;j<a_array_len;j++)
    {
	if(a_array[j])     /* if worth doing s.th. */
	{
                VecSetValue(b,
			    j,
			    a_array[j],  
			    ADD_VALUES);
            }   
    }

    VecRestoreArray(a,&a_array);

    PetscFunctionReturn(0);

}

int FetiMatAddValues(Mat A, Vec b, int row_offset, int col)  /* meant for "A" MPI-Mat and "b" Seq-Vec */
{

    PetscScalar *b_array;
    int b_len;

    PetscFunctionBegin;

    VecGetArray(b,&b_array);
    VecGetSize (b,&b_len); 

    for(int k=0;k<b_len;k++)
    {
	if(b_array[k])     /* if worth doing s.th. */
	{
	    MatSetValue(A,
			k+row_offset,
			col,
			b_array[k],
			ADD_VALUES);
	}   
    }

    VecRestoreArray(b,&b_array);

    PetscFunctionReturn(0);

}

int MatMult_Feti(Mat A, Vec src_lambda, Vec dst_lambda)
{
    static long call=0;

    Mat_Feti * const matfeti=(Mat_Feti*)A->data;
    int its;

    PetscFunctionBegin;

    call++;

    ASSERT(src_lambda!=dst_lambda,"Source and destination vectors can't be the same in MatMult_Feti.");

    PetscScalar zero=0;
    VecSet(dst_lambda,zero);

    if(matfeti->domains->use_Q)
	VecSet(matfeti->mu,zero);

    MatFetiScatter   (A,                            /* apply all Br */
		      src_lambda,                   /* src_lambda ---> all ur */
		      FETI_SCATTER_REVERSE_INSERT);  

    for(int i=0;i<matfeti->n_dom;i++)
    {
    	FetiDomain * domain=&matfeti->domains[i];

    /* Standard Feti */

	SLESSolve        (domain->Krrinv,
			  domain->ur,               /* not needed any more until scatter reverse    */
	                  domain->ur_tmp1,&its);    /* this is important; Krrinv*Br*lambda; keep it */
	ASSERT(its>0,"SLES error.");

    /* DP Coarse Problem */

	MatMultTranspose (domain->Krc,
			  domain->ur_tmp1,
			  domain->uc);

	if(domain->use_Q)   /* new, second path */
	{
	    PetscPrintf(MPI_COMM_WORLD,"using Q\n");
	    MatMultTranspose (domain->Q,
			      domain->ur_tmp1,
			      domain->mu_tmp);

	    FetiVecAddValues     (domain->mu_tmp,
			      matfeti->mu);  /* scatter the values to the MPI-Vector */
	}

    }

    if(matfeti->domains->use_Q)                        /* Assemble MPI-mu */
    {
	VecAssemblyBegin(matfeti->mu);  /* new */
	VecAssemblyEnd  (matfeti->mu);

    }

    VecSet(matfeti->uc_ass_tmp,zero);              /* zero uc_ass_tmp               */ 

    MatFetiScatterBc (A,                            /* all domain->uc --> uc_ass_tmp */
		      matfeti->uc_ass_tmp,          /* taken out of the loop         */
		      FETI_SCATTER_FORWARD_ADD);  

    /* --------- transformation to sequential ---------- */

    if(matfeti->uc_tmp1)
	VecDestroy(matfeti->uc_tmp1);               /* unfortunately VecConvertMPIToSeqAll always creates a new Vec */
    VecConvertMPIToSeqAll(matfeti->uc_ass_tmp,&matfeti->uc_tmp1);

    if(matfeti->domains->use_Q)                     /* convert also mu to seq */
    {
	if(matfeti->mu_seq)
	    VecDestroy(matfeti->mu_seq);            /* unfortunately VecConvertMPIToSeqAll always creates a new Vec */
	VecConvertMPIToSeqAll(matfeti->mu,&matfeti->mu_seq);

	FetiVecAppend(matfeti->uc_tmp1,matfeti->mu_seq,matfeti->ucmu_tmp1);

	SLESSolve          (matfeti->Sccinv,     
			    matfeti->ucmu_tmp1,
			    matfeti->ucmu_tmp2,&its); 

	FetiVecSplit(matfeti->ucmu_tmp2,matfeti->uc_tmp2,matfeti->mu_seq);

    }
    else
    {
	SLESSolve          (matfeti->Sccinv,     
			    matfeti->uc_tmp1,
			    matfeti->uc_tmp2,&its); 
    }

    ASSERT(its>0,"SLES error.");       

    for(int i=0;i<matfeti->n_dom;i++)
    {
	FetiDomain * const domain=&matfeti->domains[i];

	MatMult          (domain->Bc,               /* no scatter this time; just take plain MatMult */
			  matfeti->uc_tmp2,    
			  domain->uc);

	MatMult          (domain->Krc,
			  domain->uc,
			  domain->ur);

	if(matfeti->domains->use_Q)
	{
	    MatMultAdd   (domain->Q,
		          matfeti->mu_seq,
		          domain->ur,
		          domain->ur); 
	}

	SLESSolve        (domain->Krrinv,       
			  domain->ur,
			  domain->ur_tmp2,&its);
	ASSERT(its>0,"SLES error.");

	PetscScalar const unity=1;
	VecWAXPY         (domain->ur,unity,                   /* ur = 1*ur_tmp1 + ur_tmp2 */
			  domain->ur_tmp1,
			  domain->ur_tmp2);

    }

    MatFetiScatter (A,                              /* apply BrT */
		    dst_lambda,                     /* all ur --->(add) lambda  */
		    FETI_SCATTER_FORWARD_ADD);     

    PetscFunctionReturn(0);
}

           /* like VecConvertMPIToSeqAll */
int MatConvertMPI2SeqAll(const Mat src, Mat * const dst) 
{
    int M=src->M,N=src->N;
    IS is_row, is_col;
    Mat * seq_tmp;

    PetscFunctionBegin;

    ISCreateStride(PETSC_COMM_WORLD,M,0,1,&is_row);
    ISCreateStride(PETSC_COMM_WORLD,N,0,1,&is_col); 

    MatGetSubMatrices(src,
		      1, /* n in manpage of MatGetSubMatrices; MatGetSubMatrices_MPIAIJ calls this ismax (!) */
		      &is_row,
		      &is_col,
		      MAT_INITIAL_MATRIX,
		      &seq_tmp);                                     /* creates an Seq-Matrix, takes (Mat**) */
    /* note: calls MatGetSubMatrix_MPIAIJ_All (this one is spelled MatriX again); uses then MPIAllreduce     */

    ISDestroy(is_col);
    ISDestroy(is_row);

    *dst=*seq_tmp;

    PetscFunctionReturn(0);

}

/* This function can be used by the user */
/* The generic version through MatCreate() is not yet supported */

           /* This could also have been made using two IS and a scaling in between */
           /* internal use; called by MatLoad 
              assumes BrT to contain the connectivity information 
              assumes that the partitioning information for the scatter can be taken from lambda_copy */
int MatFetiCreateScatter(Mat A)         /* name may be misleading */
{
    Mat_Feti * const matfeti=(Mat_Feti*)A->data;

    /* storage for MatGetRow */
    int n_cols, *cols;   /* n_cols with cubes max. 6; in the general case more */
    PetscScalar *vals; 
    MatInfo info;

    int is_len=0;        /* size of IS */
    int *is_idx_local;   /* index array for IS local  */
    int *is_idx_global;  /* index array for IS global */
    IS ISlocal;
    IS ISglobal;

    int pos=0;

    PetscFunctionBegin;

    /* length of IS has to be calculated beforehand */
    for(int i=0;i<matfeti->n_dom;i++)
    {
	const FetiDomain * const domain = &(matfeti->domains[i]); /* shortcut in the absence of references in C */
	const Mat * const BrT=&(domain->BrT);
	MatGetInfo(*BrT, MAT_LOCAL, &info);
	is_len+=(int)info.nz_used;                                /* cast from PetscLongDouble (double) */
    }

    PetscMalloc(is_len*sizeof(int),&matfeti->scatter_src_domain);    /* source domain number */
    PetscMalloc(is_len*sizeof(int),&matfeti->scatter_src);           /* source index */
    PetscMalloc(is_len*sizeof(PetscScalar),&matfeti->scatter_scale); /* scaling */

    PetscMalloc(is_len*sizeof(int),&is_idx_local);      /* for scatter to lambda */
    PetscMalloc(is_len*sizeof(int),&is_idx_global);     /* for scatter to lambda */

    VecCreateSeq(PETSC_COMM_SELF, is_len, &matfeti->contrib);

    /* in case by any reason nz_used is not correct, flag the unused portion */
    for(int i=0;i<is_len;i++) matfeti->scatter_src_domain[i]=-1;  /* to be respected by MatApplyScatter_Feti */

    for(int i=0;i<matfeti->n_dom;i++)                             /* for all domains on processor */
    {
	const FetiDomain * domain = &(matfeti->domains[i]);       /* shortcut in the absence of references in C */
	const Mat * BrT=&(domain->BrT);

	for(int row=0;row<(*BrT)->M;row++)            /* (Br_1)^T; MatLoadSeq_AIJ stores nrows in M */
	{
	    MatGetRow(*BrT,row,&n_cols,&cols,&vals);  /* loop through all rows only; rows of Br would be many more */

	    for(int k=0;k<n_cols;k++)
	    {
		matfeti->scatter_src_domain[pos]=i;       /* its ony a little waste of memory... */
		matfeti->scatter_src[pos]=row;
		matfeti->scatter_scale[pos]=vals[k];      /* only +1 -1 for now, *o* fix later */

		is_idx_local[pos]=pos;                    /* obvious, nevertheless necessary for ISCreateGeneral */
		is_idx_global[pos]=cols[k];
		pos++;
	    }  /* could also use a VecScatter (per domain) (Seq to Seq) plus a scale-Vector out of this */

	    MatRestoreRow(*BrT,row,&n_cols,&cols,&vals); 
	}

    }
    matfeti->scatter_len=pos;  /* test with < */
    ASSERT2(pos==is_len,"pos==is_len",pos,is_len);

    ISCreateGeneral(PETSC_COMM_SELF ,is_len,is_idx_local ,&ISlocal);     /* this is local */
    ISCreateGeneral(PETSC_COMM_SELF ,is_len,is_idx_global,&ISglobal);    /* this is going to be distributed */

    VecScatterCreate(matfeti->contrib,
		     ISlocal,
		     matfeti->lambda_copy,
		     ISglobal,
		     &matfeti->Br_scatter); 
    /* note: both index sets are local (PETSC_COMM_SELF) to each processor,
       even though lambda is shared/distributed (PETSC_COMM_WORLD).
       make sure you understand the difference */

#if 0 /* The PetscScatterView can not view a scatter on only one processor => does not work for  mpiexec -n 1 */
    {
    char a[256]={0};
    int rank;

    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);  
    sprintf(a,"Petsc_scatterinspect_%d",rank);
    PetscViewer file;
    PetscViewerASCIIOpen(PETSC_COMM_SELF,a,&file); 
    PetscViewerSetFormat(file,PETSC_VIEWER_ASCII_MATLAB);
    PetscObjectSetName((PetscObject)matfeti->lambda_copy,"Petsc_inspectscatter"); /* name for matlab */

    VecScatterView(matfeti->Br_scatter,file);
    PetscViewerDestroy(file);
    }
#endif

    PetscFree(is_idx_local);       /* not need the memory any more */
    PetscFree(is_idx_global);

    ISDestroy(ISlocal);
    ISDestroy(ISglobal);

    PetscFunctionReturn(0);
}

           /* Must be called only AFTER Scc ist calculated */

int MatFetiCreateScatterBc(Mat A)         
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

    PetscFunctionBegin;

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

    for(int i=0;i<matfeti->n_dom;i++)           /* for all domains on processor */
    {
	const FetiDomain * domain = &(matfeti->domains[i]); /* shortcut in the absence of references in C */
	const Mat * Bc=&(domain->Bc);

	for(int row=0;row<(*Bc)->M;row++)       /* (Bc_1); MatLoadSeq_AIJ stores n_rows in M */
	{
	    MatGetRow(*Bc,row,&n_cols,&cols,&vals);  /* loop through all rows only; rows of Bc would be many more */

	    for(int k=0;k<n_cols;k++)
	    {
		matfeti->BcT_scatter_src_domain[pos]=i;                 /* its ony a little waste of memory... */
		matfeti->BcT_scatter_src[pos]=row;
		matfeti->BcT_scatter_scale[pos]=vals[k];                /* only +1 -1 for now, *o* fix later   */

		is_idx_local[pos]=pos;                /* obvious, nevertheless necessary for ISCreateGeneral   */
		is_idx_global[pos]=cols[k];
		pos++;
	    }  

	    MatRestoreRow(*Bc,row,&n_cols,&cols,&vals); 
	}

    }
    matfeti->BcT_scatter_len=pos; 
    ASSERT2(pos==is_len,"pos==is_len",pos,is_len);

    ISCreateGeneral(PETSC_COMM_SELF ,is_len,is_idx_local ,&ISlocal);     /* this is local */
    ISCreateGeneral(PETSC_COMM_SELF ,is_len,is_idx_global,&ISglobal);    /* this is going to be distributed */

    VecScatterCreate(matfeti->BcT_contrib,
		     ISlocal,
		     matfeti->fc_ass, 
		     ISglobal,
		     &matfeti->BcT_scatter);

    PetscFree(is_idx_local);
    PetscFree(is_idx_global);

    ISDestroy(ISlocal);
    ISDestroy(ISglobal);

    PetscFunctionReturn(0);

}

/* obsolete */
int CreatePetscCommTown()  /* 2 Processors for now */
{
    PetscFunctionBegin;

    int world_size,town_size=2;    /* if PCLU is to be used, then town_size=2^k */
    int *ranks;
    MPI_Group town_group, world_group; 
    MPI_Comm_group(PETSC_COMM_WORLD, &world_group);  /* extract group from comm */
    MPI_Comm_size(PETSC_COMM_WORLD,&world_size);

    town_size=(world_size>town_size)?town_size:world_size;

    PetscMalloc(town_size*sizeof(int),&ranks);
    for(int i=0;i<town_size;i++) ranks[i]=i;

    MPI_Group_incl(world_group,town_size,ranks,&town_group);
    MPI_Comm_create(PETSC_COMM_WORLD,town_group,&PETSC_COMM_WORLD); /* create now comm */

    PetscFree(ranks);

    PetscFunctionReturn(0);
}

int MatView_Feti(Mat A, PetscViewer)
{
    Mat_Feti * const matfeti=(Mat_Feti*)A->data;  
    FetiDomain * const domain=matfeti->domains;

    int rank;

    PetscFunctionBegin;

    MPI_Comm_rank(PETSC_COMM_WORLD,&rank); 
    PetscSynchronizedPrintf(PETSC_COMM_WORLD,"MatView_Feti on processor %d\n",rank);

    PetscViewer view;
    PetscViewerDrawOpen(matfeti->((PetscObject)Kcc_ass)->comm,FETIDP_VIEWER_XDISPLAY,"K",PETSC_DECIDE,PETSC_DECIDE,400,400,&view);  
    MatView(matfeti->Kcc_ass,view);   
    WAIT("Kcc_ass");

  PetscSequentialPhaseBegin(PETSC_COMM_WORLD,1); 

    PetscViewer viewer;
    PetscViewerDrawOpen(PETSC_COMM_SELF,FETIDP_VIEWER_XDISPLAY,"K",PETSC_DECIDE,PETSC_DECIDE,500,500,&viewer);  

    for(int i=0;i<matfeti->n_dom;i++)
    {
	PetscSynchronizedPrintf(PETSC_COMM_WORLD,"Domain %d (%d)\n",domain[i].domain_num,i);
	FetiDomainLoad(matfeti->domains+i,FETIDP_PREFIX);

	MatView(domain[i].Krr,viewer); 
	WAIT("Krr");

	MatView(domain[i].Krc,viewer); 
	WAIT("Krc");

	MatView(domain[i].BrT,viewer); 
	WAIT("BrT");

	MatView(domain[i].Bc,viewer); 
	WAIT("Bc");

    }

    PetscViewerDestroy(viewer);

    PetscSequentialPhaseEnd(PETSC_COMM_WORLD,1);

    PetscSynchronizedFlush(PETSC_COMM_WORLD); /* PetscSynchronizedPrintf needs flush */

    PetscViewerDestroy(view);

    PetscFunctionReturn(0);

}

int MatFetiSolveLocal(Mat A)   
{
    Mat_Feti * const matfeti=(Mat_Feti*)A->data;
    PC pc;
    KSP ksp;

    PetscFunctionBegin;

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
	PCSetType(pc,PCLU);              
	PCFactorSetUseInPlace(pc);           
	PCFactorSetMatOrderingType(pc,MATORDERING_NATURAL); /* MATORDERING_RCM, MATORDERING_QMD, MATORDERING_ND */

	KSPSetTolerances(ksp,1e-7,1e-50,1e+5,10000); 
	SLESAppendOptionsPrefix(*sles,"local_");    /* uses options with prefix -local_ */
	SLESSetFromOptions(*sles);                

    }  

    PetscFunctionReturn(0); 

}

/*

*/

/* New strategy: sum BcT (KrcT (Krrinv Krc)) Bc */

int MatFetiSetUpCoarse(Mat A) /* only internal use; anyway takes a generic Mat A  */
{
    Mat_Feti * matfeti=(Mat_Feti*)A->data;

    MatFetiSetUpScc(A);
    if(matfeti->domains->use_Q)
	MatFetiSetUpSccTilde(A);  /* modifies Scc */
    MatFetiConvertScc2Seq(A);

}

int MatFetiSetUpScc(Mat A) /* only internal use; anyway takes a generic Mat A  */
{
    Mat_Feti * const matfeti=(Mat_Feti*)A->data;
    int its;

    PetscFunctionBegin;

    PetscSynchronizedPrintf(PETSC_COMM_WORLD,"Entering Coarse Setup (+local solves)... ");

    MatDuplicate(matfeti->Kcc_ass,MAT_COPY_VALUES,&matfeti->Scc_ass);

    for(int i=0;i<matfeti->n_dom;i++)
    {
	FetiDomain * domain=&matfeti->domains[i];

	int n_c=domain->Krc->N;                   /* is small */
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

	    {
		int reason=0;
		KSP ksp;
		SLESGetKSP(domain->Krrinv,&ksp);
		KSPGetConvergedReason(ksp,(KSPConvergedReason*)&reason);

		WARN_IF((reason!=2 && reason!=3)&&(reason==4&&its!=1),"Fetidp local solve convergence is bad.");
	    }

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

	for(int j=0;j<n_c;j++)          /* --- traverse all columns of BcT; */
	{                               /* determines the target row and source vector */
	    int n_cols;
	    int * cols;
	    PetscScalar * vals;

	    int n_col;
	    int col;
	    PetscScalar val;

	    int pos=0;

	    MatGetRow(domain->Bc,j,&n_cols,&cols,&vals);      /* n_cols is either zero or one, in fact */
	    ASSERT(n_cols==0||n_cols==1,"Bc corrupt: n_cols not 0 or 1");
	    n_col=n_cols;
	    if(n_col)
	    {
		col=*cols;
		val=*vals;
	    }
	    MatRestoreRow(domain->Krc,j,&n_cols,&cols,&vals); /* only one call to MatGetRow can be pending */

	    for(int k=0;k<n_col;k++)        /* --- if found s.th. (could be an if also) */
	    {                               /* this is a loop for psychological reasons */

		for(int l=0;l<n_c;l++)  /* --- traverse all rows of Bc; also all rows of vecs1[]  */
		{                       /* determines the target column and the vectors component */
		    MatGetRow(domain->Bc,l,&n_cols,&cols,&vals); 

		    for(int m=0;m<n_cols;m++)   /* --- if found s.th. */
		    {
			col_idx[pos]=cols[m];
			values[pos]= - vecs1_array[j][l];  /* don't forget the minus */
			pos++;
		    }

		}

	    }

	    if(n_col)
	    {
		ASSERT(pos==n_c,"Bc corrupt: pos!=n_c when building Scc");
		/* row oriented by default; kept this: this inserts a row */
		int * row_idx=&col;
		MatSetValues(matfeti->Scc_ass,          /* not collective */
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

	PetscFree(vecs1);  
	PetscFree(vecs2);
	PetscFree(vecs1_array);

	PetscFree(col_idx);
	PetscFree(values);

    }

    MatAssemblyBegin(matfeti->Scc_ass,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (matfeti->Scc_ass,MAT_FINAL_ASSEMBLY);

}    

/* should work for seq and mpi; could also be made with ISes to assign at all places */
int MatAssignSubMatrix(Mat A, Mat B, int m_offset, int n_offset, InsertMode mode)
{

    int from,to;
    int ncols;
    int *cols;
    PetscScalar *vals;
    int dst_row;

    MatGetOwnershipRange(A,&from,&to);       
    for(int row=from;row<to;row++)           
    {
	MatGetRow(A,row,&ncols,&cols,&vals); 
	for(int j=0;j<ncols;j++)
	{
	    cols+=n_offset;
	}
	dst_row=row+m_offset;
	MatSetValues(B,1,&dst_row,ncols,cols,vals,mode);
	MatRestoreRow(A,row,&ncols,&cols,&vals); 
    }

    MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (B,MAT_FINAL_ASSEMBLY);
}

int MatFetiSetUpSccTilde(Mat A)
{
    Mat_Feti * const matfeti=(Mat_Feti*)A->data;

    Mat Scctilde;

    int c_len=matfeti->Scc_ass->M;
    int m_len=0;

    if(matfeti->domains->use_Q)
	m_len=matfeti->domains->Q->N;  
    else
	ASSERT(0,"MatFetiSetUpSccTilde called without use_Q. Unnecessary.\n"); 

    int cm_len=c_len+m_len; 

    int its;
    PetscReal norm;
    PetscScalar minus_one=-1;

    MatCreate(PETSC_COMM_WORLD,&Scctilde);
    MatSetSizes(Scctilde,PETSC_DECIDE,PETSC_DECIDE,cm_len,cm_len);
    MatSetType(Scctilde,MATMPIAIJ);

    MatAssignSubMatrix (matfeti->Scc_ass,  /* copy into          */
		        Scctilde,
		        0,0,               /* upper, left corner */
		        INSERT_VALUES);

    MatDestroy(matfeti->Scc_ass);
    matfeti->Scc_ass=Scctilde;             /* exchange matrix */

#if 1  /* build the three other parts of Scctilde */
    for(int i=0;i<matfeti->n_dom;i++)
    {
	FetiDomain * domain=&matfeti->domains[i];	
	for(int j=0;j<m_len;j++)                
	{

	    /* upper right corner */
	    MatGetColumnVector(domain->Q,
			       domain->ur_tmp1, 
			       j);              

	    VecNorm(domain->ur_tmp1,NORM_1,&norm);  
	    if(norm != 0.0) continue;                 

	    VecScale(domain->ur_tmp1,minus_one);

	    SLESSolve(domain->Krrinv,
		      domain->ur_tmp1,
		      domain->ur_tmp2,&its);

	    MatMultTranspose(domain->Krc,
			     domain->ur_tmp2,
			     domain->uc);

	    MatMultTranspose(domain->Bc,
			     domain->uc,
			     domain->ucg_tmp1);

	    FetiMatAddValues(matfeti->Scc_ass, 	    
			     domain->ucg_tmp1,
			     0,                   /* upper */ 
			     j+c_len);            /* right corner */

	    /* lower right corner */
	    MatMultTranspose(domain->Q,
			     domain->ur_tmp2, /* reuse result from above */
			     domain->mu_tmp);  

	    FetiMatAddValues(matfeti->Scc_ass,
			     domain->mu_tmp,
			     c_len,               /* lower */
			     j+c_len);            /* right corner */
	}

	/* lower left corner */
	if(m_len)  
	{
	    for(int j=0;j<domain->Bc->N;j++)  
	    {

		MatGetColumnVector(domain->Bc,
				   domain->uc,      
				   j);              

		VecNorm(domain->uc,NORM_1,&norm);
		if(norm != 0.0) continue;               

		VecScale(domain->uc,minus_one);

		MatMult   (domain->Krc,
			   domain->uc,
			   domain->ur_tmp1);

		SLESSolve (domain->Krrinv,
			   domain->ur_tmp1,
			   domain->ur_tmp2,&its);

		MatMultTranspose(domain->Q,
				 domain->ur_tmp2,
				 domain->mu_tmp);

		FetiMatAddValues(matfeti->Scc_ass,
				 domain->mu_tmp,
				 c_len,                 /* lower */ 
				 j);                    /* left corner */
	    }
	}
    }

    MatAssemblyBegin(matfeti->Scc_ass,MAT_FINAL_ASSEMBLY);  
    MatAssemblyEnd  (matfeti->Scc_ass,MAT_FINAL_ASSEMBLY);

    MatlabInspect(matfeti->Scc_ass,"Scc_ass",4711);
#endif

}

/* converts and inverts */
int MatFetiConvertScc2Seq(Mat A) /* only internal use; anyway takes a generic Mat A  */
{
    Mat_Feti * const matfeti=(Mat_Feti*)A->data;

    /* convert from parallel to sequential matrix */

    MatConvertMPI2SeqAll(matfeti->Scc_ass, &matfeti->Scc);

    /* --------------  Now we have Scc  --------------- */

    MatDestroy(matfeti->Scc_ass);             /* Kcc_ass could also be destroyed, but it is small */
    matfeti->Scc_ass=0;

    /*        ---------                                                                           */

    /* invert Scc; could also be done iteratively */
    SLES * sles=&matfeti->Sccinv;

    SLESCreate(matfeti->((PetscObject)Scc)->comm,sles); 

    SLESSetOperators(*sles,
		     matfeti->Scc,
		     matfeti->Scc,
		     DIFFERENT_NONZERO_PATTERN); 	/* flag is not relevant */

    KSP ksp;
    PC pc;

    SLESGetKSP(*sles,&ksp);
    SLESGetPC(*sles,&pc);

    KSPSetType(ksp,KSPPREONLY); 
    PCSetType(pc,PCLU);  
    PCFactorSetDamping(pc,1e-9); 

#if 0
    KSPSetType(ksp,KSPCG); 
    PCSetType(pc,PCJACOBI);  
#endif

      /* calculate fScc => see MatCalculateRHS_Feti */

    PetscSynchronizedPrintf(PETSC_COMM_WORLD,"...Coarse Setup finished. ");    

    PetscFunctionReturn(0);
}
/*
   1. full matrices for Scc would use too much memory; (if every proc. holds a copy)
   2. cross-processor assembly of the results has to be done by SetValues anyway (no matrix scatter!)
   3. SLES does not take matrices as argument 
*/

           /* internal use; called by MatMult and MatCalculateRHS_Feti */

           /* scatters from matfeti->domains[i].ur to lambda (FETI_SCATTER_FORWARD_ADD)
                                                    and back (FETI_SCATTER_REVERSE_INSERT) */

int MatFetiScatter(Mat A, Vec lambda, FetiScatterMode mode) /* src the local seq. Vector */
{
    PetscFunctionBegin;
    Mat_Feti * const matfeti=(Mat_Feti*)A->data;
    PetscScalar * contrib_array;

    if(mode==FETI_SCATTER_FORWARD_ADD || mode==FETI_SCATTER_FORWARD_ADD_SCALE)
    {   /*  u ---> lambda  */
	VecGetArray(matfeti->contrib,&contrib_array);        /* length must have been set */

	for(int i=0;i<matfeti->n_dom;i++)
	{
	    VecGetArray(matfeti->domains[i].ur,&matfeti->domains[i].ur_array);
	}

	if(mode==FETI_SCATTER_FORWARD_ADD_SCALE)
	{
	    for(int j=0;j<matfeti->scatter_len && matfeti->scatter_src[j]!=-1;j++)
	    {  
		WARN_IF(matfeti->scatter_src[j]==-1,"There seem to be unnecessary (zero?) entries in BrT.");
		PetscScalar result;
		result=matfeti->domains[matfeti->scatter_src_domain[j]].ur_array[matfeti->scatter_src[j]];
		result*=matfeti->scatter_scale[j];           /* this is why I could not use a single VecScatter */
		contrib_array[j]=result;
	    }	    
	}
	else
	{
	    for(int j=0;j<matfeti->scatter_len && matfeti->scatter_src[j]!=-1;j++)
	    {  
		PetscScalar result;
		result=matfeti->domains[matfeti->scatter_src_domain[j]].ur_array[matfeti->scatter_src[j]];
		result*=(matfeti->scatter_scale[j]>=0)?1:-1;
		contrib_array[j]=result;
	    }	    
	}

		    /* VecScatter */
	VecRestoreArray(matfeti->contrib,&contrib_array);
	VecScatterBegin(matfeti->Br_scatter,matfeti->contrib,lambda,ADD_VALUES,SCATTER_FORWARD);
	VecScatterEnd  (matfeti->Br_scatter,matfeti->contrib,lambda,ADD_VALUES,SCATTER_FORWARD);

    }
    else
    {
	if(mode==FETI_SCATTER_REVERSE_INSERT || mode==FETI_SCATTER_REVERSE_INSERT_SCALE)
	{   /*  lambda ---> u  */

	    PetscScalar const zero=0;
	    for(int i=0;i<matfeti->n_dom;i++)
	    {
              VecSet(matfeti->domains[i].ur,zero);
		VecGetArray(matfeti->domains[i].ur,&matfeti->domains[i].ur_array);
	    }

	            /* VecScatter */
	    VecSet(matfeti->contrib,zero);  
	    VecScatterBegin(matfeti->Br_scatter,lambda,matfeti->contrib,INSERT_VALUES,SCATTER_REVERSE);
	    VecScatterEnd  (matfeti->Br_scatter,lambda,matfeti->contrib,INSERT_VALUES,SCATTER_REVERSE);

	    VecGetArray(matfeti->contrib,&contrib_array);

	    if(mode==FETI_SCATTER_REVERSE_INSERT_SCALE)	    
	    {
		for(int j=0;j<matfeti->scatter_len && matfeti->scatter_src[j]!=-1;j++)
		{
		    PetscScalar result;
		    result=contrib_array[j];
		    result*=matfeti->scatter_scale[j];  
		    matfeti->domains[matfeti->scatter_src_domain[j]].ur_array[matfeti->scatter_src[j]]+=result ;
		}	    
	    }
	    else
	    {
		for(int j=0;j<matfeti->scatter_len && matfeti->scatter_src[j]!=-1;j++)
		{
		    PetscScalar result;
		    result=contrib_array[j];
		    result*=(matfeti->scatter_scale[j]>=0)?1:-1;  
		    matfeti->domains[matfeti->scatter_src_domain[j]].ur_array[matfeti->scatter_src[j]]+=result ;
		}	    
	    }

	    VecRestoreArray(matfeti->contrib,&contrib_array);

	}
	else
	{
	    /* complain about the error */
	    ASSERT(0,"Unknown mode in MatScatter_Feti");

	}
    }

    for(int i=0;i<matfeti->n_dom;i++)
    {
	VecRestoreArray(matfeti->domains[i].ur,&matfeti->domains[i].ur_array);
    }
    PetscFunctionReturn(0);
}

int MatFetiScatterBc(Mat A, Vec ucg, FetiScatterMode mode)  /* uc_ass/ucg here; (same comm as fc_ass) */
{
    PetscFunctionBegin;

    Mat_Feti * const matfeti=(Mat_Feti*)A->data;
    PetscScalar * contrib_array;

    if(mode==FETI_SCATTER_FORWARD_ADD)  /* uc-->uc_ass  (domains[0].uc-->matfeti->uc_ass) */
    {   
	VecGetArray(matfeti->BcT_contrib,&contrib_array);    /* length must have been set */

	for(int i=0;i<matfeti->n_dom;i++)
	{
	    VecGetArray(matfeti->domains[i].uc,&matfeti->domains[i].uc_array);
	}

	for(int j=0;j<matfeti->BcT_scatter_len && matfeti->BcT_scatter_src[j]!=-1;j++)
	{  
	    WARN_IF(matfeti->BcT_scatter_src[j]==-1,"There seem to be unnecessary (zero?) entries in Bc.");
	    PetscScalar result;
	    result=matfeti->domains[matfeti->BcT_scatter_src_domain[j]].uc_array[matfeti->BcT_scatter_src[j]];
	    result*=matfeti->BcT_scatter_scale[j];           
	    contrib_array[j]=result;

	}	    

		    /* VecScatter */
	VecRestoreArray(matfeti->BcT_contrib,&contrib_array);
	VecScatterBegin(matfeti->BcT_scatter,matfeti->BcT_contrib,ucg,ADD_VALUES,SCATTER_FORWARD);
	VecScatterEnd  (matfeti->BcT_scatter,matfeti->BcT_contrib,ucg,ADD_VALUES,SCATTER_FORWARD);

    }
    else
    {
	if(mode==FETI_SCATTER_REVERSE_INSERT) /* uc_ass-->uc */
	{   

	    PetscScalar const zero=0;
	    for(int i=0;i<matfeti->n_dom;i++)
	    {
              VecSet(matfeti->domains[i].uc,zero);
		VecGetArray(matfeti->domains[i].uc,&matfeti->domains[i].uc_array);
	    }

	            /* VecScatter */
	    VecScatterBegin(matfeti->BcT_scatter,ucg,matfeti->BcT_contrib,INSERT_VALUES,SCATTER_REVERSE);
	    VecScatterEnd  (matfeti->BcT_scatter,ucg,matfeti->BcT_contrib,INSERT_VALUES,SCATTER_REVERSE);

	    VecGetArray(matfeti->BcT_contrib,&contrib_array);

	    for(int j=0;j<matfeti->BcT_scatter_len && matfeti->BcT_scatter_src[j]!=-1;j++)
	    {
		PetscScalar result;
		result=contrib_array[j];
		result*=matfeti->BcT_scatter_scale[j];  
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

           /*  internal use; called by MatLoad; provides space for coarse-solve aso.  */
int MatFetiSetUpTemporarySpace(Mat A)
{

    Mat_Feti * const matfeti=(Mat_Feti*)A->data;    

    PetscFunctionBegin;

    for(int i=0;i<matfeti->n_dom;i++)
    {
	FetiDomain * const domain=&matfeti->domains[i];
	VecCreateSeq(PETSC_COMM_SELF,domain->Krr->M     ,&domain->ur_tmp1);
	VecCreateSeq(PETSC_COMM_SELF,domain->Krr->M     ,&domain->ur_tmp2);
	VecCreateSeq(PETSC_COMM_SELF,matfeti->Kcc_ass->M,&domain->ucg_tmp1);  
	VecCreateSeq(PETSC_COMM_SELF,matfeti->Kcc_ass->M,&domain->ucg_tmp2);

	if(domain->use_Q)
	    VecCreateSeq(PETSC_COMM_SELF,domain->Q->N,&domain->mu_tmp); 

    }
    VecDuplicate(matfeti->fc_ass,&matfeti->uc_ass_tmp);                  /* MPI */
    VecCreateSeq(PETSC_COMM_SELF,matfeti->Kcc_ass->M,&matfeti->uc_tmp1); /* Seq */
    VecCreateSeq(PETSC_COMM_SELF,matfeti->Kcc_ass->M,&matfeti->uc_tmp2);

    if(matfeti->domains->use_Q)
    {
	VecCreateSeq(PETSC_COMM_SELF,matfeti->Kcc_ass->M + matfeti->domains->Q->N,&matfeti->ucmu_tmp1); 
	VecCreateSeq(PETSC_COMM_SELF,matfeti->Kcc_ass->M + matfeti->domains->Q->N,&matfeti->ucmu_tmp2); 
    }

    PetscFunctionReturn(0);
}

int MatFetiDestroyTemporarySpace(Mat A)
{
    Mat_Feti * const matfeti=(Mat_Feti*)A->data;    
    PetscFunctionBegin;

    for(int i=0;i<matfeti->n_dom;i++)
    {
	FetiDomain * const domain=&matfeti->domains[i];
	VecDestroy(domain->ur_tmp1);
	VecDestroy(domain->ur_tmp2);
	VecDestroy(domain->ucg_tmp1);  
	VecDestroy(domain->ucg_tmp2);

	if(domain->use_Q)
	   VecDestroy(domain->mu_tmp); 

    }

    VecDestroy(matfeti->uc_ass_tmp);
    VecDestroy(matfeti->uc_tmp1);
    VecDestroy(matfeti->uc_tmp2);

    if(matfeti->domains->use_Q)
    {
	VecDestroy(matfeti->ucmu_tmp1); 
	VecDestroy(matfeti->ucmu_tmp2); 
    }

    PetscFunctionReturn(0);

}

           /* for debugging purposes only; obsolete; Petsc-ExplicitOperator functions do the same */
int AssembleSystemMatrix(Mat A, Mat *system_matrix)
{
    Mat_Feti * const matfeti=(Mat_Feti*)A->data;
    const int N=matfeti->lambda_copy->N;
    int low,high;
    const PetscScalar zero=0;
    Vec e,dst;
    PetscScalar *dst_array;

    PetscFunctionBegin;

    MatCreate(PETSC_COMM_WORLD,system_matrix);
    MatSetSizes(*system_matrix,PETSC_DECIDE,PETSC_DECIDE,N,N);
    MatSetType(*system_matrix,MATMPIAIJ);

    VecDuplicate(matfeti->lambda_copy,&e);
    VecDuplicate(matfeti->lambda_copy,&dst);

    VecSet(e,zero);

    for(int i=0;i<N;i++)
    {
	PetscPrintf(PETSC_COMM_WORLD,"%d ",i);
	VecSetValue(e,i,1.,INSERT_VALUES);
	VecAssemblyBegin(e);
	VecAssemblyEnd(e);

	MatMult_Feti(A,e,dst);

	VecGetOwnershipRange(dst,&low,&high);
	VecGetArray(dst,&dst_array);
	for(int j=0;j<high-low;j++)
	{
	    if(dst_array[j]!=0.)
	    {
		MatSetValue(*system_matrix,low+j,i,dst_array[j],INSERT_VALUES);
	    }
	}
	VecRestoreArray(dst,&dst_array);

	VecSetValue(e,i,0.,INSERT_VALUES);
	VecAssemblyBegin(e);
	VecAssemblyEnd(e);

    }
    MatAssemblyBegin(*system_matrix,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd  (*system_matrix,MAT_FINAL_ASSEMBLY);

    MatlabInspect(*system_matrix,"system_matrix",4711);

    PetscFunctionReturn(0);
}

           /* the RHS is of the size of lambda and should therefore have the same parallel layout as lambda:

              In the KSP the initial residual is r=d_rhs-A*lambda, so it should have the same layout as lambda;
              the user must take care of this */

           /* fScc = fc - \sum BcT KrcT Krrinv fr   must be assembled over all processors!! */
           /* this is dr_Scc the rhs for lambda */

int MatFetiCalculatefScc(Mat A)   /* calculates fScc and leaves it as sequential Vec in matfeti->fScc */
{
    Mat_Feti * const matfeti=(Mat_Feti*)A->data;      
    PetscScalar const minus_one=-1;
    int its;

    PetscFunctionBegin;

    VecDuplicate(matfeti->fc_ass,&matfeti->fScc_ass);
    VecCopy     (matfeti->fc_ass, matfeti->fScc_ass);

    /* First we have to calculate fScc */
    for(int i=0;i<matfeti->n_dom;i++)
    {
	FetiDomain * const domain=&matfeti->domains[i];    

	SLESSolve        (domain->Krrinv,
			  domain->fr,
			  domain->ur_tmp1,&its);

	MatMultTranspose (domain->Krc,
			  domain->ur_tmp1,
			  domain->uc);

	VecScale(domain->uc,minus_one);
    }

#if 1  /* both alternatives work; maybe this is the smarter way to do the scatter? Bc is not very big... */
    MatFetiScatterBc (A,
		      matfeti->fScc_ass,   
		      FETI_SCATTER_FORWARD_ADD);
#else
    for(int i=0;i<matfeti->n_dom;i++)
    {
	FetiDomain * const domain=&matfeti->domains[i];    

	MatMultTranspose (domain->Bc,        /* this is the slow but the safe way: actually performing Bc*ucl */
			  domain->uc,
			  domain->ucg_tmp1); 

	FetiVecAddValues (domain->ucg_tmp1,      
		      matfeti->fScc_ass);
    }
    VecAssemblyBegin       (matfeti->fScc_ass);  
    VecAssemblyEnd         (matfeti->fScc_ass);
#endif

    if(matfeti->fScc)
	VecDestroy(matfeti->fScc);
    VecConvertMPIToSeqAll  (matfeti->fScc_ass,
			    &matfeti->fScc);

#if 0
    MatlabInspectVec(matfeti->fScc);  
#endif

}

int MatFetiCalculateRHS(Mat A, Vec * dr_Scc)   /* dr with additional correction from Scc */
{
    int its;
    Mat_Feti * const matfeti=(Mat_Feti*)A->data;      
    PetscScalar const zero=0, minus_one=-1;

    PetscFunctionBegin;

    PetscSynchronizedPrintf(PETSC_COMM_WORLD,"Calculating RHS...");

    ASSERT(matfeti->lambda_copy,"Need MatLoad first: lambda_copy==0");
    VecDuplicate(matfeti->lambda_copy,dr_Scc);  /* give dr_Scc the same layout as lambda; does not copy values */
    VecSet(*dr_Scc,zero);                     

    MatFetiCalculatefScc(A);   /* calculates fScc and leaves it as sequential Vec in matfeti->fScc */

    /* --- okay, we've got fScc now; what we really want ist dr_Scc, so we continue --- */

    /* first calculate dr; dr and dr_Scc have the same size as lambda

       originally I factored out Br and also Krrinv so I get
       dr_Scc=Br Krrinv (fr - Krc Bc Sccinv fScc) 

       now with Q and mu we get
       dr = \sum Br Krrinv fr      and
       dr_Scc = dr - \sum Br Krrinv [Krc Bc , QT] Sccinv [fScc ; \sum - QT Krrinv fr ]) 

       okay so we calculate  Krrinv fr  and scatter three times for dr,
       then for QT and last for the sum again

    */

    for(int i=0;i<matfeti->n_dom;i++)
    {
	FetiDomain * const domain=&matfeti->domains[i];
	/* first calculate  Krrinv fr */

	SLESSolve      (domain->Krrinv,
			domain->fr,
			domain->ur,&its);

	{
	    int reason=0;
	    KSP ksp;
	    SLESGetKSP(domain->Krrinv,&ksp);
	    KSPGetConvergedReason(ksp,(KSPConvergedReason*)&reason);

	    WARN_IF((reason!=2 && reason!=3)&&(reason==4&&its!=1),"RHS: Fetidp local solve convergence is bad.");
	}

    }

    VecSet(*dr_Scc,zero);
    MatFetiScatter  (A,                           /* apply Br */
		     *dr_Scc,                     /* ur --> dr_Scc */
		     FETI_SCATTER_FORWARD_ADD);   /* note: *dr_Scc is an mpi-vector */
    /* okay, now we have dr in dr_Scc; this is a MPI-Vec of the size of lambda */

    if(matfeti->domains->use_Q)
    {
	VecSet(matfeti->mu,zero);
	for(int i=0;i<matfeti->n_dom;i++)
	{
	    FetiDomain * const domain=&matfeti->domains[i];

	    MatMultTranspose(domain->Q,
			     domain->ur,          /* we still have Krrinv fr in domain->ur */
			     domain->mu_tmp);

	    FetiVecAddValues(domain->mu_tmp,
			 matfeti->mu);

	}

	VecAssemblyBegin(matfeti->mu);    /* Assemble MPI-mu */
	VecAssemblyEnd  (matfeti->mu);

	VecScale(matfeti->mu,minus_one); /* - \sum Q Krrinv fr */

	if(matfeti->mu_seq)
	    VecDestroy(matfeti->mu_seq);
	VecConvertMPIToSeqAll(matfeti->mu,&matfeti->mu_seq);
	/* okay, now we have mu */

	FetiVecAppend(matfeti->fScc,matfeti->mu_seq,matfeti->ucmu_tmp1);

	SLESSolve          (matfeti->Sccinv,     
			    matfeti->ucmu_tmp1,
			    matfeti->ucmu_tmp2,&its); 

	FetiVecSplit(matfeti->ucmu_tmp2,matfeti->uc_tmp1,matfeti->mu_seq);

    }
    else
    {
    SLESSolve      (matfeti->Sccinv,
		    matfeti->fScc,
		    matfeti->uc_tmp1,&its);
    }

    for(int i=0;i<matfeti->n_dom;i++)
    {
	FetiDomain * const domain=&matfeti->domains[i];

	if(domain->use_Q)
	{
	    MatMult(domain->Q,
		    matfeti->mu_seq,
		    domain->ur_tmp1);        /* the contribution from mu for later */
	}
	else
	{
	    VecSet(domain->ur_tmp1,zero);   /* no contribution */
	}

	MatMult        (domain->Bc,          
			matfeti->uc_tmp1,       
			domain->uc);   

	MatMultAdd     (domain->Krc,    
			domain->uc,
			domain->ur_tmp1,     /* already holds the contribution from mu if use_Q */
			domain->ur_tmp1);    /* ur_tmp1 = Krc uc + ur_tmp1 */

	SLESSolve      (domain->Krrinv,
			domain->ur_tmp1,
			domain->ur,&its);	

	VecScale       (                     /* minus somewhere else could be faster */
			domain->ur,minus_one);

#if 0
	MatlabInspectVecs(domain->ur,i); 

#endif

    }

    /*  note that MatFetiScatter can only be called 
        when all domain contributions are calculated  */

    MatFetiScatter  (A,                           /* apply Br */
		     *dr_Scc,                     /* ur --> dr_Scc */
		     FETI_SCATTER_FORWARD_ADD);   /* note: *dr_Scc is an mpi-vector */

#if 0
    PetscPrintf(PETSC_COMM_WORLD,"Writing dr_Scc\n");
    MatlabInspectVec(*dr_Scc);            /*  *dr_Scc seems to be correct now also  */
#endif

    PetscSynchronizedPrintf(PETSC_COMM_WORLD,"...RHS finished. ");

    PetscFunctionReturn(0);
}

/*  Make Kcc a MPI-Matrix and put the values there by MatSetValues 
    (only nonzero) then convert it to Seq-Matrix (P. 58 manual)     */

int MatSetUp_Feti(Mat A)   /*  MatLoad_Feti calls this  */
{
    static PetscTruth matsetup_called=PETSC_FALSE;

    PetscFunctionBegin;

    ASSERT(!matsetup_called,"MatSetUp_Feti was already called.");
    matsetup_called=PETSC_TRUE;

    Mat_Feti * const matfeti=(Mat_Feti*)A->data;
    PetscSynchronizedPrintf(PETSC_COMM_WORLD,"Entering Setup... ");

    MatFetiCreateScatter(A);      /* transform BrT into the two scatters */
    MatFetiCreateScatterBc(A);    /* transform Bc into the two scatters */

    MatFetiSetUpTemporarySpace(A);

    MatFetiSolveLocal(A);         /* sets up the SLES */
    MatFetiSetUpCoarse(A);   
    PetscSynchronizedPrintf(PETSC_COMM_WORLD,"...Setup finished. ");

    PetscFunctionReturn(0);
}

           /* it leaves the multiplicity in domain->ur_multiplicity                       */
int MatFetiCalculateMultiplicity(Mat A)
{
    Mat_Feti * const matfeti=(Mat_Feti*)A->data;
    const PetscScalar one=1;
    Vec lam;
    int llen;

    PetscFunctionBegin;

    llen=matfeti->domains->BrT->N;
    VecCreateSeq(PETSC_COMM_SELF,llen,&lam);                   /* all BrT have the same N */

    for(int i=0;i<matfeti->n_dom;i++)
    {
	FetiDomain * const domain = &(matfeti->domains[i]);

	if(!domain->ur_multiplicity)
	    VecDuplicate(domain->ur,&domain->ur_multiplicity);

	VecSet(domain->ur_tmp1,one);                          /*  11...1  */

	MatMultTranspose(domain->BrT,domain->ur_tmp1,lam);     /*  B_1^T * B_1 * (11...1)^T   */
	MatMult(domain->BrT,lam,domain->ur_multiplicity);             

	VecAXPY(domain->ur_multiplicity,one,domain->ur_tmp1); /*  ur_multiplicity=1*ur_tmp1+ur_multiplicity */

    }
    VecDestroy(lam);

    PetscFunctionReturn(0);
}

int MatDestroy_Feti(Mat A)   /* destroy domains and scatters, works with MatDestroy */
{
    Mat_Feti * const matfeti=(Mat_Feti*)A->data;

    PetscFunctionBegin;

    /* domains */
    for(int i=0;i<matfeti->n_dom;i++)
    {
	FetiDomain * const domain = &(matfeti->domains[i]); 
#if 0
	MatlabInspect(domain->BrT,"BrT",i); 
#endif
	MatDestroy(domain->Krr);
	MatDestroy(domain->Krc);
	MatDestroy(domain->BrT);
	MatDestroy(domain->Bc);
	SLESDestroy(domain->Krrinv);

	VecDestroy(domain->ur);
	VecDestroy(domain->uc);
	VecDestroy(domain->fr);
	if(domain->ur_multiplicity)
	    VecDestroy(domain->ur_multiplicity);
	if(domain->use_Q)
	    MatDestroy(domain->Q);

    }

    MatFetiDestroyTemporarySpace(A);

    /* global */
    MatDestroy(matfeti->Kcc_ass);
    VecDestroy(matfeti->fc_ass);

    if(matfeti->Scc_ass)
	MatDestroy(matfeti->Scc_ass);  
    VecDestroy(matfeti->fScc_ass);
    SLESDestroy(matfeti->Sccinv);

    MatDestroy(matfeti->Scc); 
    VecDestroy(matfeti->fScc);

    /* Scatter */
    VecDestroy(matfeti->contrib);
    PetscFree(matfeti->scatter_src_domain);
    PetscFree(matfeti->scatter_src);
    PetscFree(matfeti->scatter_scale);
    VecScatterDestroy(matfeti->Br_scatter);

    /* Bc Scatter */
    VecDestroy(matfeti->BcT_contrib);
    PetscFree(matfeti->BcT_scatter_src_domain);
    PetscFree(matfeti->BcT_scatter_src);
    PetscFree(matfeti->BcT_scatter_scale);
    VecScatterDestroy(matfeti->BcT_scatter);

    if(matfeti->domains->use_Q)
    {
	VecDestroy(matfeti->mu);
	VecDestroy(matfeti->mu_seq);
    }

    PetscFree(matfeti->domains);  /* point of no return */
    PetscFree(matfeti);

    PetscFunctionReturn(0);

}

int MatlabWrite_lambda(Mat A) 
{
    PetscFunctionBegin;

    Mat_Feti * const matfeti=(Mat_Feti*)A->data;

    PetscViewer file;
    PetscViewerASCIIOpen(PETSC_COMM_WORLD,"Petsc_lambda.m",&file); 
    PetscViewerSetFormat(file,PETSC_VIEWER_ASCII_MATLAB);
    PetscObjectSetName((PetscObject)matfeti->lambda_copy,"Petsc_lambda"); /* name for matlab */

    VecView(matfeti->lambda_copy,file);
    PetscViewerDestroy(file);

    PetscFunctionReturn(0);
}

/* Caveat: Dense matrices are only written with 5 significant digits => errors of 1e-4 */
/*         Sparse Matrices with 12                                                     */
int MatlabInspect(Mat mat, char const * const name, int const domain_num)
{
    PetscFunctionBegin;

    PetscSynchronizedFlush(PETSC_COMM_WORLD); /*  flush beforehand  */
    char fname[256]={};

    int rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);  /*  not ((PetscObject)mat)->comm; all processors would write to the same file
                                                 if mat was sequential; now if mat is mpi, only the first 
						 processor will have a nonempty file  */

    PetscViewer viewer;

    sprintf(fname,"Petsc_%s_%d__%d.m",name,domain_num,rank);
    PetscSynchronizedPrintf(PETSC_COMM_WORLD,"Writing %s (%d)\n",fname,rank); 

    PetscSynchronizedFlush(PETSC_COMM_WORLD);  /* PetscSynchronizedPrintf needs flush */
    PetscViewerASCIIOpen(((PetscObject)mat)->comm,fname,&viewer);
    PetscViewerSetFormat(viewer,PETSC_VIEWER_ASCII_MATLAB);

    char oname[256]={};
    sprintf(oname,"Petsc_%s_%d__%d",name,domain_num,rank);
    PetscObjectSetName((PetscObject)mat,oname);

    MatView(mat,viewer);

    PetscSynchronizedFlush(PETSC_COMM_WORLD);  /* PetscSynchronizedPrintf needs flush */
    PetscViewerDestroy(viewer);

    PetscFunctionReturn(0);
}

int MatFetiMatlabInspect(Mat A)                /* write all data to files; not used now */
{
    Mat_Feti * const matfeti=(Mat_Feti*)A->data;    
    char fname[256]={};
    int rank;

    PetscFunctionBegin;
    MPI_Comm_rank(((PetscObject)A)->comm, &rank);             /* ((PetscObject)A)->comm ist PETSC_COMM_WORLD anyway */
#if 1
    PetscViewer viewer;

    char name[]="Scc";
    Mat mat=matfeti->Scc;

    sprintf(fname,"Petsc_%s__%d.m",name,rank);
    PetscSynchronizedPrintf(PETSC_COMM_WORLD,"Writing %s (%d)\n",fname,rank); 

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
	PetscViewerDestroy(viewer);
    }
#endif
    PetscSynchronizedFlush(PETSC_COMM_WORLD); 
    PetscViewerDestroy(viewer);
    PetscFunctionReturn(0);
}

int MatlabInspectVec(Vec v)
{
    PetscFunctionBegin;

    MatlabInspectVecs(v,-1); 

    PetscFunctionReturn(0);
}

int MatlabInspectVecs(Vec v, int dom)
{

    char fname[256]={};
    int rank; 

    PetscFunctionBegin;

    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    char name[]="vecinspect";
    char domnum[256]={};      /* this way domnum can be an empty string */
    if(dom>=0)
	sprintf(domnum,"%d_",dom);

    sprintf(fname,"Petsc_%s_%s_%d.m",name,domnum,rank);

    PetscViewer viewer;

    PetscViewerASCIIOpen(((PetscObject)v)->comm,fname,&viewer);
    PetscViewerSetFormat(viewer,PETSC_VIEWER_ASCII_MATLAB);

    char oname[256]={};
    sprintf(oname,"Petsc_%s_%s_%d",name,domnum,rank);
    PetscObjectSetName((PetscObject)v,oname);

    VecView(v,viewer);
    PetscViewerDestroy(viewer);

    PetscFunctionReturn(0);
}

int MatlabPut(PetscObject A) /* obsolete */
    {
/*
    PetscMatlabEngine *matlab;
    PetscMatlabEngineCreate(PETSC_COMM_WORLD,PETSC_NULL,matlab);
    PetscMatlabEnginePut(PETSC_MATLAB_ENGINE_WORLD,A);
    PETSC_MATLAB_ENGINE_WORLD
    PetscMatlabEngine PETSC_MATLAB_ENGINE_(MPI_Comm comm)
    int PetscMatlabEngineEvaluate(PetscMatlabEngine mengine,char *string,...)
    PetscMatlabEngineDestroy(matlab);
*/
    }

int MatFetiScatterTest(Mat A,Vec lambda)          /* calculates lambda=lambda+Br*BrT*lambda */
{
    PetscFunctionBegin;

    MatFetiScatter   (A,                           /* apply all Br */
		      lambda,                      /* src_lambda ---> all ur */
		      FETI_SCATTER_REVERSE_INSERT);  

    MatFetiScatter   (A,                           /* apply BrT */
		      lambda,                      /* src_lambda ---> all ur */
		      FETI_SCATTER_FORWARD_ADD);     

    PetscFunctionReturn(0);
}

int MatFetiCalculateRhoNeighbor(Mat A)
{
    Mat_Feti * const matfeti=(Mat_Feti*)A->data;
    const PetscScalar one=1;
    const PetscScalar zero=0;
    Vec lambda;                                    /* temporary space */
    Vec contrib_tmp;
    PetscScalar * contrib_array;
    PetscScalar * contrib_tmp_array;

    PetscFunctionBegin;

    ASSERT(matfeti->lambda_copy,"MatFetiCalculateRhoNeighbor needs lambda_copy as template for temp-space");
    VecDuplicate(matfeti->lambda_copy,&lambda);
    VecSet(lambda,zero);

    for(int i=0;i<matfeti->n_dom;i++)              /* set all to the same rho */
    {
	FetiDomain * const domain = &(matfeti->domains[i]);
	VecSet(domain->ur,domain->my_rho);
    }

    MatFetiScatter   (A,                           /* apply BrT */
		      lambda,                      /* all ur --->(add) lambda */
		      FETI_SCATTER_FORWARD_ADD);     

    /* now we have all the coefficient-differences in lambda
       and all the coefficients still in matfeti->contrib    */

    VecDuplicate(matfeti->contrib,&contrib_tmp);    /* save them; cave: they are scaled */
    VecCopy     (matfeti->contrib,contrib_tmp);

    MatFetiScatter   (A,                            /* apply all Br */
		      lambda,                       /* lambda ---> all ur */
		      FETI_SCATTER_REVERSE_INSERT);      

    /* now we have the differences back in ur; but more importantly in matfeti->contrib */

    VecGetArray (contrib_tmp,&contrib_tmp_array);
    VecGetArray (matfeti->contrib,&contrib_array);

    for(int i=0;i<matfeti->scatter_len;i++)  /* restore neighbors coeff. from the difference and own coeff. */
    {   /* b = -(-(-a+b)) + |-a|     and     b = -(+(a-b)) + |a| */
	PetscScalar a=matfeti->scatter_scale[i];
	PetscScalar b=contrib_array[i];
	PetscScalar c=contrib_tmp_array[i];  
	contrib_tmp_array[i]=-a*b+a*c;

    }

    VecRestoreArray (contrib_tmp,&contrib_tmp_array);
    VecRestoreArray (matfeti->contrib,&contrib_array);

    /* get rid of memory */
    VecDestroy(lambda);

    /* now I have rho from the neighbor in contrib_tmp */
    VecCopy(contrib_tmp,matfeti->contrib);          /* save it in contrib */
    VecDestroy(contrib_tmp);
}

int MatFetiCalculateRhoSum(Mat A)
{
    Mat_Feti * const matfeti=(Mat_Feti*)A->data;
    const PetscScalar minus_one=-1;
    const PetscScalar minus_two=-2;
    const PetscScalar zero=0;
    const PetscScalar one=1;

    PetscScalar * scale;
    Vec lambda;                                      /* temporary space */
    PetscScalar * contrib_array;

    Vec * ur_rhos;                                   /* temporary space for ur */

    ASSERT(matfeti->lambda_copy,"MatFetiCalculateRhoSum needs lambda_copy as template for temp-space");
    VecDuplicate(matfeti->lambda_copy,&lambda);
    VecSet(lambda,zero);

    PetscMalloc(matfeti->scatter_len*sizeof(PetscScalar),&scale);        /* save scatter_scale */
#if 0
    PetscMemcpy(matfeti->scatter_scale,scale,matfeti->scatter_len*sizeof(PetscScalar));
#else
    for(int i=0;i<matfeti->scatter_len;i++) scale[i]=matfeti->scatter_scale[i];
#endif

    for(int i=0;i<matfeti->scatter_len;i++) matfeti->scatter_scale[i]=1; /* all will be added */

    for(int i=0;i<matfeti->n_dom;i++)                /* set all to the same rho */
    {
	FetiDomain * const domain = &(matfeti->domains[i]);
	VecSet(domain->ur,domain->my_rho);
    }

    /* save own rho for later */
    PetscMalloc((matfeti->n_dom)*sizeof(Vec),&ur_rhos);    /* unfortunatly need temp-space */
    for(int i=0;i<matfeti->n_dom;i++)                      /* for psychological reasons a separate loop */
    {
	FetiDomain * const domain = &(matfeti->domains[i]);
	VecDuplicate(domain->ur,&ur_rhos[i]);
	VecCopy(domain->ur,ur_rhos[i]);              /* save it for later */
    }

    MatFetiScatter   (A,                             /* apply BrT */
		      lambda,                        /* all ur --->(add) lambda */
		      FETI_SCATTER_FORWARD_ADD);     

    /* now we have in contrib all own rhos, but we will use the ur_rhos */

    MatFetiScatter   (A,                             /* apply all Br */
		      lambda,                        /* lambda ---> all ur */
		      FETI_SCATTER_REVERSE_INSERT);      

    /* now we have in contrib all added pairs of rhos */
    /* we have all the sums in domain->ur             */

    /* we will use these sums that are now in domain->ur                 */
    /* to extract the sums we want by using the multiplicity and own rho */

    if(!matfeti->domains->ur_multiplicity)           /* if not done yet, do it now */
	MatFetiCalculateMultiplicity(A);

    for(int i=0;i<matfeti->n_dom;i++)               
    {
	FetiDomain * const domain = &(matfeti->domains[i]);
	Vec ur_m;                                    /* multiplicity - 2 */
	VecDuplicate(domain->ur_multiplicity,&ur_m);
	VecSet(ur_m,minus_two);
	VecAXPY(ur_m,one,domain->ur_multiplicity);  /* ur_m=1*ur_multiplicity+ur_m */

	VecPointwiseMult(ur_rhos[i],
                         ur_m,
			 ur_rhos[i]);                /* ur_rhos[i]=ur_m*ur_rhos[i] */

	VecAXPY(domain->ur,minus_one,ur_rhos[i]);    /* rho_sum=rho_sum_all-(multiplicity-2)*my_rho

	/* now we have the desired sums in domain->ur */

	VecCopy(domain->ur,domain->ur_multiplicity); /* save it; destroy multiplicity */
	VecDestroy(ur_m);
    }

    for(int i=0;i<matfeti->n_dom;i++)                /* get rid of temporary space */
    {
	VecDestroy(ur_rhos[i]);
    }
    PetscFree(ur_rhos);

    VecDestroy(lambda);

    /* restore the standard behavior for MatFetiScatter */

#if 0
    PetscMemcpy(scale,matfeti->scatter_scale,matfeti->scatter_len*sizeof(PetscScalar));
#else
    for(int i=0;i<matfeti->scatter_len;i++) matfeti->scatter_scale[i]=scale[i];
#endif
    PetscFree(scale);

}

int MatFetiCalculateRhoScaling(Mat A)
{
    Mat_Feti * const matfeti=(Mat_Feti*)A->data;
    PetscScalar * contrib_array;

    MatFetiCalculateRhoSum(A);          /* leaves result in ur_multiplicity */

    MatFetiCalculateRhoNeighbor(A);     /* leaves result in contrib */

    /* get the arrays themselves */
    VecGetArray(matfeti->contrib,&contrib_array);
    for(int i=0;i<matfeti->n_dom;i++)
    {
	VecGetArray(matfeti->domains[i].ur_multiplicity,&matfeti->domains[i].ur_array);  /* misuse of ur_array */
    }

    /* now  calculate the new scaling rho_neighbor/rho_sum */
    for(int j=0;j<matfeti->scatter_len && matfeti->scatter_src[j]!=-1;j++)
    {
	PetscScalar result;
	result=matfeti->domains[matfeti->scatter_src_domain[j]].ur_array[matfeti->scatter_src[j]];
	result=(contrib_array[j]/result);     /* rho_scale=rho_neighbor/rho_sum */

	matfeti->scatter_scale[j]*=result;    /* scale +1/-1 by the rho_scale   */
    }

    VecRestoreArray(matfeti->contrib,&contrib_array);
    for(int i=0;i<matfeti->n_dom;i++)
    {
	VecRestoreArray(matfeti->domains[i].ur_multiplicity,&matfeti->domains[i].ur_array);  

	VecDestroy(matfeti->domains[i].ur_multiplicity);
	matfeti->domains[i].ur_multiplicity=0;   
    }

}

/*  ---------------------------------   functions for internal use   ---------------------------------  */
/* has to read K_1.mat; f_1.mat; Kcc_1; Krr_1; Krc_1; BrT_1; Bc_1; rho; Future Q */
int FetiLoadMatSeq(char const * const prefix, char const * const name, char const * const postfix, Mat* A)
{
    int ierr;
    char fname[256]={};
    PetscViewer viewer;

    PetscFunctionBegin;
    strcat(fname,prefix);
    strcat(fname,name);
    strcat(fname,postfix);

    ierr=PetscViewerBinaryOpen(PETSC_COMM_SELF, fname,PETSC_BINARY_RDONLY,&viewer);CHKERRQ(ierr);
         /* needs comm */
    ierr=MatLoad(viewer,MATSEQAIJ,A);CHKERRQ(ierr);  
    /* MATSEQAIJ must get a communicator with one processor only; so PETSC_COMM_SELF */
    PetscViewerDestroy(viewer);

    PetscFunctionReturn(0);

}

int FetiLoadMatMPI(char const * const prefix, char const * const name, char const * const postfix, Mat* A)
{
    PetscFunctionBegin;
    char fname[256]={};
    PetscViewer viewer;
    strcat(fname,prefix);
    strcat(fname,name);
    strcat(fname,postfix);
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, fname,PETSC_BINARY_RDONLY,&viewer); /* needs comm */
    MatLoad(viewer,MATMPIAIJ,A);  /* MATSEQAIJ */
    PetscViewerDestroy(viewer);
    PetscFunctionReturn(0);
}  

int FetiLoadVecSeq(char const * const prefix, char const * const name, char const * const postfix, Vec* v)
{
    char fname[256]={};
    PetscViewer viewer;
    PetscFunctionBegin;
    strcat(fname,prefix);
    strcat(fname,name);
    strcat(fname,postfix);
    PetscViewerBinaryOpen(PETSC_COMM_SELF, fname,PETSC_BINARY_RDONLY,&viewer);
    VecLoad(viewer,v);    
    PetscViewerDestroy(viewer);
    PetscFunctionReturn(0);
}

int FetiLoadVecMPI(char const * const prefix, char const * const name, char const * const postfix, Vec* v)
{
    PetscFunctionBegin;
    char fname[256]={};
    PetscViewer viewer;
    strcat(fname,prefix);
    strcat(fname,name);
    strcat(fname,postfix);
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, fname,PETSC_BINARY_RDONLY,&viewer);
    VecLoad(viewer,v);    
    PetscViewerDestroy(viewer);
    PetscFunctionReturn(0);
}

int FetiDomainLoad(FetiDomain * domain, const char * const prefix) /* Get all matrices from files */
{
    char num[8]={0};
    Vec rho;  /* right now one rho per domain; */
    PetscScalar *rho_array;

    PetscFunctionBegin;

    sprintf(num,"_%d.mat",domain->domain_num);           /* domain_num should start at 1 */ 
    FetiLoadMatSeq(prefix,FETIDP_KRR,num,&(domain->Krr)); 
    FetiLoadMatSeq(prefix,FETIDP_KRC,num,&(domain->Krc));
    FetiLoadVecSeq(prefix,FETIDP_FR,num,&(domain->fr));

    FetiLoadMatSeq(prefix,FETIDP_BRT,num,&(domain->BrT));

    FetiLoadMatSeq(prefix,FETIDP_BC,num,&(domain->Bc)); 

    if(domain->use_Q)
	FetiLoadMatSeq(prefix,FETIDP_Q,num,&(domain->Q)); 

    FetiLoadVecSeq(prefix,FETIDP_RHO,".mat",&rho); /* all processors read from the same file; anyway... */

    VecGetArray(rho,&rho_array);

    domain->my_rho=rho_array[domain->domain_num-1]; /* set my own rho */
            /* filenames of domains start at 1; here we start at zero */

    VecRestoreArray(rho,&rho_array);
    VecDestroy(rho);

    VecCreateSeq(PETSC_COMM_SELF,domain->Krr->M,&domain->ur);  /* set ur to the right size 
                                                                  Krr->M should be the right choice */
    domain->ur_multiplicity=0;      /* flag it empty until multiplicity calculated */

    VecCreateSeq(PETSC_COMM_SELF,domain->Krc->N,&domain->uc); 

    PetscFunctionReturn(0);
}
