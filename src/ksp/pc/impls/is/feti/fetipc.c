#define PETSCKSP_DLL

#include "fetipc.h"
#include "src/mat/impls/feti/feti.h" 

int PETSCKSP_DLLEXPORT PCCreateFeti(PC *pc)
{
    PetscFunctionBegin;
    PCCreate(PETSC_COMM_WORLD,pc);                     /* call the base constructor */ 
    PCCreate_Feti(*pc);
    PetscFunctionReturn(0);
}

int PCCreate_Feti(PC pc)
{
    PC_Feti * pcfeti;

    PetscTruth flag;

    PetscFunctionBegin;

    PetscNew(PC_Feti,&pcfeti);
    PetscMemzero(pc->ops,sizeof(struct _PCOps));

    pc->ops->apply               = PCApply_Feti;
    pc->ops->applytranspose      = 0;
    pc->ops->destroy             = PCDestroy_Feti;     /* destructur is called by Petsc only if SetType is called;  
						          normally user has to call it before exit */
    pc->ops->setup               = 0;

    pc->data=(void *)pcfeti;

    PetscObjectChangeTypeName((PetscObject)pc,"pcfeti"); /* PCSetUp checks this; sets it to BJacobi if 0 */

    pcfeti->lumped=PETSC_FALSE;
    PetscOptionsGetLogical("pcfeti_","-lumped",&pcfeti->lumped,&flag);   /* -pcfeti_lumped "" yes true 1 on */

    PetscFunctionReturn(0);
} 

int PCDestroy_Feti(PC pc) /* Destroy PC, works with PCDestroy */
{
    PC_Feti * pcfeti;

    PetscFunctionBegin;

    pc->ops->destroy=0;               /* avoid second call through PCDestroy 
					 in case this one is called by user   */
    if(!pc) PetscFunctionReturn(0);

    pcfeti=(PC_Feti*)pc->data;
    PCFetiDestroyTemporarySpace(pc);

    for(int i=0;i<pcfeti->n_dom;i++)
    {
	PC_FetiDomain * pcdomain = &(pcfeti->pcdomains[i]);
	PCFetiDomainDestroy(pcdomain);
    }

    if(pcfeti->pcdomains)
	PetscFree(pcfeti->pcdomains);
    pcfeti->pcdomains=0;
    pc->data=0;
    PetscFree(pcfeti);  

    PetscFunctionReturn(0); 
}

int PCLoad_Feti(PC pc)
{
    PC_Feti * pcfeti;
    Mat_Feti * matfeti;

    PetscFunctionBegin;

    pcfeti=(PC_Feti*)pc->data;

    ASSERT(pc->mat,"Either PCFetiSetMatFeti, SLESSetOperators or PCSetOperators must be called before PCLoad_Feti\n");

    matfeti=(Mat_Feti *)pc->mat->data;

    if(!pcfeti->n_dom) /* PCSetOperators has been called, but not PCFetiSetMatFeti */
	PCFetiSetMatFeti(pc,pc->mat);

    ASSERT2(matfeti->n_dom==pcfeti->n_dom,"PCFETI n_dom not equal MATFETI n_dom",matfeti->n_dom,pcfeti->n_dom);

    for(int i=0; i<matfeti->n_dom;i++)
    {
	PCFetiDomainLoad(pcfeti->pcdomains+i,FETIDP_PREFIX);
    }

    PCFetiSetUpTemporarySpace(pc);
    PCFetiSetUp(pc);                 

    PetscFunctionReturn(0);
}

int PETSCKSP_DLLEXPORT PCFetiSetMatFeti(PC pc, Mat A) /* set pointer to MatFeti for access to B and copy some information    */
{                                  /* PCSetOperators also sets pc->mat: always called on SLESSetOperators */
    int dom_per_proc;
    PC_Feti * pcfeti=(PC_Feti*)pc->data;
    Mat_Feti * matfeti=(Mat_Feti*)A->data;

    PetscFunctionBegin;

    pc->mat=A;

              /* same as in MatCreateFeti */
    dom_per_proc=matfeti->n_dom;  

              /* get memory for pcdomains */
    PetscMalloc(dom_per_proc*sizeof(PC_FetiDomain),&pcfeti->pcdomains);
    PetscMemzero(pcfeti->pcdomains,dom_per_proc*sizeof(PC_FetiDomain));  /* just for safety */

    pcfeti->n_dom=matfeti->n_dom;

    for(int i=0;i<dom_per_proc;i++)
    {
	FetiDomain    * domain   = &(matfeti->domains[i]);
	PC_FetiDomain * pcdomain = &(pcfeti->pcdomains[i]);

	pcdomain->domain_num=domain->domain_num;                         /* copy local number */
    }

              /* calculate Multiplicity */
    if(!matfeti->domains->ur_multiplicity)
    {
	MatFetiCalculateMultiplicity(A);                     /* leaves it in ur_multiplicity */
    }

    for(int i=0;i<dom_per_proc;i++)
    {
	FetiDomain    * domain   = &(matfeti->domains[i]);
	PC_FetiDomain * pcdomain = &(pcfeti->pcdomains[i]);

	VecDuplicate(domain->ur_multiplicity,&pcdomain->D);  

	VecCopy     (domain->ur_multiplicity,pcdomain->D);

    }

    MatFetiCalculateRhoScaling(A);               /* do this also, since it will destroy ur_multiplicity */

    PetscFunctionReturn(0);

} 

int PETSCKSP_DLLEXPORT PCFetiDomainLoad(PC_FetiDomain *pcdomain,const char * prefix)
{
    char num[256]={};

    PetscFunctionBegin;

    sprintf(num,"_%d.mat",pcdomain->domain_num); /* domain_num should start at 1 */ 

    FetiLoadMatSeq(prefix,FETIDP_KBB,num,&(pcdomain->Kbb)); 
    FetiLoadMatSeq(prefix,FETIDP_KII,num,&(pcdomain->Kii));
    FetiLoadMatSeq(prefix,FETIDP_KIB,num,&(pcdomain->Kib));
    FetiLoadMatSeq(prefix,FETIDP_BBR,num,&(pcdomain->Bbr));

    PetscFunctionReturn(0);
}

int PETSCKSP_DLLEXPORT PCFetiDomainSetUp(PC_FetiDomain *pcdomain)
{
    PC pc;
    KSP ksp;
    SLES * sles=&(pcdomain->Kii_inv);

    PetscFunctionBegin;

    SLESCreate(PETSC_COMM_SELF,sles);
    SLESSetOperators(*sles,
		     pcdomain->Kii,
		     pcdomain->Kii,
		     DIFFERENT_NONZERO_PATTERN);

    SLESGetKSP(*sles,&ksp);
    SLESGetPC(*sles,&pc);

    KSPSetType(ksp,KSPPREONLY);
    PCSetType(pc,PCLU);

    KSPSetTolerances(ksp,1e-7,1e-50,1e+5,10000); 
    SLESAppendOptionsPrefix(*sles,"pcfeti_schur_");  /* uses options with prefix -pcfeti_ */
    SLESSetFromOptions(*sles);

    PetscFunctionReturn(0);
}

int PETSCKSP_DLLEXPORT PCFetiSetUp(PC pc)  /* setup SLES Kii_inv */
{
    PC_Feti * pcfeti=(PC_Feti*)pc->data;
    PetscFunctionBegin;
    for(int i=0;i<pcfeti->n_dom;i++)
    {
	PCFetiDomainSetUp(pcfeti->pcdomains+i);
    }
    PetscFunctionReturn(0);
}

int PETSCKSP_DLLEXPORT PCFetiSetUpTemporarySpace(PC pc)
{
    PC_Feti * pcfeti=(PC_Feti*)pc->data;

    PetscFunctionBegin;
    for(int i=0;i<pcfeti->n_dom;i++)
    {
	PC_FetiDomain * pcdomain=pcfeti->pcdomains+i;
	VecCreateSeq(PETSC_COMM_SELF,pcdomain->Bbr->M,&pcdomain->ubr_tmp1);
	VecCreateSeq(PETSC_COMM_SELF,pcdomain->Bbr->M,&pcdomain->ubr_tmp2);
	VecCreateSeq(PETSC_COMM_SELF,pcdomain->Kii->M,&pcdomain->uii_tmp1);
	VecCreateSeq(PETSC_COMM_SELF,pcdomain->Kii->M,&pcdomain->uii_tmp2);
    }
    PetscFunctionReturn(0);

}

int PETSCKSP_DLLEXPORT PCFetiDestroyTemporarySpace(PC pc)
{
    PC_Feti * pcfeti;

    PetscFunctionBegin;
    if(!pc) PetscFunctionReturn(0);

    pcfeti=(PC_Feti*)pc->data;

    if(!pcfeti->pcdomains) PetscFunctionReturn(0);
    for(int i=0;i<pcfeti->n_dom;i++)
    {
	PC_FetiDomain * pcdomain=pcfeti->pcdomains+i;
	VecDestroy(pcdomain->ubr_tmp1);
	VecDestroy(pcdomain->ubr_tmp2);
	VecDestroy(pcdomain->uii_tmp1);
	VecDestroy(pcdomain->uii_tmp2);
    }
    PetscFunctionReturn(0);
}

int PCApply_Feti(PC pc, Vec src_lambda, Vec dst_lambda)
{
    PC_Feti * pcfeti=(PC_Feti*)pc->data;
    Mat_Feti * matfeti=(Mat_Feti*)pc->mat->data;
    int its=0;

    PetscFunctionBegin;

    PetscScalar zero=0;
    PetscScalar minusone=-1;
    VecSet(dst_lambda,zero);

    MatFetiScatter   (pc->mat,                     /* apply all Br */
		      src_lambda,                      /* src_lambda ---> all ur */
		      (pcfeti->scaling==FETI_SCALING_RHO)
		      ?FETI_SCATTER_REVERSE_INSERT_SCALE
		      :FETI_SCATTER_REVERSE_INSERT);  

    for(int i=0;i<pcfeti->n_dom;i++)
    {
	PC_FetiDomain * pcdomain=pcfeti->pcdomains+i;
	FetiDomain * domain=matfeti->domains+i;

	/* scaling */
	if(pcfeti->scaling==FETI_SCALING_MULTIPLICITY)
	{
	    VecPointwiseDivide(domain->ur,
			       pcdomain->D,
			       domain->ur);            /* divide by multiplicity */
	}
	/*         */

	MatMult(pcdomain->Bbr,
		domain->ur,
		pcdomain->ubr_tmp1);

	if(!pcfeti->lumped)                            
	{
	    /* --- */
	    MatMult(pcdomain->Kib,
		    pcdomain->ubr_tmp1,
		    pcdomain->uii_tmp1);

	    SLESSolve(pcdomain->Kii_inv,
		      pcdomain->uii_tmp1,
		      pcdomain->uii_tmp2,&its);

	    VecScale(pcdomain->uii_tmp2,minusone);                   /* scale by -1 */

	    MatMultTranspose(pcdomain->Kib,
			     pcdomain->uii_tmp2,
			     pcdomain->ubr_tmp2);

	    MatMultAdd(pcdomain->Kbb,
		       pcdomain->ubr_tmp1,
		       pcdomain->ubr_tmp2,
		       pcdomain->ubr_tmp2);     /* ubr_tmp2 = Kbb*ubr_tmp1 + ubr_tmp2; */
	    /* --- */
	}
	else
	{
	    MatMult(pcdomain->Kbb,
		    pcdomain->ubr_tmp1,
		    pcdomain->ubr_tmp2);   

	}

	MatMultTranspose(pcdomain->Bbr,
			 pcdomain->ubr_tmp2,
			 domain->ur);

	/* scaling */	
	if(pcfeti->scaling==FETI_SCALING_MULTIPLICITY)
	{
	    VecPointwiseDivide(domain->ur,
			       pcdomain->D,
			       domain->ur); /* divide by multiplicity */
	}
	/*         */
    }

    MatFetiScatter (pc->mat,            /* apply BrT */
		    dst_lambda,             /* all ur --->(add) dst_lambda */
		    (pcfeti->scaling==FETI_SCALING_RHO)
		    ?FETI_SCATTER_FORWARD_ADD_SCALE
		    :FETI_SCATTER_FORWARD_ADD);     
    PetscFunctionReturn(0);
}

int PETSCKSP_DLLEXPORT PCFetiApplySchurComplement(Vec xb, Vec yb) { /**/ }

int PETSCKSP_DLLEXPORT PCFetiDomainDestroy(PC_FetiDomain *pcdomain)
{
    PetscFunctionBegin;

    if(pcdomain->Kbb)
	MatDestroy (pcdomain->Kbb);
    if(pcdomain->Kib)
	MatDestroy (pcdomain->Kib);
    if(pcdomain->Kii)
	MatDestroy (pcdomain->Kii);
    if(pcdomain->Bbr)
	MatDestroy (pcdomain->Bbr);
    if(pcdomain->Kii_inv)
	SLESDestroy(pcdomain->Kii_inv);
    if(pcdomain->D)
    {

	VecDestroy (pcdomain->D);
    }

    pcdomain->Kii=0;
    pcdomain->Kib=0;
    pcdomain->Kbb=0;
    pcdomain->Bbr=0;
    pcdomain->Kii_inv=0;
    pcdomain->D=0;

    PetscFunctionReturn(0);
}

