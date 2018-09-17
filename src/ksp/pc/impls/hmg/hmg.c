
#include <petscdm.h>
#include <petscctable.h>
#include <petsc/private/matimpl.h>
/* Need to access the hypre private data */
#include <_hypre_parcsr_ls.h>
#include <petsc/private/pcmgimpl.h>
#include <petsc/private/pcimpl.h>      /*I "petscpc.h" I*/

typedef struct {
  PC          hypre;
  PetscBool   reuseinterp;
  PetscInt    blocksize;
} PC_HMG;

PetscErrorCode MatConvert_ParCSRMatrix_AIJ(MPI_Comm, hypre_ParCSRMatrix*, MatType, MatReuse, Mat*);
PetscErrorCode PCSetFromOptions_HMG(PetscOptionItems*,PC);
PetscErrorCode PCSetFromOptions_HYPRE(PetscOptionItems*,PC);
PetscErrorCode PCReset_MG(PC);
PetscErrorCode PCHYPREGetSolver(PC,HYPRE_Solver*);

static PetscErrorCode PCHMGExtractSubMatrix_HMG(Mat pmat,Mat *submat,MatReuse reuse,PetscInt blocksize)
{
  IS             isrow;
  PetscErrorCode ierr;
  PetscInt       rstart,rend, row,subsize, *rowsindices;
  MPI_Comm       comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)pmat,&comm);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(pmat,&rstart,&rend);CHKERRQ(ierr);
  if ((rend-rstart)%blocksize != 0) SETERRQ3(comm,PETSC_ERR_ARG_INCOMP,"Block size %d is inconsisent for [%d, %d) \n",blocksize,rstart,rend);
  subsize = (rend-rstart)/blocksize;
  ierr = PetscCalloc1(subsize,&rowsindices);CHKERRQ(ierr);
  subsize = 0;
  for (row=rstart; row<rend; row+=blocksize){
    rowsindices[subsize++]=row;
  }
  ierr = ISCreateGeneral(comm,subsize,rowsindices,PETSC_OWN_POINTER,&isrow);CHKERRQ(ierr);
  ierr = MatCreateSubMatrix(pmat,isrow,isrow,reuse,submat);CHKERRQ(ierr);
  ierr = ISDestroy(&isrow);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCHMGExpandInterpolation_HMG(MPI_Comm comm,hypre_ParCSRMatrix *subinterp, Mat *interp, PetscInt blocksize)
{
  PetscInt        subrstart,subrend,subrowsize,subcolsize,subcstart,subcend,rowsize,colsize;
  PetscInt        subrow,row,*idx,nz,*d_nnz,*o_nnz,i,j,dnz,onz,max_nz,*indices;
  PetscScalar     *values;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  subrstart = hypre_ParCSRMatrixFirstRowIndex(subinterp);
  subrend = hypre_ParCSRMatrixLastRowIndex(subinterp);
  subrowsize = subrend-subrstart+1;
  rowsize = subrowsize*blocksize;
  subcstart = hypre_ParCSRMatrixFirstColDiag(subinterp);
  subcend = hypre_ParCSRMatrixLastColDiag(subinterp);
  subcolsize = subcend-subcstart+1;
  colsize = subcolsize*blocksize;
  ierr = PetscCalloc2(rowsize,&d_nnz,rowsize,&o_nnz);CHKERRQ(ierr);
  max_nz = 0;
  for(subrow=subrstart; subrow<=subrend; subrow++){
    HYPRE_ParCSRMatrixGetRow(subinterp,subrow,(HYPRE_Int*)&nz,(HYPRE_Int**)&idx,NULL);
    if (max_nz<nz) max_nz = nz;
    dnz = 0; onz = 0;
    for(i=0;i<nz;i++){
      if(idx[i]<subcstart || idx[i]>subcend) onz++;
      else dnz++;
    }
    for(i=0;i<blocksize;i++){
      d_nnz[(subrow-subrstart)*blocksize+i] = dnz;
      o_nnz[(subrow-subrstart)*blocksize+i] = onz;
    }
    HYPRE_ParCSRMatrixRestoreRow(subinterp,subrow,(HYPRE_Int*)&nz,(HYPRE_Int**)&idx,NULL);
  }
  ierr = MatCreateAIJ(comm,rowsize,colsize,PETSC_DETERMINE,PETSC_DETERMINE,0,d_nnz,0,o_nnz,interp);CHKERRQ(ierr);
  ierr = MatSetOption(*interp,MAT_IGNORE_OFF_PROC_ENTRIES,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatSetOption(*interp,MAT_IGNORE_ZERO_ENTRIES,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatSetOption(*interp,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatSetFromOptions(*interp);CHKERRQ(ierr);

  ierr = MatSetUp(*interp);CHKERRQ(ierr);
  ierr = PetscFree2(d_nnz,o_nnz);CHKERRQ(ierr);
  ierr = PetscCalloc1(max_nz,&indices);CHKERRQ(ierr);
  for(subrow=subrstart; subrow<=subrend; subrow++){
    HYPRE_ParCSRMatrixGetRow(subinterp,subrow,(HYPRE_Int*)&nz,(HYPRE_Int**)&idx,&values);
    for(i=0;i<blocksize;i++){
      row = subrow*blocksize+i;
      for (j=0;j<nz;j++){
        indices[j] = idx[j]*blocksize+i;
      }
      ierr = MatSetValues(*interp,1,&row,nz,indices,values,INSERT_VALUES);CHKERRQ(ierr);
    }
    HYPRE_ParCSRMatrixRestoreRow(subinterp,subrow,(HYPRE_Int*)&nz,(HYPRE_Int**)&idx,&values);
  }
  ierr = PetscFree(indices);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*interp,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*interp,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PCSetUp_HMG(PC pc)
{
  PetscErrorCode     ierr;
  Mat                PA, submat;
  PC_MG              *mg   = (PC_MG*)pc->data;
  PC_HMG             *hmg   = (PC_HMG*) mg->innerctx;
  MPI_Comm           comm;
  PetscInt           level;
  hypre_ParCSRMatrix **P_array, **A_array;
  HYPRE_Solver       hsolver;
  hypre_ParAMGData   *amg_data;
  PetscInt           num_levels;
  PetscReal          global_nonzeros, num_rows;
  PetscReal          *sparse;

  PetscFunctionBegin;

  ierr = PetscObjectGetComm((PetscObject)pc,&comm);CHKERRQ(ierr);

  if(pc->setupcalled){
   if (pc->flag == SAME_NONZERO_PATTERN && hmg->reuseinterp) {
    ierr = PCMGSetGalerkin(pc,PC_MG_GALERKIN_PMAT);CHKERRQ(ierr);
    ierr = PCSetUp_MG(pc);CHKERRQ(ierr);
    PetscFunctionReturn(0);
   }else {
     ierr = PCReset_MG(pc);CHKERRQ(ierr);
     pc->setupcalled = PETSC_FALSE;
   }
  }

  if (!hmg->hypre){
    ierr = PCCreate(comm,&hmg->hypre);CHKERRQ(ierr);
    ierr = PCSetType(hmg->hypre,PCHYPRE);CHKERRQ(ierr);
    ierr = PCHYPRESetType(hmg->hypre,"boomeramg");CHKERRQ(ierr);
  }
  ierr = PCGetOperators(pc,NULL,&PA);CHKERRQ(ierr);
  if(hmg->blocksize>1) {
    ierr = PCHMGExtractSubMatrix_HMG(PA,&submat,MAT_INITIAL_MATRIX,hmg->blocksize);CHKERRQ(ierr);
    PA = submat;
  }
  ierr = PCSetOperators(hmg->hypre,PA,PA);CHKERRQ(ierr);
  if (hmg->blocksize>1){
   ierr = MatDestroy(&PA);CHKERRQ(ierr);
  }
  ierr = PCSetUseAmat(hmg->hypre,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscObjectOptionsBegin((PetscObject)hmg->hypre);CHKERRQ(ierr);
  ierr = PCSetFromOptions_HYPRE(PetscOptionsObject,hmg->hypre);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  ierr = PCSetUp(hmg->hypre);

  ierr = PCHYPREGetSolver(hmg->hypre,&hsolver);CHKERRQ(ierr);
  amg_data = (hypre_ParAMGData*) (hsolver);

  if (!amg_data) SETERRQ(comm,PETSC_ERR_ARG_BADPTR,"Fails to setup Hypre \n");

  num_levels = hypre_ParAMGDataNumLevels(amg_data);
  ierr = PetscCalloc1(num_levels,&sparse);CHKERRQ(ierr);

  A_array= hypre_ParAMGDataAArray(amg_data);
  P_array = hypre_ParAMGDataPArray(amg_data);
  for(level=0;level<num_levels;level++){
    global_nonzeros = (PetscReal) hypre_ParCSRMatrixDNumNonzeros(A_array[level]);
    num_rows = (PetscReal) hypre_ParCSRMatrixGlobalNumRows(A_array[level]);
    sparse[num_levels-1-level] =global_nonzeros/(num_rows*num_rows);
  }
  ierr = PCMGSetLevels_MG(pc,num_levels,NULL);CHKERRQ(ierr);
  ierr = PetscFree(sparse);CHKERRQ(ierr);

  for(level=num_levels-1;level>0;level--){
    Mat P=0, pmat=0;
    Vec b, x,r;
    if (hmg->blocksize>1){
     ierr = PCHMGExpandInterpolation_HMG(comm,P_array[num_levels-1-level],&P,hmg->blocksize);CHKERRQ(ierr);
    }else{
     ierr = MatConvert_ParCSRMatrix_AIJ(comm, P_array[num_levels-1-level],MATAIJ, MAT_INPLACE_MATRIX,&P);CHKERRQ(ierr);
    }
    /*ierr = MatView(P,NULL);CHKERRQ(ierr);*/
    ierr = MatCreateVecs(P,&b,&r);CHKERRQ(ierr);
    ierr = PCMGSetInterpolation(pc,level,P);CHKERRQ(ierr);
    ierr = PCMGSetRestriction(pc,level,P);CHKERRQ(ierr);
    ierr = MatDestroy(&P);CHKERRQ(ierr);
    if ((level-1)>=0 && hmg->blocksize<=1) {
      ierr = MatConvert_ParCSRMatrix_AIJ(comm, A_array[num_levels-level],MATAIJ, MAT_INPLACE_MATRIX,&pmat);CHKERRQ(ierr);
      ierr = PCMGSetOperators(pc,level-1,pmat);CHKERRQ(ierr);
      ierr = MatDestroy(&pmat);CHKERRQ(ierr);
    }
    ierr = PCMGSetRhs(pc,level-1,b);CHKERRQ(ierr);

    ierr = PCMGSetR(pc,level,r);CHKERRQ(ierr);
    ierr = VecDestroy(&r);CHKERRQ(ierr);

    ierr = VecDuplicate(b,&x);
    ierr = PCMGSetX(pc,level-1,x);CHKERRQ(ierr);
    ierr = VecDestroy(&x);CHKERRQ(ierr);
    ierr = VecDestroy(&b);CHKERRQ(ierr);
  }
  ierr = PCDestroy(&hmg->hypre);CHKERRQ(ierr);
  hmg->hypre = 0;
  ierr = PCMGSetGalerkin(pc,(hmg->blocksize>1) ? PC_MG_GALERKIN_PMAT:PC_MG_GALERKIN_NONE);CHKERRQ(ierr);
  ierr = PCSetDM(pc,NULL);CHKERRQ(ierr);
  ierr = PCSetUseAmat(pc,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscObjectOptionsBegin((PetscObject)pc);CHKERRQ(ierr);
  ierr = PCSetFromOptions_MG(PetscOptionsObject,pc);CHKERRQ(ierr); /* should be called in PCSetFromOptions_HMG(), but cannot be called prior to PCMGSetLevels() */
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  ierr = PCSetUp_MG(pc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PCDestroy_HMG(PC pc)
{
  PetscErrorCode ierr;
  PC_MG          *mg  = (PC_MG*)pc->data;
  PC_HMG         *ctx = (PC_HMG*) mg->innerctx;

  PetscFunctionBegin;
  ierr = PCDestroy(&ctx->hypre);CHKERRQ(ierr);
  ierr = PetscFree(ctx);CHKERRQ(ierr);
  ierr = PCDestroy_MG(pc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PCView_HMG(PC pc,PetscViewer viewer)
{
  PC_MG          *mg = (PC_MG*)pc->data;
  PC_HMG         *ctx = (PC_HMG*) mg->innerctx;
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer," Reuse interpolation %d\n",ctx->reuseinterp);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer," Matrix block size %D \n",ctx->blocksize);CHKERRQ(ierr);
  }
  ierr = PCView_MG(pc,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PCSetFromOptions_HMG(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PetscErrorCode ierr;
  PC_MG          *mg = (PC_MG*)pc->data;
  PC_HMG         *ctx = (PC_HMG*) mg->innerctx;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"HMG");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_hmg_reuse_interpolation","Reuse the interpolation operators when possible (cheaper, weaker when matrix entries change a lot)","None",ctx->reuseinterp,&ctx->reuseinterp,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_hmg_pmat_blocksize","Block size for each grid point","hmg",ctx->blocksize,&ctx->blocksize,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
   PCHMG - Hybrid of PETSc preconditioners (such as ASM, BJacobi, SOR, etc.) and Hypre BoomerAMG. BoomerAMG is used to coarsen matrix and generate
           a sequence of coarse matrices and interpolations. The matrices and interpolations are employed to construct PCMG, and then any available
           PETSc preconditioners can be chosen as smoothers and the coarse solver.

   Options Database Keys:
+  -pc_hmg_reuse_interpolation <true | false> - Whether or not or not to reuse the interpolations. If true, it potentially save the compute time.
.  -pc_hmg_pmat_blocksize - Block size of the underlying matrix.


   Notes:
    For multicomponent problems, we can just coarsen one submatrix associated with one particular component. In this way, the preconditioner setup
    time is significantly reduced. One typical use case is neutron transport equations. There are many variables on each mesh vertex due to the
    of angle and energy. Each variable, in fact, corresponds to the same PDEs but with different material properties.

   Level: beginner

   Concepts: additive Schwarz method

    References:
+   1. - Fande Kong, Yaqi Wang, Derek R Gaston, Cody J Permann, Andrew E Slaughter, Alexander D Lindsay, Richard C Martineau, A highly parallel multilevel
    Newton-Krylov-Schwarz method with subspace-based coarsening and partition-based balancing for the multigroup neutron transport equations on
    3D unstructured meshes, arXiv preprint arXiv:1903.03659, 2019

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC, PCMG, PCHYPRE

M*/
PETSC_EXTERN PetscErrorCode PCCreate_HMG(PC pc)
{
  PetscErrorCode ierr;
  PC_HMG         *hmg;
  PC_MG          *mg;

  PetscFunctionBegin;
  /* if type was previously mg; must manually destroy it because call to PCSetType(pc,PCMG) will not destroy it */
  if (pc->ops->destroy) {
    ierr =  (*pc->ops->destroy)(pc);CHKERRQ(ierr);
    pc->data = 0;
  }
  ierr = PetscFree(((PetscObject)pc)->type_name);CHKERRQ(ierr);
  ((PetscObject)pc)->type_name = 0;

  ierr         = PCSetType(pc,PCMG);CHKERRQ(ierr);
  ierr         = PetscNew(&hmg);CHKERRQ(ierr); \

  mg                      = (PC_MG*) pc->data;
  mg->innerctx            = hmg;
  hmg->reuseinterp        = PETSC_FALSE;
  hmg->blocksize          = 1;

  pc->ops->setfromoptions = PCSetFromOptions_HMG;
  pc->ops->view           = PCView_HMG;
  pc->ops->destroy        = PCDestroy_HMG;
  pc->ops->setup          = PCSetUp_HMG;
  PetscFunctionReturn(0);
}
