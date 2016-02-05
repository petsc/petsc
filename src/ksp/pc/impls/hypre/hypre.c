
/*
   Provides an interface to the LLNL package hypre
*/

/* Must use hypre 2.0.0 or more recent. */

#include <petsc/private/pcimpl.h>          /*I "petscpc.h" I*/
#include <../src/dm/impls/da/hypre/mhyp.h>
#include <_hypre_parcsr_ls.h>

static PetscBool cite = PETSC_FALSE;
static const char hypreCitation[] = "@manual{hypre-web-page,\n  title  = {{\\sl hypre}: High Performance Preconditioners},\n  organization = {Lawrence Livermore National Laboratory},\n  note  = {\\url{http://www.llnl.gov/CASC/hypre/}}\n}\n";

/*
   Private context (data structure) for the  preconditioner.
*/
typedef struct {
  HYPRE_Solver   hsolver;
  HYPRE_IJMatrix ij;
  HYPRE_IJVector b,x;

  HYPRE_Int (*destroy)(HYPRE_Solver);
  HYPRE_Int (*solve)(HYPRE_Solver,HYPRE_ParCSRMatrix,HYPRE_ParVector,HYPRE_ParVector);
  HYPRE_Int (*setup)(HYPRE_Solver,HYPRE_ParCSRMatrix,HYPRE_ParVector,HYPRE_ParVector);
  HYPRE_Int (*setdgrad)(HYPRE_Solver,HYPRE_ParCSRMatrix);
  HYPRE_Int (*setdcurl)(HYPRE_Solver,HYPRE_ParCSRMatrix);
  HYPRE_Int (*setcoord)(HYPRE_Solver,HYPRE_ParVector,HYPRE_ParVector,HYPRE_ParVector);
  HYPRE_Int (*setdim)(HYPRE_Solver,HYPRE_Int);

  MPI_Comm comm_hypre;
  char     *hypre_type;

  /* options for Pilut and BoomerAMG*/
  PetscInt maxiter;
  double   tol;

  /* options for Pilut */
  PetscInt factorrowsize;

  /* options for ParaSails */
  PetscInt nlevels;
  double   threshhold;
  double   filter;
  PetscInt sym;
  double   loadbal;
  PetscInt logging;
  PetscInt ruse;
  PetscInt symt;

  /* options for BoomerAMG */
  PetscBool printstatistics;

  /* options for BoomerAMG */
  PetscInt  cycletype;
  PetscInt  maxlevels;
  double    strongthreshold;
  double    maxrowsum;
  PetscInt  gridsweeps[3];
  PetscInt  coarsentype;
  PetscInt  measuretype;
  PetscInt  smoothtype;
  PetscInt  smoothnumlevels;
  PetscInt  eu_level;   /* Number of levels for ILU(k) in Euclid */
  double    eu_droptolerance; /* Drop tolerance for ILU(k) in Euclid */
  PetscInt  eu_bj;      /* Defines use of Block Jacobi ILU in Euclid */
  PetscInt  relaxtype[3];
  double    relaxweight;
  double    outerrelaxweight;
  PetscInt  relaxorder;
  double    truncfactor;
  PetscBool applyrichardson;
  PetscInt  pmax;
  PetscInt  interptype;
  PetscInt  agg_nl;
  PetscInt  agg_num_paths;
  PetscInt  nodal_coarsen;
  PetscBool nodal_relax;
  PetscInt  nodal_relax_levels;

  PetscInt  nodal_coarsening;
  PetscInt  vec_interp_variant;
  HYPRE_IJVector  *hmnull;
  HYPRE_ParVector *phmnull;  /* near null space passed to hypre */  
  PetscInt        n_hmnull;
  Vec             hmnull_constant;
  PetscScalar     **hmnull_hypre_data_array;   /* this is the space in hmnull that was allocated by hypre, it is restored to hypre just before freeing the phmnull vectors */

  /* options for AS (Auxiliary Space preconditioners) */
  PetscInt  as_print;
  PetscInt  as_max_iter;
  PetscReal as_tol;
  PetscInt  as_relax_type;
  PetscInt  as_relax_times;
  PetscReal as_relax_weight;
  PetscReal as_omega;
  PetscInt  as_amg_alpha_opts[5]; /* AMG coarsen type, agg_levels, relax_type, interp_type, Pmax for vector Poisson (AMS) or Curl problem (ADS) */
  PetscReal as_amg_alpha_theta;   /* AMG strength for vector Poisson (AMS) or Curl problem (ADS) */
  PetscInt  as_amg_beta_opts[5];  /* AMG coarsen type, agg_levels, relax_type, interp_type, Pmax for scalar Poisson (AMS) or vector Poisson (ADS) */
  PetscReal as_amg_beta_theta;    /* AMG strength for scalar Poisson (AMS) or vector Poisson (ADS)  */
  PetscInt  ams_cycle_type;
  PetscInt  ads_cycle_type;

  /* additional data */
  HYPRE_IJVector coords[3];
  HYPRE_IJVector constants[3];
  HYPRE_IJMatrix G;
  HYPRE_IJMatrix C;
  HYPRE_IJMatrix alpha_Poisson;
  HYPRE_IJMatrix beta_Poisson;
  PetscBool      ams_beta_is_zero;
} PC_HYPRE;

#undef __FUNCT__
#define __FUNCT__ "PCHYPREGetSolver"
PetscErrorCode PCHYPREGetSolver(PC pc,HYPRE_Solver *hsolver)
{
  PC_HYPRE *jac = (PC_HYPRE*)pc->data;

  PetscFunctionBegin;
  *hsolver = jac->hsolver;
  PetscFunctionReturn(0);
}

/*
    Replaces the address where the HYPRE vector points to its data with the address of
  PETSc's data. Saves the old address so it can be reset when we are finished with it.
  Allows use to get the data into a HYPRE vector without the cost of memcopies
*/
#define HYPREReplacePointer(b,newvalue,savedvalue) { \
    hypre_ParVector *par_vector   = (hypre_ParVector*)hypre_IJVectorObject(((hypre_IJVector*)b)); \
    hypre_Vector    *local_vector = hypre_ParVectorLocalVector(par_vector); \
    savedvalue         = local_vector->data; \
    local_vector->data = newvalue;          \
}

#undef __FUNCT__
#define __FUNCT__ "PCSetUp_HYPRE"
static PetscErrorCode PCSetUp_HYPRE(PC pc)
{
  PC_HYPRE           *jac = (PC_HYPRE*)pc->data;
  PetscErrorCode     ierr;
  HYPRE_ParCSRMatrix hmat;
  HYPRE_ParVector    bv,xv;
  PetscInt           bs;

  PetscFunctionBegin;
  if (!jac->hypre_type) {
    ierr = PCHYPRESetType(pc,"boomeramg");CHKERRQ(ierr);
  }

  if (pc->setupcalled) {
    /* always destroy the old matrix and create a new memory;
       hope this does not churn the memory too much. The problem
       is I do not know if it is possible to put the matrix back to
       its initial state so that we can directly copy the values
       the second time through. */
    PetscStackCallStandard(HYPRE_IJMatrixDestroy,(jac->ij));
    jac->ij = 0;
  }

  if (!jac->ij) { /* create the matrix the first time through */
    ierr = MatHYPRE_IJMatrixCreate(pc->pmat,&jac->ij);CHKERRQ(ierr);
  }
  if (!jac->b) { /* create the vectors the first time through */
    Vec x,b;
    ierr = MatCreateVecs(pc->pmat,&x,&b);CHKERRQ(ierr);
    ierr = VecHYPRE_IJVectorCreate(x,&jac->x);CHKERRQ(ierr);
    ierr = VecHYPRE_IJVectorCreate(b,&jac->b);CHKERRQ(ierr);
    ierr = VecDestroy(&x);CHKERRQ(ierr);
    ierr = VecDestroy(&b);CHKERRQ(ierr);
  }

  /* special case for BoomerAMG */
  if (jac->setup == HYPRE_BoomerAMGSetup) {
    MatNullSpace    mnull;
    PetscBool       has_const;
    PetscInt        nvec,i;
    const Vec       *vecs;
    PetscScalar     *petscvecarray;
    
    ierr = MatGetBlockSize(pc->pmat,&bs);CHKERRQ(ierr);
    if (bs > 1) PetscStackCallStandard(HYPRE_BoomerAMGSetNumFunctions,(jac->hsolver,bs));
    ierr = MatGetNearNullSpace(pc->mat, &mnull);CHKERRQ(ierr);
    if (mnull) {
      ierr = MatNullSpaceGetVecs(mnull, &has_const, &nvec, &vecs);CHKERRQ(ierr);
      ierr = PetscMalloc1(nvec+1,&jac->hmnull);CHKERRQ(ierr);
      ierr = PetscMalloc1(nvec+1,&jac->hmnull_hypre_data_array);CHKERRQ(ierr);
      ierr = PetscMalloc1(nvec+1,&jac->phmnull);CHKERRQ(ierr);
      for (i=0; i<nvec; i++) {
        ierr = VecHYPRE_IJVectorCreate(vecs[i],&jac->hmnull[i]);CHKERRQ(ierr);
        ierr = VecGetArrayRead(vecs[i],(const PetscScalar **)&petscvecarray);CHKERRQ(ierr);
        HYPREReplacePointer(jac->hmnull[i],petscvecarray,jac->hmnull_hypre_data_array[i]);
        ierr = VecRestoreArrayRead(vecs[i],(const PetscScalar **)&petscvecarray);CHKERRQ(ierr);
        PetscStackCallStandard(HYPRE_IJVectorGetObject,(jac->hmnull[i],(void**)&jac->phmnull[i]));
      }
      if (has_const) {
        ierr = MatCreateVecs(pc->pmat,&jac->hmnull_constant,NULL);CHKERRQ(ierr);
        ierr = VecSet(jac->hmnull_constant,1);CHKERRQ(ierr);
        ierr = VecNormalize(jac->hmnull_constant,NULL);
        ierr = VecHYPRE_IJVectorCreate(jac->hmnull_constant,&jac->hmnull[nvec]);CHKERRQ(ierr);
        ierr = VecGetArrayRead(jac->hmnull_constant,(const PetscScalar **)&petscvecarray);CHKERRQ(ierr);
        HYPREReplacePointer(jac->hmnull[nvec],petscvecarray,jac->hmnull_hypre_data_array[nvec]);
        ierr = VecRestoreArrayRead(jac->hmnull_constant,(const PetscScalar **)&petscvecarray);CHKERRQ(ierr);
        PetscStackCallStandard(HYPRE_IJVectorGetObject,(jac->hmnull[nvec],(void**)&jac->phmnull[nvec]));
        nvec++;
      }
      PetscStackCallStandard(HYPRE_BoomerAMGSetInterpVectors,(jac->hsolver,nvec,jac->phmnull));
      jac->n_hmnull = nvec;
    }
  }

  /* special case for AMS */
  if (jac->setup == HYPRE_AMSSetup) {
    if (!jac->coords[0] && !jac->constants[0]) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_USER,"HYPRE AMS preconditioner needs either coordinate vectors via PCSetCoordinates() or edge constant vectors via PCHYPRESetEdgeConstantVectors()");
    if (!jac->G) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_USER,"HYPRE AMS preconditioner needs discrete gradient operator via PCHYPRESetDiscreteGradient");
  }
  /* special case for ADS */
  if (jac->setup == HYPRE_ADSSetup) {
    if (!jac->coords[0]) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_USER,"HYPRE ADS preconditioner needs coordinate vectors via PCSetCoordinates()");
    else if (!jac->coords[1] || !jac->coords[2]) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_USER,"HYPRE ADS preconditioner has been designed for three dimensional problems! For two dimensional problems, use HYPRE AMS instead");
    if (!jac->G) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_USER,"HYPRE ADS preconditioner needs discrete gradient operator via PCHYPRESetDiscreteGradient");
    if (!jac->C) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_USER,"HYPRE ADS preconditioner needs discrete curl operator via PCHYPRESetDiscreteGradient");
  }
  ierr = MatHYPRE_IJMatrixCopy(pc->pmat,jac->ij);CHKERRQ(ierr);
  PetscStackCallStandard(HYPRE_IJMatrixGetObject,(jac->ij,(void**)&hmat));
  PetscStackCallStandard(HYPRE_IJVectorGetObject,(jac->b,(void**)&bv));
  PetscStackCallStandard(HYPRE_IJVectorGetObject,(jac->x,(void**)&xv));
  PetscStackCall("HYPRE_SetupXXX",ierr = (*jac->setup)(jac->hsolver,hmat,bv,xv);CHKERRQ(ierr););
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCApply_HYPRE"
static PetscErrorCode PCApply_HYPRE(PC pc,Vec b,Vec x)
{
  PC_HYPRE           *jac = (PC_HYPRE*)pc->data;
  PetscErrorCode     ierr;
  HYPRE_ParCSRMatrix hmat;
  PetscScalar        *xv;
  const PetscScalar  *bv,*sbv;
  HYPRE_ParVector    jbv,jxv;
  PetscScalar        *sxv;
  PetscInt           hierr;

  PetscFunctionBegin;
  ierr = PetscCitationsRegister(hypreCitation,&cite);CHKERRQ(ierr);
  if (!jac->applyrichardson) {ierr = VecSet(x,0.0);CHKERRQ(ierr);}
  ierr = VecGetArrayRead(b,&bv);CHKERRQ(ierr);
  ierr = VecGetArray(x,&xv);CHKERRQ(ierr);
  HYPREReplacePointer(jac->b,(PetscScalar*)bv,sbv);
  HYPREReplacePointer(jac->x,xv,sxv);

  PetscStackCallStandard(HYPRE_IJMatrixGetObject,(jac->ij,(void**)&hmat));
  PetscStackCallStandard(HYPRE_IJVectorGetObject,(jac->b,(void**)&jbv));
  PetscStackCallStandard(HYPRE_IJVectorGetObject,(jac->x,(void**)&jxv));
  PetscStackCall("Hypre solve",hierr = (*jac->solve)(jac->hsolver,hmat,jbv,jxv);
                               if (hierr && hierr != HYPRE_ERROR_CONV) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in HYPRE solver, error code %d",hierr);
                               if (hierr) hypre__global_error = 0;);

  HYPREReplacePointer(jac->b,(PetscScalar*)sbv,bv);
  HYPREReplacePointer(jac->x,sxv,xv);
  ierr = VecRestoreArray(x,&xv);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(b,&bv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCReset_HYPRE"
static PetscErrorCode PCReset_HYPRE(PC pc)
{
  PC_HYPRE       *jac = (PC_HYPRE*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (jac->ij) PetscStackCallStandard(HYPRE_IJMatrixDestroy,(jac->ij)); jac->ij = NULL;
  if (jac->b) PetscStackCallStandard(HYPRE_IJVectorDestroy,(jac->b)); jac->b = NULL;
  if (jac->x) PetscStackCallStandard(HYPRE_IJVectorDestroy,(jac->x)); jac->x = NULL;
  if (jac->coords[0]) PetscStackCallStandard(HYPRE_IJVectorDestroy,(jac->coords[0])); jac->coords[0] = NULL;
  if (jac->coords[1]) PetscStackCallStandard(HYPRE_IJVectorDestroy,(jac->coords[1])); jac->coords[1] = NULL;
  if (jac->coords[2]) PetscStackCallStandard(HYPRE_IJVectorDestroy,(jac->coords[2])); jac->coords[2] = NULL;
  if (jac->constants[0]) PetscStackCallStandard(HYPRE_IJVectorDestroy,(jac->constants[0])); jac->constants[0] = NULL;
  if (jac->constants[1]) PetscStackCallStandard(HYPRE_IJVectorDestroy,(jac->constants[1])); jac->constants[1] = NULL;
  if (jac->constants[2]) PetscStackCallStandard(HYPRE_IJVectorDestroy,(jac->constants[2])); jac->constants[2] = NULL;
  if (jac->G) PetscStackCallStandard(HYPRE_IJMatrixDestroy,(jac->G)); jac->G = NULL;
  if (jac->C) PetscStackCallStandard(HYPRE_IJMatrixDestroy,(jac->C)); jac->C = NULL;
  if (jac->alpha_Poisson) PetscStackCallStandard(HYPRE_IJMatrixDestroy,(jac->alpha_Poisson)); jac->alpha_Poisson = NULL;
  if (jac->beta_Poisson) PetscStackCallStandard(HYPRE_IJMatrixDestroy,(jac->beta_Poisson)); jac->beta_Poisson = NULL;
  if (jac->n_hmnull && jac->hmnull) {
    PetscInt                 i;
    PETSC_UNUSED PetscScalar *petscvecarray;

    for (i=0; i<jac->n_hmnull; i++) {
      HYPREReplacePointer(jac->hmnull[i],jac->hmnull_hypre_data_array[i],petscvecarray);
      PetscStackCallStandard(HYPRE_IJVectorDestroy,(jac->hmnull[i]));
    }
    ierr = PetscFree(jac->hmnull);CHKERRQ(ierr);
    ierr = PetscFree(jac->hmnull_hypre_data_array);CHKERRQ(ierr);
    ierr = PetscFree(jac->phmnull);CHKERRQ(ierr);
    ierr = VecDestroy(&jac->hmnull_constant);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCDestroy_HYPRE"
static PetscErrorCode PCDestroy_HYPRE(PC pc)
{
  PC_HYPRE                 *jac = (PC_HYPRE*)pc->data;
  PetscErrorCode           ierr;

  PetscFunctionBegin;
  ierr = PCReset_HYPRE(pc);CHKERRQ(ierr);
  if (jac->destroy) PetscStackCall("HYPRE_DestroyXXX",ierr = (*jac->destroy)(jac->hsolver);CHKERRQ(ierr););
  ierr = PetscFree(jac->hypre_type);CHKERRQ(ierr);
  if (jac->comm_hypre != MPI_COMM_NULL) { ierr = MPI_Comm_free(&(jac->comm_hypre));CHKERRQ(ierr);}
  ierr = PetscFree(pc->data);CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)pc,0);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCHYPRESetType_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCHYPREGetType_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCHYPRESetCoordinates_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCHYPRESetDiscreteGradient_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCHYPRESetDiscreteCurl_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCHYPRESetConstantEdgeVectors_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCHYPRESetAlphaPoissonMatrix_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCHYPRESetBetaPoissonMatrix_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "PCSetFromOptions_HYPRE_Pilut"
static PetscErrorCode PCSetFromOptions_HYPRE_Pilut(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PC_HYPRE       *jac = (PC_HYPRE*)pc->data;
  PetscErrorCode ierr;
  PetscBool      flag;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"HYPRE Pilut Options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_hypre_pilut_maxiter","Number of iterations","None",jac->maxiter,&jac->maxiter,&flag);CHKERRQ(ierr);
  if (flag) PetscStackCallStandard(HYPRE_ParCSRPilutSetMaxIter,(jac->hsolver,jac->maxiter));
  ierr = PetscOptionsReal("-pc_hypre_pilut_tol","Drop tolerance","None",jac->tol,&jac->tol,&flag);CHKERRQ(ierr);
  if (flag) PetscStackCallStandard(HYPRE_ParCSRPilutSetDropTolerance,(jac->hsolver,jac->tol));
  ierr = PetscOptionsInt("-pc_hypre_pilut_factorrowsize","FactorRowSize","None",jac->factorrowsize,&jac->factorrowsize,&flag);CHKERRQ(ierr);
  if (flag) PetscStackCallStandard(HYPRE_ParCSRPilutSetFactorRowSize,(jac->hsolver,jac->factorrowsize));
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCView_HYPRE_Pilut"
static PetscErrorCode PCView_HYPRE_Pilut(PC pc,PetscViewer viewer)
{
  PC_HYPRE       *jac = (PC_HYPRE*)pc->data;
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE Pilut preconditioning\n");CHKERRQ(ierr);
    if (jac->maxiter != PETSC_DEFAULT) {
      ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE Pilut: maximum number of iterations %d\n",jac->maxiter);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE Pilut: default maximum number of iterations \n");CHKERRQ(ierr);
    }
    if (jac->tol != PETSC_DEFAULT) {
      ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE Pilut: drop tolerance %g\n",(double)jac->tol);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE Pilut: default drop tolerance \n");CHKERRQ(ierr);
    }
    if (jac->factorrowsize != PETSC_DEFAULT) {
      ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE Pilut: factor row size %d\n",jac->factorrowsize);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE Pilut: default factor row size \n");CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------------*/

#undef __FUNCT__
#define __FUNCT__ "PCApplyTranspose_HYPRE_BoomerAMG"
static PetscErrorCode PCApplyTranspose_HYPRE_BoomerAMG(PC pc,Vec b,Vec x)
{
  PC_HYPRE           *jac = (PC_HYPRE*)pc->data;
  PetscErrorCode     ierr;
  HYPRE_ParCSRMatrix hmat;
  PetscScalar        *xv;
  const PetscScalar  *bv;
  HYPRE_ParVector    jbv,jxv;
  PetscScalar        *sbv,*sxv;
  PetscInt           hierr;

  PetscFunctionBegin;
  ierr = PetscCitationsRegister(hypreCitation,&cite);CHKERRQ(ierr);
  ierr = VecSet(x,0.0);CHKERRQ(ierr);
  ierr = VecGetArrayRead(b,&bv);CHKERRQ(ierr);
  ierr = VecGetArray(x,&xv);CHKERRQ(ierr);
  HYPREReplacePointer(jac->b,(PetscScalar*)bv,sbv);
  HYPREReplacePointer(jac->x,xv,sxv);

  PetscStackCallStandard(HYPRE_IJMatrixGetObject,(jac->ij,(void**)&hmat));
  PetscStackCallStandard(HYPRE_IJVectorGetObject,(jac->b,(void**)&jbv));
  PetscStackCallStandard(HYPRE_IJVectorGetObject,(jac->x,(void**)&jxv));

  hierr = HYPRE_BoomerAMGSolveT(jac->hsolver,hmat,jbv,jxv);
  /* error code of 1 in BoomerAMG merely means convergence not achieved */
  if (hierr && (hierr != 1)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in HYPRE solver, error code %d",hierr);
  if (hierr) hypre__global_error = 0;

  HYPREReplacePointer(jac->b,sbv,bv);
  HYPREReplacePointer(jac->x,sxv,xv);
  ierr = VecRestoreArray(x,&xv);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(b,&bv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* static array length */
#define ALEN(a) (sizeof(a)/sizeof((a)[0]))

static const char *HYPREBoomerAMGCycleType[]   = {"","V","W"};
static const char *HYPREBoomerAMGCoarsenType[] = {"CLJP","Ruge-Stueben","","modifiedRuge-Stueben","","","Falgout", "", "PMIS", "", "HMIS"};
static const char *HYPREBoomerAMGMeasureType[] = {"local","global"};
/* The following corresponds to HYPRE_BoomerAMGSetRelaxType which has many missing numbers in the enum */
static const char *HYPREBoomerAMGSmoothType[]   = {"Schwarz-smoothers","Pilut","ParaSails","Euclid"};
static const char *HYPREBoomerAMGRelaxType[]   = {"Jacobi","sequential-Gauss-Seidel","seqboundary-Gauss-Seidel","SOR/Jacobi","backward-SOR/Jacobi",
                                                  "" /* [5] hybrid chaotic Gauss-Seidel (works only with OpenMP) */,"symmetric-SOR/Jacobi",
                                                  "" /* 7 */,"l1scaled-SOR/Jacobi","Gaussian-elimination",
                                                  "" /* 10 */, "" /* 11 */, "" /* 12 */, "" /* 13 */, "" /* 14 */,
                                                  "CG" /* non-stationary */,"Chebyshev","FCF-Jacobi","l1scaled-Jacobi"};
static const char *HYPREBoomerAMGInterpType[]  = {"classical", "", "", "direct", "multipass", "multipass-wts", "ext+i",
                                                  "ext+i-cc", "standard", "standard-wts", "", "", "FF", "FF1"};
#undef __FUNCT__
#define __FUNCT__ "PCSetFromOptions_HYPRE_BoomerAMG"
static PetscErrorCode PCSetFromOptions_HYPRE_BoomerAMG(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PC_HYPRE       *jac = (PC_HYPRE*)pc->data;
  PetscErrorCode ierr;
  PetscInt       n,indx,level;
  PetscBool      flg, tmp_truth;
  double         tmpdbl, twodbl[2];

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"HYPRE BoomerAMG Options");CHKERRQ(ierr);
  ierr = PetscOptionsEList("-pc_hypre_boomeramg_cycle_type","Cycle type","None",HYPREBoomerAMGCycleType+1,2,HYPREBoomerAMGCycleType[jac->cycletype],&indx,&flg);CHKERRQ(ierr);
  if (flg) {
    jac->cycletype = indx+1;
    PetscStackCallStandard(HYPRE_BoomerAMGSetCycleType,(jac->hsolver,jac->cycletype));
  }
  ierr = PetscOptionsInt("-pc_hypre_boomeramg_max_levels","Number of levels (of grids) allowed","None",jac->maxlevels,&jac->maxlevels,&flg);CHKERRQ(ierr);
  if (flg) {
    if (jac->maxlevels < 2) SETERRQ1(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_OUTOFRANGE,"Number of levels %d must be at least two",jac->maxlevels);
    PetscStackCallStandard(HYPRE_BoomerAMGSetMaxLevels,(jac->hsolver,jac->maxlevels));
  }
  ierr = PetscOptionsInt("-pc_hypre_boomeramg_max_iter","Maximum iterations used PER hypre call","None",jac->maxiter,&jac->maxiter,&flg);CHKERRQ(ierr);
  if (flg) {
    if (jac->maxiter < 1) SETERRQ1(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_OUTOFRANGE,"Number of iterations %d must be at least one",jac->maxiter);
    PetscStackCallStandard(HYPRE_BoomerAMGSetMaxIter,(jac->hsolver,jac->maxiter));
  }
  ierr = PetscOptionsScalar("-pc_hypre_boomeramg_tol","Convergence tolerance PER hypre call (0.0 = use a fixed number of iterations)","None",jac->tol,&jac->tol,&flg);CHKERRQ(ierr);
  if (flg) {
    if (jac->tol < 0.0) SETERRQ1(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_OUTOFRANGE,"Tolerance %g must be greater than or equal to zero",(double)jac->tol);
    PetscStackCallStandard(HYPRE_BoomerAMGSetTol,(jac->hsolver,jac->tol));
  }

  ierr = PetscOptionsScalar("-pc_hypre_boomeramg_truncfactor","Truncation factor for interpolation (0=no truncation)","None",jac->truncfactor,&jac->truncfactor,&flg);CHKERRQ(ierr);
  if (flg) {
    if (jac->truncfactor < 0.0) SETERRQ1(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_OUTOFRANGE,"Truncation factor %g must be great than or equal zero",(double)jac->truncfactor);
    PetscStackCallStandard(HYPRE_BoomerAMGSetTruncFactor,(jac->hsolver,jac->truncfactor));
  }

  ierr = PetscOptionsInt("-pc_hypre_boomeramg_P_max","Max elements per row for interpolation operator (0=unlimited)","None",jac->pmax,&jac->pmax,&flg);CHKERRQ(ierr);
  if (flg) {
    if (jac->pmax < 0) SETERRQ1(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_OUTOFRANGE,"P_max %g must be greater than or equal to zero",(double)jac->pmax);
    PetscStackCallStandard(HYPRE_BoomerAMGSetPMaxElmts,(jac->hsolver,jac->pmax));
  }

  ierr = PetscOptionsInt("-pc_hypre_boomeramg_agg_nl","Number of levels of aggressive coarsening","None",jac->agg_nl,&jac->agg_nl,&flg);CHKERRQ(ierr);
  if (flg) {
    if (jac->agg_nl < 0) SETERRQ1(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_OUTOFRANGE,"Number of levels %g must be greater than or equal to zero",(double)jac->agg_nl);

    PetscStackCallStandard(HYPRE_BoomerAMGSetAggNumLevels,(jac->hsolver,jac->agg_nl));
  }


  ierr = PetscOptionsInt("-pc_hypre_boomeramg_agg_num_paths","Number of paths for aggressive coarsening","None",jac->agg_num_paths,&jac->agg_num_paths,&flg);CHKERRQ(ierr);
  if (flg) {
    if (jac->agg_num_paths < 1) SETERRQ1(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_OUTOFRANGE,"Number of paths %g must be greater than or equal to 1",(double)jac->agg_num_paths);

    PetscStackCallStandard(HYPRE_BoomerAMGSetNumPaths,(jac->hsolver,jac->agg_num_paths));
  }


  ierr = PetscOptionsScalar("-pc_hypre_boomeramg_strong_threshold","Threshold for being strongly connected","None",jac->strongthreshold,&jac->strongthreshold,&flg);CHKERRQ(ierr);
  if (flg) {
    if (jac->strongthreshold < 0.0) SETERRQ1(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_OUTOFRANGE,"Strong threshold %g must be great than or equal zero",(double)jac->strongthreshold);
    PetscStackCallStandard(HYPRE_BoomerAMGSetStrongThreshold,(jac->hsolver,jac->strongthreshold));
  }
  ierr = PetscOptionsScalar("-pc_hypre_boomeramg_max_row_sum","Maximum row sum","None",jac->maxrowsum,&jac->maxrowsum,&flg);CHKERRQ(ierr);
  if (flg) {
    if (jac->maxrowsum < 0.0) SETERRQ1(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_OUTOFRANGE,"Maximum row sum %g must be greater than zero",(double)jac->maxrowsum);
    if (jac->maxrowsum > 1.0) SETERRQ1(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_OUTOFRANGE,"Maximum row sum %g must be less than or equal one",(double)jac->maxrowsum);
    PetscStackCallStandard(HYPRE_BoomerAMGSetMaxRowSum,(jac->hsolver,jac->maxrowsum));
  }

  /* Grid sweeps */
  ierr = PetscOptionsInt("-pc_hypre_boomeramg_grid_sweeps_all","Number of sweeps for the up and down grid levels","None",jac->gridsweeps[0],&indx,&flg);CHKERRQ(ierr);
  if (flg) {
    PetscStackCallStandard(HYPRE_BoomerAMGSetNumSweeps,(jac->hsolver,indx));
    /* modify the jac structure so we can view the updated options with PC_View */
    jac->gridsweeps[0] = indx;
    jac->gridsweeps[1] = indx;
    /*defaults coarse to 1 */
    jac->gridsweeps[2] = 1;
  }

  ierr = PetscOptionsInt("-pc_hypre_boomeramg_nodal_coarsen","Use a nodal based coarsening 1-6","HYPRE_BoomerAMGSetNodal",jac->nodal_coarsening, &jac->nodal_coarsening,&flg);CHKERRQ(ierr);
  if (flg) {
    PetscStackCallStandard(HYPRE_BoomerAMGSetNodal,(jac->hsolver,jac->nodal_coarsening));
  }

  ierr = PetscOptionsInt("-pc_hypre_boomeramg_vec_interp_variant","Variant of algorithm 1-3","HYPRE_BoomerAMGSetInterpVecVariant",jac->vec_interp_variant, &jac->vec_interp_variant,&flg);CHKERRQ(ierr);
  if (flg) {
    PetscStackCallStandard(HYPRE_BoomerAMGSetInterpVecVariant,(jac->hsolver,jac->vec_interp_variant));
  }

  ierr = PetscOptionsInt("-pc_hypre_boomeramg_grid_sweeps_down","Number of sweeps for the down cycles","None",jac->gridsweeps[0], &indx,&flg);CHKERRQ(ierr);
  if (flg) {
    PetscStackCallStandard(HYPRE_BoomerAMGSetCycleNumSweeps,(jac->hsolver,indx, 1));
    jac->gridsweeps[0] = indx;
  }
  ierr = PetscOptionsInt("-pc_hypre_boomeramg_grid_sweeps_up","Number of sweeps for the up cycles","None",jac->gridsweeps[1],&indx,&flg);CHKERRQ(ierr);
  if (flg) {
    PetscStackCallStandard(HYPRE_BoomerAMGSetCycleNumSweeps,(jac->hsolver,indx, 2));
    jac->gridsweeps[1] = indx;
  }
  ierr = PetscOptionsInt("-pc_hypre_boomeramg_grid_sweeps_coarse","Number of sweeps for the coarse level","None",jac->gridsweeps[2],&indx,&flg);CHKERRQ(ierr);
  if (flg) {
    PetscStackCallStandard(HYPRE_BoomerAMGSetCycleNumSweeps,(jac->hsolver,indx, 3));
    jac->gridsweeps[2] = indx;
  }

  /* Smooth type */
  ierr = PetscOptionsEList("-pc_hypre_boomeramg_smooth_type","Enable more complex smoothers","None",HYPREBoomerAMGSmoothType,ALEN(HYPREBoomerAMGSmoothType),HYPREBoomerAMGSmoothType[0],&indx,&flg);
  if (flg) {
    jac->smoothtype = indx;
    PetscStackCallStandard(HYPRE_BoomerAMGSetSmoothType,(jac->hsolver,indx+6)); 
    jac->smoothnumlevels = 25;
    PetscStackCallStandard(HYPRE_BoomerAMGSetSmoothNumLevels,(jac->hsolver,25));
  }

  /* Number of smoothing levels */
  ierr = PetscOptionsInt("-pc_hypre_boomeramg_smooth_num_levels","Number of levels on which more complex smoothers are used","None",25,&indx,&flg);CHKERRQ(ierr);
  if (flg && (jac->smoothtype != -1)) {
    jac->smoothnumlevels = indx;
    PetscStackCallStandard(HYPRE_BoomerAMGSetSmoothNumLevels,(jac->hsolver,indx));
  }

  /* Number of levels for ILU(k) for Euclid */
  ierr = PetscOptionsInt("-pc_hypre_boomeramg_eu_level","Number of levels for ILU(k) in Euclid smoother","None",0,&indx,&flg);CHKERRQ(ierr);
  if (flg && (jac->smoothtype == 3)) {
    jac->eu_level = indx;
    PetscStackCallStandard(HYPRE_BoomerAMGSetEuLevel,(jac->hsolver,indx));
  }

  /* Filter for ILU(k) for Euclid */
  double droptolerance;
  ierr = PetscOptionsScalar("-pc_hypre_boomeramg_eu_droptolerance","Drop tolerance for ILU(k) in Euclid smoother","None",0,&droptolerance,&flg);CHKERRQ(ierr);
  if (flg && (jac->smoothtype == 3)) {
    jac->eu_droptolerance = droptolerance;
    PetscStackCallStandard(HYPRE_BoomerAMGSetEuLevel,(jac->hsolver,droptolerance));
  }

  /* Use Block Jacobi ILUT for Euclid */
  ierr = PetscOptionsBool("-pc_hypre_boomeramg_eu_bj", "Use Block Jacobi for ILU in Euclid smoother?", "None", PETSC_FALSE, &tmp_truth, &flg);CHKERRQ(ierr);
  if (flg && (jac->smoothtype == 3)) {
    jac->eu_bj = tmp_truth;
    PetscStackCallStandard(HYPRE_BoomerAMGSetEuBJ,(jac->hsolver,jac->eu_bj));
  }

  /* Relax type */
  ierr = PetscOptionsEList("-pc_hypre_boomeramg_relax_type_all","Relax type for the up and down cycles","None",HYPREBoomerAMGRelaxType,ALEN(HYPREBoomerAMGRelaxType),HYPREBoomerAMGRelaxType[6],&indx,&flg);CHKERRQ(ierr);
  if (flg) {
    jac->relaxtype[0] = jac->relaxtype[1]  = indx;
    PetscStackCallStandard(HYPRE_BoomerAMGSetRelaxType,(jac->hsolver, indx));
    /* by default, coarse type set to 9 */
    jac->relaxtype[2] = 9;

  }
  ierr = PetscOptionsEList("-pc_hypre_boomeramg_relax_type_down","Relax type for the down cycles","None",HYPREBoomerAMGRelaxType,ALEN(HYPREBoomerAMGRelaxType),HYPREBoomerAMGRelaxType[6],&indx,&flg);CHKERRQ(ierr);
  if (flg) {
    jac->relaxtype[0] = indx;
    PetscStackCallStandard(HYPRE_BoomerAMGSetCycleRelaxType,(jac->hsolver, indx, 1));
  }
  ierr = PetscOptionsEList("-pc_hypre_boomeramg_relax_type_up","Relax type for the up cycles","None",HYPREBoomerAMGRelaxType,ALEN(HYPREBoomerAMGRelaxType),HYPREBoomerAMGRelaxType[6],&indx,&flg);CHKERRQ(ierr);
  if (flg) {
    jac->relaxtype[1] = indx;
    PetscStackCallStandard(HYPRE_BoomerAMGSetCycleRelaxType,(jac->hsolver, indx, 2));
  }
  ierr = PetscOptionsEList("-pc_hypre_boomeramg_relax_type_coarse","Relax type on coarse grid","None",HYPREBoomerAMGRelaxType,ALEN(HYPREBoomerAMGRelaxType),HYPREBoomerAMGRelaxType[9],&indx,&flg);CHKERRQ(ierr);
  if (flg) {
    jac->relaxtype[2] = indx;
    PetscStackCallStandard(HYPRE_BoomerAMGSetCycleRelaxType,(jac->hsolver, indx, 3));
  }

  /* Relaxation Weight */
  ierr = PetscOptionsReal("-pc_hypre_boomeramg_relax_weight_all","Relaxation weight for all levels (0 = hypre estimates, -k = determined with k CG steps)","None",jac->relaxweight, &tmpdbl,&flg);CHKERRQ(ierr);
  if (flg) {
    PetscStackCallStandard(HYPRE_BoomerAMGSetRelaxWt,(jac->hsolver,tmpdbl));
    jac->relaxweight = tmpdbl;
  }

  n         = 2;
  twodbl[0] = twodbl[1] = 1.0;
  ierr      = PetscOptionsRealArray("-pc_hypre_boomeramg_relax_weight_level","Set the relaxation weight for a particular level (weight,level)","None",twodbl, &n, &flg);CHKERRQ(ierr);
  if (flg) {
    if (n == 2) {
      indx =  (int)PetscAbsReal(twodbl[1]);
      PetscStackCallStandard(HYPRE_BoomerAMGSetLevelRelaxWt,(jac->hsolver,twodbl[0],indx));
    } else SETERRQ1(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_OUTOFRANGE,"Relax weight level: you must provide 2 values separated by a comma (and no space), you provided %d",n);
  }

  /* Outer relaxation Weight */
  ierr = PetscOptionsReal("-pc_hypre_boomeramg_outer_relax_weight_all","Outer relaxation weight for all levels (-k = determined with k CG steps)","None",jac->outerrelaxweight, &tmpdbl,&flg);CHKERRQ(ierr);
  if (flg) {
    PetscStackCallStandard(HYPRE_BoomerAMGSetOuterWt,(jac->hsolver, tmpdbl));
    jac->outerrelaxweight = tmpdbl;
  }

  n         = 2;
  twodbl[0] = twodbl[1] = 1.0;
  ierr      = PetscOptionsRealArray("-pc_hypre_boomeramg_outer_relax_weight_level","Set the outer relaxation weight for a particular level (weight,level)","None",twodbl, &n, &flg);CHKERRQ(ierr);
  if (flg) {
    if (n == 2) {
      indx =  (int)PetscAbsReal(twodbl[1]);
      PetscStackCallStandard(HYPRE_BoomerAMGSetLevelOuterWt,(jac->hsolver, twodbl[0], indx));
    } else SETERRQ1(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_OUTOFRANGE,"Relax weight outer level: You must provide 2 values separated by a comma (and no space), you provided %d",n);
  }

  /* the Relax Order */
  ierr = PetscOptionsBool("-pc_hypre_boomeramg_no_CF", "Do not use CF-relaxation", "None", PETSC_FALSE, &tmp_truth, &flg);CHKERRQ(ierr);

  if (flg && tmp_truth) {
    jac->relaxorder = 0;
    PetscStackCallStandard(HYPRE_BoomerAMGSetRelaxOrder,(jac->hsolver, jac->relaxorder));
  }
  ierr = PetscOptionsEList("-pc_hypre_boomeramg_measure_type","Measure type","None",HYPREBoomerAMGMeasureType,ALEN(HYPREBoomerAMGMeasureType),HYPREBoomerAMGMeasureType[0],&indx,&flg);CHKERRQ(ierr);
  if (flg) {
    jac->measuretype = indx;
    PetscStackCallStandard(HYPRE_BoomerAMGSetMeasureType,(jac->hsolver,jac->measuretype));
  }
  /* update list length 3/07 */
  ierr = PetscOptionsEList("-pc_hypre_boomeramg_coarsen_type","Coarsen type","None",HYPREBoomerAMGCoarsenType,ALEN(HYPREBoomerAMGCoarsenType),HYPREBoomerAMGCoarsenType[6],&indx,&flg);CHKERRQ(ierr);
  if (flg) {
    jac->coarsentype = indx;
    PetscStackCallStandard(HYPRE_BoomerAMGSetCoarsenType,(jac->hsolver,jac->coarsentype));
  }

  /* new 3/07 */
  ierr = PetscOptionsEList("-pc_hypre_boomeramg_interp_type","Interpolation type","None",HYPREBoomerAMGInterpType,ALEN(HYPREBoomerAMGInterpType),HYPREBoomerAMGInterpType[0],&indx,&flg);CHKERRQ(ierr);
  if (flg) {
    jac->interptype = indx;
    PetscStackCallStandard(HYPRE_BoomerAMGSetInterpType,(jac->hsolver,jac->interptype));
  }

  ierr = PetscOptionsName("-pc_hypre_boomeramg_print_statistics","Print statistics","None",&flg);CHKERRQ(ierr);
  if (flg) {
    level = 3;
    ierr = PetscOptionsInt("-pc_hypre_boomeramg_print_statistics","Print statistics","None",level,&level,NULL);CHKERRQ(ierr);

    jac->printstatistics = PETSC_TRUE;
    PetscStackCallStandard(HYPRE_BoomerAMGSetPrintLevel,(jac->hsolver,level));
  }

  ierr = PetscOptionsName("-pc_hypre_boomeramg_print_debug","Print debug information","None",&flg);CHKERRQ(ierr);
  if (flg) {
    level = 3;
    ierr = PetscOptionsInt("-pc_hypre_boomeramg_print_debug","Print debug information","None",level,&level,NULL);CHKERRQ(ierr);

    jac->printstatistics = PETSC_TRUE;
    PetscStackCallStandard(HYPRE_BoomerAMGSetDebugFlag,(jac->hsolver,level));
  }

  ierr = PetscOptionsBool("-pc_hypre_boomeramg_nodal_relaxation", "Nodal relaxation via Schwarz", "None", PETSC_FALSE, &tmp_truth, &flg);CHKERRQ(ierr);
  if (flg && tmp_truth) {
    PetscInt tmp_int;
    ierr = PetscOptionsInt("-pc_hypre_boomeramg_nodal_relaxation", "Nodal relaxation via Schwarz", "None",jac->nodal_relax_levels,&tmp_int,&flg);CHKERRQ(ierr);
    if (flg) jac->nodal_relax_levels = tmp_int;
    PetscStackCallStandard(HYPRE_BoomerAMGSetSmoothType,(jac->hsolver,6));
    PetscStackCallStandard(HYPRE_BoomerAMGSetDomainType,(jac->hsolver,1));
    PetscStackCallStandard(HYPRE_BoomerAMGSetOverlap,(jac->hsolver,0));
    PetscStackCallStandard(HYPRE_BoomerAMGSetSmoothNumLevels,(jac->hsolver,jac->nodal_relax_levels));
  }

  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCApplyRichardson_HYPRE_BoomerAMG"
static PetscErrorCode PCApplyRichardson_HYPRE_BoomerAMG(PC pc,Vec b,Vec y,Vec w,PetscReal rtol,PetscReal abstol, PetscReal dtol,PetscInt its,PetscBool guesszero,PetscInt *outits,PCRichardsonConvergedReason *reason)
{
  PC_HYPRE       *jac = (PC_HYPRE*)pc->data;
  PetscErrorCode ierr;
  PetscInt       oits;

  PetscFunctionBegin;
  ierr = PetscCitationsRegister(hypreCitation,&cite);CHKERRQ(ierr);
  PetscStackCallStandard(HYPRE_BoomerAMGSetMaxIter,(jac->hsolver,its*jac->maxiter));
  PetscStackCallStandard(HYPRE_BoomerAMGSetTol,(jac->hsolver,rtol));
  jac->applyrichardson = PETSC_TRUE;
  ierr                 = PCApply_HYPRE(pc,b,y);CHKERRQ(ierr);
  jac->applyrichardson = PETSC_FALSE;
  PetscStackCallStandard(HYPRE_BoomerAMGGetNumIterations,(jac->hsolver,(HYPRE_Int *)&oits));
  *outits = oits;
  if (oits == its) *reason = PCRICHARDSON_CONVERGED_ITS;
  else             *reason = PCRICHARDSON_CONVERGED_RTOL;
  PetscStackCallStandard(HYPRE_BoomerAMGSetTol,(jac->hsolver,jac->tol));
  PetscStackCallStandard(HYPRE_BoomerAMGSetMaxIter,(jac->hsolver,jac->maxiter));
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PCView_HYPRE_BoomerAMG"
static PetscErrorCode PCView_HYPRE_BoomerAMG(PC pc,PetscViewer viewer)
{
  PC_HYPRE       *jac = (PC_HYPRE*)pc->data;
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG preconditioning\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Cycle type %s\n",HYPREBoomerAMGCycleType[jac->cycletype]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Maximum number of levels %d\n",jac->maxlevels);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Maximum number of iterations PER hypre call %d\n",jac->maxiter);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Convergence tolerance PER hypre call %g\n",(double)jac->tol);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Threshold for strong coupling %g\n",(double)jac->strongthreshold);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Interpolation truncation factor %g\n",(double)jac->truncfactor);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Interpolation: max elements per row %d\n",jac->pmax);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Number of levels of aggressive coarsening %d\n",jac->agg_nl);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Number of paths for aggressive coarsening %d\n",jac->agg_num_paths);CHKERRQ(ierr);

    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Maximum row sums %g\n",(double)jac->maxrowsum);CHKERRQ(ierr);

    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Sweeps down         %d\n",jac->gridsweeps[0]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Sweeps up           %d\n",jac->gridsweeps[1]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Sweeps on coarse    %d\n",jac->gridsweeps[2]);CHKERRQ(ierr);

    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Relax down          %s\n",HYPREBoomerAMGRelaxType[jac->relaxtype[0]]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Relax up            %s\n",HYPREBoomerAMGRelaxType[jac->relaxtype[1]]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Relax on coarse     %s\n",HYPREBoomerAMGRelaxType[jac->relaxtype[2]]);CHKERRQ(ierr);

    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Relax weight  (all)      %g\n",(double)jac->relaxweight);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Outer relax weight (all) %g\n",(double)jac->outerrelaxweight);CHKERRQ(ierr);

    if (jac->relaxorder) {
      ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Using CF-relaxation\n");CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Not using CF-relaxation\n");CHKERRQ(ierr);
    }
    if (jac->smoothtype!=-1) {
      ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Smooth type          %s\n",HYPREBoomerAMGSmoothType[jac->smoothtype]);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Smooth num levels    %d\n",jac->smoothnumlevels);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Not using more complex smoothers.\n");CHKERRQ(ierr);
    }
    if (jac->smoothtype==3) {
      ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Euclid ILU(k) levels %d\n",jac->eu_level);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Euclid ILU(k) drop tolerance %g\n",jac->eu_droptolerance);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Euclid ILU use Block-Jacobi? %d\n",jac->eu_bj);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Measure type        %s\n",HYPREBoomerAMGMeasureType[jac->measuretype]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Coarsen type        %s\n",HYPREBoomerAMGCoarsenType[jac->coarsentype]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Interpolation type  %s\n",HYPREBoomerAMGInterpType[jac->interptype]);CHKERRQ(ierr);
    if (jac->nodal_coarsening) {
      ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Using nodal coarsening (with HYPRE_BOOMERAMGSetNodal() %D\n",jac->nodal_coarsening);CHKERRQ(ierr);
    }
    if (jac->vec_interp_variant) {
      ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: HYPRE_BoomerAMGSetInterpVecVariant() %D\n",jac->vec_interp_variant);CHKERRQ(ierr);
    }
    if (jac->nodal_relax) {
      ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Using nodal relaxation via Schwarz smoothing on levels %d\n",jac->nodal_relax_levels);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "PCSetFromOptions_HYPRE_ParaSails"
static PetscErrorCode PCSetFromOptions_HYPRE_ParaSails(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PC_HYPRE       *jac = (PC_HYPRE*)pc->data;
  PetscErrorCode ierr;
  PetscInt       indx;
  PetscBool      flag;
  const char     *symtlist[] = {"nonsymmetric","SPD","nonsymmetric,SPD"};

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"HYPRE ParaSails Options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_hypre_parasails_nlevels","Number of number of levels","None",jac->nlevels,&jac->nlevels,0);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-pc_hypre_parasails_thresh","Threshold","None",jac->threshhold,&jac->threshhold,&flag);CHKERRQ(ierr);
  if (flag) PetscStackCallStandard(HYPRE_ParaSailsSetParams,(jac->hsolver,jac->threshhold,jac->nlevels));

  ierr = PetscOptionsReal("-pc_hypre_parasails_filter","filter","None",jac->filter,&jac->filter,&flag);CHKERRQ(ierr);
  if (flag) PetscStackCallStandard(HYPRE_ParaSailsSetFilter,(jac->hsolver,jac->filter));

  ierr = PetscOptionsReal("-pc_hypre_parasails_loadbal","Load balance","None",jac->loadbal,&jac->loadbal,&flag);CHKERRQ(ierr);
  if (flag) PetscStackCallStandard(HYPRE_ParaSailsSetLoadbal,(jac->hsolver,jac->loadbal));

  ierr = PetscOptionsBool("-pc_hypre_parasails_logging","Print info to screen","None",(PetscBool)jac->logging,(PetscBool*)&jac->logging,&flag);CHKERRQ(ierr);
  if (flag) PetscStackCallStandard(HYPRE_ParaSailsSetLogging,(jac->hsolver,jac->logging));

  ierr = PetscOptionsBool("-pc_hypre_parasails_reuse","Reuse nonzero pattern in preconditioner","None",(PetscBool)jac->ruse,(PetscBool*)&jac->ruse,&flag);CHKERRQ(ierr);
  if (flag) PetscStackCallStandard(HYPRE_ParaSailsSetReuse,(jac->hsolver,jac->ruse));

  ierr = PetscOptionsEList("-pc_hypre_parasails_sym","Symmetry of matrix and preconditioner","None",symtlist,ALEN(symtlist),symtlist[0],&indx,&flag);CHKERRQ(ierr);
  if (flag) {
    jac->symt = indx;
    PetscStackCallStandard(HYPRE_ParaSailsSetSym,(jac->hsolver,jac->symt));
  }

  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCView_HYPRE_ParaSails"
static PetscErrorCode PCView_HYPRE_ParaSails(PC pc,PetscViewer viewer)
{
  PC_HYPRE       *jac = (PC_HYPRE*)pc->data;
  PetscErrorCode ierr;
  PetscBool      iascii;
  const char     *symt = 0;;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE ParaSails preconditioning\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE ParaSails: nlevels %d\n",jac->nlevels);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE ParaSails: threshold %g\n",(double)jac->threshhold);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE ParaSails: filter %g\n",(double)jac->filter);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE ParaSails: load balance %g\n",(double)jac->loadbal);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE ParaSails: reuse nonzero structure %s\n",PetscBools[jac->ruse]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE ParaSails: print info to screen %s\n",PetscBools[jac->logging]);CHKERRQ(ierr);
    if (!jac->symt) symt = "nonsymmetric matrix and preconditioner";
    else if (jac->symt == 1) symt = "SPD matrix and preconditioner";
    else if (jac->symt == 2) symt = "nonsymmetric matrix but SPD preconditioner";
    else SETERRQ1(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONG,"Unknown HYPRE ParaSails symmetric option %d",jac->symt);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE ParaSails: %s\n",symt);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
/* --------------------------------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "PCSetFromOptions_HYPRE_AMS"
static PetscErrorCode PCSetFromOptions_HYPRE_AMS(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PC_HYPRE       *jac = (PC_HYPRE*)pc->data;
  PetscErrorCode ierr;
  PetscInt       n;
  PetscBool      flag,flag2,flag3,flag4;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"HYPRE AMS Options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_hypre_ams_print_level","Debugging output level for AMS","None",jac->as_print,&jac->as_print,&flag);CHKERRQ(ierr);
  if (flag) PetscStackCallStandard(HYPRE_AMSSetPrintLevel,(jac->hsolver,jac->as_print));
  ierr = PetscOptionsInt("-pc_hypre_ams_max_iter","Maximum number of AMS multigrid iterations within PCApply","None",jac->as_max_iter,&jac->as_max_iter,&flag);CHKERRQ(ierr);
  if (flag) PetscStackCallStandard(HYPRE_AMSSetMaxIter,(jac->hsolver,jac->as_max_iter));
  ierr = PetscOptionsInt("-pc_hypre_ams_cycle_type","Cycle type for AMS multigrid","None",jac->ams_cycle_type,&jac->ams_cycle_type,&flag);CHKERRQ(ierr);
  if (flag) PetscStackCallStandard(HYPRE_AMSSetCycleType,(jac->hsolver,jac->ams_cycle_type));
  ierr = PetscOptionsReal("-pc_hypre_ams_tol","Error tolerance for AMS multigrid","None",jac->as_tol,&jac->as_tol,&flag);CHKERRQ(ierr);
  if (flag) PetscStackCallStandard(HYPRE_AMSSetTol,(jac->hsolver,jac->as_tol));
  ierr = PetscOptionsInt("-pc_hypre_ams_relax_type","Relaxation type for AMS smoother","None",jac->as_relax_type,&jac->as_relax_type,&flag);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_hypre_ams_relax_times","Number of relaxation steps for AMS smoother","None",jac->as_relax_times,&jac->as_relax_times,&flag2);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-pc_hypre_ams_relax_weight","Relaxation weight for AMS smoother","None",jac->as_relax_weight,&jac->as_relax_weight,&flag3);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-pc_hypre_ams_omega","SSOR coefficient for AMS smoother","None",jac->as_omega,&jac->as_omega,&flag4);CHKERRQ(ierr);
  if (flag || flag2 || flag3 || flag4) {
    PetscStackCallStandard(HYPRE_AMSSetSmoothingOptions,(jac->hsolver,jac->as_relax_type,
                                                                      jac->as_relax_times,
                                                                      jac->as_relax_weight,
                                                                      jac->as_omega));
  }
  ierr = PetscOptionsReal("-pc_hypre_ams_amg_alpha_theta","Threshold for strong coupling of vector Poisson AMG solver","None",jac->as_amg_alpha_theta,&jac->as_amg_alpha_theta,&flag);CHKERRQ(ierr);
  n = 5;
  ierr = PetscOptionsIntArray("-pc_hypre_ams_amg_alpha_options","AMG options for vector Poisson","None",jac->as_amg_alpha_opts,&n,&flag2);CHKERRQ(ierr);
  if (flag || flag2) {
    PetscStackCallStandard(HYPRE_AMSSetAlphaAMGOptions,(jac->hsolver,jac->as_amg_alpha_opts[0],       /* AMG coarsen type */
                                                                     jac->as_amg_alpha_opts[1],       /* AMG agg_levels */
                                                                     jac->as_amg_alpha_opts[2],       /* AMG relax_type */
                                                                     jac->as_amg_alpha_theta,
                                                                     jac->as_amg_alpha_opts[3],       /* AMG interp_type */
                                                                     jac->as_amg_alpha_opts[4]));     /* AMG Pmax */
  }
  ierr = PetscOptionsReal("-pc_hypre_ams_amg_beta_theta","Threshold for strong coupling of scalar Poisson AMG solver","None",jac->as_amg_beta_theta,&jac->as_amg_beta_theta,&flag);CHKERRQ(ierr);
  n = 5;
  ierr = PetscOptionsIntArray("-pc_hypre_ams_amg_beta_options","AMG options for scalar Poisson solver","None",jac->as_amg_beta_opts,&n,&flag2);CHKERRQ(ierr);
  if (flag || flag2) {
    PetscStackCallStandard(HYPRE_AMSSetBetaAMGOptions,(jac->hsolver,jac->as_amg_beta_opts[0],       /* AMG coarsen type */
                                                                    jac->as_amg_beta_opts[1],       /* AMG agg_levels */
                                                                    jac->as_amg_beta_opts[2],       /* AMG relax_type */
                                                                    jac->as_amg_beta_theta,
                                                                    jac->as_amg_beta_opts[3],       /* AMG interp_type */
                                                                    jac->as_amg_beta_opts[4]));     /* AMG Pmax */
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCView_HYPRE_AMS"
static PetscErrorCode PCView_HYPRE_AMS(PC pc,PetscViewer viewer)
{
  PC_HYPRE       *jac = (PC_HYPRE*)pc->data;
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE AMS preconditioning\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE AMS: subspace iterations per application %d\n",jac->as_max_iter);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE AMS: subspace cycle type %d\n",jac->ams_cycle_type);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE AMS: subspace iteration tolerance %g\n",jac->as_tol);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE AMS: smoother type %d\n",jac->as_relax_type);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE AMS: number of smoothing steps %d\n",jac->as_relax_times);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE AMS: smoother weight %g\n",jac->as_relax_weight);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE AMS: smoother omega %g\n",jac->as_omega);CHKERRQ(ierr);
    if (jac->alpha_Poisson) {
      ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE AMS: vector Poisson solver (passed in by user)\n");CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE AMS: vector Poisson solver (computed) \n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE AMS:     boomerAMG coarsening type %d\n",jac->as_amg_alpha_opts[0]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE AMS:     boomerAMG levels of aggressive coarsening %d\n",jac->as_amg_alpha_opts[1]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE AMS:     boomerAMG relaxation type %d\n",jac->as_amg_alpha_opts[2]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE AMS:     boomerAMG interpolation type %d\n",jac->as_amg_alpha_opts[3]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE AMS:     boomerAMG max nonzero elements in interpolation rows %d\n",jac->as_amg_alpha_opts[4]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE AMS:     boomerAMG strength threshold %g\n",jac->as_amg_alpha_theta);CHKERRQ(ierr);
    if (!jac->ams_beta_is_zero) {
      if (jac->beta_Poisson) {
        ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE AMS: scalar Poisson solver (passed in by user)\n");CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE AMS: scalar Poisson solver (computed) \n");CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE AMS:     boomerAMG coarsening type %d\n",jac->as_amg_beta_opts[0]);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE AMS:     boomerAMG levels of aggressive coarsening %d\n",jac->as_amg_beta_opts[1]);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE AMS:     boomerAMG relaxation type %d\n",jac->as_amg_beta_opts[2]);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE AMS:     boomerAMG interpolation type %d\n",jac->as_amg_beta_opts[3]);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE AMS:     boomerAMG max nonzero elements in interpolation rows %d\n",jac->as_amg_beta_opts[4]);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE AMS:     boomerAMG strength threshold %g\n",jac->as_amg_beta_theta);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCSetFromOptions_HYPRE_ADS"
static PetscErrorCode PCSetFromOptions_HYPRE_ADS(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PC_HYPRE       *jac = (PC_HYPRE*)pc->data;
  PetscErrorCode ierr;
  PetscInt       n;
  PetscBool      flag,flag2,flag3,flag4;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"HYPRE ADS Options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_hypre_ads_print_level","Debugging output level for ADS","None",jac->as_print,&jac->as_print,&flag);CHKERRQ(ierr);
  if (flag) PetscStackCallStandard(HYPRE_ADSSetPrintLevel,(jac->hsolver,jac->as_print));
  ierr = PetscOptionsInt("-pc_hypre_ads_max_iter","Maximum number of ADS multigrid iterations within PCApply","None",jac->as_max_iter,&jac->as_max_iter,&flag);CHKERRQ(ierr);
  if (flag) PetscStackCallStandard(HYPRE_ADSSetMaxIter,(jac->hsolver,jac->as_max_iter));
  ierr = PetscOptionsInt("-pc_hypre_ads_cycle_type","Cycle type for ADS multigrid","None",jac->ads_cycle_type,&jac->ads_cycle_type,&flag);CHKERRQ(ierr);
  if (flag) PetscStackCallStandard(HYPRE_ADSSetCycleType,(jac->hsolver,jac->ads_cycle_type));
  ierr = PetscOptionsReal("-pc_hypre_ads_tol","Error tolerance for ADS multigrid","None",jac->as_tol,&jac->as_tol,&flag);CHKERRQ(ierr);
  if (flag) PetscStackCallStandard(HYPRE_ADSSetTol,(jac->hsolver,jac->as_tol));
  ierr = PetscOptionsInt("-pc_hypre_ads_relax_type","Relaxation type for ADS smoother","None",jac->as_relax_type,&jac->as_relax_type,&flag);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_hypre_ads_relax_times","Number of relaxation steps for ADS smoother","None",jac->as_relax_times,&jac->as_relax_times,&flag2);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-pc_hypre_ads_relax_weight","Relaxation weight for ADS smoother","None",jac->as_relax_weight,&jac->as_relax_weight,&flag3);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-pc_hypre_ads_omega","SSOR coefficient for ADS smoother","None",jac->as_omega,&jac->as_omega,&flag4);CHKERRQ(ierr);
  if (flag || flag2 || flag3 || flag4) {
    PetscStackCallStandard(HYPRE_ADSSetSmoothingOptions,(jac->hsolver,jac->as_relax_type,
                                                                      jac->as_relax_times,
                                                                      jac->as_relax_weight,
                                                                      jac->as_omega));
  }
  ierr = PetscOptionsReal("-pc_hypre_ads_ams_theta","Threshold for strong coupling of AMS solver inside ADS","None",jac->as_amg_alpha_theta,&jac->as_amg_alpha_theta,&flag);CHKERRQ(ierr);
  n = 5;
  ierr = PetscOptionsIntArray("-pc_hypre_ads_ams_options","AMG options for AMS solver inside ADS","None",jac->as_amg_alpha_opts,&n,&flag2);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_hypre_ads_ams_cycle_type","Cycle type for AMS solver inside ADS","None",jac->ams_cycle_type,&jac->ams_cycle_type,&flag3);CHKERRQ(ierr);
  if (flag || flag2 || flag3) {
    PetscStackCallStandard(HYPRE_ADSSetAMSOptions,(jac->hsolver,jac->ams_cycle_type,             /* AMS cycle type */
                                                                jac->as_amg_alpha_opts[0],       /* AMG coarsen type */
                                                                jac->as_amg_alpha_opts[1],       /* AMG agg_levels */
                                                                jac->as_amg_alpha_opts[2],       /* AMG relax_type */
                                                                jac->as_amg_alpha_theta,
                                                                jac->as_amg_alpha_opts[3],       /* AMG interp_type */
                                                                jac->as_amg_alpha_opts[4]));     /* AMG Pmax */
  }
  ierr = PetscOptionsReal("-pc_hypre_ads_amg_theta","Threshold for strong coupling of vector AMG solver inside ADS","None",jac->as_amg_beta_theta,&jac->as_amg_beta_theta,&flag);CHKERRQ(ierr);
  n = 5;
  ierr = PetscOptionsIntArray("-pc_hypre_ads_amg_options","AMG options for vector AMG solver inside ADS","None",jac->as_amg_beta_opts,&n,&flag2);CHKERRQ(ierr);
  if (flag || flag2) {
    PetscStackCallStandard(HYPRE_ADSSetAMGOptions,(jac->hsolver,jac->as_amg_beta_opts[0],       /* AMG coarsen type */
                                                                jac->as_amg_beta_opts[1],       /* AMG agg_levels */
                                                                jac->as_amg_beta_opts[2],       /* AMG relax_type */
                                                                jac->as_amg_beta_theta,
                                                                jac->as_amg_beta_opts[3],       /* AMG interp_type */
                                                                jac->as_amg_beta_opts[4]));     /* AMG Pmax */
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCView_HYPRE_ADS"
static PetscErrorCode PCView_HYPRE_ADS(PC pc,PetscViewer viewer)
{
  PC_HYPRE       *jac = (PC_HYPRE*)pc->data;
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE ADS preconditioning\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE ADS: subspace iterations per application %d\n",jac->as_max_iter);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE ADS: subspace cycle type %d\n",jac->ads_cycle_type);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE ADS: subspace iteration tolerance %g\n",jac->as_tol);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE ADS: smoother type %d\n",jac->as_relax_type);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE ADS: number of smoothing steps %d\n",jac->as_relax_times);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE ADS: smoother weight %g\n",jac->as_relax_weight);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE ADS: smoother omega %g\n",jac->as_omega);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE ADS: AMS solver\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE ADS:     subspace cycle type %d\n",jac->ams_cycle_type);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE ADS:     boomerAMG coarsening type %d\n",jac->as_amg_alpha_opts[0]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE ADS:     boomerAMG levels of aggressive coarsening %d\n",jac->as_amg_alpha_opts[1]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE ADS:     boomerAMG relaxation type %d\n",jac->as_amg_alpha_opts[2]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE ADS:     boomerAMG interpolation type %d\n",jac->as_amg_alpha_opts[3]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE ADS:     boomerAMG max nonzero elements in interpolation rows %d\n",jac->as_amg_alpha_opts[4]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE ADS:     boomerAMG strength threshold %g\n",jac->as_amg_alpha_theta);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE ADS: vector Poisson solver\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE ADS:     boomerAMG coarsening type %d\n",jac->as_amg_beta_opts[0]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE ADS:     boomerAMG levels of aggressive coarsening %d\n",jac->as_amg_beta_opts[1]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE ADS:     boomerAMG relaxation type %d\n",jac->as_amg_beta_opts[2]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE ADS:     boomerAMG interpolation type %d\n",jac->as_amg_beta_opts[3]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE ADS:     boomerAMG max nonzero elements in interpolation rows %d\n",jac->as_amg_beta_opts[4]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE ADS:     boomerAMG strength threshold %g\n",jac->as_amg_beta_theta);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCHYPRESetDiscreteGradient_HYPRE"
static PetscErrorCode PCHYPRESetDiscreteGradient_HYPRE(PC pc, Mat G)
{
  PC_HYPRE           *jac = (PC_HYPRE*)pc->data;
  HYPRE_ParCSRMatrix parcsr_G;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  /* throw away any discrete gradient if already set */
  if (jac->G) PetscStackCallStandard(HYPRE_IJMatrixDestroy,(jac->G));
  ierr = MatHYPRE_IJMatrixCreate(G,&jac->G);CHKERRQ(ierr);
  ierr = MatHYPRE_IJMatrixCopy(G,jac->G);CHKERRQ(ierr);
  PetscStackCallStandard(HYPRE_IJMatrixGetObject,(jac->G,(void**)(&parcsr_G)));
  PetscStackCall("Hypre set gradient",ierr = (*jac->setdgrad)(jac->hsolver,parcsr_G);CHKERRQ(ierr););
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCHYPRESetDiscreteGradient"
/*@
 PCHYPRESetDiscreteGradient - Set discrete gradient matrix

   Collective on PC

   Input Parameters:
+  pc - the preconditioning context
-  G - the discrete gradient

   Level: intermediate

   Notes: G should have as many rows as the number of edges and as many columns as the number of vertices in the mesh
          Each row of G has 2 nonzeros, with column indexes being the global indexes of edge's endpoints: matrix entries are +1 and -1 depending on edge orientation

.seealso:
@*/
PetscErrorCode PCHYPRESetDiscreteGradient(PC pc, Mat G)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(G,MAT_CLASSID,2);
  PetscCheckSameComm(pc,1,G,2);
  ierr = PetscTryMethod(pc,"PCHYPRESetDiscreteGradient_C",(PC,Mat),(pc,G));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCHYPRESetDiscreteCurl_HYPRE"
static PetscErrorCode PCHYPRESetDiscreteCurl_HYPRE(PC pc, Mat C)
{
  PC_HYPRE           *jac = (PC_HYPRE*)pc->data;
  HYPRE_ParCSRMatrix parcsr_C;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  /* throw away any discrete curl if already set */
  if (jac->C) PetscStackCallStandard(HYPRE_IJMatrixDestroy,(jac->C));
  ierr = MatHYPRE_IJMatrixCreate(C,&jac->C);CHKERRQ(ierr);
  ierr = MatHYPRE_IJMatrixCopy(C,jac->C);CHKERRQ(ierr);
  PetscStackCallStandard(HYPRE_IJMatrixGetObject,(jac->C,(void**)(&parcsr_C)));
  PetscStackCall("Hypre set curl",ierr = (*jac->setdcurl)(jac->hsolver,parcsr_C);CHKERRQ(ierr););
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCHYPRESetDiscreteCurl"
/*@
 PCHYPRESetDiscreteCurl - Set discrete curl matrix

   Collective on PC

   Input Parameters:
+  pc - the preconditioning context
-  C - the discrete curl

   Level: intermediate

   Notes: C should have as many rows as the number of faces and as many columns as the number of edges in the mesh
          Each row of G has as many nonzeros as the number of edges of a face, with column indexes being the global indexes of the corresponding edge: matrix entries are +1 and -1 depending on edge orientation with respect to the face orientation

.seealso:
@*/
PetscErrorCode PCHYPRESetDiscreteCurl(PC pc, Mat C)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(C,MAT_CLASSID,2);
  PetscCheckSameComm(pc,1,C,2);
  ierr = PetscTryMethod(pc,"PCHYPRESetDiscreteCurl_C",(PC,Mat),(pc,C));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCHYPRESetAlphaPoissonMatrix_HYPRE_AMS"
static PetscErrorCode PCHYPRESetAlphaPoissonMatrix_HYPRE_AMS(PC pc, Mat A)
{
  PC_HYPRE           *jac = (PC_HYPRE*)pc->data;
  HYPRE_ParCSRMatrix parcsr_alpha_Poisson;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  /* throw away any matrix if already set */
  if (jac->alpha_Poisson) PetscStackCallStandard(HYPRE_IJMatrixDestroy,(jac->alpha_Poisson));
  ierr = MatHYPRE_IJMatrixCreate(A,&jac->alpha_Poisson);CHKERRQ(ierr);
  ierr = MatHYPRE_IJMatrixCopy(A,jac->alpha_Poisson);CHKERRQ(ierr);
  PetscStackCallStandard(HYPRE_IJMatrixGetObject,(jac->alpha_Poisson,(void**)(&parcsr_alpha_Poisson)));
  PetscStackCallStandard(HYPRE_AMSSetAlphaPoissonMatrix,(jac->hsolver,parcsr_alpha_Poisson));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCHYPRESetAlphaPoissonMatrix"
/*@
 PCHYPRESetAlphaPoissonMatrix - Set vector Poisson matrix

   Collective on PC

   Input Parameters:
+  pc - the preconditioning context
-  A - the matrix

   Level: intermediate

   Notes: A should be obtained by discretizing the vector valued Poisson problem with linear finite elements

.seealso:
@*/
PetscErrorCode PCHYPRESetAlphaPoissonMatrix(PC pc, Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(A,MAT_CLASSID,2);
  PetscCheckSameComm(pc,1,A,2);
  ierr = PetscTryMethod(pc,"PCHYPRESetAlphaPoissonMatrix_C",(PC,Mat),(pc,A));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCHYPRESetBetaPoissonMatrix_HYPRE_AMS"
static PetscErrorCode PCHYPRESetBetaPoissonMatrix_HYPRE_AMS(PC pc, Mat A)
{
  PC_HYPRE           *jac = (PC_HYPRE*)pc->data;
  HYPRE_ParCSRMatrix parcsr_beta_Poisson;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (!A) {
    PetscStackCallStandard(HYPRE_AMSSetBetaPoissonMatrix,(jac->hsolver,NULL));
    jac->ams_beta_is_zero = PETSC_TRUE;
    PetscFunctionReturn(0);
  }
  jac->ams_beta_is_zero = PETSC_FALSE;
  /* throw away any matrix if already set */
  if (jac->beta_Poisson) PetscStackCallStandard(HYPRE_IJMatrixDestroy,(jac->beta_Poisson));
  ierr = MatHYPRE_IJMatrixCreate(A,&jac->beta_Poisson);CHKERRQ(ierr);
  ierr = MatHYPRE_IJMatrixCopy(A,jac->beta_Poisson);CHKERRQ(ierr);
  PetscStackCallStandard(HYPRE_IJMatrixGetObject,(jac->beta_Poisson,(void**)(&parcsr_beta_Poisson)));
  PetscStackCallStandard(HYPRE_AMSSetBetaPoissonMatrix,(jac->hsolver,parcsr_beta_Poisson));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCHYPRESetBetaPoissonMatrix"
/*@
 PCHYPRESetBetaPoissonMatrix - Set Poisson matrix

   Collective on PC

   Input Parameters:
+  pc - the preconditioning context
-  A - the matrix

   Level: intermediate

   Notes: A should be obtained by discretizing the Poisson problem with linear finite elements.
          Following HYPRE convention, the scalar Poisson solver of AMS can be turned off by passing NULL.

.seealso:
@*/
PetscErrorCode PCHYPRESetBetaPoissonMatrix(PC pc, Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  if (A) {
    PetscValidHeaderSpecific(A,MAT_CLASSID,2);
    PetscCheckSameComm(pc,1,A,2);
  }
  ierr = PetscTryMethod(pc,"PCHYPRESetBetaPoissonMatrix_C",(PC,Mat),(pc,A));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCHYPRESetEdgeConstantVectors_HYPRE_AMS"
static PetscErrorCode PCHYPRESetEdgeConstantVectors_HYPRE_AMS(PC pc,Vec ozz, Vec zoz, Vec zzo)
{
  PC_HYPRE           *jac = (PC_HYPRE*)pc->data;
  HYPRE_ParVector    par_ozz,par_zoz,par_zzo;
  PetscInt           dim;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  /* throw away any vector if already set */
  if (jac->constants[0]) PetscStackCallStandard(HYPRE_IJVectorDestroy,(jac->constants[0]));
  if (jac->constants[1]) PetscStackCallStandard(HYPRE_IJVectorDestroy,(jac->constants[1]));
  if (jac->constants[2]) PetscStackCallStandard(HYPRE_IJVectorDestroy,(jac->constants[2]));
  jac->constants[0] = NULL;
  jac->constants[1] = NULL;
  jac->constants[2] = NULL;
  ierr = VecHYPRE_IJVectorCreate(ozz,&jac->constants[0]);CHKERRQ(ierr);
  ierr = VecHYPRE_IJVectorCopy(ozz,jac->constants[0]);CHKERRQ(ierr);
  PetscStackCallStandard(HYPRE_IJVectorGetObject,(jac->constants[0],(void**)(&par_ozz)));
  ierr = VecHYPRE_IJVectorCreate(zoz,&jac->constants[1]);CHKERRQ(ierr);
  ierr = VecHYPRE_IJVectorCopy(zoz,jac->constants[1]);CHKERRQ(ierr);
  PetscStackCallStandard(HYPRE_IJVectorGetObject,(jac->constants[1],(void**)(&par_zoz)));
  dim = 2;
  if (zzo) {
    ierr = VecHYPRE_IJVectorCreate(zzo,&jac->constants[2]);CHKERRQ(ierr);
    ierr = VecHYPRE_IJVectorCopy(zzo,jac->constants[2]);CHKERRQ(ierr);
    PetscStackCallStandard(HYPRE_IJVectorGetObject,(jac->constants[2],(void**)(&par_zzo)));
    dim++;
  }
  PetscStackCallStandard(HYPRE_AMSSetEdgeConstantVectors,(jac->hsolver,par_ozz,par_zoz,par_zzo));
  PetscStackCallStandard(HYPRE_AMSSetDimension,(jac->hsolver,dim));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCHYPRESetEdgeConstantVectors"
/*@
 PCHYPRESetEdgeConstantVectors - Set the representation of the constant vector fields in edge element basis

   Collective on PC

   Input Parameters:
+  pc - the preconditioning context
-  ozz - vector representing (1,0,0) (or (1,0) in 2D)
-  zoz - vector representing (0,1,0) (or (0,1) in 2D)
-  zzo - vector representing (0,0,1) (use NULL in 2D)

   Level: intermediate

   Notes:

.seealso:
@*/
PetscErrorCode PCHYPRESetEdgeConstantVectors(PC pc, Vec ozz, Vec zoz, Vec zzo)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(ozz,VEC_CLASSID,2);
  PetscValidHeaderSpecific(zoz,VEC_CLASSID,3);
  if (zzo) PetscValidHeaderSpecific(zzo,VEC_CLASSID,4);
  PetscCheckSameComm(pc,1,ozz,2);
  PetscCheckSameComm(pc,1,zoz,3);
  if (zzo) PetscCheckSameComm(pc,1,zzo,4);
  ierr = PetscTryMethod(pc,"PCHYPRESetEdgeConstantVectors_C",(PC,Vec,Vec,Vec),(pc,ozz,zoz,zzo));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCSetCoordinates_HYPRE"
static PetscErrorCode PCSetCoordinates_HYPRE(PC pc, PetscInt dim, PetscInt nloc, PetscReal *coords)
{
  PC_HYPRE        *jac = (PC_HYPRE*)pc->data;
  Vec             tv;
  HYPRE_ParVector par_coords[3];
  PetscInt        i;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  /* throw away any coordinate vector if already set */
  if (jac->coords[0]) PetscStackCallStandard(HYPRE_IJVectorDestroy,(jac->coords[0]));
  if (jac->coords[1]) PetscStackCallStandard(HYPRE_IJVectorDestroy,(jac->coords[1]));
  if (jac->coords[2]) PetscStackCallStandard(HYPRE_IJVectorDestroy,(jac->coords[2]));
  /* set problem's dimension */
  if (jac->setdim) {
    PetscStackCall("Hypre set dim",ierr = (*jac->setdim)(jac->hsolver,dim);CHKERRQ(ierr););
  }
  /* compute IJ vector for coordinates */
  ierr = VecCreate(PetscObjectComm((PetscObject)pc),&tv);CHKERRQ(ierr);
  ierr = VecSetType(tv,VECSTANDARD);CHKERRQ(ierr);
  ierr = VecSetSizes(tv,nloc,PETSC_DECIDE);CHKERRQ(ierr);
  for (i=0;i<dim;i++) {
    PetscScalar *array;
    PetscInt    j;

    ierr = VecHYPRE_IJVectorCreate(tv,&jac->coords[i]);CHKERRQ(ierr);
    ierr = VecGetArray(tv,&array);CHKERRQ(ierr);
    for (j=0;j<nloc;j++) {
      array[j] = coords[j*dim+i];
    }
    PetscStackCallStandard(HYPRE_IJVectorSetValues,(jac->coords[i],nloc,NULL,array));
    PetscStackCallStandard(HYPRE_IJVectorAssemble,(jac->coords[i]));
    ierr = VecRestoreArray(tv,&array);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&tv);CHKERRQ(ierr);
  /* pass parCSR vectors to AMS solver */
  par_coords[0] = NULL;
  par_coords[1] = NULL;
  par_coords[2] = NULL;
  if (jac->coords[0]) PetscStackCallStandard(HYPRE_IJVectorGetObject,(jac->coords[0],(void**)(&par_coords[0])));
  if (jac->coords[1]) PetscStackCallStandard(HYPRE_IJVectorGetObject,(jac->coords[1],(void**)(&par_coords[1])));
  if (jac->coords[2]) PetscStackCallStandard(HYPRE_IJVectorGetObject,(jac->coords[2],(void**)(&par_coords[2])));
  PetscStackCall("Hypre set coords",ierr = (*jac->setcoord)(jac->hsolver,par_coords[0],par_coords[1],par_coords[2]);CHKERRQ(ierr););
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------*/

#undef __FUNCT__
#define __FUNCT__ "PCHYPREGetType_HYPRE"
static PetscErrorCode  PCHYPREGetType_HYPRE(PC pc,const char *name[])
{
  PC_HYPRE *jac = (PC_HYPRE*)pc->data;

  PetscFunctionBegin;
  *name = jac->hypre_type;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCHYPRESetType_HYPRE"
static PetscErrorCode  PCHYPRESetType_HYPRE(PC pc,const char name[])
{
  PC_HYPRE       *jac = (PC_HYPRE*)pc->data;
  PetscErrorCode ierr;
  PetscBool      flag;

  PetscFunctionBegin;
  if (jac->hypre_type) {
    ierr = PetscStrcmp(jac->hypre_type,name,&flag);CHKERRQ(ierr);
    if (!flag) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_ORDER,"Cannot reset the HYPRE preconditioner type once it has been set");
    PetscFunctionReturn(0);
  } else {
    ierr = PetscStrallocpy(name, &jac->hypre_type);CHKERRQ(ierr);
  }

  jac->maxiter         = PETSC_DEFAULT;
  jac->tol             = PETSC_DEFAULT;
  jac->printstatistics = PetscLogPrintInfo;

  ierr = PetscStrcmp("pilut",jac->hypre_type,&flag);CHKERRQ(ierr);
  if (flag) {
    PetscStackCallStandard(HYPRE_ParCSRPilutCreate,(jac->comm_hypre,&jac->hsolver));
    pc->ops->setfromoptions = PCSetFromOptions_HYPRE_Pilut;
    pc->ops->view           = PCView_HYPRE_Pilut;
    jac->destroy            = HYPRE_ParCSRPilutDestroy;
    jac->setup              = HYPRE_ParCSRPilutSetup;
    jac->solve              = HYPRE_ParCSRPilutSolve;
    jac->factorrowsize      = PETSC_DEFAULT;
    PetscFunctionReturn(0);
  }
  ierr = PetscStrcmp("parasails",jac->hypre_type,&flag);CHKERRQ(ierr);
  if (flag) {
    PetscStackCallStandard(HYPRE_ParaSailsCreate,(jac->comm_hypre,&jac->hsolver));
    pc->ops->setfromoptions = PCSetFromOptions_HYPRE_ParaSails;
    pc->ops->view           = PCView_HYPRE_ParaSails;
    jac->destroy            = HYPRE_ParaSailsDestroy;
    jac->setup              = HYPRE_ParaSailsSetup;
    jac->solve              = HYPRE_ParaSailsSolve;
    /* initialize */
    jac->nlevels    = 1;
    jac->threshhold = .1;
    jac->filter     = .1;
    jac->loadbal    = 0;
    if (PetscLogPrintInfo) jac->logging = (int) PETSC_TRUE;
    else jac->logging = (int) PETSC_FALSE;

    jac->ruse = (int) PETSC_FALSE;
    jac->symt = 0;
    PetscStackCallStandard(HYPRE_ParaSailsSetParams,(jac->hsolver,jac->threshhold,jac->nlevels));
    PetscStackCallStandard(HYPRE_ParaSailsSetFilter,(jac->hsolver,jac->filter));
    PetscStackCallStandard(HYPRE_ParaSailsSetLoadbal,(jac->hsolver,jac->loadbal));
    PetscStackCallStandard(HYPRE_ParaSailsSetLogging,(jac->hsolver,jac->logging));
    PetscStackCallStandard(HYPRE_ParaSailsSetReuse,(jac->hsolver,jac->ruse));
    PetscStackCallStandard(HYPRE_ParaSailsSetSym,(jac->hsolver,jac->symt));
    PetscFunctionReturn(0);
  }
  ierr = PetscStrcmp("boomeramg",jac->hypre_type,&flag);CHKERRQ(ierr);
  if (flag) {
    ierr                     = HYPRE_BoomerAMGCreate(&jac->hsolver);
    pc->ops->setfromoptions  = PCSetFromOptions_HYPRE_BoomerAMG;
    pc->ops->view            = PCView_HYPRE_BoomerAMG;
    pc->ops->applytranspose  = PCApplyTranspose_HYPRE_BoomerAMG;
    pc->ops->applyrichardson = PCApplyRichardson_HYPRE_BoomerAMG;
    jac->destroy             = HYPRE_BoomerAMGDestroy;
    jac->setup               = HYPRE_BoomerAMGSetup;
    jac->solve               = HYPRE_BoomerAMGSolve;
    jac->applyrichardson     = PETSC_FALSE;
    /* these defaults match the hypre defaults */
    jac->cycletype        = 1;
    jac->maxlevels        = 25;
    jac->maxiter          = 1;
    jac->tol              = 0.0; /* tolerance of zero indicates use as preconditioner (suppresses convergence errors) */
    jac->truncfactor      = 0.0;
    jac->strongthreshold  = .25;
    jac->maxrowsum        = .9;
    jac->coarsentype      = 6;
    jac->measuretype      = 0;
    jac->gridsweeps[0]    = jac->gridsweeps[1] = jac->gridsweeps[2] = 1;
    jac->smoothtype       = -1; /* Not set by default */
    jac->smoothnumlevels  = 25;
    jac->eu_level         = 0;
    jac->eu_droptolerance = 0;
    jac->eu_bj            = 0;
    jac->relaxtype[0]     = jac->relaxtype[1] = 6; /* Defaults to SYMMETRIC since in PETSc we are using a a PC - most likely with CG */
    jac->relaxtype[2]     = 9; /*G.E. */
    jac->relaxweight      = 1.0;
    jac->outerrelaxweight = 1.0;
    jac->relaxorder       = 1;
    jac->interptype       = 0;
    jac->agg_nl           = 0;
    jac->pmax             = 0;
    jac->truncfactor      = 0.0;
    jac->agg_num_paths    = 1;

    jac->nodal_coarsen      = 0;
    jac->nodal_relax        = PETSC_FALSE;
    jac->nodal_relax_levels = 1;
    PetscStackCallStandard(HYPRE_BoomerAMGSetCycleType,(jac->hsolver,jac->cycletype));
    PetscStackCallStandard(HYPRE_BoomerAMGSetMaxLevels,(jac->hsolver,jac->maxlevels));
    PetscStackCallStandard(HYPRE_BoomerAMGSetMaxIter,(jac->hsolver,jac->maxiter));
    PetscStackCallStandard(HYPRE_BoomerAMGSetTol,(jac->hsolver,jac->tol));
    PetscStackCallStandard(HYPRE_BoomerAMGSetTruncFactor,(jac->hsolver,jac->truncfactor));
    PetscStackCallStandard(HYPRE_BoomerAMGSetStrongThreshold,(jac->hsolver,jac->strongthreshold));
    PetscStackCallStandard(HYPRE_BoomerAMGSetMaxRowSum,(jac->hsolver,jac->maxrowsum));
    PetscStackCallStandard(HYPRE_BoomerAMGSetCoarsenType,(jac->hsolver,jac->coarsentype));
    PetscStackCallStandard(HYPRE_BoomerAMGSetMeasureType,(jac->hsolver,jac->measuretype));
    PetscStackCallStandard(HYPRE_BoomerAMGSetRelaxOrder,(jac->hsolver, jac->relaxorder));
    PetscStackCallStandard(HYPRE_BoomerAMGSetInterpType,(jac->hsolver,jac->interptype));
    PetscStackCallStandard(HYPRE_BoomerAMGSetAggNumLevels,(jac->hsolver,jac->agg_nl));
    PetscStackCallStandard(HYPRE_BoomerAMGSetPMaxElmts,(jac->hsolver,jac->pmax));
    PetscStackCallStandard(HYPRE_BoomerAMGSetNumPaths,(jac->hsolver,jac->agg_num_paths));
    PetscStackCallStandard(HYPRE_BoomerAMGSetRelaxType,(jac->hsolver, jac->relaxtype[0]));  /*defaults coarse to 9*/
    PetscStackCallStandard(HYPRE_BoomerAMGSetNumSweeps,(jac->hsolver, jac->gridsweeps[0])); /*defaults coarse to 1 */
    PetscFunctionReturn(0);
  }
  ierr = PetscStrcmp("ams",jac->hypre_type,&flag);CHKERRQ(ierr);
  if (flag) {
    ierr                     = HYPRE_AMSCreate(&jac->hsolver);
    pc->ops->setfromoptions  = PCSetFromOptions_HYPRE_AMS;
    pc->ops->view            = PCView_HYPRE_AMS;
    jac->destroy             = HYPRE_AMSDestroy;
    jac->setup               = HYPRE_AMSSetup;
    jac->solve               = HYPRE_AMSSolve;
    jac->setdgrad            = HYPRE_AMSSetDiscreteGradient;
    jac->setcoord            = HYPRE_AMSSetCoordinateVectors;
    jac->setdim              = HYPRE_AMSSetDimension;
    jac->coords[0]           = NULL;
    jac->coords[1]           = NULL;
    jac->coords[2]           = NULL;
    jac->G                   = NULL;
    /* solver parameters: these are borrowed from mfem package, and they are not the default values from HYPRE AMS */
    jac->as_print           = 0;
    jac->as_max_iter        = 1; /* used as a preconditioner */
    jac->as_tol             = 0.; /* used as a preconditioner */
    jac->ams_cycle_type     = 13;
    /* Smoothing options */
    jac->as_relax_type      = 2;
    jac->as_relax_times     = 1;
    jac->as_relax_weight    = 1.0;
    jac->as_omega           = 1.0;
    /* Vector valued Poisson AMG solver parameters: coarsen type, agg_levels, relax_type, interp_type, Pmax */
    jac->as_amg_alpha_opts[0] = 10;
    jac->as_amg_alpha_opts[1] = 1;
    jac->as_amg_alpha_opts[2] = 6;
    jac->as_amg_alpha_opts[3] = 6;
    jac->as_amg_alpha_opts[4] = 4;
    jac->as_amg_alpha_theta   = 0.25;
    /* Scalar Poisson AMG solver parameters: coarsen type, agg_levels, relax_type, interp_type, Pmax */
    jac->ams_beta_is_zero = PETSC_FALSE;
    jac->as_amg_beta_opts[0] = 10;
    jac->as_amg_beta_opts[1] = 1;
    jac->as_amg_beta_opts[2] = 6;
    jac->as_amg_beta_opts[3] = 6;
    jac->as_amg_beta_opts[4] = 4;
    jac->as_amg_beta_theta   = 0.25;
    PetscStackCallStandard(HYPRE_AMSSetPrintLevel,(jac->hsolver,jac->as_print));
    PetscStackCallStandard(HYPRE_AMSSetMaxIter,(jac->hsolver,jac->as_max_iter));
    PetscStackCallStandard(HYPRE_AMSSetCycleType,(jac->hsolver,jac->ams_cycle_type));
    PetscStackCallStandard(HYPRE_AMSSetTol,(jac->hsolver,jac->as_tol));
    PetscStackCallStandard(HYPRE_AMSSetSmoothingOptions,(jac->hsolver,jac->as_relax_type,
                                                                      jac->as_relax_times,
                                                                      jac->as_relax_weight,
                                                                      jac->as_omega));
    PetscStackCallStandard(HYPRE_AMSSetAlphaAMGOptions,(jac->hsolver,jac->as_amg_alpha_opts[0],       /* AMG coarsen type */
                                                                     jac->as_amg_alpha_opts[1],       /* AMG agg_levels */
                                                                     jac->as_amg_alpha_opts[2],       /* AMG relax_type */
                                                                     jac->as_amg_alpha_theta,
                                                                     jac->as_amg_alpha_opts[3],       /* AMG interp_type */
                                                                     jac->as_amg_alpha_opts[4]));     /* AMG Pmax */
    PetscStackCallStandard(HYPRE_AMSSetBetaAMGOptions,(jac->hsolver,jac->as_amg_beta_opts[0],       /* AMG coarsen type */
                                                                    jac->as_amg_beta_opts[1],       /* AMG agg_levels */
                                                                    jac->as_amg_beta_opts[2],       /* AMG relax_type */
                                                                    jac->as_amg_beta_theta,
                                                                    jac->as_amg_beta_opts[3],       /* AMG interp_type */
                                                                    jac->as_amg_beta_opts[4]));     /* AMG Pmax */
    ierr = PetscObjectComposeFunction((PetscObject)pc,"PCSetCoordinates_C",PCSetCoordinates_HYPRE);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)pc,"PCHYPRESetDiscreteGradient_C",PCHYPRESetDiscreteGradient_HYPRE);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)pc,"PCHYPRESetEdgeConstantVectors_C",PCHYPRESetEdgeConstantVectors_HYPRE_AMS);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)pc,"PCHYPRESetAlphaPoissonMatrix_C",PCHYPRESetAlphaPoissonMatrix_HYPRE_AMS);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)pc,"PCHYPRESetBetaPoissonMatrix_C",PCHYPRESetBetaPoissonMatrix_HYPRE_AMS);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = PetscStrcmp("ads",jac->hypre_type,&flag);CHKERRQ(ierr);
  if (flag) {
    ierr                     = HYPRE_ADSCreate(&jac->hsolver);
    pc->ops->setfromoptions  = PCSetFromOptions_HYPRE_ADS;
    pc->ops->view            = PCView_HYPRE_ADS;
    jac->destroy             = HYPRE_ADSDestroy;
    jac->setup               = HYPRE_ADSSetup;
    jac->solve               = HYPRE_ADSSolve;
    jac->setdgrad            = HYPRE_ADSSetDiscreteGradient;
    jac->setdcurl            = HYPRE_ADSSetDiscreteCurl;
    jac->setcoord            = HYPRE_ADSSetCoordinateVectors;
    jac->coords[0]           = NULL;
    jac->coords[1]           = NULL;
    jac->coords[2]           = NULL;
    jac->G                   = NULL;
    jac->C                   = NULL;
    /* solver parameters: these are borrowed from mfem package, and they are not the default values from HYPRE ADS */
    jac->as_print           = 0;
    jac->as_max_iter        = 1; /* used as a preconditioner */
    jac->as_tol             = 0.; /* used as a preconditioner */
    jac->ads_cycle_type     = 13;
    /* Smoothing options */
    jac->as_relax_type      = 2;
    jac->as_relax_times     = 1;
    jac->as_relax_weight    = 1.0;
    jac->as_omega           = 1.0;
    /* AMS solver parameters: cycle_type, coarsen type, agg_levels, relax_type, interp_type, Pmax */
    jac->ams_cycle_type       = 14;
    jac->as_amg_alpha_opts[0] = 10;
    jac->as_amg_alpha_opts[1] = 1;
    jac->as_amg_alpha_opts[2] = 6;
    jac->as_amg_alpha_opts[3] = 6;
    jac->as_amg_alpha_opts[4] = 4;
    jac->as_amg_alpha_theta   = 0.25;
    /* Vector Poisson AMG solver parameters: coarsen type, agg_levels, relax_type, interp_type, Pmax */
    jac->as_amg_beta_opts[0] = 10;
    jac->as_amg_beta_opts[1] = 1;
    jac->as_amg_beta_opts[2] = 6;
    jac->as_amg_beta_opts[3] = 6;
    jac->as_amg_beta_opts[4] = 4;
    jac->as_amg_beta_theta   = 0.25;
    PetscStackCallStandard(HYPRE_ADSSetPrintLevel,(jac->hsolver,jac->as_print));
    PetscStackCallStandard(HYPRE_ADSSetMaxIter,(jac->hsolver,jac->as_max_iter));
    PetscStackCallStandard(HYPRE_ADSSetCycleType,(jac->hsolver,jac->ams_cycle_type));
    PetscStackCallStandard(HYPRE_ADSSetTol,(jac->hsolver,jac->as_tol));
    PetscStackCallStandard(HYPRE_ADSSetSmoothingOptions,(jac->hsolver,jac->as_relax_type,
                                                                      jac->as_relax_times,
                                                                      jac->as_relax_weight,
                                                                      jac->as_omega));
    PetscStackCallStandard(HYPRE_ADSSetAMSOptions,(jac->hsolver,jac->ams_cycle_type,             /* AMG coarsen type */
                                                                jac->as_amg_alpha_opts[0],       /* AMG coarsen type */
                                                                jac->as_amg_alpha_opts[1],       /* AMG agg_levels */
                                                                jac->as_amg_alpha_opts[2],       /* AMG relax_type */
                                                                jac->as_amg_alpha_theta,
                                                                jac->as_amg_alpha_opts[3],       /* AMG interp_type */
                                                                jac->as_amg_alpha_opts[4]));     /* AMG Pmax */
    PetscStackCallStandard(HYPRE_ADSSetAMGOptions,(jac->hsolver,jac->as_amg_beta_opts[0],       /* AMG coarsen type */
                                                                jac->as_amg_beta_opts[1],       /* AMG agg_levels */
                                                                jac->as_amg_beta_opts[2],       /* AMG relax_type */
                                                                jac->as_amg_beta_theta,
                                                                jac->as_amg_beta_opts[3],       /* AMG interp_type */
                                                                jac->as_amg_beta_opts[4]));     /* AMG Pmax */
    ierr = PetscObjectComposeFunction((PetscObject)pc,"PCSetCoordinates_C",PCSetCoordinates_HYPRE);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)pc,"PCHYPRESetDiscreteGradient_C",PCHYPRESetDiscreteGradient_HYPRE);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)pc,"PCHYPRESetDiscreteCurl_C",PCHYPRESetDiscreteCurl_HYPRE);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = PetscFree(jac->hypre_type);CHKERRQ(ierr);

  jac->hypre_type = NULL;
  SETERRQ1(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown HYPRE preconditioner %s; Choices are pilut, parasails, boomeramg, ams",name);
  PetscFunctionReturn(0);
}

/*
    It only gets here if the HYPRE type has not been set before the call to
   ...SetFromOptions() which actually is most of the time
*/
#undef __FUNCT__
#define __FUNCT__ "PCSetFromOptions_HYPRE"
static PetscErrorCode PCSetFromOptions_HYPRE(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PetscErrorCode ierr;
  PetscInt       indx;
  const char     *type[] = {"pilut","parasails","boomeramg","ams","ads"};
  PetscBool      flg;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"HYPRE preconditioner options");CHKERRQ(ierr);
  ierr = PetscOptionsEList("-pc_hypre_type","HYPRE preconditioner type","PCHYPRESetType",type,4,"boomeramg",&indx,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PCHYPRESetType_HYPRE(pc,type[indx]);CHKERRQ(ierr);
  } else {
    ierr = PCHYPRESetType_HYPRE(pc,"boomeramg");CHKERRQ(ierr);
  }
  if (pc->ops->setfromoptions) {
    ierr = pc->ops->setfromoptions(PetscOptionsObject,pc);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCHYPRESetType"
/*@C
     PCHYPRESetType - Sets which hypre preconditioner you wish to use

   Input Parameters:
+     pc - the preconditioner context
-     name - either  pilut, parasails, boomeramg, ams, ads

   Options Database Keys:
   -pc_hypre_type - One of pilut, parasails, boomeramg, ams, ads

   Level: intermediate

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC,
           PCHYPRE

@*/
PetscErrorCode  PCHYPRESetType(PC pc,const char name[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidCharPointer(name,2);
  ierr = PetscTryMethod(pc,"PCHYPRESetType_C",(PC,const char[]),(pc,name));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCHYPREGetType"
/*@C
     PCHYPREGetType - Gets which hypre preconditioner you are using

   Input Parameter:
.     pc - the preconditioner context

   Output Parameter:
.     name - either  pilut, parasails, boomeramg, ams, ads

   Level: intermediate

.seealso:  PCCreate(), PCHYPRESetType(), PCType (for list of available types), PC,
           PCHYPRE

@*/
PetscErrorCode  PCHYPREGetType(PC pc,const char *name[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidPointer(name,2);
  ierr = PetscTryMethod(pc,"PCHYPREGetType_C",(PC,const char*[]),(pc,name));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
     PCHYPRE - Allows you to use the matrix element based preconditioners in the LLNL package hypre

   Options Database Keys:
+   -pc_hypre_type - One of pilut, parasails, boomeramg, ams, ads
-   Too many others to list, run with -pc_type hypre -pc_hypre_type XXX -help to see options for the XXX
          preconditioner

   Level: intermediate

   Notes: Apart from pc_hypre_type (for which there is PCHYPRESetType()),
          the many hypre options can ONLY be set via the options database (e.g. the command line
          or with PetscOptionsSetValue(), there are no functions to set them)

          The options -pc_hypre_boomeramg_max_iter and -pc_hypre_boomeramg_rtol refer to the number of iterations
          (V-cycles) and tolerance that boomeramg does EACH time it is called. So for example, if
          -pc_hypre_boomeramg_max_iter is set to 2 then 2-V-cycles are being used to define the preconditioner
          (-pc_hypre_boomeramg_rtol should be set to 0.0 - the default - to strictly use a fixed number of
          iterations per hypre call). -ksp_max_it and -ksp_rtol STILL determine the total number of iterations
          and tolerance for the Krylov solver. For example, if -pc_hypre_boomeramg_max_iter is 2 and -ksp_max_it is 10
          then AT MOST twenty V-cycles of boomeramg will be called.

           Note that the option -pc_hypre_boomeramg_relax_type_all defaults to symmetric relaxation
           (symmetric-SOR/Jacobi), which is required for Krylov solvers like CG that expect symmetry.
           Otherwise, you may want to use -pc_hypre_boomeramg_relax_type_all SOR/Jacobi.
          If you wish to use BoomerAMG WITHOUT a Krylov method use -ksp_type richardson NOT -ksp_type preonly
          and use -ksp_max_it to control the number of V-cycles.
          (see the PETSc FAQ.html at the PETSc website under the Documentation tab).

          2007-02-03 Using HYPRE-1.11.1b, the routine HYPRE_BoomerAMGSolveT and the option
          -pc_hypre_parasails_reuse were failing with SIGSEGV. Dalcin L.

          MatSetNearNullSpace() - if you provide a near null space to your matrix it is ignored by hypre UNLESS you also use
          the two options:
+   -pc_hypre_boomeramg_nodal_coarsen <n> - where n is from 1 to 6 (see HYPRE_BOOMERAMGSetNodal())
-   -pc_hypre_boomeramg_vec_interp_variant <v> where v is from 1 to 3 (see HYPRE_BoomerAMGSetInterpVecVariant())

          Depending on the linear system you may see the same or different convergence depending on the values you use.

          See PCPFMG for access to the hypre Struct PFMG solver

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC,
           PCHYPRESetType(), PCPFMG

M*/

#undef __FUNCT__
#define __FUNCT__ "PCCreate_HYPRE"
PETSC_EXTERN PetscErrorCode PCCreate_HYPRE(PC pc)
{
  PC_HYPRE       *jac;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(pc,&jac);CHKERRQ(ierr);

  pc->data                = jac;
  pc->ops->reset          = PCReset_HYPRE;
  pc->ops->destroy        = PCDestroy_HYPRE;
  pc->ops->setfromoptions = PCSetFromOptions_HYPRE;
  pc->ops->setup          = PCSetUp_HYPRE;
  pc->ops->apply          = PCApply_HYPRE;
  jac->comm_hypre         = MPI_COMM_NULL;
  jac->hypre_type         = NULL;
  jac->coords[0]          = NULL;
  jac->coords[1]          = NULL;
  jac->coords[2]          = NULL;
  jac->constants[0]       = NULL;
  jac->constants[1]       = NULL;
  jac->constants[2]       = NULL;
  jac->G                  = NULL;
  jac->C                  = NULL;
  jac->alpha_Poisson      = NULL;
  jac->beta_Poisson       = NULL;
  jac->setdim             = NULL;
  jac->hmnull             = NULL;
  jac->n_hmnull           = 0;
  /* duplicate communicator for hypre */
  ierr = MPI_Comm_dup(PetscObjectComm((PetscObject)pc),&(jac->comm_hypre));CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCHYPRESetType_C",PCHYPRESetType_HYPRE);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCHYPREGetType_C",PCHYPREGetType_HYPRE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------------------------------------------------------*/

/* this include is needed ONLY to allow access to the private data inside the Mat object specific to hypre */
#include <petsc/private/matimpl.h>

typedef struct {
  MPI_Comm           hcomm;        /* does not share comm with HYPRE_StructMatrix because need to create solver before getting matrix */
  HYPRE_StructSolver hsolver;

  /* keep copy of PFMG options used so may view them */
  PetscInt its;
  double   tol;
  PetscInt relax_type;
  PetscInt rap_type;
  PetscInt num_pre_relax,num_post_relax;
  PetscInt max_levels;
} PC_PFMG;

#undef __FUNCT__
#define __FUNCT__ "PCDestroy_PFMG"
PetscErrorCode PCDestroy_PFMG(PC pc)
{
  PetscErrorCode ierr;
  PC_PFMG        *ex = (PC_PFMG*) pc->data;

  PetscFunctionBegin;
  if (ex->hsolver) PetscStackCallStandard(HYPRE_StructPFMGDestroy,(ex->hsolver));
  ierr = MPI_Comm_free(&ex->hcomm);CHKERRQ(ierr);
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static const char *PFMGRelaxType[] = {"Jacobi","Weighted-Jacobi","symmetric-Red/Black-Gauss-Seidel","Red/Black-Gauss-Seidel"};
static const char *PFMGRAPType[] = {"Galerkin","non-Galerkin"};

#undef __FUNCT__
#define __FUNCT__ "PCView_PFMG"
PetscErrorCode PCView_PFMG(PC pc,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      iascii;
  PC_PFMG        *ex = (PC_PFMG*) pc->data;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE PFMG preconditioning\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE PFMG: max iterations %d\n",ex->its);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE PFMG: tolerance %g\n",ex->tol);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE PFMG: relax type %s\n",PFMGRelaxType[ex->relax_type]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE PFMG: RAP type %s\n",PFMGRAPType[ex->rap_type]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE PFMG: number pre-relax %d post-relax %d\n",ex->num_pre_relax,ex->num_post_relax);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE PFMG: max levels %d\n",ex->max_levels);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PCSetFromOptions_PFMG"
PetscErrorCode PCSetFromOptions_PFMG(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PetscErrorCode ierr;
  PC_PFMG        *ex = (PC_PFMG*) pc->data;
  PetscBool      flg = PETSC_FALSE;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"PFMG options");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_pfmg_print_statistics","Print statistics","HYPRE_StructPFMGSetPrintLevel",flg,&flg,NULL);CHKERRQ(ierr);
  if (flg) {
    int level=3;
    PetscStackCallStandard(HYPRE_StructPFMGSetPrintLevel,(ex->hsolver,level));
  }
  ierr = PetscOptionsInt("-pc_pfmg_its","Number of iterations of PFMG to use as preconditioner","HYPRE_StructPFMGSetMaxIter",ex->its,&ex->its,NULL);CHKERRQ(ierr);
  PetscStackCallStandard(HYPRE_StructPFMGSetMaxIter,(ex->hsolver,ex->its));
  ierr = PetscOptionsInt("-pc_pfmg_num_pre_relax","Number of smoothing steps before coarse grid","HYPRE_StructPFMGSetNumPreRelax",ex->num_pre_relax,&ex->num_pre_relax,NULL);CHKERRQ(ierr);
  PetscStackCallStandard(HYPRE_StructPFMGSetNumPreRelax,(ex->hsolver,ex->num_pre_relax));
  ierr = PetscOptionsInt("-pc_pfmg_num_post_relax","Number of smoothing steps after coarse grid","HYPRE_StructPFMGSetNumPostRelax",ex->num_post_relax,&ex->num_post_relax,NULL);CHKERRQ(ierr);
  PetscStackCallStandard(HYPRE_StructPFMGSetNumPostRelax,(ex->hsolver,ex->num_post_relax));

  ierr = PetscOptionsInt("-pc_pfmg_max_levels","Max Levels for MG hierarchy","HYPRE_StructPFMGSetMaxLevels",ex->max_levels,&ex->max_levels,NULL);CHKERRQ(ierr);
  PetscStackCallStandard(HYPRE_StructPFMGSetMaxLevels,(ex->hsolver,ex->max_levels));

  ierr = PetscOptionsReal("-pc_pfmg_tol","Tolerance of PFMG","HYPRE_StructPFMGSetTol",ex->tol,&ex->tol,NULL);CHKERRQ(ierr);
  PetscStackCallStandard(HYPRE_StructPFMGSetTol,(ex->hsolver,ex->tol));
  ierr = PetscOptionsEList("-pc_pfmg_relax_type","Relax type for the up and down cycles","HYPRE_StructPFMGSetRelaxType",PFMGRelaxType,ALEN(PFMGRelaxType),PFMGRelaxType[ex->relax_type],&ex->relax_type,NULL);CHKERRQ(ierr);
  PetscStackCallStandard(HYPRE_StructPFMGSetRelaxType,(ex->hsolver, ex->relax_type));
  ierr = PetscOptionsEList("-pc_pfmg_rap_type","RAP type","HYPRE_StructPFMGSetRAPType",PFMGRAPType,ALEN(PFMGRAPType),PFMGRAPType[ex->rap_type],&ex->rap_type,NULL);CHKERRQ(ierr);
  PetscStackCallStandard(HYPRE_StructPFMGSetRAPType,(ex->hsolver, ex->rap_type));
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCApply_PFMG"
PetscErrorCode PCApply_PFMG(PC pc,Vec x,Vec y)
{
  PetscErrorCode    ierr;
  PC_PFMG           *ex = (PC_PFMG*) pc->data;
  PetscScalar       *yy;
  const PetscScalar *xx;
  PetscInt          ilower[3],iupper[3];
  Mat_HYPREStruct   *mx = (Mat_HYPREStruct*)(pc->pmat->data);

  PetscFunctionBegin;
  ierr       = PetscCitationsRegister(hypreCitation,&cite);CHKERRQ(ierr);
  ierr       = DMDAGetCorners(mx->da,&ilower[0],&ilower[1],&ilower[2],&iupper[0],&iupper[1],&iupper[2]);CHKERRQ(ierr);
  iupper[0] += ilower[0] - 1;
  iupper[1] += ilower[1] - 1;
  iupper[2] += ilower[2] - 1;

  /* copy x values over to hypre */
  PetscStackCallStandard(HYPRE_StructVectorSetConstantValues,(mx->hb,0.0));
  ierr = VecGetArrayRead(x,&xx);CHKERRQ(ierr);
  PetscStackCallStandard(HYPRE_StructVectorSetBoxValues,(mx->hb,(HYPRE_Int *)ilower,(HYPRE_Int *)iupper,(PetscScalar*)xx));
  ierr = VecRestoreArrayRead(x,&xx);CHKERRQ(ierr);
  PetscStackCallStandard(HYPRE_StructVectorAssemble,(mx->hb));
  PetscStackCallStandard(HYPRE_StructPFMGSolve,(ex->hsolver,mx->hmat,mx->hb,mx->hx));

  /* copy solution values back to PETSc */
  ierr = VecGetArray(y,&yy);CHKERRQ(ierr);
  PetscStackCallStandard(HYPRE_StructVectorGetBoxValues,(mx->hx,(HYPRE_Int *)ilower,(HYPRE_Int *)iupper,yy));
  ierr = VecRestoreArray(y,&yy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCApplyRichardson_PFMG"
static PetscErrorCode PCApplyRichardson_PFMG(PC pc,Vec b,Vec y,Vec w,PetscReal rtol,PetscReal abstol, PetscReal dtol,PetscInt its,PetscBool guesszero,PetscInt *outits,PCRichardsonConvergedReason *reason)
{
  PC_PFMG        *jac = (PC_PFMG*)pc->data;
  PetscErrorCode ierr;
  PetscInt       oits;

  PetscFunctionBegin;
  ierr = PetscCitationsRegister(hypreCitation,&cite);CHKERRQ(ierr);
  PetscStackCallStandard(HYPRE_StructPFMGSetMaxIter,(jac->hsolver,its*jac->its));
  PetscStackCallStandard(HYPRE_StructPFMGSetTol,(jac->hsolver,rtol));

  ierr = PCApply_PFMG(pc,b,y);CHKERRQ(ierr);
  PetscStackCallStandard(HYPRE_StructPFMGGetNumIterations,(jac->hsolver,(HYPRE_Int *)&oits));
  *outits = oits;
  if (oits == its) *reason = PCRICHARDSON_CONVERGED_ITS;
  else             *reason = PCRICHARDSON_CONVERGED_RTOL;
  PetscStackCallStandard(HYPRE_StructPFMGSetTol,(jac->hsolver,jac->tol));
  PetscStackCallStandard(HYPRE_StructPFMGSetMaxIter,(jac->hsolver,jac->its));
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PCSetUp_PFMG"
PetscErrorCode PCSetUp_PFMG(PC pc)
{
  PetscErrorCode  ierr;
  PC_PFMG         *ex = (PC_PFMG*) pc->data;
  Mat_HYPREStruct *mx = (Mat_HYPREStruct*)(pc->pmat->data);
  PetscBool       flg;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)pc->pmat,MATHYPRESTRUCT,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_INCOMP,"Must use MATHYPRESTRUCT with this preconditioner");

  /* create the hypre solver object and set its information */
  if (ex->hsolver) PetscStackCallStandard(HYPRE_StructPFMGDestroy,(ex->hsolver));
  PetscStackCallStandard(HYPRE_StructPFMGCreate,(ex->hcomm,&ex->hsolver));
  PetscStackCallStandard(HYPRE_StructPFMGSetup,(ex->hsolver,mx->hmat,mx->hb,mx->hx));
  PetscStackCallStandard(HYPRE_StructPFMGSetZeroGuess,(ex->hsolver));
  PetscFunctionReturn(0);
}


/*MC
     PCPFMG - the hypre PFMG multigrid solver

   Level: advanced

   Options Database:
+ -pc_pfmg_its <its> number of iterations of PFMG to use as preconditioner
. -pc_pfmg_num_pre_relax <steps> number of smoothing steps before coarse grid
. -pc_pfmg_num_post_relax <steps> number of smoothing steps after coarse grid
. -pc_pfmg_tol <tol> tolerance of PFMG
. -pc_pfmg_relax_type -relaxation type for the up and down cycles, one of Jacobi,Weighted-Jacobi,symmetric-Red/Black-Gauss-Seidel,Red/Black-Gauss-Seidel
- -pc_pfmg_rap_type - type of coarse matrix generation, one of Galerkin,non-Galerkin

   Notes:  This is for CELL-centered descretizations

           This must be used with the MATHYPRESTRUCT matrix type.
           This is less general than in hypre, it supports only one block per process defined by a PETSc DMDA.

.seealso:  PCMG, MATHYPRESTRUCT
M*/

#undef __FUNCT__
#define __FUNCT__ "PCCreate_PFMG"
PETSC_EXTERN PetscErrorCode PCCreate_PFMG(PC pc)
{
  PetscErrorCode ierr;
  PC_PFMG        *ex;

  PetscFunctionBegin;
  ierr     = PetscNew(&ex);CHKERRQ(ierr); \
  pc->data = ex;

  ex->its            = 1;
  ex->tol            = 1.e-8;
  ex->relax_type     = 1;
  ex->rap_type       = 0;
  ex->num_pre_relax  = 1;
  ex->num_post_relax = 1;
  ex->max_levels     = 0;

  pc->ops->setfromoptions  = PCSetFromOptions_PFMG;
  pc->ops->view            = PCView_PFMG;
  pc->ops->destroy         = PCDestroy_PFMG;
  pc->ops->apply           = PCApply_PFMG;
  pc->ops->applyrichardson = PCApplyRichardson_PFMG;
  pc->ops->setup           = PCSetUp_PFMG;

  ierr = MPI_Comm_dup(PetscObjectComm((PetscObject)pc),&(ex->hcomm));CHKERRQ(ierr);
  PetscStackCallStandard(HYPRE_StructPFMGCreate,(ex->hcomm,&ex->hsolver));
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------------------------------------------------------------------------------------*/

/* we know we are working with a HYPRE_SStructMatrix */
typedef struct {
  MPI_Comm            hcomm;       /* does not share comm with HYPRE_SStructMatrix because need to create solver before getting matrix */
  HYPRE_SStructSolver ss_solver;

  /* keep copy of SYSPFMG options used so may view them */
  PetscInt its;
  double   tol;
  PetscInt relax_type;
  PetscInt num_pre_relax,num_post_relax;
} PC_SysPFMG;

#undef __FUNCT__
#define __FUNCT__ "PCDestroy_SysPFMG"
PetscErrorCode PCDestroy_SysPFMG(PC pc)
{
  PetscErrorCode ierr;
  PC_SysPFMG     *ex = (PC_SysPFMG*) pc->data;

  PetscFunctionBegin;
  if (ex->ss_solver) PetscStackCallStandard(HYPRE_SStructSysPFMGDestroy,(ex->ss_solver));
  ierr = MPI_Comm_free(&ex->hcomm);CHKERRQ(ierr);
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static const char *SysPFMGRelaxType[] = {"Weighted-Jacobi","Red/Black-Gauss-Seidel"};

#undef __FUNCT__
#define __FUNCT__ "PCView_SysPFMG"
PetscErrorCode PCView_SysPFMG(PC pc,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      iascii;
  PC_SysPFMG     *ex = (PC_SysPFMG*) pc->data;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE SysPFMG preconditioning\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE SysPFMG: max iterations %d\n",ex->its);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE SysPFMG: tolerance %g\n",ex->tol);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE SysPFMG: relax type %s\n",PFMGRelaxType[ex->relax_type]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE SysPFMG: number pre-relax %d post-relax %d\n",ex->num_pre_relax,ex->num_post_relax);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PCSetFromOptions_SysPFMG"
PetscErrorCode PCSetFromOptions_SysPFMG(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PetscErrorCode ierr;
  PC_SysPFMG     *ex = (PC_SysPFMG*) pc->data;
  PetscBool      flg = PETSC_FALSE;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"SysPFMG options");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_syspfmg_print_statistics","Print statistics","HYPRE_SStructSysPFMGSetPrintLevel",flg,&flg,NULL);CHKERRQ(ierr);
  if (flg) {
    int level=3;
    PetscStackCallStandard(HYPRE_SStructSysPFMGSetPrintLevel,(ex->ss_solver,level));
  }
  ierr = PetscOptionsInt("-pc_syspfmg_its","Number of iterations of SysPFMG to use as preconditioner","HYPRE_SStructSysPFMGSetMaxIter",ex->its,&ex->its,NULL);CHKERRQ(ierr);
  PetscStackCallStandard(HYPRE_SStructSysPFMGSetMaxIter,(ex->ss_solver,ex->its));
  ierr = PetscOptionsInt("-pc_syspfmg_num_pre_relax","Number of smoothing steps before coarse grid","HYPRE_SStructSysPFMGSetNumPreRelax",ex->num_pre_relax,&ex->num_pre_relax,NULL);CHKERRQ(ierr);
  PetscStackCallStandard(HYPRE_SStructSysPFMGSetNumPreRelax,(ex->ss_solver,ex->num_pre_relax));
  ierr = PetscOptionsInt("-pc_syspfmg_num_post_relax","Number of smoothing steps after coarse grid","HYPRE_SStructSysPFMGSetNumPostRelax",ex->num_post_relax,&ex->num_post_relax,NULL);CHKERRQ(ierr);
  PetscStackCallStandard(HYPRE_SStructSysPFMGSetNumPostRelax,(ex->ss_solver,ex->num_post_relax));

  ierr = PetscOptionsReal("-pc_syspfmg_tol","Tolerance of SysPFMG","HYPRE_SStructSysPFMGSetTol",ex->tol,&ex->tol,NULL);CHKERRQ(ierr);
  PetscStackCallStandard(HYPRE_SStructSysPFMGSetTol,(ex->ss_solver,ex->tol));
  ierr = PetscOptionsEList("-pc_syspfmg_relax_type","Relax type for the up and down cycles","HYPRE_SStructSysPFMGSetRelaxType",SysPFMGRelaxType,4,SysPFMGRelaxType[ex->relax_type],&ex->relax_type,NULL);CHKERRQ(ierr);
  PetscStackCallStandard(HYPRE_SStructSysPFMGSetRelaxType,(ex->ss_solver, ex->relax_type));
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCApply_SysPFMG"
PetscErrorCode PCApply_SysPFMG(PC pc,Vec x,Vec y)
{
  PetscErrorCode    ierr;
  PC_SysPFMG        *ex = (PC_SysPFMG*) pc->data;
  PetscScalar       *yy;
  const PetscScalar *xx;
  PetscInt          ilower[3],iupper[3];
  Mat_HYPRESStruct  *mx     = (Mat_HYPRESStruct*)(pc->pmat->data);
  PetscInt          ordering= mx->dofs_order;
  PetscInt          nvars   = mx->nvars;
  PetscInt          part    = 0;
  PetscInt          size;
  PetscInt          i;

  PetscFunctionBegin;
  ierr       = PetscCitationsRegister(hypreCitation,&cite);CHKERRQ(ierr);
  ierr       = DMDAGetCorners(mx->da,&ilower[0],&ilower[1],&ilower[2],&iupper[0],&iupper[1],&iupper[2]);CHKERRQ(ierr);
  iupper[0] += ilower[0] - 1;
  iupper[1] += ilower[1] - 1;
  iupper[2] += ilower[2] - 1;

  size = 1;
  for (i= 0; i< 3; i++) size *= (iupper[i]-ilower[i]+1);

  /* copy x values over to hypre for variable ordering */
  if (ordering) {
    PetscStackCallStandard(HYPRE_SStructVectorSetConstantValues,(mx->ss_b,0.0));
    ierr = VecGetArrayRead(x,&xx);CHKERRQ(ierr);
    for (i= 0; i< nvars; i++) PetscStackCallStandard(HYPRE_SStructVectorSetBoxValues,(mx->ss_b,part,(HYPRE_Int *)ilower,(HYPRE_Int *)iupper,i,(PetscScalar*)xx+(size*i)));
    ierr = VecRestoreArrayRead(x,&xx);CHKERRQ(ierr);
    PetscStackCallStandard(HYPRE_SStructVectorAssemble,(mx->ss_b));
    PetscStackCallStandard(HYPRE_SStructMatrixMatvec,(1.0,mx->ss_mat,mx->ss_b,0.0,mx->ss_x));
    PetscStackCallStandard(HYPRE_SStructSysPFMGSolve,(ex->ss_solver,mx->ss_mat,mx->ss_b,mx->ss_x));

    /* copy solution values back to PETSc */
    ierr = VecGetArray(y,&yy);CHKERRQ(ierr);
    for (i= 0; i< nvars; i++) PetscStackCallStandard(HYPRE_SStructVectorGetBoxValues,(mx->ss_x,part,(HYPRE_Int *)ilower,(HYPRE_Int *)iupper,i,yy+(size*i)));
    ierr = VecRestoreArray(y,&yy);CHKERRQ(ierr);
  } else {      /* nodal ordering must be mapped to variable ordering for sys_pfmg */
    PetscScalar *z;
    PetscInt    j, k;

    ierr = PetscMalloc1(nvars*size,&z);CHKERRQ(ierr);
    PetscStackCallStandard(HYPRE_SStructVectorSetConstantValues,(mx->ss_b,0.0));
    ierr = VecGetArrayRead(x,&xx);CHKERRQ(ierr);

    /* transform nodal to hypre's variable ordering for sys_pfmg */
    for (i= 0; i< size; i++) {
      k= i*nvars;
      for (j= 0; j< nvars; j++) z[j*size+i]= xx[k+j];
    }
    for (i= 0; i< nvars; i++) PetscStackCallStandard(HYPRE_SStructVectorSetBoxValues,(mx->ss_b,part,(HYPRE_Int *)ilower,(HYPRE_Int *)iupper,i,z+(size*i)));
    ierr = VecRestoreArrayRead(x,&xx);CHKERRQ(ierr);
    PetscStackCallStandard(HYPRE_SStructVectorAssemble,(mx->ss_b));
    PetscStackCallStandard(HYPRE_SStructSysPFMGSolve,(ex->ss_solver,mx->ss_mat,mx->ss_b,mx->ss_x));

    /* copy solution values back to PETSc */
    ierr = VecGetArray(y,&yy);CHKERRQ(ierr);
    for (i= 0; i< nvars; i++) PetscStackCallStandard(HYPRE_SStructVectorGetBoxValues,(mx->ss_x,part,(HYPRE_Int *)ilower,(HYPRE_Int *)iupper,i,z+(size*i)));
    /* transform hypre's variable ordering for sys_pfmg to nodal ordering */
    for (i= 0; i< size; i++) {
      k= i*nvars;
      for (j= 0; j< nvars; j++) yy[k+j]= z[j*size+i];
    }
    ierr = VecRestoreArray(y,&yy);CHKERRQ(ierr);
    ierr = PetscFree(z);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCApplyRichardson_SysPFMG"
static PetscErrorCode PCApplyRichardson_SysPFMG(PC pc,Vec b,Vec y,Vec w,PetscReal rtol,PetscReal abstol, PetscReal dtol,PetscInt its,PetscBool guesszero,PetscInt *outits,PCRichardsonConvergedReason *reason)
{
  PC_SysPFMG     *jac = (PC_SysPFMG*)pc->data;
  PetscErrorCode ierr;
  PetscInt       oits;

  PetscFunctionBegin;
  ierr = PetscCitationsRegister(hypreCitation,&cite);CHKERRQ(ierr);
  PetscStackCallStandard(HYPRE_SStructSysPFMGSetMaxIter,(jac->ss_solver,its*jac->its));
  PetscStackCallStandard(HYPRE_SStructSysPFMGSetTol,(jac->ss_solver,rtol));
  ierr = PCApply_SysPFMG(pc,b,y);CHKERRQ(ierr);
  PetscStackCallStandard(HYPRE_SStructSysPFMGGetNumIterations,(jac->ss_solver,(HYPRE_Int *)&oits));
  *outits = oits;
  if (oits == its) *reason = PCRICHARDSON_CONVERGED_ITS;
  else             *reason = PCRICHARDSON_CONVERGED_RTOL;
  PetscStackCallStandard(HYPRE_SStructSysPFMGSetTol,(jac->ss_solver,jac->tol));
  PetscStackCallStandard(HYPRE_SStructSysPFMGSetMaxIter,(jac->ss_solver,jac->its));
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PCSetUp_SysPFMG"
PetscErrorCode PCSetUp_SysPFMG(PC pc)
{
  PetscErrorCode   ierr;
  PC_SysPFMG       *ex = (PC_SysPFMG*) pc->data;
  Mat_HYPRESStruct *mx = (Mat_HYPRESStruct*)(pc->pmat->data);
  PetscBool        flg;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)pc->pmat,MATHYPRESSTRUCT,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_INCOMP,"Must use MATHYPRESSTRUCT with this preconditioner");

  /* create the hypre sstruct solver object and set its information */
  if (ex->ss_solver) PetscStackCallStandard(HYPRE_SStructSysPFMGDestroy,(ex->ss_solver));
  PetscStackCallStandard(HYPRE_SStructSysPFMGCreate,(ex->hcomm,&ex->ss_solver));
  PetscStackCallStandard(HYPRE_SStructSysPFMGSetZeroGuess,(ex->ss_solver));
  PetscStackCallStandard(HYPRE_SStructSysPFMGSetup,(ex->ss_solver,mx->ss_mat,mx->ss_b,mx->ss_x));
  PetscFunctionReturn(0);
}


/*MC
     PCSysPFMG - the hypre SysPFMG multigrid solver

   Level: advanced

   Options Database:
+ -pc_syspfmg_its <its> number of iterations of SysPFMG to use as preconditioner
. -pc_syspfmg_num_pre_relax <steps> number of smoothing steps before coarse grid
. -pc_syspfmg_num_post_relax <steps> number of smoothing steps after coarse grid
. -pc_syspfmg_tol <tol> tolerance of SysPFMG
. -pc_syspfmg_relax_type -relaxation type for the up and down cycles, one of Weighted-Jacobi,Red/Black-Gauss-Seidel

   Notes:  This is for CELL-centered descretizations

           This must be used with the MATHYPRESSTRUCT matrix type.
           This is less general than in hypre, it supports only one part, and one block per process defined by a PETSc DMDA.
           Also, only cell-centered variables.

.seealso:  PCMG, MATHYPRESSTRUCT
M*/

#undef __FUNCT__
#define __FUNCT__ "PCCreate_SysPFMG"
PETSC_EXTERN PetscErrorCode PCCreate_SysPFMG(PC pc)
{
  PetscErrorCode ierr;
  PC_SysPFMG     *ex;

  PetscFunctionBegin;
  ierr     = PetscNew(&ex);CHKERRQ(ierr); \
  pc->data = ex;

  ex->its            = 1;
  ex->tol            = 1.e-8;
  ex->relax_type     = 1;
  ex->num_pre_relax  = 1;
  ex->num_post_relax = 1;

  pc->ops->setfromoptions  = PCSetFromOptions_SysPFMG;
  pc->ops->view            = PCView_SysPFMG;
  pc->ops->destroy         = PCDestroy_SysPFMG;
  pc->ops->apply           = PCApply_SysPFMG;
  pc->ops->applyrichardson = PCApplyRichardson_SysPFMG;
  pc->ops->setup           = PCSetUp_SysPFMG;

  ierr = MPI_Comm_dup(PetscObjectComm((PetscObject)pc),&(ex->hcomm));CHKERRQ(ierr);
  PetscStackCallStandard(HYPRE_SStructSysPFMGCreate,(ex->hcomm,&ex->ss_solver));
  PetscFunctionReturn(0);
}
