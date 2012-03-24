
/* 
   Provides an interface to the LLNL package hypre
*/

/* Must use hypre 2.0.0 or more recent. */

#include <petsc-private/pcimpl.h>          /*I "petscpc.h" I*/
#include <../src/dm/impls/da/hypre/mhyp.h>

/* 
   Private context (data structure) for the  preconditioner.  
*/
typedef struct {
  HYPRE_Solver       hsolver;
  HYPRE_IJMatrix     ij;
  HYPRE_IJVector     b,x;

  HYPRE_Int          (*destroy)(HYPRE_Solver);
  HYPRE_Int          (*solve)(HYPRE_Solver,HYPRE_ParCSRMatrix,HYPRE_ParVector,HYPRE_ParVector);
  HYPRE_Int          (*setup)(HYPRE_Solver,HYPRE_ParCSRMatrix,HYPRE_ParVector,HYPRE_ParVector);
  
  MPI_Comm           comm_hypre;
  char              *hypre_type;

  /* options for Pilut and BoomerAMG*/
  PetscInt           maxiter;
  double             tol;

  /* options for Pilut */
  PetscInt           factorrowsize;

  /* options for ParaSails */
  PetscInt           nlevels;
  double             threshhold;
  double             filter;
  PetscInt           sym;
  double             loadbal;
  PetscInt           logging;
  PetscInt           ruse;
  PetscInt           symt;

  /* options for Euclid */
  PetscBool          bjilu;
  PetscInt           levels;

  /* options for Euclid and BoomerAMG */
  PetscBool          printstatistics;

  /* options for BoomerAMG */
  PetscInt           cycletype;
  PetscInt           maxlevels;
  double             strongthreshold;
  double             maxrowsum;
  PetscInt           gridsweeps[3];
  PetscInt           coarsentype;
  PetscInt           measuretype;
  PetscInt           relaxtype[3];
  double             relaxweight;
  double             outerrelaxweight;
  PetscInt           relaxorder;
  double             truncfactor;
  PetscBool          applyrichardson;
  PetscInt           pmax;
  PetscInt           interptype;
  PetscInt           agg_nl;
  PetscInt           agg_num_paths;
  PetscInt           nodal_coarsen;
  PetscBool          nodal_relax;
  PetscInt           nodal_relax_levels;
} PC_HYPRE;


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
    PetscStackCallHypre(0,HYPRE_IJMatrixDestroy,(jac->ij));
    jac->ij = 0;
  }

  if (!jac->ij) { /* create the matrix the first time through */ 
    ierr = MatHYPRE_IJMatrixCreate(pc->pmat,&jac->ij);CHKERRQ(ierr);
  }
  if (!jac->b) { /* create the vectors the first time through */ 
    Vec x,b;
    ierr = MatGetVecs(pc->pmat,&x,&b);CHKERRQ(ierr);
    ierr = VecHYPRE_IJVectorCreate(x,&jac->x);CHKERRQ(ierr);
    ierr = VecHYPRE_IJVectorCreate(b,&jac->b);CHKERRQ(ierr);
    ierr = VecDestroy(&x);CHKERRQ(ierr);
    ierr = VecDestroy(&b);CHKERRQ(ierr);
  }

  /* special case for BoomerAMG */
  if (jac->setup == HYPRE_BoomerAMGSetup) {
    ierr = MatGetBlockSize(pc->pmat,&bs);CHKERRQ(ierr);
    if (bs > 1) {
      PetscStackCallHypre(0,HYPRE_BoomerAMGSetNumFunctions,(jac->hsolver,bs));
    }
  };
  ierr = MatHYPRE_IJMatrixCopy(pc->pmat,jac->ij);CHKERRQ(ierr);
  PetscStackCallHypre(0,HYPRE_IJMatrixGetObject,(jac->ij,(void**)&hmat));
  PetscStackCallHypre(0,HYPRE_IJVectorGetObject,(jac->b,(void**)&bv));
  PetscStackCallHypre(0,HYPRE_IJVectorGetObject,(jac->x,(void**)&xv));
  PetscStackCallHypre("HYPRE_SetupXXX",(*jac->setup),(jac->hsolver,hmat,bv,xv));
  PetscFunctionReturn(0);
}

/*
    Replaces the address where the HYPRE vector points to its data with the address of
  PETSc's data. Saves the old address so it can be reset when we are finished with it.
  Allows use to get the data into a HYPRE vector without the cost of memcopies 
*/
#define HYPREReplacePointer(b,newvalue,savedvalue) {\
   hypre_ParVector *par_vector   = (hypre_ParVector *)hypre_IJVectorObject(((hypre_IJVector*)b));\
   hypre_Vector    *local_vector = hypre_ParVectorLocalVector(par_vector);\
   savedvalue         = local_vector->data;\
   local_vector->data = newvalue;}

#undef __FUNCT__  
#define __FUNCT__ "PCApply_HYPRE"
static PetscErrorCode PCApply_HYPRE(PC pc,Vec b,Vec x)
{
  PC_HYPRE           *jac = (PC_HYPRE*)pc->data;
  PetscErrorCode     ierr;
  HYPRE_ParCSRMatrix hmat;
  PetscScalar        *bv,*xv;
  HYPRE_ParVector    jbv,jxv;
  PetscScalar        *sbv,*sxv; 
  PetscInt           hierr;

  PetscFunctionBegin;
  if (!jac->applyrichardson) {ierr = VecSet(x,0.0);CHKERRQ(ierr);}
  ierr = VecGetArray(b,&bv);CHKERRQ(ierr);
  ierr = VecGetArray(x,&xv);CHKERRQ(ierr);
  HYPREReplacePointer(jac->b,bv,sbv);
  HYPREReplacePointer(jac->x,xv,sxv);

  PetscStackCallHypre(0,HYPRE_IJMatrixGetObject,(jac->ij,(void**)&hmat));
  PetscStackCallHypre(0,HYPRE_IJVectorGetObject,(jac->b,(void**)&jbv));
  PetscStackCallHypre(0,HYPRE_IJVectorGetObject,(jac->x,(void**)&jxv));
  hierr = (*jac->solve)(jac->hsolver,hmat,jbv,jxv);
  if (hierr && hierr != HYPRE_ERROR_CONV) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in HYPRE solver, error code %d",hierr);
  if (hierr) hypre__global_error = 0;
 
  HYPREReplacePointer(jac->b,sbv,bv);
  HYPREReplacePointer(jac->x,sxv,xv);
  ierr = VecRestoreArray(x,&xv);CHKERRQ(ierr);
  ierr = VecRestoreArray(b,&bv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCDestroy_HYPRE"
static PetscErrorCode PCDestroy_HYPRE(PC pc)
{
  PC_HYPRE       *jac = (PC_HYPRE*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (jac->ij) PetscStackCallHypre(0,HYPRE_IJMatrixDestroy,(jac->ij));
  if (jac->b)  PetscStackCallHypre(0,HYPRE_IJVectorDestroy,(jac->b));
  if (jac->x)  PetscStackCallHypre(0,HYPRE_IJVectorDestroy,(jac->x));
  if (jac->destroy) PetscStackCallHypre("HYPRE_DistroyXXX",(*jac->destroy),(jac->hsolver));
  ierr = PetscFree(jac->hypre_type);CHKERRQ(ierr);
  if (jac->comm_hypre != MPI_COMM_NULL) { ierr = MPI_Comm_free(&(jac->comm_hypre));CHKERRQ(ierr);}
  ierr = PetscFree(pc->data);CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)pc,0);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCHYPRESetType_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCHYPREGetType_C","",PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "PCSetFromOptions_HYPRE_Pilut"
static PetscErrorCode PCSetFromOptions_HYPRE_Pilut(PC pc)
{
  PC_HYPRE       *jac = (PC_HYPRE*)pc->data;
  PetscErrorCode ierr;
  PetscBool      flag;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("HYPRE Pilut Options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_hypre_pilut_maxiter","Number of iterations","None",jac->maxiter,&jac->maxiter,&flag);CHKERRQ(ierr);
  if (flag) PetscStackCallHypre(0,HYPRE_ParCSRPilutSetMaxIter,(jac->hsolver,jac->maxiter));
  ierr = PetscOptionsReal("-pc_hypre_pilut_tol","Drop tolerance","None",jac->tol,&jac->tol,&flag);CHKERRQ(ierr);
  if (flag) PetscStackCallHypre(0,HYPRE_ParCSRPilutSetDropTolerance,(jac->hsolver,jac->tol));
  ierr = PetscOptionsInt("-pc_hypre_pilut_factorrowsize","FactorRowSize","None",jac->factorrowsize,&jac->factorrowsize,&flag);CHKERRQ(ierr);
  if (flag) PetscStackCallHypre(0,HYPRE_ParCSRPilutSetFactorRowSize,(jac->hsolver,jac->factorrowsize));
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
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE Pilut preconditioning\n");CHKERRQ(ierr);
    if (jac->maxiter != PETSC_DEFAULT) {
      ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE Pilut: maximum number of iterations %d\n",jac->maxiter);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE Pilut: default maximum number of iterations \n");CHKERRQ(ierr);
    }
    if (jac->tol != PETSC_DEFAULT) {
      ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE Pilut: drop tolerance %G\n",jac->tol);CHKERRQ(ierr);
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
#define __FUNCT__ "PCSetFromOptions_HYPRE_Euclid"
static PetscErrorCode PCSetFromOptions_HYPRE_Euclid(PC pc)
{
  PC_HYPRE       *jac = (PC_HYPRE*)pc->data;
  PetscErrorCode ierr;
  PetscBool      flag;
  char           *args[8],levels[16];
  PetscInt       cnt = 0;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("HYPRE Euclid Options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_hypre_euclid_levels","Number of levels of fill ILU(k)","None",jac->levels,&jac->levels,&flag);CHKERRQ(ierr);
  if (flag) {
    if (jac->levels < 0) SETERRQ1(((PetscObject)pc)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Number of levels %d must be nonegative",jac->levels);
    ierr = PetscSNPrintf(levels,sizeof levels,"%D",jac->levels);CHKERRQ(ierr);
    args[cnt++] = (char*)"-level"; args[cnt++] = levels;
  } 
  ierr = PetscOptionsBool("-pc_hypre_euclid_bj","Use block Jacobi ILU(k)","None",jac->bjilu,&jac->bjilu,PETSC_NULL);CHKERRQ(ierr);
  if (jac->bjilu) {
    args[cnt++] =(char*) "-bj"; args[cnt++] = (char*)"1";
  } 
    
  ierr = PetscOptionsBool("-pc_hypre_euclid_print_statistics","Print statistics","None",jac->printstatistics,&jac->printstatistics,PETSC_NULL);CHKERRQ(ierr);
  if (jac->printstatistics) {
    args[cnt++] = (char*)"-eu_stats"; args[cnt++] = (char*)"1";
    args[cnt++] = (char*)"-eu_mem"; args[cnt++] = (char*)"1";
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  if (cnt) PetscStackCallHypre(0,HYPRE_EuclidSetParams,(jac->hsolver,cnt,args));
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCView_HYPRE_Euclid"
static PetscErrorCode PCView_HYPRE_Euclid(PC pc,PetscViewer viewer)
{
  PC_HYPRE       *jac = (PC_HYPRE*)pc->data;
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE Euclid preconditioning\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE Euclid: number of levels %d\n",jac->levels);CHKERRQ(ierr);
    if (jac->bjilu) {
      ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE Euclid: Using block Jacobi ILU instead of parallel ILU\n");CHKERRQ(ierr);
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
  PetscScalar        *bv,*xv;
  HYPRE_ParVector    jbv,jxv;
  PetscScalar        *sbv,*sxv; 
  PetscInt           hierr;

  PetscFunctionBegin;
  ierr = VecSet(x,0.0);CHKERRQ(ierr);
  ierr = VecGetArray(b,&bv);CHKERRQ(ierr);
  ierr = VecGetArray(x,&xv);CHKERRQ(ierr);
  HYPREReplacePointer(jac->b,bv,sbv);
  HYPREReplacePointer(jac->x,xv,sxv);

  PetscStackCallHypre(0,HYPRE_IJMatrixGetObject,(jac->ij,(void**)&hmat));
  PetscStackCallHypre(0,HYPRE_IJVectorGetObject,(jac->b,(void**)&jbv));
  PetscStackCallHypre(0,HYPRE_IJVectorGetObject,(jac->x,(void**)&jxv));
  
  hierr = HYPRE_BoomerAMGSolveT(jac->hsolver,hmat,jbv,jxv);
  /* error code of 1 in BoomerAMG merely means convergence not achieved */
  if (hierr && (hierr != 1)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in HYPRE solver, error code %d",hierr);
  if (hierr) hypre__global_error = 0;
  
  HYPREReplacePointer(jac->b,sbv,bv);
  HYPREReplacePointer(jac->x,sxv,xv);
  ierr = VecRestoreArray(x,&xv);CHKERRQ(ierr);
  ierr = VecRestoreArray(b,&bv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static const char *HYPREBoomerAMGCycleType[]   = {"","V","W"};
static const char *HYPREBoomerAMGCoarsenType[] = {"CLJP","Ruge-Stueben","","modifiedRuge-Stueben","","","Falgout", "", "PMIS", "", "HMIS"};                                 
static const char *HYPREBoomerAMGMeasureType[] = {"local","global"};                                                                                                        
static const char *HYPREBoomerAMGRelaxType[]   = {"Jacobi","sequential-Gauss-Seidel","","SOR/Jacobi","backward-SOR/Jacobi","","symmetric-SOR/Jacobi",
                                                  "","","Gaussian-elimination"};                                                                                            
static const char *HYPREBoomerAMGInterpType[]  = {"classical", "", "", "direct", "multipass", "multipass-wts", "ext+i",
                                                  "ext+i-cc", "standard", "standard-wts", "", "", "FF", "FF1"};                                                             
#undef __FUNCT__  
#define __FUNCT__ "PCSetFromOptions_HYPRE_BoomerAMG"
static PetscErrorCode PCSetFromOptions_HYPRE_BoomerAMG(PC pc)
{
  PC_HYPRE       *jac = (PC_HYPRE*)pc->data;
  PetscErrorCode ierr;
  PetscInt       n,indx,level;
  PetscBool      flg, tmp_truth;
  double         tmpdbl, twodbl[2];

  PetscFunctionBegin;
  ierr = PetscOptionsHead("HYPRE BoomerAMG Options");CHKERRQ(ierr);
  ierr = PetscOptionsEList("-pc_hypre_boomeramg_cycle_type","Cycle type","None",HYPREBoomerAMGCycleType+1,2,HYPREBoomerAMGCycleType[jac->cycletype],&indx,&flg);CHKERRQ(ierr);
  if (flg) {
    jac->cycletype = indx+1;
    PetscStackCallHypre(0,HYPRE_BoomerAMGSetCycleType,(jac->hsolver,jac->cycletype)); 
  }
  ierr = PetscOptionsInt("-pc_hypre_boomeramg_max_levels","Number of levels (of grids) allowed","None",jac->maxlevels,&jac->maxlevels,&flg);CHKERRQ(ierr);
  if (flg) {
    if (jac->maxlevels < 2) SETERRQ1(((PetscObject)pc)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Number of levels %d must be at least two",jac->maxlevels);
    PetscStackCallHypre(0,HYPRE_BoomerAMGSetMaxLevels,(jac->hsolver,jac->maxlevels));
  }
  ierr = PetscOptionsInt("-pc_hypre_boomeramg_max_iter","Maximum iterations used PER hypre call","None",jac->maxiter,&jac->maxiter,&flg);CHKERRQ(ierr);
  if (flg) {
    if (jac->maxiter < 1) SETERRQ1(((PetscObject)pc)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Number of iterations %d must be at least one",jac->maxiter);
    PetscStackCallHypre(0,HYPRE_BoomerAMGSetMaxIter,(jac->hsolver,jac->maxiter));
  }
  ierr = PetscOptionsScalar("-pc_hypre_boomeramg_tol","Convergence tolerance PER hypre call (0.0 = use a fixed number of iterations)","None",jac->tol,&jac->tol,&flg);CHKERRQ(ierr);
  if (flg) {
    if (jac->tol < 0.0) SETERRQ1(((PetscObject)pc)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Tolerance %G must be greater than or equal to zero",jac->tol);
    PetscStackCallHypre(0,HYPRE_BoomerAMGSetTol,(jac->hsolver,jac->tol));
  }

  ierr = PetscOptionsScalar("-pc_hypre_boomeramg_truncfactor","Truncation factor for interpolation (0=no truncation)","None",jac->truncfactor,&jac->truncfactor,&flg);CHKERRQ(ierr);
  if (flg) {
    if (jac->truncfactor < 0.0) SETERRQ1(((PetscObject)pc)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Truncation factor %G must be great than or equal zero",jac->truncfactor);
    PetscStackCallHypre(0,HYPRE_BoomerAMGSetTruncFactor,(jac->hsolver,jac->truncfactor));
  }

 ierr = PetscOptionsInt("-pc_hypre_boomeramg_P_max","Max elements per row for interpolation operator ( 0=unlimited )","None",jac->pmax,&jac->pmax,&flg);CHKERRQ(ierr);
  if (flg) {
    if (jac->pmax < 0) SETERRQ1(((PetscObject)pc)->comm,PETSC_ERR_ARG_OUTOFRANGE,"P_max %G must be greater than or equal to zero",jac->pmax);
    PetscStackCallHypre(0,HYPRE_BoomerAMGSetPMaxElmts,(jac->hsolver,jac->pmax));
  }

 ierr = PetscOptionsInt("-pc_hypre_boomeramg_agg_nl","Number of levels of aggressive coarsening","None",jac->agg_nl,&jac->agg_nl,&flg);CHKERRQ(ierr);
  if (flg) {
     if (jac->agg_nl < 0) SETERRQ1(((PetscObject)pc)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Number of levels %G must be greater than or equal to zero",jac->agg_nl);

     PetscStackCallHypre(0,HYPRE_BoomerAMGSetAggNumLevels,(jac->hsolver,jac->agg_nl));
  }


 ierr = PetscOptionsInt("-pc_hypre_boomeramg_agg_num_paths","Number of paths for aggressive coarsening","None",jac->agg_num_paths,&jac->agg_num_paths,&flg);CHKERRQ(ierr);
  if (flg) {
     if (jac->agg_num_paths < 1) SETERRQ1(((PetscObject)pc)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Number of paths %G must be greater than or equal to 1",jac->agg_num_paths);

     PetscStackCallHypre(0,HYPRE_BoomerAMGSetNumPaths,(jac->hsolver,jac->agg_num_paths));
  }


  ierr = PetscOptionsScalar("-pc_hypre_boomeramg_strong_threshold","Threshold for being strongly connected","None",jac->strongthreshold,&jac->strongthreshold,&flg);CHKERRQ(ierr);
  if (flg) {
    if (jac->strongthreshold < 0.0) SETERRQ1(((PetscObject)pc)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Strong threshold %G must be great than or equal zero",jac->strongthreshold);
    PetscStackCallHypre(0,HYPRE_BoomerAMGSetStrongThreshold,(jac->hsolver,jac->strongthreshold));
  }
  ierr = PetscOptionsScalar("-pc_hypre_boomeramg_max_row_sum","Maximum row sum","None",jac->maxrowsum,&jac->maxrowsum,&flg);CHKERRQ(ierr);
  if (flg) {
    if (jac->maxrowsum < 0.0) SETERRQ1(((PetscObject)pc)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Maximum row sum %G must be greater than zero",jac->maxrowsum);
    if (jac->maxrowsum > 1.0) SETERRQ1(((PetscObject)pc)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Maximum row sum %G must be less than or equal one",jac->maxrowsum);
    PetscStackCallHypre(0,HYPRE_BoomerAMGSetMaxRowSum,(jac->hsolver,jac->maxrowsum));
  } 

  /* Grid sweeps */
  ierr = PetscOptionsInt("-pc_hypre_boomeramg_grid_sweeps_all","Number of sweeps for the up and down grid levels","None",jac->gridsweeps[0],&indx,&flg);CHKERRQ(ierr);
  if (flg) {
    PetscStackCallHypre(0,HYPRE_BoomerAMGSetNumSweeps,(jac->hsolver,indx));
    /* modify the jac structure so we can view the updated options with PC_View */ 
    jac->gridsweeps[0] = indx;
    jac->gridsweeps[1] = indx;
    /*defaults coarse to 1 */
    jac->gridsweeps[2] = 1;
  } 

  ierr = PetscOptionsInt("-pc_hypre_boomeramg_grid_sweeps_down","Number of sweeps for the down cycles","None",jac->gridsweeps[0], &indx ,&flg);CHKERRQ(ierr);
  if (flg) {
    PetscStackCallHypre(0,HYPRE_BoomerAMGSetCycleNumSweeps,(jac->hsolver,indx, 1));
    jac->gridsweeps[0] = indx;
  }
  ierr = PetscOptionsInt("-pc_hypre_boomeramg_grid_sweeps_up","Number of sweeps for the up cycles","None",jac->gridsweeps[1],&indx,&flg);CHKERRQ(ierr);
  if (flg) {
    PetscStackCallHypre(0,HYPRE_BoomerAMGSetCycleNumSweeps,(jac->hsolver,indx, 2));
    jac->gridsweeps[1] = indx;
  }
  ierr = PetscOptionsInt("-pc_hypre_boomeramg_grid_sweeps_coarse","Number of sweeps for the coarse level","None",jac->gridsweeps[2],&indx,&flg);CHKERRQ(ierr);
  if (flg) {
    PetscStackCallHypre(0,HYPRE_BoomerAMGSetCycleNumSweeps,(jac->hsolver,indx, 3));
    jac->gridsweeps[2] = indx;
  }

  /* Relax type */
  ierr = PetscOptionsEList("-pc_hypre_boomeramg_relax_type_all","Relax type for the up and down cycles","None",HYPREBoomerAMGRelaxType,10,HYPREBoomerAMGRelaxType[6],&indx,&flg);CHKERRQ(ierr);
  if (flg) {
    jac->relaxtype[0] = jac->relaxtype[1]  = indx;
    PetscStackCallHypre(0,HYPRE_BoomerAMGSetRelaxType,(jac->hsolver, indx)); 
    /* by default, coarse type set to 9 */
    jac->relaxtype[2] = 9;
    
  }
  ierr = PetscOptionsEList("-pc_hypre_boomeramg_relax_type_down","Relax type for the down cycles","None",HYPREBoomerAMGRelaxType,10,HYPREBoomerAMGRelaxType[6],&indx,&flg);CHKERRQ(ierr);
  if (flg) {
    jac->relaxtype[0] = indx;
    PetscStackCallHypre(0,HYPRE_BoomerAMGSetCycleRelaxType,(jac->hsolver, indx, 1)); 
  }
  ierr = PetscOptionsEList("-pc_hypre_boomeramg_relax_type_up","Relax type for the up cycles","None",HYPREBoomerAMGRelaxType,10,HYPREBoomerAMGRelaxType[6],&indx,&flg);CHKERRQ(ierr);
  if (flg) {
    jac->relaxtype[1] = indx;
    PetscStackCallHypre(0,HYPRE_BoomerAMGSetCycleRelaxType,(jac->hsolver, indx, 2)); 
  }
  ierr = PetscOptionsEList("-pc_hypre_boomeramg_relax_type_coarse","Relax type on coarse grid","None",HYPREBoomerAMGRelaxType,10,HYPREBoomerAMGRelaxType[9],&indx,&flg);CHKERRQ(ierr);
  if (flg) {
    jac->relaxtype[2] = indx;
    PetscStackCallHypre(0,HYPRE_BoomerAMGSetCycleRelaxType,(jac->hsolver, indx, 3)); 
  }

  /* Relaxation Weight */
  ierr = PetscOptionsReal("-pc_hypre_boomeramg_relax_weight_all","Relaxation weight for all levels (0 = hypre estimates, -k = determined with k CG steps)","None",jac->relaxweight, &tmpdbl ,&flg);CHKERRQ(ierr);
  if (flg) {
    PetscStackCallHypre(0,HYPRE_BoomerAMGSetRelaxWt,(jac->hsolver,tmpdbl)); 
    jac->relaxweight = tmpdbl;
  }

  n=2;
  twodbl[0] = twodbl[1] = 1.0;
  ierr = PetscOptionsRealArray("-pc_hypre_boomeramg_relax_weight_level","Set the relaxation weight for a particular level (weight,level)","None",twodbl, &n, &flg);CHKERRQ(ierr);
  if (flg) {
    if (n == 2) {
      indx =  (int)PetscAbsReal(twodbl[1]); 
      PetscStackCallHypre(0,HYPRE_BoomerAMGSetLevelRelaxWt,(jac->hsolver,twodbl[0],indx));
    } else {
      SETERRQ1(((PetscObject)pc)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Relax weight level: you must provide 2 values separated by a comma (and no space), you provided %d",n);
    }
  }

  /* Outer relaxation Weight */
  ierr = PetscOptionsReal("-pc_hypre_boomeramg_outer_relax_weight_all","Outer relaxation weight for all levels ( -k = determined with k CG steps)","None",jac->outerrelaxweight, &tmpdbl ,&flg);CHKERRQ(ierr);
  if (flg) {
    PetscStackCallHypre(0,HYPRE_BoomerAMGSetOuterWt,( jac->hsolver, tmpdbl));
    jac->outerrelaxweight = tmpdbl;
  }

  n=2;
  twodbl[0] = twodbl[1] = 1.0;
  ierr = PetscOptionsRealArray("-pc_hypre_boomeramg_outer_relax_weight_level","Set the outer relaxation weight for a particular level (weight,level)","None",twodbl, &n, &flg);CHKERRQ(ierr);
  if (flg) {
    if (n == 2) {
      indx =  (int)PetscAbsReal(twodbl[1]); 
      PetscStackCallHypre(0,HYPRE_BoomerAMGSetLevelOuterWt,( jac->hsolver, twodbl[0], indx)); 
    } else {
      SETERRQ1(((PetscObject)pc)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Relax weight outer level: You must provide 2 values separated by a comma (and no space), you provided %d",n);
    }
  }

  /* the Relax Order */ 
  ierr = PetscOptionsBool( "-pc_hypre_boomeramg_no_CF", "Do not use CF-relaxation", "None", PETSC_FALSE, &tmp_truth, &flg);CHKERRQ(ierr);

  if (flg) {
    jac->relaxorder = 0;
    PetscStackCallHypre(0,HYPRE_BoomerAMGSetRelaxOrder,(jac->hsolver, jac->relaxorder)); 
  }
  ierr = PetscOptionsEList("-pc_hypre_boomeramg_measure_type","Measure type","None",HYPREBoomerAMGMeasureType,2,HYPREBoomerAMGMeasureType[0],&indx,&flg);CHKERRQ(ierr);
  if (flg) {
    jac->measuretype = indx;
    PetscStackCallHypre(0,HYPRE_BoomerAMGSetMeasureType,(jac->hsolver,jac->measuretype)); 
  }
  /* update list length 3/07 */
  ierr = PetscOptionsEList("-pc_hypre_boomeramg_coarsen_type","Coarsen type","None",HYPREBoomerAMGCoarsenType,11,HYPREBoomerAMGCoarsenType[6],&indx,&flg);CHKERRQ(ierr);
  if (flg) {
    jac->coarsentype = indx;
    PetscStackCallHypre(0,HYPRE_BoomerAMGSetCoarsenType,(jac->hsolver,jac->coarsentype)); 
  }
 
  /* new 3/07 */
  ierr = PetscOptionsEList("-pc_hypre_boomeramg_interp_type","Interpolation type","None",HYPREBoomerAMGInterpType,14,HYPREBoomerAMGInterpType[0],&indx,&flg);CHKERRQ(ierr);
  if (flg) {
    jac->interptype = indx;
    PetscStackCallHypre(0,HYPRE_BoomerAMGSetInterpType,(jac->hsolver,jac->interptype)); 
  }

  ierr = PetscOptionsName("-pc_hypre_boomeramg_print_statistics","Print statistics","None",&flg);CHKERRQ(ierr);
  if (flg) {
    level = 3;
    ierr = PetscOptionsInt("-pc_hypre_boomeramg_print_statistics","Print statistics","None",level,&level,PETSC_NULL);CHKERRQ(ierr);
    jac->printstatistics = PETSC_TRUE;
    PetscStackCallHypre(0,HYPRE_BoomerAMGSetPrintLevel,(jac->hsolver,level));
  }

  ierr = PetscOptionsName("-pc_hypre_boomeramg_print_debug","Print debug information","None",&flg);CHKERRQ(ierr);
  if (flg) {
    level = 3;
    ierr = PetscOptionsInt("-pc_hypre_boomeramg_print_debug","Print debug information","None",level,&level,PETSC_NULL);CHKERRQ(ierr);
    jac->printstatistics = PETSC_TRUE;
    PetscStackCallHypre(0,HYPRE_BoomerAMGSetDebugFlag,(jac->hsolver,level));
  }

  ierr = PetscOptionsBool( "-pc_hypre_boomeramg_nodal_coarsen", "HYPRE_BoomerAMGSetNodal()", "None", PETSC_FALSE, &tmp_truth, &flg);CHKERRQ(ierr);
  if (flg && tmp_truth) {
    jac->nodal_coarsen = 1;
    PetscStackCallHypre(0,HYPRE_BoomerAMGSetNodal,(jac->hsolver,1));
  }

  ierr = PetscOptionsBool( "-pc_hypre_boomeramg_nodal_relaxation", "Nodal relaxation via Schwarz", "None", PETSC_FALSE, &tmp_truth, &flg);CHKERRQ(ierr);
  if (flg && tmp_truth) {
    PetscInt tmp_int;
    ierr = PetscOptionsInt( "-pc_hypre_boomeramg_nodal_relaxation", "Nodal relaxation via Schwarz", "None",jac->nodal_relax_levels,&tmp_int,&flg);CHKERRQ(ierr);
    if (flg) jac->nodal_relax_levels = tmp_int;
    PetscStackCallHypre(0,HYPRE_BoomerAMGSetSmoothType,(jac->hsolver,6));
    PetscStackCallHypre(0,HYPRE_BoomerAMGSetDomainType,(jac->hsolver,1));
    PetscStackCallHypre(0,HYPRE_BoomerAMGSetOverlap,(jac->hsolver,0));
    PetscStackCallHypre(0,HYPRE_BoomerAMGSetSmoothNumLevels,(jac->hsolver,jac->nodal_relax_levels));
  }

  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApplyRichardson_HYPRE_BoomerAMG"
static PetscErrorCode PCApplyRichardson_HYPRE_BoomerAMG(PC pc,Vec b,Vec y,Vec w,PetscReal rtol,PetscReal abstol, PetscReal dtol,PetscInt its,PetscBool  guesszero,PetscInt *outits,PCRichardsonConvergedReason *reason)
{
  PC_HYPRE       *jac = (PC_HYPRE*)pc->data;
  PetscErrorCode ierr;
  PetscInt       oits;

  PetscFunctionBegin;
  PetscStackCallHypre(0,HYPRE_BoomerAMGSetMaxIter,(jac->hsolver,its*jac->maxiter));
  PetscStackCallHypre(0,HYPRE_BoomerAMGSetTol,(jac->hsolver,rtol));
  jac->applyrichardson = PETSC_TRUE;
  ierr = PCApply_HYPRE(pc,b,y);CHKERRQ(ierr);
  jac->applyrichardson = PETSC_FALSE;
  PetscStackCallHypre(0,HYPRE_BoomerAMGGetNumIterations,(jac->hsolver,&oits));
  *outits = oits;
  if (oits == its) *reason = PCRICHARDSON_CONVERGED_ITS;
  else             *reason = PCRICHARDSON_CONVERGED_RTOL;
  PetscStackCallHypre(0,HYPRE_BoomerAMGSetTol,(jac->hsolver,jac->tol));
  PetscStackCallHypre(0,HYPRE_BoomerAMGSetMaxIter,(jac->hsolver,jac->maxiter));
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
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG preconditioning\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Cycle type %s\n",HYPREBoomerAMGCycleType[jac->cycletype]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Maximum number of levels %d\n",jac->maxlevels);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Maximum number of iterations PER hypre call %d\n",jac->maxiter);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Convergence tolerance PER hypre call %G\n",jac->tol);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Threshold for strong coupling %G\n",jac->strongthreshold);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Interpolation truncation factor %G\n",jac->truncfactor);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Interpolation: max elements per row %d\n",jac->pmax);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Number of levels of aggressive coarsening %d\n",jac->agg_nl);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Number of paths for aggressive coarsening %d\n",jac->agg_num_paths);CHKERRQ(ierr);
 
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Maximum row sums %G\n",jac->maxrowsum);CHKERRQ(ierr);

    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Sweeps down         %d\n",jac->gridsweeps[0]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Sweeps up           %d\n",jac->gridsweeps[1]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Sweeps on coarse    %d\n",jac->gridsweeps[2]);CHKERRQ(ierr);

    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Relax down          %s\n",HYPREBoomerAMGRelaxType[jac->relaxtype[0]]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Relax up            %s\n",HYPREBoomerAMGRelaxType[jac->relaxtype[1]]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Relax on coarse     %s\n",HYPREBoomerAMGRelaxType[jac->relaxtype[2]]);CHKERRQ(ierr);

    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Relax weight  (all)      %G\n",jac->relaxweight);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Outer relax weight (all) %G\n",jac->outerrelaxweight);CHKERRQ(ierr); 

    if (jac->relaxorder) {
      ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Using CF-relaxation\n");CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Not using CF-relaxation\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Measure type        %s\n",HYPREBoomerAMGMeasureType[jac->measuretype]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Coarsen type        %s\n",HYPREBoomerAMGCoarsenType[jac->coarsentype]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE BoomerAMG: Interpolation type  %s\n",HYPREBoomerAMGInterpType[jac->interptype]);CHKERRQ(ierr);
    if (jac->nodal_coarsen) {
      ierr = PetscViewerASCIIPrintf(viewer," HYPRE BoomerAMG: Using nodal coarsening (with HYPRE_BOOMERAMGSetNodal())\n");CHKERRQ(ierr);
    }
    if (jac->nodal_relax) {
      ierr = PetscViewerASCIIPrintf(viewer," HYPRE BoomerAMG: Using nodal relaxation via Schwarz smoothing on levels %d\n",jac->nodal_relax_levels);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "PCSetFromOptions_HYPRE_ParaSails"
static PetscErrorCode PCSetFromOptions_HYPRE_ParaSails(PC pc)
{
  PC_HYPRE       *jac = (PC_HYPRE*)pc->data;
  PetscErrorCode ierr;
  PetscInt       indx;
  PetscBool      flag;
  const char     *symtlist[] = {"nonsymmetric","SPD","nonsymmetric,SPD"};

  PetscFunctionBegin;
  ierr = PetscOptionsHead("HYPRE ParaSails Options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-pc_hypre_parasails_nlevels","Number of number of levels","None",jac->nlevels,&jac->nlevels,0);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-pc_hypre_parasails_thresh","Threshold","None",jac->threshhold,&jac->threshhold,&flag);CHKERRQ(ierr);
  if (flag) {
    PetscStackCallHypre(0,HYPRE_ParaSailsSetParams,(jac->hsolver,jac->threshhold,jac->nlevels));
  }

  ierr = PetscOptionsReal("-pc_hypre_parasails_filter","filter","None",jac->filter,&jac->filter,&flag);CHKERRQ(ierr);
  if (flag) {
    PetscStackCallHypre(0,HYPRE_ParaSailsSetFilter,(jac->hsolver,jac->filter));
  }

  ierr = PetscOptionsReal("-pc_hypre_parasails_loadbal","Load balance","None",jac->loadbal,&jac->loadbal,&flag);CHKERRQ(ierr);
  if (flag) {
    PetscStackCallHypre(0,HYPRE_ParaSailsSetLoadbal,(jac->hsolver,jac->loadbal));
  }

  ierr = PetscOptionsBool("-pc_hypre_parasails_logging","Print info to screen","None",(PetscBool)jac->logging,(PetscBool*)&jac->logging,&flag);CHKERRQ(ierr);
  if (flag) {
    PetscStackCallHypre(0,HYPRE_ParaSailsSetLogging,(jac->hsolver,jac->logging));
  }

  ierr = PetscOptionsBool("-pc_hypre_parasails_reuse","Reuse nonzero pattern in preconditioner","None",(PetscBool)jac->ruse,(PetscBool*)&jac->ruse,&flag);CHKERRQ(ierr);
  if (flag) {
    PetscStackCallHypre(0,HYPRE_ParaSailsSetReuse,(jac->hsolver,jac->ruse));
  }

  ierr = PetscOptionsEList("-pc_hypre_parasails_sym","Symmetry of matrix and preconditioner","None",symtlist,3,symtlist[0],&indx,&flag);CHKERRQ(ierr);
  if (flag) {
    jac->symt = indx;
    PetscStackCallHypre(0,HYPRE_ParaSailsSetSym,(jac->hsolver,jac->symt));
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
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE ParaSails preconditioning\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE ParaSails: nlevels %d\n",jac->nlevels);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE ParaSails: threshold %G\n",jac->threshhold);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE ParaSails: filter %G\n",jac->filter);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE ParaSails: load balance %G\n",jac->loadbal);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE ParaSails: reuse nonzero structure %s\n",PetscBools[jac->ruse]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE ParaSails: print info to screen %s\n",PetscBools[jac->logging]);CHKERRQ(ierr);
    if (!jac->symt) {
      symt = "nonsymmetric matrix and preconditioner";
    } else if (jac->symt == 1) {
      symt = "SPD matrix and preconditioner";
    } else if (jac->symt == 2) {
      symt = "nonsymmetric matrix but SPD preconditioner";
    } else {
      SETERRQ1(((PetscObject)pc)->comm,PETSC_ERR_ARG_WRONG,"Unknown HYPRE ParaSails symmetric option %d",jac->symt);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  HYPRE ParaSails: %s\n",symt);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
/* ---------------------------------------------------------------------------------*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCHYPREGetType_HYPRE"
PetscErrorCode  PCHYPREGetType_HYPRE(PC pc,const char *name[])
{
  PC_HYPRE       *jac = (PC_HYPRE*)pc->data;

  PetscFunctionBegin;
  *name = jac->hypre_type;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCHYPRESetType_HYPRE"
PetscErrorCode  PCHYPRESetType_HYPRE(PC pc,const char name[])
{
  PC_HYPRE       *jac = (PC_HYPRE*)pc->data;
  PetscErrorCode ierr;
  PetscBool      flag;

  PetscFunctionBegin;
  if (jac->hypre_type) {
    ierr = PetscStrcmp(jac->hypre_type,name,&flag);CHKERRQ(ierr);
    if (!flag) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ORDER,"Cannot reset the HYPRE preconditioner type once it has been set");
    PetscFunctionReturn(0);
  } else {
    ierr = PetscStrallocpy(name, &jac->hypre_type);CHKERRQ(ierr);
  }
  
  jac->maxiter            = PETSC_DEFAULT;
  jac->tol                = PETSC_DEFAULT;
  jac->printstatistics    = PetscLogPrintInfo;

  ierr = PetscStrcmp("pilut",jac->hypre_type,&flag);CHKERRQ(ierr);
  if (flag) {
    PetscStackCallHypre(0,HYPRE_ParCSRPilutCreate,(jac->comm_hypre,&jac->hsolver));
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
    PetscStackCallHypre(0,HYPRE_ParaSailsCreate,(jac->comm_hypre,&jac->hsolver));
    pc->ops->setfromoptions = PCSetFromOptions_HYPRE_ParaSails;
    pc->ops->view           = PCView_HYPRE_ParaSails;
    jac->destroy            = HYPRE_ParaSailsDestroy;
    jac->setup              = HYPRE_ParaSailsSetup;
    jac->solve              = HYPRE_ParaSailsSolve;
    /* initialize */
    jac->nlevels     = 1;
    jac->threshhold  = .1;
    jac->filter      = .1;
    jac->loadbal     = 0;
    if (PetscLogPrintInfo) {
      jac->logging     = (int) PETSC_TRUE;
    } else {
      jac->logging     = (int) PETSC_FALSE;
    }
    jac->ruse = (int) PETSC_FALSE;
    jac->symt   = 0;
    PetscStackCallHypre(0,HYPRE_ParaSailsSetParams,(jac->hsolver,jac->threshhold,jac->nlevels));
    PetscStackCallHypre(0,HYPRE_ParaSailsSetFilter,(jac->hsolver,jac->filter));
    PetscStackCallHypre(0,HYPRE_ParaSailsSetLoadbal,(jac->hsolver,jac->loadbal));
    PetscStackCallHypre(0,HYPRE_ParaSailsSetLogging,(jac->hsolver,jac->logging));
    PetscStackCallHypre(0,HYPRE_ParaSailsSetReuse,(jac->hsolver,jac->ruse));
    PetscStackCallHypre(0,HYPRE_ParaSailsSetSym,(jac->hsolver,jac->symt));
    PetscFunctionReturn(0);
  }
  ierr = PetscStrcmp("euclid",jac->hypre_type,&flag);CHKERRQ(ierr);
  if (flag) {
    ierr                    = HYPRE_EuclidCreate(jac->comm_hypre,&jac->hsolver);
    pc->ops->setfromoptions = PCSetFromOptions_HYPRE_Euclid;
    pc->ops->view           = PCView_HYPRE_Euclid;
    jac->destroy            = HYPRE_EuclidDestroy;
    jac->setup              = HYPRE_EuclidSetup;
    jac->solve              = HYPRE_EuclidSolve;
    /* initialization */
    jac->bjilu              = PETSC_FALSE;
    jac->levels             = 1;
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

    jac->nodal_coarsen    = 0;
    jac->nodal_relax      = PETSC_FALSE;
    jac->nodal_relax_levels = 1;
    PetscStackCallHypre(0,HYPRE_BoomerAMGSetCycleType,(jac->hsolver,jac->cycletype));
    PetscStackCallHypre(0,HYPRE_BoomerAMGSetMaxLevels,(jac->hsolver,jac->maxlevels));
    PetscStackCallHypre(0,HYPRE_BoomerAMGSetMaxIter,(jac->hsolver,jac->maxiter));
    PetscStackCallHypre(0,HYPRE_BoomerAMGSetTol,(jac->hsolver,jac->tol));
    PetscStackCallHypre(0,HYPRE_BoomerAMGSetTruncFactor,(jac->hsolver,jac->truncfactor));
    PetscStackCallHypre(0,HYPRE_BoomerAMGSetStrongThreshold,(jac->hsolver,jac->strongthreshold));
    PetscStackCallHypre(0,HYPRE_BoomerAMGSetMaxRowSum,(jac->hsolver,jac->maxrowsum));
    PetscStackCallHypre(0,HYPRE_BoomerAMGSetCoarsenType,(jac->hsolver,jac->coarsentype));
    PetscStackCallHypre(0,HYPRE_BoomerAMGSetMeasureType,(jac->hsolver,jac->measuretype));
    PetscStackCallHypre(0,HYPRE_BoomerAMGSetRelaxOrder,(jac->hsolver, jac->relaxorder));
    PetscStackCallHypre(0,HYPRE_BoomerAMGSetInterpType,(jac->hsolver,jac->interptype)); 
    PetscStackCallHypre(0,HYPRE_BoomerAMGSetAggNumLevels,(jac->hsolver,jac->agg_nl));
    PetscStackCallHypre(0,HYPRE_BoomerAMGSetPMaxElmts,(jac->hsolver,jac->pmax));
    PetscStackCallHypre(0,HYPRE_BoomerAMGSetNumPaths,(jac->hsolver,jac->agg_num_paths));
    PetscStackCallHypre(0,HYPRE_BoomerAMGSetRelaxType,(jac->hsolver, jac->relaxtype[0]));  /*defaults coarse to 9*/
    PetscStackCallHypre(0,HYPRE_BoomerAMGSetNumSweeps,(jac->hsolver, jac->gridsweeps[0])); /*defaults coarse to 1 */
    PetscFunctionReturn(0);
  }
  ierr = PetscFree(jac->hypre_type);CHKERRQ(ierr);
  jac->hypre_type = PETSC_NULL;
  SETERRQ1(((PetscObject)pc)->comm,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unknown HYPRE preconditioner %s; Choices are pilut, parasails, euclid, boomeramg",name);
  PetscFunctionReturn(0);
}
EXTERN_C_END

/*
    It only gets here if the HYPRE type has not been set before the call to 
   ...SetFromOptions() which actually is most of the time
*/
#undef __FUNCT__  
#define __FUNCT__ "PCSetFromOptions_HYPRE"
static PetscErrorCode PCSetFromOptions_HYPRE(PC pc)
{
  PetscErrorCode ierr;
  PetscInt       indx;
  const char     *type[] = {"pilut","parasails","boomeramg","euclid"};
  PetscBool      flg;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("HYPRE preconditioner options");CHKERRQ(ierr);
  ierr = PetscOptionsEList("-pc_hypre_type","HYPRE preconditioner type","PCHYPRESetType",type,4,"boomeramg",&indx,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PCHYPRESetType_HYPRE(pc,type[indx]);CHKERRQ(ierr);
  } else {
    ierr = PCHYPRESetType_HYPRE(pc,"boomeramg");CHKERRQ(ierr);
  }
  if (pc->ops->setfromoptions) {
    ierr = pc->ops->setfromoptions(pc);CHKERRQ(ierr);
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
-     name - either  pilut, parasails, boomeramg, euclid

   Options Database Keys:
   -pc_hypre_type - One of pilut, parasails, boomeramg, euclid
 
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
.     name - either  pilut, parasails, boomeramg, euclid

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
+   -pc_hypre_type - One of pilut, parasails, boomeramg, euclid
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

          See PCPFMG for access to the hypre Struct PFMG solver

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC,
           PCHYPRESetType(), PCPFMG

M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCreate_HYPRE"
PetscErrorCode  PCCreate_HYPRE(PC pc)
{
  PC_HYPRE       *jac;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(pc,PC_HYPRE,&jac);CHKERRQ(ierr);
  pc->data                 = jac;
  pc->ops->destroy         = PCDestroy_HYPRE;
  pc->ops->setfromoptions  = PCSetFromOptions_HYPRE;
  pc->ops->setup           = PCSetUp_HYPRE;
  pc->ops->apply           = PCApply_HYPRE;
  jac->comm_hypre          = MPI_COMM_NULL;
  jac->hypre_type          = PETSC_NULL;
  /* duplicate communicator for hypre */
  ierr = MPI_Comm_dup(((PetscObject)pc)->comm,&(jac->comm_hypre));CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCHYPRESetType_C","PCHYPRESetType_HYPRE",PCHYPRESetType_HYPRE);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCHYPREGetType_C","PCHYPREGetType_HYPRE",PCHYPREGetType_HYPRE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

/* ---------------------------------------------------------------------------------------------------------------------------------*/

/* this include is needed ONLY to allow access to the private data inside the Mat object specific to hypre */
#include <petsc-private/matimpl.h>

typedef struct {
  MPI_Comm            hcomm;       /* does not share comm with HYPRE_StructMatrix because need to create solver before getting matrix */
  HYPRE_StructSolver  hsolver;

  /* keep copy of PFMG options used so may view them */
  PetscInt            its;   
  double              tol;
  PetscInt            relax_type;
  PetscInt            rap_type;
  PetscInt            num_pre_relax,num_post_relax;
  PetscInt            max_levels;
} PC_PFMG;

#undef __FUNCT__
#define __FUNCT__ "PCDestroy_PFMG"
PetscErrorCode PCDestroy_PFMG(PC pc)
{
  PetscErrorCode ierr;
  PC_PFMG        *ex = (PC_PFMG*) pc->data;

  PetscFunctionBegin;
  if (ex->hsolver) {PetscStackCallHypre(0,HYPRE_StructPFMGDestroy,(ex->hsolver));}
  ierr = MPI_Comm_free(&ex->hcomm);CHKERRQ(ierr);
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static const char *PFMGRelaxType[]   = {"Jacobi","Weighted-Jacobi","symmetric-Red/Black-Gauss-Seidel","Red/Black-Gauss-Seidel"};
static const char *PFMGRAPType[]   = {"Galerkin","non-Galerkin"};

#undef __FUNCT__  
#define __FUNCT__ "PCView_PFMG"
PetscErrorCode PCView_PFMG(PC pc,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      iascii;
  PC_PFMG        *ex = (PC_PFMG*) pc->data;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
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
PetscErrorCode PCSetFromOptions_PFMG(PC pc)
{
  PetscErrorCode ierr;
  PC_PFMG        *ex = (PC_PFMG*) pc->data;
  PetscBool      flg = PETSC_FALSE;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("PFMG options");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_pfmg_print_statistics","Print statistics","HYPRE_StructPFMGSetPrintLevel",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg) {
    int level=3;
    PetscStackCallHypre(0,HYPRE_StructPFMGSetPrintLevel,(ex->hsolver,level));
  }
  ierr = PetscOptionsInt("-pc_pfmg_its","Number of iterations of PFMG to use as preconditioner","HYPRE_StructPFMGSetMaxIter",ex->its,&ex->its,PETSC_NULL);CHKERRQ(ierr);
  PetscStackCallHypre(0,HYPRE_StructPFMGSetMaxIter,(ex->hsolver,ex->its));
  ierr = PetscOptionsInt("-pc_pfmg_num_pre_relax","Number of smoothing steps before coarse grid","HYPRE_StructPFMGSetNumPreRelax",ex->num_pre_relax,&ex->num_pre_relax,PETSC_NULL);CHKERRQ(ierr);
  PetscStackCallHypre(0,HYPRE_StructPFMGSetNumPreRelax,(ex->hsolver,ex->num_pre_relax));
  ierr = PetscOptionsInt("-pc_pfmg_num_post_relax","Number of smoothing steps after coarse grid","HYPRE_StructPFMGSetNumPostRelax",ex->num_post_relax,&ex->num_post_relax,PETSC_NULL);CHKERRQ(ierr);
  PetscStackCallHypre(0,HYPRE_StructPFMGSetNumPostRelax,(ex->hsolver,ex->num_post_relax));

  ierr = PetscOptionsInt("-pc_pfmg_max_levels","Max Levels for MG hierarchy","HYPRE_StructPFMGSetMaxLevels",ex->max_levels,&ex->max_levels,PETSC_NULL);CHKERRQ(ierr);
  PetscStackCallHypre(0,HYPRE_StructPFMGSetMaxLevels,(ex->hsolver,ex->max_levels));

  ierr = PetscOptionsReal("-pc_pfmg_tol","Tolerance of PFMG","HYPRE_StructPFMGSetTol",ex->tol,&ex->tol,PETSC_NULL);CHKERRQ(ierr);
  PetscStackCallHypre(0,HYPRE_StructPFMGSetTol,(ex->hsolver,ex->tol)); 
  ierr = PetscOptionsEList("-pc_pfmg_relax_type","Relax type for the up and down cycles","HYPRE_StructPFMGSetRelaxType",PFMGRelaxType,4,PFMGRelaxType[ex->relax_type],&ex->relax_type,PETSC_NULL);CHKERRQ(ierr);
  PetscStackCallHypre(0,HYPRE_StructPFMGSetRelaxType,(ex->hsolver, ex->relax_type)); 
  ierr = PetscOptionsEList("-pc_pfmg_rap_type","RAP type","HYPRE_StructPFMGSetRAPType",PFMGRAPType,2,PFMGRAPType[ex->rap_type],&ex->rap_type,PETSC_NULL);CHKERRQ(ierr);
  PetscStackCallHypre(0,HYPRE_StructPFMGSetRAPType,(ex->hsolver, ex->rap_type)); 
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApply_PFMG"
PetscErrorCode PCApply_PFMG(PC pc,Vec x,Vec y)
{
  PetscErrorCode  ierr;
  PC_PFMG         *ex = (PC_PFMG*) pc->data;
  PetscScalar     *xx,*yy;
  PetscInt        ilower[3],iupper[3];
  Mat_HYPREStruct *mx = (Mat_HYPREStruct *)(pc->pmat->data);

  PetscFunctionBegin;
  ierr = DMDAGetCorners(mx->da,&ilower[0],&ilower[1],&ilower[2],&iupper[0],&iupper[1],&iupper[2]);CHKERRQ(ierr);
  iupper[0] += ilower[0] - 1;    
  iupper[1] += ilower[1] - 1;    
  iupper[2] += ilower[2] - 1;    

  /* copy x values over to hypre */
  PetscStackCallHypre(0,HYPRE_StructVectorSetConstantValues,(mx->hb,0.0));
  ierr = VecGetArray(x,&xx);CHKERRQ(ierr);
  PetscStackCallHypre(0,HYPRE_StructVectorSetBoxValues,(mx->hb,ilower,iupper,xx));
  ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);
  PetscStackCallHypre(0,HYPRE_StructVectorAssemble,(mx->hb));
  PetscStackCallHypre(0,HYPRE_StructPFMGSolve,(ex->hsolver,mx->hmat,mx->hb,mx->hx));

  /* copy solution values back to PETSc */
  ierr = VecGetArray(y,&yy);CHKERRQ(ierr);
  PetscStackCallHypre(0,HYPRE_StructVectorGetBoxValues,(mx->hx,ilower,iupper,yy));
  ierr = VecRestoreArray(y,&yy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApplyRichardson_PFMG"
static PetscErrorCode PCApplyRichardson_PFMG(PC pc,Vec b,Vec y,Vec w,PetscReal rtol,PetscReal abstol, PetscReal dtol,PetscInt its,PetscBool  guesszero,PetscInt *outits,PCRichardsonConvergedReason *reason)
{
  PC_PFMG        *jac = (PC_PFMG*)pc->data;
  PetscErrorCode ierr;
  PetscInt       oits;

  PetscFunctionBegin;
  PetscStackCallHypre(0,HYPRE_StructPFMGSetMaxIter,(jac->hsolver,its*jac->its));
  PetscStackCallHypre(0,HYPRE_StructPFMGSetTol,(jac->hsolver,rtol));

  ierr = PCApply_PFMG(pc,b,y);CHKERRQ(ierr);
  PetscStackCallHypre(0,HYPRE_StructPFMGGetNumIterations,(jac->hsolver,&oits));
  *outits = oits;
  if (oits == its) *reason = PCRICHARDSON_CONVERGED_ITS;
  else             *reason = PCRICHARDSON_CONVERGED_RTOL;
  PetscStackCallHypre(0,HYPRE_StructPFMGSetTol,(jac->hsolver,jac->tol));
  PetscStackCallHypre(0,HYPRE_StructPFMGSetMaxIter,(jac->hsolver,jac->its));
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PCSetUp_PFMG"
PetscErrorCode PCSetUp_PFMG(PC pc)
{
  PetscErrorCode  ierr;
  PC_PFMG         *ex = (PC_PFMG*) pc->data;
  Mat_HYPREStruct *mx = (Mat_HYPREStruct *)(pc->pmat->data);
  PetscBool       flg;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)pc->pmat,MATHYPRESTRUCT,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_INCOMP,"Must use MATHYPRESTRUCT with this preconditioner");

  /* create the hypre solver object and set its information */
  if (ex->hsolver) {
    PetscStackCallHypre(0,HYPRE_StructPFMGDestroy,(ex->hsolver));
  }
  PetscStackCallHypre(0,HYPRE_StructPFMGCreate,(ex->hcomm,&ex->hsolver));
  ierr = PCSetFromOptions_PFMG(pc);CHKERRQ(ierr);
  PetscStackCallHypre(0,HYPRE_StructPFMGSetup,(ex->hsolver,mx->hmat,mx->hb,mx->hx));
  PetscStackCallHypre(0,HYPRE_StructPFMGSetZeroGuess,(ex->hsolver));
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

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCreate_PFMG"
PetscErrorCode  PCCreate_PFMG(PC pc)
{
  PetscErrorCode ierr;
  PC_PFMG        *ex;

  PetscFunctionBegin;
  ierr = PetscNew(PC_PFMG,&ex);CHKERRQ(ierr);\
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
  ierr = MPI_Comm_dup(((PetscObject)pc)->comm,&(ex->hcomm));CHKERRQ(ierr);
  PetscStackCallHypre(0,HYPRE_StructPFMGCreate,(ex->hcomm,&ex->hsolver));
  PetscFunctionReturn(0);
}
EXTERN_C_END

/* ---------------------------------------------------------------------------------------------------------------------------------------------------*/

/* we know we are working with a HYPRE_SStructMatrix */
typedef struct {
  MPI_Comm            hcomm;       /* does not share comm with HYPRE_SStructMatrix because need to create solver before getting matrix */
  HYPRE_SStructSolver  ss_solver;

  /* keep copy of SYSPFMG options used so may view them */
  PetscInt            its;
  double              tol;
  PetscInt            relax_type;
  PetscInt            num_pre_relax,num_post_relax;
} PC_SysPFMG;

#undef __FUNCT__
#define __FUNCT__ "PCDestroy_SysPFMG"
PetscErrorCode PCDestroy_SysPFMG(PC pc)
{
  PetscErrorCode ierr;
  PC_SysPFMG    *ex = (PC_SysPFMG*) pc->data;

  PetscFunctionBegin;
  if (ex->ss_solver) {PetscStackCallHypre(0,HYPRE_SStructSysPFMGDestroy,(ex->ss_solver));}
  ierr = MPI_Comm_free(&ex->hcomm);CHKERRQ(ierr);
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static const char *SysPFMGRelaxType[]   = {"Weighted-Jacobi","Red/Black-Gauss-Seidel"};

#undef __FUNCT__  
#define __FUNCT__ "PCView_SysPFMG"
PetscErrorCode PCView_SysPFMG(PC pc,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      iascii;
  PC_SysPFMG    *ex = (PC_SysPFMG*) pc->data;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
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
PetscErrorCode PCSetFromOptions_SysPFMG(PC pc)
{
  PetscErrorCode ierr;
  PC_SysPFMG    *ex = (PC_SysPFMG*) pc->data;
  PetscBool      flg = PETSC_FALSE;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("SysPFMG options");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-pc_syspfmg_print_statistics","Print statistics","HYPRE_SStructSysPFMGSetPrintLevel",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg) {
    int level=3;
    PetscStackCallHypre(0,HYPRE_SStructSysPFMGSetPrintLevel,(ex->ss_solver,level));
  }
  ierr = PetscOptionsInt("-pc_syspfmg_its","Number of iterations of SysPFMG to use as preconditioner","HYPRE_SStructSysPFMGSetMaxIter",ex->its,&ex->its,PETSC_NULL);CHKERRQ(ierr);
  PetscStackCallHypre(0,HYPRE_SStructSysPFMGSetMaxIter,(ex->ss_solver,ex->its));
  ierr = PetscOptionsInt("-pc_syspfmg_num_pre_relax","Number of smoothing steps before coarse grid","HYPRE_SStructSysPFMGSetNumPreRelax",ex->num_pre_relax,&ex->num_pre_relax,PETSC_NULL);CHKERRQ(ierr);
  PetscStackCallHypre(0,HYPRE_SStructSysPFMGSetNumPreRelax,(ex->ss_solver,ex->num_pre_relax));
  ierr = PetscOptionsInt("-pc_syspfmg_num_post_relax","Number of smoothing steps after coarse grid","HYPRE_SStructSysPFMGSetNumPostRelax",ex->num_post_relax,&ex->num_post_relax,PETSC_NULL);CHKERRQ(ierr);
  PetscStackCallHypre(0,HYPRE_SStructSysPFMGSetNumPostRelax,(ex->ss_solver,ex->num_post_relax));

  ierr = PetscOptionsReal("-pc_syspfmg_tol","Tolerance of SysPFMG","HYPRE_SStructSysPFMGSetTol",ex->tol,&ex->tol,PETSC_NULL);CHKERRQ(ierr);
  PetscStackCallHypre(0,HYPRE_SStructSysPFMGSetTol,(ex->ss_solver,ex->tol)); 
  ierr = PetscOptionsEList("-pc_syspfmg_relax_type","Relax type for the up and down cycles","HYPRE_SStructSysPFMGSetRelaxType",SysPFMGRelaxType,4,SysPFMGRelaxType[ex->relax_type],&ex->relax_type,PETSC_NULL);CHKERRQ(ierr);
  PetscStackCallHypre(0,HYPRE_SStructSysPFMGSetRelaxType,(ex->ss_solver, ex->relax_type)); 
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApply_SysPFMG"
PetscErrorCode PCApply_SysPFMG(PC pc,Vec x,Vec y)
{
  PetscErrorCode    ierr;
  PC_SysPFMG       *ex = (PC_SysPFMG*) pc->data;
  PetscScalar      *xx,*yy;
  PetscInt          ilower[3],iupper[3];
  Mat_HYPRESStruct *mx = (Mat_HYPRESStruct *)(pc->pmat->data);
  PetscInt          ordering= mx->dofs_order;
  PetscInt          nvars= mx->nvars;
  PetscInt          part= 0;
  PetscInt          size;
  PetscInt          i;

  PetscFunctionBegin;
  ierr = DMDAGetCorners(mx->da,&ilower[0],&ilower[1],&ilower[2],&iupper[0],&iupper[1],&iupper[2]);CHKERRQ(ierr);
  iupper[0] += ilower[0] - 1;    
  iupper[1] += ilower[1] - 1;    
  iupper[2] += ilower[2] - 1;    

  size= 1;
  for (i= 0; i< 3; i++) {
     size*= (iupper[i]-ilower[i]+1);
  }
  /* copy x values over to hypre for variable ordering */
  if (ordering) {
    PetscStackCallHypre(0,HYPRE_SStructVectorSetConstantValues,(mx->ss_b,0.0));
     ierr = VecGetArray(x,&xx);CHKERRQ(ierr);
     for (i= 0; i< nvars; i++) {
       PetscStackCallHypre(0,HYPRE_SStructVectorSetBoxValues,(mx->ss_b,part,ilower,iupper,i,xx+(size*i)));
     }
     ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);
     PetscStackCallHypre(0,HYPRE_SStructVectorAssemble,(mx->ss_b));
     PetscStackCallHypre(0,HYPRE_SStructMatrixMatvec,(1.0,mx->ss_mat,mx->ss_b,0.0,mx->ss_x));
     PetscStackCallHypre(0,HYPRE_SStructSysPFMGSolve,(ex->ss_solver,mx->ss_mat,mx->ss_b,mx->ss_x));

     /* copy solution values back to PETSc */
     ierr = VecGetArray(y,&yy);CHKERRQ(ierr);
     for (i= 0; i< nvars; i++) {
       PetscStackCallHypre(0,HYPRE_SStructVectorGetBoxValues,(mx->ss_x,part,ilower,iupper,i,yy+(size*i)));
     }
     ierr = VecRestoreArray(y,&yy);CHKERRQ(ierr);
  } else {      /* nodal ordering must be mapped to variable ordering for sys_pfmg */
     PetscScalar     *z;
     PetscInt         j, k;

     ierr = PetscMalloc(nvars*size*sizeof(PetscScalar),&z);CHKERRQ(ierr);
     PetscStackCallHypre(0,HYPRE_SStructVectorSetConstantValues,(mx->ss_b,0.0));
     ierr = VecGetArray(x,&xx);CHKERRQ(ierr);

     /* transform nodal to hypre's variable ordering for sys_pfmg */
     for (i= 0; i< size; i++) {
        k= i*nvars;
        for (j= 0; j< nvars; j++) {
           z[j*size+i]= xx[k+j];
        }
     }
     for (i= 0; i< nvars; i++) {
       PetscStackCallHypre(0,HYPRE_SStructVectorSetBoxValues,(mx->ss_b,part,ilower,iupper,i,z+(size*i)));
     }
     ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);
     PetscStackCallHypre(0,HYPRE_SStructVectorAssemble,(mx->ss_b));
     PetscStackCallHypre(0,HYPRE_SStructSysPFMGSolve,(ex->ss_solver,mx->ss_mat,mx->ss_b,mx->ss_x));

     /* copy solution values back to PETSc */
     ierr = VecGetArray(y,&yy);CHKERRQ(ierr);
     for (i= 0; i< nvars; i++) {
       PetscStackCallHypre(0,HYPRE_SStructVectorGetBoxValues,(mx->ss_x,part,ilower,iupper,i,z+(size*i)));
     }
     /* transform hypre's variable ordering for sys_pfmg to nodal ordering */
     for (i= 0; i< size; i++) {
        k= i*nvars;
        for (j= 0; j< nvars; j++) {
           yy[k+j]= z[j*size+i];
        }
     }
     ierr = VecRestoreArray(y,&yy);CHKERRQ(ierr);
     ierr = PetscFree(z);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApplyRichardson_SysPFMG"
static PetscErrorCode PCApplyRichardson_SysPFMG(PC pc,Vec b,Vec y,Vec w,PetscReal rtol,PetscReal abstol, PetscReal dtol,PetscInt its,PetscBool  guesszero,PetscInt *outits,PCRichardsonConvergedReason *reason)
{
  PC_SysPFMG    *jac = (PC_SysPFMG*)pc->data;
  PetscErrorCode ierr;
  PetscInt       oits;

  PetscFunctionBegin;
  PetscStackCallHypre(0,HYPRE_SStructSysPFMGSetMaxIter,(jac->ss_solver,its*jac->its));
  PetscStackCallHypre(0,HYPRE_SStructSysPFMGSetTol,(jac->ss_solver,rtol));
  ierr = PCApply_SysPFMG(pc,b,y);CHKERRQ(ierr);
  PetscStackCallHypre(0,HYPRE_SStructSysPFMGGetNumIterations,(jac->ss_solver,&oits));
  *outits = oits;
  if (oits == its) *reason = PCRICHARDSON_CONVERGED_ITS;
  else             *reason = PCRICHARDSON_CONVERGED_RTOL;
  PetscStackCallHypre(0,HYPRE_SStructSysPFMGSetTol,(jac->ss_solver,jac->tol));
  PetscStackCallHypre(0,HYPRE_SStructSysPFMGSetMaxIter,(jac->ss_solver,jac->its));
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PCSetUp_SysPFMG"
PetscErrorCode PCSetUp_SysPFMG(PC pc)
{
  PetscErrorCode    ierr;
  PC_SysPFMG        *ex = (PC_SysPFMG*) pc->data;
  Mat_HYPRESStruct  *mx = (Mat_HYPRESStruct *)(pc->pmat->data);
  PetscBool         flg;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)pc->pmat,MATHYPRESSTRUCT,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_INCOMP,"Must use MATHYPRESSTRUCT with this preconditioner");

  /* create the hypre sstruct solver object and set its information */
  if (ex->ss_solver) {
    PetscStackCallHypre(0,HYPRE_SStructSysPFMGDestroy,(ex->ss_solver));
  }
  PetscStackCallHypre(0,HYPRE_SStructSysPFMGCreate,(ex->hcomm,&ex->ss_solver));
  ierr = PCSetFromOptions_SysPFMG(pc);CHKERRQ(ierr);
  PetscStackCallHypre(0,HYPRE_SStructSysPFMGSetZeroGuess,(ex->ss_solver));
  PetscStackCallHypre(0,HYPRE_SStructSysPFMGSetup,(ex->ss_solver,mx->ss_mat,mx->ss_b,mx->ss_x));
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

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCreate_SysPFMG"
PetscErrorCode  PCCreate_SysPFMG(PC pc)
{
  PetscErrorCode     ierr;
  PC_SysPFMG        *ex;

  PetscFunctionBegin;
  ierr = PetscNew(PC_SysPFMG,&ex);CHKERRQ(ierr);\
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
  ierr = MPI_Comm_dup(((PetscObject)pc)->comm,&(ex->hcomm));CHKERRQ(ierr);
  PetscStackCallHypre(0,HYPRE_SStructSysPFMGCreate,(ex->hcomm,&ex->ss_solver));
  PetscFunctionReturn(0);
}
EXTERN_C_END
