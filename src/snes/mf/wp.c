#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: wp.c,v 1.5 1998/12/17 22:12:27 bsmith Exp bsmith $";
#endif
/*
      Implements an alternative approach for computing the h
  parameter used with the finite difference based matrix free Jacobian.

      This uses the Walker-Pernice strategy. See snesmfjdef.c for 
  a full set of comments on the routines below.

        h = error_rel * sqrt(1 + ||U||) / ||a||

      Notes:
   1) || U || does not change between linear iterations so can be reused
   2) In GMRES || a || == 1 and so does not need to ever be computed

*/

/*
    This include file defines the data structure  MatSNESFDMF that 
   includes information about the computation of h. It is shared by 
   all implementations that people provide
*/
#include "src/snes/mf/snesmfj.h"   /*I  "snes.h"   I*/

typedef struct {
  double     normUfact;                   /* previous sqrt(1.0 + || U ||) */
  PetscTruth computenorma,computenormU;   
} MatSNESFDMFWP;

extern int VecNorm_Seq(Vec,NormType,double *);

#undef __FUNC__  
#define __FUNC__ "MatSNESFDMFCompute_WP"
/*
     MatSNESFDMFCompute_WP - Standard PETSc code for 
   computing h with matrix-free finite differences.

  Input Parameters:
+   ctx - the matrix free context
.   U - the location at which you want the Jacobian
-   a - the direction you want the derivative

  Output Parameter:
.   h - the scale computed

*/
static int MatSNESFDMFCompute_WP(MatSNESFDMFCtx ctx,Vec U,Vec a,Scalar *h)
{
  MatSNESFDMFWP      *hctx = (MatSNESFDMFWP *) ctx->hctx;
  MPI_Comm           comm = ctx->comm;
  double             normU, ovalues[2],values[2];
  double             norma = 1.0, normUfact = hctx->normUfact;
  int                ierr;

  if (hctx->computenorma && (hctx->computenormU || !ctx->ncurrenth)) {
    /*
       This algorithm requires 2 norms. Rather than
       use directly the VecNorm()routine (and thus have 
       two separate collective operations, we use the sequential routines
       and manually call MPI for the collective phase.
 
       We sparately log the VecNorm() stages to get 
       accurate profiling. 
    */

    PLogEventBegin(VEC_Norm,a,0,0,0);
    ierr = VecNorm_Seq(U,NORM_2,&normU); CHKERRQ(ierr);
    ierr = VecNorm_Seq(a,NORM_2,&norma); CHKERRQ(ierr);
    ovalues[0] = normU*normU;
    ovalues[1] = norma*norma;
    PLogEventBarrierBegin(VEC_NormBarrier,0,0,0,0,comm);
#if !defined(USE_PETSC_COMPLEX)
    ierr = MPI_Allreduce(ovalues,values,2,MPI_DOUBLE,MPI_SUM,comm );CHKERRQ(ierr);
#else
    ierr = MPI_Allreduce(ovalues,values,4,MPI_DOUBLE,MPI_SUM,comm );CHKERRQ(ierr);
#endif
    PLogEventBarrierEnd(VEC_NormBarrier,0,0,0,0,comm);
    normU = sqrt(values[0]);
    norma = sqrt(values[1]);
    PLogEventEnd(VEC_Norm,a,0,0,0);
    hctx->normUfact = normUfact = sqrt(1.0+normU);
  } else if (hctx->computenormU || !ctx->ncurrenth) {
    ierr = VecNorm(U,NORM_2,&normU); CHKERRQ(ierr);
    hctx->normUfact = normUfact = sqrt(1.0+normU);
  } else if (hctx->computenorma) {
    ierr = VecNorm(a,NORM_2,&norma); CHKERRQ(ierr);
  }

  *h = ctx->error_rel*normUfact/norma;
  PetscFunctionReturn(0);
} 

#undef __FUNC__  
#define __FUNC__ "MatSNESFDMFView_WP"
/*
   MatSNESFDMFView_WP - Prints information about this particular 
     method for computing h. Note that this does not print the general
     information about the matrix free, that is printed by the calling
     routine.

  Input Parameters:
+   ctx - the matrix free context
-   viewer - the PETSc viewer

*/   
static int MatSNESFDMFView_WP(MatSNESFDMFCtx ctx,Viewer viewer)
{
  FILE          *fd;
  ViewerType    vtype;
  MatSNESFDMFWP *hctx = (MatSNESFDMFWP *)ctx->hctx;
  int           ierr;

  PetscFunctionBegin;
  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
  if (PetscTypeCompare(vtype,ASCII_VIEWER)) {
    if (hctx->computenorma) PetscFPrintf(ctx->comm,fd,"    Computes normA\n");  
    else                    PetscFPrintf(ctx->comm,fd,"    Does not compute normA\n");  
    if (hctx->computenormU) PetscFPrintf(ctx->comm,fd,"    Computes normU\n");  
    else                    PetscFPrintf(ctx->comm,fd,"    Does not compute normU\n");  
  } else {
    SETERRQ(1,1,"Viewer type not supported for this object");
  }    
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSNESFDMFPrintHelp_WP"
/*
   MatSNESFDMFPrintHelp_WP - Prints a list of all the options 
      this particular method supports.

  Input Parameter:
.  ctx - the matrix free context

*/
static int MatSNESFDMFPrintHelp_WP(MatSNESFDMFCtx ctx)
{
  char*         p;
  int           ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetOptionsPrefix((PetscObject)ctx->snes,&p); CHKERRQ(ierr);
  if (!p) p = "";
  (*PetscHelpPrintf)(ctx->comm,"   -%ssnes_mf_compute_norma <true or false>\n",p);
  (*PetscHelpPrintf)(ctx->comm,"   -%ssnes_mf_compute_normu <true or false>\n",p);

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSNESFDMFSetFromOptions_WP"
/*
   MatSNESFDMFSetFromOptions_WP - Looks in the options database for 
     any options appropriate for this method

  Input Parameter:
.  ctx - the matrix free context

*/
static int MatSNESFDMFSetFromOptions_WP(MatSNESFDMFCtx ctx)
{
  int        flag, ierr;
  PetscTruth set;
  char       *p;

  PetscFunctionBegin;
  ierr = PetscObjectGetOptionsPrefix((PetscObject)ctx->snes,&p); CHKERRQ(ierr);
  ierr = OptionsGetLogical(p,"-snes_mf_compute_norma",&set,&flag);CHKERRQ(ierr);
  if (flag) {
    ierr = MatSNESFDMFWPSetComputeNormA(ctx->mat,set);CHKERRQ(ierr);
  }
  ierr = OptionsGetLogical(p,"-snes_mf_compute_normu",&set,&flag);CHKERRQ(ierr);
  if (flag) {
    ierr = MatSNESFDMFWPSetComputeNormU(ctx->mat,set);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSNESFDMFDestroy_WP"
/*
   MatSNESFDMFDestroy_WP - Frees the space allocated by 
       MatSNESFDMFCreate_WP(). 

  Input Parameter:
.  ctx - the matrix free context

   Notes: does not free the ctx, that is handled by the calling routine

*/
static int MatSNESFDMFDestroy_WP(MatSNESFDMFCtx ctx)
{
  PetscFunctionBegin;
  PetscFree(ctx->hctx);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "MatSNESFDMFWPSetComputeNormA_P"
int MatSNESFDMFWPSetComputeNormA_P(Mat mat,PetscTruth flag)
{
  MatSNESFDMFCtx ctx;
  MatSNESFDMFWP  *hctx;
  int            ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(ierr);
  if (!ctx) {
    SETERRQ(1,1,"MatSNESFDMFWPSetComputeNormA() attached to non-shell matrix");
  }
  hctx               = (MatSNESFDMFWP *) ctx->hctx;
  hctx->computenorma = flag;

 PetscFunctionReturn(0);
} 
EXTERN_C_END

#undef __FUNC__  
#define __FUNC__ "MatSNESFDMFWPSetComputeNormA"
/*@
    MatSNESFDMFWPSetComputeNormA - Sets whether it computes the ||a|| used by the WP
             PETSc routine for computing h. With GMRES since the ||a|| is always
             one, you can save communication by setting this to false.

  Input Parameters:
+   A - the matrix created with MatCreateSNESFDMF()
-   flag - PETSC_TRUE causes it to compute ||a||, PETSC_FALSE assumes it is 1.

  Level: advanced

  Notes:
   See the manual page for MatCreateSNESFDMF() for a complete description of the
   algorithm used to compute h.

.seealso: MatSNESFDMFSetFunctionError(), MatCreateSNESFDMF()

@*/
int MatSNESFDMFWPSetComputeNormA(Mat A,PetscTruth flag)
{
  int ierr, (*f)(Mat,PetscTruth);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)A,"MatSNESFDMFWPSetComputeNormA_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(A,flag);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "MatSNESFDMFWPSetComputeNormU_P"
int MatSNESFDMFWPSetComputeNormU_P(Mat mat,PetscTruth flag)
{
  MatSNESFDMFCtx ctx;
  MatSNESFDMFWP  *hctx;
  int            ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(mat,(void **)&ctx);CHKERRQ(ierr);
  if (!ctx) {
    SETERRQ(1,1,"MatSNESFDMFWPSetComputeNormU() attached to non-shell matrix");
  }
  hctx               = (MatSNESFDMFWP *) ctx->hctx;
  hctx->computenormU = flag;

 PetscFunctionReturn(0);
} 
EXTERN_C_END

#undef __FUNC__  
#define __FUNC__ "MatSNESFDMFWPSetComputeNormU"
/*@
    MatSNESFDMFWPSetComputeNormU - Sets whether it computes the ||U|| used by the WP
             PETSc routine for computing h. With any Krylov solver this need only 
             be computed during the first iteration and kept for later.

  Input Parameters:
+   A - the matrix created with MatCreateSNESFDMF()
-   flag - PETSC_TRUE causes it to compute ||U||, PETSC_FALSE uses the previous value

  Level: advanced

  Notes:
   See the manual page for MatCreateSNESFDMF() for a complete description of the
   algorithm used to compute h.

.seealso: MatSNESFDMFSetFunctionError(), MatCreateSNESFDMF()

@*/
int MatSNESFDMFWPSetComputeNormU(Mat A,PetscTruth flag)
{
  int ierr, (*f)(Mat,PetscTruth);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)A,"MatSNESFDMFWPSetComputeNormU_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(A,flag);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "MatSNESFDMFCreate_WP"
/*
     MatSNESFDMFCreate_WP - Standard PETSc code for 
   computing h with matrix-free finite differences.

   Input Parameter:
.  ctx - the matrix free context created by MatSNESFDMFCreate()

*/
int MatSNESFDMFCreate_WP(MatSNESFDMFCtx ctx)
{
  int           ierr;
  MatSNESFDMFWP *hctx;

  PetscFunctionBegin;

  /* allocate my own private data structure */
  hctx                     = (MatSNESFDMFWP *)PetscMalloc(sizeof(MatSNESFDMFWP));CHKPTRQ(hctx);
  ctx->hctx                = (void *) hctx;
  hctx->computenormU       = PETSC_TRUE;
  hctx->computenorma       = PETSC_TRUE;

  /* set the functions I am providing */
  ctx->ops->compute        = MatSNESFDMFCompute_WP;
  ctx->ops->destroy        = MatSNESFDMFDestroy_WP;
  ctx->ops->view           = MatSNESFDMFView_WP;  
  ctx->ops->printhelp      = MatSNESFDMFPrintHelp_WP;  
  ctx->ops->setfromoptions = MatSNESFDMFSetFromOptions_WP;  

  ierr = PetscObjectComposeFunction((PetscObject)ctx->mat,"MatSNESFDMFWPSetComputeNormA_C",
                            "MatSNESFDMFWPSetComputeNormA_P",
                            (void *) MatSNESFDMFWPSetComputeNormA_P);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)ctx->mat,"MatSNESFDMFWPSetComputeNormU_C",
                            "MatSNESFDMFWPSetComputeNormU_P",
                            (void *) MatSNESFDMFWPSetComputeNormU_P);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END



