#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: jacob.c,v 1.7 1997/10/16 00:48:48 curfman Exp curfman $";
#endif

#include "user.h"
extern int DAGetColoring(DA,ISColoring*,Mat*);

#undef __FUNC__  
#define __FUNC__ "UserSetJacobian"
/***************************************************************************/
/*
   UserSetJacobian - Forms Jacobian matrix context and sets Jacobian
   evaluation routine.  We also compute space for preallocation of
   matrix memory.

   Input Parameters:
   snes - SNES context
   app - application-defined context

   ----------------
    General Notes:
   ----------------
    Although it is not required to preallocate matrix memory, this step
    is crucial for efficient matrix assembly.  See the "Matrices" chapter
    of the PETSc users manual for detailed information on this topic.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
 */
int UserSetJacobian(SNES snes,Euler *app)
{
  MatType       mtype = MATMPIBAIJ;     /* matrix format */
  MPI_Comm      comm = app->comm;       /* comunicator */
  Mat           J;                      /* Jacobian matrix context */
  int           ldim = app->ldim;       /* local dimension of vectors and matrix */
  int           gdim = app->gdim;	/* global dimension of vectors and matrix */
  int           ndof = app->ndof;	/* DoF per node */
  int           ndof_block;             /* size of matrix blocks (ndof, except when
                                           experimenting with block size = 1) */
  int           ndof_euler;             /* DoF per node for Euler model */
  int           istart, iend;           /* range of locally owned matrix rows */
  int           *nnz_d = 0, *nnz_o = 0; /* arrays for preallocating matrix memory */
  int           wkdim;                  /* dimension of nnz_d and nnz_o */
  MatFDColoring fdc;                    /* finite difference coloring context */
  PetscTruth    mset; 
  int           jac_snes_fd = 0, ierr, flg;
  ISColoring    iscoloring;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Form Jacobian matrix data structure with preallocated space
         Two choices:
            - finite differencing via coloring
            - application-defined sparse finite differences
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = OptionsHasName(PETSC_NULL,"-jac_snes_fd",&jac_snes_fd); CHKERRQ(ierr);
  if (jac_snes_fd) {
     /* Set up coloring information needed for sparse finite difference
        approximation of the Jacobian */

    ierr = DAGetColoring(app->da,&iscoloring,&J); CHKERRQ(ierr);
    ierr = MatFDColoringCreate(J,iscoloring,&fdc); CHKERRQ(ierr); 
    app->fdcoloring = fdc;
    ierr = ISColoringDestroy(iscoloring); CHKERRQ(ierr);

    ierr = MatFDColoringSetFunction(fdc,
         (int (*)(void *,Vec,Vec,void *))ComputeFunctionNoWake,app); CHKERRQ(ierr);
    ierr = MatFDColoringSetParameters(fdc,app->eps_jac,PETSC_DEFAULT); CHKERRQ(ierr);
    ierr = MatFDColoringSetFromOptions(fdc); CHKERRQ(ierr); 

    ierr = ViewerPushFormat(VIEWER_STDOUT_WORLD,VIEWER_FORMAT_ASCII_INFO,PETSC_NULL);CHKERRQ(ierr);
    ierr = MatFDColoringView(fdc,VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
    ierr = ViewerPopFormat(VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

    PetscPrintf(comm,"Jacobian for preconditioner formed via PETSc FD approx\n"); 
  } 
  else {
    /* Use the alternative old application-defined approach */

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       First, compute amount of space for matrix preallocation, to enable
       fast matrix assembly without continual dynamic memory allocation.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    /* Determine matrix format, where we choose block AIJ as the default if
       no runtime option is specified */
    ierr = MatGetTypeFromOptions(comm,PETSC_NULL,&mtype,&mset); CHKERRQ(ierr);
    if (mset != PETSC_TRUE) {
      if (app->size == 1) mtype = MATSEQBAIJ;
      else                mtype = MATMPIBAIJ;
    }
    ierr = OptionsHasName(PETSC_NULL,"-mat_block_size_1",&flg); CHKERRQ(ierr);
    if (flg) ndof_block = 1;
    else     ndof_block = ndof;

    /* Row-based matrix formats */
    if (mtype == MATSEQAIJ || mtype == MATMPIAIJ || mtype == MATMPIROWBS) {
      wkdim = app->ldim;
    } else if (mtype == MATSEQBAIJ || mtype == MATMPIBAIJ) { /* block row formats */
      if (ndof_block == ndof) wkdim = app->lbkdim;
      else                    wkdim = app->ldim;
    }
    else SETERRQ(1,1,"Matrix format not currently supported.");

    /* Allocate work arrays */
    nnz_d = (int *)PetscMalloc(2*wkdim * sizeof(int)); CHKPTRQ(nnz_d);
    PetscMemzero(nnz_d,2*wkdim * sizeof(int));
    nnz_o = nnz_d + wkdim;

    /* Note that vector and matrix partitionings are the same (see note below) */
    ierr = VecGetOwnershipRange(app->X,&istart,&iend); CHKERRQ(ierr);

    /* We mimic the matrix assembly code to determine precise locations 
       of nonzero matrix entries */

    ndof_euler = 5;
    ierr = nzmat_(&mtype,&app->mmtype,&ndof_euler,&ndof_block,&istart,&iend,
                  app->is1,app->ltog,&app->nloc,&wkdim,nnz_d,nnz_o,
                  &app->fort_ao); CHKERRQ(ierr);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Form Jacobian matrix data structure
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    /*
       Note:  For the parallel case, vectors and matrices MUST be partitioned
       accordingly.  When using distributed arrays (DAs) to create vectors,
       the DAs determine the problem partitioning.  We must explicitly
       specify the local matrix dimensions upon its creation for compatibility
       with the vector distribution.  Thus, the generic MatCreate() routine
       is NOT sufficient when working with distributed arrays.
    */

    if (mtype == MATSEQAIJ) {
      /* Rough estimate of nonzeros per row is:  nd * ndof */
      /* ierr = MatCreateSeqAIJ(comm,gdim,gdim,nd*ndof,PETSC_NULL,&J); CHKERRQ(ierr); */
         ierr = MatCreateSeqAIJ(comm,gdim,gdim,PETSC_NULL,nnz_d,&J); CHKERRQ(ierr);
    } 
    else if (mtype == MATMPIAIJ) {
      ierr = MatCreateMPIAIJ(comm,ldim,ldim,gdim,
                             gdim,PETSC_NULL,nnz_d,PETSC_NULL,nnz_o,&J); CHKERRQ(ierr);
    } 
    else if (mtype == MATMPIROWBS) {
      ierr = MatCreateMPIRowbs(comm,ldim,gdim,PETSC_NULL,nnz_d,
                               PETSC_NULL,&J); CHKERRQ(ierr);
    }
    else if (mtype == MATSEQBAIJ) {
      /* Rough estimate of block nonzeros per row is:  # of diagonals, nd */
      /* ierr = MatCreateSeqBAIJ(comm,ndof_block,gdim,gdim,nd,PETSC_NULL,&J); CHKERRQ(ierr); */
      ierr = MatCreateSeqBAIJ(comm,ndof_block,gdim,gdim,PETSC_NULL,nnz_d,&J); CHKERRQ(ierr);
    } 
    else if (mtype == MATMPIBAIJ) {
      ierr = MatCreateMPIBAIJ(comm,ndof_block,ldim,ldim,
             gdim,gdim,PETSC_NULL,nnz_d,PETSC_NULL,nnz_o,&J); CHKERRQ(ierr);
    } else {
      SETERRQ(1,1,"Matrix format not currently supported.");
    }
    if (nnz_d) PetscFree(nnz_d);
    PetscPrintf(comm,"Jacobian for preconditioner formed via application code FD approx\n"); 
  }
  app->J = J;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        Set data structures and routine for Jacobian evaluation 
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = OptionsHasName(PETSC_NULL,"-matrix_free",&app->matrix_free); CHKERRQ(ierr);
  if (!app->matrix_free) {
    /* Use explicit (approx) Jacobian to define Newton system and preconditioner */
    if (jac_snes_fd) {
      ierr = SNESSetJacobian(snes,J,J,ComputeJacobianFDColoring,app); CHKERRQ(ierr);
    } else {
      ierr = SNESSetJacobian(snes,J,J,ComputeJacobianFDBasic,app); CHKERRQ(ierr);
    }
    PetscPrintf(comm,"Linear system matrix = preconditioner matrix (not matrix-free);\n"); 
  } else {
    /* Use matrix-free Jacobian to define Newton system; use finite difference
       approximation of Jacobian for preconditioner */
    if (app->bctype != IMPLICIT) SETERRQ(1,1,"Matrix-free method requires implicit BCs!");

    if (app->mmtype == MMFP) {
      ierr = SNESDefaultMatrixFreeMatCreate(snes,app->X,&app->Jmf); CHKERRQ(ierr); 
    } else if (app->mmtype == MMEULER) {
      ierr = UserMatrixFreeMatCreate(snes,app,app->X,&app->Jmf); CHKERRQ(ierr); 
    } else {
      SETERRQ(1,0,"Need a new matrix-free option\n");
    }

    if (jac_snes_fd) {
     ierr = SNESSetJacobian(snes,app->Jmf,J,ComputeJacobianFDColoring,app); CHKERRQ(ierr);
    } else {
     ierr = SNESSetJacobian(snes,app->Jmf,J,ComputeJacobianFDBasic,app); CHKERRQ(ierr);
    }

    /* Set matrix-free parameters and view matrix context */
    ierr = OptionsGetDouble(PETSC_NULL,"-snes_mf_err",&app->eps_mf_default,&flg); CHKERRQ(ierr);
    ierr = UserSetMatrixFreeParameters(snes,app->eps_mf_default,PETSC_DEFAULT); CHKERRQ(ierr);
    ierr = ViewerPushFormat(VIEWER_STDOUT_WORLD,VIEWER_FORMAT_ASCII_INFO,0); CHKERRQ(ierr);
    PetscPrintf(comm,"Using matrix-free KSP method: linear system matrix:\n");
    ierr = MatView(app->Jmf,VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
    ierr = ViewerPopFormat(VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
  }
  return 0;
}


extern int MatView_Hybrid(Mat,Viewer);

#undef __FUNC__
#define __FUNC__ "ComputeJacobianFDColoring"
/***************************************************************************/
/* 
   ComputeJacobianFDColoring - Computes Jacobian matrix.  The user sets this routine
   by calling SNESSetJacobian().

   Input Parameters:
   X     - input vector
   jac   - Jacobian matrix
   pjac  - preconditioner matrix (same as Jacobian matrix except when
	   we use matrix-free Newton-Krylov methods)
   flag  - flag indicating information about the preconditioner matrix
	   structure.  See SLESSetOperators() for important information 
	   about setting this flag.
   ptr   - optional user-defined context, as set by SNESSetJacobian()

   Output Parameters:
   pjac  - fully assembled preconditioner matrix
   flag  - flag indicating information about the preconditioner matrix structure

   This routine supports only the implicit mode of handling boundary
   conditions.    Another version of code also supports explicit boundary
   conditions; we omit this capability here to reduce code complexity.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
 */
int ComputeJacobianFDColoring(SNES snes,Vec X,Mat *jac,Mat *pjac,MatStructure *flag,void *ptr)
{
  Euler         *app = (Euler *)ptr;
  int           iter, ierr;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set some options; do some preliminary work
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  if (app->bctype != IMPLICIT) SETERRQ(1,0,"This version supports only implicit BCs!");
  ierr = SNESGetIterationNumber(snes,&iter); CHKERRQ(ierr);

  /* We want to verify that our preallocation was 100% correct by not allowing
     any additional nonzeros into the matrix. */
  /* ierr = MatSetOption(*pjac,MAT_NEW_NONZERO_ALLOCATION_ERROR); CHKERRQ(ierr); */

  /* For better efficiency, we hold the Jacobian matrix fixed over several
     nonlinear iterations.  The flag SAME_PRECONDITIONER indicates that in
     this case the current preconditioner should be retained. */
  if (iter > 1) {
    if (app->use_jratio) {
      if (iter != (app->bcswitch)) {  /* force Jacobian eval at iteration bcswitch, since BCs change there */
        if (app->fnorm_last_jac/app->fnorm_last < app->jratio) {
          if (iter - app->iter_last_jac < app->jfreq) {
            *flag = SAME_PRECONDITIONER;
            return 0;
          } 
        } 
      }
    } else {
      /* Form Jacobian every few nonlinear iterations (as set by -jfreq option) */
      if ((iter-(app->bcswitch))%app->jfreq) {
        *flag = SAME_PRECONDITIONER;
        return 0;
      }
    }
  }

  /* Convert vector */
  ierr = UnpackWork(app,app->da,app->xx,app->localX,X); CHKERRQ(ierr);

  /* Indicate that we're now using an unfactored matrix.  This is needed only
     when using in-place ILU(0) preconditioning to allow repeated assembly of
     the matrix. */
  ierr = MatSetUnfactored(*pjac); CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Form Jacobian matrix
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */


  /* Do finite differencing with coloring */
  ierr = MatFDColoringApply(*pjac,app->fdcoloring,X,flag,snes); CHKERRQ(ierr);

  /* Fix points on edges of the 3D domain, where diagonal of Jacobian is 1.
     edges are:  (k=1, j=1, i=1 to ni1;  k=nk1, j=1, i=1 to ni1; etc.) */

  /*  if ((app->mmtype != MMFP) && */
    if ((app->problem == 1 || app->problem == 2 || app->problem == 3 || app->problem == 5)) {
    ierr = FixJacobianEdges(app,*pjac); CHKERRQ(ierr);
  }

  /* Finish the matrix assembly process.  For the Euler code, the matrix
     assembly is done completely locally, so no message-pasing is performed
     during these phases. */
  ierr = MatAssemblyBegin(*pjac,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*pjac,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  /* Fix points on edges of the 3D domain, where diagonal of Jacobian is 1.
     edges are:  (k=1, j=1, i=1 to ni1;  k=nk1, j=1, i=1 to ni1; etc.) */
  { Vec diagv; int i, rstart, rend; Scalar *diagv_a, one = 1.0;
  ierr = VecDuplicate(X,&diagv); CHKERRQ(ierr);
  ierr = MatGetDiagonal(*pjac,diagv); CHKERRQ(ierr);
  ierr = VecGetArray(diagv,&diagv_a); CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(*pjac,&rstart,&rend); CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) {
    if (!diagv_a[i-rstart]) {
      ierr = MatSetValues(*pjac,1,&i,1,&i,&one,INSERT_VALUES); CHKERRQ(ierr);
    }
  }
  ierr = VecRestoreArray(diagv,&diagv_a); CHKERRQ(ierr);
  ierr = VecDestroy(diagv); CHKERRQ(ierr);
  }

  /* Finish the matrix assembly process.  For the Euler code, the matrix
     assembly is done completely locally, so no message-pasing is performed
     during these phases. */
  ierr = MatAssemblyBegin(*pjac,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*pjac,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  /* Indicate that the preconditioner matrix has the same nonzero
     structure each time it is formed */
  /*  *flag = SAME_NONZERO_PATTERN; */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Output - primarily for debugging
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  if (!app->no_output) {

    /* View matrix (for debugging only) */
    if (app->print_vecs) {
      char filename[64]; Viewer view; MatType mtype;
      sprintf(filename,"mat.%d.out",iter);
      ierr = ViewerFileOpenASCII(app->comm,filename,&view); CHKERRQ(ierr);
      ierr = ViewerSetFormat(view,VIEWER_FORMAT_ASCII_COMMON,PETSC_NULL); CHKERRQ(ierr);
      ierr = MatGetType(*pjac,&mtype,PETSC_NULL); CHKERRQ(ierr);

      /* These routines are being superceded by GVec capabilities; these are in the
         file regmpimat.c for now */
      /*
      if (mtype == MATMPIAIJ)       {ierr = MatViewDFVec_MPIAIJ(*pjac,X,view); CHKERRQ(ierr);}
      else if (mtype == MATMPIBAIJ) {ierr = MatViewDFVec_MPIBAIJ(*pjac,X,view); CHKERRQ(ierr);}
      else                          {ierr = MatView(*pjac,view); CHKERRQ(ierr);} */

      if ((app->mmtype == MMHYBRID_E || app->mmtype == MMHYBRID_F || app->mmtype == MMHYBRID_EF1)
        && mtype == MATSEQAIJ) {
        ierr = MatView_Hybrid(*pjac,view); CHKERRQ(ierr);
      } else {
        ierr = MatView(*pjac,view); CHKERRQ(ierr);
      }
      ierr = ViewerDestroy(view); CHKERRQ(ierr);
      /* PetscFinalize(); exit(0); */
    }
  }

  return 0;
}

#undef __FUNC__
#define __FUNC__ "ComputeJacobianFDBasic"
/***************************************************************************/
/* 
   ComputeJacobianFDBasic - Computes Jacobian matrix.  The user sets this routine
   by calling SNESSetJacobian().

   Input Parameters:
   X     - input vector
   jac   - Jacobian matrix
   pjac  - preconditioner matrix (same as Jacobian matrix except when
	   we use matrix-free Newton-Krylov methods)
   flag  - flag indicating information about the preconditioner matrix
	   structure.  See SLESSetOperators() for important information 
	   about setting this flag.
   ptr   - optional user-defined context, as set by SNESSetJacobian()

   Output Parameters:
   pjac  - fully assembled preconditioner matrix
   flag  - flag indicating information about the preconditioner matrix structure

  -----------------------
   Notes for Euler code:
  -----------------------
   This routine supports two appropaches for matrix assembly:
     (1) Assembling the Jacobian directly into the PETSc matrix data 
	 structure (the default approach -- more efficient)
     (2) Forming the matrix in the original Eagle data structures and
	 converting these later to asssemble a PETSc matrix.  This approach
	 is expensive in terms of both memory and time and is retained only
	 as a demonstration of how to quickly revise an existing code that
	 uses the Eagle format.

   This routine supports only the implicit mode of handling boundary
   conditions.    Another version of code also supports explicit boundary
   conditions; we omit this capability here to reduce code complexity.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
 */
int ComputeJacobianFDBasic(SNES snes,Vec X,Mat *jac,Mat *pjac,MatStructure *flag,void *ptr)
{
  Euler  *app = (Euler *)ptr;
  int    iter;                   /* nonlinear solver iteration number */
  int    fortmat, flg, ierr, i, rstart, rend;
  Vec    fvec;
  Scalar *fvec_array, one = 1.0;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set some options; do some preliminary work
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  if (app->bctype != IMPLICIT) SETERRQ(1,0,"This version supports only implicit BCs!");
  ierr = SNESGetIterationNumber(snes,&iter); CHKERRQ(ierr);

  /* We want to verify that our preallocation was 100% correct by not allowing
     any additional nonzeros into the matrix. */
  ierr = MatSetOption(*pjac,MAT_NEW_NONZERO_ALLOCATION_ERROR); CHKERRQ(ierr);

  /* For better efficiency, we hold the Jacobian matrix fixed over several
     nonlinear iterations.  The flag SAME_PRECONDITIONER indicates that in
     this case the current preconditioner should be retained. */
  if (iter > 1) {
    if (app->use_jratio) {
      if (iter != (app->bcswitch)) {  /* force Jacobian eval at iteration bcswitch, since BCs change there */
        if (app->fnorm_last_jac/app->fnorm_last < app->jratio) {
          if (iter - app->iter_last_jac < app->jfreq) {
            *flag = SAME_PRECONDITIONER;
            return 0;
          } 
        } 
      }
    } else {
      /* Form Jacobian every few nonlinear iterations (as set by -jfreq option) */
      if ((iter-(app->bcswitch))%app->jfreq) {
        *flag = SAME_PRECONDITIONER;
        return 0;
      }
    }
  }

  if (iter == app->bcswitch) {
    ierr = OptionsHasName(PETSC_NULL,"-switch_matrix_free",&flg); CHKERRQ(ierr);
    if (flg) {
      /* Use matrix-free Jacobian to define Newton system; use finite difference
         approximation of Jacobian for preconditioner */
      if (app->bctype != IMPLICIT) SETERRQ(1,1,"Matrix-free method requires implicit BCs!");
      ierr = UserMatrixFreeMatCreate(snes,app,app->X,&app->Jmf); CHKERRQ(ierr); 
      ierr = SNESSetJacobian(snes,app->Jmf,*pjac,ComputeJacobianFDBasic,app); CHKERRQ(ierr);

      /* Set matrix-free parameters and view matrix context */
      ierr = OptionsGetDouble(PETSC_NULL,"-snes_mf_err",&app->eps_mf_default,&flg); CHKERRQ(ierr);
      ierr = UserSetMatrixFreeParameters(snes,app->eps_mf_default,PETSC_DEFAULT); CHKERRQ(ierr);
      if (!app->no_output) {
        ierr = ViewerPushFormat(VIEWER_STDOUT_WORLD,VIEWER_FORMAT_ASCII_INFO,0); CHKERRQ(ierr);
        PetscPrintf(app->comm,"Using matrix-free KSP method: linear system matrix:\n");
        ierr = MatView(app->Jmf,VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
        ierr = ViewerPopFormat(VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
      }
      jac = &app->Jmf;
    }
  }

  /* Convert vector */
  ierr = UnpackWork(app,app->da,app->xx,app->localX,X); CHKERRQ(ierr);

  /* Form Fortran matrix object */
  ierr = PetscCObjectToFortranObject(*pjac,&fortmat); CHKERRQ(ierr);

  /* Indicate that we're now using an unfactored matrix.  This is needed only
     when using in-place ILU(0) preconditioning to allow repeated assembly of
     the matrix. */
  ierr = MatSetUnfactored(*pjac); CHKERRQ(ierr);

  /* ierr = ComputeFunctionNoWake(snes,X,app->diagv,ptr); CHKERRQ(ierr);
  ierr = VecGetArray(app->diagv,&fvec_array); CHKERRQ(ierr); */

  ierr = SNESGetFunction(snes,&fvec); CHKERRQ(ierr);
  ierr = VecGetArray(fvec,&fvec_array); CHKERRQ(ierr); 

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Form Jacobian matrix
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* First, initialize the diagonal to 1.   These values will be overwritten
     everywhere EXCEPT for the edges of the 3D problem domain, where the
     edges are:  (k=1, j=1, i=1 to ni1;  k=nk1, j=1, i=1 to ni1; etc.)
     We need to do this the first time the Jacobian is assembled.  We
     could alternatively do this just for the edges and could use
     MatZeroRows(). */

  ierr = MatSetOption(*pjac,MAT_COLUMNS_SORTED); CHKERRQ(ierr);
  ierr = MatSetOption(*pjac,MAT_ROWS_SORTED); CHKERRQ(ierr);
  if (iter == 1) {
    ierr = MatGetOwnershipRange(*pjac,&rstart,&rend); CHKERRQ(ierr);
    for (i=rstart; i<rend; i++) {
      ierr = MatSetValues(*pjac,1,&i,1,&i,&one,INSERT_VALUES); CHKERRQ(ierr);
    }
  }

  if (app->mmtype != MMFP && app->mmtype != MMHYBRID_F) {
    /* As long as we're not doing just the full potential model, we must
       compute the Euler components of the Jacobian */

#if defined(ACTIVATE_OLD_ASSEMBLY)
    if (app->mat_assemble_direct) {
#endif
      /* Either assemble the matrix directly (the more efficient route) ... */
      /* We must zero the diagonal block here, since this is not done within jformdt2 */
      PetscMemzero(app->diag,app->diag_len);
      ierr = jformdt2_(&app->eps_jac,&app->eps_jac_inv,app->ltog,&app->nloc,&fortmat,app->is1,
             app->b1bc,app->b2bc,app->b3bc,app->b2bc_tmp,app->diag,
	     app->dt,app->xx,app->p,app->xx_bc,app->p_bc,
	     app->br,app->bl,app->be,app->sadai,app->sadaj,app->sadak,
	     app->aix,app->ajx,app->akx,app->aiy,app->ajy,app->aky,
	     app->aiz,app->ajz,app->akz,app->f1,app->g1,app->h1,
	     app->sp,app->sm,app->sp1,app->sp2,app->sm1,app->sm2,fvec_array,&app->fort_ao); CHKERRQ(ierr);

#if defined(ACTIVATE_OLD_ASSEMBLY)
    /* Or store the matrix in the intermediate Eagle format for later conversion ... */
    } else {
      MatType type; /* matrix format */
      ierr = jformdt_(&app->eps_jac,&app->eps_jac_inv,
	     app->b1,app->b2,app->b3,app->b4,app->b5,app->b6,
	     app->b1bc,app->b2bc,app->b3bc,app->b2bc_tmp,
	     app->diag,app->dt,app->xx,app->p,
	     app->r_bc,app->ru_bc,app->rv_bc,app->rw_bc,app->e_bc,app->p_bc,
	     app->br,app->bl,app->be,app->sadai,app->sadaj,app->sadak,
	     app->aix,app->ajx,app->akx,app->aiy,app->ajy,app->aky,
	     app->aiz,app->ajz,app->akz,app->f1,app->g1,app->h1,
	     app->sp,app->sm,app->sp1,app->sp2,app->sm1,app->sm2,fvec_array, &app->fort_ao); CHKERRQ(ierr);
      /* Convert Jacobian from Eagle format */
      if (!app->no_output) PetscPrintf(app->comm,"Building PETSc matrix ...\n");
      ierr = MatGetType(*pjac,&type,PETSC_NULL); CHKERRQ(ierr);
      ierr = buildmat_(&fortmat,&app->sctype,app->is1,
	     app->b1,app->b2,app->b3,app->b4,app->b5,app->b6,app->diag,
             app->dt,app->ltog,&app->nloc,
             app->b1bc,app->b2bc,app->b3bc,app->b2bc_tmp,&app->fort_ao); CHKERRQ(ierr);
    }
#endif
  }
  if (app->mmtype != MMEULER) {
    /* DAVID */
    /* PetscPrintf(app->comm,"Dummy FP: Setting all full potential Jacobian diagonal components to 1\n"); */
  }

  /* Finish the matrix assembly process.  For the Euler code, the matrix
     assembly is done completely locally, so no message-pasing is performed
     during these phases. */
  ierr = MatAssemblyBegin(*pjac,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*pjac,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  /* Indicate that the preconditioner matrix has the same nonzero
     structure each time it is formed */
  *flag = SAME_NONZERO_PATTERN;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Output - primarily for debugging
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  if (!app->no_output) {

    /* View matrix (for debugging only) */
    if (app->print_vecs) {
      char filename[64]; Viewer view; MatType mtype;
      sprintf(filename,"mat.%d.out",iter);
      ierr = ViewerFileOpenASCII(app->comm,filename,&view); CHKERRQ(ierr);
      ierr = ViewerSetFormat(view,VIEWER_FORMAT_ASCII_COMMON,PETSC_NULL); CHKERRQ(ierr);
      ierr = MatGetType(*pjac,&mtype,PETSC_NULL); CHKERRQ(ierr);

      /* These routines are being superceded by GVec capabilities; these are in the
         file regmpimat.c for now */
      /*
      if (mtype == MATMPIAIJ)       {ierr = MatViewDFVec_MPIAIJ(*pjac,X,view); CHKERRQ(ierr);}
      else if (mtype == MATMPIBAIJ) {ierr = MatViewDFVec_MPIBAIJ(*pjac,X,view); CHKERRQ(ierr);}
      else                          {ierr = MatView(*pjac,view); CHKERRQ(ierr);} */

      if ((app->mmtype == MMHYBRID_E || app->mmtype == MMHYBRID_F || app->mmtype == MMHYBRID_EF1)
        && mtype == MATSEQAIJ) {
        ierr = MatView_Hybrid(*pjac,view); CHKERRQ(ierr);
      } else {
        ierr = MatView(*pjac,view); CHKERRQ(ierr);
      }
      ierr = ViewerDestroy(view); CHKERRQ(ierr);
      /* PetscFinalize(); exit(0); */
    }
  
    /* Dump Jacobian and residual in binary format to file euler.dat 
       (for use in separate experiments with linear system solves) */
    ierr = OptionsHasName(PETSC_NULL,"-mat_dump",&flg); CHKERRQ(ierr);
    if (flg && app->cfl_switch <= app->cfl) {
      Viewer viewer;
      PetscPrintf(app->comm,"writing matrix in binary to euler.dat ...\n"); 
      ierr = ViewerFileOpenBinary(app->comm,"euler.dat",BINARY_CREATE,&viewer); 
  	   CHKERRQ(ierr);
      ierr = MatView(*pjac,viewer); CHKERRQ(ierr);
  
      ierr = ComputeFunctionBasic(snes,X,app->F,ptr); CHKERRQ(ierr);
      PetscPrintf(app->comm,"writing vector in binary to euler.dat ...\n"); 
      ierr = VecView(app->F,viewer); CHKERRQ(ierr);
      ierr = ViewerDestroy(viewer); CHKERRQ(ierr);
      PetscFinalize(); exit(0);
    }
  
    /* Check matrix-vector products, run with -matrix_free and -debug option */
    if (app->matrix_free && app->print_debug) {
      int    loc, m, j ,k, ijkx, jkx, ijkxl, jkxl, *ltog, dc[6];
      Vec    yy1, yy2, xx1;
      Viewer view2;
      Scalar *yy1a, *yy2a, di, diff, md[5];
  
      ierr = DAGetGlobalIndices(app->da,&loc,&ltog); CHKERRQ(ierr);
      ierr = VecDuplicate(X,&yy1); CHKERRQ(ierr);
      ierr = VecDuplicate(X,&xx1); CHKERRQ(ierr);
      ierr = VecDuplicate(X,&yy2); CHKERRQ(ierr);
      for (k=app->zs; k<app->ze; k++) {
        for (j=app->ys; j<app->ye; j++) {
  	jkx  = j*app->mx + k*app->mx*app->my;
  	jkxl = (j-app->gys)*app->gxm + (k-app->gzs)*app->gxm*app->gym;
  	for (i=app->xs; i<app->xe; i++) {
  	  ijkx  = (jkx + i)*app->ndof;
  	  ijkxl = (jkxl + i-app->gxs)*app->ndof;
  	  for (m=0;m<app->ndof;m++) {
  	    di = one*ijkx;
  	    loc = ltog[ijkxl];
  	    ierr = VecSetValues(xx1,1,&loc,&di,INSERT_VALUES); CHKERRQ(ierr);
  	 printf("[%d] k=%d, j=%d, i=%d, ijkx=%d, ijkxl=%d\n",app->rank,k,j,i,ijkx,ijkxl);
  	    ijkx++; ijkxl++;
  	  }
  	}
        }
      }
      ierr = VecAssemblyBegin(xx1); CHKERRQ(ierr);
      ierr = VecAssemblyEnd(xx1); CHKERRQ(ierr);
      ierr = ViewerFileOpenASCII(app->comm,"xx1.out",&view2); CHKERRQ(ierr);
      ierr = ViewerSetFormat(view2,VIEWER_FORMAT_ASCII_COMMON,PETSC_NULL); CHKERRQ(ierr);
      ierr = DFVecView(xx1,view2); CHKERRQ(ierr);
      ierr = ViewerDestroy(view2); CHKERRQ(ierr);
  
      ierr = MatMult(*pjac,xx1,yy1); CHKERRQ(ierr);
      ierr = ViewerFileOpenASCII(app->comm,"v1.out",&view2); CHKERRQ(ierr);
      ierr = ViewerSetFormat(view2,VIEWER_FORMAT_ASCII_COMMON,PETSC_NULL); CHKERRQ(ierr);
      ierr = DFVecView(yy1,view2); CHKERRQ(ierr);
      ierr = ViewerDestroy(view2); CHKERRQ(ierr);
  
      ierr = MatMult(*jac,xx1,yy2); CHKERRQ(ierr);
      ierr = ViewerFileOpenASCII(app->comm,"v2.out",&view2); CHKERRQ(ierr);
      ierr = ViewerSetFormat(view2,VIEWER_FORMAT_ASCII_COMMON,PETSC_NULL); CHKERRQ(ierr);
      ierr = DFVecView(yy2,view2); CHKERRQ(ierr);
      ierr = ViewerDestroy(view2); CHKERRQ(ierr);
  
      ierr = VecGetArray(yy1,&yy1a); CHKERRQ(ierr);
      ierr = VecGetArray(yy2,&yy2a); CHKERRQ(ierr);
  
      for (m=0;m<app->ndof;m++) {
        dc[m] = 0;
        md[m] = 0;
        for (k=app->zs; k<app->ze; k++) {
  	for (j=app->ys; j<app->ye; j++) {
  	  jkx = (j-app->ys)*app->xm + (k-app->zs)*app->xm*app->ym;
  	  for (i=app->xs; i<app->xe; i++) {
  	    ijkx = (jkx + i-app->xs)*app->ndof + m;
  	    diff = (PetscAbsScalar(yy1a[ijkx])-PetscAbsScalar(yy2a[ijkx])) /
  		  PetscAbsScalar(yy1a[ijkx]);
  	    if (diff > 0.1) {
  	      printf("k=%d, j=%d, i=%d, m=%d, ijkx=%d,     diff=%6.3e       yy1=%6.3e,   yy2=%6.3e\n",
  		      k,j,i,m,ijkx,diff,yy1a[ijkx],yy2a[ijkx]);
  	      if (diff > md[m]) md[m] = diff;
  	      dc[m]++;
  	    }
  	  }
  	}
        }
      }
      printf("[%d] maxdiff = %g, %g, %g, %g, %g\n\
      dcount = %d, %d, %d, %d, %d\n",
  	   app->rank,md[0],md[1],md[2],md[3],md[4],dc[0],dc[1],dc[2],dc[3],dc[4]);
  
      ierr = VecDestroy(xx1); CHKERRQ(ierr);
      ierr = VecDestroy(yy1); CHKERRQ(ierr);
      ierr = VecDestroy(yy2); CHKERRQ(ierr);
  
      PetscFinalize(); exit(0);
    }
  
    PetscPrintf(app->comm,
       "Done building PETSc matrix: last Jac iter = %d, fnorm ratio = %g, tol = %g\n",
       app->iter_last_jac,app->fnorm_last_jac/app->fnorm_last,app->jratio);
  }
  app->iter_last_jac  = iter;
  app->fnorm_last_jac = app->fnorm_last;

  return 0;
}

/* --------------------------------------------------------------- */
#include "src/snes/snesimpl.h"

typedef struct {
  int complete_print;
} SNES_Test;

/*
     SNESSolve_Test - Tests whether a hand computed Jacobian 
     matches one compute via finite differences.
*/
#undef __FUNC__  
#define __FUNC__ "SNESSolve_Test"
int SNESSolve_Test(SNES snes,int *its)
{
  Mat          A = snes->jacobian,B;
  Vec          x = snes->vec_sol;
  int          ierr,i;
  MatStructure flg;
  Scalar       mone = -1.0,one = 1.0;
  double       norm,gnorm;
  SNES_Test    *neP = (SNES_Test*) snes->data;

  *its = 0;

  if (A != snes->jacobian_pre) 
    SETERRQ(1,0,"Cannot test with alternative preconditioner");

  PetscPrintf(snes->comm,"Testing hand-coded Jacobian, if the ratio is\n");
  PetscPrintf(snes->comm,"O(1.e-8), the hand-coded Jacobian is probably correct.\n");
  if (!neP->complete_print) {
    PetscPrintf(snes->comm,"Run with -snes_test_display to show difference\n");
    PetscPrintf(snes->comm,"of hand-coded and finite difference Jacobian.\n");
  }

  for ( i=0; i<3; i++ ) {
    if (i == 1) {ierr = VecSet(&mone,x); CHKERRQ(ierr);}
    else if (i == 2) {ierr = VecSet(&one,x); CHKERRQ(ierr);}
 
    /* compute both versions of Jacobian */
    ierr = SNESComputeJacobian(snes,x,&A,&A,&flg);CHKERRQ(ierr);
    if (i == 0) {ierr = MatConvert(A,MATSAME,&B); CHKERRQ(ierr);}
    ierr = ComputeJacobianFDBasic(snes,x,&B,&B,&flg,snes->funP);CHKERRQ(ierr);
    /*    ierr = ComputeJacobianFDColoring(snes,x,&B,&B,&flg,snes->funP);CHKERRQ(ierr); */
    /* if (neP->complete_print) {
      PetscPrintf(snes->comm,"Finite difference Jacobian\n");
      ierr = MatView(B,VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
    } */
    /* compare */
    ierr = MatAXPY(&mone,A,B); CHKERRQ(ierr);
    ierr = MatNorm(B,NORM_FROBENIUS,&norm); CHKERRQ(ierr);
    ierr = MatNorm(A,NORM_FROBENIUS,&gnorm); CHKERRQ(ierr);
    PetscPrintf(snes->comm,"Norm of matrix ratio %g difference %g\n",norm/gnorm,norm);
    if (neP->complete_print) {
      PetscPrintf(snes->comm,"Difference between Jacobians\n");
      ierr = ViewerSetFormat(VIEWER_STDOUT_WORLD,VIEWER_FORMAT_ASCII_COMMON,PETSC_NULL); CHKERRA(ierr);
      ierr = MatView(B,VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
      ierr = ViewerPopFormat(VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
    }
  }
  ierr = MatDestroy(B); CHKERRQ(ierr);
  return 0;
}
/* ------------------------------------------------------------ */
#undef __FUNC__  
#define __FUNC__ "SNESDestroy_Test"
int SNESDestroy_Test(PetscObject obj)
{
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "SNESPrintHelp_Test"
static int SNESPrintHelp_Test(SNES snes,char *p)
{
  PetscPrintf(snes->comm,"Test code to compute Jacobian\n");
  PetscPrintf(snes->comm,"-snes_test_display - display difference between\n");
  return 0;
}

#undef __FUNC__  
#define __FUNC__ "SNESSetFromOptions_Test"
static int SNESSetFromOptions_Test(SNES snes)
{
  SNES_Test *ls = (SNES_Test *)snes->data;
  int       ierr,flg;

  ierr = OptionsHasName(PETSC_NULL,"-snes_test_display",&flg); CHKERRQ(ierr);
  if (flg) {
    ls->complete_print = 1;
  }
  return 0;
}

/* ------------------------------------------------------------ */
#undef __FUNC__  
#define __FUNC__ "SNESCreate_Test"
int SNESCreate_Test(SNES  snes )
{
  SNES_Test *neP;

  if (snes->method_class != SNES_NONLINEAR_EQUATIONS) SETERRQ(1,0,"SNES_NONLINEAR_EQUATIONS only");
  snes->type		= SNES_EQ_TEST;
  snes->setup		= 0;
  snes->solve		= SNESSolve_Test;
  snes->destroy		= SNESDestroy_Test;
  snes->converged	= SNESConverged_EQ_LS;
  snes->printhelp       = SNESPrintHelp_Test;
  snes->setfromoptions  = SNESSetFromOptions_Test;

  neP			= PetscNew(SNES_Test);   CHKPTRQ(neP);
  PLogObjectMemory(snes,sizeof(SNES_Test));
  snes->data    	= (void *) neP;
  neP->complete_print   = 0;
  return 0;
}
