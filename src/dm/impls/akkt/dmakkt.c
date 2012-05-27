#include <../src/ksp/pc/impls/gamg/gamg.h>        /*I "petscpc.h" I*/
#include <../src/dm/impls/akkt/dmakkt.h>          /*I "petscdmakkt.h" I*/
#include <petsc-private/dmimpl.h>                 /*I "petscdm.h" I*/

#undef  __FUNCT__
#define __FUNCT__ "DMAKKTSetDM"
PetscErrorCode DMAKKTSetDM(DM dm, DM ddm) {
  PetscBool iskkt;
  DM_AKKT *kkt = (DM_AKKT*)(dm->data);
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID,1);
  PetscValidHeaderSpecific(ddm, DM_CLASSID,2);
  ierr = PetscObjectTypeCompare((PetscObject)dm, DMAKKT, &iskkt); CHKERRQ(ierr);
  if(!iskkt) SETERRQ(((PetscObject)dm)->comm, PETSC_ERR_ARG_WRONG, "DM not of type DMAKKT");
  ierr = PetscObjectReference((PetscObject)ddm); CHKERRQ(ierr);
  if(kkt->dm) {
    ierr = DMDestroy(&(kkt->dm)); CHKERRQ(ierr);
  }
  kkt->dm = ddm;
  ierr = DMDestroy(&(kkt->cdm));  CHKERRQ(ierr);
  ierr = MatDestroy(&(kkt->Pfc)); CHKERRQ(ierr);
  dm->setupcalled = PETSC_FALSE;
  
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "DMAKKTGetDM"
PetscErrorCode DMAKKTGetDM(DM dm, DM *ddm) {
  PetscBool iskkt;
  DM_AKKT *kkt = (DM_AKKT*)(dm->data);
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)dm, DMAKKT, &iskkt); CHKERRQ(ierr);
  if(!iskkt) SETERRQ(((PetscObject)dm)->comm, PETSC_ERR_ARG_WRONG, "DM not of type DMAKKT");
  if(ddm) {
    ierr = PetscObjectReference((PetscObject)(kkt->dm)); CHKERRQ(ierr);
    *ddm = kkt->dm;
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "DMAKKTSetMatrix"
PetscErrorCode DMAKKTSetMatrix(DM dm, Mat Aff) {
  PetscBool iskkt;
  PetscErrorCode ierr;
  DM_AKKT *kkt = (DM_AKKT*)(dm->data);
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID,1);
  PetscValidHeaderSpecific(Aff, MAT_CLASSID,2);
  ierr = PetscObjectTypeCompare((PetscObject)dm, DMAKKT, &iskkt); CHKERRQ(ierr);
  if(!iskkt) SETERRQ(((PetscObject)dm)->comm, PETSC_ERR_ARG_WRONG, "DM not of type DMAKKT");
  ierr = PetscObjectReference((PetscObject)Aff); CHKERRQ(ierr);
  if(kkt->Aff) {
    ierr = MatDestroy(&(kkt->Aff)); CHKERRQ(ierr);
  }
  kkt->Aff = Aff;

  ierr = DMDestroy(&(kkt->cdm));  CHKERRQ(ierr);
  ierr = MatDestroy(&(kkt->Pfc)); CHKERRQ(ierr);
  dm->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "DMAKKTGetMatrix"
PetscErrorCode DMAKKTGetMatrix(DM dm, Mat *Aff) {
  PetscBool iskkt;
  PetscErrorCode ierr;
  DM_AKKT *kkt = (DM_AKKT*)(dm->data);
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)dm, DMAKKT, &iskkt); CHKERRQ(ierr);
  if(!iskkt) SETERRQ(((PetscObject)dm)->comm, PETSC_ERR_ARG_WRONG, "DM not of type DMAKKT");
  if(Aff) {
    ierr = PetscObjectReference((PetscObject)(kkt->Aff)); CHKERRQ(ierr);
    *Aff = kkt->Aff;
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "DMAKKTSetFieldDecompositionName"
PetscErrorCode DMAKKTSetFieldDecompositionName(DM dm, const char* dname) {
  PetscBool iskkt;
  DM_AKKT *kkt = (DM_AKKT*)(dm->data);
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID,1);
  PetscValidCharPointer(dname,2);
  ierr = PetscObjectTypeCompare((PetscObject)dm, DMAKKT, &iskkt); CHKERRQ(ierr);
  if(!iskkt) SETERRQ(((PetscObject)dm)->comm, PETSC_ERR_ARG_WRONG, "DM not of type DMAKKT");
  if(kkt->dname) {
    ierr = PetscStrncpy(kkt->dname, dname, DMAKKT_DECOMPOSITION_NAME_LEN); CHKERRQ(ierr);
  }
  ierr = DMDestroy(&(kkt->cdm));  CHKERRQ(ierr);
  ierr = MatDestroy(&(kkt->Pfc)); CHKERRQ(ierr);
  dm->setupcalled = PETSC_FALSE;
  
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "DMAKKTGetFieldDecompositionName"
PetscErrorCode DMAKKTGetFieldDecompositionName(DM dm, char** dname) {
  PetscBool iskkt;
  DM_AKKT *kkt = (DM_AKKT*)(dm->data);
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID,1);
  PetscValidCharPointer(dname,2);
  ierr = PetscObjectTypeCompare((PetscObject)dm, DMAKKT, &iskkt); CHKERRQ(ierr);
  if(!iskkt) SETERRQ(((PetscObject)dm)->comm, PETSC_ERR_ARG_WRONG, "DM not of type DMAKKT");
  if(dname) {
    *dname = PETSC_NULL;
    if(kkt->dname) {
      ierr = PetscStrallocpy(kkt->dname, dname); CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "DMAKKTSetFieldDecomposition"
PetscErrorCode DMAKKTSetFieldDecomposition(DM dm, PetscInt n, const char* const *names, IS *iss, DM *dms) {
  PetscBool iskkt;
  DM_AKKT *kkt = (DM_AKKT*)(dm->data);
  PetscInt i;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID,1);
  PetscValidPointer(names,3);
  PetscValidPointer(iss,4);
  PetscValidPointer(dms,5);
  ierr = PetscObjectTypeCompare((PetscObject)dm, DMAKKT, &iskkt); CHKERRQ(ierr);
  if(!iskkt) SETERRQ(((PetscObject)dm)->comm, PETSC_ERR_ARG_WRONG, "DM not of type DMAKKT");
  if(n < 1 || n > 2) SETERRQ1(((PetscObject)dm)->comm, PETSC_ERR_ARG_WRONG, "Number of parts in decomposition must be between 1 and 2.  Got %D instead",n);
  for(i = 0; i < 2; ++i) {
    if(kkt->names[i]) {
      ierr = PetscFree(kkt->names[i]); CHKERRQ(ierr);
    }
    if(names[i]){
      ierr = PetscStrallocpy(names[i], &(kkt->names[i])); CHKERRQ(ierr);
    }
    if(iss[i]) {
      ierr = PetscObjectReference((PetscObject)iss[i]); CHKERRQ(ierr);
    }
    if(kkt->isf[i]) {
      ierr = ISDestroy(&(kkt->isf[i])); CHKERRQ(ierr);
    }
    kkt->isf[i] = iss[i];
    if(dms[i]) {
      ierr = PetscObjectReference((PetscObject)dms[i]); CHKERRQ(ierr);
    }
    if(kkt->dmf[i]) {
      ierr = DMDestroy(&(kkt->dmf[i])); CHKERRQ(ierr);
    }
    kkt->dmf[i] = dms[i];
  }
  ierr = DMDestroy(&(kkt->cdm));  CHKERRQ(ierr);
  ierr = MatDestroy(&(kkt->Pfc)); CHKERRQ(ierr);
  dm->setupcalled = PETSC_FALSE;
  
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "DMAKKTGetFieldDecomposition"
PetscErrorCode DMAKKTGetFieldDecomposition(DM dm, PetscInt *n, char*** names, IS **iss, DM **dms) {
  PetscBool iskkt;
  DM_AKKT *kkt = (DM_AKKT*)(dm->data);
  PetscInt i;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID,1);
  PetscValidPointer(names,3);
  PetscValidPointer(iss,4);
  PetscValidPointer(dms,5);
  ierr = PetscObjectTypeCompare((PetscObject)dm, DMAKKT, &iskkt); CHKERRQ(ierr);
  if(!iskkt) SETERRQ(((PetscObject)dm)->comm, PETSC_ERR_ARG_WRONG, "DM not of type DMAKKT");
  if(n) *n = 2;
  if(names) {
    if(kkt->names) {
      ierr = PetscMalloc(sizeof(char*)*2, names); CHKERRQ(ierr);
    }
    else {
      *names = PETSC_NULL;
    }
  }
  if(iss) {
    if(kkt->isf) {
      ierr = PetscMalloc(sizeof(IS)*2, iss); CHKERRQ(ierr);
    }
    else {
      *iss = PETSC_NULL;
    }
  }
  if(dms) {
    if(kkt->dmf) {
      ierr = PetscMalloc(sizeof(DM)*2, dms); CHKERRQ(ierr);
    }
    else {
      *dms = PETSC_NULL;
    }
  }
  for(i = 0; i < 2; ++i) {
    if(names && kkt->names){ 
      ierr = PetscStrallocpy(kkt->names[i],(*names)+i); CHKERRQ(ierr);
    }
    if(iss && kkt->isf) {
      ierr = PetscObjectReference((PetscObject)kkt->isf[i]); CHKERRQ(ierr);
      (*iss)[i] = kkt->isf[i];
    }
    if(dms && kkt->dmf) {
      ierr = PetscObjectReference((PetscObject)kkt->dmf[i]); CHKERRQ(ierr);
      (*dms)[i] = kkt->dmf[i];
    }
  }
  PetscFunctionReturn(0);
}


#undef  __FUNCT__
#define __FUNCT__ "DMSetFromOptions_AKKT"
PetscErrorCode DMSetFromOptions_AKKT(DM dm) {
  PetscErrorCode ierr;
  DM_AKKT* kkt = (DM_AKKT*)(dm->data);
  PetscFunctionBegin;
  ierr = PetscOptionsBool("-dm_akkt_duplicate_mat",
                          "Duplicate underlying Mat in DMCreateMatrix",
                          "DMAKKTSetDupulicateMat",
                          kkt->duplicate_mat,
                          &kkt->duplicate_mat,
                          PETSC_NULL);
  CHKERRQ(ierr);
  ierr = PetscOptionsBool("-dm_akkt_detect_saddle_point",
                          "Identify dual variables by zero diagonal entries",
                          "DMAKKTSetDetectSaddlePoint",
                          kkt->detect_saddle_point,
                          &kkt->detect_saddle_point,
                          PETSC_NULL);
  CHKERRQ(ierr);
  ierr = PetscOptionsString("-dm_akkt_decomposition_name",
                            "Name of primal-dual decomposition to request from DM",
                            "DMAKKTSetFieldDecompositionName",
                            kkt->dname,
                            kkt->dname,
                            DMAKKT_DECOMPOSITION_NAME_LEN,
                            PETSC_NULL);
  CHKERRQ(ierr);
  dm->setupcalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "DMSetUp_AKKT"
PetscErrorCode DMSetUp_AKKT(DM dm) {
  DM_AKKT *kkt = (DM_AKKT*)(dm->data);
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if(dm->setupcalled) PetscFunctionReturn(0);
  if(!kkt->Aff){
    if(kkt->dm) {
      ierr = DMCreateMatrix(kkt->dm, MATAIJ, &kkt->Aff); CHKERRQ(ierr);
    }
    else SETERRQ(((PetscObject)dm)->comm, PETSC_ERR_ARG_WRONGSTATE, "Neither matrix nor DM set");
  }
  if(!kkt->isf[0] && !kkt->isf[0]) {
    if(kkt->detect_saddle_point) {
        ierr = MatFindZeroDiagonals(kkt->Aff,&kkt->isf[1]);CHKERRQ(ierr);
    }
    else if(kkt->dm && kkt->dname) {
      DM ddm;
      PetscInt n;
      char **names;
      IS *iss;
      DM *dms;
      PetscInt i;
      ierr = DMCreateFieldDecompositionDM(kkt->dm, kkt->dname, &ddm); CHKERRQ(ierr);
      ierr = DMCreateFieldDecomposition(ddm, &n, &names, &iss, &dms); CHKERRQ(ierr);
      if(n < 1 || n > 2) 
        SETERRQ2(((PetscObject)dm)->comm, PETSC_ERR_ARG_WRONG, "Number of parts in decomposition %s must be between 1 and 2.  Got %D instead",kkt->dname, n);
      for(i = 0; i < n; ++i) {
        if(!iss[i] && dms[i]) {
          const char* label;
          if(i == 0) 
            label = "primal";
          else
            label = "dual";
          SETERRQ1(((PetscObject)dm)->comm, PETSC_ERR_ARG_WRONG, "Decomposition defines %s subDM, but no embedding IS is given", label);
        }
      }
      ierr = DMAKKTSetFieldDecomposition(dm, n, (const char**)names, iss, dms);     CHKERRQ(ierr);
      for(i = 0; i < n; ++i) {
        ierr = PetscFree(names[i]);   CHKERRQ(ierr);
        ierr = ISDestroy(&(iss[i]));  CHKERRQ(ierr);
        ierr = DMDestroy(&(dms[i]));  CHKERRQ(ierr);
      }
    }
  }
  if(!kkt->isf[0] && !kkt->isf[1]) SETERRQ(((PetscObject)dm)->comm, PETSC_ERR_ARG_WRONGSTATE, "Decomposition ISs not set and could not be derived. ");
  if(!kkt->isf[0] || !kkt->isf[1]) {
    PetscInt lstart, lend;
    ierr = MatGetOwnershipRange(kkt->Aff, &lstart, &lend); CHKERRQ(ierr);
    if(!kkt->isf[0]) {
      ierr = ISComplement(kkt->isf[0], lstart, lend, kkt->isf+1); CHKERRQ(ierr);
    }
    else {
      ierr = ISComplement(kkt->isf[1], lstart, lend, kkt->isf+0); CHKERRQ(ierr);
    }
  }
  /* FIX: Should we allow a combination of empty kkt->dmf[0] and non-empty kkt->dmf[1]? */
  if(!kkt->dmf[0]) {
    /* Construct a GAMG proxy to coarsen the primal block. */
    Mat A0f0f;
    IS  is00;
    PetscInt lstart, lend;
    const char* primal = {"all"};
    ierr = DMCreate(((PetscObject)dm)->comm, kkt->dmf+0); CHKERRQ(ierr);
    ierr = DMSetType(kkt->dmf[0],DMAKKT);                 CHKERRQ(ierr);
    ierr = MatGetSubMatrix(kkt->Aff, kkt->isf[0], kkt->isf[0], MAT_INITIAL_MATRIX, &A0f0f); CHKERRQ(ierr);
    ierr = DMAKKTSetMatrix(kkt->dmf[0], A0f0f);                                             CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(A0f0f, &lstart, &lend);                                     CHKERRQ(ierr);
    ierr = ISCreateStride(((PetscObject)A0f0f)->comm, lend-lstart, lstart, 1, &is00);       CHKERRQ(ierr);
    ierr = DMAKKTSetFieldDecomposition(kkt->dmf[0], 1, &primal, &is00, PETSC_NULL);              CHKERRQ(ierr);
  }
  dm->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/* 
 This routine will coarsen the 11-block only using the 00-block prolongation (P0f0c), the 10 block and GAMG. 
 The result is the 11-block prolongator (P1f1c). 
 */
#undef  __FUNCT__
#define __FUNCT__ "DMCoarsen_AKKT_GAMG11"
PetscErrorCode DMCoarsen_AKKT_GAMG11(DM dm, Mat P0f0c, Mat *P1f1c_out) {
  PetscErrorCode ierr;
  DM_AKKT* kkt = (DM_AKKT*)(dm->data);
  Mat Aff   = kkt->Aff;   /* fine-level KKT matrix */ 
  Mat A1f0f;              /* fine-level dual (constraint) Jacobian */
  Mat A1f0c;              /* = A1f0f*P0f0c coarsen only primal indices */
  Mat B1f1f;              /* = A1f0c'*A1f0c */
  PC  gamg11;/* Use PCGAMG internally to get access to some of its methods to operate on B1f1f = A1f0c*A1f0c', where A1f0c = A1f0f*P0f0c. */
  PC_GAMG* pc_gamg11;
  Mat G1f1f; /* = Graph(B1f1f) */
  Mat P1f1c; /* = Prolongator(G1f1f); */
  PetscCoarsenData *coarsening;
  PetscFunctionBegin;

  /* 
   What is faster: 
     - A0c1f = P0f0c'*A0f1f followed by B1f1f = A0c1f'*A0c1f, or
     - A1f0c = A1f0f*P0f0c  followed by B1f1f = A1f0c*A1f0c'?
   My bet is on the latter: 
     - fewer transpositions inside MatMatMult and row indices are always local.
   */

  ierr = MatGetSubMatrix(Aff, kkt->isf[1], kkt->isf[0], MAT_INITIAL_MATRIX, &A1f0f);      CHKERRQ(ierr);
  if(kkt->transposeP) {
    ierr = MatMatTransposeMult(A1f0f,P0f0c,MAT_INITIAL_MATRIX, PETSC_DEFAULT, &A1f0c);    CHKERRQ(ierr); 
  }
  ierr = MatMatMult(A1f0f,P0f0c,MAT_INITIAL_MATRIX, PETSC_DEFAULT, &A1f0c);               CHKERRQ(ierr); 
  ierr = MatMatTransposeMult(A1f0c, A1f0c, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &B1f1f);    CHKERRQ(ierr); 
  
  /* We create PCGAMG here since it is only needed for coarsening and we don't want to have to carry the attendant data structures, if we don't need them. */
  ierr = PCCreate(((PetscObject)dm)->comm, &gamg11); CHKERRQ(ierr);
  /* This must be an aggregating GAMG. */
  ierr = PCSetType(gamg11, PCGAMG);                  CHKERRQ(ierr);
  ierr = PCGAMGSetSquareGraph(gamg11, PETSC_FALSE);  CHKERRQ(ierr);
  /* 
   Observe that we want to "square" A1f0c before passing it (B1f1f) to GAMG.
   This is not because we are not sure how GAMG will deal with a (potentially) non-square matrix,
   but rather because if we asked GAMG to square it, it would also smooth the resulting prolongator.
   At least PC_GAMG_AGG would, and we need an unsmoothed prolongator. 
   */
  ierr = PCSetOperators(gamg11, B1f1f, B1f1f, DIFFERENT_NONZERO_PATTERN);            CHKERRQ(ierr);
  /* FIX: Currently there is no way to tell GAMG to coarsen onto a give comm, but it shouldn't be hard to hack that stuff in. */
  pc_gamg11 = (PC_GAMG*)(gamg11->data);
  ierr = pc_gamg11->graph(gamg11, B1f1f, &G1f1f);                                   CHKERRQ(ierr);
  ierr = pc_gamg11->coarsen(gamg11, &G1f1f, &coarsening);                           CHKERRQ(ierr);
  ierr = pc_gamg11->prolongator(gamg11, B1f1f, G1f1f, coarsening, &P1f1c);          CHKERRQ(ierr);

  ierr = MatDestroy(&A1f0f); CHKERRQ(ierr);
  ierr = MatDestroy(&A1f0c); CHKERRQ(ierr);
  ierr = MatDestroy(&B1f1f); CHKERRQ(ierr);
  ierr = MatDestroy(&G1f1f); CHKERRQ(ierr);
  ierr = PCDestroy(&gamg11); CHKERRQ(ierr);
  
  *P1f1c_out = P1f1c;

  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "DMCoarsen_AKKT"
PetscErrorCode DMCoarsen_AKKT(DM dm, MPI_Comm comm, DM *cdm) {
  PetscErrorCode ierr;
  DM_AKKT* kkt = (DM_AKKT*)(dm->data);
  Mat Acc;                                /* coarse-level KKT matrix */
  Mat P0f0c, P1f1c;                       /* Primal and dual block prolongator    */
  DM  dmc[2] = {PETSC_NULL, PETSC_NULL};  /* Coarse subDMs defining the block prolongators and the coarsened decomposition. */
  PetscInt M0,N0,M1,N1;   /* Sizes of P0f0c and P1f1c. */
  PetscInt start0,end0,start1,end1; /* Ownership ranges for P0f0c and P1f1c. */
  static Mat mats[4] = {PETSC_NULL, PETSC_NULL, PETSC_NULL, PETSC_NULL}; /* Used to construct MatNest out of pieces. */
  IS isc[2];  /* Used to construct MatNest out of pieces and to define the coarsened decomposition. */
  PetscFunctionBegin;
  if(!cdm) PetscFunctionReturn(0);
  if(kkt->cdm) {
    ierr = PetscObjectReference((PetscObject)(kkt->cdm)); CHKERRQ(ierr);
    *cdm = kkt->cdm;
    PetscFunctionReturn(0);
  }
  /* Coarsen the 00 block with the attached DM and obtain the primal prolongator. */
  if(kkt->dmf[0]) {
    ierr = DMCoarsen(kkt->dmf[0], comm,  dmc+0);                           CHKERRQ(ierr);
    ierr = DMCreateInterpolation(dmc[0], kkt->dmf[0], &P0f0c, PETSC_NULL); CHKERRQ(ierr);
  }
  else SETERRQ(((PetscObject)dm)->comm, PETSC_ERR_ARG_WRONGSTATE, "Could not coarsen the primal block: primal subDM not set.");
  
  /* Should P0f0c be transposed to act as a prolongator (i.e., to map from coarse to fine). */
  ierr = MatGetSize(P0f0c, &M0, &N0);                                         CHKERRQ(ierr);
  if(M0 == N0) SETERRQ1(((PetscObject)dm)->comm, PETSC_ERR_ARG_WRONGSTATE,"Primal prolongator is square with size %D: cannot distinguish coarse from fine",M0);
  if(M0 < N0) kkt->transposeP = PETSC_TRUE;
  else        kkt->transposeP = PETSC_FALSE;
  /* See if the 11 block can be coarsened with an attached DM. If so, we are done. Otherwise, use GAMG to coarsen 11. */
  if(kkt->dmf[1]) {
     ierr = DMCoarsen(kkt->dmf[1], comm, &dmc[1]);                              CHKERRQ(ierr);
     ierr = DMCreateInterpolation(dmc[1], kkt->dmf[1], &P1f1c, PETSC_NULL);     CHKERRQ(ierr);
  }
  else {
    ierr = DMCoarsen_AKKT_GAMG11(dm, P0f0c, &P1f1c); CHKERRQ(ierr);
  }
  /* Determine whether P1f1c should be transposed in order to act as a prolongator (i.e., to map from coarse to fine). */
  ierr = MatGetSize(P1f1c, &M1, &N1);                                          CHKERRQ(ierr);
  if(M1 == N1) SETERRQ1(((PetscObject)dm)->comm, PETSC_ERR_ARG_WRONGSTATE, "Dual prlongator is square with size %D: cannot distinguish coarse from fine", M1);
  if((M1 < N1 && !kkt->transposeP) || (M1 >= N1 && kkt->transposeP)) {
    Mat P1f1ct;
    ierr = MatTranspose(P1f1c, MAT_INITIAL_MATRIX, &P1f1ct); CHKERRQ(ierr);
    ierr = MatDestroy(&P1f1c);                               CHKERRQ(ierr);
    P1f1c = P1f1ct;
  }
  /* MatNest P0f0c, P1f1c together into Pfc. */
  mats[0] = P0f0c; mats[3] = P1f1c;
  ierr = MatGetOwnershipRange(P0f0c, &start0, &end0);   CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(P1f1c, &start1, &end1);   CHKERRQ(ierr);
  ierr = ISCreateStride(((PetscObject)dm)->comm, end0-start0,start0,1,isc+0); CHKERRQ(ierr);
  ierr = ISCreateStride(((PetscObject)dm)->comm, end1-start1,start1,1,isc+1); CHKERRQ(ierr);
  if(kkt->transposeP) {
    ierr = MatCreateNest(((PetscObject)dm)->comm,2,isc,2,kkt->isf,mats,&(kkt->Pfc)); CHKERRQ(ierr);
  }
  else {
    ierr = MatCreateNest(((PetscObject)dm)->comm,2,kkt->isf,2,isc,mats,&(kkt->Pfc)); CHKERRQ(ierr);
  }
  ierr = MatDestroy(&P0f0c); CHKERRQ(ierr);
  ierr = MatDestroy(&P1f1c); CHKERRQ(ierr);
  /* Coarsening the underlying matrix and primal-dual decomposition. */
  /* 
    We do not coarsen the underlying DM because 
    (a) Its coarsening may be incompatible with the specialized KKT-aware coarsening of the blocks defined here.
    (b) Even if the coarsening  the decomposition is compatible with the decomposition of the coarsening, we can 
        pick the former without loss of generality.
    (c) Even if (b) is true, the embeddings (IS) of the coarsened subDMs are potentially different now from what 
        they would be in the coarsened DM; thus, embeddings would have to be supplied manually anyhow.
    (d) In the typical situation we should only use the primal subDM for coarsening -- the whole point of 
        DMAKKT is that the dual block coarsening should be derived from the primal block coarsening for compatibility.
        If we are given both subDMs, DMAKKT essentially becomes a version of DMComposite, in which case the composition
        of the coarsened decomposition is by definition the coarsening of the whole system DM.
   */
  /* Create the coarser DM. */
  ierr = DMCreate(((PetscObject)dm)->comm, &(kkt->cdm)); CHKERRQ(ierr);
  ierr = DMSetType(kkt->cdm, DMAKKT);                    CHKERRQ(ierr);
  /* Coarsen the underlying matrix. */
  ierr = MatPtAP(kkt->Aff, kkt->Pfc, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Acc); CHKERRQ(ierr);
  ierr = DMAKKTSetMatrix(dm, Acc);                                             CHKERRQ(ierr);
  /* Set the coarsened decomposition. */
  ierr = DMAKKTSetFieldDecomposition(kkt->cdm, 2, (const char**)kkt->names, isc, dmc); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "DMCreateInterpolation_AKKT"
PetscErrorCode DMCreateInterpolation_AKKT(DM cdm, DM fdm, Mat *interp, Vec* rscale) {
  PetscBool iskkt;
  PetscErrorCode ierr;
  DM_AKKT* fkkt = (DM_AKKT*)(fdm->data);
  PetscFunctionBegin;
  PetscValidHeaderSpecific(cdm, DM_CLASSID,1);
  PetscValidHeaderSpecific(fdm, DM_CLASSID,2);
  ierr = PetscObjectTypeCompare((PetscObject)cdm, DMAKKT, &iskkt); CHKERRQ(ierr);
  if(!iskkt) SETERRQ(((PetscObject)cdm)->comm, PETSC_ERR_ARG_WRONG, "Coarse DM not of type DMAKKT");
  ierr = PetscObjectTypeCompare((PetscObject)fdm, DMAKKT, &iskkt); CHKERRQ(ierr);
  if(!iskkt) SETERRQ(((PetscObject)fdm)->comm, PETSC_ERR_ARG_WRONG, "Fine   DM not of type DMAKKT");
  if(fkkt->cdm != cdm) SETERRQ(((PetscObject)cdm)->comm, PETSC_ERR_ARG_WRONG, "Coarse DM must be obtained from fine via DMCoarsen");
  if(interp) {
    ierr = PetscObjectReference((PetscObject)(fkkt->Pfc)); CHKERRQ(ierr);
    *interp = fkkt->Pfc;
  }
  if(rscale) *rscale = PETSC_NULL;

  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "DMCreateMatrix_AKKT"
PetscErrorCode DMCreateMatrix_AKKT(DM dm, const MatType type, Mat *A) {
  PetscBool iskkt;
  PetscErrorCode ierr;
  DM_AKKT* kkt = (DM_AKKT*)(dm->data);
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)dm, DMAKKT, &iskkt); CHKERRQ(ierr);
  if(!iskkt) SETERRQ(((PetscObject)dm)->comm, PETSC_ERR_ARG_WRONG, "DM not of type DMAKKT");
  if(A) {
    if(kkt->duplicate_mat) {
      ierr = MatDuplicate(kkt->Aff, MAT_SHARE_NONZERO_PATTERN, A); CHKERRQ(ierr);
    }
    else {
      ierr = PetscObjectReference((PetscObject)(kkt->Aff)); CHKERRQ(ierr);
      *A = kkt->Aff;
    }
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "DMCreateGlobalVector_AKKT"
PetscErrorCode DMCreateGlobalVector_AKKT(DM dm, Vec *v) {
  PetscBool iskkt;
  PetscErrorCode ierr;
  DM_AKKT* kkt = (DM_AKKT*)(dm->data);
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)dm, DMAKKT, &iskkt); CHKERRQ(ierr);
  if(!iskkt) SETERRQ(((PetscObject)dm)->comm, PETSC_ERR_ARG_WRONG, "DM not of type DMAKKT");
  if(v) {
    ierr = MatGetVecs(kkt->Aff, v, PETSC_NULL); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "DMDestroy_AKKT"
PetscErrorCode DMDestroy_AKKT(DM dm) {
  DM_AKKT *kkt = (DM_AKKT*)(dm->data);
  PetscInt i;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MatDestroy(&(kkt->Aff));       CHKERRQ(ierr);
  ierr = DMDestroy(&(kkt->dm));         CHKERRQ(ierr);
  for(i = 0; i < 2; ++i) {
    ierr = DMDestroy(&(kkt->dmf[i]));   CHKERRQ(ierr);
    ierr = ISDestroy(&(kkt->isf[i]));   CHKERRQ(ierr);
    ierr = PetscFree(kkt->names[i]);    CHKERRQ(ierr);
  }
  ierr = DMDestroy(&(kkt->cdm));        CHKERRQ(ierr);
  ierr = MatDestroy(&(kkt->Pfc));       CHKERRQ(ierr);
  ierr = PetscFree(kkt); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "DMView_AKKT"
PetscErrorCode DMView_AKKT(DM dm, PetscViewer v) {
  DM_AKKT* kkt = (DM_AKKT*)(dm->data);
  PetscErrorCode ierr;
  PetscBool isascii;
  PetscInt  i, tab, vtab;
  const char* name, *prefix;
  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)v, PETSCVIEWERASCII, &isascii); CHKERRQ(ierr);
  if(!isascii) SETERRQ(((PetscObject)dm)->comm, PETSC_ERR_SUP, "No support for non-ASCII viewers"); 
  ierr = PetscObjectGetTabLevel((PetscObject)dm, &tab);      CHKERRQ(ierr);
  ierr = PetscObjectGetName((PetscObject)dm, &name);      CHKERRQ(ierr);
  ierr = PetscObjectGetOptionsPrefix((PetscObject)dm, &prefix);      CHKERRQ(ierr);
  ierr = PetscViewerASCIIUseTabs(v,PETSC_TRUE); CHKERRQ(ierr);
  ierr = PetscViewerASCIIGetTab(v,&vtab);       CHKERRQ(ierr);
  ierr = PetscViewerASCIISetTab(v,tab);         CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(v, "DM Algebraic KKT, name: %s, prefix: %s\n", ((PetscObject)dm)->name, ((PetscObject)dm)->prefix);         CHKERRQ(ierr);
  if(kkt->dm) {
    ierr = PetscViewerASCIIPrintf(v, "DM:\n");     CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(v);             CHKERRQ(ierr);
    ierr = DMView(kkt->dm,v);                      CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(v);              CHKERRQ(ierr);
  }
  if(kkt->Aff) {
    ierr = PetscViewerASCIIPrintf(v, "Aff:\n");    CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(v);             CHKERRQ(ierr);
    ierr = MatView(kkt->Aff,v);                    CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(v);              CHKERRQ(ierr);
  }
  if(kkt->dname) {
    ierr = PetscViewerASCIIPrintf(v, "Decomposition, name %s:\n");    CHKERRQ(ierr);
  }
  for(i = 0; i < 2; ++i) {
    const char* label;
    if(i == 0) {
      label = "Primal";
    }
    else {
      label = "Dual";
    }
    if(kkt->names[i]) {
      ierr = PetscViewerASCIIPrintf(v, "%s, name %s:\n", label, kkt->names[i]); CHKERRQ(ierr);
    }
    if(kkt->isf[i]){
      ierr = PetscViewerASCIIPrintf(v, "%s, IS:\n",label);                    CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushTab(v);                                      CHKERRQ(ierr);
      ierr = ISView(kkt->isf[i],v);                                           CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(v);                                       CHKERRQ(ierr);
    }
    if(kkt->dmf[i]){
      ierr = PetscViewerASCIIPrintf(v, "%s, DM:\n", label);                   CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushTab(v);                                      CHKERRQ(ierr);
      ierr = DMView(kkt->dmf[i],v);                                           CHKERRQ(ierr);
      ierr = PetscViewerASCIIPopTab(v);                                       CHKERRQ(ierr);
    }
  }
  if(kkt->Pfc) {
    ierr = PetscViewerASCIIPrintf(v, "Prolongation:\n");          CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(v);                            CHKERRQ(ierr);
    ierr = MatView(kkt->Pfc,v);                                   CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(v);                             CHKERRQ(ierr);
  }
  
  ierr = PetscViewerASCIISetTab(v,vtab);                          CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

#undef  __FUNCT__
#define __FUNCT__ "DMCreate_AKKT"
PetscErrorCode DMCreate_AKKT(DM dm) {
  DM_AKKT *kkt;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = PetscNewLog(dm, DM_AKKT, &kkt);
  dm->data = kkt;
  ierr = PetscObjectChangeTypeName((PetscObject)kkt,DMAKKT);CHKERRQ(ierr);
  kkt->dmf[0] = kkt->dmf[1] = PETSC_NULL;
  kkt->isf[0] = kkt->isf[1] = PETSC_NULL;
  kkt->names[0] = kkt->names[1] = PETSC_NULL;

  kkt->duplicate_mat       = PETSC_FALSE;
  kkt->detect_saddle_point = PETSC_FALSE;

  dm->ops->createglobalvector  = DMCreateGlobalVector_AKKT;
  dm->ops->creatematrix        = DMCreateMatrix_AKKT;
  dm->ops->createinterpolation = DMCreateInterpolation_AKKT;
  dm->ops->coarsen             = DMCoarsen_AKKT;
  dm->ops->destroy             = DMDestroy_AKKT;
  dm->ops->view                = DMView_AKKT;
  dm->ops->setfromoptions      = DMSetFromOptions_AKKT;
  dm->ops->setup               = DMSetUp_AKKT;

  PetscFunctionReturn(0);
}
