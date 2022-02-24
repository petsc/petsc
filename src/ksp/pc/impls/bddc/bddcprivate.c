#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/ksp/pc/impls/bddc/bddc.h>
#include <../src/ksp/pc/impls/bddc/bddcprivate.h>
#include <../src/mat/impls/dense/seq/dense.h>
#include <petscdmplex.h>
#include <petscblaslapack.h>
#include <petsc/private/sfimpl.h>
#include <petsc/private/dmpleximpl.h>
#include <petscdmda.h>

static PetscErrorCode MatMPIAIJRestrict(Mat,MPI_Comm,Mat*);

/* if range is true,  it returns B s.t. span{B} = range(A)
   if range is false, it returns B s.t. range(B) _|_ range(A) */
PetscErrorCode MatDenseOrthogonalRangeOrComplement(Mat A, PetscBool range, PetscInt lw, PetscScalar *work, PetscReal *rwork, Mat *B)
{
  PetscScalar    *uwork,*data,*U, ds = 0.;
  PetscReal      *sing;
  PetscBLASInt   bM,bN,lwork,lierr,di = 1;
  PetscInt       ulw,i,nr,nc,n;
#if defined(PETSC_USE_COMPLEX)
  PetscReal      *rwork2;
#endif

  PetscFunctionBegin;
  CHKERRQ(MatGetSize(A,&nr,&nc));
  if (!nr || !nc) PetscFunctionReturn(0);

  /* workspace */
  if (!work) {
    ulw  = PetscMax(PetscMax(1,5*PetscMin(nr,nc)),3*PetscMin(nr,nc)+PetscMax(nr,nc));
    CHKERRQ(PetscMalloc1(ulw,&uwork));
  } else {
    ulw   = lw;
    uwork = work;
  }
  n = PetscMin(nr,nc);
  if (!rwork) {
    CHKERRQ(PetscMalloc1(n,&sing));
  } else {
    sing = rwork;
  }

  /* SVD */
  CHKERRQ(PetscMalloc1(nr*nr,&U));
  CHKERRQ(PetscBLASIntCast(nr,&bM));
  CHKERRQ(PetscBLASIntCast(nc,&bN));
  CHKERRQ(PetscBLASIntCast(ulw,&lwork));
  CHKERRQ(MatDenseGetArray(A,&data));
  CHKERRQ(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
#if !defined(PETSC_USE_COMPLEX)
  PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("A","N",&bM,&bN,data,&bM,sing,U,&bM,&ds,&di,uwork,&lwork,&lierr));
#else
  CHKERRQ(PetscMalloc1(5*n,&rwork2));
  PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("A","N",&bM,&bN,data,&bM,sing,U,&bM,&ds,&di,uwork,&lwork,rwork2,&lierr));
  CHKERRQ(PetscFree(rwork2));
#endif
  CHKERRQ(PetscFPTrapPop());
  PetscCheckFalse(lierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in GESVD Lapack routine %d",(int)lierr);
  CHKERRQ(MatDenseRestoreArray(A,&data));
  for (i=0;i<n;i++) if (sing[i] < PETSC_SMALL) break;
  if (!rwork) {
    CHKERRQ(PetscFree(sing));
  }
  if (!work) {
    CHKERRQ(PetscFree(uwork));
  }
  /* create B */
  if (!range) {
    CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,nr,nr-i,NULL,B));
    CHKERRQ(MatDenseGetArray(*B,&data));
    CHKERRQ(PetscArraycpy(data,U+nr*i,(nr-i)*nr));
  } else {
    CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,nr,i,NULL,B));
    CHKERRQ(MatDenseGetArray(*B,&data));
    CHKERRQ(PetscArraycpy(data,U,i*nr));
  }
  CHKERRQ(MatDenseRestoreArray(*B,&data));
  CHKERRQ(PetscFree(U));
  PetscFunctionReturn(0);
}

/* TODO REMOVE */
#if defined(PRINT_GDET)
static int inc = 0;
static int lev = 0;
#endif

PetscErrorCode PCBDDCComputeNedelecChangeEdge(Mat lG, IS edge, IS extrow, IS extcol, IS corners, Mat* Gins, Mat* GKins, PetscScalar cvals[2], PetscScalar *work, PetscReal *rwork)
{
  Mat            GE,GEd;
  PetscInt       rsize,csize,esize;
  PetscScalar    *ptr;

  PetscFunctionBegin;
  CHKERRQ(ISGetSize(edge,&esize));
  if (!esize) PetscFunctionReturn(0);
  CHKERRQ(ISGetSize(extrow,&rsize));
  CHKERRQ(ISGetSize(extcol,&csize));

  /* gradients */
  ptr  = work + 5*esize;
  CHKERRQ(MatCreateSubMatrix(lG,extrow,extcol,MAT_INITIAL_MATRIX,&GE));
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,rsize,csize,ptr,Gins));
  CHKERRQ(MatConvert(GE,MATSEQDENSE,MAT_REUSE_MATRIX,Gins));
  CHKERRQ(MatDestroy(&GE));

  /* constants */
  ptr += rsize*csize;
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,esize,csize,ptr,&GEd));
  CHKERRQ(MatCreateSubMatrix(lG,edge,extcol,MAT_INITIAL_MATRIX,&GE));
  CHKERRQ(MatConvert(GE,MATSEQDENSE,MAT_REUSE_MATRIX,&GEd));
  CHKERRQ(MatDestroy(&GE));
  CHKERRQ(MatDenseOrthogonalRangeOrComplement(GEd,PETSC_FALSE,5*esize,work,rwork,GKins));
  CHKERRQ(MatDestroy(&GEd));

  if (corners) {
    Mat               GEc;
    const PetscScalar *vals;
    PetscScalar       v;

    CHKERRQ(MatCreateSubMatrix(lG,edge,corners,MAT_INITIAL_MATRIX,&GEc));
    CHKERRQ(MatTransposeMatMult(GEc,*GKins,MAT_INITIAL_MATRIX,1.0,&GEd));
    CHKERRQ(MatDenseGetArrayRead(GEd,&vals));
    /* v    = PetscAbsScalar(vals[0]) */;
    v    = 1.;
    cvals[0] = vals[0]/v;
    cvals[1] = vals[1]/v;
    CHKERRQ(MatDenseRestoreArrayRead(GEd,&vals));
    CHKERRQ(MatScale(*GKins,1./v));
#if defined(PRINT_GDET)
    {
      PetscViewer viewer;
      char filename[256];
      sprintf(filename,"Gdet_l%d_r%d_cc%d.m",lev,PetscGlobalRank,inc++);
      CHKERRQ(PetscViewerASCIIOpen(PETSC_COMM_SELF,filename,&viewer));
      CHKERRQ(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB));
      CHKERRQ(PetscObjectSetName((PetscObject)GEc,"GEc"));
      CHKERRQ(MatView(GEc,viewer));
      CHKERRQ(PetscObjectSetName((PetscObject)(*GKins),"GK"));
      CHKERRQ(MatView(*GKins,viewer));
      CHKERRQ(PetscObjectSetName((PetscObject)GEd,"Gproj"));
      CHKERRQ(MatView(GEd,viewer));
      CHKERRQ(PetscViewerDestroy(&viewer));
    }
#endif
    CHKERRQ(MatDestroy(&GEd));
    CHKERRQ(MatDestroy(&GEc));
  }

  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCNedelecSupport(PC pc)
{
  PC_BDDC                *pcbddc = (PC_BDDC*)pc->data;
  Mat_IS                 *matis = (Mat_IS*)pc->pmat->data;
  Mat                    G,T,conn,lG,lGt,lGis,lGall,lGe,lGinit;
  Vec                    tvec;
  PetscSF                sfv;
  ISLocalToGlobalMapping el2g,vl2g,fl2g,al2g;
  MPI_Comm               comm;
  IS                     lned,primals,allprimals,nedfieldlocal;
  IS                     *eedges,*extrows,*extcols,*alleedges;
  PetscBT                btv,bte,btvc,btb,btbd,btvcand,btvi,btee,bter;
  PetscScalar            *vals,*work;
  PetscReal              *rwork;
  const PetscInt         *idxs,*ii,*jj,*iit,*jjt;
  PetscInt               ne,nv,Lv,order,n,field;
  PetscInt               n_neigh,*neigh,*n_shared,**shared;
  PetscInt               i,j,extmem,cum,maxsize,nee;
  PetscInt               *extrow,*extrowcum,*marks,*vmarks,*gidxs;
  PetscInt               *sfvleaves,*sfvroots;
  PetscInt               *corners,*cedges;
  PetscInt               *ecount,**eneighs,*vcount,**vneighs;
  PetscInt               *emarks;
  PetscBool              print,eerr,done,lrc[2],conforming,global,singular,setprimal;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  /* If the discrete gradient is defined for a subset of dofs and global is true,
     it assumes G is given in global ordering for all the dofs.
     Otherwise, the ordering is global for the Nedelec field */
  order      = pcbddc->nedorder;
  conforming = pcbddc->conforming;
  field      = pcbddc->nedfield;
  global     = pcbddc->nedglobal;
  setprimal  = PETSC_FALSE;
  print      = PETSC_FALSE;
  singular   = PETSC_FALSE;

  /* Command line customization */
  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)pc),((PetscObject)pc)->prefix,"BDDC Nedelec options","PC");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsBool("-pc_bddc_nedelec_field_primal","All edge dofs set as primals: Toselli's algorithm C",NULL,setprimal,&setprimal,NULL));
  CHKERRQ(PetscOptionsBool("-pc_bddc_nedelec_singular","Infer nullspace from discrete gradient",NULL,singular,&singular,NULL));
  CHKERRQ(PetscOptionsInt("-pc_bddc_nedelec_order","Test variable order code (to be removed)",NULL,order,&order,NULL));
  /* print debug info TODO: to be removed */
  CHKERRQ(PetscOptionsBool("-pc_bddc_nedelec_print","Print debug info",NULL,print,&print,NULL));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* Return if there are no edges in the decomposition and the problem is not singular */
  CHKERRQ(MatISGetLocalToGlobalMapping(pc->pmat,&al2g,NULL));
  CHKERRQ(ISLocalToGlobalMappingGetSize(al2g,&n));
  CHKERRQ(PetscObjectGetComm((PetscObject)pc,&comm));
  if (!singular) {
    CHKERRQ(VecGetArrayRead(matis->counter,(const PetscScalar**)&vals));
    lrc[0] = PETSC_FALSE;
    for (i=0;i<n;i++) {
      if (PetscRealPart(vals[i]) > 2.) {
        lrc[0] = PETSC_TRUE;
        break;
      }
    }
    CHKERRQ(VecRestoreArrayRead(matis->counter,(const PetscScalar**)&vals));
    CHKERRMPI(MPIU_Allreduce(&lrc[0],&lrc[1],1,MPIU_BOOL,MPI_LOR,comm));
    if (!lrc[1]) PetscFunctionReturn(0);
  }

  /* Get Nedelec field */
  PetscCheckFalse(pcbddc->n_ISForDofsLocal && field >= pcbddc->n_ISForDofsLocal,comm,PETSC_ERR_USER,"Invalid field for Nedelec %D: number of fields is %D",field,pcbddc->n_ISForDofsLocal);
  if (pcbddc->n_ISForDofsLocal && field >= 0) {
    CHKERRQ(PetscObjectReference((PetscObject)pcbddc->ISForDofsLocal[field]));
    nedfieldlocal = pcbddc->ISForDofsLocal[field];
    CHKERRQ(ISGetLocalSize(nedfieldlocal,&ne));
  } else if (!pcbddc->n_ISForDofsLocal && field != PETSC_DECIDE) {
    ne            = n;
    nedfieldlocal = NULL;
    global        = PETSC_TRUE;
  } else if (field == PETSC_DECIDE) {
    PetscInt rst,ren,*idx;

    CHKERRQ(PetscArrayzero(matis->sf_leafdata,n));
    CHKERRQ(PetscArrayzero(matis->sf_rootdata,pc->pmat->rmap->n));
    CHKERRQ(MatGetOwnershipRange(pcbddc->discretegradient,&rst,&ren));
    for (i=rst;i<ren;i++) {
      PetscInt nc;

      CHKERRQ(MatGetRow(pcbddc->discretegradient,i,&nc,NULL,NULL));
      if (nc > 1) matis->sf_rootdata[i-rst] = 1;
      CHKERRQ(MatRestoreRow(pcbddc->discretegradient,i,&nc,NULL,NULL));
    }
    CHKERRQ(PetscSFBcastBegin(matis->sf,MPIU_INT,matis->sf_rootdata,matis->sf_leafdata,MPI_REPLACE));
    CHKERRQ(PetscSFBcastEnd(matis->sf,MPIU_INT,matis->sf_rootdata,matis->sf_leafdata,MPI_REPLACE));
    CHKERRQ(PetscMalloc1(n,&idx));
    for (i=0,ne=0;i<n;i++) if (matis->sf_leafdata[i]) idx[ne++] = i;
    CHKERRQ(ISCreateGeneral(comm,ne,idx,PETSC_OWN_POINTER,&nedfieldlocal));
  } else {
    SETERRQ(comm,PETSC_ERR_USER,"When multiple fields are present, the Nedelec field has to be specified");
  }

  /* Sanity checks */
  PetscCheckFalse(!order && !conforming,comm,PETSC_ERR_SUP,"Variable order and non-conforming spaces are not supported at the same time");
  PetscCheckFalse(pcbddc->user_ChangeOfBasisMatrix,comm,PETSC_ERR_SUP,"Cannot generate Nedelec support with user defined change of basis");
  PetscCheckFalse(order && ne%order,PETSC_COMM_SELF,PETSC_ERR_USER,"The number of local edge dofs %D it's not a multiple of the order %D",ne,order);

  /* Just set primal dofs and return */
  if (setprimal) {
    IS       enedfieldlocal;
    PetscInt *eidxs;

    CHKERRQ(PetscMalloc1(ne,&eidxs));
    CHKERRQ(VecGetArrayRead(matis->counter,(const PetscScalar**)&vals));
    if (nedfieldlocal) {
      CHKERRQ(ISGetIndices(nedfieldlocal,&idxs));
      for (i=0,cum=0;i<ne;i++) {
        if (PetscRealPart(vals[idxs[i]]) > 2.) {
          eidxs[cum++] = idxs[i];
        }
      }
      CHKERRQ(ISRestoreIndices(nedfieldlocal,&idxs));
    } else {
      for (i=0,cum=0;i<ne;i++) {
        if (PetscRealPart(vals[i]) > 2.) {
          eidxs[cum++] = i;
        }
      }
    }
    CHKERRQ(VecRestoreArrayRead(matis->counter,(const PetscScalar**)&vals));
    CHKERRQ(ISCreateGeneral(comm,cum,eidxs,PETSC_COPY_VALUES,&enedfieldlocal));
    CHKERRQ(PCBDDCSetPrimalVerticesLocalIS(pc,enedfieldlocal));
    CHKERRQ(PetscFree(eidxs));
    CHKERRQ(ISDestroy(&nedfieldlocal));
    CHKERRQ(ISDestroy(&enedfieldlocal));
    PetscFunctionReturn(0);
  }

  /* Compute some l2g maps */
  if (nedfieldlocal) {
    IS is;

    /* need to map from the local Nedelec field to local numbering */
    CHKERRQ(ISLocalToGlobalMappingCreateIS(nedfieldlocal,&fl2g));
    /* need to map from the local Nedelec field to global numbering for the whole dofs*/
    CHKERRQ(ISLocalToGlobalMappingApplyIS(al2g,nedfieldlocal,&is));
    CHKERRQ(ISLocalToGlobalMappingCreateIS(is,&al2g));
    /* need to map from the local Nedelec field to global numbering (for Nedelec only) */
    if (global) {
      CHKERRQ(PetscObjectReference((PetscObject)al2g));
      el2g = al2g;
    } else {
      IS gis;

      CHKERRQ(ISRenumber(is,NULL,NULL,&gis));
      CHKERRQ(ISLocalToGlobalMappingCreateIS(gis,&el2g));
      CHKERRQ(ISDestroy(&gis));
    }
    CHKERRQ(ISDestroy(&is));
  } else {
    /* restore default */
    pcbddc->nedfield = -1;
    /* one ref for the destruction of al2g, one for el2g */
    CHKERRQ(PetscObjectReference((PetscObject)al2g));
    CHKERRQ(PetscObjectReference((PetscObject)al2g));
    el2g = al2g;
    fl2g = NULL;
  }

  /* Start communication to drop connections for interior edges (for cc analysis only) */
  CHKERRQ(PetscArrayzero(matis->sf_leafdata,n));
  CHKERRQ(PetscArrayzero(matis->sf_rootdata,pc->pmat->rmap->n));
  if (nedfieldlocal) {
    CHKERRQ(ISGetIndices(nedfieldlocal,&idxs));
    for (i=0;i<ne;i++) matis->sf_leafdata[idxs[i]] = 1;
    CHKERRQ(ISRestoreIndices(nedfieldlocal,&idxs));
  } else {
    for (i=0;i<ne;i++) matis->sf_leafdata[i] = 1;
  }
  CHKERRQ(PetscSFReduceBegin(matis->sf,MPIU_INT,matis->sf_leafdata,matis->sf_rootdata,MPI_SUM));
  CHKERRQ(PetscSFReduceEnd(matis->sf,MPIU_INT,matis->sf_leafdata,matis->sf_rootdata,MPI_SUM));

  if (!singular) { /* drop connections with interior edges to avoid unneeded communications and memory movements */
    CHKERRQ(MatDuplicate(pcbddc->discretegradient,MAT_COPY_VALUES,&G));
    CHKERRQ(MatSetOption(G,MAT_KEEP_NONZERO_PATTERN,PETSC_FALSE));
    if (global) {
      PetscInt rst;

      CHKERRQ(MatGetOwnershipRange(G,&rst,NULL));
      for (i=0,cum=0;i<pc->pmat->rmap->n;i++) {
        if (matis->sf_rootdata[i] < 2) {
          matis->sf_rootdata[cum++] = i + rst;
        }
      }
      CHKERRQ(MatSetOption(G,MAT_NO_OFF_PROC_ZERO_ROWS,PETSC_TRUE));
      CHKERRQ(MatZeroRows(G,cum,matis->sf_rootdata,0.,NULL,NULL));
    } else {
      PetscInt *tbz;

      CHKERRQ(PetscMalloc1(ne,&tbz));
      CHKERRQ(PetscSFBcastBegin(matis->sf,MPIU_INT,matis->sf_rootdata,matis->sf_leafdata,MPI_REPLACE));
      CHKERRQ(PetscSFBcastEnd(matis->sf,MPIU_INT,matis->sf_rootdata,matis->sf_leafdata,MPI_REPLACE));
      CHKERRQ(ISGetIndices(nedfieldlocal,&idxs));
      for (i=0,cum=0;i<ne;i++)
        if (matis->sf_leafdata[idxs[i]] == 1)
          tbz[cum++] = i;
      CHKERRQ(ISRestoreIndices(nedfieldlocal,&idxs));
      CHKERRQ(ISLocalToGlobalMappingApply(el2g,cum,tbz,tbz));
      CHKERRQ(MatZeroRows(G,cum,tbz,0.,NULL,NULL));
      CHKERRQ(PetscFree(tbz));
    }
  } else { /* we need the entire G to infer the nullspace */
    CHKERRQ(PetscObjectReference((PetscObject)pcbddc->discretegradient));
    G    = pcbddc->discretegradient;
  }

  /* Extract subdomain relevant rows of G */
  CHKERRQ(ISLocalToGlobalMappingGetIndices(el2g,&idxs));
  CHKERRQ(ISCreateGeneral(comm,ne,idxs,PETSC_USE_POINTER,&lned));
  CHKERRQ(MatCreateSubMatrix(G,lned,NULL,MAT_INITIAL_MATRIX,&lGall));
  CHKERRQ(ISLocalToGlobalMappingRestoreIndices(el2g,&idxs));
  CHKERRQ(ISDestroy(&lned));
  CHKERRQ(MatConvert(lGall,MATIS,MAT_INITIAL_MATRIX,&lGis));
  CHKERRQ(MatDestroy(&lGall));
  CHKERRQ(MatISGetLocalMat(lGis,&lG));

  /* SF for nodal dofs communications */
  CHKERRQ(MatGetLocalSize(G,NULL,&Lv));
  CHKERRQ(MatISGetLocalToGlobalMapping(lGis,NULL,&vl2g));
  CHKERRQ(PetscObjectReference((PetscObject)vl2g));
  CHKERRQ(ISLocalToGlobalMappingGetSize(vl2g,&nv));
  CHKERRQ(PetscSFCreate(comm,&sfv));
  CHKERRQ(ISLocalToGlobalMappingGetIndices(vl2g,&idxs));
  CHKERRQ(PetscSFSetGraphLayout(sfv,lGis->cmap,nv,NULL,PETSC_OWN_POINTER,idxs));
  CHKERRQ(ISLocalToGlobalMappingRestoreIndices(vl2g,&idxs));
  i    = singular ? 2 : 1;
  CHKERRQ(PetscMalloc2(i*nv,&sfvleaves,i*Lv,&sfvroots));

  /* Destroy temporary G created in MATIS format and modified G */
  CHKERRQ(PetscObjectReference((PetscObject)lG));
  CHKERRQ(MatDestroy(&lGis));
  CHKERRQ(MatDestroy(&G));

  if (print) {
    CHKERRQ(PetscObjectSetName((PetscObject)lG,"initial_lG"));
    CHKERRQ(MatView(lG,NULL));
  }

  /* Save lG for values insertion in change of basis */
  CHKERRQ(MatDuplicate(lG,MAT_COPY_VALUES,&lGinit));

  /* Analyze the edge-nodes connections (duplicate lG) */
  CHKERRQ(MatDuplicate(lG,MAT_COPY_VALUES,&lGe));
  CHKERRQ(MatSetOption(lGe,MAT_KEEP_NONZERO_PATTERN,PETSC_FALSE));
  CHKERRQ(PetscBTCreate(nv,&btv));
  CHKERRQ(PetscBTCreate(ne,&bte));
  CHKERRQ(PetscBTCreate(ne,&btb));
  CHKERRQ(PetscBTCreate(ne,&btbd));
  CHKERRQ(PetscBTCreate(nv,&btvcand));
  /* need to import the boundary specification to ensure the
     proper detection of coarse edges' endpoints */
  if (pcbddc->DirichletBoundariesLocal) {
    IS is;

    if (fl2g) {
      CHKERRQ(ISGlobalToLocalMappingApplyIS(fl2g,IS_GTOLM_MASK,pcbddc->DirichletBoundariesLocal,&is));
    } else {
      is = pcbddc->DirichletBoundariesLocal;
    }
    CHKERRQ(ISGetLocalSize(is,&cum));
    CHKERRQ(ISGetIndices(is,&idxs));
    for (i=0;i<cum;i++) {
      if (idxs[i] >= 0) {
        CHKERRQ(PetscBTSet(btb,idxs[i]));
        CHKERRQ(PetscBTSet(btbd,idxs[i]));
      }
    }
    CHKERRQ(ISRestoreIndices(is,&idxs));
    if (fl2g) {
      CHKERRQ(ISDestroy(&is));
    }
  }
  if (pcbddc->NeumannBoundariesLocal) {
    IS is;

    if (fl2g) {
      CHKERRQ(ISGlobalToLocalMappingApplyIS(fl2g,IS_GTOLM_MASK,pcbddc->NeumannBoundariesLocal,&is));
    } else {
      is = pcbddc->NeumannBoundariesLocal;
    }
    CHKERRQ(ISGetLocalSize(is,&cum));
    CHKERRQ(ISGetIndices(is,&idxs));
    for (i=0;i<cum;i++) {
      if (idxs[i] >= 0) {
        CHKERRQ(PetscBTSet(btb,idxs[i]));
      }
    }
    CHKERRQ(ISRestoreIndices(is,&idxs));
    if (fl2g) {
      CHKERRQ(ISDestroy(&is));
    }
  }

  /* Count neighs per dof */
  CHKERRQ(ISLocalToGlobalMappingGetNodeInfo(el2g,NULL,&ecount,&eneighs));
  CHKERRQ(ISLocalToGlobalMappingGetNodeInfo(vl2g,NULL,&vcount,&vneighs));

  /* need to remove coarse faces' dofs and coarse edges' dirichlet dofs
     for proper detection of coarse edges' endpoints */
  CHKERRQ(PetscBTCreate(ne,&btee));
  for (i=0;i<ne;i++) {
    if ((ecount[i] > 2 && !PetscBTLookup(btbd,i)) || (ecount[i] == 2 && PetscBTLookup(btb,i))) {
      CHKERRQ(PetscBTSet(btee,i));
    }
  }
  CHKERRQ(PetscMalloc1(ne,&marks));
  if (!conforming) {
    CHKERRQ(MatTranspose(lGe,MAT_INITIAL_MATRIX,&lGt));
    CHKERRQ(MatGetRowIJ(lGt,0,PETSC_FALSE,PETSC_FALSE,&i,&iit,&jjt,&done));
  }
  CHKERRQ(MatGetRowIJ(lGe,0,PETSC_FALSE,PETSC_FALSE,&i,&ii,&jj,&done));
  CHKERRQ(MatSeqAIJGetArray(lGe,&vals));
  cum  = 0;
  for (i=0;i<ne;i++) {
    /* eliminate rows corresponding to edge dofs belonging to coarse faces */
    if (!PetscBTLookup(btee,i)) {
      marks[cum++] = i;
      continue;
    }
    /* set badly connected edge dofs as primal */
    if (!conforming) {
      if (ii[i+1]-ii[i] != order + 1) { /* every row of G on the coarse edge should list order+1 nodal dofs */
        marks[cum++] = i;
        CHKERRQ(PetscBTSet(bte,i));
        for (j=ii[i];j<ii[i+1];j++) {
          CHKERRQ(PetscBTSet(btv,jj[j]));
        }
      } else {
        /* every edge dofs should be connected trough a certain number of nodal dofs
           to other edge dofs belonging to coarse edges
           - at most 2 endpoints
           - order-1 interior nodal dofs
           - no undefined nodal dofs (nconn < order)
        */
        PetscInt ends = 0,ints = 0, undef = 0;
        for (j=ii[i];j<ii[i+1];j++) {
          PetscInt v = jj[j],k;
          PetscInt nconn = iit[v+1]-iit[v];
          for (k=iit[v];k<iit[v+1];k++) if (!PetscBTLookup(btee,jjt[k])) nconn--;
          if (nconn > order) ends++;
          else if (nconn == order) ints++;
          else undef++;
        }
        if (undef || ends > 2 || ints != order -1) {
          marks[cum++] = i;
          CHKERRQ(PetscBTSet(bte,i));
          for (j=ii[i];j<ii[i+1];j++) {
            CHKERRQ(PetscBTSet(btv,jj[j]));
          }
        }
      }
    }
    /* We assume the order on the element edge is ii[i+1]-ii[i]-1 */
    if (!order && ii[i+1] != ii[i]) {
      PetscScalar val = 1./(ii[i+1]-ii[i]-1);
      for (j=ii[i];j<ii[i+1];j++) vals[j] = val;
    }
  }
  CHKERRQ(PetscBTDestroy(&btee));
  CHKERRQ(MatSeqAIJRestoreArray(lGe,&vals));
  CHKERRQ(MatRestoreRowIJ(lGe,0,PETSC_FALSE,PETSC_FALSE,&i,&ii,&jj,&done));
  if (!conforming) {
    CHKERRQ(MatRestoreRowIJ(lGt,0,PETSC_FALSE,PETSC_FALSE,&i,&iit,&jjt,&done));
    CHKERRQ(MatDestroy(&lGt));
  }
  CHKERRQ(MatZeroRows(lGe,cum,marks,0.,NULL,NULL));

  /* identify splitpoints and corner candidates */
  CHKERRQ(MatTranspose(lGe,MAT_INITIAL_MATRIX,&lGt));
  if (print) {
    CHKERRQ(PetscObjectSetName((PetscObject)lGe,"edgerestr_lG"));
    CHKERRQ(MatView(lGe,NULL));
    CHKERRQ(PetscObjectSetName((PetscObject)lGt,"edgerestr_lGt"));
    CHKERRQ(MatView(lGt,NULL));
  }
  CHKERRQ(MatGetRowIJ(lGt,0,PETSC_FALSE,PETSC_FALSE,&i,&ii,&jj,&done));
  CHKERRQ(MatSeqAIJGetArray(lGt,&vals));
  for (i=0;i<nv;i++) {
    PetscInt  ord = order, test = ii[i+1]-ii[i], vc = vcount[i];
    PetscBool sneighs = PETSC_TRUE, bdir = PETSC_FALSE;
    if (!order) { /* variable order */
      PetscReal vorder = 0.;

      for (j=ii[i];j<ii[i+1];j++) vorder += PetscRealPart(vals[j]);
      test = PetscFloorReal(vorder+10.*PETSC_SQRT_MACHINE_EPSILON);
      PetscCheckFalse(vorder-test > PETSC_SQRT_MACHINE_EPSILON,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unexpected value for vorder: %g (%D)",vorder,test);
      ord  = 1;
    }
    PetscAssert(test%ord == 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unexpected number of edge dofs %" PetscInt_FMT " connected with nodal dof %" PetscInt_FMT " with order %" PetscInt_FMT,test,i,ord);
    for (j=ii[i];j<ii[i+1] && sneighs;j++) {
      if (PetscBTLookup(btbd,jj[j])) {
        bdir = PETSC_TRUE;
        break;
      }
      if (vc != ecount[jj[j]]) {
        sneighs = PETSC_FALSE;
      } else {
        PetscInt k,*vn = vneighs[i], *en = eneighs[jj[j]];
        for (k=0;k<vc;k++) {
          if (vn[k] != en[k]) {
            sneighs = PETSC_FALSE;
            break;
          }
        }
      }
    }
    if (!sneighs || test >= 3*ord || bdir) { /* splitpoints */
      if (print) PetscPrintf(PETSC_COMM_SELF,"SPLITPOINT %D (%D %D %D)\n",i,!sneighs,test >= 3*ord,bdir);
      CHKERRQ(PetscBTSet(btv,i));
    } else if (test == ord) {
      if (order == 1 || (!order && ii[i+1]-ii[i] == 1)) {
        if (print) PetscPrintf(PETSC_COMM_SELF,"ENDPOINT %D\n",i);
        CHKERRQ(PetscBTSet(btv,i));
      } else {
        if (print) PetscPrintf(PETSC_COMM_SELF,"CORNER CANDIDATE %D\n",i);
        CHKERRQ(PetscBTSet(btvcand,i));
      }
    }
  }
  CHKERRQ(ISLocalToGlobalMappingRestoreNodeInfo(el2g,NULL,&ecount,&eneighs));
  CHKERRQ(ISLocalToGlobalMappingRestoreNodeInfo(vl2g,NULL,&vcount,&vneighs));
  CHKERRQ(PetscBTDestroy(&btbd));

  /* a candidate is valid if it is connected to another candidate via a non-primal edge dof */
  if (order != 1) {
    if (print) PetscPrintf(PETSC_COMM_SELF,"INSPECTING CANDIDATES\n");
    CHKERRQ(MatGetRowIJ(lGe,0,PETSC_FALSE,PETSC_FALSE,&i,&iit,&jjt,&done));
    for (i=0;i<nv;i++) {
      if (PetscBTLookup(btvcand,i)) {
        PetscBool found = PETSC_FALSE;
        for (j=ii[i];j<ii[i+1] && !found;j++) {
          PetscInt k,e = jj[j];
          if (PetscBTLookup(bte,e)) continue;
          for (k=iit[e];k<iit[e+1];k++) {
            PetscInt v = jjt[k];
            if (v != i && PetscBTLookup(btvcand,v)) {
              found = PETSC_TRUE;
              break;
            }
          }
        }
        if (!found) {
          if (print) PetscPrintf(PETSC_COMM_SELF,"  CANDIDATE %D CLEARED\n",i);
          CHKERRQ(PetscBTClear(btvcand,i));
        } else {
          if (print) PetscPrintf(PETSC_COMM_SELF,"  CANDIDATE %D ACCEPTED\n",i);
        }
      }
    }
    CHKERRQ(MatRestoreRowIJ(lGe,0,PETSC_FALSE,PETSC_FALSE,&i,&iit,&jjt,&done));
  }
  CHKERRQ(MatSeqAIJRestoreArray(lGt,&vals));
  CHKERRQ(MatRestoreRowIJ(lGt,0,PETSC_FALSE,PETSC_FALSE,&i,&ii,&jj,&done));
  CHKERRQ(MatDestroy(&lGe));

  /* Get the local G^T explicitly */
  CHKERRQ(MatDestroy(&lGt));
  CHKERRQ(MatTranspose(lG,MAT_INITIAL_MATRIX,&lGt));
  CHKERRQ(MatSetOption(lGt,MAT_KEEP_NONZERO_PATTERN,PETSC_FALSE));

  /* Mark interior nodal dofs */
  CHKERRQ(ISLocalToGlobalMappingGetInfo(vl2g,&n_neigh,&neigh,&n_shared,&shared));
  CHKERRQ(PetscBTCreate(nv,&btvi));
  for (i=1;i<n_neigh;i++) {
    for (j=0;j<n_shared[i];j++) {
      CHKERRQ(PetscBTSet(btvi,shared[i][j]));
    }
  }
  CHKERRQ(ISLocalToGlobalMappingRestoreInfo(vl2g,&n_neigh,&neigh,&n_shared,&shared));

  /* communicate corners and splitpoints */
  CHKERRQ(PetscMalloc1(nv,&vmarks));
  CHKERRQ(PetscArrayzero(sfvleaves,nv));
  CHKERRQ(PetscArrayzero(sfvroots,Lv));
  for (i=0;i<nv;i++) if (PetscUnlikely(PetscBTLookup(btv,i))) sfvleaves[i] = 1;

  if (print) {
    IS tbz;

    cum = 0;
    for (i=0;i<nv;i++)
      if (sfvleaves[i])
        vmarks[cum++] = i;

    CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,cum,vmarks,PETSC_COPY_VALUES,&tbz));
    CHKERRQ(PetscObjectSetName((PetscObject)tbz,"corners_to_be_zeroed_local"));
    CHKERRQ(ISView(tbz,NULL));
    CHKERRQ(ISDestroy(&tbz));
  }

  CHKERRQ(PetscSFReduceBegin(sfv,MPIU_INT,sfvleaves,sfvroots,MPI_SUM));
  CHKERRQ(PetscSFReduceEnd(sfv,MPIU_INT,sfvleaves,sfvroots,MPI_SUM));
  CHKERRQ(PetscSFBcastBegin(sfv,MPIU_INT,sfvroots,sfvleaves,MPI_REPLACE));
  CHKERRQ(PetscSFBcastEnd(sfv,MPIU_INT,sfvroots,sfvleaves,MPI_REPLACE));

  /* Zero rows of lGt corresponding to identified corners
     and interior nodal dofs */
  cum = 0;
  for (i=0;i<nv;i++) {
    if (sfvleaves[i]) {
      vmarks[cum++] = i;
      CHKERRQ(PetscBTSet(btv,i));
    }
    if (!PetscBTLookup(btvi,i)) vmarks[cum++] = i;
  }
  CHKERRQ(PetscBTDestroy(&btvi));
  if (print) {
    IS tbz;

    CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,cum,vmarks,PETSC_COPY_VALUES,&tbz));
    CHKERRQ(PetscObjectSetName((PetscObject)tbz,"corners_to_be_zeroed_with_interior"));
    CHKERRQ(ISView(tbz,NULL));
    CHKERRQ(ISDestroy(&tbz));
  }
  CHKERRQ(MatZeroRows(lGt,cum,vmarks,0.,NULL,NULL));
  CHKERRQ(PetscFree(vmarks));
  CHKERRQ(PetscSFDestroy(&sfv));
  CHKERRQ(PetscFree2(sfvleaves,sfvroots));

  /* Recompute G */
  CHKERRQ(MatDestroy(&lG));
  CHKERRQ(MatTranspose(lGt,MAT_INITIAL_MATRIX,&lG));
  if (print) {
    CHKERRQ(PetscObjectSetName((PetscObject)lG,"used_lG"));
    CHKERRQ(MatView(lG,NULL));
    CHKERRQ(PetscObjectSetName((PetscObject)lGt,"used_lGt"));
    CHKERRQ(MatView(lGt,NULL));
  }

  /* Get primal dofs (if any) */
  cum = 0;
  for (i=0;i<ne;i++) {
    if (PetscUnlikely(PetscBTLookup(bte,i))) marks[cum++] = i;
  }
  if (fl2g) {
    CHKERRQ(ISLocalToGlobalMappingApply(fl2g,cum,marks,marks));
  }
  CHKERRQ(ISCreateGeneral(comm,cum,marks,PETSC_COPY_VALUES,&primals));
  if (print) {
    CHKERRQ(PetscObjectSetName((PetscObject)primals,"prescribed_primal_dofs"));
    CHKERRQ(ISView(primals,NULL));
  }
  CHKERRQ(PetscBTDestroy(&bte));
  /* TODO: what if the user passed in some of them ?  */
  CHKERRQ(PCBDDCSetPrimalVerticesLocalIS(pc,primals));
  CHKERRQ(ISDestroy(&primals));

  /* Compute edge connectivity */
  CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject)lG,"econn_"));

  /* Symbolic conn = lG*lGt */
  CHKERRQ(MatProductCreate(lG,lGt,NULL,&conn));
  CHKERRQ(MatProductSetType(conn,MATPRODUCT_AB));
  CHKERRQ(MatProductSetAlgorithm(conn,"default"));
  CHKERRQ(MatProductSetFill(conn,PETSC_DEFAULT));
  CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject)conn,"econn_"));
  CHKERRQ(MatProductSetFromOptions(conn));
  CHKERRQ(MatProductSymbolic(conn));

  CHKERRQ(MatGetRowIJ(conn,0,PETSC_FALSE,PETSC_FALSE,&i,&ii,&jj,&done));
  if (fl2g) {
    PetscBT   btf;
    PetscInt  *iia,*jja,*iiu,*jju;
    PetscBool rest = PETSC_FALSE,free = PETSC_FALSE;

    /* create CSR for all local dofs */
    CHKERRQ(PetscMalloc1(n+1,&iia));
    if (pcbddc->mat_graph->nvtxs_csr) { /* the user has passed in a CSR graph */
      PetscCheckFalse(pcbddc->mat_graph->nvtxs_csr != n,PETSC_COMM_SELF,PETSC_ERR_USER,"Invalid size of CSR graph %D. Should be %D",pcbddc->mat_graph->nvtxs_csr,n);
      iiu = pcbddc->mat_graph->xadj;
      jju = pcbddc->mat_graph->adjncy;
    } else if (pcbddc->use_local_adj) {
      rest = PETSC_TRUE;
      CHKERRQ(MatGetRowIJ(matis->A,0,PETSC_TRUE,PETSC_FALSE,&i,(const PetscInt**)&iiu,(const PetscInt**)&jju,&done));
    } else {
      free   = PETSC_TRUE;
      CHKERRQ(PetscMalloc2(n+1,&iiu,n,&jju));
      iiu[0] = 0;
      for (i=0;i<n;i++) {
        iiu[i+1] = i+1;
        jju[i]   = -1;
      }
    }

    /* import sizes of CSR */
    iia[0] = 0;
    for (i=0;i<n;i++) iia[i+1] = iiu[i+1]-iiu[i];

    /* overwrite entries corresponding to the Nedelec field */
    CHKERRQ(PetscBTCreate(n,&btf));
    CHKERRQ(ISGetIndices(nedfieldlocal,&idxs));
    for (i=0;i<ne;i++) {
      CHKERRQ(PetscBTSet(btf,idxs[i]));
      iia[idxs[i]+1] = ii[i+1]-ii[i];
    }

    /* iia in CSR */
    for (i=0;i<n;i++) iia[i+1] += iia[i];

    /* jja in CSR */
    CHKERRQ(PetscMalloc1(iia[n],&jja));
    for (i=0;i<n;i++)
      if (!PetscBTLookup(btf,i))
        for (j=0;j<iiu[i+1]-iiu[i];j++)
          jja[iia[i]+j] = jju[iiu[i]+j];

    /* map edge dofs connectivity */
    if (jj) {
      CHKERRQ(ISLocalToGlobalMappingApply(fl2g,ii[ne],jj,(PetscInt *)jj));
      for (i=0;i<ne;i++) {
        PetscInt e = idxs[i];
        for (j=0;j<ii[i+1]-ii[i];j++) jja[iia[e]+j] = jj[ii[i]+j];
      }
    }
    CHKERRQ(ISRestoreIndices(nedfieldlocal,&idxs));
    CHKERRQ(PCBDDCSetLocalAdjacencyGraph(pc,n,iia,jja,PETSC_OWN_POINTER));
    if (rest) {
      CHKERRQ(MatRestoreRowIJ(matis->A,0,PETSC_TRUE,PETSC_FALSE,&i,(const PetscInt**)&iiu,(const PetscInt**)&jju,&done));
    }
    if (free) {
      CHKERRQ(PetscFree2(iiu,jju));
    }
    CHKERRQ(PetscBTDestroy(&btf));
  } else {
    CHKERRQ(PCBDDCSetLocalAdjacencyGraph(pc,n,ii,jj,PETSC_USE_POINTER));
  }

  /* Analyze interface for edge dofs */
  CHKERRQ(PCBDDCAnalyzeInterface(pc));
  pcbddc->mat_graph->twodim = PETSC_FALSE;

  /* Get coarse edges in the edge space */
  CHKERRQ(PCBDDCGraphGetCandidatesIS(pcbddc->mat_graph,NULL,NULL,&nee,&alleedges,&allprimals));
  CHKERRQ(MatRestoreRowIJ(conn,0,PETSC_FALSE,PETSC_FALSE,&i,&ii,&jj,&done));

  if (fl2g) {
    CHKERRQ(ISGlobalToLocalMappingApplyIS(fl2g,IS_GTOLM_DROP,allprimals,&primals));
    CHKERRQ(PetscMalloc1(nee,&eedges));
    for (i=0;i<nee;i++) {
      CHKERRQ(ISGlobalToLocalMappingApplyIS(fl2g,IS_GTOLM_DROP,alleedges[i],&eedges[i]));
    }
  } else {
    eedges  = alleedges;
    primals = allprimals;
  }

  /* Mark fine edge dofs with their coarse edge id */
  CHKERRQ(PetscArrayzero(marks,ne));
  CHKERRQ(ISGetLocalSize(primals,&cum));
  CHKERRQ(ISGetIndices(primals,&idxs));
  for (i=0;i<cum;i++) marks[idxs[i]] = nee+1;
  CHKERRQ(ISRestoreIndices(primals,&idxs));
  if (print) {
    CHKERRQ(PetscObjectSetName((PetscObject)primals,"obtained_primal_dofs"));
    CHKERRQ(ISView(primals,NULL));
  }

  maxsize = 0;
  for (i=0;i<nee;i++) {
    PetscInt size,mark = i+1;

    CHKERRQ(ISGetLocalSize(eedges[i],&size));
    CHKERRQ(ISGetIndices(eedges[i],&idxs));
    for (j=0;j<size;j++) marks[idxs[j]] = mark;
    CHKERRQ(ISRestoreIndices(eedges[i],&idxs));
    maxsize = PetscMax(maxsize,size);
  }

  /* Find coarse edge endpoints */
  CHKERRQ(MatGetRowIJ(lG,0,PETSC_FALSE,PETSC_FALSE,&i,&ii,&jj,&done));
  CHKERRQ(MatGetRowIJ(lGt,0,PETSC_FALSE,PETSC_FALSE,&i,&iit,&jjt,&done));
  for (i=0;i<nee;i++) {
    PetscInt mark = i+1,size;

    CHKERRQ(ISGetLocalSize(eedges[i],&size));
    if (!size && nedfieldlocal) continue;
    PetscCheckFalse(!size,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unexpected zero sized edge %D",i);
    CHKERRQ(ISGetIndices(eedges[i],&idxs));
    if (print) {
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"ENDPOINTS ANALYSIS EDGE %D\n",i));
      CHKERRQ(ISView(eedges[i],NULL));
    }
    for (j=0;j<size;j++) {
      PetscInt k, ee = idxs[j];
      if (print) PetscPrintf(PETSC_COMM_SELF,"  idx %D\n",ee);
      for (k=ii[ee];k<ii[ee+1];k++) {
        if (print) PetscPrintf(PETSC_COMM_SELF,"    inspect %D\n",jj[k]);
        if (PetscBTLookup(btv,jj[k])) {
          if (print) PetscPrintf(PETSC_COMM_SELF,"      corner found (already set) %D\n",jj[k]);
        } else if (PetscBTLookup(btvcand,jj[k])) { /* is it ok? */
          PetscInt  k2;
          PetscBool corner = PETSC_FALSE;
          for (k2 = iit[jj[k]];k2 < iit[jj[k]+1];k2++) {
            if (print) PetscPrintf(PETSC_COMM_SELF,"        INSPECTING %D: mark %D (ref mark %D), boundary %D\n",jjt[k2],marks[jjt[k2]],mark,!!PetscBTLookup(btb,jjt[k2]));
            /* it's a corner if either is connected with an edge dof belonging to a different cc or
               if the edge dof lie on the natural part of the boundary */
            if ((marks[jjt[k2]] && marks[jjt[k2]] != mark) || (!marks[jjt[k2]] && PetscBTLookup(btb,jjt[k2]))) {
              corner = PETSC_TRUE;
              break;
            }
          }
          if (corner) { /* found the nodal dof corresponding to the endpoint of the edge */
            if (print) PetscPrintf(PETSC_COMM_SELF,"        corner found %D\n",jj[k]);
            CHKERRQ(PetscBTSet(btv,jj[k]));
          } else {
            if (print) PetscPrintf(PETSC_COMM_SELF,"        no corners found\n");
          }
        }
      }
    }
    CHKERRQ(ISRestoreIndices(eedges[i],&idxs));
  }
  CHKERRQ(MatRestoreRowIJ(lGt,0,PETSC_FALSE,PETSC_FALSE,&i,&iit,&jjt,&done));
  CHKERRQ(MatRestoreRowIJ(lG,0,PETSC_FALSE,PETSC_FALSE,&i,&ii,&jj,&done));
  CHKERRQ(PetscBTDestroy(&btb));

  /* Reset marked primal dofs */
  CHKERRQ(ISGetLocalSize(primals,&cum));
  CHKERRQ(ISGetIndices(primals,&idxs));
  for (i=0;i<cum;i++) marks[idxs[i]] = 0;
  CHKERRQ(ISRestoreIndices(primals,&idxs));

  /* Now use the initial lG */
  CHKERRQ(MatDestroy(&lG));
  CHKERRQ(MatDestroy(&lGt));
  lG   = lGinit;
  CHKERRQ(MatTranspose(lG,MAT_INITIAL_MATRIX,&lGt));

  /* Compute extended cols indices */
  CHKERRQ(PetscBTCreate(nv,&btvc));
  CHKERRQ(PetscBTCreate(nee,&bter));
  CHKERRQ(MatGetRowIJ(lG,0,PETSC_FALSE,PETSC_FALSE,&i,&ii,&jj,&done));
  CHKERRQ(MatSeqAIJGetMaxRowNonzeros(lG,&i));
  i   *= maxsize;
  CHKERRQ(PetscCalloc1(nee,&extcols));
  CHKERRQ(PetscMalloc2(i,&extrow,i,&gidxs));
  eerr = PETSC_FALSE;
  for (i=0;i<nee;i++) {
    PetscInt size,found = 0;

    cum  = 0;
    CHKERRQ(ISGetLocalSize(eedges[i],&size));
    if (!size && nedfieldlocal) continue;
    PetscCheckFalse(!size,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unexpected zero sized edge %D",i);
    CHKERRQ(ISGetIndices(eedges[i],&idxs));
    CHKERRQ(PetscBTMemzero(nv,btvc));
    for (j=0;j<size;j++) {
      PetscInt k,ee = idxs[j];
      for (k=ii[ee];k<ii[ee+1];k++) {
        PetscInt vv = jj[k];
        if (!PetscBTLookup(btv,vv)) extrow[cum++] = vv;
        else if (!PetscBTLookupSet(btvc,vv)) found++;
      }
    }
    CHKERRQ(ISRestoreIndices(eedges[i],&idxs));
    CHKERRQ(PetscSortRemoveDupsInt(&cum,extrow));
    CHKERRQ(ISLocalToGlobalMappingApply(vl2g,cum,extrow,gidxs));
    CHKERRQ(PetscSortIntWithArray(cum,gidxs,extrow));
    CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,cum,extrow,PETSC_COPY_VALUES,&extcols[i]));
    /* it may happen that endpoints are not defined at this point
       if it is the case, mark this edge for a second pass */
    if (cum != size -1 || found != 2) {
      CHKERRQ(PetscBTSet(bter,i));
      if (print) {
        CHKERRQ(PetscObjectSetName((PetscObject)eedges[i],"error_edge"));
        CHKERRQ(ISView(eedges[i],NULL));
        CHKERRQ(PetscObjectSetName((PetscObject)extcols[i],"error_extcol"));
        CHKERRQ(ISView(extcols[i],NULL));
      }
      eerr = PETSC_TRUE;
    }
  }
  /* PetscCheckFalse(eerr,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unexpected SIZE OF EDGE > EXTCOL FIRST PASS"); */
  CHKERRMPI(MPIU_Allreduce(&eerr,&done,1,MPIU_BOOL,MPI_LOR,comm));
  if (done) {
    PetscInt *newprimals;

    CHKERRQ(PetscMalloc1(ne,&newprimals));
    CHKERRQ(ISGetLocalSize(primals,&cum));
    CHKERRQ(ISGetIndices(primals,&idxs));
    CHKERRQ(PetscArraycpy(newprimals,idxs,cum));
    CHKERRQ(ISRestoreIndices(primals,&idxs));
    CHKERRQ(MatGetRowIJ(lGt,0,PETSC_FALSE,PETSC_FALSE,&i,&iit,&jjt,&done));
    if (print) PetscPrintf(PETSC_COMM_SELF,"DOING SECOND PASS (eerr %D)\n",eerr);
    for (i=0;i<nee;i++) {
      PetscBool has_candidates = PETSC_FALSE;
      if (PetscBTLookup(bter,i)) {
        PetscInt size,mark = i+1;

        CHKERRQ(ISGetLocalSize(eedges[i],&size));
        CHKERRQ(ISGetIndices(eedges[i],&idxs));
        /* for (j=0;j<size;j++) newprimals[cum++] = idxs[j]; */
        for (j=0;j<size;j++) {
          PetscInt k,ee = idxs[j];
          if (print) PetscPrintf(PETSC_COMM_SELF,"Inspecting edge dof %D [%D %D)\n",ee,ii[ee],ii[ee+1]);
          for (k=ii[ee];k<ii[ee+1];k++) {
            /* set all candidates located on the edge as corners */
            if (PetscBTLookup(btvcand,jj[k])) {
              PetscInt k2,vv = jj[k];
              has_candidates = PETSC_TRUE;
              if (print) PetscPrintf(PETSC_COMM_SELF,"  Candidate set to vertex %D\n",vv);
              CHKERRQ(PetscBTSet(btv,vv));
              /* set all edge dofs connected to candidate as primals */
              for (k2=iit[vv];k2<iit[vv+1];k2++) {
                if (marks[jjt[k2]] == mark) {
                  PetscInt k3,ee2 = jjt[k2];
                  if (print) PetscPrintf(PETSC_COMM_SELF,"    Connected edge dof set to primal %D\n",ee2);
                  newprimals[cum++] = ee2;
                  /* finally set the new corners */
                  for (k3=ii[ee2];k3<ii[ee2+1];k3++) {
                    if (print) PetscPrintf(PETSC_COMM_SELF,"      Connected nodal dof set to vertex %D\n",jj[k3]);
                    CHKERRQ(PetscBTSet(btv,jj[k3]));
                  }
                }
              }
            } else {
              if (print) PetscPrintf(PETSC_COMM_SELF,"  Not a candidate vertex %D\n",jj[k]);
            }
          }
        }
        if (!has_candidates) { /* circular edge */
          PetscInt k, ee = idxs[0],*tmarks;

          CHKERRQ(PetscCalloc1(ne,&tmarks));
          if (print) PetscPrintf(PETSC_COMM_SELF,"  Circular edge %D\n",i);
          for (k=ii[ee];k<ii[ee+1];k++) {
            PetscInt k2;
            if (print) PetscPrintf(PETSC_COMM_SELF,"    Set to corner %D\n",jj[k]);
            CHKERRQ(PetscBTSet(btv,jj[k]));
            for (k2=iit[jj[k]];k2<iit[jj[k]+1];k2++) tmarks[jjt[k2]]++;
          }
          for (j=0;j<size;j++) {
            if (tmarks[idxs[j]] > 1) {
              if (print) PetscPrintf(PETSC_COMM_SELF,"  Edge dof set to primal %D\n",idxs[j]);
              newprimals[cum++] = idxs[j];
            }
          }
          CHKERRQ(PetscFree(tmarks));
        }
        CHKERRQ(ISRestoreIndices(eedges[i],&idxs));
      }
      CHKERRQ(ISDestroy(&extcols[i]));
    }
    CHKERRQ(PetscFree(extcols));
    CHKERRQ(MatRestoreRowIJ(lGt,0,PETSC_FALSE,PETSC_FALSE,&i,&iit,&jjt,&done));
    CHKERRQ(PetscSortRemoveDupsInt(&cum,newprimals));
    if (fl2g) {
      CHKERRQ(ISLocalToGlobalMappingApply(fl2g,cum,newprimals,newprimals));
      CHKERRQ(ISDestroy(&primals));
      for (i=0;i<nee;i++) {
        CHKERRQ(ISDestroy(&eedges[i]));
      }
      CHKERRQ(PetscFree(eedges));
    }
    CHKERRQ(PCBDDCGraphRestoreCandidatesIS(pcbddc->mat_graph,NULL,NULL,&nee,&alleedges,&allprimals));
    CHKERRQ(ISCreateGeneral(comm,cum,newprimals,PETSC_COPY_VALUES,&primals));
    CHKERRQ(PetscFree(newprimals));
    CHKERRQ(PCBDDCSetPrimalVerticesLocalIS(pc,primals));
    CHKERRQ(ISDestroy(&primals));
    CHKERRQ(PCBDDCAnalyzeInterface(pc));
    pcbddc->mat_graph->twodim = PETSC_FALSE;
    CHKERRQ(PCBDDCGraphGetCandidatesIS(pcbddc->mat_graph,NULL,NULL,&nee,&alleedges,&allprimals));
    if (fl2g) {
      CHKERRQ(ISGlobalToLocalMappingApplyIS(fl2g,IS_GTOLM_DROP,allprimals,&primals));
      CHKERRQ(PetscMalloc1(nee,&eedges));
      for (i=0;i<nee;i++) {
        CHKERRQ(ISGlobalToLocalMappingApplyIS(fl2g,IS_GTOLM_DROP,alleedges[i],&eedges[i]));
      }
    } else {
      eedges  = alleedges;
      primals = allprimals;
    }
    CHKERRQ(PetscCalloc1(nee,&extcols));

    /* Mark again */
    CHKERRQ(PetscArrayzero(marks,ne));
    for (i=0;i<nee;i++) {
      PetscInt size,mark = i+1;

      CHKERRQ(ISGetLocalSize(eedges[i],&size));
      CHKERRQ(ISGetIndices(eedges[i],&idxs));
      for (j=0;j<size;j++) marks[idxs[j]] = mark;
      CHKERRQ(ISRestoreIndices(eedges[i],&idxs));
    }
    if (print) {
      CHKERRQ(PetscObjectSetName((PetscObject)primals,"obtained_primal_dofs_secondpass"));
      CHKERRQ(ISView(primals,NULL));
    }

    /* Recompute extended cols */
    eerr = PETSC_FALSE;
    for (i=0;i<nee;i++) {
      PetscInt size;

      cum  = 0;
      CHKERRQ(ISGetLocalSize(eedges[i],&size));
      if (!size && nedfieldlocal) continue;
      PetscCheckFalse(!size,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unexpected zero sized edge %D",i);
      CHKERRQ(ISGetIndices(eedges[i],&idxs));
      for (j=0;j<size;j++) {
        PetscInt k,ee = idxs[j];
        for (k=ii[ee];k<ii[ee+1];k++) if (!PetscBTLookup(btv,jj[k])) extrow[cum++] = jj[k];
      }
      CHKERRQ(ISRestoreIndices(eedges[i],&idxs));
      CHKERRQ(PetscSortRemoveDupsInt(&cum,extrow));
      CHKERRQ(ISLocalToGlobalMappingApply(vl2g,cum,extrow,gidxs));
      CHKERRQ(PetscSortIntWithArray(cum,gidxs,extrow));
      CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,cum,extrow,PETSC_COPY_VALUES,&extcols[i]));
      if (cum != size -1) {
        if (print) {
          CHKERRQ(PetscObjectSetName((PetscObject)eedges[i],"error_edge_secondpass"));
          CHKERRQ(ISView(eedges[i],NULL));
          CHKERRQ(PetscObjectSetName((PetscObject)extcols[i],"error_extcol_secondpass"));
          CHKERRQ(ISView(extcols[i],NULL));
        }
        eerr = PETSC_TRUE;
      }
    }
  }
  CHKERRQ(MatRestoreRowIJ(lG,0,PETSC_FALSE,PETSC_FALSE,&i,&ii,&jj,&done));
  CHKERRQ(PetscFree2(extrow,gidxs));
  CHKERRQ(PetscBTDestroy(&bter));
  if (print) CHKERRQ(PCBDDCGraphASCIIView(pcbddc->mat_graph,5,PETSC_VIEWER_STDOUT_SELF));
  /* an error should not occur at this point */
  PetscCheckFalse(eerr,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unexpected SIZE OF EDGE > EXTCOL SECOND PASS");

  /* Check the number of endpoints */
  CHKERRQ(MatGetRowIJ(lG,0,PETSC_FALSE,PETSC_FALSE,&i,&ii,&jj,&done));
  CHKERRQ(PetscMalloc1(2*nee,&corners));
  CHKERRQ(PetscMalloc1(nee,&cedges));
  for (i=0;i<nee;i++) {
    PetscInt size, found = 0, gc[2];

    /* init with defaults */
    cedges[i] = corners[i*2] = corners[i*2+1] = -1;
    CHKERRQ(ISGetLocalSize(eedges[i],&size));
    if (!size && nedfieldlocal) continue;
    PetscCheckFalse(!size,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unexpected zero sized edge %D",i);
    CHKERRQ(ISGetIndices(eedges[i],&idxs));
    CHKERRQ(PetscBTMemzero(nv,btvc));
    for (j=0;j<size;j++) {
      PetscInt k,ee = idxs[j];
      for (k=ii[ee];k<ii[ee+1];k++) {
        PetscInt vv = jj[k];
        if (PetscBTLookup(btv,vv) && !PetscBTLookupSet(btvc,vv)) {
          PetscCheckFalse(found == 2,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Found more then two corners for edge %D",i);
          corners[i*2+found++] = vv;
        }
      }
    }
    if (found != 2) {
      PetscInt e;
      if (fl2g) {
        CHKERRQ(ISLocalToGlobalMappingApply(fl2g,1,idxs,&e));
      } else {
        e = idxs[0];
      }
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Found %D corners for edge %D (astart %D, estart %D)",found,i,e,idxs[0]);
    }

    /* get primal dof index on this coarse edge */
    CHKERRQ(ISLocalToGlobalMappingApply(vl2g,2,corners+2*i,gc));
    if (gc[0] > gc[1]) {
      PetscInt swap  = corners[2*i];
      corners[2*i]   = corners[2*i+1];
      corners[2*i+1] = swap;
    }
    cedges[i] = idxs[size-1];
    CHKERRQ(ISRestoreIndices(eedges[i],&idxs));
    if (print) PetscPrintf(PETSC_COMM_SELF,"EDGE %D: ce %D, corners (%D,%D)\n",i,cedges[i],corners[2*i],corners[2*i+1]);
  }
  CHKERRQ(MatRestoreRowIJ(lG,0,PETSC_FALSE,PETSC_FALSE,&i,&ii,&jj,&done));
  CHKERRQ(PetscBTDestroy(&btvc));

  if (PetscDefined(USE_DEBUG)) {
    /* Inspects columns of lG (rows of lGt) and make sure the change of basis will
     not interfere with neighbouring coarse edges */
    CHKERRQ(PetscMalloc1(nee+1,&emarks));
    CHKERRQ(MatGetRowIJ(lGt,0,PETSC_FALSE,PETSC_FALSE,&i,&ii,&jj,&done));
    for (i=0;i<nv;i++) {
      PetscInt emax = 0,eemax = 0;

      if (ii[i+1]==ii[i] || PetscBTLookup(btv,i)) continue;
      CHKERRQ(PetscArrayzero(emarks,nee+1));
      for (j=ii[i];j<ii[i+1];j++) emarks[marks[jj[j]]]++;
      for (j=1;j<nee+1;j++) {
        if (emax < emarks[j]) {
          emax = emarks[j];
          eemax = j;
        }
      }
      /* not relevant for edges */
      if (!eemax) continue;

      for (j=ii[i];j<ii[i+1];j++) {
        if (marks[jj[j]] && marks[jj[j]] != eemax) {
          SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Found 2 coarse edges (id %D and %D) connected through the %D nodal dof at edge dof %D",marks[jj[j]]-1,eemax,i,jj[j]);
        }
      }
    }
    CHKERRQ(PetscFree(emarks));
    CHKERRQ(MatRestoreRowIJ(lGt,0,PETSC_FALSE,PETSC_FALSE,&i,&ii,&jj,&done));
  }

  /* Compute extended rows indices for edge blocks of the change of basis */
  CHKERRQ(MatGetRowIJ(lGt,0,PETSC_FALSE,PETSC_FALSE,&i,&ii,&jj,&done));
  CHKERRQ(MatSeqAIJGetMaxRowNonzeros(lGt,&extmem));
  extmem *= maxsize;
  CHKERRQ(PetscMalloc1(extmem*nee,&extrow));
  CHKERRQ(PetscMalloc1(nee,&extrows));
  CHKERRQ(PetscCalloc1(nee,&extrowcum));
  for (i=0;i<nv;i++) {
    PetscInt mark = 0,size,start;

    if (ii[i+1]==ii[i] || PetscBTLookup(btv,i)) continue;
    for (j=ii[i];j<ii[i+1];j++)
      if (marks[jj[j]] && !mark)
        mark = marks[jj[j]];

    /* not relevant */
    if (!mark) continue;

    /* import extended row */
    mark--;
    start = mark*extmem+extrowcum[mark];
    size = ii[i+1]-ii[i];
    PetscCheckFalse(extrowcum[mark] + size > extmem,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Not enough memory allocated %D > %D",extrowcum[mark] + size,extmem);
    CHKERRQ(PetscArraycpy(extrow+start,jj+ii[i],size));
    extrowcum[mark] += size;
  }
  CHKERRQ(MatRestoreRowIJ(lGt,0,PETSC_FALSE,PETSC_FALSE,&i,&ii,&jj,&done));
  CHKERRQ(MatDestroy(&lGt));
  CHKERRQ(PetscFree(marks));

  /* Compress extrows */
  cum  = 0;
  for (i=0;i<nee;i++) {
    PetscInt size = extrowcum[i],*start = extrow + i*extmem;
    CHKERRQ(PetscSortRemoveDupsInt(&size,start));
    CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,size,start,PETSC_USE_POINTER,&extrows[i]));
    cum  = PetscMax(cum,size);
  }
  CHKERRQ(PetscFree(extrowcum));
  CHKERRQ(PetscBTDestroy(&btv));
  CHKERRQ(PetscBTDestroy(&btvcand));

  /* Workspace for lapack inner calls and VecSetValues */
  CHKERRQ(PetscMalloc2((5+cum+maxsize)*maxsize,&work,maxsize,&rwork));

  /* Create change of basis matrix (preallocation can be improved) */
  CHKERRQ(MatCreate(comm,&T));
  ierr = MatSetSizes(T,pc->pmat->rmap->n,pc->pmat->rmap->n,
                       pc->pmat->rmap->N,pc->pmat->rmap->N);CHKERRQ(ierr);
  CHKERRQ(MatSetType(T,MATAIJ));
  CHKERRQ(MatSeqAIJSetPreallocation(T,10,NULL));
  CHKERRQ(MatMPIAIJSetPreallocation(T,10,NULL,10,NULL));
  CHKERRQ(MatSetLocalToGlobalMapping(T,al2g,al2g));
  CHKERRQ(MatSetOption(T,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE));
  CHKERRQ(MatSetOption(T,MAT_ROW_ORIENTED,PETSC_FALSE));
  CHKERRQ(ISLocalToGlobalMappingDestroy(&al2g));

  /* Defaults to identity */
  CHKERRQ(MatCreateVecs(pc->pmat,&tvec,NULL));
  CHKERRQ(VecSet(tvec,1.0));
  CHKERRQ(MatDiagonalSet(T,tvec,INSERT_VALUES));
  CHKERRQ(VecDestroy(&tvec));

  /* Create discrete gradient for the coarser level if needed */
  CHKERRQ(MatDestroy(&pcbddc->nedcG));
  CHKERRQ(ISDestroy(&pcbddc->nedclocal));
  if (pcbddc->current_level < pcbddc->max_levels) {
    ISLocalToGlobalMapping cel2g,cvl2g;
    IS                     wis,gwis;
    PetscInt               cnv,cne;

    CHKERRQ(ISCreateGeneral(comm,nee,cedges,PETSC_COPY_VALUES,&wis));
    if (fl2g) {
      CHKERRQ(ISLocalToGlobalMappingApplyIS(fl2g,wis,&pcbddc->nedclocal));
    } else {
      CHKERRQ(PetscObjectReference((PetscObject)wis));
      pcbddc->nedclocal = wis;
    }
    CHKERRQ(ISLocalToGlobalMappingApplyIS(el2g,wis,&gwis));
    CHKERRQ(ISDestroy(&wis));
    CHKERRQ(ISRenumber(gwis,NULL,&cne,&wis));
    CHKERRQ(ISLocalToGlobalMappingCreateIS(wis,&cel2g));
    CHKERRQ(ISDestroy(&wis));
    CHKERRQ(ISDestroy(&gwis));

    CHKERRQ(ISCreateGeneral(comm,2*nee,corners,PETSC_USE_POINTER,&wis));
    CHKERRQ(ISLocalToGlobalMappingApplyIS(vl2g,wis,&gwis));
    CHKERRQ(ISDestroy(&wis));
    CHKERRQ(ISRenumber(gwis,NULL,&cnv,&wis));
    CHKERRQ(ISLocalToGlobalMappingCreateIS(wis,&cvl2g));
    CHKERRQ(ISDestroy(&wis));
    CHKERRQ(ISDestroy(&gwis));

    CHKERRQ(MatCreate(comm,&pcbddc->nedcG));
    CHKERRQ(MatSetSizes(pcbddc->nedcG,PETSC_DECIDE,PETSC_DECIDE,cne,cnv));
    CHKERRQ(MatSetType(pcbddc->nedcG,MATAIJ));
    CHKERRQ(MatSeqAIJSetPreallocation(pcbddc->nedcG,2,NULL));
    CHKERRQ(MatMPIAIJSetPreallocation(pcbddc->nedcG,2,NULL,2,NULL));
    CHKERRQ(MatSetLocalToGlobalMapping(pcbddc->nedcG,cel2g,cvl2g));
    CHKERRQ(ISLocalToGlobalMappingDestroy(&cel2g));
    CHKERRQ(ISLocalToGlobalMappingDestroy(&cvl2g));
  }
  CHKERRQ(ISLocalToGlobalMappingDestroy(&vl2g));

#if defined(PRINT_GDET)
  inc = 0;
  lev = pcbddc->current_level;
#endif

  /* Insert values in the change of basis matrix */
  for (i=0;i<nee;i++) {
    Mat         Gins = NULL, GKins = NULL;
    IS          cornersis = NULL;
    PetscScalar cvals[2];

    if (pcbddc->nedcG) {
      CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,2,corners+2*i,PETSC_USE_POINTER,&cornersis));
    }
    CHKERRQ(PCBDDCComputeNedelecChangeEdge(lG,eedges[i],extrows[i],extcols[i],cornersis,&Gins,&GKins,cvals,work,rwork));
    if (Gins && GKins) {
      const PetscScalar *data;
      const PetscInt    *rows,*cols;
      PetscInt          nrh,nch,nrc,ncc;

      CHKERRQ(ISGetIndices(eedges[i],&cols));
      /* H1 */
      CHKERRQ(ISGetIndices(extrows[i],&rows));
      CHKERRQ(MatGetSize(Gins,&nrh,&nch));
      CHKERRQ(MatDenseGetArrayRead(Gins,&data));
      CHKERRQ(MatSetValuesLocal(T,nrh,rows,nch,cols,data,INSERT_VALUES));
      CHKERRQ(MatDenseRestoreArrayRead(Gins,&data));
      CHKERRQ(ISRestoreIndices(extrows[i],&rows));
      /* complement */
      CHKERRQ(MatGetSize(GKins,&nrc,&ncc));
      PetscCheckFalse(!ncc,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Constant function has not been generated for coarse edge %D",i);
      PetscCheckFalse(ncc + nch != nrc,PETSC_COMM_SELF,PETSC_ERR_PLIB,"The sum of the number of columns of GKins %D and Gins %D does not match %D for coarse edge %D",ncc,nch,nrc,i);
      PetscCheckFalse(ncc != 1 && pcbddc->nedcG,PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot generate the coarse discrete gradient for coarse edge %D with ncc %D",i,ncc);
      CHKERRQ(MatDenseGetArrayRead(GKins,&data));
      CHKERRQ(MatSetValuesLocal(T,nrc,cols,ncc,cols+nch,data,INSERT_VALUES));
      CHKERRQ(MatDenseRestoreArrayRead(GKins,&data));

      /* coarse discrete gradient */
      if (pcbddc->nedcG) {
        PetscInt cols[2];

        cols[0] = 2*i;
        cols[1] = 2*i+1;
        CHKERRQ(MatSetValuesLocal(pcbddc->nedcG,1,&i,2,cols,cvals,INSERT_VALUES));
      }
      CHKERRQ(ISRestoreIndices(eedges[i],&cols));
    }
    CHKERRQ(ISDestroy(&extrows[i]));
    CHKERRQ(ISDestroy(&extcols[i]));
    CHKERRQ(ISDestroy(&cornersis));
    CHKERRQ(MatDestroy(&Gins));
    CHKERRQ(MatDestroy(&GKins));
  }
  CHKERRQ(ISLocalToGlobalMappingDestroy(&el2g));

  /* Start assembling */
  CHKERRQ(MatAssemblyBegin(T,MAT_FINAL_ASSEMBLY));
  if (pcbddc->nedcG) {
    CHKERRQ(MatAssemblyBegin(pcbddc->nedcG,MAT_FINAL_ASSEMBLY));
  }

  /* Free */
  if (fl2g) {
    CHKERRQ(ISDestroy(&primals));
    for (i=0;i<nee;i++) {
      CHKERRQ(ISDestroy(&eedges[i]));
    }
    CHKERRQ(PetscFree(eedges));
  }

  /* hack mat_graph with primal dofs on the coarse edges */
  {
    PCBDDCGraph graph   = pcbddc->mat_graph;
    PetscInt    *oqueue = graph->queue;
    PetscInt    *ocptr  = graph->cptr;
    PetscInt    ncc,*idxs;

    /* find first primal edge */
    if (pcbddc->nedclocal) {
      CHKERRQ(ISGetIndices(pcbddc->nedclocal,(const PetscInt**)&idxs));
    } else {
      if (fl2g) {
        CHKERRQ(ISLocalToGlobalMappingApply(fl2g,nee,cedges,cedges));
      }
      idxs = cedges;
    }
    cum = 0;
    while (cum < nee && cedges[cum] < 0) cum++;

    /* adapt connected components */
    CHKERRQ(PetscMalloc2(graph->nvtxs+1,&graph->cptr,ocptr[graph->ncc],&graph->queue));
    graph->cptr[0] = 0;
    for (i=0,ncc=0;i<graph->ncc;i++) {
      PetscInt lc = ocptr[i+1]-ocptr[i];
      if (cum != nee && oqueue[ocptr[i+1]-1] == cedges[cum]) { /* this cc has a primal dof */
        graph->cptr[ncc+1] = graph->cptr[ncc]+1;
        graph->queue[graph->cptr[ncc]] = cedges[cum];
        ncc++;
        lc--;
        cum++;
        while (cum < nee && cedges[cum] < 0) cum++;
      }
      graph->cptr[ncc+1] = graph->cptr[ncc] + lc;
      for (j=0;j<lc;j++) graph->queue[graph->cptr[ncc]+j] = oqueue[ocptr[i]+j];
      ncc++;
    }
    graph->ncc = ncc;
    if (pcbddc->nedclocal) {
      CHKERRQ(ISRestoreIndices(pcbddc->nedclocal,(const PetscInt**)&idxs));
    }
    CHKERRQ(PetscFree2(ocptr,oqueue));
  }
  CHKERRQ(ISLocalToGlobalMappingDestroy(&fl2g));
  CHKERRQ(PCBDDCGraphRestoreCandidatesIS(pcbddc->mat_graph,NULL,NULL,&nee,&alleedges,&allprimals));
  CHKERRQ(PCBDDCGraphResetCSR(pcbddc->mat_graph));
  CHKERRQ(MatDestroy(&conn));

  CHKERRQ(ISDestroy(&nedfieldlocal));
  CHKERRQ(PetscFree(extrow));
  CHKERRQ(PetscFree2(work,rwork));
  CHKERRQ(PetscFree(corners));
  CHKERRQ(PetscFree(cedges));
  CHKERRQ(PetscFree(extrows));
  CHKERRQ(PetscFree(extcols));
  CHKERRQ(MatDestroy(&lG));

  /* Complete assembling */
  CHKERRQ(MatAssemblyEnd(T,MAT_FINAL_ASSEMBLY));
  if (pcbddc->nedcG) {
    CHKERRQ(MatAssemblyEnd(pcbddc->nedcG,MAT_FINAL_ASSEMBLY));
#if 0
    CHKERRQ(PetscObjectSetName((PetscObject)pcbddc->nedcG,"coarse_G"));
    CHKERRQ(MatView(pcbddc->nedcG,NULL));
#endif
  }

  /* set change of basis */
  CHKERRQ(PCBDDCSetChangeOfBasisMat(pc,T,singular));
  CHKERRQ(MatDestroy(&T));

  PetscFunctionReturn(0);
}

/* the near-null space of BDDC carries information on quadrature weights,
   and these can be collinear -> so cheat with MatNullSpaceCreate
   and create a suitable set of basis vectors first */
PetscErrorCode PCBDDCNullSpaceCreate(MPI_Comm comm, PetscBool has_const, PetscInt nvecs, Vec quad_vecs[], MatNullSpace *nnsp)
{
  PetscInt       i;

  PetscFunctionBegin;
  for (i=0;i<nvecs;i++) {
    PetscInt first,last;

    CHKERRQ(VecGetOwnershipRange(quad_vecs[i],&first,&last));
    PetscCheckFalse(last-first < 2*nvecs && has_const,PETSC_COMM_SELF,PETSC_ERR_SUP,"Not implemented");
    if (i>=first && i < last) {
      PetscScalar *data;
      CHKERRQ(VecGetArray(quad_vecs[i],&data));
      if (!has_const) {
        data[i-first] = 1.;
      } else {
        data[2*i-first] = 1./PetscSqrtReal(2.);
        data[2*i-first+1] = -1./PetscSqrtReal(2.);
      }
      CHKERRQ(VecRestoreArray(quad_vecs[i],&data));
    }
    CHKERRQ(PetscObjectStateIncrease((PetscObject)quad_vecs[i]));
  }
  CHKERRQ(MatNullSpaceCreate(comm,has_const,nvecs,quad_vecs,nnsp));
  for (i=0;i<nvecs;i++) { /* reset vectors */
    PetscInt first,last;
    CHKERRQ(VecLockReadPop(quad_vecs[i]));
    CHKERRQ(VecGetOwnershipRange(quad_vecs[i],&first,&last));
    if (i>=first && i < last) {
      PetscScalar *data;
      CHKERRQ(VecGetArray(quad_vecs[i],&data));
      if (!has_const) {
        data[i-first] = 0.;
      } else {
        data[2*i-first] = 0.;
        data[2*i-first+1] = 0.;
      }
      CHKERRQ(VecRestoreArray(quad_vecs[i],&data));
    }
    CHKERRQ(PetscObjectStateIncrease((PetscObject)quad_vecs[i]));
    CHKERRQ(VecLockReadPush(quad_vecs[i]));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCComputeNoNetFlux(Mat A, Mat divudotp, PetscBool transpose, IS vl2l, PCBDDCGraph graph, MatNullSpace *nnsp)
{
  Mat                    loc_divudotp;
  Vec                    p,v,vins,quad_vec,*quad_vecs;
  ISLocalToGlobalMapping map;
  PetscScalar            *vals;
  const PetscScalar      *array;
  PetscInt               i,maxneighs = 0,maxsize,*gidxs;
  PetscInt               n_neigh,*neigh,*n_shared,**shared;
  PetscMPIInt            rank;

  PetscFunctionBegin;
  CHKERRQ(ISLocalToGlobalMappingGetInfo(graph->l2gmap,&n_neigh,&neigh,&n_shared,&shared));
  for (i=0;i<n_neigh;i++) maxneighs = PetscMax(graph->count[shared[i][0]]+1,maxneighs);
  CHKERRMPI(MPIU_Allreduce(MPI_IN_PLACE,&maxneighs,1,MPIU_INT,MPI_MAX,PetscObjectComm((PetscObject)A)));
  if (!maxneighs) {
    CHKERRQ(ISLocalToGlobalMappingRestoreInfo(graph->l2gmap,&n_neigh,&neigh,&n_shared,&shared));
    *nnsp = NULL;
    PetscFunctionReturn(0);
  }
  maxsize = 0;
  for (i=0;i<n_neigh;i++) maxsize = PetscMax(n_shared[i],maxsize);
  CHKERRQ(PetscMalloc2(maxsize,&gidxs,maxsize,&vals));
  /* create vectors to hold quadrature weights */
  CHKERRQ(MatCreateVecs(A,&quad_vec,NULL));
  if (!transpose) {
    CHKERRQ(MatISGetLocalToGlobalMapping(A,&map,NULL));
  } else {
    CHKERRQ(MatISGetLocalToGlobalMapping(A,NULL,&map));
  }
  CHKERRQ(VecDuplicateVecs(quad_vec,maxneighs,&quad_vecs));
  CHKERRQ(VecDestroy(&quad_vec));
  CHKERRQ(PCBDDCNullSpaceCreate(PetscObjectComm((PetscObject)A),PETSC_FALSE,maxneighs,quad_vecs,nnsp));
  for (i=0;i<maxneighs;i++) {
    CHKERRQ(VecLockReadPop(quad_vecs[i]));
  }

  /* compute local quad vec */
  CHKERRQ(MatISGetLocalMat(divudotp,&loc_divudotp));
  if (!transpose) {
    CHKERRQ(MatCreateVecs(loc_divudotp,&v,&p));
  } else {
    CHKERRQ(MatCreateVecs(loc_divudotp,&p,&v));
  }
  CHKERRQ(VecSet(p,1.));
  if (!transpose) {
    CHKERRQ(MatMultTranspose(loc_divudotp,p,v));
  } else {
    CHKERRQ(MatMult(loc_divudotp,p,v));
  }
  if (vl2l) {
    Mat        lA;
    VecScatter sc;

    CHKERRQ(MatISGetLocalMat(A,&lA));
    CHKERRQ(MatCreateVecs(lA,&vins,NULL));
    CHKERRQ(VecScatterCreate(v,NULL,vins,vl2l,&sc));
    CHKERRQ(VecScatterBegin(sc,v,vins,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(sc,v,vins,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterDestroy(&sc));
  } else {
    vins = v;
  }
  CHKERRQ(VecGetArrayRead(vins,&array));
  CHKERRQ(VecDestroy(&p));

  /* insert in global quadrature vecs */
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)A),&rank));
  for (i=1;i<n_neigh;i++) {
    const PetscInt    *idxs;
    PetscInt          idx,nn,j;

    idxs = shared[i];
    nn   = n_shared[i];
    for (j=0;j<nn;j++) vals[j] = array[idxs[j]];
    CHKERRQ(PetscFindInt(rank,graph->count[idxs[0]],graph->neighbours_set[idxs[0]],&idx));
    idx  = -(idx+1);
    PetscCheckFalse(idx < 0 || idx >= maxneighs,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Invalid index %D not in [0,%D)",idx,maxneighs);
    CHKERRQ(ISLocalToGlobalMappingApply(map,nn,idxs,gidxs));
    CHKERRQ(VecSetValues(quad_vecs[idx],nn,gidxs,vals,INSERT_VALUES));
  }
  CHKERRQ(ISLocalToGlobalMappingRestoreInfo(graph->l2gmap,&n_neigh,&neigh,&n_shared,&shared));
  CHKERRQ(VecRestoreArrayRead(vins,&array));
  if (vl2l) {
    CHKERRQ(VecDestroy(&vins));
  }
  CHKERRQ(VecDestroy(&v));
  CHKERRQ(PetscFree2(gidxs,vals));

  /* assemble near null space */
  for (i=0;i<maxneighs;i++) {
    CHKERRQ(VecAssemblyBegin(quad_vecs[i]));
  }
  for (i=0;i<maxneighs;i++) {
    CHKERRQ(VecAssemblyEnd(quad_vecs[i]));
    CHKERRQ(VecViewFromOptions(quad_vecs[i],NULL,"-pc_bddc_quad_vecs_view"));
    CHKERRQ(VecLockReadPush(quad_vecs[i]));
  }
  CHKERRQ(VecDestroyVecs(maxneighs,&quad_vecs));
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCAddPrimalVerticesLocalIS(PC pc, IS primalv)
{
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;

  PetscFunctionBegin;
  if (primalv) {
    if (pcbddc->user_primal_vertices_local) {
      IS list[2], newp;

      list[0] = primalv;
      list[1] = pcbddc->user_primal_vertices_local;
      CHKERRQ(ISConcatenate(PetscObjectComm((PetscObject)pc),2,list,&newp));
      CHKERRQ(ISSortRemoveDups(newp));
      CHKERRQ(ISDestroy(&list[1]));
      pcbddc->user_primal_vertices_local = newp;
    } else {
      CHKERRQ(PCBDDCSetPrimalVerticesLocalIS(pc,primalv));
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode func_coords_private(PetscInt dim, PetscReal t, const PetscReal X[], PetscInt Nf, PetscScalar *out, void *ctx)
{
  PetscInt f, *comp  = (PetscInt *)ctx;

  PetscFunctionBegin;
  for (f=0;f<Nf;f++) out[f] = X[*comp];
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCComputeLocalTopologyInfo(PC pc)
{
  PetscErrorCode ierr;
  Vec            local,global;
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;
  Mat_IS         *matis = (Mat_IS*)pc->pmat->data;
  PetscBool      monolithic = PETSC_FALSE;

  PetscFunctionBegin;
  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)pc),((PetscObject)pc)->prefix,"BDDC topology options","PC");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsBool("-pc_bddc_monolithic","Discard any information on dofs splitting",NULL,monolithic,&monolithic,NULL));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  /* need to convert from global to local topology information and remove references to information in global ordering */
  CHKERRQ(MatCreateVecs(pc->pmat,&global,NULL));
  CHKERRQ(MatCreateVecs(matis->A,&local,NULL));
  CHKERRQ(VecBindToCPU(global,PETSC_TRUE));
  CHKERRQ(VecBindToCPU(local,PETSC_TRUE));
  if (monolithic) { /* just get block size to properly compute vertices */
    if (pcbddc->vertex_size == 1) {
      CHKERRQ(MatGetBlockSize(pc->pmat,&pcbddc->vertex_size));
    }
    goto boundary;
  }

  if (pcbddc->user_provided_isfordofs) {
    if (pcbddc->n_ISForDofs) {
      PetscInt i;

      CHKERRQ(PetscMalloc1(pcbddc->n_ISForDofs,&pcbddc->ISForDofsLocal));
      for (i=0;i<pcbddc->n_ISForDofs;i++) {
        PetscInt bs;

        CHKERRQ(PCBDDCGlobalToLocal(matis->rctx,global,local,pcbddc->ISForDofs[i],&pcbddc->ISForDofsLocal[i]));
        CHKERRQ(ISGetBlockSize(pcbddc->ISForDofs[i],&bs));
        CHKERRQ(ISSetBlockSize(pcbddc->ISForDofsLocal[i],bs));
        CHKERRQ(ISDestroy(&pcbddc->ISForDofs[i]));
      }
      pcbddc->n_ISForDofsLocal = pcbddc->n_ISForDofs;
      pcbddc->n_ISForDofs = 0;
      CHKERRQ(PetscFree(pcbddc->ISForDofs));
    }
  } else {
    if (!pcbddc->n_ISForDofsLocal) { /* field split not present */
      DM dm;

      CHKERRQ(MatGetDM(pc->pmat, &dm));
      if (!dm) {
        CHKERRQ(PCGetDM(pc, &dm));
      }
      if (dm) {
        IS      *fields;
        PetscInt nf,i;

        CHKERRQ(DMCreateFieldDecomposition(dm,&nf,NULL,&fields,NULL));
        CHKERRQ(PetscMalloc1(nf,&pcbddc->ISForDofsLocal));
        for (i=0;i<nf;i++) {
          PetscInt bs;

          CHKERRQ(PCBDDCGlobalToLocal(matis->rctx,global,local,fields[i],&pcbddc->ISForDofsLocal[i]));
          CHKERRQ(ISGetBlockSize(fields[i],&bs));
          CHKERRQ(ISSetBlockSize(pcbddc->ISForDofsLocal[i],bs));
          CHKERRQ(ISDestroy(&fields[i]));
        }
        CHKERRQ(PetscFree(fields));
        pcbddc->n_ISForDofsLocal = nf;
      } else { /* See if MATIS has fields attached by the conversion from MatNest */
        PetscContainer   c;

        CHKERRQ(PetscObjectQuery((PetscObject)pc->pmat,"_convert_nest_lfields",(PetscObject*)&c));
        if (c) {
          MatISLocalFields lf;
          CHKERRQ(PetscContainerGetPointer(c,(void**)&lf));
          CHKERRQ(PCBDDCSetDofsSplittingLocal(pc,lf->nr,lf->rf));
        } else { /* fallback, create the default fields if bs > 1 */
          PetscInt i, n = matis->A->rmap->n;
          CHKERRQ(MatGetBlockSize(pc->pmat,&i));
          if (i > 1) {
            pcbddc->n_ISForDofsLocal = i;
            CHKERRQ(PetscMalloc1(pcbddc->n_ISForDofsLocal,&pcbddc->ISForDofsLocal));
            for (i=0;i<pcbddc->n_ISForDofsLocal;i++) {
              CHKERRQ(ISCreateStride(PetscObjectComm((PetscObject)pc),n/pcbddc->n_ISForDofsLocal,i,pcbddc->n_ISForDofsLocal,&pcbddc->ISForDofsLocal[i]));
            }
          }
        }
      }
    } else {
      PetscInt i;
      for (i=0;i<pcbddc->n_ISForDofsLocal;i++) {
        CHKERRQ(PCBDDCConsistencyCheckIS(pc,MPI_LAND,&pcbddc->ISForDofsLocal[i]));
      }
    }
  }

boundary:
  if (!pcbddc->DirichletBoundariesLocal && pcbddc->DirichletBoundaries) {
    CHKERRQ(PCBDDCGlobalToLocal(matis->rctx,global,local,pcbddc->DirichletBoundaries,&pcbddc->DirichletBoundariesLocal));
  } else if (pcbddc->DirichletBoundariesLocal) {
    CHKERRQ(PCBDDCConsistencyCheckIS(pc,MPI_LAND,&pcbddc->DirichletBoundariesLocal));
  }
  if (!pcbddc->NeumannBoundariesLocal && pcbddc->NeumannBoundaries) {
    CHKERRQ(PCBDDCGlobalToLocal(matis->rctx,global,local,pcbddc->NeumannBoundaries,&pcbddc->NeumannBoundariesLocal));
  } else if (pcbddc->NeumannBoundariesLocal) {
    CHKERRQ(PCBDDCConsistencyCheckIS(pc,MPI_LOR,&pcbddc->NeumannBoundariesLocal));
  }
  if (!pcbddc->user_primal_vertices_local && pcbddc->user_primal_vertices) {
    CHKERRQ(PCBDDCGlobalToLocal(matis->rctx,global,local,pcbddc->user_primal_vertices,&pcbddc->user_primal_vertices_local));
  }
  CHKERRQ(VecDestroy(&global));
  CHKERRQ(VecDestroy(&local));
  /* detect local disconnected subdomains if requested (use matis->A) */
  if (pcbddc->detect_disconnected) {
    IS        primalv = NULL;
    PetscInt  i;
    PetscBool filter = pcbddc->detect_disconnected_filter;

    for (i=0;i<pcbddc->n_local_subs;i++) {
      CHKERRQ(ISDestroy(&pcbddc->local_subs[i]));
    }
    CHKERRQ(PetscFree(pcbddc->local_subs));
    CHKERRQ(PCBDDCDetectDisconnectedComponents(pc,filter,&pcbddc->n_local_subs,&pcbddc->local_subs,&primalv));
    CHKERRQ(PCBDDCAddPrimalVerticesLocalIS(pc,primalv));
    CHKERRQ(ISDestroy(&primalv));
  }
  /* early stage corner detection */
  {
    DM dm;

    CHKERRQ(MatGetDM(pc->pmat,&dm));
    if (!dm) {
      CHKERRQ(PCGetDM(pc,&dm));
    }
    if (dm) {
      PetscBool isda;

      CHKERRQ(PetscObjectTypeCompare((PetscObject)dm,DMDA,&isda));
      if (isda) {
        ISLocalToGlobalMapping l2l;
        IS                     corners;
        Mat                    lA;
        PetscBool              gl,lo;

        {
          Vec               cvec;
          const PetscScalar *coords;
          PetscInt          dof,n,cdim;
          PetscBool         memc = PETSC_TRUE;

          CHKERRQ(DMDAGetInfo(dm,NULL,NULL,NULL,NULL,NULL,NULL,NULL,&dof,NULL,NULL,NULL,NULL,NULL));
          CHKERRQ(DMGetCoordinates(dm,&cvec));
          CHKERRQ(VecGetLocalSize(cvec,&n));
          CHKERRQ(VecGetBlockSize(cvec,&cdim));
          n   /= cdim;
          CHKERRQ(PetscFree(pcbddc->mat_graph->coords));
          CHKERRQ(PetscMalloc1(dof*n*cdim,&pcbddc->mat_graph->coords));
          CHKERRQ(VecGetArrayRead(cvec,&coords));
#if defined(PETSC_USE_COMPLEX)
          memc = PETSC_FALSE;
#endif
          if (dof != 1) memc = PETSC_FALSE;
          if (memc) {
            CHKERRQ(PetscArraycpy(pcbddc->mat_graph->coords,coords,cdim*n*dof));
          } else { /* BDDC graph does not use any blocked information, we need to replicate the data */
            PetscReal *bcoords = pcbddc->mat_graph->coords;
            PetscInt  i, b, d;

            for (i=0;i<n;i++) {
              for (b=0;b<dof;b++) {
                for (d=0;d<cdim;d++) {
                  bcoords[i*dof*cdim + b*cdim + d] = PetscRealPart(coords[i*cdim+d]);
                }
              }
            }
          }
          CHKERRQ(VecRestoreArrayRead(cvec,&coords));
          pcbddc->mat_graph->cdim  = cdim;
          pcbddc->mat_graph->cnloc = dof*n;
          pcbddc->mat_graph->cloc  = PETSC_FALSE;
        }
        CHKERRQ(DMDAGetSubdomainCornersIS(dm,&corners));
        CHKERRQ(MatISGetLocalMat(pc->pmat,&lA));
        CHKERRQ(MatGetLocalToGlobalMapping(lA,&l2l,NULL));
        CHKERRQ(MatISRestoreLocalMat(pc->pmat,&lA));
        lo   = (PetscBool)(l2l && corners);
        CHKERRMPI(MPIU_Allreduce(&lo,&gl,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)pc)));
        if (gl) { /* From PETSc's DMDA */
          const PetscInt    *idx;
          PetscInt          dof,bs,*idxout,n;

          CHKERRQ(DMDAGetInfo(dm,NULL,NULL,NULL,NULL,NULL,NULL,NULL,&dof,NULL,NULL,NULL,NULL,NULL));
          CHKERRQ(ISLocalToGlobalMappingGetBlockSize(l2l,&bs));
          CHKERRQ(ISGetLocalSize(corners,&n));
          CHKERRQ(ISGetIndices(corners,&idx));
          if (bs == dof) {
            CHKERRQ(PetscMalloc1(n,&idxout));
            CHKERRQ(ISLocalToGlobalMappingApplyBlock(l2l,n,idx,idxout));
          } else { /* the original DMDA local-to-local map have been modified */
            PetscInt i,d;

            CHKERRQ(PetscMalloc1(dof*n,&idxout));
            for (i=0;i<n;i++) for (d=0;d<dof;d++) idxout[dof*i+d] = dof*idx[i]+d;
            CHKERRQ(ISLocalToGlobalMappingApply(l2l,dof*n,idxout,idxout));

            bs = 1;
            n *= dof;
          }
          CHKERRQ(ISRestoreIndices(corners,&idx));
          CHKERRQ(DMDARestoreSubdomainCornersIS(dm,&corners));
          CHKERRQ(ISCreateBlock(PetscObjectComm((PetscObject)pc),bs,n,idxout,PETSC_OWN_POINTER,&corners));
          CHKERRQ(PCBDDCAddPrimalVerticesLocalIS(pc,corners));
          CHKERRQ(ISDestroy(&corners));
          pcbddc->corner_selected  = PETSC_TRUE;
          pcbddc->corner_selection = PETSC_TRUE;
        }
        if (corners) {
          CHKERRQ(DMDARestoreSubdomainCornersIS(dm,&corners));
        }
      }
    }
  }
  if (pcbddc->corner_selection && !pcbddc->mat_graph->cdim) {
    DM dm;

    CHKERRQ(MatGetDM(pc->pmat,&dm));
    if (!dm) {
      CHKERRQ(PCGetDM(pc,&dm));
    }
    if (dm) { /* this can get very expensive, I need to find a faster alternative */
      Vec            vcoords;
      PetscSection   section;
      PetscReal      *coords;
      PetscInt       d,cdim,nl,nf,**ctxs;
      PetscErrorCode (**funcs)(PetscInt, PetscReal, const PetscReal *, PetscInt, PetscScalar *, void *);

      CHKERRQ(DMGetCoordinateDim(dm,&cdim));
      CHKERRQ(DMGetLocalSection(dm,&section));
      CHKERRQ(PetscSectionGetNumFields(section,&nf));
      CHKERRQ(DMCreateGlobalVector(dm,&vcoords));
      CHKERRQ(VecGetLocalSize(vcoords,&nl));
      CHKERRQ(PetscMalloc1(nl*cdim,&coords));
      CHKERRQ(PetscMalloc2(nf,&funcs,nf,&ctxs));
      CHKERRQ(PetscMalloc1(nf,&ctxs[0]));
      for (d=0;d<nf;d++) funcs[d] = func_coords_private;
      for (d=1;d<nf;d++) ctxs[d] = ctxs[d-1] + 1;
      for (d=0;d<cdim;d++) {
        PetscInt          i;
        const PetscScalar *v;

        for (i=0;i<nf;i++) ctxs[i][0] = d;
        CHKERRQ(DMProjectFunction(dm,0.0,funcs,(void**)ctxs,INSERT_VALUES,vcoords));
        CHKERRQ(VecGetArrayRead(vcoords,&v));
        for (i=0;i<nl;i++) coords[i*cdim+d] = PetscRealPart(v[i]);
        CHKERRQ(VecRestoreArrayRead(vcoords,&v));
      }
      CHKERRQ(VecDestroy(&vcoords));
      CHKERRQ(PCSetCoordinates(pc,cdim,nl,coords));
      CHKERRQ(PetscFree(coords));
      CHKERRQ(PetscFree(ctxs[0]));
      CHKERRQ(PetscFree2(funcs,ctxs));
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCConsistencyCheckIS(PC pc, MPI_Op mop, IS *is)
{
  Mat_IS          *matis = (Mat_IS*)(pc->pmat->data);
  IS              nis;
  const PetscInt  *idxs;
  PetscInt        i,nd,n = matis->A->rmap->n,*nidxs,nnd;

  PetscFunctionBegin;
  PetscCheckFalse(mop != MPI_LAND && mop != MPI_LOR,PetscObjectComm((PetscObject)(pc)),PETSC_ERR_SUP,"Supported are MPI_LAND and MPI_LOR");
  if (mop == MPI_LAND) {
    /* init rootdata with true */
    for (i=0;i<pc->pmat->rmap->n;i++) matis->sf_rootdata[i] = 1;
  } else {
    CHKERRQ(PetscArrayzero(matis->sf_rootdata,pc->pmat->rmap->n));
  }
  CHKERRQ(PetscArrayzero(matis->sf_leafdata,n));
  CHKERRQ(ISGetLocalSize(*is,&nd));
  CHKERRQ(ISGetIndices(*is,&idxs));
  for (i=0;i<nd;i++)
    if (-1 < idxs[i] && idxs[i] < n)
      matis->sf_leafdata[idxs[i]] = 1;
  CHKERRQ(ISRestoreIndices(*is,&idxs));
  CHKERRQ(PetscSFReduceBegin(matis->sf,MPIU_INT,matis->sf_leafdata,matis->sf_rootdata,mop));
  CHKERRQ(PetscSFReduceEnd(matis->sf,MPIU_INT,matis->sf_leafdata,matis->sf_rootdata,mop));
  CHKERRQ(PetscSFBcastBegin(matis->sf,MPIU_INT,matis->sf_rootdata,matis->sf_leafdata,MPI_REPLACE));
  CHKERRQ(PetscSFBcastEnd(matis->sf,MPIU_INT,matis->sf_rootdata,matis->sf_leafdata,MPI_REPLACE));
  if (mop == MPI_LAND) {
    CHKERRQ(PetscMalloc1(nd,&nidxs));
  } else {
    CHKERRQ(PetscMalloc1(n,&nidxs));
  }
  for (i=0,nnd=0;i<n;i++)
    if (matis->sf_leafdata[i])
      nidxs[nnd++] = i;
  CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)(*is)),nnd,nidxs,PETSC_OWN_POINTER,&nis));
  CHKERRQ(ISDestroy(is));
  *is  = nis;
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCBenignRemoveInterior(PC pc,Vec r,Vec z)
{
  PC_IS             *pcis = (PC_IS*)(pc->data);
  PC_BDDC           *pcbddc = (PC_BDDC*)(pc->data);

  PetscFunctionBegin;
  if (!pcbddc->benign_have_null) {
    PetscFunctionReturn(0);
  }
  if (pcbddc->ChangeOfBasisMatrix) {
    Vec swap;

    CHKERRQ(MatMultTranspose(pcbddc->ChangeOfBasisMatrix,r,pcbddc->work_change));
    swap = pcbddc->work_change;
    pcbddc->work_change = r;
    r = swap;
  }
  CHKERRQ(VecScatterBegin(pcis->global_to_D,r,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(pcis->global_to_D,r,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(PetscLogEventBegin(PC_BDDC_Solves[pcbddc->current_level][0],pc,0,0,0));
  CHKERRQ(KSPSolve(pcbddc->ksp_D,pcis->vec1_D,pcis->vec2_D));
  CHKERRQ(PetscLogEventEnd(PC_BDDC_Solves[pcbddc->current_level][0],pc,0,0,0));
  CHKERRQ(KSPCheckSolve(pcbddc->ksp_D,pc,pcis->vec2_D));
  CHKERRQ(VecSet(z,0.));
  CHKERRQ(VecScatterBegin(pcis->global_to_D,pcis->vec2_D,z,INSERT_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecScatterEnd(pcis->global_to_D,pcis->vec2_D,z,INSERT_VALUES,SCATTER_REVERSE));
  if (pcbddc->ChangeOfBasisMatrix) {
    pcbddc->work_change = r;
    CHKERRQ(VecCopy(z,pcbddc->work_change));
    CHKERRQ(MatMult(pcbddc->ChangeOfBasisMatrix,pcbddc->work_change,z));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCBenignMatMult_Private_Private(Mat A, Vec x, Vec y, PetscBool transpose)
{
  PCBDDCBenignMatMult_ctx ctx;
  PetscBool               apply_right,apply_left,reset_x;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(A,&ctx));
  if (transpose) {
    apply_right = ctx->apply_left;
    apply_left = ctx->apply_right;
  } else {
    apply_right = ctx->apply_right;
    apply_left = ctx->apply_left;
  }
  reset_x = PETSC_FALSE;
  if (apply_right) {
    const PetscScalar *ax;
    PetscInt          nl,i;

    CHKERRQ(VecGetLocalSize(x,&nl));
    CHKERRQ(VecGetArrayRead(x,&ax));
    CHKERRQ(PetscArraycpy(ctx->work,ax,nl));
    CHKERRQ(VecRestoreArrayRead(x,&ax));
    for (i=0;i<ctx->benign_n;i++) {
      PetscScalar    sum,val;
      const PetscInt *idxs;
      PetscInt       nz,j;
      CHKERRQ(ISGetLocalSize(ctx->benign_zerodiag_subs[i],&nz));
      CHKERRQ(ISGetIndices(ctx->benign_zerodiag_subs[i],&idxs));
      sum = 0.;
      if (ctx->apply_p0) {
        val = ctx->work[idxs[nz-1]];
        for (j=0;j<nz-1;j++) {
          sum += ctx->work[idxs[j]];
          ctx->work[idxs[j]] += val;
        }
      } else {
        for (j=0;j<nz-1;j++) {
          sum += ctx->work[idxs[j]];
        }
      }
      ctx->work[idxs[nz-1]] -= sum;
      CHKERRQ(ISRestoreIndices(ctx->benign_zerodiag_subs[i],&idxs));
    }
    CHKERRQ(VecPlaceArray(x,ctx->work));
    reset_x = PETSC_TRUE;
  }
  if (transpose) {
    CHKERRQ(MatMultTranspose(ctx->A,x,y));
  } else {
    CHKERRQ(MatMult(ctx->A,x,y));
  }
  if (reset_x) {
    CHKERRQ(VecResetArray(x));
  }
  if (apply_left) {
    PetscScalar *ay;
    PetscInt    i;

    CHKERRQ(VecGetArray(y,&ay));
    for (i=0;i<ctx->benign_n;i++) {
      PetscScalar    sum,val;
      const PetscInt *idxs;
      PetscInt       nz,j;
      CHKERRQ(ISGetLocalSize(ctx->benign_zerodiag_subs[i],&nz));
      CHKERRQ(ISGetIndices(ctx->benign_zerodiag_subs[i],&idxs));
      val = -ay[idxs[nz-1]];
      if (ctx->apply_p0) {
        sum = 0.;
        for (j=0;j<nz-1;j++) {
          sum += ay[idxs[j]];
          ay[idxs[j]] += val;
        }
        ay[idxs[nz-1]] += sum;
      } else {
        for (j=0;j<nz-1;j++) {
          ay[idxs[j]] += val;
        }
        ay[idxs[nz-1]] = 0.;
      }
      CHKERRQ(ISRestoreIndices(ctx->benign_zerodiag_subs[i],&idxs));
    }
    CHKERRQ(VecRestoreArray(y,&ay));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCBenignMatMultTranspose_Private(Mat A, Vec x, Vec y)
{
  PetscFunctionBegin;
  CHKERRQ(PCBDDCBenignMatMult_Private_Private(A,x,y,PETSC_TRUE));
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCBenignMatMult_Private(Mat A, Vec x, Vec y)
{
  PetscFunctionBegin;
  CHKERRQ(PCBDDCBenignMatMult_Private_Private(A,x,y,PETSC_FALSE));
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCBenignShellMat(PC pc, PetscBool restore)
{
  PC_IS                   *pcis = (PC_IS*)pc->data;
  PC_BDDC                 *pcbddc = (PC_BDDC*)pc->data;
  PCBDDCBenignMatMult_ctx ctx;

  PetscFunctionBegin;
  if (!restore) {
    Mat                A_IB,A_BI;
    PetscScalar        *work;
    PCBDDCReuseSolvers reuse = pcbddc->sub_schurs ? pcbddc->sub_schurs->reuse_solver : NULL;

    PetscCheckFalse(pcbddc->benign_original_mat,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Benign original mat has not been restored");
    if (!pcbddc->benign_change || !pcbddc->benign_n || pcbddc->benign_change_explicit) PetscFunctionReturn(0);
    CHKERRQ(PetscMalloc1(pcis->n,&work));
    CHKERRQ(MatCreate(PETSC_COMM_SELF,&A_IB));
    CHKERRQ(MatSetSizes(A_IB,pcis->n-pcis->n_B,pcis->n_B,PETSC_DECIDE,PETSC_DECIDE));
    CHKERRQ(MatSetType(A_IB,MATSHELL));
    CHKERRQ(MatShellSetOperation(A_IB,MATOP_MULT,(void (*)(void))PCBDDCBenignMatMult_Private));
    CHKERRQ(MatShellSetOperation(A_IB,MATOP_MULT_TRANSPOSE,(void (*)(void))PCBDDCBenignMatMultTranspose_Private));
    CHKERRQ(PetscNew(&ctx));
    CHKERRQ(MatShellSetContext(A_IB,ctx));
    ctx->apply_left = PETSC_TRUE;
    ctx->apply_right = PETSC_FALSE;
    ctx->apply_p0 = PETSC_FALSE;
    ctx->benign_n = pcbddc->benign_n;
    if (reuse) {
      ctx->benign_zerodiag_subs = reuse->benign_zerodiag_subs;
      ctx->free = PETSC_FALSE;
    } else { /* TODO: could be optimized for successive solves */
      ISLocalToGlobalMapping N_to_D;
      PetscInt               i;

      CHKERRQ(ISLocalToGlobalMappingCreateIS(pcis->is_I_local,&N_to_D));
      CHKERRQ(PetscMalloc1(pcbddc->benign_n,&ctx->benign_zerodiag_subs));
      for (i=0;i<pcbddc->benign_n;i++) {
        CHKERRQ(ISGlobalToLocalMappingApplyIS(N_to_D,IS_GTOLM_DROP,pcbddc->benign_zerodiag_subs[i],&ctx->benign_zerodiag_subs[i]));
      }
      CHKERRQ(ISLocalToGlobalMappingDestroy(&N_to_D));
      ctx->free = PETSC_TRUE;
    }
    ctx->A = pcis->A_IB;
    ctx->work = work;
    CHKERRQ(MatSetUp(A_IB));
    CHKERRQ(MatAssemblyBegin(A_IB,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(A_IB,MAT_FINAL_ASSEMBLY));
    pcis->A_IB = A_IB;

    /* A_BI as A_IB^T */
    CHKERRQ(MatCreateTranspose(A_IB,&A_BI));
    pcbddc->benign_original_mat = pcis->A_BI;
    pcis->A_BI = A_BI;
  } else {
    if (!pcbddc->benign_original_mat) {
      PetscFunctionReturn(0);
    }
    CHKERRQ(MatShellGetContext(pcis->A_IB,&ctx));
    CHKERRQ(MatDestroy(&pcis->A_IB));
    pcis->A_IB = ctx->A;
    ctx->A = NULL;
    CHKERRQ(MatDestroy(&pcis->A_BI));
    pcis->A_BI = pcbddc->benign_original_mat;
    pcbddc->benign_original_mat = NULL;
    if (ctx->free) {
      PetscInt i;
      for (i=0;i<ctx->benign_n;i++) {
        CHKERRQ(ISDestroy(&ctx->benign_zerodiag_subs[i]));
      }
      CHKERRQ(PetscFree(ctx->benign_zerodiag_subs));
    }
    CHKERRQ(PetscFree(ctx->work));
    CHKERRQ(PetscFree(ctx));
  }
  PetscFunctionReturn(0);
}

/* used just in bddc debug mode */
PetscErrorCode PCBDDCBenignProject(PC pc, IS is1, IS is2, Mat *B)
{
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;
  Mat_IS         *matis = (Mat_IS*)pc->pmat->data;
  Mat            An;

  PetscFunctionBegin;
  CHKERRQ(MatPtAP(matis->A,pcbddc->benign_change,MAT_INITIAL_MATRIX,2.0,&An));
  CHKERRQ(MatZeroRowsColumns(An,pcbddc->benign_n,pcbddc->benign_p0_lidx,1.0,NULL,NULL));
  if (is1) {
    CHKERRQ(MatCreateSubMatrix(An,is1,is2,MAT_INITIAL_MATRIX,B));
    CHKERRQ(MatDestroy(&An));
  } else {
    *B = An;
  }
  PetscFunctionReturn(0);
}

/* TODO: add reuse flag */
PetscErrorCode MatSeqAIJCompress(Mat A, Mat *B)
{
  Mat            Bt;
  PetscScalar    *a,*bdata;
  const PetscInt *ii,*ij;
  PetscInt       m,n,i,nnz,*bii,*bij;
  PetscBool      flg_row;

  PetscFunctionBegin;
  CHKERRQ(MatGetSize(A,&n,&m));
  CHKERRQ(MatGetRowIJ(A,0,PETSC_FALSE,PETSC_FALSE,&n,&ii,&ij,&flg_row));
  CHKERRQ(MatSeqAIJGetArray(A,&a));
  nnz = n;
  for (i=0;i<ii[n];i++) {
    if (PetscLikely(PetscAbsScalar(a[i]) > PETSC_SMALL)) nnz++;
  }
  CHKERRQ(PetscMalloc1(n+1,&bii));
  CHKERRQ(PetscMalloc1(nnz,&bij));
  CHKERRQ(PetscMalloc1(nnz,&bdata));
  nnz = 0;
  bii[0] = 0;
  for (i=0;i<n;i++) {
    PetscInt j;
    for (j=ii[i];j<ii[i+1];j++) {
      PetscScalar entry = a[j];
      if (PetscLikely(PetscAbsScalar(entry) > PETSC_SMALL) || (n == m && ij[j] == i)) {
        bij[nnz] = ij[j];
        bdata[nnz] = entry;
        nnz++;
      }
    }
    bii[i+1] = nnz;
  }
  CHKERRQ(MatSeqAIJRestoreArray(A,&a));
  CHKERRQ(MatCreateSeqAIJWithArrays(PetscObjectComm((PetscObject)A),n,m,bii,bij,bdata,&Bt));
  CHKERRQ(MatRestoreRowIJ(A,0,PETSC_FALSE,PETSC_FALSE,&n,&ii,&ij,&flg_row));
  {
    Mat_SeqAIJ *b = (Mat_SeqAIJ*)(Bt->data);
    b->free_a = PETSC_TRUE;
    b->free_ij = PETSC_TRUE;
  }
  if (*B == A) {
    CHKERRQ(MatDestroy(&A));
  }
  *B = Bt;
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCDetectDisconnectedComponents(PC pc, PetscBool filter, PetscInt *ncc, IS* cc[], IS* primalv)
{
  Mat                    B = NULL;
  DM                     dm;
  IS                     is_dummy,*cc_n;
  ISLocalToGlobalMapping l2gmap_dummy;
  PCBDDCGraph            graph;
  PetscInt               *xadj_filtered = NULL,*adjncy_filtered = NULL;
  PetscInt               i,n;
  PetscInt               *xadj,*adjncy;
  PetscBool              isplex = PETSC_FALSE;

  PetscFunctionBegin;
  if (ncc) *ncc = 0;
  if (cc) *cc = NULL;
  if (primalv) *primalv = NULL;
  CHKERRQ(PCBDDCGraphCreate(&graph));
  CHKERRQ(MatGetDM(pc->pmat,&dm));
  if (!dm) {
    CHKERRQ(PCGetDM(pc,&dm));
  }
  if (dm) {
    CHKERRQ(PetscObjectTypeCompare((PetscObject)dm,DMPLEX,&isplex));
  }
  if (filter) isplex = PETSC_FALSE;

  if (isplex) { /* this code has been modified from plexpartition.c */
    PetscInt       p, pStart, pEnd, a, adjSize, idx, size, nroots;
    PetscInt      *adj = NULL;
    IS             cellNumbering;
    const PetscInt *cellNum;
    PetscBool      useCone, useClosure;
    PetscSection   section;
    PetscSegBuffer adjBuffer;
    PetscSF        sfPoint;

    CHKERRQ(DMPlexGetHeightStratum(dm, 0, &pStart, &pEnd));
    CHKERRQ(DMGetPointSF(dm, &sfPoint));
    CHKERRQ(PetscSFGetGraph(sfPoint, &nroots, NULL, NULL, NULL));
    /* Build adjacency graph via a section/segbuffer */
    CHKERRQ(PetscSectionCreate(PetscObjectComm((PetscObject) dm), &section));
    CHKERRQ(PetscSectionSetChart(section, pStart, pEnd));
    CHKERRQ(PetscSegBufferCreate(sizeof(PetscInt),1000,&adjBuffer));
    /* Always use FVM adjacency to create partitioner graph */
    CHKERRQ(DMGetBasicAdjacency(dm, &useCone, &useClosure));
    CHKERRQ(DMSetBasicAdjacency(dm, PETSC_TRUE, PETSC_FALSE));
    CHKERRQ(DMPlexGetCellNumbering(dm, &cellNumbering));
    CHKERRQ(ISGetIndices(cellNumbering, &cellNum));
    for (n = 0, p = pStart; p < pEnd; p++) {
      /* Skip non-owned cells in parallel (ParMetis expects no overlap) */
      if (nroots > 0) {if (cellNum[p] < 0) continue;}
      adjSize = PETSC_DETERMINE;
      CHKERRQ(DMPlexGetAdjacency(dm, p, &adjSize, &adj));
      for (a = 0; a < adjSize; ++a) {
        const PetscInt point = adj[a];
        if (pStart <= point && point < pEnd) {
          PetscInt *PETSC_RESTRICT pBuf;
          CHKERRQ(PetscSectionAddDof(section, p, 1));
          CHKERRQ(PetscSegBufferGetInts(adjBuffer, 1, &pBuf));
          *pBuf = point;
        }
      }
      n++;
    }
    CHKERRQ(DMSetBasicAdjacency(dm, useCone, useClosure));
    /* Derive CSR graph from section/segbuffer */
    CHKERRQ(PetscSectionSetUp(section));
    CHKERRQ(PetscSectionGetStorageSize(section, &size));
    CHKERRQ(PetscMalloc1(n+1, &xadj));
    for (idx = 0, p = pStart; p < pEnd; p++) {
      if (nroots > 0) {if (cellNum[p] < 0) continue;}
      CHKERRQ(PetscSectionGetOffset(section, p, &(xadj[idx++])));
    }
    xadj[n] = size;
    CHKERRQ(PetscSegBufferExtractAlloc(adjBuffer, &adjncy));
    /* Clean up */
    CHKERRQ(PetscSegBufferDestroy(&adjBuffer));
    CHKERRQ(PetscSectionDestroy(&section));
    CHKERRQ(PetscFree(adj));
    graph->xadj = xadj;
    graph->adjncy = adjncy;
  } else {
    Mat       A;
    PetscBool isseqaij, flg_row;

    CHKERRQ(MatISGetLocalMat(pc->pmat,&A));
    if (!A->rmap->N || !A->cmap->N) {
      CHKERRQ(PCBDDCGraphDestroy(&graph));
      PetscFunctionReturn(0);
    }
    CHKERRQ(PetscObjectBaseTypeCompare((PetscObject)A,MATSEQAIJ,&isseqaij));
    if (!isseqaij && filter) {
      PetscBool isseqdense;

      CHKERRQ(PetscObjectTypeCompare((PetscObject)A,MATSEQDENSE,&isseqdense));
      if (!isseqdense) {
        CHKERRQ(MatConvert(A,MATSEQAIJ,MAT_INITIAL_MATRIX,&B));
      } else { /* TODO: rectangular case and LDA */
        PetscScalar *array;
        PetscReal   chop=1.e-6;

        CHKERRQ(MatDuplicate(A,MAT_COPY_VALUES,&B));
        CHKERRQ(MatDenseGetArray(B,&array));
        CHKERRQ(MatGetSize(B,&n,NULL));
        for (i=0;i<n;i++) {
          PetscInt j;
          for (j=i+1;j<n;j++) {
            PetscReal thresh = chop*(PetscAbsScalar(array[i*(n+1)])+PetscAbsScalar(array[j*(n+1)]));
            if (PetscAbsScalar(array[i*n+j]) < thresh) array[i*n+j] = 0.;
            if (PetscAbsScalar(array[j*n+i]) < thresh) array[j*n+i] = 0.;
          }
        }
        CHKERRQ(MatDenseRestoreArray(B,&array));
        CHKERRQ(MatConvert(B,MATSEQAIJ,MAT_INPLACE_MATRIX,&B));
      }
    } else {
      CHKERRQ(PetscObjectReference((PetscObject)A));
      B = A;
    }
    CHKERRQ(MatGetRowIJ(B,0,PETSC_TRUE,PETSC_FALSE,&n,(const PetscInt**)&xadj,(const PetscInt**)&adjncy,&flg_row));

    /* if filter is true, then removes entries lower than PETSC_SMALL in magnitude */
    if (filter) {
      PetscScalar *data;
      PetscInt    j,cum;

      CHKERRQ(PetscCalloc2(n+1,&xadj_filtered,xadj[n],&adjncy_filtered));
      CHKERRQ(MatSeqAIJGetArray(B,&data));
      cum = 0;
      for (i=0;i<n;i++) {
        PetscInt t;

        for (j=xadj[i];j<xadj[i+1];j++) {
          if (PetscUnlikely(PetscAbsScalar(data[j]) < PETSC_SMALL)) {
            continue;
          }
          adjncy_filtered[cum+xadj_filtered[i]++] = adjncy[j];
        }
        t = xadj_filtered[i];
        xadj_filtered[i] = cum;
        cum += t;
      }
      CHKERRQ(MatSeqAIJRestoreArray(B,&data));
      graph->xadj = xadj_filtered;
      graph->adjncy = adjncy_filtered;
    } else {
      graph->xadj = xadj;
      graph->adjncy = adjncy;
    }
  }
  /* compute local connected components using PCBDDCGraph */
  CHKERRQ(ISCreateStride(PETSC_COMM_SELF,n,0,1,&is_dummy));
  CHKERRQ(ISLocalToGlobalMappingCreateIS(is_dummy,&l2gmap_dummy));
  CHKERRQ(ISDestroy(&is_dummy));
  CHKERRQ(PCBDDCGraphInit(graph,l2gmap_dummy,n,PETSC_MAX_INT));
  CHKERRQ(ISLocalToGlobalMappingDestroy(&l2gmap_dummy));
  CHKERRQ(PCBDDCGraphSetUp(graph,1,NULL,NULL,0,NULL,NULL));
  CHKERRQ(PCBDDCGraphComputeConnectedComponents(graph));

  /* partial clean up */
  CHKERRQ(PetscFree2(xadj_filtered,adjncy_filtered));
  if (B) {
    PetscBool flg_row;
    CHKERRQ(MatRestoreRowIJ(B,0,PETSC_TRUE,PETSC_FALSE,&n,(const PetscInt**)&xadj,(const PetscInt**)&adjncy,&flg_row));
    CHKERRQ(MatDestroy(&B));
  }
  if (isplex) {
    CHKERRQ(PetscFree(xadj));
    CHKERRQ(PetscFree(adjncy));
  }

  /* get back data */
  if (isplex) {
    if (ncc) *ncc = graph->ncc;
    if (cc || primalv) {
      Mat          A;
      PetscBT      btv,btvt;
      PetscSection subSection;
      PetscInt     *ids,cum,cump,*cids,*pids;

      CHKERRQ(DMPlexGetSubdomainSection(dm,&subSection));
      CHKERRQ(MatISGetLocalMat(pc->pmat,&A));
      CHKERRQ(PetscMalloc3(A->rmap->n,&ids,graph->ncc+1,&cids,A->rmap->n,&pids));
      CHKERRQ(PetscBTCreate(A->rmap->n,&btv));
      CHKERRQ(PetscBTCreate(A->rmap->n,&btvt));

      cids[0] = 0;
      for (i = 0, cump = 0, cum = 0; i < graph->ncc; i++) {
        PetscInt j;

        CHKERRQ(PetscBTMemzero(A->rmap->n,btvt));
        for (j = graph->cptr[i]; j < graph->cptr[i+1]; j++) {
          PetscInt k, size, *closure = NULL, cell = graph->queue[j];

          CHKERRQ(DMPlexGetTransitiveClosure(dm,cell,PETSC_TRUE,&size,&closure));
          for (k = 0; k < 2*size; k += 2) {
            PetscInt s, pp, p = closure[k], off, dof, cdof;

            CHKERRQ(PetscSectionGetConstraintDof(subSection,p,&cdof));
            CHKERRQ(PetscSectionGetOffset(subSection,p,&off));
            CHKERRQ(PetscSectionGetDof(subSection,p,&dof));
            for (s = 0; s < dof-cdof; s++) {
              if (PetscBTLookupSet(btvt,off+s)) continue;
              if (!PetscBTLookup(btv,off+s)) ids[cum++] = off+s;
              else pids[cump++] = off+s; /* cross-vertex */
            }
            CHKERRQ(DMPlexGetTreeParent(dm,p,&pp,NULL));
            if (pp != p) {
              CHKERRQ(PetscSectionGetConstraintDof(subSection,pp,&cdof));
              CHKERRQ(PetscSectionGetOffset(subSection,pp,&off));
              CHKERRQ(PetscSectionGetDof(subSection,pp,&dof));
              for (s = 0; s < dof-cdof; s++) {
                if (PetscBTLookupSet(btvt,off+s)) continue;
                if (!PetscBTLookup(btv,off+s)) ids[cum++] = off+s;
                else pids[cump++] = off+s; /* cross-vertex */
              }
            }
          }
          CHKERRQ(DMPlexRestoreTransitiveClosure(dm,cell,PETSC_TRUE,&size,&closure));
        }
        cids[i+1] = cum;
        /* mark dofs as already assigned */
        for (j = cids[i]; j < cids[i+1]; j++) {
          CHKERRQ(PetscBTSet(btv,ids[j]));
        }
      }
      if (cc) {
        CHKERRQ(PetscMalloc1(graph->ncc,&cc_n));
        for (i = 0; i < graph->ncc; i++) {
          CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,cids[i+1]-cids[i],ids+cids[i],PETSC_COPY_VALUES,&cc_n[i]));
        }
        *cc = cc_n;
      }
      if (primalv) {
        CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)pc),cump,pids,PETSC_COPY_VALUES,primalv));
      }
      CHKERRQ(PetscFree3(ids,cids,pids));
      CHKERRQ(PetscBTDestroy(&btv));
      CHKERRQ(PetscBTDestroy(&btvt));
    }
  } else {
    if (ncc) *ncc = graph->ncc;
    if (cc) {
      CHKERRQ(PetscMalloc1(graph->ncc,&cc_n));
      for (i=0;i<graph->ncc;i++) {
        CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,graph->cptr[i+1]-graph->cptr[i],graph->queue+graph->cptr[i],PETSC_COPY_VALUES,&cc_n[i]));
      }
      *cc = cc_n;
    }
  }
  /* clean up graph */
  graph->xadj = NULL;
  graph->adjncy = NULL;
  CHKERRQ(PCBDDCGraphDestroy(&graph));
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCBenignCheck(PC pc, IS zerodiag)
{
  PC_BDDC*       pcbddc = (PC_BDDC*)pc->data;
  PC_IS*         pcis = (PC_IS*)(pc->data);
  IS             dirIS = NULL;
  PetscInt       i;

  PetscFunctionBegin;
  CHKERRQ(PCBDDCGraphGetDirichletDofs(pcbddc->mat_graph,&dirIS));
  if (zerodiag) {
    Mat            A;
    Vec            vec3_N;
    PetscScalar    *vals;
    const PetscInt *idxs;
    PetscInt       nz,*count;

    /* p0 */
    CHKERRQ(VecSet(pcis->vec1_N,0.));
    CHKERRQ(PetscMalloc1(pcis->n,&vals));
    CHKERRQ(ISGetLocalSize(zerodiag,&nz));
    CHKERRQ(ISGetIndices(zerodiag,&idxs));
    for (i=0;i<nz;i++) vals[i] = 1.;
    CHKERRQ(VecSetValues(pcis->vec1_N,nz,idxs,vals,INSERT_VALUES));
    CHKERRQ(VecAssemblyBegin(pcis->vec1_N));
    CHKERRQ(VecAssemblyEnd(pcis->vec1_N));
    /* v_I */
    CHKERRQ(VecSetRandom(pcis->vec2_N,NULL));
    for (i=0;i<nz;i++) vals[i] = 0.;
    CHKERRQ(VecSetValues(pcis->vec2_N,nz,idxs,vals,INSERT_VALUES));
    CHKERRQ(ISRestoreIndices(zerodiag,&idxs));
    CHKERRQ(ISGetIndices(pcis->is_B_local,&idxs));
    for (i=0;i<pcis->n_B;i++) vals[i] = 0.;
    CHKERRQ(VecSetValues(pcis->vec2_N,pcis->n_B,idxs,vals,INSERT_VALUES));
    CHKERRQ(ISRestoreIndices(pcis->is_B_local,&idxs));
    if (dirIS) {
      PetscInt n;

      CHKERRQ(ISGetLocalSize(dirIS,&n));
      CHKERRQ(ISGetIndices(dirIS,&idxs));
      for (i=0;i<n;i++) vals[i] = 0.;
      CHKERRQ(VecSetValues(pcis->vec2_N,n,idxs,vals,INSERT_VALUES));
      CHKERRQ(ISRestoreIndices(dirIS,&idxs));
    }
    CHKERRQ(VecAssemblyBegin(pcis->vec2_N));
    CHKERRQ(VecAssemblyEnd(pcis->vec2_N));
    CHKERRQ(VecDuplicate(pcis->vec1_N,&vec3_N));
    CHKERRQ(VecSet(vec3_N,0.));
    CHKERRQ(MatISGetLocalMat(pc->pmat,&A));
    CHKERRQ(MatMult(A,pcis->vec1_N,vec3_N));
    CHKERRQ(VecDot(vec3_N,pcis->vec2_N,&vals[0]));
    PetscCheckFalse(PetscAbsScalar(vals[0]) > 1.e-1,PETSC_COMM_SELF,PETSC_ERR_SUP,"Benign trick can not be applied! b(v_I,p_0) = %1.6e (should be numerically 0.)",PetscAbsScalar(vals[0]));
    CHKERRQ(PetscFree(vals));
    CHKERRQ(VecDestroy(&vec3_N));

    /* there should not be any pressure dofs lying on the interface */
    CHKERRQ(PetscCalloc1(pcis->n,&count));
    CHKERRQ(ISGetIndices(pcis->is_B_local,&idxs));
    for (i=0;i<pcis->n_B;i++) count[idxs[i]]++;
    CHKERRQ(ISRestoreIndices(pcis->is_B_local,&idxs));
    CHKERRQ(ISGetIndices(zerodiag,&idxs));
    for (i=0;i<nz;i++) PetscCheckFalse(count[idxs[i]],PETSC_COMM_SELF,PETSC_ERR_SUP,"Benign trick can not be applied! pressure dof %D is an interface dof",idxs[i]);
    CHKERRQ(ISRestoreIndices(zerodiag,&idxs));
    CHKERRQ(PetscFree(count));
  }
  CHKERRQ(ISDestroy(&dirIS));

  /* check PCBDDCBenignGetOrSetP0 */
  CHKERRQ(VecSetRandom(pcis->vec1_global,NULL));
  for (i=0;i<pcbddc->benign_n;i++) pcbddc->benign_p0[i] = -PetscGlobalRank-i;
  CHKERRQ(PCBDDCBenignGetOrSetP0(pc,pcis->vec1_global,PETSC_FALSE));
  for (i=0;i<pcbddc->benign_n;i++) pcbddc->benign_p0[i] = 1;
  CHKERRQ(PCBDDCBenignGetOrSetP0(pc,pcis->vec1_global,PETSC_TRUE));
  for (i=0;i<pcbddc->benign_n;i++) {
    PetscInt val = PetscRealPart(pcbddc->benign_p0[i]);
    PetscCheckFalse(val != -PetscGlobalRank-i,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error testing PCBDDCBenignGetOrSetP0! Found %g at %D instead of %g",PetscRealPart(pcbddc->benign_p0[i]),i,-PetscGlobalRank-i);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCBenignDetectSaddlePoint(PC pc, PetscBool reuse, IS *zerodiaglocal)
{
  PC_BDDC*       pcbddc = (PC_BDDC*)pc->data;
  Mat_IS*        matis = (Mat_IS*)(pc->pmat->data);
  IS             pressures = NULL,zerodiag = NULL,*bzerodiag = NULL,zerodiag_save,*zerodiag_subs;
  PetscInt       nz,n,benign_n,bsp = 1;
  PetscInt       *interior_dofs,n_interior_dofs,nneu;
  PetscBool      sorted,have_null,has_null_pressures,recompute_zerodiag,checkb;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (reuse) goto project_b0;
  CHKERRQ(PetscSFDestroy(&pcbddc->benign_sf));
  CHKERRQ(MatDestroy(&pcbddc->benign_B0));
  for (n=0;n<pcbddc->benign_n;n++) {
    CHKERRQ(ISDestroy(&pcbddc->benign_zerodiag_subs[n]));
  }
  CHKERRQ(PetscFree(pcbddc->benign_zerodiag_subs));
  has_null_pressures = PETSC_TRUE;
  have_null = PETSC_TRUE;
  /* if a local information on dofs is present, gets pressure dofs from command line (uses the last field is not provided)
     Without local information, it uses only the zerodiagonal dofs (ok if the pressure block is all zero and it is a scalar field)
     Checks if all the pressure dofs in each subdomain have a zero diagonal
     If not, a change of basis on pressures is not needed
     since the local Schur complements are already SPD
  */
  if (pcbddc->n_ISForDofsLocal) {
    IS        iP = NULL;
    PetscInt  p,*pp;
    PetscBool flg;

    CHKERRQ(PetscMalloc1(pcbddc->n_ISForDofsLocal,&pp));
    n    = pcbddc->n_ISForDofsLocal;
    ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)pc),((PetscObject)pc)->prefix,"BDDC benign options","PC");CHKERRQ(ierr);
    CHKERRQ(PetscOptionsIntArray("-pc_bddc_pressure_field","Field id for pressures",NULL,pp,&n,&flg));
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
    if (!flg) {
      n = 1;
      pp[0] = pcbddc->n_ISForDofsLocal-1;
    }

    bsp = 0;
    for (p=0;p<n;p++) {
      PetscInt bs;

      PetscCheckFalse(pp[p] < 0 || pp[p] > pcbddc->n_ISForDofsLocal-1,PetscObjectComm((PetscObject)pc),PETSC_ERR_USER,"Invalid field id for pressures %D",pp[p]);
      CHKERRQ(ISGetBlockSize(pcbddc->ISForDofsLocal[pp[p]],&bs));
      bsp += bs;
    }
    CHKERRQ(PetscMalloc1(bsp,&bzerodiag));
    bsp  = 0;
    for (p=0;p<n;p++) {
      const PetscInt *idxs;
      PetscInt       b,bs,npl,*bidxs;

      CHKERRQ(ISGetBlockSize(pcbddc->ISForDofsLocal[pp[p]],&bs));
      CHKERRQ(ISGetLocalSize(pcbddc->ISForDofsLocal[pp[p]],&npl));
      CHKERRQ(ISGetIndices(pcbddc->ISForDofsLocal[pp[p]],&idxs));
      CHKERRQ(PetscMalloc1(npl/bs,&bidxs));
      for (b=0;b<bs;b++) {
        PetscInt i;

        for (i=0;i<npl/bs;i++) bidxs[i] = idxs[bs*i+b];
        CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,npl/bs,bidxs,PETSC_COPY_VALUES,&bzerodiag[bsp]));
        bsp++;
      }
      CHKERRQ(PetscFree(bidxs));
      CHKERRQ(ISRestoreIndices(pcbddc->ISForDofsLocal[pp[p]],&idxs));
    }
    CHKERRQ(ISConcatenate(PETSC_COMM_SELF,bsp,bzerodiag,&pressures));

    /* remove zeroed out pressures if we are setting up a BDDC solver for a saddle-point FETI-DP */
    CHKERRQ(PetscObjectQuery((PetscObject)pc,"__KSPFETIDP_lP",(PetscObject*)&iP));
    if (iP) {
      IS newpressures;

      CHKERRQ(ISDifference(pressures,iP,&newpressures));
      CHKERRQ(ISDestroy(&pressures));
      pressures = newpressures;
    }
    CHKERRQ(ISSorted(pressures,&sorted));
    if (!sorted) {
      CHKERRQ(ISSort(pressures));
    }
    CHKERRQ(PetscFree(pp));
  }

  /* pcis has not been setup yet, so get the local size from the subdomain matrix */
  CHKERRQ(MatGetLocalSize(pcbddc->local_mat,&n,NULL));
  if (!n) pcbddc->benign_change_explicit = PETSC_TRUE;
  CHKERRQ(MatFindZeroDiagonals(pcbddc->local_mat,&zerodiag));
  CHKERRQ(ISSorted(zerodiag,&sorted));
  if (!sorted) {
    CHKERRQ(ISSort(zerodiag));
  }
  CHKERRQ(PetscObjectReference((PetscObject)zerodiag));
  zerodiag_save = zerodiag;
  CHKERRQ(ISGetLocalSize(zerodiag,&nz));
  if (!nz) {
    if (n) have_null = PETSC_FALSE;
    has_null_pressures = PETSC_FALSE;
    CHKERRQ(ISDestroy(&zerodiag));
  }
  recompute_zerodiag = PETSC_FALSE;

  /* in case disconnected subdomains info is present, split the pressures accordingly (otherwise the benign trick could fail) */
  zerodiag_subs    = NULL;
  benign_n         = 0;
  n_interior_dofs  = 0;
  interior_dofs    = NULL;
  nneu             = 0;
  if (pcbddc->NeumannBoundariesLocal) {
    CHKERRQ(ISGetLocalSize(pcbddc->NeumannBoundariesLocal,&nneu));
  }
  checkb = (PetscBool)(!pcbddc->NeumannBoundariesLocal || pcbddc->current_level);
  if (checkb) { /* need to compute interior nodes */
    PetscInt n,i,j;
    PetscInt n_neigh,*neigh,*n_shared,**shared;
    PetscInt *iwork;

    CHKERRQ(ISLocalToGlobalMappingGetSize(matis->rmapping,&n));
    CHKERRQ(ISLocalToGlobalMappingGetInfo(matis->rmapping,&n_neigh,&neigh,&n_shared,&shared));
    CHKERRQ(PetscCalloc1(n,&iwork));
    CHKERRQ(PetscMalloc1(n,&interior_dofs));
    for (i=1;i<n_neigh;i++)
      for (j=0;j<n_shared[i];j++)
          iwork[shared[i][j]] += 1;
    for (i=0;i<n;i++)
      if (!iwork[i])
        interior_dofs[n_interior_dofs++] = i;
    CHKERRQ(PetscFree(iwork));
    CHKERRQ(ISLocalToGlobalMappingRestoreInfo(matis->rmapping,&n_neigh,&neigh,&n_shared,&shared));
  }
  if (has_null_pressures) {
    IS             *subs;
    PetscInt       nsubs,i,j,nl;
    const PetscInt *idxs;
    PetscScalar    *array;
    Vec            *work;

    subs  = pcbddc->local_subs;
    nsubs = pcbddc->n_local_subs;
    /* these vectors are needed to check if the constant on pressures is in the kernel of the local operator B (i.e. B(v_I,p0) should be zero) */
    if (checkb) {
      CHKERRQ(VecDuplicateVecs(matis->y,2,&work));
      CHKERRQ(ISGetLocalSize(zerodiag,&nl));
      CHKERRQ(ISGetIndices(zerodiag,&idxs));
      /* work[0] = 1_p */
      CHKERRQ(VecSet(work[0],0.));
      CHKERRQ(VecGetArray(work[0],&array));
      for (j=0;j<nl;j++) array[idxs[j]] = 1.;
      CHKERRQ(VecRestoreArray(work[0],&array));
      /* work[0] = 1_v */
      CHKERRQ(VecSet(work[1],1.));
      CHKERRQ(VecGetArray(work[1],&array));
      for (j=0;j<nl;j++) array[idxs[j]] = 0.;
      CHKERRQ(VecRestoreArray(work[1],&array));
      CHKERRQ(ISRestoreIndices(zerodiag,&idxs));
    }

    if (nsubs > 1 || bsp > 1) {
      IS       *is;
      PetscInt b,totb;

      totb  = bsp;
      is    = bsp > 1 ? bzerodiag : &zerodiag;
      nsubs = PetscMax(nsubs,1);
      CHKERRQ(PetscCalloc1(nsubs*totb,&zerodiag_subs));
      for (b=0;b<totb;b++) {
        for (i=0;i<nsubs;i++) {
          ISLocalToGlobalMapping l2g;
          IS                     t_zerodiag_subs;
          PetscInt               nl;

          if (subs) {
            CHKERRQ(ISLocalToGlobalMappingCreateIS(subs[i],&l2g));
          } else {
            IS tis;

            CHKERRQ(MatGetLocalSize(pcbddc->local_mat,&nl,NULL));
            CHKERRQ(ISCreateStride(PETSC_COMM_SELF,nl,0,1,&tis));
            CHKERRQ(ISLocalToGlobalMappingCreateIS(tis,&l2g));
            CHKERRQ(ISDestroy(&tis));
          }
          CHKERRQ(ISGlobalToLocalMappingApplyIS(l2g,IS_GTOLM_DROP,is[b],&t_zerodiag_subs));
          CHKERRQ(ISGetLocalSize(t_zerodiag_subs,&nl));
          if (nl) {
            PetscBool valid = PETSC_TRUE;

            if (checkb) {
              CHKERRQ(VecSet(matis->x,0));
              CHKERRQ(ISGetLocalSize(subs[i],&nl));
              CHKERRQ(ISGetIndices(subs[i],&idxs));
              CHKERRQ(VecGetArray(matis->x,&array));
              for (j=0;j<nl;j++) array[idxs[j]] = 1.;
              CHKERRQ(VecRestoreArray(matis->x,&array));
              CHKERRQ(ISRestoreIndices(subs[i],&idxs));
              CHKERRQ(VecPointwiseMult(matis->x,work[0],matis->x));
              CHKERRQ(MatMult(matis->A,matis->x,matis->y));
              CHKERRQ(VecPointwiseMult(matis->y,work[1],matis->y));
              CHKERRQ(VecGetArray(matis->y,&array));
              for (j=0;j<n_interior_dofs;j++) {
                if (PetscAbsScalar(array[interior_dofs[j]]) > PETSC_SMALL) {
                  valid = PETSC_FALSE;
                  break;
                }
              }
              CHKERRQ(VecRestoreArray(matis->y,&array));
            }
            if (valid && nneu) {
              const PetscInt *idxs;
              PetscInt       nzb;

              CHKERRQ(ISGetIndices(pcbddc->NeumannBoundariesLocal,&idxs));
              CHKERRQ(ISGlobalToLocalMappingApply(l2g,IS_GTOLM_DROP,nneu,idxs,&nzb,NULL));
              CHKERRQ(ISRestoreIndices(pcbddc->NeumannBoundariesLocal,&idxs));
              if (nzb) valid = PETSC_FALSE;
            }
            if (valid && pressures) {
              IS       t_pressure_subs,tmp;
              PetscInt i1,i2;

              CHKERRQ(ISGlobalToLocalMappingApplyIS(l2g,IS_GTOLM_DROP,pressures,&t_pressure_subs));
              CHKERRQ(ISEmbed(t_zerodiag_subs,t_pressure_subs,PETSC_TRUE,&tmp));
              CHKERRQ(ISGetLocalSize(tmp,&i1));
              CHKERRQ(ISGetLocalSize(t_zerodiag_subs,&i2));
              if (i2 != i1) valid = PETSC_FALSE;
              CHKERRQ(ISDestroy(&t_pressure_subs));
              CHKERRQ(ISDestroy(&tmp));
            }
            if (valid) {
              CHKERRQ(ISLocalToGlobalMappingApplyIS(l2g,t_zerodiag_subs,&zerodiag_subs[benign_n]));
              benign_n++;
            } else recompute_zerodiag = PETSC_TRUE;
          }
          CHKERRQ(ISDestroy(&t_zerodiag_subs));
          CHKERRQ(ISLocalToGlobalMappingDestroy(&l2g));
        }
      }
    } else { /* there's just one subdomain (or zero if they have not been detected */
      PetscBool valid = PETSC_TRUE;

      if (nneu) valid = PETSC_FALSE;
      if (valid && pressures) {
        CHKERRQ(ISEqual(pressures,zerodiag,&valid));
      }
      if (valid && checkb) {
        CHKERRQ(MatMult(matis->A,work[0],matis->x));
        CHKERRQ(VecPointwiseMult(matis->x,work[1],matis->x));
        CHKERRQ(VecGetArray(matis->x,&array));
        for (j=0;j<n_interior_dofs;j++) {
          if (PetscAbsScalar(array[interior_dofs[j]]) > PETSC_SMALL) {
            valid = PETSC_FALSE;
            break;
          }
        }
        CHKERRQ(VecRestoreArray(matis->x,&array));
      }
      if (valid) {
        benign_n = 1;
        CHKERRQ(PetscMalloc1(benign_n,&zerodiag_subs));
        CHKERRQ(PetscObjectReference((PetscObject)zerodiag));
        zerodiag_subs[0] = zerodiag;
      }
    }
    if (checkb) {
      CHKERRQ(VecDestroyVecs(2,&work));
    }
  }
  CHKERRQ(PetscFree(interior_dofs));

  if (!benign_n) {
    PetscInt n;

    CHKERRQ(ISDestroy(&zerodiag));
    recompute_zerodiag = PETSC_FALSE;
    CHKERRQ(MatGetLocalSize(pcbddc->local_mat,&n,NULL));
    if (n) have_null = PETSC_FALSE;
  }

  /* final check for null pressures */
  if (zerodiag && pressures) {
    CHKERRQ(ISEqual(pressures,zerodiag,&have_null));
  }

  if (recompute_zerodiag) {
    CHKERRQ(ISDestroy(&zerodiag));
    if (benign_n == 1) {
      CHKERRQ(PetscObjectReference((PetscObject)zerodiag_subs[0]));
      zerodiag = zerodiag_subs[0];
    } else {
      PetscInt i,nzn,*new_idxs;

      nzn = 0;
      for (i=0;i<benign_n;i++) {
        PetscInt ns;
        CHKERRQ(ISGetLocalSize(zerodiag_subs[i],&ns));
        nzn += ns;
      }
      CHKERRQ(PetscMalloc1(nzn,&new_idxs));
      nzn = 0;
      for (i=0;i<benign_n;i++) {
        PetscInt ns,*idxs;
        CHKERRQ(ISGetLocalSize(zerodiag_subs[i],&ns));
        CHKERRQ(ISGetIndices(zerodiag_subs[i],(const PetscInt**)&idxs));
        CHKERRQ(PetscArraycpy(new_idxs+nzn,idxs,ns));
        CHKERRQ(ISRestoreIndices(zerodiag_subs[i],(const PetscInt**)&idxs));
        nzn += ns;
      }
      CHKERRQ(PetscSortInt(nzn,new_idxs));
      CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,nzn,new_idxs,PETSC_OWN_POINTER,&zerodiag));
    }
    have_null = PETSC_FALSE;
  }

  /* determines if the coarse solver will be singular or not */
  CHKERRMPI(MPIU_Allreduce(&have_null,&pcbddc->benign_null,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)pc)));

  /* Prepare matrix to compute no-net-flux */
  if (pcbddc->compute_nonetflux && !pcbddc->divudotp) {
    Mat                    A,loc_divudotp;
    ISLocalToGlobalMapping rl2g,cl2g,l2gmap;
    IS                     row,col,isused = NULL;
    PetscInt               M,N,n,st,n_isused;

    if (pressures) {
      isused = pressures;
    } else {
      isused = zerodiag_save;
    }
    CHKERRQ(MatISGetLocalToGlobalMapping(pc->pmat,&l2gmap,NULL));
    CHKERRQ(MatISGetLocalMat(pc->pmat,&A));
    CHKERRQ(MatGetLocalSize(A,&n,NULL));
    PetscCheckFalse(!isused && n,PETSC_COMM_SELF,PETSC_ERR_USER,"Don't know how to extract div u dot p! Please provide the pressure field");
    n_isused = 0;
    if (isused) {
      CHKERRQ(ISGetLocalSize(isused,&n_isused));
    }
    CHKERRMPI(MPI_Scan(&n_isused,&st,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)pc)));
    st = st-n_isused;
    if (n) {
      const PetscInt *gidxs;

      CHKERRQ(MatCreateSubMatrix(A,isused,NULL,MAT_INITIAL_MATRIX,&loc_divudotp));
      CHKERRQ(ISLocalToGlobalMappingGetIndices(l2gmap,&gidxs));
      /* TODO: extend ISCreateStride with st = PETSC_DECIDE */
      CHKERRQ(ISCreateStride(PetscObjectComm((PetscObject)pc),n_isused,st,1,&row));
      CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)pc),n,gidxs,PETSC_COPY_VALUES,&col));
      CHKERRQ(ISLocalToGlobalMappingRestoreIndices(l2gmap,&gidxs));
    } else {
      CHKERRQ(MatCreateSeqAIJ(PETSC_COMM_SELF,0,0,1,NULL,&loc_divudotp));
      CHKERRQ(ISCreateStride(PetscObjectComm((PetscObject)pc),n_isused,st,1,&row));
      CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)pc),0,NULL,PETSC_COPY_VALUES,&col));
    }
    CHKERRQ(MatGetSize(pc->pmat,NULL,&N));
    CHKERRQ(ISGetSize(row,&M));
    CHKERRQ(ISLocalToGlobalMappingCreateIS(row,&rl2g));
    CHKERRQ(ISLocalToGlobalMappingCreateIS(col,&cl2g));
    CHKERRQ(ISDestroy(&row));
    CHKERRQ(ISDestroy(&col));
    CHKERRQ(MatCreate(PetscObjectComm((PetscObject)pc),&pcbddc->divudotp));
    CHKERRQ(MatSetType(pcbddc->divudotp,MATIS));
    CHKERRQ(MatSetSizes(pcbddc->divudotp,PETSC_DECIDE,PETSC_DECIDE,M,N));
    CHKERRQ(MatSetLocalToGlobalMapping(pcbddc->divudotp,rl2g,cl2g));
    CHKERRQ(ISLocalToGlobalMappingDestroy(&rl2g));
    CHKERRQ(ISLocalToGlobalMappingDestroy(&cl2g));
    CHKERRQ(MatISSetLocalMat(pcbddc->divudotp,loc_divudotp));
    CHKERRQ(MatDestroy(&loc_divudotp));
    CHKERRQ(MatAssemblyBegin(pcbddc->divudotp,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(pcbddc->divudotp,MAT_FINAL_ASSEMBLY));
  }
  CHKERRQ(ISDestroy(&zerodiag_save));
  CHKERRQ(ISDestroy(&pressures));
  if (bzerodiag) {
    PetscInt i;

    for (i=0;i<bsp;i++) {
      CHKERRQ(ISDestroy(&bzerodiag[i]));
    }
    CHKERRQ(PetscFree(bzerodiag));
  }
  pcbddc->benign_n = benign_n;
  pcbddc->benign_zerodiag_subs = zerodiag_subs;

  /* determines if the problem has subdomains with 0 pressure block */
  have_null = (PetscBool)(!!pcbddc->benign_n);
  CHKERRMPI(MPIU_Allreduce(&have_null,&pcbddc->benign_have_null,1,MPIU_BOOL,MPI_LOR,PetscObjectComm((PetscObject)pc)));

project_b0:
  CHKERRQ(MatGetLocalSize(pcbddc->local_mat,&n,NULL));
  /* change of basis and p0 dofs */
  if (pcbddc->benign_n) {
    PetscInt i,s,*nnz;

    /* local change of basis for pressures */
    CHKERRQ(MatDestroy(&pcbddc->benign_change));
    CHKERRQ(MatCreate(PetscObjectComm((PetscObject)pcbddc->local_mat),&pcbddc->benign_change));
    CHKERRQ(MatSetType(pcbddc->benign_change,MATAIJ));
    CHKERRQ(MatSetSizes(pcbddc->benign_change,n,n,PETSC_DECIDE,PETSC_DECIDE));
    CHKERRQ(PetscMalloc1(n,&nnz));
    for (i=0;i<n;i++) nnz[i] = 1; /* defaults to identity */
    for (i=0;i<pcbddc->benign_n;i++) {
      const PetscInt *idxs;
      PetscInt       nzs,j;

      CHKERRQ(ISGetLocalSize(pcbddc->benign_zerodiag_subs[i],&nzs));
      CHKERRQ(ISGetIndices(pcbddc->benign_zerodiag_subs[i],&idxs));
      for (j=0;j<nzs-1;j++) nnz[idxs[j]] = 2; /* change on pressures */
      nnz[idxs[nzs-1]] = nzs; /* last local pressure dof in subdomain */
      CHKERRQ(ISRestoreIndices(pcbddc->benign_zerodiag_subs[i],&idxs));
    }
    CHKERRQ(MatSeqAIJSetPreallocation(pcbddc->benign_change,0,nnz));
    CHKERRQ(MatSetOption(pcbddc->benign_change,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE));
    CHKERRQ(PetscFree(nnz));
    /* set identity by default */
    for (i=0;i<n;i++) {
      CHKERRQ(MatSetValue(pcbddc->benign_change,i,i,1.,INSERT_VALUES));
    }
    CHKERRQ(PetscFree3(pcbddc->benign_p0_lidx,pcbddc->benign_p0_gidx,pcbddc->benign_p0));
    CHKERRQ(PetscMalloc3(pcbddc->benign_n,&pcbddc->benign_p0_lidx,pcbddc->benign_n,&pcbddc->benign_p0_gidx,pcbddc->benign_n,&pcbddc->benign_p0));
    /* set change on pressures */
    for (s=0;s<pcbddc->benign_n;s++) {
      PetscScalar    *array;
      const PetscInt *idxs;
      PetscInt       nzs;

      CHKERRQ(ISGetLocalSize(pcbddc->benign_zerodiag_subs[s],&nzs));
      CHKERRQ(ISGetIndices(pcbddc->benign_zerodiag_subs[s],&idxs));
      for (i=0;i<nzs-1;i++) {
        PetscScalar vals[2];
        PetscInt    cols[2];

        cols[0] = idxs[i];
        cols[1] = idxs[nzs-1];
        vals[0] = 1.;
        vals[1] = 1.;
        CHKERRQ(MatSetValues(pcbddc->benign_change,1,cols,2,cols,vals,INSERT_VALUES));
      }
      CHKERRQ(PetscMalloc1(nzs,&array));
      for (i=0;i<nzs-1;i++) array[i] = -1.;
      array[nzs-1] = 1.;
      CHKERRQ(MatSetValues(pcbddc->benign_change,1,idxs+nzs-1,nzs,idxs,array,INSERT_VALUES));
      /* store local idxs for p0 */
      pcbddc->benign_p0_lidx[s] = idxs[nzs-1];
      CHKERRQ(ISRestoreIndices(pcbddc->benign_zerodiag_subs[s],&idxs));
      CHKERRQ(PetscFree(array));
    }
    CHKERRQ(MatAssemblyBegin(pcbddc->benign_change,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(pcbddc->benign_change,MAT_FINAL_ASSEMBLY));

    /* project if needed */
    if (pcbddc->benign_change_explicit) {
      Mat M;

      CHKERRQ(MatPtAP(pcbddc->local_mat,pcbddc->benign_change,MAT_INITIAL_MATRIX,2.0,&M));
      CHKERRQ(MatDestroy(&pcbddc->local_mat));
      CHKERRQ(MatSeqAIJCompress(M,&pcbddc->local_mat));
      CHKERRQ(MatDestroy(&M));
    }
    /* store global idxs for p0 */
    CHKERRQ(ISLocalToGlobalMappingApply(matis->rmapping,pcbddc->benign_n,pcbddc->benign_p0_lidx,pcbddc->benign_p0_gidx));
  }
  *zerodiaglocal = zerodiag;
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCBenignGetOrSetP0(PC pc, Vec v, PetscBool get)
{
  PC_BDDC*       pcbddc = (PC_BDDC*)pc->data;
  PetscScalar    *array;

  PetscFunctionBegin;
  if (!pcbddc->benign_sf) {
    CHKERRQ(PetscSFCreate(PetscObjectComm((PetscObject)pc),&pcbddc->benign_sf));
    CHKERRQ(PetscSFSetGraphLayout(pcbddc->benign_sf,pc->pmat->rmap,pcbddc->benign_n,NULL,PETSC_OWN_POINTER,pcbddc->benign_p0_gidx));
  }
  if (get) {
    CHKERRQ(VecGetArrayRead(v,(const PetscScalar**)&array));
    CHKERRQ(PetscSFBcastBegin(pcbddc->benign_sf,MPIU_SCALAR,array,pcbddc->benign_p0,MPI_REPLACE));
    CHKERRQ(PetscSFBcastEnd(pcbddc->benign_sf,MPIU_SCALAR,array,pcbddc->benign_p0,MPI_REPLACE));
    CHKERRQ(VecRestoreArrayRead(v,(const PetscScalar**)&array));
  } else {
    CHKERRQ(VecGetArray(v,&array));
    CHKERRQ(PetscSFReduceBegin(pcbddc->benign_sf,MPIU_SCALAR,pcbddc->benign_p0,array,MPI_REPLACE));
    CHKERRQ(PetscSFReduceEnd(pcbddc->benign_sf,MPIU_SCALAR,pcbddc->benign_p0,array,MPI_REPLACE));
    CHKERRQ(VecRestoreArray(v,&array));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCBenignPopOrPushB0(PC pc, PetscBool pop)
{
  PC_BDDC*       pcbddc = (PC_BDDC*)pc->data;

  PetscFunctionBegin;
  /* TODO: add error checking
    - avoid nested pop (or push) calls.
    - cannot push before pop.
    - cannot call this if pcbddc->local_mat is NULL
  */
  if (!pcbddc->benign_n) {
    PetscFunctionReturn(0);
  }
  if (pop) {
    if (pcbddc->benign_change_explicit) {
      IS       is_p0;
      MatReuse reuse;

      /* extract B_0 */
      reuse = MAT_INITIAL_MATRIX;
      if (pcbddc->benign_B0) {
        reuse = MAT_REUSE_MATRIX;
      }
      CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,pcbddc->benign_n,pcbddc->benign_p0_lidx,PETSC_COPY_VALUES,&is_p0));
      CHKERRQ(MatCreateSubMatrix(pcbddc->local_mat,is_p0,NULL,reuse,&pcbddc->benign_B0));
      /* remove rows and cols from local problem */
      CHKERRQ(MatSetOption(pcbddc->local_mat,MAT_KEEP_NONZERO_PATTERN,PETSC_TRUE));
      CHKERRQ(MatSetOption(pcbddc->local_mat,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_FALSE));
      CHKERRQ(MatZeroRowsColumnsIS(pcbddc->local_mat,is_p0,1.0,NULL,NULL));
      CHKERRQ(ISDestroy(&is_p0));
    } else {
      Mat_IS      *matis = (Mat_IS*)pc->pmat->data;
      PetscScalar *vals;
      PetscInt    i,n,*idxs_ins;

      CHKERRQ(VecGetLocalSize(matis->y,&n));
      CHKERRQ(PetscMalloc2(n,&idxs_ins,n,&vals));
      if (!pcbddc->benign_B0) {
        PetscInt *nnz;
        CHKERRQ(MatCreate(PetscObjectComm((PetscObject)pcbddc->local_mat),&pcbddc->benign_B0));
        CHKERRQ(MatSetType(pcbddc->benign_B0,MATAIJ));
        CHKERRQ(MatSetSizes(pcbddc->benign_B0,pcbddc->benign_n,n,PETSC_DECIDE,PETSC_DECIDE));
        CHKERRQ(PetscMalloc1(pcbddc->benign_n,&nnz));
        for (i=0;i<pcbddc->benign_n;i++) {
          CHKERRQ(ISGetLocalSize(pcbddc->benign_zerodiag_subs[i],&nnz[i]));
          nnz[i] = n - nnz[i];
        }
        CHKERRQ(MatSeqAIJSetPreallocation(pcbddc->benign_B0,0,nnz));
        CHKERRQ(MatSetOption(pcbddc->benign_B0,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE));
        CHKERRQ(PetscFree(nnz));
      }

      for (i=0;i<pcbddc->benign_n;i++) {
        PetscScalar *array;
        PetscInt    *idxs,j,nz,cum;

        CHKERRQ(VecSet(matis->x,0.));
        CHKERRQ(ISGetLocalSize(pcbddc->benign_zerodiag_subs[i],&nz));
        CHKERRQ(ISGetIndices(pcbddc->benign_zerodiag_subs[i],(const PetscInt**)&idxs));
        for (j=0;j<nz;j++) vals[j] = 1.;
        CHKERRQ(VecSetValues(matis->x,nz,idxs,vals,INSERT_VALUES));
        CHKERRQ(VecAssemblyBegin(matis->x));
        CHKERRQ(VecAssemblyEnd(matis->x));
        CHKERRQ(VecSet(matis->y,0.));
        CHKERRQ(MatMult(matis->A,matis->x,matis->y));
        CHKERRQ(VecGetArray(matis->y,&array));
        cum = 0;
        for (j=0;j<n;j++) {
          if (PetscUnlikely(PetscAbsScalar(array[j]) > PETSC_SMALL)) {
            vals[cum] = array[j];
            idxs_ins[cum] = j;
            cum++;
          }
        }
        CHKERRQ(MatSetValues(pcbddc->benign_B0,1,&i,cum,idxs_ins,vals,INSERT_VALUES));
        CHKERRQ(VecRestoreArray(matis->y,&array));
        CHKERRQ(ISRestoreIndices(pcbddc->benign_zerodiag_subs[i],(const PetscInt**)&idxs));
      }
      CHKERRQ(MatAssemblyBegin(pcbddc->benign_B0,MAT_FINAL_ASSEMBLY));
      CHKERRQ(MatAssemblyEnd(pcbddc->benign_B0,MAT_FINAL_ASSEMBLY));
      CHKERRQ(PetscFree2(idxs_ins,vals));
    }
  } else { /* push */
    if (pcbddc->benign_change_explicit) {
      PetscInt i;

      for (i=0;i<pcbddc->benign_n;i++) {
        PetscScalar *B0_vals;
        PetscInt    *B0_cols,B0_ncol;

        CHKERRQ(MatGetRow(pcbddc->benign_B0,i,&B0_ncol,(const PetscInt**)&B0_cols,(const PetscScalar**)&B0_vals));
        CHKERRQ(MatSetValues(pcbddc->local_mat,1,pcbddc->benign_p0_lidx+i,B0_ncol,B0_cols,B0_vals,INSERT_VALUES));
        CHKERRQ(MatSetValues(pcbddc->local_mat,B0_ncol,B0_cols,1,pcbddc->benign_p0_lidx+i,B0_vals,INSERT_VALUES));
        CHKERRQ(MatSetValue(pcbddc->local_mat,pcbddc->benign_p0_lidx[i],pcbddc->benign_p0_lidx[i],0.0,INSERT_VALUES));
        CHKERRQ(MatRestoreRow(pcbddc->benign_B0,i,&B0_ncol,(const PetscInt**)&B0_cols,(const PetscScalar**)&B0_vals));
      }
      CHKERRQ(MatAssemblyBegin(pcbddc->local_mat,MAT_FINAL_ASSEMBLY));
      CHKERRQ(MatAssemblyEnd(pcbddc->local_mat,MAT_FINAL_ASSEMBLY));
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Cannot push B0!");
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCAdaptiveSelection(PC pc)
{
  PC_BDDC*        pcbddc = (PC_BDDC*)pc->data;
  PCBDDCSubSchurs sub_schurs = pcbddc->sub_schurs;
  PetscBLASInt    B_dummyint,B_neigs,B_ierr,B_lwork;
  PetscBLASInt    *B_iwork,*B_ifail;
  PetscScalar     *work,lwork;
  PetscScalar     *St,*S,*eigv;
  PetscScalar     *Sarray,*Starray;
  PetscReal       *eigs,thresh,lthresh,uthresh;
  PetscInt        i,nmax,nmin,nv,cum,mss,cum2,cumarray,maxneigs;
  PetscBool       allocated_S_St;
#if defined(PETSC_USE_COMPLEX)
  PetscReal       *rwork;
#endif
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscCheckFalse(!sub_schurs,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Adaptive selection of constraints requires SubSchurs data");
  PetscCheckFalse(!sub_schurs->schur_explicit,PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Adaptive selection of constraints requires MUMPS and/or MKL_CPARDISO");
  PetscCheckFalse(sub_schurs->n_subs && (!sub_schurs->is_symmetric),PETSC_COMM_SELF,PETSC_ERR_SUP,"Adaptive selection not yet implemented for this matrix pencil (herm %d, symm %d, posdef %d)",sub_schurs->is_hermitian,sub_schurs->is_symmetric,sub_schurs->is_posdef);
  CHKERRQ(PetscLogEventBegin(PC_BDDC_AdaptiveSetUp[pcbddc->current_level],pc,0,0,0));

  if (pcbddc->dbg_flag) {
    CHKERRQ(PetscViewerFlush(pcbddc->dbg_viewer));
    CHKERRQ(PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"--------------------------------------------------\n"));
    CHKERRQ(PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"Check adaptive selection of constraints\n"));
    CHKERRQ(PetscViewerASCIIPushSynchronized(pcbddc->dbg_viewer));
  }

  if (pcbddc->dbg_flag) {
    CHKERRQ(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d cc %D (%d,%d).\n",PetscGlobalRank,sub_schurs->n_subs,sub_schurs->is_hermitian,sub_schurs->is_posdef));
  }

  /* max size of subsets */
  mss = 0;
  for (i=0;i<sub_schurs->n_subs;i++) {
    PetscInt subset_size;

    CHKERRQ(ISGetLocalSize(sub_schurs->is_subs[i],&subset_size));
    mss = PetscMax(mss,subset_size);
  }

  /* min/max and threshold */
  nmax = pcbddc->adaptive_nmax > 0 ? pcbddc->adaptive_nmax : mss;
  nmin = pcbddc->adaptive_nmin > 0 ? pcbddc->adaptive_nmin : 0;
  nmax = PetscMax(nmin,nmax);
  allocated_S_St = PETSC_FALSE;
  if (nmin || !sub_schurs->is_posdef) { /* XXX */
    allocated_S_St = PETSC_TRUE;
  }

  /* allocate lapack workspace */
  cum = cum2 = 0;
  maxneigs = 0;
  for (i=0;i<sub_schurs->n_subs;i++) {
    PetscInt n,subset_size;

    CHKERRQ(ISGetLocalSize(sub_schurs->is_subs[i],&subset_size));
    n = PetscMin(subset_size,nmax);
    cum += subset_size;
    cum2 += subset_size*n;
    maxneigs = PetscMax(maxneigs,n);
  }
  lwork = 0;
  if (mss) {
    if (sub_schurs->is_symmetric) {
      PetscScalar  sdummy = 0.;
      PetscBLASInt B_itype = 1;
      PetscBLASInt B_N = mss, idummy = 0;
      PetscReal    rdummy = 0.,zero = 0.0;
      PetscReal    eps = 0.0; /* dlamch? */

      B_lwork = -1;
      /* some implementations may complain about NULL pointers, even if we are querying */
      S = &sdummy;
      St = &sdummy;
      eigs = &rdummy;
      eigv = &sdummy;
      B_iwork = &idummy;
      B_ifail = &idummy;
#if defined(PETSC_USE_COMPLEX)
      rwork = &rdummy;
#endif
      thresh = 1.0;
      CHKERRQ(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
#if defined(PETSC_USE_COMPLEX)
      PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","V","L",&B_N,St,&B_N,S,&B_N,&zero,&thresh,&B_dummyint,&B_dummyint,&eps,&B_neigs,eigs,eigv,&B_N,&lwork,&B_lwork,rwork,B_iwork,B_ifail,&B_ierr));
#else
      PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","V","L",&B_N,St,&B_N,S,&B_N,&zero,&thresh,&B_dummyint,&B_dummyint,&eps,&B_neigs,eigs,eigv,&B_N,&lwork,&B_lwork,B_iwork,B_ifail,&B_ierr));
#endif
      PetscCheckFalse(B_ierr != 0,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in query to SYGVX Lapack routine %d",(int)B_ierr);
      CHKERRQ(PetscFPTrapPop());
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not yet implemented");
  }

  nv = 0;
  if (sub_schurs->is_vertices && pcbddc->use_vertices) { /* complement set of active subsets, each entry is a vertex (boundary made by active subsets, vertices and dirichlet dofs) */
    CHKERRQ(ISGetLocalSize(sub_schurs->is_vertices,&nv));
  }
  CHKERRQ(PetscBLASIntCast((PetscInt)PetscRealPart(lwork),&B_lwork));
  if (allocated_S_St) {
    CHKERRQ(PetscMalloc2(mss*mss,&S,mss*mss,&St));
  }
  CHKERRQ(PetscMalloc5(mss*mss,&eigv,mss,&eigs,B_lwork,&work,5*mss,&B_iwork,mss,&B_ifail));
#if defined(PETSC_USE_COMPLEX)
  CHKERRQ(PetscMalloc1(7*mss,&rwork));
#endif
  ierr = PetscMalloc5(nv+sub_schurs->n_subs,&pcbddc->adaptive_constraints_n,
                      nv+sub_schurs->n_subs+1,&pcbddc->adaptive_constraints_idxs_ptr,
                      nv+sub_schurs->n_subs+1,&pcbddc->adaptive_constraints_data_ptr,
                      nv+cum,&pcbddc->adaptive_constraints_idxs,
                      nv+cum2,&pcbddc->adaptive_constraints_data);CHKERRQ(ierr);
  CHKERRQ(PetscArrayzero(pcbddc->adaptive_constraints_n,nv+sub_schurs->n_subs));

  maxneigs = 0;
  cum = cumarray = 0;
  pcbddc->adaptive_constraints_idxs_ptr[0] = 0;
  pcbddc->adaptive_constraints_data_ptr[0] = 0;
  if (sub_schurs->is_vertices && pcbddc->use_vertices) {
    const PetscInt *idxs;

    CHKERRQ(ISGetIndices(sub_schurs->is_vertices,&idxs));
    for (cum=0;cum<nv;cum++) {
      pcbddc->adaptive_constraints_n[cum] = 1;
      pcbddc->adaptive_constraints_idxs[cum] = idxs[cum];
      pcbddc->adaptive_constraints_data[cum] = 1.0;
      pcbddc->adaptive_constraints_idxs_ptr[cum+1] = pcbddc->adaptive_constraints_idxs_ptr[cum]+1;
      pcbddc->adaptive_constraints_data_ptr[cum+1] = pcbddc->adaptive_constraints_data_ptr[cum]+1;
    }
    CHKERRQ(ISRestoreIndices(sub_schurs->is_vertices,&idxs));
  }

  if (mss) { /* multilevel */
    CHKERRQ(MatSeqAIJGetArray(sub_schurs->sum_S_Ej_inv_all,&Sarray));
    CHKERRQ(MatSeqAIJGetArray(sub_schurs->sum_S_Ej_tilda_all,&Starray));
  }

  lthresh = pcbddc->adaptive_threshold[0];
  uthresh = pcbddc->adaptive_threshold[1];
  for (i=0;i<sub_schurs->n_subs;i++) {
    const PetscInt *idxs;
    PetscReal      upper,lower;
    PetscInt       j,subset_size,eigs_start = 0;
    PetscBLASInt   B_N;
    PetscBool      same_data = PETSC_FALSE;
    PetscBool      scal = PETSC_FALSE;

    if (pcbddc->use_deluxe_scaling) {
      upper = PETSC_MAX_REAL;
      lower = uthresh;
    } else {
      PetscCheckFalse(!sub_schurs->is_posdef,PETSC_COMM_SELF,PETSC_ERR_SUP,"Not yet implemented without deluxe scaling");
      upper = 1./uthresh;
      lower = 0.;
    }
    CHKERRQ(ISGetLocalSize(sub_schurs->is_subs[i],&subset_size));
    CHKERRQ(ISGetIndices(sub_schurs->is_subs[i],&idxs));
    CHKERRQ(PetscBLASIntCast(subset_size,&B_N));
    /* this is experimental: we assume the dofs have been properly grouped to have
       the diagonal blocks Schur complements either positive or negative definite (true for Stokes) */
    if (!sub_schurs->is_posdef) {
      Mat T;

      for (j=0;j<subset_size;j++) {
        if (PetscRealPart(*(Sarray+cumarray+j*(subset_size+1))) < 0.0) {
          CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,subset_size,subset_size,Sarray+cumarray,&T));
          CHKERRQ(MatScale(T,-1.0));
          CHKERRQ(MatDestroy(&T));
          CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,subset_size,subset_size,Starray+cumarray,&T));
          CHKERRQ(MatScale(T,-1.0));
          CHKERRQ(MatDestroy(&T));
          if (sub_schurs->change_primal_sub) {
            PetscInt       nz,k;
            const PetscInt *idxs;

            CHKERRQ(ISGetLocalSize(sub_schurs->change_primal_sub[i],&nz));
            CHKERRQ(ISGetIndices(sub_schurs->change_primal_sub[i],&idxs));
            for (k=0;k<nz;k++) {
              *( Sarray + cumarray + idxs[k]*(subset_size+1)) *= -1.0;
              *(Starray + cumarray + idxs[k]*(subset_size+1))  = 0.0;
            }
            CHKERRQ(ISRestoreIndices(sub_schurs->change_primal_sub[i],&idxs));
          }
          scal = PETSC_TRUE;
          break;
        }
      }
    }

    if (allocated_S_St) { /* S and S_t should be copied since we could need them later */
      if (sub_schurs->is_symmetric) {
        PetscInt j,k;
        if (sub_schurs->n_subs == 1) { /* zeroing memory to use PetscArraycmp() later */
          CHKERRQ(PetscArrayzero(S,subset_size*subset_size));
          CHKERRQ(PetscArrayzero(St,subset_size*subset_size));
        }
        for (j=0;j<subset_size;j++) {
          for (k=j;k<subset_size;k++) {
            S [j*subset_size+k] = Sarray [cumarray+j*subset_size+k];
            St[j*subset_size+k] = Starray[cumarray+j*subset_size+k];
          }
        }
      } else {
        CHKERRQ(PetscArraycpy(S,Sarray+cumarray,subset_size*subset_size));
        CHKERRQ(PetscArraycpy(St,Starray+cumarray,subset_size*subset_size));
      }
    } else {
      S = Sarray + cumarray;
      St = Starray + cumarray;
    }
    /* see if we can save some work */
    if (sub_schurs->n_subs == 1 && pcbddc->use_deluxe_scaling) {
      CHKERRQ(PetscArraycmp(S,St,subset_size*subset_size,&same_data));
    }

    if (same_data && !sub_schurs->change) { /* there's no need of constraints here */
      B_neigs = 0;
    } else {
      if (sub_schurs->is_symmetric) {
        PetscBLASInt B_itype = 1;
        PetscBLASInt B_IL, B_IU;
        PetscReal    eps = -1.0; /* dlamch? */
        PetscInt     nmin_s;
        PetscBool    compute_range;

        B_neigs = 0;
        compute_range = (PetscBool)!same_data;
        if (nmin >= subset_size) compute_range = PETSC_FALSE;

        if (pcbddc->dbg_flag) {
          PetscInt nc = 0;

          if (sub_schurs->change_primal_sub) {
            CHKERRQ(ISGetLocalSize(sub_schurs->change_primal_sub[i],&nc));
          }
          CHKERRQ(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Computing for sub %D/%D size %D count %D fid %D (range %d) (change %D).\n",i,sub_schurs->n_subs,subset_size,pcbddc->mat_graph->count[idxs[0]]+1,pcbddc->mat_graph->which_dof[idxs[0]],compute_range,nc));
        }

        CHKERRQ(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
        if (compute_range) {

          /* ask for eigenvalues larger than thresh */
          if (sub_schurs->is_posdef) {
#if defined(PETSC_USE_COMPLEX)
            PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","V","L",&B_N,St,&B_N,S,&B_N,&lower,&upper,&B_IL,&B_IU,&eps,&B_neigs,eigs,eigv,&B_N,work,&B_lwork,rwork,B_iwork,B_ifail,&B_ierr));
#else
            PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","V","L",&B_N,St,&B_N,S,&B_N,&lower,&upper,&B_IL,&B_IU,&eps,&B_neigs,eigs,eigv,&B_N,work,&B_lwork,B_iwork,B_ifail,&B_ierr));
#endif
            CHKERRQ(PetscLogFlops((4.0*subset_size*subset_size*subset_size)/3.0));
          } else { /* no theory so far, but it works nicely */
            PetscInt  recipe = 0,recipe_m = 1;
            PetscReal bb[2];

            CHKERRQ(PetscOptionsGetInt(NULL,((PetscObject)pc)->prefix,"-pc_bddc_adaptive_recipe",&recipe,NULL));
            switch (recipe) {
            case 0:
              if (scal) { bb[0] = PETSC_MIN_REAL; bb[1] = lthresh; }
              else { bb[0] = uthresh; bb[1] = PETSC_MAX_REAL; }
#if defined(PETSC_USE_COMPLEX)
              PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","V","L",&B_N,St,&B_N,S,&B_N,&bb[0],&bb[1],&B_IL,&B_IU,&eps,&B_neigs,eigs,eigv,&B_N,work,&B_lwork,rwork,B_iwork,B_ifail,&B_ierr));
#else
              PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","V","L",&B_N,St,&B_N,S,&B_N,&bb[0],&bb[1],&B_IL,&B_IU,&eps,&B_neigs,eigs,eigv,&B_N,work,&B_lwork,B_iwork,B_ifail,&B_ierr));
#endif
              CHKERRQ(PetscLogFlops((4.0*subset_size*subset_size*subset_size)/3.0));
              break;
            case 1:
              bb[0] = PETSC_MIN_REAL; bb[1] = lthresh*lthresh;
#if defined(PETSC_USE_COMPLEX)
              PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","V","L",&B_N,St,&B_N,S,&B_N,&bb[0],&bb[1],&B_IL,&B_IU,&eps,&B_neigs,eigs,eigv,&B_N,work,&B_lwork,rwork,B_iwork,B_ifail,&B_ierr));
#else
              PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","V","L",&B_N,St,&B_N,S,&B_N,&bb[0],&bb[1],&B_IL,&B_IU,&eps,&B_neigs,eigs,eigv,&B_N,work,&B_lwork,B_iwork,B_ifail,&B_ierr));
#endif
              CHKERRQ(PetscLogFlops((4.0*subset_size*subset_size*subset_size)/3.0));
              if (!scal) {
                PetscBLASInt B_neigs2 = 0;

                bb[0] = PetscMax(lthresh*lthresh,uthresh); bb[1] = PETSC_MAX_REAL;
                CHKERRQ(PetscArraycpy(S,Sarray+cumarray,subset_size*subset_size));
                CHKERRQ(PetscArraycpy(St,Starray+cumarray,subset_size*subset_size));
#if defined(PETSC_USE_COMPLEX)
                PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","V","L",&B_N,St,&B_N,S,&B_N,&bb[0],&bb[1],&B_IL,&B_IU,&eps,&B_neigs2,eigs+B_neigs,eigv+B_neigs*B_N,&B_N,work,&B_lwork,rwork,B_iwork,B_ifail,&B_ierr));
#else
                PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","V","L",&B_N,St,&B_N,S,&B_N,&bb[0],&bb[1],&B_IL,&B_IU,&eps,&B_neigs2,eigs+B_neigs,eigv+B_neigs*B_N,&B_N,work,&B_lwork,B_iwork,B_ifail,&B_ierr));
#endif
                CHKERRQ(PetscLogFlops((4.0*subset_size*subset_size*subset_size)/3.0));
                B_neigs += B_neigs2;
              }
              break;
            case 2:
              if (scal) {
                bb[0] = PETSC_MIN_REAL;
                bb[1] = 0;
#if defined(PETSC_USE_COMPLEX)
                PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","V","L",&B_N,St,&B_N,S,&B_N,&bb[0],&bb[1],&B_IL,&B_IU,&eps,&B_neigs,eigs,eigv,&B_N,work,&B_lwork,rwork,B_iwork,B_ifail,&B_ierr));
#else
                PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","V","L",&B_N,St,&B_N,S,&B_N,&bb[0],&bb[1],&B_IL,&B_IU,&eps,&B_neigs,eigs,eigv,&B_N,work,&B_lwork,B_iwork,B_ifail,&B_ierr));
#endif
                CHKERRQ(PetscLogFlops((4.0*subset_size*subset_size*subset_size)/3.0));
              } else {
                PetscBLASInt B_neigs2 = 0;
                PetscBool    import = PETSC_FALSE;

                lthresh = PetscMax(lthresh,0.0);
                if (lthresh > 0.0) {
                  bb[0] = PETSC_MIN_REAL;
                  bb[1] = lthresh*lthresh;

                  import = PETSC_TRUE;
#if defined(PETSC_USE_COMPLEX)
                  PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","V","L",&B_N,St,&B_N,S,&B_N,&bb[0],&bb[1],&B_IL,&B_IU,&eps,&B_neigs,eigs,eigv,&B_N,work,&B_lwork,rwork,B_iwork,B_ifail,&B_ierr));
#else
                  PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","V","L",&B_N,St,&B_N,S,&B_N,&bb[0],&bb[1],&B_IL,&B_IU,&eps,&B_neigs,eigs,eigv,&B_N,work,&B_lwork,B_iwork,B_ifail,&B_ierr));
#endif
                  CHKERRQ(PetscLogFlops((4.0*subset_size*subset_size*subset_size)/3.0));
                }
                bb[0] = PetscMax(lthresh*lthresh,uthresh);
                bb[1] = PETSC_MAX_REAL;
                if (import) {
                  CHKERRQ(PetscArraycpy(S,Sarray+cumarray,subset_size*subset_size));
                  CHKERRQ(PetscArraycpy(St,Starray+cumarray,subset_size*subset_size));
                }
#if defined(PETSC_USE_COMPLEX)
                PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","V","L",&B_N,St,&B_N,S,&B_N,&bb[0],&bb[1],&B_IL,&B_IU,&eps,&B_neigs2,eigs+B_neigs,eigv+B_neigs*B_N,&B_N,work,&B_lwork,rwork,B_iwork,B_ifail,&B_ierr));
#else
                PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","V","L",&B_N,St,&B_N,S,&B_N,&bb[0],&bb[1],&B_IL,&B_IU,&eps,&B_neigs2,eigs+B_neigs,eigv+B_neigs*B_N,&B_N,work,&B_lwork,B_iwork,B_ifail,&B_ierr));
#endif
                CHKERRQ(PetscLogFlops((4.0*subset_size*subset_size*subset_size)/3.0));
                B_neigs += B_neigs2;
              }
              break;
            case 3:
              if (scal) {
                CHKERRQ(PetscOptionsGetInt(NULL,((PetscObject)pc)->prefix,"-pc_bddc_adaptive_recipe3_min_scal",&recipe_m,NULL));
              } else {
                CHKERRQ(PetscOptionsGetInt(NULL,((PetscObject)pc)->prefix,"-pc_bddc_adaptive_recipe3_min",&recipe_m,NULL));
              }
              if (!scal) {
                bb[0] = uthresh;
                bb[1] = PETSC_MAX_REAL;
#if defined(PETSC_USE_COMPLEX)
                PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","V","L",&B_N,St,&B_N,S,&B_N,&bb[0],&bb[1],&B_IL,&B_IU,&eps,&B_neigs,eigs,eigv,&B_N,work,&B_lwork,rwork,B_iwork,B_ifail,&B_ierr));
#else
                PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","V","L",&B_N,St,&B_N,S,&B_N,&bb[0],&bb[1],&B_IL,&B_IU,&eps,&B_neigs,eigs,eigv,&B_N,work,&B_lwork,B_iwork,B_ifail,&B_ierr));
#endif
                CHKERRQ(PetscLogFlops((4.0*subset_size*subset_size*subset_size)/3.0));
              }
              if (recipe_m > 0 && B_N - B_neigs > 0) {
                PetscBLASInt B_neigs2 = 0;

                B_IL = 1;
                CHKERRQ(PetscBLASIntCast(PetscMin(recipe_m,B_N - B_neigs),&B_IU));
                CHKERRQ(PetscArraycpy(S,Sarray+cumarray,subset_size*subset_size));
                CHKERRQ(PetscArraycpy(St,Starray+cumarray,subset_size*subset_size));
#if defined(PETSC_USE_COMPLEX)
                PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","I","L",&B_N,St,&B_N,S,&B_N,&lower,&upper,&B_IL,&B_IU,&eps,&B_neigs2,eigs+B_neigs,eigv+B_neigs*B_N,&B_N,work,&B_lwork,rwork,B_iwork,B_ifail,&B_ierr));
#else
                PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","I","L",&B_N,St,&B_N,S,&B_N,&lower,&upper,&B_IL,&B_IU,&eps,&B_neigs2,eigs+B_neigs,eigv+B_neigs*B_N,&B_N,work,&B_lwork,B_iwork,B_ifail,&B_ierr));
#endif
                CHKERRQ(PetscLogFlops((4.0*subset_size*subset_size*subset_size)/3.0));
                B_neigs += B_neigs2;
              }
              break;
            case 4:
              bb[0] = PETSC_MIN_REAL; bb[1] = lthresh;
#if defined(PETSC_USE_COMPLEX)
              PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","V","L",&B_N,St,&B_N,S,&B_N,&bb[0],&bb[1],&B_IL,&B_IU,&eps,&B_neigs,eigs,eigv,&B_N,work,&B_lwork,rwork,B_iwork,B_ifail,&B_ierr));
#else
              PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","V","L",&B_N,St,&B_N,S,&B_N,&bb[0],&bb[1],&B_IL,&B_IU,&eps,&B_neigs,eigs,eigv,&B_N,work,&B_lwork,B_iwork,B_ifail,&B_ierr));
#endif
              CHKERRQ(PetscLogFlops((4.0*subset_size*subset_size*subset_size)/3.0));
              {
                PetscBLASInt B_neigs2 = 0;

                bb[0] = PetscMax(lthresh+PETSC_SMALL,uthresh); bb[1] = PETSC_MAX_REAL;
                CHKERRQ(PetscArraycpy(S,Sarray+cumarray,subset_size*subset_size));
                CHKERRQ(PetscArraycpy(St,Starray+cumarray,subset_size*subset_size));
#if defined(PETSC_USE_COMPLEX)
                PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","V","L",&B_N,St,&B_N,S,&B_N,&bb[0],&bb[1],&B_IL,&B_IU,&eps,&B_neigs2,eigs+B_neigs,eigv+B_neigs*B_N,&B_N,work,&B_lwork,rwork,B_iwork,B_ifail,&B_ierr));
#else
                PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","V","L",&B_N,St,&B_N,S,&B_N,&bb[0],&bb[1],&B_IL,&B_IU,&eps,&B_neigs2,eigs+B_neigs,eigv+B_neigs*B_N,&B_N,work,&B_lwork,B_iwork,B_ifail,&B_ierr));
#endif
                CHKERRQ(PetscLogFlops((4.0*subset_size*subset_size*subset_size)/3.0));
                B_neigs += B_neigs2;
              }
              break;
            case 5: /* same as before: first compute all eigenvalues, then filter */
#if defined(PETSC_USE_COMPLEX)
              PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","A","L",&B_N,St,&B_N,S,&B_N,&bb[0],&bb[1],&B_IL,&B_IU,&eps,&B_neigs,eigs,eigv,&B_N,work,&B_lwork,rwork,B_iwork,B_ifail,&B_ierr));
#else
              PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","A","L",&B_N,St,&B_N,S,&B_N,&bb[0],&bb[1],&B_IL,&B_IU,&eps,&B_neigs,eigs,eigv,&B_N,work,&B_lwork,B_iwork,B_ifail,&B_ierr));
#endif
              CHKERRQ(PetscLogFlops((4.0*subset_size*subset_size*subset_size)/3.0));
              {
                PetscInt e,k,ne;
                for (e=0,ne=0;e<B_neigs;e++) {
                  if (eigs[e] < lthresh || eigs[e] > uthresh) {
                    for (k=0;k<B_N;k++) S[ne*B_N+k] = eigv[e*B_N+k];
                    eigs[ne] = eigs[e];
                    ne++;
                  }
                }
                CHKERRQ(PetscArraycpy(eigv,S,B_N*ne));
                B_neigs = ne;
              }
              break;
            default:
              SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Unknown recipe %D",recipe);
            }
          }
        } else if (!same_data) { /* this is just to see all the eigenvalues */
          B_IU = PetscMax(1,PetscMin(B_N,nmax));
          B_IL = 1;
#if defined(PETSC_USE_COMPLEX)
          PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","I","L",&B_N,St,&B_N,S,&B_N,&lower,&upper,&B_IL,&B_IU,&eps,&B_neigs,eigs,eigv,&B_N,work,&B_lwork,rwork,B_iwork,B_ifail,&B_ierr));
#else
          PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","I","L",&B_N,St,&B_N,S,&B_N,&lower,&upper,&B_IL,&B_IU,&eps,&B_neigs,eigs,eigv,&B_N,work,&B_lwork,B_iwork,B_ifail,&B_ierr));
#endif
          CHKERRQ(PetscLogFlops((4.0*subset_size*subset_size*subset_size)/3.0));
        } else { /* same_data is true, so just get the adaptive functional requested by the user */
          PetscInt k;
          PetscCheckFalse(!sub_schurs->change_primal_sub,PETSC_COMM_SELF,PETSC_ERR_PLIB,"This should not happen");
          CHKERRQ(ISGetLocalSize(sub_schurs->change_primal_sub[i],&nmax));
          CHKERRQ(PetscBLASIntCast(nmax,&B_neigs));
          nmin = nmax;
          CHKERRQ(PetscArrayzero(eigv,subset_size*nmax));
          for (k=0;k<nmax;k++) {
            eigs[k] = 1./PETSC_SMALL;
            eigv[k*(subset_size+1)] = 1.0;
          }
        }
        CHKERRQ(PetscFPTrapPop());
        if (B_ierr) {
          PetscCheckFalse(B_ierr < 0,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in SYGVX Lapack routine: illegal value for argument %d",-(int)B_ierr);
          else PetscCheckFalse(B_ierr <= B_N,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in SYGVX Lapack routine: %d eigenvalues failed to converge",(int)B_ierr);
          else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in SYGVX Lapack routine: leading minor of order %d is not positive definite",(int)B_ierr-B_N-1);
        }

        if (B_neigs > nmax) {
          if (pcbddc->dbg_flag) {
            CHKERRQ(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"   found %d eigs, more than maximum required %D.\n",B_neigs,nmax));
          }
          if (pcbddc->use_deluxe_scaling) eigs_start = scal ? 0 : B_neigs-nmax;
          B_neigs = nmax;
        }

        nmin_s = PetscMin(nmin,B_N);
        if (B_neigs < nmin_s) {
          PetscBLASInt B_neigs2 = 0;

          if (pcbddc->use_deluxe_scaling) {
            if (scal) {
              B_IU = nmin_s;
              B_IL = B_neigs + 1;
            } else {
              B_IL = B_N - nmin_s + 1;
              B_IU = B_N - B_neigs;
            }
          } else {
            B_IL = B_neigs + 1;
            B_IU = nmin_s;
          }
          if (pcbddc->dbg_flag) {
            CHKERRQ(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"   found %d eigs, less than minimum required %D. Asking for %d to %d incl (fortran like)\n",B_neigs,nmin,B_IL,B_IU));
          }
          if (sub_schurs->is_symmetric) {
            PetscInt j,k;
            for (j=0;j<subset_size;j++) {
              for (k=j;k<subset_size;k++) {
                S [j*subset_size+k] = Sarray [cumarray+j*subset_size+k];
                St[j*subset_size+k] = Starray[cumarray+j*subset_size+k];
              }
            }
          } else {
            CHKERRQ(PetscArraycpy(S,Sarray+cumarray,subset_size*subset_size));
            CHKERRQ(PetscArraycpy(St,Starray+cumarray,subset_size*subset_size));
          }
          CHKERRQ(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
#if defined(PETSC_USE_COMPLEX)
          PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","I","L",&B_N,St,&B_N,S,&B_N,&lower,&upper,&B_IL,&B_IU,&eps,&B_neigs2,eigs+B_neigs,eigv+B_neigs*subset_size,&B_N,work,&B_lwork,rwork,B_iwork,B_ifail,&B_ierr));
#else
          PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","I","L",&B_N,St,&B_N,S,&B_N,&lower,&upper,&B_IL,&B_IU,&eps,&B_neigs2,eigs+B_neigs,eigv+B_neigs*subset_size,&B_N,work,&B_lwork,B_iwork,B_ifail,&B_ierr));
#endif
          CHKERRQ(PetscLogFlops((4.0*subset_size*subset_size*subset_size)/3.0));
          CHKERRQ(PetscFPTrapPop());
          B_neigs += B_neigs2;
        }
        if (B_ierr) {
          PetscCheckFalse(B_ierr < 0,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in SYGVX Lapack routine: illegal value for argument %d",-(int)B_ierr);
          else PetscCheckFalse(B_ierr <= B_N,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in SYGVX Lapack routine: %d eigenvalues failed to converge",(int)B_ierr);
          else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in SYGVX Lapack routine: leading minor of order %d is not positive definite",(int)B_ierr-B_N-1);
        }
        if (pcbddc->dbg_flag) {
          CHKERRQ(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"   -> Got %d eigs\n",B_neigs));
          for (j=0;j<B_neigs;j++) {
            if (eigs[j] == 0.0) {
              CHKERRQ(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"     Inf\n"));
            } else {
              if (pcbddc->use_deluxe_scaling) {
                CHKERRQ(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"     %1.6e\n",eigs[j+eigs_start]));
              } else {
                CHKERRQ(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"     %1.6e\n",1./eigs[j+eigs_start]));
              }
            }
          }
        }
      } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not yet implemented");
    }
    /* change the basis back to the original one */
    if (sub_schurs->change) {
      Mat change,phi,phit;

      if (pcbddc->dbg_flag > 2) {
        PetscInt ii;
        for (ii=0;ii<B_neigs;ii++) {
          CHKERRQ(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"   -> Eigenvector (old basis) %d/%d (%d)\n",ii,B_neigs,B_N));
          for (j=0;j<B_N;j++) {
#if defined(PETSC_USE_COMPLEX)
            PetscReal r = PetscRealPart(eigv[(ii+eigs_start)*subset_size+j]);
            PetscReal c = PetscImaginaryPart(eigv[(ii+eigs_start)*subset_size+j]);
            CHKERRQ(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"       %1.4e + %1.4e i\n",r,c));
#else
            CHKERRQ(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"       %1.4e\n",eigv[(ii+eigs_start)*subset_size+j]));
#endif
          }
        }
      }
      CHKERRQ(KSPGetOperators(sub_schurs->change[i],&change,NULL));
      CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,subset_size,B_neigs,eigv+eigs_start*subset_size,&phit));
      CHKERRQ(MatMatMult(change,phit,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&phi));
      CHKERRQ(MatCopy(phi,phit,SAME_NONZERO_PATTERN));
      CHKERRQ(MatDestroy(&phit));
      CHKERRQ(MatDestroy(&phi));
    }
    maxneigs = PetscMax(B_neigs,maxneigs);
    pcbddc->adaptive_constraints_n[i+nv] = B_neigs;
    if (B_neigs) {
      CHKERRQ(PetscArraycpy(pcbddc->adaptive_constraints_data+pcbddc->adaptive_constraints_data_ptr[cum],eigv+eigs_start*subset_size,B_neigs*subset_size));

      if (pcbddc->dbg_flag > 1) {
        PetscInt ii;
        for (ii=0;ii<B_neigs;ii++) {
          CHKERRQ(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"   -> Eigenvector %d/%d (%d)\n",ii,B_neigs,B_N));
          for (j=0;j<B_N;j++) {
#if defined(PETSC_USE_COMPLEX)
            PetscReal r = PetscRealPart(pcbddc->adaptive_constraints_data[ii*subset_size+j+pcbddc->adaptive_constraints_data_ptr[cum]]);
            PetscReal c = PetscImaginaryPart(pcbddc->adaptive_constraints_data[ii*subset_size+j+pcbddc->adaptive_constraints_data_ptr[cum]]);
            CHKERRQ(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"       %1.4e + %1.4e i\n",r,c));
#else
            CHKERRQ(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"       %1.4e\n",pcbddc->adaptive_constraints_data[ii*subset_size+j+pcbddc->adaptive_constraints_data_ptr[cum]]));
#endif
          }
        }
      }
      CHKERRQ(PetscArraycpy(pcbddc->adaptive_constraints_idxs+pcbddc->adaptive_constraints_idxs_ptr[cum],idxs,subset_size));
      pcbddc->adaptive_constraints_idxs_ptr[cum+1] = pcbddc->adaptive_constraints_idxs_ptr[cum] + subset_size;
      pcbddc->adaptive_constraints_data_ptr[cum+1] = pcbddc->adaptive_constraints_data_ptr[cum] + subset_size*B_neigs;
      cum++;
    }
    CHKERRQ(ISRestoreIndices(sub_schurs->is_subs[i],&idxs));
    /* shift for next computation */
    cumarray += subset_size*subset_size;
  }
  if (pcbddc->dbg_flag) {
    CHKERRQ(PetscViewerFlush(pcbddc->dbg_viewer));
  }

  if (mss) {
    CHKERRQ(MatSeqAIJRestoreArray(sub_schurs->sum_S_Ej_inv_all,&Sarray));
    CHKERRQ(MatSeqAIJRestoreArray(sub_schurs->sum_S_Ej_tilda_all,&Starray));
    /* destroy matrices (junk) */
    CHKERRQ(MatDestroy(&sub_schurs->sum_S_Ej_inv_all));
    CHKERRQ(MatDestroy(&sub_schurs->sum_S_Ej_tilda_all));
  }
  if (allocated_S_St) {
    CHKERRQ(PetscFree2(S,St));
  }
  CHKERRQ(PetscFree5(eigv,eigs,work,B_iwork,B_ifail));
#if defined(PETSC_USE_COMPLEX)
  CHKERRQ(PetscFree(rwork));
#endif
  if (pcbddc->dbg_flag) {
    PetscInt maxneigs_r;
    CHKERRMPI(MPIU_Allreduce(&maxneigs,&maxneigs_r,1,MPIU_INT,MPI_MAX,PetscObjectComm((PetscObject)pc)));
    CHKERRQ(PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"Maximum number of constraints per cc %D\n",maxneigs_r));
  }
  CHKERRQ(PetscLogEventEnd(PC_BDDC_AdaptiveSetUp[pcbddc->current_level],pc,0,0,0));
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCSetUpSolvers(PC pc)
{
  PetscScalar    *coarse_submat_vals;

  PetscFunctionBegin;
  /* Setup local scatters R_to_B and (optionally) R_to_D */
  /* PCBDDCSetUpLocalWorkVectors should be called first! */
  CHKERRQ(PCBDDCSetUpLocalScatters(pc));

  /* Setup local neumann solver ksp_R */
  /* PCBDDCSetUpLocalScatters should be called first! */
  CHKERRQ(PCBDDCSetUpLocalSolvers(pc,PETSC_FALSE,PETSC_TRUE));

  /*
     Setup local correction and local part of coarse basis.
     Gives back the dense local part of the coarse matrix in column major ordering
  */
  CHKERRQ(PCBDDCSetUpCorrection(pc,&coarse_submat_vals));

  /* Compute total number of coarse nodes and setup coarse solver */
  CHKERRQ(PCBDDCSetUpCoarseSolver(pc,coarse_submat_vals));

  /* free */
  CHKERRQ(PetscFree(coarse_submat_vals));
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCResetCustomization(PC pc)
{
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;

  PetscFunctionBegin;
  CHKERRQ(ISDestroy(&pcbddc->user_primal_vertices));
  CHKERRQ(ISDestroy(&pcbddc->user_primal_vertices_local));
  CHKERRQ(ISDestroy(&pcbddc->NeumannBoundaries));
  CHKERRQ(ISDestroy(&pcbddc->NeumannBoundariesLocal));
  CHKERRQ(ISDestroy(&pcbddc->DirichletBoundaries));
  CHKERRQ(MatNullSpaceDestroy(&pcbddc->onearnullspace));
  CHKERRQ(PetscFree(pcbddc->onearnullvecs_state));
  CHKERRQ(ISDestroy(&pcbddc->DirichletBoundariesLocal));
  CHKERRQ(PCBDDCSetDofsSplitting(pc,0,NULL));
  CHKERRQ(PCBDDCSetDofsSplittingLocal(pc,0,NULL));
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCResetTopography(PC pc)
{
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;
  PetscInt       i;

  PetscFunctionBegin;
  CHKERRQ(MatDestroy(&pcbddc->nedcG));
  CHKERRQ(ISDestroy(&pcbddc->nedclocal));
  CHKERRQ(MatDestroy(&pcbddc->discretegradient));
  CHKERRQ(MatDestroy(&pcbddc->user_ChangeOfBasisMatrix));
  CHKERRQ(MatDestroy(&pcbddc->ChangeOfBasisMatrix));
  CHKERRQ(MatDestroy(&pcbddc->switch_static_change));
  CHKERRQ(VecDestroy(&pcbddc->work_change));
  CHKERRQ(MatDestroy(&pcbddc->ConstraintMatrix));
  CHKERRQ(MatDestroy(&pcbddc->divudotp));
  CHKERRQ(ISDestroy(&pcbddc->divudotp_vl2l));
  CHKERRQ(PCBDDCGraphDestroy(&pcbddc->mat_graph));
  for (i=0;i<pcbddc->n_local_subs;i++) {
    CHKERRQ(ISDestroy(&pcbddc->local_subs[i]));
  }
  pcbddc->n_local_subs = 0;
  CHKERRQ(PetscFree(pcbddc->local_subs));
  CHKERRQ(PCBDDCSubSchursDestroy(&pcbddc->sub_schurs));
  pcbddc->graphanalyzed        = PETSC_FALSE;
  pcbddc->recompute_topography = PETSC_TRUE;
  pcbddc->corner_selected      = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCResetSolvers(PC pc)
{
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;

  PetscFunctionBegin;
  CHKERRQ(VecDestroy(&pcbddc->coarse_vec));
  if (pcbddc->coarse_phi_B) {
    PetscScalar *array;
    CHKERRQ(MatDenseGetArray(pcbddc->coarse_phi_B,&array));
    CHKERRQ(PetscFree(array));
  }
  CHKERRQ(MatDestroy(&pcbddc->coarse_phi_B));
  CHKERRQ(MatDestroy(&pcbddc->coarse_phi_D));
  CHKERRQ(MatDestroy(&pcbddc->coarse_psi_B));
  CHKERRQ(MatDestroy(&pcbddc->coarse_psi_D));
  CHKERRQ(VecDestroy(&pcbddc->vec1_P));
  CHKERRQ(VecDestroy(&pcbddc->vec1_C));
  CHKERRQ(MatDestroy(&pcbddc->local_auxmat2));
  CHKERRQ(MatDestroy(&pcbddc->local_auxmat1));
  CHKERRQ(VecDestroy(&pcbddc->vec1_R));
  CHKERRQ(VecDestroy(&pcbddc->vec2_R));
  CHKERRQ(ISDestroy(&pcbddc->is_R_local));
  CHKERRQ(VecScatterDestroy(&pcbddc->R_to_B));
  CHKERRQ(VecScatterDestroy(&pcbddc->R_to_D));
  CHKERRQ(VecScatterDestroy(&pcbddc->coarse_loc_to_glob));
  CHKERRQ(KSPReset(pcbddc->ksp_D));
  CHKERRQ(KSPReset(pcbddc->ksp_R));
  CHKERRQ(KSPReset(pcbddc->coarse_ksp));
  CHKERRQ(MatDestroy(&pcbddc->local_mat));
  CHKERRQ(PetscFree(pcbddc->primal_indices_local_idxs));
  CHKERRQ(PetscFree2(pcbddc->local_primal_ref_node,pcbddc->local_primal_ref_mult));
  CHKERRQ(PetscFree(pcbddc->global_primal_indices));
  CHKERRQ(ISDestroy(&pcbddc->coarse_subassembling));
  CHKERRQ(MatDestroy(&pcbddc->benign_change));
  CHKERRQ(VecDestroy(&pcbddc->benign_vec));
  CHKERRQ(PCBDDCBenignShellMat(pc,PETSC_TRUE));
  CHKERRQ(MatDestroy(&pcbddc->benign_B0));
  CHKERRQ(PetscSFDestroy(&pcbddc->benign_sf));
  if (pcbddc->benign_zerodiag_subs) {
    PetscInt i;
    for (i=0;i<pcbddc->benign_n;i++) {
      CHKERRQ(ISDestroy(&pcbddc->benign_zerodiag_subs[i]));
    }
    CHKERRQ(PetscFree(pcbddc->benign_zerodiag_subs));
  }
  CHKERRQ(PetscFree3(pcbddc->benign_p0_lidx,pcbddc->benign_p0_gidx,pcbddc->benign_p0));
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCSetUpLocalWorkVectors(PC pc)
{
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;
  PC_IS          *pcis = (PC_IS*)pc->data;
  VecType        impVecType;
  PetscInt       n_constraints,n_R,old_size;

  PetscFunctionBegin;
  n_constraints = pcbddc->local_primal_size - pcbddc->benign_n - pcbddc->n_vertices;
  n_R = pcis->n - pcbddc->n_vertices;
  CHKERRQ(VecGetType(pcis->vec1_N,&impVecType));
  /* local work vectors (try to avoid unneeded work)*/
  /* R nodes */
  old_size = -1;
  if (pcbddc->vec1_R) {
    CHKERRQ(VecGetSize(pcbddc->vec1_R,&old_size));
  }
  if (n_R != old_size) {
    CHKERRQ(VecDestroy(&pcbddc->vec1_R));
    CHKERRQ(VecDestroy(&pcbddc->vec2_R));
    CHKERRQ(VecCreate(PetscObjectComm((PetscObject)pcis->vec1_N),&pcbddc->vec1_R));
    CHKERRQ(VecSetSizes(pcbddc->vec1_R,PETSC_DECIDE,n_R));
    CHKERRQ(VecSetType(pcbddc->vec1_R,impVecType));
    CHKERRQ(VecDuplicate(pcbddc->vec1_R,&pcbddc->vec2_R));
  }
  /* local primal dofs */
  old_size = -1;
  if (pcbddc->vec1_P) {
    CHKERRQ(VecGetSize(pcbddc->vec1_P,&old_size));
  }
  if (pcbddc->local_primal_size != old_size) {
    CHKERRQ(VecDestroy(&pcbddc->vec1_P));
    CHKERRQ(VecCreate(PetscObjectComm((PetscObject)pcis->vec1_N),&pcbddc->vec1_P));
    CHKERRQ(VecSetSizes(pcbddc->vec1_P,PETSC_DECIDE,pcbddc->local_primal_size));
    CHKERRQ(VecSetType(pcbddc->vec1_P,impVecType));
  }
  /* local explicit constraints */
  old_size = -1;
  if (pcbddc->vec1_C) {
    CHKERRQ(VecGetSize(pcbddc->vec1_C,&old_size));
  }
  if (n_constraints && n_constraints != old_size) {
    CHKERRQ(VecDestroy(&pcbddc->vec1_C));
    CHKERRQ(VecCreate(PetscObjectComm((PetscObject)pcis->vec1_N),&pcbddc->vec1_C));
    CHKERRQ(VecSetSizes(pcbddc->vec1_C,PETSC_DECIDE,n_constraints));
    CHKERRQ(VecSetType(pcbddc->vec1_C,impVecType));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCSetUpCorrection(PC pc, PetscScalar **coarse_submat_vals_n)
{
  /* pointers to pcis and pcbddc */
  PC_IS*          pcis = (PC_IS*)pc->data;
  PC_BDDC*        pcbddc = (PC_BDDC*)pc->data;
  PCBDDCSubSchurs sub_schurs = pcbddc->sub_schurs;
  /* submatrices of local problem */
  Mat             A_RV,A_VR,A_VV,local_auxmat2_R;
  /* submatrices of local coarse problem */
  Mat             S_VV,S_CV,S_VC,S_CC;
  /* working matrices */
  Mat             C_CR;
  /* additional working stuff */
  PC              pc_R;
  Mat             F,Brhs = NULL;
  Vec             dummy_vec;
  PetscBool       isLU,isCHOL,need_benign_correction,sparserhs;
  PetscScalar     *coarse_submat_vals; /* TODO: use a PETSc matrix */
  PetscScalar     *work;
  PetscInt        *idx_V_B;
  PetscInt        lda_rhs,n,n_vertices,n_constraints,*p0_lidx_I;
  PetscInt        i,n_R,n_D,n_B;
  PetscScalar     one=1.0,m_one=-1.0;

  PetscFunctionBegin;
  PetscCheckFalse(!pcbddc->symmetric_primal && pcbddc->benign_n,PETSC_COMM_SELF,PETSC_ERR_SUP,"Non-symmetric primal basis computation with benign trick not yet implemented");
  CHKERRQ(PetscLogEventBegin(PC_BDDC_CorrectionSetUp[pcbddc->current_level],pc,0,0,0));

  /* Set Non-overlapping dimensions */
  n_vertices = pcbddc->n_vertices;
  n_constraints = pcbddc->local_primal_size - pcbddc->benign_n - n_vertices;
  n_B = pcis->n_B;
  n_D = pcis->n - n_B;
  n_R = pcis->n - n_vertices;

  /* vertices in boundary numbering */
  CHKERRQ(PetscMalloc1(n_vertices,&idx_V_B));
  CHKERRQ(ISGlobalToLocalMappingApply(pcis->BtoNmap,IS_GTOLM_DROP,n_vertices,pcbddc->local_primal_ref_node,&i,idx_V_B));
  PetscCheckFalse(i != n_vertices,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error in boundary numbering for BDDC vertices! %D != %D",n_vertices,i);

  /* Subdomain contribution (Non-overlapping) to coarse matrix  */
  CHKERRQ(PetscCalloc1(pcbddc->local_primal_size*pcbddc->local_primal_size,&coarse_submat_vals));
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,n_vertices,n_vertices,coarse_submat_vals,&S_VV));
  CHKERRQ(MatDenseSetLDA(S_VV,pcbddc->local_primal_size));
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,n_constraints,n_vertices,coarse_submat_vals+n_vertices,&S_CV));
  CHKERRQ(MatDenseSetLDA(S_CV,pcbddc->local_primal_size));
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,n_vertices,n_constraints,coarse_submat_vals+pcbddc->local_primal_size*n_vertices,&S_VC));
  CHKERRQ(MatDenseSetLDA(S_VC,pcbddc->local_primal_size));
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,n_constraints,n_constraints,coarse_submat_vals+(pcbddc->local_primal_size+1)*n_vertices,&S_CC));
  CHKERRQ(MatDenseSetLDA(S_CC,pcbddc->local_primal_size));

  /* determine if can use MatSolve routines instead of calling KSPSolve on ksp_R */
  CHKERRQ(KSPGetPC(pcbddc->ksp_R,&pc_R));
  CHKERRQ(PCSetUp(pc_R));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)pc_R,PCLU,&isLU));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)pc_R,PCCHOLESKY,&isCHOL));
  lda_rhs = n_R;
  need_benign_correction = PETSC_FALSE;
  if (isLU || isCHOL) {
    CHKERRQ(PCFactorGetMatrix(pc_R,&F));
  } else if (sub_schurs && sub_schurs->reuse_solver) {
    PCBDDCReuseSolvers reuse_solver = sub_schurs->reuse_solver;
    MatFactorType      type;

    F = reuse_solver->F;
    CHKERRQ(MatGetFactorType(F,&type));
    if (type == MAT_FACTOR_CHOLESKY) isCHOL = PETSC_TRUE;
    if (type == MAT_FACTOR_LU) isLU = PETSC_TRUE;
    CHKERRQ(MatGetSize(F,&lda_rhs,NULL));
    need_benign_correction = (PetscBool)(!!reuse_solver->benign_n);
  } else F = NULL;

  /* determine if we can use a sparse right-hand side */
  sparserhs = PETSC_FALSE;
  if (F) {
    MatSolverType solver;

    CHKERRQ(MatFactorGetSolverType(F,&solver));
    CHKERRQ(PetscStrcmp(solver,MATSOLVERMUMPS,&sparserhs));
  }

  /* allocate workspace */
  n = 0;
  if (n_constraints) {
    n += lda_rhs*n_constraints;
  }
  if (n_vertices) {
    n = PetscMax(2*lda_rhs*n_vertices,n);
    n = PetscMax((lda_rhs+n_B)*n_vertices,n);
  }
  if (!pcbddc->symmetric_primal) {
    n = PetscMax(2*lda_rhs*pcbddc->local_primal_size,n);
  }
  CHKERRQ(PetscMalloc1(n,&work));

  /* create dummy vector to modify rhs and sol of MatMatSolve (work array will never be used) */
  dummy_vec = NULL;
  if (need_benign_correction && lda_rhs != n_R && F) {
    CHKERRQ(VecCreate(PetscObjectComm((PetscObject)pcis->vec1_N),&dummy_vec));
    CHKERRQ(VecSetSizes(dummy_vec,lda_rhs,PETSC_DECIDE));
    CHKERRQ(VecSetType(dummy_vec,((PetscObject)pcis->vec1_N)->type_name));
  }

  CHKERRQ(MatDestroy(&pcbddc->local_auxmat1));
  CHKERRQ(MatDestroy(&pcbddc->local_auxmat2));

  /* Precompute stuffs needed for preprocessing and application of BDDC*/
  if (n_constraints) {
    Mat         M3,C_B;
    IS          is_aux;

    /* Extract constraints on R nodes: C_{CR}  */
    CHKERRQ(ISCreateStride(PETSC_COMM_SELF,n_constraints,n_vertices,1,&is_aux));
    CHKERRQ(MatCreateSubMatrix(pcbddc->ConstraintMatrix,is_aux,pcbddc->is_R_local,MAT_INITIAL_MATRIX,&C_CR));
    CHKERRQ(MatCreateSubMatrix(pcbddc->ConstraintMatrix,is_aux,pcis->is_B_local,MAT_INITIAL_MATRIX,&C_B));

    /* Assemble         local_auxmat2_R =        (- A_{RR}^{-1} C^T_{CR}) needed by BDDC setup */
    /* Assemble pcbddc->local_auxmat2   = R_to_B (- A_{RR}^{-1} C^T_{CR}) needed by BDDC application */
    if (!sparserhs) {
      CHKERRQ(PetscArrayzero(work,lda_rhs*n_constraints));
      for (i=0;i<n_constraints;i++) {
        const PetscScalar *row_cmat_values;
        const PetscInt    *row_cmat_indices;
        PetscInt          size_of_constraint,j;

        CHKERRQ(MatGetRow(C_CR,i,&size_of_constraint,&row_cmat_indices,&row_cmat_values));
        for (j=0;j<size_of_constraint;j++) {
          work[row_cmat_indices[j]+i*lda_rhs] = -row_cmat_values[j];
        }
        CHKERRQ(MatRestoreRow(C_CR,i,&size_of_constraint,&row_cmat_indices,&row_cmat_values));
      }
      CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,lda_rhs,n_constraints,work,&Brhs));
    } else {
      Mat tC_CR;

      CHKERRQ(MatScale(C_CR,-1.0));
      if (lda_rhs != n_R) {
        PetscScalar *aa;
        PetscInt    r,*ii,*jj;
        PetscBool   done;

        CHKERRQ(MatGetRowIJ(C_CR,0,PETSC_FALSE,PETSC_FALSE,&r,(const PetscInt**)&ii,(const PetscInt**)&jj,&done));
        PetscCheckFalse(!done,PETSC_COMM_SELF,PETSC_ERR_PLIB,"GetRowIJ failed");
        CHKERRQ(MatSeqAIJGetArray(C_CR,&aa));
        CHKERRQ(MatCreateSeqAIJWithArrays(PETSC_COMM_SELF,n_constraints,lda_rhs,ii,jj,aa,&tC_CR));
        CHKERRQ(MatRestoreRowIJ(C_CR,0,PETSC_FALSE,PETSC_FALSE,&r,(const PetscInt**)&ii,(const PetscInt**)&jj,&done));
        PetscCheckFalse(!done,PETSC_COMM_SELF,PETSC_ERR_PLIB,"RestoreRowIJ failed");
      } else {
        CHKERRQ(PetscObjectReference((PetscObject)C_CR));
        tC_CR = C_CR;
      }
      CHKERRQ(MatCreateTranspose(tC_CR,&Brhs));
      CHKERRQ(MatDestroy(&tC_CR));
    }
    CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,lda_rhs,n_constraints,NULL,&local_auxmat2_R));
    if (F) {
      if (need_benign_correction) {
        PCBDDCReuseSolvers reuse_solver = sub_schurs->reuse_solver;

        /* rhs is already zero on interior dofs, no need to change the rhs */
        CHKERRQ(PetscArrayzero(reuse_solver->benign_save_vals,pcbddc->benign_n));
      }
      CHKERRQ(MatMatSolve(F,Brhs,local_auxmat2_R));
      if (need_benign_correction) {
        PetscScalar        *marr;
        PCBDDCReuseSolvers reuse_solver = sub_schurs->reuse_solver;

        CHKERRQ(MatDenseGetArray(local_auxmat2_R,&marr));
        if (lda_rhs != n_R) {
          for (i=0;i<n_constraints;i++) {
            CHKERRQ(VecPlaceArray(dummy_vec,marr+i*lda_rhs));
            CHKERRQ(PCBDDCReuseSolversBenignAdapt(reuse_solver,dummy_vec,NULL,PETSC_TRUE,PETSC_TRUE));
            CHKERRQ(VecResetArray(dummy_vec));
          }
        } else {
          for (i=0;i<n_constraints;i++) {
            CHKERRQ(VecPlaceArray(pcbddc->vec1_R,marr+i*lda_rhs));
            CHKERRQ(PCBDDCReuseSolversBenignAdapt(reuse_solver,pcbddc->vec1_R,NULL,PETSC_TRUE,PETSC_TRUE));
            CHKERRQ(VecResetArray(pcbddc->vec1_R));
          }
        }
        CHKERRQ(MatDenseRestoreArray(local_auxmat2_R,&marr));
      }
    } else {
      PetscScalar *marr;

      CHKERRQ(MatDenseGetArray(local_auxmat2_R,&marr));
      for (i=0;i<n_constraints;i++) {
        CHKERRQ(VecPlaceArray(pcbddc->vec1_R,work+i*lda_rhs));
        CHKERRQ(VecPlaceArray(pcbddc->vec2_R,marr+i*lda_rhs));
        CHKERRQ(KSPSolve(pcbddc->ksp_R,pcbddc->vec1_R,pcbddc->vec2_R));
        CHKERRQ(KSPCheckSolve(pcbddc->ksp_R,pc,pcbddc->vec2_R));
        CHKERRQ(VecResetArray(pcbddc->vec1_R));
        CHKERRQ(VecResetArray(pcbddc->vec2_R));
      }
      CHKERRQ(MatDenseRestoreArray(local_auxmat2_R,&marr));
    }
    if (sparserhs) {
      CHKERRQ(MatScale(C_CR,-1.0));
    }
    CHKERRQ(MatDestroy(&Brhs));
    if (!pcbddc->switch_static) {
      CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,n_B,n_constraints,NULL,&pcbddc->local_auxmat2));
      for (i=0;i<n_constraints;i++) {
        Vec r, b;
        CHKERRQ(MatDenseGetColumnVecRead(local_auxmat2_R,i,&r));
        CHKERRQ(MatDenseGetColumnVec(pcbddc->local_auxmat2,i,&b));
        CHKERRQ(VecScatterBegin(pcbddc->R_to_B,r,b,INSERT_VALUES,SCATTER_FORWARD));
        CHKERRQ(VecScatterEnd(pcbddc->R_to_B,r,b,INSERT_VALUES,SCATTER_FORWARD));
        CHKERRQ(MatDenseRestoreColumnVec(pcbddc->local_auxmat2,i,&b));
        CHKERRQ(MatDenseRestoreColumnVecRead(local_auxmat2_R,i,&r));
      }
      CHKERRQ(MatMatMult(C_B,pcbddc->local_auxmat2,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&M3));
    } else {
      if (lda_rhs != n_R) {
        IS dummy;

        CHKERRQ(ISCreateStride(PETSC_COMM_SELF,n_R,0,1,&dummy));
        CHKERRQ(MatCreateSubMatrix(local_auxmat2_R,dummy,NULL,MAT_INITIAL_MATRIX,&pcbddc->local_auxmat2));
        CHKERRQ(ISDestroy(&dummy));
      } else {
        CHKERRQ(PetscObjectReference((PetscObject)local_auxmat2_R));
        pcbddc->local_auxmat2 = local_auxmat2_R;
      }
      CHKERRQ(MatMatMult(C_CR,pcbddc->local_auxmat2,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&M3));
    }
    CHKERRQ(ISDestroy(&is_aux));
    /* Assemble explicitly S_CC = ( C_{CR} A_{RR}^{-1} C^T_{CR})^{-1}  */
    CHKERRQ(MatScale(M3,m_one));
    if (isCHOL) {
      CHKERRQ(MatCholeskyFactor(M3,NULL,NULL));
    } else {
      CHKERRQ(MatLUFactor(M3,NULL,NULL,NULL));
    }
    CHKERRQ(MatSeqDenseInvertFactors_Private(M3));
    /* Assemble local_auxmat1 = S_CC*C_{CB} needed by BDDC application in KSP and in preproc */
    CHKERRQ(MatMatMult(M3,C_B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&pcbddc->local_auxmat1));
    CHKERRQ(MatDestroy(&C_B));
    CHKERRQ(MatCopy(M3,S_CC,SAME_NONZERO_PATTERN)); /* S_CC can have a different LDA, MatMatSolve doesn't support it */
    CHKERRQ(MatDestroy(&M3));
  }

  /* Get submatrices from subdomain matrix */
  if (n_vertices) {
#if defined(PETSC_HAVE_VIENNACL) || defined(PETSC_HAVE_CUDA)
    PetscBool oldpin;
#endif
    PetscBool isaij;
    IS        is_aux;

    if (sub_schurs && sub_schurs->reuse_solver) { /* is_R_local is not sorted, ISComplement doesn't like it */
      IS tis;

      CHKERRQ(ISDuplicate(pcbddc->is_R_local,&tis));
      CHKERRQ(ISSort(tis));
      CHKERRQ(ISComplement(tis,0,pcis->n,&is_aux));
      CHKERRQ(ISDestroy(&tis));
    } else {
      CHKERRQ(ISComplement(pcbddc->is_R_local,0,pcis->n,&is_aux));
    }
#if defined(PETSC_HAVE_VIENNACL) || defined(PETSC_HAVE_CUDA)
    oldpin = pcbddc->local_mat->boundtocpu;
#endif
    CHKERRQ(MatBindToCPU(pcbddc->local_mat,PETSC_TRUE));
    CHKERRQ(MatCreateSubMatrix(pcbddc->local_mat,pcbddc->is_R_local,is_aux,MAT_INITIAL_MATRIX,&A_RV));
    CHKERRQ(MatCreateSubMatrix(pcbddc->local_mat,is_aux,pcbddc->is_R_local,MAT_INITIAL_MATRIX,&A_VR));
    CHKERRQ(PetscObjectBaseTypeCompare((PetscObject)A_VR,MATSEQAIJ,&isaij));
    if (!isaij) { /* TODO REMOVE: MatMatMult(A_VR,A_RRmA_RV) below may raise an error */
      CHKERRQ(MatConvert(A_VR,MATSEQAIJ,MAT_INPLACE_MATRIX,&A_VR));
    }
    CHKERRQ(MatCreateSubMatrix(pcbddc->local_mat,is_aux,is_aux,MAT_INITIAL_MATRIX,&A_VV));
#if defined(PETSC_HAVE_VIENNACL) || defined(PETSC_HAVE_CUDA)
    CHKERRQ(MatBindToCPU(pcbddc->local_mat,oldpin));
#endif
    CHKERRQ(ISDestroy(&is_aux));
  }

  /* Matrix of coarse basis functions (local) */
  if (pcbddc->coarse_phi_B) {
    PetscInt on_B,on_primal,on_D=n_D;
    if (pcbddc->coarse_phi_D) {
      CHKERRQ(MatGetSize(pcbddc->coarse_phi_D,&on_D,NULL));
    }
    CHKERRQ(MatGetSize(pcbddc->coarse_phi_B,&on_B,&on_primal));
    if (on_B != n_B || on_primal != pcbddc->local_primal_size || on_D != n_D) {
      PetscScalar *marray;

      CHKERRQ(MatDenseGetArray(pcbddc->coarse_phi_B,&marray));
      CHKERRQ(PetscFree(marray));
      CHKERRQ(MatDestroy(&pcbddc->coarse_phi_B));
      CHKERRQ(MatDestroy(&pcbddc->coarse_psi_B));
      CHKERRQ(MatDestroy(&pcbddc->coarse_phi_D));
      CHKERRQ(MatDestroy(&pcbddc->coarse_psi_D));
    }
  }

  if (!pcbddc->coarse_phi_B) {
    PetscScalar *marr;

    /* memory size */
    n = n_B*pcbddc->local_primal_size;
    if (pcbddc->switch_static || pcbddc->dbg_flag) n += n_D*pcbddc->local_primal_size;
    if (!pcbddc->symmetric_primal) n *= 2;
    CHKERRQ(PetscCalloc1(n,&marr));
    CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,n_B,pcbddc->local_primal_size,marr,&pcbddc->coarse_phi_B));
    marr += n_B*pcbddc->local_primal_size;
    if (pcbddc->switch_static || pcbddc->dbg_flag) {
      CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,n_D,pcbddc->local_primal_size,marr,&pcbddc->coarse_phi_D));
      marr += n_D*pcbddc->local_primal_size;
    }
    if (!pcbddc->symmetric_primal) {
      CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,n_B,pcbddc->local_primal_size,marr,&pcbddc->coarse_psi_B));
      marr += n_B*pcbddc->local_primal_size;
      if (pcbddc->switch_static || pcbddc->dbg_flag) {
        CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,n_D,pcbddc->local_primal_size,marr,&pcbddc->coarse_psi_D));
      }
    } else {
      CHKERRQ(PetscObjectReference((PetscObject)pcbddc->coarse_phi_B));
      pcbddc->coarse_psi_B = pcbddc->coarse_phi_B;
      if (pcbddc->switch_static || pcbddc->dbg_flag) {
        CHKERRQ(PetscObjectReference((PetscObject)pcbddc->coarse_phi_D));
        pcbddc->coarse_psi_D = pcbddc->coarse_phi_D;
      }
    }
  }

  /* We are now ready to evaluate coarse basis functions and subdomain contribution to coarse problem */
  p0_lidx_I = NULL;
  if (pcbddc->benign_n && (pcbddc->switch_static || pcbddc->dbg_flag)) {
    const PetscInt *idxs;

    CHKERRQ(ISGetIndices(pcis->is_I_local,&idxs));
    CHKERRQ(PetscMalloc1(pcbddc->benign_n,&p0_lidx_I));
    for (i=0;i<pcbddc->benign_n;i++) {
      CHKERRQ(PetscFindInt(pcbddc->benign_p0_lidx[i],pcis->n-pcis->n_B,idxs,&p0_lidx_I[i]));
    }
    CHKERRQ(ISRestoreIndices(pcis->is_I_local,&idxs));
  }

  /* vertices */
  if (n_vertices) {
    PetscBool restoreavr = PETSC_FALSE;

    CHKERRQ(MatConvert(A_VV,MATDENSE,MAT_INPLACE_MATRIX,&A_VV));

    if (n_R) {
      Mat               A_RRmA_RV,A_RV_bcorr=NULL,S_VVt; /* S_VVt with LDA=N */
      PetscBLASInt      B_N,B_one = 1;
      const PetscScalar *x;
      PetscScalar       *y;

      CHKERRQ(MatScale(A_RV,m_one));
      if (need_benign_correction) {
        ISLocalToGlobalMapping RtoN;
        IS                     is_p0;
        PetscInt               *idxs_p0,n;

        CHKERRQ(PetscMalloc1(pcbddc->benign_n,&idxs_p0));
        CHKERRQ(ISLocalToGlobalMappingCreateIS(pcbddc->is_R_local,&RtoN));
        CHKERRQ(ISGlobalToLocalMappingApply(RtoN,IS_GTOLM_DROP,pcbddc->benign_n,pcbddc->benign_p0_lidx,&n,idxs_p0));
        PetscCheckFalse(n != pcbddc->benign_n,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error in R numbering for benign p0! %D != %D",n,pcbddc->benign_n);
        CHKERRQ(ISLocalToGlobalMappingDestroy(&RtoN));
        CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,n,idxs_p0,PETSC_OWN_POINTER,&is_p0));
        CHKERRQ(MatCreateSubMatrix(A_RV,is_p0,NULL,MAT_INITIAL_MATRIX,&A_RV_bcorr));
        CHKERRQ(ISDestroy(&is_p0));
      }

      CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,lda_rhs,n_vertices,work,&A_RRmA_RV));
      if (!sparserhs || need_benign_correction) {
        if (lda_rhs == n_R) {
          CHKERRQ(MatConvert(A_RV,MATDENSE,MAT_INPLACE_MATRIX,&A_RV));
        } else {
          PetscScalar    *av,*array;
          const PetscInt *xadj,*adjncy;
          PetscInt       n;
          PetscBool      flg_row;

          array = work+lda_rhs*n_vertices;
          CHKERRQ(PetscArrayzero(array,lda_rhs*n_vertices));
          CHKERRQ(MatConvert(A_RV,MATSEQAIJ,MAT_INPLACE_MATRIX,&A_RV));
          CHKERRQ(MatGetRowIJ(A_RV,0,PETSC_FALSE,PETSC_FALSE,&n,&xadj,&adjncy,&flg_row));
          CHKERRQ(MatSeqAIJGetArray(A_RV,&av));
          for (i=0;i<n;i++) {
            PetscInt j;
            for (j=xadj[i];j<xadj[i+1];j++) array[lda_rhs*adjncy[j]+i] = av[j];
          }
          CHKERRQ(MatRestoreRowIJ(A_RV,0,PETSC_FALSE,PETSC_FALSE,&n,&xadj,&adjncy,&flg_row));
          CHKERRQ(MatDestroy(&A_RV));
          CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,lda_rhs,n_vertices,array,&A_RV));
        }
        if (need_benign_correction) {
          PCBDDCReuseSolvers reuse_solver = sub_schurs->reuse_solver;
          PetscScalar        *marr;

          CHKERRQ(MatDenseGetArray(A_RV,&marr));
          /* need \Phi^T A_RV = (I+L)A_RV, L given by

                 | 0 0  0 | (V)
             L = | 0 0 -1 | (P-p0)
                 | 0 0 -1 | (p0)

          */
          for (i=0;i<reuse_solver->benign_n;i++) {
            const PetscScalar *vals;
            const PetscInt    *idxs,*idxs_zero;
            PetscInt          n,j,nz;

            CHKERRQ(ISGetLocalSize(reuse_solver->benign_zerodiag_subs[i],&nz));
            CHKERRQ(ISGetIndices(reuse_solver->benign_zerodiag_subs[i],&idxs_zero));
            CHKERRQ(MatGetRow(A_RV_bcorr,i,&n,&idxs,&vals));
            for (j=0;j<n;j++) {
              PetscScalar val = vals[j];
              PetscInt    k,col = idxs[j];
              for (k=0;k<nz;k++) marr[idxs_zero[k]+lda_rhs*col] -= val;
            }
            CHKERRQ(MatRestoreRow(A_RV_bcorr,i,&n,&idxs,&vals));
            CHKERRQ(ISRestoreIndices(reuse_solver->benign_zerodiag_subs[i],&idxs_zero));
          }
          CHKERRQ(MatDenseRestoreArray(A_RV,&marr));
        }
        CHKERRQ(PetscObjectReference((PetscObject)A_RV));
        Brhs = A_RV;
      } else {
        Mat tA_RVT,A_RVT;

        if (!pcbddc->symmetric_primal) {
          /* A_RV already scaled by -1 */
          CHKERRQ(MatTranspose(A_RV,MAT_INITIAL_MATRIX,&A_RVT));
        } else {
          restoreavr = PETSC_TRUE;
          CHKERRQ(MatScale(A_VR,-1.0));
          CHKERRQ(PetscObjectReference((PetscObject)A_VR));
          A_RVT = A_VR;
        }
        if (lda_rhs != n_R) {
          PetscScalar *aa;
          PetscInt    r,*ii,*jj;
          PetscBool   done;

          CHKERRQ(MatGetRowIJ(A_RVT,0,PETSC_FALSE,PETSC_FALSE,&r,(const PetscInt**)&ii,(const PetscInt**)&jj,&done));
          PetscCheckFalse(!done,PETSC_COMM_SELF,PETSC_ERR_PLIB,"GetRowIJ failed");
          CHKERRQ(MatSeqAIJGetArray(A_RVT,&aa));
          CHKERRQ(MatCreateSeqAIJWithArrays(PETSC_COMM_SELF,n_vertices,lda_rhs,ii,jj,aa,&tA_RVT));
          CHKERRQ(MatRestoreRowIJ(A_RVT,0,PETSC_FALSE,PETSC_FALSE,&r,(const PetscInt**)&ii,(const PetscInt**)&jj,&done));
          PetscCheckFalse(!done,PETSC_COMM_SELF,PETSC_ERR_PLIB,"RestoreRowIJ failed");
        } else {
          CHKERRQ(PetscObjectReference((PetscObject)A_RVT));
          tA_RVT = A_RVT;
        }
        CHKERRQ(MatCreateTranspose(tA_RVT,&Brhs));
        CHKERRQ(MatDestroy(&tA_RVT));
        CHKERRQ(MatDestroy(&A_RVT));
      }
      if (F) {
        /* need to correct the rhs */
        if (need_benign_correction) {
          PCBDDCReuseSolvers reuse_solver = sub_schurs->reuse_solver;
          PetscScalar        *marr;

          CHKERRQ(MatDenseGetArray(Brhs,&marr));
          if (lda_rhs != n_R) {
            for (i=0;i<n_vertices;i++) {
              CHKERRQ(VecPlaceArray(dummy_vec,marr+i*lda_rhs));
              CHKERRQ(PCBDDCReuseSolversBenignAdapt(reuse_solver,dummy_vec,NULL,PETSC_FALSE,PETSC_TRUE));
              CHKERRQ(VecResetArray(dummy_vec));
            }
          } else {
            for (i=0;i<n_vertices;i++) {
              CHKERRQ(VecPlaceArray(pcbddc->vec1_R,marr+i*lda_rhs));
              CHKERRQ(PCBDDCReuseSolversBenignAdapt(reuse_solver,pcbddc->vec1_R,NULL,PETSC_FALSE,PETSC_TRUE));
              CHKERRQ(VecResetArray(pcbddc->vec1_R));
            }
          }
          CHKERRQ(MatDenseRestoreArray(Brhs,&marr));
        }
        CHKERRQ(MatMatSolve(F,Brhs,A_RRmA_RV));
        if (restoreavr) {
          CHKERRQ(MatScale(A_VR,-1.0));
        }
        /* need to correct the solution */
        if (need_benign_correction) {
          PCBDDCReuseSolvers reuse_solver = sub_schurs->reuse_solver;
          PetscScalar        *marr;

          CHKERRQ(MatDenseGetArray(A_RRmA_RV,&marr));
          if (lda_rhs != n_R) {
            for (i=0;i<n_vertices;i++) {
              CHKERRQ(VecPlaceArray(dummy_vec,marr+i*lda_rhs));
              CHKERRQ(PCBDDCReuseSolversBenignAdapt(reuse_solver,dummy_vec,NULL,PETSC_TRUE,PETSC_TRUE));
              CHKERRQ(VecResetArray(dummy_vec));
            }
          } else {
            for (i=0;i<n_vertices;i++) {
              CHKERRQ(VecPlaceArray(pcbddc->vec1_R,marr+i*lda_rhs));
              CHKERRQ(PCBDDCReuseSolversBenignAdapt(reuse_solver,pcbddc->vec1_R,NULL,PETSC_TRUE,PETSC_TRUE));
              CHKERRQ(VecResetArray(pcbddc->vec1_R));
            }
          }
          CHKERRQ(MatDenseRestoreArray(A_RRmA_RV,&marr));
        }
      } else {
        CHKERRQ(MatDenseGetArray(Brhs,&y));
        for (i=0;i<n_vertices;i++) {
          CHKERRQ(VecPlaceArray(pcbddc->vec1_R,y+i*lda_rhs));
          CHKERRQ(VecPlaceArray(pcbddc->vec2_R,work+i*lda_rhs));
          CHKERRQ(KSPSolve(pcbddc->ksp_R,pcbddc->vec1_R,pcbddc->vec2_R));
          CHKERRQ(KSPCheckSolve(pcbddc->ksp_R,pc,pcbddc->vec2_R));
          CHKERRQ(VecResetArray(pcbddc->vec1_R));
          CHKERRQ(VecResetArray(pcbddc->vec2_R));
        }
        CHKERRQ(MatDenseRestoreArray(Brhs,&y));
      }
      CHKERRQ(MatDestroy(&A_RV));
      CHKERRQ(MatDestroy(&Brhs));
      /* S_VV and S_CV */
      if (n_constraints) {
        Mat B;

        CHKERRQ(PetscArrayzero(work+lda_rhs*n_vertices,n_B*n_vertices));
        for (i=0;i<n_vertices;i++) {
          CHKERRQ(VecPlaceArray(pcbddc->vec1_R,work+i*lda_rhs));
          CHKERRQ(VecPlaceArray(pcis->vec1_B,work+lda_rhs*n_vertices+i*n_B));
          CHKERRQ(VecScatterBegin(pcbddc->R_to_B,pcbddc->vec1_R,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD));
          CHKERRQ(VecScatterEnd(pcbddc->R_to_B,pcbddc->vec1_R,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD));
          CHKERRQ(VecResetArray(pcis->vec1_B));
          CHKERRQ(VecResetArray(pcbddc->vec1_R));
        }
        CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,n_B,n_vertices,work+lda_rhs*n_vertices,&B));
        /* Reuse dense S_C = pcbddc->local_auxmat1 * B */
        CHKERRQ(MatProductCreateWithMat(pcbddc->local_auxmat1,B,NULL,S_CV));
        CHKERRQ(MatProductSetType(S_CV,MATPRODUCT_AB));
        CHKERRQ(MatProductSetFromOptions(S_CV));
        CHKERRQ(MatProductSymbolic(S_CV));
        CHKERRQ(MatProductNumeric(S_CV));
        CHKERRQ(MatProductClear(S_CV));

        CHKERRQ(MatDestroy(&B));
        CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,lda_rhs,n_vertices,work+lda_rhs*n_vertices,&B));
        /* Reuse B = local_auxmat2_R * S_CV */
        CHKERRQ(MatProductCreateWithMat(local_auxmat2_R,S_CV,NULL,B));
        CHKERRQ(MatProductSetType(B,MATPRODUCT_AB));
        CHKERRQ(MatProductSetFromOptions(B));
        CHKERRQ(MatProductSymbolic(B));
        CHKERRQ(MatProductNumeric(B));

        CHKERRQ(MatScale(S_CV,m_one));
        CHKERRQ(PetscBLASIntCast(lda_rhs*n_vertices,&B_N));
        PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&B_N,&one,work+lda_rhs*n_vertices,&B_one,work,&B_one));
        CHKERRQ(MatDestroy(&B));
      }
      if (lda_rhs != n_R) {
        CHKERRQ(MatDestroy(&A_RRmA_RV));
        CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,n_R,n_vertices,work,&A_RRmA_RV));
        CHKERRQ(MatDenseSetLDA(A_RRmA_RV,lda_rhs));
      }
      CHKERRQ(MatMatMult(A_VR,A_RRmA_RV,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&S_VVt));
      /* need A_VR * \Phi * A_RRmA_RV = A_VR * (I+L)^T * A_RRmA_RV, L given as before */
      if (need_benign_correction) {
        PCBDDCReuseSolvers reuse_solver = sub_schurs->reuse_solver;
        PetscScalar        *marr,*sums;

        CHKERRQ(PetscMalloc1(n_vertices,&sums));
        CHKERRQ(MatDenseGetArray(S_VVt,&marr));
        for (i=0;i<reuse_solver->benign_n;i++) {
          const PetscScalar *vals;
          const PetscInt    *idxs,*idxs_zero;
          PetscInt          n,j,nz;

          CHKERRQ(ISGetLocalSize(reuse_solver->benign_zerodiag_subs[i],&nz));
          CHKERRQ(ISGetIndices(reuse_solver->benign_zerodiag_subs[i],&idxs_zero));
          for (j=0;j<n_vertices;j++) {
            PetscInt k;
            sums[j] = 0.;
            for (k=0;k<nz;k++) sums[j] += work[idxs_zero[k]+j*lda_rhs];
          }
          CHKERRQ(MatGetRow(A_RV_bcorr,i,&n,&idxs,&vals));
          for (j=0;j<n;j++) {
            PetscScalar val = vals[j];
            PetscInt k;
            for (k=0;k<n_vertices;k++) {
              marr[idxs[j]+k*n_vertices] += val*sums[k];
            }
          }
          CHKERRQ(MatRestoreRow(A_RV_bcorr,i,&n,&idxs,&vals));
          CHKERRQ(ISRestoreIndices(reuse_solver->benign_zerodiag_subs[i],&idxs_zero));
        }
        CHKERRQ(PetscFree(sums));
        CHKERRQ(MatDenseRestoreArray(S_VVt,&marr));
        CHKERRQ(MatDestroy(&A_RV_bcorr));
      }
      CHKERRQ(MatDestroy(&A_RRmA_RV));
      CHKERRQ(PetscBLASIntCast(n_vertices*n_vertices,&B_N));
      CHKERRQ(MatDenseGetArrayRead(A_VV,&x));
      CHKERRQ(MatDenseGetArray(S_VVt,&y));
      PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&B_N,&one,x,&B_one,y,&B_one));
      CHKERRQ(MatDenseRestoreArrayRead(A_VV,&x));
      CHKERRQ(MatDenseRestoreArray(S_VVt,&y));
      CHKERRQ(MatCopy(S_VVt,S_VV,SAME_NONZERO_PATTERN));
      CHKERRQ(MatDestroy(&S_VVt));
    } else {
      CHKERRQ(MatCopy(A_VV,S_VV,SAME_NONZERO_PATTERN));
    }
    CHKERRQ(MatDestroy(&A_VV));

    /* coarse basis functions */
    for (i=0;i<n_vertices;i++) {
      Vec         v;
      PetscScalar one = 1.0,zero = 0.0;

      CHKERRQ(VecPlaceArray(pcbddc->vec1_R,work+lda_rhs*i));
      CHKERRQ(MatDenseGetColumnVec(pcbddc->coarse_phi_B,i,&v));
      CHKERRQ(VecScatterBegin(pcbddc->R_to_B,pcbddc->vec1_R,v,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterEnd(pcbddc->R_to_B,pcbddc->vec1_R,v,INSERT_VALUES,SCATTER_FORWARD));
      if (PetscDefined(USE_DEBUG)) { /* The following VecSetValues() expects a sequential matrix */
        PetscMPIInt rank;
        CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)pcbddc->coarse_phi_B),&rank));
        PetscCheckFalse(rank > 1,PetscObjectComm((PetscObject)pcbddc->coarse_phi_B),PETSC_ERR_PLIB,"Expected a sequential dense matrix");
      }
      CHKERRQ(VecSetValues(v,1,&idx_V_B[i],&one,INSERT_VALUES));
      CHKERRQ(VecAssemblyBegin(v)); /* If v is on device, hope VecSetValues() eventually implemented by a host to device memcopy */
      CHKERRQ(VecAssemblyEnd(v));
      CHKERRQ(MatDenseRestoreColumnVec(pcbddc->coarse_phi_B,i,&v));

      if (pcbddc->switch_static || pcbddc->dbg_flag) {
        PetscInt j;

        CHKERRQ(MatDenseGetColumnVec(pcbddc->coarse_phi_D,i,&v));
        CHKERRQ(VecScatterBegin(pcbddc->R_to_D,pcbddc->vec1_R,v,INSERT_VALUES,SCATTER_FORWARD));
        CHKERRQ(VecScatterEnd(pcbddc->R_to_D,pcbddc->vec1_R,v,INSERT_VALUES,SCATTER_FORWARD));
        if (PetscDefined(USE_DEBUG)) { /* The following VecSetValues() expects a sequential matrix */
          PetscMPIInt rank;
          CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)pcbddc->coarse_phi_D),&rank));
          PetscCheckFalse(rank > 1,PetscObjectComm((PetscObject)pcbddc->coarse_phi_D),PETSC_ERR_PLIB,"Expected a sequential dense matrix");
        }
        for (j=0;j<pcbddc->benign_n;j++) CHKERRQ(VecSetValues(v,1,&p0_lidx_I[j],&zero,INSERT_VALUES));
        CHKERRQ(VecAssemblyBegin(v));
        CHKERRQ(VecAssemblyEnd(v));
        CHKERRQ(MatDenseRestoreColumnVec(pcbddc->coarse_phi_D,i,&v));
      }
      CHKERRQ(VecResetArray(pcbddc->vec1_R));
    }
    /* if n_R == 0 the object is not destroyed */
    CHKERRQ(MatDestroy(&A_RV));
  }
  CHKERRQ(VecDestroy(&dummy_vec));

  if (n_constraints) {
    Mat B;

    CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,lda_rhs,n_constraints,work,&B));
    CHKERRQ(MatScale(S_CC,m_one));
    CHKERRQ(MatProductCreateWithMat(local_auxmat2_R,S_CC,NULL,B));
    CHKERRQ(MatProductSetType(B,MATPRODUCT_AB));
    CHKERRQ(MatProductSetFromOptions(B));
    CHKERRQ(MatProductSymbolic(B));
    CHKERRQ(MatProductNumeric(B));

    CHKERRQ(MatScale(S_CC,m_one));
    if (n_vertices) {
      if (isCHOL || need_benign_correction) { /* if we can solve the interior problem with cholesky, we should also be fine with transposing here */
        CHKERRQ(MatTranspose(S_CV,MAT_REUSE_MATRIX,&S_VC));
      } else {
        Mat S_VCt;

        if (lda_rhs != n_R) {
          CHKERRQ(MatDestroy(&B));
          CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,n_R,n_constraints,work,&B));
          CHKERRQ(MatDenseSetLDA(B,lda_rhs));
        }
        CHKERRQ(MatMatMult(A_VR,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&S_VCt));
        CHKERRQ(MatCopy(S_VCt,S_VC,SAME_NONZERO_PATTERN));
        CHKERRQ(MatDestroy(&S_VCt));
      }
    }
    CHKERRQ(MatDestroy(&B));
    /* coarse basis functions */
    for (i=0;i<n_constraints;i++) {
      Vec v;

      CHKERRQ(VecPlaceArray(pcbddc->vec1_R,work+lda_rhs*i));
      CHKERRQ(MatDenseGetColumnVec(pcbddc->coarse_phi_B,i+n_vertices,&v));
      CHKERRQ(VecScatterBegin(pcbddc->R_to_B,pcbddc->vec1_R,v,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterEnd(pcbddc->R_to_B,pcbddc->vec1_R,v,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(MatDenseRestoreColumnVec(pcbddc->coarse_phi_B,i+n_vertices,&v));
      if (pcbddc->switch_static || pcbddc->dbg_flag) {
        PetscInt    j;
        PetscScalar zero = 0.0;
        CHKERRQ(MatDenseGetColumnVec(pcbddc->coarse_phi_D,i+n_vertices,&v));
        CHKERRQ(VecScatterBegin(pcbddc->R_to_D,pcbddc->vec1_R,v,INSERT_VALUES,SCATTER_FORWARD));
        CHKERRQ(VecScatterEnd(pcbddc->R_to_D,pcbddc->vec1_R,v,INSERT_VALUES,SCATTER_FORWARD));
        for (j=0;j<pcbddc->benign_n;j++) CHKERRQ(VecSetValues(v,1,&p0_lidx_I[j],&zero,INSERT_VALUES));
        CHKERRQ(VecAssemblyBegin(v));
        CHKERRQ(VecAssemblyEnd(v));
        CHKERRQ(MatDenseRestoreColumnVec(pcbddc->coarse_phi_D,i+n_vertices,&v));
      }
      CHKERRQ(VecResetArray(pcbddc->vec1_R));
    }
  }
  if (n_constraints) {
    CHKERRQ(MatDestroy(&local_auxmat2_R));
  }
  CHKERRQ(PetscFree(p0_lidx_I));

  /* coarse matrix entries relative to B_0 */
  if (pcbddc->benign_n) {
    Mat               B0_B,B0_BPHI;
    IS                is_dummy;
    const PetscScalar *data;
    PetscInt          j;

    CHKERRQ(ISCreateStride(PETSC_COMM_SELF,pcbddc->benign_n,0,1,&is_dummy));
    CHKERRQ(MatCreateSubMatrix(pcbddc->benign_B0,is_dummy,pcis->is_B_local,MAT_INITIAL_MATRIX,&B0_B));
    CHKERRQ(ISDestroy(&is_dummy));
    CHKERRQ(MatMatMult(B0_B,pcbddc->coarse_phi_B,MAT_INITIAL_MATRIX,1.0,&B0_BPHI));
    CHKERRQ(MatConvert(B0_BPHI,MATSEQDENSE,MAT_INPLACE_MATRIX,&B0_BPHI));
    CHKERRQ(MatDenseGetArrayRead(B0_BPHI,&data));
    for (j=0;j<pcbddc->benign_n;j++) {
      PetscInt primal_idx = pcbddc->local_primal_size - pcbddc->benign_n + j;
      for (i=0;i<pcbddc->local_primal_size;i++) {
        coarse_submat_vals[primal_idx*pcbddc->local_primal_size+i] = data[i*pcbddc->benign_n+j];
        coarse_submat_vals[i*pcbddc->local_primal_size+primal_idx] = data[i*pcbddc->benign_n+j];
      }
    }
    CHKERRQ(MatDenseRestoreArrayRead(B0_BPHI,&data));
    CHKERRQ(MatDestroy(&B0_B));
    CHKERRQ(MatDestroy(&B0_BPHI));
  }

  /* compute other basis functions for non-symmetric problems */
  if (!pcbddc->symmetric_primal) {
    Mat         B_V=NULL,B_C=NULL;
    PetscScalar *marray;

    if (n_constraints) {
      Mat S_CCT,C_CRT;

      CHKERRQ(MatTranspose(C_CR,MAT_INITIAL_MATRIX,&C_CRT));
      CHKERRQ(MatTranspose(S_CC,MAT_INITIAL_MATRIX,&S_CCT));
      CHKERRQ(MatMatMult(C_CRT,S_CCT,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&B_C));
      CHKERRQ(MatDestroy(&S_CCT));
      if (n_vertices) {
        Mat S_VCT;

        CHKERRQ(MatTranspose(S_VC,MAT_INITIAL_MATRIX,&S_VCT));
        CHKERRQ(MatMatMult(C_CRT,S_VCT,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&B_V));
        CHKERRQ(MatDestroy(&S_VCT));
      }
      CHKERRQ(MatDestroy(&C_CRT));
    } else {
      CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,n_R,n_vertices,NULL,&B_V));
    }
    if (n_vertices && n_R) {
      PetscScalar    *av,*marray;
      const PetscInt *xadj,*adjncy;
      PetscInt       n;
      PetscBool      flg_row;

      /* B_V = B_V - A_VR^T */
      CHKERRQ(MatConvert(A_VR,MATSEQAIJ,MAT_INPLACE_MATRIX,&A_VR));
      CHKERRQ(MatGetRowIJ(A_VR,0,PETSC_FALSE,PETSC_FALSE,&n,&xadj,&adjncy,&flg_row));
      CHKERRQ(MatSeqAIJGetArray(A_VR,&av));
      CHKERRQ(MatDenseGetArray(B_V,&marray));
      for (i=0;i<n;i++) {
        PetscInt j;
        for (j=xadj[i];j<xadj[i+1];j++) marray[i*n_R + adjncy[j]] -= av[j];
      }
      CHKERRQ(MatDenseRestoreArray(B_V,&marray));
      CHKERRQ(MatRestoreRowIJ(A_VR,0,PETSC_FALSE,PETSC_FALSE,&n,&xadj,&adjncy,&flg_row));
      CHKERRQ(MatDestroy(&A_VR));
    }

    /* currently there's no support for MatTransposeMatSolve(F,B,X) */
    if (n_vertices) {
      CHKERRQ(MatDenseGetArray(B_V,&marray));
      for (i=0;i<n_vertices;i++) {
        CHKERRQ(VecPlaceArray(pcbddc->vec1_R,marray+i*n_R));
        CHKERRQ(VecPlaceArray(pcbddc->vec2_R,work+i*n_R));
        CHKERRQ(KSPSolveTranspose(pcbddc->ksp_R,pcbddc->vec1_R,pcbddc->vec2_R));
        CHKERRQ(KSPCheckSolve(pcbddc->ksp_R,pc,pcbddc->vec2_R));
        CHKERRQ(VecResetArray(pcbddc->vec1_R));
        CHKERRQ(VecResetArray(pcbddc->vec2_R));
      }
      CHKERRQ(MatDenseRestoreArray(B_V,&marray));
    }
    if (B_C) {
      CHKERRQ(MatDenseGetArray(B_C,&marray));
      for (i=n_vertices;i<n_constraints+n_vertices;i++) {
        CHKERRQ(VecPlaceArray(pcbddc->vec1_R,marray+(i-n_vertices)*n_R));
        CHKERRQ(VecPlaceArray(pcbddc->vec2_R,work+i*n_R));
        CHKERRQ(KSPSolveTranspose(pcbddc->ksp_R,pcbddc->vec1_R,pcbddc->vec2_R));
        CHKERRQ(KSPCheckSolve(pcbddc->ksp_R,pc,pcbddc->vec2_R));
        CHKERRQ(VecResetArray(pcbddc->vec1_R));
        CHKERRQ(VecResetArray(pcbddc->vec2_R));
      }
      CHKERRQ(MatDenseRestoreArray(B_C,&marray));
    }
    /* coarse basis functions */
    for (i=0;i<pcbddc->local_primal_size;i++) {
      Vec  v;

      CHKERRQ(VecPlaceArray(pcbddc->vec1_R,work+i*n_R));
      CHKERRQ(MatDenseGetColumnVec(pcbddc->coarse_psi_B,i,&v));
      CHKERRQ(VecScatterBegin(pcbddc->R_to_B,pcbddc->vec1_R,v,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterEnd(pcbddc->R_to_B,pcbddc->vec1_R,v,INSERT_VALUES,SCATTER_FORWARD));
      if (i<n_vertices) {
        PetscScalar one = 1.0;
        CHKERRQ(VecSetValues(v,1,&idx_V_B[i],&one,INSERT_VALUES));
        CHKERRQ(VecAssemblyBegin(v));
        CHKERRQ(VecAssemblyEnd(v));
      }
      CHKERRQ(MatDenseRestoreColumnVec(pcbddc->coarse_psi_B,i,&v));

      if (pcbddc->switch_static || pcbddc->dbg_flag) {
        CHKERRQ(MatDenseGetColumnVec(pcbddc->coarse_psi_D,i,&v));
        CHKERRQ(VecScatterBegin(pcbddc->R_to_D,pcbddc->vec1_R,v,INSERT_VALUES,SCATTER_FORWARD));
        CHKERRQ(VecScatterEnd(pcbddc->R_to_D,pcbddc->vec1_R,v,INSERT_VALUES,SCATTER_FORWARD));
        CHKERRQ(MatDenseRestoreColumnVec(pcbddc->coarse_psi_D,i,&v));
      }
      CHKERRQ(VecResetArray(pcbddc->vec1_R));
    }
    CHKERRQ(MatDestroy(&B_V));
    CHKERRQ(MatDestroy(&B_C));
  }

  /* free memory */
  CHKERRQ(PetscFree(idx_V_B));
  CHKERRQ(MatDestroy(&S_VV));
  CHKERRQ(MatDestroy(&S_CV));
  CHKERRQ(MatDestroy(&S_VC));
  CHKERRQ(MatDestroy(&S_CC));
  CHKERRQ(PetscFree(work));
  if (n_vertices) {
    CHKERRQ(MatDestroy(&A_VR));
  }
  if (n_constraints) {
    CHKERRQ(MatDestroy(&C_CR));
  }
  CHKERRQ(PetscLogEventEnd(PC_BDDC_CorrectionSetUp[pcbddc->current_level],pc,0,0,0));

  /* Checking coarse_sub_mat and coarse basis functios */
  /* Symmetric case     : It should be \Phi^{(j)^T} A^{(j)} \Phi^{(j)}=coarse_sub_mat */
  /* Non-symmetric case : It should be \Psi^{(j)^T} A^{(j)} \Phi^{(j)}=coarse_sub_mat */
  if (pcbddc->dbg_flag) {
    Mat         coarse_sub_mat;
    Mat         AUXMAT,TM1,TM2,TM3,TM4;
    Mat         coarse_phi_D,coarse_phi_B;
    Mat         coarse_psi_D,coarse_psi_B;
    Mat         A_II,A_BB,A_IB,A_BI;
    Mat         C_B,CPHI;
    IS          is_dummy;
    Vec         mones;
    MatType     checkmattype=MATSEQAIJ;
    PetscReal   real_value;

    if (pcbddc->benign_n && !pcbddc->benign_change_explicit) {
      Mat A;
      CHKERRQ(PCBDDCBenignProject(pc,NULL,NULL,&A));
      CHKERRQ(MatCreateSubMatrix(A,pcis->is_I_local,pcis->is_I_local,MAT_INITIAL_MATRIX,&A_II));
      CHKERRQ(MatCreateSubMatrix(A,pcis->is_I_local,pcis->is_B_local,MAT_INITIAL_MATRIX,&A_IB));
      CHKERRQ(MatCreateSubMatrix(A,pcis->is_B_local,pcis->is_I_local,MAT_INITIAL_MATRIX,&A_BI));
      CHKERRQ(MatCreateSubMatrix(A,pcis->is_B_local,pcis->is_B_local,MAT_INITIAL_MATRIX,&A_BB));
      CHKERRQ(MatDestroy(&A));
    } else {
      CHKERRQ(MatConvert(pcis->A_II,checkmattype,MAT_INITIAL_MATRIX,&A_II));
      CHKERRQ(MatConvert(pcis->A_IB,checkmattype,MAT_INITIAL_MATRIX,&A_IB));
      CHKERRQ(MatConvert(pcis->A_BI,checkmattype,MAT_INITIAL_MATRIX,&A_BI));
      CHKERRQ(MatConvert(pcis->A_BB,checkmattype,MAT_INITIAL_MATRIX,&A_BB));
    }
    CHKERRQ(MatConvert(pcbddc->coarse_phi_D,checkmattype,MAT_INITIAL_MATRIX,&coarse_phi_D));
    CHKERRQ(MatConvert(pcbddc->coarse_phi_B,checkmattype,MAT_INITIAL_MATRIX,&coarse_phi_B));
    if (!pcbddc->symmetric_primal) {
      CHKERRQ(MatConvert(pcbddc->coarse_psi_D,checkmattype,MAT_INITIAL_MATRIX,&coarse_psi_D));
      CHKERRQ(MatConvert(pcbddc->coarse_psi_B,checkmattype,MAT_INITIAL_MATRIX,&coarse_psi_B));
    }
    CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,pcbddc->local_primal_size,pcbddc->local_primal_size,coarse_submat_vals,&coarse_sub_mat));

    CHKERRQ(PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"--------------------------------------------------\n"));
    CHKERRQ(PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"Check coarse sub mat computation (symmetric %d)\n",pcbddc->symmetric_primal));
    CHKERRQ(PetscViewerFlush(pcbddc->dbg_viewer));
    if (!pcbddc->symmetric_primal) {
      CHKERRQ(MatMatMult(A_II,coarse_phi_D,MAT_INITIAL_MATRIX,1.0,&AUXMAT));
      CHKERRQ(MatTransposeMatMult(coarse_psi_D,AUXMAT,MAT_INITIAL_MATRIX,1.0,&TM1));
      CHKERRQ(MatDestroy(&AUXMAT));
      CHKERRQ(MatMatMult(A_BB,coarse_phi_B,MAT_INITIAL_MATRIX,1.0,&AUXMAT));
      CHKERRQ(MatTransposeMatMult(coarse_psi_B,AUXMAT,MAT_INITIAL_MATRIX,1.0,&TM2));
      CHKERRQ(MatDestroy(&AUXMAT));
      CHKERRQ(MatMatMult(A_IB,coarse_phi_B,MAT_INITIAL_MATRIX,1.0,&AUXMAT));
      CHKERRQ(MatTransposeMatMult(coarse_psi_D,AUXMAT,MAT_INITIAL_MATRIX,1.0,&TM3));
      CHKERRQ(MatDestroy(&AUXMAT));
      CHKERRQ(MatMatMult(A_BI,coarse_phi_D,MAT_INITIAL_MATRIX,1.0,&AUXMAT));
      CHKERRQ(MatTransposeMatMult(coarse_psi_B,AUXMAT,MAT_INITIAL_MATRIX,1.0,&TM4));
      CHKERRQ(MatDestroy(&AUXMAT));
    } else {
      CHKERRQ(MatPtAP(A_II,coarse_phi_D,MAT_INITIAL_MATRIX,1.0,&TM1));
      CHKERRQ(MatPtAP(A_BB,coarse_phi_B,MAT_INITIAL_MATRIX,1.0,&TM2));
      CHKERRQ(MatMatMult(A_IB,coarse_phi_B,MAT_INITIAL_MATRIX,1.0,&AUXMAT));
      CHKERRQ(MatTransposeMatMult(coarse_phi_D,AUXMAT,MAT_INITIAL_MATRIX,1.0,&TM3));
      CHKERRQ(MatDestroy(&AUXMAT));
      CHKERRQ(MatMatMult(A_BI,coarse_phi_D,MAT_INITIAL_MATRIX,1.0,&AUXMAT));
      CHKERRQ(MatTransposeMatMult(coarse_phi_B,AUXMAT,MAT_INITIAL_MATRIX,1.0,&TM4));
      CHKERRQ(MatDestroy(&AUXMAT));
    }
    CHKERRQ(MatAXPY(TM1,one,TM2,DIFFERENT_NONZERO_PATTERN));
    CHKERRQ(MatAXPY(TM1,one,TM3,DIFFERENT_NONZERO_PATTERN));
    CHKERRQ(MatAXPY(TM1,one,TM4,DIFFERENT_NONZERO_PATTERN));
    CHKERRQ(MatConvert(TM1,MATSEQDENSE,MAT_INPLACE_MATRIX,&TM1));
    if (pcbddc->benign_n) {
      Mat               B0_B,B0_BPHI;
      const PetscScalar *data2;
      PetscScalar       *data;
      PetscInt          j;

      CHKERRQ(ISCreateStride(PETSC_COMM_SELF,pcbddc->benign_n,0,1,&is_dummy));
      CHKERRQ(MatCreateSubMatrix(pcbddc->benign_B0,is_dummy,pcis->is_B_local,MAT_INITIAL_MATRIX,&B0_B));
      CHKERRQ(MatMatMult(B0_B,coarse_phi_B,MAT_INITIAL_MATRIX,1.0,&B0_BPHI));
      CHKERRQ(MatConvert(B0_BPHI,MATSEQDENSE,MAT_INPLACE_MATRIX,&B0_BPHI));
      CHKERRQ(MatDenseGetArray(TM1,&data));
      CHKERRQ(MatDenseGetArrayRead(B0_BPHI,&data2));
      for (j=0;j<pcbddc->benign_n;j++) {
        PetscInt primal_idx = pcbddc->local_primal_size - pcbddc->benign_n + j;
        for (i=0;i<pcbddc->local_primal_size;i++) {
          data[primal_idx*pcbddc->local_primal_size+i] += data2[i*pcbddc->benign_n+j];
          data[i*pcbddc->local_primal_size+primal_idx] += data2[i*pcbddc->benign_n+j];
        }
      }
      CHKERRQ(MatDenseRestoreArray(TM1,&data));
      CHKERRQ(MatDenseRestoreArrayRead(B0_BPHI,&data2));
      CHKERRQ(MatDestroy(&B0_B));
      CHKERRQ(ISDestroy(&is_dummy));
      CHKERRQ(MatDestroy(&B0_BPHI));
    }
#if 0
  {
    PetscViewer viewer;
    char filename[256];
    sprintf(filename,"details_local_coarse_mat%d_level%d.m",PetscGlobalRank,pcbddc->current_level);
    CHKERRQ(PetscViewerASCIIOpen(PETSC_COMM_SELF,filename,&viewer));
    CHKERRQ(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB));
    CHKERRQ(PetscObjectSetName((PetscObject)coarse_sub_mat,"computed"));
    CHKERRQ(MatView(coarse_sub_mat,viewer));
    CHKERRQ(PetscObjectSetName((PetscObject)TM1,"projected"));
    CHKERRQ(MatView(TM1,viewer));
    if (pcbddc->coarse_phi_B) {
      CHKERRQ(PetscObjectSetName((PetscObject)pcbddc->coarse_phi_B,"phi_B"));
      CHKERRQ(MatView(pcbddc->coarse_phi_B,viewer));
    }
    if (pcbddc->coarse_phi_D) {
      CHKERRQ(PetscObjectSetName((PetscObject)pcbddc->coarse_phi_D,"phi_D"));
      CHKERRQ(MatView(pcbddc->coarse_phi_D,viewer));
    }
    if (pcbddc->coarse_psi_B) {
      CHKERRQ(PetscObjectSetName((PetscObject)pcbddc->coarse_psi_B,"psi_B"));
      CHKERRQ(MatView(pcbddc->coarse_psi_B,viewer));
    }
    if (pcbddc->coarse_psi_D) {
      CHKERRQ(PetscObjectSetName((PetscObject)pcbddc->coarse_psi_D,"psi_D"));
      CHKERRQ(MatView(pcbddc->coarse_psi_D,viewer));
    }
    CHKERRQ(PetscObjectSetName((PetscObject)pcbddc->local_mat,"A"));
    CHKERRQ(MatView(pcbddc->local_mat,viewer));
    CHKERRQ(PetscObjectSetName((PetscObject)pcbddc->ConstraintMatrix,"C"));
    CHKERRQ(MatView(pcbddc->ConstraintMatrix,viewer));
    CHKERRQ(PetscObjectSetName((PetscObject)pcis->is_I_local,"I"));
    CHKERRQ(ISView(pcis->is_I_local,viewer));
    CHKERRQ(PetscObjectSetName((PetscObject)pcis->is_B_local,"B"));
    CHKERRQ(ISView(pcis->is_B_local,viewer));
    CHKERRQ(PetscObjectSetName((PetscObject)pcbddc->is_R_local,"R"));
    CHKERRQ(ISView(pcbddc->is_R_local,viewer));
    CHKERRQ(PetscViewerDestroy(&viewer));
  }
#endif
    CHKERRQ(MatAXPY(TM1,m_one,coarse_sub_mat,DIFFERENT_NONZERO_PATTERN));
    CHKERRQ(MatNorm(TM1,NORM_FROBENIUS,&real_value));
    CHKERRQ(PetscViewerASCIIPushSynchronized(pcbddc->dbg_viewer));
    CHKERRQ(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d          matrix error % 1.14e\n",PetscGlobalRank,real_value));

    /* check constraints */
    CHKERRQ(ISCreateStride(PETSC_COMM_SELF,pcbddc->local_primal_size-pcbddc->benign_n,0,1,&is_dummy));
    CHKERRQ(MatCreateSubMatrix(pcbddc->ConstraintMatrix,is_dummy,pcis->is_B_local,MAT_INITIAL_MATRIX,&C_B));
    if (!pcbddc->benign_n) { /* TODO: add benign case */
      CHKERRQ(MatMatMult(C_B,coarse_phi_B,MAT_INITIAL_MATRIX,1.0,&CPHI));
    } else {
      PetscScalar *data;
      Mat         tmat;
      CHKERRQ(MatDenseGetArray(pcbddc->coarse_phi_B,&data));
      CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,pcis->n_B,pcbddc->local_primal_size-pcbddc->benign_n,data,&tmat));
      CHKERRQ(MatDenseRestoreArray(pcbddc->coarse_phi_B,&data));
      CHKERRQ(MatMatMult(C_B,tmat,MAT_INITIAL_MATRIX,1.0,&CPHI));
      CHKERRQ(MatDestroy(&tmat));
    }
    CHKERRQ(MatCreateVecs(CPHI,&mones,NULL));
    CHKERRQ(VecSet(mones,-1.0));
    CHKERRQ(MatDiagonalSet(CPHI,mones,ADD_VALUES));
    CHKERRQ(MatNorm(CPHI,NORM_FROBENIUS,&real_value));
    CHKERRQ(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d phi constraints error % 1.14e\n",PetscGlobalRank,real_value));
    if (!pcbddc->symmetric_primal) {
      CHKERRQ(MatMatMult(C_B,coarse_psi_B,MAT_REUSE_MATRIX,1.0,&CPHI));
      CHKERRQ(VecSet(mones,-1.0));
      CHKERRQ(MatDiagonalSet(CPHI,mones,ADD_VALUES));
      CHKERRQ(MatNorm(CPHI,NORM_FROBENIUS,&real_value));
      CHKERRQ(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d psi constraints error % 1.14e\n",PetscGlobalRank,real_value));
    }
    CHKERRQ(MatDestroy(&C_B));
    CHKERRQ(MatDestroy(&CPHI));
    CHKERRQ(ISDestroy(&is_dummy));
    CHKERRQ(VecDestroy(&mones));
    CHKERRQ(PetscViewerFlush(pcbddc->dbg_viewer));
    CHKERRQ(MatDestroy(&A_II));
    CHKERRQ(MatDestroy(&A_BB));
    CHKERRQ(MatDestroy(&A_IB));
    CHKERRQ(MatDestroy(&A_BI));
    CHKERRQ(MatDestroy(&TM1));
    CHKERRQ(MatDestroy(&TM2));
    CHKERRQ(MatDestroy(&TM3));
    CHKERRQ(MatDestroy(&TM4));
    CHKERRQ(MatDestroy(&coarse_phi_D));
    CHKERRQ(MatDestroy(&coarse_phi_B));
    if (!pcbddc->symmetric_primal) {
      CHKERRQ(MatDestroy(&coarse_psi_D));
      CHKERRQ(MatDestroy(&coarse_psi_B));
    }
    CHKERRQ(MatDestroy(&coarse_sub_mat));
  }
  /* FINAL CUDA support (we cannot currently mix viennacl and cuda vectors */
  {
    PetscBool gpu;

    CHKERRQ(PetscObjectTypeCompare((PetscObject)pcis->vec1_N,VECSEQCUDA,&gpu));
    if (gpu) {
      if (pcbddc->local_auxmat1) {
        CHKERRQ(MatConvert(pcbddc->local_auxmat1,MATSEQDENSECUDA,MAT_INPLACE_MATRIX,&pcbddc->local_auxmat1));
      }
      if (pcbddc->local_auxmat2) {
        CHKERRQ(MatConvert(pcbddc->local_auxmat2,MATSEQDENSECUDA,MAT_INPLACE_MATRIX,&pcbddc->local_auxmat2));
      }
      if (pcbddc->coarse_phi_B) {
        CHKERRQ(MatConvert(pcbddc->coarse_phi_B,MATSEQDENSECUDA,MAT_INPLACE_MATRIX,&pcbddc->coarse_phi_B));
      }
      if (pcbddc->coarse_phi_D) {
        CHKERRQ(MatConvert(pcbddc->coarse_phi_D,MATSEQDENSECUDA,MAT_INPLACE_MATRIX,&pcbddc->coarse_phi_D));
      }
      if (pcbddc->coarse_psi_B) {
        CHKERRQ(MatConvert(pcbddc->coarse_psi_B,MATSEQDENSECUDA,MAT_INPLACE_MATRIX,&pcbddc->coarse_psi_B));
      }
      if (pcbddc->coarse_psi_D) {
        CHKERRQ(MatConvert(pcbddc->coarse_psi_D,MATSEQDENSECUDA,MAT_INPLACE_MATRIX,&pcbddc->coarse_psi_D));
      }
    }
  }
  /* get back data */
  *coarse_submat_vals_n = coarse_submat_vals;
  PetscFunctionReturn(0);
}

PetscErrorCode MatCreateSubMatrixUnsorted(Mat A, IS isrow, IS iscol, Mat* B)
{
  Mat            *work_mat;
  IS             isrow_s,iscol_s;
  PetscBool      rsorted,csorted;
  PetscInt       rsize,*idxs_perm_r=NULL,csize,*idxs_perm_c=NULL;

  PetscFunctionBegin;
  CHKERRQ(ISSorted(isrow,&rsorted));
  CHKERRQ(ISSorted(iscol,&csorted));
  CHKERRQ(ISGetLocalSize(isrow,&rsize));
  CHKERRQ(ISGetLocalSize(iscol,&csize));

  if (!rsorted) {
    const PetscInt *idxs;
    PetscInt *idxs_sorted,i;

    CHKERRQ(PetscMalloc1(rsize,&idxs_perm_r));
    CHKERRQ(PetscMalloc1(rsize,&idxs_sorted));
    for (i=0;i<rsize;i++) {
      idxs_perm_r[i] = i;
    }
    CHKERRQ(ISGetIndices(isrow,&idxs));
    CHKERRQ(PetscSortIntWithPermutation(rsize,idxs,idxs_perm_r));
    for (i=0;i<rsize;i++) {
      idxs_sorted[i] = idxs[idxs_perm_r[i]];
    }
    CHKERRQ(ISRestoreIndices(isrow,&idxs));
    CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,rsize,idxs_sorted,PETSC_OWN_POINTER,&isrow_s));
  } else {
    CHKERRQ(PetscObjectReference((PetscObject)isrow));
    isrow_s = isrow;
  }

  if (!csorted) {
    if (isrow == iscol) {
      CHKERRQ(PetscObjectReference((PetscObject)isrow_s));
      iscol_s = isrow_s;
    } else {
      const PetscInt *idxs;
      PetscInt       *idxs_sorted,i;

      CHKERRQ(PetscMalloc1(csize,&idxs_perm_c));
      CHKERRQ(PetscMalloc1(csize,&idxs_sorted));
      for (i=0;i<csize;i++) {
        idxs_perm_c[i] = i;
      }
      CHKERRQ(ISGetIndices(iscol,&idxs));
      CHKERRQ(PetscSortIntWithPermutation(csize,idxs,idxs_perm_c));
      for (i=0;i<csize;i++) {
        idxs_sorted[i] = idxs[idxs_perm_c[i]];
      }
      CHKERRQ(ISRestoreIndices(iscol,&idxs));
      CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,csize,idxs_sorted,PETSC_OWN_POINTER,&iscol_s));
    }
  } else {
    CHKERRQ(PetscObjectReference((PetscObject)iscol));
    iscol_s = iscol;
  }

  CHKERRQ(MatCreateSubMatrices(A,1,&isrow_s,&iscol_s,MAT_INITIAL_MATRIX,&work_mat));

  if (!rsorted || !csorted) {
    Mat      new_mat;
    IS       is_perm_r,is_perm_c;

    if (!rsorted) {
      PetscInt *idxs_r,i;
      CHKERRQ(PetscMalloc1(rsize,&idxs_r));
      for (i=0;i<rsize;i++) {
        idxs_r[idxs_perm_r[i]] = i;
      }
      CHKERRQ(PetscFree(idxs_perm_r));
      CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,rsize,idxs_r,PETSC_OWN_POINTER,&is_perm_r));
    } else {
      CHKERRQ(ISCreateStride(PETSC_COMM_SELF,rsize,0,1,&is_perm_r));
    }
    CHKERRQ(ISSetPermutation(is_perm_r));

    if (!csorted) {
      if (isrow_s == iscol_s) {
        CHKERRQ(PetscObjectReference((PetscObject)is_perm_r));
        is_perm_c = is_perm_r;
      } else {
        PetscInt *idxs_c,i;
        PetscCheckFalse(!idxs_perm_c,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Permutation array not present");
        CHKERRQ(PetscMalloc1(csize,&idxs_c));
        for (i=0;i<csize;i++) {
          idxs_c[idxs_perm_c[i]] = i;
        }
        CHKERRQ(PetscFree(idxs_perm_c));
        CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,csize,idxs_c,PETSC_OWN_POINTER,&is_perm_c));
      }
    } else {
      CHKERRQ(ISCreateStride(PETSC_COMM_SELF,csize,0,1,&is_perm_c));
    }
    CHKERRQ(ISSetPermutation(is_perm_c));

    CHKERRQ(MatPermute(work_mat[0],is_perm_r,is_perm_c,&new_mat));
    CHKERRQ(MatDestroy(&work_mat[0]));
    work_mat[0] = new_mat;
    CHKERRQ(ISDestroy(&is_perm_r));
    CHKERRQ(ISDestroy(&is_perm_c));
  }

  CHKERRQ(PetscObjectReference((PetscObject)work_mat[0]));
  *B = work_mat[0];
  CHKERRQ(MatDestroyMatrices(1,&work_mat));
  CHKERRQ(ISDestroy(&isrow_s));
  CHKERRQ(ISDestroy(&iscol_s));
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCComputeLocalMatrix(PC pc, Mat ChangeOfBasisMatrix)
{
  Mat_IS*        matis = (Mat_IS*)pc->pmat->data;
  PC_BDDC*       pcbddc = (PC_BDDC*)pc->data;
  Mat            new_mat,lA;
  IS             is_local,is_global;
  PetscInt       local_size;
  PetscBool      isseqaij;

  PetscFunctionBegin;
  CHKERRQ(MatDestroy(&pcbddc->local_mat));
  CHKERRQ(MatGetSize(matis->A,&local_size,NULL));
  CHKERRQ(ISCreateStride(PetscObjectComm((PetscObject)matis->A),local_size,0,1,&is_local));
  CHKERRQ(ISLocalToGlobalMappingApplyIS(matis->rmapping,is_local,&is_global));
  CHKERRQ(ISDestroy(&is_local));
  CHKERRQ(MatCreateSubMatrixUnsorted(ChangeOfBasisMatrix,is_global,is_global,&new_mat));
  CHKERRQ(ISDestroy(&is_global));

  if (pcbddc->dbg_flag) {
    Vec       x,x_change;
    PetscReal error;

    CHKERRQ(MatCreateVecs(ChangeOfBasisMatrix,&x,&x_change));
    CHKERRQ(VecSetRandom(x,NULL));
    CHKERRQ(MatMult(ChangeOfBasisMatrix,x,x_change));
    CHKERRQ(VecScatterBegin(matis->cctx,x,matis->x,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(matis->cctx,x,matis->x,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(MatMult(new_mat,matis->x,matis->y));
    if (!pcbddc->change_interior) {
      const PetscScalar *x,*y,*v;
      PetscReal         lerror = 0.;
      PetscInt          i;

      CHKERRQ(VecGetArrayRead(matis->x,&x));
      CHKERRQ(VecGetArrayRead(matis->y,&y));
      CHKERRQ(VecGetArrayRead(matis->counter,&v));
      for (i=0;i<local_size;i++)
        if (PetscRealPart(v[i]) < 1.5 && PetscAbsScalar(x[i]-y[i]) > lerror)
          lerror = PetscAbsScalar(x[i]-y[i]);
      CHKERRQ(VecRestoreArrayRead(matis->x,&x));
      CHKERRQ(VecRestoreArrayRead(matis->y,&y));
      CHKERRQ(VecRestoreArrayRead(matis->counter,&v));
      CHKERRMPI(MPIU_Allreduce(&lerror,&error,1,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)pc)));
      if (error > PETSC_SMALL) {
        if (!pcbddc->user_ChangeOfBasisMatrix || pcbddc->current_level) {
          SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_PLIB,"Error global vs local change on I: %1.6e",error);
        } else {
          SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_USER,"Error global vs local change on I: %1.6e",error);
        }
      }
    }
    CHKERRQ(VecScatterBegin(matis->rctx,matis->y,x,INSERT_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecScatterEnd(matis->rctx,matis->y,x,INSERT_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecAXPY(x,-1.0,x_change));
    CHKERRQ(VecNorm(x,NORM_INFINITY,&error));
    if (error > PETSC_SMALL) {
      if (!pcbddc->user_ChangeOfBasisMatrix || pcbddc->current_level) {
        SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_PLIB,"Error global vs local change on N: %1.6e",error);
      } else {
        SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_USER,"Error global vs local change on N: %1.6e",error);
      }
    }
    CHKERRQ(VecDestroy(&x));
    CHKERRQ(VecDestroy(&x_change));
  }

  /* lA is present if we are setting up an inner BDDC for a saddle point FETI-DP */
  CHKERRQ(PetscObjectQuery((PetscObject)pc,"__KSPFETIDP_lA" ,(PetscObject*)&lA));

  /* TODO: HOW TO WORK WITH BAIJ and SBAIJ and SEQDENSE? */
  CHKERRQ(PetscObjectBaseTypeCompare((PetscObject)matis->A,MATSEQAIJ,&isseqaij));
  if (isseqaij) {
    CHKERRQ(MatDestroy(&pcbddc->local_mat));
    CHKERRQ(MatPtAP(matis->A,new_mat,MAT_INITIAL_MATRIX,2.0,&pcbddc->local_mat));
    if (lA) {
      Mat work;
      CHKERRQ(MatPtAP(lA,new_mat,MAT_INITIAL_MATRIX,2.0,&work));
      CHKERRQ(PetscObjectCompose((PetscObject)pc,"__KSPFETIDP_lA" ,(PetscObject)work));
      CHKERRQ(MatDestroy(&work));
    }
  } else {
    Mat work_mat;

    CHKERRQ(MatDestroy(&pcbddc->local_mat));
    CHKERRQ(MatConvert(matis->A,MATSEQAIJ,MAT_INITIAL_MATRIX,&work_mat));
    CHKERRQ(MatPtAP(work_mat,new_mat,MAT_INITIAL_MATRIX,2.0,&pcbddc->local_mat));
    CHKERRQ(MatDestroy(&work_mat));
    if (lA) {
      Mat work;
      CHKERRQ(MatConvert(lA,MATSEQAIJ,MAT_INITIAL_MATRIX,&work_mat));
      CHKERRQ(MatPtAP(work_mat,new_mat,MAT_INITIAL_MATRIX,2.0,&work));
      CHKERRQ(PetscObjectCompose((PetscObject)pc,"__KSPFETIDP_lA" ,(PetscObject)work));
      CHKERRQ(MatDestroy(&work));
    }
  }
  if (matis->A->symmetric_set) {
    CHKERRQ(MatSetOption(pcbddc->local_mat,MAT_SYMMETRIC,matis->A->symmetric));
#if !defined(PETSC_USE_COMPLEX)
    CHKERRQ(MatSetOption(pcbddc->local_mat,MAT_HERMITIAN,matis->A->symmetric));
#endif
  }
  CHKERRQ(MatDestroy(&new_mat));
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCSetUpLocalScatters(PC pc)
{
  PC_IS*          pcis = (PC_IS*)(pc->data);
  PC_BDDC*        pcbddc = (PC_BDDC*)pc->data;
  PCBDDCSubSchurs sub_schurs = pcbddc->sub_schurs;
  PetscInt        *idx_R_local=NULL;
  PetscInt        n_vertices,i,j,n_R,n_D,n_B;
  PetscInt        vbs,bs;
  PetscBT         bitmask=NULL;

  PetscFunctionBegin;
  /*
    No need to setup local scatters if
      - primal space is unchanged
        AND
      - we actually have locally some primal dofs (could not be true in multilevel or for isolated subdomains)
        AND
      - we are not in debugging mode (this is needed since there are Synchronized prints at the end of the subroutine
  */
  if (!pcbddc->new_primal_space_local && pcbddc->local_primal_size && !pcbddc->dbg_flag) {
    PetscFunctionReturn(0);
  }
  /* destroy old objects */
  CHKERRQ(ISDestroy(&pcbddc->is_R_local));
  CHKERRQ(VecScatterDestroy(&pcbddc->R_to_B));
  CHKERRQ(VecScatterDestroy(&pcbddc->R_to_D));
  /* Set Non-overlapping dimensions */
  n_B = pcis->n_B;
  n_D = pcis->n - n_B;
  n_vertices = pcbddc->n_vertices;

  /* Dohrmann's notation: dofs splitted in R (Remaining: all dofs but the vertices) and V (Vertices) */

  /* create auxiliary bitmask and allocate workspace */
  if (!sub_schurs || !sub_schurs->reuse_solver) {
    CHKERRQ(PetscMalloc1(pcis->n-n_vertices,&idx_R_local));
    CHKERRQ(PetscBTCreate(pcis->n,&bitmask));
    for (i=0;i<n_vertices;i++) {
      CHKERRQ(PetscBTSet(bitmask,pcbddc->local_primal_ref_node[i]));
    }

    for (i=0, n_R=0; i<pcis->n; i++) {
      if (!PetscBTLookup(bitmask,i)) {
        idx_R_local[n_R++] = i;
      }
    }
  } else { /* A different ordering (already computed) is present if we are reusing the Schur solver */
    PCBDDCReuseSolvers reuse_solver = sub_schurs->reuse_solver;

    CHKERRQ(ISGetIndices(reuse_solver->is_R,(const PetscInt**)&idx_R_local));
    CHKERRQ(ISGetLocalSize(reuse_solver->is_R,&n_R));
  }

  /* Block code */
  vbs = 1;
  CHKERRQ(MatGetBlockSize(pcbddc->local_mat,&bs));
  if (bs>1 && !(n_vertices%bs)) {
    PetscBool is_blocked = PETSC_TRUE;
    PetscInt  *vary;
    if (!sub_schurs || !sub_schurs->reuse_solver) {
      CHKERRQ(PetscMalloc1(pcis->n/bs,&vary));
      CHKERRQ(PetscArrayzero(vary,pcis->n/bs));
      /* Verify that the vertex indices correspond to each element in a block (code taken from sbaij2.c) */
      /* it is ok to check this way since local_primal_ref_node are always sorted by local numbering and idx_R_local is obtained as a complement */
      for (i=0; i<n_vertices; i++) vary[pcbddc->local_primal_ref_node[i]/bs]++;
      for (i=0; i<pcis->n/bs; i++) {
        if (vary[i]!=0 && vary[i]!=bs) {
          is_blocked = PETSC_FALSE;
          break;
        }
      }
      CHKERRQ(PetscFree(vary));
    } else {
      /* Verify directly the R set */
      for (i=0; i<n_R/bs; i++) {
        PetscInt j,node=idx_R_local[bs*i];
        for (j=1; j<bs; j++) {
          if (node != idx_R_local[bs*i+j]-j) {
            is_blocked = PETSC_FALSE;
            break;
          }
        }
      }
    }
    if (is_blocked) { /* build compressed IS for R nodes (complement of vertices) */
      vbs = bs;
      for (i=0;i<n_R/vbs;i++) {
        idx_R_local[i] = idx_R_local[vbs*i]/vbs;
      }
    }
  }
  CHKERRQ(ISCreateBlock(PETSC_COMM_SELF,vbs,n_R/vbs,idx_R_local,PETSC_COPY_VALUES,&pcbddc->is_R_local));
  if (sub_schurs && sub_schurs->reuse_solver) {
    PCBDDCReuseSolvers reuse_solver = sub_schurs->reuse_solver;

    CHKERRQ(ISRestoreIndices(reuse_solver->is_R,(const PetscInt**)&idx_R_local));
    CHKERRQ(ISDestroy(&reuse_solver->is_R));
    CHKERRQ(PetscObjectReference((PetscObject)pcbddc->is_R_local));
    reuse_solver->is_R = pcbddc->is_R_local;
  } else {
    CHKERRQ(PetscFree(idx_R_local));
  }

  /* print some info if requested */
  if (pcbddc->dbg_flag) {
    CHKERRQ(PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"--------------------------------------------------\n"));
    CHKERRQ(PetscViewerFlush(pcbddc->dbg_viewer));
    CHKERRQ(PetscViewerASCIIPushSynchronized(pcbddc->dbg_viewer));
    CHKERRQ(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d local dimensions\n",PetscGlobalRank));
    CHKERRQ(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"local_size = %D, dirichlet_size = %D, boundary_size = %D\n",pcis->n,n_D,n_B));
    CHKERRQ(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"r_size = %D, v_size = %D, constraints = %D, local_primal_size = %D\n",n_R,n_vertices,pcbddc->local_primal_size-n_vertices-pcbddc->benign_n,pcbddc->local_primal_size));
    CHKERRQ(PetscViewerFlush(pcbddc->dbg_viewer));
  }

  /* VecScatters pcbddc->R_to_B and (optionally) pcbddc->R_to_D */
  if (!sub_schurs || !sub_schurs->reuse_solver) {
    IS       is_aux1,is_aux2;
    PetscInt *aux_array1,*aux_array2,*is_indices,*idx_R_local;

    CHKERRQ(ISGetIndices(pcbddc->is_R_local,(const PetscInt**)&idx_R_local));
    CHKERRQ(PetscMalloc1(pcis->n_B-n_vertices,&aux_array1));
    CHKERRQ(PetscMalloc1(pcis->n_B-n_vertices,&aux_array2));
    CHKERRQ(ISGetIndices(pcis->is_I_local,(const PetscInt**)&is_indices));
    for (i=0; i<n_D; i++) {
      CHKERRQ(PetscBTSet(bitmask,is_indices[i]));
    }
    CHKERRQ(ISRestoreIndices(pcis->is_I_local,(const PetscInt**)&is_indices));
    for (i=0, j=0; i<n_R; i++) {
      if (!PetscBTLookup(bitmask,idx_R_local[i])) {
        aux_array1[j++] = i;
      }
    }
    CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,j,aux_array1,PETSC_OWN_POINTER,&is_aux1));
    CHKERRQ(ISGetIndices(pcis->is_B_local,(const PetscInt**)&is_indices));
    for (i=0, j=0; i<n_B; i++) {
      if (!PetscBTLookup(bitmask,is_indices[i])) {
        aux_array2[j++] = i;
      }
    }
    CHKERRQ(ISRestoreIndices(pcis->is_B_local,(const PetscInt**)&is_indices));
    CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,j,aux_array2,PETSC_OWN_POINTER,&is_aux2));
    CHKERRQ(VecScatterCreate(pcbddc->vec1_R,is_aux1,pcis->vec1_B,is_aux2,&pcbddc->R_to_B));
    CHKERRQ(ISDestroy(&is_aux1));
    CHKERRQ(ISDestroy(&is_aux2));

    if (pcbddc->switch_static || pcbddc->dbg_flag) {
      CHKERRQ(PetscMalloc1(n_D,&aux_array1));
      for (i=0, j=0; i<n_R; i++) {
        if (PetscBTLookup(bitmask,idx_R_local[i])) {
          aux_array1[j++] = i;
        }
      }
      CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,j,aux_array1,PETSC_OWN_POINTER,&is_aux1));
      CHKERRQ(VecScatterCreate(pcbddc->vec1_R,is_aux1,pcis->vec1_D,(IS)0,&pcbddc->R_to_D));
      CHKERRQ(ISDestroy(&is_aux1));
    }
    CHKERRQ(PetscBTDestroy(&bitmask));
    CHKERRQ(ISRestoreIndices(pcbddc->is_R_local,(const PetscInt**)&idx_R_local));
  } else {
    PCBDDCReuseSolvers reuse_solver = sub_schurs->reuse_solver;
    IS                 tis;
    PetscInt           schur_size;

    CHKERRQ(ISGetLocalSize(reuse_solver->is_B,&schur_size));
    CHKERRQ(ISCreateStride(PETSC_COMM_SELF,schur_size,n_D,1,&tis));
    CHKERRQ(VecScatterCreate(pcbddc->vec1_R,tis,pcis->vec1_B,reuse_solver->is_B,&pcbddc->R_to_B));
    CHKERRQ(ISDestroy(&tis));
    if (pcbddc->switch_static || pcbddc->dbg_flag) {
      CHKERRQ(ISCreateStride(PETSC_COMM_SELF,n_D,0,1,&tis));
      CHKERRQ(VecScatterCreate(pcbddc->vec1_R,tis,pcis->vec1_D,(IS)0,&pcbddc->R_to_D));
      CHKERRQ(ISDestroy(&tis));
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatNullSpacePropagateAny_Private(Mat A, IS is, Mat B)
{
  MatNullSpace   NullSpace;
  Mat            dmat;
  const Vec      *nullvecs;
  Vec            v,v2,*nullvecs2;
  VecScatter     sct = NULL;
  PetscContainer c;
  PetscScalar    *ddata;
  PetscInt       k,nnsp_size,bsiz,bsiz2,n,N,bs;
  PetscBool      nnsp_has_cnst;

  PetscFunctionBegin;
  if (!is && !B) { /* MATIS */
    Mat_IS* matis = (Mat_IS*)A->data;

    if (!B) {
      CHKERRQ(MatISGetLocalMat(A,&B));
    }
    sct  = matis->cctx;
    CHKERRQ(PetscObjectReference((PetscObject)sct));
  } else {
    CHKERRQ(MatGetNullSpace(B,&NullSpace));
    if (!NullSpace) {
      CHKERRQ(MatGetNearNullSpace(B,&NullSpace));
    }
    if (NullSpace) PetscFunctionReturn(0);
  }
  CHKERRQ(MatGetNullSpace(A,&NullSpace));
  if (!NullSpace) {
    CHKERRQ(MatGetNearNullSpace(A,&NullSpace));
  }
  if (!NullSpace) PetscFunctionReturn(0);

  CHKERRQ(MatCreateVecs(A,&v,NULL));
  CHKERRQ(MatCreateVecs(B,&v2,NULL));
  if (!sct) {
    CHKERRQ(VecScatterCreate(v,is,v2,NULL,&sct));
  }
  CHKERRQ(MatNullSpaceGetVecs(NullSpace,&nnsp_has_cnst,&nnsp_size,(const Vec**)&nullvecs));
  bsiz = bsiz2 = nnsp_size+!!nnsp_has_cnst;
  CHKERRQ(PetscMalloc1(bsiz,&nullvecs2));
  CHKERRQ(VecGetBlockSize(v2,&bs));
  CHKERRQ(VecGetSize(v2,&N));
  CHKERRQ(VecGetLocalSize(v2,&n));
  CHKERRQ(PetscMalloc1(n*bsiz,&ddata));
  for (k=0;k<nnsp_size;k++) {
    CHKERRQ(VecCreateMPIWithArray(PetscObjectComm((PetscObject)B),bs,n,N,ddata + n*k,&nullvecs2[k]));
    CHKERRQ(VecScatterBegin(sct,nullvecs[k],nullvecs2[k],INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(sct,nullvecs[k],nullvecs2[k],INSERT_VALUES,SCATTER_FORWARD));
  }
  if (nnsp_has_cnst) {
    CHKERRQ(VecCreateMPIWithArray(PetscObjectComm((PetscObject)B),bs,n,N,ddata + n*nnsp_size,&nullvecs2[nnsp_size]));
    CHKERRQ(VecSet(nullvecs2[nnsp_size],1.0));
  }
  CHKERRQ(PCBDDCOrthonormalizeVecs(&bsiz2,nullvecs2));
  CHKERRQ(MatNullSpaceCreate(PetscObjectComm((PetscObject)B),PETSC_FALSE,bsiz2,nullvecs2,&NullSpace));

  CHKERRQ(MatCreateDense(PetscObjectComm((PetscObject)B),n,PETSC_DECIDE,N,bsiz2,ddata,&dmat));
  CHKERRQ(PetscContainerCreate(PetscObjectComm((PetscObject)B),&c));
  CHKERRQ(PetscContainerSetPointer(c,ddata));
  CHKERRQ(PetscContainerSetUserDestroy(c,PetscContainerUserDestroyDefault));
  CHKERRQ(PetscObjectCompose((PetscObject)dmat,"_PBDDC_Null_dmat_arr",(PetscObject)c));
  CHKERRQ(PetscContainerDestroy(&c));
  CHKERRQ(PetscObjectCompose((PetscObject)NullSpace,"_PBDDC_Null_dmat",(PetscObject)dmat));
  CHKERRQ(MatDestroy(&dmat));

  for (k=0;k<bsiz;k++) {
    CHKERRQ(VecDestroy(&nullvecs2[k]));
  }
  CHKERRQ(PetscFree(nullvecs2));
  CHKERRQ(MatSetNearNullSpace(B,NullSpace));
  CHKERRQ(MatNullSpaceDestroy(&NullSpace));
  CHKERRQ(VecDestroy(&v));
  CHKERRQ(VecDestroy(&v2));
  CHKERRQ(VecScatterDestroy(&sct));
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCSetUpLocalSolvers(PC pc, PetscBool dirichlet, PetscBool neumann)
{
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;
  PC_IS          *pcis = (PC_IS*)pc->data;
  PC             pc_temp;
  Mat            A_RR;
  MatNullSpace   nnsp;
  MatReuse       reuse;
  PetscScalar    m_one = -1.0;
  PetscReal      value;
  PetscInt       n_D,n_R;
  PetscBool      issbaij,opts;
  void           (*f)(void) = NULL;
  char           dir_prefix[256],neu_prefix[256],str_level[16];
  size_t         len;

  PetscFunctionBegin;
  CHKERRQ(PetscLogEventBegin(PC_BDDC_LocalSolvers[pcbddc->current_level],pc,0,0,0));
  /* approximate solver, propagate NearNullSpace if needed */
  if (!pc->setupcalled && (pcbddc->NullSpace_corr[0] || pcbddc->NullSpace_corr[2])) {
    MatNullSpace gnnsp1,gnnsp2;
    PetscBool    lhas,ghas;

    CHKERRQ(MatGetNearNullSpace(pcbddc->local_mat,&nnsp));
    CHKERRQ(MatGetNearNullSpace(pc->pmat,&gnnsp1));
    CHKERRQ(MatGetNullSpace(pc->pmat,&gnnsp2));
    lhas = nnsp ? PETSC_TRUE : PETSC_FALSE;
    CHKERRMPI(MPIU_Allreduce(&lhas,&ghas,1,MPIU_BOOL,MPI_LOR,PetscObjectComm((PetscObject)pc)));
    if (!ghas && (gnnsp1 || gnnsp2)) {
      CHKERRQ(MatNullSpacePropagateAny_Private(pc->pmat,NULL,NULL));
    }
  }

  /* compute prefixes */
  CHKERRQ(PetscStrcpy(dir_prefix,""));
  CHKERRQ(PetscStrcpy(neu_prefix,""));
  if (!pcbddc->current_level) {
    CHKERRQ(PetscStrncpy(dir_prefix,((PetscObject)pc)->prefix,sizeof(dir_prefix)));
    CHKERRQ(PetscStrncpy(neu_prefix,((PetscObject)pc)->prefix,sizeof(neu_prefix)));
    CHKERRQ(PetscStrlcat(dir_prefix,"pc_bddc_dirichlet_",sizeof(dir_prefix)));
    CHKERRQ(PetscStrlcat(neu_prefix,"pc_bddc_neumann_",sizeof(neu_prefix)));
  } else {
    CHKERRQ(PetscSNPrintf(str_level,sizeof(str_level),"l%d_",(int)(pcbddc->current_level)));
    CHKERRQ(PetscStrlen(((PetscObject)pc)->prefix,&len));
    len -= 15; /* remove "pc_bddc_coarse_" */
    if (pcbddc->current_level>1) len -= 3; /* remove "lX_" with X level number */
    if (pcbddc->current_level>10) len -= 1; /* remove another char from level number */
    /* Nonstandard use of PetscStrncpy() to only copy a portion of the input string */
    CHKERRQ(PetscStrncpy(dir_prefix,((PetscObject)pc)->prefix,len+1));
    CHKERRQ(PetscStrncpy(neu_prefix,((PetscObject)pc)->prefix,len+1));
    CHKERRQ(PetscStrlcat(dir_prefix,"pc_bddc_dirichlet_",sizeof(dir_prefix)));
    CHKERRQ(PetscStrlcat(neu_prefix,"pc_bddc_neumann_",sizeof(neu_prefix)));
    CHKERRQ(PetscStrlcat(dir_prefix,str_level,sizeof(dir_prefix)));
    CHKERRQ(PetscStrlcat(neu_prefix,str_level,sizeof(neu_prefix)));
  }

  /* DIRICHLET PROBLEM */
  if (dirichlet) {
    PCBDDCSubSchurs sub_schurs = pcbddc->sub_schurs;
    if (pcbddc->benign_n && !pcbddc->benign_change_explicit) {
      PetscCheckFalse(!sub_schurs || !sub_schurs->reuse_solver,PETSC_COMM_SELF,PETSC_ERR_SUP,"Not yet implemented");
      if (pcbddc->dbg_flag) {
        Mat    A_IIn;

        CHKERRQ(PCBDDCBenignProject(pc,pcis->is_I_local,pcis->is_I_local,&A_IIn));
        CHKERRQ(MatDestroy(&pcis->A_II));
        pcis->A_II = A_IIn;
      }
    }
    if (pcbddc->local_mat->symmetric_set) {
      CHKERRQ(MatSetOption(pcis->A_II,MAT_SYMMETRIC,pcbddc->local_mat->symmetric));
    }
    /* Matrix for Dirichlet problem is pcis->A_II */
    n_D  = pcis->n - pcis->n_B;
    opts = PETSC_FALSE;
    if (!pcbddc->ksp_D) { /* create object if not yet build */
      opts = PETSC_TRUE;
      CHKERRQ(KSPCreate(PETSC_COMM_SELF,&pcbddc->ksp_D));
      CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)pcbddc->ksp_D,(PetscObject)pc,1));
      /* default */
      CHKERRQ(KSPSetType(pcbddc->ksp_D,KSPPREONLY));
      CHKERRQ(KSPSetOptionsPrefix(pcbddc->ksp_D,dir_prefix));
      CHKERRQ(PetscObjectTypeCompare((PetscObject)pcis->pA_II,MATSEQSBAIJ,&issbaij));
      CHKERRQ(KSPGetPC(pcbddc->ksp_D,&pc_temp));
      if (issbaij) {
        CHKERRQ(PCSetType(pc_temp,PCCHOLESKY));
      } else {
        CHKERRQ(PCSetType(pc_temp,PCLU));
      }
      CHKERRQ(KSPSetErrorIfNotConverged(pcbddc->ksp_D,pc->erroriffailure));
    }
    CHKERRQ(MatSetOptionsPrefix(pcis->pA_II,((PetscObject)pcbddc->ksp_D)->prefix));
    CHKERRQ(KSPSetOperators(pcbddc->ksp_D,pcis->A_II,pcis->pA_II));
    /* Allow user's customization */
    if (opts) {
      CHKERRQ(KSPSetFromOptions(pcbddc->ksp_D));
    }
    CHKERRQ(MatGetNearNullSpace(pcis->pA_II,&nnsp));
    if (pcbddc->NullSpace_corr[0] && !nnsp) { /* approximate solver, propagate NearNullSpace */
      CHKERRQ(MatNullSpacePropagateAny_Private(pcbddc->local_mat,pcis->is_I_local,pcis->pA_II));
    }
    CHKERRQ(MatGetNearNullSpace(pcis->pA_II,&nnsp));
    CHKERRQ(KSPGetPC(pcbddc->ksp_D,&pc_temp));
    CHKERRQ(PetscObjectQueryFunction((PetscObject)pc_temp,"PCSetCoordinates_C",&f));
    if (f && pcbddc->mat_graph->cloc && !nnsp) {
      PetscReal      *coords = pcbddc->mat_graph->coords,*scoords;
      const PetscInt *idxs;
      PetscInt       cdim = pcbddc->mat_graph->cdim,nl,i,d;

      CHKERRQ(ISGetLocalSize(pcis->is_I_local,&nl));
      CHKERRQ(ISGetIndices(pcis->is_I_local,&idxs));
      CHKERRQ(PetscMalloc1(nl*cdim,&scoords));
      for (i=0;i<nl;i++) {
        for (d=0;d<cdim;d++) {
          scoords[i*cdim+d] = coords[idxs[i]*cdim+d];
        }
      }
      CHKERRQ(ISRestoreIndices(pcis->is_I_local,&idxs));
      CHKERRQ(PCSetCoordinates(pc_temp,cdim,nl,scoords));
      CHKERRQ(PetscFree(scoords));
    }
    if (sub_schurs && sub_schurs->reuse_solver) {
      PCBDDCReuseSolvers reuse_solver = sub_schurs->reuse_solver;

      CHKERRQ(KSPSetPC(pcbddc->ksp_D,reuse_solver->interior_solver));
    }

    /* umfpack interface has a bug when matrix dimension is zero. TODO solve from umfpack interface */
    if (!n_D) {
      CHKERRQ(KSPGetPC(pcbddc->ksp_D,&pc_temp));
      CHKERRQ(PCSetType(pc_temp,PCNONE));
    }
    CHKERRQ(KSPSetUp(pcbddc->ksp_D));
    /* set ksp_D into pcis data */
    CHKERRQ(PetscObjectReference((PetscObject)pcbddc->ksp_D));
    CHKERRQ(KSPDestroy(&pcis->ksp_D));
    pcis->ksp_D = pcbddc->ksp_D;
  }

  /* NEUMANN PROBLEM */
  A_RR = NULL;
  if (neumann) {
    PCBDDCSubSchurs sub_schurs = pcbddc->sub_schurs;
    PetscInt        ibs,mbs;
    PetscBool       issbaij, reuse_neumann_solver;
    Mat_IS*         matis = (Mat_IS*)pc->pmat->data;

    reuse_neumann_solver = PETSC_FALSE;
    if (sub_schurs && sub_schurs->reuse_solver) {
      IS iP;

      reuse_neumann_solver = PETSC_TRUE;
      CHKERRQ(PetscObjectQuery((PetscObject)sub_schurs->A,"__KSPFETIDP_iP",(PetscObject*)&iP));
      if (iP) reuse_neumann_solver = PETSC_FALSE;
    }
    /* Matrix for Neumann problem is A_RR -> we need to create/reuse it at this point */
    CHKERRQ(ISGetSize(pcbddc->is_R_local,&n_R));
    if (pcbddc->ksp_R) { /* already created ksp */
      PetscInt nn_R;
      CHKERRQ(KSPGetOperators(pcbddc->ksp_R,NULL,&A_RR));
      CHKERRQ(PetscObjectReference((PetscObject)A_RR));
      CHKERRQ(MatGetSize(A_RR,&nn_R,NULL));
      if (nn_R != n_R) { /* old ksp is not reusable, so reset it */
        CHKERRQ(KSPReset(pcbddc->ksp_R));
        CHKERRQ(MatDestroy(&A_RR));
        reuse = MAT_INITIAL_MATRIX;
      } else { /* same sizes, but nonzero pattern depend on primal vertices so it can be changed */
        if (pcbddc->new_primal_space_local) { /* we are not sure the matrix will have the same nonzero pattern */
          CHKERRQ(MatDestroy(&A_RR));
          reuse = MAT_INITIAL_MATRIX;
        } else { /* safe to reuse the matrix */
          reuse = MAT_REUSE_MATRIX;
        }
      }
      /* last check */
      if (pc->flag == DIFFERENT_NONZERO_PATTERN) {
        CHKERRQ(MatDestroy(&A_RR));
        reuse = MAT_INITIAL_MATRIX;
      }
    } else { /* first time, so we need to create the matrix */
      reuse = MAT_INITIAL_MATRIX;
    }
    /* convert pcbddc->local_mat if needed later in PCBDDCSetUpCorrection
       TODO: Get Rid of these conversions */
    CHKERRQ(MatGetBlockSize(pcbddc->local_mat,&mbs));
    CHKERRQ(ISGetBlockSize(pcbddc->is_R_local,&ibs));
    CHKERRQ(PetscObjectTypeCompare((PetscObject)pcbddc->local_mat,MATSEQSBAIJ,&issbaij));
    if (ibs != mbs) { /* need to convert to SEQAIJ to extract any submatrix with is_R_local */
      if (matis->A == pcbddc->local_mat) {
        CHKERRQ(MatDestroy(&pcbddc->local_mat));
        CHKERRQ(MatConvert(matis->A,MATSEQAIJ,MAT_INITIAL_MATRIX,&pcbddc->local_mat));
      } else {
        CHKERRQ(MatConvert(pcbddc->local_mat,MATSEQAIJ,MAT_INPLACE_MATRIX,&pcbddc->local_mat));
      }
    } else if (issbaij) { /* need to convert to BAIJ to get offdiagonal blocks */
      if (matis->A == pcbddc->local_mat) {
        CHKERRQ(MatDestroy(&pcbddc->local_mat));
        CHKERRQ(MatConvert(matis->A,mbs > 1 ? MATSEQBAIJ : MATSEQAIJ,MAT_INITIAL_MATRIX,&pcbddc->local_mat));
      } else {
        CHKERRQ(MatConvert(pcbddc->local_mat,mbs > 1 ? MATSEQBAIJ : MATSEQAIJ,MAT_INPLACE_MATRIX,&pcbddc->local_mat));
      }
    }
    /* extract A_RR */
    if (reuse_neumann_solver) {
      PCBDDCReuseSolvers reuse_solver = sub_schurs->reuse_solver;

      if (pcbddc->dbg_flag) { /* we need A_RR to test the solver later */
        CHKERRQ(MatDestroy(&A_RR));
        if (reuse_solver->benign_n) { /* we are not using the explicit change of basis on the pressures */
          CHKERRQ(PCBDDCBenignProject(pc,pcbddc->is_R_local,pcbddc->is_R_local,&A_RR));
        } else {
          CHKERRQ(MatCreateSubMatrix(pcbddc->local_mat,pcbddc->is_R_local,pcbddc->is_R_local,MAT_INITIAL_MATRIX,&A_RR));
        }
      } else {
        CHKERRQ(MatDestroy(&A_RR));
        CHKERRQ(PCGetOperators(reuse_solver->correction_solver,&A_RR,NULL));
        CHKERRQ(PetscObjectReference((PetscObject)A_RR));
      }
    } else { /* we have to build the neumann solver, so we need to extract the relevant matrix */
      CHKERRQ(MatCreateSubMatrix(pcbddc->local_mat,pcbddc->is_R_local,pcbddc->is_R_local,reuse,&A_RR));
    }
    if (pcbddc->local_mat->symmetric_set) {
      CHKERRQ(MatSetOption(A_RR,MAT_SYMMETRIC,pcbddc->local_mat->symmetric));
    }
    opts = PETSC_FALSE;
    if (!pcbddc->ksp_R) { /* create object if not present */
      opts = PETSC_TRUE;
      CHKERRQ(KSPCreate(PETSC_COMM_SELF,&pcbddc->ksp_R));
      CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)pcbddc->ksp_R,(PetscObject)pc,1));
      /* default */
      CHKERRQ(KSPSetType(pcbddc->ksp_R,KSPPREONLY));
      CHKERRQ(KSPSetOptionsPrefix(pcbddc->ksp_R,neu_prefix));
      CHKERRQ(KSPGetPC(pcbddc->ksp_R,&pc_temp));
      CHKERRQ(PetscObjectTypeCompare((PetscObject)A_RR,MATSEQSBAIJ,&issbaij));
      if (issbaij) {
        CHKERRQ(PCSetType(pc_temp,PCCHOLESKY));
      } else {
        CHKERRQ(PCSetType(pc_temp,PCLU));
      }
      CHKERRQ(KSPSetErrorIfNotConverged(pcbddc->ksp_R,pc->erroriffailure));
    }
    CHKERRQ(KSPSetOperators(pcbddc->ksp_R,A_RR,A_RR));
    CHKERRQ(MatSetOptionsPrefix(A_RR,((PetscObject)pcbddc->ksp_R)->prefix));
    if (opts) { /* Allow user's customization once */
      CHKERRQ(KSPSetFromOptions(pcbddc->ksp_R));
    }
    CHKERRQ(MatGetNearNullSpace(A_RR,&nnsp));
    if (pcbddc->NullSpace_corr[2] && !nnsp) { /* approximate solver, propagate NearNullSpace */
      CHKERRQ(MatNullSpacePropagateAny_Private(pcbddc->local_mat,pcbddc->is_R_local,A_RR));
    }
    CHKERRQ(MatGetNearNullSpace(A_RR,&nnsp));
    CHKERRQ(KSPGetPC(pcbddc->ksp_R,&pc_temp));
    CHKERRQ(PetscObjectQueryFunction((PetscObject)pc_temp,"PCSetCoordinates_C",&f));
    if (f && pcbddc->mat_graph->cloc && !nnsp) {
      PetscReal      *coords = pcbddc->mat_graph->coords,*scoords;
      const PetscInt *idxs;
      PetscInt       cdim = pcbddc->mat_graph->cdim,nl,i,d;

      CHKERRQ(ISGetLocalSize(pcbddc->is_R_local,&nl));
      CHKERRQ(ISGetIndices(pcbddc->is_R_local,&idxs));
      CHKERRQ(PetscMalloc1(nl*cdim,&scoords));
      for (i=0;i<nl;i++) {
        for (d=0;d<cdim;d++) {
          scoords[i*cdim+d] = coords[idxs[i]*cdim+d];
        }
      }
      CHKERRQ(ISRestoreIndices(pcbddc->is_R_local,&idxs));
      CHKERRQ(PCSetCoordinates(pc_temp,cdim,nl,scoords));
      CHKERRQ(PetscFree(scoords));
    }

    /* umfpack interface has a bug when matrix dimension is zero. TODO solve from umfpack interface */
    if (!n_R) {
      CHKERRQ(KSPGetPC(pcbddc->ksp_R,&pc_temp));
      CHKERRQ(PCSetType(pc_temp,PCNONE));
    }
    /* Reuse solver if it is present */
    if (reuse_neumann_solver) {
      PCBDDCReuseSolvers reuse_solver = sub_schurs->reuse_solver;

      CHKERRQ(KSPSetPC(pcbddc->ksp_R,reuse_solver->correction_solver));
    }
    CHKERRQ(KSPSetUp(pcbddc->ksp_R));
  }

  if (pcbddc->dbg_flag) {
    CHKERRQ(PetscViewerFlush(pcbddc->dbg_viewer));
    CHKERRQ(PetscViewerASCIIPushSynchronized(pcbddc->dbg_viewer));
    CHKERRQ(PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"--------------------------------------------------\n"));
  }
  CHKERRQ(PetscLogEventEnd(PC_BDDC_LocalSolvers[pcbddc->current_level],pc,0,0,0));

  /* adapt Dirichlet and Neumann solvers if a nullspace correction has been requested */
  if (pcbddc->NullSpace_corr[0]) {
    CHKERRQ(PCBDDCSetUseExactDirichlet(pc,PETSC_FALSE));
  }
  if (dirichlet && pcbddc->NullSpace_corr[0] && !pcbddc->switch_static) {
    CHKERRQ(PCBDDCNullSpaceAssembleCorrection(pc,PETSC_TRUE,pcbddc->NullSpace_corr[1]));
  }
  if (neumann && pcbddc->NullSpace_corr[2]) {
    CHKERRQ(PCBDDCNullSpaceAssembleCorrection(pc,PETSC_FALSE,pcbddc->NullSpace_corr[3]));
  }
  /* check Dirichlet and Neumann solvers */
  if (pcbddc->dbg_flag) {
    if (dirichlet) { /* Dirichlet */
      CHKERRQ(VecSetRandom(pcis->vec1_D,NULL));
      CHKERRQ(MatMult(pcis->A_II,pcis->vec1_D,pcis->vec2_D));
      CHKERRQ(KSPSolve(pcbddc->ksp_D,pcis->vec2_D,pcis->vec2_D));
      CHKERRQ(KSPCheckSolve(pcbddc->ksp_D,pc,pcis->vec2_D));
      CHKERRQ(VecAXPY(pcis->vec1_D,m_one,pcis->vec2_D));
      CHKERRQ(VecNorm(pcis->vec1_D,NORM_INFINITY,&value));
      CHKERRQ(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d infinity error for Dirichlet solve (%s) = % 1.14e \n",PetscGlobalRank,((PetscObject)(pcbddc->ksp_D))->prefix,value));
      CHKERRQ(PetscViewerFlush(pcbddc->dbg_viewer));
    }
    if (neumann) { /* Neumann */
      CHKERRQ(VecSetRandom(pcbddc->vec1_R,NULL));
      CHKERRQ(MatMult(A_RR,pcbddc->vec1_R,pcbddc->vec2_R));
      CHKERRQ(KSPSolve(pcbddc->ksp_R,pcbddc->vec2_R,pcbddc->vec2_R));
      CHKERRQ(KSPCheckSolve(pcbddc->ksp_R,pc,pcbddc->vec2_R));
      CHKERRQ(VecAXPY(pcbddc->vec1_R,m_one,pcbddc->vec2_R));
      CHKERRQ(VecNorm(pcbddc->vec1_R,NORM_INFINITY,&value));
      CHKERRQ(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d infinity error for Neumann solve (%s) = % 1.14e\n",PetscGlobalRank,((PetscObject)(pcbddc->ksp_R))->prefix,value));
      CHKERRQ(PetscViewerFlush(pcbddc->dbg_viewer));
    }
  }
  /* free Neumann problem's matrix */
  CHKERRQ(MatDestroy(&A_RR));
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCBDDCSolveSubstructureCorrection(PC pc, Vec inout_B, Vec inout_D, PetscBool applytranspose)
{
  PC_BDDC*        pcbddc = (PC_BDDC*)(pc->data);
  PCBDDCSubSchurs sub_schurs = pcbddc->sub_schurs;
  PetscBool       reuse_solver = sub_schurs ? ( sub_schurs->reuse_solver ? PETSC_TRUE : PETSC_FALSE) : PETSC_FALSE;

  PetscFunctionBegin;
  if (!reuse_solver) {
    CHKERRQ(VecSet(pcbddc->vec1_R,0.));
  }
  if (!pcbddc->switch_static) {
    if (applytranspose && pcbddc->local_auxmat1) {
      CHKERRQ(MatMultTranspose(pcbddc->local_auxmat2,inout_B,pcbddc->vec1_C));
      CHKERRQ(MatMultTransposeAdd(pcbddc->local_auxmat1,pcbddc->vec1_C,inout_B,inout_B));
    }
    if (!reuse_solver) {
      CHKERRQ(VecScatterBegin(pcbddc->R_to_B,inout_B,pcbddc->vec1_R,INSERT_VALUES,SCATTER_REVERSE));
      CHKERRQ(VecScatterEnd(pcbddc->R_to_B,inout_B,pcbddc->vec1_R,INSERT_VALUES,SCATTER_REVERSE));
    } else {
      PCBDDCReuseSolvers reuse_solver = sub_schurs->reuse_solver;

      CHKERRQ(VecScatterBegin(reuse_solver->correction_scatter_B,inout_B,reuse_solver->rhs_B,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterEnd(reuse_solver->correction_scatter_B,inout_B,reuse_solver->rhs_B,INSERT_VALUES,SCATTER_FORWARD));
    }
  } else {
    CHKERRQ(VecScatterBegin(pcbddc->R_to_B,inout_B,pcbddc->vec1_R,INSERT_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecScatterEnd(pcbddc->R_to_B,inout_B,pcbddc->vec1_R,INSERT_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecScatterBegin(pcbddc->R_to_D,inout_D,pcbddc->vec1_R,INSERT_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecScatterEnd(pcbddc->R_to_D,inout_D,pcbddc->vec1_R,INSERT_VALUES,SCATTER_REVERSE));
    if (applytranspose && pcbddc->local_auxmat1) {
      CHKERRQ(MatMultTranspose(pcbddc->local_auxmat2,pcbddc->vec1_R,pcbddc->vec1_C));
      CHKERRQ(MatMultTransposeAdd(pcbddc->local_auxmat1,pcbddc->vec1_C,inout_B,inout_B));
      CHKERRQ(VecScatterBegin(pcbddc->R_to_B,inout_B,pcbddc->vec1_R,INSERT_VALUES,SCATTER_REVERSE));
      CHKERRQ(VecScatterEnd(pcbddc->R_to_B,inout_B,pcbddc->vec1_R,INSERT_VALUES,SCATTER_REVERSE));
    }
  }
  CHKERRQ(PetscLogEventBegin(PC_BDDC_Solves[pcbddc->current_level][1],pc,0,0,0));
  if (!reuse_solver || pcbddc->switch_static) {
    if (applytranspose) {
      CHKERRQ(KSPSolveTranspose(pcbddc->ksp_R,pcbddc->vec1_R,pcbddc->vec1_R));
    } else {
      CHKERRQ(KSPSolve(pcbddc->ksp_R,pcbddc->vec1_R,pcbddc->vec1_R));
    }
    CHKERRQ(KSPCheckSolve(pcbddc->ksp_R,pc,pcbddc->vec1_R));
  } else {
    PCBDDCReuseSolvers reuse_solver = sub_schurs->reuse_solver;

    if (applytranspose) {
      CHKERRQ(MatFactorSolveSchurComplementTranspose(reuse_solver->F,reuse_solver->rhs_B,reuse_solver->sol_B));
    } else {
      CHKERRQ(MatFactorSolveSchurComplement(reuse_solver->F,reuse_solver->rhs_B,reuse_solver->sol_B));
    }
  }
  CHKERRQ(PetscLogEventEnd(PC_BDDC_Solves[pcbddc->current_level][1],pc,0,0,0));
  CHKERRQ(VecSet(inout_B,0.));
  if (!pcbddc->switch_static) {
    if (!reuse_solver) {
      CHKERRQ(VecScatterBegin(pcbddc->R_to_B,pcbddc->vec1_R,inout_B,INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterEnd(pcbddc->R_to_B,pcbddc->vec1_R,inout_B,INSERT_VALUES,SCATTER_FORWARD));
    } else {
      PCBDDCReuseSolvers reuse_solver = sub_schurs->reuse_solver;

      CHKERRQ(VecScatterBegin(reuse_solver->correction_scatter_B,reuse_solver->sol_B,inout_B,INSERT_VALUES,SCATTER_REVERSE));
      CHKERRQ(VecScatterEnd(reuse_solver->correction_scatter_B,reuse_solver->sol_B,inout_B,INSERT_VALUES,SCATTER_REVERSE));
    }
    if (!applytranspose && pcbddc->local_auxmat1) {
      CHKERRQ(MatMult(pcbddc->local_auxmat1,inout_B,pcbddc->vec1_C));
      CHKERRQ(MatMultAdd(pcbddc->local_auxmat2,pcbddc->vec1_C,inout_B,inout_B));
    }
  } else {
    CHKERRQ(VecScatterBegin(pcbddc->R_to_B,pcbddc->vec1_R,inout_B,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(pcbddc->R_to_B,pcbddc->vec1_R,inout_B,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterBegin(pcbddc->R_to_D,pcbddc->vec1_R,inout_D,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(pcbddc->R_to_D,pcbddc->vec1_R,inout_D,INSERT_VALUES,SCATTER_FORWARD));
    if (!applytranspose && pcbddc->local_auxmat1) {
      CHKERRQ(MatMult(pcbddc->local_auxmat1,inout_B,pcbddc->vec1_C));
      CHKERRQ(MatMultAdd(pcbddc->local_auxmat2,pcbddc->vec1_C,pcbddc->vec1_R,pcbddc->vec1_R));
    }
    CHKERRQ(VecScatterBegin(pcbddc->R_to_B,pcbddc->vec1_R,inout_B,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(pcbddc->R_to_B,pcbddc->vec1_R,inout_B,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterBegin(pcbddc->R_to_D,pcbddc->vec1_R,inout_D,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(pcbddc->R_to_D,pcbddc->vec1_R,inout_D,INSERT_VALUES,SCATTER_FORWARD));
  }
  PetscFunctionReturn(0);
}

/* parameter apply transpose determines if the interface preconditioner should be applied transposed or not */
PetscErrorCode  PCBDDCApplyInterfacePreconditioner(PC pc, PetscBool applytranspose)
{
  PC_BDDC*        pcbddc = (PC_BDDC*)(pc->data);
  PC_IS*            pcis = (PC_IS*)  (pc->data);
  const PetscScalar zero = 0.0;

  PetscFunctionBegin;
  /* Application of PSI^T or PHI^T (depending on applytranspose, see comment above) */
  if (!pcbddc->benign_apply_coarse_only) {
    if (applytranspose) {
      CHKERRQ(MatMultTranspose(pcbddc->coarse_phi_B,pcis->vec1_B,pcbddc->vec1_P));
      if (pcbddc->switch_static) CHKERRQ(MatMultTransposeAdd(pcbddc->coarse_phi_D,pcis->vec1_D,pcbddc->vec1_P,pcbddc->vec1_P));
    } else {
      CHKERRQ(MatMultTranspose(pcbddc->coarse_psi_B,pcis->vec1_B,pcbddc->vec1_P));
      if (pcbddc->switch_static) CHKERRQ(MatMultTransposeAdd(pcbddc->coarse_psi_D,pcis->vec1_D,pcbddc->vec1_P,pcbddc->vec1_P));
    }
  } else {
    CHKERRQ(VecSet(pcbddc->vec1_P,zero));
  }

  /* add p0 to the last value of vec1_P holding the coarse dof relative to p0 */
  if (pcbddc->benign_n) {
    PetscScalar *array;
    PetscInt    j;

    CHKERRQ(VecGetArray(pcbddc->vec1_P,&array));
    for (j=0;j<pcbddc->benign_n;j++) array[pcbddc->local_primal_size-pcbddc->benign_n+j] += pcbddc->benign_p0[j];
    CHKERRQ(VecRestoreArray(pcbddc->vec1_P,&array));
  }

  /* start communications from local primal nodes to rhs of coarse solver */
  CHKERRQ(VecSet(pcbddc->coarse_vec,zero));
  CHKERRQ(PCBDDCScatterCoarseDataBegin(pc,ADD_VALUES,SCATTER_FORWARD));
  CHKERRQ(PCBDDCScatterCoarseDataEnd(pc,ADD_VALUES,SCATTER_FORWARD));

  /* Coarse solution -> rhs and sol updated inside PCBDDCScattarCoarseDataBegin/End */
  if (pcbddc->coarse_ksp) {
    Mat          coarse_mat;
    Vec          rhs,sol;
    MatNullSpace nullsp;
    PetscBool    isbddc = PETSC_FALSE;

    if (pcbddc->benign_have_null) {
      PC        coarse_pc;

      CHKERRQ(KSPGetPC(pcbddc->coarse_ksp,&coarse_pc));
      CHKERRQ(PetscObjectTypeCompare((PetscObject)coarse_pc,PCBDDC,&isbddc));
      /* we need to propagate to coarser levels the need for a possible benign correction */
      if (isbddc && pcbddc->benign_apply_coarse_only && !pcbddc->benign_skip_correction) {
        PC_BDDC* coarsepcbddc = (PC_BDDC*)(coarse_pc->data);
        coarsepcbddc->benign_skip_correction = PETSC_FALSE;
        coarsepcbddc->benign_apply_coarse_only = PETSC_TRUE;
      }
    }
    CHKERRQ(KSPGetRhs(pcbddc->coarse_ksp,&rhs));
    CHKERRQ(KSPGetSolution(pcbddc->coarse_ksp,&sol));
    CHKERRQ(KSPGetOperators(pcbddc->coarse_ksp,&coarse_mat,NULL));
    if (applytranspose) {
      PetscCheckFalse(pcbddc->benign_apply_coarse_only,PetscObjectComm((PetscObject)pcbddc->coarse_ksp),PETSC_ERR_SUP,"Not yet implemented");
      CHKERRQ(PetscLogEventBegin(PC_BDDC_Solves[pcbddc->current_level][2],pc,0,0,0));
      CHKERRQ(KSPSolveTranspose(pcbddc->coarse_ksp,rhs,sol));
      CHKERRQ(PetscLogEventEnd(PC_BDDC_Solves[pcbddc->current_level][2],pc,0,0,0));
      CHKERRQ(KSPCheckSolve(pcbddc->coarse_ksp,pc,sol));
      CHKERRQ(MatGetTransposeNullSpace(coarse_mat,&nullsp));
      if (nullsp) {
        CHKERRQ(MatNullSpaceRemove(nullsp,sol));
      }
    } else {
      CHKERRQ(MatGetNullSpace(coarse_mat,&nullsp));
      if (pcbddc->benign_apply_coarse_only && isbddc) { /* need just to apply the coarse preconditioner during presolve */
        PC        coarse_pc;

        if (nullsp) {
          CHKERRQ(MatNullSpaceRemove(nullsp,rhs));
        }
        CHKERRQ(KSPGetPC(pcbddc->coarse_ksp,&coarse_pc));
        CHKERRQ(PCPreSolve(coarse_pc,pcbddc->coarse_ksp));
        CHKERRQ(PCBDDCBenignRemoveInterior(coarse_pc,rhs,sol));
        CHKERRQ(PCPostSolve(coarse_pc,pcbddc->coarse_ksp));
      } else {
        CHKERRQ(PetscLogEventBegin(PC_BDDC_Solves[pcbddc->current_level][2],pc,0,0,0));
        CHKERRQ(KSPSolve(pcbddc->coarse_ksp,rhs,sol));
        CHKERRQ(PetscLogEventEnd(PC_BDDC_Solves[pcbddc->current_level][2],pc,0,0,0));
        CHKERRQ(KSPCheckSolve(pcbddc->coarse_ksp,pc,sol));
        if (nullsp) {
          CHKERRQ(MatNullSpaceRemove(nullsp,sol));
        }
      }
    }
    /* we don't need the benign correction at coarser levels anymore */
    if (pcbddc->benign_have_null && isbddc) {
      PC        coarse_pc;
      PC_BDDC*  coarsepcbddc;

      CHKERRQ(KSPGetPC(pcbddc->coarse_ksp,&coarse_pc));
      coarsepcbddc = (PC_BDDC*)(coarse_pc->data);
      coarsepcbddc->benign_skip_correction = PETSC_TRUE;
      coarsepcbddc->benign_apply_coarse_only = PETSC_FALSE;
    }
  }

  /* Local solution on R nodes */
  if (pcis->n && !pcbddc->benign_apply_coarse_only) {
    CHKERRQ(PCBDDCSolveSubstructureCorrection(pc,pcis->vec1_B,pcis->vec1_D,applytranspose));
  }
  /* communications from coarse sol to local primal nodes */
  CHKERRQ(PCBDDCScatterCoarseDataBegin(pc,INSERT_VALUES,SCATTER_REVERSE));
  CHKERRQ(PCBDDCScatterCoarseDataEnd(pc,INSERT_VALUES,SCATTER_REVERSE));

  /* Sum contributions from the two levels */
  if (!pcbddc->benign_apply_coarse_only) {
    if (applytranspose) {
      CHKERRQ(MatMultAdd(pcbddc->coarse_psi_B,pcbddc->vec1_P,pcis->vec1_B,pcis->vec1_B));
      if (pcbddc->switch_static) CHKERRQ(MatMultAdd(pcbddc->coarse_psi_D,pcbddc->vec1_P,pcis->vec1_D,pcis->vec1_D));
    } else {
      CHKERRQ(MatMultAdd(pcbddc->coarse_phi_B,pcbddc->vec1_P,pcis->vec1_B,pcis->vec1_B));
      if (pcbddc->switch_static) CHKERRQ(MatMultAdd(pcbddc->coarse_phi_D,pcbddc->vec1_P,pcis->vec1_D,pcis->vec1_D));
    }
    /* store p0 */
    if (pcbddc->benign_n) {
      PetscScalar *array;
      PetscInt    j;

      CHKERRQ(VecGetArray(pcbddc->vec1_P,&array));
      for (j=0;j<pcbddc->benign_n;j++) pcbddc->benign_p0[j] = array[pcbddc->local_primal_size-pcbddc->benign_n+j];
      CHKERRQ(VecRestoreArray(pcbddc->vec1_P,&array));
    }
  } else { /* expand the coarse solution */
    if (applytranspose) {
      CHKERRQ(MatMult(pcbddc->coarse_psi_B,pcbddc->vec1_P,pcis->vec1_B));
    } else {
      CHKERRQ(MatMult(pcbddc->coarse_phi_B,pcbddc->vec1_P,pcis->vec1_B));
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCScatterCoarseDataBegin(PC pc,InsertMode imode, ScatterMode smode)
{
  PC_BDDC*          pcbddc = (PC_BDDC*)(pc->data);
  Vec               from,to;
  const PetscScalar *array;

  PetscFunctionBegin;
  if (smode == SCATTER_REVERSE) { /* from global to local -> get data from coarse solution */
    from = pcbddc->coarse_vec;
    to = pcbddc->vec1_P;
    if (pcbddc->coarse_ksp) { /* get array from coarse processes */
      Vec tvec;

      CHKERRQ(KSPGetRhs(pcbddc->coarse_ksp,&tvec));
      CHKERRQ(VecResetArray(tvec));
      CHKERRQ(KSPGetSolution(pcbddc->coarse_ksp,&tvec));
      CHKERRQ(VecGetArrayRead(tvec,&array));
      CHKERRQ(VecPlaceArray(from,array));
      CHKERRQ(VecRestoreArrayRead(tvec,&array));
    }
  } else { /* from local to global -> put data in coarse right hand side */
    from = pcbddc->vec1_P;
    to = pcbddc->coarse_vec;
  }
  CHKERRQ(VecScatterBegin(pcbddc->coarse_loc_to_glob,from,to,imode,smode));
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCScatterCoarseDataEnd(PC pc, InsertMode imode, ScatterMode smode)
{
  PC_BDDC*          pcbddc = (PC_BDDC*)(pc->data);
  Vec               from,to;
  const PetscScalar *array;

  PetscFunctionBegin;
  if (smode == SCATTER_REVERSE) { /* from global to local -> get data from coarse solution */
    from = pcbddc->coarse_vec;
    to = pcbddc->vec1_P;
  } else { /* from local to global -> put data in coarse right hand side */
    from = pcbddc->vec1_P;
    to = pcbddc->coarse_vec;
  }
  CHKERRQ(VecScatterEnd(pcbddc->coarse_loc_to_glob,from,to,imode,smode));
  if (smode == SCATTER_FORWARD) {
    if (pcbddc->coarse_ksp) { /* get array from coarse processes */
      Vec tvec;

      CHKERRQ(KSPGetRhs(pcbddc->coarse_ksp,&tvec));
      CHKERRQ(VecGetArrayRead(to,&array));
      CHKERRQ(VecPlaceArray(tvec,array));
      CHKERRQ(VecRestoreArrayRead(to,&array));
    }
  } else {
    if (pcbddc->coarse_ksp) { /* restore array of pcbddc->coarse_vec */
     CHKERRQ(VecResetArray(from));
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCConstraintsSetUp(PC pc)
{
  PetscErrorCode    ierr;
  PC_IS*            pcis = (PC_IS*)(pc->data);
  PC_BDDC*          pcbddc = (PC_BDDC*)pc->data;
  Mat_IS*           matis = (Mat_IS*)pc->pmat->data;
  /* one and zero */
  PetscScalar       one=1.0,zero=0.0;
  /* space to store constraints and their local indices */
  PetscScalar       *constraints_data;
  PetscInt          *constraints_idxs,*constraints_idxs_B;
  PetscInt          *constraints_idxs_ptr,*constraints_data_ptr;
  PetscInt          *constraints_n;
  /* iterators */
  PetscInt          i,j,k,total_counts,total_counts_cc,cum;
  /* BLAS integers */
  PetscBLASInt      lwork,lierr;
  PetscBLASInt      Blas_N,Blas_M,Blas_K,Blas_one=1;
  PetscBLASInt      Blas_LDA,Blas_LDB,Blas_LDC;
  /* reuse */
  PetscInt          olocal_primal_size,olocal_primal_size_cc;
  PetscInt          *olocal_primal_ref_node,*olocal_primal_ref_mult;
  /* change of basis */
  PetscBool         qr_needed;
  PetscBT           change_basis,qr_needed_idx;
  /* auxiliary stuff */
  PetscInt          *nnz,*is_indices;
  PetscInt          ncc;
  /* some quantities */
  PetscInt          n_vertices,total_primal_vertices,valid_constraints;
  PetscInt          size_of_constraint,max_size_of_constraint=0,max_constraints,temp_constraints;
  PetscReal         tol; /* tolerance for retaining eigenmodes */

  PetscFunctionBegin;
  tol  = PetscSqrtReal(PETSC_SMALL);
  /* Destroy Mat objects computed previously */
  CHKERRQ(MatDestroy(&pcbddc->ChangeOfBasisMatrix));
  CHKERRQ(MatDestroy(&pcbddc->ConstraintMatrix));
  CHKERRQ(MatDestroy(&pcbddc->switch_static_change));
  /* save info on constraints from previous setup (if any) */
  olocal_primal_size = pcbddc->local_primal_size;
  olocal_primal_size_cc = pcbddc->local_primal_size_cc;
  CHKERRQ(PetscMalloc2(olocal_primal_size_cc,&olocal_primal_ref_node,olocal_primal_size_cc,&olocal_primal_ref_mult));
  CHKERRQ(PetscArraycpy(olocal_primal_ref_node,pcbddc->local_primal_ref_node,olocal_primal_size_cc));
  CHKERRQ(PetscArraycpy(olocal_primal_ref_mult,pcbddc->local_primal_ref_mult,olocal_primal_size_cc));
  CHKERRQ(PetscFree2(pcbddc->local_primal_ref_node,pcbddc->local_primal_ref_mult));
  CHKERRQ(PetscFree(pcbddc->primal_indices_local_idxs));

  if (!pcbddc->adaptive_selection) {
    IS           ISForVertices,*ISForFaces,*ISForEdges;
    MatNullSpace nearnullsp;
    const Vec    *nearnullvecs;
    Vec          *localnearnullsp;
    PetscScalar  *array;
    PetscInt     n_ISForFaces,n_ISForEdges,nnsp_size;
    PetscBool    nnsp_has_cnst;
    /* LAPACK working arrays for SVD or POD */
    PetscBool    skip_lapack,boolforchange;
    PetscScalar  *work;
    PetscReal    *singular_vals;
#if defined(PETSC_USE_COMPLEX)
    PetscReal    *rwork;
#endif
    PetscScalar  *temp_basis = NULL,*correlation_mat = NULL;
    PetscBLASInt dummy_int=1;
    PetscScalar  dummy_scalar=1.;
    PetscBool    use_pod = PETSC_FALSE;

    /* MKL SVD with same input gives different results on different processes! */
#if defined(PETSC_MISSING_LAPACK_GESVD) || defined(PETSC_HAVE_MKL_LIBS)
    use_pod = PETSC_TRUE;
#endif
    /* Get index sets for faces, edges and vertices from graph */
    CHKERRQ(PCBDDCGraphGetCandidatesIS(pcbddc->mat_graph,&n_ISForFaces,&ISForFaces,&n_ISForEdges,&ISForEdges,&ISForVertices));
    /* print some info */
    if (pcbddc->dbg_flag && (!pcbddc->sub_schurs || pcbddc->sub_schurs_rebuild)) {
      PetscInt nv;

      CHKERRQ(PCBDDCGraphASCIIView(pcbddc->mat_graph,pcbddc->dbg_flag,pcbddc->dbg_viewer));
      CHKERRQ(ISGetSize(ISForVertices,&nv));
      CHKERRQ(PetscViewerASCIIPushSynchronized(pcbddc->dbg_viewer));
      CHKERRQ(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"--------------------------------------------------------------\n"));
      CHKERRQ(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d got %02d local candidate vertices (%D)\n",PetscGlobalRank,nv,pcbddc->use_vertices));
      CHKERRQ(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d got %02d local candidate edges    (%D)\n",PetscGlobalRank,n_ISForEdges,pcbddc->use_edges));
      CHKERRQ(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d got %02d local candidate faces    (%D)\n",PetscGlobalRank,n_ISForFaces,pcbddc->use_faces));
      CHKERRQ(PetscViewerFlush(pcbddc->dbg_viewer));
      CHKERRQ(PetscViewerASCIIPopSynchronized(pcbddc->dbg_viewer));
    }

    /* free unneeded index sets */
    if (!pcbddc->use_vertices) {
      CHKERRQ(ISDestroy(&ISForVertices));
    }
    if (!pcbddc->use_edges) {
      for (i=0;i<n_ISForEdges;i++) {
        CHKERRQ(ISDestroy(&ISForEdges[i]));
      }
      CHKERRQ(PetscFree(ISForEdges));
      n_ISForEdges = 0;
    }
    if (!pcbddc->use_faces) {
      for (i=0;i<n_ISForFaces;i++) {
        CHKERRQ(ISDestroy(&ISForFaces[i]));
      }
      CHKERRQ(PetscFree(ISForFaces));
      n_ISForFaces = 0;
    }

    /* check if near null space is attached to global mat */
    if (pcbddc->use_nnsp) {
      CHKERRQ(MatGetNearNullSpace(pc->pmat,&nearnullsp));
    } else nearnullsp = NULL;

    if (nearnullsp) {
      CHKERRQ(MatNullSpaceGetVecs(nearnullsp,&nnsp_has_cnst,&nnsp_size,&nearnullvecs));
      /* remove any stored info */
      CHKERRQ(MatNullSpaceDestroy(&pcbddc->onearnullspace));
      CHKERRQ(PetscFree(pcbddc->onearnullvecs_state));
      /* store information for BDDC solver reuse */
      CHKERRQ(PetscObjectReference((PetscObject)nearnullsp));
      pcbddc->onearnullspace = nearnullsp;
      CHKERRQ(PetscMalloc1(nnsp_size,&pcbddc->onearnullvecs_state));
      for (i=0;i<nnsp_size;i++) {
        CHKERRQ(PetscObjectStateGet((PetscObject)nearnullvecs[i],&pcbddc->onearnullvecs_state[i]));
      }
    } else { /* if near null space is not provided BDDC uses constants by default */
      nnsp_size = 0;
      nnsp_has_cnst = PETSC_TRUE;
    }
    /* get max number of constraints on a single cc */
    max_constraints = nnsp_size;
    if (nnsp_has_cnst) max_constraints++;

    /*
         Evaluate maximum storage size needed by the procedure
         - Indices for connected component i stored at "constraints_idxs + constraints_idxs_ptr[i]"
         - Values for constraints on connected component i stored at "constraints_data + constraints_data_ptr[i]"
         There can be multiple constraints per connected component
                                                                                                                                                           */
    n_vertices = 0;
    if (ISForVertices) {
      CHKERRQ(ISGetSize(ISForVertices,&n_vertices));
    }
    ncc = n_vertices+n_ISForFaces+n_ISForEdges;
    CHKERRQ(PetscMalloc3(ncc+1,&constraints_idxs_ptr,ncc+1,&constraints_data_ptr,ncc,&constraints_n));

    total_counts = n_ISForFaces+n_ISForEdges;
    total_counts *= max_constraints;
    total_counts += n_vertices;
    CHKERRQ(PetscBTCreate(total_counts,&change_basis));

    total_counts = 0;
    max_size_of_constraint = 0;
    for (i=0;i<n_ISForEdges+n_ISForFaces;i++) {
      IS used_is;
      if (i<n_ISForEdges) {
        used_is = ISForEdges[i];
      } else {
        used_is = ISForFaces[i-n_ISForEdges];
      }
      CHKERRQ(ISGetSize(used_is,&j));
      total_counts += j;
      max_size_of_constraint = PetscMax(j,max_size_of_constraint);
    }
    CHKERRQ(PetscMalloc3(total_counts*max_constraints+n_vertices,&constraints_data,total_counts+n_vertices,&constraints_idxs,total_counts+n_vertices,&constraints_idxs_B));

    /* get local part of global near null space vectors */
    CHKERRQ(PetscMalloc1(nnsp_size,&localnearnullsp));
    for (k=0;k<nnsp_size;k++) {
      CHKERRQ(VecDuplicate(pcis->vec1_N,&localnearnullsp[k]));
      CHKERRQ(VecScatterBegin(matis->rctx,nearnullvecs[k],localnearnullsp[k],INSERT_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterEnd(matis->rctx,nearnullvecs[k],localnearnullsp[k],INSERT_VALUES,SCATTER_FORWARD));
    }

    /* whether or not to skip lapack calls */
    skip_lapack = PETSC_TRUE;
    if (n_ISForFaces+n_ISForEdges && max_constraints > 1 && !pcbddc->use_nnsp_true) skip_lapack = PETSC_FALSE;

    /* First we issue queries to allocate optimal workspace for LAPACKgesvd (or LAPACKsyev if SVD is missing) */
    if (!skip_lapack) {
      PetscScalar temp_work;

      if (use_pod) {
        /* Proper Orthogonal Decomposition (POD) using the snapshot method */
        CHKERRQ(PetscMalloc1(max_constraints*max_constraints,&correlation_mat));
        CHKERRQ(PetscMalloc1(max_constraints,&singular_vals));
        CHKERRQ(PetscMalloc1(max_size_of_constraint*max_constraints,&temp_basis));
#if defined(PETSC_USE_COMPLEX)
        CHKERRQ(PetscMalloc1(3*max_constraints,&rwork));
#endif
        /* now we evaluate the optimal workspace using query with lwork=-1 */
        CHKERRQ(PetscBLASIntCast(max_constraints,&Blas_N));
        CHKERRQ(PetscBLASIntCast(max_constraints,&Blas_LDA));
        lwork = -1;
        CHKERRQ(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
#if !defined(PETSC_USE_COMPLEX)
        PetscStackCallBLAS("LAPACKsyev",LAPACKsyev_("V","U",&Blas_N,correlation_mat,&Blas_LDA,singular_vals,&temp_work,&lwork,&lierr));
#else
        PetscStackCallBLAS("LAPACKsyev",LAPACKsyev_("V","U",&Blas_N,correlation_mat,&Blas_LDA,singular_vals,&temp_work,&lwork,rwork,&lierr));
#endif
        CHKERRQ(PetscFPTrapPop());
        PetscCheckFalse(lierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in query to SYEV Lapack routine %d",(int)lierr);
      } else {
#if !defined(PETSC_MISSING_LAPACK_GESVD)
        /* SVD */
        PetscInt max_n,min_n;
        max_n = max_size_of_constraint;
        min_n = max_constraints;
        if (max_size_of_constraint < max_constraints) {
          min_n = max_size_of_constraint;
          max_n = max_constraints;
        }
        CHKERRQ(PetscMalloc1(min_n,&singular_vals));
#if defined(PETSC_USE_COMPLEX)
        CHKERRQ(PetscMalloc1(5*min_n,&rwork));
#endif
        /* now we evaluate the optimal workspace using query with lwork=-1 */
        lwork = -1;
        CHKERRQ(PetscBLASIntCast(max_n,&Blas_M));
        CHKERRQ(PetscBLASIntCast(min_n,&Blas_N));
        CHKERRQ(PetscBLASIntCast(max_n,&Blas_LDA));
        CHKERRQ(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
#if !defined(PETSC_USE_COMPLEX)
        PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("O","N",&Blas_M,&Blas_N,&constraints_data[0],&Blas_LDA,singular_vals,&dummy_scalar,&dummy_int,&dummy_scalar,&dummy_int,&temp_work,&lwork,&lierr));
#else
        PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("O","N",&Blas_M,&Blas_N,&constraints_data[0],&Blas_LDA,singular_vals,&dummy_scalar,&dummy_int,&dummy_scalar,&dummy_int,&temp_work,&lwork,rwork,&lierr));
#endif
        CHKERRQ(PetscFPTrapPop());
        PetscCheckFalse(lierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in query to GESVD Lapack routine %d",(int)lierr);
#else
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"This should not happen");
#endif /* on missing GESVD */
      }
      /* Allocate optimal workspace */
      CHKERRQ(PetscBLASIntCast((PetscInt)PetscRealPart(temp_work),&lwork));
      CHKERRQ(PetscMalloc1(lwork,&work));
    }
    /* Now we can loop on constraining sets */
    total_counts = 0;
    constraints_idxs_ptr[0] = 0;
    constraints_data_ptr[0] = 0;
    /* vertices */
    if (n_vertices) {
      CHKERRQ(ISGetIndices(ISForVertices,(const PetscInt**)&is_indices));
      CHKERRQ(PetscArraycpy(constraints_idxs,is_indices,n_vertices));
      for (i=0;i<n_vertices;i++) {
        constraints_n[total_counts] = 1;
        constraints_data[total_counts] = 1.0;
        constraints_idxs_ptr[total_counts+1] = constraints_idxs_ptr[total_counts]+1;
        constraints_data_ptr[total_counts+1] = constraints_data_ptr[total_counts]+1;
        total_counts++;
      }
      CHKERRQ(ISRestoreIndices(ISForVertices,(const PetscInt**)&is_indices));
      n_vertices = total_counts;
    }

    /* edges and faces */
    total_counts_cc = total_counts;
    for (ncc=0;ncc<n_ISForEdges+n_ISForFaces;ncc++) {
      IS        used_is;
      PetscBool idxs_copied = PETSC_FALSE;

      if (ncc<n_ISForEdges) {
        used_is = ISForEdges[ncc];
        boolforchange = pcbddc->use_change_of_basis; /* change or not the basis on the edge */
      } else {
        used_is = ISForFaces[ncc-n_ISForEdges];
        boolforchange = (PetscBool)(pcbddc->use_change_of_basis && pcbddc->use_change_on_faces); /* change or not the basis on the face */
      }
      temp_constraints = 0;          /* zero the number of constraints I have on this conn comp */

      CHKERRQ(ISGetSize(used_is,&size_of_constraint));
      CHKERRQ(ISGetIndices(used_is,(const PetscInt**)&is_indices));
      /* change of basis should not be performed on local periodic nodes */
      if (pcbddc->mat_graph->mirrors && pcbddc->mat_graph->mirrors[is_indices[0]]) boolforchange = PETSC_FALSE;
      if (nnsp_has_cnst) {
        PetscScalar quad_value;

        CHKERRQ(PetscArraycpy(constraints_idxs + constraints_idxs_ptr[total_counts_cc],is_indices,size_of_constraint));
        idxs_copied = PETSC_TRUE;

        if (!pcbddc->use_nnsp_true) {
          quad_value = (PetscScalar)(1.0/PetscSqrtReal((PetscReal)size_of_constraint));
        } else {
          quad_value = 1.0;
        }
        for (j=0;j<size_of_constraint;j++) {
          constraints_data[constraints_data_ptr[total_counts_cc]+j] = quad_value;
        }
        temp_constraints++;
        total_counts++;
      }
      for (k=0;k<nnsp_size;k++) {
        PetscReal real_value;
        PetscScalar *ptr_to_data;

        CHKERRQ(VecGetArrayRead(localnearnullsp[k],(const PetscScalar**)&array));
        ptr_to_data = &constraints_data[constraints_data_ptr[total_counts_cc]+temp_constraints*size_of_constraint];
        for (j=0;j<size_of_constraint;j++) {
          ptr_to_data[j] = array[is_indices[j]];
        }
        CHKERRQ(VecRestoreArrayRead(localnearnullsp[k],(const PetscScalar**)&array));
        /* check if array is null on the connected component */
        CHKERRQ(PetscBLASIntCast(size_of_constraint,&Blas_N));
        PetscStackCallBLAS("BLASasum",real_value = BLASasum_(&Blas_N,ptr_to_data,&Blas_one));
        if (real_value > tol*size_of_constraint) { /* keep indices and values */
          temp_constraints++;
          total_counts++;
          if (!idxs_copied) {
            CHKERRQ(PetscArraycpy(constraints_idxs + constraints_idxs_ptr[total_counts_cc],is_indices,size_of_constraint));
            idxs_copied = PETSC_TRUE;
          }
        }
      }
      CHKERRQ(ISRestoreIndices(used_is,(const PetscInt**)&is_indices));
      valid_constraints = temp_constraints;
      if (!pcbddc->use_nnsp_true && temp_constraints) {
        if (temp_constraints == 1) { /* just normalize the constraint */
          PetscScalar norm,*ptr_to_data;

          ptr_to_data = &constraints_data[constraints_data_ptr[total_counts_cc]];
          CHKERRQ(PetscBLASIntCast(size_of_constraint,&Blas_N));
          PetscStackCallBLAS("BLASdot",norm = BLASdot_(&Blas_N,ptr_to_data,&Blas_one,ptr_to_data,&Blas_one));
          norm = 1.0/PetscSqrtReal(PetscRealPart(norm));
          PetscStackCallBLAS("BLASscal",BLASscal_(&Blas_N,&norm,ptr_to_data,&Blas_one));
        } else { /* perform SVD */
          PetscScalar *ptr_to_data = &constraints_data[constraints_data_ptr[total_counts_cc]];

          if (use_pod) {
            /* SVD: Y = U*S*V^H                -> U (eigenvectors of Y*Y^H) = Y*V*(S)^\dag
               POD: Y^H*Y = V*D*V^H, D = S^H*S -> U = Y*V*D^(-1/2)
               -> When PETSC_USE_COMPLEX and PETSC_MISSING_LAPACK_GESVD are defined
                  the constraints basis will differ (by a complex factor with absolute value equal to 1)
                  from that computed using LAPACKgesvd
               -> This is due to a different computation of eigenvectors in LAPACKheev
               -> The quality of the POD-computed basis will be the same */
            CHKERRQ(PetscArrayzero(correlation_mat,temp_constraints*temp_constraints));
            /* Store upper triangular part of correlation matrix */
            CHKERRQ(PetscBLASIntCast(size_of_constraint,&Blas_N));
            CHKERRQ(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
            for (j=0;j<temp_constraints;j++) {
              for (k=0;k<j+1;k++) {
                PetscStackCallBLAS("BLASdot",correlation_mat[j*temp_constraints+k] = BLASdot_(&Blas_N,ptr_to_data+k*size_of_constraint,&Blas_one,ptr_to_data+j*size_of_constraint,&Blas_one));
              }
            }
            /* compute eigenvalues and eigenvectors of correlation matrix */
            CHKERRQ(PetscBLASIntCast(temp_constraints,&Blas_N));
            CHKERRQ(PetscBLASIntCast(temp_constraints,&Blas_LDA));
#if !defined(PETSC_USE_COMPLEX)
            PetscStackCallBLAS("LAPACKsyev",LAPACKsyev_("V","U",&Blas_N,correlation_mat,&Blas_LDA,singular_vals,work,&lwork,&lierr));
#else
            PetscStackCallBLAS("LAPACKsyev",LAPACKsyev_("V","U",&Blas_N,correlation_mat,&Blas_LDA,singular_vals,work,&lwork,rwork,&lierr));
#endif
            CHKERRQ(PetscFPTrapPop());
            PetscCheckFalse(lierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in SYEV Lapack routine %d",(int)lierr);
            /* retain eigenvalues greater than tol: note that LAPACKsyev gives eigs in ascending order */
            j = 0;
            while (j < temp_constraints && singular_vals[j]/singular_vals[temp_constraints-1] < tol) j++;
            total_counts = total_counts-j;
            valid_constraints = temp_constraints-j;
            /* scale and copy POD basis into used quadrature memory */
            CHKERRQ(PetscBLASIntCast(size_of_constraint,&Blas_M));
            CHKERRQ(PetscBLASIntCast(temp_constraints,&Blas_N));
            CHKERRQ(PetscBLASIntCast(temp_constraints,&Blas_K));
            CHKERRQ(PetscBLASIntCast(size_of_constraint,&Blas_LDA));
            CHKERRQ(PetscBLASIntCast(temp_constraints,&Blas_LDB));
            CHKERRQ(PetscBLASIntCast(size_of_constraint,&Blas_LDC));
            if (j<temp_constraints) {
              PetscInt ii;
              for (k=j;k<temp_constraints;k++) singular_vals[k] = 1.0/PetscSqrtReal(singular_vals[k]);
              CHKERRQ(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
              PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&Blas_M,&Blas_N,&Blas_K,&one,ptr_to_data,&Blas_LDA,correlation_mat,&Blas_LDB,&zero,temp_basis,&Blas_LDC));
              CHKERRQ(PetscFPTrapPop());
              for (k=0;k<temp_constraints-j;k++) {
                for (ii=0;ii<size_of_constraint;ii++) {
                  ptr_to_data[k*size_of_constraint+ii] = singular_vals[temp_constraints-1-k]*temp_basis[(temp_constraints-1-k)*size_of_constraint+ii];
                }
              }
            }
          } else {
#if !defined(PETSC_MISSING_LAPACK_GESVD)
            CHKERRQ(PetscBLASIntCast(size_of_constraint,&Blas_M));
            CHKERRQ(PetscBLASIntCast(temp_constraints,&Blas_N));
            CHKERRQ(PetscBLASIntCast(size_of_constraint,&Blas_LDA));
            CHKERRQ(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
#if !defined(PETSC_USE_COMPLEX)
            PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("O","N",&Blas_M,&Blas_N,ptr_to_data,&Blas_LDA,singular_vals,&dummy_scalar,&dummy_int,&dummy_scalar,&dummy_int,work,&lwork,&lierr));
#else
            PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("O","N",&Blas_M,&Blas_N,ptr_to_data,&Blas_LDA,singular_vals,&dummy_scalar,&dummy_int,&dummy_scalar,&dummy_int,work,&lwork,rwork,&lierr));
#endif
            PetscCheckFalse(lierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in GESVD Lapack routine %d",(int)lierr);
            CHKERRQ(PetscFPTrapPop());
            /* retain eigenvalues greater than tol: note that LAPACKgesvd gives eigs in descending order */
            k = temp_constraints;
            if (k > size_of_constraint) k = size_of_constraint;
            j = 0;
            while (j < k && singular_vals[k-j-1]/singular_vals[0] < tol) j++;
            valid_constraints = k-j;
            total_counts = total_counts-temp_constraints+valid_constraints;
#else
            SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"This should not happen");
#endif /* on missing GESVD */
          }
        }
      }
      /* update pointers information */
      if (valid_constraints) {
        constraints_n[total_counts_cc] = valid_constraints;
        constraints_idxs_ptr[total_counts_cc+1] = constraints_idxs_ptr[total_counts_cc]+size_of_constraint;
        constraints_data_ptr[total_counts_cc+1] = constraints_data_ptr[total_counts_cc]+size_of_constraint*valid_constraints;
        /* set change_of_basis flag */
        if (boolforchange) {
          PetscBTSet(change_basis,total_counts_cc);
        }
        total_counts_cc++;
      }
    }
    /* free workspace */
    if (!skip_lapack) {
      CHKERRQ(PetscFree(work));
#if defined(PETSC_USE_COMPLEX)
      CHKERRQ(PetscFree(rwork));
#endif
      CHKERRQ(PetscFree(singular_vals));
      CHKERRQ(PetscFree(correlation_mat));
      CHKERRQ(PetscFree(temp_basis));
    }
    for (k=0;k<nnsp_size;k++) {
      CHKERRQ(VecDestroy(&localnearnullsp[k]));
    }
    CHKERRQ(PetscFree(localnearnullsp));
    /* free index sets of faces, edges and vertices */
    for (i=0;i<n_ISForFaces;i++) {
      CHKERRQ(ISDestroy(&ISForFaces[i]));
    }
    if (n_ISForFaces) {
      CHKERRQ(PetscFree(ISForFaces));
    }
    for (i=0;i<n_ISForEdges;i++) {
      CHKERRQ(ISDestroy(&ISForEdges[i]));
    }
    if (n_ISForEdges) {
      CHKERRQ(PetscFree(ISForEdges));
    }
    CHKERRQ(ISDestroy(&ISForVertices));
  } else {
    PCBDDCSubSchurs sub_schurs = pcbddc->sub_schurs;

    total_counts = 0;
    n_vertices = 0;
    if (sub_schurs->is_vertices && pcbddc->use_vertices) {
      CHKERRQ(ISGetLocalSize(sub_schurs->is_vertices,&n_vertices));
    }
    max_constraints = 0;
    total_counts_cc = 0;
    for (i=0;i<sub_schurs->n_subs+n_vertices;i++) {
      total_counts += pcbddc->adaptive_constraints_n[i];
      if (pcbddc->adaptive_constraints_n[i]) total_counts_cc++;
      max_constraints = PetscMax(max_constraints,pcbddc->adaptive_constraints_n[i]);
    }
    constraints_idxs_ptr = pcbddc->adaptive_constraints_idxs_ptr;
    constraints_data_ptr = pcbddc->adaptive_constraints_data_ptr;
    constraints_idxs = pcbddc->adaptive_constraints_idxs;
    constraints_data = pcbddc->adaptive_constraints_data;
    /* constraints_n differs from pcbddc->adaptive_constraints_n */
    CHKERRQ(PetscMalloc1(total_counts_cc,&constraints_n));
    total_counts_cc = 0;
    for (i=0;i<sub_schurs->n_subs+n_vertices;i++) {
      if (pcbddc->adaptive_constraints_n[i]) {
        constraints_n[total_counts_cc++] = pcbddc->adaptive_constraints_n[i];
      }
    }

    max_size_of_constraint = 0;
    for (i=0;i<total_counts_cc;i++) max_size_of_constraint = PetscMax(max_size_of_constraint,constraints_idxs_ptr[i+1]-constraints_idxs_ptr[i]);
    CHKERRQ(PetscMalloc1(constraints_idxs_ptr[total_counts_cc],&constraints_idxs_B));
    /* Change of basis */
    CHKERRQ(PetscBTCreate(total_counts_cc,&change_basis));
    if (pcbddc->use_change_of_basis) {
      for (i=0;i<sub_schurs->n_subs;i++) {
        if (PetscBTLookup(sub_schurs->is_edge,i) || pcbddc->use_change_on_faces) {
          CHKERRQ(PetscBTSet(change_basis,i+n_vertices));
        }
      }
    }
  }
  pcbddc->local_primal_size = total_counts;
  CHKERRQ(PetscMalloc1(pcbddc->local_primal_size+pcbddc->benign_n,&pcbddc->primal_indices_local_idxs));

  /* map constraints_idxs in boundary numbering */
  CHKERRQ(ISGlobalToLocalMappingApply(pcis->BtoNmap,IS_GTOLM_DROP,constraints_idxs_ptr[total_counts_cc],constraints_idxs,&i,constraints_idxs_B));
  PetscCheckFalse(i != constraints_idxs_ptr[total_counts_cc],PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error in boundary numbering for constraints indices %D != %D",constraints_idxs_ptr[total_counts_cc],i);

  /* Create constraint matrix */
  CHKERRQ(MatCreate(PETSC_COMM_SELF,&pcbddc->ConstraintMatrix));
  CHKERRQ(MatSetType(pcbddc->ConstraintMatrix,MATAIJ));
  CHKERRQ(MatSetSizes(pcbddc->ConstraintMatrix,pcbddc->local_primal_size,pcis->n,pcbddc->local_primal_size,pcis->n));

  /* find primal_dofs: subdomain corners plus dofs selected as primal after change of basis */
  /* determine if a QR strategy is needed for change of basis */
  qr_needed = pcbddc->use_qr_single;
  CHKERRQ(PetscBTCreate(total_counts_cc,&qr_needed_idx));
  total_primal_vertices=0;
  pcbddc->local_primal_size_cc = 0;
  for (i=0;i<total_counts_cc;i++) {
    size_of_constraint = constraints_idxs_ptr[i+1]-constraints_idxs_ptr[i];
    if (size_of_constraint == 1 && pcbddc->mat_graph->custom_minimal_size) {
      pcbddc->primal_indices_local_idxs[total_primal_vertices++] = constraints_idxs[constraints_idxs_ptr[i]];
      pcbddc->local_primal_size_cc += 1;
    } else if (PetscBTLookup(change_basis,i)) {
      for (k=0;k<constraints_n[i];k++) {
        pcbddc->primal_indices_local_idxs[total_primal_vertices++] = constraints_idxs[constraints_idxs_ptr[i]+k];
      }
      pcbddc->local_primal_size_cc += constraints_n[i];
      if (constraints_n[i] > 1 || pcbddc->use_qr_single) {
        PetscBTSet(qr_needed_idx,i);
        qr_needed = PETSC_TRUE;
      }
    } else {
      pcbddc->local_primal_size_cc += 1;
    }
  }
  /* note that the local variable n_vertices used below stores the number of pointwise constraints */
  pcbddc->n_vertices = total_primal_vertices;
  /* permute indices in order to have a sorted set of vertices */
  CHKERRQ(PetscSortInt(total_primal_vertices,pcbddc->primal_indices_local_idxs));
  CHKERRQ(PetscMalloc2(pcbddc->local_primal_size_cc+pcbddc->benign_n,&pcbddc->local_primal_ref_node,pcbddc->local_primal_size_cc+pcbddc->benign_n,&pcbddc->local_primal_ref_mult));
  CHKERRQ(PetscArraycpy(pcbddc->local_primal_ref_node,pcbddc->primal_indices_local_idxs,total_primal_vertices));
  for (i=0;i<total_primal_vertices;i++) pcbddc->local_primal_ref_mult[i] = 1;

  /* nonzero structure of constraint matrix */
  /* and get reference dof for local constraints */
  CHKERRQ(PetscMalloc1(pcbddc->local_primal_size,&nnz));
  for (i=0;i<total_primal_vertices;i++) nnz[i] = 1;

  j = total_primal_vertices;
  total_counts = total_primal_vertices;
  cum = total_primal_vertices;
  for (i=n_vertices;i<total_counts_cc;i++) {
    if (!PetscBTLookup(change_basis,i)) {
      pcbddc->local_primal_ref_node[cum] = constraints_idxs[constraints_idxs_ptr[i]];
      pcbddc->local_primal_ref_mult[cum] = constraints_n[i];
      cum++;
      size_of_constraint = constraints_idxs_ptr[i+1]-constraints_idxs_ptr[i];
      for (k=0;k<constraints_n[i];k++) {
        pcbddc->primal_indices_local_idxs[total_counts++] = constraints_idxs[constraints_idxs_ptr[i]+k];
        nnz[j+k] = size_of_constraint;
      }
      j += constraints_n[i];
    }
  }
  CHKERRQ(MatSeqAIJSetPreallocation(pcbddc->ConstraintMatrix,0,nnz));
  CHKERRQ(MatSetOption(pcbddc->ConstraintMatrix,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE));
  CHKERRQ(PetscFree(nnz));

  /* set values in constraint matrix */
  for (i=0;i<total_primal_vertices;i++) {
    CHKERRQ(MatSetValue(pcbddc->ConstraintMatrix,i,pcbddc->local_primal_ref_node[i],1.0,INSERT_VALUES));
  }
  total_counts = total_primal_vertices;
  for (i=n_vertices;i<total_counts_cc;i++) {
    if (!PetscBTLookup(change_basis,i)) {
      PetscInt *cols;

      size_of_constraint = constraints_idxs_ptr[i+1]-constraints_idxs_ptr[i];
      cols = constraints_idxs+constraints_idxs_ptr[i];
      for (k=0;k<constraints_n[i];k++) {
        PetscInt    row = total_counts+k;
        PetscScalar *vals;

        vals = constraints_data+constraints_data_ptr[i]+k*size_of_constraint;
        CHKERRQ(MatSetValues(pcbddc->ConstraintMatrix,1,&row,size_of_constraint,cols,vals,INSERT_VALUES));
      }
      total_counts += constraints_n[i];
    }
  }
  /* assembling */
  CHKERRQ(MatAssemblyBegin(pcbddc->ConstraintMatrix,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(pcbddc->ConstraintMatrix,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatViewFromOptions(pcbddc->ConstraintMatrix,(PetscObject)pc,"-pc_bddc_constraint_mat_view"));

  /* Create matrix for change of basis. We don't need it in case pcbddc->use_change_of_basis is FALSE */
  if (pcbddc->use_change_of_basis) {
    /* dual and primal dofs on a single cc */
    PetscInt     dual_dofs,primal_dofs;
    /* working stuff for GEQRF */
    PetscScalar  *qr_basis = NULL,*qr_tau = NULL,*qr_work = NULL,lqr_work_t;
    PetscBLASInt lqr_work;
    /* working stuff for UNGQR */
    PetscScalar  *gqr_work = NULL,lgqr_work_t=0.0;
    PetscBLASInt lgqr_work;
    /* working stuff for TRTRS */
    PetscScalar  *trs_rhs = NULL;
    PetscBLASInt Blas_NRHS;
    /* pointers for values insertion into change of basis matrix */
    PetscInt     *start_rows,*start_cols;
    PetscScalar  *start_vals;
    /* working stuff for values insertion */
    PetscBT      is_primal;
    PetscInt     *aux_primal_numbering_B;
    /* matrix sizes */
    PetscInt     global_size,local_size;
    /* temporary change of basis */
    Mat          localChangeOfBasisMatrix;
    /* extra space for debugging */
    PetscScalar  *dbg_work = NULL;

    /* local temporary change of basis acts on local interfaces -> dimension is n_B x n_B */
    CHKERRQ(MatCreate(PETSC_COMM_SELF,&localChangeOfBasisMatrix));
    CHKERRQ(MatSetType(localChangeOfBasisMatrix,MATAIJ));
    CHKERRQ(MatSetSizes(localChangeOfBasisMatrix,pcis->n,pcis->n,pcis->n,pcis->n));
    /* nonzeros for local mat */
    CHKERRQ(PetscMalloc1(pcis->n,&nnz));
    if (!pcbddc->benign_change || pcbddc->fake_change) {
      for (i=0;i<pcis->n;i++) nnz[i]=1;
    } else {
      const PetscInt *ii;
      PetscInt       n;
      PetscBool      flg_row;
      CHKERRQ(MatGetRowIJ(pcbddc->benign_change,0,PETSC_FALSE,PETSC_FALSE,&n,&ii,NULL,&flg_row));
      for (i=0;i<n;i++) nnz[i] = ii[i+1]-ii[i];
      CHKERRQ(MatRestoreRowIJ(pcbddc->benign_change,0,PETSC_FALSE,PETSC_FALSE,&n,&ii,NULL,&flg_row));
    }
    for (i=n_vertices;i<total_counts_cc;i++) {
      if (PetscBTLookup(change_basis,i)) {
        size_of_constraint = constraints_idxs_ptr[i+1]-constraints_idxs_ptr[i];
        if (PetscBTLookup(qr_needed_idx,i)) {
          for (j=0;j<size_of_constraint;j++) nnz[constraints_idxs[constraints_idxs_ptr[i]+j]] = size_of_constraint;
        } else {
          nnz[constraints_idxs[constraints_idxs_ptr[i]]] = size_of_constraint;
          for (j=1;j<size_of_constraint;j++) nnz[constraints_idxs[constraints_idxs_ptr[i]+j]] = 2;
        }
      }
    }
    CHKERRQ(MatSeqAIJSetPreallocation(localChangeOfBasisMatrix,0,nnz));
    CHKERRQ(MatSetOption(localChangeOfBasisMatrix,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE));
    CHKERRQ(PetscFree(nnz));
    /* Set interior change in the matrix */
    if (!pcbddc->benign_change || pcbddc->fake_change) {
      for (i=0;i<pcis->n;i++) {
        CHKERRQ(MatSetValue(localChangeOfBasisMatrix,i,i,1.0,INSERT_VALUES));
      }
    } else {
      const PetscInt *ii,*jj;
      PetscScalar    *aa;
      PetscInt       n;
      PetscBool      flg_row;
      CHKERRQ(MatGetRowIJ(pcbddc->benign_change,0,PETSC_FALSE,PETSC_FALSE,&n,&ii,&jj,&flg_row));
      CHKERRQ(MatSeqAIJGetArray(pcbddc->benign_change,&aa));
      for (i=0;i<n;i++) {
        CHKERRQ(MatSetValues(localChangeOfBasisMatrix,1,&i,ii[i+1]-ii[i],jj+ii[i],aa+ii[i],INSERT_VALUES));
      }
      CHKERRQ(MatSeqAIJRestoreArray(pcbddc->benign_change,&aa));
      CHKERRQ(MatRestoreRowIJ(pcbddc->benign_change,0,PETSC_FALSE,PETSC_FALSE,&n,&ii,&jj,&flg_row));
    }

    if (pcbddc->dbg_flag) {
      CHKERRQ(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"--------------------------------------------------------------\n"));
      CHKERRQ(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Checking change of basis computation for subdomain %04d\n",PetscGlobalRank));
    }

    /* Now we loop on the constraints which need a change of basis */
    /*
       Change of basis matrix is evaluated similarly to the FIRST APPROACH in
       Klawonn and Widlund, Dual-primal FETI-DP methods for linear elasticity, (see Sect 6.2.1)

       Basic blocks of change of basis matrix T computed by

          - Using the following block transformation if there is only a primal dof on the cc (and -pc_bddc_use_qr_single is not specified)

            | 1        0   ...        0         s_1/S |
            | 0        1   ...        0         s_2/S |
            |              ...                        |
            | 0        ...            1     s_{n-1}/S |
            | -s_1/s_n ...    -s_{n-1}/s_n      s_n/S |

            with S = \sum_{i=1}^n s_i^2
            NOTE: in the above example, the primal dof is the last one of the edge in LOCAL ordering
                  in the current implementation, the primal dof is the first one of the edge in GLOBAL ordering

          - QR decomposition of constraints otherwise
    */
    if (qr_needed && max_size_of_constraint) {
      /* space to store Q */
      CHKERRQ(PetscMalloc1(max_size_of_constraint*max_size_of_constraint,&qr_basis));
      /* array to store scaling factors for reflectors */
      CHKERRQ(PetscMalloc1(max_constraints,&qr_tau));
      /* first we issue queries for optimal work */
      CHKERRQ(PetscBLASIntCast(max_size_of_constraint,&Blas_M));
      CHKERRQ(PetscBLASIntCast(max_constraints,&Blas_N));
      CHKERRQ(PetscBLASIntCast(max_size_of_constraint,&Blas_LDA));
      lqr_work = -1;
      PetscStackCallBLAS("LAPACKgeqrf",LAPACKgeqrf_(&Blas_M,&Blas_N,qr_basis,&Blas_LDA,qr_tau,&lqr_work_t,&lqr_work,&lierr));
      PetscCheckFalse(lierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in query to GEQRF Lapack routine %d",(int)lierr);
      CHKERRQ(PetscBLASIntCast((PetscInt)PetscRealPart(lqr_work_t),&lqr_work));
      CHKERRQ(PetscMalloc1((PetscInt)PetscRealPart(lqr_work_t),&qr_work));
      lgqr_work = -1;
      CHKERRQ(PetscBLASIntCast(max_size_of_constraint,&Blas_M));
      CHKERRQ(PetscBLASIntCast(max_size_of_constraint,&Blas_N));
      CHKERRQ(PetscBLASIntCast(max_constraints,&Blas_K));
      CHKERRQ(PetscBLASIntCast(max_size_of_constraint,&Blas_LDA));
      if (Blas_K>Blas_M) Blas_K=Blas_M; /* adjust just for computing optimal work */
      PetscStackCallBLAS("LAPACKorgqr",LAPACKorgqr_(&Blas_M,&Blas_N,&Blas_K,qr_basis,&Blas_LDA,qr_tau,&lgqr_work_t,&lgqr_work,&lierr));
      PetscCheckFalse(lierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in query to ORGQR/UNGQR Lapack routine %d",(int)lierr);
      CHKERRQ(PetscBLASIntCast((PetscInt)PetscRealPart(lgqr_work_t),&lgqr_work));
      CHKERRQ(PetscMalloc1((PetscInt)PetscRealPart(lgqr_work_t),&gqr_work));
      /* array to store rhs and solution of triangular solver */
      CHKERRQ(PetscMalloc1(max_constraints*max_constraints,&trs_rhs));
      /* allocating workspace for check */
      if (pcbddc->dbg_flag) {
        CHKERRQ(PetscMalloc1(max_size_of_constraint*(max_constraints+max_size_of_constraint),&dbg_work));
      }
    }
    /* array to store whether a node is primal or not */
    CHKERRQ(PetscBTCreate(pcis->n_B,&is_primal));
    CHKERRQ(PetscMalloc1(total_primal_vertices,&aux_primal_numbering_B));
    CHKERRQ(ISGlobalToLocalMappingApply(pcis->BtoNmap,IS_GTOLM_DROP,total_primal_vertices,pcbddc->local_primal_ref_node,&i,aux_primal_numbering_B));
    PetscCheckFalse(i != total_primal_vertices,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error in boundary numbering for BDDC vertices! %D != %D",total_primal_vertices,i);
    for (i=0;i<total_primal_vertices;i++) {
      CHKERRQ(PetscBTSet(is_primal,aux_primal_numbering_B[i]));
    }
    CHKERRQ(PetscFree(aux_primal_numbering_B));

    /* loop on constraints and see whether or not they need a change of basis and compute it */
    for (total_counts=n_vertices;total_counts<total_counts_cc;total_counts++) {
      size_of_constraint = constraints_idxs_ptr[total_counts+1]-constraints_idxs_ptr[total_counts];
      if (PetscBTLookup(change_basis,total_counts)) {
        /* get constraint info */
        primal_dofs = constraints_n[total_counts];
        dual_dofs = size_of_constraint-primal_dofs;

        if (pcbddc->dbg_flag) {
          CHKERRQ(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Constraints %D: %D need a change of basis (size %D)\n",total_counts,primal_dofs,size_of_constraint));
        }

        if (PetscBTLookup(qr_needed_idx,total_counts)) { /* QR */

          /* copy quadrature constraints for change of basis check */
          if (pcbddc->dbg_flag) {
            CHKERRQ(PetscArraycpy(dbg_work,&constraints_data[constraints_data_ptr[total_counts]],size_of_constraint*primal_dofs));
          }
          /* copy temporary constraints into larger work vector (in order to store all columns of Q) */
          CHKERRQ(PetscArraycpy(qr_basis,&constraints_data[constraints_data_ptr[total_counts]],size_of_constraint*primal_dofs));

          /* compute QR decomposition of constraints */
          CHKERRQ(PetscBLASIntCast(size_of_constraint,&Blas_M));
          CHKERRQ(PetscBLASIntCast(primal_dofs,&Blas_N));
          CHKERRQ(PetscBLASIntCast(size_of_constraint,&Blas_LDA));
          CHKERRQ(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
          PetscStackCallBLAS("LAPACKgeqrf",LAPACKgeqrf_(&Blas_M,&Blas_N,qr_basis,&Blas_LDA,qr_tau,qr_work,&lqr_work,&lierr));
          PetscCheckFalse(lierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in GEQRF Lapack routine %d",(int)lierr);
          CHKERRQ(PetscFPTrapPop());

          /* explicitly compute R^-T */
          CHKERRQ(PetscArrayzero(trs_rhs,primal_dofs*primal_dofs));
          for (j=0;j<primal_dofs;j++) trs_rhs[j*(primal_dofs+1)] = 1.0;
          CHKERRQ(PetscBLASIntCast(primal_dofs,&Blas_N));
          CHKERRQ(PetscBLASIntCast(primal_dofs,&Blas_NRHS));
          CHKERRQ(PetscBLASIntCast(size_of_constraint,&Blas_LDA));
          CHKERRQ(PetscBLASIntCast(primal_dofs,&Blas_LDB));
          CHKERRQ(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
          PetscStackCallBLAS("LAPACKtrtrs",LAPACKtrtrs_("U","T","N",&Blas_N,&Blas_NRHS,qr_basis,&Blas_LDA,trs_rhs,&Blas_LDB,&lierr));
          PetscCheckFalse(lierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in TRTRS Lapack routine %d",(int)lierr);
          CHKERRQ(PetscFPTrapPop());

          /* explicitly compute all columns of Q (Q = [Q1 | Q2]) overwriting QR factorization in qr_basis */
          CHKERRQ(PetscBLASIntCast(size_of_constraint,&Blas_M));
          CHKERRQ(PetscBLASIntCast(size_of_constraint,&Blas_N));
          CHKERRQ(PetscBLASIntCast(primal_dofs,&Blas_K));
          CHKERRQ(PetscBLASIntCast(size_of_constraint,&Blas_LDA));
          CHKERRQ(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
          PetscStackCallBLAS("LAPACKorgqr",LAPACKorgqr_(&Blas_M,&Blas_N,&Blas_K,qr_basis,&Blas_LDA,qr_tau,gqr_work,&lgqr_work,&lierr));
          PetscCheckFalse(lierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in ORGQR/UNGQR Lapack routine %d",(int)lierr);
          CHKERRQ(PetscFPTrapPop());

          /* first primal_dofs columns of Q need to be re-scaled in order to be unitary w.r.t constraints
             i.e. C_{pxn}*Q_{nxn} should be equal to [I_pxp | 0_pxd] (see check below)
             where n=size_of_constraint, p=primal_dofs, d=dual_dofs (n=p+d), I and 0 identity and null matrix resp. */
          CHKERRQ(PetscBLASIntCast(size_of_constraint,&Blas_M));
          CHKERRQ(PetscBLASIntCast(primal_dofs,&Blas_N));
          CHKERRQ(PetscBLASIntCast(primal_dofs,&Blas_K));
          CHKERRQ(PetscBLASIntCast(size_of_constraint,&Blas_LDA));
          CHKERRQ(PetscBLASIntCast(primal_dofs,&Blas_LDB));
          CHKERRQ(PetscBLASIntCast(size_of_constraint,&Blas_LDC));
          CHKERRQ(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
          PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&Blas_M,&Blas_N,&Blas_K,&one,qr_basis,&Blas_LDA,trs_rhs,&Blas_LDB,&zero,constraints_data+constraints_data_ptr[total_counts],&Blas_LDC));
          CHKERRQ(PetscFPTrapPop());
          CHKERRQ(PetscArraycpy(qr_basis,&constraints_data[constraints_data_ptr[total_counts]],size_of_constraint*primal_dofs));

          /* insert values in change of basis matrix respecting global ordering of new primal dofs */
          start_rows = &constraints_idxs[constraints_idxs_ptr[total_counts]];
          /* insert cols for primal dofs */
          for (j=0;j<primal_dofs;j++) {
            start_vals = &qr_basis[j*size_of_constraint];
            start_cols = &constraints_idxs[constraints_idxs_ptr[total_counts]+j];
            CHKERRQ(MatSetValues(localChangeOfBasisMatrix,size_of_constraint,start_rows,1,start_cols,start_vals,INSERT_VALUES));
          }
          /* insert cols for dual dofs */
          for (j=0,k=0;j<dual_dofs;k++) {
            if (!PetscBTLookup(is_primal,constraints_idxs_B[constraints_idxs_ptr[total_counts]+k])) {
              start_vals = &qr_basis[(primal_dofs+j)*size_of_constraint];
              start_cols = &constraints_idxs[constraints_idxs_ptr[total_counts]+k];
              CHKERRQ(MatSetValues(localChangeOfBasisMatrix,size_of_constraint,start_rows,1,start_cols,start_vals,INSERT_VALUES));
              j++;
            }
          }

          /* check change of basis */
          if (pcbddc->dbg_flag) {
            PetscInt   ii,jj;
            PetscBool valid_qr=PETSC_TRUE;
            CHKERRQ(PetscBLASIntCast(primal_dofs,&Blas_M));
            CHKERRQ(PetscBLASIntCast(size_of_constraint,&Blas_N));
            CHKERRQ(PetscBLASIntCast(size_of_constraint,&Blas_K));
            CHKERRQ(PetscBLASIntCast(size_of_constraint,&Blas_LDA));
            CHKERRQ(PetscBLASIntCast(size_of_constraint,&Blas_LDB));
            CHKERRQ(PetscBLASIntCast(primal_dofs,&Blas_LDC));
            CHKERRQ(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
            PetscStackCallBLAS("BLASgemm",BLASgemm_("T","N",&Blas_M,&Blas_N,&Blas_K,&one,dbg_work,&Blas_LDA,qr_basis,&Blas_LDB,&zero,&dbg_work[size_of_constraint*primal_dofs],&Blas_LDC));
            CHKERRQ(PetscFPTrapPop());
            for (jj=0;jj<size_of_constraint;jj++) {
              for (ii=0;ii<primal_dofs;ii++) {
                if (ii != jj && PetscAbsScalar(dbg_work[size_of_constraint*primal_dofs+jj*primal_dofs+ii]) > 1.e-12) valid_qr = PETSC_FALSE;
                if (ii == jj && PetscAbsScalar(dbg_work[size_of_constraint*primal_dofs+jj*primal_dofs+ii]-(PetscReal)1) > 1.e-12) valid_qr = PETSC_FALSE;
              }
            }
            if (!valid_qr) {
              CHKERRQ(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"\t-> wrong change of basis!\n"));
              for (jj=0;jj<size_of_constraint;jj++) {
                for (ii=0;ii<primal_dofs;ii++) {
                  if (ii != jj && PetscAbsScalar(dbg_work[size_of_constraint*primal_dofs+jj*primal_dofs+ii]) > 1.e-12) {
                    CHKERRQ(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"\tQr basis function %D is not orthogonal to constraint %D (%1.14e)!\n",jj,ii,PetscAbsScalar(dbg_work[size_of_constraint*primal_dofs+jj*primal_dofs+ii])));
                  }
                  if (ii == jj && PetscAbsScalar(dbg_work[size_of_constraint*primal_dofs+jj*primal_dofs+ii]-(PetscReal)1) > 1.e-12) {
                    CHKERRQ(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"\tQr basis function %D is not unitary w.r.t constraint %D (%1.14e)!\n",jj,ii,PetscAbsScalar(dbg_work[size_of_constraint*primal_dofs+jj*primal_dofs+ii])));
                  }
                }
              }
            } else {
              CHKERRQ(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"\t-> right change of basis!\n"));
            }
          }
        } else { /* simple transformation block */
          PetscInt    row,col;
          PetscScalar val,norm;

          CHKERRQ(PetscBLASIntCast(size_of_constraint,&Blas_N));
          PetscStackCallBLAS("BLASdot",norm = BLASdot_(&Blas_N,constraints_data+constraints_data_ptr[total_counts],&Blas_one,constraints_data+constraints_data_ptr[total_counts],&Blas_one));
          for (j=0;j<size_of_constraint;j++) {
            PetscInt row_B = constraints_idxs_B[constraints_idxs_ptr[total_counts]+j];
            row = constraints_idxs[constraints_idxs_ptr[total_counts]+j];
            if (!PetscBTLookup(is_primal,row_B)) {
              col = constraints_idxs[constraints_idxs_ptr[total_counts]];
              CHKERRQ(MatSetValue(localChangeOfBasisMatrix,row,row,1.0,INSERT_VALUES));
              CHKERRQ(MatSetValue(localChangeOfBasisMatrix,row,col,constraints_data[constraints_data_ptr[total_counts]+j]/norm,INSERT_VALUES));
            } else {
              for (k=0;k<size_of_constraint;k++) {
                col = constraints_idxs[constraints_idxs_ptr[total_counts]+k];
                if (row != col) {
                  val = -constraints_data[constraints_data_ptr[total_counts]+k]/constraints_data[constraints_data_ptr[total_counts]];
                } else {
                  val = constraints_data[constraints_data_ptr[total_counts]]/norm;
                }
                CHKERRQ(MatSetValue(localChangeOfBasisMatrix,row,col,val,INSERT_VALUES));
              }
            }
          }
          if (pcbddc->dbg_flag) {
            CHKERRQ(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"\t-> using standard change of basis\n"));
          }
        }
      } else {
        if (pcbddc->dbg_flag) {
          CHKERRQ(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Constraint %D does not need a change of basis (size %D)\n",total_counts,size_of_constraint));
        }
      }
    }

    /* free workspace */
    if (qr_needed) {
      if (pcbddc->dbg_flag) {
        CHKERRQ(PetscFree(dbg_work));
      }
      CHKERRQ(PetscFree(trs_rhs));
      CHKERRQ(PetscFree(qr_tau));
      CHKERRQ(PetscFree(qr_work));
      CHKERRQ(PetscFree(gqr_work));
      CHKERRQ(PetscFree(qr_basis));
    }
    CHKERRQ(PetscBTDestroy(&is_primal));
    CHKERRQ(MatAssemblyBegin(localChangeOfBasisMatrix,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(localChangeOfBasisMatrix,MAT_FINAL_ASSEMBLY));

    /* assembling of global change of variable */
    if (!pcbddc->fake_change) {
      Mat      tmat;
      PetscInt bs;

      CHKERRQ(VecGetSize(pcis->vec1_global,&global_size));
      CHKERRQ(VecGetLocalSize(pcis->vec1_global,&local_size));
      CHKERRQ(MatDuplicate(pc->pmat,MAT_DO_NOT_COPY_VALUES,&tmat));
      CHKERRQ(MatISSetLocalMat(tmat,localChangeOfBasisMatrix));
      CHKERRQ(MatAssemblyBegin(tmat,MAT_FINAL_ASSEMBLY));
      CHKERRQ(MatAssemblyEnd(tmat,MAT_FINAL_ASSEMBLY));
      CHKERRQ(MatCreate(PetscObjectComm((PetscObject)pc),&pcbddc->ChangeOfBasisMatrix));
      CHKERRQ(MatSetType(pcbddc->ChangeOfBasisMatrix,MATAIJ));
      CHKERRQ(MatGetBlockSize(pc->pmat,&bs));
      CHKERRQ(MatSetBlockSize(pcbddc->ChangeOfBasisMatrix,bs));
      CHKERRQ(MatSetSizes(pcbddc->ChangeOfBasisMatrix,local_size,local_size,global_size,global_size));
      CHKERRQ(MatISSetMPIXAIJPreallocation_Private(tmat,pcbddc->ChangeOfBasisMatrix,PETSC_TRUE));
      CHKERRQ(MatConvert(tmat,MATAIJ,MAT_REUSE_MATRIX,&pcbddc->ChangeOfBasisMatrix));
      CHKERRQ(MatDestroy(&tmat));
      CHKERRQ(VecSet(pcis->vec1_global,0.0));
      CHKERRQ(VecSet(pcis->vec1_N,1.0));
      CHKERRQ(VecScatterBegin(matis->rctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE));
      CHKERRQ(VecScatterEnd(matis->rctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE));
      CHKERRQ(VecReciprocal(pcis->vec1_global));
      CHKERRQ(MatDiagonalScale(pcbddc->ChangeOfBasisMatrix,pcis->vec1_global,NULL));

      /* check */
      if (pcbddc->dbg_flag) {
        PetscReal error;
        Vec       x,x_change;

        CHKERRQ(VecDuplicate(pcis->vec1_global,&x));
        CHKERRQ(VecDuplicate(pcis->vec1_global,&x_change));
        CHKERRQ(VecSetRandom(x,NULL));
        CHKERRQ(VecCopy(x,pcis->vec1_global));
        CHKERRQ(VecScatterBegin(matis->rctx,x,pcis->vec1_N,INSERT_VALUES,SCATTER_FORWARD));
        CHKERRQ(VecScatterEnd(matis->rctx,x,pcis->vec1_N,INSERT_VALUES,SCATTER_FORWARD));
        CHKERRQ(MatMult(localChangeOfBasisMatrix,pcis->vec1_N,pcis->vec2_N));
        CHKERRQ(VecScatterBegin(matis->rctx,pcis->vec2_N,x,INSERT_VALUES,SCATTER_REVERSE));
        CHKERRQ(VecScatterEnd(matis->rctx,pcis->vec2_N,x,INSERT_VALUES,SCATTER_REVERSE));
        CHKERRQ(MatMult(pcbddc->ChangeOfBasisMatrix,pcis->vec1_global,x_change));
        CHKERRQ(VecAXPY(x,-1.0,x_change));
        CHKERRQ(VecNorm(x,NORM_INFINITY,&error));
        if (error > PETSC_SMALL) {
          SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_PLIB,"Error global vs local change on N: %1.6e",error);
        }
        CHKERRQ(VecDestroy(&x));
        CHKERRQ(VecDestroy(&x_change));
      }
      /* adapt sub_schurs computed (if any) */
      if (pcbddc->use_deluxe_scaling) {
        PCBDDCSubSchurs sub_schurs=pcbddc->sub_schurs;

        PetscCheckFalse(pcbddc->use_change_of_basis && pcbddc->adaptive_userdefined,PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Cannot mix automatic change of basis, adaptive selection and user-defined constraints");
        if (sub_schurs && sub_schurs->S_Ej_all) {
          Mat                    S_new,tmat;
          IS                     is_all_N,is_V_Sall = NULL;

          CHKERRQ(ISLocalToGlobalMappingApplyIS(pcis->BtoNmap,sub_schurs->is_Ej_all,&is_all_N));
          CHKERRQ(MatCreateSubMatrix(localChangeOfBasisMatrix,is_all_N,is_all_N,MAT_INITIAL_MATRIX,&tmat));
          if (pcbddc->deluxe_zerorows) {
            ISLocalToGlobalMapping NtoSall;
            IS                     is_V;
            CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,pcbddc->n_vertices,pcbddc->local_primal_ref_node,PETSC_COPY_VALUES,&is_V));
            CHKERRQ(ISLocalToGlobalMappingCreateIS(is_all_N,&NtoSall));
            CHKERRQ(ISGlobalToLocalMappingApplyIS(NtoSall,IS_GTOLM_DROP,is_V,&is_V_Sall));
            CHKERRQ(ISLocalToGlobalMappingDestroy(&NtoSall));
            CHKERRQ(ISDestroy(&is_V));
          }
          CHKERRQ(ISDestroy(&is_all_N));
          CHKERRQ(MatPtAP(sub_schurs->S_Ej_all,tmat,MAT_INITIAL_MATRIX,1.0,&S_new));
          CHKERRQ(MatDestroy(&sub_schurs->S_Ej_all));
          CHKERRQ(PetscObjectReference((PetscObject)S_new));
          if (pcbddc->deluxe_zerorows) {
            const PetscScalar *array;
            const PetscInt    *idxs_V,*idxs_all;
            PetscInt          i,n_V;

            CHKERRQ(MatZeroRowsColumnsIS(S_new,is_V_Sall,1.,NULL,NULL));
            CHKERRQ(ISGetLocalSize(is_V_Sall,&n_V));
            CHKERRQ(ISGetIndices(is_V_Sall,&idxs_V));
            CHKERRQ(ISGetIndices(sub_schurs->is_Ej_all,&idxs_all));
            CHKERRQ(VecGetArrayRead(pcis->D,&array));
            for (i=0;i<n_V;i++) {
              PetscScalar val;
              PetscInt    idx;

              idx = idxs_V[i];
              val = array[idxs_all[idxs_V[i]]];
              CHKERRQ(MatSetValue(S_new,idx,idx,val,INSERT_VALUES));
            }
            CHKERRQ(MatAssemblyBegin(S_new,MAT_FINAL_ASSEMBLY));
            CHKERRQ(MatAssemblyEnd(S_new,MAT_FINAL_ASSEMBLY));
            CHKERRQ(VecRestoreArrayRead(pcis->D,&array));
            CHKERRQ(ISRestoreIndices(sub_schurs->is_Ej_all,&idxs_all));
            CHKERRQ(ISRestoreIndices(is_V_Sall,&idxs_V));
          }
          sub_schurs->S_Ej_all = S_new;
          CHKERRQ(MatDestroy(&S_new));
          if (sub_schurs->sum_S_Ej_all) {
            CHKERRQ(MatPtAP(sub_schurs->sum_S_Ej_all,tmat,MAT_INITIAL_MATRIX,1.0,&S_new));
            CHKERRQ(MatDestroy(&sub_schurs->sum_S_Ej_all));
            CHKERRQ(PetscObjectReference((PetscObject)S_new));
            if (pcbddc->deluxe_zerorows) {
              CHKERRQ(MatZeroRowsColumnsIS(S_new,is_V_Sall,1.,NULL,NULL));
            }
            sub_schurs->sum_S_Ej_all = S_new;
            CHKERRQ(MatDestroy(&S_new));
          }
          CHKERRQ(ISDestroy(&is_V_Sall));
          CHKERRQ(MatDestroy(&tmat));
        }
        /* destroy any change of basis context in sub_schurs */
        if (sub_schurs && sub_schurs->change) {
          PetscInt i;

          for (i=0;i<sub_schurs->n_subs;i++) {
            CHKERRQ(KSPDestroy(&sub_schurs->change[i]));
          }
          CHKERRQ(PetscFree(sub_schurs->change));
        }
      }
      if (pcbddc->switch_static) { /* need to save the local change */
        pcbddc->switch_static_change = localChangeOfBasisMatrix;
      } else {
        CHKERRQ(MatDestroy(&localChangeOfBasisMatrix));
      }
      /* determine if any process has changed the pressures locally */
      pcbddc->change_interior = pcbddc->benign_have_null;
    } else { /* fake change (get back change of basis into ConstraintMatrix and info on qr) */
      CHKERRQ(MatDestroy(&pcbddc->ConstraintMatrix));
      pcbddc->ConstraintMatrix = localChangeOfBasisMatrix;
      pcbddc->use_qr_single = qr_needed;
    }
  } else if (pcbddc->user_ChangeOfBasisMatrix || pcbddc->benign_saddle_point) {
    if (!pcbddc->benign_have_null && pcbddc->user_ChangeOfBasisMatrix) {
      CHKERRQ(PetscObjectReference((PetscObject)pcbddc->user_ChangeOfBasisMatrix));
      pcbddc->ChangeOfBasisMatrix = pcbddc->user_ChangeOfBasisMatrix;
    } else {
      Mat benign_global = NULL;
      if (pcbddc->benign_have_null) {
        Mat M;

        pcbddc->change_interior = PETSC_TRUE;
        CHKERRQ(VecCopy(matis->counter,pcis->vec1_N));
        CHKERRQ(VecReciprocal(pcis->vec1_N));
        CHKERRQ(MatDuplicate(pc->pmat,MAT_DO_NOT_COPY_VALUES,&benign_global));
        if (pcbddc->benign_change) {
          CHKERRQ(MatDuplicate(pcbddc->benign_change,MAT_COPY_VALUES,&M));
          CHKERRQ(MatDiagonalScale(M,pcis->vec1_N,NULL));
        } else {
          CHKERRQ(MatCreateSeqAIJ(PETSC_COMM_SELF,pcis->n,pcis->n,1,NULL,&M));
          CHKERRQ(MatDiagonalSet(M,pcis->vec1_N,INSERT_VALUES));
        }
        CHKERRQ(MatISSetLocalMat(benign_global,M));
        CHKERRQ(MatDestroy(&M));
        CHKERRQ(MatAssemblyBegin(benign_global,MAT_FINAL_ASSEMBLY));
        CHKERRQ(MatAssemblyEnd(benign_global,MAT_FINAL_ASSEMBLY));
      }
      if (pcbddc->user_ChangeOfBasisMatrix) {
        CHKERRQ(MatMatMult(pcbddc->user_ChangeOfBasisMatrix,benign_global,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&pcbddc->ChangeOfBasisMatrix));
        CHKERRQ(MatDestroy(&benign_global));
      } else if (pcbddc->benign_have_null) {
        pcbddc->ChangeOfBasisMatrix = benign_global;
      }
    }
    if (pcbddc->switch_static && pcbddc->ChangeOfBasisMatrix) { /* need to save the local change */
      IS             is_global;
      const PetscInt *gidxs;

      CHKERRQ(ISLocalToGlobalMappingGetIndices(matis->rmapping,&gidxs));
      CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)pc),pcis->n,gidxs,PETSC_COPY_VALUES,&is_global));
      CHKERRQ(ISLocalToGlobalMappingRestoreIndices(matis->rmapping,&gidxs));
      CHKERRQ(MatCreateSubMatrixUnsorted(pcbddc->ChangeOfBasisMatrix,is_global,is_global,&pcbddc->switch_static_change));
      CHKERRQ(ISDestroy(&is_global));
    }
  }
  if (!pcbddc->fake_change && pcbddc->ChangeOfBasisMatrix && !pcbddc->work_change) {
    CHKERRQ(VecDuplicate(pcis->vec1_global,&pcbddc->work_change));
  }

  if (!pcbddc->fake_change) {
    /* add pressure dofs to set of primal nodes for numbering purposes */
    for (i=0;i<pcbddc->benign_n;i++) {
      pcbddc->local_primal_ref_node[pcbddc->local_primal_size_cc] = pcbddc->benign_p0_lidx[i];
      pcbddc->primal_indices_local_idxs[pcbddc->local_primal_size] = pcbddc->benign_p0_lidx[i];
      pcbddc->local_primal_ref_mult[pcbddc->local_primal_size_cc] = 1;
      pcbddc->local_primal_size_cc++;
      pcbddc->local_primal_size++;
    }

    /* check if a new primal space has been introduced (also take into account benign trick) */
    pcbddc->new_primal_space_local = PETSC_TRUE;
    if (olocal_primal_size == pcbddc->local_primal_size) {
      CHKERRQ(PetscArraycmp(pcbddc->local_primal_ref_node,olocal_primal_ref_node,olocal_primal_size_cc,&pcbddc->new_primal_space_local));
      pcbddc->new_primal_space_local = (PetscBool)(!pcbddc->new_primal_space_local);
      if (!pcbddc->new_primal_space_local) {
        CHKERRQ(PetscArraycmp(pcbddc->local_primal_ref_mult,olocal_primal_ref_mult,olocal_primal_size_cc,&pcbddc->new_primal_space_local));
        pcbddc->new_primal_space_local = (PetscBool)(!pcbddc->new_primal_space_local);
      }
    }
    /* new_primal_space will be used for numbering of coarse dofs, so it should be the same across all subdomains */
    CHKERRMPI(MPIU_Allreduce(&pcbddc->new_primal_space_local,&pcbddc->new_primal_space,1,MPIU_BOOL,MPI_LOR,PetscObjectComm((PetscObject)pc)));
  }
  CHKERRQ(PetscFree2(olocal_primal_ref_node,olocal_primal_ref_mult));

  /* flush dbg viewer */
  if (pcbddc->dbg_flag) {
    CHKERRQ(PetscViewerFlush(pcbddc->dbg_viewer));
  }

  /* free workspace */
  CHKERRQ(PetscBTDestroy(&qr_needed_idx));
  CHKERRQ(PetscBTDestroy(&change_basis));
  if (!pcbddc->adaptive_selection) {
    CHKERRQ(PetscFree3(constraints_idxs_ptr,constraints_data_ptr,constraints_n));
    CHKERRQ(PetscFree3(constraints_data,constraints_idxs,constraints_idxs_B));
  } else {
    ierr = PetscFree5(pcbddc->adaptive_constraints_n,
                      pcbddc->adaptive_constraints_idxs_ptr,
                      pcbddc->adaptive_constraints_data_ptr,
                      pcbddc->adaptive_constraints_idxs,
                      pcbddc->adaptive_constraints_data);CHKERRQ(ierr);
    CHKERRQ(PetscFree(constraints_n));
    CHKERRQ(PetscFree(constraints_idxs_B));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCAnalyzeInterface(PC pc)
{
  ISLocalToGlobalMapping map;
  PC_BDDC                *pcbddc = (PC_BDDC*)pc->data;
  Mat_IS                 *matis  = (Mat_IS*)pc->pmat->data;
  PetscInt               i,N;
  PetscBool              rcsr = PETSC_FALSE;

  PetscFunctionBegin;
  if (pcbddc->recompute_topography) {
    pcbddc->graphanalyzed = PETSC_FALSE;
    /* Reset previously computed graph */
    CHKERRQ(PCBDDCGraphReset(pcbddc->mat_graph));
    /* Init local Graph struct */
    CHKERRQ(MatGetSize(pc->pmat,&N,NULL));
    CHKERRQ(MatISGetLocalToGlobalMapping(pc->pmat,&map,NULL));
    CHKERRQ(PCBDDCGraphInit(pcbddc->mat_graph,map,N,pcbddc->graphmaxcount));

    if (pcbddc->user_primal_vertices_local && !pcbddc->user_primal_vertices) {
      CHKERRQ(PCBDDCConsistencyCheckIS(pc,MPI_LOR,&pcbddc->user_primal_vertices_local));
    }
    /* Check validity of the csr graph passed in by the user */
    PetscCheckFalse(pcbddc->mat_graph->nvtxs_csr && pcbddc->mat_graph->nvtxs_csr != pcbddc->mat_graph->nvtxs,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid size of local CSR graph! Found %D, expected %D",pcbddc->mat_graph->nvtxs_csr,pcbddc->mat_graph->nvtxs);

    /* Set default CSR adjacency of local dofs if not provided by the user with PCBDDCSetLocalAdjacencyGraph */
    if (!pcbddc->mat_graph->xadj && pcbddc->use_local_adj) {
      PetscInt  *xadj,*adjncy;
      PetscInt  nvtxs;
      PetscBool flg_row=PETSC_FALSE;

      CHKERRQ(MatGetRowIJ(matis->A,0,PETSC_TRUE,PETSC_FALSE,&nvtxs,(const PetscInt**)&xadj,(const PetscInt**)&adjncy,&flg_row));
      if (flg_row) {
        CHKERRQ(PCBDDCSetLocalAdjacencyGraph(pc,nvtxs,xadj,adjncy,PETSC_COPY_VALUES));
        pcbddc->computed_rowadj = PETSC_TRUE;
      }
      CHKERRQ(MatRestoreRowIJ(matis->A,0,PETSC_TRUE,PETSC_FALSE,&nvtxs,(const PetscInt**)&xadj,(const PetscInt**)&adjncy,&flg_row));
      rcsr = PETSC_TRUE;
    }
    if (pcbddc->dbg_flag) {
      CHKERRQ(PetscViewerFlush(pcbddc->dbg_viewer));
    }

    if (pcbddc->mat_graph->cdim && !pcbddc->mat_graph->cloc) {
      PetscReal    *lcoords;
      PetscInt     n;
      MPI_Datatype dimrealtype;

      /* TODO: support for blocked */
      PetscCheckFalse(pcbddc->mat_graph->cnloc != pc->pmat->rmap->n,PETSC_COMM_SELF,PETSC_ERR_USER,"Invalid number of local coordinates! Got %D, expected %D",pcbddc->mat_graph->cnloc,pc->pmat->rmap->n);
      CHKERRQ(MatGetLocalSize(matis->A,&n,NULL));
      CHKERRQ(PetscMalloc1(pcbddc->mat_graph->cdim*n,&lcoords));
      CHKERRMPI(MPI_Type_contiguous(pcbddc->mat_graph->cdim,MPIU_REAL,&dimrealtype));
      CHKERRMPI(MPI_Type_commit(&dimrealtype));
      CHKERRQ(PetscSFBcastBegin(matis->sf,dimrealtype,pcbddc->mat_graph->coords,lcoords,MPI_REPLACE));
      CHKERRQ(PetscSFBcastEnd(matis->sf,dimrealtype,pcbddc->mat_graph->coords,lcoords,MPI_REPLACE));
      CHKERRMPI(MPI_Type_free(&dimrealtype));
      CHKERRQ(PetscFree(pcbddc->mat_graph->coords));

      pcbddc->mat_graph->coords = lcoords;
      pcbddc->mat_graph->cloc   = PETSC_TRUE;
      pcbddc->mat_graph->cnloc  = n;
    }
    PetscCheckFalse(pcbddc->mat_graph->cnloc && pcbddc->mat_graph->cnloc != pcbddc->mat_graph->nvtxs,PETSC_COMM_SELF,PETSC_ERR_USER,"Invalid number of local subdomain coordinates! Got %D, expected %D",pcbddc->mat_graph->cnloc,pcbddc->mat_graph->nvtxs);
    pcbddc->mat_graph->active_coords = (PetscBool)(pcbddc->corner_selection && pcbddc->mat_graph->cdim && !pcbddc->corner_selected);

    /* Setup of Graph */
    pcbddc->mat_graph->commsizelimit = 0; /* don't use the COMM_SELF variant of the graph */
    CHKERRQ(PCBDDCGraphSetUp(pcbddc->mat_graph,pcbddc->vertex_size,pcbddc->NeumannBoundariesLocal,pcbddc->DirichletBoundariesLocal,pcbddc->n_ISForDofsLocal,pcbddc->ISForDofsLocal,pcbddc->user_primal_vertices_local));

    /* attach info on disconnected subdomains if present */
    if (pcbddc->n_local_subs) {
      PetscInt *local_subs,n,totn;

      CHKERRQ(MatGetLocalSize(matis->A,&n,NULL));
      CHKERRQ(PetscMalloc1(n,&local_subs));
      for (i=0;i<n;i++) local_subs[i] = pcbddc->n_local_subs;
      for (i=0;i<pcbddc->n_local_subs;i++) {
        const PetscInt *idxs;
        PetscInt       nl,j;

        CHKERRQ(ISGetLocalSize(pcbddc->local_subs[i],&nl));
        CHKERRQ(ISGetIndices(pcbddc->local_subs[i],&idxs));
        for (j=0;j<nl;j++) local_subs[idxs[j]] = i;
        CHKERRQ(ISRestoreIndices(pcbddc->local_subs[i],&idxs));
      }
      for (i=0,totn=0;i<n;i++) totn = PetscMax(totn,local_subs[i]);
      pcbddc->mat_graph->n_local_subs = totn + 1;
      pcbddc->mat_graph->local_subs = local_subs;
    }
  }

  if (!pcbddc->graphanalyzed) {
    /* Graph's connected components analysis */
    CHKERRQ(PCBDDCGraphComputeConnectedComponents(pcbddc->mat_graph));
    pcbddc->graphanalyzed = PETSC_TRUE;
    pcbddc->corner_selected = pcbddc->corner_selection;
  }
  if (rcsr) pcbddc->mat_graph->nvtxs_csr = 0;
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCOrthonormalizeVecs(PetscInt *nio, Vec vecs[])
{
  PetscInt       i,j,n;
  PetscScalar    *alphas;
  PetscReal      norm,*onorms;

  PetscFunctionBegin;
  n = *nio;
  if (!n) PetscFunctionReturn(0);
  CHKERRQ(PetscMalloc2(n,&alphas,n,&onorms));
  CHKERRQ(VecNormalize(vecs[0],&norm));
  if (norm < PETSC_SMALL) {
    onorms[0] = 0.0;
    CHKERRQ(VecSet(vecs[0],0.0));
  } else {
    onorms[0] = norm;
  }

  for (i=1;i<n;i++) {
    CHKERRQ(VecMDot(vecs[i],i,vecs,alphas));
    for (j=0;j<i;j++) alphas[j] = PetscConj(-alphas[j]);
    CHKERRQ(VecMAXPY(vecs[i],i,alphas,vecs));
    CHKERRQ(VecNormalize(vecs[i],&norm));
    if (norm < PETSC_SMALL) {
      onorms[i] = 0.0;
      CHKERRQ(VecSet(vecs[i],0.0));
    } else {
      onorms[i] = norm;
    }
  }
  /* push nonzero vectors at the beginning */
  for (i=0;i<n;i++) {
    if (onorms[i] == 0.0) {
      for (j=i+1;j<n;j++) {
        if (onorms[j] != 0.0) {
          CHKERRQ(VecCopy(vecs[j],vecs[i]));
          onorms[j] = 0.0;
        }
      }
    }
  }
  for (i=0,*nio=0;i<n;i++) *nio += onorms[i] != 0.0 ? 1 : 0;
  CHKERRQ(PetscFree2(alphas,onorms));
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCMatISGetSubassemblingPattern(Mat mat, PetscInt *n_subdomains, PetscInt redprocs, IS* is_sends, PetscBool *have_void)
{
  ISLocalToGlobalMapping mapping;
  Mat                    A;
  PetscInt               n_neighs,*neighs,*n_shared,**shared;
  PetscMPIInt            size,rank,color;
  PetscInt               *xadj,*adjncy;
  PetscInt               *adjncy_wgt,*v_wgt,*ranks_send_to_idx;
  PetscInt               im_active,active_procs,N,n,i,j,threshold = 2;
  PetscInt               void_procs,*procs_candidates = NULL;
  PetscInt               xadj_count,*count;
  PetscBool              ismatis,use_vwgt=PETSC_FALSE;
  PetscSubcomm           psubcomm;
  MPI_Comm               subcomm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  CHKERRQ(PetscObjectTypeCompare((PetscObject)mat,MATIS,&ismatis));
  PetscCheckFalse(!ismatis,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Cannot use %s on a matrix object which is not of type MATIS",PETSC_FUNCTION_NAME);
  PetscValidLogicalCollectiveInt(mat,*n_subdomains,2);
  PetscValidLogicalCollectiveInt(mat,redprocs,3);
  PetscCheckFalse(*n_subdomains <=0,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONG,"Invalid number of subdomains requested %D",*n_subdomains);

  if (have_void) *have_void = PETSC_FALSE;
  CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)mat),&size));
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)mat),&rank));
  CHKERRQ(MatISGetLocalMat(mat,&A));
  CHKERRQ(MatGetLocalSize(A,&n,NULL));
  im_active = !!n;
  CHKERRMPI(MPIU_Allreduce(&im_active,&active_procs,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)mat)));
  void_procs = size - active_procs;
  /* get ranks of of non-active processes in mat communicator */
  if (void_procs) {
    PetscInt ncand;

    if (have_void) *have_void = PETSC_TRUE;
    CHKERRQ(PetscMalloc1(size,&procs_candidates));
    CHKERRMPI(MPI_Allgather(&im_active,1,MPIU_INT,procs_candidates,1,MPIU_INT,PetscObjectComm((PetscObject)mat)));
    for (i=0,ncand=0;i<size;i++) {
      if (!procs_candidates[i]) {
        procs_candidates[ncand++] = i;
      }
    }
    /* force n_subdomains to be not greater that the number of non-active processes */
    *n_subdomains = PetscMin(void_procs,*n_subdomains);
  }

  /* number of subdomains requested greater than active processes or matrix size -> just shift the matrix
     number of subdomains requested 1 -> send to rank-0 or first candidate in voids  */
  CHKERRQ(MatGetSize(mat,&N,NULL));
  if (active_procs < *n_subdomains || *n_subdomains == 1 || N <= *n_subdomains) {
    PetscInt issize,isidx,dest;
    if (*n_subdomains == 1) dest = 0;
    else dest = rank;
    if (im_active) {
      issize = 1;
      if (procs_candidates) { /* shift the pattern on non-active candidates (if any) */
        isidx = procs_candidates[dest];
      } else {
        isidx = dest;
      }
    } else {
      issize = 0;
      isidx = -1;
    }
    if (*n_subdomains != 1) *n_subdomains = active_procs;
    CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)mat),issize,&isidx,PETSC_COPY_VALUES,is_sends));
    CHKERRQ(PetscFree(procs_candidates));
    PetscFunctionReturn(0);
  }
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-matis_partitioning_use_vwgt",&use_vwgt,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-matis_partitioning_threshold",&threshold,NULL));
  threshold = PetscMax(threshold,2);

  /* Get info on mapping */
  CHKERRQ(MatISGetLocalToGlobalMapping(mat,&mapping,NULL));
  CHKERRQ(ISLocalToGlobalMappingGetInfo(mapping,&n_neighs,&neighs,&n_shared,&shared));

  /* build local CSR graph of subdomains' connectivity */
  CHKERRQ(PetscMalloc1(2,&xadj));
  xadj[0] = 0;
  xadj[1] = PetscMax(n_neighs-1,0);
  CHKERRQ(PetscMalloc1(xadj[1],&adjncy));
  CHKERRQ(PetscMalloc1(xadj[1],&adjncy_wgt));
  CHKERRQ(PetscCalloc1(n,&count));
  for (i=1;i<n_neighs;i++)
    for (j=0;j<n_shared[i];j++)
      count[shared[i][j]] += 1;

  xadj_count = 0;
  for (i=1;i<n_neighs;i++) {
    for (j=0;j<n_shared[i];j++) {
      if (count[shared[i][j]] < threshold) {
        adjncy[xadj_count] = neighs[i];
        adjncy_wgt[xadj_count] = n_shared[i];
        xadj_count++;
        break;
      }
    }
  }
  xadj[1] = xadj_count;
  CHKERRQ(PetscFree(count));
  CHKERRQ(ISLocalToGlobalMappingRestoreInfo(mapping,&n_neighs,&neighs,&n_shared,&shared));
  CHKERRQ(PetscSortIntWithArray(xadj[1],adjncy,adjncy_wgt));

  CHKERRQ(PetscMalloc1(1,&ranks_send_to_idx));

  /* Restrict work on active processes only */
  CHKERRQ(PetscMPIIntCast(im_active,&color));
  if (void_procs) {
    CHKERRQ(PetscSubcommCreate(PetscObjectComm((PetscObject)mat),&psubcomm));
    CHKERRQ(PetscSubcommSetNumber(psubcomm,2)); /* 2 groups, active process and not active processes */
    CHKERRQ(PetscSubcommSetTypeGeneral(psubcomm,color,rank));
    subcomm = PetscSubcommChild(psubcomm);
  } else {
    psubcomm = NULL;
    subcomm = PetscObjectComm((PetscObject)mat);
  }

  v_wgt = NULL;
  if (!color) {
    CHKERRQ(PetscFree(xadj));
    CHKERRQ(PetscFree(adjncy));
    CHKERRQ(PetscFree(adjncy_wgt));
  } else {
    Mat             subdomain_adj;
    IS              new_ranks,new_ranks_contig;
    MatPartitioning partitioner;
    PetscInt        rstart=0,rend=0;
    PetscInt        *is_indices,*oldranks;
    PetscMPIInt     size;
    PetscBool       aggregate;

    CHKERRMPI(MPI_Comm_size(subcomm,&size));
    if (void_procs) {
      PetscInt prank = rank;
      CHKERRQ(PetscMalloc1(size,&oldranks));
      CHKERRMPI(MPI_Allgather(&prank,1,MPIU_INT,oldranks,1,MPIU_INT,subcomm));
      for (i=0;i<xadj[1];i++) {
        CHKERRQ(PetscFindInt(adjncy[i],size,oldranks,&adjncy[i]));
      }
      CHKERRQ(PetscSortIntWithArray(xadj[1],adjncy,adjncy_wgt));
    } else {
      oldranks = NULL;
    }
    aggregate = ((redprocs > 0 && redprocs < size) ? PETSC_TRUE : PETSC_FALSE);
    if (aggregate) { /* TODO: all this part could be made more efficient */
      PetscInt    lrows,row,ncols,*cols;
      PetscMPIInt nrank;
      PetscScalar *vals;

      CHKERRMPI(MPI_Comm_rank(subcomm,&nrank));
      lrows = 0;
      if (nrank<redprocs) {
        lrows = size/redprocs;
        if (nrank<size%redprocs) lrows++;
      }
      CHKERRQ(MatCreateAIJ(subcomm,lrows,lrows,size,size,50,NULL,50,NULL,&subdomain_adj));
      CHKERRQ(MatGetOwnershipRange(subdomain_adj,&rstart,&rend));
      CHKERRQ(MatSetOption(subdomain_adj,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_FALSE));
      CHKERRQ(MatSetOption(subdomain_adj,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE));
      row = nrank;
      ncols = xadj[1]-xadj[0];
      cols = adjncy;
      CHKERRQ(PetscMalloc1(ncols,&vals));
      for (i=0;i<ncols;i++) vals[i] = adjncy_wgt[i];
      CHKERRQ(MatSetValues(subdomain_adj,1,&row,ncols,cols,vals,INSERT_VALUES));
      CHKERRQ(MatAssemblyBegin(subdomain_adj,MAT_FINAL_ASSEMBLY));
      CHKERRQ(MatAssemblyEnd(subdomain_adj,MAT_FINAL_ASSEMBLY));
      CHKERRQ(PetscFree(xadj));
      CHKERRQ(PetscFree(adjncy));
      CHKERRQ(PetscFree(adjncy_wgt));
      CHKERRQ(PetscFree(vals));
      if (use_vwgt) {
        Vec               v;
        const PetscScalar *array;
        PetscInt          nl;

        CHKERRQ(MatCreateVecs(subdomain_adj,&v,NULL));
        CHKERRQ(VecSetValue(v,row,(PetscScalar)n,INSERT_VALUES));
        CHKERRQ(VecAssemblyBegin(v));
        CHKERRQ(VecAssemblyEnd(v));
        CHKERRQ(VecGetLocalSize(v,&nl));
        CHKERRQ(VecGetArrayRead(v,&array));
        CHKERRQ(PetscMalloc1(nl,&v_wgt));
        for (i=0;i<nl;i++) v_wgt[i] = (PetscInt)PetscRealPart(array[i]);
        CHKERRQ(VecRestoreArrayRead(v,&array));
        CHKERRQ(VecDestroy(&v));
      }
    } else {
      CHKERRQ(MatCreateMPIAdj(subcomm,1,(PetscInt)size,xadj,adjncy,adjncy_wgt,&subdomain_adj));
      if (use_vwgt) {
        CHKERRQ(PetscMalloc1(1,&v_wgt));
        v_wgt[0] = n;
      }
    }
    /* CHKERRQ(MatView(subdomain_adj,0)); */

    /* Partition */
    CHKERRQ(MatPartitioningCreate(subcomm,&partitioner));
#if defined(PETSC_HAVE_PTSCOTCH)
    CHKERRQ(MatPartitioningSetType(partitioner,MATPARTITIONINGPTSCOTCH));
#elif defined(PETSC_HAVE_PARMETIS)
    CHKERRQ(MatPartitioningSetType(partitioner,MATPARTITIONINGPARMETIS));
#else
    CHKERRQ(MatPartitioningSetType(partitioner,MATPARTITIONINGAVERAGE));
#endif
    CHKERRQ(MatPartitioningSetAdjacency(partitioner,subdomain_adj));
    if (v_wgt) {
      CHKERRQ(MatPartitioningSetVertexWeights(partitioner,v_wgt));
    }
    *n_subdomains = PetscMin((PetscInt)size,*n_subdomains);
    CHKERRQ(MatPartitioningSetNParts(partitioner,*n_subdomains));
    CHKERRQ(MatPartitioningSetFromOptions(partitioner));
    CHKERRQ(MatPartitioningApply(partitioner,&new_ranks));
    /* CHKERRQ(MatPartitioningView(partitioner,0)); */

    /* renumber new_ranks to avoid "holes" in new set of processors */
    CHKERRQ(ISRenumber(new_ranks,NULL,NULL,&new_ranks_contig));
    CHKERRQ(ISDestroy(&new_ranks));
    CHKERRQ(ISGetIndices(new_ranks_contig,(const PetscInt**)&is_indices));
    if (!aggregate) {
      if (procs_candidates) { /* shift the pattern on non-active candidates (if any) */
        PetscAssert(oldranks,PETSC_COMM_SELF,PETSC_ERR_PLIB,"This should not happen");
        ranks_send_to_idx[0] = procs_candidates[oldranks[is_indices[0]]];
      } else if (oldranks) {
        ranks_send_to_idx[0] = oldranks[is_indices[0]];
      } else {
        ranks_send_to_idx[0] = is_indices[0];
      }
    } else {
      PetscInt    idx = 0;
      PetscMPIInt tag;
      MPI_Request *reqs;

      CHKERRQ(PetscObjectGetNewTag((PetscObject)subdomain_adj,&tag));
      CHKERRQ(PetscMalloc1(rend-rstart,&reqs));
      for (i=rstart;i<rend;i++) {
        CHKERRMPI(MPI_Isend(is_indices+i-rstart,1,MPIU_INT,i,tag,subcomm,&reqs[i-rstart]));
      }
      CHKERRMPI(MPI_Recv(&idx,1,MPIU_INT,MPI_ANY_SOURCE,tag,subcomm,MPI_STATUS_IGNORE));
      CHKERRMPI(MPI_Waitall(rend-rstart,reqs,MPI_STATUSES_IGNORE));
      CHKERRQ(PetscFree(reqs));
      if (procs_candidates) { /* shift the pattern on non-active candidates (if any) */
        PetscAssert(oldranks,PETSC_COMM_SELF,PETSC_ERR_PLIB,"This should not happen");
        ranks_send_to_idx[0] = procs_candidates[oldranks[idx]];
      } else if (oldranks) {
        ranks_send_to_idx[0] = oldranks[idx];
      } else {
        ranks_send_to_idx[0] = idx;
      }
    }
    CHKERRQ(ISRestoreIndices(new_ranks_contig,(const PetscInt**)&is_indices));
    /* clean up */
    CHKERRQ(PetscFree(oldranks));
    CHKERRQ(ISDestroy(&new_ranks_contig));
    CHKERRQ(MatDestroy(&subdomain_adj));
    CHKERRQ(MatPartitioningDestroy(&partitioner));
  }
  CHKERRQ(PetscSubcommDestroy(&psubcomm));
  CHKERRQ(PetscFree(procs_candidates));

  /* assemble parallel IS for sends */
  i = 1;
  if (!color) i=0;
  CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)mat),i,ranks_send_to_idx,PETSC_OWN_POINTER,is_sends));
  PetscFunctionReturn(0);
}

typedef enum {MATDENSE_PRIVATE=0,MATAIJ_PRIVATE,MATBAIJ_PRIVATE,MATSBAIJ_PRIVATE}MatTypePrivate;

PetscErrorCode PCBDDCMatISSubassemble(Mat mat, IS is_sends, PetscInt n_subdomains, PetscBool restrict_comm, PetscBool restrict_full, PetscBool reuse, Mat *mat_n, PetscInt nis, IS isarray[], PetscInt nvecs, Vec nnsp_vec[])
{
  Mat                    local_mat;
  IS                     is_sends_internal;
  PetscInt               rows,cols,new_local_rows;
  PetscInt               i,bs,buf_size_idxs,buf_size_idxs_is,buf_size_vals,buf_size_vecs;
  PetscBool              ismatis,isdense,newisdense,destroy_mat;
  ISLocalToGlobalMapping l2gmap;
  PetscInt*              l2gmap_indices;
  const PetscInt*        is_indices;
  MatType                new_local_type;
  /* buffers */
  PetscInt               *ptr_idxs,*send_buffer_idxs,*recv_buffer_idxs;
  PetscInt               *ptr_idxs_is,*send_buffer_idxs_is,*recv_buffer_idxs_is;
  PetscInt               *recv_buffer_idxs_local;
  PetscScalar            *ptr_vals,*recv_buffer_vals;
  const PetscScalar      *send_buffer_vals;
  PetscScalar            *ptr_vecs,*send_buffer_vecs,*recv_buffer_vecs;
  /* MPI */
  MPI_Comm               comm,comm_n;
  PetscSubcomm           subcomm;
  PetscMPIInt            n_sends,n_recvs,size;
  PetscMPIInt            *iflags,*ilengths_idxs,*ilengths_vals,*ilengths_idxs_is;
  PetscMPIInt            *onodes,*onodes_is,*olengths_idxs,*olengths_idxs_is,*olengths_vals;
  PetscMPIInt            len,tag_idxs,tag_idxs_is,tag_vals,tag_vecs,source_dest;
  MPI_Request            *send_req_idxs,*send_req_idxs_is,*send_req_vals,*send_req_vecs;
  MPI_Request            *recv_req_idxs,*recv_req_idxs_is,*recv_req_vals,*recv_req_vecs;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_CLASSID,1);
  CHKERRQ(PetscObjectTypeCompare((PetscObject)mat,MATIS,&ismatis));
  PetscCheck(ismatis,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Cannot use %s on a matrix object which is not of type MATIS",PETSC_FUNCTION_NAME);
  PetscValidLogicalCollectiveInt(mat,n_subdomains,3);
  PetscValidLogicalCollectiveBool(mat,restrict_comm,4);
  PetscValidLogicalCollectiveBool(mat,restrict_full,5);
  PetscValidLogicalCollectiveBool(mat,reuse,6);
  PetscValidLogicalCollectiveInt(mat,nis,8);
  PetscValidLogicalCollectiveInt(mat,nvecs,10);
  if (nvecs) {
    PetscCheckFalse(nvecs > 1,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Just 1 vector supported");
    PetscValidHeaderSpecific(nnsp_vec[0],VEC_CLASSID,11);
  }
  /* further checks */
  CHKERRQ(MatISGetLocalMat(mat,&local_mat));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)local_mat,MATSEQDENSE,&isdense));
  PetscCheck(isdense,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Currently cannot subassemble MATIS when local matrix type is not of type SEQDENSE");
  CHKERRQ(MatGetSize(local_mat,&rows,&cols));
  PetscCheck(rows == cols,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Local MATIS matrices should be square");
  if (reuse && *mat_n) {
    PetscInt mrows,mcols,mnrows,mncols;
    PetscValidHeaderSpecific(*mat_n,MAT_CLASSID,7);
    CHKERRQ(PetscObjectTypeCompare((PetscObject)*mat_n,MATIS,&ismatis));
    PetscCheck(ismatis,PetscObjectComm((PetscObject)*mat_n),PETSC_ERR_SUP,"Cannot reuse a matrix which is not of type MATIS");
    CHKERRQ(MatGetSize(mat,&mrows,&mcols));
    CHKERRQ(MatGetSize(*mat_n,&mnrows,&mncols));
    PetscCheck(mrows == mnrows,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Cannot reuse matrix! Wrong number of rows %D != %D",mrows,mnrows);
    PetscCheck(mcols == mncols,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Cannot reuse matrix! Wrong number of cols %D != %D",mcols,mncols);
  }
  CHKERRQ(MatGetBlockSize(local_mat,&bs));
  PetscValidLogicalCollectiveInt(mat,bs,1);

  /* prepare IS for sending if not provided */
  if (!is_sends) {
    PetscCheck(n_subdomains,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"You should specify either an IS or a target number of subdomains");
    CHKERRQ(PCBDDCMatISGetSubassemblingPattern(mat,&n_subdomains,0,&is_sends_internal,NULL));
  } else {
    CHKERRQ(PetscObjectReference((PetscObject)is_sends));
    is_sends_internal = is_sends;
  }

  /* get comm */
  CHKERRQ(PetscObjectGetComm((PetscObject)mat,&comm));

  /* compute number of sends */
  CHKERRQ(ISGetLocalSize(is_sends_internal,&i));
  CHKERRQ(PetscMPIIntCast(i,&n_sends));

  /* compute number of receives */
  CHKERRMPI(MPI_Comm_size(comm,&size));
  CHKERRQ(PetscMalloc1(size,&iflags));
  CHKERRQ(PetscArrayzero(iflags,size));
  CHKERRQ(ISGetIndices(is_sends_internal,&is_indices));
  for (i=0;i<n_sends;i++) iflags[is_indices[i]] = 1;
  CHKERRQ(PetscGatherNumberOfMessages(comm,iflags,NULL,&n_recvs));
  CHKERRQ(PetscFree(iflags));

  /* restrict comm if requested */
  subcomm = NULL;
  destroy_mat = PETSC_FALSE;
  if (restrict_comm) {
    PetscMPIInt color,subcommsize;

    color = 0;
    if (restrict_full) {
      if (!n_recvs) color = 1; /* processes not receiving anything will not partecipate in new comm (full restriction) */
    } else {
      if (!n_recvs && n_sends) color = 1; /* just those processes that are sending but not receiving anything will not partecipate in new comm */
    }
    CHKERRMPI(MPIU_Allreduce(&color,&subcommsize,1,MPI_INT,MPI_SUM,comm));
    subcommsize = size - subcommsize;
    /* check if reuse has been requested */
    if (reuse) {
      if (*mat_n) {
        PetscMPIInt subcommsize2;
        CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)*mat_n),&subcommsize2));
        PetscCheck(subcommsize == subcommsize2,PetscObjectComm((PetscObject)*mat_n),PETSC_ERR_PLIB,"Cannot reuse matrix! wrong subcomm size %d != %d",subcommsize,subcommsize2);
        comm_n = PetscObjectComm((PetscObject)*mat_n);
      } else {
        comm_n = PETSC_COMM_SELF;
      }
    } else { /* MAT_INITIAL_MATRIX */
      PetscMPIInt rank;

      CHKERRMPI(MPI_Comm_rank(comm,&rank));
      CHKERRQ(PetscSubcommCreate(comm,&subcomm));
      CHKERRQ(PetscSubcommSetNumber(subcomm,2));
      CHKERRQ(PetscSubcommSetTypeGeneral(subcomm,color,rank));
      comm_n = PetscSubcommChild(subcomm);
    }
    /* flag to destroy *mat_n if not significative */
    if (color) destroy_mat = PETSC_TRUE;
  } else {
    comm_n = comm;
  }

  /* prepare send/receive buffers */
  CHKERRQ(PetscMalloc1(size,&ilengths_idxs));
  CHKERRQ(PetscArrayzero(ilengths_idxs,size));
  CHKERRQ(PetscMalloc1(size,&ilengths_vals));
  CHKERRQ(PetscArrayzero(ilengths_vals,size));
  if (nis) {
    CHKERRQ(PetscCalloc1(size,&ilengths_idxs_is));
  }

  /* Get data from local matrices */
  PetscCheck(isdense,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Subassembling of AIJ local matrices not yet implemented");
    /* TODO: See below some guidelines on how to prepare the local buffers */
    /*
       send_buffer_vals should contain the raw values of the local matrix
       send_buffer_idxs should contain:
       - MatType_PRIVATE type
       - PetscInt        size_of_l2gmap
       - PetscInt        global_row_indices[size_of_l2gmap]
       - PetscInt        all_other_info_which_is_needed_to_compute_preallocation_and_set_values
    */
  {
    ISLocalToGlobalMapping mapping;

    CHKERRQ(MatISGetLocalToGlobalMapping(mat,&mapping,NULL));
    CHKERRQ(MatDenseGetArrayRead(local_mat,&send_buffer_vals));
    CHKERRQ(ISLocalToGlobalMappingGetSize(mapping,&i));
    CHKERRQ(PetscMalloc1(i+2,&send_buffer_idxs));
    send_buffer_idxs[0] = (PetscInt)MATDENSE_PRIVATE;
    send_buffer_idxs[1] = i;
    CHKERRQ(ISLocalToGlobalMappingGetIndices(mapping,(const PetscInt**)&ptr_idxs));
    CHKERRQ(PetscArraycpy(&send_buffer_idxs[2],ptr_idxs,i));
    CHKERRQ(ISLocalToGlobalMappingRestoreIndices(mapping,(const PetscInt**)&ptr_idxs));
    CHKERRQ(PetscMPIIntCast(i,&len));
    for (i=0;i<n_sends;i++) {
      ilengths_vals[is_indices[i]] = len*len;
      ilengths_idxs[is_indices[i]] = len+2;
    }
  }
  CHKERRQ(PetscGatherMessageLengths2(comm,n_sends,n_recvs,ilengths_idxs,ilengths_vals,&onodes,&olengths_idxs,&olengths_vals));
  /* additional is (if any) */
  if (nis) {
    PetscMPIInt psum;
    PetscInt j;
    for (j=0,psum=0;j<nis;j++) {
      PetscInt plen;
      CHKERRQ(ISGetLocalSize(isarray[j],&plen));
      CHKERRQ(PetscMPIIntCast(plen,&len));
      psum += len+1; /* indices + lenght */
    }
    CHKERRQ(PetscMalloc1(psum,&send_buffer_idxs_is));
    for (j=0,psum=0;j<nis;j++) {
      PetscInt plen;
      const PetscInt *is_array_idxs;
      CHKERRQ(ISGetLocalSize(isarray[j],&plen));
      send_buffer_idxs_is[psum] = plen;
      CHKERRQ(ISGetIndices(isarray[j],&is_array_idxs));
      CHKERRQ(PetscArraycpy(&send_buffer_idxs_is[psum+1],is_array_idxs,plen));
      CHKERRQ(ISRestoreIndices(isarray[j],&is_array_idxs));
      psum += plen+1; /* indices + lenght */
    }
    for (i=0;i<n_sends;i++) {
      ilengths_idxs_is[is_indices[i]] = psum;
    }
    CHKERRQ(PetscGatherMessageLengths(comm,n_sends,n_recvs,ilengths_idxs_is,&onodes_is,&olengths_idxs_is));
  }
  CHKERRQ(MatISRestoreLocalMat(mat,&local_mat));

  buf_size_idxs = 0;
  buf_size_vals = 0;
  buf_size_idxs_is = 0;
  buf_size_vecs = 0;
  for (i=0;i<n_recvs;i++) {
    buf_size_idxs += (PetscInt)olengths_idxs[i];
    buf_size_vals += (PetscInt)olengths_vals[i];
    if (nis) buf_size_idxs_is += (PetscInt)olengths_idxs_is[i];
    if (nvecs) buf_size_vecs += (PetscInt)olengths_idxs[i];
  }
  CHKERRQ(PetscMalloc1(buf_size_idxs,&recv_buffer_idxs));
  CHKERRQ(PetscMalloc1(buf_size_vals,&recv_buffer_vals));
  CHKERRQ(PetscMalloc1(buf_size_idxs_is,&recv_buffer_idxs_is));
  CHKERRQ(PetscMalloc1(buf_size_vecs,&recv_buffer_vecs));

  /* get new tags for clean communications */
  CHKERRQ(PetscObjectGetNewTag((PetscObject)mat,&tag_idxs));
  CHKERRQ(PetscObjectGetNewTag((PetscObject)mat,&tag_vals));
  CHKERRQ(PetscObjectGetNewTag((PetscObject)mat,&tag_idxs_is));
  CHKERRQ(PetscObjectGetNewTag((PetscObject)mat,&tag_vecs));

  /* allocate for requests */
  CHKERRQ(PetscMalloc1(n_sends,&send_req_idxs));
  CHKERRQ(PetscMalloc1(n_sends,&send_req_vals));
  CHKERRQ(PetscMalloc1(n_sends,&send_req_idxs_is));
  CHKERRQ(PetscMalloc1(n_sends,&send_req_vecs));
  CHKERRQ(PetscMalloc1(n_recvs,&recv_req_idxs));
  CHKERRQ(PetscMalloc1(n_recvs,&recv_req_vals));
  CHKERRQ(PetscMalloc1(n_recvs,&recv_req_idxs_is));
  CHKERRQ(PetscMalloc1(n_recvs,&recv_req_vecs));

  /* communications */
  ptr_idxs = recv_buffer_idxs;
  ptr_vals = recv_buffer_vals;
  ptr_idxs_is = recv_buffer_idxs_is;
  ptr_vecs = recv_buffer_vecs;
  for (i=0;i<n_recvs;i++) {
    source_dest = onodes[i];
    CHKERRMPI(MPI_Irecv(ptr_idxs,olengths_idxs[i],MPIU_INT,source_dest,tag_idxs,comm,&recv_req_idxs[i]));
    CHKERRMPI(MPI_Irecv(ptr_vals,olengths_vals[i],MPIU_SCALAR,source_dest,tag_vals,comm,&recv_req_vals[i]));
    ptr_idxs += olengths_idxs[i];
    ptr_vals += olengths_vals[i];
    if (nis) {
      source_dest = onodes_is[i];
      CHKERRMPI(MPI_Irecv(ptr_idxs_is,olengths_idxs_is[i],MPIU_INT,source_dest,tag_idxs_is,comm,&recv_req_idxs_is[i]));
      ptr_idxs_is += olengths_idxs_is[i];
    }
    if (nvecs) {
      source_dest = onodes[i];
      CHKERRMPI(MPI_Irecv(ptr_vecs,olengths_idxs[i]-2,MPIU_SCALAR,source_dest,tag_vecs,comm,&recv_req_vecs[i]));
      ptr_vecs += olengths_idxs[i]-2;
    }
  }
  for (i=0;i<n_sends;i++) {
    CHKERRQ(PetscMPIIntCast(is_indices[i],&source_dest));
    CHKERRMPI(MPI_Isend(send_buffer_idxs,ilengths_idxs[source_dest],MPIU_INT,source_dest,tag_idxs,comm,&send_req_idxs[i]));
    CHKERRMPI(MPI_Isend((PetscScalar*)send_buffer_vals,ilengths_vals[source_dest],MPIU_SCALAR,source_dest,tag_vals,comm,&send_req_vals[i]));
    if (nis) {
      CHKERRMPI(MPI_Isend(send_buffer_idxs_is,ilengths_idxs_is[source_dest],MPIU_INT,source_dest,tag_idxs_is,comm,&send_req_idxs_is[i]));
    }
    if (nvecs) {
      CHKERRQ(VecGetArray(nnsp_vec[0],&send_buffer_vecs));
      CHKERRMPI(MPI_Isend(send_buffer_vecs,ilengths_idxs[source_dest]-2,MPIU_SCALAR,source_dest,tag_vecs,comm,&send_req_vecs[i]));
    }
  }
  CHKERRQ(ISRestoreIndices(is_sends_internal,&is_indices));
  CHKERRQ(ISDestroy(&is_sends_internal));

  /* assemble new l2g map */
  CHKERRMPI(MPI_Waitall(n_recvs,recv_req_idxs,MPI_STATUSES_IGNORE));
  ptr_idxs = recv_buffer_idxs;
  new_local_rows = 0;
  for (i=0;i<n_recvs;i++) {
    new_local_rows += *(ptr_idxs+1); /* second element is the local size of the l2gmap */
    ptr_idxs += olengths_idxs[i];
  }
  CHKERRQ(PetscMalloc1(new_local_rows,&l2gmap_indices));
  ptr_idxs = recv_buffer_idxs;
  new_local_rows = 0;
  for (i=0;i<n_recvs;i++) {
    CHKERRQ(PetscArraycpy(&l2gmap_indices[new_local_rows],ptr_idxs+2,*(ptr_idxs+1)));
    new_local_rows += *(ptr_idxs+1); /* second element is the local size of the l2gmap */
    ptr_idxs += olengths_idxs[i];
  }
  CHKERRQ(PetscSortRemoveDupsInt(&new_local_rows,l2gmap_indices));
  CHKERRQ(ISLocalToGlobalMappingCreate(comm_n,1,new_local_rows,l2gmap_indices,PETSC_COPY_VALUES,&l2gmap));
  CHKERRQ(PetscFree(l2gmap_indices));

  /* infer new local matrix type from received local matrices type */
  /* currently if all local matrices are of type X, then the resulting matrix will be of type X, except for the dense case */
  /* it also assumes that if the block size is set, than it is the same among all local matrices (see checks at the beginning of the function) */
  if (n_recvs) {
    MatTypePrivate new_local_type_private = (MatTypePrivate)send_buffer_idxs[0];
    ptr_idxs = recv_buffer_idxs;
    for (i=0;i<n_recvs;i++) {
      if ((PetscInt)new_local_type_private != *ptr_idxs) {
        new_local_type_private = MATAIJ_PRIVATE;
        break;
      }
      ptr_idxs += olengths_idxs[i];
    }
    switch (new_local_type_private) {
      case MATDENSE_PRIVATE:
        new_local_type = MATSEQAIJ;
        bs = 1;
        break;
      case MATAIJ_PRIVATE:
        new_local_type = MATSEQAIJ;
        bs = 1;
        break;
      case MATBAIJ_PRIVATE:
        new_local_type = MATSEQBAIJ;
        break;
      case MATSBAIJ_PRIVATE:
        new_local_type = MATSEQSBAIJ;
        break;
      default:
        SETERRQ(comm,PETSC_ERR_SUP,"Unsupported private type %d in %s",new_local_type_private,PETSC_FUNCTION_NAME);
    }
  } else { /* by default, new_local_type is seqaij */
    new_local_type = MATSEQAIJ;
    bs = 1;
  }

  /* create MATIS object if needed */
  if (!reuse) {
    CHKERRQ(MatGetSize(mat,&rows,&cols));
    CHKERRQ(MatCreateIS(comm_n,bs,PETSC_DECIDE,PETSC_DECIDE,rows,cols,l2gmap,l2gmap,mat_n));
  } else {
    /* it also destroys the local matrices */
    if (*mat_n) {
      CHKERRQ(MatSetLocalToGlobalMapping(*mat_n,l2gmap,l2gmap));
    } else { /* this is a fake object */
      CHKERRQ(MatCreateIS(comm_n,bs,PETSC_DECIDE,PETSC_DECIDE,rows,cols,l2gmap,l2gmap,mat_n));
    }
  }
  CHKERRQ(MatISGetLocalMat(*mat_n,&local_mat));
  CHKERRQ(MatSetType(local_mat,new_local_type));

  CHKERRMPI(MPI_Waitall(n_recvs,recv_req_vals,MPI_STATUSES_IGNORE));

  /* Global to local map of received indices */
  CHKERRQ(PetscMalloc1(buf_size_idxs,&recv_buffer_idxs_local)); /* needed for values insertion */
  CHKERRQ(ISGlobalToLocalMappingApply(l2gmap,IS_GTOLM_MASK,buf_size_idxs,recv_buffer_idxs,&i,recv_buffer_idxs_local));
  CHKERRQ(ISLocalToGlobalMappingDestroy(&l2gmap));

  /* restore attributes -> type of incoming data and its size */
  buf_size_idxs = 0;
  for (i=0;i<n_recvs;i++) {
    recv_buffer_idxs_local[buf_size_idxs] = recv_buffer_idxs[buf_size_idxs];
    recv_buffer_idxs_local[buf_size_idxs+1] = recv_buffer_idxs[buf_size_idxs+1];
    buf_size_idxs += (PetscInt)olengths_idxs[i];
  }
  CHKERRQ(PetscFree(recv_buffer_idxs));

  /* set preallocation */
  CHKERRQ(PetscObjectTypeCompare((PetscObject)local_mat,MATSEQDENSE,&newisdense));
  if (!newisdense) {
    PetscInt *new_local_nnz=NULL;

    ptr_idxs = recv_buffer_idxs_local;
    if (n_recvs) {
      CHKERRQ(PetscCalloc1(new_local_rows,&new_local_nnz));
    }
    for (i=0;i<n_recvs;i++) {
      PetscInt j;
      if (*ptr_idxs == (PetscInt)MATDENSE_PRIVATE) { /* preallocation provided for dense case only */
        for (j=0;j<*(ptr_idxs+1);j++) {
          new_local_nnz[*(ptr_idxs+2+j)] += *(ptr_idxs+1);
        }
      } else {
        /* TODO */
      }
      ptr_idxs += olengths_idxs[i];
    }
    if (new_local_nnz) {
      for (i=0;i<new_local_rows;i++) new_local_nnz[i] = PetscMin(new_local_nnz[i],new_local_rows);
      CHKERRQ(MatSeqAIJSetPreallocation(local_mat,0,new_local_nnz));
      for (i=0;i<new_local_rows;i++) new_local_nnz[i] /= bs;
      CHKERRQ(MatSeqBAIJSetPreallocation(local_mat,bs,0,new_local_nnz));
      for (i=0;i<new_local_rows;i++) new_local_nnz[i] = PetscMax(new_local_nnz[i]-i,0);
      CHKERRQ(MatSeqSBAIJSetPreallocation(local_mat,bs,0,new_local_nnz));
    } else {
      CHKERRQ(MatSetUp(local_mat));
    }
    CHKERRQ(PetscFree(new_local_nnz));
  } else {
    CHKERRQ(MatSetUp(local_mat));
  }

  /* set values */
  ptr_vals = recv_buffer_vals;
  ptr_idxs = recv_buffer_idxs_local;
  for (i=0;i<n_recvs;i++) {
    if (*ptr_idxs == (PetscInt)MATDENSE_PRIVATE) { /* values insertion provided for dense case only */
      CHKERRQ(MatSetOption(local_mat,MAT_ROW_ORIENTED,PETSC_FALSE));
      CHKERRQ(MatSetValues(local_mat,*(ptr_idxs+1),ptr_idxs+2,*(ptr_idxs+1),ptr_idxs+2,ptr_vals,ADD_VALUES));
      CHKERRQ(MatAssemblyBegin(local_mat,MAT_FLUSH_ASSEMBLY));
      CHKERRQ(MatAssemblyEnd(local_mat,MAT_FLUSH_ASSEMBLY));
      CHKERRQ(MatSetOption(local_mat,MAT_ROW_ORIENTED,PETSC_TRUE));
    } else {
      /* TODO */
    }
    ptr_idxs += olengths_idxs[i];
    ptr_vals += olengths_vals[i];
  }
  CHKERRQ(MatAssemblyBegin(local_mat,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(local_mat,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatISRestoreLocalMat(*mat_n,&local_mat));
  CHKERRQ(MatAssemblyBegin(*mat_n,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(*mat_n,MAT_FINAL_ASSEMBLY));
  CHKERRQ(PetscFree(recv_buffer_vals));

#if 0
  if (!restrict_comm) { /* check */
    Vec       lvec,rvec;
    PetscReal infty_error;

    CHKERRQ(MatCreateVecs(mat,&rvec,&lvec));
    CHKERRQ(VecSetRandom(rvec,NULL));
    CHKERRQ(MatMult(mat,rvec,lvec));
    CHKERRQ(VecScale(lvec,-1.0));
    CHKERRQ(MatMultAdd(*mat_n,rvec,lvec,lvec));
    CHKERRQ(VecNorm(lvec,NORM_INFINITY,&infty_error));
    CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject)mat),"Infinity error subassembling %1.6e\n",infty_error));
    CHKERRQ(VecDestroy(&rvec));
    CHKERRQ(VecDestroy(&lvec));
  }
#endif

  /* assemble new additional is (if any) */
  if (nis) {
    PetscInt **temp_idxs,*count_is,j,psum;

    CHKERRMPI(MPI_Waitall(n_recvs,recv_req_idxs_is,MPI_STATUSES_IGNORE));
    CHKERRQ(PetscCalloc1(nis,&count_is));
    ptr_idxs = recv_buffer_idxs_is;
    psum = 0;
    for (i=0;i<n_recvs;i++) {
      for (j=0;j<nis;j++) {
        PetscInt plen = *(ptr_idxs); /* first element is the local size of IS's indices */
        count_is[j] += plen; /* increment counting of buffer for j-th IS */
        psum += plen;
        ptr_idxs += plen+1; /* shift pointer to received data */
      }
    }
    CHKERRQ(PetscMalloc1(nis,&temp_idxs));
    CHKERRQ(PetscMalloc1(psum,&temp_idxs[0]));
    for (i=1;i<nis;i++) {
      temp_idxs[i] = temp_idxs[i-1]+count_is[i-1];
    }
    CHKERRQ(PetscArrayzero(count_is,nis));
    ptr_idxs = recv_buffer_idxs_is;
    for (i=0;i<n_recvs;i++) {
      for (j=0;j<nis;j++) {
        PetscInt plen = *(ptr_idxs); /* first element is the local size of IS's indices */
        CHKERRQ(PetscArraycpy(&temp_idxs[j][count_is[j]],ptr_idxs+1,plen));
        count_is[j] += plen; /* increment starting point of buffer for j-th IS */
        ptr_idxs += plen+1; /* shift pointer to received data */
      }
    }
    for (i=0;i<nis;i++) {
      CHKERRQ(ISDestroy(&isarray[i]));
      CHKERRQ(PetscSortRemoveDupsInt(&count_is[i],temp_idxs[i]));
      CHKERRQ(ISCreateGeneral(comm_n,count_is[i],temp_idxs[i],PETSC_COPY_VALUES,&isarray[i]));
    }
    CHKERRQ(PetscFree(count_is));
    CHKERRQ(PetscFree(temp_idxs[0]));
    CHKERRQ(PetscFree(temp_idxs));
  }
  /* free workspace */
  CHKERRQ(PetscFree(recv_buffer_idxs_is));
  CHKERRMPI(MPI_Waitall(n_sends,send_req_idxs,MPI_STATUSES_IGNORE));
  CHKERRQ(PetscFree(send_buffer_idxs));
  CHKERRMPI(MPI_Waitall(n_sends,send_req_vals,MPI_STATUSES_IGNORE));
  if (isdense) {
    CHKERRQ(MatISGetLocalMat(mat,&local_mat));
    CHKERRQ(MatDenseRestoreArrayRead(local_mat,&send_buffer_vals));
    CHKERRQ(MatISRestoreLocalMat(mat,&local_mat));
  } else {
    /* CHKERRQ(PetscFree(send_buffer_vals)); */
  }
  if (nis) {
    CHKERRMPI(MPI_Waitall(n_sends,send_req_idxs_is,MPI_STATUSES_IGNORE));
    CHKERRQ(PetscFree(send_buffer_idxs_is));
  }

  if (nvecs) {
    CHKERRMPI(MPI_Waitall(n_recvs,recv_req_vecs,MPI_STATUSES_IGNORE));
    CHKERRMPI(MPI_Waitall(n_sends,send_req_vecs,MPI_STATUSES_IGNORE));
    CHKERRQ(VecRestoreArray(nnsp_vec[0],&send_buffer_vecs));
    CHKERRQ(VecDestroy(&nnsp_vec[0]));
    CHKERRQ(VecCreate(comm_n,&nnsp_vec[0]));
    CHKERRQ(VecSetSizes(nnsp_vec[0],new_local_rows,PETSC_DECIDE));
    CHKERRQ(VecSetType(nnsp_vec[0],VECSTANDARD));
    /* set values */
    ptr_vals = recv_buffer_vecs;
    ptr_idxs = recv_buffer_idxs_local;
    CHKERRQ(VecGetArray(nnsp_vec[0],&send_buffer_vecs));
    for (i=0;i<n_recvs;i++) {
      PetscInt j;
      for (j=0;j<*(ptr_idxs+1);j++) {
        send_buffer_vecs[*(ptr_idxs+2+j)] += *(ptr_vals + j);
      }
      ptr_idxs += olengths_idxs[i];
      ptr_vals += olengths_idxs[i]-2;
    }
    CHKERRQ(VecRestoreArray(nnsp_vec[0],&send_buffer_vecs));
    CHKERRQ(VecAssemblyBegin(nnsp_vec[0]));
    CHKERRQ(VecAssemblyEnd(nnsp_vec[0]));
  }

  CHKERRQ(PetscFree(recv_buffer_vecs));
  CHKERRQ(PetscFree(recv_buffer_idxs_local));
  CHKERRQ(PetscFree(recv_req_idxs));
  CHKERRQ(PetscFree(recv_req_vals));
  CHKERRQ(PetscFree(recv_req_vecs));
  CHKERRQ(PetscFree(recv_req_idxs_is));
  CHKERRQ(PetscFree(send_req_idxs));
  CHKERRQ(PetscFree(send_req_vals));
  CHKERRQ(PetscFree(send_req_vecs));
  CHKERRQ(PetscFree(send_req_idxs_is));
  CHKERRQ(PetscFree(ilengths_vals));
  CHKERRQ(PetscFree(ilengths_idxs));
  CHKERRQ(PetscFree(olengths_vals));
  CHKERRQ(PetscFree(olengths_idxs));
  CHKERRQ(PetscFree(onodes));
  if (nis) {
    CHKERRQ(PetscFree(ilengths_idxs_is));
    CHKERRQ(PetscFree(olengths_idxs_is));
    CHKERRQ(PetscFree(onodes_is));
  }
  CHKERRQ(PetscSubcommDestroy(&subcomm));
  if (destroy_mat) { /* destroy mat is true only if restrict comm is true and process will not partecipate */
    CHKERRQ(MatDestroy(mat_n));
    for (i=0;i<nis;i++) {
      CHKERRQ(ISDestroy(&isarray[i]));
    }
    if (nvecs) { /* need to match VecDestroy nnsp_vec called in the other code path */
      CHKERRQ(VecDestroy(&nnsp_vec[0]));
    }
    *mat_n = NULL;
  }
  PetscFunctionReturn(0);
}

/* temporary hack into ksp private data structure */
#include <petsc/private/kspimpl.h>

PetscErrorCode PCBDDCSetUpCoarseSolver(PC pc,PetscScalar* coarse_submat_vals)
{
  PC_BDDC                *pcbddc = (PC_BDDC*)pc->data;
  PC_IS                  *pcis = (PC_IS*)pc->data;
  Mat                    coarse_mat,coarse_mat_is,coarse_submat_dense;
  Mat                    coarsedivudotp = NULL;
  Mat                    coarseG,t_coarse_mat_is;
  MatNullSpace           CoarseNullSpace = NULL;
  ISLocalToGlobalMapping coarse_islg;
  IS                     coarse_is,*isarray,corners;
  PetscInt               i,im_active=-1,active_procs=-1;
  PetscInt               nis,nisdofs,nisneu,nisvert;
  PetscInt               coarse_eqs_per_proc;
  PC                     pc_temp;
  PCType                 coarse_pc_type;
  KSPType                coarse_ksp_type;
  PetscBool              multilevel_requested,multilevel_allowed;
  PetscBool              coarse_reuse;
  PetscInt               ncoarse,nedcfield;
  PetscBool              compute_vecs = PETSC_FALSE;
  PetscScalar            *array;
  MatReuse               coarse_mat_reuse;
  PetscBool              restr, full_restr, have_void;
  PetscMPIInt            size;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  CHKERRQ(PetscLogEventBegin(PC_BDDC_CoarseSetUp[pcbddc->current_level],pc,0,0,0));
  /* Assign global numbering to coarse dofs */
  if (pcbddc->new_primal_space || pcbddc->coarse_size == -1) { /* a new primal space is present or it is the first initialization, so recompute global numbering */
    PetscInt ocoarse_size;
    compute_vecs = PETSC_TRUE;

    pcbddc->new_primal_space = PETSC_TRUE;
    ocoarse_size = pcbddc->coarse_size;
    CHKERRQ(PetscFree(pcbddc->global_primal_indices));
    CHKERRQ(PCBDDCComputePrimalNumbering(pc,&pcbddc->coarse_size,&pcbddc->global_primal_indices));
    /* see if we can avoid some work */
    if (pcbddc->coarse_ksp) { /* coarse ksp has already been created */
      /* if the coarse size is different or we are using adaptive selection, better to not reuse the coarse matrix */
      if (ocoarse_size != pcbddc->coarse_size || pcbddc->adaptive_selection) {
        CHKERRQ(KSPReset(pcbddc->coarse_ksp));
        coarse_reuse = PETSC_FALSE;
      } else { /* we can safely reuse already computed coarse matrix */
        coarse_reuse = PETSC_TRUE;
      }
    } else { /* there's no coarse ksp, so we need to create the coarse matrix too */
      coarse_reuse = PETSC_FALSE;
    }
    /* reset any subassembling information */
    if (!coarse_reuse || pcbddc->recompute_topography) {
      CHKERRQ(ISDestroy(&pcbddc->coarse_subassembling));
    }
  } else { /* primal space is unchanged, so we can reuse coarse matrix */
    coarse_reuse = PETSC_TRUE;
  }
  if (coarse_reuse && pcbddc->coarse_ksp) {
    CHKERRQ(KSPGetOperators(pcbddc->coarse_ksp,&coarse_mat,NULL));
    CHKERRQ(PetscObjectReference((PetscObject)coarse_mat));
    coarse_mat_reuse = MAT_REUSE_MATRIX;
  } else {
    coarse_mat = NULL;
    coarse_mat_reuse = MAT_INITIAL_MATRIX;
  }

  /* creates temporary l2gmap and IS for coarse indexes */
  CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)pc),pcbddc->local_primal_size,pcbddc->global_primal_indices,PETSC_COPY_VALUES,&coarse_is));
  CHKERRQ(ISLocalToGlobalMappingCreateIS(coarse_is,&coarse_islg));

  /* creates temporary MATIS object for coarse matrix */
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,pcbddc->local_primal_size,pcbddc->local_primal_size,coarse_submat_vals,&coarse_submat_dense));
  CHKERRQ(MatCreateIS(PetscObjectComm((PetscObject)pc),1,PETSC_DECIDE,PETSC_DECIDE,pcbddc->coarse_size,pcbddc->coarse_size,coarse_islg,coarse_islg,&t_coarse_mat_is));
  CHKERRQ(MatISSetLocalMat(t_coarse_mat_is,coarse_submat_dense));
  CHKERRQ(MatAssemblyBegin(t_coarse_mat_is,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(t_coarse_mat_is,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatDestroy(&coarse_submat_dense));

  /* count "active" (i.e. with positive local size) and "void" processes */
  im_active = !!(pcis->n);
  CHKERRMPI(MPIU_Allreduce(&im_active,&active_procs,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)pc)));

  /* determine number of processes partecipating to coarse solver and compute subassembling pattern */
  /* restr : whether we want to exclude senders (which are not receivers) from the subassembling pattern */
  /* full_restr : just use the receivers from the subassembling pattern */
  CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)pc),&size));
  coarse_mat_is        = NULL;
  multilevel_allowed   = PETSC_FALSE;
  multilevel_requested = PETSC_FALSE;
  coarse_eqs_per_proc  = PetscMin(PetscMax(pcbddc->coarse_size,1),pcbddc->coarse_eqs_per_proc);
  if (coarse_eqs_per_proc < 0) coarse_eqs_per_proc = pcbddc->coarse_size;
  if (pcbddc->current_level < pcbddc->max_levels) multilevel_requested = PETSC_TRUE;
  if (pcbddc->coarse_size <= pcbddc->coarse_eqs_limit) multilevel_requested = PETSC_FALSE;
  if (multilevel_requested) {
    ncoarse    = active_procs/pcbddc->coarsening_ratio;
    restr      = PETSC_FALSE;
    full_restr = PETSC_FALSE;
  } else {
    ncoarse    = pcbddc->coarse_size/coarse_eqs_per_proc + !!(pcbddc->coarse_size%coarse_eqs_per_proc);
    restr      = PETSC_TRUE;
    full_restr = PETSC_TRUE;
  }
  if (!pcbddc->coarse_size || size == 1) multilevel_allowed = multilevel_requested = restr = full_restr = PETSC_FALSE;
  ncoarse = PetscMax(1,ncoarse);
  if (!pcbddc->coarse_subassembling) {
    if (pcbddc->coarsening_ratio > 1) {
      if (multilevel_requested) {
        CHKERRQ(PCBDDCMatISGetSubassemblingPattern(pc->pmat,&ncoarse,pcbddc->coarse_adj_red,&pcbddc->coarse_subassembling,&have_void));
      } else {
        CHKERRQ(PCBDDCMatISGetSubassemblingPattern(t_coarse_mat_is,&ncoarse,pcbddc->coarse_adj_red,&pcbddc->coarse_subassembling,&have_void));
      }
    } else {
      PetscMPIInt rank;

      CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)pc),&rank));
      have_void = (active_procs == (PetscInt)size) ? PETSC_FALSE : PETSC_TRUE;
      CHKERRQ(ISCreateStride(PetscObjectComm((PetscObject)pc),1,rank,1,&pcbddc->coarse_subassembling));
    }
  } else { /* if a subassembling pattern exists, then we can reuse the coarse ksp and compute the number of process involved */
    PetscInt    psum;
    if (pcbddc->coarse_ksp) psum = 1;
    else psum = 0;
    CHKERRMPI(MPIU_Allreduce(&psum,&ncoarse,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)pc)));
    have_void = ncoarse < size ? PETSC_TRUE : PETSC_FALSE;
  }
  /* determine if we can go multilevel */
  if (multilevel_requested) {
    if (ncoarse > 1) multilevel_allowed = PETSC_TRUE; /* found enough processes */
    else restr = full_restr = PETSC_TRUE; /* 1 subdomain, use a direct solver */
  }
  if (multilevel_allowed && have_void) restr = PETSC_TRUE;

  /* dump subassembling pattern */
  if (pcbddc->dbg_flag && multilevel_allowed) {
    CHKERRQ(ISView(pcbddc->coarse_subassembling,pcbddc->dbg_viewer));
  }
  /* compute dofs splitting and neumann boundaries for coarse dofs */
  nedcfield = -1;
  corners = NULL;
  if (multilevel_allowed && !coarse_reuse && (pcbddc->n_ISForDofsLocal || pcbddc->NeumannBoundariesLocal || pcbddc->nedclocal || pcbddc->corner_selected)) { /* protects from unneeded computations */
    PetscInt               *tidxs,*tidxs2,nout,tsize,i;
    const PetscInt         *idxs;
    ISLocalToGlobalMapping tmap;

    /* create map between primal indices (in local representative ordering) and local primal numbering */
    CHKERRQ(ISLocalToGlobalMappingCreate(PETSC_COMM_SELF,1,pcbddc->local_primal_size,pcbddc->primal_indices_local_idxs,PETSC_COPY_VALUES,&tmap));
    /* allocate space for temporary storage */
    CHKERRQ(PetscMalloc1(pcbddc->local_primal_size,&tidxs));
    CHKERRQ(PetscMalloc1(pcbddc->local_primal_size,&tidxs2));
    /* allocate for IS array */
    nisdofs = pcbddc->n_ISForDofsLocal;
    if (pcbddc->nedclocal) {
      if (pcbddc->nedfield > -1) {
        nedcfield = pcbddc->nedfield;
      } else {
        nedcfield = 0;
        PetscCheckFalse(nisdofs,PetscObjectComm((PetscObject)pc),PETSC_ERR_PLIB,"This should not happen (%D)",nisdofs);
        nisdofs = 1;
      }
    }
    nisneu = !!pcbddc->NeumannBoundariesLocal;
    nisvert = 0; /* nisvert is not used */
    nis = nisdofs + nisneu + nisvert;
    CHKERRQ(PetscMalloc1(nis,&isarray));
    /* dofs splitting */
    for (i=0;i<nisdofs;i++) {
      /* CHKERRQ(ISView(pcbddc->ISForDofsLocal[i],0)); */
      if (nedcfield != i) {
        CHKERRQ(ISGetLocalSize(pcbddc->ISForDofsLocal[i],&tsize));
        CHKERRQ(ISGetIndices(pcbddc->ISForDofsLocal[i],&idxs));
        CHKERRQ(ISGlobalToLocalMappingApply(tmap,IS_GTOLM_DROP,tsize,idxs,&nout,tidxs));
        CHKERRQ(ISRestoreIndices(pcbddc->ISForDofsLocal[i],&idxs));
      } else {
        CHKERRQ(ISGetLocalSize(pcbddc->nedclocal,&tsize));
        CHKERRQ(ISGetIndices(pcbddc->nedclocal,&idxs));
        CHKERRQ(ISGlobalToLocalMappingApply(tmap,IS_GTOLM_DROP,tsize,idxs,&nout,tidxs));
        PetscCheckFalse(tsize != nout,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Failed when mapping coarse nedelec field! %D != %D",tsize,nout);
        CHKERRQ(ISRestoreIndices(pcbddc->nedclocal,&idxs));
      }
      CHKERRQ(ISLocalToGlobalMappingApply(coarse_islg,nout,tidxs,tidxs2));
      CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)pc),nout,tidxs2,PETSC_COPY_VALUES,&isarray[i]));
      /* CHKERRQ(ISView(isarray[i],0)); */
    }
    /* neumann boundaries */
    if (pcbddc->NeumannBoundariesLocal) {
      /* CHKERRQ(ISView(pcbddc->NeumannBoundariesLocal,0)); */
      CHKERRQ(ISGetLocalSize(pcbddc->NeumannBoundariesLocal,&tsize));
      CHKERRQ(ISGetIndices(pcbddc->NeumannBoundariesLocal,&idxs));
      CHKERRQ(ISGlobalToLocalMappingApply(tmap,IS_GTOLM_DROP,tsize,idxs,&nout,tidxs));
      CHKERRQ(ISRestoreIndices(pcbddc->NeumannBoundariesLocal,&idxs));
      CHKERRQ(ISLocalToGlobalMappingApply(coarse_islg,nout,tidxs,tidxs2));
      CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)pc),nout,tidxs2,PETSC_COPY_VALUES,&isarray[nisdofs]));
      /* CHKERRQ(ISView(isarray[nisdofs],0)); */
    }
    /* coordinates */
    if (pcbddc->corner_selected) {
      CHKERRQ(PCBDDCGraphGetCandidatesIS(pcbddc->mat_graph,NULL,NULL,NULL,NULL,&corners));
      CHKERRQ(ISGetLocalSize(corners,&tsize));
      CHKERRQ(ISGetIndices(corners,&idxs));
      CHKERRQ(ISGlobalToLocalMappingApply(tmap,IS_GTOLM_DROP,tsize,idxs,&nout,tidxs));
      PetscCheckFalse(tsize != nout,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Failed when mapping corners! %D != %D",tsize,nout);
      CHKERRQ(ISRestoreIndices(corners,&idxs));
      CHKERRQ(PCBDDCGraphRestoreCandidatesIS(pcbddc->mat_graph,NULL,NULL,NULL,NULL,&corners));
      CHKERRQ(ISLocalToGlobalMappingApply(coarse_islg,nout,tidxs,tidxs2));
      CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)pc),nout,tidxs2,PETSC_COPY_VALUES,&corners));
    }
    CHKERRQ(PetscFree(tidxs));
    CHKERRQ(PetscFree(tidxs2));
    CHKERRQ(ISLocalToGlobalMappingDestroy(&tmap));
  } else {
    nis = 0;
    nisdofs = 0;
    nisneu = 0;
    nisvert = 0;
    isarray = NULL;
  }
  /* destroy no longer needed map */
  CHKERRQ(ISLocalToGlobalMappingDestroy(&coarse_islg));

  /* subassemble */
  if (multilevel_allowed) {
    Vec       vp[1];
    PetscInt  nvecs = 0;
    PetscBool reuse,reuser;

    if (coarse_mat) reuse = PETSC_TRUE;
    else reuse = PETSC_FALSE;
    CHKERRMPI(MPIU_Allreduce(&reuse,&reuser,1,MPIU_BOOL,MPI_LOR,PetscObjectComm((PetscObject)pc)));
    vp[0] = NULL;
    if (pcbddc->benign_have_null) { /* propagate no-net-flux quadrature to coarser level */
      CHKERRQ(VecCreate(PetscObjectComm((PetscObject)pc),&vp[0]));
      CHKERRQ(VecSetSizes(vp[0],pcbddc->local_primal_size,PETSC_DECIDE));
      CHKERRQ(VecSetType(vp[0],VECSTANDARD));
      nvecs = 1;

      if (pcbddc->divudotp) {
        Mat      B,loc_divudotp;
        Vec      v,p;
        IS       dummy;
        PetscInt np;

        CHKERRQ(MatISGetLocalMat(pcbddc->divudotp,&loc_divudotp));
        CHKERRQ(MatGetSize(loc_divudotp,&np,NULL));
        CHKERRQ(ISCreateStride(PETSC_COMM_SELF,np,0,1,&dummy));
        CHKERRQ(MatCreateSubMatrix(loc_divudotp,dummy,pcis->is_B_local,MAT_INITIAL_MATRIX,&B));
        CHKERRQ(MatCreateVecs(B,&v,&p));
        CHKERRQ(VecSet(p,1.));
        CHKERRQ(MatMultTranspose(B,p,v));
        CHKERRQ(VecDestroy(&p));
        CHKERRQ(MatDestroy(&B));
        CHKERRQ(VecGetArray(vp[0],&array));
        CHKERRQ(VecPlaceArray(pcbddc->vec1_P,array));
        CHKERRQ(MatMultTranspose(pcbddc->coarse_phi_B,v,pcbddc->vec1_P));
        CHKERRQ(VecResetArray(pcbddc->vec1_P));
        CHKERRQ(VecRestoreArray(vp[0],&array));
        CHKERRQ(ISDestroy(&dummy));
        CHKERRQ(VecDestroy(&v));
      }
    }
    if (reuser) {
      CHKERRQ(PCBDDCMatISSubassemble(t_coarse_mat_is,pcbddc->coarse_subassembling,0,restr,full_restr,PETSC_TRUE,&coarse_mat,nis,isarray,nvecs,vp));
    } else {
      CHKERRQ(PCBDDCMatISSubassemble(t_coarse_mat_is,pcbddc->coarse_subassembling,0,restr,full_restr,PETSC_FALSE,&coarse_mat_is,nis,isarray,nvecs,vp));
    }
    if (vp[0]) { /* vp[0] could have been placed on a different set of processes */
      PetscScalar       *arraym;
      const PetscScalar *arrayv;
      PetscInt          nl;
      CHKERRQ(VecGetLocalSize(vp[0],&nl));
      CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,1,nl,NULL,&coarsedivudotp));
      CHKERRQ(MatDenseGetArray(coarsedivudotp,&arraym));
      CHKERRQ(VecGetArrayRead(vp[0],&arrayv));
      CHKERRQ(PetscArraycpy(arraym,arrayv,nl));
      CHKERRQ(VecRestoreArrayRead(vp[0],&arrayv));
      CHKERRQ(MatDenseRestoreArray(coarsedivudotp,&arraym));
      CHKERRQ(VecDestroy(&vp[0]));
    } else {
      CHKERRQ(MatCreateSeqAIJ(PETSC_COMM_SELF,0,0,1,NULL,&coarsedivudotp));
    }
  } else {
    CHKERRQ(PCBDDCMatISSubassemble(t_coarse_mat_is,pcbddc->coarse_subassembling,0,restr,full_restr,PETSC_FALSE,&coarse_mat_is,0,NULL,0,NULL));
  }
  if (coarse_mat_is || coarse_mat) {
    if (!multilevel_allowed) {
      CHKERRQ(MatConvert(coarse_mat_is,MATAIJ,coarse_mat_reuse,&coarse_mat));
    } else {
      /* if this matrix is present, it means we are not reusing the coarse matrix */
      if (coarse_mat_is) {
        PetscCheckFalse(coarse_mat,PetscObjectComm((PetscObject)coarse_mat_is),PETSC_ERR_PLIB,"This should not happen");
        CHKERRQ(PetscObjectReference((PetscObject)coarse_mat_is));
        coarse_mat = coarse_mat_is;
      }
    }
  }
  CHKERRQ(MatDestroy(&t_coarse_mat_is));
  CHKERRQ(MatDestroy(&coarse_mat_is));

  /* create local to global scatters for coarse problem */
  if (compute_vecs) {
    PetscInt lrows;
    CHKERRQ(VecDestroy(&pcbddc->coarse_vec));
    if (coarse_mat) {
      CHKERRQ(MatGetLocalSize(coarse_mat,&lrows,NULL));
    } else {
      lrows = 0;
    }
    CHKERRQ(VecCreate(PetscObjectComm((PetscObject)pc),&pcbddc->coarse_vec));
    CHKERRQ(VecSetSizes(pcbddc->coarse_vec,lrows,PETSC_DECIDE));
    CHKERRQ(VecSetType(pcbddc->coarse_vec,coarse_mat ? coarse_mat->defaultvectype : VECSTANDARD));
    CHKERRQ(VecScatterDestroy(&pcbddc->coarse_loc_to_glob));
    CHKERRQ(VecScatterCreate(pcbddc->vec1_P,NULL,pcbddc->coarse_vec,coarse_is,&pcbddc->coarse_loc_to_glob));
  }
  CHKERRQ(ISDestroy(&coarse_is));

  /* set defaults for coarse KSP and PC */
  if (multilevel_allowed) {
    coarse_ksp_type = KSPRICHARDSON;
    coarse_pc_type  = PCBDDC;
  } else {
    coarse_ksp_type = KSPPREONLY;
    coarse_pc_type  = PCREDUNDANT;
  }

  /* print some info if requested */
  if (pcbddc->dbg_flag) {
    if (!multilevel_allowed) {
      CHKERRQ(PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"--------------------------------------------------\n"));
      if (multilevel_requested) {
        CHKERRQ(PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"Not enough active processes on level %D (active processes %D, coarsening ratio %D)\n",pcbddc->current_level,active_procs,pcbddc->coarsening_ratio));
      } else if (pcbddc->max_levels) {
        CHKERRQ(PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"Maximum number of requested levels reached (%D)\n",pcbddc->max_levels));
      }
      CHKERRQ(PetscViewerFlush(pcbddc->dbg_viewer));
    }
  }

  /* communicate coarse discrete gradient */
  coarseG = NULL;
  if (pcbddc->nedcG && multilevel_allowed) {
    MPI_Comm ccomm;
    if (coarse_mat) {
      ccomm = PetscObjectComm((PetscObject)coarse_mat);
    } else {
      ccomm = MPI_COMM_NULL;
    }
    CHKERRQ(MatMPIAIJRestrict(pcbddc->nedcG,ccomm,&coarseG));
  }

  /* create the coarse KSP object only once with defaults */
  if (coarse_mat) {
    PetscBool   isredundant,isbddc,force,valid;
    PetscViewer dbg_viewer = NULL;

    if (pcbddc->dbg_flag) {
      dbg_viewer = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)coarse_mat));
      CHKERRQ(PetscViewerASCIIAddTab(dbg_viewer,2*pcbddc->current_level));
    }
    if (!pcbddc->coarse_ksp) {
      char   prefix[256],str_level[16];
      size_t len;

      CHKERRQ(KSPCreate(PetscObjectComm((PetscObject)coarse_mat),&pcbddc->coarse_ksp));
      CHKERRQ(KSPSetErrorIfNotConverged(pcbddc->coarse_ksp,pc->erroriffailure));
      CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)pcbddc->coarse_ksp,(PetscObject)pc,1));
      CHKERRQ(KSPSetTolerances(pcbddc->coarse_ksp,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,1));
      CHKERRQ(KSPSetOperators(pcbddc->coarse_ksp,coarse_mat,coarse_mat));
      CHKERRQ(KSPSetType(pcbddc->coarse_ksp,coarse_ksp_type));
      CHKERRQ(KSPSetNormType(pcbddc->coarse_ksp,KSP_NORM_NONE));
      CHKERRQ(KSPGetPC(pcbddc->coarse_ksp,&pc_temp));
      /* TODO is this logic correct? should check for coarse_mat type */
      CHKERRQ(PCSetType(pc_temp,coarse_pc_type));
      /* prefix */
      CHKERRQ(PetscStrcpy(prefix,""));
      CHKERRQ(PetscStrcpy(str_level,""));
      if (!pcbddc->current_level) {
        CHKERRQ(PetscStrncpy(prefix,((PetscObject)pc)->prefix,sizeof(prefix)));
        CHKERRQ(PetscStrlcat(prefix,"pc_bddc_coarse_",sizeof(prefix)));
      } else {
        CHKERRQ(PetscStrlen(((PetscObject)pc)->prefix,&len));
        if (pcbddc->current_level>1) len -= 3; /* remove "lX_" with X level number */
        if (pcbddc->current_level>10) len -= 1; /* remove another char from level number */
        /* Nonstandard use of PetscStrncpy() to copy only a portion of the string */
        CHKERRQ(PetscStrncpy(prefix,((PetscObject)pc)->prefix,len+1));
        CHKERRQ(PetscSNPrintf(str_level,sizeof(str_level),"l%d_",(int)(pcbddc->current_level)));
        CHKERRQ(PetscStrlcat(prefix,str_level,sizeof(prefix)));
      }
      CHKERRQ(KSPSetOptionsPrefix(pcbddc->coarse_ksp,prefix));
      /* propagate BDDC info to the next level (these are dummy calls if pc_temp is not of type PCBDDC) */
      CHKERRQ(PCBDDCSetLevel(pc_temp,pcbddc->current_level+1));
      CHKERRQ(PCBDDCSetCoarseningRatio(pc_temp,pcbddc->coarsening_ratio));
      CHKERRQ(PCBDDCSetLevels(pc_temp,pcbddc->max_levels));
      /* allow user customization */
      CHKERRQ(KSPSetFromOptions(pcbddc->coarse_ksp));
      /* get some info after set from options */
      CHKERRQ(KSPGetPC(pcbddc->coarse_ksp,&pc_temp));
      /* multilevel cannot be done with coarse PC different from BDDC, NN, HPDDM, unless forced to */
      force = PETSC_FALSE;
      CHKERRQ(PetscOptionsGetBool(NULL,((PetscObject)pc_temp)->prefix,"-pc_type_forced",&force,NULL));
      CHKERRQ(PetscObjectTypeCompareAny((PetscObject)pc_temp,&valid,PCBDDC,PCNN,PCHPDDM,""));
      CHKERRQ(PetscObjectTypeCompare((PetscObject)pc_temp,PCBDDC,&isbddc));
      if (multilevel_allowed && !force && !valid) {
        isbddc = PETSC_TRUE;
        CHKERRQ(PCSetType(pc_temp,PCBDDC));
        CHKERRQ(PCBDDCSetLevel(pc_temp,pcbddc->current_level+1));
        CHKERRQ(PCBDDCSetCoarseningRatio(pc_temp,pcbddc->coarsening_ratio));
        CHKERRQ(PCBDDCSetLevels(pc_temp,pcbddc->max_levels));
        if (pc_temp->ops->setfromoptions) { /* need to setfromoptions again, skipping the pc_type */
          ierr = PetscObjectOptionsBegin((PetscObject)pc_temp);CHKERRQ(ierr);
          CHKERRQ((*pc_temp->ops->setfromoptions)(PetscOptionsObject,pc_temp));
          CHKERRQ(PetscObjectProcessOptionsHandlers(PetscOptionsObject,(PetscObject)pc_temp));
          ierr = PetscOptionsEnd();CHKERRQ(ierr);
          pc_temp->setfromoptionscalled++;
        }
      }
    }
    /* propagate BDDC info to the next level (these are dummy calls if pc_temp is not of type PCBDDC) */
    CHKERRQ(KSPGetPC(pcbddc->coarse_ksp,&pc_temp));
    if (nisdofs) {
      CHKERRQ(PCBDDCSetDofsSplitting(pc_temp,nisdofs,isarray));
      for (i=0;i<nisdofs;i++) {
        CHKERRQ(ISDestroy(&isarray[i]));
      }
    }
    if (nisneu) {
      CHKERRQ(PCBDDCSetNeumannBoundaries(pc_temp,isarray[nisdofs]));
      CHKERRQ(ISDestroy(&isarray[nisdofs]));
    }
    if (nisvert) {
      CHKERRQ(PCBDDCSetPrimalVerticesIS(pc_temp,isarray[nis-1]));
      CHKERRQ(ISDestroy(&isarray[nis-1]));
    }
    if (coarseG) {
      CHKERRQ(PCBDDCSetDiscreteGradient(pc_temp,coarseG,1,nedcfield,PETSC_FALSE,PETSC_TRUE));
    }

    /* get some info after set from options */
    CHKERRQ(PetscObjectTypeCompare((PetscObject)pc_temp,PCBDDC,&isbddc));

    /* multilevel can only be requested via -pc_bddc_levels or PCBDDCSetLevels */
    if (isbddc && !multilevel_allowed) {
      CHKERRQ(PCSetType(pc_temp,coarse_pc_type));
    }
    /* multilevel cannot be done with coarse PC different from BDDC, NN, HPDDM, unless forced to */
    force = PETSC_FALSE;
    CHKERRQ(PetscOptionsGetBool(NULL,((PetscObject)pc_temp)->prefix,"-pc_type_forced",&force,NULL));
    CHKERRQ(PetscObjectTypeCompareAny((PetscObject)pc_temp,&valid,PCBDDC,PCNN,PCHPDDM,""));
    if (multilevel_requested && multilevel_allowed && !valid && !force) {
      CHKERRQ(PCSetType(pc_temp,PCBDDC));
    }
    CHKERRQ(PetscObjectTypeCompare((PetscObject)pc_temp,PCREDUNDANT,&isredundant));
    if (isredundant) {
      KSP inner_ksp;
      PC  inner_pc;

      CHKERRQ(PCRedundantGetKSP(pc_temp,&inner_ksp));
      CHKERRQ(KSPGetPC(inner_ksp,&inner_pc));
    }

    /* parameters which miss an API */
    CHKERRQ(PetscObjectTypeCompare((PetscObject)pc_temp,PCBDDC,&isbddc));
    if (isbddc) {
      PC_BDDC* pcbddc_coarse = (PC_BDDC*)pc_temp->data;

      pcbddc_coarse->detect_disconnected = PETSC_TRUE;
      pcbddc_coarse->coarse_eqs_per_proc = pcbddc->coarse_eqs_per_proc;
      pcbddc_coarse->coarse_eqs_limit    = pcbddc->coarse_eqs_limit;
      pcbddc_coarse->benign_saddle_point = pcbddc->benign_have_null;
      if (pcbddc_coarse->benign_saddle_point) {
        Mat                    coarsedivudotp_is;
        ISLocalToGlobalMapping l2gmap,rl2g,cl2g;
        IS                     row,col;
        const PetscInt         *gidxs;
        PetscInt               n,st,M,N;

        CHKERRQ(MatGetSize(coarsedivudotp,&n,NULL));
        CHKERRMPI(MPI_Scan(&n,&st,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)coarse_mat)));
        st   = st-n;
        CHKERRQ(ISCreateStride(PetscObjectComm((PetscObject)coarse_mat),1,st,1,&row));
        CHKERRQ(MatISGetLocalToGlobalMapping(coarse_mat,&l2gmap,NULL));
        CHKERRQ(ISLocalToGlobalMappingGetSize(l2gmap,&n));
        CHKERRQ(ISLocalToGlobalMappingGetIndices(l2gmap,&gidxs));
        CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)coarse_mat),n,gidxs,PETSC_COPY_VALUES,&col));
        CHKERRQ(ISLocalToGlobalMappingRestoreIndices(l2gmap,&gidxs));
        CHKERRQ(ISLocalToGlobalMappingCreateIS(row,&rl2g));
        CHKERRQ(ISLocalToGlobalMappingCreateIS(col,&cl2g));
        CHKERRQ(ISGetSize(row,&M));
        CHKERRQ(MatGetSize(coarse_mat,&N,NULL));
        CHKERRQ(ISDestroy(&row));
        CHKERRQ(ISDestroy(&col));
        CHKERRQ(MatCreate(PetscObjectComm((PetscObject)coarse_mat),&coarsedivudotp_is));
        CHKERRQ(MatSetType(coarsedivudotp_is,MATIS));
        CHKERRQ(MatSetSizes(coarsedivudotp_is,PETSC_DECIDE,PETSC_DECIDE,M,N));
        CHKERRQ(MatSetLocalToGlobalMapping(coarsedivudotp_is,rl2g,cl2g));
        CHKERRQ(ISLocalToGlobalMappingDestroy(&rl2g));
        CHKERRQ(ISLocalToGlobalMappingDestroy(&cl2g));
        CHKERRQ(MatISSetLocalMat(coarsedivudotp_is,coarsedivudotp));
        CHKERRQ(MatDestroy(&coarsedivudotp));
        CHKERRQ(PCBDDCSetDivergenceMat(pc_temp,coarsedivudotp_is,PETSC_FALSE,NULL));
        CHKERRQ(MatDestroy(&coarsedivudotp_is));
        pcbddc_coarse->adaptive_userdefined = PETSC_TRUE;
        if (pcbddc->adaptive_threshold[0] == 0.0) pcbddc_coarse->deluxe_zerorows = PETSC_TRUE;
      }
    }

    /* propagate symmetry info of coarse matrix */
    CHKERRQ(MatSetOption(coarse_mat,MAT_STRUCTURALLY_SYMMETRIC,PETSC_TRUE));
    if (pc->pmat->symmetric_set) {
      CHKERRQ(MatSetOption(coarse_mat,MAT_SYMMETRIC,pc->pmat->symmetric));
    }
    if (pc->pmat->hermitian_set) {
      CHKERRQ(MatSetOption(coarse_mat,MAT_HERMITIAN,pc->pmat->hermitian));
    }
    if (pc->pmat->spd_set) {
      CHKERRQ(MatSetOption(coarse_mat,MAT_SPD,pc->pmat->spd));
    }
    if (pcbddc->benign_saddle_point && !pcbddc->benign_have_null) {
      CHKERRQ(MatSetOption(coarse_mat,MAT_SPD,PETSC_TRUE));
    }
    /* set operators */
    CHKERRQ(MatViewFromOptions(coarse_mat,(PetscObject)pc,"-pc_bddc_coarse_mat_view"));
    CHKERRQ(MatSetOptionsPrefix(coarse_mat,((PetscObject)pcbddc->coarse_ksp)->prefix));
    CHKERRQ(KSPSetOperators(pcbddc->coarse_ksp,coarse_mat,coarse_mat));
    if (pcbddc->dbg_flag) {
      CHKERRQ(PetscViewerASCIISubtractTab(dbg_viewer,2*pcbddc->current_level));
    }
  }
  CHKERRQ(MatDestroy(&coarseG));
  CHKERRQ(PetscFree(isarray));
#if 0
  {
    PetscViewer viewer;
    char filename[256];
    sprintf(filename,"coarse_mat_level%d.m",pcbddc->current_level);
    CHKERRQ(PetscViewerASCIIOpen(PetscObjectComm((PetscObject)coarse_mat),filename,&viewer));
    CHKERRQ(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB));
    CHKERRQ(MatView(coarse_mat,viewer));
    CHKERRQ(PetscViewerPopFormat(viewer));
    CHKERRQ(PetscViewerDestroy(&viewer));
  }
#endif

  if (corners) {
    Vec            gv;
    IS             is;
    const PetscInt *idxs;
    PetscInt       i,d,N,n,cdim = pcbddc->mat_graph->cdim;
    PetscScalar    *coords;

    PetscCheckFalse(!pcbddc->mat_graph->cloc,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing local coordinates");
    CHKERRQ(VecGetSize(pcbddc->coarse_vec,&N));
    CHKERRQ(VecGetLocalSize(pcbddc->coarse_vec,&n));
    CHKERRQ(VecCreate(PetscObjectComm((PetscObject)pcbddc->coarse_vec),&gv));
    CHKERRQ(VecSetBlockSize(gv,cdim));
    CHKERRQ(VecSetSizes(gv,n*cdim,N*cdim));
    CHKERRQ(VecSetType(gv,VECSTANDARD));
    CHKERRQ(VecSetFromOptions(gv));
    CHKERRQ(VecSet(gv,PETSC_MAX_REAL)); /* we only propagate coordinates from vertices constraints */

    CHKERRQ(PCBDDCGraphGetCandidatesIS(pcbddc->mat_graph,NULL,NULL,NULL,NULL,&is));
    CHKERRQ(ISGetLocalSize(is,&n));
    CHKERRQ(ISGetIndices(is,&idxs));
    CHKERRQ(PetscMalloc1(n*cdim,&coords));
    for (i=0;i<n;i++) {
      for (d=0;d<cdim;d++) {
        coords[cdim*i+d] = pcbddc->mat_graph->coords[cdim*idxs[i]+d];
      }
    }
    CHKERRQ(ISRestoreIndices(is,&idxs));
    CHKERRQ(PCBDDCGraphRestoreCandidatesIS(pcbddc->mat_graph,NULL,NULL,NULL,NULL,&is));

    CHKERRQ(ISGetLocalSize(corners,&n));
    CHKERRQ(ISGetIndices(corners,&idxs));
    CHKERRQ(VecSetValuesBlocked(gv,n,idxs,coords,INSERT_VALUES));
    CHKERRQ(ISRestoreIndices(corners,&idxs));
    CHKERRQ(PetscFree(coords));
    CHKERRQ(VecAssemblyBegin(gv));
    CHKERRQ(VecAssemblyEnd(gv));
    CHKERRQ(VecGetArray(gv,&coords));
    if (pcbddc->coarse_ksp) {
      PC        coarse_pc;
      PetscBool isbddc;

      CHKERRQ(KSPGetPC(pcbddc->coarse_ksp,&coarse_pc));
      CHKERRQ(PetscObjectTypeCompare((PetscObject)coarse_pc,PCBDDC,&isbddc));
      if (isbddc) { /* coarse coordinates have PETSC_MAX_REAL, specific for BDDC */
        PetscReal *realcoords;

        CHKERRQ(VecGetLocalSize(gv,&n));
#if defined(PETSC_USE_COMPLEX)
        CHKERRQ(PetscMalloc1(n,&realcoords));
        for (i=0;i<n;i++) realcoords[i] = PetscRealPart(coords[i]);
#else
        realcoords = coords;
#endif
        CHKERRQ(PCSetCoordinates(coarse_pc,cdim,n/cdim,realcoords));
#if defined(PETSC_USE_COMPLEX)
        CHKERRQ(PetscFree(realcoords));
#endif
      }
    }
    CHKERRQ(VecRestoreArray(gv,&coords));
    CHKERRQ(VecDestroy(&gv));
  }
  CHKERRQ(ISDestroy(&corners));

  if (pcbddc->coarse_ksp) {
    Vec crhs,csol;

    CHKERRQ(KSPGetSolution(pcbddc->coarse_ksp,&csol));
    CHKERRQ(KSPGetRhs(pcbddc->coarse_ksp,&crhs));
    if (!csol) {
      CHKERRQ(MatCreateVecs(coarse_mat,&((pcbddc->coarse_ksp)->vec_sol),NULL));
    }
    if (!crhs) {
      CHKERRQ(MatCreateVecs(coarse_mat,NULL,&((pcbddc->coarse_ksp)->vec_rhs)));
    }
  }
  CHKERRQ(MatDestroy(&coarsedivudotp));

  /* compute null space for coarse solver if the benign trick has been requested */
  if (pcbddc->benign_null) {

    CHKERRQ(VecSet(pcbddc->vec1_P,0.));
    for (i=0;i<pcbddc->benign_n;i++) {
      CHKERRQ(VecSetValue(pcbddc->vec1_P,pcbddc->local_primal_size-pcbddc->benign_n+i,1.0,INSERT_VALUES));
    }
    CHKERRQ(VecAssemblyBegin(pcbddc->vec1_P));
    CHKERRQ(VecAssemblyEnd(pcbddc->vec1_P));
    CHKERRQ(VecScatterBegin(pcbddc->coarse_loc_to_glob,pcbddc->vec1_P,pcbddc->coarse_vec,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(pcbddc->coarse_loc_to_glob,pcbddc->vec1_P,pcbddc->coarse_vec,INSERT_VALUES,SCATTER_FORWARD));
    if (coarse_mat) {
      Vec         nullv;
      PetscScalar *array,*array2;
      PetscInt    nl;

      CHKERRQ(MatCreateVecs(coarse_mat,&nullv,NULL));
      CHKERRQ(VecGetLocalSize(nullv,&nl));
      CHKERRQ(VecGetArrayRead(pcbddc->coarse_vec,(const PetscScalar**)&array));
      CHKERRQ(VecGetArray(nullv,&array2));
      CHKERRQ(PetscArraycpy(array2,array,nl));
      CHKERRQ(VecRestoreArray(nullv,&array2));
      CHKERRQ(VecRestoreArrayRead(pcbddc->coarse_vec,(const PetscScalar**)&array));
      CHKERRQ(VecNormalize(nullv,NULL));
      CHKERRQ(MatNullSpaceCreate(PetscObjectComm((PetscObject)coarse_mat),PETSC_FALSE,1,&nullv,&CoarseNullSpace));
      CHKERRQ(VecDestroy(&nullv));
    }
  }
  CHKERRQ(PetscLogEventEnd(PC_BDDC_CoarseSetUp[pcbddc->current_level],pc,0,0,0));

  CHKERRQ(PetscLogEventBegin(PC_BDDC_CoarseSolver[pcbddc->current_level],pc,0,0,0));
  if (pcbddc->coarse_ksp) {
    PetscBool ispreonly;

    if (CoarseNullSpace) {
      PetscBool isnull;
      CHKERRQ(MatNullSpaceTest(CoarseNullSpace,coarse_mat,&isnull));
      if (isnull) {
        CHKERRQ(MatSetNullSpace(coarse_mat,CoarseNullSpace));
      }
      /* TODO: add local nullspaces (if any) */
    }
    /* setup coarse ksp */
    CHKERRQ(KSPSetUp(pcbddc->coarse_ksp));
    /* Check coarse problem if in debug mode or if solving with an iterative method */
    CHKERRQ(PetscObjectTypeCompare((PetscObject)pcbddc->coarse_ksp,KSPPREONLY,&ispreonly));
    if (pcbddc->dbg_flag || (!ispreonly && pcbddc->use_coarse_estimates)) {
      KSP       check_ksp;
      KSPType   check_ksp_type;
      PC        check_pc;
      Vec       check_vec,coarse_vec;
      PetscReal abs_infty_error,infty_error,lambda_min=1.0,lambda_max=1.0;
      PetscInt  its;
      PetscBool compute_eigs;
      PetscReal *eigs_r,*eigs_c;
      PetscInt  neigs;
      const char *prefix;

      /* Create ksp object suitable for estimation of extreme eigenvalues */
      CHKERRQ(KSPCreate(PetscObjectComm((PetscObject)pcbddc->coarse_ksp),&check_ksp));
      CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)check_ksp,(PetscObject)pcbddc->coarse_ksp,0));
      CHKERRQ(KSPSetErrorIfNotConverged(pcbddc->coarse_ksp,PETSC_FALSE));
      CHKERRQ(KSPSetOperators(check_ksp,coarse_mat,coarse_mat));
      CHKERRQ(KSPSetTolerances(check_ksp,1.e-12,1.e-12,PETSC_DEFAULT,pcbddc->coarse_size));
      /* prevent from setup unneeded object */
      CHKERRQ(KSPGetPC(check_ksp,&check_pc));
      CHKERRQ(PCSetType(check_pc,PCNONE));
      if (ispreonly) {
        check_ksp_type = KSPPREONLY;
        compute_eigs = PETSC_FALSE;
      } else {
        check_ksp_type = KSPGMRES;
        compute_eigs = PETSC_TRUE;
      }
      CHKERRQ(KSPSetType(check_ksp,check_ksp_type));
      CHKERRQ(KSPSetComputeSingularValues(check_ksp,compute_eigs));
      CHKERRQ(KSPSetComputeEigenvalues(check_ksp,compute_eigs));
      CHKERRQ(KSPGMRESSetRestart(check_ksp,pcbddc->coarse_size+1));
      CHKERRQ(KSPGetOptionsPrefix(pcbddc->coarse_ksp,&prefix));
      CHKERRQ(KSPSetOptionsPrefix(check_ksp,prefix));
      CHKERRQ(KSPAppendOptionsPrefix(check_ksp,"check_"));
      CHKERRQ(KSPSetFromOptions(check_ksp));
      CHKERRQ(KSPSetUp(check_ksp));
      CHKERRQ(KSPGetPC(pcbddc->coarse_ksp,&check_pc));
      CHKERRQ(KSPSetPC(check_ksp,check_pc));
      /* create random vec */
      CHKERRQ(MatCreateVecs(coarse_mat,&coarse_vec,&check_vec));
      CHKERRQ(VecSetRandom(check_vec,NULL));
      CHKERRQ(MatMult(coarse_mat,check_vec,coarse_vec));
      /* solve coarse problem */
      CHKERRQ(KSPSolve(check_ksp,coarse_vec,coarse_vec));
      CHKERRQ(KSPCheckSolve(check_ksp,pc,coarse_vec));
      /* set eigenvalue estimation if preonly has not been requested */
      if (compute_eigs) {
        CHKERRQ(PetscMalloc1(pcbddc->coarse_size+1,&eigs_r));
        CHKERRQ(PetscMalloc1(pcbddc->coarse_size+1,&eigs_c));
        CHKERRQ(KSPComputeEigenvalues(check_ksp,pcbddc->coarse_size+1,eigs_r,eigs_c,&neigs));
        if (neigs) {
          lambda_max = eigs_r[neigs-1];
          lambda_min = eigs_r[0];
          if (pcbddc->use_coarse_estimates) {
            if (lambda_max>=lambda_min) { /* using PETSC_SMALL since lambda_max == lambda_min is not allowed by KSPChebyshevSetEigenvalues */
              CHKERRQ(KSPChebyshevSetEigenvalues(pcbddc->coarse_ksp,lambda_max+PETSC_SMALL,lambda_min));
              CHKERRQ(KSPRichardsonSetScale(pcbddc->coarse_ksp,2.0/(lambda_max+lambda_min)));
            }
          }
        }
      }

      /* check coarse problem residual error */
      if (pcbddc->dbg_flag) {
        PetscViewer dbg_viewer = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)pcbddc->coarse_ksp));
        CHKERRQ(PetscViewerASCIIAddTab(dbg_viewer,2*(pcbddc->current_level+1)));
        CHKERRQ(VecAXPY(check_vec,-1.0,coarse_vec));
        CHKERRQ(VecNorm(check_vec,NORM_INFINITY,&infty_error));
        CHKERRQ(MatMult(coarse_mat,check_vec,coarse_vec));
        CHKERRQ(VecNorm(coarse_vec,NORM_INFINITY,&abs_infty_error));
        CHKERRQ(PetscViewerASCIIPrintf(dbg_viewer,"Coarse problem details (use estimates %d)\n",pcbddc->use_coarse_estimates));
        CHKERRQ(PetscObjectPrintClassNamePrefixType((PetscObject)(pcbddc->coarse_ksp),dbg_viewer));
        CHKERRQ(PetscObjectPrintClassNamePrefixType((PetscObject)(check_pc),dbg_viewer));
        CHKERRQ(PetscViewerASCIIPrintf(dbg_viewer,"Coarse problem exact infty_error   : %1.6e\n",infty_error));
        CHKERRQ(PetscViewerASCIIPrintf(dbg_viewer,"Coarse problem residual infty_error: %1.6e\n",abs_infty_error));
        if (CoarseNullSpace) {
          CHKERRQ(PetscViewerASCIIPrintf(dbg_viewer,"Coarse problem is singular\n"));
        }
        if (compute_eigs) {
          PetscReal          lambda_max_s,lambda_min_s;
          KSPConvergedReason reason;
          CHKERRQ(KSPGetType(check_ksp,&check_ksp_type));
          CHKERRQ(KSPGetIterationNumber(check_ksp,&its));
          CHKERRQ(KSPGetConvergedReason(check_ksp,&reason));
          CHKERRQ(KSPComputeExtremeSingularValues(check_ksp,&lambda_max_s,&lambda_min_s));
          CHKERRQ(PetscViewerASCIIPrintf(dbg_viewer,"Coarse problem eigenvalues (estimated with %d iterations of %s, conv reason %d): %1.6e %1.6e (%1.6e %1.6e)\n",its,check_ksp_type,reason,lambda_min,lambda_max,lambda_min_s,lambda_max_s));
          for (i=0;i<neigs;i++) {
            CHKERRQ(PetscViewerASCIIPrintf(dbg_viewer,"%1.6e %1.6ei\n",eigs_r[i],eigs_c[i]));
          }
        }
        CHKERRQ(PetscViewerFlush(dbg_viewer));
        CHKERRQ(PetscViewerASCIISubtractTab(dbg_viewer,2*(pcbddc->current_level+1)));
      }
      CHKERRQ(VecDestroy(&check_vec));
      CHKERRQ(VecDestroy(&coarse_vec));
      CHKERRQ(KSPDestroy(&check_ksp));
      if (compute_eigs) {
        CHKERRQ(PetscFree(eigs_r));
        CHKERRQ(PetscFree(eigs_c));
      }
    }
  }
  CHKERRQ(MatNullSpaceDestroy(&CoarseNullSpace));
  /* print additional info */
  if (pcbddc->dbg_flag) {
    /* waits until all processes reaches this point */
    CHKERRQ(PetscBarrier((PetscObject)pc));
    CHKERRQ(PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"Coarse solver setup completed at level %D\n",pcbddc->current_level));
    CHKERRQ(PetscViewerFlush(pcbddc->dbg_viewer));
  }

  /* free memory */
  CHKERRQ(MatDestroy(&coarse_mat));
  CHKERRQ(PetscLogEventEnd(PC_BDDC_CoarseSolver[pcbddc->current_level],pc,0,0,0));
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCComputePrimalNumbering(PC pc,PetscInt* coarse_size_n,PetscInt** local_primal_indices_n)
{
  PC_BDDC*       pcbddc = (PC_BDDC*)pc->data;
  PC_IS*         pcis = (PC_IS*)pc->data;
  Mat_IS*        matis = (Mat_IS*)pc->pmat->data;
  IS             subset,subset_mult,subset_n;
  PetscInt       local_size,coarse_size=0;
  PetscInt       *local_primal_indices=NULL;
  const PetscInt *t_local_primal_indices;

  PetscFunctionBegin;
  /* Compute global number of coarse dofs */
  PetscCheckFalse(pcbddc->local_primal_size && !pcbddc->local_primal_ref_node,PETSC_COMM_SELF,PETSC_ERR_PLIB,"BDDC ConstraintsSetUp should be called first");
  CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)(pc->pmat)),pcbddc->local_primal_size_cc,pcbddc->local_primal_ref_node,PETSC_COPY_VALUES,&subset_n));
  CHKERRQ(ISLocalToGlobalMappingApplyIS(pcis->mapping,subset_n,&subset));
  CHKERRQ(ISDestroy(&subset_n));
  CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)(pc->pmat)),pcbddc->local_primal_size_cc,pcbddc->local_primal_ref_mult,PETSC_COPY_VALUES,&subset_mult));
  CHKERRQ(ISRenumber(subset,subset_mult,&coarse_size,&subset_n));
  CHKERRQ(ISDestroy(&subset));
  CHKERRQ(ISDestroy(&subset_mult));
  CHKERRQ(ISGetLocalSize(subset_n,&local_size));
  PetscCheckFalse(local_size != pcbddc->local_primal_size,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Invalid number of local primal indices computed %D != %D",local_size,pcbddc->local_primal_size);
  CHKERRQ(PetscMalloc1(local_size,&local_primal_indices));
  CHKERRQ(ISGetIndices(subset_n,&t_local_primal_indices));
  CHKERRQ(PetscArraycpy(local_primal_indices,t_local_primal_indices,local_size));
  CHKERRQ(ISRestoreIndices(subset_n,&t_local_primal_indices));
  CHKERRQ(ISDestroy(&subset_n));

  /* check numbering */
  if (pcbddc->dbg_flag) {
    PetscScalar coarsesum,*array,*array2;
    PetscInt    i;
    PetscBool   set_error = PETSC_FALSE,set_error_reduced = PETSC_FALSE;

    CHKERRQ(PetscViewerFlush(pcbddc->dbg_viewer));
    CHKERRQ(PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"--------------------------------------------------\n"));
    CHKERRQ(PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"Check coarse indices\n"));
    CHKERRQ(PetscViewerASCIIPushSynchronized(pcbddc->dbg_viewer));
    /* counter */
    CHKERRQ(VecSet(pcis->vec1_global,0.0));
    CHKERRQ(VecSet(pcis->vec1_N,1.0));
    CHKERRQ(VecScatterBegin(matis->rctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecScatterEnd(matis->rctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecScatterBegin(matis->rctx,pcis->vec1_global,pcis->vec2_N,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(matis->rctx,pcis->vec1_global,pcis->vec2_N,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecSet(pcis->vec1_N,0.0));
    for (i=0;i<pcbddc->local_primal_size;i++) {
      CHKERRQ(VecSetValue(pcis->vec1_N,pcbddc->primal_indices_local_idxs[i],1.0,INSERT_VALUES));
    }
    CHKERRQ(VecAssemblyBegin(pcis->vec1_N));
    CHKERRQ(VecAssemblyEnd(pcis->vec1_N));
    CHKERRQ(VecSet(pcis->vec1_global,0.0));
    CHKERRQ(VecScatterBegin(matis->rctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecScatterEnd(matis->rctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecScatterBegin(matis->rctx,pcis->vec1_global,pcis->vec1_N,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(matis->rctx,pcis->vec1_global,pcis->vec1_N,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecGetArray(pcis->vec1_N,&array));
    CHKERRQ(VecGetArray(pcis->vec2_N,&array2));
    for (i=0;i<pcis->n;i++) {
      if (array[i] != 0.0 && array[i] != array2[i]) {
        PetscInt owned = (PetscInt)PetscRealPart(array[i]),gi;
        PetscInt neigh = (PetscInt)PetscRealPart(array2[i]);
        set_error = PETSC_TRUE;
        CHKERRQ(ISLocalToGlobalMappingApply(pcis->mapping,1,&i,&gi));
        CHKERRQ(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d: local index %D (gid %D) owned by %D processes instead of %D!\n",PetscGlobalRank,i,gi,owned,neigh));
      }
    }
    CHKERRQ(VecRestoreArray(pcis->vec2_N,&array2));
    CHKERRMPI(MPIU_Allreduce(&set_error,&set_error_reduced,1,MPIU_BOOL,MPI_LOR,PetscObjectComm((PetscObject)pc)));
    CHKERRQ(PetscViewerFlush(pcbddc->dbg_viewer));
    for (i=0;i<pcis->n;i++) {
      if (PetscRealPart(array[i]) > 0.0) array[i] = 1.0/PetscRealPart(array[i]);
    }
    CHKERRQ(VecRestoreArray(pcis->vec1_N,&array));
    CHKERRQ(VecSet(pcis->vec1_global,0.0));
    CHKERRQ(VecScatterBegin(matis->rctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecScatterEnd(matis->rctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecSum(pcis->vec1_global,&coarsesum));
    CHKERRQ(PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"Size of coarse problem is %D (%lf)\n",coarse_size,PetscRealPart(coarsesum)));
    if (pcbddc->dbg_flag > 1 || set_error_reduced) {
      PetscInt *gidxs;

      CHKERRQ(PetscMalloc1(pcbddc->local_primal_size,&gidxs));
      CHKERRQ(ISLocalToGlobalMappingApply(pcis->mapping,pcbddc->local_primal_size,pcbddc->primal_indices_local_idxs,gidxs));
      CHKERRQ(PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"Distribution of local primal indices\n"));
      CHKERRQ(PetscViewerFlush(pcbddc->dbg_viewer));
      CHKERRQ(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d\n",PetscGlobalRank));
      for (i=0;i<pcbddc->local_primal_size;i++) {
        CHKERRQ(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"local_primal_indices[%D]=%D (%D,%D)\n",i,local_primal_indices[i],pcbddc->primal_indices_local_idxs[i],gidxs[i]));
      }
      CHKERRQ(PetscViewerFlush(pcbddc->dbg_viewer));
      CHKERRQ(PetscFree(gidxs));
    }
    CHKERRQ(PetscViewerFlush(pcbddc->dbg_viewer));
    CHKERRQ(PetscViewerASCIIPushSynchronized(pcbddc->dbg_viewer));
    PetscCheckFalse(set_error_reduced,PetscObjectComm((PetscObject)pc),PETSC_ERR_PLIB,"BDDC Numbering of coarse dofs failed");
  }

  /* get back data */
  *coarse_size_n = coarse_size;
  *local_primal_indices_n = local_primal_indices;
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCGlobalToLocal(VecScatter g2l_ctx,Vec gwork, Vec lwork, IS globalis, IS* localis)
{
  IS             localis_t;
  PetscInt       i,lsize,*idxs,n;
  PetscScalar    *vals;

  PetscFunctionBegin;
  /* get indices in local ordering exploiting local to global map */
  CHKERRQ(ISGetLocalSize(globalis,&lsize));
  CHKERRQ(PetscMalloc1(lsize,&vals));
  for (i=0;i<lsize;i++) vals[i] = 1.0;
  CHKERRQ(ISGetIndices(globalis,(const PetscInt**)&idxs));
  CHKERRQ(VecSet(gwork,0.0));
  CHKERRQ(VecSet(lwork,0.0));
  if (idxs) { /* multilevel guard */
    CHKERRQ(VecSetOption(gwork,VEC_IGNORE_NEGATIVE_INDICES,PETSC_TRUE));
    CHKERRQ(VecSetValues(gwork,lsize,idxs,vals,INSERT_VALUES));
  }
  CHKERRQ(VecAssemblyBegin(gwork));
  CHKERRQ(ISRestoreIndices(globalis,(const PetscInt**)&idxs));
  CHKERRQ(PetscFree(vals));
  CHKERRQ(VecAssemblyEnd(gwork));
  /* now compute set in local ordering */
  CHKERRQ(VecScatterBegin(g2l_ctx,gwork,lwork,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(g2l_ctx,gwork,lwork,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecGetArrayRead(lwork,(const PetscScalar**)&vals));
  CHKERRQ(VecGetSize(lwork,&n));
  for (i=0,lsize=0;i<n;i++) {
    if (PetscRealPart(vals[i]) > 0.5) {
      lsize++;
    }
  }
  CHKERRQ(PetscMalloc1(lsize,&idxs));
  for (i=0,lsize=0;i<n;i++) {
    if (PetscRealPart(vals[i]) > 0.5) {
      idxs[lsize++] = i;
    }
  }
  CHKERRQ(VecRestoreArrayRead(lwork,(const PetscScalar**)&vals));
  CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)gwork),lsize,idxs,PETSC_OWN_POINTER,&localis_t));
  *localis = localis_t;
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCSetUpSubSchurs(PC pc)
{
  PC_IS               *pcis=(PC_IS*)pc->data;
  PC_BDDC             *pcbddc=(PC_BDDC*)pc->data;
  PCBDDCSubSchurs     sub_schurs=pcbddc->sub_schurs;
  Mat                 S_j;
  PetscInt            *used_xadj,*used_adjncy;
  PetscBool           free_used_adj;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  CHKERRQ(PetscLogEventBegin(PC_BDDC_Schurs[pcbddc->current_level],pc,0,0,0));
  /* decide the adjacency to be used for determining internal problems for local schur on subsets */
  free_used_adj = PETSC_FALSE;
  if (pcbddc->sub_schurs_layers == -1) {
    used_xadj = NULL;
    used_adjncy = NULL;
  } else {
    if (pcbddc->sub_schurs_use_useradj && pcbddc->mat_graph->xadj) {
      used_xadj = pcbddc->mat_graph->xadj;
      used_adjncy = pcbddc->mat_graph->adjncy;
    } else if (pcbddc->computed_rowadj) {
      used_xadj = pcbddc->mat_graph->xadj;
      used_adjncy = pcbddc->mat_graph->adjncy;
    } else {
      PetscBool      flg_row=PETSC_FALSE;
      const PetscInt *xadj,*adjncy;
      PetscInt       nvtxs;

      CHKERRQ(MatGetRowIJ(pcbddc->local_mat,0,PETSC_TRUE,PETSC_FALSE,&nvtxs,&xadj,&adjncy,&flg_row));
      if (flg_row) {
        CHKERRQ(PetscMalloc2(nvtxs+1,&used_xadj,xadj[nvtxs],&used_adjncy));
        CHKERRQ(PetscArraycpy(used_xadj,xadj,nvtxs+1));
        CHKERRQ(PetscArraycpy(used_adjncy,adjncy,xadj[nvtxs]));
        free_used_adj = PETSC_TRUE;
      } else {
        pcbddc->sub_schurs_layers = -1;
        used_xadj = NULL;
        used_adjncy = NULL;
      }
      CHKERRQ(MatRestoreRowIJ(pcbddc->local_mat,0,PETSC_TRUE,PETSC_FALSE,&nvtxs,&xadj,&adjncy,&flg_row));
    }
  }

  /* setup sub_schurs data */
  CHKERRQ(MatCreateSchurComplement(pcis->A_II,pcis->pA_II,pcis->A_IB,pcis->A_BI,pcis->A_BB,&S_j));
  if (!sub_schurs->schur_explicit) {
    /* pcbddc->ksp_D up to date only if not using MatFactor with Schur complement support */
    CHKERRQ(MatSchurComplementSetKSP(S_j,pcbddc->ksp_D));
    CHKERRQ(PCBDDCSubSchursSetUp(sub_schurs,NULL,S_j,PETSC_FALSE,used_xadj,used_adjncy,pcbddc->sub_schurs_layers,NULL,pcbddc->adaptive_selection,PETSC_FALSE,PETSC_FALSE,0,NULL,NULL,NULL,NULL));
  } else {
    Mat       change = NULL;
    Vec       scaling = NULL;
    IS        change_primal = NULL, iP;
    PetscInt  benign_n;
    PetscBool reuse_solvers = (PetscBool)!pcbddc->use_change_of_basis;
    PetscBool need_change = PETSC_FALSE;
    PetscBool discrete_harmonic = PETSC_FALSE;

    if (!pcbddc->use_vertices && reuse_solvers) {
      PetscInt n_vertices;

      CHKERRQ(ISGetLocalSize(sub_schurs->is_vertices,&n_vertices));
      reuse_solvers = (PetscBool)!n_vertices;
    }
    if (!pcbddc->benign_change_explicit) {
      benign_n = pcbddc->benign_n;
    } else {
      benign_n = 0;
    }
    /* sub_schurs->change is a local object; instead, PCBDDCConstraintsSetUp and the quantities used in the test below are logically collective on pc.
       We need a global reduction to avoid possible deadlocks.
       We assume that sub_schurs->change is created once, and then reused for different solves, unless the topography has been recomputed */
    if (pcbddc->adaptive_userdefined || (pcbddc->deluxe_zerorows && !pcbddc->use_change_of_basis)) {
      PetscBool have_loc_change = (PetscBool)(!!sub_schurs->change);
      CHKERRMPI(MPIU_Allreduce(&have_loc_change,&need_change,1,MPIU_BOOL,MPI_LOR,PetscObjectComm((PetscObject)pc)));
      need_change = (PetscBool)(!need_change);
    }
    /* If the user defines additional constraints, we import them here.
       We need to compute the change of basis according to the quadrature weights attached to pmat via MatSetNearNullSpace, and this could not be done (at the moment) without some hacking */
    if (need_change) {
      PC_IS   *pcisf;
      PC_BDDC *pcbddcf;
      PC      pcf;

      PetscCheckFalse(pcbddc->sub_schurs_rebuild,PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot compute change of basis with a different graph");
      CHKERRQ(PCCreate(PetscObjectComm((PetscObject)pc),&pcf));
      CHKERRQ(PCSetOperators(pcf,pc->mat,pc->pmat));
      CHKERRQ(PCSetType(pcf,PCBDDC));

      /* hacks */
      pcisf                        = (PC_IS*)pcf->data;
      pcisf->is_B_local            = pcis->is_B_local;
      pcisf->vec1_N                = pcis->vec1_N;
      pcisf->BtoNmap               = pcis->BtoNmap;
      pcisf->n                     = pcis->n;
      pcisf->n_B                   = pcis->n_B;
      pcbddcf                      = (PC_BDDC*)pcf->data;
      CHKERRQ(PetscFree(pcbddcf->mat_graph));
      pcbddcf->mat_graph           = pcbddc->mat_graph;
      pcbddcf->use_faces           = PETSC_TRUE;
      pcbddcf->use_change_of_basis = PETSC_TRUE;
      pcbddcf->use_change_on_faces = PETSC_TRUE;
      pcbddcf->use_qr_single       = PETSC_TRUE;
      pcbddcf->fake_change         = PETSC_TRUE;

      /* setup constraints so that we can get information on primal vertices and change of basis (in local numbering) */
      CHKERRQ(PCBDDCConstraintsSetUp(pcf));
      sub_schurs->change_with_qr = pcbddcf->use_qr_single;
      CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,pcbddcf->n_vertices,pcbddcf->local_primal_ref_node,PETSC_COPY_VALUES,&change_primal));
      change = pcbddcf->ConstraintMatrix;
      pcbddcf->ConstraintMatrix = NULL;

      /* free unneeded memory allocated in PCBDDCConstraintsSetUp */
      CHKERRQ(PetscFree(pcbddcf->sub_schurs));
      CHKERRQ(MatNullSpaceDestroy(&pcbddcf->onearnullspace));
      CHKERRQ(PetscFree2(pcbddcf->local_primal_ref_node,pcbddcf->local_primal_ref_mult));
      CHKERRQ(PetscFree(pcbddcf->primal_indices_local_idxs));
      CHKERRQ(PetscFree(pcbddcf->onearnullvecs_state));
      CHKERRQ(PetscFree(pcf->data));
      pcf->ops->destroy = NULL;
      pcf->ops->reset   = NULL;
      CHKERRQ(PCDestroy(&pcf));
    }
    if (!pcbddc->use_deluxe_scaling) scaling = pcis->D;

    CHKERRQ(PetscObjectQuery((PetscObject)pc,"__KSPFETIDP_iP",(PetscObject*)&iP));
    if (iP) {
      ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)iP),sub_schurs->prefix,"BDDC sub_schurs options","PC");CHKERRQ(ierr);
      CHKERRQ(PetscOptionsBool("-sub_schurs_discrete_harmonic",NULL,NULL,discrete_harmonic,&discrete_harmonic,NULL));
      ierr = PetscOptionsEnd();CHKERRQ(ierr);
    }
    if (discrete_harmonic) {
      Mat A;
      CHKERRQ(MatDuplicate(pcbddc->local_mat,MAT_COPY_VALUES,&A));
      CHKERRQ(MatZeroRowsColumnsIS(A,iP,1.0,NULL,NULL));
      CHKERRQ(PetscObjectCompose((PetscObject)A,"__KSPFETIDP_iP",(PetscObject)iP));
      CHKERRQ(PCBDDCSubSchursSetUp(sub_schurs,A,S_j,pcbddc->sub_schurs_exact_schur,used_xadj,used_adjncy,pcbddc->sub_schurs_layers,scaling,pcbddc->adaptive_selection,reuse_solvers,pcbddc->benign_saddle_point,benign_n,pcbddc->benign_p0_lidx,pcbddc->benign_zerodiag_subs,change,change_primal));
      CHKERRQ(MatDestroy(&A));
    } else {
      CHKERRQ(PCBDDCSubSchursSetUp(sub_schurs,pcbddc->local_mat,S_j,pcbddc->sub_schurs_exact_schur,used_xadj,used_adjncy,pcbddc->sub_schurs_layers,scaling,pcbddc->adaptive_selection,reuse_solvers,pcbddc->benign_saddle_point,benign_n,pcbddc->benign_p0_lidx,pcbddc->benign_zerodiag_subs,change,change_primal));
    }
    CHKERRQ(MatDestroy(&change));
    CHKERRQ(ISDestroy(&change_primal));
  }
  CHKERRQ(MatDestroy(&S_j));

  /* free adjacency */
  if (free_used_adj) {
    CHKERRQ(PetscFree2(used_xadj,used_adjncy));
  }
  CHKERRQ(PetscLogEventEnd(PC_BDDC_Schurs[pcbddc->current_level],pc,0,0,0));
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCInitSubSchurs(PC pc)
{
  PC_IS               *pcis=(PC_IS*)pc->data;
  PC_BDDC             *pcbddc=(PC_BDDC*)pc->data;
  PCBDDCGraph         graph;

  PetscFunctionBegin;
  /* attach interface graph for determining subsets */
  if (pcbddc->sub_schurs_rebuild) { /* in case rebuild has been requested, it uses a graph generated only by the neighbouring information */
    IS       verticesIS,verticescomm;
    PetscInt vsize,*idxs;

    CHKERRQ(PCBDDCGraphGetCandidatesIS(pcbddc->mat_graph,NULL,NULL,NULL,NULL,&verticesIS));
    CHKERRQ(ISGetSize(verticesIS,&vsize));
    CHKERRQ(ISGetIndices(verticesIS,(const PetscInt**)&idxs));
    CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)pc),vsize,idxs,PETSC_COPY_VALUES,&verticescomm));
    CHKERRQ(ISRestoreIndices(verticesIS,(const PetscInt**)&idxs));
    CHKERRQ(PCBDDCGraphRestoreCandidatesIS(pcbddc->mat_graph,NULL,NULL,NULL,NULL,&verticesIS));
    CHKERRQ(PCBDDCGraphCreate(&graph));
    CHKERRQ(PCBDDCGraphInit(graph,pcbddc->mat_graph->l2gmap,pcbddc->mat_graph->nvtxs_global,pcbddc->graphmaxcount));
    CHKERRQ(PCBDDCGraphSetUp(graph,pcbddc->mat_graph->custom_minimal_size,NULL,pcbddc->DirichletBoundariesLocal,0,NULL,verticescomm));
    CHKERRQ(ISDestroy(&verticescomm));
    CHKERRQ(PCBDDCGraphComputeConnectedComponents(graph));
  } else {
    graph = pcbddc->mat_graph;
  }
  /* print some info */
  if (pcbddc->dbg_flag && !pcbddc->sub_schurs_rebuild) {
    IS       vertices;
    PetscInt nv,nedges,nfaces;
    CHKERRQ(PCBDDCGraphASCIIView(graph,pcbddc->dbg_flag,pcbddc->dbg_viewer));
    CHKERRQ(PCBDDCGraphGetCandidatesIS(graph,&nfaces,NULL,&nedges,NULL,&vertices));
    CHKERRQ(ISGetSize(vertices,&nv));
    CHKERRQ(PetscViewerASCIIPushSynchronized(pcbddc->dbg_viewer));
    CHKERRQ(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"--------------------------------------------------------------\n"));
    CHKERRQ(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d got %02d local candidate vertices (%D)\n",PetscGlobalRank,(int)nv,pcbddc->use_vertices));
    CHKERRQ(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d got %02d local candidate edges    (%D)\n",PetscGlobalRank,(int)nedges,pcbddc->use_edges));
    CHKERRQ(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d got %02d local candidate faces    (%D)\n",PetscGlobalRank,(int)nfaces,pcbddc->use_faces));
    CHKERRQ(PetscViewerFlush(pcbddc->dbg_viewer));
    CHKERRQ(PetscViewerASCIIPopSynchronized(pcbddc->dbg_viewer));
    CHKERRQ(PCBDDCGraphRestoreCandidatesIS(graph,&nfaces,NULL,&nedges,NULL,&vertices));
  }

  /* sub_schurs init */
  if (!pcbddc->sub_schurs) {
    CHKERRQ(PCBDDCSubSchursCreate(&pcbddc->sub_schurs));
  }
  CHKERRQ(PCBDDCSubSchursInit(pcbddc->sub_schurs,((PetscObject)pc)->prefix,pcis->is_I_local,pcis->is_B_local,graph,pcis->BtoNmap,pcbddc->sub_schurs_rebuild));

  /* free graph struct */
  if (pcbddc->sub_schurs_rebuild) {
    CHKERRQ(PCBDDCGraphDestroy(&graph));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCCheckOperator(PC pc)
{
  PC_IS               *pcis=(PC_IS*)pc->data;
  PC_BDDC             *pcbddc=(PC_BDDC*)pc->data;

  PetscFunctionBegin;
  if (pcbddc->n_vertices == pcbddc->local_primal_size) {
    IS             zerodiag = NULL;
    Mat            S_j,B0_B=NULL;
    Vec            dummy_vec=NULL,vec_check_B,vec_scale_P;
    PetscScalar    *p0_check,*array,*array2;
    PetscReal      norm;
    PetscInt       i;

    /* B0 and B0_B */
    if (zerodiag) {
      IS       dummy;

      CHKERRQ(ISCreateStride(PETSC_COMM_SELF,pcbddc->benign_n,0,1,&dummy));
      CHKERRQ(MatCreateSubMatrix(pcbddc->benign_B0,dummy,pcis->is_B_local,MAT_INITIAL_MATRIX,&B0_B));
      CHKERRQ(MatCreateVecs(B0_B,NULL,&dummy_vec));
      CHKERRQ(ISDestroy(&dummy));
    }
    /* I need a primal vector to scale primal nodes since BDDC sums contibutions */
    CHKERRQ(VecDuplicate(pcbddc->vec1_P,&vec_scale_P));
    CHKERRQ(VecSet(pcbddc->vec1_P,1.0));
    CHKERRQ(VecScatterBegin(pcbddc->coarse_loc_to_glob,pcbddc->vec1_P,pcbddc->coarse_vec,ADD_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(pcbddc->coarse_loc_to_glob,pcbddc->vec1_P,pcbddc->coarse_vec,ADD_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterBegin(pcbddc->coarse_loc_to_glob,pcbddc->coarse_vec,vec_scale_P,INSERT_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecScatterEnd(pcbddc->coarse_loc_to_glob,pcbddc->coarse_vec,vec_scale_P,INSERT_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecReciprocal(vec_scale_P));
    /* S_j */
    CHKERRQ(MatCreateSchurComplement(pcis->A_II,pcis->pA_II,pcis->A_IB,pcis->A_BI,pcis->A_BB,&S_j));
    CHKERRQ(MatSchurComplementSetKSP(S_j,pcbddc->ksp_D));

    /* mimic vector in \widetilde{W}_\Gamma */
    CHKERRQ(VecSetRandom(pcis->vec1_N,NULL));
    /* continuous in primal space */
    CHKERRQ(VecSetRandom(pcbddc->coarse_vec,NULL));
    CHKERRQ(VecScatterBegin(pcbddc->coarse_loc_to_glob,pcbddc->coarse_vec,pcbddc->vec1_P,INSERT_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecScatterEnd(pcbddc->coarse_loc_to_glob,pcbddc->coarse_vec,pcbddc->vec1_P,INSERT_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecGetArray(pcbddc->vec1_P,&array));
    CHKERRQ(PetscCalloc1(pcbddc->benign_n,&p0_check));
    for (i=0;i<pcbddc->benign_n;i++) p0_check[i] = array[pcbddc->local_primal_size-pcbddc->benign_n+i];
    CHKERRQ(VecSetValues(pcis->vec1_N,pcbddc->local_primal_size,pcbddc->local_primal_ref_node,array,INSERT_VALUES));
    CHKERRQ(VecRestoreArray(pcbddc->vec1_P,&array));
    CHKERRQ(VecAssemblyBegin(pcis->vec1_N));
    CHKERRQ(VecAssemblyEnd(pcis->vec1_N));
    CHKERRQ(VecScatterBegin(pcis->N_to_B,pcis->vec1_N,pcis->vec2_B,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(pcis->N_to_B,pcis->vec1_N,pcis->vec2_B,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecDuplicate(pcis->vec2_B,&vec_check_B));
    CHKERRQ(VecCopy(pcis->vec2_B,vec_check_B));

    /* assemble rhs for coarse problem */
    /* widetilde{S}_\Gamma w_\Gamma + \widetilde{B0}^T_B p0 */
    /* local with Schur */
    CHKERRQ(MatMult(S_j,pcis->vec2_B,pcis->vec1_B));
    if (zerodiag) {
      CHKERRQ(VecGetArray(dummy_vec,&array));
      for (i=0;i<pcbddc->benign_n;i++) array[i] = p0_check[i];
      CHKERRQ(VecRestoreArray(dummy_vec,&array));
      CHKERRQ(MatMultTransposeAdd(B0_B,dummy_vec,pcis->vec1_B,pcis->vec1_B));
    }
    /* sum on primal nodes the local contributions */
    CHKERRQ(VecScatterBegin(pcis->N_to_B,pcis->vec1_B,pcis->vec1_N,INSERT_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecScatterEnd(pcis->N_to_B,pcis->vec1_B,pcis->vec1_N,INSERT_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecGetArray(pcis->vec1_N,&array));
    CHKERRQ(VecGetArray(pcbddc->vec1_P,&array2));
    for (i=0;i<pcbddc->local_primal_size;i++) array2[i] = array[pcbddc->local_primal_ref_node[i]];
    CHKERRQ(VecRestoreArray(pcbddc->vec1_P,&array2));
    CHKERRQ(VecRestoreArray(pcis->vec1_N,&array));
    CHKERRQ(VecSet(pcbddc->coarse_vec,0.));
    CHKERRQ(VecScatterBegin(pcbddc->coarse_loc_to_glob,pcbddc->vec1_P,pcbddc->coarse_vec,ADD_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(pcbddc->coarse_loc_to_glob,pcbddc->vec1_P,pcbddc->coarse_vec,ADD_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterBegin(pcbddc->coarse_loc_to_glob,pcbddc->coarse_vec,pcbddc->vec1_P,INSERT_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecScatterEnd(pcbddc->coarse_loc_to_glob,pcbddc->coarse_vec,pcbddc->vec1_P,INSERT_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecGetArray(pcbddc->vec1_P,&array));
    /* scale primal nodes (BDDC sums contibutions) */
    CHKERRQ(VecPointwiseMult(pcbddc->vec1_P,vec_scale_P,pcbddc->vec1_P));
    CHKERRQ(VecSetValues(pcis->vec1_N,pcbddc->local_primal_size,pcbddc->local_primal_ref_node,array,INSERT_VALUES));
    CHKERRQ(VecRestoreArray(pcbddc->vec1_P,&array));
    CHKERRQ(VecAssemblyBegin(pcis->vec1_N));
    CHKERRQ(VecAssemblyEnd(pcis->vec1_N));
    CHKERRQ(VecScatterBegin(pcis->N_to_B,pcis->vec1_N,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(pcis->N_to_B,pcis->vec1_N,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD));
    /* global: \widetilde{B0}_B w_\Gamma */
    if (zerodiag) {
      CHKERRQ(MatMult(B0_B,pcis->vec2_B,dummy_vec));
      CHKERRQ(VecGetArray(dummy_vec,&array));
      for (i=0;i<pcbddc->benign_n;i++) pcbddc->benign_p0[i] = array[i];
      CHKERRQ(VecRestoreArray(dummy_vec,&array));
    }
    /* BDDC */
    CHKERRQ(VecSet(pcis->vec1_D,0.));
    CHKERRQ(PCBDDCApplyInterfacePreconditioner(pc,PETSC_FALSE));

    CHKERRQ(VecCopy(pcis->vec1_B,pcis->vec2_B));
    CHKERRQ(VecAXPY(pcis->vec1_B,-1.0,vec_check_B));
    CHKERRQ(VecNorm(pcis->vec1_B,NORM_INFINITY,&norm));
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"[%d] BDDC local error is %1.4e\n",PetscGlobalRank,norm));
    for (i=0;i<pcbddc->benign_n;i++) {
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"[%d] BDDC p0[%D] error is %1.4e\n",PetscGlobalRank,i,PetscAbsScalar(pcbddc->benign_p0[i]-p0_check[i])));
    }
    CHKERRQ(PetscFree(p0_check));
    CHKERRQ(VecDestroy(&vec_scale_P));
    CHKERRQ(VecDestroy(&vec_check_B));
    CHKERRQ(VecDestroy(&dummy_vec));
    CHKERRQ(MatDestroy(&S_j));
    CHKERRQ(MatDestroy(&B0_B));
  }
  PetscFunctionReturn(0);
}

#include <../src/mat/impls/aij/mpi/mpiaij.h>
PetscErrorCode MatMPIAIJRestrict(Mat A, MPI_Comm ccomm, Mat *B)
{
  Mat            At;
  IS             rows;
  PetscInt       rst,ren;
  PetscLayout    rmap;

  PetscFunctionBegin;
  rst = ren = 0;
  if (ccomm != MPI_COMM_NULL) {
    CHKERRQ(PetscLayoutCreate(ccomm,&rmap));
    CHKERRQ(PetscLayoutSetSize(rmap,A->rmap->N));
    CHKERRQ(PetscLayoutSetBlockSize(rmap,1));
    CHKERRQ(PetscLayoutSetUp(rmap));
    CHKERRQ(PetscLayoutGetRange(rmap,&rst,&ren));
  }
  CHKERRQ(ISCreateStride(PetscObjectComm((PetscObject)A),ren-rst,rst,1,&rows));
  CHKERRQ(MatCreateSubMatrix(A,rows,NULL,MAT_INITIAL_MATRIX,&At));
  CHKERRQ(ISDestroy(&rows));

  if (ccomm != MPI_COMM_NULL) {
    Mat_MPIAIJ *a,*b;
    IS         from,to;
    Vec        gvec;
    PetscInt   lsize;

    CHKERRQ(MatCreate(ccomm,B));
    CHKERRQ(MatSetSizes(*B,ren-rst,PETSC_DECIDE,PETSC_DECIDE,At->cmap->N));
    CHKERRQ(MatSetType(*B,MATAIJ));
    CHKERRQ(PetscLayoutDestroy(&((*B)->rmap)));
    CHKERRQ(PetscLayoutSetUp((*B)->cmap));
    a    = (Mat_MPIAIJ*)At->data;
    b    = (Mat_MPIAIJ*)(*B)->data;
    CHKERRMPI(MPI_Comm_size(ccomm,&b->size));
    CHKERRMPI(MPI_Comm_rank(ccomm,&b->rank));
    CHKERRQ(PetscObjectReference((PetscObject)a->A));
    CHKERRQ(PetscObjectReference((PetscObject)a->B));
    b->A = a->A;
    b->B = a->B;

    b->donotstash      = a->donotstash;
    b->roworiented     = a->roworiented;
    b->rowindices      = NULL;
    b->rowvalues       = NULL;
    b->getrowactive    = PETSC_FALSE;

    (*B)->rmap         = rmap;
    (*B)->factortype   = A->factortype;
    (*B)->assembled    = PETSC_TRUE;
    (*B)->insertmode   = NOT_SET_VALUES;
    (*B)->preallocated = PETSC_TRUE;

    if (a->colmap) {
#if defined(PETSC_USE_CTABLE)
      CHKERRQ(PetscTableCreateCopy(a->colmap,&b->colmap));
#else
      CHKERRQ(PetscMalloc1(At->cmap->N,&b->colmap));
      CHKERRQ(PetscLogObjectMemory((PetscObject)*B,At->cmap->N*sizeof(PetscInt)));
      CHKERRQ(PetscArraycpy(b->colmap,a->colmap,At->cmap->N));
#endif
    } else b->colmap = NULL;
    if (a->garray) {
      PetscInt len;
      len  = a->B->cmap->n;
      CHKERRQ(PetscMalloc1(len+1,&b->garray));
      CHKERRQ(PetscLogObjectMemory((PetscObject)(*B),len*sizeof(PetscInt)));
      if (len) CHKERRQ(PetscArraycpy(b->garray,a->garray,len));
    } else b->garray = NULL;

    CHKERRQ(PetscObjectReference((PetscObject)a->lvec));
    b->lvec = a->lvec;
    CHKERRQ(PetscLogObjectParent((PetscObject)*B,(PetscObject)b->lvec));

    /* cannot use VecScatterCopy */
    CHKERRQ(VecGetLocalSize(b->lvec,&lsize));
    CHKERRQ(ISCreateGeneral(ccomm,lsize,b->garray,PETSC_USE_POINTER,&from));
    CHKERRQ(ISCreateStride(PETSC_COMM_SELF,lsize,0,1,&to));
    CHKERRQ(MatCreateVecs(*B,&gvec,NULL));
    CHKERRQ(VecScatterCreate(gvec,from,b->lvec,to,&b->Mvctx));
    CHKERRQ(PetscLogObjectParent((PetscObject)*B,(PetscObject)b->Mvctx));
    CHKERRQ(ISDestroy(&from));
    CHKERRQ(ISDestroy(&to));
    CHKERRQ(VecDestroy(&gvec));
  }
  CHKERRQ(MatDestroy(&At));
  PetscFunctionReturn(0);
}
