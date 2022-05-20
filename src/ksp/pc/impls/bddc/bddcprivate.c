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
  PetscCall(MatGetSize(A,&nr,&nc));
  if (!nr || !nc) PetscFunctionReturn(0);

  /* workspace */
  if (!work) {
    ulw  = PetscMax(PetscMax(1,5*PetscMin(nr,nc)),3*PetscMin(nr,nc)+PetscMax(nr,nc));
    PetscCall(PetscMalloc1(ulw,&uwork));
  } else {
    ulw   = lw;
    uwork = work;
  }
  n = PetscMin(nr,nc);
  if (!rwork) {
    PetscCall(PetscMalloc1(n,&sing));
  } else {
    sing = rwork;
  }

  /* SVD */
  PetscCall(PetscMalloc1(nr*nr,&U));
  PetscCall(PetscBLASIntCast(nr,&bM));
  PetscCall(PetscBLASIntCast(nc,&bN));
  PetscCall(PetscBLASIntCast(ulw,&lwork));
  PetscCall(MatDenseGetArray(A,&data));
  PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
#if !defined(PETSC_USE_COMPLEX)
  PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("A","N",&bM,&bN,data,&bM,sing,U,&bM,&ds,&di,uwork,&lwork,&lierr));
#else
  PetscCall(PetscMalloc1(5*n,&rwork2));
  PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("A","N",&bM,&bN,data,&bM,sing,U,&bM,&ds,&di,uwork,&lwork,rwork2,&lierr));
  PetscCall(PetscFree(rwork2));
#endif
  PetscCall(PetscFPTrapPop());
  PetscCheck(!lierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in GESVD Lapack routine %d",(int)lierr);
  PetscCall(MatDenseRestoreArray(A,&data));
  for (i=0;i<n;i++) if (sing[i] < PETSC_SMALL) break;
  if (!rwork) {
    PetscCall(PetscFree(sing));
  }
  if (!work) {
    PetscCall(PetscFree(uwork));
  }
  /* create B */
  if (!range) {
    PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,nr,nr-i,NULL,B));
    PetscCall(MatDenseGetArray(*B,&data));
    PetscCall(PetscArraycpy(data,U+nr*i,(nr-i)*nr));
  } else {
    PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,nr,i,NULL,B));
    PetscCall(MatDenseGetArray(*B,&data));
    PetscCall(PetscArraycpy(data,U,i*nr));
  }
  PetscCall(MatDenseRestoreArray(*B,&data));
  PetscCall(PetscFree(U));
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
  PetscCall(ISGetSize(edge,&esize));
  if (!esize) PetscFunctionReturn(0);
  PetscCall(ISGetSize(extrow,&rsize));
  PetscCall(ISGetSize(extcol,&csize));

  /* gradients */
  ptr  = work + 5*esize;
  PetscCall(MatCreateSubMatrix(lG,extrow,extcol,MAT_INITIAL_MATRIX,&GE));
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,rsize,csize,ptr,Gins));
  PetscCall(MatConvert(GE,MATSEQDENSE,MAT_REUSE_MATRIX,Gins));
  PetscCall(MatDestroy(&GE));

  /* constants */
  ptr += rsize*csize;
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,esize,csize,ptr,&GEd));
  PetscCall(MatCreateSubMatrix(lG,edge,extcol,MAT_INITIAL_MATRIX,&GE));
  PetscCall(MatConvert(GE,MATSEQDENSE,MAT_REUSE_MATRIX,&GEd));
  PetscCall(MatDestroy(&GE));
  PetscCall(MatDenseOrthogonalRangeOrComplement(GEd,PETSC_FALSE,5*esize,work,rwork,GKins));
  PetscCall(MatDestroy(&GEd));

  if (corners) {
    Mat               GEc;
    const PetscScalar *vals;
    PetscScalar       v;

    PetscCall(MatCreateSubMatrix(lG,edge,corners,MAT_INITIAL_MATRIX,&GEc));
    PetscCall(MatTransposeMatMult(GEc,*GKins,MAT_INITIAL_MATRIX,1.0,&GEd));
    PetscCall(MatDenseGetArrayRead(GEd,&vals));
    /* v    = PetscAbsScalar(vals[0]) */;
    v    = 1.;
    cvals[0] = vals[0]/v;
    cvals[1] = vals[1]/v;
    PetscCall(MatDenseRestoreArrayRead(GEd,&vals));
    PetscCall(MatScale(*GKins,1./v));
#if defined(PRINT_GDET)
    {
      PetscViewer viewer;
      char filename[256];
      sprintf(filename,"Gdet_l%d_r%d_cc%d.m",lev,PetscGlobalRank,inc++);
      PetscCall(PetscViewerASCIIOpen(PETSC_COMM_SELF,filename,&viewer));
      PetscCall(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB));
      PetscCall(PetscObjectSetName((PetscObject)GEc,"GEc"));
      PetscCall(MatView(GEc,viewer));
      PetscCall(PetscObjectSetName((PetscObject)(*GKins),"GK"));
      PetscCall(MatView(*GKins,viewer));
      PetscCall(PetscObjectSetName((PetscObject)GEd,"Gproj"));
      PetscCall(MatView(GEd,viewer));
      PetscCall(PetscViewerDestroy(&viewer));
    }
#endif
    PetscCall(MatDestroy(&GEd));
    PetscCall(MatDestroy(&GEc));
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
  PetscOptionsBegin(PetscObjectComm((PetscObject)pc),((PetscObject)pc)->prefix,"BDDC Nedelec options","PC");
  PetscCall(PetscOptionsBool("-pc_bddc_nedelec_field_primal","All edge dofs set as primals: Toselli's algorithm C",NULL,setprimal,&setprimal,NULL));
  PetscCall(PetscOptionsBool("-pc_bddc_nedelec_singular","Infer nullspace from discrete gradient",NULL,singular,&singular,NULL));
  PetscCall(PetscOptionsInt("-pc_bddc_nedelec_order","Test variable order code (to be removed)",NULL,order,&order,NULL));
  /* print debug info TODO: to be removed */
  PetscCall(PetscOptionsBool("-pc_bddc_nedelec_print","Print debug info",NULL,print,&print,NULL));
  PetscOptionsEnd();

  /* Return if there are no edges in the decomposition and the problem is not singular */
  PetscCall(MatISGetLocalToGlobalMapping(pc->pmat,&al2g,NULL));
  PetscCall(ISLocalToGlobalMappingGetSize(al2g,&n));
  PetscCall(PetscObjectGetComm((PetscObject)pc,&comm));
  if (!singular) {
    PetscCall(VecGetArrayRead(matis->counter,(const PetscScalar**)&vals));
    lrc[0] = PETSC_FALSE;
    for (i=0;i<n;i++) {
      if (PetscRealPart(vals[i]) > 2.) {
        lrc[0] = PETSC_TRUE;
        break;
      }
    }
    PetscCall(VecRestoreArrayRead(matis->counter,(const PetscScalar**)&vals));
    PetscCall(MPIU_Allreduce(&lrc[0],&lrc[1],1,MPIU_BOOL,MPI_LOR,comm));
    if (!lrc[1]) PetscFunctionReturn(0);
  }

  /* Get Nedelec field */
  PetscCheck(!pcbddc->n_ISForDofsLocal || field < pcbddc->n_ISForDofsLocal,comm,PETSC_ERR_USER,"Invalid field for Nedelec %" PetscInt_FMT ": number of fields is %" PetscInt_FMT,field,pcbddc->n_ISForDofsLocal);
  if (pcbddc->n_ISForDofsLocal && field >= 0) {
    PetscCall(PetscObjectReference((PetscObject)pcbddc->ISForDofsLocal[field]));
    nedfieldlocal = pcbddc->ISForDofsLocal[field];
    PetscCall(ISGetLocalSize(nedfieldlocal,&ne));
  } else if (!pcbddc->n_ISForDofsLocal && field != PETSC_DECIDE) {
    ne            = n;
    nedfieldlocal = NULL;
    global        = PETSC_TRUE;
  } else if (field == PETSC_DECIDE) {
    PetscInt rst,ren,*idx;

    PetscCall(PetscArrayzero(matis->sf_leafdata,n));
    PetscCall(PetscArrayzero(matis->sf_rootdata,pc->pmat->rmap->n));
    PetscCall(MatGetOwnershipRange(pcbddc->discretegradient,&rst,&ren));
    for (i=rst;i<ren;i++) {
      PetscInt nc;

      PetscCall(MatGetRow(pcbddc->discretegradient,i,&nc,NULL,NULL));
      if (nc > 1) matis->sf_rootdata[i-rst] = 1;
      PetscCall(MatRestoreRow(pcbddc->discretegradient,i,&nc,NULL,NULL));
    }
    PetscCall(PetscSFBcastBegin(matis->sf,MPIU_INT,matis->sf_rootdata,matis->sf_leafdata,MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(matis->sf,MPIU_INT,matis->sf_rootdata,matis->sf_leafdata,MPI_REPLACE));
    PetscCall(PetscMalloc1(n,&idx));
    for (i=0,ne=0;i<n;i++) if (matis->sf_leafdata[i]) idx[ne++] = i;
    PetscCall(ISCreateGeneral(comm,ne,idx,PETSC_OWN_POINTER,&nedfieldlocal));
  } else {
    SETERRQ(comm,PETSC_ERR_USER,"When multiple fields are present, the Nedelec field has to be specified");
  }

  /* Sanity checks */
  PetscCheck(order || conforming,comm,PETSC_ERR_SUP,"Variable order and non-conforming spaces are not supported at the same time");
  PetscCheck(!pcbddc->user_ChangeOfBasisMatrix,comm,PETSC_ERR_SUP,"Cannot generate Nedelec support with user defined change of basis");
  PetscCheck(!order || (ne%order == 0),PETSC_COMM_SELF,PETSC_ERR_USER,"The number of local edge dofs %" PetscInt_FMT " is not a multiple of the order %" PetscInt_FMT,ne,order);

  /* Just set primal dofs and return */
  if (setprimal) {
    IS       enedfieldlocal;
    PetscInt *eidxs;

    PetscCall(PetscMalloc1(ne,&eidxs));
    PetscCall(VecGetArrayRead(matis->counter,(const PetscScalar**)&vals));
    if (nedfieldlocal) {
      PetscCall(ISGetIndices(nedfieldlocal,&idxs));
      for (i=0,cum=0;i<ne;i++) {
        if (PetscRealPart(vals[idxs[i]]) > 2.) {
          eidxs[cum++] = idxs[i];
        }
      }
      PetscCall(ISRestoreIndices(nedfieldlocal,&idxs));
    } else {
      for (i=0,cum=0;i<ne;i++) {
        if (PetscRealPart(vals[i]) > 2.) {
          eidxs[cum++] = i;
        }
      }
    }
    PetscCall(VecRestoreArrayRead(matis->counter,(const PetscScalar**)&vals));
    PetscCall(ISCreateGeneral(comm,cum,eidxs,PETSC_COPY_VALUES,&enedfieldlocal));
    PetscCall(PCBDDCSetPrimalVerticesLocalIS(pc,enedfieldlocal));
    PetscCall(PetscFree(eidxs));
    PetscCall(ISDestroy(&nedfieldlocal));
    PetscCall(ISDestroy(&enedfieldlocal));
    PetscFunctionReturn(0);
  }

  /* Compute some l2g maps */
  if (nedfieldlocal) {
    IS is;

    /* need to map from the local Nedelec field to local numbering */
    PetscCall(ISLocalToGlobalMappingCreateIS(nedfieldlocal,&fl2g));
    /* need to map from the local Nedelec field to global numbering for the whole dofs*/
    PetscCall(ISLocalToGlobalMappingApplyIS(al2g,nedfieldlocal,&is));
    PetscCall(ISLocalToGlobalMappingCreateIS(is,&al2g));
    /* need to map from the local Nedelec field to global numbering (for Nedelec only) */
    if (global) {
      PetscCall(PetscObjectReference((PetscObject)al2g));
      el2g = al2g;
    } else {
      IS gis;

      PetscCall(ISRenumber(is,NULL,NULL,&gis));
      PetscCall(ISLocalToGlobalMappingCreateIS(gis,&el2g));
      PetscCall(ISDestroy(&gis));
    }
    PetscCall(ISDestroy(&is));
  } else {
    /* restore default */
    pcbddc->nedfield = -1;
    /* one ref for the destruction of al2g, one for el2g */
    PetscCall(PetscObjectReference((PetscObject)al2g));
    PetscCall(PetscObjectReference((PetscObject)al2g));
    el2g = al2g;
    fl2g = NULL;
  }

  /* Start communication to drop connections for interior edges (for cc analysis only) */
  PetscCall(PetscArrayzero(matis->sf_leafdata,n));
  PetscCall(PetscArrayzero(matis->sf_rootdata,pc->pmat->rmap->n));
  if (nedfieldlocal) {
    PetscCall(ISGetIndices(nedfieldlocal,&idxs));
    for (i=0;i<ne;i++) matis->sf_leafdata[idxs[i]] = 1;
    PetscCall(ISRestoreIndices(nedfieldlocal,&idxs));
  } else {
    for (i=0;i<ne;i++) matis->sf_leafdata[i] = 1;
  }
  PetscCall(PetscSFReduceBegin(matis->sf,MPIU_INT,matis->sf_leafdata,matis->sf_rootdata,MPI_SUM));
  PetscCall(PetscSFReduceEnd(matis->sf,MPIU_INT,matis->sf_leafdata,matis->sf_rootdata,MPI_SUM));

  if (!singular) { /* drop connections with interior edges to avoid unneeded communications and memory movements */
    PetscCall(MatDuplicate(pcbddc->discretegradient,MAT_COPY_VALUES,&G));
    PetscCall(MatSetOption(G,MAT_KEEP_NONZERO_PATTERN,PETSC_FALSE));
    if (global) {
      PetscInt rst;

      PetscCall(MatGetOwnershipRange(G,&rst,NULL));
      for (i=0,cum=0;i<pc->pmat->rmap->n;i++) {
        if (matis->sf_rootdata[i] < 2) {
          matis->sf_rootdata[cum++] = i + rst;
        }
      }
      PetscCall(MatSetOption(G,MAT_NO_OFF_PROC_ZERO_ROWS,PETSC_TRUE));
      PetscCall(MatZeroRows(G,cum,matis->sf_rootdata,0.,NULL,NULL));
    } else {
      PetscInt *tbz;

      PetscCall(PetscMalloc1(ne,&tbz));
      PetscCall(PetscSFBcastBegin(matis->sf,MPIU_INT,matis->sf_rootdata,matis->sf_leafdata,MPI_REPLACE));
      PetscCall(PetscSFBcastEnd(matis->sf,MPIU_INT,matis->sf_rootdata,matis->sf_leafdata,MPI_REPLACE));
      PetscCall(ISGetIndices(nedfieldlocal,&idxs));
      for (i=0,cum=0;i<ne;i++)
        if (matis->sf_leafdata[idxs[i]] == 1)
          tbz[cum++] = i;
      PetscCall(ISRestoreIndices(nedfieldlocal,&idxs));
      PetscCall(ISLocalToGlobalMappingApply(el2g,cum,tbz,tbz));
      PetscCall(MatZeroRows(G,cum,tbz,0.,NULL,NULL));
      PetscCall(PetscFree(tbz));
    }
  } else { /* we need the entire G to infer the nullspace */
    PetscCall(PetscObjectReference((PetscObject)pcbddc->discretegradient));
    G    = pcbddc->discretegradient;
  }

  /* Extract subdomain relevant rows of G */
  PetscCall(ISLocalToGlobalMappingGetIndices(el2g,&idxs));
  PetscCall(ISCreateGeneral(comm,ne,idxs,PETSC_USE_POINTER,&lned));
  PetscCall(MatCreateSubMatrix(G,lned,NULL,MAT_INITIAL_MATRIX,&lGall));
  PetscCall(ISLocalToGlobalMappingRestoreIndices(el2g,&idxs));
  PetscCall(ISDestroy(&lned));
  PetscCall(MatConvert(lGall,MATIS,MAT_INITIAL_MATRIX,&lGis));
  PetscCall(MatDestroy(&lGall));
  PetscCall(MatISGetLocalMat(lGis,&lG));

  /* SF for nodal dofs communications */
  PetscCall(MatGetLocalSize(G,NULL,&Lv));
  PetscCall(MatISGetLocalToGlobalMapping(lGis,NULL,&vl2g));
  PetscCall(PetscObjectReference((PetscObject)vl2g));
  PetscCall(ISLocalToGlobalMappingGetSize(vl2g,&nv));
  PetscCall(PetscSFCreate(comm,&sfv));
  PetscCall(ISLocalToGlobalMappingGetIndices(vl2g,&idxs));
  PetscCall(PetscSFSetGraphLayout(sfv,lGis->cmap,nv,NULL,PETSC_OWN_POINTER,idxs));
  PetscCall(ISLocalToGlobalMappingRestoreIndices(vl2g,&idxs));
  i    = singular ? 2 : 1;
  PetscCall(PetscMalloc2(i*nv,&sfvleaves,i*Lv,&sfvroots));

  /* Destroy temporary G created in MATIS format and modified G */
  PetscCall(PetscObjectReference((PetscObject)lG));
  PetscCall(MatDestroy(&lGis));
  PetscCall(MatDestroy(&G));

  if (print) {
    PetscCall(PetscObjectSetName((PetscObject)lG,"initial_lG"));
    PetscCall(MatView(lG,NULL));
  }

  /* Save lG for values insertion in change of basis */
  PetscCall(MatDuplicate(lG,MAT_COPY_VALUES,&lGinit));

  /* Analyze the edge-nodes connections (duplicate lG) */
  PetscCall(MatDuplicate(lG,MAT_COPY_VALUES,&lGe));
  PetscCall(MatSetOption(lGe,MAT_KEEP_NONZERO_PATTERN,PETSC_FALSE));
  PetscCall(PetscBTCreate(nv,&btv));
  PetscCall(PetscBTCreate(ne,&bte));
  PetscCall(PetscBTCreate(ne,&btb));
  PetscCall(PetscBTCreate(ne,&btbd));
  PetscCall(PetscBTCreate(nv,&btvcand));
  /* need to import the boundary specification to ensure the
     proper detection of coarse edges' endpoints */
  if (pcbddc->DirichletBoundariesLocal) {
    IS is;

    if (fl2g) {
      PetscCall(ISGlobalToLocalMappingApplyIS(fl2g,IS_GTOLM_MASK,pcbddc->DirichletBoundariesLocal,&is));
    } else {
      is = pcbddc->DirichletBoundariesLocal;
    }
    PetscCall(ISGetLocalSize(is,&cum));
    PetscCall(ISGetIndices(is,&idxs));
    for (i=0;i<cum;i++) {
      if (idxs[i] >= 0) {
        PetscCall(PetscBTSet(btb,idxs[i]));
        PetscCall(PetscBTSet(btbd,idxs[i]));
      }
    }
    PetscCall(ISRestoreIndices(is,&idxs));
    if (fl2g) {
      PetscCall(ISDestroy(&is));
    }
  }
  if (pcbddc->NeumannBoundariesLocal) {
    IS is;

    if (fl2g) {
      PetscCall(ISGlobalToLocalMappingApplyIS(fl2g,IS_GTOLM_MASK,pcbddc->NeumannBoundariesLocal,&is));
    } else {
      is = pcbddc->NeumannBoundariesLocal;
    }
    PetscCall(ISGetLocalSize(is,&cum));
    PetscCall(ISGetIndices(is,&idxs));
    for (i=0;i<cum;i++) {
      if (idxs[i] >= 0) {
        PetscCall(PetscBTSet(btb,idxs[i]));
      }
    }
    PetscCall(ISRestoreIndices(is,&idxs));
    if (fl2g) {
      PetscCall(ISDestroy(&is));
    }
  }

  /* Count neighs per dof */
  PetscCall(ISLocalToGlobalMappingGetNodeInfo(el2g,NULL,&ecount,&eneighs));
  PetscCall(ISLocalToGlobalMappingGetNodeInfo(vl2g,NULL,&vcount,&vneighs));

  /* need to remove coarse faces' dofs and coarse edges' dirichlet dofs
     for proper detection of coarse edges' endpoints */
  PetscCall(PetscBTCreate(ne,&btee));
  for (i=0;i<ne;i++) {
    if ((ecount[i] > 2 && !PetscBTLookup(btbd,i)) || (ecount[i] == 2 && PetscBTLookup(btb,i))) {
      PetscCall(PetscBTSet(btee,i));
    }
  }
  PetscCall(PetscMalloc1(ne,&marks));
  if (!conforming) {
    PetscCall(MatTranspose(lGe,MAT_INITIAL_MATRIX,&lGt));
    PetscCall(MatGetRowIJ(lGt,0,PETSC_FALSE,PETSC_FALSE,&i,&iit,&jjt,&done));
  }
  PetscCall(MatGetRowIJ(lGe,0,PETSC_FALSE,PETSC_FALSE,&i,&ii,&jj,&done));
  PetscCall(MatSeqAIJGetArray(lGe,&vals));
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
        PetscCall(PetscBTSet(bte,i));
        for (j=ii[i];j<ii[i+1];j++) {
          PetscCall(PetscBTSet(btv,jj[j]));
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
          PetscCall(PetscBTSet(bte,i));
          for (j=ii[i];j<ii[i+1];j++) {
            PetscCall(PetscBTSet(btv,jj[j]));
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
  PetscCall(PetscBTDestroy(&btee));
  PetscCall(MatSeqAIJRestoreArray(lGe,&vals));
  PetscCall(MatRestoreRowIJ(lGe,0,PETSC_FALSE,PETSC_FALSE,&i,&ii,&jj,&done));
  if (!conforming) {
    PetscCall(MatRestoreRowIJ(lGt,0,PETSC_FALSE,PETSC_FALSE,&i,&iit,&jjt,&done));
    PetscCall(MatDestroy(&lGt));
  }
  PetscCall(MatZeroRows(lGe,cum,marks,0.,NULL,NULL));

  /* identify splitpoints and corner candidates */
  PetscCall(MatTranspose(lGe,MAT_INITIAL_MATRIX,&lGt));
  if (print) {
    PetscCall(PetscObjectSetName((PetscObject)lGe,"edgerestr_lG"));
    PetscCall(MatView(lGe,NULL));
    PetscCall(PetscObjectSetName((PetscObject)lGt,"edgerestr_lGt"));
    PetscCall(MatView(lGt,NULL));
  }
  PetscCall(MatGetRowIJ(lGt,0,PETSC_FALSE,PETSC_FALSE,&i,&ii,&jj,&done));
  PetscCall(MatSeqAIJGetArray(lGt,&vals));
  for (i=0;i<nv;i++) {
    PetscInt  ord = order, test = ii[i+1]-ii[i], vc = vcount[i];
    PetscBool sneighs = PETSC_TRUE, bdir = PETSC_FALSE;
    if (!order) { /* variable order */
      PetscReal vorder = 0.;

      for (j=ii[i];j<ii[i+1];j++) vorder += PetscRealPart(vals[j]);
      test = PetscFloorReal(vorder+10.*PETSC_SQRT_MACHINE_EPSILON);
      PetscCheck(vorder-test <= PETSC_SQRT_MACHINE_EPSILON,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unexpected value for vorder: %g (%" PetscInt_FMT ")",(double)vorder,test);
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
      if (print) PetscPrintf(PETSC_COMM_SELF,"SPLITPOINT %" PetscInt_FMT " (%s %s %s)\n",i,PetscBools[!sneighs],PetscBools[test >= 3*ord],PetscBools[bdir]);
      PetscCall(PetscBTSet(btv,i));
    } else if (test == ord) {
      if (order == 1 || (!order && ii[i+1]-ii[i] == 1)) {
        if (print) PetscPrintf(PETSC_COMM_SELF,"ENDPOINT %" PetscInt_FMT "\n",i);
        PetscCall(PetscBTSet(btv,i));
      } else {
        if (print) PetscPrintf(PETSC_COMM_SELF,"CORNER CANDIDATE %" PetscInt_FMT "\n",i);
        PetscCall(PetscBTSet(btvcand,i));
      }
    }
  }
  PetscCall(ISLocalToGlobalMappingRestoreNodeInfo(el2g,NULL,&ecount,&eneighs));
  PetscCall(ISLocalToGlobalMappingRestoreNodeInfo(vl2g,NULL,&vcount,&vneighs));
  PetscCall(PetscBTDestroy(&btbd));

  /* a candidate is valid if it is connected to another candidate via a non-primal edge dof */
  if (order != 1) {
    if (print) PetscPrintf(PETSC_COMM_SELF,"INSPECTING CANDIDATES\n");
    PetscCall(MatGetRowIJ(lGe,0,PETSC_FALSE,PETSC_FALSE,&i,&iit,&jjt,&done));
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
          if (print) PetscPrintf(PETSC_COMM_SELF,"  CANDIDATE %" PetscInt_FMT " CLEARED\n",i);
          PetscCall(PetscBTClear(btvcand,i));
        } else {
          if (print) PetscPrintf(PETSC_COMM_SELF,"  CANDIDATE %" PetscInt_FMT " ACCEPTED\n",i);
        }
      }
    }
    PetscCall(MatRestoreRowIJ(lGe,0,PETSC_FALSE,PETSC_FALSE,&i,&iit,&jjt,&done));
  }
  PetscCall(MatSeqAIJRestoreArray(lGt,&vals));
  PetscCall(MatRestoreRowIJ(lGt,0,PETSC_FALSE,PETSC_FALSE,&i,&ii,&jj,&done));
  PetscCall(MatDestroy(&lGe));

  /* Get the local G^T explicitly */
  PetscCall(MatDestroy(&lGt));
  PetscCall(MatTranspose(lG,MAT_INITIAL_MATRIX,&lGt));
  PetscCall(MatSetOption(lGt,MAT_KEEP_NONZERO_PATTERN,PETSC_FALSE));

  /* Mark interior nodal dofs */
  PetscCall(ISLocalToGlobalMappingGetInfo(vl2g,&n_neigh,&neigh,&n_shared,&shared));
  PetscCall(PetscBTCreate(nv,&btvi));
  for (i=1;i<n_neigh;i++) {
    for (j=0;j<n_shared[i];j++) {
      PetscCall(PetscBTSet(btvi,shared[i][j]));
    }
  }
  PetscCall(ISLocalToGlobalMappingRestoreInfo(vl2g,&n_neigh,&neigh,&n_shared,&shared));

  /* communicate corners and splitpoints */
  PetscCall(PetscMalloc1(nv,&vmarks));
  PetscCall(PetscArrayzero(sfvleaves,nv));
  PetscCall(PetscArrayzero(sfvroots,Lv));
  for (i=0;i<nv;i++) if (PetscUnlikely(PetscBTLookup(btv,i))) sfvleaves[i] = 1;

  if (print) {
    IS tbz;

    cum = 0;
    for (i=0;i<nv;i++)
      if (sfvleaves[i])
        vmarks[cum++] = i;

    PetscCall(ISCreateGeneral(PETSC_COMM_SELF,cum,vmarks,PETSC_COPY_VALUES,&tbz));
    PetscCall(PetscObjectSetName((PetscObject)tbz,"corners_to_be_zeroed_local"));
    PetscCall(ISView(tbz,NULL));
    PetscCall(ISDestroy(&tbz));
  }

  PetscCall(PetscSFReduceBegin(sfv,MPIU_INT,sfvleaves,sfvroots,MPI_SUM));
  PetscCall(PetscSFReduceEnd(sfv,MPIU_INT,sfvleaves,sfvroots,MPI_SUM));
  PetscCall(PetscSFBcastBegin(sfv,MPIU_INT,sfvroots,sfvleaves,MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(sfv,MPIU_INT,sfvroots,sfvleaves,MPI_REPLACE));

  /* Zero rows of lGt corresponding to identified corners
     and interior nodal dofs */
  cum = 0;
  for (i=0;i<nv;i++) {
    if (sfvleaves[i]) {
      vmarks[cum++] = i;
      PetscCall(PetscBTSet(btv,i));
    }
    if (!PetscBTLookup(btvi,i)) vmarks[cum++] = i;
  }
  PetscCall(PetscBTDestroy(&btvi));
  if (print) {
    IS tbz;

    PetscCall(ISCreateGeneral(PETSC_COMM_SELF,cum,vmarks,PETSC_COPY_VALUES,&tbz));
    PetscCall(PetscObjectSetName((PetscObject)tbz,"corners_to_be_zeroed_with_interior"));
    PetscCall(ISView(tbz,NULL));
    PetscCall(ISDestroy(&tbz));
  }
  PetscCall(MatZeroRows(lGt,cum,vmarks,0.,NULL,NULL));
  PetscCall(PetscFree(vmarks));
  PetscCall(PetscSFDestroy(&sfv));
  PetscCall(PetscFree2(sfvleaves,sfvroots));

  /* Recompute G */
  PetscCall(MatDestroy(&lG));
  PetscCall(MatTranspose(lGt,MAT_INITIAL_MATRIX,&lG));
  if (print) {
    PetscCall(PetscObjectSetName((PetscObject)lG,"used_lG"));
    PetscCall(MatView(lG,NULL));
    PetscCall(PetscObjectSetName((PetscObject)lGt,"used_lGt"));
    PetscCall(MatView(lGt,NULL));
  }

  /* Get primal dofs (if any) */
  cum = 0;
  for (i=0;i<ne;i++) {
    if (PetscUnlikely(PetscBTLookup(bte,i))) marks[cum++] = i;
  }
  if (fl2g) {
    PetscCall(ISLocalToGlobalMappingApply(fl2g,cum,marks,marks));
  }
  PetscCall(ISCreateGeneral(comm,cum,marks,PETSC_COPY_VALUES,&primals));
  if (print) {
    PetscCall(PetscObjectSetName((PetscObject)primals,"prescribed_primal_dofs"));
    PetscCall(ISView(primals,NULL));
  }
  PetscCall(PetscBTDestroy(&bte));
  /* TODO: what if the user passed in some of them ?  */
  PetscCall(PCBDDCSetPrimalVerticesLocalIS(pc,primals));
  PetscCall(ISDestroy(&primals));

  /* Compute edge connectivity */
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)lG,"econn_"));

  /* Symbolic conn = lG*lGt */
  PetscCall(MatProductCreate(lG,lGt,NULL,&conn));
  PetscCall(MatProductSetType(conn,MATPRODUCT_AB));
  PetscCall(MatProductSetAlgorithm(conn,"default"));
  PetscCall(MatProductSetFill(conn,PETSC_DEFAULT));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)conn,"econn_"));
  PetscCall(MatProductSetFromOptions(conn));
  PetscCall(MatProductSymbolic(conn));

  PetscCall(MatGetRowIJ(conn,0,PETSC_FALSE,PETSC_FALSE,&i,&ii,&jj,&done));
  if (fl2g) {
    PetscBT   btf;
    PetscInt  *iia,*jja,*iiu,*jju;
    PetscBool rest = PETSC_FALSE,free = PETSC_FALSE;

    /* create CSR for all local dofs */
    PetscCall(PetscMalloc1(n+1,&iia));
    if (pcbddc->mat_graph->nvtxs_csr) { /* the user has passed in a CSR graph */
      PetscCheck(pcbddc->mat_graph->nvtxs_csr == n,PETSC_COMM_SELF,PETSC_ERR_USER,"Invalid size of CSR graph %" PetscInt_FMT ". Should be %" PetscInt_FMT,pcbddc->mat_graph->nvtxs_csr,n);
      iiu = pcbddc->mat_graph->xadj;
      jju = pcbddc->mat_graph->adjncy;
    } else if (pcbddc->use_local_adj) {
      rest = PETSC_TRUE;
      PetscCall(MatGetRowIJ(matis->A,0,PETSC_TRUE,PETSC_FALSE,&i,(const PetscInt**)&iiu,(const PetscInt**)&jju,&done));
    } else {
      free   = PETSC_TRUE;
      PetscCall(PetscMalloc2(n+1,&iiu,n,&jju));
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
    PetscCall(PetscBTCreate(n,&btf));
    PetscCall(ISGetIndices(nedfieldlocal,&idxs));
    for (i=0;i<ne;i++) {
      PetscCall(PetscBTSet(btf,idxs[i]));
      iia[idxs[i]+1] = ii[i+1]-ii[i];
    }

    /* iia in CSR */
    for (i=0;i<n;i++) iia[i+1] += iia[i];

    /* jja in CSR */
    PetscCall(PetscMalloc1(iia[n],&jja));
    for (i=0;i<n;i++)
      if (!PetscBTLookup(btf,i))
        for (j=0;j<iiu[i+1]-iiu[i];j++)
          jja[iia[i]+j] = jju[iiu[i]+j];

    /* map edge dofs connectivity */
    if (jj) {
      PetscCall(ISLocalToGlobalMappingApply(fl2g,ii[ne],jj,(PetscInt *)jj));
      for (i=0;i<ne;i++) {
        PetscInt e = idxs[i];
        for (j=0;j<ii[i+1]-ii[i];j++) jja[iia[e]+j] = jj[ii[i]+j];
      }
    }
    PetscCall(ISRestoreIndices(nedfieldlocal,&idxs));
    PetscCall(PCBDDCSetLocalAdjacencyGraph(pc,n,iia,jja,PETSC_OWN_POINTER));
    if (rest) {
      PetscCall(MatRestoreRowIJ(matis->A,0,PETSC_TRUE,PETSC_FALSE,&i,(const PetscInt**)&iiu,(const PetscInt**)&jju,&done));
    }
    if (free) {
      PetscCall(PetscFree2(iiu,jju));
    }
    PetscCall(PetscBTDestroy(&btf));
  } else {
    PetscCall(PCBDDCSetLocalAdjacencyGraph(pc,n,ii,jj,PETSC_USE_POINTER));
  }

  /* Analyze interface for edge dofs */
  PetscCall(PCBDDCAnalyzeInterface(pc));
  pcbddc->mat_graph->twodim = PETSC_FALSE;

  /* Get coarse edges in the edge space */
  PetscCall(PCBDDCGraphGetCandidatesIS(pcbddc->mat_graph,NULL,NULL,&nee,&alleedges,&allprimals));
  PetscCall(MatRestoreRowIJ(conn,0,PETSC_FALSE,PETSC_FALSE,&i,&ii,&jj,&done));

  if (fl2g) {
    PetscCall(ISGlobalToLocalMappingApplyIS(fl2g,IS_GTOLM_DROP,allprimals,&primals));
    PetscCall(PetscMalloc1(nee,&eedges));
    for (i=0;i<nee;i++) {
      PetscCall(ISGlobalToLocalMappingApplyIS(fl2g,IS_GTOLM_DROP,alleedges[i],&eedges[i]));
    }
  } else {
    eedges  = alleedges;
    primals = allprimals;
  }

  /* Mark fine edge dofs with their coarse edge id */
  PetscCall(PetscArrayzero(marks,ne));
  PetscCall(ISGetLocalSize(primals,&cum));
  PetscCall(ISGetIndices(primals,&idxs));
  for (i=0;i<cum;i++) marks[idxs[i]] = nee+1;
  PetscCall(ISRestoreIndices(primals,&idxs));
  if (print) {
    PetscCall(PetscObjectSetName((PetscObject)primals,"obtained_primal_dofs"));
    PetscCall(ISView(primals,NULL));
  }

  maxsize = 0;
  for (i=0;i<nee;i++) {
    PetscInt size,mark = i+1;

    PetscCall(ISGetLocalSize(eedges[i],&size));
    PetscCall(ISGetIndices(eedges[i],&idxs));
    for (j=0;j<size;j++) marks[idxs[j]] = mark;
    PetscCall(ISRestoreIndices(eedges[i],&idxs));
    maxsize = PetscMax(maxsize,size);
  }

  /* Find coarse edge endpoints */
  PetscCall(MatGetRowIJ(lG,0,PETSC_FALSE,PETSC_FALSE,&i,&ii,&jj,&done));
  PetscCall(MatGetRowIJ(lGt,0,PETSC_FALSE,PETSC_FALSE,&i,&iit,&jjt,&done));
  for (i=0;i<nee;i++) {
    PetscInt mark = i+1,size;

    PetscCall(ISGetLocalSize(eedges[i],&size));
    if (!size && nedfieldlocal) continue;
    PetscCheck(size,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unexpected zero sized edge %" PetscInt_FMT,i);
    PetscCall(ISGetIndices(eedges[i],&idxs));
    if (print) {
      PetscCall(PetscPrintf(PETSC_COMM_SELF,"ENDPOINTS ANALYSIS EDGE %" PetscInt_FMT "\n",i));
      PetscCall(ISView(eedges[i],NULL));
    }
    for (j=0;j<size;j++) {
      PetscInt k, ee = idxs[j];
      if (print) PetscPrintf(PETSC_COMM_SELF,"  idx %" PetscInt_FMT "\n",ee);
      for (k=ii[ee];k<ii[ee+1];k++) {
        if (print) PetscPrintf(PETSC_COMM_SELF,"    inspect %" PetscInt_FMT "\n",jj[k]);
        if (PetscBTLookup(btv,jj[k])) {
          if (print) PetscPrintf(PETSC_COMM_SELF,"      corner found (already set) %" PetscInt_FMT "\n",jj[k]);
        } else if (PetscBTLookup(btvcand,jj[k])) { /* is it ok? */
          PetscInt  k2;
          PetscBool corner = PETSC_FALSE;
          for (k2 = iit[jj[k]];k2 < iit[jj[k]+1];k2++) {
            if (print) PetscPrintf(PETSC_COMM_SELF,"        INSPECTING %" PetscInt_FMT ": mark %" PetscInt_FMT " (ref mark %" PetscInt_FMT "), boundary %d\n",jjt[k2],marks[jjt[k2]],mark,(int)!!PetscBTLookup(btb,jjt[k2]));
            /* it's a corner if either is connected with an edge dof belonging to a different cc or
               if the edge dof lie on the natural part of the boundary */
            if ((marks[jjt[k2]] && marks[jjt[k2]] != mark) || (!marks[jjt[k2]] && PetscBTLookup(btb,jjt[k2]))) {
              corner = PETSC_TRUE;
              break;
            }
          }
          if (corner) { /* found the nodal dof corresponding to the endpoint of the edge */
            if (print) PetscPrintf(PETSC_COMM_SELF,"        corner found %" PetscInt_FMT "\n",jj[k]);
            PetscCall(PetscBTSet(btv,jj[k]));
          } else {
            if (print) PetscPrintf(PETSC_COMM_SELF,"        no corners found\n");
          }
        }
      }
    }
    PetscCall(ISRestoreIndices(eedges[i],&idxs));
  }
  PetscCall(MatRestoreRowIJ(lGt,0,PETSC_FALSE,PETSC_FALSE,&i,&iit,&jjt,&done));
  PetscCall(MatRestoreRowIJ(lG,0,PETSC_FALSE,PETSC_FALSE,&i,&ii,&jj,&done));
  PetscCall(PetscBTDestroy(&btb));

  /* Reset marked primal dofs */
  PetscCall(ISGetLocalSize(primals,&cum));
  PetscCall(ISGetIndices(primals,&idxs));
  for (i=0;i<cum;i++) marks[idxs[i]] = 0;
  PetscCall(ISRestoreIndices(primals,&idxs));

  /* Now use the initial lG */
  PetscCall(MatDestroy(&lG));
  PetscCall(MatDestroy(&lGt));
  lG   = lGinit;
  PetscCall(MatTranspose(lG,MAT_INITIAL_MATRIX,&lGt));

  /* Compute extended cols indices */
  PetscCall(PetscBTCreate(nv,&btvc));
  PetscCall(PetscBTCreate(nee,&bter));
  PetscCall(MatGetRowIJ(lG,0,PETSC_FALSE,PETSC_FALSE,&i,&ii,&jj,&done));
  PetscCall(MatSeqAIJGetMaxRowNonzeros(lG,&i));
  i   *= maxsize;
  PetscCall(PetscCalloc1(nee,&extcols));
  PetscCall(PetscMalloc2(i,&extrow,i,&gidxs));
  eerr = PETSC_FALSE;
  for (i=0;i<nee;i++) {
    PetscInt size,found = 0;

    cum  = 0;
    PetscCall(ISGetLocalSize(eedges[i],&size));
    if (!size && nedfieldlocal) continue;
    PetscCheck(size,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unexpected zero sized edge %" PetscInt_FMT,i);
    PetscCall(ISGetIndices(eedges[i],&idxs));
    PetscCall(PetscBTMemzero(nv,btvc));
    for (j=0;j<size;j++) {
      PetscInt k,ee = idxs[j];
      for (k=ii[ee];k<ii[ee+1];k++) {
        PetscInt vv = jj[k];
        if (!PetscBTLookup(btv,vv)) extrow[cum++] = vv;
        else if (!PetscBTLookupSet(btvc,vv)) found++;
      }
    }
    PetscCall(ISRestoreIndices(eedges[i],&idxs));
    PetscCall(PetscSortRemoveDupsInt(&cum,extrow));
    PetscCall(ISLocalToGlobalMappingApply(vl2g,cum,extrow,gidxs));
    PetscCall(PetscSortIntWithArray(cum,gidxs,extrow));
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF,cum,extrow,PETSC_COPY_VALUES,&extcols[i]));
    /* it may happen that endpoints are not defined at this point
       if it is the case, mark this edge for a second pass */
    if (cum != size -1 || found != 2) {
      PetscCall(PetscBTSet(bter,i));
      if (print) {
        PetscCall(PetscObjectSetName((PetscObject)eedges[i],"error_edge"));
        PetscCall(ISView(eedges[i],NULL));
        PetscCall(PetscObjectSetName((PetscObject)extcols[i],"error_extcol"));
        PetscCall(ISView(extcols[i],NULL));
      }
      eerr = PETSC_TRUE;
    }
  }
  /* PetscCheck(!eerr,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unexpected SIZE OF EDGE > EXTCOL FIRST PASS"); */
  PetscCall(MPIU_Allreduce(&eerr,&done,1,MPIU_BOOL,MPI_LOR,comm));
  if (done) {
    PetscInt *newprimals;

    PetscCall(PetscMalloc1(ne,&newprimals));
    PetscCall(ISGetLocalSize(primals,&cum));
    PetscCall(ISGetIndices(primals,&idxs));
    PetscCall(PetscArraycpy(newprimals,idxs,cum));
    PetscCall(ISRestoreIndices(primals,&idxs));
    PetscCall(MatGetRowIJ(lGt,0,PETSC_FALSE,PETSC_FALSE,&i,&iit,&jjt,&done));
    if (print) PetscPrintf(PETSC_COMM_SELF,"DOING SECOND PASS (eerr %s)\n",PetscBools[eerr]);
    for (i=0;i<nee;i++) {
      PetscBool has_candidates = PETSC_FALSE;
      if (PetscBTLookup(bter,i)) {
        PetscInt size,mark = i+1;

        PetscCall(ISGetLocalSize(eedges[i],&size));
        PetscCall(ISGetIndices(eedges[i],&idxs));
        /* for (j=0;j<size;j++) newprimals[cum++] = idxs[j]; */
        for (j=0;j<size;j++) {
          PetscInt k,ee = idxs[j];
          if (print) PetscPrintf(PETSC_COMM_SELF,"Inspecting edge dof %" PetscInt_FMT " [%" PetscInt_FMT " %" PetscInt_FMT ")\n",ee,ii[ee],ii[ee+1]);
          for (k=ii[ee];k<ii[ee+1];k++) {
            /* set all candidates located on the edge as corners */
            if (PetscBTLookup(btvcand,jj[k])) {
              PetscInt k2,vv = jj[k];
              has_candidates = PETSC_TRUE;
              if (print) PetscPrintf(PETSC_COMM_SELF,"  Candidate set to vertex %" PetscInt_FMT "\n",vv);
              PetscCall(PetscBTSet(btv,vv));
              /* set all edge dofs connected to candidate as primals */
              for (k2=iit[vv];k2<iit[vv+1];k2++) {
                if (marks[jjt[k2]] == mark) {
                  PetscInt k3,ee2 = jjt[k2];
                  if (print) PetscPrintf(PETSC_COMM_SELF,"    Connected edge dof set to primal %" PetscInt_FMT "\n",ee2);
                  newprimals[cum++] = ee2;
                  /* finally set the new corners */
                  for (k3=ii[ee2];k3<ii[ee2+1];k3++) {
                    if (print) PetscPrintf(PETSC_COMM_SELF,"      Connected nodal dof set to vertex %" PetscInt_FMT "\n",jj[k3]);
                    PetscCall(PetscBTSet(btv,jj[k3]));
                  }
                }
              }
            } else {
              if (print) PetscPrintf(PETSC_COMM_SELF,"  Not a candidate vertex %" PetscInt_FMT "\n",jj[k]);
            }
          }
        }
        if (!has_candidates) { /* circular edge */
          PetscInt k, ee = idxs[0],*tmarks;

          PetscCall(PetscCalloc1(ne,&tmarks));
          if (print) PetscPrintf(PETSC_COMM_SELF,"  Circular edge %" PetscInt_FMT "\n",i);
          for (k=ii[ee];k<ii[ee+1];k++) {
            PetscInt k2;
            if (print) PetscPrintf(PETSC_COMM_SELF,"    Set to corner %" PetscInt_FMT "\n",jj[k]);
            PetscCall(PetscBTSet(btv,jj[k]));
            for (k2=iit[jj[k]];k2<iit[jj[k]+1];k2++) tmarks[jjt[k2]]++;
          }
          for (j=0;j<size;j++) {
            if (tmarks[idxs[j]] > 1) {
              if (print) PetscPrintf(PETSC_COMM_SELF,"  Edge dof set to primal %" PetscInt_FMT "\n",idxs[j]);
              newprimals[cum++] = idxs[j];
            }
          }
          PetscCall(PetscFree(tmarks));
        }
        PetscCall(ISRestoreIndices(eedges[i],&idxs));
      }
      PetscCall(ISDestroy(&extcols[i]));
    }
    PetscCall(PetscFree(extcols));
    PetscCall(MatRestoreRowIJ(lGt,0,PETSC_FALSE,PETSC_FALSE,&i,&iit,&jjt,&done));
    PetscCall(PetscSortRemoveDupsInt(&cum,newprimals));
    if (fl2g) {
      PetscCall(ISLocalToGlobalMappingApply(fl2g,cum,newprimals,newprimals));
      PetscCall(ISDestroy(&primals));
      for (i=0;i<nee;i++) {
        PetscCall(ISDestroy(&eedges[i]));
      }
      PetscCall(PetscFree(eedges));
    }
    PetscCall(PCBDDCGraphRestoreCandidatesIS(pcbddc->mat_graph,NULL,NULL,&nee,&alleedges,&allprimals));
    PetscCall(ISCreateGeneral(comm,cum,newprimals,PETSC_COPY_VALUES,&primals));
    PetscCall(PetscFree(newprimals));
    PetscCall(PCBDDCSetPrimalVerticesLocalIS(pc,primals));
    PetscCall(ISDestroy(&primals));
    PetscCall(PCBDDCAnalyzeInterface(pc));
    pcbddc->mat_graph->twodim = PETSC_FALSE;
    PetscCall(PCBDDCGraphGetCandidatesIS(pcbddc->mat_graph,NULL,NULL,&nee,&alleedges,&allprimals));
    if (fl2g) {
      PetscCall(ISGlobalToLocalMappingApplyIS(fl2g,IS_GTOLM_DROP,allprimals,&primals));
      PetscCall(PetscMalloc1(nee,&eedges));
      for (i=0;i<nee;i++) {
        PetscCall(ISGlobalToLocalMappingApplyIS(fl2g,IS_GTOLM_DROP,alleedges[i],&eedges[i]));
      }
    } else {
      eedges  = alleedges;
      primals = allprimals;
    }
    PetscCall(PetscCalloc1(nee,&extcols));

    /* Mark again */
    PetscCall(PetscArrayzero(marks,ne));
    for (i=0;i<nee;i++) {
      PetscInt size,mark = i+1;

      PetscCall(ISGetLocalSize(eedges[i],&size));
      PetscCall(ISGetIndices(eedges[i],&idxs));
      for (j=0;j<size;j++) marks[idxs[j]] = mark;
      PetscCall(ISRestoreIndices(eedges[i],&idxs));
    }
    if (print) {
      PetscCall(PetscObjectSetName((PetscObject)primals,"obtained_primal_dofs_secondpass"));
      PetscCall(ISView(primals,NULL));
    }

    /* Recompute extended cols */
    eerr = PETSC_FALSE;
    for (i=0;i<nee;i++) {
      PetscInt size;

      cum  = 0;
      PetscCall(ISGetLocalSize(eedges[i],&size));
      if (!size && nedfieldlocal) continue;
      PetscCheck(size,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unexpected zero sized edge %" PetscInt_FMT,i);
      PetscCall(ISGetIndices(eedges[i],&idxs));
      for (j=0;j<size;j++) {
        PetscInt k,ee = idxs[j];
        for (k=ii[ee];k<ii[ee+1];k++) if (!PetscBTLookup(btv,jj[k])) extrow[cum++] = jj[k];
      }
      PetscCall(ISRestoreIndices(eedges[i],&idxs));
      PetscCall(PetscSortRemoveDupsInt(&cum,extrow));
      PetscCall(ISLocalToGlobalMappingApply(vl2g,cum,extrow,gidxs));
      PetscCall(PetscSortIntWithArray(cum,gidxs,extrow));
      PetscCall(ISCreateGeneral(PETSC_COMM_SELF,cum,extrow,PETSC_COPY_VALUES,&extcols[i]));
      if (cum != size -1) {
        if (print) {
          PetscCall(PetscObjectSetName((PetscObject)eedges[i],"error_edge_secondpass"));
          PetscCall(ISView(eedges[i],NULL));
          PetscCall(PetscObjectSetName((PetscObject)extcols[i],"error_extcol_secondpass"));
          PetscCall(ISView(extcols[i],NULL));
        }
        eerr = PETSC_TRUE;
      }
    }
  }
  PetscCall(MatRestoreRowIJ(lG,0,PETSC_FALSE,PETSC_FALSE,&i,&ii,&jj,&done));
  PetscCall(PetscFree2(extrow,gidxs));
  PetscCall(PetscBTDestroy(&bter));
  if (print) PetscCall(PCBDDCGraphASCIIView(pcbddc->mat_graph,5,PETSC_VIEWER_STDOUT_SELF));
  /* an error should not occur at this point */
  PetscCheck(!eerr,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unexpected SIZE OF EDGE > EXTCOL SECOND PASS");

  /* Check the number of endpoints */
  PetscCall(MatGetRowIJ(lG,0,PETSC_FALSE,PETSC_FALSE,&i,&ii,&jj,&done));
  PetscCall(PetscMalloc1(2*nee,&corners));
  PetscCall(PetscMalloc1(nee,&cedges));
  for (i=0;i<nee;i++) {
    PetscInt size, found = 0, gc[2];

    /* init with defaults */
    cedges[i] = corners[i*2] = corners[i*2+1] = -1;
    PetscCall(ISGetLocalSize(eedges[i],&size));
    if (!size && nedfieldlocal) continue;
    PetscCheck(size,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unexpected zero sized edge %" PetscInt_FMT,i);
    PetscCall(ISGetIndices(eedges[i],&idxs));
    PetscCall(PetscBTMemzero(nv,btvc));
    for (j=0;j<size;j++) {
      PetscInt k,ee = idxs[j];
      for (k=ii[ee];k<ii[ee+1];k++) {
        PetscInt vv = jj[k];
        if (PetscBTLookup(btv,vv) && !PetscBTLookupSet(btvc,vv)) {
          PetscCheck(found != 2,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Found more then two corners for edge %" PetscInt_FMT,i);
          corners[i*2+found++] = vv;
        }
      }
    }
    if (found != 2) {
      PetscInt e;
      if (fl2g) {
        PetscCall(ISLocalToGlobalMappingApply(fl2g,1,idxs,&e));
      } else {
        e = idxs[0];
      }
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Found %" PetscInt_FMT " corners for edge %" PetscInt_FMT " (astart %" PetscInt_FMT ", estart %" PetscInt_FMT ")",found,i,e,idxs[0]);
    }

    /* get primal dof index on this coarse edge */
    PetscCall(ISLocalToGlobalMappingApply(vl2g,2,corners+2*i,gc));
    if (gc[0] > gc[1]) {
      PetscInt swap  = corners[2*i];
      corners[2*i]   = corners[2*i+1];
      corners[2*i+1] = swap;
    }
    cedges[i] = idxs[size-1];
    PetscCall(ISRestoreIndices(eedges[i],&idxs));
    if (print) PetscPrintf(PETSC_COMM_SELF,"EDGE %" PetscInt_FMT ": ce %" PetscInt_FMT ", corners (%" PetscInt_FMT ",%" PetscInt_FMT ")\n",i,cedges[i],corners[2*i],corners[2*i+1]);
  }
  PetscCall(MatRestoreRowIJ(lG,0,PETSC_FALSE,PETSC_FALSE,&i,&ii,&jj,&done));
  PetscCall(PetscBTDestroy(&btvc));

  if (PetscDefined(USE_DEBUG)) {
    /* Inspects columns of lG (rows of lGt) and make sure the change of basis will
     not interfere with neighbouring coarse edges */
    PetscCall(PetscMalloc1(nee+1,&emarks));
    PetscCall(MatGetRowIJ(lGt,0,PETSC_FALSE,PETSC_FALSE,&i,&ii,&jj,&done));
    for (i=0;i<nv;i++) {
      PetscInt emax = 0,eemax = 0;

      if (ii[i+1]==ii[i] || PetscBTLookup(btv,i)) continue;
      PetscCall(PetscArrayzero(emarks,nee+1));
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
          SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Found 2 coarse edges (id %" PetscInt_FMT " and %" PetscInt_FMT ") connected through the %" PetscInt_FMT " nodal dof at edge dof %" PetscInt_FMT,marks[jj[j]]-1,eemax,i,jj[j]);
        }
      }
    }
    PetscCall(PetscFree(emarks));
    PetscCall(MatRestoreRowIJ(lGt,0,PETSC_FALSE,PETSC_FALSE,&i,&ii,&jj,&done));
  }

  /* Compute extended rows indices for edge blocks of the change of basis */
  PetscCall(MatGetRowIJ(lGt,0,PETSC_FALSE,PETSC_FALSE,&i,&ii,&jj,&done));
  PetscCall(MatSeqAIJGetMaxRowNonzeros(lGt,&extmem));
  extmem *= maxsize;
  PetscCall(PetscMalloc1(extmem*nee,&extrow));
  PetscCall(PetscMalloc1(nee,&extrows));
  PetscCall(PetscCalloc1(nee,&extrowcum));
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
    PetscCheck(extrowcum[mark] + size <= extmem,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Not enough memory allocated %" PetscInt_FMT " > %" PetscInt_FMT,extrowcum[mark] + size,extmem);
    PetscCall(PetscArraycpy(extrow+start,jj+ii[i],size));
    extrowcum[mark] += size;
  }
  PetscCall(MatRestoreRowIJ(lGt,0,PETSC_FALSE,PETSC_FALSE,&i,&ii,&jj,&done));
  PetscCall(MatDestroy(&lGt));
  PetscCall(PetscFree(marks));

  /* Compress extrows */
  cum  = 0;
  for (i=0;i<nee;i++) {
    PetscInt size = extrowcum[i],*start = extrow + i*extmem;
    PetscCall(PetscSortRemoveDupsInt(&size,start));
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF,size,start,PETSC_USE_POINTER,&extrows[i]));
    cum  = PetscMax(cum,size);
  }
  PetscCall(PetscFree(extrowcum));
  PetscCall(PetscBTDestroy(&btv));
  PetscCall(PetscBTDestroy(&btvcand));

  /* Workspace for lapack inner calls and VecSetValues */
  PetscCall(PetscMalloc2((5+cum+maxsize)*maxsize,&work,maxsize,&rwork));

  /* Create change of basis matrix (preallocation can be improved) */
  PetscCall(MatCreate(comm,&T));
  PetscCall(MatSetSizes(T,pc->pmat->rmap->n,pc->pmat->rmap->n,pc->pmat->rmap->N,pc->pmat->rmap->N));
  PetscCall(MatSetType(T,MATAIJ));
  PetscCall(MatSeqAIJSetPreallocation(T,10,NULL));
  PetscCall(MatMPIAIJSetPreallocation(T,10,NULL,10,NULL));
  PetscCall(MatSetLocalToGlobalMapping(T,al2g,al2g));
  PetscCall(MatSetOption(T,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE));
  PetscCall(MatSetOption(T,MAT_ROW_ORIENTED,PETSC_FALSE));
  PetscCall(ISLocalToGlobalMappingDestroy(&al2g));

  /* Defaults to identity */
  PetscCall(MatCreateVecs(pc->pmat,&tvec,NULL));
  PetscCall(VecSet(tvec,1.0));
  PetscCall(MatDiagonalSet(T,tvec,INSERT_VALUES));
  PetscCall(VecDestroy(&tvec));

  /* Create discrete gradient for the coarser level if needed */
  PetscCall(MatDestroy(&pcbddc->nedcG));
  PetscCall(ISDestroy(&pcbddc->nedclocal));
  if (pcbddc->current_level < pcbddc->max_levels) {
    ISLocalToGlobalMapping cel2g,cvl2g;
    IS                     wis,gwis;
    PetscInt               cnv,cne;

    PetscCall(ISCreateGeneral(comm,nee,cedges,PETSC_COPY_VALUES,&wis));
    if (fl2g) {
      PetscCall(ISLocalToGlobalMappingApplyIS(fl2g,wis,&pcbddc->nedclocal));
    } else {
      PetscCall(PetscObjectReference((PetscObject)wis));
      pcbddc->nedclocal = wis;
    }
    PetscCall(ISLocalToGlobalMappingApplyIS(el2g,wis,&gwis));
    PetscCall(ISDestroy(&wis));
    PetscCall(ISRenumber(gwis,NULL,&cne,&wis));
    PetscCall(ISLocalToGlobalMappingCreateIS(wis,&cel2g));
    PetscCall(ISDestroy(&wis));
    PetscCall(ISDestroy(&gwis));

    PetscCall(ISCreateGeneral(comm,2*nee,corners,PETSC_USE_POINTER,&wis));
    PetscCall(ISLocalToGlobalMappingApplyIS(vl2g,wis,&gwis));
    PetscCall(ISDestroy(&wis));
    PetscCall(ISRenumber(gwis,NULL,&cnv,&wis));
    PetscCall(ISLocalToGlobalMappingCreateIS(wis,&cvl2g));
    PetscCall(ISDestroy(&wis));
    PetscCall(ISDestroy(&gwis));

    PetscCall(MatCreate(comm,&pcbddc->nedcG));
    PetscCall(MatSetSizes(pcbddc->nedcG,PETSC_DECIDE,PETSC_DECIDE,cne,cnv));
    PetscCall(MatSetType(pcbddc->nedcG,MATAIJ));
    PetscCall(MatSeqAIJSetPreallocation(pcbddc->nedcG,2,NULL));
    PetscCall(MatMPIAIJSetPreallocation(pcbddc->nedcG,2,NULL,2,NULL));
    PetscCall(MatSetLocalToGlobalMapping(pcbddc->nedcG,cel2g,cvl2g));
    PetscCall(ISLocalToGlobalMappingDestroy(&cel2g));
    PetscCall(ISLocalToGlobalMappingDestroy(&cvl2g));
  }
  PetscCall(ISLocalToGlobalMappingDestroy(&vl2g));

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
      PetscCall(ISCreateGeneral(PETSC_COMM_SELF,2,corners+2*i,PETSC_USE_POINTER,&cornersis));
    }
    PetscCall(PCBDDCComputeNedelecChangeEdge(lG,eedges[i],extrows[i],extcols[i],cornersis,&Gins,&GKins,cvals,work,rwork));
    if (Gins && GKins) {
      const PetscScalar *data;
      const PetscInt    *rows,*cols;
      PetscInt          nrh,nch,nrc,ncc;

      PetscCall(ISGetIndices(eedges[i],&cols));
      /* H1 */
      PetscCall(ISGetIndices(extrows[i],&rows));
      PetscCall(MatGetSize(Gins,&nrh,&nch));
      PetscCall(MatDenseGetArrayRead(Gins,&data));
      PetscCall(MatSetValuesLocal(T,nrh,rows,nch,cols,data,INSERT_VALUES));
      PetscCall(MatDenseRestoreArrayRead(Gins,&data));
      PetscCall(ISRestoreIndices(extrows[i],&rows));
      /* complement */
      PetscCall(MatGetSize(GKins,&nrc,&ncc));
      PetscCheck(ncc,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Constant function has not been generated for coarse edge %" PetscInt_FMT,i);
      PetscCheck(ncc + nch == nrc,PETSC_COMM_SELF,PETSC_ERR_PLIB,"The sum of the number of columns of GKins %" PetscInt_FMT " and Gins %" PetscInt_FMT " does not match %" PetscInt_FMT " for coarse edge %" PetscInt_FMT,ncc,nch,nrc,i);
      PetscCheck(ncc == 1 || !pcbddc->nedcG,PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot generate the coarse discrete gradient for coarse edge %" PetscInt_FMT " with ncc %" PetscInt_FMT,i,ncc);
      PetscCall(MatDenseGetArrayRead(GKins,&data));
      PetscCall(MatSetValuesLocal(T,nrc,cols,ncc,cols+nch,data,INSERT_VALUES));
      PetscCall(MatDenseRestoreArrayRead(GKins,&data));

      /* coarse discrete gradient */
      if (pcbddc->nedcG) {
        PetscInt cols[2];

        cols[0] = 2*i;
        cols[1] = 2*i+1;
        PetscCall(MatSetValuesLocal(pcbddc->nedcG,1,&i,2,cols,cvals,INSERT_VALUES));
      }
      PetscCall(ISRestoreIndices(eedges[i],&cols));
    }
    PetscCall(ISDestroy(&extrows[i]));
    PetscCall(ISDestroy(&extcols[i]));
    PetscCall(ISDestroy(&cornersis));
    PetscCall(MatDestroy(&Gins));
    PetscCall(MatDestroy(&GKins));
  }
  PetscCall(ISLocalToGlobalMappingDestroy(&el2g));

  /* Start assembling */
  PetscCall(MatAssemblyBegin(T,MAT_FINAL_ASSEMBLY));
  if (pcbddc->nedcG) {
    PetscCall(MatAssemblyBegin(pcbddc->nedcG,MAT_FINAL_ASSEMBLY));
  }

  /* Free */
  if (fl2g) {
    PetscCall(ISDestroy(&primals));
    for (i=0;i<nee;i++) {
      PetscCall(ISDestroy(&eedges[i]));
    }
    PetscCall(PetscFree(eedges));
  }

  /* hack mat_graph with primal dofs on the coarse edges */
  {
    PCBDDCGraph graph   = pcbddc->mat_graph;
    PetscInt    *oqueue = graph->queue;
    PetscInt    *ocptr  = graph->cptr;
    PetscInt    ncc,*idxs;

    /* find first primal edge */
    if (pcbddc->nedclocal) {
      PetscCall(ISGetIndices(pcbddc->nedclocal,(const PetscInt**)&idxs));
    } else {
      if (fl2g) {
        PetscCall(ISLocalToGlobalMappingApply(fl2g,nee,cedges,cedges));
      }
      idxs = cedges;
    }
    cum = 0;
    while (cum < nee && cedges[cum] < 0) cum++;

    /* adapt connected components */
    PetscCall(PetscMalloc2(graph->nvtxs+1,&graph->cptr,ocptr[graph->ncc],&graph->queue));
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
      PetscCall(ISRestoreIndices(pcbddc->nedclocal,(const PetscInt**)&idxs));
    }
    PetscCall(PetscFree2(ocptr,oqueue));
  }
  PetscCall(ISLocalToGlobalMappingDestroy(&fl2g));
  PetscCall(PCBDDCGraphRestoreCandidatesIS(pcbddc->mat_graph,NULL,NULL,&nee,&alleedges,&allprimals));
  PetscCall(PCBDDCGraphResetCSR(pcbddc->mat_graph));
  PetscCall(MatDestroy(&conn));

  PetscCall(ISDestroy(&nedfieldlocal));
  PetscCall(PetscFree(extrow));
  PetscCall(PetscFree2(work,rwork));
  PetscCall(PetscFree(corners));
  PetscCall(PetscFree(cedges));
  PetscCall(PetscFree(extrows));
  PetscCall(PetscFree(extcols));
  PetscCall(MatDestroy(&lG));

  /* Complete assembling */
  PetscCall(MatAssemblyEnd(T,MAT_FINAL_ASSEMBLY));
  if (pcbddc->nedcG) {
    PetscCall(MatAssemblyEnd(pcbddc->nedcG,MAT_FINAL_ASSEMBLY));
#if 0
    PetscCall(PetscObjectSetName((PetscObject)pcbddc->nedcG,"coarse_G"));
    PetscCall(MatView(pcbddc->nedcG,NULL));
#endif
  }

  /* set change of basis */
  PetscCall(PCBDDCSetChangeOfBasisMat(pc,T,singular));
  PetscCall(MatDestroy(&T));

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

    PetscCall(VecGetOwnershipRange(quad_vecs[i],&first,&last));
    PetscCheck(last-first >= 2*nvecs || !has_const,PETSC_COMM_SELF,PETSC_ERR_SUP,"Not implemented");
    if (i>=first && i < last) {
      PetscScalar *data;
      PetscCall(VecGetArray(quad_vecs[i],&data));
      if (!has_const) {
        data[i-first] = 1.;
      } else {
        data[2*i-first] = 1./PetscSqrtReal(2.);
        data[2*i-first+1] = -1./PetscSqrtReal(2.);
      }
      PetscCall(VecRestoreArray(quad_vecs[i],&data));
    }
    PetscCall(PetscObjectStateIncrease((PetscObject)quad_vecs[i]));
  }
  PetscCall(MatNullSpaceCreate(comm,has_const,nvecs,quad_vecs,nnsp));
  for (i=0;i<nvecs;i++) { /* reset vectors */
    PetscInt first,last;
    PetscCall(VecLockReadPop(quad_vecs[i]));
    PetscCall(VecGetOwnershipRange(quad_vecs[i],&first,&last));
    if (i>=first && i < last) {
      PetscScalar *data;
      PetscCall(VecGetArray(quad_vecs[i],&data));
      if (!has_const) {
        data[i-first] = 0.;
      } else {
        data[2*i-first] = 0.;
        data[2*i-first+1] = 0.;
      }
      PetscCall(VecRestoreArray(quad_vecs[i],&data));
    }
    PetscCall(PetscObjectStateIncrease((PetscObject)quad_vecs[i]));
    PetscCall(VecLockReadPush(quad_vecs[i]));
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
  PetscCall(ISLocalToGlobalMappingGetInfo(graph->l2gmap,&n_neigh,&neigh,&n_shared,&shared));
  for (i=0;i<n_neigh;i++) maxneighs = PetscMax(graph->count[shared[i][0]]+1,maxneighs);
  PetscCall(MPIU_Allreduce(MPI_IN_PLACE,&maxneighs,1,MPIU_INT,MPI_MAX,PetscObjectComm((PetscObject)A)));
  if (!maxneighs) {
    PetscCall(ISLocalToGlobalMappingRestoreInfo(graph->l2gmap,&n_neigh,&neigh,&n_shared,&shared));
    *nnsp = NULL;
    PetscFunctionReturn(0);
  }
  maxsize = 0;
  for (i=0;i<n_neigh;i++) maxsize = PetscMax(n_shared[i],maxsize);
  PetscCall(PetscMalloc2(maxsize,&gidxs,maxsize,&vals));
  /* create vectors to hold quadrature weights */
  PetscCall(MatCreateVecs(A,&quad_vec,NULL));
  if (!transpose) {
    PetscCall(MatISGetLocalToGlobalMapping(A,&map,NULL));
  } else {
    PetscCall(MatISGetLocalToGlobalMapping(A,NULL,&map));
  }
  PetscCall(VecDuplicateVecs(quad_vec,maxneighs,&quad_vecs));
  PetscCall(VecDestroy(&quad_vec));
  PetscCall(PCBDDCNullSpaceCreate(PetscObjectComm((PetscObject)A),PETSC_FALSE,maxneighs,quad_vecs,nnsp));
  for (i=0;i<maxneighs;i++) {
    PetscCall(VecLockReadPop(quad_vecs[i]));
  }

  /* compute local quad vec */
  PetscCall(MatISGetLocalMat(divudotp,&loc_divudotp));
  if (!transpose) {
    PetscCall(MatCreateVecs(loc_divudotp,&v,&p));
  } else {
    PetscCall(MatCreateVecs(loc_divudotp,&p,&v));
  }
  PetscCall(VecSet(p,1.));
  if (!transpose) {
    PetscCall(MatMultTranspose(loc_divudotp,p,v));
  } else {
    PetscCall(MatMult(loc_divudotp,p,v));
  }
  if (vl2l) {
    Mat        lA;
    VecScatter sc;

    PetscCall(MatISGetLocalMat(A,&lA));
    PetscCall(MatCreateVecs(lA,&vins,NULL));
    PetscCall(VecScatterCreate(v,NULL,vins,vl2l,&sc));
    PetscCall(VecScatterBegin(sc,v,vins,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(sc,v,vins,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterDestroy(&sc));
  } else {
    vins = v;
  }
  PetscCall(VecGetArrayRead(vins,&array));
  PetscCall(VecDestroy(&p));

  /* insert in global quadrature vecs */
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)A),&rank));
  for (i=1;i<n_neigh;i++) {
    const PetscInt    *idxs;
    PetscInt          idx,nn,j;

    idxs = shared[i];
    nn   = n_shared[i];
    for (j=0;j<nn;j++) vals[j] = array[idxs[j]];
    PetscCall(PetscFindInt(rank,graph->count[idxs[0]],graph->neighbours_set[idxs[0]],&idx));
    idx  = -(idx+1);
    PetscCheck(idx >= 0 && idx < maxneighs,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Invalid index %" PetscInt_FMT " not in [0,%" PetscInt_FMT ")",idx,maxneighs);
    PetscCall(ISLocalToGlobalMappingApply(map,nn,idxs,gidxs));
    PetscCall(VecSetValues(quad_vecs[idx],nn,gidxs,vals,INSERT_VALUES));
  }
  PetscCall(ISLocalToGlobalMappingRestoreInfo(graph->l2gmap,&n_neigh,&neigh,&n_shared,&shared));
  PetscCall(VecRestoreArrayRead(vins,&array));
  if (vl2l) {
    PetscCall(VecDestroy(&vins));
  }
  PetscCall(VecDestroy(&v));
  PetscCall(PetscFree2(gidxs,vals));

  /* assemble near null space */
  for (i=0;i<maxneighs;i++) {
    PetscCall(VecAssemblyBegin(quad_vecs[i]));
  }
  for (i=0;i<maxneighs;i++) {
    PetscCall(VecAssemblyEnd(quad_vecs[i]));
    PetscCall(VecViewFromOptions(quad_vecs[i],NULL,"-pc_bddc_quad_vecs_view"));
    PetscCall(VecLockReadPush(quad_vecs[i]));
  }
  PetscCall(VecDestroyVecs(maxneighs,&quad_vecs));
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
      PetscCall(ISConcatenate(PetscObjectComm((PetscObject)pc),2,list,&newp));
      PetscCall(ISSortRemoveDups(newp));
      PetscCall(ISDestroy(&list[1]));
      pcbddc->user_primal_vertices_local = newp;
    } else {
      PetscCall(PCBDDCSetPrimalVerticesLocalIS(pc,primalv));
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
  Vec            local,global;
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;
  Mat_IS         *matis = (Mat_IS*)pc->pmat->data;
  PetscBool      monolithic = PETSC_FALSE;

  PetscFunctionBegin;
  PetscOptionsBegin(PetscObjectComm((PetscObject)pc),((PetscObject)pc)->prefix,"BDDC topology options","PC");
  PetscCall(PetscOptionsBool("-pc_bddc_monolithic","Discard any information on dofs splitting",NULL,monolithic,&monolithic,NULL));
  PetscOptionsEnd();
  /* need to convert from global to local topology information and remove references to information in global ordering */
  PetscCall(MatCreateVecs(pc->pmat,&global,NULL));
  PetscCall(MatCreateVecs(matis->A,&local,NULL));
  PetscCall(VecBindToCPU(global,PETSC_TRUE));
  PetscCall(VecBindToCPU(local,PETSC_TRUE));
  if (monolithic) { /* just get block size to properly compute vertices */
    if (pcbddc->vertex_size == 1) {
      PetscCall(MatGetBlockSize(pc->pmat,&pcbddc->vertex_size));
    }
    goto boundary;
  }

  if (pcbddc->user_provided_isfordofs) {
    if (pcbddc->n_ISForDofs) {
      PetscInt i;

      PetscCall(PetscMalloc1(pcbddc->n_ISForDofs,&pcbddc->ISForDofsLocal));
      for (i=0;i<pcbddc->n_ISForDofs;i++) {
        PetscInt bs;

        PetscCall(PCBDDCGlobalToLocal(matis->rctx,global,local,pcbddc->ISForDofs[i],&pcbddc->ISForDofsLocal[i]));
        PetscCall(ISGetBlockSize(pcbddc->ISForDofs[i],&bs));
        PetscCall(ISSetBlockSize(pcbddc->ISForDofsLocal[i],bs));
        PetscCall(ISDestroy(&pcbddc->ISForDofs[i]));
      }
      pcbddc->n_ISForDofsLocal = pcbddc->n_ISForDofs;
      pcbddc->n_ISForDofs = 0;
      PetscCall(PetscFree(pcbddc->ISForDofs));
    }
  } else {
    if (!pcbddc->n_ISForDofsLocal) { /* field split not present */
      DM dm;

      PetscCall(MatGetDM(pc->pmat, &dm));
      if (!dm) {
        PetscCall(PCGetDM(pc, &dm));
      }
      if (dm) {
        IS      *fields;
        PetscInt nf,i;

        PetscCall(DMCreateFieldDecomposition(dm,&nf,NULL,&fields,NULL));
        PetscCall(PetscMalloc1(nf,&pcbddc->ISForDofsLocal));
        for (i=0;i<nf;i++) {
          PetscInt bs;

          PetscCall(PCBDDCGlobalToLocal(matis->rctx,global,local,fields[i],&pcbddc->ISForDofsLocal[i]));
          PetscCall(ISGetBlockSize(fields[i],&bs));
          PetscCall(ISSetBlockSize(pcbddc->ISForDofsLocal[i],bs));
          PetscCall(ISDestroy(&fields[i]));
        }
        PetscCall(PetscFree(fields));
        pcbddc->n_ISForDofsLocal = nf;
      } else { /* See if MATIS has fields attached by the conversion from MatNest */
        PetscContainer   c;

        PetscCall(PetscObjectQuery((PetscObject)pc->pmat,"_convert_nest_lfields",(PetscObject*)&c));
        if (c) {
          MatISLocalFields lf;
          PetscCall(PetscContainerGetPointer(c,(void**)&lf));
          PetscCall(PCBDDCSetDofsSplittingLocal(pc,lf->nr,lf->rf));
        } else { /* fallback, create the default fields if bs > 1 */
          PetscInt i, n = matis->A->rmap->n;
          PetscCall(MatGetBlockSize(pc->pmat,&i));
          if (i > 1) {
            pcbddc->n_ISForDofsLocal = i;
            PetscCall(PetscMalloc1(pcbddc->n_ISForDofsLocal,&pcbddc->ISForDofsLocal));
            for (i=0;i<pcbddc->n_ISForDofsLocal;i++) {
              PetscCall(ISCreateStride(PetscObjectComm((PetscObject)pc),n/pcbddc->n_ISForDofsLocal,i,pcbddc->n_ISForDofsLocal,&pcbddc->ISForDofsLocal[i]));
            }
          }
        }
      }
    } else {
      PetscInt i;
      for (i=0;i<pcbddc->n_ISForDofsLocal;i++) {
        PetscCall(PCBDDCConsistencyCheckIS(pc,MPI_LAND,&pcbddc->ISForDofsLocal[i]));
      }
    }
  }

boundary:
  if (!pcbddc->DirichletBoundariesLocal && pcbddc->DirichletBoundaries) {
    PetscCall(PCBDDCGlobalToLocal(matis->rctx,global,local,pcbddc->DirichletBoundaries,&pcbddc->DirichletBoundariesLocal));
  } else if (pcbddc->DirichletBoundariesLocal) {
    PetscCall(PCBDDCConsistencyCheckIS(pc,MPI_LAND,&pcbddc->DirichletBoundariesLocal));
  }
  if (!pcbddc->NeumannBoundariesLocal && pcbddc->NeumannBoundaries) {
    PetscCall(PCBDDCGlobalToLocal(matis->rctx,global,local,pcbddc->NeumannBoundaries,&pcbddc->NeumannBoundariesLocal));
  } else if (pcbddc->NeumannBoundariesLocal) {
    PetscCall(PCBDDCConsistencyCheckIS(pc,MPI_LOR,&pcbddc->NeumannBoundariesLocal));
  }
  if (!pcbddc->user_primal_vertices_local && pcbddc->user_primal_vertices) {
    PetscCall(PCBDDCGlobalToLocal(matis->rctx,global,local,pcbddc->user_primal_vertices,&pcbddc->user_primal_vertices_local));
  }
  PetscCall(VecDestroy(&global));
  PetscCall(VecDestroy(&local));
  /* detect local disconnected subdomains if requested (use matis->A) */
  if (pcbddc->detect_disconnected) {
    IS        primalv = NULL;
    PetscInt  i;
    PetscBool filter = pcbddc->detect_disconnected_filter;

    for (i=0;i<pcbddc->n_local_subs;i++) {
      PetscCall(ISDestroy(&pcbddc->local_subs[i]));
    }
    PetscCall(PetscFree(pcbddc->local_subs));
    PetscCall(PCBDDCDetectDisconnectedComponents(pc,filter,&pcbddc->n_local_subs,&pcbddc->local_subs,&primalv));
    PetscCall(PCBDDCAddPrimalVerticesLocalIS(pc,primalv));
    PetscCall(ISDestroy(&primalv));
  }
  /* early stage corner detection */
  {
    DM dm;

    PetscCall(MatGetDM(pc->pmat,&dm));
    if (!dm) {
      PetscCall(PCGetDM(pc,&dm));
    }
    if (dm) {
      PetscBool isda;

      PetscCall(PetscObjectTypeCompare((PetscObject)dm,DMDA,&isda));
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

          PetscCall(DMDAGetInfo(dm,NULL,NULL,NULL,NULL,NULL,NULL,NULL,&dof,NULL,NULL,NULL,NULL,NULL));
          PetscCall(DMGetCoordinates(dm,&cvec));
          PetscCall(VecGetLocalSize(cvec,&n));
          PetscCall(VecGetBlockSize(cvec,&cdim));
          n   /= cdim;
          PetscCall(PetscFree(pcbddc->mat_graph->coords));
          PetscCall(PetscMalloc1(dof*n*cdim,&pcbddc->mat_graph->coords));
          PetscCall(VecGetArrayRead(cvec,&coords));
#if defined(PETSC_USE_COMPLEX)
          memc = PETSC_FALSE;
#endif
          if (dof != 1) memc = PETSC_FALSE;
          if (memc) {
            PetscCall(PetscArraycpy(pcbddc->mat_graph->coords,coords,cdim*n*dof));
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
          PetscCall(VecRestoreArrayRead(cvec,&coords));
          pcbddc->mat_graph->cdim  = cdim;
          pcbddc->mat_graph->cnloc = dof*n;
          pcbddc->mat_graph->cloc  = PETSC_FALSE;
        }
        PetscCall(DMDAGetSubdomainCornersIS(dm,&corners));
        PetscCall(MatISGetLocalMat(pc->pmat,&lA));
        PetscCall(MatGetLocalToGlobalMapping(lA,&l2l,NULL));
        PetscCall(MatISRestoreLocalMat(pc->pmat,&lA));
        lo   = (PetscBool)(l2l && corners);
        PetscCall(MPIU_Allreduce(&lo,&gl,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)pc)));
        if (gl) { /* From PETSc's DMDA */
          const PetscInt    *idx;
          PetscInt          dof,bs,*idxout,n;

          PetscCall(DMDAGetInfo(dm,NULL,NULL,NULL,NULL,NULL,NULL,NULL,&dof,NULL,NULL,NULL,NULL,NULL));
          PetscCall(ISLocalToGlobalMappingGetBlockSize(l2l,&bs));
          PetscCall(ISGetLocalSize(corners,&n));
          PetscCall(ISGetIndices(corners,&idx));
          if (bs == dof) {
            PetscCall(PetscMalloc1(n,&idxout));
            PetscCall(ISLocalToGlobalMappingApplyBlock(l2l,n,idx,idxout));
          } else { /* the original DMDA local-to-local map have been modified */
            PetscInt i,d;

            PetscCall(PetscMalloc1(dof*n,&idxout));
            for (i=0;i<n;i++) for (d=0;d<dof;d++) idxout[dof*i+d] = dof*idx[i]+d;
            PetscCall(ISLocalToGlobalMappingApply(l2l,dof*n,idxout,idxout));

            bs = 1;
            n *= dof;
          }
          PetscCall(ISRestoreIndices(corners,&idx));
          PetscCall(DMDARestoreSubdomainCornersIS(dm,&corners));
          PetscCall(ISCreateBlock(PetscObjectComm((PetscObject)pc),bs,n,idxout,PETSC_OWN_POINTER,&corners));
          PetscCall(PCBDDCAddPrimalVerticesLocalIS(pc,corners));
          PetscCall(ISDestroy(&corners));
          pcbddc->corner_selected  = PETSC_TRUE;
          pcbddc->corner_selection = PETSC_TRUE;
        }
        if (corners) {
          PetscCall(DMDARestoreSubdomainCornersIS(dm,&corners));
        }
      }
    }
  }
  if (pcbddc->corner_selection && !pcbddc->mat_graph->cdim) {
    DM dm;

    PetscCall(MatGetDM(pc->pmat,&dm));
    if (!dm) {
      PetscCall(PCGetDM(pc,&dm));
    }
    if (dm) { /* this can get very expensive, I need to find a faster alternative */
      Vec            vcoords;
      PetscSection   section;
      PetscReal      *coords;
      PetscInt       d,cdim,nl,nf,**ctxs;
      PetscErrorCode (**funcs)(PetscInt, PetscReal, const PetscReal *, PetscInt, PetscScalar *, void *);
      /* debug coordinates */
      PetscViewer       viewer;
      PetscBool         flg;
      PetscViewerFormat format;
      const char        *prefix;

      PetscCall(DMGetCoordinateDim(dm,&cdim));
      PetscCall(DMGetLocalSection(dm,&section));
      PetscCall(PetscSectionGetNumFields(section,&nf));
      PetscCall(DMCreateGlobalVector(dm,&vcoords));
      PetscCall(VecGetLocalSize(vcoords,&nl));
      PetscCall(PetscMalloc1(nl*cdim,&coords));
      PetscCall(PetscMalloc2(nf,&funcs,nf,&ctxs));
      PetscCall(PetscMalloc1(nf,&ctxs[0]));
      for (d=0;d<nf;d++) funcs[d] = func_coords_private;
      for (d=1;d<nf;d++) ctxs[d] = ctxs[d-1] + 1;

      /* debug coordinates */
      PetscCall(PCGetOptionsPrefix(pc,&prefix));
      PetscCall(PetscOptionsGetViewer(PetscObjectComm((PetscObject)vcoords),((PetscObject)vcoords)->options,prefix,"-pc_bddc_coords_vec_view",&viewer,&format,&flg));
      if (flg) PetscCall(PetscViewerPushFormat(viewer,format));
      for (d=0;d<cdim;d++) {
        PetscInt          i;
        const PetscScalar *v;
        char              name[16];

        for (i=0;i<nf;i++) ctxs[i][0] = d;
        PetscCall(PetscSNPrintf(name,sizeof(name),"bddc_coords_%d",(int)d));
        PetscCall(PetscObjectSetName((PetscObject)vcoords,name));
        PetscCall(DMProjectFunction(dm,0.0,funcs,(void**)ctxs,INSERT_VALUES,vcoords));
        if (flg) PetscCall(VecView(vcoords,viewer));
        PetscCall(VecGetArrayRead(vcoords,&v));
        for (i=0;i<nl;i++) coords[i*cdim+d] = PetscRealPart(v[i]);
        PetscCall(VecRestoreArrayRead(vcoords,&v));
      }
      PetscCall(VecDestroy(&vcoords));
      PetscCall(PCSetCoordinates(pc,cdim,nl,coords));
      PetscCall(PetscFree(coords));
      PetscCall(PetscFree(ctxs[0]));
      PetscCall(PetscFree2(funcs,ctxs));
      if (flg) {
        PetscCall(PetscViewerPopFormat(viewer));
        PetscCall(PetscViewerDestroy(&viewer));
      }
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
  PetscCheck(mop == MPI_LAND || mop == MPI_LOR,PetscObjectComm((PetscObject)(pc)),PETSC_ERR_SUP,"Supported are MPI_LAND and MPI_LOR");
  if (mop == MPI_LAND) {
    /* init rootdata with true */
    for (i=0;i<pc->pmat->rmap->n;i++) matis->sf_rootdata[i] = 1;
  } else {
    PetscCall(PetscArrayzero(matis->sf_rootdata,pc->pmat->rmap->n));
  }
  PetscCall(PetscArrayzero(matis->sf_leafdata,n));
  PetscCall(ISGetLocalSize(*is,&nd));
  PetscCall(ISGetIndices(*is,&idxs));
  for (i=0;i<nd;i++)
    if (-1 < idxs[i] && idxs[i] < n)
      matis->sf_leafdata[idxs[i]] = 1;
  PetscCall(ISRestoreIndices(*is,&idxs));
  PetscCall(PetscSFReduceBegin(matis->sf,MPIU_INT,matis->sf_leafdata,matis->sf_rootdata,mop));
  PetscCall(PetscSFReduceEnd(matis->sf,MPIU_INT,matis->sf_leafdata,matis->sf_rootdata,mop));
  PetscCall(PetscSFBcastBegin(matis->sf,MPIU_INT,matis->sf_rootdata,matis->sf_leafdata,MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(matis->sf,MPIU_INT,matis->sf_rootdata,matis->sf_leafdata,MPI_REPLACE));
  if (mop == MPI_LAND) {
    PetscCall(PetscMalloc1(nd,&nidxs));
  } else {
    PetscCall(PetscMalloc1(n,&nidxs));
  }
  for (i=0,nnd=0;i<n;i++)
    if (matis->sf_leafdata[i])
      nidxs[nnd++] = i;
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)(*is)),nnd,nidxs,PETSC_OWN_POINTER,&nis));
  PetscCall(ISDestroy(is));
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

    PetscCall(MatMultTranspose(pcbddc->ChangeOfBasisMatrix,r,pcbddc->work_change));
    swap = pcbddc->work_change;
    pcbddc->work_change = r;
    r = swap;
  }
  PetscCall(VecScatterBegin(pcis->global_to_D,r,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(pcis->global_to_D,r,pcis->vec1_D,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(PetscLogEventBegin(PC_BDDC_Solves[pcbddc->current_level][0],pc,0,0,0));
  PetscCall(KSPSolve(pcbddc->ksp_D,pcis->vec1_D,pcis->vec2_D));
  PetscCall(PetscLogEventEnd(PC_BDDC_Solves[pcbddc->current_level][0],pc,0,0,0));
  PetscCall(KSPCheckSolve(pcbddc->ksp_D,pc,pcis->vec2_D));
  PetscCall(VecSet(z,0.));
  PetscCall(VecScatterBegin(pcis->global_to_D,pcis->vec2_D,z,INSERT_VALUES,SCATTER_REVERSE));
  PetscCall(VecScatterEnd(pcis->global_to_D,pcis->vec2_D,z,INSERT_VALUES,SCATTER_REVERSE));
  if (pcbddc->ChangeOfBasisMatrix) {
    pcbddc->work_change = r;
    PetscCall(VecCopy(z,pcbddc->work_change));
    PetscCall(MatMult(pcbddc->ChangeOfBasisMatrix,pcbddc->work_change,z));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCBenignMatMult_Private_Private(Mat A, Vec x, Vec y, PetscBool transpose)
{
  PCBDDCBenignMatMult_ctx ctx;
  PetscBool               apply_right,apply_left,reset_x;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A,&ctx));
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

    PetscCall(VecGetLocalSize(x,&nl));
    PetscCall(VecGetArrayRead(x,&ax));
    PetscCall(PetscArraycpy(ctx->work,ax,nl));
    PetscCall(VecRestoreArrayRead(x,&ax));
    for (i=0;i<ctx->benign_n;i++) {
      PetscScalar    sum,val;
      const PetscInt *idxs;
      PetscInt       nz,j;
      PetscCall(ISGetLocalSize(ctx->benign_zerodiag_subs[i],&nz));
      PetscCall(ISGetIndices(ctx->benign_zerodiag_subs[i],&idxs));
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
      PetscCall(ISRestoreIndices(ctx->benign_zerodiag_subs[i],&idxs));
    }
    PetscCall(VecPlaceArray(x,ctx->work));
    reset_x = PETSC_TRUE;
  }
  if (transpose) {
    PetscCall(MatMultTranspose(ctx->A,x,y));
  } else {
    PetscCall(MatMult(ctx->A,x,y));
  }
  if (reset_x) {
    PetscCall(VecResetArray(x));
  }
  if (apply_left) {
    PetscScalar *ay;
    PetscInt    i;

    PetscCall(VecGetArray(y,&ay));
    for (i=0;i<ctx->benign_n;i++) {
      PetscScalar    sum,val;
      const PetscInt *idxs;
      PetscInt       nz,j;
      PetscCall(ISGetLocalSize(ctx->benign_zerodiag_subs[i],&nz));
      PetscCall(ISGetIndices(ctx->benign_zerodiag_subs[i],&idxs));
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
      PetscCall(ISRestoreIndices(ctx->benign_zerodiag_subs[i],&idxs));
    }
    PetscCall(VecRestoreArray(y,&ay));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCBenignMatMultTranspose_Private(Mat A, Vec x, Vec y)
{
  PetscFunctionBegin;
  PetscCall(PCBDDCBenignMatMult_Private_Private(A,x,y,PETSC_TRUE));
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCBenignMatMult_Private(Mat A, Vec x, Vec y)
{
  PetscFunctionBegin;
  PetscCall(PCBDDCBenignMatMult_Private_Private(A,x,y,PETSC_FALSE));
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

    PetscCheck(!pcbddc->benign_original_mat,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Benign original mat has not been restored");
    if (!pcbddc->benign_change || !pcbddc->benign_n || pcbddc->benign_change_explicit) PetscFunctionReturn(0);
    PetscCall(PetscMalloc1(pcis->n,&work));
    PetscCall(MatCreate(PETSC_COMM_SELF,&A_IB));
    PetscCall(MatSetSizes(A_IB,pcis->n-pcis->n_B,pcis->n_B,PETSC_DECIDE,PETSC_DECIDE));
    PetscCall(MatSetType(A_IB,MATSHELL));
    PetscCall(MatShellSetOperation(A_IB,MATOP_MULT,(void (*)(void))PCBDDCBenignMatMult_Private));
    PetscCall(MatShellSetOperation(A_IB,MATOP_MULT_TRANSPOSE,(void (*)(void))PCBDDCBenignMatMultTranspose_Private));
    PetscCall(PetscNew(&ctx));
    PetscCall(MatShellSetContext(A_IB,ctx));
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

      PetscCall(ISLocalToGlobalMappingCreateIS(pcis->is_I_local,&N_to_D));
      PetscCall(PetscMalloc1(pcbddc->benign_n,&ctx->benign_zerodiag_subs));
      for (i=0;i<pcbddc->benign_n;i++) {
        PetscCall(ISGlobalToLocalMappingApplyIS(N_to_D,IS_GTOLM_DROP,pcbddc->benign_zerodiag_subs[i],&ctx->benign_zerodiag_subs[i]));
      }
      PetscCall(ISLocalToGlobalMappingDestroy(&N_to_D));
      ctx->free = PETSC_TRUE;
    }
    ctx->A = pcis->A_IB;
    ctx->work = work;
    PetscCall(MatSetUp(A_IB));
    PetscCall(MatAssemblyBegin(A_IB,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A_IB,MAT_FINAL_ASSEMBLY));
    pcis->A_IB = A_IB;

    /* A_BI as A_IB^T */
    PetscCall(MatCreateTranspose(A_IB,&A_BI));
    pcbddc->benign_original_mat = pcis->A_BI;
    pcis->A_BI = A_BI;
  } else {
    if (!pcbddc->benign_original_mat) {
      PetscFunctionReturn(0);
    }
    PetscCall(MatShellGetContext(pcis->A_IB,&ctx));
    PetscCall(MatDestroy(&pcis->A_IB));
    pcis->A_IB = ctx->A;
    ctx->A = NULL;
    PetscCall(MatDestroy(&pcis->A_BI));
    pcis->A_BI = pcbddc->benign_original_mat;
    pcbddc->benign_original_mat = NULL;
    if (ctx->free) {
      PetscInt i;
      for (i=0;i<ctx->benign_n;i++) {
        PetscCall(ISDestroy(&ctx->benign_zerodiag_subs[i]));
      }
      PetscCall(PetscFree(ctx->benign_zerodiag_subs));
    }
    PetscCall(PetscFree(ctx->work));
    PetscCall(PetscFree(ctx));
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
  PetscCall(MatPtAP(matis->A,pcbddc->benign_change,MAT_INITIAL_MATRIX,2.0,&An));
  PetscCall(MatZeroRowsColumns(An,pcbddc->benign_n,pcbddc->benign_p0_lidx,1.0,NULL,NULL));
  if (is1) {
    PetscCall(MatCreateSubMatrix(An,is1,is2,MAT_INITIAL_MATRIX,B));
    PetscCall(MatDestroy(&An));
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
  PetscCall(MatGetSize(A,&n,&m));
  PetscCall(MatGetRowIJ(A,0,PETSC_FALSE,PETSC_FALSE,&n,&ii,&ij,&flg_row));
  PetscCall(MatSeqAIJGetArray(A,&a));
  nnz = n;
  for (i=0;i<ii[n];i++) {
    if (PetscLikely(PetscAbsScalar(a[i]) > PETSC_SMALL)) nnz++;
  }
  PetscCall(PetscMalloc1(n+1,&bii));
  PetscCall(PetscMalloc1(nnz,&bij));
  PetscCall(PetscMalloc1(nnz,&bdata));
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
  PetscCall(MatSeqAIJRestoreArray(A,&a));
  PetscCall(MatCreateSeqAIJWithArrays(PetscObjectComm((PetscObject)A),n,m,bii,bij,bdata,&Bt));
  PetscCall(MatRestoreRowIJ(A,0,PETSC_FALSE,PETSC_FALSE,&n,&ii,&ij,&flg_row));
  {
    Mat_SeqAIJ *b = (Mat_SeqAIJ*)(Bt->data);
    b->free_a = PETSC_TRUE;
    b->free_ij = PETSC_TRUE;
  }
  if (*B == A) {
    PetscCall(MatDestroy(&A));
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
  PetscCall(PCBDDCGraphCreate(&graph));
  PetscCall(MatGetDM(pc->pmat,&dm));
  if (!dm) {
    PetscCall(PCGetDM(pc,&dm));
  }
  if (dm) {
    PetscCall(PetscObjectTypeCompare((PetscObject)dm,DMPLEX,&isplex));
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

    PetscCall(DMPlexGetHeightStratum(dm, 0, &pStart, &pEnd));
    PetscCall(DMGetPointSF(dm, &sfPoint));
    PetscCall(PetscSFGetGraph(sfPoint, &nroots, NULL, NULL, NULL));
    /* Build adjacency graph via a section/segbuffer */
    PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject) dm), &section));
    PetscCall(PetscSectionSetChart(section, pStart, pEnd));
    PetscCall(PetscSegBufferCreate(sizeof(PetscInt),1000,&adjBuffer));
    /* Always use FVM adjacency to create partitioner graph */
    PetscCall(DMGetBasicAdjacency(dm, &useCone, &useClosure));
    PetscCall(DMSetBasicAdjacency(dm, PETSC_TRUE, PETSC_FALSE));
    PetscCall(DMPlexGetCellNumbering(dm, &cellNumbering));
    PetscCall(ISGetIndices(cellNumbering, &cellNum));
    for (n = 0, p = pStart; p < pEnd; p++) {
      /* Skip non-owned cells in parallel (ParMetis expects no overlap) */
      if (nroots > 0) {if (cellNum[p] < 0) continue;}
      adjSize = PETSC_DETERMINE;
      PetscCall(DMPlexGetAdjacency(dm, p, &adjSize, &adj));
      for (a = 0; a < adjSize; ++a) {
        const PetscInt point = adj[a];
        if (pStart <= point && point < pEnd) {
          PetscInt *PETSC_RESTRICT pBuf;
          PetscCall(PetscSectionAddDof(section, p, 1));
          PetscCall(PetscSegBufferGetInts(adjBuffer, 1, &pBuf));
          *pBuf = point;
        }
      }
      n++;
    }
    PetscCall(DMSetBasicAdjacency(dm, useCone, useClosure));
    /* Derive CSR graph from section/segbuffer */
    PetscCall(PetscSectionSetUp(section));
    PetscCall(PetscSectionGetStorageSize(section, &size));
    PetscCall(PetscMalloc1(n+1, &xadj));
    for (idx = 0, p = pStart; p < pEnd; p++) {
      if (nroots > 0) {if (cellNum[p] < 0) continue;}
      PetscCall(PetscSectionGetOffset(section, p, &(xadj[idx++])));
    }
    xadj[n] = size;
    PetscCall(PetscSegBufferExtractAlloc(adjBuffer, &adjncy));
    /* Clean up */
    PetscCall(PetscSegBufferDestroy(&adjBuffer));
    PetscCall(PetscSectionDestroy(&section));
    PetscCall(PetscFree(adj));
    graph->xadj = xadj;
    graph->adjncy = adjncy;
  } else {
    Mat       A;
    PetscBool isseqaij, flg_row;

    PetscCall(MatISGetLocalMat(pc->pmat,&A));
    if (!A->rmap->N || !A->cmap->N) {
      PetscCall(PCBDDCGraphDestroy(&graph));
      PetscFunctionReturn(0);
    }
    PetscCall(PetscObjectBaseTypeCompare((PetscObject)A,MATSEQAIJ,&isseqaij));
    if (!isseqaij && filter) {
      PetscBool isseqdense;

      PetscCall(PetscObjectTypeCompare((PetscObject)A,MATSEQDENSE,&isseqdense));
      if (!isseqdense) {
        PetscCall(MatConvert(A,MATSEQAIJ,MAT_INITIAL_MATRIX,&B));
      } else { /* TODO: rectangular case and LDA */
        PetscScalar *array;
        PetscReal   chop=1.e-6;

        PetscCall(MatDuplicate(A,MAT_COPY_VALUES,&B));
        PetscCall(MatDenseGetArray(B,&array));
        PetscCall(MatGetSize(B,&n,NULL));
        for (i=0;i<n;i++) {
          PetscInt j;
          for (j=i+1;j<n;j++) {
            PetscReal thresh = chop*(PetscAbsScalar(array[i*(n+1)])+PetscAbsScalar(array[j*(n+1)]));
            if (PetscAbsScalar(array[i*n+j]) < thresh) array[i*n+j] = 0.;
            if (PetscAbsScalar(array[j*n+i]) < thresh) array[j*n+i] = 0.;
          }
        }
        PetscCall(MatDenseRestoreArray(B,&array));
        PetscCall(MatConvert(B,MATSEQAIJ,MAT_INPLACE_MATRIX,&B));
      }
    } else {
      PetscCall(PetscObjectReference((PetscObject)A));
      B = A;
    }
    PetscCall(MatGetRowIJ(B,0,PETSC_TRUE,PETSC_FALSE,&n,(const PetscInt**)&xadj,(const PetscInt**)&adjncy,&flg_row));

    /* if filter is true, then removes entries lower than PETSC_SMALL in magnitude */
    if (filter) {
      PetscScalar *data;
      PetscInt    j,cum;

      PetscCall(PetscCalloc2(n+1,&xadj_filtered,xadj[n],&adjncy_filtered));
      PetscCall(MatSeqAIJGetArray(B,&data));
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
      PetscCall(MatSeqAIJRestoreArray(B,&data));
      graph->xadj = xadj_filtered;
      graph->adjncy = adjncy_filtered;
    } else {
      graph->xadj = xadj;
      graph->adjncy = adjncy;
    }
  }
  /* compute local connected components using PCBDDCGraph */
  PetscCall(ISCreateStride(PETSC_COMM_SELF,n,0,1,&is_dummy));
  PetscCall(ISLocalToGlobalMappingCreateIS(is_dummy,&l2gmap_dummy));
  PetscCall(ISDestroy(&is_dummy));
  PetscCall(PCBDDCGraphInit(graph,l2gmap_dummy,n,PETSC_MAX_INT));
  PetscCall(ISLocalToGlobalMappingDestroy(&l2gmap_dummy));
  PetscCall(PCBDDCGraphSetUp(graph,1,NULL,NULL,0,NULL,NULL));
  PetscCall(PCBDDCGraphComputeConnectedComponents(graph));

  /* partial clean up */
  PetscCall(PetscFree2(xadj_filtered,adjncy_filtered));
  if (B) {
    PetscBool flg_row;
    PetscCall(MatRestoreRowIJ(B,0,PETSC_TRUE,PETSC_FALSE,&n,(const PetscInt**)&xadj,(const PetscInt**)&adjncy,&flg_row));
    PetscCall(MatDestroy(&B));
  }
  if (isplex) {
    PetscCall(PetscFree(xadj));
    PetscCall(PetscFree(adjncy));
  }

  /* get back data */
  if (isplex) {
    if (ncc) *ncc = graph->ncc;
    if (cc || primalv) {
      Mat          A;
      PetscBT      btv,btvt;
      PetscSection subSection;
      PetscInt     *ids,cum,cump,*cids,*pids;

      PetscCall(DMPlexGetSubdomainSection(dm,&subSection));
      PetscCall(MatISGetLocalMat(pc->pmat,&A));
      PetscCall(PetscMalloc3(A->rmap->n,&ids,graph->ncc+1,&cids,A->rmap->n,&pids));
      PetscCall(PetscBTCreate(A->rmap->n,&btv));
      PetscCall(PetscBTCreate(A->rmap->n,&btvt));

      cids[0] = 0;
      for (i = 0, cump = 0, cum = 0; i < graph->ncc; i++) {
        PetscInt j;

        PetscCall(PetscBTMemzero(A->rmap->n,btvt));
        for (j = graph->cptr[i]; j < graph->cptr[i+1]; j++) {
          PetscInt k, size, *closure = NULL, cell = graph->queue[j];

          PetscCall(DMPlexGetTransitiveClosure(dm,cell,PETSC_TRUE,&size,&closure));
          for (k = 0; k < 2*size; k += 2) {
            PetscInt s, pp, p = closure[k], off, dof, cdof;

            PetscCall(PetscSectionGetConstraintDof(subSection,p,&cdof));
            PetscCall(PetscSectionGetOffset(subSection,p,&off));
            PetscCall(PetscSectionGetDof(subSection,p,&dof));
            for (s = 0; s < dof-cdof; s++) {
              if (PetscBTLookupSet(btvt,off+s)) continue;
              if (!PetscBTLookup(btv,off+s)) ids[cum++] = off+s;
              else pids[cump++] = off+s; /* cross-vertex */
            }
            PetscCall(DMPlexGetTreeParent(dm,p,&pp,NULL));
            if (pp != p) {
              PetscCall(PetscSectionGetConstraintDof(subSection,pp,&cdof));
              PetscCall(PetscSectionGetOffset(subSection,pp,&off));
              PetscCall(PetscSectionGetDof(subSection,pp,&dof));
              for (s = 0; s < dof-cdof; s++) {
                if (PetscBTLookupSet(btvt,off+s)) continue;
                if (!PetscBTLookup(btv,off+s)) ids[cum++] = off+s;
                else pids[cump++] = off+s; /* cross-vertex */
              }
            }
          }
          PetscCall(DMPlexRestoreTransitiveClosure(dm,cell,PETSC_TRUE,&size,&closure));
        }
        cids[i+1] = cum;
        /* mark dofs as already assigned */
        for (j = cids[i]; j < cids[i+1]; j++) {
          PetscCall(PetscBTSet(btv,ids[j]));
        }
      }
      if (cc) {
        PetscCall(PetscMalloc1(graph->ncc,&cc_n));
        for (i = 0; i < graph->ncc; i++) {
          PetscCall(ISCreateGeneral(PETSC_COMM_SELF,cids[i+1]-cids[i],ids+cids[i],PETSC_COPY_VALUES,&cc_n[i]));
        }
        *cc = cc_n;
      }
      if (primalv) {
        PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)pc),cump,pids,PETSC_COPY_VALUES,primalv));
      }
      PetscCall(PetscFree3(ids,cids,pids));
      PetscCall(PetscBTDestroy(&btv));
      PetscCall(PetscBTDestroy(&btvt));
    }
  } else {
    if (ncc) *ncc = graph->ncc;
    if (cc) {
      PetscCall(PetscMalloc1(graph->ncc,&cc_n));
      for (i=0;i<graph->ncc;i++) {
        PetscCall(ISCreateGeneral(PETSC_COMM_SELF,graph->cptr[i+1]-graph->cptr[i],graph->queue+graph->cptr[i],PETSC_COPY_VALUES,&cc_n[i]));
      }
      *cc = cc_n;
    }
  }
  /* clean up graph */
  graph->xadj = NULL;
  graph->adjncy = NULL;
  PetscCall(PCBDDCGraphDestroy(&graph));
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCBenignCheck(PC pc, IS zerodiag)
{
  PC_BDDC*       pcbddc = (PC_BDDC*)pc->data;
  PC_IS*         pcis = (PC_IS*)(pc->data);
  IS             dirIS = NULL;
  PetscInt       i;

  PetscFunctionBegin;
  PetscCall(PCBDDCGraphGetDirichletDofs(pcbddc->mat_graph,&dirIS));
  if (zerodiag) {
    Mat            A;
    Vec            vec3_N;
    PetscScalar    *vals;
    const PetscInt *idxs;
    PetscInt       nz,*count;

    /* p0 */
    PetscCall(VecSet(pcis->vec1_N,0.));
    PetscCall(PetscMalloc1(pcis->n,&vals));
    PetscCall(ISGetLocalSize(zerodiag,&nz));
    PetscCall(ISGetIndices(zerodiag,&idxs));
    for (i=0;i<nz;i++) vals[i] = 1.;
    PetscCall(VecSetValues(pcis->vec1_N,nz,idxs,vals,INSERT_VALUES));
    PetscCall(VecAssemblyBegin(pcis->vec1_N));
    PetscCall(VecAssemblyEnd(pcis->vec1_N));
    /* v_I */
    PetscCall(VecSetRandom(pcis->vec2_N,NULL));
    for (i=0;i<nz;i++) vals[i] = 0.;
    PetscCall(VecSetValues(pcis->vec2_N,nz,idxs,vals,INSERT_VALUES));
    PetscCall(ISRestoreIndices(zerodiag,&idxs));
    PetscCall(ISGetIndices(pcis->is_B_local,&idxs));
    for (i=0;i<pcis->n_B;i++) vals[i] = 0.;
    PetscCall(VecSetValues(pcis->vec2_N,pcis->n_B,idxs,vals,INSERT_VALUES));
    PetscCall(ISRestoreIndices(pcis->is_B_local,&idxs));
    if (dirIS) {
      PetscInt n;

      PetscCall(ISGetLocalSize(dirIS,&n));
      PetscCall(ISGetIndices(dirIS,&idxs));
      for (i=0;i<n;i++) vals[i] = 0.;
      PetscCall(VecSetValues(pcis->vec2_N,n,idxs,vals,INSERT_VALUES));
      PetscCall(ISRestoreIndices(dirIS,&idxs));
    }
    PetscCall(VecAssemblyBegin(pcis->vec2_N));
    PetscCall(VecAssemblyEnd(pcis->vec2_N));
    PetscCall(VecDuplicate(pcis->vec1_N,&vec3_N));
    PetscCall(VecSet(vec3_N,0.));
    PetscCall(MatISGetLocalMat(pc->pmat,&A));
    PetscCall(MatMult(A,pcis->vec1_N,vec3_N));
    PetscCall(VecDot(vec3_N,pcis->vec2_N,&vals[0]));
    PetscCheck(PetscAbsScalar(vals[0]) <= 1.e-1,PETSC_COMM_SELF,PETSC_ERR_SUP,"Benign trick can not be applied! b(v_I,p_0) = %1.6e (should be numerically 0.)",(double)PetscAbsScalar(vals[0]));
    PetscCall(PetscFree(vals));
    PetscCall(VecDestroy(&vec3_N));

    /* there should not be any pressure dofs lying on the interface */
    PetscCall(PetscCalloc1(pcis->n,&count));
    PetscCall(ISGetIndices(pcis->is_B_local,&idxs));
    for (i=0;i<pcis->n_B;i++) count[idxs[i]]++;
    PetscCall(ISRestoreIndices(pcis->is_B_local,&idxs));
    PetscCall(ISGetIndices(zerodiag,&idxs));
    for (i=0;i<nz;i++) PetscCheck(!count[idxs[i]],PETSC_COMM_SELF,PETSC_ERR_SUP,"Benign trick can not be applied! pressure dof %" PetscInt_FMT " is an interface dof",idxs[i]);
    PetscCall(ISRestoreIndices(zerodiag,&idxs));
    PetscCall(PetscFree(count));
  }
  PetscCall(ISDestroy(&dirIS));

  /* check PCBDDCBenignGetOrSetP0 */
  PetscCall(VecSetRandom(pcis->vec1_global,NULL));
  for (i=0;i<pcbddc->benign_n;i++) pcbddc->benign_p0[i] = -PetscGlobalRank-i;
  PetscCall(PCBDDCBenignGetOrSetP0(pc,pcis->vec1_global,PETSC_FALSE));
  for (i=0;i<pcbddc->benign_n;i++) pcbddc->benign_p0[i] = 1;
  PetscCall(PCBDDCBenignGetOrSetP0(pc,pcis->vec1_global,PETSC_TRUE));
  for (i=0;i<pcbddc->benign_n;i++) {
    PetscInt val = PetscRealPart(pcbddc->benign_p0[i]);
    PetscCheck(val == -PetscGlobalRank-i,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error testing PCBDDCBenignGetOrSetP0! Found %g at %" PetscInt_FMT " instead of %g",(double)PetscRealPart(pcbddc->benign_p0[i]),i,(double)(-PetscGlobalRank-i));
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

  PetscFunctionBegin;
  if (reuse) goto project_b0;
  PetscCall(PetscSFDestroy(&pcbddc->benign_sf));
  PetscCall(MatDestroy(&pcbddc->benign_B0));
  for (n=0;n<pcbddc->benign_n;n++) {
    PetscCall(ISDestroy(&pcbddc->benign_zerodiag_subs[n]));
  }
  PetscCall(PetscFree(pcbddc->benign_zerodiag_subs));
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

    PetscCall(PetscMalloc1(pcbddc->n_ISForDofsLocal,&pp));
    n    = pcbddc->n_ISForDofsLocal;
    PetscOptionsBegin(PetscObjectComm((PetscObject)pc),((PetscObject)pc)->prefix,"BDDC benign options","PC");
    PetscCall(PetscOptionsIntArray("-pc_bddc_pressure_field","Field id for pressures",NULL,pp,&n,&flg));
    PetscOptionsEnd();
    if (!flg) {
      n = 1;
      pp[0] = pcbddc->n_ISForDofsLocal-1;
    }

    bsp = 0;
    for (p=0;p<n;p++) {
      PetscInt bs;

      PetscCheck(pp[p] >= 0 && pp[p] < pcbddc->n_ISForDofsLocal,PetscObjectComm((PetscObject)pc),PETSC_ERR_USER,"Invalid field id for pressures %" PetscInt_FMT,pp[p]);
      PetscCall(ISGetBlockSize(pcbddc->ISForDofsLocal[pp[p]],&bs));
      bsp += bs;
    }
    PetscCall(PetscMalloc1(bsp,&bzerodiag));
    bsp  = 0;
    for (p=0;p<n;p++) {
      const PetscInt *idxs;
      PetscInt       b,bs,npl,*bidxs;

      PetscCall(ISGetBlockSize(pcbddc->ISForDofsLocal[pp[p]],&bs));
      PetscCall(ISGetLocalSize(pcbddc->ISForDofsLocal[pp[p]],&npl));
      PetscCall(ISGetIndices(pcbddc->ISForDofsLocal[pp[p]],&idxs));
      PetscCall(PetscMalloc1(npl/bs,&bidxs));
      for (b=0;b<bs;b++) {
        PetscInt i;

        for (i=0;i<npl/bs;i++) bidxs[i] = idxs[bs*i+b];
        PetscCall(ISCreateGeneral(PETSC_COMM_SELF,npl/bs,bidxs,PETSC_COPY_VALUES,&bzerodiag[bsp]));
        bsp++;
      }
      PetscCall(PetscFree(bidxs));
      PetscCall(ISRestoreIndices(pcbddc->ISForDofsLocal[pp[p]],&idxs));
    }
    PetscCall(ISConcatenate(PETSC_COMM_SELF,bsp,bzerodiag,&pressures));

    /* remove zeroed out pressures if we are setting up a BDDC solver for a saddle-point FETI-DP */
    PetscCall(PetscObjectQuery((PetscObject)pc,"__KSPFETIDP_lP",(PetscObject*)&iP));
    if (iP) {
      IS newpressures;

      PetscCall(ISDifference(pressures,iP,&newpressures));
      PetscCall(ISDestroy(&pressures));
      pressures = newpressures;
    }
    PetscCall(ISSorted(pressures,&sorted));
    if (!sorted) {
      PetscCall(ISSort(pressures));
    }
    PetscCall(PetscFree(pp));
  }

  /* pcis has not been setup yet, so get the local size from the subdomain matrix */
  PetscCall(MatGetLocalSize(pcbddc->local_mat,&n,NULL));
  if (!n) pcbddc->benign_change_explicit = PETSC_TRUE;
  PetscCall(MatFindZeroDiagonals(pcbddc->local_mat,&zerodiag));
  PetscCall(ISSorted(zerodiag,&sorted));
  if (!sorted) {
    PetscCall(ISSort(zerodiag));
  }
  PetscCall(PetscObjectReference((PetscObject)zerodiag));
  zerodiag_save = zerodiag;
  PetscCall(ISGetLocalSize(zerodiag,&nz));
  if (!nz) {
    if (n) have_null = PETSC_FALSE;
    has_null_pressures = PETSC_FALSE;
    PetscCall(ISDestroy(&zerodiag));
  }
  recompute_zerodiag = PETSC_FALSE;

  /* in case disconnected subdomains info is present, split the pressures accordingly (otherwise the benign trick could fail) */
  zerodiag_subs    = NULL;
  benign_n         = 0;
  n_interior_dofs  = 0;
  interior_dofs    = NULL;
  nneu             = 0;
  if (pcbddc->NeumannBoundariesLocal) {
    PetscCall(ISGetLocalSize(pcbddc->NeumannBoundariesLocal,&nneu));
  }
  checkb = (PetscBool)(!pcbddc->NeumannBoundariesLocal || pcbddc->current_level);
  if (checkb) { /* need to compute interior nodes */
    PetscInt n,i,j;
    PetscInt n_neigh,*neigh,*n_shared,**shared;
    PetscInt *iwork;

    PetscCall(ISLocalToGlobalMappingGetSize(matis->rmapping,&n));
    PetscCall(ISLocalToGlobalMappingGetInfo(matis->rmapping,&n_neigh,&neigh,&n_shared,&shared));
    PetscCall(PetscCalloc1(n,&iwork));
    PetscCall(PetscMalloc1(n,&interior_dofs));
    for (i=1;i<n_neigh;i++)
      for (j=0;j<n_shared[i];j++)
          iwork[shared[i][j]] += 1;
    for (i=0;i<n;i++)
      if (!iwork[i])
        interior_dofs[n_interior_dofs++] = i;
    PetscCall(PetscFree(iwork));
    PetscCall(ISLocalToGlobalMappingRestoreInfo(matis->rmapping,&n_neigh,&neigh,&n_shared,&shared));
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
      PetscCall(VecDuplicateVecs(matis->y,2,&work));
      PetscCall(ISGetLocalSize(zerodiag,&nl));
      PetscCall(ISGetIndices(zerodiag,&idxs));
      /* work[0] = 1_p */
      PetscCall(VecSet(work[0],0.));
      PetscCall(VecGetArray(work[0],&array));
      for (j=0;j<nl;j++) array[idxs[j]] = 1.;
      PetscCall(VecRestoreArray(work[0],&array));
      /* work[0] = 1_v */
      PetscCall(VecSet(work[1],1.));
      PetscCall(VecGetArray(work[1],&array));
      for (j=0;j<nl;j++) array[idxs[j]] = 0.;
      PetscCall(VecRestoreArray(work[1],&array));
      PetscCall(ISRestoreIndices(zerodiag,&idxs));
    }

    if (nsubs > 1 || bsp > 1) {
      IS       *is;
      PetscInt b,totb;

      totb  = bsp;
      is    = bsp > 1 ? bzerodiag : &zerodiag;
      nsubs = PetscMax(nsubs,1);
      PetscCall(PetscCalloc1(nsubs*totb,&zerodiag_subs));
      for (b=0;b<totb;b++) {
        for (i=0;i<nsubs;i++) {
          ISLocalToGlobalMapping l2g;
          IS                     t_zerodiag_subs;
          PetscInt               nl;

          if (subs) {
            PetscCall(ISLocalToGlobalMappingCreateIS(subs[i],&l2g));
          } else {
            IS tis;

            PetscCall(MatGetLocalSize(pcbddc->local_mat,&nl,NULL));
            PetscCall(ISCreateStride(PETSC_COMM_SELF,nl,0,1,&tis));
            PetscCall(ISLocalToGlobalMappingCreateIS(tis,&l2g));
            PetscCall(ISDestroy(&tis));
          }
          PetscCall(ISGlobalToLocalMappingApplyIS(l2g,IS_GTOLM_DROP,is[b],&t_zerodiag_subs));
          PetscCall(ISGetLocalSize(t_zerodiag_subs,&nl));
          if (nl) {
            PetscBool valid = PETSC_TRUE;

            if (checkb) {
              PetscCall(VecSet(matis->x,0));
              PetscCall(ISGetLocalSize(subs[i],&nl));
              PetscCall(ISGetIndices(subs[i],&idxs));
              PetscCall(VecGetArray(matis->x,&array));
              for (j=0;j<nl;j++) array[idxs[j]] = 1.;
              PetscCall(VecRestoreArray(matis->x,&array));
              PetscCall(ISRestoreIndices(subs[i],&idxs));
              PetscCall(VecPointwiseMult(matis->x,work[0],matis->x));
              PetscCall(MatMult(matis->A,matis->x,matis->y));
              PetscCall(VecPointwiseMult(matis->y,work[1],matis->y));
              PetscCall(VecGetArray(matis->y,&array));
              for (j=0;j<n_interior_dofs;j++) {
                if (PetscAbsScalar(array[interior_dofs[j]]) > PETSC_SMALL) {
                  valid = PETSC_FALSE;
                  break;
                }
              }
              PetscCall(VecRestoreArray(matis->y,&array));
            }
            if (valid && nneu) {
              const PetscInt *idxs;
              PetscInt       nzb;

              PetscCall(ISGetIndices(pcbddc->NeumannBoundariesLocal,&idxs));
              PetscCall(ISGlobalToLocalMappingApply(l2g,IS_GTOLM_DROP,nneu,idxs,&nzb,NULL));
              PetscCall(ISRestoreIndices(pcbddc->NeumannBoundariesLocal,&idxs));
              if (nzb) valid = PETSC_FALSE;
            }
            if (valid && pressures) {
              IS       t_pressure_subs,tmp;
              PetscInt i1,i2;

              PetscCall(ISGlobalToLocalMappingApplyIS(l2g,IS_GTOLM_DROP,pressures,&t_pressure_subs));
              PetscCall(ISEmbed(t_zerodiag_subs,t_pressure_subs,PETSC_TRUE,&tmp));
              PetscCall(ISGetLocalSize(tmp,&i1));
              PetscCall(ISGetLocalSize(t_zerodiag_subs,&i2));
              if (i2 != i1) valid = PETSC_FALSE;
              PetscCall(ISDestroy(&t_pressure_subs));
              PetscCall(ISDestroy(&tmp));
            }
            if (valid) {
              PetscCall(ISLocalToGlobalMappingApplyIS(l2g,t_zerodiag_subs,&zerodiag_subs[benign_n]));
              benign_n++;
            } else recompute_zerodiag = PETSC_TRUE;
          }
          PetscCall(ISDestroy(&t_zerodiag_subs));
          PetscCall(ISLocalToGlobalMappingDestroy(&l2g));
        }
      }
    } else { /* there's just one subdomain (or zero if they have not been detected */
      PetscBool valid = PETSC_TRUE;

      if (nneu) valid = PETSC_FALSE;
      if (valid && pressures) {
        PetscCall(ISEqual(pressures,zerodiag,&valid));
      }
      if (valid && checkb) {
        PetscCall(MatMult(matis->A,work[0],matis->x));
        PetscCall(VecPointwiseMult(matis->x,work[1],matis->x));
        PetscCall(VecGetArray(matis->x,&array));
        for (j=0;j<n_interior_dofs;j++) {
          if (PetscAbsScalar(array[interior_dofs[j]]) > PETSC_SMALL) {
            valid = PETSC_FALSE;
            break;
          }
        }
        PetscCall(VecRestoreArray(matis->x,&array));
      }
      if (valid) {
        benign_n = 1;
        PetscCall(PetscMalloc1(benign_n,&zerodiag_subs));
        PetscCall(PetscObjectReference((PetscObject)zerodiag));
        zerodiag_subs[0] = zerodiag;
      }
    }
    if (checkb) {
      PetscCall(VecDestroyVecs(2,&work));
    }
  }
  PetscCall(PetscFree(interior_dofs));

  if (!benign_n) {
    PetscInt n;

    PetscCall(ISDestroy(&zerodiag));
    recompute_zerodiag = PETSC_FALSE;
    PetscCall(MatGetLocalSize(pcbddc->local_mat,&n,NULL));
    if (n) have_null = PETSC_FALSE;
  }

  /* final check for null pressures */
  if (zerodiag && pressures) {
    PetscCall(ISEqual(pressures,zerodiag,&have_null));
  }

  if (recompute_zerodiag) {
    PetscCall(ISDestroy(&zerodiag));
    if (benign_n == 1) {
      PetscCall(PetscObjectReference((PetscObject)zerodiag_subs[0]));
      zerodiag = zerodiag_subs[0];
    } else {
      PetscInt i,nzn,*new_idxs;

      nzn = 0;
      for (i=0;i<benign_n;i++) {
        PetscInt ns;
        PetscCall(ISGetLocalSize(zerodiag_subs[i],&ns));
        nzn += ns;
      }
      PetscCall(PetscMalloc1(nzn,&new_idxs));
      nzn = 0;
      for (i=0;i<benign_n;i++) {
        PetscInt ns,*idxs;
        PetscCall(ISGetLocalSize(zerodiag_subs[i],&ns));
        PetscCall(ISGetIndices(zerodiag_subs[i],(const PetscInt**)&idxs));
        PetscCall(PetscArraycpy(new_idxs+nzn,idxs,ns));
        PetscCall(ISRestoreIndices(zerodiag_subs[i],(const PetscInt**)&idxs));
        nzn += ns;
      }
      PetscCall(PetscSortInt(nzn,new_idxs));
      PetscCall(ISCreateGeneral(PETSC_COMM_SELF,nzn,new_idxs,PETSC_OWN_POINTER,&zerodiag));
    }
    have_null = PETSC_FALSE;
  }

  /* determines if the coarse solver will be singular or not */
  PetscCall(MPIU_Allreduce(&have_null,&pcbddc->benign_null,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)pc)));

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
    PetscCall(MatISGetLocalToGlobalMapping(pc->pmat,&l2gmap,NULL));
    PetscCall(MatISGetLocalMat(pc->pmat,&A));
    PetscCall(MatGetLocalSize(A,&n,NULL));
    PetscCheck(isused || (n == 0),PETSC_COMM_SELF,PETSC_ERR_USER,"Don't know how to extract div u dot p! Please provide the pressure field");
    n_isused = 0;
    if (isused) {
      PetscCall(ISGetLocalSize(isused,&n_isused));
    }
    PetscCallMPI(MPI_Scan(&n_isused,&st,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)pc)));
    st = st-n_isused;
    if (n) {
      const PetscInt *gidxs;

      PetscCall(MatCreateSubMatrix(A,isused,NULL,MAT_INITIAL_MATRIX,&loc_divudotp));
      PetscCall(ISLocalToGlobalMappingGetIndices(l2gmap,&gidxs));
      /* TODO: extend ISCreateStride with st = PETSC_DECIDE */
      PetscCall(ISCreateStride(PetscObjectComm((PetscObject)pc),n_isused,st,1,&row));
      PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)pc),n,gidxs,PETSC_COPY_VALUES,&col));
      PetscCall(ISLocalToGlobalMappingRestoreIndices(l2gmap,&gidxs));
    } else {
      PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF,0,0,1,NULL,&loc_divudotp));
      PetscCall(ISCreateStride(PetscObjectComm((PetscObject)pc),n_isused,st,1,&row));
      PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)pc),0,NULL,PETSC_COPY_VALUES,&col));
    }
    PetscCall(MatGetSize(pc->pmat,NULL,&N));
    PetscCall(ISGetSize(row,&M));
    PetscCall(ISLocalToGlobalMappingCreateIS(row,&rl2g));
    PetscCall(ISLocalToGlobalMappingCreateIS(col,&cl2g));
    PetscCall(ISDestroy(&row));
    PetscCall(ISDestroy(&col));
    PetscCall(MatCreate(PetscObjectComm((PetscObject)pc),&pcbddc->divudotp));
    PetscCall(MatSetType(pcbddc->divudotp,MATIS));
    PetscCall(MatSetSizes(pcbddc->divudotp,PETSC_DECIDE,PETSC_DECIDE,M,N));
    PetscCall(MatSetLocalToGlobalMapping(pcbddc->divudotp,rl2g,cl2g));
    PetscCall(ISLocalToGlobalMappingDestroy(&rl2g));
    PetscCall(ISLocalToGlobalMappingDestroy(&cl2g));
    PetscCall(MatISSetLocalMat(pcbddc->divudotp,loc_divudotp));
    PetscCall(MatDestroy(&loc_divudotp));
    PetscCall(MatAssemblyBegin(pcbddc->divudotp,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(pcbddc->divudotp,MAT_FINAL_ASSEMBLY));
  }
  PetscCall(ISDestroy(&zerodiag_save));
  PetscCall(ISDestroy(&pressures));
  if (bzerodiag) {
    PetscInt i;

    for (i=0;i<bsp;i++) {
      PetscCall(ISDestroy(&bzerodiag[i]));
    }
    PetscCall(PetscFree(bzerodiag));
  }
  pcbddc->benign_n = benign_n;
  pcbddc->benign_zerodiag_subs = zerodiag_subs;

  /* determines if the problem has subdomains with 0 pressure block */
  have_null = (PetscBool)(!!pcbddc->benign_n);
  PetscCall(MPIU_Allreduce(&have_null,&pcbddc->benign_have_null,1,MPIU_BOOL,MPI_LOR,PetscObjectComm((PetscObject)pc)));

project_b0:
  PetscCall(MatGetLocalSize(pcbddc->local_mat,&n,NULL));
  /* change of basis and p0 dofs */
  if (pcbddc->benign_n) {
    PetscInt i,s,*nnz;

    /* local change of basis for pressures */
    PetscCall(MatDestroy(&pcbddc->benign_change));
    PetscCall(MatCreate(PetscObjectComm((PetscObject)pcbddc->local_mat),&pcbddc->benign_change));
    PetscCall(MatSetType(pcbddc->benign_change,MATAIJ));
    PetscCall(MatSetSizes(pcbddc->benign_change,n,n,PETSC_DECIDE,PETSC_DECIDE));
    PetscCall(PetscMalloc1(n,&nnz));
    for (i=0;i<n;i++) nnz[i] = 1; /* defaults to identity */
    for (i=0;i<pcbddc->benign_n;i++) {
      const PetscInt *idxs;
      PetscInt       nzs,j;

      PetscCall(ISGetLocalSize(pcbddc->benign_zerodiag_subs[i],&nzs));
      PetscCall(ISGetIndices(pcbddc->benign_zerodiag_subs[i],&idxs));
      for (j=0;j<nzs-1;j++) nnz[idxs[j]] = 2; /* change on pressures */
      nnz[idxs[nzs-1]] = nzs; /* last local pressure dof in subdomain */
      PetscCall(ISRestoreIndices(pcbddc->benign_zerodiag_subs[i],&idxs));
    }
    PetscCall(MatSeqAIJSetPreallocation(pcbddc->benign_change,0,nnz));
    PetscCall(MatSetOption(pcbddc->benign_change,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE));
    PetscCall(PetscFree(nnz));
    /* set identity by default */
    for (i=0;i<n;i++) {
      PetscCall(MatSetValue(pcbddc->benign_change,i,i,1.,INSERT_VALUES));
    }
    PetscCall(PetscFree3(pcbddc->benign_p0_lidx,pcbddc->benign_p0_gidx,pcbddc->benign_p0));
    PetscCall(PetscMalloc3(pcbddc->benign_n,&pcbddc->benign_p0_lidx,pcbddc->benign_n,&pcbddc->benign_p0_gidx,pcbddc->benign_n,&pcbddc->benign_p0));
    /* set change on pressures */
    for (s=0;s<pcbddc->benign_n;s++) {
      PetscScalar    *array;
      const PetscInt *idxs;
      PetscInt       nzs;

      PetscCall(ISGetLocalSize(pcbddc->benign_zerodiag_subs[s],&nzs));
      PetscCall(ISGetIndices(pcbddc->benign_zerodiag_subs[s],&idxs));
      for (i=0;i<nzs-1;i++) {
        PetscScalar vals[2];
        PetscInt    cols[2];

        cols[0] = idxs[i];
        cols[1] = idxs[nzs-1];
        vals[0] = 1.;
        vals[1] = 1.;
        PetscCall(MatSetValues(pcbddc->benign_change,1,cols,2,cols,vals,INSERT_VALUES));
      }
      PetscCall(PetscMalloc1(nzs,&array));
      for (i=0;i<nzs-1;i++) array[i] = -1.;
      array[nzs-1] = 1.;
      PetscCall(MatSetValues(pcbddc->benign_change,1,idxs+nzs-1,nzs,idxs,array,INSERT_VALUES));
      /* store local idxs for p0 */
      pcbddc->benign_p0_lidx[s] = idxs[nzs-1];
      PetscCall(ISRestoreIndices(pcbddc->benign_zerodiag_subs[s],&idxs));
      PetscCall(PetscFree(array));
    }
    PetscCall(MatAssemblyBegin(pcbddc->benign_change,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(pcbddc->benign_change,MAT_FINAL_ASSEMBLY));

    /* project if needed */
    if (pcbddc->benign_change_explicit) {
      Mat M;

      PetscCall(MatPtAP(pcbddc->local_mat,pcbddc->benign_change,MAT_INITIAL_MATRIX,2.0,&M));
      PetscCall(MatDestroy(&pcbddc->local_mat));
      PetscCall(MatSeqAIJCompress(M,&pcbddc->local_mat));
      PetscCall(MatDestroy(&M));
    }
    /* store global idxs for p0 */
    PetscCall(ISLocalToGlobalMappingApply(matis->rmapping,pcbddc->benign_n,pcbddc->benign_p0_lidx,pcbddc->benign_p0_gidx));
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
    PetscCall(PetscSFCreate(PetscObjectComm((PetscObject)pc),&pcbddc->benign_sf));
    PetscCall(PetscSFSetGraphLayout(pcbddc->benign_sf,pc->pmat->rmap,pcbddc->benign_n,NULL,PETSC_OWN_POINTER,pcbddc->benign_p0_gidx));
  }
  if (get) {
    PetscCall(VecGetArrayRead(v,(const PetscScalar**)&array));
    PetscCall(PetscSFBcastBegin(pcbddc->benign_sf,MPIU_SCALAR,array,pcbddc->benign_p0,MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(pcbddc->benign_sf,MPIU_SCALAR,array,pcbddc->benign_p0,MPI_REPLACE));
    PetscCall(VecRestoreArrayRead(v,(const PetscScalar**)&array));
  } else {
    PetscCall(VecGetArray(v,&array));
    PetscCall(PetscSFReduceBegin(pcbddc->benign_sf,MPIU_SCALAR,pcbddc->benign_p0,array,MPI_REPLACE));
    PetscCall(PetscSFReduceEnd(pcbddc->benign_sf,MPIU_SCALAR,pcbddc->benign_p0,array,MPI_REPLACE));
    PetscCall(VecRestoreArray(v,&array));
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
      PetscCall(ISCreateGeneral(PETSC_COMM_SELF,pcbddc->benign_n,pcbddc->benign_p0_lidx,PETSC_COPY_VALUES,&is_p0));
      PetscCall(MatCreateSubMatrix(pcbddc->local_mat,is_p0,NULL,reuse,&pcbddc->benign_B0));
      /* remove rows and cols from local problem */
      PetscCall(MatSetOption(pcbddc->local_mat,MAT_KEEP_NONZERO_PATTERN,PETSC_TRUE));
      PetscCall(MatSetOption(pcbddc->local_mat,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_FALSE));
      PetscCall(MatZeroRowsColumnsIS(pcbddc->local_mat,is_p0,1.0,NULL,NULL));
      PetscCall(ISDestroy(&is_p0));
    } else {
      Mat_IS      *matis = (Mat_IS*)pc->pmat->data;
      PetscScalar *vals;
      PetscInt    i,n,*idxs_ins;

      PetscCall(VecGetLocalSize(matis->y,&n));
      PetscCall(PetscMalloc2(n,&idxs_ins,n,&vals));
      if (!pcbddc->benign_B0) {
        PetscInt *nnz;
        PetscCall(MatCreate(PetscObjectComm((PetscObject)pcbddc->local_mat),&pcbddc->benign_B0));
        PetscCall(MatSetType(pcbddc->benign_B0,MATAIJ));
        PetscCall(MatSetSizes(pcbddc->benign_B0,pcbddc->benign_n,n,PETSC_DECIDE,PETSC_DECIDE));
        PetscCall(PetscMalloc1(pcbddc->benign_n,&nnz));
        for (i=0;i<pcbddc->benign_n;i++) {
          PetscCall(ISGetLocalSize(pcbddc->benign_zerodiag_subs[i],&nnz[i]));
          nnz[i] = n - nnz[i];
        }
        PetscCall(MatSeqAIJSetPreallocation(pcbddc->benign_B0,0,nnz));
        PetscCall(MatSetOption(pcbddc->benign_B0,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE));
        PetscCall(PetscFree(nnz));
      }

      for (i=0;i<pcbddc->benign_n;i++) {
        PetscScalar *array;
        PetscInt    *idxs,j,nz,cum;

        PetscCall(VecSet(matis->x,0.));
        PetscCall(ISGetLocalSize(pcbddc->benign_zerodiag_subs[i],&nz));
        PetscCall(ISGetIndices(pcbddc->benign_zerodiag_subs[i],(const PetscInt**)&idxs));
        for (j=0;j<nz;j++) vals[j] = 1.;
        PetscCall(VecSetValues(matis->x,nz,idxs,vals,INSERT_VALUES));
        PetscCall(VecAssemblyBegin(matis->x));
        PetscCall(VecAssemblyEnd(matis->x));
        PetscCall(VecSet(matis->y,0.));
        PetscCall(MatMult(matis->A,matis->x,matis->y));
        PetscCall(VecGetArray(matis->y,&array));
        cum = 0;
        for (j=0;j<n;j++) {
          if (PetscUnlikely(PetscAbsScalar(array[j]) > PETSC_SMALL)) {
            vals[cum] = array[j];
            idxs_ins[cum] = j;
            cum++;
          }
        }
        PetscCall(MatSetValues(pcbddc->benign_B0,1,&i,cum,idxs_ins,vals,INSERT_VALUES));
        PetscCall(VecRestoreArray(matis->y,&array));
        PetscCall(ISRestoreIndices(pcbddc->benign_zerodiag_subs[i],(const PetscInt**)&idxs));
      }
      PetscCall(MatAssemblyBegin(pcbddc->benign_B0,MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(pcbddc->benign_B0,MAT_FINAL_ASSEMBLY));
      PetscCall(PetscFree2(idxs_ins,vals));
    }
  } else { /* push */
    if (pcbddc->benign_change_explicit) {
      PetscInt i;

      for (i=0;i<pcbddc->benign_n;i++) {
        PetscScalar *B0_vals;
        PetscInt    *B0_cols,B0_ncol;

        PetscCall(MatGetRow(pcbddc->benign_B0,i,&B0_ncol,(const PetscInt**)&B0_cols,(const PetscScalar**)&B0_vals));
        PetscCall(MatSetValues(pcbddc->local_mat,1,pcbddc->benign_p0_lidx+i,B0_ncol,B0_cols,B0_vals,INSERT_VALUES));
        PetscCall(MatSetValues(pcbddc->local_mat,B0_ncol,B0_cols,1,pcbddc->benign_p0_lidx+i,B0_vals,INSERT_VALUES));
        PetscCall(MatSetValue(pcbddc->local_mat,pcbddc->benign_p0_lidx[i],pcbddc->benign_p0_lidx[i],0.0,INSERT_VALUES));
        PetscCall(MatRestoreRow(pcbddc->benign_B0,i,&B0_ncol,(const PetscInt**)&B0_cols,(const PetscScalar**)&B0_vals));
      }
      PetscCall(MatAssemblyBegin(pcbddc->local_mat,MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(pcbddc->local_mat,MAT_FINAL_ASSEMBLY));
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

  PetscFunctionBegin;
  PetscCheck(sub_schurs,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Adaptive selection of constraints requires SubSchurs data");
  PetscCheck(sub_schurs->schur_explicit,PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Adaptive selection of constraints requires MUMPS and/or MKL_CPARDISO");
  PetscCheck(!sub_schurs->n_subs || !(!sub_schurs->is_symmetric),PETSC_COMM_SELF,PETSC_ERR_SUP,"Adaptive selection not yet implemented for this matrix pencil (herm %d, symm %d, posdef %d)",sub_schurs->is_hermitian,sub_schurs->is_symmetric,sub_schurs->is_posdef);
  PetscCall(PetscLogEventBegin(PC_BDDC_AdaptiveSetUp[pcbddc->current_level],pc,0,0,0));

  if (pcbddc->dbg_flag) {
    PetscCall(PetscViewerFlush(pcbddc->dbg_viewer));
    PetscCall(PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"--------------------------------------------------\n"));
    PetscCall(PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"Check adaptive selection of constraints\n"));
    PetscCall(PetscViewerASCIIPushSynchronized(pcbddc->dbg_viewer));
  }

  if (pcbddc->dbg_flag) {
    PetscCall(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d cc %" PetscInt_FMT " (%d,%d).\n",PetscGlobalRank,sub_schurs->n_subs,sub_schurs->is_hermitian,sub_schurs->is_posdef));
  }

  /* max size of subsets */
  mss = 0;
  for (i=0;i<sub_schurs->n_subs;i++) {
    PetscInt subset_size;

    PetscCall(ISGetLocalSize(sub_schurs->is_subs[i],&subset_size));
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

    PetscCall(ISGetLocalSize(sub_schurs->is_subs[i],&subset_size));
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
      PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
#if defined(PETSC_USE_COMPLEX)
      PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","V","L",&B_N,St,&B_N,S,&B_N,&zero,&thresh,&B_dummyint,&B_dummyint,&eps,&B_neigs,eigs,eigv,&B_N,&lwork,&B_lwork,rwork,B_iwork,B_ifail,&B_ierr));
#else
      PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","V","L",&B_N,St,&B_N,S,&B_N,&zero,&thresh,&B_dummyint,&B_dummyint,&eps,&B_neigs,eigs,eigv,&B_N,&lwork,&B_lwork,B_iwork,B_ifail,&B_ierr));
#endif
      PetscCheck(B_ierr == 0,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in query to SYGVX Lapack routine %d",(int)B_ierr);
      PetscCall(PetscFPTrapPop());
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not yet implemented");
  }

  nv = 0;
  if (sub_schurs->is_vertices && pcbddc->use_vertices) { /* complement set of active subsets, each entry is a vertex (boundary made by active subsets, vertices and dirichlet dofs) */
    PetscCall(ISGetLocalSize(sub_schurs->is_vertices,&nv));
  }
  PetscCall(PetscBLASIntCast((PetscInt)PetscRealPart(lwork),&B_lwork));
  if (allocated_S_St) {
    PetscCall(PetscMalloc2(mss*mss,&S,mss*mss,&St));
  }
  PetscCall(PetscMalloc5(mss*mss,&eigv,mss,&eigs,B_lwork,&work,5*mss,&B_iwork,mss,&B_ifail));
#if defined(PETSC_USE_COMPLEX)
  PetscCall(PetscMalloc1(7*mss,&rwork));
#endif
  PetscCall(PetscMalloc5(nv+sub_schurs->n_subs,&pcbddc->adaptive_constraints_n,
                         nv+sub_schurs->n_subs+1,&pcbddc->adaptive_constraints_idxs_ptr,
                         nv+sub_schurs->n_subs+1,&pcbddc->adaptive_constraints_data_ptr,
                         nv+cum,&pcbddc->adaptive_constraints_idxs,
                         nv+cum2,&pcbddc->adaptive_constraints_data));
  PetscCall(PetscArrayzero(pcbddc->adaptive_constraints_n,nv+sub_schurs->n_subs));

  maxneigs = 0;
  cum = cumarray = 0;
  pcbddc->adaptive_constraints_idxs_ptr[0] = 0;
  pcbddc->adaptive_constraints_data_ptr[0] = 0;
  if (sub_schurs->is_vertices && pcbddc->use_vertices) {
    const PetscInt *idxs;

    PetscCall(ISGetIndices(sub_schurs->is_vertices,&idxs));
    for (cum=0;cum<nv;cum++) {
      pcbddc->adaptive_constraints_n[cum] = 1;
      pcbddc->adaptive_constraints_idxs[cum] = idxs[cum];
      pcbddc->adaptive_constraints_data[cum] = 1.0;
      pcbddc->adaptive_constraints_idxs_ptr[cum+1] = pcbddc->adaptive_constraints_idxs_ptr[cum]+1;
      pcbddc->adaptive_constraints_data_ptr[cum+1] = pcbddc->adaptive_constraints_data_ptr[cum]+1;
    }
    PetscCall(ISRestoreIndices(sub_schurs->is_vertices,&idxs));
  }

  if (mss) { /* multilevel */
    PetscCall(MatSeqAIJGetArray(sub_schurs->sum_S_Ej_inv_all,&Sarray));
    PetscCall(MatSeqAIJGetArray(sub_schurs->sum_S_Ej_tilda_all,&Starray));
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
      PetscCheck(sub_schurs->is_posdef,PETSC_COMM_SELF,PETSC_ERR_SUP,"Not yet implemented without deluxe scaling");
      upper = 1./uthresh;
      lower = 0.;
    }
    PetscCall(ISGetLocalSize(sub_schurs->is_subs[i],&subset_size));
    PetscCall(ISGetIndices(sub_schurs->is_subs[i],&idxs));
    PetscCall(PetscBLASIntCast(subset_size,&B_N));
    /* this is experimental: we assume the dofs have been properly grouped to have
       the diagonal blocks Schur complements either positive or negative definite (true for Stokes) */
    if (!sub_schurs->is_posdef) {
      Mat T;

      for (j=0;j<subset_size;j++) {
        if (PetscRealPart(*(Sarray+cumarray+j*(subset_size+1))) < 0.0) {
          PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,subset_size,subset_size,Sarray+cumarray,&T));
          PetscCall(MatScale(T,-1.0));
          PetscCall(MatDestroy(&T));
          PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,subset_size,subset_size,Starray+cumarray,&T));
          PetscCall(MatScale(T,-1.0));
          PetscCall(MatDestroy(&T));
          if (sub_schurs->change_primal_sub) {
            PetscInt       nz,k;
            const PetscInt *idxs;

            PetscCall(ISGetLocalSize(sub_schurs->change_primal_sub[i],&nz));
            PetscCall(ISGetIndices(sub_schurs->change_primal_sub[i],&idxs));
            for (k=0;k<nz;k++) {
              *( Sarray + cumarray + idxs[k]*(subset_size+1)) *= -1.0;
              *(Starray + cumarray + idxs[k]*(subset_size+1))  = 0.0;
            }
            PetscCall(ISRestoreIndices(sub_schurs->change_primal_sub[i],&idxs));
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
          PetscCall(PetscArrayzero(S,subset_size*subset_size));
          PetscCall(PetscArrayzero(St,subset_size*subset_size));
        }
        for (j=0;j<subset_size;j++) {
          for (k=j;k<subset_size;k++) {
            S [j*subset_size+k] = Sarray [cumarray+j*subset_size+k];
            St[j*subset_size+k] = Starray[cumarray+j*subset_size+k];
          }
        }
      } else {
        PetscCall(PetscArraycpy(S,Sarray+cumarray,subset_size*subset_size));
        PetscCall(PetscArraycpy(St,Starray+cumarray,subset_size*subset_size));
      }
    } else {
      S = Sarray + cumarray;
      St = Starray + cumarray;
    }
    /* see if we can save some work */
    if (sub_schurs->n_subs == 1 && pcbddc->use_deluxe_scaling) {
      PetscCall(PetscArraycmp(S,St,subset_size*subset_size,&same_data));
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
            PetscCall(ISGetLocalSize(sub_schurs->change_primal_sub[i],&nc));
          }
          PetscCall(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Computing for sub %" PetscInt_FMT "/%" PetscInt_FMT " size %" PetscInt_FMT " count %" PetscInt_FMT " fid %" PetscInt_FMT " (range %d) (change %" PetscInt_FMT ").\n",i,sub_schurs->n_subs,subset_size,pcbddc->mat_graph->count[idxs[0]]+1,pcbddc->mat_graph->which_dof[idxs[0]],compute_range,nc));
        }

        PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
        if (compute_range) {

          /* ask for eigenvalues larger than thresh */
          if (sub_schurs->is_posdef) {
#if defined(PETSC_USE_COMPLEX)
            PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","V","L",&B_N,St,&B_N,S,&B_N,&lower,&upper,&B_IL,&B_IU,&eps,&B_neigs,eigs,eigv,&B_N,work,&B_lwork,rwork,B_iwork,B_ifail,&B_ierr));
#else
            PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","V","L",&B_N,St,&B_N,S,&B_N,&lower,&upper,&B_IL,&B_IU,&eps,&B_neigs,eigs,eigv,&B_N,work,&B_lwork,B_iwork,B_ifail,&B_ierr));
#endif
            PetscCall(PetscLogFlops((4.0*subset_size*subset_size*subset_size)/3.0));
          } else { /* no theory so far, but it works nicely */
            PetscInt  recipe = 0,recipe_m = 1;
            PetscReal bb[2];

            PetscCall(PetscOptionsGetInt(NULL,((PetscObject)pc)->prefix,"-pc_bddc_adaptive_recipe",&recipe,NULL));
            switch (recipe) {
            case 0:
              if (scal) { bb[0] = PETSC_MIN_REAL; bb[1] = lthresh; }
              else { bb[0] = uthresh; bb[1] = PETSC_MAX_REAL; }
#if defined(PETSC_USE_COMPLEX)
              PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","V","L",&B_N,St,&B_N,S,&B_N,&bb[0],&bb[1],&B_IL,&B_IU,&eps,&B_neigs,eigs,eigv,&B_N,work,&B_lwork,rwork,B_iwork,B_ifail,&B_ierr));
#else
              PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","V","L",&B_N,St,&B_N,S,&B_N,&bb[0],&bb[1],&B_IL,&B_IU,&eps,&B_neigs,eigs,eigv,&B_N,work,&B_lwork,B_iwork,B_ifail,&B_ierr));
#endif
              PetscCall(PetscLogFlops((4.0*subset_size*subset_size*subset_size)/3.0));
              break;
            case 1:
              bb[0] = PETSC_MIN_REAL; bb[1] = lthresh*lthresh;
#if defined(PETSC_USE_COMPLEX)
              PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","V","L",&B_N,St,&B_N,S,&B_N,&bb[0],&bb[1],&B_IL,&B_IU,&eps,&B_neigs,eigs,eigv,&B_N,work,&B_lwork,rwork,B_iwork,B_ifail,&B_ierr));
#else
              PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","V","L",&B_N,St,&B_N,S,&B_N,&bb[0],&bb[1],&B_IL,&B_IU,&eps,&B_neigs,eigs,eigv,&B_N,work,&B_lwork,B_iwork,B_ifail,&B_ierr));
#endif
              PetscCall(PetscLogFlops((4.0*subset_size*subset_size*subset_size)/3.0));
              if (!scal) {
                PetscBLASInt B_neigs2 = 0;

                bb[0] = PetscMax(lthresh*lthresh,uthresh); bb[1] = PETSC_MAX_REAL;
                PetscCall(PetscArraycpy(S,Sarray+cumarray,subset_size*subset_size));
                PetscCall(PetscArraycpy(St,Starray+cumarray,subset_size*subset_size));
#if defined(PETSC_USE_COMPLEX)
                PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","V","L",&B_N,St,&B_N,S,&B_N,&bb[0],&bb[1],&B_IL,&B_IU,&eps,&B_neigs2,eigs+B_neigs,eigv+B_neigs*B_N,&B_N,work,&B_lwork,rwork,B_iwork,B_ifail,&B_ierr));
#else
                PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","V","L",&B_N,St,&B_N,S,&B_N,&bb[0],&bb[1],&B_IL,&B_IU,&eps,&B_neigs2,eigs+B_neigs,eigv+B_neigs*B_N,&B_N,work,&B_lwork,B_iwork,B_ifail,&B_ierr));
#endif
                PetscCall(PetscLogFlops((4.0*subset_size*subset_size*subset_size)/3.0));
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
                PetscCall(PetscLogFlops((4.0*subset_size*subset_size*subset_size)/3.0));
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
                  PetscCall(PetscLogFlops((4.0*subset_size*subset_size*subset_size)/3.0));
                }
                bb[0] = PetscMax(lthresh*lthresh,uthresh);
                bb[1] = PETSC_MAX_REAL;
                if (import) {
                  PetscCall(PetscArraycpy(S,Sarray+cumarray,subset_size*subset_size));
                  PetscCall(PetscArraycpy(St,Starray+cumarray,subset_size*subset_size));
                }
#if defined(PETSC_USE_COMPLEX)
                PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","V","L",&B_N,St,&B_N,S,&B_N,&bb[0],&bb[1],&B_IL,&B_IU,&eps,&B_neigs2,eigs+B_neigs,eigv+B_neigs*B_N,&B_N,work,&B_lwork,rwork,B_iwork,B_ifail,&B_ierr));
#else
                PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","V","L",&B_N,St,&B_N,S,&B_N,&bb[0],&bb[1],&B_IL,&B_IU,&eps,&B_neigs2,eigs+B_neigs,eigv+B_neigs*B_N,&B_N,work,&B_lwork,B_iwork,B_ifail,&B_ierr));
#endif
                PetscCall(PetscLogFlops((4.0*subset_size*subset_size*subset_size)/3.0));
                B_neigs += B_neigs2;
              }
              break;
            case 3:
              if (scal) {
                PetscCall(PetscOptionsGetInt(NULL,((PetscObject)pc)->prefix,"-pc_bddc_adaptive_recipe3_min_scal",&recipe_m,NULL));
              } else {
                PetscCall(PetscOptionsGetInt(NULL,((PetscObject)pc)->prefix,"-pc_bddc_adaptive_recipe3_min",&recipe_m,NULL));
              }
              if (!scal) {
                bb[0] = uthresh;
                bb[1] = PETSC_MAX_REAL;
#if defined(PETSC_USE_COMPLEX)
                PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","V","L",&B_N,St,&B_N,S,&B_N,&bb[0],&bb[1],&B_IL,&B_IU,&eps,&B_neigs,eigs,eigv,&B_N,work,&B_lwork,rwork,B_iwork,B_ifail,&B_ierr));
#else
                PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","V","L",&B_N,St,&B_N,S,&B_N,&bb[0],&bb[1],&B_IL,&B_IU,&eps,&B_neigs,eigs,eigv,&B_N,work,&B_lwork,B_iwork,B_ifail,&B_ierr));
#endif
                PetscCall(PetscLogFlops((4.0*subset_size*subset_size*subset_size)/3.0));
              }
              if (recipe_m > 0 && B_N - B_neigs > 0) {
                PetscBLASInt B_neigs2 = 0;

                B_IL = 1;
                PetscCall(PetscBLASIntCast(PetscMin(recipe_m,B_N - B_neigs),&B_IU));
                PetscCall(PetscArraycpy(S,Sarray+cumarray,subset_size*subset_size));
                PetscCall(PetscArraycpy(St,Starray+cumarray,subset_size*subset_size));
#if defined(PETSC_USE_COMPLEX)
                PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","I","L",&B_N,St,&B_N,S,&B_N,&lower,&upper,&B_IL,&B_IU,&eps,&B_neigs2,eigs+B_neigs,eigv+B_neigs*B_N,&B_N,work,&B_lwork,rwork,B_iwork,B_ifail,&B_ierr));
#else
                PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","I","L",&B_N,St,&B_N,S,&B_N,&lower,&upper,&B_IL,&B_IU,&eps,&B_neigs2,eigs+B_neigs,eigv+B_neigs*B_N,&B_N,work,&B_lwork,B_iwork,B_ifail,&B_ierr));
#endif
                PetscCall(PetscLogFlops((4.0*subset_size*subset_size*subset_size)/3.0));
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
              PetscCall(PetscLogFlops((4.0*subset_size*subset_size*subset_size)/3.0));
              {
                PetscBLASInt B_neigs2 = 0;

                bb[0] = PetscMax(lthresh+PETSC_SMALL,uthresh); bb[1] = PETSC_MAX_REAL;
                PetscCall(PetscArraycpy(S,Sarray+cumarray,subset_size*subset_size));
                PetscCall(PetscArraycpy(St,Starray+cumarray,subset_size*subset_size));
#if defined(PETSC_USE_COMPLEX)
                PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","V","L",&B_N,St,&B_N,S,&B_N,&bb[0],&bb[1],&B_IL,&B_IU,&eps,&B_neigs2,eigs+B_neigs,eigv+B_neigs*B_N,&B_N,work,&B_lwork,rwork,B_iwork,B_ifail,&B_ierr));
#else
                PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","V","L",&B_N,St,&B_N,S,&B_N,&bb[0],&bb[1],&B_IL,&B_IU,&eps,&B_neigs2,eigs+B_neigs,eigv+B_neigs*B_N,&B_N,work,&B_lwork,B_iwork,B_ifail,&B_ierr));
#endif
                PetscCall(PetscLogFlops((4.0*subset_size*subset_size*subset_size)/3.0));
                B_neigs += B_neigs2;
              }
              break;
            case 5: /* same as before: first compute all eigenvalues, then filter */
#if defined(PETSC_USE_COMPLEX)
              PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","A","L",&B_N,St,&B_N,S,&B_N,&bb[0],&bb[1],&B_IL,&B_IU,&eps,&B_neigs,eigs,eigv,&B_N,work,&B_lwork,rwork,B_iwork,B_ifail,&B_ierr));
#else
              PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","A","L",&B_N,St,&B_N,S,&B_N,&bb[0],&bb[1],&B_IL,&B_IU,&eps,&B_neigs,eigs,eigv,&B_N,work,&B_lwork,B_iwork,B_ifail,&B_ierr));
#endif
              PetscCall(PetscLogFlops((4.0*subset_size*subset_size*subset_size)/3.0));
              {
                PetscInt e,k,ne;
                for (e=0,ne=0;e<B_neigs;e++) {
                  if (eigs[e] < lthresh || eigs[e] > uthresh) {
                    for (k=0;k<B_N;k++) S[ne*B_N+k] = eigv[e*B_N+k];
                    eigs[ne] = eigs[e];
                    ne++;
                  }
                }
                PetscCall(PetscArraycpy(eigv,S,B_N*ne));
                B_neigs = ne;
              }
              break;
            default:
              SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Unknown recipe %" PetscInt_FMT,recipe);
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
          PetscCall(PetscLogFlops((4.0*subset_size*subset_size*subset_size)/3.0));
        } else { /* same_data is true, so just get the adaptive functional requested by the user */
          PetscInt k;
          PetscCheck(sub_schurs->change_primal_sub,PETSC_COMM_SELF,PETSC_ERR_PLIB,"This should not happen");
          PetscCall(ISGetLocalSize(sub_schurs->change_primal_sub[i],&nmax));
          PetscCall(PetscBLASIntCast(nmax,&B_neigs));
          nmin = nmax;
          PetscCall(PetscArrayzero(eigv,subset_size*nmax));
          for (k=0;k<nmax;k++) {
            eigs[k] = 1./PETSC_SMALL;
            eigv[k*(subset_size+1)] = 1.0;
          }
        }
        PetscCall(PetscFPTrapPop());
        if (B_ierr) {
          PetscCheck(B_ierr >= 0,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in SYGVX Lapack routine: illegal value for argument %" PetscBLASInt_FMT,-B_ierr);
          PetscCheck(B_ierr > B_N,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in SYGVX Lapack routine: %" PetscBLASInt_FMT " eigenvalues failed to converge",B_ierr);
          SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in SYGVX Lapack routine: leading minor of order %" PetscBLASInt_FMT " is not positive definite",B_ierr-B_N-1);
        }

        if (B_neigs > nmax) {
          if (pcbddc->dbg_flag) {
            PetscCall(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"   found %" PetscBLASInt_FMT " eigs, more than maximum required %" PetscInt_FMT ".\n",B_neigs,nmax));
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
            PetscCall(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"   found %" PetscBLASInt_FMT " eigs, less than minimum required %" PetscInt_FMT ". Asking for %" PetscBLASInt_FMT " to %" PetscBLASInt_FMT " incl (fortran like)\n",B_neigs,nmin,B_IL,B_IU));
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
            PetscCall(PetscArraycpy(S,Sarray+cumarray,subset_size*subset_size));
            PetscCall(PetscArraycpy(St,Starray+cumarray,subset_size*subset_size));
          }
          PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
#if defined(PETSC_USE_COMPLEX)
          PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","I","L",&B_N,St,&B_N,S,&B_N,&lower,&upper,&B_IL,&B_IU,&eps,&B_neigs2,eigs+B_neigs,eigv+B_neigs*subset_size,&B_N,work,&B_lwork,rwork,B_iwork,B_ifail,&B_ierr));
#else
          PetscStackCallBLAS("LAPACKsygvx",LAPACKsygvx_(&B_itype,"V","I","L",&B_N,St,&B_N,S,&B_N,&lower,&upper,&B_IL,&B_IU,&eps,&B_neigs2,eigs+B_neigs,eigv+B_neigs*subset_size,&B_N,work,&B_lwork,B_iwork,B_ifail,&B_ierr));
#endif
          PetscCall(PetscLogFlops((4.0*subset_size*subset_size*subset_size)/3.0));
          PetscCall(PetscFPTrapPop());
          B_neigs += B_neigs2;
        }
        if (B_ierr) {
          PetscCheck(B_ierr >= 0,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in SYGVX Lapack routine: illegal value for argument %" PetscBLASInt_FMT,-B_ierr);
          PetscCheck(B_ierr > B_N,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in SYGVX Lapack routine: %" PetscBLASInt_FMT " eigenvalues failed to converge",B_ierr);
          SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in SYGVX Lapack routine: leading minor of order %" PetscBLASInt_FMT " is not positive definite",B_ierr-B_N-1);
        }
        if (pcbddc->dbg_flag) {
          PetscCall(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"   -> Got %" PetscBLASInt_FMT " eigs\n",B_neigs));
          for (j=0;j<B_neigs;j++) {
            if (eigs[j] == 0.0) {
              PetscCall(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"     Inf\n"));
            } else {
              if (pcbddc->use_deluxe_scaling) {
                PetscCall(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"     %1.6e\n",(double)eigs[j+eigs_start]));
              } else {
                PetscCall(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"     %1.6e\n",(double)(1./eigs[j+eigs_start])));
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
          PetscCall(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"   -> Eigenvector (old basis) %" PetscInt_FMT "/%" PetscBLASInt_FMT " (%" PetscBLASInt_FMT ")\n",ii,B_neigs,B_N));
          for (j=0;j<B_N;j++) {
#if defined(PETSC_USE_COMPLEX)
            PetscReal r = PetscRealPart(eigv[(ii+eigs_start)*subset_size+j]);
            PetscReal c = PetscImaginaryPart(eigv[(ii+eigs_start)*subset_size+j]);
            PetscCall(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"       %1.4e + %1.4e i\n",(double)r,(double)c));
#else
            PetscCall(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"       %1.4e\n",(double)(eigv[(ii+eigs_start)*subset_size+j])));
#endif
          }
        }
      }
      PetscCall(KSPGetOperators(sub_schurs->change[i],&change,NULL));
      PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,subset_size,B_neigs,eigv+eigs_start*subset_size,&phit));
      PetscCall(MatMatMult(change,phit,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&phi));
      PetscCall(MatCopy(phi,phit,SAME_NONZERO_PATTERN));
      PetscCall(MatDestroy(&phit));
      PetscCall(MatDestroy(&phi));
    }
    maxneigs = PetscMax(B_neigs,maxneigs);
    pcbddc->adaptive_constraints_n[i+nv] = B_neigs;
    if (B_neigs) {
      PetscCall(PetscArraycpy(pcbddc->adaptive_constraints_data+pcbddc->adaptive_constraints_data_ptr[cum],eigv+eigs_start*subset_size,B_neigs*subset_size));

      if (pcbddc->dbg_flag > 1) {
        PetscInt ii;
        for (ii=0;ii<B_neigs;ii++) {
          PetscCall(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"   -> Eigenvector %" PetscInt_FMT "/%" PetscBLASInt_FMT " (%" PetscBLASInt_FMT ")\n",ii,B_neigs,B_N));
          for (j=0;j<B_N;j++) {
#if defined(PETSC_USE_COMPLEX)
            PetscReal r = PetscRealPart(pcbddc->adaptive_constraints_data[ii*subset_size+j+pcbddc->adaptive_constraints_data_ptr[cum]]);
            PetscReal c = PetscImaginaryPart(pcbddc->adaptive_constraints_data[ii*subset_size+j+pcbddc->adaptive_constraints_data_ptr[cum]]);
            PetscCall(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"       %1.4e + %1.4e i\n",(double)r,(double)c));
#else
            PetscCall(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"       %1.4e\n",(double)PetscRealPart(pcbddc->adaptive_constraints_data[ii*subset_size+j+pcbddc->adaptive_constraints_data_ptr[cum]])));
#endif
          }
        }
      }
      PetscCall(PetscArraycpy(pcbddc->adaptive_constraints_idxs+pcbddc->adaptive_constraints_idxs_ptr[cum],idxs,subset_size));
      pcbddc->adaptive_constraints_idxs_ptr[cum+1] = pcbddc->adaptive_constraints_idxs_ptr[cum] + subset_size;
      pcbddc->adaptive_constraints_data_ptr[cum+1] = pcbddc->adaptive_constraints_data_ptr[cum] + subset_size*B_neigs;
      cum++;
    }
    PetscCall(ISRestoreIndices(sub_schurs->is_subs[i],&idxs));
    /* shift for next computation */
    cumarray += subset_size*subset_size;
  }
  if (pcbddc->dbg_flag) {
    PetscCall(PetscViewerFlush(pcbddc->dbg_viewer));
  }

  if (mss) {
    PetscCall(MatSeqAIJRestoreArray(sub_schurs->sum_S_Ej_inv_all,&Sarray));
    PetscCall(MatSeqAIJRestoreArray(sub_schurs->sum_S_Ej_tilda_all,&Starray));
    /* destroy matrices (junk) */
    PetscCall(MatDestroy(&sub_schurs->sum_S_Ej_inv_all));
    PetscCall(MatDestroy(&sub_schurs->sum_S_Ej_tilda_all));
  }
  if (allocated_S_St) {
    PetscCall(PetscFree2(S,St));
  }
  PetscCall(PetscFree5(eigv,eigs,work,B_iwork,B_ifail));
#if defined(PETSC_USE_COMPLEX)
  PetscCall(PetscFree(rwork));
#endif
  if (pcbddc->dbg_flag) {
    PetscInt maxneigs_r;
    PetscCall(MPIU_Allreduce(&maxneigs,&maxneigs_r,1,MPIU_INT,MPI_MAX,PetscObjectComm((PetscObject)pc)));
    PetscCall(PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"Maximum number of constraints per cc %" PetscInt_FMT "\n",maxneigs_r));
  }
  PetscCall(PetscLogEventEnd(PC_BDDC_AdaptiveSetUp[pcbddc->current_level],pc,0,0,0));
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCSetUpSolvers(PC pc)
{
  PetscScalar    *coarse_submat_vals;

  PetscFunctionBegin;
  /* Setup local scatters R_to_B and (optionally) R_to_D */
  /* PCBDDCSetUpLocalWorkVectors should be called first! */
  PetscCall(PCBDDCSetUpLocalScatters(pc));

  /* Setup local neumann solver ksp_R */
  /* PCBDDCSetUpLocalScatters should be called first! */
  PetscCall(PCBDDCSetUpLocalSolvers(pc,PETSC_FALSE,PETSC_TRUE));

  /*
     Setup local correction and local part of coarse basis.
     Gives back the dense local part of the coarse matrix in column major ordering
  */
  PetscCall(PCBDDCSetUpCorrection(pc,&coarse_submat_vals));

  /* Compute total number of coarse nodes and setup coarse solver */
  PetscCall(PCBDDCSetUpCoarseSolver(pc,coarse_submat_vals));

  /* free */
  PetscCall(PetscFree(coarse_submat_vals));
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCResetCustomization(PC pc)
{
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;

  PetscFunctionBegin;
  PetscCall(ISDestroy(&pcbddc->user_primal_vertices));
  PetscCall(ISDestroy(&pcbddc->user_primal_vertices_local));
  PetscCall(ISDestroy(&pcbddc->NeumannBoundaries));
  PetscCall(ISDestroy(&pcbddc->NeumannBoundariesLocal));
  PetscCall(ISDestroy(&pcbddc->DirichletBoundaries));
  PetscCall(MatNullSpaceDestroy(&pcbddc->onearnullspace));
  PetscCall(PetscFree(pcbddc->onearnullvecs_state));
  PetscCall(ISDestroy(&pcbddc->DirichletBoundariesLocal));
  PetscCall(PCBDDCSetDofsSplitting(pc,0,NULL));
  PetscCall(PCBDDCSetDofsSplittingLocal(pc,0,NULL));
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCResetTopography(PC pc)
{
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;
  PetscInt       i;

  PetscFunctionBegin;
  PetscCall(MatDestroy(&pcbddc->nedcG));
  PetscCall(ISDestroy(&pcbddc->nedclocal));
  PetscCall(MatDestroy(&pcbddc->discretegradient));
  PetscCall(MatDestroy(&pcbddc->user_ChangeOfBasisMatrix));
  PetscCall(MatDestroy(&pcbddc->ChangeOfBasisMatrix));
  PetscCall(MatDestroy(&pcbddc->switch_static_change));
  PetscCall(VecDestroy(&pcbddc->work_change));
  PetscCall(MatDestroy(&pcbddc->ConstraintMatrix));
  PetscCall(MatDestroy(&pcbddc->divudotp));
  PetscCall(ISDestroy(&pcbddc->divudotp_vl2l));
  PetscCall(PCBDDCGraphDestroy(&pcbddc->mat_graph));
  for (i=0;i<pcbddc->n_local_subs;i++) {
    PetscCall(ISDestroy(&pcbddc->local_subs[i]));
  }
  pcbddc->n_local_subs = 0;
  PetscCall(PetscFree(pcbddc->local_subs));
  PetscCall(PCBDDCSubSchursDestroy(&pcbddc->sub_schurs));
  pcbddc->graphanalyzed        = PETSC_FALSE;
  pcbddc->recompute_topography = PETSC_TRUE;
  pcbddc->corner_selected      = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCResetSolvers(PC pc)
{
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;

  PetscFunctionBegin;
  PetscCall(VecDestroy(&pcbddc->coarse_vec));
  if (pcbddc->coarse_phi_B) {
    PetscScalar *array;
    PetscCall(MatDenseGetArray(pcbddc->coarse_phi_B,&array));
    PetscCall(PetscFree(array));
  }
  PetscCall(MatDestroy(&pcbddc->coarse_phi_B));
  PetscCall(MatDestroy(&pcbddc->coarse_phi_D));
  PetscCall(MatDestroy(&pcbddc->coarse_psi_B));
  PetscCall(MatDestroy(&pcbddc->coarse_psi_D));
  PetscCall(VecDestroy(&pcbddc->vec1_P));
  PetscCall(VecDestroy(&pcbddc->vec1_C));
  PetscCall(MatDestroy(&pcbddc->local_auxmat2));
  PetscCall(MatDestroy(&pcbddc->local_auxmat1));
  PetscCall(VecDestroy(&pcbddc->vec1_R));
  PetscCall(VecDestroy(&pcbddc->vec2_R));
  PetscCall(ISDestroy(&pcbddc->is_R_local));
  PetscCall(VecScatterDestroy(&pcbddc->R_to_B));
  PetscCall(VecScatterDestroy(&pcbddc->R_to_D));
  PetscCall(VecScatterDestroy(&pcbddc->coarse_loc_to_glob));
  PetscCall(KSPReset(pcbddc->ksp_D));
  PetscCall(KSPReset(pcbddc->ksp_R));
  PetscCall(KSPReset(pcbddc->coarse_ksp));
  PetscCall(MatDestroy(&pcbddc->local_mat));
  PetscCall(PetscFree(pcbddc->primal_indices_local_idxs));
  PetscCall(PetscFree2(pcbddc->local_primal_ref_node,pcbddc->local_primal_ref_mult));
  PetscCall(PetscFree(pcbddc->global_primal_indices));
  PetscCall(ISDestroy(&pcbddc->coarse_subassembling));
  PetscCall(MatDestroy(&pcbddc->benign_change));
  PetscCall(VecDestroy(&pcbddc->benign_vec));
  PetscCall(PCBDDCBenignShellMat(pc,PETSC_TRUE));
  PetscCall(MatDestroy(&pcbddc->benign_B0));
  PetscCall(PetscSFDestroy(&pcbddc->benign_sf));
  if (pcbddc->benign_zerodiag_subs) {
    PetscInt i;
    for (i=0;i<pcbddc->benign_n;i++) {
      PetscCall(ISDestroy(&pcbddc->benign_zerodiag_subs[i]));
    }
    PetscCall(PetscFree(pcbddc->benign_zerodiag_subs));
  }
  PetscCall(PetscFree3(pcbddc->benign_p0_lidx,pcbddc->benign_p0_gidx,pcbddc->benign_p0));
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
  PetscCall(VecGetType(pcis->vec1_N,&impVecType));
  /* local work vectors (try to avoid unneeded work)*/
  /* R nodes */
  old_size = -1;
  if (pcbddc->vec1_R) {
    PetscCall(VecGetSize(pcbddc->vec1_R,&old_size));
  }
  if (n_R != old_size) {
    PetscCall(VecDestroy(&pcbddc->vec1_R));
    PetscCall(VecDestroy(&pcbddc->vec2_R));
    PetscCall(VecCreate(PetscObjectComm((PetscObject)pcis->vec1_N),&pcbddc->vec1_R));
    PetscCall(VecSetSizes(pcbddc->vec1_R,PETSC_DECIDE,n_R));
    PetscCall(VecSetType(pcbddc->vec1_R,impVecType));
    PetscCall(VecDuplicate(pcbddc->vec1_R,&pcbddc->vec2_R));
  }
  /* local primal dofs */
  old_size = -1;
  if (pcbddc->vec1_P) {
    PetscCall(VecGetSize(pcbddc->vec1_P,&old_size));
  }
  if (pcbddc->local_primal_size != old_size) {
    PetscCall(VecDestroy(&pcbddc->vec1_P));
    PetscCall(VecCreate(PetscObjectComm((PetscObject)pcis->vec1_N),&pcbddc->vec1_P));
    PetscCall(VecSetSizes(pcbddc->vec1_P,PETSC_DECIDE,pcbddc->local_primal_size));
    PetscCall(VecSetType(pcbddc->vec1_P,impVecType));
  }
  /* local explicit constraints */
  old_size = -1;
  if (pcbddc->vec1_C) {
    PetscCall(VecGetSize(pcbddc->vec1_C,&old_size));
  }
  if (n_constraints && n_constraints != old_size) {
    PetscCall(VecDestroy(&pcbddc->vec1_C));
    PetscCall(VecCreate(PetscObjectComm((PetscObject)pcis->vec1_N),&pcbddc->vec1_C));
    PetscCall(VecSetSizes(pcbddc->vec1_C,PETSC_DECIDE,n_constraints));
    PetscCall(VecSetType(pcbddc->vec1_C,impVecType));
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
  PetscCheck(pcbddc->symmetric_primal || !pcbddc->benign_n,PETSC_COMM_SELF,PETSC_ERR_SUP,"Non-symmetric primal basis computation with benign trick not yet implemented");
  PetscCall(PetscLogEventBegin(PC_BDDC_CorrectionSetUp[pcbddc->current_level],pc,0,0,0));

  /* Set Non-overlapping dimensions */
  n_vertices = pcbddc->n_vertices;
  n_constraints = pcbddc->local_primal_size - pcbddc->benign_n - n_vertices;
  n_B = pcis->n_B;
  n_D = pcis->n - n_B;
  n_R = pcis->n - n_vertices;

  /* vertices in boundary numbering */
  PetscCall(PetscMalloc1(n_vertices,&idx_V_B));
  PetscCall(ISGlobalToLocalMappingApply(pcis->BtoNmap,IS_GTOLM_DROP,n_vertices,pcbddc->local_primal_ref_node,&i,idx_V_B));
  PetscCheck(i == n_vertices,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error in boundary numbering for BDDC vertices! %" PetscInt_FMT " != %" PetscInt_FMT,n_vertices,i);

  /* Subdomain contribution (Non-overlapping) to coarse matrix  */
  PetscCall(PetscCalloc1(pcbddc->local_primal_size*pcbddc->local_primal_size,&coarse_submat_vals));
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,n_vertices,n_vertices,coarse_submat_vals,&S_VV));
  PetscCall(MatDenseSetLDA(S_VV,pcbddc->local_primal_size));
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,n_constraints,n_vertices,coarse_submat_vals+n_vertices,&S_CV));
  PetscCall(MatDenseSetLDA(S_CV,pcbddc->local_primal_size));
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,n_vertices,n_constraints,coarse_submat_vals+pcbddc->local_primal_size*n_vertices,&S_VC));
  PetscCall(MatDenseSetLDA(S_VC,pcbddc->local_primal_size));
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,n_constraints,n_constraints,coarse_submat_vals+(pcbddc->local_primal_size+1)*n_vertices,&S_CC));
  PetscCall(MatDenseSetLDA(S_CC,pcbddc->local_primal_size));

  /* determine if can use MatSolve routines instead of calling KSPSolve on ksp_R */
  PetscCall(KSPGetPC(pcbddc->ksp_R,&pc_R));
  PetscCall(PCSetUp(pc_R));
  PetscCall(PetscObjectTypeCompare((PetscObject)pc_R,PCLU,&isLU));
  PetscCall(PetscObjectTypeCompare((PetscObject)pc_R,PCCHOLESKY,&isCHOL));
  lda_rhs = n_R;
  need_benign_correction = PETSC_FALSE;
  if (isLU || isCHOL) {
    PetscCall(PCFactorGetMatrix(pc_R,&F));
  } else if (sub_schurs && sub_schurs->reuse_solver) {
    PCBDDCReuseSolvers reuse_solver = sub_schurs->reuse_solver;
    MatFactorType      type;

    F = reuse_solver->F;
    PetscCall(MatGetFactorType(F,&type));
    if (type == MAT_FACTOR_CHOLESKY) isCHOL = PETSC_TRUE;
    if (type == MAT_FACTOR_LU) isLU = PETSC_TRUE;
    PetscCall(MatGetSize(F,&lda_rhs,NULL));
    need_benign_correction = (PetscBool)(!!reuse_solver->benign_n);
  } else F = NULL;

  /* determine if we can use a sparse right-hand side */
  sparserhs = PETSC_FALSE;
  if (F) {
    MatSolverType solver;

    PetscCall(MatFactorGetSolverType(F,&solver));
    PetscCall(PetscStrcmp(solver,MATSOLVERMUMPS,&sparserhs));
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
  PetscCall(PetscMalloc1(n,&work));

  /* create dummy vector to modify rhs and sol of MatMatSolve (work array will never be used) */
  dummy_vec = NULL;
  if (need_benign_correction && lda_rhs != n_R && F) {
    PetscCall(VecCreate(PetscObjectComm((PetscObject)pcis->vec1_N),&dummy_vec));
    PetscCall(VecSetSizes(dummy_vec,lda_rhs,PETSC_DECIDE));
    PetscCall(VecSetType(dummy_vec,((PetscObject)pcis->vec1_N)->type_name));
  }

  PetscCall(MatDestroy(&pcbddc->local_auxmat1));
  PetscCall(MatDestroy(&pcbddc->local_auxmat2));

  /* Precompute stuffs needed for preprocessing and application of BDDC*/
  if (n_constraints) {
    Mat         M3,C_B;
    IS          is_aux;

    /* Extract constraints on R nodes: C_{CR}  */
    PetscCall(ISCreateStride(PETSC_COMM_SELF,n_constraints,n_vertices,1,&is_aux));
    PetscCall(MatCreateSubMatrix(pcbddc->ConstraintMatrix,is_aux,pcbddc->is_R_local,MAT_INITIAL_MATRIX,&C_CR));
    PetscCall(MatCreateSubMatrix(pcbddc->ConstraintMatrix,is_aux,pcis->is_B_local,MAT_INITIAL_MATRIX,&C_B));

    /* Assemble         local_auxmat2_R =        (- A_{RR}^{-1} C^T_{CR}) needed by BDDC setup */
    /* Assemble pcbddc->local_auxmat2   = R_to_B (- A_{RR}^{-1} C^T_{CR}) needed by BDDC application */
    if (!sparserhs) {
      PetscCall(PetscArrayzero(work,lda_rhs*n_constraints));
      for (i=0;i<n_constraints;i++) {
        const PetscScalar *row_cmat_values;
        const PetscInt    *row_cmat_indices;
        PetscInt          size_of_constraint,j;

        PetscCall(MatGetRow(C_CR,i,&size_of_constraint,&row_cmat_indices,&row_cmat_values));
        for (j=0;j<size_of_constraint;j++) {
          work[row_cmat_indices[j]+i*lda_rhs] = -row_cmat_values[j];
        }
        PetscCall(MatRestoreRow(C_CR,i,&size_of_constraint,&row_cmat_indices,&row_cmat_values));
      }
      PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,lda_rhs,n_constraints,work,&Brhs));
    } else {
      Mat tC_CR;

      PetscCall(MatScale(C_CR,-1.0));
      if (lda_rhs != n_R) {
        PetscScalar *aa;
        PetscInt    r,*ii,*jj;
        PetscBool   done;

        PetscCall(MatGetRowIJ(C_CR,0,PETSC_FALSE,PETSC_FALSE,&r,(const PetscInt**)&ii,(const PetscInt**)&jj,&done));
        PetscCheck(done,PETSC_COMM_SELF,PETSC_ERR_PLIB,"GetRowIJ failed");
        PetscCall(MatSeqAIJGetArray(C_CR,&aa));
        PetscCall(MatCreateSeqAIJWithArrays(PETSC_COMM_SELF,n_constraints,lda_rhs,ii,jj,aa,&tC_CR));
        PetscCall(MatRestoreRowIJ(C_CR,0,PETSC_FALSE,PETSC_FALSE,&r,(const PetscInt**)&ii,(const PetscInt**)&jj,&done));
        PetscCheck(done,PETSC_COMM_SELF,PETSC_ERR_PLIB,"RestoreRowIJ failed");
      } else {
        PetscCall(PetscObjectReference((PetscObject)C_CR));
        tC_CR = C_CR;
      }
      PetscCall(MatCreateTranspose(tC_CR,&Brhs));
      PetscCall(MatDestroy(&tC_CR));
    }
    PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,lda_rhs,n_constraints,NULL,&local_auxmat2_R));
    if (F) {
      if (need_benign_correction) {
        PCBDDCReuseSolvers reuse_solver = sub_schurs->reuse_solver;

        /* rhs is already zero on interior dofs, no need to change the rhs */
        PetscCall(PetscArrayzero(reuse_solver->benign_save_vals,pcbddc->benign_n));
      }
      PetscCall(MatMatSolve(F,Brhs,local_auxmat2_R));
      if (need_benign_correction) {
        PetscScalar        *marr;
        PCBDDCReuseSolvers reuse_solver = sub_schurs->reuse_solver;

        PetscCall(MatDenseGetArray(local_auxmat2_R,&marr));
        if (lda_rhs != n_R) {
          for (i=0;i<n_constraints;i++) {
            PetscCall(VecPlaceArray(dummy_vec,marr+i*lda_rhs));
            PetscCall(PCBDDCReuseSolversBenignAdapt(reuse_solver,dummy_vec,NULL,PETSC_TRUE,PETSC_TRUE));
            PetscCall(VecResetArray(dummy_vec));
          }
        } else {
          for (i=0;i<n_constraints;i++) {
            PetscCall(VecPlaceArray(pcbddc->vec1_R,marr+i*lda_rhs));
            PetscCall(PCBDDCReuseSolversBenignAdapt(reuse_solver,pcbddc->vec1_R,NULL,PETSC_TRUE,PETSC_TRUE));
            PetscCall(VecResetArray(pcbddc->vec1_R));
          }
        }
        PetscCall(MatDenseRestoreArray(local_auxmat2_R,&marr));
      }
    } else {
      PetscScalar *marr;

      PetscCall(MatDenseGetArray(local_auxmat2_R,&marr));
      for (i=0;i<n_constraints;i++) {
        PetscCall(VecPlaceArray(pcbddc->vec1_R,work+i*lda_rhs));
        PetscCall(VecPlaceArray(pcbddc->vec2_R,marr+i*lda_rhs));
        PetscCall(KSPSolve(pcbddc->ksp_R,pcbddc->vec1_R,pcbddc->vec2_R));
        PetscCall(KSPCheckSolve(pcbddc->ksp_R,pc,pcbddc->vec2_R));
        PetscCall(VecResetArray(pcbddc->vec1_R));
        PetscCall(VecResetArray(pcbddc->vec2_R));
      }
      PetscCall(MatDenseRestoreArray(local_auxmat2_R,&marr));
    }
    if (sparserhs) {
      PetscCall(MatScale(C_CR,-1.0));
    }
    PetscCall(MatDestroy(&Brhs));
    if (!pcbddc->switch_static) {
      PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,n_B,n_constraints,NULL,&pcbddc->local_auxmat2));
      for (i=0;i<n_constraints;i++) {
        Vec r, b;
        PetscCall(MatDenseGetColumnVecRead(local_auxmat2_R,i,&r));
        PetscCall(MatDenseGetColumnVec(pcbddc->local_auxmat2,i,&b));
        PetscCall(VecScatterBegin(pcbddc->R_to_B,r,b,INSERT_VALUES,SCATTER_FORWARD));
        PetscCall(VecScatterEnd(pcbddc->R_to_B,r,b,INSERT_VALUES,SCATTER_FORWARD));
        PetscCall(MatDenseRestoreColumnVec(pcbddc->local_auxmat2,i,&b));
        PetscCall(MatDenseRestoreColumnVecRead(local_auxmat2_R,i,&r));
      }
      PetscCall(MatMatMult(C_B,pcbddc->local_auxmat2,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&M3));
    } else {
      if (lda_rhs != n_R) {
        IS dummy;

        PetscCall(ISCreateStride(PETSC_COMM_SELF,n_R,0,1,&dummy));
        PetscCall(MatCreateSubMatrix(local_auxmat2_R,dummy,NULL,MAT_INITIAL_MATRIX,&pcbddc->local_auxmat2));
        PetscCall(ISDestroy(&dummy));
      } else {
        PetscCall(PetscObjectReference((PetscObject)local_auxmat2_R));
        pcbddc->local_auxmat2 = local_auxmat2_R;
      }
      PetscCall(MatMatMult(C_CR,pcbddc->local_auxmat2,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&M3));
    }
    PetscCall(ISDestroy(&is_aux));
    /* Assemble explicitly S_CC = ( C_{CR} A_{RR}^{-1} C^T_{CR})^{-1}  */
    PetscCall(MatScale(M3,m_one));
    if (isCHOL) {
      PetscCall(MatCholeskyFactor(M3,NULL,NULL));
    } else {
      PetscCall(MatLUFactor(M3,NULL,NULL,NULL));
    }
    PetscCall(MatSeqDenseInvertFactors_Private(M3));
    /* Assemble local_auxmat1 = S_CC*C_{CB} needed by BDDC application in KSP and in preproc */
    PetscCall(MatMatMult(M3,C_B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&pcbddc->local_auxmat1));
    PetscCall(MatDestroy(&C_B));
    PetscCall(MatCopy(M3,S_CC,SAME_NONZERO_PATTERN)); /* S_CC can have a different LDA, MatMatSolve doesn't support it */
    PetscCall(MatDestroy(&M3));
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

      PetscCall(ISDuplicate(pcbddc->is_R_local,&tis));
      PetscCall(ISSort(tis));
      PetscCall(ISComplement(tis,0,pcis->n,&is_aux));
      PetscCall(ISDestroy(&tis));
    } else {
      PetscCall(ISComplement(pcbddc->is_R_local,0,pcis->n,&is_aux));
    }
#if defined(PETSC_HAVE_VIENNACL) || defined(PETSC_HAVE_CUDA)
    oldpin = pcbddc->local_mat->boundtocpu;
#endif
    PetscCall(MatBindToCPU(pcbddc->local_mat,PETSC_TRUE));
    PetscCall(MatCreateSubMatrix(pcbddc->local_mat,pcbddc->is_R_local,is_aux,MAT_INITIAL_MATRIX,&A_RV));
    PetscCall(MatCreateSubMatrix(pcbddc->local_mat,is_aux,pcbddc->is_R_local,MAT_INITIAL_MATRIX,&A_VR));
    PetscCall(PetscObjectBaseTypeCompare((PetscObject)A_VR,MATSEQAIJ,&isaij));
    if (!isaij) { /* TODO REMOVE: MatMatMult(A_VR,A_RRmA_RV) below may raise an error */
      PetscCall(MatConvert(A_VR,MATSEQAIJ,MAT_INPLACE_MATRIX,&A_VR));
    }
    PetscCall(MatCreateSubMatrix(pcbddc->local_mat,is_aux,is_aux,MAT_INITIAL_MATRIX,&A_VV));
#if defined(PETSC_HAVE_VIENNACL) || defined(PETSC_HAVE_CUDA)
    PetscCall(MatBindToCPU(pcbddc->local_mat,oldpin));
#endif
    PetscCall(ISDestroy(&is_aux));
  }

  /* Matrix of coarse basis functions (local) */
  if (pcbddc->coarse_phi_B) {
    PetscInt on_B,on_primal,on_D=n_D;
    if (pcbddc->coarse_phi_D) {
      PetscCall(MatGetSize(pcbddc->coarse_phi_D,&on_D,NULL));
    }
    PetscCall(MatGetSize(pcbddc->coarse_phi_B,&on_B,&on_primal));
    if (on_B != n_B || on_primal != pcbddc->local_primal_size || on_D != n_D) {
      PetscScalar *marray;

      PetscCall(MatDenseGetArray(pcbddc->coarse_phi_B,&marray));
      PetscCall(PetscFree(marray));
      PetscCall(MatDestroy(&pcbddc->coarse_phi_B));
      PetscCall(MatDestroy(&pcbddc->coarse_psi_B));
      PetscCall(MatDestroy(&pcbddc->coarse_phi_D));
      PetscCall(MatDestroy(&pcbddc->coarse_psi_D));
    }
  }

  if (!pcbddc->coarse_phi_B) {
    PetscScalar *marr;

    /* memory size */
    n = n_B*pcbddc->local_primal_size;
    if (pcbddc->switch_static || pcbddc->dbg_flag) n += n_D*pcbddc->local_primal_size;
    if (!pcbddc->symmetric_primal) n *= 2;
    PetscCall(PetscCalloc1(n,&marr));
    PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,n_B,pcbddc->local_primal_size,marr,&pcbddc->coarse_phi_B));
    marr += n_B*pcbddc->local_primal_size;
    if (pcbddc->switch_static || pcbddc->dbg_flag) {
      PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,n_D,pcbddc->local_primal_size,marr,&pcbddc->coarse_phi_D));
      marr += n_D*pcbddc->local_primal_size;
    }
    if (!pcbddc->symmetric_primal) {
      PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,n_B,pcbddc->local_primal_size,marr,&pcbddc->coarse_psi_B));
      marr += n_B*pcbddc->local_primal_size;
      if (pcbddc->switch_static || pcbddc->dbg_flag) {
        PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,n_D,pcbddc->local_primal_size,marr,&pcbddc->coarse_psi_D));
      }
    } else {
      PetscCall(PetscObjectReference((PetscObject)pcbddc->coarse_phi_B));
      pcbddc->coarse_psi_B = pcbddc->coarse_phi_B;
      if (pcbddc->switch_static || pcbddc->dbg_flag) {
        PetscCall(PetscObjectReference((PetscObject)pcbddc->coarse_phi_D));
        pcbddc->coarse_psi_D = pcbddc->coarse_phi_D;
      }
    }
  }

  /* We are now ready to evaluate coarse basis functions and subdomain contribution to coarse problem */
  p0_lidx_I = NULL;
  if (pcbddc->benign_n && (pcbddc->switch_static || pcbddc->dbg_flag)) {
    const PetscInt *idxs;

    PetscCall(ISGetIndices(pcis->is_I_local,&idxs));
    PetscCall(PetscMalloc1(pcbddc->benign_n,&p0_lidx_I));
    for (i=0;i<pcbddc->benign_n;i++) {
      PetscCall(PetscFindInt(pcbddc->benign_p0_lidx[i],pcis->n-pcis->n_B,idxs,&p0_lidx_I[i]));
    }
    PetscCall(ISRestoreIndices(pcis->is_I_local,&idxs));
  }

  /* vertices */
  if (n_vertices) {
    PetscBool restoreavr = PETSC_FALSE;

    PetscCall(MatConvert(A_VV,MATDENSE,MAT_INPLACE_MATRIX,&A_VV));

    if (n_R) {
      Mat               A_RRmA_RV,A_RV_bcorr=NULL,S_VVt; /* S_VVt with LDA=N */
      PetscBLASInt      B_N,B_one = 1;
      const PetscScalar *x;
      PetscScalar       *y;

      PetscCall(MatScale(A_RV,m_one));
      if (need_benign_correction) {
        ISLocalToGlobalMapping RtoN;
        IS                     is_p0;
        PetscInt               *idxs_p0,n;

        PetscCall(PetscMalloc1(pcbddc->benign_n,&idxs_p0));
        PetscCall(ISLocalToGlobalMappingCreateIS(pcbddc->is_R_local,&RtoN));
        PetscCall(ISGlobalToLocalMappingApply(RtoN,IS_GTOLM_DROP,pcbddc->benign_n,pcbddc->benign_p0_lidx,&n,idxs_p0));
        PetscCheck(n == pcbddc->benign_n,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error in R numbering for benign p0! %" PetscInt_FMT " != %" PetscInt_FMT,n,pcbddc->benign_n);
        PetscCall(ISLocalToGlobalMappingDestroy(&RtoN));
        PetscCall(ISCreateGeneral(PETSC_COMM_SELF,n,idxs_p0,PETSC_OWN_POINTER,&is_p0));
        PetscCall(MatCreateSubMatrix(A_RV,is_p0,NULL,MAT_INITIAL_MATRIX,&A_RV_bcorr));
        PetscCall(ISDestroy(&is_p0));
      }

      PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,lda_rhs,n_vertices,work,&A_RRmA_RV));
      if (!sparserhs || need_benign_correction) {
        if (lda_rhs == n_R) {
          PetscCall(MatConvert(A_RV,MATDENSE,MAT_INPLACE_MATRIX,&A_RV));
        } else {
          PetscScalar    *av,*array;
          const PetscInt *xadj,*adjncy;
          PetscInt       n;
          PetscBool      flg_row;

          array = work+lda_rhs*n_vertices;
          PetscCall(PetscArrayzero(array,lda_rhs*n_vertices));
          PetscCall(MatConvert(A_RV,MATSEQAIJ,MAT_INPLACE_MATRIX,&A_RV));
          PetscCall(MatGetRowIJ(A_RV,0,PETSC_FALSE,PETSC_FALSE,&n,&xadj,&adjncy,&flg_row));
          PetscCall(MatSeqAIJGetArray(A_RV,&av));
          for (i=0;i<n;i++) {
            PetscInt j;
            for (j=xadj[i];j<xadj[i+1];j++) array[lda_rhs*adjncy[j]+i] = av[j];
          }
          PetscCall(MatRestoreRowIJ(A_RV,0,PETSC_FALSE,PETSC_FALSE,&n,&xadj,&adjncy,&flg_row));
          PetscCall(MatDestroy(&A_RV));
          PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,lda_rhs,n_vertices,array,&A_RV));
        }
        if (need_benign_correction) {
          PCBDDCReuseSolvers reuse_solver = sub_schurs->reuse_solver;
          PetscScalar        *marr;

          PetscCall(MatDenseGetArray(A_RV,&marr));
          /* need \Phi^T A_RV = (I+L)A_RV, L given by

                 | 0 0  0 | (V)
             L = | 0 0 -1 | (P-p0)
                 | 0 0 -1 | (p0)

          */
          for (i=0;i<reuse_solver->benign_n;i++) {
            const PetscScalar *vals;
            const PetscInt    *idxs,*idxs_zero;
            PetscInt          n,j,nz;

            PetscCall(ISGetLocalSize(reuse_solver->benign_zerodiag_subs[i],&nz));
            PetscCall(ISGetIndices(reuse_solver->benign_zerodiag_subs[i],&idxs_zero));
            PetscCall(MatGetRow(A_RV_bcorr,i,&n,&idxs,&vals));
            for (j=0;j<n;j++) {
              PetscScalar val = vals[j];
              PetscInt    k,col = idxs[j];
              for (k=0;k<nz;k++) marr[idxs_zero[k]+lda_rhs*col] -= val;
            }
            PetscCall(MatRestoreRow(A_RV_bcorr,i,&n,&idxs,&vals));
            PetscCall(ISRestoreIndices(reuse_solver->benign_zerodiag_subs[i],&idxs_zero));
          }
          PetscCall(MatDenseRestoreArray(A_RV,&marr));
        }
        PetscCall(PetscObjectReference((PetscObject)A_RV));
        Brhs = A_RV;
      } else {
        Mat tA_RVT,A_RVT;

        if (!pcbddc->symmetric_primal) {
          /* A_RV already scaled by -1 */
          PetscCall(MatTranspose(A_RV,MAT_INITIAL_MATRIX,&A_RVT));
        } else {
          restoreavr = PETSC_TRUE;
          PetscCall(MatScale(A_VR,-1.0));
          PetscCall(PetscObjectReference((PetscObject)A_VR));
          A_RVT = A_VR;
        }
        if (lda_rhs != n_R) {
          PetscScalar *aa;
          PetscInt    r,*ii,*jj;
          PetscBool   done;

          PetscCall(MatGetRowIJ(A_RVT,0,PETSC_FALSE,PETSC_FALSE,&r,(const PetscInt**)&ii,(const PetscInt**)&jj,&done));
          PetscCheck(done,PETSC_COMM_SELF,PETSC_ERR_PLIB,"GetRowIJ failed");
          PetscCall(MatSeqAIJGetArray(A_RVT,&aa));
          PetscCall(MatCreateSeqAIJWithArrays(PETSC_COMM_SELF,n_vertices,lda_rhs,ii,jj,aa,&tA_RVT));
          PetscCall(MatRestoreRowIJ(A_RVT,0,PETSC_FALSE,PETSC_FALSE,&r,(const PetscInt**)&ii,(const PetscInt**)&jj,&done));
          PetscCheck(done,PETSC_COMM_SELF,PETSC_ERR_PLIB,"RestoreRowIJ failed");
        } else {
          PetscCall(PetscObjectReference((PetscObject)A_RVT));
          tA_RVT = A_RVT;
        }
        PetscCall(MatCreateTranspose(tA_RVT,&Brhs));
        PetscCall(MatDestroy(&tA_RVT));
        PetscCall(MatDestroy(&A_RVT));
      }
      if (F) {
        /* need to correct the rhs */
        if (need_benign_correction) {
          PCBDDCReuseSolvers reuse_solver = sub_schurs->reuse_solver;
          PetscScalar        *marr;

          PetscCall(MatDenseGetArray(Brhs,&marr));
          if (lda_rhs != n_R) {
            for (i=0;i<n_vertices;i++) {
              PetscCall(VecPlaceArray(dummy_vec,marr+i*lda_rhs));
              PetscCall(PCBDDCReuseSolversBenignAdapt(reuse_solver,dummy_vec,NULL,PETSC_FALSE,PETSC_TRUE));
              PetscCall(VecResetArray(dummy_vec));
            }
          } else {
            for (i=0;i<n_vertices;i++) {
              PetscCall(VecPlaceArray(pcbddc->vec1_R,marr+i*lda_rhs));
              PetscCall(PCBDDCReuseSolversBenignAdapt(reuse_solver,pcbddc->vec1_R,NULL,PETSC_FALSE,PETSC_TRUE));
              PetscCall(VecResetArray(pcbddc->vec1_R));
            }
          }
          PetscCall(MatDenseRestoreArray(Brhs,&marr));
        }
        PetscCall(MatMatSolve(F,Brhs,A_RRmA_RV));
        if (restoreavr) {
          PetscCall(MatScale(A_VR,-1.0));
        }
        /* need to correct the solution */
        if (need_benign_correction) {
          PCBDDCReuseSolvers reuse_solver = sub_schurs->reuse_solver;
          PetscScalar        *marr;

          PetscCall(MatDenseGetArray(A_RRmA_RV,&marr));
          if (lda_rhs != n_R) {
            for (i=0;i<n_vertices;i++) {
              PetscCall(VecPlaceArray(dummy_vec,marr+i*lda_rhs));
              PetscCall(PCBDDCReuseSolversBenignAdapt(reuse_solver,dummy_vec,NULL,PETSC_TRUE,PETSC_TRUE));
              PetscCall(VecResetArray(dummy_vec));
            }
          } else {
            for (i=0;i<n_vertices;i++) {
              PetscCall(VecPlaceArray(pcbddc->vec1_R,marr+i*lda_rhs));
              PetscCall(PCBDDCReuseSolversBenignAdapt(reuse_solver,pcbddc->vec1_R,NULL,PETSC_TRUE,PETSC_TRUE));
              PetscCall(VecResetArray(pcbddc->vec1_R));
            }
          }
          PetscCall(MatDenseRestoreArray(A_RRmA_RV,&marr));
        }
      } else {
        PetscCall(MatDenseGetArray(Brhs,&y));
        for (i=0;i<n_vertices;i++) {
          PetscCall(VecPlaceArray(pcbddc->vec1_R,y+i*lda_rhs));
          PetscCall(VecPlaceArray(pcbddc->vec2_R,work+i*lda_rhs));
          PetscCall(KSPSolve(pcbddc->ksp_R,pcbddc->vec1_R,pcbddc->vec2_R));
          PetscCall(KSPCheckSolve(pcbddc->ksp_R,pc,pcbddc->vec2_R));
          PetscCall(VecResetArray(pcbddc->vec1_R));
          PetscCall(VecResetArray(pcbddc->vec2_R));
        }
        PetscCall(MatDenseRestoreArray(Brhs,&y));
      }
      PetscCall(MatDestroy(&A_RV));
      PetscCall(MatDestroy(&Brhs));
      /* S_VV and S_CV */
      if (n_constraints) {
        Mat B;

        PetscCall(PetscArrayzero(work+lda_rhs*n_vertices,n_B*n_vertices));
        for (i=0;i<n_vertices;i++) {
          PetscCall(VecPlaceArray(pcbddc->vec1_R,work+i*lda_rhs));
          PetscCall(VecPlaceArray(pcis->vec1_B,work+lda_rhs*n_vertices+i*n_B));
          PetscCall(VecScatterBegin(pcbddc->R_to_B,pcbddc->vec1_R,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD));
          PetscCall(VecScatterEnd(pcbddc->R_to_B,pcbddc->vec1_R,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD));
          PetscCall(VecResetArray(pcis->vec1_B));
          PetscCall(VecResetArray(pcbddc->vec1_R));
        }
        PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,n_B,n_vertices,work+lda_rhs*n_vertices,&B));
        /* Reuse dense S_C = pcbddc->local_auxmat1 * B */
        PetscCall(MatProductCreateWithMat(pcbddc->local_auxmat1,B,NULL,S_CV));
        PetscCall(MatProductSetType(S_CV,MATPRODUCT_AB));
        PetscCall(MatProductSetFromOptions(S_CV));
        PetscCall(MatProductSymbolic(S_CV));
        PetscCall(MatProductNumeric(S_CV));
        PetscCall(MatProductClear(S_CV));

        PetscCall(MatDestroy(&B));
        PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,lda_rhs,n_vertices,work+lda_rhs*n_vertices,&B));
        /* Reuse B = local_auxmat2_R * S_CV */
        PetscCall(MatProductCreateWithMat(local_auxmat2_R,S_CV,NULL,B));
        PetscCall(MatProductSetType(B,MATPRODUCT_AB));
        PetscCall(MatProductSetFromOptions(B));
        PetscCall(MatProductSymbolic(B));
        PetscCall(MatProductNumeric(B));

        PetscCall(MatScale(S_CV,m_one));
        PetscCall(PetscBLASIntCast(lda_rhs*n_vertices,&B_N));
        PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&B_N,&one,work+lda_rhs*n_vertices,&B_one,work,&B_one));
        PetscCall(MatDestroy(&B));
      }
      if (lda_rhs != n_R) {
        PetscCall(MatDestroy(&A_RRmA_RV));
        PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,n_R,n_vertices,work,&A_RRmA_RV));
        PetscCall(MatDenseSetLDA(A_RRmA_RV,lda_rhs));
      }
      PetscCall(MatMatMult(A_VR,A_RRmA_RV,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&S_VVt));
      /* need A_VR * \Phi * A_RRmA_RV = A_VR * (I+L)^T * A_RRmA_RV, L given as before */
      if (need_benign_correction) {
        PCBDDCReuseSolvers reuse_solver = sub_schurs->reuse_solver;
        PetscScalar        *marr,*sums;

        PetscCall(PetscMalloc1(n_vertices,&sums));
        PetscCall(MatDenseGetArray(S_VVt,&marr));
        for (i=0;i<reuse_solver->benign_n;i++) {
          const PetscScalar *vals;
          const PetscInt    *idxs,*idxs_zero;
          PetscInt          n,j,nz;

          PetscCall(ISGetLocalSize(reuse_solver->benign_zerodiag_subs[i],&nz));
          PetscCall(ISGetIndices(reuse_solver->benign_zerodiag_subs[i],&idxs_zero));
          for (j=0;j<n_vertices;j++) {
            PetscInt k;
            sums[j] = 0.;
            for (k=0;k<nz;k++) sums[j] += work[idxs_zero[k]+j*lda_rhs];
          }
          PetscCall(MatGetRow(A_RV_bcorr,i,&n,&idxs,&vals));
          for (j=0;j<n;j++) {
            PetscScalar val = vals[j];
            PetscInt k;
            for (k=0;k<n_vertices;k++) {
              marr[idxs[j]+k*n_vertices] += val*sums[k];
            }
          }
          PetscCall(MatRestoreRow(A_RV_bcorr,i,&n,&idxs,&vals));
          PetscCall(ISRestoreIndices(reuse_solver->benign_zerodiag_subs[i],&idxs_zero));
        }
        PetscCall(PetscFree(sums));
        PetscCall(MatDenseRestoreArray(S_VVt,&marr));
        PetscCall(MatDestroy(&A_RV_bcorr));
      }
      PetscCall(MatDestroy(&A_RRmA_RV));
      PetscCall(PetscBLASIntCast(n_vertices*n_vertices,&B_N));
      PetscCall(MatDenseGetArrayRead(A_VV,&x));
      PetscCall(MatDenseGetArray(S_VVt,&y));
      PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&B_N,&one,x,&B_one,y,&B_one));
      PetscCall(MatDenseRestoreArrayRead(A_VV,&x));
      PetscCall(MatDenseRestoreArray(S_VVt,&y));
      PetscCall(MatCopy(S_VVt,S_VV,SAME_NONZERO_PATTERN));
      PetscCall(MatDestroy(&S_VVt));
    } else {
      PetscCall(MatCopy(A_VV,S_VV,SAME_NONZERO_PATTERN));
    }
    PetscCall(MatDestroy(&A_VV));

    /* coarse basis functions */
    for (i=0;i<n_vertices;i++) {
      Vec         v;
      PetscScalar one = 1.0,zero = 0.0;

      PetscCall(VecPlaceArray(pcbddc->vec1_R,work+lda_rhs*i));
      PetscCall(MatDenseGetColumnVec(pcbddc->coarse_phi_B,i,&v));
      PetscCall(VecScatterBegin(pcbddc->R_to_B,pcbddc->vec1_R,v,INSERT_VALUES,SCATTER_FORWARD));
      PetscCall(VecScatterEnd(pcbddc->R_to_B,pcbddc->vec1_R,v,INSERT_VALUES,SCATTER_FORWARD));
      if (PetscDefined(USE_DEBUG)) { /* The following VecSetValues() expects a sequential matrix */
        PetscMPIInt rank;
        PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)pcbddc->coarse_phi_B),&rank));
        PetscCheck(rank <= 1,PetscObjectComm((PetscObject)pcbddc->coarse_phi_B),PETSC_ERR_PLIB,"Expected a sequential dense matrix");
      }
      PetscCall(VecSetValues(v,1,&idx_V_B[i],&one,INSERT_VALUES));
      PetscCall(VecAssemblyBegin(v)); /* If v is on device, hope VecSetValues() eventually implemented by a host to device memcopy */
      PetscCall(VecAssemblyEnd(v));
      PetscCall(MatDenseRestoreColumnVec(pcbddc->coarse_phi_B,i,&v));

      if (pcbddc->switch_static || pcbddc->dbg_flag) {
        PetscInt j;

        PetscCall(MatDenseGetColumnVec(pcbddc->coarse_phi_D,i,&v));
        PetscCall(VecScatterBegin(pcbddc->R_to_D,pcbddc->vec1_R,v,INSERT_VALUES,SCATTER_FORWARD));
        PetscCall(VecScatterEnd(pcbddc->R_to_D,pcbddc->vec1_R,v,INSERT_VALUES,SCATTER_FORWARD));
        if (PetscDefined(USE_DEBUG)) { /* The following VecSetValues() expects a sequential matrix */
          PetscMPIInt rank;
          PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)pcbddc->coarse_phi_D),&rank));
          PetscCheck(rank <= 1,PetscObjectComm((PetscObject)pcbddc->coarse_phi_D),PETSC_ERR_PLIB,"Expected a sequential dense matrix");
        }
        for (j=0;j<pcbddc->benign_n;j++) PetscCall(VecSetValues(v,1,&p0_lidx_I[j],&zero,INSERT_VALUES));
        PetscCall(VecAssemblyBegin(v));
        PetscCall(VecAssemblyEnd(v));
        PetscCall(MatDenseRestoreColumnVec(pcbddc->coarse_phi_D,i,&v));
      }
      PetscCall(VecResetArray(pcbddc->vec1_R));
    }
    /* if n_R == 0 the object is not destroyed */
    PetscCall(MatDestroy(&A_RV));
  }
  PetscCall(VecDestroy(&dummy_vec));

  if (n_constraints) {
    Mat B;

    PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,lda_rhs,n_constraints,work,&B));
    PetscCall(MatScale(S_CC,m_one));
    PetscCall(MatProductCreateWithMat(local_auxmat2_R,S_CC,NULL,B));
    PetscCall(MatProductSetType(B,MATPRODUCT_AB));
    PetscCall(MatProductSetFromOptions(B));
    PetscCall(MatProductSymbolic(B));
    PetscCall(MatProductNumeric(B));

    PetscCall(MatScale(S_CC,m_one));
    if (n_vertices) {
      if (isCHOL || need_benign_correction) { /* if we can solve the interior problem with cholesky, we should also be fine with transposing here */
        PetscCall(MatTranspose(S_CV,MAT_REUSE_MATRIX,&S_VC));
      } else {
        Mat S_VCt;

        if (lda_rhs != n_R) {
          PetscCall(MatDestroy(&B));
          PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,n_R,n_constraints,work,&B));
          PetscCall(MatDenseSetLDA(B,lda_rhs));
        }
        PetscCall(MatMatMult(A_VR,B,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&S_VCt));
        PetscCall(MatCopy(S_VCt,S_VC,SAME_NONZERO_PATTERN));
        PetscCall(MatDestroy(&S_VCt));
      }
    }
    PetscCall(MatDestroy(&B));
    /* coarse basis functions */
    for (i=0;i<n_constraints;i++) {
      Vec v;

      PetscCall(VecPlaceArray(pcbddc->vec1_R,work+lda_rhs*i));
      PetscCall(MatDenseGetColumnVec(pcbddc->coarse_phi_B,i+n_vertices,&v));
      PetscCall(VecScatterBegin(pcbddc->R_to_B,pcbddc->vec1_R,v,INSERT_VALUES,SCATTER_FORWARD));
      PetscCall(VecScatterEnd(pcbddc->R_to_B,pcbddc->vec1_R,v,INSERT_VALUES,SCATTER_FORWARD));
      PetscCall(MatDenseRestoreColumnVec(pcbddc->coarse_phi_B,i+n_vertices,&v));
      if (pcbddc->switch_static || pcbddc->dbg_flag) {
        PetscInt    j;
        PetscScalar zero = 0.0;
        PetscCall(MatDenseGetColumnVec(pcbddc->coarse_phi_D,i+n_vertices,&v));
        PetscCall(VecScatterBegin(pcbddc->R_to_D,pcbddc->vec1_R,v,INSERT_VALUES,SCATTER_FORWARD));
        PetscCall(VecScatterEnd(pcbddc->R_to_D,pcbddc->vec1_R,v,INSERT_VALUES,SCATTER_FORWARD));
        for (j=0;j<pcbddc->benign_n;j++) PetscCall(VecSetValues(v,1,&p0_lidx_I[j],&zero,INSERT_VALUES));
        PetscCall(VecAssemblyBegin(v));
        PetscCall(VecAssemblyEnd(v));
        PetscCall(MatDenseRestoreColumnVec(pcbddc->coarse_phi_D,i+n_vertices,&v));
      }
      PetscCall(VecResetArray(pcbddc->vec1_R));
    }
  }
  if (n_constraints) {
    PetscCall(MatDestroy(&local_auxmat2_R));
  }
  PetscCall(PetscFree(p0_lidx_I));

  /* coarse matrix entries relative to B_0 */
  if (pcbddc->benign_n) {
    Mat               B0_B,B0_BPHI;
    IS                is_dummy;
    const PetscScalar *data;
    PetscInt          j;

    PetscCall(ISCreateStride(PETSC_COMM_SELF,pcbddc->benign_n,0,1,&is_dummy));
    PetscCall(MatCreateSubMatrix(pcbddc->benign_B0,is_dummy,pcis->is_B_local,MAT_INITIAL_MATRIX,&B0_B));
    PetscCall(ISDestroy(&is_dummy));
    PetscCall(MatMatMult(B0_B,pcbddc->coarse_phi_B,MAT_INITIAL_MATRIX,1.0,&B0_BPHI));
    PetscCall(MatConvert(B0_BPHI,MATSEQDENSE,MAT_INPLACE_MATRIX,&B0_BPHI));
    PetscCall(MatDenseGetArrayRead(B0_BPHI,&data));
    for (j=0;j<pcbddc->benign_n;j++) {
      PetscInt primal_idx = pcbddc->local_primal_size - pcbddc->benign_n + j;
      for (i=0;i<pcbddc->local_primal_size;i++) {
        coarse_submat_vals[primal_idx*pcbddc->local_primal_size+i] = data[i*pcbddc->benign_n+j];
        coarse_submat_vals[i*pcbddc->local_primal_size+primal_idx] = data[i*pcbddc->benign_n+j];
      }
    }
    PetscCall(MatDenseRestoreArrayRead(B0_BPHI,&data));
    PetscCall(MatDestroy(&B0_B));
    PetscCall(MatDestroy(&B0_BPHI));
  }

  /* compute other basis functions for non-symmetric problems */
  if (!pcbddc->symmetric_primal) {
    Mat         B_V=NULL,B_C=NULL;
    PetscScalar *marray;

    if (n_constraints) {
      Mat S_CCT,C_CRT;

      PetscCall(MatTranspose(C_CR,MAT_INITIAL_MATRIX,&C_CRT));
      PetscCall(MatTranspose(S_CC,MAT_INITIAL_MATRIX,&S_CCT));
      PetscCall(MatMatMult(C_CRT,S_CCT,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&B_C));
      PetscCall(MatDestroy(&S_CCT));
      if (n_vertices) {
        Mat S_VCT;

        PetscCall(MatTranspose(S_VC,MAT_INITIAL_MATRIX,&S_VCT));
        PetscCall(MatMatMult(C_CRT,S_VCT,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&B_V));
        PetscCall(MatDestroy(&S_VCT));
      }
      PetscCall(MatDestroy(&C_CRT));
    } else {
      PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,n_R,n_vertices,NULL,&B_V));
    }
    if (n_vertices && n_R) {
      PetscScalar    *av,*marray;
      const PetscInt *xadj,*adjncy;
      PetscInt       n;
      PetscBool      flg_row;

      /* B_V = B_V - A_VR^T */
      PetscCall(MatConvert(A_VR,MATSEQAIJ,MAT_INPLACE_MATRIX,&A_VR));
      PetscCall(MatGetRowIJ(A_VR,0,PETSC_FALSE,PETSC_FALSE,&n,&xadj,&adjncy,&flg_row));
      PetscCall(MatSeqAIJGetArray(A_VR,&av));
      PetscCall(MatDenseGetArray(B_V,&marray));
      for (i=0;i<n;i++) {
        PetscInt j;
        for (j=xadj[i];j<xadj[i+1];j++) marray[i*n_R + adjncy[j]] -= av[j];
      }
      PetscCall(MatDenseRestoreArray(B_V,&marray));
      PetscCall(MatRestoreRowIJ(A_VR,0,PETSC_FALSE,PETSC_FALSE,&n,&xadj,&adjncy,&flg_row));
      PetscCall(MatDestroy(&A_VR));
    }

    /* currently there's no support for MatTransposeMatSolve(F,B,X) */
    if (n_vertices) {
      PetscCall(MatDenseGetArray(B_V,&marray));
      for (i=0;i<n_vertices;i++) {
        PetscCall(VecPlaceArray(pcbddc->vec1_R,marray+i*n_R));
        PetscCall(VecPlaceArray(pcbddc->vec2_R,work+i*n_R));
        PetscCall(KSPSolveTranspose(pcbddc->ksp_R,pcbddc->vec1_R,pcbddc->vec2_R));
        PetscCall(KSPCheckSolve(pcbddc->ksp_R,pc,pcbddc->vec2_R));
        PetscCall(VecResetArray(pcbddc->vec1_R));
        PetscCall(VecResetArray(pcbddc->vec2_R));
      }
      PetscCall(MatDenseRestoreArray(B_V,&marray));
    }
    if (B_C) {
      PetscCall(MatDenseGetArray(B_C,&marray));
      for (i=n_vertices;i<n_constraints+n_vertices;i++) {
        PetscCall(VecPlaceArray(pcbddc->vec1_R,marray+(i-n_vertices)*n_R));
        PetscCall(VecPlaceArray(pcbddc->vec2_R,work+i*n_R));
        PetscCall(KSPSolveTranspose(pcbddc->ksp_R,pcbddc->vec1_R,pcbddc->vec2_R));
        PetscCall(KSPCheckSolve(pcbddc->ksp_R,pc,pcbddc->vec2_R));
        PetscCall(VecResetArray(pcbddc->vec1_R));
        PetscCall(VecResetArray(pcbddc->vec2_R));
      }
      PetscCall(MatDenseRestoreArray(B_C,&marray));
    }
    /* coarse basis functions */
    for (i=0;i<pcbddc->local_primal_size;i++) {
      Vec  v;

      PetscCall(VecPlaceArray(pcbddc->vec1_R,work+i*n_R));
      PetscCall(MatDenseGetColumnVec(pcbddc->coarse_psi_B,i,&v));
      PetscCall(VecScatterBegin(pcbddc->R_to_B,pcbddc->vec1_R,v,INSERT_VALUES,SCATTER_FORWARD));
      PetscCall(VecScatterEnd(pcbddc->R_to_B,pcbddc->vec1_R,v,INSERT_VALUES,SCATTER_FORWARD));
      if (i<n_vertices) {
        PetscScalar one = 1.0;
        PetscCall(VecSetValues(v,1,&idx_V_B[i],&one,INSERT_VALUES));
        PetscCall(VecAssemblyBegin(v));
        PetscCall(VecAssemblyEnd(v));
      }
      PetscCall(MatDenseRestoreColumnVec(pcbddc->coarse_psi_B,i,&v));

      if (pcbddc->switch_static || pcbddc->dbg_flag) {
        PetscCall(MatDenseGetColumnVec(pcbddc->coarse_psi_D,i,&v));
        PetscCall(VecScatterBegin(pcbddc->R_to_D,pcbddc->vec1_R,v,INSERT_VALUES,SCATTER_FORWARD));
        PetscCall(VecScatterEnd(pcbddc->R_to_D,pcbddc->vec1_R,v,INSERT_VALUES,SCATTER_FORWARD));
        PetscCall(MatDenseRestoreColumnVec(pcbddc->coarse_psi_D,i,&v));
      }
      PetscCall(VecResetArray(pcbddc->vec1_R));
    }
    PetscCall(MatDestroy(&B_V));
    PetscCall(MatDestroy(&B_C));
  }

  /* free memory */
  PetscCall(PetscFree(idx_V_B));
  PetscCall(MatDestroy(&S_VV));
  PetscCall(MatDestroy(&S_CV));
  PetscCall(MatDestroy(&S_VC));
  PetscCall(MatDestroy(&S_CC));
  PetscCall(PetscFree(work));
  if (n_vertices) {
    PetscCall(MatDestroy(&A_VR));
  }
  if (n_constraints) {
    PetscCall(MatDestroy(&C_CR));
  }
  PetscCall(PetscLogEventEnd(PC_BDDC_CorrectionSetUp[pcbddc->current_level],pc,0,0,0));

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
      PetscCall(PCBDDCBenignProject(pc,NULL,NULL,&A));
      PetscCall(MatCreateSubMatrix(A,pcis->is_I_local,pcis->is_I_local,MAT_INITIAL_MATRIX,&A_II));
      PetscCall(MatCreateSubMatrix(A,pcis->is_I_local,pcis->is_B_local,MAT_INITIAL_MATRIX,&A_IB));
      PetscCall(MatCreateSubMatrix(A,pcis->is_B_local,pcis->is_I_local,MAT_INITIAL_MATRIX,&A_BI));
      PetscCall(MatCreateSubMatrix(A,pcis->is_B_local,pcis->is_B_local,MAT_INITIAL_MATRIX,&A_BB));
      PetscCall(MatDestroy(&A));
    } else {
      PetscCall(MatConvert(pcis->A_II,checkmattype,MAT_INITIAL_MATRIX,&A_II));
      PetscCall(MatConvert(pcis->A_IB,checkmattype,MAT_INITIAL_MATRIX,&A_IB));
      PetscCall(MatConvert(pcis->A_BI,checkmattype,MAT_INITIAL_MATRIX,&A_BI));
      PetscCall(MatConvert(pcis->A_BB,checkmattype,MAT_INITIAL_MATRIX,&A_BB));
    }
    PetscCall(MatConvert(pcbddc->coarse_phi_D,checkmattype,MAT_INITIAL_MATRIX,&coarse_phi_D));
    PetscCall(MatConvert(pcbddc->coarse_phi_B,checkmattype,MAT_INITIAL_MATRIX,&coarse_phi_B));
    if (!pcbddc->symmetric_primal) {
      PetscCall(MatConvert(pcbddc->coarse_psi_D,checkmattype,MAT_INITIAL_MATRIX,&coarse_psi_D));
      PetscCall(MatConvert(pcbddc->coarse_psi_B,checkmattype,MAT_INITIAL_MATRIX,&coarse_psi_B));
    }
    PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,pcbddc->local_primal_size,pcbddc->local_primal_size,coarse_submat_vals,&coarse_sub_mat));

    PetscCall(PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"--------------------------------------------------\n"));
    PetscCall(PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"Check coarse sub mat computation (symmetric %d)\n",pcbddc->symmetric_primal));
    PetscCall(PetscViewerFlush(pcbddc->dbg_viewer));
    if (!pcbddc->symmetric_primal) {
      PetscCall(MatMatMult(A_II,coarse_phi_D,MAT_INITIAL_MATRIX,1.0,&AUXMAT));
      PetscCall(MatTransposeMatMult(coarse_psi_D,AUXMAT,MAT_INITIAL_MATRIX,1.0,&TM1));
      PetscCall(MatDestroy(&AUXMAT));
      PetscCall(MatMatMult(A_BB,coarse_phi_B,MAT_INITIAL_MATRIX,1.0,&AUXMAT));
      PetscCall(MatTransposeMatMult(coarse_psi_B,AUXMAT,MAT_INITIAL_MATRIX,1.0,&TM2));
      PetscCall(MatDestroy(&AUXMAT));
      PetscCall(MatMatMult(A_IB,coarse_phi_B,MAT_INITIAL_MATRIX,1.0,&AUXMAT));
      PetscCall(MatTransposeMatMult(coarse_psi_D,AUXMAT,MAT_INITIAL_MATRIX,1.0,&TM3));
      PetscCall(MatDestroy(&AUXMAT));
      PetscCall(MatMatMult(A_BI,coarse_phi_D,MAT_INITIAL_MATRIX,1.0,&AUXMAT));
      PetscCall(MatTransposeMatMult(coarse_psi_B,AUXMAT,MAT_INITIAL_MATRIX,1.0,&TM4));
      PetscCall(MatDestroy(&AUXMAT));
    } else {
      PetscCall(MatPtAP(A_II,coarse_phi_D,MAT_INITIAL_MATRIX,1.0,&TM1));
      PetscCall(MatPtAP(A_BB,coarse_phi_B,MAT_INITIAL_MATRIX,1.0,&TM2));
      PetscCall(MatMatMult(A_IB,coarse_phi_B,MAT_INITIAL_MATRIX,1.0,&AUXMAT));
      PetscCall(MatTransposeMatMult(coarse_phi_D,AUXMAT,MAT_INITIAL_MATRIX,1.0,&TM3));
      PetscCall(MatDestroy(&AUXMAT));
      PetscCall(MatMatMult(A_BI,coarse_phi_D,MAT_INITIAL_MATRIX,1.0,&AUXMAT));
      PetscCall(MatTransposeMatMult(coarse_phi_B,AUXMAT,MAT_INITIAL_MATRIX,1.0,&TM4));
      PetscCall(MatDestroy(&AUXMAT));
    }
    PetscCall(MatAXPY(TM1,one,TM2,DIFFERENT_NONZERO_PATTERN));
    PetscCall(MatAXPY(TM1,one,TM3,DIFFERENT_NONZERO_PATTERN));
    PetscCall(MatAXPY(TM1,one,TM4,DIFFERENT_NONZERO_PATTERN));
    PetscCall(MatConvert(TM1,MATSEQDENSE,MAT_INPLACE_MATRIX,&TM1));
    if (pcbddc->benign_n) {
      Mat               B0_B,B0_BPHI;
      const PetscScalar *data2;
      PetscScalar       *data;
      PetscInt          j;

      PetscCall(ISCreateStride(PETSC_COMM_SELF,pcbddc->benign_n,0,1,&is_dummy));
      PetscCall(MatCreateSubMatrix(pcbddc->benign_B0,is_dummy,pcis->is_B_local,MAT_INITIAL_MATRIX,&B0_B));
      PetscCall(MatMatMult(B0_B,coarse_phi_B,MAT_INITIAL_MATRIX,1.0,&B0_BPHI));
      PetscCall(MatConvert(B0_BPHI,MATSEQDENSE,MAT_INPLACE_MATRIX,&B0_BPHI));
      PetscCall(MatDenseGetArray(TM1,&data));
      PetscCall(MatDenseGetArrayRead(B0_BPHI,&data2));
      for (j=0;j<pcbddc->benign_n;j++) {
        PetscInt primal_idx = pcbddc->local_primal_size - pcbddc->benign_n + j;
        for (i=0;i<pcbddc->local_primal_size;i++) {
          data[primal_idx*pcbddc->local_primal_size+i] += data2[i*pcbddc->benign_n+j];
          data[i*pcbddc->local_primal_size+primal_idx] += data2[i*pcbddc->benign_n+j];
        }
      }
      PetscCall(MatDenseRestoreArray(TM1,&data));
      PetscCall(MatDenseRestoreArrayRead(B0_BPHI,&data2));
      PetscCall(MatDestroy(&B0_B));
      PetscCall(ISDestroy(&is_dummy));
      PetscCall(MatDestroy(&B0_BPHI));
    }
#if 0
  {
    PetscViewer viewer;
    char filename[256];
    sprintf(filename,"details_local_coarse_mat%d_level%d.m",PetscGlobalRank,pcbddc->current_level);
    PetscCall(PetscViewerASCIIOpen(PETSC_COMM_SELF,filename,&viewer));
    PetscCall(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB));
    PetscCall(PetscObjectSetName((PetscObject)coarse_sub_mat,"computed"));
    PetscCall(MatView(coarse_sub_mat,viewer));
    PetscCall(PetscObjectSetName((PetscObject)TM1,"projected"));
    PetscCall(MatView(TM1,viewer));
    if (pcbddc->coarse_phi_B) {
      PetscCall(PetscObjectSetName((PetscObject)pcbddc->coarse_phi_B,"phi_B"));
      PetscCall(MatView(pcbddc->coarse_phi_B,viewer));
    }
    if (pcbddc->coarse_phi_D) {
      PetscCall(PetscObjectSetName((PetscObject)pcbddc->coarse_phi_D,"phi_D"));
      PetscCall(MatView(pcbddc->coarse_phi_D,viewer));
    }
    if (pcbddc->coarse_psi_B) {
      PetscCall(PetscObjectSetName((PetscObject)pcbddc->coarse_psi_B,"psi_B"));
      PetscCall(MatView(pcbddc->coarse_psi_B,viewer));
    }
    if (pcbddc->coarse_psi_D) {
      PetscCall(PetscObjectSetName((PetscObject)pcbddc->coarse_psi_D,"psi_D"));
      PetscCall(MatView(pcbddc->coarse_psi_D,viewer));
    }
    PetscCall(PetscObjectSetName((PetscObject)pcbddc->local_mat,"A"));
    PetscCall(MatView(pcbddc->local_mat,viewer));
    PetscCall(PetscObjectSetName((PetscObject)pcbddc->ConstraintMatrix,"C"));
    PetscCall(MatView(pcbddc->ConstraintMatrix,viewer));
    PetscCall(PetscObjectSetName((PetscObject)pcis->is_I_local,"I"));
    PetscCall(ISView(pcis->is_I_local,viewer));
    PetscCall(PetscObjectSetName((PetscObject)pcis->is_B_local,"B"));
    PetscCall(ISView(pcis->is_B_local,viewer));
    PetscCall(PetscObjectSetName((PetscObject)pcbddc->is_R_local,"R"));
    PetscCall(ISView(pcbddc->is_R_local,viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }
#endif
    PetscCall(MatAXPY(TM1,m_one,coarse_sub_mat,DIFFERENT_NONZERO_PATTERN));
    PetscCall(MatNorm(TM1,NORM_FROBENIUS,&real_value));
    PetscCall(PetscViewerASCIIPushSynchronized(pcbddc->dbg_viewer));
    PetscCall(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d          matrix error % 1.14e\n",PetscGlobalRank,(double)real_value));

    /* check constraints */
    PetscCall(ISCreateStride(PETSC_COMM_SELF,pcbddc->local_primal_size-pcbddc->benign_n,0,1,&is_dummy));
    PetscCall(MatCreateSubMatrix(pcbddc->ConstraintMatrix,is_dummy,pcis->is_B_local,MAT_INITIAL_MATRIX,&C_B));
    if (!pcbddc->benign_n) { /* TODO: add benign case */
      PetscCall(MatMatMult(C_B,coarse_phi_B,MAT_INITIAL_MATRIX,1.0,&CPHI));
    } else {
      PetscScalar *data;
      Mat         tmat;
      PetscCall(MatDenseGetArray(pcbddc->coarse_phi_B,&data));
      PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,pcis->n_B,pcbddc->local_primal_size-pcbddc->benign_n,data,&tmat));
      PetscCall(MatDenseRestoreArray(pcbddc->coarse_phi_B,&data));
      PetscCall(MatMatMult(C_B,tmat,MAT_INITIAL_MATRIX,1.0,&CPHI));
      PetscCall(MatDestroy(&tmat));
    }
    PetscCall(MatCreateVecs(CPHI,&mones,NULL));
    PetscCall(VecSet(mones,-1.0));
    PetscCall(MatDiagonalSet(CPHI,mones,ADD_VALUES));
    PetscCall(MatNorm(CPHI,NORM_FROBENIUS,&real_value));
    PetscCall(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d phi constraints error % 1.14e\n",PetscGlobalRank,(double)real_value));
    if (!pcbddc->symmetric_primal) {
      PetscCall(MatMatMult(C_B,coarse_psi_B,MAT_REUSE_MATRIX,1.0,&CPHI));
      PetscCall(VecSet(mones,-1.0));
      PetscCall(MatDiagonalSet(CPHI,mones,ADD_VALUES));
      PetscCall(MatNorm(CPHI,NORM_FROBENIUS,&real_value));
      PetscCall(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d psi constraints error % 1.14e\n",PetscGlobalRank,(double)real_value));
    }
    PetscCall(MatDestroy(&C_B));
    PetscCall(MatDestroy(&CPHI));
    PetscCall(ISDestroy(&is_dummy));
    PetscCall(VecDestroy(&mones));
    PetscCall(PetscViewerFlush(pcbddc->dbg_viewer));
    PetscCall(MatDestroy(&A_II));
    PetscCall(MatDestroy(&A_BB));
    PetscCall(MatDestroy(&A_IB));
    PetscCall(MatDestroy(&A_BI));
    PetscCall(MatDestroy(&TM1));
    PetscCall(MatDestroy(&TM2));
    PetscCall(MatDestroy(&TM3));
    PetscCall(MatDestroy(&TM4));
    PetscCall(MatDestroy(&coarse_phi_D));
    PetscCall(MatDestroy(&coarse_phi_B));
    if (!pcbddc->symmetric_primal) {
      PetscCall(MatDestroy(&coarse_psi_D));
      PetscCall(MatDestroy(&coarse_psi_B));
    }
    PetscCall(MatDestroy(&coarse_sub_mat));
  }
  /* FINAL CUDA support (we cannot currently mix viennacl and cuda vectors */
  {
    PetscBool gpu;

    PetscCall(PetscObjectTypeCompare((PetscObject)pcis->vec1_N,VECSEQCUDA,&gpu));
    if (gpu) {
      if (pcbddc->local_auxmat1) {
        PetscCall(MatConvert(pcbddc->local_auxmat1,MATSEQDENSECUDA,MAT_INPLACE_MATRIX,&pcbddc->local_auxmat1));
      }
      if (pcbddc->local_auxmat2) {
        PetscCall(MatConvert(pcbddc->local_auxmat2,MATSEQDENSECUDA,MAT_INPLACE_MATRIX,&pcbddc->local_auxmat2));
      }
      if (pcbddc->coarse_phi_B) {
        PetscCall(MatConvert(pcbddc->coarse_phi_B,MATSEQDENSECUDA,MAT_INPLACE_MATRIX,&pcbddc->coarse_phi_B));
      }
      if (pcbddc->coarse_phi_D) {
        PetscCall(MatConvert(pcbddc->coarse_phi_D,MATSEQDENSECUDA,MAT_INPLACE_MATRIX,&pcbddc->coarse_phi_D));
      }
      if (pcbddc->coarse_psi_B) {
        PetscCall(MatConvert(pcbddc->coarse_psi_B,MATSEQDENSECUDA,MAT_INPLACE_MATRIX,&pcbddc->coarse_psi_B));
      }
      if (pcbddc->coarse_psi_D) {
        PetscCall(MatConvert(pcbddc->coarse_psi_D,MATSEQDENSECUDA,MAT_INPLACE_MATRIX,&pcbddc->coarse_psi_D));
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
  PetscCall(ISSorted(isrow,&rsorted));
  PetscCall(ISSorted(iscol,&csorted));
  PetscCall(ISGetLocalSize(isrow,&rsize));
  PetscCall(ISGetLocalSize(iscol,&csize));

  if (!rsorted) {
    const PetscInt *idxs;
    PetscInt *idxs_sorted,i;

    PetscCall(PetscMalloc1(rsize,&idxs_perm_r));
    PetscCall(PetscMalloc1(rsize,&idxs_sorted));
    for (i=0;i<rsize;i++) {
      idxs_perm_r[i] = i;
    }
    PetscCall(ISGetIndices(isrow,&idxs));
    PetscCall(PetscSortIntWithPermutation(rsize,idxs,idxs_perm_r));
    for (i=0;i<rsize;i++) {
      idxs_sorted[i] = idxs[idxs_perm_r[i]];
    }
    PetscCall(ISRestoreIndices(isrow,&idxs));
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF,rsize,idxs_sorted,PETSC_OWN_POINTER,&isrow_s));
  } else {
    PetscCall(PetscObjectReference((PetscObject)isrow));
    isrow_s = isrow;
  }

  if (!csorted) {
    if (isrow == iscol) {
      PetscCall(PetscObjectReference((PetscObject)isrow_s));
      iscol_s = isrow_s;
    } else {
      const PetscInt *idxs;
      PetscInt       *idxs_sorted,i;

      PetscCall(PetscMalloc1(csize,&idxs_perm_c));
      PetscCall(PetscMalloc1(csize,&idxs_sorted));
      for (i=0;i<csize;i++) {
        idxs_perm_c[i] = i;
      }
      PetscCall(ISGetIndices(iscol,&idxs));
      PetscCall(PetscSortIntWithPermutation(csize,idxs,idxs_perm_c));
      for (i=0;i<csize;i++) {
        idxs_sorted[i] = idxs[idxs_perm_c[i]];
      }
      PetscCall(ISRestoreIndices(iscol,&idxs));
      PetscCall(ISCreateGeneral(PETSC_COMM_SELF,csize,idxs_sorted,PETSC_OWN_POINTER,&iscol_s));
    }
  } else {
    PetscCall(PetscObjectReference((PetscObject)iscol));
    iscol_s = iscol;
  }

  PetscCall(MatCreateSubMatrices(A,1,&isrow_s,&iscol_s,MAT_INITIAL_MATRIX,&work_mat));

  if (!rsorted || !csorted) {
    Mat      new_mat;
    IS       is_perm_r,is_perm_c;

    if (!rsorted) {
      PetscInt *idxs_r,i;
      PetscCall(PetscMalloc1(rsize,&idxs_r));
      for (i=0;i<rsize;i++) {
        idxs_r[idxs_perm_r[i]] = i;
      }
      PetscCall(PetscFree(idxs_perm_r));
      PetscCall(ISCreateGeneral(PETSC_COMM_SELF,rsize,idxs_r,PETSC_OWN_POINTER,&is_perm_r));
    } else {
      PetscCall(ISCreateStride(PETSC_COMM_SELF,rsize,0,1,&is_perm_r));
    }
    PetscCall(ISSetPermutation(is_perm_r));

    if (!csorted) {
      if (isrow_s == iscol_s) {
        PetscCall(PetscObjectReference((PetscObject)is_perm_r));
        is_perm_c = is_perm_r;
      } else {
        PetscInt *idxs_c,i;
        PetscCheck(idxs_perm_c,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Permutation array not present");
        PetscCall(PetscMalloc1(csize,&idxs_c));
        for (i=0;i<csize;i++) {
          idxs_c[idxs_perm_c[i]] = i;
        }
        PetscCall(PetscFree(idxs_perm_c));
        PetscCall(ISCreateGeneral(PETSC_COMM_SELF,csize,idxs_c,PETSC_OWN_POINTER,&is_perm_c));
      }
    } else {
      PetscCall(ISCreateStride(PETSC_COMM_SELF,csize,0,1,&is_perm_c));
    }
    PetscCall(ISSetPermutation(is_perm_c));

    PetscCall(MatPermute(work_mat[0],is_perm_r,is_perm_c,&new_mat));
    PetscCall(MatDestroy(&work_mat[0]));
    work_mat[0] = new_mat;
    PetscCall(ISDestroy(&is_perm_r));
    PetscCall(ISDestroy(&is_perm_c));
  }

  PetscCall(PetscObjectReference((PetscObject)work_mat[0]));
  *B = work_mat[0];
  PetscCall(MatDestroyMatrices(1,&work_mat));
  PetscCall(ISDestroy(&isrow_s));
  PetscCall(ISDestroy(&iscol_s));
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
  PetscCall(MatDestroy(&pcbddc->local_mat));
  PetscCall(MatGetSize(matis->A,&local_size,NULL));
  PetscCall(ISCreateStride(PetscObjectComm((PetscObject)matis->A),local_size,0,1,&is_local));
  PetscCall(ISLocalToGlobalMappingApplyIS(matis->rmapping,is_local,&is_global));
  PetscCall(ISDestroy(&is_local));
  PetscCall(MatCreateSubMatrixUnsorted(ChangeOfBasisMatrix,is_global,is_global,&new_mat));
  PetscCall(ISDestroy(&is_global));

  if (pcbddc->dbg_flag) {
    Vec       x,x_change;
    PetscReal error;

    PetscCall(MatCreateVecs(ChangeOfBasisMatrix,&x,&x_change));
    PetscCall(VecSetRandom(x,NULL));
    PetscCall(MatMult(ChangeOfBasisMatrix,x,x_change));
    PetscCall(VecScatterBegin(matis->cctx,x,matis->x,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(matis->cctx,x,matis->x,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(MatMult(new_mat,matis->x,matis->y));
    if (!pcbddc->change_interior) {
      const PetscScalar *x,*y,*v;
      PetscReal         lerror = 0.;
      PetscInt          i;

      PetscCall(VecGetArrayRead(matis->x,&x));
      PetscCall(VecGetArrayRead(matis->y,&y));
      PetscCall(VecGetArrayRead(matis->counter,&v));
      for (i=0;i<local_size;i++)
        if (PetscRealPart(v[i]) < 1.5 && PetscAbsScalar(x[i]-y[i]) > lerror)
          lerror = PetscAbsScalar(x[i]-y[i]);
      PetscCall(VecRestoreArrayRead(matis->x,&x));
      PetscCall(VecRestoreArrayRead(matis->y,&y));
      PetscCall(VecRestoreArrayRead(matis->counter,&v));
      PetscCall(MPIU_Allreduce(&lerror,&error,1,MPIU_REAL,MPIU_MAX,PetscObjectComm((PetscObject)pc)));
      if (error > PETSC_SMALL) {
        if (!pcbddc->user_ChangeOfBasisMatrix || pcbddc->current_level) {
          SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_PLIB,"Error global vs local change on I: %1.6e",(double)error);
        } else {
          SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_USER,"Error global vs local change on I: %1.6e",(double)error);
        }
      }
    }
    PetscCall(VecScatterBegin(matis->rctx,matis->y,x,INSERT_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterEnd(matis->rctx,matis->y,x,INSERT_VALUES,SCATTER_REVERSE));
    PetscCall(VecAXPY(x,-1.0,x_change));
    PetscCall(VecNorm(x,NORM_INFINITY,&error));
    if (error > PETSC_SMALL) {
      if (!pcbddc->user_ChangeOfBasisMatrix || pcbddc->current_level) {
        SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_PLIB,"Error global vs local change on N: %1.6e",(double)error);
      } else {
        SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_USER,"Error global vs local change on N: %1.6e",(double)error);
      }
    }
    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroy(&x_change));
  }

  /* lA is present if we are setting up an inner BDDC for a saddle point FETI-DP */
  PetscCall(PetscObjectQuery((PetscObject)pc,"__KSPFETIDP_lA" ,(PetscObject*)&lA));

  /* TODO: HOW TO WORK WITH BAIJ and SBAIJ and SEQDENSE? */
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)matis->A,MATSEQAIJ,&isseqaij));
  if (isseqaij) {
    PetscCall(MatDestroy(&pcbddc->local_mat));
    PetscCall(MatPtAP(matis->A,new_mat,MAT_INITIAL_MATRIX,2.0,&pcbddc->local_mat));
    if (lA) {
      Mat work;
      PetscCall(MatPtAP(lA,new_mat,MAT_INITIAL_MATRIX,2.0,&work));
      PetscCall(PetscObjectCompose((PetscObject)pc,"__KSPFETIDP_lA" ,(PetscObject)work));
      PetscCall(MatDestroy(&work));
    }
  } else {
    Mat work_mat;

    PetscCall(MatDestroy(&pcbddc->local_mat));
    PetscCall(MatConvert(matis->A,MATSEQAIJ,MAT_INITIAL_MATRIX,&work_mat));
    PetscCall(MatPtAP(work_mat,new_mat,MAT_INITIAL_MATRIX,2.0,&pcbddc->local_mat));
    PetscCall(MatDestroy(&work_mat));
    if (lA) {
      Mat work;
      PetscCall(MatConvert(lA,MATSEQAIJ,MAT_INITIAL_MATRIX,&work_mat));
      PetscCall(MatPtAP(work_mat,new_mat,MAT_INITIAL_MATRIX,2.0,&work));
      PetscCall(PetscObjectCompose((PetscObject)pc,"__KSPFETIDP_lA" ,(PetscObject)work));
      PetscCall(MatDestroy(&work));
    }
  }
  if (matis->A->symmetric_set) {
    PetscCall(MatSetOption(pcbddc->local_mat,MAT_SYMMETRIC,matis->A->symmetric));
#if !defined(PETSC_USE_COMPLEX)
    PetscCall(MatSetOption(pcbddc->local_mat,MAT_HERMITIAN,matis->A->symmetric));
#endif
  }
  PetscCall(MatDestroy(&new_mat));
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
  PetscCall(ISDestroy(&pcbddc->is_R_local));
  PetscCall(VecScatterDestroy(&pcbddc->R_to_B));
  PetscCall(VecScatterDestroy(&pcbddc->R_to_D));
  /* Set Non-overlapping dimensions */
  n_B = pcis->n_B;
  n_D = pcis->n - n_B;
  n_vertices = pcbddc->n_vertices;

  /* Dohrmann's notation: dofs splitted in R (Remaining: all dofs but the vertices) and V (Vertices) */

  /* create auxiliary bitmask and allocate workspace */
  if (!sub_schurs || !sub_schurs->reuse_solver) {
    PetscCall(PetscMalloc1(pcis->n-n_vertices,&idx_R_local));
    PetscCall(PetscBTCreate(pcis->n,&bitmask));
    for (i=0;i<n_vertices;i++) {
      PetscCall(PetscBTSet(bitmask,pcbddc->local_primal_ref_node[i]));
    }

    for (i=0, n_R=0; i<pcis->n; i++) {
      if (!PetscBTLookup(bitmask,i)) {
        idx_R_local[n_R++] = i;
      }
    }
  } else { /* A different ordering (already computed) is present if we are reusing the Schur solver */
    PCBDDCReuseSolvers reuse_solver = sub_schurs->reuse_solver;

    PetscCall(ISGetIndices(reuse_solver->is_R,(const PetscInt**)&idx_R_local));
    PetscCall(ISGetLocalSize(reuse_solver->is_R,&n_R));
  }

  /* Block code */
  vbs = 1;
  PetscCall(MatGetBlockSize(pcbddc->local_mat,&bs));
  if (bs>1 && !(n_vertices%bs)) {
    PetscBool is_blocked = PETSC_TRUE;
    PetscInt  *vary;
    if (!sub_schurs || !sub_schurs->reuse_solver) {
      PetscCall(PetscMalloc1(pcis->n/bs,&vary));
      PetscCall(PetscArrayzero(vary,pcis->n/bs));
      /* Verify that the vertex indices correspond to each element in a block (code taken from sbaij2.c) */
      /* it is ok to check this way since local_primal_ref_node are always sorted by local numbering and idx_R_local is obtained as a complement */
      for (i=0; i<n_vertices; i++) vary[pcbddc->local_primal_ref_node[i]/bs]++;
      for (i=0; i<pcis->n/bs; i++) {
        if (vary[i]!=0 && vary[i]!=bs) {
          is_blocked = PETSC_FALSE;
          break;
        }
      }
      PetscCall(PetscFree(vary));
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
  PetscCall(ISCreateBlock(PETSC_COMM_SELF,vbs,n_R/vbs,idx_R_local,PETSC_COPY_VALUES,&pcbddc->is_R_local));
  if (sub_schurs && sub_schurs->reuse_solver) {
    PCBDDCReuseSolvers reuse_solver = sub_schurs->reuse_solver;

    PetscCall(ISRestoreIndices(reuse_solver->is_R,(const PetscInt**)&idx_R_local));
    PetscCall(ISDestroy(&reuse_solver->is_R));
    PetscCall(PetscObjectReference((PetscObject)pcbddc->is_R_local));
    reuse_solver->is_R = pcbddc->is_R_local;
  } else {
    PetscCall(PetscFree(idx_R_local));
  }

  /* print some info if requested */
  if (pcbddc->dbg_flag) {
    PetscCall(PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"--------------------------------------------------\n"));
    PetscCall(PetscViewerFlush(pcbddc->dbg_viewer));
    PetscCall(PetscViewerASCIIPushSynchronized(pcbddc->dbg_viewer));
    PetscCall(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d local dimensions\n",PetscGlobalRank));
    PetscCall(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"local_size = %" PetscInt_FMT ", dirichlet_size = %" PetscInt_FMT ", boundary_size = %" PetscInt_FMT "\n",pcis->n,n_D,n_B));
    PetscCall(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"r_size = %" PetscInt_FMT ", v_size = %" PetscInt_FMT ", constraints = %" PetscInt_FMT ", local_primal_size = %" PetscInt_FMT "\n",n_R,n_vertices,pcbddc->local_primal_size-n_vertices-pcbddc->benign_n,pcbddc->local_primal_size));
    PetscCall(PetscViewerFlush(pcbddc->dbg_viewer));
  }

  /* VecScatters pcbddc->R_to_B and (optionally) pcbddc->R_to_D */
  if (!sub_schurs || !sub_schurs->reuse_solver) {
    IS       is_aux1,is_aux2;
    PetscInt *aux_array1,*aux_array2,*is_indices,*idx_R_local;

    PetscCall(ISGetIndices(pcbddc->is_R_local,(const PetscInt**)&idx_R_local));
    PetscCall(PetscMalloc1(pcis->n_B-n_vertices,&aux_array1));
    PetscCall(PetscMalloc1(pcis->n_B-n_vertices,&aux_array2));
    PetscCall(ISGetIndices(pcis->is_I_local,(const PetscInt**)&is_indices));
    for (i=0; i<n_D; i++) {
      PetscCall(PetscBTSet(bitmask,is_indices[i]));
    }
    PetscCall(ISRestoreIndices(pcis->is_I_local,(const PetscInt**)&is_indices));
    for (i=0, j=0; i<n_R; i++) {
      if (!PetscBTLookup(bitmask,idx_R_local[i])) {
        aux_array1[j++] = i;
      }
    }
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF,j,aux_array1,PETSC_OWN_POINTER,&is_aux1));
    PetscCall(ISGetIndices(pcis->is_B_local,(const PetscInt**)&is_indices));
    for (i=0, j=0; i<n_B; i++) {
      if (!PetscBTLookup(bitmask,is_indices[i])) {
        aux_array2[j++] = i;
      }
    }
    PetscCall(ISRestoreIndices(pcis->is_B_local,(const PetscInt**)&is_indices));
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF,j,aux_array2,PETSC_OWN_POINTER,&is_aux2));
    PetscCall(VecScatterCreate(pcbddc->vec1_R,is_aux1,pcis->vec1_B,is_aux2,&pcbddc->R_to_B));
    PetscCall(ISDestroy(&is_aux1));
    PetscCall(ISDestroy(&is_aux2));

    if (pcbddc->switch_static || pcbddc->dbg_flag) {
      PetscCall(PetscMalloc1(n_D,&aux_array1));
      for (i=0, j=0; i<n_R; i++) {
        if (PetscBTLookup(bitmask,idx_R_local[i])) {
          aux_array1[j++] = i;
        }
      }
      PetscCall(ISCreateGeneral(PETSC_COMM_SELF,j,aux_array1,PETSC_OWN_POINTER,&is_aux1));
      PetscCall(VecScatterCreate(pcbddc->vec1_R,is_aux1,pcis->vec1_D,(IS)0,&pcbddc->R_to_D));
      PetscCall(ISDestroy(&is_aux1));
    }
    PetscCall(PetscBTDestroy(&bitmask));
    PetscCall(ISRestoreIndices(pcbddc->is_R_local,(const PetscInt**)&idx_R_local));
  } else {
    PCBDDCReuseSolvers reuse_solver = sub_schurs->reuse_solver;
    IS                 tis;
    PetscInt           schur_size;

    PetscCall(ISGetLocalSize(reuse_solver->is_B,&schur_size));
    PetscCall(ISCreateStride(PETSC_COMM_SELF,schur_size,n_D,1,&tis));
    PetscCall(VecScatterCreate(pcbddc->vec1_R,tis,pcis->vec1_B,reuse_solver->is_B,&pcbddc->R_to_B));
    PetscCall(ISDestroy(&tis));
    if (pcbddc->switch_static || pcbddc->dbg_flag) {
      PetscCall(ISCreateStride(PETSC_COMM_SELF,n_D,0,1,&tis));
      PetscCall(VecScatterCreate(pcbddc->vec1_R,tis,pcis->vec1_D,(IS)0,&pcbddc->R_to_D));
      PetscCall(ISDestroy(&tis));
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
      PetscCall(MatISGetLocalMat(A,&B));
    }
    sct  = matis->cctx;
    PetscCall(PetscObjectReference((PetscObject)sct));
  } else {
    PetscCall(MatGetNullSpace(B,&NullSpace));
    if (!NullSpace) {
      PetscCall(MatGetNearNullSpace(B,&NullSpace));
    }
    if (NullSpace) PetscFunctionReturn(0);
  }
  PetscCall(MatGetNullSpace(A,&NullSpace));
  if (!NullSpace) {
    PetscCall(MatGetNearNullSpace(A,&NullSpace));
  }
  if (!NullSpace) PetscFunctionReturn(0);

  PetscCall(MatCreateVecs(A,&v,NULL));
  PetscCall(MatCreateVecs(B,&v2,NULL));
  if (!sct) {
    PetscCall(VecScatterCreate(v,is,v2,NULL,&sct));
  }
  PetscCall(MatNullSpaceGetVecs(NullSpace,&nnsp_has_cnst,&nnsp_size,(const Vec**)&nullvecs));
  bsiz = bsiz2 = nnsp_size+!!nnsp_has_cnst;
  PetscCall(PetscMalloc1(bsiz,&nullvecs2));
  PetscCall(VecGetBlockSize(v2,&bs));
  PetscCall(VecGetSize(v2,&N));
  PetscCall(VecGetLocalSize(v2,&n));
  PetscCall(PetscMalloc1(n*bsiz,&ddata));
  for (k=0;k<nnsp_size;k++) {
    PetscCall(VecCreateMPIWithArray(PetscObjectComm((PetscObject)B),bs,n,N,ddata + n*k,&nullvecs2[k]));
    PetscCall(VecScatterBegin(sct,nullvecs[k],nullvecs2[k],INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(sct,nullvecs[k],nullvecs2[k],INSERT_VALUES,SCATTER_FORWARD));
  }
  if (nnsp_has_cnst) {
    PetscCall(VecCreateMPIWithArray(PetscObjectComm((PetscObject)B),bs,n,N,ddata + n*nnsp_size,&nullvecs2[nnsp_size]));
    PetscCall(VecSet(nullvecs2[nnsp_size],1.0));
  }
  PetscCall(PCBDDCOrthonormalizeVecs(&bsiz2,nullvecs2));
  PetscCall(MatNullSpaceCreate(PetscObjectComm((PetscObject)B),PETSC_FALSE,bsiz2,nullvecs2,&NullSpace));

  PetscCall(MatCreateDense(PetscObjectComm((PetscObject)B),n,PETSC_DECIDE,N,bsiz2,ddata,&dmat));
  PetscCall(PetscContainerCreate(PetscObjectComm((PetscObject)B),&c));
  PetscCall(PetscContainerSetPointer(c,ddata));
  PetscCall(PetscContainerSetUserDestroy(c,PetscContainerUserDestroyDefault));
  PetscCall(PetscObjectCompose((PetscObject)dmat,"_PBDDC_Null_dmat_arr",(PetscObject)c));
  PetscCall(PetscContainerDestroy(&c));
  PetscCall(PetscObjectCompose((PetscObject)NullSpace,"_PBDDC_Null_dmat",(PetscObject)dmat));
  PetscCall(MatDestroy(&dmat));

  for (k=0;k<bsiz;k++) {
    PetscCall(VecDestroy(&nullvecs2[k]));
  }
  PetscCall(PetscFree(nullvecs2));
  PetscCall(MatSetNearNullSpace(B,NullSpace));
  PetscCall(MatNullSpaceDestroy(&NullSpace));
  PetscCall(VecDestroy(&v));
  PetscCall(VecDestroy(&v2));
  PetscCall(VecScatterDestroy(&sct));
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
  PetscCall(PetscLogEventBegin(PC_BDDC_LocalSolvers[pcbddc->current_level],pc,0,0,0));
  /* approximate solver, propagate NearNullSpace if needed */
  if (!pc->setupcalled && (pcbddc->NullSpace_corr[0] || pcbddc->NullSpace_corr[2])) {
    MatNullSpace gnnsp1,gnnsp2;
    PetscBool    lhas,ghas;

    PetscCall(MatGetNearNullSpace(pcbddc->local_mat,&nnsp));
    PetscCall(MatGetNearNullSpace(pc->pmat,&gnnsp1));
    PetscCall(MatGetNullSpace(pc->pmat,&gnnsp2));
    lhas = nnsp ? PETSC_TRUE : PETSC_FALSE;
    PetscCall(MPIU_Allreduce(&lhas,&ghas,1,MPIU_BOOL,MPI_LOR,PetscObjectComm((PetscObject)pc)));
    if (!ghas && (gnnsp1 || gnnsp2)) {
      PetscCall(MatNullSpacePropagateAny_Private(pc->pmat,NULL,NULL));
    }
  }

  /* compute prefixes */
  PetscCall(PetscStrcpy(dir_prefix,""));
  PetscCall(PetscStrcpy(neu_prefix,""));
  if (!pcbddc->current_level) {
    PetscCall(PetscStrncpy(dir_prefix,((PetscObject)pc)->prefix,sizeof(dir_prefix)));
    PetscCall(PetscStrncpy(neu_prefix,((PetscObject)pc)->prefix,sizeof(neu_prefix)));
    PetscCall(PetscStrlcat(dir_prefix,"pc_bddc_dirichlet_",sizeof(dir_prefix)));
    PetscCall(PetscStrlcat(neu_prefix,"pc_bddc_neumann_",sizeof(neu_prefix)));
  } else {
    PetscCall(PetscSNPrintf(str_level,sizeof(str_level),"l%d_",(int)(pcbddc->current_level)));
    PetscCall(PetscStrlen(((PetscObject)pc)->prefix,&len));
    len -= 15; /* remove "pc_bddc_coarse_" */
    if (pcbddc->current_level>1) len -= 3; /* remove "lX_" with X level number */
    if (pcbddc->current_level>10) len -= 1; /* remove another char from level number */
    /* Nonstandard use of PetscStrncpy() to only copy a portion of the input string */
    PetscCall(PetscStrncpy(dir_prefix,((PetscObject)pc)->prefix,len+1));
    PetscCall(PetscStrncpy(neu_prefix,((PetscObject)pc)->prefix,len+1));
    PetscCall(PetscStrlcat(dir_prefix,"pc_bddc_dirichlet_",sizeof(dir_prefix)));
    PetscCall(PetscStrlcat(neu_prefix,"pc_bddc_neumann_",sizeof(neu_prefix)));
    PetscCall(PetscStrlcat(dir_prefix,str_level,sizeof(dir_prefix)));
    PetscCall(PetscStrlcat(neu_prefix,str_level,sizeof(neu_prefix)));
  }

  /* DIRICHLET PROBLEM */
  if (dirichlet) {
    PCBDDCSubSchurs sub_schurs = pcbddc->sub_schurs;
    if (pcbddc->benign_n && !pcbddc->benign_change_explicit) {
      PetscCheck(sub_schurs && sub_schurs->reuse_solver,PETSC_COMM_SELF,PETSC_ERR_SUP,"Not yet implemented");
      if (pcbddc->dbg_flag) {
        Mat    A_IIn;

        PetscCall(PCBDDCBenignProject(pc,pcis->is_I_local,pcis->is_I_local,&A_IIn));
        PetscCall(MatDestroy(&pcis->A_II));
        pcis->A_II = A_IIn;
      }
    }
    if (pcbddc->local_mat->symmetric_set) {
      PetscCall(MatSetOption(pcis->A_II,MAT_SYMMETRIC,pcbddc->local_mat->symmetric));
    }
    /* Matrix for Dirichlet problem is pcis->A_II */
    n_D  = pcis->n - pcis->n_B;
    opts = PETSC_FALSE;
    if (!pcbddc->ksp_D) { /* create object if not yet build */
      opts = PETSC_TRUE;
      PetscCall(KSPCreate(PETSC_COMM_SELF,&pcbddc->ksp_D));
      PetscCall(PetscObjectIncrementTabLevel((PetscObject)pcbddc->ksp_D,(PetscObject)pc,1));
      /* default */
      PetscCall(KSPSetType(pcbddc->ksp_D,KSPPREONLY));
      PetscCall(KSPSetOptionsPrefix(pcbddc->ksp_D,dir_prefix));
      PetscCall(PetscObjectTypeCompare((PetscObject)pcis->pA_II,MATSEQSBAIJ,&issbaij));
      PetscCall(KSPGetPC(pcbddc->ksp_D,&pc_temp));
      if (issbaij) {
        PetscCall(PCSetType(pc_temp,PCCHOLESKY));
      } else {
        PetscCall(PCSetType(pc_temp,PCLU));
      }
      PetscCall(KSPSetErrorIfNotConverged(pcbddc->ksp_D,pc->erroriffailure));
    }
    PetscCall(MatSetOptionsPrefix(pcis->pA_II,((PetscObject)pcbddc->ksp_D)->prefix));
    PetscCall(KSPSetOperators(pcbddc->ksp_D,pcis->A_II,pcis->pA_II));
    /* Allow user's customization */
    if (opts) {
      PetscCall(KSPSetFromOptions(pcbddc->ksp_D));
    }
    PetscCall(MatGetNearNullSpace(pcis->pA_II,&nnsp));
    if (pcbddc->NullSpace_corr[0] && !nnsp) { /* approximate solver, propagate NearNullSpace */
      PetscCall(MatNullSpacePropagateAny_Private(pcbddc->local_mat,pcis->is_I_local,pcis->pA_II));
    }
    PetscCall(MatGetNearNullSpace(pcis->pA_II,&nnsp));
    PetscCall(KSPGetPC(pcbddc->ksp_D,&pc_temp));
    PetscCall(PetscObjectQueryFunction((PetscObject)pc_temp,"PCSetCoordinates_C",&f));
    if (f && pcbddc->mat_graph->cloc && !nnsp) {
      PetscReal      *coords = pcbddc->mat_graph->coords,*scoords;
      const PetscInt *idxs;
      PetscInt       cdim = pcbddc->mat_graph->cdim,nl,i,d;

      PetscCall(ISGetLocalSize(pcis->is_I_local,&nl));
      PetscCall(ISGetIndices(pcis->is_I_local,&idxs));
      PetscCall(PetscMalloc1(nl*cdim,&scoords));
      for (i=0;i<nl;i++) {
        for (d=0;d<cdim;d++) {
          scoords[i*cdim+d] = coords[idxs[i]*cdim+d];
        }
      }
      PetscCall(ISRestoreIndices(pcis->is_I_local,&idxs));
      PetscCall(PCSetCoordinates(pc_temp,cdim,nl,scoords));
      PetscCall(PetscFree(scoords));
    }
    if (sub_schurs && sub_schurs->reuse_solver) {
      PCBDDCReuseSolvers reuse_solver = sub_schurs->reuse_solver;

      PetscCall(KSPSetPC(pcbddc->ksp_D,reuse_solver->interior_solver));
    }

    /* umfpack interface has a bug when matrix dimension is zero. TODO solve from umfpack interface */
    if (!n_D) {
      PetscCall(KSPGetPC(pcbddc->ksp_D,&pc_temp));
      PetscCall(PCSetType(pc_temp,PCNONE));
    }
    PetscCall(KSPSetUp(pcbddc->ksp_D));
    /* set ksp_D into pcis data */
    PetscCall(PetscObjectReference((PetscObject)pcbddc->ksp_D));
    PetscCall(KSPDestroy(&pcis->ksp_D));
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
      PetscCall(PetscObjectQuery((PetscObject)sub_schurs->A,"__KSPFETIDP_iP",(PetscObject*)&iP));
      if (iP) reuse_neumann_solver = PETSC_FALSE;
    }
    /* Matrix for Neumann problem is A_RR -> we need to create/reuse it at this point */
    PetscCall(ISGetSize(pcbddc->is_R_local,&n_R));
    if (pcbddc->ksp_R) { /* already created ksp */
      PetscInt nn_R;
      PetscCall(KSPGetOperators(pcbddc->ksp_R,NULL,&A_RR));
      PetscCall(PetscObjectReference((PetscObject)A_RR));
      PetscCall(MatGetSize(A_RR,&nn_R,NULL));
      if (nn_R != n_R) { /* old ksp is not reusable, so reset it */
        PetscCall(KSPReset(pcbddc->ksp_R));
        PetscCall(MatDestroy(&A_RR));
        reuse = MAT_INITIAL_MATRIX;
      } else { /* same sizes, but nonzero pattern depend on primal vertices so it can be changed */
        if (pcbddc->new_primal_space_local) { /* we are not sure the matrix will have the same nonzero pattern */
          PetscCall(MatDestroy(&A_RR));
          reuse = MAT_INITIAL_MATRIX;
        } else { /* safe to reuse the matrix */
          reuse = MAT_REUSE_MATRIX;
        }
      }
      /* last check */
      if (pc->flag == DIFFERENT_NONZERO_PATTERN) {
        PetscCall(MatDestroy(&A_RR));
        reuse = MAT_INITIAL_MATRIX;
      }
    } else { /* first time, so we need to create the matrix */
      reuse = MAT_INITIAL_MATRIX;
    }
    /* convert pcbddc->local_mat if needed later in PCBDDCSetUpCorrection
       TODO: Get Rid of these conversions */
    PetscCall(MatGetBlockSize(pcbddc->local_mat,&mbs));
    PetscCall(ISGetBlockSize(pcbddc->is_R_local,&ibs));
    PetscCall(PetscObjectTypeCompare((PetscObject)pcbddc->local_mat,MATSEQSBAIJ,&issbaij));
    if (ibs != mbs) { /* need to convert to SEQAIJ to extract any submatrix with is_R_local */
      if (matis->A == pcbddc->local_mat) {
        PetscCall(MatDestroy(&pcbddc->local_mat));
        PetscCall(MatConvert(matis->A,MATSEQAIJ,MAT_INITIAL_MATRIX,&pcbddc->local_mat));
      } else {
        PetscCall(MatConvert(pcbddc->local_mat,MATSEQAIJ,MAT_INPLACE_MATRIX,&pcbddc->local_mat));
      }
    } else if (issbaij) { /* need to convert to BAIJ to get offdiagonal blocks */
      if (matis->A == pcbddc->local_mat) {
        PetscCall(MatDestroy(&pcbddc->local_mat));
        PetscCall(MatConvert(matis->A,mbs > 1 ? MATSEQBAIJ : MATSEQAIJ,MAT_INITIAL_MATRIX,&pcbddc->local_mat));
      } else {
        PetscCall(MatConvert(pcbddc->local_mat,mbs > 1 ? MATSEQBAIJ : MATSEQAIJ,MAT_INPLACE_MATRIX,&pcbddc->local_mat));
      }
    }
    /* extract A_RR */
    if (reuse_neumann_solver) {
      PCBDDCReuseSolvers reuse_solver = sub_schurs->reuse_solver;

      if (pcbddc->dbg_flag) { /* we need A_RR to test the solver later */
        PetscCall(MatDestroy(&A_RR));
        if (reuse_solver->benign_n) { /* we are not using the explicit change of basis on the pressures */
          PetscCall(PCBDDCBenignProject(pc,pcbddc->is_R_local,pcbddc->is_R_local,&A_RR));
        } else {
          PetscCall(MatCreateSubMatrix(pcbddc->local_mat,pcbddc->is_R_local,pcbddc->is_R_local,MAT_INITIAL_MATRIX,&A_RR));
        }
      } else {
        PetscCall(MatDestroy(&A_RR));
        PetscCall(PCGetOperators(reuse_solver->correction_solver,&A_RR,NULL));
        PetscCall(PetscObjectReference((PetscObject)A_RR));
      }
    } else { /* we have to build the neumann solver, so we need to extract the relevant matrix */
      PetscCall(MatCreateSubMatrix(pcbddc->local_mat,pcbddc->is_R_local,pcbddc->is_R_local,reuse,&A_RR));
    }
    if (pcbddc->local_mat->symmetric_set) {
      PetscCall(MatSetOption(A_RR,MAT_SYMMETRIC,pcbddc->local_mat->symmetric));
    }
    opts = PETSC_FALSE;
    if (!pcbddc->ksp_R) { /* create object if not present */
      opts = PETSC_TRUE;
      PetscCall(KSPCreate(PETSC_COMM_SELF,&pcbddc->ksp_R));
      PetscCall(PetscObjectIncrementTabLevel((PetscObject)pcbddc->ksp_R,(PetscObject)pc,1));
      /* default */
      PetscCall(KSPSetType(pcbddc->ksp_R,KSPPREONLY));
      PetscCall(KSPSetOptionsPrefix(pcbddc->ksp_R,neu_prefix));
      PetscCall(KSPGetPC(pcbddc->ksp_R,&pc_temp));
      PetscCall(PetscObjectTypeCompare((PetscObject)A_RR,MATSEQSBAIJ,&issbaij));
      if (issbaij) {
        PetscCall(PCSetType(pc_temp,PCCHOLESKY));
      } else {
        PetscCall(PCSetType(pc_temp,PCLU));
      }
      PetscCall(KSPSetErrorIfNotConverged(pcbddc->ksp_R,pc->erroriffailure));
    }
    PetscCall(KSPSetOperators(pcbddc->ksp_R,A_RR,A_RR));
    PetscCall(MatSetOptionsPrefix(A_RR,((PetscObject)pcbddc->ksp_R)->prefix));
    if (opts) { /* Allow user's customization once */
      PetscCall(KSPSetFromOptions(pcbddc->ksp_R));
    }
    PetscCall(MatGetNearNullSpace(A_RR,&nnsp));
    if (pcbddc->NullSpace_corr[2] && !nnsp) { /* approximate solver, propagate NearNullSpace */
      PetscCall(MatNullSpacePropagateAny_Private(pcbddc->local_mat,pcbddc->is_R_local,A_RR));
    }
    PetscCall(MatGetNearNullSpace(A_RR,&nnsp));
    PetscCall(KSPGetPC(pcbddc->ksp_R,&pc_temp));
    PetscCall(PetscObjectQueryFunction((PetscObject)pc_temp,"PCSetCoordinates_C",&f));
    if (f && pcbddc->mat_graph->cloc && !nnsp) {
      PetscReal      *coords = pcbddc->mat_graph->coords,*scoords;
      const PetscInt *idxs;
      PetscInt       cdim = pcbddc->mat_graph->cdim,nl,i,d;

      PetscCall(ISGetLocalSize(pcbddc->is_R_local,&nl));
      PetscCall(ISGetIndices(pcbddc->is_R_local,&idxs));
      PetscCall(PetscMalloc1(nl*cdim,&scoords));
      for (i=0;i<nl;i++) {
        for (d=0;d<cdim;d++) {
          scoords[i*cdim+d] = coords[idxs[i]*cdim+d];
        }
      }
      PetscCall(ISRestoreIndices(pcbddc->is_R_local,&idxs));
      PetscCall(PCSetCoordinates(pc_temp,cdim,nl,scoords));
      PetscCall(PetscFree(scoords));
    }

    /* umfpack interface has a bug when matrix dimension is zero. TODO solve from umfpack interface */
    if (!n_R) {
      PetscCall(KSPGetPC(pcbddc->ksp_R,&pc_temp));
      PetscCall(PCSetType(pc_temp,PCNONE));
    }
    /* Reuse solver if it is present */
    if (reuse_neumann_solver) {
      PCBDDCReuseSolvers reuse_solver = sub_schurs->reuse_solver;

      PetscCall(KSPSetPC(pcbddc->ksp_R,reuse_solver->correction_solver));
    }
    PetscCall(KSPSetUp(pcbddc->ksp_R));
  }

  if (pcbddc->dbg_flag) {
    PetscCall(PetscViewerFlush(pcbddc->dbg_viewer));
    PetscCall(PetscViewerASCIIPushSynchronized(pcbddc->dbg_viewer));
    PetscCall(PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"--------------------------------------------------\n"));
  }
  PetscCall(PetscLogEventEnd(PC_BDDC_LocalSolvers[pcbddc->current_level],pc,0,0,0));

  /* adapt Dirichlet and Neumann solvers if a nullspace correction has been requested */
  if (pcbddc->NullSpace_corr[0]) {
    PetscCall(PCBDDCSetUseExactDirichlet(pc,PETSC_FALSE));
  }
  if (dirichlet && pcbddc->NullSpace_corr[0] && !pcbddc->switch_static) {
    PetscCall(PCBDDCNullSpaceAssembleCorrection(pc,PETSC_TRUE,pcbddc->NullSpace_corr[1]));
  }
  if (neumann && pcbddc->NullSpace_corr[2]) {
    PetscCall(PCBDDCNullSpaceAssembleCorrection(pc,PETSC_FALSE,pcbddc->NullSpace_corr[3]));
  }
  /* check Dirichlet and Neumann solvers */
  if (pcbddc->dbg_flag) {
    if (dirichlet) { /* Dirichlet */
      PetscCall(VecSetRandom(pcis->vec1_D,NULL));
      PetscCall(MatMult(pcis->A_II,pcis->vec1_D,pcis->vec2_D));
      PetscCall(KSPSolve(pcbddc->ksp_D,pcis->vec2_D,pcis->vec2_D));
      PetscCall(KSPCheckSolve(pcbddc->ksp_D,pc,pcis->vec2_D));
      PetscCall(VecAXPY(pcis->vec1_D,m_one,pcis->vec2_D));
      PetscCall(VecNorm(pcis->vec1_D,NORM_INFINITY,&value));
      PetscCall(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d infinity error for Dirichlet solve (%s) = % 1.14e \n",PetscGlobalRank,((PetscObject)(pcbddc->ksp_D))->prefix,(double)value));
      PetscCall(PetscViewerFlush(pcbddc->dbg_viewer));
    }
    if (neumann) { /* Neumann */
      PetscCall(VecSetRandom(pcbddc->vec1_R,NULL));
      PetscCall(MatMult(A_RR,pcbddc->vec1_R,pcbddc->vec2_R));
      PetscCall(KSPSolve(pcbddc->ksp_R,pcbddc->vec2_R,pcbddc->vec2_R));
      PetscCall(KSPCheckSolve(pcbddc->ksp_R,pc,pcbddc->vec2_R));
      PetscCall(VecAXPY(pcbddc->vec1_R,m_one,pcbddc->vec2_R));
      PetscCall(VecNorm(pcbddc->vec1_R,NORM_INFINITY,&value));
      PetscCall(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d infinity error for Neumann solve (%s) = % 1.14e\n",PetscGlobalRank,((PetscObject)(pcbddc->ksp_R))->prefix,(double)value));
      PetscCall(PetscViewerFlush(pcbddc->dbg_viewer));
    }
  }
  /* free Neumann problem's matrix */
  PetscCall(MatDestroy(&A_RR));
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCBDDCSolveSubstructureCorrection(PC pc, Vec inout_B, Vec inout_D, PetscBool applytranspose)
{
  PC_BDDC*        pcbddc = (PC_BDDC*)(pc->data);
  PCBDDCSubSchurs sub_schurs = pcbddc->sub_schurs;
  PetscBool       reuse_solver = sub_schurs ? ( sub_schurs->reuse_solver ? PETSC_TRUE : PETSC_FALSE) : PETSC_FALSE;

  PetscFunctionBegin;
  if (!reuse_solver) {
    PetscCall(VecSet(pcbddc->vec1_R,0.));
  }
  if (!pcbddc->switch_static) {
    if (applytranspose && pcbddc->local_auxmat1) {
      PetscCall(MatMultTranspose(pcbddc->local_auxmat2,inout_B,pcbddc->vec1_C));
      PetscCall(MatMultTransposeAdd(pcbddc->local_auxmat1,pcbddc->vec1_C,inout_B,inout_B));
    }
    if (!reuse_solver) {
      PetscCall(VecScatterBegin(pcbddc->R_to_B,inout_B,pcbddc->vec1_R,INSERT_VALUES,SCATTER_REVERSE));
      PetscCall(VecScatterEnd(pcbddc->R_to_B,inout_B,pcbddc->vec1_R,INSERT_VALUES,SCATTER_REVERSE));
    } else {
      PCBDDCReuseSolvers reuse_solver = sub_schurs->reuse_solver;

      PetscCall(VecScatterBegin(reuse_solver->correction_scatter_B,inout_B,reuse_solver->rhs_B,INSERT_VALUES,SCATTER_FORWARD));
      PetscCall(VecScatterEnd(reuse_solver->correction_scatter_B,inout_B,reuse_solver->rhs_B,INSERT_VALUES,SCATTER_FORWARD));
    }
  } else {
    PetscCall(VecScatterBegin(pcbddc->R_to_B,inout_B,pcbddc->vec1_R,INSERT_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterEnd(pcbddc->R_to_B,inout_B,pcbddc->vec1_R,INSERT_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterBegin(pcbddc->R_to_D,inout_D,pcbddc->vec1_R,INSERT_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterEnd(pcbddc->R_to_D,inout_D,pcbddc->vec1_R,INSERT_VALUES,SCATTER_REVERSE));
    if (applytranspose && pcbddc->local_auxmat1) {
      PetscCall(MatMultTranspose(pcbddc->local_auxmat2,pcbddc->vec1_R,pcbddc->vec1_C));
      PetscCall(MatMultTransposeAdd(pcbddc->local_auxmat1,pcbddc->vec1_C,inout_B,inout_B));
      PetscCall(VecScatterBegin(pcbddc->R_to_B,inout_B,pcbddc->vec1_R,INSERT_VALUES,SCATTER_REVERSE));
      PetscCall(VecScatterEnd(pcbddc->R_to_B,inout_B,pcbddc->vec1_R,INSERT_VALUES,SCATTER_REVERSE));
    }
  }
  PetscCall(PetscLogEventBegin(PC_BDDC_Solves[pcbddc->current_level][1],pc,0,0,0));
  if (!reuse_solver || pcbddc->switch_static) {
    if (applytranspose) {
      PetscCall(KSPSolveTranspose(pcbddc->ksp_R,pcbddc->vec1_R,pcbddc->vec1_R));
    } else {
      PetscCall(KSPSolve(pcbddc->ksp_R,pcbddc->vec1_R,pcbddc->vec1_R));
    }
    PetscCall(KSPCheckSolve(pcbddc->ksp_R,pc,pcbddc->vec1_R));
  } else {
    PCBDDCReuseSolvers reuse_solver = sub_schurs->reuse_solver;

    if (applytranspose) {
      PetscCall(MatFactorSolveSchurComplementTranspose(reuse_solver->F,reuse_solver->rhs_B,reuse_solver->sol_B));
    } else {
      PetscCall(MatFactorSolveSchurComplement(reuse_solver->F,reuse_solver->rhs_B,reuse_solver->sol_B));
    }
  }
  PetscCall(PetscLogEventEnd(PC_BDDC_Solves[pcbddc->current_level][1],pc,0,0,0));
  PetscCall(VecSet(inout_B,0.));
  if (!pcbddc->switch_static) {
    if (!reuse_solver) {
      PetscCall(VecScatterBegin(pcbddc->R_to_B,pcbddc->vec1_R,inout_B,INSERT_VALUES,SCATTER_FORWARD));
      PetscCall(VecScatterEnd(pcbddc->R_to_B,pcbddc->vec1_R,inout_B,INSERT_VALUES,SCATTER_FORWARD));
    } else {
      PCBDDCReuseSolvers reuse_solver = sub_schurs->reuse_solver;

      PetscCall(VecScatterBegin(reuse_solver->correction_scatter_B,reuse_solver->sol_B,inout_B,INSERT_VALUES,SCATTER_REVERSE));
      PetscCall(VecScatterEnd(reuse_solver->correction_scatter_B,reuse_solver->sol_B,inout_B,INSERT_VALUES,SCATTER_REVERSE));
    }
    if (!applytranspose && pcbddc->local_auxmat1) {
      PetscCall(MatMult(pcbddc->local_auxmat1,inout_B,pcbddc->vec1_C));
      PetscCall(MatMultAdd(pcbddc->local_auxmat2,pcbddc->vec1_C,inout_B,inout_B));
    }
  } else {
    PetscCall(VecScatterBegin(pcbddc->R_to_B,pcbddc->vec1_R,inout_B,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(pcbddc->R_to_B,pcbddc->vec1_R,inout_B,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterBegin(pcbddc->R_to_D,pcbddc->vec1_R,inout_D,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(pcbddc->R_to_D,pcbddc->vec1_R,inout_D,INSERT_VALUES,SCATTER_FORWARD));
    if (!applytranspose && pcbddc->local_auxmat1) {
      PetscCall(MatMult(pcbddc->local_auxmat1,inout_B,pcbddc->vec1_C));
      PetscCall(MatMultAdd(pcbddc->local_auxmat2,pcbddc->vec1_C,pcbddc->vec1_R,pcbddc->vec1_R));
    }
    PetscCall(VecScatterBegin(pcbddc->R_to_B,pcbddc->vec1_R,inout_B,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(pcbddc->R_to_B,pcbddc->vec1_R,inout_B,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterBegin(pcbddc->R_to_D,pcbddc->vec1_R,inout_D,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(pcbddc->R_to_D,pcbddc->vec1_R,inout_D,INSERT_VALUES,SCATTER_FORWARD));
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
      PetscCall(MatMultTranspose(pcbddc->coarse_phi_B,pcis->vec1_B,pcbddc->vec1_P));
      if (pcbddc->switch_static) PetscCall(MatMultTransposeAdd(pcbddc->coarse_phi_D,pcis->vec1_D,pcbddc->vec1_P,pcbddc->vec1_P));
    } else {
      PetscCall(MatMultTranspose(pcbddc->coarse_psi_B,pcis->vec1_B,pcbddc->vec1_P));
      if (pcbddc->switch_static) PetscCall(MatMultTransposeAdd(pcbddc->coarse_psi_D,pcis->vec1_D,pcbddc->vec1_P,pcbddc->vec1_P));
    }
  } else {
    PetscCall(VecSet(pcbddc->vec1_P,zero));
  }

  /* add p0 to the last value of vec1_P holding the coarse dof relative to p0 */
  if (pcbddc->benign_n) {
    PetscScalar *array;
    PetscInt    j;

    PetscCall(VecGetArray(pcbddc->vec1_P,&array));
    for (j=0;j<pcbddc->benign_n;j++) array[pcbddc->local_primal_size-pcbddc->benign_n+j] += pcbddc->benign_p0[j];
    PetscCall(VecRestoreArray(pcbddc->vec1_P,&array));
  }

  /* start communications from local primal nodes to rhs of coarse solver */
  PetscCall(VecSet(pcbddc->coarse_vec,zero));
  PetscCall(PCBDDCScatterCoarseDataBegin(pc,ADD_VALUES,SCATTER_FORWARD));
  PetscCall(PCBDDCScatterCoarseDataEnd(pc,ADD_VALUES,SCATTER_FORWARD));

  /* Coarse solution -> rhs and sol updated inside PCBDDCScattarCoarseDataBegin/End */
  if (pcbddc->coarse_ksp) {
    Mat          coarse_mat;
    Vec          rhs,sol;
    MatNullSpace nullsp;
    PetscBool    isbddc = PETSC_FALSE;

    if (pcbddc->benign_have_null) {
      PC        coarse_pc;

      PetscCall(KSPGetPC(pcbddc->coarse_ksp,&coarse_pc));
      PetscCall(PetscObjectTypeCompare((PetscObject)coarse_pc,PCBDDC,&isbddc));
      /* we need to propagate to coarser levels the need for a possible benign correction */
      if (isbddc && pcbddc->benign_apply_coarse_only && !pcbddc->benign_skip_correction) {
        PC_BDDC* coarsepcbddc = (PC_BDDC*)(coarse_pc->data);
        coarsepcbddc->benign_skip_correction = PETSC_FALSE;
        coarsepcbddc->benign_apply_coarse_only = PETSC_TRUE;
      }
    }
    PetscCall(KSPGetRhs(pcbddc->coarse_ksp,&rhs));
    PetscCall(KSPGetSolution(pcbddc->coarse_ksp,&sol));
    PetscCall(KSPGetOperators(pcbddc->coarse_ksp,&coarse_mat,NULL));
    if (applytranspose) {
      PetscCheck(!pcbddc->benign_apply_coarse_only,PetscObjectComm((PetscObject)pcbddc->coarse_ksp),PETSC_ERR_SUP,"Not yet implemented");
      PetscCall(PetscLogEventBegin(PC_BDDC_Solves[pcbddc->current_level][2],pc,0,0,0));
      PetscCall(KSPSolveTranspose(pcbddc->coarse_ksp,rhs,sol));
      PetscCall(PetscLogEventEnd(PC_BDDC_Solves[pcbddc->current_level][2],pc,0,0,0));
      PetscCall(KSPCheckSolve(pcbddc->coarse_ksp,pc,sol));
      PetscCall(MatGetTransposeNullSpace(coarse_mat,&nullsp));
      if (nullsp) {
        PetscCall(MatNullSpaceRemove(nullsp,sol));
      }
    } else {
      PetscCall(MatGetNullSpace(coarse_mat,&nullsp));
      if (pcbddc->benign_apply_coarse_only && isbddc) { /* need just to apply the coarse preconditioner during presolve */
        PC        coarse_pc;

        if (nullsp) {
          PetscCall(MatNullSpaceRemove(nullsp,rhs));
        }
        PetscCall(KSPGetPC(pcbddc->coarse_ksp,&coarse_pc));
        PetscCall(PCPreSolve(coarse_pc,pcbddc->coarse_ksp));
        PetscCall(PCBDDCBenignRemoveInterior(coarse_pc,rhs,sol));
        PetscCall(PCPostSolve(coarse_pc,pcbddc->coarse_ksp));
      } else {
        PetscCall(PetscLogEventBegin(PC_BDDC_Solves[pcbddc->current_level][2],pc,0,0,0));
        PetscCall(KSPSolve(pcbddc->coarse_ksp,rhs,sol));
        PetscCall(PetscLogEventEnd(PC_BDDC_Solves[pcbddc->current_level][2],pc,0,0,0));
        PetscCall(KSPCheckSolve(pcbddc->coarse_ksp,pc,sol));
        if (nullsp) {
          PetscCall(MatNullSpaceRemove(nullsp,sol));
        }
      }
    }
    /* we don't need the benign correction at coarser levels anymore */
    if (pcbddc->benign_have_null && isbddc) {
      PC        coarse_pc;
      PC_BDDC*  coarsepcbddc;

      PetscCall(KSPGetPC(pcbddc->coarse_ksp,&coarse_pc));
      coarsepcbddc = (PC_BDDC*)(coarse_pc->data);
      coarsepcbddc->benign_skip_correction = PETSC_TRUE;
      coarsepcbddc->benign_apply_coarse_only = PETSC_FALSE;
    }
  }

  /* Local solution on R nodes */
  if (pcis->n && !pcbddc->benign_apply_coarse_only) {
    PetscCall(PCBDDCSolveSubstructureCorrection(pc,pcis->vec1_B,pcis->vec1_D,applytranspose));
  }
  /* communications from coarse sol to local primal nodes */
  PetscCall(PCBDDCScatterCoarseDataBegin(pc,INSERT_VALUES,SCATTER_REVERSE));
  PetscCall(PCBDDCScatterCoarseDataEnd(pc,INSERT_VALUES,SCATTER_REVERSE));

  /* Sum contributions from the two levels */
  if (!pcbddc->benign_apply_coarse_only) {
    if (applytranspose) {
      PetscCall(MatMultAdd(pcbddc->coarse_psi_B,pcbddc->vec1_P,pcis->vec1_B,pcis->vec1_B));
      if (pcbddc->switch_static) PetscCall(MatMultAdd(pcbddc->coarse_psi_D,pcbddc->vec1_P,pcis->vec1_D,pcis->vec1_D));
    } else {
      PetscCall(MatMultAdd(pcbddc->coarse_phi_B,pcbddc->vec1_P,pcis->vec1_B,pcis->vec1_B));
      if (pcbddc->switch_static) PetscCall(MatMultAdd(pcbddc->coarse_phi_D,pcbddc->vec1_P,pcis->vec1_D,pcis->vec1_D));
    }
    /* store p0 */
    if (pcbddc->benign_n) {
      PetscScalar *array;
      PetscInt    j;

      PetscCall(VecGetArray(pcbddc->vec1_P,&array));
      for (j=0;j<pcbddc->benign_n;j++) pcbddc->benign_p0[j] = array[pcbddc->local_primal_size-pcbddc->benign_n+j];
      PetscCall(VecRestoreArray(pcbddc->vec1_P,&array));
    }
  } else { /* expand the coarse solution */
    if (applytranspose) {
      PetscCall(MatMult(pcbddc->coarse_psi_B,pcbddc->vec1_P,pcis->vec1_B));
    } else {
      PetscCall(MatMult(pcbddc->coarse_phi_B,pcbddc->vec1_P,pcis->vec1_B));
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

      PetscCall(KSPGetRhs(pcbddc->coarse_ksp,&tvec));
      PetscCall(VecResetArray(tvec));
      PetscCall(KSPGetSolution(pcbddc->coarse_ksp,&tvec));
      PetscCall(VecGetArrayRead(tvec,&array));
      PetscCall(VecPlaceArray(from,array));
      PetscCall(VecRestoreArrayRead(tvec,&array));
    }
  } else { /* from local to global -> put data in coarse right hand side */
    from = pcbddc->vec1_P;
    to = pcbddc->coarse_vec;
  }
  PetscCall(VecScatterBegin(pcbddc->coarse_loc_to_glob,from,to,imode,smode));
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
  PetscCall(VecScatterEnd(pcbddc->coarse_loc_to_glob,from,to,imode,smode));
  if (smode == SCATTER_FORWARD) {
    if (pcbddc->coarse_ksp) { /* get array from coarse processes */
      Vec tvec;

      PetscCall(KSPGetRhs(pcbddc->coarse_ksp,&tvec));
      PetscCall(VecGetArrayRead(to,&array));
      PetscCall(VecPlaceArray(tvec,array));
      PetscCall(VecRestoreArrayRead(to,&array));
    }
  } else {
    if (pcbddc->coarse_ksp) { /* restore array of pcbddc->coarse_vec */
     PetscCall(VecResetArray(from));
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCConstraintsSetUp(PC pc)
{
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
  PetscCall(MatDestroy(&pcbddc->ChangeOfBasisMatrix));
  PetscCall(MatDestroy(&pcbddc->ConstraintMatrix));
  PetscCall(MatDestroy(&pcbddc->switch_static_change));
  /* save info on constraints from previous setup (if any) */
  olocal_primal_size = pcbddc->local_primal_size;
  olocal_primal_size_cc = pcbddc->local_primal_size_cc;
  PetscCall(PetscMalloc2(olocal_primal_size_cc,&olocal_primal_ref_node,olocal_primal_size_cc,&olocal_primal_ref_mult));
  PetscCall(PetscArraycpy(olocal_primal_ref_node,pcbddc->local_primal_ref_node,olocal_primal_size_cc));
  PetscCall(PetscArraycpy(olocal_primal_ref_mult,pcbddc->local_primal_ref_mult,olocal_primal_size_cc));
  PetscCall(PetscFree2(pcbddc->local_primal_ref_node,pcbddc->local_primal_ref_mult));
  PetscCall(PetscFree(pcbddc->primal_indices_local_idxs));

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
    PetscCall(PCBDDCGraphGetCandidatesIS(pcbddc->mat_graph,&n_ISForFaces,&ISForFaces,&n_ISForEdges,&ISForEdges,&ISForVertices));
    /* print some info */
    if (pcbddc->dbg_flag && (!pcbddc->sub_schurs || pcbddc->sub_schurs_rebuild)) {
      PetscInt nv;

      PetscCall(PCBDDCGraphASCIIView(pcbddc->mat_graph,pcbddc->dbg_flag,pcbddc->dbg_viewer));
      PetscCall(ISGetSize(ISForVertices,&nv));
      PetscCall(PetscViewerASCIIPushSynchronized(pcbddc->dbg_viewer));
      PetscCall(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"--------------------------------------------------------------\n"));
      PetscCall(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d got %02" PetscInt_FMT " local candidate vertices (%d)\n",PetscGlobalRank,nv,pcbddc->use_vertices));
      PetscCall(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d got %02" PetscInt_FMT " local candidate edges    (%d)\n",PetscGlobalRank,n_ISForEdges,pcbddc->use_edges));
      PetscCall(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d got %02" PetscInt_FMT " local candidate faces    (%d)\n",PetscGlobalRank,n_ISForFaces,pcbddc->use_faces));
      PetscCall(PetscViewerFlush(pcbddc->dbg_viewer));
      PetscCall(PetscViewerASCIIPopSynchronized(pcbddc->dbg_viewer));
    }

    /* free unneeded index sets */
    if (!pcbddc->use_vertices) {
      PetscCall(ISDestroy(&ISForVertices));
    }
    if (!pcbddc->use_edges) {
      for (i=0;i<n_ISForEdges;i++) {
        PetscCall(ISDestroy(&ISForEdges[i]));
      }
      PetscCall(PetscFree(ISForEdges));
      n_ISForEdges = 0;
    }
    if (!pcbddc->use_faces) {
      for (i=0;i<n_ISForFaces;i++) {
        PetscCall(ISDestroy(&ISForFaces[i]));
      }
      PetscCall(PetscFree(ISForFaces));
      n_ISForFaces = 0;
    }

    /* check if near null space is attached to global mat */
    if (pcbddc->use_nnsp) {
      PetscCall(MatGetNearNullSpace(pc->pmat,&nearnullsp));
    } else nearnullsp = NULL;

    if (nearnullsp) {
      PetscCall(MatNullSpaceGetVecs(nearnullsp,&nnsp_has_cnst,&nnsp_size,&nearnullvecs));
      /* remove any stored info */
      PetscCall(MatNullSpaceDestroy(&pcbddc->onearnullspace));
      PetscCall(PetscFree(pcbddc->onearnullvecs_state));
      /* store information for BDDC solver reuse */
      PetscCall(PetscObjectReference((PetscObject)nearnullsp));
      pcbddc->onearnullspace = nearnullsp;
      PetscCall(PetscMalloc1(nnsp_size,&pcbddc->onearnullvecs_state));
      for (i=0;i<nnsp_size;i++) {
        PetscCall(PetscObjectStateGet((PetscObject)nearnullvecs[i],&pcbddc->onearnullvecs_state[i]));
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
      PetscCall(ISGetSize(ISForVertices,&n_vertices));
    }
    ncc = n_vertices+n_ISForFaces+n_ISForEdges;
    PetscCall(PetscMalloc3(ncc+1,&constraints_idxs_ptr,ncc+1,&constraints_data_ptr,ncc,&constraints_n));

    total_counts = n_ISForFaces+n_ISForEdges;
    total_counts *= max_constraints;
    total_counts += n_vertices;
    PetscCall(PetscBTCreate(total_counts,&change_basis));

    total_counts = 0;
    max_size_of_constraint = 0;
    for (i=0;i<n_ISForEdges+n_ISForFaces;i++) {
      IS used_is;
      if (i<n_ISForEdges) {
        used_is = ISForEdges[i];
      } else {
        used_is = ISForFaces[i-n_ISForEdges];
      }
      PetscCall(ISGetSize(used_is,&j));
      total_counts += j;
      max_size_of_constraint = PetscMax(j,max_size_of_constraint);
    }
    PetscCall(PetscMalloc3(total_counts*max_constraints+n_vertices,&constraints_data,total_counts+n_vertices,&constraints_idxs,total_counts+n_vertices,&constraints_idxs_B));

    /* get local part of global near null space vectors */
    PetscCall(PetscMalloc1(nnsp_size,&localnearnullsp));
    for (k=0;k<nnsp_size;k++) {
      PetscCall(VecDuplicate(pcis->vec1_N,&localnearnullsp[k]));
      PetscCall(VecScatterBegin(matis->rctx,nearnullvecs[k],localnearnullsp[k],INSERT_VALUES,SCATTER_FORWARD));
      PetscCall(VecScatterEnd(matis->rctx,nearnullvecs[k],localnearnullsp[k],INSERT_VALUES,SCATTER_FORWARD));
    }

    /* whether or not to skip lapack calls */
    skip_lapack = PETSC_TRUE;
    if (n_ISForFaces+n_ISForEdges && max_constraints > 1 && !pcbddc->use_nnsp_true) skip_lapack = PETSC_FALSE;

    /* First we issue queries to allocate optimal workspace for LAPACKgesvd (or LAPACKsyev if SVD is missing) */
    if (!skip_lapack) {
      PetscScalar temp_work;

      if (use_pod) {
        /* Proper Orthogonal Decomposition (POD) using the snapshot method */
        PetscCall(PetscMalloc1(max_constraints*max_constraints,&correlation_mat));
        PetscCall(PetscMalloc1(max_constraints,&singular_vals));
        PetscCall(PetscMalloc1(max_size_of_constraint*max_constraints,&temp_basis));
#if defined(PETSC_USE_COMPLEX)
        PetscCall(PetscMalloc1(3*max_constraints,&rwork));
#endif
        /* now we evaluate the optimal workspace using query with lwork=-1 */
        PetscCall(PetscBLASIntCast(max_constraints,&Blas_N));
        PetscCall(PetscBLASIntCast(max_constraints,&Blas_LDA));
        lwork = -1;
        PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
#if !defined(PETSC_USE_COMPLEX)
        PetscStackCallBLAS("LAPACKsyev",LAPACKsyev_("V","U",&Blas_N,correlation_mat,&Blas_LDA,singular_vals,&temp_work,&lwork,&lierr));
#else
        PetscStackCallBLAS("LAPACKsyev",LAPACKsyev_("V","U",&Blas_N,correlation_mat,&Blas_LDA,singular_vals,&temp_work,&lwork,rwork,&lierr));
#endif
        PetscCall(PetscFPTrapPop());
        PetscCheck(!lierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in query to SYEV Lapack routine %d",(int)lierr);
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
        PetscCall(PetscMalloc1(min_n,&singular_vals));
#if defined(PETSC_USE_COMPLEX)
        PetscCall(PetscMalloc1(5*min_n,&rwork));
#endif
        /* now we evaluate the optimal workspace using query with lwork=-1 */
        lwork = -1;
        PetscCall(PetscBLASIntCast(max_n,&Blas_M));
        PetscCall(PetscBLASIntCast(min_n,&Blas_N));
        PetscCall(PetscBLASIntCast(max_n,&Blas_LDA));
        PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
#if !defined(PETSC_USE_COMPLEX)
        PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("O","N",&Blas_M,&Blas_N,&constraints_data[0],&Blas_LDA,singular_vals,&dummy_scalar,&dummy_int,&dummy_scalar,&dummy_int,&temp_work,&lwork,&lierr));
#else
        PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("O","N",&Blas_M,&Blas_N,&constraints_data[0],&Blas_LDA,singular_vals,&dummy_scalar,&dummy_int,&dummy_scalar,&dummy_int,&temp_work,&lwork,rwork,&lierr));
#endif
        PetscCall(PetscFPTrapPop());
        PetscCheck(!lierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in query to GESVD Lapack routine %d",(int)lierr);
#else
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"This should not happen");
#endif /* on missing GESVD */
      }
      /* Allocate optimal workspace */
      PetscCall(PetscBLASIntCast((PetscInt)PetscRealPart(temp_work),&lwork));
      PetscCall(PetscMalloc1(lwork,&work));
    }
    /* Now we can loop on constraining sets */
    total_counts = 0;
    constraints_idxs_ptr[0] = 0;
    constraints_data_ptr[0] = 0;
    /* vertices */
    if (n_vertices) {
      PetscCall(ISGetIndices(ISForVertices,(const PetscInt**)&is_indices));
      PetscCall(PetscArraycpy(constraints_idxs,is_indices,n_vertices));
      for (i=0;i<n_vertices;i++) {
        constraints_n[total_counts] = 1;
        constraints_data[total_counts] = 1.0;
        constraints_idxs_ptr[total_counts+1] = constraints_idxs_ptr[total_counts]+1;
        constraints_data_ptr[total_counts+1] = constraints_data_ptr[total_counts]+1;
        total_counts++;
      }
      PetscCall(ISRestoreIndices(ISForVertices,(const PetscInt**)&is_indices));
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

      PetscCall(ISGetSize(used_is,&size_of_constraint));
      PetscCall(ISGetIndices(used_is,(const PetscInt**)&is_indices));
      /* change of basis should not be performed on local periodic nodes */
      if (pcbddc->mat_graph->mirrors && pcbddc->mat_graph->mirrors[is_indices[0]]) boolforchange = PETSC_FALSE;
      if (nnsp_has_cnst) {
        PetscScalar quad_value;

        PetscCall(PetscArraycpy(constraints_idxs + constraints_idxs_ptr[total_counts_cc],is_indices,size_of_constraint));
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

        PetscCall(VecGetArrayRead(localnearnullsp[k],(const PetscScalar**)&array));
        ptr_to_data = &constraints_data[constraints_data_ptr[total_counts_cc]+temp_constraints*size_of_constraint];
        for (j=0;j<size_of_constraint;j++) {
          ptr_to_data[j] = array[is_indices[j]];
        }
        PetscCall(VecRestoreArrayRead(localnearnullsp[k],(const PetscScalar**)&array));
        /* check if array is null on the connected component */
        PetscCall(PetscBLASIntCast(size_of_constraint,&Blas_N));
        PetscStackCallBLAS("BLASasum",real_value = BLASasum_(&Blas_N,ptr_to_data,&Blas_one));
        if (real_value > tol*size_of_constraint) { /* keep indices and values */
          temp_constraints++;
          total_counts++;
          if (!idxs_copied) {
            PetscCall(PetscArraycpy(constraints_idxs + constraints_idxs_ptr[total_counts_cc],is_indices,size_of_constraint));
            idxs_copied = PETSC_TRUE;
          }
        }
      }
      PetscCall(ISRestoreIndices(used_is,(const PetscInt**)&is_indices));
      valid_constraints = temp_constraints;
      if (!pcbddc->use_nnsp_true && temp_constraints) {
        if (temp_constraints == 1) { /* just normalize the constraint */
          PetscScalar norm,*ptr_to_data;

          ptr_to_data = &constraints_data[constraints_data_ptr[total_counts_cc]];
          PetscCall(PetscBLASIntCast(size_of_constraint,&Blas_N));
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
            PetscCall(PetscArrayzero(correlation_mat,temp_constraints*temp_constraints));
            /* Store upper triangular part of correlation matrix */
            PetscCall(PetscBLASIntCast(size_of_constraint,&Blas_N));
            PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
            for (j=0;j<temp_constraints;j++) {
              for (k=0;k<j+1;k++) {
                PetscStackCallBLAS("BLASdot",correlation_mat[j*temp_constraints+k] = BLASdot_(&Blas_N,ptr_to_data+k*size_of_constraint,&Blas_one,ptr_to_data+j*size_of_constraint,&Blas_one));
              }
            }
            /* compute eigenvalues and eigenvectors of correlation matrix */
            PetscCall(PetscBLASIntCast(temp_constraints,&Blas_N));
            PetscCall(PetscBLASIntCast(temp_constraints,&Blas_LDA));
#if !defined(PETSC_USE_COMPLEX)
            PetscStackCallBLAS("LAPACKsyev",LAPACKsyev_("V","U",&Blas_N,correlation_mat,&Blas_LDA,singular_vals,work,&lwork,&lierr));
#else
            PetscStackCallBLAS("LAPACKsyev",LAPACKsyev_("V","U",&Blas_N,correlation_mat,&Blas_LDA,singular_vals,work,&lwork,rwork,&lierr));
#endif
            PetscCall(PetscFPTrapPop());
            PetscCheck(!lierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in SYEV Lapack routine %d",(int)lierr);
            /* retain eigenvalues greater than tol: note that LAPACKsyev gives eigs in ascending order */
            j = 0;
            while (j < temp_constraints && singular_vals[j]/singular_vals[temp_constraints-1] < tol) j++;
            total_counts = total_counts-j;
            valid_constraints = temp_constraints-j;
            /* scale and copy POD basis into used quadrature memory */
            PetscCall(PetscBLASIntCast(size_of_constraint,&Blas_M));
            PetscCall(PetscBLASIntCast(temp_constraints,&Blas_N));
            PetscCall(PetscBLASIntCast(temp_constraints,&Blas_K));
            PetscCall(PetscBLASIntCast(size_of_constraint,&Blas_LDA));
            PetscCall(PetscBLASIntCast(temp_constraints,&Blas_LDB));
            PetscCall(PetscBLASIntCast(size_of_constraint,&Blas_LDC));
            if (j<temp_constraints) {
              PetscInt ii;
              for (k=j;k<temp_constraints;k++) singular_vals[k] = 1.0/PetscSqrtReal(singular_vals[k]);
              PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
              PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&Blas_M,&Blas_N,&Blas_K,&one,ptr_to_data,&Blas_LDA,correlation_mat,&Blas_LDB,&zero,temp_basis,&Blas_LDC));
              PetscCall(PetscFPTrapPop());
              for (k=0;k<temp_constraints-j;k++) {
                for (ii=0;ii<size_of_constraint;ii++) {
                  ptr_to_data[k*size_of_constraint+ii] = singular_vals[temp_constraints-1-k]*temp_basis[(temp_constraints-1-k)*size_of_constraint+ii];
                }
              }
            }
          } else {
#if !defined(PETSC_MISSING_LAPACK_GESVD)
            PetscCall(PetscBLASIntCast(size_of_constraint,&Blas_M));
            PetscCall(PetscBLASIntCast(temp_constraints,&Blas_N));
            PetscCall(PetscBLASIntCast(size_of_constraint,&Blas_LDA));
            PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
#if !defined(PETSC_USE_COMPLEX)
            PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("O","N",&Blas_M,&Blas_N,ptr_to_data,&Blas_LDA,singular_vals,&dummy_scalar,&dummy_int,&dummy_scalar,&dummy_int,work,&lwork,&lierr));
#else
            PetscStackCallBLAS("LAPACKgesvd",LAPACKgesvd_("O","N",&Blas_M,&Blas_N,ptr_to_data,&Blas_LDA,singular_vals,&dummy_scalar,&dummy_int,&dummy_scalar,&dummy_int,work,&lwork,rwork,&lierr));
#endif
            PetscCheck(!lierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in GESVD Lapack routine %d",(int)lierr);
            PetscCall(PetscFPTrapPop());
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
      PetscCall(PetscFree(work));
#if defined(PETSC_USE_COMPLEX)
      PetscCall(PetscFree(rwork));
#endif
      PetscCall(PetscFree(singular_vals));
      PetscCall(PetscFree(correlation_mat));
      PetscCall(PetscFree(temp_basis));
    }
    for (k=0;k<nnsp_size;k++) {
      PetscCall(VecDestroy(&localnearnullsp[k]));
    }
    PetscCall(PetscFree(localnearnullsp));
    /* free index sets of faces, edges and vertices */
    for (i=0;i<n_ISForFaces;i++) {
      PetscCall(ISDestroy(&ISForFaces[i]));
    }
    if (n_ISForFaces) {
      PetscCall(PetscFree(ISForFaces));
    }
    for (i=0;i<n_ISForEdges;i++) {
      PetscCall(ISDestroy(&ISForEdges[i]));
    }
    if (n_ISForEdges) {
      PetscCall(PetscFree(ISForEdges));
    }
    PetscCall(ISDestroy(&ISForVertices));
  } else {
    PCBDDCSubSchurs sub_schurs = pcbddc->sub_schurs;

    total_counts = 0;
    n_vertices = 0;
    if (sub_schurs->is_vertices && pcbddc->use_vertices) {
      PetscCall(ISGetLocalSize(sub_schurs->is_vertices,&n_vertices));
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
    PetscCall(PetscMalloc1(total_counts_cc,&constraints_n));
    total_counts_cc = 0;
    for (i=0;i<sub_schurs->n_subs+n_vertices;i++) {
      if (pcbddc->adaptive_constraints_n[i]) {
        constraints_n[total_counts_cc++] = pcbddc->adaptive_constraints_n[i];
      }
    }

    max_size_of_constraint = 0;
    for (i=0;i<total_counts_cc;i++) max_size_of_constraint = PetscMax(max_size_of_constraint,constraints_idxs_ptr[i+1]-constraints_idxs_ptr[i]);
    PetscCall(PetscMalloc1(constraints_idxs_ptr[total_counts_cc],&constraints_idxs_B));
    /* Change of basis */
    PetscCall(PetscBTCreate(total_counts_cc,&change_basis));
    if (pcbddc->use_change_of_basis) {
      for (i=0;i<sub_schurs->n_subs;i++) {
        if (PetscBTLookup(sub_schurs->is_edge,i) || pcbddc->use_change_on_faces) {
          PetscCall(PetscBTSet(change_basis,i+n_vertices));
        }
      }
    }
  }
  pcbddc->local_primal_size = total_counts;
  PetscCall(PetscMalloc1(pcbddc->local_primal_size+pcbddc->benign_n,&pcbddc->primal_indices_local_idxs));

  /* map constraints_idxs in boundary numbering */
  PetscCall(ISGlobalToLocalMappingApply(pcis->BtoNmap,IS_GTOLM_DROP,constraints_idxs_ptr[total_counts_cc],constraints_idxs,&i,constraints_idxs_B));
  PetscCheck(i == constraints_idxs_ptr[total_counts_cc],PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error in boundary numbering for constraints indices %" PetscInt_FMT " != %" PetscInt_FMT,constraints_idxs_ptr[total_counts_cc],i);

  /* Create constraint matrix */
  PetscCall(MatCreate(PETSC_COMM_SELF,&pcbddc->ConstraintMatrix));
  PetscCall(MatSetType(pcbddc->ConstraintMatrix,MATAIJ));
  PetscCall(MatSetSizes(pcbddc->ConstraintMatrix,pcbddc->local_primal_size,pcis->n,pcbddc->local_primal_size,pcis->n));

  /* find primal_dofs: subdomain corners plus dofs selected as primal after change of basis */
  /* determine if a QR strategy is needed for change of basis */
  qr_needed = pcbddc->use_qr_single;
  PetscCall(PetscBTCreate(total_counts_cc,&qr_needed_idx));
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
  PetscCall(PetscSortInt(total_primal_vertices,pcbddc->primal_indices_local_idxs));
  PetscCall(PetscMalloc2(pcbddc->local_primal_size_cc+pcbddc->benign_n,&pcbddc->local_primal_ref_node,pcbddc->local_primal_size_cc+pcbddc->benign_n,&pcbddc->local_primal_ref_mult));
  PetscCall(PetscArraycpy(pcbddc->local_primal_ref_node,pcbddc->primal_indices_local_idxs,total_primal_vertices));
  for (i=0;i<total_primal_vertices;i++) pcbddc->local_primal_ref_mult[i] = 1;

  /* nonzero structure of constraint matrix */
  /* and get reference dof for local constraints */
  PetscCall(PetscMalloc1(pcbddc->local_primal_size,&nnz));
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
  PetscCall(MatSeqAIJSetPreallocation(pcbddc->ConstraintMatrix,0,nnz));
  PetscCall(MatSetOption(pcbddc->ConstraintMatrix,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE));
  PetscCall(PetscFree(nnz));

  /* set values in constraint matrix */
  for (i=0;i<total_primal_vertices;i++) {
    PetscCall(MatSetValue(pcbddc->ConstraintMatrix,i,pcbddc->local_primal_ref_node[i],1.0,INSERT_VALUES));
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
        PetscCall(MatSetValues(pcbddc->ConstraintMatrix,1,&row,size_of_constraint,cols,vals,INSERT_VALUES));
      }
      total_counts += constraints_n[i];
    }
  }
  /* assembling */
  PetscCall(MatAssemblyBegin(pcbddc->ConstraintMatrix,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(pcbddc->ConstraintMatrix,MAT_FINAL_ASSEMBLY));
  PetscCall(MatViewFromOptions(pcbddc->ConstraintMatrix,(PetscObject)pc,"-pc_bddc_constraint_mat_view"));

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
    PetscCall(MatCreate(PETSC_COMM_SELF,&localChangeOfBasisMatrix));
    PetscCall(MatSetType(localChangeOfBasisMatrix,MATAIJ));
    PetscCall(MatSetSizes(localChangeOfBasisMatrix,pcis->n,pcis->n,pcis->n,pcis->n));
    /* nonzeros for local mat */
    PetscCall(PetscMalloc1(pcis->n,&nnz));
    if (!pcbddc->benign_change || pcbddc->fake_change) {
      for (i=0;i<pcis->n;i++) nnz[i]=1;
    } else {
      const PetscInt *ii;
      PetscInt       n;
      PetscBool      flg_row;
      PetscCall(MatGetRowIJ(pcbddc->benign_change,0,PETSC_FALSE,PETSC_FALSE,&n,&ii,NULL,&flg_row));
      for (i=0;i<n;i++) nnz[i] = ii[i+1]-ii[i];
      PetscCall(MatRestoreRowIJ(pcbddc->benign_change,0,PETSC_FALSE,PETSC_FALSE,&n,&ii,NULL,&flg_row));
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
    PetscCall(MatSeqAIJSetPreallocation(localChangeOfBasisMatrix,0,nnz));
    PetscCall(MatSetOption(localChangeOfBasisMatrix,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE));
    PetscCall(PetscFree(nnz));
    /* Set interior change in the matrix */
    if (!pcbddc->benign_change || pcbddc->fake_change) {
      for (i=0;i<pcis->n;i++) {
        PetscCall(MatSetValue(localChangeOfBasisMatrix,i,i,1.0,INSERT_VALUES));
      }
    } else {
      const PetscInt *ii,*jj;
      PetscScalar    *aa;
      PetscInt       n;
      PetscBool      flg_row;
      PetscCall(MatGetRowIJ(pcbddc->benign_change,0,PETSC_FALSE,PETSC_FALSE,&n,&ii,&jj,&flg_row));
      PetscCall(MatSeqAIJGetArray(pcbddc->benign_change,&aa));
      for (i=0;i<n;i++) {
        PetscCall(MatSetValues(localChangeOfBasisMatrix,1,&i,ii[i+1]-ii[i],jj+ii[i],aa+ii[i],INSERT_VALUES));
      }
      PetscCall(MatSeqAIJRestoreArray(pcbddc->benign_change,&aa));
      PetscCall(MatRestoreRowIJ(pcbddc->benign_change,0,PETSC_FALSE,PETSC_FALSE,&n,&ii,&jj,&flg_row));
    }

    if (pcbddc->dbg_flag) {
      PetscCall(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"--------------------------------------------------------------\n"));
      PetscCall(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Checking change of basis computation for subdomain %04d\n",PetscGlobalRank));
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
      PetscCall(PetscMalloc1(max_size_of_constraint*max_size_of_constraint,&qr_basis));
      /* array to store scaling factors for reflectors */
      PetscCall(PetscMalloc1(max_constraints,&qr_tau));
      /* first we issue queries for optimal work */
      PetscCall(PetscBLASIntCast(max_size_of_constraint,&Blas_M));
      PetscCall(PetscBLASIntCast(max_constraints,&Blas_N));
      PetscCall(PetscBLASIntCast(max_size_of_constraint,&Blas_LDA));
      lqr_work = -1;
      PetscStackCallBLAS("LAPACKgeqrf",LAPACKgeqrf_(&Blas_M,&Blas_N,qr_basis,&Blas_LDA,qr_tau,&lqr_work_t,&lqr_work,&lierr));
      PetscCheck(!lierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in query to GEQRF Lapack routine %d",(int)lierr);
      PetscCall(PetscBLASIntCast((PetscInt)PetscRealPart(lqr_work_t),&lqr_work));
      PetscCall(PetscMalloc1((PetscInt)PetscRealPart(lqr_work_t),&qr_work));
      lgqr_work = -1;
      PetscCall(PetscBLASIntCast(max_size_of_constraint,&Blas_M));
      PetscCall(PetscBLASIntCast(max_size_of_constraint,&Blas_N));
      PetscCall(PetscBLASIntCast(max_constraints,&Blas_K));
      PetscCall(PetscBLASIntCast(max_size_of_constraint,&Blas_LDA));
      if (Blas_K>Blas_M) Blas_K=Blas_M; /* adjust just for computing optimal work */
      PetscStackCallBLAS("LAPACKorgqr",LAPACKorgqr_(&Blas_M,&Blas_N,&Blas_K,qr_basis,&Blas_LDA,qr_tau,&lgqr_work_t,&lgqr_work,&lierr));
      PetscCheck(!lierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in query to ORGQR/UNGQR Lapack routine %d",(int)lierr);
      PetscCall(PetscBLASIntCast((PetscInt)PetscRealPart(lgqr_work_t),&lgqr_work));
      PetscCall(PetscMalloc1((PetscInt)PetscRealPart(lgqr_work_t),&gqr_work));
      /* array to store rhs and solution of triangular solver */
      PetscCall(PetscMalloc1(max_constraints*max_constraints,&trs_rhs));
      /* allocating workspace for check */
      if (pcbddc->dbg_flag) {
        PetscCall(PetscMalloc1(max_size_of_constraint*(max_constraints+max_size_of_constraint),&dbg_work));
      }
    }
    /* array to store whether a node is primal or not */
    PetscCall(PetscBTCreate(pcis->n_B,&is_primal));
    PetscCall(PetscMalloc1(total_primal_vertices,&aux_primal_numbering_B));
    PetscCall(ISGlobalToLocalMappingApply(pcis->BtoNmap,IS_GTOLM_DROP,total_primal_vertices,pcbddc->local_primal_ref_node,&i,aux_primal_numbering_B));
    PetscCheck(i == total_primal_vertices,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error in boundary numbering for BDDC vertices! %" PetscInt_FMT " != %" PetscInt_FMT,total_primal_vertices,i);
    for (i=0;i<total_primal_vertices;i++) {
      PetscCall(PetscBTSet(is_primal,aux_primal_numbering_B[i]));
    }
    PetscCall(PetscFree(aux_primal_numbering_B));

    /* loop on constraints and see whether or not they need a change of basis and compute it */
    for (total_counts=n_vertices;total_counts<total_counts_cc;total_counts++) {
      size_of_constraint = constraints_idxs_ptr[total_counts+1]-constraints_idxs_ptr[total_counts];
      if (PetscBTLookup(change_basis,total_counts)) {
        /* get constraint info */
        primal_dofs = constraints_n[total_counts];
        dual_dofs = size_of_constraint-primal_dofs;

        if (pcbddc->dbg_flag) {
          PetscCall(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Constraints %" PetscInt_FMT ": %" PetscInt_FMT " need a change of basis (size %" PetscInt_FMT ")\n",total_counts,primal_dofs,size_of_constraint));
        }

        if (PetscBTLookup(qr_needed_idx,total_counts)) { /* QR */

          /* copy quadrature constraints for change of basis check */
          if (pcbddc->dbg_flag) {
            PetscCall(PetscArraycpy(dbg_work,&constraints_data[constraints_data_ptr[total_counts]],size_of_constraint*primal_dofs));
          }
          /* copy temporary constraints into larger work vector (in order to store all columns of Q) */
          PetscCall(PetscArraycpy(qr_basis,&constraints_data[constraints_data_ptr[total_counts]],size_of_constraint*primal_dofs));

          /* compute QR decomposition of constraints */
          PetscCall(PetscBLASIntCast(size_of_constraint,&Blas_M));
          PetscCall(PetscBLASIntCast(primal_dofs,&Blas_N));
          PetscCall(PetscBLASIntCast(size_of_constraint,&Blas_LDA));
          PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
          PetscStackCallBLAS("LAPACKgeqrf",LAPACKgeqrf_(&Blas_M,&Blas_N,qr_basis,&Blas_LDA,qr_tau,qr_work,&lqr_work,&lierr));
          PetscCheck(!lierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in GEQRF Lapack routine %d",(int)lierr);
          PetscCall(PetscFPTrapPop());

          /* explicitly compute R^-T */
          PetscCall(PetscArrayzero(trs_rhs,primal_dofs*primal_dofs));
          for (j=0;j<primal_dofs;j++) trs_rhs[j*(primal_dofs+1)] = 1.0;
          PetscCall(PetscBLASIntCast(primal_dofs,&Blas_N));
          PetscCall(PetscBLASIntCast(primal_dofs,&Blas_NRHS));
          PetscCall(PetscBLASIntCast(size_of_constraint,&Blas_LDA));
          PetscCall(PetscBLASIntCast(primal_dofs,&Blas_LDB));
          PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
          PetscStackCallBLAS("LAPACKtrtrs",LAPACKtrtrs_("U","T","N",&Blas_N,&Blas_NRHS,qr_basis,&Blas_LDA,trs_rhs,&Blas_LDB,&lierr));
          PetscCheck(!lierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in TRTRS Lapack routine %d",(int)lierr);
          PetscCall(PetscFPTrapPop());

          /* explicitly compute all columns of Q (Q = [Q1 | Q2]) overwriting QR factorization in qr_basis */
          PetscCall(PetscBLASIntCast(size_of_constraint,&Blas_M));
          PetscCall(PetscBLASIntCast(size_of_constraint,&Blas_N));
          PetscCall(PetscBLASIntCast(primal_dofs,&Blas_K));
          PetscCall(PetscBLASIntCast(size_of_constraint,&Blas_LDA));
          PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
          PetscStackCallBLAS("LAPACKorgqr",LAPACKorgqr_(&Blas_M,&Blas_N,&Blas_K,qr_basis,&Blas_LDA,qr_tau,gqr_work,&lgqr_work,&lierr));
          PetscCheck(!lierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in ORGQR/UNGQR Lapack routine %d",(int)lierr);
          PetscCall(PetscFPTrapPop());

          /* first primal_dofs columns of Q need to be re-scaled in order to be unitary w.r.t constraints
             i.e. C_{pxn}*Q_{nxn} should be equal to [I_pxp | 0_pxd] (see check below)
             where n=size_of_constraint, p=primal_dofs, d=dual_dofs (n=p+d), I and 0 identity and null matrix resp. */
          PetscCall(PetscBLASIntCast(size_of_constraint,&Blas_M));
          PetscCall(PetscBLASIntCast(primal_dofs,&Blas_N));
          PetscCall(PetscBLASIntCast(primal_dofs,&Blas_K));
          PetscCall(PetscBLASIntCast(size_of_constraint,&Blas_LDA));
          PetscCall(PetscBLASIntCast(primal_dofs,&Blas_LDB));
          PetscCall(PetscBLASIntCast(size_of_constraint,&Blas_LDC));
          PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
          PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&Blas_M,&Blas_N,&Blas_K,&one,qr_basis,&Blas_LDA,trs_rhs,&Blas_LDB,&zero,constraints_data+constraints_data_ptr[total_counts],&Blas_LDC));
          PetscCall(PetscFPTrapPop());
          PetscCall(PetscArraycpy(qr_basis,&constraints_data[constraints_data_ptr[total_counts]],size_of_constraint*primal_dofs));

          /* insert values in change of basis matrix respecting global ordering of new primal dofs */
          start_rows = &constraints_idxs[constraints_idxs_ptr[total_counts]];
          /* insert cols for primal dofs */
          for (j=0;j<primal_dofs;j++) {
            start_vals = &qr_basis[j*size_of_constraint];
            start_cols = &constraints_idxs[constraints_idxs_ptr[total_counts]+j];
            PetscCall(MatSetValues(localChangeOfBasisMatrix,size_of_constraint,start_rows,1,start_cols,start_vals,INSERT_VALUES));
          }
          /* insert cols for dual dofs */
          for (j=0,k=0;j<dual_dofs;k++) {
            if (!PetscBTLookup(is_primal,constraints_idxs_B[constraints_idxs_ptr[total_counts]+k])) {
              start_vals = &qr_basis[(primal_dofs+j)*size_of_constraint];
              start_cols = &constraints_idxs[constraints_idxs_ptr[total_counts]+k];
              PetscCall(MatSetValues(localChangeOfBasisMatrix,size_of_constraint,start_rows,1,start_cols,start_vals,INSERT_VALUES));
              j++;
            }
          }

          /* check change of basis */
          if (pcbddc->dbg_flag) {
            PetscInt   ii,jj;
            PetscBool valid_qr=PETSC_TRUE;
            PetscCall(PetscBLASIntCast(primal_dofs,&Blas_M));
            PetscCall(PetscBLASIntCast(size_of_constraint,&Blas_N));
            PetscCall(PetscBLASIntCast(size_of_constraint,&Blas_K));
            PetscCall(PetscBLASIntCast(size_of_constraint,&Blas_LDA));
            PetscCall(PetscBLASIntCast(size_of_constraint,&Blas_LDB));
            PetscCall(PetscBLASIntCast(primal_dofs,&Blas_LDC));
            PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
            PetscStackCallBLAS("BLASgemm",BLASgemm_("T","N",&Blas_M,&Blas_N,&Blas_K,&one,dbg_work,&Blas_LDA,qr_basis,&Blas_LDB,&zero,&dbg_work[size_of_constraint*primal_dofs],&Blas_LDC));
            PetscCall(PetscFPTrapPop());
            for (jj=0;jj<size_of_constraint;jj++) {
              for (ii=0;ii<primal_dofs;ii++) {
                if (ii != jj && PetscAbsScalar(dbg_work[size_of_constraint*primal_dofs+jj*primal_dofs+ii]) > 1.e-12) valid_qr = PETSC_FALSE;
                if (ii == jj && PetscAbsScalar(dbg_work[size_of_constraint*primal_dofs+jj*primal_dofs+ii]-(PetscReal)1) > 1.e-12) valid_qr = PETSC_FALSE;
              }
            }
            if (!valid_qr) {
              PetscCall(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"\t-> wrong change of basis!\n"));
              for (jj=0;jj<size_of_constraint;jj++) {
                for (ii=0;ii<primal_dofs;ii++) {
                  if (ii != jj && PetscAbsScalar(dbg_work[size_of_constraint*primal_dofs+jj*primal_dofs+ii]) > 1.e-12) {
                    PetscCall(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"\tQr basis function %" PetscInt_FMT " is not orthogonal to constraint %" PetscInt_FMT " (%1.14e)!\n",jj,ii,(double)PetscAbsScalar(dbg_work[size_of_constraint*primal_dofs+jj*primal_dofs+ii])));
                  }
                  if (ii == jj && PetscAbsScalar(dbg_work[size_of_constraint*primal_dofs+jj*primal_dofs+ii]-(PetscReal)1) > 1.e-12) {
                    PetscCall(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"\tQr basis function %" PetscInt_FMT " is not unitary w.r.t constraint %" PetscInt_FMT " (%1.14e)!\n",jj,ii,(double)PetscAbsScalar(dbg_work[size_of_constraint*primal_dofs+jj*primal_dofs+ii])));
                  }
                }
              }
            } else {
              PetscCall(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"\t-> right change of basis!\n"));
            }
          }
        } else { /* simple transformation block */
          PetscInt    row,col;
          PetscScalar val,norm;

          PetscCall(PetscBLASIntCast(size_of_constraint,&Blas_N));
          PetscStackCallBLAS("BLASdot",norm = BLASdot_(&Blas_N,constraints_data+constraints_data_ptr[total_counts],&Blas_one,constraints_data+constraints_data_ptr[total_counts],&Blas_one));
          for (j=0;j<size_of_constraint;j++) {
            PetscInt row_B = constraints_idxs_B[constraints_idxs_ptr[total_counts]+j];
            row = constraints_idxs[constraints_idxs_ptr[total_counts]+j];
            if (!PetscBTLookup(is_primal,row_B)) {
              col = constraints_idxs[constraints_idxs_ptr[total_counts]];
              PetscCall(MatSetValue(localChangeOfBasisMatrix,row,row,1.0,INSERT_VALUES));
              PetscCall(MatSetValue(localChangeOfBasisMatrix,row,col,constraints_data[constraints_data_ptr[total_counts]+j]/norm,INSERT_VALUES));
            } else {
              for (k=0;k<size_of_constraint;k++) {
                col = constraints_idxs[constraints_idxs_ptr[total_counts]+k];
                if (row != col) {
                  val = -constraints_data[constraints_data_ptr[total_counts]+k]/constraints_data[constraints_data_ptr[total_counts]];
                } else {
                  val = constraints_data[constraints_data_ptr[total_counts]]/norm;
                }
                PetscCall(MatSetValue(localChangeOfBasisMatrix,row,col,val,INSERT_VALUES));
              }
            }
          }
          if (pcbddc->dbg_flag) {
            PetscCall(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"\t-> using standard change of basis\n"));
          }
        }
      } else {
        if (pcbddc->dbg_flag) {
          PetscCall(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Constraint %" PetscInt_FMT " does not need a change of basis (size %" PetscInt_FMT ")\n",total_counts,size_of_constraint));
        }
      }
    }

    /* free workspace */
    if (qr_needed) {
      if (pcbddc->dbg_flag) {
        PetscCall(PetscFree(dbg_work));
      }
      PetscCall(PetscFree(trs_rhs));
      PetscCall(PetscFree(qr_tau));
      PetscCall(PetscFree(qr_work));
      PetscCall(PetscFree(gqr_work));
      PetscCall(PetscFree(qr_basis));
    }
    PetscCall(PetscBTDestroy(&is_primal));
    PetscCall(MatAssemblyBegin(localChangeOfBasisMatrix,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(localChangeOfBasisMatrix,MAT_FINAL_ASSEMBLY));

    /* assembling of global change of variable */
    if (!pcbddc->fake_change) {
      Mat      tmat;
      PetscInt bs;

      PetscCall(VecGetSize(pcis->vec1_global,&global_size));
      PetscCall(VecGetLocalSize(pcis->vec1_global,&local_size));
      PetscCall(MatDuplicate(pc->pmat,MAT_DO_NOT_COPY_VALUES,&tmat));
      PetscCall(MatISSetLocalMat(tmat,localChangeOfBasisMatrix));
      PetscCall(MatAssemblyBegin(tmat,MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(tmat,MAT_FINAL_ASSEMBLY));
      PetscCall(MatCreate(PetscObjectComm((PetscObject)pc),&pcbddc->ChangeOfBasisMatrix));
      PetscCall(MatSetType(pcbddc->ChangeOfBasisMatrix,MATAIJ));
      PetscCall(MatGetBlockSize(pc->pmat,&bs));
      PetscCall(MatSetBlockSize(pcbddc->ChangeOfBasisMatrix,bs));
      PetscCall(MatSetSizes(pcbddc->ChangeOfBasisMatrix,local_size,local_size,global_size,global_size));
      PetscCall(MatISSetMPIXAIJPreallocation_Private(tmat,pcbddc->ChangeOfBasisMatrix,PETSC_TRUE));
      PetscCall(MatConvert(tmat,MATAIJ,MAT_REUSE_MATRIX,&pcbddc->ChangeOfBasisMatrix));
      PetscCall(MatDestroy(&tmat));
      PetscCall(VecSet(pcis->vec1_global,0.0));
      PetscCall(VecSet(pcis->vec1_N,1.0));
      PetscCall(VecScatterBegin(matis->rctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE));
      PetscCall(VecScatterEnd(matis->rctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE));
      PetscCall(VecReciprocal(pcis->vec1_global));
      PetscCall(MatDiagonalScale(pcbddc->ChangeOfBasisMatrix,pcis->vec1_global,NULL));

      /* check */
      if (pcbddc->dbg_flag) {
        PetscReal error;
        Vec       x,x_change;

        PetscCall(VecDuplicate(pcis->vec1_global,&x));
        PetscCall(VecDuplicate(pcis->vec1_global,&x_change));
        PetscCall(VecSetRandom(x,NULL));
        PetscCall(VecCopy(x,pcis->vec1_global));
        PetscCall(VecScatterBegin(matis->rctx,x,pcis->vec1_N,INSERT_VALUES,SCATTER_FORWARD));
        PetscCall(VecScatterEnd(matis->rctx,x,pcis->vec1_N,INSERT_VALUES,SCATTER_FORWARD));
        PetscCall(MatMult(localChangeOfBasisMatrix,pcis->vec1_N,pcis->vec2_N));
        PetscCall(VecScatterBegin(matis->rctx,pcis->vec2_N,x,INSERT_VALUES,SCATTER_REVERSE));
        PetscCall(VecScatterEnd(matis->rctx,pcis->vec2_N,x,INSERT_VALUES,SCATTER_REVERSE));
        PetscCall(MatMult(pcbddc->ChangeOfBasisMatrix,pcis->vec1_global,x_change));
        PetscCall(VecAXPY(x,-1.0,x_change));
        PetscCall(VecNorm(x,NORM_INFINITY,&error));
        if (error > PETSC_SMALL) {
          SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_PLIB,"Error global vs local change on N: %1.6e",(double)error);
        }
        PetscCall(VecDestroy(&x));
        PetscCall(VecDestroy(&x_change));
      }
      /* adapt sub_schurs computed (if any) */
      if (pcbddc->use_deluxe_scaling) {
        PCBDDCSubSchurs sub_schurs=pcbddc->sub_schurs;

        PetscCheck(!pcbddc->use_change_of_basis || !pcbddc->adaptive_userdefined,PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Cannot mix automatic change of basis, adaptive selection and user-defined constraints");
        if (sub_schurs && sub_schurs->S_Ej_all) {
          Mat                    S_new,tmat;
          IS                     is_all_N,is_V_Sall = NULL;

          PetscCall(ISLocalToGlobalMappingApplyIS(pcis->BtoNmap,sub_schurs->is_Ej_all,&is_all_N));
          PetscCall(MatCreateSubMatrix(localChangeOfBasisMatrix,is_all_N,is_all_N,MAT_INITIAL_MATRIX,&tmat));
          if (pcbddc->deluxe_zerorows) {
            ISLocalToGlobalMapping NtoSall;
            IS                     is_V;
            PetscCall(ISCreateGeneral(PETSC_COMM_SELF,pcbddc->n_vertices,pcbddc->local_primal_ref_node,PETSC_COPY_VALUES,&is_V));
            PetscCall(ISLocalToGlobalMappingCreateIS(is_all_N,&NtoSall));
            PetscCall(ISGlobalToLocalMappingApplyIS(NtoSall,IS_GTOLM_DROP,is_V,&is_V_Sall));
            PetscCall(ISLocalToGlobalMappingDestroy(&NtoSall));
            PetscCall(ISDestroy(&is_V));
          }
          PetscCall(ISDestroy(&is_all_N));
          PetscCall(MatPtAP(sub_schurs->S_Ej_all,tmat,MAT_INITIAL_MATRIX,1.0,&S_new));
          PetscCall(MatDestroy(&sub_schurs->S_Ej_all));
          PetscCall(PetscObjectReference((PetscObject)S_new));
          if (pcbddc->deluxe_zerorows) {
            const PetscScalar *array;
            const PetscInt    *idxs_V,*idxs_all;
            PetscInt          i,n_V;

            PetscCall(MatZeroRowsColumnsIS(S_new,is_V_Sall,1.,NULL,NULL));
            PetscCall(ISGetLocalSize(is_V_Sall,&n_V));
            PetscCall(ISGetIndices(is_V_Sall,&idxs_V));
            PetscCall(ISGetIndices(sub_schurs->is_Ej_all,&idxs_all));
            PetscCall(VecGetArrayRead(pcis->D,&array));
            for (i=0;i<n_V;i++) {
              PetscScalar val;
              PetscInt    idx;

              idx = idxs_V[i];
              val = array[idxs_all[idxs_V[i]]];
              PetscCall(MatSetValue(S_new,idx,idx,val,INSERT_VALUES));
            }
            PetscCall(MatAssemblyBegin(S_new,MAT_FINAL_ASSEMBLY));
            PetscCall(MatAssemblyEnd(S_new,MAT_FINAL_ASSEMBLY));
            PetscCall(VecRestoreArrayRead(pcis->D,&array));
            PetscCall(ISRestoreIndices(sub_schurs->is_Ej_all,&idxs_all));
            PetscCall(ISRestoreIndices(is_V_Sall,&idxs_V));
          }
          sub_schurs->S_Ej_all = S_new;
          PetscCall(MatDestroy(&S_new));
          if (sub_schurs->sum_S_Ej_all) {
            PetscCall(MatPtAP(sub_schurs->sum_S_Ej_all,tmat,MAT_INITIAL_MATRIX,1.0,&S_new));
            PetscCall(MatDestroy(&sub_schurs->sum_S_Ej_all));
            PetscCall(PetscObjectReference((PetscObject)S_new));
            if (pcbddc->deluxe_zerorows) {
              PetscCall(MatZeroRowsColumnsIS(S_new,is_V_Sall,1.,NULL,NULL));
            }
            sub_schurs->sum_S_Ej_all = S_new;
            PetscCall(MatDestroy(&S_new));
          }
          PetscCall(ISDestroy(&is_V_Sall));
          PetscCall(MatDestroy(&tmat));
        }
        /* destroy any change of basis context in sub_schurs */
        if (sub_schurs && sub_schurs->change) {
          PetscInt i;

          for (i=0;i<sub_schurs->n_subs;i++) {
            PetscCall(KSPDestroy(&sub_schurs->change[i]));
          }
          PetscCall(PetscFree(sub_schurs->change));
        }
      }
      if (pcbddc->switch_static) { /* need to save the local change */
        pcbddc->switch_static_change = localChangeOfBasisMatrix;
      } else {
        PetscCall(MatDestroy(&localChangeOfBasisMatrix));
      }
      /* determine if any process has changed the pressures locally */
      pcbddc->change_interior = pcbddc->benign_have_null;
    } else { /* fake change (get back change of basis into ConstraintMatrix and info on qr) */
      PetscCall(MatDestroy(&pcbddc->ConstraintMatrix));
      pcbddc->ConstraintMatrix = localChangeOfBasisMatrix;
      pcbddc->use_qr_single = qr_needed;
    }
  } else if (pcbddc->user_ChangeOfBasisMatrix || pcbddc->benign_saddle_point) {
    if (!pcbddc->benign_have_null && pcbddc->user_ChangeOfBasisMatrix) {
      PetscCall(PetscObjectReference((PetscObject)pcbddc->user_ChangeOfBasisMatrix));
      pcbddc->ChangeOfBasisMatrix = pcbddc->user_ChangeOfBasisMatrix;
    } else {
      Mat benign_global = NULL;
      if (pcbddc->benign_have_null) {
        Mat M;

        pcbddc->change_interior = PETSC_TRUE;
        PetscCall(VecCopy(matis->counter,pcis->vec1_N));
        PetscCall(VecReciprocal(pcis->vec1_N));
        PetscCall(MatDuplicate(pc->pmat,MAT_DO_NOT_COPY_VALUES,&benign_global));
        if (pcbddc->benign_change) {
          PetscCall(MatDuplicate(pcbddc->benign_change,MAT_COPY_VALUES,&M));
          PetscCall(MatDiagonalScale(M,pcis->vec1_N,NULL));
        } else {
          PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF,pcis->n,pcis->n,1,NULL,&M));
          PetscCall(MatDiagonalSet(M,pcis->vec1_N,INSERT_VALUES));
        }
        PetscCall(MatISSetLocalMat(benign_global,M));
        PetscCall(MatDestroy(&M));
        PetscCall(MatAssemblyBegin(benign_global,MAT_FINAL_ASSEMBLY));
        PetscCall(MatAssemblyEnd(benign_global,MAT_FINAL_ASSEMBLY));
      }
      if (pcbddc->user_ChangeOfBasisMatrix) {
        PetscCall(MatMatMult(pcbddc->user_ChangeOfBasisMatrix,benign_global,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&pcbddc->ChangeOfBasisMatrix));
        PetscCall(MatDestroy(&benign_global));
      } else if (pcbddc->benign_have_null) {
        pcbddc->ChangeOfBasisMatrix = benign_global;
      }
    }
    if (pcbddc->switch_static && pcbddc->ChangeOfBasisMatrix) { /* need to save the local change */
      IS             is_global;
      const PetscInt *gidxs;

      PetscCall(ISLocalToGlobalMappingGetIndices(matis->rmapping,&gidxs));
      PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)pc),pcis->n,gidxs,PETSC_COPY_VALUES,&is_global));
      PetscCall(ISLocalToGlobalMappingRestoreIndices(matis->rmapping,&gidxs));
      PetscCall(MatCreateSubMatrixUnsorted(pcbddc->ChangeOfBasisMatrix,is_global,is_global,&pcbddc->switch_static_change));
      PetscCall(ISDestroy(&is_global));
    }
  }
  if (!pcbddc->fake_change && pcbddc->ChangeOfBasisMatrix && !pcbddc->work_change) {
    PetscCall(VecDuplicate(pcis->vec1_global,&pcbddc->work_change));
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
      PetscCall(PetscArraycmp(pcbddc->local_primal_ref_node,olocal_primal_ref_node,olocal_primal_size_cc,&pcbddc->new_primal_space_local));
      pcbddc->new_primal_space_local = (PetscBool)(!pcbddc->new_primal_space_local);
      if (!pcbddc->new_primal_space_local) {
        PetscCall(PetscArraycmp(pcbddc->local_primal_ref_mult,olocal_primal_ref_mult,olocal_primal_size_cc,&pcbddc->new_primal_space_local));
        pcbddc->new_primal_space_local = (PetscBool)(!pcbddc->new_primal_space_local);
      }
    }
    /* new_primal_space will be used for numbering of coarse dofs, so it should be the same across all subdomains */
    PetscCall(MPIU_Allreduce(&pcbddc->new_primal_space_local,&pcbddc->new_primal_space,1,MPIU_BOOL,MPI_LOR,PetscObjectComm((PetscObject)pc)));
  }
  PetscCall(PetscFree2(olocal_primal_ref_node,olocal_primal_ref_mult));

  /* flush dbg viewer */
  if (pcbddc->dbg_flag) {
    PetscCall(PetscViewerFlush(pcbddc->dbg_viewer));
  }

  /* free workspace */
  PetscCall(PetscBTDestroy(&qr_needed_idx));
  PetscCall(PetscBTDestroy(&change_basis));
  if (!pcbddc->adaptive_selection) {
    PetscCall(PetscFree3(constraints_idxs_ptr,constraints_data_ptr,constraints_n));
    PetscCall(PetscFree3(constraints_data,constraints_idxs,constraints_idxs_B));
  } else {
    PetscCall(PetscFree5(pcbddc->adaptive_constraints_n,pcbddc->adaptive_constraints_idxs_ptr,pcbddc->adaptive_constraints_data_ptr,pcbddc->adaptive_constraints_idxs,pcbddc->adaptive_constraints_data));
    PetscCall(PetscFree(constraints_n));
    PetscCall(PetscFree(constraints_idxs_B));
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
    PetscCall(PCBDDCGraphReset(pcbddc->mat_graph));
    /* Init local Graph struct */
    PetscCall(MatGetSize(pc->pmat,&N,NULL));
    PetscCall(MatISGetLocalToGlobalMapping(pc->pmat,&map,NULL));
    PetscCall(PCBDDCGraphInit(pcbddc->mat_graph,map,N,pcbddc->graphmaxcount));

    if (pcbddc->user_primal_vertices_local && !pcbddc->user_primal_vertices) {
      PetscCall(PCBDDCConsistencyCheckIS(pc,MPI_LOR,&pcbddc->user_primal_vertices_local));
    }
    /* Check validity of the csr graph passed in by the user */
    PetscCheck(!pcbddc->mat_graph->nvtxs_csr || pcbddc->mat_graph->nvtxs_csr == pcbddc->mat_graph->nvtxs,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid size of local CSR graph! Found %" PetscInt_FMT ", expected %" PetscInt_FMT,pcbddc->mat_graph->nvtxs_csr,pcbddc->mat_graph->nvtxs);

    /* Set default CSR adjacency of local dofs if not provided by the user with PCBDDCSetLocalAdjacencyGraph */
    if (!pcbddc->mat_graph->xadj && pcbddc->use_local_adj) {
      PetscInt  *xadj,*adjncy;
      PetscInt  nvtxs;
      PetscBool flg_row=PETSC_FALSE;

      PetscCall(MatGetRowIJ(matis->A,0,PETSC_TRUE,PETSC_FALSE,&nvtxs,(const PetscInt**)&xadj,(const PetscInt**)&adjncy,&flg_row));
      if (flg_row) {
        PetscCall(PCBDDCSetLocalAdjacencyGraph(pc,nvtxs,xadj,adjncy,PETSC_COPY_VALUES));
        pcbddc->computed_rowadj = PETSC_TRUE;
      }
      PetscCall(MatRestoreRowIJ(matis->A,0,PETSC_TRUE,PETSC_FALSE,&nvtxs,(const PetscInt**)&xadj,(const PetscInt**)&adjncy,&flg_row));
      rcsr = PETSC_TRUE;
    }
    if (pcbddc->dbg_flag) {
      PetscCall(PetscViewerFlush(pcbddc->dbg_viewer));
    }

    if (pcbddc->mat_graph->cdim && !pcbddc->mat_graph->cloc) {
      PetscReal    *lcoords;
      PetscInt     n;
      MPI_Datatype dimrealtype;

      /* TODO: support for blocked */
      PetscCheck(pcbddc->mat_graph->cnloc == pc->pmat->rmap->n,PETSC_COMM_SELF,PETSC_ERR_USER,"Invalid number of local coordinates! Got %" PetscInt_FMT ", expected %" PetscInt_FMT,pcbddc->mat_graph->cnloc,pc->pmat->rmap->n);
      PetscCall(MatGetLocalSize(matis->A,&n,NULL));
      PetscCall(PetscMalloc1(pcbddc->mat_graph->cdim*n,&lcoords));
      PetscCallMPI(MPI_Type_contiguous(pcbddc->mat_graph->cdim,MPIU_REAL,&dimrealtype));
      PetscCallMPI(MPI_Type_commit(&dimrealtype));
      PetscCall(PetscSFBcastBegin(matis->sf,dimrealtype,pcbddc->mat_graph->coords,lcoords,MPI_REPLACE));
      PetscCall(PetscSFBcastEnd(matis->sf,dimrealtype,pcbddc->mat_graph->coords,lcoords,MPI_REPLACE));
      PetscCallMPI(MPI_Type_free(&dimrealtype));
      PetscCall(PetscFree(pcbddc->mat_graph->coords));

      pcbddc->mat_graph->coords = lcoords;
      pcbddc->mat_graph->cloc   = PETSC_TRUE;
      pcbddc->mat_graph->cnloc  = n;
    }
    PetscCheck(!pcbddc->mat_graph->cnloc || pcbddc->mat_graph->cnloc == pcbddc->mat_graph->nvtxs,PETSC_COMM_SELF,PETSC_ERR_USER,"Invalid number of local subdomain coordinates! Got %" PetscInt_FMT ", expected %" PetscInt_FMT,pcbddc->mat_graph->cnloc,pcbddc->mat_graph->nvtxs);
    pcbddc->mat_graph->active_coords = (PetscBool)(pcbddc->corner_selection && pcbddc->mat_graph->cdim && !pcbddc->corner_selected);

    /* Setup of Graph */
    pcbddc->mat_graph->commsizelimit = 0; /* don't use the COMM_SELF variant of the graph */
    PetscCall(PCBDDCGraphSetUp(pcbddc->mat_graph,pcbddc->vertex_size,pcbddc->NeumannBoundariesLocal,pcbddc->DirichletBoundariesLocal,pcbddc->n_ISForDofsLocal,pcbddc->ISForDofsLocal,pcbddc->user_primal_vertices_local));

    /* attach info on disconnected subdomains if present */
    if (pcbddc->n_local_subs) {
      PetscInt *local_subs,n,totn;

      PetscCall(MatGetLocalSize(matis->A,&n,NULL));
      PetscCall(PetscMalloc1(n,&local_subs));
      for (i=0;i<n;i++) local_subs[i] = pcbddc->n_local_subs;
      for (i=0;i<pcbddc->n_local_subs;i++) {
        const PetscInt *idxs;
        PetscInt       nl,j;

        PetscCall(ISGetLocalSize(pcbddc->local_subs[i],&nl));
        PetscCall(ISGetIndices(pcbddc->local_subs[i],&idxs));
        for (j=0;j<nl;j++) local_subs[idxs[j]] = i;
        PetscCall(ISRestoreIndices(pcbddc->local_subs[i],&idxs));
      }
      for (i=0,totn=0;i<n;i++) totn = PetscMax(totn,local_subs[i]);
      pcbddc->mat_graph->n_local_subs = totn + 1;
      pcbddc->mat_graph->local_subs = local_subs;
    }
  }

  if (!pcbddc->graphanalyzed) {
    /* Graph's connected components analysis */
    PetscCall(PCBDDCGraphComputeConnectedComponents(pcbddc->mat_graph));
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
  PetscCall(PetscMalloc2(n,&alphas,n,&onorms));
  PetscCall(VecNormalize(vecs[0],&norm));
  if (norm < PETSC_SMALL) {
    onorms[0] = 0.0;
    PetscCall(VecSet(vecs[0],0.0));
  } else {
    onorms[0] = norm;
  }

  for (i=1;i<n;i++) {
    PetscCall(VecMDot(vecs[i],i,vecs,alphas));
    for (j=0;j<i;j++) alphas[j] = PetscConj(-alphas[j]);
    PetscCall(VecMAXPY(vecs[i],i,alphas,vecs));
    PetscCall(VecNormalize(vecs[i],&norm));
    if (norm < PETSC_SMALL) {
      onorms[i] = 0.0;
      PetscCall(VecSet(vecs[i],0.0));
    } else {
      onorms[i] = norm;
    }
  }
  /* push nonzero vectors at the beginning */
  for (i=0;i<n;i++) {
    if (onorms[i] == 0.0) {
      for (j=i+1;j<n;j++) {
        if (onorms[j] != 0.0) {
          PetscCall(VecCopy(vecs[j],vecs[i]));
          onorms[j] = 0.0;
        }
      }
    }
  }
  for (i=0,*nio=0;i<n;i++) *nio += onorms[i] != 0.0 ? 1 : 0;
  PetscCall(PetscFree2(alphas,onorms));
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
  PetscCall(PetscObjectTypeCompare((PetscObject)mat,MATIS,&ismatis));
  PetscCheck(ismatis,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Cannot use %s on a matrix object which is not of type MATIS",PETSC_FUNCTION_NAME);
  PetscValidLogicalCollectiveInt(mat,*n_subdomains,2);
  PetscValidLogicalCollectiveInt(mat,redprocs,3);
  PetscCheck(*n_subdomains >0,PetscObjectComm((PetscObject)mat),PETSC_ERR_ARG_WRONG,"Invalid number of subdomains requested %" PetscInt_FMT,*n_subdomains);

  if (have_void) *have_void = PETSC_FALSE;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)mat),&size));
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)mat),&rank));
  PetscCall(MatISGetLocalMat(mat,&A));
  PetscCall(MatGetLocalSize(A,&n,NULL));
  im_active = !!n;
  PetscCall(MPIU_Allreduce(&im_active,&active_procs,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)mat)));
  void_procs = size - active_procs;
  /* get ranks of of non-active processes in mat communicator */
  if (void_procs) {
    PetscInt ncand;

    if (have_void) *have_void = PETSC_TRUE;
    PetscCall(PetscMalloc1(size,&procs_candidates));
    PetscCallMPI(MPI_Allgather(&im_active,1,MPIU_INT,procs_candidates,1,MPIU_INT,PetscObjectComm((PetscObject)mat)));
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
  PetscCall(MatGetSize(mat,&N,NULL));
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
    PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)mat),issize,&isidx,PETSC_COPY_VALUES,is_sends));
    PetscCall(PetscFree(procs_candidates));
    PetscFunctionReturn(0);
  }
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-matis_partitioning_use_vwgt",&use_vwgt,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-matis_partitioning_threshold",&threshold,NULL));
  threshold = PetscMax(threshold,2);

  /* Get info on mapping */
  PetscCall(MatISGetLocalToGlobalMapping(mat,&mapping,NULL));
  PetscCall(ISLocalToGlobalMappingGetInfo(mapping,&n_neighs,&neighs,&n_shared,&shared));

  /* build local CSR graph of subdomains' connectivity */
  PetscCall(PetscMalloc1(2,&xadj));
  xadj[0] = 0;
  xadj[1] = PetscMax(n_neighs-1,0);
  PetscCall(PetscMalloc1(xadj[1],&adjncy));
  PetscCall(PetscMalloc1(xadj[1],&adjncy_wgt));
  PetscCall(PetscCalloc1(n,&count));
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
  PetscCall(PetscFree(count));
  PetscCall(ISLocalToGlobalMappingRestoreInfo(mapping,&n_neighs,&neighs,&n_shared,&shared));
  PetscCall(PetscSortIntWithArray(xadj[1],adjncy,adjncy_wgt));

  PetscCall(PetscMalloc1(1,&ranks_send_to_idx));

  /* Restrict work on active processes only */
  PetscCall(PetscMPIIntCast(im_active,&color));
  if (void_procs) {
    PetscCall(PetscSubcommCreate(PetscObjectComm((PetscObject)mat),&psubcomm));
    PetscCall(PetscSubcommSetNumber(psubcomm,2)); /* 2 groups, active process and not active processes */
    PetscCall(PetscSubcommSetTypeGeneral(psubcomm,color,rank));
    subcomm = PetscSubcommChild(psubcomm);
  } else {
    psubcomm = NULL;
    subcomm = PetscObjectComm((PetscObject)mat);
  }

  v_wgt = NULL;
  if (!color) {
    PetscCall(PetscFree(xadj));
    PetscCall(PetscFree(adjncy));
    PetscCall(PetscFree(adjncy_wgt));
  } else {
    Mat             subdomain_adj;
    IS              new_ranks,new_ranks_contig;
    MatPartitioning partitioner;
    PetscInt        rstart=0,rend=0;
    PetscInt        *is_indices,*oldranks;
    PetscMPIInt     size;
    PetscBool       aggregate;

    PetscCallMPI(MPI_Comm_size(subcomm,&size));
    if (void_procs) {
      PetscInt prank = rank;
      PetscCall(PetscMalloc1(size,&oldranks));
      PetscCallMPI(MPI_Allgather(&prank,1,MPIU_INT,oldranks,1,MPIU_INT,subcomm));
      for (i=0;i<xadj[1];i++) {
        PetscCall(PetscFindInt(adjncy[i],size,oldranks,&adjncy[i]));
      }
      PetscCall(PetscSortIntWithArray(xadj[1],adjncy,adjncy_wgt));
    } else {
      oldranks = NULL;
    }
    aggregate = ((redprocs > 0 && redprocs < size) ? PETSC_TRUE : PETSC_FALSE);
    if (aggregate) { /* TODO: all this part could be made more efficient */
      PetscInt    lrows,row,ncols,*cols;
      PetscMPIInt nrank;
      PetscScalar *vals;

      PetscCallMPI(MPI_Comm_rank(subcomm,&nrank));
      lrows = 0;
      if (nrank<redprocs) {
        lrows = size/redprocs;
        if (nrank<size%redprocs) lrows++;
      }
      PetscCall(MatCreateAIJ(subcomm,lrows,lrows,size,size,50,NULL,50,NULL,&subdomain_adj));
      PetscCall(MatGetOwnershipRange(subdomain_adj,&rstart,&rend));
      PetscCall(MatSetOption(subdomain_adj,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_FALSE));
      PetscCall(MatSetOption(subdomain_adj,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE));
      row = nrank;
      ncols = xadj[1]-xadj[0];
      cols = adjncy;
      PetscCall(PetscMalloc1(ncols,&vals));
      for (i=0;i<ncols;i++) vals[i] = adjncy_wgt[i];
      PetscCall(MatSetValues(subdomain_adj,1,&row,ncols,cols,vals,INSERT_VALUES));
      PetscCall(MatAssemblyBegin(subdomain_adj,MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(subdomain_adj,MAT_FINAL_ASSEMBLY));
      PetscCall(PetscFree(xadj));
      PetscCall(PetscFree(adjncy));
      PetscCall(PetscFree(adjncy_wgt));
      PetscCall(PetscFree(vals));
      if (use_vwgt) {
        Vec               v;
        const PetscScalar *array;
        PetscInt          nl;

        PetscCall(MatCreateVecs(subdomain_adj,&v,NULL));
        PetscCall(VecSetValue(v,row,(PetscScalar)n,INSERT_VALUES));
        PetscCall(VecAssemblyBegin(v));
        PetscCall(VecAssemblyEnd(v));
        PetscCall(VecGetLocalSize(v,&nl));
        PetscCall(VecGetArrayRead(v,&array));
        PetscCall(PetscMalloc1(nl,&v_wgt));
        for (i=0;i<nl;i++) v_wgt[i] = (PetscInt)PetscRealPart(array[i]);
        PetscCall(VecRestoreArrayRead(v,&array));
        PetscCall(VecDestroy(&v));
      }
    } else {
      PetscCall(MatCreateMPIAdj(subcomm,1,(PetscInt)size,xadj,adjncy,adjncy_wgt,&subdomain_adj));
      if (use_vwgt) {
        PetscCall(PetscMalloc1(1,&v_wgt));
        v_wgt[0] = n;
      }
    }
    /* PetscCall(MatView(subdomain_adj,0)); */

    /* Partition */
    PetscCall(MatPartitioningCreate(subcomm,&partitioner));
#if defined(PETSC_HAVE_PTSCOTCH)
    PetscCall(MatPartitioningSetType(partitioner,MATPARTITIONINGPTSCOTCH));
#elif defined(PETSC_HAVE_PARMETIS)
    PetscCall(MatPartitioningSetType(partitioner,MATPARTITIONINGPARMETIS));
#else
    PetscCall(MatPartitioningSetType(partitioner,MATPARTITIONINGAVERAGE));
#endif
    PetscCall(MatPartitioningSetAdjacency(partitioner,subdomain_adj));
    if (v_wgt) {
      PetscCall(MatPartitioningSetVertexWeights(partitioner,v_wgt));
    }
    *n_subdomains = PetscMin((PetscInt)size,*n_subdomains);
    PetscCall(MatPartitioningSetNParts(partitioner,*n_subdomains));
    PetscCall(MatPartitioningSetFromOptions(partitioner));
    PetscCall(MatPartitioningApply(partitioner,&new_ranks));
    /* PetscCall(MatPartitioningView(partitioner,0)); */

    /* renumber new_ranks to avoid "holes" in new set of processors */
    PetscCall(ISRenumber(new_ranks,NULL,NULL,&new_ranks_contig));
    PetscCall(ISDestroy(&new_ranks));
    PetscCall(ISGetIndices(new_ranks_contig,(const PetscInt**)&is_indices));
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

      PetscCall(PetscObjectGetNewTag((PetscObject)subdomain_adj,&tag));
      PetscCall(PetscMalloc1(rend-rstart,&reqs));
      for (i=rstart;i<rend;i++) {
        PetscCallMPI(MPI_Isend(is_indices+i-rstart,1,MPIU_INT,i,tag,subcomm,&reqs[i-rstart]));
      }
      PetscCallMPI(MPI_Recv(&idx,1,MPIU_INT,MPI_ANY_SOURCE,tag,subcomm,MPI_STATUS_IGNORE));
      PetscCallMPI(MPI_Waitall(rend-rstart,reqs,MPI_STATUSES_IGNORE));
      PetscCall(PetscFree(reqs));
      if (procs_candidates) { /* shift the pattern on non-active candidates (if any) */
        PetscAssert(oldranks,PETSC_COMM_SELF,PETSC_ERR_PLIB,"This should not happen");
        ranks_send_to_idx[0] = procs_candidates[oldranks[idx]];
      } else if (oldranks) {
        ranks_send_to_idx[0] = oldranks[idx];
      } else {
        ranks_send_to_idx[0] = idx;
      }
    }
    PetscCall(ISRestoreIndices(new_ranks_contig,(const PetscInt**)&is_indices));
    /* clean up */
    PetscCall(PetscFree(oldranks));
    PetscCall(ISDestroy(&new_ranks_contig));
    PetscCall(MatDestroy(&subdomain_adj));
    PetscCall(MatPartitioningDestroy(&partitioner));
  }
  PetscCall(PetscSubcommDestroy(&psubcomm));
  PetscCall(PetscFree(procs_candidates));

  /* assemble parallel IS for sends */
  i = 1;
  if (!color) i=0;
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)mat),i,ranks_send_to_idx,PETSC_OWN_POINTER,is_sends));
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
  PetscCall(PetscObjectTypeCompare((PetscObject)mat,MATIS,&ismatis));
  PetscCheck(ismatis,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Cannot use %s on a matrix object which is not of type MATIS",PETSC_FUNCTION_NAME);
  PetscValidLogicalCollectiveInt(mat,n_subdomains,3);
  PetscValidLogicalCollectiveBool(mat,restrict_comm,4);
  PetscValidLogicalCollectiveBool(mat,restrict_full,5);
  PetscValidLogicalCollectiveBool(mat,reuse,6);
  PetscValidLogicalCollectiveInt(mat,nis,8);
  PetscValidLogicalCollectiveInt(mat,nvecs,10);
  if (nvecs) {
    PetscCheck(nvecs <= 1,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Just 1 vector supported");
    PetscValidHeaderSpecific(nnsp_vec[0],VEC_CLASSID,11);
  }
  /* further checks */
  PetscCall(MatISGetLocalMat(mat,&local_mat));
  PetscCall(PetscObjectTypeCompare((PetscObject)local_mat,MATSEQDENSE,&isdense));
  PetscCheck(isdense,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Currently cannot subassemble MATIS when local matrix type is not of type SEQDENSE");
  PetscCall(MatGetSize(local_mat,&rows,&cols));
  PetscCheck(rows == cols,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Local MATIS matrices should be square");
  if (reuse && *mat_n) {
    PetscInt mrows,mcols,mnrows,mncols;
    PetscValidHeaderSpecific(*mat_n,MAT_CLASSID,7);
    PetscCall(PetscObjectTypeCompare((PetscObject)*mat_n,MATIS,&ismatis));
    PetscCheck(ismatis,PetscObjectComm((PetscObject)*mat_n),PETSC_ERR_SUP,"Cannot reuse a matrix which is not of type MATIS");
    PetscCall(MatGetSize(mat,&mrows,&mcols));
    PetscCall(MatGetSize(*mat_n,&mnrows,&mncols));
    PetscCheck(mrows == mnrows,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Cannot reuse matrix! Wrong number of rows %" PetscInt_FMT " != %" PetscInt_FMT,mrows,mnrows);
    PetscCheck(mcols == mncols,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"Cannot reuse matrix! Wrong number of cols %" PetscInt_FMT " != %" PetscInt_FMT,mcols,mncols);
  }
  PetscCall(MatGetBlockSize(local_mat,&bs));
  PetscValidLogicalCollectiveInt(mat,bs,1);

  /* prepare IS for sending if not provided */
  if (!is_sends) {
    PetscCheck(n_subdomains,PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"You should specify either an IS or a target number of subdomains");
    PetscCall(PCBDDCMatISGetSubassemblingPattern(mat,&n_subdomains,0,&is_sends_internal,NULL));
  } else {
    PetscCall(PetscObjectReference((PetscObject)is_sends));
    is_sends_internal = is_sends;
  }

  /* get comm */
  PetscCall(PetscObjectGetComm((PetscObject)mat,&comm));

  /* compute number of sends */
  PetscCall(ISGetLocalSize(is_sends_internal,&i));
  PetscCall(PetscMPIIntCast(i,&n_sends));

  /* compute number of receives */
  PetscCallMPI(MPI_Comm_size(comm,&size));
  PetscCall(PetscMalloc1(size,&iflags));
  PetscCall(PetscArrayzero(iflags,size));
  PetscCall(ISGetIndices(is_sends_internal,&is_indices));
  for (i=0;i<n_sends;i++) iflags[is_indices[i]] = 1;
  PetscCall(PetscGatherNumberOfMessages(comm,iflags,NULL,&n_recvs));
  PetscCall(PetscFree(iflags));

  /* restrict comm if requested */
  subcomm = NULL;
  destroy_mat = PETSC_FALSE;
  if (restrict_comm) {
    PetscMPIInt color,subcommsize;

    color = 0;
    if (restrict_full) {
      if (!n_recvs) color = 1; /* processes not receiving anything will not participate in new comm (full restriction) */
    } else {
      if (!n_recvs && n_sends) color = 1; /* just those processes that are sending but not receiving anything will not participate in new comm */
    }
    PetscCall(MPIU_Allreduce(&color,&subcommsize,1,MPI_INT,MPI_SUM,comm));
    subcommsize = size - subcommsize;
    /* check if reuse has been requested */
    if (reuse) {
      if (*mat_n) {
        PetscMPIInt subcommsize2;
        PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)*mat_n),&subcommsize2));
        PetscCheck(subcommsize == subcommsize2,PetscObjectComm((PetscObject)*mat_n),PETSC_ERR_PLIB,"Cannot reuse matrix! wrong subcomm size %d != %d",subcommsize,subcommsize2);
        comm_n = PetscObjectComm((PetscObject)*mat_n);
      } else {
        comm_n = PETSC_COMM_SELF;
      }
    } else { /* MAT_INITIAL_MATRIX */
      PetscMPIInt rank;

      PetscCallMPI(MPI_Comm_rank(comm,&rank));
      PetscCall(PetscSubcommCreate(comm,&subcomm));
      PetscCall(PetscSubcommSetNumber(subcomm,2));
      PetscCall(PetscSubcommSetTypeGeneral(subcomm,color,rank));
      comm_n = PetscSubcommChild(subcomm);
    }
    /* flag to destroy *mat_n if not significative */
    if (color) destroy_mat = PETSC_TRUE;
  } else {
    comm_n = comm;
  }

  /* prepare send/receive buffers */
  PetscCall(PetscMalloc1(size,&ilengths_idxs));
  PetscCall(PetscArrayzero(ilengths_idxs,size));
  PetscCall(PetscMalloc1(size,&ilengths_vals));
  PetscCall(PetscArrayzero(ilengths_vals,size));
  if (nis) {
    PetscCall(PetscCalloc1(size,&ilengths_idxs_is));
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

    PetscCall(MatISGetLocalToGlobalMapping(mat,&mapping,NULL));
    PetscCall(MatDenseGetArrayRead(local_mat,&send_buffer_vals));
    PetscCall(ISLocalToGlobalMappingGetSize(mapping,&i));
    PetscCall(PetscMalloc1(i+2,&send_buffer_idxs));
    send_buffer_idxs[0] = (PetscInt)MATDENSE_PRIVATE;
    send_buffer_idxs[1] = i;
    PetscCall(ISLocalToGlobalMappingGetIndices(mapping,(const PetscInt**)&ptr_idxs));
    PetscCall(PetscArraycpy(&send_buffer_idxs[2],ptr_idxs,i));
    PetscCall(ISLocalToGlobalMappingRestoreIndices(mapping,(const PetscInt**)&ptr_idxs));
    PetscCall(PetscMPIIntCast(i,&len));
    for (i=0;i<n_sends;i++) {
      ilengths_vals[is_indices[i]] = len*len;
      ilengths_idxs[is_indices[i]] = len+2;
    }
  }
  PetscCall(PetscGatherMessageLengths2(comm,n_sends,n_recvs,ilengths_idxs,ilengths_vals,&onodes,&olengths_idxs,&olengths_vals));
  /* additional is (if any) */
  if (nis) {
    PetscMPIInt psum;
    PetscInt j;
    for (j=0,psum=0;j<nis;j++) {
      PetscInt plen;
      PetscCall(ISGetLocalSize(isarray[j],&plen));
      PetscCall(PetscMPIIntCast(plen,&len));
      psum += len+1; /* indices + length */
    }
    PetscCall(PetscMalloc1(psum,&send_buffer_idxs_is));
    for (j=0,psum=0;j<nis;j++) {
      PetscInt plen;
      const PetscInt *is_array_idxs;
      PetscCall(ISGetLocalSize(isarray[j],&plen));
      send_buffer_idxs_is[psum] = plen;
      PetscCall(ISGetIndices(isarray[j],&is_array_idxs));
      PetscCall(PetscArraycpy(&send_buffer_idxs_is[psum+1],is_array_idxs,plen));
      PetscCall(ISRestoreIndices(isarray[j],&is_array_idxs));
      psum += plen+1; /* indices + length */
    }
    for (i=0;i<n_sends;i++) {
      ilengths_idxs_is[is_indices[i]] = psum;
    }
    PetscCall(PetscGatherMessageLengths(comm,n_sends,n_recvs,ilengths_idxs_is,&onodes_is,&olengths_idxs_is));
  }
  PetscCall(MatISRestoreLocalMat(mat,&local_mat));

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
  PetscCall(PetscMalloc1(buf_size_idxs,&recv_buffer_idxs));
  PetscCall(PetscMalloc1(buf_size_vals,&recv_buffer_vals));
  PetscCall(PetscMalloc1(buf_size_idxs_is,&recv_buffer_idxs_is));
  PetscCall(PetscMalloc1(buf_size_vecs,&recv_buffer_vecs));

  /* get new tags for clean communications */
  PetscCall(PetscObjectGetNewTag((PetscObject)mat,&tag_idxs));
  PetscCall(PetscObjectGetNewTag((PetscObject)mat,&tag_vals));
  PetscCall(PetscObjectGetNewTag((PetscObject)mat,&tag_idxs_is));
  PetscCall(PetscObjectGetNewTag((PetscObject)mat,&tag_vecs));

  /* allocate for requests */
  PetscCall(PetscMalloc1(n_sends,&send_req_idxs));
  PetscCall(PetscMalloc1(n_sends,&send_req_vals));
  PetscCall(PetscMalloc1(n_sends,&send_req_idxs_is));
  PetscCall(PetscMalloc1(n_sends,&send_req_vecs));
  PetscCall(PetscMalloc1(n_recvs,&recv_req_idxs));
  PetscCall(PetscMalloc1(n_recvs,&recv_req_vals));
  PetscCall(PetscMalloc1(n_recvs,&recv_req_idxs_is));
  PetscCall(PetscMalloc1(n_recvs,&recv_req_vecs));

  /* communications */
  ptr_idxs = recv_buffer_idxs;
  ptr_vals = recv_buffer_vals;
  ptr_idxs_is = recv_buffer_idxs_is;
  ptr_vecs = recv_buffer_vecs;
  for (i=0;i<n_recvs;i++) {
    source_dest = onodes[i];
    PetscCallMPI(MPI_Irecv(ptr_idxs,olengths_idxs[i],MPIU_INT,source_dest,tag_idxs,comm,&recv_req_idxs[i]));
    PetscCallMPI(MPI_Irecv(ptr_vals,olengths_vals[i],MPIU_SCALAR,source_dest,tag_vals,comm,&recv_req_vals[i]));
    ptr_idxs += olengths_idxs[i];
    ptr_vals += olengths_vals[i];
    if (nis) {
      source_dest = onodes_is[i];
      PetscCallMPI(MPI_Irecv(ptr_idxs_is,olengths_idxs_is[i],MPIU_INT,source_dest,tag_idxs_is,comm,&recv_req_idxs_is[i]));
      ptr_idxs_is += olengths_idxs_is[i];
    }
    if (nvecs) {
      source_dest = onodes[i];
      PetscCallMPI(MPI_Irecv(ptr_vecs,olengths_idxs[i]-2,MPIU_SCALAR,source_dest,tag_vecs,comm,&recv_req_vecs[i]));
      ptr_vecs += olengths_idxs[i]-2;
    }
  }
  for (i=0;i<n_sends;i++) {
    PetscCall(PetscMPIIntCast(is_indices[i],&source_dest));
    PetscCallMPI(MPI_Isend(send_buffer_idxs,ilengths_idxs[source_dest],MPIU_INT,source_dest,tag_idxs,comm,&send_req_idxs[i]));
    PetscCallMPI(MPI_Isend((PetscScalar*)send_buffer_vals,ilengths_vals[source_dest],MPIU_SCALAR,source_dest,tag_vals,comm,&send_req_vals[i]));
    if (nis) {
      PetscCallMPI(MPI_Isend(send_buffer_idxs_is,ilengths_idxs_is[source_dest],MPIU_INT,source_dest,tag_idxs_is,comm,&send_req_idxs_is[i]));
    }
    if (nvecs) {
      PetscCall(VecGetArray(nnsp_vec[0],&send_buffer_vecs));
      PetscCallMPI(MPI_Isend(send_buffer_vecs,ilengths_idxs[source_dest]-2,MPIU_SCALAR,source_dest,tag_vecs,comm,&send_req_vecs[i]));
    }
  }
  PetscCall(ISRestoreIndices(is_sends_internal,&is_indices));
  PetscCall(ISDestroy(&is_sends_internal));

  /* assemble new l2g map */
  PetscCallMPI(MPI_Waitall(n_recvs,recv_req_idxs,MPI_STATUSES_IGNORE));
  ptr_idxs = recv_buffer_idxs;
  new_local_rows = 0;
  for (i=0;i<n_recvs;i++) {
    new_local_rows += *(ptr_idxs+1); /* second element is the local size of the l2gmap */
    ptr_idxs += olengths_idxs[i];
  }
  PetscCall(PetscMalloc1(new_local_rows,&l2gmap_indices));
  ptr_idxs = recv_buffer_idxs;
  new_local_rows = 0;
  for (i=0;i<n_recvs;i++) {
    PetscCall(PetscArraycpy(&l2gmap_indices[new_local_rows],ptr_idxs+2,*(ptr_idxs+1)));
    new_local_rows += *(ptr_idxs+1); /* second element is the local size of the l2gmap */
    ptr_idxs += olengths_idxs[i];
  }
  PetscCall(PetscSortRemoveDupsInt(&new_local_rows,l2gmap_indices));
  PetscCall(ISLocalToGlobalMappingCreate(comm_n,1,new_local_rows,l2gmap_indices,PETSC_COPY_VALUES,&l2gmap));
  PetscCall(PetscFree(l2gmap_indices));

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
    PetscCall(MatGetSize(mat,&rows,&cols));
    PetscCall(MatCreateIS(comm_n,bs,PETSC_DECIDE,PETSC_DECIDE,rows,cols,l2gmap,l2gmap,mat_n));
  } else {
    /* it also destroys the local matrices */
    if (*mat_n) {
      PetscCall(MatSetLocalToGlobalMapping(*mat_n,l2gmap,l2gmap));
    } else { /* this is a fake object */
      PetscCall(MatCreateIS(comm_n,bs,PETSC_DECIDE,PETSC_DECIDE,rows,cols,l2gmap,l2gmap,mat_n));
    }
  }
  PetscCall(MatISGetLocalMat(*mat_n,&local_mat));
  PetscCall(MatSetType(local_mat,new_local_type));

  PetscCallMPI(MPI_Waitall(n_recvs,recv_req_vals,MPI_STATUSES_IGNORE));

  /* Global to local map of received indices */
  PetscCall(PetscMalloc1(buf_size_idxs,&recv_buffer_idxs_local)); /* needed for values insertion */
  PetscCall(ISGlobalToLocalMappingApply(l2gmap,IS_GTOLM_MASK,buf_size_idxs,recv_buffer_idxs,&i,recv_buffer_idxs_local));
  PetscCall(ISLocalToGlobalMappingDestroy(&l2gmap));

  /* restore attributes -> type of incoming data and its size */
  buf_size_idxs = 0;
  for (i=0;i<n_recvs;i++) {
    recv_buffer_idxs_local[buf_size_idxs] = recv_buffer_idxs[buf_size_idxs];
    recv_buffer_idxs_local[buf_size_idxs+1] = recv_buffer_idxs[buf_size_idxs+1];
    buf_size_idxs += (PetscInt)olengths_idxs[i];
  }
  PetscCall(PetscFree(recv_buffer_idxs));

  /* set preallocation */
  PetscCall(PetscObjectTypeCompare((PetscObject)local_mat,MATSEQDENSE,&newisdense));
  if (!newisdense) {
    PetscInt *new_local_nnz=NULL;

    ptr_idxs = recv_buffer_idxs_local;
    if (n_recvs) {
      PetscCall(PetscCalloc1(new_local_rows,&new_local_nnz));
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
      PetscCall(MatSeqAIJSetPreallocation(local_mat,0,new_local_nnz));
      for (i=0;i<new_local_rows;i++) new_local_nnz[i] /= bs;
      PetscCall(MatSeqBAIJSetPreallocation(local_mat,bs,0,new_local_nnz));
      for (i=0;i<new_local_rows;i++) new_local_nnz[i] = PetscMax(new_local_nnz[i]-i,0);
      PetscCall(MatSeqSBAIJSetPreallocation(local_mat,bs,0,new_local_nnz));
    } else {
      PetscCall(MatSetUp(local_mat));
    }
    PetscCall(PetscFree(new_local_nnz));
  } else {
    PetscCall(MatSetUp(local_mat));
  }

  /* set values */
  ptr_vals = recv_buffer_vals;
  ptr_idxs = recv_buffer_idxs_local;
  for (i=0;i<n_recvs;i++) {
    if (*ptr_idxs == (PetscInt)MATDENSE_PRIVATE) { /* values insertion provided for dense case only */
      PetscCall(MatSetOption(local_mat,MAT_ROW_ORIENTED,PETSC_FALSE));
      PetscCall(MatSetValues(local_mat,*(ptr_idxs+1),ptr_idxs+2,*(ptr_idxs+1),ptr_idxs+2,ptr_vals,ADD_VALUES));
      PetscCall(MatAssemblyBegin(local_mat,MAT_FLUSH_ASSEMBLY));
      PetscCall(MatAssemblyEnd(local_mat,MAT_FLUSH_ASSEMBLY));
      PetscCall(MatSetOption(local_mat,MAT_ROW_ORIENTED,PETSC_TRUE));
    } else {
      /* TODO */
    }
    ptr_idxs += olengths_idxs[i];
    ptr_vals += olengths_vals[i];
  }
  PetscCall(MatAssemblyBegin(local_mat,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(local_mat,MAT_FINAL_ASSEMBLY));
  PetscCall(MatISRestoreLocalMat(*mat_n,&local_mat));
  PetscCall(MatAssemblyBegin(*mat_n,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*mat_n,MAT_FINAL_ASSEMBLY));
  PetscCall(PetscFree(recv_buffer_vals));

#if 0
  if (!restrict_comm) { /* check */
    Vec       lvec,rvec;
    PetscReal infty_error;

    PetscCall(MatCreateVecs(mat,&rvec,&lvec));
    PetscCall(VecSetRandom(rvec,NULL));
    PetscCall(MatMult(mat,rvec,lvec));
    PetscCall(VecScale(lvec,-1.0));
    PetscCall(MatMultAdd(*mat_n,rvec,lvec,lvec));
    PetscCall(VecNorm(lvec,NORM_INFINITY,&infty_error));
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)mat),"Infinity error subassembling %1.6e\n",infty_error));
    PetscCall(VecDestroy(&rvec));
    PetscCall(VecDestroy(&lvec));
  }
#endif

  /* assemble new additional is (if any) */
  if (nis) {
    PetscInt **temp_idxs,*count_is,j,psum;

    PetscCallMPI(MPI_Waitall(n_recvs,recv_req_idxs_is,MPI_STATUSES_IGNORE));
    PetscCall(PetscCalloc1(nis,&count_is));
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
    PetscCall(PetscMalloc1(nis,&temp_idxs));
    PetscCall(PetscMalloc1(psum,&temp_idxs[0]));
    for (i=1;i<nis;i++) {
      temp_idxs[i] = temp_idxs[i-1]+count_is[i-1];
    }
    PetscCall(PetscArrayzero(count_is,nis));
    ptr_idxs = recv_buffer_idxs_is;
    for (i=0;i<n_recvs;i++) {
      for (j=0;j<nis;j++) {
        PetscInt plen = *(ptr_idxs); /* first element is the local size of IS's indices */
        PetscCall(PetscArraycpy(&temp_idxs[j][count_is[j]],ptr_idxs+1,plen));
        count_is[j] += plen; /* increment starting point of buffer for j-th IS */
        ptr_idxs += plen+1; /* shift pointer to received data */
      }
    }
    for (i=0;i<nis;i++) {
      PetscCall(ISDestroy(&isarray[i]));
      PetscCall(PetscSortRemoveDupsInt(&count_is[i],temp_idxs[i]));
      PetscCall(ISCreateGeneral(comm_n,count_is[i],temp_idxs[i],PETSC_COPY_VALUES,&isarray[i]));
    }
    PetscCall(PetscFree(count_is));
    PetscCall(PetscFree(temp_idxs[0]));
    PetscCall(PetscFree(temp_idxs));
  }
  /* free workspace */
  PetscCall(PetscFree(recv_buffer_idxs_is));
  PetscCallMPI(MPI_Waitall(n_sends,send_req_idxs,MPI_STATUSES_IGNORE));
  PetscCall(PetscFree(send_buffer_idxs));
  PetscCallMPI(MPI_Waitall(n_sends,send_req_vals,MPI_STATUSES_IGNORE));
  if (isdense) {
    PetscCall(MatISGetLocalMat(mat,&local_mat));
    PetscCall(MatDenseRestoreArrayRead(local_mat,&send_buffer_vals));
    PetscCall(MatISRestoreLocalMat(mat,&local_mat));
  } else {
    /* PetscCall(PetscFree(send_buffer_vals)); */
  }
  if (nis) {
    PetscCallMPI(MPI_Waitall(n_sends,send_req_idxs_is,MPI_STATUSES_IGNORE));
    PetscCall(PetscFree(send_buffer_idxs_is));
  }

  if (nvecs) {
    PetscCallMPI(MPI_Waitall(n_recvs,recv_req_vecs,MPI_STATUSES_IGNORE));
    PetscCallMPI(MPI_Waitall(n_sends,send_req_vecs,MPI_STATUSES_IGNORE));
    PetscCall(VecRestoreArray(nnsp_vec[0],&send_buffer_vecs));
    PetscCall(VecDestroy(&nnsp_vec[0]));
    PetscCall(VecCreate(comm_n,&nnsp_vec[0]));
    PetscCall(VecSetSizes(nnsp_vec[0],new_local_rows,PETSC_DECIDE));
    PetscCall(VecSetType(nnsp_vec[0],VECSTANDARD));
    /* set values */
    ptr_vals = recv_buffer_vecs;
    ptr_idxs = recv_buffer_idxs_local;
    PetscCall(VecGetArray(nnsp_vec[0],&send_buffer_vecs));
    for (i=0;i<n_recvs;i++) {
      PetscInt j;
      for (j=0;j<*(ptr_idxs+1);j++) {
        send_buffer_vecs[*(ptr_idxs+2+j)] += *(ptr_vals + j);
      }
      ptr_idxs += olengths_idxs[i];
      ptr_vals += olengths_idxs[i]-2;
    }
    PetscCall(VecRestoreArray(nnsp_vec[0],&send_buffer_vecs));
    PetscCall(VecAssemblyBegin(nnsp_vec[0]));
    PetscCall(VecAssemblyEnd(nnsp_vec[0]));
  }

  PetscCall(PetscFree(recv_buffer_vecs));
  PetscCall(PetscFree(recv_buffer_idxs_local));
  PetscCall(PetscFree(recv_req_idxs));
  PetscCall(PetscFree(recv_req_vals));
  PetscCall(PetscFree(recv_req_vecs));
  PetscCall(PetscFree(recv_req_idxs_is));
  PetscCall(PetscFree(send_req_idxs));
  PetscCall(PetscFree(send_req_vals));
  PetscCall(PetscFree(send_req_vecs));
  PetscCall(PetscFree(send_req_idxs_is));
  PetscCall(PetscFree(ilengths_vals));
  PetscCall(PetscFree(ilengths_idxs));
  PetscCall(PetscFree(olengths_vals));
  PetscCall(PetscFree(olengths_idxs));
  PetscCall(PetscFree(onodes));
  if (nis) {
    PetscCall(PetscFree(ilengths_idxs_is));
    PetscCall(PetscFree(olengths_idxs_is));
    PetscCall(PetscFree(onodes_is));
  }
  PetscCall(PetscSubcommDestroy(&subcomm));
  if (destroy_mat) { /* destroy mat is true only if restrict comm is true and process will not participate */
    PetscCall(MatDestroy(mat_n));
    for (i=0;i<nis;i++) {
      PetscCall(ISDestroy(&isarray[i]));
    }
    if (nvecs) { /* need to match VecDestroy nnsp_vec called in the other code path */
      PetscCall(VecDestroy(&nnsp_vec[0]));
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

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(PC_BDDC_CoarseSetUp[pcbddc->current_level],pc,0,0,0));
  /* Assign global numbering to coarse dofs */
  if (pcbddc->new_primal_space || pcbddc->coarse_size == -1) { /* a new primal space is present or it is the first initialization, so recompute global numbering */
    PetscInt ocoarse_size;
    compute_vecs = PETSC_TRUE;

    pcbddc->new_primal_space = PETSC_TRUE;
    ocoarse_size = pcbddc->coarse_size;
    PetscCall(PetscFree(pcbddc->global_primal_indices));
    PetscCall(PCBDDCComputePrimalNumbering(pc,&pcbddc->coarse_size,&pcbddc->global_primal_indices));
    /* see if we can avoid some work */
    if (pcbddc->coarse_ksp) { /* coarse ksp has already been created */
      /* if the coarse size is different or we are using adaptive selection, better to not reuse the coarse matrix */
      if (ocoarse_size != pcbddc->coarse_size || pcbddc->adaptive_selection) {
        PetscCall(KSPReset(pcbddc->coarse_ksp));
        coarse_reuse = PETSC_FALSE;
      } else { /* we can safely reuse already computed coarse matrix */
        coarse_reuse = PETSC_TRUE;
      }
    } else { /* there's no coarse ksp, so we need to create the coarse matrix too */
      coarse_reuse = PETSC_FALSE;
    }
    /* reset any subassembling information */
    if (!coarse_reuse || pcbddc->recompute_topography) {
      PetscCall(ISDestroy(&pcbddc->coarse_subassembling));
    }
  } else { /* primal space is unchanged, so we can reuse coarse matrix */
    coarse_reuse = PETSC_TRUE;
  }
  if (coarse_reuse && pcbddc->coarse_ksp) {
    PetscCall(KSPGetOperators(pcbddc->coarse_ksp,&coarse_mat,NULL));
    PetscCall(PetscObjectReference((PetscObject)coarse_mat));
    coarse_mat_reuse = MAT_REUSE_MATRIX;
  } else {
    coarse_mat = NULL;
    coarse_mat_reuse = MAT_INITIAL_MATRIX;
  }

  /* creates temporary l2gmap and IS for coarse indexes */
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)pc),pcbddc->local_primal_size,pcbddc->global_primal_indices,PETSC_COPY_VALUES,&coarse_is));
  PetscCall(ISLocalToGlobalMappingCreateIS(coarse_is,&coarse_islg));

  /* creates temporary MATIS object for coarse matrix */
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,pcbddc->local_primal_size,pcbddc->local_primal_size,coarse_submat_vals,&coarse_submat_dense));
  PetscCall(MatCreateIS(PetscObjectComm((PetscObject)pc),1,PETSC_DECIDE,PETSC_DECIDE,pcbddc->coarse_size,pcbddc->coarse_size,coarse_islg,coarse_islg,&t_coarse_mat_is));
  PetscCall(MatISSetLocalMat(t_coarse_mat_is,coarse_submat_dense));
  PetscCall(MatAssemblyBegin(t_coarse_mat_is,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(t_coarse_mat_is,MAT_FINAL_ASSEMBLY));
  PetscCall(MatDestroy(&coarse_submat_dense));

  /* count "active" (i.e. with positive local size) and "void" processes */
  im_active = !!(pcis->n);
  PetscCall(MPIU_Allreduce(&im_active,&active_procs,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)pc)));

  /* determine number of processes partecipating to coarse solver and compute subassembling pattern */
  /* restr : whether we want to exclude senders (which are not receivers) from the subassembling pattern */
  /* full_restr : just use the receivers from the subassembling pattern */
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)pc),&size));
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
        PetscCall(PCBDDCMatISGetSubassemblingPattern(pc->pmat,&ncoarse,pcbddc->coarse_adj_red,&pcbddc->coarse_subassembling,&have_void));
      } else {
        PetscCall(PCBDDCMatISGetSubassemblingPattern(t_coarse_mat_is,&ncoarse,pcbddc->coarse_adj_red,&pcbddc->coarse_subassembling,&have_void));
      }
    } else {
      PetscMPIInt rank;

      PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)pc),&rank));
      have_void = (active_procs == (PetscInt)size) ? PETSC_FALSE : PETSC_TRUE;
      PetscCall(ISCreateStride(PetscObjectComm((PetscObject)pc),1,rank,1,&pcbddc->coarse_subassembling));
    }
  } else { /* if a subassembling pattern exists, then we can reuse the coarse ksp and compute the number of process involved */
    PetscInt    psum;
    if (pcbddc->coarse_ksp) psum = 1;
    else psum = 0;
    PetscCall(MPIU_Allreduce(&psum,&ncoarse,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)pc)));
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
    PetscCall(ISView(pcbddc->coarse_subassembling,pcbddc->dbg_viewer));
  }
  /* compute dofs splitting and neumann boundaries for coarse dofs */
  nedcfield = -1;
  corners = NULL;
  if (multilevel_allowed && !coarse_reuse && (pcbddc->n_ISForDofsLocal || pcbddc->NeumannBoundariesLocal || pcbddc->nedclocal || pcbddc->corner_selected)) { /* protects from unneeded computations */
    PetscInt               *tidxs,*tidxs2,nout,tsize,i;
    const PetscInt         *idxs;
    ISLocalToGlobalMapping tmap;

    /* create map between primal indices (in local representative ordering) and local primal numbering */
    PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_SELF,1,pcbddc->local_primal_size,pcbddc->primal_indices_local_idxs,PETSC_COPY_VALUES,&tmap));
    /* allocate space for temporary storage */
    PetscCall(PetscMalloc1(pcbddc->local_primal_size,&tidxs));
    PetscCall(PetscMalloc1(pcbddc->local_primal_size,&tidxs2));
    /* allocate for IS array */
    nisdofs = pcbddc->n_ISForDofsLocal;
    if (pcbddc->nedclocal) {
      if (pcbddc->nedfield > -1) {
        nedcfield = pcbddc->nedfield;
      } else {
        nedcfield = 0;
        PetscCheck(!nisdofs,PetscObjectComm((PetscObject)pc),PETSC_ERR_PLIB,"This should not happen (%" PetscInt_FMT ")",nisdofs);
        nisdofs = 1;
      }
    }
    nisneu = !!pcbddc->NeumannBoundariesLocal;
    nisvert = 0; /* nisvert is not used */
    nis = nisdofs + nisneu + nisvert;
    PetscCall(PetscMalloc1(nis,&isarray));
    /* dofs splitting */
    for (i=0;i<nisdofs;i++) {
      /* PetscCall(ISView(pcbddc->ISForDofsLocal[i],0)); */
      if (nedcfield != i) {
        PetscCall(ISGetLocalSize(pcbddc->ISForDofsLocal[i],&tsize));
        PetscCall(ISGetIndices(pcbddc->ISForDofsLocal[i],&idxs));
        PetscCall(ISGlobalToLocalMappingApply(tmap,IS_GTOLM_DROP,tsize,idxs,&nout,tidxs));
        PetscCall(ISRestoreIndices(pcbddc->ISForDofsLocal[i],&idxs));
      } else {
        PetscCall(ISGetLocalSize(pcbddc->nedclocal,&tsize));
        PetscCall(ISGetIndices(pcbddc->nedclocal,&idxs));
        PetscCall(ISGlobalToLocalMappingApply(tmap,IS_GTOLM_DROP,tsize,idxs,&nout,tidxs));
        PetscCheck(tsize == nout,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Failed when mapping coarse nedelec field! %" PetscInt_FMT " != %" PetscInt_FMT,tsize,nout);
        PetscCall(ISRestoreIndices(pcbddc->nedclocal,&idxs));
      }
      PetscCall(ISLocalToGlobalMappingApply(coarse_islg,nout,tidxs,tidxs2));
      PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)pc),nout,tidxs2,PETSC_COPY_VALUES,&isarray[i]));
      /* PetscCall(ISView(isarray[i],0)); */
    }
    /* neumann boundaries */
    if (pcbddc->NeumannBoundariesLocal) {
      /* PetscCall(ISView(pcbddc->NeumannBoundariesLocal,0)); */
      PetscCall(ISGetLocalSize(pcbddc->NeumannBoundariesLocal,&tsize));
      PetscCall(ISGetIndices(pcbddc->NeumannBoundariesLocal,&idxs));
      PetscCall(ISGlobalToLocalMappingApply(tmap,IS_GTOLM_DROP,tsize,idxs,&nout,tidxs));
      PetscCall(ISRestoreIndices(pcbddc->NeumannBoundariesLocal,&idxs));
      PetscCall(ISLocalToGlobalMappingApply(coarse_islg,nout,tidxs,tidxs2));
      PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)pc),nout,tidxs2,PETSC_COPY_VALUES,&isarray[nisdofs]));
      /* PetscCall(ISView(isarray[nisdofs],0)); */
    }
    /* coordinates */
    if (pcbddc->corner_selected) {
      PetscCall(PCBDDCGraphGetCandidatesIS(pcbddc->mat_graph,NULL,NULL,NULL,NULL,&corners));
      PetscCall(ISGetLocalSize(corners,&tsize));
      PetscCall(ISGetIndices(corners,&idxs));
      PetscCall(ISGlobalToLocalMappingApply(tmap,IS_GTOLM_DROP,tsize,idxs,&nout,tidxs));
      PetscCheck(tsize == nout,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Failed when mapping corners! %" PetscInt_FMT " != %" PetscInt_FMT,tsize,nout);
      PetscCall(ISRestoreIndices(corners,&idxs));
      PetscCall(PCBDDCGraphRestoreCandidatesIS(pcbddc->mat_graph,NULL,NULL,NULL,NULL,&corners));
      PetscCall(ISLocalToGlobalMappingApply(coarse_islg,nout,tidxs,tidxs2));
      PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)pc),nout,tidxs2,PETSC_COPY_VALUES,&corners));
    }
    PetscCall(PetscFree(tidxs));
    PetscCall(PetscFree(tidxs2));
    PetscCall(ISLocalToGlobalMappingDestroy(&tmap));
  } else {
    nis = 0;
    nisdofs = 0;
    nisneu = 0;
    nisvert = 0;
    isarray = NULL;
  }
  /* destroy no longer needed map */
  PetscCall(ISLocalToGlobalMappingDestroy(&coarse_islg));

  /* subassemble */
  if (multilevel_allowed) {
    Vec       vp[1];
    PetscInt  nvecs = 0;
    PetscBool reuse,reuser;

    if (coarse_mat) reuse = PETSC_TRUE;
    else reuse = PETSC_FALSE;
    PetscCall(MPIU_Allreduce(&reuse,&reuser,1,MPIU_BOOL,MPI_LOR,PetscObjectComm((PetscObject)pc)));
    vp[0] = NULL;
    if (pcbddc->benign_have_null) { /* propagate no-net-flux quadrature to coarser level */
      PetscCall(VecCreate(PetscObjectComm((PetscObject)pc),&vp[0]));
      PetscCall(VecSetSizes(vp[0],pcbddc->local_primal_size,PETSC_DECIDE));
      PetscCall(VecSetType(vp[0],VECSTANDARD));
      nvecs = 1;

      if (pcbddc->divudotp) {
        Mat      B,loc_divudotp;
        Vec      v,p;
        IS       dummy;
        PetscInt np;

        PetscCall(MatISGetLocalMat(pcbddc->divudotp,&loc_divudotp));
        PetscCall(MatGetSize(loc_divudotp,&np,NULL));
        PetscCall(ISCreateStride(PETSC_COMM_SELF,np,0,1,&dummy));
        PetscCall(MatCreateSubMatrix(loc_divudotp,dummy,pcis->is_B_local,MAT_INITIAL_MATRIX,&B));
        PetscCall(MatCreateVecs(B,&v,&p));
        PetscCall(VecSet(p,1.));
        PetscCall(MatMultTranspose(B,p,v));
        PetscCall(VecDestroy(&p));
        PetscCall(MatDestroy(&B));
        PetscCall(VecGetArray(vp[0],&array));
        PetscCall(VecPlaceArray(pcbddc->vec1_P,array));
        PetscCall(MatMultTranspose(pcbddc->coarse_phi_B,v,pcbddc->vec1_P));
        PetscCall(VecResetArray(pcbddc->vec1_P));
        PetscCall(VecRestoreArray(vp[0],&array));
        PetscCall(ISDestroy(&dummy));
        PetscCall(VecDestroy(&v));
      }
    }
    if (reuser) {
      PetscCall(PCBDDCMatISSubassemble(t_coarse_mat_is,pcbddc->coarse_subassembling,0,restr,full_restr,PETSC_TRUE,&coarse_mat,nis,isarray,nvecs,vp));
    } else {
      PetscCall(PCBDDCMatISSubassemble(t_coarse_mat_is,pcbddc->coarse_subassembling,0,restr,full_restr,PETSC_FALSE,&coarse_mat_is,nis,isarray,nvecs,vp));
    }
    if (vp[0]) { /* vp[0] could have been placed on a different set of processes */
      PetscScalar       *arraym;
      const PetscScalar *arrayv;
      PetscInt          nl;
      PetscCall(VecGetLocalSize(vp[0],&nl));
      PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,1,nl,NULL,&coarsedivudotp));
      PetscCall(MatDenseGetArray(coarsedivudotp,&arraym));
      PetscCall(VecGetArrayRead(vp[0],&arrayv));
      PetscCall(PetscArraycpy(arraym,arrayv,nl));
      PetscCall(VecRestoreArrayRead(vp[0],&arrayv));
      PetscCall(MatDenseRestoreArray(coarsedivudotp,&arraym));
      PetscCall(VecDestroy(&vp[0]));
    } else {
      PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF,0,0,1,NULL,&coarsedivudotp));
    }
  } else {
    PetscCall(PCBDDCMatISSubassemble(t_coarse_mat_is,pcbddc->coarse_subassembling,0,restr,full_restr,PETSC_FALSE,&coarse_mat_is,0,NULL,0,NULL));
  }
  if (coarse_mat_is || coarse_mat) {
    if (!multilevel_allowed) {
      PetscCall(MatConvert(coarse_mat_is,MATAIJ,coarse_mat_reuse,&coarse_mat));
    } else {
      /* if this matrix is present, it means we are not reusing the coarse matrix */
      if (coarse_mat_is) {
        PetscCheck(!coarse_mat,PetscObjectComm((PetscObject)coarse_mat_is),PETSC_ERR_PLIB,"This should not happen");
        PetscCall(PetscObjectReference((PetscObject)coarse_mat_is));
        coarse_mat = coarse_mat_is;
      }
    }
  }
  PetscCall(MatDestroy(&t_coarse_mat_is));
  PetscCall(MatDestroy(&coarse_mat_is));

  /* create local to global scatters for coarse problem */
  if (compute_vecs) {
    PetscInt lrows;
    PetscCall(VecDestroy(&pcbddc->coarse_vec));
    if (coarse_mat) {
      PetscCall(MatGetLocalSize(coarse_mat,&lrows,NULL));
    } else {
      lrows = 0;
    }
    PetscCall(VecCreate(PetscObjectComm((PetscObject)pc),&pcbddc->coarse_vec));
    PetscCall(VecSetSizes(pcbddc->coarse_vec,lrows,PETSC_DECIDE));
    PetscCall(VecSetType(pcbddc->coarse_vec,coarse_mat ? coarse_mat->defaultvectype : VECSTANDARD));
    PetscCall(VecScatterDestroy(&pcbddc->coarse_loc_to_glob));
    PetscCall(VecScatterCreate(pcbddc->vec1_P,NULL,pcbddc->coarse_vec,coarse_is,&pcbddc->coarse_loc_to_glob));
  }
  PetscCall(ISDestroy(&coarse_is));

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
      PetscCall(PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"--------------------------------------------------\n"));
      if (multilevel_requested) {
        PetscCall(PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"Not enough active processes on level %" PetscInt_FMT " (active processes %" PetscInt_FMT ", coarsening ratio %" PetscInt_FMT ")\n",pcbddc->current_level,active_procs,pcbddc->coarsening_ratio));
      } else if (pcbddc->max_levels) {
        PetscCall(PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"Maximum number of requested levels reached (%" PetscInt_FMT ")\n",pcbddc->max_levels));
      }
      PetscCall(PetscViewerFlush(pcbddc->dbg_viewer));
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
    PetscCall(MatMPIAIJRestrict(pcbddc->nedcG,ccomm,&coarseG));
  }

  /* create the coarse KSP object only once with defaults */
  if (coarse_mat) {
    PetscBool   isredundant,isbddc,force,valid;
    PetscViewer dbg_viewer = NULL;

    if (pcbddc->dbg_flag) {
      dbg_viewer = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)coarse_mat));
      PetscCall(PetscViewerASCIIAddTab(dbg_viewer,2*pcbddc->current_level));
    }
    if (!pcbddc->coarse_ksp) {
      char   prefix[256],str_level[16];
      size_t len;

      PetscCall(KSPCreate(PetscObjectComm((PetscObject)coarse_mat),&pcbddc->coarse_ksp));
      PetscCall(KSPSetErrorIfNotConverged(pcbddc->coarse_ksp,pc->erroriffailure));
      PetscCall(PetscObjectIncrementTabLevel((PetscObject)pcbddc->coarse_ksp,(PetscObject)pc,1));
      PetscCall(KSPSetTolerances(pcbddc->coarse_ksp,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,1));
      PetscCall(KSPSetOperators(pcbddc->coarse_ksp,coarse_mat,coarse_mat));
      PetscCall(KSPSetType(pcbddc->coarse_ksp,coarse_ksp_type));
      PetscCall(KSPSetNormType(pcbddc->coarse_ksp,KSP_NORM_NONE));
      PetscCall(KSPGetPC(pcbddc->coarse_ksp,&pc_temp));
      /* TODO is this logic correct? should check for coarse_mat type */
      PetscCall(PCSetType(pc_temp,coarse_pc_type));
      /* prefix */
      PetscCall(PetscStrcpy(prefix,""));
      PetscCall(PetscStrcpy(str_level,""));
      if (!pcbddc->current_level) {
        PetscCall(PetscStrncpy(prefix,((PetscObject)pc)->prefix,sizeof(prefix)));
        PetscCall(PetscStrlcat(prefix,"pc_bddc_coarse_",sizeof(prefix)));
      } else {
        PetscCall(PetscStrlen(((PetscObject)pc)->prefix,&len));
        if (pcbddc->current_level>1) len -= 3; /* remove "lX_" with X level number */
        if (pcbddc->current_level>10) len -= 1; /* remove another char from level number */
        /* Nonstandard use of PetscStrncpy() to copy only a portion of the string */
        PetscCall(PetscStrncpy(prefix,((PetscObject)pc)->prefix,len+1));
        PetscCall(PetscSNPrintf(str_level,sizeof(str_level),"l%d_",(int)(pcbddc->current_level)));
        PetscCall(PetscStrlcat(prefix,str_level,sizeof(prefix)));
      }
      PetscCall(KSPSetOptionsPrefix(pcbddc->coarse_ksp,prefix));
      /* propagate BDDC info to the next level (these are dummy calls if pc_temp is not of type PCBDDC) */
      PetscCall(PCBDDCSetLevel(pc_temp,pcbddc->current_level+1));
      PetscCall(PCBDDCSetCoarseningRatio(pc_temp,pcbddc->coarsening_ratio));
      PetscCall(PCBDDCSetLevels(pc_temp,pcbddc->max_levels));
      /* allow user customization */
      PetscCall(KSPSetFromOptions(pcbddc->coarse_ksp));
      /* get some info after set from options */
      PetscCall(KSPGetPC(pcbddc->coarse_ksp,&pc_temp));
      /* multilevel cannot be done with coarse PC different from BDDC, NN, HPDDM, unless forced to */
      force = PETSC_FALSE;
      PetscCall(PetscOptionsGetBool(NULL,((PetscObject)pc_temp)->prefix,"-pc_type_forced",&force,NULL));
      PetscCall(PetscObjectTypeCompareAny((PetscObject)pc_temp,&valid,PCBDDC,PCNN,PCHPDDM,""));
      PetscCall(PetscObjectTypeCompare((PetscObject)pc_temp,PCBDDC,&isbddc));
      if (multilevel_allowed && !force && !valid) {
        isbddc = PETSC_TRUE;
        PetscCall(PCSetType(pc_temp,PCBDDC));
        PetscCall(PCBDDCSetLevel(pc_temp,pcbddc->current_level+1));
        PetscCall(PCBDDCSetCoarseningRatio(pc_temp,pcbddc->coarsening_ratio));
        PetscCall(PCBDDCSetLevels(pc_temp,pcbddc->max_levels));
        if (pc_temp->ops->setfromoptions) { /* need to setfromoptions again, skipping the pc_type */
          PetscObjectOptionsBegin((PetscObject)pc_temp);
          PetscCall((*pc_temp->ops->setfromoptions)(PetscOptionsObject,pc_temp));
          PetscCall(PetscObjectProcessOptionsHandlers(PetscOptionsObject,(PetscObject)pc_temp));
          PetscOptionsEnd();
          pc_temp->setfromoptionscalled++;
        }
      }
    }
    /* propagate BDDC info to the next level (these are dummy calls if pc_temp is not of type PCBDDC) */
    PetscCall(KSPGetPC(pcbddc->coarse_ksp,&pc_temp));
    if (nisdofs) {
      PetscCall(PCBDDCSetDofsSplitting(pc_temp,nisdofs,isarray));
      for (i=0;i<nisdofs;i++) {
        PetscCall(ISDestroy(&isarray[i]));
      }
    }
    if (nisneu) {
      PetscCall(PCBDDCSetNeumannBoundaries(pc_temp,isarray[nisdofs]));
      PetscCall(ISDestroy(&isarray[nisdofs]));
    }
    if (nisvert) {
      PetscCall(PCBDDCSetPrimalVerticesIS(pc_temp,isarray[nis-1]));
      PetscCall(ISDestroy(&isarray[nis-1]));
    }
    if (coarseG) {
      PetscCall(PCBDDCSetDiscreteGradient(pc_temp,coarseG,1,nedcfield,PETSC_FALSE,PETSC_TRUE));
    }

    /* get some info after set from options */
    PetscCall(PetscObjectTypeCompare((PetscObject)pc_temp,PCBDDC,&isbddc));

    /* multilevel can only be requested via -pc_bddc_levels or PCBDDCSetLevels */
    if (isbddc && !multilevel_allowed) {
      PetscCall(PCSetType(pc_temp,coarse_pc_type));
    }
    /* multilevel cannot be done with coarse PC different from BDDC, NN, HPDDM, unless forced to */
    force = PETSC_FALSE;
    PetscCall(PetscOptionsGetBool(NULL,((PetscObject)pc_temp)->prefix,"-pc_type_forced",&force,NULL));
    PetscCall(PetscObjectTypeCompareAny((PetscObject)pc_temp,&valid,PCBDDC,PCNN,PCHPDDM,""));
    if (multilevel_requested && multilevel_allowed && !valid && !force) {
      PetscCall(PCSetType(pc_temp,PCBDDC));
    }
    PetscCall(PetscObjectTypeCompare((PetscObject)pc_temp,PCREDUNDANT,&isredundant));
    if (isredundant) {
      KSP inner_ksp;
      PC  inner_pc;

      PetscCall(PCRedundantGetKSP(pc_temp,&inner_ksp));
      PetscCall(KSPGetPC(inner_ksp,&inner_pc));
    }

    /* parameters which miss an API */
    PetscCall(PetscObjectTypeCompare((PetscObject)pc_temp,PCBDDC,&isbddc));
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

        PetscCall(MatGetSize(coarsedivudotp,&n,NULL));
        PetscCallMPI(MPI_Scan(&n,&st,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)coarse_mat)));
        st   = st-n;
        PetscCall(ISCreateStride(PetscObjectComm((PetscObject)coarse_mat),1,st,1,&row));
        PetscCall(MatISGetLocalToGlobalMapping(coarse_mat,&l2gmap,NULL));
        PetscCall(ISLocalToGlobalMappingGetSize(l2gmap,&n));
        PetscCall(ISLocalToGlobalMappingGetIndices(l2gmap,&gidxs));
        PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)coarse_mat),n,gidxs,PETSC_COPY_VALUES,&col));
        PetscCall(ISLocalToGlobalMappingRestoreIndices(l2gmap,&gidxs));
        PetscCall(ISLocalToGlobalMappingCreateIS(row,&rl2g));
        PetscCall(ISLocalToGlobalMappingCreateIS(col,&cl2g));
        PetscCall(ISGetSize(row,&M));
        PetscCall(MatGetSize(coarse_mat,&N,NULL));
        PetscCall(ISDestroy(&row));
        PetscCall(ISDestroy(&col));
        PetscCall(MatCreate(PetscObjectComm((PetscObject)coarse_mat),&coarsedivudotp_is));
        PetscCall(MatSetType(coarsedivudotp_is,MATIS));
        PetscCall(MatSetSizes(coarsedivudotp_is,PETSC_DECIDE,PETSC_DECIDE,M,N));
        PetscCall(MatSetLocalToGlobalMapping(coarsedivudotp_is,rl2g,cl2g));
        PetscCall(ISLocalToGlobalMappingDestroy(&rl2g));
        PetscCall(ISLocalToGlobalMappingDestroy(&cl2g));
        PetscCall(MatISSetLocalMat(coarsedivudotp_is,coarsedivudotp));
        PetscCall(MatDestroy(&coarsedivudotp));
        PetscCall(PCBDDCSetDivergenceMat(pc_temp,coarsedivudotp_is,PETSC_FALSE,NULL));
        PetscCall(MatDestroy(&coarsedivudotp_is));
        pcbddc_coarse->adaptive_userdefined = PETSC_TRUE;
        if (pcbddc->adaptive_threshold[0] == 0.0) pcbddc_coarse->deluxe_zerorows = PETSC_TRUE;
      }
    }

    /* propagate symmetry info of coarse matrix */
    PetscCall(MatSetOption(coarse_mat,MAT_STRUCTURALLY_SYMMETRIC,PETSC_TRUE));
    if (pc->pmat->symmetric_set) {
      PetscCall(MatSetOption(coarse_mat,MAT_SYMMETRIC,pc->pmat->symmetric));
    }
    if (pc->pmat->hermitian_set) {
      PetscCall(MatSetOption(coarse_mat,MAT_HERMITIAN,pc->pmat->hermitian));
    }
    if (pc->pmat->spd_set) {
      PetscCall(MatSetOption(coarse_mat,MAT_SPD,pc->pmat->spd));
    }
    if (pcbddc->benign_saddle_point && !pcbddc->benign_have_null) {
      PetscCall(MatSetOption(coarse_mat,MAT_SPD,PETSC_TRUE));
    }
    /* set operators */
    PetscCall(MatViewFromOptions(coarse_mat,(PetscObject)pc,"-pc_bddc_coarse_mat_view"));
    PetscCall(MatSetOptionsPrefix(coarse_mat,((PetscObject)pcbddc->coarse_ksp)->prefix));
    PetscCall(KSPSetOperators(pcbddc->coarse_ksp,coarse_mat,coarse_mat));
    if (pcbddc->dbg_flag) {
      PetscCall(PetscViewerASCIISubtractTab(dbg_viewer,2*pcbddc->current_level));
    }
  }
  PetscCall(MatDestroy(&coarseG));
  PetscCall(PetscFree(isarray));
#if 0
  {
    PetscViewer viewer;
    char filename[256];
    sprintf(filename,"coarse_mat_level%d.m",pcbddc->current_level);
    PetscCall(PetscViewerASCIIOpen(PetscObjectComm((PetscObject)coarse_mat),filename,&viewer));
    PetscCall(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_MATLAB));
    PetscCall(MatView(coarse_mat,viewer));
    PetscCall(PetscViewerPopFormat(viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }
#endif

  if (corners) {
    Vec            gv;
    IS             is;
    const PetscInt *idxs;
    PetscInt       i,d,N,n,cdim = pcbddc->mat_graph->cdim;
    PetscScalar    *coords;

    PetscCheck(pcbddc->mat_graph->cloc,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing local coordinates");
    PetscCall(VecGetSize(pcbddc->coarse_vec,&N));
    PetscCall(VecGetLocalSize(pcbddc->coarse_vec,&n));
    PetscCall(VecCreate(PetscObjectComm((PetscObject)pcbddc->coarse_vec),&gv));
    PetscCall(VecSetBlockSize(gv,cdim));
    PetscCall(VecSetSizes(gv,n*cdim,N*cdim));
    PetscCall(VecSetType(gv,VECSTANDARD));
    PetscCall(VecSetFromOptions(gv));
    PetscCall(VecSet(gv,PETSC_MAX_REAL)); /* we only propagate coordinates from vertices constraints */

    PetscCall(PCBDDCGraphGetCandidatesIS(pcbddc->mat_graph,NULL,NULL,NULL,NULL,&is));
    PetscCall(ISGetLocalSize(is,&n));
    PetscCall(ISGetIndices(is,&idxs));
    PetscCall(PetscMalloc1(n*cdim,&coords));
    for (i=0;i<n;i++) {
      for (d=0;d<cdim;d++) {
        coords[cdim*i+d] = pcbddc->mat_graph->coords[cdim*idxs[i]+d];
      }
    }
    PetscCall(ISRestoreIndices(is,&idxs));
    PetscCall(PCBDDCGraphRestoreCandidatesIS(pcbddc->mat_graph,NULL,NULL,NULL,NULL,&is));

    PetscCall(ISGetLocalSize(corners,&n));
    PetscCall(ISGetIndices(corners,&idxs));
    PetscCall(VecSetValuesBlocked(gv,n,idxs,coords,INSERT_VALUES));
    PetscCall(ISRestoreIndices(corners,&idxs));
    PetscCall(PetscFree(coords));
    PetscCall(VecAssemblyBegin(gv));
    PetscCall(VecAssemblyEnd(gv));
    PetscCall(VecGetArray(gv,&coords));
    if (pcbddc->coarse_ksp) {
      PC        coarse_pc;
      PetscBool isbddc;

      PetscCall(KSPGetPC(pcbddc->coarse_ksp,&coarse_pc));
      PetscCall(PetscObjectTypeCompare((PetscObject)coarse_pc,PCBDDC,&isbddc));
      if (isbddc) { /* coarse coordinates have PETSC_MAX_REAL, specific for BDDC */
        PetscReal *realcoords;

        PetscCall(VecGetLocalSize(gv,&n));
#if defined(PETSC_USE_COMPLEX)
        PetscCall(PetscMalloc1(n,&realcoords));
        for (i=0;i<n;i++) realcoords[i] = PetscRealPart(coords[i]);
#else
        realcoords = coords;
#endif
        PetscCall(PCSetCoordinates(coarse_pc,cdim,n/cdim,realcoords));
#if defined(PETSC_USE_COMPLEX)
        PetscCall(PetscFree(realcoords));
#endif
      }
    }
    PetscCall(VecRestoreArray(gv,&coords));
    PetscCall(VecDestroy(&gv));
  }
  PetscCall(ISDestroy(&corners));

  if (pcbddc->coarse_ksp) {
    Vec crhs,csol;

    PetscCall(KSPGetSolution(pcbddc->coarse_ksp,&csol));
    PetscCall(KSPGetRhs(pcbddc->coarse_ksp,&crhs));
    if (!csol) {
      PetscCall(MatCreateVecs(coarse_mat,&((pcbddc->coarse_ksp)->vec_sol),NULL));
    }
    if (!crhs) {
      PetscCall(MatCreateVecs(coarse_mat,NULL,&((pcbddc->coarse_ksp)->vec_rhs)));
    }
  }
  PetscCall(MatDestroy(&coarsedivudotp));

  /* compute null space for coarse solver if the benign trick has been requested */
  if (pcbddc->benign_null) {

    PetscCall(VecSet(pcbddc->vec1_P,0.));
    for (i=0;i<pcbddc->benign_n;i++) {
      PetscCall(VecSetValue(pcbddc->vec1_P,pcbddc->local_primal_size-pcbddc->benign_n+i,1.0,INSERT_VALUES));
    }
    PetscCall(VecAssemblyBegin(pcbddc->vec1_P));
    PetscCall(VecAssemblyEnd(pcbddc->vec1_P));
    PetscCall(VecScatterBegin(pcbddc->coarse_loc_to_glob,pcbddc->vec1_P,pcbddc->coarse_vec,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(pcbddc->coarse_loc_to_glob,pcbddc->vec1_P,pcbddc->coarse_vec,INSERT_VALUES,SCATTER_FORWARD));
    if (coarse_mat) {
      Vec         nullv;
      PetscScalar *array,*array2;
      PetscInt    nl;

      PetscCall(MatCreateVecs(coarse_mat,&nullv,NULL));
      PetscCall(VecGetLocalSize(nullv,&nl));
      PetscCall(VecGetArrayRead(pcbddc->coarse_vec,(const PetscScalar**)&array));
      PetscCall(VecGetArray(nullv,&array2));
      PetscCall(PetscArraycpy(array2,array,nl));
      PetscCall(VecRestoreArray(nullv,&array2));
      PetscCall(VecRestoreArrayRead(pcbddc->coarse_vec,(const PetscScalar**)&array));
      PetscCall(VecNormalize(nullv,NULL));
      PetscCall(MatNullSpaceCreate(PetscObjectComm((PetscObject)coarse_mat),PETSC_FALSE,1,&nullv,&CoarseNullSpace));
      PetscCall(VecDestroy(&nullv));
    }
  }
  PetscCall(PetscLogEventEnd(PC_BDDC_CoarseSetUp[pcbddc->current_level],pc,0,0,0));

  PetscCall(PetscLogEventBegin(PC_BDDC_CoarseSolver[pcbddc->current_level],pc,0,0,0));
  if (pcbddc->coarse_ksp) {
    PetscBool ispreonly;

    if (CoarseNullSpace) {
      PetscBool isnull;
      PetscCall(MatNullSpaceTest(CoarseNullSpace,coarse_mat,&isnull));
      if (isnull) {
        PetscCall(MatSetNullSpace(coarse_mat,CoarseNullSpace));
      }
      /* TODO: add local nullspaces (if any) */
    }
    /* setup coarse ksp */
    PetscCall(KSPSetUp(pcbddc->coarse_ksp));
    /* Check coarse problem if in debug mode or if solving with an iterative method */
    PetscCall(PetscObjectTypeCompare((PetscObject)pcbddc->coarse_ksp,KSPPREONLY,&ispreonly));
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
      PetscCall(KSPCreate(PetscObjectComm((PetscObject)pcbddc->coarse_ksp),&check_ksp));
      PetscCall(PetscObjectIncrementTabLevel((PetscObject)check_ksp,(PetscObject)pcbddc->coarse_ksp,0));
      PetscCall(KSPSetErrorIfNotConverged(pcbddc->coarse_ksp,PETSC_FALSE));
      PetscCall(KSPSetOperators(check_ksp,coarse_mat,coarse_mat));
      PetscCall(KSPSetTolerances(check_ksp,1.e-12,1.e-12,PETSC_DEFAULT,pcbddc->coarse_size));
      /* prevent from setup unneeded object */
      PetscCall(KSPGetPC(check_ksp,&check_pc));
      PetscCall(PCSetType(check_pc,PCNONE));
      if (ispreonly) {
        check_ksp_type = KSPPREONLY;
        compute_eigs = PETSC_FALSE;
      } else {
        check_ksp_type = KSPGMRES;
        compute_eigs = PETSC_TRUE;
      }
      PetscCall(KSPSetType(check_ksp,check_ksp_type));
      PetscCall(KSPSetComputeSingularValues(check_ksp,compute_eigs));
      PetscCall(KSPSetComputeEigenvalues(check_ksp,compute_eigs));
      PetscCall(KSPGMRESSetRestart(check_ksp,pcbddc->coarse_size+1));
      PetscCall(KSPGetOptionsPrefix(pcbddc->coarse_ksp,&prefix));
      PetscCall(KSPSetOptionsPrefix(check_ksp,prefix));
      PetscCall(KSPAppendOptionsPrefix(check_ksp,"check_"));
      PetscCall(KSPSetFromOptions(check_ksp));
      PetscCall(KSPSetUp(check_ksp));
      PetscCall(KSPGetPC(pcbddc->coarse_ksp,&check_pc));
      PetscCall(KSPSetPC(check_ksp,check_pc));
      /* create random vec */
      PetscCall(MatCreateVecs(coarse_mat,&coarse_vec,&check_vec));
      PetscCall(VecSetRandom(check_vec,NULL));
      PetscCall(MatMult(coarse_mat,check_vec,coarse_vec));
      /* solve coarse problem */
      PetscCall(KSPSolve(check_ksp,coarse_vec,coarse_vec));
      PetscCall(KSPCheckSolve(check_ksp,pc,coarse_vec));
      /* set eigenvalue estimation if preonly has not been requested */
      if (compute_eigs) {
        PetscCall(PetscMalloc1(pcbddc->coarse_size+1,&eigs_r));
        PetscCall(PetscMalloc1(pcbddc->coarse_size+1,&eigs_c));
        PetscCall(KSPComputeEigenvalues(check_ksp,pcbddc->coarse_size+1,eigs_r,eigs_c,&neigs));
        if (neigs) {
          lambda_max = eigs_r[neigs-1];
          lambda_min = eigs_r[0];
          if (pcbddc->use_coarse_estimates) {
            if (lambda_max>=lambda_min) { /* using PETSC_SMALL since lambda_max == lambda_min is not allowed by KSPChebyshevSetEigenvalues */
              PetscCall(KSPChebyshevSetEigenvalues(pcbddc->coarse_ksp,lambda_max+PETSC_SMALL,lambda_min));
              PetscCall(KSPRichardsonSetScale(pcbddc->coarse_ksp,2.0/(lambda_max+lambda_min)));
            }
          }
        }
      }

      /* check coarse problem residual error */
      if (pcbddc->dbg_flag) {
        PetscViewer dbg_viewer = PETSC_VIEWER_STDOUT_(PetscObjectComm((PetscObject)pcbddc->coarse_ksp));
        PetscCall(PetscViewerASCIIAddTab(dbg_viewer,2*(pcbddc->current_level+1)));
        PetscCall(VecAXPY(check_vec,-1.0,coarse_vec));
        PetscCall(VecNorm(check_vec,NORM_INFINITY,&infty_error));
        PetscCall(MatMult(coarse_mat,check_vec,coarse_vec));
        PetscCall(VecNorm(coarse_vec,NORM_INFINITY,&abs_infty_error));
        PetscCall(PetscViewerASCIIPrintf(dbg_viewer,"Coarse problem details (use estimates %d)\n",pcbddc->use_coarse_estimates));
        PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)(pcbddc->coarse_ksp),dbg_viewer));
        PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject)(check_pc),dbg_viewer));
        PetscCall(PetscViewerASCIIPrintf(dbg_viewer,"Coarse problem exact infty_error   : %1.6e\n",(double)infty_error));
        PetscCall(PetscViewerASCIIPrintf(dbg_viewer,"Coarse problem residual infty_error: %1.6e\n",(double)abs_infty_error));
        if (CoarseNullSpace) {
          PetscCall(PetscViewerASCIIPrintf(dbg_viewer,"Coarse problem is singular\n"));
        }
        if (compute_eigs) {
          PetscReal          lambda_max_s,lambda_min_s;
          KSPConvergedReason reason;
          PetscCall(KSPGetType(check_ksp,&check_ksp_type));
          PetscCall(KSPGetIterationNumber(check_ksp,&its));
          PetscCall(KSPGetConvergedReason(check_ksp,&reason));
          PetscCall(KSPComputeExtremeSingularValues(check_ksp,&lambda_max_s,&lambda_min_s));
          PetscCall(PetscViewerASCIIPrintf(dbg_viewer,"Coarse problem eigenvalues (estimated with %" PetscInt_FMT " iterations of %s, conv reason %d): %1.6e %1.6e (%1.6e %1.6e)\n",its,check_ksp_type,reason,(double)lambda_min,(double)lambda_max,(double)lambda_min_s,(double)lambda_max_s));
          for (i=0;i<neigs;i++) {
            PetscCall(PetscViewerASCIIPrintf(dbg_viewer,"%1.6e %1.6ei\n",(double)eigs_r[i],(double)eigs_c[i]));
          }
        }
        PetscCall(PetscViewerFlush(dbg_viewer));
        PetscCall(PetscViewerASCIISubtractTab(dbg_viewer,2*(pcbddc->current_level+1)));
      }
      PetscCall(VecDestroy(&check_vec));
      PetscCall(VecDestroy(&coarse_vec));
      PetscCall(KSPDestroy(&check_ksp));
      if (compute_eigs) {
        PetscCall(PetscFree(eigs_r));
        PetscCall(PetscFree(eigs_c));
      }
    }
  }
  PetscCall(MatNullSpaceDestroy(&CoarseNullSpace));
  /* print additional info */
  if (pcbddc->dbg_flag) {
    /* waits until all processes reaches this point */
    PetscCall(PetscBarrier((PetscObject)pc));
    PetscCall(PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"Coarse solver setup completed at level %" PetscInt_FMT "\n",pcbddc->current_level));
    PetscCall(PetscViewerFlush(pcbddc->dbg_viewer));
  }

  /* free memory */
  PetscCall(MatDestroy(&coarse_mat));
  PetscCall(PetscLogEventEnd(PC_BDDC_CoarseSolver[pcbddc->current_level],pc,0,0,0));
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
  PetscCheck(!pcbddc->local_primal_size || pcbddc->local_primal_ref_node,PETSC_COMM_SELF,PETSC_ERR_PLIB,"BDDC ConstraintsSetUp should be called first");
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)(pc->pmat)),pcbddc->local_primal_size_cc,pcbddc->local_primal_ref_node,PETSC_COPY_VALUES,&subset_n));
  PetscCall(ISLocalToGlobalMappingApplyIS(pcis->mapping,subset_n,&subset));
  PetscCall(ISDestroy(&subset_n));
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)(pc->pmat)),pcbddc->local_primal_size_cc,pcbddc->local_primal_ref_mult,PETSC_COPY_VALUES,&subset_mult));
  PetscCall(ISRenumber(subset,subset_mult,&coarse_size,&subset_n));
  PetscCall(ISDestroy(&subset));
  PetscCall(ISDestroy(&subset_mult));
  PetscCall(ISGetLocalSize(subset_n,&local_size));
  PetscCheck(local_size == pcbddc->local_primal_size,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Invalid number of local primal indices computed %" PetscInt_FMT " != %" PetscInt_FMT,local_size,pcbddc->local_primal_size);
  PetscCall(PetscMalloc1(local_size,&local_primal_indices));
  PetscCall(ISGetIndices(subset_n,&t_local_primal_indices));
  PetscCall(PetscArraycpy(local_primal_indices,t_local_primal_indices,local_size));
  PetscCall(ISRestoreIndices(subset_n,&t_local_primal_indices));
  PetscCall(ISDestroy(&subset_n));

  /* check numbering */
  if (pcbddc->dbg_flag) {
    PetscScalar coarsesum,*array,*array2;
    PetscInt    i;
    PetscBool   set_error = PETSC_FALSE,set_error_reduced = PETSC_FALSE;

    PetscCall(PetscViewerFlush(pcbddc->dbg_viewer));
    PetscCall(PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"--------------------------------------------------\n"));
    PetscCall(PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"Check coarse indices\n"));
    PetscCall(PetscViewerASCIIPushSynchronized(pcbddc->dbg_viewer));
    /* counter */
    PetscCall(VecSet(pcis->vec1_global,0.0));
    PetscCall(VecSet(pcis->vec1_N,1.0));
    PetscCall(VecScatterBegin(matis->rctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterEnd(matis->rctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterBegin(matis->rctx,pcis->vec1_global,pcis->vec2_N,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(matis->rctx,pcis->vec1_global,pcis->vec2_N,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecSet(pcis->vec1_N,0.0));
    for (i=0;i<pcbddc->local_primal_size;i++) {
      PetscCall(VecSetValue(pcis->vec1_N,pcbddc->primal_indices_local_idxs[i],1.0,INSERT_VALUES));
    }
    PetscCall(VecAssemblyBegin(pcis->vec1_N));
    PetscCall(VecAssemblyEnd(pcis->vec1_N));
    PetscCall(VecSet(pcis->vec1_global,0.0));
    PetscCall(VecScatterBegin(matis->rctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterEnd(matis->rctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterBegin(matis->rctx,pcis->vec1_global,pcis->vec1_N,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(matis->rctx,pcis->vec1_global,pcis->vec1_N,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecGetArray(pcis->vec1_N,&array));
    PetscCall(VecGetArray(pcis->vec2_N,&array2));
    for (i=0;i<pcis->n;i++) {
      if (array[i] != 0.0 && array[i] != array2[i]) {
        PetscInt owned = (PetscInt)PetscRealPart(array[i]),gi;
        PetscInt neigh = (PetscInt)PetscRealPart(array2[i]);
        set_error = PETSC_TRUE;
        PetscCall(ISLocalToGlobalMappingApply(pcis->mapping,1,&i,&gi));
        PetscCall(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d: local index %" PetscInt_FMT " (gid %" PetscInt_FMT ") owned by %" PetscInt_FMT " processes instead of %" PetscInt_FMT "!\n",PetscGlobalRank,i,gi,owned,neigh));
      }
    }
    PetscCall(VecRestoreArray(pcis->vec2_N,&array2));
    PetscCall(MPIU_Allreduce(&set_error,&set_error_reduced,1,MPIU_BOOL,MPI_LOR,PetscObjectComm((PetscObject)pc)));
    PetscCall(PetscViewerFlush(pcbddc->dbg_viewer));
    for (i=0;i<pcis->n;i++) {
      if (PetscRealPart(array[i]) > 0.0) array[i] = 1.0/PetscRealPart(array[i]);
    }
    PetscCall(VecRestoreArray(pcis->vec1_N,&array));
    PetscCall(VecSet(pcis->vec1_global,0.0));
    PetscCall(VecScatterBegin(matis->rctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterEnd(matis->rctx,pcis->vec1_N,pcis->vec1_global,ADD_VALUES,SCATTER_REVERSE));
    PetscCall(VecSum(pcis->vec1_global,&coarsesum));
    PetscCall(PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"Size of coarse problem is %" PetscInt_FMT " (%lf)\n",coarse_size,(double)PetscRealPart(coarsesum)));
    if (pcbddc->dbg_flag > 1 || set_error_reduced) {
      PetscInt *gidxs;

      PetscCall(PetscMalloc1(pcbddc->local_primal_size,&gidxs));
      PetscCall(ISLocalToGlobalMappingApply(pcis->mapping,pcbddc->local_primal_size,pcbddc->primal_indices_local_idxs,gidxs));
      PetscCall(PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"Distribution of local primal indices\n"));
      PetscCall(PetscViewerFlush(pcbddc->dbg_viewer));
      PetscCall(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d\n",PetscGlobalRank));
      for (i=0;i<pcbddc->local_primal_size;i++) {
        PetscCall(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"local_primal_indices[%" PetscInt_FMT "]=%" PetscInt_FMT " (%" PetscInt_FMT ",%" PetscInt_FMT ")\n",i,local_primal_indices[i],pcbddc->primal_indices_local_idxs[i],gidxs[i]));
      }
      PetscCall(PetscViewerFlush(pcbddc->dbg_viewer));
      PetscCall(PetscFree(gidxs));
    }
    PetscCall(PetscViewerFlush(pcbddc->dbg_viewer));
    PetscCall(PetscViewerASCIIPushSynchronized(pcbddc->dbg_viewer));
    PetscCheck(!set_error_reduced,PetscObjectComm((PetscObject)pc),PETSC_ERR_PLIB,"BDDC Numbering of coarse dofs failed");
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
  PetscCall(ISGetLocalSize(globalis,&lsize));
  PetscCall(PetscMalloc1(lsize,&vals));
  for (i=0;i<lsize;i++) vals[i] = 1.0;
  PetscCall(ISGetIndices(globalis,(const PetscInt**)&idxs));
  PetscCall(VecSet(gwork,0.0));
  PetscCall(VecSet(lwork,0.0));
  if (idxs) { /* multilevel guard */
    PetscCall(VecSetOption(gwork,VEC_IGNORE_NEGATIVE_INDICES,PETSC_TRUE));
    PetscCall(VecSetValues(gwork,lsize,idxs,vals,INSERT_VALUES));
  }
  PetscCall(VecAssemblyBegin(gwork));
  PetscCall(ISRestoreIndices(globalis,(const PetscInt**)&idxs));
  PetscCall(PetscFree(vals));
  PetscCall(VecAssemblyEnd(gwork));
  /* now compute set in local ordering */
  PetscCall(VecScatterBegin(g2l_ctx,gwork,lwork,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(g2l_ctx,gwork,lwork,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecGetArrayRead(lwork,(const PetscScalar**)&vals));
  PetscCall(VecGetSize(lwork,&n));
  for (i=0,lsize=0;i<n;i++) {
    if (PetscRealPart(vals[i]) > 0.5) {
      lsize++;
    }
  }
  PetscCall(PetscMalloc1(lsize,&idxs));
  for (i=0,lsize=0;i<n;i++) {
    if (PetscRealPart(vals[i]) > 0.5) {
      idxs[lsize++] = i;
    }
  }
  PetscCall(VecRestoreArrayRead(lwork,(const PetscScalar**)&vals));
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)gwork),lsize,idxs,PETSC_OWN_POINTER,&localis_t));
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

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(PC_BDDC_Schurs[pcbddc->current_level],pc,0,0,0));
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

      PetscCall(MatGetRowIJ(pcbddc->local_mat,0,PETSC_TRUE,PETSC_FALSE,&nvtxs,&xadj,&adjncy,&flg_row));
      if (flg_row) {
        PetscCall(PetscMalloc2(nvtxs+1,&used_xadj,xadj[nvtxs],&used_adjncy));
        PetscCall(PetscArraycpy(used_xadj,xadj,nvtxs+1));
        PetscCall(PetscArraycpy(used_adjncy,adjncy,xadj[nvtxs]));
        free_used_adj = PETSC_TRUE;
      } else {
        pcbddc->sub_schurs_layers = -1;
        used_xadj = NULL;
        used_adjncy = NULL;
      }
      PetscCall(MatRestoreRowIJ(pcbddc->local_mat,0,PETSC_TRUE,PETSC_FALSE,&nvtxs,&xadj,&adjncy,&flg_row));
    }
  }

  /* setup sub_schurs data */
  PetscCall(MatCreateSchurComplement(pcis->A_II,pcis->pA_II,pcis->A_IB,pcis->A_BI,pcis->A_BB,&S_j));
  if (!sub_schurs->schur_explicit) {
    /* pcbddc->ksp_D up to date only if not using MatFactor with Schur complement support */
    PetscCall(MatSchurComplementSetKSP(S_j,pcbddc->ksp_D));
    PetscCall(PCBDDCSubSchursSetUp(sub_schurs,NULL,S_j,PETSC_FALSE,used_xadj,used_adjncy,pcbddc->sub_schurs_layers,NULL,pcbddc->adaptive_selection,PETSC_FALSE,PETSC_FALSE,0,NULL,NULL,NULL,NULL));
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

      PetscCall(ISGetLocalSize(sub_schurs->is_vertices,&n_vertices));
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
      PetscCall(MPIU_Allreduce(&have_loc_change,&need_change,1,MPIU_BOOL,MPI_LOR,PetscObjectComm((PetscObject)pc)));
      need_change = (PetscBool)(!need_change);
    }
    /* If the user defines additional constraints, we import them here.
       We need to compute the change of basis according to the quadrature weights attached to pmat via MatSetNearNullSpace, and this could not be done (at the moment) without some hacking */
    if (need_change) {
      PC_IS   *pcisf;
      PC_BDDC *pcbddcf;
      PC      pcf;

      PetscCheck(!pcbddc->sub_schurs_rebuild,PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot compute change of basis with a different graph");
      PetscCall(PCCreate(PetscObjectComm((PetscObject)pc),&pcf));
      PetscCall(PCSetOperators(pcf,pc->mat,pc->pmat));
      PetscCall(PCSetType(pcf,PCBDDC));

      /* hacks */
      pcisf                        = (PC_IS*)pcf->data;
      pcisf->is_B_local            = pcis->is_B_local;
      pcisf->vec1_N                = pcis->vec1_N;
      pcisf->BtoNmap               = pcis->BtoNmap;
      pcisf->n                     = pcis->n;
      pcisf->n_B                   = pcis->n_B;
      pcbddcf                      = (PC_BDDC*)pcf->data;
      PetscCall(PetscFree(pcbddcf->mat_graph));
      pcbddcf->mat_graph           = pcbddc->mat_graph;
      pcbddcf->use_faces           = PETSC_TRUE;
      pcbddcf->use_change_of_basis = PETSC_TRUE;
      pcbddcf->use_change_on_faces = PETSC_TRUE;
      pcbddcf->use_qr_single       = PETSC_TRUE;
      pcbddcf->fake_change         = PETSC_TRUE;

      /* setup constraints so that we can get information on primal vertices and change of basis (in local numbering) */
      PetscCall(PCBDDCConstraintsSetUp(pcf));
      sub_schurs->change_with_qr = pcbddcf->use_qr_single;
      PetscCall(ISCreateGeneral(PETSC_COMM_SELF,pcbddcf->n_vertices,pcbddcf->local_primal_ref_node,PETSC_COPY_VALUES,&change_primal));
      change = pcbddcf->ConstraintMatrix;
      pcbddcf->ConstraintMatrix = NULL;

      /* free unneeded memory allocated in PCBDDCConstraintsSetUp */
      PetscCall(PetscFree(pcbddcf->sub_schurs));
      PetscCall(MatNullSpaceDestroy(&pcbddcf->onearnullspace));
      PetscCall(PetscFree2(pcbddcf->local_primal_ref_node,pcbddcf->local_primal_ref_mult));
      PetscCall(PetscFree(pcbddcf->primal_indices_local_idxs));
      PetscCall(PetscFree(pcbddcf->onearnullvecs_state));
      PetscCall(PetscFree(pcf->data));
      pcf->ops->destroy = NULL;
      pcf->ops->reset   = NULL;
      PetscCall(PCDestroy(&pcf));
    }
    if (!pcbddc->use_deluxe_scaling) scaling = pcis->D;

    PetscCall(PetscObjectQuery((PetscObject)pc,"__KSPFETIDP_iP",(PetscObject*)&iP));
    if (iP) {
      PetscOptionsBegin(PetscObjectComm((PetscObject)iP),sub_schurs->prefix,"BDDC sub_schurs options","PC");
      PetscCall(PetscOptionsBool("-sub_schurs_discrete_harmonic",NULL,NULL,discrete_harmonic,&discrete_harmonic,NULL));
      PetscOptionsEnd();
    }
    if (discrete_harmonic) {
      Mat A;
      PetscCall(MatDuplicate(pcbddc->local_mat,MAT_COPY_VALUES,&A));
      PetscCall(MatZeroRowsColumnsIS(A,iP,1.0,NULL,NULL));
      PetscCall(PetscObjectCompose((PetscObject)A,"__KSPFETIDP_iP",(PetscObject)iP));
      PetscCall(PCBDDCSubSchursSetUp(sub_schurs,A,S_j,pcbddc->sub_schurs_exact_schur,used_xadj,used_adjncy,pcbddc->sub_schurs_layers,scaling,pcbddc->adaptive_selection,reuse_solvers,pcbddc->benign_saddle_point,benign_n,pcbddc->benign_p0_lidx,pcbddc->benign_zerodiag_subs,change,change_primal));
      PetscCall(MatDestroy(&A));
    } else {
      PetscCall(PCBDDCSubSchursSetUp(sub_schurs,pcbddc->local_mat,S_j,pcbddc->sub_schurs_exact_schur,used_xadj,used_adjncy,pcbddc->sub_schurs_layers,scaling,pcbddc->adaptive_selection,reuse_solvers,pcbddc->benign_saddle_point,benign_n,pcbddc->benign_p0_lidx,pcbddc->benign_zerodiag_subs,change,change_primal));
    }
    PetscCall(MatDestroy(&change));
    PetscCall(ISDestroy(&change_primal));
  }
  PetscCall(MatDestroy(&S_j));

  /* free adjacency */
  if (free_used_adj) {
    PetscCall(PetscFree2(used_xadj,used_adjncy));
  }
  PetscCall(PetscLogEventEnd(PC_BDDC_Schurs[pcbddc->current_level],pc,0,0,0));
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

    PetscCall(PCBDDCGraphGetCandidatesIS(pcbddc->mat_graph,NULL,NULL,NULL,NULL,&verticesIS));
    PetscCall(ISGetSize(verticesIS,&vsize));
    PetscCall(ISGetIndices(verticesIS,(const PetscInt**)&idxs));
    PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)pc),vsize,idxs,PETSC_COPY_VALUES,&verticescomm));
    PetscCall(ISRestoreIndices(verticesIS,(const PetscInt**)&idxs));
    PetscCall(PCBDDCGraphRestoreCandidatesIS(pcbddc->mat_graph,NULL,NULL,NULL,NULL,&verticesIS));
    PetscCall(PCBDDCGraphCreate(&graph));
    PetscCall(PCBDDCGraphInit(graph,pcbddc->mat_graph->l2gmap,pcbddc->mat_graph->nvtxs_global,pcbddc->graphmaxcount));
    PetscCall(PCBDDCGraphSetUp(graph,pcbddc->mat_graph->custom_minimal_size,NULL,pcbddc->DirichletBoundariesLocal,0,NULL,verticescomm));
    PetscCall(ISDestroy(&verticescomm));
    PetscCall(PCBDDCGraphComputeConnectedComponents(graph));
  } else {
    graph = pcbddc->mat_graph;
  }
  /* print some info */
  if (pcbddc->dbg_flag && !pcbddc->sub_schurs_rebuild) {
    IS       vertices;
    PetscInt nv,nedges,nfaces;
    PetscCall(PCBDDCGraphASCIIView(graph,pcbddc->dbg_flag,pcbddc->dbg_viewer));
    PetscCall(PCBDDCGraphGetCandidatesIS(graph,&nfaces,NULL,&nedges,NULL,&vertices));
    PetscCall(ISGetSize(vertices,&nv));
    PetscCall(PetscViewerASCIIPushSynchronized(pcbddc->dbg_viewer));
    PetscCall(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"--------------------------------------------------------------\n"));
    PetscCall(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d got %02" PetscInt_FMT " local candidate vertices (%d)\n",PetscGlobalRank,nv,pcbddc->use_vertices));
    PetscCall(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d got %02" PetscInt_FMT " local candidate edges    (%d)\n",PetscGlobalRank,nedges,pcbddc->use_edges));
    PetscCall(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d got %02" PetscInt_FMT " local candidate faces    (%d)\n",PetscGlobalRank,nfaces,pcbddc->use_faces));
    PetscCall(PetscViewerFlush(pcbddc->dbg_viewer));
    PetscCall(PetscViewerASCIIPopSynchronized(pcbddc->dbg_viewer));
    PetscCall(PCBDDCGraphRestoreCandidatesIS(graph,&nfaces,NULL,&nedges,NULL,&vertices));
  }

  /* sub_schurs init */
  if (!pcbddc->sub_schurs) {
    PetscCall(PCBDDCSubSchursCreate(&pcbddc->sub_schurs));
  }
  PetscCall(PCBDDCSubSchursInit(pcbddc->sub_schurs,((PetscObject)pc)->prefix,pcis->is_I_local,pcis->is_B_local,graph,pcis->BtoNmap,pcbddc->sub_schurs_rebuild));

  /* free graph struct */
  if (pcbddc->sub_schurs_rebuild) {
    PetscCall(PCBDDCGraphDestroy(&graph));
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

      PetscCall(ISCreateStride(PETSC_COMM_SELF,pcbddc->benign_n,0,1,&dummy));
      PetscCall(MatCreateSubMatrix(pcbddc->benign_B0,dummy,pcis->is_B_local,MAT_INITIAL_MATRIX,&B0_B));
      PetscCall(MatCreateVecs(B0_B,NULL,&dummy_vec));
      PetscCall(ISDestroy(&dummy));
    }
    /* I need a primal vector to scale primal nodes since BDDC sums contibutions */
    PetscCall(VecDuplicate(pcbddc->vec1_P,&vec_scale_P));
    PetscCall(VecSet(pcbddc->vec1_P,1.0));
    PetscCall(VecScatterBegin(pcbddc->coarse_loc_to_glob,pcbddc->vec1_P,pcbddc->coarse_vec,ADD_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(pcbddc->coarse_loc_to_glob,pcbddc->vec1_P,pcbddc->coarse_vec,ADD_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterBegin(pcbddc->coarse_loc_to_glob,pcbddc->coarse_vec,vec_scale_P,INSERT_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterEnd(pcbddc->coarse_loc_to_glob,pcbddc->coarse_vec,vec_scale_P,INSERT_VALUES,SCATTER_REVERSE));
    PetscCall(VecReciprocal(vec_scale_P));
    /* S_j */
    PetscCall(MatCreateSchurComplement(pcis->A_II,pcis->pA_II,pcis->A_IB,pcis->A_BI,pcis->A_BB,&S_j));
    PetscCall(MatSchurComplementSetKSP(S_j,pcbddc->ksp_D));

    /* mimic vector in \widetilde{W}_\Gamma */
    PetscCall(VecSetRandom(pcis->vec1_N,NULL));
    /* continuous in primal space */
    PetscCall(VecSetRandom(pcbddc->coarse_vec,NULL));
    PetscCall(VecScatterBegin(pcbddc->coarse_loc_to_glob,pcbddc->coarse_vec,pcbddc->vec1_P,INSERT_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterEnd(pcbddc->coarse_loc_to_glob,pcbddc->coarse_vec,pcbddc->vec1_P,INSERT_VALUES,SCATTER_REVERSE));
    PetscCall(VecGetArray(pcbddc->vec1_P,&array));
    PetscCall(PetscCalloc1(pcbddc->benign_n,&p0_check));
    for (i=0;i<pcbddc->benign_n;i++) p0_check[i] = array[pcbddc->local_primal_size-pcbddc->benign_n+i];
    PetscCall(VecSetValues(pcis->vec1_N,pcbddc->local_primal_size,pcbddc->local_primal_ref_node,array,INSERT_VALUES));
    PetscCall(VecRestoreArray(pcbddc->vec1_P,&array));
    PetscCall(VecAssemblyBegin(pcis->vec1_N));
    PetscCall(VecAssemblyEnd(pcis->vec1_N));
    PetscCall(VecScatterBegin(pcis->N_to_B,pcis->vec1_N,pcis->vec2_B,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(pcis->N_to_B,pcis->vec1_N,pcis->vec2_B,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecDuplicate(pcis->vec2_B,&vec_check_B));
    PetscCall(VecCopy(pcis->vec2_B,vec_check_B));

    /* assemble rhs for coarse problem */
    /* widetilde{S}_\Gamma w_\Gamma + \widetilde{B0}^T_B p0 */
    /* local with Schur */
    PetscCall(MatMult(S_j,pcis->vec2_B,pcis->vec1_B));
    if (zerodiag) {
      PetscCall(VecGetArray(dummy_vec,&array));
      for (i=0;i<pcbddc->benign_n;i++) array[i] = p0_check[i];
      PetscCall(VecRestoreArray(dummy_vec,&array));
      PetscCall(MatMultTransposeAdd(B0_B,dummy_vec,pcis->vec1_B,pcis->vec1_B));
    }
    /* sum on primal nodes the local contributions */
    PetscCall(VecScatterBegin(pcis->N_to_B,pcis->vec1_B,pcis->vec1_N,INSERT_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterEnd(pcis->N_to_B,pcis->vec1_B,pcis->vec1_N,INSERT_VALUES,SCATTER_REVERSE));
    PetscCall(VecGetArray(pcis->vec1_N,&array));
    PetscCall(VecGetArray(pcbddc->vec1_P,&array2));
    for (i=0;i<pcbddc->local_primal_size;i++) array2[i] = array[pcbddc->local_primal_ref_node[i]];
    PetscCall(VecRestoreArray(pcbddc->vec1_P,&array2));
    PetscCall(VecRestoreArray(pcis->vec1_N,&array));
    PetscCall(VecSet(pcbddc->coarse_vec,0.));
    PetscCall(VecScatterBegin(pcbddc->coarse_loc_to_glob,pcbddc->vec1_P,pcbddc->coarse_vec,ADD_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(pcbddc->coarse_loc_to_glob,pcbddc->vec1_P,pcbddc->coarse_vec,ADD_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterBegin(pcbddc->coarse_loc_to_glob,pcbddc->coarse_vec,pcbddc->vec1_P,INSERT_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterEnd(pcbddc->coarse_loc_to_glob,pcbddc->coarse_vec,pcbddc->vec1_P,INSERT_VALUES,SCATTER_REVERSE));
    PetscCall(VecGetArray(pcbddc->vec1_P,&array));
    /* scale primal nodes (BDDC sums contibutions) */
    PetscCall(VecPointwiseMult(pcbddc->vec1_P,vec_scale_P,pcbddc->vec1_P));
    PetscCall(VecSetValues(pcis->vec1_N,pcbddc->local_primal_size,pcbddc->local_primal_ref_node,array,INSERT_VALUES));
    PetscCall(VecRestoreArray(pcbddc->vec1_P,&array));
    PetscCall(VecAssemblyBegin(pcis->vec1_N));
    PetscCall(VecAssemblyEnd(pcis->vec1_N));
    PetscCall(VecScatterBegin(pcis->N_to_B,pcis->vec1_N,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(pcis->N_to_B,pcis->vec1_N,pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD));
    /* global: \widetilde{B0}_B w_\Gamma */
    if (zerodiag) {
      PetscCall(MatMult(B0_B,pcis->vec2_B,dummy_vec));
      PetscCall(VecGetArray(dummy_vec,&array));
      for (i=0;i<pcbddc->benign_n;i++) pcbddc->benign_p0[i] = array[i];
      PetscCall(VecRestoreArray(dummy_vec,&array));
    }
    /* BDDC */
    PetscCall(VecSet(pcis->vec1_D,0.));
    PetscCall(PCBDDCApplyInterfacePreconditioner(pc,PETSC_FALSE));

    PetscCall(VecCopy(pcis->vec1_B,pcis->vec2_B));
    PetscCall(VecAXPY(pcis->vec1_B,-1.0,vec_check_B));
    PetscCall(VecNorm(pcis->vec1_B,NORM_INFINITY,&norm));
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"[%d] BDDC local error is %1.4e\n",PetscGlobalRank,(double)norm));
    for (i=0;i<pcbddc->benign_n;i++) {
      PetscCall(PetscPrintf(PETSC_COMM_SELF,"[%d] BDDC p0[%" PetscInt_FMT "] error is %1.4e\n",PetscGlobalRank,i,(double)PetscAbsScalar(pcbddc->benign_p0[i]-p0_check[i])));
    }
    PetscCall(PetscFree(p0_check));
    PetscCall(VecDestroy(&vec_scale_P));
    PetscCall(VecDestroy(&vec_check_B));
    PetscCall(VecDestroy(&dummy_vec));
    PetscCall(MatDestroy(&S_j));
    PetscCall(MatDestroy(&B0_B));
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
    PetscCall(PetscLayoutCreate(ccomm,&rmap));
    PetscCall(PetscLayoutSetSize(rmap,A->rmap->N));
    PetscCall(PetscLayoutSetBlockSize(rmap,1));
    PetscCall(PetscLayoutSetUp(rmap));
    PetscCall(PetscLayoutGetRange(rmap,&rst,&ren));
  }
  PetscCall(ISCreateStride(PetscObjectComm((PetscObject)A),ren-rst,rst,1,&rows));
  PetscCall(MatCreateSubMatrix(A,rows,NULL,MAT_INITIAL_MATRIX,&At));
  PetscCall(ISDestroy(&rows));

  if (ccomm != MPI_COMM_NULL) {
    Mat_MPIAIJ *a,*b;
    IS         from,to;
    Vec        gvec;
    PetscInt   lsize;

    PetscCall(MatCreate(ccomm,B));
    PetscCall(MatSetSizes(*B,ren-rst,PETSC_DECIDE,PETSC_DECIDE,At->cmap->N));
    PetscCall(MatSetType(*B,MATAIJ));
    PetscCall(PetscLayoutDestroy(&((*B)->rmap)));
    PetscCall(PetscLayoutSetUp((*B)->cmap));
    a    = (Mat_MPIAIJ*)At->data;
    b    = (Mat_MPIAIJ*)(*B)->data;
    PetscCallMPI(MPI_Comm_size(ccomm,&b->size));
    PetscCallMPI(MPI_Comm_rank(ccomm,&b->rank));
    PetscCall(PetscObjectReference((PetscObject)a->A));
    PetscCall(PetscObjectReference((PetscObject)a->B));
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
      PetscCall(PetscTableCreateCopy(a->colmap,&b->colmap));
#else
      PetscCall(PetscMalloc1(At->cmap->N,&b->colmap));
      PetscCall(PetscLogObjectMemory((PetscObject)*B,At->cmap->N*sizeof(PetscInt)));
      PetscCall(PetscArraycpy(b->colmap,a->colmap,At->cmap->N));
#endif
    } else b->colmap = NULL;
    if (a->garray) {
      PetscInt len;
      len  = a->B->cmap->n;
      PetscCall(PetscMalloc1(len+1,&b->garray));
      PetscCall(PetscLogObjectMemory((PetscObject)(*B),len*sizeof(PetscInt)));
      if (len) PetscCall(PetscArraycpy(b->garray,a->garray,len));
    } else b->garray = NULL;

    PetscCall(PetscObjectReference((PetscObject)a->lvec));
    b->lvec = a->lvec;
    PetscCall(PetscLogObjectParent((PetscObject)*B,(PetscObject)b->lvec));

    /* cannot use VecScatterCopy */
    PetscCall(VecGetLocalSize(b->lvec,&lsize));
    PetscCall(ISCreateGeneral(ccomm,lsize,b->garray,PETSC_USE_POINTER,&from));
    PetscCall(ISCreateStride(PETSC_COMM_SELF,lsize,0,1,&to));
    PetscCall(MatCreateVecs(*B,&gvec,NULL));
    PetscCall(VecScatterCreate(gvec,from,b->lvec,to,&b->Mvctx));
    PetscCall(PetscLogObjectParent((PetscObject)*B,(PetscObject)b->Mvctx));
    PetscCall(ISDestroy(&from));
    PetscCall(ISDestroy(&to));
    PetscCall(VecDestroy(&gvec));
  }
  PetscCall(MatDestroy(&At));
  PetscFunctionReturn(0);
}
