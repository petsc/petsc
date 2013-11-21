#include <petsc-private/dmmbimpl.h> /*I  "petscdm.h"   I*/
#include <petsc-private/vecimpl.h> /*I  "petscdm.h"   I*/

#include <petscdmmoab.h>
#include <MBTagConventions.hpp>

static PetscErrorCode DMMoab_Compute_NNZ_From_Connectivity(DM,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscBool);
static PetscErrorCode DMMoab_MatFillMatrixEntries_Private(DM,Mat);

#undef __FUNCT__
#define __FUNCT__ "DMCreateMatrix_Moab"
PetscErrorCode DMCreateMatrix_Moab(DM dm,Mat *J)
{
  PetscErrorCode  ierr;
  ISLocalToGlobalMapping ltogb;
  PetscInt        innz,ionz,nlsiz;
  DM_Moab         *dmmoab=(DM_Moab*)dm->data;
  PetscInt        *nnz=0,*onz=0;
  char            *tmp=0;
  MatType         mtype;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(J,3);

  /* next, need to allocate the non-zero arrays to enable pre-allocation */
  mtype = dm->mattype;
  ierr = PetscStrstr(mtype, "baij", &tmp);CHKERRQ(ierr);
  nlsiz = (tmp ? dmmoab->nloc:dmmoab->nloc*dmmoab->bs);

  /* allocate the nnz, onz arrays based on block size and local nodes */
  ierr = PetscMalloc((nlsiz)*sizeof(PetscInt),&nnz);CHKERRQ(ierr);
  ierr = PetscMemzero(nnz,sizeof(PetscInt)*(nlsiz));CHKERRQ(ierr);
  ierr = PetscMalloc(nlsiz*sizeof(PetscInt),&onz);CHKERRQ(ierr);
  ierr = PetscMemzero(onz,sizeof(PetscInt)*nlsiz);CHKERRQ(ierr);

  /* compute the nonzero pattern based on MOAB connectivity data for local elements */
  ierr = DMMoab_Compute_NNZ_From_Connectivity(dm,&innz,nnz,&ionz,onz,(tmp?PETSC_TRUE:PETSC_FALSE));CHKERRQ(ierr);

  /* create the Matrix and set its type as specified by user */
  ierr = MatCreate(dmmoab->pcomm->comm(), J);CHKERRQ(ierr);
  ierr = MatSetSizes(*J, dmmoab->nloc*dmmoab->numFields, dmmoab->nloc*dmmoab->numFields, PETSC_DETERMINE, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetBlockSize(*J, dmmoab->bs);CHKERRQ(ierr);
  ierr = MatSetType(*J, mtype);CHKERRQ(ierr);
  ierr = MatSetFromOptions(*J);CHKERRQ(ierr);

  if (!dmmoab->ltog_map) SETERRQ(dmmoab->pcomm->comm(), PETSC_ERR_ORDER, "Cannot create a DMMoab Mat without calling DMSetUp first.");
  ierr = MatSetLocalToGlobalMapping(*J,dmmoab->ltog_map,dmmoab->ltog_map);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingBlock(dmmoab->ltog_map,dmmoab->bs,&ltogb);
  ierr = MatSetLocalToGlobalMappingBlock(*J,ltogb,ltogb);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&ltogb);CHKERRQ(ierr);

  /* set preallocation based on different supported Mat types */
  ierr = MatSeqAIJSetPreallocation(*J, innz, nnz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(*J, innz, nnz, ionz, onz);CHKERRQ(ierr);
  ierr = MatSeqBAIJSetPreallocation(*J, dmmoab->bs, innz, nnz);CHKERRQ(ierr);
  ierr = MatMPIBAIJSetPreallocation(*J, dmmoab->bs, innz, nnz, ionz, onz);CHKERRQ(ierr);

  /* clean up temporary memory */
  ierr = PetscFree(nnz);CHKERRQ(ierr);
  ierr = PetscFree(onz);CHKERRQ(ierr);

  /* set up internal matrix data-structures */
  ierr = MatSetUp(*J);CHKERRQ(ierr);

  /* set DM reference */
  ierr = MatSetDM(*J, dm);CHKERRQ(ierr);

  /* set the correct NNZ pattern by setting matrix entries - make the matrix ready to use */
  ierr = DMMoab_MatFillMatrixEntries_Private(dm,*J);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoab_Compute_NNZ_From_Connectivity"
PetscErrorCode DMMoab_Compute_NNZ_From_Connectivity(DM dm,PetscInt* innz,PetscInt* nnz,PetscInt* ionz,PetscInt* onz,PetscBool isbaij)
{
  PetscInt        i,f,nloc,vpere,bs,nsize,ivtx,n_nnz,n_onz;
  DM_Moab         *dmmoab = (DM_Moab*)dm->data;
  const moab::EntityHandle *connect;
  moab::Range     adjs,found,allvlocal,allvghost;
  moab::Range::iterator iter,jter;
  std::vector<moab::EntityHandle> storage;
  moab::EntityHandle vtx;
  moab::ErrorCode merr;

  PetscFunctionBegin;
  bs = dmmoab->bs;
  nloc = dmmoab->nloc;
  nsize = (isbaij ? nloc:nloc*bs);

  /* find the truly user-expected layer of ghosted entities to decipher NNZ pattern */
  merr = dmmoab->mbiface->get_entities_by_type(dmmoab->fileset,moab::MBVERTEX,allvlocal,true);MBERRNM(merr);
  merr = dmmoab->pcomm->filter_pstatus(allvlocal,PSTATUS_NOT_OWNED,PSTATUS_NOT,-1,&adjs);MBERRNM(merr);
  allvghost = moab::subtract(allvlocal, adjs);

  /* loop over the locally owned vertices and figure out the NNZ pattern using connectivity information */
  for(iter = dmmoab->vowned->begin(),ivtx=0; iter != dmmoab->vowned->end(); iter++,ivtx++) {

    vtx = *iter;
    adjs.clear();
    /* Get adjacency information for current vertex - i.e., all elements of dimension (dim) that connects
       to the current vertex. We can then decipher if a vertex is ghosted or not and compute the 
       non-zero pattern accordingly. */
    merr = dmmoab->mbiface->get_adjacencies(&vtx,1,dmmoab->dim,false,adjs,moab::Interface::INTERSECT);

    /* reset counters */
    n_nnz=n_onz=0;
    found.clear();

    /* loop over vertices and update the number of connectivity */
    for(jter = adjs.begin(); jter != adjs.end(); jter++) {
      
      /* Get connectivity information in canonical ordering for the local element */
      merr = dmmoab->mbiface->get_connectivity(*jter,connect,vpere,false,&storage);MBERRNM(merr);

      /* loop over each element connected to the adjacent vertex and update as needed */
      for (i=0; i<vpere; ++i) {
        if (connect[i] == vtx || found.find(connect[i]) != found.end()) continue; /* make sure we don't double count shared vertices */
        if (allvghost.find(connect[i]) != allvghost.end()) n_onz++; /* update out-of-proc onz */
        else n_nnz++; /* else local vertex */
        found.insert(connect[i]);
      }
    }
 
    if (isbaij) {
      nnz[ivtx]=n_nnz;      /* leave out self to avoid repeats -> node shared by multiple elements */
      if (onz) onz[ivtx]=n_onz;  /* add ghost non-owned nodes */
    }
    else { /* AIJ matrices */
      for (f=0;f<dmmoab->numFields;f++) {
        nnz[dmmoab->numFields*ivtx+f]=n_nnz;      /* leave out self to avoid repeats -> node shared by multiple elements */
        if (onz) onz[dmmoab->numFields*ivtx+f]=n_onz;  /* add ghost non-owned nodes */
      }
    }
  }

  if (innz) *innz=0;
  if (ionz) *ionz=0;
  for (i=0;i<nsize;i++) {
    nnz[i]+=1;  /* self count the node */
    /* check if we got overzealous */
    nnz[i]=(nnz[i]>dmmoab->nloc ? dmmoab->nloc:nnz[i]);
    if (!isbaij) {
      nnz[i]*=bs;
      if (onz) onz[i]*=bs;
    }

    /* update innz and ionz based on local maxima */
    if (innz && (nnz[i]>*innz)) *innz=nnz[i];
    if ((ionz && onz) && (onz[i]>*ionz)) *ionz=onz[i];
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoab_MatFillMatrixEntries_Private"
PetscErrorCode DMMoab_MatFillMatrixEntries_Private(DM dm, Mat A)
{
  DM_Moab                   *dmmoab = (DM_Moab*)dm->data;
  PetscInt                  nconn = 0,prev_nconn = 0;
  const moab::EntityHandle  *connect;
  PetscScalar               *locala=NULL;
  PetscInt                  *dof_indices=NULL;
  PetscErrorCode            ierr;

  PetscFunctionBegin;
  /* loop over local elements */
  for(moab::Range::iterator iter = dmmoab->elocal->begin(); iter != dmmoab->elocal->end(); iter++) {
    const moab::EntityHandle ehandle = *iter;

    /* Get connectivity information: */
    ierr = DMMoabGetElementConnectivity(dm, ehandle, &nconn, &connect);CHKERRQ(ierr);

    /* if we have mixed elements or arrays have not been initialized - Allocate now */
    if (prev_nconn != nconn) {
      if (locala) {
        ierr = PetscFree(locala);CHKERRQ(ierr);
        ierr = PetscFree(dof_indices);CHKERRQ(ierr);
      }
      ierr = PetscMalloc(sizeof(PetscScalar)*nconn*nconn*dmmoab->numFields*dmmoab->numFields,&locala);CHKERRQ(ierr);
      ierr = PetscMemzero(locala,sizeof(PetscScalar)*nconn*nconn*dmmoab->numFields*dmmoab->numFields);CHKERRQ(ierr);
      ierr = PetscMalloc(sizeof(PetscInt)*nconn,&dof_indices);CHKERRQ(ierr);
      prev_nconn=nconn;
    }

    /* get the global DOF number to appropriately set the element contribution in the RHS vector */
    ierr = DMMoabGetDofsBlockedLocal(dm, nconn, connect, dof_indices);CHKERRQ(ierr);

    /* set the values directly into appropriate locations. Can alternately use VecSetValues */
    ierr = MatSetValuesBlockedLocal(A, nconn, dof_indices, nconn, dof_indices, locala, INSERT_VALUES);CHKERRQ(ierr);
  }

  /* clean up memory */
  ierr = PetscFree(locala);CHKERRQ(ierr);
  ierr = PetscFree(dof_indices);CHKERRQ(ierr);

  /* finish assembly */
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

