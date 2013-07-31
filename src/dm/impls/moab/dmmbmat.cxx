#include <petsc-private/dmmbimpl.h> /*I  "petscdm.h"   I*/
#include <petsc-private/vecimpl.h> /*I  "petscdm.h"   I*/

#include <petscdmmoab.h>

static PetscErrorCode DMMoab_Compute_NNZ_From_Connectivity(DM,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscBool);

#undef __FUNCT__
#define __FUNCT__ "DMCreateMatrix_Moab"
PetscErrorCode DMCreateMatrix_Moab(DM dm, MatType mtype,Mat *J)
{
  PetscErrorCode  ierr;
  ISLocalToGlobalMapping ltog;
  MatType         mltype;
  PetscInt        i,nloc,count,dof,innz,ionz,nsize;
  DM_Moab         *dmmoab = (DM_Moab*)dm->data;
  moab::Range     *range=dmmoab->vowned;
  PetscInt        *gindices,*nnz,*onz;
  char            *tmp=0;
  moab::ErrorCode merr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(J,3);
  nloc = dmmoab->vowned->size() * dmmoab->bs;

  /* create the Matrix and set its type as specified by user */
  ierr = MatCreate(dmmoab->pcomm->comm(), J);CHKERRQ(ierr);
  ierr = MatSetSizes(*J, nloc, nloc, PETSC_DETERMINE, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetType(*J, mtype);CHKERRQ(ierr);
  ierr = MatSetFromOptions(*J);CHKERRQ(ierr);
  
  /* next, need to allocate the non-zero arrays to enable pre-allocation */
  ierr = MatGetType(*J, &mltype);CHKERRQ(ierr); /* in case user overrode the default type from command-line, re-check the type */
  ierr = PetscStrstr(mltype, "baij", &tmp);CHKERRQ(ierr);
  nsize = (tmp ? dmmoab->nloc:dmmoab->nloc*dmmoab->bs);

  /* allocate the nnz, onz arrays based on block size and local nodes */
  ierr = PetscMalloc(nsize*sizeof(PetscInt),&nnz);CHKERRQ(ierr);
  ierr = PetscMemzero(nnz,sizeof(PetscInt)*nsize);CHKERRQ(ierr);
  ierr = PetscMalloc(nsize*sizeof(PetscInt),&onz);CHKERRQ(ierr);
  ierr = PetscMemzero(onz,sizeof(PetscInt)*nsize);CHKERRQ(ierr);

  /* compute the nonzero pattern based on MOAB connectivity data for local elements */
  ierr = DMMoab_Compute_NNZ_From_Connectivity(dm,&innz,nnz,&ionz,onz,(tmp?PETSC_TRUE:PETSC_FALSE));CHKERRQ(ierr);

  if (dmmoab->bs > 1 && tmp) {
    /* Block matrix created, now set local to global mapping */
    ierr = PetscMalloc(sizeof(PetscInt)*range->size()*dmmoab->bs, &gindices);CHKERRQ(ierr);
    moab::Range::iterator  iter;
    for(iter = range->begin(),count=0; iter != range->end(); iter++,count+=dmmoab->bs) {
      merr = dmmoab->mbiface->tag_get_data(dmmoab->ltog_tag,&(*iter),1,&dof);MBERRNM(merr);
      for(i=0; i<dmmoab->bs; ++i)
        gindices[count+i] = (dof)*dmmoab->bs+i;
    }

    ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_SELF,range->size(),gindices,PETSC_COPY_VALUES,&ltog);CHKERRQ(ierr);
    ierr = MatSetLocalToGlobalMappingBlock(*J,ltog,ltog);CHKERRQ(ierr);

    /* Clean up */
    ierr = ISLocalToGlobalMappingDestroy(&ltog);CHKERRQ(ierr);
    ierr = PetscFree(gindices);CHKERRQ(ierr);
    
  } else {
    ierr = MatSetBlockSize(*J, dmmoab->bs);CHKERRQ(ierr);
  }

  /* set preallocation based on different supported Mat types */
//  ierr = MatSeqAIJSetPreallocation(*J, innz, nnz);CHKERRQ(ierr);
//  ierr = MatMPIAIJSetPreallocation(*J, innz, nnz, ionz, onz);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(*J, innz, 0);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(*J, innz, 0, ionz, 0);CHKERRQ(ierr);
  ierr = MatSeqBAIJSetPreallocation(*J, dmmoab->bs, innz, nnz);CHKERRQ(ierr);
  ierr = MatMPIBAIJSetPreallocation(*J, dmmoab->bs, innz, nnz, ionz, onz);CHKERRQ(ierr);

  ierr = MatSetDM(*J, dm);CHKERRQ(ierr);
  ierr = PetscFree(nnz);CHKERRQ(ierr);
  ierr = PetscFree(onz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMMoab_Compute_NNZ_From_Connectivity"
PetscErrorCode DMMoab_Compute_NNZ_From_Connectivity(DM dm,PetscInt* innz,PetscInt* nnz,PetscInt* ionz,PetscInt* onz,PetscBool isbaij)
{
  PetscErrorCode  ierr;
  PetscInt        i,j,f,nloc,vpere,bs,nsize,nghost_found;
  DM_Moab         *dmmoab = (DM_Moab*)dm->data;
  const moab::EntityHandle *connect;
  moab::Range     *vowned=dmmoab->vowned,*elocal=dmmoab->elocal,*eghost=dmmoab->eghost,adjs,visited;
  moab::Range::iterator iter;
  PetscInt        dof,doff,ndofs;
  moab::Tag       id_tag;
  moab::ErrorCode merr;
  PetscSection section;

  PetscFunctionBegin;
  bs = dmmoab->bs;
  nloc = dmmoab->nloc;
  nsize = (isbaij ? nloc:nloc*bs);

  ierr = DMMoabGetLocalToGlobalTag(dm,&id_tag);CHKERRQ(ierr);

  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  
  for(iter = elocal->begin(),i=0; iter != elocal->end(); iter++,i++) {
    /* Get connectivity information in canonical ordering for the local element */
    merr = dmmoab->mbiface->get_connectivity((*iter),connect,vpere);MBERRNM(merr);
    
    nghost_found=0;
    /* loop over vertices and update the number of connectivity */
    for (j=0;j<vpere;j++) {
      moab::Range::const_iterator giter = dmmoab->vghost->find(connect[j]);
      if (giter != dmmoab->vghost->end()) nghost_found++;
    }
//      PetscPrintf(PETSC_COMM_WORLD, "Element %D, VPERE %D, nGhost %D\n", i,vpere,nghost_found);

    /* loop over vertices and update the number of connectivity */
    for (j=0;j<vpere;j++) {

      /*
      moab::Range::const_iterator giter = visited.find(connect[j]);
      if (giter != visited.end()) continue;

      visited.insert(connect[j]);
      */

      merr = dmmoab->mbiface->get_adjacencies(&connect[j],1,1,true,adjs,moab::Interface::INTERSECT);
//      PetscPrintf(PETSC_COMM_WORLD, "Point %D, nADJS %D\n", connect[j], adjs.size());

      merr = dmmoab->mbiface->tag_get_data(id_tag,&connect[j],1,&dof);MBERRNM(merr);
      ierr = PetscSectionGetDof(section, connect[j], &ndofs);

      for (f=0;f<dmmoab->nfields;f++) {
        ierr = PetscSectionGetFieldOffset(section, connect[j], f, &doff);

//        nnz[doff]+=(vpere-nghost_found-1);      /* leave out self to avoid repeats -> node shared by multiple elements */
        nnz[doff]+=adjs.size();      /* leave out self to avoid repeats -> node shared by multiple elements */
        onz[doff]+=nghost_found;  /* add ghost non-owned nodes */

//        PetscPrintf(PETSC_COMM_WORLD, "Point %D, Field %D, DOFPP %D \t DOF %D, OFFSET %D : NNZ = %D, ONZ = %D\n", connect[j], f, ndofs, dof, doff, nnz[doff], onz[doff]);
        adjs.clear();
      }
    }
  }

  visited.clear();

  if (innz) *innz=0;
  if (ionz) *ionz=0;
  for (i=0;i<nsize;i++) {
    nnz[i]+=1;  /* self count the node */
    if (!isbaij) {
      nnz[i]*=bs;
      onz[i]*=bs;
    }
//    PetscPrintf(PETSC_COMM_WORLD, "Index %D: NNZ = %D, ONZ = %D\n", i, nnz[i], onz[i]);

    if (innz && nnz[i]>*innz) *innz=nnz[i];
    if (ionz && onz[i]>*ionz) *ionz=onz[i];
  }
//    PetscPrintf(PETSC_COMM_WORLD, "MAX: NNZ = %D, ONZ = %D\n", *innz, *ionz);
  PetscFunctionReturn(0);
}


#if 0

#undef __FUNCT__
#define __FUNCT__ "DMMoab_Compute_NNZ_From_Connectivity_OLD"
PetscErrorCode DMMoab_Compute_NNZ_From_Connectivity_OLD(DM dm,PetscInt* innz,PetscInt* nnz,PetscInt* ionz,PetscInt* onz,PetscBool isbaij)
{
  PetscErrorCode  ierr;
  PetscInt        i,j,k,f,nloc,count,vpere,bs,nsize,nghost_found;
  DM_Moab         *dmmoab = (DM_Moab*)dm->data;
  const moab::EntityHandle *connect;
  moab::Range     *vowned=dmmoab->vowned,*elocal=dmmoab->elocal,*eghost=dmmoab->eghost,adjs;
  moab::Range::iterator iter;
  PetscInt        *vertex_ids,firstvtx,dof,doff,ndofs,offset;
  moab::Tag       id_tag;
  std::vector<moab::EntityHandle> storage;
  moab::ErrorCode merr;
  PetscSection section;

  PetscFunctionBegin;
  bs = dmmoab->bs;
  nloc = dmmoab->nloc;
  nsize = (isbaij ? nloc:nloc*bs);

  ierr = DMMoabGetLocalToGlobalTag(dm,&id_tag);CHKERRQ(ierr);

  merr = dmmoab->mbiface->tag_get_data(id_tag,&(*vowned->begin()),1,&firstvtx);MBERRNM(merr);
  //firstvtx = vertex_ids[0];

  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);

  for(iter = elocal->begin(),i=0; iter != elocal->end(); iter++,i++) {
    /* Get connectivity information in canonical ordering for the local element */
    merr = dmmoab->mbiface->get_connectivity((*iter),connect,vpere,false,&storage);MBERRNM(merr);

    nghost_found=0;
    /* loop over vertices and update the number of connectivity */
    for (j=0;j<vpere;j++) {
      moab::Range::const_iterator giter = dmmoab->vghost->find(connect[j]);
      if (giter != dmmoab->vghost->end()) nghost_found++;
    }

    /* loop over vertices and update the number of connectivity */
    for (j=0;j<vpere;j++) {

      merr = dmmoab->mbiface->get_adjacencies(&connect[j],1,1,true,adjs,moab::Interface::INTERSECT);
      PetscPrintf(PETSC_COMM_WORLD, "Point %D, nADJS %D\n", connect[j], adjs.size());

      moab::Range::const_iterator giter = dmmoab->vghost->find(connect[j]);
      if (giter != dmmoab->vghost->end()) continue;

      merr = dmmoab->mbiface->tag_get_data(id_tag,&connect[j],1,&dof);MBERRNM(merr);
      ierr = PetscSectionGetDof(section, connect[j], &ndofs);

      for (f=0;f<dmmoab->nfields;f++) {
        ierr = PetscSectionGetFieldOffset(section, connect[j], f, &doff);

//        nnz[doff]+=ndofs*(vpere-nghost_found-1);      /* leave out self to avoid repeats -> node shared by multiple elements */
        nnz[doff]+=adjs.size()-nghost_found-1;      /* leave out self to avoid repeats -> node shared by multiple elements */
        onz[doff]+=nghost_found;  /* add ghost non-owned nodes */

        PetscPrintf(PETSC_COMM_WORLD, "Point %D, Field %D, DOFPP %D \t DOF %D, OFFSET %D : NNZ = %D, ONZ = %D\n", connect[j], f, ndofs, dof, doff, nnz[doff], onz[doff]);
        adjs.clear();
      }

//      /* if block format, then all the block data are local only */
//      if (!isbaij) {
//        offset=(dof-firstvtx)*bs;
//        for (k=0;k<bs;k++) {
//          nnz[offset+k]+=vpere-nghost_found-1;      /* leave out self to avoid repeats -> node shared by multiple elements */
//          onz[offset+k]+=nghost_found;  /* found a ghost non-owned node */
//        }
//      }
//      else {
//        nnz[(dof-firstvtx)]+=vpere-nghost_found-1;  /* leave out self to avoid repeats -> node shared by multiple elements */
//        onz[(dof-firstvtx)]+=nghost_found;  /* found a ghost non-owned node */
//      }

    }
  }

  /* loop through ghosted elements that contain non-ghosted (locally owned) vertices */
  for(iter = eghost->begin(),i=0; iter != eghost->end(); iter++,i++) {
    /* Get connectivity information in canonical ordering for the local element */
    merr = dmmoab->mbiface->get_connectivity((*iter),connect,vpere,false,&storage);MBERRNM(merr);

    nghost_found=0;
    /* loop over vertices and update the number of connectivity */
    for (j=0;j<vpere;j++) {
      moab::Range::const_iterator giter = dmmoab->vghost->find(connect[j]);
      if (giter != dmmoab->vghost->end()) nghost_found++;
    }

    if (nghost_found == vpere) continue;  /* all vertices are ghosted.. */

    /* loop over vertices and update the number of connectivity */
    for (j=0;j<vpere;j++) {
      moab::Range::const_iterator giter = dmmoab->vghost->find(connect[j]);
      if (giter != dmmoab->vghost->end()) continue;

      merr = dmmoab->mbiface->tag_get_data(id_tag,&connect[j],1,&dof);MBERRNM(merr);
      ierr = PetscSectionGetDof(section, connect[j], &ndofs);
      
      for (f=0;f<dmmoab->nfields;f++) {
        ierr = PetscSectionGetFieldOffset(section, connect[j], f, &doff);

        onz[doff]+=nghost_found;  /* add ghost non-owned nodes */

        PetscPrintf(PETSC_COMM_WORLD, "Point %D, Field %D, DOF %D, OFFSET %D : ONZ = %D\n", connect[j], f, dof, doff, onz[doff]);

      }

//      if (dof-firstvtx < 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_LIB,"Invalid local vertex ordering. Found ghost node with ID less than first vertex: [%i, %i]",dof,firstvtx);
//
//      /* if block format, then all the block data are local only */
//      if (!isbaij) {
//        offset=(dof-firstvtx)*bs;
//        for (k=0;k<bs;k++)
//          onz[offset+k]+=nghost_found;
//      }
//      else {
//        onz[(dof-firstvtx)]+=nghost_found;
//      }
    }
  }

  if (innz) *innz=0;
  if (ionz) *ionz=0;
  for (i=0;i<nsize;i++) {
    nnz[i]+=1;  /* self count the node */
    if (!isbaij) {
      nnz[i]*=bs;
      onz[i]*=bs;
    }
    PetscPrintf(PETSC_COMM_WORLD, "Index %D: NNZ = %D, ONZ = %D\n", i, nnz[i], onz[i]);
    
    if (innz && nnz[i]>*innz) *innz=nnz[i];
    if (ionz && onz[i]>*ionz) *ionz=onz[i];
  }
    PetscPrintf(PETSC_COMM_WORLD, "MAX: NNZ = %D, ONZ = %D\n", *innz, *ionz);
  PetscFunctionReturn(0);
}

#endif

