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
  ierr = MatSeqAIJSetPreallocation(*J, innz, nnz);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(*J, innz, nnz, ionz, onz);CHKERRQ(ierr);
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

    /* loop over vertices and update the number of connectivity */
    for (j=0;j<vpere;j++) {

      /*
      moab::Range::const_iterator giter = visited.find(connect[j]);
      if (giter != visited.end()) continue;

      visited.insert(connect[j]);
      */

      merr = dmmoab->mbiface->get_adjacencies(&connect[j],1,1,true,adjs,moab::Interface::INTERSECT);

      merr = dmmoab->mbiface->tag_get_data(id_tag,&connect[j],1,&dof);MBERRNM(merr);
      ierr = PetscSectionGetDof(section, connect[j], &ndofs);

      for (f=0;f<dmmoab->nfields;f++) {
        ierr = PetscSectionGetFieldOffset(section, connect[j], f, &doff);

//        nnz[doff]+=(vpere-nghost_found-1);      /* leave out self to avoid repeats -> node shared by multiple elements */
        nnz[doff]+=adjs.size();      /* leave out self to avoid repeats -> node shared by multiple elements */
        onz[doff]+=nghost_found;  /* add ghost non-owned nodes */

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

    if (innz && nnz[i]>*innz) *innz=nnz[i];
    if (ionz && onz[i]>*ionz) *ionz=onz[i];
  }
  PetscFunctionReturn(0);
}

