
/*
  This file defines an "generalized" additive Schwarz preconditioner for any Mat implementation.
  In this version each processor may have any number of subdomains and a subdomain may live on multiple
  processors.

       N    - total number of subdomains on all processors
       n    - actual number of subdomains on this processor
       nmax - maximum number of subdomains per processor
*/
#include <petsc-private/pcimpl.h>     /*I "petscpc.h" I*/

typedef struct {
  PetscInt   N,n,nmax;
  PetscInt   overlap;             /* overlap requested by user */
  KSP        *ksp;                /* linear solvers for each block */
  Vec        gx,gy;               /* Merged work vectors */
  Vec        *x,*y;               /* Split work vectors; storage aliases pieces of storage of the above merged vectors. */
  IS         gis, gis_local;      /* merged ISs */
  VecScatter grestriction;        /* merged restriction */
  VecScatter gprolongation;       /* merged prolongation */
  IS         *is;                 /* index set that defines each overlapping subdomain */
  IS         *is_local;           /* index set that defines each local subdomain (same as subdomain with the overlap removed); may be NULL */
  Mat        *pmat;               /* subdomain block matrices */
  PCGASMType  type;               /* use reduced interpolation, restriction or both */
  PetscBool  type_set;            /* if user set this value (so won't change it for symmetric problems) */
  PetscBool  same_local_solves;   /* flag indicating whether all local solvers are same */
  PetscBool  sort_indices;        /* flag to sort subdomain indices */
} PC_GASM;

#undef __FUNCT__
#define __FUNCT__ "PCGASMPrintSubdomains"
static PetscErrorCode  PCGASMPrintSubdomains(PC pc)
{
  PC_GASM         *osm  = (PC_GASM*)pc->data;
  const char     *prefix;
  char           fname[PETSC_MAX_PATH_LEN+1];
  PetscViewer    viewer, sviewer;
  PetscInt       i,j,nidx;
  const PetscInt *idx;
  char           *cidx;
  PetscBool      found;
  PetscMPIInt    size, rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(((PetscObject)pc)->comm, &size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(((PetscObject)pc)->comm, &rank);CHKERRQ(ierr);
  ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(prefix,"-pc_gasm_print_subdomains",fname,PETSC_MAX_PATH_LEN,&found);CHKERRQ(ierr);
  if (!found) { ierr = PetscStrcpy(fname,"stdout");CHKERRQ(ierr); };
  for (i=0;i<osm->n;++i) {
    ierr = PetscViewerASCIIOpen(((PetscObject)((osm->is)[i]))->comm,fname,&viewer);CHKERRQ(ierr);
    ierr = ISGetLocalSize(osm->is[i], &nidx);CHKERRQ(ierr);
    /* 
     No more than 15 characters per index plus a space.
     PetscViewerStringSPrintf requires a string of size at least 2, so use (nidx+1) instead of nidx, 
     in case nidx == 0. That will take care of the space for the trailing '\0' as well. 
     For nidx == 0, the whole string 16 '\0'.
     */
    ierr = PetscMalloc(sizeof(char)*(16*(nidx+1)+1), &cidx);CHKERRQ(ierr);  
    ierr = ISGetIndices(osm->is[i], &idx);CHKERRQ(ierr);
    ierr = PetscViewerStringOpen(((PetscObject)(osm->is[i]))->comm, cidx, 16*(nidx+1), &sviewer);CHKERRQ(ierr);
    for(j = 0; j < nidx; ++j) {
      ierr = PetscViewerStringSPrintf(sviewer, "%D ", idx[j]);CHKERRQ(ierr);
    }
    ierr = PetscViewerDestroy(&sviewer);CHKERRQ(ierr);
    ierr = ISRestoreIndices(osm->is[i],&idx);CHKERRQ(ierr);

    ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "Subdomain with overlap\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer, "%s", cidx);CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_FALSE);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
    ierr = PetscFree(cidx);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    if (osm->is_local) {
      ierr = PetscViewerASCIIOpen(((PetscObject)((osm->is)[i]))->comm,fname,&viewer);CHKERRQ(ierr);
      ierr = ISGetLocalSize(osm->is_local[i], &nidx);CHKERRQ(ierr);
      /* 
       No more than 15 characters per index plus a space.
       PetscViewerStringSPrintf requires a string of size at least 2, so use (nidx+1) instead of nidx, 
       in case nidx == 0. That will take care of the space for the trailing '\0' as well. 
       For nidx == 0, the whole string 16 '\0'.
       */
      ierr = PetscMalloc(sizeof(char)*(16*(nidx+1)+1), &cidx);CHKERRQ(ierr);  
      ierr = ISGetIndices(osm->is_local[i], &idx);CHKERRQ(ierr);
      ierr = PetscViewerStringOpen(((PetscObject)(osm->is_local[i]))->comm, cidx, 16*(nidx+1), &sviewer);CHKERRQ(ierr);
      for(j = 0; j < nidx; ++j) {
        ierr = PetscViewerStringSPrintf(sviewer, "%D ", idx[j]);CHKERRQ(ierr);
      }
      ierr = PetscViewerDestroy(&sviewer);CHKERRQ(ierr);
      ierr = ISRestoreIndices(osm->is_local[i],&idx);CHKERRQ(ierr);

      ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_TRUE);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer, "Subdomain without overlap\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer, "%s", cidx);CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_FALSE);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
      ierr = PetscFree(cidx);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PCView_GASM"
static PetscErrorCode PCView_GASM(PC pc,PetscViewer viewer)
{
  PC_GASM         *osm = (PC_GASM*)pc->data;
  const char     *prefix;
  PetscErrorCode ierr;
  PetscMPIInt    rank, size;
  PetscInt       i,bsz;
  PetscBool      iascii,isstring, print_subdomains=PETSC_FALSE;
  PetscViewer    sviewer;


  PetscFunctionBegin;
  ierr = MPI_Comm_size(((PetscObject)pc)->comm, &size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(((PetscObject)pc)->comm, &rank);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERSTRING,&isstring);CHKERRQ(ierr);
  ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(prefix,"-pc_gasm_print_subdomains",&print_subdomains,PETSC_NULL);CHKERRQ(ierr);
  if (iascii) {
    char overlaps[256] = "user-defined overlap",subdomains[256] = "total subdomains set";
    if (osm->overlap >= 0) {ierr = PetscSNPrintf(overlaps,sizeof overlaps,"amount of overlap = %D",osm->overlap);CHKERRQ(ierr);}
    if (osm->nmax > 0)     {ierr = PetscSNPrintf(subdomains,sizeof subdomains,"max number of local subdomains = %D",osm->nmax);CHKERRQ(ierr);}
    ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"  [%d:%d] number of locally-supported subdomains = %D\n",(int)rank,(int)size,osm->n);CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_FALSE);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  Generalized additive Schwarz: %s, %s\n",subdomains,overlaps);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  Generalized additive Schwarz: restriction/interpolation type - %s\n",PCGASMTypes[osm->type]);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(((PetscObject)pc)->comm,&rank);CHKERRQ(ierr);
    if (osm->same_local_solves) {
      if (osm->ksp) {
        ierr = PetscViewerASCIIPrintf(viewer,"  Local solve is same for all subdomains, in the following KSP and PC objects:\n");CHKERRQ(ierr);
        ierr = PetscViewerGetSingleton(viewer,&sviewer);CHKERRQ(ierr);
        if (!rank) {
          ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
          ierr = KSPView(osm->ksp[0],sviewer);CHKERRQ(ierr);
          ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
        }
        ierr = PetscViewerRestoreSingleton(viewer,&sviewer);CHKERRQ(ierr);
      }
    } else {
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  Local solve info for each subdomain is in the following KSP and PC objects:\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"- - - - - - - - - - - - - - - - - -\n");CHKERRQ(ierr);
      for (i=0; i<osm->nmax; i++) {
        ierr = PetscViewerGetSingleton(viewer,&sviewer);CHKERRQ(ierr);
        if (i < osm->n) {
          ierr = ISGetLocalSize(osm->is[i],&bsz);CHKERRQ(ierr);
          ierr = PetscViewerASCIISynchronizedAllow(sviewer,PETSC_TRUE);CHKERRQ(ierr);
          ierr = PetscViewerASCIISynchronizedPrintf(sviewer,"[%d:%d] local subdomain number %D, size = %D\n",(int)rank,(int)size,i,bsz);CHKERRQ(ierr);
          ierr = KSPView(osm->ksp[i],sviewer);CHKERRQ(ierr);
          ierr = PetscViewerASCIISynchronizedPrintf(sviewer,"- - - - - - - - - - - - - - - - - -\n");CHKERRQ(ierr);
          ierr = PetscViewerFlush(sviewer);CHKERRQ(ierr);
          ierr = PetscViewerASCIISynchronizedAllow(sviewer,PETSC_FALSE);CHKERRQ(ierr);
        }
        ierr = PetscViewerRestoreSingleton(viewer,&sviewer);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    }
  } else if (isstring) {
    ierr = PetscViewerStringSPrintf(viewer," subdomains=%D, overlap=%D, type=%s",osm->n,osm->overlap,PCGASMTypes[osm->type]);CHKERRQ(ierr);
    ierr = PetscViewerGetSingleton(viewer,&sviewer);CHKERRQ(ierr);
    if (osm->ksp) {ierr = KSPView(osm->ksp[0],sviewer);CHKERRQ(ierr);}
    ierr = PetscViewerRestoreSingleton(viewer,&sviewer);CHKERRQ(ierr);
  } else {
    SETERRQ1(((PetscObject)pc)->comm,PETSC_ERR_SUP,"Viewer type %s not supported for PCGASM",((PetscObject)viewer)->type_name);
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  if(print_subdomains) {
    ierr = PCGASMPrintSubdomains(pc);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}





#undef __FUNCT__
#define __FUNCT__ "PCSetUp_GASM"
static PetscErrorCode PCSetUp_GASM(PC pc)
{
  PC_GASM         *osm  = (PC_GASM*)pc->data;
  PetscErrorCode ierr;
  PetscBool      symset,flg;
  PetscInt       i;
  PetscMPIInt    rank, size;
  MatReuse       scall = MAT_REUSE_MATRIX;
  KSP            ksp;
  PC             subpc;
  const char     *prefix,*pprefix;
  PetscInt       dn;       /* Number of indices in a single subdomain assigned to this processor. */
  const PetscInt *didx;    /* Indices from a single subdomain assigned to this processor. */
  PetscInt       ddn;      /* Number of indices in all subdomains assigned to this processor. */
  PetscInt       *ddidx;   /* Indices of all subdomains assigned to this processor. */
  IS             gid;      /* Identity IS of the size of all subdomains assigned to this processor. */
  Vec            x,y;
  PetscScalar    *gxarray, *gyarray;
  PetscInt       gfirst, glast;
  DM             *domain_dm = PETSC_NULL;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(((PetscObject)pc)->comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(((PetscObject)pc)->comm,&rank);CHKERRQ(ierr);
  if (!pc->setupcalled) {

    if (!osm->type_set) {
      ierr = MatIsSymmetricKnown(pc->pmat,&symset,&flg);CHKERRQ(ierr);
      if (symset && flg) { osm->type = PC_GASM_BASIC; }
    }

    /* Note: if osm->n has been set, it is at least 1. */
    if (osm->N == PETSC_DECIDE && osm->n < 1) { 
      /* no subdomains given */
      /* try pc->dm first */
      if(pc->dm) {
        char      ddm_name[1024];
        DM        ddm;
        PetscBool flg;
        PetscInt     num_domains, d;
        char         **domain_names;
        IS           *domain_is;
        /* Allow the user to request a decomposition DM by name */
        ierr = PetscStrncpy(ddm_name, "", 1024); CHKERRQ(ierr);
        ierr = PetscOptionsString("-pc_asm_decomposition_dm", "Name of the DM defining the composition", "PCSetDM", ddm_name, ddm_name,1024,&flg); CHKERRQ(ierr);
        if(flg) {
          ierr = DMCreateDecompositionDM(pc->dm, ddm_name, &ddm); CHKERRQ(ierr);
          if(!ddm) {
            SETERRQ1(((PetscObject)pc)->comm, PETSC_ERR_ARG_WRONGSTATE, "Uknown DM decomposition name %s", ddm_name);
          }
          ierr = PetscInfo(pc,"Using decomposition DM defined using options database\n");CHKERRQ(ierr);
          ierr = PCSetDM(pc,ddm); CHKERRQ(ierr);
        }
        ierr = DMCreateDecomposition(pc->dm, &num_domains, &domain_names, &domain_is, &domain_dm);    CHKERRQ(ierr);
        if(num_domains) {
          ierr = PCGASMSetLocalSubdomains(pc, num_domains, domain_is, PETSC_NULL);CHKERRQ(ierr);
        }
        for(d = 0; d < num_domains; ++d) {
          ierr = PetscFree(domain_names[d]); CHKERRQ(ierr);
          ierr = ISDestroy(&domain_is[d]);   CHKERRQ(ierr);
        }
        ierr = PetscFree(domain_names);CHKERRQ(ierr);
        ierr = PetscFree(domain_is);CHKERRQ(ierr);
      }
      if (osm->N == PETSC_DECIDE && osm->n < 1) { /* still no subdomains; use one per processor */
        osm->nmax = osm->n = 1;
        ierr = MPI_Comm_size(((PetscObject)pc)->comm,&size);CHKERRQ(ierr);
        osm->N = size;
      }
    } else if (osm->N == PETSC_DECIDE) {
      PetscInt inwork[2], outwork[2];
      /* determine global number of subdomains and the max number of local subdomains */
      inwork[0] = inwork[1] = osm->n;
      ierr = MPI_Allreduce(inwork,outwork,1,MPIU_2INT,PetscMaxSum_Op,((PetscObject)pc)->comm);CHKERRQ(ierr);
      osm->nmax = outwork[0];
      osm->N    = outwork[1];
    }
    if (!osm->is){ /* create the index sets */
      ierr = PCGASMCreateSubdomains(pc->pmat,osm->n,&osm->is);CHKERRQ(ierr);
    }
    if (!osm->is_local) {
      /*
	 This indicates that osm->is should define a nonoverlapping decomposition
	 (there is no way to really guarantee that if subdomains are set by the user through PCGASMSetLocalSubdomains,
	  but the assumption is that either the user does the right thing, or subdomains in ossm->is have been created
	  via PCGASMCreateSubdomains, which guarantees a nonoverlapping decomposition).
	 Therefore, osm->is will be used to define osm->is_local.
	 If a nonzero overlap has been requested by the user, then osm->is will be expanded and will overlap,
	 so osm->is_local should obtain a copy of osm->is while they are still (presumably) nonoverlapping.
	 Otherwise (no overlap has been requested), osm->is_local are simply aliases for osm->is.
      */
      ierr = PetscMalloc(osm->n*sizeof(IS),&osm->is_local);CHKERRQ(ierr);
      for (i=0; i<osm->n; i++) {
        if (osm->overlap > 0) { /* With positive overlap, osm->is[i] will be modified */
          ierr = ISDuplicate(osm->is[i],&osm->is_local[i]);CHKERRQ(ierr);
          ierr = ISCopy(osm->is[i],osm->is_local[i]);CHKERRQ(ierr);
        } else {
          ierr = PetscObjectReference((PetscObject)osm->is[i]);CHKERRQ(ierr);
          osm->is_local[i] = osm->is[i];
        }
      }
    }
    /* Beyond this point osm->is_local is not null. */
    if (osm->overlap > 0) {
      /* Extend the "overlapping" regions by a number of steps */
      ierr = MatIncreaseOverlap(pc->pmat,osm->n,osm->is,osm->overlap);CHKERRQ(ierr);
    }
    ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);
    flg  = PETSC_FALSE;
    ierr = PetscOptionsGetBool(prefix,"-pc_gasm_print_subdomains",&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) { ierr = PCGASMPrintSubdomains(pc);CHKERRQ(ierr); }

    if (osm->sort_indices) {
      for (i=0; i<osm->n; i++) {
        ierr = ISSort(osm->is[i]);CHKERRQ(ierr);
	ierr = ISSort(osm->is_local[i]);CHKERRQ(ierr);
      }
    }
    /* Merge the ISs, create merged vectors and scatter contexts. */
    /* Restriction ISs. */
    ddn = 0;
    for (i=0; i<osm->n; i++) {
      ierr = ISGetLocalSize(osm->is[i],&dn);CHKERRQ(ierr);
      ddn += dn;
    }
    ierr = PetscMalloc(ddn*sizeof(PetscInt), &ddidx);CHKERRQ(ierr);
    ddn = 0;
    for (i=0; i<osm->n; i++) {
      ierr = ISGetLocalSize(osm->is[i],&dn);CHKERRQ(ierr);
      ierr = ISGetIndices(osm->is[i],&didx);CHKERRQ(ierr);
      ierr = PetscMemcpy(ddidx+ddn, didx, sizeof(PetscInt)*dn);CHKERRQ(ierr);
      ierr = ISRestoreIndices(osm->is[i], &didx);CHKERRQ(ierr);
      ddn += dn;
    }
    ierr = ISCreateGeneral(((PetscObject)(pc))->comm, ddn, ddidx, PETSC_OWN_POINTER, &osm->gis);CHKERRQ(ierr);
    ierr = MatGetVecs(pc->pmat,&x,&y);CHKERRQ(ierr);
    ierr = VecCreateMPI(((PetscObject)pc)->comm, ddn, PETSC_DECIDE, &osm->gx);CHKERRQ(ierr);
    ierr = VecDuplicate(osm->gx,&osm->gy);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(osm->gx, &gfirst, &glast);CHKERRQ(ierr);
    ierr = ISCreateStride(((PetscObject)pc)->comm,ddn,gfirst,1, &gid);CHKERRQ(ierr);
    ierr = VecScatterCreate(x,osm->gis,osm->gx,gid, &(osm->grestriction));CHKERRQ(ierr);
    ierr = ISDestroy(&gid);CHKERRQ(ierr);
    /* Prolongation ISs */
    { PetscInt       dn_local;       /* Number of indices in the local part of a single domain assigned to this processor. */
      const PetscInt *didx_local;    /* Global indices from the local part of a single domain assigned to this processor. */
      PetscInt       ddn_local;      /* Number of indices in the local part of the disjoint union all domains assigned to this processor. */
      PetscInt       *ddidx_local;   /* Global indices of the local part of the disjoint union of all domains assigned to this processor. */
      /**/
      ISLocalToGlobalMapping ltog;          /* Map from global to local indices on the disjoint union of subdomains: "local" ind's run from 0 to ddn-1. */
      PetscInt              *ddidx_llocal;  /* Mapped local indices of the disjoint union of local parts of subdomains. */
      PetscInt               ddn_llocal;    /* Number of indices in ddidx_llocal; must equal ddn_local, or else gis_local is not a sub-IS of gis. */
      IS                     gis_llocal;    /* IS with ddidx_llocal indices. */
      PetscInt               j;
      ddn_local = 0;
      for (i=0; i<osm->n; i++) {
	ierr = ISGetLocalSize(osm->is_local[i],&dn_local);CHKERRQ(ierr);
	ddn_local += dn_local;
      }
      ierr = PetscMalloc(ddn_local*sizeof(PetscInt), &ddidx_local);CHKERRQ(ierr);
      ddn_local = 0;
      for (i=0; i<osm->n; i++) {
	ierr = ISGetLocalSize(osm->is_local[i],&dn_local);CHKERRQ(ierr);
	ierr = ISGetIndices(osm->is_local[i],&didx_local);CHKERRQ(ierr);
	ierr = PetscMemcpy(ddidx_local+ddn_local, didx_local, sizeof(PetscInt)*dn_local);CHKERRQ(ierr);
	ierr = ISRestoreIndices(osm->is_local[i], &didx_local);CHKERRQ(ierr);
	ddn_local += dn_local;
      }
      ierr = PetscMalloc(sizeof(PetscInt)*ddn_local, &ddidx_llocal);CHKERRQ(ierr);
      ierr = ISCreateGeneral(((PetscObject)pc)->comm, ddn_local, ddidx_local, PETSC_OWN_POINTER, &(osm->gis_local));CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingCreateIS(osm->gis,&ltog);CHKERRQ(ierr);
      ierr = ISGlobalToLocalMappingApply(ltog,IS_GTOLM_DROP,ddn_local,ddidx_local,&ddn_llocal,ddidx_llocal);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingDestroy(&ltog);CHKERRQ(ierr);
      if (ddn_llocal != ddn_local) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"gis_local contains %D indices outside of gis", ddn_llocal - ddn_local);
      /* Now convert these localized indices into the global indices into the merged output vector. */
      ierr = VecGetOwnershipRange(osm->gy, &gfirst, &glast);CHKERRQ(ierr);
      for(j=0; j < ddn_llocal; ++j) {
	ddidx_llocal[j] += gfirst;
      }
      ierr = ISCreateGeneral(PETSC_COMM_SELF,ddn_llocal,ddidx_llocal,PETSC_OWN_POINTER,&gis_llocal);CHKERRQ(ierr);
      ierr = VecScatterCreate(y,osm->gis_local,osm->gy,gis_llocal,&osm->gprolongation);CHKERRQ(ierr);
      ierr = ISDestroy(&gis_llocal);CHKERRQ(ierr);
    }
    /* Create the subdomain work vectors. */
    ierr = PetscMalloc(osm->n*sizeof(Vec),&osm->x);CHKERRQ(ierr);
    ierr = PetscMalloc(osm->n*sizeof(Vec),&osm->y);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(osm->gx, &gfirst, &glast);CHKERRQ(ierr);
    ierr = VecGetArray(osm->gx, &gxarray);CHKERRQ(ierr);
    ierr = VecGetArray(osm->gy, &gyarray);CHKERRQ(ierr);
    for (i=0, ddn=0; i<osm->n; ++i, ddn += dn) {
      PetscInt dN;
      ierr = ISGetLocalSize(osm->is[i],&dn);CHKERRQ(ierr);
      ierr = ISGetSize(osm->is[i],&dN);CHKERRQ(ierr);
      ierr = VecCreateMPIWithArray(((PetscObject)(osm->is[i]))->comm,1,dn,dN,gxarray+ddn,&osm->x[i]);CHKERRQ(ierr); 
      ierr = VecCreateMPIWithArray(((PetscObject)(osm->is[i]))->comm,1,dn,dN,gyarray+ddn,&osm->y[i]);CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(osm->gx, &gxarray);CHKERRQ(ierr);
    ierr = VecRestoreArray(osm->gy, &gyarray);CHKERRQ(ierr);
    ierr = VecDestroy(&x);CHKERRQ(ierr);
    ierr = VecDestroy(&y);CHKERRQ(ierr);
    /* Create the local solvers */
    ierr = PetscMalloc(osm->n*sizeof(KSP *),&osm->ksp);CHKERRQ(ierr);
    for (i=0; i<osm->n; i++) { /* KSPs are local */
      ierr = KSPCreate(((PetscObject)(osm->is[i]))->comm,&ksp);CHKERRQ(ierr);
      ierr = PetscLogObjectParent(pc,ksp);CHKERRQ(ierr);
      ierr = PetscObjectIncrementTabLevel((PetscObject)ksp,(PetscObject)pc,1);CHKERRQ(ierr);
      ierr = KSPSetType(ksp,KSPPREONLY);CHKERRQ(ierr);
      ierr = KSPGetPC(ksp,&subpc);CHKERRQ(ierr);
      ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);
      ierr = KSPSetOptionsPrefix(ksp,prefix);CHKERRQ(ierr);
      ierr = KSPAppendOptionsPrefix(ksp,"sub_");CHKERRQ(ierr);
      osm->ksp[i] = ksp;
    }
    scall = MAT_INITIAL_MATRIX;

  } else {
    /* 
       Destroy the blocks from the previous iteration
    */
    if (pc->flag == DIFFERENT_NONZERO_PATTERN) {
      ierr = MatDestroyMatrices(osm->n,&osm->pmat);CHKERRQ(ierr);
      scall = MAT_INITIAL_MATRIX;
    }
  }

  /* 
     Extract out the submatrices. 
  */
  if(size > 1) {
    ierr = MatGetSubMatricesParallel(pc->pmat,osm->n,osm->is, osm->is,scall,&osm->pmat);CHKERRQ(ierr);
  }
  else {
    ierr = MatGetSubMatrices(pc->pmat,osm->n,osm->is, osm->is,scall,&osm->pmat);CHKERRQ(ierr);
  }
  if (scall == MAT_INITIAL_MATRIX) {
    ierr = PetscObjectGetOptionsPrefix((PetscObject)pc->pmat,&pprefix);CHKERRQ(ierr);
    for (i=0; i<osm->n; i++) {
      ierr = PetscLogObjectParent(pc,osm->pmat[i]);CHKERRQ(ierr);
      ierr = PetscObjectSetOptionsPrefix((PetscObject)osm->pmat[i],pprefix);CHKERRQ(ierr);
    }
  }
  
  /* Return control to the user so that the submatrices can be modified (e.g., to apply
     different boundary conditions for the submatrices than for the global problem) */
  ierr = PCModifySubMatrices(pc,osm->n,osm->is,osm->is,osm->pmat,pc->modifysubmatricesP);CHKERRQ(ierr);

  /* 
     Loop over submatrices putting them into local ksp
  */
  for (i=0; i<osm->n; i++) {
    ierr = KSPSetOperators(osm->ksp[i],osm->pmat[i],osm->pmat[i],pc->flag);CHKERRQ(ierr);
    if (!pc->setupcalled) {
      ierr = KSPSetFromOptions(osm->ksp[i]);CHKERRQ(ierr);
    }
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetUpOnBlocks_GASM"
static PetscErrorCode PCSetUpOnBlocks_GASM(PC pc)
{
  PC_GASM         *osm = (PC_GASM*)pc->data;
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  for (i=0; i<osm->n; i++) {
    ierr = KSPSetUp(osm->ksp[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApply_GASM"
static PetscErrorCode PCApply_GASM(PC pc,Vec x,Vec y)
{
  PC_GASM         *osm = (PC_GASM*)pc->data;
  PetscErrorCode ierr;
  PetscInt       i;
  ScatterMode    forward = SCATTER_FORWARD,reverse = SCATTER_REVERSE;

  PetscFunctionBegin;
  /*
     Support for limiting the restriction or interpolation to only local 
     subdomain values (leaving the other values 0). 
  */
  if (!(osm->type & PC_GASM_RESTRICT)) {
    forward = SCATTER_FORWARD_LOCAL;
    /* have to zero the work RHS since scatter may leave some slots empty */
    ierr = VecZeroEntries(osm->gx);CHKERRQ(ierr);
  }
  if (!(osm->type & PC_GASM_INTERPOLATE)) {
    reverse = SCATTER_REVERSE_LOCAL;
  }

  ierr = VecScatterBegin(osm->grestriction,x,osm->gx,INSERT_VALUES,forward);CHKERRQ(ierr);
  ierr = VecZeroEntries(y);CHKERRQ(ierr);
  ierr = VecScatterEnd(osm->grestriction,x,osm->gx,INSERT_VALUES,forward);CHKERRQ(ierr);
  /* do the subdomain solves */
  for (i=0; i<osm->n; ++i) { 
    ierr = KSPSolve(osm->ksp[i],osm->x[i],osm->y[i]);CHKERRQ(ierr);
  }
  ierr = VecScatterBegin(osm->gprolongation,osm->gy,y,ADD_VALUES,reverse);CHKERRQ(ierr);
  ierr = VecScatterEnd(osm->gprolongation,osm->gy,y,ADD_VALUES,reverse);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApplyTranspose_GASM"
static PetscErrorCode PCApplyTranspose_GASM(PC pc,Vec x,Vec y)
{
  PC_GASM         *osm = (PC_GASM*)pc->data;
  PetscErrorCode ierr;
  PetscInt       i;
  ScatterMode    forward = SCATTER_FORWARD,reverse = SCATTER_REVERSE;

  PetscFunctionBegin;
  /*
     Support for limiting the restriction or interpolation to only local 
     subdomain values (leaving the other values 0).

     Note: these are reversed from the PCApply_GASM() because we are applying the 
     transpose of the three terms 
  */
  if (!(osm->type & PC_GASM_INTERPOLATE)) {
    forward = SCATTER_FORWARD_LOCAL;
    /* have to zero the work RHS since scatter may leave some slots empty */
    ierr = VecZeroEntries(osm->gx);CHKERRQ(ierr);
  }
  if (!(osm->type & PC_GASM_RESTRICT)) {
    reverse = SCATTER_REVERSE_LOCAL;
  }

  ierr = VecScatterBegin(osm->grestriction,x,osm->gx,INSERT_VALUES,forward);CHKERRQ(ierr);
  ierr = VecZeroEntries(y);CHKERRQ(ierr);
  ierr = VecScatterEnd(osm->grestriction,x,osm->gx,INSERT_VALUES,forward);CHKERRQ(ierr);
  /* do the local solves */
  for (i=0; i<osm->n; ++i) { /* Note that the solves are local, so we can go to osm->n, rather than osm->nmax. */
    ierr = KSPSolveTranspose(osm->ksp[i],osm->x[i],osm->y[i]);CHKERRQ(ierr); 
  }
  ierr = VecScatterBegin(osm->gprolongation,osm->gy,y,ADD_VALUES,reverse);CHKERRQ(ierr);
  ierr = VecScatterEnd(osm->gprolongation,osm->gy,y,ADD_VALUES,reverse);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCReset_GASM"
static PetscErrorCode PCReset_GASM(PC pc)
{
  PC_GASM        *osm = (PC_GASM*)pc->data;
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  if (osm->ksp) {
    for (i=0; i<osm->n; i++) {
      ierr = KSPReset(osm->ksp[i]);CHKERRQ(ierr);
    }
  }
  if (osm->pmat) {
    if (osm->n > 0) {
      ierr = MatDestroyMatrices(osm->n,&osm->pmat);CHKERRQ(ierr);
    }
  }
  if (osm->x) {
    for (i=0; i<osm->n; i++) {
      ierr = VecDestroy(&osm->x[i]);CHKERRQ(ierr);
      ierr = VecDestroy(&osm->y[i]);CHKERRQ(ierr);
    }
  }
  ierr = VecDestroy(&osm->gx);CHKERRQ(ierr);
  ierr = VecDestroy(&osm->gy);CHKERRQ(ierr);
  
  ierr = VecScatterDestroy(&osm->grestriction);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&osm->gprolongation);CHKERRQ(ierr);
  if (osm->is) {
    ierr = PCGASMDestroySubdomains(osm->n,osm->is,osm->is_local);CHKERRQ(ierr); 
    osm->is = 0;
    osm->is_local = 0;
  }
  ierr = ISDestroy(&osm->gis);CHKERRQ(ierr);
  ierr = ISDestroy(&osm->gis_local);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCDestroy_GASM"
static PetscErrorCode PCDestroy_GASM(PC pc)
{
  PC_GASM         *osm = (PC_GASM*)pc->data;
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  ierr = PCReset_GASM(pc);CHKERRQ(ierr);
  if (osm->ksp) {
    for (i=0; i<osm->n; i++) {
      ierr = KSPDestroy(&osm->ksp[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(osm->ksp);CHKERRQ(ierr);
  }
  ierr = PetscFree(osm->x);CHKERRQ(ierr);
  ierr = PetscFree(osm->y);CHKERRQ(ierr);
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetFromOptions_GASM"
static PetscErrorCode PCSetFromOptions_GASM(PC pc) {
  PC_GASM         *osm = (PC_GASM*)pc->data;
  PetscErrorCode ierr;
  PetscInt       blocks,ovl;
  PetscBool      symset,flg;
  PCGASMType      gasmtype;

  PetscFunctionBegin;
  /* set the type to symmetric if matrix is symmetric */
  if (!osm->type_set && pc->pmat) {
    ierr = MatIsSymmetricKnown(pc->pmat,&symset,&flg);CHKERRQ(ierr);
    if (symset && flg) { osm->type = PC_GASM_BASIC; }
  }
  ierr = PetscOptionsHead("Generalized additive Schwarz options");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-pc_gasm_blocks","Number of subdomains","PCGASMSetTotalSubdomains",osm->n,&blocks,&flg);CHKERRQ(ierr);
    if (flg) {ierr = PCGASMSetTotalSubdomains(pc,blocks);CHKERRQ(ierr); }
    ierr = PetscOptionsInt("-pc_gasm_overlap","Number of grid points overlap","PCGASMSetOverlap",osm->overlap,&ovl,&flg);CHKERRQ(ierr);
    if (flg) {ierr = PCGASMSetOverlap(pc,ovl);CHKERRQ(ierr); }
    flg  = PETSC_FALSE;
    ierr = PetscOptionsEnum("-pc_gasm_type","Type of restriction/extension","PCGASMSetType",PCGASMTypes,(PetscEnum)osm->type,(PetscEnum*)&gasmtype,&flg);CHKERRQ(ierr);
    if (flg) {ierr = PCGASMSetType(pc,gasmtype);CHKERRQ(ierr); }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------------------------------*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCGASMSetLocalSubdomains_GASM"
PetscErrorCode  PCGASMSetLocalSubdomains_GASM(PC pc,PetscInt n,IS is[],IS is_local[])
{
  PC_GASM         *osm = (PC_GASM*)pc->data;
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  if (n < 1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Each process must have 1 or more subdomains, n = %D",n);
  if (pc->setupcalled && (n != osm->n || is)) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_WRONGSTATE,"PCGASMSetLocalSubdomains() should be called before calling PCSetUp().");

  if (!pc->setupcalled) {
    osm->n            = n;
    osm->is           = 0;
    osm->is_local     = 0;
    if (is) {
      for (i=0; i<n; i++) {ierr = PetscObjectReference((PetscObject)is[i]);CHKERRQ(ierr);}
    }
    if (is_local) {
      for (i=0; i<n; i++) {ierr = PetscObjectReference((PetscObject)is_local[i]);CHKERRQ(ierr);}
    }
    if (osm->is) {
      ierr = PCGASMDestroySubdomains(osm->n,osm->is,osm->is_local);CHKERRQ(ierr);
    }
    if (is) {
      ierr = PetscMalloc(n*sizeof(IS),&osm->is);CHKERRQ(ierr);
      for (i=0; i<n; i++) { osm->is[i] = is[i]; }
      /* Flag indicating that the user has set overlapping subdomains so PCGASM should not increase their size. */
      osm->overlap = -1;
    }
    if (is_local) {
      ierr = PetscMalloc(n*sizeof(IS),&osm->is_local);CHKERRQ(ierr);
      for (i=0; i<n; i++) { osm->is_local[i] = is_local[i]; }
    }
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCGASMSetTotalSubdomains_GASM"
PetscErrorCode  PCGASMSetTotalSubdomains_GASM(PC pc,PetscInt N) {
  PC_GASM         *osm = (PC_GASM*)pc->data;
  PetscErrorCode ierr;
  PetscMPIInt    rank,size;
  PetscInt       n;

  PetscFunctionBegin;
  if (N < 1) SETERRQ1(((PetscObject)pc)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Number of total blocks must be > 0, N = %D",N);

  /*
     Split the subdomains equally among all processors
  */
  ierr = MPI_Comm_rank(((PetscObject)pc)->comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(((PetscObject)pc)->comm,&size);CHKERRQ(ierr);
  n = N/size + ((N % size) > rank);
  if (!n) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Process %d must have at least one block: total processors %d total blocks %D",(int)rank,(int)size,N);
  if (pc->setupcalled && n != osm->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"PCGASMSetTotalSubdomains() should be called before PCSetUp().");
  if (!pc->setupcalled) {
    if (osm->is) {
      ierr = PCGASMDestroySubdomains(osm->n,osm->is,osm->is_local);CHKERRQ(ierr);
    }
    osm->N            = N;
    osm->n            = n;
    osm->is           = 0;
    osm->is_local     = 0;
  }
  PetscFunctionReturn(0);
}/* PCGASMSetTotalSubdomains_GASM() */
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCGASMSetOverlap_GASM"
PetscErrorCode  PCGASMSetOverlap_GASM(PC pc,PetscInt ovl)
{
  PC_GASM *osm = (PC_GASM*)pc->data;

  PetscFunctionBegin;
  if (ovl < 0) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Negative overlap value requested");
  if (pc->setupcalled && ovl != osm->overlap) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_WRONGSTATE,"PCGASMSetOverlap() should be called before PCSetUp().");
  if (!pc->setupcalled) {
    osm->overlap = ovl;
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCGASMSetType_GASM"
PetscErrorCode  PCGASMSetType_GASM(PC pc,PCGASMType type)
{
  PC_GASM *osm = (PC_GASM*)pc->data;

  PetscFunctionBegin;
  osm->type     = type;
  osm->type_set = PETSC_TRUE;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCGASMSetSortIndices_GASM"
PetscErrorCode  PCGASMSetSortIndices_GASM(PC pc,PetscBool  doSort)
{
  PC_GASM *osm = (PC_GASM*)pc->data;

  PetscFunctionBegin;
  osm->sort_indices = doSort;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCGASMGetSubKSP_GASM"
/* 
   FIX: This routine might need to be modified once multiple ranks per subdomain are allowed.
        In particular, it would upset the global subdomain number calculation.
*/
PetscErrorCode  PCGASMGetSubKSP_GASM(PC pc,PetscInt *n,PetscInt *first,KSP **ksp) 
{
  PC_GASM         *osm = (PC_GASM*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (osm->n < 1) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ORDER,"Need to call PCSetUP() on PC (or KSPSetUp() on the outer KSP object) before calling here");

  if (n) {
    *n = osm->n;
  }
  if (first) {
    ierr = MPI_Scan(&osm->n,first,1,MPIU_INT,MPI_SUM,((PetscObject)pc)->comm);CHKERRQ(ierr);
    *first -= osm->n;
  }
  if (ksp) {
    /* Assume that local solves are now different; not necessarily
       true though!  This flag is used only for PCView_GASM() */
    *ksp                   = osm->ksp;
    osm->same_local_solves = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}/* PCGASMGetSubKSP_GASM() */
EXTERN_C_END


#undef __FUNCT__  
#define __FUNCT__ "PCGASMSetLocalSubdomains"
/*@C
    PCGASMSetLocalSubdomains - Sets the local subdomains (for this processor
    only) for the additive Schwarz preconditioner. 

    Collective on PC 

    Input Parameters:
+   pc - the preconditioner context
.   n - the number of subdomains for this processor (default value = 1)
.   is - the index set that defines the subdomains for this processor
         (or PETSC_NULL for PETSc to determine subdomains)
-   is_local - the index sets that define the local part of the subdomains for this processor
         (or PETSC_NULL to use the default of 1 subdomain per process)

    Notes:
    The IS numbering is in the parallel, global numbering of the vector.

    By default the GASM preconditioner uses 1 block per processor.  

    Use PCGASMSetTotalSubdomains() to set the subdomains for all processors.

    Level: advanced

.keywords: PC, GASM, set, local, subdomains, additive Schwarz

.seealso: PCGASMSetTotalSubdomains(), PCGASMSetOverlap(), PCGASMGetSubKSP(),
          PCGASMCreateSubdomains2D(), PCGASMGetLocalSubdomains()
@*/
PetscErrorCode  PCGASMSetLocalSubdomains(PC pc,PetscInt n,IS is[],IS is_local[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCGASMSetLocalSubdomains_C",(PC,PetscInt,IS[],IS[]),(pc,n,is,is_local));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCGASMSetTotalSubdomains"
/*@C
    PCGASMSetTotalSubdomains - Sets the subdomains for all processor for the 
    additive Schwarz preconditioner.  Either all or no processors in the
    PC communicator must call this routine, with the same index sets.

    Collective on PC

    Input Parameters:
+   pc - the preconditioner context
.   n - the number of subdomains for all processors
.   is - the index sets that define the subdomains for all processor
         (or PETSC_NULL for PETSc to determine subdomains)
-   is_local - the index sets that define the local part of the subdomains for this processor
         (or PETSC_NULL to use the default of 1 subdomain per process)

    Options Database Key:
    To set the total number of subdomain blocks rather than specify the
    index sets, use the option
.    -pc_gasm_blocks <blks> - Sets total blocks

    Notes:
    Currently you cannot use this to set the actual subdomains with the argument is.

    By default the GASM preconditioner uses 1 block per processor.  

    These index sets cannot be destroyed until after completion of the
    linear solves for which the GASM preconditioner is being used.

    Use PCGASMSetLocalSubdomains() to set local subdomains.

    Level: advanced

.keywords: PC, GASM, set, total, global, subdomains, additive Schwarz

.seealso: PCGASMSetLocalSubdomains(), PCGASMSetOverlap(), PCGASMGetSubKSP(),
          PCGASMCreateSubdomains2D()
@*/
PetscErrorCode  PCGASMSetTotalSubdomains(PC pc,PetscInt N)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCGASMSetTotalSubdomains_C",(PC,PetscInt),(pc,N));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCGASMSetOverlap"
/*@
    PCGASMSetOverlap - Sets the overlap between a pair of subdomains for the
    additive Schwarz preconditioner.  Either all or no processors in the
    PC communicator must call this routine. 

    Logically Collective on PC

    Input Parameters:
+   pc  - the preconditioner context
-   ovl - the amount of overlap between subdomains (ovl >= 0, default value = 1)

    Options Database Key:
.   -pc_gasm_overlap <ovl> - Sets overlap

    Notes:
    By default the GASM preconditioner uses 1 block per processor.  To use
    multiple blocks per perocessor, see PCGASMSetTotalSubdomains() and
    PCGASMSetLocalSubdomains() (and the option -pc_gasm_blocks <blks>).

    The overlap defaults to 1, so if one desires that no additional
    overlap be computed beyond what may have been set with a call to
    PCGASMSetTotalSubdomains() or PCGASMSetLocalSubdomains(), then ovl
    must be set to be 0.  In particular, if one does not explicitly set
    the subdomains an application code, then all overlap would be computed
    internally by PETSc, and using an overlap of 0 would result in an GASM 
    variant that is equivalent to the block Jacobi preconditioner.  

    Note that one can define initial index sets with any overlap via
    PCGASMSetTotalSubdomains() or PCGASMSetLocalSubdomains(); the routine
    PCGASMSetOverlap() merely allows PETSc to extend that overlap further
    if desired.

    Level: intermediate

.keywords: PC, GASM, set, overlap

.seealso: PCGASMSetTotalSubdomains(), PCGASMSetLocalSubdomains(), PCGASMGetSubKSP(),
          PCGASMCreateSubdomains2D(), PCGASMGetLocalSubdomains()
@*/
PetscErrorCode  PCGASMSetOverlap(PC pc,PetscInt ovl)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveInt(pc,ovl,2);
  ierr = PetscTryMethod(pc,"PCGASMSetOverlap_C",(PC,PetscInt),(pc,ovl));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCGASMSetType"
/*@
    PCGASMSetType - Sets the type of restriction and interpolation used
    for local problems in the additive Schwarz method.

    Logically Collective on PC

    Input Parameters:
+   pc  - the preconditioner context
-   type - variant of GASM, one of
.vb
      PC_GASM_BASIC       - full interpolation and restriction
      PC_GASM_RESTRICT    - full restriction, local processor interpolation
      PC_GASM_INTERPOLATE - full interpolation, local processor restriction
      PC_GASM_NONE        - local processor restriction and interpolation
.ve

    Options Database Key:
.   -pc_gasm_type [basic,restrict,interpolate,none] - Sets GASM type

    Level: intermediate

.keywords: PC, GASM, set, type

.seealso: PCGASMSetTotalSubdomains(), PCGASMSetTotalSubdomains(), PCGASMGetSubKSP(),
          PCGASMCreateSubdomains2D()
@*/
PetscErrorCode  PCGASMSetType(PC pc,PCGASMType type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveEnum(pc,type,2);
  ierr = PetscTryMethod(pc,"PCGASMSetType_C",(PC,PCGASMType),(pc,type));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCGASMSetSortIndices"
/*@
    PCGASMSetSortIndices - Determines whether subdomain indices are sorted.

    Logically Collective on PC

    Input Parameters:
+   pc  - the preconditioner context
-   doSort - sort the subdomain indices

    Level: intermediate

.keywords: PC, GASM, set, type

.seealso: PCGASMSetLocalSubdomains(), PCGASMSetTotalSubdomains(), PCGASMGetSubKSP(),
          PCGASMCreateSubdomains2D()
@*/
PetscErrorCode  PCGASMSetSortIndices(PC pc,PetscBool  doSort)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveBool(pc,doSort,2);
  ierr = PetscTryMethod(pc,"PCGASMSetSortIndices_C",(PC,PetscBool),(pc,doSort));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCGASMGetSubKSP"
/*@C
   PCGASMGetSubKSP - Gets the local KSP contexts for all blocks on
   this processor.
   
   Collective on PC iff first_local is requested

   Input Parameter:
.  pc - the preconditioner context

   Output Parameters:
+  n_local - the number of blocks on this processor or PETSC_NULL
.  first_local - the global number of the first block on this processor or PETSC_NULL,
                 all processors must request or all must pass PETSC_NULL
-  ksp - the array of KSP contexts

   Note:  
   After PCGASMGetSubKSP() the array of KSPes is not to be freed

   Currently for some matrix implementations only 1 block per processor 
   is supported.
   
   You must call KSPSetUp() before calling PCGASMGetSubKSP().

   Level: advanced

.keywords: PC, GASM, additive Schwarz, get, sub, KSP, context

.seealso: PCGASMSetTotalSubdomains(), PCGASMSetTotalSubdomains(), PCGASMSetOverlap(),
          PCGASMCreateSubdomains2D(),
@*/
PetscErrorCode  PCGASMGetSubKSP(PC pc,PetscInt *n_local,PetscInt *first_local,KSP *ksp[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscUseMethod(pc,"PCGASMGetSubKSP_C",(PC,PetscInt*,PetscInt*,KSP **),(pc,n_local,first_local,ksp));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------------*/
/*MC
   PCGASM - Use the (restricted) additive Schwarz method, each block is (approximately) solved with 
           its own KSP object.

   Options Database Keys:
+  -pc_gasm_truelocal - Activates PCGGASMSetUseTrueLocal()
.  -pc_gasm_blocks <blks> - Sets total blocks
.  -pc_gasm_overlap <ovl> - Sets overlap
-  -pc_gasm_type [basic,restrict,interpolate,none] - Sets GASM type

     IMPORTANT: If you run with, for example, 3 blocks on 1 processor or 3 blocks on 3 processors you 
      will get a different convergence rate due to the default option of -pc_gasm_type restrict. Use
      -pc_gasm_type basic to use the standard GASM. 

   Notes: Each processor can have one or more blocks, but a block cannot be shared by more
     than one processor. Defaults to one block per processor.

     To set options on the solvers for each block append -sub_ to all the KSP, and PC
        options database keys. For example, -sub_pc_type ilu -sub_pc_factor_levels 1 -sub_ksp_type preonly
        
     To set the options on the solvers separate for each block call PCGASMGetSubKSP()
         and set the options directly on the resulting KSP object (you can access its PC
         with KSPGetPC())


   Level: beginner

   Concepts: additive Schwarz method

    References:
    An additive variant of the Schwarz alternating method for the case of many subregions
    M Dryja, OB Widlund - Courant Institute, New York University Technical report

    Domain Decompositions: Parallel Multilevel Methods for Elliptic Partial Differential Equations, 
    Barry Smith, Petter Bjorstad, and William Gropp, Cambridge University Press, ISBN 0-521-49589-X.

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC,
           PCBJACOBI, PCGASMSetUseTrueLocal(), PCGASMGetSubKSP(), PCGASMSetLocalSubdomains(),
           PCGASMSetTotalSubdomains(), PCSetModifySubmatrices(), PCGASMSetOverlap(), PCGASMSetType()

M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCreate_GASM"
PetscErrorCode  PCCreate_GASM(PC pc)
{
  PetscErrorCode ierr;
  PC_GASM         *osm;

  PetscFunctionBegin;
  ierr = PetscNewLog(pc,PC_GASM,&osm);CHKERRQ(ierr);
  osm->N                 = PETSC_DECIDE;
  osm->n                 = 0;
  osm->nmax              = 0;
  osm->overlap           = 1;
  osm->ksp               = 0;
  osm->grestriction      = 0;
  osm->gprolongation     = 0;
  osm->gx                = 0;
  osm->gy                = 0;
  osm->x                 = 0;
  osm->y                 = 0;
  osm->is                = 0;
  osm->is_local          = 0;
  osm->pmat              = 0;
  osm->type              = PC_GASM_RESTRICT;
  osm->same_local_solves = PETSC_TRUE;
  osm->sort_indices      = PETSC_TRUE;

  pc->data                   = (void*)osm;
  pc->ops->apply             = PCApply_GASM;
  pc->ops->applytranspose    = PCApplyTranspose_GASM;
  pc->ops->setup             = PCSetUp_GASM;
  pc->ops->reset             = PCReset_GASM;
  pc->ops->destroy           = PCDestroy_GASM;
  pc->ops->setfromoptions    = PCSetFromOptions_GASM;
  pc->ops->setuponblocks     = PCSetUpOnBlocks_GASM;
  pc->ops->view              = PCView_GASM;
  pc->ops->applyrichardson   = 0;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCGASMSetLocalSubdomains_C","PCGASMSetLocalSubdomains_GASM",
                    PCGASMSetLocalSubdomains_GASM);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCGASMSetTotalSubdomains_C","PCGASMSetTotalSubdomains_GASM",
                    PCGASMSetTotalSubdomains_GASM);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCGASMSetOverlap_C","PCGASMSetOverlap_GASM",
                    PCGASMSetOverlap_GASM);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCGASMSetType_C","PCGASMSetType_GASM",
                    PCGASMSetType_GASM);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCGASMSetSortIndices_C","PCGASMSetSortIndices_GASM",
                    PCGASMSetSortIndices_GASM);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCGASMGetSubKSP_C","PCGASMGetSubKSP_GASM",
                    PCGASMGetSubKSP_GASM);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END


#undef __FUNCT__  
#define __FUNCT__ "PCGASMCreateSubdomains"
/*@C
   PCGASMCreateSubdomains - Creates the index sets for the overlapping Schwarz 
   preconditioner for a any problem on a general grid.

   Collective

   Input Parameters:
+  A - The global matrix operator
-  n - the number of local blocks

   Output Parameters:
.  outis - the array of index sets defining the subdomains

   Level: advanced

   Note: this generates nonoverlapping subdomains; the PCGASM will generate the overlap
    from these if you use PCGASMSetLocalSubdomains()

    In the Fortran version you must provide the array outis[] already allocated of length n.

.keywords: PC, GASM, additive Schwarz, create, subdomains, unstructured grid

.seealso: PCGASMSetLocalSubdomains(), PCGASMDestroySubdomains()
@*/
PetscErrorCode  PCGASMCreateSubdomains(Mat A, PetscInt n, IS* outis[])
{
  MatPartitioning           mpart;
  const char                *prefix;
  PetscErrorCode            (*f)(Mat,MatReuse,Mat*);
  PetscMPIInt               size;
  PetscInt                  i,j,rstart,rend,bs;
  PetscBool                 isbaij = PETSC_FALSE,foundpart = PETSC_FALSE;
  Mat                       Ad = PETSC_NULL, adj;
  IS                        ispart,isnumb,*is;
  PetscErrorCode            ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(outis,3);
  if (n < 1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"number of local blocks must be > 0, n = %D",n);

  /* Get prefix, row distribution, and block size */
  ierr = MatGetOptionsPrefix(A,&prefix);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
  ierr = MatGetBlockSize(A,&bs);CHKERRQ(ierr);
  if (rstart/bs*bs != rstart || rend/bs*bs != rend) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"bad row distribution [%D,%D) for matrix block size %D",rstart,rend,bs);

  /* Get diagonal block from matrix if possible */
  ierr = MPI_Comm_size(((PetscObject)A)->comm,&size);CHKERRQ(ierr);
  ierr = PetscObjectQueryFunction((PetscObject)A,"MatGetDiagonalBlock_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = MatGetDiagonalBlock(A,&Ad);CHKERRQ(ierr);
  } else if (size == 1) {
    Ad = A;
  }
  if (Ad) {
    ierr = PetscTypeCompare((PetscObject)Ad,MATSEQBAIJ,&isbaij);CHKERRQ(ierr);
    if (!isbaij) {ierr = PetscTypeCompare((PetscObject)Ad,MATSEQSBAIJ,&isbaij);CHKERRQ(ierr);}
  }
  if (Ad && n > 1) {
    PetscBool  match,done;
    /* Try to setup a good matrix partitioning if available */
    ierr = MatPartitioningCreate(PETSC_COMM_SELF,&mpart);CHKERRQ(ierr);
    ierr = PetscObjectSetOptionsPrefix((PetscObject)mpart,prefix);CHKERRQ(ierr);
    ierr = MatPartitioningSetFromOptions(mpart);CHKERRQ(ierr);
    ierr = PetscTypeCompare((PetscObject)mpart,MATPARTITIONINGCURRENT,&match);CHKERRQ(ierr);
    if (!match) {
      ierr = PetscTypeCompare((PetscObject)mpart,MATPARTITIONINGSQUARE,&match);CHKERRQ(ierr);
    }
    if (!match) { /* assume a "good" partitioner is available */
      PetscInt na,*ia,*ja;
      ierr = MatGetRowIJ(Ad,0,PETSC_TRUE,isbaij,&na,&ia,&ja,&done);CHKERRQ(ierr);
      if (done) {
        /* Build adjacency matrix by hand. Unfortunately a call to
           MatConvert(Ad,MATMPIADJ,MAT_INITIAL_MATRIX,&adj) will
           remove the block-aij structure and we cannot expect
           MatPartitioning to split vertices as we need */
        PetscInt i,j,*row,len,nnz,cnt,*iia=0,*jja=0;
        nnz = 0;
        for (i=0; i<na; i++) { /* count number of nonzeros */
          len = ia[i+1] - ia[i];
          row = ja + ia[i];
          for (j=0; j<len; j++) {
            if (row[j] == i) { /* don't count diagonal */
              len--; break;
            }
          }
          nnz += len;
        }
        ierr = PetscMalloc((na+1)*sizeof(PetscInt),&iia);CHKERRQ(ierr);
        ierr = PetscMalloc((nnz)*sizeof(PetscInt),&jja);CHKERRQ(ierr);
        nnz    = 0;
        iia[0] = 0;
        for (i=0; i<na; i++) { /* fill adjacency */
          cnt = 0;
          len = ia[i+1] - ia[i];
          row = ja + ia[i];
          for (j=0; j<len; j++) {
            if (row[j] != i) { /* if not diagonal */
              jja[nnz+cnt++] = row[j];
            }
          }
          nnz += cnt;
          iia[i+1] = nnz;
        }
        /* Partitioning of the adjacency matrix */
        ierr = MatCreateMPIAdj(PETSC_COMM_SELF,na,na,iia,jja,PETSC_NULL,&adj);CHKERRQ(ierr);
        ierr = MatPartitioningSetAdjacency(mpart,adj);CHKERRQ(ierr);
        ierr = MatPartitioningSetNParts(mpart,n);CHKERRQ(ierr);
        ierr = MatPartitioningApply(mpart,&ispart);CHKERRQ(ierr);
        ierr = ISPartitioningToNumbering(ispart,&isnumb);CHKERRQ(ierr);
        ierr = MatDestroy(&adj);CHKERRQ(ierr);
        foundpart = PETSC_TRUE;
      }
      ierr = MatRestoreRowIJ(Ad,0,PETSC_TRUE,isbaij,&na,&ia,&ja,&done);CHKERRQ(ierr);
    }
    ierr = MatPartitioningDestroy(&mpart);CHKERRQ(ierr);
  }

  ierr = PetscMalloc(n*sizeof(IS),&is);CHKERRQ(ierr);
  *outis = is;

  if (!foundpart) {

    /* Partitioning by contiguous chunks of rows */

    PetscInt mbs   = (rend-rstart)/bs;
    PetscInt start = rstart;
    for (i=0; i<n; i++) {
      PetscInt count = (mbs/n + ((mbs % n) > i)) * bs;
      ierr   = ISCreateStride(PETSC_COMM_SELF,count,start,1,&is[i]);CHKERRQ(ierr);
      start += count;
    }

  } else {

    /* Partitioning by adjacency of diagonal block  */

    const PetscInt *numbering;
    PetscInt       *count,nidx,*indices,*newidx,start=0;
    /* Get node count in each partition */
    ierr = PetscMalloc(n*sizeof(PetscInt),&count);CHKERRQ(ierr);
    ierr = ISPartitioningCount(ispart,n,count);CHKERRQ(ierr);
    if (isbaij && bs > 1) { /* adjust for the block-aij case */
      for (i=0; i<n; i++) count[i] *= bs;
    }
    /* Build indices from node numbering */
    ierr = ISGetLocalSize(isnumb,&nidx);CHKERRQ(ierr);
    ierr = PetscMalloc(nidx*sizeof(PetscInt),&indices);CHKERRQ(ierr);
    for (i=0; i<nidx; i++) indices[i] = i; /* needs to be initialized */
    ierr = ISGetIndices(isnumb,&numbering);CHKERRQ(ierr);
    ierr = PetscSortIntWithPermutation(nidx,numbering,indices);CHKERRQ(ierr);
    ierr = ISRestoreIndices(isnumb,&numbering);CHKERRQ(ierr);
    if (isbaij && bs > 1) { /* adjust for the block-aij case */
      ierr = PetscMalloc(nidx*bs*sizeof(PetscInt),&newidx);CHKERRQ(ierr);
      for (i=0; i<nidx; i++)
        for (j=0; j<bs; j++)
          newidx[i*bs+j] = indices[i]*bs + j;
      ierr = PetscFree(indices);CHKERRQ(ierr);
      nidx   *= bs;
      indices = newidx;
    }
    /* Shift to get global indices */
    for (i=0; i<nidx; i++) indices[i] += rstart;

    /* Build the index sets for each block */
    for (i=0; i<n; i++) {
      ierr   = ISCreateGeneral(PETSC_COMM_SELF,count[i],&indices[start],PETSC_COPY_VALUES,&is[i]);CHKERRQ(ierr);
      ierr   = ISSort(is[i]);CHKERRQ(ierr);
      start += count[i];
    }

    ierr = PetscFree(count);
    ierr = PetscFree(indices);
    ierr = ISDestroy(&isnumb);CHKERRQ(ierr);
    ierr = ISDestroy(&ispart);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCGASMDestroySubdomains"
/*@C
   PCGASMDestroySubdomains - Destroys the index sets created with
   PCGASMCreateSubdomains(). Should be called after setting subdomains
   with PCGASMSetLocalSubdomains().

   Collective

   Input Parameters:
+  n - the number of index sets
.  is - the array of index sets
-  is_local - the array of local index sets, can be PETSC_NULL

   Level: advanced

.keywords: PC, GASM, additive Schwarz, create, subdomains, unstructured grid

.seealso: PCGASMCreateSubdomains(), PCGASMSetLocalSubdomains()
@*/
PetscErrorCode  PCGASMDestroySubdomains(PetscInt n, IS is[], IS is_local[])
{
  PetscInt       i;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (n <= 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"n must be > 0: n = %D",n);
  PetscValidPointer(is,2);
  for (i=0; i<n; i++) { ierr = ISDestroy(&is[i]);CHKERRQ(ierr); }
  ierr = PetscFree(is);CHKERRQ(ierr);
  if (is_local) {
    PetscValidPointer(is_local,3);
    for (i=0; i<n; i++) { ierr = ISDestroy(&is_local[i]);CHKERRQ(ierr); }
    ierr = PetscFree(is_local);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#define PCGASMLocalSubdomainBounds2D(M,N,xleft,ylow,xright,yhigh,first,last,xleft_loc,ylow_loc,xright_loc,yhigh_loc,n) \
{                                                                                                       \
 PetscInt first_row = first/M, last_row = last/M+1;                                                     \
  /*                                                                                                    \
   Compute ylow_loc and yhigh_loc so that (ylow_loc,xleft) and (yhigh_loc,xright) are the corners       \
   of the bounding box of the intersection of the subdomain with the local ownership range (local       \
   subdomain).                                                                                          \
   Also compute xleft_loc and xright_loc as the lower and upper bounds on the first and last rows       \
   of the intersection.                                                                                 \
  */                                                                                                    \
  /* ylow_loc is the grid row containing the first element of the local sumbdomain */                   \
  *ylow_loc = PetscMax(first_row,ylow);                                                                    \
  /* xleft_loc is the offset of first element of the local subdomain within its grid row (might actually be outside the local subdomain) */ \
  *xleft_loc = *ylow_loc==first_row?PetscMax(first%M,xleft):xleft;                                                                            \
  /* yhigh_loc is the grid row above the last local subdomain element */                                                                    \
  *yhigh_loc = PetscMin(last_row,yhigh);                                                                                                     \
  /* xright is the offset of the end of the  local subdomain within its grid row (might actually be outside the local subdomain) */         \
  *xright_loc = *yhigh_loc==last_row?PetscMin(xright,last%M):xright;                                                                          \
  /* Now compute the size of the local subdomain n. */ \
  *n = 0;                                               \
  if(*ylow_loc < *yhigh_loc) {                           \
    PetscInt width = xright-xleft;                     \
    *n += width*(*yhigh_loc-*ylow_loc-1);                 \
    *n += PetscMin(PetscMax(*xright_loc-xleft,0),width); \
    *n -= PetscMin(PetscMax(*xleft_loc-xleft,0), width); \
  }\
}



#undef __FUNCT__  
#define __FUNCT__ "PCGASMCreateSubdomains2D"
/*@
   PCGASMCreateSubdomains2D - Creates the index sets for the overlapping Schwarz 
   preconditioner for a two-dimensional problem on a regular grid.

   Collective

   Input Parameters:
+  M, N - the global number of mesh points in the x and y directions
.  Mdomains, Ndomains - the global number of subdomains in the x and y directions
.  dof - degrees of freedom per node
-  overlap - overlap in mesh lines

   Output Parameters:
+  Nsub - the number of local subdomains created
.  is - array of index sets defining overlapping (if overlap > 0) subdomains
-  is_local - array of index sets defining non-overlapping subdomains

   Note:
   Presently PCAMSCreateSubdomains2d() is valid only for sequential
   preconditioners.  More general related routines are
   PCGASMSetTotalSubdomains() and PCGASMSetLocalSubdomains().

   Level: advanced

.keywords: PC, GASM, additive Schwarz, create, subdomains, 2D, regular grid

.seealso: PCGASMSetTotalSubdomains(), PCGASMSetLocalSubdomains(), PCGASMGetSubKSP(),
          PCGASMSetOverlap()
@*/
PetscErrorCode  PCGASMCreateSubdomains2D(PC pc, PetscInt M,PetscInt N,PetscInt Mdomains,PetscInt Ndomains,PetscInt dof,PetscInt overlap, PetscInt *nsub,IS **is,IS **is_local)
{
  PetscErrorCode ierr;
  PetscMPIInt    size, rank;
  PetscInt       i, j;
  PetscInt       maxheight, maxwidth;
  PetscInt       xstart, xleft, xright, xleft_loc, xright_loc;
  PetscInt       ystart, ylow,  yhigh,  ylow_loc,  yhigh_loc;
  PetscInt       x[2][2], y[2][2], n[2];
  PetscInt       first, last;
  PetscInt       nidx, *idx;
  PetscInt       ii,jj,s,q,d;
  PetscInt       k,kk;
  PetscMPIInt    color;
  MPI_Comm       comm, subcomm;
  IS             **iis = 0;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)pc, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(pc->pmat, &first, &last);CHKERRQ(ierr);
  if (first%dof || last%dof) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Matrix row partitioning unsuitable for domain decomposition: local row range (%D,%D) "
	     "does not respect the number of degrees of freedom per grid point %D", first, last, dof);

  /* Determine the number of domains with nonzero intersections with the local ownership range. */
  s = 0;
  ystart = 0;
  for (j=0; j<Ndomains; ++j) {
    maxheight = N/Ndomains + ((N % Ndomains) > j); /* Maximal height of subdomain */
    if (maxheight < 2) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Too many %D subdomains in the vertical directon for mesh height %D", Ndomains, N);
    /* Vertical domain limits with an overlap. */
    ylow = PetscMax(ystart - overlap,0);    
    yhigh = PetscMin(ystart + maxheight + overlap,N);
    xstart = 0;
    for (i=0; i<Mdomains; ++i) {
      maxwidth = M/Mdomains + ((M % Mdomains) > i); /* Maximal width of subdomain */
      if (maxwidth < 2) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Too many %D subdomains in the horizontal direction for mesh width %D", Mdomains, M);
      /* Horizontal domain limits with an overlap. */
      xleft   = PetscMax(xstart - overlap,0);
      xright  = PetscMin(xstart + maxwidth + overlap,M);
      /* 
	 Determine whether this subdomain intersects this processor's ownership range of pc->pmat.
      */
      PCGASMLocalSubdomainBounds2D(M,N,xleft,ylow,xright,yhigh,first,last,(&xleft_loc),(&ylow_loc),(&xright_loc),(&yhigh_loc),(&nidx));
      if(nidx) {
        ++s;
      }
      xstart += maxwidth;
    }/* for(i = 0; i < Mdomains; ++i) */
    ystart += maxheight;
  }/* for(j = 0; j < Ndomains; ++j) */
  /* Now we can allocate the necessary number of ISs. */
  *nsub = s;
  ierr = PetscMalloc((*nsub)*sizeof(IS*),is);CHKERRQ(ierr);
  ierr = PetscMalloc((*nsub)*sizeof(IS*),is_local);CHKERRQ(ierr);
  s = 0;
  ystart = 0;
  for (j=0; j<Ndomains; ++j) {
    maxheight = N/Ndomains + ((N % Ndomains) > j); /* Maximal height of subdomain */
    if (maxheight < 2) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Too many %D subdomains in the vertical directon for mesh height %D", Ndomains, N);
    /* Vertical domain limits with an overlap. */
    y[0][0] = PetscMax(ystart - overlap,0);    
    y[0][1] = PetscMin(ystart + maxheight + overlap,N);
    /* Vertical domain limits without an overlap. */
    y[1][0] = ystart;                      
    y[1][1] = PetscMin(ystart + maxheight,N); 
    xstart = 0;
    for (i=0; i<Mdomains; ++i) {
      maxwidth = M/Mdomains + ((M % Mdomains) > i); /* Maximal width of subdomain */
      if (maxwidth < 2) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Too many %D subdomains in the horizontal direction for mesh width %D", Mdomains, M);
      /* Horizontal domain limits with an overlap. */
      x[0][0]  = PetscMax(xstart - overlap,0);
      x[0][1]  = PetscMin(xstart + maxwidth + overlap,M);
      /* Horizontal domain limits without an overlap. */
      x[1][0] = xstart;                   
      x[1][1] = PetscMin(xstart+maxwidth,M); 
      /* 
	 Determine whether this domain intersects this processor's ownership range of pc->pmat.
	 Do this twice: first for the domains with overlaps, and once without.
	 During the first pass create the subcommunicators, and use them on the second pass as well.
      */
      for(q = 0; q < 2; ++q) {
	/*
	  domain limits, (xleft, xright) and (ylow, yheigh) are adjusted 
	  according to whether the domain with an overlap or without is considered. 
	*/
	xleft = x[q][0]; xright = x[q][1];
	ylow  = y[q][0]; yhigh  = y[q][1];
        PCGASMLocalSubdomainBounds2D(M,N,xleft,ylow,xright,yhigh,first,last,(&xleft_loc),(&ylow_loc),(&xright_loc),(&yhigh_loc),(&nidx));
	nidx *= dof;
        n[q] = nidx;
        /*
         Based on the counted number of indices in the local domain *with an overlap*,
         construct a subcommunicator of all the processors supporting this domain. 
         Observe that a domain with an overlap might have nontrivial local support,
         while the domain without an overlap might not.  Hence, the decision to participate
         in the subcommunicator must be based on the domain with an overlap.
         */
	if (q == 0) {
	  if(nidx) {
	    color = 1;
	  } else {
	    color = MPI_UNDEFINED;
	  }
	  ierr = MPI_Comm_split(comm, color, rank, &subcomm);CHKERRQ(ierr);
	}
        /*
         Proceed only if the number of local indices *with an overlap* is nonzero.
         */
        if (n[0]) {
          if(q == 0) {
            iis = is;
          }
          if (q == 1) {
            /* 
             The IS for the no-overlap subdomain shares a communicator with the overlapping domain.
             Moreover, if the overlap is zero, the two ISs are identical.
             */
            if (overlap == 0) {
              (*is_local)[s] = (*is)[s];
              ierr = PetscObjectReference((PetscObject)(*is)[s]);CHKERRQ(ierr);
              continue;
            } else {
              iis = is_local;
              subcomm = ((PetscObject)(*is)[s])->comm;
            }
          }/* if(q == 1) */
          idx = PETSC_NULL;
	  ierr = PetscMalloc(nidx*sizeof(PetscInt),&idx);CHKERRQ(ierr);
          if(nidx) {
            k    = 0;
            for (jj=ylow_loc; jj<yhigh_loc; ++jj) {
              PetscInt x0 = (jj==ylow_loc)?xleft_loc:xleft;
              PetscInt x1 = (jj==yhigh_loc-1)?xright_loc:xright;
              kk = dof*(M*jj + x0);
              for (ii=x0; ii<x1; ++ii) {
                for(d = 0; d < dof; ++d) {
                  idx[k++] = kk++;
                }
              }
            }
          }
	  ierr = ISCreateGeneral(subcomm,nidx,idx,PETSC_OWN_POINTER,(*iis)+s);CHKERRQ(ierr);
	}/* if(n[0]) */
      }/* for(q = 0; q < 2; ++q) */
      if(n[0]) {
        ++s;
      }
      xstart += maxwidth;
    }/* for(i = 0; i < Mdomains; ++i) */
    ystart += maxheight;
  }/* for(j = 0; j < Ndomains; ++j) */
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGASMGetLocalSubdomains"
/*@C
    PCGASMGetLocalSubdomains - Gets the local subdomains (for this processor
    only) for the additive Schwarz preconditioner. 

    Not Collective

    Input Parameter:
.   pc - the preconditioner context

    Output Parameters:
+   n - the number of subdomains for this processor (default value = 1)
.   is - the index sets that define the subdomains for this processor
-   is_local - the index sets that define the local part of the subdomains for this processor (can be PETSC_NULL)
         

    Notes:
    The IS numbering is in the parallel, global numbering of the vector.

    Level: advanced

.keywords: PC, GASM, set, local, subdomains, additive Schwarz

.seealso: PCGASMSetTotalSubdomains(), PCGASMSetOverlap(), PCGASMGetSubKSP(),
          PCGASMCreateSubdomains2D(), PCGASMSetLocalSubdomains(), PCGASMGetLocalSubmatrices()
@*/
PetscErrorCode  PCGASMGetLocalSubdomains(PC pc,PetscInt *n,IS *is[],IS *is_local[])
{
  PC_GASM         *osm;
  PetscErrorCode ierr;
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidIntPointer(n,2);
  if (is) PetscValidPointer(is,3);
  ierr = PetscTypeCompare((PetscObject)pc,PCGASM,&match);CHKERRQ(ierr);
  if (!match) {
    if (n)  *n  = 0;
    if (is) *is = PETSC_NULL;
  } else {
    osm = (PC_GASM*)pc->data;
    if (n)  *n  = osm->n;
    if (is) *is = osm->is;
    if (is_local) *is_local = osm->is_local;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCGASMGetLocalSubmatrices"
/*@C
    PCGASMGetLocalSubmatrices - Gets the local submatrices (for this processor
    only) for the additive Schwarz preconditioner. 

    Not Collective

    Input Parameter:
.   pc - the preconditioner context

    Output Parameters:
+   n - the number of matrices for this processor (default value = 1)
-   mat - the matrices
         

    Level: advanced

.keywords: PC, GASM, set, local, subdomains, additive Schwarz, block Jacobi

.seealso: PCGASMSetTotalSubdomains(), PCGASMSetOverlap(), PCGASMGetSubKSP(),
          PCGASMCreateSubdomains2D(), PCGASMSetLocalSubdomains(), PCGASMGetLocalSubdomains()
@*/
PetscErrorCode  PCGASMGetLocalSubmatrices(PC pc,PetscInt *n,Mat *mat[])
{
  PC_GASM         *osm;
  PetscErrorCode ierr;
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidIntPointer(n,2);
  if (mat) PetscValidPointer(mat,3);
  if (!pc->setupcalled) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_WRONGSTATE,"Must call after KSPSetUP() or PCSetUp().");
  ierr = PetscTypeCompare((PetscObject)pc,PCGASM,&match);CHKERRQ(ierr);
  if (!match) {
    if (n)   *n   = 0;
    if (mat) *mat = PETSC_NULL;
  } else {
    osm = (PC_GASM*)pc->data;
    if (n)   *n   = osm->n;
    if (mat) *mat = osm->pmat;
  }
  PetscFunctionReturn(0);
}
