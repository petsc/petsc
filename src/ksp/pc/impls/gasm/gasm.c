/*
  This file defines an "generalized" additive Schwarz preconditioner for any Mat implementation.
  In this version each processor may intersect multiple subdomains and any subdomain may 
  intersect multiple processors.  Intersections of subdomains with processors are called *local 
  subdomains*.

       N    - total number of local subdomains on all processors  (set in PCGASMSetTotalSubdomains() or calculated in PCSetUp_GASM())
       n    - actual number of local subdomains on this processor (set in PCGASMSetSubdomains() or calculated in PCGASMSetTotalSubdomains())
       nmax - maximum number of local subdomains per processor    (calculated in PCGASMSetTotalSubdomains() or in PCSetUp_GASM())
*/
#include <petsc-private/pcimpl.h>     /*I "petscpc.h" I*/

typedef struct {
  PetscInt   N,n,nmax;
  PetscInt   overlap;             /* overlap requested by user */
  KSP        *ksp;                /* linear solvers for each block */
  Vec        gx,gy;               /* Merged work vectors */
  Vec        *x,*y;               /* Split work vectors; storage aliases pieces of storage of the above merged vectors. */
  VecScatter gorestriction;       /* merged restriction to disjoint union of outer subdomains */
  VecScatter girestriction;       /* merged restriction to disjoint union of inner subdomains */
  IS         *ois;                /* index sets that define the outer (conceptually, overlapping) subdomains */
  IS         *iis;                /* index sets that define the inner (conceptually, nonoverlapping) subdomains */
  Mat        *pmat;               /* subdomain block matrices */
  PCGASMType type;                /* use reduced interpolation, restriction or both */
  PetscBool  create_local;           /* whether the autocreated subdomains are local or not. */
  PetscBool  type_set;               /* if user set this value (so won't change it for symmetric problems) */
  PetscBool  same_subdomain_solvers; /* flag indicating whether all local solvers are same */
  PetscBool  sort_indices;           /* flag to sort subdomain indices */
} PC_GASM;

#undef __FUNCT__
#define __FUNCT__ "PCGASMSubdomainView_Private"
static PetscErrorCode  PCGASMSubdomainView_Private(PC pc, PetscInt i, PetscViewer viewer)
{
  PC_GASM        *osm  = (PC_GASM*)pc->data;
  PetscInt       j,nidx;
  const PetscInt *idx;
  PetscViewer    sviewer;
  char           *cidx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if(i < 0 || i > osm->n) SETERRQ2(((PetscObject)viewer)->comm, PETSC_ERR_ARG_WRONG, "Invalid subdomain %D: must nonnegative and less than %D", i, osm->n);
  /* Inner subdomains. */
  ierr = ISGetLocalSize(osm->iis[i], &nidx);CHKERRQ(ierr);
  /* 
   No more than 15 characters per index plus a space.
   PetscViewerStringSPrintf requires a string of size at least 2, so use (nidx+1) instead of nidx, 
   in case nidx == 0. That will take care of the space for the trailing '\0' as well. 
   For nidx == 0, the whole string 16 '\0'.
   */
  ierr = PetscMalloc(sizeof(char)*(16*(nidx+1)+1), &cidx);CHKERRQ(ierr);  
  ierr = PetscViewerStringOpen(PETSC_COMM_SELF, cidx, 16*(nidx+1)+1, &sviewer); CHKERRQ(ierr);
  ierr = ISGetIndices(osm->iis[i], &idx);CHKERRQ(ierr);
  for(j = 0; j < nidx; ++j) {
    ierr = PetscViewerStringSPrintf(sviewer, "%D ", idx[j]);CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(osm->iis[i],&idx);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&sviewer); CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "Inner subdomain:\n");  CHKERRQ(ierr);
  ierr = PetscViewerFlush(viewer);                              CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_TRUE);  CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedPrintf(viewer, "%s", cidx);CHKERRQ(ierr);
  ierr = PetscViewerFlush(viewer);                              CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_FALSE); CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "\n");                  CHKERRQ(ierr);
  ierr = PetscViewerFlush(viewer);                              CHKERRQ(ierr);
  ierr = PetscFree(cidx);                                       CHKERRQ(ierr);
  /* Outer subdomains. */
  ierr = ISGetLocalSize(osm->ois[i], &nidx);CHKERRQ(ierr);
  /* 
   No more than 15 characters per index plus a space.
   PetscViewerStringSPrintf requires a string of size at least 2, so use (nidx+1) instead of nidx, 
   in case nidx == 0. That will take care of the space for the trailing '\0' as well. 
   For nidx == 0, the whole string 16 '\0'.
   */
  ierr = PetscMalloc(sizeof(char)*(16*(nidx+1)+1), &cidx);CHKERRQ(ierr);
  ierr = PetscViewerStringOpen(PETSC_COMM_SELF, cidx, 16*(nidx+1)+1, &sviewer); CHKERRQ(ierr);
  ierr = ISGetIndices(osm->ois[i], &idx);CHKERRQ(ierr);
  for(j = 0; j < nidx; ++j) {
    ierr = PetscViewerStringSPrintf(sviewer,"%D ", idx[j]); CHKERRQ(ierr);
  }
  ierr = PetscViewerDestroy(&sviewer); CHKERRQ(ierr);
  ierr = ISRestoreIndices(osm->ois[i],&idx);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "Outer subdomain:\n");CHKERRQ(ierr);
  ierr = PetscViewerFlush(viewer);                            CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedPrintf(viewer, "%s", cidx);CHKERRQ(ierr);
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_FALSE);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  ierr = PetscFree(cidx);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGASMPrintSubdomains"
static PetscErrorCode  PCGASMPrintSubdomains(PC pc)
{
  PC_GASM        *osm  = (PC_GASM*)pc->data;
  const char     *prefix;
  char           fname[PETSC_MAX_PATH_LEN+1];
  PetscInt       i, l, d, count, gcount, *permutation, *numbering;
  PetscBool      found;
  PetscViewer    viewer, sviewer = PETSC_NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc2(osm->n, PetscInt, &permutation, osm->n, PetscInt, &numbering); CHKERRQ(ierr);
  for(i = 0; i < osm->n; ++i) permutation[i] = i;
  ierr = PetscObjectsGetGlobalNumbering(((PetscObject)pc)->comm, osm->n, (PetscObject*)osm->ois, &gcount, numbering); CHKERRQ(ierr);
  ierr = PetscSortIntWithPermutation(osm->n, numbering, permutation); CHKERRQ(ierr);
  ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(prefix,"-pc_gasm_print_subdomains",fname,PETSC_MAX_PATH_LEN,&found);CHKERRQ(ierr);
  if (!found) { ierr = PetscStrcpy(fname,"stdout");CHKERRQ(ierr); };
  ierr = PetscViewerASCIIOpen(((PetscObject)pc)->comm,fname,&viewer);CHKERRQ(ierr);
  /*
   Make sure the viewer has a name. Otherwise this may cause a deadlock or other weird errors when creating a subcomm viewer: 
   the subcomm viewer will attempt to inherit the viewer's name, which, if not set, will be constructed collectively on the comm.
  */
  ierr = PetscObjectName((PetscObject)viewer);                  CHKERRQ(ierr);
  l = 0;
  for(count = 0; count < gcount; ++count) {
    /* Now let subdomains go one at a time in the global numbering order and print their subdomain/solver info. */
    if(l<osm->n){
      d = permutation[l]; /* d is the local number of the l-th smallest (in the global ordering) among the locally supported subdomains */
      if(numbering[d] == count) {
        ierr = PetscViewerGetSubcomm(viewer,((PetscObject)osm->ois[d])->comm, &sviewer);CHKERRQ(ierr);
        ierr = PCGASMSubdomainView_Private(pc,d,sviewer); CHKERRQ(ierr);
        ierr = PetscViewerRestoreSubcomm(viewer,((PetscObject)osm->ois[d])->comm, &sviewer);CHKERRQ(ierr);
        ++l;
      }
    }
    ierr = MPI_Barrier(((PetscObject)pc)->comm); CHKERRQ(ierr);
  }
  ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
  ierr = PetscFree2(permutation,numbering); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PCView_GASM"
static PetscErrorCode PCView_GASM(PC pc,PetscViewer viewer)
{
  PC_GASM        *osm = (PC_GASM*)pc->data;
  const char     *prefix;
  PetscErrorCode ierr;
  PetscMPIInt    rank, size;
  PetscInt       i,bsz;
  PetscBool      iascii,view_subdomains=PETSC_FALSE;
  PetscViewer    sviewer;
  PetscInt       count, l, gcount, *numbering, *permutation;
  char overlap[256]     = "user-defined overlap";
  char gsubdomains[256] = "unknown total number of subdomains";
  char lsubdomains[256] = "unknown number of local  subdomains";
  char msubdomains[256] = "unknown max number of local subdomains";
  PetscFunctionBegin;
  ierr = MPI_Comm_size(((PetscObject)pc)->comm, &size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(((PetscObject)pc)->comm, &rank);CHKERRQ(ierr);


  ierr = PetscMalloc2(osm->n, PetscInt, &permutation, osm->n, PetscInt, &numbering); CHKERRQ(ierr);
  for(i = 0; i < osm->n; ++i) permutation[i] = i;
  ierr = PetscObjectsGetGlobalNumbering(((PetscObject)pc)->comm, osm->n, (PetscObject*)osm->ois, &gcount, numbering); CHKERRQ(ierr);
  ierr = PetscSortIntWithPermutation(osm->n, numbering, permutation); CHKERRQ(ierr);

  if(osm->overlap >= 0) {
    ierr = PetscSNPrintf(overlap,sizeof(overlap),"requested amount of overlap = %D",osm->overlap);        CHKERRQ(ierr);
  }
  ierr = PetscSNPrintf(gsubdomains, sizeof(gsubdomains), "total number of subdomains = %D",gcount);      CHKERRQ(ierr);
  if(osm->N > 0) {
    ierr = PetscSNPrintf(lsubdomains, sizeof(gsubdomains), "number of local subdomains = %D",osm->N);     CHKERRQ(ierr);
  }
  if(osm->nmax > 0){
    ierr = PetscSNPrintf(msubdomains,sizeof(msubdomains),"max number of local subdomains = %D",osm->nmax);CHKERRQ(ierr);
  }

  ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(prefix,"-pc_gasm_view_subdomains",&view_subdomains,PETSC_NULL);CHKERRQ(ierr);

  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii); CHKERRQ(ierr);
  if (iascii) {
    /* 
     Make sure the viewer has a name. Otherwise this may cause a deadlock when creating a subcomm viewer: 
     the subcomm viewer will attempt to inherit the viewer's name, which, if not set, will be constructed 
     collectively on the comm.
     */
    ierr = PetscObjectName((PetscObject)viewer);                  CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Generalized additive Schwarz:\n"); CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Restriction/interpolation type: %s\n",PCGASMTypes[osm->type]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"%s\n",overlap);    CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"%s\n",gsubdomains);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"%s\n",lsubdomains);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"%s\n",msubdomains);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"[%d:%d] number of locally-supported subdomains = %D\n",(int)rank,(int)size,osm->n);CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_FALSE);CHKERRQ(ierr);
    /* Cannot take advantage of osm->same_subdomain_solvers without a global numbering of subdomains. */
    ierr = PetscViewerASCIIPrintf(viewer,"Subdomain solver info is as follows:\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"- - - - - - - - - - - - - - - - - -\n");CHKERRQ(ierr);
    /* Make sure that everybody waits for the banner to be printed. */
    ierr = MPI_Barrier(((PetscObject)viewer)->comm); CHKERRQ(ierr);
    /* Now let subdomains go one at a time in the global numbering order and print their subdomain/solver info. */
    l = 0; 
    for(count = 0; count < gcount; ++count) {
      PetscMPIInt srank, ssize;
      if(l<osm->n){
        PetscInt d = permutation[l]; /* d is the local number of the l-th smallest (in the global ordering) among the locally supported subdomains */
        if(numbering[d] == count) {
          ierr = MPI_Comm_size(((PetscObject)osm->ois[d])->comm, &ssize); CHKERRQ(ierr);
          ierr = MPI_Comm_rank(((PetscObject)osm->ois[d])->comm, &srank); CHKERRQ(ierr);
          ierr = PetscViewerGetSubcomm(viewer,((PetscObject)osm->ois[d])->comm, &sviewer);CHKERRQ(ierr);
          ierr = ISGetLocalSize(osm->ois[d],&bsz);CHKERRQ(ierr);
          ierr = PetscViewerASCIISynchronizedAllow(sviewer,PETSC_TRUE);CHKERRQ(ierr);
          ierr = PetscViewerASCIISynchronizedPrintf(sviewer,"[%D:%D] (subcomm [%D:%D]) local subdomain number %D, local size = %D\n",(int)rank,(int)size,(int)srank,(int)ssize,d,bsz);CHKERRQ(ierr);
          ierr = PetscViewerFlush(sviewer); CHKERRQ(ierr);
          ierr = PetscViewerASCIISynchronizedAllow(sviewer,PETSC_FALSE);CHKERRQ(ierr);
          if(view_subdomains) {
            ierr = PCGASMSubdomainView_Private(pc,d,sviewer); CHKERRQ(ierr);
          }
          if(!pc->setupcalled) {
            PetscViewerASCIIPrintf(sviewer, "Solver not set up yet: PCSetUp() not yet called\n"); CHKERRQ(ierr);
          }
          else {
            ierr = KSPView(osm->ksp[d],sviewer);CHKERRQ(ierr);
          }
          ierr = PetscViewerASCIIPrintf(sviewer,"- - - - - - - - - - - - - - - - - -\n");CHKERRQ(ierr);
          ierr = PetscViewerFlush(sviewer);CHKERRQ(ierr);
          ierr = PetscViewerRestoreSubcomm(viewer,((PetscObject)osm->ois[d])->comm, &sviewer);CHKERRQ(ierr);
          ++l;
        }
      }
      ierr = MPI_Barrier(((PetscObject)pc)->comm); CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  } else {
    SETERRQ1(((PetscObject)pc)->comm,PETSC_ERR_SUP,"Viewer type %s not supported for PCGASM",((PetscObject)viewer)->type_name);
  }
  ierr = PetscFree2(permutation,numbering); CHKERRQ(ierr);
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
  Vec            x,y;
  PetscInt       oni;       /* Number of indices in the i-th local outer subdomain.               */
  const PetscInt *oidxi;    /* Indices from the i-th subdomain local outer subdomain.             */
  PetscInt       on;        /* Number of indices in the disjoint union of local outer subdomains. */
  PetscInt       *oidx;     /* Indices in the disjoint union of local outer subdomains. */
  IS             gois;      /* Disjoint union the global indices of outer subdomains.             */
  IS             goid;      /* Identity IS of the size of the disjoint union of outer subdomains. */
  PetscScalar    *gxarray, *gyarray;
  PetscInt       gofirst;   /* Start of locally-owned indices in the vectors -- osm->gx,osm->gy -- 
                             over the disjoint union of outer subdomains. */
  DM             *domain_dm = PETSC_NULL;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(((PetscObject)pc)->comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(((PetscObject)pc)->comm,&rank);CHKERRQ(ierr);
  if (!pc->setupcalled) {

    if (!osm->type_set) {
      ierr = MatIsSymmetricKnown(pc->pmat,&symset,&flg);CHKERRQ(ierr);
      if (symset && flg) { osm->type = PC_GASM_BASIC; }
    }

    /* 
     If subdomains have been set, then the local number of subdomains, osm->n, is NOT PETSC_DECIDE and is at least 1.
     The total number of subdomains, osm->N is not necessarily set, might be PETSC_DECIDE, and then will have to be calculated from osm->n.
     */
    if (osm->n == PETSC_DECIDE) { 
      /* no subdomains given */
      /* try pc->dm first */
      if(pc->dm) {
        char      ddm_name[1024];
        DM        ddm;
        PetscBool flg;
        PetscInt     num_domains, d;
        char         **domain_names;
        IS           *inner_domain_is, *outer_domain_is;
        /* Allow the user to request a decomposition DM by name */
        ierr = PetscStrncpy(ddm_name, "", 1024); CHKERRQ(ierr);
        ierr = PetscOptionsString("-pc_gasm_decomposition","Name of the DM defining the composition", "PCSetDM", ddm_name, ddm_name,1024,&flg); CHKERRQ(ierr);
        if(flg) {
          ierr = DMCreateDomainDecompositionDM(pc->dm, ddm_name, &ddm); CHKERRQ(ierr);
          if(!ddm) {
            SETERRQ1(((PetscObject)pc)->comm, PETSC_ERR_ARG_WRONGSTATE, "Uknown DM decomposition name %s", ddm_name);
          }
          ierr = PetscInfo(pc,"Using decomposition DM defined using options database\n");CHKERRQ(ierr);
          ierr = PCSetDM(pc,ddm); CHKERRQ(ierr);
        }
        ierr = DMCreateDomainDecomposition(pc->dm, &num_domains, &domain_names, &inner_domain_is, &outer_domain_is, &domain_dm);    CHKERRQ(ierr);
        if(num_domains) {
          ierr = PCGASMSetSubdomains(pc, num_domains, inner_domain_is, outer_domain_is);CHKERRQ(ierr);
        }
        for(d = 0; d < num_domains; ++d) {
          ierr = PetscFree(domain_names[d]); CHKERRQ(ierr);
          ierr = ISDestroy(&inner_domain_is[d]);   CHKERRQ(ierr);
          ierr = ISDestroy(&outer_domain_is[d]);   CHKERRQ(ierr);
        }
        ierr = PetscFree(domain_names);CHKERRQ(ierr);
        ierr = PetscFree(inner_domain_is);CHKERRQ(ierr);
        ierr = PetscFree(outer_domain_is);CHKERRQ(ierr);
      }
      if (osm->n == PETSC_DECIDE) { /* still no subdomains; use one per processor */
        osm->nmax = osm->n = 1;
        ierr = MPI_Comm_size(((PetscObject)pc)->comm,&size);CHKERRQ(ierr);
        osm->N = size;
      }
    } 
    if (osm->N == PETSC_DECIDE) {
      PetscInt inwork[2], outwork[2];
      /* determine global number of subdomains and the max number of local subdomains */
      inwork[0] = inwork[1] = osm->n;
      ierr = MPI_Allreduce(inwork,outwork,1,MPIU_2INT,PetscMaxSum_Op,((PetscObject)pc)->comm);CHKERRQ(ierr);
      osm->nmax = outwork[0];
      osm->N    = outwork[1];
    }
    if (!osm->iis){ 
      /* 
       The local number of subdomains was set in PCGASMSetTotalSubdomains() or PCGASMSetSubdomains(), 
       but the actual subdomains have not been supplied (in PCGASMSetSubdomains()).
       We create the requisite number of inner subdomains on PETSC_COMM_SELF (for now).
       */
      ierr = PCGASMCreateLocalSubdomains(pc->pmat,osm->overlap,osm->n,&osm->iis,&osm->ois);CHKERRQ(ierr);
    }

    ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);
    flg  = PETSC_FALSE;
    ierr = PetscOptionsGetBool(prefix,"-pc_gasm_print_subdomains",&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) { ierr = PCGASMPrintSubdomains(pc);CHKERRQ(ierr); }

    if (osm->sort_indices) {
      for (i=0; i<osm->n; i++) {
        ierr = ISSort(osm->ois[i]);CHKERRQ(ierr);
	ierr = ISSort(osm->iis[i]);CHKERRQ(ierr);
      }
    }
    /* 
     Merge the ISs, create merged vectors and restrictions. 
     */
    /* Merge outer subdomain ISs and construct a restriction onto the disjoint union of local outer subdomains. */
    on = 0;
    for (i=0; i<osm->n; i++) {
      ierr = ISGetLocalSize(osm->ois[i],&oni);CHKERRQ(ierr);
      on += oni;
    }
    ierr = PetscMalloc(on*sizeof(PetscInt), &oidx);CHKERRQ(ierr);
    on = 0;
    for (i=0; i<osm->n; i++) {
      ierr = ISGetLocalSize(osm->ois[i],&oni);CHKERRQ(ierr);
      ierr = ISGetIndices(osm->ois[i],&oidxi);CHKERRQ(ierr);
      ierr = PetscMemcpy(oidx+on, oidxi, sizeof(PetscInt)*oni);CHKERRQ(ierr);
      ierr = ISRestoreIndices(osm->ois[i], &oidxi);CHKERRQ(ierr);
      on += oni;
    }
    ierr = ISCreateGeneral(((PetscObject)(pc))->comm, on, oidx, PETSC_OWN_POINTER, &gois);CHKERRQ(ierr);
    ierr = MatGetVecs(pc->pmat,&x,&y);CHKERRQ(ierr);
    ierr = VecCreateMPI(((PetscObject)pc)->comm, on, PETSC_DECIDE, &osm->gx);CHKERRQ(ierr);
    ierr = VecDuplicate(osm->gx,&osm->gy);CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(osm->gx, &gofirst, PETSC_NULL);CHKERRQ(ierr);
    ierr = ISCreateStride(((PetscObject)pc)->comm,on,gofirst,1, &goid);CHKERRQ(ierr);
    ierr = VecScatterCreate(x,gois,osm->gx,goid, &(osm->gorestriction));CHKERRQ(ierr);
    ierr = VecDestroy(&x); CHKERRQ(ierr);
    ierr = ISDestroy(&gois); CHKERRQ(ierr);
    /* Merge inner subdomain ISs and construct a restriction onto the disjoint union of local inner subdomains. */
    { PetscInt       ini;     /* Number of indices the i-th a local inner subdomain. */
      PetscInt       in;      /* Number of indices in the disjoint uniont of local inner subdomains. */
      PetscInt       *iidx;   /* Global indices in the merged local inner subdomain. */
      PetscInt       *ioidx;  /* Global indices of the disjoint union of inner subdomains within the disjoint union of outer subdomains. */
      IS             giis;    /* IS for the disjoint union of inner subdomains. */
      IS             giois;   /* IS for the disjoint union of inner subdomains within the disjoint union of outer subdomains. */
      /**/
      in = 0;
      for (i=0; i<osm->n; i++) {
	ierr = ISGetLocalSize(osm->iis[i],&ini);CHKERRQ(ierr);
	in += ini;
      }
      ierr = PetscMalloc(in*sizeof(PetscInt), &iidx); CHKERRQ(ierr);
      ierr = PetscMalloc(in*sizeof(PetscInt), &ioidx);CHKERRQ(ierr);
      ierr = VecGetOwnershipRange(osm->gx,&gofirst, PETSC_NULL); CHKERRQ(ierr);
      in = 0;
      on = 0;
      for (i=0; i<osm->n; i++) {
        const PetscInt *iidxi;        /* Global indices of the i-th local inner subdomain. */
        ISLocalToGlobalMapping ltogi; /* Map from global to local indices of the i-th outer local subdomain. */
        PetscInt       *ioidxi;       /* Local indices of the i-th local inner subdomain within the local outer subdomain. */
        PetscInt       ioni;          /* Number of indices in ioidxi; if ioni != ini the inner subdomain is not a subdomain of the outer subdomain (error). */
        PetscInt       k;
	ierr = ISGetLocalSize(osm->iis[i],&ini);CHKERRQ(ierr);
	ierr = ISGetLocalSize(osm->ois[i],&oni);CHKERRQ(ierr);
	ierr = ISGetIndices(osm->iis[i],&iidxi);CHKERRQ(ierr);
	ierr = PetscMemcpy(iidx+in, iidxi, sizeof(PetscInt)*ini);CHKERRQ(ierr);
        ierr = ISLocalToGlobalMappingCreateIS(osm->ois[i],&ltogi);CHKERRQ(ierr);
        ioidxi = ioidx+in;
        ierr = ISGlobalToLocalMappingApply(ltogi,IS_GTOLM_DROP,ini,iidxi,&ioni,ioidxi);CHKERRQ(ierr);
        ierr = ISLocalToGlobalMappingDestroy(&ltogi);CHKERRQ(ierr);
	ierr = ISRestoreIndices(osm->iis[i], &iidxi);CHKERRQ(ierr);
        if (ioni != ini) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Inner subdomain %D contains %D indices outside of its outer subdomain", i, ini - ioni);
        for(k = 0; k < ini; ++k) {
          ioidxi[k] += gofirst+on;
        }
	in += ini;
        on += oni;
      }
      ierr = ISCreateGeneral(((PetscObject)pc)->comm, in, iidx,  PETSC_OWN_POINTER, &giis);CHKERRQ(ierr);
      ierr = ISCreateGeneral(((PetscObject)pc)->comm, in, ioidx, PETSC_OWN_POINTER, &giois);CHKERRQ(ierr);
      ierr = VecScatterCreate(y,giis,osm->gy,giois,&osm->girestriction);CHKERRQ(ierr);
      ierr = VecDestroy(&y); CHKERRQ(ierr);
      ierr = ISDestroy(&giis);  CHKERRQ(ierr);
      ierr = ISDestroy(&giois); CHKERRQ(ierr);
    }
    ierr = ISDestroy(&goid);CHKERRQ(ierr);
    /* Create the subdomain work vectors. */
    ierr = PetscMalloc(osm->n*sizeof(Vec),&osm->x);CHKERRQ(ierr);
    ierr = PetscMalloc(osm->n*sizeof(Vec),&osm->y);CHKERRQ(ierr);
    ierr = VecGetArray(osm->gx, &gxarray);CHKERRQ(ierr);
    ierr = VecGetArray(osm->gy, &gyarray);CHKERRQ(ierr);
    for (i=0, on=0; i<osm->n; ++i, on += oni) {
      PetscInt oNi;
      ierr = ISGetLocalSize(osm->ois[i],&oni);CHKERRQ(ierr);
      ierr = ISGetSize(osm->ois[i],&oNi);CHKERRQ(ierr);
      ierr = VecCreateMPIWithArray(((PetscObject)(osm->ois[i]))->comm,1,oni,oNi,gxarray+on,&osm->x[i]);CHKERRQ(ierr); 
      ierr = VecCreateMPIWithArray(((PetscObject)(osm->ois[i]))->comm,1,oni,oNi,gyarray+on,&osm->y[i]);CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(osm->gx, &gxarray);CHKERRQ(ierr);
    ierr = VecRestoreArray(osm->gy, &gyarray);CHKERRQ(ierr);
    /* Create the local solvers */
    ierr = PetscMalloc(osm->n*sizeof(KSP *),&osm->ksp);CHKERRQ(ierr);
    for (i=0; i<osm->n; i++) { /* KSPs are local */
      ierr = KSPCreate(((PetscObject)(osm->ois[i]))->comm,&ksp);CHKERRQ(ierr);
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

  }/*if(!pc->setupcalled)*/ 
  else {
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
    ierr = MatGetSubMatricesParallel(pc->pmat,osm->n,osm->ois, osm->ois,scall,&osm->pmat);CHKERRQ(ierr);
  }
  else {
    ierr = MatGetSubMatrices(pc->pmat,osm->n,osm->ois, osm->ois,scall,&osm->pmat);CHKERRQ(ierr);
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
  ierr = PCModifySubMatrices(pc,osm->n,osm->ois,osm->ois,osm->pmat,pc->modifysubmatricesP);CHKERRQ(ierr);

  /* 
     Loop over submatrices putting them into local ksps
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
     Support for limiting the restriction or interpolation only to the inner
     subdomain values (leaving the other values 0). 
  */
  if(!(osm->type & PC_GASM_RESTRICT)) {
    /* have to zero the work RHS since scatter may leave some slots empty */
    ierr = VecZeroEntries(osm->gx);CHKERRQ(ierr);
    ierr = VecScatterBegin(osm->girestriction,x,osm->gx,INSERT_VALUES,forward);CHKERRQ(ierr);
  }
  else {
    ierr = VecScatterBegin(osm->gorestriction,x,osm->gx,INSERT_VALUES,forward);CHKERRQ(ierr);
  }
  ierr = VecZeroEntries(osm->gy);CHKERRQ(ierr);
  if(!(osm->type & PC_GASM_RESTRICT)) {
    ierr = VecScatterEnd(osm->girestriction,x,osm->gx,INSERT_VALUES,forward);CHKERRQ(ierr);
  }
  else {
    ierr = VecScatterEnd(osm->gorestriction,x,osm->gx,INSERT_VALUES,forward);CHKERRQ(ierr);
  }
  /* do the subdomain solves */
  for (i=0; i<osm->n; ++i) { 
    ierr = KSPSolve(osm->ksp[i],osm->x[i],osm->y[i]);CHKERRQ(ierr);
  }
  ierr = VecZeroEntries(y); CHKERRQ(ierr);
  if(!(osm->type & PC_GASM_INTERPOLATE)) {
    ierr = VecScatterBegin(osm->girestriction,osm->gy,y,ADD_VALUES,reverse);CHKERRQ(ierr);
    ierr = VecScatterEnd(osm->girestriction,osm->gy,y,ADD_VALUES,reverse);CHKERRQ(ierr);  PetscFunctionReturn(0);
  }
  else {
    ierr = VecScatterBegin(osm->gorestriction,osm->gy,y,ADD_VALUES,reverse);CHKERRQ(ierr);
    ierr = VecScatterEnd(osm->gorestriction,osm->gy,y,ADD_VALUES,reverse);CHKERRQ(ierr);  PetscFunctionReturn(0);
  }
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
    /* have to zero the work RHS since scatter may leave some slots empty */
    ierr = VecZeroEntries(osm->gx);CHKERRQ(ierr);
    ierr = VecScatterBegin(osm->girestriction,x,osm->gx,INSERT_VALUES,forward);CHKERRQ(ierr);
  }
  else {
    ierr = VecScatterBegin(osm->gorestriction,x,osm->gx,INSERT_VALUES,forward);CHKERRQ(ierr);
  }
  ierr = VecZeroEntries(osm->gy);CHKERRQ(ierr);
  if (!(osm->type & PC_GASM_INTERPOLATE)) {
    ierr = VecScatterEnd(osm->girestriction,x,osm->gx,INSERT_VALUES,forward);CHKERRQ(ierr);
  }
  else {
    ierr = VecScatterEnd(osm->gorestriction,x,osm->gx,INSERT_VALUES,forward);CHKERRQ(ierr);
  }
  /* do the local solves */
  for (i=0; i<osm->n; ++i) { /* Note that the solves are local, so we can go to osm->n, rather than osm->nmax. */
    ierr = KSPSolveTranspose(osm->ksp[i],osm->x[i],osm->y[i]);CHKERRQ(ierr); 
  }
  ierr = VecZeroEntries(y); CHKERRQ(ierr);
  if (!(osm->type & PC_GASM_RESTRICT)) {
    ierr = VecScatterBegin(osm->girestriction,osm->gy,y,ADD_VALUES,reverse);CHKERRQ(ierr);
    ierr = VecScatterEnd(osm->girestriction,osm->gy,y,ADD_VALUES,reverse);CHKERRQ(ierr);
  }
  else {
    ierr = VecScatterBegin(osm->gorestriction,osm->gy,y,ADD_VALUES,reverse);CHKERRQ(ierr);
    ierr = VecScatterEnd(osm->gorestriction,osm->gy,y,ADD_VALUES,reverse);CHKERRQ(ierr);
  }

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
  
  ierr = VecScatterDestroy(&osm->gorestriction);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&osm->girestriction);CHKERRQ(ierr);
  if (osm->ois) {
    ierr = PCGASMDestroySubdomains(osm->n,osm->ois,osm->iis);CHKERRQ(ierr); 
    osm->ois = 0;
    osm->iis = 0;
  }
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
    ierr = PetscOptionsInt("-pc_gasm_total_subdomains","Total number of subdomains across communicator","PCGASMSetTotalSubdomains",osm->n,&blocks,&flg);CHKERRQ(ierr);
    osm->create_local = PETSC_TRUE;
    ierr = PetscOptionsBool("-pc_gasm_subdomains_create_local","Whether to make autocreated subdomains local (true by default)","PCGASMSetTotalSubdomains",osm->create_local,&osm->create_local,&flg);CHKERRQ(ierr);
    if(!osm->create_local) SETERRQ(((PetscObject)pc)->comm, PETSC_ERR_SUP, "No support for autocreation of nonlocal subdomains yet.");
 
    if (flg) {ierr = PCGASMSetTotalSubdomains(pc,blocks,osm->create_local);CHKERRQ(ierr); }
    ierr = PetscOptionsInt("-pc_gasm_overlap","Number of overlapping degrees of freedom","PCGASMSetOverlap",osm->overlap,&ovl,&flg);CHKERRQ(ierr);
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
#define __FUNCT__ "PCGASMSetSubdomains_GASM"
PetscErrorCode  PCGASMSetSubdomains_GASM(PC pc,PetscInt n,IS is_local[],IS is[])
{
  PC_GASM         *osm = (PC_GASM*)pc->data;
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  if (n < 1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Each process must have 1 or more subdomains, n = %D",n);
  if (pc->setupcalled && (n != osm->n || is_local)) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_WRONGSTATE,"PCGASMSetSubdomains() should be called before calling PCSetUp().");

  if (!pc->setupcalled) {
    osm->n            = n;
    osm->ois           = 0;
    osm->iis     = 0;
    if (is) {
      for (i=0; i<n; i++) {ierr = PetscObjectReference((PetscObject)is[i]);CHKERRQ(ierr);}
    }
    if (is_local) {
      for (i=0; i<n; i++) {ierr = PetscObjectReference((PetscObject)is_local[i]);CHKERRQ(ierr);}
    }
    if (osm->ois) {
      ierr = PCGASMDestroySubdomains(osm->n,osm->iis,osm->ois);CHKERRQ(ierr);
    }
    if (is) {
      ierr = PetscMalloc(n*sizeof(IS),&osm->ois);CHKERRQ(ierr);
      for (i=0; i<n; i++) { osm->ois[i] = is[i]; }
      /* Flag indicating that the user has set outer subdomains, so PCGASM should not increase their size. */
      osm->overlap = -1;
    }
    if (is_local) {
      ierr = PetscMalloc(n*sizeof(IS),&osm->iis);CHKERRQ(ierr);
      for (i=0; i<n; i++) { osm->iis[i] = is_local[i]; }
    }
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCGASMSetTotalSubdomains_GASM"
PetscErrorCode  PCGASMSetTotalSubdomains_GASM(PC pc,PetscInt N, PetscBool create_local) {
  PC_GASM         *osm = (PC_GASM*)pc->data;
  PetscErrorCode ierr;
  PetscMPIInt    rank,size;
  PetscInt       n;
  PetscInt       Nmin, Nmax;
  PetscFunctionBegin;
  if(!create_local) SETERRQ(((PetscObject)pc)->comm, PETSC_ERR_SUP, "No suppor for autocreation of nonlocal subdomains.");
  if (N < 1) SETERRQ1(((PetscObject)pc)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Total number of subdomains must be > 0, N = %D",N);
  ierr = MPI_Allreduce(&N,&Nmin,1,MPI_INT,MPI_MIN,((PetscObject)pc)->comm); CHKERRQ(ierr);
  ierr = MPI_Allreduce(&N,&Nmax,1,MPI_INT,MPI_MAX,((PetscObject)pc)->comm); CHKERRQ(ierr);
  if(Nmin != Nmax) 
    SETERRQ2(((PetscObject)pc)->comm, PETSC_ERR_ARG_WRONG, "All processors must use the same number of subdomains.  min(N) = %D != %D = max(N)", Nmin, Nmax);

  osm->create_local = create_local;
  /*
     Split the subdomains equally among all processors
  */
  ierr = MPI_Comm_rank(((PetscObject)pc)->comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(((PetscObject)pc)->comm,&size);CHKERRQ(ierr);
  n = N/size + ((N % size) > rank);
  if (!n) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Process %d must have at least one subdomain: total processors %d total blocks %D",(int)rank,(int)size,N);
  if (pc->setupcalled && n != osm->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"PCGASMSetTotalSubdomains() should be called before PCSetUp().");
  if (!pc->setupcalled) {
    if (osm->ois) {
      ierr = PCGASMDestroySubdomains(osm->n,osm->iis,osm->ois);CHKERRQ(ierr);
    }
    osm->N            = N;
    osm->n            = n;
    osm->nmax         = N/size + ((N%size)?1:0);
    osm->ois           = 0;
    osm->iis     = 0;
  }
  PetscFunctionReturn(0);
}
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
       true, though!  This flag is used only for PCView_GASM() */
    *ksp                   = osm->ksp;
    osm->same_subdomain_solvers = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}/* PCGASMGetSubKSP_GASM() */
EXTERN_C_END


#undef __FUNCT__  
#define __FUNCT__ "PCGASMSetSubdomains"
/*@C
    PCGASMSetSubdomains - Sets the subdomains for this processor
    for the additive Schwarz preconditioner. 

    Collective on PC 

    Input Parameters:
+   pc  - the preconditioner context
.   n   - the number of subdomains for this processor
.   iis - the index sets that define this processor's local inner subdomains
         (or PETSC_NULL for PETSc to determine subdomains)
-   ois- the index sets that define this processor's local outer subdomains 
         (or PETSC_NULL to use the same as iis)

    Notes:
    The IS indices use the parallel, global numbering of the vector entries.
    Inner subdomains are those where the correction is applied.
    Outer subdomains are those where the residual necessary to obtain the 
    corrections is obtained (see PCGASMType for the use of inner/outer subdomains).
    Both inner and outer subdomains can extend over several processors. 
    This processor's portion of a subdomain is known as a local subdomain.

    By default the GASM preconditioner uses 1 (local) subdomain per processor.
    Use PCGASMSetTotalSubdomains() to set the total number of subdomains across 
    all processors that PCGASM will create automatically, and to specify whether 
    they should be local or not.


    Level: advanced

.keywords: PC, GASM, set, subdomains, additive Schwarz

.seealso: PCGASMSetTotalSubdomains(), PCGASMSetOverlap(), PCGASMGetSubKSP(),
          PCGASMCreateSubdomains2D(), PCGASMGetSubdomains()
@*/
PetscErrorCode  PCGASMSetSubdomains(PC pc,PetscInt n,IS iis[],IS ois[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCGASMSetSubdomains_C",(PC,PetscInt,IS[],IS[]),(pc,n,iis,ois));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCGASMSetTotalSubdomains"
/*@C
    PCGASMSetTotalSubdomains - Sets the total number of subdomains to use in the generalized additive 
    Schwarz preconditioner.  The number of subdomains is cumulative across all processors in pc's
    communicator. Either all or no processors in the PC communicator must call this routine with 
    the same N.  The subdomains will be created automatically during PCSetUp().

    Collective on PC

    Input Parameters:
+   pc           - the preconditioner context
.   N            - the total number of subdomains cumulative across all processors
-   create_local - whether the subdomains to be created are to be local

    Options Database Key:
    To set the total number of subdomains and let PCGASM autocreate them, rather than specify the index sets, use the following options:
+    -pc_gasm_total_subdomains <n>                  - sets the total number of subdomains to be autocreated by PCGASM
-    -pc_gasm_subdomains_create_local <true|false>  - whether autocreated subdomains should be local or not (default is true)

    By default the GASM preconditioner uses 1 subdomain per processor.  


    Use PCGASMSetSubdomains() to set subdomains explicitly or to set different numbers
    of subdomains per processor.

    Level: advanced

.keywords: PC, GASM, set, total, global, subdomains, additive Schwarz

.seealso: PCGASMSetSubdomains(), PCGASMSetOverlap(), PCGASMGetSubKSP(),
          PCGASMCreateSubdomains2D()
@*/
PetscErrorCode  PCGASMSetTotalSubdomains(PC pc,PetscInt N, PetscBool create_local)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCGASMSetTotalSubdomains_C",(PC,PetscInt,PetscBool),(pc,N,create_local));CHKERRQ(ierr);
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
.   -pc_gasm_overlap <overlap> - Sets overlap

    Notes:
    By default the GASM preconditioner uses 1 subdomain per processor.  To use
    multiple subdomain per perocessor, see PCGASMSetTotalSubdomains() or
    PCGASMSetSubdomains() (and the option -pc_gasm_total_subdomains <n>).

    The overlap defaults to 1, so if one desires that no additional
    overlap be computed beyond what may have been set with a call to
    PCGASMSetTotalSubdomains() or PCGASMSetSubdomains(), then ovl
    must be set to be 0.  In particular, if one does not explicitly set
    the subdomains in application code, then all overlap would be computed
    internally by PETSc, and using an overlap of 0 would result in an GASM 
    variant that is equivalent to the block Jacobi preconditioner.  

    Note that one can define initial index sets with any overlap via
    PCGASMSetSubdomains(); the routine PCGASMSetOverlap() merely allows 
    PETSc to extend that overlap further, if desired.

    Level: intermediate

.keywords: PC, GASM, set, overlap

.seealso: PCGASMSetTotalSubdomains(), PCGASMSetSubdomains(), PCGASMGetSubKSP(),
          PCGASMCreateSubdomains2D(), PCGASMGetSubdomains()
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

.seealso: PCGASMSetTotalSubdomains(), PCGASMSetSubdomains(), PCGASMGetSubKSP(),
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

.seealso: PCGASMSetSubdomains(), PCGASMSetTotalSubdomains(), PCGASMGetSubKSP(),
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

.seealso: PCGASMSetTotalSubdomains(), PCGASMSetSubdomains(), PCGASMSetOverlap(),
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
+  -pc_gasm_total_block_count <n> - Sets total number of local subdomains (known as blocks) to be distributed among processors
.  -pc_gasm_view_subdomains       - activates the printing of subdomain indices in PCView(), -ksp_view or -snes_view
.  -pc_gasm_print_subdomains      - activates the printing of subdomain indices in PCSetUp()
.  -pc_gasm_overlap <ovl>         - Sets overlap by which to (automatically) extend local subdomains
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
           PCBJACOBI, PCGASMSetUseTrueLocal(), PCGASMGetSubKSP(), PCGASMSetSubdomains(),
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
  osm->n                 = PETSC_DECIDE;
  osm->nmax              = 0;
  osm->overlap           = 1;
  osm->ksp               = 0;
  osm->gorestriction     = 0;
  osm->girestriction     = 0;
  osm->gx                = 0;
  osm->gy                = 0;
  osm->x                 = 0;
  osm->y                 = 0;
  osm->ois               = 0;
  osm->iis               = 0;
  osm->pmat              = 0;
  osm->type              = PC_GASM_RESTRICT;
  osm->same_subdomain_solvers = PETSC_TRUE;
  osm->sort_indices           = PETSC_TRUE;

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

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCGASMSetSubdomains_C","PCGASMSetSubdomains_GASM",
                    PCGASMSetSubdomains_GASM);CHKERRQ(ierr);
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
#define __FUNCT__ "PCGASMCreateLocalSubdomains"
/*@C
   PCGASMCreateLocalSubdomains - Creates n local index sets for the overlapping 
   Schwarz preconditioner for a any problem based on its matrix.

   Collective

   Input Parameters:
+  A       - The global matrix operator
.  overlap - amount of overlap in outer subdomains
-  n       - the number of local subdomains

   Output Parameters:
+  iis - the array of index sets defining the local inner subdomains (on which the correction is applied)
-  ois - the array of index sets defining the local outer subdomains (on which the residual is computed)

   Level: advanced

   Note: this generates n nonoverlapping local inner subdomains on PETSC_COMM_SELF; 
         PCGASM will generate the overlap from these if you use them in PCGASMSetSubdomains() and set a 
         nonzero overlap with PCGASMSetOverlap()

    In the Fortran version you must provide the array outis[] already allocated of length n.

.keywords: PC, GASM, additive Schwarz, create, subdomains, unstructured grid

.seealso: PCGASMSetSubdomains(), PCGASMDestroySubdomains()
@*/
PetscErrorCode  PCGASMCreateLocalSubdomains(Mat A, PetscInt overlap, PetscInt n, IS* iis[], IS* ois[])
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
  PetscValidPointer(iis,4);
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
    ierr = PetscObjectTypeCompare((PetscObject)Ad,MATSEQBAIJ,&isbaij);CHKERRQ(ierr);
    if (!isbaij) {ierr = PetscObjectTypeCompare((PetscObject)Ad,MATSEQSBAIJ,&isbaij);CHKERRQ(ierr);}
  }
  if (Ad && n > 1) {
    PetscBool  match,done;
    /* Try to setup a good matrix partitioning if available */
    ierr = MatPartitioningCreate(PETSC_COMM_SELF,&mpart);CHKERRQ(ierr);
    ierr = PetscObjectSetOptionsPrefix((PetscObject)mpart,prefix);CHKERRQ(ierr);
    ierr = MatPartitioningSetFromOptions(mpart);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)mpart,MATPARTITIONINGCURRENT,&match);CHKERRQ(ierr);
    if (!match) {
      ierr = PetscObjectTypeCompare((PetscObject)mpart,MATPARTITIONINGSQUARE,&match);CHKERRQ(ierr);
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
  *iis = is;
  if(!ois) PetscFunctionReturn(0);
  /*
   Initially make outer subdomains the same as inner subdomains. If nonzero additional overlap
   has been requested, copy the inner subdomains over so they can be modified.
   */
  ierr = PetscMalloc(n*sizeof(IS),ois);CHKERRQ(ierr);
  for (i=0; i<n; ++i) {
    if (overlap > 0) { /* With positive overlap, (*iis)[i] will be modified */
      ierr = ISDuplicate((*iis)[i],(*ois)+i);CHKERRQ(ierr);
      ierr = ISCopy((*iis)[i],(*ois)[i]);CHKERRQ(ierr);
    } else {
      ierr = PetscObjectReference((PetscObject)(*iis)[i]);CHKERRQ(ierr);
      (*ois)[i] = (*iis)[i];
    }
  }
  if (overlap > 0) {
    /* Extend the "overlapping" regions by a number of steps */
    ierr = MatIncreaseOverlap(A,n,*ois,overlap);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCGASMDestroySubdomains"
/*@C
   PCGASMDestroySubdomains - Destroys the index sets created with
   PCGASMCreateLocalSubdomains() or PCGASMCreateSubdomains2D. Should be 
   called after setting subdomains with PCGASMSetSubdomains().

   Collective

   Input Parameters:
+  n   - the number of index sets
.  iis - the array of inner subdomains,
-  ois - the array of outer subdomains, can be PETSC_NULL

   Level: intermediate

   Notes: this is merely a convenience subroutine that walks each list,
   destroys each IS on the list, and then frees the list.

.keywords: PC, GASM, additive Schwarz, create, subdomains, unstructured grid

.seealso: PCGASMCreateLocalSubdomains(), PCGASMSetSubdomains()
@*/
PetscErrorCode  PCGASMDestroySubdomains(PetscInt n, IS iis[], IS ois[])
{
  PetscInt       i;
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (n <= 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"n must be > 0: n = %D",n);
  PetscValidPointer(iis,2);
  for (i=0; i<n; i++) { ierr = ISDestroy(&iis[i]);CHKERRQ(ierr); }
  ierr = PetscFree(iis);CHKERRQ(ierr);
  if (ois) {
    for (i=0; i<n; i++) { ierr = ISDestroy(&ois[i]);CHKERRQ(ierr); }
    ierr = PetscFree(ois);CHKERRQ(ierr);
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
+  M, N               - the global number of grid points in the x and y directions
.  Mdomains, Ndomains - the global number of subdomains in the x and y directions
.  dof                - degrees of freedom per node
-  overlap            - overlap in mesh lines

   Output Parameters:
+  Nsub - the number of local subdomains created
.  iis  - array of index sets defining inner (nonoverlapping) subdomains
-  ois  - array of index sets defining outer (overlapping, if overlap > 0) subdomains


   Level: advanced

.keywords: PC, GASM, additive Schwarz, create, subdomains, 2D, regular grid

.seealso: PCGASMSetTotalSubdomains(), PCGASMSetSubdomains(), PCGASMGetSubKSP(),
          PCGASMSetOverlap()
@*/
PetscErrorCode  PCGASMCreateSubdomains2D(PC pc, PetscInt M,PetscInt N,PetscInt Mdomains,PetscInt Ndomains,PetscInt dof,PetscInt overlap, PetscInt *nsub,IS **iis,IS **ois)
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
  IS             **xis = 0, **is = ois, **is_local = iis;

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
            xis = is;
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
              xis = is_local;
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
	  ierr = ISCreateGeneral(subcomm,nidx,idx,PETSC_OWN_POINTER,(*xis)+s);CHKERRQ(ierr);
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
#define __FUNCT__ "PCGASMGetSubdomains"
/*@C
    PCGASMGetSubdomains - Gets the subdomains supported on this processor
    for the additive Schwarz preconditioner. 

    Not Collective

    Input Parameter:
.   pc - the preconditioner context

    Output Parameters:
+   n   - the number of subdomains for this processor (default value = 1)
.   iis - the index sets that define the inner subdomains (without overlap) supported on this processor (can be PETSC_NULL)
-   ois - the index sets that define the outer subdomains (with overlap) supported on this processor (can be PETSC_NULL)
         

    Notes:
    The user is responsible for destroying the ISs and freeing the returned arrays.
    The IS numbering is in the parallel, global numbering of the vector.

    Level: advanced

.keywords: PC, GASM, get, subdomains, additive Schwarz

.seealso: PCGASMSetTotalSubdomains(), PCGASMSetOverlap(), PCGASMGetSubKSP(),
          PCGASMCreateSubdomains2D(), PCGASMSetSubdomains(), PCGASMGetSubmatrices()
@*/
PetscErrorCode  PCGASMGetSubdomains(PC pc,PetscInt *n,IS *iis[],IS *ois[])
{
  PC_GASM         *osm;
  PetscErrorCode ierr;
  PetscBool      match;
  PetscInt       i;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscObjectTypeCompare((PetscObject)pc,PCGASM,&match);CHKERRQ(ierr);
  if (!match) 
    SETERRQ2(((PetscObject)pc)->comm, PETSC_ERR_ARG_WRONG, "Incorrect object type: expected %s, got %s instead", PCGASM, ((PetscObject)pc)->type_name);
  osm = (PC_GASM*)pc->data;
  if (n)  *n  = osm->n;
  if(iis) {
    ierr = PetscMalloc(osm->n*sizeof(IS), iis); CHKERRQ(ierr);
  }
  if(ois) {
    ierr = PetscMalloc(osm->n*sizeof(IS), ois); CHKERRQ(ierr);
  }
  if(iis || ois) {
    for(i = 0; i < osm->n; ++i) {
      if(iis) (*iis)[i] = osm->iis[i];
      if(ois) (*ois)[i] = osm->ois[i];
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCGASMGetSubmatrices"
/*@C
    PCGASMGetSubmatrices - Gets the local submatrices (for this processor
    only) for the additive Schwarz preconditioner. 

    Not Collective

    Input Parameter:
.   pc - the preconditioner context

    Output Parameters:
+   n   - the number of matrices for this processor (default value = 1)
-   mat - the matrices
         
    Notes: matrices returned by this routine have the same communicators as the index sets (IS) 
           used to define subdomains in PCGASMSetSubdomains(), or PETSC_COMM_SELF, if the 
           subdomains were defined using PCGASMSetTotalSubdomains().
    Level: advanced

.keywords: PC, GASM, set, local, subdomains, additive Schwarz, block Jacobi

.seealso: PCGASMSetTotalSubdomain(), PCGASMSetOverlap(), PCGASMGetSubKSP(),
          PCGASMCreateSubdomains2D(), PCGASMSetSubdomains(), PCGASMGetSubdomains()
@*/
PetscErrorCode  PCGASMGetSubmatrices(PC pc,PetscInt *n,Mat *mat[])
{
  PC_GASM         *osm;
  PetscErrorCode ierr;
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidIntPointer(n,2);
  if (mat) PetscValidPointer(mat,3);
  if (!pc->setupcalled) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_WRONGSTATE,"Must call after KSPSetUP() or PCSetUp().");
  ierr = PetscObjectTypeCompare((PetscObject)pc,PCGASM,&match);CHKERRQ(ierr);
  if (!match) SETERRQ2(((PetscObject)pc)->comm, PETSC_ERR_ARG_WRONG, "Expected %s, got %s instead", PCGASM, ((PetscObject)pc)->type_name);
  osm = (PC_GASM*)pc->data;
  if (n)   *n   = osm->n;
  if (mat) *mat = osm->pmat;

  PetscFunctionReturn(0);
}
