



#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: aodatabasic.c,v 1.12 1997/10/19 03:31:10 bsmith Exp bsmith $";
#endif

/*
    The most basic AOData routines. These store the 
  entire database on each processor. These are very simple, not that
  we do not even use a private data structure for AOData and the 
  private datastructure for AODataSegment is just used as a simple array.

*/

#include "src/ao/aoimpl.h"
#include "pinclude/pviewer.h"
#include "sys.h"

#undef __FUNC__  
#define __FUNC__ "AODataDestroy_Basic"
int AODataDestroy_Basic(PetscObject obj)
{
  int    ierr,i,j;
  AOData ao = (AOData) obj;

  PetscFunctionBegin;
  for (i=0; i<ao->nkeys; i++ ) {
    PetscFree(ao->keys[i].name);
    if (ao->keys[i].ltog) {ierr = ISLocalToGlobalMappingDestroy(ao->keys[i].ltog);CHKERRQ(ierr);}
    for (j=0; j<ao->keys[i].nsegments; j++ ) {
      PetscFree(ao->keys[i].segments[j].data);
      PetscFree(ao->keys[i].segments[j].name);
    }
    PetscFree(ao->keys[i].segments);
    PetscFree(ao->keys[i].rowners);
  }
  PetscFree(ao->keys);
  
  PLogObjectDestroy(ao);
  PetscHeaderDestroy(ao);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODataView_Basic_Binary"
int AODataView_Basic_Binary(PetscObject obj,Viewer viewer)
{
  AOData          ao = (AOData) obj;
  int             ierr,i,j, N, fd;
  AODataSegment   *segment;
  AODataKey       *key;
  char            paddedname[256];

  PetscFunctionBegin;
  if (!ao->datacomplete) SETERRQ(1,1,"Not all segments set in AOData");

  ierr  = ViewerBinaryGetDescriptor(viewer,&fd); CHKERRQ(ierr);

  /* write out number of keys */
  ierr = PetscBinaryWrite(fd,&ao->nkeys,1,PETSC_INT,0);CHKERRQ(ierr);

  for ( i=0; i<ao->nkeys; i++ ) {
    key = ao->keys + i;
    N   = key->N;
    /* 
       Write out name of key - use a fixed length for the name in the binary 
       file to make seeking easier
    */
    PetscMemzero(paddedname,256*sizeof(char));
    PetscStrncpy(paddedname,key->name,255);
    ierr = PetscBinaryWrite(fd,paddedname,256,PETSC_CHAR,0);CHKERRQ(ierr);
    /* write out the number of keys */
    ierr = PetscBinaryWrite(fd,&key->N,1,PETSC_INT,0);CHKERRQ(ierr);
    /* write out number of segments */
    ierr = PetscBinaryWrite(fd,&key->nsegments,1,PETSC_INT,0);CHKERRQ(ierr);
   
    /* loop over segments writing them out */
    for (j=0; j<key->nsegments; j++) {
      segment = key->segments + j;
      /* 
         Write out name of segment - use a fixed length for the name in the binary 
         file to make seeking easier
      */
      PetscMemzero(paddedname,256*sizeof(char));
      PetscStrncpy(paddedname,segment->name,255);
      ierr = PetscBinaryWrite(fd,paddedname,256,PETSC_CHAR,0);CHKERRQ(ierr);
      ierr = PetscBinaryWrite(fd,&segment->bs,1,PETSC_INT,0);CHKERRQ(ierr);
      ierr = PetscBinaryWrite(fd,&segment->datatype,1,PETSC_INT,0);CHKERRQ(ierr);
      /* write out the data */
      ierr = PetscBinaryWrite(fd,segment->data,N*segment->bs,segment->datatype,0);CHKERRQ(ierr);
    }
  }

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODataView_Basic_ASCII"
int AODataView_Basic_ASCII(PetscObject obj,Viewer viewer)
{
  AOData          ao = (AOData) obj;
  int             ierr,format,i,j,k,l,rank,size;
  FILE            *fd;
  char            *dt;
  AODataSegment   *segment;
  AODataKey       *key;

  PetscFunctionBegin;
  MPI_Comm_rank(ao->comm,&rank); if (rank) PetscFunctionReturn(0);  
  MPI_Comm_size(ao->comm,&size);

  ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
  ierr = ViewerGetFormat(viewer,&format);
  if (format == VIEWER_FORMAT_ASCII_INFO) {
    for ( i=0; i<ao->nkeys; i++ ) {
      key  = ao->keys + i;
      fprintf(fd,"AOData Key: %s Length %d Ownership: ",key->name,key->N);
      for (j=0; j<size+1; j++) {fprintf(fd,"%d ",key->rowners[j]);}
      fprintf(fd,"\n");
       
      for ( j=0; j<key->nsegments; j++ ) {
        segment = key->segments + j;
        ierr = PetscDataTypeGetName(segment->datatype,&dt); CHKERRQ(ierr);    
        fprintf(fd,"AOData Segment: %s Blocksize %d datatype %s\n",segment->name,segment->bs,dt);
      }
    }
    PetscFunctionReturn(0);
  } 

  for ( i=0; i<ao->nkeys; i++ ) {
    key  = ao->keys + i;
    fprintf(fd,"AOData Key: %s Length %d Ownership: ",key->name,key->N);
    for (j=0; j<size+1; j++) {fprintf(fd,"%d ",key->rowners[j]);}
    fprintf(fd,"\n");
      
    for ( j=0; j<key->nsegments; j++ ) {
      segment = key->segments + j;
      ierr = PetscDataTypeGetName(segment->datatype,&dt); CHKERRQ(ierr);    
      fprintf(fd,"  AOData Segment: %s Blocksize %d datatype %s\n",segment->name,segment->bs,dt);
      if (segment->datatype == PETSC_INT) {
        int *mdata = (int *) segment->data;
        for ( k=0; k<key->N; k++ ) {
          fprintf(fd," %d: ",k);
          for ( l=0; l<segment->bs; l++ ) {
            fprintf(fd,"   %d ",mdata[k*segment->bs + l]);
          }
          fprintf(fd,"\n");
        }
      } else if (segment->datatype == PETSC_DOUBLE) {
        double *mdata = (double *) segment->data;
        for ( k=0; k<key->N; k++ ) {
          fprintf(fd," %d: ",k);
          for ( l=0; l<segment->bs; l++ ) {
            fprintf(fd,"   %18.16e ",mdata[k*segment->bs + l]);
          }
          fprintf(fd,"\n");
        }
      } else if (segment->datatype == PETSC_SCALAR) {
        Scalar *mdata = (Scalar *) segment->data;
        for ( k=0; k<key->N; k++ ) {
          fprintf(fd," %d: ",k);
          for ( l=0; l<segment->bs; l++ ) {
#if !defined(USE_PETSC_COMPLEX)
            fprintf(fd,"   %18.16e ",mdata[k*segment->bs + l]);
#else
            Scalar x = mdata[k*segment->bs + l];
            if (imag(x) > 0.0) {
              fprintf(fd," %18.16e + %18.16e i \n",real(x),imag(x));
            } else if (imag(x) < 0.0) {
              fprintf(fd,"   %18.16e - %18.16e i \n",real(x),-imag(x));
            } else {
              fprintf(fd,"   %18.16e \n",real(x));
            }
#endif
          }
        }
        fprintf(fd,"\n");
      } else {
        SETERRQ(1,1,"Unknown PETSc data format");
      }
    }
  }
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "AODataView_Basic"
int AODataView_Basic(PetscObject obj,Viewer viewer)
{
  AOData          ao = (AOData) obj;
  int             rank,ierr;
  ViewerType      vtype;

  PetscFunctionBegin;
  MPI_Comm_rank(ao->comm,&rank); if (rank) PetscFunctionReturn(0);

  if (!viewer) {
    viewer = VIEWER_STDOUT_SELF; 
  }

  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (vtype  == ASCII_FILE_VIEWER || vtype == ASCII_FILES_VIEWER) { 
    ierr = AODataView_Basic_ASCII(obj,viewer); CHKERRQ(ierr);
  } else if (vtype == BINARY_FILE_VIEWER) {
    ierr = AODataView_Basic_Binary(obj,viewer); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

int AODataSegmentAdd_Basic(AOData aodata,char *name,char *segname,int bs,int n,int *keys,void *data,
                           PetscDataType dtype)
{
  AODataSegment    *segment;
  AODataKey        *key;
  int              N,size,rank,ierr,*lens,i,*disp,*akeys,datasize,*fkeys,len,ikey,iseg,flag;
  MPI_Datatype     mtype;
  char             *adata,*fdata;
  MPI_Comm         comm = aodata->comm;

  PetscFunctionBegin;
  ierr = AODataFindSegment_Private(aodata,name,segname,&flag,&ikey,&iseg);CHKERRQ(ierr);
  if (flag == 0) SETERRQ(1,1,"Segment already defined");
  if (flag == -1) SETERRQ(1,1,"No room for additional segments");

  key     = aodata->keys + ikey;
  segment = key->segments + iseg;

  segment->bs       = bs;
  segment->datatype = dtype;

  /* transmit all lengths to all processors */
  MPI_Comm_size(comm,&size);
  MPI_Comm_rank(comm,&rank);
  lens = (int *) PetscMalloc( 2*size*sizeof(int) ); CHKPTRQ(lens);
  disp = lens + size;
  MPI_Allgather(&n,1,MPI_INT,lens,1,MPI_INT,comm);
  N =  0;
  for ( i=0; i<size; i++ ) {
    disp[i]  = N;
    N       += lens[i];
  }
  if (N != key->N) SETERRQ(1,1,"Did not provide correct number of keys for keyname");

  ierr = PetscDataTypeToMPIDataType(dtype,&mtype);CHKERRQ(ierr);
  ierr = PetscDataTypeGetSize(dtype,&datasize); CHKERRQ(ierr);

  /*
    Allocate space for all keys and all data 
  */
  akeys = (int *) PetscMalloc((key->N+1)*sizeof(int)); CHKPTRQ(akeys);
  adata = (char *) PetscMalloc((N*bs+1)*datasize); CHKPTRQ(adata);


  MPI_Allgatherv(keys,n,MPI_INT,akeys,lens,disp,MPI_INT,comm);
  for ( i=0; i<size; i++ ) {
    disp[i] *= bs;
    lens[i] *= bs;
  }
  MPI_Allgatherv(data,n*bs,mtype,adata,lens,disp,mtype,comm);
  PetscFree(lens);

  /*
    Now we have all the keys and data we need to put it in order
  */
  fkeys = (int *) PetscMalloc((key->N+1)*sizeof(int)); CHKPTRQ(fkeys);
  PetscMemzero(fkeys, (key->N+1)*sizeof(int));
  fdata = (char *) PetscMalloc((N*bs+1)*datasize); CHKPTRQ(fdata);

  for ( i=0; i<N; i++ ) {
    if (fkeys[akeys[i]] != 0) {
      SETERRQ(1,1,"Duplicate key");
    }
    if (fkeys[akeys[i]] >= N) {
      SETERRQ(1,1,"Key out of range");
    }
    fkeys[akeys[i]] = 1;
    PetscMemcpy(fdata+i*bs*datasize,adata+i*bs*datasize,bs*datasize);
  }
  for ( i=0; i<N; i++ ) {
    if (!fkeys[i]) {
      SETERRQ(1,1,"Missing key");
    }
  }
  PetscFree(akeys);
  PetscFree(adata);
  PetscFree(fkeys);

  segment->data = (void *) fdata;
  key->nsegments++;

  len           = PetscStrlen(segname);
  segment->name = (char *) PetscMalloc((len+1)*sizeof(char));CHKPTRQ(segment->name);
  PetscStrcpy(segment->name,segname);


  PetscFunctionReturn(0);
}

int AODataSegmentGet_Basic(AOData ao,char *name,char *segname,int n,int *keys,void **data)
{
  AODataSegment    *segment; 
  int              ierr,dsize,i,bs,ikey,iseg,flag;
  char             *idata, *odata;
  
  PetscFunctionBegin;
  /* find the correct segment */
  ierr = AODataFindSegment_Private(ao,name,segname,&flag,&ikey,&iseg);CHKERRQ(ierr);
  if (flag) SETERRQ(1,1,"Cannot locate segment");

  segment = ao->keys[ikey].segments+iseg;

  ierr  = PetscDataTypeGetSize(segment->datatype,&dsize); CHKERRQ(ierr);
  bs    = segment->bs;
  odata = (char *) PetscMalloc((n+1)*bs*dsize);CHKPTRQ(odata);
  idata = (char *) segment->data;
  for ( i=0; i<n; i++ ) {
    PetscMemcpy(odata + i*bs*dsize,idata + keys[i]*bs*dsize,bs*dsize);
  }
  *data = (void *) odata;
  PetscFunctionReturn(0);
}

int AODataSegmentRestore_Basic(AOData aodata,char *name,char *segname,int n,int *keys,void **data)
{
  PetscFunctionBegin;
  PetscFree(*data);
  PetscFunctionReturn(0);
}

int AODataSegmentGetLocal_Basic(AOData ao,char *name,char *segname,int n,int *keys,void **data)
{
  int           ierr,flag,ikey,*globals,*locals,bs;
  PetscDataType dtype;

  PetscFunctionBegin;
  ierr = AODataFindKey_Private(ao,segname,&flag,&ikey);CHKERRQ(ierr);
  if (flag) SETERRQ(1,1,"Segment does not have corresponding key");
  if (!ao->keys[ikey].ltog) SETERRQ(1,1,"No local to global mapping set for key");
  ierr = AODataSegmentGetInfo(ao,name,segname,PETSC_NULL,PETSC_NULL,&bs,&dtype); CHKERRQ(ierr);
  if (dtype != PETSC_INT) SETERRQ(1,1,"Datatype of segment must be PETSC_INT");

  /* get the values in global indexing */
  ierr = AODataSegmentGet_Basic(ao,name,segname,n,keys,(void **)&globals);CHKERRQ(ierr);
  
  /* allocate space to store them in local indexing */
  locals = (int *) PetscMalloc((n+1)*bs*sizeof(int));CHKPTRQ(locals);

  ierr = ISGlobalToLocalMappingApply(ao->keys[ikey].ltog,IS_GTOLM_MASK,n*bs,globals,PETSC_NULL,locals);
         CHKERRQ(ierr);

  ierr = AODataSegmentRestore_Basic(ao,name,segname,n,keys,(void **)&globals);CHKERRQ(ierr);

  *data = (void *) locals;
  PetscFunctionReturn(0);
}

int AODataSegmentRestoreLocal_Basic(AOData aodata,char *name,char *segname,int n,int *keys,void **data)
{
  PetscFunctionBegin;
  PetscFree(*data);
  PetscFunctionReturn(0);
}

int AODataKeyRemap_Basic(AOData aodata, char *key,IS is)
{
  IS   isall;
  int  ierr,*gindices,i,j,nkey,nseg,k,*ii,nk,ikey,nisall,*igindices,flag,dsize,bs;
  char *data,*tmpdata;

  PetscFunctionBegin;

  /* get the old to new mapping for all key indices */
  ierr = ISAllGather(is,&isall);CHKERRQ(ierr);
  ierr = ISGetIndices(isall,&gindices);CHKERRQ(ierr);

  /* invert the gindices */
  ierr = ISGetSize(isall,&nisall);CHKERRQ(ierr);
  igindices = (int *) PetscMalloc( nisall*sizeof(int) ); CHKPTRQ(igindices);
  for ( i=0; i<nisall; i++ ) {
    igindices[gindices[i]] = i;
  }

  /* remap all the values in the segments that match the key */
  nkey = aodata->nkeys;
  for ( i=0; i<nkey; i++ ) {
    nseg = aodata->keys[i].nsegments;
    for ( j=0; j<nseg; j++ ) {
      if (PetscStrcmp(key,aodata->keys[i].segments[j].name)) continue;
      if (aodata->keys[i].segments[j].datatype != PETSC_INT) {
        SETERRQ(1,1,"Segment name same as key but not integer type");
      }
      /* Could also create and use a AOCreateXXX() for these */
      nk = aodata->keys[i].segments[j].bs*aodata->keys[i].N;
      ii = (int *) aodata->keys[i].segments[j].data;
      for ( k=0; k<nk; k++ ) {
        if (ii[k] >= 0) ii[k] = igindices[ii[k]];
      }
    }
  }
  PetscFree(igindices);

  /* reorder in the arrays all the values for the key */
  ierr = AODataFindKey_Private(aodata,key,&flag,&ikey);CHKERRQ(ierr);
  if (flag) SETERRQ(1,1,"Could not find key");
  nseg = aodata->keys[ikey].nsegments;
  nk   = aodata->keys[ikey].N;
  for ( j=0; j<nseg; j++ ) {
    ierr = PetscDataTypeGetSize(aodata->keys[ikey].segments[j].datatype,&dsize);CHKERRQ(ierr);
    bs      = aodata->keys[ikey].segments[j].bs;
    data    = (char *) aodata->keys[ikey].segments[j].data;
    tmpdata = (char *) PetscMalloc((nk+1)*bs*dsize);CHKPTRQ(tmpdata);

    for ( k=0; k<nk; k++ ) {
      PetscMemcpy(tmpdata+k*bs*dsize,data+gindices[k]*bs*dsize,bs*dsize);
    }
    
    PetscMemcpy(data,tmpdata,bs*nk*dsize);
    PetscFree(tmpdata);
  }

  ierr = ISRestoreIndices(isall,&gindices);CHKERRQ(ierr);
  ierr = ISDestroy(isall);
  PetscFunctionReturn(0);
}

extern int AODataSegmentGetReduced_Basic(AOData,char *,char *,int,int*,IS *);

static struct _AODataOps myops = {AODataSegmentAdd_Basic,
                                  AODataSegmentGet_Basic,
                                  AODataSegmentRestore_Basic,
                                  AODataSegmentGetLocal_Basic,
                                  AODataSegmentRestoreLocal_Basic,
                                  AODataSegmentGetReduced_Basic,
                                  AODataKeyRemap_Basic};

#undef __FUNC__  
#define __FUNC__ "AODataCreateBasic" 
/*@C
   AODataCreateBasic - Creates a AO datastructure.

   Input Parameters:
.  comm  - MPI communicator that is to share AO
.  n - total number of keys that will be added

   Output Parameter:
.  aoout - the new database

   Options Database Key:
$   -ao_data_view : call AODataView() at the conclusion of AODataAdd()
$   -ao_data_view_info : call AODataView() at the conclusion of AODataAdd()

.keywords: AOData, create

.seealso: AODataAdd(), AODataDestroy()
@*/
int AODataCreateBasic(MPI_Comm comm,int n,AOData *aoout)
{
  AOData        ao;

  PetscFunctionBegin;
  *aoout = 0;
  PetscHeaderCreate(ao, _p_AOData,AODATA_COOKIE,AODATA_BASIC,comm,AODataDestroy,AODataView); 
  PLogObjectCreate(ao);
  PLogObjectMemory(ao,sizeof(struct _p_AOData) + n*sizeof(void *));

  PetscMemcpy(&ao->ops,&myops,sizeof(myops));
  ao->destroy  = AODataDestroy_Basic;
  ao->view     = AODataView_Basic;

  ao->nkeys        = 0;
  ao->keys         = (AODataKey*) PetscMalloc((n+1)*sizeof(AODataKey));CHKPTRQ(ao->keys);
  ao->nkeys_max    = n;
  ao->datacomplete = 0;

  *aoout = ao; PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "AODataLoadBasic" 
/*@C
   AODataLoadBasic - Loads a AO database from a file.

   Input Parameters:
.  viewer - the binary file containing the data

   Output Parameter:
.  aoout - the new database

   Options Database Key:
$   -ao_data_view : call AODataView() at the conclusion of AODataLoadBasic()
$   -ao_data_view_info : call AODataView() at the conclusion of AODataLoadBasic()

.keywords: AOData, create, load

.seealso: AODataAdd(), AODataDestroy(), AODataCreateBasic()
@*/
int AODataLoadBasic(Viewer viewer,AOData *aoout)
{
  AOData        ao;
  int           fd,nkeys,i,len,flg1,ierr,dsize,j,size,rank;
  char          paddedname[256];
  AODataSegment *seg;
  AODataKey     *key;
  MPI_Comm      comm;
  ViewerType    vtype;

  PetscFunctionBegin;
  *aoout = 0;
  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (vtype  != BINARY_FILE_VIEWER) {
    SETERRQ(1,1,"Viewer must be obtained from ViewerFileOpenBinary()");
  }

  ierr = PetscObjectGetComm((PetscObject)viewer,&comm); CHKERRQ(ierr);
  MPI_Comm_size(comm,&size);
  MPI_Comm_rank(comm,&rank);

  ierr = ViewerBinaryGetDescriptor(viewer,&fd); CHKERRQ(ierr);

  /* read in number of segments */
  ierr = PetscBinaryRead(fd,&nkeys,1,PETSC_INT);CHKERRQ(ierr);

  PetscHeaderCreate(ao, _p_AOData,AODATA_COOKIE,AODATA_BASIC,comm,AODataDestroy,AODataView); 
  PLogObjectCreate(ao);
  PLogObjectMemory(ao,sizeof(struct _p_AOData) + nkeys*sizeof(void *));

  PetscMemcpy(&ao->ops,&myops,sizeof(myops));
  ao->destroy  = AODataDestroy_Basic;
  ao->view     = AODataView_Basic;

  ao->nkeys      = nkeys;
  ao->keys       = (AODataKey*) PetscMalloc((nkeys+1)*sizeof(AODataKey));CHKPTRQ(ao->keys);
  ao->nkeys_max  = nkeys;
  
  for ( i=0; i<nkeys; i++ ) {
    key = ao->keys + i;

    /* read in key name */
    ierr = PetscBinaryRead(fd,paddedname,256,PETSC_CHAR); CHKERRQ(ierr);
    len  = PetscStrlen(paddedname);
    key->name = (char *) PetscMalloc((len+1)*sizeof(char));CHKPTRQ(key->name);
    PetscStrcpy(key->name,paddedname);

    ierr = PetscBinaryRead(fd,&key->N,1,PETSC_INT); CHKERRQ(ierr);    

    /* determine Nlocal and rowners for key */
    key->nlocal = key->N/size + ((key->N % size) > rank);
    key->rowners = (int *) PetscMalloc((size+1)*sizeof(int));CHKPTRQ(key->rowners);
    MPI_Allgather(&key->nlocal,1,MPI_INT,key->rowners+1,1,MPI_INT,comm);
    key->rowners[0] = 0;
    for (j=2; j<=size; j++ ) {
      key->rowners[j] += key->rowners[j-1];
    }
    key->rstart        = key->rowners[rank];
    key->rend          = key->rowners[rank+1];

    /* loop keys segments, reading them in */
    ierr = PetscBinaryRead(fd,&key->nsegments,1,PETSC_INT); CHKERRQ(ierr);    
    seg = key->segments = (AODataSegment *) PetscMalloc(key->nsegments*sizeof(AODataSegment));CHKPTRQ(seg);
    for ( j=0; j<key->nsegments; j++ ) {

      /* read in segment name */
      ierr = PetscBinaryRead(fd,paddedname,256,PETSC_CHAR); CHKERRQ(ierr);
      len  = PetscStrlen(paddedname);
      seg->name = (char *) PetscMalloc((len+1)*sizeof(char));CHKPTRQ(seg->name);
      PetscStrcpy(seg->name,paddedname);

      /* read in segment blocksize and datatype */
      ierr = PetscBinaryRead(fd,&seg->bs,1,PETSC_INT);CHKERRQ(ierr);
      ierr = PetscBinaryRead(fd,&seg->datatype,1,PETSC_INT);CHKERRQ(ierr);

      /* allocate the space for the data */
      ierr = PetscDataTypeGetSize(seg->datatype,&dsize); CHKERRQ(ierr);
      seg->data = (void *) PetscMalloc(key->N*seg->bs*dsize);CHKPTRQ(seg->data);
      /* read in the data */
      ierr = PetscBinaryRead(fd,seg->data,key->N*seg->bs,seg->datatype);CHKERRQ(ierr);
      seg++;
    }
  }
  *aoout = ao; 

  ierr = OptionsHasName(PETSC_NULL,"-ao_data_view",&flg1); CHKERRQ(ierr);
  if (flg1) {
    ierr = AODataView(ao,VIEWER_STDOUT_(comm)); CHKERRQ(ierr);
  }
  ierr = OptionsHasName(PETSC_NULL,"-ao_data_view_info",&flg1); CHKERRQ(ierr);
  if (flg1) {
    ierr = ViewerPushFormat(VIEWER_STDOUT_(comm),VIEWER_FORMAT_ASCII_INFO,0);CHKERRQ(ierr);
    ierr = AODataView(ao,VIEWER_STDOUT_(comm)); CHKERRQ(ierr);
    ierr = ViewerPopFormat(VIEWER_STDOUT_(comm));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

