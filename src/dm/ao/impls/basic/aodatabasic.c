#define PETSCDM_DLL

/*
  The most basic AOData routines. These store the entire database on each processor.
  These routines are very simple; note that we do not even use a private data structure
  for AOData, and the private datastructure for AODataSegment is just used as a simple array.

  These are made slightly complicated by having to be able to handle logical variables
  stored in bit arrays. Thus,
    - Before mallocing to hold a bit array, we shrunk the array length by a factor
      of 8 using PetscBTLength()
    - We use PetscBitMemcpy() to allow us to copy at the individual bit level;
      for regular datatypes this just does a regular memcpy().
*/

#include "src/dm/ao/aoimpl.h"          /*I  "petscao.h"  I*/
#include "petscsys.h"
#include "petscbt.h"

#undef __FUNCT__  
#define __FUNCT__ "AODataDestroy_Basic"
PetscErrorCode AODataDestroy_Basic(AOData ao)
{
  PetscErrorCode ierr;
  AODataKey     *key = ao->keys,*nextkey;
  AODataSegment *seg,*nextseg;

  PetscFunctionBegin;
  while (key) {
    ierr = PetscFree(key->name);CHKERRQ(ierr);
    if (key->ltog) {
      ierr = ISLocalToGlobalMappingDestroy(key->ltog);CHKERRQ(ierr);
    }
    seg = key->segments;
    while (seg) {
      ierr    = PetscFree(seg->data);CHKERRQ(ierr);
      ierr    = PetscFree(seg->name);CHKERRQ(ierr);
      nextseg = seg->next;
      ierr    = PetscFree(seg);CHKERRQ(ierr);
      seg     = nextseg;
    }
    ierr = PetscFree(key->rowners);CHKERRQ(ierr);
    nextkey = key->next;
    ierr = PetscFree(key);CHKERRQ(ierr);
    key     = nextkey;
  }
  ierr = PetscHeaderDestroy(ao);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AODataView_Basic_Binary"
PetscErrorCode AODataView_Basic_Binary(AOData ao,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscInt       N;
  int            fd;
  AODataSegment  *segment;
  AODataKey      *key = ao->keys;
  char           paddedname[256];

  PetscFunctionBegin;

  ierr  = PetscViewerBinaryGetDescriptor(viewer,&fd);CHKERRQ(ierr);

  /* write out number of keys */
  ierr = PetscBinaryWrite(fd,&ao->nkeys,1,PETSC_INT,PETSC_FALSE);CHKERRQ(ierr);

  while (key) {
    N   = key->N;
    /* 
       Write out name of key - use a fixed length for the name in the binary 
       file to make seeking easier
    */
    ierr = PetscMemzero(paddedname,256*sizeof(char));CHKERRQ(ierr);
    ierr = PetscStrncpy(paddedname,key->name,255);CHKERRQ(ierr);
    ierr = PetscBinaryWrite(fd,paddedname,256,PETSC_CHAR,PETSC_FALSE);CHKERRQ(ierr);
    /* write out the number of indices */
    ierr = PetscBinaryWrite(fd,&key->N,1,PETSC_INT,PETSC_FALSE);CHKERRQ(ierr);
    /* write out number of segments */
    ierr = PetscBinaryWrite(fd,&key->nsegments,1,PETSC_INT,PETSC_FALSE);CHKERRQ(ierr);
   
    /* loop over segments writing them out */
    segment = key->segments;
    while (segment) {
      /* 
         Write out name of segment - use a fixed length for the name in the binary 
         file to make seeking easier
      */
      ierr = PetscMemzero(paddedname,256*sizeof(char));CHKERRQ(ierr);
      ierr = PetscStrncpy(paddedname,segment->name,255);CHKERRQ(ierr);
      ierr = PetscBinaryWrite(fd,paddedname,256,PETSC_CHAR,PETSC_FALSE);CHKERRQ(ierr);
      ierr = PetscBinaryWrite(fd,&segment->bs,1,PETSC_INT,PETSC_FALSE);CHKERRQ(ierr);
      ierr = PetscBinaryWrite(fd,&segment->datatype,1,PETSC_INT,PETSC_FALSE);CHKERRQ(ierr);
      /* write out the data */
      ierr = PetscBinaryWrite(fd,segment->data,N*segment->bs,segment->datatype,PETSC_FALSE);CHKERRQ(ierr);
      segment = segment->next;
    }
    key = key->next;
  }

  PetscFunctionReturn(0);
}

/*
      All processors have the same data so processor 0 prints it
*/
#undef __FUNCT__  
#define __FUNCT__ "AODataView_Basic_ASCII"
PetscErrorCode AODataView_Basic_ASCII(AOData ao,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscMPIInt       rank,size;
  PetscInt          j,k,l,nkeys,nsegs,i,N,bs,zero = 0;
  char              **keynames,**segnames,*segvalue;
  const char        *stype,*dt;
  AODataSegment     *segment;
  AODataKey         *key = ao->keys;
  PetscDataType     dtype;
  PetscViewerFormat format;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(ao->comm,&rank);CHKERRQ(ierr);
  if (rank) PetscFunctionReturn(0);  
  ierr = MPI_Comm_size(ao->comm,&size);CHKERRQ(ierr);

  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_INFO) {
    ierr = AODataGetInfo(ao,&nkeys,&keynames);CHKERRQ(ierr);
    for (i=0; i<nkeys; i++) {
      ierr = AODataKeyGetInfo(ao,keynames[i],&N,0,&nsegs,&segnames);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  %s: (%D)\n",keynames[i],N);CHKERRQ(ierr);
      for (j=0; j<nsegs; j++) {
        ierr = AODataSegmentGetInfo(ao,keynames[i],segnames[j],&bs,&dtype);CHKERRQ(ierr);
        ierr = PetscDataTypeGetName(dtype,&stype);CHKERRQ(ierr);
        if (dtype == PETSC_CHAR) {
          ierr = AODataSegmentGet(ao,keynames[i],segnames[j],1,&zero,(void **)&segvalue);CHKERRQ(ierr);
          ierr = PetscViewerASCIIPrintf(viewer,"      %s: (%D) %s -> %s\n",segnames[j],bs,stype,segvalue);CHKERRQ(ierr);
          ierr = AODataSegmentRestore(ao,keynames[i],segnames[j],1,&zero,(void **)&segvalue);CHKERRQ(ierr);
        } else {
          ierr = PetscViewerASCIIPrintf(viewer,"      %s: (%D) %s\n",segnames[j],bs,stype);CHKERRQ(ierr);
        }
      }
    }
    ierr = PetscFree(keynames);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
    while (key) {
      ierr = PetscViewerASCIIPrintf(viewer,"AOData Key: %s Length %D Ownership: ",key->name,key->N);CHKERRQ(ierr);
      for (j=0; j<size+1; j++) {ierr = PetscViewerASCIIPrintf(viewer,"%D ",key->rowners[j]);CHKERRQ(ierr);}
      ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);

      segment = key->segments;
      while (segment) {      
        ierr = PetscDataTypeGetName(segment->datatype,&dt);CHKERRQ(ierr);    
        ierr = PetscViewerASCIIPrintf(viewer,"  AOData Segment: %s Blocksize %D datatype %s\n",segment->name,segment->bs,dt);CHKERRQ(ierr);
        if (segment->datatype == PETSC_INT) {
          PetscInt *mdata = (PetscInt*)segment->data;
          for (k=0; k<key->N; k++) {
            ierr = PetscViewerASCIIPrintf(viewer," %D: ",k);CHKERRQ(ierr);
            for (l=0; l<segment->bs; l++) {
              ierr = PetscViewerASCIIPrintf(viewer,"   %D ",mdata[k*segment->bs + l]);CHKERRQ(ierr);
            }
            ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
          }
        } else if (segment->datatype == PETSC_DOUBLE) {
          PetscReal *mdata = (PetscReal*)segment->data;
          for (k=0; k<key->N; k++) {
            ierr = PetscViewerASCIIPrintf(viewer," %D: ",k);CHKERRQ(ierr);
            for (l=0; l<segment->bs; l++) {
              ierr = PetscViewerASCIIPrintf(viewer,"   %18.16e ",mdata[k*segment->bs + l]);CHKERRQ(ierr);
            }
            ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
          }
        } else if (segment->datatype == PETSC_SCALAR) {
          PetscScalar *mdata = (PetscScalar*)segment->data;
          for (k=0; k<key->N; k++) {
            ierr = PetscViewerASCIIPrintf(viewer," %D: ",k);CHKERRQ(ierr);
            for (l=0; l<segment->bs; l++) {
#if !defined(PETSC_USE_COMPLEX)
              ierr = PetscViewerASCIIPrintf(viewer,"   %18.16e ",mdata[k*segment->bs + l]);CHKERRQ(ierr);
#else
              PetscScalar x = mdata[k*segment->bs + l];
              if (PetscImaginaryPart(x) > 0.0) {
                ierr = PetscViewerASCIIPrintf(viewer," %18.16e + %18.16e i \n",PetscRealPart(x),PetscImaginaryPart(x));CHKERRQ(ierr);
              } else if (PetscImaginaryPart(x) < 0.0) {
                ierr = PetscViewerASCIIPrintf(viewer,"   %18.16e - %18.16e i \n",PetscRealPart(x),-PetscImaginaryPart(x));CHKERRQ(ierr);
              } else {
                ierr = PetscViewerASCIIPrintf(viewer,"   %18.16e \n",PetscRealPart(x));CHKERRQ(ierr);
              }
#endif
            }
          }
          ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
        } else if (segment->datatype == PETSC_LOGICAL) {
          PetscBT mdata = (PetscBT) segment->data;
          for (k=0; k<key->N; k++) {
            ierr = PetscViewerASCIIPrintf(viewer," %D: ",k);CHKERRQ(ierr);
            for (l=0; l<segment->bs; l++) {
              ierr = PetscViewerASCIIPrintf(viewer,"   %d ",(int)PetscBTLookup(mdata,k*segment->bs + l));CHKERRQ(ierr);
            }
            ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
          }
        } else if (segment->datatype == PETSC_CHAR) {
          char * mdata = (char*)segment->data;
          for (k=0; k<key->N; k++) {
            ierr = PetscViewerASCIIPrintf(viewer,"  %s ",mdata + k*segment->bs);CHKERRQ(ierr);
          }
          ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
        } else {
          SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Unknown PETSc data format");
        }
        segment = segment->next;
      }
      key = key->next;
    }
  }
  ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AODataView_Basic"
PetscErrorCode AODataView_Basic(AOData ao,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscTruth     iascii,isbinary;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(ao->comm,&rank);CHKERRQ(ierr);
  if (rank) PetscFunctionReturn(0);

  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_BINARY,&isbinary);CHKERRQ(ierr);
  if (iascii) { 
    ierr = AODataView_Basic_ASCII(ao,viewer);CHKERRQ(ierr);
  } else if (isbinary) {
    ierr = AODataView_Basic_Binary(ao,viewer);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported for AOData basic",((PetscObject)viewer)->type_name);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AODataKeyRemove_Basic"
PetscErrorCode AODataKeyRemove_Basic(AOData aodata,const char name[])
{
  AODataSegment    *segment,*iseg;
  AODataKey        *key,*ikey;
  PetscErrorCode ierr;
  PetscTruth       flag;

  PetscFunctionBegin;
  ierr = AODataKeyFind_Private(aodata,name,&flag,&key);CHKERRQ(ierr);
  if (flag != 1)  PetscFunctionReturn(0);
  aodata->nkeys--;

  segment = key->segments;
  while (segment) {
    iseg    = segment->next;
    ierr = PetscFree(segment->name);CHKERRQ(ierr);
    ierr = PetscFree(segment->data);CHKERRQ(ierr);
    ierr = PetscFree(segment);CHKERRQ(ierr);
    segment = iseg;
  }
  ikey = aodata->keys;
  if (key == ikey) {
    aodata->keys = key->next;
    goto finishup1;
  }
  while (ikey->next != key) {
    ikey = ikey->next;
  }
  ikey->next = key->next;

  finishup1:

  ierr = PetscFree(key->name);CHKERRQ(ierr);
  ierr = PetscFree(key->rowners);CHKERRQ(ierr);
  ierr = PetscFree(key);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AODataSegmentRemove_Basic"
PetscErrorCode AODataSegmentRemove_Basic(AOData aodata,const char name[],const char segname[])
{
  AODataSegment    *segment,*iseg;
  AODataKey        *key;
  PetscErrorCode ierr;
  PetscTruth       flag;

  PetscFunctionBegin;
  ierr = AODataSegmentFind_Private(aodata,name,segname,&flag,&key,&iseg);CHKERRQ(ierr);
  if (!flag)  PetscFunctionReturn(0);
  key->nsegments--;

  segment = key->segments;
  if (segment == iseg) {
    key->segments = segment->next;
    goto finishup2;
  }
  while (segment->next != iseg) {
    segment = segment->next;
  }
  segment->next = iseg->next;
  segment       = iseg;
  
  finishup2:

  ierr = PetscFree(segment->name);CHKERRQ(ierr);
  ierr = PetscFree(segment->data);CHKERRQ(ierr);
  ierr = PetscFree(segment);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "AODataSegmentAdd_Basic"
PetscErrorCode AODataSegmentAdd_Basic(AOData aodata,const char name[],const char segname[],PetscInt bs,PetscInt n,PetscInt *keys,void *data,PetscDataType dtype)
{
  AODataSegment  *segment,*iseg;
  AODataKey      *key;
  PetscErrorCode ierr;
  PetscInt       N,i,*akeys,datasize,*fkeys,Nb,j;
  PetscMPIInt    size,*lens,*disp,nn;
  MPI_Datatype   mtype;
  char           *adata;
  MPI_Comm       comm = aodata->comm;
  PetscTruth     flag;

  PetscFunctionBegin;
  ierr  = AODataKeyFind_Private(aodata,name,&flag,&key);CHKERRQ(ierr);
  if (!flag)  SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Key %s doesn't exist",name);
  ierr = AODataSegmentFind_Private(aodata,name,segname,&flag,&key,&iseg);CHKERRQ(ierr);
  if (flag)  SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Segment %s in key %s already exists",name,segname);

  ierr = PetscNew(AODataSegment,&segment);CHKERRQ(ierr);
  if (iseg) {
    iseg->next    = segment;
  } else {
    key->segments = segment;
  }
  segment->next     = 0;
  segment->bs       = bs;
  segment->datatype = dtype;

  ierr = PetscDataTypeGetSize(dtype,&datasize);CHKERRQ(ierr);

  /*
     If keys not given, assume each processor provides entire data 
  */
  if (!keys && n == key->N) {
    char *fdata1;
    if (dtype == PETSC_LOGICAL) Nb = PetscBTLength(key->N); else Nb = key->N;
    ierr = PetscMalloc((Nb*bs+1)*datasize,&fdata1);CHKERRQ(ierr);
    ierr = PetscBitMemcpy(fdata1,0,data,0,key->N*bs,dtype);CHKERRQ(ierr);
    segment->data = (void*)fdata1;
  } else if (!keys) {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Keys not given, but not all data given on each processor");
  } else {
    /* transmit all lengths to all processors */
    ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
    ierr = PetscMalloc(2*size*sizeof(PetscMPIInt),&lens);CHKERRQ(ierr);
    disp = lens + size;
    nn   = n;
    ierr = MPI_Allgather(&nn,1,MPI_INT,lens,1,MPI_INT,comm);CHKERRQ(ierr);
    N =  0;
    for (i=0; i<size; i++) {
      disp[i]  = N;
      N       += lens[i];
    }
    if (N != key->N) {
      SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Did not provide correct number of keys for keyname");
    }

    /*
      Allocate space for all keys and all data 
    */
    ierr = PetscMalloc((N+1)*sizeof(PetscInt),&akeys);CHKERRQ(ierr);
    ierr = PetscMalloc((N*bs+1)*datasize,&adata);CHKERRQ(ierr);

    ierr = MPI_Allgatherv(keys,n,MPIU_INT,akeys,lens,disp,MPIU_INT,comm);CHKERRQ(ierr);
    for (i=0; i<size; i++) {
      disp[i] *= bs;
      lens[i] *= bs;
    }
   
    if (dtype != PETSC_LOGICAL) {
      char *fdata2;

      ierr = PetscDataTypeToMPIDataType(dtype,&mtype);CHKERRQ(ierr);
      ierr = MPI_Allgatherv(data,n*bs,mtype,adata,lens,disp,mtype,comm);CHKERRQ(ierr);
      ierr = PetscFree(lens);CHKERRQ(ierr);

      /*
        Now we have all the keys and data we need to put it in order
      */
      ierr = PetscMalloc((key->N+1)*sizeof(PetscInt),&fkeys);CHKERRQ(ierr);
      ierr = PetscMemzero(fkeys,(key->N+1)*sizeof(PetscInt));CHKERRQ(ierr);
      ierr = PetscMalloc((key->N*bs+1)*datasize,&fdata2);CHKERRQ(ierr);

      for (i=0; i<N; i++) {
        if (fkeys[akeys[i]] != 0) {
          SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Duplicate key");
        }
        if (fkeys[akeys[i]] >= N) {
          SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Key out of range");
        }
        fkeys[akeys[i]] = 1;
        ierr = PetscBitMemcpy(fdata2,akeys[i]*bs,adata,i*bs,bs,dtype);CHKERRQ(ierr);
      }
      for (i=0; i<N; i++) {
        if (!fkeys[i]) {
          SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Missing key");
        }
      }
      segment->data = (void*)fdata2;
    } else {
      /*
            For logical input the length is given by the user in bits; we need to 
            convert to bytes to send with MPI
      */
      PetscBT fdata3;
      PetscBT mvalues = (PetscBT) data;
      char *values;
      ierr = PetscMalloc((n+1)*bs*sizeof(char),&values);CHKERRQ(ierr);
      for (i=0; i<n; i++) {
        for (j=0; j<bs; j++) {
          if (PetscBTLookup(mvalues,i*bs+j)) values[i*bs+j] = 1; else values[i*bs+j] = 0;
        }
      }

      ierr = MPI_Allgatherv(values,n*bs,MPI_BYTE,adata,lens,disp,MPI_BYTE,comm);CHKERRQ(ierr);
      ierr = PetscFree(lens);CHKERRQ(ierr);
      ierr = PetscFree(values);CHKERRQ(ierr);

      /*
        Now we have all the keys and data we need to put it in order
      */
      ierr = PetscMalloc((key->N+1)*sizeof(PetscInt),&fkeys);CHKERRQ(ierr);
      ierr = PetscMemzero(fkeys,(key->N+1)*sizeof(PetscInt));CHKERRQ(ierr);
      ierr = PetscBTCreate(N*bs,fdata3);CHKERRQ(ierr);

      for (i=0; i<N; i++) {
        if (fkeys[akeys[i]] != 0) {
          SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Duplicate key");
        }
        if (fkeys[akeys[i]] >= N) {
          SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Key out of range");
        }
        fkeys[akeys[i]] = 1;
        for (j=0; j<bs; j++) {
          if (adata[i*bs+j]) { ierr = PetscBTSet(fdata3,i*bs+j);CHKERRQ(ierr); }
        }
      }
      for (i=0; i<N; i++) {
        if (!fkeys[i]) {
          SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Missing key");
        }
      }
      segment->data = (void*)fdata3;
    }
    ierr = PetscFree(akeys);CHKERRQ(ierr);
    ierr = PetscFree(adata);CHKERRQ(ierr);
    ierr = PetscFree(fkeys);CHKERRQ(ierr);
  }

  key->nsegments++;

  ierr = PetscStrallocpy(segname,&segment->name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AODataSegmentExtrema_Basic"
PetscErrorCode AODataSegmentGetExtrema_Basic(AOData ao,const char name[],const char segname[],void *xmax,void *xmin)
{
  AODataSegment  *segment; 
  AODataKey      *key;
  PetscErrorCode ierr;
  PetscInt       i,bs,n,j;
  PetscTruth     flag;

  PetscFunctionBegin;
  /* find the correct segment */
  ierr = AODataSegmentFind_Private(ao,name,segname,&flag,&key,&segment);CHKERRQ(ierr);
  if (!flag) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Cannot locate segment");

  n       = key->N;
  bs      = segment->bs;

  if (segment->datatype == PETSC_INT) {
    PetscInt *vmax = (PetscInt*)xmax,*vmin = (PetscInt*)xmin,*values = (PetscInt*)segment->data;
    for (j=0; j<bs; j++) {
      vmax[j] = vmin[j] = values[j];
    }
    for (i=1; i<n; i++) {
      for (j=0; j<bs; j++) {
        vmax[j] = PetscMax(vmax[j],values[bs*i+j]);
        vmin[j] = PetscMin(vmin[j],values[bs*i+j]);
      }
    }
  } else if (segment->datatype == PETSC_DOUBLE) {
    PetscReal *vmax = (PetscReal*)xmax,*vmin = (PetscReal*)xmin,*values = (PetscReal*)segment->data;
    for (j=0; j<bs; j++) {
      vmax[j] = vmin[j] = values[j];
    }
    for (i=1; i<n; i++) {
      for (j=0; j<bs; j++) {
        vmax[j] = PetscMax(vmax[j],values[bs*i+j]);
        vmin[j] = PetscMin(vmin[j],values[bs*i+j]);
      }
    }
  } else SETERRQ(PETSC_ERR_SUP,"Cannot find extrema for this data type");

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AODataSegmentGet_Basic"
PetscErrorCode AODataSegmentGet_Basic(AOData ao,const char name[],const char segname[],PetscInt n,PetscInt *keys,void **data)
{
  AODataSegment  *segment; 
  AODataKey      *key;
  PetscErrorCode ierr;
  PetscInt       dsize,i,bs,nb;
  char           *idata,*odata;
  PetscTruth     flag;
  
  PetscFunctionBegin;
  /* find the correct segment */
  ierr = AODataSegmentFind_Private(ao,name,segname,&flag,&key,&segment);CHKERRQ(ierr);
  if (!flag) SETERRQ2(PETSC_ERR_ARG_WRONG,"Cannot locate segment %s in key %s",segname,name);

  ierr  = PetscDataTypeGetSize(segment->datatype,&dsize);CHKERRQ(ierr);
  bs    = segment->bs;
  if (segment->datatype == PETSC_LOGICAL) nb = PetscBTLength(n); else nb = n;
  ierr = PetscMalloc((nb+1)*bs*dsize,&odata);CHKERRQ(ierr);
  idata = (char*)segment->data;
  for (i=0; i<n; i++) {
    ierr = PetscBitMemcpy(odata,i*bs,idata,keys[i]*bs,bs,segment->datatype);CHKERRQ(ierr);
  }
  *data = (void*)odata;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AODataSegmentRestore_Basic"
PetscErrorCode AODataSegmentRestore_Basic(AOData aodata,const char name[],const char segname[],PetscInt n,PetscInt *keys,void **data)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(*data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AODataSegmentGetLocal_Basic"
PetscErrorCode AODataSegmentGetLocal_Basic(AOData ao,const char name[],const char segname[],PetscInt n,PetscInt *keys,void **data)
{
  PetscErrorCode ierr;
  PetscInt       *globals,*locals,bs;
  PetscDataType  dtype;
  AODataKey      *key;
  PetscTruth     flag;

  PetscFunctionBegin;
  ierr = AODataKeyFind_Private(ao,segname,&flag,&key);CHKERRQ(ierr);
  if (!flag) SETERRQ(PETSC_ERR_ARG_WRONG,"Segment does not have corresponding key");
  if (!key->ltog) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"No local to global mapping set for key");
  ierr = AODataSegmentGetInfo(ao,name,segname,&bs,&dtype);CHKERRQ(ierr);
  if (dtype != PETSC_INT) SETERRQ(PETSC_ERR_ARG_WRONG,"Datatype of segment must be PETSC_INT");

  /* get the values in global indexing */
  ierr = AODataSegmentGet_Basic(ao,name,segname,n,keys,(void **)&globals);CHKERRQ(ierr);
  
  /* allocate space to store them in local indexing */
  ierr = PetscMalloc((n+1)*bs*sizeof(PetscInt),&locals);CHKERRQ(ierr);

  ierr = ISGlobalToLocalMappingApply(key->ltog,IS_GTOLM_MASK,n*bs,globals,PETSC_NULL,locals);CHKERRQ(ierr);

  ierr = AODataSegmentRestore_Basic(ao,name,segname,n,keys,(void **)&globals);CHKERRQ(ierr);

  *data = (void*)locals;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AODataSegmentRestoreLocal_Basic"
PetscErrorCode AODataSegmentRestoreLocal_Basic(AOData aodata,const char name[],const char segname[],PetscInt n,PetscInt *keys,void **data)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(*data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN PetscErrorCode AOBasicGetIndices_Private(AO,PetscInt **,PetscInt **);

#undef __FUNCT__  
#define __FUNCT__ "AODataKeyRemap_Basic"
PetscErrorCode AODataKeyRemap_Basic(AOData aodata,const char keyname[],AO ao)
{
  PetscErrorCode ierr;
  PetscInt       *inew,k,*ii,nk,dsize,bs,nkb;
  char           *data,*tmpdata;
  AODataKey      *key;
  AODataSegment  *seg;
  PetscTruth     flag,match;

  PetscFunctionBegin;

  /* remap all the values in the segments that match the key */
  key = aodata->keys;
  while (key) {
    seg = key->segments;
    while (seg) {
      ierr = PetscStrcmp(seg->name,keyname,&match);CHKERRQ(ierr);
      if (!match) {
        seg = seg->next;
        continue;
      }
      if (seg->datatype != PETSC_INT) {
        SETERRQ(PETSC_ERR_ARG_WRONG,"Segment name same as key but not integer type");
      }
      nk   = seg->bs*key->N;
      ii   = (PetscInt*)seg->data;
      ierr = AOPetscToApplication(ao,nk,ii);CHKERRQ(ierr);
      seg  = seg->next;
    }
    key = key->next;
  }
  
  ierr = AOBasicGetIndices_Private(ao,&inew,PETSC_NULL);CHKERRQ(ierr);
  /* reorder in the arrays all the values for the key */
  ierr = AODataKeyFind_Private(aodata,keyname,&flag,&key);CHKERRQ(ierr);
  if (!flag) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Could not find key");
  nk  = key->N;
  seg = key->segments;
  while (seg) {
    ierr    = PetscDataTypeGetSize(seg->datatype,&dsize);CHKERRQ(ierr);
    bs      = seg->bs;
    data    = (char*)seg->data;
    if (seg->datatype == PETSC_LOGICAL) nkb = PetscBTLength(nk*bs); else nkb = nk*bs;
    ierr = PetscMalloc((nkb+1)*dsize,&tmpdata);CHKERRQ(ierr);

    for (k=0; k<nk; k++) {
      ierr = PetscBitMemcpy(tmpdata,inew[k]*bs,data,k*bs,bs,seg->datatype);CHKERRQ(ierr);
    }
    ierr = PetscMemcpy(data,tmpdata,nkb*dsize);CHKERRQ(ierr);
    ierr = PetscFree(tmpdata);CHKERRQ(ierr);
    seg = seg->next;
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AODataKeyGetAdjacency_Basic"
PetscErrorCode AODataKeyGetAdjacency_Basic(AOData aodata,const char keyname[],Mat *adj)
{
  PetscErrorCode ierr;
  PetscInt       cnt,i,j,*jj,*ii,nlocal,n,*nb,bs,ls;
  AODataKey     *key;
  AODataSegment *seg;
  PetscTruth    flag;

  PetscFunctionBegin;
  ierr = AODataSegmentFind_Private(aodata,keyname,keyname,&flag,&key,&seg);CHKERRQ(ierr);
  if (!flag) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Cannot locate key with neighbor segment");

  /*
     Get the beginning of the neighbor list for this processor 
  */
  bs     = seg->bs;
  nb     = (PetscInt*)seg->data;
  nb    += bs*key->rstart;
  nlocal = key->rend - key->rstart;
  n      = bs*key->N;

  /*
      Assemble the adjacency graph: first we determine total number of entries
  */
  cnt = 0;
  for (i=0; i<bs*nlocal; i++) {
    if (nb[i] >= 0) cnt++;
  }
  ierr = PetscMalloc((nlocal + 1)*sizeof(PetscInt),&ii);CHKERRQ(ierr);
  ierr = PetscMalloc((cnt+1)*sizeof(PetscInt),&jj);CHKERRQ(ierr);
  ii[0] = 0;
  cnt   = 0;
  for (i=0; i<nlocal; i++) {
    ls = 0;
    for (j=0; j<bs; j++) {
      if (nb[bs*i+j] >= 0) {
        jj[cnt++] = nb[bs*i+j];
        ls++;
      }
    }
    /* now sort the column indices for this row */
    ierr = PetscSortInt(ls,jj+cnt-ls);CHKERRQ(ierr);
    ii[i+1] = cnt;
  }

  ierr = MatCreate(aodata->comm,nlocal,n,PETSC_DETERMINE,n,adj);CHKERRQ(ierr);
  ierr = MatSetType(*adj,MATMPIADJ);CHKERRQ(ierr);
  ierr = MatMPIAdjSetPreallocation(*adj,ii,jj,PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "AODataSegmentPartition_Basic"
PetscErrorCode AODataSegmentPartition_Basic(AOData aodata,const char keyname[],const char segname[])
{
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscInt        bs,i,j,*idx,nc,*isc;
  AO             ao;
  AODataKey      *key,*keyseg;
  AODataSegment  *segment;
  PetscTruth     flag;

  PetscFunctionBegin;

  ierr = AODataKeyFind_Private(aodata,segname,&flag,&keyseg);CHKERRQ(ierr);
  if (!flag) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Cannot locate segment as a key");
  ierr = PetscMalloc(keyseg->N*sizeof(PetscInt),&isc);CHKERRQ(ierr);
  ierr = PetscMemzero(isc,keyseg->N*sizeof(PetscInt));CHKERRQ(ierr);

  ierr = AODataSegmentFind_Private(aodata,keyname,segname,&flag,&key,&segment);CHKERRQ(ierr);
  if (flag != 1) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Cannot locate segment");
  ierr = MPI_Comm_size(aodata->comm,&size);CHKERRQ(ierr);

  bs                = segment->bs;

  idx = (PetscInt*)segment->data;
  nc  = 0;
  for (i=0; i<size; i++) {
    for (j=bs*key->rowners[i]; j<bs*key->rowners[i+1]; j++) {
      /* allow some keys to have fewer data than others, indicate with a -1 */
      if (idx[j] >= 0) {
	if (!isc[idx[j]]) {
	  isc[idx[j]] = ++nc;
	}
      }
    }
  }
  for (i=0; i<nc; i++) {
    isc[i]--;
  }

  ierr = AOCreateBasic(aodata->comm,keyseg->nlocal,isc+keyseg->rstart,PETSC_NULL,&ao);CHKERRQ(ierr);
  ierr = PetscFree(isc);CHKERRQ(ierr);

  ierr = AODataKeyRemap(aodata,segname,ao);CHKERRQ(ierr);
  ierr = AODestroy(ao);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AODataKeyGetActive_Basic" 
PetscErrorCode AODataKeyGetActive_Basic(AOData aodata,const char name[],const char segname[],PetscInt n,PetscInt *keys,PetscInt wl,IS *is)
{
  PetscErrorCode ierr;
  PetscInt       i,cnt,*fnd,bs;
  AODataKey      *key;
  AODataSegment  *segment;
  PetscBT        bt;
  PetscTruth     flag;

  PetscFunctionBegin;
  ierr = AODataSegmentFind_Private(aodata,name,segname,&flag,&key,&segment);CHKERRQ(ierr);
  if (!flag) SETERRQ(PETSC_ERR_ARG_WRONG,"Cannot locate segment");

  bt = (PetscBT) segment->data;
  bs = segment->bs;

  if (wl >= bs) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Bit field (wl) argument too large");

  /* count the active ones */
  cnt = 0;
  for (i=0; i<n; i++) {
    if (PetscBTLookup(bt,keys[i]*bs+wl)) {
      cnt++;
    }
  }

  ierr = PetscMalloc((cnt+1)*sizeof(PetscInt),&fnd);CHKERRQ(ierr);
  cnt = 0;
  for (i=0; i<n; i++) {
    if (PetscBTLookup(bt,keys[i]*bs+wl)) {
      fnd[cnt++] = keys[i];
    }
  }
  
  ierr = ISCreateGeneral(aodata->comm,cnt,fnd,is);CHKERRQ(ierr);
  ierr = PetscFree(fnd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AODataKeyGetActiveLocal_Basic" 
PetscErrorCode AODataKeyGetActiveLocal_Basic(AOData aodata,const char name[],const char segname[],PetscInt n,PetscInt *keys,PetscInt wl,IS *is)
{
  PetscErrorCode ierr;
  PetscInt       i,cnt,*fnd,bs,*locals;
  AODataKey      *key;
  AODataSegment  *segment;
  PetscBT        bt;
  PetscTruth     flag;

  PetscFunctionBegin;
  ierr = AODataSegmentFind_Private(aodata,name,segname,&flag,&key,&segment);CHKERRQ(ierr);
  if (!flag) SETERRQ(PETSC_ERR_ARG_WRONG,"Cannot locate segment");

  bt = (PetscBT) segment->data;
  bs = segment->bs;

  if (wl >= bs) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Bit field (wl) argument too large");

  /* count the active ones */
  cnt = 0;
  for (i=0; i<n; i++) {
    if (PetscBTLookup(bt,keys[i]*bs+wl)) {
      cnt++;
    }
  }

  ierr = PetscMalloc((cnt+1)*sizeof(PetscInt),&fnd);CHKERRQ(ierr);
  cnt = 0;
  for (i=0; i<n; i++) {
    if (PetscBTLookup(bt,keys[i]*bs+wl)) {
      fnd[cnt++] = keys[i];
    }
  }
  
  ierr = PetscMalloc((n+1)*sizeof(PetscInt),&locals);CHKERRQ(ierr);
  ierr = ISGlobalToLocalMappingApply(key->ltog,IS_GTOLM_MASK,cnt,fnd,PETSC_NULL,locals);CHKERRQ(ierr);  
  ierr = PetscFree(fnd);CHKERRQ(ierr);
  ierr = ISCreateGeneral(aodata->comm,cnt,locals,is);CHKERRQ(ierr);
  ierr = PetscFree(locals);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN PetscErrorCode AODataSegmentGetReduced_Basic(AOData,const char[],const char[],PetscInt,PetscInt*,IS *);
EXTERN PetscErrorCode AODataPublish_Petsc(PetscObject);

static struct _AODataOps myops = {AODataSegmentAdd_Basic,
                                  AODataSegmentGet_Basic,
                                  AODataSegmentRestore_Basic,
                                  AODataSegmentGetLocal_Basic,
                                  AODataSegmentRestoreLocal_Basic,
                                  AODataSegmentGetReduced_Basic,
                                  AODataSegmentGetExtrema_Basic,
                                  AODataKeyRemap_Basic,
                                  AODataKeyGetAdjacency_Basic,
                                  AODataKeyGetActive_Basic,
                                  AODataKeyGetActiveLocal_Basic,
                                  AODataSegmentPartition_Basic,
                                  AODataKeyRemove_Basic,
                                  AODataSegmentRemove_Basic,
                                  AODataDestroy_Basic,
                                  AODataView_Basic};

#undef __FUNCT__  
#define __FUNCT__ "AODataCreateBasic" 
/*@C
   AODataCreateBasic - Creates an AO datastructure.

   Collective on MPI_Comm

   Input Parameters:
.  comm  - MPI communicator that is to share AO

   Output Parameter:
.  aoout - the new database

   Options Database Keys:
+  -ao_data_view - Prints entire database at the conclusion of AODataSegmentAdd()
-  -ao_data_view_info - Prints info about database at the conclusion of AODataSegmentAdd()

   Level: intermediate

.keywords: AOData, create

.seealso: AODataSegmentAdd(), AODataDestroy()
@*/
PetscErrorCode PETSCDM_DLLEXPORT AODataCreateBasic(MPI_Comm comm,AOData *aoout)
{
  AOData         ao;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(aoout,2);
  *aoout = 0;
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = DMInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif

  ierr = PetscHeaderCreate(ao,_p_AOData,struct _AODataOps,AODATA_COOKIE,AODATA_BASIC,"AOData",comm,AODataDestroy,AODataView);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(ao,sizeof(struct _p_AOData));CHKERRQ(ierr);

  ierr = PetscMemcpy(ao->ops,&myops,sizeof(myops));CHKERRQ(ierr);
  ao->bops->publish = AODataPublish_Petsc;

  ao->nkeys        = 0;
  ao->keys         = 0;
  ao->datacomplete = 0;

  ierr = PetscPublishAll(ao);CHKERRQ(ierr);
  *aoout = ao; 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "AODataLoadBasic" 
/*@C
   AODataLoadBasic - Loads an AO database from a file.

   Collective on PetscViewer

   Input Parameters:
.  viewer - the binary file containing the data

   Output Parameter:
.  aoout - the new database

   Options Database Keys:
+  -ao_data_view - Prints entire database at the conclusion of AODataLoadBasic()
-  -ao_data_view_info - Prints info about database at the conclusion of AODataLoadBasic()

   Level: intermediate

.keywords: AOData, create, load, basic

.seealso: AODataSegmentAdd(), AODataDestroy(), AODataCreateBasic(), AODataView() 
@*/
PetscErrorCode PETSCDM_DLLEXPORT AODataLoadBasic(PetscViewer viewer,AOData *aoout)
{
  AOData        ao;
  PetscErrorCode ierr;
  int            fd;
  PetscMPIInt    size,rank;
  PetscInt       nkeys,i,dsize,j,Nb;
  char           paddedname[256];
  AODataSegment  *seg = 0;
  AODataKey      *key = 0;
  MPI_Comm       comm;
  PetscTruth     isbinary,flg1;

  PetscFunctionBegin;
  *aoout = 0;
#ifndef PETSC_USE_DYNAMIC_LIBRARIES
  ierr = DMInitializePackage(PETSC_NULL);CHKERRQ(ierr);
#endif
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_BINARY,&isbinary);CHKERRQ(ierr);
  if (!isbinary) {
    SETERRQ(PETSC_ERR_ARG_WRONG,"Viewer must be obtained from PetscViewerBinaryOpen()");
  }

  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

  ierr = PetscViewerBinaryGetDescriptor(viewer,&fd);CHKERRQ(ierr);

  /* read in number of segments */
  ierr = PetscBinaryRead(fd,&nkeys,1,PETSC_INT);CHKERRQ(ierr);

  ierr = PetscHeaderCreate(ao,_p_AOData,struct _AODataOps,AODATA_COOKIE,AODATA_BASIC,"AOData",comm,AODataDestroy,AODataView);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(ao,sizeof(struct _p_AOData) + nkeys*sizeof(void*));CHKERRQ(ierr);

  ierr = PetscMemcpy(ao->ops,&myops,sizeof(myops));CHKERRQ(ierr);
  ao->bops->publish  = AODataPublish_Petsc;

  ao->nkeys      = nkeys;

  for (i=0; i<nkeys; i++) {
    if (!i) {
      ierr     = PetscNew(AODataKey,&key);CHKERRQ(ierr);
      ao->keys = key;
    } else {
      ierr = PetscNew(AODataKey,&key->next);CHKERRQ(ierr);
      key  = key->next;
    }
    key->ltog = 0;
    key->next = 0;

    /* read in key name */
    ierr = PetscBinaryRead(fd,paddedname,256,PETSC_CHAR);CHKERRQ(ierr);
    ierr = PetscStrallocpy(paddedname,&key->name);CHKERRQ(ierr);
    ierr = PetscBinaryRead(fd,&key->N,1,PETSC_INT);CHKERRQ(ierr);    

    /* determine Nlocal and rowners for key */
    key->nlocal  = key->N/size + ((key->N % size) > rank);
    ierr = PetscMalloc((size+1)*sizeof(PetscInt),&key->rowners);CHKERRQ(ierr);
    ierr = MPI_Allgather(&key->nlocal,1,MPI_INT,key->rowners+1,1,MPI_INT,comm);CHKERRQ(ierr);
    key->rowners[0] = 0;
    for (j=2; j<=size; j++) {
      key->rowners[j] += key->rowners[j-1];
    }
    key->rstart        = key->rowners[rank];
    key->rend          = key->rowners[rank+1];

    /* loop key's segments, reading them in */
    ierr = PetscBinaryRead(fd,&key->nsegments,1,PETSC_INT);CHKERRQ(ierr);    

    for (j=0; j<key->nsegments; j++) {
      if (!j) {
        ierr          = PetscNew(AODataSegment,&seg);CHKERRQ(ierr);
        key->segments = seg;
      } else {
        ierr = PetscNew(AODataSegment,&seg->next);CHKERRQ(ierr);
        seg  = seg->next;
      }

      /* read in segment name */
      ierr = PetscBinaryRead(fd,paddedname,256,PETSC_CHAR);CHKERRQ(ierr);
      ierr = PetscStrallocpy(paddedname,&seg->name);CHKERRQ(ierr);

      /* read in segment blocksize and datatype */
      ierr = PetscBinaryRead(fd,&seg->bs,1,PETSC_INT);CHKERRQ(ierr);
      ierr = PetscBinaryRead(fd,&seg->datatype,1,PETSC_INT);CHKERRQ(ierr);

      /* allocate the space for the data */
      ierr = PetscDataTypeGetSize(seg->datatype,&dsize);CHKERRQ(ierr);
      if (seg->datatype == PETSC_LOGICAL) Nb = PetscBTLength(key->N*seg->bs); else Nb = key->N*seg->bs;
      ierr = PetscMalloc(Nb*dsize,&seg->data);CHKERRQ(ierr);
      /* read in the data */
      ierr = PetscBinaryRead(fd,seg->data,key->N*seg->bs,seg->datatype);CHKERRQ(ierr);
      seg->next = 0;
    }
  }
  *aoout = ao; 

  ierr = PetscOptionsHasName(PETSC_NULL,"-ao_data_view",&flg1);CHKERRQ(ierr);
  if (flg1) {
    ierr = AODataView(ao,PETSC_VIEWER_STDOUT_(comm));CHKERRQ(ierr);
  }
  ierr = PetscOptionsHasName(PETSC_NULL,"-ao_data_view_info",&flg1);CHKERRQ(ierr);
  if (flg1) {
    ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_(comm),PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);
    ierr = AODataView(ao,PETSC_VIEWER_STDOUT_(comm));CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_(comm));CHKERRQ(ierr);
  }
  ierr = PetscPublishAll(ao);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}




