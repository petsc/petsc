#include <petscwebclient.h>
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma gcc diagnostic ignored "-Wdeprecated-declarations"

/*
    Encodes and decodes from MIME Base64
*/
static char encoding_table[] = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                                'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
                                'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
                                'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f',
                                'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                                'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
                                'w', 'x', 'y', 'z', '0', '1', '2', '3',
                                '4', '5', '6', '7', '8', '9', '+', '/'};

static PetscErrorCode base64_encode(const unsigned char *data,unsigned char *encoded_data,size_t len)
{
  static size_t  mod_table[] = {0, 2, 1};
  size_t         i,j;
  size_t         input_length,output_length;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStrlen((const char*)data,&input_length);CHKERRQ(ierr);
  output_length = 4 * ((input_length + 2) / 3);
  if (output_length > len) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Output length not large enough");

  for (i = 0, j = 0; i < input_length;) {
     uint32_t octet_a = i < input_length ? (unsigned char)data[i++] : 0;
     uint32_t octet_b = i < input_length ? (unsigned char)data[i++] : 0;
     uint32_t octet_c = i < input_length ? (unsigned char)data[i++] : 0;
     uint32_t triple = (octet_a << 0x10) + (octet_b << 0x08) + octet_c;

     encoded_data[j++] = encoding_table[(triple >> 3 * 6) & 0x3F];
     encoded_data[j++] = encoding_table[(triple >> 2 * 6) & 0x3F];
     encoded_data[j++] = encoding_table[(triple >> 1 * 6) & 0x3F];
     encoded_data[j++] = encoding_table[(triple >> 0 * 6) & 0x3F];
  }
  encoded_data[j] = 0;
  for (i = 0; i < mod_table[input_length % 3]; i++) encoded_data[output_length - 1 - i] = '=';
  PetscFunctionReturn(0);
}

PETSC_UNUSED static PetscErrorCode base64_decode(const unsigned char *data,unsigned char* decoded_data, size_t length)
{
  static char    decoding_table[257];
  static int     decode_table_built = 0;
  size_t         i,j;
  PetscErrorCode ierr;
  size_t         input_length,output_length;

  PetscFunctionBegin;
  if (!decode_table_built) {
    for (i = 0; i < 64; i++) decoding_table[(unsigned char) encoding_table[i]] = i;
    decode_table_built = 1;
  }

  ierr = PetscStrlen((const char*)data,&input_length);CHKERRQ(ierr);
  if (input_length % 4 != 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Input length must be divisible by 4");

  output_length = input_length / 4 * 3;
  if (data[input_length - 1] == '=') (output_length)--;
  if (data[input_length - 2] == '=') (output_length)--;
  if (output_length > length) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Output length too shore");

  for (i = 0, j = 0; i < input_length;) {
    uint32_t sextet_a = data[i] == '=' ? 0 & i++ : decoding_table[(int)data[i++]];
    uint32_t sextet_b = data[i] == '=' ? 0 & i++ : decoding_table[(int)data[i++]];
    uint32_t sextet_c = data[i] == '=' ? 0 & i++ : decoding_table[(int)data[i++]];
    uint32_t sextet_d = data[i] == '=' ? 0 & i++ : decoding_table[(int)data[i++]];
    uint32_t triple = (sextet_a << 3 * 6) + (sextet_b << 2 * 6) + (sextet_c << 1 * 6) + (sextet_d << 0 * 6);

    if (j < output_length) decoded_data[j++] = (triple >> 2 * 8) & 0xFF;
    if (j < output_length) decoded_data[j++] = (triple >> 1 * 8) & 0xFF;
    if (j < output_length) decoded_data[j++] = (triple >> 0 * 8) & 0xFF;
  }
  decoded_data[j] = 0;
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_UNISTD_H)
#include <unistd.h>
#endif

/*@C
     PetscGlobusAuthorize - Get an access token allowing PETSc applications to make Globus file transfer requests

   Not collective, only the first process in MPI_Comm does anything

   Input Parameters:
+  comm - the MPI communicator
-  tokensize - size of the token array

   Output Parameters:
.  access_token - can be used with PetscGlobusUpLoad() for 30 days

   Notes:
    This call requires stdout and stdin access from process 0 on the MPI communicator

   You can run src/sys/webclient/tutorials/globusobtainaccesstoken to get an access token

   Level: intermediate

.seealso: PetscGoogleDriveRefresh(), PetscGoogleDriveUpload(), PetscURLShorten(), PetscGlobusUpload()

@*/
PetscErrorCode PetscGlobusAuthorize(MPI_Comm comm,char access_token[],size_t tokensize)
{
  SSL_CTX        *ctx;
  SSL            *ssl;
  int            sock;
  PetscErrorCode ierr;
  char           buff[8*1024],*ptr,head[1024];
  PetscMPIInt    rank;
  size_t         len;
  PetscBool      found;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
  if (!rank) {
    if (!isatty(fileno(PETSC_STDOUT))) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Requires users input/output");
    ierr = PetscPrintf(comm,"Enter globus username:");CHKERRQ(ierr);
    ptr  = fgets(buff, 1024, stdin);
    if (!ptr) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_FILE_READ, "Error reading from stdin: %d", errno);
    ierr = PetscStrlen(buff,&len);CHKERRQ(ierr);
    buff[len-1] = ':'; /* remove carriage return at end of line */

    ierr = PetscPrintf(comm,"Enter globus password:");CHKERRQ(ierr);
    ptr  = fgets(buff+len, 1024-len, stdin);
    if (!ptr) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_FILE_READ, "Error reading from stdin: %d", errno);
    ierr = PetscStrlen(buff,&len);CHKERRQ(ierr);
    buff[len-1] = '\0'; /* remove carriage return at end of line */
    ierr = PetscStrcpy(head,"Authorization: Basic ");CHKERRQ(ierr);
    ierr = base64_encode((const unsigned char*)buff,(unsigned char*)(head+21),sizeof(head)-21);CHKERRQ(ierr);
    ierr = PetscStrcat(head,"\r\n");CHKERRQ(ierr);

    ierr = PetscSSLInitializeContext(&ctx);CHKERRQ(ierr);
    ierr = PetscHTTPSConnect("nexus.api.globusonline.org",443,ctx,&sock,&ssl);CHKERRQ(ierr);
    ierr = PetscHTTPSRequest("GET","nexus.api.globusonline.org/goauth/token?grant_type=client_credentials",head,"application/x-www-form-urlencoded",NULL,ssl,buff,sizeof(buff));CHKERRQ(ierr);
    ierr = PetscSSLDestroyContext(ctx);CHKERRQ(ierr);
    close(sock);

    ierr   = PetscPullJSONValue(buff,"access_token",access_token,tokensize,&found);CHKERRQ(ierr);
    if (!found) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Globus did not return access token");

    ierr = PetscPrintf(comm,"Here is your Globus access token, save it in a save place, in the future you can run PETSc\n");CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"programs with the option -globus_access_token %s\n",access_token);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"to access Globus automatically\n");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


/*@C
     PetscGlobusGetTransfers - Get a record of current transfers requested from Globus

   Not collective, only the first process in MPI_Comm does anything

   Input Parameters:
+  comm - the MPI communicator
.  access_token - Globus access token, if NULL will check in options database for -globus_access_token XXX otherwise
                  will call PetscGlobusAuthorize().
-  buffsize - size of the buffer

   Output Parameters:
.  buff - location to put Globus information

   Level: intermediate

.seealso: PetscGoogleDriveRefresh(), PetscGoogleDriveUpload(), PetscURLShorten(), PetscGlobusUpload(), PetscGlobusAuthorize()

@*/
PetscErrorCode PetscGlobusGetTransfers(MPI_Comm comm,const char access_token[],char buff[],size_t buffsize)
{
  SSL_CTX        *ctx;
  SSL            *ssl;
  int            sock;
  PetscErrorCode ierr;
  char           head[4096];
  PetscMPIInt    rank;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
  if (!rank) {
    ierr = PetscStrcpy(head,"Authorization : Globus-Goauthtoken ");CHKERRQ(ierr);
    if (access_token) {
      ierr = PetscStrcat(head,access_token);CHKERRQ(ierr);
    } else {
      PetscBool set;
      char      accesstoken[4096];
      ierr = PetscOptionsGetString(NULL,NULL,"-globus_access_token",accesstoken,sizeof(accesstoken),&set);CHKERRQ(ierr);
      if (!set) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Pass in Globus accesstoken or use -globus_access_token XXX");
      ierr = PetscStrcat(head,accesstoken);CHKERRQ(ierr);
    }
    ierr = PetscStrcat(head,"\r\n");CHKERRQ(ierr);

    ierr = PetscSSLInitializeContext(&ctx);CHKERRQ(ierr);
    ierr = PetscHTTPSConnect("transfer.api.globusonline.org",443,ctx,&sock,&ssl);CHKERRQ(ierr);
    ierr = PetscHTTPSRequest("GET","transfer.api.globusonline.org/v0.10/tasksummary",head,"application/json",NULL,ssl,buff,buffsize);CHKERRQ(ierr);
    ierr = PetscSSLDestroyContext(ctx);CHKERRQ(ierr);
    close(sock);
  }
  PetscFunctionReturn(0);
}

/*@C
     PetscGlobusUpload - Loads a file to Globus

     Not collective, only the first process in the MPI_Comm uploads the file

  Input Parameters:
+   comm - MPI communicator
.   access_token - obtained with PetscGlobusAuthorize(), pass NULL to use -globus_access_token XXX from the PETSc database
-   filename - file to upload

  Options Database:
.  -globus_access_token XXX - the Globus token

   Level: intermediate

.seealso: PetscURLShorten(), PetscGoogleDriveAuthorize(), PetscGoogleDriveRefresh(), PetscGlobusAuthorize()

@*/
PetscErrorCode PetscGlobusUpload(MPI_Comm comm,const char access_token[],const char filename[])
{
  SSL_CTX        *ctx;
  SSL            *ssl;
  int            sock;
  PetscErrorCode ierr;
  char           head[4096],buff[8*1024],body[4096],submission_id[4096];
  PetscMPIInt    rank;
  PetscBool      flg,found;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
  if (!rank) {
    ierr = PetscTestFile(filename,'r',&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to find file: %s",filename);

    ierr = PetscStrcpy(head,"Authorization : Globus-Goauthtoken ");CHKERRQ(ierr);
    if (access_token) {
      ierr = PetscStrcat(head,access_token);CHKERRQ(ierr);
    } else {
      PetscBool set;
      char      accesstoken[4096];
      ierr = PetscOptionsGetString(NULL,NULL,"-globus_access_token",accesstoken,sizeof(accesstoken),&set);CHKERRQ(ierr);
      if (!set) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Pass in Globus accesstoken or use -globus_access_token XXX");
      ierr = PetscStrcat(head,accesstoken);CHKERRQ(ierr);
    }
    ierr = PetscStrcat(head,"\r\n");CHKERRQ(ierr);

    /* Get Globus submission id */
    ierr = PetscSSLInitializeContext(&ctx);CHKERRQ(ierr);
    ierr = PetscHTTPSConnect("transfer.api.globusonline.org",443,ctx,&sock,&ssl);CHKERRQ(ierr);
    ierr = PetscHTTPSRequest("GET","transfer.api.globusonline.org/v0.10/submission_id",head,"application/json",NULL,ssl,buff,sizeof(buff));CHKERRQ(ierr);
    ierr = PetscSSLDestroyContext(ctx);CHKERRQ(ierr);
    close(sock);
    ierr   = PetscPullJSONValue(buff,"value",submission_id,sizeof(submission_id),&found);CHKERRQ(ierr);
    if (!found) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Globus did not return submission id");

    /* build JSON body of transfer request */
    ierr = PetscStrcpy(body,"{");CHKERRQ(ierr);
    ierr = PetscPushJSONValue(body,"submission_id",submission_id,sizeof(body));CHKERRQ(ierr);                 ierr = PetscStrcat(body,",");CHKERRQ(ierr);
    ierr = PetscPushJSONValue(body,"DATA_TYPE","transfer",sizeof(body));CHKERRQ(ierr);                        ierr = PetscStrcat(body,",");CHKERRQ(ierr);
    ierr = PetscPushJSONValue(body,"sync_level","null",sizeof(body));CHKERRQ(ierr);                           ierr = PetscStrcat(body,",");CHKERRQ(ierr);
    ierr = PetscPushJSONValue(body,"source_endpoint","barryfsmith#MacBookPro",sizeof(body));CHKERRQ(ierr);    ierr = PetscStrcat(body,",");CHKERRQ(ierr);
    ierr = PetscPushJSONValue(body,"label","PETSc transfer label",sizeof(body));CHKERRQ(ierr);                ierr = PetscStrcat(body,",");CHKERRQ(ierr);
    ierr = PetscPushJSONValue(body,"length","1",sizeof(body));CHKERRQ(ierr);                                  ierr = PetscStrcat(body,",");CHKERRQ(ierr);
    ierr = PetscPushJSONValue(body,"destination_endpoint","mcs#home",sizeof(body));CHKERRQ(ierr);             ierr = PetscStrcat(body,",");CHKERRQ(ierr);

    ierr = PetscStrcat(body,"\"DATA\": [ {");CHKERRQ(ierr);
    ierr = PetscPushJSONValue(body,"source_path","/~/FEM_GPU.pdf",sizeof(body));CHKERRQ(ierr);                ierr = PetscStrcat(body,",");CHKERRQ(ierr);
    ierr = PetscPushJSONValue(body,"destination_path","/~/FEM_GPU.pdf",sizeof(body));CHKERRQ(ierr);           ierr = PetscStrcat(body,",");CHKERRQ(ierr);
    ierr = PetscPushJSONValue(body,"verify_size","null",sizeof(body));CHKERRQ(ierr);                          ierr = PetscStrcat(body,",");CHKERRQ(ierr);
    ierr = PetscPushJSONValue(body,"recursive","false",sizeof(body));CHKERRQ(ierr);                           ierr = PetscStrcat(body,",");CHKERRQ(ierr);
    ierr = PetscPushJSONValue(body,"DATA_TYPE","transfer_item",sizeof(body));CHKERRQ(ierr);
    ierr = PetscStrcat(body,"} ] }");CHKERRQ(ierr);

    ierr = PetscSSLInitializeContext(&ctx);CHKERRQ(ierr);
    ierr = PetscHTTPSConnect("transfer.api.globusonline.org",443,ctx,&sock,&ssl);CHKERRQ(ierr);
    ierr = PetscHTTPSRequest("POST","transfer.api.globusonline.org/v0.10/transfer",head,"application/json",body,ssl,buff,sizeof(buff));CHKERRQ(ierr);
    ierr = PetscSSLDestroyContext(ctx);CHKERRQ(ierr);
    close(sock);
    ierr   = PetscPullJSONValue(buff,"code",submission_id,sizeof(submission_id),&found);CHKERRQ(ierr);
    if (!found) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Globus did not return code on transfer");
    ierr = PetscStrcmp(submission_id,"Accepted",&found);CHKERRQ(ierr);
    if (!found) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Globus did not accept transfer");
  }
  PetscFunctionReturn(0);
}


