
#include <petscwebclient.h>
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma gcc diagnostic ignored "-Wdeprecated-declarations"

/*
   These variables identify the code as a PETSc application to Box.

   See -   https://stackoverflow.com/questions/4616553/using-oauth-in-free-open-source-software
   Users can get their own application IDs - goto https://developer.box.com

*/
#define PETSC_BOX_CLIENT_ID  "sse42nygt4zqgrdwi0luv79q1u1f0xza"
#define PETSC_BOX_CLIENT_ST  "A0Dy4KgOYLB2JIYZqpbze4EzjeIiX5k4"

#if defined(PETSC_HAVE_SAWS)
#include <mongoose.h>

static volatile char *result = NULL;

static int PetscBoxWebServer_Private(struct mg_connection *conn)
{
  const struct mg_request_info *request_info = mg_get_request_info(conn);
  result = (char*) request_info->query_string;
  return 1;  /* Mongoose will now not handle the request */
}

/*
    Box can only return an authorization code to a Webserver, hence we need to start one up and wait for
    the authorization code to arrive from Box
*/
static PetscErrorCode PetscBoxStartWebServer_Private(void)
{
  PetscErrorCode      ierr;
  int                 optionsLen = 5;
  const char          *options[optionsLen];
  struct mg_callbacks callbacks;
  struct mg_context   *ctx;
  char                keyfile[PETSC_MAX_PATH_LEN];
  PetscBool           exists;

  PetscFunctionBegin;
  options[0] = "listening_ports";
  options[1] = "8081s";

  ierr = PetscStrcpy(keyfile,"sslclient.pem");CHKERRQ(ierr);
  ierr = PetscTestFile(keyfile,'r',&exists);CHKERRQ(ierr);
  if (!exists) {
    ierr = PetscGetHomeDirectory(keyfile,PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
    ierr = PetscStrcat(keyfile,"/");CHKERRQ(ierr);
    ierr = PetscStrcat(keyfile,"sslclient.pem");CHKERRQ(ierr);
    ierr = PetscTestFile(keyfile,'r',&exists);CHKERRQ(ierr);
    if (!exists) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to locate sslclient.pem file in current directory or home directory");
  }

  options[2] = "ssl_certificate";
  options[3] = keyfile;
  options[4] = NULL;

  /* Prepare callbacks structure. We have only one callback, the rest are NULL. */
  ierr = PetscMemzero(&callbacks, sizeof(callbacks));CHKERRQ(ierr);
  callbacks.begin_request = PetscBoxWebServer_Private;
  ctx = mg_start(&callbacks, NULL, options);
  if (!ctx) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Unable to start up webserver");
  while (!result) {};
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_UNISTD_H)
#include <unistd.h>
#endif

/*@C
     PetscBoxAuthorize - Get authorization and refresh token for accessing Box drive from PETSc

   Not collective, only the first process in MPI_Comm does anything

   Input Parameters:
+  comm - the MPI communicator
-  tokensize - size of the token arrays

   Output Parameters:
+  access_token - can be used with PetscBoxUpload() for this one session
-  refresh_token - can be used for ever to obtain new access_tokens with PetscBoxRefresh(), guard this like a password
                   it gives access to your Box Drive

   Notes:
    This call requires stdout and stdin access from process 0 on the MPI communicator

   You can run src/sys/webclient/tutorials/boxobtainrefreshtoken to get a refresh token and then in the future pass it to
   PETSc programs with -box_refresh_token XXX

   This requires PETSc be installed using --with-saws or --download-saws

   Requires the user have created a self-signed ssl certificate with

$    saws/CA.pl  -newcert  (using the passphrase of password)
$    cat newkey.pem newcert.pem > sslclient.pem

    and put the resulting file in either the current directory (with the application) or in the home directory. This seems kind of
    silly but it was all I could figure out.

   Level: intermediate

.seealso: PetscBoxRefresh(), PetscBoxUpload(), PetscURLShorten()

@*/
PetscErrorCode PetscBoxAuthorize(MPI_Comm comm,char access_token[],char refresh_token[],size_t tokensize)
{
  SSL_CTX        *ctx;
  SSL            *ssl;
  int            sock;
  PetscErrorCode ierr;
  char           buff[8*1024],body[1024];
  PetscMPIInt    rank;
  PetscBool      flg,found;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    if (!isatty(fileno(PETSC_STDOUT))) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Requires users input/output");
    ierr = PetscPrintf(comm,"Cut and paste the following into your browser:\n\n"
                            "https://www.box.com/api/oauth2/authorize?"
                            "response_type=code&"
                            "client_id="
                            PETSC_BOX_CLIENT_ID
                            "&state=PETScState"
                            "\n\n");CHKERRQ(ierr);
    ierr = PetscBoxStartWebServer_Private();CHKERRQ(ierr);
    ierr = PetscStrbeginswith((const char*)result,"state=PETScState&code=",&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Did not get expected string from Box got %s",result);
    ierr = PetscStrncpy(buff,(const char*)result+22,sizeof(buff));CHKERRQ(ierr);

    ierr = PetscSSLInitializeContext(&ctx);CHKERRQ(ierr);
    ierr = PetscHTTPSConnect("www.box.com",443,ctx,&sock,&ssl);CHKERRQ(ierr);
    ierr = PetscStrcpy(body,"code=");CHKERRQ(ierr);
    ierr = PetscStrcat(body,buff);CHKERRQ(ierr);
    ierr = PetscStrcat(body,"&client_id=");CHKERRQ(ierr);
    ierr = PetscStrcat(body,PETSC_BOX_CLIENT_ID);CHKERRQ(ierr);
    ierr = PetscStrcat(body,"&client_secret=");CHKERRQ(ierr);
    ierr = PetscStrcat(body,PETSC_BOX_CLIENT_ST);CHKERRQ(ierr);
    ierr = PetscStrcat(body,"&grant_type=authorization_code");CHKERRQ(ierr);

    ierr = PetscHTTPSRequest("POST","www.box.com/api/oauth2/token",NULL,"application/x-www-form-urlencoded",body,ssl,buff,sizeof(buff));CHKERRQ(ierr);
    ierr = PetscSSLDestroyContext(ctx);CHKERRQ(ierr);
    close(sock);

    ierr   = PetscPullJSONValue(buff,"access_token",access_token,tokensize,&found);CHKERRQ(ierr);
    if (!found) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Box did not return access token");
    ierr   = PetscPullJSONValue(buff,"refresh_token",refresh_token,tokensize,&found);CHKERRQ(ierr);
    if (!found) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Box did not return refresh token");

    ierr = PetscPrintf(comm,"Here is your Box refresh token, save it in a save place, in the future you can run PETSc\n");CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"programs with the option -box_refresh_token %s\n",refresh_token);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"to access Box Drive automatically\n");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
#endif

/*@C
     PetscBoxRefresh - Get a new authorization token for accessing Box drive from PETSc from a refresh token

   Not collective, only the first process in the MPI_Comm does anything

   Input Parameters:
+   comm - MPI communicator
.   refresh token - obtained with PetscBoxAuthorize(), if NULL PETSc will first look for one in the options data
                    if not found it will call PetscBoxAuthorize()
-   tokensize - size of the output string access_token

   Output Parameter:
+   access_token - token that can be passed to PetscBoxUpload()
-   new_refresh_token - the old refresh token is no longer valid, not this is different than Google where the same refresh_token is used forever

   Level: intermediate

.seealso: PetscURLShorten(), PetscBoxAuthorize(), PetscBoxUpload()

@*/
PetscErrorCode PetscBoxRefresh(MPI_Comm comm,const char refresh_token[],char access_token[],char new_refresh_token[],size_t tokensize)
{
  SSL_CTX        *ctx;
  SSL            *ssl;
  int            sock;
  PetscErrorCode ierr;
  char           buff[8*1024],body[1024];
  PetscMPIInt    rank;
  char           *refreshtoken = (char*)refresh_token;
  PetscBool      found;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    if (!refresh_token) {
      PetscBool set;
      ierr = PetscMalloc1(512,&refreshtoken);CHKERRQ(ierr);
      ierr = PetscOptionsGetString(NULL,NULL,"-box_refresh_token",refreshtoken,sizeof(refreshtoken),&set);CHKERRQ(ierr);
#if defined(PETSC_HAVE_SAWS)
      if (!set) {
        ierr = PetscBoxAuthorize(comm,access_token,new_refresh_token,512*sizeof(char));CHKERRQ(ierr);
        ierr = PetscFree(refreshtoken);CHKERRQ(ierr);
        PetscFunctionReturn(0);
      }
#else
      if (!set) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Must provide refresh token with -box_refresh_token XXX");
#endif
    }
    ierr = PetscSSLInitializeContext(&ctx);CHKERRQ(ierr);
    ierr = PetscHTTPSConnect("www.box.com",443,ctx,&sock,&ssl);CHKERRQ(ierr);
    ierr = PetscStrcpy(body,"client_id=");CHKERRQ(ierr);
    ierr = PetscStrcat(body,PETSC_BOX_CLIENT_ID);CHKERRQ(ierr);
    ierr = PetscStrcat(body,"&client_secret=");CHKERRQ(ierr);
    ierr = PetscStrcat(body,PETSC_BOX_CLIENT_ST);CHKERRQ(ierr);
    ierr = PetscStrcat(body,"&refresh_token=");CHKERRQ(ierr);
    ierr = PetscStrcat(body,refreshtoken);CHKERRQ(ierr);
    if (!refresh_token) {ierr = PetscFree(refreshtoken);CHKERRQ(ierr);}
    ierr = PetscStrcat(body,"&grant_type=refresh_token");CHKERRQ(ierr);

    ierr = PetscHTTPSRequest("POST","www.box.com/api/oauth2/token",NULL,"application/x-www-form-urlencoded",body,ssl,buff,sizeof(buff));CHKERRQ(ierr);
    ierr = PetscSSLDestroyContext(ctx);CHKERRQ(ierr);
    close(sock);

    ierr   = PetscPullJSONValue(buff,"access_token",access_token,tokensize,&found);CHKERRQ(ierr);
    if (!found) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Box did not return access token");
    ierr   = PetscPullJSONValue(buff,"refresh_token",new_refresh_token,tokensize,&found);CHKERRQ(ierr);
    if (!found) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Box did not return refresh token");

    ierr = PetscPrintf(comm,"Here is your new Box refresh token, save it in a save place, in the future you can run PETSc\n");CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"programs with the option -box_refresh_token %s\n",new_refresh_token);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"to access Box Drive automatically\n");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#include <sys/stat.h>

/*@C
     PetscBoxUpload - Loads a file to the Box Drive

     This routine has not yet been written; it is just copied from Google Drive

     Not collective, only the first process in the MPI_Comm uploads the file

  Input Parameters:
+   comm - MPI communicator
.   access_token - obtained with PetscBoxRefresh(), pass NULL to have PETSc generate one
-   filename - file to upload; if you upload multiple times it will have different names each time on Box Drive

  Options Database:
.  -box_refresh_token   XXX

  Usage Patterns:
    With PETSc option -box_refresh_token  XXX given
    PetscBoxUpload(comm,NULL,filename);        will upload file with no user interaction

    Without PETSc option -box_refresh_token XXX given
    PetscBoxUpload(comm,NULL,filename);        for first use will prompt user to authorize access to Box Drive with their processor

    With PETSc option -box_refresh_token  XXX given
    PetscBoxRefresh(comm,NULL,access_token,sizeof(access_token));
    PetscBoxUpload(comm,access_token,filename);

    With refresh token entered in some way by the user
    PetscBoxRefresh(comm,refresh_token,access_token,sizeof(access_token));
    PetscBoxUpload(comm,access_token,filename);

    PetscBoxAuthorize(comm,access_token,refresh_token,sizeof(access_token));
    PetscBoxUpload(comm,access_token,filename);

   Level: intermediate

.seealso: PetscURLShorten(), PetscBoxAuthorize(), PetscBoxRefresh()

@*/
PetscErrorCode PetscBoxUpload(MPI_Comm comm,const char access_token[],const char filename[])
{
  SSL_CTX        *ctx;
  SSL            *ssl;
  int            sock;
  PetscErrorCode ierr;
  char           head[1024],buff[8*1024],*body,*title;
  PetscMPIInt    rank;
  struct stat    sb;
  size_t         len,blen,rd;
  FILE           *fd;
  int            err;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    ierr = PetscStrcpy(head,"Authorization: Bearer ");CHKERRQ(ierr);
    ierr = PetscStrcat(head,access_token);CHKERRQ(ierr);
    ierr = PetscStrcat(head,"\r\n");CHKERRQ(ierr);
    ierr = PetscStrcat(head,"uploadType: multipart\r\n");CHKERRQ(ierr);

    err = stat(filename,&sb);
    if (err) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to stat file: %s",filename);
    len = 1024 + sb.st_size;
    ierr = PetscMalloc1(len,&body);CHKERRQ(ierr);
    ierr = PetscStrcpy(body,"--foo_bar_baz\r\n"
                            "Content-Type: application/json\r\n\r\n"
                            "{");CHKERRQ(ierr);
    ierr = PetscPushJSONValue(body,"title",filename,len);CHKERRQ(ierr);
    ierr = PetscStrcat(body,",");CHKERRQ(ierr);
    ierr = PetscPushJSONValue(body,"mimeType","text.html",len);CHKERRQ(ierr);
    ierr = PetscStrcat(body,",");CHKERRQ(ierr);
    ierr = PetscPushJSONValue(body,"description","a file",len);CHKERRQ(ierr);
    ierr = PetscStrcat(body, "}\r\n\r\n"
                             "--foo_bar_baz\r\n"
                             "Content-Type: text/html\r\n\r\n");CHKERRQ(ierr);
    ierr = PetscStrlen(body,&blen);CHKERRQ(ierr);
    fd = fopen (filename, "r");
    if (!fd) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to open file: %s",filename);
    rd = fread (body+blen, sizeof (unsigned char), sb.st_size, fd);
    if (rd != (size_t)sb.st_size) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to read entire file: %s %d %d",filename,(int)rd,(int)sb.st_size);
    fclose(fd);
    body[blen + rd] = 0;
    ierr = PetscStrcat(body,"\r\n\r\n"
                            "--foo_bar_baz\r\n");CHKERRQ(ierr);
    ierr = PetscSSLInitializeContext(&ctx);CHKERRQ(ierr);
    ierr = PetscHTTPSConnect("www.boxapis.com",443,ctx,&sock,&ssl);CHKERRQ(ierr);
    ierr = PetscHTTPSRequest("POST","www.boxapis.com/upload/drive/v2/files/",head,"multipart/related; boundary=\"foo_bar_baz\"",body,ssl,buff,sizeof(buff));CHKERRQ(ierr);
    ierr = PetscFree(body);CHKERRQ(ierr);
    ierr = PetscSSLDestroyContext(ctx);CHKERRQ(ierr);
    close(sock);
    ierr   = PetscStrstr(buff,"\"title\"",&title);CHKERRQ(ierr);
    if (!title) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Upload of file %s failed",filename);
  }
  PetscFunctionReturn(0);
}
