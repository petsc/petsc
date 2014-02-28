
#include <petscsys.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <netdb.h>
#include <fcntl.h>
#include <signal.h>
#include <unistd.h>
#include <string.h>

#include <openssl/ssl.h>
#include <openssl/err.h>

static BIO *bio_err = NULL;

#define PASSWORD "password"

#if defined(PETSC_USE_CERTIFICATE)
static int password_cb(char *buf,int num, int rwflag,void *userdata)
{
  if (num < strlen(PASSWORD)+1) return(0);
  strcpy(buf,PASSWORD);
  return(strlen(PASSWORD));
}
#endif

static void sigpipe_handle(int x)
{
}

#undef __FUNCT__
#define __FUNCT__ "PetscSSLInitializeContext"
/*
    PetscSSLInitializeContext - Set up an SSL context suitable for initiating HTTPS requests.

    If built with PETSC_USE_CERTIFICATE requires the user have created a self-signed certificate with

$    ./CA.pl  -newcert  (using the passphrase of password)
$    cat newkey.pem newcert.pem > sslclient.pem

    and put the resulting file in either the current directory (with the application) or in the home directory. This seems kind of
    silly but it was all I could figure out.

*/
PetscErrorCode PetscSSLInitializeContext(SSL_CTX **octx)
{
    SSL_METHOD     *meth;
    SSL_CTX        *ctx;
#if defined(PETSC_USE_CERTIFICATE)
    char           keyfile[PETSC_MAX_PATH_LEN];
    PetscBool      exists;
    PetscErrorCode ierr;
#endif

    PetscFunctionBegin;
    if (!bio_err){
      SSL_library_init();
      SSL_load_error_strings();
      bio_err = BIO_new_fp(stderr,BIO_NOCLOSE);
    }

    /* Set up a SIGPIPE handler */
    signal(SIGPIPE,sigpipe_handle);

    meth = SSLv23_method();
    ctx  = SSL_CTX_new(meth);

#if defined(PETSC_USE_CERTIFICATE)
    /* Locate keyfile */
    ierr = PetscStrcpy(keyfile,"sslclient.pem");CHKERRQ(ierr);
    ierr = PetscTestFile(keyfile,'r',&exists);CHKERRQ(ierr);
    if (!exists) {
      ierr = PetscGetHomeDirectory(keyfile,PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
      ierr = PetscStrcat(keyfile,"/");CHKERRQ(ierr);
      ierr = PetscStrcat(keyfile,"sslclient.pem");CHKERRQ(ierr);
      ierr = PetscTestFile(keyfile,'r',&exists);CHKERRQ(ierr);
      if (!exists) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to locate sslclient.pem file in current directory or home directory");
    }

    /* Load our keys and certificates*/
    if (!(SSL_CTX_use_certificate_chain_file(ctx,keyfile))) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Cannot read certificate file");

    SSL_CTX_set_default_passwd_cb(ctx,password_cb);
    if (!(SSL_CTX_use_PrivateKey_file(ctx,keyfile,SSL_FILETYPE_PEM))) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Cannot read key file");
#endif

    *octx = ctx;
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscSSLDestroyContext"
PetscErrorCode PetscSSLDestroyContext(SSL_CTX *ctx)
{
  PetscFunctionBegin;
  SSL_CTX_free(ctx);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscHTTPSRequest"
/*
     PetscHTTPSRequest - Send a request to an HTTPS server

   Input Parameters:
+   type - either "POST" or "GET"
.   url - complete URL of request including https://
.   header - additional header information, may be NULL
.   ctype - data type of body, for example application/json
.   body - data to send to server
.   ssl - obtained with PetscHTTPSConnect()
-   buffsize - size of buffer

   Output Parameter:
.   buff - everything returned from server
 */
static PetscErrorCode PetscHTTPSRequest(const char type[],const char url[],const char header[],const char ctype[],const char body[],SSL *ssl,char buff[],size_t buffsize)
{
  char           *request=0;
  char           contentlength[40],contenttype[80];
  int            r;
  size_t         request_len,len,headlen,bodylen,contentlen,urllen,typelen,contenttypelen = 0;
  PetscErrorCode ierr;
  PetscBool      flg;

  PetscFunctionBegin;
  ierr = PetscStrbeginswith(url,"https://",&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"URL must begin with https://");
  if (header) {
    ierr = PetscStrendswith(header,"\r\n",&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"header must end with \\r\\n");
  }

  ierr = PetscStrlen(type,&typelen);CHKERRQ(ierr);
  ierr = PetscStrlen(url,&urllen);CHKERRQ(ierr);
  if (ctype) {
    ierr = PetscSNPrintf(contenttype,80,"Content-Type: %s\r\n",ctype);CHKERRQ(ierr);
    ierr = PetscStrlen(contenttype,&contenttypelen);CHKERRQ(ierr);
  }
  ierr = PetscStrlen(header,&headlen);CHKERRQ(ierr);
  ierr = PetscStrlen(body,&bodylen);CHKERRQ(ierr);
  ierr = PetscSNPrintf(contentlength,40,"Content-Length: %d\r\n\r\n",(int)bodylen);CHKERRQ(ierr);
  ierr = PetscStrlen(contentlength,&contentlen);CHKERRQ(ierr);

  /* Now construct our HTTP request */
  request_len = typelen + 1 + urllen + 35 + headlen + contenttypelen + contentlen + bodylen + 1;
  ierr = PetscMalloc1(request_len,&request);CHKERRQ(ierr);
  ierr = PetscStrcpy(request,type);CHKERRQ(ierr);
  ierr = PetscStrcat(request," ");CHKERRQ(ierr);
  ierr = PetscStrcat(request,url);CHKERRQ(ierr);
  ierr = PetscStrcat(request," HTTP/1.1\r\nUser-Agent:PETScClient\r\n");CHKERRQ(ierr);
  ierr = PetscStrcat(request,header);CHKERRQ(ierr);
  if (ctype) {
    ierr = PetscStrcat(request,contenttype);CHKERRQ(ierr);
  }
  ierr = PetscStrcat(request,contentlength);CHKERRQ(ierr);
  ierr = PetscStrcat(request,body);CHKERRQ(ierr);
  ierr = PetscStrlen(request,&request_len);CHKERRQ(ierr);
  ierr = PetscInfo1(NULL,"HTTPS request follows: \n%s\n",request);CHKERRQ(ierr);

  r = SSL_write(ssl,request,request_len);
  switch (SSL_get_error(ssl,r)){
    case SSL_ERROR_NONE:
      if (request_len != r) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Incomplete write to SSL socket");
      break;
      default:
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"SSL socket write problem");
  }

  /* Now read the server's response, assuming  that it's terminated by a close */
  r = SSL_read(ssl,buff,(int)buffsize);
  len = r;
  switch (SSL_get_error(ssl,r)){
  case SSL_ERROR_NONE:
    break;
  case SSL_ERROR_ZERO_RETURN:
    SSL_shutdown(ssl);  /* ignore shutdown error message */
    break;
  case SSL_ERROR_SYSCALL:
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"SSL read problem");
  }
  buff[len] = 0; /* null terminate string */
  ierr = PetscInfo1(NULL,"HTTPS result follows: \n%s\n",buff);CHKERRQ(ierr);

  SSL_free(ssl);
  ierr = PetscFree(request);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscHTTPSConnect"
PetscErrorCode PetscHTTPSConnect(const char host[],int port,SSL_CTX *ctx,int *sock,SSL **ssl)
{
  BIO            *sbio;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Connect the TCP socket*/
  ierr = PetscOpenSocket(host,port,sock);CHKERRQ(ierr);

  /* Connect the SSL socket */
  *ssl = SSL_new(ctx);
  sbio = BIO_new_socket(*sock,BIO_NOCLOSE);
  SSL_set_bio(*ssl,sbio,sbio);
  if (SSL_connect(*ssl) <= 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"SSL connect error");
  PetscFunctionReturn(0);
}

/*
    This file is not included in the respository since it contains authorization secrets
*/
#include <../src/sys/webclient/authorization.h>

#undef __FUNCT__
#define __FUNCT__ "PetscGoogleDriveRefresh"
/*@C
     PetscGoogleDriveRefresh - Get a new authorization token for accessing Google drive from PETSc from a refresh token

   Not collective, only the first process in the MPI_Comm does anything

   Input Parameters:
+   comm - MPI communicator
.   refresh token - obtained with PetscGoogleDriveAuthorize(), if NULL PETSc will first look for one in the options data 
                    if not found it will call PetscGoogleDriveAuthorize()
-   tokensize - size of the output string access_token

   Output Parameter:
.   access_token - token that can be passed to PetscGoogleDriveUpload()

.seealso: PetscURLShorten(), PetscGoogleDriveAuthorize(), PetscGoogleDriveUpload()

@*/
PetscErrorCode PetscGoogleDriveRefresh(MPI_Comm comm,const char refresh_token[],char access_token[],size_t tokensize)
{
  SSL_CTX        *ctx;
  SSL            *ssl;
  int            sock;
  PetscErrorCode ierr;
  char           buff[8*1024],body[1024],*access,*ctmp;
  PetscMPIInt    rank;
  char           *refreshtoken = (char*)refresh_token;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    if (!refresh_token) {
      PetscBool set;
      ierr = PetscMalloc1(512,&refreshtoken);CHKERRQ(ierr);
      ierr = PetscOptionsGetString(NULL,"-refresh_token",refreshtoken,512,&set);CHKERRQ(ierr);
      if (!set) {
        ierr = PetscGoogleDriveAuthorize(comm,access_token,refreshtoken,512*sizeof(char));CHKERRQ(ierr);
        ierr = PetscFree(refreshtoken);CHKERRQ(ierr);
        PetscFunctionReturn(0);
      }
    }
    ierr = PetscSSLInitializeContext(&ctx);CHKERRQ(ierr);
    ierr = PetscHTTPSConnect("accounts.google.com",443,ctx,&sock,&ssl);CHKERRQ(ierr);
    ierr = PetscStrcpy(body,"&client_id=");CHKERRQ(ierr);
    ierr = PetscStrcat(body,PETSC_CLIENT_ID);CHKERRQ(ierr);
    ierr = PetscStrcat(body,"&client_secret=");CHKERRQ(ierr);
    ierr = PetscStrcat(body,PETSC_CLIENT_SECRET);CHKERRQ(ierr);
    ierr = PetscStrcat(body,"&refresh_token=");CHKERRQ(ierr);
    ierr = PetscStrcat(body,refreshtoken);CHKERRQ(ierr);
    if (!refresh_token) {ierr = PetscFree(refreshtoken);CHKERRQ(ierr);}
    ierr = PetscStrcat(body,"&grant_type=refresh_token");CHKERRQ(ierr);

    ierr = PetscHTTPSRequest("POST","https://accounts.google.com/o/oauth2/token",NULL,"application/x-www-form-urlencoded",body,ssl,buff,sizeof(buff));CHKERRQ(ierr);
    ierr = PetscSSLDestroyContext(ctx);CHKERRQ(ierr);
    close(sock);

    ierr   = PetscStrstr(buff,"\"access_token\" : \"",&access);CHKERRQ(ierr);
    if (!access) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Did not receive access token from Google");
    access += 18;
    ierr   = PetscStrchr(access,'\"',&ctmp);CHKERRQ(ierr);
    if (!ctmp) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Access token from Google is misformed");
    *ctmp  = 0;
    ierr   = PetscStrncpy(access_token,access,tokensize);CHKERRQ(ierr);
    *ctmp  = '\"';
  }
  PetscFunctionReturn(0);
}

#include <sys/stat.h>

#undef __FUNCT__
#define __FUNCT__ "PetscGoogleDriveUpload"
/*@C
     PetscGoogleDriveUpload - Loads a file to the Google Drive

     Not collective, only the first process in the MPI_Comm uploads the file

  Input Parameters:
+   comm - MPI communicator
.   access_token - obtained with PetscGoogleDriveRefresh(), pass NULL to have PETSc generate one
-   filename - file to upload; if you upload multiple times it will have different names each time on Google Drive

  Options Database:
.  -refresh_token   XXX

  Usage Patterns:
    With PETSc option -refresh_token  XXX given
    PetscGoogleDriveUpload(comm,NULL,filename);        will upload file with no user interaction

    Without PETSc option -refresh_token XXX given
    PetscGoogleDriveUpload(comm,NULL,filename);        for first use will prompt user to authorize access to Google Drive with their processor

    With PETSc option -refresh_token  XXX given
    PetscGoogleDriveRefresh(comm,NULL,access_token,sizeof(access_token));
    PetscGoogleDriveUpload(comm,access_token,filename);

    With refresh token entered in some way by the user
    PetscGoogleDriveRefresh(comm,refresh_token,access_token,sizeof(access_token));
    PetscGoogleDriveUpload(comm,access_token,filename);

    PetscGoogleDriveAuthorize(comm,access_token,refresh_token,sizeof(access_token));
    PetscGoogleDriveUpload(comm,access_token,filename);

.seealso: PetscURLShorten(), PetscGoogleDriveAuthorize(), PetscGoogleDriveRefresh()

@*/
PetscErrorCode PetscGoogleDriveUpload(MPI_Comm comm,const char access_token[],const char filename[])
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

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    ierr = PetscStrcpy(head,"Authorization: Bearer ");CHKERRQ(ierr);
    ierr = PetscStrcat(head,access_token);CHKERRQ(ierr);
    ierr = PetscStrcat(head,"\r\n");CHKERRQ(ierr);
    ierr = PetscStrcat(head,"uploadType: multipart\r\n");CHKERRQ(ierr);

    ierr = stat(filename,&sb);
    if (ierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to stat file: %s",filename);
    len = 1024 + sb.st_size;
    ierr = PetscMalloc1(len,&body);CHKERRQ(ierr);
    ierr = PetscStrcpy(body,"--foo_bar_baz\r\n"
                         "Content-Type: application/json\r\n\r\n"
                         "{"
                         "\"title\": \"");
    ierr = PetscStrcat(body,filename);
    ierr = PetscStrcat(body,"\","
                         "\"mimeType\": \"text.html\","
                         "\"description\": \" a file\""
                         "}\r\n\r\n"
                         "--foo_bar_baz\r\n"
                         "Content-Type: text/html\r\n\r\n");
    ierr = PetscStrlen(body,&blen);CHKERRQ(ierr);
    fd = fopen (filename, "r");
    if (!fd) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to open file: %s",filename);
    rd = fread (body+blen, sizeof (unsigned char), sb.st_size, fd);
    if (rd != sb.st_size) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to read entire file: %s %d %d",filename,(int)rd,sb.st_size);
    fclose(fd);
    body[blen + rd] = 0;
    ierr = PetscStrcat(body,"\r\n\r\n"
                            "--foo_bar_baz\r\n");
    ierr = PetscSSLInitializeContext(&ctx);CHKERRQ(ierr);
    ierr = PetscHTTPSConnect("www.googleapis.com",443,ctx,&sock,&ssl);CHKERRQ(ierr);
    ierr = PetscHTTPSRequest("POST","https://www.googleapis.com/upload/drive/v2/files/",head,"multipart/related; boundary=\"foo_bar_baz\"",body,ssl,buff,sizeof(buff));CHKERRQ(ierr);
    ierr = PetscFree(body);CHKERRQ(ierr);
    ierr = PetscSSLDestroyContext(ctx);CHKERRQ(ierr);
    close(sock);
    ierr   = PetscStrstr(buff,"\"title\"",&title);CHKERRQ(ierr);
    if (!title) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Upload of file %s failed",filename);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PetscGoogleDriveAuthorize"
/*@C
     PetscGoogleDriveAuthorize - Get authorization and refresh token for accessing Google drive from PETSc

   Not collective, only the first process in MPI_Comm does anything

   Input Parameters:
+  comm - the MPI communicator
-  tokensize - size of the token arrays

   Output Parameters:
+  access_token - can be used with PetscGoogleDriveUpload() for this one session
-  refresh_token - can be used for ever to obtain new access_tokens with PetscGoogleDriveRefresh(), guard this like a password
                   it gives access to your Google Drive

   Notes: This call requires stdout and stdin access from process 0 on the MPI communicator

   You can run src/sys/webclient/examples/tutorials/obtainrefreshtoken to get a refresh token and then in the future pass it to
   PETSc programs with -refresh_token XXX

.seealso: PetscGoogleDriveRefresh(), PetscGoogleDriveUpload(), PetscURLShorten()

@*/
PetscErrorCode PetscGoogleDriveAuthorize(MPI_Comm comm,char access_token[],char refresh_token[],size_t tokensize)
{
  SSL_CTX        *ctx;
  SSL            *ssl;
  int            sock;
  PetscErrorCode ierr;
  char           buff[8*1024],*ptr,body[1024],*access,*refresh,*ctmp;
  PetscMPIInt    rank;
  size_t         len;

  PetscFunctionBegin;
  ierr = PetscPrintf(comm,"Cut and paste the following into your browser:\n"
                          "https://accounts.google.com/o/oauth2/auth?"
                          "scope=https%%3A%%2F%%2Fwww.googleapis.com%%2Fauth%%2Fdrive.file&"
                          "redirect_uri=urn:ietf:wg:oauth:2.0:oob&"
                          "response_type=code&"
                          "client_id="
                          PETSC_CLIENT_ID
                          "\n\n");CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"Paste the result here:");CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    ptr  = fgets(buff, 1024, stdin);
    if (!ptr) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_FILE_READ, "Error reading from stdin: %d", errno);
    ierr = PetscStrlen(buff,&len);CHKERRQ(ierr);
    buff[len-1] = 0; /* remove carriage return at end of line */

    ierr = PetscSSLInitializeContext(&ctx);CHKERRQ(ierr);
    ierr = PetscHTTPSConnect("accounts.google.com",443,ctx,&sock,&ssl);CHKERRQ(ierr);
    ierr = PetscStrcpy(body,"code=");CHKERRQ(ierr);
    ierr = PetscStrcat(body,buff);CHKERRQ(ierr);
    ierr = PetscStrcat(body,"&client_id=");CHKERRQ(ierr);
    ierr = PetscStrcat(body,PETSC_CLIENT_ID);CHKERRQ(ierr);
    ierr = PetscStrcat(body,"&client_secret=");CHKERRQ(ierr);
    ierr = PetscStrcat(body,PETSC_CLIENT_SECRET);CHKERRQ(ierr);
    ierr = PetscStrcat(body,"&redirect_uri=urn:ietf:wg:oauth:2.0:oob&");CHKERRQ(ierr);
    ierr = PetscStrcat(body,"grant_type=authorization_code");CHKERRQ(ierr);

    ierr = PetscHTTPSRequest("POST","https://accounts.google.com/o/oauth2/token",NULL,"application/x-www-form-urlencoded",body,ssl,buff,sizeof(buff));CHKERRQ(ierr);
    ierr = PetscSSLDestroyContext(ctx);CHKERRQ(ierr);
    close(sock);

    ierr   = PetscStrstr(buff,"\"access_token\" : \"",&access);CHKERRQ(ierr);
    if (!access) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Did not receive access token from Google");
    access += 18;
    ierr   = PetscStrchr(access,'\"',&ctmp);CHKERRQ(ierr);
    if (!ctmp) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Access token from Google is misformed");
    *ctmp  = 0;
    ierr   = PetscStrncpy(access_token,access,tokensize);CHKERRQ(ierr);
    *ctmp  = '\"';

    ierr   = PetscStrstr(buff,"\"refresh_token\" : \"",&refresh);CHKERRQ(ierr);
    if (!refresh) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Did not receive refresh token from Google");
    refresh += 19;
    ierr   = PetscStrchr(refresh,'\"',&ctmp);CHKERRQ(ierr);
    if (!ctmp) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Refresh token from Google is misformed");
    *ctmp  = 0;
    ierr = PetscStrncpy(refresh_token,refresh,tokensize);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PetscURLShorten"
/*@C
     PetscURLShorten - Uses Google's service to get a short url for a long url

    Input Parameters:
+    url - long URL you want shortened
-    lenshorturl - length of buffer to contain short URL

    Output Parameter:
.    shorturl - the shortened URL

.seealso: PetscGoogleDriveRefresh(), PetscGoogleDriveUpload(), PetscGoogleDriveAuthorize()
@*/
PetscErrorCode PetscURLShorten(const char url[],char shorturl[],size_t lenshorturl)
{
  SSL_CTX        *ctx;
  SSL            *ssl;
  int            sock;
  PetscErrorCode ierr;
  char           buff[1024],body[512],*sub1,*sub2;

  PetscFunctionBegin;
  ierr = PetscSSLInitializeContext(&ctx);CHKERRQ(ierr);
  ierr = PetscHTTPSConnect("www.googleapis.com",443,ctx,&sock,&ssl);CHKERRQ(ierr);
  ierr = PetscSNPrintf(body,512,"{\"longUrl\": \"%s\"}",url);CHKERRQ(ierr);
  ierr = PetscHTTPSRequest("POST","https://www.googleapis.com/urlshortener/v1/url",NULL,"application/json",body,ssl,buff,sizeof(buff));CHKERRQ(ierr);
  ierr = PetscSSLDestroyContext(ctx);CHKERRQ(ierr);
  close(sock);
  ierr = PetscStrstr(buff,"\"id\": \"",&sub1);CHKERRQ(ierr);
  if (sub1) {
    sub1 += 7;
    ierr = PetscStrstr(sub1,"\"",&sub2);CHKERRQ(ierr);
    if (!sub2) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Google did not shorten URL");
    sub2[0] = 0;
    ierr = PetscStrncpy(shorturl,sub1,lenshorturl);CHKERRQ(ierr);
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Google did not shorten URL");
  PetscFunctionReturn(0);
}

