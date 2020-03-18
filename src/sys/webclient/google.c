
#include <petscwebclient.h>
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#pragma gcc diagnostic ignored "-Wdeprecated-declarations"

/*
   These variables identify the code as a PETSc application to Google.

   See -   https://stackoverflow.com/questions/4616553/using-oauth-in-free-open-source-software
   Users can get their own application IDs - https://code.google.com/p/google-apps-manager/wiki/GettingAnOAuthConsoleKey

*/
#define PETSC_GOOGLE_CLIENT_ID  "521429262559-i19i57eek8tnt9ftpp4p91rcl0bo9ag5.apps.googleusercontent.com"
#define PETSC_GOOGLE_CLIENT_ST  "vOds_A71I3_S_aHMq_kZAI0t"
#define PETSC_GOOGLE_API_KEY    "AIzaSyDRZsOcySpWVzsUvIBL2UG3J2tcg-MXbyk"


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

   Options Database:
.  -google_refresh_token XXX   where XXX was obtained from PetscGoogleDriveAuthorize()

   Level: intermediate

.seealso: PetscURLShorten(), PetscGoogleDriveAuthorize(), PetscGoogleDriveUpload()

@*/
PetscErrorCode PetscGoogleDriveRefresh(MPI_Comm comm,const char refresh_token[],char access_token[],size_t tokensize)
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
      ierr = PetscOptionsGetString(NULL,NULL,"-google_refresh_token",refreshtoken,512,&set);CHKERRQ(ierr);
      if (!set) {
        ierr = PetscGoogleDriveAuthorize(comm,access_token,refreshtoken,512*sizeof(char));CHKERRQ(ierr);
        ierr = PetscFree(refreshtoken);CHKERRQ(ierr);
        PetscFunctionReturn(0);
      }
    }
    ierr = PetscSSLInitializeContext(&ctx);CHKERRQ(ierr);
    ierr = PetscHTTPSConnect("accounts.google.com",443,ctx,&sock,&ssl);CHKERRQ(ierr);
    ierr = PetscStrcpy(body,"client_id=");CHKERRQ(ierr);
    ierr = PetscStrcat(body,PETSC_GOOGLE_CLIENT_ID);CHKERRQ(ierr);
    ierr = PetscStrcat(body,"&client_secret=");CHKERRQ(ierr);
    ierr = PetscStrcat(body,PETSC_GOOGLE_CLIENT_ST);CHKERRQ(ierr);
    ierr = PetscStrcat(body,"&refresh_token=");CHKERRQ(ierr);
    ierr = PetscStrcat(body,refreshtoken);CHKERRQ(ierr);
    if (!refresh_token) {ierr = PetscFree(refreshtoken);CHKERRQ(ierr);}
    ierr = PetscStrcat(body,"&grant_type=refresh_token");CHKERRQ(ierr);

    ierr = PetscHTTPSRequest("POST","accounts.google.com/o/oauth2/token",NULL,"application/x-www-form-urlencoded",body,ssl,buff,sizeof(buff));CHKERRQ(ierr);
    ierr = PetscSSLDestroyContext(ctx);CHKERRQ(ierr);
    close(sock);

    ierr   = PetscPullJSONValue(buff,"access_token",access_token,tokensize,&found);CHKERRQ(ierr);
    if (!found) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Google drive did not return access_token");
  }
  PetscFunctionReturn(0);
}

#include <sys/stat.h>

/*@C
     PetscGoogleDriveUpload - Loads a file to the Google Drive

     Not collective, only the first process in the MPI_Comm uploads the file

  Input Parameters:
+   comm - MPI communicator
.   access_token - obtained with PetscGoogleDriveRefresh(), pass NULL to have PETSc generate one
-   filename - file to upload; if you upload multiple times it will have different names each time on Google Drive

  Options Database:
.  -google_refresh_token   XXX

  Usage Patterns:
    With PETSc option -google_refresh_token  XXX given
    PetscGoogleDriveUpload(comm,NULL,filename);        will upload file with no user interaction

    Without PETSc option -google_refresh_token XXX given
    PetscGoogleDriveUpload(comm,NULL,filename);        for first use will prompt user to authorize access to Google Drive with their browser

    With PETSc option -google_refresh_token  XXX given
    PetscGoogleDriveRefresh(comm,NULL,access_token,sizeof(access_token));
    PetscGoogleDriveUpload(comm,access_token,filename);

    With refresh token entered in some way by the user
    PetscGoogleDriveRefresh(comm,refresh_token,access_token,sizeof(access_token));
    PetscGoogleDriveUpload(comm,access_token,filename);

    PetscGoogleDriveAuthorize(comm,access_token,refresh_token,sizeof(access_token));
    PetscGoogleDriveUpload(comm,access_token,filename);

   Level: intermediate

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
                            "{");CHKERRQ(ierr);
    ierr = PetscPushJSONValue(body,"title",filename,len);CHKERRQ(ierr);
    ierr = PetscStrcat(body,",");CHKERRQ(ierr);
    ierr = PetscPushJSONValue(body,"mimeType","text.html",len);CHKERRQ(ierr);
    ierr = PetscStrcat(body,",");CHKERRQ(ierr);
    ierr = PetscPushJSONValue(body,"description","a file",len);CHKERRQ(ierr);
    ierr = PetscStrcat(body,"}\r\n\r\n"
                            "--foo_bar_baz\r\n"
                            "Content-Type: text/html\r\n\r\n");CHKERRQ(ierr);
    ierr = PetscStrlen(body,&blen);CHKERRQ(ierr);
    fd = fopen (filename, "r");
    if (!fd) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to open file: %s",filename);
    rd = fread (body+blen, sizeof (unsigned char), sb.st_size, fd);
    if (rd != (size_t) sb.st_size) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to read entire file: %s %d %d",filename,(int)rd,sb.st_size);
    fclose(fd);
    body[blen + rd] = 0;
    ierr = PetscStrcat(body,"\r\n\r\n"
                            "--foo_bar_baz\r\n");CHKERRQ(ierr);
    ierr = PetscSSLInitializeContext(&ctx);CHKERRQ(ierr);
    ierr = PetscHTTPSConnect("www.googleapis.com",443,ctx,&sock,&ssl);CHKERRQ(ierr);
    ierr = PetscHTTPSRequest("POST","www.googleapis.com/upload/drive/v2/files/",head,"multipart/related; boundary=\"foo_bar_baz\"",body,ssl,buff,sizeof(buff));CHKERRQ(ierr);
    ierr = PetscFree(body);CHKERRQ(ierr);
    ierr = PetscSSLDestroyContext(ctx);CHKERRQ(ierr);
    close(sock);
    ierr   = PetscStrstr(buff,"\"title\"",&title);CHKERRQ(ierr);
    if (!title) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Upload of file %s failed",filename);
  }
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_UNISTD_H)
#include <unistd.h>
#endif

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

   Notes:
    This call requires stdout and stdin access from process 0 on the MPI communicator

   You can run src/sys/webclient/tutorials/googleobtainrefreshtoken to get a refresh token and then in the future pass it to
   PETSc programs with -google_refresh_token XXX

   Level: intermediate

.seealso: PetscGoogleDriveRefresh(), PetscGoogleDriveUpload(), PetscURLShorten()

@*/
PetscErrorCode PetscGoogleDriveAuthorize(MPI_Comm comm,char access_token[],char refresh_token[],size_t tokensize)
{
  SSL_CTX        *ctx;
  SSL            *ssl;
  int            sock;
  PetscErrorCode ierr;
  char           buff[8*1024],*ptr,body[1024];
  PetscMPIInt    rank;
  size_t         len;
  PetscBool      found;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    if (!isatty(fileno(PETSC_STDOUT))) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Requires users input/output");
    ierr = PetscPrintf(comm,"Cut and paste the following into your browser:\n\n"
                            "https://accounts.google.com/o/oauth2/auth?"
                            "scope=https%%3A%%2F%%2Fwww.googleapis.com%%2Fauth%%2Fdrive.file&"
                            "redirect_uri=urn:ietf:wg:oauth:2.0:oob&"
                            "response_type=code&"
                            "client_id="
                            PETSC_GOOGLE_CLIENT_ID
                            "\n\n");CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"Paste the result here:");CHKERRQ(ierr);
    ptr  = fgets(buff, 1024, stdin);
    if (!ptr) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_FILE_READ, "Error reading from stdin: %d", errno);
    ierr = PetscStrlen(buff,&len);CHKERRQ(ierr);
    buff[len-1] = 0; /* remove carriage return at end of line */

    ierr = PetscSSLInitializeContext(&ctx);CHKERRQ(ierr);
    ierr = PetscHTTPSConnect("accounts.google.com",443,ctx,&sock,&ssl);CHKERRQ(ierr);
    ierr = PetscStrcpy(body,"code=");CHKERRQ(ierr);
    ierr = PetscStrcat(body,buff);CHKERRQ(ierr);
    ierr = PetscStrcat(body,"&client_id=");CHKERRQ(ierr);
    ierr = PetscStrcat(body,PETSC_GOOGLE_CLIENT_ID);CHKERRQ(ierr);
    ierr = PetscStrcat(body,"&client_secret=");CHKERRQ(ierr);
    ierr = PetscStrcat(body,PETSC_GOOGLE_CLIENT_ST);CHKERRQ(ierr);
    ierr = PetscStrcat(body,"&redirect_uri=urn:ietf:wg:oauth:2.0:oob&");CHKERRQ(ierr);
    ierr = PetscStrcat(body,"grant_type=authorization_code");CHKERRQ(ierr);

    ierr = PetscHTTPSRequest("POST","accounts.google.com/o/oauth2/token",NULL,"application/x-www-form-urlencoded",body,ssl,buff,sizeof(buff));CHKERRQ(ierr);
    ierr = PetscSSLDestroyContext(ctx);CHKERRQ(ierr);
    close(sock);

    ierr   = PetscPullJSONValue(buff,"access_token",access_token,tokensize,&found);CHKERRQ(ierr);
    if (!found) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Google drive did not return access_token");
    ierr   = PetscPullJSONValue(buff,"refresh_token",refresh_token,tokensize,&found);CHKERRQ(ierr);
    if (!found) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Google drive did not return refresh_token");

    ierr = PetscPrintf(comm,"Here is your Google refresh token, save it in a save place, in the future you can run PETSc\n");CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"programs with the option -google_refresh_token %s\n",refresh_token);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"to access Google Drive automatically\n");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


/*@C
     PetscURLShorten - Uses Google's service to get a short url for a long url

    Input Parameters:
+    url - long URL you want shortened
-    lenshorturl - length of buffer to contain short URL

    Output Parameter:
.    shorturl - the shortened URL

   Level: intermediate

.seealso: PetscGoogleDriveRefresh(), PetscGoogleDriveUpload(), PetscGoogleDriveAuthorize()
@*/
PetscErrorCode PetscURLShorten(const char url[],char shorturl[],size_t lenshorturl)
{
  SSL_CTX        *ctx;
  SSL            *ssl;
  int            sock;
  PetscErrorCode ierr;
  char           buff[1024],body[512],post[1024];
  PetscBool      found;

  PetscFunctionBegin;
  ierr = PetscSSLInitializeContext(&ctx);CHKERRQ(ierr);
  ierr = PetscHTTPSConnect("www.googleapis.com",443,ctx,&sock,&ssl);CHKERRQ(ierr);
  ierr = PetscStrcpy(body,"{");CHKERRQ(ierr);
  ierr = PetscPushJSONValue(body,"longUrl",url,sizeof(body)-2);CHKERRQ(ierr);
  ierr = PetscStrcat(body,"}");CHKERRQ(ierr);
  ierr = PetscSNPrintf(post,sizeof(post),"www.googleapis.com/urlshortener/v1/url?key=%s",PETSC_GOOGLE_API_KEY);CHKERRQ(ierr);
  ierr = PetscHTTPSRequest("POST",post,NULL,"application/json",body,ssl,buff,sizeof(buff));CHKERRQ(ierr);
  ierr = PetscSSLDestroyContext(ctx);CHKERRQ(ierr);
  close(sock);

  ierr   = PetscPullJSONValue(buff,"id",shorturl,lenshorturl,&found);CHKERRQ(ierr);
  if (!found) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Google drive did not return short URL");
  PetscFunctionReturn(0);
}

