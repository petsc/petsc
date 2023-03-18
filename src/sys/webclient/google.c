
#include <petscwebclient.h>
PETSC_PRAGMA_DIAGNOSTIC_IGNORED_BEGIN("-Wdeprecated-declarations")

/*
   These variables identify the code as a PETSc application to Google.

   See -   https://stackoverflow.com/questions/4616553/using-oauth-in-free-open-source-software
   Users can get their own application IDs - https://code.google.com/p/google-apps-manager/wiki/GettingAnOAuthConsoleKey

*/
#define PETSC_GOOGLE_CLIENT_ID "521429262559-i19i57eek8tnt9ftpp4p91rcl0bo9ag5.apps.googleusercontent.com"
#define PETSC_GOOGLE_CLIENT_ST "vOds_A71I3_S_aHMq_kZAI0t"
#define PETSC_GOOGLE_API_KEY   "AIzaSyDRZsOcySpWVzsUvIBL2UG3J2tcg-MXbyk"

/*@C
     PetscGoogleDriveRefresh - Get a new authorization token for accessing Google drive from PETSc from a refresh token

   Not Collective, only the first process in the `MPI_Comm` does anything

   Input Parameters:
+   comm - MPI communicator
.   refresh token - obtained with `PetscGoogleDriveAuthorize()`, if NULL PETSc will first look for one in the options data
                    if not found it will call `PetscGoogleDriveAuthorize()`
-   tokensize - size of the output string access_token

   Output Parameter:
.   access_token - token that can be passed to `PetscGoogleDriveUpload()`

   Options Database Key:
.  -google_refresh_token XXX - where XXX was obtained from `PetscGoogleDriveAuthorize()`

   Level: intermediate

.seealso: `PetscURLShorten()`, `PetscGoogleDriveAuthorize()`, `PetscGoogleDriveUpload()`
@*/
PetscErrorCode PetscGoogleDriveRefresh(MPI_Comm comm, const char refresh_token[], char access_token[], size_t tokensize)
{
  SSL_CTX    *ctx;
  SSL        *ssl;
  int         sock;
  char        buff[8 * 1024], body[1024];
  PetscMPIInt rank;
  char       *refreshtoken = (char *)refresh_token;
  PetscBool   found;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  if (rank == 0) {
    if (!refresh_token) {
      PetscBool set;
      PetscCall(PetscMalloc1(512, &refreshtoken));
      PetscCall(PetscOptionsGetString(NULL, NULL, "-google_refresh_token", refreshtoken, sizeof(refreshtoken), &set));
      if (!set) {
        PetscCall(PetscGoogleDriveAuthorize(comm, access_token, refreshtoken, 512 * sizeof(char)));
        PetscCall(PetscFree(refreshtoken));
        PetscFunctionReturn(PETSC_SUCCESS);
      }
    }
    PetscCall(PetscSSLInitializeContext(&ctx));
    PetscCall(PetscHTTPSConnect("accounts.google.com", 443, ctx, &sock, &ssl));
    PetscCall(PetscStrncpy(body, "client_id=", sizeof(body)));
    PetscCall(PetscStrlcat(body, PETSC_GOOGLE_CLIENT_ID, sizeof(body)));
    PetscCall(PetscStrlcat(body, "&client_secret=", sizeof(body)));
    PetscCall(PetscStrlcat(body, PETSC_GOOGLE_CLIENT_ST, sizeof(body)));
    PetscCall(PetscStrlcat(body, "&refresh_token=", sizeof(body)));
    PetscCall(PetscStrlcat(body, refreshtoken, sizeof(body)));
    if (!refresh_token) PetscCall(PetscFree(refreshtoken));
    PetscCall(PetscStrlcat(body, "&grant_type=refresh_token", sizeof(body)));

    PetscCall(PetscHTTPSRequest("POST", "accounts.google.com/o/oauth2/token", NULL, "application/x-www-form-urlencoded", body, ssl, buff, sizeof(buff)));
    PetscCall(PetscSSLDestroyContext(ctx));
    close(sock);

    PetscCall(PetscPullJSONValue(buff, "access_token", access_token, tokensize, &found));
    PetscCheck(found, PETSC_COMM_SELF, PETSC_ERR_LIB, "Google drive did not return access_token");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#include <sys/stat.h>

/*@C
     PetscGoogleDriveUpload - Loads a file to the Google Drive

     Not Collective, only the first process in the `MPI_Comm` uploads the file

  Input Parameters:
+   comm - MPI communicator
.   access_token - obtained with PetscGoogleDriveRefresh(), pass `NULL` to have PETSc generate one
-   filename - file to upload; if you upload multiple times it will have different names each time on Google Drive

  Options Database Key:
.  -google_refresh_token XXX - pass the access token for the operation

  Usage Patterns:
.vb
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
.ve

   Level: intermediate

.seealso: `PetscURLShorten()`, `PetscGoogleDriveAuthorize()`, `PetscGoogleDriveRefresh()`
@*/
PetscErrorCode PetscGoogleDriveUpload(MPI_Comm comm, const char access_token[], const char filename[])
{
  SSL_CTX    *ctx;
  SSL        *ssl;
  int         sock;
  char        head[1024], buff[8 * 1024], *body, *title;
  PetscMPIInt rank;
  struct stat sb;
  size_t      len, blen, rd;
  FILE       *fd;
  int         err;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  if (rank == 0) {
    PetscCall(PetscStrncpy(head, "Authorization: Bearer ", sizeof(head)));
    PetscCall(PetscStrlcat(head, access_token, sizeof(head)));
    PetscCall(PetscStrlcat(head, "\r\n", sizeof(head)));
    PetscCall(PetscStrlcat(head, "uploadType: multipart\r\n", sizeof(head)));

    err = stat(filename, &sb);
    PetscCheck(!err, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Unable to stat file: %s", filename);
    len = 1024 + sb.st_size;
    PetscCall(PetscMalloc1(len, &body));
    PetscCall(PetscStrncpy(body,
                           "--foo_bar_baz\r\n"
                           "Content-Type: application/json\r\n\r\n"
                           "{",
                           sizeof(body)));
    PetscCall(PetscPushJSONValue(body, "title", filename, len));
    PetscCall(PetscStrlcat(body, ",", sizeof(body)));
    PetscCall(PetscPushJSONValue(body, "mimeType", "text.html", len));
    PetscCall(PetscStrlcat(body, ",", sizeof(body)));
    PetscCall(PetscPushJSONValue(body, "description", "a file", len));
    PetscCall(PetscStrlcat(body,
                           "}\r\n\r\n"
                           "--foo_bar_baz\r\n"
                           "Content-Type: text/html\r\n\r\n",
                           sizeof(body)));
    PetscCall(PetscStrlen(body, &blen));
    fd = fopen(filename, "r");
    PetscCheck(fd, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Unable to open file: %s", filename);
    rd = fread(body + blen, sizeof(unsigned char), sb.st_size, fd);
    PetscCheck(rd == (size_t)sb.st_size, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Unable to read entire file: %s %d %d", filename, (int)rd, (int)sb.st_size);
    fclose(fd);
    body[blen + rd] = 0;
    PetscCall(PetscStrlcat(body,
                           "\r\n\r\n"
                           "--foo_bar_baz\r\n",
                           sizeof(body)));
    PetscCall(PetscSSLInitializeContext(&ctx));
    PetscCall(PetscHTTPSConnect("www.googleapis.com", 443, ctx, &sock, &ssl));
    PetscCall(PetscHTTPSRequest("POST", "www.googleapis.com/upload/drive/v2/files/", head, "multipart/related; boundary=\"foo_bar_baz\"", body, ssl, buff, sizeof(buff)));
    PetscCall(PetscFree(body));
    PetscCall(PetscSSLDestroyContext(ctx));
    close(sock);
    PetscCall(PetscStrstr(buff, "\"title\"", &title));
    PetscCheck(title, PETSC_COMM_SELF, PETSC_ERR_LIB, "Upload of file %s failed", filename);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if defined(PETSC_HAVE_UNISTD_H)
  #include <unistd.h>
#endif

/*@C
     PetscGoogleDriveAuthorize - Get authorization and refresh token for accessing Google drive from PETSc

   Not Collective, only the first process in `MPI_Comm` does anything

   Input Parameters:
+  comm - the MPI communicator
-  tokensize - size of the token arrays

   Output Parameters:
+  access_token - can be used with `PetscGoogleDriveUpload()` for this one session
-  refresh_token - can be used for ever to obtain new access_tokens with `PetscGoogleDriveRefresh()`, guard this like a password
                   it gives access to your Google Drive

   Level: intermediate

   Notes:
    This call requires `stdout` and `stdin` access from process 0 on the MPI communicator

   You can run src/sys/webclient/tutorials/googleobtainrefreshtoken to get a refresh token and then in the future pass it to
   PETSc programs with `-google_refresh_token XXX`

.seealso: `PetscGoogleDriveRefresh()`, `PetscGoogleDriveUpload()`, `PetscURLShorten()`
@*/
PetscErrorCode PetscGoogleDriveAuthorize(MPI_Comm comm, char access_token[], char refresh_token[], size_t tokensize)
{
  SSL_CTX    *ctx;
  SSL        *ssl;
  int         sock;
  char        buff[8 * 1024], *ptr, body[1024];
  PetscMPIInt rank;
  size_t      len;
  PetscBool   found;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  if (rank == 0) {
    PetscCheck(isatty(fileno(PETSC_STDOUT)), PETSC_COMM_SELF, PETSC_ERR_USER, "Requires users input/output");
    PetscCall(PetscPrintf(comm, "Cut and paste the following into your browser:\n\n"
                                "https://accounts.google.com/o/oauth2/auth?"
                                "scope=https%%3A%%2F%%2Fwww.googleapis.com%%2Fauth%%2Fdrive.file&"
                                "redirect_uri=urn:ietf:wg:oauth:2.0:oob&"
                                "response_type=code&"
                                "client_id=" PETSC_GOOGLE_CLIENT_ID "\n\n"));
    PetscCall(PetscPrintf(comm, "Paste the result here:"));
    ptr = fgets(buff, 1024, stdin);
    PetscCheck(ptr, PETSC_COMM_SELF, PETSC_ERR_FILE_READ, "Error reading from stdin: %d", errno);
    PetscCall(PetscStrlen(buff, &len));
    buff[len - 1] = 0; /* remove carriage return at end of line */

    PetscCall(PetscSSLInitializeContext(&ctx));
    PetscCall(PetscHTTPSConnect("accounts.google.com", 443, ctx, &sock, &ssl));
    PetscCall(PetscStrncpy(body, "code=", sizeof(body)));
    PetscCall(PetscStrlcat(body, buff, sizeof(body)));
    PetscCall(PetscStrlcat(body, "&client_id=", sizeof(body)));
    PetscCall(PetscStrlcat(body, PETSC_GOOGLE_CLIENT_ID, sizeof(body)));
    PetscCall(PetscStrlcat(body, "&client_secret=", sizeof(body)));
    PetscCall(PetscStrlcat(body, PETSC_GOOGLE_CLIENT_ST, sizeof(body)));
    PetscCall(PetscStrlcat(body, "&redirect_uri=urn:ietf:wg:oauth:2.0:oob&", sizeof(body)));
    PetscCall(PetscStrlcat(body, "grant_type=authorization_code", sizeof(body)));

    PetscCall(PetscHTTPSRequest("POST", "accounts.google.com/o/oauth2/token", NULL, "application/x-www-form-urlencoded", body, ssl, buff, sizeof(buff)));
    PetscCall(PetscSSLDestroyContext(ctx));
    close(sock);

    PetscCall(PetscPullJSONValue(buff, "access_token", access_token, tokensize, &found));
    PetscCheck(found, PETSC_COMM_SELF, PETSC_ERR_LIB, "Google drive did not return access_token");
    PetscCall(PetscPullJSONValue(buff, "refresh_token", refresh_token, tokensize, &found));
    PetscCheck(found, PETSC_COMM_SELF, PETSC_ERR_LIB, "Google drive did not return refresh_token");

    PetscCall(PetscPrintf(comm, "Here is your Google refresh token, save it in a save place, in the future you can run PETSc\n"));
    PetscCall(PetscPrintf(comm, "programs with the option -google_refresh_token %s\n", refresh_token));
    PetscCall(PetscPrintf(comm, "to access Google Drive automatically\n"));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
     PetscURLShorten - Uses Google's service to get a short url for a long url

    Input Parameters:
+    url - long URL you want shortened
-    lenshorturl - length of buffer to contain short URL

    Output Parameter:
.    shorturl - the shortened URL

   Level: intermediate

   Note:
   Google no longer provides this service so this routine will no longer function

.seealso: `PetscGoogleDriveRefresh()`, `PetscGoogleDriveUpload()`, `PetscGoogleDriveAuthorize()`
@*/
PetscErrorCode PetscURLShorten(const char url[], char shorturl[], size_t lenshorturl)
{
  SSL_CTX  *ctx;
  SSL      *ssl;
  int       sock;
  char      buff[1024], body[512], post[1024];
  PetscBool found;

  PetscFunctionBegin;
  PetscCall(PetscSSLInitializeContext(&ctx));
  PetscCall(PetscHTTPSConnect("www.googleapis.com", 443, ctx, &sock, &ssl));
  PetscCall(PetscStrncpy(body, "{", sizeof(body)));
  PetscCall(PetscPushJSONValue(body, "longUrl", url, sizeof(body) - 2));
  PetscCall(PetscStrlcat(body, "}", sizeof(body)));
  PetscCall(PetscSNPrintf(post, sizeof(post), "www.googleapis.com/urlshortener/v1/url?key=%s", PETSC_GOOGLE_API_KEY));
  PetscCall(PetscHTTPSRequest("POST", post, NULL, "application/json", body, ssl, buff, sizeof(buff)));
  PetscCall(PetscSSLDestroyContext(ctx));
  close(sock);

  PetscCall(PetscPullJSONValue(buff, "id", shorturl, lenshorturl, &found));
  PetscCheck(found, PETSC_COMM_SELF, PETSC_ERR_LIB, "Google drive did not return short URL");
  PetscFunctionReturn(PETSC_SUCCESS);
}
