#if !defined(__TAO_VERSION_H)
#define __TAO_VERSION_H

/* ========================================================================== */
#define TAO_VERSION_RELEASE  0 /*  1 if official release */
/* 
   Current TAO version number and release date
*/
#define TAO_VERSION_NUMBER "TAO Version 2.1"
#define TAO_VERSION_MAJOR    2
#define TAO_VERSION_MINOR    1
#define TAO_VERSION_SUBMINOR 0
#define TAO_PATCH_LEVEL      0
#define TAO_VERSION_(MAJOR,MINOR,SUBMINOR) \
    ((TAO_VERSION_MAJOR == (MAJOR)) &&       \
    (TAO_VERSION_MINOR == (MINOR)) &&       \
     (TAO_VERSION_SUBMINOR == (SUBMINOR)))
#define TAO_VERSION_DATE     "Jun 28, 2012"
#define TAO_AUTHOR_INFO      "The TAO Team:\
 Todd Munson, Jason Sarich, Stefan Wild\n\
Bug reports, questions: tao-comments@mcs.anl.gov\n\
Web page: http://www.mcs.anl.gov/tao/\n"

#endif
