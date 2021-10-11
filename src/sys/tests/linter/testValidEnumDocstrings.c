#include <petscsys.h>

/*E
  WellFormedEnum - Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor

$ LOREM - A lorem
$ IPSUM - An ipsum
$ DOLOR - A dolor

  Level: advanced

.seealso: Lorem
E*/
typedef enum {
  LOREM,
  IPSUM,
  DOLOR
} WellFormedEnum;

/*E
  IllFormedEnum -

$ SIT- A sit
$ CONSECTETUR - A consectetur
 $ AMET - An amet
$ADAPISCING - an adapiscing
Level: advanced
*/
typedef enum {
  SIT,
  AMET,
  CONSECTETUR,
  ADAPISCING
} IllFormedEnum;

/*E
  bdSpllingenUm - Lorem ipsum dolor

  Not Collective

$ FOO - a foo

  Notes:
  a note

.seealso:                         IllFormedEnum,WellFormedEnum,WellFormedEnum,WellFormedEnum,BadSpellingEnum,BadSpellingEnum
*/
typedef enum {
  FOO
} BadSpellingEnum;
