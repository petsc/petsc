
/*
   Obtained from https://opensource.apple.com/source/Libc/Libc-1353.41.1/stdio/FreeBSD/mktemp.c.auto.html

   The only line changed is mkdirat() to mkdir() because mkdirat() fails under valgrind
*/
#include <sys/cdefs.h>

#include <assert.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <unistd.h>

#define ALLOWED_MKOSTEMP_FLAGS (O_APPEND | O_SHLOCK | O_EXLOCK | O_CLOEXEC)

char *_mktemp(char *);

typedef enum {
        FTPP_DONE, FTPP_TRY_NEXT, FTPP_ERROR
} find_temp_path_progress_t;

/* A contract for actions that find_temp_path performs for every path from
 * the template.
 *
 * If the desired path was found, set result and return FTPP_DONE.
 * If an IO/FS error ocurred, set errno and return FTPP_ERROR.
 * Otherwise return FTPP_TRY_NEXT.
 */
typedef find_temp_path_progress_t (*find_temp_path_action_t)(
                int dfd, char *path, void *ctx, void *result);

static int find_temp_path(int dfd, char *path, int slen, int stat_base_dir,
                find_temp_path_action_t action, void *action_ctx, void *action_result);

static find_temp_path_progress_t
_mkdtemp_action(int dfd, char *path, void *ctx __unused, void *result __unused)
{
        if (mkdir(path, 0700) == 0)
                return FTPP_DONE;
        return (errno == EEXIST) ?
                        FTPP_TRY_NEXT :
                        FTPP_ERROR; // errno is set already
}

static const char padchar[] =
"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";


static int
find_temp_path(int dfd, char *path, int slen, int stat_base_dir,
                find_temp_path_action_t action, void *action_ctx, void *action_result)
 {
        char *start, *trv, *suffp, *carryp;
        const char *pad;
        struct stat sbuf;
        int rval;
        uint32_t rand;
        char carrybuf[MAXPATHLEN];

        if (slen < 0) {
                errno = EINVAL;
                return (0);
        }

        for (trv = path; *trv != '\0'; ++trv)
                ;
        if (trv - path >= MAXPATHLEN) {
                errno = ENAMETOOLONG;
                return (0);
        }
        trv -= slen;
        suffp = trv;
        --trv;
        if (trv < path || NULL != strchr(suffp, '/')) {
                errno = EINVAL;
                return (0);
        }

        /* Fill space with random characters */
        while (trv >= path && *trv == 'X') {
                rand = arc4random_uniform(sizeof(padchar) - 1);
                *trv-- = padchar[rand];
        }
        start = trv + 1;

        /* save first combination of random characters */
        memcpy(carrybuf, start, suffp - start);

        /*
         * check the target directory.
         */
        if (stat_base_dir) {
                for (; trv > path; --trv) {
                        if (*trv == '/') {
                                *trv = '\0';
                                rval = fstatat(dfd, path, &sbuf, 0);
                                *trv = '/';
                                if (rval != 0)
                                        return (0);
                                if (!S_ISDIR(sbuf.st_mode)) {
                                        errno = ENOTDIR;
                                        return (0);
                                }
                                break;
                        }
                }
        }

        for (;;) {
                switch (action(dfd, path, action_ctx, action_result)) {
                case FTPP_DONE:
                        return (1);
                case FTPP_ERROR:
                        return (0); // errno must be set by the action
                default:
                        ; // FTPP_TRY_NEXT, fall-through
                }

                /* If we have a collision, cycle through the space of filenames */
                for (trv = start, carryp = carrybuf;;) {
                        /* have we tried all possible permutations? */
                        if (trv == suffp) {
                                /* yes - exit with EEXIST */
                                errno = EEXIST;
                                return (0);
                        }
                        pad = strchr(padchar, *trv);
                        if (pad == NULL) {
                                /* this should never happen */
                                errno = EIO;
                                return (0);
                        }
                        /* increment character */
                        *trv = (*++pad == '\0') ? padchar[0] : *pad;
                        /* carry to next position? */
                        if (*trv == *carryp) {
                                /* increment position and loop */
                                ++trv;
                                ++carryp;
                        } else {
                                /* try with new name */
                                break;
                        }
                }
        }
        /*NOTREACHED*/
}

char *
mkdtemp(char *path)
{
        return (find_temp_path(AT_FDCWD, path, 0, 1, _mkdtemp_action, NULL, NULL) ?
                        path : (char *)NULL);
}
