
/* This file contains routines written by Bob Olson
   for the VRML interface of the PETSc demo at SC96 */

#include "user.h"

void dump_angle_vrml(float angle)
{
    FILE *fp;
    fp = fopen("angle.wrl", "w");

    fprintf(fp, "\n\
#VRML V1.0 ascii\n\
    Separator {\n\
	PickStyle {\n\
            style BOUNDING_BOX\n\
	}\n\
	Scale { scaleFactor .3 .3 .3 }\n\
	Material {\n\
            diffuseColor .58 .855 .44\n\
	}\n\
	Text3 {\n\
	    string \"Angle of attack: %f\"\n\
	}\n\
    }\n\
", angle);
    fclose(fp);
}
/* --------------------------------------------------------------- */

void MonitorDumpIter(int iter)
{
    FILE *fp;
    fp = fopen("iter.wrl", "w");

    fprintf(fp, "\n\
#VRML V1.0 ascii\n\
    Separator {\n\
	PickStyle {\n\
            style BOUNDING_BOX\n\
	}\n\
	Scale { scaleFactor .3 .3 .3 }\n\
	Material {\n\
            diffuseColor .58 .855 .44\n\
	}\n\
	Text3 {\n\
	    string \"Iteration: %d\"\n\
	}\n\
    }\n\
", iter);
    fclose(fp);
}
