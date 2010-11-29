function [mat,err] = PetscMatCreateSeqAIJ(mx)

[err,mx,m] = calllib('libpetsc', 'MatCreateSeqAIJFromMatlab',mx',0);PetscCHKERRQ(err);
mat  = PetscMat(m,'pobj');

