function[x] = MSA_InitialGuess(user,x)

%% The initial guess is the average of the boundary conditions
mx = user.mx; my = user.my;
for(j = 0:my-1)
    for(i = 0:mx-1)
        row=j*mx + i;
        a1 = ((j+1)*user.bottom(i+1)+(my-j+1)*user.top(i+1))/(my+2);
        a2 = ((i+1)*user.left(j+1)+(mx-i+1)*user.right(j+1))/(mx+2);
        x(row) = (a1 + a2)/2.0;
    end
end