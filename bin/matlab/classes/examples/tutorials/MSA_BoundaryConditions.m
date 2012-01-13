function[user] = MSA_BoundaryConditions(user)
%%% Sets the boundary conditions for the minimum surface area problem

maxits = 5;
tol = 1e-10;
b = -0.5;t = 0.5; l = -0.5; r = 0.5;
user.bedge = b; user.tedge = t; user.ledge = l; user.redge = r;
mx = user.mx; my = user.my;
bsize = user.mx+2; lsize = user.my+2;
tsize = user.mx+2; rsize = user.my+2;

user.bottom = PetscVec();
user.bottom.SetType('seq');
user.bottom.SetSizes(bsize,bsize);
user.top = user.bottom.Duplicate();
user.right = user.bottom.Duplicate();
user.left  = user.bottom.Duplicate();

hx = (r-l)/(mx+1); hy = (t-b)/(my+1);
user.hx = hx; user.hy = hy;

for(j=1:4)
    switch j
        case 1
            xt = l;yt = b;
            limit = bsize;
        case 2
            xt = l; yt = t;
            limit = tsize;
        case 3
            xt = l; yt = b;
            limit = lsize;
        case 4
            xt = r; yt = b;
            limit = rsize;
    end
    
    for(i = 1:limit)
        u1 = xt; u2 = -yt;
        for (k = 1:maxits)
            nf1 = u1 + u1*u2^2 - u1^3/3 - xt;
            nf2 = -u2 - u1^2*u2 + u2^3/3 - yt;
            fnorm = sqrt(nf1^2 + nf2^2);
            if (fnorm < tol)
                break;
            end
            jac =    [1+u2^2-u1^2, 2*u1*u2;
                      -2*u1*u2   , -1-u1^2+u2^2];
            det_jac = det(jac);
            u1 = u1 - (jac(2,2)*nf1 - jac(1,2)*nf2)/det_jac;
            u2 = u2 - (jac(1,1)*nf2 - jac(2,1)*nf1)/det_jac;
        end
        
        switch j
            case 1
                xt = xt + hx;
                user.bottom(i) = u1^2-u2^2;
            case 2
                xt = xt + hx;
                user.top(i) = u1^2 - u2^2;
            case 3
                yt = yt + hy;
                user.left(i) = u1^2 - u2^2;
            case 4
                yt = yt + hy;
                user.right(i) = u1^2 - u2^2;
        end
    end
end        