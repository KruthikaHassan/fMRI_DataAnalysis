import numpy as np 


def shooting(X,y,lambda):
    tol = 1e-6
    found = 0
    beta = np.linalg.pinv((np.transpose(X)*X + 2*lambda)) * (np.transpose(X)*y)
    
    while(found == 0):
        betaold = beta
        for index in range(0,voxels):
        
            xi = X[:,index]
            
            #get residual excluding ith col
            yi = (y - X*beta) + xi*beta[index]           
            
            #calulate xi'*yi and see where it falls
            deltai = (np.transpose(xi)*yi) # 1 by 1 scalar
            
            if (deltai < -lambda):
                beta[index] = (deltai + lambda )/(np.transpose(xi)*xi)
            
            elif(deltai > lambda):
                beta[index] = (deltai - lambda )/(np.transpose(xi)*xi)
            
            else:
                beta[index] = 0;
            
        
        #check difference between beta and beta_old
        if(max(abs(beta - beta_old)) <= tol):
            found = 1
    

    return beta