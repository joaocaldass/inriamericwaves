*** Some general remarks valid for all the notebooks ***

### STORAGE OF THE SOLUTION
 The main function for solving the models implemented (serre.splitSerre, kdv.kdv_fvfourier etc.) return the solution in the form of three (or two, in the case of KdV) arrays :
	* hall,uall : 2D array MxT, where M is the number of spatial point and T is the number of timeteps 
	* tall :  1D array of size T containing all the instants of simulation
 Therefore, the solution in the instant tall[i] is stored in hall[:,i] and uall[:,i]



### FUNCTIONS FOR PLOTTING ANIMATIONS

 For plotting animations of N functions in the same domain x, use the function
	plotAnimationNSolutions(N,x,u,t,xmin,xmax,ymin,ymax,lb,ylabel,location=0,savePath=None)
     - N : number of solutions
     - x : array of spatial points
     - t : array of instants
     - u = np.array([u1,u2,...]) : array of arrays containing the solutions. Each solution ui must have the shape MxT,
            where M = x.size and T = t.size
     - xmin,xmax : x interval for plotting
     - ymin,ymax : y interval for plotting
     - lb = ["lb1",...,"lbN"] : labels for the legend
     - ylabel = labelf for y axis
     - location (optional) : position of the legend (location = 0 as default gives an optimal position)
     - savePath (optional) : if not None, save the animation in a video specified by savePath

  If the functions have different domains x1,...xN, use the function
	plotAnimationNSolutionsDiffDomain(N,x,u,t,xmin,xmax,ymin,ymax,lb,ylabel,location=0,savePath=None) 
     - x = np.array([x1,x2,...])
     Nevertheless, all of the xi must have the same size and all the ui must have the same shape
    (if necessary, complete xi with values outside of [xmin,xmax] and ui with zeros).
