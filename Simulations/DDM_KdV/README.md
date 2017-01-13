FOLDER DDM_KdV

- Contains the notebooks used for the content of the paper
- Contains the file with the results of the optimziation tests


- Files with results : the results of each tests is stored in a library. It is a simple library indexed by the variable
                     parameter (let's say t0).
                     For example, the test with the parameter t0 has the format
                     tests[str(t0)] =    (array([cL,niterCV,errorCV] with growing cL),
                                          array([cL,niterCV,errorCV] with growing niterCV),
                                          [cL,niterCV] for min niterCV,
                                          [cL,niterCV] for max niterCV,
                                          [cL,errorCV] for min errorCV,
                                          [cL,errorCV] for max errorCV,
                                          [cL,niterCV] for min niterCV for negative coefficients,
                                          [cL,niterCV] for min niterCV for positive coefficients)
  * The files are :
	** resumePositive_optimDt2em2DxVar.json/resumeNegative_optimDt2em2DxVar.json : dt=2e-2 fixed, variable dx
	** resumePositive_optimDtv8em2DxVar.json/resumeNegative_optimDt8em2DxVar.json : dt=8e-2 fixed, variable dx
	** resumePositive_optimDtVarDx250.json/resumeNegative_optimDtVarDx250.json : variable dt, dx=12/250 fixed
	** resumePositive_optimDtVarDx500.json/resumeNegative_optimDtVar500.json : variable dt, dx=12/500 fixed
 
 * The files can be readen using the function besseTBC.loadTests(filename) (returns the library)

