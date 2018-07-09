FOLDER Serre

* NOTEBOOKS :
	- serre.ipynb : (--> serre.py)
		:: Finite Volume solver for the advection step : serre.RK4
		:: Finite Difference solver for the dispersive step : serre.EFDSolverFM4 (Order 4, scheme proposed by Fabien Marche);
								      serre.EFDSolverFM4Bottom(...,eta=eta) for non horizontal bottom
								      (EFDSolverFM4 = EFDSolverFM4Bottom(eta=0))
		:: Splitting solver : serre.splitSerre; calls the above functions
		:: NSWE solver : serre.NSWE; calls only the FV solvers
		:: functions to impose boundary conditions
		:: Validation of the Serre model with the analytical solution (cnoidal and solitary)
	- cnoidal.ipynb : (--> cnoidal.py)
		:: Implementation of the analytical solutions for the Serre equations
	- serre_validation_cnoidal.ipynb/serre_validation_solitary.ipynb : more detailed validations
	- serre_NonHorizontal
		:: validation of the NSWE and Serre with a non-horizontal bottom
* PYTHON SCRIPTS :
	- serre.py
	- cnoidal.py

* Updates Gaspard :
  - added the notebook (and python files associated) serre_DTBC and serre_DDM
    - serre_DTBC : contains a modified version of serre file, to focus on the
      DTBC we worked on with Antoine. This is meant to study their efficiency
      when used as boundary conditions.
    - serre_DDM : to study the efficiency of the DTBC when used as interface
      conditions. Note that this is not a coupling between Serre and NSWE but a
      stricly domain decomposition for the Serre equations.
  - saved some videos to present the results in different cases. The names
    should be pretty explicit.
  - added some python scripts (test_roots, test_Y, convolution and Zt_tools)
    that are meant either to build the coefficients in our DTBC or to study
    their behaviour.
