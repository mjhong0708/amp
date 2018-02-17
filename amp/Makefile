python2:
	+$(MAKE) -C model
	+$(MAKE) -C descriptor
	f2py -c -m fmodules model.f90 descriptor/cutoffs.f90 descriptor/gaussian.f90 descriptor/zernike.f90 model/neuralnetwork.f90

python3:
	+$(MAKE) -C model
	+$(MAKE) -C descriptor
	f2py3 -c -m fmodules model.f90 descriptor/cutoffs.f90 descriptor/gaussian.f90 descriptor/zernike.f90 model/neuralnetwork.f90

py2tests:
	rm -fr ../tests/py2_amptests
	mkdir -p ../tests/py2_amptests
	cd ../tests/py2_amptests && nosetests -v ../../

py3tests:
	rm -fr ../tests/py3_amptests
	mkdir -p ../tests/py3_amptests
	cd ../tests/py3_amptests && nosetests3 -v ../../
