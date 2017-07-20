python2:
	+$(MAKE) -C model
	+$(MAKE) -C descriptor
	f2py -c -m fmodules model.f90 descriptor/cutoffs.f90 descriptor/gaussian.f90 descriptor/zernike.f90 model/neuralnetwork.f90

python3:
	+$(MAKE) -C model
	+$(MAKE) -C descriptor
	f2py3 -c -m fmodules model.f90 descriptor/cutoffs.f90 descriptor/gaussian.f90 descriptor/zernike.f90 model/neuralnetwork.f90

AMPDIR:=$(shell pwd)

py2tests:
	rm -fr /tmp/py2_amptests
	mkdir -p /tmp/py2_amptests
	cd /tmp/py2_amptests && nosetests $(AMPDIR)/..

py3tests:
	rm -fr /tmp/py3_amptests
	mkdir -p /tmp/py3_amptests
	cd /tmp/py3_amptests && nosetests3 $(AMPDIR)/..
