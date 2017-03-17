fortran:
	+$(MAKE) -C model
	+$(MAKE) -C descriptor
	f2py -c -m fmodules model.f90 descriptor/cutoffs.f90 descriptor/gaussian.f90 descriptor/zernike.f90 model/neuralnetwork.f90

AMPDIR:=$(shell pwd)
tests:
	rm -fr /tmp/amptests
	mkdir -p /tmp/amptests
	cd /tmp/amptests && nosetests --with-timer $(AMPDIR)/..
