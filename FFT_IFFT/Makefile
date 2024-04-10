all:
	LDSHARED="icpx -shared -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -liomp5 -ldl -lpthread" CC=icpx CXX=icpx python3 setup.py build_ext --inplace

clean:
	rm *.so
	rm -rf build
	rm fft.cpp