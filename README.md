DustFilaments
=====

DustFilaments is a code for painting filaments in the Celestial Sphere, to generate a full sky map of the Thermal Dust emission at millimeter frequencies by integrating a population of 3D filaments.

Available maps
--------------

Our maps are located at ...
The units of our maps are uK Thermo units.
We create full sky maps of T,Q,U emission at 20, 27, 39, 93, 145, 225, 280 GHz. Also we include maps at 217 and 353 GHz. We include maps with Q,U large scale filled by the Planck template, as well as the calibrated map without this large-scale filled and before filtering by a high-pass filter. This is the raw Q,U map directly from our code, but calibrated to uK thermo units. 

The code
--------

The code consists of 3 main methods:
* **get_MagField**: this function will return a magnetic field cube. The random isotropic component can be made deterministic by using the same seed again. Optionally you can provide a precalculated cube for e.g. the large scale Galactic component. This must be as a numpy array saved as a npz file, with a key label `Hcube`. Also note that the convention for the entire code is that a magnetic field cube has shape `[Npix,Npix,Npix,3]` and the indices ordering is `[index_z,index_y,index_x,:]`. You can also skip this `get_MagField` function all together if you want, if you want to calculate a magnetic field cube on your own, provided that these convention are followed. 

* **get_FilPop**: this function will create a filament population and has the magnetic field cube as input. It will create a random population with a given seed. Remember to always use the same magnetic field and population seed if you want to make two or more runs of the code if you are running too many frequency channels. 

* **Paint_Filament** this is the main method of the code, and it will paint a single filament into a healpix map provided as input. The healpix map is updated in place. The `test` directory has an example of a script that will run the code in a cluster using mpi4py with mpiexec.

Install
-------

**Requirements** : Standard python modules
* numpy
* healpy
* mpi4py
* yaml

Also, the healpix c++ library is needed for compiling the filament paint code.
To install, run 
```
python setup.py install
```
After install, you should also install [this](https://github.com/huffenberger-cosmology/magnetic-field-power-spectrum) code that generates the isotropic random magnetic field box. This code is needed if you use the *get_MagField* function.

Using the code
--------------
