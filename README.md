# Spatial Maths for Python

[![A Python Robotics Package](https://raw.githubusercontent.com/petercorke/robotics-toolbox-python/master/.github/svg/py_collection.min.svg)](https://github.com/petercorke/robotics-toolbox-python)
[![QUT Centre for Robotics Open Source](https://github.com/qcr/qcr.github.io/raw/master/misc/badge.svg)](https://qcr.github.io)

[![PyPI version](https://badge.fury.io/py/spatialmath-python.svg)](https://badge.fury.io/py/spatialmath-python)
[![Anaconda version](https://anaconda.org/conda-forge/spatialmath-python/badges/version.svg)](https://anaconda.org/conda-forge/spatialmath-python)
![Python Version](https://img.shields.io/pypi/pyversions/spatialmath-python.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[![Build Status](https://github.com/rai-opensource/spatialmath-python/actions/workflows/master.yml/badge.svg?branch=master)](https://github.com/rai-opensource/spatialmath-python/actions/workflows/master.yml?query=workflow%3Abuild+branch%3Amaster)
[![Coverage](https://codecov.io/github/rai-opensource/spatialmath-python/graph/badge.svg?token=W15FGBA059)](https://codecov.io/github/rai-opensource/spatialmath-python)
[![PyPI - Downloads](https://img.shields.io/pypi/dw/spatialmath-python)](https://pypistats.org/packages/spatialmath-python)
[![GitHub stars](https://img.shields.io/github/stars/rai-opensource/spatialmath-python.svg?style=social&label=Star)](https://GitHub.com/rai-opensource/spatialmath-python/stargazers/)

<table style="border:0px">
<tr style="border:0px">
<td style="border:0px">
<img src="https://github.com/rai-opensource/spatialmath-python/raw/master/docs/figs/CartesianSnakes_LogoW.png" width="200"></td>
<td style="border:0px">
A Python implementation of the <a href="https://github.com/petercorke/spatial-math">Spatial Math Toolbox for MATLAB<sup>&reg;</sup></a>
<ul>
<li><a href="https://github.com/rai-opensource/spatialmath-python">GitHub repository </a></li>
<li><a href="https://https://spatialmath-python.rai-inst.com">Documentation</a></li>
<li><a href=https://github.com/rai-opensource/spatialmath-python/discussions/categories/changes>Recent changes</a>
<li><a href="https://github.com/rai-opensource/spatialmath-python/wiki">Wiki (examples and details)</a></li>
<li><a href="installation#">Installation</a></li>
</ul>
</td>
</tr>
</table>


Spatial mathematics capability underpins all of robotics and robotic vision where we need to describe the position, orientation or pose of objects in 2D or 3D spaces.



# What it does

The package provides classes to represent pose and orientation in 3D and 2D
space:

| Represents   | in 3D            |   in 2D  |
| ------------ | ---------------- | -------- |
| pose         | ``SE3`` ``Twist3`` ``UnitDualQuaternion``   |   ``SE2`` ``Twist2`` |
| orientation  | ``SO3`` ``UnitQuaternion`` |            ``SO2``  |


More specifically:

 * `SE3` matrices belonging to the group $\mathbf{SE}(3)$ for position and orientation (pose) in 3-dimensions
 * `SO3` matrices belonging to the group $\mathbf{SO}(3)$ for orientation in 3-dimensions
 *  `UnitQuaternion` belonging to the group $\mathbf{S}^3$ for orientation in 3-dimensions
 * `Twist3` vectors belonging to the group $\mathbf{se}(3)$ for pose in 3-dimensions
 * `UnitDualQuaternion` maps to the group $\mathbf{SE}(3)$ for position and orientation (pose) in 3-dimensions
 * `SE2` matrices belonging to the group $\mathbf{SE}(2)$ for position and orientation (pose) in 2-dimensions
 * `SO2` matrices belonging to the group $\mathbf{SO}(2)$ for orientation in 2-dimensions
 * `Twist2` vectors belonging to the group $\mathbf{se}(2)$ for pose in 2-dimensions


These classes provide convenience and type safety, as well as methods and overloaded operators to support:

 * composition, using the `*` operator
 * point transformation, using the `*` operator
 * exponent, using the `**` operator
 * normalization
 * inversion
 * connection to the Lie algebra via matrix exponential and logarithm operations
 * conversion of orientation to/from Euler angles, roll-pitch-yaw angles and angle-axis forms.
 * list operations such as append, insert and get

These are layered over a set of base functions that perform many of the same operations but represent data explicitly in terms of `numpy` arrays.

The class, method and functions names largely mirror those of the MATLAB toolboxes, and the semantics are quite similar.

![trplot](https://github.com/rai-opensource/spatialmath-python/raw/master/docs/figs/fig1.png)

![animation video](./docs/figs/animate.gif)

# Citing

Check out our ICRA 2021 paper on [IEEE Xplore](https://ieeexplore.ieee.org/document/9561366) or get the PDF from [Peter's website](https://bit.ly/icra_rtb).  This describes the [Robotics Toolbox for Python](https://github.com/petercorke/robotics-toolbox-python) as well Spatial Maths.

If the toolbox helped you in your research, please cite

```
@inproceedings{rtb,
  title={Not your grandmother’s toolbox--the Robotics Toolbox reinvented for Python},
  author={Corke, Peter and Haviland, Jesse},
  booktitle={2021 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={11357--11363},
  year={2021},
  organization={IEEE}
}
```

<br>

<a id='6'></a>

## Using the Toolbox in your Open Source Code?

If you are using the Toolbox in your open source code, feel free to add our badge to your readme!

[![Powered by the Spatial Math Toolbox](https://github.com/rai-opensource/spatialmath-python/raw/master/.github/svg/sm_powered.min.svg)](https://github.com/rai-opensource/spatialmath-python)

Simply copy the following

```
[![Powered by the Spatial Math Toolbox](https://github.com/rai-opensource/spatialmath-python/raw/master/.github/svg/sm_powered.min.svg)](https://github.com/rai-opensource/spatialmath-python)
```


# Installation

## Using pip

Install a snapshot from PyPI

```
pip install spatialmath-python
```

Note that if you are using ROS2, you may run into version conflicts when using `rosdep`, particularly
concerning `matplotlib`. If this happens, you can enable optional version pinning with

```
pip install spatialmath-python[ros-humble]
```

## From GitHub

Install the current code base from GitHub and pip install a link to that cloned copy

```
git clone https://github.com/rai-opensource/spatialmath-python.git
cd spatialmath-python
pip install -e .
# Optional: if you would like to contribute and commit code changes to the repository,
# pre-commit install
```

## Dependencies

`numpy`, `scipy`, `matplotlib`, `ffmpeg` (if rendering animations as a movie)

# Examples


## High-level classes

These classes abstract the low-level numpy arrays into objects that obey the rules associated with the mathematical groups SO(2), SE(2), SO(3), SE(3) as well as twists and quaternions.

Using classes ensures type safety, for example it stops us mixing a 2D homogeneous transformation with a 3D rotation matrix -- both of which are 3x3 matrices.  It also ensures that the internal matrix representation is always a valid member of the relevant group.

For example, to create an object representing a rotation of 0.3 radians about the x-axis is simply

```python
>>> from spatialmath import SO3, SE3
>>> R1 = SO3.Rx(0.3)
>>> R1
   1         0         0
   0         0.955336 -0.29552
   0         0.29552   0.955336
```
while a rotation of 30 deg about the z-axis is

```python
>>> R2 = SO3.Rz(30, 'deg')
>>> R2
   0.866025 -0.5       0
   0.5       0.866025  0
   0         0         1
```
and the composition of these two rotations is

```python
>>> R = R1 * R2
   0.866025 -0.5       0
   0.433013  0.75     -0.5
   0.25      0.433013  0.866025
```

We can find the corresponding Euler angles (in radians)

```python
>> R.eul()
array([-1.57079633,  0.52359878,  2.0943951 ])
```

Frequently in robotics we want a sequence, a trajectory, of rotation matrices or poses. These pose classes inherit capability from the `list` class

```python
>>> R = SO3()   # the null rotation or identity matrix
>>> R.append(R1)
>>> R.append(R2)
>>> len(R)
 3
>>> R[1]
   1         0         0
   0         0.955336 -0.29552
   0         0.29552   0.955336
```
and this can be used in `for` loops and list comprehensions.

An alternative way of constructing this would be (`R1`, `R2` defined above)

```python
>>> R = SO3( [ SO3(), R1, R2 ] )
>>> len(R)
 3
```

Many of the constructors such as `.Rx`, `.Ry` and `.Rz` support vectorization

```python
>>> R = SO3.Rx( np.arange(0, 2*np.pi, 0.2))
>>> len(R)
 32
```
which has created, in a single line, a list of rotation matrices.

Vectorization also applies to the operators, for instance

```python
>>> A = R * SO3.Ry(0.5)
>>> len(R)
 32
```
will produce a result where each element is the product of each element of the left-hand side with the right-hand side, ie. `R[i] * SO3.Ry(0.5)`.

Similarly

```python
>>> A = SO3.Ry(0.5) * R
>>> len(R)
 32
```
will produce a result where each element is the product of the left-hand side with each element of the right-hand side , ie. `SO3.Ry(0.5) * R[i] `.

Finally

```python
>>> A = R * R
>>> len(R)
 32
```
will produce a result where each element is the product of each element of the left-hand side with each element of the right-hand side , ie. `R[i] * R[i] `.

The underlying representation of these classes is a numpy matrix, but the class ensures that the structure of that matrix is valid for the particular group represented: SO(2), SE(2), SO(3), SE(3).  Any operation that is not valid for the group will return a matrix rather than a pose class, for example

```python
>>> SO3.Rx(0.3) * 2
array([[ 2.        ,  0.        ,  0.        ],
       [ 0.        ,  1.91067298, -0.59104041],
       [ 0.        ,  0.59104041,  1.91067298]])

>>> SO3.Rx(0.3) - 1
array([[ 0.        , -1.        , -1.        ],
       [-1.        , -0.04466351, -1.29552021],
       [-1.        , -0.70447979, -0.04466351]])
```

We can print and plot these objects as well

```
>>> T = SE3(1,2,3) * SE3.Rx(30, 'deg')
>>> T.print()
   1         0         0         1
   0         0.866025 -0.5       2
   0         0.5       0.866025  3
   0         0         0         1

>>> T.printline()
t =        1,        2,        3; rpy/zyx =       30,        0,        0 deg

>>> T.plot()
```

![trplot](https://github.com/rai-opensource/spatialmath-python/raw/master/docs/figs/fig1.png)

`printline` is a compact single line format for tabular listing, whereas `print` shows the underlying matrix and for consoles that support it, it is colorised, with rotational elements in red and translational elements in blue.

For more detail checkout the shipped Python notebooks:

* [gentle introduction](https://github.com/rai-opensource/spatialmath-python/blob/master/notebooks/gentle-introduction.ipynb)
* [deeper introduction](https://github.com/rai-opensource/spatialmath-python/blob/master/notebooks/introduction.ipynb)


You can browse it statically through the links above, or clone the toolbox and run them interactively using [Jupyter](https://jupyter.org) or [JupyterLab](https://jupyter.org).


## Low-level spatial math


Import the low-level transform functions

```
>>> from spatialmath.base import *
```

We can create a 3D rotation matrix

```
>>> rotx(0.3)
array([[ 1.        ,  0.        ,  0.        ],
       [ 0.        ,  0.95533649, -0.29552021],
       [ 0.        ,  0.29552021,  0.95533649]])

>>> rotx(30, unit='deg')
array([[ 1.       ,  0.       ,  0.       ],
       [ 0.       ,  0.8660254, -0.5      ],
       [ 0.       ,  0.5      ,  0.8660254]])
```
The results are `numpy` arrays so to perform matrix multiplication you need to use the `@` operator, for example

```
rotx(0.3) @ roty(0.2)
```

We also support multiple ways of passing vector information to functions that require it:

* as separate positional arguments

```
transl2(1, 2)
array([[1., 0., 1.],
       [0., 1., 2.],
       [0., 0., 1.]])
```

* as a list or a tuple

```
transl2( [1,2] )
array([[1., 0., 1.],
       [0., 1., 2.],
       [0., 0., 1.]])

transl2( (1,2) )
Out[444]:
array([[1., 0., 1.],
       [0., 1., 2.],
       [0., 0., 1.]])
```

* or as a `numpy` array

```
transl2( np.array([1,2]) )
Out[445]:
array([[1., 0., 1.],
       [0., 1., 2.],
       [0., 0., 1.]])
```


There is a single module that deals with quaternions, unit or not, and the representation is a `numpy` array of four elements.  As above, functions can accept the `numpy` array, a list, dict or `numpy` row or column vectors.

```
>>> from spatialmath.base.quaternion import *
>>> q = qqmul([1,2,3,4], [5,6,7,8])
>>> q
array([-60,  12,  30,  24])
>>> qprint(q)
-60.000000 < 12.000000, 30.000000, 24.000000 >
>>> qnorm(q)
72.24956747275377
```

## Graphics

![trplot](https://github.com/rai-opensource/spatialmath-python/raw/master/docs/figs/transforms3d.png)

The functions support various plotting styles

```
trplot( transl(1,2,3), frame='A', rviz=True, width=1, dims=[0, 10, 0, 10, 0, 10])
trplot( transl(3,1, 2), color='red', width=3, frame='B')
trplot( transl(4, 3, 1)@trotx(math.pi/3), color='green', frame='c', dims=[0,4,0,4,0,4])
```

Animation is straightforward

```
tranimate(transl(4, 3, 4)@trotx(2)@troty(-2), frame='A', arrow=False, dims=[0, 5], nframes=200)
```

and it can be saved to a file by

```
tranimate(transl(4, 3, 4)@trotx(2)@troty(-2), frame='A', arrow=False, dims=[0, 5], nframes=200, movie='out.mp4')
```

![animation video](./docs/figs/animate.gif)

At the moment we can only save as an MP4, but the following incantation will covert that to an animated GIF for embedding in web pages

```
ffmpeg -i out -r 20 -vf "fps=10,scale=640:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" out.gif
```

For use in a Jupyter notebook, or on Colab, you can display an animation by
```
from IPython.core.display import HTML
HTML(tranimate(transl(4, 3, 4)@trotx(2)@troty(-2), frame='A', arrow=False, dims=[0, 5], nframes=200, movie=True))
```
The `movie=True` option causes `tranimate` to output an HTML5 fragment which
is displayed inline by the `HTML` function.

## Symbolic support

Some functions have support for symbolic variables, for example

```
import sympy

theta = sym.symbols('theta')
print(rotx(theta))
[[1 0 0]
 [0 cos(theta) -sin(theta)]
 [0 sin(theta) cos(theta)]]
```

The resulting `numpy` array is an array of symbolic objects not numbers &ndash; the constants are also symbolic objects.  You can read the elements of the matrix

```
a = T[0,0]

a
Out[258]: 1

type(a)
Out[259]: int

a = T[1,1]
a
Out[256]:
cos(theta)
type(a)
Out[255]: cos
```
We see that the symbolic constants are converted back to Python numeric types on read.

Similarly when we assign an element or slice of the symbolic matrix to a numeric value, they are converted to symbolic constants on the way in.

## Making a New Release
- Make a pull request that bumps the version number in `pyproject.toml`, which follows the major.minor.patch semantic versioning convention. 
- Then, go to the URL https://github.com/rai-opensource/spatialmath-python/releases, click draft new release. Click the tag selection dialouge, then click create new tag, and make a tag that matches the semantic version numbers in the pull request you just made. Click "generate release notes", then click publish release. This will automatically kick off the workflow in `.github/workflows/publish.yml` and the new version of the code on PyPi (https://pypi.org/project/spatialmath-python).

## Migration Notice

On April 28, 2026 this repository moved to the **RAI-Opensource** GitHub organization.  If you are still pulling from the previous bdaiinstitute GitHub organization, please update your git remote:
   ```bash
   git remote set-url origin https://github.com/RAI-Opensource/spatialmath-python.git
   ```

Also update any links of `https://bdaiinstitute.github.io/spatialmath-python` to  <https://spatialmath-python.rai-inst.com/>.

If you maintain downstream docs/scripts, update links from this repository to the new org.


## History & Contributors

This package was originally created by [Peter Corke](https://github.com/petercorke) and [Jesse Haviland](https://github.com/jhavl) and was inspired by the [Spatial Math Toolbox for MATLAB](https://github.com/petercorke/spatialmath-matlab).  It supports the textbook [Robotics, Vision & Control in Python 3e](https://github.com/petercorke/RVC3-python).

The package is now a collaboration with the [Robotics and AI Institute](https://rai-inst.com/).
