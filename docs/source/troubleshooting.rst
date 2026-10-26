Troubleshooting
===============

.. _slow-first-plot:

Slow first plot
---------------

The first plot in a fresh environment can take several seconds while Matplotlib
scans installed fonts and builds its font cache. You may see the message
``Matplotlib is building the font cache; this may take a moment``. The time
depends on the system and the installed fonts.

SpatialMath imports Matplotlib's plotting interface lazily, so this cost is
usually deferred until plotting is first requested. A fast ``import spatialmath``
does not mean that the font cache is already populated. If the delay recurs in
each new process, check whether the cache directory is writable and persists
between runs, especially in containers, CI runners and sandboxes.

Find the cache directory
^^^^^^^^^^^^^^^^^^^^^^^^

Use the same Python environment as your application:

.. code-block:: shell

   python -c "import matplotlib; print(matplotlib.get_cachedir())"

``MPLCONFIGDIR`` overrides the directory used for Matplotlib's configuration and
cache. Set it **before importing Matplotlib**. If the usual directories are not
writable, Matplotlib can fall back to a temporary directory, which may prevent
reuse between runs. See Matplotlib's
`environment variable documentation <https://matplotlib.org/stable/install/environment_variables_faq.html#envvar-MPLCONFIGDIR>`_
for platform-specific defaults.

Keep and prepopulate the cache
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a Linux or macOS shell, choose a persistent, writable directory and run a
one-time warm-up:

.. code-block:: shell

   export MPLCONFIGDIR="$HOME/.cache/spatialmath-matplotlib"
   mkdir -p "$MPLCONFIGDIR"
   MPLBACKEND=Agg python -c "import matplotlib.pyplot"

Use the same ``MPLCONFIGDIR`` when running the application. The warm-up still
pays the initial scan cost; retaining the cache lets later processes reuse it.
``Agg`` allows this command to run without a display and is selected only for
the warm-up command, so it does not change the application's plotting backend.

* **Docker:** after installing Matplotlib and the image's fonts, prepopulate the
  cache during the image build:

  .. code-block:: dockerfile

     RUN MPLBACKEND=Agg python -c "import matplotlib.pyplot"

  The build step and application must use the same cache directory, with
  appropriate permissions for the runtime user. A cache built in root's home
  will not help an application using another user's default cache directory.
  A volume mounted over the populated directory also hides the built-in cache.

* **CI:** restore the cache directory before the first plot or warm-up command,
  and save it afterwards using the CI system's cache facility. Include the
  operating system and Matplotlib version in the cache key, and refresh it when
  the installed fonts change.

* **Sandboxes:** configure ``MPLCONFIGDIR`` to point to a writable directory that
  survives between runs, rather than a temporary directory recreated each time.
