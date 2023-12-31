# napari-GapSeq2

[![License MIT](https://img.shields.io/pypi/l/napari-GapSeq2.svg?color=green)](https://github.com/piedrro/napari-GapSeq2/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-GapSeq2.svg?color=green)](https://pypi.org/project/napari-GapSeq2)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-GapSeq2.svg?color=green)](https://python.org)
[![tests](https://github.com/piedrro/napari-GapSeq2/workflows/tests/badge.svg)](https://github.com/piedrro/napari-GapSeq2/actions)
[![codecov](https://codecov.io/gh/piedrro/napari-GapSeq2/branch/main/graph/badge.svg)](https://codecov.io/gh/piedrro/napari-GapSeq2)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-GapSeq2)](https://napari-hub.org/plugins/napari-GapSeq2)

A **Napari** plugin for extracting time series traces from **Single Molecule Localisation Microsocpy** (SMLM) data, using **Picasso** (picassosr) as a backend. Includes features for **aligning** image channels/datasets, **undrifting** images, **detecting/fitting** localisations and extracting **traces**, and supports both **ALEX** and **FRET** data. Traces can be exported in different formats for downstream analysis. 

This was built by Dr Piers Turner from the Kapanidis Lab, University of Oxford.

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Installation

You can install `napari-GapSeq2` via [GitHub]:

    conda create –-name napari-gapseq2 python==3.9
    conda activate napari-gapseq2
    conda install -c anaconda git
    conda update --all

    pip install git+https://github.com/piedrro/napari-GapSeq2.git


## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [MIT] license,
"napari-GapSeq2" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/piedrro/napari-GapSeq2/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
