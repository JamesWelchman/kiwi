# Kiwi Model Analyzer

Kiwi is a port of the excellent [LIME](https://github.com/marcotcr/lime)
algorithm. It is a Python library for use in scripts/notebooks but the hard
numerical processing is implemented in Rust for performance.

It is particularly designed for processing large numbers of records in
a timely and efficient manner. The current benchmark is the adult dataset
with XDGBoost, currently standing at `~10_000 records per minute`.

Currently only classification of tabular data is supported.


## Building

Currently there is no packaged solution for building Kiwi, here is a list of
instructions which _should_ work cross platform.


### 1. Obtaining tools

You will need:

	- The Rust compiler
	- Python development headers
	- A BLAS implementation (possibly with a fortran compiler)

You will first need a the Rust compiler, see [here](https://www.rust-lang.org/tools/install)
for details.

Documentation on obtaining Python headers is [here](https://pyo3.rs/v0.13.2/).

Avaliable BLAS implementations are documented [here](https://github.com/rust-ndarray/ndarray-linalg#backend-features). By default this library uses openblas-static for building.
(And thus, from the documentation requires the GNU C compiler and the GNU fortran compiler as well).

### 2. Building

Having obtained all these tools, this should work.

```bash
  $ git clone https://github.com/jameswelchman/kiwi.git
  $ cd kiwi
  $ cargo build --release
```

Assuming the build was succesful, one can now "install" the library to the main repo path.


#### Linux
```
  $ cp target/release/libkiwi.so kiwi.so
```

#### OS/X
```
  $ cp target/release/libkiwi.dylib kiwi.so
```

#### Win32
```
  $ cp target/release/kiwi.dll kiwi.pyd
```

## Getting started

There are two notebooks [iris](../blob/master/Iris.ipynb) and the more in-depth
[adult](../blob/master/adult_xdg.ipynb). These two notebooks descibe most of the tabular
API for Kiwi.

## API Docs

Currently API notes are maintained in the file [lib.rs](../blob/master/src/lib.rs).

The author is optimistic about the pyo3/sphinx ecosystem providing a better solution
in the future.

## Troubleshooting

### Kernel Crashed / Memory allocation error with `explain_instance_iter`

Try reducing the `max_memory` keyword parameter from the default.

```python
    explainer = KiwiTabularExplainer(training_data)
    explainer.explain_instance_iter(data, predict_fn, max_memory=16_000_000)
```

### `explain_instance_iter` running very slowly

Try using the `sample_background_thread=True` when creating the tabular explainer.
Usually this decreases performance, however it never falls below a certain minimum.

```python
    explainer = KiwiTabularExplainer(training_data, sample_background_thread=True)
    explainer.explain_instance_iter(data, predict_fn)
```