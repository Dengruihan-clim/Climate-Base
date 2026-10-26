# Validation

Validation completed on 26 October 2026 with Python 3.12 and synthetic inputs.

- **36 automated tests passed.** These cover analytic pressure integration,
  pressure and wind units, longitude sectors, the order of intensity calculation
  and regression, common-year sample standardization, OLS and p-values, seasonal
  weights, edge diagnostics, malformed inputs, and metadata privacy.
- **Real spectral backend passed:** skyborn 0.4.5 recovered the analytic divergent
  meridional wind with relative RMS error `9.682837447967361e-08`; the maximum
  absolute divergent zonal-wind error was `0 m/s`. The test tolerance is `2e-5`.
- **Both complete CLI routes passed:** the same 12-year synthetic example was run
  using total winds and using its known divergent component. Both produced
  NetCDF, CSV, and horizontal/vertical PNG outputs, which were visually checked.
  The relative RMS differences were approximately `5.22e-8` for streamfunction,
  `3.10e-7` for horizontal response, and `3.36e-7` for the main regional response.
- Source and generated metadata were checked for private paths and inherited
  source information. Only reusable source, documentation, and tests are published.

Reproduce the checks from this directory:

```bash
python -m pip install -r requirements.txt pytest
python -m pytest tests -q
python examples/generate_synthetic.py --output-dir demo_input
python hadley_ma2025.py --winds demo_input/winds.nc --input-kind divergent --v-var vd --index demo_input/index.csv --output-dir demo_output/divergent
# Requires a real optional spectral backend:
python tools/validate_spectral_backend.py
python hadley_ma2025.py --winds demo_input/winds.nc --input-kind total --truncation 6 --index demo_input/index.csv --output-dir demo_output/total
```

These checks validate this implementation's numerical behavior. They do not
reproduce the complete CP/EP ENSO experiments or scientific conclusions of the
reference paper.
