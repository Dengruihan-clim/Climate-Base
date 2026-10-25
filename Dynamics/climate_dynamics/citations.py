"""Original method and upstream software references for the diagnostics.

Authors: RH_Deng; Codex (OpenAI, AI-assisted organization and refactoring).
Scientific formulations and upstream solvers remain credited to their authors.
Full bibliographic records and implementation boundaries are in REFERENCES.md.
"""

CITATIONS = {
    "takaya_nakamura_2001": {
        "authors": "Takaya, K., and H. Nakamura",
        "year": 2001,
        "title": "A Formulation of a Phase-Independent Wave-Activity Flux for Stationary and Migratory Quasigeostrophic Eddies on a Zonally Varying Basic Flow",
        "journal": "Journal of the Atmospheric Sciences",
        "volume": "58", "pages": "608-627",
        "doi": "10.1175/1520-0469(2001)058<0608:AFOAPI>2.0.CO;2",
        "url": "https://doi.org/10.1175/1520-0469%282001%29058%3C0608%3AAFOAPI%3E2.0.CO%3B2",
    },
    "sardeshmukh_hoskins_1988": {
        "authors": "Sardeshmukh, P. D., and B. J. Hoskins",
        "year": 1988,
        "title": "The Generation of Global Rotational Flow by Steady Idealized Tropical Divergence",
        "journal": "Journal of the Atmospheric Sciences",
        "volume": "45", "pages": "1228-1251",
        "doi": "10.1175/1520-0469(1988)045<1228:TGOGRF>2.0.CO;2",
        "url": "https://doi.org/10.1175/1520-0469%281988%29045%3C1228%3ATGOGRF%3E2.0.CO%3B2",
    },
    "dawson_2016": {
        "authors": "Dawson, A.", "year": 2016,
        "title": "Windspharm: A High-Level Library for Global Wind Field Computations Using Spherical Harmonics",
        "journal": "Journal of Open Research Software",
        "volume": "4", "article": "e31",
        "doi": "10.5334/jors.129", "url": "https://doi.org/10.5334/jors.129",
    },
    "adams_swarztrauber_1999": {
        "authors": "Adams, J. C., and P. N. Swarztrauber", "year": 1999,
        "title": "SPHEREPACK 3.0: A Model Development Facility",
        "journal": "Monthly Weather Review", "volume": "127", "pages": "1872-1878",
        "doi": "10.1175/1520-0493(1999)127<1872:SAMDF>2.0.CO;2",
        "url": "https://doi.org/10.1175/1520-0493%281999%29127%3C1872%3ASAMDF%3E2.0.CO%3B2",
    },
    "oort_yienger_1996": {
        "authors": "Oort, A. H., and J. J. Yienger", "year": 1996,
        "title": "Observed Interannual Variability in the Hadley Circulation and Its Connection to ENSO",
        "journal": "Journal of Climate", "volume": "9", "pages": "2751-2767",
        "doi": "10.1175/1520-0442(1996)009<2751:OIVITH>2.0.CO;2",
        "url": "https://doi.org/10.1175/1520-0442%281996%29009%3C2751%3AOIVITH%3E2.0.CO%3B2",
    },
    "adam_et_al_2018": {
        "authors": "Adam, O., K. M. Grise, P. Staten, I. R. Simpson, S. M. Davis, N. A. Davis, D. W. Waugh, T. Birner, and A. Ming",
        "year": 2018,
        "title": "The TropD software package (v1): standardized methods for calculating tropical-width diagnostics",
        "journal": "Geoscientific Model Development", "volume": "11", "pages": "4339-4357",
        "doi": "10.5194/gmd-11-4339-2018", "url": "https://doi.org/10.5194/gmd-11-4339-2018",
    },
}

METHOD_REFERENCES = {
    "tn_waf": ("takaya_nakamura_2001",),
    "rossby_wave_source": ("sardeshmukh_hoskins_1988",),
    "linear_rws": ("sardeshmukh_hoskins_1988",),
    "wind": ("dawson_2016", "adams_swarztrauber_1999"),
    "mass_streamfunction": ("oort_yienger_1996", "adam_et_al_2018"),
}
