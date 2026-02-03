"""
Test that we can import buddi_v2 and any submodules without errors.
"""

import pytest

def test_import_buddi_v2():
    import buddi_v2
    import buddi_v2.preprocessing as preproc
    import buddi_v2.plotting as plotting
    from buddi_v2.models import buddi4
    from buddi_v2.preprocessing import sc_augmentor
    from buddi_v2.plotting import plot_latent_space as pls

    assert buddi_v2 is not None
    assert buddi4 is not None
    assert sc_augmentor is not None
    assert pls is not None
    assert preproc is not None
    assert plotting is not None
