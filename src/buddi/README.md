# Development Notes

## What's done:

1. Preprocessing module under `buddi.preprocessing.sc_augmentor`, which does:
   - Random uniform pseudobulk generation
   - Realistic pseudobulk generation
   - Cell type dominant pseudobulk generation
   - Some control over noise injection
   - Visualization of Integrated pseudobulk and bulk data with `buddi.plotting.plot_data`
1. `BuDDI3`/`BuDDI4` class that works with:
   - generic data class `buddi.data.BuDDINData`
   - generic fitter function `buddi.model.fit.fit_buddi`
   - generic visualization `buddi.plotting.plot_latent_space` and `buddi.plotting.plot_loss`
   - dataset classes `buddi.dataset.buddi3_dataset.*` and `buddi.dataset.buddi4_dataset.*`
1. Some training Examples

## Missing:

- Generic BuDDI with arbitrary number of encoder branches (which can be simply implemented by subclassing `buddi.model.buddi_abstract_class`, see existing `BuDDI3` and `BuDDI4` as reference), but will need custom dataset classes for use with generic fitter function (also reference `buddi.dataset.buddi3_dataset.*` and `buddi.dataset.buddi4_dataset.*`)
- BuDDI validation helper (currently in analysis repo)
- Installation
