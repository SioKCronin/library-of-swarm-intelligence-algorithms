# ETDA Examples

## High-Dimensional ROI Example

`high_dim_roi_example.py` demonstrates the complete ETDA workflow:

1. **Large-Scale Multidimensional ROI**: Creates a 50-dimensional region of interest with 1000 samples
2. **Persistence Homology Mapping**: Computes topological structure using giotto-tda
3. **Manifold Reduction**: Reduces 50D space to 10D using PCA
4. **Global Maxima Identification**: Uses TDA to identify candidate optima on the manifold
5. **Swarm Optimization**: Runs Bat Algorithm on the reduced space

### Running the Example

```bash
cd etda
pip install -e .
python examples/high_dim_roi_example.py
```

### Output

The example will show:
- ROI creation statistics
- Persistence homology results (entropy, critical points)
- TDA-identified candidate maxima
- Swarm optimization results
- Comparison between TDA candidates and swarm optimization

### Use Cases

This approach is particularly useful for:
- **Health Data**: High-dimensional biomarker spaces
- **Treatment Optimization**: Finding optimal parameter combinations
- **Feature Selection**: Identifying important regions in high-D spaces
- **Drug Discovery**: Exploring chemical/biological spaces

