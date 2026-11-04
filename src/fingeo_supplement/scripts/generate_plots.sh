# River plot, lineage tree, data scatter, geodesic paths:
uv run python scripts/export_plots.py --dataset zebrafish_cns --finsler
uv run python scripts/export_plots.py --dataset zebrafish_arch --finsler
uv run python scripts/export_plots.py --dataset mouse_blood --finsler

# Plus classifier confidence/entropy heatmaps (2D synthetic):
uv run python scripts/export_plots.py --dataset synthetic --finsler --dim 2

# Non-Finsler (CFM-only) variant, for the side-by-side river-plot comparison:
uv run python scripts/export_plots.py --dataset zebrafish_cns --no-finsler

# uv run python scripts/export_plots.py --dataset synthetic --finsler --dim 2 --embed-epochs 1
