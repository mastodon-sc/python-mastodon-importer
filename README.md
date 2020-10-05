# Mastodon Reader for Python
only GraphReader is implemented, yet

ported from: https://github.com/mastodon-sc/matlab-mastodon-importer

# Example
```python
from mastodon_reader import MastodonReader

mr = MastodonReader("H:/projects/066_nikhil_lineage/mastodon_playground/ilastik_bdv_hdf5_smooth.mastodon")
V, E = mr.read_graph()
```
