# Mastodon Reader for Python
no tags/features import is implemented, yet

ported from: https://github.com/mastodon-sc/matlab-mastodon-importer

# Example
```python
from mastodon_reader import MastodonReader

mr = MastodonReader("path/to/file.mastodon")

# show meta data
print(mr.read_metadata())

# get Graph (pandas tables)
V, E = mr.read_graph()

# get networkx Graph
G = mr.read_graph_networkx()
```
