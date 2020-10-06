# Mastodon Reader for Python
only GraphReader is implemented, yet

ported from: https://github.com/mastodon-sc/matlab-mastodon-importer

# Example
```python
from mastodon_reader import MastodonReader

mr = MastodonReader("path/to/file.mastodon")

# show meta data
print(mr.read_metadata())

# get Graph
V, E = mr.read_graph()
```
