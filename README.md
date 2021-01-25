# Mastodon Reader for Python
Read graph representation, lineage tables, meta data, and tags from a Mastodon project file.

(no feature import implemented, yet)

ported from: https://github.com/mastodon-sc/matlab-mastodon-importer

# Example
Read the mastodon file:
```python
from mastodon_reader import MastodonReader

mr = MastodonReader("path/to/file.mastodon")

# show meta data
mr.read_metadata()

# read (networkX) Graph representation, Nodes and Edge tables, and the tag set definitions
Graph, Nodes, Edges, TagSets = mr.read()

```
or read information separately

```python
# read only Nodes and Edge tables
Nodes, Edges = mr.read_tables()

# Read Tags and add found tags as columns to the Nodes and Edge tables
TagSets = mr.read_tags(Nodes, Edges)

# Create netwworkX Graph DiGraph representation
Graph = mr.create_nx_graph(Nodes, Edges)
```
