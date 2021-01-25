import io
import numpy; np = numpy
import pandas
import struct
import zipfile
import xml.etree.ElementTree as ET

SUPPORTED_MASTODON_VERSIONS = ["0.3"]


def ismember(a_vec, b_vec, method=None):
    """

    Description
    -----------
    MATLAB equivalent ismember function
    [LIA,LOCB] = ISMEMBER(A,B) also returns an array LOCB containing the
    lowest absolute index in B for each element in A which is a member of
    B and 0 if there is no such index.
    Parameters
    ----------
    a_vec : list or array
    b_vec : list or array
    method : None or 'rows' (default: None).
        rows can be used for row-wise matrice comparison.
    Returns an array containing logical 1 (true) where the data in A is found
    in B. Elsewhere, the array contains logical 0 (false)
    -------
    Tuple

    Example
    -------
    >>> a_vec = np.array([1,2,3,None])
    >>> b_vec = np.array([4,1,2])
    >>> Iloc,idx = ismember(a_vec,b_vec)
    >>> a_vec[Iloc] == b_vec[idx]

    """
    # Set types
    a_vec, b_vec = _settypes(a_vec, b_vec)

    # Compute
    if method is None:
        Iloc, idx = _compute(a_vec, b_vec)
    elif method == "rows":
        if a_vec.shape[0] != b_vec.shape[0]:
            raise Exception("Error: Input matrices should have same number of columns.")
        # Compute row-wise over the matrices
        out = list(map(lambda x, y: _compute(x, y), a_vec, b_vec))
        # Unzipping
        Iloc, idx = list(zip(*out))
    else:
        Iloc, idx = None, None

    return (Iloc, idx)

def _settypes(a_vec, b_vec):
    if "pandas" in str(type(a_vec)):
        a_vec.values[np.where(a_vec.values == None)] = "NaN"
        a_vec = np.array(a_vec.values)
    if "pandas" in str(type(b_vec)):
        b_vec.values[np.where(b_vec.values == None)] = "NaN"
        b_vec = np.array(b_vec.values)
    if isinstance(a_vec, list):
        a_vec = np.array(a_vec)
        # a_vec[a_vec==None]='NaN'
    if isinstance(b_vec, list):
        b_vec = np.array(b_vec)
        # b_vec[b_vec==None]='NaN'

    return a_vec, b_vec


def _compute(a_vec, b_vec):
    bool_ind = np.isin(a_vec, b_vec)
    common = a_vec[bool_ind]
    [common_unique, common_inv] = np.unique(common, return_inverse=True)
    [b_unique, b_ind] = np.unique(b_vec, return_index=True)
    common_ind = b_ind[np.isin(b_unique, common_unique, assume_unique=True)]

    return bool_ind, common_ind[common_inv]


def RGBint2RGB(rgb_int):
    B = rgb_int & 255
    G = (rgb_int >> 8) & 255
    R = (rgb_int >> 16) & 255
    return (R, G, B)


class JavaRawReader:
    MAGIC = -21267
    VERSION = 5

    def __init__(self, raw_file):
        if isinstance(raw_file, io.IOBase):
            self._fh = raw_file
        elif isinstance(raw_file, str):
            self._fh = open(raw_file, "rb")
        else:
            raise RuntimeError("asdf")

        self.block = b""
        self.index = 0

        if not struct.unpack(">h", self._fh.read(2))[0] == JavaRawReader.MAGIC:
            raise RuntimeError("Wrong format")

        if not struct.unpack(">h", self._fh.read(2))[0] == JavaRawReader.VERSION:
            raise RuntimeError("Wrong version")

    def read(self, size, fmt):
        if size > len(self.block) - self.index:
            self._fetch_block()

        res = struct.unpack(fmt, self.block[self.index : self.index + size])
        self.index += size

        return res

    def read_int(self):
        return self.read(4, ">i")[0]

    def read_short(self):
        return self.read(2, ">h")[0]

    def read_utf8(self):
        str_length = self.read_short()

        return b"".join(self.read(str_length, ">" + "c" * str_length)).decode("utf-8")

    def _fetch_block(self):
        block_key = struct.unpack(">b", self._fh.read(1))[0]
        if block_key == 122:
            block_length = struct.unpack(">i", self._fh.read(4))[0]
        elif block_key == 119:
            block_length = struct.unpack(">b", self._fh.read(1))[0]
        else:
            raise RuntimeError("block not understood")

        self.block = self.block[self.index :] + self._fh.read(block_length)

        self.index = 0

    def close(self):
        self._fh.close()

    def __del__(self):
        self.close()


class MastodonReader:
    def __init__(self, source_file):
        self.source_file = source_file

    def read_metadata(self):
        with zipfile.ZipFile(self.source_file) as masto_zip:
            with masto_zip.open("project.xml", "r") as proj_xml:
                xml_root = ET.fromstring(proj_xml.read())

        mamut_version = xml_root.attrib["version"]
        if not mamut_version in SUPPORTED_MASTODON_VERSIONS:
            raise RuntimeWarning(
                f"Warning: Version mismatch with found version '{mamut_version}'. Supported are {' '.join(SUPPORTED_MASTODON_VERSIONS)}."
            )

        movie_filename = xml_root.findall("SpimDataFile")[0].text
        space_units = xml_root.findall("SpaceUnits")[0].text
        time_units = xml_root.findall("TimeUnits")[0].text

        return {
            "version": mamut_version,
            "file": movie_filename,
            "space unit": space_units,
            "time unit": time_units,
        }

    def read_tags(self, V, E):
        with zipfile.ZipFile(self.source_file) as masto_zip:
            fh = masto_zip.open("tags.raw", "r")
            jr = JavaRawReader(fh)

            tss = self._read_tag_set_structure(jr)

            label_sets_vertices, map_vertices = self._read_label_set_property_map(jr)
            label_sets_edges, map_edges = self._read_label_set_property_map(jr)

            self._append_tags_to_table(map_vertices, label_sets_vertices, tss, V)
            self._append_tags_to_table(map_edges, label_sets_edges, tss, E)

            return tss

    def _append_tags_to_table(self, map_, label_sets, tss, tab):
        # Prepare columns.
        n_tag_set = len(tss)
        columns = {}
        for i in range(n_tag_set):
            columns[i] = numpy.zeros(len(tab), dtype="int32")
            columns[i].fill(-1)

        # Map tag ids to tag_set.
        n_total_tags = 0
        for ts in tss:
            n_total_tags = n_total_tags + len(ts["tags"])

        tag_map = {}
        tag_set_map = {}

        for i in range(len(tss)):
            for j in range(len(tss[i]["tags"])):
                tag_map[tss[i]["tags"][j]["id"]] = tss[i]["tags"][j]["label"]
                tag_set_map[tss[i]["tags"][j]["id"]] = i  ###

        n_label_sets = len(label_sets)

        #  Process label-set by label-set.
        for i in range(n_label_sets):

            label_set = label_sets[i]
            n_labels = len(label_set)

            for j in range(n_labels):

                # What is the tag-id of this element in the label set?
                tag_id = label_set[j]

                # What tag-set column are we editing for this tag_id?
                tag_set = tag_set_map[tag_id]

                # % What rows, in the map, have this label-set?
                idx2 = numpy.nonzero(map_[:, 1] == i)[0]  # orig: ( i - 1  ) ); % 1 -> 0

                #  What object ids correspond to these rows?
                object_ids = map_[idx2, 0]

                # % What are the rows, in the table, that have these ids?

                _, idx1 = ismember(object_ids, tab.index.to_numpy())

                #% Fill these rows with the tag_id
                columns[tag_set][idx1] = tag_id

        for i, ts in enumerate(tss):
            tab[ts["name"] + "_ID"] = columns[i]
            tab[ts["name"] + "_NAME"] = ""

            tab[ts["name"] + "_NAME"] = tab[ts["name"] + "_ID"].apply(
                lambda xxx: tag_map[xxx] if xxx > -1 else ""
            )

        return tss

    def _read_label_set_property_map(self, jr):
        num_sets = jr.read_int()
        label_sets = []

        for i in range(num_sets):
            num_labels = jr.read_int()
            labels = numpy.zeros(num_labels, dtype="int32")

            for j in range(num_labels):
                # The labels are ints in Mastodon -> tag linear index
                labels[j] = jr.read_int()

            label_sets.append(labels)

        # Read entries.
        size = jr.read_int()
        map_ = numpy.zeros((size, 2), dtype="int32")

        for i in range(size):
            map_[i, 0] = jr.read_int()
            map_[i, 1] = jr.read_int()

        return label_sets, map_

    def _read_tag_set_structure(self, jr):
        n_tag_sets = jr.read_int()

        tag_sets = []

        for _ in range(n_tag_sets):
            tag_set_id = jr.read_int()
            tag_set_name = jr.read_utf8()
            n_tags = jr.read_int()

            tag_set = {"name": tag_set_name, "label": tag_set_id}

            tags = []
            for _ in range(n_tags):
                tag_id = jr.read_int()
                tag_label = jr.read_utf8()
                tag_color = jr.read_int()
                tag = {
                    "id": tag_id,
                    "label": tag_label,
                    "color": tag_color,
                    "color_rgb": RGBint2RGB(tag_color),
                }
                tags.append(tag)

            tag_set["tags"] = tags

            tag_sets.append(tag_set)

        return tag_sets

    def read_features(self):
        pass

    def create_nx_graph(self, V, E):
        import networkx as nx

        G = nx.from_pandas_edgelist(
            E, source="source_idx", target="target_idx", create_using=nx.DiGraph
        )
        nx.set_node_attributes(G, V.to_dict("index"))

        return G

    def read(self):
        V, E = self.read_tables()
        tss = self.read_tags(V, E)
        G = self.create_nx_graph(V, E)

        return G, V, E, tss

    def read_tables(self):
        with zipfile.ZipFile(self.source_file) as masto_zip:
            fh = masto_zip.open("model.raw", "r")

            jr = JavaRawReader(fh)

            try:
                n_verticices = jr.read_int()
                V = numpy.zeros(shape=(n_verticices, 11), dtype=numpy.float32)

                for i in range(n_verticices):
                    V[i, 0:3] = jr.read(8 * 3, "<ddd")
                    V[i, 3] = jr.read(4, "<i")[0]
                    V[i, 4:11] = jr.read(8 * 7, "<ddddddd")

                V = pandas.DataFrame(
                    V,
                    columns=[
                        "x",
                        "y",
                        "z",
                        "t",
                        "c_11",
                        "c_12",
                        "c_13",
                        "c_22",
                        "c_23",
                        "c_33",
                        "bsrs",
                    ],
                )

                n_edges = jr.read_int()
                E = numpy.zeros((n_edges, 3), dtype=numpy.int32)

                for i in range(n_edges):
                    E[i, :2] = jr.read(4 * 4, ">iiii")[:2]
                    E[i, 2] = i - 1

                E = E[numpy.argsort(E[:, 0]), :]

                E = pandas.DataFrame(E, columns=["source_idx", "target_idx", "id"])

                return V, E

            except:
                raise
            finally:
                jr.close()


if __name__ == "__main__":
    pass
