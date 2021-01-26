import io
import re
import numpy
import pandas
import struct
import zipfile
import xml.etree.ElementTree as ET

from .utils import ismember, RGBint2RGB

SUPPORTED_MASTODON_VERSIONS = ["0.3"]

class JavaRawReader:
    MAGIC = -21267
    VERSION = 5

    def __init__(self, raw_file):
        if isinstance(raw_file, io.IOBase):
            self._fh = raw_file
        elif isinstance(raw_file, str):
            self._fh = open(raw_file, "rb")
        else:
            raise RuntimeError("raw_file needs to be instance of str or IOBase")

        self.block = b""
        self.index = 0

        if not struct.unpack(">h", self._fh.read(2))[0] == JavaRawReader.MAGIC:
            raise RuntimeError("Wrong format")

        if not struct.unpack(">h", self._fh.read(2))[0] == JavaRawReader.VERSION:
            raise RuntimeError("Wrong version")

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        self._fh.close()

    def __del__(self):
        self.close()

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

    def read_double_rev(self):
        return self.read(8, ">d")[0]


    def _fetch_block(self):
        block_key = struct.unpack(">b", self._fh.read(1))[0]
        if block_key == 122:
            block_length = struct.unpack(">i", self._fh.read(4))[0]
        elif block_key == 119:
            block_length = struct.unpack(">b", self._fh.read(1))[0]
        else:
            raise RuntimeError("Java bytle block key not understood")

        self.block = self.block[self.index :] + self._fh.read(block_length)

        self.index = 0




class MastodonReader:
    def __init__(self, source_file):
        self.source_file = source_file

    def read_metadata(self):
        with zipfile.ZipFile(self.source_file) as masto_zip:
            with masto_zip.open("project.xml", "r") as proj_xml:
                xml_root = ET.fromstring(proj_xml.read())

        mastodon_version = xml_root.attrib["version"]
        if not mastodon_version in SUPPORTED_MASTODON_VERSIONS:
            raise RuntimeWarning(
                f"Warning: Version mismatch with found version '{mastodon_version}'. Supported are {' '.join(SUPPORTED_MASTODON_VERSIONS)}."
            )

        movie_filename = xml_root.findall("SpimDataFile")[0].text
        space_units = xml_root.findall("SpaceUnits")[0].text
        time_units = xml_root.findall("TimeUnits")[0].text

        return {
            "version": mastodon_version,
            "file": movie_filename,
            "space unit": space_units,
            "time unit": time_units,
        }

    def read_tags(self, V, E):
        with zipfile.ZipFile(self.source_file) as masto_zip:
            fh = masto_zip.open("tags.raw", "r")
            with JavaRawReader(fh) as jr:

                tss = self._read_tag_set_structure(jr)

                label_sets_vertices, map_vertices = self._read_label_set_property_map(
                    jr
                )
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

    def read_features(self, V, E):
        with zipfile.ZipFile(self.source_file) as masto_zip:
            feature_files = filter(lambda fn: fn.startswith("features/") and
            fn.endswith(".raw"), masto_zip.namelist())

            features = {}
            pat  = re.compile(r"^features/(.+?)\.raw$")
            for fn in feature_files:
                match = pat.search(fn)
                if match:
                    features[match.group(1)] = masto_zip.open(fn, "r")

            mff = MastodonFeatureFactory()

            for feature_name, feature_fn in features.items():
                masto_feature_class = mff(feature_name)
                if masto_feature_class:
                    masto_feature = masto_feature_class(feature_fn)
                    masto_feature.read(V, E)





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

            with JavaRawReader(fh) as jr:
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
                    E[i, 2] = i

                E = E[numpy.argsort(E[:, 0]), :]

                E = pandas.DataFrame(E, columns=["source_idx", "target_idx", "id"])

                return V, E


class MastodonFeatureFactory:
    def __init__(self):
        self._lookup = {}

        # self.register_class(LinkVelocity)
        # self.register_class(LinkDisplacement)
        # self.register_class(SpotGaussianFilteredIntensity)
        # self.register_class(SpotMedianIntensity)
        # self.register_class(SpotNLinks)
        # self.register_class(SpotSumIntensity)
        # self.register_class(SpotTrackID)
        # self.register_class(TrackNSpots)
        self.register_class(UpdateStackLink)
        self.register_class(UpdateStackSpot)


    def register_class(self, klass):
        self._lookup[klass.name] = klass

    def __call__(self, name):
        if name in self._lookup:
            print("found", name)
            return self._lookup[name]


class MastodonFeature:
    name = None
    add_to = None
    info = None

    def __init__(self, mastodon_feature_file):
        self.mastodon_feature_file = mastodon_feature_file

    def read(self, V, E):
        pass

    def add_projections_to_table(self, projections, V, E):
        if self.add_to == "Link":
            tab = E
        elif self.add_to == "Spot":
            tab = V
        else:
            raise RuntimeWarning(f"Mastodon Feature '{self.name}': cannot add features to table")
            return

        for projection in projections:
            tab[projection["key"]] = -1
            _, idx  = ismember( projection["map"][:, 0].astype("int32") , tab.index.to_numpy())
            tab.loc[idx, projection["key"] ] = projection["map"][:, 1]



class LinkVelocity(MastodonFeature):
    name = 'Link velocity'
    add_to = "Link"
    info = 'Computes the link velocity as the distance between the source and target spots divided by their frame difference. Units are in physical distance per frame.'

    def read(self, V, E):
        projections = []
        with JavaRawReader(self.mastodon_feature_file) as jr:
            projection = dict()
            projection["key"]         = 'Link velocity'
            projection["info"]        = self.info
            projection["dimension"]   = 'VELOCITY'
            projection["units"]       = jr.read_utf8()
            projection["map"]         = import_double_map( jr )

            projections.append(projection)

        self.add_projections_to_table(projections, V, E)



class LinkDisplacement(MastodonFeature):
    name = 'Link displacement'
    add_to = "Link"
    info = 'Computes the link displacement in physical units as the distance between the source spot and the target spot.'

    def read(self, V, E):
        projections = []
        with JavaRawReader(self.mastodon_feature_file) as jr:
            projection = dict()
            projection["key"]         = 'Link displacement'
            projection["info"]        = self.info
            projection["dimension"]   = 'LENGTH'
            projection["units"]       = jr.read_utf8()
            projection["map"]         = import_double_map( jr )

            projections.append(projection)

        self.add_projections_to_table(projections, V, E)


class SpotGaussianFilteredIntensity(MastodonFeature):
    name = 'Spot gaussian-filtered intensity'
    add_to = "Spot"
    info = """Computes the average intensity and its standard deviation inside spots over all
            sources of the dataset. The average is calculated by a weighted mean over the pixels
            of the spot, weighted by a gaussian centered in the spot and with a sigma value equal
            to the minimal radius of the ellipsoid divided by 2."""

    def read(self, V, E):
        projections = []
        with JavaRawReader(self.mastodon_feature_file) as jr:
            n_sources = jr.read_int();
            for ch in range(n_sources):
                # Mean.
                projection = dict()

                projection["key"]        = f'Spot gaussian filtered intensity Mean ch{ch}'
                projection["info"]       = self.info
                projection["dimension"]  = 'INTENSITY'
                projection["units"]      = 'Counts'
                projection["map"]        = import_double_map( jr )

                projections.append(projection)

                # Std.
                projection = dict()

                projection["key"]         = f'Spot gaussian filtered intensity Std ch{ch}'
                projection["info"]        = self.info
                projection["dimension"]   = 'INTENSITY'
                projection["units"]       = 'Counts'
                projection["map"]         = import_double_map( jr)

                projections.append(projection)

        self.add_projections_to_table(projections, V, E)

class SpotMedianIntensity(MastodonFeature):
    name = 'Spot median intensity'
    add_to = "Spot"
    info = """Computes the median intensity inside a spot,
              for the pixels inside the largest box that fits into the spot ellipsoid."""

    def read(self, V, E):
        projections = []
        with JavaRawReader(self.mastodon_feature_file) as jr:
            n_sources = jr.read_int();
            for ch in range(n_sources):
                # Mean.
                projection = dict()

                projection["key"]        = f'Spot median intensity ch{ch}'
                projection["info"]       = self.info
                projection["dimension"]  = 'INTENSITY'
                projection["units"]      = 'Counts'
                projection["map"]        = import_double_map( jr )

                projections.append(projection)


        self.add_projections_to_table(projections, V, E)

class SpotNLinks(MastodonFeature):
    name = 'Spot N links'
    add_to = "Spot"
    info = 'Computes the number of links that touch a spot.'

    def read(self, V, E):
        projections = []
        with JavaRawReader(self.mastodon_feature_file) as jr:
                # Mean.
            projection = dict()

            projection["key"]        = 'Spot N links'
            projection["info"]       = self.info
            projection["dimension"]  = 'None'
            projection["units"]      = ''
            projection["map"]        = import_int_map( jr )

            projections.append(projection)


        self.add_projections_to_table(projections, V, E)




class SpotSumIntensity(MastodonFeature):
    name = 'Spot sum intensity'
    add_to = "Spot"
    info = 'Computes the total intensity inside a spot, for the pixels inside the spot ellipsoid.'

    def read(self, V, E):
        projections = []
        with JavaRawReader(self.mastodon_feature_file) as jr:
            n_sources = jr.read_int();
            for ch in range(n_sources):
                # Mean.
                projection = dict()

                projection["key"]        = f'Spot sum intensity ch{ch}'
                projection["info"]       = self.info
                projection["dimension"]  = 'INTENSITY'
                projection["units"]      = 'Counts'
                projection["map"]        = import_double_map( jr )

                projections.append(projection)

        self.add_projections_to_table(projections, V, E)


class SpotTrackID(MastodonFeature):
    name = 'Spot track ID'
    add_to = "Spot"
    info = 'Returns the ID of the track each spot belongs to.'

    def read(self, V, E):
        projections = []
        with JavaRawReader(self.mastodon_feature_file) as jr:
                # Mean.
            projection = dict()

            projection["key"]        = 'Spot track ID'
            projection["info"]       = self.info
            projection["dimension"]  = 'None'
            projection["units"]      = ''
            projection["map"]        = import_int_map( jr )

            projections.append(projection)


        self.add_projections_to_table(projections, V, E)



class TrackNSpots(MastodonFeature):
    name = 'Track N spots'
    add_to = "Spot"
    info = 'Returns the number of spots in a track.'

    def read(self, V, E):
        projections = []
        with JavaRawReader(self.mastodon_feature_file) as jr:
                # Mean.
            projection = dict()

            projection["key"]        = 'Spot track ID'
            projection["info"]       = self.info
            projection["dimension"]  = 'None'
            projection["units"]      = ''
            projection["map"]        = import_int_map( jr )

            projections.append(projection)


        self.add_projections_to_table(projections, V, E)



class UpdateStackLink(MastodonFeature):
    name = 'Update stack Link'
    add_to = "Link"
    info = "Do not import"

class UpdateStackSpot(MastodonFeature):
    name = 'Update stack Spot'
    add_to = "Spot"
    info = "Do not import"


def import_double_map(jr):
    n_entries   = jr.read_int()
    map_ = numpy.zeros((n_entries, 2) )
    for proj in  range(n_entries):
        map_[proj , 0] = jr.read_int()
        map_[proj , 1] = jr.read_double_rev()

    return map_

def import_int_map(jr):
    n_entries   = jr.read_int()
    map_ = numpy.zeros((n_entries, 2), dtype="int32")
    for proj in  range(n_entries):
        map_[proj , 0] = jr.read_int()
        map_[proj , 1] = jr.read_int()

    return map_








if __name__ == "__main__":
    pass
