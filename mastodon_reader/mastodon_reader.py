import io
import numpy
import pandas
import struct
import zipfile
import xml.etree.ElementTree as ET

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

    def read_tags(self):
        pass

    def read_features(self):
        pass

    def read_graph_networkx(self):
        import networkx as nx

        V, E = self.read_graph()

        G = nx.from_pandas_edgelist(E, source="source_idx", target="target_idx")
        nx.set_node_attributes(G, V.to_dict("index"))

        return G

    def read_graph(self):
        with zipfile.ZipFile(self.source_file) as masto_zip:
            fh = masto_zip.open("model.raw", "r")

            jr = JavaRawReader(fh)

            try:
                n_verticices = jr.read(4, ">i")[0]
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

                n_edges = jr.read(4, ">i")[0]
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
