import io
import numpy
import struct
import zipfile


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
        pass

    def read_tags(self):
        pass

    def read_features(self):
        pass

    def read_graph(self):
        with zipfile.ZipFile(self.source_file) as masto_zip:
            fh = masto_zip.open("model.raw", "r")
            jr = JavaRawReader(fh)

            n_verticices = jr.read(4, ">i")[0]
            V = numpy.zeros(shape=(n_verticices, 12), dtype=numpy.float32)

            for i in range(n_verticices):
                V[i, 0:3] = jr.read(8 * 3, "<ddd")
                V[i, 3] = jr.read(4, "<i")[0]
                V[i, 4:11] = jr.read(8 * 7, "<ddddddd")
                V[i, 11] = i

            n_edges = jr.read(4, ">i")[0]
            E = numpy.zeros((n_edges, 3), dtype=numpy.int32)

            for i in range(n_edges):
                E[i, :2] = jr.read(4 * 4, ">iiii")[:2]
                E[i, 2] = i - 1

            E = E[numpy.argsort(E[:, 0]), :]

            jr.close()
            return V, E


if __name__ == "__main__":

    mr = MastodonReader(
        "H:/projects/066_nikhil_lineage/mastodon_playground/ilastik_bdv_hdf5_smooth.mastodon"
    )
    print(mr.read_graph())
