# Python implementations of the stream module's native functions.


def fold_source(step, init, produce):
    acc = [init]

    def sink(xs):
        acc[0] = step(acc[0], xs)

    produce(sink)
    return acc[0]


def collect_source(produce):
    out = []
    produce(out.extend)
    return out


class _Merges:
    """Summaries merged in stream order as a binary counter (see
    stream-rust): adjacent results of equal rank merge, so each summary
    takes part in O(log n) merges."""

    def __init__(self, merge):
        self.merge = merge
        self.stack = []

    def push(self, b):
        r = 0
        while self.stack and self.stack[-1][0] == r:
            _, top = self.stack.pop()
            b = self.merge(top, b)
            r += 1
        self.stack.append((r, b))

    def finish(self, empty):
        if not self.stack:
            return empty
        acc = self.stack[0][1]
        for _, b in self.stack[1:]:
            acc = self.merge(acc, b)
        return acc


def reduce_native(summarize, merge, pull):
    acc = _Merges(merge)
    while True:
        ok, msg, xs = pull()
        if not ok:
            return (False, msg, acc.finish(summarize([])))
        if len(xs) == 0:
            return (True, "", acc.finish(summarize([])))
        acc.push(summarize(xs))


def take_pull_native(n, pull):
    out = []
    while len(out) < n:
        ok, msg, xs = pull()
        if not ok:
            return (False, msg, out)
        if len(xs) == 0:
            break
        out.extend(xs[: n - len(out)])
    return (True, "", out)


_SORTERS = {}
_NEXT_SORTER = __import__("itertools").count(1)
_SORTED_BATCH = 1 << 16
_MAX_RUNS = 64


class _Sorter:
    """Records of `width` u64 words, sorted in runs of at most `budget`
    words written to unlinked temporary files and merged when read."""

    def __init__(self, d, width, budget):
        self.dir, self.width, self.budget = d, width, budget
        self.buf, self.runs, self.reader = [], [], None

    def _records(self, xs):
        w = self.width
        return [tuple(xs[i : i + w]) for i in range(0, len(xs), w)]

    def _write(self, recs):
        import array
        import tempfile
        f = tempfile.TemporaryFile(dir=self.dir)
        for i in range(0, len(recs), _SORTED_BATCH):
            a = array.array("Q", [x for r in recs[i : i + _SORTED_BATCH] for x in r])
            f.write(a.tobytes())
        f.flush()
        return f

    def _read(self, f):
        import array
        f.seek(0)
        size = 8 * self.width * _SORTED_BATCH
        while True:
            b = f.read(size)
            if not b:
                return
            if len(b) % (8 * self.width):
                raise IOError("a sort run ends inside a record")
            a = array.array("Q")
            a.frombytes(b)
            for i in range(0, len(a), self.width):
                yield tuple(a[i : i + self.width])

    def push(self, xs):
        if self.reader is not None:
            raise ValueError("pushSorter: the sorter is already being read")
        if len(xs) % self.width:
            raise ValueError("pushSorter: %d words are not whole records of %d" % (len(xs), self.width))
        if any(x < 0 or x >= 1 << 64 for x in xs):
            raise ValueError("pushSorter: a word is not an unsigned 64-bit integer")
        self.buf.extend(xs)
        if len(self.buf) >= self.budget:
            self._spill()

    def _spill(self):
        if self.buf:
            self.runs.append(self._write(sorted(self._records(self.buf))))
            self.buf = []
        if len(self.runs) >= _MAX_RUNS:
            import heapq
            merged = list(heapq.merge(*[self._read(f) for f in self.runs]))
            for f in self.runs:
                f.close()
            self.runs = [self._write(merged)]

    def next(self):
        import heapq
        import itertools
        if self.reader is None:
            if not self.runs:
                self.reader = iter(sorted(self._records(self.buf)))
            else:
                self._spill()
                self.reader = heapq.merge(*[self._read(f) for f in self.runs])
            self.buf = []
        out = []
        for r in itertools.islice(self.reader, _SORTED_BATCH):
            out.extend(r)
        return out

    def close(self):
        for f in self.runs:
            f.close()


def _sorter(h):
    if h not in _SORTERS:
        raise ValueError("no open sorter %d" % h)
    return _SORTERS[h]


def open_sorter(d, width, kib):
    import os
    if not 1 <= width <= 4:
        raise ValueError("openSorter: record width %d is not 1 to 4 words" % width)
    if kib < 1:
        raise ValueError("openSorter: a memory budget of %d KiB is less than 1" % kib)
    if not os.path.isdir(d):
        raise ValueError("openSorter: %s is not a directory" % d)
    words = kib * 1024 // 8
    h = next(_NEXT_SORTER)
    _SORTERS[h] = _Sorter(d, width, max(width, words - words % width))
    return h


def push_sorter(h, xs):
    _sorter(h).push(xs)


def next_sorted(h):
    try:
        return (True, "", _sorter(h).next())
    except (IOError, OSError, ValueError) as e:
        return (False, str(e), [])


def close_sorter(h):
    s = _SORTERS.pop(h, None)
    if s is not None:
        s.close()
