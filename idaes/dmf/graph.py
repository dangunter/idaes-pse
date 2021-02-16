"""
Graph related functions
"""
from idaes.dmf.errors import DMFError


class Graphviz:
    """Create output in Graphviz DOT language.

    To get the output DOT, simply convert the instance to a string.

    Sample usage::

        # assume 'dmf' represents a DMF instance, and 'rsrc' represents
        # a root Resource instance
        gv = Graphviz.from_dmf_related(dmf, rsrc)
        print(str(gv))

        # a more complicated example, where the first alias(name) is
        # pulled out as the node label

        def node_label(meta):
            result = {}
            if meta["aliases"]:
                result ["label"] = meta["aliases"][0]
            return result

        gv = Graphviz.from_dmf_related(dmf, rsrc, node_attr_fn=node_label)
        print(str(gv))
    """

    def __init__(self, node_attr_fn=None, edge_attr_fn=None):
        """Create Graphviz DOT exporter.

        The signature of the `*_fn` arguments should be:

            def func(meta: Dict) -> Dict:
                # logic goes here
                return result

        You can achieve the same result by using a subclass and overriding
        the public methods `get_node_attr` and `get_edge_attr`. But this may
        be too cumbersome in simple cases.

        Args:
            node_attr_fn: Return a dict of attributes for a node,
                          given an input dict of metadata from the DMF
            edge_attr_fn: Return a dict of attributes for an edge,
                          given an input dict of metadata from the DMF
        """
        # internal representation of the DOT graph
        self._nodes, self._edges = {}, []
        # user-provided functions to add attributes
        self._get_node_attr, self._get_edge_attr = node_attr_fn, edge_attr_fn

    def __str__(self):
        lines = []
        lines.append("digraph {")

        # nodes
        for node_id, node_attr in self._nodes.items():
            if node_attr:
                node_attr_str = ", ".join([f"{k} = \"{v}\"" for k, v in node_attr.items()])
                lines.append(f"{node_id} [ {node_attr_str} ];")
            else:
                id_pfx = node_id[:8]
                lines.append(f"{node_id} [ label = {id_pfx} ];")

        # edges
        for src, dst, edge_attr in self._edges:
            edge_str = f"{src} -> {dst}"
            if edge_attr:
                edge_attr_str = ", ".join([f"{k} = \"{v}\"" for k, v in edge_attr.items()])
                lines.append(f"{edge_str} [ {edge_attr_str} ];")
            else:
                lines.append(f"{edge_str};")

        lines.append("}")
        return "\n".join(lines)

    @staticmethod
    def from_related(triple_data, **gv_kwargs) -> "Graphviz":
        """Primary way to construct this class.

        Args:
            triple_data: List of (triple, metadata)
            gv_kwargs: Keyword arguments for Graphviz constructor

        Returns:
            Graphviz instance

        Raises:
            DMFError: error fetching the data from DMF
        """
        # Create Graphviz instance
        gv = Graphviz(**gv_kwargs)
        # Run query and feed results to Graphviz instance
        for triple, meta in triple_data:
            src, dst = triple.subject, triple.object
            gv.add_node(src, meta)
            gv.add_node(dst, meta)
            gv.add_edge(src, dst, meta)

        return gv

    def add_node(self, node_id, meta=None):
        node_id = self._gv_id(node_id)
        if node_id in self._nodes and len(self._nodes[node_id]) > 0:
            return  # nothing to do, already in dict with attributes
        node_attr = {} if meta is None else self.get_node_attr(meta)
        self._nodes[node_id] = node_attr

    def add_edge(self, src, dst, meta=None):
        edge_attr = {} if meta is None else self.get_edge_attr(meta)
        self._edges.append((self._gv_id(src), self._gv_id(dst), edge_attr))

    _DIGITS = {str(i) for i in range(10)}

    def _gv_id(self, s):
        if s[0] in self._DIGITS:
            s = "_" + s
        return s

    def get_node_attr(self, meta):
        return self._get_node_attr(meta) if self._get_node_attr else {}

    def get_edge_attr(self, meta):
        return self._get_edge_attr(meta) if self._get_edge_attr else {}
