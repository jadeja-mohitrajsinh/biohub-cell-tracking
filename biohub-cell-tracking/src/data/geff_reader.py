"""GEFF graph reader and writer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import zarr

from data.models import AxisMetadata, CellNode, GeffMetadata, TrackingEdge, TrackingGraph, VoxelScale


class GeffReader:
    """Load tracking graphs stored in GEFF (Zarr v3)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"GEFF store not found: {self.path}")
        self._root = zarr.open_group(str(self.path), mode="r")
        self.metadata = self._parse_metadata()

    def _parse_metadata(self) -> GeffMetadata:
        """Parse GEFF metadata from store attributes."""
        attrs = dict(self._root.attrs)
        geff = attrs.get("geff", {})
        axes = tuple(
            AxisMetadata(
                name=axis["name"],
                axis_type=axis.get("type", "space"),
                unit=axis.get("unit"),
                min_value=axis.get("min"),
                max_value=axis.get("max"),
                scale=float(axis.get("scale", 1.0)),
            )
            for axis in geff.get("axes", [])
        )
        estimated = geff.get("extra", {}).get("estimated_number_of_nodes")
        return GeffMetadata(
            geff_version=str(geff.get("geff_version", "1.1")),
            directed=bool(geff.get("directed", True)),
            axes=axes,
            estimated_number_of_nodes=int(estimated) if estimated is not None else None,
        )

    def read_graph(self) -> TrackingGraph:
        """Load the full tracking graph into memory."""
        node_ids = np.asarray(self._root["nodes"]["ids"])
        props_group = self._root["nodes"]["props"]
        t_values = np.asarray(props_group["t"]["values"])
        z_values = np.asarray(props_group["z"]["values"])
        y_values = np.asarray(props_group["y"]["values"])
        x_values = np.asarray(props_group["x"]["values"])

        nodes = {
            int(node_id): CellNode(
                node_id=int(node_id),
                t=int(t_values[index]),
                z=int(z_values[index]),
                y=int(y_values[index]),
                x=int(x_values[index]),
            )
            for index, node_id in enumerate(node_ids)
        }

        edge_ids = np.asarray(self._root["edges"]["ids"])
        edges = [
            TrackingEdge(source_id=int(source), target_id=int(target))
            for source, target in edge_ids
        ]
        return TrackingGraph(nodes=nodes, edges=edges, metadata=self.metadata)


class GeffWriter:
    """Write tracking predictions as GEFF Zarr stores."""

    def __init__(
        self,
        path: str | Path,
        scale: VoxelScale | None = None,
        estimated_number_of_nodes: int | None = None,
        overwrite: bool = True,
        zarr_format: int = 3,
    ) -> None:
        self.path = Path(path)
        self.scale = scale or VoxelScale()
        self.estimated_number_of_nodes = estimated_number_of_nodes
        self.overwrite = overwrite
        self.zarr_format = zarr_format

    def write(self, graph: TrackingGraph) -> Path:
        """Serialize a tracking graph to GEFF."""
        if self.path.exists():
            if not self.overwrite:
                raise FileExistsError(f"Refusing to overwrite existing GEFF: {self.path}")
            import shutil

            shutil.rmtree(self.path)

        self.path.mkdir(parents=True, exist_ok=True)
        root = zarr.open_group(str(self.path), mode="w")

        node_ids = np.array(sorted(graph.nodes), dtype=np.uint64)
        t_values = np.array([graph.nodes[int(node_id)].t for node_id in node_ids], dtype=np.int64)
        z_values = np.array([graph.nodes[int(node_id)].z for node_id in node_ids], dtype=np.int64)
        y_values = np.array([graph.nodes[int(node_id)].y for node_id in node_ids], dtype=np.int64)
        x_values = np.array([graph.nodes[int(node_id)].x for node_id in node_ids], dtype=np.int64)

        nodes_group = root.create_group("nodes")
        nodes_group.create_array("ids", data=node_ids)
        props_group = nodes_group.create_group("props")
        for prop_name, values in ("t", t_values), ("z", z_values), ("y", y_values), ("x", x_values):
            prop_group = props_group.create_group(prop_name)
            prop_group.create_array("values", data=values)

        edge_array = np.array(
            [[edge.source_id, edge.target_id] for edge in graph.edges],
            dtype=np.uint64,
        )
        edges_group = root.create_group("edges")
        edges_group.create_array("ids", data=edge_array)
        edges_group.create_group("props")

        t_max = int(t_values.max()) if len(t_values) else 0
        z_min = int(z_values.min()) if len(z_values) else 0
        z_max = int(z_values.max()) if len(z_values) else 0
        y_min = int(y_values.min()) if len(y_values) else 0
        y_max = int(y_values.max()) if len(y_values) else 0
        x_min = int(x_values.min()) if len(x_values) else 0
        x_max = int(x_values.max()) if len(x_values) else 0

        estimated_nodes = self.estimated_number_of_nodes
        if estimated_nodes is None and graph.metadata is not None:
            estimated_nodes = graph.metadata.estimated_number_of_nodes

        root.attrs.update(
            {
                "geff": {
                    "geff_version": "1.1",
                    "directed": True,
                    "axes": [
                        {
                            "name": "t",
                            "type": "time",
                            "unit": None,
                            "min": 0.0,
                            "max": float(t_max),
                            "scale": self.scale.t,
                        },
                        {
                            "name": "z",
                            "type": "space",
                            "unit": None,
                            "min": float(z_min),
                            "max": float(z_max),
                            "scale": self.scale.z,
                        },
                        {
                            "name": "y",
                            "type": "space",
                            "unit": None,
                            "min": float(y_min),
                            "max": float(y_max),
                            "scale": self.scale.y,
                        },
                        {
                            "name": "x",
                            "type": "space",
                            "unit": None,
                            "min": float(x_min),
                            "max": float(x_max),
                            "scale": self.scale.x,
                        },
                    ],
                    "node_props_metadata": {
                        key: {
                            "identifier": key,
                            "dtype": "int64",
                            "varlength": False,
                            "unit": None,
                        }
                        for key in ("t", "z", "y", "x")
                    },
                    "edge_props_metadata": {},
                    "extra": {
                        "estimated_number_of_nodes": int(
                            estimated_nodes if estimated_nodes is not None else len(node_ids)
                        )
                    },
                }
            }
        )

        self._write_zarr_json_sidecars()
        return self.path

    def _write_zarr_json_sidecars(self) -> None:
        """Write minimal Zarr v3 sidecar metadata for external tools."""
        root_meta = {
            "attributes": dict(zarr.open_group(str(self.path), mode="r").attrs),
            "zarr_format": self.zarr_format,
            "node_type": "group",
        }
        with (self.path / "zarr.json").open("w", encoding="utf-8") as handle:
            json.dump(root_meta, handle, indent=2)
