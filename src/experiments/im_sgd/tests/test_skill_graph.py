import unittest
from typing import override

from experiments.im_sgd.components import Node, SkillGraph


class TestEngine(unittest.TestCase):
    @override
    def setUp(self) -> None:
        self.skill_graph = SkillGraph()
        Node._next_id = 1 # reset node id
        self.nodes = [Node() for _ in range(5)]
        [node.terminal_states.append(1) for node in self.nodes]
        [self.skill_graph.add_node(node) for node in self.nodes]
        self.skill_graph.add_edge(1,2)
        self.skill_graph.add_edge(2,3)
        self.skill_graph.add_edge(2,5)
        self.skill_graph.add_edge(1,3)
        self.skill_graph.add_edge(5,4)

    def test_nodes_created(self) -> None:
        self.assertEqual(len(self.skill_graph.nodes), 5)
    
    def test_paths(self) -> None:
        self.assertTrue(self.skill_graph.has_path(1,2))
        self.assertTrue(self.skill_graph.has_path(1,3))
        self.assertTrue(self.skill_graph.has_path(1,5))
        self.assertFalse(self.skill_graph.has_path(3,4))
        self.assertTrue(self.skill_graph.has_path(5,4))
        self.assertFalse(self.skill_graph.has_path(4,5))

    def test_node_removal(self) -> None:
        self.skill_graph.remove_node(self.nodes[0])
        self.assertFalse(self.skill_graph.has_path(1,2))
        self.assertFalse(self.skill_graph.has_path(1,3))
        self.assertFalse(self.skill_graph.has_path(1,5))

    def test_edge_removal(self) -> None:
        self.skill_graph.remove_edge(1,2)
        self.assertFalse(self.skill_graph.has_path(1,2))
        self.assertTrue(self.skill_graph.has_path(1,3))
        self.assertFalse(self.skill_graph.has_path(1,5))
    
    def test_state_mapping(self) -> None:
        self.assertListEqual(self.skill_graph.map_state_to_nodes(1), self.skill_graph.nodes)

if __name__ == "__main__":
    unittest.main(verbosity=2)
